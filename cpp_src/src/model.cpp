#include "model.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include "instance.hpp"

#ifdef USE_GUROBI

GurobiModel::GurobiModel(const Instance& instance) : instance_(instance), env_(), model_(env_) {}

void GurobiModel::formulate() {
    addVariables();
    addConstraints();
    addObjective();
    formulated_ = true;
}

void GurobiModel::addVariables() {
    for (const auto& [edge, cost] : instance_.getEdges()) {
        auto [i, j] = edge;
        std::string name = "x_" + std::to_string(i) + "_" + std::to_string(j);
        x_[edge] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    for (int i = 0; i < instance_.getN(); ++i) {
        std::string name = "u_" + std::to_string(i);
        double ub = (i == 0) ? 0.0 : instance_.getN() - 1;
        u_[i] = model_.addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
    }

    for (const auto& [relation, cost] : instance_.getRelations()) {
        auto [b0, b1, a0, a1] = relation;
        std::string name = "y_" + std::to_string(b0) + "_" + std::to_string(b1) + "_" +
                           std::to_string(a0) + "_" + std::to_string(a1);
        y_[relation] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    for (const auto& [a1, a2, b1, b2] : instance_.getZVarIndices()) {
        std::string name = "z_" + std::to_string(a1) + "_" + std::to_string(a2) + "_" +
                           std::to_string(b1) + "_" + std::to_string(b2);
        z_[{a1, a2, b1, b2}] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    model_.update();
}

void GurobiModel::addConstraints() {
    // (Your constraint code here; no change except wrap inside #ifdef)
    // ... all constraints ...
}

void GurobiModel::addObjective() {
    GRBLinExpr obj = 0;

    for (const auto& [edge, cost] : instance_.getEdges()) {
        obj += x_[edge] * cost;
    }

    for (const auto& [relation, cost] : instance_.getRelations()) {
        obj += y_[relation] * cost;
    }

    model_.setObjective(obj, GRB_MINIMIZE);
}

void GurobiModel::solveModelWithParameters() {
    solveModelWithParameters(SolverParameters());
}

void GurobiModel::solveModelWithParameters(const SolverParameters& params) {
    checkModelIsFormulated();

    if (params.mipStart) {
        auto vars = instance_.getMipStart(true);
        provideMipStart(vars);
    }

    if (params.timeLimitSec > 0) {
        model_.set(GRB_DoubleParam_TimeLimit, params.timeLimitSec);
    }
    model_.set(GRB_DoubleParam_Heuristics, params.heuristicEffort);
    model_.set(GRB_IntParam_Presolve, params.presolve);

    if (!params.logs) {
        model_.set(GRB_IntParam_OutputFlag, 0);
    }

    model_.optimize();
    checkModelStatus();
}

std::pair<std::vector<int>, double> GurobiModel::getSolutionAndCost() const {
    const_cast<GRBModel&>(model_).update();

    std::vector<std::pair<int, double>> nodePositions;
    for (int i = 1; i < instance_.getN(); ++i) {
        nodePositions.emplace_back(i, u_.at(i).get(GRB_DoubleAttr_Xn));
    }

    std::sort(nodePositions.begin(), nodePositions.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> tour = {0};
    for (const auto& [node, _] : nodePositions) {
        tour.push_back(node);
    }

    double cost = 0;
    for (const auto& [edge, edgeCost] : instance_.getEdges()) {
        if (x_.at(edge).get(GRB_DoubleAttr_Xn) > 0.5) {
            cost += edgeCost;
        }
    }
    for (const auto& [relation, relationCost] : instance_.getRelations()) {
        if (y_.at(relation).get(GRB_DoubleAttr_Xn) > 0.5) {
            cost += relationCost;
        }
    }

    return {tour, cost};
}

void GurobiModel::checkModelIsFormulated() const {
    if (!formulated_) {
        throw std::runtime_error("Model is not formulated");
    }
}

void GurobiModel::checkModelStatus() const {
    int status = model_.get(GRB_IntAttr_Status);
    if (status == GRB_INFEASIBLE) {
        const_cast<GRBModel&>(model_).computeIIS();
        const_cast<GRBModel&>(model_).write("model.ilp");
        throw std::runtime_error("Model is infeasible");
    }
}

void GurobiModel::provideMipStart(
    const std::vector<boost::unordered_map<std::string, double>>& vars) const {
    std::cout << "Providing MIP start" << std::endl;
    assert(vars.size() == 4);

    for (const auto& [key, val] : vars[0]) {
        size_t pos = key.find(',');
        int i = std::stoi(key.substr(0, pos));
        int j = std::stoi(key.substr(pos + 1));
        const_cast<GRBVar&>(x_.at({i, j})).set(GRB_DoubleAttr_Start, val);
    }

    for (const auto& [key, val] : vars[1]) {
        size_t pos1 = key.find(',');
        size_t pos2 = key.find(',', pos1 + 1);
        size_t pos3 = key.find(',', pos2 + 1);
        int b0 = std::stoi(key.substr(0, pos1));
        int b1 = std::stoi(key.substr(pos1 + 1, pos2 - pos1 - 1));
        int a0 = std::stoi(key.substr(pos2 + 1, pos3 - pos2 - 1));
        int a1 = std::stoi(key.substr(pos3 + 1));
        const_cast<GRBVar&>(y_.at({b0, b1, a0, a1})).set(GRB_DoubleAttr_Start, val);
    }

    for (const auto& [key, val] : vars[2]) {
        int i = std::stoi(key);
        const_cast<GRBVar&>(u_.at(i)).set(GRB_DoubleAttr_Start, val);
    }

    for (const auto& [key, val] : vars[3]) {
        size_t pos1 = key.find(',');
        size_t pos2 = key.find(',', pos1 + 1);
        size_t pos3 = key.find(',', pos2 + 1);
        int a1 = std::stoi(key.substr(0, pos1));
        int a2 = std::stoi(key.substr(pos1 + 1, pos2 - pos1 - 1));
        int b1 = std::stoi(key.substr(pos2 + 1, pos3 - pos2 - 1));
        int b2 = std::stoi(key.substr(pos3 + 1));
        const_cast<GRBVar&>(z_.at({a1, a2, b1, b2})).set(GRB_DoubleAttr_Start, val);
    }
}

#else  // No Gurobi, stub implementations

GurobiModel::GurobiModel(const Instance& instance) : instance_(instance) {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::formulate() {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::addVariables() {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::addConstraints() {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::addObjective() {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::solveModelWithParameters() {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::solveModelWithParameters(const SolverParameters&) {
    throw std::runtime_error("Gurobi not enabled");
}

std::pair<std::vector<int>, double> GurobiModel::getSolutionAndCost() const {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::checkModelIsFormulated() const {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::checkModelStatus() const {
    throw std::runtime_error("Gurobi not enabled");
}

void GurobiModel::provideMipStart(const std::vector<boost::unordered_map<std::string, double>>&) const {
    throw std::runtime_error("Gurobi not enabled");
}

#endif
