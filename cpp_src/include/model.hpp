#pragma once

#include <gurobi_c++.h>

#include <boost/unordered_map.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct SolverParameters {
    int timeLimitSec = 60;
    double heuristicEffort = 0.05;
    int presolve = -1;
    bool mipStart = false;
    bool logs = false;
};

class Instance;  // Forward declaration

class GurobiModel {
public:
    explicit GurobiModel(const Instance& instance);

    // Model formulation
    void formulate();

    // Variable management
    void addVariables();

    // Constraint management
    void addConstraints();

    // Objective management
    void addObjective();

    // Model solving
    void solveModelWithParameters();  // Default parameters
    void solveModelWithParameters(const SolverParameters& params);

    // Solution retrieval
    std::pair<std::vector<int>, double> getSolutionAndCost() const;

    // Model status checks
    void checkModelIsFormulated() const;
    void checkModelStatus() const;

    // Getters for model and variables
    const GRBModel& getModel() const {
        return model_;
    }
    const boost::unordered_map<std::pair<int, int>, GRBVar>& getX() const {
        return x_;
    }
    const boost::unordered_map<std::tuple<int, int, int, int>, GRBVar>& getY() const {
        return y_;
    }

    double getMIPGap() const {
        return model_.get(GRB_DoubleAttr_MIPGap);
    }

    double getWallTime() const {
        return model_.get(GRB_DoubleAttr_Runtime);
    }

    double getObjBound() const {
        return model_.get(GRB_DoubleAttr_ObjBound);
    }

private:
    const Instance& instance_;
    GRBEnv env_;
    GRBModel model_;
    bool formulated_ = false;

    // Variable maps
    boost::unordered_map<std::pair<int, int>, GRBVar> x_;             // Edge variables
    boost::unordered_map<int, GRBVar> u_;                             // Node position variables
    boost::unordered_map<std::tuple<int, int, int, int>, GRBVar> y_;  // Relation variables
    boost::unordered_map<std::tuple<int, int, int, int>, GRBVar> z_;  // Precedence variables

    // Helper methods
    void provideMipStart(const std::vector<boost::unordered_map<std::string, double>>& vars) const;
};