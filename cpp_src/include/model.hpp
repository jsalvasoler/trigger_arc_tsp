#pragma once

#ifdef USE_GUROBI
#include <gurobi_c++.h>
#else
// Dummy placeholders to allow compilation without Gurobi
struct GRBEnv {};
struct GRBModel {
    void update() {}
    void optimize() {}
    double get(int) const { return 0.0; }
};
struct GRBVar {};
#endif

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
#ifdef USE_GUROBI
    const GRBModel& getModel() const {
        return model_;
    }
    const boost::unordered_map<std::pair<int, int>, GRBVar>& getX() const {
        return x_;
    }
    const boost::unordered_map<std::tuple<int, int, int, int>, GRBVar>& getY() const {
        return y_;
    }
#endif

    double getMIPGap() const;
    double getWallTime() const;
    double getObjBound() const;

private:
    const Instance& instance_;
#ifdef USE_GUROBI
    GRBEnv env_;
    GRBModel model_;
#endif
    bool formulated_ = false;

    // Variable maps
#ifdef USE_GUROBI
    boost::unordered_map<std::pair<int, int>, GRBVar> x_;             // Edge variables
    boost::unordered_map<int, GRBVar> u_;                             // Node position variables
    boost::unordered_map<std::tuple<int, int, int, int>, GRBVar> y_;  // Relation variables
    boost::unordered_map<std::tuple<int, int, int, int>, GRBVar> z_;  // Precedence variables
#endif

    // Helper methods
    void provideMipStart(const std::vector<boost::unordered_map<std::string, double>>& vars) const;
};
