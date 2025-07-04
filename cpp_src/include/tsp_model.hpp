#pragma once

#ifdef USE_GUROBI
#include <gurobi_c++.h>
using GurobiVarType = GRBVar;
#else
// Dummy type so code compiles without Gurobi
struct DummyGRBVar {};
using GurobiVarType = DummyGRBVar;
#endif

#include <boost/unordered_map.hpp>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "instance.hpp"

class GurobiTSPModel {
public:
    explicit GurobiTSPModel(const Instance& instance);

    void formulate();
    void solveToFeasibleSolution();
    void solveToOptimality(std::optional<int> timeLimitSec = std::nullopt, bool logs = false);

    std::vector<int> getBestTour() const;
    std::vector<std::vector<int>> getBestNTours(int n) const;

    const boost::unordered_map<std::pair<int, int>, GurobiVarType>& getX() const {
        return x_;
    }
    const boost::unordered_map<int, GurobiVarType>& getU() const {
        return u_;
    }

#ifdef USE_GUROBI
    const GRBModel& getModel() const {
        return model_;
    }
#endif

private:
    void checkModelIsFormulated() const;
    void checkModelStatus() const;
    int getEdgeIndex(const std::pair<int, int>& edge) const;
    int getNodeIndex(int node) const;

    const Instance& instance_;

#ifdef USE_GUROBI
    GRBEnv env_;
    GRBModel model_;
#endif

    bool formulated_ = false;

    boost::unordered_map<std::pair<int, int>, GurobiVarType> x_;  // Binary variables for edges
    boost::unordered_map<int, GurobiVarType> u_;  // Continuous variables for node positions
};
