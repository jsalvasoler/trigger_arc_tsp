#pragma once

#include <gurobi_c++.h>

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

    const boost::unordered_map<std::pair<int, int>, GRBVar>& getX() const {
        return x_;
    }
    const boost::unordered_map<int, GRBVar>& getU() const {
        return u_;
    }
    const GRBModel& getModel() const {
        return model_;
    }

private:
    void checkModelIsFormulated() const;
    void checkModelStatus() const;
    int getEdgeIndex(const std::pair<int, int>& edge) const;
    int getNodeIndex(int node) const;
    GRBEnv createSilentEnvironmentSilently();

    const Instance& instance_;
    GRBEnv env_;
    GRBModel model_;
    bool formulated_ = false;

    boost::unordered_map<std::pair<int, int>, GRBVar> x_;  // Binary variables for edges
    boost::unordered_map<int, GRBVar> u_;  // Continuous variables for node positions
};