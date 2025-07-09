#ifndef RANDOMIZED_GREEDY_HPP
#define RANDOMIZED_GREEDY_HPP

#include <boost/unordered_set.hpp>
#include <random>
#include <vector>

#include "method.hpp"

class RandomizedGreedyConstruction : public Method {
public:
    explicit RandomizedGreedyConstruction(const Instance& instance, double alpha = 0.2);
    ~RandomizedGreedyConstruction() override = default;

    void run() override;
    std::vector<int> getSolution() const override;

private:
    std::vector<int> constructSolution();
    std::vector<std::pair<int, int>> getFeasibleEdges(
        const std::vector<int>& partialTour, const boost::unordered_set<int>& visitedNodes) const;
    std::pair<int, int> selectRandomizedGreedyEdge(
        const std::vector<std::pair<int, int>>& feasibleEdges,
        const std::vector<int>& partialTour) const;

    double alpha_;  // Randomization parameter (0 = pure greedy, 1 = pure random)
    std::vector<int> bestSolution_;
    double bestCost_;
    mutable std::mt19937 rng_;
};

#endif  // RANDOMIZED_GREEDY_HPP