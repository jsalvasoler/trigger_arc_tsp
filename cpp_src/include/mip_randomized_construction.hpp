#ifndef MIP_RANDOMIZED_CONSTRUCTION_HPP
#define MIP_RANDOMIZED_CONSTRUCTION_HPP

#include <boost/unordered_map.hpp>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "method.hpp"
#include "tsp_model.hpp"

struct TSPPrior {
    std::vector<int> priorities;
    double alpha;
    double beta;
    double cost;
    double relGap;
    std::vector<int> bestTour;

    TSPPrior(const std::vector<int>& prio, double a, double b)
        : priorities(prio),
          alpha(a),
          beta(b),
          cost(std::numeric_limits<double>::infinity()),
          relGap(0.0),
          bestTour() {}
};

class MIPRandomizedConstruction : public Method {
public:
    explicit MIPRandomizedConstruction(const Instance& instance,
                                       int nTrials = 100,
                                       int timeLimitSec = 10);
    ~MIPRandomizedConstruction() override = default;

    void run() override;
    std::vector<int> getSolution() const;
    double getBestCost() const;

    // Core evaluation methods (public for testing)
    std::pair<std::vector<int>, double> evaluateIndividual(TSPPrior& tspPrior, int timeLimitSec);
    boost::unordered_map<std::pair<int, int>, double> getEdgesForTSPSearch(
        const TSPPrior& tspPrior);

    // Node distance computation (public for testing)
    boost::unordered_map<std::pair<int, int>, double> computeNodeDist(
        const std::vector<int>& nodePriorities);

    // Solution generation methods (public for testing)
    std::vector<int> generateRandomPermutation();
    std::vector<std::vector<int>> generateNRandomPermutations(int n);

private:
    // Utility methods
    std::unique_ptr<Instance> createTSPInstance(
        const boost::unordered_map<std::pair<int, int>, double>& tspEdges);

    // Member variables
    int nTrials_;
    int timeLimitSec_;
    std::vector<int> bestSolution_;
    double bestCost_;
    mutable std::mt19937 rng_;
};

#endif  // MIP_RANDOMIZED_CONSTRUCTION_HPP