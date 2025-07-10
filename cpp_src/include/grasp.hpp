#pragma once
#include <memory>

#include "instance.hpp"
#include "method.hpp"

enum class ConstructiveHeuristicType {
    RandomizedGreedy,
    MIPRandomizedGreedyBias,
    MIPRandomizedGreedyRandom,
    SimpleRandomized
};

enum class LocalSearch {
    TwoOpt,    // 2-opt
    SwapTwo,   // swap two nodes
    Relocate,  // relocate a node to a new position
};

class GRASP : public Method {
public:
    GRASP(const Instance& instance,
          int n_trials,
          double alpha,
          double beta,
          ConstructiveHeuristicType constructive_heuristic_type,
          const std::vector<LocalSearch>& local_searches,
          bool log_to_stdout = false,
          bool start_with_best_saved_solution = false,
          int time_limit_sec = 3600);
    ~GRASP() override = default;

    void run() override;
    std::vector<int> getSolution() const override;

private:
    std::optional<std::vector<int>> localSearch(const std::vector<int>& tour);
    int n_trials_;
    double alpha_;
    double beta_;
    ConstructiveHeuristicType constructive_heuristic_type_;
    std::vector<LocalSearch> local_searches_;
    bool log_to_stdout_;
    std::vector<int> best_tour_;
    double best_cost_;
    bool start_with_best_saved_solution_;
    int time_limit_sec_;
    int iteration_ = 0;

    std::unique_ptr<Method> constructive_heuristic_;
    void log_iteration(double cost,
                       double best_cost,
                       double time_elapsed,
                       bool improvement = false);
};
