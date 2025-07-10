#pragma once

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class NeighborhoodIterator {
public:
    NeighborhoodIterator(int n, bool include_zero, bool allow_all_pairs)
        : n_(n), include_zero_(include_zero), allow_all_pairs_(allow_all_pairs) {
        reset();
    }

    std::optional<std::pair<int, int>> next() {
        if (allow_all_pairs_) {
            while (i_ < n_) {
                if (j_ >= n_) {
                    i_++;
                    j_ = start_i_;
                    continue;
                }
                if (i_ == j_) {
                    j_++;
                    continue;
                }
                return std::make_pair(i_, j_++);
            }
        } else {  // i < j
            if (j_ >= n_) {
                i_++;
                j_ = i_ + 1;
            }

            if (i_ >= n_ - 1) {
                return std::nullopt;
            }

            return std::make_pair(i_, j_++);
        }
        return std::nullopt;
    }

    void reset() {
        start_i_ = include_zero_ ? 0 : 1;
        i_ = start_i_;
        j_ = allow_all_pairs_ ? start_i_ : i_ + 1;
    }

private:
    const int n_;
    const bool include_zero_;
    const bool allow_all_pairs_;
    int start_i_;
    int i_;
    int j_;
};

class Instance {
public:
    // Constructor
    Instance(int N,
             const boost::unordered_map<std::pair<int, int>, double>& edges,
             const boost::unordered_map<std::tuple<int, int, int, int>, double>& relations,
             const std::string& name);

    // Static factory method
    static std::unique_ptr<Instance> loadInstanceFromFile(const std::string& filePath);

    // Getters
    int getN() const {
        return N_;
    }
    int getA() const {
        return A_;
    }
    int getR() const {
        return R_;
    }
    const std::string& getName() const {
        return name_;
    }
    const std::string& getModelName() const {
        return modelName_;
    }
    const boost::unordered_map<std::pair<int, int>, double>& getEdges() const {
        return edges_;
    }
    const boost::unordered_map<std::tuple<int, int, int, int>, double>& getRelations() const {
        return relations_;
    }
    const boost::unordered_set<int>& getDeltaIn(int node) const {
        return deltaIn_.at(node);
    }
    const boost::unordered_set<int>& getDeltaOut(int node) const {
        return deltaOut_.at(node);
    }
    const std::vector<std::tuple<int, int, int, int>>& getZVarIndices() const {
        if (zVarIndices_.empty()) {
            generateZVarIndices();
        }
        return zVarIndices_;
    }
    const boost::unordered_map<std::pair<int, int>, std::vector<std::pair<int, int>>>& getRA()
        const {
        return R_a_;
    }

    void resetTrackers() const {
        twoOptIteratorTracker_.reset();
        swapTwoIteratorTracker_.reset();
        relocateIteratorTracker_.reset();
    }

    // Core methods
    double computeObjective(const std::vector<int>& tour) const;
    bool checkSolutionCorrectness(const std::vector<int>& tour) const;
    bool testSolution(const std::vector<int>& tour, double proposedObjective) const;
    void saveSolution(const std::vector<int>& tour,
                      std::optional<double> objective = std::nullopt) const;
    std::optional<std::vector<int>> getBestKnownSolution(int idx = 0) const;
    std::vector<boost::unordered_map<std::string, double>> getMipStart(
        bool useTspOnly = false) const;
    std::vector<boost::unordered_map<std::string, double>> getVariablesFromTour(
        const std::vector<int>& tour) const;
    std::vector<int> tspSolution() const;
    float computePartialTourCost(const std::vector<int>& partialTour, int startIdx = 0) const;
    void generateZVarIndices() const;
    void get_two_opt_neighbor(std::vector<int>& tour, int i, int j) const;
    void get_swap_two_neighbor(std::vector<int>& tour, int i, int j) const;
    void get_relocate_neighbor(std::vector<int>& tour, int i, int j) const;

    mutable NeighborhoodIterator twoOptIteratorTracker_;
    mutable NeighborhoodIterator swapTwoIteratorTracker_;
    mutable NeighborhoodIterator relocateIteratorTracker_;

private:
    // Member variables
    std::string name_;
    std::string modelName_;
    int N_;  // Number of nodes
    int A_;  // Number of edges
    int R_;  // Number of relations

    boost::unordered_map<std::pair<int, int>, double> edges_;
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations_;
    boost::unordered_map<std::pair<int, int>, std::vector<std::pair<int, int>>> R_a_;

    boost::unordered_map<int, boost::unordered_set<int>> deltaIn_;
    boost::unordered_map<int, boost::unordered_set<int>> deltaOut_;
    mutable std::vector<std::tuple<int, int, int, int>> zVarIndices_;
};