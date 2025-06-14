#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <optional>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <tuple>

class Instance {
public:
    // Constructor
    Instance(int N, 
            const boost::unordered_map<std::pair<int, int>, double>& edges,
            const boost::unordered_map<std::tuple<int, int, int, int>, double>& relations,
            const std::string& name);

    // Static factory method
    static std::unique_ptr<Instance> loadInstanceFromFile(const std::string& filePath);

    // Getters for TSP model
    int getN() const { return N_; }
    const boost::unordered_map<std::pair<int, int>, double>& getEdges() const { return edges_; }
    const boost::unordered_map<std::tuple<int, int, int, int>, double>& getRelations() const { return relations_; }
    const boost::unordered_set<int>& getDeltaIn(int node) const { return deltaIn_.at(node); }
    const boost::unordered_set<int>& getDeltaOut(int node) const { return deltaOut_.at(node); }
    const std::vector<std::tuple<int, int, int, int>>& getZVarIndices() const { return zVarIndices_; }
    const boost::unordered_map<std::pair<int, int>, std::vector<std::pair<int, int>>>& getRA() const { return R_a_; }

    // Core methods
    double computeObjective(const std::vector<int>& tour) const;
    bool checkSolutionCorrectness(const std::vector<int>& tour) const;
    bool testSolution(const std::vector<int>& tour, double proposedObjective) const;
    void saveSolution(const std::vector<int>& tour, std::optional<double> objective = std::nullopt);
    std::optional<std::vector<int>> getBestKnownSolution(int idx = 0) const;
    std::vector<boost::unordered_map<std::string, double>> getMipStart(bool useTspOnly = false) const;
    std::vector<boost::unordered_map<std::string, double>> getVariablesFromTour(const std::vector<int>& tour) const;
    std::vector<int> tspSolution() const;

    // Getters
    int getA() const { return A_; }
    int getR() const { return R_; }
    const std::string& getName() const { return name_; }
    const std::string& getModelName() const { return modelName_; }
    

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
    std::vector<std::tuple<int, int, int, int>> zVarIndices_;
}; 