#include "instance.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

namespace fs = std::filesystem;

Instance::Instance(int N,
                 const boost::unordered_map<std::pair<int, int>, double>& edges,
                 const boost::unordered_map<std::tuple<int, int, int, int>, double>& relations,
                 const std::string& name)
    : name_(name)
    , modelName_(name.substr(0, name.find_last_of('.')) + ".mps")
    , N_(N)
    , A_(edges.size())
    , R_(relations.size())
    , edges_(edges)
    , relations_(relations) {
    
    // Initialize delta_in and delta_out
    for (int node = 0; node < N_; ++node) {
        deltaIn_[node] = boost::unordered_set<int>();
        deltaOut_[node] = boost::unordered_set<int>();
    }

    // Fill delta_in and delta_out
    for (const auto& [edge, _] : edges_) {
        deltaIn_[edge.second].insert(edge.first);
        deltaOut_[edge.first].insert(edge.second);
    }

    // Initialize R_a
    for (const auto& [rel, cost] : relations_) {
        auto [b0, b1, a0, a1] = rel;
        R_a_[{a0, a1}].push_back({b0, b1});
    }

    // Update relations by subtracting edge costs
    for (auto& [rel, cost] : relations_) {
        auto [b0, b1, a0, a1] = rel;
        cost -= edges_[{a0, a1}];
    }

    // Generate z variable indices
    boost::unordered_set<std::tuple<int, int, int, int>> zVarIndicesSet;
    for (const auto& [a, bList] : R_a_) {
        for (const auto& b : bList) {
            if (b != std::make_pair(0, 0)) {
                zVarIndicesSet.insert({a.first, a.second, b.first, b.second});
                zVarIndicesSet.insert({b.first, b.second, a.first, a.second});
            }
            for (const auto& c : bList) {
                if (b != c && b != std::make_pair(0, 0)) {
                    zVarIndicesSet.insert({b.first, b.second, c.first, c.second});
                }
            }
        }
    }
    zVarIndices_ = std::vector<std::tuple<int, int, int, int>>(zVarIndicesSet.begin(), zVarIndicesSet.end());
}

std::unique_ptr<Instance> Instance::loadInstanceFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    int N, A, R;
    iss >> N >> A >> R;

    boost::unordered_map<std::pair<int, int>, double> edges;
    for (int i = 0; i < A; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        int idx, from, to;
        double weight;
        iss >> idx >> from >> to >> weight;
        edges[{from, to}] = weight;
    }

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    for (int i = 0; i < R; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        int idx1, idx2, fromTrigger, toTrigger, idx3, fromArc, toArc;
        double cost;
        iss >> idx1 >> idx2 >> fromTrigger >> toTrigger >> idx3 >> fromArc >> toArc >> cost;
        relations[{fromTrigger, toTrigger, fromArc, toArc}] = cost;
    }

    // Get filename from path
    std::string name = filePath.substr(filePath.find_last_of("/\\") + 1);
    return std::make_unique<Instance>(N, edges, relations, name);
}

double Instance::computeObjective(const std::vector<int>& tour) const {
    std::vector<std::pair<int, int>> path;
    for (int i = 0; i < N_; ++i) {
        path.push_back({tour[i], (i < N_ - 1) ? tour[i + 1] : 0});
    }

    double cost = 0.0;
    for (const auto& edge : path) {
        cost += edges_.at(edge);
    }

    for (const auto& a : path) {
        auto it = R_a_.find(a);
        if (it == R_a_.end()) continue;

        std::vector<std::pair<int, int>> triggering;
        for (const auto& b : it->second) {
            if (std::find(path.begin(), path.end(), b) != path.end()) {
                triggering.push_back(b);
            }
        }

        if (triggering.empty()) continue;

        // Sort triggering by their index in path
        std::sort(triggering.begin(), triggering.end(),
                 [&path](const auto& b1, const auto& b2) {
                     return std::find(path.begin(), path.end(), b1) <
                            std::find(path.begin(), path.end(), b2);
                 });

        // Remove triggering arcs that happen after arc a
        auto aPos = std::find(path.begin(), path.end(), a);
        triggering.erase(
            std::remove_if(triggering.begin(), triggering.end(),
                          [&path, aPos](const auto& b) {
                              return std::find(path.begin(), path.end(), b) >= aPos;
                          }),
            triggering.end());

        if (!triggering.empty()) {
            auto lastTrigger = triggering.back();
            cost += relations_.at({lastTrigger.first, lastTrigger.second, a.first, a.second});
        }
    }

    return cost;
}

bool Instance::checkSolutionCorrectness(const std::vector<int>& tour) const {
    if (tour.back() == 0) {
        std::vector<int> tourCopy = tour;
        tourCopy.pop_back();
        return checkSolutionCorrectness(tourCopy);
    }

    if (tour.size() != static_cast<size_t>(N_) || tour[0] != 0) {
        return false;
    }

    boost::unordered_set<int> uniqueNodes(tour.begin(), tour.end());
    return uniqueNodes.size() == static_cast<size_t>(N_) && 
           std::all_of(uniqueNodes.begin(), uniqueNodes.end(),
                      [this](int node) { return node >= 0 && node < N_; });
}

bool Instance::testSolution(const std::vector<int>& tour, double proposedObjective) const {
    if (!checkSolutionCorrectness(tour)) {
        return false;
    }

    double cost = computeObjective(tour);
    return std::abs(cost - proposedObjective) < 1e-6;
}

void Instance::saveSolution(const std::vector<int>& tour, std::optional<double> objective) {
    if (!checkSolutionCorrectness(tour)) {
        throw std::runtime_error("Invalid solution for instance " + name_);
    }

    double obj = objective.value_or(computeObjective(tour));
    if (!testSolution(tour, obj)) {
        std::cerr << "Warning: Solution does not have correct objective value for instance " 
                  << name_ << " (computed: " << computeObjective(tour) 
                  << ", proposed: " << obj << ")" << std::endl;
        obj = computeObjective(tour);
    }

    // Create solutions directory if it doesn't exist
    fs::path solutionsDir = fs::current_path().parent_path().parent_path() / "solutions";
    fs::create_directories(solutionsDir);

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    std::string timestamp = ss.str();

    // Write solution to file
    std::ofstream file(solutionsDir / name_, std::ios::app);
    if (file.tellp() != 0) {
        file << "\n";
    }

    for (size_t i = 0; i < tour.size(); ++i) {
        file << tour[i];
        if (i < tour.size() - 1) file << ",";
    }
    file << " | " << obj << " | " << timestamp;
}

std::optional<std::vector<int>> Instance::getBestKnownSolution(int idx) const {
    fs::path solutionPath = fs::path("solutions") / name_;
    if (!fs::exists(solutionPath)) {
        return std::nullopt;
    }

    std::vector<std::pair<std::string, double>> solutions;
    std::ifstream file(solutionPath);
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find(" | ");
        if (pos != std::string::npos) {
            std::string tour = line.substr(0, pos);
            double obj = std::stod(line.substr(pos + 3));
            if (std::find_if(solutions.begin(), solutions.end(),
                           [&tour](const auto& s) { return s.first == tour; }) == solutions.end()) {
                solutions.push_back({tour, obj});
            }
        }
    }

    std::sort(solutions.begin(), solutions.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (static_cast<size_t>(idx) >= solutions.size()) {
        throw std::runtime_error("Solution index out of range");
    }

    std::vector<int> tour;
    std::stringstream ss(solutions[idx].first);
    std::string item;
    while (std::getline(ss, item, ',')) {
        tour.push_back(std::stoi(item));
    }

    std::cout << "Best known solution for " << name_ << ": ";
    for (int node : tour) {
        std::cout << node << " ";
    }
    std::cout << "with objective " << solutions[idx].second << std::endl;

    return tour;
}

std::vector<boost::unordered_map<std::string, double>> Instance::getMipStart(bool useTspOnly) const {
    auto bestKnownSolution = getBestKnownSolution();
    std::vector<int> tour = bestKnownSolution && !useTspOnly ? 
                           bestKnownSolution.value() : tspSolution();
    return getVariablesFromTour(tour);
}

std::vector<boost::unordered_map<std::string, double>> Instance::getVariablesFromTour(
    const std::vector<int>& tour) const {
    
    std::vector<std::pair<int, int>> tourEdges;
    for (size_t i = 0; i < tour.size(); ++i) {
        tourEdges.push_back({tour[i], (i < tour.size() - 1) ? tour[i + 1] : 0});
    }

    boost::unordered_map<std::string, double> x;
    for (const auto& [edge, _] : edges_) {
        x[std::to_string(edge.first) + "," + std::to_string(edge.second)] = 
            std::find(tourEdges.begin(), tourEdges.end(), edge) != tourEdges.end() ? 1.0 : 0.0;
    }

    boost::unordered_map<std::string, double> u;
    for (size_t i = 0; i < tour.size(); ++i) {
        u[std::to_string(tour[i])] = static_cast<double>(i);
    }

    boost::unordered_map<std::string, double> y;
    for (const auto& [rel, _] : relations_) {
        y[std::to_string(std::get<0>(rel)) + "," + std::to_string(std::get<1>(rel)) + "," +
          std::to_string(std::get<2>(rel)) + "," + std::to_string(std::get<3>(rel))] = 0.0;
    }

    for (const auto& a : tourEdges) {
        auto it = R_a_.find(a);
        if (it == R_a_.end()) continue;

        std::vector<std::pair<int, int>> triggering;
        for (const auto& b : it->second) {
            if (std::find(tourEdges.begin(), tourEdges.end(), b) != tourEdges.end()) {
                triggering.push_back(b);
            }
        }

        if (triggering.empty()) continue;

        std::sort(triggering.begin(), triggering.end(),
                 [&tourEdges](const auto& b1, const auto& b2) {
                     return std::find(tourEdges.begin(), tourEdges.end(), b1) <
                            std::find(tourEdges.begin(), tourEdges.end(), b2);
                 });

        auto aPos = std::find(tourEdges.begin(), tourEdges.end(), a);
        triggering.erase(
            std::remove_if(triggering.begin(), triggering.end(),
                          [&tourEdges, aPos](const auto& b) {
                              return std::find(tourEdges.begin(), tourEdges.end(), b) >= aPos;
                          }),
            triggering.end());

        if (!triggering.empty()) {
            auto lastTrigger = triggering.back();
            std::string key = std::to_string(lastTrigger.first) + "," +
                             std::to_string(lastTrigger.second) + "," +
                             std::to_string(a.first) + "," + std::to_string(a.second);
            y[key] = 1.0;
        }
    }

    boost::unordered_map<std::string, double> z;
    for (const auto& [a1, a2, b1, b2] : zVarIndices_) {
        std::string key = std::to_string(a1) + "," + std::to_string(a2) + "," +
                         std::to_string(b1) + "," + std::to_string(b2);
        z[key] = u[std::to_string(a1)] + 1 <= u[std::to_string(b1)] ? 1.0 : 0.0;
    }

    return {x, y, u, z};
}

std::vector<int> Instance::tspSolution() const {
    // TODO: Implement TSP solution using Gurobi
    // This will require implementing the GurobiTSPModel class in C++
    throw std::runtime_error("TSP solution not implemented yet");
} 