#include <boost/json/src.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "grasp.hpp"
#include "instance.hpp"
#include "mip_randomized_construction.hpp"
#include "model.hpp"
#include "randomized_greedy.hpp"

namespace po = boost::program_options;

std::string tourToString(const std::vector<int>& tour) {
    std::stringstream ss;
    for (size_t i = 0; i < tour.size(); ++i) {
        ss << tour[i];
        if (i < tour.size() - 1) {
            ss << ",";
        }
    }
    return ss.str();
}

int main(int argc, char* argv[]) {
    po::variables_map vm;
    po::options_description desc("Trigger Arc TSP Solver");

    desc.add_options()("help,h", "Show help message")(
        "instance-file,i", po::value<std::string>()->required(), "Path to the instance file")(
        "method",
        po::value<std::string>()->default_value("gurobi"),
        "Solution method (gurobi, randomized_greedy, mip_randomized_construction, or grasp)")(
        "alpha",
        po::value<double>()->default_value(0.3),
        "Alpha parameter for randomized greedy (0.0 to 1.0)")(
        "time-limit,t", po::value<int>()->default_value(60), "Time limit in seconds")(
        "heuristic-effort,e",
        po::value<double>()->default_value(0.05),
        "Heuristic effort (0.0 to 1.0)")(
        "presolve,p",
        po::value<int>()->default_value(-1),
        "Presolve level (-1: automatic, 0: off, 1: conservative, 2: aggressive)")(
        "mip-start,m", "Use MIP start if available")("logs,l", "Show solver logs")(
        "n-trials", po::value<int>()->default_value(10), "Number of trials for GRASP")(
        "beta", po::value<double>()->default_value(0.5), "Beta parameter for GRASP")(
        "start-with-best", "Start GRASP with the best known solution")(
        "constructive-heuristic",
        po::value<std::string>()->default_value("RandomizedGreedy"),
        "Constructive heuristic for GRASP (RandomizedGreedy, MIPRandomizedGreedyBias, "
        "MIPRandomizedGreedyRandom)")("local-searches",
                                      po::value<std::vector<std::string>>()->multitoken(),
                                      "Local searches for GRASP (TwoOpt, SwapTwo, Relocate)")(
        "output-dir,o",
        po::value<std::string>()->default_value("output"),
        "Path to the output directory");

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error parsing command line: " << e.what() << "\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    auto instance = Instance::loadInstanceFromFile(vm["instance-file"].as<std::string>());
    std::vector<int> tour;
    double cost = std::numeric_limits<double>::infinity();
    double mipGap = 0.0;
    double wallTime = 0.0;
    double objBound = 0.0;
    std::string methodName = vm["method"].as<std::string>();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (methodName == "gurobi") {
        SolverParameters params = {.timeLimitSec = vm["time-limit"].as<int>(),
                                   .heuristicEffort = vm["heuristic-effort"].as<double>(),
                                   .presolve = vm["presolve"].as<int>(),
                                   .mipStart = vm.count("mip-start") > 0,
                                   .logs = vm.count("logs") > 0};

        GurobiModel model(*instance);
        model.formulate();
        model.solveModelWithParameters(params);

        std::tie(tour, cost) = model.getSolutionAndCost();
        mipGap = model.getMIPGap();
        wallTime = model.getWallTime();
        objBound = model.getObjBound();
    } else if (methodName == "randomized_greedy") {
        double alpha = vm["alpha"].as<double>();
        RandomizedGreedyConstruction greedy(*instance, alpha);
        greedy.run();
        tour = greedy.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else if (methodName == "mip_randomized_construction") {
        int timeLimitSec = vm["time-limit"].as<int>();

        MIPRandomizedConstruction mipRC(*instance, timeLimitSec);
        mipRC.run();
        tour = mipRC.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else if (methodName == "grasp") {
        int n_trials = vm["n-trials"].as<int>();
        double alpha = vm["alpha"].as<double>();
        double beta = vm["beta"].as<double>();
        bool start_with_best = vm.count("start-with-best") > 0;
        int time_limit_sec = vm["time-limit"].as<int>();

        std::string constructive_heuristic_str = vm["constructive-heuristic"].as<std::string>();
        ConstructiveHeuristicType constructive_heuristic;
        if (constructive_heuristic_str == "RandomizedGreedy") {
            constructive_heuristic = ConstructiveHeuristicType::RandomizedGreedy;
        } else if (constructive_heuristic_str == "MIPRandomizedGreedyBias") {
            constructive_heuristic = ConstructiveHeuristicType::MIPRandomizedGreedyBias;
        } else if (constructive_heuristic_str == "MIPRandomizedGreedyRandom") {
            constructive_heuristic = ConstructiveHeuristicType::MIPRandomizedGreedyRandom;
        } else {
            std::cerr << "Error: Unknown constructive heuristic '" << constructive_heuristic_str
                      << "'\n";
            return 1;
        }

        std::vector<LocalSearch> local_searches;
        if (vm.count("local-searches")) {
            for (const auto& ls_str : vm["local-searches"].as<std::vector<std::string>>()) {
                if (ls_str == "TwoOpt") {
                    local_searches.push_back(LocalSearch::TwoOpt);
                } else if (ls_str == "SwapTwo") {
                    local_searches.push_back(LocalSearch::SwapTwo);
                } else if (ls_str == "Relocate") {
                    local_searches.push_back(LocalSearch::Relocate);
                } else {
                    std::cerr << "Error: Unknown local search '" << ls_str << "'\n";
                    return 1;
                }
            }
        }

        GRASP grasp(*instance,
                    n_trials,
                    alpha,
                    beta,
                    constructive_heuristic,
                    local_searches,
                    vm.count("logs") > 0,
                    start_with_best,
                    time_limit_sec);
        grasp.run();
        tour = grasp.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else {
        std::cerr << "Error: Unknown method '" << methodName << "'\n";
        return 1;
    }

    if (!tour.empty()) {
        instance->saveSolution(tour, cost);
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    boost::json::object json;
    json["instance_name"] = instance->getName();
    json["instance_file"] = vm["instance-file"].as<std::string>();
    json["method"] = methodName;
    if (methodName == "randomized_greedy") {
        json["alpha"] = vm["alpha"].as<double>();
    } else if (methodName == "mip_randomized_construction") {
        json["time_limit"] = vm["time-limit"].as<int>();
    } else {
        json["time_limit"] = vm["time-limit"].as<int>();
        json["heuristic_effort"] = vm["heuristic-effort"].as<double>();
        json["presolve"] = vm["presolve"].as<int>();
        json["mip_start"] = vm.count("mip-start") > 0 ? "true" : "false";
        json["mip_gap"] = mipGap;
        json["obj_bound"] = objBound;
    }
    json["tour"] = tourToString(tour);
    json["cost"] = cost;
    json["wall_time"] = wallTime;
    json["timestamp"] = timestamp;

    std::string outputDir = vm["output-dir"].as<std::string>();
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    std::string outputPath = outputDir + "/" + timestamp + "_" + instance->getName() + ".json";
    std::ofstream outputFile(outputPath);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open output file: " << outputPath << std::endl;
        return 1;
    }

    outputFile << std::setw(4) << json << std::endl;
    outputFile.close();

    return 0;
}
