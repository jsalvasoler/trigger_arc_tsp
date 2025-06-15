#include <boost/json/src.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "instance.hpp"
#include "model.hpp"

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
        "time-limit,t", po::value<int>()->default_value(60), "Time limit in seconds")(
        "heuristic-effort,e",
        po::value<double>()->default_value(0.05),
        "Heuristic effort (0.0 to 1.0)")(
        "presolve,p",
        po::value<int>()->default_value(-1),
        "Presolve level (-1: automatic, 0: off, 1: conservative, 2: aggressive)")(
        "mip-start,m", "Use MIP start if available")("logs,l", "Show solver logs")(
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

    SolverParameters params = {.timeLimitSec = vm["time-limit"].as<int>(),
                               .heuristicEffort = vm["heuristic-effort"].as<double>(),
                               .presolve = vm["presolve"].as<int>(),
                               .mipStart = vm.count("mip-start") > 0,
                               .logs = vm.count("logs") > 0};

    GurobiModel model(*instance);
    model.formulate();
    model.solveModelWithParameters(params);

    auto [tour, cost] = model.getSolutionAndCost();
    instance->saveSolution(tour, cost);

    // save output to file

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    boost::json::object json;
    json["instance_name"] = instance->getName();
    json["instance_file"] = vm["instance-file"].as<std::string>();
    json["method"] = "gurobi";
    json["time_limit"] = params.timeLimitSec;
    json["heuristic_effort"] = params.heuristicEffort;
    json["presolve"] = params.presolve;
    json["mip_start"] = params.mipStart ? "true" : "false";
    json["tour"] = tourToString(tour);
    json["cost"] = cost;
    json["mip_gap"] = model.getMIPGap();
    json["wall_time"] = model.getWallTime();
    json["obj_bound"] = model.getObjBound();
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
