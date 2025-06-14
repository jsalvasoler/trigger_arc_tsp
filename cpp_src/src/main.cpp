#include <boost/program_options.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "instance.hpp"
#include "model.hpp"

namespace po = boost::program_options;

void printSolution(const std::vector<int>& tour, double cost) {
    std::cout << "Tour: ";
    for (size_t i = 0; i < tour.size(); ++i) {
        std::cout << tour[i];
        if (i < tour.size() - 1) {
            std::cout << " -> ";
        }
    }
    std::cout << "\nTotal cost: " << cost << std::endl;
}

int main(int argc, char* argv[]) {
    try {
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
            "mip-start,m", "Use MIP start if available")("logs,l", "Show solver logs");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);

        // Create instance from file
        auto instance = Instance::loadInstanceFromFile(vm["instance-file"].as<std::string>());

        // Set up solver parameters
        SolverParameters params = {.timeLimitSec = vm["time-limit"].as<int>(),
                                   .heuristicEffort = vm["heuristic-effort"].as<double>(),
                                   .presolve = vm["presolve"].as<int>(),
                                   .mipStart = vm.count("mip-start") > 0,
                                   .logs = vm.count("logs") > 0};

        // Create and solve model
        GurobiModel model(*instance);
        model.formulate();
        model.solveModelWithParameters(params);

        // Get and print solution
        auto [tour, cost] = model.getSolutionAndCost();
        printSolution(tour, cost);
        instance->saveSolution(tour, cost);

    } catch (const po::error& e) {
        std::cerr << "Error parsing command line: " << e.what() << "\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}