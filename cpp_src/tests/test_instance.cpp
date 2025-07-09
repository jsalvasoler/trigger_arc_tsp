#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>

#include "../include/instance.hpp"

namespace fs = std::filesystem;

class InstanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get the source directory path (two levels up from build directory)
        sourceDir_ = fs::current_path().parent_path();
        instancePath_ = sourceDir_ / "tests" / "instances" / "example_1.txt";
        solutionsDir_ = sourceDir_.parent_path() / "cpp_src" / "build" / "solutions";
    }

    void TearDown() override {
        // Clean up test files
        if (fs::exists(solutionsDir_ / "example_1.txt")) {
            fs::remove(solutionsDir_ / "example_1.txt");
        }
    }

    fs::path sourceDir_;
    fs::path instancePath_;
    fs::path solutionsDir_;
};

TEST_F(InstanceTest, LoadInstanceFromFile) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    EXPECT_EQ(instance->getN(), 5);
    EXPECT_EQ(instance->getA(), 8);
    EXPECT_EQ(instance->getR(), 4);
    EXPECT_EQ(instance->getName(), "example_1.txt");
    EXPECT_EQ(instance->getEdges().size(), 8);
    EXPECT_EQ(instance->getRelations().size(), 4);
}

TEST_F(InstanceTest, SolutionCorrectness) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    // Invalid because does not start at 0
    std::vector<int> invalidTour1 = {2, 1, 4, 3, 0};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour1));

    // Invalid because of wrong length
    std::vector<int> invalidTour2 = {0, 2, 1, 4, 3, 2, 1};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour2));

    // Invalid because of duplicate nodes
    std::vector<int> invalidTour3 = {0, 2, 1, 4, 3, 3};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour3));

    // Invalid because of wrong node
    std::vector<int> invalidTour4 = {0, 2, 1, 4, 18};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour4));
}

TEST_F(InstanceTest, WriteSolutionToFile) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    // Test invalid tour
    std::vector<int> invalidTour = {2, 1, 4, 3};
    EXPECT_THROW(instance->saveSolution(invalidTour, 71.0), std::runtime_error);

    // Test valid tour
    std::vector<int> validTour = {0, 2, 1, 4, 3};
    instance->saveSolution(validTour, 71.0);

    // Verify file contents
    std::ifstream file(solutionsDir_ / "example_1.txt");
    std::string line;
    std::getline(file, line);
    EXPECT_TRUE(line.find("0,2,1,4,3 | 71 | ") == 0);
}

TEST_F(InstanceTest, MIPStartFromTsp1) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    std::vector<int> tour;
    std::vector<int> partial_tour;
    std::vector<double> expected_partial_costs;

    // Case 1
    tour = {0, 2, 1, 4, 3, 0};
    partial_tour = {0};
    expected_partial_costs = {10, 25, 36, 56, 71};
    for (int i = 1; i < (int)tour.size(); ++i) {
        partial_tour.push_back(tour[i]);
        for (int j = 0; j < i; ++j) {  // j is the start index
            double cost = instance->computePartialTourCost(partial_tour, j);
            EXPECT_DOUBLE_EQ(cost,
                             j == 0
                                 ? expected_partial_costs[i - 1]
                                 : expected_partial_costs[i - 1] - expected_partial_costs[j - 1]);
        }
    }

    // Case 2
    tour = {0, 3, 2, 1, 4, 0};
    partial_tour = {0};
    expected_partial_costs = {10, 30, 45, 56, 62};
    for (int i = 1; i < (int)tour.size(); ++i) {
        partial_tour.push_back(tour[i]);
        for (int j = 0; j < i; ++j) {  // j is the start index
            double cost = instance->computePartialTourCost(partial_tour, j);
            EXPECT_DOUBLE_EQ(cost,
                             j == 0
                                 ? expected_partial_costs[i - 1]
                                 : expected_partial_costs[i - 1] - expected_partial_costs[j - 1]);
        }
    }
}

TEST_F(InstanceTest, TwoOptMethod) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    std::vector<int> tour = {0, 2, 1, 4, 3, 0};
    auto tour_tmp = tour;

    instance->get_two_opt_neigbhor(tour);  // Apply 2-opt mutation

    // Check that the tour length remains the same
    EXPECT_EQ(tour.size(), tour_tmp.size());
}

TEST_F(InstanceTest, AllTwoOptNeighbors) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    std::vector<int> tour = {0, 2, 1, 4, 3};

    // Call method under test
    auto neighbors = instance->get_all_two_opt_neigbhor(tour);

    // Total expected neighbors = n * (n - 1) / 2 for swap(i, j)
    int n = tour.size();
    int expected_count = n * (n - 1) / 2;

    // Check number of neighbors
    EXPECT_EQ(neighbors.size(), expected_count);

    // All neighbors should be different from original tour but same size
    for (const auto& neighbor : neighbors) {
        EXPECT_EQ(neighbor.size(), tour.size());
        EXPECT_NE(neighbor, tour); // Ensure at least one element changed
    }

    // Optional: Check that all neighbors are permutations of the original tour
    for (const auto& neighbor : neighbors) {
        auto sorted_neighbor = neighbor;
        auto sorted_original = tour;
        std::sort(sorted_neighbor.begin(), sorted_neighbor.end());
        std::sort(sorted_original.begin(), sorted_original.end());
        EXPECT_EQ(sorted_neighbor, sorted_original);
    }
}