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
        solutionsDir_ = sourceDir_.parent_path() / "solutions";
        
        // Create necessary directories
        fs::create_directories("solutions/examples");
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 