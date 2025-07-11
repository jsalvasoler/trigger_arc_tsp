#pragma once

#include <random>
#include <vector>

#include "method.hpp"

class SimpleRandomizedConstruction : public Method {
public:
    explicit SimpleRandomizedConstruction(const Instance& instance);
    ~SimpleRandomizedConstruction() override = default;

    void run() override;
    std::vector<int> getSolution() const override;

private:
    std::vector<int> constructSolution();

    std::vector<int> tour_;
    mutable std::mt19937 rng_;
};