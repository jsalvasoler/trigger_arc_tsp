#ifndef METHOD_HPP
#define METHOD_HPP

#include "instance.hpp"

class Method {
public:
    explicit Method(const Instance& instance);
    virtual ~Method() = default;

    virtual void run() = 0;
    virtual std::vector<int> getSolution() const = 0;

protected:
    const Instance& instance_;
};

#endif  // METHOD_HPP
