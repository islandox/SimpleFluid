/** @file OrdinaryScalar.hh
 * @brief Finite scalar arithmetic with explicit validation and work accounting.
 */
#pragma once
#include <cmath>

namespace SimpleFluid::detail::ordinary_fp
{
template<class Policy> struct Scalar
{
    static_assert(Policy::ordinary());
    long double value = 0;
    explicit Scalar(long double input = 0) : value(input) { Policy::finite(input); }
    template<class OtherPolicy> Scalar(const Scalar<OtherPolicy>& other) : value(other.value) {}
    bool zero() const { return value == 0; }
};

template<class Policy> Scalar<Policy> operator-(Scalar<Policy> a)
{
    return Scalar<Policy>(-a.value);
}

template<class Policy> Scalar<Policy> operator+(Scalar<Policy> a, Scalar<Policy> b)
{
    Policy::scalar_operation();
    Policy::operation();
    return Scalar<Policy>(a.value + b.value);
}

template<class Policy> Scalar<Policy> operator-(Scalar<Policy> a, Scalar<Policy> b)
{
    return a + -b;
}

template<class Policy> Scalar<Policy> operator*(Scalar<Policy> a, Scalar<Policy> b)
{
    Policy::scalar_operation();
    Policy::operation();
    return Scalar<Policy>(a.value * b.value);
}

template<class Policy> Scalar<Policy> operator/(Scalar<Policy> a, Scalar<Policy> b)
{
    Policy::scalar_operation();
    if (b.value == 0)
        Policy::reject("zero divisor in ordinary geometry.");
    Policy::operation();
    return Scalar<Policy>(a.value / b.value);
}

template<class Policy> Scalar<Policy> scaled(Scalar<Policy> a, int exponent)
{
    Policy::scalar_operation();
    Policy::operation();
    return Scalar<Policy>(std::scalbn(a.value, exponent));
}
} // namespace SimpleFluid::detail::ordinary_fp
