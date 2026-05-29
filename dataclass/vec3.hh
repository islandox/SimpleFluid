/**
 * @file vec3.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief A generic 3D vector class with arithmetic and geometric operations.
 * @version 0.1
 * @date 2026-05-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include "typedefs.hh"

#include <array>
#include <cstddef>
#include <cmath>
#include <limits>

namespace SimpleFluid
{

/**
 * @brief A 3D vector class
 * 
 * @tparam Scalar 
 */
template <std::convertible_to<double> Scalar = real_t>
struct vec3
{
    using scalar_t = Scalar;

    scalar_t x, y, z;

    constexpr vec3() : x(0), y(0), z(0) {}

    constexpr vec3(scalar_t x_, scalar_t y_, scalar_t z_) 
        : x(x_), y(y_), z(z_) {}

    constexpr vec3(const std::array<scalar_t, 3>& arr)
        : x(arr[0]), y(arr[1]), z(arr[2]) {}

    constexpr scalar_t& component(std::size_t index) {
        return index == 0 ? x : (index == 1 ? y : z);
    }

    constexpr scalar_t component(std::size_t index) const {
        return index == 0 ? x : (index == 1 ? y : z);
    }

    // Arithmetic operators
    constexpr vec3 operator+(const vec3& v) const {
        return vec3(x + v.x, y + v.y, z + v.z);
    }

    constexpr vec3 operator-(const vec3& v) const {
        return vec3(x - v.x, y - v.y, z - v.z);
    }

    constexpr vec3 operator*(scalar_t s) const {
        return vec3(x * s, y * s, z * s);
    }

    constexpr vec3 operator/(scalar_t s) const {
        return vec3(x / s, y / s, z / s);
    }

    constexpr auto operator<=> (const vec3&) const = default;

    // Dot product
    constexpr scalar_t dot(const vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Cross product
    constexpr vec3 cross(const vec3& v) const {
        return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    // 2-Norm
    constexpr scalar_t norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};
}
