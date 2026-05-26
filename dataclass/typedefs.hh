/**
 * @file typedefs.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <cstdint>
#include <concepts>
#include <array>
#include <vector>
#include <string>
#include <memory>

namespace SimpleFluid
{
    using real_t = double;
    using global_index_t = long long;
    using local_index_t = int;

    using ArrReal = std::vector<real_t>;
    using ArrInt = std::vector<int>;
    using ArrString = std::vector<std::string>;
    using ArrBool = std::vector<bool>;
    template <class T>
    using Arr = std::vector<T>;

    using Vec3DReal = std::array<real_t, 3>;
    template <class T>
    using Vec3D = std::array<T, 3>;

    template <class T>
    using UP = std::unique_ptr<T>;

    template <class T>
    using SP = std::shared_ptr<T>;

    template <class T>
    using WP = std::weak_ptr<T>;

    enum Dimension : uint8_t
    {
        X = 0,
        Y = 1,
        Z = 2
    };
}
