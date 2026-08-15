/**
 * @file testTypedefs.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for common SimpleFluid type aliases.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "dataclass/typedefs.hh"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

static_assert(std::is_same_v<SimpleFluid::real_t, double>);
static_assert(std::is_same_v<SimpleFluid::global_index_t, long long>);
static_assert(std::is_same_v<SimpleFluid::local_index_t, int>);
static_assert(std::is_same_v<SimpleFluid::ArrReal, std::vector<double>>);
static_assert(std::is_same_v<SimpleFluid::ArrInt, std::vector<int>>);
static_assert(std::is_same_v<SimpleFluid::ArrString, std::vector<std::string>>);
static_assert(std::is_same_v<SimpleFluid::ArrBool, std::vector<bool>>);
static_assert(std::is_same_v<SimpleFluid::Vec3DReal, std::array<double, 3>>);

/** @brief Verifies container and pointer aliases preserve their element types. */
TEST(TypedefsTest, TemplateAliasesPreserveElementTypes)
{
    EXPECT_TRUE((std::is_same_v<SimpleFluid::Arr<float>, std::vector<float>>));
    EXPECT_TRUE((std::is_same_v<SimpleFluid::Vec3D<int>, std::array<int, 3>>));
    EXPECT_TRUE((std::is_same_v<SimpleFluid::UP<int>, std::unique_ptr<int>>));
    EXPECT_TRUE((std::is_same_v<SimpleFluid::SP<int>, std::shared_ptr<int>>));
    EXPECT_TRUE((std::is_same_v<SimpleFluid::WP<int>, std::weak_ptr<int>>));
}

/** @brief Locks the dimension-enum values to X/Y/Z array ordering. */
TEST(TypedefsTest, DimensionValuesMatchCoordinateOrder)
{
    EXPECT_EQ(SimpleFluid::Dimension::X, 0);
    EXPECT_EQ(SimpleFluid::Dimension::Y, 1);
    EXPECT_EQ(SimpleFluid::Dimension::Z, 2);
}

} // namespace
