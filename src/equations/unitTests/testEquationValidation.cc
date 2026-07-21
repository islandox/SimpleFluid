/**
 * @file testEquationValidation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for shared equation validation helpers.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/EquationValidation.hh"

#include <memory>
#include <stdexcept>

namespace
{

/** @brief Minimal mesh token used by pointer-validation tests. */
struct DummyMesh
{
};

/** @brief Minimal field exposing the mesh-reference validation contract. */
struct DummyField
{
    const DummyMesh& mesh() const noexcept { return *mesh_ptr; }

    const DummyMesh* mesh_ptr = nullptr;
};

/** @brief Verifies that non-null mesh validation returns the supplied pointer. */
TEST(EquationValidationTest, RequireNonNullMeshReturnsPointer)
{
    auto mesh = std::make_shared<DummyMesh>();
    EXPECT_EQ(SimpleFluid::EquationValidation::require_non_null_mesh(
                  mesh, "TestEquation"),
              mesh);
}

/** @brief Verifies that non-null mesh validation rejects a null pointer. */
TEST(EquationValidationTest, RequireNonNullMeshThrowsForNull)
{
    std::shared_ptr<DummyMesh> mesh;
    EXPECT_THROW(SimpleFluid::EquationValidation::require_non_null_mesh(
                     mesh, "TestEquation"),
                 std::invalid_argument);
}

/** @brief Verifies that mesh matching accepts a field on the expected mesh. */
TEST(EquationValidationTest, RequireMeshMatchAcceptsSameMesh)
{
    DummyMesh mesh;
    DummyField field{&mesh};

    EXPECT_NO_THROW(SimpleFluid::EquationValidation::require_mesh_match(
        mesh, field, "TestEquation"));
}

/** @brief Verifies that mesh matching rejects a field on a different mesh. */
TEST(EquationValidationTest, RequireMeshMatchRejectsDifferentMesh)
{
    DummyMesh expected;
    DummyMesh actual;
    DummyField field{&actual};

    EXPECT_THROW(SimpleFluid::EquationValidation::require_mesh_match(
                     expected, field, "TestEquation"),
                 std::invalid_argument);
}

/** @brief Verifies that non-negative validation accepts zero and rejects negatives. */
TEST(EquationValidationTest, RequireNonNegativeRejectsNegative)
{
    EXPECT_NO_THROW(SimpleFluid::EquationValidation::require_non_negative(
        0.0, "coefficient", "TestEquation"));
    EXPECT_NO_THROW(SimpleFluid::EquationValidation::require_non_negative(
        1.0, "coefficient", "TestEquation"));
    EXPECT_THROW(SimpleFluid::EquationValidation::require_non_negative(
                     -1.0, "coefficient", "TestEquation"),
                 std::invalid_argument);
}

/** @brief Verifies that cache-size validation accepts sufficient storage. */
TEST(EquationValidationTest, AssertSufficientCacheSizeAllowsLargeEnoughCache)
{
    EXPECT_NO_THROW(SimpleFluid::EquationValidation::assert_sufficient_cache_size(
        4, 4));
    EXPECT_NO_THROW(SimpleFluid::EquationValidation::assert_sufficient_cache_size(
        5, 4));
}

} // namespace
