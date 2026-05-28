/**
 * @file testEquationValidation.cc
 * @brief Unit tests for shared equation validation helpers.
 */

#include <gtest/gtest.h>

#include "equations/EquationValidation.hh"

#include <memory>
#include <stdexcept>

namespace
{

struct DummyMesh
{
};

struct DummyField
{
    const DummyMesh& mesh() const noexcept { return *mesh_ptr; }

    const DummyMesh* mesh_ptr = nullptr;
};

TEST(EquationValidationTest, RequireNonNullMeshReturnsPointer)
{
    auto mesh = std::make_shared<DummyMesh>();
    EXPECT_EQ(SimpleFluid::EquationValidation::require_non_null_mesh(
                  mesh, "TestEquation"),
              mesh);
}

TEST(EquationValidationTest, RequireNonNullMeshThrowsForNull)
{
    std::shared_ptr<DummyMesh> mesh;
    EXPECT_THROW(SimpleFluid::EquationValidation::require_non_null_mesh(
                     mesh, "TestEquation"),
                 std::invalid_argument);
}

TEST(EquationValidationTest, RequireMeshMatchAcceptsSameMesh)
{
    DummyMesh mesh;
    DummyField field{&mesh};

    EXPECT_NO_THROW(SimpleFluid::EquationValidation::require_mesh_match(
        mesh, field, "TestEquation"));
}

TEST(EquationValidationTest, RequireMeshMatchRejectsDifferentMesh)
{
    DummyMesh expected;
    DummyMesh actual;
    DummyField field{&actual};

    EXPECT_THROW(SimpleFluid::EquationValidation::require_mesh_match(
                     expected, field, "TestEquation"),
                 std::invalid_argument);
}

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

TEST(EquationValidationTest, AssertSufficientCacheSizeAllowsLargeEnoughCache)
{
    EXPECT_NO_THROW(SimpleFluid::EquationValidation::assert_sufficient_cache_size(
        4, 4));
    EXPECT_NO_THROW(SimpleFluid::EquationValidation::assert_sufficient_cache_size(
        5, 4));
}

} // namespace
