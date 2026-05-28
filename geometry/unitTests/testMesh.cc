/**
 * @file testMesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for Mesh class
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/Mesh.hh"
#include "geometry/MeshUtils.hh"

#include "utils/testing_environment.hh"

#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace

TEST(MeshTest, InvalidIdUsesNegativeOneForSignedTypes)
{
    EXPECT_EQ(SimpleFluid::invalid_id<int>(), -1);
    EXPECT_EQ(SimpleFluid::invalid_id<long>(), -1L);
}

TEST(MeshTest, InvalidIdUsesMaxForUnsignedTypes)
{
    EXPECT_EQ(SimpleFluid::invalid_id<unsigned>(),
              std::numeric_limits<unsigned>::max());
    EXPECT_EQ(SimpleFluid::invalid_id<std::size_t>(),
              std::numeric_limits<std::size_t>::max());
}

TEST(MeshTest, InvalidBoundaryIdMatchesInvalidInt)
{
    EXPECT_EQ(MeshType::invalid_boundary_id, SimpleFluid::invalid_id<int>());
}

TEST(MeshTest, VtuCellTypeCodeMapsSupportedCells)
{
    EXPECT_EQ(SimpleFluid::MeshUtils::vtu_cell_type_code(
                  SimpleFluid::MeshUtils::CellType::HEXAHEDRON),
              12);
    EXPECT_EQ(SimpleFluid::MeshUtils::vtu_cell_type_code(
                  SimpleFluid::MeshUtils::CellType::TRIPRISM),
              13);
}

TEST(MeshTest, VtuCellTypeCodeRejectsUnsupportedCells)
{
    EXPECT_THROW(SimpleFluid::MeshUtils::vtu_cell_type_code(
                     SimpleFluid::MeshUtils::CellType::INVALID),
                 std::runtime_error);
}
