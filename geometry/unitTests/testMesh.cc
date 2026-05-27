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

#include "utils/testing_environment.hh"

#include <stdexcept>
#include <string>

namespace
{

using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace