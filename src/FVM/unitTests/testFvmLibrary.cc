/**
 * @file testFvmLibrary.cc
 * @author islandox
 * @brief Verify compiled linear algebra is available to FVM-only consumers.
 * @version 0.1
 * @date 2026-09-17
 * @copyright Copyright (c) 2026
 */

#include <gtest/gtest.h>

#include "FVM/Operators.hh"
#include "solvers/DICPreconditioner.hh"
#include "utils/testing_environment.hh"

#include <Tpetra_Core.hpp>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(FvmLibraryTest, AppliesCompiledDICWithoutSolverLibrary)
{
    auto map = Teuchos::rcp(new Pack::map_type(3, 0, Tpetra::getDefaultComm()));
    auto matrix = SimpleFluid::FVM::identity_matrix<Pack>(map);
    matrix->scale(2.0);
    SimpleFluid::detail::DICPreconditioner<Pack> inverse(*matrix);
    Pack::vector_type input(map, true), output(map, true);
    input.putScalar(3.0);

    inverse.apply(input, output);

    const auto values = output.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
        EXPECT_DOUBLE_EQ(values(row, 0), 1.5);
}
} // namespace
