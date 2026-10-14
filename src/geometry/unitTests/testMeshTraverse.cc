/** @file testMeshTraverse.cc
 * @brief Register the environment for the split traversal tests.
 */
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>

namespace
{
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
} // namespace
