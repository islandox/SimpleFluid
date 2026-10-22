/** @file testGeneralVoFExactPressure.cc
 * @brief Independent rational graph pressure, empty owners and bounded failure.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "solvers/details/GeneralVoFExactPressure.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid;
using M = Meshes::ExactGeometryMeasure;
using LO = DefaultTpetraTypes::local_ordinal_type;
namespace AMR = Meshes::AMR;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
M ratio(double a, double b)
{
    return M::from_value(a).divided_by(M::from_value(b));
}
SP<const MeshHandle<>> line_mesh(bool all_on_root = false)
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
            grid.xy_nodes.push_back({double(x), double(y), 0});
    for (unsigned x = 0; x < 3; ++x)
        grid.base_cells.push_back({x, x + 1, x + 5, x + 4});
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR source(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 3; ++gid)
        if ((all_on_root ? 0 : int(gid % std::min(3, comm->getSize()))) == comm->getRank())
            owned.push_back(gid);
    auto boot = source.detach_owned(owned, {'e', 'x', 'a', 'c', 't', '_', 'p', 'r', 'e', 's', 's', 'u', 'r', 'e'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    return AMR::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor()).mesh;
}
std::vector<detail::GeneralVoFExactPressureFace> faces(const MeshHandle<>& mesh)
{
    std::vector<detail::GeneralVoFExactPressureFace> result;
    for (std::size_t i = 0; i < mesh.num_owned_faces(); ++i)
    {
        const auto face = LO(i);
        if (mesh.is_geometry_exterior_face(face))
            continue;
        const auto point = mesh.face_centroid(face);
        result.push_back({mesh.face_global_id(face), mesh.cell_global_id(mesh.owner_cell(face)),
            mesh.cell_global_id(mesh.neighbor_cell(face)), point.x == 1 ? ratio(3, 7) : ratio(5, 11)});
    }
    return result;
}
std::vector<M> rhs(const MeshHandle<>& mesh, double scale)
{
    std::vector<M> result;
    for (std::size_t i = 0; i < mesh.num_owned_cells(); ++i)
    {
        const auto x = mesh.cell_centroid(LO(i)).x;
        result.push_back((x < 1 ? ratio(-1, 7) : x < 2 ? ratio(1, 7) - ratio(5, 66) : ratio(5, 66)).scaled(scale));
    }
    return result;
}
M expected(double x, double scale)
{
    // Analytic solution (0,1/3,1/2) with its arithmetic mean5/18 removed.
    return ((x < 1 ? M{} : x < 2 ? ratio(1, 3) : ratio(1, 2)) - ratio(5, 18)).scaled(scale);
}
TEST(GeneralVoFExactPressureTest, RationalThreeCellGraphReachesExactlyZeroOriginalResidual)
{
    const auto mesh = line_mesh();
    const auto input = faces(*mesh);
    const auto b = rhs(*mesh, 1);
    const auto result = detail::solve_general_vof_exact_pressure(mesh, input, b, 2, 16 * 1024 * 1024);
    EXPECT_LE(result.iterations, 2);
    EXPECT_GT(result.iterations, 0);
    EXPECT_GE(result.operator_applications, std::size_t(result.iterations + 1));
    if (result.owned_pressure.empty() && result.overlap_pressure.empty() && result.face_rate_increment.empty())
        EXPECT_EQ(result.storage_bytes, 0u);
    else
        EXPECT_GT(result.storage_bytes, 0u);
    EXPECT_LE(result.storage_bytes, result.admitted_storage_bytes);
    ASSERT_EQ(result.owned_pressure.size(), mesh->num_owned_cells());
    ASSERT_EQ(result.overlap_pressure.size(), mesh->num_cells());
    for (std::size_t i = 0; i < result.owned_pressure.size(); ++i)
        EXPECT_EQ(result.owned_pressure[i].compare(expected(mesh->cell_centroid(LO(i)).x, 1)), 0);
    for (std::size_t i = 0; i < result.overlap_pressure.size(); ++i)
        EXPECT_EQ(result.overlap_pressure[i].compare(expected(mesh->cell_centroid(LO(i)).x, 1)), 0);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const auto local = mesh->owned_face_map()->getLocalElement(input[i].face);
        const auto first = mesh->cell_centroid(mesh->owner_cell(local)).x;
        const auto second = mesh->cell_centroid(mesh->neighbor_cell(local)).x;
        const auto difference = expected(first, 1) - expected(second, 1);
        const auto oracle =
            difference.sign() ? difference.divided_by(M::from_value(1).divided_by(input[i].mobility)) : M{};
        EXPECT_EQ(result.face_rate_increment[i].compare(oracle), 0);
    }
    int local_empty = mesh->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (mesh->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_GT(empty, 0);
}
TEST(GeneralVoFExactPressureTest, BelowRangePressureAndBudgetFailureNeverRoundTheRationalSolution)
{
    const auto mesh = line_mesh();
    auto input = faces(*mesh);
    const double eta = std::numeric_limits<double>::denorm_min();
    const auto b = rhs(*mesh, eta);
    EXPECT_THROW(
        (void) detail::solve_general_vof_exact_pressure(mesh, input, b, 0, 16 * 1024 * 1024), std::runtime_error);
    EXPECT_THROW(
        (void) detail::solve_general_vof_exact_pressure(mesh, input, b, 1, 16 * 1024 * 1024), std::runtime_error);
    EXPECT_THROW((void) detail::solve_general_vof_exact_pressure(mesh, input, b, 2, 1), std::runtime_error);
    auto invalid = input;
    if (!invalid.empty())
        invalid.front().mobility = M::from_value(-1);
    EXPECT_THROW(
        (void) detail::solve_general_vof_exact_pressure(mesh, invalid, b, 2, 16 * 1024 * 1024), std::runtime_error);
    const auto result = detail::solve_general_vof_exact_pressure(mesh, input, b, 2, 16 * 1024 * 1024);
    for (std::size_t i = 0; i < result.owned_pressure.size(); ++i)
    {
        EXPECT_EQ(result.owned_pressure[i].compare(expected(mesh->cell_centroid(LO(i)).x, eta)), 0);
        EXPECT_NE(result.owned_pressure[i].sign(), 0);
        EXPECT_THROW((void) result.owned_pressure[i].certified().as_double(), std::exception);
    }
    auto incompatible = b;
    if (!incompatible.empty())
        incompatible.front() = incompatible.front() + M::from_value(eta);
    EXPECT_THROW(
        detail::solve_general_vof_exact_pressure(mesh, input, incompatible, 3, 16 * 1024 * 1024), std::runtime_error);
    const auto recovered = detail::solve_general_vof_exact_pressure(mesh, input, b, 2, 16 * 1024 * 1024);
    for (std::size_t i = 0; i < result.owned_pressure.size(); ++i)
        EXPECT_EQ(recovered.owned_pressure[i].compare(result.owned_pressure[i]), 0);
}
TEST(GeneralVoFExactPressureTest, RepeatedEndpointAndFaceQueriesAuthenticateEveryOriginalIncidence)
{
    // Both canonical faces and all three rows on rank0 force repeated middle
    // endpoint and repeated face-authority queries, including MPI empty owners.
    const auto mesh = line_mesh(true);
    const auto input = faces(*mesh);
    const auto b = rhs(*mesh, 1);
    const auto result = detail::solve_general_vof_exact_pressure(mesh, input, b, 2, 16 * 1024 * 1024);
    for (std::size_t i = 0; i < result.owned_pressure.size(); ++i)
        EXPECT_EQ(result.owned_pressure[i].compare(expected(mesh->cell_centroid(LO(i)).x, 1)), 0);
    if (mesh->owned_cell_map()->getComm()->getRank() != 0)
        EXPECT_TRUE(result.owned_pressure.empty());
}
TEST(GeneralVoFExactPressureTest, InnerProductHasSeparateAggregateLeaseAndRecoversAfterLocalBudgetRejection)
{
    const auto mesh = line_mesh(true);
    auto input = faces(*mesh);
    for (auto& face : input)
        face.mobility = M::from_value(1);
    // The line-graph eigenmode (1,0,-1) has eigenvalue1. Its below-range
    // common scale grows inner-product temporaries while alpha cancels to1.
    auto denominator = M::from_value(2);
    for (unsigned i = 0; i < 13; ++i)
        denominator = denominator.squared();
    const auto pressure = [&](double x) { return M::from_value(x < 1 ? 1 : x < 2 ? 0 : -1).divided_by(denominator); };
    std::vector<M> b;
    for (std::size_t i = 0; i < mesh->num_owned_cells(); ++i)
        b.push_back(pressure(mesh->cell_centroid(LO(i)).x));
    M norm;
    std::size_t largest_row = 0;
    for (const auto& value : b)
    {
        largest_row = std::max(largest_row, value.storage_bytes());
        norm = norm + value.squared();
    }
    if (!b.empty())
        EXPECT_GT(norm.storage_bytes(), largest_row);
    constexpr std::size_t budget = 768 * 1024;
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    EXPECT_THROW(
        (void) detail::solve_general_vof_exact_pressure(mesh, input, b, 1, rank == 0 ? 1 : budget), std::runtime_error);
    const auto result = detail::solve_general_vof_exact_pressure(mesh, input, b, 1, budget);
    EXPECT_EQ(result.iterations, 1);
    for (std::size_t i = 0; i < result.owned_pressure.size(); ++i)
        EXPECT_EQ(result.owned_pressure[i].compare(pressure(mesh->cell_centroid(LO(i)).x)), 0);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const auto face = mesh->owned_face_map()->getLocalElement(input[i].face);
        const auto first = pressure(mesh->cell_centroid(mesh->owner_cell(face)).x);
        const auto second = pressure(mesh->cell_centroid(mesh->neighbor_cell(face)).x);
        const auto oracle = first - second;
        EXPECT_EQ(result.face_rate_increment[i].compare(oracle), 0);
    }
    EXPECT_LE(result.admitted_storage_bytes, budget);
}
TEST(GeneralVoFExactPressureTest, VectorProductCompactsOnlyAllocationSlackWithinOriginalLease)
{
    const auto mesh = line_mesh(true);
    const auto input = faces(*mesh);
    auto denominator = M::from_value(2);
    for (unsigned i = 0; i < 12; ++i)
        denominator = denominator.squared();
    auto b = rhs(*mesh, 1);
    for (auto& value : b)
        value = value.divided_by(denominator);
    constexpr std::size_t budget = 768 * 1024;
    const auto result = detail::solve_general_vof_exact_pressure(mesh, input, b, 2, budget);
    for (std::size_t i = 0; i < result.owned_pressure.size(); ++i)
        EXPECT_EQ(
            result.owned_pressure[i].compare(expected(mesh->cell_centroid(LO(i)).x, 1).divided_by(denominator)), 0);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const auto face = mesh->owned_face_map()->getLocalElement(input[i].face);
        const auto difference = (expected(mesh->cell_centroid(mesh->owner_cell(face)).x, 1) -
                                 expected(mesh->cell_centroid(mesh->neighbor_cell(face)).x, 1))
                                    .divided_by(denominator);
        const auto oracle = difference.divided_by(M::from_value(1).divided_by(input[i].mobility));
        EXPECT_EQ(result.face_rate_increment[i].compare(oracle), 0);
    }
    EXPECT_LE(result.admitted_storage_bytes, budget);
}
} // namespace
