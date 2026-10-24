/** @file testGeneralVoFPhiClosure.cc
 * @brief Arithmetic provenance, nonzero circulation, and bounded closure rollback.
 */
#include "FVM/AdaptiveVoF.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid;
using LO = DefaultTpetraTypes::local_ordinal_type;
using GO = DefaultTpetraTypes::global_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

} // namespace

namespace
{
TEST(GeneralVoFPhiClosureTest, CertifiedClosureRequiresExplicitBoundedMode)
{
    Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const std::vector<long double> rate(mesh->num_owned_faces()), radius(rate.size()),
        cell_radius(mesh->num_owned_cells());
    EXPECT_THROW((void) detail::close_general_vof_finite_rate(
                     mesh, rate, radius, cell_radius, 4 * 1024 * 1024, ArithmeticMode::Ordinary),
        std::runtime_error);
    const auto closed = detail::close_general_vof_finite_rate(
        mesh, rate, radius, cell_radius, 4 * 1024 * 1024, ArithmeticMode::BoundedCertification);
    EXPECT_EQ(closed.rate, rate);
}

TEST(GeneralVoFPhiClosureTest, FiniteSparseForestPreservesCirculationAndRejectsIndependentCapacityExhaustion)
{
    Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const long double capacity = std::ldexp(1.L, -45);
    std::vector<long double> rate(mesh->num_owned_faces()), radius(rate.size(), capacity);
    std::vector<long double> cell_radius(mesh->num_owned_cells(), 6 * capacity);
    GO local_first = std::numeric_limits<GO>::max(), first = 0;
    for (std::size_t f = 0; f < rate.size(); ++f)
    {
        rate[f] = mesh->face_normal(LO(f)).x;
        local_first = std::min(local_first, mesh->face_global_id(LO(f)));
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_first, &first);
    const auto base = rate;
    const auto check_rows = [&](const std::vector<long double>& canonical, long double scale)
    {
        DefaultTpetraTypes::vector_type owned(mesh->owned_face_map()), overlap(mesh->overlap_face_map());
        DefaultTpetraTypes::import_type importer(mesh->owned_face_map(), mesh->overlap_face_map());
        {
            auto data = owned.getDataNonConst();
            for (std::size_t f = 0; f < canonical.size(); ++f)
            {
                data[f] = static_cast<double>(canonical[f] / scale);
                // This fixture's dyadic values are exactly representable after
                // scaling, so the independent import cannot erase a defect.
                EXPECT_EQ(static_cast<long double>(data[f]), canonical[f] / scale);
            }
        }
        overlap.doImport(owned, importer, Tpetra::INSERT);
        const auto data = overlap.getData();
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            long double net = 0;
            for (auto face : mesh->faces(LO(c)))
                net += (mesh->owner_cell(face) == LO(c) ? 1 : -1) * static_cast<long double>(data[face]);
            EXPECT_EQ(net, 0);
        }
    };
    for (int multiplier : {0, 1, 1024, 1})
    {
        rate = base;
        for (std::size_t f = 0; f < rate.size(); ++f)
            if (mesh->face_global_id(LO(f)) == first)
                rate[f] += multiplier * capacity / 8;
        const auto original = rate;
        if (multiplier == 1024)
        {
            EXPECT_THROW((void) detail::close_general_vof_finite_rate(
                             mesh, rate, radius, cell_radius, 4 * 1024 * 1024, ArithmeticMode::BoundedCertification),
                detail::GeneralVoFPhiInfeasible);
        }
        else
        {
            const auto closed = detail::close_general_vof_finite_rate(
                mesh, rate, radius, cell_radius, 4 * 1024 * 1024, ArithmeticMode::BoundedCertification);
            ASSERT_EQ(closed.rate.size(), rate.size());
            check_rows(closed.rate, 1);
            for (std::size_t f = 0; f < rate.size(); ++f)
            {
                EXPECT_LE(std::abs(closed.correction[f]), radius[f]);
                EXPECT_EQ(closed.rate[f], rate[f] + closed.correction[f]);
            }
        }
        EXPECT_EQ(rate, original);
    }
    // Fixed-width mantissa/exponent packets retain products far below binary64
    // range. This is an accounting test and imposes no thin-feature fit duty.
    const long double tiny =
        static_cast<long double>(std::numeric_limits<double>::denorm_min()) * std::numeric_limits<double>::denorm_min();
    ASSERT_GT(tiny, 0);
    rate = base;
    for (auto& value : rate)
        value *= tiny;
    std::fill(radius.begin(), radius.end(), 0);
    std::fill(cell_radius.begin(), cell_radius.end(), 0);
    const auto closed = detail::close_general_vof_finite_rate(
        mesh, rate, radius, cell_radius, 4 * 1024 * 1024, ArithmeticMode::BoundedCertification);
    EXPECT_EQ(closed.rate, rate);
    check_rows(closed.rate, tiny);
    EXPECT_THROW((void) detail::close_general_vof_finite_rate(
                     mesh, rate, radius, cell_radius, 1, ArithmeticMode::BoundedCertification),
        std::length_error);
}
} // namespace

namespace
{
TEST(GeneralVoFPhiClosureTest, FiniteCapacityRoutingPreservesFixedEdgesAndAvoidsNarrowShortcuts)
{
    const auto native = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{
            Meshes::cartesian_region("cycle", {{{0, 1, 2}, {0, 1, 2}, {0, 1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    const auto mesh = std::make_shared<MeshHandle<>>(native);
    const auto index = [](const auto& x) { return unsigned(x.x > 1) + 2 * unsigned(x.y > 1); };
    detail::GeneralVoFFiniteProjectionClosureTopology topology;
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        topology.cells.push_back({mesh->cell_global_id(LO(c)), {char(index(mesh->cell_centroid(LO(c))))}, 0});
    const long double delta = std::ldexp(1.L, -40);
    std::vector<long double> rate(mesh->num_owned_faces()), radius(rate.size()),
        cell_radius(mesh->num_owned_cells(), 4 * delta);
    std::vector<bool> shortcut(rate.size());
    for (std::size_t f = 0; f < rate.size(); ++f)
    {
        const auto face = LO(f);
        if (mesh->is_geometry_exterior_face(face))
            continue;
        const auto first = mesh->owner_cell(face), second = mesh->neighbor_cell(face);
        const auto a = index(mesh->cell_centroid(first)), b = index(mesh->cell_centroid(second));
        shortcut[f] = std::min(a, b) == 0 && std::max(a, b) == 1;
        if (shortcut[f])
            rate[f] = a == 0 ? delta : -delta;
        topology.faces.push_back({mesh->face_global_id(face), mesh->cell_global_id(first), mesh->cell_global_id(second),
            {char(std::min(a, b)), char(std::max(a, b))}, 0, 0});
    }
    const auto original = rate;
    for (long double shortcut_capacity : {0.L, delta / 16})
    {
        for (std::size_t f = 0; f < rate.size(); ++f)
            radius[f] = mesh->is_geometry_exterior_face(LO(f)) ? 0 : (shortcut[f] ? shortcut_capacity : 2 * delta);
        detail::GeneralVoFFiniteRateClosure closed;
        ASSERT_NO_THROW(closed = detail::close_general_vof_finite_rate(mesh, rate, radius, cell_radius, 4 * 1024 * 1024,
                            ArithmeticMode::BoundedCertification, &topology));
        DefaultTpetraTypes::vector_type owned(mesh->owned_face_map()), overlap(mesh->overlap_face_map());
        DefaultTpetraTypes::import_type importer(mesh->owned_face_map(), mesh->overlap_face_map());
        auto owned_data = owned.getDataNonConst();
        for (std::size_t f = 0; f < rate.size(); ++f)
        {
            EXPECT_LE(std::abs(closed.rate[f] - rate[f]), radius[f]);
            EXPECT_EQ(static_cast<long double>(double(closed.rate[f])), closed.rate[f]);
            owned_data[f] = double(closed.rate[f]);
            if (shortcut[f])
                EXPECT_EQ(closed.rate[f], rate[f]);
        }
        owned_data = {};
        overlap.doImport(owned, importer, Tpetra::INSERT);
        const auto data = overlap.getData();
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            long double sum = 0;
            for (const auto face : mesh->faces(LO(c)))
                sum += (mesh->owner_cell(face) == LO(c) ? 1 : -1) * static_cast<long double>(data[face]);
            EXPECT_EQ(sum, 0);
        }
        EXPECT_EQ(rate, original);
    }
    std::fill(radius.begin(), radius.end(), 0);
    EXPECT_THROW((void) detail::close_general_vof_finite_rate(
                     mesh, rate, radius, cell_radius, 4 * 1024 * 1024, ArithmeticMode::BoundedCertification, &topology),
        detail::GeneralVoFPhiInfeasible);
    EXPECT_EQ(rate, original);
}
} // namespace

namespace
{
TEST(GeneralVoFPhiClosureTest, FiniteWidestParentDepthPropagatesThroughLowerCapacityLeaf)
{
    const auto native = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{
            Meshes::cartesian_region("leaf", {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    const auto mesh = std::make_shared<MeshHandle<>>(native);
    const auto index = [](const auto& x) { return unsigned(x.x > 1) + 2 * unsigned(x.y > 1) + 4 * unsigned(x.z > 1); };
    const auto delta = std::ldexp(1.L, -40);
    detail::GeneralVoFFiniteProjectionClosureTopology topology;
    std::vector<long double> rate(mesh->num_owned_faces()), radius(rate.size()),
        cell_radius(mesh->num_owned_cells(), 32 * delta);
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        topology.cells.push_back({mesh->cell_global_id(LO(c)), {char(index(mesh->cell_centroid(LO(c))))}, 0});
    for (std::size_t f = 0; f < rate.size(); ++f)
    {
        const auto face = LO(f);
        if (mesh->is_geometry_exterior_face(face))
            continue;
        const auto first = mesh->owner_cell(face), second = mesh->neighbor_cell(face);
        const auto a = index(mesh->cell_centroid(first)), b = index(mesh->cell_centroid(second));
        const auto lo = std::min(a, b), hi = std::max(a, b);
        if (lo == 0 && hi == 1)
            radius[f] = 2 * delta;
        if ((lo == 0 && hi == 2) || (lo == 2 && hi == 3) || (lo == 1 && hi == 3))
            radius[f] = 8 * delta;
        if (lo == 1 && hi == 5)
        {
            radius[f] = delta;
            rate[f] = a == 1 ? delta : -delta;
        }
        topology.faces.push_back({mesh->face_global_id(face), mesh->cell_global_id(first), mesh->cell_global_id(second),
            {char(lo), char(hi)}, 0, 0});
    }
    // The parent1 first sees the short narrow0->1 path, then upgrades through
    // 0->2->3->1. Its leaf5 keeps the same smaller bottleneck but must also
    // update its depth so its defect reaches1 before1 is routed upstream.
    const auto original = rate;
    detail::GeneralVoFFiniteRateClosure closed;
    ASSERT_NO_THROW(closed = detail::close_general_vof_finite_rate(mesh, rate, radius, cell_radius, 4 * 1024 * 1024,
                        ArithmeticMode::BoundedCertification, &topology));
    for (std::size_t f = 0; f < rate.size(); ++f)
    {
        EXPECT_EQ(closed.rate[f], 0);
        EXPECT_EQ(closed.correction[f], closed.rate[f] - original[f]);
        EXPECT_LE(std::abs(closed.correction[f]), radius[f]);
    }
    EXPECT_EQ(rate, original);
}
} // namespace
