/** @file testGeneralVoFPhiClosure.cc
 * @brief Arithmetic provenance, nonzero circulation, and bounded closure rollback.
 */
#include "FVM/AdaptiveVoF.hh"
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
            EXPECT_THROW((void) detail::close_general_vof_finite_rate(mesh, rate, radius, cell_radius, 4 * 1024 * 1024),
                detail::GeneralVoFPhiInfeasible);
        }
        else
        {
            const auto closed = detail::close_general_vof_finite_rate(mesh, rate, radius, cell_radius, 4 * 1024 * 1024);
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
    const auto closed = detail::close_general_vof_finite_rate(mesh, rate, radius, cell_radius, 4 * 1024 * 1024);
    EXPECT_EQ(closed.rate, rate);
    check_rows(closed.rate, tiny);
    EXPECT_THROW((void) detail::close_general_vof_finite_rate(mesh, rate, radius, cell_radius, 1), std::length_error);
}
} // namespace

namespace
{

} // namespace

namespace
{

} // namespace
