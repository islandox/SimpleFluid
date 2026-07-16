/**
 * @file testFeedbackMapMultiRank.cc
 * @brief MPI tests for globally conservative feedback mapping.
 */

#include <gtest/gtest.h>

#include "equations/FeedbackMap.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using FeedbackCell = SimpleFluid::FeedbackMap::FeedbackCell<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace

/**
 * @brief A coarse region spanning ranks preserves its global source integral,
 *        while a rank with no local portion still receives the mapped value.
 */
TEST(FeedbackMapMultiRankTest,
     VolumeWeightedAverageIsGlobalForDistributedCoarseRegions)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType source(mesh, 0.0, "distributed_feedback_source");
    std::vector<Pack::local_ordinal_type> whole_domain_lids;
    std::vector<Pack::local_ordinal_type> single_cell_lids;
    std::array<double, 2> local_integrals{};
    std::array<double, 2> local_volumes{};

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        const auto value =
            2.0 + 0.75 * static_cast<double>(cell_gid)
          + 0.1 * static_cast<double>(cell_gid * cell_gid);
        const auto volume = mesh->cell_volume(cell_lid);
        source.set_owned_value(cell_lid, value);

        whole_domain_lids.push_back(cell_lid);
        local_integrals[0] += value * volume;
        local_volumes[0] += volume;
        if (cell_gid == 0)
        {
            single_cell_lids.push_back(cell_lid);
            local_integrals[1] += value * volume;
            local_volumes[1] += volume;
        }
    }

    std::array<double, 2> global_integrals{};
    std::array<double, 2> global_volumes{};
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_SUM,
        static_cast<int>(local_integrals.size()),
        local_integrals.data(),
        global_integrals.data());
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_SUM,
        static_cast<int>(local_volumes.size()),
        local_volumes.data(),
        global_volumes.data());

    const int local_whole_domain_count =
        static_cast<int>(whole_domain_lids.size());
    const int local_single_cell_count =
        static_cast<int>(single_cell_lids.size());
    int minimum_whole_domain_count = 0;
    int minimum_single_cell_count = 0;
    int maximum_single_cell_count = 0;
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_whole_domain_count,
        &minimum_whole_domain_count);
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_single_cell_count,
        &minimum_single_cell_count);
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_single_cell_count,
        &maximum_single_cell_count);

    ASSERT_GT(minimum_whole_domain_count, 0)
        << "The whole-domain coarse region must span every test rank.";
    ASSERT_EQ(minimum_single_cell_count, 0)
        << "At least one rank must have no local contribution.";
    ASSERT_EQ(maximum_single_cell_count, 1)
        << "Exactly one rank should own the selected fine cell.";
    ASSERT_GT(global_volumes[0], 0.0);
    ASSERT_GT(global_volumes[1], 0.0);

    const std::vector<FeedbackCell> feedback_cells{
        {"whole_domain", std::move(whole_domain_lids)},
        {"single_global_cell", std::move(single_cell_lids)}};
    const auto averages =
        SimpleFluid::FeedbackMap::volume_weighted_average<Pack>(
            source, feedback_cells);

    ASSERT_EQ(averages.size(), feedback_cells.size());
    for (size_t feedback_id = 0;
         feedback_id < feedback_cells.size();
         ++feedback_id)
    {
        const auto tolerance = std::max(
            1.0e-14,
            std::abs(global_integrals[feedback_id]) * 1.0e-12);
        EXPECT_NEAR(
            averages[feedback_id] * global_volumes[feedback_id],
            global_integrals[feedback_id],
            tolerance);
    }
}

/**
 * @brief Equal-sized but differently ordered coarse definitions are rejected
 *        collectively instead of being reduced index-wise.
 */
TEST(FeedbackMapMultiRankTest,
     RejectsInconsistentCoarseRegionNamesAndOrdering)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType source(mesh, 1.0, "inconsistent_feedback_source");
    std::vector<Pack::local_ordinal_type> lower_half_lids;
    std::vector<Pack::local_ordinal_type> upper_half_lids;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        if (cell_gid < 4)
        {
            lower_half_lids.push_back(cell_lid);
        }
        else
        {
            upper_half_lids.push_back(cell_lid);
        }
    }

    std::vector<FeedbackCell> feedback_cells;
    if (comm->getRank() == 0)
    {
        feedback_cells = {
            {"lower_half", std::move(lower_half_lids)},
            {"upper_half", std::move(upper_half_lids)}};
    }
    else
    {
        feedback_cells = {
            {"upper_half", std::move(upper_half_lids)},
            {"lower_half", std::move(lower_half_lids)}};
    }

    try
    {
        (void)SimpleFluid::FeedbackMap::volume_weighted_average<Pack>(
            source, feedback_cells);
        FAIL() << "Expected inconsistent distributed region ordering to fail.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(
            std::string(error.what()).find(
                "identical coarse-cell names and ordering"),
            std::string::npos);
    }
}
