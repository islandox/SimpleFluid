/**
 * @file testFeedbackMapMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI tests for globally conservative feedback mapping.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/FeedbackMap.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using FeedbackCell = SimpleFluid::FeedbackMap::FeedbackCell<Pack>;
using CouplingDriver =
    SimpleFluid::FeedbackMap::PlaceholderOuterCouplingDriver<Pack>;

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

/** @brief Rank-local invalid imported power is rejected collectively. */
TEST(FeedbackMapMultiRankTest,
     RejectsInvalidPowerWithoutStrandingPeerRanks)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType power(mesh, 5.0, "qdot_fission");
    std::vector<double> imported(mesh->num_owned_cells(), 8.0);
    ASSERT_FALSE(imported.empty());
    if (comm->getRank() == 1)
    {
        imported.front() = -1.0;
    }

    try
    {
        SimpleFluid::FeedbackMap::import_power_density<Pack>(
            power, imported);
        FAIL() << "Expected rank-local invalid power to fail collectively.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(
            std::string(error.what()).find("finite and non-negative"),
            std::string::npos);
    }
}

/** @brief Rank-dependent registry fields fail before per-field mapping. */
TEST(FeedbackMapMultiRankTest,
     RejectsInconsistentFeedbackFieldRegistry)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType gas_fraction(mesh, 0.1, "alpha_g");
    FieldType density(mesh, 950.0, "rhoFeedback");
    FieldType precursor(mesh, 2.0, "C_1");
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);
    if (comm->getRank() == 0)
    {
        registry.register_precursor_group(1, precursor);
    }

    std::vector<Pack::local_ordinal_type> local_cells;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_cells.push_back(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    const std::vector<FeedbackCell> feedback_cells{
        {"whole_domain", std::move(local_cells)}};

    try
    {
        (void)registry.export_snapshot(feedback_cells);
        FAIL()
            << "Expected inconsistent feedback fields to fail collectively.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(
            std::string(error.what()).find("identical field names"),
            std::string::npos);
    }
}

/** @brief Rank-dependent coupling controls are rejected on every rank. */
TEST(FeedbackMapMultiRankTest,
     RejectsInconsistentOuterCouplingConfiguration)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType gas_fraction(mesh, 0.1, "alpha_g");
    FieldType density(mesh, 950.0, "rhoFeedback");
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);

    std::vector<Pack::local_ordinal_type> local_cells;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_cells.push_back(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    const std::vector<FeedbackCell> feedback_cells{
        {"whole_domain", std::move(local_cells)}};

    const auto expect_collective_rejection =
        [](auto&& operation, const std::string& expected_message)
        {
            bool rejected = false;
            try
            {
                operation();
            }
            catch (const std::invalid_argument& error)
            {
                rejected = true;
                EXPECT_NE(
                    std::string(error.what()).find(expected_message),
                    std::string::npos);
            }
            EXPECT_TRUE(rejected);
        };

    const auto rank_dependent_group_count =
        comm->getRank() == 0 ? size_t{0} : size_t{1};
    expect_collective_rejection(
        [&]
        {
            registry.require_standard_fields(
                rank_dependent_group_count);
        },
        "precursor group count");

    SimpleFluid::FeedbackMap::PlaceholderOuterCouplingOptions options;
    options.outer_iterations =
        comm->getRank() == 0 ? size_t{1} : size_t{2};
    expect_collective_rejection(
        [&]
        {
            (void)CouplingDriver(
                registry, feedback_cells, options);
        },
        "iteration count");

    options = {};
    options.thermal_hydraulic_subcycles =
        comm->getRank() == 0 ? size_t{1} : size_t{2};
    expect_collective_rejection(
        [&]
        {
            (void)CouplingDriver(
                registry, feedback_cells, options);
        },
        "TH subcycle count");

    options = {};
    options.precursor_group_count = rank_dependent_group_count;
    expect_collective_rejection(
        [&]
        {
            (void)CouplingDriver(
                registry, feedback_cells, options);
        },
        "precursor group count");
}

/** @brief A rank-local alternate power mesh is rejected collectively. */
TEST(FeedbackMapMultiRankTest,
     RejectsRankLocalOuterCouplingPowerMeshMismatch)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    auto alternate_mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType gas_fraction(mesh, 0.1, "alpha_g");
    FieldType density(mesh, 950.0, "rhoFeedback");
    FieldType power(mesh, 0.0, "qdot_fission");
    FieldType alternate_power(
        alternate_mesh, 0.0, "alternate_qdot_fission");
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);

    std::vector<Pack::local_ordinal_type> local_cells;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_cells.push_back(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    CouplingDriver driver(
        registry,
        {{"whole_domain", std::move(local_cells)}});

    auto& selected_power =
        comm->getRank() == 0 ? power : alternate_power;
    size_t thermal_hydraulic_calls = 0;
    size_t neutronics_calls = 0;
    bool rejected = false;
    try
    {
        (void)driver.run(
            selected_power,
            std::vector<double>(
                selected_power.mesh().num_owned_cells(), 1.0),
            [&](size_t, size_t)
            {
                ++thermal_hydraulic_calls;
            },
            [&](const auto&)
            {
                ++neutronics_calls;
                return std::vector<double>(
                    selected_power.mesh().num_owned_cells(), 1.0);
            });
    }
    catch (const std::invalid_argument& error)
    {
        rejected = true;
        EXPECT_NE(
            std::string(error.what()).find("must share a mesh"),
            std::string::npos);
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(thermal_hydraulic_calls, 0u);
    EXPECT_EQ(neutronics_calls, 0u);
}

/** @brief A rank-local NaN is rejected before collective feedback sums. */
TEST(FeedbackMapMultiRankTest,
     RejectsRankLocalNonFiniteFeedbackValue)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType temperature(mesh, 300.0, "temperature");
    if (comm->getRank() == 1)
    {
        temperature.set_owned_value(
            0, std::numeric_limits<double>::quiet_NaN());
    }
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);

    std::vector<Pack::local_ordinal_type> local_cells;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_cells.push_back(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    const std::vector<FeedbackCell> feedback_cells{
        {"whole_domain", std::move(local_cells)}};

    bool rejected = false;
    try
    {
        (void)registry.export_snapshot(feedback_cells);
    }
    catch (const std::invalid_argument& error)
    {
        rejected = true;
        EXPECT_NE(
            std::string(error.what()).find("finite field values"),
            std::string::npos);
    }
    EXPECT_TRUE(rejected);
}

/** @brief The placeholder exchange advances successfully on exactly two ranks. */
TEST(FeedbackMapMultiRankTest,
     RunsSuccessfulOuterCouplingExchangeOnTwoRanks)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    FieldType power(mesh, 0.0, "qdot_fission");
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType gas_fraction(mesh, 0.1, "alpha_g");
    FieldType density(mesh, 950.0, "rhoFeedback");
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);

    std::vector<Pack::local_ordinal_type> local_cells;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_cells.push_back(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    SimpleFluid::FeedbackMap::PlaceholderOuterCouplingOptions options;
    options.outer_iterations = 2;
    CouplingDriver driver(
        registry,
        {{"whole_domain", std::move(local_cells)}},
        options);

    size_t thermal_hydraulic_calls = 0;
    size_t neutronics_calls = 0;
    const auto records = driver.run(
        power,
        std::vector<double>(mesh->num_owned_cells(), 1.0),
        [&](size_t iteration, size_t subcycle)
        {
            EXPECT_EQ(subcycle, 0u);
            EXPECT_EQ(iteration, thermal_hydraulic_calls);
            for (size_t owned = 0;
                 owned < mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<Pack::local_ordinal_type>(owned);
                temperature.set_owned_value(
                    cell_lid,
                    temperature.value(cell_lid)
                    + power.value(cell_lid));
            }
            temperature.sync_ghosts();
            ++thermal_hydraulic_calls;
        },
        [&](const auto& feedback)
        {
            EXPECT_EQ(feedback.sequence_index(), neutronics_calls);
            ++neutronics_calls;
            return std::vector<double>(
                mesh->num_owned_cells(),
                static_cast<double>(neutronics_calls + 1));
        });

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(thermal_hydraulic_calls, 2u);
    EXPECT_EQ(neutronics_calls, 2u);
    EXPECT_DOUBLE_EQ(records[0].feedback.field("T_liquid")[0], 301.0);
    EXPECT_DOUBLE_EQ(records[1].feedback.field("T_liquid")[0], 303.0);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(power.value(cell_lid), 3.0);
    }
}

/** @brief Rank-local callback exceptions become synchronized driver failures. */
TEST(FeedbackMapMultiRankTest,
     SynchronizesRankLocalOuterCouplingCallbackFailures)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    FieldType power(mesh, 0.0, "qdot_fission");
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType gas_fraction(mesh, 0.1, "alpha_g");
    FieldType density(mesh, 950.0, "rhoFeedback");
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);

    std::vector<Pack::local_ordinal_type> local_cells;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_cells.push_back(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    CouplingDriver driver(
        registry,
        {{"whole_domain", std::move(local_cells)}});
    const std::vector<double> initial_power(
        mesh->num_owned_cells(), 1.0);

    size_t thermal_hydraulic_calls = 0;
    size_t neutronics_calls = 0;
    bool thermal_hydraulic_rejected = false;
    try
    {
        (void)driver.run(
            power,
            initial_power,
            [&](size_t, size_t)
            {
                ++thermal_hydraulic_calls;
                if (comm->getRank() == 1)
                {
                    throw std::runtime_error("rank-local TH failure");
                }
            },
            [&](const auto&)
            {
                ++neutronics_calls;
                return initial_power;
            });
    }
    catch (const std::runtime_error& error)
    {
        thermal_hydraulic_rejected = true;
        EXPECT_NE(
            std::string(error.what()).find(
                "thermal-hydraulic callback"),
            std::string::npos);
    }
    EXPECT_TRUE(thermal_hydraulic_rejected);
    EXPECT_EQ(thermal_hydraulic_calls, 1u);
    EXPECT_EQ(neutronics_calls, 0u);

    thermal_hydraulic_calls = 0;
    neutronics_calls = 0;
    bool neutronics_rejected = false;
    try
    {
        (void)driver.run(
            power,
            initial_power,
            [&](size_t, size_t)
            {
                ++thermal_hydraulic_calls;
            },
            [&](const auto&) -> std::vector<double>
            {
                ++neutronics_calls;
                if (comm->getRank() == 1)
                {
                    throw std::runtime_error(
                        "rank-local neutronics failure");
                }
                return initial_power;
            });
    }
    catch (const std::runtime_error& error)
    {
        neutronics_rejected = true;
        EXPECT_NE(
            std::string(error.what()).find("neutronics callback"),
            std::string::npos);
    }
    EXPECT_TRUE(neutronics_rejected);
    EXPECT_EQ(thermal_hydraulic_calls, 1u);
    EXPECT_EQ(neutronics_calls, 1u);
}
