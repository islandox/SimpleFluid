/**
 * @file testYPlusBoundaryLayerController.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for rank-consistent y-plus boundary-layer adaptation.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/YPlusBoundaryLayerController.hh"
#include "utils/testing_environment.hh"

#include <Tpetra_Core.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Spec =
    SimpleFluid::BoundaryLayerMeshFactory::BoundaryLayerSpec;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::WallYPlusStatistics make_statistics(
    std::string boundary_name,
    double maximum,
    double area_weighted_mean)
{
    return {
        std::move(boundary_name),
        4,
        2.0,
        std::min(maximum, area_weighted_mean),
        maximum,
        area_weighted_mean};
}

SimpleFluid::MeshQualityMetrics valid_quality()
{
    return {
        8,
        36,
        1.0,
        1.0,
        0.5,
        1.2,
        5.0,
        0.1,
        20.0};
}

SimpleFluid::Arr<Spec> initial_specs(double height = 0.4)
{
    return {{"xmin", 8, height, 1.2}};
}

} // namespace

TEST(WallYPlusStatisticsTest, ReducesAreaWeightedSamples)
{
    const auto communicator = Tpetra::getDefaultComm();
    ASSERT_EQ(communicator->getSize(), 1);
    const SimpleFluid::Arr<SimpleFluid::WallYPlusSample> samples{
        {1.0, 1.0},
        {3.0, 3.0}};

    const auto statistics =
        SimpleFluid::reduce_wall_y_plus_statistics(
            *communicator, "heated_wall", samples);

    EXPECT_EQ(statistics.boundary_name, "heated_wall");
    EXPECT_EQ(statistics.global_face_count, 2U);
    EXPECT_DOUBLE_EQ(statistics.total_face_area, 4.0);
    EXPECT_DOUBLE_EQ(statistics.minimum, 1.0);
    EXPECT_DOUBLE_EQ(statistics.maximum, 3.0);
    EXPECT_DOUBLE_EQ(statistics.area_weighted_mean, 2.5);
}

TEST(WallYPlusStatisticsTest, AcceptsZeroAndRepresentsEmptyPatch)
{
    const auto communicator = Tpetra::getDefaultComm();
    ASSERT_EQ(communicator->getSize(), 1);
    const SimpleFluid::Arr<SimpleFluid::WallYPlusSample> zero_sample{
        {0.0, 2.0}};

    const auto zero_statistics =
        SimpleFluid::reduce_wall_y_plus_statistics(
            *communicator, "stationary_wall", zero_sample);
    EXPECT_FALSE(zero_statistics.empty());
    EXPECT_DOUBLE_EQ(zero_statistics.minimum, 0.0);
    EXPECT_DOUBLE_EQ(zero_statistics.maximum, 0.0);
    EXPECT_DOUBLE_EQ(zero_statistics.area_weighted_mean, 0.0);

    const auto empty_statistics =
        SimpleFluid::reduce_wall_y_plus_statistics(
            *communicator, "empty_wall",
            std::span<const SimpleFluid::WallYPlusSample>{});
    EXPECT_TRUE(empty_statistics.empty());
    EXPECT_EQ(empty_statistics.global_face_count, 0U);
    EXPECT_DOUBLE_EQ(empty_statistics.total_face_area, 0.0);
    EXPECT_TRUE(std::isnan(empty_statistics.minimum));
    EXPECT_TRUE(std::isnan(empty_statistics.maximum));
    EXPECT_TRUE(std::isnan(empty_statistics.area_weighted_mean));
}

TEST(
    WallYPlusStatisticsTest,
    ReducesAcrossRanksWhenSomeRanksOwnNoWallFaces)
{
    const auto communicator = Tpetra::getDefaultComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    SimpleFluid::Arr<SimpleFluid::WallYPlusSample> local_samples;
    if (communicator->getRank() == 0)
    {
        local_samples = {{0.5, 1.0}, {2.5, 3.0}};
    }
    const auto statistics =
        SimpleFluid::reduce_wall_y_plus_statistics(
            *communicator, "rank_zero_wall", local_samples);

    EXPECT_EQ(statistics.global_face_count, 2U);
    EXPECT_DOUBLE_EQ(statistics.total_face_area, 4.0);
    EXPECT_DOUBLE_EQ(statistics.minimum, 0.5);
    EXPECT_DOUBLE_EQ(statistics.maximum, 2.5);
    EXPECT_DOUBLE_EQ(statistics.area_weighted_mean, 2.0);

    const auto globally_empty =
        SimpleFluid::reduce_wall_y_plus_statistics(
            *communicator, "globally_empty_wall",
            std::span<const SimpleFluid::WallYPlusSample>{});
    EXPECT_TRUE(globally_empty.empty());
    EXPECT_TRUE(std::isnan(globally_empty.minimum));
    EXPECT_TRUE(std::isnan(globally_empty.maximum));
    EXPECT_TRUE(std::isnan(globally_empty.area_weighted_mean));

    SimpleFluid::YPlusBoundaryLayerControllerOptions options;
    options.target_y_plus = 1.0;
    options.adaptation_exponent = 1.0;
    options.minimum_height_ratio = 0.1;
    const SimpleFluid::YPlusBoundaryLayerController controller(options);
    const auto update = controller.update_height(0.01, statistics);
    EXPECT_DOUBLE_EQ(update.measured_y_plus, 2.5);
    EXPECT_DOUBLE_EQ(update.next_height, 0.004);
}

TEST(WallYPlusStatisticsTest, RejectsInvalidSamplesCollectively)
{
    const auto communicator = Tpetra::getDefaultComm();
    const SimpleFluid::Arr<SimpleFluid::WallYPlusSample> samples{
        {-1.0, 1.0}};
    EXPECT_THROW(
        SimpleFluid::reduce_wall_y_plus_statistics(
            *communicator, "wall", samples),
        std::invalid_argument);
}

TEST(
    YPlusBoundaryLayerControllerTest,
    RejectsZeroMeasurementThatCannotDefineAHeightRatio)
{
    const SimpleFluid::YPlusBoundaryLayerController controller;
    EXPECT_THROW(
        controller.update_height(
            0.01, make_statistics("stationary_wall", 0.0, 0.0)),
        std::invalid_argument);
}

TEST(YPlusBoundaryLayerControllerTest, UsesMaximumForResolvedWallControl)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions options;
    options.target_y_plus = 1.0;
    options.adaptation_exponent = 0.5;
    options.minimum_height_ratio = 0.1;
    const SimpleFluid::YPlusBoundaryLayerController controller(options);
    const auto statistics =
        make_statistics("wall", 4.0, 1.0);

    const auto update =
        controller.update_height(0.01, statistics);

    EXPECT_DOUBLE_EQ(update.measured_y_plus, 4.0);
    EXPECT_DOUBLE_EQ(update.undamped_height_ratio, 0.5);
    EXPECT_DOUBLE_EQ(update.applied_height_ratio, 0.5);
    EXPECT_DOUBLE_EQ(update.next_height, 0.005);
    EXPECT_FALSE(update.converged);
}

TEST(
    YPlusBoundaryLayerControllerTest,
    SupportsAreaWeightedWallFunctionControl)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions options;
    options.target_y_plus = 1.0;
    options.control_statistic =
        SimpleFluid::WallYPlusControlStatistic::AreaWeightedMean;
    const SimpleFluid::YPlusBoundaryLayerController controller(options);

    const auto update = controller.update_height(
        0.01, make_statistics("wall", 4.0, 1.0));

    EXPECT_TRUE(update.converged);
    EXPECT_DOUBLE_EQ(update.next_height, 0.01);
}

TEST(
    YPlusBoundaryLayerControllerTest,
    AppliesLogDampingRatioAndAbsoluteClamps)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions options;
    options.target_y_plus = 1.0;
    options.adaptation_exponent = 0.5;
    options.damping = 0.5;
    options.minimum_height_ratio = 0.75;
    options.minimum_first_cell_height = 0.08;
    const SimpleFluid::YPlusBoundaryLayerController controller(options);

    const auto update = controller.update_height(
        0.1, make_statistics("wall", 16.0, 16.0));

    EXPECT_DOUBLE_EQ(update.undamped_height_ratio, 0.25);
    EXPECT_DOUBLE_EQ(update.damped_height_ratio, 0.5);
    EXPECT_DOUBLE_EQ(update.applied_height_ratio, 0.75);
    EXPECT_DOUBLE_EQ(update.next_height, 0.08);
    EXPECT_TRUE(update.ratio_was_limited);
    EXPECT_TRUE(update.absolute_height_was_limited);
}

TEST(YPlusBoundaryLayerControllerTest, MatchesLayerSpecsByBoundaryName)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions options;
    options.target_y_plus = 1.0;
    options.adaptation_exponent = 1.0;
    options.minimum_height_ratio = 0.1;
    options.maximum_height_ratio = 10.0;
    const SimpleFluid::YPlusBoundaryLayerController controller(options);
    const SimpleFluid::Arr<Spec> specs{
        {"xmin", 8, 0.4, 1.2},
        {"xmax", 8, 0.2, 1.2}};
    const SimpleFluid::Arr<SimpleFluid::WallYPlusStatistics> statistics{
        make_statistics("xmax", 1.0, 1.0),
        make_statistics("xmin", 4.0, 4.0)};

    const auto update =
        controller.update_layer_specs(specs, statistics);

    EXPECT_FALSE(update.converged);
    EXPECT_DOUBLE_EQ(update.layer_specs[0].first_cell_height, 0.1);
    EXPECT_DOUBLE_EQ(update.layer_specs[1].first_cell_height, 0.2);
    EXPECT_FALSE(update.patch_updates[0].converged);
    EXPECT_TRUE(update.patch_updates[1].converged);
}

TEST(MeshQualityTest, ReportsGrowthAspectAndOrthogonality)
{
    const auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 3.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const Handle mesh(geometry);

    const auto quality =
        SimpleFluid::evaluate_mesh_quality(mesh);

    EXPECT_EQ(quality.global_cell_count, 2U);
    EXPECT_EQ(quality.global_face_count, 11U);
    EXPECT_DOUBLE_EQ(quality.minimum_cell_volume, 1.0);
    EXPECT_DOUBLE_EQ(quality.minimum_face_area, 1.0);
    EXPECT_DOUBLE_EQ(quality.minimum_normal_distance, 0.5);
    EXPECT_DOUBLE_EQ(quality.maximum_growth_ratio, 2.0);
    EXPECT_NEAR(
        quality.maximum_non_orthogonality_degrees, 0.0, 1.0e-12);
    EXPECT_NEAR(quality.maximum_skewness, 0.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(quality.maximum_aspect_ratio, 2.0);
}

TEST(MeshQualityTest, GateRejectsMandatoryAndConfiguredFailures)
{
    auto quality = valid_quality();
    quality.minimum_face_area = 0.0;
    quality.maximum_growth_ratio = 2.0;
    SimpleFluid::MeshQualityLimits limits;
    limits.maximum_growth_ratio = 1.5;
    const SimpleFluid::MeshQualityGate gate(limits);

    const auto assessment = gate.assess(quality);

    EXPECT_FALSE(assessment.accepted());
    ASSERT_EQ(assessment.violations.size(), 2U);
    EXPECT_NE(
        assessment.report(quality).find("face areas"),
        std::string::npos);
    EXPECT_NE(
        assessment.report(quality).find("growth ratio"),
        std::string::npos);
    EXPECT_THROW(gate.require(quality), std::runtime_error);
}

TEST(
    YPlusBoundaryLayerAdaptationDriverTest,
    RebuildsPristineArtifactsUntilConverged)
{
    struct Artifact
    {
        double first_cell_height = {};
        size_t cycle = {};
    };

    SimpleFluid::YPlusBoundaryLayerControllerOptions controller_options;
    controller_options.target_y_plus = 1.0;
    controller_options.adaptation_exponent = 1.0;
    controller_options.minimum_height_ratio = 0.1;
    controller_options.maximum_height_ratio = 10.0;
    controller_options.relative_tolerance = 1.0e-12;
    SimpleFluid::YPlusBoundaryLayerAdaptationOptions driver_options;
    driver_options.maximum_cycles = 4;
    const SimpleFluid::YPlusBoundaryLayerAdaptationDriver driver(
        SimpleFluid::YPlusBoundaryLayerController(controller_options),
        driver_options);

    std::weak_ptr<Artifact> previous_artifact;
    std::vector<std::string> events;
    const auto report = driver.run(
        initial_specs(),
        [&](const SimpleFluid::Arr<Spec>& specs, size_t cycle)
        {
            EXPECT_TRUE(previous_artifact.expired());
            events.push_back("build" + std::to_string(cycle));
            auto artifact = std::make_shared<Artifact>(
                Artifact{specs.front().first_cell_height, cycle});
            previous_artifact = artifact;
            return artifact;
        },
        [&](const auto& artifact, size_t cycle)
        {
            EXPECT_EQ(artifact->cycle, cycle);
            events.push_back("quality" + std::to_string(cycle));
            return valid_quality();
        },
        [&](const auto& artifact, size_t cycle)
        {
            EXPECT_EQ(artifact->cycle, cycle);
            events.push_back("pilot" + std::to_string(cycle));
            const auto y_plus =
                artifact->first_cell_height * 10.0;
            return SimpleFluid::Arr<
                SimpleFluid::WallYPlusStatistics>{
                make_statistics("xmin", y_plus, y_plus)};
        });

    EXPECT_TRUE(report.converged());
    ASSERT_EQ(report.cycles.size(), 2U);
    ASSERT_EQ(report.evaluated_layer_specs.size(), 1U);
    EXPECT_DOUBLE_EQ(
        report.evaluated_layer_specs.front().first_cell_height, 0.1);
    EXPECT_DOUBLE_EQ(
        report.recommended_layer_specs.front().first_cell_height, 0.1);
    EXPECT_EQ(
        events,
        (std::vector<std::string>{
            "build0", "quality0", "pilot0",
            "build1", "quality1", "pilot1"}));
    EXPECT_TRUE(previous_artifact.expired());
}

TEST(
    YPlusBoundaryLayerAdaptationDriverTest,
    ReportsMaximumCyclesWithoutClaimingUntestedMesh)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions controller_options;
    controller_options.target_y_plus = 1.0;
    controller_options.adaptation_exponent = 1.0;
    controller_options.minimum_height_ratio = 0.5;
    SimpleFluid::YPlusBoundaryLayerAdaptationOptions driver_options;
    driver_options.maximum_cycles = 2;
    const SimpleFluid::YPlusBoundaryLayerAdaptationDriver driver(
        SimpleFluid::YPlusBoundaryLayerController(controller_options),
        driver_options);

    const auto report = driver.run(
        initial_specs(1.0),
        [](const SimpleFluid::Arr<Spec>& specs, size_t)
        {
            return specs.front().first_cell_height;
        },
        [](double&, size_t) { return valid_quality(); },
        [](double&, size_t)
        {
            return SimpleFluid::Arr<
                SimpleFluid::WallYPlusStatistics>{
                make_statistics("xmin", 10.0, 10.0)};
        });

    EXPECT_EQ(
        report.status,
        SimpleFluid::YPlusBoundaryLayerAdaptationStatus::
            MaximumCyclesReached);
    ASSERT_EQ(report.cycles.size(), 2U);
    EXPECT_DOUBLE_EQ(
        report.evaluated_layer_specs.front().first_cell_height, 0.5);
    EXPECT_DOUBLE_EQ(
        report.recommended_layer_specs.front().first_cell_height, 0.25);
}

TEST(
    YPlusBoundaryLayerAdaptationDriverTest,
    RejectsBadMeshBeforePilotSolver)
{
    SimpleFluid::YPlusBoundaryLayerAdaptationOptions driver_options;
    driver_options.mesh_quality_limits.maximum_growth_ratio = 1.5;
    const SimpleFluid::YPlusBoundaryLayerAdaptationDriver driver(
        SimpleFluid::YPlusBoundaryLayerController{},
        driver_options);
    size_t pilot_calls = 0;

    const auto report = driver.run(
        initial_specs(),
        [](const SimpleFluid::Arr<Spec>&, size_t) { return 0; },
        [](int&, size_t)
        {
            auto quality = valid_quality();
            quality.maximum_growth_ratio = 2.0;
            return quality;
        },
        [&](int&, size_t)
        {
            ++pilot_calls;
            return SimpleFluid::Arr<
                SimpleFluid::WallYPlusStatistics>{};
        });

    EXPECT_EQ(
        report.status,
        SimpleFluid::YPlusBoundaryLayerAdaptationStatus::
            MeshQualityRejected);
    EXPECT_EQ(pilot_calls, 0U);
    ASSERT_EQ(report.cycles.size(), 1U);
    EXPECT_FALSE(
        report.cycles.front().mesh_quality_assessment.accepted());
    EXPECT_FALSE(report.cycles.front().update.has_value());
    EXPECT_TRUE(report.evaluated_layer_specs.empty());
    EXPECT_TRUE(report.recommended_layer_specs.empty());
}

TEST(
    YPlusBoundaryLayerAdaptationDriverTest,
    PreservesLastAcceptedSpecsWhenLaterMeshFailsQuality)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions controller_options;
    controller_options.adaptation_exponent = 1.0;
    controller_options.minimum_height_ratio = 0.5;
    SimpleFluid::YPlusBoundaryLayerAdaptationOptions driver_options;
    driver_options.maximum_cycles = 3;
    driver_options.mesh_quality_limits.maximum_growth_ratio = 1.5;
    const SimpleFluid::YPlusBoundaryLayerAdaptationDriver driver(
        SimpleFluid::YPlusBoundaryLayerController(controller_options),
        driver_options);
    size_t pilot_calls = 0;

    const auto report = driver.run(
        initial_specs(1.0),
        [](const SimpleFluid::Arr<Spec>& specs, size_t)
        {
            return specs.front().first_cell_height;
        },
        [](double&, size_t cycle)
        {
            auto quality = valid_quality();
            if (cycle == 1)
            {
                quality.maximum_growth_ratio = 2.0;
            }
            return quality;
        },
        [&](double&, size_t)
        {
            ++pilot_calls;
            return SimpleFluid::Arr<
                SimpleFluid::WallYPlusStatistics>{
                make_statistics("xmin", 10.0, 10.0)};
        });

    EXPECT_EQ(
        report.status,
        SimpleFluid::YPlusBoundaryLayerAdaptationStatus::
            MeshQualityRejected);
    EXPECT_EQ(pilot_calls, 1U);
    ASSERT_EQ(report.cycles.size(), 2U);
    EXPECT_DOUBLE_EQ(
        report.cycles.back()
            .evaluated_layer_specs.front()
            .first_cell_height,
        0.5);
    EXPECT_FALSE(
        report.cycles.back().mesh_quality_assessment.accepted());
    ASSERT_EQ(report.evaluated_layer_specs.size(), 1U);
    ASSERT_EQ(report.recommended_layer_specs.size(), 1U);
    EXPECT_DOUBLE_EQ(
        report.evaluated_layer_specs.front().first_cell_height,
        1.0);
    EXPECT_DOUBLE_EQ(
        report.recommended_layer_specs.front().first_cell_height,
        1.0);
}

TEST(
    YPlusBoundaryLayerAdaptationDriverTest,
    DefaultQualityEvaluatorAcceptsMeshHandleArtifact)
{
    SimpleFluid::YPlusBoundaryLayerControllerOptions controller_options;
    controller_options.target_y_plus = 1.0;
    const SimpleFluid::YPlusBoundaryLayerAdaptationDriver driver{
        SimpleFluid::YPlusBoundaryLayerController(controller_options)};

    const auto report = driver.run(
        initial_specs(0.1),
        [](const SimpleFluid::Arr<Spec>&, size_t)
        {
            auto geometry = std::make_shared<Cartesian>(
                SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                    {0.0, 1.0},
                    {0.0, 1.0},
                    {0.0, 1.0}}});
            return std::make_shared<Handle>(geometry);
        },
        [](const auto&, size_t)
        {
            return SimpleFluid::Arr<
                SimpleFluid::WallYPlusStatistics>{
                make_statistics("xmin", 1.0, 1.0)};
        });

    EXPECT_TRUE(report.converged());
    ASSERT_EQ(report.cycles.size(), 1U);
    EXPECT_TRUE(
        report.cycles.front().mesh_quality_assessment.accepted());
}
