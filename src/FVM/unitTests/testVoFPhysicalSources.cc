/**
 * @file testVoFPhysicalSources.cc
 * @brief Physical phase-volume, interface-film, and distributed patch contracts.
 */
#include "FVM/AdaptiveVoF.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using Hierarchy = Meshes::CartesianAMR;
using Grid = Meshes::AMRGrid;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

double global_sum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

Grid unit_cell_grid()
{
    Grid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2.0, 2.0, 2.0};
    grid.max_level = 1;
    return grid;
}

std::shared_ptr<const Hierarchy> periodic_refined_grid()
{
    auto grid = unit_cell_grid();
    grid.root_blocks = {4, 1, 1};
    grid.extent = {1.0, 1.0, 1.0};
    grid.periodic = {true, true, true};
    const auto root = std::make_shared<const Hierarchy>(grid);
    return root->adapt({1});
}

std::vector<double> face_velocity(const AdaptiveVoF& geometry, bool expanding)
{
    std::vector<double> velocity(geometry.mesh()->num_faces(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t face = 0; face < geometry.mesh()->num_owned_faces(); ++face)
    {
        const auto id = static_cast<LO>(face);
        const auto point = geometry.mesh()->face_centroid(id);
        const auto normal = geometry.mesh()->face_normal(id);
        velocity[face] = geometry.mesh()->is_geometry_exterior_face(id)
                             ? 0.0
                             : (expanding ? 0.2 * std::sin(2.0 * std::numbers::pi * point.x) : 0.1) * normal.x;
    }
    return geometry.exchange_faces(velocity);
}

PhaseVolumeIncrement zero_sources(const AdaptiveVoF& geometry)
{
    return {std::vector<double>(geometry.mesh()->num_owned_cells(), 0.0),
        std::vector<double>(geometry.mesh()->num_owned_cells(), 0.0)};
}

TEST(VoFPhysicalSourcesTest, PlaneSectionsHavePhysicalAreaCentroidAndOrientation)
{
    const auto axis = GeometricVoF::plane_section({0, 0, 0}, {1, 1, 1}, {2, 0, 0}, 0.5);
    EXPECT_DOUBLE_EQ(axis.area, 1.0);
    EXPECT_NEAR(axis.centroid[0], 0.25, 1.0e-15);
    EXPECT_NEAR(axis.centroid[1], 0.5, 1.0e-15);
    EXPECT_NEAR(axis.centroid[2], 0.5, 1.0e-15);
    EXPECT_EQ(axis.normal, (Vector{1.0, 0.0, 0.0}));
    const auto diagonal = GeometricVoF::plane_section({0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1.5);
    EXPECT_NEAR(diagonal.area, 3.0 * std::sqrt(3.0) / 4.0, 2.0e-15);
    for (std::size_t a = 0; a < 3; ++a)
    {
        EXPECT_NEAR(diagonal.centroid[a], 0.5, 2.0e-15);
        EXPECT_NEAR(diagonal.normal[a], 1.0 / std::sqrt(3.0), 2.0e-15);
    }
    const auto complement = GeometricVoF::plane_section({0, 0, 0}, {1, 1, 1}, {-1, -1, -1}, -1.5);
    EXPECT_NEAR(complement.area, diagonal.area, 2.0e-15);
    EXPECT_DOUBLE_EQ(GeometricVoF::plane_section({0, 0, 0}, {1, 1, 1}, {1, 0, 0}, 2.0).area, 0.0);
}

TEST(VoFPhysicalSourcesTest, RetainedEndpointRoundoffCannotTransportAnAbsentPhase)
{
    AdaptiveVoF geometry(periodic_refined_grid());
    const auto velocity = face_velocity(geometry, false);
    constexpr double dt = 0.01;
    for (double value : {std::nextafter(1.0, 2.0), -std::numeric_limits<double>::epsilon()})
    {
        SCOPED_TRACE(value);
        const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), value);
        const auto result = geometry.advect(fraction, velocity, dt);
        EXPECT_EQ(result.fraction, fraction);
        for (std::size_t face = 0; face < geometry.mesh()->num_owned_faces(); ++face)
        {
            const double total = velocity[face] * dt * geometry.mesh()->face_area(static_cast<LO>(face));
            EXPECT_DOUBLE_EQ(result.gas_volume[face] + result.liquid_volume[face], total);
            EXPECT_DOUBLE_EQ(value > 1 ? result.liquid_volume[face] : result.gas_volume[face], 0.0);
        }
    }

    // The uniform solver has the same endpoint partition and retains f too.
    UniformGrid uniform;
    uniform.cells = {2, 2, 2};
    uniform.spacing = {0.5, 0.5, 0.5};
    uniform.periodic = {true, true, true};
    GeometricVoF vof(uniform);
    FaceField face_speed;
    for (std::size_t a = 0; a < 3; ++a)
        face_speed[a].assign(uniform.face_count(a), a == 0 ? 0.1 : 0.0);
    for (double value : {std::nextafter(1.0, 2.0), -std::numeric_limits<double>::epsilon()})
    {
        const std::vector<double> fraction(uniform.cell_count(), value);
        const auto result = vof.advect(fraction, face_speed, dt);
        EXPECT_EQ(result.fraction, fraction);
        for (std::size_t face = 0; face < face_speed[0].size(); ++face)
        {
            const double total = face_speed[0][face] * dt * uniform.face_area(0);
            EXPECT_DOUBLE_EQ(result.gas_volume[0][face] + result.liquid_volume[0][face], total);
            EXPECT_DOUBLE_EQ(value > 1 ? result.liquid_volume[0][face] : result.gas_volume[0][face], 0.0);
        }
    }
}

TEST(VoFPhysicalSourcesTest, PlanarAndObliqueFilmsIntegrateTheActualCarrierGeometry)
{
    const auto tree = std::make_shared<const Hierarchy>(unit_cell_grid());
    AdaptiveVoF geometry(tree);
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < fraction.size(); ++c)
        fraction[c] = geometry.mesh()->cell_centroid(static_cast<LO>(c)).x < 1.0 ? 0.25 : 0.0;
    const auto planar = geometry.interface_patches(fraction);
    EXPECT_DOUBLE_EQ(global_sum(static_cast<double>(planar.size())), 4.0);
    double area = 0.0;
    for (const auto& patch : planar)
    {
        EXPECT_EQ(patch.kind, InterfacePatch::Kind::Plic);
        EXPECT_EQ(patch.gas_cell, patch.carrier_cell);
        EXPECT_NEAR(patch.centroid[0], 0.25, 2.0e-15);
        EXPECT_NEAR(patch.normal[0], 1.0, 2.0e-15);
        EXPECT_NEAR(geometry.film_length(patch, 100.0), 0.375, 2.0e-14);
        EXPECT_NEAR(geometry.film_length(patch, 0.2), 0.2 - 0.04 / 1.5, 2.0e-14);
        EXPECT_NEAR(geometry.film_length(patch, 1.0e-18), 1.0e-18, 1.0e-30);
        EXPECT_THROW((void) geometry.film_length(patch, 0.0), std::invalid_argument);
        area += patch.area;
    }
    EXPECT_NEAR(global_sum(area), 4.0, 2.0e-14);

    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto point = geometry.mesh()->cell_centroid(static_cast<LO>(c));
        fraction[c] = point.x < 1.0 && point.y < 1.0 ? 0.5 : 0.0;
    }
    const auto oblique = geometry.interface_patches(fraction);
    EXPECT_DOUBLE_EQ(global_sum(static_cast<double>(oblique.size())), 2.0);
    area = 0.0;
    for (const auto& patch : oblique)
    {
        EXPECT_NEAR(patch.area, std::sqrt(2.0), 2.0e-14);
        EXPECT_NEAR(patch.normal[0], 1.0 / std::sqrt(2.0), 2.0e-14);
        EXPECT_NEAR(patch.normal[1], 1.0 / std::sqrt(2.0), 2.0e-14);
        constexpr double delta = 0.2;
        const double expected = delta - std::sqrt(2.0) * delta * delta + 2.0 * delta * delta * delta / 3.0;
        EXPECT_NEAR(geometry.film_length(patch, delta), expected, 2.0e-14);
        EXPECT_NEAR(geometry.film_length(patch, 10.0), 1.0 / (3.0 * std::sqrt(2.0)), 2.0e-14);
        area += patch.area;
    }
    EXPECT_NEAR(global_sum(area), 2.0 * std::sqrt(2.0), 2.0e-14);
}

TEST(VoFPhysicalSourcesTest, AlignedCoarseFineAndPeriodicPatchesScatterAndSampleOnce)
{
    auto config = unit_cell_grid();
    config.root_blocks = {4, 1, 1};
    config.extent = {2.0, 1.0, 1.0};
    config.periodic = {true, false, false};
    const auto tree = std::make_shared<const Hierarchy>(config);
    AdaptiveVoF geometry(tree->adapt({1}));
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    std::vector<double> scalar(fraction.size());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        fraction[c] = geometry.mesh()->cell_centroid(static_cast<LO>(c)).x < 0.5 ? 1.0 : 0.0;
        scalar[c] = geometry.mesh()->cell_global_id(static_cast<LO>(c)) + 0.125;
    }
    const auto patches = geometry.interface_patches(fraction);
    EXPECT_DOUBLE_EQ(global_sum(static_cast<double>(patches.size())), 20.0);
    const auto gas_values = geometry.sample_patch_values(patches, scalar, InterfaceSide::Gas);
    const auto carrier_values = geometry.sample_patch_values(patches, scalar, InterfaceSide::Carrier);
    std::vector<double> amounts(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        const auto& patch = patches[i];
        EXPECT_EQ(patch.kind, InterfacePatch::Kind::MeshFace);
        EXPECT_DOUBLE_EQ(gas_values[i], patch.gas_cell + 0.125);
        EXPECT_DOUBLE_EQ(carrier_values[i], patch.carrier_cell + 0.125);
        if (patch.centroid[0] < 1.0)
        {
            EXPECT_NEAR(patch.centroid[0], 0.5, 1.0e-15);
            EXPECT_DOUBLE_EQ(patch.normal[0], 1.0);
            EXPECT_NEAR(geometry.film_length(patch, 10.0), 0.0625, 2.0e-15);
        }
        else
        {
            EXPECT_NEAR(patch.centroid[0], 2.0, 1.0e-15);
            EXPECT_DOUBLE_EQ(patch.normal[0], -1.0);
            EXPECT_NEAR(geometry.film_length(patch, 10.0), 0.125, 2.0e-15);
        }
        amounts[i] = patch.area;
    }
    const auto gas = geometry.scatter_patch_amounts(patches, amounts, InterfaceSide::Gas);
    for (auto& amount : amounts)
        amount = -amount;
    const auto carrier = geometry.scatter_patch_amounts(patches, amounts, InterfaceSide::Carrier);
    EXPECT_NEAR(global_sum(std::accumulate(gas.begin(), gas.end(), 0.0)), 2.0, 2.0e-14);
    EXPECT_NEAR(global_sum(std::accumulate(carrier.begin(), carrier.end(), 0.0)), -2.0, 2.0e-14);
    auto repeated = patches;
    if (!repeated.empty())
        repeated.push_back(repeated.front());
    EXPECT_THROW((void) geometry.sample_patch_values(repeated, scalar, InterfaceSide::Gas), std::invalid_argument);
    const auto aperture = geometry.carrier_face_aperture(fraction);
    const auto halo = geometry.exchange_cells(fraction);
    for (std::size_t face = 0; face < geometry.mesh()->num_owned_faces(); ++face)
    {
        const auto id = static_cast<LO>(face);
        const bool blocked = geometry.mesh()->is_geometry_exterior_face(id) ||
                             halo[geometry.mesh()->owner_cell(id)] >= 1.0 ||
                             halo[geometry.mesh()->neighbor_cell(id)] >= 1.0;
        EXPECT_DOUBLE_EQ(aperture[face], blocked ? 0.0 : 1.0);
    }
}

TEST(VoFPhysicalSourcesTest, SharedCarrierApertureRejectsDisjointHalfFaceTraces)
{
    auto config = unit_cell_grid();
    config.block_cells = {2, 4, 2};
    config.extent = {2.0, 4.0, 2.0};
    AdaptiveVoF geometry(std::make_shared<const Hierarchy>(config));
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto point = geometry.mesh()->cell_centroid(static_cast<LO>(c));
        const double lower_gas = point.y < 1.0 ? 1.0 : (point.y < 2.0 ? 0.5 : 0.0);
        fraction[c] = point.x < 1.0 ? lower_gas : 1.0 - lower_gas;
    }
    const auto aperture = geometry.carrier_face_aperture(fraction);
    double checked = 0.0;
    for (std::size_t face = 0; face < geometry.mesh()->num_owned_faces(); ++face)
    {
        const auto point = geometry.mesh()->face_centroid(static_cast<LO>(face));
        const auto normal = geometry.mesh()->face_normal(static_cast<LO>(face));
        if (std::abs(normal.x) > 0.5 && point.x == 1.0 && point.y == 1.5)
        {
            EXPECT_NEAR(aperture[face], 0.0, 2.0e-14);
            checked += 1.0;
        }
    }
    EXPECT_DOUBLE_EQ(global_sum(checked), 2.0);
}

TEST(VoFPhysicalSourcesTest, CompatibleNonzeroDivergencePreservesPureGasAndPhaseFluxes)
{
    AdaptiveVoF geometry(periodic_refined_grid());
    const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), 1.0);
    const auto velocity = face_velocity(geometry, true);
    constexpr double dt = 0.01;
    auto source = zero_sources(geometry);
    for (std::size_t c = 0; c < fraction.size(); ++c)
        for (const auto face : geometry.mesh()->faces(static_cast<LO>(c)))
        {
            const double sign = geometry.mesh()->owner_cell(face) == static_cast<LO>(c) ? 1.0 : -1.0;
            source.gas[c] += sign * velocity[face] * dt * geometry.mesh()->face_area(face);
        }
    EXPECT_THROW((void) geometry.advect(fraction, velocity, dt), std::invalid_argument);
    const auto next = geometry.advect(fraction, velocity, dt, source);
    EXPECT_GT(next.maximum_volume_divergence, 1.0e-3);
    EXPECT_LT(next.maximum_volume_source_residual, 2.0e-14);
    EXPECT_EQ(next.fraction, fraction);
    for (std::size_t face = 0; face < velocity.size(); ++face)
    {
        const double total = velocity[face] * dt * geometry.mesh()->face_area(static_cast<LO>(face));
        EXPECT_DOUBLE_EQ(next.gas_volume[face], total);
        EXPECT_DOUBLE_EQ(next.liquid_volume[face], 0.0);
    }
}

TEST(VoFPhysicalSourcesTest, ZeroSourceOverloadRetainsExistingGeometricTransport)
{
    AdaptiveVoF geometry(periodic_refined_grid());
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto box = geometry.hierarchy()->cell_box(geometry.mesh()->cell_global_id(static_cast<LO>(c)));
        fraction[c] =
            std::max(0.0, std::min(box.upper[0], 0.63) - std::max(box.lower[0], 0.19)) / (box.upper[0] - box.lower[0]);
    }
    const auto velocity = face_velocity(geometry, false);
    const auto baseline = geometry.advect(fraction, velocity, 0.02, 2);
    const auto sourced = geometry.advect(fraction, velocity, 0.02, zero_sources(geometry), 2);
    EXPECT_EQ(sourced.fraction, baseline.fraction);
    EXPECT_EQ(sourced.gas_volume, baseline.gas_volume);
    EXPECT_EQ(sourced.liquid_volume, baseline.liquid_volume);
    EXPECT_DOUBLE_EQ(sourced.maximum_volume_divergence, baseline.maximum_volume_divergence);
}

TEST(VoFPhysicalSourcesTest, SourceOnlyConversionClosesBothPhaseLedgersWithoutClipping)
{
    AdaptiveVoF geometry(std::make_shared<const Hierarchy>(unit_cell_grid()));
    const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), 0.4);
    const std::vector<double> velocity(geometry.mesh()->num_faces(), 0.0);
    auto source = zero_sources(geometry);
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const double increment = geometry.mesh()->cell_global_id(static_cast<LO>(c)) % 2 == 0 ? 0.1 : -0.05;
        source.gas[c] = increment * geometry.mesh()->cell_volume(static_cast<LO>(c));
        source.carrier[c] = -source.gas[c];
    }
    const auto next = geometry.advect(fraction, velocity, 0.1, source);
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const double v = geometry.mesh()->cell_volume(static_cast<LO>(c));
        EXPECT_DOUBLE_EQ(next.fraction[c], fraction[c] + source.gas[c] / v);
        EXPECT_NEAR(v * (next.fraction[c] - fraction[c]), source.gas[c], 2.0e-15);
        EXPECT_NEAR(v * ((1.0 - next.fraction[c]) - (1.0 - fraction[c])), source.carrier[c], 2.0e-15);
    }
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.0, source), std::invalid_argument);
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        source.gas[c] = 0.7 * geometry.mesh()->cell_volume(static_cast<LO>(c));
        source.carrier[c] = -source.gas[c];
    }
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1, source), std::invalid_argument);
    EXPECT_TRUE(std::all_of(fraction.begin(), fraction.end(), [](double value) { return value == 0.4; }));
}

TEST(VoFPhysicalSourcesTest, GlobalAndLocalSourceIncompatibilityAreRejected)
{
    AdaptiveVoF geometry(std::make_shared<const Hierarchy>(unit_cell_grid()));
    const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), 0.4);
    const std::vector<double> velocity(geometry.mesh()->num_faces(), 0.0);
    auto source = zero_sources(geometry);
    for (std::size_t c = 0; c < fraction.size(); ++c)
        source.gas[c] = 0.01 * geometry.mesh()->cell_volume(static_cast<LO>(c));
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1, source), std::invalid_argument);
    for (std::size_t c = 0; c < fraction.size(); ++c)
        if (geometry.mesh()->cell_global_id(static_cast<LO>(c)) % 2 != 0)
            source.gas[c] = -source.gas[c];
    EXPECT_NEAR(global_sum(std::accumulate(source.gas.begin(), source.gas.end(), 0.0)), 0.0, 1.0e-15);
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1, source), std::invalid_argument);
}

TEST(VoFPhysicalSourcesMultiRankTest, DivergentSourceShapesAndOverloadsRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires multiple MPI ranks.";
    AdaptiveVoF geometry(std::make_shared<const Hierarchy>(unit_cell_grid()));
    const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), 0.4);
    const std::vector<double> velocity(geometry.mesh()->num_faces(), 0.0);
    auto source = zero_sources(geometry);
    if (comm->getRank() == 0)
        source.gas.push_back(0.0);
    EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1, source), std::invalid_argument);
    source = zero_sources(geometry);
    if (comm->getRank() == 0)
        EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1, source), std::invalid_argument);
    else
        EXPECT_THROW((void) geometry.advect(fraction, velocity, 0.1), std::invalid_argument);
}
} // namespace
