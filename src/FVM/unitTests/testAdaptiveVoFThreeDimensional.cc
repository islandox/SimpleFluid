/**
 * @file testAdaptiveVoFThreeDimensional.cc
 * @brief Oblique sphere transport through native coarse/fine and MPI interfaces.
 */
#include "FVM/AdaptiveVoF.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using Hierarchy = SimpleFluid::Meshes::AMR::CartesianAMR;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

/** Independent quadrature for the physical sphere, never a reconstructed PLIC. */
std::vector<std::pair<double, double>> gauss_points()
{
    constexpr unsigned order = 256;
    std::vector<std::pair<double, double>> points;
    for (unsigned i = 0; i < order / 2; ++i)
    {
        double root = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (order + 0.5));
        double derivative = 0.0;
        for (unsigned iteration = 0; iteration < 20; ++iteration)
        {
            double previous = 1.0, value = root;
            for (unsigned k = 2; k <= order; ++k)
            {
                const double next = ((2.0 * k - 1.0) * root * value - (k - 1.0) * previous) / k;
                previous = value;
                value = next;
            }
            derivative = order * (root * value - previous) / (root * root - 1.0);
            const double correction = value / derivative;
            root -= correction;
            if (std::abs(correction) < 1.0e-15)
                break;
        }
        const double weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        points.emplace_back(-root, weight);
        points.emplace_back(root, weight);
    }
    return points;
}

std::vector<double> sphere_fractions(const AdaptiveVoF& geometry, Vector center, double radius)
{
    const auto points = gauss_points();
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto box = geometry.hierarchy()->cell_box(geometry.mesh()->cell_global_id(static_cast<LO>(cell)));
        Vector lower{}, upper{}, width{};
        double near2 = 0.0, far2 = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
        {
            lower[a] = box.lower[a] - center[a];
            upper[a] = box.upper[a] - center[a];
            width[a] = upper[a] - lower[a];
            const double near =
                lower[a] <= 0.0 && upper[a] >= 0.0 ? 0.0 : std::min(std::abs(lower[a]), std::abs(upper[a]));
            const double far = std::max(std::abs(lower[a]), std::abs(upper[a]));
            near2 += near * near;
            far2 += far * far;
        }
        if (near2 >= radius * radius)
            continue;
        if (far2 <= radius * radius)
        {
            fraction[cell] = 1.0;
            continue;
        }
        for (const auto& [y, wy] : points)
            for (const auto& [z, wz] : points)
            {
                const double py = 0.5 * (lower[1] + upper[1] + width[1] * y);
                const double pz = 0.5 * (lower[2] + upper[2] + width[2] * z);
                const double chord2 = radius * radius - py * py - pz * pz;
                if (chord2 <= 0.0)
                    continue;
                const double chord = std::sqrt(chord2);
                const double length = std::max(0.0, std::min(upper[0], chord) - std::max(lower[0], -chord));
                fraction[cell] += 0.25 * wy * wz * length / width[0];
            }
    }
    return fraction;
}

double sum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

double maximum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &result);
    return result;
}

double gas_volume(const AdaptiveVoF& geometry, const std::vector<double>& fraction)
{
    double total = 0.0;
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        total += fraction[cell] * geometry.mesh()->cell_volume(static_cast<LO>(cell));
    return sum(total);
}

TEST(AdaptiveVoFThreeDimensionalTest, ObliqueSphereCrossesCoarseFineAndPartitionFacesConservatively)
{
    SimpleFluid::Meshes::AMR::AMRGrid config;
    config.root_blocks = {4, 2, 2};
    config.block_cells = {4, 4, 4};
    config.extent = {1.0, 0.5, 0.5};
    config.periodic = {true, true, true};
    config.max_level = 1;
    const auto initial_tree = std::make_shared<const Hierarchy>(config);
    std::vector<std::size_t> refine;
    for (std::size_t block = 0; block < initial_tree->blocks().size(); ++block)
        if (initial_tree->blocks()[block].index[0] == 1)
            refine.push_back(block);
    const auto tree = initial_tree->adapt(refine);
    AdaptiveVoF geometry(tree);
    EXPECT_EQ(tree->blocks().size(), 44U);
    EXPECT_EQ(geometry.mesh()->owned_cell_map()->getGlobalNumElements(), 2816U);

    constexpr double radius = 0.09;
    constexpr Vector initial_center{0.19, 0.25, 0.25};
    constexpr Vector velocity{0.1, 0.03, 0.02};
    constexpr double dt = 0.075;
    constexpr unsigned steps = 12;
    auto fraction = sphere_fractions(geometry, initial_center, radius);
    const double initial_volume = gas_volume(geometry, fraction);
    const double analytical_volume = 4.0 * std::numbers::pi * radius * radius * radius / 3.0;
    EXPECT_NEAR(initial_volume, analytical_volume, 5.0e-5 * analytical_volume);
    std::vector<double> face_velocity(geometry.mesh()->num_faces(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t face = 0; face < geometry.mesh()->num_owned_faces(); ++face)
    {
        const auto normal = geometry.mesh()->face_normal(static_cast<LO>(face));
        face_velocity[face] = velocity[0] * normal.x + velocity[1] * normal.y + velocity[2] * normal.z;
    }
    const auto canonical_velocity = geometry.exchange_faces(face_velocity);
    double maximum_ledger_error = 0.0;
    double maximum_volume_error = 0.0;
    for (unsigned step = 0; step < steps; ++step)
    {
        const auto next = geometry.advect(fraction, face_velocity, dt, step % 3);
        double local_bound_error = 0.0;
        double local_ledger_error = 0.0;
        for (std::size_t cell = 0; cell < fraction.size(); ++cell)
        {
            const auto cid = static_cast<LO>(cell);
            double expected = fraction[cell];
            for (const auto face : geometry.mesh()->faces(cid))
            {
                const double sign = geometry.mesh()->owner_cell(face) == cid ? 1.0 : -1.0;
                expected -= sign * next.gas_volume[face] / geometry.mesh()->cell_volume(cid);
            }
            local_ledger_error = std::max(local_ledger_error, std::abs(next.fraction[cell] - expected));
            local_bound_error = std::max({local_bound_error, -next.fraction[cell], next.fraction[cell] - 1.0});
        }
        double local_flux_error = 0.0;
        for (std::size_t face = 0; face < geometry.mesh()->num_owned_faces(); ++face)
        {
            const double total = canonical_velocity[face] * dt * geometry.mesh()->face_area(static_cast<LO>(face));
            local_flux_error = std::max(
                local_flux_error, std::abs(next.gas_volume[face] + next.liquid_volume[face] - total) / std::abs(total));
        }
        EXPECT_LE(maximum(local_bound_error), 2.0e-12);
        EXPECT_LE(maximum(local_flux_error), 8.0 * std::numeric_limits<double>::epsilon());
        maximum_ledger_error = std::max(maximum_ledger_error, maximum(local_ledger_error));
        EXPECT_LE(maximum_ledger_error, 2.0e-12);
        const double volume_error = std::abs(gas_volume(geometry, next.fraction) - initial_volume) / initial_volume;
        maximum_volume_error = std::max(maximum_volume_error, volume_error);
        EXPECT_LE(maximum_volume_error, 2.0e-12);
        fraction = next.fraction;
    }

    Vector final_center{};
    for (std::size_t a = 0; a < 3; ++a)
        final_center[a] = initial_center[a] + steps * dt * velocity[a];
    const auto exact = sphere_fractions(geometry, final_center, radius);
    EXPECT_NEAR(gas_volume(geometry, exact), analytical_volume, 5.0e-5 * analytical_volume);
    double l1 = 0.0;
    Vector actual_moment{}, reference_moment{};
    for (std::size_t cell = 0; cell < fraction.size(); ++cell)
    {
        const auto cid = static_cast<LO>(cell);
        const double volume = geometry.mesh()->cell_volume(cid);
        const auto center = geometry.mesh()->cell_centroid(cid);
        l1 += volume * std::abs(fraction[cell] - exact[cell]);
        for (std::size_t a = 0; a < 3; ++a)
        {
            actual_moment[a] += volume * fraction[cell] * center.component(a);
            reference_moment[a] += volume * exact[cell] * center.component(a);
        }
    }
    const double relative_l1 = sum(l1) / analytical_volume;
    const double coarse_width = config.extent[0] / (config.root_blocks[0] * config.block_cells[0]);
    // This deliberately underresolved traversal has 2.88/5.76 cells across the
    // diameter. Bound its mean interface displacement by one eighth coarse cell;
    // resolving a centimetric bubble physically requires the separate 24--32 target.
    const double shape_gate = 3.0 * coarse_width / (8.0 * radius);
    EXPECT_LT(relative_l1, shape_gate);
    const double reference_volume = gas_volume(geometry, exact);
    double centroid_error = 0.0;
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double error = sum(actual_moment[a]) / initial_volume - sum(reference_moment[a]) / reference_volume;
        centroid_error += error * error;
    }
    centroid_error = std::sqrt(centroid_error);
    EXPECT_LT(centroid_error, coarse_width / 8.0);
    EXPECT_EQ(geometry.mesh()->connectivity_storage_bytes(), 0U);
    RecordProperty("relative_shape_l1", relative_l1);
    RecordProperty("mean_interface_error_m", relative_l1 * radius / 3.0);
    RecordProperty("centroid_error_m", centroid_error);
    RecordProperty("maximum_relative_gas_volume_error", maximum_volume_error);
    RecordProperty("maximum_cell_phase_ledger_error", maximum_ledger_error);
}
} // namespace
