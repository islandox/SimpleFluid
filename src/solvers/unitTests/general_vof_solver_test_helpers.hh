/** Shared test-local native fixtures; no test cases or environment registration. */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryInstrumentation.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/GeneralVoFSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
namespace AMR = SimpleFluid::Meshes::AMR;
std::shared_ptr<const VoF::GeneralVoF> phases(unsigned fixture, bool skew = false, bool single_layer = false)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    const auto owning_ranks = std::min(2, comm->getSize());
    for (std::uint64_t gid = 0; gid < (single_layer ? 9 : 27); ++gid)
        if (int(gid % owning_ranks) == comm->getRank())
            owned.push_back(gid);
    AMR::ExtrudedAMRGrid grid;
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned i = 0; i < 4; ++i)
            grid.xy_nodes.push_back({double(i) + (skew ? .3 * j : 0), double(j), 0});
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 3; ++i)
        {
            const unsigned a = 4 * j + i;
            grid.base_cells.push_back({a, a + 1, a + 5, a + 4});
        }
    for (unsigned i = 0; i < 3; ++i)
    {
        grid.boundaries.push_back({i, i + 1, "wall"});
        grid.boundaries.push_back({12 + i, 13 + i, "wall"});
        grid.boundaries.push_back({4 * i, 4 * (i + 1), "wall"});
        grid.boundaries.push_back({4 * i + 3, 4 * (i + 1) + 3, "wall"});
    }
    grid.z_edges = {0, 1, 2, 3};
    if (single_layer)
    {
        // Four boundary cells, one layer; two MPI ranks deliberately own none.
        grid.xy_nodes = {
            {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}};
        grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
        grid.boundaries = {{0, 1, "wall"}, {1, 2, "wall"}, {6, 7, "wall"}, {7, 8, "wall"}, {0, 3, "wall"},
            {3, 6, "wall"}, {2, 5, "wall"}, {5, 8, "wall"}};
        grid.z_edges = {0, 1};
        owned.clear();
        for (std::uint64_t gid = 0; gid < 4; ++gid)
            if (int(gid % owning_ranks) == comm->getRank())
                owned.push_back(gid);
    }
    auto source = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap = source->detach_owned(owned, {'g', 'e', 'n', 'e', 'r', 'a', 'l', 'f', 'l', 'o', 'w'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = hierarchy->geometry(leaf);
        const auto volume = geometry->moments().volume.as_double();
        double g = 0, c = volume;
        if (fixture == 1)
        {
            const bool gas_cell = geometry->centroid()[0] < 1;
            g = gas_cell ? volume : 0;
            c = gas_cell ? 0 : volume;
        }
        if (fixture == 2)
        {
            const auto cut = geometry->clip({{1, 0, 0}, {0, 0, 0}, .75});
            g = cut.inside ? cut.inside->moments().volume.as_double() : 0;
            c = cut.outside ? cut.outside->moments().volume.as_double() : 0;
        }
        order.push_back(key);
        gas.push_back(g);
        carrier.push_back(c);
    }
    auto result =
        std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
    int local_present = result->mesh()->num_owned_cells() > 0, present = 0;
    int local_empty = !local_present, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_present, &present);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(present, owning_ranks);
    EXPECT_EQ(empty, comm->getSize() - owning_ranks);
    return result;
}
std::shared_ptr<const VoF::GeneralVoF> cylinder_sector_phases()
{
    AMR::ExtrudedAMRGrid grid;
    const std::array<double, 4> radii = {.8, .95, 1.05, 1.2};
    for (unsigned a = 0; a < 5; ++a)
    {
        const double angle = -.2 + .1 * a;
        for (double radius : radii)
            grid.xy_nodes.push_back({radius * std::cos(angle), radius * std::sin(angle), 0});
    }
    for (unsigned a = 0; a < 4; ++a)
        for (unsigned r = 0; r < 3; ++r)
        {
            const unsigned p = 4 * a + r;
            grid.base_cells.push_back({p, p + 1, p + 5, p + 4});
        }
    for (unsigned a = 0; a < 4; ++a)
    {
        grid.boundaries.push_back({4 * a, 4 * (a + 1), "wall"});
        grid.boundaries.push_back({4 * a + 3, 4 * (a + 1) + 3, "wall"});
    }
    for (unsigned r = 0; r < 3; ++r)
    {
        grid.boundaries.push_back({r, r + 1, "symmetry"});
        grid.boundaries.push_back({16 + r, 17 + r, "symmetry"});
    }
    grid.z_edges = {0, .2 / 3, 2 * .2 / 3, .2};
    grid.boundary_aliases = {{"zmin", "symmetry_zmin"}, {"zmax", "symmetry_zmax"}};
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 36; ++c)
        if (int(c % comm->getSize()) == comm->getRank())
            owned.push_back(c);
    auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap =
        provider->detach_owned(owned, {'c', 'y', 'l', 'i', 'n', 'd', 'e', 'r', 's', 'e', 'c', 't', 'o', 'r'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto shell = hierarchy->geometry(leaf);
        const auto center = shell->centroid();
        const double radial = std::hypot(center[0], center[1]);
        const unsigned layer = radial < .95 ? 0 : radial < 1.05 ? 1 : 2;
        const double height = shell->axis_spans()[2].as_double(), half_angle = .05;
        double gas_area = 0, carrier_area = 0;
        if (layer == 0)
            gas_area = .5 * (radii[1] * radii[1] - radii[0] * radii[0]) * std::sin(.1);
        else if (layer == 2)
            carrier_area = .5 * (radii[3] * radii[3] - radii[2] * radii[2]) * std::sin(.1);
        else
        {
            // Independent polar integrals: inner gas is 1/2∫(1-r_in²)dθ;
            // outer carrier is 1/2∫(r_out²-1)dθ. The chord bounds are
            // r_i*cos(Δθ/2)/cos(θ-θ_mid), integrated analytically by tan.
            const double chord_integral = std::cos(half_angle) * std::cos(half_angle) * std::tan(half_angle);
            gas_area = half_angle - radii[1] * radii[1] * chord_integral;
            carrier_area = radii[2] * radii[2] * chord_integral - half_angle;
        }
        order.push_back(key);
        gas.push_back(layer == 0 ? shell->moments().volume.as_double() : gas_area * height);
        carrier.push_back(layer == 2 ? shell->moments().volume.as_double() : carrier_area * height);
    }
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
}
std::shared_ptr<const VoF::GeneralVoF> pure_cartesian_phases(unsigned nx, unsigned ny, unsigned nz)
{
    AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {nx, ny, nz};
    grid.extent = {1, 1, 1};
    auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < std::uint64_t(nx) * ny * nz; ++c)
        if (int(c % comm->getSize()) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider->detach_owned(owned, {'v', 'i', 's', 'c', 'o', 'u', 's', 'm', 'o', 'd', 'e'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        order.push_back(key);
        gas.push_back(0);
        carrier.push_back(hierarchy->geometry(leaf)->moments().volume.as_double());
    }
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
}
std::shared_ptr<const VoF::GeneralVoF> skew_coarse_fine_plane()
{
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = ArithmeticMode::Ordinary;
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 5; ++x)
            grid.xy_nodes.push_back({x / 4.0 + .2 * y / 2.0, y / 2.0, 0});
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
        {
            const unsigned p = 5 * y + x;
            grid.base_cells.push_back({p, p + 1, p + 6, p + 5});
        }
    for (unsigned x = 0; x < 4; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({10 + x, 11 + x, "wall"});
    }
    for (unsigned y = 0; y < 2; ++y)
    {
        grid.boundaries.push_back({5 * y, 5 * (y + 1), "wall"});
        grid.boundaries.push_back({5 * y + 4, 5 * (y + 1) + 4, "wall"});
    }
    grid.z_edges = {0, .5, 1};
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 16; ++c)
        if (int(c % comm->getSize()) == comm->getRank())
            owned.push_back(c);
    auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap = provider->detach_owned(owned, {'s', 'k', 'e', 'w', 'p', 'a', 's', 's', 'a', 'g', 'e'});
    auto raw = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> zero, full;
    for (const auto& [key, leaf] : raw->owned())
    {
        order.push_back(key);
        zero.push_back(0);
        full.push_back(PLIC::DynamicConvexGeometry::from_geometry(*raw->geometry(leaf), limits)->volume().as_double());
    }
    auto bound = std::make_shared<const VoF::GeneralVoF>(raw, order, zero, full);
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& [key, leaf] : bound->hierarchy()->owned())
    {
        const auto center =
            PLIC::DynamicConvexGeometry::from_geometry(*bound->hierarchy()->geometry(leaf), limits)->centroid();
        const double xi = center[0] - .2 * center[1];
        if (xi > .5 && xi < .75)
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    const auto plan = bound->hierarchy()->plan_adaptation(marks);
    const auto refined = bound->hierarchy()->apply(plan);
    order.clear();
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : refined->owned())
    {
        // Initialize both phases with ordinary FP geometry. Native authority
        // import supplies supports only; no rational reference clip or measure.
        const auto shell = PLIC::DynamicConvexGeometry::from_geometry(*refined->geometry(leaf), limits);
        const auto amounts = shell->phases({{1, .17, .11}, {0, 0, 0}, .65});
        order.push_back(key);
        gas.push_back(amounts.gas.as_double());
        carrier.push_back(amounts.carrier.as_double());
    }
    return std::make_shared<const VoF::GeneralVoF>(refined, std::move(order), std::move(gas), std::move(carrier));
}
VoFMaterial inviscid()
{
    VoFMaterial material;
    material.gas_viscosity = material.liquid_viscosity = material.surface_tension = 0;
    return material;
}
void enable_observed_exact_geometry_statistics(bool enabled)
{
    SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(enabled);
    VoF::GeneralVoF::enable_geometry_statistics(enabled);
    GeneralVoFSolver::enable_geometry_statistics(enabled);
}

void reset_observed_geometry_statistics()
{
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    PLIC::reset_geometry_statistics();
    VoF::GeneralVoF::reset_geometry_statistics();
    GeneralVoFSolver::reset_geometry_statistics();
}

std::array<Meshes::GeometryStatisticsSnapshot, 3> geometry_image_snapshots()
{
    return {
        {{SimpleFluid::PLIC::backend_statistics_domain<SimpleFluid::ArithmeticMode::Exact>(),
             PLIC::geometry_statistics_domain(),
             SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>(), PLIC::geometry_statistics()},
            VoF::GeneralVoF::geometry_statistics(), GeneralVoFSolver::geometry_statistics()}};
}

Meshes::GeometryStatisticsSnapshot observed_geometry_statistics()
{
    Meshes::GeometryStatisticsSnapshot result;
    std::set<const void*> exact_domains, bounded_domains;
    for (const auto& image : geometry_image_snapshots())
    {
        if (exact_domains.insert(image.exact_domain).second)
        {
            result.exact.source_coordinate_exports += image.exact.source_coordinate_exports;
            result.exact.source_support_exports += image.exact.source_support_exports;
            result.exact.volume_evaluations += image.exact.volume_evaluations;
            result.exact.moment_evaluations += image.exact.moment_evaluations;
            result.exact.clip_constructions += image.exact.clip_constructions;
            result.exact.intersection_constructions += image.exact.intersection_constructions;
            result.exact.face_measure_evaluations += image.exact.face_measure_evaluations;
            result.exact.source_membership_predicates += image.exact.source_membership_predicates;
            result.exact.measure_values += image.exact.measure_values;
            result.exact.measure_arithmetic += image.exact.measure_arithmetic;
            result.exact.cover_certifications += image.exact.cover_certifications;
            result.exact.maximum_result_numerator_bits =
                std::max(result.exact.maximum_result_numerator_bits, image.exact.maximum_result_numerator_bits);
            result.exact.maximum_result_denominator_bits =
                std::max(result.exact.maximum_result_denominator_bits, image.exact.maximum_result_denominator_bits);
        }
        if (bounded_domains.insert(image.bounded_domain).second)
        {
            result.bounded.source_imports += image.bounded.source_imports;
            result.bounded.constructions += image.bounded.constructions;
            result.bounded.intersections += image.bounded.intersections;
            result.bounded.support_distances += image.bounded.support_distances;
            result.bounded.volume_evaluations += image.bounded.volume_evaluations;
            result.bounded.moment_evaluations += image.bounded.moment_evaluations;
            result.bounded.partition_checks += image.bounded.partition_checks;
            result.bounded.structural_partition_reuses += image.bounded.structural_partition_reuses;
            result.bounded.floating_operations += image.bounded.floating_operations;
            result.bounded.interval_operations += image.bounded.interval_operations;
            result.bounded.ledger_operations += image.bounded.ledger_operations;
            result.bounded.wider_routes += image.bounded.wider_routes;
            result.bounded.rejections += image.bounded.rejections;
            result.bounded.exact_predicates += image.bounded.exact_predicates;
            result.bounded.rational_constructions += image.bounded.rational_constructions;
            result.bounded.rational_measures += image.bounded.rational_measures;
            result.bounded.rational_ledgers += image.bounded.rational_ledgers;
        }
    }
    return result;
}
} // namespace
