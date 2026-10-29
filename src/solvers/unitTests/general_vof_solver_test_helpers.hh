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
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
namespace AMR = SimpleFluid::Meshes::AMR;
// Qualification-only controls: component trace/rejection fixtures keep their defaults.
bool qualification_phase_cleanup_enabled()
{
    const auto value = std::getenv("SIMPLEFLUID_QUALIFICATION_PHASE_CLEANUP");
    const int enabled = value && std::string(value) == "1";
    const int local[2]{enabled, -enabled};
    int global[2]{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 2, local, global);
    if (global[0] != -global[1])
        throw std::runtime_error("Qualification phase-cleanup environment must agree across ranks.");
    return enabled;
}
struct QualificationPhaseReference
{
    std::vector<double> gas, carrier;
    std::array<double, 2> global_volume{};
    VoFStepDiagnostics physical_reference(VoFStepDiagnostics initial, const VoFMaterial& material) const
    {
        initial.gas_volume = global_volume[0];
        initial.gas_mass = material.gas_density * global_volume[0];
        initial.liquid_mass = material.liquid_density * global_volume[1];
        return initial;
    }
};
void qualification_phase_policy(VoF::GeneralVoFOptions& options, const std::vector<double>& gas,
    const std::vector<double>& carrier, const Teuchos::Comm<int>& comm, bool enabled,
    QualificationPhaseReference* raw = nullptr)
{
    // Every rank, including empty owners, contributes before initialization cleanup.
    double local[2]{}, global[2]{};
    for (const auto amount : gas)
        local[0] += amount;
    for (const auto amount : carrier)
        local[1] += amount;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 2, local, global);
    if (raw)
        *raw = {gas, carrier, {global[0], global[1]}};
    options.phase_cleanup = {0, {0, 0}};
    if (enabled && options.transport_backend != VoF::GeneralVoFTransportBackend::DonorCell)
    {
        options.phase_cleanup.relative_cutoff = 1e-14;
        options.phase_cleanup.maximum_absolute_volume = {1e-13 * global[0], 1e-13 * global[1]};
    }
}
void qualification_cleanup_metric(const std::string& name, long double value)
{
    std::ostringstream output;
    output << std::setprecision(20) << value;
    testing::Test::RecordProperty(name, output.str());
}
void record_qualification_cleanup(const GeneralVoFSolver& solver, const std::string& prefix = "")
{
    const auto& policy = solver.phases()->options().phase_cleanup;
    const auto& state = solver.phases()->phase_cleanup_state();
    const auto record = [&](const std::string& name, long double value)
    { qualification_cleanup_metric(prefix + "cleanup_" + name, value); };
    record("enabled", policy.relative_cutoff > 0);
    record("relative_cutoff", policy.relative_cutoff);
    record("cells", state.cleaned_cells);
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        const std::string name = phase == 0 ? "gas_" : "carrier_";
        record(name + "budget", policy.maximum_absolute_volume[phase]);
        record(name + "signed_volume", state.signed_volume[phase]);
        record(name + "absolute_volume", state.absolute_volume[phase]);
        record(name + "largest_absolute_volume", state.largest_absolute_volume[phase]);
        EXPECT_LE(state.absolute_volume[phase], policy.maximum_absolute_volume[phase]);
    }
    const auto prepared = solver.prepared_transport();
    if (!prepared)
        return;
    // These are reporting reductions; conservation gates retain raw physical references.
    double local[6]{}, global[6]{};
    for (std::size_t c = 0; c < prepared->new_mass.size(); ++c)
    {
        local[0] += static_cast<double>(prepared->raw_new_mass[c]);
        local[1] += static_cast<double>(prepared->new_mass[c]);
        local[2] += static_cast<double>(prepared->cleanup_mass[c]);
        if (c < prepared->cleanup_momentum.size())
            for (unsigned axis = 0; axis < 3; ++axis)
                local[3 + axis] += static_cast<double>(prepared->cleanup_momentum[c][axis]);
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 6, local, global);
    record("prepared_raw_mass", global[0]);
    record("prepared_cleaned_mass", global[1]);
    record("prepared_mass_correction", global[2]);
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const auto& momentum = solver.momentum_transport_diagnostics()[axis];
        const auto name = std::to_string(axis);
        record("prepared_momentum_" + name, global[3 + axis]);
        record("momentum_" + name, momentum.cleanup_momentum);
        record("accumulated_momentum_" + name, momentum.accumulated_cleanup_momentum);
        record("identity_relative_residual_" + name, momentum.cleanup_identity_relative_residual);
        record("physical_relative_residual_" + name, momentum.physical_relative_residual);
        EXPECT_LE(momentum.physical_relative_residual + momentum.cleanup_identity_relative_residual, 1e-12);
    }
}
std::shared_ptr<const VoF::GeneralVoF> phases(
    unsigned fixture, bool skew = false, bool single_layer = false, bool cleanup = true)
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
    VoF::GeneralVoFOptions options;
    if (!cleanup)
        options.phase_cleanup.relative_cutoff = 0;
    auto result = std::make_shared<const VoF::GeneralVoF>(
        hierarchy, std::move(order), std::move(gas), std::move(carrier), options);
    int local_present = result->mesh()->num_owned_cells() > 0, present = 0;
    int local_empty = !local_present, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_present, &present);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(present, owning_ranks);
    EXPECT_EQ(empty, comm->getSize() - owning_ranks);
    return result;
}
std::shared_ptr<const VoF::GeneralVoF> cylinder_sector_phases(bool spatial_partition = false,
    VoF::GeneralVoFTransportBackend backend = VoF::GeneralVoFTransportBackend::SweptPartition,
    bool cleanup = false, QualificationPhaseReference* raw = nullptr)
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
    auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 36; ++c)
    {
        int owner = int(c % comm->getSize());
        if (spatial_partition)
        {
            const auto layer = std::get<AMR::ProductCellAncestry>(provider->cell_key(c).ancestry).axial.root;
            owner = int(layer * std::min(3, comm->getSize()) / 3);
        }
        if (owner == comm->getRank())
            owned.push_back(c);
    }
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
    VoF::GeneralVoFOptions options;
    options.transport_backend = backend;
    qualification_phase_policy(options, gas, carrier, *comm, cleanup, raw);
    return std::make_shared<const VoF::GeneralVoF>(
        hierarchy, std::move(order), std::move(gas), std::move(carrier), options);
}
std::shared_ptr<const VoF::GeneralVoF> pure_cartesian_phases(
    unsigned nx, unsigned ny, unsigned nz, bool cleanup = true)
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
    VoF::GeneralVoFOptions options;
    if (!cleanup)
        options.phase_cleanup.relative_cutoff = 0;
    return std::make_shared<const VoF::GeneralVoF>(
        hierarchy, std::move(order), std::move(gas), std::move(carrier), options);
}
std::shared_ptr<const VoF::GeneralVoF> skew_coarse_fine_plane(std::optional<bool> cleanup,
    VoF::GeneralVoFTransportBackend backend, QualificationPhaseReference* raw_reference = nullptr,
    bool record_balance_work = false)
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
        full.push_back(PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_geometry(*raw->geometry(leaf), limits)
                ->volume()
                .as_double());
    }
    auto bound = std::make_shared<const VoF::GeneralVoF>(raw, order, zero, full);
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& [key, leaf] : bound->hierarchy()->owned())
    {
        const auto center = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_geometry(
            *bound->hierarchy()->geometry(leaf), limits)
                                ->centroid();
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
        const auto shell =
            PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_geometry(*refined->geometry(leaf), limits);
        const auto amounts = shell->phases({{1, .17, .11}, {0, 0, 0}, .65});
        order.push_back(key);
        gas.push_back(amounts.gas.as_double());
        carrier.push_back(amounts.carrier.as_double());
    }
    VoF::GeneralVoFOptions options;
    options.record_balance_work = record_balance_work;
    options.transport_backend = backend;
    if (cleanup)
        qualification_phase_policy(options, gas, carrier, *comm, *cleanup, raw_reference);
    return std::make_shared<const VoF::GeneralVoF>(
        refined, std::move(order), std::move(gas), std::move(carrier), options);
}
std::shared_ptr<const VoF::GeneralVoF> skew_coarse_fine_plane(bool record_balance_work = false)
{
    return skew_coarse_fine_plane(
        std::nullopt, VoF::GeneralVoFTransportBackend::SweptPartition, nullptr, record_balance_work);
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
