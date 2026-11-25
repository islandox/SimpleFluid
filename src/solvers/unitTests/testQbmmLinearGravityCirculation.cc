/** Heated, closed circulation in two native cylindrical meshes, with a passive bulk gas and inert bubble family. */
#include "QbmmCylindricalTestMesh.hh"
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using Controller = R::ReactingMixtureCoupler;
using LO = Flow::Pack::local_ordinal_type;
using Phase = R::SpeciesPhase;
using FamilyTransport = R::QbmmModelOptions::FamilyTransport;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double inner = 1., outer = 2., sector_angle = 1., height = 1.;
constexpr double p0 = 1000., t0 = 300., gas_constant = 1., liquid_density = 10., gas_cv = 20.;
constexpr double liquid_u0 = 30000., liquid_cp = 100., family_fraction = .375, bulk_gas_fraction = .125;
constexpr double gas_fraction = family_fraction + bulk_gas_fraction, mass_scale = 1. / 1024;
constexpr double gravity = -9.81, conductivity = 12., wall_power_per_volume = 3200.;
constexpr double horizon = 1. / 8, timestep = 1. / 64;
constexpr unsigned sector_wedges = 4, polygon_sides = 8;
constexpr size_t phases = 3, family = 1;

namespace Cylinder = SimpleFluid::Testing::QbmmCylinder;
using Geometry = Cylinder::Geometry;

ArrReal edges(unsigned count, double start, double end)
{
    ArrReal cuts;
    for (unsigned i = 0; i <= count; ++i)
        cuts.push_back(std::lerp(start, end, double(i) / count));
    return cuts;
}

std::shared_ptr<Flow::Mesh> make_mesh(Geometry geometry, unsigned count)
{
    const auto radial = edges(count, inner, outer);
    const auto axial = edges(count, 0., height);
    if (geometry == Geometry::AnnularSector)
        return std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cylindrical>(
            Vec3D<ArrReal>{radial, edges(sector_wedges, -.5 * sector_angle, .5 * sector_angle), axial}));

    // The same octagonal boundaries at both resolutions. Radial strips between
    // homothetic regular polygons give orthogonal adjacent centers.
    using Semi = Meshes::SemiStructuredXY_Z;
    Arr<Semi::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    for (unsigned wedge = 0; wedge < polygon_sides; ++wedge)
    {
        const double theta = 2 * std::numbers::pi * wedge / polygon_sides;
        for (const double r : radial)
            nodes.push_back({r * std::cos(theta), r * std::sin(theta), 0.});
    }
    for (unsigned wedge = 0; wedge < polygon_sides; ++wedge)
        for (unsigned i = 0; i < count; ++i)
        {
            const unsigned left = wedge * (count + 1) + i;
            const unsigned right = ((wedge + 1) % polygon_sides) * (count + 1) + i;
            cells.push_back({left, left + 1, right + 1, right});
        }
    return std::make_shared<Flow::Mesh>(std::make_shared<Semi>(nodes, cells, axial));
}

struct FourierSample
{
    double owner_temperature, neighbor_temperature, time, dt;
};

struct Vessel
{
    Geometry geometry;
    unsigned count;
    double acceleration, heating;
    bool kinetic;
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<Flow> flow;
    std::vector<std::array<double, 2>> wall_area;
    std::vector<double> hydrostatic_head;
    mutable std::vector<std::optional<FourierSample>> fourier_samples;

    Vessel(Geometry kind, unsigned resolution, double g, double heat_sign, bool kinetic_method = false)
        : geometry(kind), count(resolution), acceleration(g), heating(heat_sign), kinetic(kinetic_method),
          mesh(kinetic ? Cylinder::mesh(kind, resolution, edges(resolution, 0., height)) : make_mesh(kind, resolution)),
          wall_area(mesh->num_owned_cells()), hydrostatic_head(count), fourier_samples(mesh->num_owned_faces())
    {
        const auto reduced = initialize_wall_areas();
        const auto expected = wall_areas();
        for (size_t wall = 0; wall < expected.size(); ++wall)
            EXPECT_NEAR(reduced[wall], expected[wall], 2e-13 * expected[wall]);

        // Isothermal column with fixed initial void fractions: rho(p)=a+b*p.
        // Solve the trapezoidal hydrostatic jumps with volume-mean pressure p0.
        constexpr double a = (1. - gas_fraction) * liquid_density;
        constexpr double b = gas_fraction / (gas_constant * t0);
        const double factor = .5 * acceleration * (height / count) * b;
        hydrostatic_head.front() = 1.;
        for (size_t j = 1; j < hydrostatic_head.size(); ++j)
            hydrostatic_head[j] = hydrostatic_head[j - 1] * (1. + factor) / (1. - factor);
        const double mean = std::accumulate(hydrostatic_head.begin(), hydrostatic_head.end(), 0.) / count;
        for (double& head : hydrostatic_head)
            head = (p0 + a / b) * (head / mean - 1.);

        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {45, "liquid", 1., {{"L", 1}}, {Phase::Liquid}}, {46, "bubble gas", 1., {{"F", 1}}, {Phase::Dispersed}},
            {47, "passive gas", 1., {{"G", 1}}, {Phase::Resolved}}});
        constexpr R::ThermodynamicBounds bounds{250., 400., 750., 1250.};
        constexpr auto energy = R::EnergyConvention::SensibleEnergy;
        const auto liquid = R::make_linear_compressible_mixture_liquid(
            catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{t0, p0, liquid_density, liquid_cp, 0., 0., liquid_u0}; },
            energy);
        const auto gas = R::make_ideal_gas_mixture(catalog,
            std::vector<R::IdealGasCaloricSpecies>(phases, {gas_cv, t0, gas_cv * t0}), bounds, energy, gas_constant);
        Flow::Options options;
        options.minimum_pressure = bounds.minimum_pressure;
        options.maximum_pressure = bounds.maximum_pressure;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.qbmm_model.family_transport = FamilyTransport::RealizableLinearSSPRK2;
        options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SharedCarrier;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "two sizes"},
            {Flow::PhaseRole::BulkGas, "passive gas"}};
        options.qbmm_eos_law_identities = {
            R::ThermalLawIdentity{"Q8c constant-density liquid", 1, catalog->fingerprint()}, std::nullopt,
            std::nullopt};
        R::QbmmClosureVersions versions;
        if (kinetic)
        {
            options.qbmm_model.family_transport = FamilyTransport::KineticEuler;
            options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::NodeDependentSlip;
            const R::QbmmLawMetadata law{
                "heated vessel axial slip", 1, R::QbmmLawKind::NodeSlip, "mass-dependent axial transport", "m/s"};
            versions.laws.push_back(law);
            options.qbmm_node_slip_laws = {nullptr,
                std::make_shared<const R::QbmmNodeSlipLaw>(R::QbmmNodeSlipLaw{law, [](const auto& atom)
                    { return std::array{0., 0., atom.mass_kg_per_bubble < 2. * mass_scale ? .015625 : .03125}; }}),
                nullptr};
        }
        const auto population = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(),
            "two sizes", options.phase_roles[family], mass_scale,
            std::vector<R::QbmmMassFraction>{{45, 0.}, {46, 1.}, {47, 0.}}, versions);
        options.qbmm_families = {nullptr, population, nullptr};
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas}, options);
        Flow::State initial;
        initial.thermodynamic_pressure = p0;
        initial.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*population), std::nullopt};
        initial.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const auto id = static_cast<LO>(c);
            auto& cell = initial.cells[c];
            const double volume = mesh->cell_volume(id);
            cell.mechanical_pressure = initial_head(mesh->cell_centroid(id).z);
            const double pressure = p0 + cell.mechanical_pressure;
            const double liquid_moles = (1. - gas_fraction) * liquid_density * volume;
            const double gas_moles = pressure * volume / (gas_constant * t0);
            const double bubbles = family_fraction * gas_moles, background = bulk_gas_fraction * gas_moles;
            cell.phases = {{liquid, {liquid_moles, 0., 0.}, pressure, liquid_moles * liquid_u0, t0},
                {gas, {0., bubbles, 0.}, pressure, bubbles * gas_cv * t0, t0},
                {gas, {0., 0., background}, pressure, background * gas_cv * t0, t0}};
            // Two atoms at x=1 and x=4 with counts 2w and w.
            const double weight = bubbles / (6. * mass_scale);
            cell.phase_numbers = {std::nullopt, 3. * weight, std::nullopt};
            cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{18. * weight, 66. * weight}, std::nullopt};
        }
        flow->initialize(std::move(initial));
    }

    double initial_head(double z) const { return hydrostatic_head.at(static_cast<size_t>(count * z / height)); }

    double volume() const
    {
        const double angle = Cylinder::native(geometry)
                                 ? Cylinder::span(geometry)
                                 : polygon_sides * std::sin(2. * std::numbers::pi / polygon_sides);
        const double r0 = Cylinder::inner_radius(geometry);
        return .5 * angle * (outer * outer - r0 * r0) * height;
    }

    std::array<double, 2> wall_areas() const
    {
        const double circumference_per_radius = Cylinder::native(geometry)
                                                    ? Cylinder::span(geometry)
                                                    : 2. * polygon_sides * std::sin(std::numbers::pi / polygon_sides);
        if (geometry == Geometry::PolygonalDisk)
            return {.5 * circumference_per_radius * outer * height, .5 * circumference_per_radius * outer * height};
        return {circumference_per_radius * inner * height, circumference_per_radius * outer * height};
    }

    std::array<double, 2> wall_powers(size_t c) const
    {
        const auto areas = wall_areas();
        // Equal-area half annuli receive different wall flux densities, giving the
        // two neighbors across the actual theta seam a resolved thermal contrast.
        const double variation = kinetic && geometry == Geometry::NativeAnnulus
                                     ? (mesh->cell_centroid(static_cast<LO>(c)).y > 0. ? 1.25 : .75)
                                     : 1.;
        const double power = heating * wall_power_per_volume * volume() * variation;
        return {power * wall_area[c][0] / areas[0], -power * wall_area[c][1] / areas[1]};
    }

    // Actual centroids include the angular averaging of a sector or polygon.
    double averaged_radius_scale() const
    {
        if (geometry == Geometry::PolygonalAnnulus)
            return std::cos(std::numbers::pi / polygon_sides);
        const double half_angle = .5 * sector_angle / sector_wedges;
        return std::sin(half_angle) / half_angle;
    }

    template<class Values> Values sum(const Values& local) const
    {
        auto global = local;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local.size()),
            local.data(), global.data());
        return global;
    }

    std::array<double, 2> initialize_wall_areas()
    {
        std::array<double, 2> local{};
        collective_detail::collective_local_validation(*mesh, "Q8c radial wall faces",
            [&]
            {
                for (size_t f = 0; f < mesh->num_owned_faces(); ++f)
                {
                    const auto id = static_cast<LO>(f);
                    if (!mesh->is_geometry_exterior_face(id))
                        continue;
                    const auto normal = mesh->face_normal(id);
                    const auto center = mesh->face_centroid(id);
                    const double alignment = normal.dot(center) / std::hypot(center.x, center.y);
                    if (normal.z != 0. || std::abs(alignment) < .5)
                        continue; // End caps and native sector's azimuthal walls.
                    const size_t wall =
                        geometry == Geometry::PolygonalDisk ? size_t(center.x < 0.) : size_t(alignment > 0.);
                    const double area = mesh->face_area(id);
                    wall_area.at(mesh->owner_cell(id))[wall] += area;
                    local[wall] += area;
                }
            });
        return sum(local);
    }

    R::ReactingMixtureOptions controller_options() const
    {
        R::ReactingMixtureOptions options;
        options.maximum_correctors = flow->options().maximum_correctors;
        options.mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::SensibleEnergy);
        options.phases = {{Phase::Liquid, {}, {}}, {Phase::Dispersed, "two sizes", {}}, {Phase::Resolved, {}, {}}};
        if (acceleration != 0.)
            options.body_acceleration = Flow::Vector{0., 0., acceleration};
        options.source_identity = "Q8c equal and opposite radial wall powers";
        options.source = [this](const R::CoupledReactionContext& point, double)
        {
            const auto c = point.owned_cell;
            const auto watts = wall_powers(c);
            R::ReactionSourceInput input;
            input.heat.rate = R::VolumetricHeatRate{
                (watts[0] + watts[1]) / mesh->cell_volume(static_cast<LO>(c)), R::SourceVolumeBasis::BulkVolume};
            return input;
        };
        R::MixtureFaceTransportOptions contact;
        contact.identity = "Q8c conducting liquid faces";
        contact.schema_fingerprint = catalog->fingerprint();
        contact.physical_phase = 0;
        contact.fourier =
            R::MixtureFourierLaw{"Q8c k*A/d", 1, catalog->fingerprint(), [this](const R::MixtureFaceContext& face)
                {
                    const auto row = mesh->owned_face_map()->getLocalElement(face.algebraic_identity);
                    fourier_samples.at(row) = FourierSample{
                        face.owner.temperature, face.neighbor.temperature, face.accepted_time, face.time_step};
                    return conductivity * face.area / face.center_distance;
                }};
        options.face_transport = {std::move(contact)};
        // A zero interphase transfer exposes the per-cell conditioned energy
        // receipt, including the raw wall and face operands used by the solver.
        const auto thermal = std::make_shared<const R::InterphaseConfiguration>(
            R::InterphaseConfiguration{{"Q8c wall and Fourier energy accounting", 1, catalog->fingerprint()},
                {{"zero interphase heat", R::InterphaseStageKind::Heat, {}, {{0, 2}}}}, {nullptr, nullptr, nullptr},
                nullptr});
        options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
            R::MixtureInterphaseEventProvider{thermal->metadata, thermal,
                [](const R::MixtureInterphaseContext&) { return std::vector<std::vector<double>>{{0.}}; }});
        return options;
    }
};

double moles(const R::PhaseThermalState& phase)
{
    return std::accumulate(phase.moles.begin(), phase.moles.end(), 0.);
}

double phase_volume(const R::PhaseThermalState& phase, size_t p)
{
    return p == 0 ? moles(phase) / liquid_density
                  : moles(phase) * gas_constant * phase.temperature / phase.absolute_pressure;
}

struct Inventory
{
    std::array<double, phases> species{};
    R::QbmmMomentBlock population{};
    std::array<double, 3> momentum{};
    double energy = 0.;

    double mass() const { return std::accumulate(species.begin(), species.end(), 0.); }
};

Inventory inventory(const Vessel& vessel, const Flow::State& state)
{
    Inventory local;
    for (const auto& cell : state.cells)
    {
        local.population[0] += *cell.phase_numbers[family];
        local.population[1] += cell.phases[family].moles[family] / mass_scale;
        local.population[2] += cell.qbmm_shape[family]->H2;
        local.population[3] += cell.qbmm_shape[family]->H3;
        double mass = 0.;
        for (const auto& phase : cell.phases)
        {
            for (size_t s = 0; s < phases; ++s)
                local.species[s] += phase.moles[s];
            mass += moles(phase);
            local.energy += phase.internal_energy;
        }
        for (size_t d = 0; d < local.momentum.size(); ++d)
        {
            local.momentum[d] += mass * cell.velocity.component(d);
            if (vessel.kinetic && d == 2)
            {
                const double n = *cell.phase_numbers[family], m = moles(cell.phases[family]) / mass_scale;
                const double coarse = (m - n) / 3.;
                local.momentum[d] += mass_scale * (.015625 * (n - coarse) + .03125 * 4. * coarse);
            }
        }
    }
    return {vessel.sum(local.species), vessel.sum(local.population), vessel.sum(local.momentum),
        vessel.sum(std::array{local.energy})[0]};
}

struct Metrics
{
    double pressure_work = 0., fourier_exchange = 0., maximum_pair_defect = 0., maximum_occupancy = 0.;
    double maximum_continuity = 0., maximum_solver_residual = 0., maximum_linear_cfl = 0.;
    double maximum_momentum_defect = 0., seam_fourier_exchange = 0., partition_seam_fourier_exchange = 0.;
    double maximum_frozen_fourier_discrepancy = 0., maximum_recovered_fourier_discrepancy = 0.;
    double maximum_recovered_fourier_discrepancy_ulps = 0., maximum_relative_recovered_fourier_discrepancy = 0.;
    std::array<double, 2> wall_joules{};
    int maximum_evaluations = 0;
};

void compare_fourier_temperatures(const Vessel& vessel, const Controller::Trial& trial,
    const std::vector<double>& applied_owner_heat, double time, double dt, Metrics& metrics)
{
    using Field = Flow::Traits::scalar_cell_type;
    const auto& mesh = *vessel.mesh;
    Field recovered(vessel.mesh, 0., "Q8c_recovered_liquid_temperature");
    collective_detail::collective_local_validation(mesh, "Q8c recovered Fourier temperature field",
        [&]
        {
            for (size_t c = 0; c < trial.state().cells.size(); ++c)
                recovered.set_owned_value(static_cast<LO>(c), trial.state().cells[c].phases.at(0).temperature);
        });
    recovered.sync_ghosts();
    std::array<double, 4> local{}, global{};
    collective_detail::collective_local_validation(mesh, "Q8c applied and recovered Fourier heat",
        [&]
        {
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const auto face = static_cast<LO>(f);
                if (mesh.is_geometry_exterior_face(face))
                    continue;
                const auto row = mesh.owned_face_map()->getLocalElement(mesh.face_global_id(face));
                const auto& frozen = vessel.fourier_samples.at(row).value();
                EXPECT_EQ(frozen.time, time);
                EXPECT_EQ(frozen.dt, dt);
                const double weight = dt * (conductivity * mesh.face_area(face) / mesh.face_cell_center_distance(face));
                const double applied = applied_owner_heat.at(mesh.face_geometry_global_id(face));
                const double expected = weight * (frozen.neighbor_temperature - frozen.owner_temperature);
                EXPECT_NEAR(applied, expected, 8. * std::numeric_limits<double>::epsilon() * std::abs(expected));
                local[0] = std::max(local[0], std::abs(applied - expected));

                // The recovered endpoint may differ from the last frozen source
                // operand. Report that difference without imposing a new gate.
                const double owner = recovered.local_value(mesh.owner_cell(face));
                const double neighbor = recovered.local_value(mesh.neighbor_cell(face));
                const double recovered_heat = weight * (neighbor - owner);
                const double discrepancy = std::abs(applied - recovered_heat);
                const double largest_temperature =
                    std::max({owner, neighbor, frozen.owner_temperature, frozen.neighbor_temperature});
                const double ulp =
                    std::nextafter(largest_temperature, std::numeric_limits<double>::infinity()) - largest_temperature;
                const double heat_scale = std::max(std::abs(applied), std::abs(recovered_heat));
                local[1] = std::max(local[1], discrepancy);
                local[2] = std::max(local[2], discrepancy / (weight * ulp));
                if (heat_scale > 0.)
                    local[3] = std::max(local[3], discrepancy / heat_scale);
            }
        });
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, static_cast<int>(local.size()),
        local.data(), global.data());
    metrics.maximum_frozen_fourier_discrepancy = std::max(metrics.maximum_frozen_fourier_discrepancy, global[0]);
    metrics.maximum_recovered_fourier_discrepancy = std::max(metrics.maximum_recovered_fourier_discrepancy, global[1]);
    metrics.maximum_recovered_fourier_discrepancy_ulps =
        std::max(metrics.maximum_recovered_fourier_discrepancy_ulps, global[2]);
    metrics.maximum_relative_recovered_fourier_discrepancy =
        std::max(metrics.maximum_relative_recovered_fourier_discrepancy, global[3]);
}

void check_step(const Vessel& vessel, Controller& controller, double dt, Metrics& metrics)
{
    const auto& mesh = *vessel.mesh;
    const auto& accepted = vessel.flow->state();
    const auto revision = accepted.revision;
    const auto time = accepted.time;
    const auto before = inventory(vessel, accepted);
    std::ranges::fill(vessel.fourier_samples, std::nullopt);
    const auto trial = controller.preview(dt);
    EXPECT_EQ(vessel.flow->state().revision, revision);
    EXPECT_EQ(vessel.flow->state().time, time);
    const auto after = inventory(vessel, trial.state());
    const auto& diagnostics = trial.diagnostics().flow;
    const auto& controls = vessel.flow->options();
    if (vessel.kinetic)
    {
        EXPECT_EQ(diagnostics.family_transport, FamilyTransport::KineticEuler);
        EXPECT_GE(diagnostics.kinetic_substeps, 1);
        EXPECT_GT(diagnostics.maximum_kinetic_cfl, 0.);
        EXPECT_LE(diagnostics.maximum_kinetic_cfl, controls.qbmm_model.kinetic_cfl);
    }
    else
    {
        EXPECT_EQ(diagnostics.family_transport, FamilyTransport::RealizableLinearSSPRK2);
        EXPECT_GE(diagnostics.linear_substeps, 1);
        EXPECT_EQ(diagnostics.linear_stages, 2 * static_cast<std::uint64_t>(diagnostics.linear_substeps));
        EXPECT_LE(diagnostics.maximum_linear_cfl, controls.qbmm_model.linear_cfl);
    }
    EXPECT_GE(diagnostics.source_evaluations, 2);
    EXPECT_LE(diagnostics.source_evaluations, controls.maximum_correctors);
    EXPECT_TRUE(diagnostics.pressure_linear.converged);
    EXPECT_TRUE(diagnostics.momentum_linear.converged);
    EXPECT_TRUE(diagnostics.transport_linear.converged);
    const std::array gates{diagnostics.maximum_source_residual, diagnostics.maximum_transport_residual,
        diagnostics.maximum_thermal_residual, diagnostics.maximum_conditioned_thermal_residual,
        diagnostics.maximum_event_residual, diagnostics.maximum_shape_source_residual,
        diagnostics.maximum_shape_transport_residual, diagnostics.maximum_species_transport_residual,
        diagnostics.maximum_energy_transport_residual, diagnostics.maximum_population_transport_residual,
        diagnostics.maximum_linear_stage_residual, diagnostics.maximum_linear_step_residual};
    for (const double residual : gates)
        EXPECT_LE(residual, 1.);
    EXPECT_LE(diagnostics.maximum_occupancy_error, controls.volume_relative_tolerance);
    EXPECT_LE(diagnostics.maximum_continuity_residual, controls.continuity_tolerance);
    for (size_t s = 0; s < phases; ++s)
        EXPECT_NEAR(after.species[s], before.species[s], controls.conservation_relative_tolerance * before.species[s]);
    for (size_t k = 0; k < before.population.size(); ++k)
        EXPECT_NEAR(
            after.population[k], before.population[k], controls.conservation_relative_tolerance * before.population[k]);

    const auto nc = mesh.owned_cell_map()->getGlobalNumElements();
    const auto nf = mesh.owned_face_map()->getGlobalNumElements();
    std::vector<double> volume_transfer(phases * nc);
    std::vector<double> owned_contact(nf), left(nf), right(nf), left_count(nf), right_count(nf);
    const auto packets = trial.owned_phase_transport();
    collective_detail::collective_local_validation(mesh, "Q8c advected material volume from owned faces",
        [&]
        {
            if (packets.size() != mesh.num_owned_faces())
                throw std::runtime_error("Missing cylindrical phase transport packets.");
            for (size_t f = 0; f < packets.size(); ++f)
            {
                const auto id = static_cast<LO>(f);
                const auto& packet = packets[f];
                const bool exterior = mesh.is_geometry_exterior_face(id);
                const auto owner = mesh.cell_geometry_global_id(mesh.owner_cell(id));
                owned_contact.at(mesh.face_geometry_global_id(id)) = exterior ? 0. : 1.;
                EXPECT_EQ(packet.mesh_volume_flux, 0.);
                for (size_t p = 0; p < phases; ++p)
                {
                    const double rate = p == family
                                            ? packet.qbmm_flux.at(p).value().material_volume_flux
                                            : packet.relative_phase_volume_flux.at(p) * packet.material_fraction.at(p);
                    if (exterior)
                    {
                        EXPECT_EQ(packet.phase_volume_flux.at(p), 0.);
                        EXPECT_EQ(rate, 0.);
                    }
                    else
                    {
                        const auto neighbor = mesh.cell_geometry_global_id(mesh.neighbor_cell(id));
                        volume_transfer.at(phases * owner + p) -= dt * rate;
                        volume_transfer.at(phases * neighbor + p) += dt * rate;
                    }
                }
                if (exterior)
                {
                    const auto& qbmm = packet.qbmm_flux.at(family).value();
                    EXPECT_EQ(qbmm.population_number_flux, 0.);
                    EXPECT_EQ(qbmm.shape_flux.H2, 0.);
                    EXPECT_EQ(qbmm.shape_flux.H3, 0.);
                    EXPECT_EQ(qbmm.internal_energy_flux, 0.);
                    for (const double molar_flux : qbmm.species_molar_flux)
                        EXPECT_EQ(molar_flux, 0.);
                }
            }
        });
    volume_transfer = vessel.sum(volume_transfer);
    std::array<double, 3> budget{}; // Inner and outer external joules, then EOS work.
    collective_detail::collective_local_validation(mesh, "Q8c cylindrical caloric and Fourier receipts",
        [&]
        {
            const auto& receipt = trial.interphase_receipt();
            if (!receipt || receipt->cells.size() != accepted.cells.size())
                throw std::runtime_error("Missing cylindrical Fourier cell receipts.");
            for (size_t c = 0; c < accepted.cells.size(); ++c)
            {
                const auto id = static_cast<LO>(c);
                const auto gid = mesh.cell_geometry_global_id(id);
                const auto cell_volume = mesh.cell_volume(id);
                const auto watts = vessel.wall_powers(c);
                const double wall_heat = dt * (watts[0] + watts[1]);
                const double reported_heat = trial.local_sources().at(c).externalHeat;
                EXPECT_NEAR(reported_heat, wall_heat, 1e-12 * std::max(1., std::abs(wall_heat)));
                for (size_t wall = 0; wall < watts.size(); ++wall)
                    if (watts[wall] != 0.)
                        budget[wall] += reported_heat;

                std::vector<LO> contacts;
                for (const auto contact : mesh.faces(id))
                    if (!mesh.is_geometry_exterior_face(contact))
                        contacts.push_back(contact);
                std::ranges::sort(contacts, {}, [&](LO face) { return mesh.face_geometry_global_id(face); });
                const auto& raw = receipt->cells.at(c).raw_static_energy.at(0);
                if (raw.size() < contacts.size())
                    throw std::runtime_error("Missing cylindrical Fourier contact samples.");
                // Only one static Fourier stage is configured. Its per-cell raw
                // entries follow the registered wall terms in global face order.
                const auto start = raw.end() - static_cast<std::ptrdiff_t>(contacts.size());
                EXPECT_NEAR(
                    std::accumulate(raw.begin(), start, 0.), wall_heat, 1e-12 * std::max(1., std::abs(wall_heat)));
                for (size_t i = 0; i < contacts.size(); ++i)
                {
                    const auto face = contacts[i];
                    const auto face_gid = mesh.face_geometry_global_id(face);
                    if (mesh.owner_cell(face) == id)
                    {
                        left.at(face_gid) = start[i];
                        ++left_count.at(face_gid);
                    }
                    else
                    {
                        right.at(face_gid) = start[i];
                        ++right_count.at(face_gid);
                    }
                }

                double occupied = 0.;
                for (size_t p = 0; p < phases; ++p)
                {
                    const auto& phase = trial.state().cells.at(c).phases.at(p);
                    const auto& transported = trial.transport_snapshot().cells.at(c).phases.at(p);
                    const double v = phase_volume(phase, p);
                    occupied += v;
                    const double advected =
                        phase_volume(accepted.cells[c].phases.at(p), p) + volume_transfer.at(phases * gid + p);
                    EXPECT_NEAR(transported.advected_material_volume, advected, 2e-12 * cell_volume);
                    const double work = -phase.absolute_pressure * (v - advected);
                    budget[2] += work;
                    const double heat = trial.final_thermal_evaluations()[c].applied_phase_energy.at(p);
                    EXPECT_NEAR(phase.internal_energy, transported.internal_energy + heat + work,
                        4e-9 * std::abs(phase.internal_energy));
                }
                EXPECT_NEAR(occupied, cell_volume, controls.volume_relative_tolerance * cell_volume);
            }
        });
    budget = vessel.sum(budget);
    left = vessel.sum(left);
    right = vessel.sum(right);
    left_count = vessel.sum(left_count);
    right_count = vessel.sum(right_count);
    owned_contact = vessel.sum(owned_contact);
    for (size_t f = 0; f < nf; ++f)
    {
        EXPECT_EQ(left_count[f], owned_contact[f]) << "owner incidence of geometry face " << f;
        EXPECT_EQ(right_count[f], owned_contact[f]) << "neighbor incidence of geometry face " << f;
        EXPECT_EQ(left[f], -right[f]) << "Fourier transfer on geometry face " << f;
        metrics.maximum_pair_defect = std::max(metrics.maximum_pair_defect, std::abs(left[f] + right[f]));
        metrics.fourier_exchange += std::abs(left[f]);
    }
    compare_fourier_temperatures(vessel, trial, left, time, dt, metrics);
    if (vessel.kinetic && vessel.geometry == Geometry::NativeAnnulus)
    {
        std::array<double, 2> exchange{};
        for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
        {
            const auto face = static_cast<LO>(f);
            const auto x = mesh.face_centroid(face);
            if (mesh.is_geometry_exterior_face(face) || x.x <= 0. || std::abs(x.y) > 1e-12 ||
                std::abs(mesh.face_normal(face).y) < .99)
                continue;
            const auto l = mesh.owner_cell(face), r = mesh.neighbor_cell(face);
            EXPECT_LT(mesh.cell_centroid(l).y * mesh.cell_centroid(r).y, 0.);
            const auto gid = mesh.face_geometry_global_id(face);
            const double joules = std::abs(left.at(gid));
            exchange[0] += joules;
            if (!mesh.is_owned_cell(l) || !mesh.is_owned_cell(r))
                exchange[1] += joules;
        }
        const auto totals = vessel.sum(exchange);
        metrics.seam_fourier_exchange += totals[0];
        metrics.partition_seam_fourier_exchange += totals[1];
    }
    const double expected = vessel.heating * wall_power_per_volume * vessel.volume() * dt;
    EXPECT_NEAR(budget[0], expected, 2e-12 * std::abs(expected));
    EXPECT_NEAR(budget[1], -expected, 2e-12 * std::abs(expected));
    EXPECT_NEAR(trial.diagnostics().external_heat, 0., 2e-12 * std::abs(expected));
    EXPECT_NEAR(diagnostics.pressure_work, budget[2], 2e-9 * vessel.volume());
    EXPECT_NEAR(after.energy - before.energy, budget[0] + budget[1] + budget[2],
        controls.conservation_relative_tolerance * diagnostics.caloric_energy_scale +
            diagnostics.internal_energy_roundoff_bound);
    for (size_t d = 0; d < before.momentum.size(); ++d)
    {
        const double impulse = d == 2 ? dt * vessel.acceleration * before.mass() : 0.;
        EXPECT_NEAR(diagnostics.external_impulse.component(d), impulse, 2e-12 * before.mass());
        if (vessel.acceleration != 0.)
        {
            EXPECT_TRUE(diagnostics.impulse_parts.has_value());
            if (diagnostics.impulse_parts)
            {
                EXPECT_NEAR(diagnostics.impulse_parts->body.component(d), impulse, 2e-12 * before.mass());
                EXPECT_EQ(diagnostics.impulse_parts->other.component(d), 0.);
            }
        }
        const double defect =
            after.momentum[d] - before.momentum[d] - impulse - diagnostics.pressure_impulse.component(d);
        EXPECT_NEAR(defect, 0., 4e-10 * before.mass());
        metrics.maximum_momentum_defect = std::max(metrics.maximum_momentum_defect, std::abs(defect));
    }

    metrics.wall_joules[0] += budget[0];
    metrics.wall_joules[1] += budget[1];
    metrics.pressure_work += budget[2];
    metrics.maximum_evaluations = std::max(metrics.maximum_evaluations, diagnostics.source_evaluations);
    metrics.maximum_occupancy = std::max(metrics.maximum_occupancy, diagnostics.maximum_occupancy_error);
    metrics.maximum_continuity = std::max(metrics.maximum_continuity, diagnostics.maximum_continuity_residual);
    metrics.maximum_solver_residual = std::max(metrics.maximum_solver_residual, std::ranges::max(gates));
    metrics.maximum_linear_cfl = std::max(metrics.maximum_linear_cfl, diagnostics.maximum_linear_cfl);
    controller.commit(trial);
    EXPECT_EQ(vessel.flow->state().time, time + dt);
    EXPECT_EQ(vessel.flow->state().revision, revision + 1);
}

struct Observables
{
    double inner_upflow, outer_upflow, circulation, thermal_dipole, pressure_head, pressure_dipole;
    double gas_fraction, area, d32;
};

Observables observe(const Vessel& vessel, const Controller& controller)
{
    const auto& mesh = *vessel.mesh;
    const auto report = controller.qbmm_report();
    const auto& state = vessel.flow->state();
    const double scale = vessel.averaged_radius_scale();
    const double mean_radius =
        scale * (2. / 3.) * (outer * outer * outer - inner * inner * inner) / (outer * outer - inner * inner);
    enum Integral
    {
        Volume,
        InnerRate,
        InnerArea,
        OuterRate,
        OuterArea,
        Circulation,
        TemperatureDipole,
        PressureHead,
        PressureDipole,
        GasVolume,
        Diameter2,
        Diameter3,
        Count
    };
    std::array<double, Count> local{};
    collective_detail::collective_local_validation(mesh, "Q8c cylindrical circulation observables",
        [&]
        {
            for (size_t c = 0; c < state.cells.size(); ++c)
            {
                const auto id = static_cast<LO>(c);
                const auto center = mesh.cell_centroid(id);
                const auto& cell = state.cells[c];
                const double radius = std::hypot(center.x, center.y);
                const double r = mean_radius - radius;
                const double z = center.z - .5 * height;
                const double volume = mesh.cell_volume(id);
                const double radial_velocity = (center.x * cell.velocity.x + center.y * cell.velocity.y) / radius;
                local[Volume] += volume;
                local[Circulation] += volume * (z * radial_velocity + r * cell.velocity.z);
                local[TemperatureDipole] += volume * r * (cell.phases[0].temperature - t0);
                local[PressureHead] += volume * z * cell.mechanical_pressure;
                local[PressureDipole] += volume * r * z * (cell.mechanical_pressure - vessel.initial_head(center.z));
                const double gas_volume = phase_volume(cell.phases[family], family) + phase_volume(cell.phases[2], 2);
                local[GasVolume] += gas_volume;
                const auto& cell_report = report.cells.at(c);
                EXPECT_NEAR(cell_report.gas_volume_fraction, gas_volume / volume, 2e-12);

                const auto& population = cell_report.families.at(family).value();
                EXPECT_EQ(population.quadrature.status, R::InversionStatus::TwoNodes);
                for (const auto residual : population.quadrature.reconstruction.relative)
                    EXPECT_LE(residual, R::qbmm_reconstruction_tolerance);

                // Recover the counts from H0,H1 on the independently known
                // support {1,4}. Evaluate spherical geometry directly from the EOS.
                const auto& phase = cell.phases[family];
                const double number = cell.phase_numbers[family].value();
                const double scaled_mass = moles(phase) / mass_scale;
                const double large = (scaled_mass - number) / 3.;
                const double small = number - large;
                EXPECT_GT(small, 0.);
                EXPECT_GT(large, 0.);
                EXPECT_NEAR(cell.qbmm_shape[family]->H2, small + 16. * large, 2e-10 * scaled_mass);
                EXPECT_NEAR(cell.qbmm_shape[family]->H3, small + 64. * large, 2e-10 * scaled_mass);
                const double diameter = std::cbrt(
                    6. / std::numbers::pi * mass_scale * gas_constant * phase.temperature / phase.absolute_pressure);
                const double larger_diameter = diameter * std::cbrt(4.);
                const double m2 = small * diameter * diameter + large * larger_diameter * larger_diameter;
                const double m3 = 6. / std::numbers::pi * phase_volume(phase, family);
                const auto& geometry = population.geometry.value();
                EXPECT_NEAR(geometry.area_density * volume, std::numbers::pi * m2, 2e-10 * m2);
                EXPECT_NEAR(geometry.d32.value(), m3 / m2, 2e-10 * diameter);
                local[Diameter2] += m2;
                local[Diameter3] += m3;
            }

            // Independently integrate the accepted owner-oriented carrier flux
            // across the middle plane, using signed rates and physical areas.
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const auto id = static_cast<LO>(f);
                const auto normal = mesh.face_normal(id);
                const auto center = mesh.face_centroid(id);
                if (normal.z == 0. || std::abs(center.z - .5 * height) > 1e-14)
                    continue;
                const double radius = std::hypot(center.x, center.y) / scale;
                const double rate = normal.z * vessel.flow->volume_flux().owned_value(id);
                const double area = mesh.face_area(id);
                if (radius < .5 * (inner + outer))
                {
                    local[InnerRate] += rate;
                    local[InnerArea] += area;
                }
                else
                {
                    local[OuterRate] += rate;
                    local[OuterArea] += area;
                }
            }
        });
    const auto global = vessel.sum(local);
    const double volume = global[Volume];
    EXPECT_NEAR(volume, vessel.volume(), 2e-13 * vessel.volume());
    EXPECT_GT(global[InnerArea], 0.);
    EXPECT_GT(global[OuterArea], 0.);
    EXPECT_NEAR(global[InnerArea] + global[OuterArea], volume / height, 2e-13 * volume);
    return {global[InnerRate] / global[InnerArea], global[OuterRate] / global[OuterArea], global[Circulation] / volume,
        global[TemperatureDipole] / volume, global[PressureHead] / volume, global[PressureDipole] / volume,
        global[GasVolume] / volume, std::numbers::pi * global[Diameter2], global[Diameter3] / global[Diameter2]};
}

Observables run(Geometry geometry, unsigned n, double dt, double acceleration, double heating, const std::string& name)
{
    SCOPED_TRACE(name);
    Vessel vessel(geometry, n, acceleration, heating);
    Controller controller(vessel.flow, vessel.controller_options());
    const auto comm = vessel.mesh->owned_cell_map()->getComm();
    const int ranks = comm->getSize();
    if (const char* expected = std::getenv("SIMPLEFLUID_Q8C_EXPECTED_RANKS"))
        EXPECT_EQ(std::to_string(ranks), expected);
    testing::Test::RecordProperty("mpi_ranks", ranks);
    const auto initial = observe(vessel, controller);
    EXPECT_EQ(initial.inner_upflow, 0.);
    EXPECT_EQ(initial.outer_upflow, 0.);
    EXPECT_EQ(initial.circulation, 0.);
    EXPECT_NEAR(initial.gas_fraction, gas_fraction, 2e-14);
    Metrics metrics;
    const int steps = std::lround(horizon / dt);
    for (int step = 0; step < steps; ++step)
    {
        SCOPED_TRACE(step);
        try
        {
            check_step(vessel, controller, dt, metrics);
        }
        catch (const Flow::SourceCorrectorExhausted& error)
        {
            if (comm->getRank() == 0)
            {
                std::ostringstream trace;
                trace << std::setprecision(17);
                for (const auto& eval : error.gross_residual_evaluations())
                {
                    trace << "Q8c " << name << " rejected time=" << vessel.flow->state().time
                          << " evaluation=" << eval.evaluation << " gross=" << eval.normalized_residual
                          << " flux=" << eval.normalized_flux_change;
                    if (eval.winner)
                    {
                        const auto& w = *eval.winner;
                        trace << " stage=" << static_cast<int>(w.channel_metadata.stage)
                              << " stage_index=" << w.channel_metadata.stage_index << " channel=" << w.channel
                              << " cell=" << w.geometry_cell << " face=" << w.geometry_face.value_or(-1)
                              << " Fourier_stage=" << w.fourier_stage.value_or(-1)
                              << " current_direction=" << w.current_direction.value_or(0)
                              << " previous_direction=" << w.previous_direction.value_or(0)
                              << " current=" << w.current_packet << " previous=" << w.previous_packet
                              << " scale=" << w.scale << " tolerance=" << w.relative_tolerance
                              << " current_decrease=" << w.current_decrease
                              << " current_increase=" << w.current_increase
                              << " previous_decrease=" << w.previous_decrease
                              << " previous_increase=" << w.previous_increase;
                    }
                    trace << '\n';
                }
                std::cout << trace.str();
            }
            throw;
        }
        catch (const std::exception& error)
        {
            std::ostringstream message;
            message << "Q8c " << name << " n=" << n << " dt=" << dt << " acceleration=" << acceleration
                    << " heating=" << heating << " accepted time=" << vessel.flow->state().time << ": " << error.what();
            throw std::runtime_error(message.str());
        }
    }
    EXPECT_EQ(vessel.flow->state().time, horizon);
    EXPECT_GT(metrics.fourier_exchange, 0.);
    EXPECT_EQ(metrics.maximum_pair_defect, 0.);
    EXPECT_GT(metrics.maximum_linear_cfl, 0.);
    const double expected_wall_heat = heating * wall_power_per_volume * vessel.volume() * horizon;
    EXPECT_NEAR(metrics.wall_joules[0], expected_wall_heat, 2e-12 * std::abs(expected_wall_heat));
    EXPECT_NEAR(metrics.wall_joules[1], -expected_wall_heat, 2e-12 * std::abs(expected_wall_heat));
    const auto endpoint = observe(vessel, controller);
    EXPECT_NEAR(endpoint.gas_fraction, gas_fraction, 2e-9);
    EXPECT_GT(endpoint.area, 0.);
    EXPECT_GT(endpoint.d32, 0.);
    if (acceleration != 0.)
    {
        EXPECT_GT(heating * endpoint.inner_upflow, 1e-7);
        EXPECT_LT(heating * endpoint.outer_upflow, -1e-7);
        EXPECT_GT(heating * endpoint.circulation, 1e-7);
        EXPECT_LT(endpoint.pressure_head, 0.);
        EXPECT_NEAR(endpoint.pressure_head, initial.pressure_head, .15 * std::abs(initial.pressure_head));
        EXPECT_GT(heating * endpoint.pressure_dipole, 1e-7);
    }
    EXPECT_GT(heating * endpoint.thermal_dipole, 1e-3);

    const std::array values{std::pair{"radial_and_axial_cells", double(n)}, std::pair{"dt_s", dt},
        std::pair{"horizon_s", horizon}, std::pair{"inner_upflow_m_per_s", endpoint.inner_upflow},
        std::pair{"outer_upflow_m_per_s", endpoint.outer_upflow},
        std::pair{"circulation_m2_per_s", endpoint.circulation},
        std::pair{"thermal_dipole_K_m", endpoint.thermal_dipole},
        std::pair{"pressure_head_Pa_m", endpoint.pressure_head},
        std::pair{"pressure_dipole_Pa_m2", endpoint.pressure_dipole}, std::pair{"gas_fraction", endpoint.gas_fraction},
        std::pair{"bubble_area_m2", endpoint.area}, std::pair{"d32_m", endpoint.d32},
        std::pair{"inner_wall_J", metrics.wall_joules[0]}, std::pair{"outer_wall_J", metrics.wall_joules[1]},
        std::pair{"fourier_exchange_J", metrics.fourier_exchange},
        std::pair{"fourier_pair_defect_J", metrics.maximum_pair_defect},
        std::pair{"maximum_frozen_fourier_discrepancy_J", metrics.maximum_frozen_fourier_discrepancy},
        std::pair{"maximum_recovered_fourier_discrepancy_J", metrics.maximum_recovered_fourier_discrepancy},
        std::pair{"maximum_recovered_fourier_discrepancy_temperature_ulps",
            metrics.maximum_recovered_fourier_discrepancy_ulps},
        std::pair{
            "maximum_relative_recovered_fourier_discrepancy", metrics.maximum_relative_recovered_fourier_discrepancy},
        std::pair{"pressure_work_J", metrics.pressure_work},
        std::pair{"maximum_momentum_defect_kg_m_per_s", metrics.maximum_momentum_defect},
        std::pair{"maximum_occupancy_error", metrics.maximum_occupancy},
        std::pair{"maximum_continuity_residual_per_s", metrics.maximum_continuity},
        std::pair{"maximum_normalized_solver_residual", metrics.maximum_solver_residual},
        std::pair{"maximum_linear_cfl", metrics.maximum_linear_cfl},
        std::pair{"maximum_corrector_evaluations", double(metrics.maximum_evaluations)}};
    std::ostringstream summary;
    summary << std::setprecision(17) << "Q8c " << name << " ranks=" << ranks;
    for (const auto& [key, value] : values)
    {
        summary << ' ' << key << '=' << value;
        testing::Test::RecordProperty(name + '_' + key, value);
    }
    if (comm->getRank() == 0)
        std::cout << summary.str() << '\n';
    return endpoint;
}

void circulation_cases(Geometry geometry)
{
    // Each control uses the same physical vessel and wall powers, with a common
    // horizon. Compare spatial refinement at fixed dt, and dt refinement on
    // the fine mesh while leaving the polygonal wall geometry fixed.
    const auto coarse = run(geometry, 2, timestep, gravity, 1., "coarse");
    const auto fine = run(geometry, 4, timestep, gravity, 1., "fine");
    const auto short_step = run(geometry, 4, timestep / 2., gravity, 1., "short_step");
    const auto reverse = run(geometry, 2, timestep, gravity, -1., "reversed");
    const auto unforced = run(geometry, 2, timestep, 0., 1., "gravity_off");

    const auto relative_change = [](const std::string& name, double a, double b)
    {
        const double delta = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
        testing::Test::RecordProperty(name, delta);
        return delta;
    };
    const auto sensitivity = [&](const std::string& name, double a, double b, double limit)
    { EXPECT_LE(relative_change(name, a, b), limit) << name; };
    sensitivity("spatial_circulation_change", coarse.circulation, fine.circulation, .65);
    sensitivity("spatial_thermal_dipole_change", coarse.thermal_dipole, fine.thermal_dipole, .50);
    sensitivity("spatial_pressure_dipole_change", coarse.pressure_dipole, fine.pressure_dipole, .75);
    sensitivity("spatial_area_change", coarse.area, fine.area, .02);
    sensitivity("spatial_d32_change", coarse.d32, fine.d32, .02);
    constexpr double circulation_tolerance = .15;
    const double coarse_time_change =
        relative_change("temporal_circulation_change", fine.circulation, short_step.circulation);
    testing::Test::RecordProperty(
        "temporal_circulation_coarse_pair_within_tolerance", coarse_time_change <= circulation_tolerance);
    if (geometry == Geometry::AnnularSector)
        EXPECT_LE(coarse_time_change, circulation_tolerance);
    sensitivity("temporal_thermal_dipole_change", fine.thermal_dipole, short_step.thermal_dipole, .10);
    sensitivity("temporal_pressure_dipole_change", fine.pressure_dipole, short_step.pressure_dipole, .20);
    sensitivity("temporal_area_change", fine.area, short_step.area, .01);
    sensitivity("temporal_d32_change", fine.d32, short_step.d32, .01);
    // Reflection across mid-height removes vertical motion on the center
    // plane when gravity is off. Heating still generates a radial flow.
    EXPECT_LT(std::abs(unforced.inner_upflow), .05 * std::abs(coarse.inner_upflow));
    EXPECT_LT(std::abs(unforced.outer_upflow), .05 * std::abs(coarse.outer_upflow));
    EXPECT_LT(std::abs(unforced.circulation), .05 * std::abs(coarse.circulation));
    EXPECT_LT(std::abs(unforced.pressure_dipole), .05 * std::abs(coarse.pressure_dipole));
    EXPECT_NEAR(unforced.pressure_head, 0., 1e-7);
    sensitivity("reversed_thermal_dipole_change", coarse.thermal_dipole, -reverse.thermal_dipole, .10);
    sensitivity("gravity_off_thermal_dipole_change", coarse.thermal_dipole, unforced.thermal_dipole, .10);

    if (geometry == Geometry::PolygonalAnnulus)
    {
        // The first temporal pair is underresolved for polygonal circulation.
        // Retain both points and verify that the finer difference decreases.
        const auto quarter_step = run(geometry, 4, timestep / 4., gravity, 1., "quarter_step");
        EXPECT_LT(std::abs(quarter_step.circulation - short_step.circulation),
            std::abs(short_step.circulation - fine.circulation));
        sensitivity("temporal_finest_circulation_change", short_step.circulation, quarter_step.circulation,
            circulation_tolerance);
        sensitivity(
            "temporal_finest_thermal_dipole_change", short_step.thermal_dipole, quarter_step.thermal_dipole, .10);
        sensitivity(
            "temporal_finest_pressure_dipole_change", short_step.pressure_dipole, quarter_step.pressure_dipole, .20);
        sensitivity("temporal_finest_area_change", short_step.area, quarter_step.area, .01);
        sensitivity("temporal_finest_d32_change", short_step.d32, quarter_step.d32, .01);
    }
}

TEST(QbmmCylindricalKineticHeat, HeatedClosedVessels)
{
    for (const auto kind : Cylinder::geometries)
    {
        const std::string name = Cylinder::names[static_cast<size_t>(kind)];
        SCOPED_TRACE(name);
        Vessel vessel(kind, 2, gravity, 1., true);
        Controller controller(vessel.flow, vessel.controller_options());
        Metrics metrics;
        for (int step = 0; step < 2; ++step)
            check_step(vessel, controller, timestep, metrics);
        EXPECT_GT(metrics.fourier_exchange, 0.);
        EXPECT_EQ(metrics.maximum_pair_defect, 0.);
        EXPECT_GT(std::abs(metrics.pressure_work), 0.);
        if (kind == Geometry::NativeAnnulus)
        {
            EXPECT_GT(metrics.seam_fourier_exchange, 1e-6);
            if (vessel.mesh->owned_cell_map()->getComm()->getSize() > 1)
                EXPECT_NEAR(metrics.partition_seam_fourier_exchange, metrics.seam_fourier_exchange, 1e-12);
        }
        EXPECT_GT(metrics.wall_joules[0], 0.);
        EXPECT_LT(metrics.wall_joules[1], 0.);
        const std::array values{std::pair{"hot_wall_J", metrics.wall_joules[0]},
            std::pair{"cold_wall_J", metrics.wall_joules[1]}, std::pair{"Fourier_exchange_J", metrics.fourier_exchange},
            std::pair{"maximum_Fourier_pair_defect_J", metrics.maximum_pair_defect},
            std::pair{"seam_Fourier_exchange_J", metrics.seam_fourier_exchange},
            std::pair{"partition_seam_Fourier_exchange_J", metrics.partition_seam_fourier_exchange},
            std::pair{"pressure_work_J", metrics.pressure_work},
            std::pair{"maximum_occupancy", metrics.maximum_occupancy},
            std::pair{"maximum_continuity", metrics.maximum_continuity},
            std::pair{"maximum_residual", metrics.maximum_solver_residual},
            std::pair{"maximum_momentum_defect_kg_m_s", metrics.maximum_momentum_defect}};
        for (const auto& [label, value] : values)
        {
            std::ostringstream formatted;
            formatted << std::setprecision(17) << value;
            testing::Test::RecordProperty(name + "_" + label, formatted.str());
            if (vessel.mesh->owned_cell_map()->getComm()->getRank() == 0)
                std::cout << "Q8d " << name << '_' << label << '=' << formatted.str() << '\n';
        }
    }
}

TEST(QbmmLinearGravityCirculation, NativeAnnularSector)
{
    circulation_cases(Geometry::AnnularSector);
}

TEST(QbmmLinearGravityCirculation, PolygonalAnnulusExtrusion)
{
    circulation_cases(Geometry::PolygonalAnnulus);
}
} // namespace
