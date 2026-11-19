/** Synthetic release gate for common-pressure, common-temperature, fixed-grid, zero-slip QBMM. */
#include "ConditionedThermalTestSupport.hh"
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace Thermal = Testing::ConditionedTransfer;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using Phase = R::SpeciesPhase;
using LO = Flow::Pack::local_ordinal_type;
using Operations = std::vector<R::QbmmEventOperation>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double scale = 1. / 1024; // kg per bubble at unit coordinate.
constexpr double liquid_u0 = 2 * Thermal::cv / Thermal::gas_constant * Thermal::p_old;
constexpr double reaction_k = 4, reaction_heat = 8192, heat_power = 512;
constexpr double dt = 1. / 64; // All restart and refinement times are exactly representable.
constexpr std::array<double, 3> molar_mass{1, .028, .056};
constexpr double initial_dimer = .125; // mol/m3 of bulk volume.
enum class Law
{
    Moving,
    Empty,
    Disappear
};

R::QbmmMaterialTransfer solvent(double scaled_mass)
{
    return {R::QbmmPhaseExchange{Phase::Liquid}, {scale * scaled_mass, 0, 0}};
}

struct ChemistryFixtureOptions
{
    double dimer_molar_mass = molar_mass[2], dimer_atom_count = 2;
    double dimer_bulk_concentration = initial_dimer;
    double conservation_relative_tolerance = 2e-10;
};

struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    std::shared_ptr<const R::InterphaseConfiguration> slots;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<Flow> flow;
    bool reject = false;
    int kinetic_calls = 0, partial_calls = 0;

    explicit Fixture(bool circulating = true, ChemistryFixtureOptions chemistry = {})
    {
        const ArrReal cuts = circulating ? ArrReal{0, .5, 1} : ArrReal{0, 1};
        Flow::Mesh::DistributionOptions partition;
        partition.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{cuts, ArrReal{0, 1}, cuts}), partition);
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {44, "solvent and vapor", 1, {{"S", 1}}, {Phase::Liquid, Phase::Dispersed}},
            {709, "monomer", .028, {{"G", 1}}, {Phase::Dissolved, Phase::Resolved}},
            {710, "dimer", chemistry.dimer_molar_mass, {{"G", chemistry.dimer_atom_count}}, {Phase::Resolved}}});
        const R::ThermodynamicBounds bounds{250, 1200, 50000, 1000000};
        constexpr auto convention = R::EnergyConvention::SensibleEnergy;
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, Thermal::p_old, 1, 1000, 0, 0, liquid_u0, 0, 0}; },
            convention);
        gas = R::make_ideal_gas_mixture(catalog,
            std::vector<R::IdealGasCaloricSpecies>(3, {Thermal::cv, Thermal::T0, Thermal::cv * Thermal::T0}), bounds,
            convention);
        mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{{19, "background gas dimer dissociation",
                {{710, Phase::Resolved, -1, {}}, {709, Phase::Resolved, 2, {}}}, reaction_heat}},
            convention);
        R::InterphaseConfiguration config;
        config.metadata = {"bidirectional solvent exchange", 1, catalog->fingerprint()};
        config.stages = {{"family material", R::InterphaseStageKind::Material, {{44, 0, 1}, {44, 1, 0}}, {}}};
        auto gas_partials = std::make_shared<const R::PartialMolarEnthalpyLaw>(
            R::PartialMolarEnthalpyLaw{{"ideal vapor enthalpy", 1, catalog->fingerprint()}, bounds,
                [this](double t, double, std::span<const double>)
                {
                    ++partial_calls;
                    return R::PartialMolarEnthalpyPoint{std::vector<double>(3, Thermal::cp * t),
                        std::vector<double>(3, Thermal::cp), std::vector<double>(3, 0)};
                }});
        config.phase_enthalpy = {Thermal::liquid_partials(catalog, liquid_u0, partial_calls), gas_partials, nullptr};
        slots = std::make_shared<const R::InterphaseConfiguration>(std::move(config));
        Flow::Options controls;
        controls.minimum_pressure = bounds.minimum_pressure;
        controls.maximum_pressure = bounds.maximum_pressure;
        controls.conservation_relative_tolerance = chemistry.conservation_relative_tolerance;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solution"}, {Flow::PhaseRole::DispersedGas, "vapor"},
            {Flow::PhaseRole::BulkGas, "background"}};
        auto family = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "vapor",
            controls.phase_roles[1], scale, std::vector<R::QbmmMassFraction>{{44, 1}, {709, 0}, {710, 0}});
        controls.qbmm_families = {nullptr, family, nullptr};
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas}, controls);
        Flow::State state;
        state.thermodynamic_pressure = Thermal::p_old;
        state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*family), std::nullopt};
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            auto& cell = state.cells[c];
            const auto center = mesh->cell_centroid(static_cast<LO>(c));
            const double v = mesh->cell_volume(static_cast<LO>(c));
            // Two and three atoms per m3 at masses x* and 4x*, with vacuum in the right column.
            const double factor = !circulating || center.x < .5 ? v : 0;
            const double n = 14 * scale * factor, dimer = chemistry.dimer_bulk_concentration * v, liq = .375 * v;
            const double background = (v - liq) * Thermal::p_old / (Thermal::gas_constant * Thermal::T0) - n;
            cell.phases = {{liquid, {liq, 0, 0}, Thermal::p_old, liq * liquid_u0, 300},
                {gas, {n, 0, 0}, Thermal::p_old, n * Thermal::cv * 300, n ? 300. : 0.},
                {gas, {0, background - dimer, dimer}, Thermal::p_old, background * Thermal::cv * 300, 300}};
            cell.phase_numbers = {std::nullopt, 5 * factor, std::nullopt};
            cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{50 * factor, 194 * factor}, std::nullopt};
            if (circulating)
                cell.velocity = Flow::Vector{.5 - center.z, 0, center.x < .5 ? -.25 : .25};
        }
        flow->initialize(std::move(state));
    }

    double sum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    }

    R::ReactingMixtureOptions options(Law law = Law::Moving, double chemistry_step = dt)
    {
        R::ReactingMixtureOptions opts;
        opts.mechanism = mechanism;
        opts.phases = {{Phase::Liquid, {}, {Phase::Liquid, Phase::Dissolved, Phase::Dissolved}},
            {Phase::Dispersed, "vapor", {}}, {Phase::Resolved, {}, {}}};
        opts.source_identity = "first-order dissociation and independent heater";
        opts.integration.maximumSubstep = chemistry_step;
        opts.source = [this](const auto&, double)
        {
            R::ReactionRateProvider rate;
            rate.identity = "dimer decay";
            rate.schemaRevision = catalog->revision();
            rate.schemaFingerprint = catalog->fingerprint();
            rate.validityDescription = "finite nonnegative background gas dimer";
            rate.parameterUnits = "mol/(m3 s)";
            rate.validate = [](const auto&) {};
            rate.evaluate = [](const R::ReactionContext& in)
            {
                const double n = in.state.amount(in.catalog, 710, Phase::Resolved);
                return std::vector<R::ReactionRate>{
                    {19, reaction_k * n / in.environment.bulkVolume, R::SourceVolumeBasis::BulkVolume}};
            };
            R::ReactionSourceInput input;
            input.rates = std::move(rate);
            input.heat.rate = R::VolumetricHeatRate{heat_power, R::SourceVolumeBasis::BulkVolume};
            return input;
        };
        opts.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{{"constructive", 1, catalog->fingerprint()}, slots,
                [this, law](const R::MixtureQbmmContext& in) -> Operations
                {
                    ++kinetic_calls;
                    const auto& q = in.quadrature;
                    if (reject && mesh->cell_centroid(static_cast<LO>(in.owned_cell)).x <= .5)
                        return {R::QbmmCoalescence{0, 0, 1e6}};
                    if (law == Law::Empty)
                        return {};
                    if (law == Law::Disappear)
                    {
                        Operations death;
                        for (unsigned i = 0; i < q.active_atoms; ++i)
                            death.emplace_back(R::QbmmDisappearance{
                                i, q.atoms[i].count, solvent(q.atoms[i].count * q.atoms[i].mass_over_scale)});
                        return death;
                    }
                    const double v = mesh->cell_volume(static_cast<LO>(in.owned_cell));
                    // Left-endpoint integration intentionally exposes a known first-order count error.
                    const double increment = v * in.time_step * (1 + 4 * in.accepted_time);
                    if (!q.active_atoms)
                        return {R::QbmmBirth{{{{increment, 1}, solvent(increment)}}}};
                    const unsigned last = q.active_atoms - 1;
                    const auto [na, x] = q.atoms[0];
                    const auto [nb, y] = q.atoms[last];
                    const double a = in.time_step * std::min(na, nb) / 4;
                    const double born = increment + a;
                    return {R::QbmmBirth{{{{born, 1}, solvent(born)}}}, R::QbmmResize{0, a, 2 * x, solvent(a * x)},
                        R::QbmmResize{last, a, y / 2, solvent(a * y / 2)}, R::QbmmDisappearance{0, a, solvent(a * x)},
                        R::QbmmCoalescence{0, last, a}, R::QbmmBreakup{last, a, {{2, y / 2}}}};
                }});
        return opts;
    }
};

double phase_mass(const R::PhaseThermalState& phase)
{
    return std::inner_product(phase.moles.begin(), phase.moles.end(), molar_mass.begin(), 0.);
}

double phase_volume(const R::PhaseThermalState& phase, size_t p)
{
    if (p == 0)
        return phase_mass(phase); // Constant unit liquid density.
    return std::accumulate(phase.moles.begin(), phase.moles.end(), 0.) * Thermal::gas_constant * phase.temperature /
           phase.absolute_pressure;
}

struct Inventory
{
    std::array<double, 3> species{};
    double mass = 0, elements_g = 0, elements_s = 0, energy = 0, count = 0;
};

Inventory inventory(const Fixture& f, const Flow::State& state)
{
    Inventory total;
    for (const auto& cell : state.cells)
    {
        total.count += *cell.phase_numbers[1];
        for (const auto& phase : cell.phases)
        {
            total.energy += phase.internal_energy;
            for (size_t i = 0; i < 3; ++i)
                total.species[i] += phase.moles[i];
        }
    }
    total.count = f.sum(total.count);
    total.energy = f.sum(total.energy);
    for (auto& n : total.species)
        n = f.sum(n);
    total.mass = std::inner_product(total.species.begin(), total.species.end(), molar_mass.begin(), 0.);
    total.elements_s = total.species[0];
    total.elements_g = total.species[1] + 2 * total.species[2];
    return total;
}

void near(double actual, double expected, double relative = 2e-10)
{
    EXPECT_NEAR(actual, expected, relative * std::max(1., std::abs(expected)));
}

// Independent two-node oracle: the polynomial s² - b*s + d is orthogonal to 1 and s.
// This synthetic case uses separated positive nodes, away from the kernel's rank thresholds.
std::array<R::QbmmAtom, 2> roots(const R::QbmmMomentBlock& m)
{
    const auto [n, s, t, u] = m;
    const double det = n * t - s * s, b = (n * u - s * t) / det, d = (s * u - t * t) / det;
    const double hi = .5 * (b + std::sqrt(b * b - 4 * d)), lo = d / hi;
    const double nhi = (s - n * lo) / (hi - lo);
    return {R::QbmmAtom{n - nhi, lo}, R::QbmmAtom{nhi, hi}};
}

void check_report(const Fixture& f, const Flow::State& state, const Flow::QbmmReport& report, Flow::QbmmStateKind kind)
{
    EXPECT_EQ(report.provenance.kind, kind);
    EXPECT_EQ(report.provenance.time, state.time);
    EXPECT_EQ(report.provenance.revision, state.revision);
    EXPECT_EQ(report.provenance.thermodynamic_pressure_pa, state.thermodynamic_pressure);
    EXPECT_EQ(report.provenance.geometry_epoch, f.mesh->geometry_epoch());
    EXPECT_EQ(report.provenance.catalog_fingerprint, f.catalog->fingerprint());
    EXPECT_EQ(report.species_ids, (std::vector<R::SpeciesId>{44, 709, 710}));
    double vacuum = 0, occupied = 0;
    std::array<double, 2> minima{1, 1};
    collective_detail::collective_local_validation(*f.mesh, "Release gate cell report layout",
        [&]
        {
            EXPECT_EQ(report.cells.size(), state.cells.size());
            for (size_t c = 0; c < std::min(report.cells.size(), state.cells.size()); ++c)
            {
                const auto& row = report.cells[c];
                const auto& cell = state.cells[c];
                const auto& phase = cell.phases[1];
                const auto& family = row.families.at(1).value();
                const double v = f.mesh->cell_volume(static_cast<LO>(c));
                const double n = phase.moles[0], vol = phase_volume(phase, 1);
                EXPECT_EQ(row.geometry_cell_id, f.mesh->cell_geometry_global_id(static_cast<LO>(c)));
                EXPECT_EQ(row.bulk_volume_m3, v);
                EXPECT_EQ(row.temperature_K, cell.phases[0].temperature);
                EXPECT_EQ(row.mechanical_pressure_pa, cell.mechanical_pressure);
                near(row.gas_volume_fraction, (vol + phase_volume(cell.phases[2], 2)) / v);
                EXPECT_FALSE(row.families[0]);
                EXPECT_FALSE(row.families[2]);
                near(row.mixture_density_kg_per_m3, (phase_mass(cell.phases[0]) + n + phase_mass(cell.phases[2])) / v);
                const R::QbmmMomentBlock m{
                    *cell.phase_numbers[1], n / scale, cell.qbmm_shape[1]->H2, cell.qbmm_shape[1]->H3};
                EXPECT_EQ(family.moments, m);
                EXPECT_EQ(family.species_moles, phase.moles);
                EXPECT_EQ(family.bulk_concentration_mol_per_m3, (std::vector<double>{n / v, 0, 0}));
                EXPECT_EQ(family.phase_mass_kg, n);
                EXPECT_EQ(family.phase_internal_energy_j, phase.internal_energy);
                EXPECT_EQ(family.absolute_pressure_pa, phase.absolute_pressure);
                near(family.bulk_mass_density_kg_per_m3, n / v);
                near(family.phase_volume_m3, vol);
                near(family.phase_fraction, vol / v);
                for (unsigned k = 0; k < 4; ++k)
                {
                    const double expected = m[k] * std::pow(scale, k) / v;
                    EXPECT_NEAR(family.mass_moment_density_si[k], expected, 1e-13 * expected);
                }
                EXPECT_FALSE(family.closure_validated);
                const auto& geometry = family.geometry.value();
                if (n == 0)
                {
                    ++vacuum;
                    EXPECT_FALSE(family.closure_evaluated);
                    EXPECT_EQ(family.quadrature.status, R::InversionStatus::Absent);
                    EXPECT_FALSE(family.phase_density_kg_per_m3);
                    EXPECT_FALSE(family.phase_concentration_mol_per_m3);
                    EXPECT_FALSE(family.temperature_K);
                    EXPECT_EQ(geometry.number_density, 0);
                    EXPECT_EQ(geometry.area_density, 0);
                    EXPECT_FALSE(geometry.d32);
                    EXPECT_FALSE(geometry.min_diameter);
                    EXPECT_FALSE(geometry.max_diameter);
                    continue;
                }
                ++occupied;
                const double det0 = m[0] * m[2], sq0 = m[1] * m[1], det1 = m[1] * m[3], sq1 = m[2] * m[2];
                minima[0] = std::min(minima[0], (det0 - sq0) / (det0 + sq0));
                minima[1] = std::min(minima[1], (det1 - sq1) / (det1 + sq1));
                EXPECT_TRUE(family.closure_evaluated);
                EXPECT_EQ(family.quadrature.status, R::InversionStatus::TwoNodes);
                EXPECT_EQ(geometry.active_atoms, 2u);
                EXPECT_EQ(family.temperature_K, phase.temperature);
                near(family.phase_density_kg_per_m3.value(), n / vol);
                near(family.phase_concentration_mol_per_m3->at(0), n / vol);
                const auto atoms = roots(m);
                std::array<double, 4> diameter_density{};
                std::array<double, 2> diameter{};
                for (size_t i = 0; i < 2; ++i)
                {
                    diameter[i] = std::cbrt(6 * scale * atoms[i].mass_over_scale * vol / (n * std::numbers::pi));
                    near(geometry.radius_m[i], diameter[i] / 2);
                    near(geometry.diameter_m[i], diameter[i]);
                    near(family.quadrature.atoms[i].count, atoms[i].count);
                    for (unsigned k = 0; k < 4; ++k)
                        diameter_density[k] += atoms[i].count * std::pow(diameter[i], k) / v;
                }
                for (unsigned k = 0; k < 4; ++k)
                    near(geometry.diameter_moment_density[k], diameter_density[k]);
                near(geometry.min_diameter.value(), diameter[0]);
                near(geometry.max_diameter.value(), diameter[1]);
                near(geometry.area_density, std::numbers::pi * diameter_density[2]);
                near(geometry.d32.value(), diameter_density[3] / diameter_density[2]);
            }
        });
    const auto& global = report.families.at(1).value();
    if (f.sum(occupied))
    {
        std::array<double, 2> global_minima{};
        Teuchos::reduceAll(
            *f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 2, minima.data(), global_minima.data());
        near(global.minimum_hankel_margin.value(), global_minima[0]);
        near(global.minimum_shifted_hankel_margin.value(), global_minima[1]);
    }
    else
    {
        EXPECT_FALSE(global.minimum_hankel_margin);
        EXPECT_FALSE(global.minimum_shifted_hankel_margin);
    }
    for (long double residual : global.maximum_reconstruction.relative)
        EXPECT_LE(residual, R::qbmm_reconstruction_tolerance);
    EXPECT_EQ(global.population, "vapor");
    EXPECT_EQ(global.stamp, *state.qbmm_families[1]);
    EXPECT_EQ(global.inversion_counts.at(static_cast<size_t>(R::InversionStatus::Absent)), f.sum(vacuum));
    EXPECT_EQ(global.inversion_counts.at(static_cast<size_t>(R::InversionStatus::TwoNodes)), f.sum(occupied));
}

void check_interval(const Fixture& f, const Coupler::Trial& trial, const Flow::QbmmReport& report, double h)
{
    const auto& before = f.flow->state();
    const auto& snap = trial.transport_snapshot();
    const auto& diag = trial.diagnostics();
    const auto& interval = *report.interval;
    EXPECT_EQ(report.receipt_history, Flow::QbmmReceiptHistory::Available);
    EXPECT_EQ(interval.transported_from_revision, before.revision);
    EXPECT_EQ(interval.transported_from_generation, snap.accepted_generation);
    EXPECT_EQ(interval.transported_from_time, before.time);
    EXPECT_EQ(interval.time_step, h);
    EXPECT_EQ(interval.old_geometry_epoch, snap.old_geometry_epoch);
    EXPECT_EQ(interval.new_geometry_epoch, snap.new_geometry_epoch);
    EXPECT_EQ(interval.source_stage, snap.stage);
    EXPECT_EQ(interval.preview_id, report.provenance.preview_id);
    EXPECT_EQ(interval.local_gross_events.size(), before.cells.size());
    ASSERT_TRUE(report.convergence);
    EXPECT_GE(report.convergence->source_evaluations, 2);
    EXPECT_LE(report.convergence->maximum_event_residual, 1);
    EXPECT_LE(report.convergence->maximum_shape_transport_residual.value(), 1);
    EXPECT_LE(report.convergence->maximum_shape_source_residual.value(), 1);
    EXPECT_LE(diag.flow.maximum_conditioned_thermal_residual, 1);

    double work = 0, ingressed = 0, crossing = 0;
    for (size_t face = 0; face < trial.owned_phase_transport().size(); ++face)
    {
        const auto id = static_cast<LO>(face);
        const auto& flux = trial.owned_phase_transport()[face].qbmm_flux[1].value();
        if (f.mesh->is_geometry_exterior_face(id))
        {
            EXPECT_EQ(flux.population_number_flux, 0);
            EXPECT_EQ(flux.species_molar_flux[0], 0);
            EXPECT_EQ(flux.internal_energy_flux, 0);
            EXPECT_EQ(flux.shape_flux, R::QbmmShapeState{});
        }
        else if (static_cast<size_t>(f.mesh->owner_cell(id)) >= f.mesh->num_owned_cells() ||
                 static_cast<size_t>(f.mesh->neighbor_cell(id)) >= f.mesh->num_owned_cells())
            crossing += std::abs(flux.population_number_flux);
    }
    if (f.mesh->owned_cell_map()->getComm()->getSize() > 1)
        EXPECT_GT(f.sum(crossing), 1e-6);
    collective_detail::collective_local_validation(*f.mesh, "Release gate source and thermal receipt layout",
        [&]
        {
            for (size_t c = 0; c < before.cells.size(); ++c)
            {
                const double v = f.mesh->cell_volume(static_cast<LO>(c));
                const auto& cell = trial.state().cells[c];
                const auto& transported = snap.cells[c].phases[1];
                const R::QbmmMomentBlock m{*transported.population_number, transported.moles[0] / scale,
                    transported.qbmm_shape->H2, transported.qbmm_shape->H3};
                if (*before.cells[c].phase_numbers[1] == 0 && m[0] > 0 && transported.internal_energy > 0)
                    ++ingressed;
                const auto [small, large] = roots(m);
                const double x = small.mass_over_scale, y = large.mass_over_scale;
                const double a = h * std::min(small.count, large.count) / 4, birth = a + v * h * (1 + 4 * before.time);
                const auto& receipt = interval.local_receipts.at(c).at(1).value();
                const std::array kinds{R::QbmmEventKind::Birth, R::QbmmEventKind::Growth, R::QbmmEventKind::Shrinkage,
                    R::QbmmEventKind::Disappearance, R::QbmmEventKind::Coalescence, R::QbmmEventKind::Breakup};
                EXPECT_EQ(receipt.events.size(), kinds.size());
                for (size_t i = 0; i < std::min(receipt.events.size(), kinds.size()); ++i)
                {
                    EXPECT_EQ(receipt.events[i].kind, kinds[i]);
                    EXPECT_EQ(receipt.events[i].duration, h);
                    EXPECT_EQ(receipt.events[i].provider, "constructive");
                    EXPECT_EQ(receipt.events[i].family_fingerprint, before.qbmm_families[1]->family_fingerprint);
                }
                const auto gross =
                    std::span(interval.local_gross_events.at(c)).last(Flow::QbmmSourceReceipt::gross_channels);
                near(gross[0], birth);
                for (unsigned i = 1; i <= 3; ++i)
                    near(gross[i], a); // Disappearances, coalescing pairs and splitting parents.
                for (unsigned i = 10; i < gross.size(); ++i)
                    EXPECT_EQ(gross[i], 0); // No reservoir energy or impulse.
                const auto& actual = report.cells[c].families[1]->moments;
                for (unsigned k = 0; k < 4; ++k)
                {
                    const auto power = [k](double s) { return std::pow(s, k); };
                    const double delta =
                        birth + a * (power(2 * x) - power(x) + power(y / 2) - power(y) - power(x) + power(x + y) -
                                        power(x) - power(y) + 2 * power(y / 2) - power(y));
                    near(actual[k], m[k] + delta);
                    if (k >= 2)
                    {
                        const double created = birth + a * (power(2 * x) + 3 * power(y / 2) + power(x + y));
                        const double consumed = 3 * a * (power(x) + power(y));
                        near(gross[6 + 2 * (k - 2)], created);
                        near(gross[7 + 2 * (k - 2)], consumed);
                    }
                }
                const double into = scale * (birth + a * x), out = scale * a * (x + y / 2);
                near(gross[4], into);
                near(gross[5], out);
                const auto& thermal = trial.final_thermal_evaluations()[c];
                const auto& plan = trial.interphase_receipt()->cells[c];
                near(plan.amounts[0][0], into);
                near(plan.amounts[0][1], out);
                const auto& anchor = thermal.anchors.front().temperature;
                const double liquid_h = liquid_u0 + 1000 * (anchor[0] - Thermal::T0) + thermal.absolute_pressure;
                const double gas_h = Thermal::cp * anchor[1];
                near(thermal.raw_material_enthalpy[1], into * liquid_h - out * gas_h);
                near(thermal.raw_material_enthalpy[0], out * gas_h - into * liquid_h);
                EXPECT_EQ(thermal.material_events.size(), 2u);
                for (const auto& event : thermal.material_events)
                    near(event.carried_energy, event.slot == 0 ? into * liquid_h : out * gas_h);
                double occupied = 0;
                for (size_t p = 0; p < cell.phases.size(); ++p)
                {
                    const auto& phase = cell.phases[p];
                    const auto& adv = snap.cells[c].phases[p];
                    const double volume = phase_volume(phase, p);
                    occupied += volume;
                    const double pdv = -phase.absolute_pressure * (volume - adv.advected_material_volume);
                    work += pdv;
                    EXPECT_NEAR(
                        phase.internal_energy, adv.internal_energy + thermal.applied_phase_energy[p] + pdv, 8e-5);
                }
                near(occupied, v, 2e-9);
            }
        });
    if (before.time == 0)
        EXPECT_EQ(f.sum(ingressed), 2);
    const auto start = inventory(f, before), end = inventory(f, trial.state());
    // One SSPRK2 reaction substep, separately from the macro-step integration of the bubble law.
    const double extent = start.species[2] * (reaction_k * h - .5 * std::pow(reaction_k * h, 2));
    near(end.species[0], start.species[0]);
    near(end.species[1], start.species[1] + 2 * extent);
    near(end.species[2], start.species[2] - extent);
    near(end.elements_g, start.elements_g);
    near(end.elements_s, start.elements_s);
    near(end.mass, start.mass);
    near(end.count, start.count + h * (1 + 4 * before.time));
    near(diag.reaction_extents.at(0), extent);
    near(diag.external_heat, heat_power * h);
    near(diag.chemical_heat, reaction_heat * extent);
    near(diag.net_thermal_energy, heat_power * h + reaction_heat * extent, 2e-8);
    near(diag.flow.pressure_work, f.sum(work), 6e-8);
    EXPECT_NEAR(end.energy - start.energy, heat_power * h + reaction_heat * extent + f.sum(work), 2e-4);
    const auto& balance = report.balances.value();
    EXPECT_EQ(balance.material_basis, Flow::QbmmMaterialBalanceBasis::DeclaredReservoirs);
    near(balance.mass_defect_kg, end.mass - start.mass);
    near(balance.element_defects_mol.at("S"), end.elements_s - start.elements_s);
    near(balance.element_defects_mol.at("G"), end.elements_g - start.elements_g);
    EXPECT_NEAR(balance.internal_energy_defect_j,
        end.energy - start.energy - heat_power * h - reaction_heat * extent - f.sum(work), 1e-6);
    const std::array reaction_change{0., 2 * extent, -extent};
    for (size_t s = 0; s < reaction_change.size(); ++s)
    {
        EXPECT_EQ(balance.external_species_moles.at(s), 0);
        near(balance.species_defects_mol.at(s), end.species[s] - start.species[s] - reaction_change[s]);
    }
    near(balance.pressure_work_j, f.sum(work), 6e-8);
}

struct Endpoint
{
    Flow::State state;
    std::vector<double> flux;
    explicit Endpoint(const Fixture& f) : state(f.flow->state())
    {
        for (size_t i = 0; i < f.mesh->num_owned_faces(); ++i)
            flux.push_back(f.flow->volume_flux().owned_value(static_cast<LO>(i)));
    }

    void expect_same(const Fixture& f) const
    {
        const auto& other = f.flow->state();
        EXPECT_EQ(other.time, state.time);
        EXPECT_EQ(other.revision, state.revision);
        EXPECT_EQ(other.thermodynamic_pressure, state.thermodynamic_pressure);
        EXPECT_EQ(other.qbmm_families, state.qbmm_families);
        EXPECT_EQ(other.cells.size(), state.cells.size());
        for (size_t c = 0; c < std::min(state.cells.size(), other.cells.size()); ++c)
        {
            const auto& a = state.cells[c];
            const auto& b = other.cells[c];
            EXPECT_EQ(a.phase_numbers, b.phase_numbers);
            EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
            EXPECT_EQ(a.mechanical_pressure, b.mechanical_pressure);
            for (size_t d = 0; d < 3; ++d)
                EXPECT_EQ(a.velocity.component(d), b.velocity.component(d));
            for (size_t p = 0; p < a.phases.size(); ++p)
            {
                EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
                EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
                EXPECT_EQ(a.phases[p].absolute_pressure, b.phases[p].absolute_pressure);
                EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
                for (size_t d = 0; d < 3; ++d)
                    EXPECT_EQ(a.drift_velocity[p].component(d), b.drift_velocity[p].component(d));
            }
        }
        for (size_t face = 0; face < flux.size(); ++face)
            EXPECT_EQ(flux[face], f.flow->volume_flux().owned_value(static_cast<LO>(face)));
    }
};

void checked_step(Fixture& f, Coupler& coupler)
{
    const auto before = coupler.qbmm_report();
    const auto trial = coupler.preview(dt);
    const auto candidate = coupler.qbmm_report(trial);
    check_report(f, trial.state(), candidate, Flow::QbmmStateKind::Trial);
    check_interval(f, trial, candidate, dt);
    EXPECT_EQ(coupler.qbmm_report().provenance.generation, before.provenance.generation);
    const auto calls = f.kinetic_calls;
    coupler.commit(trial);
    EXPECT_EQ(f.kinetic_calls, calls);
    const auto accepted = coupler.qbmm_report();
    check_report(f, f.flow->state(), accepted, Flow::QbmmStateKind::Accepted);
    EXPECT_EQ(accepted.interval->preview_id, candidate.interval->preview_id);
    EXPECT_EQ(accepted.interval->local_gross_events, candidate.interval->local_gross_events);
    EXPECT_EQ(accepted.provenance.origin, Flow::QbmmStateOrigin::Corrector);
    EXPECT_GT(accepted.provenance.generation, before.provenance.generation);
    EXPECT_EQ(before.provenance.time + dt, accepted.provenance.time);
}

TEST(ReactingMixtureQbmmReleaseGateTest, MovingClosedTrajectoryAndBothMidpointContinuations)
{
    Fixture f;
    Coupler controller(f.flow, f.options());
    const auto original = controller.qbmm_report();
    check_report(f, f.flow->state(), original, Flow::QbmmStateKind::Accepted);
    EXPECT_EQ(original.receipt_history, Flow::QbmmReceiptHistory::NoInterval);
    checked_step(f, controller);
    checked_step(f, controller);
    const auto middle = controller.qbmm_report();
    const auto save = controller.checkpoint();
    const auto archive = f.flow->serialize_qbmm_restart();
    Endpoint midpoint(f);
    checked_step(f, controller);
    checked_step(f, controller);
    Endpoint uninterrupted(f);
    controller.restore(save);
    midpoint.expect_same(f);
    const auto restored = controller.qbmm_report();
    EXPECT_EQ(restored.provenance.origin, Flow::QbmmStateOrigin::Checkpoint);
    EXPECT_GT(restored.provenance.generation, middle.provenance.generation);
    EXPECT_EQ(restored.interval->preview_id, middle.interval->preview_id);
    EXPECT_EQ(restored.interval->local_gross_events, middle.interval->local_gross_events);
    controller.advance(dt);
    controller.advance(dt);
    uninterrupted.expect_same(f);

    Fixture fresh;
    fresh.flow->restore_qbmm_restart(archive);
    midpoint.expect_same(fresh);
    Coupler resumed(fresh.flow, fresh.options());
    const auto restart = resumed.qbmm_report();
    check_report(fresh, fresh.flow->state(), restart, Flow::QbmmStateKind::Accepted);
    EXPECT_EQ(restart.provenance.origin, Flow::QbmmStateOrigin::PortableRestart);
    EXPECT_EQ(restart.provenance.archived_generation, middle.provenance.generation);
    EXPECT_EQ(restart.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    EXPECT_FALSE(restart.interval);
    ASSERT_TRUE(restart.convergence);
    ASSERT_TRUE(middle.convergence);
    EXPECT_EQ(restart.convergence->correctors, middle.convergence->correctors);
    EXPECT_EQ(restart.convergence->source_evaluations, middle.convergence->source_evaluations);
    EXPECT_EQ(restart.convergence->maximum_event_residual, middle.convergence->maximum_event_residual);
    EXPECT_EQ(restart.convergence->maximum_shape_source_residual, middle.convergence->maximum_shape_source_residual);
    EXPECT_EQ(
        restart.convergence->maximum_shape_transport_residual, middle.convergence->maximum_shape_transport_residual);
    ASSERT_TRUE(restart.balances);
    ASSERT_TRUE(middle.balances);
    EXPECT_EQ(restart.balances->material_basis, middle.balances->material_basis);
    EXPECT_EQ(restart.balances->mass_defect_kg, middle.balances->mass_defect_kg);
    EXPECT_EQ(restart.balances->internal_energy_defect_j, middle.balances->internal_energy_defect_j);
    EXPECT_EQ(restart.balances->external_species_moles, middle.balances->external_species_moles);
    EXPECT_EQ(restart.balances->element_defects_mol, middle.balances->element_defects_mol);
    resumed.advance(dt);
    resumed.advance(dt);
    uninterrupted.expect_same(fresh);
    EXPECT_EQ(original.provenance.time, 0);
    EXPECT_FALSE(original.interval);
    EXPECT_FALSE(original.convergence);
    EXPECT_EQ(middle.provenance.time, 2 * dt);
    EXPECT_EQ(middle.interval->transported_from_time, dt);
    EXPECT_EQ(middle.interval->source_stage, restored.interval->source_stage);
}

TEST(ReactingMixtureQbmmReleaseGateTest, TimestepAndSourceSubstepsConvergeAgainstIndependentSolutions)
{
    const double horizon = 4 * dt;
    const auto run = [&](double macro, double chemistry)
    {
        Fixture f;
        Coupler controller(f.flow, f.options(Law::Moving, chemistry));
        for (int step = 0; step < std::lround(horizon / macro); ++step)
            controller.advance(macro);
        EXPECT_EQ(f.flow->state().time, horizon);
        const auto final = inventory(f, f.flow->state());
        const double count_exact = 2.5 + horizon + 2 * horizon * horizon;
        const double dimer_exact = initial_dimer * std::exp(-reaction_k * horizon);
        return std::array{count_exact - final.count, final.species[2] - dimer_exact};
    };
    const auto coarse = run(2 * dt, 2 * dt);
    const auto fine = run(dt, dt);
    const auto subcycled = run(2 * dt, dt);
    EXPECT_GT(fine[0], 0);
    EXPECT_GT(fine[1], 0);
    // Composite left-rule count error is 2*T*h. SSPRK2 decay has second-order global error.
    near(coarse[0], 2 * horizon * (2 * dt));
    near(fine[0], 2 * horizon * dt);
    near(subcycled[0], coarse[0]);
    EXPECT_GT(coarse[1] / fine[1], 3.8);
    EXPECT_LT(coarse[1] / fine[1], 4.5);
    near(subcycled[1], fine[1]);
}

TEST(ReactingMixtureQbmmReleaseGateTest, EmptyOwnersParticipateInRejectionExtinctionAndCallbackFreeReplay)
{
    Fixture f(false);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    EXPECT_EQ(f.sum(f.mesh->num_owned_cells() == 0), comm->getSize() - 1);
    Coupler controller(f.flow, f.options(Law::Disappear));
    const auto before = controller.qbmm_report();
    const auto trial = controller.preview(dt);
    const auto candidate = controller.qbmm_report(trial);
    const auto alternative = controller.preview(2 * dt);
    const auto other_report = controller.qbmm_report(alternative);
    EXPECT_NE(other_report.provenance.preview_id, candidate.provenance.preview_id);
    EXPECT_EQ(other_report.provenance.generation, candidate.provenance.generation);
    EXPECT_EQ(candidate.interval->time_step, dt);
    if (comm->getSize() > 1)
    {
        const auto& selected = comm->getRank() == 0 ? trial : alternative;
        EXPECT_THROW((void) controller.qbmm_report(selected), std::exception);
    }
    const auto archive = f.flow->serialize_qbmm_restart();
    f.reject = true; // Only the single physical owner evaluates the overdraw.
    EXPECT_THROW((void) controller.preview(dt), std::exception);
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), archive);
    const auto retained = controller.qbmm_report();
    EXPECT_EQ(retained.provenance.generation, before.provenance.generation);
    EXPECT_FALSE(retained.interval);
    f.reject = false;
    const auto checkpoint = controller.checkpoint();
    const auto calls = std::array{f.kinetic_calls, f.partial_calls};
    controller.commit(trial);
    EXPECT_EQ((std::array{f.kinetic_calls, f.partial_calls}), calls);
    EXPECT_THROW((void) controller.qbmm_report(alternative), std::exception);
    auto absent = controller.qbmm_report();
    check_report(f, f.flow->state(), absent, Flow::QbmmStateKind::Accepted);
    EXPECT_EQ(absent.families[1]->inversion_counts[static_cast<size_t>(R::InversionStatus::Absent)], 1u);
    for (const auto& row : absent.cells)
        EXPECT_EQ(row.families[1]->moments, R::QbmmMomentBlock{});
    controller.restore(checkpoint);
    EXPECT_EQ(f.kinetic_calls, calls[0]);
    EXPECT_EQ(controller.qbmm_report().receipt_history, Flow::QbmmReceiptHistory::NoInterval);
    EXPECT_THROW(controller.commit(trial), std::exception);
    EXPECT_THROW((void) controller.qbmm_report(trial), std::exception);
    EXPECT_EQ(candidate.families[1]->inversion_counts[static_cast<size_t>(R::InversionStatus::Absent)], 1u);
    EXPECT_EQ(before.families[1]->inversion_counts[static_cast<size_t>(R::InversionStatus::TwoNodes)], 1u);
}

TEST(ReactingMixtureQbmmReleaseGateTest, NativeAndCoupledReportsTrackThermalGeometryAtFixedMassCoordinates)
{
    Fixture f(false);
    const auto native = f.flow->preview(dt);
    const auto trial_report = f.flow->qbmm_report(native);
    check_report(f, native.state(), trial_report, Flow::QbmmStateKind::Trial);
    const auto initial = f.flow->qbmm_report();
    check_report(f, f.flow->state(), initial, Flow::QbmmStateKind::Accepted);
    // Changing only the shape at fixed material, temperature and pressure changes
    // interfacial area. The physical EOS volume and phase fraction stay identical.
    Flow monodisperse(f.mesh, {f.liquid, f.gas, f.gas}, f.flow->options());
    auto same_material = f.flow->state();
    for (auto& cell : same_material.cells)
        cell.qbmm_shape[1] = R::qbmm_initialize_monodisperse(
            *f.flow->options().qbmm_families[1], *cell.phase_numbers[1], cell.phases[1].moles);
    monodisperse.initialize(std::move(same_material));
    const auto mono = monodisperse.qbmm_report();
    for (size_t c = 0; c < mono.cells.size(); ++c)
    {
        const auto& poly = *initial.cells[c].families[1];
        const auto& single = *mono.cells[c].families[1];
        EXPECT_EQ(single.quadrature.status, R::InversionStatus::RankOne);
        EXPECT_EQ(poly.phase_volume_m3, single.phase_volume_m3);
        EXPECT_EQ(poly.phase_fraction, single.phase_fraction);
        EXPECT_NE(poly.geometry->area_density, single.geometry->area_density);
    }
    EXPECT_EQ(f.kinetic_calls, 0);
    Coupler controller(f.flow, f.options(Law::Empty));
    const auto trial = controller.preview(dt);
    check_report(f, trial.state(), controller.qbmm_report(trial), Flow::QbmmStateKind::Trial);
    controller.commit(trial);
    const auto heated = controller.qbmm_report();
    check_report(f, f.flow->state(), heated, Flow::QbmmStateKind::Accepted);
    EXPECT_NE(heated.provenance.thermodynamic_pressure_pa, initial.provenance.thermodynamic_pressure_pa);
    for (size_t c = 0; c < heated.cells.size(); ++c)
    {
        const auto& old = initial.cells[c].families[1].value();
        const auto& now = heated.cells[c].families[1].value();
        EXPECT_EQ(old.moments, now.moments);
        EXPECT_GT(*now.temperature_K, *old.temperature_K);
        EXPECT_EQ(now.species_moles, old.species_moles);
        // The extra unbound gas molecules compress this family in a closed cell of fixed liquid volume.
        EXPECT_LT(now.phase_volume_m3, old.phase_volume_m3);
        EXPECT_LT(now.geometry->d32.value(), old.geometry->d32.value());
    }
}

TEST(ReactingMixtureQbmmReleaseGateTest, ActualMassAndElementDefectsRejectBeforePublishing)
{
    struct Example
    {
        double dimer_molar_mass, dimer_atom_count;
        const char* failure;
    };
    const Example examples[]{
        {molar_mass[2], 2, nullptr},
        {molar_mass[2] * (1 - 1.5e-12), 2, "QBMM physical total mass budget failed."},
        {molar_mass[2], 2 - 3e-12, "QBMM physical element budget failed."},
    };
    for (const auto& example : examples)
    {
        // The mechanism's relative tolerance is 1e-12 of the sum of reactant and product operands.
        // Consuming 16 mol of a 20 mol stock makes either allowed bias exceed the native physical gate.
        Fixture f(false, {example.dimer_molar_mass, example.dimer_atom_count, 20, 2.5e-13});
        auto options = f.options(Law::Empty);
        options.qbmm_population.reset();
        options.source = [](const auto&, double)
        {
            R::ReactionSourceInput source;
            source.integratedExtents = {{19, 16}};
            return source;
        };
        Coupler controller(f.flow, std::move(options));
        if (!example.failure)
        {
            EXPECT_NO_THROW(controller.advance(dt));
            EXPECT_EQ(f.flow->state().time, dt);
            continue;
        }
        const auto accepted = f.flow->serialize_qbmm_restart();
        try
        {
            (void) controller.preview(dt);
            ADD_FAILURE() << "Expected " << example.failure;
        }
        catch (const std::exception& error)
        {
            EXPECT_STREQ(error.what(), example.failure);
        }
        EXPECT_EQ(f.flow->serialize_qbmm_restart(), accepted);
    }
}
} // namespace
