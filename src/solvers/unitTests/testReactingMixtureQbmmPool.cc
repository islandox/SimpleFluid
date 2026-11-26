/** Coupled fixed-grid BulkGas conversion with independent extensive and caloric oracles. */
#include "QbmmRestartTestSupport.hh"
#include "equations/QbmmPopulationLawAdapters.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>
#include <numeric>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace IO = Testing::QbmmRestart;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using Phase = R::SpeciesPhase;
using Direction = R::QbmmPoolConversionDirection;
using Operations = std::vector<R::QbmmEventOperation>;
using Context = R::MixtureQbmmContext;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr R::SpeciesId A = 17, B = 31, solvent = 53;
constexpr double scale = 8., T0 = 256., gas_R = 8., dt = 1. / 64., liquid_cv = 512.;
constexpr R::ThermodynamicBounds bounds{100., 800., 4000., 500000.};

struct FixtureOptions
{
    double liquid_fraction = 1. / 16., pool_stock = 0;
    bool different_eos = false, two_families = false, circulating = false, two_cells = false, batched = true;
    bool extra_permitted = false, dry_chemistry = false, declare_liquid_law = false;
    Direction direction = Direction::Bidirectional;
    std::uint64_t law_revision = 1;
    bool legacy = false;
    std::string pool_name = "pool";
    bool passive = false;
};

struct Fixture
{
    FixtureOptions setup;
    std::shared_ptr<const R::SpeciesCatalog> catalog =
        std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {A, "monomer", 1, {{"X", 1}}, {Phase::Dissolved, Phase::Dispersed, Phase::Resolved}},
            {B, "dimer", 2, {{"X", 2}}, {Phase::Dissolved, Phase::Dispersed, Phase::Resolved}},
            {solvent, "solvent", 1, {{"S", 1}}, {Phase::Liquid, Phase::Resolved}}});
    std::shared_ptr<Flow::Mesh> mesh;
    Flow::Options controls;
    std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos;
    std::shared_ptr<Flow> flow;
    R::ReactingMixtureOptions options;
    std::shared_ptr<const R::QbmmBulkGasPoolBinding> pool;
    std::shared_ptr<R::InterphaseConfiguration> slots;
    std::unique_ptr<Coupler> controller;
    std::function<Operations(const Context&)> events;
    R::QbmmLawMetadata liquid_law{"liquid tracer", 1, R::QbmmLawKind::MassTransfer, "liquid mol/m3", "mol/cell",
        R::QbmmApplicabilityGate::RejectOutside, true,
        {{R::QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3, 0, 1000, 0}}};
    int calls = 0, partial_calls = 0, liquid_calls = 0;
    double pressure;

    static std::vector<double> moles(double scaled_mass) { return {4 * scaled_mass, 2 * scaled_mass, 0}; }
    static double gas_capacity(const std::vector<double>& n, double factor)
    {
        return factor * (16 * n[0] + 32 * n[1] + 16 * n[2]);
    }
    explicit Fixture(FixtureOptions s = {})
        : setup(s), pressure((5 + s.pool_stock + (s.passive ? 2 : 0)) * 6 * gas_R * T0 / (1 - s.liquid_fraction))
    {
        const auto cuts = s.circulating ? ArrReal{0, .5, 1} : ArrReal{0, 1};
        const auto z = s.two_cells ? ArrReal{0, .5, 1} : cuts;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{cuts, ArrReal{0, 1}, z}),
            Flow::Mesh::DistributionOptions{.allow_empty_partitions = true});
        const auto fp = catalog->fingerprint();
        const auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds, [this](auto)
            { return R::LinearCompressibleMixtureLiquid{T0, pressure, 1, liquid_cv, 0, 0, liquid_cv * T0, 0, 0}; });
        auto gas = [&](double factor)
        {
            return R::make_ideal_gas_mixture(catalog,
                {{16 * factor, T0, 16 * factor * T0}, {32 * factor, T0, 32 * factor * T0},
                    {16 * factor, T0, 16 * factor * T0}},
                bounds, R::EnergyConvention::IncludesFormationEnergy, gas_R);
        };
        controls.qbmm_model.bulk_gas_conversion = !s.legacy;
        controls.use_multi_rhs_transport = s.batched;
        controls.minimum_pressure = bounds.minimum_pressure;
        controls.maximum_pressure = bounds.maximum_pressure;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "fine"}};
        if (s.two_families)
            controls.phase_roles.push_back({Flow::PhaseRole::DispersedGas, "coarse"});
        controls.phase_roles.push_back({Flow::PhaseRole::BulkGas, s.pool_name});
        const size_t last = controls.phase_roles.size() - 1;
        if (s.passive)
            controls.phase_roles.push_back({Flow::PhaseRole::BulkGas, "passive gas"});
        const auto nph = controls.phase_roles.size();
        const std::array enabled{false, true, s.two_families, false, false};
        std::vector<R::SpeciesId> permitted{A, B};
        if (s.extra_permitted)
            permitted.push_back(solvent);
        pool = std::make_shared<const R::QbmmBulkGasPoolBinding>(
            catalog, fp, std::span(controls.phase_roles), last, permitted, s.direction, std::span(enabled).first(nph));
        if (!s.legacy)
            controls.qbmm_bulk_gas_pool = pool;
        controls.qbmm_eos_law_identities.resize(nph);
        controls.qbmm_eos_law_identities[0] = R::ThermalLawIdentity{"constant density", 1, fp};
        controls.qbmm_families.resize(nph);
        options.phases = {{Phase::Liquid, {}, {Phase::Dissolved, Phase::Dissolved, Phase::Liquid}}};
        eos = {liquid};
        slots = std::make_shared<R::InterphaseConfiguration>();
        slots->metadata = {"ideal partial molar enthalpy", 1, fp};
        slots->phase_enthalpy.resize(nph);
        slots->stages = {{"conversion", R::InterphaseStageKind::Material, {}, {}}};
        for (size_t p = 1; p < nph; ++p)
        {
            const bool is_pool = p == last;
            const double factor = is_pool && s.different_eos ? 1.5 : 1.;
            eos.push_back(gas(factor));
            slots->phase_enthalpy[p] = std::make_shared<const R::PartialMolarEnthalpyLaw>(
                R::PartialMolarEnthalpyLaw{{is_pool ? "pool ideal enthalpy" : "bubble ideal enthalpy", 1, fp}, bounds,
                    [this, factor](double t, double, std::span<const double>)
                    {
                        ++partial_calls;
                        const double ca = 16 * factor + gas_R, cb = 32 * factor + gas_R;
                        return R::PartialMolarEnthalpyPoint{{ca * t, cb * t, ca * t}, {ca, cb, ca}, {0, 0, 0}};
                    }});
            if (p >= last)
            {
                options.phases.push_back({Phase::Resolved, {}, {}});
                continue;
            }
            auto name = controls.phase_roles[p].identity;
            R::QbmmClosureVersions versions;
            versions.source_identity = "pool kinetics";
            versions.laws = {{"prescribed conversion", s.law_revision, R::QbmmLawKind::BulkGasConversion,
                "fixed SI node masses and frozen stock", "mol/cell", R::QbmmApplicabilityGate::ReportOnly, false,
                {{R::QbmmRangeVariable::BulkGasPoolVolumeFraction, 0, 1, {}}}}};
            if (s.declare_liquid_law)
                versions.laws.push_back(liquid_law);
            if (s.legacy)
                versions.laws.clear();
            controls.qbmm_families[p] =
                std::make_shared<const R::QbmmPopulationSchema>(catalog, fp, name, controls.phase_roles[p], scale,
                    std::vector<R::QbmmMassFraction>{{A, .5}, {B, .5}, {solvent, 0}}, versions);
            options.phases.push_back({Phase::Dispersed, name, {}});
            for (auto id : {A, B})
            {
                if (pool->allows(Direction::DispersedToPool))
                    slots->stages[0].material.push_back({id, p, last});
                if (pool->allows(Direction::PoolToDispersed))
                    slots->stages[0].material.push_back({id, last, p});
            }
        }
        std::vector<R::ReactionDefinition> reactions;
        if (s.dry_chemistry)
            reactions.push_back(
                {7, "resolved dimerization", {{A, Phase::Resolved, -2, {}}, {B, Phase::Resolved, 1, {}}}, 0.});
        options.mechanism =
            std::make_shared<const R::ReactionMechanism>(catalog, reactions, eos[0]->energy_convention());
        options.heat_phase = last;
        flow = std::make_shared<Flow>(mesh, eos, controls);
        Flow::State state;
        state.thermodynamic_pressure = pressure;
        state.qbmm_families.resize(nph);
        for (size_t p = 1; p < last; ++p)
            state.qbmm_families[p] = Flow::qbmm_family_stamp(*controls.qbmm_families[p]);
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            auto& cell = state.cells.emplace_back();
            const auto x = mesh->cell_centroid(c);
            const double v = mesh->cell_volume(c), wl = s.liquid_fraction * v;
            cell.phase_numbers.resize(nph);
            cell.qbmm_shape.resize(nph);
            cell.phases = {{eos[0], {0, 0, wl}, pressure, wl * liquid_cv * T0, wl > 0 ? T0 : 0.}};
            double fine = v, pool_mass = s.pool_stock * v;
            if (s.circulating)
            {
                fine = x.x < .5 ? v : 0;
                pool_mass += (v - fine) * 5;
                cell.velocity = Flow::Vector{.5 - x.z, 0, x.x < .5 ? -.25 : .25};
            }
            for (size_t p = 1; p < nph; ++p)
            {
                const double f = p == 1 ? fine : 0.;
                const auto n = moles(p > last ? 2 * v : p == last ? pool_mass : 5 * f);
                cell.phases.push_back({eos[p], n, pressure,
                    gas_capacity(n, p == last && s.different_eos ? 1.5 : 1.) * T0, n[0] > 0 ? T0 : 0.});
                if (p < last)
                {
                    cell.phase_numbers[p] = 3 * f;
                    cell.qbmm_shape[p] = R::QbmmShapeState{11 * f, 29 * f};
                }
            }
        }
        flow->initialize(std::move(state));
        events = [this](const Context& ctx) { return trajectory(ctx); };
    }
    void start()
    {
        const R::ThermalLawIdentity id{"pool kinetics", 1, catalog->fingerprint()};
        std::vector<std::shared_ptr<const R::MixtureQbmmLaw>> laws;
        if (setup.declare_liquid_law)
            laws.push_back(std::make_shared<const R::MixtureQbmmLaw>(R::MixtureQbmmLaw{liquid_law, [this](const auto&)
                {
                    ++liquid_calls;
                    return Operations{};
                }}));
        options.qbmm_population =
            std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{id, setup.legacy ? nullptr : slots,
                [this](const Context& ctx)
                {
                    ++calls;
                    return events(ctx);
                },
                laws});
        controller = std::make_unique<Coupler>(flow, options);
    }
    Operations drain(const Context& in, std::vector<double> fractions) const
    {
        return R::qbmm_pool_drain_from_fraction(in.family, *in.law_inputs, *pool, fractions);
    }
    Operations birth(const Context& in, double count, double size) const
    {
        return R::qbmm_pool_birth_from_spectrum(
            in.family, *in.law_inputs, *pool, std::array{R::QbmmPoolBirthAtom{count, size * scale}});
    }
    Operations trajectory(const Context& in) const
    {
        const size_t step = static_cast<size_t>(std::lround(in.accepted_time / dt));
        if (step == 0)
            return drain(in, {0, 1});
        if (step == 1)
            return birth(in, 2, 1.5);
        if (step == 2)
            return drain(in, std::vector<double>(in.quadrature.active_atoms, 1));
        return birth(in, 10, .5);
    }
    double sum(double local) const
    {
        double total = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &total);
        return total;
    }
    void check_ranks(bool one_cell = true) const
    {
        const auto comm = mesh->owned_cell_map()->getComm();
        if (const char* ranks = std::getenv("SIMPLEFLUID_Q9B_RANKS"))
            EXPECT_EQ(comm->getSize(), std::stoi(ranks));
        if (one_cell)
            EXPECT_EQ(sum(mesh->num_owned_cells() == 0), comm->getSize() - 1);
    }
};

void near(double actual, double expected, double tol = 3e-9)
{
    EXPECT_NEAR(actual, expected, tol * std::max(1., std::abs(expected)));
}

TEST(ReactingMixtureQbmmPool, BidirectionalSpectrumExactAbsenceAndRegeneration)
{
    Fixture f;
    f.check_ranks();
    f.start();
    const std::array<std::array<double, 4>, 4> moments{
        {{2, 2, 2, 2}, {4, 5, 6.5, 8.75}, {0, 0, 0, 0}, {10, 5, 2.5, 1.25}}};
    const std::array<std::array<double, 4>, 4> gross{
        {{1, 3, 9, 27}, {2, 3, 4.5, 6.75}, {4, 5, 6.5, 8.75}, {10, 5, 2.5, 1.25}}};
    for (size_t i = 0; i < moments.size(); ++i)
    {
        const auto trial = f.controller->preview(dt);
        const auto report = f.controller->qbmm_report(trial);
        const auto& diag = trial.diagnostics();
        ASSERT_TRUE(diag.flow.pool_conversion);
        const auto& dir = diag.flow.pool_conversion->directions[i % 2];
        near(dir.count, gross[i][0]);
        near(dir.mass_kg, gross[i][1] * scale);
        near(dir.H2, gross[i][2]);
        near(dir.H3, gross[i][3]);
        EXPECT_EQ(dir.species_moles, Fixture::moles(gross[i][1]));
        near(dir.carried_enthalpy_j, gross[i][1] * (4 * 24 + 2 * 40) * T0);
        EXPECT_EQ(diag.flow.pool_conversion->directions[1 - i % 2].count, 0);
        EXPECT_GT(diag.flow.pool_conversion->law_evaluations, 0u);
        near(diag.external_mass, 0);
        near(diag.transfer_energy, 0);
        near(diag.external_heat, 0);
        near(trial.state().thermodynamic_pressure, f.pressure);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& cell = trial.state().cells[c];
            const auto& row = report.cells[c];
            const auto& family = *row.families[1];
            EXPECT_EQ(family.pool_conversion_law_evaluated, true);
            for (size_t k = 0; k < 4; ++k)
                near(family.moments[k], moments[i][k], 2e-13);
            EXPECT_EQ(cell.phases[1].moles, Fixture::moles(moments[i][1]));
            EXPECT_EQ(cell.phases[2].moles, Fixture::moles(5 - moments[i][1]));
            EXPECT_FALSE(cell.phase_numbers[2]);
            EXPECT_FALSE(cell.qbmm_shape[2]);
            EXPECT_FALSE(row.families[2]);
            const double gas_void = 1 - f.setup.liquid_fraction;
            near(row.gas_volume_fraction, gas_void);
            near(row.dispersed_gas_volume_fraction, gas_void * moments[i][1] / 5);
            near(row.bulk_gas_pool_volume_fraction, gas_void * (5 - moments[i][1]) / 5);
            EXPECT_EQ(row.passive_gas_volume_fraction, 0);
            near(row.temperature_K.value(), T0);
            if (i == 2)
            {
                EXPECT_EQ(family.quadrature.status, R::InversionStatus::Absent);
                EXPECT_FALSE(family.temperature_K);
                EXPECT_EQ(cell.phases[1].internal_energy, 0);
            }
            if (i % 2 == 1)
            {
                EXPECT_EQ(cell.phases[2].temperature, 0);
                EXPECT_EQ(cell.phases[2].internal_energy, 0);
            }
        }
        const auto evaluations = f.calls;
        f.controller->commit(trial);
        EXPECT_EQ(f.calls, evaluations);
    }
}

TEST(ReactingMixtureQbmmPool, PressureRootReplaysEachDonorPartialAndPhaseVolumeWork)
{
    Fixture f({.different_eos = true});
    f.start();
    const auto trial = f.controller->preview(dt);
    const double liquid_capacity = f.setup.liquid_fraction * liquid_cv;
    const double initial_capacity = liquid_capacity + 5 * 128;
    const double final_capacity = liquid_capacity + 2 * 128 + 3 * 192;
    const double t = initial_capacity * T0 / final_capacity;
    const double pressure = f.pressure * t / T0;
    const double anchor =
        (initial_capacity * T0 + pressure * (1 - f.setup.liquid_fraction)) / (initial_capacity + 30 * gas_R);
    near(trial.state().thermodynamic_pressure, pressure);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& thermal = trial.final_thermal_evaluations()[c];
        const auto& before = f.flow->state().cells[c];
        const auto& after = trial.state().cells[c];
        const auto& config = *trial.interphase_receipt()->cells[c].configuration;
        double carried = 0;
        size_t positive = 0;
        for (const auto& event : thermal.material_events)
        {
            const auto& slot = config.stages[event.stage].material[event.slot];
            if (event.carried_energy == 0)
                continue;
            EXPECT_EQ(slot.donor, 1u);
            EXPECT_EQ(slot.recipient, 2u);
            double h = anchor * (slot.species == A ? 24. : 40.);
            near(event.molar_enthalpy, h);
            near(event.carried_energy, h * (slot.species == A ? 12. : 6.));
            carried += event.carried_energy;
            ++positive;
        }
        EXPECT_EQ(positive, 2u);
        const std::array old_v{f.setup.liquid_fraction, 1 - f.setup.liquid_fraction, 0.};
        const std::array new_v{f.setup.liquid_fraction, 12 * gas_R * t / pressure, 18 * gas_R * t / pressure};
        const std::array material{0., -carried, carried};
        double work = 0., redistributed = 0.;
        for (size_t p = 0; p < 3; ++p)
        {
            const double pdv = -pressure * (new_v[p] - old_v[p]);
            near(after.phases[p].temperature, t);
            near(thermal.volumes[p], new_v[p]);
            near(thermal.raw_material_enthalpy[p], material[p]);
            const double du = after.phases[p].internal_energy - before.phases[p].internal_energy;
            near(thermal.raw_equilibrium_redistribution[p], du - material[p] - pdv);
            work += pdv;
            redistributed += thermal.raw_equilibrium_redistribution[p];
        }
        EXPECT_NEAR(work, 0., 1e-8);
        EXPECT_NEAR(redistributed, 0., 1e-8);
        EXPECT_GT(std::abs(thermal.raw_equilibrium_redistribution[2]), 1.);
    }
}

TEST(ReactingMixtureQbmmPool, TypedAndJointSubstepsDrainEachActualPredecessor)
{
    std::optional<Flow::QbmmPoolDirectionDiagnostics> typed_totals;
    size_t typed_steps = 0;
    for (bool joint : {false, true})
    {
        SCOPED_TRACE(joint ? "joint provider" : "typed provider");
        Fixture f;
        f.check_ranks();
        const auto& family = *f.controls.qbmm_families.at(1);
        const auto& contract = family.closure_versions().laws.at(0);
        const auto drain = [&](const R::QbmmLawInputs& input)
        {
            ++f.calls;
            const double fraction = .5 * input.time_step_s.value() / dt;
            return R::qbmm_pool_drain_from_fraction(
                family, input, *f.pool, std::vector<double>(input.quadrature.active_atoms, fraction));
        };
        const R::ThermalLawIdentity provider{"pool kinetics", 1, f.catalog->fingerprint()};
        if (joint)
            f.options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
                R::MixtureQbmmProvider{provider, f.slots, {}, {}, [&](const R::MixtureQbmmJointContext& ctx)
                    {
                        std::vector<Operations> rows;
                        for (const auto& row : ctx.families)
                            rows.push_back(drain(row.law_inputs));
                        return rows;
                    }});
        else
        {
            auto typed = std::make_shared<const R::MixtureQbmmLaw>(R::MixtureQbmmLaw{contract, drain});
            f.options.qbmm_population =
                std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{provider, f.slots, {}, {typed}});
        }
        f.options.qbmm_source_substep_relative_tolerance = .005;
        Coupler controller(f.flow, f.options);
        const auto trial = controller.preview(dt);
        const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
        ASSERT_GT(n, 1u);
        const auto report = controller.qbmm_report(trial);
        const auto& totals = report.pool_conversion.value().directions[0];
        const double retained = 1 - .5 / n;
        constexpr std::array initial{3., 5., 11., 29.};
        const double lost = 1 - std::pow(retained, n);
        near(totals.count, initial[0] * lost);
        near(totals.mass_kg, scale * initial[1] * lost);
        near(totals.H2, initial[2] * lost);
        near(totals.H3, initial[3] * lost);
        near(totals.carried_enthalpy_j, 5 * lost * (4 * 24 + 2 * 40) * T0);
        EXPECT_EQ(report.pool_conversion->law_evaluations, 1u);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& receipt = trial.local_qbmm_sources()[c][1].value();
            ASSERT_EQ(receipt.substeps.size(), n);
            ASSERT_EQ(receipt.events.size(), 2 * n);
            EXPECT_EQ(report.cells[c].families[1]->pool_conversion_law_evaluated, true);
            for (size_t i = 0; i < n; ++i)
            {
                const auto& stage = receipt.substeps[i];
                const double fraction = std::pow(retained, i + 1);
                near(stage.duration, dt / n);
                EXPECT_EQ(stage.event_end, 2 * (i + 1));
                EXPECT_EQ(receipt.events.at(2 * i).provider, joint ? provider.identity : contract.identity);
                near(stage.final_count, initial[0] * fraction);
                near(stage.final_shape.H2, initial[2] * fraction);
                near(stage.final_shape.H3, initial[3] * fraction);
                for (size_t s = 0; s < stage.final_moles.size(); ++s)
                    near(stage.final_moles[s], Fixture::moles(initial[1])[s] * fraction);
            }
            for (size_t s = 0; s < totals.species_moles.size(); ++s)
            {
                const double amount = Fixture::moles(initial[1])[s] * lost;
                near(totals.species_moles[s], amount);
                near(trial.state().cells[c].phases[2].moles[s], amount);
            }
        }
        if (joint)
        {
            EXPECT_EQ(n, typed_steps);
            EXPECT_EQ(totals, typed_totals.value());
        }
        else
        {
            typed_steps = n;
            typed_totals = totals;
        }
    }
}

TEST(ReactingMixtureQbmmPool, GasFractionAndTracesUsePhysicalVolumeAtPositiveLiquidOccupancy)
{
    for (const double liquid : {.75, .5, 1. / 16., 1. / 65536.})
    {
        SCOPED_TRACE(liquid);
        Fixture f({.liquid_fraction = liquid, .pool_stock = 1, .declare_liquid_law = true});
        f.events = [&](const Context& ctx)
        {
            const auto& input = *ctx.law_inputs;
            near(input.gas_volume_fraction.value(), 1 - liquid);
            near(input.carrier_liquid_volume_m3.value(), liquid);
            near(input.carrier_liquid_temperature_K.value(), T0);
            near(input.carrier_liquid_concentration_mol_per_m3.value()[2], 1.);
            const double pool_volume = (1 - liquid) / 6.;
            near(input.bulk_gas_pool_volume_m3.value(), pool_volume);
            near(input.bulk_gas_pool_temperature_K.value(), T0);
            near(input.bulk_gas_pool_concentration_mol_per_m3.value()[0], 4. / pool_volume);
            EXPECT_EQ(input.bulk_gas_pool_moles.value(), Fixture::moles(1));
            return Operations{};
        };
        f.start();
        const auto trial = f.controller->preview(dt);
        EXPECT_GT(f.sum(f.liquid_calls), 0.);
        const auto report = f.controller->qbmm_report(trial);
        for (const auto& cell : report.cells)
        {
            near(cell.gas_volume_fraction, 1 - liquid);
            near(cell.dispersed_gas_volume_fraction, 5 * (1 - liquid) / 6);
            near(cell.bulk_gas_pool_volume_fraction, (1 - liquid) / 6);
            EXPECT_EQ(cell.families[1]->pool_conversion_law_evaluated, true);
        }
    }
}

TEST(ReactingMixtureQbmmPool, SharedOriginalSpeciesAndAtomDebitsRejectWithoutPublishing)
{
    Fixture f({.pool_stock = 1, .two_families = true});
    f.check_ranks();
    for (auto id : {A, B})
        f.slots->stages.front().material.push_back({id, 1, 0});
    f.start();
    const auto unchanged = f.controller->serialize_qbmm_restart();
    for (int failure = 0; failure < 7; ++failure)
    {
        SCOPED_TRACE(failure);
        f.events = [&](const Context& in) -> Operations
        {
            if (failure == 0)
                return f.birth(in, .75, 1.);
            if (failure == 1)
                return in.phase == 1 ? f.drain(in, {0, 1}) : f.birth(in, 2, 1);
            if (in.phase != 1)
                return {};
            if (failure == 2)
            {
                auto ops = f.drain(in, {.25, 0.});
                ops.emplace_back(R::QbmmCoalescence{0, 0, .5});
                ops.emplace_back(
                    R::QbmmDisappearance{0, .75, {R::QbmmPhaseExchange{Phase::Dissolved}, Fixture::moles(.75)}});
                return ops;
            }
            auto ops = f.drain(in, {0, 1});
            auto& packet = std::get<R::QbmmDisappearance>(ops[0]).transfer;
            auto& tag = std::get<R::QbmmPhaseExchange>(packet.counterpart).resolved_pool;
            if (failure == 3)
                tag.reset();
            if (failure == 4)
                tag->direction = Direction::PoolToDispersed;
            if (failure == 5)
                tag->pool_fingerprint++;
            if (failure == 6)
            {
                packet.moles[0] -= 2;
                packet.moles[1] += 1;
            }
            return ops;
        };
        EXPECT_THROW((void) f.controller->preview(dt), std::exception);
        EXPECT_EQ(f.controller->serialize_qbmm_restart(), unchanged);
    }
    f.events = [&](const Context& in) { return f.birth(in, .5, 1); };
    const auto trial = f.controller->preview(dt);
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(cell.phases[1].moles, Fixture::moles(5.5));
        EXPECT_EQ(cell.phases[2].moles, Fixture::moles(.5));
        EXPECT_EQ(cell.phases[3].moles, Fixture::moles(0));
    }
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), unchanged);

    // Exactly two original unit parents. Coalescence consumes 1.5 and the other two uses split the remaining half.
    constexpr double dissolved = 1. / 1024.;
    for (bool overdraw : {true, false})
    {
        f.events = [&](const Context& in) -> Operations
        {
            if (in.phase != 1)
                return {};
            const double loss = overdraw ? std::nextafter(dissolved, 1.) : dissolved;
            auto ops = f.drain(in, {(.5 - dissolved) / 2, 0.});
            ops.emplace_back(R::QbmmCoalescence{0, 0, .75});
            ops.emplace_back(
                R::QbmmDisappearance{0, loss, {R::QbmmPhaseExchange{Phase::Dissolved}, Fixture::moles(loss)}});
            return ops;
        };
        if (overdraw)
        {
            EXPECT_THROW((void) f.controller->preview(dt), std::exception);
            EXPECT_EQ(f.controller->serialize_qbmm_restart(), unchanged);
            continue;
        }
        const auto exact = f.controller->preview(dt);
        for (const auto& cell : exact.state().cells)
        {
            EXPECT_EQ(cell.phase_numbers[1], 1.75);
            EXPECT_EQ(cell.qbmm_shape[1], (R::QbmmShapeState{12., 33.}));
            EXPECT_EQ(cell.phases[1].moles, Fixture::moles(4.5));
            EXPECT_EQ(cell.phases[3].moles, Fixture::moles(1.5 - dissolved));
            near(cell.phases[0].moles[0], 4 * dissolved);
            near(cell.phases[0].moles[1], 2 * dissolved);
        }
    }
}

TEST(ReactingMixtureQbmmPool, PassiveBulkGasRetainsItsStockAndCannotSupplyPoolBirths)
{
    Fixture f({.pool_stock = 1, .passive = true});
    const double gas_fraction = 1 - f.setup.liquid_fraction;
    f.events = [&](const Context& in)
    {
        const auto& inputs = *in.law_inputs;
        EXPECT_EQ(inputs.bulk_gas_pool_moles.value(), Fixture::moles(1));
        near(inputs.bulk_gas_pool_volume_m3.value(), gas_fraction / 8);
        near(inputs.gas_volume_fraction.value(), gas_fraction);
        return f.birth(in, 2, 1);
    };
    f.start();
    const auto before = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
    f.events = [&](const Context& in) { return f.birth(in, 1, 1); };
    const auto trial = f.controller->preview(dt);
    const auto report = f.controller->qbmm_report(trial);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& cell = trial.state().cells[c];
        EXPECT_EQ(cell.phases[1].moles, Fixture::moles(6));
        EXPECT_EQ(cell.phases[2].moles, Fixture::moles(0));
        EXPECT_EQ(cell.phases[3].moles, Fixture::moles(2));
        EXPECT_FALSE(cell.phase_numbers[3]);
        EXPECT_FALSE(cell.qbmm_shape[3]);
        const auto& row = report.cells[c];
        near(row.gas_volume_fraction, gas_fraction);
        near(row.dispersed_gas_volume_fraction, .75 * gas_fraction);
        EXPECT_EQ(row.bulk_gas_pool_volume_fraction, 0.);
        near(row.passive_gas_volume_fraction, .25 * gas_fraction);
    }
}

TEST(ReactingMixtureQbmmPool, EqualAndOppositeConversionsRetainDirectionalCorrectorActivity)
{
    Fixture f({.pool_stock = 1});
    f.events = [&](const Context& in)
    {
        const double n = in.stage.corrector == 0 ? .25 : .5;
        auto out = f.drain(in, {n / 2., 0});
        auto back = f.birth(in, n, 1.);
        out.push_back(std::move(back.front()));
        return out;
    };
    f.start();
    const auto trial = f.controller->preview(dt);
    EXPECT_GE(trial.diagnostics().correctors, 3);
    const auto& diag = trial.diagnostics().flow.pool_conversion.value();
    for (const auto& direction : diag.directions)
    {
        near(direction.count, .5);
        near(direction.mass_kg, .5 * scale);
        near(direction.H2, .5);
        near(direction.H3, .5);
        EXPECT_EQ(direction.species_moles, Fixture::moles(.5));
        near(direction.carried_enthalpy_j, .5 * (4 * 24 + 2 * 40) * T0);
    }
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(cell.phases[1].moles, Fixture::moles(5));
        EXPECT_EQ(cell.phases[2].moles, Fixture::moles(1));
        EXPECT_EQ(cell.phase_numbers[1], 3);
        EXPECT_EQ(cell.qbmm_shape[1], (R::QbmmShapeState{11, 29}));
    }
}

TEST(ReactingMixtureQbmmPool, FailedPreviewPreservesAcceptedConversionAndACompetingTrial)
{
    Fixture f;
    f.start();
    f.controller->advance(dt);
    const auto checkpoint = f.controller->checkpoint();
    const auto before = f.controller->serialize_qbmm_restart();
    const auto accepted = f.controller->diagnostics().flow.pool_conversion;
    const auto competing = f.controller->preview(dt);
    f.events = [&](const Context& in) { return f.birth(in, 8, 1); };
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(before, f.controller->serialize_qbmm_restart());
    EXPECT_EQ(f.controller->diagnostics().flow.pool_conversion, accepted);
    const auto evaluated = f.calls;
    f.controller->commit(competing);
    near(f.controller->diagnostics().flow.pool_conversion.value().directions[1].mass_kg, 3 * scale);
    f.controller->restore(checkpoint);
    EXPECT_EQ(f.calls, evaluated);
    EXPECT_EQ(f.flow->state().time, dt);
    EXPECT_EQ(f.controller->diagnostics().flow.pool_conversion, accepted);
}

TEST(ReactingMixtureQbmmPool, DirectionalPoolWitnessDetectsChangingCounterpartAtIdenticalFamilyGross)
{
    Fixture f({.pool_stock = 1});
    f.events = [&](const Context& in)
    {
        auto death = f.drain(in, {.25, 0.});
        auto birth = f.birth(in, .5, 1.);
        // Identical parents and products on every corrector. The balanced external packets carry zero energy/impulse.
        if (in.stage.corrector == 0)
        {
            std::get<R::QbmmDisappearance>(death.front()).transfer.counterpart = R::QbmmReservoirExchange{"out"};
            std::get<R::QbmmBirth>(birth.front()).atoms.front().transfer.counterpart = R::QbmmReservoirExchange{"in"};
        }
        death.push_back(std::move(birth.front()));
        return death;
    };
    f.start();
    const auto trial = f.controller->preview(dt);
    const auto& flow = trial.diagnostics().flow;
    const auto& trace = flow.gross_residual_evaluations;
    ASSERT_GE(trace.size(), 3u);
    ASSERT_TRUE(trace.at(1).winner);
    EXPECT_EQ(trace[1].winner->channel_metadata.stage, Flow::GrossResidualStage::QbmmPoolConversion);
    EXPECT_EQ(trace[1].winner->previous_packet, 0.);
    EXPECT_GT(trace[1].winner->current_packet, 0.);
    EXPECT_GT(trace[1].normalized_residual, 1.);
    EXPECT_EQ(trace.back().normalized_residual, 0.);
    for (const auto& direction : flow.pool_conversion.value().directions)
    {
        EXPECT_EQ(direction.count, .5);
        EXPECT_EQ(direction.species_moles, Fixture::moles(.5));
    }
    EXPECT_EQ(trial.diagnostics().external_mass, 0.);
    EXPECT_EQ(trial.diagnostics().transfer_energy, 0.);
}

TEST(ReactingMixtureQbmmPool, DryLiquidCallbacksStayIdleWhileGasChemistryAndHeatAdvance)
{
    for (bool liquid_rate : {false, true})
    {
        Fixture f({.liquid_fraction = 0,
            .pool_stock = 1,
            .two_cells = true,
            .dry_chemistry = true,
            .declare_liquid_law = true});
        int rate_calls = 0, contact_calls = 0;
        f.events = [](const Context& in)
        {
            EXPECT_EQ(in.law_inputs->carrier_liquid_volume_m3, 0.);
            EXPECT_FALSE(in.law_inputs->carrier_liquid_concentration_mol_per_m3);
            EXPECT_FALSE(in.law_inputs->carrier_liquid_temperature_K);
            EXPECT_TRUE(in.law_inputs->bulk_gas_pool_concentration_mol_per_m3);
            return Operations{};
        };
        f.options.source_identity = "dry rate and heat";
        f.options.heat_phase = 0; // The nominal liquid heat recipient is absent.
        f.options.source = [&](const R::CoupledReactionContext& context, double)
        {
            EXPECT_EQ(context.reaction.environment.liquidVolume, 0.);
            EXPECT_GT(context.reaction.environment.gasVolume.value_or(0.), 0.);
            EXPECT_GT(context.reaction.environment.gasVolume.value(), 0.);
            R::ReactionRateProvider rate;
            rate.identity = "dimerization";
            rate.schemaFingerprint = f.catalog->fingerprint();
            rate.schemaRevision = f.catalog->revision();
            rate.validityDescription = "positive gas volume";
            rate.parameterUnits = "mol/(m3 s)";
            rate.requiresLiquid = liquid_rate;
            rate.validate = [&](const auto&) { ++rate_calls; };
            rate.evaluate = [&](const auto& rxn)
            {
                ++rate_calls;
                EXPECT_EQ(rxn.environment.liquidVolume, 0.);
                return std::vector<R::ReactionRate>{{7, 1., R::SourceVolumeBasis::BulkVolume}};
            };
            R::ReactionSourceInput result;
            result.rates = rate;
            result.heat.rate = R::VolumetricHeatRate{64., R::SourceVolumeBasis::BulkVolume};
            return result;
        };
        R::MixtureFaceTransportOptions contacts;
        contacts.identity = "liquid Fourier";
        contacts.physical_phase = 0;
        contacts.schema_fingerprint = f.catalog->fingerprint();
        R::MixtureDiffusionLaw diffusion;
        diffusion.binding.identity = "dry liquid solute";
        diffusion.binding.schemaFingerprint = f.catalog->fingerprint();
        diffusion.binding.counterSpecies = solvent;
        diffusion.primary_species = {A};
        diffusion.molar_conductance = [&](const auto&)
        {
            ++contact_calls;
            return std::vector<double>{1., 0., 0.};
        };
        diffusion.partial_molar_enthalpy = [&](auto, const auto&, const auto&)
        {
            ++contact_calls;
            return 0.;
        };
        contacts.diffusion = std::move(diffusion);
        contacts.fourier = R::MixtureFourierLaw{"dry contact", 1, f.catalog->fingerprint(), [&](const auto&)
            {
                ++contact_calls;
                return 1.;
            }};
        f.options.face_transport = {contacts};
        auto heat_exchange = std::make_shared<R::InterphaseConfiguration>();
        heat_exchange->metadata = {"liquid contact heat stage", 1, f.catalog->fingerprint()};
        heat_exchange->phase_enthalpy.resize(f.eos.size());
        heat_exchange->stages = {{"liquid to pool", R::InterphaseStageKind::Heat, {}, {{0, 2}}}};
        f.options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
            R::MixtureInterphaseEventProvider{heat_exchange->metadata, heat_exchange, [&](const auto&)
                {
                    ++contact_calls;
                    return std::vector<std::vector<double>>{{1.}};
                }});
        f.start();
        const auto trial = f.controller->preview(dt);
        EXPECT_EQ(f.liquid_calls, 0);
        EXPECT_EQ(contact_calls, 0);
        if (liquid_rate)
            EXPECT_EQ(rate_calls, 0);
        else
            EXPECT_GT(f.sum(rate_calls), 0);
        near(trial.diagnostics().reaction_extents.at(0), liquid_rate ? 0. : dt);
        near(trial.diagnostics().external_heat, 64 * dt);
        const auto report = f.controller->qbmm_report(trial);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& cell = trial.state().cells[c];
            const double v = f.mesh->cell_volume(c), extent = liquid_rate ? 0. : v * dt;
            near(cell.phases[2].moles[0], 4 * v - 2 * extent);
            near(cell.phases[2].moles[1], 2 * v + extent);
            EXPECT_EQ(cell.phases[0].temperature, 0);
            near(report.cells[c].gas_volume_fraction, 1);
            EXPECT_GT(cell.phases[2].temperature, T0);
        }
    }
}

TEST(ReactingMixtureQbmmPool, PackedAndScalarTransportCarryConversionAcrossPartitionFaces)
{
    Fixture packed({.circulating = true}), scalar({.circulating = true, .batched = false});
    packed.check_ranks(false);
    auto run = [](Fixture& f)
    {
        f.events = [&f](const Context& in)
        { return f.drain(in, std::vector<double>(in.quadrature.active_atoms, .125)); };
        f.start();
        return f.controller->preview(dt);
    };
    const auto a = run(packed), b = run(scalar);
    EXPECT_GT(a.diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
    EXPECT_EQ(b.diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
    const auto comm = packed.mesh->owned_cell_map()->getComm();
    double interiors = 0, crossing = 0, ingress = 0, total_pool = 0, total_number = 0;
    for (size_t face = 0; face < a.owned_phase_transport().size(); ++face)
    {
        if (packed.mesh->is_geometry_exterior_face(face))
            continue;
        const double flux = a.owned_phase_transport()[face].qbmm_flux.at(1)->population_number_flux;
        near(flux, b.owned_phase_transport()[face].qbmm_flux.at(1)->population_number_flux);
        interiors += std::abs(flux);
        if (static_cast<size_t>(packed.mesh->owner_cell(face)) >= packed.mesh->num_owned_cells() ||
            static_cast<size_t>(packed.mesh->neighbor_cell(face)) >= packed.mesh->num_owned_cells())
            crossing += std::abs(flux);
    }
    EXPECT_GT(packed.sum(interiors), 1e-6);
    if (comm->getSize() > 1)
        EXPECT_GT(packed.sum(crossing), 1e-6);
    near(a.state().thermodynamic_pressure, b.state().thermodynamic_pressure);
    for (size_t c = 0; c < a.state().cells.size(); ++c)
    {
        const auto& ca = a.state().cells[c];
        const auto& cb = b.state().cells[c];
        for (size_t p = 0; p < ca.phases.size(); ++p)
        {
            for (size_t s = 0; s < 3; ++s)
                near(ca.phases[p].moles[s], cb.phases[p].moles[s]);
            near(ca.phases[p].temperature, cb.phases[p].temperature);
            near(ca.phases[p].internal_energy, cb.phases[p].internal_energy);
        }
        near(*ca.phase_numbers[1], *cb.phase_numbers[1]);
        near(ca.qbmm_shape[1]->H2, cb.qbmm_shape[1]->H2);
        near(ca.qbmm_shape[1]->H3, cb.qbmm_shape[1]->H3);
        total_number += *ca.phase_numbers[1];
        total_pool += ca.phases[2].moles[0];
        if (packed.mesh->cell_centroid(c).x > .5)
            ingress += *ca.phase_numbers[1];
    }
    EXPECT_GT(packed.sum(ingress), 1e-6);
    near(packed.sum(total_number), 1.5 * .875);
    near(packed.sum(total_pool), 10 * (1 + .125));
    near(a.diagnostics().flow.pool_conversion->directions[0].mass_kg, 2.5 * .125 * scale);
}

TEST(ReactingMixtureQbmmPool, ResidualDissolvedStockWithoutSolventRejectsAtSetupAndAfterChemistry)
{
    Fixture invalid;
    invalid.check_ranks();
    auto state = invalid.flow->state();
    for (auto& cell : state.cells)
    {
        auto& liquid = cell.phases[0];
        // Equal molar masses and a composition-independent liquid EOS preserve U, density and occupied volume.
        liquid.moles[0] = std::exchange(liquid.moles[2], 0.);
    }
    invalid.flow->initialize(std::move(state));
    const auto initial = invalid.flow->serialize_qbmm_restart();
    EXPECT_THROW(invalid.start(), std::exception);
    EXPECT_EQ(invalid.calls, 0);
    EXPECT_EQ(invalid.flow->serialize_qbmm_restart(), initial);

    Fixture f;
    f.options.source_identity = "solvent removal with residual solute";
    f.options.source = [](const R::CoupledReactionContext& context, double)
    {
        const auto& rxn = context.reaction;
        const double stock = rxn.state.amount(rxn.catalog, solvent, Phase::Liquid);
        R::ReactionSourceInput input;
        input.externalTransfers = {{"solvent withdrawal", solvent, Phase::Liquid, -stock, -stock},
            {"solute supply", A, Phase::Dissolved, stock, stock}};
        return input;
    };
    f.start();
    const auto before = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(f.calls, 0);
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
}

TEST(ReactingMixtureQbmmPool, UnsupportedModelsAndUnpermittedSlotsFailCollectively)
{
    using M = R::QbmmModelOptions;
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    for (int mode = 0; mode < 9; ++mode)
    {
        SCOPED_TRACE(mode);
        auto options = f.controls;
        if (comm->getRank() == comm->getSize() - 1)
        {
            auto& model = options.qbmm_model;
            if (mode == 0)
                model.family_transport = M::FamilyTransport::KineticEuler;
            if (mode == 1 || mode == 2)
                model.family_transport = M::FamilyTransport::RealizableLinearSSPRK2;
            if (mode == 2)
                model.linear_slope_limiter = M::LinearSlopeLimiter::TvdMc;
            if (mode == 3)
                model.velocity = M::Velocity::SuppliedPhaseSlip;
            if (mode == 4)
                model.velocity = M::Velocity::NodeDependentSlip;
            if (mode == 5)
                model.velocity = M::Velocity::IndependentPhaseMomentum;
            if (mode == 6)
                model.transport.time = FVM::ScalarTimeScheme::BDF2;
            if (mode == 7)
                model.transport.convection = FVM::ScalarConvectionScheme::BoundedLinearUpwind;
            if (mode == 8)
                options.qbmm_bulk_gas_pool.reset();
        }
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    }
    for (auto acceleration : {Flow::Vector{}, Flow::Vector{0., 0., -1.}})
    {
        int direct_calls = 0;
        auto direct = std::make_shared<const Flow::SourceStageProvider>(
            Flow::SourceStageProvider{"direct acceleration", 1, f.catalog->fingerprint(),
                [&](const auto&)
                {
                    ++direct_calls;
                    return Flow::SourceStageResult{};
                },
                0., 0, acceleration});
        EXPECT_THROW((void) f.flow->preview_with_sources(dt, direct), std::exception);
        EXPECT_EQ(direct_calls, 0);
        f.options.body_acceleration = acceleration;
        EXPECT_THROW(f.start(), std::exception);
    }
    f.options.body_acceleration.reset();
    {
        Flow::PlanarMotion motion(f.mesh);
        EXPECT_THROW((void) f.flow->preview_ale(dt, motion, 1.001), std::exception);
        EXPECT_FALSE(motion.has_active_trial());
    }
    int motion_calls = 0;
    f.options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
        R::MixturePlanarMotionProvider{f.mesh, {}, "prescribed", 1, [&](double, double)
            {
                ++motion_calls;
                return 1.;
            }});
    EXPECT_THROW(f.start(), std::exception);
    EXPECT_EQ(motion_calls, 0);
    Fixture wrong({.direction = Direction::DispersedToPool});
    if (comm->getRank() == comm->getSize() - 1)
        wrong.slots->stages[0].material.push_back({A, 2, 1});
    EXPECT_THROW(wrong.start(), std::exception);
    Fixture missing;
    missing.slots->stages[0].material.erase(missing.slots->stages[0].material.begin());
    EXPECT_THROW(
        {
            missing.start();
            (void) missing.controller->preview(dt);
        },
        std::exception);
}

TEST(ReactingMixtureQbmmPool, PortableContinuation)
{
    Fixture f;
    f.check_ranks();
    f.start();
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const auto* writer = std::getenv("SIMPLEFLUID_Q9B_WRITE");
    const auto* reader = std::getenv("SIMPLEFLUID_Q9B_READ");
    ASSERT_FALSE(writer && reader);
    std::vector<std::byte> first;
    if (reader)
    {
        first = IO::read_restart(*comm, reader, ".deposit");
        f.controller->restore_qbmm_restart(IO::read_restart(*comm, reader, ".return"));
        EXPECT_EQ(f.calls, 0);
    }
    else
    {
        f.controller->advance(dt);
        first = f.controller->serialize_qbmm_restart();
        f.controller->advance(dt);
        if (writer)
        {
            IO::write_restart(*comm, writer, ".deposit", first);
            IO::write_restart(*comm, writer, ".return", f.controller->serialize_qbmm_restart());
            return;
        }
    }
    ASSERT_GE(first.size(), 24u);
    EXPECT_EQ(first[8], std::byte{10});
    Fixture fresh;
    fresh.start();
    fresh.controller->restore_qbmm_restart(first);
    EXPECT_EQ(fresh.calls, 0);
    for (const auto& cell : fresh.controller->qbmm_report().cells)
        EXPECT_FALSE(cell.families[1]->pool_conversion_law_evaluated.has_value());
    near(fresh.controller->diagnostics().flow.pool_conversion.value().directions[0].mass_kg, 3 * scale);
    const auto candidate = fresh.controller->preview(dt);
    const auto saved = fresh.controller->serialize_qbmm_restart();
    auto dry_archive = first;
    dry_archive.resize(dry_archive.size() - 16);
    bool mutated = false;
    for (const auto& cell : fresh.flow->state().cells)
    {
        const auto& liquid = cell.phases[0];
        std::vector<std::byte> row;
        for (double value : {liquid.absolute_pressure, liquid.internal_energy, liquid.temperature,
                 cell.drift_velocity[0].x, cell.drift_velocity[0].y, cell.drift_velocity[0].z})
            IO::append_restart_word(row, std::bit_cast<std::uint64_t>(value));
        const auto moles_offset = row.size();
        for (double moles : liquid.moles)
            IO::append_restart_word(row, std::bit_cast<std::uint64_t>(moles));
        IO::append_restart_word(row, 0); // Liquid has neither number nor shape.
        IO::append_restart_word(row, 0);
        const auto match = std::ranges::search(dry_archive, row);
        if (!match.empty() && std::ranges::search(std::ranges::subrange(match.end(), dry_archive.end()), row).empty())
        {
            auto monomer = match.begin() + moles_offset;
            std::swap_ranges(monomer, monomer + 8, monomer + 16); // Swap dissolved A and the equal-mass solvent.
            mutated = true;
        }
    }
    ASSERT_EQ(fresh.sum(mutated), 1.);
    IO::append_restart_checksums(*comm, dry_archive);
    const auto evaluations = fresh.calls;
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(dry_archive), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), saved);
    fresh.controller->commit(candidate);
    EXPECT_EQ(fresh.calls, evaluations);
    EXPECT_DOUBLE_EQ(fresh.flow->state().time, 2 * dt);
    EXPECT_DOUBLE_EQ(fresh.flow->state().thermodynamic_pressure, f.flow->state().thermodynamic_pressure);
    EXPECT_EQ(fresh.controller->diagnostics().flow.pool_conversion, f.controller->diagnostics().flow.pool_conversion);
    for (size_t c = 0; c < fresh.mesh->num_owned_cells(); ++c)
    {
        const auto& a = fresh.flow->state().cells[c];
        const auto& b = f.flow->state().cells[c];
        EXPECT_EQ(a.phase_numbers, b.phase_numbers);
        EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
        for (size_t p = 0; p < a.phases.size(); ++p)
        {
            EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
            EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
            EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
        }
    }
    const auto archive = fresh.controller->serialize_qbmm_restart();
    auto invalid = first;
    invalid.resize(invalid.size() - 16);
    if (comm->getRank() == comm->getSize() - 1)
        invalid[16] ^= std::byte{0x10};
    IO::append_restart_checksums(*comm, invalid);
    const int calls = fresh.calls;
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(invalid), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), archive);
    EXPECT_EQ(fresh.calls, calls);
    for (auto setup : {FixtureOptions{.extra_permitted = true}, FixtureOptions{.direction = Direction::DispersedToPool},
             FixtureOptions{.law_revision = 2}, FixtureOptions{.different_eos = true},
             FixtureOptions{.pool_name = "other gas"}, FixtureOptions{.two_families = true}})
    {
        Fixture mismatch(setup);
        mismatch.start();
        const auto before = mismatch.controller->serialize_qbmm_restart();
        EXPECT_THROW(mismatch.controller->restore_qbmm_restart(first), std::exception);
        EXPECT_EQ(mismatch.controller->serialize_qbmm_restart(), before);
        EXPECT_EQ(mismatch.calls, 0);
    }
    Fixture legacy({.legacy = true});
    legacy.events = [](const Context&) { return Operations{}; };
    legacy.start();
    legacy.controller->advance(dt);
    const auto old_archive = legacy.controller->serialize_qbmm_restart();
    EXPECT_EQ(old_archive[8], std::byte{5});
    EXPECT_THROW(legacy.controller->restore_qbmm_restart(first), std::exception);
    EXPECT_EQ(legacy.controller->serialize_qbmm_restart(), old_archive);
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(old_archive), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), archive);
}
} // namespace
