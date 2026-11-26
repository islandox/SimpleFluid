#pragma once

/** Small ideal-gas family/pool fixture with independently known extensive and caloric inventories. */
#include "QbmmRestartTestSupport.hh"
#include "equations/QbmmPopulationLawAdapters.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>
#include <numeric>

namespace SimpleFluid::Testing::QbmmPool
{
namespace R = Reacting;
namespace IO = Testing::QbmmRestart;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using Phase = R::SpeciesPhase;
using Direction = R::QbmmPoolConversionDirection;
using Operations = std::vector<R::QbmmEventOperation>;
using Context = R::MixtureQbmmContext;
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
    bool passive = false, supplied_slip = false;
    std::optional<Flow::Vector> acceleration{};
    bool four_layers = false, periodic_z = false;
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
        const auto z = s.four_layers ? ArrReal{0, .25, .5, .75, 1} : s.two_cells ? ArrReal{0, .5, 1} : cuts;
        mesh =
            std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{cuts, ArrReal{0, 1}, z},
                                             Vec3D<bool>{false, false, s.periodic_z}),
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
        if (s.supplied_slip)
            controls.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
        options.body_acceleration = s.acceleration;
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
                const Flow::Vector circulation{.5 - x.z, 0, x.x < .5 ? -.25 : .25};
                if (s.supplied_slip)
                {
                    cell.drift_velocity.resize(nph);
                    cell.drift_velocity[1] = circulation;
                    cell.drift_velocity[last] = circulation * -.75;
                }
                else
                    cell.velocity = circulation;
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

inline void near(double actual, double expected, double tol = 3e-9)
{
    EXPECT_NEAR(actual, expected, tol * std::max(1., std::abs(expected)));
}
} // namespace SimpleFluid::Testing::QbmmPool
