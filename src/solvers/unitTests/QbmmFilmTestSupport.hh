/** Shared one-cell finite-liquid film fixture with configurable population laws. */
#pragma once
#include "ConditionedThermalTestSupport.hh"
#include "QbmmRestartTestSupport.hh"
#include "equations/QbmmPopulationLawAdapters.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <numeric>

namespace SimpleFluid::Testing::QbmmFilm
{
namespace R = Reacting;
namespace T = Testing::ConditionedTransfer;
namespace IO = Testing::QbmmRestart;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using Phase = R::SpeciesPhase;
using Operations = std::vector<R::QbmmEventOperation>;
using Nodes = std::vector<R::QbmmFilmNodeValues>;

constexpr R::SpeciesId tracer = 44, carrier = 709;
constexpr double mass_scale = 1. / 1024, dt = .01, dissolved = .02, solvent_moles = 12.5, carrier_molar_mass = .028;
constexpr double liquid_volume = dissolved + solvent_moles * carrier_molar_mass; // rho_liquid = 1 kg/m3.
constexpr double liquid_u = 2 * (T::cv / T::gas_constant) * T::p_old;
const std::array fine{R::QbmmAtom{2., 1.}, R::QbmmAtom{1., 3.}};
const std::array coarse{R::QbmmAtom{1., 5.}};
inline std::span<const R::QbmmAtom> seed(std::string_view name)
{
    if (name == "fine")
        return fine;
    return coarse;
}

inline double film_coefficient(std::string_view name)
{
    return name == "fine" ? 12000. : 8000.; // Supplied m/s, deliberately stiff for the tested interval.
}

// Independent ideal-gas sphere area. Each bubble contains pure tracer at 1 kg/mol.
inline double area(double scaled_mass, double temperature, double pressure)
{
    const double volume = scaled_mass * mass_scale * T::gas_constant * temperature / pressure;
    return std::cbrt(36 * std::numbers::pi * volume * volume);
}

enum class Failure
{
    None,
    Explicit,
    SeparateSolves,
    WrongRowCount
};

enum class Evaluation
{
    Integrated,
    Typed,
    Joint
};
const R::QbmmLawMetadata film_law{"finite liquid film", 1, R::QbmmLawKind::MassTransfer,
    "frozen bubble geometry and dissolved inventory", "mol/cell", R::QbmmApplicabilityGate::RejectOutside, true,
    {{R::QbmmRangeVariable::TimeStepS, dt, dt, {}}}};

struct Fixture
{
    int calls = 0, partial_calls = 0, motion_calls = 0;
    Failure failure = Failure::None;
    double attempted = 0;
    std::array<double, 4> frozen_temperature{}, frozen_pressure{};
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<Coupler> controller;

    Fixture(Evaluation mode, bool moving, bool reversed = false,
        const std::function<void(R::ReactingMixtureOptions&)>& configure = {},
        std::span<const R::QbmmAtom> coarse_seed = coarse, const R::QbmmLawMetadata& law_contract = film_law,
        ArrReal z_faces = {0, 1})
    {
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {tracer, "film tracer", 1., {{"X", 1}}, {Phase::Dissolved, Phase::Dispersed}},
            {carrier, "carrier", carrier_molar_mass, {{"Y", 1}}, {Phase::Liquid, Phase::Resolved}}});
        const R::ThermodynamicBounds bounds{250, 1200, 50000, 1000000};
        const auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{T::T0, T::p_old, 1, 1000, 0, 0, liquid_u, 0, 0}; });
        const auto gas =
            R::make_ideal_gas_mixture(catalog, {{T::cv, T::T0, T::cv * T::T0}, {T::cv, T::T0, T::cv * T::T0}}, bounds);
        Flow::Mesh::DistributionOptions partition;
        partition.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, std::move(z_faces)}),
            partition);

        Flow::Options controls;
        controls.minimum_pressure = bounds.minimum_pressure;
        controls.maximum_pressure = bounds.maximum_pressure;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}};
        controls.qbmm_families = {nullptr};
        R::ReactingMixtureOptions options;
        options.mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, gas->energy_convention());
        options.phases = {{Phase::Liquid, {}, {Phase::Dissolved, Phase::Liquid}}};
        R::InterphaseConfiguration transfers;
        transfers.metadata = {"liquid film enthalpy", 1, catalog->fingerprint()};
        transfers.stages = {{"dissolved tracer into bubbles", R::InterphaseStageKind::Material, {}, {}}};
        std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos{liquid};
        Flow::State state;
        state.thermodynamic_pressure = T::p_old;
        state.qbmm_families = {std::nullopt};
        Flow::CellState cell;
        cell.phases = {{liquid, {dissolved, solvent_moles}, T::p_old, liquid_volume * liquid_u, T::T0}};
        cell.phase_numbers = {std::nullopt};
        cell.qbmm_shape = {std::nullopt};
        double inert = (1 - liquid_volume) * T::p_old / (T::gas_constant * T::T0);
        std::vector<std::string> names{"fine"};
        if (mode == Evaluation::Joint)
            names.push_back("coarse");
        if (reversed)
            std::reverse(names.begin(), names.end());
        for (const auto& name : names)
        {
            transfers.stages.front().material.push_back({tracer, 0, eos.size()});
            eos.push_back(gas);
            controls.phase_roles.push_back({Flow::PhaseRole::DispersedGas, name});
            R::QbmmClosureVersions policies;
            policies.source_identity = "film";
            if (name == "coarse" || mode == Evaluation::Typed)
                policies.laws = {law_contract};
            auto family = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), name,
                controls.phase_roles.back(), mass_scale, std::vector<R::QbmmMassFraction>{{tracer, 1}, {carrier, 0}},
                policies);
            controls.qbmm_families.push_back(family);
            state.qbmm_families.emplace_back(Flow::qbmm_family_stamp(*family));
            options.phases.push_back({Phase::Dispersed, name, {}});
            std::array<double, 4> moments{};
            for (const auto [count, x] : name == "coarse" ? coarse_seed : seed(name))
                for (int k = 0; k < 4; ++k)
                    moments[k] += count * std::pow(x, k);
            const double moles = mass_scale * moments[1];
            inert -= moles;
            cell.phases.push_back({gas, {moles, 0}, T::p_old, moles * T::cv * T::T0, moles > 0 ? T::T0 : 0.});
            cell.phase_numbers.emplace_back(moments[0]);
            cell.qbmm_shape.emplace_back(R::QbmmShapeState{moments[2], moments[3]});
        }
        eos.push_back(gas);
        controls.phase_roles.push_back({Flow::PhaseRole::BulkGas, "inert"});
        controls.qbmm_families.push_back(nullptr);
        options.phases.push_back({Phase::Resolved, {}, {}});
        state.qbmm_families.emplace_back();
        cell.phases.push_back({gas, {0, inert}, T::p_old, inert * T::cv * T::T0, T::T0});
        cell.phase_numbers.emplace_back();
        cell.qbmm_shape.emplace_back();
        state.cells.resize(mesh->num_owned_cells(), cell);
        flow = std::make_shared<Flow>(mesh, eos, controls);
        flow->initialize(std::move(state));
        transfers.phase_enthalpy.resize(eos.size());
        transfers.phase_enthalpy.front() = T::liquid_partials(catalog, liquid_u, partial_calls);
        const auto slots = std::make_shared<const R::InterphaseConfiguration>(std::move(transfers));
        const R::ThermalLawIdentity identity{"film", 1, catalog->fingerprint()};
        if (mode == Evaluation::Joint)
            options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{identity,
                slots, {}, {}, [this](const R::MixtureQbmmJointContext& context) { return solve_joint(context); }});
        else if (mode == Evaluation::Typed)
        {
            const auto law = std::make_shared<const R::MixtureQbmmLaw>(
                R::MixtureQbmmLaw{law_contract, [this](const R::QbmmLawInputs& input)
                    {
                        ++calls;
                        if (flow->state().time == 0 && input.time_step_s == dt)
                            EXPECT_EQ(input.carrier_liquid_dissolved_moles, std::optional(std::vector{dissolved, 0.}));
                        const auto values = coefficients(1, input);
                        return R::qbmm_implicit_film_transfer(*flow->options().qbmm_families[1], input, tracer, values);
                    }});
            options.qbmm_population =
                std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{identity, slots, {}, {law}});
        }
        else
            options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{
                identity, slots, [this](const R::MixtureQbmmContext& context) { return solve_single(context); }});
        if (moving)
            options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
                R::MixturePlanarMotionProvider{mesh, {}, "expanding film cell", 1, [this](double time, double step)
                    {
                        ++motion_calls;
                        return 1. + .05 * (time + step);
                    }});
        if (configure)
            configure(options);
        controller = std::make_unique<Coupler>(flow, std::move(options));
    }

    Nodes coefficients(size_t phase, const R::QbmmLawInputs& input)
    {
        if (input.quadrature.active_atoms == 0)
            return {};
        frozen_temperature[phase] = input.temperature_K.value();
        frozen_pressure[phase] = input.absolute_pressure_pa;
        return Nodes(input.quadrature.active_atoms, {film_coefficient(input.population), 0});
    }

    Operations solve_single(const R::MixtureQbmmContext& context)
    {
        ++calls;
        const auto& input = *context.law_inputs;
        const double stock = context.material.amount(*catalog, tracer, Phase::Dissolved);
        const auto values = coefficients(context.phase, input);
        if (failure == Failure::Explicit)
        {
            std::vector<double> rates;
            attempted = 0;
            for (const auto& atom : std::span(input.quadrature.atoms).first(input.quadrature.active_atoms))
            {
                const double rate = film_coefficient(input.population) * stock /
                                    input.carrier_liquid_volume_m3.value() *
                                    area(atom.mass_over_scale, input.temperature_K.value(), input.absolute_pressure_pa);
                rates.push_back(rate);
                attempted += context.time_step * atom.count * rate;
            }
            return R::qbmm_resize_from_mass_rate(context.family, input, rates,
                std::vector<R::QbmmTransferCounterpart>(rates.size(), R::QbmmPhaseExchange{Phase::Dissolved}));
        }
        return R::qbmm_implicit_film_transfer(context.family, input, tracer, stock, values);
    }

    std::vector<Operations> solve_joint(const R::MixtureQbmmJointContext& context)
    {
        ++calls;
        EXPECT_TRUE(std::ranges::is_sorted(
            context.families, {}, [](const auto& row) -> const std::string& { return row.family.population(); }));
        if (failure == Failure::WrongRowCount)
            return {};
        const double stock = context.material.amount(*catalog, tracer, Phase::Dissolved);
        std::vector<Nodes> values;
        std::vector<R::QbmmImplicitFilmFamily> families;
        std::vector<Operations> separate;
        attempted = 0;
        for (const auto& row : context.families)
        {
            values.push_back(coefficients(row.phase, row.law_inputs));
            families.push_back({row.family, row.law_inputs, tracer, stock, values.back()});
            if (failure == Failure::SeparateSolves)
            {
                separate.push_back(
                    R::qbmm_implicit_film_transfer(row.family, row.law_inputs, tracer, stock, values.back()));
                for (const auto& op : separate.back())
                    attempted += std::get<R::QbmmResize>(op).transfer.moles[0];
            }
        }
        if (failure == Failure::SeparateSolves)
            return separate;
        return R::qbmm_implicit_film_transfer(families);
    }
};

inline void check_partition(const Fixture& f, const char* ranks_variable = "SIMPLEFLUID_FILM_RANKS")
{
    const auto comm = f.mesh->owned_cell_map()->getComm();
    if (const char* ranks = std::getenv(ranks_variable))
        EXPECT_EQ(comm->getSize(), std::stoi(ranks));
    const int empty = f.mesh->num_owned_cells() == 0;
    int total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, comm->getSize() - 1);
}

inline const R::MaterialThermalReceipt* film_receipt(
    const R::FrozenInterphasePlan& plan, const R::PressureConditionedThermalEvaluation& thermal, size_t recipient)
{
    for (const auto& receipt : thermal.material_events)
    {
        const auto slot = plan.configuration->stages.at(receipt.stage).material.at(receipt.slot);
        if (slot == R::MaterialTransferSlot{tracer, 0, recipient})
            return &receipt;
    }
    return nullptr;
}

} // namespace SimpleFluid::Testing::QbmmFilm
