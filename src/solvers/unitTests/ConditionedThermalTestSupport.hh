/** Synthetic independent thermodynamic fixtures for local transfer and flow contracts. */
#pragma once
#include "equations/PressureConditionedThermal.hh"
#include <numeric>
namespace SimpleFluid::Testing::ConditionedTransfer
{
namespace R = Reacting;
inline constexpr double gas_constant = 8.31446261815324, cv = 20, cp = cv + gas_constant, p_old = 100000, T0 = 300;
struct Fixture
{
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    std::shared_ptr<const R::InterphaseConfiguration> configuration;
    std::vector<R::PhaseThermalState> before;
    std::vector<double> Uadv, Vadv{.4, 0, .6};
    double nl = .4, ng = p_old * .6 / (gas_constant * T0), physical_U0 = (cv / gas_constant) * p_old * 1.4;
    int constitutive_calls = 0;
    explicit Fixture(double vapor_reference_shift = 0, double inert_reference_shift = 0)
    {
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {44, "solvent_vapor", 1, {{"S", 1}}, {R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed}},
            {709, "retained_noncondensable", .028, {{"G", 1}},
                {R::SpeciesPhase::Dissolved, R::SpeciesPhase::Resolved}}});
        const double liquid_e = (2 * (cv / gas_constant) * p_old * .4) / .4 + vapor_reference_shift;
        liquid = R::make_linear_compressible_mixture_liquid(catalog, {250, 1200, 50000, 1000000},
            [liquid_e](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, p_old, 1, 1000, 0, 0, liquid_e, 0, 0}; });
        gas = R::make_ideal_gas_mixture(catalog,
            {{cv, 300, cv * 300 + vapor_reference_shift}, {cv, 300, cv * 300 + inert_reference_shift}},
            {250, 1200, 50000, 1000000});
        before = {{liquid, {nl, 0}, p_old, .4 * liquid_e, 300}, {gas, {0, 0}, p_old, 0, 0},
            {gas, {0, ng}, p_old, ng * (cv * 300 + inert_reference_shift), 300}};
        for (const auto& phase : before)
            Uadv.push_back(phase.internal_energy);
        auto partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(
            R::PartialMolarEnthalpyLaw{{"analytic-incompressible-liquid-partials", 1, catalog->fingerprint()},
                {250, 1200, 50000, 1000000}, [this, liquid_e](double t, double p, std::span<const double>)
                {
                    ++constitutive_calls;
                    R::PartialMolarEnthalpyPoint result;
                    const double h = liquid_e + 1000 * (t - 300) + p;
                    for (const auto& species : catalog->species())
                    {
                        result.value.push_back(species.molarMass * h);
                        result.temperature.push_back(species.molarMass * 1000);
                        result.pressure.push_back(species.molarMass);
                    }
                    return result;
                }});
        R::InterphaseConfiguration config;
        config.metadata = {"frozen-full-vapor-transfer", 1, catalog->fingerprint()};
        config.phase_enthalpy = {partial, nullptr, nullptr};
        config.stages = {{"declared-liquid-to-vapor", R::InterphaseStageKind::Material, {{44, 0, 1}}, {}}};
        configuration = std::make_shared<const R::InterphaseConfiguration>(std::move(config));
    }
    R::FrozenInterphasePlan plan(double heat = 0, double amount = -1) const
    {
        return R::compile_interphase_plan(
            configuration, before, {{amount < 0 ? nl : amount}}, {{heat}, {}, {}}, {heat});
    }
    std::vector<R::PhaseThermalState> final_templates(const R::FrozenInterphasePlan& plan) const
    {
        auto phases = before;
        for (size_t p = 0; p < phases.size(); ++p)
            phases[p].moles = plan.final_moles[p];
        return phases;
    }
    R::PressureConditionedThermalEvaluation evaluate(const R::FrozenInterphasePlan& plan, double p,
        R::PhaseThermalMode mode = R::PhaseThermalMode::Equilibrium) const
    {
        return R::evaluate_pressure_conditioned_thermal(plan, final_templates(plan), Uadv, Vadv, p, mode);
    }
    double root(const R::FrozenInterphasePlan& plan) const
    {
        double p = p_old;
        for (int iteration = 0; iteration < 30; ++iteration)
        {
            const auto result = evaluate(plan, p);
            if (std::abs(result.total_volume - 1) < 1e-12)
                return p;
            if (!(result.volume_pressure_derivative < 0))
                throw std::runtime_error("Independent test solve received unstable tangent.");
            p -= (result.total_volume - 1) / result.volume_pressure_derivative;
        }
        throw std::runtime_error("Independent test Newton solve did not close.");
    }
};

} // namespace SimpleFluid::Testing::ConditionedTransfer
