/** Pressure-conditioned local interphase thermal recovery and conservative receipts. */
#include "equations/PressureConditionedThermal.hh"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace SimpleFluid::Reacting
{
namespace
{
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
/** Feed high AND correction when merging accumulators; never collapse positive totals
 * before a later cancellation. Original event channels remain retained separately.
 */
struct Acc
{
    long double high = 0, low = 0, magnitude = 0, inherited_roundoff = 0;
    std::size_t terms = 0;
    void add(long double value)
    {
        require(std::isfinite(value), "Nonfinite raw thermal operand.");
        const auto next = high + value;
        low += std::abs(high) >= std::abs(value) ? (high - next) + value : (value - next) + high;
        high = next;
        magnitude += std::abs(value);
        ++terms;
        require(
            std::isfinite(high) && std::isfinite(low) && std::isfinite(magnitude), "Raw thermal accumulator overflow.");
    }
    long double error_bound() const
    {
        const long double eps = std::numeric_limits<long double>::epsilon();
        return inherited_roundoff + 16 * eps * eps * static_cast<long double>(terms) * terms * magnitude +
               4 * eps * std::abs(value());
    }
    void append(const Acc& other, long double sign = 1)
    {
        add(sign * other.high);
        add(sign * other.low);
        inherited_roundoff += std::abs(sign) * other.error_bound();
    }
    long double value() const { return high + low; }
};
double narrow(long double value)
{
    const double result = static_cast<double>(value);
    require(std::isfinite(value) && std::isfinite(result) && (value == 0 || result != 0),
        "Thermal quantity is not representable.");
    return result;
}
double narrow_bound(long double value)
{
    require(std::isfinite(value) && value >= 0, "Invalid numerical thermal error bound.");
    double result = static_cast<double>(value);
    if (static_cast<long double>(result) < value)
        result = std::nextafter(result, std::numeric_limits<double>::infinity());
    require(std::isfinite(result), "Thermal error bound is not representable.");
    return result; // Directed rounding of a numerical bound, never a physical energy/count floor.
}
void valid_bounds(const ThermodynamicBounds& b)
{
    require(std::isfinite(b.minimum_temperature) && std::isfinite(b.maximum_temperature) &&
                std::isfinite(b.minimum_pressure) && std::isfinite(b.maximum_pressure) && b.minimum_temperature > 0 &&
                b.minimum_temperature < b.maximum_temperature && b.minimum_pressure > 0 &&
                b.minimum_pressure < b.maximum_pressure,
        "Invalid supplied thermal law bounds.");
}
struct Point
{
    std::vector<double> x;
    long double amount = 0, mass = 0, H = 0, Ht = 0, Hp = 0, V = 0, Vt = 0, Vp = 0, U = 0, caloric = 0;
};
Point point(const PhaseThermalState& phase, std::span<const double> moles, double t, double p)
{
    const auto& eos = *phase.thermodynamics;
    const auto& catalog = eos.catalog();
    require(moles.size() == catalog.species().size(), "Thermal species layout differs.");
    Point result;
    Acc amount, mass;
    for (size_t i = 0; i < moles.size(); ++i)
    {
        require(std::isfinite(moles[i]) && moles[i] >= 0, "Invalid thermal species inventory.");
        amount.add(moles[i]);
        mass.add(static_cast<long double>(moles[i]) * catalog.species()[i].molarMass);
    }
    result.amount = amount.value();
    result.mass = mass.value();
    result.x.assign(moles.size(), 0);
    if (result.amount == 0)
        return result;
    require(result.mass > 0, "Occupied phase mass is not representable.");
    for (size_t i = 0; i < moles.size(); ++i)
        result.x[i] = static_cast<double>(moles[i] / result.amount);
    const auto properties = eos.evaluate(t, p, result.x);
    const long double rho = properties.density, rho2 = rho * rho;
    result.H = result.mass * properties.enthalpy;
    result.U = result.mass * properties.internal_energy;
    result.Ht = result.mass * (properties.internal_energy_temperature - p * properties.density_temperature / rho2);
    result.Hp = result.mass * (properties.internal_energy_pressure + 1 / rho - p * properties.density_pressure / rho2);
    result.V = result.mass / rho;
    result.Vt = -result.mass * properties.density_temperature / rho2;
    result.Vp = -result.mass * properties.density_pressure / rho2;
    result.caloric = std::abs(t * result.mass * properties.internal_energy_temperature);
    require(std::isfinite(result.H) && std::isfinite(result.U) && std::isfinite(result.Ht) && result.Ht > 0 &&
                std::isfinite(result.Hp) && std::isfinite(result.V) && result.V > 0 && std::isfinite(result.Vt) &&
                std::isfinite(result.Vp),
        "Invalid thermal EOS enthalpy/volume derivatives.");
    return result;
}
void endpoint_matches(double applied, double raw)
{
    Acc residual;
    residual.add(applied);
    residual.add(-static_cast<long double>(raw));
    const auto bound = 64 * std::numeric_limits<double>::epsilon() *
                       (std::abs(static_cast<long double>(applied)) + std::abs(static_cast<long double>(raw)));
    require(applied >= 0 && raw >= 0 && (applied == 0) == (raw == 0) && std::abs(residual.value()) <= bound,
        "Applied thermal material endpoint differs from frozen finite events.");
}
/** Persistent predecessor stock for every configured material stage. Rounded
 * phase/EOS views never seed a later donor reservation.
 */
struct MaterialStocks
{
    const SpeciesCatalog& catalog;
    std::vector<std::vector<Acc>> amounts;

    MaterialStocks(const SpeciesCatalog& schema, std::span<const std::vector<double>> before,
        std::span<const std::vector<std::array<long double, 2>>> raw = {})
        : catalog(schema), amounts(before.size())
    {
        require(raw.empty() || raw.size() == before.size(), "Raw thermal predecessor phase layout differs.");
        for (size_t p = 0; p < before.size(); ++p)
        {
            require(before[p].size() == catalog.species().size(), "Material donor species layout differs.");
            require(raw.empty() || raw[p].size() == before[p].size(),
                "Raw thermal predecessor species layout differs.");
            amounts[p].resize(before[p].size());
            for (size_t s = 0; s < before[p].size(); ++s)
            {
                require(std::isfinite(before[p][s]) && before[p][s] >= 0, "Invalid material predecessor stock.");
                if (raw.empty())
                    amounts[p][s].add(before[p][s]);
                else
                {
                    amounts[p][s].add(raw[p][s][0]);
                    amounts[p][s].add(raw[p][s][1]);
                    require(amounts[p][s].value() >= 0 && narrow(amounts[p][s].value()) == before[p][s],
                        "Raw thermal predecessor does not project to its actual represented phase.");
                }
            }
        }
    }

    std::vector<std::vector<double>> project() const
    {
        std::vector<std::vector<double>> result(amounts.size());
        for (size_t p = 0; p < amounts.size(); ++p)
        {
            result[p].resize(amounts[p].size());
            for (size_t s = 0; s < amounts[p].size(); ++s)
            {
                result[p][s] = narrow(amounts[p][s].value());
                require(result[p][s] >= 0, "Negative interphase material endpoint.");
            }
        }
        return result;
    }

    std::vector<std::vector<double>> apply(const ConfiguredInterphaseStage& stage, std::span<const double> events)
    {
        require(stage.kind == InterphaseStageKind::Material && stage.material.size() == events.size(),
            "Material stage layout differs.");
        auto margins = amounts;
        auto next = amounts;
        for (size_t i = 0; i < events.size(); ++i)
        {
            require(
                std::isfinite(events[i]) && events[i] >= 0, "Material event amount must be finite nonnegative mol.");
            const auto& slot = stage.material[i];
            const auto s = catalog.index(slot.species);
            margins.at(slot.donor).at(s).add(-static_cast<long double>(events[i]));
            next.at(slot.donor).at(s).add(-static_cast<long double>(events[i]));
            next.at(slot.recipient).at(s).add(events[i]);
        }
        for (const auto& phase : margins)
            for (const auto& stock : phase)
                require(
                    stock.value() >= 0, "Interphase gross outgoing events overdraw their actual predecessor donor.");
        amounts = std::move(next);
        return project();
    }
};
} // namespace
void validate_interphase_configuration(const InterphaseConfiguration& config, std::span<const PhaseThermalState> phases)
{
    require(!phases.empty() && !config.metadata.identity.empty() && config.metadata.revision > 0 &&
                config.phase_enthalpy.size() == phases.size() && !config.stages.empty(),
        "Invalid immutable interphase configuration.");
    require(phases.front().thermodynamics != nullptr, "Interphase configuration lacks an EOS.");
    const auto& catalog = phases.front().thermodynamics->catalog();
    require(config.metadata.schema_fingerprint == catalog.fingerprint(), "Interphase configuration schema differs.");
    for (size_t p = 0; p < phases.size(); ++p)
    {
        require(phases[p].thermodynamics && &phases[p].thermodynamics->catalog() == &catalog &&
                    phases[p].thermodynamics->energy_convention() == phases.front().thermodynamics->energy_convention(),
            "Interphase phases require one catalog and formation convention.");
        if (config.phase_enthalpy[p])
        {
            const auto& law = *config.phase_enthalpy[p];
            valid_bounds(law.bounds);
            require(!law.metadata.identity.empty() && law.metadata.revision > 0 &&
                        law.metadata.schema_fingerprint == catalog.fingerprint() && law.evaluate,
                "Invalid immutable partial-h law.");
        }
    }
    if (config.caloric_domain)
    {
        const auto& law = *config.caloric_domain;
        require(!law.metadata.identity.empty() && law.metadata.revision > 0 &&
                    law.metadata.schema_fingerprint == catalog.fingerprint() && law.evaluate,
            "Invalid whole-plan caloric domain provider metadata.");
    }
    std::set<std::string> ids;
    for (const auto& stage : config.stages)
    {
        require(!stage.identity.empty() && ids.insert(stage.identity).second,
            "Interphase stage identities must be explicit and unique.");
        require(stage.kind == InterphaseStageKind::Heat || stage.kind == InterphaseStageKind::Material,
            "Unknown interphase stage kind.");
        if (stage.kind == InterphaseStageKind::Heat)
        {
            require(stage.material.empty() && !stage.heat.empty(), "Heat stage cannot alter material bindings.");
            for (const auto& slot : stage.heat)
                require(slot.first < phases.size() && slot.second < phases.size() && slot.first != slot.second,
                    "Invalid configured heat endpoints.");
        }
        else
        {
            require(stage.heat.empty() && !stage.material.empty(), "Material stage cannot alter heat bindings.");
            for (const auto& slot : stage.material)
            {
                (void) catalog.index(slot.species);
                require(slot.donor < phases.size() && slot.recipient < phases.size() && slot.donor != slot.recipient &&
                            config.phase_enthalpy[slot.donor],
                    "Material donor requires valid endpoints and a supplied partial-h law.");
            }
        }
    }
}
FrozenInterphasePlan compile_interphase_plan(std::shared_ptr<const InterphaseConfiguration> configuration,
    std::span<const PhaseThermalState> pre_phases, std::vector<std::vector<double>> amounts,
    std::vector<std::vector<long double>> raw_static_energy, std::vector<long double> raw_declared_energy)
{
    std::vector<std::vector<std::array<long double, 2>>> represented;
    represented.reserve(pre_phases.size());
    for (const auto& phase : pre_phases)
    {
        auto& parts = represented.emplace_back();
        for (double value : phase.moles)
            parts.push_back({value, 0});
    }
    return compile_interphase_plan(std::move(configuration), pre_phases, std::move(amounts),
        std::move(raw_static_energy), std::move(raw_declared_energy), represented);
}

FrozenInterphasePlan compile_interphase_plan(std::shared_ptr<const InterphaseConfiguration> configuration,
    std::span<const PhaseThermalState> pre_phases, std::vector<std::vector<double>> amounts,
    std::vector<std::vector<long double>> raw_static_energy, std::vector<long double> raw_declared_energy,
    std::span<const std::vector<std::array<long double, 2>>> predecessor_parts)
{
    require(configuration != nullptr, "Missing immutable interphase configuration.");
    require(predecessor_parts.size() == pre_phases.size(), "Raw thermal predecessor requires every actual phase.");
    validate_interphase_configuration(*configuration, pre_phases);
    require(amounts.size() == configuration->stages.size() && raw_static_energy.size() == pre_phases.size(),
        "Interphase stage/energy layout differs.");
    FrozenInterphasePlan result;
    result.configuration = std::move(configuration);
    result.amounts = std::move(amounts);
    result.raw_static_energy = std::move(raw_static_energy);
    result.raw_declared_energy = std::move(raw_declared_energy);
    result.raw_pre_moles.assign(predecessor_parts.begin(), predecessor_parts.end());
    Acc allocated, declared;
    for (size_t p = 0; p < pre_phases.size(); ++p)
    {
        result.pre_moles.push_back(pre_phases[p].moles);
        for (auto e : result.raw_static_energy[p])
            allocated.add(e);
    }
    for (auto e : result.raw_declared_energy)
        declared.add(e);
    Acc mismatch;
    mismatch.append(allocated);
    mismatch.append(declared, -1);
    require(std::abs(mismatch.value()) <=
                64 * std::numeric_limits<double>::epsilon() * (allocated.magnitude + declared.magnitude),
        "Raw allocated static energy differs from its declared physical channels.");
    MaterialStocks stocks(pre_phases.front().thermodynamics->catalog(), result.pre_moles, result.raw_pre_moles);
    result.final_moles = stocks.project();
    for (size_t j = 0; j < result.amounts.size(); ++j)
    {
        const auto& stage = result.configuration->stages[j];
        if (stage.kind == InterphaseStageKind::Material)
            result.final_moles = stocks.apply(stage, result.amounts[j]);
        else
        {
            require(result.amounts[j].size() == stage.heat.size(), "Configured heat amount layout differs.");
            for (double q : result.amounts[j])
                require(std::isfinite(q), "Nonfinite supplied integrated interphase heat.");
        }
    }
    return result;
}

PressureConditionedThermalEvaluation evaluate_pressure_conditioned_thermal(const FrozenInterphasePlan& plan,
    std::span<const PhaseThermalState> final_templates, std::span<const double> advected_energy,
    std::span<const double> advected_volumes, double pressure, PhaseThermalMode mode,
    ThermalEvaluationControls controls)
{
    require(plan.configuration != nullptr, "Missing frozen thermal plan.");
    validate_interphase_configuration(*plan.configuration, final_templates);
    const auto np = final_templates.size();
    const auto& catalog = final_templates.front().thermodynamics->catalog();
    require(advected_energy.size() == np && advected_volumes.size() == np && plan.pre_moles.size() == np &&
                plan.final_moles.size() == np && plan.raw_static_energy.size() == np &&
                plan.amounts.size() == plan.configuration->stages.size(),
        "Conditioned thermal owned-cell phase layout differs.");
    require(mode == PhaseThermalMode::Equilibrium || mode == PhaseThermalMode::SeparatePhaseTemperatures,
        "Invalid thermal mode.");
    require(std::isfinite(pressure) && pressure > 0 && std::isfinite(controls.physical_relative_tolerance) &&
                controls.physical_relative_tolerance > 0 && std::isfinite(controls.caloric_relative_tolerance) &&
                controls.caloric_relative_tolerance > 0,
        "Invalid conditioned thermal pressure/accuracy controls.");
    PressureConditionedThermalEvaluation result;
    result.absolute_pressure = pressure;
    result.phases.assign(final_templates.begin(), final_templates.end());
    result.volumes.resize(np);
    result.applied_phase_energy.resize(np);
    result.raw_applied_phase_energy.resize(np);
    result.raw_material_enthalpy.resize(np);
    result.raw_conductive_heat.resize(np);
    result.raw_equilibrium_redistribution.resize(np);
    std::vector<Acc> sources(np), targets(np), material(np), heat(np), equilibrium(np), empty_roundoff(np);
    std::vector<long double> target_derivative(np), last_caloric(np), temperature_derivative(np);
    MaterialStocks stocks(catalog, plan.pre_moles, plan.raw_pre_moles);
    auto moles = stocks.project();
    for (size_t a = 0; a < np; ++a)
    {
        require(std::isfinite(advected_energy[a]) && std::isfinite(advected_volumes[a]) && advected_volumes[a] >= 0,
            "Invalid actual advected energy/work volume.");
        require(
            final_templates[a].moles.size() == plan.final_moles[a].size(), "Final thermal material layout differs.");
        for (size_t i = 0; i < plan.final_moles[a].size(); ++i)
            endpoint_matches(final_templates[a].moles[i], plan.final_moles[a][i]);
        if (std::none_of(plan.pre_moles[a].begin(), plan.pre_moles[a].end(), [](double n) { return n > 0; }) &&
            advected_volumes[a] != 0)
            throw std::invalid_argument("Upstream depleted phase has unowned pressure-dependent volume work; a "
                                        "chemistry/face thermal trace is required.");
        targets[a].add(advected_energy[a]);
        targets[a].add(static_cast<long double>(pressure) * advected_volumes[a]);
        result.pressure_operands += std::abs(pressure * advected_volumes[a]);
        for (auto value : plan.raw_static_energy[a])
            sources[a].add(value);
        targets[a].append(sources[a]);
        target_derivative[a] = advected_volumes[a];
    }
    std::vector<Point> points(np);
    auto recover = [&]()
    {
        std::vector<int> occupied(np, 0);
        double common_lower = 0, common_upper = std::numeric_limits<double>::infinity();
        size_t active = 0;
        for (size_t a = 0; a < np; ++a)
        {
            for (double n : moles[a])
                require(std::isfinite(n) && n >= 0, "Invalid staged thermal material stock.");
            occupied[a] = std::any_of(moles[a].begin(), moles[a].end(), [](double n) { return n > 0; });
            if (occupied[a])
            {
                ++active;
                const auto& b = final_templates[a].thermodynamics->bounds();
                require(pressure >= b.minimum_pressure && pressure <= b.maximum_pressure,
                    "Pressure is outside occupied anchor EOS bounds.");
                common_lower = std::max(common_lower, b.minimum_temperature);
                common_upper = std::min(common_upper, b.maximum_temperature);
            }
            else
            {
                const long double value = targets[a].value();
                const long double bound =
                    32 * std::numeric_limits<double>::epsilon() * targets[a].magnitude + targets[a].error_bound();
                require(std::abs(value) <= bound, "Empty staged phase has unallocated physical enthalpy/work.");
                if (value != 0)
                {
                    require(
                        last_caloric[a] > 0 && bound <= .25L * controls.physical_relative_tolerance * last_caloric[a],
                        "Empty-phase cancellation is too ill-conditioned for requested caloric accuracy.");
                    empty_roundoff[a].add(-value);
                    targets[a].add(-value);
                }
                result.phases[a].moles = moles[a];
                result.phases[a].temperature = 0;
                result.phases[a].internal_energy = 0;
                result.phases[a].absolute_pressure = pressure;
                temperature_derivative[a] = 0;
                points[a] = {};
            }
        }
        require(active > 0, "A closed control volume has no occupied caloric phase.");
        auto invert = [&](size_t selected, double lo, double hi)
        {
            if (lo > hi)
                throw ThermalCaloricRangeError("Thermal anchors lack a common supplied temperature interval.");
            Acc target;
            for (size_t a = 0; a < np; ++a)
                if (occupied[a] && (selected == np || selected == a))
                    target.append(targets[a]);
            auto enthalpy = [&](double t, long double* physical = nullptr)
            {
                Acc value;
                long double scale = 0;
                for (size_t a = 0; a < np; ++a)
                    if (occupied[a] && (selected == np || selected == a))
                    {
                        const auto state = point(final_templates[a], moles[a], t, pressure);
                        value.add(state.H);
                        scale += std::abs(t * state.Ht);
                    }
                if (physical)
                    *physical = scale;
                return value;
            };
            const auto low = enthalpy(lo), high = enthalpy(hi);
            if (target.value() < low.value() || target.value() > high.value())
                throw ThermalCaloricRangeError(
                    "Conditioned phase enthalpy is outside supplied caloric validity bounds.");
            for (int iteration = 0; iteration < 180; ++iteration)
            {
                const double t = std::midpoint(lo, hi);
                long double physical = 0;
                const auto value = enthalpy(t, &physical);
                Acc residual;
                residual.append(value);
                residual.append(target, -1);
                const auto roundoff = 32 * std::numeric_limits<double>::epsilon() * std::abs(value.value()) +
                                      value.error_bound() + target.error_bound() + residual.error_bound();
                require(roundoff <=
                            .25L * std::min(controls.physical_relative_tolerance, controls.caloric_relative_tolerance) *
                                physical,
                    "Caloric reference/operands cannot resolve requested conditioned temperature accuracy.");
                // Operand roundoff bounds certify conditioning; they must not become a reference-dependent
                // temperature stopping tolerance. Resolve the supplied monotone EOS to its representable bracket.
                if (residual.value() == 0 || t == lo || t == hi)
                    return t;
                if (residual.value() < 0)
                    lo = t;
                else
                    hi = t;
            }
            throw std::runtime_error("Bounded conditioned enthalpy inversion did not converge.");
        };
        const double common = mode == PhaseThermalMode::Equilibrium ? invert(np, common_lower, common_upper) : 0;
        long double Ht = 0, Hp = 0, derivative = 0;
        for (size_t a = 0; a < np; ++a)
            if (occupied[a])
            {
                const auto& b = final_templates[a].thermodynamics->bounds();
                const double t = mode == PhaseThermalMode::Equilibrium
                                     ? common
                                     : invert(a, b.minimum_temperature, b.maximum_temperature);
                points[a] = point(final_templates[a], moles[a], t, pressure);
                last_caloric[a] = points[a].caloric;
                result.phases[a].moles = moles[a];
                result.phases[a].temperature = t;
                result.phases[a].absolute_pressure = pressure;
                result.phases[a].internal_energy = narrow(points[a].U);
                Ht += points[a].Ht;
                Hp += points[a].Hp;
                derivative += target_derivative[a];
                if (mode == PhaseThermalMode::SeparatePhaseTemperatures)
                    temperature_derivative[a] = (target_derivative[a] - points[a].Hp) / points[a].Ht;
            }
        if (mode == PhaseThermalMode::Equilibrium)
        {
            const auto common_derivative = (derivative - Hp) / Ht;
            for (size_t a = 0; a < np; ++a)
                if (occupied[a])
                {
                    temperature_derivative[a] = common_derivative;
                    Acc allocation;
                    allocation.add(points[a].H);
                    allocation.append(targets[a], -1);
                    // Merge raw high/correction, not a prematurely collapsed allocation endpoint.
                    sources[a].append(allocation);
                    equilibrium[a].append(allocation);
                    targets[a] = {};
                    targets[a].add(points[a].H);
                    target_derivative[a] = points[a].Ht * common_derivative + points[a].Hp;
                }
        }
        ThermalAnchorReceipt anchor;
        anchor.temperature.resize(np);
        anchor.enthalpy.resize(np);
        anchor.temperature_pressure_derivative.resize(np);
        for (size_t a = 0; a < np; ++a)
        {
            require(std::isfinite(temperature_derivative[a]), "Nonfinite full thermal pressure tangent.");
            anchor.temperature_pressure_derivative[a] = narrow(temperature_derivative[a]);
            anchor.temperature[a] = result.phases[a].temperature;
            anchor.enthalpy[a] = points[a].H;
        }
        result.anchors.push_back(std::move(anchor));
    };
    recover();
    for (size_t j = 0; j < plan.amounts.size(); ++j)
    {
        const auto& stage = plan.configuration->stages[j];
        const auto& amounts = plan.amounts[j];
        if (stage.kind == InterphaseStageKind::Heat)
        {
            require(amounts.size() == stage.heat.size(), "Frozen heat slots changed.");
            // Gross outgoing sensible heat reserves actual bounded predecessor capacity; incoming heat
            // cannot finance it. This is invariant to additive caloric references, including negative U.
            std::vector<Acc> heat_margin(np), heat_headroom(np);
            std::vector<int> heat_active(np, 0);
            for (size_t i = 0; i < amounts.size(); ++i)
            {
                const double q = amounts[i];
                require(std::isfinite(q), "Nonfinite frozen interphase heat.");
                if (q == 0)
                    continue;
                const auto& slot = stage.heat[i];
                const size_t donor = q > 0 ? slot.first : slot.second, receiver = q > 0 ? slot.second : slot.first;
                for (const size_t a : {donor, receiver})
                    if (!heat_active[a])
                    {
                        require(points[a].amount > 0, "Heat reservation needs an actual occupied predecessor phase.");
                        const auto& bounds = final_templates[a].thermodynamics->bounds();
                        const auto lower = point(final_templates[a], moles[a], bounds.minimum_temperature, pressure);
                        const auto upper = point(final_templates[a], moles[a], bounds.maximum_temperature, pressure);
                        heat_margin[a].add(points[a].H);
                        heat_margin[a].add(-lower.H);
                        heat_headroom[a].add(upper.H);
                        heat_headroom[a].add(-points[a].H);
                        heat_active[a] = 1;
                    }
                heat_margin[donor].add(-std::abs(static_cast<long double>(q)));
                heat_headroom[receiver].add(-std::abs(static_cast<long double>(q)));
            }
            for (size_t a = 0; a < np; ++a)
                if (heat_active[a] && (heat_margin[a].value() < 0 || heat_headroom[a].value() < 0))
                    throw ThermalCaloricRangeError(
                        "Joint interphase heat lies outside bounded donor/receiver caloric capacity.");
            for (size_t i = 0; i < amounts.size(); ++i)
            {
                const auto& slot = stage.heat[i];
                const long double q = amounts[i];
                require(std::isfinite(q), "Nonfinite frozen interphase heat.");
                if (q == 0)
                    continue;
                require(points[slot.first].amount > 0 && points[slot.second].amount > 0,
                    "Heat exchange needs both actual occupied predecessor phases; no absent phase temperature is "
                    "invented.");
                targets[slot.first].add(-q);
                targets[slot.second].add(q);
                sources[slot.first].add(-q);
                sources[slot.second].add(q);
                heat[slot.first].add(-q);
                heat[slot.second].add(q);
            }
        }
        else
        {
            const auto after = stocks.apply(stage, amounts);
            std::vector<PartialMolarEnthalpyPoint> properties(np);
            std::vector<int> evaluated(np, 0);
            for (size_t i = 0; i < amounts.size(); ++i)
            {
                if (amounts[i] == 0)
                    continue;
                const auto& slot = stage.material[i];
                const auto a = slot.donor;
                if (!evaluated[a])
                {
                    require(points[a].amount > 0, "Material donor has no actual anchor stock.");
                    const auto& law = *plan.configuration->phase_enthalpy[a];
                    const auto t = result.phases[a].temperature;
                    if (t < law.bounds.minimum_temperature || t > law.bounds.maximum_temperature ||
                        pressure < law.bounds.minimum_pressure || pressure > law.bounds.maximum_pressure)
                        throw ThermalCaloricRangeError(
                            "Occupied donor anchor lies outside supplied partial-h law domain.");
                    properties[a] = law.evaluate(t, pressure, points[a].x);
                    const auto& h = properties[a];
                    require(h.value.size() == catalog.species().size() && h.temperature.size() == h.value.size() &&
                                h.pressure.size() == h.value.size(),
                        "Supplied partial-h species/derivative layout differs.");
                    Acc euler, temperature, pressure_partial;
                    for (size_t k = 0; k < h.value.size(); ++k)
                    {
                        require(std::isfinite(h.value[k]) && std::isfinite(h.temperature[k]) &&
                                    std::isfinite(h.pressure[k]),
                            "Nonfinite supplied partial enthalpy or derivative.");
                        euler.add(static_cast<long double>(moles[a][k]) * h.value[k]);
                        temperature.add(static_cast<long double>(moles[a][k]) * h.temperature[k]);
                        pressure_partial.add(static_cast<long double>(moles[a][k]) * h.pressure[k]);
                    }
                    const auto compatible = [&](const Acc& sum, long double expected, long double physical)
                    {
                        Acc residual;
                        residual.append(sum);
                        residual.add(-expected);
                        const auto roundoff = 64 * std::numeric_limits<double>::epsilon() *
                                                  std::max(std::abs(expected), std::abs(sum.value())) +
                                              residual.error_bound();
                        require(roundoff <= .25L * controls.physical_relative_tolerance * physical,
                            "Partial-h consistency is too poorly conditioned for requested caloric accuracy.");
                        require(
                            std::abs(residual.value()) <= controls.physical_relative_tolerance * physical + roundoff,
                            "Supplied partial-h violates full donor Euler or temperature/pressure derivative "
                            "identity.");
                    };
                    compatible(euler, points[a].H, points[a].caloric);
                    compatible(temperature, points[a].Ht, points[a].Ht);
                    // A genuinely zero pressure derivative must be zero within its independent operand roundoff.
                    compatible(pressure_partial, points[a].Hp, std::max(std::abs(points[a].Hp), std::abs(points[a].V)));
                    evaluated[a] = 1;
                }
                const auto species = catalog.index(slot.species);
                const auto& h = properties[a];
                const long double energy = static_cast<long double>(amounts[i]) * h.value[species];
                const long double tangent = static_cast<long double>(amounts[i]) *
                                            (h.temperature[species] * temperature_derivative[a] + h.pressure[species]);
                targets[a].add(-energy);
                targets[slot.recipient].add(energy);
                sources[a].add(-energy);
                sources[slot.recipient].add(energy);
                material[a].add(-energy);
                material[slot.recipient].add(energy);
                target_derivative[a] -= tangent;
                target_derivative[slot.recipient] += tangent;
                result.material_events.push_back({j, i, h.value[species], h.temperature[species], h.pressure[species],
                    narrow(temperature_derivative[a]), energy, tangent});
            }
            moles = after;
        }
        if (j + 1 == plan.amounts.size())
        {
            // Retain the exact applied source-packet material projection; its raw endpoint was independently checked.
            for (size_t a = 0; a < np; ++a)
                moles[a] = final_templates[a].moles;
        }
        recover();
    }
    Acc energy_error, total_roundoff, operand_bound, caloric, work, work_magnitude, pressure_operands;
    Acc applied_total, declared_total, material_pair, heat_pair, equilibrium_pair;
    Acc raw_roundoff, volume_derivative;
    for (auto e : plan.raw_declared_energy)
        declared_total.add(e);
    for (size_t a = 0; a < np; ++a)
    {
        result.volumes[a] = narrow(points[a].V);
        result.total_volume += result.volumes[a];
        volume_derivative.add(points[a].Vp + points[a].Vt * temperature_derivative[a]);
        result.applied_phase_energy[a] = narrow(sources[a].value());
        result.raw_applied_phase_energy[a] = {sources[a].high, sources[a].low};
        raw_roundoff.add(sources[a].error_bound());
        applied_total.append(sources[a]);
        material_pair.append(material[a]);
        heat_pair.append(heat[a]);
        equilibrium_pair.append(equilibrium[a]);
        result.raw_material_enthalpy[a] = material[a].value();
        result.raw_conductive_heat[a] = heat[a].value();
        result.raw_equilibrium_redistribution[a] = equilibrium[a].value();
        const long double w = -static_cast<long double>(pressure) * (points[a].V - advected_volumes[a]);
        work.add(w);
        work_magnitude.add(std::abs(w));
        pressure_operands.add(std::abs(pressure * points[a].V));
        caloric.add(points[a].caloric);
        energy_error.add(result.phases[a].internal_energy);
        energy_error.add(-static_cast<long double>(advected_energy[a]));
        energy_error.append(sources[a], -1);
        energy_error.add(-w);
        energy_error.append(empty_roundoff[a], -1);
        total_roundoff.append(empty_roundoff[a]);
        operand_bound.add(
            sources[a].error_bound() + 32 * std::numeric_limits<double>::epsilon() *
                                           (std::abs(static_cast<long double>(result.phases[a].internal_energy)) +
                                               std::abs(static_cast<long double>(advected_energy[a])) +
                                               std::abs(sources[a].value()) + std::abs(w)));
    }
    const long double physical_budget =
        controls.physical_relative_tolerance * (caloric.value() + work_magnitude.value());
    Acc declared_error;
    declared_error.append(applied_total);
    declared_error.append(declared_total, -1);
    require(std::abs(material_pair.value()) <= material_pair.error_bound() &&
                std::abs(heat_pair.value()) <= heat_pair.error_bound(),
        "Interphase material enthalpy/heat ledger is not paired conservative.");
    require(std::abs(equilibrium_pair.value()) <= physical_budget + operand_bound.value() &&
                std::abs(declared_error.value()) <= physical_budget + operand_bound.value(),
        "Conditioned equilibrium allocation changed declared physical energy.");
    require(operand_bound.value() <= .25L * physical_budget,
        "Conditioned thermal reference cannot resolve requested physical energy accuracy.");
    require(
        std::abs(energy_error.value()) <=
            controls.physical_relative_tolerance * (caloric.value() + work_magnitude.value()) + operand_bound.value(),
        "Conditioned phase thermal/work budget failed.");
    result.volume_pressure_derivative = narrow(volume_derivative.value());
    require(std::isfinite(result.volume_pressure_derivative), "Invalid full conditioned volume-pressure tangent.");
    result.pressure_work = narrow(work.value());
    result.work_magnitude = narrow(work_magnitude.value());
    result.pressure_operands += narrow(pressure_operands.value());
    result.caloric_scale = narrow(caloric.value());
    result.empty_phase_energy_roundoff = narrow(total_roundoff.value());
    result.energy_operand_roundoff_bound = narrow_bound(operand_bound.value());
    result.raw_channel_roundoff_bound = narrow(raw_roundoff.value());
    return result;
}
CaloricPressureInterval interphase_pressure_bounds(
    const FrozenInterphasePlan& plan, std::span<const PhaseThermalState> phases)
{
    require(plan.configuration != nullptr, "Missing frozen pressure-domain plan.");
    validate_interphase_configuration(*plan.configuration, phases);
    CaloricPressureInterval result{0, std::numeric_limits<double>::infinity()};
    const auto& catalog = phases.front().thermodynamics->catalog();
    MaterialStocks stocks(catalog, plan.pre_moles, plan.raw_pre_moles);
    auto moles = stocks.project();
    auto include_phases = [&]()
    {
        for (size_t a = 0; a < phases.size(); ++a)
            if (std::any_of(moles[a].begin(), moles[a].end(), [](double n) { return n > 0; }))
            {
                const auto& b = phases[a].thermodynamics->bounds();
                result.lower = std::max(result.lower, b.minimum_pressure);
                result.upper = std::min(result.upper, b.maximum_pressure);
            }
    };
    include_phases();
    for (size_t j = 0; j < plan.amounts.size(); ++j)
    {
        const auto& stage = plan.configuration->stages[j];
        if (stage.kind == InterphaseStageKind::Material)
        {
            for (size_t i = 0; i < stage.material.size(); ++i)
                if (plan.amounts[j][i] > 0)
                {
                    const auto& b = plan.configuration->phase_enthalpy[stage.material[i].donor]->bounds;
                    result.lower = std::max(result.lower, b.minimum_pressure);
                    result.upper = std::min(result.upper, b.maximum_pressure);
                }
            moles = stocks.apply(stage, plan.amounts[j]);
        }
        include_phases();
    }
    require(std::isfinite(result.lower) && std::isfinite(result.upper) && result.lower <= result.upper,
        "Occupied thermal anchors/laws have no common pressure domain.");
    return result;
}
} // namespace SimpleFluid::Reacting
