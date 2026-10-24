#include "equations/ReactingThermodynamics.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace SimpleFluid::Reacting
{
namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

class CaloricRangeError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

bool finite(double value)
{
    return std::isfinite(value);
}

void validate_bounds(const ThermodynamicBounds& b)
{
    require(finite(b.minimum_temperature) && finite(b.maximum_temperature) && finite(b.minimum_pressure) &&
                finite(b.maximum_pressure) && b.minimum_temperature > 0 &&
                b.maximum_temperature > b.minimum_temperature && b.minimum_pressure > 0 &&
                b.maximum_pressure > b.minimum_pressure,
        "Mixture EOS requires finite positive temperature/pressure validity intervals.");
}

bool close(double a, double b, double tolerance = 2e-10)
{
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

struct Composition
{
    std::vector<double> fractions;
    double mass = 0, amount = 0;
};

Composition composition(const SpeciesCatalog& catalog, std::span<const double> moles)
{
    require(moles.size() == catalog.species().size(), "Phase composition does not match its species catalog.");
    Composition result;
    result.fractions.resize(moles.size());
    for (std::size_t i = 0; i < moles.size(); ++i)
    {
        require(finite(moles[i]) && moles[i] >= 0, "Phase moles must be finite and nonnegative.");
        result.amount += moles[i];
        result.mass += moles[i] * catalog.species()[i].molarMass;
    }
    require(finite(result.amount) && finite(result.mass), "Phase mass/amount overflow.");
    if (result.amount > 0)
        for (std::size_t i = 0; i < moles.size(); ++i)
            result.fractions[i] = moles[i] / result.amount;
    return result;
}

void validate_phase(const PhaseThermalState& phase)
{
    require(static_cast<bool>(phase.thermodynamics), "Phase EOS is null.");
    require(finite(phase.internal_energy), "Phase internal energy must be finite.");
    const auto c = composition(phase.thermodynamics->catalog(), phase.moles);
    require(c.amount > 0 || (phase.internal_energy == 0 && phase.temperature == 0),
        "An empty phase must have exactly zero energy and temperature.");
    const auto& b = phase.thermodynamics->bounds();
    require(finite(phase.absolute_pressure) && phase.absolute_pressure >= b.minimum_pressure &&
                phase.absolute_pressure <= b.maximum_pressure,
        "Phase pressure is outside its EOS range.");
}

// Monotone caloric law, guaranteed by every evaluated positive de/dT. No extrapolation or floors.
template<class Energy> double invert_energy(double lower, double upper, double target, Energy energy)
{
    require(lower <= upper, "Occupied phases have no common temperature validity interval.");
    const double low_energy = energy(lower), high_energy = energy(upper);
    require(finite(low_energy) && finite(high_energy) && high_energy >= low_energy,
        "Caloric inversion has invalid endpoint energies.");
    const double tolerance =
        32 * std::numeric_limits<double>::epsilon() *
        std::max({std::numeric_limits<double>::min(), std::abs(target), std::abs(low_energy), std::abs(high_energy)});
    if (target < low_energy || target > high_energy)
    {
        if (!(target >= low_energy - tolerance && target <= high_energy + tolerance))
            throw CaloricRangeError("Requested phase energy is outside the declared temperature validity interval.");
        return target < low_energy ? lower : upper;
    }
    if (target == low_energy)
        return lower;
    if (target == high_energy)
        return upper;
    for (int iteration = 0; iteration < 180; ++iteration)
    {
        const double middle = std::midpoint(lower, upper);
        const double value = energy(middle);
        require(finite(value), "Caloric inversion returned nonfinite energy.");
        if (std::abs(value - target) <= tolerance)
            return middle;
        if (middle == lower || middle == upper)
            break;
        if (value < target)
            lower = middle;
        else
            upper = middle;
    }
    const double result = std::midpoint(lower, upper);
    require(std::abs(energy(result) - target) <= 8 * tolerance, "Caloric inversion did not converge.");
    return result;
}
} // namespace

MixtureThermodynamics::MixtureThermodynamics(std::shared_ptr<const SpeciesCatalog> catalog, ThermodynamicBounds bounds,
    EOSKind kind, Evaluator evaluator, EnergyConvention convention, ChemicalPotentialEvaluator chemical_potentials)
    : MixtureThermodynamics(std::move(catalog), bounds, kind, EnthalpyPressureDependence::Unspecified,
          std::move(evaluator), convention, std::move(chemical_potentials))
{
}

MixtureThermodynamics::MixtureThermodynamics(std::shared_ptr<const SpeciesCatalog> catalog, ThermodynamicBounds bounds,
    EOSKind kind, EnthalpyPressureDependence dependence, Evaluator evaluator, EnergyConvention convention,
    ChemicalPotentialEvaluator chemical_potentials)
    : catalog_(std::move(catalog)), bounds_(bounds), kind_(kind), enthalpy_pressure_dependence_(dependence),
      evaluator_(std::move(evaluator)), convention_(convention), chemical_potentials_(std::move(chemical_potentials))
{
    require(static_cast<bool>(catalog_) && !catalog_->species().empty(), "Mixture EOS species catalog is empty.");
    require(kind_ == EOSKind::Liquid || kind_ == EOSKind::Gas, "Unknown mixture EOS kind.");
    require(convention_ == EnergyConvention::SensibleEnergy || convention_ == EnergyConvention::IncludesFormationEnergy,
        "Unknown mixture caloric energy convention.");
    require(enthalpy_pressure_dependence_ == EnthalpyPressureDependence::Unspecified ||
                enthalpy_pressure_dependence_ == EnthalpyPressureDependence::PressureIndependent,
        "Unknown immutable enthalpy-pressure capability.");
    validate_bounds(bounds_);
    require(static_cast<bool>(evaluator_), "Mixture EOS callback is null.");
}

EnthalpyPressureDependence MixtureThermodynamics::enthalpy_pressure_dependence() const noexcept
{
    return enthalpy_pressure_dependence_;
}

const ThermodynamicBounds& MixtureThermodynamics::bounds() const noexcept
{
    return bounds_;
}
const SpeciesCatalog& MixtureThermodynamics::catalog() const noexcept
{
    return *catalog_;
}
EOSKind MixtureThermodynamics::kind() const noexcept
{
    return kind_;
}
EnergyConvention MixtureThermodynamics::energy_convention() const noexcept
{
    return convention_;
}

MixtureThermodynamicPoint MixtureThermodynamics::evaluate(
    double temperature, double pressure, std::span<const double> x) const
{
    require(finite(temperature) && finite(pressure) && temperature >= bounds_.minimum_temperature &&
                temperature <= bounds_.maximum_temperature && pressure >= bounds_.minimum_pressure &&
                pressure <= bounds_.maximum_pressure,
        "Mixture EOS state is outside its declared T/P range.");
    require(x.size() == catalog_->species().size(), "Mixture mole fractions do not match catalog.");
    double sum = 0;
    for (double value : x)
    {
        require(finite(value) && value >= 0 && value <= 1, "Mixture mole fraction is invalid.");
        sum += value;
    }
    require(close(sum, 1.0, 2e-12), "Mixture mole fractions must sum to one.");
    auto point = evaluator_(temperature, pressure, x);
    const std::array values{point.density, point.internal_energy, point.enthalpy, point.internal_energy_temperature,
        point.internal_energy_pressure, point.density_temperature, point.density_pressure, point.dynamic_viscosity,
        point.thermal_conductivity};
    require(std::all_of(values.begin(), values.end(), finite) && point.density > 0 &&
                point.internal_energy_temperature > 0 && point.density_pressure >= 0 &&
                (kind_ == EOSKind::Liquid || point.density_pressure > 0) && point.dynamic_viscosity >= 0 &&
                point.thermal_conductivity >= 0,
        "Mixture EOS returned invalid density, caloric derivative, or compressibility.");
    const double rho2 = point.density * point.density;
    require(close(point.enthalpy, point.internal_energy + pressure / point.density, 2e-11),
        "Mixture EOS violates h=e+p/rho.");
    const double expected_ep = (temperature * point.density_temperature + pressure * point.density_pressure) / rho2;
    require(close(point.internal_energy_pressure, expected_ep, 2e-11),
        "Mixture EOS pressure/temperature derivatives violate the caloric identity.");
    require(point.internal_energy_temperature - pressure * point.density_temperature / rho2 > 0,
        "Mixture EOS has nonpositive isobaric heat capacity.");
    if (point.density_pressure == 0)
        require(kind_ == EOSKind::Liquid && point.density_temperature == 0 && point.internal_energy_pressure == 0,
            "An incompressible liquid requires constant material volume and pressure-independent internal energy.");
    else
        require(point.internal_energy_temperature -
                        point.internal_energy_pressure * point.density_temperature / point.density_pressure >
                    0,
            "Mixture EOS has nonpositive isochoric heat capacity.");
    if (enthalpy_pressure_dependence_ == EnthalpyPressureDependence::PressureIndependent)
    {
        const double v = 1 / point.density, thermal = temperature * point.density_temperature / rho2;
        require(
            std::abs(v + thermal) <= 256 * std::numeric_limits<double>::epsilon() * (std::abs(v) + std::abs(thermal)),
            "EOS contradicts its immutable pressure-independent enthalpy capability.");
    }
    if (chemical_potentials_)
        point.chemical_potentials = chemical_potentials_(temperature, pressure, x);
    if (!point.fugacities.empty())
    {
        require(point.fugacities.size() == x.size(), "Mixture fugacities do not match catalog.");
        for (std::size_t i = 0; i < x.size(); ++i)
            require(finite(point.fugacities[i]) && point.fugacities[i] >= 0 && (x[i] == 0 || point.fugacities[i] > 0),
                "Mixture fugacity is invalid.");
    }
    if (!point.chemical_potentials.empty())
        require(point.chemical_potentials.size() == x.size() &&
                    std::all_of(point.chemical_potentials.begin(), point.chemical_potentials.end(), finite),
            "Mixture chemical potentials are invalid.");
    require(kind_ != EOSKind::Gas || !point.fugacities.empty() || !point.chemical_potentials.empty(),
        "Gas EOS must supply species fugacities or an explicit chemical-potential hook.");
    return point;
}

std::shared_ptr<const MixtureThermodynamics> make_ideal_gas_mixture(std::shared_ptr<const SpeciesCatalog> catalog,
    std::vector<IdealGasCaloricSpecies> caloric, ThermodynamicBounds bounds, EnergyConvention convention,
    double gas_constant, MixtureTransportEvaluator transport)
{
    require(static_cast<bool>(catalog) && caloric.size() == catalog->species().size(),
        "Ideal gas caloric properties must match catalog.");
    require(finite(gas_constant) && gas_constant > 0, "Ideal gas constant must be positive.");
    for (const auto& species : caloric)
        require(finite(species.molar_heat_capacity_v) && species.molar_heat_capacity_v > 0 &&
                    finite(species.reference_temperature) && species.reference_temperature > 0 &&
                    finite(species.reference_internal_energy),
            "Ideal gas caloric properties are invalid.");
    auto evaluate = [catalog, caloric = std::move(caloric), gas_constant, transport = std::move(transport)](
                        double t, double p, std::span<const double> x)
    {
        double molecular_weight = 0, cv = 0, e = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            molecular_weight += x[i] * catalog->species()[i].molarMass;
            cv += x[i] * caloric[i].molar_heat_capacity_v;
            e += x[i] * (caloric[i].reference_internal_energy +
                            caloric[i].molar_heat_capacity_v * (t - caloric[i].reference_temperature));
        }
        const double rho = p * molecular_weight / (gas_constant * t);
        MixtureThermodynamicPoint result{rho, e / molecular_weight, (e + gas_constant * t) / molecular_weight,
            cv / molecular_weight, 0, -rho / t, rho / p, {}, {}};
        if (transport)
            std::tie(result.dynamic_viscosity, result.thermal_conductivity) = transport(t, p, x);
        for (double fraction : x)
            result.fugacities.push_back(fraction * p);
        return result;
    };
    return std::make_shared<const MixtureThermodynamics>(std::move(catalog), bounds, EOSKind::Gas,
        EnthalpyPressureDependence::PressureIndependent, std::move(evaluate), convention);
}

std::shared_ptr<const MixtureThermodynamics> make_linear_compressible_mixture_liquid(
    std::shared_ptr<const SpeciesCatalog> catalog, ThermodynamicBounds bounds,
    LiquidMixturePropertyEvaluator properties, EnergyConvention convention)
{
    require(static_cast<bool>(properties), "Liquid mixture property callback is null.");
    auto evaluate = [properties = std::move(properties)](double t, double p, std::span<const double> x)
    {
        const auto q = properties(x);
        const std::array values{q.reference_temperature, q.reference_pressure, q.reference_density,
            q.isobaric_heat_capacity, q.thermal_expansion, q.compressibility, q.reference_internal_energy,
            q.dynamic_viscosity, q.thermal_conductivity};
        require(std::all_of(values.begin(), values.end(), finite) && q.reference_temperature > 0 &&
                    q.reference_pressure > 0 && q.reference_density > 0 && q.isobaric_heat_capacity > 0 &&
                    q.compressibility >= 0 && (q.compressibility > 0 || q.thermal_expansion == 0) &&
                    q.dynamic_viscosity >= 0 && q.thermal_conductivity >= 0,
            "Liquid mixture supplied properties are invalid.");
        const double v0 = 1.0 / q.reference_density;
        const double dp = p - q.reference_pressure;
        const double v = v0 * (1 + q.thermal_expansion * (t - q.reference_temperature) - q.compressibility * dp);
        require(v > 0 && finite(v), "Linear liquid volume is outside its supplied validity domain.");
        const double h =
            q.reference_internal_energy + q.reference_pressure * v0 +
            q.isobaric_heat_capacity * (t - q.reference_temperature) +
            v0 * ((1 - q.thermal_expansion * q.reference_temperature) * dp - 0.5 * q.compressibility * dp * dp);
        return MixtureThermodynamicPoint{1 / v, h - p * v, h, q.isobaric_heat_capacity - p * v0 * q.thermal_expansion,
            -t * v0 * q.thermal_expansion + p * v0 * q.compressibility, -v0 * q.thermal_expansion / (v * v),
            v0 * q.compressibility / (v * v), {}, {}, q.dynamic_viscosity, q.thermal_conductivity};
    };
    return std::make_shared<const MixtureThermodynamics>(
        std::move(catalog), bounds, EOSKind::Liquid, std::move(evaluate), convention);
}

namespace
{
struct CaloricGroup
{
    std::vector<std::size_t> phases;
    double minimum_temperature = 0, maximum_temperature = 0;
};
struct CaloricDescription
{
    CaloricPressureInterval pressure;
    std::vector<Composition> mixtures;
    std::vector<CaloricGroup> groups;
    bool empty = false;
};
CaloricDescription describe_caloric_query(const CaloricPressureQuery& query)
{
    require(!query.phases.empty() && query.phases.size() == query.advected_volumes.size() &&
                query.phases.size() == query.pressure_offsets.size(),
        "Caloric query phase/volume/offset layouts differ.");
    require(query.mode == PhaseThermalMode::Equilibrium || query.mode == PhaseThermalMode::SeparatePhaseTemperatures,
        "Unknown caloric query thermal mode.");
    require(finite(query.pressure.lower) && finite(query.pressure.upper) && query.pressure.lower > 0 &&
                query.pressure.lower <= query.pressure.upper,
        "Invalid caloric query pressure interval.");
    CaloricDescription result;
    result.pressure = query.pressure;
    result.mixtures.reserve(query.phases.size());
    CaloricGroup common;
    common.maximum_temperature = std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < query.phases.size(); ++a)
    {
        const auto& phase = query.phases[a];
        require(bool(phase.thermodynamics) && finite(phase.internal_energy) && finite(query.advected_volumes[a]) &&
                    finite(query.pressure_offsets[a]),
            "Invalid caloric query inventory, volume packet or offset.");
        require(phase.thermodynamics->energy_convention() == query.phases.front().thermodynamics->energy_convention(),
            "Caloric query phases use different formation-energy conventions.");
        result.mixtures.push_back(composition(phase.thermodynamics->catalog(), phase.moles));
        const auto& bounds = phase.thermodynamics->bounds();
        if (result.mixtures.back().amount > 0)
        {
            result.pressure.lower =
                std::max(result.pressure.lower, bounds.minimum_pressure - query.pressure_offsets[a]);
            result.pressure.upper =
                std::min(result.pressure.upper, bounds.maximum_pressure - query.pressure_offsets[a]);
            if (query.mode == PhaseThermalMode::Equilibrium)
            {
                common.phases.push_back(a);
                common.minimum_temperature = std::max(common.minimum_temperature, bounds.minimum_temperature);
                common.maximum_temperature = std::min(common.maximum_temperature, bounds.maximum_temperature);
            }
            else
                result.groups.push_back({{a}, bounds.minimum_temperature, bounds.maximum_temperature});
        }
        else
            result.groups.push_back({{a}, 0, 0}); // Exact empty-phase work/energy equation, no EOS call.
    }
    if (!common.phases.empty())
    {
        result.empty = common.minimum_temperature > common.maximum_temperature;
        result.groups.push_back(std::move(common));
    }
    result.empty = result.empty || result.pressure.lower > result.pressure.upper;
    return result;
}
CaloricPressureMargins caloric_margins(
    const CaloricPressureQuery& query, const CaloricDescription& description, double p0)
{
    require(!description.empty && finite(p0) && p0 >= description.pressure.lower && p0 <= description.pressure.upper,
        "Caloric margin query is outside the supplied pressure/temperature domain.");
    CaloricPressureMargins result;
    for (const auto& group : description.groups)
    {
        long double target = 0, low = 0, high = 0;
        for (const auto a : group.phases)
        {
            const auto& phase = query.phases[a];
            const double pressure = p0 + query.pressure_offsets[a];
            target += static_cast<long double>(phase.internal_energy) +
                      static_cast<long double>(pressure) * query.advected_volumes[a];
            const auto& mix = description.mixtures[a];
            if (mix.amount == 0)
                continue;
            const auto low_point = phase.thermodynamics->evaluate(group.minimum_temperature, pressure, mix.fractions);
            const auto high_point = phase.thermodynamics->evaluate(group.maximum_temperature, pressure, mix.fractions);
            require(high_point.enthalpy >= low_point.enthalpy, "Caloric endpoint enthalpy is not monotone.");
            low += static_cast<long double>(mix.mass) * low_point.enthalpy;
            high += static_cast<long double>(mix.mass) * high_point.enthalpy;
        }
        require(std::isfinite(target) && std::isfinite(low) && std::isfinite(high), "Caloric margin overflow.");
        result.lower.push_back(target - low);
        result.upper.push_back(high - target);
    }
    return result;
}
struct AffineCaloricConstraint
{
    long double constant = 0, coefficient = 0;
};
CaloricPressureDomain intersect_affine_caloric_constraints(
    CaloricPressureInterval supplied, std::span<const AffineCaloricConstraint> constraints)
{
    long double lower = supplied.lower, upper = supplied.upper;
    for (const auto constraint : constraints)
    {
        require(std::isfinite(constraint.constant) && std::isfinite(constraint.coefficient),
            "Nonfinite affine caloric constraint.");
        const auto at_lower = std::fma(constraint.coefficient, lower, constraint.constant);
        const auto at_upper = std::fma(constraint.coefficient, upper, constraint.constant);
        require(std::isfinite(at_lower) && std::isfinite(at_upper), "Affine caloric margin overflow.");
        if (constraint.coefficient == 0)
        {
            if (constraint.constant < 0)
                return {CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
            continue;
        }
        if (constraint.coefficient > 0)
        {
            if (at_upper < 0)
                return {CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
            if (at_lower < 0)
                lower = std::max(lower, -constraint.constant / constraint.coefficient);
        }
        else
        {
            if (at_lower < 0)
                return {CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
            if (at_upper < 0)
                upper = std::min(upper, -constraint.constant / constraint.coefficient);
        }
        if (lower > upper)
            return {CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
    }
    double lo = static_cast<double>(lower), hi = static_cast<double>(upper);
    if (static_cast<long double>(lo) < lower)
        lo = std::nextafter(lo, std::numeric_limits<double>::infinity());
    if (static_cast<long double>(hi) > upper)
        hi = std::nextafter(hi, -std::numeric_limits<double>::infinity());
    if (lo > hi)
        return {CaloricDomainKnowledge::Unresolved, std::nullopt};
    return {CaloricDomainKnowledge::CertifiedConnected, CaloricPressureInterval{lo, hi}};
}
} // namespace

CaloricPressureMargins evaluate_caloric_pressure_margins(const CaloricPressureQuery& query, double pressure)
{
    return caloric_margins(query, describe_caloric_query(query), pressure);
}

CaloricPressureDomain exact_caloric_pressure_domain(const CaloricPressureQuery& query)
{
    const auto description = describe_caloric_query(query);
    if (description.empty)
        return {CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
    for (std::size_t a = 0; a < query.phases.size(); ++a)
        if (description.mixtures[a].amount > 0 && query.phases[a].thermodynamics->enthalpy_pressure_dependence() !=
                                                      EnthalpyPressureDependence::PressureIndependent)
            return {CaloricDomainKnowledge::Unresolved, std::nullopt};
    const double anchor = description.pressure.lower;
    const auto margins = caloric_margins(query, description, anchor);
    std::vector<AffineCaloricConstraint> constraints;
    for (std::size_t g = 0; g < description.groups.size(); ++g)
    {
        long double volume = 0;
        for (const auto a : description.groups[g].phases)
            volume += query.advected_volumes[a];
        constraints.push_back({std::fma(-volume, static_cast<long double>(anchor), margins.lower[g]), volume});
        constraints.push_back({std::fma(volume, static_cast<long double>(anchor), margins.upper[g]), -volume});
    }
    return intersect_affine_caloric_constraints(description.pressure, constraints);
}

PhaseThermalRecovery recover_phase_temperatures(
    std::span<const PhaseThermalState> phases, double total_energy, PhaseThermalMode mode)
{
    require(finite(total_energy), "Total phase internal energy is nonfinite.");
    require(mode == PhaseThermalMode::Equilibrium || mode == PhaseThermalMode::SeparatePhaseTemperatures,
        "Unknown phase thermal mode.");
    PhaseThermalRecovery result;
    result.phases.assign(phases.begin(), phases.end());
    result.volumes.resize(phases.size());
    std::vector<Composition> mixtures;
    double lower = 0, upper = std::numeric_limits<double>::infinity(), given_energy = 0, given_energy_scale = 0;
    bool occupied = false;
    for (const auto& phase : phases)
    {
        validate_phase(phase);
        require(phase.thermodynamics->energy_convention() == phases.front().thermodynamics->energy_convention(),
            "All phases must use one common chemical formation-energy convention.");
        mixtures.push_back(composition(phase.thermodynamics->catalog(), phase.moles));
        given_energy += phase.internal_energy;
        given_energy_scale += std::abs(phase.internal_energy);
        if (mixtures.back().amount > 0)
        {
            occupied = true;
            lower = std::max(lower, phase.thermodynamics->bounds().minimum_temperature);
            upper = std::min(upper, phase.thermodynamics->bounds().maximum_temperature);
        }
    }
    if (!occupied)
    {
        require(total_energy == 0, "Empty phases cannot store internal energy.");
        return result;
    }
    auto phase_energy = [&](std::size_t i, double t)
    {
        return mixtures[i].mass * phases[i]
                                      .thermodynamics->evaluate(t, phases[i].absolute_pressure, mixtures[i].fractions)
                                      .internal_energy;
    };
    double common_t = 0;
    if (mode == PhaseThermalMode::Equilibrium)
        common_t = invert_energy(lower, upper, total_energy,
            [&](double t)
            {
                double sum = 0;
                for (std::size_t i = 0; i < phases.size(); ++i)
                    if (mixtures[i].amount > 0)
                        sum += phase_energy(i, t);
                return sum;
            });
    else
        require(std::abs(given_energy - total_energy) <=
                    2e-11 * std::max({std::numeric_limits<double>::min(), given_energy_scale, std::abs(total_energy)}),
            "Separate phase energies do not sum to supplied total internal energy.");
    double energy_t = 0, energy_p = 0, volume_t = 0, volume_p = 0, energy_scale = 0;
    for (std::size_t i = 0; i < phases.size(); ++i)
    {
        if (mixtures[i].amount == 0)
            continue;
        const auto& bounds = phases[i].thermodynamics->bounds();
        const double t = mode == PhaseThermalMode::Equilibrium
                             ? common_t
                             : invert_energy(bounds.minimum_temperature, bounds.maximum_temperature,
                                   phases[i].internal_energy, [&](double trial_t) { return phase_energy(i, trial_t); });
        const auto point = phases[i].thermodynamics->evaluate(t, phases[i].absolute_pressure, mixtures[i].fractions);
        const double mass = mixtures[i].mass;
        const double vt = -mass * point.density_temperature / (point.density * point.density);
        const double vp = -mass * point.density_pressure / (point.density * point.density);
        result.phases[i].temperature = t;
        result.phases[i].internal_energy = mass * point.internal_energy;
        result.volumes[i] = mass / point.density;
        result.total_internal_energy += result.phases[i].internal_energy;
        result.total_volume += result.volumes[i];
        energy_scale += mass * (std::abs(point.internal_energy) + t * point.internal_energy_temperature);
        energy_t += mass * point.internal_energy_temperature;
        energy_p += mass * point.internal_energy_pressure;
        volume_t += vt;
        volume_p += vp;
        if (mode == PhaseThermalMode::SeparatePhaseTemperatures)
            result.volume_pressure_susceptibility +=
                vp - vt * point.internal_energy_pressure / point.internal_energy_temperature;
    }
    if (mode == PhaseThermalMode::Equilibrium)
        result.volume_pressure_susceptibility = volume_p - volume_t * energy_p / energy_t;
    require(std::abs(result.total_internal_energy - total_energy) <=
                2e-10 * std::max({std::numeric_limits<double>::min(), energy_scale, std::abs(total_energy)}),
        "Recovered phase energies do not close.");
    require(finite(result.total_volume) && finite(result.volume_pressure_susceptibility),
        "Recovered phase volume or susceptibility is invalid.");
    return result;
}

PhasePressureRecovery solve_phase_pressure(std::span<const PhaseThermalState> phases, double energy, double volume,
    double minimum, double maximum, PhaseThermalMode mode)
{
    require(!phases.empty() && finite(volume) && volume > 0 && finite(minimum) && finite(maximum) && minimum > 0 &&
                maximum > minimum,
        "Pressure inversion requires a positive volume and pressure bracket.");
    const double reference = phases.front().absolute_pressure;
    for (const auto& phase : phases)
    {
        validate_phase(phase);
        const double offset = phase.absolute_pressure - reference;
        minimum = std::max(minimum, phase.thermodynamics->bounds().minimum_pressure - offset);
        maximum = std::min(maximum, phase.thermodynamics->bounds().maximum_pressure - offset);
    }
    require(minimum < maximum, "Phase pressure validity intervals do not intersect.");
    auto evaluate = [&](double pressure)
    {
        auto shifted = std::vector<PhaseThermalState>(phases.begin(), phases.end());
        for (auto& phase : shifted)
            phase.absolute_pressure += pressure - reference;
        auto thermal = recover_phase_temperatures(shifted, energy, mode);
        require(thermal.volume_pressure_susceptibility <= 0,
            "Pressure inversion requires a stable nonpositive volume susceptibility.");
        return PhasePressureRecovery{pressure, std::move(thermal)};
    };
    // A pressure validity interval need not be calorically feasible at both endpoints at fixed U.
    // Restrict to the connected feasible branch containing the supplied accepted pressure.
    const auto feasible = [&](double pressure) -> std::optional<PhasePressureRecovery>
    {
        try
        {
            return evaluate(pressure);
        }
        catch (const CaloricRangeError&)
        {
            return std::nullopt;
        }
    };
    const auto accepted = feasible(std::clamp(reference, minimum, maximum));
    if (accepted)
    {
        if (std::abs(accepted->thermal.total_volume - volume) <= 8e-12 * volume)
            return *accepted;
        require(accepted->thermal.volume_pressure_susceptibility < 0,
            "An incompressible closed material cannot accommodate the requested incompatible volume.");
    }
    auto low = feasible(minimum), high = feasible(maximum);
    if (!low || !high)
    {
        auto anchor = feasible(std::clamp(reference, minimum, maximum));
        for (int probe = 1; !anchor && probe < 128; ++probe)
            anchor = feasible(std::lerp(minimum, maximum, probe / 128.0));
        require(anchor.has_value(), "No calorically feasible state found within supplied pressure bounds.");
        if (std::abs(anchor->thermal.total_volume - volume) <= 8e-12 * volume)
            return *anchor;
        const auto contract = [&](double invalid, double valid)
        {
            auto candidate = evaluate(valid);
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                const double middle = std::midpoint(invalid, valid);
                if (middle == invalid || middle == valid)
                    break;
                auto next = feasible(middle);
                if (next)
                {
                    valid = middle;
                    candidate = std::move(*next);
                }
                else
                    invalid = middle;
            }
            return candidate;
        };
        if (!low)
            low = contract(minimum, anchor->reference_pressure);
        if (!high)
            high = contract(maximum, anchor->reference_pressure);
        minimum = low->reference_pressure;
        maximum = high->reference_pressure;
    }
    require(low->thermal.total_volume >= volume && high->thermal.total_volume <= volume,
        "Requested material volume has no pressure root on the calorically feasible branch.");
    for (int iteration = 0; iteration < 180; ++iteration)
    {
        const double pressure = std::midpoint(minimum, maximum);
        auto trial = evaluate(pressure);
        const double residual = trial.thermal.total_volume - volume;
        if (std::abs(residual) <= 8e-12 * volume || pressure == minimum || pressure == maximum)
            return trial;
        if (residual > 0)
            minimum = pressure;
        else
            maximum = pressure;
    }
    throw std::runtime_error("Material pressure inversion did not converge.");
}

BubbleThermodynamicState solve_bubble_radius(const MixtureThermodynamics& gas, std::span<const double> moles,
    double temperature, double liquid_pressure, double sigma, double number, double minimum, double maximum)
{
    require(gas.kind() == EOSKind::Gas, "Bubble radius requires a gas EOS.");
    require(finite(liquid_pressure) && liquid_pressure > 0 && finite(sigma) && sigma >= 0 && finite(number) &&
                number > 0 && finite(minimum) && minimum > 0 && finite(maximum) && maximum > minimum,
        "Bubble radius inputs/bounds are invalid.");
    const auto c = composition(gas.catalog(), moles);
    require(c.amount > 0, "A populated bubble cannot contain zero gas.");
    // Intersect supplied radius bounds with EOS pressure validity before evaluating endpoints.
    const auto& bounds = gas.bounds();
    if (sigma > 0)
    {
        require(bounds.maximum_pressure > liquid_pressure, "No bubble radius satisfies EOS pressure bounds.");
        minimum = std::max(minimum, 2 * sigma / (bounds.maximum_pressure - liquid_pressure));
        if (bounds.minimum_pressure > liquid_pressure)
            maximum = std::min(maximum, 2 * sigma / (bounds.minimum_pressure - liquid_pressure));
        // The inverse pressure bound can round outward when mapped back through
        // p_l + 2*sigma/r. Move only the radius endpoint inward; the EOS domain
        // and the subsequent signed volume bracket remain strict.
        while (minimum <= maximum && liquid_pressure + 2 * sigma / minimum > bounds.maximum_pressure)
            minimum = std::nextafter(minimum, std::numeric_limits<double>::infinity());
        while (minimum <= maximum && liquid_pressure + 2 * sigma / maximum < bounds.minimum_pressure)
            maximum = std::nextafter(maximum, 0.0);
    }
    require(minimum <= maximum, "Bubble radius and EOS pressure bounds do not intersect.");
    auto evaluate = [&](double radius)
    {
        const double pressure = liquid_pressure + 2 * sigma / radius;
        const auto point = gas.evaluate(temperature, pressure, c.fractions);
        return BubbleThermodynamicState{radius, c.mass / point.density, pressure, point};
    };
    auto residual = [&](const BubbleThermodynamicState& state)
    { return number * (4 * std::numbers::pi / 3) * state.radius * state.radius * state.radius - state.volume; };
    auto low = evaluate(minimum), high = evaluate(maximum);
    double fl = residual(low), fh = residual(high);
    require(fl <= 0 && fh >= 0, "Bubble volume equation has no bracket in supplied radius bounds.");
    for (int i = 0; i < 180; ++i)
    {
        const double radius = std::midpoint(minimum, maximum);
        const auto trial = evaluate(radius);
        const double f = residual(trial);
        // Film k_L*A can depend directly on radius, so resolve this
        // numerical inversion near arithmetic accuracy before using its area.
        if (std::abs(f) <= 32 * std::numeric_limits<double>::epsilon() * trial.volume || radius == minimum ||
            radius == maximum)
            return trial;
        if (f < 0)
            minimum = radius;
        else
            maximum = radius;
    }
    throw std::runtime_error("Bubble radius inversion did not converge.");
}

InterphaseExchangeResult preview_interphase_exchange(std::span<const PhaseThermalState> phases, std::size_t first,
    std::size_t second, const InterphaseExchangeClosure& closure, PhaseThermalMode mode)
{
    require(first < phases.size() && second < phases.size() && first != second && static_cast<bool>(closure),
        "Interphase exchange requires two distinct phases and a closure.");
    double total = 0;
    for (const auto& phase : phases)
    {
        validate_phase(phase);
        total += phase.internal_energy;
    }
    const auto initial = recover_phase_temperatures(phases, total, mode);
    const auto& a = initial.phases[first];
    const auto& b = initial.phases[second];
    require(&a.thermodynamics->catalog() == &b.thermodynamics->catalog(),
        "Interphase exchange requires one shared immutable species catalog.");
    require(a.thermodynamics->energy_convention() == b.thermodynamics->energy_convention(),
        "Interphase exchange phases use different chemical formation-energy conventions.");
    const auto ca = composition(a.thermodynamics->catalog(), a.moles);
    const auto cb = composition(b.thermodynamics->catalog(), b.moles);
    // Empty receiver can be initialized by a closure, but has no invented EOS state or temperature.
    const auto pa = ca.amount > 0 ? a.thermodynamics->evaluate(a.temperature, a.absolute_pressure, ca.fractions)
                                  : MixtureThermodynamicPoint{};
    const auto pb = cb.amount > 0 ? b.thermodynamics->evaluate(b.temperature, b.absolute_pressure, cb.fractions)
                                  : MixtureThermodynamicPoint{};
    const auto exchange = closure(a, pa, b, pb);
    require(exchange.moles_first_to_second.size() == a.moles.size() &&
                exchange.carried_molar_enthalpy.size() == a.moles.size() && finite(exchange.heat_first_to_second),
        "Interphase closure packet has invalid species shape or heat.");
    auto candidate = initial.phases;
    InterphaseExchangeResult result;
    for (std::size_t i = 0; i < a.moles.size(); ++i)
    {
        const double delta = exchange.moles_first_to_second[i], h = exchange.carried_molar_enthalpy[i];
        require(finite(delta) && finite(h), "Interphase closure packet has nonfinite transfer/enthalpy.");
        require(delta <= a.moles[i] && -delta <= b.moles[i], "Interphase exchange overdraws a species donor.");
        candidate[first].moles[i] -= delta;
        candidate[second].moles[i] += delta;
        result.mass_first_to_second += delta * a.thermodynamics->catalog().species()[i].molarMass;
        result.enthalpy_first_to_second += delta * h;
    }
    result.heat_first_to_second = exchange.heat_first_to_second;
    require(finite(result.mass_first_to_second) && finite(result.enthalpy_first_to_second),
        "Interphase exchange mass/enthalpy overflow.");
    const double energy = result.enthalpy_first_to_second + result.heat_first_to_second;
    candidate[first].internal_energy -= energy;
    candidate[second].internal_energy += energy;
    if (exchange.phase_volume_work)
    {
        const auto work = *exchange.phase_volume_work;
        require(finite(work.first_volume_change) && finite(work.second_volume_change),
            "Isobaric phase-volume work is nonfinite.");
        result.pressure_work_on_first = -a.absolute_pressure * work.first_volume_change;
        result.pressure_work_on_second = -b.absolute_pressure * work.second_volume_change;
        result.external_mechanical_work = result.pressure_work_on_first + result.pressure_work_on_second;
        candidate[first].internal_energy += result.pressure_work_on_first;
        candidate[second].internal_energy += result.pressure_work_on_second;
        total += result.external_mechanical_work;
    }
    for (size_t phase_index = 0; phase_index < candidate.size(); ++phase_index)
        if (auto& phase = candidate[phase_index]; composition(phase.thermodynamics->catalog(), phase.moles).amount == 0)
        {
            const double phase_work = phase_index == first
                                          ? result.pressure_work_on_first
                                          : (phase_index == second ? result.pressure_work_on_second : 0);
            const double transferred_energy = phase_index == first || phase_index == second ? energy : 0;
            const double operand_bound = 32 * std::numeric_limits<double>::epsilon() *
                                         (std::abs(initial.phases[phase_index].internal_energy) +
                                             std::abs(transferred_energy) + std::abs(phase_work));
            require(exchange.phase_volume_work ? std::abs(phase.internal_energy) <= operand_bound
                                               : phase.internal_energy == 0,
                "Interphase exchange leaves physical energy in a completely depleted phase.");
            result.empty_phase_energy_roundoff -= phase.internal_energy;
            total -= phase.internal_energy;
            phase.internal_energy = 0;
            phase.temperature = 0;
        }
    // Even equilibrium mode must satisfy individual finite-energy availability before common-T relaxation.
    (void) recover_phase_temperatures(candidate, total, PhaseThermalMode::SeparatePhaseTemperatures);
    result.thermal = recover_phase_temperatures(candidate, total, mode);
    if (exchange.phase_volume_work)
    {
        const auto check_volume = [&](size_t i, double change)
        {
            const double error = result.thermal.volumes[i] - initial.volumes[i] - change;
            const double scale = std::max({initial.volumes[i], result.thermal.volumes[i], std::abs(change)});
            require(std::abs(error) <= 2e-10 * scale,
                "Supplied isobaric work does not match recovered phase-volume changes.");
        };
        check_volume(first, exchange.phase_volume_work->first_volume_change);
        check_volume(second, exchange.phase_volume_work->second_volume_change);
    }
    return result;
}

} // namespace SimpleFluid::Reacting
