/** @file ClosedMixtureHeadspaceModel.cc
 * @brief Bounded prescribed-temperature pressure closure and accepted species accounting.
 */
#include "solvers/ClosedMixtureHeadspaceModel.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace SimpleFluid
{
namespace
{
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
bool close(double a, double b, double tolerance)
{
    return std::isfinite(a) && std::isfinite(b) &&
        std::abs(static_cast<long double>(a) - b) <=
            tolerance * std::max(std::abs(static_cast<long double>(a)), std::abs(static_cast<long double>(b)));
}
} // namespace

ClosedMixtureHeadspaceModel::ClosedMixtureHeadspaceModel(HeadspaceOptions options,
    std::shared_ptr<const Reacting::MixtureThermodynamics> thermodynamics, std::string provider_identity,
    std::uint64_t provider_revision)
    : HeadspaceModel(std::move(options),
          {thermodynamics ? thermodynamics->catalog().fingerprint() : 0, std::move(provider_identity),
              provider_revision,
              thermodynamics ? std::array<real_t, 4>{thermodynamics->bounds().minimum_temperature,
                                   thermodynamics->bounds().maximum_temperature,
                                   thermodynamics->bounds().minimum_pressure, thermodynamics->bounds().maximum_pressure}
                             : std::array<real_t, 4>{},
              true}),
      d_thermodynamics(std::move(thermodynamics))
{
    require(!metadata().provider_identity.empty() && metadata().provider_revision > 0,
        "A supplied headspace EOS requires explicit immutable identity and revision.");
    require(d_options.mode == HeadspaceMode::Closed && !d_options.infer_initial_moles &&
        !d_options.initial_moles.empty(), "A mixture headspace requires explicit initial species moles.");
    require(d_options.temperature_mode == HeadspaceTemperatureMode::Fixed ||
        d_options.temperature_mode == HeadspaceTemperatureMode::Prescribed ||
        d_options.temperature_mode == HeadspaceTemperatureMode::BulkLiquid ||
        d_options.temperature_mode == HeadspaceTemperatureMode::External,
        "Unsupported headspace temperature policy.");
    require(d_thermodynamics && d_thermodynamics->kind() == Reacting::EOSKind::Gas,
        "A mixture headspace requires a supplied gas-mixture EOS.");
}

std::unique_ptr<HeadspaceModel> ClosedMixtureHeadspaceModel::clone() const
{
    return std::make_unique<ClosedMixtureHeadspaceModel>(*this);
}
HeadspaceMode ClosedMixtureHeadspaceModel::mode() const noexcept { return HeadspaceMode::Closed; }
const GasMolesBySpecies& ClosedMixtureHeadspaceModel::headspaceMoles() const noexcept { return d_moles; }
const GasMolesBySpecies& ClosedMixtureHeadspaceModel::ventedMoles() const noexcept { return d_empty_vent; }
const GasMolesBySpecies& ClosedMixtureHeadspaceModel::accumulatedEscapedMoles() const noexcept { return d_escaped; }

ClosedMixtureHeadspaceModel::InventoryCandidate ClosedMixtureHeadspaceModel::prospective(
    const GasMolesBySpecies& escaped) const
{
    if (!d_initialized)
        throw std::logic_error("A mixture headspace must be initialized before trial or commit.");
    InventoryCandidate result{d_moles, d_escaped, d_amount_sums, d_escape_sums};
    const auto add = [](auto& sums, auto& amounts, const auto& name, double amount)
    {
        auto& accumulator = sums[name];
        accumulator += amount;
        const long double sum = accumulator.value();
        const double stored = static_cast<double>(sum);
        require(std::isfinite(sum) && std::isfinite(stored) && (sum == 0 || stored > 0),
            "Headspace species addition exceeds representable storage.");
        amounts[name] = stored;
    };
    for (const auto& [name, amount] : escaped)
    {
        require(std::isfinite(amount) && amount >= 0, "Headspace escape amounts must be finite and nonnegative.");
        add(result.amount_sums, result.moles, name, amount);
        add(result.escape_sums, result.escaped, name, amount);
    }
    return result;
}

HeadspaceState ClosedMixtureHeadspaceModel::recover(real_t volume, real_t gas_temperature,
    const GasMolesBySpecies& amounts) const
{
    require(std::isfinite(volume) && volume > 0 && std::isfinite(gas_temperature) && gas_temperature > 0,
        "A closed mixture headspace requires positive finite volume and temperature.");
    const auto& catalog = d_thermodynamics->catalog();
    std::vector<double> moles(catalog.species().size());
    detail::CompensatedSum<long double> total, mass;
    for (const auto& [name, amount] : amounts)
    {
        const auto found = std::ranges::find(catalog.species(), name, &Reacting::SpeciesDefinition::name);
        require(found != catalog.species().end() && std::isfinite(amount) && amount >= 0,
            "Headspace inventories must name catalog species with finite nonnegative amounts.");
        require(amount == 0 || catalog.allows(found->id, Reacting::SpeciesPhase::Resolved) ||
            catalog.allows(found->id, Reacting::SpeciesPhase::Dispersed),
            "A headspace species must permit a gas phase.");
        moles[catalog.index(found->id)] = amount;
        total += amount;
        mass += static_cast<long double>(amount) * found->molarMass;
    }
    require(std::isfinite(total.value()) && total.value() > 0 && std::isfinite(mass.value()) && mass.value() > 0,
        "A closed headspace needs a finite positive gas inventory.");
    const double total_moles = static_cast<double>(total.value());
    require(std::isfinite(total_moles) && total_moles > 0, "Headspace molar total is not representable.");
    for (auto& amount : moles)
    {
        const auto fraction = static_cast<double>(static_cast<long double>(amount) / total.value());
        require(amount == 0 || fraction > 0,
            "Headspace mole fraction is not representable; trace species cannot be dropped.");
        amount = fraction;
    }
    const auto& bounds = d_thermodynamics->bounds();
    const auto volume_at = [&](double pressure)
    {
        const auto point = d_thermodynamics->evaluate(gas_temperature, pressure, moles);
        const auto value = static_cast<double>(mass.value() / point.density);
        require(std::isfinite(value) && value > 0, "Headspace EOS volume is not representable.");
        return value;
    };
    double lower = bounds.minimum_pressure, upper = bounds.maximum_pressure;
    const double low_volume = volume_at(lower), high_volume = volume_at(upper);
    require(low_volume >= volume && high_volume <= volume,
        "Headspace pressure lies outside the supplied gas EOS pressure range.");
    double pressure = lower;
    // Accept exact bounds without asking an iterative root to approach an endpoint.
    bool converged = close(low_volume, volume, 2e-12);
    if (!converged && close(high_volume, volume, 2e-12))
    {
        pressure = upper;
        converged = true;
    }
    // Covers the finite double exponent range without introducing a pressure floor.
    for (int iteration = 0; !converged && iteration < 2048; ++iteration)
    {
        pressure = std::midpoint(lower, upper);
        const double candidate = volume_at(pressure);
        if (close(candidate, volume, 2e-12))
        {
            converged = true;
            break;
        }
        if (candidate > volume)
            lower = pressure;
        else
            upper = pressure;
    }
    require(converged, "Headspace mixture pressure did not meet its physical volume closure.");
    return {.pressure = pressure, .volume = volume, .temperature = gas_temperature, .total_moles = total_moles};
}

void ClosedMixtureHeadspaceModel::initialize(real_t pool_volume, real_t supplied_temperature)
{
    const auto candidate =
        recover(availableVolume(pool_volume), temperature(supplied_temperature), d_options.initial_moles);
    require(close(candidate.pressure, d_options.initial_pressure, 1e-10),
        "Initial headspace species, volume, temperature and pressure disagree with the supplied mixture EOS.");
    auto amounts = d_options.initial_moles;
    AmountSums accumulators;
    for (const auto& [name, amount] : amounts)
        accumulators[name] += amount;
    d_moles.swap(amounts);
    d_amount_sums.swap(accumulators);
    d_escaped.clear();
    d_escape_sums.clear();
    setState(candidate);
    d_initialized = true;
}
HeadspaceState ClosedMixtureHeadspaceModel::trialState(real_t pool_volume, real_t supplied_temperature,
    const GasMolesBySpecies& escaped_moles) const
{
    const auto candidate = prospective(escaped_moles);
    return recover(availableVolume(pool_volume), temperature(supplied_temperature), candidate.moles);
}
void ClosedMixtureHeadspaceModel::commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles)
{
    require(std::isfinite(state.pressure) && state.pressure > 0 && std::isfinite(state.total_moles) &&
        state.total_moles > 0 && state.volume <= d_options.total_internal_volume,
        "Invalid headspace state or volume exceeds vessel capacity.");
    auto candidate = prospective(escaped_moles);
    const auto expected = recover(state.volume, temperature(state.temperature), candidate.moles);
    require(close(expected.pressure, state.pressure, 1e-10) && close(expected.total_moles, state.total_moles, 1e-12) &&
        expected.temperature == state.temperature,
        "Headspace state does not close the declared mixture inventory and EOS.");
    d_moles.swap(candidate.moles);
    d_escaped.swap(candidate.escaped);
    d_amount_sums.swap(candidate.amount_sums);
    d_escape_sums.swap(candidate.escape_sums);
    setState(expected);
}
} // namespace SimpleFluid
