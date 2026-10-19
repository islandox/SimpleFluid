/** @file RadiolyticBubbleKinetics.cc
 * @brief Conservative value kernels without field, MPI, or material-provider ownership.
 */
#include "equations/RadiolyticBubbleKinetics.hh"

#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::RadiolyticGasPhysics
{
namespace
{
using Sum = detail::CompensatedSum<real_t>;
constexpr real_t ledger_tolerance = 1e-12;

void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

void require_result(bool valid, const char* message)
{
    if (!valid)
        throw std::runtime_error(message);
}

bool nonnegative(real_t x)
{
    return std::isfinite(x) && x >= 0.0;
}

void validate(const MolarPivotGrid& grid, const UpperBubblePopulation& state)
{
    require(!grid.mol_per_bubble.empty(), "Bubble molar pivot grid must be nonempty.");
    for (std::size_t i = 0; i < grid.mol_per_bubble.size(); ++i)
        require(std::isfinite(grid.mol_per_bubble[i]) && grid.mol_per_bubble[i] > 0.0 &&
                    (i == 0 || grid.mol_per_bubble[i] > grid.mol_per_bubble[i - 1]),
            "Bubble molar pivots must be finite, positive and strictly increasing.");
    require(nonnegative(state.dissolved_inventory) && state.number_density.size() == grid.mol_per_bubble.size(),
        "Bubble population requires a nonnegative dissolved inventory and one number density per pivot.");
    for (std::size_t i = 0; i < state.number_density.size(); ++i)
    {
        require(nonnegative(state.number_density[i]), "Bubble number densities must be finite and nonnegative.");
        const real_t moles = grid.mol_per_bubble[i] * state.number_density[i];
        require(std::isfinite(moles) && (state.number_density[i] == 0.0 || moles > 0.0),
            "A populated bubble bin must have a representable positive molar inventory.");
    }
}

void check_balance(real_t error, real_t scale, const char* message)
{
    require_result(std::isfinite(error) && std::isfinite(scale) &&
                       std::abs(error) <= ledger_tolerance * std::max(scale, std::numeric_limits<real_t>::min()),
        message);
}

struct Accounting
{
    Sum micro_number, micro_moles, birth_number, birth_moles;
    Sum growth, dissolution, coalescence, breakup, breakup_created_number, dissolved_number;
};

void deposit(const MolarPivotGrid& grid, real_t number, real_t moles, BubbleUnderflowPolicy underflow,
    std::vector<Sum>& bins, Accounting& accounting, std::vector<Sum>* changes = nullptr)
{
    require(nonnegative(number) && nonnegative(moles), "Bubble births must have finite nonnegative amounts.");
    require((number == 0.0) == (moles == 0.0), "A bubble birth must carry both number and moles, or neither.");
    if (number == 0.0)
        return;
    const real_t size = moles / number;
    if (size < grid.mol_per_bubble.front())
    {
        if (underflow == BubbleUnderflowPolicy::Reject)
            throw std::out_of_range("Bubble birth or post-growth size lies below the molar pivot grid.");
        accounting.micro_number += number;
        accounting.micro_moles += moles;
        return;
    }
    if (!std::isfinite(size) || size > grid.mol_per_bubble.back())
        throw std::out_of_range("Bubble birth or post-growth size lies above the molar pivot grid.");
    const auto upper = std::lower_bound(grid.mol_per_bubble.begin(), grid.mol_per_bubble.end(), size);
    const auto hi = static_cast<std::size_t>(upper - grid.mol_per_bubble.begin());
    if (*upper == size)
    {
        bins[hi] += number;
        if (changes)
            (*changes)[hi] += number;
        return;
    }
    const auto lo = hi - 1;
    const real_t high_number =
        number * ((size - grid.mol_per_bubble[lo]) / (grid.mol_per_bubble[hi] - grid.mol_per_bubble[lo]));
    bins[lo] += number - high_number;
    bins[hi] += high_number;
    if (changes)
    {
        (*changes)[lo] += number - high_number;
        (*changes)[hi] += high_number;
    }
}

std::vector<real_t> values(const std::vector<Sum>& sums)
{
    std::vector<real_t> result;
    result.reserve(sums.size());
    for (const auto& sum : sums)
    {
        const auto value = sum.value();
        require_result(nonnegative(value), "Bubble reaction produced an invalid number density.");
        result.push_back(value);
    }
    return result;
}

BubblePopulationContext context(
    const MolarPivotGrid& grid, const UpperBubblePopulation& state, const BubblePopulationEnvironment& environment)
{
    require(environment.liquid_fraction > 0.0 || state.dissolved_inventory == 0.0,
        "Dissolved hydrogen requires positive aqueous liquid volume.");
    require(environment.carrier_fraction > 0.0 || std::all_of(state.number_density.begin(), state.number_density.end(),
                                                      [](real_t number) { return number == 0.0; }),
        "Populated bubble bins require positive nonresolved carrier volume.");
    const auto concentration =
        environment.liquid_fraction > 0.0 ? state.dissolved_inventory / environment.liquid_fraction : real_t{};
    require(nonnegative(concentration), "Dissolved concentration must remain finite and nonnegative.");
    return {grid, state, environment, concentration};
}

void validate_environment(const BubblePopulationEnvironment& e)
{
    constexpr real_t fraction_roundoff = 64 * std::numeric_limits<real_t>::epsilon();
    require(std::isfinite(e.time) && nonnegative(e.temperature) && nonnegative(e.liquid_pressure) &&
                nonnegative(e.liquid_fraction) && nonnegative(e.carrier_fraction) &&
                e.liquid_fraction <= e.carrier_fraction + fraction_roundoff &&
                e.carrier_fraction <= 1.0 + fraction_roundoff && nonnegative(e.liquid_density) &&
                nonnegative(e.dynamic_viscosity) && nonnegative(e.surface_tension) &&
                nonnegative(e.hydrogen_diffusivity) && std::isfinite(e.gas_constant) && e.gas_constant > 0.0,
        "Bubble population environment contains an invalid value.");
}

} // namespace

MicrobubbleSubstepResult advance_microbubble_substep(const MicrobubbleState& initial, const MicrobubbleSubstep& step)
{
    require(nonnegative(initial.dissolved_inventory) && nonnegative(initial.number_density) &&
                nonnegative(initial.molar_inventory),
        "Microbubble inventories must be finite and nonnegative.");
    require(std::isfinite(step.duration) && step.duration > 0.0 && nonnegative(step.bulk_production_rate) &&
                std::isfinite(step.liquid_fraction) && step.liquid_fraction > 0.0 && step.liquid_fraction <= 1.0 &&
                std::isfinite(step.nucleation_moles) && step.nucleation_moles > 0.0 &&
                std::isfinite(step.critical_concentration) && step.critical_concentration > 0.0 &&
                nonnegative(step.micro_decay_fraction) && step.micro_decay_fraction <= 1.0 &&
                nonnegative(step.micro_retention) && step.micro_retention <= 1.0 &&
                std::abs(step.micro_decay_fraction + step.micro_retention - 1.0) <=
                    8.0 * std::numeric_limits<real_t>::epsilon() &&
                nonnegative(step.conversion_pressure_factor),
        "Invalid microbubble substep coefficients.");
    require(step.heaviside_mode == RadiolyticHeavisideMode::Exact ||
                step.heaviside_mode == RadiolyticHeavisideMode::Smoothed,
        "Invalid microbubble threshold model.");
    require(step.heaviside_mode != RadiolyticHeavisideMode::Smoothed ||
                (std::isfinite(step.smooth_heaviside_width) && step.smooth_heaviside_width > 0.0),
        "Invalid microbubble threshold width.");

    MicrobubbleSubstepResult result;
    result.state = initial;
    auto& state = result.state;
    // Keep these operations in the same order as the original legacy loop.
    const auto produced = step.bulk_production_rate * step.duration;
    const auto produced_number = produced / step.nucleation_moles;
    state.molar_inventory += produced;
    state.number_density += produced_number;
    const auto before_decay_number = state.number_density;
    const auto micro_decay_moles = state.molar_inventory * step.micro_decay_fraction;
    state.molar_inventory -= micro_decay_moles;
    state.number_density *= step.micro_retention;
    state.dissolved_inventory += micro_decay_moles;
    const auto dissolved_concentration = state.dissolved_inventory / step.liquid_fraction;
    const auto supersaturation = dissolved_concentration / step.critical_concentration - 1.0;
    const auto activation = smoothed_heaviside(supersaturation, step.heaviside_mode, step.smooth_heaviside_width);
    const auto conversion_frequency =
        step.conversion_pressure_factor * std::max(supersaturation, real_t{}) * activation;
    const auto conversion_fraction = 1.0 - std::exp(-conversion_frequency * step.duration);
    const auto number_to_large = state.number_density * conversion_fraction;
    const auto moles_to_large = std::min(state.molar_inventory, step.nucleation_moles * number_to_large);
    state.number_density -= number_to_large;
    state.molar_inventory -= moles_to_large;
    result.converted = {number_to_large, moles_to_large};
    Sum number_change, molar_change;
    number_change += produced_number;
    number_change += -before_decay_number * step.micro_decay_fraction;
    number_change += -number_to_large;
    molar_change += produced;
    molar_change += -micro_decay_moles;
    molar_change += -moles_to_large;
    result.number_change = number_change.value();
    result.molar_change = molar_change.value();
    result.produced_moles = produced;
    result.dissolved_moles = micro_decay_moles;
    result.decayed_number = before_decay_number - (before_decay_number * step.micro_retention);
    require_result(nonnegative(state.dissolved_inventory) && nonnegative(state.number_density) &&
                       nonnegative(state.molar_inventory) && nonnegative(number_to_large) &&
                       nonnegative(moles_to_large),
        "Microbubble substep produced a nonfinite or negative inventory.");
    Sum balance;
    balance += state.dissolved_inventory;
    balance += state.molar_inventory;
    balance += moles_to_large;
    balance += -initial.dissolved_inventory;
    balance += -initial.molar_inventory;
    balance += -produced;
    result.inventory_error = balance.value();
    check_balance(result.inventory_error, initial.dissolved_inventory + initial.molar_inventory + produced,
        "Microbubble substep failed its hydrogen balance.");
    Sum number_balance;
    number_balance += state.number_density;
    number_balance += number_to_large;
    number_balance += result.decayed_number;
    number_balance += -initial.number_density;
    number_balance += -produced_number;
    result.number_error = number_balance.value();
    check_balance(result.number_error, initial.number_density + produced_number,
        "Microbubble substep failed its physical number balance.");
    return result;
}

BubblePopulationMoments population_moments(const MolarPivotGrid& grid, const UpperBubblePopulation& state)
{
    validate(grid, state);
    Sum number, moles;
    for (std::size_t i = 0; i < state.number_density.size(); ++i)
    {
        number += state.number_density[i];
        moles += state.number_density[i] * grid.mol_per_bubble[i];
    }
    require_result(std::isfinite(number.value()) && std::isfinite(moles.value()), "Bubble moments overflow.");
    return {number.value(), moles.value()};
}

std::vector<real_t> population_bin_moles(const MolarPivotGrid& grid, const UpperBubblePopulation& state)
{
    validate(grid, state);
    std::vector<real_t> result(state.number_density.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = state.number_density[i] * grid.mol_per_bubble[i];
    return result;
}

BubblePopulationStepResult advance_upper_bubble_population(const MolarPivotGrid& grid,
    const UpperBubblePopulation& initial, const BubblePopulationEnvironment& environment, real_t duration,
    const BubblePopulationKernels& kernels, const std::vector<BubblePopulationBirth>& births,
    BubblePopulationStepOptions options)
{
    validate(grid, initial);
    validate_environment(environment);
    require(std::isfinite(duration) && duration > 0.0, "Bubble population substep must be positive and finite.");
    require(
        options.underflow == BubbleUnderflowPolicy::Reject || options.underflow == BubbleUnderflowPolicy::ReturnToMicro,
        "Invalid bubble underflow policy.");
    const bool any_kernel = bool(kernels.growth_rate) || bool(kernels.coalescence_kernel) ||
                            bool(kernels.breakup_frequency) || bool(kernels.breakup_daughters);
    require(
        !any_kernel || (!kernels.model_name.empty() && !kernels.validity_description.empty() && bool(kernels.validate)),
        "Enabled population kernels require an external model name, validity declaration and validator.");
    require(bool(kernels.breakup_frequency) == bool(kernels.breakup_daughters),
        "Breakup requires both frequency and daughter distribution callbacks.");
    BubblePopulationStepResult result;
    result.state = initial;
    auto& state = result.state;
    Accounting accounting;
    std::vector<Sum> changes(initial.number_density.size());
    const auto initial_moments = population_moments(grid, initial);
    (void) context(grid, state, environment);
    std::vector<Sum> deposited(state.number_density.size());
    for (std::size_t i = 0; i < deposited.size(); ++i)
        deposited[i] += state.number_density[i];
    for (const auto& birth : births)
    {
        require(environment.carrier_fraction > 0.0 || birth.number_density == 0.0,
            "A positive bubble birth requires nonresolved carrier volume.");
        deposit(grid, birth.number_density, birth.molar_inventory, options.underflow, deposited, accounting, &changes);
        accounting.birth_number += birth.number_density;
        accounting.birth_moles += birth.molar_inventory;
    }
    state.number_density = values(deposited);

    if (kernels.growth_rate)
    {
        const auto view = context(grid, state, environment);
        kernels.validate(view);
        std::vector<Sum> grown(state.number_density.size());
        Sum uptake, release;
        for (std::size_t i = 0; i < state.number_density.size(); ++i)
        {
            const auto number = state.number_density[i];
            if (number == 0.0)
                continue;
            const auto rate = kernels.growth_rate(i, view);
            require(std::isfinite(rate), "Bubble growth rate must be finite.");
            const auto change = duration * rate;
            if (change == 0.0)
            {
                // Preserve equilibrium exactly, including at the grid endpoints.
                // A multiply/divide round trip through the molar moment need not
                // recover the same representative pivot in floating arithmetic.
                grown[i] += number;
                continue;
            }
            const auto size = grid.mol_per_bubble[i] + change;
            require_result(std::isfinite(size) && size >= 0.0,
                "Bubble growth substep exhausts its gas donor; reduce the substep.");
            const auto amount = number * change;
            require_result(std::isfinite(amount), "Bubble growth amount overflows.");
            if (amount >= 0.0)
                uptake += amount;
            else
                release += -amount;
            // Use the physical size increment directly inside an adjacent
            // interval: xi + change may round back to xi for a small event.
            // This preserves the event in both bins without subtracting two
            // large stored populations to recover its source afterwards.
            if (change > 0 && i + 1 < state.number_density.size() &&
                change <= grid.mol_per_bubble[i + 1] - grid.mol_per_bubble[i])
            {
                const real_t moved = number * (change / (grid.mol_per_bubble[i + 1] - grid.mol_per_bubble[i]));
                grown[i] += number - moved;
                grown[i + 1] += moved;
                changes[i] += -moved;
                changes[i + 1] += moved;
                continue;
            }
            if (change < 0 && i > 0 && -change <= grid.mol_per_bubble[i] - grid.mol_per_bubble[i - 1])
            {
                const real_t moved = number * ((-change) / (grid.mol_per_bubble[i] - grid.mol_per_bubble[i - 1]));
                grown[i] += number - moved;
                grown[i - 1] += moved;
                changes[i] += -moved;
                changes[i - 1] += moved;
                continue;
            }
            if (change > 0 && i + 1 == state.number_density.size())
                throw std::out_of_range("Bubble post-growth size lies above the molar pivot grid.");
            changes[i] += -number;
            if (size == 0.0)
                accounting.dissolved_number += number;
            else if (change < 0 && i == 0)
            {
                if (options.underflow == BubbleUnderflowPolicy::Reject)
                    throw std::out_of_range("Bubble post-growth size lies below the molar pivot grid.");
                accounting.micro_number += number;
                accounting.micro_moles += number * size;
            }
            else
                deposit(grid, number, number * size, options.underflow, grown, accounting, &changes);
        }
        require_result(std::isfinite(uptake.value()) && std::isfinite(release.value()) &&
                           uptake.value() <= state.dissolved_inventory + release.value(),
            "Bubble growth substep exhausts dissolved hydrogen; reduce the substep.");
        state.dissolved_inventory = (state.dissolved_inventory + release.value()) - uptake.value();
        state.number_density = values(grown);
        accounting.growth += uptake.value();
        accounting.dissolution += release.value();
    }

    if (kernels.coalescence_kernel)
    {
        const auto view = context(grid, state, environment);
        kernels.validate(view);
        std::vector<Sum> consumed(state.number_density.size()), born(state.number_density.size());
        for (std::size_t i = 0; i < state.number_density.size(); ++i)
            for (std::size_t j = i; j < state.number_density.size(); ++j)
            {
                if (state.number_density[i] == 0.0 || state.number_density[j] == 0.0)
                    continue;
                const auto rate = kernels.coalescence_kernel(i, j, view);
                require(nonnegative(rate), "Coalescence kernel must be finite and nonnegative.");
                if (rate == 0.0)
                    continue;
                const auto events = (rate * state.number_density[i]) *
                                    (state.number_density[j] / environment.carrier_fraction) * duration *
                                    (i == j ? 0.5 : 1.0);
                require_result(nonnegative(events), "Coalescence event count overflows.");
                if (events == 0.0)
                    continue;
                consumed[i] += events;
                consumed[j] += events;
                changes[i] += -events;
                changes[j] += -events;
                deposit(grid, events, events * (grid.mol_per_bubble[i] + grid.mol_per_bubble[j]), options.underflow,
                    born, accounting, &changes);
                accounting.coalescence += events;
            }
        for (std::size_t i = 0; i < consumed.size(); ++i)
        {
            require_result(consumed[i].value() <= state.number_density[i],
                "Coalescence substep exhausts a number donor; reduce the substep.");
            born[i] += state.number_density[i] - consumed[i].value();
        }
        state.number_density = values(born);
    }

    if (kernels.breakup_frequency)
    {
        const auto view = context(grid, state, environment);
        kernels.validate(view);
        std::vector<Sum> remaining(state.number_density.size());
        for (std::size_t i = 0; i < state.number_density.size(); ++i)
        {
            const auto number = state.number_density[i];
            if (number == 0.0)
                continue;
            const auto frequency = kernels.breakup_frequency(i, view);
            require(nonnegative(frequency), "Breakup frequency must be finite and nonnegative.");
            const auto events = number * (frequency * duration);
            require_result(nonnegative(events) && events <= number,
                "Breakup substep exhausts a number donor; reduce the substep.");
            remaining[i] += number - events;
            if (events == 0.0)
                continue;
            changes[i] += -events;
            const auto daughters = kernels.breakup_daughters(i, view);
            require(!daughters.empty(), "An active breakup must declare its daughter distribution.");
            Sum daughter_number, daughter_moles;
            for (const auto& daughter : daughters)
            {
                require(std::isfinite(daughter.mol_per_bubble) && daughter.mol_per_bubble > 0.0 &&
                            nonnegative(daughter.multiplicity),
                    "Invalid breakup daughter.");
                daughter_number += daughter.multiplicity;
                daughter_moles += daughter.multiplicity * daughter.mol_per_bubble;
                deposit(grid, events * daughter.multiplicity, events * daughter.multiplicity * daughter.mol_per_bubble,
                    options.underflow, remaining, accounting, &changes);
            }
            require(daughter_number.value() >= 1.0, "Breakup daughters cannot reduce the particle count.");
            check_balance(daughter_moles.value() - grid.mol_per_bubble[i], grid.mol_per_bubble[i],
                "Breakup daughters do not conserve the parent's moles.");
            accounting.breakup += events;
            accounting.breakup_created_number += events * (daughter_number.value() - 1.0);
        }
        state.number_density = values(remaining);
    }

    const auto final_moments = population_moments(grid, state);
    result.number_change.resize(changes.size());
    result.hydrogen_change.resize(changes.size());
    for (std::size_t i = 0; i < changes.size(); ++i)
    {
        result.number_change[i] = changes[i].value();
        result.hydrogen_change[i] = grid.mol_per_bubble[i] * result.number_change[i];
        require_result(std::isfinite(result.number_change[i]) && std::isfinite(result.hydrogen_change[i]),
            "Bubble population event ledger overflowed.");
    }
    result.returned_to_micro = {accounting.micro_number.value(), accounting.micro_moles.value()};
    result.dissolved_to_gas_moles = accounting.growth.value();
    result.gas_to_dissolved_moles = accounting.dissolution.value();
    result.coalescence_events = accounting.coalescence.value();
    result.breakup_events = accounting.breakup.value();
    result.breakup_created_number = accounting.breakup_created_number.value();
    result.dissolved_bubble_number = accounting.dissolved_number.value();
    require_result(nonnegative(result.returned_to_micro.number_density) &&
                       nonnegative(result.returned_to_micro.molar_inventory) &&
                       nonnegative(result.dissolved_to_gas_moles) && nonnegative(result.gas_to_dissolved_moles) &&
                       nonnegative(result.coalescence_events) && nonnegative(result.breakup_events) &&
                       nonnegative(result.breakup_created_number) && nonnegative(result.dissolved_bubble_number),
        "Bubble reaction accounting overflowed.");
    result.hydrogen_before = initial.dissolved_inventory + initial_moments.molar_inventory;
    result.hydrogen_added_by_births = accounting.birth_moles.value();
    result.hydrogen_after =
        state.dissolved_inventory + final_moments.molar_inventory + result.returned_to_micro.molar_inventory;
    Sum hydrogen_error, number_error;
    hydrogen_error += state.dissolved_inventory;
    hydrogen_error += final_moments.molar_inventory;
    hydrogen_error += result.returned_to_micro.molar_inventory;
    hydrogen_error += -initial.dissolved_inventory;
    hydrogen_error += -initial_moments.molar_inventory;
    hydrogen_error += -result.hydrogen_added_by_births;
    number_error += final_moments.number_density;
    number_error += result.returned_to_micro.number_density;
    number_error += -initial_moments.number_density;
    number_error += -accounting.birth_number.value();
    number_error += result.coalescence_events;
    number_error += -result.breakup_created_number;
    number_error += result.dissolved_bubble_number;
    result.hydrogen_error = hydrogen_error.value();
    result.number_error = number_error.value();
    check_balance(result.hydrogen_error,
        result.hydrogen_before + result.hydrogen_added_by_births + result.hydrogen_after,
        "Upper-population reactions failed the hydrogen ledger.");
    check_balance(result.number_error,
        initial_moments.number_density + accounting.birth_number.value() + final_moments.number_density +
            result.returned_to_micro.number_density + result.coalescence_events + result.breakup_created_number +
            result.dissolved_bubble_number,
        "Upper-population reactions failed the physical number ledger.");
    return result;
}

} // namespace SimpleFluid::RadiolyticGasPhysics
