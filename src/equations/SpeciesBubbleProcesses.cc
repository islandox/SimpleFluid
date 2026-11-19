#include "equations/SpeciesBubbleProcesses.hh"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numbers>
#include <set>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::BubbleGasPhysics
{
namespace
{
using Phase = Reacting::SpeciesPhase;

void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

bool nonnegative(double value)
{
    return std::isfinite(value) && value >= 0;
}

void validate_step(const MultispeciesPopulationContext& context, double duration)
{
    require(nonnegative(duration), "Population-process duration must be finite and nonnegative.");
    context.chemistry.state.validate(context.chemistry.catalog);
    require(std::isfinite(context.chemistry.environment.bulkVolume) && context.chemistry.environment.bulkVolume > 0,
        "Population processes require a finite positive physical bulk volume.");
}

const PopulationCondition& population(const MultispeciesPopulationContext& context, std::string_view id)
{
    const auto condition = std::ranges::find(context.populations, id, &PopulationCondition::id);
    const auto material = std::ranges::find(
        context.chemistry.state.dispersedPopulations, id, &Reacting::SpeciesPopulationState::id);
    require(condition != context.populations.end() && material != context.chemistry.state.dispersedPopulations.end(),
        "Population process refers to an unknown material/count population.");
    require(std::ranges::count(context.populations, id, &PopulationCondition::id) == 1,
        "Population-process count context must have unique population names.");
    require(nonnegative(condition->number), "Population-process count must be finite and nonnegative.");
    return *condition;
}

void validate_species(const MultispeciesPopulationContext& context, Reacting::SpeciesId id)
{
    const auto& catalog = context.chemistry.catalog;
    require(catalog.allows(id, Phase::Dissolved) && catalog.allows(id, Phase::Dispersed),
        "Population transfer species must allow both Dissolved and Dispersed phases.");
}

double rate_value(const PopulationRate& rate, const MultispeciesPopulationContext& context)
{
    const auto value = rate ? rate(context) : 0;
    require(nonnegative(value), "Supplied population rate must be finite and nonnegative.");
    return value;
}

double integrated(double rate, double duration, double volume = 1)
{
    const double result = rate * duration * volume;
    require(nonnegative(result), "Integrated population request must be finite and nonnegative.");
    return result;
}

double fraction(const PopulationRate& rate, const MultispeciesPopulationContext& context, double duration)
{
    return -std::expm1(-integrated(rate_value(rate, context), duration));
}

void append(MultispeciesPopulationPacket& target, MultispeciesPopulationPacket source)
{
    target.transfers.insert(target.transfers.end(), std::make_move_iterator(source.transfers.begin()),
        std::make_move_iterator(source.transfers.end()));
    target.numberEvents.insert(target.numberEvents.end(), std::make_move_iterator(source.numberEvents.begin()),
        std::make_move_iterator(source.numberEvents.end()));
}
} // namespace

PopulationProcess compose_population_processes(std::vector<PopulationProcess> processes)
{
    for (const auto& process : processes)
        require(static_cast<bool>(process), "A composed population process must be nonempty.");
    return [processes = std::move(processes)](const MultispeciesPopulationContext& context, double duration)
    {
        validate_step(context, duration);
        MultispeciesPopulationPacket result;
        for (const auto& process : processes)
            append(result, process(context, duration));
        return result;
    };
}

PopulationProcess make_nucleation_process(
    std::string id, std::vector<SpeciesMolarRate> species, PopulationRate number_rate)
{
    require(!id.empty() && !species.empty() && number_rate, "Nucleation requires named population, species and count law.");
    std::set<Reacting::SpeciesId> ids;
    for (const auto& law : species)
        require(law.rate && ids.insert(law.species).second, "Nucleation species require unique IDs and supplied rates.");
    return [id = std::move(id), species = std::move(species), number_rate = std::move(number_rate)](
               const MultispeciesPopulationContext& context, double duration)
    {
        validate_step(context, duration);
        (void) population(context, id);
        MultispeciesPopulationPacket result;
        const double volume = context.chemistry.environment.bulkVolume;
        bool has_gas = false;
        for (const auto& law : species)
        {
            validate_species(context, law.species);
            const auto moles = integrated(rate_value(law.rate, context), duration, volume);
            if (moles > 0)
            {
                result.transfers.push_back({law.species, Phase::Dissolved, Phase::Dispersed, {}, id, moles});
                has_gas = true;
            }
        }
        const auto births = integrated(rate_value(number_rate, context), duration, volume);
        require(has_gas == (births > 0), "Nucleation must pair positive gas amounts and physical count births.");
        if (births > 0)
            result.numberEvents.push_back({id, births, 0});
        return result;
    };
}

PopulationProcess make_population_number_process(std::string id, PopulationRate birth_rate, PopulationRate death_rate)
{
    require(!id.empty() && (birth_rate || death_rate), "Population number process requires a name and count law.");
    return [id = std::move(id), birth_rate = std::move(birth_rate), death_rate = std::move(death_rate)](
               const MultispeciesPopulationContext& context, double duration)
    {
        validate_step(context, duration);
        (void) population(context, id);
        const auto volume = context.chemistry.environment.bulkVolume;
        const auto births = integrated(rate_value(birth_rate, context), duration, volume);
        const auto deaths = integrated(rate_value(death_rate, context), duration, volume);
        MultispeciesPopulationPacket result;
        if (births > 0 || deaths > 0)
            result.numberEvents.push_back({id, births, deaths});
        return result;
    };
}

PopulationProcess make_population_dissolution_process(std::string id, PopulationRate frequency)
{
    require(!id.empty() && frequency, "Population dissolution requires a named population and supplied frequency.");
    return [id = std::move(id), frequency = std::move(frequency)](
               const MultispeciesPopulationContext& context, double duration)
    {
        validate_step(context, duration);
        const auto& condition = population(context, id);
        const auto converted = fraction(frequency, context, duration);
        MultispeciesPopulationPacket result;
        for (const auto& species : context.chemistry.catalog.species())
        {
            const auto amount = context.chemistry.state.amount(context.chemistry.catalog, species.id, Phase::Dispersed, id);
            if (amount > 0)
            {
                validate_species(context, species.id);
                const auto moles = integrated(amount, converted);
                if (moles > 0)
                    result.transfers.push_back({species.id, Phase::Dispersed, Phase::Dissolved, id, {}, moles});
            }
        }
        const auto deaths = integrated(condition.number, converted);
        if (deaths > 0)
            result.numberEvents.push_back({id, 0, deaths});
        return result;
    };
}

PopulationProcess make_population_conversion_process(std::string donor, std::string receiver, PopulationRate frequency)
{
    require(!donor.empty() && !receiver.empty() && donor != receiver && frequency,
        "Population conversion requires distinct named populations and a supplied frequency.");
    return [donor = std::move(donor), receiver = std::move(receiver), frequency = std::move(frequency)](
               const MultispeciesPopulationContext& context, double duration)
    {
        validate_step(context, duration);
        const auto& from = population(context, donor);
        (void) population(context, receiver);
        const auto converted = fraction(frequency, context, duration);
        MultispeciesPopulationPacket result;
        for (const auto& species : context.chemistry.catalog.species())
        {
            const auto amount = context.chemistry.state.amount(
                context.chemistry.catalog, species.id, Phase::Dispersed, donor);
            const auto moles = integrated(amount, converted);
            if (moles > 0)
                result.transfers.push_back({species.id, Phase::Dispersed, Phase::Dispersed, donor, receiver, moles});
        }
        const auto number = integrated(from.number, converted);
        if (number > 0)
        {
            result.numberEvents.push_back({donor, 0, number});
            result.numberEvents.push_back({receiver, number, 0});
        }
        return result;
    };
}

PopulationProcess make_film_transfer_process(std::string id, std::vector<SpeciesFilmLaw> species)
{
    require(!id.empty() && !species.empty(), "Film transfer requires a named population and supplied species laws.");
    std::set<Reacting::SpeciesId> ids;
    for (const auto& law : species)
        require(law.coefficient && law.equilibrium_concentration && ids.insert(law.species).second,
            "Film species require unique IDs, supplied coefficients and mixture equilibrium laws.");
    return [id = std::move(id), species = std::move(species)](
               const MultispeciesPopulationContext& context, double duration)
    {
        validate_step(context, duration);
        const auto& condition = population(context, id);
        MultispeciesPopulationPacket result;
        for (const auto& law : species)
        {
            validate_species(context, law.species);
            if (condition.number == 0)
                continue;
            require(std::isfinite(condition.radius) && condition.radius > 0,
                "Occupied film population requires a finite positive radius.");
            const auto volume = context.chemistry.environment.liquidVolume;
            require(std::isfinite(volume) && volume > 0, "Film transfer requires finite positive actual liquid volume.");
            const auto coefficient = law.coefficient(context, condition);
            const auto equilibrium = law.equilibrium_concentration(context, condition);
            require(nonnegative(coefficient) && nonnegative(equilibrium),
                "Supplied film coefficient and equilibrium concentration must be finite and nonnegative.");
            const auto dissolved = context.chemistry.state.amount(
                context.chemistry.catalog, law.species, Phase::Dissolved);
            const double moles = duration * coefficient * 4 * std::numbers::pi * condition.radius * condition.radius *
                                 condition.number * (dissolved / volume - equilibrium);
            require(std::isfinite(moles), "Supplied film request must be finite.");
            if (moles > 0)
                result.transfers.push_back({law.species, Phase::Dissolved, Phase::Dispersed, {}, id, moles});
            else if (moles < 0)
                result.transfers.push_back({law.species, Phase::Dispersed, Phase::Dissolved, id, {}, -moles});
        }
        return result;
    };
}
} // namespace SimpleFluid::BubbleGasPhysics
