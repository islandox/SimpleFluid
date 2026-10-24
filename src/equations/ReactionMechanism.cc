/** @file ReactionMechanism.cc
 * @brief Validated species schema and conservative local reaction previews.
 */
#include "equations/ReactionMechanism.hh"

#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace SimpleFluid::Reacting
{
namespace
{
using Sum = detail::CompensatedSum<long double>;
using Amounts = std::vector<long double>;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

bool nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

bool valid_phase(SpeciesPhase phase)
{
    return static_cast<std::size_t>(phase) < 4;
}

std::uint64_t fingerprint(const std::vector<SpeciesDefinition>& species, std::uint64_t revision)
{
    std::uint64_t hash = 14695981039346656037ULL;
    const auto number = [&hash](std::uint64_t value)
    {
        for (unsigned int byte = 0; byte < 8; ++byte)
        {
            hash ^= (value >> (8 * byte)) & 0xff;
            hash *= 1099511628211ULL;
        }
    };
    const auto string = [&hash, &number](std::string_view value)
    {
        number(value.size());
        for (const unsigned char byte : value)
        {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
    };
    number(revision);
    number(species.size());
    for (const auto& entry : species)
    {
        number(entry.id);
        string(entry.name);
        number(std::bit_cast<std::uint64_t>(entry.molarMass));
        number(entry.elements.size());
        for (const auto& [element, count] : entry.elements)
        {
            string(element);
            number(std::bit_cast<std::uint64_t>(count));
        }
        number(entry.allowedPhases.size());
        for (const auto phase : entry.allowedPhases)
            number(static_cast<std::uint64_t>(phase));
    }
    return hash;
}
double checked(long double value)
{
    const auto result = static_cast<double>(value);
    require(std::isfinite(value) && std::isfinite(result) && (value == 0.0L || result != 0.0),
        "Reaction arithmetic is nonfinite or overflows/underflows storage.");
    return result;
}

void balanced(long double residual, long double scale, const char* message)
{
    constexpr long double tolerance = 1e-12L;
    require(std::isfinite(residual) && std::isfinite(scale) && std::abs(residual) <= tolerance * scale, message);
}

std::size_t slot(const SpeciesCatalog& catalog, const CellSpeciesState& state, SpeciesId species, SpeciesPhase phase,
    std::string_view population)
{
    require(valid_phase(phase), "Invalid species phase.");
    require(state.schemaRevision == catalog.revision() && state.schemaFingerprint == catalog.fingerprint() &&
                state.amounts.size() == 4 * catalog.species().size(),
        "Species state has the wrong schema binding or size.");
    const auto speciesIndex = catalog.index(species);
    const auto count = catalog.species().size();
    if (population.empty())
        return static_cast<std::size_t>(phase) * count + speciesIndex;
    require(phase == SpeciesPhase::Dispersed, "Only dispersed species can select a named population.");
    for (std::size_t i = 0; i < state.dispersedPopulations.size(); ++i)
        if (state.dispersedPopulations[i].id == population)
            return state.amounts.size() + i * count + speciesIndex;
    throw std::invalid_argument("Unknown dispersed species population.");
}

Amounts flatten(const CellSpeciesState& state)
{
    Amounts result(state.amounts.begin(), state.amounts.end());
    for (const auto& population : state.dispersedPopulations)
        result.insert(result.end(), population.amounts.begin(), population.amounts.end());
    return result;
}

CellSpeciesState store(const CellSpeciesState& topology, const Amounts& amounts)
{
    auto result = topology;
    std::size_t index = 0;
    for (auto& value : result.amounts)
        value = checked(amounts.at(index++));
    for (auto& population : result.dispersedPopulations)
        for (auto& value : population.amounts)
            value = checked(amounts.at(index++));
    return result;
}

CellSpeciesState zeros(const CellSpeciesState& topology)
{
    auto result = topology;
    std::fill(result.amounts.begin(), result.amounts.end(), 0.0);
    for (auto& population : result.dispersedPopulations)
        std::fill(population.amounts.begin(), population.amounts.end(), 0.0);
    return result;
}

double volume(const ReactionEnvironment& environment, SourceVolumeBasis basis)
{
    switch (basis)
    {
        case SourceVolumeBasis::LiquidVolume:
            return environment.liquidVolume;
        case SourceVolumeBasis::BulkVolume:
            return environment.bulkVolume;
    }
    throw std::invalid_argument("Invalid source volume basis.");
}

class DonorExhaustion : public std::runtime_error
{
public:
    DonorExhaustion() : std::runtime_error("Reaction step exceeds a jointly reserved donor inventory.") {}
};

struct Event
{
    struct Term
    {
        std::size_t slot;
        long double moles;
    };
    std::vector<Term> terms; ///< Sparse raw integrated stoichiometric/external molar events.
};
using Stocks = std::vector<Sum>;

Amounts inventory_values(const Stocks& stocks)
{
    Amounts result(stocks.size());
    for (std::size_t i = 0; i < stocks.size(); ++i)
    {
        result[i] = stocks[i].value();
        checked(result[i]);
    }
    return result;
}

Event reaction_event(const ReactionMechanism& mechanism, const CellSpeciesState& topology, const Amounts& extents)
{
    Event result;
    for (std::size_t r = 0; r < mechanism.reactions().size(); ++r)
        for (const auto& term : mechanism.reactions()[r].changes)
        {
            const auto i = slot(mechanism.catalog(), topology, term.species, term.phase, term.population);
            const long double delta = static_cast<long double>(term.coefficient) * extents[r];
            checked(delta);
            if (delta != 0.0L)
                result.terms.push_back({i, delta});
        }
    return result;
}

Stocks apply_event(const Stocks& initial, const Event& event)
{
    auto result = initial;
    auto margins = initial;
    for (const auto& term : event.terms)
    {
        result[term.slot] += term.moles;
        if (term.moles < 0.0L)
            margins[term.slot] += term.moles; // Never fund simultaneous outgoing demands with incoming credits.
    }
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        // The comparison is on the residual, not an already-rounded positive gross sum.
        if (margins[i].value() < 0.0L)
            throw DonorExhaustion();
        checked(result[i].value());
        if (result[i].value() < 0.0L)
            throw DonorExhaustion();
    }
    return result;
}

Amounts rates(const ReactionMechanism& mechanism, const CellSpeciesState& state, const ReactionEnvironment& environment,
    const ReactionRateProvider& provider)
{
    const ReactionContext context{mechanism.catalog(), state, environment};
    provider.validate(context);
    const auto supplied = provider.evaluate(context);
    Amounts result(mechanism.reactions().size(), 0.0L);
    std::set<ReactionId> seen;
    for (const auto& rate : supplied)
    {
        require(seen.insert(rate.reaction).second, "A reaction rate was supplied more than once.");
        require(std::isfinite(rate.value), "Reaction rates must be finite.");
        const auto i = mechanism.index(rate.reaction);
        result[i] = static_cast<long double>(rate.value) * volume(environment, rate.basis);
        checked(result[i]);
    }
    return result;
}

Amounts multiply(const Amounts& values, double scale)
{
    Amounts result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        result[i] = values[i] * scale;
        checked(result[i]);
    }
    return result;
}

} // namespace

SpeciesCatalog::SpeciesCatalog(std::vector<SpeciesDefinition> species, std::uint64_t schemaRevision)
    : d_species(std::move(species)), d_revision(schemaRevision), d_fingerprint(0)
{
    require(!d_species.empty() && d_revision != 0, "A species catalog needs species and a nonzero schema revision.");
    std::set<SpeciesId> ids;
    std::set<std::string> names;
    for (const auto& entry : d_species)
    {
        require(ids.insert(entry.id).second && !entry.name.empty() && names.insert(entry.name).second,
            "Species IDs and nonempty names must be unique.");
        require(std::isfinite(entry.molarMass) && entry.molarMass > 0.0,
            "Species molar masses must be finite and positive.");
        require(!entry.allowedPhases.empty(), "Every species needs an allowed phase.");
        std::set<SpeciesPhase> phases;
        for (const auto phase : entry.allowedPhases)
            require(valid_phase(phase) && phases.insert(phase).second, "Allowed phases must be valid and unique.");
        for (const auto& [element, count] : entry.elements)
            require(!element.empty() && std::isfinite(count) && count > 0.0,
                "Element names and positive finite atom counts must be declared.");
    }
    d_fingerprint = Reacting::fingerprint(d_species, d_revision);
}

const std::vector<SpeciesDefinition>& SpeciesCatalog::species() const noexcept
{
    return d_species;
}

std::uint64_t SpeciesCatalog::fingerprint() const noexcept
{
    return d_fingerprint;
}

std::size_t SpeciesCatalog::index(SpeciesId id) const
{
    const auto found = std::find_if(
        d_species.begin(), d_species.end(), [id](const SpeciesDefinition& species) { return species.id == id; });
    require(found != d_species.end(), "Unknown species ID.");
    return static_cast<std::size_t>(found - d_species.begin());
}

std::uint64_t SpeciesCatalog::revision() const noexcept
{
    return d_revision;
}

bool SpeciesCatalog::allows(SpeciesId id, SpeciesPhase phase) const
{
    const auto& allowed = d_species[index(id)].allowedPhases;
    return std::find(allowed.begin(), allowed.end(), phase) != allowed.end();
}

CellSpeciesState::CellSpeciesState(const SpeciesCatalog& catalog)
    : schemaRevision(catalog.revision()), schemaFingerprint(catalog.fingerprint()),
      amounts(4 * catalog.species().size(), 0.0)
{
}

double CellSpeciesState::amount(
    const SpeciesCatalog& catalog, SpeciesId species, SpeciesPhase phase, std::string_view population) const
{
    const auto index = slot(catalog, *this, species, phase, population);
    if (index < amounts.size())
        return amounts.at(index);
    const auto offset = index - amounts.size();
    return dispersedPopulations.at(offset / catalog.species().size()).amounts.at(offset % catalog.species().size());
}

double CellSpeciesState::phase_amount(const SpeciesCatalog& catalog, SpeciesId species, SpeciesPhase phase) const
{
    Sum sum;
    sum += amount(catalog, species, phase);
    if (phase == SpeciesPhase::Dispersed)
        for (const auto& population : dispersedPopulations)
            sum += population.amounts.at(catalog.index(species));
    return checked(sum.value());
}

void CellSpeciesState::set_amount(
    const SpeciesCatalog& catalog, SpeciesId species, SpeciesPhase phase, double value, std::string_view population)
{
    require(nonnegative(value) && (value == 0.0 || catalog.allows(species, phase)),
        "Species amounts must be finite, nonnegative and in an allowed phase.");
    const auto index = slot(catalog, *this, species, phase, population);
    if (index < amounts.size())
        amounts.at(index) = value;
    else
    {
        const auto offset = index - amounts.size();
        dispersedPopulations.at(offset / catalog.species().size()).amounts.at(offset % catalog.species().size()) =
            value;
    }
}

void CellSpeciesState::validate(const SpeciesCatalog& catalog) const
{
    const auto count = catalog.species().size();
    require(schemaRevision == catalog.revision() && schemaFingerprint == catalog.fingerprint() &&
                amounts.size() == 4 * count,
        "Species state has the wrong schema revision or size.");
    for (std::size_t p = 0; p < 4; ++p)
        for (std::size_t i = 0; i < count; ++i)
            require(nonnegative(amounts[p * count + i]) &&
                        (amounts[p * count + i] == 0.0 ||
                            catalog.allows(catalog.species()[i].id, static_cast<SpeciesPhase>(p))),
                "Species inventory is nonfinite, negative or in a forbidden phase.");
    std::set<std::string> ids;
    for (const auto& population : dispersedPopulations)
    {
        require(!population.id.empty() && ids.insert(population.id).second && population.amounts.size() == count,
            "Dispersed populations require unique nonempty IDs and matching species schema.");
        for (std::size_t i = 0; i < count; ++i)
            require(
                nonnegative(population.amounts[i]) &&
                    (population.amounts[i] == 0.0 || catalog.allows(catalog.species()[i].id, SpeciesPhase::Dispersed)),
                "Dispersed species inventory is invalid or forbidden.");
    }
}

ReactionMechanism::ReactionMechanism(std::shared_ptr<const SpeciesCatalog> catalog,
    std::vector<ReactionDefinition> reactions, EnergyConvention convention)
    : d_catalog(std::move(catalog)), d_reactions(std::move(reactions)), d_convention(convention)
{
    require(d_catalog != nullptr, "A reaction mechanism needs a species catalog.");
    require(convention == EnergyConvention::SensibleEnergy || convention == EnergyConvention::IncludesFormationEnergy,
        "Invalid reaction energy convention.");
    std::set<ReactionId> ids;
    std::set<std::string> names;
    for (const auto& reaction : d_reactions)
    {
        require(ids.insert(reaction.id).second && !reaction.name.empty() && names.insert(reaction.name).second &&
                    !reaction.changes.empty(),
            "Reactions need unique IDs, names and nonempty stoichiometry.");
        require(std::isfinite(reaction.heatPerExtent) &&
                    (convention == EnergyConvention::SensibleEnergy || reaction.heatPerExtent == 0.0),
            "Formation-energy storage cannot also receive reaction heat.");
        Sum mass, massScale;
        std::map<std::string, Sum> elements, elementScales;
        std::set<std::tuple<SpeciesId, SpeciesPhase, std::string>> entries;
        bool consumes = false, produces = false;
        for (const auto& term : reaction.changes)
        {
            const auto& species = d_catalog->species()[d_catalog->index(term.species)];
            require(d_catalog->allows(term.species, term.phase) && valid_phase(term.phase) &&
                        std::isfinite(term.coefficient) && term.coefficient != 0.0 &&
                        (term.population.empty() || term.phase == SpeciesPhase::Dispersed) &&
                        entries.emplace(term.species, term.phase, term.population).second,
                "Reaction stoichiometry needs unique valid species compartments and finite nonzero coefficients.");
            consumes = consumes || term.coefficient < 0.0;
            produces = produces || term.coefficient > 0.0;
            const long double weightedMass = static_cast<long double>(species.molarMass) * term.coefficient;
            checked(weightedMass);
            mass += weightedMass;
            massScale += std::abs(weightedMass);
            for (const auto& [element, count] : species.elements)
            {
                const long double weightedCount = static_cast<long double>(count) * term.coefficient;
                checked(weightedCount);
                elements[element] += weightedCount;
                elementScales[element] += std::abs(weightedCount);
            }
        }
        require(consumes && produces, "Internal reactions need both donors and products.");
        balanced(mass.value(), massScale.value(), "Reaction does not conserve declared species mass.");
        for (const auto& [element, sum] : elements)
            balanced(sum.value(), elementScales[element].value(), "Reaction does not conserve a declared element.");
    }
}

const SpeciesCatalog& ReactionMechanism::catalog() const noexcept
{
    return *d_catalog;
}
const std::vector<ReactionDefinition>& ReactionMechanism::reactions() const noexcept
{
    return d_reactions;
}
EnergyConvention ReactionMechanism::energy_convention() const noexcept
{
    return d_convention;
}

std::size_t ReactionMechanism::index(ReactionId id) const
{
    const auto found = std::find_if(
        d_reactions.begin(), d_reactions.end(), [id](const ReactionDefinition& reaction) { return reaction.id == id; });
    require(found != d_reactions.end(), "Unknown reaction ID.");
    return static_cast<std::size_t>(found - d_reactions.begin());
}

double SpeciesSourceLedger::thermal_energy() const
{
    Sum sum;
    sum += externalHeat;
    sum += reactionHeat;
    sum += transferEnergy;
    return checked(sum.value());
}

ReactionAdvance advance_reactions(const ReactionMechanism& mechanism, const CellSpeciesState& initial,
    const ReactionEnvironment& environment, double duration, const ReactionSourceInput& input,
    const ReactionAdvanceOptions& options)
{
    initial.validate(mechanism.catalog());
    require(std::isfinite(environment.time) && std::isfinite(environment.temperature) &&
                environment.temperature > 0.0 && std::isfinite(environment.absolutePressure) &&
                environment.absolutePressure > 0.0 && nonnegative(environment.liquidVolume) &&
                nonnegative(environment.bulkVolume) &&
                (environment.volumeClosure == ReactionVolumeClosure::Accepted ||
                    environment.volumeClosure == ReactionVolumeClosure::ThermodynamicPredictor) &&
                (environment.volumeClosure == ReactionVolumeClosure::ThermodynamicPredictor ||
                    environment.liquidVolume <= environment.bulkVolume) &&
                (environment.liquidVolume == 0 || environment.bulkVolume > 0) && nonnegative(duration) &&
                std::isfinite(environment.time + duration) &&
                (duration == 0.0 || environment.time + duration > environment.time),
        "Reaction context needs finite time, positive temperature/pressure and valid volumes/duration.");
    const auto initialValues = flatten(initial);
    const auto liquidEntries = 2 * mechanism.catalog().species().size();
    require(
        environment.liquidVolume > 0.0 || std::none_of(initial.amounts.begin(), initial.amounts.begin() + liquidEntries,
                                              [](double value) { return value > 0.0; }),
        "Initially occupied liquid/dissolved species require positive physical liquid volume.");
    require(environment.bulkVolume > 0.0 || std::none_of(initialValues.begin(), initialValues.end(),
                                                [](long double value) { return value > 0.0L; }),
        "Occupied species inventories require positive physical bulk volume.");
    require(options.maximumSubstep > 0.0 && !std::isnan(options.maximumSubstep) && options.maximumSubsteps > 0,
        "Reaction subcycling needs a positive step size and a nonzero step limit.");
    require(!input.rates || input.integratedExtents.empty(), "Rates and integrated extents are alternative inputs.");
    require(
        !input.heat.rate || !input.heat.integratedEnergy, "Heat rate and integrated energy are alternative inputs.");
    if (input.rates)
    {
        const auto& provider = *input.rates;
        require(!provider.identity.empty() && provider.revision != 0 &&
                    provider.schemaRevision == mechanism.catalog().revision() &&
                    provider.schemaFingerprint == mechanism.catalog().fingerprint() &&
                    !provider.validityDescription.empty() && !provider.parameterUnits.empty() && provider.validate &&
                    provider.evaluate,
            "Reaction provider requires identity/revision, matching schema, units, domain and pure callbacks.");
    }
    const auto count = flatten(initial).size();
    std::vector<Sum> chemical(count), external(count), extentSums(mechanism.reactions().size());
    std::vector<Sum> forwardExtents(mechanism.reactions().size()), reverseExtents(mechanism.reactions().size());
    const auto record_direction = [&](std::size_t reaction, long double extent)
    {
        if (extent >= 0.0L)
            forwardExtents[reaction] += extent;
        else
            reverseExtents[reaction] += -extent;
    };
    Sum externalHeat, transferEnergy, externalMass;
    std::array<Sum, 3> momentum;
    Amounts extents(mechanism.reactions().size(), 0.0L);
    std::set<ReactionId> seen;
    for (const auto& extent : input.integratedExtents)
    {
        require(std::isfinite(extent.moles) && seen.insert(extent.reaction).second,
            "Integrated reaction extents must be finite and unique.");
        extents[mechanism.index(extent.reaction)] = extent.moles;
    }
    auto combined = reaction_event(mechanism, initial, extents);
    for (const auto& term : combined.terms)
        chemical[term.slot] += term.moles;
    for (const auto& transfer : input.externalTransfers)
    {
        require(!transfer.reservoir.empty() && std::isfinite(transfer.moles) && std::isfinite(transfer.mass) &&
                    std::isfinite(transfer.energy) && mechanism.catalog().allows(transfer.species, transfer.phase),
            "External transfers need a reservoir, allowed species phase and finite physical budgets.");
        const auto i = slot(mechanism.catalog(), initial, transfer.species, transfer.phase, transfer.population);
        const long double expected =
            static_cast<long double>(transfer.moles) *
            mechanism.catalog().species()[mechanism.catalog().index(transfer.species)].molarMass;
        checked(expected);
        balanced(static_cast<long double>(transfer.mass) - expected,
            std::abs(expected) + std::abs(static_cast<long double>(transfer.mass)),
            "External mass must match its signed species amount.");
        external[i] += transfer.moles;
        if (transfer.moles != 0.0)
            combined.terms.push_back({i, static_cast<long double>(transfer.moles)});
        transferEnergy += transfer.energy;
        externalMass += transfer.mass;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            require(std::isfinite(transfer.momentum[axis]), "External momentum must be finite.");
            momentum[axis] += transfer.momentum[axis];
        }
    }
    Stocks initialStocks(count);
    for (std::size_t i = 0; i < count; ++i)
        initialStocks[i] += initialValues[i];
    auto current = apply_event(initialStocks, combined);
    for (std::size_t r = 0; r < extents.size(); ++r)
    {
        extentSums[r] += extents[r];
        record_direction(r, extents[r]);
    }
    if (input.heat.rate)
    {
        require(std::isfinite(input.heat.rate->value), "External heat rate must be finite.");
        externalHeat +=
            static_cast<long double>(input.heat.rate->value) * volume(environment, input.heat.rate->basis) * duration;
    }
    if (input.heat.integratedEnergy)
    {
        require(std::isfinite(*input.heat.integratedEnergy), "Integrated external heat must be finite.");
        externalHeat += *input.heat.integratedEnergy;
    }
    std::size_t substeps = 0, retries = 0;
    double elapsed = 0.0;
    double nextStep = std::min(duration, options.maximumSubstep);
    while (input.rates && elapsed < duration)
    {
        require(substeps < options.maximumSubsteps, "Reaction local subcycle limit exceeded.");
        double step = std::min(nextStep, duration - elapsed);
        std::size_t trialRetries = 0;
        for (;;)
        {
            require(step > 0.0 && elapsed + step > elapsed, "Reaction substep cannot advance representable time.");
            try
            {
                auto stageEnvironment = environment;
                stageEnvironment.time = environment.time + elapsed;
                const double stageEndTime = environment.time + (elapsed + step);
                require(stageEndTime > stageEnvironment.time,
                    "Reaction substep cannot advance representable absolute callback time.");
                const auto firstRates =
                    rates(mechanism, store(initial, inventory_values(current)), stageEnvironment, *input.rates);
                const auto firstExtents = multiply(firstRates, step);
                const auto firstEvent = reaction_event(mechanism, initial, firstExtents);
                const auto stage = apply_event(current, firstEvent);
                stageEnvironment.time = stageEndTime;
                const auto secondRates =
                    rates(mechanism, store(initial, inventory_values(stage)), stageEnvironment, *input.rates);
                const auto secondExtents = multiply(secondRates, step);
                const auto secondEvent = reaction_event(mechanism, initial, secondExtents);
                static_cast<void>(apply_event(stage, secondEvent));
                // Each Euler stage above reserves donors against its actual stage stocks.
                // SSPRK2 then combines the two feasible stages; do not reserve these
                // averaged events against the original stock (stage one may create a stage-two donor).
                auto acceptedChemical = chemical;
                auto accepted = current;
                for (const auto* event : {&firstEvent, &secondEvent})
                    for (const auto& term : event->terms)
                    {
                        const long double delta = term.moles * 0.5L;
                        require(std::isfinite(delta) && (term.moles == 0.0L || delta != 0.0L),
                            "Weighted reaction event is not representable in local accumulation.");
                        acceptedChemical[term.slot] += delta;
                        accepted[term.slot] += delta;
                    }
                for (const auto& stock : accepted)
                {
                    checked(stock.value());
                    if (stock.value() < 0.0L)
                        throw DonorExhaustion();
                }
                current = std::move(accepted);
                chemical = std::move(acceptedChemical);
                for (std::size_t r = 0; r < extents.size(); ++r)
                {
                    // Preserve the independent stage operands until final reporting.
                    extentSums[r] += firstExtents[r] * 0.5L;
                    extentSums[r] += secondExtents[r] * 0.5L;
                    record_direction(r, firstExtents[r] * 0.5L);
                    record_direction(r, secondExtents[r] * 0.5L);
                }
                ++substeps;
                elapsed = step == duration - elapsed ? duration : elapsed + step;
                nextStep = std::min(options.maximumSubstep, step * 2.0);
                break;
            }
            catch (const DonorExhaustion&)
            {
                if (trialRetries++ >= options.maximumRetries)
                    throw;
                ++retries;
                step *= 0.5;
            }
        }
    }
    ReactionAdvance result;
    require(environment.bulkVolume > 0.0 ||
                std::none_of(current.begin(), current.end(), [](const Sum& value) { return value.value() > 0.0L; }),
        "Produced species inventories require positive physical bulk volume.");
    result.state = store(initial, inventory_values(current));
    result.state.validate(mechanism.catalog());
    auto& ledger = result.ledger;
    Amounts chemicalValues(count), externalValues(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        chemicalValues[i] = chemical[i].value();
        externalValues[i] = external[i].value();
    }
    // Carry the actual accepted chemical sums into composition before the reporting state narrows them.
    auto& local = result.localReceipt;
    local.schemaFingerprint = mechanism.catalog().fingerprint();
    local.chemicalChanges.reserve(count);
    const auto speciesCount = mechanism.catalog().species().size();
    std::size_t chemicalIndex = 0;
    for (std::size_t phase = 0; phase < static_cast<std::size_t>(SpeciesPhase::Count); ++phase)
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            const auto value = chemicalValues[chemicalIndex++];
            if (value != 0)
                local.chemicalChanges.push_back({mechanism.catalog().species()[s].id,
                    static_cast<SpeciesPhase>(phase), {}, value});
        }
    for (const auto& population : initial.dispersedPopulations)
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            const auto value = chemicalValues[chemicalIndex++];
            if (value != 0)
                local.chemicalChanges.push_back(
                    {mechanism.catalog().species()[s].id, SpeciesPhase::Dispersed, population.id, value});
        }
    local.externalHeat = externalHeat.value();
    local.reactionHeatContributions.reserve(extentSums.size());

    ledger.chemicalChange = store(zeros(initial), chemicalValues);
    ledger.externalChange = store(zeros(initial), externalValues);
    Sum reactionHeat;
    for (std::size_t r = 0; r < extentSums.size(); ++r)
    {
        ledger.reactionExtents.push_back(checked(extentSums[r].value()));
        ledger.forwardReactionExtents.push_back(checked(forwardExtents[r].value()));
        ledger.reverseReactionExtents.push_back(checked(reverseExtents[r].value()));
        const auto contribution = extentSums[r].value() * mechanism.reactions()[r].heatPerExtent;
        local.reactionHeatContributions.push_back(contribution);
        reactionHeat += contribution;
    }
    ledger.externalTransfers = input.externalTransfers;
    ledger.externalHeat = checked(externalHeat.value());
    ledger.reactionHeat = checked(reactionHeat.value());
    ledger.transferEnergy = checked(transferEnergy.value());
    ledger.externalMass = checked(externalMass.value());
    for (std::size_t axis = 0; axis < 3; ++axis)
        ledger.externalMomentum[axis] = checked(momentum[axis].value());
    checked(static_cast<long double>(ledger.externalHeat) + ledger.reactionHeat + ledger.transferEnergy);
    ledger.substeps = substeps;
    ledger.retries = retries;
    return result;
}

} // namespace SimpleFluid::Reacting
