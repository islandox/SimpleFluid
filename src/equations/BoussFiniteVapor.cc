#include "equations/BoussFiniteVapor.hh"
#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace SimpleFluid::Reacting
{
namespace
{
using Sum = detail::CompensatedSum<long double>;
using Key = std::tuple<SpeciesId, SpeciesPhase, std::string>;
void require(bool value, const char* message)
{
    if (!value)
        throw std::invalid_argument(message);
}
bool positive(double value)
{
    return std::isfinite(value) && value > 0;
}
bool nonnegative(double value)
{
    return std::isfinite(value) && value >= 0;
}
double narrow(long double value)
{
    const double result = static_cast<double>(value);
    require(std::isfinite(value) && std::isfinite(result) && (value == 0 || result != 0),
        "Bouss material/energy receipt cannot be represented.");
    return result;
}
double admitted(long double value)
{
    require(std::isfinite(value) && value >= 0, "Invalid Bouss admission bound.");
    double result = narrow(value);
    if (static_cast<long double>(result) > value)
        result = std::nextafter(result, 0.0);
    require(value == 0 || result > 0, "Bouss admission underflows its positive physical event.");
    return result;
}
bool gas(SpeciesPhase phase)
{
    return phase == SpeciesPhase::Dispersed || phase == SpeciesPhase::Resolved;
}
using RawAmount = std::array<long double, 2>;
void add_raw(RawAmount& total, long double value)
{
    const auto next = total[0] + value;
    if (std::abs(total[0]) >= std::abs(value))
        total[1] += (total[0] - next) + value;
    else
        total[1] += (value - next) + total[0];
    total[0] = next;
}
long double raw_value(const RawAmount& value)
{
    Sum result;
    result += value[0];
    result += value[1];
    return result.value();
}
std::vector<double> flattened(const CellSpeciesState& state)
{
    auto result = state.amounts;
    for (const auto& population : state.dispersedPopulations)
        result.insert(result.end(), population.amounts.begin(), population.amounts.end());
    return result;
}
std::size_t raw_index(const SpeciesCatalog& catalog, const CellSpeciesState& state, SpeciesId species,
    SpeciesPhase phase, std::string_view population)
{
    const auto speciesIndex = catalog.index(species), size = catalog.species().size();
    if (population.empty())
    {
        require(static_cast<std::size_t>(phase) < 4, "Raw Bouss stock has an invalid phase.");
        return static_cast<std::size_t>(phase) * size + speciesIndex;
    }
    require(phase == SpeciesPhase::Dispersed, "Only dispersed raw stocks have named population ownership.");
    const auto found = std::ranges::find(state.dispersedPopulations, population, &SpeciesPopulationState::id);
    require(found != state.dispersedPopulations.end(), "Raw Bouss stock names an unknown population.");
    return state.amounts.size() + static_cast<std::size_t>(found - state.dispersedPopulations.begin()) * size +
           speciesIndex;
}
void validate_raw_stock(const ReactionAdvance& advance)
{
    const auto represented = flattened(advance.state);
    require(advance.localReceipt.postSourceAmounts.size() == represented.size(),
        "Bouss raw post-chemistry stock has an incompatible dense schema.");
    for (std::size_t i = 0; i < represented.size(); ++i)
    {
        const auto& parts = advance.localReceipt.postSourceAmounts[i];
        const auto amount = raw_value(parts);
        require(std::isfinite(parts[0]) && std::isfinite(parts[1]) && amount >= 0 && narrow(amount) == represented[i],
            "Bouss raw post-chemistry stock does not project to its actual state.");
    }
}
Sum volume_margin(const SpeciesCatalog& catalog, const BoussReferenceVolume& volumes, const CellSpeciesState& state,
    long double capacity)
{
    Sum margin;
    margin += capacity;
    for (std::size_t s = 0; s < catalog.species().size(); ++s)
    {
        const auto id = catalog.species()[s].id;
        margin +=
            -static_cast<long double>(state.amount(catalog, id, SpeciesPhase::Resolved)) * volumes.gasMolarVolumes[s];
        margin +=
            -static_cast<long double>(state.amount(catalog, id, SpeciesPhase::Dispersed)) * volumes.gasMolarVolumes[s];
        for (const auto& population : state.dispersedPopulations)
            margin += -static_cast<long double>(population.amounts[s]) * volumes.gasMolarVolumes[s];
    }
    return margin;
}
Sum free_volume(const SpeciesCatalog& catalog, const BoussReferenceVolume& volumes, const CellSpeciesState& state,
    long double capacity)
{
    auto margin = volume_margin(catalog, volumes, state, capacity);
    require(margin.value() >= 0, "Actual Bouss gas inventory exceeds shared scalar capacity.");
    return margin;
}
void debit_gas_budget(
    Sum& margin, const SpeciesCatalog& catalog, const BoussReferenceVolume& volumes, const ReactionAdvance& advance)
{
    for (const auto& change : advance.localReceipt.chemicalChanges)
        if (gas(change.phase))
            margin += -change.moles * volumes.gasMolarVolumes[catalog.index(change.species)];
    for (const auto& change : advance.ledger.externalTransfers)
        if (gas(change.phase))
            margin += -static_cast<long double>(change.moles) * volumes.gasMolarVolumes[catalog.index(change.species)];
    require(margin.value() >= 0, "Actual source events exceed Bouss gas capacity, including sub-ULP material events.");
}
void append_heat(std::vector<long double>& events, const ReactionAdvance& advance)
{
    for (const auto part : advance.localReceipt.externalHeat)
        events.push_back(part);
    for (const auto& value : advance.localReceipt.reactionHeatContributions)
        for (const auto part : value)
            events.push_back(part);
    for (const auto& transfer : advance.ledger.externalTransfers)
        events.push_back(transfer.energy);
}
void combine(ReactionAdvance& first, const ReactionAdvance& second, const SpeciesCatalog& catalog)
{
    validate_raw_stock(first);
    validate_raw_stock(second);
    const auto input = flattened(first.state); // Actual rounded stock used by the fresh admitted kernel.
    auto rawEndpoint = first.localReceipt.postSourceAmounts;
    require(rawEndpoint.size() == second.localReceipt.postSourceAmounts.size(),
        "Admitted chemistry changed the raw compartment schema.");
    for (std::size_t i = 0; i < rawEndpoint.size(); ++i)
    {
        // Keep earlier sub-ULP material while appending the next kernel's actual
        // event delta: its raw final stock minus its actual rounded input.
        add_raw(rawEndpoint[i], second.localReceipt.postSourceAmounts[i][0]);
        add_raw(rawEndpoint[i], second.localReceipt.postSourceAmounts[i][1]);
        add_raw(rawEndpoint[i], -static_cast<long double>(input[i]));
    }
    first.state = second.state;
    std::size_t slot = 0;
    const auto publish = [&](double& value)
    {
        const auto amount = raw_value(rawEndpoint[slot++]);
        require(std::isfinite(amount) && amount >= 0, "Admitted chemistry overdraws raw prior stock.");
        value = narrow(amount);
    };
    for (auto& value : first.state.amounts)
        publish(value);
    for (auto& population : first.state.dispersedPopulations)
        for (auto& value : population.amounts)
            publish(value);
    first.localReceipt.postSourceAmounts = std::move(rawEndpoint);
    // Retain individual raw keyed entries; do not reconstruct them from rounded extents.
    first.localReceipt.chemicalChanges.insert(first.localReceipt.chemicalChanges.end(),
        second.localReceipt.chemicalChanges.begin(), second.localReceipt.chemicalChanges.end());
    for (std::size_t r = 0; r < first.ledger.reactionExtents.size(); ++r)
    {
        const auto add = [](auto a, auto b)
        {
            Sum sum;
            sum += a;
            sum += b;
            return sum.value();
        };
        first.ledger.reactionExtents[r] =
            narrow(add(first.ledger.reactionExtents[r], second.ledger.reactionExtents[r]));
        first.ledger.forwardReactionExtents[r] =
            narrow(add(first.ledger.forwardReactionExtents[r], second.ledger.forwardReactionExtents[r]));
        first.ledger.reverseReactionExtents[r] =
            narrow(add(first.ledger.reverseReactionExtents[r], second.ledger.reverseReactionExtents[r]));
        for (const auto part : second.localReceipt.reactionHeatContributions[r])
            add_raw(first.localReceipt.reactionHeatContributions[r], part);
    }
    std::map<Key, Sum> changes;
    for (const auto& entry : first.localReceipt.chemicalChanges)
        changes[{entry.species, entry.phase, entry.population}] += entry.moles;
    std::fill(first.ledger.chemicalChange.amounts.begin(), first.ledger.chemicalChange.amounts.end(), 0);
    for (auto& population : first.ledger.chemicalChange.dispersedPopulations)
        std::fill(population.amounts.begin(), population.amounts.end(), 0);
    for (const auto& [key, value] : changes)
    {
        const auto species = catalog.index(std::get<0>(key));
        if (std::get<2>(key).empty())
            first.ledger.chemicalChange
                .amounts[static_cast<std::size_t>(std::get<1>(key)) * catalog.species().size() + species] =
                narrow(value.value());
        else
        {
            auto& populations = first.ledger.chemicalChange.dispersedPopulations;
            const auto found = std::ranges::find(populations, std::get<2>(key), &SpeciesPopulationState::id);
            require(found != populations.end(), "Gas request names an unknown population.");
            found->amounts[species] = narrow(value.value());
        }
    }
    Sum reactionHeat;
    for (const auto& value : first.localReceipt.reactionHeatContributions)
        for (const auto part : value)
            reactionHeat += part;
    first.ledger.reactionHeat = narrow(reactionHeat.value());
    first.ledger.substeps += second.ledger.substeps;
    first.ledger.retries += second.ledger.retries;
}
std::vector<BubbleGasPhysics::PopulationNumberState> advance_numbers(const SpeciesCatalog& catalog,
    const CellSpeciesState& before, const CellSpeciesState& after,
    std::span<const BubbleGasPhysics::PopulationNumberState> initial,
    std::span<const BubbleGasPhysics::PopulationNumberEvent> events)
{
    require(
        initial.size() == after.dispersedPopulations.size(), "Bouss population number topology differs from material.");
    for (const auto& species : catalog.species())
        require(before.amount(catalog, species.id, SpeciesPhase::Dispersed) == 0 &&
                    after.amount(catalog, species.id, SpeciesPhase::Dispersed) == 0,
            "Bouss dispersed species must have an explicit named count owner.");
    std::vector<BubbleGasPhysics::PopulationNumberState> result(initial.begin(), initial.end());
    std::map<std::string, std::size_t> index;
    std::vector<Sum> margins(initial.size()), endpoints(initial.size());
    for (std::size_t p = 0; p < initial.size(); ++p)
    {
        require(initial[p].population == after.dispersedPopulations[p].id && nonnegative(initial[p].number) &&
                    index.emplace(initial[p].population, p).second,
            "Bouss population identity/count is invalid.");
        margins[p] += initial[p].number;
        endpoints[p] += initial[p].number;
        Sum moles;
        for (double amount : before.dispersedPopulations[p].amounts)
            moles += amount;
        require(
            (moles.value() == 0) == (initial[p].number == 0), "Transported Bouss gas/count occupancy is inconsistent.");
    }
    for (const auto& event : events)
    {
        require(index.contains(event.population) && nonnegative(event.births) && nonnegative(event.deaths),
            "Bouss population event is invalid.");
        const auto p = index.at(event.population);
        margins[p] += -static_cast<long double>(event.deaths); // Never borrow births.
        endpoints[p] += -static_cast<long double>(event.deaths);
        endpoints[p] += event.births;
    }
    for (std::size_t p = 0; p < initial.size(); ++p)
    {
        require(margins[p].value() >= 0 && endpoints[p].value() >= 0,
            "Bouss population deaths overdraw actual transported count.");
        result[p].number = narrow(endpoints[p].value());
        Sum moles;
        for (const double amount : after.dispersedPopulations[p].amounts)
            moles += amount;
        require((moles.value() == 0) == (result[p].number == 0),
            "Bouss gas birth/disappearance requires explicit paired population events.");
    }
    return result;
}
} // namespace

std::uint64_t bouss_configuration_signature(const BoussReferenceVolume& volumes, const BoussCoordinatedSources& sources)
{
    std::uint64_t result = 14695981039346656037ULL;
    const auto integer = [&result](std::uint64_t value)
    {
        for (unsigned int byte = 0; byte < 8; ++byte)
        {
            result ^= (value >> (byte * 8)) & 255;
            result *= 1099511628211ULL;
        }
    };
    const auto number = [&](double value) { integer(std::bit_cast<std::uint64_t>(value)); };
    const auto text = [&](std::string_view value)
    {
        integer(value.size());
        for (const unsigned char byte : value)
            integer(byte);
    };
    text(volumes.identity);
    integer(volumes.revision);
    integer(volumes.schemaFingerprint);
    number(volumes.liquidDensity);
    number(volumes.thermalDensity);
    number(volumes.heatCapacity);
    integer(volumes.gasMolarVolumes.size());
    for (const double value : volumes.gasMolarVolumes)
        number(value);
    text(sources.identity);
    integer(sources.revision);
    integer(sources.schemaFingerprint);
    integer(bool(sources.gas));
    integer(bool(sources.vapor));
    integer(bool(sources.populationEvents));
    integer(bool(sources.populationProcess));
    if (sources.gas)
    {
        text(sources.gas->identity);
        integer(sources.gas->revision);
        integer(sources.gas->schemaFingerprint);
        integer(bool(sources.gas->request));
    }
    if (sources.vapor)
    {
        integer(sources.vapor->solvent);
        integer(static_cast<std::uint64_t>(sources.vapor->vaporPhase));
        text(sources.vapor->population);
        number(sources.vapor->saturationTemperature);
        number(sources.vapor->latentHeat);
        number(sources.vapor->timeScale);
        number(sources.vapor->sensibleCorrection);
        integer(sources.vapor->bulkEnabled);
        integer(static_cast<std::uint64_t>(sources.vapor->admissionPolicy));
        integer(sources.vapor->walls.size());
        for (const auto& wall : sources.vapor->walls)
        {
            text(wall.patch);
            number(wall.inwardHeatFlux);
            number(wall.evaporationFraction);
        }
    }
    return result;
}
void validate_bouss_reference(const ReactionMechanism& mechanism, const BoussReferenceVolume& volumes)
{
    const auto& catalog = mechanism.catalog();
    require(mechanism.energy_convention() == EnergyConvention::SensibleEnergy && !volumes.identity.empty() &&
                volumes.revision > 0 && volumes.schemaFingerprint == catalog.fingerprint() &&
                positive(volumes.liquidDensity) && positive(volumes.thermalDensity) && positive(volumes.heatCapacity) &&
                volumes.gasMolarVolumes.size() == catalog.species().size(),
        "Bouss reference-volume/scalar-energy binding is invalid.");
    for (std::size_t s = 0; s < catalog.species().size(); ++s)
    {
        const auto id = catalog.species()[s].id;
        const bool allowed = catalog.allows(id, SpeciesPhase::Resolved) || catalog.allows(id, SpeciesPhase::Dispersed);
        require(allowed ? positive(volumes.gasMolarVolumes[s]) : volumes.gasMolarVolumes[s] == 0,
            "Supply an explicit positive molar volume exactly for gas-capable catalog species.");
    }
}
void validate_bouss_sources(
    const ReactionMechanism& mechanism, const BoussReferenceVolume& volumes, const BoussCoordinatedSources& sources)
{
    validate_bouss_reference(mechanism, volumes);
    require(!sources.identity.empty() && sources.revision > 0 &&
                sources.schemaFingerprint == mechanism.catalog().fingerprint(),
        "Bouss source configuration metadata is invalid.");
    if (sources.gas)
        require(!sources.gas->identity.empty() && sources.gas->revision > 0 && sources.gas->request &&
                    sources.gas->schemaFingerprint == mechanism.catalog().fingerprint(),
            "Bouss gas-request provider metadata is invalid.");
    if (sources.vapor)
    {
        const auto& vapor = *sources.vapor;
        const auto& catalog = mechanism.catalog();
        require(catalog.allows(vapor.solvent, SpeciesPhase::Liquid) && gas(vapor.vaporPhase) &&
                    catalog.allows(vapor.solvent, vapor.vaporPhase) &&
                    ((vapor.vaporPhase == SpeciesPhase::Dispersed) == !vapor.population.empty()) &&
                    positive(vapor.saturationTemperature) && positive(vapor.latentHeat) && positive(vapor.timeScale) &&
                    std::isfinite(vapor.sensibleCorrection) && vapor.latentHeat - vapor.sensibleCorrection > 0,
            "Bouss finite solvent/vapor binding is invalid.");
        require(vapor.admissionPolicy == BoussVaporAdmissionPolicy::ProportionalRequests,
            "Bouss bulk/wall proposals require an explicit supported joint admission policy.");
        std::vector<std::string> patches;
        for (const auto& wall : vapor.walls)
        {
            require(!wall.patch.empty() && std::ranges::find(patches, wall.patch) == patches.end() &&
                        nonnegative(wall.inwardHeatFlux) && nonnegative(wall.evaporationFraction) &&
                        wall.evaporationFraction <= 1,
                "Bouss wall heat binding needs unique patches, actual nonnegative heat and a fraction in [0,1].");
            patches.push_back(wall.patch);
        }
    }
}
double bouss_gas_volume(
    const SpeciesCatalog& catalog, const BoussReferenceVolume& volumes, const CellSpeciesState& state)
{
    state.validate(catalog);
    require(volumes.schemaFingerprint == catalog.fingerprint() &&
                volumes.gasMolarVolumes.size() == catalog.species().size() && positive(volumes.liquidDensity),
        "Bouss volume query uses a foreign or malformed reference binding.");
    Sum value;
    for (std::size_t s = 0; s < catalog.species().size(); ++s)
    {
        const auto id = catalog.species()[s].id;
        value += static_cast<long double>(state.phase_amount(catalog, id, SpeciesPhase::Resolved)) *
                 volumes.gasMolarVolumes.at(s);
        // Avoid aggregating rounded phase_amount for independently retained named bins.
        value +=
            static_cast<long double>(state.amount(catalog, id, SpeciesPhase::Dispersed)) * volumes.gasMolarVolumes[s];
        for (const auto& p : state.dispersedPopulations)
            value += static_cast<long double>(p.amounts[s]) * volumes.gasMolarVolumes[s];
    }
    return narrow(value.value());
}
double bouss_liquid_volume(
    const SpeciesCatalog& catalog, const BoussReferenceVolume& volumes, const CellSpeciesState& state)
{
    state.validate(catalog);
    require(volumes.schemaFingerprint == catalog.fingerprint() &&
                volumes.gasMolarVolumes.size() == catalog.species().size() && positive(volumes.liquidDensity),
        "Bouss volume query uses a foreign or malformed reference binding.");
    Sum value;
    for (const auto& species : catalog.species())
        for (const auto phase : {SpeciesPhase::Liquid, SpeciesPhase::Dissolved})
            value += static_cast<long double>(state.amount(catalog, species.id, phase)) * species.molarMass /
                     volumes.liquidDensity;
    return narrow(value.value());
}
BubbleGasPhysics::MultispeciesPopulationResult advance_bouss_population_exchange(const SpeciesCatalog& catalog,
    const BoussReferenceVolume& volumes, const CellSpeciesState& initial,
    std::span<const BubbleGasPhysics::PopulationNumberState> numbers, const ReactionEnvironment& environment,
    double alphaMaximum, const BubbleGasPhysics::MultispeciesPopulationPacket& packet)
{
    std::vector<ReactionDefinition> definitions;
    ReactionSourceInput input;
    for (std::size_t i = 0; i < packet.transfers.size(); ++i)
    {
        const auto& transfer = packet.transfers[i];
        require(i <= std::numeric_limits<ReactionId>::max() && nonnegative(transfer.moles) &&
                    (transfer.donor != transfer.receiver || transfer.donorPopulation != transfer.receiverPopulation),
            "Bouss paired population transfer is invalid.");
        definitions.push_back({static_cast<ReactionId>(i), "supplied paired Bouss phase transfer " + std::to_string(i),
            {{transfer.species, transfer.donor, -1, transfer.donorPopulation},
                {transfer.species, transfer.receiver, 1, transfer.receiverPopulation}},
            0});
        input.integratedExtents.push_back({static_cast<ReactionId>(i), transfer.moles});
    }
    const ReactionMechanism transferMechanism(
        std::make_shared<const SpeciesCatalog>(catalog), std::move(definitions), EnergyConvention::SensibleEnergy);
    validate_bouss_reference(transferMechanism, volumes);
    require(positive(environment.bulkVolume) && nonnegative(alphaMaximum) && alphaMaximum < 1,
        "Bouss population packet requires a finite reference volume/capacity.");
    auto capacity =
        free_volume(catalog, volumes, initial, static_cast<long double>(environment.bulkVolume) * alphaMaximum);
    const auto material = advance_reactions(transferMechanism, initial, environment, 0, input);
    debit_gas_budget(capacity, catalog, volumes, material);
    auto acceptedNumbers = advance_numbers(catalog, initial, material.state, numbers, packet.numberEvents);
    std::vector<Sum> changes(numbers.size());
    for (const auto& event : packet.numberEvents)
        for (std::size_t p = 0; p < numbers.size(); ++p)
            if (numbers[p].population == event.population)
            {
                changes[p] += -static_cast<long double>(event.deaths);
                changes[p] += event.births;
            }
    std::vector<double> numberChanges;
    for (const auto& value : changes)
        numberChanges.push_back(narrow(value.value()));
    // Requires the separately owned field raw-count receipt API. Feed every
    // original signed event before a reporting sum can erase a small remainder.
    std::vector<BubbleGasPhysics::PopulationNumberChangeReceipt> rawChanges(numbers.size());
    const auto add = [](auto& sum, long double value)
    {
        const auto next = sum.high + value;
        if (std::abs(sum.high) >= std::abs(value))
            sum.correction += (sum.high - next) + value;
        else
            sum.correction += (value - next) + sum.high;
        sum.high = next;
    };
    for (const auto& event : packet.numberEvents)
        for (std::size_t p = 0; p < numbers.size(); ++p)
            if (numbers[p].population == event.population)
            {
                add(rawChanges[p], -static_cast<long double>(event.deaths));
                add(rawChanges[p], event.births);
            }
    return {
        material.state, std::move(acceptedNumbers), material.ledger, std::move(numberChanges), std::move(rawChanges)};
}
BoussSourceAdvance advance_bouss_sources(const ReactionMechanism& mechanism, const BoussReferenceVolume& volumes,
    const BoussCoordinatedSources& sources, const CellSpeciesState& initial,
    std::span<const BubbleGasPhysics::PopulationNumberState> numbers, const ReactionEnvironment& environment,
    double duration, double alphaMaximum, const ReactionSourceInput& actual, const ReactionAdvanceOptions& options,
    const BoussWallHeatBudget* wallHeat)
{
    validate_bouss_sources(mechanism, volumes, sources);
    if (sources.vapor && !sources.vapor->walls.empty())
        require(wallHeat && wallHeat->producer &&
                    wallHeat->configurationSignature == bouss_configuration_signature(volumes, sources) &&
                    wallHeat->duration == duration && wallHeat->intervalEnd == environment.time + duration,
            "Wall evaporation needs its certified heat-owner packet before any source callback.");
    if (wallHeat)
    {
        require(sources.vapor && wallHeat->producer &&
                    wallHeat->configurationSignature == bouss_configuration_signature(volumes, sources) &&
                    wallHeat->duration == duration && wallHeat->intervalEnd == environment.time + duration,
            "Foreign/stale wall heat packet before source evaluation.");
        for (const auto& packet : wallHeat->packets)
            require(packet.wall < sources.vapor->walls.size() && packet.geometryFace >= 0 &&
                        std::isfinite(packet.inwardEnergy) && packet.inwardEnergy >= 0,
                "Malformed wall heat event before source evaluation.");
    }
    require(positive(duration) && positive(environment.bulkVolume) && nonnegative(alphaMaximum) && alphaMaximum < 1,
        "Bouss admission needs positive duration/reference CV and a valid scalar capacity.");
    const auto& catalog = mechanism.catalog();
    const long double capacity = static_cast<long double>(environment.bulkVolume) * alphaMaximum;
    auto capacityMargin = free_volume(catalog, volumes, initial, capacity);
    (void) advance_numbers(catalog, initial, initial, numbers, {}); // Validate before any supplied callback.

    BoussSourceAdvance result;
    result.chemistry = advance_reactions(mechanism, initial, environment, duration, actual, options);
    append_heat(result.thermalContributions, result.chemistry);
    validate_raw_stock(result.chemistry);
    auto state = result.chemistry.state;
    debit_gas_budget(capacityMargin, catalog, volumes, result.chemistry); // Actual source never receives scaling.
    (void) free_volume(catalog, volumes, state, capacity);
    // Chemistry is one preceding atomic stage. All constitutively admitted gas
    // requests and vapor then share one donor reservation seeded from its actual
    // accepted endpoint. No admitted request/vapor credit funds another outgoing
    // event in this admission stage, including sub-ULP debits beside huge stocks.
    std::map<Key, Sum> donorMargins;
    for (const auto& species : catalog.species())
    {
        for (std::size_t p = 0; p < 4; ++p)
        {
            const auto phase = static_cast<SpeciesPhase>(p);
            const auto& parts =
                result.chemistry.localReceipt.postSourceAmounts[raw_index(catalog, state, species.id, phase, {})];
            donorMargins[{species.id, phase, {}}] += parts[0];
            donorMargins[{species.id, phase, {}}] += parts[1];
        }
        for (const auto& population : state.dispersedPopulations)
        {
            const auto& parts =
                result.chemistry.localReceipt
                    .postSourceAmounts[raw_index(catalog, state, species.id, SpeciesPhase::Dispersed, population.id)];
            donorMargins[{species.id, SpeciesPhase::Dispersed, population.id}] += parts[0];
            donorMargins[{species.id, SpeciesPhase::Dispersed, population.id}] += parts[1];
        }
    }
    auto admissionEnvironment = environment;
    admissionEnvironment.liquidVolume = bouss_liquid_volume(catalog, volumes, state);
    if (sources.gas)
    {
        const ReactionContext context{catalog, state, admissionEnvironment};
        const auto requests = sources.gas->request(context, duration);
        for (const auto& request : requests)
        {
            require(nonnegative(request.moles), "Bouss gas request must be a finite forward extent.");
            const auto& reaction = mechanism.reactions().at(mechanism.index(request.reaction));
            Sum gasGrowth;
            std::map<Key, Sum> demand;
            for (const auto& term : reaction.changes)
            {
                if (gas(term.phase))
                    gasGrowth += static_cast<long double>(term.coefficient) *
                                 volumes.gasMolarVolumes[catalog.index(term.species)];
                if (term.coefficient < 0)
                    demand[{term.species, term.phase, term.population}] += -static_cast<long double>(term.coefficient);
            }
            require(gasGrowth.value() > 0, "Only net gas-producing reactions may request Bouss admission.");
            long double limit = request.moles;
            limit = std::min(limit, capacityMargin.value() / gasGrowth.value());
            for (const auto& [key, amount] : demand)
                limit = std::min(limit, donorMargins.at(key).value() / amount.value());
            double accepted = admitted(limit);
            if (accepted > 0)
            {
                const auto reserveDonors = [&](double extent)
                {
                    auto margins = donorMargins;
                    for (const auto& term : reaction.changes)
                        if (term.coefficient < 0)
                        {
                            auto& margin = margins.at({term.species, term.phase, term.population});
                            margin += static_cast<long double>(extent) * term.coefficient;
                            require(margin.value() >= 0,
                                "Admitted gas request exhausts a raw finite donor; event needs a "
                                "smaller representable extent.");
                        }
                    return margins;
                };
                // A collapsed positive bound must not hide a raw finite-donor
                // overdraw. Reject it before considering stored-volume rounding.
                (void) reserveDonors(accepted);
                struct RequestTrial
                {
                    ReactionAdvance next;
                    ReactionAdvance aggregate;
                    Sum capacity;
                };
                const auto preview = [&](double extent)
                {
                    ReactionSourceInput input;
                    input.integratedExtents.push_back({request.reaction, extent});
                    auto next = advance_reactions(mechanism, state, admissionEnvironment, duration, input, options);
                    auto nextCapacity = capacityMargin;
                    debit_gas_budget(nextCapacity, catalog, volumes, next);
                    auto aggregate = result.chemistry;
                    combine(aggregate, next, catalog);
                    return RequestTrial{std::move(next), std::move(aggregate), std::move(nextCapacity)};
                };
                auto trial = preview(accepted);
                if (volume_margin(catalog, volumes, trial.aggregate.state, capacity).value() < 0)
                {
                    // Event-space admission can fit while rounding each stored
                    // product independently crosses the same strict capacity.
                    // Search only this requested extent, retaining a tested
                    // feasible endpoint. Actual chemistry is never rescaled.
                    double low = 0, high = accepted;
                    std::optional<RequestTrial> feasible;
                    for (unsigned iteration = 0; iteration < 80; ++iteration)
                    {
                        const double middle = std::midpoint(low, high);
                        if (middle == low || middle == high)
                            break;
                        auto candidate = preview(middle);
                        if (volume_margin(catalog, volumes, candidate.aggregate.state, capacity).value() >= 0)
                        {
                            low = middle;
                            feasible = std::move(candidate);
                        }
                        else
                            high = middle;
                    }
                    accepted = low;
                    if (feasible)
                        trial = std::move(*feasible);
                }
                if (accepted > 0)
                {
                    // Trial evaluation has no provider callback or publication;
                    // only the final accepted event enters receipts and budgets.
                    donorMargins = reserveDonors(accepted);
                    capacityMargin = std::move(trial.capacity);
                    append_heat(result.thermalContributions, trial.next);
                    result.chemistry = std::move(trial.aggregate);
                    state = result.chemistry.state;
                    admissionEnvironment.liquidVolume = bouss_liquid_volume(catalog, volumes, state);
                    (void) free_volume(catalog, volumes, state, capacity);
                }
            }
            result.admissions.push_back({request.reaction, request.moles, accepted});
        }
    }
    Sum thermal;
    for (const auto event : result.thermalContributions)
        thermal += event;
    if (sources.vapor)
    {
        const auto& vapor = *sources.vapor;
        const auto species = catalog.index(vapor.solvent);
        const double molarMass = catalog.species()[species].molarMass;
        const long double capacityT =
            static_cast<long double>(environment.bulkVolume) * volumes.thermalDensity * volumes.heatCapacity;
        const long double delta = environment.temperature - vapor.saturationTemperature;
        const long double sensible = capacityT * std::abs(delta);
        const long double latentPerMole = static_cast<long double>(vapor.latentHeat) * molarMass;
        const long double coolingPerMole =
            static_cast<long double>(vapor.latentHeat - vapor.sensibleCorrection) * molarMass;
        require(vapor.walls.empty() ||
                    (wallHeat && wallHeat->producer &&
                        wallHeat->configurationSignature == bouss_configuration_signature(volumes, sources) &&
                        wallHeat->duration == duration && wallHeat->intervalEnd == environment.time + duration),
            "Wall evaporation requires its exact host-certified heat-owner interval/configuration packet.");
        if (wallHeat)
            require(wallHeat->configurationSignature == bouss_configuration_signature(volumes, sources) &&
                        wallHeat->duration == duration && wallHeat->intervalEnd == environment.time + duration,
                "Foreign/stale Bouss wall heat packet.");
        Sum liquidStock, vaporStock, latentEnergy, sensibleEnergy, evaporationMoles;
        const auto& liquidParts =
            result.chemistry.localReceipt
                .postSourceAmounts[raw_index(catalog, state, vapor.solvent, SpeciesPhase::Liquid, {})];
        const auto& vaporParts =
            result.chemistry.localReceipt
                .postSourceAmounts[raw_index(catalog, state, vapor.solvent, vapor.vaporPhase, vapor.population)];
        liquidStock += liquidParts[0];
        liquidStock += liquidParts[1];
        vaporStock += vaporParts[0];
        vaporStock += vaporParts[1];
        auto& liquidDonor = donorMargins.at({vapor.solvent, SpeciesPhase::Liquid, {}});
        auto& vaporDonor = donorMargins.at({vapor.solvent, vapor.vaporPhase, vapor.population});
        const auto transfer = [&](bool evaporation, double amount)
        {
            if (amount == 0)
                return;
            auto& margin = evaporation ? liquidDonor : vaporDonor;
            margin += -static_cast<long double>(amount);
            require(margin.value() >= 0, "Combined admitted vapor proposals overdraw the raw donor margin.");
            liquidStock += (evaporation ? -1.0L : 1.0L) * amount;
            vaporStock += (evaporation ? 1.0L : -1.0L) * amount;
            capacityMargin += (evaporation ? -1.0L : 1.0L) * amount * volumes.gasMolarVolumes[species];
            require(liquidStock.value() >= 0 && vaporStock.value() >= 0 && capacityMargin.value() >= 0,
                "Combined vapor material/capacity endpoint is infeasible.");
            result.transfers.push_back({vapor.solvent, evaporation ? SpeciesPhase::Liquid : vapor.vaporPhase,
                evaporation ? vapor.vaporPhase : SpeciesPhase::Liquid, evaporation ? "" : vapor.population,
                evaporation ? vapor.population : "", amount});
            const long double sign = evaporation ? -1 : 1;
            const long double latent = sign * amount * molarMass * vapor.latentHeat;
            const long double correction = -sign * amount * molarMass * vapor.sensibleCorrection;
            latentEnergy += latent;
            sensibleEnergy += correction;
            result.vapor.thermalContributions.push_back(latent);
            result.vapor.thermalContributions.push_back(correction);
            result.thermalContributions.push_back(latent);
            result.thermalContributions.push_back(correction);
            thermal += latent;
            thermal += correction;
        };
        // Condensation reserves only the actual pre-admission vapor. Its incoming
        // liquid is deliberately not credited to the shared outgoing solvent margin.
        if (vapor.bulkEnabled && delta < 0)
        {
            const long double requested = sensible * duration / vapor.timeScale / latentPerMole;
            require(std::isfinite(requested), "Bouss condensation request overflows.");
            // Accepted source heating already spends the initial subcooling.
            // Preserve raw channels so a small remaining budget survives large
            // canceling heats. Cooling cannot enlarge the constitutive request.
            Sum subcooling;
            subcooling += sensible;
            for (const auto event : result.thermalContributions)
                subcooling += -event;
            const auto available = std::clamp(subcooling.value(), 0.0L, sensible);
            const long double limit = std::min({requested, vaporDonor.value(),
                static_cast<long double>(state.amount(catalog, vapor.solvent, vapor.vaporPhase, vapor.population)),
                available / coolingPerMole});
            result.vapor.requestedCondensation = narrow(requested);
            result.vapor.condensation = admitted(limit);
            transfer(false, result.vapor.condensation);
        }
        std::vector<BoussEvaporationAdmission> proposals;
        Sum requestedMoles, allocatedWallHeat;
        if (vapor.bulkEnabled && delta > 0)
        {
            const auto request = sensible * duration / vapor.timeScale / latentPerMole;
            require(std::isfinite(request), "Bouss bulk evaporation request overflows.");
            // The bulk constitutive request cannot spend more than its own
            // initial sensible superheat, even when wall heat is also present.
            const double requested = admitted(std::min(request, sensible / coolingPerMole));
            proposals.push_back({false, {}, 0, requested});
            requestedMoles += requested;
        }
        if (wallHeat)
            for (const auto& packet : wallHeat->packets)
            {
                require(
                    packet.wall < vapor.walls.size() && std::isfinite(packet.inwardEnergy) && packet.inwardEnergy >= 0,
                    "Malformed certified wall heat event.");
                const auto& binding = vapor.walls[packet.wall];
                const long double allocated = packet.inwardEnergy * binding.evaporationFraction;
                const auto requested = admitted(allocated / latentPerMole);
                proposals.push_back(
                    {true, binding.patch, packet.geometryFace, requested, 0, packet.inwardEnergy, allocated});
                requestedMoles += requested;
                allocatedWallHeat += allocated;
            }
        long double scale = 1, admittedTotal = requestedMoles.value();
        if (requestedMoles.value() > 0)
        {
            scale = std::min(scale, liquidDonor.value() / requestedMoles.value());
            scale =
                std::min(scale, static_cast<long double>(state.amount(catalog, vapor.solvent, SpeciesPhase::Liquid)) /
                                    requestedMoles.value());
            scale =
                std::min(scale, capacityMargin.value() / (requestedMoles.value() * volumes.gasMolarVolumes[species]));
            const long double superheat = delta > 0 ? sensible : 0;
            // Wall latent allowance is delivered by that SAME boundary owner;
            // this packet is never added again to the thermal source.
            const auto available =
                std::max(0.0L, superheat + std::min(0.0L, thermal.value()) + allocatedWallHeat.value());
            scale = std::min(scale, available / (requestedMoles.value() * coolingPerMole));
            admittedTotal = std::min({requestedMoles.value(), liquidDonor.value(),
                static_cast<long double>(state.amount(catalog, vapor.solvent, SpeciesPhase::Liquid)),
                capacityMargin.value() / volumes.gasMolarVolumes[species], available / coolingPerMole});
        }
        require(std::isfinite(scale) && scale >= 0 && scale <= 1, "Invalid joint wall/bulk admission scale.");
        const auto fits_evaporation = [&](double amount)
        {
            auto nextCapacity = capacityMargin;
            nextCapacity += -static_cast<long double>(amount) * volumes.gasMolarVolumes[species];
            auto nextVapor = vaporStock;
            nextVapor += amount;
            auto represented = state;
            represented.set_amount(
                catalog, vapor.solvent, vapor.vaporPhase, narrow(nextVapor.value()), vapor.population);
            return nextCapacity.value() >= 0 && volume_margin(catalog, volumes, represented, capacity).value() >= 0;
        };
        for (auto& proposal : proposals)
        {
            proposal.admittedMoles = proposals.size() == 1
                                         ? admitted(admittedTotal)
                                         : admitted(static_cast<long double>(proposal.requestedMoles) * scale);
            if (!fits_evaporation(proposal.admittedMoles))
            {
                // As for gas requests, admit only an extent whose raw event and
                // rounded stored vapor both fit the unchanged physical capacity.
                double low = 0, high = proposal.admittedMoles;
                require(fits_evaporation(low), "Pre-evaporation gas inventory exceeds shared scalar capacity.");
                for (unsigned iteration = 0; iteration < 80; ++iteration)
                {
                    const auto middle = std::midpoint(low, high);
                    if (middle == low || middle == high)
                        break;
                    if (fits_evaporation(middle))
                        low = middle;
                    else
                        high = middle;
                }
                proposal.admittedMoles = low;
            }
            transfer(true, proposal.admittedMoles);
            evaporationMoles += proposal.admittedMoles;
        }
        result.vapor.requestedEvaporation = narrow(requestedMoles.value());
        result.vapor.evaporation = narrow(evaporationMoles.value());
        result.vapor.evaporationAdmissions = std::move(proposals);
        result.vapor.latentEnergy = latentEnergy.value();
        result.vapor.sensibleEnergy = sensibleEnergy.value();
        state.set_amount(catalog, vapor.solvent, SpeciesPhase::Liquid, narrow(liquidStock.value()));
        state.set_amount(catalog, vapor.solvent, vapor.vaporPhase, narrow(vaporStock.value()), vapor.population);
    }
    state.validate(catalog);
    (void) free_volume(catalog, volumes, state, capacity);
    const ReactionContext transported{catalog, initial, environment};
    if (sources.populationEvents)
        result.numberEvents = sources.populationEvents({transported, state, numbers, result.admissions, result.vapor});
    result.numbers = advance_numbers(catalog, initial, state, numbers, result.numberEvents);
    result.state = std::move(state);
    result.thermalEnergy = thermal.value();
    require(std::isfinite(result.thermalEnergy), "Bouss accepted thermal budget is nonfinite.");
    return result;
}
} // namespace SimpleFluid::Reacting
