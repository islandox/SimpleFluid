/** @file SpeciesDiffusion.cc */
#include "equations/SpeciesDiffusion.hh"

#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Reacting
{
namespace
{
using Sum = detail::CompensatedSum<long double>;
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
double narrow(long double value)
{
    const double result = static_cast<double>(value);
    require(std::isfinite(value) && std::isfinite(result) && (value == 0.0L || result != 0.0),
        "Diffusion arithmetic is nonfinite or cannot be represented without overflow/underflow.");
    return result;
}
void validate_compartment(const DiffusionCompartment& compartment)
{
    require(static_cast<std::size_t>(compartment.phase) < 4 &&
                (compartment.population.empty() || compartment.phase == SpeciesPhase::Dispersed),
        "Diffusion compartments require a valid phase and an explicitly dispersed population name.");
}
std::vector<long double> flatten(const CellSpeciesState& state)
{
    std::vector<long double> result(state.amounts.begin(), state.amounts.end());
    for (const auto& population : state.dispersedPopulations)
        result.insert(result.end(), population.amounts.begin(), population.amounts.end());
    return result;
}
void store(CellSpeciesState& state, std::span<const long double> values)
{
    std::size_t index = 0;
    for (auto& value : state.amounts)
        value = narrow(values[index++]);
    for (auto& population : state.dispersedPopulations)
        for (auto& value : population.amounts)
            value = narrow(values[index++]);
}
std::size_t slot(const SpeciesCatalog& catalog, const CellSpeciesState& state, std::size_t species,
    const DiffusionCompartment& compartment)
{
    const auto ns = catalog.species().size();
    if (compartment.population.empty())
        return static_cast<std::size_t>(compartment.phase) * ns + species;
    for (std::size_t p = 0; p < state.dispersedPopulations.size(); ++p)
        if (state.dispersedPopulations[p].id == compartment.population)
            return state.amounts.size() + p * ns + species;
    throw std::invalid_argument("Diffusion compartment selects an absent named population.");
}
} // namespace

SpeciesDiffusionPreview preview_species_diffusion(const SpeciesCatalog& catalog,
    std::span<const CellSpeciesState> owned, const SpeciesDiffusionBinding& binding,
    std::span<const CanonicalDiffusionFace> faces)
{
    validate_compartment(binding.transported);
    validate_compartment(binding.counter);
    require(!binding.identity.empty() && binding.revision != 0 && binding.schemaFingerprint == catalog.fingerprint(),
        "Diffusion packets require provider identity/revision and the exact ordered species schema.");
    require(binding.energyConvention == EnergyConvention::SensibleEnergy ||
                binding.energyConvention == EnergyConvention::IncludesFormationEnergy,
        "Diffusion packets need an explicit valid caloric energy convention.");
    require(binding.counterSpecies.has_value(), "Diffusion requires an explicitly selected finite counter-species.");
    const auto counterSpecies = *binding.counterSpecies;
    const auto ns = catalog.species().size(), counter = catalog.index(counterSpecies);
    require(catalog.allows(counterSpecies, binding.counter.phase),
        "Diffusion counter-species is forbidden in its declared finite donor phase.");
    const bool sameCompartment = binding.transported.phase == binding.counter.phase &&
                                 binding.transported.population == binding.counter.population;
    std::vector<std::vector<long double>> initial;
    std::vector<std::vector<Sum>> changes, donors, stocks;
    std::vector<Sum> energy(owned.size());
    std::vector<std::array<Sum, 3>> momentum(owned.size());
    for (const auto& state : owned)
    {
        state.validate(catalog);
        static_cast<void>(slot(catalog, state, counter, binding.counter));
        static_cast<void>(slot(catalog, state, 0, binding.transported));
        initial.push_back(flatten(state));
        changes.emplace_back(initial.back().size());
        donors.emplace_back(initial.back().size());
        stocks.emplace_back(initial.back().size());
        for (std::size_t i = 0; i < initial.back().size(); ++i)
            stocks.back()[i] += initial.back()[i];
    }
    SpeciesDiffusionPreview result;
    result.identity = binding.identity;
    result.revision = binding.revision;
    result.schemaFingerprint = catalog.fingerprint();
    result.state.assign(owned.begin(), owned.end());
    result.cells.resize(owned.size());
    std::vector<std::size_t> order(faces.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](std::size_t a, std::size_t b) { return faces[a].identity < faces[b].identity; });
    std::set<std::uint64_t> seen;
    for (const auto faceIndex : order)
    {
        const auto& face = faces[faceIndex];
        require(seen.insert(face.identity).second && (face.owner || face.neighbor) &&
                    (!face.owner || *face.owner < owned.size()) && (!face.neighbor || *face.neighbor < owned.size()),
            "Canonical diffusion faces require unique IDs and valid incident owned rows.");
        require(
            face.moles.size() == ns && face.ownerMolarEnthalpy.size() == ns && face.neighborMolarEnthalpy.size() == ns,
            "Diffusion face species/enthalpy arrays must match the exact catalog ordering.");
        require(std::isfinite(face.ownerCounterMolarEnthalpy) && std::isfinite(face.neighborCounterMolarEnthalpy),
            "Counter-species molar enthalpies must be finite.");
        for (std::size_t a = 0; a < 3; ++a)
            require(std::isfinite(face.ownerVelocity[a]) && std::isfinite(face.neighborVelocity[a]),
                "Diffusive donor velocities must be finite.");
        Sum primaryMass, massScale;
        for (std::size_t s = 0; s < ns; ++s)
        {
            require(std::isfinite(face.moles[s]) && std::isfinite(face.ownerMolarEnthalpy[s]) &&
                        std::isfinite(face.neighborMolarEnthalpy[s]),
                "Diffusion fluxes and molar enthalpies must be finite.");
            require(face.moles[s] == 0.0 || catalog.allows(catalog.species()[s].id, binding.transported.phase),
                "A diffusing species is forbidden in its transported compartment.");
            const long double mass = static_cast<long double>(face.moles[s]) * catalog.species()[s].molarMass;
            narrow(mass);
            primaryMass += mass;
            massScale += std::abs(mass);
        }
        require(!sameCompartment || face.moles[counter] == 0.0,
            "A counter-species primary flux cannot also be prescribed in the identical compartment.");
        const long double mass = primaryMass.value();
        const double counterMoles = mass == 0.0L ? 0.0 : narrow(-mass / catalog.species()[counter].molarMass);
        const long double counterMass = static_cast<long double>(counterMoles) * catalog.species()[counter].molarMass;
        narrow(counterMass);
        Sum residual;
        residual += mass;
        residual += counterMass;
        massScale += std::abs(counterMass);
        require(std::abs(residual.value()) <= 64.0L * std::numeric_limits<double>::epsilon() * massScale.value(),
            "Stored diffusion counterflux violates zero net mixture mass beyond arithmetic roundoff.");
        Sum faceEnergy;
        std::array<Sum, 3> faceMomentum;
        const auto transfer = [&](std::size_t species, const DiffusionCompartment& compartment, double moles,
                                  double ownerH, double neighborH)
        {
            if (moles == 0.0)
                return;
            const auto& velocity = moles > 0.0 ? face.ownerVelocity : face.neighborVelocity;
            const double donorH = moles > 0.0 ? ownerH : neighborH;
            faceEnergy += static_cast<long double>(moles) * donorH;
            for (std::size_t a = 0; a < 3; ++a)
                faceMomentum[a] += static_cast<long double>(moles) * catalog.species()[species].molarMass * velocity[a];
            const auto apply = [&](std::optional<std::size_t> row, double sign)
            {
                if (!row)
                    return;
                const auto index = slot(catalog, owned[*row], species, compartment);
                const long double delta = static_cast<long double>(sign) * moles;
                changes[*row][index] += delta;
                // Keep the accepted stock and each raw energy/impulse operand in
                // one accumulator. Rounded net/face reports cannot seed state.
                stocks[*row][index] += delta;
                energy[*row] += delta * donorH;
                for (std::size_t a = 0; a < 3; ++a)
                    momentum[*row][a] += delta * catalog.species()[species].molarMass * velocity[a];
                if (delta < 0.0L)
                    donors[*row][index] += -delta;
            };
            apply(face.owner, -1.0);
            apply(face.neighbor, 1.0);
        };
        for (std::size_t s = 0; s < ns; ++s)
            transfer(s, binding.transported, face.moles[s], face.ownerMolarEnthalpy[s], face.neighborMolarEnthalpy[s]);
        transfer(
            counter, binding.counter, counterMoles, face.ownerCounterMolarEnthalpy, face.neighborCounterMolarEnthalpy);
        SpeciesDiffusionFaceLedger ledger;
        ledger.identity = face.identity;
        ledger.owner = face.owner;
        ledger.neighbor = face.neighbor;
        ledger.moles = face.moles;
        ledger.counterMoles = counterMoles;
        ledger.enthalpy = narrow(faceEnergy.value());
        ledger.massResidual = narrow(residual.value());
        ledger.absoluteMassFlux = narrow(massScale.value());
        for (std::size_t a = 0; a < 3; ++a)
            ledger.momentum[a] = narrow(faceMomentum[a].value());
        result.faces.push_back(std::move(ledger));
    }
    for (std::size_t c = 0; c < owned.size(); ++c)
    {
        std::vector<long double> accepted(initial[c].size()), deltas(initial[c].size());
        for (std::size_t i = 0; i < initial[c].size(); ++i)
        {
            if (donors[c][i].value() > initial[c][i])
                throw std::runtime_error("Canonical diffusion jointly exhausts a finite species/counter donor; "
                                         "recompute a smaller physical step.");
            deltas[i] = changes[c][i].value();
            accepted[i] = stocks[c][i].value();
            if (accepted[i] < 0.0L)
                throw std::runtime_error("Diffusion produced a negative finite inventory.");
        }
        store(result.state[c], accepted);
        result.state[c].validate(catalog);
        result.cells[c].change = owned[c];
        store(result.cells[c].change, deltas);
        result.cells[c].enthalpy = narrow(energy[c].value());
        for (std::size_t a = 0; a < 3; ++a)
            result.cells[c].momentum[a] = narrow(momentum[c][a].value());
    }
    return result;
}
} // namespace SimpleFluid::Reacting
