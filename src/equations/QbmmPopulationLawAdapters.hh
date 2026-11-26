/** @file QbmmPopulationLawAdapters.hh
 * @brief Pure SI rate-to-event adapters. No empirical coefficients or donor clipping.
 */
#pragma once

#include "equations/QbmmPopulationEvents.hh"
#include "utils/CompensatedSum.hh"

#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <set>

namespace SimpleFluid::Reacting
{
namespace qbmm_law_detail
{
/** A valid resize rate requests a destination that a shorter step can represent. */
class SIMPLEFLUID_EQUATIONS_EXPORT StepSizeError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

inline double narrow(long double value)
{
    const double result = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0) ||
        std::abs(value - result) > qbmm_reconstruction_tolerance * std::abs(value))
        throw std::invalid_argument("QBMM supplied law result is not representable.");
    return result;
}

inline double duration(const QbmmLawInputs& input)
{
    if (!input.quadrature.valid() || input.quadrature.active_atoms > input.quadrature.atoms.size() ||
        !input.time_step_s || !std::isfinite(*input.time_step_s) || *input.time_step_s <= 0 ||
        !std::isfinite(input.bulk_volume_m3) || input.bulk_volume_m3 <= 0 || !std::isfinite(input.mass_scale_kg) ||
        input.mass_scale_kg <= 0)
        throw std::invalid_argument("QBMM event adapter requires valid frozen quadrature, volume, scale and dt.");
    for (unsigned a = 0; a < input.quadrature.active_atoms; ++a)
    {
        const auto& atom = input.quadrature.atoms[a];
        if (!std::isfinite(atom.count) || atom.count <= 0 || !std::isfinite(atom.mass_over_scale) ||
            atom.mass_over_scale <= 0)
            throw std::invalid_argument("QBMM event adapter requires positive finite active atoms.");
    }
    return *input.time_step_s;
}

inline void validate_family_binding(const QbmmPopulationSchema& family, const QbmmLawInputs& input)
{
    if (input.family_fingerprint == 0 || input.family_fingerprint != family.fingerprint() ||
        input.catalog_fingerprint == 0 || input.catalog_fingerprint != family.catalog().fingerprint() ||
        input.population != family.population() || input.mass_scale_kg != family.mass_scale_kg())
        throw std::invalid_argument(
            "QBMM material rate adapter requires matching frozen family, catalog, population and scale.");
}

inline QbmmResize resize_from_mass_change(const QbmmPopulationSchema& family, const QbmmLawInputs& input, unsigned a,
    long double requested_per_bubble, const QbmmTransferCounterpart& counterpart)
{
    const auto& atom = input.quadrature.atoms[a];
    const auto proposed = atom.mass_over_scale + requested_per_bubble / input.mass_scale_kg;
    if (std::isinf(static_cast<double>(proposed)))
        throw StepSizeError("QBMM supplied law result is not representable.");
    const double destination = narrow(proposed);
    if (destination <= 0)
        throw StepSizeError("QBMM growth needs a representable positive destination or explicit disappearance.");
    if (destination == atom.mass_over_scale)
        throw std::invalid_argument(
            "QBMM growth needs a representable positive destination or explicit disappearance.");
    const long double scaled_mass = (static_cast<long double>(destination) - atom.mass_over_scale) * atom.count;
    const long double represented_mass = scaled_mass * input.mass_scale_kg;
    const long double requested_mass = requested_per_bubble * atom.count;
    if (!std::isfinite(represented_mass) || !std::isfinite(requested_mass) || requested_mass == 0 ||
        std::abs(represented_mass - requested_mass) > qbmm_reconstruction_tolerance * std::abs(requested_mass))
        throw std::invalid_argument("QBMM resize destination cannot represent the requested integrated mass.");
    return {a, atom.count, destination, {counterpart, family.species_moles(narrow(std::abs(scaled_mass)))}};
}
} // namespace qbmm_law_detail

/** K_ab in m3/s on bulk number densities; unordered equal-node pairs use K_aa*N_a^2/2.
 * Counts are expected counts, so there is no integer N*(N-1) correction. The supplied
 * symmetric kernel includes any collision efficiency. The owner reserves all donors
 * across these events and competing laws; an overdraw is rejected, never clipped here.
 */
[[nodiscard]] inline std::vector<QbmmEventOperation> qbmm_coalescence_from_kernel(
    const QbmmLawInputs& input, const std::array<std::array<double, 2>, 2>& kernel_m3_per_s)
{
    const double dt = qbmm_law_detail::duration(input);
    std::vector<QbmmEventOperation> result;
    for (unsigned a = 0; a < input.quadrature.active_atoms; ++a)
        for (unsigned b = a; b < input.quadrature.active_atoms; ++b)
        {
            const double kernel = kernel_m3_per_s[a][b];
            if (!std::isfinite(kernel) || kernel < 0 || kernel != kernel_m3_per_s[b][a])
                throw std::invalid_argument("QBMM coalescence kernel must be nonnegative, finite and symmetric.");
            const auto pairs = static_cast<long double>(dt) * kernel * input.quadrature.atoms[a].count *
                               input.quadrature.atoms[b].count / input.bulk_volume_m3 / (a == b ? 2 : 1);
            if (pairs != 0)
                result.emplace_back(QbmmCoalescence{a, b, qbmm_law_detail::narrow(pairs)});
        }
    return result;
}

struct QbmmDaughterFraction
{
    double multiplicity, mass_fraction; ///< Per parent; sum(multiplicity*mass_fraction) must equal one.
};

/** Frequencies in s^-1, daughter masses expressed as fractions of their parent's mass.
 * Every daughter has positive multiplicity and mass. The event kernel verifies mass conservation.
 */
[[nodiscard]] inline std::vector<QbmmEventOperation> qbmm_breakup_from_frequency(const QbmmLawInputs& input,
    std::span<const double> frequency_per_s, std::span<const std::vector<QbmmDaughterFraction>> daughters)
{
    const double dt = qbmm_law_detail::duration(input);
    const auto active = input.quadrature.active_atoms;
    if (frequency_per_s.size() != active || daughters.size() != active)
        throw std::invalid_argument("QBMM breakup law must supply exactly the active node layout.");
    std::vector<QbmmEventOperation> result;
    for (unsigned a = 0; a < active; ++a)
    {
        const double frequency = frequency_per_s[a];
        if (!std::isfinite(frequency) || frequency < 0)
            throw std::invalid_argument("QBMM breakup frequency must be finite and nonnegative.");
        if (frequency == 0)
            continue;
        QbmmBreakup event{
            a, qbmm_law_detail::narrow(static_cast<long double>(dt) * frequency * input.quadrature.atoms[a].count), {}};
        long double mass_fraction = 0;
        for (const auto& daughter : daughters[a])
        {
            if (!std::isfinite(daughter.multiplicity) || daughter.multiplicity <= 0 ||
                !std::isfinite(daughter.mass_fraction) || daughter.mass_fraction <= 0)
                throw std::invalid_argument("QBMM breakup daughters require positive finite multiplicities and mass.");
            mass_fraction += static_cast<long double>(daughter.multiplicity) * daughter.mass_fraction;
            event.daughters.push_back({daughter.multiplicity,
                qbmm_law_detail::narrow(
                    static_cast<long double>(input.quadrature.atoms[a].mass_over_scale) * daughter.mass_fraction)});
        }
        if (std::abs(mass_fraction - 1) > qbmm_reconstruction_tolerance)
            throw std::invalid_argument("QBMM supplied daughter fractions must conserve parent mass.");
        result.emplace_back(std::move(event));
    }
    return result;
}

/** Signed kg/(bubble s) at each active node. A complete disappearance must be supplied
 * explicitly through QbmmDisappearance. The destination must be positive and must
 * reproduce the requested signed integrated mass within the transfer roundoff budget.
 * Counterpart energy and momentum, if supplied, are integrated amounts per node.
 */
[[nodiscard]] inline std::vector<QbmmEventOperation> qbmm_resize_from_mass_rate(const QbmmPopulationSchema& family,
    const QbmmLawInputs& input, std::span<const double> mass_rate_kg_per_bubble_s,
    std::span<const QbmmTransferCounterpart> counterparts)
{
    qbmm_law_detail::validate_family_binding(family, input);
    const double dt = qbmm_law_detail::duration(input);
    if (mass_rate_kg_per_bubble_s.size() != input.quadrature.active_atoms ||
        counterparts.size() != input.quadrature.active_atoms)
        throw std::invalid_argument("QBMM growth law must supply exactly the active node layout.");
    std::vector<QbmmEventOperation> result;
    for (unsigned a = 0; a < input.quadrature.active_atoms; ++a)
    {
        const double rate = mass_rate_kg_per_bubble_s[a];
        if (!std::isfinite(rate))
            throw std::invalid_argument("QBMM mass rate must be finite.");
        if (rate == 0)
            continue;
        const long double requested_per_bubble = static_cast<long double>(rate) * dt;
        result.emplace_back(
            qbmm_law_detail::resize_from_mass_change(family, input, a, requested_per_bubble, counterparts[a]));
    }
    return result;
}

/** Supplied film values for each active frozen quadrature node, in node order. */
struct QbmmFilmNodeValues
{
    double coefficient_m_per_s, equilibrium_concentration_mol_per_m3;
};

/** One named family contributing node films to the shared original Dissolved stock. */
struct QbmmImplicitFilmFamily
{
    const QbmmPopulationSchema& family;
    const QbmmLawInputs& input;
    SpeciesId species;
    double dissolved_moles; ///< Original extensive inventory in mol/cell.
    std::span<const QbmmFilmNodeValues> nodes;
};

inline void validate_qbmm_implicit_film_species(const QbmmPopulationSchema& family, SpeciesId species)
{
    const auto& catalog = family.catalog();
    const auto index = catalog.index(species);
    const auto fractions = family.mass_fractions();
    if (!catalog.allows(species, SpeciesPhase::Dissolved) || !catalog.allows(species, SpeciesPhase::Dispersed))
        throw std::invalid_argument("QBMM film species must allow both Dissolved and Dispersed phases.");
    for (std::size_t s = 0; s < fractions.size(); ++s)
        if ((fractions[s] > 0) != (s == index))
            throw std::invalid_argument("QBMM film requires a family composed of exactly the transferred species.");
}

/** Backward-Euler film exchange with one end-of-stage Dissolved inventory per species.
 * Geometry, liquid EOS volume, coefficients and equilibrium concentrations are frozen. The output
 * follows caller family order and active node order. An absent family supplies no node values.
 * The resize events preserve count and pair material with Dissolved. The event owner reserves gross
 * original donors and owns thermal/momentum exchange. Complete depletion requires explicit disappearance.
 */
[[nodiscard]] inline std::vector<std::vector<QbmmEventOperation>> qbmm_implicit_film_transfer(
    std::span<const QbmmImplicitFilmFamily> families)
{
    std::vector<std::vector<QbmmEventOperation>> result(families.size());
    if (families.empty())
        return result;

    // Canonical arithmetic order also resolves ties when selecting the stiffest film.
    std::vector<std::size_t> order(families.size());
    std::iota(order.begin(), order.end(), std::size_t{});
    std::ranges::sort(order, {}, [&](std::size_t i) -> const std::string& { return families[i].family.population(); });

    struct Request
    {
        std::size_t family_index;
        unsigned atom;
        long double alpha, equilibrium_concentration;
    };
    struct Pool
    {
        double dissolved;
        std::optional<double> concentration;
        std::vector<Request> requests;
    };
    std::map<SpeciesId, Pool> pools;
    std::set<std::uint64_t> fingerprints;
    const auto& shared = families[order.front()];
    std::optional<double> shared_volume;
    for (std::size_t rank = 0; rank < order.size(); ++rank)
    {
        const std::size_t i = order[rank];
        const auto& [family, input, species, dissolved, nodes] = families[i];
        qbmm_law_detail::validate_family_binding(family, input);
        validate_qbmm_implicit_film_species(family, species);
        const double dt = qbmm_law_detail::duration(input);
        if ((rank > 0 && family.population() == families[order[rank - 1]].family.population()) ||
            !fingerprints.insert(family.fingerprint()).second ||
            input.catalog_fingerprint != shared.input.catalog_fingerprint ||
            input.time_step_s != shared.input.time_step_s || input.bulk_volume_m3 != shared.input.bulk_volume_m3 ||
            input.location != shared.input.location || !std::isfinite(dissolved) || dissolved < 0)
            throw std::invalid_argument("Joint QBMM films need distinct families from the same frozen source stage.");
        const auto species_index = family.catalog().index(species);
        if (const auto& stock = input.carrier_liquid_dissolved_moles;
            stock &&
            (!input.carrier_liquid_volume_m3 || !std::isfinite(*input.carrier_liquid_volume_m3) ||
                *input.carrier_liquid_volume_m3 <= 0 || stock->size() != family.catalog().species().size() ||
                !std::ranges::all_of(*stock, [](double value) { return std::isfinite(value) && value >= 0; }) ||
                (*stock)[species_index] != dissolved))
            throw std::invalid_argument("QBMM film supplied Dissolved inventory disagrees with its frozen stock.");
        auto& pool = pools.try_emplace(species, Pool{dissolved, {}, {}}).first->second;
        if (dissolved != pool.dissolved)
            throw std::invalid_argument("Joint QBMM films for one species need the same frozen Dissolved inventory.");

        const auto active = input.quadrature.active_atoms;
        const auto status = input.quadrature.status;
        if (nodes.size() != active || (status == InversionStatus::Absent && active != 0) ||
            (status == InversionStatus::RankOne && active != 1) || (status == InversionStatus::TwoNodes && active != 2))
            throw std::invalid_argument("QBMM film values must match the frozen active node layout.");
        if (active == 0)
            continue;
        if (!input.geometry || input.geometry->active_atoms != active || !input.carrier_liquid_volume_m3 ||
            !input.carrier_liquid_concentration_mol_per_m3 ||
            input.carrier_liquid_concentration_mol_per_m3->size() != family.catalog().species().size() ||
            !std::ranges::all_of(*input.carrier_liquid_concentration_mol_per_m3,
                [](double value) { return std::isfinite(value) && value >= 0; }))
            throw std::invalid_argument("Occupied QBMM film needs frozen node geometry and a physical liquid trace.");

        const double volume = *input.carrier_liquid_volume_m3;
        if (!std::isfinite(volume) || volume <= 0 || (shared_volume && volume != *shared_volume))
            throw std::invalid_argument("Occupied QBMM films require one finite positive carrier liquid EOS volume.");
        const double concentration = (*input.carrier_liquid_concentration_mol_per_m3)[species_index];
        const double expected_concentration = dissolved / volume;
        if (!std::isfinite(expected_concentration) || (dissolved != 0 && expected_concentration == 0) ||
            std::abs(static_cast<long double>(concentration) - expected_concentration) >
                qbmm_reconstruction_tolerance * static_cast<long double>(expected_concentration) ||
            (pool.concentration && concentration != *pool.concentration))
            throw std::invalid_argument("QBMM film liquid trace disagrees with the frozen Dissolved inventory.");
        shared_volume = volume;
        pool.concentration = concentration;

        for (unsigned a = 0; a < active; ++a)
        {
            const auto [coefficient, equilibrium] = nodes[a];
            const long double diameter = input.geometry->diameter_m[a];
            if (!std::isfinite(diameter) || diameter <= 0 || !std::isfinite(coefficient) || coefficient < 0 ||
                !std::isfinite(equilibrium) || equilibrium < 0)
                throw std::invalid_argument(
                    "QBMM film requires positive finite diameters and nonnegative finite supplied values.");
            if (coefficient == 0)
                continue;
            const long double area =
                std::numbers::pi_v<long double> * diameter * diameter * input.quadrature.atoms[a].count;
            const long double alpha = dt * static_cast<long double>(coefficient) * area / volume;
            const long double equilibrium_moles = static_cast<long double>(volume) * equilibrium;
            if (!std::isfinite(alpha) || alpha <= 0 || !std::isfinite(equilibrium_moles) ||
                (equilibrium != 0 && equilibrium_moles == 0))
                throw std::invalid_argument("QBMM film weight or equilibrium inventory is not representable.");
            pool.requests.push_back({i, a, alpha, equilibrium});
        }
    }
    for (const auto& [species, pool] : pools)
    {
        if (pool.requests.empty())
            continue;
        // Solve for the offset from the stiffest film's equilibrium to retain its small driving difference.
        const long double reference =
            std::ranges::max_element(pool.requests, {}, &Request::alpha)->equilibrium_concentration;
        const long double volume = *shared_volume;
        const auto equilibrium_difference = [volume](long double concentration, long double center)
        {
            // Subtract concentrations before conversion to preserve close equilibria at the common volume.
            const long double difference = volume * (concentration - center);
            if (!std::isfinite(difference) || (concentration != center && difference == 0))
                throw std::invalid_argument("QBMM film equilibrium difference is not representable.");
            return difference;
        };
        SimpleFluid::detail::CompensatedSum<long double> denominator;
        denominator += 1;
        for (const auto& request : pool.requests)
            denominator += request.alpha;
        const long double denominator_value = denominator.value();
        if (!std::isfinite(denominator_value))
            throw std::invalid_argument("QBMM film dissolved endpoint is not representable.");
        const auto offset_from_equilibrium = [&](long double equilibrium)
        {
            SimpleFluid::detail::CompensatedSum<long double> numerator;
            const long double equilibrium_moles = volume * equilibrium;
            numerator += pool.dissolved;
            numerator += -equilibrium_moles;
            numerator += -std::fma(volume, equilibrium, -equilibrium_moles);
            for (const auto& request : pool.requests)
            {
                const long double difference = equilibrium_difference(request.equilibrium_concentration, equilibrium);
                const long double weighted_difference = request.alpha * difference;
                if (!std::isfinite(weighted_difference) || (difference != 0 && weighted_difference == 0))
                    throw std::invalid_argument("QBMM film weighted equilibrium difference is not representable.");
                numerator += weighted_difference;
            }
            const long double numerator_value = numerator.value();
            const long double offset = numerator_value / denominator_value;
            if (!std::isfinite(offset) || (numerator_value != 0 && offset == 0))
                throw std::invalid_argument("QBMM film dissolved endpoint is not representable.");
            return offset;
        };
        const long double endpoint_offset = offset_from_equilibrium(reference);
        for (const auto& request : pool.requests)
        {
            const long double equilibrium_offset = equilibrium_difference(request.equilibrium_concentration, reference);
            long double driving_difference = endpoint_offset - equilibrium_offset;
            // Recenter the compensated numerator when the offset subtraction
            // cancels, since alpha can amplify even one lost bit into a finite transfer.
            if (equilibrium_offset != 0 &&
                std::abs(driving_difference) <= .5L * std::max(std::abs(endpoint_offset), std::abs(equilibrium_offset)))
                driving_difference = offset_from_equilibrium(request.equilibrium_concentration);
            if (driving_difference == 0)
                continue;
            const long double transfer = request.alpha * driving_difference;
            if (transfer == 0)
                throw std::invalid_argument("QBMM film transfer underflows its integrated molar amount.");
            (void) qbmm_law_detail::narrow(transfer);
            const auto& film = families[request.family_index];
            const auto& catalog = film.family.catalog();
            const long double atom_mass_change = transfer * catalog.species()[catalog.index(species)].molarMass /
                                                 film.input.quadrature.atoms[request.atom].count;
            result[request.family_index].emplace_back(qbmm_law_detail::resize_from_mass_change(
                film.family, film.input, request.atom, atom_mass_change, QbmmPhaseExchange{SpeciesPhase::Dissolved}));
        }
    }
    return result;
}

[[nodiscard]] inline std::vector<QbmmEventOperation> qbmm_implicit_film_transfer(const QbmmPopulationSchema& family,
    const QbmmLawInputs& input, SpeciesId species, double dissolved_moles, std::span<const QbmmFilmNodeValues> nodes)
{
    const std::array films{QbmmImplicitFilmFamily{family, input, species, dissolved_moles, nodes}};
    auto operations = qbmm_implicit_film_transfer(films);
    return std::move(operations.front());
}

/** Single-family adapter using the original Dissolved inventory supplied with an occupied liquid trace. */
[[nodiscard]] inline std::vector<QbmmEventOperation> qbmm_implicit_film_transfer(const QbmmPopulationSchema& family,
    const QbmmLawInputs& input, SpeciesId species, std::span<const QbmmFilmNodeValues> nodes)
{
    if (const auto& dissolved = input.carrier_liquid_dissolved_moles)
    {
        if (dissolved->size() != family.catalog().species().size())
            throw std::invalid_argument("Occupied QBMM film needs original catalogued Dissolved moles.");
        return qbmm_implicit_film_transfer(
            family, input, species, (*dissolved)[family.catalog().index(species)], nodes);
    }
    if (input.quadrature.status != InversionStatus::Absent)
        throw std::invalid_argument("Occupied QBMM film needs original catalogued Dissolved moles.");
    return qbmm_implicit_film_transfer(family, input, species, 0, nodes);
}

struct QbmmBirthRate
{
    double number_per_bulk_m3_s, mass_kg_per_bubble;
    QbmmTransferCounterpart counterpart;
};

/** Explicit nonnegative birth spectrum and material allocation; no yield or size is inferred. */
[[nodiscard]] inline std::vector<QbmmEventOperation> qbmm_birth_from_rate(
    const QbmmPopulationSchema& family, const QbmmLawInputs& input, std::span<const QbmmBirthRate> births)
{
    qbmm_law_detail::validate_family_binding(family, input);
    const double dt = qbmm_law_detail::duration(input);
    QbmmBirth event;
    for (const auto& birth : births)
    {
        if (!std::isfinite(birth.number_per_bulk_m3_s) || birth.number_per_bulk_m3_s < 0 ||
            !std::isfinite(birth.mass_kg_per_bubble) || birth.mass_kg_per_bubble <= 0)
            throw std::invalid_argument("QBMM birth needs finite nonnegative rate and positive SI bubble mass.");
        if (birth.number_per_bulk_m3_s == 0)
            continue;
        const double count =
            qbmm_law_detail::narrow(static_cast<long double>(birth.number_per_bulk_m3_s) * input.bulk_volume_m3 * dt);
        const double size =
            qbmm_law_detail::narrow(static_cast<long double>(birth.mass_kg_per_bubble) / input.mass_scale_kg);
        event.atoms.push_back({{count, size}, {birth.counterpart, family.species_moles(qbmm_law_detail::narrow(
                                                                      static_cast<long double>(count) * size))}});
    }
    if (event.atoms.empty())
        return {};
    return {std::move(event)};
}
} // namespace SimpleFluid::Reacting
