/** @file QbmmPopulationLawAdapters.hh
 * @brief Pure SI rate-to-event adapters. No empirical coefficients or donor clipping.
 */
#pragma once

#include "equations/QbmmPopulationEvents.hh"

#include <limits>

namespace SimpleFluid::Reacting
{
namespace qbmm_law_detail
{
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
        const auto& atom = input.quadrature.atoms[a];
        const long double requested_per_bubble = static_cast<long double>(rate) * dt;
        const double destination =
            qbmm_law_detail::narrow(atom.mass_over_scale + requested_per_bubble / input.mass_scale_kg);
        if (destination <= 0 || destination == atom.mass_over_scale)
            throw std::invalid_argument(
                "QBMM growth needs a representable positive destination or explicit disappearance.");
        const long double scaled_mass = (static_cast<long double>(destination) - atom.mass_over_scale) * atom.count;
        const long double represented_mass = scaled_mass * input.mass_scale_kg;
        const long double requested_mass = requested_per_bubble * atom.count;
        if (!std::isfinite(represented_mass) || !std::isfinite(requested_mass) || requested_mass == 0 ||
            std::abs(represented_mass - requested_mass) > qbmm_reconstruction_tolerance * std::abs(requested_mass))
            throw std::invalid_argument("QBMM resize destination cannot represent the requested integrated mass.");
        result.emplace_back(QbmmResize{a, atom.count, destination,
            {counterparts[a], family.species_moles(qbmm_law_detail::narrow(std::abs(scaled_mass)))}});
    }
    return result;
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
