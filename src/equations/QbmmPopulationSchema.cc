/** @file QbmmPopulationSchema.cc
 * @brief Family binding, fixed-composition gates, and explicit moment migrations.
 */
#include "equations/QbmmPopulationSchema.hh"

#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Reacting
{
namespace
{
constexpr long double relative_tolerance = 64 * std::numeric_limits<double>::epsilon();

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

bool agree(long double actual, long double expected)
{
    if (actual == 0 || expected == 0)
        return actual == expected;
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= relative_tolerance * std::max(std::abs(actual), std::abs(expected));
}

double checked_double(long double value, bool nonzero)
{
    require(std::isfinite(value) && std::abs(value) <= std::numeric_limits<double>::max(),
        "QBMM value exceeds the representable range.");
    const auto result = static_cast<double>(value);
    require(result != 0 || !nonzero, "A nonzero QBMM value underflows to zero.");
    require(agree(result, value), "The QBMM value cannot be represented within the roundoff budget.");
    return result;
}

std::uint64_t family_fingerprint(const QbmmPopulationSchema& schema)
{
    std::uint64_t hash = 14695981039346656037ULL;
    const auto byte = [&hash](unsigned char value)
    {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    const auto number = [&byte](std::uint64_t value)
    {
        for (unsigned shift = 0; shift < 64; shift += 8)
            byte(static_cast<unsigned char>(value >> shift));
    };
    const auto string = [&byte, &number](std::string_view value)
    {
        number(value.size());
        for (const char c : value)
            byte(static_cast<unsigned char>(c));
    };

    string("QBMM dispersed gas, fixed mass fractions, moments 0 through 3, two nodes");
    number(schema.catalog().fingerprint());
    string(schema.population());
    number(std::bit_cast<std::uint64_t>(schema.mass_scale_kg()));
    for (const double fraction : schema.mass_fractions())
        number(std::bit_cast<std::uint64_t>(fraction));
    const auto& versions = schema.closure_versions();
    number(versions.inversion);
    number(versions.thermodynamics);
    number(versions.transport);
    string(versions.source_identity);
    number(versions.source);
    return hash;
}

// Apply the scale after multiplying mantissas so a large scale ratio can act on a small moment.
double convert_coordinate(double value, long double numerator, long double denominator, unsigned order)
{
    if (value == 0)
        return 0;
    int numerator_exponent = 0, denominator_exponent = 0, value_exponent = 0;
    const long double numerator_mantissa = std::frexp(numerator, &numerator_exponent);
    const long double denominator_mantissa = std::frexp(denominator, &denominator_exponent);
    const long double ratio_mantissa = numerator_mantissa / denominator_mantissa;
    auto mantissa = std::frexp(static_cast<long double>(value), &value_exponent);
    for (unsigned k = 0; k < order; ++k)
        mantissa *= ratio_mantissa;
    const int exponent = value_exponent + static_cast<int>(order) * (numerator_exponent - denominator_exponent);
    const double converted = checked_double(std::scalbn(mantissa, exponent), true);
    require(agree(std::scalbn(static_cast<long double>(converted), -exponent), mantissa),
        "The QBMM coordinate conversion exceeds the roundoff budget.");
    return converted;
}

QbmmMomentBlock convert_moments(QbmmMomentBlock moments, long double numerator, long double denominator)
{
    for (unsigned order = 1; order < moments.size(); ++order)
        moments[order] = convert_coordinate(moments[order], numerator, denominator, order);
    return moments;
}

void require_realizable(QbmmMomentBlock moments)
{
    require(invert_qbmm_moments(moments).valid(), "The QBMM migration does not give a realizable moment block.");
}

QbmmShapeState checked_shape(
    const QbmmPopulationSchema& schema, double number, std::span<const double> moles, QbmmMomentBlock distribution)
{
    const double mass = schema.scaled_mass(moles);
    require(std::isfinite(number) && number >= 0 && agree(distribution[0], number) && agree(distribution[1], mass),
        "QBMM migration must preserve the authoritative count and species mass.");
    require_realizable(distribution);
    if (number != distribution[0] || mass != distribution[1])
        require_realizable({number, mass, distribution[2], distribution[3]});
    return {distribution[2], distribution[3]};
}
} // namespace

QbmmPopulationSchema::QbmmPopulationSchema(std::shared_ptr<const SpeciesCatalog> catalog,
    std::uint64_t catalog_fingerprint, std::string population, std::string_view phase_identity, bool dispersed_phase,
    double mass_scale_kg, std::vector<QbmmMassFraction> fractions, QbmmClosureVersions versions)
    : d_catalog(std::move(catalog)), d_population(std::move(population)), d_mass_scale_kg(mass_scale_kg),
      d_versions(std::move(versions)), d_mixture_molar_mass(0), d_fingerprint(0)
{
    require(d_catalog && catalog_fingerprint == d_catalog->fingerprint(),
        "The QBMM composition catalog fingerprint must match the host catalog.");
    require(dispersed_phase && !d_population.empty() && d_population == phase_identity,
        "A QBMM family requires a matching named dispersed gas phase.");
    require(std::isfinite(d_mass_scale_kg) && d_mass_scale_kg > 0, "The QBMM mass scale must be finite and positive.");
    require(d_versions.inversion == 1 && d_versions.thermodynamics == 1 && d_versions.transport == 1 &&
                !d_versions.source_identity.empty() && d_versions.source != 0,
        "Unknown QBMM built-in policy version or missing source policy identity and revision.");
    const auto& species = d_catalog->species();
    require(fractions.size() == species.size(), "The QBMM composition must declare every catalog species.");

    SimpleFluid::detail::CompensatedSum total_fraction, moles_per_kg;
    d_mass_fractions.reserve(fractions.size());
    for (std::size_t i = 0; i < fractions.size(); ++i)
    {
        const auto [id, fraction] = fractions[i];
        require(id == species[i].id, "The QBMM fractions must be in the supplied catalog's species order.");
        require(std::isfinite(fraction) && fraction >= 0 && fraction <= 1,
            "QBMM mass fractions must be finite and between zero and one.");
        require(fraction == 0 || d_catalog->allows(id, SpeciesPhase::Dispersed),
            "A positive QBMM mass fraction requires a species admitted to dispersed gas.");
        d_mass_fractions.push_back(fraction == 0 ? 0.0 : fraction);
        total_fraction += fraction;
        moles_per_kg += static_cast<long double>(fraction) / species[i].molarMass;
    }
    require(agree(total_fraction.value(), 1), "QBMM mass fractions must sum to one.");
    d_mixture_molar_mass = 1 / moles_per_kg.value();
    require(std::isfinite(d_mixture_molar_mass) && d_mixture_molar_mass > 0,
        "The QBMM mixture molar mass must be finite and positive.");
    d_fingerprint = family_fingerprint(*this);
}

double QbmmPopulationSchema::scaled_mass(std::span<const double> moles) const
{
    require(
        std::ranges::all_of(moles, [](double n) { return n >= 0; }), "QBMM stock species amounts must be nonnegative.");
    return scaled_mass_change(moles);
}

double QbmmPopulationSchema::scaled_mass_change(std::span<const double> signed_moles) const
{
    require(signed_moles.size() == d_mass_fractions.size(), "The QBMM species packet must have the catalog size.");
    const auto& species = d_catalog->species();
    SimpleFluid::detail::CompensatedSum total_mass;
    for (std::size_t i = 0; i < signed_moles.size(); ++i)
    {
        const double moles = signed_moles[i];
        require(std::isfinite(moles) && (d_mass_fractions[i] != 0 || moles == 0),
            "A QBMM species packet must be finite and exactly zero for zero fractions.");
        const long double mass = static_cast<long double>(moles) * species[i].molarMass;
        require(std::isfinite(mass) && (mass != 0 || moles == 0), "Unrepresentable QBMM species mass.");
        total_mass += mass;
    }
    const auto mass = total_mass.value();
    require(std::isfinite(mass), "Unrepresentable QBMM total mass.");
    for (std::size_t i = 0; i < signed_moles.size(); ++i)
        require(agree(static_cast<long double>(signed_moles[i]) * species[i].molarMass, d_mass_fractions[i] * mass),
            "The QBMM species packet violates the fixed composition.");
    return checked_double(mass / d_mass_scale_kg, mass != 0);
}

std::vector<double> QbmmPopulationSchema::species_moles(double scaled_mass) const
{
    require(std::isfinite(scaled_mass), "QBMM packet generation requires finite scaled mass.");
    const auto& species = d_catalog->species();
    std::vector<double> moles(species.size());
    for (std::size_t i = 0; i < moles.size(); ++i)
    {
        const double fraction = d_mass_fractions[i];
        if (fraction != 0 && scaled_mass != 0)
            moles[i] = checked_double(
                static_cast<long double>(scaled_mass) * d_mass_scale_kg * fraction / species[i].molarMass, true);
    }
    require(agree(scaled_mass_change(moles), scaled_mass), "The QBMM packet cannot represent its requested mass.");
    return moles;
}

QbmmShapeState qbmm_initialize_monodisperse(
    const QbmmPopulationSchema& schema, double number, std::span<const double> moles)
{
    const double mass = schema.scaled_mass(moles);
    require(std::isfinite(number) && number >= 0 && (number == 0) == (mass == 0),
        "QBMM monodisperse initialization requires both material and count, or exact absence.");
    if (number == 0)
        return {};
    const long double mean = static_cast<long double>(mass) / number;
    const long double second = mass * mean;
    const QbmmShapeState shape{checked_double(second, true), checked_double(second * mean, true)};
    require_realizable({number, mass, shape.H2, shape.H3});
    return shape;
}

QbmmShapeState qbmm_migrate_mass_pivots(
    const QbmmPopulationSchema& schema, double number, std::span<const double> moles, std::span<const QbmmAtom> pivots)
{
    return checked_shape(schema, number, moles, qbmm_atom_moments(pivots));
}

QbmmShapeState qbmm_migrate_molar_pivots(const QbmmPopulationSchema& schema, double number,
    std::span<const double> moles, std::span<const QbmmMolarPivot> pivots)
{
    std::vector<QbmmAtom> atoms;
    atoms.reserve(pivots.size());
    for (const auto [count, mol_per_bubble] : pivots)
    {
        require(std::isfinite(count) && count >= 0 && std::isfinite(mol_per_bubble) && mol_per_bubble > 0,
            "A QBMM molar pivot requires positive finite moles per bubble and nonnegative finite count.");
        if (count != 0)
            atoms.push_back({count,
                convert_coordinate(mol_per_bubble, schema.mixture_molar_mass_kg_per_mol(), schema.mass_scale_kg(), 1)});
    }
    return qbmm_migrate_mass_pivots(schema, number, moles, atoms);
}

QbmmShapeState qbmm_convert_molar_moments(const QbmmPopulationSchema& schema, double number,
    std::span<const double> moles, QbmmMomentBlock unscaled_molar_moments)
{
    require_realizable(unscaled_molar_moments);
    const auto moments =
        convert_moments(unscaled_molar_moments, schema.mixture_molar_mass_kg_per_mol(), schema.mass_scale_kg());
    return checked_shape(schema, number, moles, moments);
}

QbmmMomentBlock qbmm_rescale_moments(
    const QbmmPopulationSchema& from, const QbmmPopulationSchema& to, QbmmMomentBlock moments)
{
    require(from.catalog().fingerprint() == to.catalog().fingerprint() && from.population() == to.population() &&
                std::ranges::equal(from.mass_fractions(), to.mass_fractions()) &&
                from.closure_versions() == to.closure_versions(),
        "QBMM mass rescaling requires the same catalog, family binding, composition, and policies.");
    require_realizable(moments);
    const auto converted = convert_moments(moments, from.mass_scale_kg(), to.mass_scale_kg());
    require_realizable(converted);
    return converted;
}
} // namespace SimpleFluid::Reacting
