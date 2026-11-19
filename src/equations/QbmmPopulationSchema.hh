/** @file QbmmPopulationSchema.hh
 * @brief Immutable fixed-composition QBMM family and explicit coordinate migration.
 */
#pragma once

#include "equations/QbmmConstitutiveLaws.hh"
#include "equations/QbmmMoments.hh"
#include "equations/ReactionMechanism.hh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::Reacting
{
/** One declared mass fraction, identified by its catalog species ID. */
struct QbmmMassFraction
{
    SpeciesId species;
    double mass_fraction;
};

/** Versioned policies for the four-moment, two-node mass-coordinate representation. */
struct QbmmClosureVersions
{
    std::uint64_t inversion = 1, thermodynamics = 1, transport = 1;
    std::string source_identity = "constructive"; ///< Caller-declared source policy identity.
    std::uint64_t source = 1;                     ///< Revision of that source policy.
    std::vector<QbmmLawMetadata> laws; ///< Ordered supplied SI contracts, including ranges and rejection policy.
    bool operator==(const QbmmClosureVersions&) const = default;
};

/** Mesh-independent family binding. Composition and conditioning scale are fixed at construction. */
class SIMPLEFLUID_EQUATIONS_EXPORT QbmmPopulationSchema
{
public:
    /** Bind to a named dispersed gas role from the host phase schema.
     * @tparam PhaseRoleDescriptor Host descriptor with `role` (an enum with `DispersedGas`) and `identity`.
     * @param catalog Ordered species catalog shared with the host.
     * @param catalog_fingerprint Schema stamp supplied with the declared composition.
     * @param population Exact named dispersed compartment.
     * @param phase Corresponding host physical phase role.
     * @param mass_scale_kg Positive finite reference mass in kg per bubble.
     * @param fractions One nonnegative mass fraction per catalog species, in catalog order, including zeros.
     * @param versions Supported built-in policies and a declared source policy identity and revision.
     * @throws std::invalid_argument For an incompatible binding or invalid fractions, scale or policies.
     */
    template<class PhaseRoleDescriptor>
    QbmmPopulationSchema(std::shared_ptr<const SpeciesCatalog> catalog, std::uint64_t catalog_fingerprint,
        std::string population, const PhaseRoleDescriptor& phase, double mass_scale_kg,
        std::vector<QbmmMassFraction> fractions, QbmmClosureVersions versions = {})
        : QbmmPopulationSchema(std::move(catalog), catalog_fingerprint, std::move(population), phase.identity,
              phase.role == decltype(phase.role)::DispersedGas, mass_scale_kg, std::move(fractions),
              std::move(versions))
    {
    }

    QbmmPopulationSchema(const QbmmPopulationSchema&) = default;
    QbmmPopulationSchema& operator=(const QbmmPopulationSchema&) = delete;

    [[nodiscard]] const SpeciesCatalog& catalog() const noexcept { return *d_catalog; }
    [[nodiscard]] const std::string& population() const noexcept { return d_population; }
    [[nodiscard]] std::uint64_t fingerprint() const noexcept { return d_fingerprint; }
    [[nodiscard]] double mass_scale_kg() const noexcept { return d_mass_scale_kg; }
    [[nodiscard]] std::span<const double> mass_fractions() const noexcept { return d_mass_fractions; }
    [[nodiscard]] const QbmmClosureVersions& closure_versions() const noexcept { return d_versions; }
    [[nodiscard]] bool declares_law(const QbmmLawMetadata& law) const
    {
        return std::ranges::find(d_versions.laws, law) != d_versions.laws.end();
    }
    [[nodiscard]] long double mixture_molar_mass_kg_per_mol() const noexcept { return d_mixture_molar_mass; }

    /** Derive H1 from the existing nonnegative mol/cell stock in catalog order.
     * Fractions and each species mass match to 64 double epsilons in relative terms.
     * Exact zeros are required for zero fractions, including during signed transfers.
     * @throws std::invalid_argument For incompatible composition or an unrepresentable scaled mass.
     */
    [[nodiscard]] double scaled_mass(std::span<const double> moles) const;

    /** Derive a signed H1 change from a compatible packet of mol/cell in catalog order.
     * @throws std::invalid_argument For incompatible composition or an unrepresentable scaled mass change.
     */
    [[nodiscard]] double scaled_mass_change(std::span<const double> signed_moles) const;

    /** Generate an extensive fixed-composition packet in mol/cell, in catalog order, from signed H1.
     * @throws std::invalid_argument If the packet cannot represent the requested mass and composition.
     */
    [[nodiscard]] std::vector<double> species_moles(double scaled_mass) const;

private:
    QbmmPopulationSchema(std::shared_ptr<const SpeciesCatalog> catalog, std::uint64_t catalog_fingerprint,
        std::string population, std::string_view phase_identity, bool dispersed_phase, double mass_scale_kg,
        std::vector<QbmmMassFraction> fractions, QbmmClosureVersions versions);

    std::shared_ptr<const SpeciesCatalog> d_catalog;
    std::string d_population;
    double d_mass_scale_kg;
    QbmmClosureVersions d_versions;
    std::vector<double> d_mass_fractions;
    long double d_mixture_molar_mass;
    std::uint64_t d_fingerprint;
};

/** One old fixed-composition pivot with an actual extensive count and unscaled moles per bubble. */
struct QbmmMolarPivot
{
    double count, mol_per_bubble;
};

/** Initialize missing shape with the explicit assumption of monodispersity.
 * @param schema The binding for the proposed family.
 * @param number Existing extensive bubble count.
 * @param moles Existing material inventory in schema catalog order, mol/cell.
 * @return The shape moments only. Count and species remain with their existing owners.
 * @throws std::invalid_argument For inconsistent count or composition or an unrepresentable moment block.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmShapeState qbmm_initialize_monodisperse(
    const QbmmPopulationSchema& schema, double number, std::span<const double> moles);

/** Convert externally supplied active pivots with extensive count and mass divided by `schema.mass_scale_kg()`.
 * `number` and `moles` are the existing stock. Supply the active positive pivots only, or an empty span for absence.
 * @throws std::invalid_argument If the pivots disagree with stock or their moment block cannot be represented.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmShapeState qbmm_migrate_mass_pivots(
    const QbmmPopulationSchema& schema, double number, std::span<const double> moles, std::span<const QbmmAtom> pivots);

/** Convert fixed-composition molar pivots with the schema's mixture molar mass.
 * `number` and `moles` are the existing stock. Zero-count legacy bins are allowed with a positive finite pivot.
 * @throws std::invalid_argument If the pivots disagree with stock or their moment block cannot be represented.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmShapeState qbmm_migrate_molar_pivots(const QbmmPopulationSchema& schema,
    double number, std::span<const double> moles, std::span<const QbmmMolarPivot> pivots);

/** Convert unscaled extensive moments in (mol/bubble)^k times count, using the declared fixed composition.
 * `number` and `moles` are the existing stock. The old molar-coordinate population must share the schema composition.
 * @throws std::invalid_argument If the moments disagree with stock or fail representability or realizability checks.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmShapeState qbmm_convert_molar_moments(const QbmmPopulationSchema& schema,
    double number, std::span<const double> moles, QbmmMomentBlock unscaled_molar_moments);

/** Explicitly change the mass scaling of a temporary moment block for the same family.
 * The caller retains the authoritative stock and carries only the resulting shape across schema migration.
 * @throws std::invalid_argument If the schemas differ in other properties or either block is inadmissible.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmMomentBlock qbmm_rescale_moments(
    const QbmmPopulationSchema& from, const QbmmPopulationSchema& to, QbmmMomentBlock moments);
} // namespace SimpleFluid::Reacting
