/** @file QbmmBulkGasPool.hh
 * @brief Explicit binding and frozen physical state of the unnamed Resolved gas pool.
 */
#pragma once

#include "equations/QbmmPopulationSchema.hh"

namespace SimpleFluid::Reacting
{
enum class QbmmPoolConversionDirection
{
    DispersedToPool,
    PoolToDispersed,
    Bidirectional
};

/** Binds the single physical BulkGas phase to the catalog's unnamed Resolved stock.
 * The population binding mask is aligned with the host's phase order.
 */
class QbmmBulkGasPoolBinding
{
public:
    template<class PhaseRoleDescriptor, std::size_t PhaseExtent>
    QbmmBulkGasPoolBinding(std::shared_ptr<const SpeciesCatalog> catalog, std::uint64_t catalog_fingerprint,
        std::span<PhaseRoleDescriptor, PhaseExtent> phases, std::size_t pool_phase, std::span<const SpeciesId> species,
        QbmmPoolConversionDirection direction, std::span<const bool> qbmm_bound)
        : d_catalog(std::move(catalog)), d_phase_index(pool_phase), d_direction(direction)
    {
        using Role = decltype(std::declval<PhaseRoleDescriptor>().role);
        if (!d_catalog || d_catalog->fingerprint() != catalog_fingerprint || pool_phase >= phases.size() ||
            phases[pool_phase].role != Role::BulkGas || qbmm_bound.size() != phases.size() || qbmm_bound[pool_phase] ||
            species.empty() || direction < QbmmPoolConversionDirection::DispersedToPool ||
            direction > QbmmPoolConversionDirection::Bidirectional)
            throw std::invalid_argument(
                "QBMM pool requires one unbound BulkGas phase and an explicit Resolved binding.");

        d_phase_identity = phases[pool_phase].identity;
        d_species.resize(d_catalog->species().size());
        for (const auto id : species)
        {
            const auto index = d_catalog->index(id);
            if (d_species[index] || !d_catalog->allows(id, SpeciesPhase::Resolved))
                throw std::invalid_argument("QBMM pool species must be unique and permitted in Resolved.");
            d_species[index] = true;
        }

        const auto hash = [this](std::uint64_t value)
        {
            for (unsigned shift = 0; shift < 64; shift += 8)
            {
                d_fingerprint ^= static_cast<std::uint8_t>(value >> shift);
                d_fingerprint *= 1099511628211ULL;
            }
        };
        hash(catalog_fingerprint);
        hash(pool_phase);
        hash(static_cast<std::uint64_t>(direction));
        hash(phases[pool_phase].identity.size());
        for (const unsigned char byte : phases[pool_phase].identity)
            hash(byte);
        for (const bool permitted : d_species)
            hash(permitted);
    }

    QbmmBulkGasPoolBinding(const QbmmBulkGasPoolBinding&) = default;
    QbmmBulkGasPoolBinding& operator=(const QbmmBulkGasPoolBinding&) = delete;

    [[nodiscard]] const SpeciesCatalog& catalog() const noexcept { return *d_catalog; }
    [[nodiscard]] std::size_t phase_index() const noexcept { return d_phase_index; }
    [[nodiscard]] const std::string& phase_identity() const noexcept { return d_phase_identity; }
    [[nodiscard]] QbmmPoolConversionDirection direction() const noexcept { return d_direction; }
    [[nodiscard]] std::uint64_t fingerprint() const noexcept { return d_fingerprint; }
    [[nodiscard]] bool permits(SpeciesId id) const { return d_species[d_catalog->index(id)]; }

    [[nodiscard]] bool allows(QbmmPoolConversionDirection direction) const noexcept
    {
        return direction >= QbmmPoolConversionDirection::DispersedToPool &&
               direction <= QbmmPoolConversionDirection::Bidirectional &&
               (d_direction == QbmmPoolConversionDirection::Bidirectional || d_direction == direction);
    }

    void validate_family(const QbmmPopulationSchema& family, QbmmPoolConversionDirection direction) const
    {
        if (family.catalog().fingerprint() != catalog().fingerprint() || !allows(direction))
            throw std::invalid_argument("QBMM pool and family require a shared catalog and an allowed direction.");
        const auto fractions = family.mass_fractions();
        for (std::size_t i = 0; i < fractions.size(); ++i)
            if (fractions[i] != 0 && !d_species[i])
                throw std::invalid_argument("QBMM pool does not permit the family's fixed composition.");
    }

    void validate_inventory(const CellSpeciesState& state) const
    {
        state.validate(catalog());
        for (const auto& species : catalog().species())
            if (state.amount(catalog(), species.id, SpeciesPhase::Resolved) != 0 && !permits(species.id))
                throw std::invalid_argument("QBMM pool stock contains a species outside its binding.");
    }

private:
    std::shared_ptr<const SpeciesCatalog> d_catalog;
    std::size_t d_phase_index;
    QbmmPoolConversionDirection d_direction;
    std::string d_phase_identity;
    std::vector<bool> d_species;
    std::uint64_t d_fingerprint = 14695981039346656037ULL;
};

struct QbmmBulkGasPoolState
{
    const double volume_m3;
    const std::vector<double> moles; ///< Extensive unnamed Resolved row in catalog order.
    const std::optional<std::vector<double>> concentration_mol_per_m3;
    const std::optional<double> temperature_K;
    const std::uint64_t binding_fingerprint;
};

[[nodiscard]] inline QbmmBulkGasPoolState make_qbmm_bulk_gas_pool_state(const QbmmBulkGasPoolBinding& binding,
    const CellSpeciesState& state, double volume_m3, std::optional<double> temperature_K = {},
    std::optional<double> count = {}, std::optional<QbmmShapeState> shape = {})
{
    if (count || shape)
        throw std::invalid_argument("QBMM bulk gas pool cannot own count or shape.");
    binding.validate_inventory(state);
    std::vector<double> moles;
    moles.reserve(binding.catalog().species().size());
    for (const auto& species : binding.catalog().species())
        moles.push_back(state.amount(binding.catalog(), species.id, SpeciesPhase::Resolved));
    auto concentration = qbmm_phase_concentrations(moles, volume_m3);
    if (concentration.has_value() != temperature_K.has_value() ||
        (temperature_K && (!std::isfinite(*temperature_K) || *temperature_K <= 0)))
        throw std::invalid_argument("QBMM pool temperature must be positive finite and present exactly on occupancy.");
    return {volume_m3, std::move(moles), std::move(concentration), temperature_K, binding.fingerprint()};
}
} // namespace SimpleFluid::Reacting
