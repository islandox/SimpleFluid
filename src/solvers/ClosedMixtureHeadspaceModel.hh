/** @file ClosedMixtureHeadspaceModel.hh
 * @brief Isothermal composition-aware headspace closure with an explicit supplied gas EOS.
 */
#pragma once
#include "equations/ReactingThermodynamics.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"
#include "utils/CompensatedSum.hh"

namespace SimpleFluid
{
/** Lumped prescribed-temperature pressure closure, not a headspace energy equation.
 * Initial moles must identify actual catalog species; inferring an unnamed gas is
 * unsupported. Escaped inventories are nonnegative integrated mol/species, not rates.
 * Temperature follows HeadspaceOptions; no incoming enthalpy, compression work, or
 * heat-reservoir energy is computed. Thus this adapter supplies pressure closure
 * only, and cannot establish energy conservation of a closed adiabatic headspace.
 * The supplied EOS replaces gas_constant/compressibility_factor in HeadspaceOptions.
 *
 * trialState is side-effect free; commit applies one accepted event. The enclosing
 * PlanarFreeSurfaceModel preview owner/generation checks provide commit-once
 * protection. Direct callers must provide that transaction discipline themselves.
 */
class SIMPLEFLUID_SOLVERS_EXPORT ClosedMixtureHeadspaceModel final : public HeadspaceModel
{
public:
    ClosedMixtureHeadspaceModel(HeadspaceOptions options,
        std::shared_ptr<const Reacting::MixtureThermodynamics> thermodynamics, std::string provider_identity,
        std::uint64_t provider_revision);
    [[nodiscard]] std::unique_ptr<HeadspaceModel> clone() const override;
    [[nodiscard]] HeadspaceMode mode() const noexcept override;
    void initialize(real_t pool_volume, real_t supplied_temperature) override;
    [[nodiscard]] HeadspaceState trialState(real_t pool_volume, real_t supplied_temperature,
        const GasMolesBySpecies& escaped_moles) const override;
    void commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles) override;
    [[nodiscard]] const GasMolesBySpecies& headspaceMoles() const noexcept override;
    [[nodiscard]] const GasMolesBySpecies& ventedMoles() const noexcept override;
    /** Accepted integrated escapes [mol/species], accumulated separately from initial inventory. */
    [[nodiscard]] const GasMolesBySpecies& accumulatedEscapedMoles() const noexcept;

private:
    using AmountSums = std::map<std::string, detail::CompensatedSum<long double>>;
    struct InventoryCandidate
    {
        GasMolesBySpecies moles, escaped;
        AmountSums amount_sums, escape_sums;
    };
    SIMPLEFLUID_SOLVERS_LOCAL InventoryCandidate prospective(const GasMolesBySpecies& escaped) const;
    SIMPLEFLUID_SOLVERS_LOCAL HeadspaceState recover(real_t volume, real_t temperature,
        const GasMolesBySpecies& amounts) const;
    std::shared_ptr<const Reacting::MixtureThermodynamics> d_thermodynamics;
    GasMolesBySpecies d_moles, d_escaped, d_empty_vent;
    AmountSums d_amount_sums, d_escape_sums;
    bool d_initialized = false;
};
} // namespace SimpleFluid
