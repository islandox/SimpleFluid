/** @file PhaseCleanup.hh
 * @brief Separate, budgeted physical phase relabeling on private Ordinary candidates.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
inline void phase_cleanup_key_storage(const Key& key, wire::Budget& budget)
{
    std::visit(
        [&](const auto& ancestry)
        {
            using Ancestry = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<Ancestry, AMR::ConvexCellAncestry>)
                budget.add(ancestry.path.capacity() * sizeof(std::uint64_t));
            else if constexpr (std::is_same_v<Ancestry, AMR::ProductCellAncestry>)
            {
                budget.add(ancestry.base.path.capacity() * sizeof(std::uint64_t));
                budget.add(ancestry.axial.path.capacity() * sizeof(std::uint64_t));
            }
        },
        key.ancestry);
}
SIMPLEFLUID_FVM_LOCAL void validate_phase_cleanup_options(const GeneralVoFOptions&);
/** Collective. Raw controls and prior ledger must agree before automatic inventory reduction. */
SIMPLEFLUID_FVM_LOCAL void resolve_phase_cleanup_options(GeneralVoF::Data&);
/** Collective. Owner-only proposed batch is admitted globally before private candidate mutation. */
SIMPLEFLUID_FVM_LOCAL void apply_phase_cleanup(GeneralVoF::Data&);
} // namespace SimpleFluid::VoF::general_vof_detail
