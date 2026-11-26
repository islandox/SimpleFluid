#pragma once

#include "equations/QbmmPopulationEvents.hh"
#include "utils/CompensatedSum.hh"

namespace SimpleFluid::Reacting::qbmm_event_detail
{
struct StageMomentChanges
{
    detail::CompensatedSum<long double> count;
    std::array<detail::CompensatedSum<long double>, 2> shape;
};

/** Owner-local species predecessor in CellSpeciesState order: unnamed compartments, then named populations.
 * Supplying adaptive_changes requests the unrounded signed substep totals and permits duration retries.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_LOCAL QbmmPopulationAdvance advance_with_raw_species(
    const QbmmPopulationSchema& schema, const CellSpeciesState& initial, QbmmCellView pre_cell,
    const QbmmEventBatch& batch, std::span<const std::array<long double, 2>> raw_species,
    StageMomentChanges* adaptive_changes);
using RawSpeciesStocks = std::vector<detail::CompensatedSum<long double>>;

/** Seed from the owner's raw high/correction rows and return the accepted stocks in the same order.
 * The stock and moment outputs are unchanged on rejection.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_LOCAL QbmmJointPopulationAdvance advance_joint_with_raw_species(
    const CellSpeciesState& initial, std::span<const QbmmFamilyEventInput> families, const QbmmBulkGasPoolBinding& pool,
    std::span<const std::array<long double, 2>> raw_species, RawSpeciesStocks& accepted_stocks,
    std::span<StageMomentChanges> adaptive_changes, std::span<std::array<long double, 4>> moment_changes);
} // namespace SimpleFluid::Reacting::qbmm_event_detail
