#pragma once

#include "equations/QbmmPopulationEvents.hh"
#include "equations/details/LowMachScaledFaceSum.hh"
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
using RawSpeciesExactStocks = std::vector<qbmm_face_detail::ScaledFaceSum>;

/** The exact stocks carry successful joint reservations into later source substeps. On the first
 * substep their empty vector requests initialization from the owner's raw high/correction rows.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_LOCAL QbmmJointPopulationAdvance advance_joint_with_raw_species(
    const CellSpeciesState& initial, std::span<const QbmmFamilyEventInput> families, const QbmmBulkGasPoolBinding& pool,
    std::span<const std::array<long double, 2>> raw_species, RawSpeciesExactStocks& exact_stocks,
    std::span<StageMomentChanges> adaptive_changes,
    std::span<std::array<long double, 4>> moment_changes);
} // namespace SimpleFluid::Reacting::qbmm_event_detail
