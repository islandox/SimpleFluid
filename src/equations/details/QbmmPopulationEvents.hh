#pragma once

#include "equations/QbmmPopulationEvents.hh"

namespace SimpleFluid::Reacting::qbmm_event_detail
{
/** Owner-local species predecessor in CellSpeciesState order: unnamed compartments, then named populations. */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_LOCAL QbmmPopulationAdvance advance_with_raw_species(
    const QbmmPopulationSchema& schema, const CellSpeciesState& initial, QbmmCellView pre_cell,
    const QbmmEventBatch& batch, std::span<const std::array<long double, 2>> raw_species);
} // namespace SimpleFluid::Reacting::qbmm_event_detail
