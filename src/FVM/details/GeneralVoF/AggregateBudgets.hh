/** @file AggregateBudgets.hh
 * @brief Sparse owner authority for one shared Ordinary phase-quality allowance.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"

namespace SimpleFluid::VoF::general_vof_detail
{
/** Each local count includes every geometric face candidate and both stream
 * signs. Owners sum counts across canonical providers before any requester
 * divides an old-phase allowance. All ranks perform the same two exchanges. */
SIMPLEFLUID_FVM_LOCAL std::map<std::uint64_t, std::size_t> aggregate_authority_counts(const GeneralVoF::Data&,
    const std::map<std::uint64_t, std::size_t>& local_counts, const PLIC::GeometryPacketContext&);
} // namespace SimpleFluid::VoF::general_vof_detail
