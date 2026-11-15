/** @file STKFaceAuthority.hh
 *  @brief STK-free agreement of detached input face authority.
 */
#pragma once

#include "utils/export_helper.hh"
#include "geometry/mesh/UnstructuredMeshData.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

namespace SimpleFluid::detail
{
/**
 * @brief Select and copy one complete source record for every owned-incident face.
 *
 * Call collectively after global boundary-ID remapping. The producer with the
 * smallest incident locally owned source-cell ID wins (rank breaks ties).
 * Equivalent cyclic/reversed loops, unordered cell incidence, effective
 * algebraic IDs, boundary labels/catalog and source node coordinates must
 * agree. Metrics and ordered loops are copied together from the winner.
 * Aura-only records, source nodes and cell face IDs remain unchanged.
 * A single-rank communicator returns the input unchanged.
 */
SIMPLEFLUID_LOCAL void synchronize_stk_face_authority(
    Meshes::UnstructuredMeshData& data, const Teuchos::RCP<const Teuchos::Comm<int>>& comm);
} // namespace SimpleFluid::detail
