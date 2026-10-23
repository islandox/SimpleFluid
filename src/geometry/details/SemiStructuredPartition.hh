/** @file SemiStructuredPartition.hh
 * @brief Compact base-XY graph partitioning for layered native mesh views.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "dataclass/TpetraTypes.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"
#include <cstddef>
#include <vector>

namespace SimpleFluid::detail
{
/** Collectively partition only the base-cell adjacency graph with Zoltan2/ParMETIS.
 * Rank-local graph rows exist on a prefix communicator; the full communicator
 * receives only one compact base-cell owner vector. Labels are canonicalized
 * by minimum base-cell ordinal. No extruded incidence or geometry is built.
 */
SIMPLEFLUID_LOCAL std::vector<size_t> partition_semi_structured_xy(const Meshes::SemiStructMeshTopo& topology,
    size_t parts, bool allow_empty, const Teuchos::RCP<const Teuchos::Comm<int>>& comm);
} // namespace SimpleFluid::detail
