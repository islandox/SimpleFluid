/** @file STKMeshInput.hh
 * @brief STK-free options and detached records for mesh-file input.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "geometry/mesh/UnstructuredMeshData.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <string>
#include <unordered_map>

namespace SimpleFluid
{
struct STKMeshInputOptions
{
    std::unordered_map<std::string, int> boundary_name_to_id;
    bool auto_assign_boundary_ids = true;
    // auto: retain a complete existing decomposed set; otherwise read a
    // single base file with LINEAR input decomposition. Partial sets fail.
    // Empty explicitly selects decomposed files; named Ioss methods pass on.
    std::string decomposition_method = "auto";
};
namespace detail
{
/** Read collectively on comm; no STK object survives the returned records. */
SIMPLEFLUID_LOCAL Meshes::UnstructuredMeshData read_stk_mesh_input(const std::string& filename,
    const STKMeshInputOptions& options, const Teuchos::RCP<const Teuchos::Comm<int>>& comm);
} // namespace detail
} // namespace SimpleFluid
