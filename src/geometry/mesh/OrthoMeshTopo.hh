/**
 * @file OrthoMeshTopo.hh
 * @brief Topology queries shared by three-dimensional orthogonal meshes.
 */

#pragma once

#include "geometry/mesh/BoundaryFacePatch.hh"
#include "geometry/mesh/OrthogonalIndexer.hh"

#include <array>
#include <string>
#include <unordered_map>

namespace SimpleFluid::Mesh
{

/**
 * @brief Connectivity and boundary patches for an orthogonal 3D mesh.
 *
 * Boundary patch IDs are assigned in dimension order: lower and upper I,
 * lower and upper J, then lower and upper K. Periodic dimensions have no
 * boundary patches and wrap the face at coordinate zero.
 */
class OrthoMeshTopo
{
public:
    using Indexer = OrthogonalIndexer;
    using Ordinal = Indexer::Ordinal;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using BoundaryPatch = BoundaryFacePatch<FaceID>;
    using BoundaryPatchMap = std::unordered_map<int, BoundaryPatch>;
    using BoundaryNames = std::array<std::string, 6>;

    static constexpr int invalid_boundary_id = -1;

    OrthoMeshTopo() = default;
    OrthoMeshTopo(const Indexer& indexer, BoundaryNames boundary_names);
    OrthoMeshTopo(Ordinal ni, Ordinal nj, Ordinal nk,
                  bool periodic_i, bool periodic_j, bool periodic_k, BoundaryNames boundary_names);

    const Indexer& indexer() const noexcept { return d_indexer; }

    CellID owner_cell(FaceID face_id) const noexcept;
    CellID neighbor_cell(FaceID face_id) const noexcept;

    bool is_boundary_face(FaceID face_id) const noexcept;
    int boundary_id(FaceID face_id) const noexcept;
    const std::string& boundary_patch_name(int patch_id) const;
    const BoundaryPatch& boundary_face_patch(int patch_id) const;
    const BoundaryPatchMap& boundary_patches() const noexcept
    {
        return d_boundary_patches;
    }

private:
    void initialize_boundary_patches();

    Indexer d_indexer;
    BoundaryNames d_boundary_names{};
    BoundaryPatchMap d_boundary_patches;
};

} // namespace SimpleFluid::Mesh
