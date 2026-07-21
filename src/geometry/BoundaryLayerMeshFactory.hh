/**
 * @file BoundaryLayerMeshFactory.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief In-place, topology-family-preserving boundary-layer mesh refinement.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/Database.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace SimpleFluid
{

/**
 * @brief Rebuild an input mesh in place with graded boundary-normal layers.
 *
 * The factory preserves the concrete geometry family: Cartesian input remains
 * `OrthogonalCartesian3D`, cylindrical input remains
 * `OrthogonalCylindrial3D`, and axial refinement of a semi-structured input
 * remains `SemiStructuredXY_Z` with identical XY topology.
 *
 * Boundary-layer stacks replace input edges covered by the stack and retain
 * every original edge in the unaffected interior.  The first cell height is
 * measured from the named wall; subsequent wall-normal widths are multiplied
 * by `growth_ratio`.  Widths use the mesh's native coordinate units, so theta
 * boundary widths on a cylindrical sector are specified in radians.
 *
 * Apply this factory before constructing a `MeshHandle` or mesh-backed fields,
 * since changing cell counts invalidates their cached IDs and maps.
 */
class BoundaryLayerMeshFactory
{
public:
    /**
     * @brief One graded layer stack attached to a named mesh boundary.
     *
     * The first cell starts at the boundary and subsequent widths are
     * multiplied by @ref growth_ratio.
     */
    struct BoundaryLayerSpec
    {
        std::string boundary_name;
        size_t count = 0;
        real_t first_cell_height = 0.0;
        real_t growth_ratio = 1.0;
    };

    /**
     * @brief Construct from explicit boundary-layer specifications.
     * @param layer_specs Named layer stacks to validate and retain.
     * @throws std::invalid_argument If names are empty or duplicated, counts
     *         or first-cell heights are not positive, or ratios are below one.
     */
    explicit BoundaryLayerMeshFactory(
        Arr<BoundaryLayerSpec> layer_specs);

    /**
     * @brief Construct from the existing flat boundary-layer database keys.
     *
     * Reads `boundary_layer_boundary_names`, `boundary_layer_counts`,
     * `boundary_layer_first_cell_heights`, and
     * `boundary_layer_growth_ratios`.
     *
     * @param database Configuration database containing the parallel arrays.
     * @throws std::invalid_argument If the database or configuration is invalid.
     */
    explicit BoundaryLayerMeshFactory(SP<const Database> database);

    /**
     * @brief Refine a Cartesian mesh while retaining orthogonal topology.
     * @param mesh Mesh to rebuild in place.
     */
    void build(Meshes::OrthogonalCartesian3D& mesh) const;

    /**
     * @brief Refine a cylindrical mesh while retaining orthogonal topology.
     * @param mesh Mesh to rebuild in place.
     */
    void build(Meshes::OrthogonalCylindrial3D& mesh) const;

    /**
     * @brief Refine zmin/zmax of a semi-structured mesh in place.
     *
     * Side-layer generation is rejected because it would require changing the
     * supplied XY topology rather than preserving it.
     *
     * @param mesh Mesh to rebuild in place.
     * @throws std::invalid_argument If a configured boundary is not zmin or zmax.
     */
    void build(Meshes::SemiStructuredXY_Z& mesh) const;

    /**
     * @brief Shared-pointer convenience overload preserving object identity.
     * @tparam MeshType Supported concrete mesh family.
     * @param mesh Mesh to rebuild in place.
     * @throws std::invalid_argument If @p mesh is null.
     */
    template<class MeshType>
    void build(const SP<MeshType>& mesh) const
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "BoundaryLayerMeshFactory requires a non-null mesh.");
        }
        build(*mesh);
    }

    const Arr<BoundaryLayerSpec>& layer_specs() const noexcept
    {
        return d_layer_specs;
    }

private:
    static Arr<BoundaryLayerSpec> read_specs(
        const SP<const Database>& database);

    const BoundaryLayerSpec* find_spec(
        const std::string& boundary_name) const noexcept;

    void validate_supported_boundaries(
        const ArrString& boundary_names) const;

    static ArrReal refine_edges(
        const ArrReal& input_edges,
        const BoundaryLayerSpec* lower,
        const BoundaryLayerSpec* upper,
        const std::string& coordinate_name);

    Arr<BoundaryLayerSpec> d_layer_specs;
};

} // namespace SimpleFluid
