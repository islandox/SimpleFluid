/**
 * @file AnnularSemiStructuredMeshFactory.hh
 * @brief Polygonal annular XY meshes with graded wall layers, extruded in Z.
 */
#pragma once

#include "geometry/mesh/SemiStructuredXY_Z.hh"

#include <cstddef>

namespace SimpleFluid
{

/**
 * @brief Build an annular quadrilateral XY cross section and axial extrusion.
 *
 * All lengths are in mesh coordinate units (metres in the SI solvers). The
 * inner polygon circumscribes the inner circle and the outer polygon is
 * inscribed in the outer circle. Thus the polygonal fluid region lies inside
 * the requested analytic annulus. Native polygon areas/volumes are retained;
 * no area or volume normalization is applied.
 *
 * Wall layers have prescribed boundary-normal widths. Circular-wall layers
 * use parallel regular-polygon sides, with the apothem as the normal
 * coordinate. Every ring uses the same even angular count, at least eight;
 * its outer circumference spacing is no greater than xy_spacing. Bulk
 * radial and axial intervals are split evenly, independently of wall layers.
 * Inner-ring tangential spacing is consequently smaller than xy_spacing.
 *
 * Build before constructing fields or handles. The returned native geometry
 * is serial; a native_region inside MultiRegionMesh supplies the existing MPI
 * and composite planar-ALE path without changing the XY topology.
 */
class AnnularSemiStructuredMeshFactory
{
public:
    struct Options
    {
        real_t inner_radius = 0.0;
        real_t outer_radius = 0.0;
        real_t bottom = 0.0;
        real_t top = 0.0;
        real_t xy_spacing = 0.01;
        real_t z_spacing = 0.01;
        size_t wall_layers = 8; ///< Zero disables all boundary-layer stacks.
        real_t first_layer_height = 0.001;
        real_t growth_ratio = 1.3;
        bool refine_inner = true;
        bool refine_outer = true;
        bool refine_bottom = true;
        bool refine_top = false;
    };

    struct Counts
    {
        size_t angular_cells = 0;
        size_t radial_cells = 0;
        size_t axial_cells = 0;
        size_t xy_cells = 0;
        size_t xy_nodes = 0;
        size_t cells = 0;
        size_t faces = 0;
        size_t nodes = 0;
        bool operator==(const Counts&) const = default;
    };

    struct Result
    {
        SP<Meshes::SemiStructuredXY_Z> mesh;
        real_t cross_section_area = 0.0; ///< Actual native polygonal area.
        ArrReal axial_fractions; ///< Reference Z edges normalized to [0,1].
        ArrReal radial_apothems; ///< Normal distances of ring sides from axis.
        size_t angular_cells = 0;
        real_t analytic_cross_section_area = 0.0;
        real_t relative_area_deficit = 0.0;
        Counts counts;
    };

    /** @brief Validate finite dimensions, positive spacing and layer settings. */
    static void validate_options(const Options& options);

    explicit AnnularSemiStructuredMeshFactory(Options options);

    /**
     * @brief Plan counts using only one-dimensional coordinate arrays.
     * @throws std::invalid_argument If polygon clearance is insufficient or
     *         boundary-layer stacks overlap/cannot be represented distinctly.
     * @throws std::overflow_error If mesh IDs or size arithmetic would overflow.
     */
    Counts planned_counts() const;

    /** @brief Construct the deterministic mesh with rmin/rmax/zmin/zmax batches. */
    Result build() const;

    const Options& options() const noexcept { return d_options; }

private:
    struct Layout
    {
        ArrReal radial_apothems;
        ArrReal z_edges;
        Counts counts;
    };
    Layout plan_layout() const;
    Options d_options;
};

} // namespace SimpleFluid
