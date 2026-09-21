/**
 * @file AnnularSemiStructuredMeshFactory.hh
 * @brief Polygonal annular XY meshes with graded wall layers, extruded in Z.
 */
#pragma once

#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/MultiRegionMesh.hh"

#include <cstddef>

namespace SimpleFluid
{

/**
 * @brief Build triangular bulk prisms with hexahedral radial-wall layers.
 *
 * All lengths are in mesh coordinate units (metres in the SI solvers). The
 * inner polygon circumscribes the inner circle and the outer polygon is
 * inscribed in the outer circle. Thus the polygonal fluid region lies inside
 * the requested analytic annulus. Native polygon areas/volumes are retained;
 * no area or volume normalization is applied.
 *
 * Wall layers have prescribed boundary-normal widths. Circular-wall layers
 * use parallel regular-polygon sides, with the apothem as the normal
 * coordinate. Inner and outer stacks have independent angular counts. Auto
 * sizing uses each stack's largest circumference, with even counts of at
 * least eight, so the inner wall is not over-resolved by the outer radius.
 * Axial bulk intervals are split evenly, independently of wall layers.
 * FrontalDelaunay2D triangulates the bulk between the fixed wall interfaces.
 * Its advancing fronts generate interior points independently of the radial
 * wall bands, and its constrained edges preserve the annular hole. Inner/outer
 * wall bands remain quadrilateral, producing hexahedral wall cells. Bottom
 * and optional top stacks retain bulk triangular prisms. Where radial and
 * bottom widths are both below half the requested spacing, mitered R-Z
 * quadrilaterals replace the fine tensor-product overlap and sweep into
 * hexahedra. Every regional interface matches faces one-to-one.
 * Custom spacing and wall widths must satisfy
 * the solver's existing mesh-quality gate; skinny triangular cells can exceed
 * its non-orthogonality limit even when all native volumes remain positive.
 *
 * Build collectively on the default Tpetra communicator before constructing
 * fields or handles. The returned MultiRegionMesh owns immutable native
 * SemiStructuredXY_Z and SweptRZRegion children and supports distributed common
 * affine Z motion.
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
        size_t inner_angular_cells = 0; ///< Zero sizes the inner stack automatically.
        size_t outer_angular_cells = 0; ///< Zero sizes the outer stack automatically.
        size_t wall_layers = 8; ///< Zero disables all boundary-layer stacks.
        real_t first_layer_height = 0.002;
        real_t growth_ratio = 1.25;
        bool refine_inner = true;
        bool refine_outer = true;
        bool refine_bottom = true;
        bool refine_top = false;
        bool coarsen_bottom_corners = true;
    };

    struct Counts
    {
        size_t angular_cells = 0; ///< Outer count, retained for compatibility.
        size_t inner_angular_cells = 0;
        size_t outer_angular_cells = 0;
        size_t radial_cells = 0; ///< Reference radial guide intervals, not bulk topology.
        size_t axial_cells = 0;
        size_t xy_cells = 0; ///< Reference mixed XY template above corner transitions.
        size_t coarse_xy_cells = 0; ///< Unsplit parent sectors; not a region cell count.
        size_t mixed_xy_cells = 0;
        size_t wall_xy_cells = 0; ///< Quadrilaterals in radial wall layers.
        size_t bulk_xy_cells = 0; ///< Actual FrontalDelaunay2D triangles.
        size_t xy_nodes = 0;
        size_t bottom_axial_cells = 0;
        size_t bulk_axial_cells = 0;
        size_t top_axial_cells = 0;
        size_t hex_cells = 0;
        size_t prism_cells = 0;
        size_t corner_layers = 0;
        size_t corner_cells = 0;
        size_t removed_corner_cells = 0;
        size_t regions = 0;
        size_t cells = 0;
        size_t faces = 0;
        size_t nodes = 0;
        bool operator==(const Counts&) const = default;
    };

    struct Result
    {
        SP<Meshes::MultiRegionMesh> mesh;
        real_t cross_section_area = 0.0; ///< Actual native polygonal area.
        ArrReal axial_fractions; ///< Reference Z edges normalized to [0,1].
        ArrReal radial_apothems; ///< Wall fronts and nominal bulk radial guides.
        size_t angular_cells = 0;
        real_t analytic_cross_section_area = 0.0;
        real_t relative_area_deficit = 0.0;
        Counts counts;
    };

    /** @brief Validate finite dimensions, positive spacing and layer settings. */
    static void validate_options(const Options& options);

    explicit AnnularSemiStructuredMeshFactory(Options options);

    /**
     * @brief Plan exact counts by constructing the 2D mesh without 3D extrusion.
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
        struct Region
        {
            std::string name;
            bool swept = false;
            Arr<Meshes::SemiStructuredXY_Z::Vec3> nodes;
            Arr<Arr<unsigned>> cells;
            Arr<Meshes::SemiStructuredXY_Z::BoundaryEdge> boundaries;
            ArrReal edges;
        };
        struct Join
        {
            size_t first, second;
            std::string first_boundary, second_boundary;
        };
        ArrReal radial_apothems;
        ArrReal z_edges;
        Arr<Meshes::SemiStructuredXY_Z::Vec3> xy_nodes;
        Arr<Arr<unsigned>> xy_cells;
        Arr<Meshes::SemiStructuredXY_Z::BoundaryEdge> boundaries;
        Counts counts;
        std::vector<Region> regions;
        std::vector<Join> joins;
    };
    Layout plan_layout() const;
    Options d_options;
};

} // namespace SimpleFluid
