/** @file OrthogonalMeshDecomposition.hh
 * @brief Count-balanced tensor partitioning of shared orthogonal geometry.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/OrthogonalIndexer.hh"
#include <array>
#include <optional>
#include <vector>

namespace SimpleFluid::Meshes
{
/** Rank is iPart + pI*(jPart + pJ*kPart). Counts take priority over physical
 * widths. Every axis block has floor(n/p) or ceil(n/p) cells; widths choose
 * feasible cut locations and break equally balanced process-grid ties.
 */
class SIMPLEFLUID_PUBLIC_TYPE OrthogonalMeshDecomposition
{
public:
    using Shape = std::array<size_t, 3>;
    using Edges = std::array<std::vector<double>, 3>;
    OrthogonalMeshDecomposition(const OrthogonalIndexer& indexer, const Edges& physical_edges, size_t partitions,
        bool allow_empty = false, std::optional<Shape> factors = {});
    const Shape& factors() const noexcept { return d_factors; }
    const std::vector<size_t>& axis_cuts(size_t axis) const { return d_cuts.at(axis); }
    size_t num_partitions() const noexcept { return d_partitions; }
    Shape owned_begin(size_t rank) const;
    Shape owned_end(size_t rank) const;
    size_t num_owned_cells(size_t rank) const;
    size_t owner(OrthogonalIndexer::CellID cell) const;
    size_t storage_bytes() const noexcept;

private:
    Shape d_dimensions{}, d_factors{};
    std::array<std::vector<size_t>, 3> d_cuts;
    size_t d_partitions = 0;
};
} // namespace SimpleFluid::Meshes
