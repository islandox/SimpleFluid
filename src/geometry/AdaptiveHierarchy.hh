/** @file AdaptiveHierarchy.hh
 * @brief Immutable native hierarchy observations and persistent cell ancestry.
 */
#pragma once

#include "geometry/mesh/MultiRegionMesh.hh"
#include <compare>
#include <memory>
#include <variant>
#include <vector>

namespace SimpleFluid::Meshes
{
/** Exact retained observations, independent of solver state and population schema.
 * Identity tokens compare by retained object identity, never by a digest. */
struct HierarchyObservation
{
    std::shared_ptr<const void> identity, topology_identity, partition_identity;
    std::uint64_t topology = 0, partition = 0, geometry = 0;
    /** Effective resident geometry identity; may differ from the canonical catalog. */
    const void* native_geometry = nullptr;
    const void* communicator = nullptr;
    bool operator==(const HierarchyObservation&) const = default;
};

/** Logical dyadic cell coordinates; unlike field GIDs these survive leaf sorting. */
struct CartesianCellAncestry
{
    unsigned level = 0;
    std::array<std::uint64_t, 3> index{};
    auto operator<=>(const CartesianCellAncestry&) const = default;
    [[nodiscard]] CartesianCellAncestry parent() const;
};

/** Explicit convex-cell provenance; no polyhedral refiner is implied. */
struct ConvexCellAncestry
{
    std::uint64_t root = 0;
    std::vector<std::uint64_t> path;
    auto operator<=>(const ConvexCellAncestry&) const = default;
};

struct BaseCellKey
{
    std::uint64_t root = 0, root_template = 0;
    std::vector<std::uint64_t> path;
    auto operator<=>(const BaseCellKey&) const = default;
};
struct AxialIntervalKey
{
    std::uint64_t root = 0;
    std::vector<std::uint64_t> path;
    auto operator<=>(const AxialIntervalKey&) const = default;
};
struct BaseEdgeKey
{
    enum class Kind { Root, TemplateInterior };
    BaseCellKey creator;
    std::uint64_t ordinal = 0;
    std::vector<std::uint64_t> path;
    Kind kind = Kind::Root;
    auto operator<=>(const BaseEdgeKey&) const = default;
};
struct ProductCellAncestry
{
    BaseCellKey base;
    AxialIntervalKey axial;
    auto operator<=>(const ProductCellAncestry&) const = default;
};

struct AdaptiveCellKey
{
    std::uint64_t region = 0;
    std::variant<CartesianCellAncestry, ConvexCellAncestry, ProductCellAncestry> ancestry;
    auto operator<=>(const AdaptiveCellKey&) const = default;
};

/** Immutable native-provider seam. The implemented adaptive provider is Cartesian;
 * convex ancestry also identifies certified geometry without implying a refiner.
 * Material membership and native fragment ordinal are observations, not ancestry.
 * No connectivity or solution field is materialized by this interface. */
class AdaptiveHierarchy
{
public:
    virtual ~AdaptiveHierarchy();
    [[nodiscard]] virtual HierarchyObservation observation() const noexcept = 0;
    /** Canonical geometry catalog, retaining native providers without flattening. */
    [[nodiscard]] virtual const SP<const MultiRegionMesh>& native_geometry() const noexcept = 0;
    [[nodiscard]] virtual AdaptiveCellKey cell_key(std::uint64_t cell) const = 0;
    [[nodiscard]] virtual std::uint64_t cell_material(std::uint64_t cell) const = 0;
    [[nodiscard]] virtual std::size_t cell_fragment(std::uint64_t cell) const = 0;
};
} // namespace SimpleFluid::Meshes
