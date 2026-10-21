/** @file AdaptiveHierarchy.hh
 * @brief Immutable native hierarchy observations and persistent cell ancestry.
 */
#pragma once

#include "geometry/mesh/MultiRegionMesh.hh"
#include <compare>
#include <memory>
#include <variant>

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

struct AdaptiveCellKey
{
    std::uint64_t region = 0;
    std::variant<CartesianCellAncestry> ancestry;
    auto operator<=>(const AdaptiveCellKey&) const = default;
};

/** Immutable native-provider seam. Only Cartesian ancestry is admitted today.
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
