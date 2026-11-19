#pragma once
#include "geometry/PLIC/details/GeometrySnapshotAllocation.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>
namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
{
template<class T>
using SnapshotVector = PLIC::details::GeometrySnapshotVector<T>;
template<class Key, class Value, class Less = std::less<Key>>
using SnapshotMap = PLIC::details::GeometrySnapshotMap<Key, Value, Less>;
template<class T, class... Args> std::shared_ptr<T> snapshot_shared(Args&&... args)
{
    return PLIC::details::geometry_snapshot_shared<T>(std::forward<Args>(args)...);
}
// Attribution must not change the byte-admission model of the private stores.
static_assert(sizeof(SnapshotVector<std::byte>) == sizeof(std::vector<std::byte>));
static_assert(alignof(SnapshotVector<std::byte>) == alignof(std::vector<std::byte>));
using CatalogBuild = AdaptiveGeometrySnapshot;
using CatalogPurpose = GeometrySnapshotPurpose;
/** Borrowed only during the private acquisition of an authenticated full draft. */
struct PublicationSource
{
    const AdaptiveLeaf* leaf = nullptr;
    const AdaptiveNativeCellBinding* binding = nullptr;
};
struct BuildFacts
{
    SnapshotVector<AdaptiveCellMetrics> metrics;
    SnapshotVector<GeometryEvaluationPolicy> policy;
};
struct Builder
{
    /** `transient_catalog` marks a throwaway catalog that the caller replaces
     * with a persistent demand catalog; its representation metadata is then
     * admitted once, by that persistent catalog, not twice. */
    template<ArithmeticMode Mode>
    static CatalogBuild build(const DistributedAdaptiveHierarchy&, std::span<const AdaptiveLeaf* const>,
        adaptive_detail::Budget&, const AdaptiveGeometrySnapshot*, CatalogPurpose, PLIC::GeometryAllocationLedger&,
        PLIC::GeometryAllocationLedger&, std::span<const PublicationSource>, BuildFacts&,
        bool transient_catalog = false);
};
GeometryConstructionControls construction_controls();
std::size_t checked_sum(std::size_t, std::size_t);
std::size_t checked_product(std::size_t, std::size_t);
void charge_items(adaptive_detail::Budget&, std::size_t, std::size_t);
bool eligible(const DistributedAdaptiveHierarchy&, const AdaptiveLeaf&, ArithmeticMode);
const AdaptiveLeaf* exact_owned_cartesian_source(const DistributedAdaptiveHierarchy&, const AdaptiveLeaf&);
PLIC::GeometryLimits geometry_limits(ArithmeticMode, std::size_t);
struct RecipeAxes
{
    std::array<std::uint64_t, 3> counts{};
    std::array<std::span<const char>, 3> values;
    explicit RecipeAxes(const NativeFragmentDescriptor&);
    double value(unsigned, std::uint64_t) const;
    std::array<std::uint64_t, 3> location(std::uint64_t ordinal) const;
    std::array<std::uint64_t, 3> location(const AdaptiveLeaf&) const;
    AdaptiveCellMetrics metrics(const AdaptiveLeaf&) const;
};
} // namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
