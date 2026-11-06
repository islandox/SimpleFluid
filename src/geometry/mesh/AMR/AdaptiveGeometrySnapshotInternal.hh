#pragma once
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
{
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
    std::vector<AdaptiveCellMetrics> metrics;
    std::vector<GeometryEvaluationPolicy> policy;
};
struct Builder
{
    template<ArithmeticMode Mode>
    static CatalogBuild build(const DistributedAdaptiveHierarchy&, std::span<const AdaptiveLeaf* const>,
        adaptive_detail::Budget&, const AdaptiveGeometrySnapshot*, CatalogPurpose, PLIC::GeometryAllocationLedger&,
        PLIC::GeometryAllocationLedger&, std::span<const PublicationSource>, BuildFacts&);
};
GeometryConstructionControls construction_controls();
std::size_t checked_product(std::size_t, std::size_t);
void charge_items(adaptive_detail::Budget&, std::size_t, std::size_t);
bool eligible(const DistributedAdaptiveHierarchy&, const AdaptiveLeaf&, ArithmeticMode);
PLIC::GeometryLimits geometry_limits(ArithmeticMode, std::size_t);
struct RecipeAxes
{
    std::array<std::uint64_t, 3> counts{};
    std::array<std::span<const char>, 3> values;
    explicit RecipeAxes(const NativeFragmentDescriptor&);
    double value(unsigned, std::uint64_t) const;
    std::array<std::uint64_t, 3> location(const AdaptiveLeaf&) const;
    AdaptiveCellMetrics metrics(const AdaptiveLeaf&) const;
};
} // namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
