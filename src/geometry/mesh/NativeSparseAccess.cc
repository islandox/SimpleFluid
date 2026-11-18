#include "geometry/mesh/NativeSparseAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include <cmath>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
NativeSourceCellFragment NativeSparseAccess::source_cell_fragment(
    const AdaptiveLeaf& leaf, const AdaptiveCellMetrics& metrics, const Admission& admission,
    std::shared_ptr<const NativeSourceSubsetCoverage> coverage)
{
    if (leaf.provider != AdaptiveProviderKind::Polyhedral || !leaf.geometry || !(metrics.volume > 0) ||
        !std::isfinite(metrics.volume))
        throw std::invalid_argument("Native source-cell fragment requires immutable original polyhedral geometry.");
    auto source =
        geometry_detail::NativeAllocationEnvelopeAccess::make<ExplicitRegionProvider::NativeSourceCell>(admission);
    auto& data = *source.owner;
    data.metadata = geometry_detail::native_constructor_storage(admission);
    data.coverage = std::move(coverage);
    data.geometry = leaf.geometry;
    data.volume = metrics.volume;
    data.centroid = {metrics.centroid[0], metrics.centroid[1], metrics.centroid[2]};
    const auto original_faces = leaf.geometry->faces();
    const auto faces = original_faces.size();
    std::size_t incidence = 0;
    for (std::size_t f = 0; f < faces; ++f)
    {
        const auto count = original_faces.at(f).size();
        if (count > std::numeric_limits<std::size_t>::max() - incidence)
            throw std::invalid_argument("Native source-cell topology count overflows.");
        incidence += count;
    }
    if (faces == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Native source-cell face count overflows.");
    data.metadata.owner->reserve(data.offsets, faces + 1);
    data.metadata.owner->reserve(data.loops, incidence);
    // The incidence bound is exact before uniqueness. Read the original
    // stored loops directly, without constructing planar wrapper/point buffers.
    data.metadata.owner->reserve(data.points, incidence);
    data.offsets.push_back(0);
    for (std::size_t f = 0; f < faces; ++f)
        data.offsets.push_back(data.offsets.back() + original_faces.at(f).size());
    PLIC::details::NativeCellGeometrySourceAccess::visit_original_convex_face_points(*leaf.geometry, &data,
        [](void* raw, const auto& point)
        {
            auto& data = *static_cast<ExplicitRegionProvider::NativeSourceCell*>(raw);
            const auto at = std::find(data.points.begin(), data.points.end(), point);
            const auto ordinal = static_cast<std::uint64_t>(at - data.points.begin());
            if (at == data.points.end())
                data.points.push_back(point);
            data.loops.push_back(ordinal);
        });
    data.allocation = source.envelope;
    auto provider = geometry_detail::NativeAllocationEnvelopeAccess::make<ExplicitRegionProvider>(admission,
        ExplicitRegionProvider::NativeSourceKey{},
        std::shared_ptr<const ExplicitRegionProvider::NativeSourceCell>(source.owner));
    provider.owner->d_native_allocation = provider.envelope;
    auto metadata = geometry_detail::native_constructor_storage(admission);
    std::string name;
    metadata.owner->string(name, "native_source_cell");
    std::shared_ptr<const ExplicitRegionProvider> retained = provider.owner;
    return {DistributedExplicitRegion(std::move(name), retained, retained), std::move(metadata)};
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> NativeSparseAccess::source_provider_geometry(
    const std::shared_ptr<const ExplicitRegionProvider>& provider)
{
    if (!provider || !provider->d_native_source)
        throw std::invalid_argument("Not an admitted source-cell fragment.");
    return provider->d_native_source->geometry;
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> NativeSparseAccess::source_provider_geometry(
    const ExplicitRegionProvider& provider)
{
    const auto& source = provider.d_native_source;
    if (!source) return {};
    geometry_detail::NativeConstructorOwnerSink::require_envelope(source->allocation, source);
    return source->geometry;
}
const std::shared_ptr<const NativeSourceSubsetCoverage>& NativeSparseAccess::source_provider_coverage(
    const std::shared_ptr<const ExplicitRegionProvider>& provider)
{
    if (!provider || !provider->d_native_source)
        throw std::invalid_argument("Not an admitted source-cell fragment.");
    return provider->d_native_source->coverage;
}
bool NativeSparseAccess::visit_source_provider(const std::shared_ptr<const ExplicitRegionProvider>& provider,
    const geometry_detail::NativeConstructorOwnerSink& sink)
{
    using Kind = geometry_detail::NativeConstructorOwnerKind;
    if (!provider || !provider->d_native_source)
        throw std::invalid_argument("Explicit provider is not an admitted single-source native fragment.");
    geometry_detail::NativeConstructorOwnerSink::require_envelope(provider->d_native_allocation, provider);
    if (!sink.edge(provider, Kind::NativeMesh, 0))
        return false;
    sink.fixed(provider->d_native_allocation);
    const auto& source = provider->d_native_source;
    geometry_detail::NativeConstructorOwnerSink::require_envelope(source->allocation, source);
    if (sink.edge(source, Kind::Topology, 0))
    {
        sink.fixed(source->allocation);
        sink.storage(source->metadata);
        return true;
    }
    return false;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
