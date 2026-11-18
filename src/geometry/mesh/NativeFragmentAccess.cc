#include "geometry/mesh/NativeFragmentAccess.hh"
#include "geometry/mesh/AMR/NativeAdaptivePayloadAccess.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include "geometry/details/NativeAllocationGraph.hh"
#include "geometry/mesh/NativeFragmentDecodePlan.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include "geometry/mesh/details/NativeCartesianAccess.hh"
#include "geometry/mesh/details/NativeLegacyCartesianAccess.hh"
#include "geometry/mesh/details/NativeProviderOwnerAccess.hh"
#include "geometry/mesh/details/NativeSemiStructuredAccess.hh"
#include <cstring>
#include <limits>

namespace SimpleFluid::Meshes
{
namespace
{
using Admission = geometry_detail::NativeConstructionAdmission;
using Storage = geometry_detail::NativeConstructorStorageOwner;
using Layout = RegionLayout;
struct Cursor
{
    std::span<const char> bytes;
    template<class T> T value()
    {
        if (bytes.size() < sizeof(T))
            throw std::invalid_argument("Truncated native fragment value.");
        T result;
        std::memcpy(&result, bytes.data(), sizeof(T));
        bytes = bytes.subspan(sizeof(T));
        return result;
    }
    std::size_t count(std::size_t width = 1)
    {
        const auto n = value<std::uint64_t>();
        if (!width || n > bytes.size() / width)
            throw std::invalid_argument("Native fragment extent exceeds packet.");
        return static_cast<std::size_t>(n);
    }
    std::span<const char> packet()
    {
        const auto n = count();
        auto result = bytes.first(n);
        bytes = bytes.subspan(n);
        return result;
    }
    std::string_view text()
    {
        const auto span = packet();
        return {span.data(), span.size()};
    }
    template<class T> void array(std::vector<T>& result, const Storage& storage)
    {
        const auto n = count(sizeof(T));
        storage.owner->reserve(result, n);
        for (std::size_t i = 0; i < n; ++i)
            result.push_back(value<T>());
    }
    void finish() const
    {
        if (!bytes.empty())
            throw std::invalid_argument("Trailing native fragment data.");
    }
};
bool equivalent(const NativeFragmentDescriptor& a, const NativeFragmentDescriptor& b)
{
    if (a.kind != b.kind || a.view != b.view || a.parts.size() != b.parts.size())
        return false;
    for (std::size_t i = 0; i < a.parts.size(); ++i)
        if (!a.parts[i] || !b.parts[i] || *a.parts[i] != *b.parts[i])
            return false;
    return true;
}
Layout layout(Cursor& in)
{
    Layout result{};
    result.family = in.value<Layout::Family>();
    result.cells = in.value<std::uint64_t>();
    result.faces = in.value<std::uint64_t>();
    result.nodes = in.value<std::uint64_t>();
    for (auto& n : result.extents)
        n = in.value<std::uint64_t>();
    for (auto& p : result.periodic)
    {
        const auto v = in.value<std::uint8_t>();
        if (v > 1)
            throw std::invalid_argument("Bad native periodic flag.");
        p = v;
    }
    return result;
}
void require_layout(const MultiRegionMesh::Region& region, const Layout& expected)
{
    const auto actual = std::visit([](const auto& r) { return r.layout(); }, region);
    if (actual.family != expected.family || actual.cells != expected.cells || actual.faces != expected.faces ||
        actual.nodes != expected.nodes || actual.extents != expected.extents || actual.periodic != expected.periodic)
        throw std::invalid_argument("Detached native fragment differs from declared layout.");
}
} // namespace
std::shared_ptr<const NativeFragmentRecord> NativeFragmentAccess::freeze(
    MultiRegionMesh::Region region, Storage metadata, const Admission& admission)
{
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeFragmentRecord>(
        admission, NativeFragmentRecord::ConstructionKey{}, std::move(region), std::move(metadata));
    tracked.owner->allocation_ = tracked.envelope;
    return std::move(tracked.owner);
}
std::shared_ptr<const NativeFragmentRecord> NativeFragmentAccess::find(
    const std::shared_ptr<const NativeFragmentDescriptor>& descriptor, AMR::GeometryConstructionControls controls)
{
    if (!descriptor || !descriptor->provenance.origin_)
        return {};
    const auto& origin = descriptor->provenance.origin_;
    const auto bound = origin->descriptor_.lock();
    const auto& envelope = origin->allocation_;
    const auto& record = origin->fragment_;
    if (!bound || bound.get() != descriptor.get() || bound.owner_before(descriptor) || descriptor.owner_before(bound) ||
        !envelope || !envelope->authentic_owner(envelope) || !envelope->matches(origin) || !record ||
        !record->allocation_ || !record->allocation_->authentic_owner(record->allocation_) ||
        !record->allocation_->matches(record) || !origin->descriptor_allocation_ ||
        !origin->descriptor_allocation_->authentic_owner(origin->descriptor_allocation_) ||
        !origin->descriptor_allocation_->matches(descriptor) || controls != origin->controls_)
        return {};
    std::visit([](const auto& r) { r.validate_static(); }, record->region_);
    return record;
}
std::shared_ptr<const NativeFragmentDescriptor> NativeFragmentAccess::bind(
    const std::shared_ptr<const NativeFragmentDescriptor>& descriptor,
    std::shared_ptr<const NativeFragmentRecord> record, const Admission& admission)
{
    if (!descriptor || !record)
        throw std::invalid_argument("Missing native descriptor/fragment origin.");
    auto metadata = geometry_detail::native_constructor_storage(admission);
    // Descriptor bytes are immutable semantic input. Copy the mutable public
    // descriptor's vectors into an exact fresh control before attaching trust.
    auto controlled = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeFragmentDescriptor>(admission);
    controlled.owner->kind = descriptor->kind;
    metadata.owner->copy(controlled.owner->view, std::span<const char>(descriptor->view));
    metadata.owner->copy(
        controlled.owner->parts, std::span<const std::shared_ptr<const NativeFragmentBytes>>(descriptor->parts));
    auto origin = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeFragmentOrigin>(
        admission, NativeFragmentOrigin::ConstructionKey{});
    origin.owner->fragment_ = std::move(record);
    origin.owner->controls_ = AMR::geometry_construction_controls();
    origin.owner->descriptor_ = controlled.owner;
    origin.owner->metadata_ = std::move(metadata);
    origin.owner->descriptor_allocation_ = controlled.envelope;
    origin.owner->allocation_ = origin.envelope;
    controlled.owner->provenance.origin_ = std::move(origin.owner);
    return std::move(controlled.owner);
}
std::shared_ptr<const NativeFragmentDescriptor> NativeFragmentAccess::rebind(
    const std::shared_ptr<const NativeFragmentDescriptor>& source,
    const std::shared_ptr<const NativeFragmentDescriptor>& normalized, const Admission& admission)
{
    const auto record = find(source, AMR::geometry_construction_controls());
    if (!record || !normalized || !equivalent(*source, *normalized))
        throw std::invalid_argument("Native normalization lacks exact owner-local descriptor provenance.");
    return bind(normalized, record, admission);
}
std::shared_ptr<const NativeFragmentDescriptor> NativeFragmentAccess::extract(
    const MultiRegionMesh::Region& region, NativeFragmentCache& cache, const Admission& admission)
{
    auto descriptor = extract_native_fragment(region, cache);
    if (const auto native = std::get_if<NativeIsoRegion<OrthogonalCartesian3D>>(&region))
    {
        auto detached = details::NativeLegacyCartesianAccess::detach_region(*native, admission);
        auto record = freeze(std::move(detached.region), std::move(detached.metadata), admission);
        return bind(descriptor, std::move(record), admission);
    }
    // A const Region alone does not establish immutable provenance. Share an
    // actual factory-admitted provider only after all of its exact owner
    // envelopes validate. Mutable legacy XY/Z receives a direct detached copy
    // preserving original metric bits and fresh execution/epoch state.
    auto metadata = geometry_detail::native_constructor_storage(admission);
    std::string name;
    std::visit([&](const auto& r) { metadata.owner->string(name, r.name()); }, region);
    std::shared_ptr<const NativeFragmentRecord> record;
    if (const auto native = std::get_if<NativeIsoRegion<SemiStructuredXY_Z>>(&region))
    {
        const auto detached =
            details::NativeSemiStructuredAccess::detached_copy(native->geometry().native(), admission);
        using Provider = NativeRegionProvider<SemiStructuredXY_Z>;
        auto provider = geometry_detail::NativeAllocationEnvelopeAccess::make<Provider>(admission, detached);
        provider.owner->d_native_allocation = provider.envelope;
        record = freeze(NativeIsoRegion<SemiStructuredXY_Z>(std::move(name), provider.owner, provider.owner),
            std::move(metadata), admission);
    }
    else if (const auto cartesian = std::get_if<CartesianRegion>(&region))
        record = freeze(CartesianRegion(std::move(name), cartesian->d_topology, cartesian->d_geometry),
            std::move(metadata), admission);
    else if (const auto product = std::get_if<ExtrudedRegion>(&region))
        record = freeze(
            ExtrudedRegion(std::move(name), product->d_topology, product->d_geometry), std::move(metadata), admission);
    if (record)
    {
        geometry_detail::NativeConstructorOwnerSink inspect;
        inspect.owner_edge = [](void*, const std::shared_ptr<const void>&, geometry_detail::NativeConstructorOwnerKind,
                                 std::size_t) { return true; };
        inspect.fixed_edge = [](void*, const auto&) {};
        inspect.state_edge = [](void*, const auto& value)
        { (void) geometry_detail::NativeAllocationGraphAccess::validate_state(value); };
        try
        {
            visit(record, inspect, [](void*, const auto&) {});
            return bind(descriptor, std::move(record), admission);
        }
        catch (const std::invalid_argument&)
        {
            // An older factory has no immutable admitted child receipts. Decode
            // its canonical descriptor through the same admitted constructors.
            record.reset();
        }
    }
    return bind(descriptor, materialize(descriptor, admission), admission);
}
std::shared_ptr<const NativeFragmentRecord> NativeFragmentAccess::materialize(
    const std::shared_ptr<const NativeFragmentDescriptor>& descriptor, const Admission& admission)
{
    if (auto existing = find(descriptor, AMR::geometry_construction_controls()))
        return existing;
    if (!descriptor)
        throw std::invalid_argument("Null native fragment descriptor.");
    if (descriptor->kind == NativeFragmentDescriptor::Kind::LegacyProcedural && descriptor->parts.size() == 1 &&
        descriptor->parts[0] && descriptor->parts[0]->size() >= sizeof(std::uint32_t))
    {
        std::uint32_t tag;
        std::memcpy(&tag, descriptor->parts[0]->data(), sizeof(tag));
        if (tag == 1)
        {
            Cursor view{descriptor->view};
            const auto name = view.text();
            view.finish();
            if (name.empty())
                throw std::invalid_argument("Legacy Cartesian descriptor requires its view name.");
            auto native = details::NativeLegacyCartesianAccess::decode(*descriptor->parts[0], admission, name);
            return freeze(std::move(native.region), std::move(native.metadata), admission);
        }
    }
    auto metadata = geometry_detail::native_constructor_storage(admission);
    auto scratch_storage = geometry_detail::native_constructor_storage(admission);
    Cursor view{descriptor->view};
    const auto name = view.text();
    std::string owned_name;
    metadata.owner->string(owned_name, name);
    if (descriptor->kind == NativeFragmentDescriptor::Kind::Cartesian)
    {
        const auto plan = NativeFragmentDecodePlan::scan(descriptor);
        Cursor topology{*descriptor->parts[0]};
        std::array<unsigned, 3> counts;
        for (auto& n : counts)
            n = static_cast<unsigned>(topology.value<std::uint64_t>());
        for (unsigned i = 0; i < 3; ++i)
            if (topology.value<std::uint8_t>())
                throw std::invalid_argument("Periodic local Cartesian fragment.");
        std::array<std::string_view, 6> names;
        for (auto& n : names)
            n = topology.text();
        topology.finish();
        std::array<std::span<const real_t>, 3> axes;
        // Packet alignment is not assumed. Copy native scalars into admitted
        // scratch arrays before the final immutable provider copies them.
        std::array<std::vector<real_t>, 3> scratch;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            Cursor input{*descriptor->parts[axis + 1]};
            input.array(scratch[axis], scratch_storage);
            input.finish();
            axes[axis] = scratch[axis];
        }
        view.finish();
        if (!plan.matches(descriptor))
            throw std::invalid_argument("Native descriptor changed during Cartesian decode.");
        auto native = details::NativeCartesianAccess::make(name, axes, names, admission);
        return freeze(std::move(native.region), std::move(native.metadata), admission);
    }
    if (descriptor->kind == NativeFragmentDescriptor::Kind::ConstructedProduct)
    {
        const auto plan = NativeFragmentDecodePlan::scan(descriptor);
        const auto layers = view.value<std::uint64_t>();
        auto topology =
            details::NativeTopologyAccess::decode_base(*descriptor->parts[0], static_cast<unsigned>(layers), admission);
        std::vector<details::NativeTopologyAccess::AliasView> aliases;
        const auto n = view.count(2 * sizeof(std::uint64_t));
        scratch_storage.owner->reserve(aliases, n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto from = view.text(), to = view.text();
            aliases.emplace_back(from, to);
        }
        topology =
            details::NativeTopologyAccess::with_layers(*topology, static_cast<unsigned>(layers), aliases, admission);
        const auto run = view.packet();
        view.finish();
        auto construction = PLIC::details::ExtrudedConstructionAccess::decode_parts(
            topology, *descriptor->parts[1], *descriptor->parts[2], *descriptor->parts[3], run, admission);
        auto geometry = details::NativeTopologyAccess::geometry(topology, std::move(construction), admission);
        if (!plan.matches(descriptor))
            throw std::invalid_argument("Native descriptor changed during Product decode.");
        return freeze(ExtrudedRegion(std::move(owned_name), std::move(topology), std::move(geometry)),
            std::move(metadata), admission);
    }
    if (descriptor->kind != NativeFragmentDescriptor::Kind::LegacyProcedural || descriptor->parts.size() != 1 ||
        !descriptor->parts[0])
        throw std::invalid_argument("Unknown detached native descriptor family.");
    Cursor input{*descriptor->parts[0]};
    const auto tag = input.value<std::uint32_t>();
    if (input.text() != name)
        throw std::invalid_argument("Native descriptor names differ.");
    const auto declared = layout(input);
    view.finish();
    if (tag != 2 && tag != 5 && tag != 8)
        throw std::invalid_argument("Native legacy family requires its admitted detached constructor.");
    std::vector<MeshUtils::Vec3> nodes;
    std::vector<real_t> layers;
    input.array(nodes, scratch_storage);
    input.array(layers, scratch_storage);
    std::vector<std::vector<unsigned>> loops;
    const auto cells = input.count(sizeof(std::uint64_t));
    scratch_storage.owner->reserve(loops, cells);
    loops.resize(cells);
    for (auto& loop : loops)
        input.array(loop, scratch_storage);
    std::vector<SemiStructMeshTopo::BoundaryEdge> boundaries;
    const auto edges = input.count(2 * sizeof(unsigned) + sizeof(std::uint64_t));
    scratch_storage.owner->reserve(boundaries, edges);
    for (std::size_t edge = 0; edge < edges; ++edge)
    {
        const auto first = input.value<unsigned>(), second = input.value<unsigned>();
        std::string label;
        scratch_storage.owner->string(label, input.text());
        boundaries.push_back({first, second, std::move(label)});
    }
    std::vector<details::NativeTopologyAccess::AliasView> aliases;
    const auto names = input.count(2 * sizeof(std::uint64_t));
    scratch_storage.owner->reserve(aliases, names);
    for (std::size_t i = 0; i < names; ++i)
    {
        const auto from = input.text(), to = input.text();
        if (from != to)
            aliases.emplace_back(from, to);
    }
    std::optional<MultiRegionMesh::Region> result;
    if (tag == 2)
    {
        if (!aliases.empty())
            throw std::invalid_argument("Native XY/Z cannot alias boundary names.");
        auto native = details::NativeSemiStructuredAccess::make(nodes, loops, layers, boundaries, admission);
        using Provider = NativeRegionProvider<SemiStructuredXY_Z>;
        auto provider = geometry_detail::NativeAllocationEnvelopeAccess::make<Provider>(admission, std::move(native));
        provider.owner->d_native_allocation = provider.envelope;
        result.emplace(NativeIsoRegion<SemiStructuredXY_Z>(std::move(owned_name), provider.owner, provider.owner));
    }
    else
    {
        if (layers.size() < 2 || layers.size() - 1 > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Invalid legacy extrusion extent.");
        auto base = details::NativeTopologyAccess::make(static_cast<unsigned>(nodes.size()), loops,
            static_cast<unsigned>(layers.size() - 1), boundaries, admission);
        auto topology = details::NativeTopologyAccess::topology(std::move(base), aliases, admission);
        std::shared_ptr<const ExtrudedGeometry> geometry;
        if (tag == 8)
        {
            auto construction = PLIC::details::ExtrudedConstructionAccess::decode(topology, input.packet(), admission);
            geometry = details::NativeTopologyAccess::geometry(topology, std::move(construction), admission);
        }
        else
            geometry = details::NativeTopologyAccess::geometry(topology, nodes, layers, admission);
        result.emplace(ExtrudedRegion(std::move(owned_name), std::move(topology), std::move(geometry)));
    }
    input.finish();
    require_layout(*result, declared);
    return freeze(std::move(*result), std::move(metadata), admission);
}
bool NativeFragmentAccess::origin_bound(const std::shared_ptr<const NativeFragmentDescriptor>& descriptor) noexcept
{
    if (!descriptor || !descriptor->provenance.origin_)
        return false;
    const auto& origin = descriptor->provenance.origin_;
    const auto target = origin->descriptor_.lock();
    return target && target.get() == descriptor.get() && !target.owner_before(descriptor) &&
           !descriptor.owner_before(target) && origin->descriptor_allocation_ &&
           origin->descriptor_allocation_->authentic_owner(origin->descriptor_allocation_) &&
           origin->descriptor_allocation_->matches(descriptor);
}
void NativeFragmentAccess::visit_descriptor(const std::shared_ptr<const NativeFragmentDescriptor>& descriptor,
    const geometry_detail::NativeConstructorOwnerSink& sink,
    PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit identity, GeometryVisit geometry, CoverageVisit coverage)
{
    if (!descriptor)
        return;
    using Kind = geometry_detail::NativeConstructorOwnerKind;
    using DescriptorAccess = geometry_detail::NativeObjectOwnerAccess<NativeFragmentDescriptor>;
    const bool native = DescriptorAccess::admitted(descriptor);
    const bool bound = origin_bound(descriptor);
    const bool transition =
        native  ? sink.semantic(descriptor, Kind::NativeMesh, DescriptorAccess::envelope(descriptor), true)
        : bound ? sink.semantic(descriptor, Kind::NativeMesh, descriptor->provenance.origin_->descriptor_allocation_,
                      true)
                : sink.structural(descriptor, Kind::NativeMesh,
                      sizeof(NativeFragmentDescriptor) + descriptor->view.capacity() +
                          descriptor->parts.capacity() * sizeof(descriptor->parts[0]));
    if (!transition)
        return;
    if (native)
        DescriptorAccess::visit_children(descriptor, sink, true);
    for (const auto& part : descriptor->parts)
        if (part && !AMR::adaptive_detail::NativeAdaptivePayloadAccess::bytes(part, sink))
            sink.edge(part, Kind::ConstructionValues, sizeof(*part) + part->capacity());
    visit_origin(descriptor, sink, identity, geometry, coverage);
}
void NativeFragmentAccess::visit_origin(const std::shared_ptr<const NativeFragmentDescriptor>& descriptor,
    const geometry_detail::NativeConstructorOwnerSink& sink,
    PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit identity, GeometryVisit geometry, CoverageVisit coverage)
{
    if (!descriptor || !descriptor->provenance.origin_)
        return;
    const auto& origin = descriptor->provenance.origin_;
    using Kind = geometry_detail::NativeConstructorOwnerKind;
    geometry_detail::NativeConstructorOwnerSink::require_envelope(origin->allocation_, origin);
    if (!sink.semantic(origin, Kind::NativeMesh, origin->allocation_))
        return;
    sink.fixed(origin->allocation_);
    // Both fields are real strong edges to fixed envelopes. The original
    // descriptor target is weak and is never locked to traverse its children.
    sink.fixed(origin->descriptor_allocation_);
    sink.storage(origin->metadata_);
    visit(origin->fragment_, sink, identity, geometry, coverage);
}
void NativeFragmentAccess::visit(const std::shared_ptr<const NativeFragmentRecord>& record,
    const geometry_detail::NativeConstructorOwnerSink& sink,
    PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit identity, GeometryVisit geometry, CoverageVisit coverage)
{
    if (!record)
        return;
    using Kind = geometry_detail::NativeConstructorOwnerKind;
    geometry_detail::NativeConstructorOwnerSink::require_envelope(record->allocation_, record);
    if (!sink.semantic(record, Kind::NativeMesh, record->allocation_))
        return;
    sink.fixed(record->allocation_);
    sink.storage(record->metadata_);
    std::visit(
        [&](const auto& region)
        {
            using R = std::decay_t<decltype(region)>;
            if constexpr (std::is_same_v<R, CartesianRegion>)
            {
                details::NativeCartesianAccess::visit(region.d_topology, sink);
                details::NativeCartesianAccess::visit(region.d_geometry, sink);
            }
            else if constexpr (std::is_same_v<R, ExtrudedRegion>)
            {
                details::NativeProviderOwnerAccess::topology(region.d_topology, sink);
                details::NativeProviderOwnerAccess::geometry(region.d_geometry, sink, identity);
            }
            else if constexpr (std::is_same_v<R, NativeIsoRegion<OrthogonalCartesian3D>>)
                details::NativeLegacyCartesianAccess::visit_region(region, sink);
            else if constexpr (std::is_same_v<R, NativeIsoRegion<SemiStructuredXY_Z>>)
            {
                details::NativeProviderOwnerAccess::native(region.d_topology->d_mesh, sink);
                details::NativeProviderOwnerAccess::native(region.d_geometry->d_mesh, sink);
            }
            else if constexpr (std::is_same_v<R, DistributedExplicitRegion>)
            {
                const bool topology =
                    AMR::adaptive_detail::NativeSparseAccess::visit_source_provider(region.d_topology, sink);
                const bool physical =
                    AMR::adaptive_detail::NativeSparseAccess::visit_source_provider(region.d_geometry, sink);
                if (topology || physical)
                {
                    if (!geometry)
                        throw std::invalid_argument("Native explicit source projection lacks its typed geometry edge.");
                    if (const auto& token = AMR::adaptive_detail::NativeSparseAccess::source_provider_coverage(region.d_topology))
                    {
                        if (!coverage) throw std::invalid_argument("Native source fragment lacks subset coverage projection.");
                        coverage(sink.context, token);
                    }
                    geometry(sink.context,
                        AMR::adaptive_detail::NativeSparseAccess::source_provider_geometry(region.d_topology));
                }
            }
            else
                throw std::invalid_argument("Unqualified detached fragment owner projection.");
        },
        record->region_);
}
} // namespace SimpleFluid::Meshes
