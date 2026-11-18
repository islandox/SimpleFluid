/** @file NativeCartesianAccess.hh
 * @brief Admitted immutable rectilinear storage and exact owner projection.
 */
#pragma once
#include "geometry/details/NativeConstructorOwnerSink.hh"
#include "geometry/mesh/RegionProviders.hh"
#include <span>
#include <string_view>

namespace SimpleFluid::Meshes::details
{
struct NativeCartesianRegion
{
    geometry_detail::NativeConstructorStorageOwner metadata;
    CartesianRegion region;
};
class NativeCartesianAccess
{
public:
    using Admission = geometry_detail::NativeConstructionAdmission;
    using Sink = geometry_detail::NativeConstructorOwnerSink;
    static std::shared_ptr<const RectilinearTopology> topology(
        std::array<unsigned, 3> counts, const std::array<std::string_view, 6>& names, const Admission& admission)
    {
        auto storage = geometry_detail::native_constructor_storage(admission);
        OrthoMeshTopo::BoundaryNames owned_names;
        for (std::size_t boundary = 0; boundary != 6; ++boundary)
            storage.owner->string(owned_names[boundary], names[boundary]);
        const OrthoMeshTopo::Indexer indexer(counts[0], counts[1], counts[2], false, false, false);
        auto native = geometry_detail::NativeAllocationEnvelopeAccess::make<OrthoMeshTopo>(
            admission, OrthoMeshTopo::AdmittedStorageTag{}, indexer, std::move(owned_names), std::move(storage));
        native.owner->d_allocation_envelope = native.envelope;
        auto wrapped = geometry_detail::NativeAllocationEnvelopeAccess::make<RectilinearTopology>(
            admission, std::move(native.owner));
        wrapped.owner->d_allocation_envelope = wrapped.envelope;
        return std::move(wrapped.owner);
    }
    static std::shared_ptr<const RectilinearGeometry> geometry(
        const std::array<std::span<const real_t>, 3>& axes, const Admission& admission)
    {
        auto storage = geometry_detail::native_constructor_storage(admission);
        Vec3D<ArrReal> owned;
        for (std::size_t axis = 0; axis != 3; ++axis)
            storage.owner->copy(owned[axis], axes[axis]);
        auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<RectilinearGeometry>(
            admission, RectilinearGeometry::AdmittedStorageTag{}, std::move(owned), std::move(storage));
        tracked.owner->d_allocation_envelope = tracked.envelope;
        return std::move(tracked.owner);
    }
    static NativeCartesianRegion make(std::string_view name, const std::array<std::span<const real_t>, 3>& axes,
        const std::array<std::string_view, 6>& names, const Admission& admission)
    {
        auto physical = geometry(axes, admission);
        const auto extents = physical->layout().extents;
        auto connectivity = topology(
            {static_cast<unsigned>(extents[0]), static_cast<unsigned>(extents[1]), static_cast<unsigned>(extents[2])},
            names, admission);
        auto metadata = geometry_detail::native_constructor_storage(admission);
        std::string owned_name;
        metadata.owner->string(owned_name, name);
        return {
            std::move(metadata), CartesianRegion(std::move(owned_name), std::move(connectivity), std::move(physical))};
    }
    static NativeCartesianRegion detach(const CartesianRegion& source, const Admission& admission)
    {
        const auto topology_lease = source.topology().acquire_geometry_read();
        const auto native_lease = source.topology().native_topology().acquire_geometry_read();
        source.validate_static();
        std::array<std::span<const real_t>, 3> axes;
        std::array<std::string_view, 6> names;
        for (std::size_t axis = 0; axis != 3; ++axis)
            axes[axis] = source.geometry().cell_edges()[axis];
        for (std::size_t boundary = 0; boundary != 6; ++boundary)
            names[boundary] = source.topology().boundary_batch_name(static_cast<int>(boundary));
        auto result = make(source.name(), axes, names, admission);
        source.validate_static();
        return result;
    }
    static void visit(const std::shared_ptr<const RectilinearTopology>& owner, const Sink& sink)
    {
        using Kind = geometry_detail::NativeConstructorOwnerKind;
        if (!owner)
            return;
        const auto& native = owner->d_topology;
        if (!owner->d_allocation_envelope || !native->d_allocation_envelope || !native->d_storage.owner)
            throw std::invalid_argument("Legacy Cartesian topology requires admitted detach.");
        Sink::require_envelope(owner->d_allocation_envelope, owner);
        Sink::require_envelope(native->d_allocation_envelope, native);
        if (!sink.semantic(owner, Kind::Topology, owner->d_allocation_envelope))
            return;
        sink.fixed(owner->d_allocation_envelope);
        if (sink.semantic(native, Kind::TopologyBase, native->d_allocation_envelope))
        {
            sink.fixed(native->d_allocation_envelope);
            sink.storage(native->d_storage);
        }
    }
    static void visit(const std::shared_ptr<const RectilinearGeometry>& owner, const Sink& sink)
    {
        using Kind = geometry_detail::NativeConstructorOwnerKind;
        if (!owner)
            return;
        if (!owner->d_allocation_envelope || !owner->d_storage.owner)
            throw std::invalid_argument("Legacy Cartesian geometry requires admitted detach.");
        Sink::require_envelope(owner->d_allocation_envelope, owner);
        if (!sink.semantic(owner, Kind::Geometry, owner->d_allocation_envelope))
            return;
        sink.fixed(owner->d_allocation_envelope);
        sink.storage(owner->d_storage);
    }
};
} // namespace SimpleFluid::Meshes::details
