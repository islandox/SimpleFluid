#pragma once
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/details/NativeTopologyAccess.hh"
#include <cmath>

namespace SimpleFluid::Meshes::details
{
class NativeSemiStructuredAccess
{
    static void copy_loops(SemiStructuredXY_Z::AdmittedStorage& data, const Arr<Arr<unsigned>>& loops)
    {
        data.receipt.owner->reserve(data.loops, loops.size());
        data.loops.resize(loops.size());
        for (size_t i = 0; i < loops.size(); ++i)
            data.receipt.owner->copy(data.loops[i], std::span<const unsigned>(loops[i]));
    }
    static std::shared_ptr<const SemiStructuredXY_Z> publish(SemiStructuredXY_Z::AdmittedStorage&& data,
        const geometry_detail::NativeConstructionAdmission& admission, bool initialize)
    {
        auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<SemiStructuredXY_Z>(
            admission, SemiStructuredXY_Z::AdmittedStorageTag{}, std::move(data));
        tracked.owner->d_allocation_envelope = std::move(tracked.envelope);
        if (initialize) tracked.owner->initialize_xy_geometry();
        return std::move(tracked.owner);
    }
public:
    static std::shared_ptr<const SemiStructuredXY_Z> detached_copy(const SemiStructuredXY_Z& source,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        const auto lease = source.acquire_geometry_read();
        const auto epoch = source.geometry_epoch();
        const auto identity = source.topology().base_storage_identity();
        SemiStructuredXY_Z::AdmittedStorage data(geometry_detail::native_constructor_storage(admission),
            NativeTopologyAccess::detach(source.d_topology, admission));
        auto& receipt = *data.receipt.owner;
        receipt.copy(data.nodes, std::span<const MeshUtils::Vec3>(source.d_xy_nodes));
        copy_loops(data, source.d_xy_cell_nodes);
        receipt.copy(data.z, std::span<const real_t>(source.d_z_edges));
        receipt.copy(data.widths, std::span<const real_t>(source.d_z_widths));
        receipt.copy(data.midpoints, std::span<const real_t>(source.d_z_midpoints));
        receipt.copy(data.areas, std::span<const real_t>(source.d_xy_cell_areas));
        receipt.copy(data.cell_centers, std::span<const MeshUtils::Vec3>(source.d_xy_cell_centroids));
        receipt.copy(data.lengths, std::span<const real_t>(source.d_xy_edge_lengths));
        receipt.copy(data.edge_centers, std::span<const MeshUtils::Vec3>(source.d_xy_edge_centroids));
        receipt.copy(data.edge_normals, std::span<const MeshUtils::Vec3>(source.d_xy_edge_normals));
        if (source.geometry_epoch() != epoch || source.topology().base_storage_identity() != identity)
            throw std::invalid_argument("Native source changed during detached geometry copy.");
        // Fresh GeometryExecutionGuard/GeometryEpochState; original metric bits.
        return publish(std::move(data), admission, false);
    }
    static std::shared_ptr<const SemiStructuredXY_Z> make(std::span<const MeshUtils::Vec3> nodes,
        const Arr<Arr<unsigned>>& loops, std::span<const real_t> z,
        const Arr<SemiStructMeshTopo::BoundaryEdge>& boundaries,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        if (nodes.empty() || loops.empty() || z.size() < 2)
            throw std::invalid_argument("SemiStructuredXY_Z requires nonempty XY and at least two Z edges.");
        if (nodes.size() > std::numeric_limits<unsigned>::max() ||
            loops.size() > std::numeric_limits<unsigned>::max() || z.size() > std::numeric_limits<unsigned>::max())
            throw std::overflow_error("SemiStructuredXY_Z entity count exceeds its ID type.");
        for (const auto node : nodes)
            if (!std::isfinite(node.x) || !std::isfinite(node.y))
                throw std::invalid_argument("SemiStructuredXY_Z contains a non-finite XY coordinate.");
        for (size_t i = 0; i < z.size(); ++i)
            if (!std::isfinite(z[i]) || (i && z[i] <= z[i - 1]))
                throw std::invalid_argument("SemiStructuredXY_Z Z coordinates must be finite and strictly increasing.");
        SemiStructuredXY_Z::AdmittedStorage data(geometry_detail::native_constructor_storage(admission),
            NativeTopologyAccess::make(static_cast<unsigned>(nodes.size()), loops,
                static_cast<unsigned>(z.size() - 1), boundaries, admission));
        auto& receipt = *data.receipt.owner;
        receipt.copy(data.nodes, nodes); copy_loops(data, loops); receipt.copy(data.z, z);
        receipt.reserve(data.widths, z.size() - 1); receipt.reserve(data.midpoints, z.size() - 1);
        for (size_t i = 0; i + 1 < z.size(); ++i)
        {
            data.widths.push_back(z[i + 1] - z[i]);
            data.midpoints.push_back((z[i] + z[i + 1]) / static_cast<real_t>(2));
        }
        // initialize_xy_geometry keeps exactly the original expressions and predicates.
        const auto edges = data.topology.side_faces().size();
        receipt.reserve(data.areas, loops.size()); receipt.reserve(data.cell_centers, loops.size());
        receipt.reserve(data.lengths, edges); receipt.reserve(data.edge_centers, edges);
        receipt.reserve(data.edge_normals, edges);
        return publish(std::move(data), admission, true);
    }
};
} // namespace SimpleFluid::Meshes::details
