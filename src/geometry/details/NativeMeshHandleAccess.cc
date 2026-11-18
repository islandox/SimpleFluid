#include "geometry/details/NativeMeshHandleAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include <bits/hashtable_policy.h>
namespace SimpleFluid::geometry_detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<>;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;
using Meshes::AMR::adaptive_detail::collective;
using BatchNode = std::__detail::_Hash_node<std::pair<const int, Handle::BoundaryFaceBatch>, false>;
using NameNode = std::__detail::_Hash_node<std::pair<const int, std::string>, false>;
std::size_t boundary_bound(std::size_t n)
{
    if (n > SIZE_MAX / (sizeof(BatchNode) + sizeof(NameNode)))
        throw std::invalid_argument("Native boundary node extent overflows.");
    if (!n) return 0;
    const auto buckets = std::__detail::_Prime_rehash_policy{}._M_next_bkt(n);
    if (buckets > SIZE_MAX / (2 * sizeof(void*)))
        throw std::invalid_argument("Native boundary bucket extent overflows.");
    const auto nodes = n * (sizeof(BatchNode) + sizeof(NameNode));
    const auto index = 2 * buckets * sizeof(void*);
    if (index > SIZE_MAX - nodes)
        throw std::invalid_argument("Native boundary bookkeeping extent overflows.");
    return nodes + index;
}
} // namespace
std::shared_ptr<const Handle> NativeMeshHandleAccess::make(
    const NativePartitionAccess::Result& accepted, const NativeConstructionAdmission& admission)
{
    const auto& partition = accepted.partition;
    if (!partition) throw std::invalid_argument("Native handle lacks an admitted partition.");
    const auto& plan = partition->plan();
    const auto& comm = plan.communicator();
    const auto& geometry = partition->geometry();
    std::shared_ptr<Handle> handle;
    NativeConstructorVector<GO> cells(admission.allocator<GO>()), owned_faces(admission.allocator<GO>()),
        faces(admission.allocator<GO>()), boundaries(admission.allocator<GO>());
    std::array<std::shared_ptr<NativeTpetraMapAdmission::Prepared>, 4> prepared;
    std::shared_ptr<detail::MeshBinding<Pack>> binding;
    collective(*comm, [&]
    {
        if (!partition->geometry_unchanged() || !plan.matches_source(geometry) || accepted.actual_owned_map.is_null() ||
            accepted.actual_owned_map->getComm().getRawPtr() != comm.getRawPtr() ||
            accepted.actual_owned_map->getLocalNumElements() != plan.owned_cells().size() ||
            accepted.actual_owned_map->getGlobalNumElements() != plan.global_cells())
            throw std::invalid_argument("Native handle input differs from its admitted actual map/geometry.");
        auto owner = NativeAllocationEnvelopeAccess::make<Handle>(admission, Handle::NativePreparedTag{});
        handle = std::move(owner.owner);
        handle->d_native_allocation = std::move(owner.envelope);
        handle->d_native_storage = native_constructor_storage(admission);
        handle->d_mesh = Handle::MultiRegionPtr(geometry);
        handle->d_mutable_mesh.emplace(geometry);
        auto& storage = *handle->d_native_storage.owner;
        storage.reserve(handle->d_node_global_ids, plan.node_ids().size());
        for (const auto id : plan.node_ids()) handle->d_node_global_ids.push_back(static_cast<GO>(id));
        cells.reserve(plan.cell_ids().size());
        faces.reserve(plan.face_ids().size());
        owned_faces.reserve(plan.owned_faces().size());
        boundaries.reserve(plan.owned_faces().size());
        for (const auto id : plan.cell_ids()) cells.push_back(static_cast<GO>(id));
        for (const auto id : plan.face_ids()) faces.push_back(static_cast<GO>(id));
        for (const auto id : plan.owned_faces()) owned_faces.push_back(static_cast<GO>(id));
        // Boundary names/containers are rich bookkeeping, independently admitted.
        NativeConstructorVector<std::pair<int, std::size_t>> counts(admission.allocator<std::pair<int, std::size_t>>());
        counts.reserve(faces.size());
        for (std::size_t i = 0; i < faces.size(); ++i)
        {
            const int id = geometry->boundary_id(geometry->face_id(faces[i]));
            if (id == Handle::invalid_boundary_id) continue;
            counts.push_back({id, i});
            if (i < owned_faces.size()) boundaries.push_back(faces[i]);
        }
        std::sort(counts.begin(), counts.end());
        std::size_t batch_count = 0;
        for (std::size_t i = 0; i < counts.size(); ++i)
            batch_count += i == 0 || counts[i - 1].first != counts[i].first;
        auto phase = admission.reserve_profile({boundary_bound(batch_count)});
        auto boundary_profile = NativeAllocationEnvelopeAccess::make<NativeRetainedExtent>(
            admission, phase.retain_profile(phase.bytes()));
        handle->d_native_boundary_profile = std::move(boundary_profile.owner);
        handle->d_native_boundary_allocation = std::move(boundary_profile.envelope);
        if (batch_count)
        {
            handle->d_boundary_names.reserve(batch_count);
            handle->d_boundary_batches.reserve(batch_count);
        }
        for (std::size_t i = 0; i < counts.size();)
        {
            auto end = i + 1;
            while (end < counts.size() && counts[end].first == counts[i].first) ++end;
            const auto id = counts[i].first;
            std::string name;
            storage.string(name, geometry->boundary_batch_name(id));
            handle->d_boundary_names.emplace(id, std::move(name));
            auto& batch = handle->d_boundary_batches[id];
            batch.id = id;
            storage.reserve(batch.face_lids, end - i);
            for (auto k = i; k < end; ++k) batch.face_lids.push_back(static_cast<LO>(counts[k].second));
            i = end;
        }
        const std::array<std::span<const GO>, 4> gids{cells, owned_faces, faces, boundaries};
        for (std::size_t i = 0; i < gids.size(); ++i)
            prepared[i] = NativeTpetraMapAdmission::prepare(
                NativeTpetraMapAdmission::plan(NativeTpetraMapAdmission::Kind::Noncontiguous, gids[i]), admission,
                i ? prepared[i - 1] : nullptr);
        auto binding_owner = NativeAllocationEnvelopeAccess::make<detail::MeshBinding<Pack>>(admission);
        binding = std::move(binding_owner.owner);
        handle->d_native_binding_allocation = std::move(binding_owner.envelope);
    });
    // All four profiles/wrappers/receipts are allocated and admitted on every
    // rank before entering any of their map constructors.
    const std::array<std::span<const GO>, 4> gids{cells, owned_faces, faces, boundaries};
    std::array<Teuchos::RCP<const Pack::map_type>, 4> maps;
    for (std::size_t i = 0; i < maps.size(); ++i)
    {
        auto actual = NativeTpetraMapAdmission::construct(*prepared[i], gids[i], accepted.global_map_counts[i + 1], comm);
        maps[i] = std::move(actual.map);
        collective(*comm, [&]
        {
            if (maps[i]->getLocalNumElements() != gids[i].size() ||
                maps[i]->getGlobalNumElements() != accepted.global_map_counts[i + 1])
                throw std::invalid_argument("Actual native map extent differs from its sealed plan.");
        });
    }
    collective(*comm, [&]
    {
        handle->d_owned_cell_map = accepted.actual_owned_map;
        handle->d_overlap_cell_map = maps[0];
        handle->d_owned_face_map = maps[1];
        handle->d_overlap_face_map = maps[2];
        handle->d_boundary_face_map = maps[3];
        handle->d_map_indexing = true;
        binding->geometry_identity = geometry.get();
        binding->owned_cell_map = handle->d_owned_cell_map.getRawPtr();
        binding->overlap_cell_map = maps[0].getRawPtr();
        binding->owned_face_map = maps[1].getRawPtr();
        binding->overlap_face_map = maps[2].getRawPtr();
        binding->boundary_face_map = maps[3].getRawPtr();
        binding->num_owned_cells = binding->owned_cell_map_size = plan.owned_cells().size();
        binding->num_local_cells = binding->overlap_cell_map_size = cells.size();
        binding->num_owned_faces = binding->owned_face_map_size = owned_faces.size();
        binding->num_local_faces = binding->overlap_face_map_size = faces.size();
        binding->boundary_face_map_size = boundaries.size();
        binding->validate();
        handle->d_binding = binding;
    });
    return handle;
}
std::size_t NativeMeshHandleAccess::library_upper(const std::shared_ptr<const Handle>& handle)
{
    std::size_t total = handle->d_native_boundary_profile->bytes();
    for (const auto& map : {handle->d_owned_cell_map, handle->d_overlap_cell_map, handle->d_owned_face_map,
             handle->d_overlap_face_map, handle->d_boundary_face_map})
    {
        const auto bytes = NativeTpetraMapAdmission::retained_upper(map);
        if (bytes > SIZE_MAX - total) throw std::invalid_argument("Native library upper sum overflows.");
        total += bytes;
    }
    return total + NativeTpetraMapAdmission::library_context_bytes();
}
std::size_t NativeMeshHandleAccess::scalar_bytes(const std::shared_ptr<const Handle>& handle)
{
    std::size_t total = handle->d_node_global_ids.capacity() * sizeof(GO);
    for (const auto& [id, batch] : handle->d_boundary_batches)
    {
        (void) id;
        const auto bytes = batch.face_lids.capacity() * sizeof(LO);
        if (bytes > SIZE_MAX - total) throw std::invalid_argument("Native scalar boundary extent overflows.");
        total += bytes;
    }
    return total;
}
void NativeMeshHandleAccess::visit(const std::shared_ptr<const Handle>& handle,
    const NativeConstructorOwnerSink& sink, PLIC::details::GeometryOwnerGraphEditor& graph,
    bool retain, PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!handle || !sink.semantic(handle, NativeConstructorOwnerKind::NativeMesh, handle->d_native_allocation)) return;
    sink.fixed(handle->d_native_allocation);
    sink.storage(handle->d_native_storage);
    sink.fixed(handle->d_native_boundary_allocation);
    if (sink.semantic(handle->d_native_boundary_profile, NativeConstructorOwnerKind::Storage,
            handle->d_native_boundary_allocation))
        graph.native_state(handle->d_native_boundary_profile->state(), retain, bool(accounting));
    if (handle->d_native_boundary_profile->bytes())
    {
        const auto alias = std::shared_ptr<const void>(handle->d_native_boundary_profile, &handle->d_native_boundary_marker);
        if (accounting)
            accounting->authorize_profile_owner(0, alias, handle->d_native_boundary_profile,
                handle->d_native_boundary_allocation);
        graph.external(alias, handle->d_native_boundary_profile->bytes(), retain, bool(accounting));
    }
    const auto geometry = std::get<Handle::MultiRegionPtr>(handle->d_mesh);
    Meshes::AMR::adaptive_detail::NativeSparseAccess::visit_geometry(geometry, sink, graph, retain, accounting);
    if (handle->d_mutable_mesh)
        Meshes::AMR::adaptive_detail::NativeSparseAccess::visit_geometry(geometry, sink, graph, retain, accounting);
    if (sink.semantic(handle->d_binding, NativeConstructorOwnerKind::Topology, handle->d_native_binding_allocation))
        sink.fixed(handle->d_native_binding_allocation);
    NativeTpetraMapAdmission::visit(handle->d_owned_cell_map, graph, retain, accounting);
    NativeTpetraMapAdmission::visit(handle->d_overlap_cell_map, graph, retain, accounting);
    NativeTpetraMapAdmission::visit(handle->d_owned_face_map, graph, retain, accounting);
    NativeTpetraMapAdmission::visit(handle->d_overlap_face_map, graph, retain, accounting);
    NativeTpetraMapAdmission::visit(handle->d_boundary_face_map, graph, retain, accounting);
}
} // namespace SimpleFluid::geometry_detail
