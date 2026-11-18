#include "parallel/details/NativePartitionAccess.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include "parallel/NativePacketAdmission.hh"
#include <algorithm>
namespace SimpleFluid::geometry_detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using ID = std::uint64_t;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;
using Meshes::AMR::adaptive_detail::collective;
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > SIZE_MAX - a)
        throw std::invalid_argument("Native partition extent overflows.");
    return a + b;
}
void check_rows(std::size_t n)
{
    if (n > std::size_t(std::numeric_limits<LO>::max()))
        throw std::invalid_argument("Native partition row count exceeds local ordinal range.");
}
void check_gid(ID gid)
{
    if (gid > ID(std::numeric_limits<GO>::max()))
        throw std::invalid_argument("Native partition GID exceeds global ordinal range.");
}
template<class T> void sort_unique(T& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}
} // namespace
NativePartitionAccess::Result NativePartitionAccess::make(
    Meshes::AMR::adaptive_detail::NativePublicationSession& session, std::shared_ptr<Meshes::MultiRegionMesh> geometry,
    std::span<const ID> owned, std::span<const ID> ghosts)
{
    const auto& input = session.inputs();
    const auto& a = session.admission();
    const auto& comm = input.comm;
    CompositePartitionPlan<> plan;
    NativeConstructorVector<ID> sorted(a.allocator<ID>()), faces(a.allocator<ID>()), nodes(a.allocator<ID>());
    NativeConstructorVector<GO> owned_gids(a.allocator<GO>());
    std::shared_ptr<NativeTpetraMapAdmission::Prepared> prepared;
    std::shared_ptr<CompositePartition<>> partition;
    std::shared_ptr<const NativeAllocationEnvelope> allocation;
    std::array<ID, 5> local_counts{};
    collective(*comm,
        [&]
        {
            if (!session.qualified() || !geometry || !geometry->sparse_native() ||
                geometry->communicator().getRawPtr() != comm.getRawPtr())
                throw std::invalid_argument("Native partition requires this Session's current sparse geometry.");
            const auto provenance = Meshes::AMR::adaptive_detail::NativeSparseAccess::output_provenance(geometry);
            if (!provenance || provenance->lineage != input.lineage || provenance->publication != session.issuance_)
                throw std::invalid_argument("Native partition has foreign sparse publication provenance.");
            geometry->validate_static();
            if (owned.size() != input.layout->size())
                throw std::invalid_argument("Native partition owned count differs from the sealed layout.");
            check_rows(add(owned.size(), ghosts.size()));
            for (const auto count : {geometry->num_cells(), geometry->num_faces(), geometry->num_nodes()})
                if (count > ID(std::numeric_limits<GO>::max()) ||
                    count >= Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid())
                    throw std::invalid_argument("Native partition global count exceeds map range.");
            sorted.reserve(add(owned.size(), ghosts.size()));
            owned_gids.reserve(owned.size());
            for (std::size_t i = 0; i < owned.size(); ++i)
            {
                const auto gid = owned[i];
                check_gid(gid);
                if (gid != input.cell_begin + i || geometry->sparse_cell_owner(gid) != comm->getRank())
                    throw std::invalid_argument(
                        "Native partition owned route differs from current owner confirmation.");
                (void) geometry->native_cell(gid);
                sorted.push_back(gid);
                owned_gids.push_back(static_cast<GO>(gid));
            }
            for (const auto gid : ghosts)
            {
                check_gid(gid);
                const auto owner = geometry->sparse_cell_owner(gid);
                if (owner < 0 || owner >= comm->getSize() || owner == comm->getRank())
                    throw std::invalid_argument("Native partition ghost lacks a current remote owner.");
                (void) geometry->native_cell(gid);
                sorted.push_back(gid);
            }
            std::sort(sorted.begin(), sorted.end());
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
                throw std::invalid_argument("Native partition repeats a current full cell.");
            std::size_t face_count = 0, node_count = 0;
            for (const auto gid : sorted)
            {
                node_count = add(node_count, geometry->cell_nodes(gid).size());
                for (const auto f : geometry->cell_faces(gid))
                {
                    face_count = add(face_count, 1);
                    node_count = add(node_count, geometry->face_nodes(f).size());
                }
            }
            faces.reserve(face_count);
            nodes.reserve(node_count);
            for (const auto gid : sorted)
            {
                for (const auto n : geometry->cell_nodes(gid))
                    nodes.push_back(n);
                for (const auto f : geometry->cell_faces(gid))
                {
                    faces.push_back(f);
                    for (const auto n : geometry->face_nodes(f))
                        nodes.push_back(n);
                }
            }
            sort_unique(faces);
            sort_unique(nodes);
            check_rows(faces.size());
            check_rows(nodes.size());
            for (const auto f : faces)
                check_gid(f);
            for (const auto n : nodes)
                check_gid(n);
            plan.d_native_storage = native_constructor_storage(a);
            auto& storage = *plan.d_native_storage.owner;
            storage.reserve(plan.d_cells, sorted.size());
            storage.reserve(plan.d_cell_owners, sorted.size());
            storage.reserve(plan.d_faces, faces.size());
            storage.copy(plan.d_nodes, std::span<const ID>(nodes));
            plan.d_cells.insert(plan.d_cells.end(), owned.begin(), owned.end());
            plan.d_cells.insert(plan.d_cells.end(), ghosts.begin(), ghosts.end());
            for (const auto gid : plan.d_cells)
                plan.d_cell_owners.push_back(geometry->sparse_cell_owner(gid));
            plan.d_owned_cells = owned.size();
            for (const auto f : faces)
                if (geometry->sparse_cell_owner(geometry->owner_cell(f)) == comm->getRank())
                    plan.d_faces.push_back(f);
            plan.d_owned_faces = plan.d_faces.size();
            for (const auto f : faces)
                if (geometry->sparse_cell_owner(geometry->owner_cell(f)) != comm->getRank())
                    plan.d_faces.push_back(f);
            plan.d_global_cells = geometry->num_cells();
            plan.d_global_faces = geometry->num_faces();
            plan.d_global_nodes = geometry->num_nodes();
            plan.d_collective_id = session.issuance_;
            plan.d_comm = comm;
            plan.d_source = geometry;
            plan.d_source_epoch = geometry->geometry_epoch();
            plan.d_source_mode = CompositeMeshSource::Mode::Distributed;
            local_counts = {owned.size(), plan.d_cells.size(), plan.d_owned_faces, plan.d_faces.size(), 0};
            for (std::size_t i = 0; i < plan.d_owned_faces; ++i)
                local_counts[4] += geometry->boundary_id(geometry->face_id(plan.d_faces[i])) != -1;
            prepared = NativeTpetraMapAdmission::prepare(
                NativeTpetraMapAdmission::plan(NativeTpetraMapAdmission::Kind::ContiguousOwned, owned_gids), a);
        });
    std::array<ID, 5> global_counts{};
    ID local = owned.size(), prefix = 0, maximum = 0;
    owner_packet_partition_counts(*comm, local_counts, global_counts, prefix, maximum);
    const auto sum = global_counts[0];
    collective(*comm,
        [&]
        {
            if (sum != input.global_cells || prefix < local || prefix - local != input.cell_begin ||
                (sum == 0 && (plan.d_global_faces || plan.d_global_nodes || !plan.d_cells.empty() ||
                                 !plan.d_faces.empty() || !plan.d_nodes.empty())))
                throw std::invalid_argument(
                    "Owner-confirmed native offsets/counts do not partition the source domain.");
            plan.d_balance.total_cells = sum;
            plan.d_balance.max_owned_cells = maximum;
            plan.d_balance.measured_imbalance = sum ? double(maximum) * comm->getSize() / double(sum) : 1;
            // Allocate the actual wrapper before its first library collective.
            auto wrapper = NativeAllocationEnvelopeAccess::make<CompositePartition<>>(
                a, CompositePartition<>::NativePreparedKey{}, geometry, std::move(plan));
            partition = std::move(wrapper.owner);
            allocation = std::move(wrapper.envelope);
            partition->d_native_allocation = allocation;
        });
    auto accepted_map = NativeTpetraMapAdmission::construct(*prepared, owned_gids, sum, comm);
    collective(*comm,
        [&]
        {
            if (accepted_map.map->getGlobalNumElements() != sum || accepted_map.map->getLocalNumElements() != local ||
                accepted_map.map->getIndexBase() != GO{} ||
                (local && accepted_map.map->getMinGlobalIndex() != static_cast<GO>(input.cell_begin)))
                throw std::invalid_argument("Actual admitted owned map differs from its exact ownership receipt.");
        });
    return {std::move(partition), std::move(accepted_map.map), std::move(allocation), global_counts};
}
std::shared_ptr<const NativeAllocationEnvelope> NativePartitionAccess::allocation(
    const std::shared_ptr<const CompositePartition<>>& partition) noexcept
{
    if (!partition) return {};
    const auto& value = partition->d_native_allocation;
    return value && value->authentic_owner(value) && value->matches(partition) ? value : nullptr;
}
std::shared_ptr<const NativeAllocationEnvelope> NativePartitionAccess::checked_allocation(
    const std::shared_ptr<const CompositePartition<>>& partition)
{
    const auto value = allocation(partition);
    if (!value && partition && partition->d_native_allocation)
        throw std::invalid_argument("Native binding has a foreign partition shared control.");
    return value;
}
void NativePartitionAccess::visit(const std::shared_ptr<const CompositePartition<>>& partition,
    const NativeConstructorOwnerSink& sink, PLIC::details::GeometryOwnerGraphEditor& graph, bool retain,
    PLIC::details::NativeGeometryAccounting* accounting)
{
    if (!partition)
        return;
    const auto& envelope = partition->d_native_allocation;
    if (sink.semantic(partition, NativeConstructorOwnerKind::Topology, envelope))
    {
        sink.fixed(envelope);
        sink.storage(partition->d_plan.d_native_storage);
        Meshes::AMR::adaptive_detail::NativeSparseAccess::visit_geometry(
            partition->d_geometry, sink, graph, retain, accounting);
    }
}
} // namespace SimpleFluid::geometry_detail
