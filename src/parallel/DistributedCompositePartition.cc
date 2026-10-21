/** @file DistributedCompositePartition.cc
 * @brief Direct authority-verified sparse shards into native field/halo maps.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "parallel/CompositePartition.hh"
#include <Tpetra_Map.hpp>
#include <atomic>
#include <set>
namespace SimpleFluid
{
template<TpetraTypePack Pack>
CompositePartition<Pack> CompositePartition<Pack>::from_sparse_native(std::shared_ptr<Meshes::MultiRegionMesh> geometry,
    std::span<const uint64_t> owned, std::span<const uint64_t> ghosts)
{
    if (!geometry)
        throw std::invalid_argument("Sparse partition requires geometry.");
    const auto comm = geometry->communicator();
    if (comm.is_null())
        throw std::invalid_argument("Sparse partition requires communicator.");
    using GO = typename Pack::global_ordinal_type;
    using LO = typename Pack::local_ordinal_type;
    CompositePartitionPlan<Pack> plan;
    SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (!geometry->sparse_native())
                throw std::invalid_argument("Direct distributed partition requires sparse native geometry.");
            geometry->validate_static();
            // Counts and GIDs are separate contracts: a sparse rank may carry
            // very high entity IDs without allocating the global domain.
            const auto checked_rows = [](std::size_t n)
            {
                if (n > static_cast<std::size_t>(std::numeric_limits<LO>::max()))
                    throw std::invalid_argument("Sparse native local row count exceeds local ordinal range.");
            };
            checked_rows(owned.size());
            checked_rows(ghosts.size());
            if (ghosts.size() > std::numeric_limits<std::size_t>::max() - owned.size())
                throw std::invalid_argument("Sparse native local row count overflows.");
            checked_rows(owned.size() + ghosts.size());
            for (const auto count : {geometry->num_cells(), geometry->num_faces(), geometry->num_nodes()})
                if (count > static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) ||
                    count >= Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid())
                    throw std::invalid_argument("Sparse native global count exceeds typed map range.");
            std::set<uint64_t> unique;
            for (auto cell : owned)
            {
                if (cell > static_cast<uint64_t>(std::numeric_limits<GO>::max()) || !unique.insert(cell).second ||
                    geometry->sparse_cell_owner(cell) != comm->getRank())
                    throw std::invalid_argument("Invalid sparse partition owned cell.");
                (void) geometry->native_cell(cell);
                plan.d_cells.push_back(cell);
                plan.d_cell_owners.push_back(comm->getRank());
            }
            plan.d_owned_cells = owned.size();
            for (auto cell : ghosts)
            {
                const auto owner = geometry->sparse_cell_owner(cell);
                if (cell > static_cast<uint64_t>(std::numeric_limits<GO>::max()) || !unique.insert(cell).second ||
                    owner == comm->getRank())
                    throw std::invalid_argument("Invalid sparse partition ghost cell.");
                (void) geometry->native_cell(cell);
                plan.d_cells.push_back(cell);
                plan.d_cell_owners.push_back(owner);
            }
            plan.d_global_cells = geometry->num_cells();
            plan.d_global_faces = geometry->num_faces();
            plan.d_global_nodes = geometry->num_nodes();
            plan.d_comm = comm;
            plan.d_source = geometry;
            plan.d_source_epoch = geometry->geometry_epoch();
            plan.d_source_mode = CompositeMeshSource::Mode::Distributed;
            std::set<uint64_t> faces, nodes;
            for (auto cell : plan.d_cells)
            {
                for (auto node : geometry->cell_nodes(cell))
                    nodes.insert(node);
                for (auto face : geometry->cell_faces(cell))
                {
                    faces.insert(face);
                    for (auto node : geometry->face_nodes(face))
                        nodes.insert(node);
                }
            }
            checked_rows(faces.size());
            checked_rows(nodes.size());
            for (auto face : faces)
                if (face > static_cast<std::uint64_t>(std::numeric_limits<GO>::max()))
                    throw std::invalid_argument("Sparse native face GID exceeds global ordinal range.");
            for (auto node : nodes)
                if (node > static_cast<std::uint64_t>(std::numeric_limits<GO>::max()))
                    throw std::invalid_argument("Sparse native node GID exceeds global ordinal range.");
            for (auto face : faces)
                if (geometry->sparse_cell_owner(geometry->owner_cell(face)) == comm->getRank())
                    plan.d_faces.push_back(face);
            plan.d_owned_faces = plan.d_faces.size();
            for (auto face : faces)
                if (geometry->sparse_cell_owner(geometry->owner_cell(face)) != comm->getRank())
                    plan.d_faces.push_back(face);
            plan.d_nodes.assign(nodes.begin(), nodes.end());
        });
    std::vector<GO> gids(owned.begin(), owned.end());
    const auto map = Teuchos::rcp(new typename Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::arrayViewFromVector(gids), GO{}, comm));
    // Tpetra may classify a globally empty map as replicated on multiple
    // ranks. Its ownership is vacuously unique only for an entirely empty
    // utility domain; every nonempty domain still uses the full map check.
    const bool empty_domain = plan.d_global_cells == 0;
    const bool unique = empty_domain || map->isOneToOne();
    SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (!unique || map->getGlobalNumElements() != plan.d_global_cells ||
                (empty_domain && (plan.d_global_faces != 0 || plan.d_global_nodes != 0 || !plan.d_cells.empty() ||
                                     !plan.d_faces.empty() || !plan.d_nodes.empty())))
                throw std::invalid_argument("Sparse native owned maps do not partition the global cell domain.");
        });
    std::vector<GO> ghost_ids(ghosts.begin(), ghosts.end());
    std::vector<int> owners(ghosts.size(), -1);
    if (!empty_domain)
        (void) map->getRemoteIndexList(Teuchos::arrayViewFromVector(ghost_ids), Teuchos::arrayViewFromVector(owners));
    SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < owners.size(); ++i)
                if (owners[i] != plan.d_cell_owners[plan.d_owned_cells + i])
                    throw std::invalid_argument("Sparse native ghost owner differs from the actual distributed map.");
        });
    static std::atomic<uint64_t> next{1};
    uint64_t identity = 0;
    if (comm->getRank() == 0)
        identity = next.fetch_add(1);
    Teuchos::broadcast(*comm, 0, 1, &identity);
    plan.d_collective_id = identity;
    uint64_t local = owned.size(), maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &maximum);
    plan.d_balance.total_cells = plan.d_global_cells;
    plan.d_balance.max_owned_cells = maximum;
    plan.d_balance.measured_imbalance =
        plan.d_global_cells ? double(maximum) * comm->getSize() / double(plan.d_global_cells) : 1;
    return CompositePartition(std::move(geometry), std::move(plan));
}
template CompositePartition<DefaultTpetraTypes> CompositePartition<DefaultTpetraTypes>::from_sparse_native(
    std::shared_ptr<Meshes::MultiRegionMesh>, std::span<const uint64_t>, std::span<const uint64_t>);
} // namespace SimpleFluid
