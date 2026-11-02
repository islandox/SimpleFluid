/** @file RetainedStorage.cc
 * @brief Retained topology, geometry and phase-storage admission accounting.
 */
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "FVM/details/GeneralVoF/PhaseCleanup.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
template<ArithmeticMode Mode>
void retained_budget(const GeneralVoF::Data& data, std::optional<std::size_t> proposed_spatial_bytes,
    std::size_t extra_retained_bytes, std::size_t* remaining_bytes)
{
    // Complete the audit before admission so a rejection reports the entire
    // requirement, rather than whichever category first crossed the limit.
    // Shared geometry and typed identity owners are charged once across roots;
    // ledger bookkeeping remains part of the retained admission requirement.
    data.topology->validate_surface_tiles();
    wire::Budget budget{std::numeric_limits<std::size_t>::max()};
    std::array<std::pair<const char*, std::size_t>, 12> categories{};
    std::size_t category_count = 0, checkpoint = 0;
    const auto category = [&](const char* name)
    {
        categories.at(category_count++) = {name, budget.used - checkpoint};
        checkpoint = budget.used;
    };
    std::size_t geometry_root_bytes = 0, key_allowance_bytes = 0, map_link_allowance_bytes = 0;
    std::size_t tile_vector_bytes = 0, tile_metadata_bytes = 0;
    std::size_t tile_owned_vector_capacity_bytes = 0, tile_owned_vector_size_bytes = 0;
    const auto key_vector_bytes = [](const Key& key, bool capacity)
    {
        return std::visit(
            [&](const auto& ancestry) -> std::size_t
            {
                using Ancestry = std::decay_t<decltype(ancestry)>;
                if constexpr (std::is_same_v<Ancestry, AMR::ConvexCellAncestry>)
                    return (capacity ? ancestry.path.capacity() : ancestry.path.size()) * sizeof(std::uint64_t);
                else if constexpr (std::is_same_v<Ancestry, AMR::ProductCellAncestry>)
                    return ((capacity ? ancestry.base.path.capacity() : ancestry.base.path.size()) +
                               (capacity ? ancestry.axial.path.capacity() : ancestry.axial.path.size())) *
                           sizeof(std::uint64_t);
                else
                    return 0;
            },
            key.ancestry);
    };
    const auto key_allowance = [&](const Key& key)
    {
        const auto bytes = AMR::encode_adaptive_key(key).size();
        key_allowance_bytes += bytes;
        budget.add(bytes);
        return bytes;
    };
    const auto links = [&](std::size_t count)
    {
        const auto bytes = count * sizeof(void*);
        map_link_allowance_bytes += bytes;
        budget.add(bytes);
    };
    budget.add(extra_retained_bytes);
    std::set<const void*> geometry;
    budget.add(data.latest_phase_cleanup.capacity() * sizeof(GeneralVoFPhaseCleanupDelta));
    for (const auto& delta : data.latest_phase_cleanup)
        phase_cleanup_key_storage(delta.key, budget);
    budget.add(sizeof(data) + sizeof(GeneralVoF::Data::Topology) + data.topology->order.capacity() * sizeof(Key) +
               (data.gas.capacity() + data.carrier.capacity()) * sizeof(double) +
               data.phases.capacity() * sizeof(PhaseVolumes) +
               (data.roundoff.capacity() + data.phase_history.capacity()) * sizeof(PhaseRoundoff) +
               data.occupancy.capacity() * sizeof(PhaseError));
    if (proposed_spatial_bytes)
        budget.add(*proposed_spatial_bytes);
    else
    {
        std::lock_guard lock(data.topology->spatial_cache_mutex);
        if (data.topology->spatial_cache)
            budget.add(data.topology->spatial_cache->bytes);
    }
    PLIC::GeometryAllocationLedger native_allocations, finite_allocations;
    const auto retain_face = [&](const auto& face)
    {
        const auto before = budget.used;
        budget.add(face->exclusive_storage_bytes());
        native_allocations.retain(PLIC::planar_support<ArithmeticMode::Ordinary>(*face), budget.used, budget.limit);
        geometry_root_bytes += budget.used - before;
    };
    category("state_and_cache");
    std::set<const void*> exact_payloads;
    const auto charge_exact = [&](const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& value)
    {
        if (exact_payloads.insert(value.storage_identity()).second)
            budget.add(value.storage_bytes());
    };
    if (data.experimental_catalog)
        budget.add(sizeof(*data.experimental_catalog) +
                   data.experimental_catalog->capacity() *
                       sizeof(SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>));
    for (const auto& [key, cell] : data.cells)
    {
        budget.add(sizeof(Cell) + sizeof(Key));
        links(8);
        key_allowance(key);
        const auto before_geometry = budget.used;
        if (geometry.insert(cell.phase.geometry.get()).second)
            native_allocations.retain(cell.phase.geometry, budget.used, budget.limit);
        if constexpr (Mode != ArithmeticMode::Exact)
            if (cell.finite_geometry && geometry.insert(cell.finite_geometry.get()).second)
            {
                if constexpr (Mode == ArithmeticMode::Ordinary)
                    native_allocations.retain<Mode>(cell.finite_geometry, budget.used, budget.limit);
                else
                    finite_allocations.retain<Mode>(cell.finite_geometry, budget.used, budget.limit);
            }
        geometry_root_bytes += budget.used - before_geometry;
        if (cell.experimental_exact)
        {
            budget.add(sizeof(ExperimentalPhaseInventory));
            const auto& x = *cell.experimental_exact;
            for (const auto* q :
                {&x.phases.bulk, &x.phases.gas, &x.phases.carrier, &x.error.difference, &x.error.admitted_bound,
                    &x.error.native_conversion, &x.error.phase_input_bound, &x.error.fit_residual, &x.error.fit_bound,
                    &x.error.transferred_bound, &x.error.fit_allocation_bound})
                charge_exact(*q);
        }
    }
    category("cells_and_geometry_roots");
    for (const auto& node : data.topology->nodes)
        budget.add(sizeof(node) + node.identity.capacity() + node.coordinate_identity.payload_bytes());
    if constexpr (Mode == ArithmeticMode::Ordinary)
        native_allocations.retain(std::get<PLIC::PointSnapshot<Mode>>(data.topology->finite_nodes.snapshot),
            budget.used, budget.limit);
    else if constexpr (Mode == ArithmeticMode::BoundedCertification)
        finite_allocations.retain(std::get<PLIC::PointSnapshot<Mode>>(data.topology->finite_nodes.snapshot),
            budget.used, budget.limit);
    for (const auto& [id, slot] : data.topology->finite_nodes.slots)
    {
        budget.add(sizeof(id) + sizeof(slot));
        links(4);
    }
    for (const auto& [id, identity] : data.topology->construction_nodes)
    {
        budget.add(sizeof(id) + sizeof(identity) + identity.capacity());
        links(4);
    }
    for (const auto& [id, face] : data.topology->construction_faces)
    {
        budget.add(sizeof(id) + sizeof(face));
        links(4);
    }
    category("nodes_and_authorities");
    // Tile arrays pay every allocated element slot. Immutable metadata pays
    // actual owner/control/capacity once, including the traversal's owner ledger.
    AMR::adaptive_detail::ImmutableSequenceLedger tile_metadata;
    const auto tiles = [&](const auto& entries)
    {
        const auto allocation = entries.capacity() * sizeof(AMR::AdaptiveTile);
        budget.add(allocation);
        tile_vector_bytes += allocation;
        tile_owned_vector_capacity_bytes += allocation;
        tile_owned_vector_size_bytes += entries.size() * sizeof(AMR::AdaptiveTile);
        for (const auto& tile : entries)
        {
            std::size_t shared_owners = 0;
            const auto shared_sequence = [&](const auto& values)
            {
                const auto owner = tile_metadata.retain(values, budget);
                if (owner)
                {
                    using T = typename std::decay_t<decltype(values)>::value_type;
                    shared_owners += owner;
                    tile_owned_vector_capacity_bytes += values.storage_capacity() * sizeof(T);
                    tile_owned_vector_size_bytes += values.storage_size() * sizeof(T);
                }
            };
            shared_sequence(tile.domain);
            shared_sequence(tile.surface);
            shared_sequence(tile.node_gids);
            shared_sequence(tile.second_node_gids);
            const auto vectors = tile.identity.capacity() + tile.second_image_identity.payload_bytes();
            budget.add(vectors + tile.boundary.capacity());
            tile_metadata_bytes += shared_owners + vectors + tile.boundary.capacity();
            tile_owned_vector_capacity_bytes += vectors;
            tile_owned_vector_size_bytes +=
                tile.identity.size() + tile.second_image_identity.payload_bytes();
            const auto tile_key = [&](const Key& key)
            {
                const auto capacity = key_vector_bytes(key, true);
                tile_owned_vector_capacity_bytes += capacity;
                tile_owned_vector_size_bytes += key_vector_bytes(key, false);
                const auto encoded = key_allowance(key);
                if (capacity > encoded)
                    budget.add(capacity - encoded);
            };
            tile_key(tile.first);
            if (tile.second)
                tile_key(*tile.second);
            if (geometry.insert(tile.geometry.get()).second)
                retain_face(tile.geometry);
        }
    };
    tiles(data.topology->tiles);
    category("owned_tiles");
    tiles(data.topology->extra_surface_tiles);
    const auto reference_bytes =
        data.topology->surface_tile_references.capacity() * sizeof(GeneralVoF::Data::Topology::SurfaceTileReference);
    budget.add(reference_bytes);
    tile_owned_vector_capacity_bytes += reference_bytes;
    tile_owned_vector_size_bytes +=
        data.topology->surface_tile_references.size() * sizeof(GeneralVoF::Data::Topology::SurfaceTileReference);
    category("surface_extras_and_references");
    budget.add(
        data.topology->frontier_tile_references.capacity() * sizeof(GeneralVoF::Data::Topology::SurfaceTileReference));
    category("frontier_references");
    for (const auto& key : data.topology->graph_keys)
    {
        budget.add(sizeof(Key) + key_vector_bytes(key, true));
        links(4);
        key_allowance(key);
    }
    auto graph = [&](const auto& entries)
    {
        for (const auto& [key, values] : entries)
        {
            budget.add(sizeof(key) + sizeof(values));
            links(4);
            if constexpr (std::is_same_v<std::decay_t<decltype(key)>, Key>)
            {
                budget.add(key_vector_bytes(key, true));
                key_allowance(key);
            }
            budget.add(values.storage_bytes());
        }
    };
    graph(data.topology->adjacency);
    graph(data.topology->node_fans);
    for (const auto& [gid, key] : data.topology->by_gid)
    {
        budget.add(sizeof(gid) + sizeof(Key));
        links(4);
        key_allowance(key);
    }
    category("graphs_and_ids");
    const auto incidence = [&](const auto& entries)
    {
        for (const auto& [key, indices] : entries)
        {
            budget.add(sizeof(key) + sizeof(indices) + indices.capacity() * sizeof(std::size_t));
            links(4);
            if constexpr (std::is_same_v<std::decay_t<decltype(key)>, Key>)
                key_allowance(key);
        }
    };
    incidence(data.topology->cell_tiles);
    incidence(data.topology->node_tiles);
    category("incidence");
    for (const auto& [node, walls] : data.topology->node_walls)
    {
        budget.add(sizeof(node) + sizeof(walls) + walls.capacity() * sizeof(GeneralVoF::Data::Wall));
        for (const auto& wall : walls)
        {
            if (const auto* identity = std::get_if<AMR::AdaptiveBytes>(&wall.authority))
                budget.add(identity->capacity());
            if (geometry.insert(wall.geometry.get()).second)
                retain_face(wall.geometry);
        }
    }
    category("walls");
    if (budget.used > data.options.byte_budget)
    {
        namespace failure = ::SimpleFluid::detail::vof_failure;
        failure::Scope context("retained_storage_admission");
        context.record.arithmetic_mode = Mode;
        context.record.mark(failure::Field::Arithmetic);
        context.record.byte_budget = data.options.byte_budget;
        context.record.mark(failure::Field::ByteBudget);
        context.record.bytes_used = budget.used;
        context.record.mark(failure::Field::BytesUsed);
        std::ostringstream message;
        message << "General VoF retained payload byte budget exhausted: accounted_requirement=" << budget.used
                << " limit=" << data.options.byte_budget;
        for (std::size_t i = 0; i < category_count; ++i)
            message << ' ' << categories[i].first << '=' << categories[i].second;
        const auto vector_shape = [&](const char* name, const auto& entries)
        { message << ' ' << name << "_size=" << entries.size() << ' ' << name << "_capacity=" << entries.capacity(); };
        vector_shape("owned_tiles", data.topology->tiles);
        vector_shape("surface_extras", data.topology->extra_surface_tiles);
        vector_shape("surface_references", data.topology->surface_tile_references);
        vector_shape("frontier_references", data.topology->frontier_tile_references);
        message << " tile_vector_bytes=" << tile_vector_bytes << " tile_metadata_bytes=" << tile_metadata_bytes
                << " tile_owned_vector_capacity_bytes=" << tile_owned_vector_capacity_bytes
                << " tile_owned_vector_size_bytes=" << tile_owned_vector_size_bytes
                << " geometry_owner_bytes=" << geometry_root_bytes << " geometry_root_count=" << geometry.size()
                << " encoded_key_allowance_bytes=" << key_allowance_bytes
                << " map_link_allowance_bytes=" << map_link_allowance_bytes
                << " geometry_accounting=unique_shared_owners_with_ledger";
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, message.str());
    }
    if (remaining_bytes)
        *remaining_bytes = data.options.byte_budget - budget.used;
}
template void retained_budget<ArithmeticMode::Ordinary>(
    const GeneralVoF::Data&, std::optional<std::size_t>, std::size_t, std::size_t*);
template void retained_budget<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&, std::optional<std::size_t>, std::size_t, std::size_t*);
template void retained_budget<ArithmeticMode::Exact>(
    const GeneralVoF::Data&, std::optional<std::size_t>, std::size_t, std::size_t*);
} // namespace SimpleFluid::VoF::general_vof_detail
