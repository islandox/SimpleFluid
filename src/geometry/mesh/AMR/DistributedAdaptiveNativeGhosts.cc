/** @file DistributedAdaptiveNativeGhosts.cc
 * @brief Owner-authenticated complete native halos with metric-only outer cells.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNativeRetention.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <bit>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using adaptive_detail::GeometryRecordKind;
using adaptive_detail::read_geometry_record_header;
using adaptive_detail::Reader;
using adaptive_detail::write_geometry_record_header;
using adaptive_detail::Writer;
void node_write(Writer& w, const AdaptiveNode& node)
{
    w.word(node.gid);
    w.bytes(node.identity);
    w.bytes(PLIC::encode_identity(node.coordinate_identity));
    for (auto coordinate : node.coordinate)
        w.word(std::bit_cast<std::uint64_t>(coordinate));
}
AdaptiveNode node_read(Reader& r, std::size_t limit)
{
    auto gid = r.word();
    auto identity = r.bytes();
    auto point = PLIC::decode_identity<PLIC::PointIdentity<ArithmeticMode::Ordinary>>(r.bytes(), limit);
    AdaptiveNode node{gid, std::move(identity), {}, std::move(point)};
    for (auto& coordinate : node.coordinate)
        coordinate = std::bit_cast<double>(r.word());
    if (node.coordinate != PLIC::point_display<ArithmeticMode::Ordinary>(node.coordinate_identity))
        throw std::invalid_argument("Invalid exact native ghost node display.");
    return node;
}
AdaptiveNativeDraft fetch_native_draft(const DistributedAdaptiveHierarchy& hierarchy,
    std::span<const AdaptiveCellKey> additional_full_ghosts, bool compact_procedural)
{
    const auto comm = hierarchy.communicator();
    const auto limit = hierarchy.options().byte_budget;
    auto draft =
        compact_procedural ? adaptive_detail::compact_adaptive_native_draft(hierarchy) : hierarchy.native_draft();
    std::map<AdaptiveCellKey, const AdaptiveLeaf*> owned;
    std::map<AdaptiveCellKey, std::vector<const AdaptiveTile*>> rows;
    std::map<std::uint64_t, const AdaptiveNode*> local_nodes;
    std::vector<AdaptiveCellKey> ghost_keys;
    adaptive_detail::collective(*comm,
        [&]
        {
            for (const auto& leaf : draft.owned)
                owned.emplace(leaf.key, &leaf);
            std::set<AdaptiveCellKey> remote;
            adaptive_detail::Budget halo_budget{limit};
            for (const auto& key : additional_full_ghosts)
            {
                halo_budget.add(sizeof(AdaptiveCellKey) + encode_adaptive_key(key).size() + 4 * sizeof(void*));
                if (!owned.contains(key))
                    remote.insert(key);
            }
            for (const auto& tile : draft.tiles)
            {
                if (owned.contains(tile.first))
                    rows[tile.first].push_back(&tile);
                else
                    remote.insert(tile.first);
                if (tile.second)
                {
                    if (owned.contains(*tile.second))
                        rows[*tile.second].push_back(&tile);
                    else
                        remote.insert(*tile.second);
                }
            }
            ghost_keys.assign(remote.begin(), remote.end());
            for (const auto& node : draft.nodes)
                local_nodes.emplace(node.gid, &node);
        });
    const auto owners = hierarchy.lookup_cells(ghost_keys);
    auto ghosts = compact_procedural ? adaptive_detail::fetch_native_source_leaves(hierarchy, ghost_keys)
                                     : hierarchy.fetch_leaves(ghost_keys);
    std::vector<OwnerPacket> requests;
    adaptive_detail::collective(*comm,
        [&]
        {
            adaptive_detail::Budget budget{limit};
            for (std::size_t i = 0; i < ghost_keys.size(); ++i)
            {
                Writer w;
                write_geometry_record_header(w, GeometryRecordKind::GhostRequest);
                w.word(i);
                w.bytes(encode_adaptive_key(ghost_keys[i]));
                w.word(owners[i].revision);
                budget.packet(w.data.size());
                requests.push_back({owners[i].owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket>().swap(requests);
    std::vector<OwnerPacket> responses;
    adaptive_detail::collective(*comm,
        [&]
        {
            adaptive_detail::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                read_geometry_record_header(r, GeometryRecordKind::GhostRequest);
                const auto ordinal = r.word();
                auto key_bytes = r.bytes();
                const auto revision = r.word();
                r.done();
                const auto key = decode_adaptive_key(key_bytes);
                const auto leaf = owned.at(key);
                if (leaf->revision != revision)
                    throw std::invalid_argument("Stale native full-ghost request.");
                Writer w;
                write_geometry_record_header(w, GeometryRecordKind::GhostReply);
                w.word(ordinal);
                w.bytes(key_bytes);
                w.word(revision);
                const auto& tiles = rows.at(key);
                w.word(tiles.size());
                std::set<std::uint64_t> needed;
                for (const auto* tile : tiles)
                {
                    auto bytes = encode_adaptive_tile(*tile, limit);
                    budget.add(bytes.size() + 8);
                    w.bytes(bytes);
                    needed.insert(tile->node_gids.begin(), tile->node_gids.end());
                    needed.insert(tile->second_node_gids.begin(), tile->second_node_gids.end());
                }
                w.word(needed.size());
                for (auto gid : needed)
                {
                    const auto& node = *local_nodes.at(gid);
                    budget.add(
                        sizeof(node) + node.identity.size() + PLIC::encoded_identity_size(node.coordinate_identity));
                    node_write(w, node);
                }
                budget.packet(w.data.size());
                responses.push_back({packet.peer, std::move(w.data)});
            }
        });
    std::vector<OwnerPacket>().swap(incoming.packets);
    auto returned = exchange_owner_packets(responses, comm, limit);
    adaptive_detail::collective(*comm,
        [&]
        {
            std::map<std::uint64_t, AdaptiveTile> tiles;
            std::map<std::uint64_t, AdaptiveNode> nodes;
            for (auto& tile : draft.tiles)
                tiles.emplace(tile.face_gid, std::move(tile));
            for (auto& node : draft.nodes)
                nodes.emplace(node.gid, std::move(node));
            std::vector<bool> seen(ghost_keys.size());
            adaptive_detail::Budget budget{limit};
            for (const auto& packet : returned.packets)
            {
                Reader r{packet.bytes};
                read_geometry_record_header(r, GeometryRecordKind::GhostReply);
                const auto ordinal = r.word();
                const auto key = decode_adaptive_key(r.bytes());
                const auto revision = r.word();
                if (ordinal >= ghost_keys.size() || seen[ordinal] || packet.peer != owners[ordinal].owner ||
                    key != ghost_keys[ordinal] || revision != owners[ordinal].revision)
                    throw std::invalid_argument("Foreign or duplicate complete native ghost package.");
                seen[ordinal] = true;
                const auto count = r.word();
                if (count > (packet.bytes.size() - r.offset) / 8)
                    throw std::invalid_argument("Invalid native ghost tile count.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto bytes = r.bytes();
                    budget.add(bytes.size());
                    auto tile = decode_adaptive_tile(bytes, limit);
                    if (tile.first != key && (!tile.second || *tile.second != key))
                        throw std::invalid_argument("Nonincident tile in native ghost package.");
                    auto found = tiles.find(tile.face_gid);
                    if (found != tiles.end())
                    {
                        if (encode_adaptive_tile(found->second, limit) != bytes)
                            throw std::invalid_argument("Native ghost canonical tile replicas disagree.");
                    }
                    else
                        tiles.emplace(tile.face_gid, std::move(tile));
                }
                const auto node_count = r.word();
                if (node_count > (packet.bytes.size() - r.offset) / 48)
                    throw std::invalid_argument("Invalid native ghost node count.");
                for (std::uint64_t i = 0; i < node_count; ++i)
                {
                    auto node = node_read(r, limit);
                    budget.add(sizeof(node) + node.identity.size() + node.coordinate_identity.payload_bytes());
                    auto found = nodes.find(node.gid);
                    if (found != nodes.end())
                    {
                        if (found->second.identity != node.identity ||
                            found->second.coordinate_identity != node.coordinate_identity ||
                            found->second.coordinate != node.coordinate)
                            throw std::invalid_argument("Native ghost canonical node replicas disagree.");
                    }
                    else
                        nodes.emplace(node.gid, std::move(node));
                }
                r.done();
            }
            if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
                throw std::invalid_argument("Missing complete native ghost package.");
            draft.tiles.clear();
            for (auto& [gid, tile] : tiles)
                draft.tiles.push_back(std::move(tile));
            draft.nodes.clear();
            for (auto& [gid, node] : nodes)
                draft.nodes.push_back(std::move(node));
            for (auto& leaf : ghosts)
            {
                const bool compact = adaptive_detail::compact_native_geometry(leaf, compact_procedural);
                leaf.geometry = hierarchy.geometry(leaf);
                leaf.surfaces = hierarchy.surfaces(leaf);
                if (compact)
                    leaf.geometry.reset();
                draft.ghosts.push_back(std::move(leaf));
            }
        });
    std::vector<AdaptiveCellKey> keys;
    adaptive_detail::collective(*comm,
        [&]
        {
            std::set<AdaptiveCellKey> needed;
            for (const auto& leaf : draft.owned)
                needed.insert(leaf.key);
            for (const auto& leaf : draft.ghosts)
                needed.insert(leaf.key);
            for (const auto& tile : draft.tiles)
            {
                needed.insert(tile.first);
                if (tile.second)
                    needed.insert(*tile.second);
            }
            keys.assign(needed.begin(), needed.end());
        });
    const auto locations = hierarchy.lookup_cells(keys);
    const auto records = compact_procedural ? adaptive_detail::fetch_native_source_leaves(hierarchy, keys)
                                            : hierarchy.fetch_leaves(keys);
    adaptive_detail::collective(*comm,
        [&]
        {
            std::set<AdaptiveCellKey> full;
            for (const auto& leaf : draft.owned)
                full.insert(leaf.key);
            for (const auto& leaf : draft.ghosts)
                full.insert(leaf.key);
            draft.cell_bindings.clear();
            draft.global_cells = hierarchy.global_cells();
            // Use this exact fetch/repack selection for both metadata and full
            // records. Earlier ghost-only requests can have different ordinals.
            draft.owned.clear();
            draft.ghosts.clear();
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (records[i].key != keys[i] || records[i].revision != locations[i].revision)
                    throw std::invalid_argument("Native metric owner record mismatch.");
                const auto metric = hierarchy.metrics(records[i]);
                AdaptiveNativeCellBinding binding;
                binding.key = keys[i];
                binding.gid = locations[i].address;
                binding.material = records[i].material;
                binding.owner = locations[i].owner;
                binding.provider = records[i].provider;
                binding.full = full.contains(keys[i]);
                binding.volume = metric.volume;
                binding.centroid = metric.centroid;
                binding.native_cell = records[i].native_cell;
                if (binding.full)
                    binding.native_recipe = records[i].native_recipe;
                if (binding.full)
                {
                    auto full_record = records[i];
                    full_record.geometry = hierarchy.geometry(full_record);
                    full_record.surfaces = hierarchy.surfaces(full_record);
                    if (adaptive_detail::compact_native_geometry(records[i], compact_procedural))
                        full_record.geometry.reset();
                    if (binding.owner == comm->getRank())
                        draft.owned.push_back(std::move(full_record));
                    else
                        draft.ghosts.push_back(std::move(full_record));
                }
                draft.cell_bindings.push_back(std::move(binding));
            }
        });
    return draft;
}
} // namespace
AdaptiveNativeDraft fetch_adaptive_native_draft(
    const DistributedAdaptiveHierarchy& hierarchy, std::span<const AdaptiveCellKey> additional_full_ghosts)
{
    return fetch_native_draft(hierarchy, additional_full_ghosts, false);
}
AdaptiveNativeDraft adaptive_detail::fetch_compact_adaptive_native_draft(
    const DistributedAdaptiveHierarchy& hierarchy, std::span<const AdaptiveCellKey> additional_full_ghosts)
{
    return fetch_native_draft(hierarchy, additional_full_ghosts, true);
}
AdaptiveNativePublication assemble_adaptive_native(const DistributedAdaptiveHierarchy& hierarchy,
    std::span<const char> root_schema, std::span<const AdaptiveCellKey> additional_full_ghosts)
{
    auto draft = adaptive_detail::fetch_compact_adaptive_native_draft(hierarchy, additional_full_ghosts);
    return assemble_adaptive_native(
        std::move(draft), root_schema, hierarchy.communicator(), hierarchy.options().byte_budget);
}
} // namespace SimpleFluid::Meshes::AMR
