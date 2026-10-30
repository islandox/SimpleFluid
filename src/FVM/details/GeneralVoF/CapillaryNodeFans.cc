/** @file CapillaryNodeFans.cc
 * @brief Bounded sparse subscriptions to an incident cell owner's complete fan.
 */
#include "FVM/details/GeneralVoF/CapillaryNodeFans.hh"

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
constexpr std::uint64_t request_tag = 0x5346434e46414e01ULL, reply_tag = 0x5346434e46414e02ULL;
constexpr std::size_t tree_links = 4 * sizeof(void*);
using Supports = std::map<std::uint64_t, const Cell*>;
using GO = MeshHandle<>::global_ordinal_type;

std::size_t key_size(const Key& key)
{
    // Match the canonical typed key encoding without allocating a temporary.
    wire::Budget size{std::numeric_limits<std::size_t>::max()};
    size.add(5 * sizeof(std::uint64_t));
    std::visit(
        [&](const auto& ancestry)
        {
            using T = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<T, AMR::CartesianCellAncestry>)
                size.add(4 * sizeof(std::uint64_t));
            else if constexpr (std::is_same_v<T, AMR::ConvexCellAncestry>)
            {
                size.add(2 * sizeof(std::uint64_t));
                for (auto word : ancestry.path)
                {
                    (void) word;
                    size.add(sizeof(std::uint64_t));
                }
            }
            else if constexpr (std::is_same_v<T, AMR::ProductCellAncestry>)
            {
                size.add(5 * sizeof(std::uint64_t));
                for (const auto* path : {&ancestry.base.path, &ancestry.axial.path})
                    for (auto word : *path)
                    {
                        (void) word;
                        size.add(sizeof(std::uint64_t));
                    }
            }
            else
                size.add(2 * sizeof(std::uint64_t));
        },
        key.ancestry);
    return size.used;
}
std::size_t key_capacity(const Key& key)
{
    return std::visit(
        [](const auto& ancestry) -> std::size_t
        {
            using T = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<T, AMR::ConvexCellAncestry>)
                return ancestry.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (std::is_same_v<T, AMR::ProductCellAncestry>)
                return (ancestry.base.path.capacity() + ancestry.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
void encoded_key(wire::Writer& writer, const Key& key, const wire::Budget& live)
{
    auto temporary = live;
    // Canonical encoding grows a byte vector geometrically; its capacity is
    // bounded by twice the serialized size, including its transient old buffer.
    const auto bytes = key_size(key);
    temporary.add(bytes);
    temporary.add(bytes);
    temporary.add(bytes);
    writer.bytes(AMR::encode_adaptive_key(key));
}
std::span<const char> bytes(wire::Reader& reader)
{
    const auto count = reader.word();
    if (count > reader.data.size() - reader.offset)
        throw std::invalid_argument("Capillary node fan byte field exceeds its packet.");
    const auto value = reader.data.subspan(reader.offset, static_cast<std::size_t>(count));
    reader.offset += static_cast<std::size_t>(count);
    return value;
}
Key decoded_key(wire::Reader& reader, const wire::Budget& live)
{
    const auto encoded = bytes(reader);
    auto temporary = live;
    // Decoder path vectors also grow geometrically; include the old buffer
    // while growing and both product paths before decoding any allocation.
    temporary.add(encoded.size());
    temporary.add(encoded.size());
    temporary.add(encoded.size());
    return AMR::decode_adaptive_key(encoded);
}
void stamp(wire::Writer& writer, std::uint64_t tag, const GeneralVoF::Data& data)
{
    writer.word(tag);
    writer.word(data.control);
    writer.word(data.hierarchy->topology_generation());
    writer.word(data.hierarchy->partition_generation());
}
void stamp(wire::Reader& reader, std::uint64_t tag, const GeneralVoF::Data& data)
{
    if (reader.word() != tag || reader.word() != data.control ||
        reader.word() != data.hierarchy->topology_generation() ||
        reader.word() != data.hierarchy->partition_generation())
        throw std::invalid_argument("Capillary node fan source/topology/partition context differs.");
}
bool same_identity(std::span<const char> a, const AMR::AdaptiveBytes& b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
std::size_t header_size(std::uint64_t node, const Cell& cell, const GeneralVoF::Data& data)
{
    wire::Budget size{std::numeric_limits<std::size_t>::max()};
    size.add(8 * sizeof(std::uint64_t)); // stamp, node, identity length, key length, revision
    size.add(data.topology->construction_nodes.at(node).size());
    size.add(key_size(cell.key));
    return size.used;
}
void header(wire::Writer& writer, std::uint64_t tag, std::uint64_t node, const Cell& cell, const GeneralVoF::Data& data,
    const wire::Budget& live)
{
    stamp(writer, tag, data);
    writer.word(node);
    writer.bytes(data.topology->construction_nodes.at(node));
    encoded_key(writer, cell.key, live);
    writer.word(cell.revision);
}
void reserve_packets(std::vector<OwnerPacket>& packets, std::size_t count, wire::Budget& budget)
{
    if (count > (budget.limit - budget.used) / sizeof(OwnerPacket))
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Capillary node fan packet capacity exceeds its byte budget.");
    budget.add(count * sizeof(OwnerPacket));
    packets.reserve(count);
}
std::size_t packet_storage(const std::vector<OwnerPacket>& packets, std::size_t limit)
{
    wire::Budget budget{limit};
    if (packets.capacity() > limit / sizeof(OwnerPacket))
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Capillary node fan packet capacity exceeds its byte budget.");
    budget.add(packets.capacity() * sizeof(OwnerPacket));
    for (const auto& packet : packets)
        budget.add(packet.bytes.capacity());
    return budget.used;
}
void release(std::vector<OwnerPacket>& packets)
{
    std::vector<OwnerPacket>{}.swap(packets);
}
} // namespace

CapillaryNodeFans import_capillary_node_fans(const GeneralVoF::Data& data, std::size_t byte_budget)
{
    const auto comm = data.hierarchy->communicator();
    const auto rank = comm->getRank();
    const auto limit = byte_budget ? std::min(byte_budget, data.options.byte_budget) : data.options.byte_budget;
    CapillaryNodeFans result;
    Supports supports;
    std::size_t support_bytes = 0;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (data.options.arithmetic_mode != ArithmeticMode::Ordinary || !data.topology ||
                !data.options.maximum_stencil_cells)
                throw std::invalid_argument("Capillary node fans require a current Ordinary source.");
            wire::Budget budget{limit};
            const auto retain = [&](std::uint64_t node, const Key& key)
            {
                if (!data.topology->finite_nodes.contains(node) || data.topology->node_fans.contains(node))
                    return;
                const auto found = data.cells.find(key);
                if (found == data.cells.end())
                    return;
                const auto& cell = found->second;
                if (cell.key != key || cell.owner < 0 || cell.owner >= comm->getSize() ||
                    !data.topology->construction_nodes.contains(node) || !data.topology->by_gid.contains(cell.gid) ||
                    data.topology->by_gid.at(cell.gid) != key)
                    throw std::invalid_argument("Capillary node fan supporting cell identity/owner differs.");
                if (cell.owner == rank)
                    throw std::invalid_argument("Owned incident cell lacks its authoritative node fan.");
                const auto prior = supports.find(node);
                if (prior == supports.end())
                {
                    budget.add(sizeof(Supports::value_type) + tree_links);
                    supports.emplace(node, &cell);
                }
                else if (cell.gid < prior->second->gid)
                    prior->second = &cell;
            };
            const auto retain_tile = [&](const AMR::AdaptiveTile& tile)
            {
                for (auto node : tile.node_gids)
                {
                    retain(node, tile.first);
                    if (tile.second && tile.second_image_identity.is_zero())
                        retain(node, *tile.second);
                }
                if (tile.second)
                    for (auto node : tile.second_node_gids)
                        retain(node, *tile.second);
            };
            for (const auto& tile : data.topology->surface_tiles())
                retain_tile(tile);
            for (const auto& tile : data.topology->frontier_tiles())
                retain_tile(tile);
            for (const auto& [node, coordinate] : data.topology->finite_nodes)
                if (!data.topology->node_fans.contains(node) && !supports.contains(node))
                    throw std::invalid_argument("Retained finite node lacks a full supporting cell.");
            support_bytes = budget.used;
        });
    agree_transaction(comm, data.control);
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(support_bytes);
            reserve_packets(requests, supports.size(), budget);
            for (const auto& [node, cell] : supports)
            {
                const auto size = header_size(node, *cell, data);
                budget.add(size);
                wire::Writer writer;
                writer.data.reserve(size);
                header(writer, request_tag, node, *cell, data, budget);
                requests.push_back({cell->owner, std::move(writer.data)});
            }
        });
    const auto exchange_limit = [&](const std::vector<OwnerPacket>& packets)
    {
        std::size_t available = 0;
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                budget.add(support_bytes);
                budget.add(packet_storage(packets, limit));
                available = limit - budget.used;
            });
        return available;
    };
    auto incoming = exchange_owner_packets(requests, comm, exchange_limit(requests));
    release(requests);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(support_bytes);
            budget.add(packet_storage(incoming.packets, limit));
            reserve_packets(replies, incoming.packets.size(), budget);
            std::set<std::pair<int, std::uint64_t>> seen;
            for (const auto& packet : incoming.packets)
            {
                wire::Reader reader{packet.bytes};
                stamp(reader, request_tag, data);
                const auto node = reader.word();
                const auto identity = bytes(reader);
                auto key = decoded_key(reader, budget);
                auto working = budget;
                working.add(key_capacity(key));
                const auto revision = reader.word();
                reader.done();
                const auto source_cell = data.cells.find(key);
                const auto source_fan = data.topology->node_fans.find(node);
                const auto source_identity = data.topology->construction_nodes.find(node);
                if (source_cell == data.cells.end() || source_fan == data.topology->node_fans.end() ||
                    source_identity == data.topology->construction_nodes.end())
                    throw std::invalid_argument("Capillary node fan request lacks retained owner metadata.");
                const auto& cell = source_cell->second;
                const auto& fan = source_fan->second;
                if (packet.peer < 0 || packet.peer >= comm->getSize() || packet.peer == rank || cell.owner != rank ||
                    cell.revision != revision || !data.hierarchy->owned().contains(key) ||
                    data.hierarchy->owned().at(key).revision != revision || !fan.contains(key) ||
                    !same_identity(identity, source_identity->second) || !data.topology->finite_nodes.contains(node) ||
                    cell.gid > std::uint64_t(std::numeric_limits<GO>::max()) ||
                    !data.topology->native.mesh->owned_cell_map()->isNodeGlobalElement(GO(cell.gid)))
                    throw std::invalid_argument("Capillary node fan request lacks current owned-cell authority.");
                if (fan.empty() || fan.size() > data.options.maximum_stencil_cells)
                    throw PLIC::GeometryError(
                        PLIC::GeometryErrorCode::ResourceLimit, "Capillary node fan exceeds its stencil budget.");
                budget.add(sizeof(decltype(seen)::value_type) + tree_links);
                working.add(sizeof(decltype(seen)::value_type) + tree_links);
                if (!seen.emplace(packet.peer, node).second)
                    throw std::invalid_argument("Capillary node fan request repeats a peer/node pair.");
                wire::Budget size{std::numeric_limits<std::size_t>::max()};
                size.add(header_size(node, cell, data));
                size.add(sizeof(std::uint64_t));
                for (const auto& member : fan)
                {
                    const auto found = data.cells.find(member);
                    if (found == data.cells.end())
                        throw std::invalid_argument("Capillary authoritative node fan lacks retained member metadata.");
                    const auto& source = found->second;
                    if (source.key != member || source.owner < 0 || source.owner >= comm->getSize())
                        throw std::invalid_argument("Capillary node fan member owner identity differs.");
                    size.add(3 * sizeof(std::uint64_t)); // key length, owner, revision
                    size.add(key_size(member));
                }
                budget.add(size.used);
                working.add(size.used);
                wire::Writer writer;
                writer.data.reserve(size.used);
                header(writer, reply_tag, node, cell, data, working);
                writer.word(fan.size());
                for (const auto& member : fan)
                {
                    encoded_key(writer, member, working);
                    writer.word(std::uint64_t(data.cells.at(member).owner));
                    writer.word(data.cells.at(member).revision);
                }
                replies.push_back({packet.peer, std::move(writer.data)});
            }
        });
    release(incoming.packets);
    auto returned = exchange_owner_packets(replies, comm, exchange_limit(replies));
    release(replies);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(support_bytes);
            budget.add(packet_storage(returned.packets, limit));
            for (const auto& packet : returned.packets)
            {
                wire::Reader reader{packet.bytes};
                stamp(reader, reply_tag, data);
                const auto node = reader.word();
                const auto identity = bytes(reader);
                const auto support = supports.find(node);
                if (support == supports.end() || support->second->owner != packet.peer ||
                    !same_identity(identity, data.topology->construction_nodes.at(node)) || result.fans.contains(node))
                    throw std::invalid_argument("Capillary node fan reply lacks its requested node/owner identity.");
                auto key = decoded_key(reader, budget);
                auto working = budget;
                working.add(key_capacity(key));
                const auto revision = reader.word();
                if (key != support->second->key || revision != support->second->revision)
                    throw std::invalid_argument("Capillary node fan supporting cell revision differs.");
                const auto count = reader.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw PLIC::GeometryError(
                        PLIC::GeometryErrorCode::ResourceLimit, "Capillary node fan reply exceeds its stencil budget.");
                const auto row_bytes = sizeof(decltype(result.fans)::value_type) + tree_links;
                budget.add(row_bytes);
                working.add(row_bytes);
                result.allocation_bytes += row_bytes;
                auto& fan = result.fans[node];
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto member = decoded_key(reader, working);
                    const auto owner = reader.word(), member_revision = reader.word();
                    if (owner >= std::uint64_t(comm->getSize()))
                        throw std::invalid_argument("Capillary node fan member owner is outside the communicator.");
                    if (const auto found = data.cells.find(member); found != data.cells.end())
                        if (found->second.owner != int(owner) || found->second.revision != member_revision)
                            throw std::invalid_argument("Capillary node fan member owner/revision differs.");
                    const auto member_bytes = sizeof(Key) + tree_links + key_capacity(member);
                    budget.add(member_bytes);
                    working.add(member_bytes);
                    result.allocation_bytes += member_bytes;
                    if (!fan.insert(std::move(member)).second)
                        throw std::invalid_argument("Capillary node fan reply repeats a member.");
                }
                reader.done();
                if (!fan.contains(key))
                    throw std::invalid_argument("Capillary node fan reply omits its supporting cell.");
            }
            if (result.fans.size() != supports.size())
                throw std::invalid_argument("Capillary node fan replies are incomplete.");
            result.source_control = data.control;
        });
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail
