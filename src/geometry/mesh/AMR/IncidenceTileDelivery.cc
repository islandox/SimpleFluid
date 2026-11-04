#include "IncidenceTileDelivery.hh"
#include "BudgetedOwnerExchange.hh"
#include "DistributedAdaptiveClosureRecords.hh"
#include "DistributedAdaptiveIdentityBudget.hh"
#include "ResidentSupportCertification.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
struct Charge
{
    Budget& b;
    std::size_t n = 0;
    void add(std::size_t x)
    {
        if (x > std::numeric_limits<std::size_t>::max() - n)
            throw std::invalid_argument("Delivery charge overflow.");
        b.add(x);
        n += x;
    }
    ~Charge() { b.used -= n; }
};
void header(Writer& w, ContactHaloContext c, std::uint64_t generation, std::uint64_t kind)
{
    w.word(1);
    w.word(c.epoch);
    w.word(c.topology);
    w.word(c.partition);
    w.word(c.round);
    w.word(c.transaction);
    w.word(generation);
    w.word(kind);
}
std::pair<std::uint64_t, std::uint64_t> header(Reader& r, ContactHaloContext c)
{
    if (r.word() != 1 || r.word() != c.epoch || r.word() != c.topology || r.word() != c.partition ||
        r.word() != c.round || r.word() != c.transaction)
        throw std::invalid_argument("Stale incidence delivery context.");
    auto generation = r.word(), kind = r.word();
    if (kind > 1)
        throw std::invalid_argument("Invalid incidence delivery kind.");
    return {generation, kind};
}
std::size_t source_recipe_size(const IncidenceDeliverySource& source, const PrivateTileRecipe& item)
{
    if (!item.face || item.first_member >= source.members.size() ||
        (item.second_member && *item.second_member >= source.members.size()))
        throw std::invalid_argument("Invalid incidence recipe.");
    std::size_t size = 88;
    const auto add = [&](std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() - size)
            throw std::invalid_argument("Incidence recipe size overflow.");
        size += n;
    };
    add(source.support.size());
    add(source.domain.size());
    add(key_wire_size(source.members[item.first_member].cell));
    if (item.second_member)
    {
        add(32);
        add(key_wire_size(source.members[*item.second_member].cell));
    }
    add(PLIC::encoded_identity_size(item.second_image));
    add(item.face->encoded_incidence_size());
    return size;
}
// ResidentSupportCertification retains a key set, a cell-keyed coverage map,
// and translated/reversed temporary faces while checking a group. Admit their
// allocation envelope before entering it. Native PLIC's own nested construction
// and partition-certificate scratch remains governed by its existing provider
// policy, which does not expose a shared-arena cap.
std::size_t certification_caller_scratch(
    std::span<const char> support, std::span<const RegistryMember> members, std::span<const ReceivedTileRecipe> tiles)
{
    std::size_t total = 0;
    const auto add = [&](std::size_t n, std::size_t factor)
    {
        if (factor && n > (std::numeric_limits<std::size_t>::max() - total) / factor)
            throw std::invalid_argument("Incidence certification scratch overflow.");
        total += n * factor;
    };
    add(support.size(), 4);
    add(members.size(), 512);
    add(tiles.size(), 512);
    for (const auto& member : members)
        add(key_wire_size(member.cell), 4);
    for (const auto& tile : tiles)
    {
        if (!tile.face)
            throw std::invalid_argument("Missing certification face.");
        // Canonical key temporaries, both directions, the set-owned key, and
        // their vector growth overlap. Incidence encoding is an upper bound on
        // canonical encoding because canonicalization only removes vertices.
        add(tile.face->encoded_incidence_size(), 8);
        add(tile.support.size(), 4);
        add(key_wire_size(tile.first.cell), 4);
        if (tile.second)
            add(key_wire_size(tile.second->cell), 4);
        // Reversal and translated physical owner images overlap their source.
        add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(tile.face->points().size()), 4);
        add(tile.face->storage_bytes(), 4);
    }
    return total;
}
bool same_member(const RegistryMember& a, const RegistryMember& b)
{
    return a.cell == b.cell && a.revision == b.revision && a.surface == b.surface && a.owner == b.owner;
}
} // namespace
void deliver_incidence_tiles(std::span<const IncidenceDeliverySource> sources,
    std::span<const IncidenceDeliveryExpected> expected, ContactHaloContext context,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget, IdentityBudget& identities,
    const OwnedLeafLookup& lookup,
    const std::function<void(std::span<const char>, std::uint64_t, std::span<const ReceivedTileRecipe>)>& publish)
{
    if (comm.is_null())
        throw std::invalid_argument("Incidence delivery requires communicator.");
    struct Group
    {
        bool terminal = false, has_generation = false;
        std::uint64_t generation = 0, count = 0;
        std::vector<ReceivedTileRecipe> tiles;
    };
    Charge retained{budget};
    // Decoded faces are temporary. Canonical owners retained by the callback
    // enter the session ledger only after typed pool interning; discarded
    // equal planes must not remain pinned for the lifetime of the session.
    IdentityBudget decoded_identities;
    (void) identities;
    std::vector<Group> groups;
    collective(*comm,
        [&]
        {
            if (expected.size() > std::numeric_limits<std::size_t>::max() / sizeof(Group))
                throw std::invalid_argument("Delivery group overflow.");
            retained.add(expected.size() * sizeof(Group));
            groups.resize(expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                if (expected[i].support.empty() || expected[i].local_members.empty())
                    throw std::invalid_argument("Empty expected owner group.");
                for (std::size_t j = 0; j < i; ++j)
                    if (std::ranges::equal(expected[i].support, expected[j].support))
                        throw std::invalid_argument("Duplicate expected owner group.");
                for (const auto& m : expected[i].local_members)
                    if (m.owner != comm->getRank())
                        throw std::invalid_argument("Foreign expected owner member.");
            }
            for (std::size_t i = 0; i < sources.size(); ++i)
            {
                const auto& source = sources[i];
                if (source.support.empty() || source.domain.empty() ||
                    authority(source.support, comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("Foreign delivery source.");
                for (std::size_t j = 0; j < i; ++j)
                    if (std::ranges::equal(source.support, sources[j].support))
                        throw std::invalid_argument("Duplicate delivery source group.");
            }
        });
    // A logical sequence of tile records followed by one terminal per distinct
    // destination. Re-scan members to avoid an unbounded destination table.
    std::size_t source_index = 0, recipe = 0, side = 0, marker = 0;
    for (;;)
    {
        // A soft per-rank page target leaves room for simultaneous input from
        // all authorities and for the exchange's transport and decode buffers.
        // It never rejects an individually admissible large record.
        const auto free_credit = static_cast<unsigned long long>(budget.limit - budget.used);
        unsigned long long minimum_credit = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &free_credit, &minimum_credit);
        const auto page_target = static_cast<std::size_t>(minimum_credit / static_cast<unsigned>(comm->getSize()) / 8);
        std::vector<OwnerPacket> outgoing;
        std::size_t charge = 0, logical_bytes = 0, framed_bytes = 0;
        collective(*comm,
            [&]
            {
                // Return false without moving the cursor when a nonempty page
                // should be exchanged before this record. The empty-page path
                // admits large records under the original shared cap.
                const auto can_append = [&](std::size_t bytes, std::size_t scratch)
                {
                    constexpr auto max = std::numeric_limits<std::size_t>::max();
                    constexpr auto wire_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
                    if (bytes > wire_max - sizeof(std::uint64_t))
                        throw std::invalid_argument("Incidence record exceeds transport count range.");
                    const auto framed = bytes + sizeof(std::uint64_t);
                    if (!outgoing.empty() && (logical_bytes >= page_target || framed > page_target - logical_bytes ||
                                                 framed > wire_max - framed_bytes || outgoing.size() >= wire_max))
                        return false;
                    std::size_t growth = 0;
                    if (outgoing.size() == outgoing.capacity())
                    {
                        const auto old = outgoing.capacity();
                        if (old > max / 2 / sizeof(OwnerPacket))
                            throw std::invalid_argument("Incidence packet vector capacity overflow.");
                        growth = (old ? old * 2 : 1) * sizeof(OwnerPacket);
                    }
                    const auto available = budget.limit - budget.used;
                    if (growth > available || bytes > available - growth || scratch > available - growth - bytes)
                    {
                        if (!outgoing.empty())
                            return false;
                        throw std::invalid_argument("Incidence page record exceeds live byte credit.");
                    }
                    return true;
                };
                const auto begin_packet = [&](std::size_t bytes)
                {
                    const auto before = budget.used;
                    budget.append_capacity(outgoing);
                    charge += budget.used - before;
                    budget.add(bytes);
                    charge += bytes;
                    logical_bytes += bytes + sizeof(std::uint64_t);
                    framed_bytes += bytes + sizeof(std::uint64_t);
                };
                while (source_index < sources.size())
                {
                    const auto* source = &sources[source_index];
                    while (recipe < source->recipes.size())
                    {
                        const auto& item = source->recipes[recipe];
                        if (item.first_member >= source->members.size() ||
                            (item.second_member && *item.second_member >= source->members.size()))
                            throw std::invalid_argument("Invalid incidence recipe membership.");
                        const int first = source->members[item.first_member].owner;
                        const int second = item.second_member ? source->members[*item.second_member].owner : first;
                        if (first < 0 || first >= comm->getSize() || second < 0 || second >= comm->getSize())
                            throw std::invalid_argument("Invalid incidence destination.");
                        const bool send_second = side == 1;
                        if (send_second && second == first)
                        {
                            ++recipe;
                            side = 0;
                            continue;
                        }
                        const int peer = send_second ? second : first;
                        const auto body = source_recipe_size(*source, item);
                        if (body > (std::numeric_limits<std::size_t>::max() - 72) / 6)
                            throw std::invalid_argument("Incidence packet overflow.");
                        const auto bytes = body + 72;
                        if (!can_append(bytes, body * 6))
                            return;
                        Charge scratch{budget};
                        scratch.add(body * 6);
                        begin_packet(bytes);
                        Writer w;
                        w.data.reserve(bytes);
                        header(w, context, source->generation, 1);
                        w.bytes(encode_tile_recipe(source->support, source->domain, source->members, item, recipe));
                        if (w.data.capacity() != bytes || w.data.size() != bytes)
                            throw std::invalid_argument("Delivery packet capacity mismatch.");
                        outgoing.push_back({peer, std::move(w.data)});
                        if (!side)
                            side = 1;
                        else
                        {
                            ++recipe;
                            side = 0;
                        }
                    }
                    while (marker < source->members.size())
                    {
                        const auto index = marker;
                        const int peer = source->members[index].owner;
                        if (peer < 0 || peer >= comm->getSize())
                            throw std::invalid_argument("Invalid terminal destination.");
                        bool earlier = false;
                        for (std::size_t i = 0; i < index; ++i)
                            earlier |= source->members[i].owner == peer;
                        if (earlier)
                        {
                            ++marker;
                            continue;
                        }
                        std::uint64_t count = 0;
                        for (const auto& item : source->recipes)
                            if (source->members[item.first_member].owner == peer ||
                                (item.second_member && source->members[*item.second_member].owner == peer))
                                ++count;
                        if (source->support.size() > std::numeric_limits<std::size_t>::max() - 80)
                            throw std::invalid_argument("Terminal overflow.");
                        const auto bytes = source->support.size() + 80;
                        if (!can_append(bytes, 0))
                            return;
                        begin_packet(bytes);
                        Writer w;
                        w.data.reserve(bytes);
                        header(w, context, source->generation, 0);
                        w.bytes(source->support);
                        w.word(count);
                        if (w.data.capacity() != bytes || w.data.size() != bytes)
                            throw std::invalid_argument("Terminal capacity mismatch.");
                        outgoing.push_back({peer, std::move(w.data)});
                        ++marker;
                    }
                    ++source_index;
                    recipe = side = marker = 0;
                }
            });
        int active = outgoing.empty() ? 0 : 1, any = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &active, &any);
        if (!any)
            break;
        auto incoming = consume_budgeted_owner_packets(std::move(outgoing), comm, budget, charge);
        collective(*comm,
            [&]
            {
                for (const auto& packet : incoming.packets)
                {
                    Reader r{packet.bytes};
                    const auto [generation, kind] = header(r, context);
                    const auto body = r.bytes_view();
                    std::span<const char> key = body;
                    if (kind == 1)
                    {
                        Reader peek{body};
                        if (peek.word() != 1)
                            throw std::invalid_argument("Invalid tile version.");
                        key = peek.bytes_view();
                    }
                    std::size_t index = 0;
                    while (index < expected.size() && !std::ranges::equal(key, expected[index].support))
                        ++index;
                    if (index == expected.size() || authority(key, comm->getSize()) != packet.peer ||
                        generation < expected[index].minimum_generation)
                        throw std::invalid_argument("Unexpected incidence group receipt.");
                    auto& group = groups[index];
                    if (group.has_generation && group.generation != generation)
                        throw std::invalid_argument("Mixed incidence group generations.");
                    group.has_generation = true;
                    group.generation = generation;
                    if (group.terminal)
                        throw std::invalid_argument("Incidence record after terminal.");
                    if (!kind)
                    {
                        group.count = r.word();
                        r.done();
                        group.terminal = true;
                        continue;
                    }
                    r.done();
                    const auto available = budget.limit - budget.used;
                    Charge decode{budget};
                    decode.add(available);
                    // The decoder constructs keys, a native face, and a canonical
                    // re-encoding concurrently. Give each nested codec only a
                    // fraction of the already admitted aggregate scratch envelope.
                    constexpr std::size_t decode_envelope_factor = 16;
                    const auto nested_limit = available / decode_envelope_factor;
                    auto tile = decode_tile_recipe(body, packet.peer, comm->getRank(), comm->getSize(), nested_limit);
                    std::size_t bytes = sizeof(ReceivedTileRecipe);
                    const auto add_bytes = [&](std::size_t value)
                    {
                        if (bytes > available || value > available - bytes)
                            throw std::invalid_argument("Decoded incidence exceeds admission.");
                        bytes += value;
                    };
                    add_bytes(tile.support.capacity());
                    add_bytes(tile.domain.capacity());
                    add_bytes(key_capacity(tile.first.cell));
                    if (tile.second)
                        add_bytes(key_capacity(tile.second->cell));
                    add_bytes(tile.second_image.payload_bytes());
                    add_bytes(tile.face->exclusive_storage_bytes());
                    add_bytes(sizeof(PLIC::PlanarFace<ArithmeticMode::Ordinary>));
                    budget.used -= available - bytes;
                    decode.n = 0;
                    retained.n += bytes;
                    const auto match = [&](const RegistryMember& m)
                    {
                        return m.owner != comm->getRank() || std::ranges::any_of(expected[index].local_members,
                                                                 [&](const auto& e) { return same_member(e, m); });
                    };
                    if (!match(tile.first) || (tile.second && !match(*tile.second)))
                        throw std::invalid_argument("Stale incidence member receipt.");
                    const auto old = budget.used;
                    budget.append_capacity(group.tiles);
                    retained.n += budget.used - old;
                    const auto identity_before = budget.used;
                    decoded_identities.retain(budget, PLIC::planar_support<ArithmeticMode::Ordinary>(*tile.face));
                    retained.n += budget.used - identity_before;
                    group.tiles.push_back(std::move(tile));
                }
            });
        retire_budgeted_owner_packets(incoming, budget);
    }
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < groups.size(); ++i)
            {
                auto& group = groups[i];
                if (!group.terminal || group.count != group.tiles.size())
                    throw std::invalid_argument("Incomplete incidence group.");
                for (std::size_t j = 0; j < group.tiles.size(); ++j)
                    for (std::size_t k = 0; k < j; ++k)
                        if (group.tiles[j].emission == group.tiles[k].emission)
                            throw std::invalid_argument("Duplicate incidence receipt.");
                Charge certification{budget};
                certification.add(
                    certification_caller_scratch(expected[i].support, expected[i].local_members, group.tiles));
                certify_resident_support(
                    expected[i].support, expected[i].local_members, group.tiles, comm->getRank(), lookup);
            }
        });
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < groups.size(); ++i)
                publish(expected[i].support, groups[i].generation, groups[i].tiles);
        });
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
