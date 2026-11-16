#include "geometry/mesh/AMR/SupportGroupFetch.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/PooledSupportRegistry.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::uint64_t magic = 0x5346475250464554ULL;
constexpr std::size_t words = 8 * 8;
struct Scope
{
    Budget& budget;
    std::size_t held = 0;
    ~Scope() { budget.used -= held; }
    void add(std::size_t n)
    {
        budget.add(n);
        held += n;
    }
    void drop(std::size_t n)
    {
        held -= n;
        budget.used -= n;
    }
    template<class T> void append(std::vector<T>& values)
    {
        const auto before = budget.used;
        try
        {
            budget.append_capacity(values);
        }
        catch (...)
        {
            held += budget.used - before;
            throw;
        }
        held += budget.used - before;
    }
};
void add_size(std::size_t& total, std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() - total)
        throw std::invalid_argument("Support fetch byte count overflows.");
    total += n;
}
void header(Writer& writer, ContactHaloContext context, std::uint64_t page)
{
    writer.word(magic);
    writer.word(1);
    writer.word(context.epoch);
    writer.word(context.topology);
    writer.word(context.partition);
    writer.word(context.round);
    writer.word(context.transaction);
    writer.word(page);
}
void check_header(Reader& reader, ContactHaloContext context, std::uint64_t page)
{
    if (reader.word() != magic || reader.word() != 1 || reader.word() != context.epoch ||
        reader.word() != context.topology || reader.word() != context.partition || reader.word() != context.round ||
        reader.word() != context.transaction || reader.word() != page)
        throw std::invalid_argument("Invalid support fetch context or page.");
}
std::size_t key_bound(const AdaptiveCellKey& key)
{
    const auto dynamic = key_capacity(key);
    if (dynamic > std::numeric_limits<std::size_t>::max() / 4 - 128)
        throw std::invalid_argument("Support fetch cell key overflows.");
    return 4 * (dynamic + 128);
}
std::size_t packet_bytes(const std::vector<OwnerPacket>& packets)
{
    if (packets.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(OwnerPacket))
        throw std::invalid_argument("Support fetch packet metadata overflows.");
    auto bytes = packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
        add_size(bytes, packet.bytes.capacity());
    return bytes;
}
OwnerExchange exchange(std::span<const OwnerPacket> packets, Teuchos::RCP<const Teuchos::Comm<int>> comm, Scope& scope)
{
    std::size_t cap = 0;
    collective(*comm,
        [&]
        {
            cap = scope.budget.limit - scope.budget.used;
            if (!cap)
                throw std::invalid_argument("Support fetch transport lease exhausted.");
            scope.add(cap);
        });
    auto result = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(packets, comm, cap);
    collective(*comm,
        [&]
        {
            const auto retained = packet_bytes(result.packets);
            if (retained > cap)
                throw std::invalid_argument("Support fetch transport exceeded lease.");
            scope.drop(cap - retained);
        });
    return result;
}
void prepare(Writer& writer, std::size_t bytes, Scope& scope)
{
    scope.add(bytes);
    writer.data.reserve(bytes);
    if (writer.data.capacity() != bytes)
        throw std::invalid_argument("Support fetch packet capacity differs from admission.");
}
AdaptiveBytes encoded_key(const AdaptiveCellKey& key, Scope& scope)
{
    const auto bound = key_bound(key);
    scope.add(bound);
    auto result = encode_adaptive_key(key);
    if (result.capacity() > bound)
        throw std::invalid_argument("Support fetch key exceeds admission.");
    return result;
}
struct Group
{
    PooledSupportRegistry::Slot slot;
    std::size_t first, count;
};
struct Expected
{
    PooledSupportRegistry::Slot group;
    std::size_t member;
    std::span<const char> payload;
    bool seen = false;
};
} // namespace

void fetch_dirty_support_groups(const PooledSupportRegistry& registry, const OwnedLeafLookup& lookup,
    ContactHaloContext context, Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget, IdentityBudget& identities,
    std::size_t page_members, const CompleteGroupExpected& expect, const CompleteGroupEmit& emit,
    const CompleteGroupFlush& flush)
{
    if (comm.is_null())
        throw std::invalid_argument("Support fetch requires a communicator.");
    collective(*comm,
        [&]
        {
            if (!page_members || !lookup || !expect || !emit || !flush)
                throw std::invalid_argument("Invalid support fetch callbacks or page size.");
        });
    // The session-wide ledger is owned by callbacks that retain outputs. A
    // separate group ledger is essential: it releases temporary face pins at
    // the end of each group, while both ledgers use this same live Budget.
    (void) identities;
    const auto& dirty = registry.dirty_slots();
    auto next = dirty.begin();
    std::uint64_t page = 0;
    for (;;)
    {
        const int local = next != dirty.end();
        int any = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &any);
        if (!any)
            break;
        Scope page_memory{budget};
        std::vector<Group> groups;
        std::vector<Expected> expected;
        std::vector<OwnerPacket> requests;
        collective(*comm,
            [&]
            {
                const auto initial_free = budget.limit - budget.used;
                std::size_t projected = 0, page_count = 0;
                while (next != dirty.end())
                {
                    const auto slot = *next;
                    const auto& support = registry.support_key(slot);
                    const auto* members = registry.find(slot);
                    const auto count = members ? members->size() : 0;
                    std::size_t cost = sizeof(Group);
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        const auto& member = (*members)[i];
                        if (member.owner < 0 || member.owner >= comm->getSize() || !member.encoded_size)
                            throw std::invalid_argument("Invalid support member metadata.");
                        const auto key = key_bound(registry.member_cell(member));
                        // Four transport streams and one encoder result can overlap.
                        // Include vector growth and each framed key/support copy.
                        add_size(cost, 2 * sizeof(Expected) + 8 * sizeof(OwnerPacket) + sizeof(SupportOccurrence) +
                                           sizeof(RegistryMember) + 8 * words + 512);
                        for (unsigned copy = 0; copy < 4; ++copy)
                        {
                            add_size(cost, support.size());
                            add_size(cost, key);
                        }
                        for (unsigned copy = 0; copy < 3; ++copy)
                            add_size(cost, member.encoded_size);
                    }
                    // A group is indivisible. The soft member target batches small
                    // groups; a larger group gets its own page if it fits the arena.
                    if (!groups.empty() && (page_count >= page_members || cost > initial_free - projected))
                        break;
                    if (cost > initial_free - projected)
                        throw std::invalid_argument("Complete support group exceeds available fetch arena.");
                    add_size(projected, cost);
                    page_memory.append(groups);
                    groups.push_back({slot, expected.size(), count});
                    ++next;
                    const auto units = std::max<std::size_t>(count, 1);
                    if (units > std::numeric_limits<std::size_t>::max() - page_count)
                        throw std::invalid_argument("Support fetch page member count overflows.");
                    page_count += units;
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        const auto& member = (*members)[i];
                        page_memory.append(expected);
                        expected.push_back({slot, i, {}, false});
                        auto key = encoded_key(registry.member_cell(member), page_memory);
                        std::size_t packet_size = words + 8 * 6;
                        add_size(packet_size, support.size());
                        add_size(packet_size, key.size());
                        page_memory.append(requests);
                        Writer writer;
                        prepare(writer, packet_size, page_memory);
                        header(writer, context, page);
                        writer.bytes(support.bytes());
                        writer.bytes(key);
                        writer.word(member.revision);
                        writer.word(member.surface);
                        writer.word(member.encoded_size);
                        writer.word(expected.size() - 1);
                        if (writer.data.size() != packet_size)
                            throw std::logic_error("Support request framing differs.");
                        requests.push_back({member.owner, std::move(writer.data)});
                    }
                }
            });
        auto incoming = exchange(requests, comm, page_memory);
        std::vector<OwnerPacket> replies;
        collective(*comm,
            [&]
            {
                for (const auto& packet : incoming.packets)
                {
                    Reader reader{packet.bytes};
                    check_header(reader, context, page);
                    const auto support = reader.bytes_view();
                    const auto encoded = reader.bytes_view();
                    const auto revision = reader.word(), ordinal = reader.word(), size = reader.word(),
                               index = reader.word();
                    reader.done();
                    if (support.empty() || authority(support, comm->getSize()) != packet.peer ||
                        encoded.size() > budget.limit / 2)
                        throw std::invalid_argument("Foreign support fetch request.");
                    page_memory.add(2 * encoded.size());
                    const auto key = decode_adaptive_key(encoded);
                    const auto& leaf = lookup(key);
                    if (leaf.key != key || leaf.revision != revision || ordinal >= leaf.surfaces.size() ||
                        !std::ranges::equal(leaf.surfaces[ordinal].key, support))
                        throw std::invalid_argument("Stale support fetch membership.");
                    expect(support);
                    page_memory.append(replies);
                    const auto cap = budget.limit - budget.used;
                    if (!cap || size > cap)
                        throw std::invalid_argument("Support encoding lease exhausted.");
                    page_memory.add(cap);
                    auto geometry = encode_support_surface(leaf, ordinal);
                    if (geometry.size() != size || geometry.capacity() > cap)
                        throw std::invalid_argument("Support encoded size changed or exceeds lease.");
                    page_memory.drop(cap - geometry.capacity());
                    std::size_t response_size = words + 8 * 7;
                    add_size(response_size, support.size());
                    add_size(response_size, encoded.size());
                    add_size(response_size, geometry.size());
                    Writer writer;
                    prepare(writer, response_size, page_memory);
                    header(writer, context, page);
                    writer.bytes(support);
                    writer.bytes(encoded);
                    writer.word(revision);
                    writer.word(ordinal);
                    writer.word(size);
                    writer.word(index);
                    writer.bytes(geometry);
                    if (writer.data.size() != response_size)
                        throw std::logic_error("Support reply framing differs.");
                    replies.push_back({packet.peer, std::move(writer.data)});
                }
            });
        auto returned = exchange(replies, comm, page_memory);
        // Validate every response, sender and complete member slot collectively
        // before any local complete-group callback is invoked.
        collective(*comm,
            [&]
            {
                for (const auto& packet : returned.packets)
                {
                    Reader reader{packet.bytes};
                    check_header(reader, context, page);
                    const auto support = reader.bytes_view();
                    const auto encoded = reader.bytes_view();
                    const auto revision = reader.word(), ordinal = reader.word(), size = reader.word(),
                               index = reader.word();
                    const auto payload = reader.bytes_view();
                    reader.done();
                    if (index >= expected.size())
                        throw std::invalid_argument("Foreign support fetch reply slot.");
                    auto& e = expected[index];
                    const auto* group = registry.find(e.group);
                    if (!group || e.member >= group->size())
                        throw std::invalid_argument("Stale support fetch group.");
                    const auto& member = (*group)[e.member];
                    if (e.seen || packet.peer != member.owner || revision != member.revision ||
                        ordinal != member.surface || size != member.encoded_size || payload.size() != size ||
                        !std::ranges::equal(support, registry.support_key(e.group)) ||
                        encoded.size() > budget.limit / 2)
                        throw std::invalid_argument("Mismatched or duplicate support fetch reply.");
                    page_memory.add(2 * encoded.size());
                    if (decode_adaptive_key(encoded) != registry.member_cell(member))
                        throw std::invalid_argument("Mismatched support fetch member key.");
                    e.payload = payload;
                    e.seen = true;
                }
                for (const auto& e : expected)
                    if (!e.seen)
                        throw std::invalid_argument("Incomplete support fetch page.");
            });
        collective(*comm,
            [&]
            {
                for (const auto& group : groups)
                {
                    Scope group_memory{budget};
                    IdentityBudget group_identities;
                    std::vector<RegistryMember> members;
                    std::vector<SupportOccurrence> occurrences;
                    const auto& support = registry.support_key(group.slot);
                    const auto* source = registry.find(group.slot);
                    if (group.count && (!source || source->size() != group.count))
                        throw std::invalid_argument("Support group changed during fetch.");
                    if (group.count > budget.limit / sizeof(RegistryMember) ||
                        group.count > budget.limit / sizeof(SupportOccurrence))
                        throw std::invalid_argument("Support group vectors exceed available arena.");
                    std::size_t group_bound = group.count * sizeof(RegistryMember);
                    add_size(group_bound, group.count * sizeof(SupportOccurrence));
                    for (std::size_t i = 0; i < group.count; ++i)
                    {
                        add_size(group_bound, (*source)[i].encoded_size);
                        add_size(group_bound, key_capacity(registry.member_cell((*source)[i])));
                    }
                    if (group_bound > budget.limit - budget.used)
                        throw std::invalid_argument("Complete support group cannot fit the remaining arena.");
                    group_memory.add(group.count * sizeof(RegistryMember));
                    group_memory.add(group.count * sizeof(SupportOccurrence));
                    members.reserve(group.count);
                    occurrences.reserve(group.count);
                    if (members.capacity() != group.count || occurrences.capacity() != group.count)
                        throw std::invalid_argument("Support group vector capacity differs from admission.");
                    for (std::size_t i = 0; i < group.count; ++i)
                    {
                        const auto& member = (*source)[i];
                        const auto& cell = registry.member_cell(member);
                        group_memory.add(key_capacity(cell));
                        members.push_back(registry.materialize(member));
                        const auto cap = budget.limit - budget.used;
                        if (!cap)
                            throw std::invalid_argument("Support geometry decode lease exhausted.");
                        group_memory.add(cap);
                        auto occurrence = decode_support_surface(
                            expected[group.first + i].payload, support.bytes(), member.owner, cap);
                        if (occurrence.cell != cell)
                            throw std::invalid_argument("Decoded support member changed.");
                        Budget measured{cap};
                        measured.add(key_capacity(occurrence.cell));
                        measured.add(occurrence.domain.capacity());
                        measured.add(occurrence.boundary.capacity());
                        auto admit_face = [&](const SupportFace& face)
                        {
                            if (face)
                            {
                                measured.add(sizeof(*face) + 4 * sizeof(void*));
                                group_identities.face(measured, *face);
                            }
                        };
                        admit_face(occurrence.face);
                        if (occurrence.whole_geometry != occurrence.face)
                            admit_face(occurrence.whole_geometry);
                        group_memory.drop(cap - measured.used);
                        occurrences.push_back(std::move(occurrence));
                    }
                    emit(support.bytes(), members, occurrences);
                }
            });
        flush();
        if (page == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Support fetch page identity exhausted.");
        ++page;
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
