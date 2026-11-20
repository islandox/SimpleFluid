#include "PooledLineExchange.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/BudgetedOwnerExchange.hh"
#include "geometry/mesh/AMR/ContactPointWire.hh"
#include <algorithm>
#include <cfenv>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using ReceiptAllocation = cartesian_detail::Allocation;
using ReceiptControl = contact_funding_detail::ControlCredit;
constexpr auto receipt_control_bytes = ReceiptControl::allowance();
bool original_member_equal(const PooledLineSubscriber& a, const PooledLineSubscriber& b)
{
    return a.resident == b.resident && a.support_authority == b.support_authority && a.support == b.support &&
           a.emission == b.emission && a.image == b.image && a.edge == b.edge &&
           a.source_generation == b.source_generation && a.native_source == b.native_source;
}
void header(Writer& w, unsigned layer, ContactHaloContext c, std::uint64_t kind, std::uint64_t wave, bool /*source*/)
{
    w.word(2);
    w.word(c.epoch);
    w.word(c.topology);
    w.word(c.partition);
    w.word(c.round);
    w.word(c.transaction);
    w.word(layer);
    w.word(wave);
    w.word(kind);
}
void header(Reader& r, unsigned layer, ContactHaloContext c, std::uint64_t kind, std::uint64_t wave, bool /*source*/)
{
    if (r.word() != 2 || r.word() != c.epoch || r.word() != c.topology || r.word() != c.partition ||
        r.word() != c.round || r.word() != c.transaction || r.word() != layer || r.word() != wave || r.word() != kind)
        throw std::invalid_argument("Invalid pooled line context.");
}
void subscriber(Writer& w, const PooledLineSubscriber& sub, const IncidencePools& pools, bool /*source*/)
{
    w.word(sub.resident);
    w.word(sub.support_authority);
    const auto support = pools.supports.get(sub.support);
    w.bytes(support.bytes());
    w.word(sub.emission);
    w.word(sub.image);
    w.word(sub.edge);
}
std::size_t intern_bytes(std::span<const char> bytes, IdentityPool<AdaptiveBytes>& pool, Budget& budget)
{
    budget.add(bytes.size());
    AdaptiveBytes value(bytes.begin(), bytes.end());
    if (value.capacity() > value.size())
        budget.add(value.capacity() - value.size());
    const auto charge = value.capacity();
    const auto result = pool.intern(std::move(value));
    budget.used -= charge;
    return result;
}
std::size_t intern_point(std::span<const char> bytes, IncidencePools& pools, Budget& budget)
{
    const auto reserved = budget.limit - budget.used;
    budget.add(reserved);
    auto value = PLIC::decode_identity<IncidencePools::Point>(bytes, reserved);
    const auto input = value.payload_bytes();
    if (input > reserved)
        throw std::invalid_argument("Decoded point exceeds live budget.");
    budget.used -= reserved - input;
    const auto result = pools.points.intern(std::move(value));
    budget.used -= input;
    return result;
}
PooledLineSubscriber subscriber(Reader& r, IncidencePools& pools, int ranks, bool /*source*/)
{
    PooledLineSubscriber sub;
    const auto resident = r.word(), owner = r.word();
    if (resident >= std::uint64_t(ranks) || owner >= std::uint64_t(ranks))
        throw std::invalid_argument("Invalid pooled line participant.");
    sub.resident = int(resident);
    sub.support_authority = int(owner);
    const auto support = r.bytes_view();
    if (support.empty() || authority(support, ranks) != sub.support_authority)
        throw std::invalid_argument("Invalid pooled line support.");
    sub.support = pools.supports.intern_bytes(support);
    sub.emission = r.word();
    const auto image = r.word();
    sub.edge = r.word();
    if (image > 1)
        throw std::invalid_argument("Invalid pooled line image.");
    sub.image = static_cast<std::uint8_t>(image);
    return sub;
}
std::size_t add_size(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Pooled line packet length overflow.");
    return a + b;
}
template<class T> struct TemporaryCredit
{
    Budget& budget;
    T value;
    std::size_t bytes = 0;
    explicit TemporaryCredit(Budget& b) : budget(b) {}
    ~TemporaryCredit() { reset(); }
    void reset()
    {
        {
            T retired;
            std::swap(retired, value);
        }
        budget.used -= bytes;
        bytes = 0;
    }
    void add(std::size_t amount)
    {
        const auto next = add_size(bytes, amount);
        budget.add(amount);
        bytes = next;
    }
    template<class U> void append(std::vector<U>& values)
    {
        const auto before = budget.used;
        try
        {
            budget.append_capacity(values);
        }
        catch (...)
        {
            bytes = add_size(bytes, budget.used - before);
            throw;
        }
        bytes = add_size(bytes, budget.used - before);
    }
    void observe(std::size_t before)
    {
        if (budget.used >= before)
            bytes = add_size(bytes, budget.used - before);
        else
            bytes -= before - budget.used;
    }
};
struct NativeStamp
{
    std::uint64_t generation = 0;
    bool native = false;
};
/** New control family. Existing version-two data packets are never modified.
 * SparseOwnerExchange preserves record order within each peer stream, including
 * segmented decoding; full member keys below additionally forbid reordering. */
constexpr std::uint64_t native_control_magic = 0x53464e4c434e5431ULL;
constexpr std::size_t native_control_header_bytes = 14 * 8;
struct NativeReference
{
    std::span<const char> line, support;
    std::array<std::uint64_t, 5> member{}; // resident/authority/emission/image/edge
    std::array<std::uint64_t, 3> page{};   // remove, or total/begin/count
};
NativeReference data_reference(
    std::span<const char> bytes, unsigned layer, ContactHaloContext context, std::uint64_t wave, bool reply)
{
    Reader reader{bytes};
    header(reader, layer, context, reply, wave, false);
    NativeReference ref;
    ref.line = reader.bytes_view();
    ref.member[0] = reader.word();
    ref.member[1] = reader.word();
    ref.support = reader.bytes_view();
    for (unsigned i = 2; i < ref.member.size(); ++i)
        ref.member[i] = reader.word();
    if (reply)
        for (auto& value : ref.page)
            value = reader.word();
    else
        ref.page[0] = reader.word();
    return ref;
}
std::size_t reference_bytes(const NativeReference& ref, bool reply)
{
    return add_size(reply ? 104 : 88, add_size(ref.line.size(), ref.support.size()));
}
void write_reference(Writer& writer, const NativeReference& ref, bool reply)
{
    writer.bytes(ref.line);
    writer.bytes(ref.support);
    for (auto value : ref.member)
        writer.word(value);
    for (unsigned i = 0; i < (reply ? 3U : 1U); ++i)
        writer.word(ref.page[i]);
}
NativeReference read_reference(Reader& reader, bool reply)
{
    NativeReference ref;
    ref.line = reader.bytes_view();
    ref.support = reader.bytes_view();
    for (auto& value : ref.member)
        value = reader.word();
    for (unsigned i = 0; i < (reply ? 3U : 1U); ++i)
        ref.page[i] = reader.word();
    return ref;
}
bool same_reference(const NativeReference& first, const NativeReference& second)
{
    return first.member == second.member && first.page == second.page && std::ranges::equal(first.line, second.line) &&
           std::ranges::equal(first.support, second.support);
}
void build_native_controls(TemporaryCredit<std::vector<OwnerPacket>>& controls, std::span<const OwnerPacket> data,
    std::span<const NativeStamp> stamps, unsigned layer, ContactHaloContext context, std::uint64_t wave,
    std::uint64_t page, bool reply, int sender, int ranks)
{
    if (data.size() != stamps.size() || ranks <= 0)
        throw std::invalid_argument("Native line control source inventory differs.");
    TemporaryCredit<std::vector<std::array<std::size_t, 2>>> peers(controls.budget);
    peers.add(std::size_t(ranks) * sizeof(peers.value[0]));
    peers.value.resize(ranks);
    if (peers.value.capacity() != std::size_t(ranks))
        throw std::invalid_argument("Native line peer capacity differs from admission.");
    for (const auto& packet : data)
    {
        if (packet.peer < 0 || packet.peer >= ranks)
            throw std::invalid_argument("Native line control peer is invalid.");
        const auto ref = data_reference(packet.bytes, layer, context, wave, reply);
        auto& peer = peers.value[packet.peer];
        ++peer[0];
        peer[1] = add_size(peer[1], reference_bytes(ref, reply));
    }
    controls.add(std::size_t(ranks) * sizeof(OwnerPacket));
    controls.value.reserve(ranks);
    if (controls.value.capacity() != std::size_t(ranks))
        throw std::invalid_argument("Native control packet capacity differs from admission.");
    for (int peer = 0; peer < ranks; ++peer)
    {
        const auto size = add_size(native_control_header_bytes, peers.value[peer][1]);
        controls.add(size);
        Writer writer;
        writer.data.reserve(size);
        if (writer.data.capacity() != size)
            throw std::invalid_argument("Native line control capacity differs from admission.");
        for (auto value : {native_control_magic, std::uint64_t(1), context.epoch, context.topology, context.partition,
                 context.round, context.transaction, std::uint64_t(layer), wave, std::uint64_t(reply ? 2 : 1), page,
                 std::uint64_t(sender), std::uint64_t(peer), std::uint64_t(peers.value[peer][0])})
            writer.word(value);
        std::size_t ordinal = 0;
        for (std::size_t i = 0; i < data.size(); ++i)
            if (data[i].peer == peer)
            {
                writer.word(ordinal++);
                writer.word(stamps[i].generation);
                writer.word(stamps[i].native);
                write_reference(writer, data_reference(data[i].bytes, layer, context, wave, reply), reply);
            }
        if (writer.data.size() != size)
            throw std::logic_error("Native line control framing differs from admission.");
        controls.value.push_back({peer, std::move(writer.data)});
    }
}
void match_native_controls(TemporaryCredit<std::vector<NativeStamp>>& stamps, std::span<const OwnerPacket> data,
    std::span<const OwnerPacket> controls, unsigned layer, ContactHaloContext context, std::uint64_t wave,
    std::uint64_t page, bool reply, int receiver, int ranks)
{
    if (controls.size() != std::size_t(ranks))
        throw std::invalid_argument("Missing native line peer completion, including empty peers.");
    const auto n = data.size(), peers = std::size_t(ranks), absent = std::numeric_limits<std::size_t>::max();
    if (peers > (std::numeric_limits<std::size_t>::max() - n) / 3 ||
        n + 3 * peers > stamps.budget.limit / sizeof(std::size_t) || n > stamps.budget.limit / sizeof(NativeStamp))
        throw std::invalid_argument("Native line control lookup capacity overflows.");
    TemporaryCredit<std::vector<std::size_t>> lookup(stamps.budget);
    lookup.add((n + 3 * peers) * sizeof(std::size_t));
    lookup.value.resize(n + 3 * peers);
    if (lookup.value.capacity() != n + 3 * peers)
        throw std::invalid_argument("Native control lookup capacity differs from admission.");
    auto* count = lookup.value.data();
    auto* head = count + peers;
    auto* seen = head + peers;
    auto* next = seen + peers;
    std::fill(head, head + peers, absent);
    for (std::size_t i = n; i--;)
    {
        const auto peer = data[i].peer;
        if (peer < 0 || peer >= ranks)
            throw std::invalid_argument("Invalid native line data peer.");
        next[i] = head[peer];
        head[peer] = i;
        ++count[peer];
    }
    stamps.add(n * sizeof(NativeStamp));
    stamps.value.resize(n);
    if (stamps.value.capacity() != n)
        throw std::invalid_argument("Native stamp capacity differs from admission.");
    for (const auto& packet : controls)
    {
        Reader reader{packet.bytes};
        for (auto expected :
            {native_control_magic, std::uint64_t(1), context.epoch, context.topology, context.partition, context.round,
                context.transaction, std::uint64_t(layer), wave, std::uint64_t(reply ? 2 : 1), page})
            if (reader.word() != expected)
                throw std::invalid_argument("Stale native line control context.");
        const auto sender = reader.word(), destination = reader.word(), records = reader.word();
        if (packet.peer < 0 || packet.peer >= ranks || sender != std::uint64_t(packet.peer) ||
            destination != std::uint64_t(receiver) || seen[packet.peer]++ || records != count[packet.peer])
            throw std::invalid_argument("Duplicate, missing or foreign native line peer completion.");
        auto index = head[packet.peer];
        for (std::size_t ordinal = 0; ordinal < records; ++ordinal)
        {
            if (reader.word() != ordinal || index == absent)
                throw std::invalid_argument("Reordered native line control ordinal.");
            const auto generation = reader.word(), native = reader.word();
            if (native > 1 || (native && !generation))
                throw std::invalid_argument("Invalid native source stamp.");
            if (!same_reference(
                    read_reference(reader, reply), data_reference(data[index].bytes, layer, context, wave, reply)))
                throw std::invalid_argument("Native line reference differs from full version-two member identity.");
            stamps.value[index] = {generation, bool(native)};
            index = next[index];
        }
        reader.done();
        if (index != absent)
            throw std::invalid_argument("Incomplete native line page binding.");
    }
    if (std::find(seen, seen + peers, 0) != seen + peers)
        throw std::invalid_argument("Missing empty native line peer completion.");
}
void transfer_packets(TemporaryCredit<std::vector<OwnerPacket>>& outgoing, TemporaryCredit<OwnerExchange>& incoming,
    Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    const auto before = outgoing.budget.used;
    try
    {
        incoming.value =
            consume_budgeted_owner_packets(std::move(outgoing.value), comm, outgoing.budget, outgoing.bytes);
    }
    catch (...)
    {
        outgoing.observe(before);
        throw;
    }
    outgoing.observe(before);
    incoming.bytes = std::exchange(outgoing.bytes, 0);
}
void inject_native_fault(TemporaryCredit<std::vector<OwnerPacket>>& controls, PooledNativeLineFault fault, bool reply,
    int rank, std::uint64_t wave, std::uint64_t page)
{
    if (fault.kind == PooledNativeLineFault::Kind::None || fault.reply != reply || fault.rank != rank ||
        fault.wave != wave || fault.page != page || controls.value.empty())
        return;
    switch (fault.kind)
    {
        case PooledNativeLineFault::Kind::MissingCompletion: {
            const auto released = controls.value.back().bytes.capacity();
            controls.value.pop_back(); // Retain capacity until the admitted batch dies.
            controls.budget.used -= released;
            controls.bytes -= released;
            break;
        }
        case PooledNativeLineFault::Kind::DuplicateCompletion: {
            const auto bytes = controls.value.front().bytes.size();
            controls.add(bytes);
            controls.append(controls.value);
            auto duplicate = controls.value.front();
            controls.value.push_back(std::move(duplicate));
            break;
        }
        case PooledNativeLineFault::Kind::StaleContext:
            controls.value.front().bytes.at(16) ^= 1; // Exact epoch word.
            break;
        case PooledNativeLineFault::Kind::TruncatedCompletion:
            controls.value.front().bytes.pop_back(); // Existing admitted capacity is retained.
            break;
        case PooledNativeLineFault::Kind::ReorderedReference:
            for (auto& packet : controls.value)
                if (packet.bytes.size() > native_control_header_bytes)
                {
                    packet.bytes[native_control_header_bytes] ^= 1; // First ordinal must be zero.
                    break;
                }
            break;
        default:
            break;
    }
}
std::size_t edit_size(const PooledLineEdit& edit, const IncidencePools& pools, bool /*source*/)
{
    std::size_t size = add_size(136, pools.lines.get(edit.line).size());
    size = add_size(size, pools.supports.get(edit.subscriber.support).size());
    if (!edit.remove)
    {
        size = add_size(size, 16);
        size = add_size(size, PLIC::encoded_identity_size(pools.points.get(edit.subscriber.first)));
        size = add_size(size, PLIC::encoded_identity_size(pools.points.get(edit.subscriber.last)));
    }
    return size;
}
std::size_t live_page_credit(const Teuchos::Comm<int>& comm, const Budget& budget, std::size_t requested)
{
    unsigned long long local = budget.limit - budget.used, common = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local, &common);
    // Every source may target one receiver. Keep room for that fan-in, sender
    // and receiver framing, simultaneous decode, and persistent decode growth.
    // SparseOwnerExchange still performs its exact allocation preflight.
    return std::min<std::size_t>(requested, common / static_cast<unsigned>(comm.getSize()) / 8);
}
/** Independently versioned controls for read-only cached native query receipts.
 * These envelopes never enter the original version-two data streams. Every
 * wave/page includes one envelope for EVERY peer, including empty senders. */
constexpr std::uint64_t cached_control_magic = 0x5346434c52444331ULL;
constexpr std::size_t cached_header_bytes = 14 * 8;
void cached_header(Writer& writer, unsigned layer, ContactHaloContext context, std::uint64_t wave, std::uint64_t page,
    bool reply, int sender, int destination, bool record)
{
    for (auto word : {cached_control_magic, std::uint64_t(1), std::uint64_t(reply ? 2 : 1), context.epoch,
             context.topology, context.partition, context.round, context.transaction, std::uint64_t(layer), wave, page,
             std::uint64_t(sender), std::uint64_t(destination), std::uint64_t(record)})
        writer.word(word);
}
bool cached_header(Reader& reader, unsigned layer, ContactHaloContext context, std::uint64_t wave, std::uint64_t page,
    bool reply, int sender, int destination)
{
    for (auto expected : {cached_control_magic, std::uint64_t(1), std::uint64_t(reply ? 2 : 1), context.epoch,
             context.topology, context.partition, context.round, context.transaction, std::uint64_t(layer), wave, page,
             std::uint64_t(sender), std::uint64_t(destination)})
        if (reader.word() != expected)
            throw std::invalid_argument("Stale or foreign cached native line control.");
    const auto records = reader.word();
    if (records > 1)
        throw std::invalid_argument("Invalid cached native line control count.");
    return records;
}
std::size_t cached_request_bytes(const PooledLineCompletionRequest& request, const IncidencePools& pools)
{
    auto size = add_size(
        88, add_size(pools.lines.get(request.line).size(), pools.supports.get(request.original.support).size()));
    size = add_size(size, PLIC::encoded_identity_size(pools.points.get(request.original.first)));
    return add_size(size, PLIC::encoded_identity_size(pools.points.get(request.original.last)));
}
void write_cached_point(Writer& writer, const IncidencePools::Point& point)
{
    append_framed_contact_point_identity(writer, point);
}
void write_cached_request(
    Writer& writer, const PooledLineCompletionRequest& request, const IncidencePools& pools)
{
    writer.bytes(pools.lines.get(request.line));
    subscriber(writer, request.original, pools, false);
    writer.word(request.original.source_generation);
    writer.word(request.original.native_source);
    write_cached_point(writer, pools.points.get(request.original.first));
    write_cached_point(writer, pools.points.get(request.original.last));
}
std::size_t find_cached_bytes(std::span<const char> bytes, const IdentityPool<AdaptiveBytes>& pool, Budget& budget)
{
    ReceiptAllocation scratch(budget);
    AdaptiveBytes key;
    scratch.copy(key, bytes);
    const auto slot = pool.find(key);
    if (!slot)
        throw std::invalid_argument("Unknown full key in cached native line request.");
    return *slot;
}
std::size_t read_cached_point(std::span<const char> bytes, IncidencePools& pools, Budget& budget, bool existing)
{
    // Ordinary canonical point decoding owns no expanded geometry or support
    // tree. Admit its complete packet while checking the inline canonical value.
    ReceiptAllocation scratch(budget);
    scratch.retain(bytes.size());
    auto point = PLIC::decode_identity<IncidencePools::Point>(bytes, bytes.size());
    if (point.payload_bytes())
        throw std::invalid_argument("Unexpected retained payload in ordinary cached point decoding.");
    if (existing)
    {
        const auto slot = pools.points.find(point);
        if (!slot)
            throw std::invalid_argument("Unknown original endpoint in cached native line request.");
        return *slot;
    }
    return pools.points.intern(std::move(point));
}
PooledLineCompletionRequest read_cached_request(Reader& reader, IncidencePools& pools, Budget& budget, int ranks)
{
    PooledLineCompletionRequest request;
    const auto line = reader.bytes_view();
    if (line.empty())
        throw std::invalid_argument("Empty cached native line key.");
    request.line = find_cached_bytes(line, pools.lines, budget);
    auto& member = request.original;
    const auto resident = reader.word(), support_authority = reader.word();
    const auto support = reader.bytes_view();
    if (resident >= std::uint64_t(ranks) || support_authority >= std::uint64_t(ranks) || support.empty() ||
        authority(support, ranks) != int(support_authority))
        throw std::invalid_argument("Invalid cached native support participant or full key.");
    member.resident = int(resident);
    member.support_authority = int(support_authority);
    const auto support_slot = pools.supports.find_bytes(support);
    if (!support_slot)
        throw std::invalid_argument("Unknown full key in cached native line request.");
    member.support = *support_slot;
    member.emission = reader.word();
    const auto image = reader.word();
    member.edge = reader.word();
    member.source_generation = reader.word();
    const auto native = reader.word();
    if (image > 1 || native != 1 || !member.source_generation)
        throw std::invalid_argument("Invalid cached native member generation.");
    member.image = std::uint8_t(image);
    member.native_source = true;
    member.first = read_cached_point(reader.bytes_view(), pools, budget, true);
    member.last = read_cached_point(reader.bytes_view(), pools, budget, true);
    return request;
}
bool cached_same_request(const PooledLineCompletionRequest& a, const PooledLineCompletionRequest& b)
{
    return a.line == b.line && original_member_equal(a.original, b.original) && a.original.first == b.original.first &&
           a.original.last == b.original.last;
}
using CachedMemberKey = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, int, int, std::uint64_t,
    std::uint8_t, std::uint64_t, std::uint64_t>;
CachedMemberKey cached_member_key(const PooledLineCompletionRequest& request)
{
    const auto& member = request.original;
    return {request.line, member.support, member.first, member.last, member.resident, member.support_authority,
        member.emission, member.image, member.edge, member.source_generation};
}
void remember_cached_member(
    TemporaryCredit<std::set<CachedMemberKey>>& seen, const PooledLineCompletionRequest& request)
{
    const auto key = cached_member_key(request);
    if (seen.value.contains(key))
        throw std::invalid_argument("Duplicate cached native member request.");
    seen.add(sizeof(CachedMemberKey) + 4 * sizeof(void*));
    seen.value.insert(key);
}
void cached_reserve_packets(TemporaryCredit<std::vector<OwnerPacket>>& packets, int ranks)
{
    packets.add(cartesian_detail::product(std::size_t(ranks), sizeof(OwnerPacket)));
    packets.value.reserve(ranks);
    if (packets.value.capacity() != std::size_t(ranks))
        throw std::invalid_argument("Cached control peer capacity differs from admission.");
}
template<class Write>
void cached_packet(TemporaryCredit<std::vector<OwnerPacket>>& packets, std::size_t bytes, int peer, Write&& write)
{
    packets.add(bytes);
    Writer writer;
    writer.data.reserve(bytes);
    if (writer.data.capacity() != bytes)
        throw std::invalid_argument("Cached line control capacity differs from admission.");
    write(writer);
    if (writer.data.size() != bytes)
        throw std::logic_error("Cached line control framing differs from admission.");
    packets.value.push_back({peer, std::move(writer.data)});
}
void cached_word(std::vector<char>& bytes, std::size_t offset, std::uint64_t word)
{
    if (offset > bytes.size() || 8 > bytes.size() - offset)
        throw std::invalid_argument("Cached fault target is truncated.");
    for (unsigned i = 0; i < 8; ++i)
        bytes[offset + i] = char(word >> (8 * i));
}
void inject_cached_fault(TemporaryCredit<std::vector<OwnerPacket>>& packets, PooledCachedLineFault fault, bool reply,
    int rank, std::uint64_t wave, std::uint64_t page)
{
    if (fault.kind == PooledCachedLineFault::Kind::None || fault.reply != reply || fault.rank != rank ||
        fault.wave != wave || fault.page != page || packets.value.empty())
        return;
    auto index = packets.value.size();
    for (std::size_t i = 0; i < packets.value.size(); ++i)
    {
        const auto& packet = packets.value[i];
        if (fault.peer >= 0 && packet.peer != fault.peer)
            continue;
        const bool empty = packet.bytes.size() == cached_header_bytes;
        if ((fault.kind == PooledCachedLineFault::Kind::InvalidEmptyCount && !empty) ||
            ((fault.kind == PooledCachedLineFault::Kind::StaleGeneration ||
                 fault.kind == PooledCachedLineFault::Kind::DuplicateChunk ||
                 fault.kind == PooledCachedLineFault::Kind::MissingCompletion) &&
                empty))
            continue;
        index = i;
        break;
    }
    if (index == packets.value.size())
        return;
    auto& packet = packets.value[index];
    switch (fault.kind)
    {
        case PooledCachedLineFault::Kind::MissingEnvelope: {
            const auto released = packet.bytes.capacity();
            packets.value.erase(packets.value.begin() + index);
            packets.budget.used -= released;
            packets.bytes -= released;
            break;
        }
        case PooledCachedLineFault::Kind::DuplicateEnvelope: {
            const auto bytes = packet.bytes.size();
            packets.add(bytes);
            packets.append(packets.value);
            auto duplicate = packets.value[index];
            packets.value.push_back(std::move(duplicate));
            break;
        }
        case PooledCachedLineFault::Kind::StaleContext:
            packet.bytes.at(24) ^= 1; // first context word in this distinct family
            break;
        case PooledCachedLineFault::Kind::TruncatedEnvelope:
            packet.bytes.pop_back();
            break;
        case PooledCachedLineFault::Kind::InvalidEmptyCount:
            cached_word(packet.bytes, cached_header_bytes - 8, 1);
            break;
        case PooledCachedLineFault::Kind::MissingCompletion:
            cached_word(packet.bytes, cached_header_bytes - 8, 0);
            packet.bytes.resize(cached_header_bytes); // capacity remains fully admitted
            break;
        case PooledCachedLineFault::Kind::StaleGeneration:
        case PooledCachedLineFault::Kind::DuplicateChunk: {
            Reader reader{packet.bytes};
            reader.offset = cached_header_bytes;
            (void) reader.bytes_view();
            (void) reader.word();
            (void) reader.word();
            (void) reader.bytes_view();
            for (unsigned i = 0; i < 3; ++i)
                (void) reader.word();
            if (fault.kind == PooledCachedLineFault::Kind::StaleGeneration)
                packet.bytes.at(reader.offset) ^= 1;
            else
            {
                (void) reader.word();
                (void) reader.word();
                (void) reader.bytes_view();
                (void) reader.bytes_view();
                (void) reader.word(); // actual kernel witness
                (void) reader.word(); // total
                const auto offset = reader.offset;
                const auto begin = reader.word();
                cached_word(packet.bytes, offset, begin ? 0 : 1); // guaranteed different chunk offset
            }
            break;
        }
        default:
            break;
    }
}
class CachedMetadataEnvironment
{
    std::fenv_t caller_{};

public:
    CachedMetadataEnvironment()
    {
        if (std::feholdexcept(&caller_))
            throw std::invalid_argument("Cannot isolate cached native control metadata.");
    }
    ~CachedMetadataEnvironment() { (void) std::fesetenv(&caller_); }
};
} // namespace
PooledLineCompletion::Proof::Proof(
    PooledLineSubscriber original, ContactHaloContext context, unsigned layer, const IncidencePools& pools)
    : original_(std::move(original)), context_(context), layer_(layer), kernel_layer_(layer),
      endpoints_{pools.points.get(original_.first), pools.points.get(original_.last)}, point_pool_(&pools.points)
{
}
PooledLineCompletion::Proof::~Proof()
{
    AdaptiveBytes{}.swap(line_);
    AdaptiveBytes{}.swap(support_);
    if (budget_)
        budget_->used -= charge_;
}
std::size_t PooledLineCompletion::Proof::metadata_bytes() const noexcept
{
    return charge_ + receipt_control_bytes;
}
NativeContactOwnerFunding PooledLineCompletion::Proof::native_funding(
    const std::shared_ptr<const Proof>& owner, std::size_t bytes) const
{
    return NativeContactFundingAccess::split_named(owner, bytes, [](const Proof& value) { return value.charge_; },
        [](const Proof& value) { return static_cast<const ReceiptControl*>(value.control_); });
}
std::size_t PooledLineCompletion::Proof::known_payload_bytes() const noexcept
{
    return sizeof(Proof) + sizeof(ReceiptControl) + line_.capacity() + support_.capacity();
}
std::pair<const void*, std::size_t> PooledLineCompletion::Proof::borrowed_control_payload() const noexcept
{
    return {control_, sizeof(ReceiptControl)};
}
void PooledLineCompletion::Proof::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(visitor, this, ContactPayloadRole::PooledLineCompletionProofObject, sizeof(*this), true);
    contact_payload_object(
        visitor, control_, ContactPayloadRole::CartesianControlCreditObject, sizeof(ReceiptControl), true);
    contact_payload_buffer(visitor, this, ContactPayloadRole::PooledLineCompletionKeys, line_);
    contact_payload_buffer(visitor, this, ContactPayloadRole::PooledLineCompletionKeys, support_);
}
PooledLineCompletion::PooledLineCompletion(std::shared_ptr<const Proof> proof, std::vector<std::size_t>&& result,
    Budget& budget, std::shared_ptr<Budget> keeper)
    : proof_(std::move(proof)), result_(std::move(result)), output_budget_(&budget), output_keeper_(std::move(keeper)),
      output_charge_(result_.capacity() * sizeof(std::size_t))
{
}
PooledLineCompletion::PooledLineCompletion(PooledLineCompletion&& other) noexcept
    : proof_(std::move(other.proof_)), result_(std::move(other.result_)),
      output_budget_(std::exchange(other.output_budget_, nullptr)), output_keeper_(std::move(other.output_keeper_)),
      output_charge_(std::exchange(other.output_charge_, 0)), wrapper_charge_(std::exchange(other.wrapper_charge_, 0)),
      wrapper_control_(std::exchange(other.wrapper_control_, nullptr))
{
}
PooledLineCompletion& PooledLineCompletion::operator=(PooledLineCompletion&& other) noexcept
{
    if (this != &other)
    {
        std::vector<std::size_t>{}.swap(result_);
        if (output_budget_)
            output_budget_->used -= output_charge_ + wrapper_charge_;
        proof_ = std::move(other.proof_);
        result_ = std::move(other.result_);
        output_budget_ = std::exchange(other.output_budget_, nullptr);
        output_keeper_ = std::move(other.output_keeper_);
        output_charge_ = std::exchange(other.output_charge_, 0);
        wrapper_charge_ = std::exchange(other.wrapper_charge_, 0);
        wrapper_control_ = std::exchange(other.wrapper_control_, nullptr);
    }
    return *this;
}
PooledLineCompletion::~PooledLineCompletion()
{
    std::vector<std::size_t>{}.swap(result_);
    if (output_budget_)
        output_budget_->used -= output_charge_ + wrapper_charge_;
}
PooledLineCompletionView PooledLineCompletion::share_immutable() &&
{
    if (!output_budget_ || !proof_ || wrapper_control_)
        throw std::invalid_argument("Completed line reply has no unshared receiver-owned result.");
    ReceiptAllocation body(*output_budget_);
    body.retain(sizeof(PooledLineCompletion));
    auto [result, control] = contact_funding_detail::make_controlled(*output_budget_, output_keeper_,
        [&] { return new PooledLineCompletion(std::move(*this)); });
    result->wrapper_control_ = control.get();
    result->wrapper_charge_ = std::exchange(body.bytes, 0);
    return result;
}
std::size_t PooledLineCompletion::metadata_bytes() const noexcept
{
    return output_charge_ + wrapper_charge_ + (wrapper_control_ ? receipt_control_bytes : 0);
}
NativeContactOwnerFunding PooledLineCompletion::native_funding(
    const PooledLineCompletionView& owner, std::size_t bytes) const
{
    return NativeContactFundingAccess::split_named(owner, bytes,
        [](const PooledLineCompletion& value) { return value.output_charge_ + value.wrapper_charge_; },
        [](const PooledLineCompletion& value) { return static_cast<const ReceiptControl*>(value.wrapper_control_); });
}
std::size_t PooledLineCompletion::known_payload_bytes() const noexcept
{
    return (wrapper_control_ ? sizeof(PooledLineCompletion) + sizeof(ReceiptControl) : 0) +
           (proof_ ? proof_->known_payload_bytes() : 0);
}
std::pair<const void*, std::size_t> PooledLineCompletion::borrowed_control_payload() const noexcept
{
    return {wrapper_control_, wrapper_control_ ? sizeof(ReceiptControl) : 0};
}
void PooledLineCompletion::retain_dependencies(cartesian_detail::CreditConstruction& credits,
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases,
    cartesian_detail::Allocation& memory) const
{
    credits.retain(
        proof_, proof_->allocation_budget(), proof_->metadata_bytes(), proof_->has_arena_keeper(), leases, memory);
}
bool PooledLineCompletion::imports_owner(const void* id) const noexcept
{
    return id == this || id == proof_.get();
}
PooledLineCompletion PooledLineCompletion::seal(std::size_t line, PooledLineSubscriber original,
    std::vector<std::size_t>&& result, ContactHaloContext context, unsigned layer, const IncidencePools& pools,
    Budget& budget, std::shared_ptr<Budget> keeper, const std::shared_ptr<CartesianAuthorityArena>& arena,
    unsigned kernel_witness_layer, bool cached_read)
{
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget)
        throw std::invalid_argument("Completed line replies require their explicit destination arena.");
    cartesian_detail::check_keeper(budget, arena->keeper);
    if (!keeper)
        keeper = arena->keeper;
    if (!keeper || !original.native_source || original.has_result() || original.result.capacity() ||
        kernel_witness_layer > layer || (!cached_read && kernel_witness_layer != layer))
        throw std::invalid_argument("Completed line replies require an original native request and lifetime keeper.");
    ReceiptAllocation body(budget);
    body.retain(sizeof(Proof));
    auto [proof, control] = contact_funding_detail::make_controlled(budget, keeper,
        [&] { return new Proof(std::move(original), context, layer, pools); });
    proof->control_ = control.get();
    proof->kernel_layer_ = kernel_witness_layer;
    proof->cached_read_ = cached_read;
    body.copy(proof->line_, pools.lines.get(line));
    const auto support = pools.supports.get(proof->original_.support);
    body.copy(proof->support_, support.bytes());
    proof->keeper_ = keeper;
    proof->charge_ = std::exchange(body.bytes, 0);
    proof->budget_ = &budget;
    return PooledLineCompletion(std::move(proof), std::move(result), budget, std::move(keeper));
}
void PooledLineCompletion::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    if (wrapper_control_)
    {
        contact_payload_object(visitor, this, ContactPayloadRole::PooledLineCompletionObject, sizeof(*this), true);
        contact_payload_object(
            visitor, wrapper_control_, ContactPayloadRole::CartesianControlCreditObject, sizeof(ReceiptControl), true);
    }
    if (proof_)
        proof_->visit_borrowed_payload(visitor);
    contact_payload_buffer(visitor, this, ContactPayloadRole::PooledLineCompletionResults, result_);
}
PooledLineBorrowedCompletion::PooledLineBorrowedCompletion(std::size_t line, const PooledLineSubscriber& original,
    std::span<const std::size_t> result, ContactHaloContext context, unsigned layer, const IncidencePools& pools)
    : line_(line), original_(original), result_(result), context_(context), layer_(layer), pools_(pools),
      support_(pools.supports.get(original.support))
{
}
std::span<const char> PooledLineBorrowedCompletion::line_key() const { return pools_.lines.get(line_); }
std::span<const char> PooledLineBorrowedCompletion::support_key() const { return support_.bytes(); }
ContactSourcePoint PooledLineBorrowedCompletion::endpoint(unsigned i) const
{
    if (i > 1)
        throw std::out_of_range("Invalid borrowed line endpoint ordinal.");
    return pools_.points.get(i ? original_.last : original_.first);
}
const void* PooledLineBorrowedCompletion::point_pool_identity() const noexcept { return &pools_.points; }
void exchange_pooled_line_edits(PooledLineRegistry& registry, IncidencePools& pools, const NextPooledLineEdit& next,
    unsigned layer, ContactHaloContext context, Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget,
    std::size_t page_bytes, const PooledLineReply& publish, const PooledLineBeforeSend& before_send,
    bool source_protocol, PooledNativeLineFault source_fault, const PooledLineDataObserve& observe_data,
    const PooledLineOriginalRequest& original_request, const PooledLineCompletionReceive& completed_reply,
    std::shared_ptr<Budget> completion_keeper, std::shared_ptr<CartesianAuthorityArena> completion_arena,
    const PooledLineCompletionConsume& consume_completion)
{
    if (comm.is_null())
        throw std::invalid_argument("Pooled line exchange requires communicator.");
    collective(*comm,
        [&]
        {
            if (!page_bytes)
                throw std::invalid_argument("Invalid pooled line page size.");
            if (consume_completion && !completed_reply)
                throw std::invalid_argument("Borrowed completion consumers require the normal receipt fallback.");
            if (completed_reply)
            {
                cartesian_detail::check_keeper(budget, completion_keeper);
                if (!source_protocol || !original_request || !completion_arena || completion_arena->budget != &budget)
                    throw std::invalid_argument(
                        "Completed line replies require native controls, original facts and arena.");
                cartesian_detail::check_keeper(budget, completion_arena->keeper);
                if (!completion_keeper)
                    completion_keeper = completion_arena->keeper;
                if (!completion_keeper)
                    throw std::invalid_argument("Completed line reply lifetime requires a budget keeper.");
            }
            registry.set_layer(layer);
        });
    int source_local = source_protocol, source_min = 0, source_max = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &source_local, &source_min);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &source_local, &source_max);
    if (source_min != source_max)
        throw std::invalid_argument("Inconsistent native line control option.");
    bool ended = false;
    TemporaryCredit<std::vector<PooledLineEdit>> staged_memory(budget);
    auto& staged = staged_memory.value;
    auto& staged_charge = staged_memory.bytes;
    std::size_t staged_cursor = 0;
    std::uint64_t edit_page = 0;
    for (;;)
    {
        if (staged_cursor == staged.size())
        {
            std::vector<PooledLineEdit>().swap(staged);
            budget.used -= staged_charge;
            staged_charge = staged_cursor = 0;
        }
        const auto staging_target = live_page_credit(*comm, budget, page_bytes);
        collective(*comm,
            [&]
            {
                if (!staged.empty() || ended)
                    return;
                std::size_t wire_bytes = 0;
                for (;;)
                {
                    PooledLineEdit item;
                    if (!next(item))
                    {
                        ended = true;
                        break;
                    }
                    if (item.subscriber.resident != comm->getRank() || item.subscriber.image > 1 ||
                        item.subscriber.has_result() || !item.subscriber.result.empty())
                        throw std::invalid_argument("Invalid local pooled line edit.");
                    const auto bytes = edit_size(item, pools, source_protocol);
                    if (bytes > page_bytes || sizeof(OwnerPacket) > page_bytes - bytes)
                        throw std::invalid_argument("Pooled line edit exceeds page limit.");
                    const auto before = budget.used;
                    budget.append_capacity(staged);
                    staged_charge += budget.used - before;
                    staged.push_back(std::move(item));
                    wire_bytes = add_size(wire_bytes, bytes + sizeof(OwnerPacket));
                    const auto local_credit = (budget.limit - budget.used) / static_cast<unsigned>(comm->getSize()) / 8;
                    if (wire_bytes >= std::min(staging_target, local_credit))
                        break;
                }
            });
        // A cursor can create persistent tile, identity and subscription state.
        // Refresh only after that bounded batch is staged, then encode without
        // advancing the cursor. Unsent compact edits remain charged for the
        // next page; no stale pre-cursor credit controls the wire allocation.
        const auto wire_credit = live_page_credit(*comm, budget, page_bytes);
        TemporaryCredit<std::vector<OwnerPacket>> outgoing_memory(budget), controls(budget);
        TemporaryCredit<std::vector<NativeStamp>> outgoing_stamps(budget), incoming_stamps(budget);
        auto& outgoing = outgoing_memory.value;
        auto& charge = outgoing_memory.bytes;
        std::size_t page_used = 0;
        collective(*comm,
            [&]
            {
                while (staged_cursor < staged.size())
                {
                    const auto& item = staged[staged_cursor];
                    const auto bytes = edit_size(item, pools, source_protocol);
                    const auto cost = add_size(bytes, sizeof(OwnerPacket));
                    if (cost > wire_credit)
                        throw std::invalid_argument("Pooled line edit exceeds live page credit: rank=" +
                            std::to_string(comm->getRank()) + " used=" + std::to_string(budget.used) +
                            " request=" + std::to_string(cost) + " limit=" + std::to_string(budget.limit) +
                            " wire_credit=" + std::to_string(wire_credit) +
                            " cumulative_spent=" + std::to_string(budget.cumulative) + ".");
                    if (cost > wire_credit - page_used)
                        break;
                    page_used += cost;
                    budget.add(bytes);
                    charge += bytes;
                    Writer w;
                    w.data.reserve(bytes);
                    if (w.data.capacity() != bytes)
                        throw std::invalid_argument("Line edit capacity mismatch.");
                    header(w, layer, context, 0, 0, source_protocol);
                    w.bytes(pools.lines.get(item.line));
                    subscriber(w, item.subscriber, pools, source_protocol);
                    w.word(item.remove);
                    if (!item.remove)
                    {
                        append_framed_contact_point_identity(w, pools.points.get(item.subscriber.first));
                        append_framed_contact_point_identity(w, pools.points.get(item.subscriber.last));
                    }
                    if (w.data.size() != bytes)
                        throw std::invalid_argument("Line edit size mismatch.");
                    const auto before = budget.used;
                    budget.append_capacity(outgoing);
                    charge += budget.used - before;
                    outgoing.push_back({authority(pools.lines.get(item.line), comm->getSize()), std::move(w.data)});
                    if (source_protocol)
                    {
                        outgoing_stamps.append(outgoing_stamps.value);
                        outgoing_stamps.value.push_back(
                            {item.subscriber.source_generation, item.subscriber.native_source});
                    }
                    ++staged_cursor;
                }
            });
        collective(*comm,
            [&]
            {
                if (observe_data)
                    observe_data(false, layer, 0, edit_page, outgoing);
                if (source_protocol)
                {
                    build_native_controls(controls, outgoing, outgoing_stamps.value, layer, context, 0, edit_page,
                        false, comm->getRank(), comm->getSize());
                    inject_native_fault(controls, source_fault, false, comm->getRank(), 0, edit_page);
                }
            });
        int local = !outgoing.empty(), any = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &any);
        TemporaryCredit<OwnerExchange> incoming_memory(budget), received_controls(budget);
        auto& incoming = incoming_memory.value;
        if (any)
            transfer_packets(outgoing_memory, incoming_memory, comm);
        if (source_protocol)
        {
            transfer_packets(controls, received_controls, comm);
            collective(*comm,
                [&]
                {
                    match_native_controls(incoming_stamps, incoming.packets, received_controls.value.packets, layer,
                        context, 0, edit_page, false, comm->getRank(), comm->getSize());
                });
        }
        if (!any)
            break;
        collective(*comm,
            [&]
            {
                for (std::size_t packet_index = 0; packet_index < incoming.packets.size(); ++packet_index)
                {
                    const auto& packet = incoming.packets[packet_index];
                    Reader r{packet.bytes};
                    header(r, layer, context, 0, 0, source_protocol);
                    const auto line = r.bytes_view();
                    if (line.empty() || authority(line, comm->getSize()) != comm->getRank())
                        throw std::invalid_argument("Foreign pooled line edit.");
                    PooledLineEdit item;
                    item.line = intern_bytes(line, pools.lines, budget);
                    item.subscriber = subscriber(r, pools, comm->getSize(), source_protocol);
                    if (source_protocol)
                    {
                        item.subscriber.source_generation = incoming_stamps.value[packet_index].generation;
                        item.subscriber.native_source = incoming_stamps.value[packet_index].native;
                    }
                    const auto remove = r.word();
                    if (remove > 1 || item.subscriber.resident != packet.peer)
                        throw std::invalid_argument("Invalid pooled line edit source.");
                    item.remove = remove;
                    if (!item.remove)
                    {
                        item.subscriber.first = intern_point(r.bytes_view(), pools, budget);
                        item.subscriber.last = intern_point(r.bytes_view(), pools, budget);
                    }
                    r.done();
                    registry.apply(std::span<const PooledLineEdit>(&item, 1));
                }
            });
        incoming_memory.reset();
        if (edit_page == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Native line edit page identity exhausted.");
        ++edit_page;
    }
    std::uint64_t wave = 0;
    // Query one complete line at a time, then stream its changed replies in
    // bounded chunks. The registry owns packed reply bodies while only query
    // ordinals are staged here; even one long segment is fragmented on wire.
    for (;;)
    {
        const auto dirty = registry.next_dirty();
        int local = dirty.has_value(), any = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &any);
        if (!any)
            break;
        TemporaryCredit<std::vector<std::size_t>> changed_memory(budget);
        auto& changed = changed_memory.value;
        auto& changed_charge = changed_memory.bytes;
        collective(*comm,
            [&]
            {
                if (dirty)
                    registry.recompute_one(*dirty,
                        [&](std::size_t, std::size_t subscriber)
                        {
                            const auto before = budget.used;
                            budget.append_capacity(changed);
                            changed_charge += budget.used - before;
                            changed.push_back(subscriber);
                        });
            });
        using ReplyKey =
            std::tuple<std::size_t, std::size_t, int, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool>;
        struct Partial
        {
            PooledLineSubscriber subscriber;
            std::size_t total = 0;
        };
        struct PendingCompletion
        {
            std::size_t line;
            PooledLineSubscriber original;
            std::vector<std::size_t> result;
        };
        TemporaryCredit<std::vector<PendingCompletion>> pending_memory(budget);
        using PartialMap = std::map<ReplyKey, Partial>;
        TemporaryCredit<PartialMap> partial_memory(budget);
        auto& partial = partial_memory.value;
        struct ReceiptBuffers
        {
            std::set<ReplyKey> completed;
            std::vector<unsigned long long> completed_by_source, global_completed;
        };
        TemporaryCredit<ReceiptBuffers> receipt_memory(budget);
        auto& completed = receipt_memory.value.completed;
        auto& completed_by_source = receipt_memory.value.completed_by_source;
        auto& global_completed = receipt_memory.value.global_completed;
        unsigned long long minted = 0;
        collective(*comm,
            [&]
            {
                const auto ranks = std::size_t(comm->getSize());
                if (changed.size() > std::numeric_limits<unsigned long long>::max() / ranks ||
                    ranks > budget.limit / (2 * sizeof(unsigned long long)))
                    throw std::invalid_argument("Line completion count capacity overflow.");
                minted = changed.size();
                receipt_memory.add(2 * ranks * sizeof(unsigned long long));
                completed_by_source.assign(ranks, 0);
                global_completed.assign(ranks, 0);
                if (completed_by_source.capacity() != ranks || global_completed.capacity() != ranks)
                    throw std::invalid_argument("Line completion capacity differs from admission.");
            });
        auto& partial_charge = partial_memory.bytes;
        std::size_t cursor = 0, offset = 0;
        std::uint64_t reply_page = 0;
        for (;;)
        {
            const auto wire_credit = live_page_credit(*comm, budget, page_bytes);
            TemporaryCredit<std::vector<OwnerPacket>> reply_memory(budget), controls(budget);
            TemporaryCredit<std::vector<NativeStamp>> outgoing_stamps(budget), incoming_stamps(budget);
            auto& replies = reply_memory.value;
            auto& charge = reply_memory.bytes;
            std::size_t page_used = 0;
            collective(*comm,
                [&]
                {
                    while (cursor < changed.size())
                    {
                        const auto result = registry.query_result(*dirty, changed[cursor]);
                        const auto& sub = result.subscriber;
                        const auto& line = pools.lines.get(*dirty);
                        std::size_t bytes = add_size(152, line.size());
                        bytes = add_size(bytes, pools.supports.get(sub.support).size());
                        const auto base = bytes;
                        if (base + sizeof(OwnerPacket) >= wire_credit)
                            throw std::invalid_argument("Line reply header exceeds live page credit.");
                        const auto available = wire_credit - page_used;
                        if (base + sizeof(OwnerPacket) >= available)
                            break;
                        std::size_t count = 0;
                        while (offset + count < result.result_size())
                        {
                            const auto point_bytes = add_size(
                                8, PLIC::encoded_identity_size(pools.points.get(result.result_at(offset + count))));
                            if (point_bytes > available - sizeof(OwnerPacket) - bytes)
                                break;
                            bytes += point_bytes;
                            ++count;
                        }
                        if (!count)
                        {
                            if (!page_used)
                                throw std::invalid_argument("Line reply point exceeds page limit.");
                            break;
                        }
                        budget.add(bytes);
                        charge += bytes;
                        Writer w;
                        w.data.reserve(bytes);
                        if (w.data.capacity() != bytes)
                            throw std::invalid_argument("Line reply capacity mismatch.");
                        header(w, layer, context, 1, wave, source_protocol);
                        w.bytes(line);
                        subscriber(w, sub, pools, source_protocol);
                        w.word(result.result_size());
                        w.word(offset);
                        w.word(count);
                        for (std::size_t p = 0; p < count; ++p)
                        {
                            const auto& point = pools.points.get(result.result_at(offset + p));
                            append_framed_contact_point_identity(w, point);
                        }
                        if (w.data.size() != bytes)
                            throw std::invalid_argument("Line reply size mismatch.");
                        const auto before = budget.used;
                        budget.append_capacity(replies);
                        charge += budget.used - before;
                        replies.push_back({sub.resident, std::move(w.data)});
                        if (source_protocol)
                        {
                            outgoing_stamps.append(outgoing_stamps.value);
                            outgoing_stamps.value.push_back({sub.source_generation, sub.native_source});
                        }
                        page_used += bytes + sizeof(OwnerPacket);
                        offset += count;
                        if (offset == result.result_size())
                        {
                            ++cursor;
                            offset = 0;
                        }
                    }
                });
            collective(*comm,
                [&]
                {
                    if (observe_data)
                        observe_data(true, layer, wave, reply_page, replies);
                    if (source_protocol)
                    {
                        build_native_controls(controls, replies, outgoing_stamps.value, layer, context, wave,
                            reply_page, true, comm->getRank(), comm->getSize());
                        inject_native_fault(controls, source_fault, true, comm->getRank(), wave, reply_page);
                    }
                    if (!before_send)
                        return;
                    before_send(layer, wave, replies);
                    std::size_t actual = replies.capacity() * sizeof(OwnerPacket);
                    for (const auto& packet : replies)
                        actual = add_size(actual, packet.bytes.capacity());
                    if (actual > charge)
                        budget.add(actual - charge);
                    else
                        budget.used -= charge - actual;
                    charge = actual;
                });
            int pending = !replies.empty(), active = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &pending, &active);
            TemporaryCredit<OwnerExchange> incoming_memory(budget), received_controls(budget);
            auto& incoming = incoming_memory.value;
            if (active)
                transfer_packets(reply_memory, incoming_memory, comm);
            if (source_protocol)
            {
                transfer_packets(controls, received_controls, comm);
                collective(*comm,
                    [&]
                    {
                        match_native_controls(incoming_stamps, incoming.packets, received_controls.value.packets, layer,
                            context, wave, reply_page, true, comm->getRank(), comm->getSize());
                    });
            }
            if (!active)
                break;
            collective(*comm,
                [&]
                {
                    for (std::size_t packet_index = 0; packet_index < incoming.packets.size(); ++packet_index)
                    {
                        const auto& packet = incoming.packets[packet_index];
                        Reader r{packet.bytes};
                        header(r, layer, context, 1, wave, source_protocol);
                        const auto line = r.bytes_view();
                        if (authority(line, comm->getSize()) != packet.peer)
                            throw std::invalid_argument("Foreign pooled line reply.");
                        const auto line_slot = intern_bytes(line, pools.lines, budget);
                        auto sub = subscriber(r, pools, comm->getSize(), source_protocol);
                        if (source_protocol)
                        {
                            sub.source_generation = incoming_stamps.value[packet_index].generation;
                            sub.native_source = incoming_stamps.value[packet_index].native;
                        }
                        const auto total = r.word(), begin = r.word(), count = r.word();
                        if (sub.resident != comm->getRank() || !count || count > packet.bytes.size() / 8 ||
                            begin > total || count > total - begin || total > budget.limit / sizeof(std::size_t))
                            throw std::invalid_argument("Invalid pooled line reply.");
                        const ReplyKey key{line_slot, sub.support, sub.support_authority, sub.emission, sub.image,
                            sub.edge, sub.source_generation, sub.native_source};
                        if (completed.contains(key))
                            throw std::invalid_argument("Repeated completed pooled line query.");
                        auto found = partial.find(key);
                        if (found == partial.end())
                        {
                            if (begin)
                                throw std::invalid_argument("Missing initial pooled line reply chunk.");
                            const auto bytes =
                                sizeof(PartialMap::value_type) + 4 * sizeof(void*) + total * sizeof(std::size_t);
                            budget.add(bytes);
                            partial_charge += bytes;
                            sub.result_state = PooledLineSubscriber::ResultState::Explicit;
                            sub.result.reserve(total);
                            if (sub.result.capacity() != total)
                                throw std::invalid_argument("Line reply capacity mismatch.");
                            found = partial.emplace(key, Partial{std::move(sub), std::size_t(total)}).first;
                        }
                        if (found->second.total != total || found->second.subscriber.result.size() != begin)
                            throw std::invalid_argument("Repeated or out-of-order pooled line reply chunk.");
                        auto& result = found->second.subscriber.result;
                        for (std::uint64_t p = 0; p < count; ++p)
                            result.push_back(intern_point(r.bytes_view(), pools, budget));
                        r.done();
                        if (result.size() == total)
                        {
                            const auto receipt = sizeof(ReplyKey) + 4 * sizeof(void*);
                            receipt_memory.add(receipt);
                            if (!completed.insert(key).second)
                                throw std::invalid_argument("Repeated completed pooled line query.");
                            auto& source_count = completed_by_source.at(std::size_t(packet.peer));
                            if (source_count ==
                                std::numeric_limits<unsigned long long>::max() / static_cast<unsigned>(comm->getSize()))
                                throw std::invalid_argument("Line completion count overflow.");
                            ++source_count;
                            const auto result_bytes = result.capacity() * sizeof(std::size_t);
                            const bool retain_completion = completed_reply && found->second.subscriber.native_source;
                            if (retain_completion)
                            {
                                auto original = original_request(line_slot, found->second.subscriber);
                                if (!original || !original_member_equal(*original, found->second.subscriber) ||
                                    original->has_result() || original->result.capacity())
                                    throw std::invalid_argument("Missing or mismatched original native line request.");
                                // Validate endpoint slots NOW against the original local pool. The
                                // wire subscriber has default0/0, which is never endpoint evidence.
                                (void) pools.points.get(original->first);
                                (void) pools.points.get(original->last);
                                pending_memory.append(pending_memory.value);
                                pending_memory.value.push_back({line_slot, std::move(*original), std::move(result)});
                                // Transfer the one existing admitted output allocation. No add,
                                // duplicate output, or gap in the old/new overlap accounting.
                                pending_memory.bytes = add_size(pending_memory.bytes, result_bytes);
                            }
                            else
                                publish(line_slot, found->second.subscriber);
                            const auto bytes = sizeof(PartialMap::value_type) + 4 * sizeof(void*) + result_bytes;
                            partial.erase(found);
                            budget.used -= bytes - (retain_completion ? result_bytes : 0);
                            partial_charge -= bytes;
                        }
                    }
                });
            incoming_memory.reset();
            if (reply_page == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Native line reply page identity exhausted.");
            ++reply_page;
        }
        collective(*comm,
            [&]
            {
                if (!partial.empty() || partial_charge)
                    throw std::invalid_argument("Missing terminal pooled line reply chunk.");
            });
        Teuchos::reduceAll(
            *comm, Teuchos::REDUCE_SUM, comm->getSize(), completed_by_source.data(), global_completed.data());
        collective(*comm,
            [&]
            {
                if (global_completed[std::size_t(comm->getRank())] != minted)
                    throw std::invalid_argument("Incomplete changed line-query delivery.");
                if (wave == std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument("Line completion wave identity exhausted.");
            });
        // Only now are terminal chunks, peer controls and globally minted
        // counts all authenticated. A receiver-issued proof cannot escape early.
        collective(*comm,
            [&]
            {
                for (auto& pending : pending_memory.value)
                {
                    const auto bytes = pending.result.capacity() * sizeof(std::size_t);
                    bool consumed = false;
                    if (consume_completion)
                    {
                        const PooledLineBorrowedCompletion completion(
                            pending.line, pending.original, pending.result, context, layer, pools);
                        consumed = consume_completion(pending.line, completion);
                    }
                    if (consumed)
                    {
                        // The synchronous borrow has ended. Physically retire the
                        // one admitted output before returning its live credit.
                        std::vector<std::size_t>().swap(pending.result);
                        pending_memory.bytes -= bytes;
                        budget.used -= bytes;
                        continue;
                    }
                    auto receipt =
                        PooledLineCompletion::seal(pending.line, std::move(pending.original), std::move(pending.result),
                            context, layer, pools, budget, completion_keeper, completion_arena, layer, false);
                    pending_memory.bytes -= bytes; // ownership moved, charge stays live in receipt
                    completed_reply(pending.line, std::move(receipt));
                }
            });
        ++wave;
        receipt_memory.reset();
        changed_memory.reset();
    }
}
void fetch_pooled_line_completions(PooledLineRegistry& registry, IncidencePools& pools,
    const NextPooledLineCompletionRequest& next_request, unsigned effective_layer, ContactHaloContext context,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget, std::size_t page_bytes,
    const PooledLineCompletionReceive& completed_reply, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena, PooledCachedLineFault fault)
{
    if (comm.is_null())
        throw std::invalid_argument("Cached native line controls require a communicator.");
    std::optional<CachedMetadataEnvironment> environment;
    const auto ranks = comm->getSize(), rank = comm->getRank();
    std::size_t empty_page_bytes = 0;
    collective(*comm,
        [&]
        {
            if (!page_bytes || effective_layer >= 64 || !completed_reply || !arena || arena->budget != &budget ||
                !PLIC::geometry_encoding_capacity_profile_supported())
                throw std::invalid_argument("Invalid cached native line control destination.");
            cartesian_detail::check_keeper(budget, keeper);
            cartesian_detail::check_keeper(budget, arena->keeper);
            if (!keeper)
                keeper = arena->keeper;
            if (!keeper)
                throw std::invalid_argument("Cached native line controls require a Budget lifetime keeper.");
            if (ranks <= 0)
                throw std::invalid_argument("Cached native controls require positive communicator size.");
            empty_page_bytes =
                cartesian_detail::product(std::size_t(ranks), add_size(cached_header_bytes, sizeof(OwnerPacket)));
            environment.emplace();
        });
    TemporaryCredit<std::set<CachedMemberKey>> requested_members(budget), served_members(budget);
    bool ended = false;
    std::uint64_t wave = 0;
    for (;;)
    {
        PooledLineCompletionRequest local_request;
        bool has_local = false;
        int request_authority = -1;
        collective(*comm,
            [&]
            {
                if (!ended)
                {
                    has_local = next_request && next_request(local_request);
                    ended = !has_local;
                }
                if (!has_local)
                    return;
                const auto& member = local_request.original;
                if (member.resident != rank || member.support_authority < 0 || member.support_authority >= ranks ||
                    member.image > 1 || !member.native_source || !member.source_generation || member.has_result() ||
                    member.result.capacity())
                    throw std::invalid_argument("Invalid original cached native request.");
                const auto& line = pools.lines.get(local_request.line);
                const auto support = pools.supports.get(member.support);
                if (line.empty() || support.empty() || authority(support.bytes(), ranks) != member.support_authority)
                    throw std::invalid_argument("Invalid full cached native request keys.");
                (void) pools.points.get(member.first);
                (void) pools.points.get(member.last);
                request_authority = authority(line, ranks);
                remember_cached_member(requested_members, local_request);
            });
        unsigned long long local_expected = has_local, expected = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_expected, &expected);
        TemporaryCredit<std::vector<OwnerPacket>> outgoing(budget);
        const auto request_credit = live_page_credit(*comm, budget, page_bytes);
        collective(*comm,
            [&]
            {
                const auto body = has_local ? cached_request_bytes(local_request, pools) : 0;
                if (empty_page_bytes > request_credit || body > request_credit - empty_page_bytes)
                    throw std::invalid_argument("Cached native request exceeds bounded live page credit.");
                cached_reserve_packets(outgoing, ranks);
                for (int peer = 0; peer < ranks; ++peer)
                {
                    const bool present = has_local && peer == request_authority;
                    const auto bytes = add_size(cached_header_bytes, present ? body : 0);
                    cached_packet(outgoing, bytes, peer,
                        [&](Writer& writer)
                        {
                            cached_header(writer, effective_layer, context, wave, 0, false, rank, peer, present);
                            if (present)
                                write_cached_request(writer, local_request, pools);
                        });
                }
                inject_cached_fault(outgoing, fault, false, rank, wave, 0);
            });
        TemporaryCredit<OwnerExchange> incoming(budget);
        transfer_packets(outgoing, incoming, comm);
        struct StoredReply
        {
            PooledLineCompletionRequest request;
            PooledLineRegistry::CompletedMember cached;
            std::size_t offset = 0;
        };
        TemporaryCredit<std::vector<StoredReply>> stored(budget);
        collective(*comm,
            [&]
            {
                if (incoming.value.packets.size() != std::size_t(ranks))
                    throw std::invalid_argument(
                        "Missing or duplicate cached native request envelope, including empty peers.");
                ReceiptAllocation lookup(budget);
                std::vector<unsigned char> seen;
                lookup.reserve(seen, ranks);
                seen.resize(ranks, 0);
                for (const auto& packet : incoming.value.packets)
                {
                    if (packet.peer < 0 || packet.peer >= ranks || seen[packet.peer]++ ||
                        packet.bytes.size() > page_bytes)
                        throw std::invalid_argument("Duplicate, foreign or oversized cached native request envelope.");
                    Reader reader{packet.bytes};
                    if (cached_header(reader, effective_layer, context, wave, 0, false, packet.peer, rank))
                    {
                        auto request = read_cached_request(reader, pools, budget, ranks);
                        if (request.original.resident != packet.peer ||
                            authority(pools.lines.get(request.line), ranks) != rank)
                            throw std::invalid_argument("Foreign cached native line request member.");
                        remember_cached_member(served_members, request);
                        auto actual =
                            registry.find_completed_member(request.line, effective_layer, request.original, pools);
                        if (!actual)
                            throw std::invalid_argument(
                                "Missing, dirty, stale or uncomputed cached native line member.");
                        stored.append(stored.value);
                        stored.value.push_back({std::move(request), std::move(*actual), 0});
                    }
                    reader.done();
                }
            });
        incoming.reset();
        unsigned long long local_accepted = stored.value.size(), accepted = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_accepted, &accepted);
        collective(*comm,
            [&]
            {
                if (accepted != expected)
                    throw std::invalid_argument(
                        "Cached native request inventory differs from completed private lookups.");
            });
        TemporaryCredit<std::vector<std::size_t>> result(budget);
        bool complete = !has_local;
        std::optional<std::size_t> result_total;
        std::optional<unsigned> kernel_witness;
        std::uint64_t page = 0;
        for (;;)
        {
            int remaining = 0;
            collective(*comm,
                [&]
                {
                    for (const auto& value : stored.value)
                        remaining |= value.offset < value.cached.result.result_size();
                });
            int any_remaining = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &remaining, &any_remaining);
            const auto reply_credit = live_page_credit(*comm, budget, page_bytes);
            TemporaryCredit<std::vector<OwnerPacket>> replies(budget);
            collective(*comm,
                [&]
                {
                    if (empty_page_bytes > reply_credit)
                        throw std::invalid_argument("Cached native empty receipts exceed bounded live page credit.");
                    cached_reserve_packets(replies, ranks);
                    std::size_t used = empty_page_bytes;
                    bool emitted = false;
                    for (int peer = 0; peer < ranks; ++peer)
                    {
                        auto found = std::find_if(stored.value.begin(), stored.value.end(),
                            [&](const auto& value) { return value.request.original.resident == peer; });
                        std::size_t count = 0, body = 0;
                        if (found != stored.value.end() && found->offset < found->cached.result.result_size())
                        {
                            const auto reference = add_size(cached_request_bytes(found->request, pools), 32);
                            if (reference < reply_credit - used)
                            {
                                body = reference;
                                while (found->offset + count < found->cached.result.result_size())
                                {
                                    const auto slot = found->cached.result.result_at(found->offset + count);
                                    const auto bytes = add_size(8, PLIC::encoded_identity_size(pools.points.get(slot)));
                                    if (bytes > reply_credit - used - body)
                                        break;
                                    body += bytes;
                                    ++count;
                                }
                            }
                            if (!count)
                                body = 0;
                        }
                        cached_packet(replies, add_size(cached_header_bytes, body), peer,
                            [&](Writer& writer)
                            {
                                cached_header(
                                    writer, effective_layer, context, wave, page, true, rank, peer, count != 0);
                                if (!count)
                                    return;
                                write_cached_request(writer, found->request, pools);
                                writer.word(found->cached.kernel_witness_layer);
                                writer.word(found->cached.result.result_size());
                                writer.word(found->offset);
                                writer.word(count);
                                for (std::size_t p = 0; p < count; ++p)
                                    write_cached_point(writer,
                                        pools.points.get(found->cached.result.result_at(found->offset + p)));
                            });
                        if (count)
                        {
                            found->offset += count;
                            used += body;
                            emitted = true;
                        }
                    }
                    if (remaining && !emitted)
                        throw std::invalid_argument("Cached native result point cannot fit bounded live page credit.");
                    inject_cached_fault(replies, fault, true, rank, wave, page);
                });
            TemporaryCredit<OwnerExchange> received(budget);
            transfer_packets(replies, received, comm);
            collective(*comm,
                [&]
                {
                    if (received.value.packets.size() != std::size_t(ranks))
                        throw std::invalid_argument(
                            "Missing or duplicate cached native reply envelope, including empty peers.");
                    ReceiptAllocation lookup(budget);
                    std::vector<unsigned char> seen;
                    lookup.reserve(seen, ranks);
                    seen.resize(ranks, 0);
                    for (const auto& packet : received.value.packets)
                    {
                        if (packet.peer < 0 || packet.peer >= ranks || seen[packet.peer]++ ||
                            packet.bytes.size() > page_bytes)
                            throw std::invalid_argument(
                                "Duplicate, foreign or oversized cached native reply envelope.");
                        Reader reader{packet.bytes};
                        if (cached_header(reader, effective_layer, context, wave, page, true, packet.peer, rank))
                        {
                            if (!any_remaining || !has_local || packet.peer != request_authority || complete)
                                throw std::invalid_argument("Unknown or repeated cached native result completion.");
                            const auto reference = read_cached_request(reader, pools, budget, ranks);
                            if (!cached_same_request(local_request, reference))
                                throw std::invalid_argument(
                                    "Cached native result original member or endpoints differ.");
                            const auto witness = reader.word(), total = reader.word(), begin = reader.word(),
                                       count = reader.word();
                            if (witness > effective_layer || !total || !count || begin > total ||
                                count > total - begin || count > packet.bytes.size() / 8 ||
                                total > budget.limit / sizeof(std::size_t))
                                throw std::invalid_argument("Invalid cached native result witness or chunk inventory.");
                            if (!result_total)
                            {
                                if (begin)
                                    throw std::invalid_argument("Missing initial cached native result chunk.");
                                const auto bytes = cartesian_detail::product(std::size_t(total), sizeof(std::size_t));
                                result.add(bytes);
                                result.value.reserve(total);
                                if (result.value.capacity() != total)
                                    throw std::invalid_argument(
                                        "Cached native result capacity differs from admission.");
                                result_total = total;
                                kernel_witness = unsigned(witness);
                            }
                            if (*result_total != total || *kernel_witness != witness || begin != result.value.size())
                                throw std::invalid_argument(
                                    "Duplicate, reordered or stale cached native result chunk.");
                            for (std::uint64_t p = 0; p < count; ++p)
                                result.value.push_back(read_cached_point(reader.bytes_view(), pools, budget, false));
                            complete = result.value.size() == total;
                        }
                        reader.done();
                    }
                });
            received.reset();
            if (!any_remaining)
                break; // authenticated all-peer EMPTY terminal page
            collective(*comm,
                [&]
                {
                    if (page == std::numeric_limits<std::uint64_t>::max())
                        throw std::invalid_argument("Cached native reply page identity exhausted.");
                });
            ++page;
        }
        unsigned long long local_completed = has_local && complete, global_completed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_completed, &global_completed);
        collective(*comm,
            [&]
            {
                if (!complete || global_completed != expected ||
                    (has_local && (!result_total || !kernel_witness || result.value.size() != *result_total)))
                    throw std::invalid_argument("Missing terminal cached native line result or completion count.");
            });
        collective(*comm,
            [&]
            {
                if (has_local)
                {
                    const auto bytes = result.value.capacity() * sizeof(std::size_t);
                    auto completion = PooledLineCompletion::seal(local_request.line, std::move(local_request.original),
                        std::move(result.value), context, effective_layer, pools, budget, keeper, arena,
                        *kernel_witness, true);
                    result.bytes -= bytes; // ONE existing output owner transfers to the sealed receipt
                    completed_reply(local_request.line, std::move(completion));
                }
                if (wave == std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument("Cached native request wave identity exhausted.");
            });
        if (!expected)
            break; // even an entirely empty invocation exchanged both envelope families
        ++wave;
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
