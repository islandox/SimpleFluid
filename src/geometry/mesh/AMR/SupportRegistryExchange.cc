#include "geometry/mesh/AMR/SupportRegistryExchange.hh"
#include "geometry/mesh/AMR/BudgetedOwnerExchange.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "geometry/mesh/AMR/PooledSupportRegistry.hh"
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::uint64_t magic = 0x5346524547535550ULL;
constexpr std::size_t fixed_bytes = 14 * 8;
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
    template<class T> void append(std::vector<T>& v)
    {
        const auto before = budget.used;
        try
        {
            budget.append_capacity(v);
        }
        catch (...)
        {
            held += budget.used - before;
            throw;
        }
        held += budget.used - before;
    }
    void observe(std::size_t before)
    {
        if (budget.used >= before)
            held += budget.used - before;
        else
            held -= before - budget.used;
    }
};
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Support registry message size overflows.");
    return a + b;
}
std::size_t packet_storage(const std::vector<OwnerPacket>& packets)
{
    if (packets.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(OwnerPacket))
        throw std::invalid_argument("Support registry packet capacity overflows.");
    auto size = packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
        size = sum(size, packet.bytes.capacity());
    return size;
}
std::size_t key_encoding_bound(const AdaptiveCellKey& key)
{
    const auto dynamic = key_capacity(key);
    if (dynamic > std::numeric_limits<std::size_t>::max() / 4 - 128)
        throw std::invalid_argument("Support registry key size overflows.");
    return 4 * (dynamic + 128);
}
void header(Writer& w, ContactHaloContext c, std::uint64_t page)
{
    w.word(magic);
    w.word(1);
    w.word(c.epoch);
    w.word(c.topology);
    w.word(c.partition);
    w.word(c.round);
    w.word(c.transaction);
    w.word(page);
}
void verify_header(Reader& r, ContactHaloContext c, std::uint64_t page)
{
    if (r.word() != magic || r.word() != 1 || r.word() != c.epoch || r.word() != c.topology ||
        r.word() != c.partition || r.word() != c.round || r.word() != c.transaction || r.word() != page)
        throw std::invalid_argument("Invalid support registry context or page.");
}
} // namespace

void exchange_support_registry_stream(PooledSupportRegistry& registry, const std::function<bool(RegistryChange&)>& next,
    ContactHaloContext context, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t page_bytes, Budget& budget,
    std::size_t cursor_capacity)
{
    if (comm.is_null())
        throw std::invalid_argument("Support registry requires communicator.");
    collective(*comm,
        [&]
        {
            if (!next || !page_bytes || page_bytes > budget.limit || !cursor_capacity || cursor_capacity > budget.limit)
                throw std::invalid_argument("Invalid support registry page limit or cursor.");
        });
    Scope pending_memory{budget};
    std::optional<RegistryChange> pending;
    std::size_t pending_charge = 0;
    bool ended = false;
    std::uint64_t page_number = 0;
    for (;;)
    {
        Scope page_memory{budget};
        std::vector<OwnerPacket> packets;
        collective(*comm,
            [&]
            {
                std::size_t logical_bytes = 0;
                for (;;)
                {
                    if (!pending && !ended)
                    {
                        Scope callback_memory{budget};
                        const auto cap = cursor_capacity;
                        callback_memory.add(cap);
                        RegistryChange value;
                        if (next(value))
                        {
                            const auto charge = sum(value.support.capacity(), key_capacity(value.member.cell));
                            if (charge > cap)
                                throw std::invalid_argument("Support registry cursor exceeded lease.");
                            callback_memory.drop(cap - charge);
                            callback_memory.held -= charge;
                            pending_memory.held += charge;
                            pending_charge = charge;
                            pending.emplace(std::move(value));
                        }
                        else
                            ended = true;
                    }
                    if (!pending)
                        break;
                    if (pending->support.empty() || pending->member.owner != comm->getRank() ||
                        !pending->member.encoded_size)
                        throw std::invalid_argument("Invalid support registry source membership.");
                    const auto key_bound = key_encoding_bound(pending->member.cell);
                    page_memory.add(key_bound);
                    auto key = encode_adaptive_key(pending->member.cell);
                    if (key.capacity() > key_bound)
                        throw std::invalid_argument("Support registry key exceeded admission.");
                    auto bytes = sum(fixed_bytes, pending->support.size());
                    bytes = sum(bytes, key.size());
                    const auto cost = sum(sizeof(OwnerPacket), bytes);
                    if (cost > page_bytes)
                        throw std::invalid_argument("Support registry record exceeds page limit.");
                    if (cost > page_bytes - logical_bytes)
                        break;
                    logical_bytes += cost;
                    page_memory.append(packets);
                    page_memory.add(bytes);
                    Writer writer;
                    writer.data.reserve(bytes);
                    if (writer.data.capacity() != bytes)
                        throw std::invalid_argument("Support registry packet capacity differs from admission.");
                    header(writer, context, page_number);
                    writer.word(pending->remove);
                    writer.bytes(pending->support);
                    writer.bytes(key);
                    writer.word(pending->member.revision);
                    writer.word(pending->member.surface);
                    writer.word(pending->member.encoded_size);
                    if (writer.data.size() != bytes)
                        throw std::logic_error("Support registry packet framing differs.");
                    packets.push_back({authority(pending->support, comm->getSize()), std::move(writer.data)});
                    pending.reset();
                    pending_memory.drop(pending_charge);
                    pending_charge = 0;
                }
            });
        const int active = !packets.empty();
        int any = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &active, &any);
        if (!any)
            break;
        const auto sent = packet_storage(packets);
        const auto before = budget.used;
        OwnerExchange incoming;
        try
        {
            incoming = consume_budgeted_owner_packets(std::move(packets), comm, budget, sent);
        }
        catch (...)
        {
            page_memory.observe(before);
            throw;
        }
        page_memory.observe(before);
        collective(*comm,
            [&]
            {
                for (const auto& packet : incoming.packets)
                {
                    Scope decoded_memory{budget};
                    Reader reader{packet.bytes};
                    verify_header(reader, context, page_number);
                    const auto remove = reader.word();
                    const auto support = reader.bytes_view();
                    const auto key = reader.bytes_view();
                    const auto revision = reader.word(), ordinal = reader.word(), size = reader.word();
                    reader.done();
                    if (remove > 1 || support.empty() || !size ||
                        authority(support, comm->getSize()) != comm->getRank() || key.size() > budget.limit / 2)
                        throw std::invalid_argument("Invalid support registry routed membership.");
                    decoded_memory.add(support.size());
                    decoded_memory.add(2 * key.size());
                    RegistryChange change;
                    change.remove = bool(remove);
                    change.support.assign(support.begin(), support.end());
                    if (change.support.capacity() != support.size())
                        throw std::invalid_argument("Support registry decoded capacity differs from admission.");
                    change.member = {decode_adaptive_key(key), revision, ordinal, size, packet.peer};
                    registry.apply(std::span<const RegistryChange>(&change, 1));
                }
            });
        if (page_number == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Support registry page identity exhausted.");
        ++page_number;
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
