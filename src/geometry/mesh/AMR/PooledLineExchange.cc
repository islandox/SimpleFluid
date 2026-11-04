#include "PooledLineExchange.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/BudgetedOwnerExchange.hh"
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
void header(Writer& w, unsigned layer, ContactHaloContext c, std::uint64_t kind, std::uint64_t wave = 0)
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
void header(Reader& r, unsigned layer, ContactHaloContext c, std::uint64_t kind, std::uint64_t wave = 0)
{
    if (r.word() != 2 || r.word() != c.epoch || r.word() != c.topology || r.word() != c.partition ||
        r.word() != c.round || r.word() != c.transaction || r.word() != layer || r.word() != wave || r.word() != kind)
        throw std::invalid_argument("Invalid pooled line context.");
}
void subscriber(Writer& w, const PooledLineSubscriber& sub, const IncidencePools& pools)
{
    w.word(sub.resident);
    w.word(sub.support_authority);
    w.bytes(pools.supports.get(sub.support));
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
PooledLineSubscriber subscriber(Reader& r, IncidencePools& pools, int ranks, Budget& budget)
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
    sub.support = intern_bytes(support, pools.supports, budget);
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
std::size_t edit_size(const PooledLineEdit& edit, const IncidencePools& pools)
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
} // namespace
void exchange_pooled_line_edits(PooledLineRegistry& registry, IncidencePools& pools, const NextPooledLineEdit& next,
    unsigned layer, ContactHaloContext context, Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget,
    std::size_t page_bytes, const PooledLineReply& publish, const PooledLineBeforeSend& before_send)
{
    if (comm.is_null())
        throw std::invalid_argument("Pooled line exchange requires communicator.");
    collective(*comm,
        [&]
        {
            if (!page_bytes)
                throw std::invalid_argument("Invalid pooled line page size.");
            registry.set_layer(layer);
        });
    bool ended = false;
    std::vector<PooledLineEdit> staged;
    std::size_t staged_charge = 0, staged_cursor = 0;
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
                    const auto bytes = edit_size(item, pools);
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
        std::vector<OwnerPacket> outgoing;
        std::size_t charge = 0, page_used = 0;
        collective(*comm,
            [&]
            {
                while (staged_cursor < staged.size())
                {
                    const auto& item = staged[staged_cursor];
                    const auto bytes = edit_size(item, pools);
                    if (bytes > wire_credit || sizeof(OwnerPacket) > wire_credit - bytes)
                        throw std::invalid_argument("Pooled line edit exceeds live page credit.");
                    const auto cost = bytes + sizeof(OwnerPacket);
                    if (cost > wire_credit - page_used)
                        break;
                    page_used += cost;
                    budget.add(bytes);
                    charge += bytes;
                    Writer w;
                    w.data.reserve(bytes);
                    if (w.data.capacity() != bytes)
                        throw std::invalid_argument("Line edit capacity mismatch.");
                    header(w, layer, context, 0);
                    w.bytes(pools.lines.get(item.line));
                    subscriber(w, item.subscriber, pools);
                    w.word(item.remove);
                    if (!item.remove)
                    {
                        const auto scratch =
                            std::max(PLIC::encoded_identity_size(pools.points.get(item.subscriber.first)),
                                PLIC::encoded_identity_size(pools.points.get(item.subscriber.last)));
                        budget.add(scratch);
                        w.bytes(PLIC::encode_identity(pools.points.get(item.subscriber.first)));
                        w.bytes(PLIC::encode_identity(pools.points.get(item.subscriber.last)));
                        budget.used -= scratch;
                    }
                    if (w.data.size() != bytes)
                        throw std::invalid_argument("Line edit size mismatch.");
                    const auto before = budget.used;
                    budget.append_capacity(outgoing);
                    charge += budget.used - before;
                    outgoing.push_back({authority(pools.lines.get(item.line), comm->getSize()), std::move(w.data)});
                    ++staged_cursor;
                }
            });
        int local = !outgoing.empty(), any = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &any);
        if (!any)
            break;
        auto incoming = consume_budgeted_owner_packets(std::move(outgoing), comm, budget, charge);
        collective(*comm,
            [&]
            {
                for (const auto& packet : incoming.packets)
                {
                    Reader r{packet.bytes};
                    header(r, layer, context, 0);
                    const auto line = r.bytes_view();
                    if (line.empty() || authority(line, comm->getSize()) != comm->getRank())
                        throw std::invalid_argument("Foreign pooled line edit.");
                    PooledLineEdit item;
                    item.line = intern_bytes(line, pools.lines, budget);
                    item.subscriber = subscriber(r, pools, comm->getSize(), budget);
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
        retire_budgeted_owner_packets(incoming, budget);
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
        std::vector<std::size_t> changed;
        std::size_t changed_charge = 0;
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
        using ReplyKey = std::tuple<std::size_t, std::size_t, int, std::uint64_t, std::uint64_t, std::uint64_t>;
        struct Partial
        {
            PooledLineSubscriber subscriber;
            std::size_t total = 0;
        };
        std::map<ReplyKey, Partial> partial;
        std::set<ReplyKey> completed;
        std::vector<unsigned long long> completed_by_source, global_completed;
        std::size_t receipt_charge = 0;
        unsigned long long minted = 0;
        collective(*comm,
            [&]
            {
                const auto ranks = std::size_t(comm->getSize());
                if (changed.size() > std::numeric_limits<unsigned long long>::max() / ranks ||
                    ranks > budget.limit / (2 * sizeof(unsigned long long)))
                    throw std::invalid_argument("Line completion count capacity overflow.");
                minted = changed.size();
                receipt_charge = 2 * ranks * sizeof(unsigned long long);
                budget.add(receipt_charge);
                completed_by_source.assign(ranks, 0);
                global_completed.assign(ranks, 0);
                if (completed_by_source.capacity() != ranks || global_completed.capacity() != ranks)
                    throw std::invalid_argument("Line completion capacity differs from admission.");
            });
        std::size_t partial_charge = 0, cursor = 0, offset = 0;
        for (;;)
        {
            const auto wire_credit = live_page_credit(*comm, budget, page_bytes);
            std::vector<OwnerPacket> replies;
            std::size_t charge = 0, page_used = 0;
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
                        header(w, layer, context, 1, wave);
                        w.bytes(line);
                        subscriber(w, sub, pools);
                        w.word(result.result_size());
                        w.word(offset);
                        w.word(count);
                        for (std::size_t p = 0; p < count; ++p)
                        {
                            const auto& point = pools.points.get(result.result_at(offset + p));
                            const auto scratch = PLIC::encoded_identity_size(point);
                            budget.add(scratch);
                            w.bytes(PLIC::encode_identity(point));
                            budget.used -= scratch;
                        }
                        if (w.data.size() != bytes)
                            throw std::invalid_argument("Line reply size mismatch.");
                        const auto before = budget.used;
                        budget.append_capacity(replies);
                        charge += budget.used - before;
                        replies.push_back({sub.resident, std::move(w.data)});
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
            if (!active)
            {
                std::vector<OwnerPacket>().swap(replies);
                budget.used -= charge;
                break;
            }
            auto incoming = consume_budgeted_owner_packets(std::move(replies), comm, budget, charge);
            collective(*comm,
                [&]
                {
                    for (const auto& packet : incoming.packets)
                    {
                        Reader r{packet.bytes};
                        header(r, layer, context, 1, wave);
                        const auto line = r.bytes_view();
                        if (authority(line, comm->getSize()) != packet.peer)
                            throw std::invalid_argument("Foreign pooled line reply.");
                        const auto line_slot = intern_bytes(line, pools.lines, budget);
                        auto sub = subscriber(r, pools, comm->getSize(), budget);
                        const auto total = r.word(), begin = r.word(), count = r.word();
                        if (sub.resident != comm->getRank() || !count || count > packet.bytes.size() / 8 ||
                            begin > total || count > total - begin || total > budget.limit / sizeof(std::size_t))
                            throw std::invalid_argument("Invalid pooled line reply.");
                        const ReplyKey key{
                            line_slot, sub.support, sub.support_authority, sub.emission, sub.image, sub.edge};
                        if (completed.contains(key))
                            throw std::invalid_argument("Repeated completed pooled line query.");
                        auto found = partial.find(key);
                        if (found == partial.end())
                        {
                            if (begin)
                                throw std::invalid_argument("Missing initial pooled line reply chunk.");
                            const auto bytes =
                                sizeof(decltype(partial)::value_type) + 4 * sizeof(void*) + total * sizeof(std::size_t);
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
                            budget.add(receipt);
                            receipt_charge += receipt;
                            if (!completed.insert(key).second)
                                throw std::invalid_argument("Repeated completed pooled line query.");
                            auto& source_count = completed_by_source.at(std::size_t(packet.peer));
                            if (source_count ==
                                std::numeric_limits<unsigned long long>::max() / static_cast<unsigned>(comm->getSize()))
                                throw std::invalid_argument("Line completion count overflow.");
                            ++source_count;
                            publish(line_slot, found->second.subscriber);
                            const auto bytes = sizeof(decltype(partial)::value_type) + 4 * sizeof(void*) +
                                               result.capacity() * sizeof(std::size_t);
                            partial.erase(found);
                            budget.used -= bytes;
                            partial_charge -= bytes;
                        }
                    }
                });
            retire_budgeted_owner_packets(incoming, budget);
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
        ++wave;
        completed.clear();
        std::vector<unsigned long long>().swap(completed_by_source);
        std::vector<unsigned long long>().swap(global_completed);
        budget.used -= receipt_charge;
        std::vector<std::size_t>().swap(changed);
        budget.used -= changed_charge;
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
