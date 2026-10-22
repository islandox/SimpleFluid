/** @file DistributedAncestryDirectory.cc
 * @brief Owner-directed exact registration and lookup without global catalogs.
 */
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <limits>
namespace SimpleFluid::Meshes::AMR
{
using namespace adaptive_detail;
namespace
{
// Ignored staging helper; paste into Directory.cc's anonymous namespace.
// The immutable caller keys and resident shard are views, not copied inputs.
// The explicit cap bounds this service's simultaneous owned BUFFER capacities;
// it does not claim MPI/backend allocator/process RSS.
std::span<const char> directory_key_view(Reader& reader)
{
    const auto count = reader.word();
    if (count > reader.data.size() - reader.offset)
        throw std::invalid_argument("Invalid bounded directory key length.");
    const auto result = reader.data.subspan(reader.offset, std::size_t(count));
    reader.offset += std::size_t(count);
    return result;
}
void directory_array(Budget& budget, std::size_t count, std::size_t width)
{
    if (width && count > budget.limit / width)
        throw std::invalid_argument("Bounded directory array exceeds transient lease.");
    budget.add(count * width);
}
std::size_t directory_packets(const std::vector<OwnerPacket>& packets, std::size_t cap)
{
    Budget budget{cap};
    directory_array(budget, packets.capacity(), sizeof(OwnerPacket));
    for (const auto& packet : packets)
        budget.add(packet.bytes.capacity());
    return budget.used;
}
void directory_word(AdaptiveBytes& bytes, std::size_t& offset, std::uint64_t word)
{
    if (offset > bytes.size() || bytes.size() - offset < 8)
        throw std::logic_error("Bounded directory writer exceeded preadmitted capacity.");
    for (unsigned i = 0; i < 8; ++i)
        bytes[offset++] = char(word >> (8 * i));
}
void directory_key(AdaptiveBytes& bytes, std::size_t& offset, std::span<const char> key)
{
    directory_word(bytes, offset, key.size());
    if (key.size() > bytes.size() - offset)
        throw std::logic_error("Bounded directory key writer exceeded capacity.");
    std::copy(key.begin(), key.end(), bytes.begin() + offset);
    offset += key.size();
}
std::vector<AdaptiveOwner> directory_lookup_leased(std::span<const AdaptiveBytes> keys, bool require_all,
    std::size_t cap, const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
    const std::map<AdaptiveBytes, AdaptiveOwner>& shard)
{
    // Disjoint service arenas: persistent results R=cap/4, caller packets
    // P=cap/4, transport E=cap/2. Logical encodings B=cap/16 leave capacity
    // growth/lookup-copy margin in P and give E the packed-send +two-receive
    // budget it actually needs. During reply construction returned old E
    // replaces the engine, and its lookup key is INCLUDED in P. No old stream
    // survives another exchange. Integer remainders stay unused.
    const auto result_cap = cap / 4, packet_cap = cap / 4, exchange_cap = cap / 2;
    const auto logical_packet_cap = cap / 16;
    std::vector<AdaptiveOwner> result;
    std::vector<unsigned char> seen;
    std::size_t result_used = 0;
    std::vector<OwnerPacket> requests;
    collective(*comm,
        [&]
        {
            Budget output{result_cap};
            directory_array(output, keys.size(), sizeof(AdaptiveOwner) + sizeof(unsigned char));
            Budget input{logical_packet_cap};
            directory_array(input, keys.size(), sizeof(OwnerPacket));
            for (const auto& key : keys)
            {
                if (key.empty())
                    throw std::invalid_argument("Empty adaptive owner lookup key.");
                // Include EVERY result key copy, even duplicate queried keys.
                output.add(key.size());
                input.add(16);
                input.add(key.size());
            }
            result.resize(keys.size());
            seen.resize(keys.size());
            requests.resize(keys.size());
            Budget actual_output{result_cap};
            directory_array(actual_output, result.capacity(), sizeof(AdaptiveOwner));
            directory_array(actual_output, seen.capacity(), sizeof(unsigned char));
            result_used = actual_output.used;
            for (const auto& key : keys)
                actual_output.add(key.size());
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                auto& packet = requests[i];
                packet.peer = authority(keys[i], comm->getSize());
                packet.bytes.resize(16 + keys[i].size());
                std::size_t offset = 0;
                directory_word(packet.bytes, offset, i);
                directory_key(packet.bytes, offset, keys[i]);
                if (offset != packet.bytes.size())
                    throw std::logic_error("Bounded request framing differs.");
            }
            (void) directory_packets(requests, packet_cap);
        });
    auto incoming = exchange_owner_packets(requests, comm, exchange_cap);
    std::vector<OwnerPacket>().swap(requests);
    std::vector<OwnerPacket> replies;
    collective(*comm,
        [&]
        {
            (void) directory_packets(incoming.packets, exchange_cap);
            Budget outgoing{logical_packet_cap};
            directory_array(outgoing, incoming.packets.size(), sizeof(OwnerPacket));
            std::size_t largest_key = 0;
            for (const auto& packet : incoming.packets)
            {
                Reader reader{packet.bytes};
                (void) reader.word();
                const auto key = directory_key_view(reader);
                reader.done();
                if (key.empty() || authority(key, comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("Misrouted bounded owner lookup.");
                outgoing.add(40);
                outgoing.add(key.size());
                largest_key = std::max(largest_key, key.size());
            }
            // Map lookup requires one owned AdaptiveBytes temporary because the
            // resident shard's comparator is not transparent. Admit it BEFORE copy.
            outgoing.add(largest_key);
            replies.resize(incoming.packets.size());
            std::size_t reply_resident = directory_packets(replies, packet_cap);
            for (std::size_t i = 0; i < incoming.packets.size(); ++i)
            {
                const auto& packet = incoming.packets[i];
                Reader reader{packet.bytes};
                const auto index = reader.word();
                const auto key_view = directory_key_view(reader);
                reader.done();
                AdaptiveBytes key(key_view.begin(), key_view.end());
                if (key.capacity() > largest_key)
                    throw std::invalid_argument("Bounded directory lookup key capacity grew.");
                const auto found = shard.find(key);
                auto& reply = replies[i];
                reply.peer = packet.peer;
                reply.bytes.resize(40 + key.size());
                std::size_t offset = 0;
                directory_word(reply.bytes, offset, index);
                directory_key(reply.bytes, offset, key);
                directory_word(reply.bytes, offset, found == shard.end() ? 0 : std::uint64_t(found->second.owner) + 1);
                directory_word(reply.bytes, offset, found == shard.end() ? 0 : found->second.revision);
                directory_word(reply.bytes, offset,
                    found == shard.end() ? std::numeric_limits<std::uint64_t>::max() : found->second.address);
                if (offset != reply.bytes.size())
                    throw std::logic_error("Bounded reply framing differs.");
                if (reply.bytes.capacity() > packet_cap - reply_resident)
                    throw std::invalid_argument("Bounded reply capacities exceed transient share.");
                reply_resident += reply.bytes.capacity();
                if (key.capacity() > packet_cap - reply_resident)
                    throw std::invalid_argument("Bounded replies and live lookup key exceed transient share.");
            }
        });
    incoming = {}; // Release old return buffers BEFORE reply exchange.
    collective(*comm, [&] { (void) directory_packets(replies, packet_cap); });
    auto incoming_replies = exchange_owner_packets(replies, comm, exchange_cap);
    std::vector<OwnerPacket>().swap(replies);
    collective(*comm,
        [&]
        {
            (void) directory_packets(incoming_replies.packets, exchange_cap);
            for (const auto& packet : incoming_replies.packets)
            {
                Reader reader{packet.bytes};
                const auto index = reader.word();
                const auto key = directory_key_view(reader);
                const auto owner = reader.word(), revision = reader.word(), address = reader.word();
                reader.done();
                if (index >= keys.size() || seen[std::size_t(index)] || key.size() != keys[std::size_t(index)].size() ||
                    !std::equal(key.begin(), key.end(), keys[std::size_t(index)].begin()) ||
                    packet.peer != authority(key, comm->getSize()) || owner > std::uint64_t(comm->getSize()))
                    throw std::invalid_argument("Invalid bounded adaptive owner reply.");
                if (require_all && !owner)
                    throw std::invalid_argument("Missing adaptive owner.");
                // No allocating Reader::bytes() temporary: validate spans first,
                // then allocate this one result copy inside its preadmitted R slot.
                if (key.size() > result_cap - result_used)
                    throw std::invalid_argument("Bounded directory duplicate-key result payload exceeds lease.");
                auto& value = result[std::size_t(index)];
                value.key.assign(key.begin(), key.end());
                if (value.key.capacity() > result_cap - result_used)
                    throw std::invalid_argument("Bounded directory result key capacity exceeds lease.");
                result_used += value.key.capacity();
                value.owner = owner ? int(owner - 1) : -1;
                value.revision = revision;
                value.address = address;
                seen[std::size_t(index)] = 1;
            }
            for (auto present : seen)
                if (!present)
                    throw std::invalid_argument("Missing adaptive directory reply.");
        });
    return result;
}

} // namespace

DistributedAncestryDirectory::DistributedAncestryDirectory(
    std::span<const AdaptiveOwner> owned, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget)
    : comm_(std::move(comm)), budget_(budget)
{
    if (comm_.is_null())
        throw std::invalid_argument("Directory needs a communicator.");
    std::vector<OwnerPacket> packets;
    collective(*comm_,
        [&]
        {
            Budget allocation{budget_};
            for (const auto& entry : owned)
            {
                allocation.packet(24);
                allocation.add(entry.key.size());
            }

            for (const auto& entry : owned)
            {
                if (entry.key.empty() || entry.owner != comm_->getRank())
                    throw std::invalid_argument("Directory registration must name actual local owner.");
                Writer w;
                w.bytes(entry.key);
                w.word(entry.revision);
                w.word(entry.address);
                packets.push_back({authority(entry.key, comm_->getSize()), std::move(w.data)});
            }
        });
    auto exchange = exchange_owner_packets(packets, comm_, budget_);
    registration_ = exchange.stats;
    collective(*comm_,
        [&]
        {
            Budget storage{budget_};
            storage.add(sizeof(*this));
            for (const auto& packet : exchange.packets)
            {
                storage.add(sizeof(AdaptiveOwner) + sizeof(AdaptiveBytes) + 3 * sizeof(void*));
                storage.add(packet.bytes.size());
                storage.add(packet.bytes.size());
                Reader r{packet.bytes};
                auto key = r.bytes();
                auto revision = r.word(), address = r.word();
                r.done();
                if (key.empty() || authority(key, comm_->getSize()) != comm_->getRank() ||
                    !shard_.emplace(key, AdaptiveOwner{key, packet.peer, revision, address}).second)
                    throw std::invalid_argument("Duplicate or misrouted adaptive owner.");
            }
            if (storage_bytes() > budget_)
                throw std::invalid_argument("Directory resident storage budget exhausted.");
        });
}
std::vector<AdaptiveOwner> DistributedAncestryDirectory::lookup(
    std::span<const AdaptiveBytes> keys, bool require_all, std::optional<std::size_t> transient_cap) const
{
    const int mode = transient_cap.has_value();
    const int local_mode[2]{mode, -mode};
    int global_mode[2]{};
    Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MAX, 2, local_mode, global_mode);
    if (global_mode[0] != -global_mode[1])
        throw std::invalid_argument("Directory transient-lease presence differs across ranks.");
    if (transient_cap)
    {
        collective(*comm_,
            [&]
            {
                if (!*transient_cap || *transient_cap > budget_)
                    throw std::invalid_argument("Invalid ancestry transient lease.");
            });
        return directory_lookup_leased(keys, require_all, *transient_cap, comm_, shard_);
    }
    // Unchanged legacy path when no explicit transient lease was requested.

    std::vector<OwnerPacket> requests;
    collective(*comm_,
        [&]
        {
            Budget allocation{budget_};
            for (const auto& key : keys)
            {
                allocation.packet(16);
                allocation.add(key.size());
            }
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (keys[i].empty())
                    throw std::invalid_argument("Empty adaptive owner lookup key.");
                Writer w;
                w.word(i);
                w.bytes(keys[i]);
                requests.push_back({authority(keys[i], comm_->getSize()), std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, comm_, budget_);
    std::vector<OwnerPacket> replies;
    collective(*comm_,
        [&]
        {
            Budget allocation{budget_};
            for (const auto& packet : incoming.packets)
            {
                allocation.packet(24);
                allocation.add(packet.bytes.size());
            }
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto index = r.word();
                auto key = r.bytes();
                r.done();
                if (authority(key, comm_->getSize()) != comm_->getRank())
                    throw std::invalid_argument("Misrouted owner lookup.");
                auto it = shard_.find(key);
                Writer w;
                w.word(index);
                w.bytes(key);
                w.word(it == shard_.end() ? 0 : std::uint64_t(it->second.owner) + 1);
                w.word(it == shard_.end() ? 0 : it->second.revision);
                w.word(it == shard_.end() ? std::numeric_limits<std::uint64_t>::max() : it->second.address);
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto incoming_replies = exchange_owner_packets(replies, comm_, budget_);
    std::vector<AdaptiveOwner> result;
    std::vector<bool> seen;
    collective(*comm_,
        [&]
        {
            Budget allocation{budget_};
            for (const auto& key : keys)
            {
                allocation.add(sizeof(AdaptiveOwner) + 1);
                allocation.add(key.size());
            }
            result.resize(keys.size());
            seen.resize(keys.size());
            for (const auto& packet : incoming_replies.packets)
            {
                Reader r{packet.bytes};
                auto index = r.word();
                auto key = r.bytes();
                auto owner = r.word(), revision = r.word(), address = r.word();
                r.done();
                if (index >= keys.size() || seen[index] || key != keys[index] ||
                    packet.peer != authority(key, comm_->getSize()) || owner > std::uint64_t(comm_->getSize()))
                    throw std::invalid_argument("Invalid adaptive owner reply.");
                if (require_all && !owner)
                    throw std::invalid_argument("Missing adaptive owner.");
                seen[index] = true;
                result[index] = {std::move(key), owner ? int(owner - 1) : -1, revision, address};
            }
            for (bool value : seen)
                if (!value)
                    throw std::invalid_argument("Missing adaptive directory reply.");
        });
    return result;
}

std::size_t DistributedAncestryDirectory::storage_bytes() const noexcept
{
    std::size_t result = sizeof(*this);
    for (const auto& [key, value] : shard_)
        result += sizeof(key) + sizeof(value) + key.capacity() + value.key.capacity() + 3 * sizeof(void*);
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
