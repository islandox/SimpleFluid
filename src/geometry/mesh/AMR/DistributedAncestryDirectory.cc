/** @file DistributedAncestryDirectory.cc
 * @brief Owner-directed exact registration and lookup without global catalogs.
 */
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <limits>
namespace SimpleFluid::Meshes::AMR
{
using namespace adaptive_detail;
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
    std::span<const AdaptiveBytes> keys, bool require_all) const
{
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
