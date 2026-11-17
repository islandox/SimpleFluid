#include "geometry/mesh/AMR/DistributedAdaptiveOwnerOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <limits>
#include <utility>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t links = 4 * sizeof(void*);
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
    void transfer(std::size_t n) { held -= n; }
};
std::size_t product(std::size_t n, std::size_t width)
{
    if (width && n > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("Adaptive overlay allocation overflows.");
    return n * width;
}
template<class T> void exact_resize(std::vector<T>& v, std::size_t n, Scope& scope)
{
    scope.add(product(n, sizeof(T)));
    v.resize(n);
    if (v.capacity() != n)
        throw std::invalid_argument("Adaptive overlay vector capacity differs from admission.");
}
void allocate_bytes(AdaptiveBytes& value, std::size_t size, Scope& scope)
{
    scope.add(size);
    value.resize(size);
    if (value.capacity() != size)
        throw std::invalid_argument("Adaptive overlay byte capacity differs from admission.");
}
void word(AdaptiveBytes& out, std::size_t& offset, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        out.at(offset++) = char(value >> (8 * i));
}
void bytes(AdaptiveBytes& out, std::size_t& offset, std::span<const char> key)
{
    word(out, offset, key.size());
    for (char c : key)
        out.at(offset++) = c;
}
std::span<const char> view(Reader& r)
{
    return r.bytes_view();
}
bool equal(std::span<const char> a, const AdaptiveBytes& b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
std::size_t packets_bytes(const std::vector<OwnerPacket>& packets)
{
    std::size_t n = product(packets.capacity(), sizeof(OwnerPacket));
    for (const auto& p : packets)
    {
        if (p.bytes.capacity() > std::numeric_limits<std::size_t>::max() - n)
            throw std::invalid_argument("Adaptive overlay packet capacities overflow.");
        n += p.bytes.capacity();
    }
    return n;
}
// The transport's cap excludes borrowed packets, already charged by caller.
// Its whole remaining lease is charged throughout the collective; returned
// packet capacity takes over the corresponding part before the lease is freed.
OwnerExchange exchange(
    std::span<const OwnerPacket> input, const Teuchos::RCP<const Teuchos::Comm<int>>& comm, Scope& live)
{
    std::size_t cap = 0;
    collective(*comm,
        [&]
        {
            cap = live.budget.limit - live.budget.used;
            if (!cap)
                throw std::invalid_argument("Adaptive overlay transport lease exhausted.");
            live.add(cap);
        });
    auto result = ::SimpleFluid::Meshes::AMR::adaptive_detail::exchange_adaptive_packets(input, comm, cap);
    collective(*comm,
        [&]
        {
            const auto retained = packets_bytes(result.packets);
            if (retained > cap)
                throw std::invalid_argument("Adaptive overlay transport exceeded lease.");
            live.drop(cap - retained);
        });
    return result;
}
} // namespace
DistributedAdaptiveOwnerOverlay::Lookup::Lookup(Lookup&& other) noexcept
    : budget_(std::exchange(other.budget_, nullptr)), charge_(std::exchange(other.charge_, 0)),
      values_(std::move(other.values_))
{
}
DistributedAdaptiveOwnerOverlay::Lookup::~Lookup()
{
    // Destroy the actual storage before releasing the live charge.
    std::vector<AdaptiveOwner>().swap(values_);
    if (budget_)
        budget_->used -= charge_;
}
DistributedAdaptiveOwnerOverlay::DistributedAdaptiveOwnerOverlay(
    const DistributedAncestryDirectory& base, Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget)
    : base_(base), comm_(std::move(comm)), budget_(budget)
{
    if (comm_.is_null())
        throw std::invalid_argument("Invalid adaptive owner overlay communicator.");
}
DistributedAdaptiveOwnerOverlay::~DistributedAdaptiveOwnerOverlay()
{
    shard_.clear();
    budget_.used -= retained_;
}
DistributedAdaptiveOwnerOverlay::Lookup DistributedAdaptiveOwnerOverlay::lookup(
    std::span<const AdaptiveBytes> keys, bool require_all) const
{
    Scope live{budget_};
    Lookup result;
    std::vector<OwnerPacket> requests;
    collective(*comm_,
        [&]
        {
            exact_resize(requests, keys.size(), live);
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (keys[i].empty() || keys[i].size() > std::numeric_limits<std::size_t>::max() - 16)
                    throw std::invalid_argument("Invalid adaptive overlay query key.");
                auto& p = requests[i];
                p.peer = authority(keys[i], comm_->getSize());
                allocate_bytes(p.bytes, 16 + keys[i].size(), live);
                std::size_t off = 0;
                word(p.bytes, off, i);
                bytes(p.bytes, off, keys[i]);
            }
        });
    auto incoming = exchange(requests, comm_, live);
    auto request_charge = packets_bytes(requests);
    std::vector<OwnerPacket>().swap(requests);
    live.drop(request_charge);
    std::vector<OwnerPacket> replies;
    collective(*comm_,
        [&]
        {
            exact_resize(replies, incoming.packets.size(), live);
            for (std::size_t i = 0; i < incoming.packets.size(); ++i)
            {
                const auto& p = incoming.packets[i];
                Reader r{p.bytes};
                auto index = r.word();
                auto key = view(r);
                r.done();
                if (key.empty() || authority(key, comm_->getSize()) != comm_->getRank() ||
                    key.size() > std::numeric_limits<std::size_t>::max() - 40)
                    throw std::invalid_argument("Misrouted adaptive overlay query.");
                // Map comparator needs one owning key; charge before constructing it.
                Scope temporary{budget_};
                temporary.add(key.size());
                AdaptiveBytes probe(key.begin(), key.end());
                if (probe.capacity() != key.size())
                    throw std::invalid_argument("Overlay lookup key capacity grew.");
                auto it = shard_.find(probe);
                auto& reply = replies[i];
                reply.peer = p.peer;
                allocate_bytes(reply.bytes, 40 + key.size(), live);
                std::size_t off = 0;
                word(reply.bytes, off, index);
                bytes(reply.bytes, off, key);
                word(reply.bytes, off, it == shard_.end() ? 0 : it->second.active ? 2 : 1);
                word(reply.bytes, off,
                    it != shard_.end() && it->second.active ? std::uint64_t(it->second.value.owner) + 1 : 0);
                word(reply.bytes, off, it == shard_.end() ? 0 : it->second.value.revision);
            }
        });
    auto incoming_charge = packets_bytes(incoming.packets);
    incoming = {};
    live.drop(incoming_charge);
    auto received = exchange(replies, comm_, live);
    auto reply_charge = packets_bytes(replies);
    std::vector<OwnerPacket>().swap(replies);
    live.drop(reply_charge);
    std::vector<unsigned char> seen;
    std::vector<AdaptiveBytes> missing;
    std::vector<std::size_t> positions;
    std::size_t result_charge = 0;
    collective(*comm_,
        [&]
        {
            exact_resize(result.values_, keys.size(), live);
            result_charge = product(keys.size(), sizeof(AdaptiveOwner));
            exact_resize(seen, keys.size(), live);
            // Preallocate the maximum possible missing slots. Keys are copied only
            // for true overlay misses; tombstones never reach the base directory.
            exact_resize(missing, keys.size(), live);
            exact_resize(positions, keys.size(), live);
            std::size_t n = 0;
            for (const auto& p : received.packets)
            {
                Reader r{p.bytes};
                auto index = r.word();
                auto key = view(r);
                auto tag = r.word(), owner = r.word(), revision = r.word();
                r.done();
                if (index >= keys.size() || seen[index] || !equal(key, keys[index]) ||
                    p.peer != authority(key, comm_->getSize()) || tag > 2 ||
                    (tag == 2 ? !owner || owner > std::uint64_t(comm_->getSize()) : owner != 0))
                    throw std::invalid_argument("Invalid adaptive overlay reply.");
                seen[index] = 1;
                auto& value = result.values_[index];
                allocate_bytes(value.key, key.size(), live);
                result_charge += key.size();
                std::copy(key.begin(), key.end(), value.key.begin());
                value.owner = tag == 2 ? int(owner - 1) : -1;
                value.revision = revision;
                if (!tag)
                {
                    allocate_bytes(missing[n], key.size(), live);
                    std::copy(key.begin(), key.end(), missing[n].begin());
                    positions[n++] = index;
                }
            }
            for (auto value : seen)
                if (!value)
                    throw std::invalid_argument("Missing adaptive overlay reply.");
            missing.resize(n);
            positions.resize(n);
        });
    auto receive_charge = packets_bytes(received.packets);
    received = {};
    live.drop(receive_charge);
    std::size_t cap = 0;
    collective(*comm_,
        [&]
        {
            cap = budget_.limit - budget_.used;
            if (!cap)
                throw std::invalid_argument("Adaptive overlay base lookup lease exhausted.");
            live.add(cap);
        });
    auto base = base_.lookup(missing, false, cap);
    collective(*comm_,
        [&]
        {
            if (base.size() != positions.size())
                throw std::logic_error("Invalid base lookup size.");
            for (std::size_t i = 0; i < base.size(); ++i)
            {
                if (base[i].key != result.values_[positions[i]].key)
                    throw std::invalid_argument("Invalid base lookup key.");
                auto& target = result.values_[positions[i]];
                target.owner = base[i].owner;
                target.revision = base[i].revision;
                target.address = base[i].address;
            }
            if (require_all)
                for (const auto& value : result.values_)
                    if (value.owner < 0)
                        throw std::invalid_argument("Missing adaptive overlay owner.");
        });
    std::vector<AdaptiveOwner>().swap(base);
    live.drop(cap);
    result.budget_ = &budget_;
    result.charge_ = result_charge;
    live.transfer(result_charge);
    return result;
}
void DistributedAdaptiveOwnerOverlay::apply(
    std::span<const AdaptiveOwner> retired, std::span<const AdaptiveOwner> inserted)
{
    Scope live{budget_};
    std::vector<AdaptiveBytes> keys;
    collective(*comm_,
        [&]
        {
            if (inserted.size() > std::numeric_limits<std::size_t>::max() - retired.size())
                throw std::invalid_argument("Adaptive overlay delta size overflows.");
            exact_resize(keys, retired.size() + inserted.size(), live);
            std::size_t i = 0;
            for (auto group : {retired, inserted})
                for (const auto& value : group)
                {
                    if (value.key.empty() || value.owner != comm_->getRank())
                        throw std::invalid_argument("Invalid local adaptive overlay delta.");
                    allocate_bytes(keys[i], value.key.size(), live);
                    std::copy(value.key.begin(), value.key.end(), keys[i++].begin());
                }
        });
    auto before = lookup(keys, false);
    std::vector<OwnerPacket> packets;
    collective(*comm_,
        [&]
        {
            exact_resize(packets, keys.size(), live);
            std::size_t i = 0;
            for (unsigned kind = 0; kind < 2; ++kind)
                for (const auto& value : kind ? inserted : retired)
                {
                    const auto& previous = before.values()[i];
                    if (!kind && (previous.owner != value.owner || previous.revision != value.revision))
                        throw std::invalid_argument("Stale adaptive overlay retirement.");
                    if (value.key.size() > std::numeric_limits<std::size_t>::max() - 40)
                        throw std::invalid_argument("Adaptive overlay delta packet overflows.");
                    auto& packet = packets[i++];
                    packet.peer = authority(value.key, comm_->getSize());
                    allocate_bytes(packet.bytes, 40 + value.key.size(), live);
                    std::size_t offset = 0;
                    bytes(packet.bytes, offset, value.key);
                    word(packet.bytes, offset, kind);
                    word(packet.bytes, offset, value.revision);
                    word(packet.bytes, offset, previous.owner < 0 ? 0 : std::uint64_t(previous.owner) + 1);
                    word(packet.bytes, offset, previous.revision);
                }
        });
    auto updates = exchange(packets, comm_, live);
    const auto packet_charge = packets_bytes(packets);
    std::vector<OwnerPacket>().swap(packets);
    live.drop(packet_charge);
    struct Change
    {
        bool has_retired = false, has_inserted = false;
        int prior_owner = -2, retire_owner = -1, insert_owner = -1;
        std::uint64_t prior_revision = 0, retire_revision = 0, insert_revision = 0;
    };
    std::map<AdaptiveBytes, Change> changes;
    std::map<AdaptiveBytes, Entry> staged;
    std::size_t staged_charge = 0, replaced_charge = 0;
    collective(*comm_,
        [&]
        {
            for (const auto& packet : updates.packets)
            {
                Reader r{packet.bytes};
                auto key = view(r);
                auto kind = r.word(), revision = r.word();
                auto previous = r.word(), previous_revision = r.word();
                r.done();
                if (key.empty() || kind > 1 || previous > std::uint64_t(comm_->getSize()) ||
                    authority(key, comm_->getSize()) != comm_->getRank())
                    throw std::invalid_argument("Invalid adaptive overlay update packet.");
                Scope temporary{budget_};
                temporary.add(key.size());
                AdaptiveBytes probe(key.begin(), key.end());
                if (probe.capacity() != key.size())
                    throw std::invalid_argument("Overlay update key capacity grew.");
                auto it = changes.find(probe);
                if (it == changes.end())
                {
                    live.add(sizeof(decltype(changes)::value_type) + links);
                    live.add(key.size());
                    it = changes.emplace(probe, Change{}).first;
                    if (it->first.capacity() != key.size())
                        throw std::invalid_argument("Overlay change key capacity grew.");
                }
                auto& change = it->second;
                const int prior = previous ? int(previous - 1) : -1;
                if (change.prior_owner != -2 &&
                    (change.prior_owner != prior || change.prior_revision != previous_revision))
                    throw std::invalid_argument("Inconsistent adaptive overlay observations.");
                change.prior_owner = prior;
                change.prior_revision = previous_revision;
                bool& present = kind ? change.has_inserted : change.has_retired;
                if (present)
                    throw std::invalid_argument("Duplicate adaptive overlay delta.");
                present = true;
                if (kind)
                {
                    change.insert_owner = packet.peer;
                    change.insert_revision = revision;
                }
                else
                {
                    change.retire_owner = packet.peer;
                    change.retire_revision = revision;
                }
            }
            for (const auto& [key, change] : changes)
            {
                if (change.has_retired &&
                    (change.prior_owner != change.retire_owner || change.prior_revision != change.retire_revision))
                    throw std::invalid_argument("Stale adaptive overlay retirement.");
                if (change.has_inserted && change.prior_owner >= 0 && !change.has_retired)
                    throw std::invalid_argument("Duplicate adaptive overlay insertion.");
                const auto before_charge = live.held;
                live.add(sizeof(decltype(staged)::value_type) + links);
                live.add(key.size());
                live.add(key.size());
                staged_charge += live.held - before_charge;
                auto [it, ok] = staged.emplace(
                    key, Entry{change.has_inserted,
                             AdaptiveOwner{key, change.has_inserted ? change.insert_owner : change.retire_owner,
                                 change.has_inserted ? change.insert_revision : change.retire_revision}});
                if (!ok || it->first.capacity() != key.size() || it->second.value.key.capacity() != key.size())
                    throw std::invalid_argument("Adaptive overlay staging capacity mismatch.");
                const auto old = shard_.find(key);
                if (old != shard_.end())
                    replaced_charge += sizeof(decltype(shard_)::value_type) + links + old->first.capacity() +
                                       old->second.value.key.capacity();
            }
        });
    // The collective above is the final error vote. All replacement objects
    // and map nodes are allocated; publication uses swaps and node transfers.
    while (!staged.empty())
    {
        auto node = staged.extract(staged.begin());
        auto old = shard_.find(node.key());
        if (old == shard_.end())
            shard_.insert(std::move(node));
        else
            std::swap(old->second, node.mapped());
    }
    // For existing keys the staged map node was destroyed, while the original
    // key remains and the new value moved into place. Both have equal key size.
    retained_ += staged_charge;
    retained_ -= replaced_charge;
    budget_.used -= replaced_charge;
    live.transfer(staged_charge);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
