/** @file DistributedAncestryDirectory.cc
 * @brief Owner-directed exact registration and lookup without global catalogs.
 */
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "geometry/mesh/AMR/DirectoryFactPages.hh"
#include "geometry/mesh/AMR/DirectoryRegistrationAccess.hh"
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
template<class Find>
std::vector<AdaptiveOwner> directory_lookup_leased(std::span<const AdaptiveBytes> keys, bool require_all,
    std::size_t cap, const Teuchos::RCP<const Teuchos::Comm<int>>& comm, Find&& find)
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
                const auto found = find(key);
                auto& reply = replies[i];
                reply.peer = packet.peer;
                reply.bytes.resize(40 + key.size());
                std::size_t offset = 0;
                directory_word(reply.bytes, offset, index);
                directory_key(reply.bytes, offset, key);
                directory_word(reply.bytes, offset, !found ? 0 : std::uint64_t(found->owner) + 1);
                directory_word(reply.bytes, offset, !found ? 0 : found->revision);
                directory_word(
                    reply.bytes, offset, !found ? std::numeric_limits<std::uint64_t>::max() : found->address);
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

struct DistributedAncestryDirectory::SmallShard
{
    std::vector<AdaptiveOwner> entries;
};
struct DistributedAncestryDirectory::Storage
{
    ContactGroupAllocation header;
    DirectoryFactPages facts;
    Storage(ContactGroupAllocation admission, DirectoryFactPages pages)
        : header(std::move(admission)), facts(std::move(pages))
    {
    }
    static constexpr std::size_t header_bytes() noexcept
    {
        return sizeof(Storage) + sizeof(Budget) + 8 * sizeof(void*);
    }
};

DistributedAncestryDirectory::~DistributedAncestryDirectory() = default;
DistributedAncestryDirectory::DistributedAncestryDirectory(DistributedAncestryDirectory&&) = default;
DistributedAncestryDirectory& DistributedAncestryDirectory::operator=(DistributedAncestryDirectory&&) = default;
DistributedAncestryDirectory::DistributedAncestryDirectory(const DistributedAncestryDirectory& source)
    : comm_(source.comm_), budget_(source.budget_), entries_(source.entries_), resident_bytes_(source.resident_bytes_),
      storage_(source.storage_), registration_(source.registration_)
{
    if (source.small_)
        small_ = std::make_unique<SmallShard>(*source.small_);
}
DistributedAncestryDirectory& DistributedAncestryDirectory::operator=(const DistributedAncestryDirectory& source)
{
    if (this != &source)
    {
        DistributedAncestryDirectory copy(source);
        *this = std::move(copy);
    }
    return *this;
}

DistributedAncestryDirectory::DistributedAncestryDirectory(
    std::span<const AdaptiveOwner> owned, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget)
    : DistributedAncestryDirectory(owned, std::move(comm), budget, {})
{
}
DistributedAncestryDirectory::DistributedAncestryDirectory(std::span<const AdaptiveOwner> owned,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget,
    std::optional<std::size_t> registration_transient_cap)
    : DistributedAncestryDirectory(owned, std::move(comm), budget, registration_transient_cap, nullptr, nullptr, false)
{
}

DistributedAncestryDirectory::DistributedAncestryDirectory(std::span<const AdaptiveOwner> owned,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget,
    std::optional<std::size_t> registration_transient_cap, const DirectorySourceCredit* source,
    const FamilyDirectoryProjection* projection, bool persistent_registration)
    : comm_(std::move(comm)), budget_(budget), resident_bytes_(sizeof(*this))
{
    if (comm_.is_null())
        throw std::invalid_argument("Directory needs a communicator.");
    const int mode[4]{int(registration_transient_cap.has_value()), -int(registration_transient_cap.has_value()),
        int(projection != nullptr), -int(projection != nullptr)};
    int agreed[4]{};
    Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MAX, 4, mode, agreed);
    if (agreed[0] != -agreed[1] || agreed[2] != -agreed[3])
        throw std::invalid_argument("Directory registration lease/projection presence differs across ranks.");
    const auto epoch = projection ? projection->topology_ : 0;
    const std::uint64_t epoch_pair[2]{epoch, ~epoch};
    std::uint64_t epoch_agreed[2]{};
    Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MAX, 2, epoch_pair, epoch_agreed);
    if (epoch_agreed[0] != ~epoch_agreed[1])
        throw std::invalid_argument("Directory projection epoch differs across ranks.");
    const auto cap = registration_transient_cap.value_or(budget_);
    const auto extra = source ? source->facts_ : 0;
    collective(*comm_,
        [&]
        {
            if (!cap || cap > budget_ || (projection && owned.size() != projection->size()))
                throw std::invalid_argument("Invalid directory registration lease or projection coverage.");
            if (source)
            {
                const auto& previous = *source->predecessor_;
                const auto root =
                    previous.storage_ ? previous.storage_->facts.root_owner() : std::shared_ptr<const void>{};
                const auto bytes = previous.storage_ ? previous.storage_->facts.fact_bytes() : 0;
                if (previous.comm_.getRawPtr() != comm_.getRawPtr() || previous.budget_ != budget_ ||
                    root.get() != source->root_.get() || root.owner_before(source->root_) ||
                    source->root_.owner_before(root) || bytes != extra || extra > *source->held_ ||
                    *source->held_ > source->arena_->used || source->arena_->used > source->arena_->limit)
                    throw std::invalid_argument("Directory predecessor lacks its exact live fact credit.");
            }
        });

    std::vector<OwnerPacket> packets;
    collective(*comm_,
        [&]
        {
            const auto fixed = projection ? std::size_t{32} : std::size_t{24};
            Budget input{cap};
            directory_array(input, owned.size(), sizeof(OwnerPacket));
            packets.resize(owned.size());
            if (packets.capacity() != owned.size())
                throw std::invalid_argument("Directory registration packet capacity grew.");
            for (std::size_t i = 0; i < owned.size(); ++i)
            {
                const auto& entry = owned[i];
                if (entry.key.empty() || entry.owner != comm_->getRank())
                    throw std::invalid_argument("Directory registration must name actual local owner.");
                const bool marker = projection && projection->marker(entry, comm_->getRank());
                input.add(persistent_ordered_detail::checked_sum(fixed, entry.key.size()));
                auto& packet = packets[i];
                packet.peer = authority(entry.key, comm_->getSize());
                packet.bytes.resize(fixed + entry.key.size());
                if (packet.bytes.capacity() != packet.bytes.size())
                    throw std::invalid_argument("Directory registration payload capacity grew.");
                std::size_t offset = 0;
                directory_key(packet.bytes, offset, entry.key);
                directory_word(packet.bytes, offset, entry.revision);
                directory_word(packet.bytes, offset, entry.address);
                if (projection)
                    directory_word(packet.bytes, offset, marker);
            }
        });
    std::size_t exchange_cap = budget_;
    collective(*comm_,
        [&]
        {
            if (registration_transient_cap)
            {
                const auto resident = directory_packets(packets, cap);
                if (resident && resident >= cap)
                    throw std::invalid_argument("Directory registration exchange lease exhausted.");
                exchange_cap = cap - resident;
            }
        });
    auto incoming = exchange_owner_packets(packets, comm_, exchange_cap);
    registration_ = incoming.stats;
    std::vector<OwnerPacket>().swap(packets);
    collective(*comm_,
        [&]
        {
            const auto packet_bytes = directory_packets(incoming.packets, cap);
            Budget transient{cap};
            transient.add(sizeof(*this));
            transient.add(packet_bytes);
            directory_array(transient, incoming.packets.size(), sizeof(DirectoryRegistrationView));
            std::vector<DirectoryRegistrationView> rows;
            rows.reserve(incoming.packets.size());
            if (rows.capacity() != incoming.packets.size())
                throw std::invalid_argument("Directory validation view capacity differs from admission.");
            for (const auto& packet : incoming.packets)
            {
                Reader reader{packet.bytes};
                const auto key = reader.bytes_view();
                const auto revision = reader.word(), address = reader.word();
                const auto marker = projection ? reader.word() : 0;
                reader.done();
                if (key.empty() || packet.peer < 0 || packet.peer >= comm_->getSize() ||
                    authority(key, comm_->getSize()) != comm_->getRank() || marker > 1 || (marker && revision != epoch))
                    throw std::invalid_argument("Duplicate, malformed or misrouted adaptive registration.");
                rows.push_back({key, packet.peer, revision, address, bool(marker)});
            }
            std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b)
                { return std::lexicographical_compare(a.key.begin(), a.key.end(), b.key.begin(), b.key.end()); });
            if (std::adjacent_find(rows.begin(), rows.end(),
                    [](const auto& a, const auto& b) { return std::ranges::equal(a.key, b.key); }) != rows.end())
                throw std::invalid_argument("Duplicate adaptive owner.");
            entries_ = rows.size();
            // Bounded literal compatibility has one key per value and no
            // arena or page controller. Its entire storage remains metadata.
            const bool raw_compatibility = !PLIC::geometry_encoding_capacity_profile_supported();
            if (!rows.empty() &&
                (raw_compatibility || (!persistent_registration && !projection && !source && rows.size() <= 32)))
            {
                using persistent_ordered_detail::checked_product;
                using persistent_ordered_detail::checked_sum;
                const auto header = sizeof(SmallShard);
                auto bytes = checked_sum(header, checked_product(rows.size(), sizeof(AdaptiveOwner)));
                for (const auto& row : rows)
                    bytes = checked_sum(bytes, row.key.size());
                transient.add(bytes);
                auto small = std::make_unique<SmallShard>();
                small->entries.reserve(rows.size());
                if (small->entries.capacity() != rows.size())
                    throw std::invalid_argument("Small directory capacity differs from admission.");
                for (const auto& row : rows)
                {
                    small->entries.push_back(
                        {AdaptiveBytes(row.key.begin(), row.key.end()), row.owner, row.revision, row.address});
                    if (small->entries.back().key.capacity() != row.key.size())
                        throw std::invalid_argument("Small directory key capacity differs from admission.");
                }
                small_ = std::move(small);
                resident_bytes_ = checked_sum(sizeof(*this), bytes);
                return;
            }
            const auto* predecessor =
                source && source->predecessor_->storage_ ? &source->predecessor_->storage_->facts : nullptr;
            if (rows.empty() && (!predecessor || !predecessor->fact_bytes()))
                return;
            const auto nested_cap = persistent_ordered_detail::checked_sum(cap - transient.used, extra);
            if (Storage::header_bytes() > nested_cap)
                throw std::invalid_argument("Directory storage header budget exhausted.");
            auto arena = std::make_shared<Budget>(Budget{nested_cap});
            ContactGroupAllocation header(arena);
            header.add(Storage::header_bytes());
            auto facts = DirectoryFactPages::reconcile(predecessor, rows, arena, epoch);
            storage_ = std::make_shared<Storage>(std::move(header), std::move(facts));
            resident_bytes_ = sizeof(*this) + Storage::header_bytes() + storage_->facts.storage_bytes();
            const auto shared = storage_->facts.shared_fact_bytes();
            if (shared > extra || resident_bytes_ - shared > cap || resident_bytes_ > budget_)
                throw std::invalid_argument("Bounded directory resident storage exceeds lease.");
        });
}

std::optional<DistributedAncestryDirectory::Value> DistributedAncestryDirectory::find(
    std::span<const char> key) const noexcept
{
    if (small_)
    {
        const auto found = std::lower_bound(small_->entries.begin(), small_->entries.end(), key,
            [](const AdaptiveOwner& row, std::span<const char> value)
            { return std::lexicographical_compare(row.key.begin(), row.key.end(), value.begin(), value.end()); });
        if (found != small_->entries.end() && std::ranges::equal(found->key, key))
            return Value{found->owner, found->revision, found->address};
        return {};
    }
    if (!storage_)
        return {};
    if (const auto row = storage_->facts.find(key))
        return Value{row->owner, row->revision, row->address};
    return {};
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
        return directory_lookup_leased(
            keys, require_all, *transient_cap, comm_, [&](std::span<const char> key) { return find(key); });
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
                const auto it = find(key);
                Writer w;
                w.word(index);
                w.bytes(key);
                w.word(!it ? 0 : std::uint64_t(it->owner) + 1);
                w.word(!it ? 0 : it->revision);
                w.word(!it ? std::numeric_limits<std::uint64_t>::max() : it->address);
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
    return resident_bytes_;
}
DirectorySourceCredit::DirectorySourceCredit(
    const DistributedAncestryDirectory& source, Budget& arena, std::size_t& held)
    : predecessor_(&source),
      root_(source.storage_ ? source.storage_->facts.root_owner() : std::shared_ptr<const void>{}), arena_(&arena),
      held_(&held), facts_(source.storage_ ? source.storage_->facts.fact_bytes() : 0)
{
    if (facts_ > held || held > arena.used || arena.used > arena.limit)
        throw std::invalid_argument("Directory source credit exceeds its live source lease.");
}

bool FamilyDirectoryProjection::marker(const AdaptiveOwner& entry, int rank) const
{
    const auto check = [&](const auto& family)
    {
        if (family.parent.provider == AdaptiveProviderKind::Product && family.revision != topology_)
            throw std::invalid_argument("Family directory token differs from its exact source epoch.");
        const auto revision = family.revision;
        if (entry.owner != rank || entry.key != family.key || entry.revision != revision ||
            entry.address != std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Family directory projection differs from its exact source record.");
        return family.parent.provider == AdaptiveProviderKind::Product;
    };
    if (root_)
    {
        const auto family = root_->find(entry.key);
        if (!family)
            throw std::invalid_argument("Family directory projection has a foreign source key.");
        return check(*family);
    }
    const auto family = bootstrap_->find(entry.key);
    if (family == bootstrap_->end())
        throw std::invalid_argument("Bootstrap directory projection has a foreign source key.");
    return check(family->second);
}

DistributedAncestryDirectory DirectoryRegistrationAccess::derive(std::span<const AdaptiveOwner> rows,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget, std::size_t transient,
    const DirectorySourceCredit* source, const FamilyDirectoryProjection* projection)
{
    return DistributedAncestryDirectory(rows, std::move(comm), budget, transient, source, projection, true);
}
void DirectoryRegistrationAccess::transfer_shared(const DirectorySourceCredit& source,
    const DistributedAncestryDirectory& target, Budget& arena, std::size_t& candidate_held)
{
    const auto& predecessor = *source.predecessor_;
    const auto root = predecessor.storage_ ? predecessor.storage_->facts.root_owner() : std::shared_ptr<const void>{};
    const auto shared = target.storage_ ? target.storage_->facts.shared_fact_bytes() : 0;
    if (source.arena_ != &arena || source.root_.get() != root.get() || source.root_.owner_before(root) ||
        root.owner_before(source.root_) || shared > source.facts_ || shared > *source.held_ ||
        shared > std::numeric_limits<std::size_t>::max() - candidate_held)
        throw std::invalid_argument("Directory shared-fact transfer lacks its exact source lease.");
    *source.held_ -= shared;
    candidate_held += shared;
}
void DirectoryRegistrationAccess::finish_overlap(const DistributedAncestryDirectory& directory) noexcept
{
    if (directory.storage_)
        directory.storage_->facts.finish_overlap();
}
DirectoryPersistenceStatistics DirectoryRegistrationAccess::statistics(
    const DistributedAncestryDirectory& directory) noexcept
{
    if (!directory.storage_)
        return {directory.resident_bytes_ - sizeof(directory), 0, 0, 0, 0, 0, 0};
    const auto& facts = directory.storage_->facts;
    const auto& work = facts.work();
    return {facts.fact_bytes(), facts.numeric_bytes(), facts.shared_fact_bytes(), facts.overlap_bytes(),
        work.created_pages, work.created_page_bytes, work.fact_edits};
}
const DirectoryFactPages* DirectoryRegistrationAccess::facts(const DistributedAncestryDirectory& directory) noexcept
{
    return directory.storage_ ? &directory.storage_->facts : nullptr;
}
void DirectoryRegistrationAccess::visit_storage(
    const DistributedAncestryDirectory& directory, const std::function<void(const void*, std::size_t, bool)>& visit)
{
    visit(&directory, sizeof(directory), false);
    if (directory.small_)
    {
        const auto& entries = directory.small_->entries;
        visit(directory.small_.get(), sizeof(DistributedAncestryDirectory::SmallShard), false);
        if (entries.capacity())
            visit(entries.data(), entries.capacity() * sizeof(AdaptiveOwner), false);
        for (const auto& row : entries)
            if (row.key.capacity())
                visit(row.key.data(), row.key.capacity(), false);
    }
    if (!directory.storage_)
        return;
    visit(directory.storage_.get(), sizeof(DistributedAncestryDirectory::Storage), false);
    visit(directory.storage_->header.keeper().get(), sizeof(Budget), false);
    directory.storage_->facts.visit_storage(
        [&](const void* address, std::size_t bytes, std::size_t, bool numeric) { visit(address, bytes, numeric); });
}
} // namespace SimpleFluid::Meshes::AMR
