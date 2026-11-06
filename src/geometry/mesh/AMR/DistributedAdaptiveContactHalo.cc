/** @file DistributedAdaptiveContactHalo.cc
 * @brief Bounded owner queries over complete canonical point incidence.
 */
#include "DistributedAdaptiveContactHalo.hh"
#include "AdaptiveContactSnapshot.hh"
#include "AdaptiveLeafLayout.hh"
#include "BudgetedOwnerExchange.hh"
#include "ClosedContactIndex.hh"
#include "ContactCoverIndex.hh"
#include "ContactPairSet.hh"
#include "ContactRecordIndex.hh"
#include "ContactPointGroups.hh"
#include "ContactTracePool.hh"
#include "DeferredOwnerCharges.hh"
#include "DistributedAdaptiveClosureRecords.hh"
#include "DistributedAdaptiveIncidence.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::uint64_t halo_magic = 0x53464348414c4f31ULL;
using Kind = ContactHaloMessageKind;
constexpr std::size_t header_bytes = 13 * 8;
constexpr std::size_t map_links = 4 * sizeof(void*);
struct Bounds
{
    AdaptiveContactBounds box;
    bool present = false;
};
void include(Bounds& target, const AdaptiveContactBounds& box)
{
    if (!target.present)
        target = {box, true};
    else
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            target.box.lower[axis] = std::min(target.box.lower[axis], box.lower[axis]);
            target.box.upper[axis] = std::max(target.box.upper[axis], box.upper[axis]);
        }
}
bool overlaps(const Bounds& first, const Bounds& second)
{
    return first.present && second.present && closed_contact_overlap(first.box, second.box);
}
void write_bounds(Writer& w, const Bounds& value)
{
    w.word(value.present);
    if (value.present)
        for (const auto* array : {&value.box.lower, &value.box.upper})
            for (const auto scalar : *array)
                for (const auto word : PLIC::scalar_packet(scalar))
                    w.word(word);
}
Bounds read_bounds(Reader& r)
{
    Bounds result;
    const auto present = r.word();
    if (present > 1)
        throw std::invalid_argument("Invalid spatial closure cover presence.");
    result.present = present;
    if (result.present)
    {
        for (auto* array : {&result.box.lower, &result.box.upper})
            for (auto& scalar : *array)
            {
                PLIC::ScalarPacketWords words;
                for (auto& word : words)
                    word = r.word();
                scalar = PLIC::scalar_from_packet(words);
                if (!std::isfinite(scalar))
                    throw std::invalid_argument("Nonfinite spatial closure cover.");
            }
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (result.box.upper[axis] < result.box.lower[axis])
                throw std::invalid_argument("Inverted spatial closure cover.");
    }
    return result;
}
bool contains(const AdaptiveContactBounds& box, const std::array<long double, 3>& point)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (point[axis] < box.lower[axis] || box.upper[axis] < point[axis])
            return false;
    return true;
}
void replace_word(AdaptiveBytes& bytes, std::size_t offset, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.at(offset + i) = char(value >> (8 * i));
}
} // namespace

namespace
{
constexpr auto no_slot = std::numeric_limits<std::size_t>::max();

struct ContactOwnedIdentity
{
    ContactGroupAllocation admission;
    AdaptiveCellKey key;
    AdaptiveBytes domain;
    std::uint64_t revision;
    ContactOwnedIdentity(ContactGroupAllocation charge, const AdaptiveCellKey& key_, std::span<const char> domain_,
        std::uint64_t revision_)
        : admission(std::move(charge)), key(key_), domain(domain_.begin(), domain_.end()), revision(revision_) {}
    static std::shared_ptr<const ContactOwnedIdentity> capture(const std::shared_ptr<Budget>& arena,
        const AdaptiveCellKey& key, std::span<const char> domain, std::uint64_t revision)
    {
        ContactGroupAllocation charge(arena);
        charge.add(sizeof(ContactOwnedIdentity) + 128 + key_capacity(key) + domain.size());
        auto result = std::make_shared<ContactOwnedIdentity>(std::move(charge), key, domain, revision);
        const auto actual = sizeof(ContactOwnedIdentity) + 128 + key_capacity(result->key) + result->domain.capacity();
        if (actual > result->admission.bytes())
            throw std::invalid_argument("Contact identity capacity differs from admission.");
        result->admission.release(result->admission.bytes() - actual);
        return result;
    }
};
class ContactRowGroups
{
public:
    explicit ContactRowGroups(ContactGroupAllocation charge) : admission_(std::move(charge)) {}
    [[nodiscard]] bool full() const noexcept { return rows_ == 64; }
    [[nodiscard]] std::size_t begin_row(std::size_t points)
    {
        const auto start = wide_.capacity() ? reserve_row(wide_, points) : reserve_row(compact_, points);
        ++rows_;
        return start;
    }
    void append(std::size_t slot)
    {
        if (slot > std::numeric_limits<std::uint32_t>::max() && !wide_.capacity())
        {
            admission_.reserve(wide_, compact_.capacity());
            wide_.assign(compact_.begin(), compact_.end());
            const auto bytes = compact_.capacity() * sizeof(std::uint32_t);
            std::vector<std::uint32_t>().swap(compact_);
            admission_.release(bytes);
        }
        if (!wide_.capacity())
            compact_.push_back(static_cast<std::uint32_t>(slot));
        else
            wide_.push_back(slot);
    }
    [[nodiscard]] std::size_t at(std::size_t i) const
    {
        return wide_.capacity() ? wide_.at(i) : compact_.at(i);
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return admission_.bytes(); }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    template<class T> std::size_t reserve_row(std::vector<T>& values, std::size_t points)
    {
        const auto start = values.size();
        if (points > values.max_size() - start)
            throw std::invalid_argument("Contact point-group selector extent overflows.");
        if (start + points > values.capacity())
        {
            const auto doubled = values.capacity() <= values.max_size() / 2 ? values.capacity() * 2 : 0;
            admission_.reserve(values, std::max(start + points, doubled));
        }
        return start;
    }
    ContactGroupAllocation admission_;
    std::vector<std::uint32_t> compact_;
    std::vector<std::size_t> wide_;
    std::size_t rows_ = 0;
};
struct FrozenContactGroup
{
    std::shared_ptr<const ContactFanBlock> block;
    std::size_t slice = 0;
};
struct ContactRowSlot
{
    IncidenceRowHandle row;
    std::shared_ptr<const ContactRowGroups> groups;
    std::size_t group_start = 0;
    const Budget* origin = nullptr;
};
struct ContactRecordSlot
{
    const AdaptiveLeaf* local_leaf = nullptr;
    std::shared_ptr<const ContactOwnedIdentity> owned;
    std::size_t row = no_slot;
    ContactBindingId binding{no_slot, 0};
    int owner = -1;
    [[nodiscard]] const AdaptiveCellKey& key() const { return local_leaf ? local_leaf->key : owned->key; }
    [[nodiscard]] const AdaptiveBytes& domain() const { return local_leaf ? local_leaf->domain : owned->domain; }
    [[nodiscard]] std::uint64_t revision() const { return local_leaf ? local_leaf->revision : owned->revision; }
};
struct ContactWorkSlot
{
    std::shared_ptr<const ContactBody> contact;
    const Budget* origin = nullptr;
    std::size_t bytes = 0, remote_refs = 0;
};
} // namespace

struct AdaptiveContactSnapshot::Storage
{
    ContactGroupAllocation admission;
    std::shared_ptr<const AdaptiveLeafLayout> layout;
    ContactSlotPages<ContactRecordSlot> records;
    ContactSlotPages<ContactRowSlot> rows;
    ContactRecordIndex index;
    ContactSlotPages<FrozenContactGroup> groups;
    ContactSlotPages<std::size_t> bindings;
    ContactSlotPages<PointFactDirectory::FrozenPoint> remote_points;
    int rank;
    struct Fan
    {
        const Storage& storage;
        const FrozenContactGroup& group;
        [[nodiscard]] std::size_t local_count() const { return group.block->local_count(group.slice); }
        [[nodiscard]] std::size_t remote_count() const { return group.block->remote_count(group.slice); }
        struct Local
        {
            std::size_t id;
            const ContactPointArray& points;
            std::size_t point;
            [[nodiscard]] std::size_t record() const noexcept { return id; }
            [[nodiscard]] ContactPointIdentity physical() const { return points.at(point).physical; }
        };
        [[nodiscard]] Local local(std::size_t i) const
        {
            const auto [binding, point] = group.block->local(group.slice, i);
            const auto record_slot = storage.bindings[binding];
            if (!record_slot)
                throw std::invalid_argument("Frozen contact binding has no record.");
            const auto id = record_slot - 1;
            const auto& record = storage.records[id];
            return {id, storage.rows[record.row].row->points, point};
        }
        [[nodiscard]] std::size_t remote_record(std::size_t i) const
        {
            return group.block->remote(group.slice, i).first;
        }
        [[nodiscard]] ContactFanView::Remote remote(std::size_t i) const
        {
            const auto [record, point] = group.block->remote(group.slice, i);
            return {record, storage.remote_points[point].value()};
        }
    };
    [[nodiscard]] Fan fan(std::size_t row, std::size_t point) const
    {
        const auto& selection = rows[row];
        const auto& group = groups[selection.groups->at(selection.group_start + point)];
        if (!group.block)
            throw std::invalid_argument("Frozen contact row has no exact point group.");
        return {*this, group};
    }
    Storage(ContactGroupAllocation charge, int rank_)
        : admission(std::move(charge)), records(admission.keeper()), rows(admission.keeper()),
          index(admission.keeper()), groups(admission.keeper()), bindings(admission.keeper()),
          remote_points(admission.keeper()), rank(rank_) {}
    Storage(ContactGroupAllocation charge, const Storage& source)
        : admission(std::move(charge)), layout(source.layout), records(source.records, admission.keeper()),
          rows(source.rows, admission.keeper()), index(source.index, admission.keeper()),
          groups(source.groups, admission.keeper()), bindings(source.bindings, admission.keeper()),
          remote_points(source.remote_points, admission.keeper()), rank(source.rank) {}
    auto key_at() const { return [this](std::size_t id) -> const AdaptiveCellKey& { return records[id].key(); }; }
    [[nodiscard]] std::optional<std::size_t> find(const AdaptiveCellKey& key) const { return index.find(key, key_at()); }
    void insert(std::size_t id) { index.insert(id, key_at()); }
    void erase(const AdaptiveCellKey& key, std::size_t id) { index.erase(key, id); }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return records.overlap_bytes() + rows.overlap_bytes() + index.overlap_bytes() +
               groups.overlap_bytes() + bindings.overlap_bytes() + remote_points.overlap_bytes();
    }
    void finish_overlap() noexcept
    {
        records.finish_overlap();
        rows.finish_overlap();
        index.finish_overlap();
        groups.finish_overlap();
        bindings.finish_overlap();
        remote_points.finish_overlap();
    }
};

struct DistributedAdaptiveContactHalo::Impl
{
    using Groups = ContactPointGroups<>;
    using GroupId = ContactGroupId;
    struct GroupIdLess
    {
        bool operator()(GroupId a, GroupId b) const noexcept
        {
            return a.slot < b.slot || (a.slot == b.slot && a.generation < b.generation);
        }
    };
    struct PendingFanEdit
    {
        GroupId group;
        std::vector<ContactBindingId> removed;
        std::vector<ContactLocalOccurrence> appended;
        std::vector<ContactRemoteOccurrence> remote;
        bool replace_remote = false;
    };
    struct DirectoryNode
    {
        Bounds left, right;
    };
    struct Frame
    {
        std::uint64_t query, page, total;
        bool final;
    };
    const AdaptiveProviderKernel& kernel;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    AdaptiveClosureOptions options;
    ContactHaloContext context;
    ContactHaloBeforeSend before_send;
    std::shared_ptr<Budget> arena;
    std::shared_ptr<std::size_t> external_lease_bytes = std::make_shared<std::size_t>(0);
    Budget& budget;
    DeferredOwnerCharges deferred_owners;
    IdentityBudget identities;
    std::optional<ContactTracePool> trace_pool;
    ContactHaloStatistics statistics;
    AdaptiveClosureStats diagnostics;
    std::shared_ptr<AdaptiveContactSnapshot::Storage> query;
    std::optional<ContactSlotPages<ContactWorkSlot>> work;
    std::shared_ptr<const AdaptiveContactSnapshot> sealed_contacts;
    std::shared_ptr<const AdaptiveLeafLayout> pending_contact_layout;
    std::size_t unbound_owned_records = 0;
    std::unique_ptr<DistributedAdaptiveIncidenceSession> incidence;
    std::set<GroupId, GroupIdLess> dirty_points;
    ContactPairSet affected_pairs;
    std::set<std::size_t> possibly_unused_records;
    std::set<std::pair<int, std::size_t>> missing_records;
    std::map<std::pair<std::size_t, std::size_t>, ClosureRequirements> failing_pairs;
    std::vector<std::size_t> free_records, retired_records, free_rows;
    std::unique_ptr<Groups> groups;
    std::vector<PendingFanEdit> pending_edits;
    std::vector<std::size_t> pending_positions;
    std::unique_ptr<ContactCoverIndex> index;
    std::vector<ClosedContactCover> covers;
    std::map<int, DirectoryNode> directory;
    std::map<int, std::vector<ClosedContactCover>> neighbors, previous_neighbors;
    Bounds local_cover;
    int ranks, rank, tree_width = 1;
    [[nodiscard]] const ContactRecordSlot& record(std::size_t id) const { return query->records[id]; }
    [[nodiscard]] ContactRecordSlot& edit_record(std::size_t id) { return query->records.edit(id); }
    [[nodiscard]] const ContactWorkSlot& body(std::size_t id) const { return (*work)[id]; }
    [[nodiscard]] ContactWorkSlot& edit_body(std::size_t id) { return work->edit(id); }
    template<class T> void release_borrowed(std::shared_ptr<const T>& owner, std::size_t bytes)
    {
        if (bytes && owner.use_count() > 1)
        {
            if (!deferred_owners.adopt(owner, bytes))
                throw std::invalid_argument("Duplicate borrowed contact owner admission.");
            bytes = 0;
        }
        owner.reset();
        budget.used -= bytes;
    }
    std::size_t page_reservation = 0, external_retained_bytes = 0, fork_overlap_bytes = 0;
    bool ready = false, evaluated = false;
    const char* diagnostic_phase = "construct";

    Impl(const AdaptiveProviderKernel& kernel_, Teuchos::RCP<const Teuchos::Comm<int>> comm_,
        AdaptiveClosureOptions options_, ContactHaloContext context_, ContactHaloBeforeSend before_send_)
        : kernel(kernel_), comm(std::move(comm_)), options(options_), context(context_),
          before_send(std::move(before_send_)), arena(std::make_shared<Budget>(Budget{options.byte_budget})),
          budget(*arena), deferred_owners(budget), trace_pool(std::in_place, budget), affected_pairs(budget),
          work(std::in_place, arena), ranks(comm.is_null() ? 0 : comm->getSize()), rank(comm.is_null() ? -1 : comm->getRank())
    {
        budget.add(sizeof(Impl) + sizeof(Budget) + sizeof(std::size_t) + 128 + sizeof(DistributedAdaptiveContactHalo) +
                   sizeof(Groups));
        groups = std::make_unique<Groups>(arena);
        ContactGroupAllocation charge(arena);
        charge.add(sizeof(AdaptiveContactSnapshot::Storage) + 128);
        query = std::make_shared<AdaptiveContactSnapshot::Storage>(std::move(charge), rank);
    }
    void initialize_collective()
    {
        std::array<std::uint64_t, 10> semantics{context.epoch, context.topology, context.partition, context.round,
            context.transaction, options.byte_budget, options.contact_budget, options.tile_budget,
            std::bit_cast<std::uint64_t>(options.maximum_span_ratio), options.maximum_trace_jump};
        auto root = semantics;
        Teuchos::broadcast(*comm, 0, int(root.size()), root.data());
        collective(*comm,
            [&]
            {
                if (root != semantics || !options.byte_budget || !options.contact_budget || !options.tile_budget ||
                    !std::isfinite(options.maximum_span_ratio) || options.maximum_span_ratio < 1 ||
                    options.maximum_trace_jump != 1)
                    throw std::invalid_argument("Invalid or inconsistent spatial closure context/options.");
                while (tree_width < ranks)
                {
                    if (tree_width > std::numeric_limits<int>::max() / 2)
                        throw std::invalid_argument("Spatial directory rank extent overflows.");
                    tree_width *= 2;
                }
            });
    }
    ~Impl()
    {
        if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            std::fprintf(stderr,
                "ADAPTIVE_HALO rank=%d phase=%s ready=%d evaluated=%d owned=%zu remote=%zu points=%zu dirty_points=%zu "
                "pairs=%zu pair_bytes=%zu records_bytes=%zu index_bytes=%zu used=%zu peak=%zu cumulative=%zu\n",
                rank, diagnostic_phase, int(ready), int(evaluated), statistics.owned_records, statistics.remote_records,
                groups ? groups->size() : 0, dirty_points.size(), affected_pairs.size(), affected_pairs.storage_bytes(),
                (query ? query->records.storage_bytes() : 0), index ? index->storage_bytes() : 0, budget.used, budget.peak,
                budget.cumulative);
        statistics.retained_bytes = budget.used;
        statistics.peak_bytes = budget.peak;
        statistics.cumulative_bytes = budget.cumulative;
        if (options.statistics)
        {
            auto& output = *options.statistics;
            output.cells = diagnostics.cells;
            output.contacts += diagnostics.contacts;
            output.geometry_records += diagnostics.geometry_records;
            output.trace_records += diagnostics.trace_records;
            output.transmitted_geometry_records += diagnostics.transmitted_geometry_records;
            output.decoded_geometry_records += diagnostics.decoded_geometry_records;
            output.unique_record_bytes += diagnostics.unique_record_bytes;
            output.replicated_record_bytes += diagnostics.replicated_record_bytes;
            output.decoded_geometry_bytes += diagnostics.decoded_geometry_bytes;
            output.cumulative_bytes += budget.cumulative;
            output.live_peak_bytes = std::max(output.live_peak_bytes, budget.peak);
            output.transient_geometry_peak_bytes =
                std::max(output.transient_geometry_peak_bytes, diagnostics.transient_geometry_peak_bytes);
            output.work += statistics.work;
        }
    }
    Writer writer(
        Kind kind, std::uint64_t query = 0, std::uint64_t page = 0, std::uint64_t total = 0, bool final = true)
    {
        Writer w;
        constexpr std::size_t initial = header_bytes + 8 + 6 * PLIC::ScalarPacketWords{}.size() * 8;
        if (page_reservation)
            throw std::invalid_argument("Overlapping spatial page reservations.");
        budget.add(initial);
        page_reservation = initial;
        w.data.reserve(initial);
        if (w.data.capacity() != initial)
            throw std::invalid_argument("Spatial header capacity differs from admission.");
        w.word(halo_magic);
        w.word(2);
        w.word(std::uint64_t(kind));
        w.word(context.epoch);
        w.word(context.topology);
        w.word(context.partition);
        w.word(context.round);
        w.word(context.transaction);
        w.word(rank);
        w.word(query);
        w.word(page);
        w.word(total);
        w.word(final);
        return w;
    }
    Frame read(Reader& r, Kind kind, int source)
    {
        if (r.word() != halo_magic || r.word() != 2 || r.word() != std::uint64_t(kind) || r.word() != context.epoch ||
            r.word() != context.topology || r.word() != context.partition || r.word() != context.round ||
            r.word() != context.transaction || r.word() != std::uint64_t(source))
            throw std::invalid_argument("Foreign or stale spatial closure page.");
        Frame frame{r.word(), r.word(), r.word(), false};
        const auto final = r.word();
        if (final > 1)
            throw std::invalid_argument("Invalid spatial closure completion flag.");
        frame.final = final;
        return frame;
    }
    bool any(bool local)
    {
        int value = local, result = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &value, &result);
        return result;
    }
    std::size_t credit()
    {
        std::uint64_t available = budget.limit - budget.used, common = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &available, &common);
        // Reserve simultaneous sender, receiver, framing, decode and transport
        // scratch. Pages are bounded before packet vectors are constructed.
        return std::min<std::size_t>(256 * 1024, common / 16);
    }
    OwnerExchange exchange(int destination, Writer&& w)
    {
        std::vector<OwnerPacket> outgoing;
        std::size_t outgoing_charge = 0;
        collective(*comm,
            [&]
            {
                if (destination >= 0)
                {
                    Reader header{w.data};
                    (void) header.word();
                    (void) header.word();
                    const auto kind = static_cast<Kind>(header.word());
                    outgoing_charge = sizeof(OwnerPacket) + w.data.capacity();
                    if (page_reservation)
                    {
                        if (page_reservation != w.data.capacity())
                            throw std::invalid_argument("Spatial page reservation disagreement.");
                        budget.add(sizeof(OwnerPacket));
                    }
                    else
                        budget.add(outgoing_charge);
                    outgoing.reserve(1);
                    outgoing.push_back({destination, std::move(w.data)});
                    if (before_send)
                    {
                        before_send(context, kind, outgoing);
                        std::size_t actual = outgoing.capacity() * sizeof(OwnerPacket);
                        for (const auto& packet : outgoing)
                        {
                            if (packet.bytes.capacity() > std::numeric_limits<std::size_t>::max() - actual)
                                throw std::invalid_argument("Spatial packet capacity overflows.");
                            actual += packet.bytes.capacity();
                        }
                        if (actual > outgoing_charge)
                            budget.add(actual - outgoing_charge);
                        else
                            budget.used -= outgoing_charge - actual;
                        outgoing_charge = actual;
                    }
                    ++statistics.pages;
                }
            });
        auto incoming = consume_budgeted_owner_packets(std::move(outgoing), comm, budget, outgoing_charge);
        page_reservation = 0;
        return incoming;
    }
    void retire(OwnerExchange& exchange) { retire_budgeted_owner_packets(exchange, budget); }
    void admit_writer(Writer& w, std::size_t capacity)
    {
        if (capacity > budget.limit - budget.used)
            throw std::invalid_argument("Spatial closure page exceeds available byte credit.");
        const auto previous = page_reservation;
        if (!previous)
            throw std::invalid_argument("Missing spatial page header reservation.");
        if (capacity < previous)
            throw std::invalid_argument("Spatial page credit cannot hold header.");
        budget.add(capacity);
        page_reservation = capacity;
        w.data.reserve(capacity);
        if (w.data.capacity() != capacity)
            throw std::invalid_argument("Spatial closure page capacity differs from admission.");
        budget.used -= previous;
    }
    std::size_t add_record(const AdaptiveCellKey& key, std::uint64_t revision, int owner, std::span<const char> domain)
    {
        if (const auto found = query->find(key))
        {
            const auto& previous = record(*found);
            if (previous.owner == owner && previous.revision() == revision &&
                std::ranges::equal(previous.domain(), domain))
                return *found;
            if (previous.owner == rank || (owner != rank && body(*found).remote_refs))
                throw std::invalid_argument("Conflicting spatial closure cell owner/revision/domain.");
            discard_record(*found);
        }
        const auto id = free_records.empty() ? query->records.size() : free_records.back();
        if (!free_records.empty())
            free_records.pop_back();
        auto& entry = edit_record(id);
        (void) edit_body(id);
        if (owner == rank && query->layout)
        {
            const auto* leaf = query->layout->find(key);
            if (leaf && leaf->revision == revision && std::ranges::equal(leaf->domain, domain))
                entry.local_leaf = leaf;
            else
                ++unbound_owned_records;
        }
        if (!entry.local_leaf)
            entry.owned = ContactOwnedIdentity::capture(arena, key, domain, revision);
        entry.owner = owner;
        query->insert(id);
        if (owner != rank)
        {
            budget.add(sizeof(decltype(missing_records)::value_type) + map_links);
            missing_records.emplace(owner, id);
        }
        return id;
    }
    void build_directory()
    {
        Bounds accumulated = local_cover;
        for (int stride = 1; stride < tree_width; stride *= 2)
        {
            int destination = -1;
            Writer outgoing;
            collective(*comm,
                [&]
                {
                    if (rank % (2 * stride) == stride)
                    {
                        destination = rank - stride;
                        outgoing = writer(Kind::Cover, stride);
                        write_bounds(outgoing, accumulated);
                    }
                });
            auto incoming = exchange(destination, std::move(outgoing));
            collective(*comm,
                [&]
                {
                    if (rank % (2 * stride) == 0)
                    {
                        Bounds right;
                        if (rank + stride < ranks)
                        {
                            if (incoming.packets.size() != 1 || incoming.packets.front().peer != rank + stride)
                                throw std::invalid_argument("Missing spatial directory child cover.");
                            Reader r{incoming.packets.front().bytes};
                            const auto frame = read(r, Kind::Cover, rank + stride);
                            if (frame.query != std::uint64_t(stride) || frame.page || frame.total || !frame.final)
                                throw std::invalid_argument("Invalid spatial directory child completion.");
                            right = read_bounds(r);
                            r.done();
                        }
                        else if (!incoming.packets.empty())
                            throw std::invalid_argument("Unexpected spatial directory child.");
                        budget.add(sizeof(decltype(directory)::value_type) + map_links);
                        directory.emplace(stride, DirectoryNode{accumulated, right});
                        if (right.present)
                            include(accumulated, right.box);
                    }
                    else if (!incoming.packets.empty())
                        throw std::invalid_argument("Foreign spatial directory cover delivery.");
                });
            retire(incoming);
        }
    }
    void discover_neighbors()
    {
        // One requester's metadata envelope traverses the distributed tree at
        // a time. Only overlapping owner fragments return to that requester.
        for (int requester = 0; requester < ranks; ++requester)
        {
            Writer outgoing;
            int destination = -1;
            collective(*comm,
                [&]
                {
                    if (rank == requester)
                    {
                        destination = 0;
                        outgoing = writer(Kind::DirectoryQuery, requester);
                        write_bounds(outgoing, local_cover);
                    }
                });
            auto incoming = exchange(destination, std::move(outgoing));
            Bounds query;
            bool active = false;
            collective(*comm,
                [&]
                {
                    if (rank == 0)
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != requester)
                            throw std::invalid_argument("Missing spatial directory root query.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::DirectoryQuery, requester);
                        if (frame.query != std::uint64_t(requester) || !frame.final)
                            throw std::invalid_argument("Invalid spatial directory root query.");
                        query = read_bounds(r);
                        r.done();
                        active = query.present;
                    }
                    else if (!incoming.packets.empty())
                        throw std::invalid_argument("Foreign spatial directory root query.");
                });
            retire(incoming);
            for (int stride = tree_width / 2; stride; stride /= 2)
            {
                outgoing = Writer{};
                destination = -1;
                bool left_active = false;
                collective(*comm,
                    [&]
                    {
                        if (rank % (2 * stride) == 0)
                        {
                            const auto& node = directory.at(stride);
                            left_active = active && overlaps(query, node.left);
                            if (rank + stride < ranks)
                            {
                                destination = rank + stride;
                                outgoing = writer(Kind::DirectoryQuery, requester, stride);
                                write_bounds(outgoing, active && overlaps(query, node.right) ? query : Bounds{});
                            }
                        }
                    });
                incoming = exchange(destination, std::move(outgoing));
                collective(*comm,
                    [&]
                    {
                        if (rank % (2 * stride) == 0)
                            active = left_active;
                        else if (rank % (2 * stride) == stride)
                        {
                            active = false;
                            if (incoming.packets.size() != 1 || incoming.packets.front().peer != rank - stride)
                                throw std::invalid_argument("Missing spatial directory branch terminal response.");
                            if (!incoming.packets.empty())
                            {
                                if (incoming.packets.size() != 1 || incoming.packets.front().peer != rank - stride)
                                    throw std::invalid_argument("Foreign spatial directory branch.");
                                Reader r{incoming.packets.front().bytes};
                                const auto frame = read(r, Kind::DirectoryQuery, rank - stride);
                                if (frame.query != std::uint64_t(requester) || frame.page != std::uint64_t(stride) ||
                                    !frame.final)
                                    throw std::invalid_argument("Invalid spatial directory branch completion.");
                                query = read_bounds(r);
                                r.done();
                                active = query.present;
                            }
                        }
                        else if (!incoming.packets.empty())
                            throw std::invalid_argument("Unexpected spatial directory branch query.");
                    });
                retire(incoming);
            }
            // Cyclic service bounds both fan-in and the aggregate receive page.
            for (int owner = 0; owner < ranks; ++owner)
            {
                std::size_t cursor = 0, page = 0;
                bool done = !(rank == owner && active && owner != requester);
                bool received_final = owner == requester;
                std::uint64_t expected_cover_page = 0;
                int subscribed = rank == owner && active && owner != requester;
                Teuchos::broadcast(*comm, owner, 1, &subscribed);
                while (any(!done))
                {
                    const auto cap = credit();
                    outgoing = Writer{};
                    destination = -1;
                    collective(*comm,
                        [&]
                        {
                            if (!done)
                            {
                                if (cap <= header_bytes + 8)
                                    throw std::invalid_argument("Insufficient spatial cover page credit.");
                                destination = requester;
                                outgoing = writer(Kind::LeafCovers, requester, page++);
                                admit_writer(outgoing, cap);
                                outgoing.word(0);
                                std::size_t count = 0;
                                while (cursor < covers.size())
                                {
                                    const auto& cover = covers[cursor];
                                    if (!closed_contact_overlap(query.box, cover.bounds))
                                    {
                                        ++cursor;
                                        continue;
                                    }
                                    const auto need =
                                        8 + cover.domain.size() + 8 + 6 * PLIC::ScalarPacketWords{}.size() * 8;
                                    if (need > cap - outgoing.data.size())
                                    {
                                        if (!count)
                                            throw std::invalid_argument("A spatial cover exceeds page credit.");
                                        break;
                                    }
                                    outgoing.bytes(cover.domain);
                                    write_bounds(outgoing, {cover.bounds, true});
                                    ++cursor;
                                    ++count;
                                }
                                done = cursor == covers.size();
                                replace_word(outgoing.data, header_bytes - 8, done);
                                replace_word(outgoing.data, header_bytes, count);
                            }
                        });
                    incoming = exchange(destination, std::move(outgoing));
                    collective(*comm,
                        [&]
                        {
                            if (rank == requester && !incoming.packets.empty())
                            {
                                if (received_final || incoming.packets.size() != 1 ||
                                    incoming.packets.front().peer != owner)
                                    throw std::invalid_argument("Duplicate or foreign owner-cover page.");
                                Reader r{incoming.packets.front().bytes};
                                const auto frame = read(r, Kind::LeafCovers, owner);
                                if (frame.page != expected_cover_page++)
                                    throw std::invalid_argument("Missing or duplicate owner-cover page.");
                                auto found = neighbors.find(owner);
                                if (found == neighbors.end())
                                {
                                    budget.add(sizeof(decltype(neighbors)::value_type) + map_links);
                                    found = neighbors.try_emplace(owner).first;
                                }
                                auto n = r.word();
                                if (frame.query != std::uint64_t(requester) || n > r.data.size() / 16)
                                    throw std::invalid_argument("Invalid owner-cover response count.");
                                for (std::uint64_t i = 0; i < n; ++i)
                                {
                                    const auto domain = r.bytes_view();
                                    const auto bounds = read_bounds(r);
                                    if (domain.empty() || !bounds.present || !overlaps(local_cover, bounds))
                                        throw std::invalid_argument("Foreign owner-cover response.");
                                    budget.add(domain.size());
                                    budget.append_capacity(found->second);
                                    found->second.push_back(
                                        {AdaptiveBytes(domain.begin(), domain.end()), bounds.box, owner, 0});
                                }
                                r.done();
                                received_final = frame.final;
                            }
                            else if (!incoming.packets.empty())
                                throw std::invalid_argument("Foreign owner-cover page delivery.");
                        });
                    retire(incoming);
                }
                collective(*comm,
                    [&]
                    {
                        if (rank == requester && subscribed && !received_final)
                            throw std::invalid_argument("Missing owner-cover terminal response.");
                    });
                // A non-overlapping leaf has no subscription and needs no point
                // queries. Completion of tree traversal is synchronous above.
            }
        }
    }
    bool candidate(int owner, GroupId id) const
    {
        const auto found = neighbors.find(owner);
        if (found == neighbors.end())
            return false;
        const auto normalized = groups->normalized(id);
        const auto& coordinate = PLIC::details::GeometryIdentityAccess::point(normalized);
        for (const auto& cover : found->second)
            if (std::ranges::equal(groups->domain(id), cover.domain) && contains(cover.bounds, coordinate))
                return true;
        return false;
    }
    [[nodiscard]] Groups::OrderedIds ordered_dirty() const
    {
        ContactGroupAllocation scratch(arena);
        std::vector<GroupId> ids;
        scratch.reserve(ids, dirty_points.size());
        ids.assign(dirty_points.begin(), dirty_points.end());
        return groups->ordered_ids(ids);
    }
    PendingFanEdit& stage_edit(GroupId);
    void commit_edits();
    [[nodiscard]] std::size_t metadata_bytes() const;
    void install_row(IncidenceRowHandle, const AdaptiveLeaf&, std::shared_ptr<ContactRowGroups>&);
    void retire_row(std::size_t record);
    void touch_point(GroupId);
    void gather_pairs(GroupId);
    void notify_dirty_points();
    void refresh_covers();
    [[nodiscard]] ContactCoverIndex::GroupResolver cover_resolver() const
    {
        return {this,
            [](const void* pointer, GroupId id) -> ContactCoverIndex::GroupObservation
            {
                const auto& state = *static_cast<const Impl*>(pointer);
                return {PLIC::details::GeometryIdentityAccess::point(state.groups->normalized(id)),
                    &state.groups->domain(id)};
            }};
    }
    void make_cover_index()
    {
        index = query->layout ? std::make_unique<ContactCoverIndex>(arena, rank, cover_resolver()) :
                                std::make_unique<ContactCoverIndex>(budget, rank);
    }
    [[nodiscard]] std::size_t insert_cover(GroupId id)
    {
        return query->layout ? index->insert(id) :
            index->insert(groups->point_directory(), groups->normalized_reference(id), &groups->domain(id));
    }
    void capture_fan(GroupId);
    template<class Visit>
    void with_fan(const ContactRecordSlot& local, std::size_t point, const ContactPointIdentity& normalized,
        Visit&& visit) const
    {
        if (query->layout)
            visit(query->fan(local.row, point));
        else if (const auto group = groups->find({local.domain(), &normalized}))
            visit(groups->borrowed_fan(*group));
        else
            throw std::invalid_argument("Complete closed-contact row has no point group.");
    }
    void attach_record(std::size_t, ContactBody&&, std::size_t);
    void discard_record(std::size_t);
    void remove_remote_points(GroupId);
    void discover_points();
    void fetch_records();
    std::vector<AdaptiveMark> evaluate(bool retain_charge = false);
};

namespace
{
std::size_t row_storage(const ContactIncidenceRow& row)
{
    return sizeof(ContactIncidenceRow) + 128 + key_capacity(row.key) + row.domain.capacity() +
           row.points.storage_bytes();
}
} // namespace

std::size_t DistributedAdaptiveContactHalo::Impl::metadata_bytes() const
{
    std::size_t bytes =
        sizeof(Impl) + sizeof(Budget) + sizeof(std::size_t) + 128 + sizeof(DistributedAdaptiveContactHalo) +
        query->admission.bytes() + (sealed_contacts ? sizeof(AdaptiveContactSnapshot) + 128 : 0) +
        (free_rows.capacity() + free_records.capacity() + retired_records.capacity()) * sizeof(std::size_t) +
        identities.metadata_bytes() + deferred_owners.metadata_bytes();
    const auto group_storage = groups->storage_breakdown();
    bytes += sizeof(Groups) + group_storage.private_metadata_bytes() +
             pending_positions.capacity() * sizeof(std::size_t) + pending_edits.capacity() * sizeof(PendingFanEdit);
    for (const auto& edit : pending_edits)
        bytes += edit.removed.capacity() * sizeof(ContactBindingId) +
                 edit.appended.capacity() * sizeof(ContactLocalOccurrence) +
                 edit.remote.capacity() * sizeof(ContactRemoteOccurrence);
    if (incidence)
        groups->visit_admissions(
            [&](const ContactStorageAdmission& admission)
            {
                if (admission.body_owner &&
                    incidence->point_pool().owns_page_admission(admission.body_owner, admission.metadata_owner))
                    bytes -= admission.private_metadata;
            });
    bytes += covers.capacity() * sizeof(ClosedContactCover);
    for (const auto& cover : covers)
        bytes += cover.domain.capacity();
    for (const auto* map : {&neighbors, &previous_neighbors})
        for (const auto& [rank, values] : *map)
        {
            bytes +=
                sizeof(decltype(neighbors)::value_type) + map_links + values.capacity() * sizeof(ClosedContactCover);
            for (const auto& cover : values)
                bytes += cover.domain.capacity();
        }
    bytes += directory.size() * (sizeof(decltype(directory)::value_type) + map_links) +
             failing_pairs.size() * (sizeof(decltype(failing_pairs)::value_type) + map_links) +
             affected_pairs.storage_bytes() + possibly_unused_records.size() * (sizeof(std::size_t) + map_links) +
             missing_records.size() * (sizeof(decltype(missing_records)::value_type) + map_links);
    bytes += dirty_points.size() * (sizeof(GroupId) + map_links);
    if (index)
        bytes += index->private_metadata_bytes();
    if (incidence)
        bytes += incidence->metadata_bytes();
    if (trace_pool)
        bytes += trace_pool->metadata_bytes();
    return bytes;
}

void DistributedAdaptiveContactHalo::Impl::attach_record(std::size_t id, ContactBody&& value, std::size_t bytes)
{
    const auto transferred = trace_pool->adopt(value);
    if (transferred > bytes)
        throw std::invalid_argument("Contact trace transfer exceeds record admission.");
    bytes -= transferred;
    budget.add(128);
    bytes += 128;
    auto keeper = arena;
    auto& slot = edit_body(id);
    slot.contact = std::shared_ptr<const ContactBody>(new ContactBody(std::move(value)),
        [keeper, bytes](const ContactBody* record)
        {
            delete record;
            keeper->used -= bytes;
        });
    slot.origin = &budget;
    slot.bytes = bytes;
}
DistributedAdaptiveContactHalo::Impl::PendingFanEdit& DistributedAdaptiveContactHalo::Impl::stage_edit(GroupId id)
{
    constexpr auto empty = std::numeric_limits<std::size_t>::max();
    while (pending_positions.size() <= id.slot)
    {
        budget.append_capacity(pending_positions);
        pending_positions.push_back(empty);
    }
    auto& position = pending_positions[id.slot];
    if (position == empty)
    {
        budget.append_capacity(pending_edits);
        position = pending_edits.size();
        pending_edits.push_back({id, {}, {}, {}, false});
    }
    if (pending_edits[position].group != id)
        throw std::invalid_argument("Stale contact fan edit slot.");
    return pending_edits[position];
}
void DistributedAdaptiveContactHalo::Impl::commit_edits()
{
    auto catalog = groups->catalog();
    for (const auto& edit : pending_edits)
    {
        const auto old = groups->fan(edit.group);
        for (std::size_t i = 0; i < edit.appended.size(); ++i)
        {
            const auto record = catalog->binding(edit.appended[i].binding).record;
            if (this->record(record).domain() != groups->domain(edit.group))
                throw std::invalid_argument("Contact occurrence does not belong to its connected domain.");
            for (std::size_t j = 0; j < i; ++j)
                if (catalog->binding(edit.appended[j].binding).record == record)
                    throw std::invalid_argument("Duplicate cell incidence at a normalized closure point.");
            for (std::size_t j = 0; j < old.local_count(); ++j)
            {
                const auto occurrence = old.local(j);
                if (occurrence.binding.record == record &&
                    std::find(edit.removed.begin(), edit.removed.end(), occurrence.id) == edit.removed.end())
                    throw std::invalid_argument("Duplicate cell incidence at a normalized closure point.");
            }
        }
    }
    groups->replace_fans_generated(pending_edits.size(),
        [&](std::size_t i)
        {
            const auto& edit = pending_edits[i];
            return Groups::Edit{edit.group, edit.removed, edit.appended, edit.remote, edit.replace_remote};
        });
    std::size_t bytes = pending_edits.capacity() * sizeof(PendingFanEdit);
    for (const auto& edit : pending_edits)
    {
        const auto cover = groups->cover(edit.group);
        if (!groups->fan(edit.group).local_count() && cover != std::numeric_limits<std::size_t>::max())
        {
            if (!index)
                throw std::invalid_argument("Missing contact point cover index.");
            index->erase(cover);
            groups->set_cover(edit.group, std::numeric_limits<std::size_t>::max());
        }
        for (const auto& occurrence : edit.remote)
            catalog->points.release(occurrence.physical);
        pending_positions[edit.group.slot] = std::numeric_limits<std::size_t>::max();
        bytes += edit.removed.capacity() * sizeof(ContactBindingId) +
                 edit.appended.capacity() * sizeof(ContactLocalOccurrence) +
                 edit.remote.capacity() * sizeof(ContactRemoteOccurrence);
    }
    std::vector<PendingFanEdit>().swap(pending_edits);
    budget.used -= bytes;
}
void DistributedAdaptiveContactHalo::Impl::touch_point(GroupId id)
{
    if (dirty_points.contains(id))
        return;
    (void) groups->key(id);
    budget.add(sizeof(GroupId) + map_links);
    dirty_points.insert(id);
    gather_pairs(id);
}
void DistributedAdaptiveContactHalo::Impl::gather_pairs(GroupId id)
{
    const auto fan = groups->fan(id);
    auto add = [&](std::size_t a, std::size_t b)
    {
        if (a == b)
            throw std::invalid_argument("Duplicate cell in complete normalized contact fan.");
        const auto ids = std::minmax(a, b);
        (void) affected_pairs.insert({ids.first, ids.second});
    };
    for (std::size_t i = 0; i < fan.local_count(); ++i)
    {
        const auto record = fan.local(i).binding.record;
        for (std::size_t j = 0; j < i; ++j)
            add(record, fan.local(j).binding.record);
        for (std::size_t j = 0; j < fan.remote_count(); ++j)
            add(record, fan.remote(j).record);
    }
}
void DistributedAdaptiveContactHalo::Impl::remove_remote_points(GroupId id)
{
    auto& edit = stage_edit(id);
    if (edit.replace_remote)
        return;
    const auto fan = groups->fan(id);
    for (std::size_t i = 0; i < fan.remote_count(); ++i)
    {
        const auto item = fan.remote(i);
        auto& record = edit_body(item.record);
        if (!record.remote_refs)
            throw std::invalid_argument("Remote incidence reference underflow.");
        if (!--record.remote_refs && !possibly_unused_records.contains(item.record))
        {
            budget.add(sizeof(std::size_t) + map_links);
            possibly_unused_records.insert(item.record);
        }
    }
    edit.replace_remote = true;
}
void DistributedAdaptiveContactHalo::Impl::discard_record(std::size_t id)
{
    auto& entry = edit_record(id);
    if (entry.owner < 0)
        return;
    query->erase(entry.key(), id);
    // Tombstone slots are not reused until old pair obligations have retired.
    if (entry.owner != rank && body(id).contact)
        --statistics.remote_records;
    if (entry.owner == rank && query->layout && !entry.local_leaf)
        --unbound_owned_records;
    if (missing_records.erase({entry.owner, id}))
        budget.used -= sizeof(decltype(missing_records)::value_type) + map_links;
    entry.owner = -1;
    budget.append_capacity(retired_records);
    retired_records.push_back(id);
}
void DistributedAdaptiveContactHalo::Impl::install_row(
    IncidenceRowHandle handle, const AdaptiveLeaf& leaf, std::shared_ptr<ContactRowGroups>& selectors)
{
    const auto& row = *handle;
    if (leaf.domain.empty() || row.points.empty() || row.revision != leaf.revision ||
        (query->layout ? !row.domain.empty() : (row.key != leaf.key || row.domain != leaf.domain)))
        throw std::invalid_argument("Spatial incidence row disagrees with its source owner.");
    const auto row_index = free_rows.empty() ? query->rows.size() : free_rows.back();
    if (!free_rows.empty())
        free_rows.pop_back();
    auto& row_slot = query->rows.edit(row_index);
    row_slot = {std::move(handle), {}, 0, &budget};
    if (query->layout)
    {
        if (!selectors || selectors->full())
        {
            ContactGroupAllocation charge(arena);
            charge.add(sizeof(ContactRowGroups) + 128);
            selectors = std::make_shared<ContactRowGroups>(std::move(charge));
        }
        row_slot.group_start = selectors->begin_row(row.points.size());
        row_slot.groups = selectors;
    }
    const auto id = add_record(leaf.key, leaf.revision, rank, leaf.domain);
    auto& entry = edit_record(id);
    entry.row = row_index;
    if (!body(id).contact)
    {
        const auto before = budget.used, identities_before = identities.admitted_bytes();
        auto record = prepare_contact_body(
            leaf, kernel, budget, identities, diagnostics.trace_records, diagnostics.transient_geometry_peak_bytes);
        attach_record(id, std::move(record), budget.used - before - (identities.admitted_bytes() - identities_before));
        ++statistics.owned_records;
        ++diagnostics.geometry_records;
    }
    entry.binding = groups->catalog()->bind(row_slot.row, id);
    if (query->layout)
        query->bindings.edit(entry.binding.slot) = id + 1;
    for (std::size_t point_index = 0; point_index < row.points.size(); ++point_index)
    {
        const auto& point = row.points[point_index];
        const auto group = groups->intern(leaf.domain, point.normalized, row.points.normalized_locator(point_index),
            incidence ? &incidence->point_pool() : nullptr);
        if (selectors)
            selectors->append(group.slot);
        touch_point(group);
        if (ready && groups->cover(group) == std::numeric_limits<std::size_t>::max())
        {
            if (!index)
                make_cover_index();
            groups->set_cover(group, insert_cover(group));
        }
        auto& edit = stage_edit(group);
        budget.append_capacity(edit.appended);
        edit.appended.push_back({entry.binding, point_index});
        ++diagnostics.contacts;
    }
}
void DistributedAdaptiveContactHalo::Impl::retire_row(std::size_t record)
{
    const auto& entry = this->record(record);
    const auto row_index = entry.row;
    const auto& row = *query->rows[row_index].row;
    const auto binding = entry.binding;
    for (const auto& point : row.points)
    {
        const auto group = groups->find({entry.domain(), &point.normalized});
        if (!group)
            throw std::invalid_argument("Retired contact point has no group.");
        touch_point(*group);
        auto& edit = stage_edit(*group);
        budget.append_capacity(edit.removed);
        edit.removed.push_back(binding);
    }
    groups->catalog()->release(binding);
    if (query->layout)
        query->bindings.edit(binding.slot) = 0;
    auto& changing = edit_record(record);
    changing.binding = {no_slot, 0};
    changing.row = no_slot;
    auto& row_slot = query->rows.edit(row_index);
    release_borrowed(row_slot.row, row_slot.origin != &budget ? row_storage(row) : 0);
    row_slot.groups.reset();
    row_slot.group_start = 0;
    row_slot.origin = nullptr;
    budget.append_capacity(free_rows);
    free_rows.push_back(row_index);
}
void DistributedAdaptiveContactHalo::Impl::capture_fan(GroupId id)
{
    const auto& fan = groups->borrowed_fan(id);
    if (!fan.local_count())
    {
        if (const auto* old = query->groups.find(id.slot); old && old->block)
            query->groups.edit(id.slot) = {};
        return;
    }
    const auto& block = fan.owner()->frozen_block();
    query->groups.edit(id.slot) = {block, fan.group()};
    const auto& facts = groups->point_directory();
    for (std::size_t i = 0; i < fan.remote_count(); ++i)
    {
        const auto slot = block->remote(fan.group(), i).second;
        if (const auto* point = query->remote_points.find(slot); !point || !*point)
            query->remote_points.edit(slot) = facts.frozen(facts.pinned_reference(slot));
    }
    ++statistics.contact_frozen_groups;
}
void DistributedAdaptiveContactHalo::Impl::refresh_covers()
{
    std::size_t bytes = covers.capacity() * sizeof(ClosedContactCover);
    for (const auto& cover : covers)
        bytes += cover.domain.capacity();
    std::vector<ClosedContactCover>().swap(covers);
    budget.used -= bytes;
    local_cover = {};
    if (!ready && groups->size())
    {
        // Bootstrap visits the completed distinct point catalogue once. Later
        // updates maintain leaf handles at the changed point-group boundary.
        make_cover_index();
        index->reserve(groups->size());
        auto ordered = groups->ordered_ids();
        for (const auto id : ordered.ids())
        {
            if (!groups->fan(id).local_count())
                throw std::invalid_argument("Empty bootstrap point group.");
            groups->set_cover(id, insert_cover(id));
        }
    }
    if (index && !index->point_count() && !query->layout)
        index.reset();
    if (!index || !index->point_count())
        return;
    if (const auto bounds = index->aggregate_bounds())
        local_cover = {*bounds, true};
    covers = index->ownership_frontier_covers(8);
}

void DistributedAdaptiveContactHalo::Impl::notify_dirty_points()
{
    auto relevant = [&](int owner, GroupId id)
    {
        if (candidate(owner, id))
            return true;
        const auto previous = previous_neighbors.find(owner);
        if (previous != previous_neighbors.end())
            for (const auto& cover : previous->second)
                if (std::ranges::equal(groups->domain(id), cover.domain) &&
                    contains(cover.bounds, PLIC::details::GeometryIdentityAccess::point(groups->normalized(id))))
                    return true;
        return false;
    };
    // Freeze canonical outgoing order; learned dirty IDs are not echoed.
    std::optional<Groups::OrderedIds> outgoing_order;
    collective(*comm, [&] { outgoing_order.emplace(ordered_dirty()); });
    const auto& outgoing_keys = *outgoing_order;
    for (int offset = 1; offset < ranks; ++offset)
    {
        const int destination = (rank + offset) % ranks;
        const int source = (rank + ranks - offset) % ranks;
        std::size_t cursor = 0;
        std::uint64_t page = 0, expected_page = 0;
        bool sent_final = false, received_final = false;
        while (any(!sent_final))
        {
            const auto cap = credit();
            Writer outgoing;
            int peer = -1;
            collective(*comm,
                [&]
                {
                    if (sent_final)
                        return;
                    if (cap <= header_bytes + 8)
                        throw std::invalid_argument("Dirty contact page has no byte credit.");
                    peer = destination;
                    outgoing = writer(Kind::DirtyPoints, 0, page++);
                    admit_writer(outgoing, cap);
                    outgoing.word(0);
                    std::size_t count = 0;
                    while (cursor < outgoing_keys.ids().size())
                    {
                        const auto id = outgoing_keys.ids()[cursor];
                        const auto key = groups->key(id);
                        if (!relevant(destination, id))
                        {
                            ++cursor;
                            continue;
                        }
                        if (framed_contact_point_key_size(key) > cap - outgoing.data.size())
                        {
                            if (!count)
                                throw std::invalid_argument("Dirty point key exceeds page credit.");
                            break;
                        }
                        append_framed_contact_point_key(outgoing, key);
                        ++count;
                        ++cursor;
                    }
                    sent_final = cursor == outgoing_keys.ids().size();
                    replace_word(outgoing.data, header_bytes, count);
                    replace_word(outgoing.data, header_bytes - 8, sent_final);
                });
            auto incoming = exchange(peer, std::move(outgoing));
            collective(*comm,
                [&]
                {
                    if (incoming.packets.empty())
                        return;
                    if (received_final || incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                        throw std::invalid_argument("Duplicate or foreign dirty point page.");
                    Reader r{incoming.packets.front().bytes};
                    const auto frame = read(r, Kind::DirtyPoints, source);
                    if (frame.query || frame.total || frame.page != expected_page++)
                        throw std::invalid_argument("Missing dirty point page.");
                    const auto count = r.word();
                    if (count > r.data.size() / 8)
                        throw std::invalid_argument("Invalid dirty point count.");
                    for (std::uint64_t i = 0; i < count; ++i)
                    {
                        const auto field = r.bytes_view();
                        if (const auto id = groups->find_encoded(field))
                            touch_point(*id);
                    }
                    r.done();
                    received_final = frame.final;
                });
            retire(incoming);
        }
        collective(*comm,
            [&]
            {
                if (!received_final)
                    throw std::invalid_argument("Missing dirty point terminal reply.");
            });
    }
}

void DistributedAdaptiveContactHalo::Impl::discover_points()
{
    struct Pending
    {
        GroupId point;
        std::uint64_t total = 0, next = 0;
        bool started = false, done = false;
    };
    struct Query
    {
        std::uint64_t id;
        std::optional<GroupId> group;
    };
    std::optional<Groups::OrderedIds> query_order;
    collective(*comm, [&] { query_order.emplace(ordered_dirty()); });
    const auto& ordered = *query_order;
    for (int offset = 1; offset < ranks; ++offset)
    {
        const int destination = (rank + offset) % ranks;
        const int source = (rank + ranks - offset) % ranks;
        auto cursor = ordered.ids().begin();
        std::uint64_t next_query = 0, expected_query = 0, query_page = 0, expected_page = 0;
        while (true)
        {
            while (cursor != ordered.ids().end() &&
                   (!groups->fan(*cursor).local_count() || !candidate(destination, *cursor)))
                ++cursor;
            if (!any(cursor != ordered.ids().end()))
                break;
            const auto cap = credit();
            std::vector<Pending> pending;
            Writer outgoing;
            int peer = -1;
            const auto first_query = next_query;
            collective(*comm,
                [&]
                {
                    if (cursor != ordered.ids().end())
                    {
                        if (cap <= header_bytes + 8)
                            throw std::invalid_argument("Insufficient exact point query page credit.");
                        peer = destination;
                        outgoing = writer(Kind::Points, first_query, query_page++);
                        ++statistics.point_query_pages;
                        admit_writer(outgoing, cap);
                        outgoing.word(0);
                        std::size_t count = 0;
                        while (cursor != ordered.ids().end())
                        {
                            if (!groups->fan(*cursor).local_count() || !candidate(destination, *cursor))
                            {
                                ++cursor;
                                continue;
                            }
                            const auto need = 8 + framed_contact_point_key_size(groups->key(*cursor));
                            if (need > cap - outgoing.data.size())
                            {
                                if (!count)
                                    throw std::invalid_argument("An exact point key exceeds page credit.");
                                break;
                            }
                            outgoing.word(next_query++);
                            append_framed_contact_point_key(outgoing, groups->key(*cursor));
                            budget.append_capacity(pending);
                            pending.push_back({*cursor, 0, 0, false, false});
                            ++statistics.exact_point_queries;
                            ++count;
                            ++cursor;
                        }
                        replace_word(outgoing.data, header_bytes, count);
                        statistics.query_bytes += outgoing.data.size();
                    }
                });
            auto incoming = exchange(peer, std::move(outgoing));
            std::vector<Query> queries;
            collective(*comm,
                [&]
                {
                    if (!incoming.packets.empty())
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                            throw std::invalid_argument("Foreign exact point query page.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::Points, source);
                        const auto n = r.word();
                        if (frame.query != expected_query || frame.page != expected_page++ || !frame.final || !n ||
                            n > r.data.size() / 16)
                            throw std::invalid_argument("Missing, duplicate or invalid exact point query page.");
                        for (std::uint64_t i = 0; i < n; ++i)
                        {
                            const auto id = r.word();
                            if (id != expected_query++)
                                throw std::invalid_argument("Nonsequential exact point query identity.");
                            const auto bytes = r.bytes_view();
                            budget.append_capacity(queries);
                            queries.push_back({id, groups->find_encoded(bytes)});
                        }
                        r.done();
                    }
                });
            // Queries are now pointers into immutable owner-local reverse rows;
            // encoded query pages can be retired before producing any fan page.
            retire(incoming);
            std::size_t query_cursor = 0, occurrence_cursor = 0;
            std::uint64_t response_page = 0, expected_response_page = 0;
            bool received_final = pending.empty();
            while (any(query_cursor < queries.size()))
            {
                const auto reply_cap = credit();
                outgoing = Writer{};
                peer = -1;
                collective(*comm,
                    [&]
                    {
                        if (query_cursor < queries.size())
                        {
                            if (reply_cap <= header_bytes + 8)
                                throw std::invalid_argument("Insufficient exact contact fan page credit.");
                            peer = source;
                            outgoing = writer(Kind::Fan, 0, response_page++);
                            admit_writer(outgoing, reply_cap);
                            outgoing.word(0);
                            std::size_t count = 0;
                            while (query_cursor < queries.size())
                            {
                                const auto& query = queries[query_cursor];
                                const auto fan = query.group ? groups->fan(*query.group) : ContactFanView{};
                                const auto total = fan.local_count();
                                std::size_t need = 32;
                                if (total)
                                {
                                    const auto occurrence = fan.local(occurrence_cursor);
                                    const auto& record = this->record(occurrence.binding.record);
                                    need += 24 + key_wire_size(record.key()) +
                                            PLIC::encoded_identity_size(occurrence.physical());
                                }
                                if (need > reply_cap - outgoing.data.size())
                                {
                                    if (!count)
                                        throw std::invalid_argument("An exact contact fan item exceeds page credit.");
                                    break;
                                }
                                outgoing.word(query.id);
                                outgoing.word(total);
                                outgoing.word(occurrence_cursor);
                                outgoing.word(total ? 1 : 0);
                                if (total)
                                {
                                    const auto occurrence = fan.local(occurrence_cursor);
                                    const auto& record = this->record(occurrence.binding.record);
                                    const auto& physical = occurrence.physical();
                                    const auto key_size = key_wire_size(record.key());
                                    const auto point_size = PLIC::encoded_identity_size(physical);
                                    if (key_size > (budget.limit - budget.used) / 3 ||
                                        point_size > (budget.limit - budget.used - 3 * key_size) / 4)
                                        throw std::invalid_argument("Spatial fan encoding exceeds live byte budget.");
                                    const auto scratch = 3 * key_size + 4 * point_size;
                                    budget.add(scratch);
                                    {
                                        auto key = encode_adaptive_key(record.key());
                                        outgoing.bytes(key);
                                    }
                                    outgoing.word(record.revision());
                                    {
                                        auto point = PLIC::encode_identity(physical);
                                        outgoing.bytes(point);
                                    }
                                    budget.used -= scratch;
                                }
                                ++count;
                                if (!total || ++occurrence_cursor == total)
                                {
                                    occurrence_cursor = 0;
                                    ++query_cursor;
                                }
                            }
                            replace_word(outgoing.data, header_bytes, count);
                            replace_word(outgoing.data, header_bytes - 8, query_cursor == queries.size());
                            statistics.query_bytes += outgoing.data.size();
                        }
                    });
                incoming = exchange(peer, std::move(outgoing));
                collective(*comm,
                    [&]
                    {
                        if (!incoming.packets.empty())
                        {
                            if (received_final || incoming.packets.size() != 1 ||
                                incoming.packets.front().peer != destination)
                                throw std::invalid_argument("Foreign or duplicate exact contact fan page.");
                            Reader r{incoming.packets.front().bytes};
                            const auto frame = read(r, Kind::Fan, destination);
                            if (frame.query || frame.page != expected_response_page++)
                                throw std::invalid_argument("Missing exact contact fan page.");
                            const auto n = r.word();
                            if (!n || n > r.data.size() / 32)
                                throw std::invalid_argument("Invalid exact contact fan page count.");
                            for (std::uint64_t i = 0; i < n; ++i)
                            {
                                const auto id = r.word(), total = r.word(), index = r.word(), present = r.word();
                                if (id < first_query || id - first_query >= pending.size() || present > 1)
                                    throw std::invalid_argument("Foreign exact contact fan query.");
                                auto& request = pending[id - first_query];
                                if (request.done || index != request.next || (!total && present) ||
                                    (total && !present) || (request.started && total != request.total))
                                    throw std::invalid_argument("Duplicate or incomplete exact contact fan.");
                                request.started = true;
                                request.total = total;
                                if (!present)
                                {
                                    request.done = true;
                                    ++statistics.empty_point_replies;
                                    continue;
                                }
                                if (index >= total)
                                    throw std::invalid_argument("Exact contact fan offset exceeds completion count.");
                                const auto encoded_key = r.bytes_view();
                                auto key = decode_bounded_key(encoded_key, budget);
                                const auto revision = r.word();
                                const auto encoded_point = r.bytes_view();
                                auto physical = PLIC::decode_identity<ContactPointIdentity>(
                                    encoded_point, budget.limit - budget.used);
                                const auto record =
                                    add_record(key, revision, destination, groups->domain(request.point));
                                auto& edit = stage_edit(request.point);
                                edit.replace_remote = true;
                                for (const auto& existing : edit.remote)
                                    if (existing.record == record)
                                        throw std::invalid_argument(
                                            "Duplicate cell at an exact normalized contact point.");
                                if (this->record(record).owner == rank)
                                    throw std::invalid_argument("Remote contact fan claims a local cell.");
                                budget.append_capacity(edit.remote);
                                const auto reference = groups->catalog()->points.retain(physical);
                                edit.remote.push_back({record, reference});
                                ++edit_body(record).remote_refs;
                                ++statistics.matched_occurrences;
                                request.done = ++request.next == total;
                                const auto key_bytes = key_capacity(key);
                                key = {};
                                budget.used -= key_bytes;
                            }
                            r.done();
                            received_final = frame.final;
                        }
                    });
                retire(incoming);
            }
            collective(*comm,
                [&]
                {
                    if (!received_final)
                        throw std::invalid_argument("Missing final exact contact fan page.");
                    for (const auto& request : pending)
                        if (!request.done)
                            throw std::invalid_argument("Missing exact contact fan terminal response.");
                });
            const auto scratch = pending.capacity() * sizeof(Pending) + queries.capacity() * sizeof(Query);
            std::vector<Pending>().swap(pending);
            std::vector<Query>().swap(queries);
            budget.used -= scratch;
        }
    }
    collective(*comm, [&] { commit_edits(); });
}

void DistributedAdaptiveContactHalo::Impl::fetch_records()
{
    struct Request
    {
        std::uint64_t query;
        std::size_t record;
    };
    for (int offset = 1; offset < ranks; ++offset)
    {
        const int destination = (rank + offset) % ranks;
        const int source = (rank + ranks - offset) % ranks;
        auto cursor = missing_records.lower_bound({destination, 0});
        const auto end = missing_records.lower_bound({destination + 1, 0});
        std::uint64_t next_query = 0, expected_query = 0, page = 0, expected_page = 0;
        std::set<std::size_t> served;
        while (true)
        {
            if (!any(cursor != end))
                break;
            const auto cap = credit();
            const auto first_query = next_query;
            std::vector<std::size_t> pending;
            Writer outgoing;
            int peer = -1;
            collective(*comm,
                [&]
                {
                    if (cursor != end)
                    {
                        if (cap <= header_bytes + 8)
                            throw std::invalid_argument("Insufficient contact record request page credit.");
                        peer = destination;
                        outgoing = writer(Kind::Records, first_query, page++);
                        admit_writer(outgoing, cap);
                        outgoing.word(0);
                        std::size_t count = 0;
                        while (cursor != end)
                        {
                            const auto id = cursor->second;
                            const auto& record = this->record(id);
                            if (record.owner != destination || body(id).contact)
                                throw std::invalid_argument("Stale missing-record subscription.");
                            const auto key_size = key_wire_size(record.key());
                            const auto need = 24 + key_size;
                            if (need > cap - outgoing.data.size())
                            {
                                if (!count)
                                    throw std::invalid_argument("A contact record request exceeds page credit.");
                                break;
                            }
                            if (key_size > (budget.limit - budget.used) / 3)
                                throw std::invalid_argument("Spatial request key exceeds live byte budget.");
                            budget.add(3 * key_size);
                            outgoing.word(next_query++);
                            {
                                const auto key = encode_adaptive_key(record.key());
                                outgoing.bytes(key);
                            }
                            budget.used -= 3 * key_size;
                            outgoing.word(record.revision());
                            ++statistics.record_fetches;
                            budget.append_capacity(pending);
                            pending.push_back(id);
                            ++cursor;
                            ++count;
                        }
                        replace_word(outgoing.data, header_bytes, count);
                        statistics.query_bytes += outgoing.data.size();
                    }
                });
            auto incoming = exchange(peer, std::move(outgoing));
            std::vector<Request> requests;
            collective(*comm,
                [&]
                {
                    if (!incoming.packets.empty())
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                            throw std::invalid_argument("Foreign contact record request page.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::Records, source);
                        const auto n = r.word();
                        if (frame.query != expected_query || frame.page != expected_page++ || !frame.final || !n ||
                            n > r.data.size() / 24)
                            throw std::invalid_argument("Missing or duplicate contact record request page.");
                        for (std::uint64_t i = 0; i < n; ++i)
                        {
                            const auto query = r.word();
                            if (query != expected_query++)
                                throw std::invalid_argument("Nonsequential contact record request.");
                            const auto encoded_key = r.bytes_view();
                            auto key = decode_bounded_key(encoded_key, budget);
                            const auto revision = r.word();
                            const auto found = this->query->find(key);
                            if (!found || record(*found).owner != rank ||
                                record(*found).revision() != revision || !body(*found).contact ||
                                served.contains(*found))
                                throw std::invalid_argument("Foreign, stale or duplicate contact record request.");
                            budget.add(sizeof(std::size_t) + map_links);
                            served.insert(*found);
                            budget.append_capacity(requests);
                            requests.push_back({query, *found});
                            const auto key_bytes = key_capacity(key);
                            key = {};
                            budget.used -= key_bytes;
                        }
                        r.done();
                    }
                });
            retire(incoming);
            std::size_t request_cursor = 0, encoded_offset = 0, receive_cursor = 0;
            std::uint64_t response_page = 0, expected_response_page = 0;
            AdaptiveBytes encoded, assembly;
            std::size_t assembly_total = 0;
            bool received_final = pending.empty();
            while (any(request_cursor < requests.size()))
            {
                const auto reply_cap = credit();
                outgoing = Writer{};
                peer = -1;
                collective(*comm,
                    [&]
                    {
                        if (request_cursor < requests.size())
                        {
                            if (reply_cap <= header_bytes + 8 + 32)
                                throw std::invalid_argument("Insufficient contact record data page credit.");
                            peer = source;
                            outgoing = writer(Kind::RecordData, 0, response_page++);
                            ++statistics.record_pages;
                            admit_writer(outgoing, reply_cap);
                            outgoing.word(0);
                            std::size_t chunks = 0;
                            while (request_cursor < requests.size() && reply_cap - outgoing.data.size() > 32)
                            {
                                const auto& request = requests[request_cursor];
                                if (encoded.empty())
                                {
                                    const auto& record = this->record(request.record);
                                    encoded = encode_contact(record.key(), record.domain(), *body(request.record).contact, budget);
                                    ++diagnostics.transmitted_geometry_records;
                                    diagnostics.replicated_record_bytes += encoded.size();
                                    statistics.record_bytes += encoded.size();
                                }
                                const auto count =
                                    std::min(encoded.size() - encoded_offset, reply_cap - outgoing.data.size() - 32);
                                outgoing.word(request.query);
                                outgoing.word(encoded.size());
                                outgoing.word(encoded_offset);
                                outgoing.bytes(std::span<const char>(encoded).subspan(encoded_offset, count));
                                ++chunks;
                                encoded_offset += count;
                                if (encoded_offset == encoded.size())
                                {
                                    const auto bytes = encoded.capacity();
                                    AdaptiveBytes().swap(encoded);
                                    budget.used -= bytes;
                                    encoded_offset = 0;
                                    ++request_cursor;
                                }
                            }
                            replace_word(outgoing.data, header_bytes, chunks);
                            replace_word(outgoing.data, header_bytes - 8, request_cursor == requests.size());
                        }
                    });
                incoming = exchange(peer, std::move(outgoing));
                collective(*comm,
                    [&]
                    {
                        if (!incoming.packets.empty())
                        {
                            if (received_final || incoming.packets.size() != 1 ||
                                incoming.packets.front().peer != destination)
                                throw std::invalid_argument("Foreign or duplicate contact record data page.");
                            Reader r{incoming.packets.front().bytes};
                            const auto frame = read(r, Kind::RecordData, destination);
                            if (frame.query || frame.page != expected_response_page++)
                                throw std::invalid_argument("Missing contact record data page.");
                            const auto n = r.word();
                            if (!n || n > r.data.size() / 32)
                                throw std::invalid_argument("Invalid contact record data chunk count.");
                            for (std::uint64_t i = 0; i < n; ++i)
                            {
                                const auto query = r.word(), total = r.word(), offset_ = r.word();
                                const auto bytes = r.bytes_view();
                                if (receive_cursor >= pending.size() || query != first_query + receive_cursor ||
                                    offset_ != assembly.size() || !total || bytes.empty() ||
                                    bytes.size() > total - std::min<std::uint64_t>(total, offset_) ||
                                    (!assembly.empty() && total != assembly_total))
                                    throw std::invalid_argument(
                                        "Duplicate, truncated or foreign contact record chunk.");
                                if (assembly.empty())
                                {
                                    budget.add(total);
                                    assembly.reserve(total);
                                    if (assembly.capacity() != total)
                                        throw std::invalid_argument(
                                            "Contact record assembly capacity differs from admission.");
                                    assembly_total = total;
                                }
                                assembly.insert(assembly.end(), bytes.begin(), bytes.end());
                                if (assembly.size() == assembly_total)
                                {
                                    const auto& record = this->record(pending[receive_cursor]);
                                    if (body(pending[receive_cursor]).contact || record.owner != destination)
                                        throw std::invalid_argument(
                                            "Duplicate or conflicting contact record completion.");
                                    const auto before = budget.used, identities_before = identities.admitted_bytes();
                                    auto decoded = decode_contact(assembly, budget, identities);
                                    if (decoded.key != record.key() || decoded.domain != record.domain())
                                        throw std::invalid_argument(
                                            "Contact record reply key/domain differs from exact match.");
                                    auto body = std::move(static_cast<ContactBody&>(decoded));
                                    const auto wire_only = sizeof(ContactRecord) - sizeof(ContactBody) +
                                                           key_capacity(decoded.key) + decoded.domain.capacity();
                                    decoded = {};
                                    budget.used -= wire_only;
                                    attach_record(pending[receive_cursor], std::move(body),
                                        budget.used - before - (identities.admitted_bytes() - identities_before));
                                    if (!missing_records.erase({destination, pending[receive_cursor]}))
                                        throw std::invalid_argument("Unrequested contact record completion.");
                                    budget.used -= sizeof(decltype(missing_records)::value_type) + map_links;
                                    ++statistics.remote_records;
                                    ++statistics.decoded_record_replies;
                                    ++diagnostics.decoded_geometry_records;
                                    diagnostics.decoded_geometry_bytes += sizeof(ContactBody::spans);
                                    const auto capacity = assembly.capacity();
                                    AdaptiveBytes().swap(assembly);
                                    budget.used -= capacity;
                                    assembly_total = 0;
                                    ++receive_cursor;
                                }
                            }
                            r.done();
                            received_final = frame.final;
                        }
                    });
                retire(incoming);
            }
            collective(*comm,
                [&]
                {
                    if (!received_final || receive_cursor != pending.size() || !assembly.empty())
                        throw std::invalid_argument("Missing contact record terminal completion.");
                });
            const auto scratch = pending.capacity() * sizeof(std::size_t) + requests.capacity() * sizeof(Request);
            std::vector<std::size_t>().swap(pending);
            std::vector<Request>().swap(requests);
            budget.used -= scratch;
        }
        const auto served_bytes = served.size() * (sizeof(std::size_t) + map_links);
        served.clear();
        budget.used -= served_bytes;
    }
}

std::vector<AdaptiveMark> DistributedAdaptiveContactHalo::Impl::evaluate(bool retain_charge)
{
    if (!evaluated)
    {
        collective(*comm,
            [&]
            {
                diagnostic_phase = "pair_gather";
                {
                    auto ordered = ordered_dirty();
                    for (const auto id : ordered.ids())
                    {
                        gather_pairs(id);
                        if (!groups->fan(id).local_count())
                            remove_remote_points(id);
                    }
                    commit_edits();
                    for (const auto id : ordered.ids())
                    {
                        if (query->layout)
                            capture_fan(id);
                        if (!groups->borrowed_fan(id).local_count())
                            groups->retire(id);
                    }
                    const auto bytes = dirty_points.size() * (sizeof(GroupId) + map_links);
                    dirty_points.clear();
                    budget.used -= bytes;
                }
                diagnostic_phase = "pair_sort";
                auto sorted_pairs = affected_pairs.take_sorted();
                statistics.unique_cell_pairs = sorted_pairs.size();
                diagnostic_phase = "pair_evaluate";
                struct PhysicalPair
                {
                    std::size_t local_point, other_occurrence;
                    std::uint8_t flags;
                    enum : std::uint8_t
                    {
                        local_is_first = 1,
                        other_is_remote = 2
                    };
                };
                auto erase_requirement = [&](const auto& ids)
                {
                    if (failing_pairs.erase(ids))
                        budget.used -= sizeof(decltype(failing_pairs)::value_type) + map_links;
                };
                auto charge_work = [&](std::size_t n)
                {
                    if (n > options.contact_budget - statistics.work)
                        throw std::invalid_argument("Spatial closure cumulative contact budget exhausted.");
                    statistics.work += n;
                };
                for (const auto& ids : sorted_pairs)
                {
                    const auto& a = record(ids.first);
                    const auto& b = record(ids.second);
                    const auto& first_body = body(ids.first).contact;
                    const auto& second_body = body(ids.second).contact;
                    if (a.owner < 0 || b.owner < 0 || (a.owner != rank && b.owner != rank))
                    {
                        erase_requirement(ids);
                        continue;
                    }
                    const auto local_id = a.owner == rank ? ids.first : ids.second;
                    const auto other_id = local_id == ids.first ? ids.second : ids.first;
                    const auto& local = record(local_id);
                    if (local.row == no_slot || local.binding.slot == no_slot)
                        throw std::invalid_argument("Missing local spatial incidence binding.");
                    const auto& points = query->rows[local.row].row->points;
                    // Retain occurrence locators, resolving values only at use.
                    // The production query reads the same immutable fan blocks
                    // and exact point namespace that the snapshot will export.
                    std::vector<PhysicalPair> physical;
                    visit_contact_occurrences(local_id, points,
                        [&](std::size_t point, const auto& normalized, auto&& visit)
                        {
                            with_fan(local, point, normalized, [&](const auto& fan) { visit(point, fan); });
                        },
                        [&](std::size_t, std::size_t point, const auto&, std::size_t neighbor,
                            std::size_t occurrence, bool remote, const auto&)
                        {
                            if (neighbor != other_id)
                                return;
                            auto flags = std::uint8_t(local_id == ids.first ? PhysicalPair::local_is_first : 0);
                            if (remote)
                                flags |= PhysicalPair::other_is_remote;
                            budget.append_capacity(physical);
                            physical.push_back({point, occurrence, flags});
                        });
                    std::ranges::sort(physical,
                        [&](const auto& x, const auto& y)
                        {
                            const auto a = points.at(x.local_point).normalized;
                            const auto b = points.at(y.local_point).normalized;
                            return ContactPointWireLess{}({local.domain(), &a}, {local.domain(), &b});
                        });
                    if (physical.empty())
                    {
                        erase_requirement(ids);
                        continue;
                    }
                    if (!first_body || !second_body)
                        throw std::invalid_argument("Incomplete exact-match contact record cache.");
                    statistics.logical_contact_pairs += physical.size();
                    ClosureRequirements required;
                    charge_work(1);
                    ++statistics.span_pair_evaluations;
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        if (a.owner == rank && span_exceeds_ratio(first_body->spans[axis], second_body->spans[axis],
                                                   options.maximum_span_ratio))
                            join_closure_requirement(*first_body, required.first, first_body->factors[axis]);
                        if (b.owner == rank && span_exceeds_ratio(second_body->spans[axis], first_body->spans[axis],
                                                   options.maximum_span_ratio))
                            join_closure_requirement(*second_body, required.second, second_body->factors[axis]);
                    }
                    auto traces = [&](const PhysicalPair& point)
                    {
                        if (!second_body->trace_faces().empty() &&
                            first_body->trace_faces().size() >
                                (options.contact_budget - statistics.work) / second_body->trace_faces().size())
                            throw std::invalid_argument("Spatial closure cumulative trace budget exhausted.");
                        const auto count = first_body->trace_faces().size() * second_body->trace_faces().size();
                        charge_work(count);
                        statistics.trace_pair_evaluations += count;
                        const auto local_point = points.at(point.local_point);
                        const auto& local_physical = local_point.physical;
                        auto other_physical = local_physical;
                        with_fan(local, point.local_point, local_point.normalized,
                            [&](const auto& fan)
                            {
                                other_physical = point.flags & PhysicalPair::other_is_remote
                                                     ? fan.remote(point.other_occurrence).physical
                                                     : fan.local(point.other_occurrence).physical();
                            });
                        const auto& first_physical =
                            point.flags & PhysicalPair::local_is_first ? local_physical : other_physical;
                        const auto& second_physical =
                            point.flags & PhysicalPair::local_is_first ? other_physical : local_physical;
                        for (const auto& first : first_body->trace_faces())
                            for (const auto& second : second_body->trace_faces())
                            {
                                const auto request = trace_pair_requirements(
                                    first, second, first_physical, second_physical, options.maximum_trace_jump);
                                if (a.owner == rank && request.first)
                                    join_closure_requirement(*first_body, required.first, std::uint64_t(*request.first));
                                if (b.owner == rank && request.second)
                                    join_closure_requirement(
                                        *second_body, required.second, std::uint64_t(*request.second));
                            }
                    };
                    if (first_body->traces_point_independent && second_body->traces_point_independent)
                        traces(physical.front());
                    else
                        for (const auto& point : physical)
                            traces(point);
                    const auto scratch = physical.capacity() * sizeof(PhysicalPair);
                    std::vector<PhysicalPair>().swap(physical);
                    budget.used -= scratch;
                    if (required.first || required.second)
                    {
                        if (!failing_pairs.contains(ids))
                            budget.add(sizeof(decltype(failing_pairs)::value_type) + map_links);
                        failing_pairs.insert_or_assign(ids, required);
                    }
                    else
                        erase_requirement(ids);
                }
                sorted_pairs.clear();
                for (const auto id : possibly_unused_records)
                    if (record(id).owner >= 0 && record(id).owner != rank && !body(id).remote_refs)
                        discard_record(id);
                const auto unused_bytes = possibly_unused_records.size() * (sizeof(std::size_t) + map_links);
                possibly_unused_records.clear();
                budget.used -= unused_bytes;
                for (const auto id : retired_records)
                {
                    auto& entry = edit_record(id);
                    auto& cached = edit_body(id);
                    release_borrowed(cached.contact, cached.origin != &budget ? cached.bytes : 0);
                    const auto identity_credit = entry.owned && &entry.owned->admission.arena() != &budget
                                                     ? entry.owned->admission.bytes() : 0;
                    release_borrowed(entry.owned, identity_credit);
                    entry = {};
                    cached = {};
                    budget.append_capacity(free_records);
                    free_records.push_back(id);
                }
                retired_records.clear();
                groups->reclaim([&](std::size_t slot)
                    {
                        if (const auto* point = query->remote_points.find(slot); point && *point)
                            query->remote_points.edit(slot) = {};
                    });
                deferred_owners.reap();
                evaluated = true;
                diagnostic_phase = "ready";
            });
    }
    std::vector<AdaptiveMark> result;
    collective(*comm,
        [&]
        {
            std::map<std::size_t, std::optional<AdaptiveSplit>> required;
            for (const auto& [ids, requests] : failing_pairs)
            {
                auto join = [&](std::size_t id, const auto& request)
                {
                    if (!request)
                        return;
                    if (!required.contains(id))
                        budget.add(sizeof(decltype(required)::value_type) + map_links);
                    join_closure_requirement(*body(id).contact, required[id], std::uint64_t(*request));
                };
                join(ids.first, requests.first);
                join(ids.second, requests.second);
            }
            for (const auto& [id, request] : required)
            {
                budget.append_capacity(result);
                budget.add(key_capacity(record(id).key()));
                result.push_back({record(id).key(), *request});
            }
            const auto bytes = required.size() * (sizeof(decltype(required)::value_type) + map_links);
            required.clear();
            budget.used -= bytes;
            std::ranges::sort(result, {}, &AdaptiveMark::key);
        });
    std::size_t bytes = result.capacity() * sizeof(AdaptiveMark);
    for (const auto& mark : result)
        bytes += key_capacity(mark.key);
    if (!retain_charge)
        budget.used -= bytes;
    return result;
}

DistributedAdaptiveContactHalo::DistributedAdaptiveContactHalo(const AdaptiveProviderKernel& kernel,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveClosureOptions options, ContactHaloContext context,
    ContactHaloBeforeSend before_send)
{
    if (comm.is_null())
        throw std::invalid_argument("Spatial closure needs a communicator.");
    // No rank enters the constructor's first communication until every local
    // owner allocation has succeeded. Impl construction itself cannot communicate.
    collective(*comm,
        [&]
        {
            if (sizeof(Impl) > options.byte_budget)
                throw std::invalid_argument("Spatial closure session exceeds its byte budget.");
            impl_ = std::make_unique<Impl>(kernel, comm, options, context, std::move(before_send));
        });
    impl_->initialize_collective();
}
DistributedAdaptiveContactHalo::~DistributedAdaptiveContactHalo() = default;
DistributedAdaptiveContactHalo::DistributedAdaptiveContactHalo(DistributedAdaptiveContactHalo&&) noexcept = default;
DistributedAdaptiveContactHalo& DistributedAdaptiveContactHalo::operator=(
    DistributedAdaptiveContactHalo&&) noexcept = default;
Budget& DistributedAdaptiveContactHalo::allocation() noexcept
{
    impl_->deferred_owners.reap();
    return impl_->budget;
}
void DistributedAdaptiveContactHalo::visit_closed_contacts(
    const AdaptiveCellKey& key, const ContactWitnessVisitor& visitor) const
{
    const auto& state = *impl_;
    if (!state.ready || !state.evaluated)
        throw std::invalid_argument("Closed-contact query needs a complete evaluated halo.");
    if (state.sealed_contacts)
    {
        state.sealed_contacts->visit_closed_contacts(key, visitor);
        return;
    }
    const auto found = state.query->find(key);
    if (!found || state.record(*found).owner != state.rank)
        throw std::out_of_range("Closed-contact query requires an owned leaf.");
    const auto& local = state.record(*found);
    visit_closed_contact_witnesses(*found, state.query->rows[local.row].row->points,
        [&](std::size_t point, const auto& normalized, auto&& visit)
        {
            state.with_fan(local, point, normalized, [&](const auto& fan) { visit(point, fan); });
        },
        [&](std::size_t neighbor) -> const ContactRecordSlot& { return state.record(neighbor); }, visitor);
}

namespace
{
bool same_leaf_layout(const std::shared_ptr<const AdaptiveLeafLayout>& a,
    const std::shared_ptr<const AdaptiveLeafLayout>& b) noexcept
{
    return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
} // namespace

std::shared_ptr<const AdaptiveContactSnapshot> DistributedAdaptiveContactHalo::seal_contacts(
    std::shared_ptr<const AdaptiveLeafLayout> next, std::span<const AdaptiveLeaf* const> changed_leaf_pages)
{
    if (!changed_leaf_pages.empty() || !same_leaf_layout(impl_->query->layout, next))
        throw std::invalid_argument("A different contact layout requires an authenticated transition.");
    return seal_contacts_impl(next, nullptr);
}

std::shared_ptr<const AdaptiveContactSnapshot> DistributedAdaptiveContactHalo::seal_contacts(
    const AdaptiveLeafTransition& transition)
{
    if (!same_leaf_layout(impl_->query->layout, transition.source()))
        throw std::invalid_argument("Contact transition source disagrees with the pinned selection.");
    return seal_contacts_impl(transition.target(), &transition);
}

std::shared_ptr<const AdaptiveContactSnapshot> DistributedAdaptiveContactHalo::seal_contacts_impl(
    const std::shared_ptr<const AdaptiveLeafLayout>& next, const AdaptiveLeafTransition* transition)
{
    auto& state = *impl_;
    if (!state.ready || !state.evaluated || !state.failing_pairs.empty() || !state.query->layout || !next ||
        next->size() != state.statistics.owned_records)
        throw std::invalid_argument("Contact sealing requires a complete and graded canonical selection.");
    if (state.sealed_contacts)
    {
        if (!same_leaf_layout(state.query->layout, next))
            throw std::invalid_argument("Contact snapshot is already sealed for another selection.");
        return state.sealed_contacts;
    }
    if (state.pending_contact_layout && !same_leaf_layout(state.pending_contact_layout, next))
        throw std::invalid_argument("Contact seal retry supplied a different owned selection.");
    if (transition)
        transition->visit_rebound_leaves(
            [&](const AdaptiveLeaf& leaf)
            {
                if (next->find(leaf.key) != &leaf)
                    throw std::invalid_argument("Contact rebind locator belongs to another leaf selection.");
                const auto found = state.query->find(leaf.key);
                if (!found)
                    throw std::invalid_argument("Contact rebind locator has no exact record.");
                const auto& record = state.record(*found);
                if (record.owner != state.rank || record.revision() != leaf.revision || record.domain() != leaf.domain)
                    throw std::invalid_argument("Contact rebind locator disagrees with an exact record.");
            });
    ContactGroupAllocation charge(state.arena);
    charge.add(sizeof(AdaptiveContactSnapshot) + 128);
    auto snapshot = std::shared_ptr<const AdaptiveContactSnapshot>(
        new AdaptiveContactSnapshot(std::move(charge), state.query, state.context));
    // Keep both leaf roots pinned throughout every edit, including on a
    // failed page admission followed by a retry of this private candidate.
    if (transition)
    {
        state.pending_contact_layout = next;
        transition->visit_rebound_leaves(
            [&](const AdaptiveLeaf& leaf)
            {
                const auto id = *state.query->find(leaf.key);
                if (state.record(id).local_leaf == &leaf)
                    return;
                auto& record = state.edit_record(id);
                if (!record.local_leaf)
                {
                    const auto credit = &record.owned->admission.arena() != &state.budget ? record.owned->admission.bytes() : 0;
                    state.release_borrowed(record.owned, credit);
                    --state.unbound_owned_records;
                }
                record.local_leaf = &leaf;
                ++state.statistics.contact_identity_rebinds;
            });
    }
    if (state.unbound_owned_records)
        throw std::invalid_argument("Complete contact selection has unbound owned records.");
    state.query->layout = next;
    state.pending_contact_layout.reset();
    state.sealed_contacts = std::move(snapshot);
    return state.sealed_contacts;
}

void AdaptiveContactSnapshot::visit_closed_contacts(const AdaptiveCellKey& key, const ContactWitnessVisitor& visitor) const
{
    const auto found = storage_->find(key);
    if (!found || storage_->records[*found].owner != storage_->rank)
        throw std::out_of_range("Closed-contact query requires an owned leaf.");
    const auto& local = storage_->records[*found];
    visit_closed_contact_witnesses(*found, storage_->rows[local.row].row->points,
        [&](std::size_t point, const auto&, auto&& visit) { visit(point, storage_->fan(local.row, point)); },
        [&](std::size_t neighbor) -> const ContactRecordSlot& { return storage_->records[neighbor]; }, visitor);
}

const std::shared_ptr<const AdaptiveSubfaceSnapshot>& DistributedAdaptiveContactHalo::subface_snapshot() const
{
    const auto& state = *impl_;
    if (!state.incidence || !state.ready || !state.evaluated)
        throw std::invalid_argument("Canonical subface snapshot needs a complete evaluated halo.");
    return state.incidence->subface_snapshot();
}

void DistributedAdaptiveContactHalo::for_each_incidence_row(
    const std::function<void(const ContactIncidenceRow&)>& visitor) const
{
    if (impl_->query->layout)
        throw std::invalid_argument("Owning incidence row visitor requires the compatibility route.");
    for (std::size_t i = 0; i < impl_->query->rows.size(); ++i)
        if (const auto& row = impl_->query->rows[i].row)
            visitor(*row);
}
namespace
{
void visit_contact_page(const std::function<void(const ContactPayloadAllocation&)>& visitor,
    const void* page, bool branch, std::size_t bytes)
{
    contact_payload_object(visitor, page, branch ? ContactPayloadRole::ContactSlotBranch :
        ContactPayloadRole::ContactSlotLeaf, bytes - 128, true);
}
void ContactRowGroups::visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(visitor, this, ContactPayloadRole::ContactRowGroupObject, sizeof(*this), true);
    contact_payload_buffer(visitor, this, ContactPayloadRole::ContactRowGroupSlots, compact_);
    contact_payload_buffer(visitor, this, ContactPayloadRole::ContactRowGroupSlots, wide_);
}
} // namespace
void AdaptiveContactSnapshot::Storage::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(visitor, this, ContactPayloadRole::ContactSnapshotStorage, sizeof(*this), true);
    const auto page = [&](const void* value, bool branch, std::size_t bytes)
    { visit_contact_page(visitor, value, branch, bytes); };
    records.visit_pages(page);
    rows.visit_pages(page);
    index.visit_pages(page);
    groups.visit_pages(page);
    bindings.visit_pages(page);
    remote_points.visit_pages(page);
    for (std::size_t i = 0; i < records.size(); ++i)
        if (const auto& identity = records[i].owned)
        {
            contact_payload_object(visitor, identity.get(), ContactPayloadRole::ContactOwnedIdentityObject,
                sizeof(ContactOwnedIdentity), true);
            IdentityPoolBorrowedPayload<AdaptiveCellKey>::visit(identity->key, visitor);
            contact_payload_buffer(visitor, identity.get(), ContactPayloadRole::ContactOwnedIdentityDomain, identity->domain);
        }
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (const auto& row = rows[i].row)
        {
            contact_payload_object(
                visitor, row.get(), ContactPayloadRole::ContactIncidenceRowObject, sizeof(ContactIncidenceRow), true);
            IdentityPoolBorrowedPayload<AdaptiveCellKey>::visit(row->key, visitor);
            contact_payload_buffer(visitor, row.get(), ContactPayloadRole::ContactIncidenceRowDomain, row->domain);
            row->points.visit_borrowed_payload(visitor);
            if (const auto& selectors = rows[i].groups)
                selectors->visit_borrowed_payload(visitor);
        }
    for (std::size_t i = 0; i < groups.size(); ++i)
        if (const auto* group = groups.find(i); group && group->block)
            group->block->visit_borrowed_payload(visitor);
    for (std::size_t i = 0; i < remote_points.size(); ++i)
        if (const auto* point = remote_points.find(i); point && *point)
            point->visit_borrowed_payload(visitor);
}
void AdaptiveContactSnapshot::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(visitor, this, ContactPayloadRole::ContactSnapshotObject, sizeof(*this), true);
    storage_->visit_borrowed_payload(visitor);
}
void DistributedAdaptiveContactHalo::for_each_borrowed_contact_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    impl_->groups->visit_borrowed_payload(visitor, true);
    if (impl_->incidence)
        impl_->incidence->for_each_borrowed_payload(visitor, true);
    if (impl_->sealed_contacts)
        impl_->sealed_contacts->visit_borrowed_payload(visitor);
    else
        impl_->query->visit_borrowed_payload(visitor);
    impl_->work->visit_pages([&](const void* page, bool branch, std::size_t bytes)
        { visit_contact_page(visitor, page, branch, bytes); });
    if (impl_->index)
        impl_->index->visit_borrowed_pages([&](const void* page, bool branch, std::size_t bytes)
            { visit_contact_page(visitor, page, branch, bytes); });
    impl_->deferred_owners.visit_borrowed_payload(visitor);
    contact_payload_buffer(visitor, impl_.get(), ContactPayloadRole::HaloGroupEditPositions, impl_->pending_positions);
    contact_payload_buffer(visitor, impl_.get(), ContactPayloadRole::HaloFanEdits, impl_->pending_edits);
    for (const auto& edit : impl_->pending_edits)
    {
        contact_payload_buffer(visitor, &edit, ContactPayloadRole::HaloRemovedBindings, edit.removed);
        contact_payload_buffer(visitor, &edit, ContactPayloadRole::HaloAppendedLocal, edit.appended);
        contact_payload_buffer(visitor, &edit, ContactPayloadRole::HaloAppendedRemote, edit.remote);
    }
}
void DistributedAdaptiveContactHalo::for_each_borrowed_geometry_ledger(
    const std::function<void(const PLIC::GeometryAllocationLedger&)>& visitor) const
{
    visitor(impl_->identities.borrowed_geometry_ledger());
}
void DistributedAdaptiveContactHalo::for_each_borrowed_plane_identity(
    const std::function<void(const PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>&)>& visitor) const
{
    if (impl_->incidence)
        impl_->incidence->for_each_borrowed_plane_identity(visitor);
}
void DistributedAdaptiveContactHalo::for_each_borrowed_contact_face(
    const std::function<void(const PLIC::PlanarFace<ArithmeticMode::Ordinary>&)>& visitor) const
{
    for (std::size_t i = 0; i < impl_->work->size(); ++i)
        if (const auto& contact = impl_->body(i).contact)
            for (const auto& trace : contact->trace_faces())
                if (trace.face)
                    visitor(*trace.face);
}
void DistributedAdaptiveContactHalo::for_each_borrowed_source_admission(
    const std::function<void(const void*, std::size_t)>& visitor) const
{
    impl_->deferred_owners.visit_live_owners(visitor);
    if (impl_->incidence)
        impl_->incidence->for_each_borrowed_source_admission(visitor);
}
void DistributedAdaptiveContactHalo::for_each_borrowed_overlap_credit(
    const std::function<void(const void*, std::size_t, bool)>& visitor) const
{
    if (impl_->incidence)
        impl_->incidence->for_each_borrowed_overlap_credit(visitor);
}
std::size_t DistributedAdaptiveContactHalo::source_arena_overlap_metadata_bytes() const noexcept
{
    return impl_->incidence ? impl_->incidence->source_arena_overlap_metadata_bytes() : 0;
}
std::size_t DistributedAdaptiveContactHalo::current_fork_overlap_bytes() const noexcept
{
    return impl_->fork_overlap_bytes + impl_->query->overlap_bytes() + impl_->work->overlap_bytes() +
           (impl_->index ? impl_->index->overlap_bytes() : 0);
}
void DistributedAdaptiveContactHalo::for_each_borrowed_contact_group(
    const std::function<void(const ContactHaloBorrowedGroup&)>& visitor) const
{
    impl_->groups->visit_borrowed_groups(
        [&](const auto& group)
        {
            visitor({group.record, group.id.slot, group.id.generation, group.domain, group.normalized,
                group.normalized_reference.slot, group.normalized_reference.generation, group.fan, group.cover,
                group.live, group.retired});
        });
}
void DistributedAdaptiveContactHalo::for_each_borrowed_contact_fact(
    const std::function<void(const ContactHaloBorrowedFact&)>& visitor) const
{
    impl_->groups->borrowed_catalog().points.visit_borrowed_facts(
        [&](const auto& fact)
        {
            visitor({fact.record, fact.reference.slot, fact.reference.generation, fact.pins, fact.retired,
                fact.physical_page, fact.captured.value, fact.value, fact.source_snapshot, fact.source_captured.value});
        });
}
void DistributedAdaptiveContactHalo::for_each_borrowed_contact_binding(
    const std::function<void(const ContactHaloBorrowedBinding&)>& visitor) const
{
    impl_->groups->borrowed_catalog().visit_borrowed_bindings(
        [&](const auto& binding)
        {
            visitor({binding.record, binding.id.slot, binding.id.generation, binding.pins, binding.retired, binding.row,
                binding.incidence_generation, binding.record_id});
        });
}
ContactMarkLease::ContactMarkLease(std::vector<AdaptiveMark> marks, Budget& budget, std::size_t charged) noexcept
    : marks_(std::move(marks)), budget_(&budget), charged_(charged)
{
}
ContactMarkLease::~ContactMarkLease()
{
    reset();
}
ContactMarkLease::ContactMarkLease(ContactMarkLease&& other) noexcept
    : marks_(std::move(other.marks_)), budget_(std::exchange(other.budget_, nullptr)),
      charged_(std::exchange(other.charged_, 0))
{
}
ContactMarkLease& ContactMarkLease::operator=(ContactMarkLease&& other) noexcept
{
    if (this != &other)
    {
        reset();
        marks_ = std::move(other.marks_);
        budget_ = std::exchange(other.budget_, nullptr);
        charged_ = std::exchange(other.charged_, 0);
    }
    return *this;
}
void ContactMarkLease::reset() noexcept
{
    std::vector<AdaptiveMark>().swap(marks_);
    if (budget_)
        budget_->used -= charged_;
    budget_ = nullptr;
    charged_ = 0;
}

void DistributedAdaptiveContactHalo::bootstrap(
    std::span<const AdaptiveLeaf> owned, std::vector<ContactIncidenceRow> input)
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (state.ready || owned.size() != input.size())
                throw std::invalid_argument("Invalid or repeated spatial contact bootstrap.");
            state.budget.add(input.capacity() * sizeof(ContactIncidenceRow));
            std::map<AdaptiveCellKey, const AdaptiveLeaf*> source;
            std::size_t source_bytes = 0;
            for (const auto& leaf : owned)
            {
                const auto bytes = sizeof(decltype(source)::value_type) + map_links + key_capacity(leaf.key);
                state.budget.add(bytes);
                source_bytes += bytes;
                if (!source.emplace(leaf.key, &leaf).second)
                    throw std::invalid_argument("Duplicate spatial source key.");
            }
            std::shared_ptr<ContactRowGroups> selectors;
            for (auto& row : input)
            {
                const auto found = source.find(row.key);
                if (found == source.end())
                    throw std::invalid_argument("Missing spatial source row.");
                const auto bytes = row_storage(row);
                state.budget.add(bytes);
                auto keeper = state.arena;
                auto handle = std::shared_ptr<const ContactIncidenceRow>(new ContactIncidenceRow(std::move(row)),
                    [keeper, bytes](const ContactIncidenceRow* value)
                    {
                        delete value;
                        keeper->used -= bytes;
                    });
                state.install_row(std::move(handle), *found->second, selectors);
            }
            source.clear();
            state.budget.used -= source_bytes;
            const auto slots = input.capacity() * sizeof(ContactIncidenceRow);
            std::vector<ContactIncidenceRow>().swap(input);
            state.budget.used -= slots;
            state.statistics.bootstrap_owned_visits = owned.size();
            state.diagnostics.cells = owned.size();
        });
    collective(*state.comm,
        [&]
        {
            state.commit_edits();
            state.refresh_covers();
        });
    state.diagnostic_phase = "directory";
    state.build_directory();
    state.discover_neighbors();
    state.diagnostic_phase = "point_fans";
    state.discover_points();
    state.diagnostic_phase = "record_fetch";
    state.fetch_records();
    state.ready = true;
    state.diagnostic_phase = "ready";
}

void DistributedAdaptiveContactHalo::bootstrap_canonical(
    std::span<const AdaptiveLeaf> owned, const ContactOwnedLeafLookup& lookup,
    std::shared_ptr<const AdaptiveLeafLayout> layout)
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (state.ready || state.incidence)
                throw std::invalid_argument("Repeated canonical contact bootstrap.");
            if (layout && layout->size() != owned.size())
                throw std::invalid_argument("Canonical contact source extent disagrees with the owned selection.");
            state.query->layout = std::move(layout);
            state.incidence = std::make_unique<DistributedAdaptiveIncidenceSession>(
                state.kernel, state.comm, state.options, state.context, state.budget, state.identities, state.arena);
        });
    state.diagnostic_phase = "canonical_incidence";
    auto rows = state.incidence->bootstrap(owned, lookup, !state.query->layout);
    state.diagnostic_phase = "local_rows";
    collective(*state.comm,
        [&]
        {
            std::shared_ptr<ContactRowGroups> selectors;
            for (auto& delivery : rows)
                state.install_row(std::move(delivery.row), *delivery.leaf, selectors);
            if (state.unbound_owned_records)
                throw std::invalid_argument("Canonical contact bootstrap disagrees with the owned selection.");
            const auto bytes = rows.capacity() * sizeof(IncidenceRowDelivery);
            std::vector<IncidenceRowDelivery>().swap(rows);
            state.budget.used -= bytes;
            state.statistics.bootstrap_owned_visits = owned.size();
            state.diagnostics.cells = owned.size();
        });
    collective(*state.comm,
        [&]
        {
            state.commit_edits();
            state.refresh_covers();
        });
    state.diagnostic_phase = "directory";
    state.build_directory();
    state.discover_neighbors();
    state.diagnostic_phase = "point_fans";
    state.discover_points();
    state.diagnostic_phase = "record_fetch";
    state.fetch_records();
    state.ready = true;
    state.diagnostic_phase = "ready";
}

void DistributedAdaptiveContactHalo::apply_delta(std::span<const ContactRetiredLeaf> retired,
    std::span<const AdaptiveCellKey> inserted, ContactHaloContext next, const ContactOwnedLeafLookup& lookup)
{
    auto& state = *impl_;
    state.deferred_owners.reap();
    std::vector<IncidenceCellVersion> versions;
    collective(*state.comm,
        [&]
        {
            if (!state.ready || !state.incidence || !state.evaluated || state.sealed_contacts)
                throw std::invalid_argument("Incremental contact update needs a completed canonical session.");
            if (next == state.context)
                throw std::invalid_argument("Incremental contact context did not advance.");
            state.budget.add(retired.size() * sizeof(IncidenceCellVersion));
            versions.reserve(retired.size());
            for (const auto& item : retired)
            {
                const auto found = state.query->find(item.key);
                if (!found)
                    throw std::invalid_argument("Retired contact cell is absent.");
                const auto& record = state.record(*found);
                if (record.owner != state.rank || record.revision() != item.revision || record.row >= state.query->rows.size())
                    throw std::invalid_argument("Retired contact owner or revision is stale.");
                state.budget.add(key_capacity(item.key));
                versions.push_back({item.key, item.revision, state.query->rows[record.row].row->incidence_generation});
            }
        });
    auto delta = state.incidence->apply_delta(versions, inserted, next, lookup, !state.query->layout);
    collective(*state.comm,
        [&]
        {
            std::size_t bytes = versions.capacity() * sizeof(IncidenceCellVersion);
            for (const auto& item : versions)
                bytes += key_capacity(item.key);
            std::vector<IncidenceCellVersion>().swap(versions);
            state.budget.used -= bytes;
            state.context = next;
            state.evaluated = false;
            state.statistics.work = 0;
            for (const auto& item : delta.retired)
            {
                const auto found = state.query->find(item.key);
                if (!found)
                    throw std::invalid_argument("Incidence retired an unknown cell.");
                const auto id = *found;
                const auto& record = state.record(id);
                if (record.owner != state.rank || record.revision() != item.leaf_revision ||
                    record.row >= state.query->rows.size() ||
                    state.query->rows[record.row].row->incidence_generation != item.incidence_generation)
                    throw std::invalid_argument("Incidence retirement version is stale.");
                state.retire_row(id);
            }
            for (const auto& item : retired)
            {
                const auto found = state.query->find(item.key);
                if (!found)
                    throw std::invalid_argument("Missing retired geometry record.");
                if (state.record(*found).row < state.query->rows.size())
                    throw std::invalid_argument("Retired leaf still has a live incidence row.");
                state.discard_record(*found);
                --state.statistics.owned_records;
            }
            std::shared_ptr<ContactRowGroups> selectors;
            for (auto& delivery : delta.changed)
            {
                auto& row = delivery.row;
                const auto& leaf = *delivery.leaf;
                if (const auto found = state.query->find(leaf.key); found)
                {
                    const auto& record = state.record(*found);
                    if (record.owner == state.rank && record.row < state.query->rows.size())
                    {
                        if (record.revision() != row->revision ||
                            state.query->rows[record.row].row->incidence_generation >= row->incidence_generation)
                            throw std::invalid_argument("Changed incidence row did not advance its generation.");
                        state.retire_row(*found);
                    }
                }
                state.install_row(std::move(row), leaf, selectors);
            }
            bytes = delta.retired.capacity() * sizeof(IncidenceCellVersion) +
                    delta.changed.capacity() * sizeof(IncidenceRowDelivery);
            for (const auto& item : delta.retired)
                bytes += key_capacity(item.key);
            delta = {};
            state.budget.used -= bytes;
            state.statistics.delta_owned_visits += inserted.size();
            state.previous_neighbors.swap(state.neighbors);
            const auto directory_bytes =
                state.directory.size() * (sizeof(decltype(state.directory)::value_type) + map_links);
            state.directory.clear();
            state.budget.used -= directory_bytes;
        });
    state.initialize_collective();
    collective(*state.comm,
        [&]
        {
            state.commit_edits();
            state.refresh_covers();
        });
    state.build_directory();
    state.discover_neighbors();
    state.notify_dirty_points();
    collective(*state.comm,
        [&]
        {
            for (const auto id : state.dirty_points)
                state.remove_remote_points(id);
            state.commit_edits();
            std::size_t bytes =
                state.previous_neighbors.size() * (sizeof(decltype(state.previous_neighbors)::value_type) + map_links);
            for (const auto& [owner, covers] : state.previous_neighbors)
            {
                bytes += covers.capacity() * sizeof(ClosedContactCover);
                for (const auto& cover : covers)
                    bytes += cover.domain.capacity();
            }
            state.previous_neighbors.clear();
            state.budget.used -= bytes;
        });
    state.discover_points();
    state.fetch_records();
}

std::unique_ptr<DistributedAdaptiveContactHalo> DistributedAdaptiveContactHalo::fork_candidate(
    ContactHaloContext context, std::size_t caller_live_bytes) const
{
    impl_->deferred_owners.reap();
    const auto& source = *impl_;
    std::unique_ptr<DistributedAdaptiveContactHalo> candidate;
    // Allocate the outer owner on every rank before its communicating ctor.
    collective(*source.comm,
        [&]
        {
            if (!source.ready || !source.evaluated || !source.incidence)
                throw std::invalid_argument("Contact candidate needs an evaluated canonical source.");
            const auto metadata = source.metadata_bytes();
            constexpr auto header = sizeof(Impl) + sizeof(Budget) + sizeof(std::size_t) + 128 +
                                    sizeof(DistributedAdaptiveContactHalo) + sizeof(Impl::Groups) +
                                    Impl::Groups::empty_dynamic_admission_bytes();
            if (caller_live_bytes > source.options.byte_budget ||
                metadata > source.options.byte_budget - caller_live_bytes ||
                header > source.options.byte_budget - caller_live_bytes - metadata)
                throw std::invalid_argument("Contact candidate metadata overlap exceeds byte budget.");
        });
    void* storage = nullptr;
    collective(*source.comm, [&] { storage = ::operator new(sizeof(DistributedAdaptiveContactHalo)); });
    try
    {
        candidate.reset(new (storage) DistributedAdaptiveContactHalo(
            source.kernel, source.comm, source.options, context, source.before_send));
    }
    catch (...)
    {
        ::operator delete(storage);
        throw;
    }
    auto& state = *candidate->impl_;
    collective(*source.comm,
        [&]
        {
            // The empty constructor catalog is only an initialization owner.
            // Retire it before admitting source/candidate overlap so empty
            // ranks never pay for two empty catalogs simultaneously.
            state.groups.reset();
            state.query.reset();
            state.budget.used -= sizeof(Impl::Groups);
            state.budget.add(caller_live_bytes);
            state.external_retained_bytes = caller_live_bytes;
            state.fork_overlap_bytes = source.metadata_bytes();
            state.budget.add(state.fork_overlap_bytes);
        });
    // Every rank finishes the incidence fork before selective group imports can
    // reuse its independently admitted physical-page credits.
    collective(*source.comm,
        [&]
        {
            state.incidence =
                source.incidence->fork(state.budget, state.identities, state.arena, &state.fork_overlap_bytes);
        });
    collective(*source.comm,
        [&]
        {
            state.trace_pool.emplace(*source.trace_pool, state.budget);
            ContactGroupAllocation charge(state.arena);
            charge.add(sizeof(AdaptiveContactSnapshot::Storage) + 128);
            state.query = std::make_shared<AdaptiveContactSnapshot::Storage>(std::move(charge), *source.query);
            state.work.emplace(*source.work, state.arena);
            state.unbound_owned_records = source.unbound_owned_records;
            auto copy_vector = [&](auto& target, const auto& from)
            {
                using Value = typename std::decay_t<decltype(target)>::value_type;
                state.budget.add(from.size() * sizeof(Value));
                target.reserve(from.size());
                target.assign(from.begin(), from.end());
            };
            const auto borrow_owner = [&](const auto& owner, std::size_t bytes, bool from_source_overlap = false)
            {
                if (state.deferred_owners.contains(owner) ||
                    state.incidence->admits_source_metadata_owner(owner, bytes))
                    return;
                if (from_source_overlap)
                {
                    if (bytes > state.fork_overlap_bytes)
                        throw std::invalid_argument("Point-page credit exceeds source overlap admission.");
                }
                else
                    state.budget.add(bytes);
                try
                {
                    if (!state.deferred_owners.adopt(owner, bytes))
                        throw std::invalid_argument("Duplicate borrowed point-page admission.");
                }
                catch (...)
                {
                    if (!from_source_overlap)
                        state.budget.used -= bytes;
                    throw;
                }
                if (from_source_overlap)
                    state.fork_overlap_bytes -= bytes;
            };
            for (std::size_t i = 0; i < source.query->rows.size(); ++i)
                if (const auto& row = source.query->rows[i].row)
                {
                    state.budget.add(row_storage(*row));
                    if (const auto& selectors = source.query->rows[i].groups)
                        borrow_owner(selectors, selectors->bytes());
                    if (const auto view = row->points.page_view();
                        view && !state.deferred_owners.contains(view))
                    {
                        // The forked incidence pool independently admits page
                        // bodies. This credit covers the shared immutable
                        // directory once, independently of per-row buffers.
                        borrow_owner(view, view->metadata_bytes());
                        view->visit_credit_owners(
                            [&](const auto& owner, std::size_t bytes)
                            {
                                // Transfer current-source credit objects out
                                // of source-only overlap. Ancestor credits are
                                // distinct owners even when they pin one page.
                                const bool current_source = view->origin_arena() == &source.budget &&
                                                            source.incidence->owns_point_page_credit(owner.get());
                                borrow_owner(owner, bytes, current_source);
                            });
                    }
                    if (const auto view = row->points.source_page_view();
                        view && !state.deferred_owners.contains(view))
                    {
                        // Source bodies are admitted by the candidate point
                        // directory; independently retain the exact immutable
                        // source-view and ancestor credit metadata of this row.
                        borrow_owner(view, view->metadata_bytes());
                        view->visit_credit_owners(
                            [&](const auto& owner, std::size_t bytes)
                            {
                                const bool current_source = view->origin_arena() == &source.budget &&
                                                            source.incidence->owns_point_page_credit(owner.get());
                                borrow_owner(owner, bytes, current_source);
                            });
                    }
                }
            copy_vector(state.free_rows, source.free_rows);
            copy_vector(state.free_records, source.free_records);
            for (std::size_t id = 0; id < source.query->records.size(); ++id)
            {
                const auto& record = source.record(id);
                const auto& body = source.body(id);
                state.budget.add(body.bytes + (record.owned ? record.owned->admission.bytes() : 0));
                if (body.contact)
                    for (const auto& trace : body.contact->trace_faces())
                        if (trace.face)
                            state.identities.retain(
                                state.budget, PLIC::planar_support<ArithmeticMode::Ordinary>(*trace.face));
            }
            state.budget.add(sizeof(Impl::Groups));
            auto copied_groups =
                std::make_unique<Impl::Groups>(*source.groups, state.arena, &state.incidence->point_pool());
            state.groups = std::move(copied_groups);
            copy_vector(state.pending_positions, source.pending_positions);
            for (std::size_t id = 0; id < source.query->records.size(); ++id)
            {
                const auto binding = source.record(id).binding;
                if (binding.slot != no_slot)
                    state.groups->catalog()->pin(binding);
            }
            for (const auto& [ids, request] : source.failing_pairs)
            {
                state.budget.add(sizeof(decltype(state.failing_pairs)::value_type) + map_links);
                state.failing_pairs.emplace(ids, request);
            }
            if (source.index && state.query->layout)
                state.index = std::make_unique<ContactCoverIndex>(*source.index, state.arena, state.cover_resolver());
            else if (source.index)
            {
                state.index = std::make_unique<ContactCoverIndex>(*source.index, state.budget);
                auto ordered = state.groups->ordered_ids();
                for (const auto id : ordered.ids())
                {
                    const auto cover = state.groups->cover(id);
                    if (cover == std::numeric_limits<std::size_t>::max())
                        throw std::invalid_argument("Missing copied point cover leaf.");
                    state.index->rebind(cover, state.groups->point_directory(), state.groups->normalized_reference(id),
                        &state.groups->domain(id));
                }
            }
            state.budget.add(source.covers.size() * sizeof(ClosedContactCover));
            state.covers.reserve(source.covers.size());
            for (const auto& cover : source.covers)
            {
                state.budget.add(cover.domain.size());
                state.covers.push_back(cover);
            }
            for (const auto& [owner, covers] : source.neighbors)
            {
                state.budget.add(sizeof(decltype(state.neighbors)::value_type) + map_links +
                                 covers.size() * sizeof(ClosedContactCover));
                for (const auto& cover : covers)
                    state.budget.add(cover.domain.size());
                state.neighbors.emplace(owner, covers);
            }
            for (const auto& [id, node] : source.directory)
            {
                state.budget.add(sizeof(decltype(state.directory)::value_type) + map_links);
                state.directory.emplace(id, node);
            }
            state.local_cover = source.local_cover;
            state.statistics.owned_records = source.statistics.owned_records;
            state.statistics.remote_records = source.statistics.remote_records;
            state.diagnostics.cells = source.diagnostics.cells;
            state.ready = state.evaluated = true;
        });
    return candidate;
}
std::size_t DistributedAdaptiveContactHalo::cache_storage_bytes() const noexcept
{
    return impl_->budget.used - impl_->external_retained_bytes - current_fork_overlap_bytes() -
           *impl_->external_lease_bytes;
}
ContactExternalResidencyLease DistributedAdaptiveContactHalo::external_residency_lease() const
{
    return ContactExternalResidencyLease(impl_->arena, impl_->external_lease_bytes);
}
ContactExternalResidencyLease::ContactExternalResidencyLease(
    std::shared_ptr<Budget> arena, std::shared_ptr<std::size_t> total)
    : arena_(std::move(arena)), total_(std::move(total))
{
}
ContactExternalResidencyLease::~ContactExternalResidencyLease()
{
    if (arena_)
    {
        arena_->used -= bytes_;
        *total_ -= bytes_;
    }
}
ContactExternalResidencyLease::ContactExternalResidencyLease(ContactExternalResidencyLease&& other) noexcept
    : arena_(std::move(other.arena_)), total_(std::move(other.total_)), bytes_(std::exchange(other.bytes_, 0))
{
}
ContactExternalResidencyLease& ContactExternalResidencyLease::operator=(ContactExternalResidencyLease&& other) noexcept
{
    if (this != &other)
    {
        if (arena_)
        {
            arena_->used -= bytes_;
            *total_ -= bytes_;
        }
        arena_ = std::move(other.arena_);
        total_ = std::move(other.total_);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}
void ContactExternalResidencyLease::adopt_precharged(std::size_t bytes)
{
    if (!arena_ || bytes_ || bytes > arena_->used)
        throw std::logic_error("Invalid pre-admitted external residency transfer.");
    *total_ += bytes;
    bytes_ = bytes;
}
void ContactExternalResidencyLease::resize(std::size_t bytes)
{
    if (!arena_)
        throw std::logic_error("External residency lease lacks its allocation arena.");
    if (bytes > bytes_)
    {
        arena_->add(bytes - bytes_);
        *total_ += bytes - bytes_;
    }
    else
    {
        arena_->used -= bytes_ - bytes;
        *total_ -= bytes_ - bytes;
    }
    bytes_ = bytes;
}
void DistributedAdaptiveContactHalo::bind_external_retained_bytes(std::size_t bytes) noexcept
{
    impl_->budget.used -= impl_->fork_overlap_bytes;
    impl_->fork_overlap_bytes = 0;
    impl_->query->finish_overlap();
    impl_->work->finish_overlap();
    if (impl_->incidence)
        impl_->incidence->finish_inherited_overlap();
    if (impl_->index)
    {
        impl_->index->finish_overlap();
        if (!impl_->index->point_count())
            impl_->index.reset();
    }
    impl_->external_retained_bytes = bytes;
}

std::vector<AdaptiveMark> DistributedAdaptiveContactHalo::required_refinements()
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (!state.ready)
                throw std::invalid_argument("Spatial closure session has no complete bootstrap.");
        });
    return state.evaluate();
}
ContactMarkLease DistributedAdaptiveContactHalo::lease_required_refinements()
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (!state.ready)
                throw std::invalid_argument("Spatial closure session has no complete bootstrap.");
        });
    auto marks = state.evaluate(true);
    std::size_t bytes = marks.capacity() * sizeof(AdaptiveMark);
    for (const auto& mark : marks)
        bytes += key_capacity(mark.key);
    return ContactMarkLease(std::move(marks), state.budget, bytes);
}
const ContactHaloStatistics& DistributedAdaptiveContactHalo::statistics() const noexcept
{
    if (impl_->incidence)
    {
        const auto& incidence = impl_->incidence->statistics();
        impl_->statistics.bootstrap_owned_visits = incidence.bootstrap_owned_visits;
        impl_->statistics.delta_owned_visits = incidence.delta_owned_visits;
        impl_->statistics.full_owned_scans_after_bootstrap = incidence.full_owned_scans_after_bootstrap;
        impl_->statistics.dirty_supports = incidence.dirty_supports;
        impl_->statistics.dirty_support_members = incidence.dirty_support_members;
        impl_->statistics.dirty_lines = incidence.dirty_lines;
        impl_->statistics.dirty_subscriptions = incidence.dirty_subscriptions;
        impl_->statistics.recomputed_tiles = incidence.recomputed_tiles;
        impl_->statistics.source_owner_manifests = incidence.source_owner_manifests;
        impl_->statistics.source_membership_cells = incidence.source_membership_cells;
        impl_->statistics.source_coordinate_points = incidence.source_coordinate_points;
        impl_->statistics.explicit_coordinate_points = incidence.explicit_coordinate_points;
        impl_->statistics.source_tile_records = incidence.source_tile_records;
        impl_->statistics.explicit_tile_ring_slots = incidence.explicit_tile_ring_slots;
        impl_->statistics.source_line_records = incidence.source_line_records;
        impl_->statistics.source_line_nonforward_records = incidence.source_line_nonforward_records;
        impl_->statistics.source_line_families = incidence.source_line_families;
        impl_->statistics.source_axis_maps = incidence.source_axis_maps;
        impl_->statistics.source_result_selectors = incidence.source_result_selectors;
        impl_->statistics.explicit_line_records = incidence.explicit_line_records;
        impl_->statistics.stride_line_records = incidence.stride_line_records;
        impl_->statistics.retained_explicit_line_records = incidence.retained_explicit_line_records;
        impl_->statistics.retained_stride_line_records = incidence.retained_stride_line_records;
        impl_->statistics.retained_source_line_records = incidence.retained_source_line_records;
        impl_->statistics.source_point_recaptures = incidence.source_point_recaptures;
        impl_->statistics.source_point_recaptured_records = incidence.source_point_recaptured_records;
        impl_->statistics.source_point_recapture_admitted_bytes = incidence.source_point_recapture_admitted_bytes;
        impl_->statistics.active_saved_layers = incidence.active_saved_layers;
        impl_->statistics.retained_saved_layers = incidence.retained_saved_layers;
        impl_->statistics.retained_explicit_layer_input_slots = incidence.retained_explicit_layer_input_slots;
        impl_->statistics.retained_explicit_layer_output_slots = incidence.retained_explicit_layer_output_slots;
        impl_->statistics.retained_explicit_layer_edge_result_slots =
            incidence.retained_explicit_layer_edge_result_slots;
        impl_->statistics.retained_certified_layer_input_slots = incidence.retained_certified_layer_input_slots;
        impl_->statistics.retained_certified_layer_output_slots = incidence.retained_certified_layer_output_slots;
        impl_->statistics.retained_certified_layer_edge_result_slots =
            incidence.retained_certified_layer_edge_result_slots;
        impl_->statistics.retained_certified_layer_descriptor_bytes =
            incidence.retained_certified_layer_descriptor_bytes;
        impl_->statistics.retained_layer_endpoint_identity_slots = incidence.retained_layer_endpoint_identity_slots;
        impl_->statistics.cached_completion_requests = incidence.cached_completion_requests;
        impl_->statistics.cached_completion_remote_requests = incidence.cached_completion_remote_requests;
        impl_->statistics.cached_completion_received = incidence.cached_completion_received;
        impl_->statistics.retained_completion_owners = incidence.retained_completion_owners;
        impl_->statistics.retained_expanded_completion_result_slots =
            incidence.retained_expanded_completion_result_slots;
    }
    const auto storage = impl_->groups->storage_breakdown();
    std::size_t reused_metadata = 0, reused_body = 0;
    if (impl_->incidence)
        impl_->groups->visit_admissions(
            [&](const ContactStorageAdmission& admission)
            {
                if (admission.body_owner &&
                    impl_->incidence->point_pool().owns_page_admission(admission.body_owner, admission.metadata_owner))
                {
                    reused_metadata += admission.private_metadata;
                    reused_body += admission.shared_body;
                }
            });
    impl_->statistics.point_groups = impl_->groups->size();
    impl_->statistics.group_private_metadata_bytes = sizeof(Impl::Groups) + storage.private_metadata_bytes() -
                                                     reused_metadata +
                                                     impl_->pending_positions.capacity() * sizeof(std::size_t) +
                                                     impl_->pending_edits.capacity() * sizeof(Impl::PendingFanEdit);
    for (const auto& edit : impl_->pending_edits)
        impl_->statistics.group_private_metadata_bytes += edit.removed.capacity() * sizeof(ContactBindingId) +
                                                          edit.appended.capacity() * sizeof(ContactLocalOccurrence) +
                                                          edit.remote.capacity() * sizeof(ContactRemoteOccurrence);
    impl_->statistics.group_shared_body_bytes = storage.shared_body_bytes() - reused_body;
    impl_->statistics.reused_point_credit_metadata_bytes = reused_metadata;
    impl_->statistics.reused_point_body_bytes = reused_body;
    const auto& query = *impl_->query;
    impl_->statistics.contact_record_pages_copied = query.records.copied_pages();
    impl_->statistics.contact_row_pages_copied = query.rows.copied_pages();
    impl_->statistics.contact_work_pages_copied = impl_->work->copied_pages();
    impl_->statistics.contact_index_pages_copied = query.index.copied_pages();
    impl_->statistics.contact_index_split_records = query.index.split_records();
    impl_->statistics.contact_cover_pages_copied = impl_->index ? impl_->index->copied_pages() : 0;
    impl_->statistics.contact_cover_page_bytes = impl_->index
        ? impl_->index->storage_bytes() - impl_->index->private_metadata_bytes() : 0;
    impl_->statistics.contact_query_pages_copied =
        query.groups.copied_pages() + query.bindings.copied_pages() + query.remote_points.copied_pages();
    impl_->statistics.retained_bytes = impl_->budget.used;
    impl_->statistics.peak_bytes = impl_->budget.peak;
    impl_->statistics.cumulative_bytes = impl_->budget.cumulative;
    return impl_->statistics;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
