#pragma once
#include "ContactPayloadCensus.hh"
#include "ContactPointFans.hh"
#include "ContactPointWire.hh"
#include "PhysicalOwnerIndex.hh"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactGroupId
{
    std::size_t slot;
    std::uint64_t generation;
    bool operator==(const ContactGroupId&) const = default;
};
/** A stable full-value typed lookup, with no resident serialized point keys.
 * The hash policy is a compile-time parameter; hashes never decide equality. */
template<class Hasher = ContactPointWireHasher> class ContactPointGroups
{
    struct Group
    {
        std::uint64_t generation = 1;
        std::size_t domain = 0;
        ContactPointRef normalized{};
        ContactFanView fan;
        std::size_t cover = std::numeric_limits<std::size_t>::max();
        bool live = false, retired = false;
    };
    struct Readers
    {
        std::size_t active = 0;
        std::byte admission_identity{};
    };

public:
    struct Edit
    {
        ContactGroupId group;
        std::span<const ContactBindingId> remove_bindings;
        std::span<const ContactLocalOccurrence> append_local;
        std::span<const ContactRemoteOccurrence> remote;
        bool replace_remote = true;
    };
    /** IDs are ordered by the exact legacy wire comparator. While this lease
     * exists, retired groups cannot be reclaimed or rebound to another key. */
    class OrderedIds
    {
    public:
        ~OrderedIds()
        {
            if (readers_)
                --readers_->active;
        }
        OrderedIds(OrderedIds&&) = default;
        OrderedIds(const OrderedIds&) = delete;
        OrderedIds& operator=(const OrderedIds&) = delete;
        std::span<const ContactGroupId> ids() const noexcept { return ids_; }

    private:
        friend class ContactPointGroups;
        OrderedIds(std::shared_ptr<Budget> arena, std::shared_ptr<Readers> readers)
            : allocation_(std::move(arena)), readers_(std::move(readers))
        {
            ++readers_->active;
        }
        ContactGroupAllocation allocation_;
        std::vector<ContactGroupId> ids_;
        std::shared_ptr<Readers> readers_;
    };
    explicit ContactPointGroups(std::shared_ptr<Budget>, Hasher = {});
    /** Dynamic owners constructed before any group or fan is inserted. */
    [[nodiscard]] static constexpr std::size_t empty_dynamic_admission_bytes() noexcept
    {
        return sizeof(ContactFanCatalog) + 128 + sizeof(Readers) + 128;
    }
    ContactPointGroups(
        const ContactPointGroups&, std::shared_ptr<Budget>, const PointFactDirectory::PointPool* target_pool = nullptr);
    ~ContactPointGroups();
    ContactPointGroups(const ContactPointGroups&) = delete;
    ContactPointGroups& operator=(const ContactPointGroups&) = delete;
    [[nodiscard]] ContactGroupId intern(const AdaptiveBytes&, const ContactPointIdentity&,
        std::optional<ContactPointArray::Locator> = {}, const PointFactDirectory::PointPool* target_pool = nullptr);
    [[nodiscard]] std::optional<ContactGroupId> find(ContactPointKeyView) const;
    /** Any bounded byte stream is a valid probe. Malformed framing is simply
     * an unmatched key; this method never decodes untrusted point bytes. */
    [[nodiscard]] std::optional<ContactGroupId> find_encoded(std::span<const char>) const;
    [[nodiscard]] ContactPointKeyView key(ContactGroupId) const;
    [[nodiscard]] const AdaptiveBytes& domain(ContactGroupId) const;
    [[nodiscard]] ContactPointIdentity normalized(ContactGroupId) const;
    [[nodiscard]] ContactPointRef normalized_reference(ContactGroupId) const;
    [[nodiscard]] const PointFactDirectory& point_directory() const noexcept { return catalog_->points; }
    [[nodiscard]] ContactFanView fan(ContactGroupId) const;
    [[nodiscard]] std::size_t cover(ContactGroupId) const;
    void set_cover(ContactGroupId, std::size_t);
    void replace_fans(std::span<const Edit>);
    template<class EditAt> void replace_fans_generated(std::size_t, EditAt&&);
    void retire(ContactGroupId);
    /** Call only after old-pair obligations and dirty/query/evaluation readers
     * have completed. Owning fan snapshots continue pinning their catalogs. */
    void reclaim();
    [[nodiscard]] OrderedIds ordered_ids(bool include_retired = false) const;
    [[nodiscard]] OrderedIds ordered_ids(std::span<const ContactGroupId> selected) const;
    [[nodiscard]] std::size_t size() const noexcept { return live_; }
    [[nodiscard]] std::size_t slot_count() const noexcept { return groups_.size(); }
    [[nodiscard]] std::shared_ptr<ContactFanCatalog> catalog() const noexcept { return catalog_; }
    [[nodiscard]] const ContactFanCatalog& borrowed_catalog() const noexcept { return *catalog_; }
    [[nodiscard]] ContactStorageBreakdown storage_breakdown() const;
    template<class Visitor> void visit_admissions(Visitor&&) const;
    template<class Visitor> void visit_borrowed_payload(Visitor&&, bool include_object = false) const;
    struct BorrowedGroup
    {
        const void* record;
        ContactGroupId id;
        const AdaptiveBytes* domain;
        ContactPointRef normalized_reference;
        ContactPointIdentity normalized;
        const ContactFanView* fan;
        std::size_t cover;
        bool live, retired;
    };
    template<class Visitor> void visit_borrowed_groups(Visitor&& visitor) const
    {
        for (std::size_t i = 0; i < groups_.size(); ++i)
        {
            const auto& group = groups_[i];
            if (group.live || group.retired)
                visitor(BorrowedGroup{&group, {i, group.generation}, &domains_.get(group.domain), group.normalized,
                    catalog_->points.get(group.normalized), &group.fan, group.cover, group.live, group.retired});
        }
    }
    [[nodiscard]] std::size_t last_duplicate_check_work() const noexcept { return duplicate_work_; }
    [[nodiscard]] std::size_t fan_owner_lookup_work() const noexcept { return fan_owners_.probes() + fork_owner_work_; }

private:
    static constexpr auto empty = std::numeric_limits<std::size_t>::max();
    static constexpr auto tombstone = empty - 1;
    const Group& checked(ContactGroupId, bool allow_retired = false) const;
    Group& checked(ContactGroupId, bool allow_retired = false);
    ContactPointKeyView key(const Group&) const;
    void ensure_lookup();
    static std::shared_ptr<Readers> make_readers(const std::shared_ptr<Budget>&);
    template<class Probe> std::optional<ContactGroupId> find_probe(const Probe&) const;
    void insert_bucket(std::size_t);
    ContactGroupAllocation allocation_;
    IdentityPool<AdaptiveBytes> domains_;
    std::shared_ptr<ContactFanCatalog> catalog_;
    std::shared_ptr<Readers> readers_;
    [[no_unique_address]] Hasher hasher_;
    std::vector<Group> groups_;
    std::vector<std::size_t> free_, buckets_;
    PhysicalOwnerIndex<std::shared_ptr<const ContactFanOwner>> fan_owners_;
    std::size_t live_ = 0;
    std::size_t duplicate_work_ = 0, fork_owner_work_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
#include "ContactPointGroups.ipp"
