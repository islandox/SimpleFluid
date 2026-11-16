#pragma once
#include "ContactPayloadCensus.hh"
#include "ContactPointFacts.hh"
#include "ContactStorageBreakdown.hh"
#include "DistributedAdaptiveContactHalo.hh"
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactBindingId
{
    std::size_t slot;
    std::uint64_t generation;
    bool operator==(const ContactBindingId&) const = default;
};
struct ContactLocalOccurrence
{
    ContactBindingId binding;
    std::size_t point;
};
struct ContactRemoteOccurrence
{
    std::size_t record;
    ContactPointRef physical;
};
/** Exact row binding. Its generation is never narrowed into an occurrence. */
struct ContactRowBinding
{
    std::shared_ptr<const ContactIncidenceRow> row;
    std::uint64_t incidence_generation = 0;
    std::size_t record = 0;
};
class ContactFanCatalog
{
public:
    [[nodiscard]] static std::shared_ptr<ContactFanCatalog> create(std::shared_ptr<Budget>);
    [[nodiscard]] static std::shared_ptr<ContactFanCatalog> create(
        const ContactFanCatalog&, std::shared_ptr<Budget>, const PointFactDirectory::PointPool*);
    explicit ContactFanCatalog(std::shared_ptr<Budget>);
    ContactFanCatalog(const ContactFanCatalog&, std::shared_ptr<Budget>, const PointFactDirectory::PointPool*);
    [[nodiscard]] ContactBindingId bind(std::shared_ptr<const ContactIncidenceRow>, std::size_t record);
    void pin(ContactBindingId);
    void release(ContactBindingId);
    [[nodiscard]] const ContactRowBinding& binding(ContactBindingId) const;
    [[nodiscard]] ContactBindingId pinned_binding(std::size_t) const;
    [[nodiscard]] ContactBindingId staging_binding(std::size_t) const;
    void reclaim(const std::function<void(std::size_t)>& retire_fact = {});
    PointFactDirectory points;
    [[nodiscard]] const std::shared_ptr<Budget>& arena() const noexcept { return allocation_.keeper(); }
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept { return allocation_.bytes(); }
    [[nodiscard]] const void* metadata_identity() const noexcept { return allocation_.identity(); }
    struct BorrowedBinding
    {
        const void* record;
        ContactBindingId id;
        std::size_t pins;
        bool retired;
        const ContactIncidenceRow* row;
        std::uint64_t incidence_generation;
        std::size_t record_id;
    };
    template<class Visitor> void visit_borrowed_bindings(Visitor&& visitor) const
    {
        for (std::size_t i = 0; i < bindings_.size(); ++i)
        {
            const auto& entry = bindings_[i];
            if (entry.refs || entry.retired)
                visitor(BorrowedBinding{&entry, {i, entry.generation}, entry.refs, entry.retired, entry.value.row.get(),
                    entry.value.incidence_generation, entry.value.record});
        }
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = true) const
    {
        if (include_object)
            contact_payload_object(
                visitor, this, ParallelSupportPayloadRole::BindingCatalogObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::BindingEntries, bindings_);
        contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::BindingFreeSlots, free_);
        contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::BindingBuckets, buckets_);
        points.visit_borrowed_payload(visitor);
    }

private:
    struct Binding
    {
        ContactRowBinding value;
        std::uint64_t generation = 1;
        std::size_t refs = 0;
        bool retired = false;
    };
    Binding& checked(ContactBindingId);
    const Binding& checked(ContactBindingId) const;
    static std::uint64_t binding_hash(const ContactIncidenceRow*, std::uint64_t, std::size_t) noexcept;
    void ensure_binding_lookup();
    static constexpr auto empty_binding = std::numeric_limits<std::size_t>::max();
    static constexpr auto retired_bucket = empty_binding - 1;
    ContactGroupAllocation allocation_;
    ContactVector<Binding> bindings_;
    ContactVector<std::size_t> free_;
    ContactVector<std::size_t> buckets_;
    std::size_t retained_ = 0;
};
/** Immutable, multi-group flat arrays. Each compact occurrence is eight bytes;
 * offsets and slots widen together when any admitted value needs 64 bits. */
class ContactFanBlock
{
public:
    template<class UInt> struct Local
    {
        UInt binding, point;
    };
    template<class UInt> struct Remote
    {
        UInt record, physical;
    };
    template<class UInt> struct Slice
    {
        UInt local, locals, remote, remotes;
    };
    struct Input
    {
        std::span<const ContactLocalOccurrence> local;
        std::span<const ContactRemoteOccurrence> remote;
    };
    [[nodiscard]] static std::shared_ptr<const ContactFanBlock> build(
        std::span<const Input>, ContactGroupAllocation&, bool detached_charge = false);
    [[nodiscard]] bool wide() const noexcept;
    [[nodiscard]] std::size_t groups() const noexcept;
    [[nodiscard]] std::size_t local_count(std::size_t group) const;
    [[nodiscard]] std::size_t remote_count(std::size_t group) const;
    [[nodiscard]] std::pair<std::size_t, std::size_t> local(std::size_t group, std::size_t occurrence) const;
    [[nodiscard]] std::pair<std::size_t, std::size_t> remote(std::size_t group, std::size_t occurrence) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] const Budget* origin() const noexcept { return &admission_.arena(); }
    [[nodiscard]] static bool needs_wide(std::uint64_t value) noexcept;
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ParallelSupportPayloadRole::FanBlockObject, sizeof(*this), true);
        std::visit(
            [&](const auto& arrays)
            {
                contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::FanLocal, arrays.local);
                contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::FanRemote, arrays.remote);
                contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::FanSlices, arrays.slices);
            },
            data_);
    }
    template<class Visitor> void visit_borrowed_arrays(Visitor&& visitor) const
    {
        std::visit(
            [&](const auto& arrays)
            {
                visitor(std::span(arrays.local), arrays.local.capacity(), std::span(arrays.remote),
                    arrays.remote.capacity(), std::span(arrays.slices), arrays.slices.capacity());
            },
            data_);
    }

private:
    template<class UInt> struct Arrays
    {
        ContactVector<Local<UInt>> local;
        ContactVector<Remote<UInt>> remote;
        ContactVector<Slice<UInt>> slices;
    };
    explicit ContactFanBlock(std::shared_ptr<Budget> arena) : admission_(std::move(arena)) {}
    ContactGroupAllocation admission_;
    std::variant<Arrays<std::uint32_t>, Arrays<std::uint64_t>> data_;
};
/** One arena's block credit and all occurrence pins. An escaped owner retains
 * its own catalog, so it cannot resolve a binding in a later catalog. */
class ContactFanOwner
{
public:
    [[nodiscard]] static std::shared_ptr<const ContactFanOwner> build(
        std::shared_ptr<ContactFanCatalog>, std::span<const ContactFanBlock::Input>);
    [[nodiscard]] static std::shared_ptr<const ContactFanOwner> fork(
        const ContactFanOwner&, std::shared_ptr<ContactFanCatalog>);
    ~ContactFanOwner();
    [[nodiscard]] const ContactFanBlock& block() const noexcept { return *block_; }
    [[nodiscard]] const std::shared_ptr<const ContactFanBlock>& frozen_block() const noexcept { return block_; }
    [[nodiscard]] const ContactFanCatalog* catalog_identity() const noexcept { return catalog_.get(); }
    /** End the construction/fork barrier after every selected slice has a
     * view. Unselected slices become permanently unavailable in this owner. */
    void finish_staging() const;
    [[nodiscard]] bool has_live_slices() const noexcept;
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept { return allocation_.bytes(); }
    [[nodiscard]] const void* metadata_identity() const noexcept { return allocation_.identity(); }
    [[nodiscard]] std::size_t shared_body_bytes() const noexcept { return block_ ? block_->storage_bytes() : 0; }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ParallelSupportPayloadRole::FanOwnerObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ParallelSupportPayloadRole::FanViews, views_);
        if (block_)
            block_->visit_borrowed_payload(visitor);
    }
    template<class Visitor> void visit_borrowed_slices(Visitor&& visitor) const
    {
        for (std::size_t i = 0; i < views_.size(); ++i)
            visitor(i, views_[i], block_.get());
    }

private:
    friend struct ContactFanView;
    explicit ContactFanOwner(std::shared_ptr<ContactFanCatalog>);
    [[nodiscard]] static std::shared_ptr<ContactFanOwner> create(std::shared_ptr<ContactFanCatalog>);
    void pin_slice(std::size_t) const;
    void unpin_slice(std::size_t) const;
    void release_slice(std::size_t) const;
    [[nodiscard]] const ContactRowBinding& binding(std::size_t slot) const;
    [[nodiscard]] ContactPointIdentity physical(std::size_t slot) const;
    ContactGroupAllocation allocation_;
    ContactGroupAllocation body_admission_;
    std::shared_ptr<ContactFanCatalog> catalog_;
    std::shared_ptr<const ContactFanBlock> block_;
    mutable ContactVector<std::size_t> views_;
    mutable bool staging_ = true;
};
struct ContactFanView
{
private:
    std::shared_ptr<const ContactFanOwner> owner_;
    std::size_t group_ = 0;

public:
    [[nodiscard]] const std::shared_ptr<const ContactFanOwner>& owner() const noexcept { return owner_; }
    [[nodiscard]] std::size_t group() const noexcept { return group_; }
    ContactFanView() = default;
    ContactFanView(std::shared_ptr<const ContactFanOwner>, std::size_t);
    ContactFanView(const ContactFanView&);
    ContactFanView(ContactFanView&&) noexcept;
    ContactFanView& operator=(ContactFanView) noexcept;
    ~ContactFanView();
    [[nodiscard]] std::size_t local_count() const { return owner_ ? owner_->block().local_count(group_) : 0; }
    [[nodiscard]] std::size_t remote_count() const { return owner_ ? owner_->block().remote_count(group_) : 0; }
    struct Local
    {
        ContactBindingId id;
        const ContactRowBinding& binding;
        std::size_t point;
        [[nodiscard]] std::size_t record() const noexcept { return binding.record; }
        [[nodiscard]] ContactPointIdentity physical() const { return binding.row->points.at(point).physical; }
    };
    struct Remote
    {
        std::size_t record;
        ContactPointIdentity physical;
    };
    [[nodiscard]] Local local(std::size_t i) const;
    [[nodiscard]] Remote remote(std::size_t i) const;
    [[nodiscard]] std::size_t remote_record(std::size_t i) const { return owner_->block().remote(group_, i).first; }
};
static_assert(sizeof(ContactFanBlock::Local<std::uint32_t>) == 8);
static_assert(sizeof(ContactFanBlock::Remote<std::uint32_t>) == 8);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
