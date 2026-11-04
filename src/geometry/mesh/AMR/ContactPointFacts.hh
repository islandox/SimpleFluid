#pragma once
#include "ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/ContactGroupAllocation.hh"
#include "geometry/mesh/AMR/ContactPointArray.hh"
#include "geometry/mesh/AMR/ContactPointWire.hh"
#include "geometry/mesh/AMR/ContactStorageBreakdown.hh"
#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <map>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactPointRef
{
    std::size_t slot;
    std::uint64_t generation;
    bool operator==(const ContactPointRef&) const = default;
};
/** Owns only referenced immutable point pages. Refcounts include staging and
 * immutable fan owners. release() retires; reclaim() is the evaluation barrier. */
class PointFactDirectory
{
public:
    using PointPool = IdentityPool<ContactPointIdentity>;
    explicit PointFactDirectory(std::shared_ptr<Budget>);
    PointFactDirectory(const PointFactDirectory&, std::shared_ptr<Budget>, const PointPool* target_pool = nullptr,
        bool preserve_pins = true);
    ~PointFactDirectory();
    PointFactDirectory(const PointFactDirectory&) = delete;
    PointFactDirectory& operator=(const PointFactDirectory&) = delete;
    [[nodiscard]] ContactPointRef retain(ContactPointArray::Locator, const PointPool* target_pool = nullptr);
    [[nodiscard]] ContactPointRef retain(const ContactPointIdentity&);
    void pin(ContactPointRef);
    void release(ContactPointRef);
    [[nodiscard]] const ContactPointIdentity& get(ContactPointRef) const;
    /** Synchronous readers may resolve a slot only while their immutable fan
     * owns a pin. Escaped snapshots retain this entire directory/catalog. */
    [[nodiscard]] ContactPointRef pinned_reference(std::size_t slot) const;
    [[nodiscard]] ContactPointRef staging_reference(std::size_t slot) const;
    void reclaim();
    void seal_fallback() noexcept;
    [[nodiscard]] std::size_t fallback_pages() const noexcept;
    [[nodiscard]] std::size_t live_points() const noexcept;
    [[nodiscard]] std::size_t live_pages() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return allocation_.bytes(); }
    [[nodiscard]] const void* credit_identity(ContactPointRef) const;
    [[nodiscard]] std::size_t inherited_credit_overhead_bytes() const noexcept { return 0; }
    [[nodiscard]] const void* metadata_identity() const noexcept { return allocation_.identity(); }
    void contribute_breakdown(ContactStorageBreakdown&) const;
    template<class Visitor> void visit_admissions(Visitor&& visitor) const
    {
        visitor(ContactStorageAdmission{allocation_.identity(), allocation_.bytes(), nullptr, 0});
        for (const auto& page : pages_)
        {
            if (page.view)
            {
                visitor(ContactStorageAdmission{page.view->metadata_identity(), page.view->metadata_bytes(), nullptr, 0});
                page.view->visit_page_admissions([&](const void* credit, const void* body,
                    std::size_t metadata, std::size_t bytes)
                    { visitor(ContactStorageAdmission{credit, metadata, body, bytes}); });
            }
            else if (page.fallback)
                visitor(ContactStorageAdmission{page.fallback->metadata.identity(), page.fallback->metadata.bytes(),
                    page.fallback->body.get(), page.fallback->body_admission.bytes(), page.fallback->body->sealed});
        }
    }
    struct BorrowedFact
    {
        const void* record;
        ContactPointRef reference;
        std::size_t pins;
        bool retired;
        const void* physical_page;
        CapturedPointReference captured;
        const ContactPointIdentity* value;
    };
    template<class Visitor> void visit_borrowed_facts(Visitor&& visitor) const
    {
        for (std::size_t i = 0; i < entries_.size(); ++i)
        {
            const auto& entry = entries_[i];
            if (entry.refs || entry.retired)
                visitor(BorrowedFact{&entry, {i, entry.generation}, entry.refs, entry.retired,
                    pages_[entry.page].identity, entry.reference, &pages_[entry.page].get(entry.reference)});
        }
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visitor, this, ContactPayloadRole::FactDirectoryObject, sizeof(*this));
        contact_payload_buffer(visitor, this, ContactPayloadRole::FactPages, pages_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::FactEntries, entries_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::FactFreeEntries, free_entries_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::FactFreePages, free_pages_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::FactBuckets, buckets_);
        for (const auto& value : page_index_)
            contact_payload_node_value(visitor, this, ContactPayloadRole::FactPageIndexValues, value);
        for (const auto& page : pages_)
            if (page.view) page.view->visit_borrowed_payload(visitor);
            else if (page.fallback)
            {
                contact_payload_object(visitor, page.fallback.get(), ContactPayloadRole::FallbackCreditObject,
                    sizeof(FallbackCredit), true);
                contact_payload_object(visitor, page.fallback->body.get(), ContactPayloadRole::FallbackBodyObject,
                    sizeof(FallbackBody), true);
                contact_payload_buffer(visitor, page.fallback->body.get(), ContactPayloadRole::FallbackValues,
                    page.fallback->body->values);
            }
    }

private:
    static constexpr std::size_t fallback_records_per_page = 16;
    struct FallbackBody
    {
        std::vector<ContactPointIdentity> values;
        std::size_t size = 0;
        bool sealed = false;
        FallbackBody()
        {
            values.reserve(fallback_records_per_page);
            if (values.capacity() != fallback_records_per_page)
                throw std::invalid_argument("Fallback point capacity differs from admission.");
        }
        FallbackBody(const FallbackBody& source) : FallbackBody()
        {
            values.assign(source.values.begin(), source.values.end());
            size = source.size;
            sealed = source.sealed;
        }
    };
    struct FallbackCredit
    {
        explicit FallbackCredit(std::shared_ptr<Budget> arena) : metadata(arena), body_admission(std::move(arena)) {}
        ContactGroupAllocation metadata, body_admission;
        std::shared_ptr<FallbackBody> body;
    };
    struct Page
    {
        std::shared_ptr<const ContactPointPageView> view;
        std::shared_ptr<FallbackCredit> fallback;
        const void* identity = nullptr;
        std::size_t live = 0;
        [[nodiscard]] bool present() const noexcept { return bool(view) || bool(fallback); }
        [[nodiscard]] const ContactPointIdentity& get(CapturedPointReference reference) const;
    };
    struct Entry
    {
        std::uint64_t generation = 1;
        std::size_t refs = 0, page = 0;
        CapturedPointReference reference{};
        std::uint64_t hash = 0;
        bool retired = false;
    };
    Entry& checked(ContactPointRef);
    const Entry& checked(ContactPointRef) const;
    ContactPointRef retain_page(
        std::shared_ptr<const ContactPointPageView>, CapturedPointReference, std::uint64_t hash);
    std::shared_ptr<FallbackCredit> make_fallback(const FallbackCredit* source = nullptr);
    std::size_t add_page(Page);
    ContactPointRef add_entry(std::size_t page, CapturedPointReference, std::uint64_t hash);
    std::optional<ContactPointRef> find(const ContactPointIdentity&, std::uint64_t hash) const;
    void ensure_lookup();
    void insert_bucket(std::size_t);
    static constexpr auto empty = std::numeric_limits<std::size_t>::max();
    static constexpr auto tombstone = empty - 1;
    static constexpr auto page_node_bytes = sizeof(std::pair<const void* const, std::size_t>) + 4 * sizeof(void*);
    ContactGroupAllocation allocation_;
    std::vector<Page> pages_;
    std::vector<Entry> entries_;
    std::vector<std::size_t> free_entries_;
    std::vector<std::size_t> free_pages_;
    std::vector<std::size_t> buckets_;
    std::map<const void*, std::size_t> page_index_;
    std::size_t retained_ = 0;
    std::size_t fallback_tail_ = empty;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
