#pragma once
#include "IncidencePointDirectory.hh"
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** References belong to this NEW composite namespace. A stored suffix is
 * encoded relative to this view's immutable source prefix; the original
 * physical CapturedReference is recovered unchanged before physical reads. */
struct CapturedSourcePointReference
{
    std::uint64_t value = 0;
    bool operator==(const CapturedSourcePointReference&) const = default;
};
class ContactSourcePointView
{
public:
    using Pool = IncidencePointDirectory<>;
    using StoredPool = Pool::Store;
    using PhysicalView = StoredPool::PageView;
    using PhysicalReference = StoredPool::CapturedReference;
    struct StoredLocation
    {
        const PhysicalView* view;
        PhysicalReference reference;
    };
    static std::shared_ptr<ContactSourcePointView> create(std::shared_ptr<const ContactSourcePointSnapshot> source,
        Budget& budget, std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
    {
        cartesian_detail::check_keeper(budget, keeper);
        if (!source || !arena || arena->budget != &budget)
            throw std::invalid_argument("Unbound composite source point view.");
        cartesian_detail::Allocation memory(budget);
        memory.retain(sizeof(ContactSourcePointView) + 128);
        cartesian_detail::CreditConstruction credits(arena);
        auto result = std::shared_ptr<ContactSourcePointView>(new ContactSourcePointView);
        result->source_ = std::move(source);
        result->budget_ = &budget;
        result->keeper_ = std::move(keeper);
        result->arena_ = std::move(arena);
        credits.retain(result->source_, result->source_->allocation_budget(), result->source_->metadata_bytes(),
            result->source_->has_arena_keeper(), result->leases_, memory);
        result->source_->visit_dependencies(
            [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
            {
                using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
                const std::shared_ptr<const std::remove_const_t<Element>> fixed = owner;
                credits.retain(fixed, origin, bytes, keeper, result->leases_, memory);
            });
        result->charge_ = std::exchange(memory.bytes, 0);
        credits.committed = true;
        return result;
    }
    ~ContactSourcePointView()
    {
        stored_.reset();
        source_.reset();
        decltype(leases_){}.swap(leases_);
        if (budget_)
            budget_->used -= charge_;
    }
    [[nodiscard]] CapturedSourcePointReference capture(Pool& pool, std::size_t logical_slot)
    {
        if (frozen_ || &pool.allocation() != budget_ || pool.source() != source_)
            throw std::invalid_argument("Invalid composite point capture binding.");
        (void) pool.get(logical_slot);
        if (logical_slot < source_->size())
            return {logical_slot};
        ensure_stored();
        return stored_reference(stored_->capture(pool.physical_pool(), logical_slot - source_->size()));
    }
    [[nodiscard]] CapturedSourcePointReference source_reference(ContactSourcePointReference reference) const
    {
        if (reference.value >= source_->size())
            throw std::out_of_range("Invalid source point location.");
        return {reference.value};
    }
    [[nodiscard]] CapturedSourcePointReference import_reference(
        const ContactSourcePointView& source, CapturedSourcePointReference reference, const Pool* target = nullptr)
    {
        if (frozen_ || !source.frozen_ || source.source_ != source_ || (target && &target->allocation() != budget_))
            throw std::invalid_argument("Invalid immutable composite point import.");
        (void) source.value(reference);
        if (reference.value < source_->size())
            return reference;
        ensure_stored();
        const auto old = source.stored_location(reference);
        return stored_reference(
            stored_->import_reference(*old->view, old->reference, target ? &target->physical_pool() : nullptr));
    }
    void freeze()
    {
        if (stored_)
            stored_->freeze();
        frozen_ = true;
        arena_.reset();
    }
    [[nodiscard]] bool frozen() const noexcept { return frozen_; }
    [[nodiscard]] ContactSourcePoint value(CapturedSourcePointReference reference) const
    {
        if (!frozen_)
            throw std::invalid_argument("Composite source point view is not frozen.");
        if (reference.value < source_->size())
            return source_->value({reference.value});
        if (!stored_)
            throw std::out_of_range("Composite point has no stored suffix.");
        return stored_->get({reference.value - source_->size()});
    }
    [[nodiscard]] ContactSourcePoint get(CapturedSourcePointReference reference) const { return value(reference); }
    [[nodiscard]] std::optional<ContactSourcePointReference> source_location(
        CapturedSourcePointReference reference) const
    {
        if (reference.value < source_->size())
            return ContactSourcePointReference{reference.value};
        return {};
    }
    [[nodiscard]] std::optional<StoredLocation> stored_location(CapturedSourcePointReference reference) const
    {
        if (reference.value < source_->size())
            return {};
        if (!stored_)
            throw std::out_of_range("Composite point has no stored location.");
        const PhysicalReference stored{reference.value - source_->size()};
        (void) stored_->get(stored);
        return StoredLocation{stored_.get(), stored};
    }
    [[nodiscard]] const std::shared_ptr<const ContactSourcePointSnapshot>& source() const noexcept { return source_; }
    [[nodiscard]] const std::shared_ptr<CartesianAuthorityArena>& arena() const noexcept { return arena_; }
    [[nodiscard]] const Budget* origin_arena() const noexcept { return budget_; }
    [[nodiscard]] const void* metadata_identity() const noexcept { return &charge_; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_; }
    [[nodiscard]] std::size_t page_count() const noexcept { return stored_ ? stored_->page_count() : 0; }
    template<class Visitor> void visit_page_admissions(Visitor&& visitor) const
    {
        if (stored_)
            stored_->visit_page_admissions(visitor);
    }
    template<class Visitor> void visit_source_admissions(Visitor&& visitor) const
    {
        if (source_->allocation_budget() == budget_)
            visitor(source_->admission_identity(), source_.get(), std::size_t(0), source_->metadata_bytes());
        else
            for (const auto& lease : leases_)
                if (lease->owner.lock().get() == source_.get())
                    visitor(lease.get(), source_.get(), lease->metadata, source_->metadata_bytes());
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ContactPayloadRole::SourcePointViewObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointLeases, leases_);
        source_->visit_borrowed_payload(visitor);
        for (const auto& lease : leases_)
            lease->visit_borrowed_payload(visitor);
        if (stored_)
            stored_->visit_borrowed_payload(visitor);
    }
    template<class Visitor> void visit_credit_owners(Visitor&& visitor) const
    {
        for (const auto& lease : leases_)
            visitor(lease, lease->metadata);
        if (stored_)
            stored_->visit_credit_owners(visitor);
    }

private:
    ContactSourcePointView() = default;
    void ensure_stored()
    {
        if (stored_)
            return;
        StoredPool factory(*budget_, keeper_);
        stored_ = factory.page_view();
    }
    CapturedSourcePointReference stored_reference(PhysicalReference reference) const
    {
        if (reference.value > std::numeric_limits<std::uint64_t>::max() - source_->size())
            throw std::invalid_argument("Composite point reference overflows.");
        return {reference.value + source_->size()};
    }
    std::shared_ptr<const ContactSourcePointSnapshot> source_;
    std::shared_ptr<PhysicalView> stored_;
    std::shared_ptr<CartesianAuthorityArena> arena_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
    std::size_t charge_ = 0;
    bool frozen_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
