/** @file PersistentContactFacts.hh
 * @brief Sparse copy-on-write exact point references for live contact fans.
 */
#pragma once
#include "IncidencePhysicalOwners.hh"
#include "ContactPointFacts.hh"
#include "ContactSlotIndex.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Production point catalogue. Only live reference counters and numeric
 * lookups change. Imported physical/source pages are frozen and shared. */
class PersistentContactFacts
{
    static constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();
    using PageBody = PointFactDirectory::Page;
    struct Page
    {
        PageBody body;
        std::size_t next_free = absent;
    };
    struct Fact
    {
        std::uint64_t generation = 1;
        std::size_t pins = 0, remotes = 0, page = absent, next_free = absent;
        PageBody::Reference reference{};
        std::uint64_t digest = 0;
        bool retired = false;
    };

public:
    using PointPool = PointFactDirectory::PointPool;
    using FrozenPoint = PointFactDirectory::FrozenPoint;
    using BorrowedFact = PointFactDirectory::BorrowedFact;
    PersistentContactFacts(std::shared_ptr<Budget> arena, ContactPhysicalResidency& residency)
        : residency_(residency), scratch_(std::move(arena)), pages_(scratch_.keeper()), facts_(scratch_.keeper()),
          page_ids_(scratch_.keeper()), fact_ids_(scratch_.keeper()) {}
    PersistentContactFacts(const PersistentContactFacts& source, std::shared_ptr<Budget> arena,
        ContactPhysicalResidency& residency)
        : residency_(residency), scratch_(std::move(arena)), pages_(source.pages_, scratch_.keeper()),
          facts_(source.facts_, scratch_.keeper()),
          page_ids_(source.page_ids_, scratch_.keeper()), fact_ids_(source.fact_ids_, scratch_.keeper()),
          free_page_(source.free_page_), free_fact_(source.free_fact_)
    {
        if (source.tail_ != absent || !source.retired_.empty())
            throw std::invalid_argument("Persistent contact facts need a sealed source.");
    }

    [[nodiscard]] ContactPointRef retain(ContactPointArray::Locator, const PointPool*);
    [[nodiscard]] ContactPointRef retain(const ContactPointIdentity&);
    void pin(ContactPointRef reference)
    {
        const auto& before = checked(reference);
        if (before.pins == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Contact point reference count overflows.");
        auto& fact = facts_.edit(reference.slot);
        ++fact.pins;
        fact.retired = false;
    }
    void release(ContactPointRef reference)
    {
        const auto& before = checked(reference);
        if (!before.pins)
            throw std::invalid_argument("Contact point reference already retired.");
        if (before.pins == 1)
        {
            scratch_.append(retired_);
            retired_.push_back(reference);
        }
        auto& fact = facts_.edit(reference.slot);
        fact.retired = !--fact.pins;
    }
    [[nodiscard]] bool pin_remote(ContactPointRef reference)
    {
        if (checked(reference).remotes == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Remote contact fact reference count overflows.");
        pin(reference);
        return ++facts_.edit(reference.slot).remotes == 1;
    }
    [[nodiscard]] bool release_remote(ContactPointRef reference)
    {
        if (!checked(reference).remotes)
            throw std::invalid_argument("Remote contact fact has no fan reference.");
        release(reference);
        return --facts_.edit(reference.slot).remotes == 0;
    }
    [[nodiscard]] ContactPointIdentity get(ContactPointRef reference) const
    {
        const auto& fact = checked(reference);
        return pages_[fact.page].body.get(fact.reference);
    }
    [[nodiscard]] FrozenPoint frozen(ContactPointRef reference) const
    {
        const auto& fact = checked(reference);
        const auto& page = pages_[fact.page].body;
        FrozenPoint result;
        if (page.view)
            result.storage_ = FrozenPoint::Stored{page.view, fact.reference.stored};
        else if (page.source_view)
            result.storage_ = FrozenPoint::Source{page.source_view, fact.reference.source};
        else if (page.fallback && page.fallback->body->sealed)
            result.storage_ = FrozenPoint::Fallback{page.fallback, std::size_t(fact.reference.stored.value)};
        else
            throw std::invalid_argument("Persistent contact fact has no immutable point owner.");
        return result;
    }
    [[nodiscard]] ContactPointRef pinned_reference(std::size_t slot) const
    {
        const auto& fact = facts_[slot];
        if (!fact.pins)
            throw std::out_of_range("Unpinned contact-point slot.");
        return {slot, fact.generation};
    }
    void seal_fallback() noexcept
    {
        if (tail_ != absent)
        {
            pages_[tail_].body.fallback->body->sealed = true;
            tail_ = absent;
        }
    }
    template<class Retire> void reclaim(Retire&& retire)
    {
        for (const auto ref : retired_)
        {
            const auto& before = facts_[ref.slot];
            if (before.generation != ref.generation || !before.retired)
                continue;
            retire(ref.slot);
            fact_ids_.erase(before.digest, ref.slot);
            auto& fact = facts_.edit(ref.slot);
            const auto page = fact.page;
            fact.retired = false;
            fact.page = absent;
            if (fact.generation != std::numeric_limits<std::uint64_t>::max())
            {
                ++fact.generation;
                fact.next_free = free_fact_;
                free_fact_ = ref.slot;
            }
            auto& entry = pages_.edit(page);
            if (!--entry.body.live)
            {
                page_ids_.erase(pointer_digest(entry.body.identity), page);
                if (tail_ == page)
                    tail_ = absent;
                observe(entry.body, false);
                entry = {{}, free_page_};
                free_page_ = page;
            }
        }
        const auto bytes = retired_.capacity() * sizeof(ContactPointRef);
        std::vector<ContactPointRef>().swap(retired_);
        scratch_.release(bytes);
    }

    [[nodiscard]] std::size_t storage_bytes() const noexcept
    {
        return pages_.storage_bytes() + facts_.storage_bytes() + page_ids_.storage_bytes() + fact_ids_.storage_bytes();
    }
    [[nodiscard]] std::size_t copied_pages() const noexcept
    {
        return pages_.copied_pages() + facts_.copied_pages() + page_ids_.copied_pages() + fact_ids_.copied_pages();
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return pages_.overlap_bytes() + facts_.overlap_bytes() + page_ids_.overlap_bytes() + fact_ids_.overlap_bytes();
    }
    void finish_overlap() noexcept
    {
        pages_.finish_overlap();
        facts_.finish_overlap();
        page_ids_.finish_overlap();
        fact_ids_.finish_overlap();
    }
    template<class Visit> void visit_pages(Visit&& visit) const
    {
        pages_.visit_pages(visit);
        facts_.visit_pages(visit);
        page_ids_.visit_pages(visit);
        fact_ids_.visit_pages(visit);
    }
    template<class Visit> void visit_facts(Visit&& visit) const
    {
        for (std::size_t slot = 0; slot < facts_.size(); ++slot)
        {
            const auto& fact = facts_[slot];
            if (!fact.pins && !fact.retired)
                continue;
            const auto& page = pages_[fact.page].body;
            visit(BorrowedFact{&fact, {slot, fact.generation}, fact.pins, fact.retired,
                page.source_view ? nullptr : page.identity,
                page.source_view ? CapturedPointReference{0} : fact.reference.stored, page.get(fact.reference),
                page.source_view ? page.identity : nullptr,
                page.source_view ? fact.reference.source : CapturedSourcePointReference{0}});
        }
    }
    template<class Visit> void visit_point_pages(Visit&& visit) const
    {
        for (std::size_t slot = 0; slot < pages_.size(); ++slot)
        {
            const auto& page = pages_[slot].body;
            if (page.present())
                visit(page);
        }
    }

private:
    [[nodiscard]] const Fact& checked(ContactPointRef reference) const
    {
        const auto& fact = facts_[reference.slot];
        if (fact.generation != reference.generation || (!fact.pins && !fact.retired))
            throw std::out_of_range("Stale contact point reference.");
        return fact;
    }
    static std::uint64_t pointer_digest(const void* pointer) noexcept
    {
        auto value = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }
    [[nodiscard]] std::optional<std::size_t> page_id(const void* identity) const
    {
        return page_ids_.find(pointer_digest(identity),
            [&](std::size_t slot) { return pages_[slot].body.identity == identity; });
    }
    [[nodiscard]] std::optional<ContactPointRef> find(const ContactPointIdentity& value, std::uint64_t digest) const
    {
        const ContactPointKeyView probe{{}, &value};
        const ContactPointWireLess less;
        const auto slot = fact_ids_.find(digest,
            [&](std::size_t candidate)
            {
                const auto& fact = facts_[candidate];
                if (fact.digest != digest)
                    return false;
                const auto physical = pages_[fact.page].body.get(fact.reference);
                const ContactPointKeyView key{{}, &physical};
                return !less(key, probe) && !less(probe, key);
            });
        if (slot)
            return ContactPointRef{*slot, facts_[*slot].generation};
        return {};
    }
    std::size_t add_page(PageBody body, cartesian_detail::ArenaCredits* credits = nullptr)
    {
        const auto slot = free_page_ == absent ? pages_.size() : free_page_;
        auto& target = pages_.edit(slot);
        free_page_ = target.next_free;
        observe(body, true, credits);
        target = {std::move(body), absent};
        page_ids_.insert(slot, [&](std::size_t id) { return pointer_digest(pages_[id].body.identity); });
        return slot;
    }
    ContactPointRef add_fact(std::size_t page, PageBody::Reference reference, std::uint64_t digest)
    {
        const auto slot = free_fact_ == absent ? facts_.size() : free_fact_;
        auto& target = facts_.edit(slot);
        free_fact_ = target.next_free;
        target = {target.generation, 1, 0, page, absent, reference, digest, false};
        ++pages_.edit(page).body.live;
        fact_ids_.insert(slot, [&](std::size_t id) { return facts_[id].digest; });
        return {slot, target.generation};
    }

    void observe(const PageBody& body, bool added, cartesian_detail::ArenaCredits* credits = nullptr)
    {
        if (body.view)
            residency_.observe_point_view(body.view, added);
        else if (body.source_view)
        {
            residency_.observe_point_view(body.source_view, added);
            IncidencePhysicalOwners(residency_, credits, scratch_.arena(), added)
                .observe_snapshot(body.source_view->source());
        }
        else if (added)
            residency_.retain(ContactPhysicalResidency::Role::FallbackPoints, body.fallback,
                body.fallback->metadata.bytes() + body.fallback->body_admission.bytes(),
                &body.fallback->metadata.arena());
        else
            residency_.release(ContactPhysicalResidency::Role::FallbackPoints, body.fallback);
    }

    ContactPhysicalResidency& residency_;
    ContactGroupAllocation scratch_;
    ContactSlotPages<Page> pages_;
    ContactSlotPages<Fact> facts_;
    ContactSlotIndex page_ids_, fact_ids_;
    std::vector<ContactPointRef> retired_;
    std::size_t free_page_ = absent, free_fact_ = absent, tail_ = absent;
};

inline ContactPointRef PersistentContactFacts::retain(ContactPointArray::Locator locator, const PointPool* pool)
{
    if (locator.source_view)
    {
        if (const auto stored = locator.source_view->stored_location(locator.source_reference))
            return retain({stored->view, stored->reference, nullptr, {}}, pool);
        const auto value = locator.source_view->value(locator.source_reference);
        const auto digest = ContactPointWireHasher{}({{}, value});
        if (const auto old = find(value, digest))
        {
            pin(*old);
            return *old;
        }
        if (!pool || !pool->source_arena())
            throw std::invalid_argument("Persistent source facts require their source arena.");
        const auto* identity = locator.source_view->source().get();
        auto page = page_id(identity);
        if (!page)
        {
            auto view = ContactSourcePointView::create(
                locator.source_view->source(), scratch_.arena(), scratch_.keeper(), pool->source_arena());
            (void) view->import_reference(*locator.source_view, locator.source_reference, pool, true);
            view->freeze();
            PageBody input;
            input.source_view = std::move(view);
            input.identity = identity;
            page = add_page(std::move(input), pool->source_arena().get());
        }
        return add_fact(*page, PageBody::Reference(locator.source_reference), digest);
    }
    if (!locator.view)
        throw std::invalid_argument("Missing persistent point-page locator.");
    const auto& value = locator.view->get(locator.reference);
    const auto digest = ContactPointWireHasher{}({{}, value});
    if (const auto old = find(value, digest))
    {
        pin(*old);
        return *old;
    }
    const auto* identity = locator.view->physical_page_identity(locator.reference);
    if (const auto page = page_id(identity))
        return add_fact(*page, PageBody::Reference(CapturedPointReference{locator.reference.value % 16}), digest);
    IdentityPool<ContactPointIdentity> factory(scratch_.arena(), scratch_.keeper());
    auto view = factory.page_view();
    const auto reference =
        view->import_reference(*locator.view, locator.reference, pool ? &pool->physical_pool() : nullptr, true);
    view->freeze();
    PageBody input;
    input.view = std::move(view);
    input.identity = identity;
    return add_fact(add_page(std::move(input)), PageBody::Reference(reference), digest);
}

inline ContactPointRef PersistentContactFacts::retain(const ContactPointIdentity& value)
{
    const auto digest = ContactPointWireHasher{}({{}, value});
    if (const auto old = find(value, digest))
    {
        pin(*old);
        return *old;
    }
    using Credit = PointFactDirectory::FallbackCredit;
    using Body = PointFactDirectory::FallbackBody;
    if (tail_ == absent)
    {
        ContactGroupAllocation staging(scratch_.keeper());
        staging.add(sizeof(Credit) + 128);
        auto owner = std::make_shared<Credit>(scratch_.keeper());
        owner->metadata.transfer_from(staging, sizeof(Credit) + 128);
        owner->body_admission.add(sizeof(Body) + 128 +
            PointFactDirectory::fallback_records_per_page * sizeof(ContactPointIdentity));
        owner->body = std::make_shared<Body>();
        PageBody input;
        input.identity = owner->body.get();
        input.fallback = std::move(owner);
        tail_ = add_page(std::move(input));
    }
    const auto page = tail_;
    auto& body = *pages_[page].body.fallback->body;
    const auto offset = body.size++;
    body.values.push_back(value);
    if (body.size == PointFactDirectory::fallback_records_per_page)
        seal_fallback();
    return add_fact(page, PageBody::Reference(CapturedPointReference{offset}), digest);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
