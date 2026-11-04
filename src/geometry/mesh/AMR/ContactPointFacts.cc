#include "geometry/mesh/AMR/ContactPointFacts.hh"
#include <limits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
PointFactDirectory::PointFactDirectory(std::shared_ptr<Budget> arena) : allocation_(std::move(arena)) {}
PointFactDirectory::~PointFactDirectory() = default;
PointFactDirectory::PointFactDirectory(
    const PointFactDirectory& source, std::shared_ptr<Budget> arena, const PointPool* target_pool, bool preserve_pins)
    : PointFactDirectory(std::move(arena))
{
    allocation_.reserve(entries_, source.entries_.size());
    allocation_.reserve(pages_, source.pages_.size());
    allocation_.reserve(free_entries_, source.free_entries_.size());
    allocation_.reserve(free_pages_, source.free_pages_.size());
    free_pages_ = source.free_pages_;
    allocation_.reserve(buckets_, source.buckets_.size());
    buckets_ = source.buckets_;
    retained_ = source.retained_;
    entries_ = source.entries_;
    if (!preserve_pins)
        for (auto& entry : entries_)
            if (entry.refs || entry.retired)
            {
                entry.refs = 0;
                entry.retired = true;
            }
    free_entries_ = source.free_entries_;
    for (const auto& page : source.pages_)
    {
        Page target;
        target.identity = page.identity;
        target.live = page.live;
        if (page.view)
        {
            PointPool factory(allocation_.arena(), allocation_.keeper());
            auto view = factory.page_view();
            // Each page view in this directory has precisely one physical page.
            (void) view->import_reference(*page.view, {0}, target_pool);
            view->freeze();
            target.view = std::move(view);
        }
        else if (page.fallback)
        {
            target.fallback = page.fallback->body->sealed &&
                page.fallback->metadata.keeper().get() == allocation_.keeper().get() ? page.fallback :
                make_fallback(page.fallback.get());
            target.identity = target.fallback->body.get();
            if (!target.fallback->body->sealed)
                fallback_tail_ = pages_.size();
        }
        if (target.present())
        {
            allocation_.add(page_node_bytes);
            page_index_.emplace(target.identity, pages_.size());
        }
        pages_.push_back(std::move(target));
    }
}
PointFactDirectory::Entry& PointFactDirectory::checked(ContactPointRef ref)
{
    return const_cast<Entry&>(std::as_const(*this).checked(ref));
}
const PointFactDirectory::Entry& PointFactDirectory::checked(ContactPointRef ref) const
{
    if (ref.slot >= entries_.size() || entries_[ref.slot].generation != ref.generation ||
        (!entries_[ref.slot].refs && !entries_[ref.slot].retired))
        throw std::out_of_range("Stale contact-point reference.");
    return entries_[ref.slot];
}
std::optional<ContactPointRef> PointFactDirectory::find(
    const ContactPointIdentity& value, std::uint64_t hash) const
{
    if (buckets_.empty())
        return {};
    const auto mask = buckets_.size() - 1;
    auto bucket = static_cast<std::size_t>(hash) & mask;
    const ContactPointWireLess less;
    const ContactPointKeyView probe{{}, &value};
    for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
    {
        const auto slot = buckets_[bucket];
        if (slot == empty)
            return {};
        if (slot != tombstone)
        {
            const auto& entry = entries_[slot];
            const ContactPointKeyView key{{}, &pages_[entry.page].get(entry.reference)};
            if (entry.hash == hash && !less(key, probe) && !less(probe, key))
                return ContactPointRef{slot, entry.generation};
        }
        bucket = (bucket + 1) & mask;
    }
    return {};
}
void PointFactDirectory::insert_bucket(std::size_t slot)
{
    const auto mask = buckets_.size() - 1;
    auto bucket = static_cast<std::size_t>(entries_[slot].hash) & mask;
    for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
    {
        if (buckets_[bucket] == empty || buckets_[bucket] == tombstone)
        {
            buckets_[bucket] = slot;
            return;
        }
        bucket = (bucket + 1) & mask;
    }
    throw std::invalid_argument("Contact-point lookup has no empty slot.");
}
void PointFactDirectory::ensure_lookup()
{
    if (!buckets_.empty() && retained_ + 1 <= buckets_.size() - buckets_.size() / 4)
        return;
    const auto old = buckets_.capacity();
    if (old > std::numeric_limits<std::size_t>::max() / 2)
        throw std::invalid_argument("Contact-point lookup capacity overflows.");
    const auto capacity = old ? 2 * old : 8;
    ContactGroupAllocation scratch(allocation_.keeper());
    std::vector<std::size_t> replacement;
    scratch.reserve(replacement, capacity);
    replacement.assign(capacity, empty);
    for (std::size_t slot = 0; slot < entries_.size(); ++slot)
        if (entries_[slot].refs || entries_[slot].retired)
        {
            auto bucket = static_cast<std::size_t>(entries_[slot].hash) & (capacity - 1);
            while (replacement[bucket] != empty)
                bucket = (bucket + 1) & (capacity - 1);
            replacement[bucket] = slot;
        }
    allocation_.transfer_from(scratch, capacity * sizeof(std::size_t));
    buckets_.swap(replacement);
    std::vector<std::size_t>().swap(replacement);
    allocation_.release(old * sizeof(std::size_t));
}
const ContactPointIdentity& PointFactDirectory::Page::get(CapturedPointReference reference) const
{
    if (view) return view->get(reference);
    if (!fallback || reference.value >= fallback->body->size)
        throw std::out_of_range("Invalid fallback point reference.");
    return fallback->body->values[reference.value];
}
std::shared_ptr<PointFactDirectory::FallbackCredit> PointFactDirectory::make_fallback(const FallbackCredit* source)
{
    // Partial tails are copied across a fork. Published/full pages are immutable
    // and share their body with an independent target-arena credit.
    ContactGroupAllocation staging(allocation_.keeper());
    staging.add(sizeof(FallbackCredit) + 128);
    auto result = std::shared_ptr<FallbackCredit>(new FallbackCredit(allocation_.keeper()));
    result->metadata.transfer_from(staging, sizeof(FallbackCredit) + 128);
    result->body_admission.add(sizeof(FallbackBody) + 128 + fallback_records_per_page * sizeof(ContactPointIdentity));
    if (source && source->body->sealed)
        result->body = source->body;
    else if (source)
        result->body = std::make_shared<FallbackBody>(*source->body);
    else
        result->body = std::make_shared<FallbackBody>();
    return result;
}
std::size_t PointFactDirectory::add_page(Page page)
{
    auto index = free_pages_.empty() ? pages_.size() : free_pages_.back();
    if (index == pages_.size()) allocation_.append(pages_);
    allocation_.add(page_node_bytes);
    try { page_index_.emplace(page.identity, index); }
    catch (...) { allocation_.release(page_node_bytes); throw; }
    if (index == pages_.size()) pages_.push_back({});
    else free_pages_.pop_back();
    pages_[index] = std::move(page);
    return index;
}
ContactPointRef PointFactDirectory::add_entry(
    std::size_t page, CapturedPointReference reference, std::uint64_t hash)
{
    const auto slot = free_entries_.empty() ? entries_.size() : free_entries_.back();
    if (free_entries_.empty()) entries_.push_back({});
    else free_entries_.pop_back();
    auto& entry = entries_[slot];
    entry.refs = 1;
    entry.retired = false;
    entry.page = page;
    entry.reference = reference;
    entry.hash = hash;
    insert_bucket(slot);
    ++pages_[page].live;
    ++retained_;
    return {slot, entry.generation};
}
ContactPointRef PointFactDirectory::retain_page(
    std::shared_ptr<const ContactPointPageView> view, CapturedPointReference reference, std::uint64_t hash)
{
    ensure_lookup();
    if (free_entries_.empty()) allocation_.append(entries_);
    const auto identity = view->physical_page_identity(reference);
    const auto found = page_index_.find(identity);
    const auto page = found == page_index_.end() ? add_page({std::move(view), {}, identity, 0}) : found->second;
    return add_entry(page, reference, hash);
}
ContactPointRef PointFactDirectory::retain(ContactPointArray::Locator locator, const PointPool* target_pool)
{
    if (!locator.view)
        throw std::invalid_argument("Missing point-page locator.");
    const auto& value = locator.view->get(locator.reference);
    const auto hash = ContactPointWireHasher{}({{}, &value});
    if (const auto existing = find(value, hash))
    {
        pin(*existing);
        return *existing;
    }
    const auto identity = locator.view->physical_page_identity(locator.reference);
    if (const auto found = page_index_.find(identity); found != page_index_.end())
        return retain_page(pages_[found->second].view, {locator.reference.value % 16}, hash);
    PointPool factory(allocation_.arena(), allocation_.keeper());
    auto view = factory.page_view();
    const auto reference = view->import_reference(*locator.view, locator.reference, target_pool);
    view->freeze();
    return retain_page(std::move(view), reference, hash);
}
ContactPointRef PointFactDirectory::retain(const ContactPointIdentity& value)
{
    const auto hash = ContactPointWireHasher{}({{}, &value});
    if (const auto existing = find(value, hash))
    {
        pin(*existing);
        return *existing;
    }
    ensure_lookup();
    if (free_entries_.empty()) allocation_.append(entries_);
    if (fallback_tail_ == empty)
    {
        auto credit = make_fallback();
        const auto* identity = credit->body.get();
        fallback_tail_ = add_page({{}, std::move(credit), identity, 0});
    }
    auto& body = *pages_[fallback_tail_].fallback->body;
    if (body.sealed || body.size == fallback_records_per_page)
        throw std::logic_error("Invalid fallback point tail.");
    const auto offset = body.size;
    body.values.push_back(value);
    ++body.size;
    const auto page = fallback_tail_;
    if (body.size == fallback_records_per_page)
    {
        body.sealed = true;
        fallback_tail_ = empty;
    }
    return add_entry(page, {offset}, hash);
}
void PointFactDirectory::seal_fallback() noexcept
{
    if (fallback_tail_ != empty)
    {
        pages_[fallback_tail_].fallback->body->sealed = true;
        fallback_tail_ = empty;
    }
}
std::size_t PointFactDirectory::fallback_pages() const noexcept
{
    std::size_t count = 0;
    for (const auto& page : pages_) count += bool(page.fallback);
    return count;
}
void PointFactDirectory::pin(ContactPointRef ref)
{
    auto& entry = checked(ref);
    if (entry.refs == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Contact-point reference count overflows.");
    ++entry.refs;
    entry.retired = false;
}
void PointFactDirectory::release(ContactPointRef ref)
{
    auto& entry = checked(ref);
    if (!entry.refs)
        throw std::invalid_argument("Contact-point reference already retired.");
    entry.retired = --entry.refs == 0;
}
const ContactPointIdentity& PointFactDirectory::get(ContactPointRef ref) const
{
    const auto& entry = checked(ref);
    return pages_[entry.page].get(entry.reference);
}
ContactPointRef PointFactDirectory::pinned_reference(std::size_t slot) const
{
    if (slot >= entries_.size() || !entries_[slot].refs)
        throw std::out_of_range("Unpinned contact-point slot.");
    return {slot, entries_[slot].generation};
}
ContactPointRef PointFactDirectory::staging_reference(std::size_t slot) const
{
    if (slot >= entries_.size() || (!entries_[slot].refs && !entries_[slot].retired))
        throw std::out_of_range("Unavailable contact-point staging slot.");
    return {slot, entries_[slot].generation};
}
void PointFactDirectory::reclaim()
{
    std::size_t retiring = 0;
    for (const auto& entry : entries_)
        retiring += entry.retired;
    if (retiring > std::numeric_limits<std::size_t>::max() - free_entries_.size())
        throw std::invalid_argument("Contact-point free list overflows.");
    allocation_.reserve(free_entries_, free_entries_.size() + retiring);
    if (retiring)
        allocation_.reserve(free_pages_, pages_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i)
    {
        auto& entry = entries_[i];
        if (!entry.retired)
            continue;
        // Exhausted generations are quarantined permanently, never wrapped.
        if (entry.generation != std::numeric_limits<std::uint64_t>::max())
        {
            ++entry.generation;
            free_entries_.push_back(i);
        }
        const auto mask = buckets_.size() - 1;
        auto bucket = static_cast<std::size_t>(entry.hash) & mask;
        while (buckets_[bucket] != i)
            bucket = (bucket + 1) & mask;
        buckets_[bucket] = tombstone;
        --retained_;
        entry.retired = false;
        auto& page = pages_[entry.page];
        if (--page.live == 0)
        {
            if (fallback_tail_ == entry.page) fallback_tail_ = empty;
            page_index_.erase(page.identity);
            allocation_.release(page_node_bytes);
            page = {};
            free_pages_.push_back(entry.page);
        }
    }
}
std::size_t PointFactDirectory::live_points() const noexcept
{
    std::size_t n = 0;
    for (const auto& entry : entries_)
        n += entry.refs || entry.retired;
    return n;
}
std::size_t PointFactDirectory::live_pages() const noexcept
{
    std::size_t n = 0;
    for (const auto& page : pages_)
        n += page.present();
    return n;
}
const void* PointFactDirectory::credit_identity(ContactPointRef ref) const
{
    const auto& entry = checked(ref);
    const auto& page = pages_[entry.page];
    return page.view ? page.view->page_credit_identity(entry.reference) : page.fallback.get();
}
void PointFactDirectory::contribute_breakdown(ContactStorageBreakdown& result) const
{
    result.fact_metadata += allocation_.bytes();
    for (const auto& page : pages_)
    {
        if (page.view)
        {
            result.fact_view_metadata += page.view->metadata_bytes();
            page.view->visit_page_admissions([&](const void*, const void*, std::size_t metadata, std::size_t body)
                {
                    result.point_credit_metadata += metadata;
                    result.point_shared_body += body;
                });
        }
        else if (page.fallback)
        {
            result.point_credit_metadata += page.fallback->metadata.bytes();
            result.point_shared_body += page.fallback->body_admission.bytes();
        }
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
