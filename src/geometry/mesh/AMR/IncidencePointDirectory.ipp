#pragma once
#include <limits>
#include <stdexcept>
#include <utility>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
template<class H, class P>
std::unique_ptr<typename IncidencePointDirectory<H, P>::Store> IncidencePointDirectory<H, P>::make_store(
    const Store* copy, cartesian_detail::Allocation& memory)
{
    memory.retain(sizeof(Store) + 128);
    return copy ? std::make_unique<Store>(*copy, *budget_, keeper_)
                : std::make_unique<Store>(*budget_, keeper_, hasher_, payload_, storage_);
}
template<class H, class P>
IncidencePointDirectory<H, P>::IncidencePointDirectory(
    Budget& budget, std::shared_ptr<Budget> keeper, H hash, P payload, IdentityPoolStorage storage)
    : budget_(&budget), keeper_(std::move(keeper)), hasher_(std::move(hash)), payload_(std::move(payload)),
      storage_(keeper_ ? storage : IdentityPoolStorage::Flat)
{
    cartesian_detail::check_keeper(budget, keeper_);
    cartesian_detail::Allocation memory(budget);
    store_ = make_store(nullptr, memory);
    charge_ = std::exchange(memory.bytes, 0);
}
template<class H, class P>
void IncidencePointDirectory<H, P>::credit_source(
    cartesian_detail::CreditConstruction& credits, cartesian_detail::Allocation& memory)
{
    if (!source_)
        return;
    credits.retain(
        source_, source_->allocation_budget(), source_->metadata_bytes(), source_->has_arena_keeper(), leases_, memory);
    if (credits.arena->prepaid(source_, source_->allocation_budget(), source_->metadata_bytes(), false))
        return;
    source_->visit_dependencies<CartesianDependencyVisit::Physical>(
        [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
        {
            using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
            const std::shared_ptr<const std::remove_const_t<Element>> fixed = owner;
            credits.retain(fixed, origin, bytes, keeper, leases_, memory);
        });
}
template<class H, class P>
IncidencePointDirectory<H, P>::IncidencePointDirectory(const IncidencePointDirectory& old, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
    : budget_(&budget), keeper_(std::move(keeper)), hasher_(old.hasher_), payload_(old.payload_),
      arena_(std::move(arena)), storage_(keeper_ ? old.storage_ : IdentityPoolStorage::Flat)
{
    cartesian_detail::check_keeper(budget, keeper_);
    if (old.source_ && (!arena_ || arena_->budget != &budget))
        throw std::invalid_argument("Source point fork requires its explicit candidate arena.");
    cartesian_detail::Allocation memory(budget);
    struct ConstructionCleanup
    {
        IncidencePointDirectory& owner;
        bool committed = false;
        ~ConstructionCleanup()
        {
            if (committed)
                return;
            owner.store_.reset();
            owner.source_.reset();
            decltype(owner.leases_){}.swap(owner.leases_);
        }
    } cleanup{*this};
    if (old.source_)
    {
        cartesian_detail::CreditConstruction credits(arena_);
        source_ = old.source_;
        credit_source(credits, memory);
        store_ = make_store(old.store_.get(), memory);
        charge_ = std::exchange(memory.bytes, 0);
        credits.committed = true;
    }
    else
    {
        store_ = make_store(old.store_.get(), memory);
        charge_ = std::exchange(memory.bytes, 0);
    }
    cleanup.committed = true;
}
template<class H, class P> IncidencePointDirectory<H, P>::~IncidencePointDirectory()
{
    store_.reset();
    source_.reset();
    decltype(leases_){}.swap(leases_);
    budget_->used -= charge_ + replaced_overlap_;
}
template<class H, class P> std::size_t IncidencePointDirectory<H, P>::size() const noexcept
{
    return source_size() + store_->size();
}
template<class H, class P>
typename IncidencePointDirectory<H, P>::Point IncidencePointDirectory<H, P>::get(std::size_t slot) const
{
    if (slot < source_size())
        return source_->value({slot});
    return store_->get(slot - source_size());
}
template<class H, class P> std::optional<std::size_t> IncidencePointDirectory<H, P>::find(const Point& point) const
{
    if (source_)
    {
        std::uint64_t hash = 0;
        if constexpr (H::enabled)
            hash = hasher_(point);
        if (const auto slot = source_->find(point, hash))
            return slot;
    }
    if (const auto slot = store_->find(point))
        return cartesian_detail::add(source_size(), *slot);
    return {};
}
template<class H, class P> std::size_t IncidencePointDirectory<H, P>::intern(const Point& point)
{
    if (const auto slot = find(point))
        return *slot;
    return cartesian_detail::add(source_size(), store_->intern(point));
}
template<class H, class P> std::size_t IncidencePointDirectory<H, P>::intern(Point&& point)
{
    if (const auto slot = find(point))
        return *slot;
    return cartesian_detail::add(source_size(), store_->intern(std::move(point)));
}
template<class H, class P>
typename IncidencePointDirectory<H, P>::PreparedSource IncidencePointDirectory<H, P>::prepare_source(
    const ContactSourcePointAdmission& admission, std::shared_ptr<CartesianAuthorityArena> arena)
{
    // A custom payload policy retains its original explicit admission semantics.
    if constexpr (!std::is_same_v<P, IdentityPoolPayload<Point>>)
        return {};
    if (!arena || arena->budget != budget_)
        throw std::invalid_argument("Foreign source point arena.");
    PreparedSource prepared;
    prepared.prefix_ = source_;
    prepared.prefix_namespace_ = source_ ? source_->slot_namespace() : nullptr;
    prepared.trusted_prefix_ = std::is_same_v<H, IdentityPoolHasher<Point>>;
    prepared.snapshot_ = ContactSourcePointSnapshot::capture_impl(
        admission, H::enabled,
        [&](std::size_t index, const ContactSourcePointEmit& emit)
        {
            const auto point = get(index);
            std::uint64_t hash = 0;
            if constexpr (H::enabled)
                hash = hasher_(point);
            emit(point, hash);
        },
        *budget_, keeper_, arena, prepared.trusted_prefix_, prepared.trusted_prefix_ ? &prepared.prefix_ : nullptr);
    arena_ = std::move(arena);
    return prepared;
}
template<class H, class P>
template<class Retire>
bool IncidencePointDirectory<H, P>::publish_source(
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot, Retire retire)
{
    return publish_source_from(std::move(snapshot), 0, std::move(retire));
}
template<class H, class P>
template<class Retire>
bool IncidencePointDirectory<H, P>::publish_source(const PreparedSource& prepared, Retire retire)
{
    if (!ContactSourcePointSnapshot::same_snapshot(prepared.prefix_, source_) ||
        prepared.prefix_namespace_ != (source_ ? source_->slot_namespace() : nullptr))
        return false;
    return publish_source_from(prepared.snapshot_, prepared.trusted_prefix_ ? source_size() : 0, std::move(retire));
}
template<class H, class P>
template<class Retire>
bool IncidencePointDirectory<H, P>::publish_source_from(
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot, std::size_t first, Retire retire)
{
    if (!ContactSourcePointSnapshot::exact_control(snapshot) || !arena_ || snapshot->size() != size() ||
        snapshot->allocation_budget() != budget_ || snapshot->hashing() != H::enabled ||
        (source_ && snapshot->generation() <= source_->generation()))
        return false;
    for (std::size_t i = first; i < snapshot->size(); ++i)
        if (snapshot->value({i}) != get(i))
            return false;
    cartesian_detail::Allocation memory(*budget_);
    auto replacement = make_store(nullptr, memory);
    if constexpr (!std::is_same_v<Retire, std::nullptr_t>)
        retire(*store_);
    replaced_overlap_ += store_->take_replaced_overlap();
    source_.swap(snapshot);
    store_.swap(replacement);
    replacement.reset();
    // Retire the old explicit-directory object only after its actual destructor.
    const auto previous = sizeof(Store) + 128;
    budget_->used -= previous;
    charge_ -= previous;
    charge_ = cartesian_detail::add(charge_, std::exchange(memory.bytes, 0));
    const auto old_lease_bytes = cartesian_detail::product(leases_.capacity(), sizeof(leases_[0]));
    decltype(leases_){}.swap(leases_);
    budget_->used -= old_lease_bytes;
    charge_ -= old_lease_bytes;
    snapshot.reset();
    arena_->reap();
    return true;
}
template<class H, class P> std::size_t IncidencePointDirectory<H, P>::retained_bytes() const
{
    auto bytes = cartesian_detail::add(charge_, store_->retained_bytes());
    if (source_ && source_->allocation_budget() == budget_)
        bytes = cartesian_detail::add(bytes, source_->metadata_bytes());
    for (const auto& credit : leases_)
        bytes = cartesian_detail::add(bytes, cartesian_detail::add(credit->payload, credit->metadata));
    return bytes;
}
template<class H, class P> std::size_t IncidencePointDirectory<H, P>::metadata_bytes() const
{
    return cartesian_detail::add(charge_, store_->metadata_bytes());
}
template<class H, class P>
template<class Visitor>
void IncidencePointDirectory<H, P>::visit_borrowed_payload(Visitor&& visitor) const
{
    store_->visit_borrowed_payload(visitor, true);
    if (source_)
        source_->visit_borrowed_payload(visitor);
    for (const auto& lease : leases_)
        lease->visit_borrowed_payload(visitor);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
