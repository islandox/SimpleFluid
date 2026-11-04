#include "TileBodyPages.hh"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t records_per_page = 16;
static_assert(std::numeric_limits<std::size_t>::digits <= std::numeric_limits<std::uint64_t>::digits);
constexpr auto absent = PooledTile::absent_cell;
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Tile body capacity product overflows.");
    return a * b;
}
std::size_t sum(std::size_t a, std::size_t b)
{
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::invalid_argument("Tile body capacity sum overflows.");
    return a + b;
}
template<class UInt> struct BodyRecord
{
    std::uint64_t emission;
    UInt support, domain, plane, normal_slot, image_slot, first_cell, second_cell, offset, count;
    std::uint8_t present;
};
template<class UInt> struct BodyData
{
    using Index = UInt;
    std::vector<BodyRecord<UInt>> records;
    std::vector<UInt> rings;
};
using CompactData = BodyData<std::uint32_t>;
using WideData = BodyData<std::uint64_t>;
using PageData = std::variant<CompactData, WideData>;
struct BodyPage
{
    PageData data;
    bool sealed = false;
};
template<class UInt> BodyRecord<UInt> record(const PooledTile& source, std::size_t offset)
{
    const auto checked = [](std::size_t value)
    {
        if (value > std::numeric_limits<UInt>::max())
            throw std::invalid_argument("Tile body index cannot be narrowed.");
        return static_cast<UInt>(value);
    };
    return {source.emission, checked(source.support), checked(source.domain), checked(source.plane),
        checked(source.normal_slot), checked(source.image_slot),
        checked(source.first_cell == absent ? 0 : source.first_cell),
        checked(source.second_cell == absent ? 0 : source.second_cell), checked(offset),
        checked(source.intrinsic_loop.size()),
        static_cast<std::uint8_t>((source.first_cell != absent ? 1 : 0) | (source.second_cell != absent ? 2 : 0))};
}
bool needs_wide(const PooledTile& source, std::size_t ring_size)
{
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    const std::array<std::size_t, 7> values{source.support, source.domain, source.plane, source.normal_slot,
        source.image_slot, source.first_cell == absent ? 0 : source.first_cell,
        source.second_cell == absent ? 0 : source.second_cell};
    return ring_size > maximum ||
           std::any_of(values.begin(), values.end(), [](auto value) { return value > maximum; }) ||
           std::any_of(
               source.intrinsic_loop.begin(), source.intrinsic_loop.end(), [](auto value) { return value > maximum; });
}
std::size_t ring_capacity(std::size_t old, std::size_t required)
{
    if (old >= required)
        return old;
    const auto doubled = old > std::numeric_limits<std::size_t>::max() / 2 ? required : 2 * old;
    return std::max(required, doubled ? doubled : std::size_t{1});
}
} // namespace
struct TileBodyPageCredit
{
    Budget* budget;
    std::shared_ptr<Budget> keeper;
    std::shared_ptr<BodyPage> page;
    std::size_t bytes = 0, ordinal = 0;
    std::array<std::atomic<std::size_t>, records_per_page> references{};
    TileBodyPageCredit(Budget& arena, std::shared_ptr<Budget> owner) : budget(&arena), keeper(std::move(owner)) {}
    ~TileBodyPageCredit()
    {
        page.reset();
        budget->used -= bytes;
    }
    void admit(std::size_t amount)
    {
        const auto next = sum(bytes, amount);
        budget->add(amount);
        bytes = next;
    }
    void release(std::size_t amount) noexcept
    {
        bytes -= amount;
        budget->used -= amount;
    }
};
namespace
{
constexpr auto credit_overhead = sizeof(TileBodyPageCredit) + 128;
std::shared_ptr<TileBodyPageCredit> make_credit(
    Budget& budget, const std::shared_ptr<Budget>& keeper, std::shared_ptr<BodyPage> page, std::size_t body_bytes)
{
    budget.add(credit_overhead);
    std::shared_ptr<TileBodyPageCredit> credit;
    try
    {
        credit = std::make_shared<TileBodyPageCredit>(budget, keeper);
    }
    catch (...)
    {
        budget.used -= credit_overhead;
        throw;
    }
    credit->bytes = credit_overhead;
    credit->admit(body_bytes);
    credit->page = std::move(page);
    return credit;
}
std::size_t occupied(const TileBodyPageCredit& credit) noexcept
{
    return std::visit([](const auto& data) { return data.records.size(); }, credit.page->data);
}
template<class UInt> void reserve_exact(std::vector<UInt>& values, std::size_t capacity)
{
    values.reserve(capacity);
    if (values.capacity() != capacity)
        throw std::invalid_argument("Tile body capacity differs from admission.");
}
template<class UInt> std::size_t buffer_bytes(const BodyData<UInt>& data)
{
    return sum(
        product(data.records.capacity(), sizeof(BodyRecord<UInt>)), product(data.rings.capacity(), sizeof(UInt)));
}
std::size_t physical_body_bytes(const BodyPage& page) noexcept
{
    return sizeof(BodyPage) + 128 + std::visit([](const auto& data) { return buffer_bytes(data); }, page.data);
}
template<class UInt> void append_values(BodyData<UInt>& data, const PooledTile& source)
{
    const auto next = record<UInt>(source, data.rings.size());
    // Capacities and every scalar/ring conversion have already been admitted
    // and checked. These integer writes cannot throw or allocate.
    for (const auto value : source.intrinsic_loop)
        data.rings.push_back(static_cast<UInt>(value));
    data.records.push_back(next);
}
void append_body(TileBodyPageCredit& credit, const PooledTile& source)
{
    auto& page = *credit.page;
    const auto size = std::visit([](const auto& data) { return data.rings.size(); }, page.data);
    const auto required = sum(size, source.intrinsic_loop.size());
    if (std::holds_alternative<CompactData>(page.data) && needs_wide(source, required))
    {
        const auto& compact = std::get<CompactData>(page.data);
        const auto capacity = ring_capacity(compact.rings.capacity(), required);
        const auto bytes =
            sum(product(records_per_page, sizeof(BodyRecord<std::uint64_t>)), product(capacity, sizeof(std::uint64_t)));
        credit.admit(bytes);
        WideData wide;
        try
        {
            reserve_exact(wide.records, records_per_page);
            reserve_exact(wide.rings, capacity);
            for (const auto& old : compact.records)
                wide.records.push_back({old.emission, old.support, old.domain, old.plane, old.normal_slot,
                    old.image_slot, old.first_cell, old.second_cell, old.offset, old.count, old.present});
            wide.rings.assign(compact.rings.begin(), compact.rings.end());
            append_values(wide, source);
        }
        catch (...)
        {
            WideData().records.swap(wide.records);
            WideData().rings.swap(wide.rings);
            credit.release(bytes);
            throw;
        }
        const auto previous = buffer_bytes(compact);
        page.data.emplace<WideData>(std::move(wide));
        credit.release(previous);
    }
    else
        std::visit(
            [&](auto& data)
            {
                using UInt = typename std::decay_t<decltype(data)>::Index;
                // Validate every conversion before changing either live buffer.
                (void) record<UInt>(source, size);
                for (const auto value : source.intrinsic_loop)
                    if (value > std::numeric_limits<UInt>::max())
                        throw std::invalid_argument("Tile ring index cannot be narrowed.");
                if (required > data.rings.capacity())
                {
                    const auto capacity = ring_capacity(data.rings.capacity(), required);
                    const auto bytes = product(capacity, sizeof(UInt));
                    const auto previous = product(data.rings.capacity(), sizeof(UInt));
                    credit.admit(bytes);
                    {
                        std::vector<UInt> next;
                        try
                        {
                            reserve_exact(next, capacity);
                            next.assign(data.rings.begin(), data.rings.end());
                        }
                        catch (...)
                        {
                            std::vector<UInt>().swap(next);
                            credit.release(bytes);
                            throw;
                        }
                        data.rings.swap(next);
                    }
                    credit.release(previous);
                }
                append_values(data, source);
            },
            page.data);
    if (occupied(credit) == records_per_page)
        page.sealed = true;
}
} // namespace
TileBodyHandle::TileBodyHandle(std::shared_ptr<TileBodyPageCredit> credit, std::uint8_t slot)
    : credit_(std::move(credit)), slot_(slot)
{
    retain();
}
void TileBodyHandle::retain()
{
    if (!credit_)
        return;
    auto& counter = credit_->references[slot_];
    auto count = counter.load(std::memory_order_relaxed);
    do
    {
        if (count == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Tile body handle count overflows.");
    } while (!counter.compare_exchange_weak(count, count + 1, std::memory_order_relaxed));
}
void TileBodyHandle::release() noexcept
{
    if (credit_)
        credit_->references[slot_].fetch_sub(1, std::memory_order_relaxed);
    credit_.reset();
}
TileBodyHandle::TileBodyHandle(const TileBodyHandle& source) : credit_(source.credit_), slot_(source.slot_)
{
    retain();
}
TileBodyHandle::TileBodyHandle(TileBodyHandle&& source) noexcept
    : credit_(std::move(source.credit_)), slot_(source.slot_)
{
}
TileBodyHandle& TileBodyHandle::operator=(const TileBodyHandle& source)
{
    if (this != &source)
    {
        TileBodyHandle copy(source);
        *this = std::move(copy);
    }
    return *this;
}
TileBodyHandle& TileBodyHandle::operator=(TileBodyHandle&& source) noexcept
{
    if (this != &source)
    {
        release();
        credit_ = std::move(source.credit_);
        slot_ = source.slot_;
    }
    return *this;
}
TileBodyHandle::~TileBodyHandle()
{
    release();
}
TileBodyView TileBodyHandle::view() const
{
    if (!credit_ || !credit_->page->sealed)
        throw std::invalid_argument("Tile body page is not frozen.");
    return TileBodyView(*this);
}
std::shared_ptr<const void> TileBodyHandle::credit_owner() const noexcept
{
    return credit_;
}
std::size_t TileBodyHandle::credit_bytes() const noexcept
{
    return credit_ ? credit_->bytes : 0;
}
const void* TileBodyHandle::page_identity() const noexcept
{
    return credit_ ? credit_->page.get() : nullptr;
}
std::weak_ptr<const void> TileBodyHandle::body_owner() const noexcept
{
    return credit_ ? std::weak_ptr<const void>(credit_->page) : std::weak_ptr<const void>{};
}
std::size_t TileBodyHandle::body_bytes() const noexcept
{
    return credit_ ? physical_body_bytes(*credit_->page) : 0;
}
TileBodyScalars TileBodyView::scalars() const
{
    return handle_.scalars();
}
TileBodyLoopView TileBodyHandle::intrinsic_loop() const
{
    if (!credit_ || !credit_->page->sealed)
        throw std::invalid_argument("Tile body page is not frozen.");
    return TileBodyLoopView(*this);
}
TileBodyScalars TileBodyHandle::scalars() const
{
    if (!credit_ || !credit_->page->sealed)
        throw std::invalid_argument("Tile body page is not frozen.");
    return std::visit(
        [&](const auto& data)
        {
            const auto& value = data.records[slot_];
            return TileBodyScalars{static_cast<std::size_t>(value.support), static_cast<std::size_t>(value.domain),
                static_cast<std::size_t>(value.plane), static_cast<std::size_t>(value.normal_slot),
                static_cast<std::size_t>(value.image_slot),
                (value.present & 1) ? static_cast<std::size_t>(value.first_cell) : absent,
                (value.present & 2) ? static_cast<std::size_t>(value.second_cell) : absent, value.emission};
        },
        credit_->page->data);
}
bool operator==(const TileBodyLoopView& first, const TileBodyLoopView& second)
{
    return first.size() == second.size() && std::equal(first.begin(), first.end(), second.begin());
}
std::size_t TileBodyLoopView::size() const noexcept
{
    if (!handle_)
        return direct_.size();
    return std::visit([&](const auto& data) { return static_cast<std::size_t>(data.records[handle_.slot_].count); },
        handle_.credit_->page->data);
}
std::size_t TileBodyLoopView::operator[](std::size_t index) const
{
    if (!handle_)
        return direct_[index];
    return std::visit(
        [&](const auto& data)
        {
            const auto offset = static_cast<std::size_t>(data.records[handle_.slot_].offset);
            return static_cast<std::size_t>(data.rings[offset + index]);
        },
        handle_.credit_->page->data);
}
std::size_t TileBodyLoopView::at(std::size_t index) const
{
    if (index >= size())
        throw std::out_of_range("Tile body ring index out of range.");
    return (*this)[index];
}
TileBodyPages::TileBodyPages(Budget& budget, std::shared_ptr<Budget> keeper)
    : budget_(&budget), keeper_(std::move(keeper))
{
    if (keeper_ && keeper_.get() != &budget)
        throw std::invalid_argument("Tile body budget keeper mismatch.");
}
TileBodyPages::TileBodyPages(const TileBodyPages& source, Budget& budget, std::shared_ptr<Budget> keeper)
    : TileBodyPages(budget, std::move(keeper))
{
    for (const auto& credit : source.pages_)
        if (credit && !credit->page->sealed)
            throw std::invalid_argument("Tile body source must be frozen before fork.");
    const auto bytes = product(source.pages_.size(), sizeof(decltype(pages_)::value_type));
    budget_->add(bytes);
    charged_ = bytes;
    try
    {
        reserve_exact(pages_, source.pages_.size());
        pages_.resize(source.pages_.size());
        for (std::size_t i = 0; i < source.pages_.size(); ++i)
            if (const auto& old = source.pages_[i])
            {
                pages_[i] = make_credit(*budget_, keeper_, old->page, old->bytes - credit_overhead);
                pages_[i]->ordinal = i;
            }
        while (first_free_ < pages_.size() && pages_[first_free_])
            ++first_free_;
        for (const auto& [owner, body_bytes] : source.deferred_)
            if (!owner.expired())
            {
                const auto admission = sum(body_bytes, deferred_node_bytes);
                budget_->add(admission);
                try
                {
                    deferred_.emplace(owner, body_bytes);
                }
                catch (...)
                {
                    budget_->used -= admission;
                    throw;
                }
                deferred_body_bytes_ += body_bytes;
                deferred_metadata_bytes_ += deferred_node_bytes;
            }
    }
    catch (...)
    {
        decltype(pages_)().swap(pages_);
        release_deferred();
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
TileBodyPages::~TileBodyPages()
{
    decltype(pages_)().swap(pages_);
    // Retiring this admission domain follows DeferredOwnerCharges: surviving
    // foreign bodies have independent admission in their remaining domains.
    release_deferred();
    budget_->used -= charged_;
}
void TileBodyPages::release_deferred() noexcept
{
    deferred_.clear();
    budget_->used -= deferred_body_bytes_ + deferred_metadata_bytes_;
    deferred_body_bytes_ = deferred_metadata_bytes_ = 0;
}
void TileBodyPages::grow_directory()
{
    if (pages_.size() != pages_.capacity())
        return;
    const auto previous = pages_.capacity();
    const auto next = previous ? sum(previous, previous) : 1;
    const auto bytes = product(next, sizeof(decltype(pages_)::value_type));
    budget_->add(bytes);
    {
        decltype(pages_) replacement;
        try
        {
            reserve_exact(replacement, next);
            replacement.assign(pages_.begin(), pages_.end());
        }
        catch (...)
        {
            decltype(pages_)().swap(replacement);
            budget_->used -= bytes;
            throw;
        }
        pages_.swap(replacement);
    }
    const auto old = product(previous, sizeof(decltype(pages_)::value_type));
    budget_->used -= old;
    charged_ = bytes;
}
std::size_t TileBodyPages::install(std::shared_ptr<TileBodyPageCredit> credit)
{
    const auto index = first_free_;
    if (index == pages_.size())
    {
        grow_directory();
        pages_.push_back(std::move(credit));
    }
    else
        pages_[index] = std::move(credit);
    pages_[index]->ordinal = index;
    while (first_free_ < pages_.size() && pages_[first_free_])
        ++first_free_;
    return index;
}
TileBodyHandle TileBodyPages::append(const PooledTile& source)
{
    if (active_ == std::numeric_limits<std::size_t>::max())
    {
        const bool wide = needs_wide(source, source.intrinsic_loop.size());
        const auto record_size = wide ? sizeof(BodyRecord<std::uint64_t>) : sizeof(BodyRecord<std::uint32_t>);
        const auto record_bytes = product(records_per_page, record_size);
        auto credit = make_credit(*budget_, keeper_, {}, sum(sizeof(BodyPage) + 128, record_bytes));
        credit->page = std::make_shared<BodyPage>();
        if (wide)
            credit->page->data.emplace<WideData>();
        std::visit([](auto& data) { reserve_exact(data.records, records_per_page); }, credit->page->data);
        append_body(*credit, source);
        active_ = install(std::move(credit));
        return TileBodyHandle(pages_[active_], 0);
    }
    auto& credit = pages_[active_];
    const auto slot = occupied(*credit);
    append_body(*credit, source);
    TileBodyHandle result(credit, static_cast<std::uint8_t>(slot));
    if (credit->page->sealed)
        active_ = std::numeric_limits<std::size_t>::max();
    return result;
}
TileBodyHandle TileBodyPages::adopt(const TileBodyHandle& source)
{
    if (!source || !source.credit_->page->sealed)
        throw std::invalid_argument("Only frozen tile bodies can be adopted.");
    const auto ordinal = source.credit_->ordinal;
    if (ordinal < pages_.size() && pages_[ordinal] && pages_[ordinal]->page == source.credit_->page)
        return TileBodyHandle(pages_[ordinal], source.slot_);
    for (const auto& credit : pages_)
        if (credit && credit->page == source.credit_->page)
            return TileBodyHandle(credit, source.slot_);
    const WeakBodyOwner owner(source.credit_->page);
    const auto deferred = deferred_.find(owner);
    const auto body_bytes = physical_body_bytes(*source.credit_->page);
    auto credit = make_credit(*budget_, keeper_, source.credit_->page, deferred == deferred_.end() ? body_bytes : 0);
    const auto index = install(std::move(credit));
    if (deferred != deferred_.end())
    {
        // The directory installation is the last potentially throwing step.
        // Transfer the already-admitted body bytes without touching used or
        // cumulative; destroy the weak node before refunding its allocation.
        pages_[index]->bytes += deferred->second;
        deferred_body_bytes_ -= deferred->second;
        deferred_.erase(deferred);
        deferred_metadata_bytes_ -= deferred_node_bytes;
        budget_->used -= deferred_node_bytes;
    }
    return TileBodyHandle(pages_[index], source.slot_);
}
void TileBodyPages::freeze() noexcept
{
    for (const auto& credit : pages_)
        if (credit && !credit->page->sealed)
            credit->page->sealed = true;
    active_ = std::numeric_limits<std::size_t>::max();
}
void TileBodyPages::defer_body(TileBodyPageCredit& credit)
{
    const WeakBodyOwner owner(credit.page);
    const auto bytes = physical_body_bytes(*credit.page);
    if (deferred_.contains(owner))
        throw std::invalid_argument("Tile body has conflicting active and deferred admission.");
    // Admit only the retirement metadata: these body bytes are already live.
    budget_->add(deferred_node_bytes);
    try
    {
        deferred_.emplace(owner, bytes);
    }
    catch (...)
    {
        budget_->used -= deferred_node_bytes;
        throw;
    }
    deferred_body_bytes_ += bytes;
    deferred_metadata_bytes_ += deferred_node_bytes;
    credit.bytes -= bytes;
}
TileBodyReapStatistics TileBodyPages::reap()
{
    TileBodyReapStatistics result;
    for (auto position = deferred_.begin(); position != deferred_.end();)
    {
        if (!position->first.expired())
        {
            ++position;
            continue;
        }
        const auto bytes = position->second;
        position = deferred_.erase(position);
        deferred_body_bytes_ -= bytes;
        deferred_metadata_bytes_ -= deferred_node_bytes;
        budget_->used -= bytes + deferred_node_bytes;
        result.released_bytes += bytes + deferred_node_bytes;
    }
    for (auto& credit : pages_)
        if (credit && credit->page->sealed && credit.use_count() == 1)
        {
            const auto occupied_records = occupied(*credit);
            if (credit->page.use_count() != 1)
            {
                const auto body_bytes = physical_body_bytes(*credit->page);
                defer_body(*credit);
                ++result.deferred_pages;
                result.deferred_body_bytes += body_bytes;
            }
            ++result.pages;
            result.occupied_records += occupied_records;
            result.released_bytes += credit->bytes;
            first_free_ = std::min(first_free_, credit->ordinal);
            credit.reset();
        }
    return result;
}
TileBodyPageStatistics TileBodyPages::statistics() const noexcept
{
    TileBodyPageStatistics result;
    result.metadata_bytes = metadata_bytes();
    result.deferred_pages = deferred_.size();
    result.deferred_body_bytes = deferred_body_bytes_;
    result.deferred_metadata_bytes = deferred_metadata_bytes_;
    for (const auto& credit : pages_)
        if (credit)
        {
            ++result.pages;
            result.wide_pages += std::holds_alternative<WideData>(credit->page->data);
            result.occupied_records += occupied(*credit);
            result.record_capacity += records_per_page;
            result.page_credit_bytes += credit->bytes;
            result.page_body_bytes += physical_body_bytes(*credit->page);
            result.page_credit_overhead_bytes += credit_overhead;
            for (const auto& refs : credit->references)
                result.live_records += refs.load(std::memory_order_relaxed) != 0;
            std::visit(
                [&](const auto& data)
                {
                    using UInt = typename std::decay_t<decltype(data)>::Index;
                    result.ring_size += data.rings.size();
                    result.ring_capacity += data.rings.capacity();
                    result.ring_bytes += data.rings.capacity() * sizeof(UInt);
                },
                credit->page->data);
        }
    return result;
}
std::size_t TileBodyPages::retained_bytes() const noexcept
{
    return metadata_bytes() + deferred_body_bytes_ + statistics().page_credit_bytes;
}
std::size_t TileBodyPages::credit_bytes(const std::shared_ptr<TileBodyPageCredit>& credit) noexcept
{
    return credit->bytes;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
