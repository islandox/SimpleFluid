#include "TileBodyPages.hh"
#include "IncidencePhysicalOwners.hh"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <exception>
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
    std::shared_ptr<const ContactSourcePointSnapshot> source;
    unsigned source_bits = 0;
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
struct TileSourceCredit
{
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::shared_ptr<const ContactSourcePointSnapshot> source;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
    std::size_t bytes = 0;
    ~TileSourceCredit()
    {
        source.reset();
        decltype(leases){}.swap(leases);
        if (budget)
            budget->used -= bytes;
    }
};
struct TileBodyPageCredit
{
    Budget* budget;
    std::shared_ptr<Budget> keeper;
    std::shared_ptr<BodyPage> page;
    std::shared_ptr<TileSourceCredit> source;
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
std::shared_ptr<TileSourceCredit> source_credit(std::shared_ptr<const ContactSourcePointSnapshot> source,
    Budget& budget, const std::shared_ptr<Budget>& keeper, const std::shared_ptr<CartesianAuthorityArena>& arena)
{
    if (!source || !arena || arena->budget != &budget)
        throw std::invalid_argument("Tile source loops require their explicit arena.");
    cartesian_detail::Allocation memory(budget);
    memory.retain(sizeof(TileSourceCredit) + 128);
    cartesian_detail::CreditConstruction credits(arena);
    auto result = std::make_shared<TileSourceCredit>();
    result->source = std::move(source);
    credits.retain(result->source, result->source->allocation_budget(), result->source->metadata_bytes(),
        result->source->has_arena_keeper(), result->leases, memory);
    if (!arena->prepaid(result->source, result->source->allocation_budget(), result->source->metadata_bytes(), false))
        result->source->visit_dependencies<CartesianDependencyVisit::Physical>(
            [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
            {
                using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
                const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
                credits.retain(fixed, origin, bytes, keep, result->leases, memory);
            });
    result->budget = &budget;
    result->keeper = keeper;
    result->bytes = std::exchange(memory.bytes, 0);
    credits.committed = true;
    return result;
}
template<class UInt> std::uint64_t quad_code(const BodyRecord<UInt>& record)
{
    if constexpr (sizeof(UInt) == 4)
        return std::uint64_t(record.offset) | (std::uint64_t(record.count) << 32);
    else
        return record.offset;
}
std::optional<std::size_t> source_corner(
    const ContactSourcePointSnapshot& source, std::uint64_t code, unsigned bits, std::size_t corner)
{
    if (corner >= 4 || bits > 12)
        return {};
    const auto mask = bits ? (std::uint64_t(1) << bits) - 1 : 0;
    std::array<std::size_t, 3> point{};
    for (unsigned a = 0; a < 3; ++a)
        point[a] = (code >> (a * bits)) & mask;
    const auto a = unsigned((code >> (5 * bits)) & 3), b = unsigned((code >> (5 * bits + 2)) & 3);
    if (a >= 3 || b >= 3 || a == b)
        return {};
    if (corner == 1 || corner == 2)
        point[a] = (code >> (3 * bits)) & mask;
    if (corner == 2 || corner == 3)
        point[b] = (code >> (4 * bits)) & mask;
    return source.find_components(point);
}
bool certify_quad(const ContactSourcePointSnapshot& source, std::span<const std::size_t, 4> points, unsigned bits,
    std::uint64_t& code)
{
    std::array<std::array<std::size_t, 3>, 4> values{};
    for (unsigned p = 0; p < 4; ++p)
    {
        const auto value = source.component_indices({points[p]});
        if (!value)
            return false;
        values[p] = *value;
    }
    unsigned a = 3, b = 3;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (values[0][axis] != values[1][axis])
        {
            if (a != 3)
                return false;
            a = axis;
        }
        if (values[0][axis] != values[3][axis])
        {
            if (b != 3)
                return false;
            b = axis;
        }
    }
    if (a == 3 || b == 3 || a == b)
        return false;
    auto opposite = values[0];
    opposite[a] = values[1][a];
    opposite[b] = values[3][b];
    if (opposite != values[2])
        return false;
    code = 0;
    const std::array<std::size_t, 5> selected{values[0][0], values[0][1], values[0][2], values[1][a], values[3][b]};
    const auto maximum = bits ? (std::uint64_t(1) << bits) - 1 : 0;
    for (unsigned field = 0; field < 5; ++field)
    {
        if (selected[field] > maximum)
            return false;
        code |= std::uint64_t(selected[field]) << (field * bits);
    }
    code |= std::uint64_t(a) << (5 * bits);
    code |= std::uint64_t(b) << (5 * bits + 2);
    for (unsigned p = 0; p < 4; ++p)
    {
        const auto replay = source_corner(source, code, bits, p);
        if (!replay || *replay != points[p] || source.value({*replay}) != source.value({points[p]}))
            return false;
    }
    return true;
}
constexpr auto credit_overhead = sizeof(TileBodyPageCredit) + 128;
std::shared_ptr<TileBodyPageCredit> make_credit(Budget& budget, const std::shared_ptr<Budget>& keeper,
    std::shared_ptr<BodyPage> page, std::size_t body_bytes, std::shared_ptr<TileSourceCredit> source = {})
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
    credit->source = std::move(source);
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
    return body_storage_owner();
}
std::shared_ptr<const void> TileBodyHandle::body_storage_owner() const noexcept
{
    return credit_ ? std::shared_ptr<const void>(credit_->page) : nullptr;
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
    if (indexed_at_)
        return indexed_count_;
    if (!handle_)
        return direct_.size();
    if (handle_.credit_->page->source)
        return 4;
    return std::visit([&](const auto& data) { return static_cast<std::size_t>(data.records[handle_.slot_].count); },
        handle_.credit_->page->data);
}
std::size_t TileBodyLoopView::operator[](std::size_t index) const
{
    if (indexed_at_)
        return indexed_at_(indexed_owner_, index);
    if (!handle_)
        return direct_[index];
    if (handle_.credit_->page->source)
        return std::visit(
            [&](const auto& data)
            {
                const auto slot = source_corner(*handle_.credit_->page->source, quad_code(data.records[handle_.slot_]),
                    handle_.credit_->page->source_bits, index);
                if (!slot)
                    throw std::invalid_argument("Certified tile source lost a complete corner identity.");
                return *slot;
            },
            handle_.credit_->page->data);
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
TileBodyPages::TileBodyPages(const TileBodyPages& source, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> source_arena)
    : TileBodyPages(budget, std::move(keeper))
{
    source_arena_ = std::move(source_arena);
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
                auto credit = make_credit(*budget_, keeper_, old->page, old->bytes - credit_overhead,
                    old->page->source ? source_context(old->page->source) : nullptr);
                retain_source_context(credit->source);
                credit->ordinal = i;
                pages_[i] = std::move(credit);
                ++active_page_credits_;
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
        source_credit_.reset();
        release_source_context_index();
        release_deferred();
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
TileBodyPages::~TileBodyPages()
{
    decltype(pages_)().swap(pages_);
    source_credit_.reset();
    release_source_context_index();
    // Retiring this admission domain follows DeferredOwnerCharges: surviving
    // foreign bodies have independent admission in their remaining domains.
    release_deferred();
    budget_->used -= charged_;
}
void TileBodyPages::retain_source_context(const std::shared_ptr<TileSourceCredit>& context)
{
    if (!context)
        return;
    const auto key = context->source.get();
    if (const auto found = source_contexts_.find(key); found != source_contexts_.end())
    {
        if (found->second.owner.owner_before(context) || context.owner_before(found->second.owner))
            throw std::logic_error("Tile source context allocation identity changed.");
        found->second.references = sum(found->second.references, 1);
        return;
    }
    const auto next = sum(source_context_bytes_, context->bytes);
    budget_->add(source_context_node_bytes);
    try
    {
        source_contexts_.emplace(key, SourceContextUse{context, 1});
    }
    catch (...)
    {
        budget_->used -= source_context_node_bytes;
        throw;
    }
    source_context_bytes_ = next;
}
void TileBodyPages::release_source_context(const std::shared_ptr<TileSourceCredit>& context) noexcept
{
    if (!context)
        return;
    const auto found = source_contexts_.find(context->source.get());
    if (found == source_contexts_.end() || !found->second.references || found->second.owner.owner_before(context) ||
        context.owner_before(found->second.owner))
        std::terminate();
    if (--found->second.references == 0)
    {
        source_context_bytes_ -= context->bytes;
        source_contexts_.erase(found);
        budget_->used -= source_context_node_bytes;
    }
}
void TileBodyPages::release_source_context_index() noexcept
{
    const auto bytes = source_contexts_.size() * source_context_node_bytes;
    source_contexts_.clear();
    budget_->used -= bytes;
    source_context_bytes_ = 0;
}
std::shared_ptr<TileSourceCredit> TileBodyPages::source_context(
    std::shared_ptr<const ContactSourcePointSnapshot> source)
{
    if (const auto found = source_contexts_.find(source.get()); found != source_contexts_.end())
    {
        auto existing = found->second.owner.lock();
        if (!existing)
            throw std::logic_error("Live tile source context lost its physical owner.");
        return existing;
    }
    auto next = source_credit(std::move(source), *budget_, keeper_, source_arena_);
    retain_source_context(next);
    release_source_context(source_credit_);
    source_credit_ = std::move(next);
    return source_credit_;
}
std::size_t TileBodyPages::certify_private_source_loops(
    std::shared_ptr<const ContactSourcePointSnapshot> source, std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (!source || !arena || arena->budget != budget_ || source->allocation_budget() != budget_)
        throw std::invalid_argument("Foreign private tile-source certification arena.");
    freeze(); // Encoded source pages can never become a later append tail.
    source_arena_ = std::move(arena);
    std::size_t maximum = 0;
    for (const auto count : source->component_counts())
        if (count)
            maximum = std::max(maximum, count - 1);
    const auto bits = unsigned(std::bit_width(maximum));
    if (bits > 12)
        return 0;
    std::size_t records = 0;
    for (auto& credit : pages_)
        if (credit && !credit->page->source && credit->page.use_count() == 1)
        {
            std::array<std::uint64_t, records_per_page> codes{};
            const bool matches = std::visit(
                [&](const auto& data)
                {
                    for (std::size_t index = 0; index < data.records.size(); ++index)
                    {
                        const auto& entry = data.records[index];
                        if (entry.count != 4 || entry.offset > data.rings.size() ||
                            data.rings.size() - entry.offset < 4)
                            return false;
                        std::array<std::size_t, 4> loop{};
                        for (unsigned p = 0; p < 4; ++p)
                            loop[p] = data.rings[entry.offset + p];
                        if (!certify_quad(*source, loop, bits, codes[index]))
                            return false;
                    }
                    return true;
                },
                credit->page->data);
            if (!matches)
                continue;
            auto context = source_context(source);
            retain_source_context(context); // All throwing admission precedes publication.
            std::size_t old_rings = 0;
            std::visit(
                [&](auto& data)
                {
                    using UInt = typename std::remove_cvref_t<decltype(data)>::Index;
                    for (std::size_t index = 0; index < data.records.size(); ++index)
                    {
                        data.records[index].offset = UInt(codes[index]);
                        data.records[index].count = sizeof(UInt) == 4 ? UInt(codes[index] >> 32) : 0;
                    }
                    old_rings = product(data.rings.capacity(), sizeof(UInt));
                    decltype(data.rings){}.swap(data.rings);
                },
                credit->page->data);
            credit->page->source = source;
            credit->page->source_bits = bits;
            credit->source = std::move(context);
            credit->release(old_rings);
            records += occupied(*credit);
        }
    return records;
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
        grow_directory();
    retain_source_context(credit->source);
    if (index == pages_.size())
        pages_.push_back(std::move(credit));
    else
        pages_[index] = std::move(credit);
    ++active_page_credits_;
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
    auto credit = make_credit(*budget_, keeper_, source.credit_->page, deferred == deferred_.end() ? body_bytes : 0,
        source.credit_->page->source ? source_context(source.credit_->page->source) : nullptr);
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
            release_source_context(credit->source);
            --active_page_credits_;
            credit.reset();
        }
    if (source_credit_ && source_credit_.use_count() == 1)
    {
        release_source_context(source_credit_);
        source_credit_.reset();
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
            if (credit->page->source)
            {
                ++result.source_pages;
                result.source_records += occupied(*credit);
            }
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
namespace
{
void visit_tile_source(const std::shared_ptr<TileSourceCredit>& source,
    const std::function<void(const ContactPayloadAllocation&)>& visitor)
{
    if (!source)
        return;
    contact_payload_object(
        visitor, source.get(), ContactPayloadRole::TileSourceCreditObject, sizeof(TileSourceCredit), true);
    contact_payload_buffer(visitor, source.get(), ContactPayloadRole::TileSourceLeaseValues, source->leases);
    for (const auto& lease : source->leases)
        lease->visit_borrowed_payload(visitor);
    if (source->source)
        source->source->visit_borrowed_payload(visitor);
}
void visit_tile_credit(const std::shared_ptr<TileBodyPageCredit>& credit,
    const std::function<void(const ContactPayloadAllocation&)>& visitor)
{
    if (!credit)
        return;
    contact_payload_object(
        visitor, credit.get(), ContactPayloadRole::TilePageCreditObject, sizeof(TileBodyPageCredit), true);
    const auto& page = *credit->page;
    contact_payload_object(visitor, &page, ContactPayloadRole::TilePageObject, sizeof(BodyPage), true);
    std::visit(
        [&](const auto& data)
        {
            contact_payload_buffer(visitor, &page, ContactPayloadRole::TilePageRecords, data.records);
            contact_payload_buffer(visitor, &page, ContactPayloadRole::TilePageRings, data.rings);
        },
        page.data);
    if (page.source)
        page.source->visit_borrowed_payload(visitor);
    visit_tile_source(credit->source, visitor);
}
} // namespace
void TileBodyHandle::observe_source_owners(IncidencePhysicalOwners& owners) const
{
    using Role = IncidencePhysicalOwners::Role;
    if (!credit_ || !credit_->source)
        return;
    const auto& source = credit_->source;
    if (!owners.observe(Role::TileSourceContext, source, source->bytes, source->budget))
        return;
    owners.observe_source(source->source);
    for (const auto& lease : source->leases)
        owners.observe(Role::CreditMetadata, lease, lease->metadata, lease->budget, lease->arena_scope);
}
const Budget* TileBodyHandle::credit_origin() const noexcept { return credit_ ? credit_->budget : nullptr; }
void TileBodyHandle::visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    visit_tile_credit(credit_, visitor);
}
void TileBodyPages::visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_buffer(visitor, this, ContactPayloadRole::TilePageDirectory, pages_);
    for (const auto& entry : deferred_)
        contact_payload_node_value(visitor, this, ContactPayloadRole::DeferredOwnerChargeNodes, entry);
    for (const auto& context : source_contexts_)
        contact_payload_node_value(visitor, this, ContactPayloadRole::TileSourceContextIndexNodes, context);
    visit_tile_source(source_credit_, visitor);
    for (const auto& credit : pages_)
        visit_tile_credit(credit, visitor);
}
bool TileBodyPages::imports_source_owner(const void* identity) const noexcept
{
    const auto contains = [&](const std::shared_ptr<TileSourceCredit>& credit)
    {
        bool found = false;
        if (credit)
            credit->source->visit_dependencies(
                [&](const auto& owner, const Budget*, std::size_t, bool) { found = found || owner.get() == identity; });
        return found;
    };
    if (contains(source_credit_))
        return true;
    for (const auto& page : pages_)
        if (page && contains(page->source))
            return true;
    return false;
}
std::size_t TileBodyPages::metadata_bytes() const noexcept
{
    return charged_ + deferred_metadata_bytes_ + source_context_bytes_ +
           source_contexts_.size() * source_context_node_bytes;
}
std::size_t TileBodyPages::source_only_metadata_bytes(std::size_t exported_context_bytes) const noexcept
{
    if (exported_context_bytes > source_context_bytes_)
        std::terminate();
    return metadata_bytes() - exported_context_bytes;
}
std::size_t TileBodyPages::page_credit_metadata_bytes() const noexcept
{
    return active_page_credits_ * credit_overhead;
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
