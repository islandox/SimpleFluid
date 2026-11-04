#pragma once
#include <algorithm>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
template<class E, std::size_t C>
template<class T>
void CompactTileIndex<E, C>::Array<T>::DestroyArray::operator()(T* values) const noexcept
{
    std::destroy_n(values, count);
    std::allocator<T>{}.deallocate(values, count);
}
template<class E, std::size_t C> template<class T> CompactTileIndex<E, C>::Array<T>::~Array()
{
    clear();
}
template<class E, std::size_t C> template<class T> void CompactTileIndex<E, C>::Array<T>::clear() noexcept
{
    data.reset();
    budget_->used -= bytes();
    capacity = count = 0;
}
template<class E, std::size_t C> template<class T> void CompactTileIndex<E, C>::Array<T>::reserve(std::size_t n)
{
    if (n <= capacity)
        return;
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::invalid_argument("Tile index capacity overflows.");
    const auto charge = n * sizeof(T);
    budget_->add(charge);
    std::unique_ptr<T[], DestroyArray> next(nullptr, DestroyArray{n});
    try
    {
        // allocator::allocate requests exactly n*sizeof(T); new T[n] can add
        // an ABI array cookie for nontrivially destructible directory entries.
        auto* values = std::allocator<T>{}.allocate(n);
        try
        {
            std::uninitialized_value_construct_n(values, n);
        }
        catch (...)
        {
            // uninitialized_value_construct_n destroys its constructed prefix.
            std::allocator<T>{}.deallocate(values, n);
            throw;
        }
        next.reset(values);
    }
    catch (...)
    {
        budget_->used -= charge;
        throw;
    }
    static_assert(std::is_nothrow_move_assignable_v<T>);
    for (std::size_t i = 0; i < count; ++i)
        next[i] = std::move(data[i]);
    data.swap(next);
    next.reset();
    budget_->used -= bytes();
    capacity = n;
}
template<class E, std::size_t C> template<class T> void CompactTileIndex<E, C>::Array<T>::append(T value)
{
    if (count == capacity)
    {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Tile index capacity overflows.");
        reserve(capacity ? 2 * capacity : 1);
    }
    data[count++] = std::move(value);
}
template<class E, std::size_t C> template<class T> void CompactTileIndex<E, C>::Array<T>::swap(Array& other) noexcept
{
    data.swap(other.data);
    std::swap(count, other.count);
    std::swap(capacity, other.capacity);
}
template<class E, std::size_t C> CompactTileIndex<E, C>::Page::~Page()
{
    for (std::size_t i = 0; i < page_size; ++i)
        if (occupied & (std::uint16_t(1) << i))
            std::destroy_at(std::addressof(cells[i].value));
}
template<class E, std::size_t C>
CompactTileIndex<E, C>::CompactTileIndex(Budget& b, std::shared_ptr<Budget> keeper)
    : budget_(&b), keeper_(std::move(keeper)), pages_(b), compact_(b), wide_(b)
{
}
template<class E, std::size_t C> CompactTileIndex<E, C>::~CompactTileIndex()
{
    // Destroy entries before returning their page admission. Directory storage
    // is released by its own Array after this body.
    for (std::size_t i = 0; i < pages_.count; ++i)
        pages_[i].reset();
    budget_->used -= page_bytes_;
}
template<class E, std::size_t C> std::size_t CompactTileIndex<E, C>::size() const noexcept
{
    return wide_mode_ ? wide_.count : compact_.count;
}
template<class E, std::size_t C> auto CompactTileIndex<E, C>::key_at(std::size_t i) const noexcept -> Key
{
    return wide_mode_ ? wide_[i].key : Key{compact_[i].support, compact_[i].emission};
}
template<class E, std::size_t C> std::size_t CompactTileIndex<E, C>::slot_at(std::size_t i) const noexcept
{
    return wide_mode_ ? wide_[i].slot : compact_[i].slot;
}
template<class E, std::size_t C> E& CompactTileIndex<E, C>::entry(std::size_t slot) noexcept
{
    return pages_[slot / page_size]->cells[slot % page_size].value;
}
template<class E, std::size_t C> const E& CompactTileIndex<E, C>::entry(std::size_t slot) const noexcept
{
    return pages_[slot / page_size]->cells[slot % page_size].value;
}
template<class E, std::size_t C> E& CompactTileIndex<E, C>::value_at(std::size_t i) noexcept
{
    return entry(slot_at(i));
}
template<class E, std::size_t C> const E& CompactTileIndex<E, C>::value_at(std::size_t i) const noexcept
{
    return entry(slot_at(i));
}
template<class E, std::size_t C> std::size_t CompactTileIndex<E, C>::lower_bound(Key key) const noexcept
{
    std::size_t first = 0, last = size();
    while (first != last)
    {
        const auto middle = first + (last - first) / 2;
        if (key_at(middle) < key)
            first = middle + 1;
        else
            last = middle;
    }
    return first;
}
template<class E, std::size_t C> E* CompactTileIndex<E, C>::find(Key key) noexcept
{
    const auto i = lower_bound(key);
    return i != size() && key_at(i) == key ? &value_at(i) : nullptr;
}
template<class E, std::size_t C> const E* CompactTileIndex<E, C>::find(Key key) const noexcept
{
    const auto i = lower_bound(key);
    return i != size() && key_at(i) == key ? &value_at(i) : nullptr;
}
template<class E, std::size_t C> E& CompactTileIndex<E, C>::at(Key key)
{
    auto* found = find(key);
    if (!found)
        throw std::out_of_range("Missing resident tile ordinal.");
    return *found;
}
template<class E, std::size_t C> std::size_t CompactTileIndex<E, C>::metadata_bytes() const noexcept
{
    return page_bytes_ + pages_.bytes() + compact_.bytes() + wide_.bytes();
}
template<class E, std::size_t C> std::size_t CompactTileIndex<E, C>::insert_value(E value)
{
    if (free_ == absent)
    {
        while (first_empty_page_ != pages_.count && pages_[first_empty_page_])
            ++first_empty_page_;
        if (pages_.count > (absent - page_size) / page_size)
            throw std::invalid_argument("Tile metadata slot count overflows.");
        if (first_empty_page_ == pages_.count && pages_.count == pages_.capacity)
            pages_.reserve(pages_.capacity ? 2 * pages_.capacity : 1);
        budget_->add(sizeof(Page));
        std::unique_ptr<Page> page;
        try
        {
            page = std::make_unique<Page>();
        }
        catch (...)
        {
            budget_->used -= sizeof(Page);
            throw;
        }
        const auto first = first_empty_page_ * page_size;
        for (std::size_t i = 0; i < page_size; ++i)
            page->cells[i].next = i + 1 == page_size ? absent : first + i + 1;
        if (first_empty_page_ == pages_.count)
            pages_.append(std::move(page));
        else
            pages_[first_empty_page_] = std::move(page);
        page_bytes_ += sizeof(Page);
        free_ = first;
    }
    const auto slot = free_;
    auto& page = *pages_[slot / page_size];
    auto& cell = page.cells[slot % page_size];
    free_ = cell.next;
    std::construct_at(std::addressof(cell.value), std::move(value));
    page.occupied |= std::uint16_t(1) << (slot % page_size);
    return slot;
}
template<class E, std::size_t C> void CompactTileIndex<E, C>::reap_empty_pages() noexcept
{
    // Run only after old readers and staging have retired. The directory is
    // packed without moving values. First record each page's new ordinal while
    // old slots remain readable, then rewrite descriptor slots, then move page
    // owners. Thus later small batches never scan a historical high-water span.
    free_ = absent;
    std::size_t live_pages = 0;
    for (std::size_t i = 0; i < pages_.count; ++i)
    {
        auto& page = pages_[i];
        if (page && !page->occupied)
        {
            page.reset();
            page_bytes_ -= sizeof(Page);
            budget_->used -= sizeof(Page);
        }
        if (page)
            page->directory_slot = live_pages++;
    }
    for (std::size_t i = 0; i < size(); ++i)
    {
        const auto old = slot_at(i);
        const auto slot = pages_[old / page_size]->directory_slot * page_size + old % page_size;
        if (wide_mode_)
            wide_[i].slot = slot;
        else
            compact_[i].slot = static_cast<std::uint32_t>(slot);
    }
    for (std::size_t i = 0; i < pages_.count; ++i)
        if (pages_[i] && pages_[i]->directory_slot != i)
            pages_[pages_[i]->directory_slot] = std::move(pages_[i]);
    pages_.count = live_pages;
    first_empty_page_ = live_pages;
    for (std::size_t i = live_pages; i != 0;)
    {
        --i;
        auto& page = pages_[i];
        for (std::size_t j = page_size; j != 0;)
        {
            --j;
            if (!(page->occupied & (std::uint16_t(1) << j)))
            {
                page->cells[j].next = free_;
                free_ = i * page_size + j;
            }
        }
    }
}
template<class E, std::size_t C> void CompactTileIndex<E, C>::retire_value(std::size_t slot) noexcept
{
    auto& page = *pages_[slot / page_size];
    auto& cell = page.cells[slot % page_size];
    std::destroy_at(std::addressof(cell.value));
    page.occupied &= ~(std::uint16_t(1) << (slot % page_size));
    cell.next = free_;
    free_ = slot;
}
template<class E, std::size_t C>
CompactTileIndex<E, C>::Batch::Batch(CompactTileIndex& owner)
    : owner_(&owner), retired_(*owner.budget_), staged_(*owner.budget_)
{
    if (owner.batch_live_)
        throw std::invalid_argument("Concurrent resident tile metadata batch.");
    owner.batch_live_ = true;
}
template<class E, std::size_t C> CompactTileIndex<E, C>::Batch::~Batch()
{
    if (!published_)
    {
        for (std::size_t i = 0; i < staged_.count; ++i)
            owner_->retire_value(staged_[i].slot);
        owner_->reap_empty_pages();
    }
    owner_->batch_live_ = false;
}
template<class E, std::size_t C> void CompactTileIndex<E, C>::Batch::replace_support(std::size_t support)
{
    if (published_)
        throw std::invalid_argument("Resident tile metadata batch already published.");
    retired_.append(support);
}
template<class E, std::size_t C> void CompactTileIndex<E, C>::Batch::insert(Key key, E value)
{
    if (published_)
        throw std::invalid_argument("Resident tile metadata batch already published.");
    // Reserve the descriptor before allocating its value so every live staged
    // slot is tracked for unwinding, including after allocation failure.
    if (staged_.count == staged_.capacity)
        staged_.reserve(staged_.capacity ? 2 * staged_.capacity : 1);
    const auto slot = owner_->insert_value(std::move(value));
    staged_.append({key, slot});
}
template<class E, std::size_t C> void CompactTileIndex<E, C>::Batch::publish(std::size_t tile_limit)
{
    if (published_)
        throw std::invalid_argument("Resident tile metadata batch already published.");
    auto& owner = *owner_;
    owner.merge_comparisons_ = 0;
    if (!retired_.count && !staged_.count)
    {
        published_ = true;
        return;
    }
    const auto compare = [&](const WideDescriptor& a, const WideDescriptor& b)
    {
        ++owner.merge_comparisons_;
        return a.key < b.key;
    };
    if (staged_.count > 1)
        std::sort(staged_.data.get(), staged_.data.get() + staged_.count, compare);
    if (retired_.count > 1)
        std::sort(retired_.data.get(), retired_.data.get() + retired_.count,
            [&](auto a, auto b)
            {
                ++owner.merge_comparisons_;
                return a < b;
            });
    for (std::size_t i = 1; i < retired_.count; ++i)
        if (retired_[i - 1] == retired_[i])
            throw std::invalid_argument("Duplicate delivered resident support.");
    for (std::size_t i = 1; i < staged_.count; ++i)
        if (staged_[i - 1].key == staged_[i].key)
            throw std::invalid_argument("Duplicate resident tile ordinal.");
    std::size_t survivors = 0, group = 0;
    bool wide = false;
    const auto retired = [&](Key key, std::size_t& cursor)
    {
        while (cursor != retired_.count && retired_[cursor] < key.first)
        {
            ++owner.merge_comparisons_;
            ++cursor;
        }
        ++owner.merge_comparisons_;
        return cursor != retired_.count && retired_[cursor] == key.first;
    };
    const auto needs_wide = [](Key key, std::size_t slot)
    { return key.first > std::numeric_limits<std::uint32_t>::max() || slot > C; };
    for (std::size_t i = 0; i < owner.size(); ++i)
        if (!retired(owner.key_at(i), group))
        {
            ++survivors;
            wide |= needs_wide(owner.key_at(i), owner.slot_at(i));
        }
    if (staged_.count > std::numeric_limits<std::size_t>::max() - survivors || survivors + staged_.count > tile_limit)
        throw std::invalid_argument("Resident tile budget exhausted.");
    for (std::size_t i = 0; i < staged_.count; ++i)
        wide |= needs_wide(staged_[i].key, staged_[i].slot);
    const auto count = survivors + staged_.count;
    Array<CompactDescriptor> next_compact(*owner.budget_);
    Array<WideDescriptor> next_wide(*owner.budget_);
    if (wide)
        next_wide.reserve(count);
    else
        next_compact.reserve(count);
    const auto emit = [&](Key key, std::size_t slot)
    {
        if (wide)
            next_wide.append({key, slot});
        else
            next_compact.append({key.second, static_cast<std::uint32_t>(key.first), static_cast<std::uint32_t>(slot)});
    };
    std::size_t old = 0, fresh = 0;
    group = 0;
    while (old != owner.size() || fresh != staged_.count)
    {
        if (old != owner.size() && retired(owner.key_at(old), group))
        {
            ++old;
            continue;
        }
        if (old == owner.size())
        {
            emit(staged_[fresh].key, staged_[fresh].slot);
            ++fresh;
        }
        else if (fresh == staged_.count)
        {
            emit(owner.key_at(old), owner.slot_at(old));
            ++old;
        }
        else
        {
            ++owner.merge_comparisons_;
            const auto old_key = owner.key_at(old);
            if (old_key == staged_[fresh].key)
                throw std::invalid_argument("Duplicate resident tile ordinal.");
            if (old_key < staged_[fresh].key)
            {
                emit(old_key, owner.slot_at(old));
                ++old;
            }
            else
            {
                emit(staged_[fresh].key, staged_[fresh].slot);
                ++fresh;
            }
        }
    }
    // All throwing work is complete. Keep the old descriptor allocation alive
    // through retirement; no erased slot can be reused while it is readable.
    owner.compact_.swap(next_compact);
    owner.wide_.swap(next_wide);
    const auto old_wide = owner.wide_mode_;
    owner.wide_mode_ = wide;
    group = 0;
    const auto old_count = old_wide ? next_wide.count : next_compact.count;
    for (std::size_t i = 0; i < old_count; ++i)
    {
        const auto key = old_wide ? next_wide[i].key : Key{next_compact[i].support, next_compact[i].emission};
        if (retired(key, group))
            owner.retire_value(old_wide ? next_wide[i].slot : next_compact[i].slot);
    }
    published_ = true;
    // The old publication has no remaining readers. Release its allocation
    // before compacting directory ordinals; stable value addresses do not move.
    next_compact.clear();
    next_wide.clear();
    owner.reap_empty_pages();
    staged_.clear();
    retired_.clear();
}
template<class E, std::size_t C>
template<class Transform>
void CompactTileIndex<E, C>::clone_from(const CompactTileIndex& source, Transform&& transform)
{
    if (size() || pages_.count || batch_live_)
        throw std::invalid_argument("Resident tile metadata clone target is not empty.");
    // Source slot holes need not be copied. Live values are packed in the same
    // canonical order, and each private descriptor is constructed exactly once.
    const auto count = source.size();
    bool wide = count && count - 1 > C;
    for (std::size_t i = 0; i < count; ++i)
        wide |= source.key_at(i).first > std::numeric_limits<std::uint32_t>::max();
    if (wide)
        wide_.reserve(count);
    else
        compact_.reserve(count);
    pages_.reserve(count / page_size + (count % page_size != 0));
    wide_mode_ = wide;
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto key = source.key_at(i);
        const auto slot = insert_value(transform(source.value_at(i)));
        if (wide)
            wide_.append({key, slot});
        else
            compact_.append({key.second, static_cast<std::uint32_t>(key.first), static_cast<std::uint32_t>(slot)});
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
