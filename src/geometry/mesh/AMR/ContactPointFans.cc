#include "ContactPointFans.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
ContactFanCatalog::ContactFanCatalog(std::shared_ptr<Budget> arena) : points(arena), allocation_(std::move(arena)) {}
std::shared_ptr<ContactFanCatalog> ContactFanCatalog::create(std::shared_ptr<Budget> arena)
{
    ContactGroupAllocation admission(arena);
    admission.add(sizeof(ContactFanCatalog) + 128);
    auto result = geometry_detail::domain_shared<geometry_detail::AllocationDomain::ContactHalo, ContactFanCatalog>(
        std::move(arena));
    result->allocation_.transfer_from(admission, sizeof(ContactFanCatalog) + 128);
    return result;
}
std::shared_ptr<ContactFanCatalog> ContactFanCatalog::create(
    const ContactFanCatalog& source, std::shared_ptr<Budget> arena, const PointFactDirectory::PointPool* target_pool)
{
    ContactGroupAllocation admission(arena);
    admission.add(sizeof(ContactFanCatalog) + 128);
    auto result = geometry_detail::domain_shared<geometry_detail::AllocationDomain::ContactHalo, ContactFanCatalog>(
        source, std::move(arena), target_pool);
    result->allocation_.transfer_from(admission, sizeof(ContactFanCatalog) + 128);
    return result;
}
ContactFanCatalog::ContactFanCatalog(
    const ContactFanCatalog& source, std::shared_ptr<Budget> arena, const PointFactDirectory::PointPool* target_pool)
    : points(source.points, arena, target_pool, false), allocation_(std::move(arena))
{
    allocation_.reserve(bindings_, source.bindings_.size());
    allocation_.reserve(free_, source.free_.size());
    allocation_.reserve(buckets_, source.buckets_.size());
    bindings_ = source.bindings_;
    free_ = source.free_;
    buckets_ = source.buckets_;
    retained_ = source.retained_;
    for (auto& entry : bindings_)
        if (entry.refs || entry.retired)
        {
            entry.refs = 0;
            entry.retired = true;
        }
}
const ContactFanCatalog::Binding& ContactFanCatalog::checked(ContactBindingId id) const
{
    if (id.slot >= bindings_.size() || bindings_[id.slot].generation != id.generation ||
        (!bindings_[id.slot].refs && !bindings_[id.slot].retired))
        throw std::out_of_range("Stale contact row binding.");
    return bindings_[id.slot];
}
ContactFanCatalog::Binding& ContactFanCatalog::checked(ContactBindingId id)
{
    return const_cast<Binding&>(std::as_const(*this).checked(id));
}
ContactBindingId ContactFanCatalog::bind(std::shared_ptr<const ContactIncidenceRow> row, std::size_t record)
{
    if (!row)
        throw std::invalid_argument("Missing immutable contact row.");
    const auto hash = binding_hash(row.get(), row->incidence_generation, record);
    if (!buckets_.empty())
    {
        const auto mask = buckets_.size() - 1;
        auto bucket = static_cast<std::size_t>(hash) & mask;
        for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
        {
            const auto slot = buckets_[bucket];
            if (slot == empty_binding)
                break;
            if (slot != retired_bucket)
            {
                const auto& entry = bindings_[slot];
                if (entry.value.row.get() == row.get() && entry.value.record == record &&
                    entry.value.incidence_generation == row->incidence_generation)
                {
                    const ContactBindingId id{slot, entry.generation};
                    pin(id);
                    return id;
                }
            }
            bucket = (bucket + 1) & mask;
        }
    }
    ensure_binding_lookup();
    if (free_.empty())
        allocation_.append(bindings_);
    const auto slot = free_.empty() ? bindings_.size() : free_.back();
    if (free_.empty())
        bindings_.push_back({});
    else
        free_.pop_back();
    auto& entry = bindings_[slot];
    entry.value = {std::move(row), 0, record};
    entry.value.incidence_generation = entry.value.row->incidence_generation;
    entry.refs = 1;
    entry.retired = false;
    const auto mask = buckets_.size() - 1;
    auto bucket = static_cast<std::size_t>(hash) & mask;
    while (buckets_[bucket] != empty_binding && buckets_[bucket] != retired_bucket)
        bucket = (bucket + 1) & mask;
    buckets_[bucket] = slot;
    ++retained_;
    return {slot, entry.generation};
}
std::uint64_t ContactFanCatalog::binding_hash(
    const ContactIncidenceRow* row, std::uint64_t generation, std::size_t record) noexcept
{
    const auto mix = [](std::uint64_t value)
    {
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    };
    return mix(reinterpret_cast<std::uintptr_t>(row)) ^ mix(generation) ^ mix(record);
}
void ContactFanCatalog::ensure_binding_lookup()
{
    if (!buckets_.empty() && retained_ + 1 <= buckets_.size() - buckets_.size() / 4)
        return;
    const auto old = buckets_.capacity();
    if (old > std::numeric_limits<std::size_t>::max() / 2)
        throw std::invalid_argument("Contact binding lookup capacity overflows.");
    const auto capacity = old ? 2 * old : 8;
    ContactGroupAllocation scratch(allocation_.keeper());
    ContactVector<std::size_t> replacement;
    scratch.reserve(replacement, capacity);
    replacement.assign(capacity, empty_binding);
    for (std::size_t slot = 0; slot < bindings_.size(); ++slot)
        if (bindings_[slot].refs || bindings_[slot].retired)
        {
            const auto& value = bindings_[slot].value;
            auto bucket =
                static_cast<std::size_t>(binding_hash(value.row.get(), value.incidence_generation, value.record)) &
                (capacity - 1);
            while (replacement[bucket] != empty_binding)
                bucket = (bucket + 1) & (capacity - 1);
            replacement[bucket] = slot;
        }
    allocation_.transfer_from(scratch, capacity * sizeof(std::size_t));
    buckets_.swap(replacement);
    ContactVector<std::size_t>().swap(replacement);
    allocation_.release(old * sizeof(std::size_t));
}
void ContactFanCatalog::pin(ContactBindingId id)
{
    auto& entry = checked(id);
    if (entry.refs == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Contact binding reference count overflows.");
    ++entry.refs;
    entry.retired = false;
}
void ContactFanCatalog::release(ContactBindingId id)
{
    auto& entry = checked(id);
    if (!entry.refs)
        throw std::invalid_argument("Contact binding already retired.");
    entry.retired = --entry.refs == 0;
}
const ContactRowBinding& ContactFanCatalog::binding(ContactBindingId id) const
{
    const auto& value = checked(id).value;
    if (!value.row || value.row->incidence_generation != value.incidence_generation)
        throw std::invalid_argument("Contact row binding generation changed.");
    return value;
}
ContactBindingId ContactFanCatalog::pinned_binding(std::size_t slot) const
{
    if (slot >= bindings_.size() || !bindings_[slot].refs)
        throw std::out_of_range("Unpinned contact binding slot.");
    return {slot, bindings_[slot].generation};
}
ContactBindingId ContactFanCatalog::staging_binding(std::size_t slot) const
{
    if (slot >= bindings_.size() || (!bindings_[slot].refs && !bindings_[slot].retired))
        throw std::out_of_range("Unavailable contact binding staging slot.");
    return {slot, bindings_[slot].generation};
}
void ContactFanCatalog::reclaim(const std::function<void(std::size_t)>& retire_fact)
{
    std::size_t n = 0;
    for (const auto& entry : bindings_)
        n += entry.retired;
    if (n > std::numeric_limits<std::size_t>::max() - free_.size())
        throw std::invalid_argument("Contact binding free list overflows.");
    allocation_.reserve(free_, free_.size() + n);
    for (std::size_t i = 0; i < bindings_.size(); ++i)
    {
        auto& entry = bindings_[i];
        if (!entry.retired)
            continue;
        const auto mask = buckets_.size() - 1;
        auto bucket = static_cast<std::size_t>(
                          binding_hash(entry.value.row.get(), entry.value.incidence_generation, entry.value.record)) &
                      mask;
        while (buckets_[bucket] != i)
            bucket = (bucket + 1) & mask;
        buckets_[bucket] = retired_bucket;
        --retained_;
        entry.value = {};
        entry.retired = false;
        if (entry.generation != std::numeric_limits<std::uint64_t>::max())
        {
            ++entry.generation;
            free_.push_back(i);
        }
    }
    points.reclaim(retire_fact);
}
bool ContactFanBlock::needs_wide(std::uint64_t value) noexcept
{
    return value > std::numeric_limits<std::uint32_t>::max();
}
std::shared_ptr<const ContactFanBlock> ContactFanBlock::build(
    std::span<const Input> input, ContactGroupAllocation& allocation, bool detached_charge)
{
    ContactGroupAllocation staged(allocation.keeper());
    std::size_t locals = 0, remotes = 0;
    bool wide = needs_wide(input.size());
    for (const auto& group : input)
    {
        if (group.local.size() > std::numeric_limits<std::size_t>::max() - locals ||
            group.remote.size() > std::numeric_limits<std::size_t>::max() - remotes)
            throw std::invalid_argument("Contact fan extent overflows.");
        locals += group.local.size();
        remotes += group.remote.size();
        for (const auto& value : group.local)
            wide = wide || needs_wide(value.binding.slot) || needs_wide(value.point);
        for (const auto& value : group.remote)
            wide = wide || needs_wide(value.record) || needs_wide(value.physical.slot);
    }
    wide = wide || needs_wide(locals) || needs_wide(remotes);
    staged.add(sizeof(ContactFanBlock) + 128);
    auto result = contact_shared_object<ContactFanBlock>(
        [&](ContactFanBlock* storage) { ::new (storage) ContactFanBlock(allocation.keeper()); });
    if (wide)
        result->data_.emplace<Arrays<std::uint64_t>>();
    std::visit(
        [&](auto& arrays)
        {
            using UInt = decltype(typename std::decay_t<decltype(arrays.slices)>::value_type{}.local);
            staged.reserve(arrays.local, locals);
            staged.reserve(arrays.remote, remotes);
            staged.reserve(arrays.slices, input.size());
            for (const auto& group : input)
            {
                arrays.slices.push_back({static_cast<UInt>(arrays.local.size()), static_cast<UInt>(group.local.size()),
                    static_cast<UInt>(arrays.remote.size()), static_cast<UInt>(group.remote.size())});
                for (const auto& value : group.local)
                    arrays.local.push_back({static_cast<UInt>(value.binding.slot), static_cast<UInt>(value.point)});
                for (const auto& value : group.remote)
                    arrays.remote.push_back({static_cast<UInt>(value.record), static_cast<UInt>(value.physical.slot)});
            }
        },
        result->data_);
    (detached_charge ? result->admission_ : allocation).transfer_from(staged, staged.bytes());
    return result;
}
bool ContactFanBlock::wide() const noexcept
{
    return data_.index() == 1;
}
std::size_t ContactFanBlock::groups() const noexcept
{
    return std::visit([](const auto& a) { return a.slices.size(); }, data_);
}
std::size_t ContactFanBlock::local_count(std::size_t group) const
{
    return std::visit([&](const auto& a) { return std::size_t(a.slices.at(group).locals); }, data_);
}
std::size_t ContactFanBlock::remote_count(std::size_t group) const
{
    return std::visit([&](const auto& a) { return std::size_t(a.slices.at(group).remotes); }, data_);
}
std::pair<std::size_t, std::size_t> ContactFanBlock::local(std::size_t group, std::size_t occurrence) const
{
    return std::visit(
        [&](const auto& a) -> std::pair<std::size_t, std::size_t>
        {
            const auto& slice = a.slices.at(group);
            if (occurrence >= slice.locals)
                throw std::out_of_range("Contact local fan index out of range.");
            const auto& value = a.local[std::size_t(slice.local) + occurrence];
            return {value.binding, value.point};
        },
        data_);
}
std::pair<std::size_t, std::size_t> ContactFanBlock::remote(std::size_t group, std::size_t occurrence) const
{
    return std::visit(
        [&](const auto& a) -> std::pair<std::size_t, std::size_t>
        {
            const auto& slice = a.slices.at(group);
            if (occurrence >= slice.remotes)
                throw std::out_of_range("Contact remote fan index out of range.");
            const auto& value = a.remote[std::size_t(slice.remote) + occurrence];
            return {value.record, value.physical};
        },
        data_);
}
std::size_t ContactFanBlock::storage_bytes() const noexcept
{
    return std::visit(
        [](const auto& a)
        {
            return sizeof(ContactFanBlock) + 128 + a.local.capacity() * sizeof(a.local[0]) +
                   a.remote.capacity() * sizeof(a.remote[0]) + a.slices.capacity() * sizeof(a.slices[0]);
        },
        data_);
}
ContactFanOwner::ContactFanOwner(std::shared_ptr<ContactFanCatalog> catalog)
    : allocation_(catalog->arena()), body_admission_(catalog->arena()), catalog_(std::move(catalog))
{
}
std::shared_ptr<ContactFanOwner> ContactFanOwner::create(std::shared_ptr<ContactFanCatalog> catalog)
{
    ContactGroupAllocation admission(catalog->arena());
    admission.add(sizeof(ContactFanOwner) + 128);
    auto result = contact_shared_object<ContactFanOwner>(
        [&](ContactFanOwner* storage) { ::new (storage) ContactFanOwner(std::move(catalog)); });
    result->allocation_.transfer_from(admission, sizeof(ContactFanOwner) + 128);
    return result;
}
std::shared_ptr<const ContactFanOwner> ContactFanOwner::build(
    std::shared_ptr<ContactFanCatalog> catalog, std::span<const ContactFanBlock::Input> input)
{
    auto result = create(std::move(catalog));
    for (const auto& group : input)
    {
        for (const auto& local : group.local)
            (void) result->catalog_->binding(local.binding).row->points.at(local.point);
        for (const auto& remote : group.remote)
            (void) result->catalog_->points.get(remote.physical);
    }
    result->block_ = ContactFanBlock::build(input, result->body_admission_);
    result->allocation_.reserve(result->views_, input.size());
    result->views_.assign(input.size(), 0);
    for (std::size_t g = 0; g < input.size(); ++g)
        result->pin_slice(g); // one staging pin per slice
    return result;
}
std::shared_ptr<const ContactFanOwner> ContactFanOwner::fork(
    const ContactFanOwner& source, std::shared_ptr<ContactFanCatalog> catalog)
{
    auto result = create(std::move(catalog));
    result->body_admission_.add(source.block_->storage_bytes());
    result->block_ = source.block_;
    result->allocation_.reserve(result->views_, source.block_->groups());
    result->views_.assign(source.block_->groups(), 0);
    for (std::size_t g = 0; g < source.block_->groups(); ++g)
    {
        // Dead slices can contain recycled slot numbers in an otherwise live
        // immutable block. They must never be reactivated by this fork.
        if (source.views_[g] == std::numeric_limits<std::size_t>::max())
            result->views_[g] = std::numeric_limits<std::size_t>::max();
        else if (source.views_[g])
            result->pin_slice(g);
    }
    return result;
}
void ContactFanOwner::pin_slice(std::size_t group) const
{
    auto& count = views_.at(group);
    constexpr auto retired = std::numeric_limits<std::size_t>::max();
    if (count == retired || (!count && !staging_))
        throw std::out_of_range("Retired contact fan slice.");
    if (count == retired - 1)
        throw std::invalid_argument("Contact fan view count overflows.");
    if (count)
    {
        ++count;
        return;
    }
    std::size_t locals = 0, remotes = 0;
    try
    {
        for (std::size_t i = 0; i < block_->local_count(group); ++i)
        {
            const auto [slot, point] = block_->local(group, i);
            const auto binding = catalog_->staging_binding(slot);
            (void) catalog_->binding(binding).row->points.at(point);
            catalog_->pin(binding);
            ++locals;
        }
        for (std::size_t i = 0; i < block_->remote_count(group); ++i)
        {
            const auto slot = block_->remote(group, i).second;
            catalog_->points.pin(catalog_->points.staging_reference(slot));
            ++remotes;
        }
    }
    catch (...)
    {
        for (std::size_t i = 0; i < locals; ++i)
            catalog_->release(catalog_->pinned_binding(block_->local(group, i).first));
        for (std::size_t i = 0; i < remotes; ++i)
            catalog_->points.release(catalog_->points.pinned_reference(block_->remote(group, i).second));
        throw;
    }
    count = 1;
}
void ContactFanOwner::release_slice(std::size_t group) const
{
    for (std::size_t i = 0; i < block_->local_count(group); ++i)
        catalog_->release(catalog_->pinned_binding(block_->local(group, i).first));
    for (std::size_t i = 0; i < block_->remote_count(group); ++i)
        catalog_->points.release(catalog_->points.pinned_reference(block_->remote(group, i).second));
}
void ContactFanOwner::unpin_slice(std::size_t group) const
{
    auto& count = views_.at(group);
    if (!count || count == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Contact fan slice is not pinned.");
    if (--count == 0)
    {
        release_slice(group);
        count = std::numeric_limits<std::size_t>::max();
    }
}
void ContactFanOwner::finish_staging() const
{
    if (!staging_)
        return;
    for (std::size_t g = 0; g < views_.size(); ++g)
        if (views_[g] && views_[g] != std::numeric_limits<std::size_t>::max())
            unpin_slice(g);
        else
            views_[g] = std::numeric_limits<std::size_t>::max();
    staging_ = false;
}
bool ContactFanOwner::has_live_slices() const noexcept
{
    for (const auto count : views_)
        if (count && count != std::numeric_limits<std::size_t>::max())
            return true;
    return false;
}
ContactFanOwner::~ContactFanOwner()
{
    for (std::size_t g = 0; g < views_.size(); ++g)
        if (views_[g] && views_[g] != std::numeric_limits<std::size_t>::max())
            release_slice(g);
}
ContactFanView::ContactFanView(std::shared_ptr<const ContactFanOwner> value, std::size_t slice)
    : owner_(std::move(value)), group_(slice)
{
    if (owner_)
        owner_->pin_slice(group_);
}
ContactFanView::ContactFanView(const ContactFanView& other) : ContactFanView(other.owner_, other.group_) {}
ContactFanView::ContactFanView(ContactFanView&& other) noexcept : owner_(std::move(other.owner_)), group_(other.group_)
{
}
ContactFanView& ContactFanView::operator=(ContactFanView other) noexcept
{
    owner_.swap(other.owner_);
    std::swap(group_, other.group_);
    return *this;
}
ContactFanView::~ContactFanView()
{
    if (owner_)
        owner_->unpin_slice(group_);
}
const ContactRowBinding& ContactFanOwner::binding(std::size_t slot) const
{
    return catalog_->binding(catalog_->pinned_binding(slot));
}
ContactPointIdentity ContactFanOwner::physical(std::size_t slot) const
{
    return catalog_->points.get(catalog_->points.pinned_reference(slot));
}
ContactFanView::Local ContactFanView::local(std::size_t i) const
{
    if (!owner_)
        throw std::out_of_range("Empty contact local fan.");
    const auto [binding, point] = owner_->block().local(group_, i);
    return {owner_->catalog_->pinned_binding(binding), owner_->binding(binding), point};
}
ContactFanView::Remote ContactFanView::remote(std::size_t i) const
{
    if (!owner_)
        throw std::out_of_range("Empty contact remote fan.");
    const auto [record, physical] = owner_->block().remote(group_, i);
    return {record, owner_->physical(physical)};
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
