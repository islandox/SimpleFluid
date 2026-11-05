#pragma once
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
template<class H>
std::shared_ptr<typename ContactPointGroups<H>::Readers> ContactPointGroups<H>::make_readers(
    const std::shared_ptr<Budget>& arena)
{
    constexpr auto bytes = sizeof(Readers) + 128;
    arena->add(bytes);
    Readers* readers;
    try
    {
        readers = new Readers;
    }
    catch (...)
    {
        arena->used -= bytes;
        throw;
    }
    return std::shared_ptr<Readers>(readers,
        [arena](Readers* value)
        {
            delete value;
            arena->used -= bytes;
        });
}
template<class H>
ContactPointGroups<H>::ContactPointGroups(std::shared_ptr<Budget> arena, H hash)
    : allocation_(arena), domains_(*arena, arena), catalog_(ContactFanCatalog::create(arena)),
      readers_(make_readers(arena)), hasher_(std::move(hash)), fan_owners_(*arena)
{
}
template<class H>
ContactPointGroups<H>::ContactPointGroups(
    const ContactPointGroups& source, std::shared_ptr<Budget> arena, const PointFactDirectory::PointPool* target_pool)
    : allocation_(arena), domains_(source.domains_, *arena, arena),
      catalog_(ContactFanCatalog::create(*source.catalog_, arena, target_pool)), readers_(make_readers(arena)),
      hasher_(source.hasher_), fan_owners_(*arena)
{
    allocation_.reserve(groups_, source.groups_.size());
    allocation_.reserve(free_, source.free_.size());
    allocation_.reserve(buckets_, source.buckets_.size());
    free_ = source.free_;
    buckets_ = source.buckets_;
    PhysicalOwnerIndex<std::shared_ptr<const ContactFanOwner>> copies(*arena);
    for (const auto& old : source.groups_)
    {
        Group group;
        group.generation = old.generation;
        group.domain = old.domain;
        group.normalized = old.normalized;
        group.cover = old.cover;
        group.live = old.live;
        group.retired = old.retired;
        if (old.live || old.retired)
        {
            catalog_->points.pin(group.normalized);
            if (old.fan.owner())
            {
                std::shared_ptr<const ContactFanOwner> owner;
                if (const auto found = copies.find(old.fan.owner().get()))
                    owner = *found;
                if (!owner)
                {
                    owner = ContactFanOwner::fork(*old.fan.owner(), catalog_);
                    copies.insert(old.fan.owner().get(), owner);
                    fan_owners_.insert(owner.get(), owner);
                }
                group.fan = {std::move(owner), old.fan.group()};
            }
        }
        groups_.push_back(std::move(group));
    }
    live_ = source.live_;
    copies.visit([](const void*, const auto& owner) { owner->finish_staging(); });
    fork_owner_work_ = copies.probes();
    // Discard catalog entries retained solely by escaped source snapshots.
    catalog_->reclaim();
}
template<class H> ContactPointGroups<H>::~ContactPointGroups()
{
    for (auto& group : groups_)
        if (group.live || group.retired)
            catalog_->points.release(group.normalized);
    // Fan owners must drop their pins before the group's catalog owner dies.
    std::vector<Group>().swap(groups_);
}
template<class H>
const typename ContactPointGroups<H>::Group& ContactPointGroups<H>::checked(ContactGroupId id, bool retired) const
{
    if (id.slot >= groups_.size() || groups_[id.slot].generation != id.generation ||
        (!groups_[id.slot].live && !(retired && groups_[id.slot].retired)))
        throw std::out_of_range("Stale contact group ID.");
    return groups_[id.slot];
}
template<class H> typename ContactPointGroups<H>::Group& ContactPointGroups<H>::checked(ContactGroupId id, bool retired)
{
    return const_cast<Group&>(std::as_const(*this).checked(id, retired));
}
template<class H> ContactPointKeyView ContactPointGroups<H>::key(const Group& group) const
{
    return {domains_.get(group.domain), catalog_->points.get(group.normalized)};
}
template<class H> ContactPointKeyView ContactPointGroups<H>::key(ContactGroupId id) const
{
    return key(checked(id, true));
}
template<class H> const AdaptiveBytes& ContactPointGroups<H>::domain(ContactGroupId id) const
{
    return domains_.get(checked(id, true).domain);
}
template<class H> ContactPointIdentity ContactPointGroups<H>::normalized(ContactGroupId id) const
{
    return catalog_->points.get(checked(id, true).normalized);
}
template<class H> ContactPointRef ContactPointGroups<H>::normalized_reference(ContactGroupId id) const
{
    return checked(id, true).normalized;
}
template<class H> ContactFanView ContactPointGroups<H>::fan(ContactGroupId id) const
{
    return checked(id, true).fan;
}
template<class H> std::size_t ContactPointGroups<H>::cover(ContactGroupId id) const
{
    return checked(id, true).cover;
}
template<class H> void ContactPointGroups<H>::set_cover(ContactGroupId id, std::size_t cover)
{
    checked(id).cover = cover;
}
template<class H>
template<class Probe>
std::optional<ContactGroupId> ContactPointGroups<H>::find_probe(const Probe& probe) const
{
    if (buckets_.empty())
        return {};
    const auto mask = buckets_.size() - 1;
    auto bucket = static_cast<std::size_t>(hasher_(probe)) & mask;
    const ContactPointWireLess less;
    for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
    {
        const auto slot = buckets_[bucket];
        if (slot == empty)
            return {};
        if (slot != tombstone)
        {
            const auto& group = groups_[slot];
            const auto value = key(group);
            if (group.live && !less(value, probe) && !less(probe, value))
                return ContactGroupId{slot, group.generation};
        }
        bucket = (bucket + 1) & mask;
    }
    return {};
}
template<class H> std::optional<ContactGroupId> ContactPointGroups<H>::find(ContactPointKeyView value) const
{
    return find_probe(value);
}
template<class H> std::optional<ContactGroupId> ContactPointGroups<H>::find_encoded(std::span<const char> value) const
{
    return find_probe(value);
}
template<class H> void ContactPointGroups<H>::insert_bucket(std::size_t slot)
{
    const auto mask = buckets_.size() - 1;
    auto bucket = static_cast<std::size_t>(hasher_(key(groups_[slot]))) & mask;
    for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
    {
        if (buckets_[bucket] == empty || buckets_[bucket] == tombstone)
        {
            buckets_[bucket] = slot;
            return;
        }
        bucket = (bucket + 1) & mask;
    }
    throw std::invalid_argument("Contact group lookup has no empty slot.");
}
template<class H> void ContactPointGroups<H>::ensure_lookup()
{
    if (!buckets_.empty() && live_ + 1 <= buckets_.size() - buckets_.size() / 4)
        return;
    const auto old = buckets_.capacity();
    if (old > std::numeric_limits<std::size_t>::max() / 2)
        throw std::invalid_argument("Contact group lookup capacity overflows.");
    const auto capacity = old ? 2 * old : 8;
    // Rehash a distinct admitted array; a failure leaves the current table intact.
    ContactGroupAllocation scratch(allocation_.keeper());
    std::vector<std::size_t> replacement;
    scratch.reserve(replacement, capacity);
    replacement.assign(capacity, empty);
    for (std::size_t slot = 0; slot < groups_.size(); ++slot)
        if (groups_[slot].live)
        {
            auto bucket = static_cast<std::size_t>(hasher_(key(groups_[slot]))) & (capacity - 1);
            while (replacement[bucket] != empty)
                bucket = (bucket + 1) & (capacity - 1);
            replacement[bucket] = slot;
        }
    // The replacement's admission transfers to this owner without re-admission.
    allocation_.transfer_from(scratch, capacity * sizeof(std::size_t));
    buckets_.swap(replacement);
    std::vector<std::size_t>().swap(replacement);
    allocation_.release(old * sizeof(std::size_t));
}
template<class H>
ContactGroupId ContactPointGroups<H>::intern(const AdaptiveBytes& domain, const ContactPointIdentity& point,
    std::optional<ContactPointArray::Locator> locator, const PointFactDirectory::PointPool* target_pool)
{
    if (const auto old = find({domain, &point}))
        return *old;
    ensure_lookup();
    if (free_.empty())
        allocation_.append(groups_);
    const auto domain_slot = domains_.intern(domain);
    const auto normalized = locator ? catalog_->points.retain(*locator, target_pool) : catalog_->points.retain(point);
    // Catch a stale/wrong locator rather than silently changing the typed key.
    if (catalog_->points.get(normalized) != point)
    {
        catalog_->points.release(normalized);
        throw std::invalid_argument("Normalized contact point locator does not match key.");
    }
    const auto slot = free_.empty() ? groups_.size() : free_.back();
    if (free_.empty())
        groups_.push_back({});
    else
        free_.pop_back();
    auto& group = groups_[slot];
    group.domain = domain_slot;
    group.normalized = normalized;
    group.live = true;
    group.retired = false;
    try
    {
        insert_bucket(slot);
    }
    catch (...)
    {
        group.live = false;
        group.retired = true;
        throw;
    }
    ++live_;
    return {slot, group.generation};
}
template<class H> void ContactPointGroups<H>::replace_fans(std::span<const Edit> edits)
{
    replace_fans_generated(edits.size(), [&](std::size_t i) { return edits[i]; });
}
template<class H>
template<class EditAt>
void ContactPointGroups<H>::replace_fans_generated(std::size_t count, EditAt&& edit_at)
{
    ContactGroupAllocation scratch(allocation_.keeper());
    struct Prepared
    {
        std::vector<ContactLocalOccurrence> local;
        std::vector<ContactRemoteOccurrence> remote;
    };
    std::vector<Prepared> prepared;
    std::vector<ContactFanBlock::Input> inputs;
    scratch.reserve(prepared, count);
    scratch.reserve(inputs, count);
    std::vector<std::size_t> ordered;
    scratch.reserve(ordered, count);
    for (std::size_t i = 0; i < count; ++i)
        ordered.push_back(i);
    duplicate_work_ = 0;
    std::sort(ordered.begin(), ordered.end(),
        [&](std::size_t a, std::size_t b)
        {
            ++duplicate_work_;
            const auto first = edit_at(a).group, second = edit_at(b).group;
            return first.slot < second.slot || (first.slot == second.slot && first.generation < second.generation);
        });
    for (std::size_t i = 1; i < ordered.size(); ++i)
        if (edit_at(ordered[i - 1]).group == edit_at(ordered[i]).group)
            throw std::invalid_argument("Duplicate touched contact group.");
    for (std::size_t e = 0; e < count; ++e)
    {
        const auto edit = edit_at(e);
        const auto& group = checked(edit.group);
        prepared.push_back({});
        auto& next = prepared.back();
        if (edit.append_local.size() > std::numeric_limits<std::size_t>::max() - group.fan.local_count())
            throw std::invalid_argument("Contact local fan size overflows.");
        if (group.fan.local_count())
            scratch.reserve(next.local, group.fan.local_count() + edit.append_local.size());
        if (!edit.replace_remote)
            scratch.reserve(next.remote, group.fan.remote_count());
        for (std::size_t i = 0; i < group.fan.local_count(); ++i)
        {
            const auto [slot, point] = group.fan.owner()->block().local(group.fan.group(), i);
            const auto binding = catalog_->pinned_binding(slot);
            if (std::find(edit.remove_bindings.begin(), edit.remove_bindings.end(), binding) ==
                edit.remove_bindings.end())
                next.local.push_back({binding, point});
        }
        if (group.fan.local_count())
            next.local.insert(next.local.end(), edit.append_local.begin(), edit.append_local.end());
        const std::span<const ContactLocalOccurrence> local_values =
            group.fan.local_count() ? std::span<const ContactLocalOccurrence>(next.local) : edit.append_local;
        for (const auto& local : local_values)
        {
            const auto& binding = catalog_->binding(local.binding);
            if (binding.row->domain != domains_.get(group.domain) ||
                binding.row->points.at(local.point).normalized != catalog_->points.get(group.normalized))
                throw std::invalid_argument("Contact occurrence does not belong to its typed group.");
        }
        if (!edit.replace_remote)
            for (std::size_t i = 0; i < group.fan.remote_count(); ++i)
            {
                const auto [record, slot] = group.fan.owner()->block().remote(group.fan.group(), i);
                next.remote.push_back({record, catalog_->points.pinned_reference(slot)});
            }
        inputs.push_back(
            {local_values, edit.replace_remote ? edit.remote : std::span<const ContactRemoteOccurrence>(next.remote)});
    }
    if (!count)
        return;
    catalog_->points.seal_fallback();
    auto owner = ContactFanOwner::build(catalog_, inputs);
    fan_owners_.insert(owner.get(), owner);
    // No fallible work remains. Old blocks stay alive through all staging.
    for (std::size_t i = 0; i < count; ++i)
        checked(edit_at(i).group).fan = {owner, i};
    owner->finish_staging();
}
template<class H> void ContactPointGroups<H>::retire(ContactGroupId id)
{
    auto& group = checked(id);
    const auto mask = buckets_.size() - 1;
    auto bucket = static_cast<std::size_t>(hasher_(key(group))) & mask;
    for (std::size_t visited = 0; visited < buckets_.size(); ++visited)
    {
        if (buckets_[bucket] == id.slot)
        {
            buckets_[bucket] = tombstone;
            break;
        }
        bucket = (bucket + 1) & mask;
    }
    group.live = false;
    group.retired = true;
    --live_;
}
template<class H> void ContactPointGroups<H>::reclaim()
{
    if (readers_->active)
        return;
    std::size_t n = 0;
    for (const auto& group : groups_)
        n += group.retired;
    if (n > std::numeric_limits<std::size_t>::max() - free_.size())
        throw std::invalid_argument("Contact group free list overflows.");
    allocation_.reserve(free_, free_.size() + n);
    for (std::size_t i = 0; i < groups_.size(); ++i)
    {
        auto& group = groups_[i];
        if (!group.retired)
            continue;
        catalog_->points.release(group.normalized);
        group.fan = {};
        group.retired = false;
        group.cover = empty;
        if (group.generation != std::numeric_limits<std::uint64_t>::max())
        {
            ++group.generation;
            free_.push_back(i);
        }
    }
    ContactGroupAllocation scratch(allocation_.keeper());
    std::vector<const void*> dead;
    scratch.reserve(dead, fan_owners_.size());
    fan_owners_.visit(
        [&](const void* identity, const auto& owner)
        {
            if (!owner->has_live_slices())
                dead.push_back(identity);
        });
    for (const auto identity : dead)
        (void) fan_owners_.erase(identity);
    catalog_->reclaim();
}
template<class H>
typename ContactPointGroups<H>::OrderedIds ContactPointGroups<H>::ordered_ids(bool include_retired) const
{
    OrderedIds result(allocation_.keeper(), readers_);
    result.allocation_.reserve(result.ids_, include_retired ? groups_.size() : live_);
    for (std::size_t i = 0; i < groups_.size(); ++i)
        if (groups_[i].live || (include_retired && groups_[i].retired))
            result.ids_.push_back({i, groups_[i].generation});
    const ContactPointWireLess less;
    std::sort(result.ids_.begin(), result.ids_.end(),
        [&](const auto& first, const auto& second) { return less(key(first), key(second)); });
    return result;
}
template<class H>
typename ContactPointGroups<H>::OrderedIds ContactPointGroups<H>::ordered_ids(
    std::span<const ContactGroupId> selected) const
{
    OrderedIds result(allocation_.keeper(), readers_);
    result.allocation_.reserve(result.ids_, selected.size());
    for (const auto id : selected)
    {
        (void) checked(id, true);
        result.ids_.push_back(id);
    }
    const ContactPointWireLess less;
    std::sort(result.ids_.begin(), result.ids_.end(),
        [&](const auto& first, const auto& second) { return less(key(first), key(second)); });
    result.ids_.erase(std::unique(result.ids_.begin(), result.ids_.end()), result.ids_.end());
    return result;
}
template<class H> ContactStorageBreakdown ContactPointGroups<H>::storage_breakdown() const
{
    ContactStorageBreakdown result;
    result.group_metadata = allocation_.bytes() + sizeof(Readers) + 128;
    result.owner_index_metadata = fan_owners_.metadata_bytes();
    result.domain_metadata = domains_.private_metadata_bytes();
    result.domain_shared_body = domains_.shared_body_bytes();
    result.binding_metadata = catalog_->private_metadata_bytes();
    catalog_->points.contribute_breakdown(result);
    fan_owners_.visit(
        [&](const void*, const auto& owner)
        {
            result.fan_owner_metadata += owner->private_metadata_bytes();
            result.fan_shared_body += owner->shared_body_bytes();
        });
    return result;
}
template<class H> template<class Visitor> void ContactPointGroups<H>::visit_admissions(Visitor&& visitor) const
{
    visitor(ContactStorageAdmission{allocation_.identity(), allocation_.bytes(), nullptr, 0});
    visitor(ContactStorageAdmission{&readers_->admission_identity, sizeof(Readers) + 128, nullptr, 0});
    visitor(ContactStorageAdmission{fan_owners_.metadata_identity(), fan_owners_.metadata_bytes(), nullptr, 0});
    visitor(ContactStorageAdmission{domains_.metadata_identity(), domains_.metadata_bytes(), nullptr, 0});
    domains_.visit_page_admissions([&](const void* metadata, const void* body, std::size_t owner, std::size_t bytes)
        { visitor(ContactStorageAdmission{metadata, owner, body, bytes}); });
    visitor(ContactStorageAdmission{catalog_->metadata_identity(), catalog_->private_metadata_bytes(), nullptr, 0});
    catalog_->points.visit_admissions(visitor);
    fan_owners_.visit(
        [&](const void*, const auto& owner)
        {
            visitor(ContactStorageAdmission{owner->metadata_identity(), owner->private_metadata_bytes(),
                &owner->block(), owner->shared_body_bytes()});
        });
}
template<class H>
template<class Visitor>
void ContactPointGroups<H>::visit_borrowed_payload(Visitor&& visitor, bool include_object) const
{
    if (include_object)
        contact_payload_object(visitor, this, ContactPayloadRole::GroupObject, sizeof(*this));
    contact_payload_buffer(visitor, this, ContactPayloadRole::GroupSlots, groups_);
    contact_payload_buffer(visitor, this, ContactPayloadRole::GroupFreeSlots, free_);
    contact_payload_buffer(visitor, this, ContactPayloadRole::GroupBuckets, buckets_);
    contact_payload_object(visitor, readers_.get(), ContactPayloadRole::ReaderObject, sizeof(Readers), true);
    fan_owners_.visit_borrowed_payload(visitor, ContactPayloadRole::FanOwnerIndexBuckets);
    domains_.visit_borrowed_payload(visitor);
    catalog_->visit_borrowed_payload(visitor);
    fan_owners_.visit([&](const void*, const auto& owner) { owner->visit_borrowed_payload(visitor); });
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
