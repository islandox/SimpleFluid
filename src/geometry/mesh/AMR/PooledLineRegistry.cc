#include "PooledLineRegistry.hh"
#include "IncidencePhysicalOwners.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <cfenv>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
thread_local std::size_t (*replay_admission_observer)(const Budget&, std::size_t) = nullptr;

class CompletedMemberEnvironment
{
    std::fenv_t caller_{};

public:
    CompletedMemberEnvironment()
    {
        if (std::feholdexcept(&caller_))
            throw std::invalid_argument("Cannot isolate completed native member metadata.");
    }
    ~CompletedMemberEnvironment() { (void) std::fesetenv(&caller_); }
};
bool completed_point_equal(const IncidencePools::Point& a, const IncidencePools::Point& b)
{
    const auto bytes = PLIC::canonical_key_size(a);
    if (bytes != PLIC::canonical_key_size(b))
        return false;
    for (std::size_t offset = 0; offset < bytes; offset += 8)
        if (PLIC::details::GeometryIdentityAccess::canonical_word_at(a, offset) !=
            PLIC::details::GeometryIdentityAccess::canonical_word_at(b, offset))
            return false;
    return true;
}
void add_checked(std::size_t& a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Pooled line allocation overflow.");
    a += b;
}
std::size_t array_bytes(std::size_t count, std::size_t element)
{
    if (count > std::numeric_limits<std::size_t>::max() / element)
        throw std::invalid_argument("Pooled line capacity overflow.");
    return count * element;
}
std::array<std::uint64_t, 6> run_values(const PooledLineSubscriber& value)
{
    return {value.support, value.first, value.last, value.emission, value.edge, value.source_generation};
}
std::optional<std::uint64_t> progressed(std::uint64_t origin, std::int64_t step, std::size_t index)
{
    const auto magnitude = step < 0 ? std::uint64_t(-(step + 1)) + 1 : std::uint64_t(step);
    if (magnitude && index > std::numeric_limits<std::uint64_t>::max() / magnitude)
        return {};
    const auto delta = magnitude * index;
    if (step < 0)
        return delta > origin ? std::optional<std::uint64_t>{} : std::optional<std::uint64_t>{origin - delta};
    if (delta > std::numeric_limits<std::uint64_t>::max() - origin)
        return {};
    return origin + delta;
}
} // namespace
PooledLineRegistry::ReplayAdmissionObserver PooledLineRegistry::exchange_replay_admission_observer(
    ReplayAdmissionObserver observer) noexcept
{
    return std::exchange(replay_admission_observer, observer);
}
PooledLineSubscriber PooledLineRegistry::SubscriberRun::at(
    std::size_t index, std::span<const std::array<std::uint64_t, 6>> patterns) const
{
    if (index < first || index - first >= count)
        throw std::out_of_range("Invalid certified subscriber run ordinal.");
    std::array<std::uint64_t, 6> values{};
    if (!period)
        throw std::invalid_argument("Invalid certified subscriber pattern period.");
    const auto phase = (index - first) % period, block = (index - first) / period;
    if (phase && std::size_t(pattern_offset) + phase - 1 >= patterns.size())
        throw std::invalid_argument("Invalid certified subscriber pattern reference.");
    const auto& base = phase ? patterns[std::size_t(pattern_offset) + phase - 1] : origin;
    for (std::size_t field = 0; field < values.size(); ++field)
    {
        const auto value = progressed(base[field], step[field], block);
        if (!value)
            throw std::invalid_argument("Certified subscriber run integer range differs from replay.");
        values[field] = *value;
    }
    for (std::size_t field = 0; field < 3; ++field)
        if (values[field] > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Certified subscriber run slot is unrepresentable.");
    PooledLineSubscriber result;
    result.support = values[0];
    result.first = values[1];
    result.last = values[2];
    result.emission = values[3];
    result.edge = values[4];
    result.source_generation = values[5];
    result.resident = resident;
    result.support_authority = support_authority;
    result.image = image;
    result.native_source = true;
    result.result_state = PooledLineSubscriber::ResultState::ForwardEndpoints;
    return result;
}
PooledLineSubscriber PooledLineRegistry::Bucket::metadata(std::size_t index, const IncidencePools& pools) const
{
    if (source_ranges)
    {
        if (source_ranges->certified_layer() != certified_layer)
            throw std::invalid_argument("Cached source line layer witness differs from its sealed bucket.");
        return source_ranges->at(index, pools);
    }
    if (runs.empty())
        return subscribers.at(index).metadata();
    const auto end = std::upper_bound(
        runs.begin(), runs.end(), index, [](auto ordinal, const auto& run) { return ordinal < run.first; });
    if (end == runs.begin())
        throw std::out_of_range("Invalid certified subscriber ordinal.");
    return std::prev(end)->at(index, patterns);
}
std::span<const std::size_t> PooledLineRegistry::Bucket::explicit_result(std::size_t index) const
{
    if (source_ranges)
    {
        if (index >= size())
            throw std::out_of_range("Invalid source line result ordinal.");
        return {};
    }
    if (!runs.empty())
    {
        if (index >= size())
            throw std::out_of_range("Invalid line run result ordinal.");
        return {};
    }
    const auto& entry = subscribers.at(index);
    return entry.spill ? std::span<const std::size_t>(entry.spill->result) : std::span<const std::size_t>{};
}
std::size_t PooledLineRegistry::Bucket::spill_bytes(std::size_t index, const IncidencePools& pools) const
{
    if (!source_ranges && runs.empty())
        return subscribers.at(index).spill_bytes();
    const auto value = metadata(index, pools);
    auto bytes =
        StoredSubscriber::needs_wide(value) || value.result_state == PooledLineSubscriber::ResultState::Explicit
            ? sizeof(SubscriberSpill)
            : 0;
    if (value.result_state == PooledLineSubscriber::ResultState::Explicit)
        add_checked(bytes, array_bytes(result_size(index, pools), sizeof(std::size_t)));
    return bytes;
}
std::size_t PooledLineRegistry::Bucket::result_size(std::size_t index, const IncidencePools& pools) const
{
    if (source_ranges)
        return source_ranges->result_size(index, pools);
    const auto value = metadata(index, pools);
    return value.result_state == PooledLineSubscriber::ResultState::ForwardEndpoints ? 2
                                                                                     : explicit_result(index).size();
}
std::size_t PooledLineRegistry::Bucket::result_at(
    std::size_t index, std::size_t output, const IncidencePools& pools) const
{
    if (source_ranges)
        return source_ranges->result_at(index, output, pools);
    const auto value = metadata(index, pools);
    if (value.result_state == PooledLineSubscriber::ResultState::ForwardEndpoints)
        return value.result_at(output);
    const auto result = explicit_result(index);
    if (output >= result.size())
        throw std::out_of_range("Invalid cached explicit line result ordinal.");
    return result[output];
}
PooledLineRegistry::StoredSubscriber PooledLineRegistry::Bucket::clone(
    std::size_t index, const IncidencePools& pools) const
{
    if (!source_ranges && runs.empty())
        return subscribers.at(index).clone();
    auto value = metadata(index, pools);
    if (value.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        const auto count = result_size(index, pools);
        value.result.reserve(count);
        if (value.result.capacity() != count)
            throw std::invalid_argument("Source line clone result capacity differs from admission.");
        for (std::size_t output = 0; output < count; ++output)
            value.result.push_back(result_at(index, output, pools));
    }
    return StoredSubscriber::pack(std::move(value));
}
bool PooledLineRegistry::StoredSubscriber::needs_wide(const PooledLineSubscriber& sub) noexcept
{
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    return sub.support > max || sub.first > max || sub.last > max || sub.emission > max || sub.edge > max ||
           sub.source_generation > std::numeric_limits<std::uint16_t>::max();
}
PooledLineSubscriber PooledLineRegistry::StoredSubscriber::metadata() const
{
    PooledLineSubscriber result;
    result.support = wide() ? spill->support : support;
    result.first = wide() ? spill->first : first;
    result.last = wide() ? spill->last : last;
    result.emission = wide() ? spill->emission : emission;
    result.edge = wide() ? spill->edge : edge;
    result.source_generation = wide() ? spill->source_generation : source_generation;
    result.resident = resident;
    result.support_authority = support_authority;
    result.image = image & 1;
    result.native_source = native();
    result.result_state = result_state;
    return result;
}
PooledLineRegistry::StoredSubscriber PooledLineRegistry::StoredSubscriber::pack(PooledLineSubscriber&& sub)
{
    if (sub.image > 1)
        throw std::invalid_argument("Invalid pooled physical line image.");
    StoredSubscriber result;
    result.resident = sub.resident;
    result.support_authority = sub.support_authority;
    result.image = sub.image | (sub.native_source ? 0x40 : 0) | (needs_wide(sub) ? 0x80 : 0);
    result.result_state = sub.result_state;
    if (result.wide() || sub.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        result.spill = std::make_unique<SubscriberSpill>();
        result.spill->result = std::move(sub.result);
    }
    if (result.wide())
    {
        result.spill->support = sub.support;
        result.spill->first = sub.first;
        result.spill->last = sub.last;
        result.spill->emission = sub.emission;
        result.spill->edge = sub.edge;
        result.spill->source_generation = sub.source_generation;
    }
    else
    {
        result.support = static_cast<std::uint32_t>(sub.support);
        result.first = static_cast<std::uint32_t>(sub.first);
        result.last = static_cast<std::uint32_t>(sub.last);
        result.emission = static_cast<std::uint32_t>(sub.emission);
        result.edge = static_cast<std::uint32_t>(sub.edge);
        result.source_generation = static_cast<std::uint16_t>(sub.source_generation);
    }
    return result;
}
PooledLineRegistry::StoredSubscriber PooledLineRegistry::StoredSubscriber::clone() const
{
    auto result = pack(metadata());
    if (spill && !spill->result.empty())
    {
        result.spill->result.reserve(spill->result.size());
        if (result.spill->result.capacity() != spill->result.size())
            throw std::invalid_argument("Pooled line spill capacity differs from admission.");
        result.spill->result.insert(result.spill->result.end(), spill->result.begin(), spill->result.end());
    }
    return result;
}
std::size_t PooledLineRegistry::StoredSubscriber::spill_bytes() const
{
    if (!spill)
        return 0;
    auto bytes = sizeof(SubscriberSpill);
    add_checked(bytes, array_bytes(spill->result.capacity(), sizeof(std::size_t)));
    return bytes;
}
PooledLineRegistry::Bucket::~Bucket()
{
    source_ranges.reset();
    decltype(patterns){}.swap(patterns);
    decltype(runs){}.swap(runs);
    decltype(subscribers){}.swap(subscribers);
    if (arena)
        arena->used -= charge;
}
namespace
{
std::shared_ptr<Budget> line_arena(Budget& budget, const IncidencePools& pools)
{
    if (const auto& keeper = pools.points.keeper())
    {
        if (keeper.get() != &budget)
            throw std::invalid_argument("Pooled line registry received a foreign arena.");
        return keeper;
    }
    return std::shared_ptr<Budget>(std::shared_ptr<Budget>{}, &budget);
}
std::shared_ptr<ContactPhysicalResidency> line_residency(ContactGroupAllocation& header,
    const std::shared_ptr<ContactPhysicalResidency>& supplied, const ContactPhysicalResidency* previous = nullptr)
{
    if (supplied)
        return supplied;
    header.add(sizeof(ContactPhysicalResidency) + 128);
    return previous ? std::make_shared<ContactPhysicalResidency>(*previous, header.keeper())
                    : std::make_shared<ContactPhysicalResidency>(header.keeper());
}
} // namespace
PooledLineRegistry::PooledLineRegistry(
    Budget& budget, IncidencePools& pools, std::shared_ptr<ContactPhysicalResidency> shared)
    : budget_(&budget), pools_(&pools), keeper_(line_arena(budget, pools)), inherited_(keeper_),
      residency_header_(keeper_), residency_(line_residency(residency_header_, shared)), shared_residency_(bool(shared))
{
}
PooledLineRegistry::PooledLineRegistry(const PooledLineRegistry& source, Budget& budget, IncidencePools& pools,
    std::shared_ptr<ContactPhysicalResidency> shared)
    : budget_(&budget), pools_(&pools), keeper_(line_arena(budget, pools)), inherited_(keeper_),
      residency_header_(keeper_),
      residency_((source.require_publishable(),
          line_residency(residency_header_, shared,
              pools.points.keeper() && source.pools_->points.keeper() ? source.residency_.get() : nullptr))),
      shared_residency_(bool(shared)), keep_overlap_(true), layer_(source.layer_), replay_(source.replay_),
      processed_lines_(source.processed_lines_)
{
    try
    {
        if (bool(shared) != source.shared_residency_)
            throw std::invalid_argument("Pooled line registry fork changed its ownership policy.");
        const bool share = bool(pools.points.keeper()) && bool(source.pools_->points.keeper());
        if (share)
        {
            lines_ = source.lines_;
            inherited_.add(lines_ ? lines_->subtree_bytes : 0);
            retained_ = source.retained_;
            retained_bucket_bytes_ = source.retained_bucket_bytes_;
        }
        else
        {
            if (shared)
                throw std::invalid_argument("Shared incidence line registries need exact lifetime keepers.");
            // Raw-budget standalone compatibility. Do not leave a shared
            // bucket or index node referring to a caller's expired arena.
            cartesian_detail::Allocation scratch(budget);
            std::map<const Bucket*, std::shared_ptr<const Bucket>> aliases;
            std::vector<Witness> rows;
            scratch.reserve(rows, source.lines_ ? source.lines_->size : 0);
            Index::visit_from(
                source.lines_, [](const auto&) { return false; },
                [&](const Witness& row)
                {
                    std::shared_ptr<const Bucket> bucket;
                    if (row.bucket)
                    {
                        const auto* old = row.bucket.get();
                        auto found = aliases.find(old);
                        if (found == aliases.end())
                        {
                            const auto bytes = old->charge;
                            scratch.retain(bytes);
                            auto next = std::make_shared<Bucket>();
                            next->certified_layer = old->certified_layer;
                            next->subscription_count = old->subscription_count;
                            next->source_ranges = old->source_ranges;
                            next->runs = old->runs;
                            next->patterns = old->patterns;
                            next->subscribers.reserve(old->subscribers.size());
                            for (const auto& sub : old->subscribers)
                                next->subscribers.push_back(sub.clone());
                            const auto actual = body(*next);
                            if (actual > bytes)
                                throw std::invalid_argument("Copied line bucket exceeds its original admission.");
                            transfer_bucket(*next, actual, scratch.bytes);
                            scratch.retain(sizeof(decltype(aliases)::value_type) + 4 * sizeof(void*));
                            found = aliases.emplace(old, std::move(next)).first;
                        }
                        bucket = found->second;
                        observe_bucket(bucket, true);
                    }
                    rows.push_back({row.key, std::move(bucket)});
                    return true;
                });
            lines_ = Index(keeper_).build(rows);
        }
        for (const auto slot : source.dirty_)
            admit(set_node_bytes);
        dirty_ = source.dirty_;
    }
    catch (...)
    {
        lines_.reset();
        decltype(dirty_){}.swap(dirty_);
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
PooledLineRegistry::~PooledLineRegistry()
{
    lines_.reset();
    decltype(dirty_){}.swap(dirty_);
    decltype(touched_){}.swap(touched_);
    budget_->used -= charged_;
}
void PooledLineRegistry::require_publishable() const
{
    if (poisoned_)
        throw std::logic_error("Discard the poisoned private line registry and its owner editor.");
}
void PooledLineRegistry::admit(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() - charged_)
        throw std::invalid_argument("Pooled line charge overflow.");
    budget_->add(n);
    charged_ += n;
}
void PooledLineRegistry::release(std::size_t n)
{
    budget_->used -= n;
    charged_ -= n;
}
void PooledLineRegistry::transfer_bucket(Bucket& bucket, std::size_t bytes, std::size_t& from) noexcept
{
    bucket.arena = keeper_;
    bucket.charge = bytes;
    from -= bytes;
}
void PooledLineRegistry::observe_bucket(const std::shared_ptr<const Bucket>& bucket, bool add)
{
    if (!bucket)
        return;
    using Role = IncidencePhysicalOwners::Role;
    const auto& credit_arena = pools_->points.source_arena();
    IncidencePhysicalOwners owners(*residency_, credit_arena.get(), *budget_, add);
    if (!owners.observe(Role::LineBucket, bucket, bucket->charge, bucket->arena.get()))
        return;
    const auto adjust = [&](std::size_t& count, std::size_t value)
    {
        if (add)
            add_checked(count, value);
        else
            count -= value;
    };
    adjust(retained_bucket_bytes_, bucket->charge);
    if (const auto& recipe = bucket->source_ranges)
    {
        adjust(retained_.retained_source_records, bucket->size());
        if (owners.observe(Role::SourceDependency, recipe, recipe->metadata_bytes(), recipe->allocation_budget()))
            owners.observe_proof(*recipe);
    }
    else if (bucket->runs.empty())
        adjust(retained_.retained_explicit_records, bucket->size());
    else
        adjust(retained_.retained_run_records, bucket->size());
}
std::size_t PooledLineRegistry::retained_bytes() const noexcept
{
    return charged_ + retained_bucket_bytes_ + (lines_ ? lines_->subtree_bytes : 0);
}
std::size_t PooledLineRegistry::overlap_bytes() const noexcept
{
    const auto live = !lines_ || lines_->charge.arena != keeper_ ? (lines_ ? lines_->subtree_bytes : 0)
                                                                 : lines_->subtree_bytes - lines_->same_arena_bytes;
    return inherited_.bytes() - live + (shared_residency_ ? 0 : residency_->overlap_bytes());
}
void PooledLineRegistry::finish_overlap() noexcept
{
    if (poisoned_)
        return;
    keep_overlap_ = false;
    sync_index_charge();
    if (!shared_residency_)
        residency_->finish_overlap();
}
void PooledLineRegistry::sync_index_charge()
{
    const auto live = !lines_ || lines_->charge.arena != keeper_ ? (lines_ ? lines_->subtree_bytes : 0)
                                                                 : lines_->subtree_bytes - lines_->same_arena_bytes;
    if (live > inherited_.bytes())
        throw std::logic_error("Line edit introduced an unadmitted inherited index page.");
    if (!keep_overlap_)
        inherited_.release(inherited_.bytes() - live);
}
void PooledLineRegistry::reap_deferred() noexcept
{
    if (const auto& arena = pools_->points.source_arena())
        arena->reap();
}
std::size_t PooledLineRegistry::body(const Bucket& bucket)
{
    std::size_t result = bucket_owner_bytes;
    add_checked(result, array_bytes(bucket.subscribers.capacity(), sizeof(StoredSubscriber)));
    add_checked(result, array_bytes(bucket.runs.capacity(), sizeof(SubscriberRun)));
    add_checked(result, array_bytes(bucket.patterns.capacity(), sizeof(std::array<std::uint64_t, 6>)));
    for (const auto& sub : bucket.subscribers)
        add_checked(result, sub.spill_bytes());
    return result;
}
PooledLineRegistry::ActiveStorage PooledLineRegistry::active_storage() const
{
    ActiveStorage result = retained_;
    std::optional<std::size_t> seen;
    Index::visit_from(
        lines_, [](const auto&) { return false; },
        [&](const Witness& row)
        {
            const auto line = row.key.first;
            if (seen == line)
                return true;
            seen = line;
            const auto* current = effective(line, layer_);
            if (!current || !current->bucket)
                return true;
            ++result.lines;
            const auto& bucket = *current->bucket;
            if (const auto& recipe = bucket.source_ranges)
            {
                result.source_records += recipe->size();
                result.source_families += recipe->family_count();
                result.source_nonforward_records += recipe->nonforward_count();
                result.source_result_selectors += recipe->result_selector_count();
                result.source_bytes += body(bucket) + recipe->metadata_bytes();
                std::optional<std::size_t> prior;
                bool first_catalog = true;
                Index::visit_from(
                    lines_, [](const auto&) { return false; },
                    [&](const Witness& old)
                    {
                        const auto old_line = old.key.first;
                        if (old_line >= line)
                            return false;
                        if (prior == old_line)
                            return true;
                        prior = old_line;
                        const auto* earlier = effective(old_line, layer_);
                        first_catalog = !earlier || !earlier->bucket || !earlier->bucket->source_ranges ||
                                        earlier->bucket->source_ranges->catalog() != recipe->catalog();
                        return first_catalog;
                    });
                if (first_catalog)
                {
                    result.source_catalog_bytes += recipe->catalog()->metadata_bytes();
                    result.source_maps += recipe->catalog()->map_count();
                    result.source_mapped_components += recipe->catalog()->mapped_components();
                }
            }
            else if (bucket.runs.empty())
            {
                result.explicit_records += bucket.size();
                result.explicit_bytes += body(bucket);
            }
            else
            {
                result.run_records += bucket.size();
                result.runs += bucket.runs.size();
                result.run_bytes += body(bucket);
            }
            return true;
        });
    return result;
}
const PooledLineRegistry::Witness* PooledLineRegistry::effective(std::size_t line, unsigned layer) const noexcept
{
    const Key target{line, layer};
    auto* node = lines_.get();
    const Witness* previous = nullptr;
    while (node && !node->leaf())
    {
        const auto child = std::partition_point(node->children.begin(), node->children.end(),
            [&](const auto& entry) { return entry->maximum->key <= target; });
        if (child == node->children.end())
        {
            previous = node->maximum;
            node = nullptr;
            break;
        }
        if (child != node->children.begin())
            previous = (*std::prev(child))->maximum;
        node = child->get();
    }
    if (node)
    {
        const auto at = std::partition_point(
            node->rows.begin(), node->rows.end(), [&](const auto& row) { return row.key <= target; });
        if (at != node->rows.begin())
            previous = &*std::prev(at);
    }
    return previous && previous->key.first == line ? previous : nullptr;
}
const PooledLineRegistry::Witness* PooledLineRegistry::exact(Key key) const noexcept
{
    const auto* found = Index::lower_bound(lines_, [&](const auto& value) { return value.key < key; });
    return found && found->key == key ? found : nullptr;
}
bool PooledLineRegistry::has_touched_raw_buckets() const noexcept
{
    if (!retained_.retained_explicit_records && !retained_.retained_run_records)
        return false;
    for (const auto line : touched_)
    {
        bool raw = false;
        Index::visit_from(
            lines_, [&](const auto& row) { return row.key.first < line; },
            [&](const Witness& row)
            {
                if (row.key.first != line)
                    return false;
                raw = row.bucket && !row.bucket->source_ranges;
                return !raw;
            });
        if (raw)
            return true;
    }
    return false;
}
void PooledLineRegistry::begin_replay(bool bootstrap)
{
    require_publishable();
    reap_deferred();
    if (!dirty_.empty())
        throw std::invalid_argument("Unfinished line replay.");
    finish_source_selection();
    certified_witnesses_ = 0;
    replay_ = !bootstrap;
    layer_ = 0;
}
void PooledLineRegistry::set_layer(unsigned layer)
{
    require_publishable();
    if (!dirty_.empty() || layer < layer_ || layer >= 64)
        throw std::invalid_argument("Invalid incremental line layer.");
    layer_ = layer;
}
void PooledLineRegistry::finish_source_selection() noexcept
{
    if (poisoned_)
        return;
    const auto bytes = touched_.size() * set_node_bytes;
    touched_.clear();
    release(bytes);
}
void PooledLineRegistry::finish_replay(unsigned last_layer)
{
    require_publishable();
    if (!dirty_.empty())
        throw std::invalid_argument("Unfinished line replay.");
    MutationGuard mutation{*this};
    for (const auto line : touched_)
        while (const auto* found =
                   Index::lower_bound(lines_, [&](const auto& row) { return row.key <= Key{line, last_layer}; }))
        {
            if (found->key.first != line)
                break;
            edit_witness(found->key, nullptr);
        }
    reap_deferred();
}
void PooledLineRegistry::edit_witness(Key key, const std::shared_ptr<const Bucket>* replacement)
{
    const auto* found = exact(key);
    if (!found && !replacement)
        throw std::logic_error("Missing exact line witness retirement.");
    const auto old = found ? found->bucket : nullptr;
    Witness next{key, replacement ? *replacement : nullptr};
    if (found && replacement && old == next.bucket)
        return;
    MutationGuard mutation{*this};
    auto root = Index(keeper_).edit(lines_, next, replacement ? &next : nullptr, found != nullptr);
    // Add new owners before retiring old. A future-layer alias is an actual
    // second reference to the same physical bucket, not a second body charge.
    if (old != next.bucket)
    {
        observe_bucket(next.bucket, true);
        observe_bucket(old, false);
    }
    lines_ = std::move(root);
    sync_index_charge();
}
void PooledLineRegistry::preserve_future(std::size_t line)
{
    if (!replay_ || layer_ == 63 || exact({line, layer_ + 1}))
        return;
    const auto* found = effective(line, layer_);
    const auto old = found ? found->bucket : nullptr;
    edit_witness({line, layer_ + 1}, &old);
}
void PooledLineRegistry::replace(std::size_t line, std::shared_ptr<const Bucket> next)
{
    edit_witness({line, layer_}, &next);
}
void PooledLineRegistry::apply(std::span<const PooledLineEdit> changes)
{
    require_publishable();
    reap_deferred();
    for (const auto& change : changes)
    {
        (void) pools_->lines.get(change.line);
        const auto* found = effective(change.line, layer_);
        const auto old = found ? found->bucket : nullptr;
        const auto count = old ? old->size() : 0;
        if (!change.remove && count == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Pooled subscriber count overflow.");
        std::size_t position = 0;
        bool same = false;
        if (old)
        {
            std::size_t lower = 0, upper = count;
            while (lower < upper)
            {
                const auto middle = lower + (upper - lower) / 2;
                if (line_subscriber_before(old->metadata(middle, *pools_), change.subscriber, *pools_))
                    lower = middle + 1;
                else
                    upper = middle;
            }
            position = lower;
            same = position < count &&
                   !line_subscriber_before(change.subscriber, old->metadata(position, *pools_), *pools_);
        }
        if (change.remove && !same)
            throw std::invalid_argument("Unknown retired line subscription.");
        if (change.remove && same)
        {
            const auto metadata = old->metadata(position, *pools_);
            if (metadata.source_generation != change.subscriber.source_generation ||
                metadata.native_source != change.subscriber.native_source)
                throw std::invalid_argument("Stale native line source retirement.");
        }
        if (!change.remove && (same || change.subscriber.has_result() || !change.subscriber.result.empty()))
            throw std::invalid_argument("Duplicate or prepopulated line subscription.");
        // Membership and generation rejection are read-only, so callers can
        // correct a stale request and retry. Admissions below are private.
        MutationGuard mutation{*this};
        if (!touched_.contains(change.line))
        {
            admit(set_node_bytes);
            touched_.insert(change.line);
        }
        preserve_future(change.line);
        const auto next_count = change.remove ? count - 1 : count + 1;
        std::size_t admitted = bucket_owner_bytes;
        add_checked(admitted, array_bytes(next_count, sizeof(StoredSubscriber)));
        if (old)
            for (std::size_t i = 0; i < count; ++i)
                if (!change.remove || i != position)
                    add_checked(admitted, old->spill_bytes(i, *pools_));
        if (!change.remove && StoredSubscriber::needs_wide(change.subscriber))
            add_checked(admitted, sizeof(SubscriberSpill));
        admit(admitted);
        auto next = std::make_shared<Bucket>();
        next->subscribers.reserve(next_count);
        if (next->subscribers.capacity() != next_count)
            throw std::invalid_argument("Pooled subscriber capacity differs from admission.");
        for (std::size_t i = 0; i <= count; ++i)
        {
            if (i == position && !change.remove)
            {
                // New subscriptions have no result buffer; this stack copy
                // preserves the public transient type without an allocation.
                auto sub = change.subscriber;
                next->subscribers.push_back(StoredSubscriber::pack(std::move(sub)));
            }
            if (i < count && (!change.remove || i != position))
                next->subscribers.push_back(old->clone(i, *pools_));
        }
        const auto actual = body(*next);
        if (actual > admitted)
            throw std::invalid_argument("Pooled line copy exceeded its admission.");
        release(admitted - actual);
        transfer_bucket(*next, actual, charged_);
        if (next->subscribers.empty())
        {
            next.reset();
            replace(change.line, {});
        }
        else
            replace(change.line, std::move(next));
        if (!dirty_.contains(change.line))
        {
            admit(set_node_bytes);
            dirty_.insert(change.line);
        }
    }
}
void PooledLineRegistry::recompute(const PooledChangedLineQuery& publish)
{
    while (const auto line = next_dirty())
        recompute_one(*line, publish);
}
std::optional<std::size_t> PooledLineRegistry::next_dirty() const
{
    require_publishable();
    if (dirty_.empty())
        return {};
    return *dirty_.begin();
}
bool PooledLineRegistry::certify_runs(
    Bucket& bucket, std::span<const PooledLineSubscriber> values, std::size_t& scratch)
{
    if (values.empty() || std::ranges::any_of(values,
                              [](const auto& value)
                              {
                                  return !value.native_source || !value.source_generation ||
                                         value.result_state != PooledLineSubscriber::ResultState::ForwardEndpoints ||
                                         !value.result.empty();
                              }))
        return false;
    auto same_constants = [](const auto& first, const auto& next)
    {
        return first.resident == next.resident && first.support_authority == next.support_authority &&
               first.image == next.image;
    };
    struct Candidate
    {
        SubscriberRun run;
        std::array<std::array<std::uint64_t, 6>, 16> bases{};
        std::size_t bytes = 0, saving = 0, loss = 0;
    };
    auto candidate = [&](std::size_t first)
    {
        Candidate best;
        best.run.first = first;
        best.run.count = 1;
        best.run.origin = run_values(values[first]);
        best.run.resident = values[first].resident;
        best.run.support_authority = values[first].support_authority;
        best.run.image = values[first].image;
        best.bases[0] = best.run.origin;
        best.bytes = sizeof(SubscriberRun);
        best.loss = sizeof(SubscriberRun) - sizeof(StoredSubscriber);
        for (unsigned period = 1; period <= 16 && 2 * std::size_t(period) <= values.size() - first; ++period)
        {
            Candidate next = best;
            next.run.count = 2 * period;
            next.run.period = period;
            next.run.first = first;
            next.run.origin = run_values(values[first]);
            next.bases[0] = next.run.origin;
            bool valid = true;
            for (unsigned phase = 0; phase < 2 * period; ++phase)
                valid = valid && same_constants(values[first], values[first + phase]);
            if (!valid)
                continue;
            const auto following = run_values(values[first + period]);
            for (std::size_t field = 0; field < following.size(); ++field)
            {
                const auto difference = following[field] >= next.run.origin[field]
                                            ? following[field] - next.run.origin[field]
                                            : next.run.origin[field] - following[field];
                if (difference > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
                {
                    valid = false;
                    break;
                }
                next.run.step[field] =
                    following[field] >= next.run.origin[field] ? std::int64_t(difference) : -std::int64_t(difference);
            }
            if (!valid)
                continue;
            for (unsigned phase = 1; phase < period; ++phase)
                next.bases[phase] = run_values(values[first + phase]);
            for (unsigned phase = 0; phase < period && valid; ++phase)
            {
                const auto actual = run_values(values[first + period + phase]);
                for (std::size_t field = 0; field < actual.size(); ++field)
                {
                    const auto expected = progressed(next.bases[phase][field], next.run.step[field], 1);
                    valid = valid && expected && *expected == actual[field];
                }
            }
            if (!valid)
                continue;
            while (
                first + next.run.count < values.size() && same_constants(values[first], values[first + next.run.count]))
            {
                const auto actual = run_values(values[first + next.run.count]);
                bool matches = true;
                const auto phase = next.run.count % period, block = next.run.count / period;
                for (std::size_t field = 0; field < actual.size(); ++field)
                {
                    const auto expected = progressed(next.bases[phase][field], next.run.step[field], block);
                    matches = matches && expected && *expected == actual[field];
                }
                if (!matches)
                    break;
                ++next.run.count;
            }
            next.bytes = sizeof(SubscriberRun) + (period - 1) * sizeof(next.bases[0]);
            const auto raw = array_bytes(next.run.count, sizeof(StoredSubscriber));
            next.saving = raw > next.bytes ? raw - next.bytes : 0;
            next.loss = next.bytes > raw ? next.bytes - raw : 0;
            if (next.saving > best.saving ||
                (next.saving == best.saving &&
                    (next.loss < best.loss || (next.loss == best.loss && next.run.count > best.run.count))))
                best = next;
        }
        return best;
    };
    std::size_t run_count = 0, pattern_count = 0;
    for (std::size_t first = 0; first < values.size();)
    {
        const auto next = candidate(first);
        ++run_count;
        add_checked(pattern_count, next.run.period - 1);
        first += next.run.count;
    }
    if (pattern_count > std::numeric_limits<std::uint32_t>::max())
        return false;
    auto bytes = array_bytes(run_count, sizeof(SubscriberRun));
    add_checked(bytes, array_bytes(pattern_count, sizeof(std::array<std::uint64_t, 6>)));
    if (bytes >= array_bytes(values.size(), sizeof(StoredSubscriber)))
        return false;
    if (bytes > std::numeric_limits<std::size_t>::max() - scratch)
        throw std::invalid_argument("Certified line run credit overflows.");
    admit(bytes);
    scratch += bytes;
    bucket.runs.reserve(run_count);
    bucket.patterns.reserve(pattern_count);
    if (bucket.runs.capacity() != run_count || bucket.patterns.capacity() != pattern_count)
        throw std::invalid_argument("Certified line run capacity differs from admission.");
    for (std::size_t first = 0; first < values.size();)
    {
        auto next = candidate(first);
        next.run.pattern_offset = static_cast<std::uint32_t>(bucket.patterns.size());
        for (unsigned phase = 1; phase < next.run.period; ++phase)
            bucket.patterns.push_back(next.bases[phase]);
        bucket.runs.push_back(next.run);
        first += next.run.count;
    }
    bucket.subscription_count = values.size();
    // Slot progressions are only a candidate codec. Full canonical records and
    // actual complete-query output identities certify the retained representation.
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const auto replayed = bucket.metadata(i, *pools_);
        const auto& actual = values[i];
        if (run_values(replayed) != run_values(actual) || replayed.resident != actual.resident ||
            replayed.support_authority != actual.support_authority || replayed.image != actual.image ||
            replayed.native_source != actual.native_source || replayed.result_state != actual.result_state ||
            pools_->supports.get(replayed.support) != pools_->supports.get(actual.support) ||
            pools_->points.get(replayed.first) != pools_->points.get(actual.first) ||
            pools_->points.get(replayed.last) != pools_->points.get(actual.last))
        {
            std::vector<SubscriberRun>().swap(bucket.runs);
            decltype(bucket.patterns){}.swap(bucket.patterns);
            bucket.subscription_count = 0;
            release(bytes);
            scratch -= bytes;
            return false;
        }
    }
    return true;
}
void PooledLineRegistry::recompute_one(std::size_t line, const PooledChangedLineQuery& publish)
{
    require_publishable();
    reap_deferred();
    const auto dirty = dirty_.find(line);
    if (dirty == dirty_.end())
        throw std::invalid_argument("Line is not dirty.");
    MutationGuard mutation{*this};
    ++processed_lines_;
    const auto found = effective(line, layer_);
    if (found && found->bucket)
    {
        const auto& old = *found->bucket;
        const auto count = old.size();
        std::size_t scratch = array_bytes(count, sizeof(PooledLineSubscriber));
        for (std::size_t i = 0; i < count; ++i)
            if (old.metadata(i, *pools_).result_state == PooledLineSubscriber::ResultState::Explicit)
                add_checked(scratch, array_bytes(old.result_size(i, *pools_), sizeof(std::size_t)));
        admit(scratch);
        struct ScratchCredit
        {
            PooledLineRegistry& registry;
            std::size_t& bytes;
            ~ScratchCredit() { registry.release(bytes); }
            void add(std::size_t amount)
            {
                if (amount > std::numeric_limits<std::size_t>::max() - bytes)
                    throw std::invalid_argument("Line scratch credit overflows.");
                registry.admit(amount);
                bytes += amount;
            }
            void drop(std::size_t amount)
            {
                registry.release(amount);
                bytes -= amount;
            }
        } credit{*this, scratch};
        std::vector<PooledLineSubscriber> working;
        working.reserve(count);
        if (working.capacity() != count)
            throw std::invalid_argument("Pooled query capacity differs from admission.");
        for (std::size_t i = 0; i < count; ++i)
        {
            auto transient = old.metadata(i, *pools_);
            if (transient.result_state == PooledLineSubscriber::ResultState::Explicit)
            {
                const auto result_count = old.result_size(i, *pools_);
                transient.result.reserve(result_count);
                if (transient.result.capacity() != result_count)
                    throw std::invalid_argument("Pooled query result capacity differs from admission.");
                for (std::size_t output = 0; output < result_count; ++output)
                    transient.result.push_back(old.result_at(i, output, *pools_));
            }
            working.push_back(std::move(transient));
        }
        std::vector<std::size_t> changed;
        query_pooled_line(
            working, *pools_, [&](std::size_t n) { credit.add(n); },
            [&](std::size_t i, std::span<const std::size_t>)
            {
                if (changed.size() == changed.capacity())
                {
                    if (changed.capacity() > std::numeric_limits<std::size_t>::max() / 2)
                        throw std::invalid_argument("Pooled changed-query capacity overflow.");
                    const auto next = changed.capacity() ? changed.capacity() * 2 : 1;
                    const auto bytes = array_bytes(next, sizeof(std::size_t));
                    const auto old_bytes = array_bytes(changed.capacity(), sizeof(std::size_t));
                    credit.add(bytes);
                    changed.reserve(next);
                    if (changed.capacity() != next)
                        throw std::invalid_argument("Pooled changed-query capacity differs from admission.");
                    credit.drop(old_bytes);
                }
                changed.push_back(i);
            });
        // The query's explicit result buffers already have scratch credit.
        // Admit only packed metadata here, then move those buffers into the
        // new immutable bucket before retiring the complete-line workspace.
        credit.add(bucket_owner_bytes);
        auto next = std::make_shared<Bucket>();
        next->certified_layer = layer_;
        if (!certify_runs(*next, working, scratch))
        {
            std::size_t packed = array_bytes(count, sizeof(StoredSubscriber));
            for (const auto& sub : working)
                if (StoredSubscriber::needs_wide(sub) ||
                    sub.result_state == PooledLineSubscriber::ResultState::Explicit)
                    add_checked(packed, sizeof(SubscriberSpill));
            credit.add(packed);
            next->subscribers.reserve(count);
            if (next->subscribers.capacity() != count)
                throw std::invalid_argument("Pooled subscriber capacity differs from admission.");
            for (auto& sub : working)
                next->subscribers.push_back(StoredSubscriber::pack(std::move(sub)));
        }
        const auto actual = body(*next);
        if (actual > scratch)
            throw std::invalid_argument("Pooled line query exceeded its admission.");
        // This scratch is part of charged_. The immutable bucket takes
        // ownership of these live bytes from both registry counters.
        transfer_bucket(*next, actual, charged_);
        scratch -= actual;
        replace(line, std::move(next));
        // A publisher sees an already committed replacement. Its exception
        // does not poison or roll back the completed line result.
        mutation.commit();
        for (const auto index : changed)
            publish(line, index);
        std::vector<PooledLineSubscriber>().swap(working);
        std::vector<std::size_t>().swap(changed);
    }
    dirty_.erase(dirty);
    release(set_node_bytes);
}
PooledLineResultView PooledLineRegistry::query_result(std::size_t line, std::size_t subscriber) const
{
    require_publishable();
    const auto found = effective(line, layer_);
    if (!found || !found->bucket)
        throw std::invalid_argument("Missing pooled line query result.");
    PooledLineResultView result{
        found->bucket->metadata(subscriber, *pools_), found->bucket->explicit_result(subscriber)};
    if (!result.subscriber.has_result())
        throw std::invalid_argument("Uncomputed pooled line query result.");
    if (found->bucket->source_ranges && result.subscriber.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        result.source_result = found->bucket->source_ranges.get();
        result.source_pools = pools_;
        result.source_ordinal = subscriber;
    }
    return result;
}
std::optional<PooledLineRegistry::CompletedMember> PooledLineRegistry::find_completed_member(std::size_t line,
    unsigned effective_layer, const PooledLineSubscriber& expected, const IncidencePools& request_namespace) const
{
    require_publishable();
    if (&request_namespace != pools_ || !expected.native_source || !expected.source_generation ||
        expected.has_result() || expected.result.capacity() || dirty_.contains(line))
        return {};
    CompletedMemberEnvironment environment;
    const auto found = effective(line, effective_layer);
    if (!found || !found->bucket)
        return {};
    const auto& bucket = *found->bucket;
    if (bucket.certified_layer > effective_layer)
        return {};
    for (std::size_t i = 0; i < bucket.size(); ++i)
    {
        const auto actual = bucket.metadata(i, *pools_);
        if (actual.resident != expected.resident || actual.support_authority != expected.support_authority ||
            pools_->supports.get(actual.support) != pools_->supports.get(expected.support) ||
            actual.emission != expected.emission || actual.image != expected.image || actual.edge != expected.edge)
            continue;
        if (!actual.has_result() || actual.source_generation != expected.source_generation ||
            actual.native_source != expected.native_source || actual.first != expected.first ||
            actual.last != expected.last ||
            !completed_point_equal(pools_->points.get(actual.first), pools_->points.get(expected.first)) ||
            !completed_point_equal(pools_->points.get(actual.last), pools_->points.get(expected.last)))
            return {};
        PooledLineResultView result{actual, bucket.explicit_result(i)};
        if (bucket.source_ranges && actual.result_state == PooledLineSubscriber::ResultState::Explicit)
        {
            result.source_result = bucket.source_ranges.get();
            result.source_pools = pools_;
            result.source_ordinal = i;
        }
        if (!result.result_size())
            return {};
        return CompletedMember{std::move(result), bucket.certified_layer};
    }
    return {};
}
std::optional<PooledLineRegistry::SavedSourceQuery> PooledLineRegistry::capture_saved_source_query(std::size_t line,
    unsigned effective_layer, unsigned kernel_witness_layer, const PooledLineSubscriber& expected) const
{
    require_publishable();
    const auto found = effective(line, effective_layer);
    if (!found || !found->bucket || !found->bucket->source_ranges)
        return {};
    const auto& bucket = *found->bucket;
    if (bucket.certified_layer != kernel_witness_layer ||
        bucket.source_ranges->certified_layer() != kernel_witness_layer)
        return {};
    for (std::size_t i = 0; i < bucket.size(); ++i)
    {
        const auto actual = bucket.metadata(i, *pools_);
        if (actual.resident != expected.resident || actual.support_authority != expected.support_authority ||
            pools_->supports.get(actual.support) != pools_->supports.get(expected.support) ||
            actual.emission != expected.emission || actual.image != expected.image || actual.edge != expected.edge ||
            actual.source_generation != expected.source_generation || actual.native_source != expected.native_source)
            continue;
        if (actual.first != expected.first || actual.last != expected.last ||
            pools_->points.get(actual.first) != pools_->points.get(expected.first) ||
            pools_->points.get(actual.last) != pools_->points.get(expected.last))
            return {};
        return SavedSourceQuery{bucket.source_ranges, i};
    }
    return {};
}
void PooledLineRegistry::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    Index::visit_pages(lines_,
        [&](const auto& page)
        {
            contact_payload_object(
                visitor, page.get(), GeometryContactPayloadRole::LineRegistryNodes, sizeof(*page), true);
            contact_payload_buffer(visitor, page.get(), GeometryContactPayloadRole::LineHistoryRows, page->rows);
            contact_payload_buffer(
                visitor, page.get(), GeometryContactPayloadRole::LineHistoryChildren, page->children);
        });
    Index::visit_from(
        lines_, [](const auto&) { return false; },
        [&](const Witness& row)
        {
            if (!row.bucket)
                return true;
            const auto& bucket = *row.bucket;
            contact_payload_object(
                visitor, &bucket, GeometryContactPayloadRole::LineBucketObject, sizeof(bucket), true);
            contact_payload_buffer(
                visitor, &bucket, GeometryContactPayloadRole::LineSubscriberValues, bucket.subscribers);
            contact_payload_buffer(visitor, &bucket, GeometryContactPayloadRole::LineRunValues, bucket.runs);
            contact_payload_buffer(visitor, &bucket, GeometryContactPayloadRole::LinePatternValues, bucket.patterns);
            for (const auto& subscriber : bucket.subscribers)
                if (subscriber.spill)
                {
                    contact_payload_object(visitor, subscriber.spill.get(), GeometryContactPayloadRole::LineSpillObject,
                        sizeof(SubscriberSpill));
                    contact_payload_buffer(visitor, subscriber.spill.get(),
                        GeometryContactPayloadRole::LineResultValues, subscriber.spill->result);
                }
            if (bucket.source_ranges)
                bucket.source_ranges->visit_borrowed_payload(visitor);
            return true;
        });
    for (const auto& entry : dirty_)
        contact_payload_node_value(visitor, this, GeometryContactPayloadRole::LineRegistryDirtyNodes, entry);
    for (const auto& entry : touched_)
        contact_payload_node_value(visitor, this, GeometryContactPayloadRole::LineRegistryTouchedNodes, entry);
}
bool PooledLineRegistry::imports_source_owner(const void* identity) const noexcept
{
    return bool(residency_->usage().find({identity, std::uint32_t(ContactPhysicalResidency::Role::CreditMetadata)}));
}
void PooledLineRegistry::certify_source_ranges(
    const CartesianLineRangeAdmission& admission, std::shared_ptr<CartesianAuthorityArena> arena)
{
    require_publishable();
    reap_deferred();
    if (!dirty_.empty())
        throw std::invalid_argument("Source line ranges require a completed actual line pass.");
    if (!has_touched_raw_buckets())
    {
        finish_source_selection();
        return;
    }
    CartesianSourceLineRangeBuilder builder(admission, *budget_, std::move(arena));
    struct Pending
    {
        Key key;
        std::shared_ptr<const Bucket> accepted;
        std::shared_ptr<const Bucket> replacement;
    };
    cartesian_detail::Allocation scratch(*budget_);
    std::vector<Pending> pending;
    const auto materialize =
        [&](const Bucket& bucket, std::vector<PooledLineSubscriber>& values, cartesian_detail::Allocation& group)
    {
        for (std::size_t i = 0; i < bucket.size(); ++i)
        {
            auto value = bucket.metadata(i, *pools_);
            if (!value.has_result())
                return false;
            if (value.result_state == PooledLineSubscriber::ResultState::Explicit)
            {
                const auto count = bucket.result_size(i, *pools_);
                group.reserve(value.result, count);
                for (std::size_t output = 0; output < count; ++output)
                    value.result.push_back(bucket.result_at(i, output, *pools_));
            }
            values.push_back(std::move(value));
        }
        return true;
    };
    for (const auto line : touched_)
        Index::visit_from(
            lines_, [&](const auto& row) { return row.key.first < line; },
            [&](const Witness& row)
            {
                if (row.key.first != line)
                    return false;
                if (!row.bucket || row.bucket->source_ranges)
                    return true;
                const auto alias = std::find_if(pending.begin(), pending.end(),
                    [&](const auto& item) { return item.key.first == line && item.accepted == row.bucket; });
                if (alias != pending.end())
                {
                    auto replacement = alias->replacement;
                    scratch.append(pending);
                    pending.push_back({row.key, row.bucket, std::move(replacement)});
                    return true;
                }
                const auto& old = *row.bucket;
                cartesian_detail::Allocation group(*budget_);
                std::vector<PooledLineSubscriber> values;
                group.reserve(values, old.size());
                if (materialize(old, values, group))
                    if (auto recipe = builder.make_recipe(line, values, *pools_, old.certified_layer))
                    {
                        scratch.append(pending);
                        scratch.retain(bucket_owner_bytes);
                        auto replacement = std::make_shared<Bucket>();
                        replacement->source_ranges = std::move(recipe);
                        replacement->certified_layer = old.certified_layer;
                        transfer_bucket(*replacement, bucket_owner_bytes, scratch.bytes);
                        pending.push_back({row.key, row.bucket, std::move(replacement)});
                    }
                return true;
            });
    builder.freeze();
    // A separate final pass proves every ordered scalar, FULL support identity,
    // endpoint identity and complete query result against real execution. Maps
    // are now immutable; no proof can be changed by a later bucket observation.
    for (const auto& item : pending)
    {
        const auto& old = *item.accepted;
        cartesian_detail::Allocation group(*budget_);
        std::vector<PooledLineSubscriber> values;
        if (replay_admission_observer)
        {
            struct RestoreLimit
            {
                Budget& budget;
                std::size_t limit;
                ~RestoreLimit() { budget.limit = limit; }
            } restore{*budget_, budget_->limit};
            const auto requested =
                replay_admission_observer(*budget_, array_bytes(old.size(), sizeof(PooledLineSubscriber)));
            budget_->limit = std::clamp(requested, budget_->used, restore.limit);
            group.reserve(values, old.size());
        }
        else
            group.reserve(values, old.size());
        if (!materialize(old, values, group))
            throw std::invalid_argument("Source line replay lost a completed real query result.");
        if (!item.replacement->source_ranges->exact_replay(values, *pools_))
            throw std::invalid_argument("Source line ranges differ from the actual complete ordered output.");
    }
    MutationGuard mutation{*this};
    for (const auto& item : pending)
    {
        const auto* accepted = exact(item.key);
        if (!accepted || accepted->bucket != item.accepted)
            throw std::logic_error("Source line witness changed before certification publication.");
        edit_witness(item.key, &item.replacement);
        ++certified_witnesses_;
    }
    finish_source_selection();
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
