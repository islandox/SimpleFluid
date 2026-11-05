#include "PooledLineRegistry.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
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
constexpr std::size_t owner = 128;
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
PooledLineRegistry::PooledLineRegistry(Budget& budget, IncidencePools& pools)
    : budget_(&budget), deferred_(budget), pools_(&pools)
{
}
PooledLineRegistry::PooledLineRegistry(const PooledLineRegistry& source, Budget& budget, IncidencePools& pools)
    : budget_(&budget), deferred_(budget), pools_(&pools), layer_(source.layer_), replay_(source.replay_),
      processed_lines_(source.processed_lines_)
{
    try
    {
        std::set<const Bucket*> unique;
        std::size_t scratch = 0;
        for (const auto& [slot, bucket] : source.lines_)
        {
            admit(index_node_bytes);
            if (bucket && !unique.contains(bucket.get()))
            {
                admit(sizeof(const Bucket*) + 4 * sizeof(void*));
                scratch += sizeof(const Bucket*) + 4 * sizeof(void*);
                unique.insert(bucket.get());
                admit(bucket->charge);
            }
        }
        unique.clear();
        release(scratch);
        for (const auto slot : source.dirty_)
            admit(set_node_bytes);
        lines_ = source.lines_;
        dirty_ = source.dirty_;
        const auto arena = pools.points.source_arena();
        const bool source_ranges = std::ranges::any_of(
            lines_, [](const auto& entry) { return entry.second && bool(entry.second->source_ranges); });
        if (source_ranges)
        {
            if (!arena || arena->budget != &budget)
                throw std::invalid_argument("Source line fork requires its destination typed arena.");
            cartesian_detail::CreditConstruction credits(arena);
            cartesian_detail::Allocation memory(budget);
            admit(array_bytes(lines_.size(), sizeof(SourceOwner)));
            source_owners_.reserve(lines_.size());
            if (source_owners_.capacity() != lines_.size())
                throw std::invalid_argument("Source line owner capacity differs from admission.");
            for (const auto& [key, bucket] : lines_)
                if (bucket && bucket->source_ranges)
                {
                    const auto& recipe = bucket->source_ranges;
                    if (std::ranges::none_of(source_owners_, [&](const auto& old) { return old.lock() == recipe; }))
                        source_owners_.push_back(recipe);
                    credits.retain(recipe, recipe->allocation_budget(), recipe->metadata_bytes(),
                        recipe->has_arena_keeper(), source_leases_, memory);
                    recipe->retain_dependencies(credits, source_leases_, memory);
                }
            source_lease_bytes_ = std::exchange(memory.bytes, 0);
            credits.committed = true;
        }
    }
    catch (...)
    {
        Index().swap(lines_);
        decltype(source_owners_){}.swap(source_owners_);
        std::set<std::size_t>().swap(dirty_);
        budget_->used -= charged_;
        charged_ = 0;
        decltype(source_leases_){}.swap(source_leases_);
        budget_->used -= source_lease_bytes_;
        source_lease_bytes_ = 0;
        throw;
    }
}
PooledLineRegistry::~PooledLineRegistry()
{
    Index().swap(lines_);
    decltype(source_owners_){}.swap(source_owners_);
    std::set<std::size_t>().swap(dirty_);
    std::set<std::size_t>().swap(touched_);
    decltype(source_leases_){}.swap(source_leases_);
    budget_->used -= source_lease_bytes_;
    budget_->used -= charged_;
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
void PooledLineRegistry::reap_source_owners() noexcept
{
    source_owners_.erase(
        std::remove_if(source_owners_.begin(), source_owners_.end(), [](const auto& owner) { return owner.expired(); }),
        source_owners_.end());
    source_leases_.erase(std::remove_if(source_leases_.begin(), source_leases_.end(),
                             [](const auto& lease) { return lease->owner.expired(); }),
        source_leases_.end());
    if (const auto& arena = pools_->points.source_arena())
        arena->reap();
}
std::size_t PooledLineRegistry::retained_bytes() const noexcept
{
    std::size_t result = charged_ + deferred_.retained_bytes() + source_lease_bytes_;
    for (std::size_t i = 0; i < source_owners_.size(); ++i)
        if (const auto recipe = source_owners_[i].lock())
        {
            result += recipe->metadata_bytes();
            bool first_catalog = true;
            for (std::size_t j = 0; j < i; ++j)
                if (const auto previous = source_owners_[j].lock();
                    previous && previous->catalog() == recipe->catalog())
                {
                    first_catalog = false;
                    break;
                }
            if (first_catalog)
                result += recipe->catalog()->metadata_bytes();
        }
        else
            result += CartesianSourceLineRangeRecipe::control_metadata_bytes();
    return result;
}
std::size_t PooledLineRegistry::body(const Bucket& bucket)
{
    std::size_t result = owner;
    add_checked(result, array_bytes(bucket.subscribers.capacity(), sizeof(StoredSubscriber)));
    add_checked(result, array_bytes(bucket.runs.capacity(), sizeof(SubscriberRun)));
    add_checked(result, array_bytes(bucket.patterns.capacity(), sizeof(std::array<std::uint64_t, 6>)));
    for (const auto& sub : bucket.subscribers)
        add_checked(result, sub.spill_bytes());
    return result;
}
PooledLineRegistry::ActiveStorage PooledLineRegistry::active_storage() const
{
    ActiveStorage result;
    for (auto entry = lines_.begin(); entry != lines_.end(); ++entry)
        if (entry->second)
        {
            bool seen = false;
            for (auto previous = lines_.begin(); previous != entry; ++previous)
                seen = seen || previous->second == entry->second;
            if (!seen)
            {
                if (entry->second->source_ranges)
                    result.retained_source_records += entry->second->size();
                else if (entry->second->runs.empty())
                    result.retained_explicit_records += entry->second->size();
                else
                    result.retained_run_records += entry->second->size();
            }
        }
    for (auto next = lines_.begin(); next != lines_.end();)
    {
        const auto line = next->first.first;
        const auto current = effective(line, layer_);
        if (current != lines_.end() && current->second)
        {
            ++result.lines;
            const auto& bucket = *current->second;
            if (bucket.source_ranges)
            {
                const auto& recipe = bucket.source_ranges;
                result.source_records += recipe->size();
                result.source_families += recipe->family_count();
                result.source_nonforward_records += recipe->nonforward_count();
                result.source_result_selectors += recipe->result_selector_count();
                result.source_bytes += body(bucket) + recipe->metadata_bytes();
                bool first_catalog = true;
                for (auto old = lines_.begin(); old != lines_.end() && old->first.first < line;)
                {
                    const auto previous_line = old->first.first;
                    const auto previous = effective(previous_line, layer_);
                    if (previous != lines_.end() && previous->second && previous->second->source_ranges &&
                        previous->second->source_ranges->catalog() == recipe->catalog())
                    {
                        first_catalog = false;
                        break;
                    }
                    old = lines_.upper_bound({previous_line, std::numeric_limits<unsigned>::max()});
                }
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
        }
        next = lines_.upper_bound({line, std::numeric_limits<unsigned>::max()});
    }
    return result;
}
PooledLineRegistry::Index::const_iterator PooledLineRegistry::effective(std::size_t line, unsigned layer) const
{
    auto position = lines_.upper_bound({line, layer});
    if (position == lines_.begin())
        return lines_.end();
    --position;
    return position->first.first == line ? position : lines_.end();
}
void PooledLineRegistry::begin_replay(bool bootstrap)
{
    reap_source_owners();
    if (!dirty_.empty())
        throw std::invalid_argument("Unfinished line replay.");
    release(touched_.size() * set_node_bytes);
    touched_.clear();
    replay_ = !bootstrap;
    layer_ = 0;
}
void PooledLineRegistry::set_layer(unsigned layer)
{
    if (!dirty_.empty() || layer < layer_ || layer >= 64)
        throw std::invalid_argument("Invalid incremental line layer.");
    layer_ = layer;
}
void PooledLineRegistry::finish_replay(unsigned last_layer)
{
    reap_source_owners();
    if (!dirty_.empty())
        throw std::invalid_argument("Unfinished line replay.");
    for (const auto line : touched_)
    {
        auto it = lines_.upper_bound({line, last_layer});
        while (it != lines_.end() && it->first.first == line)
        {
            const auto bytes = it->second ? it->second->charge : 0;
            const bool deferred = it->second && (it->second.use_count() > 1 || deferred_.contains(it->second));
            if (deferred && deferred_.adopt(it->second, bytes))
                charged_ -= bytes;
            it = lines_.erase(it);
            release(index_node_bytes);
            if (bytes && !deferred)
                release(bytes);
        }
    }
    release(touched_.size() * set_node_bytes);
    touched_.clear();
    reap_source_owners();
}
void PooledLineRegistry::preserve_future(std::size_t line)
{
    if (!replay_ || layer_ == 63 || lines_.contains({line, layer_ + 1}))
        return;
    const auto old = effective(line, layer_);
    admit(index_node_bytes);
    lines_.emplace(Key{line, layer_ + 1}, old == lines_.end() ? nullptr : old->second);
}
void PooledLineRegistry::replace(std::size_t line, std::shared_ptr<const Bucket> next)
{
    auto old = lines_.find({line, layer_});
    if (old == lines_.end())
    {
        admit(index_node_bytes);
        lines_.emplace(Key{line, layer_}, std::move(next));
        return;
    }
    const auto charge = old->second ? old->second->charge : 0;
    const bool deferred = old->second && (old->second.use_count() > 1 || deferred_.contains(old->second));
    if (deferred && deferred_.adopt(old->second, charge))
        charged_ -= charge;
    old->second = std::move(next);
    if (charge && !deferred)
        release(charge);
    reap_source_owners();
}
void PooledLineRegistry::apply(std::span<const PooledLineEdit> changes)
{
    reap_source_owners();
    for (const auto& change : changes)
    {
        (void) pools_->lines.get(change.line);
        if (!touched_.contains(change.line))
        {
            admit(set_node_bytes);
            touched_.insert(change.line);
        }
        preserve_future(change.line);
        const auto found = effective(change.line, layer_);
        const auto* old = found == lines_.end() ? nullptr : found->second.get();
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
        const auto next_count = change.remove ? count - 1 : count + 1;
        std::size_t admitted = owner;
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
        next->charge = actual;
        if (next->subscribers.empty())
        {
            next.reset();
            release(actual);
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
    reap_source_owners();
    const auto dirty = dirty_.find(line);
    if (dirty == dirty_.end())
        throw std::invalid_argument("Line is not dirty.");
    ++processed_lines_;
    const auto found = effective(line, layer_);
    if (found != lines_.end() && found->second)
    {
        const auto& old = *found->second;
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
        credit.add(owner);
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
        next->charge = actual;
        replace(line, std::move(next));
        scratch -= actual; // The immutable bucket now owns this admitted body.
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
    const auto found = effective(line, layer_);
    if (found == lines_.end() || !found->second)
        throw std::invalid_argument("Missing pooled line query result.");
    PooledLineResultView result{
        found->second->metadata(subscriber, *pools_), found->second->explicit_result(subscriber)};
    if (!result.subscriber.has_result())
        throw std::invalid_argument("Uncomputed pooled line query result.");
    if (found->second->source_ranges && result.subscriber.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        result.source_result = found->second->source_ranges.get();
        result.source_pools = pools_;
        result.source_ordinal = subscriber;
    }
    return result;
}
std::optional<PooledLineRegistry::CompletedMember> PooledLineRegistry::find_completed_member(std::size_t line,
    unsigned effective_layer, const PooledLineSubscriber& expected, const IncidencePools& request_namespace) const
{
    if (&request_namespace != pools_ || !expected.native_source || !expected.source_generation ||
        expected.has_result() || expected.result.capacity() || dirty_.contains(line))
        return {};
    CompletedMemberEnvironment environment;
    const auto found = effective(line, effective_layer);
    if (found == lines_.end() || !found->second)
        return {};
    const auto& bucket = *found->second;
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
    const auto found = effective(line, effective_layer);
    if (found == lines_.end() || !found->second || !found->second->source_ranges)
        return {};
    const auto& bucket = *found->second;
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
    deferred_.visit_borrowed_payload(visitor);
    for (const auto& entry : lines_)
    {
        contact_payload_node_value(visitor, this, ContactPayloadRole::LineRegistryNodes, entry);
        if (!entry.second)
            continue;
        const auto& bucket = *entry.second;
        contact_payload_object(visitor, &bucket, ContactPayloadRole::LineBucketObject, sizeof(bucket), true);
        contact_payload_buffer(visitor, &bucket, ContactPayloadRole::LineSubscriberValues, bucket.subscribers);
        contact_payload_buffer(visitor, &bucket, ContactPayloadRole::LineRunValues, bucket.runs);
        contact_payload_buffer(visitor, &bucket, ContactPayloadRole::LinePatternValues, bucket.patterns);
        for (const auto& subscriber : bucket.subscribers)
            if (subscriber.spill)
            {
                contact_payload_object(
                    visitor, subscriber.spill.get(), ContactPayloadRole::LineSpillObject, sizeof(SubscriberSpill));
                contact_payload_buffer(
                    visitor, subscriber.spill.get(), ContactPayloadRole::LineResultValues, subscriber.spill->result);
            }
        if (bucket.source_ranges)
            bucket.source_ranges->visit_borrowed_payload(visitor);
    }
    for (const auto& entry : dirty_)
        contact_payload_node_value(visitor, this, ContactPayloadRole::LineRegistryDirtyNodes, entry);
    for (const auto& entry : touched_)
        contact_payload_node_value(visitor, this, ContactPayloadRole::LineRegistryTouchedNodes, entry);
    contact_payload_buffer(visitor, this, ContactPayloadRole::LineSourceOwners, source_owners_);
    contact_payload_buffer(visitor, this, ContactPayloadRole::LineSourceLeases, source_leases_);
    for (const auto& owner : source_owners_)
        if (owner.control)
            contact_payload_object(
                visitor, owner.control, ContactPayloadRole::CartesianControlCreditObject, owner.control_bytes, true);
    for (const auto& lease : source_leases_)
        lease->visit_borrowed_payload(visitor);
}
bool PooledLineRegistry::imports_source_owner(const void* identity) const noexcept
{
    for (const auto& [key, bucket] : lines_)
        if (bucket && bucket->source_ranges && bucket->source_ranges->imports_owner(identity))
            return true;
    return false;
}
void PooledLineRegistry::certify_source_ranges(
    const CartesianLineRangeAdmission& admission, std::shared_ptr<CartesianAuthorityArena> arena)
{
    reap_source_owners();
    if (!dirty_.empty())
        throw std::invalid_argument("Source line ranges require a completed actual line pass.");
    CartesianSourceLineRangeBuilder builder(admission, *budget_, std::move(arena));
    struct Pending
    {
        Index::iterator position;
        std::shared_ptr<Bucket> replacement;
        bool owns_replacement = true;
    };
    cartesian_detail::Allocation scratch(*budget_);
    std::vector<Pending> pending;
    const auto materialize =
        [&](const Bucket& bucket, std::vector<PooledLineSubscriber>& values, cartesian_detail::Allocation& group)
    {
        group.reserve(values, bucket.size());
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
    for (auto current = lines_.begin(); current != lines_.end(); ++current)
        if (current->second && !current->second->source_ranges)
        {
            const auto alias = std::find_if(pending.begin(), pending.end(),
                [&](const auto& item)
                {
                    return item.position->first.first == current->first.first &&
                           item.position->second == current->second;
                });
            if (alias != pending.end())
            {
                auto replacement = alias->replacement;
                scratch.append(pending);
                pending.push_back({current, std::move(replacement), false});
                continue;
            }
            const auto& old = *current->second;
            cartesian_detail::Allocation group(*budget_);
            std::vector<PooledLineSubscriber> values;
            if (materialize(old, values, group))
                if (auto recipe = builder.make_recipe(current->first.first, values, *pools_, old.certified_layer))
                {
                    scratch.append(pending);
                    scratch.retain(owner);
                    auto replacement = std::make_shared<Bucket>();
                    replacement->source_ranges = std::move(recipe);
                    replacement->certified_layer = old.certified_layer;
                    replacement->charge = owner;
                    pending.push_back({current, std::move(replacement), true});
                }
        }
    builder.freeze();
    // A separate final pass proves every ordered scalar, FULL support identity,
    // endpoint identity and complete query result against real execution. Maps
    // are now immutable; no proof can be changed by a later bucket observation.
    for (const auto& item : pending)
    {
        const auto& old = *item.position->second;
        cartesian_detail::Allocation group(*budget_);
        std::vector<PooledLineSubscriber> values;
        if (!materialize(old, values, group))
            throw std::invalid_argument("Source line replay lost a completed real query result.");
        if (!item.replacement->source_ranges->exact_replay(values, *pools_))
            throw std::invalid_argument("Source line ranges differ from the actual complete ordered output.");
    }
    const auto new_owners = std::ranges::count_if(pending, [](const auto& item) { return item.owns_replacement; });
    const auto needed = cartesian_detail::add(source_owners_.size(), std::size_t(new_owners));
    if (needed > source_owners_.capacity())
    {
        const auto bytes = array_bytes(needed, sizeof(SourceOwner));
        const auto previous = array_bytes(source_owners_.capacity(), sizeof(SourceOwner));
        admit(bytes);
        try
        {
            source_owners_.reserve(needed);
        }
        catch (...)
        {
            release(bytes);
            throw;
        }
        if (source_owners_.capacity() != needed)
            throw std::invalid_argument("Source line owner capacity differs from admission.");
        release(previous);
    }
    // Admit deferred metadata before committing replacements. Existing keys are
    // replaced in place; this pass does not create a new fixed-point layer.
    for (const auto& item : pending)
    {
        const auto& old = item.position->second;
        if (old && (old.use_count() > 1 || deferred_.contains(old)))
            if (deferred_.adopt(old, old->charge))
                charged_ -= old->charge;
    }
    for (auto& item : pending)
    {
        auto& old = item.position->second;
        const auto old_bytes = old->charge;
        const bool deferred = deferred_.contains(old);
        if (item.owns_replacement)
            source_owners_.push_back(item.replacement->source_ranges);
        old = std::move(item.replacement);
        if (!deferred)
            release(old_bytes);
        if (item.owns_replacement)
        {
            charged_ += owner;
            scratch.bytes -= owner;
        }
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
