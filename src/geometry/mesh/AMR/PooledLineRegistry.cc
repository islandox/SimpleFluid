#include "PooledLineRegistry.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
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
} // namespace
bool PooledLineRegistry::StoredSubscriber::needs_wide(const PooledLineSubscriber& sub) noexcept
{
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    return sub.support > max || sub.first > max || sub.last > max || sub.emission > max || sub.edge > max;
}
PooledLineSubscriber PooledLineRegistry::StoredSubscriber::metadata() const
{
    PooledLineSubscriber result;
    result.support = wide ? spill->support : support;
    result.first = wide ? spill->first : first;
    result.last = wide ? spill->last : last;
    result.emission = wide ? spill->emission : emission;
    result.edge = wide ? spill->edge : edge;
    result.resident = resident;
    result.support_authority = support_authority;
    result.image = image;
    result.result_state = result_state;
    return result;
}
PooledLineRegistry::StoredSubscriber PooledLineRegistry::StoredSubscriber::pack(PooledLineSubscriber&& sub)
{
    StoredSubscriber result;
    result.resident = sub.resident;
    result.support_authority = sub.support_authority;
    result.image = sub.image;
    result.result_state = sub.result_state;
    result.wide = needs_wide(sub);
    if (result.wide || sub.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        result.spill = std::make_unique<SubscriberSpill>();
        result.spill->result = std::move(sub.result);
    }
    if (result.wide)
    {
        result.spill->support = sub.support;
        result.spill->first = sub.first;
        result.spill->last = sub.last;
        result.spill->emission = sub.emission;
        result.spill->edge = sub.edge;
    }
    else
    {
        result.support = static_cast<std::uint32_t>(sub.support);
        result.first = static_cast<std::uint32_t>(sub.first);
        result.last = static_cast<std::uint32_t>(sub.last);
        result.emission = static_cast<std::uint32_t>(sub.emission);
        result.edge = static_cast<std::uint32_t>(sub.edge);
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
    }
    catch (...)
    {
        Index().swap(lines_);
        std::set<std::size_t>().swap(dirty_);
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
PooledLineRegistry::~PooledLineRegistry()
{
    Index().swap(lines_);
    std::set<std::size_t>().swap(dirty_);
    std::set<std::size_t>().swap(touched_);
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
std::size_t PooledLineRegistry::body(const Bucket& bucket)
{
    std::size_t result = owner;
    add_checked(result, array_bytes(bucket.subscribers.capacity(), sizeof(StoredSubscriber)));
    for (const auto& sub : bucket.subscribers)
        add_checked(result, sub.spill_bytes());
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
}
void PooledLineRegistry::apply(std::span<const PooledLineEdit> changes)
{
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
        const auto count = old ? old->subscribers.size() : 0;
        if (!change.remove && count == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Pooled subscriber count overflow.");
        std::size_t position = 0;
        bool same = false;
        if (old)
        {
            const auto at = std::lower_bound(old->subscribers.begin(), old->subscribers.end(), change.subscriber,
                [&](const auto& a, const auto& b) { return line_subscriber_before(a.metadata(), b, *pools_); });
            position = std::size_t(at - old->subscribers.begin());
            same = at != old->subscribers.end() && !line_subscriber_before(change.subscriber, at->metadata(), *pools_);
        }
        if (change.remove && !same)
            throw std::invalid_argument("Unknown retired line subscription.");
        if (!change.remove && (same || change.subscriber.has_result() || !change.subscriber.result.empty()))
            throw std::invalid_argument("Duplicate or prepopulated line subscription.");
        const auto next_count = change.remove ? count - 1 : count + 1;
        std::size_t admitted = owner;
        add_checked(admitted, array_bytes(next_count, sizeof(StoredSubscriber)));
        if (old)
            for (std::size_t i = 0; i < count; ++i)
                if (!change.remove || i != position)
                    add_checked(admitted, old->subscribers[i].spill_bytes());
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
                next->subscribers.push_back(old->subscribers[i].clone());
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
void PooledLineRegistry::recompute_one(std::size_t line, const PooledChangedLineQuery& publish)
{
    const auto dirty = dirty_.find(line);
    if (dirty == dirty_.end())
        throw std::invalid_argument("Line is not dirty.");
    ++processed_lines_;
    const auto found = effective(line, layer_);
    if (found != lines_.end() && found->second)
    {
        const auto& old = *found->second;
        const auto count = old.subscribers.size();
        std::size_t scratch = array_bytes(count, sizeof(PooledLineSubscriber));
        for (const auto& sub : old.subscribers)
            if (sub.spill)
                add_checked(scratch, array_bytes(sub.spill->result.size(), sizeof(std::size_t)));
        admit(scratch);
        std::vector<PooledLineSubscriber> working;
        working.reserve(count);
        if (working.capacity() != count)
            throw std::invalid_argument("Pooled query capacity differs from admission.");
        for (const auto& sub : old.subscribers)
        {
            auto transient = sub.metadata();
            if (sub.spill && !sub.spill->result.empty())
            {
                transient.result.reserve(sub.spill->result.size());
                if (transient.result.capacity() != sub.spill->result.size())
                    throw std::invalid_argument("Pooled query result capacity differs from admission.");
                transient.result.insert(transient.result.end(), sub.spill->result.begin(), sub.spill->result.end());
            }
            working.push_back(std::move(transient));
        }
        std::vector<std::size_t> changed;
        query_pooled_line(
            working, *pools_,
            [&](std::size_t n)
            {
                admit(n);
                add_checked(scratch, n);
            },
            [&](std::size_t i, std::span<const std::size_t>)
            {
                if (changed.size() == changed.capacity())
                {
                    if (changed.capacity() > std::numeric_limits<std::size_t>::max() / 2)
                        throw std::invalid_argument("Pooled changed-query capacity overflow.");
                    const auto next = changed.capacity() ? changed.capacity() * 2 : 1;
                    const auto bytes = array_bytes(next, sizeof(std::size_t));
                    admit(bytes);
                    add_checked(scratch, bytes);
                    changed.reserve(next);
                    if (changed.capacity() != next)
                        throw std::invalid_argument("Pooled changed-query capacity differs from admission.");
                }
                changed.push_back(i);
            });
        // The query's explicit result buffers already have scratch credit.
        // Admit only packed metadata here, then move those buffers into the
        // new immutable bucket before retiring the complete-line workspace.
        std::size_t packed = owner;
        add_checked(packed, array_bytes(count, sizeof(StoredSubscriber)));
        for (const auto& sub : working)
            if (StoredSubscriber::needs_wide(sub) || sub.result_state == PooledLineSubscriber::ResultState::Explicit)
                add_checked(packed, sizeof(SubscriberSpill));
        admit(packed);
        add_checked(scratch, packed);
        auto next = std::make_shared<Bucket>();
        next->subscribers.reserve(count);
        if (next->subscribers.capacity() != count)
            throw std::invalid_argument("Pooled subscriber capacity differs from admission.");
        for (auto& sub : working)
            next->subscribers.push_back(StoredSubscriber::pack(std::move(sub)));
        const auto actual = body(*next);
        if (actual > scratch)
            throw std::invalid_argument("Pooled line query exceeded its admission.");
        next->charge = actual;
        replace(line, std::move(next));
        for (const auto index : changed)
            publish(line, index);
        std::vector<PooledLineSubscriber>().swap(working);
        std::vector<std::size_t>().swap(changed);
        release(scratch - actual);
    }
    dirty_.erase(dirty);
    release(set_node_bytes);
}
PooledLineResultView PooledLineRegistry::query_result(std::size_t line, std::size_t subscriber) const
{
    const auto found = effective(line, layer_);
    if (found == lines_.end() || !found->second)
        throw std::invalid_argument("Missing pooled line query result.");
    const auto& stored = found->second->subscribers.at(subscriber);
    PooledLineResultView result{stored.metadata(), {}};
    if (!result.subscriber.has_result())
        throw std::invalid_argument("Uncomputed pooled line query result.");
    if (stored.spill)
        result.explicit_result = stored.spill->result;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
