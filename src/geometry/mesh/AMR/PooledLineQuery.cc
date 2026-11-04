#include "PooledLineQuery.hh"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <tuple>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
bool line_subscriber_before(const PooledLineSubscriber& a, const PooledLineSubscriber& b, const IncidencePools& pools)
{
    if (a.resident != b.resident)
        return a.resident < b.resident;
    if (a.support_authority != b.support_authority)
        return a.support_authority < b.support_authority;
    const auto& as = pools.supports.get(a.support);
    const auto& bs = pools.supports.get(b.support);
    if (as != bs)
        return as < bs;
    return std::tie(a.emission, a.image, a.edge) < std::tie(b.emission, b.image, b.edge);
}
void query_pooled_line(std::span<PooledLineSubscriber> subscribers, IncidencePools& pools, const LineCharge& charge,
    const PooledLineChange& publish)
{
    for (std::size_t i = 1; i < subscribers.size(); ++i)
        if (!line_subscriber_before(subscribers[i - 1], subscribers[i], pools))
            throw std::invalid_argument("Unordered or duplicate pooled line subscriptions.");
    charge(subscribers.size() * sizeof(LineSubscription));
    std::vector<LineSubscription> segments;
    segments.reserve(subscribers.size());
    for (const auto& subscriber : subscribers)
        segments.push_back({pools.points.get(subscriber.first), pools.points.get(subscriber.last)});
    query_complete_line(segments, charge,
        [&](std::size_t i, const PLIC::OrdinaryEdgeQueryScratch& scratch)
        {
            auto& subscriber = subscribers[i];
            const auto same = [&](std::span<const std::size_t> points)
            {
                if (!subscriber.has_result() || subscriber.result_size() != points.size())
                    return false;
                for (std::size_t point = 0; point < points.size(); ++point)
                    if (subscriber.result_at(point) != points[point])
                        return false;
                return true;
            };
            if (scratch.size() == 2)
            {
                const std::array endpoints{
                    pools.points.intern(scratch.point(0)), pools.points.intern(scratch.point(1))};
                if (endpoints[0] == subscriber.first && endpoints[1] == subscriber.last)
                {
                    const bool changed = !same(endpoints);
                    std::vector<std::size_t>().swap(subscriber.result);
                    subscriber.result_state = PooledLineSubscriber::ResultState::ForwardEndpoints;
                    if (changed)
                        publish(i, endpoints);
                    return;
                }
            }
            charge(scratch.size() * sizeof(std::size_t));
            std::vector<std::size_t> next;
            next.reserve(scratch.size());
            for (std::size_t j = 0; j < scratch.size(); ++j)
                next.push_back(pools.points.intern(scratch.point(j)));
            if (same(next))
                return;
            subscriber.result = std::move(next);
            subscriber.result_state = PooledLineSubscriber::ResultState::Explicit;
            publish(i, subscriber.result);
        });
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
