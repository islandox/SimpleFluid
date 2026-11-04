#include "CompleteLineQuery.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
void query_complete_line(std::span<const LineSubscription> subscriptions, const LineCharge& charge,
    const LineQueryEmit& emit)
{
    // The old authority first interns full identities and then retains their
    // first occurrence order; EdgePointIndex independently interns coordinates.
    std::set<LinePoint> unique;
    std::vector<LinePoint> ordered;
    for (const auto& subscription : subscriptions)
        for (const auto* point : {&subscription.first, &subscription.last})
        {
            if (unique.find(*point) != unique.end())
                continue;
            charge(sizeof(LinePoint) + point->payload_bytes() + 4 * sizeof(void*));
            unique.insert(*point);
            // This conservative bound includes old/new vector allocation and
            // identity owner copies. Caller must release it at group teardown.
            if (ordered.size() == ordered.capacity())
            {
                const auto old = ordered.capacity();
                if (old > std::numeric_limits<std::size_t>::max() / 2 / sizeof(LinePoint))
                    throw std::invalid_argument("Edge endpoint capacity overflow.");
                const auto next = old ? old * 2 : 1;
                charge(next * sizeof(LinePoint));
                ordered.reserve(next);
                if (ordered.capacity() != next)
                    throw std::invalid_argument("Edge endpoint capacity differs from admission.");
            }
            charge(point->payload_bytes());
            ordered.push_back(*point);
        }
    // Use the same conservative index envelope as the existing authority.
    if (ordered.size() > std::numeric_limits<std::size_t>::max() / sizeof(LinePoint))
        throw std::invalid_argument("Edge index capacity overflow.");
    std::size_t index_bound = ordered.size() * sizeof(LinePoint);
    charge(index_bound);
    for (const auto& point : ordered)
    {
        const auto encoded = PLIC::encoded_identity_size(point);
        if (point.payload_bytes() > std::numeric_limits<std::size_t>::max() - 256 ||
            encoded > (std::numeric_limits<std::size_t>::max() - point.payload_bytes() - 256) / 4)
            throw std::invalid_argument("Edge index capacity overflow.");
        const auto extra = point.payload_bytes() + 4 * encoded + 256;
        if (extra > std::numeric_limits<std::size_t>::max() - index_bound)
            throw std::invalid_argument("Edge index capacity overflow.");
        index_bound += extra;
        charge(extra);
    }
    PLIC::EdgePointIndex<ArithmeticMode::Ordinary> index(ordered);
    std::size_t actual = ordered.capacity() * sizeof(LinePoint);
    for (const auto& point : ordered)
    {
        if (point.payload_bytes() > std::numeric_limits<std::size_t>::max() - actual)
            throw std::invalid_argument("Edge index capacity overflow.");
        actual += point.payload_bytes();
    }
    if (index.storage_bytes() > std::numeric_limits<std::size_t>::max() - actual)
        throw std::invalid_argument("Edge index capacity overflow.");
    actual += index.storage_bytes();
    if (actual > index_bound)
        charge(actual - index_bound);
    PLIC::OrdinaryEdgeQueryScratch scratch;
    const auto query_peak = scratch.reserve_required_peak(unique.size());
    charge(query_peak);
    scratch.reserve(unique.size(), query_peak);
    for (std::size_t i = 0; i < subscriptions.size(); ++i)
    {
        index.closed_segment(subscriptions[i].first, subscriptions[i].last, scratch);
        emit(i, scratch);
        scratch.clear();
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
