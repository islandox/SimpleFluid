/** @file AdaptiveContactQuery.hh
 * @brief Borrowed exact witnesses of complete owner-local closed incidence.
 */
#pragma once
#include "geometry/mesh/AMR/ContactPointArray.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <functional>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** One occurrence per neighbor and canonical contact point. The two physical
 * points are in their respective leaf frames, including at periodic seams.
 * Borrowed key and witness values are valid for the duration of the callback. */
struct ContactWitness
{
    const AdaptiveCellKey& neighbor;
    int owner;
    std::uint64_t revision;
    std::span<const char> domain;
    ContactPointIdentity normalized, local_physical, neighbor_physical;
};
using ContactWitnessVisitor = std::function<void(const ContactWitness&)>;

/** Visit the exact fan incidences of a row. The caller selects a mutable-session
 * or frozen lookup, and decides how to handle and order occurrence locators.
 * In particular, contact pairs are never inferred from keys or logical bounds. */
template<class FanAt, class Visit>
void visit_contact_occurrences(
    std::size_t local_record, const ContactPointArray& points, FanAt&& fan_at, Visit&& visit)
{
    for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
    {
        const auto point = points.at(point_index);
        fan_at(point_index, point.normalized,
            [&](const auto& group, const auto& fan)
            {
                for (std::size_t i = 0; i < fan.local_count(); ++i)
                {
                    const auto occurrence = fan.local(i);
                    if (occurrence.record() != local_record)
                        visit(group, point_index, point, occurrence.record(), i, false, fan);
                }
                for (std::size_t i = 0; i < fan.remote_count(); ++i)
                {
                    const auto record = fan.remote_record(i);
                    visit(group, point_index, point, record, i, true, fan);
                }
            });
    }
}
template<class FanAt, class RecordAt>
void visit_closed_contact_witnesses(std::size_t local, const ContactPointArray& points,
    FanAt&& fan_at, RecordAt&& record_at, const ContactWitnessVisitor& visitor)
{
    visit_contact_occurrences(local, points, std::forward<FanAt>(fan_at),
        [&](const auto&, std::size_t, const auto& point, std::size_t neighbor, std::size_t occurrence, bool remote,
            const auto& fan)
        {
            const auto& other = record_at(neighbor);
            if (other.owner < 0)
                throw std::invalid_argument("Complete closed-contact fan has an unresolved neighbor.");
            visitor({other.key(), other.owner, other.revision(), other.domain(), point.normalized, point.physical,
                remote ? fan.remote(occurrence).physical : fan.local(occurrence).physical()});
        });
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
