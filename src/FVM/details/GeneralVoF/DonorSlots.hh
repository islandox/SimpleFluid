/** @file DonorSlots.hh
 * @brief Compact lookup from full donor IDs to unchanged local donor slots.
 */
#pragma once
#include "geometry/swept_volume/FiniteVoFTransport.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include <numeric>

namespace SimpleFluid::VoF::general_vof_detail
{
/** Build only a permutation. Numerical donor and quota order stays unchanged;
 * duplicate donor validation remains with OrdinaryAggregateSource::measure. */
inline std::vector<std::size_t> make_donor_slot_order(
    std::span<const geometry::swept_volume::OrdinaryAggregateDonor> donors, geometry::swept_volume::Work& work)
{
    std::vector<std::size_t> slots;
    if (donors.size() > work.limits.maximum_atoms || donors.size() > slots.max_size() ||
        donors.size() > std::numeric_limits<std::size_t>::max() / sizeof(std::size_t))
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Aggregate donor-slot lookup exceeds its local capacity.");
    work.charge(donors.size() * sizeof(std::size_t));
    slots.resize(donors.size());
    work.charge((slots.capacity() - donors.size()) * sizeof(std::size_t));
    std::iota(slots.begin(), slots.end(), std::size_t{});
    std::sort(slots.begin(), slots.end(), [&](std::size_t a, std::size_t b)
        { return donors[a].id != donors[b].id ? donors[a].id < donors[b].id : a < b; });
    return slots;
}

/** The donor IDs and permutation must remain unchanged after construction.
 * No pointer or reference is retained beyond this lookup. */
inline std::size_t donor_slot(std::span<const geometry::swept_volume::OrdinaryAggregateDonor> donors,
    std::span<const std::size_t> slots, std::uint64_t id)
{
    if (slots.size() != donors.size())
        throw std::invalid_argument("Aggregate donor-slot lookup has a foreign domain.");
    const auto donor_id = [&](std::size_t slot)
    {
        if (slot >= donors.size())
            throw std::out_of_range("Aggregate donor slot is outside its local domain.");
        return donors[slot].id;
    };
    const auto found = std::lower_bound(slots.begin(), slots.end(), id,
        [&](std::size_t slot, std::uint64_t wanted) { return donor_id(slot) < wanted; });
    if (found == slots.end() || donor_id(*found) != id)
        throw std::out_of_range("Aggregate donor has no phase-quota slot.");
    return *found;
}
} // namespace SimpleFluid::VoF::general_vof_detail
