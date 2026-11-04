#pragma once
#include "IncidencePools.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Per-cell point refcounts keyed by the session's exact point pool slots.
 * Retired incidence slots stay pooled and charged until session destruction. */
class PooledCellPoints
{
public:
    using Slot = std::size_t;
    PooledCellPoints(Budget&, IncidencePools&);
    PooledCellPoints(const PooledCellPoints&, Budget&, IncidencePools&);
    ~PooledCellPoints();
    PooledCellPoints(const PooledCellPoints&) = delete;
    PooledCellPoints& operator=(const PooledCellPoints&) = delete;
    void begin_delta();
    void finish_delta();
    void add(Slot);
    void remove(Slot);
    [[nodiscard]] bool changed() const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charged_; }
    /** The returned row is charged to the arena and charge ownership transfers
     * to the caller. Temporary normalization/audit storage is scoped here. */
    [[nodiscard]] ContactIncidenceRow row(const AdaptiveLeaf&, const AdaptiveProviderKernel&);

private:
    void admit(std::size_t);
    void release(std::size_t);
    Budget* budget_;
    IncidencePools* pools_;
    std::size_t charged_ = 0;
    std::vector<std::pair<Slot, std::size_t>> counts_;
    std::vector<Slot> before_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
