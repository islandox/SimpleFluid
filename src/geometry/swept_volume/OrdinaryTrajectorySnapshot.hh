/** @file OrdinaryTrajectorySnapshot.hh
 * @brief Immutable canonical-node Ordinary trajectories shared by finite recipes.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include <map>
#include <span>
#include <vector>
namespace SimpleFluid::geometry::swept_volume
{
class SIMPLEFLUID_FVM_EXPORT OrdinaryTrajectorySnapshot
{
public:
    using Point = PLIC::Point<ArithmeticMode::Ordinary>;
    struct Entry
    {
        std::uint64_t node;
        Point current, departure;
    };
    using TraceMap = std::map<std::uint64_t, std::pair<Point, Point>>;
    /** Canonical node identity, rather than coordinate equality, defines entries.
     * This geometry owner grants no transport transaction authority. */
    static std::shared_ptr<const OrdinaryTrajectorySnapshot> create(std::vector<Entry> entries);
    /** Freeze a complete caller catalog. Admit the old map, new owner and
     * buffer before any noexcept point transfer; admission is cumulative. */
    static std::shared_ptr<const OrdinaryTrajectorySnapshot> freeze(
        TraceMap& traces, std::size_t& used, std::size_t limit);
    [[nodiscard]] std::span<const Entry> entries() const& noexcept { return entries_; }
    std::span<const Entry> entries() const&& = delete;
    [[nodiscard]] const Entry& at(std::size_t index) const { return entries_.at(index); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t slot(std::uint64_t node) const;
    /** Retained payload: owner object metadata and actual entry capacity.
     * Allocator and shared-ownership control-block overhead are excluded,
     * following the geometry payload accounting convention. */
    [[nodiscard]] std::size_t storage_bytes() const noexcept
    {
        return sizeof(*this) + entries_.capacity() * sizeof(Entry);
    }
    void retain_provenance(PLIC::GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit) const;

private:
    friend struct Work;
    explicit OrdinaryTrajectorySnapshot(std::vector<Entry> entries) : entries_(std::move(entries)) {}
    std::vector<Entry> entries_;
};
} // namespace SimpleFluid::geometry::swept_volume
