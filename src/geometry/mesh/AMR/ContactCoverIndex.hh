/** @file ContactCoverIndex.hh
 * @brief Borrowed owner-local point bounds for closure ownership discovery.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/AMR/ClosedContactIndex.hh"
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget;
/** The caller owns distinct normalized point coordinates and interned domains.
 * Their addresses and values remain stable until erase(). Handles survive
 * metadata growth and copies. No geometry or provenance is retained here.
 * The shared budget must outlive this object, which releases its live charge.
 */
class SIMPLEFLUID_LOCAL ContactCoverIndex
{
public:
    using Handle = std::size_t;
    explicit ContactCoverIndex(Budget&, int owner);
    ~ContactCoverIndex();
    /** Copy metadata once. Before reading or mutating the copy, rebind every
     * live handle to the copy's point/domain owners. The source's borrowed
     * pointers are copied without dereferencing them. */
    ContactCoverIndex(const ContactCoverIndex&, Budget&);
    ContactCoverIndex(const ContactCoverIndex&) = delete;
    ContactCoverIndex& operator=(const ContactCoverIndex&) = delete;
    /** Precharge exact arrays for at least this many simultaneously live
     * points. Zero needs no arrays; existing capacity is never shrunk. */
    void reserve(std::size_t maximum_points);
    [[nodiscard]] Handle insert(const std::array<long double, 3>*, const AdaptiveBytes* domain);
    void erase(Handle);
    /** Allocation-free rebinding. The caller proves coordinates and complete
     * domain bytes equal the former binding; old pointers are never read. */
    void rebind(Handle, const std::array<long double, 3>*, const AdaptiveBytes* domain);
    [[nodiscard]] std::optional<AdaptiveContactBounds> aggregate_bounds() const noexcept;
    /** Actual homogeneous-domain subtrees, refined toward a soft target.
     * Mixed domains always split, even beyond the target. Returned vector and
     * domain capacities are already charged and become the caller's charge. */
    [[nodiscard]] std::vector<ClosedContactCover> ownership_frontier_covers(std::size_t target_fragments = 8);
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] std::size_t point_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
