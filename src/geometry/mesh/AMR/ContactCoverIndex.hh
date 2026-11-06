/** @file ContactCoverIndex.hh
 * @brief Borrowed owner-local point bounds for closure ownership discovery.
 */
#pragma once
#include "ClosedContactIndex.hh"
#include "ContactPointRef.hh"
#include "utils/export_helper.hh"
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget;
class PointFactDirectory;
struct ContactGroupId;
/** Directory bindings retain only typed references; the caller keeps their
 * immutable directory generations and interned domains alive until erase().
 * Explicit coordinate pointer bindings also require stable addresses. Handles survive
 * metadata growth and copies. No geometry or provenance is retained here.
 * The shared budget must outlive this object, which releases its live charge.
 */
class SIMPLEFLUID_LOCAL ContactCoverIndex
{
public:
    using Handle = std::size_t;
    explicit ContactCoverIndex(Budget&, int owner);
    struct GroupObservation
    {
        std::array<long double, 3> coordinates;
        const AdaptiveBytes* domain;
    };
    struct GroupResolver
    {
        const void* state;
        GroupObservation (*get)(const void*, ContactGroupId);
    };
    ContactCoverIndex(std::shared_ptr<Budget>, int owner, GroupResolver);
    ContactCoverIndex(const ContactCoverIndex&, std::shared_ptr<Budget>, GroupResolver);
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
    [[nodiscard]] Handle insert(const PointFactDirectory&, ContactPointRef, const AdaptiveBytes* domain);
    [[nodiscard]] Handle insert(ContactGroupId);
    void erase(Handle);
    /** Allocation-free rebinding. The caller proves coordinates and complete
     * domain bytes equal the former binding; old pointers are never read. */
    void rebind(Handle, const std::array<long double, 3>*, const AdaptiveBytes* domain);
    void rebind(Handle, const PointFactDirectory&, ContactPointRef, const AdaptiveBytes* domain);
    [[nodiscard]] std::optional<AdaptiveContactBounds> aggregate_bounds() const noexcept;
    /** Actual homogeneous-domain subtrees, refined toward a soft target.
     * Mixed domains always split, even beyond the target. Returned vector and
     * domain capacities are already charged and become the caller's charge. */
    [[nodiscard]] std::vector<ClosedContactCover> ownership_frontier_covers(std::size_t target_fragments = 8);
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] std::size_t point_count() const noexcept;
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t overlap_bytes() const noexcept;
    [[nodiscard]] std::size_t copied_pages() const noexcept;
    void finish_overlap() noexcept;
    void visit_borrowed_pages(const std::function<void(const void*, bool, std::size_t)>&) const;

private:
    Handle insert_binding(
        const std::array<long double, 3>*, const PointFactDirectory*, ContactPointRef, const AdaptiveBytes* domain);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
