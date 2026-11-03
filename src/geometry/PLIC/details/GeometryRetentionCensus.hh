/** Private, quiescent, observational retention probe. No owner pins escape. */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <vector>
namespace SimpleFluid::PLIC::details::retention
{
enum class RootKind : std::uint8_t
{
    TypedGeometry,
    DynamicGeometry,
    DynamicPoint,
    TypedPoint,
    Snapshot,
    TypedExport,
    DynamicExport,
    Context,
    OrdinaryPlanarFace,
    IdentityBacking,
    Authority,
    OrdinaryBoundary
};
struct BorrowedRoot
{
    RootKind kind;
    const void* object = nullptr;
    ArithmeticMode mode = ArithmeticMode::Ordinary;
    std::uint32_t group = 0;
};
template<ArithmeticMode M> BorrowedRoot root(const ConvexCellGeometry<M>& p, std::uint32_t g = 0)
{
    return {RootKind::TypedGeometry, &p, M, g};
}
inline BorrowedRoot root(const DynamicConvexGeometry& p, std::uint32_t g = 0)
{
    return {RootKind::DynamicGeometry, &p, p.arithmetic_mode(), g};
}
inline BorrowedRoot root(const DynamicPoint& p, std::uint32_t g = 0)
{
    return {RootKind::DynamicPoint, &p, p.arithmetic_mode, g};
}
template<ArithmeticMode M> BorrowedRoot root(const Point<M>& p, std::uint32_t g = 0)
{
    return {RootKind::TypedPoint, &p, M, g};
}
template<ArithmeticMode M> BorrowedRoot root(const PointSnapshot<M>& p, std::uint32_t g = 0)
{
    return {RootKind::Snapshot, &p, M, g};
}
template<ArithmeticMode M> BorrowedRoot root(const PointExport<M>& p, std::uint32_t g = 0)
{
    return {RootKind::TypedExport, &p, M, g};
}
inline BorrowedRoot root(const DynamicPointExport& p, std::uint32_t g = 0)
{
    return {RootKind::DynamicExport, &p, ArithmeticMode::Ordinary, g};
}
BorrowedRoot root(const PointSnapshotContextBase&, std::uint32_t group = 0);
template<GeometryIdentityKind K, ArithmeticMode M>
BorrowedRoot root(const GeometryIdentity<K, M>& p, std::uint32_t g = 0)
{
    if constexpr (GeometryIdentity<K, M>::shared_storage)
        return {RootKind::IdentityBacking, GeometryIdentityAccess::backing(p).get(), M, g};
    else
        return {RootKind::IdentityBacking, nullptr, M, g};
}
inline BorrowedRoot root(const PlaneAuthority& p, std::uint32_t g = 0)
{
    return {RootKind::Authority, &p, ArithmeticMode::Ordinary, g};
}
inline BorrowedRoot root(const OrdinaryBoundary& p, std::uint32_t g = 0)
{
    return {RootKind::OrdinaryBoundary, &p, ArithmeticMode::Ordinary, g};
}
inline BorrowedRoot root(const PlanarFace<ArithmeticMode::Ordinary>& p, std::uint32_t g = 0)
{
    return {RootKind::OrdinaryPlanarFace, &p, ArithmeticMode::Ordinary, g};
}
/** Caller keeps roots alive and quiescent only during inspection. Scratch must
 * outlive the report; reset() is valid only after report destruction. */
class ProbeScratch final : public std::pmr::memory_resource
{
public:
    explicit ProbeScratch(std::span<std::byte> bytes) noexcept : bytes_(bytes) {}
    std::size_t used_bytes() const noexcept { return used_; }
    std::size_t capacity_bytes() const noexcept { return bytes_.size(); }
    void reset() noexcept { used_ = 0; }

private:
    void* do_allocate(std::size_t, std::size_t) override;
    void do_deallocate(void*, std::size_t, std::size_t) override {}
    bool do_is_equal(const std::pmr::memory_resource& p) const noexcept override { return this == &p; }
    std::span<std::byte> bytes_;
    std::size_t used_ = 0;
};
enum class Status : std::uint8_t
{
    Complete,
    ScratchExhausted,
    InvalidGraph,
    IntegerOverflow
};
struct ProbeLimits
{
    std::size_t maximum_owners = 1000000, maximum_records = 4000000;
    bool complete_root_manifest = false;
};
struct ByteCount
{
    std::size_t owners = 0, bytes = 0;
};
struct OwnerRow
{
    std::size_t id = 0;
    const char* kind = "";
    const char* recipe_kind = "";
    std::size_t depth = 0, dependency_depth = 0;
    ArithmeticMode mode = ArithmeticMode::Ordinary;
    std::size_t bytes = 0, metadata_bytes = 0, populated_bytes = 0, unused_capacity_bytes = 0;
    std::uint64_t root_groups = 0;
    bool application = false, ledger = false, required = false;
};
struct ContextRow
{
    std::size_t owner_id = 0;
    ArithmeticMode mode = ArithmeticMode::Ordinary;
    std::array<std::size_t, 4> retained_records{}, required_records{};
    std::size_t authorities = 0, required_authorities = 0, incidence = 0, required_incidence = 0;
    std::size_t edges = 0, required_edges = 0, fact_rows = 0, required_fact_rows = 0;
};
struct GroupRow
{
    std::uint32_t group = 0;
    std::size_t roots = 0, inclusive_bytes = 0, exclusive_bytes = 0;
};
struct Report
{
    explicit Report(ProbeScratch& scratch) : owners(&scratch), contexts(&scratch), groups(&scratch) {}
    Status status = Status::Complete;
    const char* detail = "complete";
    bool complete_root_manifest = false;
    std::size_t root_count = 0, ledger_count = 0, failed_ledgers = 0, pending_pins = 0;
    ByteCount application_only, intersection, ledger_only, owner_union, required_ledger_only, unneeded_ledger_only;
    std::size_t application_bytes = 0, ledger_bytes = 0, group_overlap_bytes = 0;
    std::size_t required_record_bytes_lower_bound = 0, required_records = 0, retained_records = 0;
    std::size_t uncovered_required_owners = 0, maximum_recipe_depth = 0, maximum_identity_depth = 0;
    /** Dependency depths count edges; an isolated root has depth zero. Point
     * dependencies are validated only within the required record closure. */
    std::size_t maximum_owner_dependency_depth = 0, maximum_context_dependency_depth = 0;
    std::size_t maximum_required_point_dependency_depth = 0, maximum_identity_dependency_depth = 0;
    std::size_t maximum_ancestry_dependency_depth = 0;
    std::size_t ledger_tracking_bytes = 0, ledger_payload_admission = 0, ledger_bookkeeping_admission = 0;
    std::size_t probe_used_bytes = 0, probe_capacity_bytes = 0;
    std::pmr::vector<OwnerRow> owners;
    std::pmr::vector<ContextRow> contexts;
    std::pmr::vector<GroupRow> groups;
    bool complete() const noexcept { return status == Status::Complete; }
};
[[nodiscard]] SIMPLEFLUID_LOCAL Report inspect_retention(std::span<const BorrowedRoot>,
    std::span<const GeometryAllocationLedger* const>, ProbeScratch&, ProbeLimits = {}) noexcept;
} // namespace SimpleFluid::PLIC::details::retention
