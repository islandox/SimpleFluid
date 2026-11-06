/** @file AdaptiveSubfaceSnapshot.hh
 * @brief Detached authenticated subfaces from the completed support and line fixed point.
 */
#pragma once
#include "AdaptiveGeometrySnapshot.hh"
#include "CartesianIncidenceSource.hh"
#include "ContactHaloContext.hh"
#include "DistributedAdaptiveTypes.hh"
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class CertifiedIncidenceLoopView;
class IdentityBudget;
class TileBodyHandle;
struct IncidencePools;
struct ReceivedTileRecipe;
struct SubfaceProvenancePage;
struct SubfacePiecePage;
struct SubfaceIndexRoot;
class SubfaceProvenanceBuilder;
class SubfacePieceBuilder;
class AdaptiveSubfaceSnapshot;
namespace cartesian_detail
{
struct CreditConstruction;
struct AuthorityOwnerCredit;
struct Allocation;
} // namespace cartesian_detail

/** Index into the immutable provenance page built by authenticated delivery. */
struct SubfaceProvenanceRef
{
    std::shared_ptr<const SubfaceProvenancePage> page;
    std::size_t slot = 0;
    explicit operator bool() const noexcept { return bool(page); }
};

/** Physical ring ownership transferred from the final explicit replay output. */
class SIMPLEFLUID_LOCAL SharedSubfaceRing
{
public:
    [[nodiscard]] static std::shared_ptr<const SharedSubfaceRing> transfer(
        std::vector<std::size_t>&, Budget&, std::shared_ptr<Budget>, std::size_t& former_charge);
    ~SharedSubfaceRing();
    SharedSubfaceRing(const SharedSubfaceRing&) = delete;
    SharedSubfaceRing& operator=(const SharedSubfaceRing&) = delete;
    [[nodiscard]] std::span<const std::size_t> slots() const noexcept { return slots_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return slots_.capacity(); }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    [[nodiscard]] const Budget* origin() const noexcept { return arena_.get(); }
    [[nodiscard]] std::size_t charged_bytes() const noexcept { return charge_; }

private:
    explicit SharedSubfaceRing(std::shared_ptr<Budget> arena) : arena_(std::move(arena)) {}
    std::shared_ptr<Budget> arena_;
    std::vector<std::size_t> slots_;
    std::size_t charge_ = 0;
};

/** A detached exact incident. Native face is present only if this resident
 * authenticated the original local source surface. */
class SIMPLEFLUID_LOCAL CanonicalSubfaceIncident
{
public:
    [[nodiscard]] const AdaptiveCellKey& key() const;
    [[nodiscard]] int owner() const;
    [[nodiscard]] std::uint64_t revision() const;
    [[nodiscard]] std::uint64_t source_surface() const;
    [[nodiscard]] std::optional<std::uint64_t> native_face() const;

private:
    friend class CanonicalSubfaceView;
    CanonicalSubfaceIncident(std::shared_ptr<const SubfaceProvenancePage> page, std::size_t slot)
        : page_(std::move(page)), slot_(slot)
    {
    }
    std::shared_ptr<const SubfaceProvenancePage> page_;
    std::size_t slot_;
};

struct MaterializedCanonicalSubface
{
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> geometry;
    AdaptiveBytes identity;
};

/** Canonical first-cell physical loop, outward from first. The view pins
 * immutable point/provenance pages and is independent of the session. */
class SIMPLEFLUID_LOCAL CanonicalSubfaceView
{
public:
    CanonicalSubfaceView() = default;
    explicit operator bool() const noexcept { return bool(page_); }
    [[nodiscard]] CanonicalSubfaceIncident first() const;
    [[nodiscard]] std::optional<CanonicalSubfaceIncident> second() const;
    [[nodiscard]] int outward_sign() const noexcept { return side_ ? -1 : 1; }
    [[nodiscard]] std::span<const char> support() const;
    [[nodiscard]] std::span<const char> domain() const;
    [[nodiscard]] std::string_view boundary() const;
    [[nodiscard]] std::uint64_t generation() const;
    [[nodiscard]] std::uint64_t emission() const;
    [[nodiscard]] PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> second_image() const;
    [[nodiscard]] std::size_t point_count() const;
    [[nodiscard]] PLIC::PointIdentity<ArithmeticMode::Ordinary> point(std::size_t) const;
    /** Explicit operation. The identity is the original authority key before
     * edge enrichment. The returned physical geometry includes the final loop. */
    [[nodiscard]] MaterializedCanonicalSubface materialize(Budget&, IdentityBudget&) const;
    [[nodiscard]] const void* page_identity() const noexcept { return page_.get(); }

private:
    friend class AdaptiveSubfaceSnapshot;
    friend class SubfacePieceBuilder;
    CanonicalSubfaceView(std::shared_ptr<const SubfacePiecePage> page, std::size_t slot)
        : page_(std::move(page)), slot_(slot)
    {
    }
    std::shared_ptr<const SubfacePiecePage> page_;
    std::size_t slot_ = 0;
    bool side_ = false;
};
struct CanonicalSubfaceChange
{
    CanonicalSubfaceView before, after;
};
using CanonicalSubfaceEmit = CartesianBorrowedCallback<const CanonicalSubfaceView&>;
using SubfaceOwnerVisit = CartesianBorrowedCallback<const std::shared_ptr<const void>&, const Budget*, std::size_t>;

struct SubfaceIndexWork
{
    std::size_t copied_pages = 0, copied_bytes = 0, owner_probes = 0;
    std::size_t retired_foreign_index_bytes = 0;
};
/** The sealed completed piece set. Query selectors borrow keys from actual
 * piece pages. A face visit preserves canonical piece ordering. */
class SIMPLEFLUID_LOCAL AdaptiveSubfaceSnapshot
{
public:
    ~AdaptiveSubfaceSnapshot();
    AdaptiveSubfaceSnapshot(const AdaptiveSubfaceSnapshot&) = delete;
    AdaptiveSubfaceSnapshot& operator=(const AdaptiveSubfaceSnapshot&) = delete;
    [[nodiscard]] static std::shared_ptr<const AdaptiveSubfaceSnapshot> seal(ContactHaloContext,
        GeometryConstructionControls, int resident, std::span<const CanonicalSubfaceView>, Budget&,
        std::shared_ptr<Budget>);
    [[nodiscard]] static std::shared_ptr<const AdaptiveSubfaceSnapshot> derive(
        const std::shared_ptr<const AdaptiveSubfaceSnapshot>&, ContactHaloContext, GeometryConstructionControls,
        int resident, std::span<const CanonicalSubfaceChange>, Budget&, std::shared_ptr<Budget>);
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    [[nodiscard]] const GeometryConstructionControls& controls() const noexcept { return controls_; }
    [[nodiscard]] const Budget* origin() const noexcept { return arena_.get(); }
    [[nodiscard]] std::size_t charged_bytes() const noexcept { return charge_; }
    [[nodiscard]] std::size_t index_bytes() const noexcept { return index_bytes_; }
    [[nodiscard]] std::size_t foreign_index_bytes() const noexcept { return foreign_index_bytes_; }
    [[nodiscard]] std::size_t exclusive_payload_bytes() const noexcept { return exclusive_payload_bytes_; }
    [[nodiscard]] std::size_t foreign_exclusive_payload_bytes() const noexcept
    {
        return foreign_exclusive_payload_bytes_;
    }
    [[nodiscard]] const SubfaceIndexWork& index_work() const noexcept { return work_; }
    /** Foreign index pages are admitted as an aggregate. This visit is only
     * for remaining captures shared with other incidence structures. */
    void visit_payload_owners(const SubfaceOwnerVisit&) const;
    void visit_pieces(const CanonicalSubfaceEmit&) const;
    void visit_face_pieces(const AdaptiveCellKey&, std::uint64_t source_surface, const CanonicalSubfaceEmit&) const;
    /** Owner visits can repeat shared pages. Admission deduplicates by identity. */
    void visit_owners(const SubfaceOwnerVisit&) const;
    void retain_source_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;
    [[nodiscard]] bool imports_source_owner(const void*) const noexcept;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    AdaptiveSubfaceSnapshot(
        ContactHaloContext context, GeometryConstructionControls controls, int resident, std::shared_ptr<Budget> arena)
        : context_(context), controls_(controls), arena_(std::move(arena)), resident_(resident)
    {
    }
    ContactHaloContext context_;
    GeometryConstructionControls controls_;
    std::shared_ptr<Budget> arena_;
    std::size_t charge_ = 0, size_ = 0, index_bytes_ = 0, foreign_index_bytes_ = 0;
    std::size_t exclusive_payload_bytes_ = 0, foreign_exclusive_payload_bytes_ = 0;
    SubfaceIndexWork work_;
    int resident_;
    std::shared_ptr<const SubfaceIndexRoot> indexes_;
};

/** Receives only fully authenticated support groups and never retains a live
 * registry or leaf lookup. Freeze once after the complete delivery phase. */
class SIMPLEFLUID_LOCAL SubfaceProvenanceBuilder
{
public:
    SubfaceProvenanceBuilder(IncidencePools&, std::shared_ptr<Budget>, int resident);
    ~SubfaceProvenanceBuilder();
    [[nodiscard]] SubfaceProvenanceRef append(const ReceivedTileRecipe&, std::uint64_t generation,
        const std::function<const AdaptiveLeaf&(const AdaptiveCellKey&)>&);
    void freeze();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** Shares certified loops, intrinsic tile pages and transferred explicit rings.
 * The namespace captures logical point slots before publishing immutable views. */
class SIMPLEFLUID_LOCAL SubfacePieceBuilder
{
public:
    SubfacePieceBuilder(IncidencePools&, std::shared_ptr<Budget>);
    ~SubfacePieceBuilder();
    [[nodiscard]] CanonicalSubfaceView append(const SubfaceProvenanceRef&, const TileBodyHandle&,
        std::shared_ptr<const SharedSubfaceRing>, const CertifiedIncidenceLoopView*);
    void freeze();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
