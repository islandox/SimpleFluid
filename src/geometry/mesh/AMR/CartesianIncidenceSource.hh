/** @file CartesianIncidenceSource.hh
 * @brief Immutable local Cartesian incidence sources and exact replay certificates.
 */
#pragma once
#include "NativeContactFunding.hh"
#include "geometry/mesh/AMR/CompleteLineQuery.hh"
#include "geometry/mesh/AMR/ContactHaloContext.hh"
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/PrivateTileRecipe.hh"
#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Optional source admission failed its exact native authentication. The legacy
 * explicit incidence path may still evaluate the selected leaf. This does not
 * classify resource failures, which retain their ordinary failure path. */
class CartesianSourceAuthenticationFailure : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};
/** Borrowed callable; construction never allocates. A temporary callable is
 * valid through the enclosing call only; callbacks are never stored. */
template<class... Args> class CartesianBorrowedCallback
{
public:
    CartesianBorrowedCallback() = default;
    template<class F>
        requires(!std::is_same_v<std::remove_cvref_t<F>, CartesianBorrowedCallback>)
    CartesianBorrowedCallback(F&& callback) noexcept
        : object_(const_cast<void*>(static_cast<const void*>(std::addressof(callback)))),
          invoke_([](void* object, Args... args)
              { (*static_cast<std::remove_reference_t<F>*>(object))(std::forward<Args>(args)...); })
    {
    }
    void operator()(Args... args) const { invoke_(object_, std::forward<Args>(args)...); }
    explicit operator bool() const noexcept { return invoke_; }

private:
    void* object_ = nullptr;
    void (*invoke_)(void*, Args...) = nullptr;
};
/** Stable control fields only; floating-point flags and padded fenv_t are never
 * canonical identity bytes. A changed dirty evaluation publishes a new source. */
struct CartesianEvaluationEnvironment
{
    std::uint64_t platform = 0, rounding = 0, primary = 0, secondary = 0;
    auto operator<=>(const CartesianEvaluationEnvironment&) const = default;
};
[[nodiscard]] CartesianEvaluationEnvironment cartesian_evaluation_environment();
/** Half-open active Cartesian ancestry box. Ranges never imply physical incidence. */
struct CartesianIncidenceRange
{
    unsigned level = 0;
    std::array<std::uint64_t, 3> lower{}, upper{};
};
/** Actual evaluated binary64 coordinates for an occupied contiguous axis interval. */
struct CartesianIncidenceAxis
{
    unsigned level = 0, axis = 0;
    std::uint64_t first = 0;
    std::vector<double> values;
};
struct CartesianIncidenceRevision
{
    CartesianCellAncestry cell;
    std::uint64_t revision = 0;
};
struct CartesianIncidenceSurface
{
    AdaptiveCellKey cell;
    std::uint64_t revision = 0;
    std::size_t ordinal = 0;
    const AdaptiveSurface& surface;
    /** Exact legacy SurfaceRequest encoding, borrowed during the callback. */
    std::span<const char> packet;
    /** Full source/version/cell/surface identity; never a routing hash. */
    std::span<const char> member;
};
using CartesianSurfaceEmit = CartesianBorrowedCallback<const CartesianIncidenceSurface&>;
class CartesianIncidenceSource;
/** One bounded actual native evaluation, issued only by the trusted source.
 * Immutable leaf/geometry values are the SAME values used by initial registry
 * construction. A caller cannot attach this witness to supplied geometry. */
class CartesianNativeIncidenceCell
{
public:
    ~CartesianNativeIncidenceCell();
    CartesianNativeIncidenceCell(const CartesianNativeIncidenceCell&) = delete;
    CartesianNativeIncidenceCell& operator=(const CartesianNativeIncidenceCell&) = delete;
    [[nodiscard]] const AdaptiveLeaf& leaf() const noexcept { return leaf_; }
    [[nodiscard]] const std::shared_ptr<const CartesianIncidenceSource>& source() const noexcept { return source_; }
    [[nodiscard]] CartesianEvaluationEnvironment evaluation_environment() const noexcept { return evaluation_; }
    [[nodiscard]] const std::fenv_t& evaluation_state() const noexcept { return environment_; }

private:
    CartesianNativeIncidenceCell() = default;
    AdaptiveLeaf leaf_;
    std::shared_ptr<const CartesianIncidenceSource> source_;
    CartesianEvaluationEnvironment evaluation_;
    std::fenv_t environment_{};
    IdentityBudget identities_;
    Budget* budget_ = nullptr;
    std::size_t charge_ = 0;
    std::shared_ptr<Budget> keeper_;
    friend class CartesianIncidenceSource;
};
struct CartesianSourceVisitStatistics
{
    std::size_t candidate_cells = 0, emitted_surfaces = 0;
};

/** Trusted Cartesian factory only. No provider enum or descriptor parser can
 * construct this source. Sources retain no leaves, geometry shells, or hierarchy.
 * The budget/optional keeper must outlive the source. */
class CartesianIncidenceSource
{
public:
    [[nodiscard]] static bool eligible(const AdaptiveLeaf&) noexcept;
    /** Every supplied leaf must belong to this owner/domain and pass the existing
     * Cartesian kernel authentication. Unsupported supplied geometry stays explicit. */
    [[nodiscard]] static std::shared_ptr<const CartesianIncidenceSource> make(const AMRGrid&,
        std::span<const AdaptiveLeaf>, const AdaptiveBytes& domain, int owner, std::uint64_t version,
        ContactHaloContext, Budget&, std::shared_ptr<Budget> keeper = {});
    [[nodiscard]] static std::unique_ptr<const CartesianNativeIncidenceCell> evaluate_cell(
        std::shared_ptr<const CartesianIncidenceSource>, const CartesianCellAncestry&, Budget&,
        std::shared_ptr<Budget> keeper = {});
    [[nodiscard]] std::shared_ptr<const CartesianIncidenceSource> reevaluated(
        Budget&, std::shared_ptr<Budget> keeper = {}) const;
    ~CartesianIncidenceSource();
    CartesianIncidenceSource(const CartesianIncidenceSource&) = delete;
    CartesianIncidenceSource& operator=(const CartesianIncidenceSource&) = delete;
    [[nodiscard]] std::span<const CartesianIncidenceRange> ranges() const noexcept { return ranges_; }
    [[nodiscard]] std::span<const CartesianIncidenceAxis> axes() const noexcept { return axes_; }
    [[nodiscard]] std::span<const CartesianIncidenceRevision> revisions() const noexcept { return revisions_; }
    [[nodiscard]] const AdaptiveBytes& identity() const noexcept { return identity_; }
    [[nodiscard]] const AdaptiveBytes& domain() const noexcept { return domain_; }
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    [[nodiscard]] int owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint64_t region() const noexcept { return grid_.region_id; }
    [[nodiscard]] CartesianEvaluationEnvironment evaluation_environment() const noexcept { return evaluation_; }
    [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_; }
    [[nodiscard]] const void* admission_identity() const noexcept { return &charge_; }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
    [[nodiscard]] std::size_t cell_count() const noexcept { return cell_count_; }
    /** Pure borrowed storage walk: never evaluate or synthesize incidence. */
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ContactPayloadRole::CartesianSourceObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianSourceDomain, domain_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianSourceIdentity, identity_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianSourceRanges, ranges_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianSourceAxes, axes_);
        for (const auto& axis : axes_)
            contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianSourceAxisValues, axis.values);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianSourceRevisions, revisions_);
        // This private owner is always issued by the exact Cartesian factory.
        // Its owning translation unit supplies the concrete sizeof, without
        // RTTI, geometry evaluation or guessing from the public AMRGrid layout.
        if (kernel_)
            contact_payload_object(visitor, kernel_.get(), ContactPayloadRole::CartesianSourceKernel,
                native_cartesian_kernel_payload_bytes(), true);
    }
    [[nodiscard]] bool contains(const CartesianCellAncestry&) const noexcept;
    [[nodiscard]] std::uint64_t revision(const CartesianCellAncestry&) const;
    /** Enumerate one bounded shell at a time in range/z/y/x then constructor-face
     * order. A nonempty key filters complete generated support keys exactly.
     * Recognized original keys locate candidate ancestry planes/ranges using
     * their encoded integer fields; unknown formats scan the selected ranges.
     * All candidates still run the unchanged constructors and full-key filter.
     * This makes no inverse-normalization or physical-neighbor claim. */
    void visit_surfaces(Budget&, const CartesianSurfaceEmit&, std::span<const char> support = {},
        std::optional<std::size_t> range = {}, CartesianSourceVisitStatistics* = nullptr) const;
    /** Generate one cell's actual faces, preserving constructors and native face ordinals. */
    void visit_cell(const CartesianCellAncestry&, Budget&, const CartesianSurfaceEmit&) const;

private:
    friend class NativeContactFundingAccess;
    static constexpr std::size_t source_control_bytes = 128;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const CartesianIncidenceSource>&, std::size_t) const;
    CartesianIncidenceSource() = default;
    void build_identity(Budget&, std::size_t& retained);
    AMRGrid grid_;
    CartesianEvaluationEnvironment evaluation_;
    std::shared_ptr<const AdaptiveProviderKernel> kernel_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> kernel_allocation_;
    AdaptiveBytes domain_, identity_;
    std::vector<CartesianIncidenceRange> ranges_;
    std::vector<CartesianIncidenceAxis> axes_;
    std::vector<CartesianIncidenceRevision> revisions_;
    std::uint64_t base_revision_ = 0, version_ = 0;
    ContactHaloContext context_;
    int owner_ = -1;
    std::size_t cell_count_ = 0, charge_ = 0;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
};

/** Source edges discover rounded line keys only. Complete runtime line groups
 * must instead use edges of actual complete-support overlay tile outputs. */
struct CartesianIncidenceEdge
{
    AdaptiveBytes key, member;
    LinePoint first, last;
    unsigned image = 0;
    std::size_t edge = 0;
};
using CartesianEdgeEmit = CartesianBorrowedCallback<const CartesianIncidenceEdge&>;
void visit_cartesian_source_edges(
    const CartesianIncidenceSource&, Budget&, const CartesianEdgeEmit&, std::span<const char> line = {});
/** `tile.face` is the existing complete-support first-cell physical output.
 * Inverse periodic translation and line identity construction match visit_edges. */
void visit_cartesian_tile_edges(std::span<const char> domain, std::span<const char> tile_member,
    const PrivateTileRecipe& tile, Budget&, const CartesianEdgeEmit&, std::span<const char> line = {});

/** Borrowed explicit fallback input; its original geometry and packet remain
 * authoritative. It is never promoted into a procedural Cartesian source. */
struct CartesianExplicitSupportMember
{
    const AdaptiveLeaf* leaf = nullptr;
    std::size_t face = 0;
    int owner = -1;
    AdaptiveBytes identity;
};
using CartesianSupportMemberEmit = CartesianBorrowedCallback<std::span<const char>, std::span<const char>, int>;
/** Complete support membership union. Input source order/range order precedes
 * explicit exception order; the caller restores authenticated legacy arrival
 * order before invoking the existing complete-support overlay. */
void visit_cartesian_support_members(std::span<const std::shared_ptr<const CartesianIncidenceSource>>,
    std::span<const CartesianExplicitSupportMember>, std::span<const char> support, Budget&,
    const CartesianSupportMemberEmit&);
struct CartesianCertifiedTileInput
{
    std::span<const char> domain, member;
    const PrivateTileRecipe* tile = nullptr;
};
struct CartesianExplicitLineMember
{
    AdaptiveBytes line, member;
    LineSubscription subscription;
};
/** Exact rounded-key union from actual complete-support tile outputs and
 * explicit exception subscriptions. Source edges cannot enter this API. */
void visit_cartesian_line_members(std::span<const CartesianCertifiedTileInput>,
    std::span<const CartesianExplicitLineMember>, std::span<const char> line, Budget&, const CartesianEdgeEmit&);
/** Emit all old and new support keys before either version is retired. Dirty
 * line keys must subsequently come from both actual old/new complete overlays;
 * the caller propagates those keys to all affected tiles through layer 63. */
void visit_cartesian_changed_supports(const CartesianIncidenceSource* previous, const CartesianIncidenceSource* next,
    Budget&, const CartesianBorrowedCallback<std::span<const char>>&);

/** Compact selectors are expanded with the existing source constructors. */
enum class CartesianRecipeOutput
{
    SurfacePacket,
    SurfaceKey,
    FaceKey,
    PointKeys,
    LineKeys
};
struct CartesianRecipeSelector
{
    std::size_t source = 0;
    std::optional<std::size_t> range;
    std::optional<CartesianCellAncestry> cell;
    std::optional<std::size_t> face;
    AdaptiveBytes support;
    CartesianRecipeOutput output = CartesianRecipeOutput::PointKeys;
    /** Constructor vertex/edge order; count zero means the complete loop. */
    std::size_t first = 0, count = 0;
    bool reverse = false, authority_image = false;
};
using CartesianIdentityEmit = CartesianBorrowedCallback<std::span<const char>>;
/** Candidate immutable recipe. Source/version dependencies are complete values,
 * and selectors describe ranges rather than permanent per-cell member arrays. */
struct CartesianReplayRecipe
{
    std::vector<std::shared_ptr<const CartesianIncidenceSource>> sources;
    std::vector<CartesianRecipeSelector> selectors;
};
void replay_cartesian_recipe(const CartesianReplayRecipe&, Budget&, const CartesianIdentityEmit&);

} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
