#pragma once
#include "ContactSourcePoints.hh"
#include "PooledLineQuery.hh"
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace cartesian_line_detail
{
/** Private-storage boundary predicate. Full UINT32_MAX is an ordinary valid
 * selector; map/fixed and empty/absent are represented separately. The forced
 * branch permits boundary qualification without allocating >4 billion rows. */
[[nodiscard]] inline bool compact_indices_fit(
    std::initializer_list<std::uint64_t> values, bool force_wide = false) noexcept
{
    if (force_wide)
        return false;
    for (auto value : values)
        if (value > std::numeric_limits<std::uint32_t>::max())
            return false;
    return true;
}
} // namespace cartesian_line_detail
class CartesianSourceManifestSet;
class CartesianSourceLineRangeBuilder;
class NativeContactFundingAccess;
struct NativeContactOwnerFunding;
/** Only the actual session may bind its closed native owner manifests, real
 * complete-line pass and published point snapshot. Eligibility bits and key
 * parsing cannot construct this admission. */
class CartesianLineRangeAdmission
{
    CartesianLineRangeAdmission(std::shared_ptr<const CartesianSourceManifestSet> manifests,
        std::shared_ptr<const ContactSourcePointSnapshot> points, ContactHaloContext context, std::uint64_t generation,
        unsigned layer)
        : manifests_(std::move(manifests)), points_(std::move(points)), context_(context), generation_(generation),
          layer_(layer)
    {
    }
    std::shared_ptr<const CartesianSourceManifestSet> manifests_;
    std::shared_ptr<const ContactSourcePointSnapshot> points_;
    ContactHaloContext context_;
    std::uint64_t generation_;
    unsigned layer_;
    friend class DistributedAdaptiveIncidenceSession;
    friend class CartesianSourceLineRangeBuilder;
};

/** A borrowed full support key with one source-coordinate word replaced.
 * size()/operator[] deliberately permit policy-preserving pool lookup without
 * allocating a reconstructed AdaptiveBytes at every read. */
struct CartesianSupportRangeKeyView
{
    std::span<const char> prototype;
    std::size_t offset = 0;
    std::uint64_t coordinate = 0;
    [[nodiscard]] std::size_t size() const noexcept { return prototype.size(); }
    [[nodiscard]] char operator[](std::size_t i) const noexcept
    {
        return i >= offset && i - offset < 8 ? char(coordinate >> (8 * (i - offset))) : prototype[i];
    }
};

/** Shared actual source-axis/component maps. Every entry came from an original
 * completed subscriber; maps never evaluate geometry or normalize a point. */
class CartesianSourceLineRangeCatalog
{
public:
    ~CartesianSourceLineRangeCatalog();
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] const Budget* allocation_budget() const noexcept;
    [[nodiscard]] bool has_arena_keeper() const noexcept;
    [[nodiscard]] std::size_t map_count() const noexcept;
    [[nodiscard]] std::size_t mapped_components() const noexcept;
    /** Actual complete-value prototype requests and distinct retained entries. */
    [[nodiscard]] std::size_t prototype_count() const noexcept;
    [[nodiscard]] std::size_t prototype_request_count() const noexcept;
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    friend class NativeContactFundingAccess;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const CartesianSourceLineRangeCatalog>&, std::size_t) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    CartesianSourceLineRangeCatalog();
    friend class CartesianSourceLineRangeBuilder;
    friend class CartesianSourceLineRangeRecipe;
};

/** One complete ordered line, described by source-coordinate families and
 * range/mask membership. There is no subscriber-slot or output-point array. */
class CartesianSourceLineRangeRecipe
{
public:
    ~CartesianSourceLineRangeRecipe();
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t family_count() const noexcept;
    /** Borrowed observation of the actually populated immutable alternative. */
    [[nodiscard]] std::size_t index_width_bytes() const noexcept;
    [[nodiscard]] std::size_t nonforward_count() const noexcept;
    [[nodiscard]] std::size_t result_selector_count() const noexcept;
    [[nodiscard]] unsigned certified_layer() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] static std::size_t control_metadata_bytes() noexcept;
    [[nodiscard]] const Budget* allocation_budget() const noexcept;
    [[nodiscard]] bool has_arena_keeper() const noexcept;
    [[nodiscard]] const std::shared_ptr<const CartesianSourceLineRangeCatalog>& catalog() const noexcept;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    [[nodiscard]] PooledLineSubscriber at(std::size_t, const IncidencePools&) const;
    [[nodiscard]] std::size_t result_size(std::size_t, const IncidencePools&) const;
    [[nodiscard]] std::size_t result_at(std::size_t, std::size_t, const IncidencePools&) const;
    [[nodiscard]] std::size_t captured_result_size(std::size_t) const;
    [[nodiscard]] std::size_t captured_result_at(std::size_t, std::size_t) const;
    [[nodiscard]] const std::shared_ptr<const ContactSourcePointSnapshot>& captured_point_snapshot() const noexcept;
    [[nodiscard]] bool exact_replay(std::span<const PooledLineSubscriber>, const IncidencePools&) const;
    /** Exact immutable owners imported by retain_dependencies; no allocation. */
    [[nodiscard]] bool imports_owner(const void*) const noexcept;
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept;
    /** A destination budget imports every immutable dependency once. */
    void retain_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;

private:
    friend class NativeContactFundingAccess;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const CartesianSourceLineRangeRecipe>&, std::size_t) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    CartesianSourceLineRangeRecipe();
    [[nodiscard]] std::pair<std::size_t, std::uint64_t> locate(std::size_t, const IncidencePools&) const;
    [[nodiscard]] std::pair<std::size_t, std::uint64_t> locate_captured(std::size_t) const;
    [[nodiscard]] std::size_t resolve_captured_point(
        const std::array<std::size_t, 3>&, std::uint8_t map_bits, std::uint64_t) const;
    [[nodiscard]] std::size_t resolve_point(
        const std::array<std::size_t, 3>&, std::uint8_t map_bits, std::uint64_t, const IncidencePools&) const;
    friend class CartesianSourceLineRangeBuilder;
};

/** A whole-pass builder shares axis proofs across buckets. freeze() is required
 * before publication. The only temporary member arrays are one bounded line. */
class CartesianSourceLineRangeBuilder
{
public:
    CartesianSourceLineRangeBuilder(
        const CartesianLineRangeAdmission&, Budget&, std::shared_ptr<CartesianAuthorityArena>);
    ~CartesianSourceLineRangeBuilder();
    [[nodiscard]] std::shared_ptr<const CartesianSourceLineRangeRecipe> make_recipe(
        std::size_t line, std::span<const PooledLineSubscriber>, const IncidencePools&, unsigned certified_layer);
    void freeze();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
