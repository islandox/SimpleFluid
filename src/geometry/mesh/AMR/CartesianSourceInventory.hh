/** @file CartesianSourceInventory.hh
 * @brief Immutable active Cartesian ancestry batches with captured native evaluation controls.
 */
#pragma once
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include "geometry/mesh/AMR/CartesianIncidenceAuthority.hh"
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Owner-local current membership. Only trusted sources enter construction;
 * subsequent transitions are derived from this exact immutable predecessor.
 * Active selectors describe ancestry, never physical incidence or adjacency.
 * The session authenticates collective context changes and the trusted grid.
 * No predecessor is retained solely for transition history. */
class CartesianSourceInventory : public std::enable_shared_from_this<CartesianSourceInventory>
{
public:
    /** Original local evaluation state, captured when the trusted source was
     * constructed. Stable controls must equal the source signature. Raw fenv_t
     * is never serialized, compared as bytes, or replaced by current controls. */
    struct CapturedSource
    {
        std::shared_ptr<const CartesianIncidenceSource> source;
        std::fenv_t environment{};
    };
    /** Exact existing key/revision retirement; each key must occur once. */
    struct Retirement
    {
        AdaptiveCellKey key;
        std::uint64_t revision = 0;
    };
    /** One original immutable source and its surviving compressed selectors. */
    class Batch
    {
    public:
        [[nodiscard]] const std::shared_ptr<const CartesianIncidenceSource>& source() const noexcept { return source_; }
        [[nodiscard]] const std::fenv_t& evaluation_state() const noexcept { return environment_; }
        [[nodiscard]] std::span<const CartesianIncidenceRange> ranges() const noexcept { return ranges_; }
        [[nodiscard]] std::size_t cell_count() const noexcept { return cells_; }
        [[nodiscard]] bool contains(const CartesianCellAncestry&) const noexcept;

    private:
        std::shared_ptr<const CartesianIncidenceSource> source_;
        std::fenv_t environment_{};
        std::vector<CartesianIncidenceRange> ranges_;
        std::size_t cells_ = 0;
        friend class CartesianSourceInventory;
    };

    /** Admit complete trusted source batches for an explicitly selected owner.
     * All batches must have this current context and the captured signature;
     * grid descriptors agree exactly, while connected domains may differ.
     * An empty inventory still binds context, owner and version. The session's
     * opaque manifest admission independently authenticates its empty grid.
     * The explicit arena must belong to budget; keeper must own that budget.
     * These are local operations with no collective or geometry evaluation. */
    [[nodiscard]] static std::shared_ptr<const CartesianSourceInventory> make(ContactHaloContext, int owner,
        std::uint64_t version, std::span<const CapturedSource>, Budget&, std::shared_ptr<Budget> keeper,
        std::shared_ptr<CartesianAuthorityArena> arena);
    /** Retire exact predecessor members and append only newly constructed delta
     * sources. New source contexts equal next; inherited sources keep original
     * context/version/identity and captured evaluation state. Context changes
     * are authenticated by the owning session, not rejected by epoch heuristics.
     * Missing, repeated or stale retirements and duplicate resulting keys reject.
     * Failure leaves this predecessor unchanged. */
    [[nodiscard]] std::shared_ptr<const CartesianSourceInventory> derive(ContactHaloContext next, std::uint64_t version,
        std::span<const Retirement>, std::span<const CapturedSource>, Budget&, std::shared_ptr<Budget> keeper,
        std::shared_ptr<CartesianAuthorityArena> arena) const;
    ~CartesianSourceInventory();
    CartesianSourceInventory(const CartesianSourceInventory&) = delete;
    CartesianSourceInventory& operator=(const CartesianSourceInventory&) = delete;
    [[nodiscard]] std::span<const Batch> batches() const noexcept { return batches_; }
    [[nodiscard]] const Batch* lookup(const AdaptiveCellKey&) const noexcept;
    [[nodiscard]] bool contains(const AdaptiveCellKey& key) const noexcept { return lookup(key) != nullptr; }
    [[nodiscard]] std::uint64_t revision(const AdaptiveCellKey&) const;
    [[nodiscard]] std::size_t cell_count() const noexcept { return cells_; }
    [[nodiscard]] int owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    /** Complete current binding: owner/context/version, ordered full live source
     * identities and ordered active selectors, including the empty case. */
    [[nodiscard]] const AdaptiveBytes& identity() const noexcept { return identity_; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_; }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ContactPayloadRole::CartesianInventoryObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianInventoryBatches, batches_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianInventoryIdentity, identity_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianInventoryLeases, leases_);
        for (const auto& batch : batches_)
        {
            contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianInventoryRanges, batch.ranges_);
            if (batch.source_)
                batch.source_->visit_borrowed_payload(visitor);
        }
        for (const auto& lease : leases_)
            if (lease)
                lease->visit_borrowed_payload(visitor);
    }
    /** Immutable source bodies and lease-token metadata are separate owners;
     * imported tokens never contribute another arena's payload credit. */
    template<class Visitor> void visit_dependencies(Visitor&& visitor) const
    {
        for (const auto& batch : batches_)
            visitor(batch.source_, batch.source_->allocation_budget(), batch.source_->metadata_bytes(),
                batch.source_->has_arena_keeper());
        for (const auto& lease : leases_)
            visitor(lease, lease->budget, lease->metadata, bool(lease->keeper));
    }

private:
    friend class OwnedSupportMembership;
    [[nodiscard]] bool derived_from(const std::shared_ptr<const CartesianSourceInventory>&) const noexcept;
    CartesianSourceInventory() = default;
    static std::shared_ptr<const CartesianSourceInventory> construct(const CartesianSourceInventory*,
        ContactHaloContext, int, std::uint64_t, std::span<const Retirement>, std::span<const CapturedSource>, Budget&,
        std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>);
    ContactHaloContext context_;
    int owner_ = -1;
    std::uint64_t version_ = 0;
    std::size_t cells_ = 0, added_begin_ = 0, maximum_domain_bytes_ = 0;
    std::weak_ptr<const CartesianSourceInventory> predecessor_;
    std::vector<Batch> batches_;
    AdaptiveBytes identity_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
    std::size_t charge_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
