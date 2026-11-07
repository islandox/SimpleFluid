#pragma once
#include "CartesianAuthorityCredits.hh"
#include "CartesianIncidenceSource.hh"
#include "CartesianSourceInventory.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <array>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveIncidenceSession;
class OwnedSupportMembership;
struct RegistryMember;
class CartesianSourceManifestAdmission
{
    CartesianSourceManifestAdmission(AMRGrid grid, ContactHaloContext context, std::uint64_t generation,
        std::size_t cells, std::shared_ptr<const CartesianSourceInventory> inventory,
        const OwnedSupportMembership* owned)
        : grid_(std::move(grid)), context_(context), generation_(generation), cells_(cells),
          inventory_(std::move(inventory)), owned_(owned)
    {
    }
    AMRGrid grid_;
    ContactHaloContext context_;
    std::uint64_t generation_;
    std::size_t cells_;
    std::shared_ptr<const CartesianSourceInventory> inventory_;
    const OwnedSupportMembership* owned_ = nullptr; // borrowed only by synchronous exchange construction
    friend class DistributedAdaptiveIncidenceSession;
    friend class CartesianSourceManifestSet;
};
/** Closed owner inventory, including authenticated empty owner completions.
 * The opaque admission comes only from actual trusted-provider construction.
 * Parsing bytes or a public provider enum cannot construct this set. */
class CartesianSourceManifestSet
{
public:
    class Source
    {
    public:
        [[nodiscard]] int owner() const noexcept { return owner_; }
        [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
        [[nodiscard]] std::size_t cell_count() const noexcept { return cells_; }
        [[nodiscard]] const AdaptiveBytes& identity() const noexcept { return identity_; }
        [[nodiscard]] std::span<const char> domain() const noexcept
        {
            return std::span<const char>(identity_).subspan(domain_offset_, domain_size_);
        }
        [[nodiscard]] std::size_t range_count() const noexcept
        {
            return active_ranges_.empty() ? ranges_ : active_ranges_.size();
        }
        [[nodiscard]] CartesianIncidenceRange range(std::size_t) const;
        [[nodiscard]] std::size_t axis_count() const noexcept { return axes_; }
        struct Axis
        {
            unsigned level, axis;
            std::uint64_t first;
            std::span<const char> words;
            [[nodiscard]] std::size_t size() const noexcept { return words.size() / 8; }
            [[nodiscard]] double value(std::size_t) const;
        };
        [[nodiscard]] Axis axis(std::size_t) const;
        [[nodiscard]] CartesianEvaluationEnvironment evaluation_environment() const noexcept { return evaluation_; }
        [[nodiscard]] std::uint64_t revision(const CartesianCellAncestry&) const;

    private:
        AdaptiveBytes identity_;
        std::vector<CartesianIncidenceRange> active_ranges_;
        CartesianEvaluationEnvironment evaluation_;
        int owner_ = -1;
        std::uint64_t version_ = 0, base_revision_ = 0;
        std::size_t cells_ = 0, domain_offset_ = 0, domain_size_ = 0, ranges_ = 0, ranges_offset_ = 0, axes_ = 0,
                    axes_offset_ = 0, revisions_ = 0, revisions_offset_ = 0;
        void authenticate(const AMRGrid&, ContactHaloContext, int, std::size_t, std::uint64_t);
        friend class CartesianSourceManifestSet;
    };
    static std::shared_ptr<const CartesianSourceManifestSet> exchange(const CartesianSourceManifestAdmission&,
        Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, std::shared_ptr<Budget>,
        std::shared_ptr<cartesian_detail::ArenaCredits>);
    /** Decoder validation seam only: never constructs an authorized manifest
     * set or a procedural answer. Tests can fault wire records without forging
     * the session-only admission token. */
    static void validate_untrusted_wire(
        const AMRGrid&, ContactHaloContext, std::uint64_t generation, std::span<const OwnerPacket>, int ranks, Budget&);
    ~CartesianSourceManifestSet();
    [[nodiscard]] const AMRGrid& grid() const noexcept { return grid_; }
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] std::size_t owner_count() const noexcept { return owners_; }
    [[nodiscard]] std::size_t source_count() const noexcept { return sources_.size(); }
    [[nodiscard]] const Source& source(std::size_t owner) const { return sources_.at(owner); }
    [[nodiscard]] bool has_explicit_support(std::span<const char>) const noexcept;
    [[nodiscard]] bool native_member(const RegistryMember&, std::span<const char> domain) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_; }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ContactPayloadRole::CartesianManifestObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianManifestSources, sources_);
        for (const auto& source : sources_)
        {
            contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianManifestIdentity, source.identity_);
            contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianManifestRanges, source.active_ranges_);
        }
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianManifestLeases, leases_);
        contact_payload_buffer(visitor, this, ContactPayloadRole::CartesianExplicitCellHeaders, explicit_cells_);
        for (const auto& cell : explicit_cells_)
        {
            contact_payload_buffer(visitor, &cell, ContactPayloadRole::CartesianExplicitMemberHeaders, cell.members);
            for (const auto& member : cell.members)
                contact_payload_buffer(
                    visitor, &cell, ContactPayloadRole::CartesianExplicitSupportKeys, member.support);
        }

        if (local_)
            local_->visit_borrowed_payload(visitor);
        for (const auto& lease : leases_)
            if (lease)
                lease->visit_borrowed_payload(visitor);
    }
    [[nodiscard]] bool local_dependencies_admitted_in(const CartesianAuthorityArena& arena) const noexcept
    {
        return budget_ == arena.budget && local_ && local_->dependencies_admitted_in(arena);
    }
    void retain_dependencies(cartesian_detail::CreditConstruction& credits,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases,
        cartesian_detail::Allocation& memory) const
    {
        if (local_dependencies_admitted_in(*credits.arena))
            return;
        visit_dependencies<CartesianDependencyVisit::Physical>(
            [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
            {
                using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
                const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
                credits.retain(fixed, origin, bytes, keep, leases, memory);
            });
    }
    template<CartesianDependencyVisit Mode = CartesianDependencyVisit::Flat, class Visitor>
    void visit_dependencies(Visitor&& visitor) const
    {
        if (local_)
        {
            const auto visit_inventory = [&]() -> decltype(auto)
            {
                return visitor(local_, local_->allocation_budget(), local_->metadata_bytes(), local_->has_arena_keeper());
            };
            const auto descend = [&]
            {
                if constexpr (Mode == CartesianDependencyVisit::Observed)
                    local_->visit_observed_dependencies(visitor);
                else if constexpr (Mode == CartesianDependencyVisit::Physical)
                    local_->visit_physical_dependencies(visitor);
                else
                    local_->visit_dependencies(visitor);
            };
            if constexpr (std::is_same_v<decltype(visit_inventory()), bool>)
            {
                if (visit_inventory())
                    descend();
            }
            else
            {
                visit_inventory();
                descend();
            }
        }
        for (const auto& lease : leases_)
            visitor(lease, lease->budget, lease->metadata, bool(lease->keeper));
    }

private:
    struct ExplicitMember
    {
        AdaptiveBytes support;
        std::size_t encoded_size = 0;
    };
    struct ExplicitCell
    {
        AdaptiveCellKey key;
        std::uint64_t revision = 0;
        int owner = -1;
        std::vector<ExplicitMember> members;
    };
    CartesianSourceManifestSet() = default;
    static void decode_wire(const AMRGrid&, ContactHaloContext, std::uint64_t, std::span<const OwnerPacket>, int,
        std::vector<Source>&, std::vector<ExplicitCell>&, cartesian_detail::Allocation&);
    static void decode_inventory(std::vector<Source>&, const AMRGrid&, ContactHaloContext, int, std::uint64_t,
        std::size_t, std::span<const char>, cartesian_detail::Allocation&);
    AMRGrid grid_;
    ContactHaloContext context_;
    std::uint64_t generation_ = 0;
    std::vector<Source> sources_;
    std::vector<ExplicitCell> explicit_cells_;
    std::shared_ptr<const CartesianSourceInventory> local_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
    std::size_t charge_ = 0;
    std::size_t owners_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
