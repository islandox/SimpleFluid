#pragma once
#include "NativeContactFunding.hh"
#include "CartesianAuthorityCredits.hh"
#include "CartesianIncidenceSource.hh"
#include "CartesianSourceInventory.hh"
#include "ContactGroupAllocation.hh"
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
class CartesianSourceManifestSet : public std::enable_shared_from_this<CartesianSourceManifestSet>
{
    struct ReceiptBytes
    {
        static constexpr std::size_t control_bytes = 128;
        ContactGroupAllocation admission;
        AdaptiveBytes bytes;
        ReceiptBytes(ContactGroupAllocation, std::span<const char>);
        [[nodiscard]] std::size_t metadata_bytes() const noexcept { return admission.bytes() + control_bytes; }
        [[nodiscard]] NativeContactOwnerFunding native_funding(
            const std::shared_ptr<const ReceiptBytes>& owner, std::size_t extent) const
        {
            return NativeContactFundingAccess::split_named<control_bytes>(owner, extent,
                [](const ReceiptBytes& value) { return value.admission.bytes(); },
                [](const ReceiptBytes& value) -> const auto& { return value.admission.keeper(); });
        }
    };
    struct RangeLess
    {
        bool operator()(const CartesianIncidenceRange&, const CartesianIncidenceRange&) const noexcept;
    };
    using RangeTree = persistent_ordered_detail::OrderedPages<CartesianIncidenceRange, RangeLess>;
    struct SourceLess;

public:
    class Source
    {
    public:
        [[nodiscard]] int owner() const noexcept { return owner_; }
        [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
        [[nodiscard]] std::size_t cell_count() const noexcept { return cells_; }
        [[nodiscard]] std::span<const char> identity() const noexcept
        {
            return identity_ ? std::span<const char>(identity_->bytes) : borrowed_identity_;
        }
        [[nodiscard]] std::span<const char> domain() const noexcept
        {
            return identity().subspan(domain_offset_, domain_size_);
        }
        [[nodiscard]] std::size_t range_count() const noexcept { return active_count_ ? active_count_ : ranges_; }
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
        std::shared_ptr<const ReceiptBytes> identity_, irregular_ranges_;
        RangeTree::Ptr active_ranges_;
        // Used only by the synchronous validating decoder. Frozen rows clear both spans.
        std::span<const char> borrowed_identity_, borrowed_ranges_;
        std::size_t active_count_ = 0, slot_ = 0;
        CartesianEvaluationEnvironment evaluation_;
        int owner_ = -1;
        std::uint64_t version_ = 0, base_revision_ = 0;
        std::size_t cells_ = 0, domain_offset_ = 0, domain_size_ = 0, ranges_ = 0, ranges_offset_ = 0, axes_ = 0,
                    axes_offset_ = 0, revisions_ = 0, revisions_offset_ = 0;
        void authenticate(const AMRGrid&, ContactHaloContext, int, std::size_t, std::uint64_t);
        [[nodiscard]] std::span<const char> raw_active_ranges() const noexcept
        {
            return irregular_ranges_ ? std::span<const char>(irregular_ranges_->bytes) : borrowed_ranges_;
        }
        friend class CartesianSourceManifestSet;
        friend struct SourceLess;
    };
    static std::shared_ptr<const CartesianSourceManifestSet> exchange(const CartesianSourceManifestAdmission&,
        Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, std::shared_ptr<Budget>,
        std::shared_ptr<cartesian_detail::ArenaCredits>,
        const std::shared_ptr<const CartesianSourceManifestSet>& previous = {});
    /** Decoder validation seam only: never constructs an authorized manifest
     * set or a procedural answer. Tests can fault wire records without forging
     * the session-only admission token. */
    static void validate_untrusted_wire(
        const AMRGrid&, ContactHaloContext, std::uint64_t generation, std::span<const OwnerPacket>, int ranks, Budget&);
    /** Validate the same bytes and exercise detached receipt storage admission.
     * The private construction is discarded and never becomes authority. */
    static void validate_receipt_storage(
        const AMRGrid&, ContactHaloContext, std::uint64_t generation, std::span<const OwnerPacket>, int ranks, Budget&);
    ~CartesianSourceManifestSet();
    [[nodiscard]] const AMRGrid& grid() const noexcept { return grid_; }
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] std::size_t owner_count() const noexcept { return owners_; }
    [[nodiscard]] std::size_t source_count() const noexcept;
    [[nodiscard]] const Source& source(std::size_t ordinal) const;
    struct ReceiptWork
    {
        std::size_t allocated_pages = 0, allocated_page_bytes = 0, copied_body_bytes = 0;
        std::size_t shared_peers = 0, validated_sources = 0, validated_explicit_cells = 0;
    };
    [[nodiscard]] ReceiptWork receipt_work() const noexcept { return receipt_work_; }

    struct DiagnosticFrame
    {
        ContactHaloContext context;
        std::uint64_t generation;
        std::span<const OwnerPacket> packets;
    };
    /** Synchronous storage observations only. This view cannot yield a
     * manifest or any construction authority. Borrowed bytes are callback-local. */
    class DiagnosticReceiptView
    {
    public:
        DiagnosticReceiptView(const DiagnosticReceiptView&) = delete;
        DiagnosticReceiptView& operator=(const DiagnosticReceiptView&) = delete;
        [[nodiscard]] std::size_t source_count() const noexcept { return result_->source_count(); }
        [[nodiscard]] int owner(std::size_t ordinal) const { return result_->source(ordinal).owner(); }
        [[nodiscard]] std::uint64_t version(std::size_t ordinal) const { return result_->source(ordinal).version(); }
        [[nodiscard]] std::span<const char> identity(std::size_t ordinal) const
        {
            return result_->source(ordinal).identity();
        }
        [[nodiscard]] std::size_t range_count(std::size_t ordinal) const
        {
            return result_->source(ordinal).range_count();
        }
        [[nodiscard]] CartesianIncidenceRange range(std::size_t ordinal, std::size_t index) const
        {
            return result_->source(ordinal).range(index);
        }
        [[nodiscard]] std::size_t irregular_range_admission(std::size_t ordinal) const
        {
            return result_->irregular_range_admission(ordinal);
        }
        [[nodiscard]] ReceiptWork work() const noexcept { return result_->receipt_work(); }

    private:
        friend class CartesianSourceManifestSet;
        explicit DiagnosticReceiptView(const CartesianSourceManifestSet& result) : result_(&result) {}
        const CartesianSourceManifestSet* result_;
    };
    /** Decode both complete frames and compare private receipt storage in one
     * admitted arena. Neither validated result escapes as authority. */
    static void inspect_receipt_transition(const AMRGrid&, DiagnosticFrame before, DiagnosticFrame after, int ranks,
        Budget&, const CartesianBorrowedCallback<const DiagnosticReceiptView&, const DiagnosticReceiptView&>&);

    [[nodiscard]] bool has_explicit_support(std::span<const char>) const noexcept;
    [[nodiscard]] bool native_member(const RegistryMember&, std::span<const char> domain) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_; }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, GeometryContactPayloadRole::CartesianManifestObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianManifestLeases, leases_);
        const auto page = [&](const auto& node, ContactPayloadRole role)
        {
            contact_payload_object(visitor, node.get(), role, sizeof(*node), true);
            contact_payload_buffer(visitor, node.get(), role, node->rows);
            contact_payload_buffer(visitor, node.get(), role, node->children);
        };
        const auto body = [&](const auto& owner, ContactPayloadRole role)
        {
            if (owner)
            {
                contact_payload_object(visitor, owner.get(), role, sizeof(*owner), true);
                contact_payload_buffer(visitor, owner.get(), role, owner->bytes);
            }
        };
        PeerTree::visit_pages(
            peers_, [&](const auto& node) { page(node, GeometryContactPayloadRole::CartesianManifestSources); });
        for (const auto& peer : peers())
        {
            SourceTree::visit_pages(peer.sources,
                [&](const auto& node) { page(node, GeometryContactPayloadRole::CartesianManifestSources); });
            for (const auto& source : SourceRows(peer.sources.get()))
            {
                body(source.identity_, GeometryContactPayloadRole::CartesianManifestIdentity);
                body(source.irregular_ranges_, GeometryContactPayloadRole::CartesianManifestRanges);
                RangeTree::visit_pages(source.active_ranges_,
                    [&](const auto& node) { page(node, GeometryContactPayloadRole::CartesianManifestRanges); });
            }
            ExplicitTree::visit_pages(peer.explicit_cells,
                [&](const auto& node) { page(node, GeometryContactPayloadRole::CartesianExplicitCellHeaders); });
            for (const auto& cell : ExplicitRows(peer.explicit_cells.get()))
                body(cell.owner, GeometryContactPayloadRole::CartesianExplicitSupportKeys);
        }

        if (local_)
            local_->visit_borrowed_payload(visitor);
        for (const auto& lease : leases_)
            if (lease)
                lease->visit_borrowed_payload(visitor);
    }
    [[nodiscard]] bool local_dependencies_admitted_in(const CartesianAuthorityArena& arena) const noexcept
    {
        return budget_ == arena.budget && local_ && local_->dependencies_admitted_in(arena) &&
               receipts_admitted_in(arena);
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
                return visitor(
                    local_, local_->allocation_budget(), local_->metadata_bytes(), local_->has_arena_keeper());
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
        if constexpr (Mode != CartesianDependencyVisit::Flat)
            visit_receipt_dependencies<Mode>(visitor);
        for (const auto& lease : leases_)
            visitor(lease, lease->budget, lease->metadata, bool(lease->keeper));
    }

private:
    struct ExplicitCell
    {
        AdaptiveCellKey key;
        std::uint64_t revision = 0;
        std::span<const char> borrowed_members;
        std::shared_ptr<const ReceiptBytes> owner;
        [[nodiscard]] std::span<const char> member_bytes() const noexcept
        {
            return owner ? std::span<const char>(owner->bytes) : borrowed_members;
        }
    };
    struct SourceLess
    {
        bool operator()(const Source& a, const Source& b) const noexcept { return a.slot_ < b.slot_; }
    };
    struct ExplicitLess
    {
        bool operator()(const ExplicitCell& a, const ExplicitCell& b) const { return a.key < b.key; }
    };
    using SourceTree = persistent_ordered_detail::OrderedPages<Source, SourceLess>;
    using ExplicitTree = persistent_ordered_detail::OrderedPages<ExplicitCell, ExplicitLess>;
    using SourceRows = CartesianInventoryRows<Source, SourceTree::Node>;
    using ExplicitRows = CartesianInventoryRows<ExplicitCell, ExplicitTree::Node>;
    struct Peer
    {
        std::size_t ordinal = 0, next_slot = 0;
        int owner = -1;
        SourceTree::Ptr sources;
        ExplicitTree::Ptr explicit_cells;
    };
    struct PeerLess
    {
        bool operator()(const Peer& a, const Peer& b) const noexcept { return a.ordinal < b.ordinal; }
    };
    struct PeerWeights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t sources = 0;
        };
        static void add_row(Summary&, const Peer&, const Budget*);
        static void add_child(Summary&, const Summary&, bool);
    };
    using PeerTree = persistent_ordered_detail::OrderedPages<Peer, PeerLess, false,
        persistent_ordered_detail::OrderedPageWork, PeerWeights>;
    using Peers = CartesianInventoryRows<Peer, PeerTree::Node>;
    [[nodiscard]] Peers peers() const noexcept { return Peers(peers_.get()); }
    struct WirePeer
    {
        int owner;
        std::size_t source_begin, source_end, explicit_begin, explicit_end;
    };
    struct Builder;
    friend class NativeContactFundingAccess;
    static constexpr std::size_t source_control_bytes = 128;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const CartesianSourceManifestSet>&, std::size_t) const;
    CartesianSourceManifestSet() = default;
    static void decode_wire(const AMRGrid&, ContactHaloContext, std::uint64_t, std::span<const OwnerPacket>, int,
        std::vector<Source>&, std::vector<ExplicitCell>&, std::vector<WirePeer>&, cartesian_detail::Allocation&);
    static void decode_inventory(std::vector<Source>&, const AMRGrid&, ContactHaloContext, int, std::uint64_t,
        std::size_t, std::span<const char>, cartesian_detail::Allocation&);
    [[nodiscard]] bool receipts_admitted_in(const CartesianAuthorityArena&) const noexcept;
    [[nodiscard]] std::size_t irregular_range_admission(std::size_t ordinal) const
    {
        const auto& stored = source(ordinal).irregular_ranges_;
        return stored ? stored->metadata_bytes() : 0;
    }

    template<CartesianDependencyVisit Mode, class Visitor> void visit_receipt_dependencies(Visitor& visitor) const
    {
        const auto enter = [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
        {
            if constexpr (Mode == CartesianDependencyVisit::Observed)
                return visitor(owner, origin, bytes, keeper);
            else
            {
                visitor(owner, origin, bytes, keeper);
                return true;
            }
        };
        const auto enter_page = [&](const auto& node)
        { return node && enter(node, node->charge.arena.get(), node->charge.bytes, bool(keeper_)); };
        const auto enter_body = [&](const auto& owner)
        {
            if (owner)
                enter(owner, &owner->admission.arena(), owner->metadata_bytes(), bool(keeper_));
        };
        const auto visit_ranges = [&](auto&& self, const auto& node) -> void
        {
            if (enter_page(node))
                for (const auto& child : node->children)
                    self(self, child);
        };
        const auto visit_sources = [&](auto&& self, const SourceTree::Ptr& node) -> void
        {
            if (!enter_page(node))
                return;
            if (!node->leaf())
                for (const auto& child : node->children)
                    self(self, child);
            else
                for (const auto& source : node->rows)
                {
                    enter_body(source.identity_);
                    enter_body(source.irregular_ranges_);
                    visit_ranges(visit_ranges, source.active_ranges_);
                }
        };
        const auto visit_explicit = [&](auto&& self, const ExplicitTree::Ptr& node) -> void
        {
            if (!enter_page(node))
                return;
            if (!node->leaf())
                for (const auto& child : node->children)
                    self(self, child);
            else
                for (const auto& cell : node->rows)
                    enter_body(cell.owner);
        };
        const auto visit_peers = [&](auto&& self, const PeerTree::Ptr& node) -> void
        {
            if (!enter_page(node))
                return;
            if (!node->leaf())
                for (const auto& child : node->children)
                    self(self, child);
            else
                for (const auto& peer : node->rows)
                {
                    visit_sources(visit_sources, peer.sources);
                    visit_explicit(visit_explicit, peer.explicit_cells);
                }
        };
        visit_peers(visit_peers, peers_);
    }

    AMRGrid grid_;
    ContactHaloContext context_;
    std::uint64_t generation_ = 0;
    PeerTree::Ptr peers_;
    ReceiptWork receipt_work_;
    std::weak_ptr<const CartesianSourceManifestSet> predecessor_;
    PLIC::details::NativeWeakOwnerObservation predecessor_control_;
    bool shared_receipts_ = false;
    std::shared_ptr<const CartesianSourceInventory> local_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
    std::size_t charge_ = 0;
    std::size_t owners_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
