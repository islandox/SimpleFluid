#pragma once
#include "CartesianAuthorityCredits.hh"
#include "CartesianIncidenceAuthority.hh"
#include "CartesianSourceManifests.hh"
#include "ContactPayloadCensus.hh"
#include "ContactPointValueIndex.hh"
#include "PersistentOrderedPages.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveIncidenceSession;
class ContactSourcePointSnapshot;
template<class Hasher, class Payload> class IncidencePointDirectory;
/** A source namespace is distinct from immutable physical(page,offset) refs. */
struct ContactSourcePointReference
{
    std::uint64_t value = 0;
    bool operator==(const ContactSourcePointReference&) const = default;
};
/** Only the actual session can close the native inventory, real support/line
 * execution and normalization publication before creating this admission. */
class ContactSourcePointAdmission
{
    ContactSourcePointAdmission(AMRGrid grid, ContactHaloContext context, int owner, std::uint64_t generation,
        unsigned layer, std::size_t count, std::shared_ptr<const CartesianIncidenceSource> source,
        std::shared_ptr<const CartesianSourceManifestSet> manifests,
        std::shared_ptr<const ContactSourcePointSnapshot> previousSnapshot = {})
        : grid_(std::move(grid)), context_(context), owner_(owner), generation_(generation), layer_(layer),
          count_(count), source_(std::move(source)), manifests_(std::move(manifests)),
          previousSnapshot_(std::move(previousSnapshot))
    {
    }
    AMRGrid grid_;
    ContactHaloContext context_;
    int owner_;
    std::uint64_t generation_;
    unsigned layer_;
    std::size_t count_;
    std::shared_ptr<const CartesianIncidenceSource> source_;
    std::shared_ptr<const CartesianSourceManifestSet> manifests_;
    std::shared_ptr<const ContactSourcePointSnapshot> previousSnapshot_;
    friend class DistributedAdaptiveIncidenceSession;
    friend class ContactSourcePointSnapshot;
};
using ContactSourcePoint = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
using ContactSourcePointEmit = CartesianBorrowedCallback<const ContactSourcePoint&, std::uint64_t>;
using ContactSourcePointRead = CartesianBorrowedCallback<std::size_t, const ContactSourcePointEmit&>;
namespace cartesian_detail
{
/** Admit the manifest and its exact live grant or foreign physical graph.
 * The manifest header's arena alone says nothing about inherited credit. */
inline void retain_point_manifest(const std::shared_ptr<const CartesianSourceManifestSet>& manifest,
    CreditConstruction& credits, std::vector<std::shared_ptr<AuthorityOwnerCredit>>& leases, Allocation& memory)
{
    credits.retain(manifest, manifest->allocation_budget(), manifest->metadata_bytes(), manifest->has_arena_keeper(),
        leases, memory);
    manifest->retain_dependencies(credits, leases, memory);
}
} // namespace cartesian_detail
/** Lossless immutable cached values, not coordinate equivalence authority.
 * Every component-composed FULL key equals its actual original identity before
 * publication; otherwise the complete actual identity remains an exception.
 * Reads copy canonical components, never normalize or allocate. */
class ContactSourcePointSnapshot
{
public:
    static void project(const std::shared_ptr<const ContactSourcePointSnapshot>& source,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
    {
        if (!source) return;
        using Funding = NativeContactFundingAccess;
        const auto source_bytes = source->metadata_bytes();
        if (!Funding::project(graph, source, source_bytes,
                Funding::capture(source, source_bytes, source->allocation_budget()), retain))
            return;
        source->visit_dependencies<CartesianDependencyVisit::Observed>(
            [&](const auto& owner, const Budget* origin, std::size_t bytes, bool)
            {
                using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
                const std::shared_ptr<const std::remove_const_t<Element>> stable = owner;
                return Funding::project(graph, stable, bytes, Funding::capture(stable, bytes, origin), retain);
            });
    }
    static std::shared_ptr<const ContactSourcePointSnapshot> capture(const ContactSourcePointAdmission&, bool hashing,
        const ContactSourcePointRead&, Budget&, std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>,
        bool canonical_hash = true);
    ~ContactSourcePointSnapshot();
    ContactSourcePointSnapshot(const ContactSourcePointSnapshot&) = delete;
    ContactSourcePointSnapshot& operator=(const ContactSourcePointSnapshot&) = delete;
    using LookupWork = ContactPointValueIndex::Probe;
    [[nodiscard]] ContactSourcePoint value(ContactSourcePointReference, LookupWork* = nullptr) const;
    /** route_hash must come from the configured deterministic lookup policy.
     * An arbitrary incorrect hash does not establish an identity lookup. */
    [[nodiscard]] std::optional<std::size_t> find(
        const ContactSourcePoint&, std::uint64_t route_hash, LookupWork* = nullptr) const;
    [[nodiscard]] std::optional<std::array<std::size_t, 3>> component_indices(ContactSourcePointReference) const;
    [[nodiscard]] std::optional<std::size_t> find_components(const std::array<std::size_t, 3>&) const;
    /** Read an already issued canonical dictionary token. No arbitrary scalar
     * can be inserted or used as an authority through this accessor. */
    [[nodiscard]] const PLIC::details::CanonicalPointComponent& canonical_component(
        unsigned axis, std::size_t index) const
    {
        return component(axis, index);
    }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] bool hashing() const noexcept { return hashing_; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge_ + control_charge_; }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return budget_; }
    [[nodiscard]] bool has_arena_keeper() const noexcept { return bool(keeper_); }
    [[nodiscard]] std::size_t exception_count() const noexcept;
    [[nodiscard]] std::size_t storage_block_count() const noexcept { return blocks_ ? blocks_->size : 0; }
    /** Local immutable slot lineage, supplemental to complete identity proof.
     * Fork/recapture shares this DATA block; no historical proof is retained. */
    [[nodiscard]] const void* slot_namespace() const noexcept { return slot_namespace_; }
    [[nodiscard]] std::size_t new_storage_records() const noexcept { return new_records_; }
    [[nodiscard]] std::size_t copied_directory_bytes() const noexcept { return directory_work_.copied_bytes; }
    [[nodiscard]] std::size_t indexed_axis_values() const noexcept { return indexed_axis_values_; }
    [[nodiscard]] std::size_t value_build_admitted_bytes() const noexcept { return value_build_bytes_; }
    [[nodiscard]] std::size_t verified_prefix_records() const noexcept { return verified_prefix_records_; }
    [[nodiscard]] const ContactPointValueIndex::Work& index_build_work() const noexcept { return index_work_; }
    [[nodiscard]] std::size_t index_metadata_bytes() const noexcept { return indexes_.metadata_bytes(); }
    [[nodiscard]] std::size_t index_build_admitted_bytes() const noexcept { return index_build_bytes_; }
    [[nodiscard]] std::size_t value_storage_bytes() const noexcept;
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept
    {
        return {control_identity_, control_payload_bytes_};
    }
    [[nodiscard]] std::array<std::size_t, 3> component_counts() const noexcept;
    [[nodiscard]] std::size_t selector_width() const noexcept;
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] unsigned certified_layer() const noexcept { return layer_; }
    [[nodiscard]] const AMRGrid& grid() const noexcept { return grid_; }
    [[nodiscard]] const std::shared_ptr<const CartesianIncidenceSource>& source() const noexcept { return source_; }
    [[nodiscard]] CartesianEvaluationEnvironment evaluation_environment() const noexcept { return evaluation_; }
    [[nodiscard]] const void* admission_identity() const noexcept { return &charge_; }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(
            visitor, this, GeometryContactPayloadRole::SourcePointSnapshotObject, sizeof(*this), true);
        BlockDirectory::visit_pages(blocks_,
            [&](const auto& page)
            {
                constexpr auto role = GeometryContactPayloadRole::SourcePointDirectoryObject;
                contact_payload_object(visitor, page.get(), role, sizeof(*page), true);
                contact_payload_buffer(visitor, page.get(), role, page->children);
                contact_payload_buffer(visitor, page.get(), role, page->rows);
                for (const auto& block : page->rows)
                    block->visit_borrowed_payload(visitor);
            });
        indexes_.visit_borrowed_payload(visitor);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::SourcePointLeases, leases_);
        if (control_identity_)
            contact_payload_object(visitor, control_identity_, GeometryContactPayloadRole::CartesianControlCreditObject,
                control_payload_bytes_, true);
        if (source_)
            source_->visit_borrowed_payload(visitor);
        if (manifests_)
            manifests_->visit_borrowed_payload(visitor);
        for (const auto& lease : leases_)
            if (lease)
                lease->visit_borrowed_payload(visitor);
    }
    template<CartesianDependencyVisit Mode = CartesianDependencyVisit::Flat, class Visitor>
    void visit_dependencies(Visitor&& visitor) const
    {
        visit_value_dependencies<Mode == CartesianDependencyVisit::Observed>(blocks_, visitor);
        indexes_.visit_dependencies<Mode == CartesianDependencyVisit::Observed>(visitor);
        if (source_)
            visitor(source_, source_->allocation_budget(), source_->metadata_bytes(), source_->has_arena_keeper());
        if (manifests_)
        {
            const auto visit_manifest = [&]() -> decltype(auto)
            {
                return visitor(manifests_, manifests_->allocation_budget(), manifests_->metadata_bytes(),
                    manifests_->has_arena_keeper());
            };
            if constexpr (std::is_same_v<decltype(visit_manifest()), bool>)
            {
                if (visit_manifest())
                    manifests_->visit_dependencies<Mode>(visitor);
            }
            else
            {
                visit_manifest();
                manifests_->visit_dependencies<Mode>(visitor);
            }
        }
        for (const auto& lease : leases_)
            visitor(lease, lease->budget, lease->metadata, bool(lease->keeper));
    }

private:
    friend class NativeContactFundingAccess;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const ContactSourcePointSnapshot>&, std::size_t) const;
    template<class Hasher, class Payload> friend class IncidencePointDirectory;
    static std::shared_ptr<const ContactSourcePointSnapshot> capture_impl(const ContactSourcePointAdmission&,
        bool hashing, const ContactSourcePointRead&, Budget&, std::shared_ptr<Budget>,
        std::shared_ptr<CartesianAuthorityArena>, bool canonical_hash,
        const std::shared_ptr<const ContactSourcePointSnapshot>* accepted_prefix);
    [[nodiscard]] static bool same_snapshot(const std::shared_ptr<const ContactSourcePointSnapshot>& a,
        const std::shared_ptr<const ContactSourcePointSnapshot>& b) noexcept
    {
        return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
    }
    [[nodiscard]] static bool exact_control(const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot) noexcept
    {
        return snapshot && !snapshot->self_.owner_before(snapshot) && !snapshot.owner_before(snapshot->self_);
    }
    ContactSourcePointSnapshot() = default;
    using Component = PLIC::details::CanonicalPointComponent;
    template<class UInt> using Selectors = std::vector<std::array<UInt, 3>>;
    /** Independent immutable values only: no proof, source, or snapshot ancestry. */
    struct ValueBlock
    {
        std::size_t first = 0, count = 0;
        std::array<std::size_t, 3> component_first{};
        std::array<std::vector<Component>, 3> components;
        std::variant<Selectors<std::uint8_t>, Selectors<std::uint16_t>, Selectors<std::uint32_t>,
            Selectors<std::uint64_t>>
            selectors;
        std::vector<std::pair<std::size_t, ContactSourcePoint>> exceptions;
        Budget* budget = nullptr;
        std::shared_ptr<Budget> keeper;
        std::size_t charge = 0, control_charge = 0;
        // Observed only while this body lives; its deleter retains the token.
        const void* control_identity = nullptr;
        std::size_t control_payload_bytes = 0;
        ~ValueBlock();
        [[nodiscard]] const Component* component_at(unsigned axis, std::size_t index) const noexcept
        {
            const auto first_slot = component_first[axis];
            const auto& values = components[axis];
            return index >= first_slot && index - first_slot < values.size() ? &values[index - first_slot] : nullptr;
        }
        [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge + control_charge; }
        [[nodiscard]] NativeContactOwnerFunding native_funding(
            const std::shared_ptr<const ValueBlock>&, std::size_t) const;
        [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept
        {
            return {control_identity, control_payload_bytes};
        }
        template<class Visitor> void visit_borrowed_payload(Visitor& visitor) const
        {
            contact_payload_object(
                visitor, this, GeometryContactPayloadRole::SourcePointValueBlockObject, sizeof(*this), true);
            if (control_identity)
                contact_payload_object(visitor, control_identity,
                    GeometryContactPayloadRole::CartesianControlCreditObject, control_payload_bytes, true);
            for (const auto& values : components)
                contact_payload_buffer(visitor, this, GeometryContactPayloadRole::SourcePointComponents, values);
            std::visit([&](const auto& values)
                { contact_payload_buffer(visitor, this, GeometryContactPayloadRole::SourcePointSelectors, values); },
                selectors);
            contact_payload_buffer(visitor, this, GeometryContactPayloadRole::SourcePointExceptions, exceptions);
        }
    };
    using Block = std::shared_ptr<const ValueBlock>;
    struct BlockLess
    {
        static constexpr auto allocation_domain = geometry_detail::AllocationDomain::ContactHalo;
        bool operator()(const Block& a, const Block& b) const noexcept { return a->first < b->first; }
    };
    using BlockDirectory = persistent_ordered_detail::OrderedPages<Block, BlockLess>;
    template<bool Prune, class Visitor>
    static void visit_value_dependencies(const BlockDirectory::Ptr& page, Visitor&& visit)
    {
        if (!page)
            return;
        const std::shared_ptr<Budget> empty;
        const bool keeper = page->charge.arena.owner_before(empty) || empty.owner_before(page->charge.arena);
        if constexpr (Prune)
        {
            if (!visit(page, page->charge.arena.get(), page->charge.bytes, keeper))
                return;
        }
        else
            visit(page, page->charge.arena.get(), page->charge.bytes, keeper);
        for (const auto& child : page->children)
            visit_value_dependencies<Prune>(child, visit);
        for (const auto& block : page->rows)
            visit(block, block->budget, block->metadata_bytes(), bool(block->keeper));
    }
    template<class Visitor> bool visit_blocks(const Visitor& visit) const
    {
        return BlockDirectory::visit_from(blocks_, [](const Block&) { return false; }, visit);
    }
    [[nodiscard]] ContactPointValueIndex::Reader index_reader() const noexcept;
    [[nodiscard]] const ValueBlock& block_at(std::size_t, LookupWork* = nullptr) const;
    [[nodiscard]] const Component& component(unsigned axis, std::size_t index, LookupWork* = nullptr) const;
    [[nodiscard]] std::optional<std::size_t> find_component(
        unsigned axis, long double value, LookupWork* = nullptr) const;
    BlockDirectory::Ptr blocks_;
    ContactPointValueIndex indexes_;
    ContactPointValueIndex::Work index_work_;
    std::size_t index_build_bytes_ = 0;
    std::weak_ptr<const ContactSourcePointSnapshot> self_;
    const void* slot_namespace_ = nullptr; // The first block is pinned by blocks_.
    std::array<std::size_t, 3> component_counts_{};
    persistent_ordered_detail::OrderedPageWork directory_work_;
    std::size_t indexed_axis_values_ = 0, verified_prefix_records_ = 0, value_build_bytes_ = 0;
    std::size_t new_records_ = 0, control_charge_ = 0;
    // Non-owning observation; custom deleter keeps this typed token alive.
    const void* control_identity_ = nullptr;
    std::size_t control_payload_bytes_ = 0;
    AMRGrid grid_;
    ContactHaloContext context_;
    CartesianEvaluationEnvironment evaluation_;
    int owner_ = -1;
    std::uint64_t generation_ = 0;
    unsigned layer_ = 0;
    std::size_t count_ = 0, charge_ = 0;
    bool hashing_ = false;
    bool canonical_hash_ = false;
    std::shared_ptr<const CartesianIncidenceSource> source_;
    std::shared_ptr<const CartesianSourceManifestSet> manifests_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    Budget* budget_ = nullptr;
    std::shared_ptr<Budget> keeper_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
