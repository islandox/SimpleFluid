#pragma once
#include "CartesianAuthorityCredits.hh"
#include "CartesianIncidenceAuthority.hh"
#include "CartesianSourceManifests.hh"
#include "ContactPayloadCensus.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <type_traits>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveIncidenceSession;
class ContactSourcePointSnapshot;
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
/** Lossless immutable cached values, not coordinate equivalence authority.
 * Every component-composed FULL key equals its actual original identity before
 * publication; otherwise the complete actual identity remains an exception.
 * Reads copy canonical components, never normalize or allocate. */
class ContactSourcePointSnapshot
{
public:
    static std::shared_ptr<const ContactSourcePointSnapshot> capture(const ContactSourcePointAdmission&, bool hashing,
        const ContactSourcePointRead&, Budget&, std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>,
        bool canonical_hash = true);
    ~ContactSourcePointSnapshot();
    ContactSourcePointSnapshot(const ContactSourcePointSnapshot&) = delete;
    ContactSourcePointSnapshot& operator=(const ContactSourcePointSnapshot&) = delete;
    [[nodiscard]] ContactSourcePoint value(ContactSourcePointReference) const;
    [[nodiscard]] std::optional<std::size_t> find(const ContactSourcePoint&, std::uint64_t route_hash) const;
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
    [[nodiscard]] std::size_t storage_block_count() const noexcept { return blocks_.size(); }
    /** Local immutable slot lineage, supplemental to complete identity proof.
     * Fork/recapture shares this DATA block; no historical proof is retained. */
    [[nodiscard]] const void* slot_namespace() const noexcept
    {
        return blocks_.empty() ? nullptr : blocks_.front().get();
    }
    [[nodiscard]] std::size_t new_storage_records() const noexcept { return new_records_; }
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
        contact_payload_object(visitor, this, ContactPayloadRole::SourcePointSnapshotObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointDirectoryObject, blocks_);
        for (const auto& block : blocks_)
            block->visit_borrowed_payload(visitor);
        contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointLeases, leases_);
        if (control_identity_)
            contact_payload_object(visitor, control_identity_, ContactPayloadRole::CartesianControlCreditObject,
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
        for (const auto& block : blocks_)
            visitor(block, block->budget, block->metadata_bytes(), bool(block->keeper));
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
        std::variant<std::vector<std::uint32_t>, std::vector<std::uint64_t>> lookup;
        std::vector<std::pair<std::size_t, ContactSourcePoint>> exceptions;
        Budget* budget = nullptr;
        std::shared_ptr<Budget> keeper;
        std::size_t charge = 0, control_charge = 0;
        // Observed only while this body lives; its deleter retains the token.
        const void* control_identity = nullptr;
        std::size_t control_payload_bytes = 0;
        ~ValueBlock();
        [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge + control_charge; }
        [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept
        {
            return {control_identity, control_payload_bytes};
        }
        template<class Visitor> void visit_borrowed_payload(Visitor& visitor) const
        {
            contact_payload_object(visitor, this, ContactPayloadRole::SourcePointValueBlockObject, sizeof(*this), true);
            if (control_identity)
                contact_payload_object(visitor, control_identity, ContactPayloadRole::CartesianControlCreditObject,
                    control_payload_bytes, true);
            for (const auto& values : components)
                contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointComponents, values);
            std::visit([&](const auto& values)
                { contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointSelectors, values); },
                selectors);
            std::visit([&](const auto& values)
                { contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointLookup, values); }, lookup);
            contact_payload_buffer(visitor, this, ContactPayloadRole::SourcePointExceptions, exceptions);
        }
    };
    [[nodiscard]] const ValueBlock& block_at(std::size_t) const;
    [[nodiscard]] const Component& component(unsigned axis, std::size_t index) const;
    std::vector<std::shared_ptr<const ValueBlock>> blocks_;
    std::array<std::size_t, 3> component_counts_{};
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
