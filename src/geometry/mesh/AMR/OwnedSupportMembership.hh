#pragma once
#include "CartesianAuthorityCredits.hh"
#include "CartesianIncidenceAuthority.hh"
#include "CartesianIncidenceSource.hh"
#include "CartesianSourceInventory.hh"
#include "DeferredOwnerCharges.hh"
#include "IncidencePools.hh"
#include "SmallMembership.hh"
#include "SupportRegistry.hh"
#include <map>
#include <memory>
#include <set>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Cached owner-local surface metadata makes retirement independent of a leaf
 * that the hierarchy may already have removed. No geometry pointers retained. */
class OwnedSupportMembership
{
public:
    using Slot = std::size_t;
    struct Member
    {
        Slot support = 0;
        std::uint64_t encoded_size = 0;
    };
    struct Cell
    {
        std::uint64_t revision = 0;
        std::vector<Member> members;
        std::size_t charge = 0;
        const CartesianIncidenceSource* construction_source = nullptr;
    };
    OwnedSupportMembership(Budget&, IncidencePools&);
    OwnedSupportMembership(const OwnedSupportMembership&, Budget&, IncidencePools&,
        std::shared_ptr<CartesianAuthorityArena> source_arena = {});
    ~OwnedSupportMembership();
    OwnedSupportMembership(const OwnedSupportMembership&) = delete;
    OwnedSupportMembership& operator=(const OwnedSupportMembership&) = delete;
    [[nodiscard]] Slot add(const AdaptiveLeaf&);
    [[nodiscard]] Slot add(const CartesianNativeIncidenceCell&);
    /** Publish proofs collected from the original native construction, after
     * actual support/line completion. A mismatch keeps every explicit record. */
    [[nodiscard]] bool publish_constructed_source();
    [[nodiscard]] bool has_complete_construction_proof() const;
    [[nodiscard]] const std::fenv_t& construction_evaluation_state() const noexcept
    {
        return construction_environment_;
    }
    [[nodiscard]] const std::shared_ptr<const CartesianIncidenceSource>& source() const noexcept { return source_; }
    [[nodiscard]] const std::shared_ptr<const CartesianSourceInventory>& inventory() const noexcept
    {
        return inventory_;
    }
    void publish_inventory(std::shared_ptr<const CartesianSourceInventory>, std::shared_ptr<CartesianAuthorityArena>);
    [[nodiscard]] const std::shared_ptr<const CartesianIncidenceSource>& construction_source() const noexcept
    {
        return construction_source_;
    }
    [[nodiscard]] std::size_t source_cell_count() const noexcept
    {
        return inventory_ ? inventory_->cell_count()
                          : (source_ ? source_->cell_count() - retired_source_cells_.size() : 0);
    }
    [[nodiscard]] std::size_t explicit_cell_count() const noexcept { return cells_.size(); }
    [[nodiscard]] const Cell& at(Slot) const;
    [[nodiscard]] std::size_t member_count(Slot) const;
    [[nodiscard]] std::uint64_t leaf_revision(Slot) const;
    void visit_members(Slot, const CartesianBorrowedCallback<std::size_t, const Member&>&) const;
    void visit_support_members(
        Slot, const CartesianBorrowedCallback<Slot, std::uint64_t, std::size_t, const Member&>&) const;
    [[nodiscard]] bool has_support(Slot) const;
    [[nodiscard]] std::size_t maximum_change_capacity() const;
    /** Called only after the real session has completed support and all line
     * layers. Exact metadata replay precedes releasing any regular records.
     * Publication is candidate-local; a throwing candidate is discarded. */
    void certify_source(
        std::shared_ptr<const CartesianIncidenceSource>, std::shared_ptr<CartesianAuthorityArena> source_arena = {});
    [[nodiscard]] Slot require(const AdaptiveCellKey&, std::uint64_t revision) const;
    [[nodiscard]] RegistryChange change(Slot, std::size_t member, int owner, bool remove) const;
    using SupportCells = SmallMembership<Slot>;
    [[nodiscard]] const SupportCells* support_cells(Slot support) const;
    void erase(Slot);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return cells_.size() * cell_node_bytes + reverse_.size() * reverse_node_bytes +
               retired_source_cells_.size() * retired_node_bytes + deferred_.metadata_bytes() +
               source_leases_.capacity() * sizeof(source_leases_[0]);
    }
    [[nodiscard]] std::size_t retained_bytes() const noexcept
    {
        return charged_ + deferred_.retained_bytes() +
               (source_ && source_->allocation_budget() == budget_ ? source_->metadata_bytes() : 0) +
               (!source_ && construction_source_ && construction_source_->allocation_budget() == budget_
                       ? construction_source_->metadata_bytes()
                       : 0);
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        for (const auto& entry : cells_)
        {
            contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipNodes, entry);
            contact_payload_object(
                visitor, entry.second.get(), ContactPayloadRole::OwnedMembershipCell, sizeof(Cell), true);
            contact_payload_buffer(
                visitor, entry.second.get(), ContactPayloadRole::OwnedMembershipMembers, entry.second->members);
        }
        for (const auto& entry : reverse_)
        {
            contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipReverseNodes, entry);
            if (const auto& spill = entry.second.spill())
            {
                contact_payload_object(
                    visitor, spill.get(), ContactPayloadRole::OwnedMembershipSpill, sizeof(SupportCells::Spill), true);
                contact_payload_buffer(visitor, spill.get(), ContactPayloadRole::OwnedMembershipValues, *spill);
            }
        }
        for (const auto& entry : retired_source_cells_)
            contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipRetiredNodes, entry);
        contact_payload_buffer(visitor, this, ContactPayloadRole::OwnedMembershipLeases, source_leases_);
        for (const auto& lease : source_leases_)
            lease->visit_borrowed_payload(visitor);
        if (inventory_)
            inventory_->visit_borrowed_payload(visitor);
        else if (source_)
            source_->visit_borrowed_payload(visitor);
        if (construction_source_)
            construction_source_->visit_borrowed_payload(visitor);
        deferred_.visit_borrowed_payload(visitor);
    }
    void reap_deferred() noexcept { deferred_.reap(); }

private:
    friend class CartesianSourceManifestSet;
    using Index = std::map<Slot, std::shared_ptr<const Cell>>;
    using ReverseIndex = std::map<Slot, SupportCells>;
    static constexpr std::size_t cell_node_bytes = sizeof(Index::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t reverse_node_bytes = sizeof(ReverseIndex::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t retired_node_bytes = sizeof(Slot) + 4 * sizeof(void*);
    [[nodiscard]] const CartesianCellAncestry* source_cell(Slot) const;
    [[nodiscard]] const CartesianSourceInventory::Batch* source_batch(Slot) const;
    [[nodiscard]] std::array<Member, 6> generated_members(Slot) const;
    void update_reverse(Slot support, Slot cell, bool remove);
    [[nodiscard]] Slot add_leaf(const AdaptiveLeaf&, const CartesianIncidenceSource*);
    void admit(std::size_t);
    void release(std::size_t);
    Budget* budget_;
    DeferredOwnerCharges deferred_;
    IncidencePools* pools_;
    std::size_t charged_ = 0;
    Index cells_;
    ReverseIndex reverse_;
    std::shared_ptr<const CartesianIncidenceSource> source_;
    std::shared_ptr<const CartesianSourceInventory> inventory_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> source_leases_;
    std::shared_ptr<const CartesianIncidenceSource> construction_source_;
    std::fenv_t construction_environment_{};
    std::set<Slot> retired_source_cells_;
    std::fenv_t source_environment_{}; // Local cached controls; never serialized.
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
