#pragma once
#include "DeferredOwnerCharges.hh"
#include "IncidencePools.hh"
#include "SmallMembership.hh"
#include "SupportRegistry.hh"
#include <map>
#include <memory>

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
    };
    OwnedSupportMembership(Budget&, IncidencePools&);
    OwnedSupportMembership(const OwnedSupportMembership&, Budget&, IncidencePools&);
    ~OwnedSupportMembership();
    OwnedSupportMembership(const OwnedSupportMembership&) = delete;
    OwnedSupportMembership& operator=(const OwnedSupportMembership&) = delete;
    [[nodiscard]] Slot add(const AdaptiveLeaf&);
    [[nodiscard]] const Cell& at(Slot) const;
    [[nodiscard]] Slot require(const AdaptiveCellKey&, std::uint64_t revision) const;
    [[nodiscard]] RegistryChange change(Slot, std::size_t member, int owner, bool remove) const;
    using SupportCells = SmallMembership<Slot>;
    [[nodiscard]] const SupportCells* support_cells(Slot support) const;
    void erase(Slot);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return cells_.size() * cell_node_bytes + reverse_.size() * reverse_node_bytes + deferred_.metadata_bytes();
    }
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return charged_ + deferred_.retained_bytes(); }
    void reap_deferred() noexcept { deferred_.reap(); }

private:
    using Index = std::map<Slot, std::shared_ptr<const Cell>>;
    using ReverseIndex = std::map<Slot, SupportCells>;
    static constexpr std::size_t cell_node_bytes = sizeof(Index::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t reverse_node_bytes = sizeof(ReverseIndex::value_type) + 4 * sizeof(void*);
    void update_reverse(Slot support, Slot cell, bool remove);
    void admit(std::size_t);
    void release(std::size_t);
    Budget* budget_;
    DeferredOwnerCharges deferred_;
    IncidencePools* pools_;
    std::size_t charged_ = 0;
    Index cells_;
    ReverseIndex reverse_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
