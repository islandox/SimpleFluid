#pragma once
#include "DeferredOwnerCharges.hh"
#include "IdentityPool.hh"
#include "SmallMembership.hh"
#include "SupportRegistry.hh"
#include <functional>
#include <map>
#include <memory>
#include <set>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Compact membership references immutable full identities through pools. */
struct PooledSupportMember
{
    std::size_t cell = 0;
    std::uint64_t revision = 0, surface = 0, encoded_size = 0;
    int owner = -1;
};
class PooledSupportRegistry
{
public:
    using BytesPool = IdentityPool<AdaptiveBytes>;
    using CellPool = IdentityPool<AdaptiveCellKey>;
    using Slot = std::size_t;
    using Members = SmallMembership<PooledSupportMember>;
    PooledSupportRegistry(Budget&, BytesPool&, CellPool&);
    PooledSupportRegistry(const PooledSupportRegistry&, Budget&, BytesPool&, CellPool&);
    ~PooledSupportRegistry();
    PooledSupportRegistry(const PooledSupportRegistry&) = delete;
    PooledSupportRegistry& operator=(const PooledSupportRegistry&) = delete;
    void apply(std::span<const RegistryChange>);
    [[nodiscard]] const Members* find(Slot support) const;
    [[nodiscard]] std::vector<Slot> dirty() const;
    [[nodiscard]] const AdaptiveBytes& support_key(Slot slot) const { return supports_->get(slot); }
    [[nodiscard]] std::uint64_t generation(Slot support) const;
    [[nodiscard]] RegistryMember materialize(const PooledSupportMember&) const;
    [[nodiscard]] const AdaptiveCellKey& member_cell(const PooledSupportMember& member) const
    {
        return cells_->get(member.cell);
    }
    [[nodiscard]] std::size_t dirty_count() const noexcept { return dirty_.size(); }
    void for_each_dirty(const std::function<void(Slot)>& visit) const
    {
        for (const auto slot : dirty_)
            visit(slot);
    }
    [[nodiscard]] const auto& dirty_slots() const noexcept { return dirty_; }
    void clear_dirty();
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return groups_.size() * group_node_bytes + dirty_.size() * dirty_node_bytes + deferred_.metadata_bytes();
    }
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return charged_ + deferred_.retained_bytes(); }
    void reap_deferred() noexcept { deferred_.reap(); }

private:
    struct Group
    {
        std::uint64_t generation = 0;
        Members members;
    };
    using Buckets = std::map<Slot, Group>;
    static constexpr std::size_t group_node_bytes = sizeof(Buckets::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t dirty_node_bytes = sizeof(std::set<Slot>::value_type) + 4 * sizeof(void*);
    void admit(std::size_t);
    void release(std::size_t);
    Budget* budget_;
    DeferredOwnerCharges deferred_;
    BytesPool* supports_;
    CellPool* cells_;
    std::size_t charged_ = 0;
    Buckets groups_;
    struct SupportLess
    {
        const BytesPool* pool;
        bool operator()(Slot a, Slot b) const { return pool->get(a) < pool->get(b); }
    };
    std::set<Slot, SupportLess> dirty_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
