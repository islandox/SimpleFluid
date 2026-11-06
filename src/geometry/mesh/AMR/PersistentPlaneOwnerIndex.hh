/** @file PersistentPlaneOwnerIndex.hh
 * @brief Selective admission leases for shared plane backing and ancestry.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** The index deduplicates physical owner allocations. A captured identity
 * page keeps its roots directly, with each root owning only its ancestry. */
class PersistentPlaneOwnerIndex
{
public:
    using Plane = PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>;
    struct Owner
    {
        persistent_ordered_detail::PageCharge charge;
        std::shared_ptr<const void> backing;
        std::shared_ptr<const Owner> parent;
        std::size_t payload;

        Owner(std::shared_ptr<Budget> arena, const PLIC::GeometryIdentityAllocation& allocation,
            std::shared_ptr<const Owner> ancestry)
            : charge(std::move(arena)), backing(allocation.owner), parent(std::move(ancestry)),
              payload(allocation.payload_bytes)
        {
        }
    };

private:
    struct Selector
    {
        const void* address;
        std::shared_ptr<const Owner> owner;
    };
    struct Less
    {
        bool operator()(const Selector& a, const Selector& b) const noexcept
        {
            return std::less<const void*>{}(a.address, b.address);
        }
    };
    using Index = persistent_ordered_detail::OrderedPages<Selector, Less>;

public:
    using Root = Index::Ptr;
    struct Retained
    {
        Root index;
        std::shared_ptr<const Owner> owner;
        std::size_t new_payload = 0;
    };

    explicit PersistentPlaneOwnerIndex(std::shared_ptr<Budget> arena) : arena_(std::move(arena)) {}

    [[nodiscard]] Retained retain(Root root, const Plane& plane) const
    {
        using namespace persistent_ordered_detail;
        PageCharge scratch(arena_);
        std::vector<PLIC::GeometryIdentityAllocation> unseen;
        std::shared_ptr<const Owner> ancestry;
        PLIC::visit_identity_allocations(plane,
            [&](const PLIC::GeometryIdentityAllocation& allocation)
            {
                const Selector key{allocation.owner.get(), {}};
                const auto* found = Index::lower_bound(root, [&](const auto& row) { return Less{}(row, key); });
                if (!found || Less{}(key, *found))
                {
                    scratch.append(unseen);
                    unseen.push_back(allocation);
                    return true;
                }
                ancestry = found->owner;
                if (ancestry->payload != allocation.payload_bytes || ancestry->backing.owner_before(allocation.owner) ||
                    allocation.owner.owner_before(ancestry->backing))
                    throw std::invalid_argument("Persistent plane owner changed its allocation.");
                return false;
            });

        std::size_t added = 0;
        const Index index(arena_);
        for (auto at = unseen.rbegin(); at != unseen.rend(); ++at)
        {
            PageCharge memory(arena_);
            memory.add(checked_sum(at->payload_bytes, sizeof(Owner) + control_bytes));
            auto owner = std::make_shared<Owner>(arena_, *at, std::move(ancestry));
            added = checked_sum(added, memory.bytes);
            owner->charge.bytes = std::exchange(memory.bytes, 0);
            const Selector selector{at->owner.get(), owner};
            root = index.edit(root, selector, &selector, false);
            ancestry = std::move(owner);
        }
        return {std::move(root), std::move(ancestry), added};
    }

    template<class Visit> static void visit_pages(const Root& root, const Visit& visit)
    {
        Index::visit_pages(root, visit);
    }

private:
    std::shared_ptr<Budget> arena_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
