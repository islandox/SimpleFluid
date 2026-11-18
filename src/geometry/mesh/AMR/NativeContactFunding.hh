/** Closed financial receipts issued before contact owner type erasure. */
#pragma once
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/NativePaidExtent.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <memory>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class ContactPhysicalResidency;
class ContactSourcePointSnapshot;
class CartesianIncidenceSource;
class CartesianSourceInventory;
class CartesianSourceManifestSet;
class CartesianSourceLineRangeCatalog;
class CartesianSourceLineRangeRecipe;
class CertifiedIncidenceResultView;
class CertifiedIncidenceResultParts;
class CertifiedIncidenceLoopView;
class SharedSubfaceRing;
class PersistentContactFacts;
class ContactFanBlock;
class ContactFanOwner;
class TileBodyHandle;
struct SubfaceProvenancePage;
struct SubfacePiecePage;
struct TileBodyPageCredit;
struct TileSourceCredit;
template<class T, class Hasher, class Payload> class IdentityPool;
namespace cartesian_detail
{
struct AuthorityOwnerCredit;
struct CreditConstruction;
} // namespace cartesian_detail
namespace persistent_ordered_detail
{
template<class Row, class Less, bool FixedLeafCapacity, class Work, class Weights, std::size_t LeafCapacity,
    class Buffers> class OrderedPages;
}

namespace contact_funding_detail
{
/** The allowance covers a scalar credit and one separate shared control.
 * Store the actual resident deleter's address for authentication. A weak
 * target here would keep the shared control alive through its own deleter. */
template<std::size_t Bound> struct FixedControlCredit
{
    Budget* budget;
    std::shared_ptr<Budget> keeper;
    const void* resident_deleter = nullptr;

    static constexpr std::size_t allowance() noexcept { return Bound ? Bound : sizeof(FixedControlCredit) + 2 * 128; }
    FixedControlCredit(Budget& budget, const std::shared_ptr<Budget>& keeper) noexcept : budget(&budget), keeper(keeper) {}
    ~FixedControlCredit() { budget->used -= allowance(); }
    FixedControlCredit(const FixedControlCredit&) = delete;
    FixedControlCredit& operator=(const FixedControlCredit&) = delete;
};
using ControlCredit = FixedControlCredit<0>;

template<class T, std::size_t Bound> struct ControlDelete
{
    std::shared_ptr<FixedControlCredit<Bound>> credit;
    void operator()(T* body) const noexcept { delete body; }
};

template<std::size_t Bound = 0, class Factory>
[[nodiscard]] auto make_controlled(Budget& budget, const std::shared_ptr<Budget>& keeper, Factory make_body)
{
    using T = std::remove_pointer_t<decltype(make_body())>;
    using ControlCredit = FixedControlCredit<Bound>;
    using Delete = ControlDelete<T, Bound>;
#if defined(__GLIBCXX__)
    static_assert(sizeof(std::_Sp_counted_ptr_inplace<ControlCredit, std::allocator<void>,
                      __gnu_cxx::__default_lock_policy>) +
            sizeof(std::_Sp_counted_deleter<T*, Delete, std::allocator<void>, __gnu_cxx::__default_lock_policy>) <=
        ControlCredit::allowance());
#endif
    if (keeper && keeper.get() != &budget)
        throw std::invalid_argument("Contact fixed control has a foreign allocation arena.");
    budget.add(ControlCredit::allowance());
    std::shared_ptr<ControlCredit> credit;
    try
    {
        credit = std::make_shared<ControlCredit>(budget, keeper);
    }
    catch (...)
    {
        budget.used -= ControlCredit::allowance();
        throw;
    }
    std::shared_ptr<T> owner(make_body(), Delete{credit});
    const auto* deleter = std::get_deleter<Delete>(owner);
    if (!deleter)
        throw std::logic_error("Contact fixed control lacks its actual deleter.");
    credit->resident_deleter = deleter;
    return std::pair(std::move(owner), std::move(credit));
}

template<std::size_t Bound = 0, class T>
[[nodiscard]] auto control(const std::shared_ptr<const T>& owner) noexcept
    -> std::shared_ptr<const FixedControlCredit<Bound>>
{
    const auto* deleter = std::get_deleter<ControlDelete<T, Bound>>(owner);
    if (!deleter || !deleter->credit || deleter->credit->resident_deleter != deleter)
        return {};
    return deleter->credit;
}
} // namespace contact_funding_detail

/** The original payload allowance is partitioned into a separately deleted
 * body and its concrete fixed control group. The group can also fund another
 * body through a second actual shared control. */
struct NativeContactOwnerFunding
{
    PLIC::details::NativePaidExtentCharge body;
    std::size_t original_bytes = 0;
    PLIC::details::NativeWeakOwnerObservation related_control;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> dependent_fixed;
    std::shared_ptr<const void> index_accounting;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> index_owner, index_extent;
    geometry_detail::NativeConstructorStorageOwner index_storage;
};

/** A side manifest owns its actual row slots as well as its native metadata.
 * Visit it after its typed bodies on entry and before them on retirement. */
inline void project_contact_side_index(const PLIC::details::geometry_demand::GeometryOwnerManifest& root,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid = false)
{
    using namespace PLIC::details::geometry_demand;
    struct Edges
    {
        PLIC::details::GeometryOwnerGraphEditor& graph;
        bool retain, prepaid;
    } edges{graph, retain, prepaid};
    MetadataSink metadata{&edges,
        [](void* raw, const auto& owner, std::size_t, std::size_t admitted)
        {
            auto& value = *static_cast<Edges*>(raw);
            return value.graph.external(owner, admitted, value.retain, value.prepaid);
        }};
    if (retain)
        metadata.paid_extent = [](void* raw, const auto& charge)
        { static_cast<Edges*>(raw)->graph.record_paid_extent(charge); };
    root.visit_dependencies(metadata.with_financial(graph, retain), {&edges, [](void* raw, const auto& row)
    {
        auto& value = *static_cast<Edges*>(raw);
        value.graph.owner(row, value.retain, value.prepaid);
    }});
}

/** Only concrete audited financial holders can call issue(). A provider tag,
 * role, caller callback or raw Budget address cannot install funding authority.
 * capture() is private and only the actual contact producer/observer can use it.
 */
class SIMPLEFLUID_LOCAL NativeContactFundingAccess final
{
    friend class ContactPhysicalResidency;
    friend class ContactSourcePointSnapshot;
    friend class CartesianIncidenceSource;
    friend class CartesianSourceInventory;
    friend class CartesianSourceManifestSet;
    friend class CartesianSourceLineRangeCatalog;
    friend class CartesianSourceLineRangeRecipe;
    friend class CertifiedIncidenceResultView;
    friend class CertifiedIncidenceResultParts;
    friend class CertifiedIncidenceLoopView;
    friend class SharedSubfaceRing;
    friend class AdaptiveSubfaceSnapshot;
    friend struct SubfaceIndexRoot;
    friend class PersistentContactFacts;
    friend class ContactFanBlock;
    friend class ContactFanOwner;
    friend struct SubfaceProvenancePage;
    friend struct SubfacePiecePage;
    friend struct TileBodyPageCredit;
    friend struct TileSourceCredit;
    friend class TileBodyHandle;
    friend struct cartesian_detail::AuthorityOwnerCredit;
    friend struct cartesian_detail::CreditConstruction;
    template<class T, class Hasher, class Payload> friend class IdentityPool;
    template<class Row, class Less, bool FixedLeafCapacity, class Work, class Weights, std::size_t LeafCapacity,
        class Buffers> friend class persistent_ordered_detail::OrderedPages;
    using Charge = PLIC::details::NativePaidExtentCharge;
    using Funding = NativeContactOwnerFunding;
    using Graph = PLIC::details::GeometryOwnerGraphEditor;
    static constexpr auto budget_extent = sizeof(Budget) + 4 * sizeof(void*);
    template<class T>
    [[nodiscard]] static Funding capture(const std::shared_ptr<const T>& owner, std::size_t bytes, const Budget* origin)
    {
        if (!owner)
            return {};
        if constexpr (requires { owner->native_funding(owner, bytes); })
        {
            auto funding = owner->native_funding(owner, bytes);
            if (funding.index_owner)
            {
                const auto& arena = funding.index_owner->state()->mirror.owner;
                if (!funding.index_owner->matches(owner) || funding.original_bytes != bytes || arena.get() != origin)
                    throw std::invalid_argument("Native contact index changed its actual owner, extent or arena.");
            }
            else if (funding.original_bytes &&
                (funding.original_bytes != bytes ||
                    !funding.body.matches(Graph::external_kind(), owner, funding.body.bytes()) ||
                    !funding.body.valid() || funding.body.origin_address() != origin))
                throw std::invalid_argument("Typed contact funding differs from the actual physical owner/arena.");
            return funding;
        }
        else
            return {};
    }
    template<class T, class Group, class Extent, class Authenticate>
    [[nodiscard]] static Funding split(const std::shared_ptr<const T>& body, std::size_t original,
        const std::shared_ptr<const Group>& group, std::size_t replaced_allowance, Extent extent, Authenticate authenticate)
    {
        if (!group || !authenticate(body, group))
            throw std::invalid_argument("Contact funding has a foreign actual shared control.");
        const auto& arena = group->keeper;
        const std::shared_ptr<Budget> empty;
        if (!arena || (!arena.owner_before(empty) && !empty.owner_before(arena)))
            return {}; // A borrowed legacy arena has no detachable native funding.
        const auto body_bytes = extent(*body);
        if (!body_bytes || replaced_allowance > original || original - replaced_allowance != body_bytes)
            throw std::invalid_argument("Contact funding has an inconsistent concrete body extent.");
        const auto fixed = observe_control(body, group);
        const auto project = +[](const std::shared_ptr<const void>&, const geometry_detail::NativeConstructorOwnerSink&) {};
        const auto inspect = +[](const std::shared_ptr<const void>&, const PLIC::details::NativeFinancialInspectionSink&) {};
        const auto body_charge = Charge::intrinsic(Graph::external_kind(), body, body_bytes, arena, group, body_bytes,
            0, project, inspect,
            [](const auto& financial, const auto& physical, const auto& origin, std::uint8_t kind, std::size_t bytes,
                std::size_t snapshot, std::uint64_t revision)
            {
                const auto token = std::static_pointer_cast<const Group>(financial);
                const auto owner = std::static_pointer_cast<const T>(physical);
                return token && owner && token->keeper.get() == origin.get() &&
                    !token->keeper.owner_before(origin) && !origin.owner_before(token->keeper) &&
                    Authenticate{}(owner, token) && Extent{}(*owner) == bytes && bytes == snapshot && !revision &&
                    kind == Graph::external_kind();
            }, fixed, {});
        Funding result;
        result.body = body_charge;
        result.original_bytes = original;
        return result;
    }

    template<class T, class Group>
    [[nodiscard]] static auto observe_control(const std::shared_ptr<T>& body, const std::shared_ptr<Group>& group)
    {
        using Concrete = std::remove_const_t<Group>;
        return PLIC::details::NativeWeakOwnerObservation::contact_group(
            body, group, Concrete::allowance(), group->keeper, budget_extent);
    }

    template<class T, class Extent, class Identity>
    [[nodiscard]] static Funding split_named(
        const std::shared_ptr<const T>& owner, std::size_t original, Extent extent, Identity identity)
    {
        using namespace contact_funding_detail;
        auto group = control(owner);
        if (group.get() != identity(*owner)) group.reset();
        return split(owner, original, group, ControlCredit::allowance(), extent,
            [](const auto& actual, const auto& token)
            { return control(actual) == token && token.get() == Identity{}(*actual); });
    }

    template<std::size_t Bound, class T, class Extent, class Keeper>
    [[nodiscard]] static Funding split_named(const std::shared_ptr<const T>& owner, std::size_t original,
        Extent extent, Keeper keeper)
    {
        using contact_funding_detail::control;
        return split(owner, original, control<Bound>(owner), Bound, extent,
            [](const auto& actual, const auto& token)
            {
                const auto& own_arena = Keeper{}(*actual);
                return control<Bound>(actual) == token && own_arena == token->keeper &&
                    !own_arena.owner_before(token->keeper) && !token->keeper.owner_before(own_arena);
            });
    }

    template<class Body, class Finance>
    [[nodiscard]] static Funding borrow_body(const std::shared_ptr<const Body>& body, std::size_t original,
        std::size_t bytes, const std::shared_ptr<const Finance>& finance, const std::shared_ptr<Budget>& arena,
        const PLIC::details::NativeWeakOwnerObservation& body_control,
        const PLIC::details::NativeWeakOwnerObservation& financial_control, Charge::FinancialValidator validator)
    {
        if (!body_control || !financial_control || bytes > original)
            throw std::invalid_argument("Borrowed contact body lacks its concrete fixed controls and extent.");
        const auto project = +[](const std::shared_ptr<const void>&, const geometry_detail::NativeConstructorOwnerSink&) {};
        const auto inspect = +[](const std::shared_ptr<const void>&, const PLIC::details::NativeFinancialInspectionSink&) {};
        auto charge = Charge::intrinsic(Graph::external_kind(), body, bytes, arena, finance, bytes, 0,
            project, inspect, validator, body_control, financial_control).as_borrowed_financial();
        if (!charge.matches(Graph::external_kind(), body, bytes))
            throw std::invalid_argument("Borrowed contact body changed its exact physical control or funding.");
        Funding result;
        result.body = std::move(charge);
        result.original_bytes = original;
        return result;
    }

    static bool project(Graph& graph, const std::shared_ptr<const void>& body, std::size_t original,
        const Funding& funding, bool retain)
    {
        if (!funding.original_bytes)
            return graph.external(body, original, retain);
        if (funding.index_owner)
        {
            if (!graph.external(body, 0, retain)) return false;
            if (graph.external(funding.index_accounting, 0, retain))
            {
                graph.native_fixed(funding.index_extent, retain, false);
                graph.native_fixed(funding.index_owner, retain, false);
                graph.native_fixed(funding.index_storage.envelope, retain, false);
                if (retain) graph.native_storage(funding.index_storage);
                if (graph.external(funding.index_storage.owner, funding.index_storage.owner->exclusive_bytes(), retain))
                    graph.native_state(funding.index_storage.owner->admission().state(), retain, false);
            }
            return true;
        }
        if (retain)
            graph.record_paid_extent(funding.body);
        const auto changed = graph.external(body, funding.body.bytes(), retain);
        if (changed)
        {
            graph.weak_observation(funding.body.physical_observation(), retain);
            graph.weak_observation(funding.related_control, retain);
            if (funding.dependent_fixed) graph.native_fixed(funding.dependent_fixed, retain, false);
        }
        return changed;
    }

    /** Only the bytes whose original charge is imported by the legacy body
     * scope. Detached state, Budget, and related weak controls did not enter
     * that import even when the typed structural visitor can reach them. */
    static bool project_reimported(Graph& graph, const std::shared_ptr<const void>& body,
        const Funding& funding, bool retain)
    {
        if (!funding.original_bytes)
            throw std::invalid_argument("Inherited contact scope lacks a typed physical allocation.");
        if (funding.index_owner)
        {
            const auto fixed = [&](const auto& owner)
            { return graph.external(owner, owner->fixed_allocation_bytes(), retain); };
            if (!fixed(funding.index_owner)) return false;
            fixed(funding.index_extent);
            fixed(funding.index_storage.envelope);
            graph.external(funding.index_storage.owner, funding.index_storage.owner->exclusive_bytes(), retain);
            return true;
        }
        if (!graph.external(body, funding.body.bytes(), retain)) return false;
        const auto* group = funding.body.physical_observation().contact_control_group();
        const auto control = funding.original_bytes - funding.body.bytes();
        if (control)
        {
            if (!group || control != group->bytes)
                throw std::logic_error("Contact reimport scope has no exact fixed/body allowance partition.");
            graph.external(group->owner, control, retain);
        }
        return true;
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
