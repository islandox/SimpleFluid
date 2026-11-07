/** @file IncidencePhysicalOwners.hh
 * @brief Exact owner events for incidence and detached source proofs.
 */
#pragma once
#include "CartesianAuthorityCredits.hh"
#include "ContactPhysicalResidency.hh"
#include "ContactSourcePoints.hh"
#include <limits>
#include <memory>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** A borrowed edit observer. The typed proof itself remains the manifest and
 * continues to pin its owners. No detached view retains this mutable editor. */
class IncidencePhysicalOwners
{
public:
    using Role = ContactPhysicalResidency::Role;
    using Credits = std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>;
    static constexpr auto tile_credit_scope = std::numeric_limits<ContactPhysicalResidency::PointScope>::max();

    IncidencePhysicalOwners(
        ContactPhysicalResidency& residency, cartesian_detail::ArenaCredits* arena, Budget& budget, bool added) noexcept
        : residency_(residency), arena_(arena), budget_(budget), added_(added)
    {
    }

    template<class T>
    bool observe(Role role, const std::shared_ptr<T>& value, std::size_t bytes, const Budget* origin,
        ContactPhysicalResidency::PointScope scope = 0)
    {
        if (!value || !bytes)
            return false;
        if (!added_)
            return residency_.release(role, value, scope).retired();
        const auto prepaid = arena_ ? arena_->find_paid_token(value, origin, bytes) : nullptr;
        return residency_.retain(role, value, bytes, origin, prepaid ? &prepaid->charged_payload : nullptr, scope)
            .inserted();
    }

    template<class View> void observe_pool_credits(const std::shared_ptr<View>& view)
    {
        if (view)
            view->visit_credit_owners(
                [&](const auto& owner, std::size_t bytes)
                {
                    if constexpr (requires { owner->persistent_owner; })
                        if (owner->persistent_owner)
                            return; // Typed append-only pools already retain this exact control.
                    const auto scope = [&]
                    {
                        if constexpr (requires { owner->pool_scope; })
                            return owner->pool_scope;
                        else
                            return ContactPhysicalResidency::PointScope(0);
                    }();
                    observe(Role::CreditMetadata, owner, bytes, owner->budget, scope);
                });
    }

    template<class View> void observe_point_credits(const std::shared_ptr<View>& view)
    {
        if (view)
            view->visit_credit_owners(
                [&](const auto& credit, std::size_t bytes)
                { residency_.observe_point_credit(credit, bytes, added_); });
    }

    template<class Proof> void observe_proof(const Proof& proof)
    {
        observe_dependencies(
            [&](auto& observer, auto& unused, auto& scratch) { proof.retain_dependencies(observer, unused, scratch); });
    }

    void observe_snapshot(const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot)
    {
        if (!snapshot)
            return;
        observe_dependencies(
            [&](auto& observer, auto& unused, auto& scratch)
            {
                if (!observer.descend(snapshot, snapshot->allocation_budget(), snapshot->metadata_bytes(),
                        snapshot->has_arena_keeper(), unused, scratch))
                    return;
                snapshot->visit_dependencies<CartesianDependencyVisit::Observed>(
                    [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
                    {
                        using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
                        const std::shared_ptr<const std::remove_const_t<Element>> stable = owner;
                        return observer.descend(stable, origin, bytes, keeper, unused, scratch);
                    });
            });
    }

    static void bind_policy(const std::shared_ptr<cartesian_detail::ArenaCredits>& arena,
        const std::shared_ptr<ContactPhysicalResidency>& residency) noexcept
    {
        if (arena && residency)
            arena->set_prepaid_policy(residency,
                [](const void* policy, const std::shared_ptr<const void>& owner, const Budget* origin,
                    std::size_t bytes, bool metadata) noexcept
                {
                    const auto& editor = *static_cast<const ContactPhysicalResidency*>(policy);
                    return editor.admits(
                        metadata ? Role::CreditMetadata : Role::SourceDependency, owner, origin, bytes);
                });
    }

private:
    template<class Visit> void observe_dependencies(const Visit& visit)
    {
        cartesian_detail::CreditConstruction observer(budget_, this,
            [](void* context, const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes,
                bool keeper, bool metadata)
            {
                if (!origin || !keeper)
                    throw std::invalid_argument("Incidence proof dependency has no immutable arena lifetime.");
                return static_cast<IncidencePhysicalOwners*>(context)->observe(
                    metadata ? Role::CreditMetadata : Role::SourceDependency, owner, bytes, origin);
            });
        Credits unused;
        cartesian_detail::Allocation scratch(budget_);
        visit(observer, unused, scratch);
    }

    ContactPhysicalResidency& residency_;
    cartesian_detail::ArenaCredits* arena_;
    Budget& budget_;
    bool added_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
