/** @file IncidencePhysicalOwners.hh
 * @brief Exact owner events for incidence and detached source proofs.
 */
#pragma once
#include "CartesianAuthorityCredits.hh"
#include "ContactPhysicalResidency.hh"
#include "ContactSourcePoints.hh"
#include <limits>
#include <memory>
#include <optional>
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
        ContactPhysicalResidency::PointScope scope = 0, const NativeContactOwnerFunding* funding = nullptr)
    {
        if (!value)
            return false;
        if constexpr (std::is_same_v<std::remove_cv_t<T>, cartesian_detail::AuthorityOwnerCredit>)
            if (!scope && role == Role::CreditMetadata)
                scope = value->arena_scope;
        if (!added_)
        {
            const auto active = residency_.usage().find({value.get(), std::uint32_t(role)});
            const bool hinted = arena_ && active && active->references == 1 && arena_->hint_retirement(value);
            try
            {
                return residency_.release(role, value, scope, funding).retired();
            }
            catch (...)
            {
                if (hinted)
                    arena_->cancel_retirement(value);
                throw;
            }
        }
        const auto prepaid = arena_ ? arena_->find_paid_token(value, origin, bytes) : nullptr;
        const auto transition = residency_.retain(
            role, value, bytes, origin, prepaid ? &prepaid->charged_payload : nullptr, scope, funding);
        if (arena_ && transition.inserted())
            arena_->cancel_retirement(value);
        return transition.inserted();
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
                [&](const auto& credit, std::size_t bytes) { residency_.observe_point_credit(credit, bytes, added_); });
    }

    template<class Proof> void observe_proof(const Proof& proof)
    {
        observe_dependencies(
            [&](auto& observer, auto& unused, auto& scratch) { proof.retain_dependencies(observer, unused, scratch); });
    }

    template<class Source> void observe_source(const std::shared_ptr<const Source>& source)
    {
        if (!source)
            return;
        observe_dependencies(
            [&](auto& observer, auto& unused, auto& scratch)
            {
                if (!observer.descend(source, source->allocation_budget(), source->metadata_bytes(),
                        source->has_arena_keeper(), unused, scratch))
                    return;
                const auto visit = [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
                {
                    using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
                    const std::shared_ptr<const std::remove_const_t<Element>> stable = owner;
                    return observer.descend(stable, origin, bytes, keeper, unused, scratch);
                };
                if constexpr (requires { source->visit_observed_dependencies(visit); })
                    source->visit_observed_dependencies(visit);
                else if constexpr (requires {
                                       source->template visit_dependencies<CartesianDependencyVisit::Observed>(visit);
                                   })
                    source->template visit_dependencies<CartesianDependencyVisit::Observed>(visit);
                else if constexpr (requires { source->visit_dependencies(visit); })
                    source->visit_dependencies(visit);
            });
    }
    void observe_snapshot(const std::shared_ptr<const ContactSourcePointSnapshot>& source) { observe_source(source); }

    static void bind_policy(const std::shared_ptr<cartesian_detail::ArenaCredits>& arena,
        const std::shared_ptr<ContactPhysicalResidency>& residency) noexcept
    {
        if (!arena || !residency)
            return;
        residency->begin_source_credit_scope(arena->scope_id);
        arena->set_prepaid_policy(residency,
            [](const void* policy, const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes,
                bool metadata) noexcept
            {
                const auto& editor = *static_cast<const ContactPhysicalResidency*>(policy);
                return editor.admits(metadata ? Role::CreditMetadata : Role::SourceDependency, owner, origin, bytes);
            });
    }

private:
    template<class Visit> void observe_dependencies(const Visit& visit)
    {
        cartesian_detail::CreditConstruction observer(
            budget_, this,
            [](void* context, const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes,
                bool keeper, bool metadata, std::uint64_t scope)
            {
                if (!origin || !keeper)
                    throw std::invalid_argument("Incidence proof dependency has no immutable arena lifetime.");
                auto& self = *static_cast<IncidencePhysicalOwners*>(context);
                const auto result = self.observe(metadata ? Role::CreditMetadata : Role::SourceDependency, owner, bytes,
                    origin, scope, self.pending_funding_ ? &*self.pending_funding_ : nullptr);
                self.pending_funding_.reset();
                return result;
            },
            [](void* raw, const NativeContactOwnerFunding& funding)
            { static_cast<IncidencePhysicalOwners*>(raw)->pending_funding_ = funding; });
        Credits unused;
        cartesian_detail::Allocation scratch(budget_);
        visit(observer, unused, scratch);
    }

    ContactPhysicalResidency& residency_;
    cartesian_detail::ArenaCredits* arena_;
    Budget& budget_;
    bool added_;
    std::optional<NativeContactOwnerFunding> pending_funding_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
