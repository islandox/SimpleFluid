/** Fixed allocation lifetime retained by an exact producer-issued weak observation. */
#pragma once
#include "geometry/details/NativeAllocationEnvelope.hh"
#include <type_traits>
#include <variant>

namespace SimpleFluid::Meshes::AMR
{
class DistributedAdaptiveHierarchy;
namespace adaptive_detail
{
class NativeContactFundingAccess;
class NativePublicationSession;
class NativeSourceOwnerStore;
}
}

namespace SimpleFluid::PLIC::details
{
class NativeWeakOwnerObservation
{
public:
    struct ContactGroup
    {
        std::shared_ptr<const void> owner, keeper;
        std::size_t bytes = 0, keeper_bytes = 0;
    };
    struct DetachedScalar
    {
        std::shared_ptr<const void> owner, keeper;
        std::size_t bytes, keeper_bytes;
    };

    NativeWeakOwnerObservation() noexcept = default;
    NativeWeakOwnerObservation(const NativeWeakOwnerObservation& other) noexcept : fixed_(copy(other.fixed_)) {}
    NativeWeakOwnerObservation& operator=(const NativeWeakOwnerObservation& other) noexcept
    {
        fixed_ = copy(other.fixed_);
        return *this;
    }
    NativeWeakOwnerObservation(NativeWeakOwnerObservation&&) noexcept = default;
    NativeWeakOwnerObservation& operator=(NativeWeakOwnerObservation&&) noexcept = default;
    [[nodiscard]] explicit operator bool() const noexcept
    { return native_envelope() || native_state() || detached_scalar() || contact_control_group(); }
    [[nodiscard]] bool matches_live(const std::shared_ptr<const void>& owner) const noexcept
    {
        if (const auto* native = std::get_if<NativeObservation>(&fixed_))
            return native->envelope && owner.get() == native->address &&
                (native->envelope_owner ? same_owner(native->envelope, owner) : native->envelope->shares_control(owner));
        if (const auto* state = native_state()) return same_owner(*state, owner);
        if (const auto* scalar = detached_scalar()) return same_owner(scalar->owner, owner);
        if (const auto* contact = std::get_if<ContactObservation>(&fixed_))
            return owner && contact->address == owner.get() && !contact->target.owner_before(owner) &&
                   !owner.owner_before(contact->target);
        return false;
    }
    [[nodiscard]] const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>* native_envelope()
        const noexcept
    {
        const auto* native = std::get_if<NativeObservation>(&fixed_);
        return native && native->envelope ? &native->envelope : nullptr;
    }
    [[nodiscard]] const ContactGroup* contact_control_group() const noexcept
    {
        const auto* contact = std::get_if<ContactObservation>(&fixed_);
        return contact && contact->group.owner ? &contact->group : nullptr;
    }
    [[nodiscard]] const std::shared_ptr<const geometry_detail::NativeAllocationState>* native_state() const noexcept
    {
        const auto* state = std::get_if<StateObservation>(&fixed_);
        return state && state->owner ? &state->owner : nullptr;
    }
    [[nodiscard]] const DetachedScalar* detached_scalar() const noexcept
    {
        const auto* scalar = std::get_if<DetachedScalar>(&fixed_);
        return scalar && scalar->owner ? scalar : nullptr;
    }
    [[nodiscard]] bool same_observation(const NativeWeakOwnerObservation& other) const noexcept
    {
        if (fixed_.index() != other.fixed_.index())
            return false;
        if (const auto* native = std::get_if<NativeObservation>(&fixed_))
        {
            const auto& right = std::get<NativeObservation>(other.fixed_);
            return native->address == right.address && native->envelope_owner == right.envelope_owner &&
                same_owner(native->envelope, right.envelope);
        }
        if (const auto* state = std::get_if<StateObservation>(&fixed_))
            return same_owner(state->owner, std::get<StateObservation>(other.fixed_).owner);
        if (const auto* scalar = std::get_if<DetachedScalar>(&fixed_))
        {
            const auto& right = std::get<DetachedScalar>(other.fixed_);
            return scalar->bytes == right.bytes && scalar->keeper_bytes == right.keeper_bytes &&
                same_owner(scalar->owner, right.owner) && same_owner(scalar->keeper, right.keeper);
        }
        if (const auto* contact = std::get_if<ContactObservation>(&fixed_))
        {
            const auto& right = std::get<ContactObservation>(other.fixed_);
            return contact->address == right.address && !contact->target.owner_before(right.target) &&
                   !right.target.owner_before(contact->target) && same_owner(contact->group.owner, right.group.owner) &&
                   same_owner(contact->group.keeper, right.group.keeper) && contact->group.bytes == right.group.bytes &&
                   contact->group.keeper_bytes == right.group.keeper_bytes;
        }
        return true;
    }

private:
    friend class NativePaidExtentCharge;
    friend class Meshes::AMR::DistributedAdaptiveHierarchy;
    friend class Meshes::AMR::adaptive_detail::NativeContactFundingAccess;
    friend class Meshes::AMR::adaptive_detail::NativePublicationSession;
    friend class Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    struct NativeObservation
    {
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> envelope;
        const void* address;
        bool envelope_owner = false;
    };
    struct StateObservation
    {
        std::shared_ptr<const geometry_detail::NativeAllocationState> owner;
    };
    struct ContactObservation
    {
        // Kept separate from the control's actual deleter group to avoid a
        // strong-group/weak-target cycle.
        std::weak_ptr<const void> target;
        const void* address;
        ContactGroup group;
    };
    using Fixed = std::variant<std::monostate, NativeObservation, ContactObservation, StateObservation, DetachedScalar>;
    static_assert(std::is_nothrow_copy_constructible_v<NativeObservation>);
    static_assert(std::is_nothrow_copy_constructible_v<ContactObservation>);
    static_assert(std::is_nothrow_copy_constructible_v<StateObservation>);
    static_assert(std::is_nothrow_copy_constructible_v<DetachedScalar>);
    static_assert(std::is_nothrow_move_assignable_v<Fixed>);
    Fixed fixed_;

    template<class A, class B>
    static bool same_owner(const std::shared_ptr<A>& a, const std::shared_ptr<B>& b) noexcept
    { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); }

    // All alternatives consist of scalars and shared/weak controls. Constructing
    // the selected alternative directly preserves that guarantee under libstdc++.
    static Fixed copy(const Fixed& value) noexcept
    {
        if (const auto* native = std::get_if<NativeObservation>(&value))
            return Fixed(std::in_place_type<NativeObservation>, *native);
        if (const auto* contact = std::get_if<ContactObservation>(&value))
            return Fixed(std::in_place_type<ContactObservation>, *contact);
        if (const auto* state = std::get_if<StateObservation>(&value))
            return Fixed(std::in_place_type<StateObservation>, *state);
        if (const auto* scalar = std::get_if<DetachedScalar>(&value))
            return Fixed(std::in_place_type<DetachedScalar>, *scalar);
        return {};
    }
    [[nodiscard]] static NativeWeakOwnerObservation envelope_control(
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& value)
    {
        if (!value || !value->authentic_owner(value))
            throw std::invalid_argument("Native fixed observation lacks its exact envelope control.");
        NativeWeakOwnerObservation result;
        result.fixed_ = NativeObservation{value, value.get(), true};
        return result;
    }
    [[nodiscard]] static NativeWeakOwnerObservation state_control(
        const std::shared_ptr<const geometry_detail::NativeAllocationState>& state)
    {
        NativeWeakOwnerObservation result;
        result.fixed_ = StateObservation{state};
        return result;
    }
    [[nodiscard]] static NativeWeakOwnerObservation scalar_control(
        const std::shared_ptr<const void>& owner, std::size_t bytes,
        const std::shared_ptr<const void>& keeper, std::size_t keeper_bytes)
    {
        NativeWeakOwnerObservation result;
        result.fixed_ = DetachedScalar{owner, keeper, bytes, keeper_bytes};
        return result;
    }

    [[nodiscard]] static NativeWeakOwnerObservation native(const std::shared_ptr<const void>& exact_target,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& exact_envelope)
    {
        if (!exact_envelope || !exact_envelope->matches(exact_target))
            throw std::invalid_argument("Native weak observation has a foreign target or allocation envelope.");
        return native_alias(exact_target, exact_envelope);
    }
    [[nodiscard]] static NativeWeakOwnerObservation native_alias(const std::shared_ptr<const void>& verified_alias,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& exact_envelope)
    {
        if (!exact_envelope || !exact_envelope->authentic_owner(exact_envelope) ||
            !exact_envelope->shares_control(verified_alias))
            throw std::invalid_argument("Native weak observation has a foreign alias control or allocation envelope.");
        NativeWeakOwnerObservation result;
        result.fixed_ = NativeObservation{exact_envelope, verified_alias.get()};
        return result;
    }
    [[nodiscard]] static NativeWeakOwnerObservation contact_group(const std::shared_ptr<const void>& exact_target,
        const std::shared_ptr<const void>& named_deleter_group, std::size_t group_upper_bytes,
        const std::shared_ptr<const void>& exact_budget_keeper, std::size_t budget_keeper_upper_bytes)
    {
        const std::shared_ptr<const void> empty;
        const auto actual = [&](const auto& owner) { return owner && (owner.owner_before(empty) || empty.owner_before(owner)); };
        if (!actual(exact_target) || !actual(named_deleter_group) || !actual(exact_budget_keeper) ||
            !group_upper_bytes || !budget_keeper_upper_bytes)
            throw std::invalid_argument("Contact weak observation lacks its actual target, control group or keeper.");
        NativeWeakOwnerObservation result;
        result.fixed_ = ContactObservation{exact_target, exact_target.get(),
            {named_deleter_group, exact_budget_keeper, group_upper_bytes, budget_keeper_upper_bytes}};
        return result;
    }
};
static_assert(std::is_nothrow_copy_constructible_v<NativeWeakOwnerObservation>);
static_assert(std::is_nothrow_copy_assignable_v<NativeWeakOwnerObservation>);
} // namespace SimpleFluid::PLIC::details
