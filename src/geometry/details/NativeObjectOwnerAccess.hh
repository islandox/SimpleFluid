/** Exact typed separate-control owners for unchanged public object layouts. */
#pragma once
#include "geometry/details/NativeConstructorOwnerSink.hh"
namespace SimpleFluid::geometry_detail
{
namespace native_object_detail
{
template<class T> struct Destroy
{
    NativeConstructorStorageOwner storage;
    void operator()(T* target) noexcept
    {
        std::destroy_at(target);
        // The real object and every directly owned concrete buffer died first.
        // External fixed/control observers must not prolong their byte charge.
        if (storage.owner)
            storage.owner->retire_all_freed_buffers();
        storage = {};
    }
};
} // namespace native_object_detail
template<class T> class NativeObjectOwnerAccess
{
public:
    using Destroy = native_object_detail::Destroy<T>;
    using Delete = native_envelope_detail::SeparateDelete<T, Destroy>;
    template<class... Args>
    [[nodiscard]] static NativeTrackedOwner<T> make_without_buffers(
        const NativeConstructionAdmission& admission, Args&&... args)
    {
        return NativeAllocationEnvelopeAccess::make_separate<T>(admission, Destroy{}, std::forward<Args>(args)...);
    }
    template<class Construct>
    [[nodiscard]] static NativeTrackedOwner<T> construct_without_buffers(
        const NativeConstructionAdmission& admission, Construct constructor)
    {
        return NativeAllocationEnvelopeAccess::make_separate_constructed<T>(
            admission, Destroy{}, std::move(constructor));
    }
    template<class... Args>
    [[nodiscard]] static NativeTrackedOwner<T> make(const NativeConstructionAdmission& admission, Args&&... args)
    {
        auto storage = native_constructor_storage(admission);
        return NativeAllocationEnvelopeAccess::make_separate<T>(
            admission, Destroy{std::move(storage)}, std::forward<Args>(args)...);
    }
    template<class Construct>
    [[nodiscard]] static NativeTrackedOwner<T> construct(
        const NativeConstructionAdmission& admission, Construct constructor)
    {
        auto storage = native_constructor_storage(admission);
        return NativeAllocationEnvelopeAccess::make_separate_constructed<T>(
            admission, Destroy{std::move(storage)}, std::move(constructor));
    }
    template<class U> [[nodiscard]] static const Delete* receipt(const std::shared_ptr<U>& owner)
    {
        if (!owner)
            return nullptr;
        const auto* value = std::get_deleter<Delete>(owner);
        if (!value)
            return nullptr;
        const auto& storage = value->destroy.storage;
        if (!value->envelope || !value->envelope->authentic_owner(value->envelope) ||
            !value->envelope->matches(owner) || bool(storage.owner) != bool(storage.envelope) ||
            (storage.owner &&
                (!storage.envelope->authentic_owner(storage.envelope) || !storage.envelope->matches(storage.owner) ||
                    storage.owner->admission().state() != value->envelope->state())))
            throw std::invalid_argument("Incomplete or foreign typed native object/control receipt.");
        return value;
    }
    template<class U> [[nodiscard]] static bool admitted(const std::shared_ptr<U>& owner)
    {
        return receipt(owner) != nullptr;
    }
    template<class U> [[nodiscard]] static const NativeConstructorStorageOwner& storage(const std::shared_ptr<U>& owner)
    {
        const auto* value = receipt(owner);
        if (!value)
            throw std::invalid_argument("Object lacks its exact typed native allocation receipt.");
        return value->destroy.storage;
    }
    template<class U>
    [[nodiscard]] static const std::shared_ptr<const NativeAllocationEnvelope>& envelope(
        const std::shared_ptr<U>& owner)
    {
        const auto* value = receipt(owner);
        if (!value)
            throw std::invalid_argument("Object lacks its exact typed native allocation envelope.");
        return value->envelope;
    }
    template<class U>
    static bool visit_children(
        const std::shared_ptr<U>& owner, const NativeConstructorOwnerSink& sink, bool branch = false)
    {
        const auto* value = receipt(owner);
        if (!value)
            return false;
        sink.fixed(value->envelope);
        sink.storage(value->destroy.storage, branch);
        return true;
    }
};
} // namespace SimpleFluid::geometry_detail
