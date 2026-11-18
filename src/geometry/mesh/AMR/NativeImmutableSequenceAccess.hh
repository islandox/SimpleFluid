/** Admitted immutable vector controls without adding fields to every sequence. */
#pragma once
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/mesh/AMR/ImmutableSequence.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeImmutableSequenceAccess
{
public:
    template<class T>
    [[nodiscard]] static ImmutableSequence<T> snapshot(
        std::span<const T> input, const geometry_detail::NativeConstructionAdmission& admission)
    {
        ImmutableSequence<T> result;
        result.count_ = ImmutableSequence<T>::narrow_size(input.size());
        if (!input.empty())
        {
            auto owner = geometry_detail::NativeObjectOwnerAccess<std::vector<T>>::make(admission);
            const auto& storage = geometry_detail::NativeObjectOwnerAccess<std::vector<T>>::storage(owner.owner);
            storage.owner->copy(*owner.owner, input);
            result.storage_ = std::move(owner.owner);
        }
        return result;
    }
    template<class T>
    static bool visit(const ImmutableSequence<T>& sequence, const geometry_detail::NativeConstructorOwnerSink& sink)
    {
        const auto& owner = sequence.storage_;
        if (!owner)
            return true;
        using Access = geometry_detail::NativeObjectOwnerAccess<std::vector<T>>;
        if (!Access::admitted(owner))
            return false;
        if (sink.semantic(owner, geometry_detail::NativeConstructorOwnerKind::Storage, Access::envelope(owner)))
            Access::visit_children(owner, sink);
        return true;
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
