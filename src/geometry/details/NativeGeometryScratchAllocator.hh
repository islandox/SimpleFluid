#pragma once
#include "geometry/details/NativeConstructionAdmission.hh"

namespace SimpleFluid::geometry_detail
{
template<class T> class SIMPLEFLUID_LOCAL NativeGeometryScratchAllocator : public NativeAdministrationAllocator<T>
{
    using Base = NativeAdministrationAllocator<T>;

public:
    using Base::Base;
    template<class U>
    NativeGeometryScratchAllocator(const NativeGeometryScratchAllocator<U>& other) noexcept : Base(other)
    {
    }
    [[nodiscard]] T* allocate(std::size_t count)
    {
        return this->allocate_for(AllocationDomain::GeometrySnapshot, count);
    }
    void deallocate(T* address, std::size_t count) noexcept
    {
        this->deallocate_for(AllocationDomain::GeometrySnapshot, address, count);
    }
};
template<class T> using NativeGeometryScratchVector = std::vector<T, NativeGeometryScratchAllocator<T>>;
} // namespace SimpleFluid::geometry_detail
