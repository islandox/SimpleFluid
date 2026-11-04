#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <cstddef>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Inline records need no additional admission beyond their page capacity.
 * Dynamically allocated records select a typed policy below or explicitly at
 * the pool instantiation. Shared plane payload belongs to IdentityBudget. */
struct NoIdentityPoolPayload
{
    template<class T> std::size_t operator()(const T&) const noexcept { return 0; }
};
template<class T> struct IdentityPoolPayload : NoIdentityPoolPayload
{
};

template<> struct IdentityPoolPayload<AdaptiveCellKey>
{
    std::size_t operator()(const AdaptiveCellKey& key) const { return key_capacity(key); }
};
template<> struct IdentityPoolPayload<AdaptiveBytes>
{
    std::size_t operator()(const AdaptiveBytes& bytes) const noexcept { return bytes.capacity(); }
};
template<> struct IdentityPoolPayload<PLIC::PointIdentity<ArithmeticMode::Ordinary>>
{
    std::size_t operator()(const PLIC::PointIdentity<ArithmeticMode::Ordinary>& point) const noexcept
    {
        return point.payload_bytes();
    }
};
template<> struct IdentityPoolPayload<PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>> : NoIdentityPoolPayload
{
};
template<> struct IdentityPoolPayload<PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>>
{
    std::size_t operator()(const PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>& image) const noexcept
    {
        return image.payload_bytes();
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
