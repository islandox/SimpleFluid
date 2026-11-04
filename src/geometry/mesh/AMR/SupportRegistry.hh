#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Compact logical membership. Geometry is fetched only for dirty supports. */
struct RegistryMember
{
    AdaptiveCellKey cell;
    std::uint64_t revision = 0, surface = 0, encoded_size = 0;
    int owner = -1;
};
struct RegistryChange
{
    AdaptiveBytes support;
    RegistryMember member;
    bool remove = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
