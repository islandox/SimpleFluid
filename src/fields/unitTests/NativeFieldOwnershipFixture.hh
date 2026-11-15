/** @file NativeFieldOwnershipFixture.hh
 * @brief Real two-cell geometry for owned and overlap face-field checks.
 */
#pragma once

#include "geometry/MeshHandle.hh"

#include <memory>

namespace SimpleFluid::test
{

/** @brief Two adjacent unit cubes: eleven faces, ten physical boundary faces. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
inline SP<MeshHandle<Pack>> native_field_ownership_mesh()
{
    using Mesh = MeshHandle<Pack>;
    return std::make_shared<Mesh>(std::make_shared<typename Mesh::Cartesian>(
        Vec3D<ArrReal>{{{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}}),
        typename Mesh::DistributionOptions{.allow_empty_partitions = true});
}

} // namespace SimpleFluid::test
