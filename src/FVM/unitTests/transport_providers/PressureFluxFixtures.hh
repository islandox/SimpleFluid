/** @file PressureFluxFixtures.hh
 * @brief Custom pressure-flux fixtures shared with their private kernel provider.
 */
#pragma once

#include "FVM/FaceFlux.hh"

namespace SimpleFluid::test::pressure_flux
{
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
using Workspace = FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack>;

template<class Map>
struct CountingBoundaryMap : Map
{
    explicit CountingBoundaryMap(const Map& entries);

    auto find(const typename Map::key_type& key) const;

    mutable size_t lookups = 0;
};

struct CountingVelocityBoundaryCache
{
    using Cache = SimpleFluid::FVM::FieldStoredVelocityBoundaryCache<Pack, Mesh>;

    explicit CountingVelocityBoundaryCache(const Cache& cache);

    CountingBoundaryMap<decltype(Cache::value)> value;
    CountingBoundaryMap<decltype(Cache::type)> type;
    decltype(Cache::type_by_name) type_by_name;
    SimpleFluid::SP<const Mesh> mesh;
};

struct FailingWorkspace : Workspace
{
    using Workspace::Workspace;
    bool reject_acquisition = false;

    auto acquire_execution() const;
};
} // namespace SimpleFluid::test::pressure_flux

#include "PressureFluxFixtures.ipp"
