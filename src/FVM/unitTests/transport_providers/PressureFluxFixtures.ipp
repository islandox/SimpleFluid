/** @file PressureFluxFixtures.ipp
 * @brief Inline custom boundary-cache queries and caller-local execution acquisition.
 */
#pragma once

namespace SimpleFluid::test::pressure_flux
{
template<class Map>
inline CountingBoundaryMap<Map>::CountingBoundaryMap(const Map& entries) : Map(entries) {}

template<class Map>
inline auto CountingBoundaryMap<Map>::find(const typename Map::key_type& key) const
{
    ++lookups;
    return Map::find(key);
}

inline CountingVelocityBoundaryCache::CountingVelocityBoundaryCache(const Cache& cache)
    : value(cache.value), type(cache.type), type_by_name(cache.type_by_name), mesh(cache.mesh)
{
}

inline auto FailingWorkspace::acquire_execution() const
{
    if (reject_acquisition)
        throw std::logic_error("Injected pressure execution acquisition failure.");
    return Workspace::acquire_execution();
}
} // namespace SimpleFluid::test::pressure_flux
