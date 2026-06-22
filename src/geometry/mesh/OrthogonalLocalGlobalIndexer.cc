/**
 * @file OrthogonalLocalGlobalIndexer.cc
 * @brief Instantiates the arithmetic orthogonal LocalGlobalIndexer.
 *
 * @details Builds the native orthogonal ordinal types and the ordinal types
 * used by DefaultTpetraTypes into SimpleFluidMesh. Other ordinal combinations
 * can be instantiated from OrthogonalLocalGlobalIndexer.tcc.
 */

#include "geometry/mesh/OrthogonalLocalGlobalIndexer.tcc"

namespace SimpleFluid::Meshes
{

/** @brief Native orthogonal-mesh ordinal types. */
template class LocalGlobalIndexer<OrthogonalMeshIndexTypePack<size_t, uint64_t>>;

/** @brief Orthogonal indexer rebound to the default Tpetra ordinal types. */
template class LocalGlobalIndexer<OrthogonalMeshIndexTypePack<int, long long>>;

} // namespace SimpleFluid::Meshes
