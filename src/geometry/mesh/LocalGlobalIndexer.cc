/**
 * @file LocalGlobalIndexer.cc
 * @brief Explicit template instantiations for LocalGlobalIndexer.
 *
 * @details Builds the common integral-ID indexers into SimpleFluidMesh so
 * callers using those packs do not need to instantiate the out-of-line
 * definitions from `LocalGlobalIndexer.tcc` themselves.
 *
 * The compiled specializations cover the library default index pack and the
 * local/global ordinal combinations used by UnstructuredMesh.
 */

#include "geometry/mesh/LocalGlobalIndexer.tcc"

#include "dataclass/typedefs.hh"

namespace SimpleFluid::Meshes
{

/** @brief Default integral-ID indexer used by MeshHandle. */
template class LocalGlobalIndexer<
    MeshIndexTypes<global_index_t, global_index_t, global_index_t,
                   local_index_t, global_index_t>>;

/**
 * @brief UnstructuredMesh indexer using the default Tpetra ordinal types.
 */
template class LocalGlobalIndexer<
    MeshIndexTypes<uint64_t, uint64_t, uint64_t, int, long long>>;

/** @brief UnstructuredMesh indexer using its native ordinal defaults. */
template class LocalGlobalIndexer<
    MeshIndexTypes<uint64_t, uint64_t, uint64_t, size_t, uint64_t>>;

} // namespace SimpleFluid::Meshes
