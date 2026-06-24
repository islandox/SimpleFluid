/**
 * @file testOrthogonalLocalGlobalIndexerHeader.cc
 * @brief Verifies the public orthogonal local/global indexer header links.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthogonalLocalGlobalIndexer.hh"

namespace
{

using OrthogonalIndexer = SimpleFluid::Meshes::OrthogonalIndexer;
using IndexTypes =
    SimpleFluid::Meshes::OrthogonalMeshIndexTypePack<int, long long>;
using LocalGlobalIndexer =
    SimpleFluid::Meshes::LocalGlobalIndexer<IndexTypes>;

static_assert(SimpleFluid::MeshIndexer<LocalGlobalIndexer>);

TEST(OrthogonalLocalGlobalIndexerHeaderTest, PublicHeaderLinksPrebuiltSpecialization)
{
    const OrthogonalIndexer global_indexer(4, 3, 2);
    const LocalGlobalIndexer indexer(
        global_indexer,
        LocalGlobalIndexer::BlockShape{2, 1, 1},
        LocalGlobalIndexer::BlockShape{1, 0, 0});

    EXPECT_EQ(indexer.num_owned_cells(), 12U);
    EXPECT_EQ(
        indexer.local_to_global_cell_id({0, 0, 0}),
        (OrthogonalIndexer::CellID{2, 0, 0}));
    EXPECT_EQ(
        indexer.global_to_local_cell_id({3, 2, 1}),
        (OrthogonalIndexer::CellID{1, 2, 1}));
}

} // namespace
