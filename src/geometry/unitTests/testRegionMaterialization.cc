#include <gtest/gtest.h>
#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <cstdlib>
#include <new>

namespace
{
thread_local bool count_allocations = false;
thread_local size_t allocated_bytes = 0;
}

// Count requested allocation volume only during explicit CSR construction.
// This catches repeated prefix copies without a machine-dependent timing gate.
void* operator new(std::size_t size)
{
    if (void* pointer = std::malloc(size ? size : 1))
    {
        if (count_allocations) allocated_bytes += size;
        return pointer;
    }
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace
{
using namespace SimpleFluid;
using Handle = MeshHandle<>;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void verify_materialization(const SP<Meshes::MultiRegionMesh>& geometry,
                            bool require_expanded_face)
{
    Handle mesh(geometry);
    const auto owned_cells = mesh.owned_cell_map();
    const auto overlap_cells = mesh.overlap_cell_map();
    const auto owned_faces = mesh.owned_face_map();
    const auto overlap_faces = mesh.overlap_face_map();
    const bool materialized_indexer = mesh.has_materialized_indexer();
    std::vector<std::vector<Handle::local_ordinal_type>> expected;
    bool expanded = false;
    for (size_t c = 0; c < mesh.num_local_cells(); ++c)
    {
        const auto faces = mesh.faces(c);
        expected.emplace_back(faces.begin(), faces.end());
        expanded = expanded || faces.size() > 6;
    }
    if (require_expanded_face && owned_cells->getComm()->getSize() == 1)
        ASSERT_TRUE(expanded);
    EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
    for (int repetition = 0; repetition < 2; ++repetition)
    {
        mesh.materialize_cell_faces();
        EXPECT_TRUE(mesh.has_materialized_connectivity());
        EXPECT_EQ(mesh.owned_cell_map(), owned_cells);
        EXPECT_EQ(mesh.overlap_cell_map(), overlap_cells);
        EXPECT_EQ(mesh.owned_face_map(), owned_faces);
        EXPECT_EQ(mesh.overlap_face_map(), overlap_faces);
        EXPECT_EQ(mesh.has_materialized_indexer(), materialized_indexer);
        for (size_t c = 0; c < mesh.num_local_cells(); ++c)
        {
            EXPECT_TRUE(std::ranges::equal(mesh.materialized_faces(c), expected[c]));
            EXPECT_TRUE(std::ranges::equal(mesh.faces(c), expected[c]));
            for (auto face : mesh.materialized_faces(c))
                EXPECT_LT(static_cast<size_t>(face), mesh.num_faces());
        }
    }
}
}

TEST(RegionMaterializationTest, PreservesCanonicalOrderingAndMaps)
{
    verify_materialization(test::two_regions(3), false);
    verify_materialization(test::periodic_regions(), false);
}

TEST(RegionMaterializationTest, RetainsEveryExpandedCoarseFineLogicalFace)
{
    verify_materialization(test::coarse_fine_regions(), true);
}

TEST(RegionMaterializationTest, AllocationVolumeGrowsWithFinalConnectivity)
{
    Handle mesh(test::two_regions(10));
    allocated_bytes = 0;
    count_allocations = true;
    try { mesh.materialize_cell_faces(); }
    catch (...) { count_allocations = false; throw; }
    count_allocations = false;
    const auto requested = allocated_bytes;
    // Acquiring leases may allocate a few small records. The generous linear
    // bound permits different vector growth factors and excludes quadratic
    // reserve(size + next_cell_faces) behavior on this 2,000-cell fixture.
    EXPECT_LT(requested, 8 * mesh.connectivity_storage_bytes() + 65536);
    EXPECT_GE(requested, mesh.connectivity_storage_bytes());
    EXPECT_GT(mesh.connectivity_storage_bytes(), 0U);
}
