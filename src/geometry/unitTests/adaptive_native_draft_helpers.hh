/** @file adaptive_native_draft_helpers.hh
 * @brief Exact comparisons of expanded and publication-only native drafts.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include <gtest/gtest.h>

namespace SimpleFluid::test
{
inline void compare_native_drafts(
    const Meshes::AMR::AdaptiveNativeDraft& expanded, const Meshes::AMR::AdaptiveNativeDraft& compact)
{
    namespace AMR = Meshes::AMR;
    EXPECT_EQ(expanded.global_cells, compact.global_cells);
    EXPECT_EQ(expanded.global_faces, compact.global_faces);
    EXPECT_EQ(expanded.global_nodes, compact.global_nodes);
    EXPECT_EQ(expanded.family_references, compact.family_references);
    auto leaves = [](const auto& full, const auto& retained)
    {
        ASSERT_EQ(full.size(), retained.size());
        for (std::size_t i = 0; i < full.size(); ++i)
        {
            ASSERT_TRUE(full[i].geometry);
            const auto& a = full[i];
            const auto& b = retained[i];
            EXPECT_EQ(a.key, b.key);
            EXPECT_EQ(a.domain, b.domain);
            EXPECT_EQ(a.material, b.material);
            EXPECT_EQ(a.revision, b.revision);
            EXPECT_EQ(a.provider, b.provider);
            EXPECT_EQ(a.parent_family, b.parent_family);
            if (b.geometry)
                EXPECT_TRUE(b.geometry->equivalent(*a.geometry));
            ASSERT_EQ(a.surfaces.size(), b.surfaces.size());
            for (std::size_t f = 0; f < a.surfaces.size(); ++f)
            {
                const auto& x = a.surfaces[f];
                const auto& y = b.surfaces[f];
                EXPECT_EQ(x.key, y.key);
                ASSERT_TRUE(x.geometry && y.geometry);
                EXPECT_TRUE(x.geometry->equivalent(*y.geometry));
                EXPECT_FALSE(x.geometry->oppositely_oriented(*y.geometry));
                EXPECT_EQ(x.orientation, y.orientation);
                EXPECT_EQ(x.exterior, y.exterior);
                EXPECT_EQ(x.boundary, y.boundary);
                EXPECT_EQ(x.authority_image, y.authority_image);
                EXPECT_EQ(x.require_native_whole_tile, y.require_native_whole_tile);
            }
        }
    };
    leaves(expanded.owned, compact.owned);
    leaves(expanded.ghosts, compact.ghosts);
    ASSERT_EQ(expanded.tiles.size(), compact.tiles.size());
    for (std::size_t i = 0; i < expanded.tiles.size(); ++i)
        EXPECT_EQ(AMR::encode_adaptive_tile(expanded.tiles[i]), AMR::encode_adaptive_tile(compact.tiles[i]));
    ASSERT_EQ(expanded.nodes.size(), compact.nodes.size());
    for (std::size_t i = 0; i < expanded.nodes.size(); ++i)
    {
        EXPECT_EQ(expanded.nodes[i].gid, compact.nodes[i].gid);
        EXPECT_EQ(expanded.nodes[i].identity, compact.nodes[i].identity);
        EXPECT_EQ(expanded.nodes[i].coordinate_identity, compact.nodes[i].coordinate_identity);
        EXPECT_EQ(expanded.nodes[i].coordinate, compact.nodes[i].coordinate);
    }
    ASSERT_EQ(expanded.cell_bindings.size(), compact.cell_bindings.size());
    for (std::size_t i = 0; i < expanded.cell_bindings.size(); ++i)
    {
        const auto& a = expanded.cell_bindings[i];
        const auto& b = compact.cell_bindings[i];
        EXPECT_EQ(a.key, b.key);
        EXPECT_EQ(a.gid, b.gid);
        EXPECT_EQ(a.material, b.material);
        EXPECT_EQ(a.owner, b.owner);
        EXPECT_EQ(a.provider, b.provider);
        EXPECT_EQ(a.full, b.full);
        EXPECT_DOUBLE_EQ(a.volume, b.volume);
        EXPECT_EQ(a.centroid, b.centroid);
    }
}
} // namespace SimpleFluid::test
