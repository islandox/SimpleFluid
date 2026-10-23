/** @file testDecomposedSemiStructuredTopology.cc
 * @brief Compact XY/Z topology roundtrips, exact halo bands and storage-scaling regressions.
 */
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/DecomposedSemiStructuredTopology.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Semi = Handle::SemiStructured;
using Topology = SimpleFluid::Meshes::DecomposedSemiStructuredTopology;
using LO = Pack::local_ordinal_type;

testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Semi> chain(unsigned layers)
{
    constexpr unsigned cells = 24;
    SimpleFluid::Arr<Semi::Vec3> nodes;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x <= cells; ++x)
            nodes.push_back({static_cast<double>(x), static_cast<double>(y), 0.});
    SimpleFluid::Arr<SimpleFluid::Arr<unsigned>> loops;
    for (unsigned parity = 0; parity < 2; ++parity)
        for (unsigned x = parity; x < cells; x += 2)
            loops.push_back({x, x + 1, cells + 2 + x, cells + 1 + x});
    SimpleFluid::ArrReal z;
    for (unsigned k = 0; k <= layers; ++k)
        z.push_back(static_cast<double>(k) + .001 * k * k);
    return std::make_shared<Semi>(nodes, loops, z);
}

std::shared_ptr<Semi> triangle()
{
    return std::make_shared<Semi>(SimpleFluid::Arr<Semi::Vec3>{{0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}}, SimpleFluid::ArrReal{0., 1.});
}

template<class T> T global_sum(T local)
{
    T global{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

Handle::DistributionOptions hybrid_options(std::size_t depth)
{
    const auto ranks = static_cast<std::size_t>(Tpetra::getDefaultComm()->getSize());
    return {.ghost_layers = depth,
        .allow_empty_partitions = true,
        .xy_partitions = ranks % 2 == 0 ? 2U : 1U,
        .z_partitions = ranks % 2 == 0 ? ranks / 2 : ranks};
}

std::set<std::size_t> owned_cells(const Handle& handle)
{
    std::set<std::size_t> result;
    for (std::size_t row = 0; row < handle.num_owned_cells(); ++row)
        result.insert(static_cast<std::size_t>(handle.cell_geometry_global_id(static_cast<LO>(row))));
    return result;
}

std::set<std::size_t> breadth_first_cells(const Semi& source, std::set<std::size_t> owned, std::size_t depth)
{
    auto result = owned;
    auto frontier = std::move(owned);
    for (std::size_t layer = 0; layer < depth; ++layer)
    {
        std::set<std::size_t> next;
        for (const auto id : frontier)
            for (const auto face : source.faces(source.indexer().cell_id(id)))
            {
                const auto neighbor = source.opposite_cell(face, source.indexer().cell_id(id));
                if (neighbor != Semi::invalid_cell_id())
                {
                    const auto global = source.indexer().cell_ordinal(neighbor);
                    if (!result.contains(global))
                        next.insert(global);
                }
            }
        result.insert(next.begin(), next.end());
        frontier = std::move(next);
    }
    return result;
}

void expect_bands_cover(const auto& bands, std::size_t count)
{
    std::size_t offset = 0;
    for (const auto& band : bands)
    {
        EXPECT_EQ(band.offset, offset);
        EXPECT_GE(band.z_end, band.z_begin);
        EXPECT_EQ(band.count(), (band.z_end - band.z_begin) * band.xy_size);
        offset += band.count();
    }
    EXPECT_EQ(offset, count);
}

/** Pure topology oracle: traverse source 3D faces, independently of the factored radius construction. */
void verify_virtual_partition(const SimpleFluid::Meshes::SemiStructMeshTopo& source, const Topology& local,
    const std::set<std::size_t>& owned, std::size_t depth)
{
    const auto& indexer = source.indexer();
    auto expected_cells = owned, frontier = owned;
    for (std::size_t layer = 0; layer < depth; ++layer)
    {
        std::set<std::size_t> next;
        for (const auto raw : frontier)
        {
            const auto cell = indexer.cell_id(raw);
            for (const auto face : source.cell_faces(cell))
            {
                const auto neighbor =
                    source.owner_cell(face) == cell ? source.neighbor_cell(face) : source.owner_cell(face);
                if (neighbor != Semi::invalid_cell_id())
                {
                    const auto id = indexer.cell_ordinal(neighbor);
                    if (!expected_cells.contains(id))
                        next.insert(id);
                }
            }
        }
        expected_cells.insert(next.begin(), next.end());
        frontier = std::move(next);
    }
    EXPECT_EQ(local.num_owned_cells(), owned.size());
    std::set<std::size_t> cells, expected_faces, expected_nodes;
    for (std::size_t row = 0; row < local.num_local_cells(); ++row)
    {
        const auto cell = local.source_cell(row);
        const auto raw = indexer.cell_ordinal(cell);
        cells.insert(raw);
        EXPECT_EQ(local.cell_local_id(cell), row);
        EXPECT_EQ(row < local.num_owned_cells(), owned.contains(raw));
        std::set<std::size_t> expected_incidence, actual_incidence;
        for (const auto face : source.cell_faces(cell))
        {
            expected_faces.insert(indexer.face_ordinal(face));
            const auto face_row = local.face_local_id(face);
            EXPECT_NE(face_row, Topology::invalid_local_id());
            if (face_row != Topology::invalid_local_id())
                expected_incidence.insert(face_row);
        }
        for (const auto face : local.cell_faces(row))
            actual_incidence.insert(face);
        EXPECT_EQ(actual_incidence, expected_incidence);
        for (const auto side : source.cell_side_faces(cell.ij))
            for (const auto node : source.side_face(side).nodes)
                for (const auto plane : {cell.k, static_cast<unsigned>((cell.k + 1) % indexer.num_node_layers)})
                    expected_nodes.insert(indexer.node_ordinal({node, plane}));
    }
    EXPECT_EQ(cells, expected_cells);
    for (std::size_t raw = 0; raw < indexer.total_cells(); ++raw)
    {
        const auto row = local.cell_local_id(indexer.cell_id(raw));
        EXPECT_EQ(row != Topology::invalid_local_id(), expected_cells.contains(raw));
        if (row != Topology::invalid_local_id())
            EXPECT_EQ(indexer.cell_ordinal(local.source_cell(row)), raw);
    }
    std::set<std::size_t> faces, nodes;
    for (std::size_t row = 0; row < local.num_local_faces(); ++row)
    {
        const auto face = local.source_face(row);
        const auto raw = indexer.face_ordinal(face);
        faces.insert(raw);
        EXPECT_EQ(local.face_local_id(face), row);
        EXPECT_EQ(local.owner_cell(row), local.cell_local_id(source.owner_cell(face)));
        EXPECT_EQ(local.neighbor_cell(row), local.cell_local_id(source.neighbor_cell(face)));
        EXPECT_EQ(local.boundary_id(row), source.boundary_id(face));
        EXPECT_EQ(row < local.num_owned_faces(), owned.contains(indexer.cell_ordinal(source.owner_cell(face))));
    }
    EXPECT_EQ(faces, expected_faces);
    for (std::size_t raw = 0; raw < indexer.total_faces(); ++raw)
    {
        const auto row = local.face_local_id(indexer.face_id(raw));
        EXPECT_EQ(row != Topology::invalid_local_id(), expected_faces.contains(raw));
        if (row != Topology::invalid_local_id())
            EXPECT_EQ(indexer.face_ordinal(local.source_face(row)), raw);
    }
    for (std::size_t row = 0; row < local.num_local_nodes(); ++row)
    {
        const auto node = local.source_node(row);
        const auto raw = indexer.node_ordinal(node);
        nodes.insert(raw);
        EXPECT_EQ(local.node_local_id(node), row);
    }
    EXPECT_EQ(nodes, expected_nodes);
    for (std::size_t raw = 0; raw < indexer.total_nodes(); ++raw)
    {
        const auto row = local.node_local_id(indexer.node_id(raw));
        EXPECT_EQ(row != Topology::invalid_local_id(), expected_nodes.contains(raw));
        if (row != Topology::invalid_local_id())
            EXPECT_EQ(indexer.node_ordinal(local.source_node(row)), raw);
    }
    expect_bands_cover(local.cell_bands(), local.num_local_cells());
    expect_bands_cover(local.face_bands(), local.num_local_faces());
    expect_bands_cover(local.node_bands(), local.num_local_nodes());
}

void verify_roundtrips(
    const std::shared_ptr<Semi>& source, const Handle& handle, const Topology& local, std::size_t depth)
{
    EXPECT_EQ(&local.source_topology(), &source->topology());
    EXPECT_EQ(local.num_owned_cells(), handle.num_owned_cells());
    EXPECT_EQ(local.num_local_cells(), handle.num_local_cells());
    EXPECT_EQ(local.num_owned_faces(), handle.num_owned_faces());
    EXPECT_EQ(local.num_local_faces(), handle.num_faces());
    const auto expected_cells = breadth_first_cells(*source, owned_cells(handle), depth);
    std::set<std::size_t> actual_cells, expected_faces, expected_nodes;
    for (std::size_t row = 0; row < local.num_local_cells(); ++row)
    {
        const auto cell = local.source_cell(row);
        const auto global = source->indexer().cell_ordinal(cell);
        actual_cells.insert(global);
        EXPECT_EQ(local.cell_local_id(cell), row);
        EXPECT_EQ(handle.cell_geometry_global_id(static_cast<LO>(row)), global);
        EXPECT_EQ(handle.is_owned_cell(static_cast<LO>(row)), row < local.num_owned_cells());
        std::set<std::size_t> actual_incidence, expected_incidence;
        for (const auto face : source->faces(cell))
        {
            const auto face_row = local.face_local_id(face);
            EXPECT_NE(face_row, Topology::invalid_local_id());
            if (face_row != Topology::invalid_local_id())
                expected_incidence.insert(face_row);
            expected_faces.insert(source->indexer().face_ordinal(face));
        }
        for (const auto face : handle.faces(static_cast<LO>(row)))
            actual_incidence.insert(static_cast<std::size_t>(face));
        EXPECT_EQ(actual_incidence, expected_incidence);
        for (const auto node : source->xy_cell_nodes()[cell.ij])
            for (const auto k : {cell.k, cell.k + 1})
                expected_nodes.insert(source->indexer().node_ordinal({node, k}));
    }
    EXPECT_EQ(actual_cells, expected_cells);
    for (std::size_t global = 0; global < source->num_cells(); ++global)
    {
        const auto row = local.cell_local_id(source->indexer().cell_id(global));
        EXPECT_EQ(row != Topology::invalid_local_id(), expected_cells.contains(global));
        if (row != Topology::invalid_local_id())
            EXPECT_EQ(source->indexer().cell_ordinal(local.source_cell(row)), global);
    }
    std::set<std::size_t> actual_faces, actual_nodes;
    int missing_halo_neighbors = 0;
    for (std::size_t row = 0; row < local.num_local_faces(); ++row)
    {
        const auto face = local.source_face(row);
        const auto global = source->indexer().face_ordinal(face);
        actual_faces.insert(global);
        EXPECT_EQ(local.face_local_id(face), row);
        EXPECT_EQ(handle.face_geometry_global_id(static_cast<LO>(row)), global);
        const auto source_owner = source->owner_cell(face), source_neighbor = source->neighbor_cell(face);
        const auto owner = local.cell_local_id(source_owner);
        const auto neighbor = source_neighbor == Semi::invalid_cell_id() ? Topology::invalid_local_id()
                                                                         : local.cell_local_id(source_neighbor);
        const auto expected_owner =
            owner == Topology::invalid_local_id() ? Handle::invalid_local_id() : static_cast<LO>(owner);
        const auto expected_neighbor =
            neighbor == Topology::invalid_local_id() ? Handle::invalid_local_id() : static_cast<LO>(neighbor);
        EXPECT_EQ(handle.owner_cell(static_cast<LO>(row)), expected_owner);
        EXPECT_EQ(handle.neighbor_cell(static_cast<LO>(row)), expected_neighbor);
        EXPECT_EQ(handle.boundary_id(static_cast<LO>(row)), source->boundary_id(face));
        const bool boundary = source_neighbor == Semi::invalid_cell_id();
        EXPECT_EQ(handle.is_geometry_exterior_face(static_cast<LO>(row)), boundary);
        EXPECT_EQ(handle.is_boundary_face(static_cast<LO>(row)), boundary);
        if (!boundary && (owner == Topology::invalid_local_id() || neighbor == Topology::invalid_local_id()))
        {
            ++missing_halo_neighbors;
            EXPECT_FALSE(handle.is_boundary_face(static_cast<LO>(row)));
        }
        const auto expected_normal = source->face_normal(face);
        const auto actual_normal = handle.face_normal(static_cast<LO>(row));
        EXPECT_DOUBLE_EQ(actual_normal.x, expected_normal.x);
        EXPECT_DOUBLE_EQ(actual_normal.y, expected_normal.y);
        EXPECT_DOUBLE_EQ(actual_normal.z, expected_normal.z);
    }
    EXPECT_EQ(actual_faces, expected_faces);
    for (std::size_t global = 0; global < source->num_faces(); ++global)
    {
        const auto row = local.face_local_id(source->indexer().face_id(global));
        EXPECT_EQ(row != Topology::invalid_local_id(), expected_faces.contains(global));
        if (row != Topology::invalid_local_id())
            EXPECT_EQ(source->indexer().face_ordinal(local.source_face(row)), global);
    }
    for (std::size_t row = 0; row < local.num_local_nodes(); ++row)
    {
        const auto node = local.source_node(row);
        const auto global = source->indexer().node_ordinal(node);
        actual_nodes.insert(global);
        EXPECT_EQ(local.node_local_id(node), row);
    }
    EXPECT_EQ(actual_nodes, expected_nodes);
    for (std::size_t global = 0; global < source->num_nodes(); ++global)
    {
        const auto row = local.node_local_id(source->indexer().node_id(global));
        EXPECT_EQ(row != Topology::invalid_local_id(), expected_nodes.contains(global));
        if (row != Topology::invalid_local_id())
            EXPECT_EQ(source->indexer().node_ordinal(local.source_node(row)), global);
    }
    expect_bands_cover(local.cell_bands(), local.num_local_cells());
    expect_bands_cover(local.face_bands(), local.num_local_faces());
    expect_bands_cover(local.node_bands(), local.num_local_nodes());
    EXPECT_GT(global_sum(missing_halo_neighbors), 0);
}
} // namespace

TEST(DecomposedSemiStructuredTopologyTest, ObserverIsAbsentForSerialAndOtherMeshFamilies)
{
    const auto source = chain(8);
    const Handle semi(source, {.allow_empty_partitions = true});
    const auto local = semi.decomposed_semi_structured_topology();
    EXPECT_EQ(static_cast<bool>(local), Tpetra::getDefaultComm()->getSize() > 1);
    const auto ranks = static_cast<std::size_t>(Tpetra::getDefaultComm()->getSize());
    SimpleFluid::ArrReal x(ranks + 1);
    std::iota(x.begin(), x.end(), 0.);
    const auto cartesian =
        std::make_shared<Handle::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{x, {0., 1.}, {0., 1.}}});
    EXPECT_FALSE(Handle(cartesian).decomposed_semi_structured_topology());
}

TEST(DecomposedSemiStructuredTopologyTest, CanonicalRoundtripsMatchExactBFSAndSourceIncidence)
{
    if (Tpetra::getDefaultComm()->getSize() == 1)
        GTEST_SKIP() << "The decomposed topology observer belongs to distributed native handles.";
    const auto source = chain(9);
    for (const std::size_t depth : {1U, 2U})
    {
        const Handle handle(source, hybrid_options(depth));
        const auto local = handle.decomposed_semi_structured_topology();
        EXPECT_TRUE(local);
        EXPECT_FALSE(handle.has_materialized_indexer());
        if (local)
            verify_roundtrips(source, handle, *local, depth);
        EXPECT_FALSE(handle.has_materialized_indexer());
        EXPECT_FALSE(handle.has_materialized_connectivity());
    }
}

TEST(DecomposedSemiStructuredTopologyTest, ManhattanRadiusBandsExcludeProductCornersAndReuseXYViews)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires a distributed XY graph and Z slabs.";
    const auto source = chain(9);
    constexpr unsigned depth = 2;
    const Handle handle(source, hybrid_options(depth));
    const auto local = handle.decomposed_semi_structured_topology();
    EXPECT_TRUE(local);
    if (!local)
        return;
    const auto factors = handle.semi_structured_decomposition();
    const auto slab = static_cast<std::size_t>(comm->getRank()) / factors.xy_partitions;
    const auto nz = source->indexer().num_layers;
    const auto first = nz / factors.z_partitions * slab + std::min(nz % factors.z_partitions, slab);
    const auto z_count = nz / factors.z_partitions + (slab < nz % factors.z_partitions);
    const auto past = first + z_count;
    std::vector<unsigned> distance(source->indexer().num_cells_per_layer, 1000000);
    std::set<unsigned> frontier;
    for (const auto cell : owned_cells(handle))
    {
        const auto xy = source->indexer().cell_id(cell).ij;
        distance[xy] = 0;
        frontier.insert(xy);
    }
    for (unsigned layer = 1; layer <= depth; ++layer)
    {
        std::set<unsigned> next;
        for (const auto xy : frontier)
            for (const auto& edge : source->topology().side_faces())
            {
                if (edge.neighbor == static_cast<unsigned>(-1))
                    continue;
                const auto other = edge.owner == xy      ? edge.neighbor
                                   : edge.neighbor == xy ? edge.owner
                                                         : static_cast<unsigned>(-1);
                if (other != static_cast<unsigned>(-1) && distance[other] > layer)
                {
                    distance[other] = layer;
                    next.insert(other);
                }
            }
        frontier = std::move(next);
    }
    unsigned forbidden_corners = 0;
    for (unsigned k = 0; k < nz; ++k)
    {
        const auto dz = k < first ? first - k : k >= past ? k - past + 1 : 0;
        for (unsigned xy = 0; xy < distance.size(); ++xy)
        {
            const bool included = dz + distance[xy] <= depth;
            EXPECT_EQ(local->cell_local_id({xy, k}) != Topology::invalid_local_id(), included);
            forbidden_corners += dz <= depth && distance[xy] <= depth && !included;
        }
    }
    if (factors.xy_partitions > 1 && factors.z_partitions > 1)
        EXPECT_GT(global_sum(forbidden_corners), 0U);
    const auto* shared_xy = local->xy_cells().data();
    const auto storage = local->storage_bytes();
    std::size_t reused_layers = 0;
    for (const auto& band : local->cell_bands())
    {
        EXPECT_EQ(local->xy_view_size(band.view), band.xy_size);
        const auto view = local->view_source_ids(band.view);
        EXPECT_EQ(view.size(), band.xy_size);
        if (band.z_end - band.z_begin > 1)
        {
            reused_layers += band.z_end - band.z_begin - 1;
            for (std::size_t xy = 0; xy < view.size(); ++xy)
            {
                const auto lower = local->source_cell(band.offset + xy);
                const auto upper =
                    local->source_cell(band.offset + (band.z_end - band.z_begin - 1) * band.xy_size + xy);
                EXPECT_EQ(lower.ij, upper.ij);
                EXPECT_EQ(lower.ij, view[xy]);
            }
        }
    }
    EXPECT_GT(global_sum(reused_layers), 0U);
    EXPECT_EQ(local->xy_cells().data(), shared_xy);
    EXPECT_EQ(local->storage_bytes(), storage);
    EXPECT_FALSE(handle.has_materialized_indexer());
}

TEST(DecomposedSemiStructuredTopologyTest, ProductBandsStorageDoesNotScaleWithExtrudedCellCount)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires the distributed compact topology observer.";
    const auto short_source = chain(8), long_source = chain(128);
    const auto ranks = static_cast<std::size_t>(comm->getSize());
    const Handle short_mesh(short_source, {.ghost_layers = 2, .xy_partitions = ranks, .z_partitions = 1});
    const Handle long_mesh(long_source, {.ghost_layers = 2, .xy_partitions = ranks, .z_partitions = 1});
    const auto short_local = short_mesh.decomposed_semi_structured_topology();
    const auto long_local = long_mesh.decomposed_semi_structured_topology();
    EXPECT_TRUE(short_local);
    EXPECT_TRUE(long_local);
    if (!short_local || !long_local)
        return;
    EXPECT_EQ(long_local->num_local_cells(), 16 * short_local->num_local_cells());
    EXPECT_EQ(long_local->xy_cells().size(), short_local->xy_cells().size());
    EXPECT_EQ(long_local->xy_sides().size(), short_local->xy_sides().size());
    EXPECT_EQ(long_local->xy_nodes().size(), short_local->xy_nodes().size());
    EXPECT_EQ(long_local->cell_bands().size(), short_local->cell_bands().size());
    EXPECT_EQ(long_local->face_bands().size(), short_local->face_bands().size());
    EXPECT_EQ(long_local->node_bands().size(), short_local->node_bands().size());
    EXPECT_EQ(long_local->storage_bytes(), short_local->storage_bytes());
    EXPECT_EQ(long_local->lookup_storage_estimate(), short_local->lookup_storage_estimate());
    EXPECT_FALSE(short_mesh.has_materialized_indexer());
    EXPECT_FALSE(long_mesh.has_materialized_indexer());
    EXPECT_FALSE(short_mesh.has_materialized_connectivity());
    EXPECT_FALSE(long_mesh.has_materialized_connectivity());
}

TEST(DecomposedSemiStructuredTopologyTest, EmptyOwnersRetainSourceAliasAndHaveNoInventedEntityBands)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires actual empty distributed owners.";
    const auto source = triangle();
    const auto ranks = static_cast<std::size_t>(comm->getSize());
    const Handle handle(source, {.ghost_layers = 2,
                                    .allow_empty_partitions = true,
                                    .xy_partitions = ranks % 2 == 0 ? 2U : 1U,
                                    .z_partitions = ranks % 2 == 0 ? ranks / 2 : ranks});
    const auto local = handle.decomposed_semi_structured_topology();
    EXPECT_TRUE(local);
    const int empty = handle.num_owned_cells() == 0 ? 1 : 0;
    EXPECT_EQ(global_sum(empty), comm->getSize() - 1);
    if (!local)
        return;
    EXPECT_EQ(&local->source_topology(), &source->topology());
    if (empty)
    {
        EXPECT_EQ(local->num_owned_cells(), 0U);
        EXPECT_EQ(local->num_local_cells(), 0U);
        EXPECT_EQ(local->num_owned_faces(), 0U);
        EXPECT_EQ(local->num_local_faces(), 0U);
        EXPECT_EQ(local->num_local_nodes(), 0U);
        EXPECT_TRUE(local->cell_bands().empty());
        EXPECT_TRUE(local->face_bands().empty());
        EXPECT_TRUE(local->node_bands().empty());
        for (std::size_t global = 0; global < source->num_cells(); ++global)
            EXPECT_EQ(local->cell_local_id(source->indexer().cell_id(global)), Topology::invalid_local_id());
        for (std::size_t global = 0; global < source->num_faces(); ++global)
            EXPECT_EQ(local->face_local_id(source->indexer().face_id(global)), Topology::invalid_local_id());
        for (std::size_t global = 0; global < source->num_nodes(); ++global)
            EXPECT_EQ(local->node_local_id(source->indexer().node_id(global)), Topology::invalid_local_id());
    }
    for (const auto id : source->boundary_batch_ids())
    {
        EXPECT_EQ(handle.boundary_batch_name(id), source->boundary_batch_name(id));
        if (empty)
            EXPECT_TRUE(handle.boundary_face_batch(id).face_lids.empty());
    }
    EXPECT_FALSE(handle.has_materialized_indexer());
}

/** Runs without MeshHandle construction or collectives, including in the serial test executable. */
TEST(DecomposedSemiStructuredTopologyTest, PureConstructorMatchesVirtualHybridRanksWithPeriodicAndClosedZ)
{
    const auto geometry = chain(9);
    constexpr std::size_t nxy = 24, xy_parts = 2, z_parts = 2;
    std::vector<std::size_t> labels(nxy), order(nxy);
    std::iota(order.begin(), order.end(), 0);
    const auto physical_x = [](std::size_t raw) { return 2 * (raw % 12) + raw / 12; };
    for (std::size_t xy = 0; xy < nxy; ++xy)
        labels[xy] = physical_x(xy) < 12 ? 0 : 1;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
        { return labels[a] != labels[b] ? labels[a] < labels[b] : physical_x(a) > physical_x(b); });
    for (const bool periodic : {false, true})
    {
        SCOPED_TRACE(periodic);
        const auto source =
            periodic ? std::make_shared<const SimpleFluid::Meshes::SemiStructMeshTopo>(geometry->xy_nodes().size(),
                           geometry->xy_cell_nodes(), 9,
                           SimpleFluid::Arr<SimpleFluid::Meshes::SemiStructMeshTopo::BoundaryEdge>{}, true)
                     : std::shared_ptr<const SimpleFluid::Meshes::SemiStructMeshTopo>(geometry, &geometry->topology());
        for (std::size_t rank = 0; rank < xy_parts * z_parts; ++rank)
        {
            SCOPED_TRACE(rank);
            const auto slab = rank / xy_parts;
            const auto begin = 9 / z_parts * slab + std::min(9 % z_parts, slab);
            const auto count = 9 / z_parts + (slab < 9 % z_parts);
            std::set<std::size_t> owned;
            for (std::size_t k = begin; k < begin + count; ++k)
                for (std::size_t xy = 0; xy < nxy; ++xy)
                    if (labels[xy] == rank % xy_parts)
                        owned.insert(
                            source->indexer().cell_ordinal({static_cast<unsigned>(xy), static_cast<unsigned>(k)}));
            for (const std::size_t depth : {1U, 2U})
            {
                SCOPED_TRACE(depth);
                const Topology local(source, labels, order, rank, xy_parts, z_parts, depth);
                EXPECT_EQ(&local.source_topology(), source.get());
                verify_virtual_partition(*source, local, owned, depth);
            }
        }
    }
}

TEST(DecomposedSemiStructuredTopologyTest, PureConstructorRejectsInvalidLabelsPermutationFactorsAndLocalIds)
{
    const auto geometry = chain(9);
    const std::shared_ptr<const SimpleFluid::Meshes::SemiStructMeshTopo> source(geometry, &geometry->topology());
    std::vector<std::size_t> labels(24), order(24);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t xy = 12; xy < labels.size(); ++xy)
        labels[xy] = 1;
    EXPECT_THROW((Topology({}, labels, order, 0, 2, 2, 1)), std::invalid_argument);
    EXPECT_THROW((Topology(source, labels, order, 0, 0, 2, 1)), std::invalid_argument);
    EXPECT_THROW((Topology(source, labels, order, 0, 2, 0, 1)), std::invalid_argument);
    EXPECT_THROW((Topology(source, labels, order, 0, 2, 2, 0)), std::invalid_argument);
    EXPECT_THROW((Topology(source, labels, order, 4, 2, 2, 1)), std::invalid_argument);
    auto bad_labels = labels;
    bad_labels.pop_back();
    EXPECT_THROW((Topology(source, bad_labels, order, 0, 2, 2, 1)), std::invalid_argument);
    bad_labels = labels;
    bad_labels[0] = 2;
    EXPECT_THROW((Topology(source, bad_labels, order, 0, 2, 2, 1)), std::invalid_argument);
    auto bad_order = order;
    bad_order[0] = bad_order[1];
    EXPECT_THROW((Topology(source, labels, bad_order, 0, 2, 2, 1)), std::invalid_argument);
    const Topology valid(source, labels, order, 0, 2, 2, 1);
    EXPECT_THROW(valid.source_cell(valid.num_local_cells()), std::out_of_range);
    EXPECT_THROW(valid.source_face(valid.num_local_faces()), std::out_of_range);
    EXPECT_THROW(valid.source_node(valid.num_local_nodes()), std::out_of_range);
    EXPECT_EQ(valid.cell_local_id(Semi::invalid_cell_id()), Topology::invalid_local_id());
    EXPECT_EQ(valid.face_local_id({0, 0, 99}), Topology::invalid_local_id());
    EXPECT_EQ(valid.node_local_id({static_cast<unsigned>(-1), 0}), Topology::invalid_local_id());
}
