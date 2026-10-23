/** @file testSemiStructuredDistributed.cc
 * @brief Independent geometry, ownership and collective-validation checks for native XY/Z partitions.
 */
#include "geometry/MeshHandle.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Semi = SimpleFluid::Meshes::SemiStructuredXY_Z;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;

testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Semi> mixed_mesh(bool single_layer = false, double width_change = 0.0)
{
    // A connected pentagon, triangle and quadrilateral tile a rectangle of area two.
    return std::make_shared<Semi>(SimpleFluid::Arr<Semi::Vec3>{{0., 0., 0.}, {1., 0., 0.}, {1.5, .5, 0.}, {1., 1., 0.},
                                      {0., 1., 0.}, {2. + width_change, 0., 0.}, {2. + width_change, 1., 0.}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2, 3, 4}, {1, 5, 2}, {2, 5, 6, 3}},
        single_layer ? SimpleFluid::ArrReal{-.5, 2.} : SimpleFluid::ArrReal{-.5, -.1, .7, 2.},
        SimpleFluid::Arr<Semi::BoundaryEdge>{
            {0, 1, "bottom"}, {1, 5, "bottom"}, {5, 6, "right"}, {6, 3, "top"}, {3, 4, "top"}, {4, 0, "left"}});
}

std::shared_ptr<Semi> one_triangle()
{
    return std::make_shared<Semi>(SimpleFluid::Arr<Semi::Vec3>{{0., 0., 0.}, {2., 0., 0.}, {0., 1., 0.}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}}, SimpleFluid::ArrReal{0., 3.});
}

template<class T> T global_sum(T local)
{
    T result{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

void expect_vector(const auto& actual, const auto& expected)
{
    EXPECT_NEAR(actual.x, expected.x, 2e-13);
    EXPECT_NEAR(actual.y, expected.y, 2e-13);
    EXPECT_NEAR(actual.z, expected.z, 2e-13);
}

std::set<std::size_t> expected_overlap(const Semi& native, std::set<std::size_t> owned, std::size_t depth)
{
    auto result = owned;
    auto frontier = std::move(owned);
    const auto& indexer = native.indexer();
    for (std::size_t layer = 0; layer < depth; ++layer)
    {
        std::set<std::size_t> next;
        for (const auto gid : frontier)
        {
            const auto cell = indexer.cell_id(gid);
            for (const auto face : native.faces(cell))
            {
                const auto other = native.opposite_cell(face, cell);
                if (other != Semi::invalid_cell_id())
                {
                    const auto id = indexer.cell_ordinal(other);
                    if (!result.contains(id))
                        next.insert(id);
                }
            }
        }
        result.insert(next.begin(), next.end());
        frontier = std::move(next);
    }
    return result;
}

/** Actual owned maps identify labels; native graph/column invariants independently validate them. */
std::vector<int> global_cell_owners(const Handle& handle, const Semi& native)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> local(native.num_cells()), owners(native.num_cells());
    for (std::size_t row = 0; row < handle.num_owned_cells(); ++row)
    {
        const auto gid = static_cast<std::size_t>(handle.cell_geometry_global_id(static_cast<LO>(row)));
        if (gid < local.size())
            local[gid] = comm->getRank() + 1;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(local.size()), local.data(), owners.data());
    for (auto& owner : owners)
    {
        EXPECT_GT(owner, 0);
        EXPECT_LE(owner, comm->getSize());
        --owner;
    }
    return owners;
}

/** Verify contiguous owned algebraic maps and exact owner-to-ghost allocation by canonical geometry identity. */
void verify_algebraic_maps(const Handle& handle, const Semi& native)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto ranks = static_cast<std::size_t>(comm->getSize());
    const auto rank = static_cast<std::size_t>(comm->getRank());
    EXPECT_TRUE(handle.owned_cell_map()->isContiguous());
    EXPECT_TRUE(handle.owned_face_map()->isContiguous());
    EXPECT_TRUE(handle.owned_cell_map()->isOneToOne());
    EXPECT_TRUE(handle.owned_face_map()->isOneToOne());
    EXPECT_EQ(handle.owned_cell_map()->getGlobalNumElements(), native.num_cells());
    EXPECT_EQ(handle.owned_face_map()->getGlobalNumElements(), native.num_faces());
    std::vector<GO> local_counts(2 * ranks), counts(2 * ranks);
    local_counts[rank] = static_cast<GO>(handle.num_owned_cells());
    local_counts[ranks + rank] = static_cast<GO>(handle.num_owned_faces());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(counts.size()), local_counts.data(), counts.data());
    GO cell_prefix = 0, face_prefix = 0;
    for (std::size_t preceding = 0; preceding < rank; ++preceding)
    {
        cell_prefix += counts[preceding];
        face_prefix += counts[ranks + preceding];
    }
    std::vector<GO> local_cell_allocation(native.num_cells()), global_cell_allocation(native.num_cells());
    std::vector<GO> local_face_allocation(native.num_faces()), global_face_allocation(native.num_faces());
    for (std::size_t row = 0; row < handle.num_owned_cells(); ++row)
    {
        const auto lid = static_cast<LO>(row);
        const auto algebraic = cell_prefix + static_cast<GO>(row);
        const auto geometry = static_cast<std::size_t>(handle.cell_geometry_global_id(lid));
        EXPECT_EQ(handle.cell_global_id(lid), algebraic);
        EXPECT_EQ(handle.owned_cell_map()->getGlobalElement(lid), algebraic);
        EXPECT_EQ(handle.overlap_cell_map()->getGlobalElement(lid), algebraic);
        if (geometry < local_cell_allocation.size())
            local_cell_allocation[geometry] = algebraic + 1;
    }
    for (std::size_t row = 0; row < handle.num_owned_faces(); ++row)
    {
        const auto lid = static_cast<LO>(row);
        const auto algebraic = face_prefix + static_cast<GO>(row);
        const auto geometry = static_cast<std::size_t>(handle.face_geometry_global_id(lid));
        EXPECT_EQ(handle.face_global_id(lid), algebraic);
        EXPECT_EQ(handle.owned_face_map()->getGlobalElement(lid), algebraic);
        EXPECT_EQ(handle.overlap_face_map()->getGlobalElement(lid), algebraic);
        if (geometry < local_face_allocation.size())
            local_face_allocation[geometry] = algebraic + 1;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(native.num_cells()), local_cell_allocation.data(),
        global_cell_allocation.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(native.num_faces()), local_face_allocation.data(),
        global_face_allocation.data());
    for (const auto allocation : global_cell_allocation)
        EXPECT_GT(allocation, 0);
    for (const auto allocation : global_face_allocation)
        EXPECT_GT(allocation, 0);
    const auto& indexer = handle.indexer();
    for (std::size_t row = 0; row < handle.num_local_cells(); ++row)
    {
        const auto lid = static_cast<LO>(row);
        const auto geometry = static_cast<std::size_t>(handle.cell_geometry_global_id(lid));
        const auto algebraic = handle.cell_global_id(lid);
        EXPECT_EQ(handle.overlap_cell_map()->getGlobalElement(lid), algebraic);
        EXPECT_EQ(handle.overlap_cell_map()->getLocalElement(algebraic), lid);
        EXPECT_EQ(indexer.cell_global_id(lid), static_cast<GO>(geometry));
        if (geometry < global_cell_allocation.size())
            EXPECT_EQ(algebraic, global_cell_allocation[geometry] - 1);
    }
    for (std::size_t row = 0; row < handle.num_faces(); ++row)
    {
        const auto lid = static_cast<LO>(row);
        const auto geometry = static_cast<std::size_t>(handle.face_geometry_global_id(lid));
        const auto algebraic = handle.face_global_id(lid);
        EXPECT_EQ(handle.overlap_face_map()->getGlobalElement(lid), algebraic);
        EXPECT_EQ(handle.overlap_face_map()->getLocalElement(algebraic), lid);
        EXPECT_EQ(indexer.face_global_id(lid), static_cast<GO>(geometry));
        if (geometry < global_face_allocation.size())
            EXPECT_EQ(algebraic, global_face_allocation[geometry] - 1);
    }
}

std::vector<std::size_t> verify_hybrid_ownership(
    const Handle& handle, const Semi& native, const std::vector<int>& owners)
{
    const auto factors = handle.semi_structured_decomposition();
    const std::size_t px = factors.xy_partitions, pz = factors.z_partitions;
    const std::size_t ranks = Tpetra::getDefaultComm()->getSize();
    const auto& indexer = native.indexer();
    const std::size_t nxy = indexer.num_cells_per_layer, nz = indexer.num_layers;
    EXPECT_GT(px, 0U);
    EXPECT_GT(pz, 0U);
    EXPECT_EQ(px * pz, ranks);
    std::vector<std::size_t> xy_owners(nxy);
    for (std::size_t xy = 0; xy < nxy; ++xy)
        xy_owners[xy] = static_cast<std::size_t>(owners[xy]) % px;
    for (std::size_t slab = 0; slab < pz; ++slab)
    {
        const auto begin = nz / pz * slab + std::min(nz % pz, slab);
        const auto count = nz / pz + (slab < nz % pz);
        for (std::size_t k = begin; k < begin + count; ++k)
            for (std::size_t xy = 0; xy < nxy; ++xy)
                EXPECT_EQ(owners[indexer.cell_ordinal({static_cast<unsigned>(xy), static_cast<unsigned>(k)})],
                    static_cast<int>(slab * px + xy_owners[xy]));
    }
    std::set<std::size_t> active_xy(xy_owners.begin(), xy_owners.end());
    EXPECT_EQ(active_xy.size(), std::min(nxy, px));
    const auto empty = global_sum(handle.num_owned_cells() == 0 ? 1 : 0);
    EXPECT_EQ(empty, static_cast<int>(ranks - std::min(nxy, px) * std::min(nz, pz)));
    return xy_owners;
}

/** A physical path whose even/odd cell ordinals sabotage contiguous-ID partitions. */
std::shared_ptr<Semi> shuffled_chain()
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
    return std::make_shared<Semi>(nodes, loops, SimpleFluid::ArrReal{-.5, -.1, .7, 2.});
}

/** Native compact geometry is the serial oracle; no serial MeshHandle is constructed on MPI. */
void verify_partition(const std::shared_ptr<Semi>& native, std::size_t ghost_depth,
    Handle::DistributionOptions options = {}, double expected_volume = 5.)
{
    const auto comm = Tpetra::getDefaultComm();
    const std::size_t ranks = comm->getSize();
    const std::size_t count = native->num_cells();
    options.ghost_layers = ghost_depth;
    options.allow_empty_partitions = true;
    const Handle handle(native, options);
    verify_algebraic_maps(handle, *native);
    const auto owners = global_cell_owners(handle, *native);
    const auto xy_owners = verify_hybrid_ownership(handle, *native, owners);
    const auto xy_order = SimpleFluid::MeshReorderingFactory<Pack>::partitioned_xy_order(native->topology(), xy_owners);
    std::vector<std::size_t> inverse_xy(xy_order.size());
    for (std::size_t local = 0; local < xy_order.size(); ++local)
        inverse_xy[xy_order[local]] = local;
    const auto owned_count = handle.num_owned_cells();
    EXPECT_EQ(handle.owned_cell_map()->getGlobalNumElements(), count);
    EXPECT_EQ(handle.owned_cell_map()->getComm().get(), comm.get());
    EXPECT_FALSE(handle.has_materialized_connectivity());
    std::set<std::size_t> owned;
    std::vector<std::size_t> owned_order;
    std::vector<int> cell_owners(count, 0), face_owners(native->num_faces(), 0);
    double volume = 0.;
    for (std::size_t local = 0; local < handle.num_owned_cells(); ++local)
    {
        const auto lid = static_cast<LO>(local);
        const auto gid = static_cast<std::size_t>(handle.cell_geometry_global_id(lid));
        owned_order.push_back(gid);
        if (gid < count)
        {
            owned.insert(gid);
            ++cell_owners[gid];
        }
        volume += handle.cell_volume(lid);
        EXPECT_TRUE(handle.is_owned_cell(lid));
    }
    const auto expected = expected_overlap(*native, owned, ghost_depth);
    std::set<std::size_t> overlap;
    std::set<GO> overlap_nodes;
    for (std::size_t local = 0; local < handle.num_local_cells(); ++local)
    {
        const auto lid = static_cast<LO>(local);
        const auto gid = static_cast<std::size_t>(handle.cell_geometry_global_id(lid));
        overlap.insert(gid);
        EXPECT_EQ(handle.is_owned_cell(lid), local < owned_count);
        if (gid >= count)
            continue;
        const auto cell = native->indexer().cell_id(gid);
        EXPECT_NEAR(handle.cell_volume(lid), native->cell_volume(cell), 2e-13);
        expect_vector(handle.cell_centroid(lid), native->cell_centroid(cell));
        const auto& nodes = native->xy_cell_nodes()[cell.ij];
        for (const auto node : nodes)
            for (const auto k : {cell.k, cell.k + 1})
                overlap_nodes.insert(static_cast<GO>(native->indexer().node_ordinal({node, k})));
        std::set<std::size_t> actual_faces;
        Semi::Vec3 sum{};
        for (const auto face_lid : handle.faces(lid))
        {
            const auto face_gid = static_cast<std::size_t>(handle.face_geometry_global_id(face_lid));
            actual_faces.insert(face_gid);
            if (face_gid >= native->num_faces())
                continue;
            const auto face = native->indexer().face_id(face_gid);
            expect_vector(handle.face_area_vector_outward(face_lid, lid), native->face_area_vector_outward(face, cell));
            sum = sum + handle.face_area_vector_outward(face_lid, lid);
        }
        // Ghost fringe incidences may be filtered to locally represented faces.
        for (const auto face_gid : actual_faces)
        {
            bool incident = false;
            for (const auto face : native->faces(cell))
                incident = incident || native->indexer().face_ordinal(face) == face_gid;
            EXPECT_TRUE(incident);
        }
        if (local < owned_count)
        {
            EXPECT_EQ(actual_faces.size(), native->faces(cell).size());
            expect_vector(sum, Semi::Vec3{});
        }
    }
    EXPECT_EQ(overlap, expected);
    const auto order_key = [&](std::size_t gid)
    {
        const auto cell = native->indexer().cell_id(gid);
        return std::pair{cell.k, inverse_xy[cell.ij]};
    };
    if (ranks > 1)
    {
        EXPECT_TRUE(std::ranges::is_sorted(owned_order, std::less{}, order_key));
        std::vector<std::size_t> ghost_order;
        for (std::size_t local = handle.num_owned_cells(); local < handle.num_local_cells(); ++local)
            ghost_order.push_back(static_cast<std::size_t>(handle.cell_geometry_global_id(static_cast<LO>(local))));
        EXPECT_TRUE(std::ranges::is_sorted(ghost_order, std::less{}, order_key));
    }
    const auto& indexer = handle.indexer();
    std::set<GO> actual_nodes;
    for (std::size_t local = 0; local < indexer.num_local_nodes(); ++local)
        actual_nodes.insert(indexer.node_global_id(static_cast<LO>(local)));
    EXPECT_EQ(actual_nodes, overlap_nodes);
    EXPECT_EQ(actual_nodes.size(), indexer.num_local_nodes());

    std::size_t partition_faces = 0;
    for (std::size_t local = 0; local < handle.num_faces(); ++local)
    {
        const auto lid = static_cast<LO>(local);
        const auto gid = static_cast<std::size_t>(handle.face_geometry_global_id(lid));
        EXPECT_LT(gid, native->num_faces());
        if (gid >= native->num_faces())
            continue;
        const auto face = native->indexer().face_id(gid);
        EXPECT_NEAR(handle.face_area(lid), native->face_area(face), 2e-13);
        expect_vector(handle.face_centroid(lid), native->face_centroid(face));
        expect_vector(handle.face_normal(lid), native->face_normal(face));
        const auto owner = handle.owner_cell(lid);
        const auto neighbor = handle.neighbor_cell(lid);
        const auto native_owner = native->indexer().cell_ordinal(native->owner_cell(face));
        if (owner != Handle::invalid_local_id())
            EXPECT_EQ(static_cast<std::size_t>(handle.cell_geometry_global_id(owner)), native_owner);
        const bool physically_exterior = native->neighbor_cell(face) == Semi::invalid_cell_id();
        EXPECT_EQ(handle.is_geometry_exterior_face(lid), physically_exterior);
        EXPECT_EQ(handle.is_boundary_face(lid), physically_exterior);
        EXPECT_EQ(handle.boundary_id(lid), native->boundary_id(face));
        if (!physically_exterior)
        {
            if (neighbor != Handle::invalid_local_id())
                EXPECT_EQ(static_cast<std::size_t>(handle.cell_geometry_global_id(neighbor)),
                    native->indexer().cell_ordinal(native->neighbor_cell(face)));
            if ((owner != Handle::invalid_local_id() && handle.is_owned_cell(owner)) !=
                (neighbor != Handle::invalid_local_id() && handle.is_owned_cell(neighbor)))
                ++partition_faces;
        }
        EXPECT_EQ(handle.is_owned_face(lid), owned.contains(native_owner));
        if (handle.is_owned_face(lid))
            ++face_owners[gid];
    }
    std::vector<int> global_cells(count), global_faces(native->num_faces());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(count), cell_owners.data(), global_cells.data());
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_SUM, static_cast<int>(native->num_faces()), face_owners.data(), global_faces.data());
    for (const auto owners : global_cells)
        EXPECT_EQ(owners, 1);
    for (const auto owners : global_faces)
        EXPECT_EQ(owners, 1);
    EXPECT_NEAR(global_sum(volume), expected_volume, 2e-12);
    const auto global_partition_faces = global_sum(static_cast<int>(partition_faces));
    const std::set<int> active_owners(owners.begin(), owners.end());
    if (active_owners.size() > 1)
        EXPECT_GT(global_partition_faces, 0);
    else
        EXPECT_EQ(global_partition_faces, 0);
    const auto native_boundary_ids = native->boundary_batch_ids();
    const std::set<int> expected_boundary_ids(native_boundary_ids.begin(), native_boundary_ids.end());
    std::set<int> actual_boundary_ids;
    for (const auto& [id, batch] : handle.boundary_batches())
    {
        actual_boundary_ids.insert(id);
        EXPECT_EQ(handle.boundary_batch_name(id), native->boundary_batch_name(id));
        for (const auto local : batch.face_lids)
        {
            EXPECT_TRUE(handle.is_geometry_exterior_face(local));
            EXPECT_EQ(handle.boundary_id(local), id);
        }
    }
    EXPECT_EQ(actual_boundary_ids, expected_boundary_ids);
    const auto output = handle.vtu_topology();
    EXPECT_EQ(output->cell_offsets.size(), owned_count);
    EXPECT_EQ(output->cell_types.size(), owned_count);
    std::set<GO> owned_nodes;
    for (const auto gid : owned)
    {
        const auto cell = native->indexer().cell_id(gid);
        for (const auto node : native->xy_cell_nodes()[cell.ij])
            for (const auto k : {cell.k, cell.k + 1})
                owned_nodes.insert(static_cast<GO>(native->indexer().node_ordinal({node, k})));
    }
    EXPECT_EQ(output->points.size(), owned_nodes.size());
    std::set<std::array<double, 3>> expected_points, actual_points;
    for (const auto node : owned_nodes)
    {
        const auto coordinate = native->node_coordinates(native->indexer().node_id(static_cast<std::size_t>(node)));
        expected_points.insert({coordinate.x, coordinate.y, coordinate.z});
    }
    for (const auto& point : output->points)
        actual_points.insert({point.x, point.y, point.z});
    EXPECT_EQ(actual_points, expected_points);
    for (const auto node : output->connectivity)
    {
        EXPECT_GE(node, 0);
        EXPECT_LT(static_cast<std::size_t>(node), output->points.size());
    }
    std::size_t vtk_cell = 0, connectivity_begin = 0;
    for (const auto gid : owned_order)
    {
        const auto cell = native->indexer().cell_id(gid);
        const auto& nodes = native->xy_cell_nodes()[cell.ij];
        const auto sides = nodes.size();
        EXPECT_EQ(output->cell_types[vtk_cell], sides == 3 ? 13 : sides == 4 ? 12 : 42);
        EXPECT_EQ(output->cell_offsets[vtk_cell], connectivity_begin + 2 * sides);
        for (std::size_t corner = 0; corner < 2 * sides; ++corner)
        {
            const auto entry = connectivity_begin + corner;
            if (entry >= output->connectivity.size())
                continue;
            const auto point = output->connectivity[entry];
            if (point < 0 || static_cast<std::size_t>(point) >= output->points.size())
                continue;
            expect_vector(output->points[point], native->node_coordinates(Semi::NodeID{nodes[corner % sides],
                                                     cell.k + static_cast<unsigned>(corner / sides)}));
        }
        connectivity_begin += 2 * sides;
        ++vtk_cell;
    }
}
} // namespace

TEST(SemiStructuredDistributedTest, UnevenPolygonTriangleNonuniformZMatchesNativeOracle)
{
    verify_partition(mixed_mesh(), 1);
}

TEST(SemiStructuredDistributedTest, TwoGhostLayersFollowTopologyBreadthFirst)
{
    verify_partition(mixed_mesh(), 2);
}

TEST(SemiStructuredDistributedTest, SingleLayerPartitionsAcrossXYAndHasActualEmptyOwners)
{
    verify_partition(mixed_mesh(true), 1);
    const Handle handle(mixed_mesh(true), {.allow_empty_partitions = true});
    const auto owners = global_cell_owners(handle, *mixed_mesh(true));
    (void) verify_hybrid_ownership(handle, *mixed_mesh(true), owners);
}

TEST(SemiStructuredDistributedTest, MoreRanksThanOneCellRetainsEmptyMapsAndOwnedOnlyOutput)
{
    const auto native = one_triangle();
    const Handle handle(native, {.ghost_layers = 2, .allow_empty_partitions = true});
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    EXPECT_EQ(global_sum(handle.num_owned_cells() == 0 ? 1 : 0), ranks - 1);
    EXPECT_EQ(global_sum(static_cast<int>(handle.num_owned_cells())), 1);
    EXPECT_EQ(global_sum(static_cast<int>(handle.num_owned_faces())), 5);
    verify_algebraic_maps(handle, *native);
    EXPECT_NEAR(global_sum(handle.num_owned_cells() ? handle.cell_volume(0) : 0.), 3., 2e-13);
    const auto output = handle.vtu_topology();
    EXPECT_EQ(output->cell_offsets.size(), handle.num_owned_cells());
    EXPECT_EQ(output->points.size(), handle.num_owned_cells() ? 6U : 0U);
    const auto native_boundary_ids = native->boundary_batch_ids();
    const std::set<int> expected_boundary_ids(native_boundary_ids.begin(), native_boundary_ids.end());
    std::set<int> actual_boundary_ids;
    for (const auto& [id, batch] : handle.boundary_batches())
    {
        actual_boundary_ids.insert(id);
        EXPECT_EQ(handle.boundary_batch_name(id), native->boundary_batch_name(id));
        EXPECT_EQ(batch.id, id);
        if (handle.num_owned_cells() == 0)
            EXPECT_TRUE(batch.face_lids.empty());
    }
    EXPECT_EQ(actual_boundary_ids, expected_boundary_ids);
    if (handle.num_owned_cells() == 0)
    {
        EXPECT_EQ(handle.num_local_cells(), 0U);
        EXPECT_EQ(handle.num_faces(), 0U);
        EXPECT_EQ(handle.indexer().num_local_nodes(), 0U);
        EXPECT_TRUE(
            handle.boundary_batches().empty() || std::ranges::all_of(handle.boundary_batches(),
                                                     [](const auto& entry) { return entry.second.face_lids.empty(); }));
    }
}

TEST(SemiStructuredDistributedTest, MutableCopiesPreserveConcreteIdentityAndConstObserverPolicy)
{
    const auto native = mixed_mesh();
    Handle handle(native, {.ghost_layers = 2});
    Handle copy = handle;
    EXPECT_TRUE(handle.has_mutable_geometry());
    EXPECT_TRUE(copy.has_mutable_geometry());
    EXPECT_EQ(handle.owned_cell_map().get(), copy.owned_cell_map().get());
    EXPECT_EQ(handle.visit([](const auto& mesh) -> const void* { return &mesh; }), native.get());
    EXPECT_EQ(copy.visit_mutable([](auto& mesh) -> const void* { return &mesh; }), native.get());
    EXPECT_EQ(handle.geometry_epoch(), copy.geometry_epoch());
    Handle observer(Handle::SemiStructuredPtr(native), {.ghost_layers = 2});
    EXPECT_FALSE(observer.has_mutable_geometry());
    EXPECT_THROW(observer.visit_mutable([](auto&) {}), std::logic_error);
}

TEST(SemiStructuredDistributedTest, InvalidAndRankAsymmetricOptionsRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto native = mixed_mesh();
    EXPECT_THROW((void) Handle(native, {.ghost_layers = 0}), std::invalid_argument);
    EXPECT_THROW(
        (void) Handle(native, {.partition = static_cast<std::size_t>(comm->getRank())}), std::invalid_argument);
    EXPECT_THROW((void) Handle(native, {.partition = static_cast<std::size_t>(comm->getRank()),
                                           .partitions = static_cast<std::size_t>(comm->getSize() + 1)}),
        std::invalid_argument);
    EXPECT_NO_THROW((void) Handle(native, {.partition = static_cast<std::size_t>(comm->getRank()),
                                              .partitions = static_cast<std::size_t>(comm->getSize()),
                                              .allow_empty_partitions = true}));
    if (comm->getSize() > 1)
    {
        EXPECT_THROW((void) Handle(native, {.ghost_layers = comm->getRank() == 0 ? 1U : 2U}), std::invalid_argument);
        EXPECT_THROW((void) Handle(native, {.allow_empty_partitions = comm->getRank() == 0}), std::invalid_argument);
        EXPECT_THROW((void) Handle(one_triangle()), std::invalid_argument);
        EXPECT_THROW(
            (void) Handle(comm->getRank() == 0 ? mixed_mesh() : mixed_mesh(false, .125)), std::invalid_argument);
        EXPECT_THROW((void) Handle(comm->getRank() == 0 ? mixed_mesh() : mixed_mesh(true)), std::invalid_argument);
        const std::shared_ptr<Semi> maybe_null = comm->getRank() == 0 ? nullptr : native;
        EXPECT_THROW((void) Handle(maybe_null), std::invalid_argument);
    }
}

/** Real graph cuts must beat a deliberately hostile contiguous-ordinal decomposition. */
TEST(SemiStructuredDistributedTest, ShuffledXYGraphCutsAndFactoryReorderingPreservePhysicalColumns)
{
    const auto native = shuffled_chain();
    const auto comm = Tpetra::getDefaultComm();
    const auto ranks = static_cast<std::size_t>(comm->getSize());
    Handle::DistributionOptions options{.ghost_layers = 2, .xy_partitions = ranks, .z_partitions = 1};
    verify_partition(native, 2, options, 60.);
    const Handle handle(native, options);
    verify_algebraic_maps(handle, *native);
    const auto owners = global_cell_owners(handle, *native);
    const auto xy_owners = verify_hybrid_ownership(handle, *native, owners);
    const std::size_t nxy = native->indexer().num_cells_per_layer;
    if (ranks > 1)
    {
        std::size_t cut = 0, flat_cut = 0;
        const auto flat_owner = [&](std::size_t xy)
        {
            const auto width = nxy / ranks, remainder = nxy % ranks, wide_end = (width + 1) * remainder;
            return xy < wide_end ? xy / (width + 1) : remainder + (xy - wide_end) / width;
        };
        for (const auto& edge : native->topology().side_faces())
        {
            if (edge.neighbor == static_cast<unsigned>(-1))
                continue;
            cut += xy_owners[edge.owner] != xy_owners[edge.neighbor];
            flat_cut += flat_owner(edge.owner) != flat_owner(edge.neighbor);
        }
        EXPECT_EQ(flat_cut, nxy - 1);
        EXPECT_LT(cut, flat_cut);
        EXPECT_LE(cut, 2 * (ranks - 1));
        std::vector<GO> owned_order;
        for (std::size_t row = 0; row < handle.num_owned_cells(); ++row)
            owned_order.push_back(handle.cell_geometry_global_id(static_cast<LO>(row)));
        EXPECT_GT(global_sum(!std::ranges::is_sorted(owned_order) ? 1 : 0), 0);
        EXPECT_GT(global_sum(handle.has_reordered_cells() ? 1 : 0), 0);
        int renumbered_cells = 0, renumbered_faces = 0;
        for (std::size_t row = 0; row < handle.num_owned_cells(); ++row)
            renumbered_cells +=
                handle.cell_global_id(static_cast<LO>(row)) != handle.cell_geometry_global_id(static_cast<LO>(row));
        for (std::size_t row = 0; row < handle.num_owned_faces(); ++row)
            renumbered_faces +=
                handle.face_global_id(static_cast<LO>(row)) != handle.face_geometry_global_id(static_cast<LO>(row));
        EXPECT_GT(global_sum(renumbered_cells), 0);
        EXPECT_GT(global_sum(renumbered_faces), 0);
    }
}

/** Factor overrides retain the same column graph and infer only the missing tensor factor. */
TEST(SemiStructuredDistributedTest, HybridFactorOverridesInferBothWaysAndRetainContiguousZSlabs)
{
    const auto native = shuffled_chain();
    const auto ranks = static_cast<std::size_t>(Tpetra::getDefaultComm()->getSize());
    std::set<std::size_t> xy_factors{1, ranks};
    if (ranks % 2 == 0)
        xy_factors.insert(2);
    for (const auto px : xy_factors)
    {
        SCOPED_TRACE(px);
        const auto pz = ranks / px;
        const Handle explicit_factors(
            native, {.allow_empty_partitions = true, .xy_partitions = px, .z_partitions = pz});
        const Handle infer_z(native, {.allow_empty_partitions = true, .xy_partitions = px});
        const Handle infer_xy(native, {.allow_empty_partitions = true, .z_partitions = pz});
        EXPECT_EQ(explicit_factors.semi_structured_decomposition().xy_partitions, px);
        EXPECT_EQ(explicit_factors.semi_structured_decomposition().z_partitions, pz);
        EXPECT_EQ(infer_z.semi_structured_decomposition().z_partitions, pz);
        EXPECT_EQ(infer_xy.semi_structured_decomposition().xy_partitions, px);
        const auto owners = global_cell_owners(explicit_factors, *native);
        EXPECT_EQ(global_cell_owners(infer_z, *native), owners);
        EXPECT_EQ(global_cell_owners(infer_xy, *native), owners);
        (void) verify_hybrid_ownership(explicit_factors, *native, owners);
        verify_partition(native, 2, {.xy_partitions = px, .z_partitions = pz}, 60.);
    }
    const Handle automatic(native);
    const auto selected = automatic.semi_structured_decomposition();
    EXPECT_EQ(selected.xy_partitions * selected.z_partitions, ranks);
    if (ranks == 4)
    {
        EXPECT_EQ(selected.xy_partitions, 2U);
        EXPECT_EQ(selected.z_partitions, 2U);
    }
}

/** Explicit slab factors can require empty owners even when total cells exceed rank count. */
TEST(SemiStructuredDistributedTest, ProductFactorEmptyOwnersRequireExplicitPolicyAndRetainMetadata)
{
    const auto native = mixed_mesh(true);
    const auto ranks = static_cast<std::size_t>(Tpetra::getDefaultComm()->getSize());
    const Handle handle(native, {.allow_empty_partitions = true, .xy_partitions = 1, .z_partitions = ranks});
    const auto owners = global_cell_owners(handle, *native);
    (void) verify_hybrid_ownership(handle, *native, owners);
    EXPECT_EQ(global_sum(handle.num_owned_cells() == 0 ? 1 : 0), static_cast<int>(ranks - 1));
    verify_partition(native, 1, {.xy_partitions = 1, .z_partitions = ranks});
    if (ranks > 1)
        EXPECT_THROW((void) Handle(native, {.xy_partitions = 1, .z_partitions = ranks}), std::invalid_argument);
}

/** Repeated construction must preserve label canonicalization and XY-order/map identity by value. */
TEST(SemiStructuredDistributedTest, RepeatedHybridConstructionHasDeterministicMapsAndCanonicalFaces)
{
    const auto native = shuffled_chain();
    const Handle first(native, {.ghost_layers = 2});
    const Handle second(native, {.ghost_layers = 2});
    EXPECT_EQ(
        first.semi_structured_decomposition().xy_partitions, second.semi_structured_decomposition().xy_partitions);
    EXPECT_EQ(first.semi_structured_decomposition().z_partitions, second.semi_structured_decomposition().z_partitions);
    EXPECT_EQ(first.num_owned_cells(), second.num_owned_cells());
    EXPECT_EQ(first.num_local_cells(), second.num_local_cells());
    EXPECT_EQ(first.num_faces(), second.num_faces());
    EXPECT_EQ(first.num_owned_faces(), second.num_owned_faces());
    for (std::size_t row = 0; row < first.num_local_cells(); ++row)
    {
        EXPECT_EQ(
            first.cell_geometry_global_id(static_cast<LO>(row)), second.cell_geometry_global_id(static_cast<LO>(row)));
        EXPECT_EQ(first.cell_global_id(static_cast<LO>(row)), second.cell_global_id(static_cast<LO>(row)));
    }
    for (std::size_t row = 0; row < first.num_faces(); ++row)
    {
        EXPECT_EQ(
            first.face_geometry_global_id(static_cast<LO>(row)), second.face_geometry_global_id(static_cast<LO>(row)));
        EXPECT_EQ(first.face_global_id(static_cast<LO>(row)), second.face_global_id(static_cast<LO>(row)));
    }
    EXPECT_EQ(global_cell_owners(first, *native), global_cell_owners(second, *native));
    EXPECT_EQ(first.visit([](const auto& geometry) -> const void* { return &geometry; }), native.get());
}

TEST(SemiStructuredDistributedTest, InvalidOrRankDivergentHybridFactorsRejectBeforeMaps)
{
    const auto native = shuffled_chain();
    const auto comm = Tpetra::getDefaultComm();
    const auto ranks = static_cast<std::size_t>(comm->getSize());
    EXPECT_THROW((void) Handle(native, {.xy_partitions = 0}), std::invalid_argument);
    EXPECT_THROW((void) Handle(native, {.z_partitions = 0}), std::invalid_argument);
    EXPECT_THROW((void) Handle(native, {.xy_partitions = ranks + 1}), std::invalid_argument);
    EXPECT_THROW((void) Handle(native, {.z_partitions = ranks + 1}), std::invalid_argument);
    if (ranks > 1)
    {
        EXPECT_THROW((void) Handle(native, {.xy_partitions = ranks, .z_partitions = ranks}), std::invalid_argument);
        EXPECT_THROW(
            (void) Handle(native, {.allow_empty_partitions = true, .xy_partitions = comm->getRank() == 0 ? 1U : ranks}),
            std::invalid_argument);
        Handle::DistributionOptions options;
        options.allow_empty_partitions = true;
        options.xy_partitions = 1;
        if (comm->getRank() != 0)
            options.z_partitions = ranks;
        EXPECT_THROW((void) Handle(native, options), std::invalid_argument);
    }
}
