/** Independent geometry and algebraic-ID oracles for orthogonal MPI decomposition. */
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/DecomposedOrthogonalTopology.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <numbers>
#include <set>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
void expect_vector(const auto& a, const auto& b)
{
    EXPECT_NEAR(a.x, b.x, 2e-12);
    EXPECT_NEAR(a.y, b.y, 2e-12);
    EXPECT_NEAR(a.z, b.z, 2e-12);
}
/** Follow the source graph, including periodic wraps, independently of the partition. */
template<class Native> auto halo(const Native& native, std::set<std::size_t> result, std::size_t depth)
{
    auto frontier = result;
    for (std::size_t layer = 0; layer < depth; ++layer)
    {
        std::set<std::size_t> next;
        for (const auto source : frontier)
        {
            const auto cell = native.indexer().cell_id(source);
            for (const auto face : native.faces(cell))
            {
                const auto other = native.opposite_cell(face, cell);
                if (other != Native::invalid_cell_id())
                {
                    const auto id = native.indexer().cell_ordinal(other);
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
template<class Native> void verify(const std::shared_ptr<Native>& native, std::size_t depth, bool explicit_grid = false)
{
    Handle::DistributionOptions options;
    options.ghost_layers = depth;
    options.allow_empty_partitions = true;
    const auto processes = std::size_t(Tpetra::getDefaultComm()->getSize());
    if (explicit_grid && (processes == 4 || processes == 8))
        options.orthogonal_partitions = std::array<std::size_t, 3>{2, 2, processes / 4};
    const Handle mesh(native, options);
    if (options.orthogonal_partitions)
        EXPECT_EQ(mesh.orthogonal_decomposition()->factors(), *options.orthogonal_partitions);
    EXPECT_FALSE(mesh.has_materialized_indexer());
    EXPECT_FALSE(mesh.has_materialized_connectivity());
    const auto compact = mesh.decomposed_orthogonal_topology();
    if (processes == 1)
        EXPECT_FALSE(compact);
    else
    {
        ASSERT_TRUE(compact);
        EXPECT_EQ(&compact->source_topology(), &native->topology());
        EXPECT_EQ(compact->num_owned_cells(), mesh.num_owned_cells());
        EXPECT_EQ(compact->num_local_cells(), mesh.num_local_cells());
        EXPECT_EQ(compact->num_owned_faces(), mesh.num_owned_faces());
        EXPECT_EQ(compact->num_local_faces(), mesh.num_faces());
        EXPECT_EQ(compact->lookup_storage_estimate(), 0U);
        for (std::size_t c = 0; c < mesh.num_local_cells(); ++c)
        {
            const auto source = compact->source_cell(c);
            EXPECT_EQ(compact->cell_local_id(source), c);
            EXPECT_EQ(native->indexer().cell_ordinal(source), std::size_t(mesh.cell_geometry_global_id(LO(c))));
            std::set<std::size_t> direct, observed;
            for (const auto f : compact->cell_faces(c))
                direct.insert(f);
            for (const auto f : mesh.faces(LO(c)))
                observed.insert(std::size_t(f));
            EXPECT_EQ(direct, observed);
        }
        for (std::size_t f = 0; f < mesh.num_faces(); ++f)
        {
            const auto source = compact->source_face(f);
            EXPECT_EQ(compact->face_local_id(source), f);
            EXPECT_EQ(native->indexer().face_ordinal(source), std::size_t(mesh.face_geometry_global_id(LO(f))));
            const auto translate = [](std::size_t value)
            {
                return value == Meshes::DecomposedOrthogonalTopology::invalid_local_id() ? Handle::invalid_local_id()
                                                                                         : LO(value);
            };
            EXPECT_EQ(translate(compact->owner_cell(f)), mesh.owner_cell(LO(f)));
            EXPECT_EQ(translate(compact->neighbor_cell(f)), mesh.neighbor_cell(LO(f)));
            EXPECT_EQ(compact->boundary_id(f), mesh.boundary_id(LO(f)));
        }
        std::set<std::size_t> source_nodes;
        for (std::size_t n = 0; n < compact->num_local_nodes(); ++n)
        {
            const auto source = compact->source_node(n);
            EXPECT_EQ(compact->node_local_id(source), n);
            EXPECT_TRUE(source_nodes.insert(native->indexer().node_ordinal(source)).second);
        }
    }
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = std::size_t(comm->getRank()), ranks = std::size_t(comm->getSize());
    std::vector<GO> local_counts(2 * ranks), counts(2 * ranks);
    local_counts[2 * rank] = GO(mesh.num_owned_cells());
    local_counts[2 * rank + 1] = GO(mesh.num_owned_faces());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(counts.size()), local_counts.data(), counts.data());
    GO cell_prefix = 0, face_prefix = 0;
    for (std::size_t r = 0; r < rank; ++r)
    {
        cell_prefix += counts[2 * r];
        face_prefix += counts[2 * r + 1];
    }
    EXPECT_TRUE(mesh.owned_cell_map()->isContiguous());
    EXPECT_TRUE(mesh.owned_face_map()->isContiguous());
    EXPECT_TRUE(mesh.owned_cell_map()->isOneToOne());
    EXPECT_TRUE(mesh.owned_face_map()->isOneToOne());
    EXPECT_EQ(mesh.owned_cell_map()->getGlobalNumElements(), native->num_cells());
    EXPECT_EQ(mesh.owned_face_map()->getGlobalNumElements(), native->num_faces());
    std::vector<int> local_owners(native->num_cells(), -1), owners(native->num_cells());
    std::vector<int> local_multiplicity(native->num_cells()), multiplicity(native->num_cells());
    std::vector<GO> local_cells(native->num_cells(), -1), cells(native->num_cells());
    std::vector<GO> local_faces(native->num_faces(), -1), faces(native->num_faces());
    std::set<std::size_t> owned;
    for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto lid = LO(c);
        const auto source = std::size_t(mesh.cell_geometry_global_id(lid));
        ASSERT_LT(source, native->num_cells());
        owned.insert(source);
        local_owners[source] = int(rank);
        local_multiplicity[source] = 1;
        local_cells[source] = mesh.cell_global_id(lid);
        EXPECT_EQ(mesh.cell_global_id(lid), cell_prefix + GO(c));
    }
    for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto lid = LO(f);
        const auto source = std::size_t(mesh.face_geometry_global_id(lid));
        ASSERT_LT(source, native->num_faces());
        local_faces[source] = mesh.face_global_id(lid);
        EXPECT_EQ(mesh.face_global_id(lid), face_prefix + GO(f));
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(owners.size()), local_owners.data(), owners.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(owners.size()), local_multiplicity.data(), multiplicity.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(cells.size()), local_cells.data(), cells.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(faces.size()), local_faces.data(), faces.data());
    for (const auto count : multiplicity)
        EXPECT_EQ(count, 1);
    const auto decomposition = mesh.orthogonal_decomposition();
    const auto factors = decomposition->factors();
    for (std::size_t source = 0; source < owners.size(); ++source)
    {
        const auto cell = native->indexer().cell_id(source);
        const std::array<std::size_t, 3> coordinate{std::size_t(cell.i), std::size_t(cell.j), std::size_t(cell.k)};
        std::array<std::size_t, 3> block{};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto& cuts = decomposition->axis_cuts(axis);
            block[axis] = std::upper_bound(cuts.begin(), cuts.end(), coordinate[axis]) - cuts.begin() - 1;
        }
        EXPECT_EQ(owners[source], int(block[0] + factors[0] * (block[1] + factors[1] * block[2])));
    }
    std::set<std::size_t> overlap, expected_faces;
    const auto expected = halo(*native, owned, depth);
    for (const auto source : expected)
        for (const auto face : native->faces(native->indexer().cell_id(source)))
            expected_faces.insert(native->indexer().face_ordinal(face));
    for (std::size_t c = 0; c < mesh.num_local_cells(); ++c)
    {
        const auto lid = LO(c);
        const auto source = std::size_t(mesh.cell_geometry_global_id(lid));
        ASSERT_LT(source, native->num_cells());
        overlap.insert(source);
        EXPECT_EQ(mesh.cell_global_id(lid), cells[source]);
        EXPECT_EQ(mesh.overlap_cell_map()->getGlobalElement(lid), cells[source]);
        EXPECT_EQ(mesh.overlap_cell_map()->getLocalElement(cells[source]), lid);
        const auto cell = native->indexer().cell_id(source);
        expect_vector(mesh.cell_centroid(lid), native->cell_centroid(cell));
        EXPECT_NEAR(mesh.cell_volume(lid), native->cell_volume(cell), 2e-12);
        std::set<std::size_t> actual, incident;
        for (const auto face : native->faces(cell))
            incident.insert(native->indexer().face_ordinal(face));
        for (const auto face : mesh.faces(lid))
        {
            const auto source_face = std::size_t(mesh.face_geometry_global_id(face));
            actual.insert(source_face);
            EXPECT_TRUE(incident.contains(source_face));
            expect_vector(mesh.face_area_vector_outward(face, lid),
                native->face_area_vector_outward(native->indexer().face_id(source_face), cell));
        }
        if (c < mesh.num_owned_cells())
            EXPECT_EQ(actual, incident);
    }
    EXPECT_EQ(overlap, expected);
    std::set<std::size_t> actual_faces;
    for (std::size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto lid = LO(f);
        const auto source = std::size_t(mesh.face_geometry_global_id(lid));
        ASSERT_LT(source, native->num_faces());
        actual_faces.insert(source);
        EXPECT_EQ(mesh.face_global_id(lid), faces[source]);
        EXPECT_EQ(mesh.overlap_face_map()->getGlobalElement(lid), faces[source]);
        const auto face = native->indexer().face_id(source);
        expect_vector(mesh.face_centroid(lid), native->face_centroid(face));
        expect_vector(mesh.face_area_vector(lid), native->face_area_vector(face));
        EXPECT_EQ(
            mesh.is_owned_face(lid), owners[native->indexer().cell_ordinal(native->owner_cell(face))] == int(rank));
        EXPECT_EQ(mesh.boundary_id(lid), native->boundary_id(face));
        EXPECT_EQ(mesh.is_geometry_exterior_face(lid), native->neighbor_cell(face) == Native::invalid_cell_id());
    }
    EXPECT_EQ(actual_faces, expected_faces);
    EXPECT_FALSE(mesh.has_materialized_indexer());
    EXPECT_FALSE(mesh.has_materialized_connectivity());
    if (owned.empty())
    {
        EXPECT_EQ(mesh.num_local_cells(), 0U);
        EXPECT_EQ(mesh.num_faces(), 0U);
    }
}
} // namespace
TEST(OrthogonalDistributedTest, NonuniformCartesianExactGeometryAndPeriodicHalos)
{
    const Vec3D<ArrReal> edges{{{0., .1, .8, 1.4, 2.}, {0., .3, .9, 1.6, 2.}, {0., .2, .7, 1.5, 2.}}};
    for (const bool periodic : {false, true})
        for (const std::size_t depth : {1U, 2U})
            verify(std::make_shared<Handle::Cartesian>(edges, Vec3D<bool>{periodic, periodic, periodic}), depth, true);
}
TEST(OrthogonalDistributedTest, NonuniformCylindricalExactGeometryAndThetaPeriodicHalos)
{
    for (const double angle : {1.3, 2 * std::numbers::pi})
        for (const std::size_t depth : {1U, 2U})
        {
            const Vec3D<ArrReal> edges{
                {{1., 1.1, 1.6, 2.1, 3.}, {0., .1 * angle, .4 * angle, .7 * angle, angle}, {0., .2, .7, 1.5, 2.}}};
            verify(std::make_shared<Handle::Cylindrical>(edges), depth, true);
        }
}
TEST(OrthogonalDistributedTest, EmptyOwnersRetainCollectiveMaps)
{
    const Vec3D<ArrReal> edges{{{1., 2.}, {0., .5}, {0., 1.}}};
    verify(std::make_shared<Handle::Cartesian>(edges), 1);
    verify(std::make_shared<Handle::Cylindrical>(edges), 1);
}
TEST(OrthogonalDistributedTest, AutomaticGridBalancesCountsBeforePhysicalAspect)
{
    const auto ranks = std::size_t(Tpetra::getDefaultComm()->getSize());
    if (ranks != 4 && ranks != 8)
        GTEST_SKIP() << "Requires four or eight MPI ranks.";
    const ArrReal regular{0., .25, .5, .75, 1.};
    auto native = std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{regular, regular, regular}});
    const Handle cube(native);
    auto factors = cube.orthogonal_decomposition()->factors();
    std::ranges::sort(factors);
    EXPECT_EQ(factors, (ranks == 4 ? std::array<std::size_t, 3>{1, 2, 2} : std::array<std::size_t, 3>{2, 2, 2}));
    if (ranks == 4)
    {
        const Handle long_box(
            std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{{0., 4., 8., 12., 16.}, regular, regular}}));
        EXPECT_EQ(long_box.orthogonal_decomposition()->factors(), (std::array<std::size_t, 3>{4, 1, 1}));
        const Handle count_first(
            std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{{0., 20., 40., 60., 80., 100.}, regular, regular}}));
        EXPECT_EQ(count_first.orthogonal_decomposition()->factors(), (std::array<std::size_t, 3>{1, 2, 2}));
    }
}

TEST(OrthogonalDistributedTest, InvalidAndRankDivergentMetadataFailCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const Vec3D<ArrReal> edges{{{0., .2, .7, 1., 2.}, {0., .3, .8, 1.2, 2.}, {0., .1, .4, .9, 2.}}};
    auto native = std::make_shared<Handle::Cartesian>(edges);
    Handle::DistributionOptions options;
    options.orthogonal_partitions = std::array<std::size_t, 3>{0, 1, std::size_t(comm->getSize())};
    EXPECT_THROW((Handle(native, options)), std::invalid_argument);
    options.orthogonal_partitions = std::array<std::size_t, 3>{1, 1, std::size_t(comm->getSize()) + 1};
    EXPECT_THROW((Handle(native, options)), std::invalid_argument);
    if (comm->getSize() == 1)
        return;
    auto changed = edges;
    if (comm->getRank() == 0)
        changed[0][1] += .01;
    EXPECT_THROW((Handle(std::make_shared<Handle::Cartesian>(changed))), std::invalid_argument);
    EXPECT_THROW((Handle(std::make_shared<Handle::Cartesian>(edges, Vec3D<bool>{comm->getRank() == 0, false, false}))),
        std::invalid_argument);
    options = {};
    options.allow_empty_partitions = true;
    if (comm->getRank() == 0)
        options.ghost_layers = 2;
    EXPECT_THROW((Handle(native, options)), std::invalid_argument);
}
TEST(OrthogonalDistributedTest, PhysicalWidthsBreakBalancedCutLocationTies)
{
    if (Tpetra::getDefaultComm()->getSize() != 2)
        GTEST_SKIP() << "Requires two MPI ranks.";
    Handle::DistributionOptions options;
    options.orthogonal_partitions = std::array<std::size_t, 3>{2, 1, 1};
    const Handle right_heavy(
        std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{{0., .1, .2, .3, .4, 10.}, {0., 1.}, {0., 1.}}}), options);
    const Handle left_heavy(
        std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{{0., 9.6, 9.7, 9.8, 9.9, 10.}, {0., 1.}, {0., 1.}}}),
        options);
    EXPECT_EQ(right_heavy.orthogonal_decomposition()->axis_cuts(0), (std::vector<std::size_t>{0, 3, 5}));
    EXPECT_EQ(left_heavy.orthogonal_decomposition()->axis_cuts(0), (std::vector<std::size_t>{0, 2, 5}));
}

TEST(OrthogonalDistributedTest, RankLocalNullGeometryFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const Vec3D<ArrReal> edges{{{1., 1.2, 1.7, 2.1, 3.}, {0., .3, .8, 1.2, 2.}, {0., .1, .4, .9, 2.}}};
    Handle::CartesianPtr cartesian;
    Handle::CylindricalPtr cylindrical;
    if (comm->getRank() != 0)
    {
        cartesian = std::make_shared<Handle::Cartesian>(edges);
        cylindrical = std::make_shared<Handle::Cylindrical>(edges);
    }
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    // Every rank must observe the failure before any rank enters map construction.
    // In serial these calls exercise the ordinary null-pointer rejection too.
    EXPECT_THROW((Handle(cartesian, options)), std::invalid_argument);
    EXPECT_THROW((Handle(cylindrical, options)), std::invalid_argument);
}

TEST(OrthogonalDistributedTest, CompactTopologyStorageDoesNotScaleWithOwnedVolume)
{
    const auto ranks = std::size_t(Tpetra::getDefaultComm()->getSize());
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    const auto partitions = std::max<std::size_t>(ranks, 2);
    options.orthogonal_partitions = std::array<std::size_t, 3>{partitions, 1, 1};
    if (ranks == 1)
    {
        options.partition = 0;
        options.partitions = partitions;
    }
    const auto make = [&](std::size_t scale)
    {
        Vec3D<ArrReal> edges;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto cells = scale * (axis == 0 ? partitions : 2);
            for (std::size_t edge = 0; edge <= cells; ++edge)
                edges[axis].push_back(double(edge) / double(cells));
        }
        return std::make_shared<Handle>(std::make_shared<Handle::Cartesian>(edges), options);
    };
    const auto small = make(4), large = make(16);
    const auto small_topology = small->decomposed_orthogonal_topology();
    const auto large_topology = large->decomposed_orthogonal_topology();
    ASSERT_TRUE(small_topology);
    ASSERT_TRUE(large_topology);
    EXPECT_GT(large->num_owned_cells(), 32 * small->num_owned_cells());
    EXPECT_LE(large_topology->storage_bytes(), 2 * small_topology->storage_bytes());
    EXPECT_LT(double(large_topology->storage_bytes()) / double(large->num_owned_cells()),
        double(small_topology->storage_bytes()) / double(small->num_owned_cells()));
    for (const auto& mesh : {small, large})
    {
        EXPECT_EQ(mesh->decomposed_orthogonal_topology()->lookup_storage_estimate(), 0U);
        EXPECT_EQ(mesh->storage_report().indexing, 0U);
        EXPECT_EQ(mesh->storage_report().indexing_estimate, 0U);
        EXPECT_FALSE(mesh->has_materialized_indexer());
        EXPECT_FALSE(mesh->has_materialized_connectivity());
    }
}

TEST(OrthogonalDistributedTest, CompactViewRetainsNativeGeometryAcrossAxialTrialAndRollback)
{
    const auto check = []<class Native>(const std::shared_ptr<Native>& native)
    {
        Handle::DistributionOptions options;
        options.allow_empty_partitions = true;
        auto mesh = std::make_shared<Handle>(native, options);
        const auto compact = mesh->decomposed_orthogonal_topology();
        if (compact)
            EXPECT_EQ(&compact->source_topology(), &native->topology());
        const auto owned_cells = mesh->owned_cell_map(), owned_faces = mesh->owned_face_map();
        std::vector<double> volumes;
        std::vector<Handle::Vec3> centers;
        std::vector<GO> cell_source, cell_algebraic, face_source, face_algebraic;
        for (std::size_t c = 0; c < mesh->num_local_cells(); ++c)
        {
            volumes.push_back(mesh->cell_volume(LO(c)));
            centers.push_back(mesh->cell_centroid(LO(c)));
            cell_source.push_back(mesh->cell_geometry_global_id(LO(c)));
            cell_algebraic.push_back(mesh->cell_global_id(LO(c)));
        }
        for (std::size_t f = 0; f < mesh->num_faces(); ++f)
        {
            face_source.push_back(mesh->face_geometry_global_id(LO(f)));
            face_algebraic.push_back(mesh->face_global_id(LO(f)));
        }
        PlanarALEMeshMotion<Pack> motion(mesh);
        motion.begin_trial(3., 1.);
        EXPECT_EQ(mesh->geometry_epoch(), 1U);
        EXPECT_EQ(native->geometry_epoch(), 1U);
        for (std::size_t c = 0; c < volumes.size(); ++c)
        {
            EXPECT_NEAR(mesh->cell_volume(LO(c)), 1.5 * volumes[c], 2e-12);
            auto expected = centers[c];
            expected.z *= 1.5;
            expect_vector(mesh->cell_centroid(LO(c)), expected);
        }
        EXPECT_EQ(mesh->decomposed_orthogonal_topology().get(), compact.get());
        EXPECT_EQ(mesh->owned_cell_map().get(), owned_cells.get());
        EXPECT_EQ(mesh->owned_face_map().get(), owned_faces.get());
        motion.rollback_trial();
        EXPECT_EQ(mesh->geometry_epoch(), 2U);
        EXPECT_EQ(native->geometry_epoch(), 2U);
        EXPECT_EQ(mesh->decomposed_orthogonal_topology().get(), compact.get());
        if (compact)
            EXPECT_EQ(&compact->source_topology(), &native->topology());
        for (std::size_t c = 0; c < volumes.size(); ++c)
        {
            EXPECT_NEAR(mesh->cell_volume(LO(c)), volumes[c], 2e-12);
            expect_vector(mesh->cell_centroid(LO(c)), centers[c]);
            EXPECT_EQ(mesh->cell_geometry_global_id(LO(c)), cell_source[c]);
            EXPECT_EQ(mesh->cell_global_id(LO(c)), cell_algebraic[c]);
        }
        for (std::size_t f = 0; f < face_source.size(); ++f)
        {
            EXPECT_EQ(mesh->face_geometry_global_id(LO(f)), face_source[f]);
            EXPECT_EQ(mesh->face_global_id(LO(f)), face_algebraic[f]);
            const auto source = native->indexer().face_id(std::size_t(face_source[f]));
            expect_vector(mesh->face_area_vector(LO(f)), native->face_area_vector(source));
        }
        EXPECT_FALSE(mesh->has_materialized_indexer());
        EXPECT_FALSE(mesh->has_materialized_connectivity());
    };
    const Vec3D<ArrReal> edges{{{1., 1.2, 1.7, 2.1, 3.}, {0., .3, .8, 1.2, 2.}, {0., .1, .4, .9, 2.}}};
    check(std::make_shared<Handle::Cartesian>(edges));
    check(std::make_shared<Handle::Cylindrical>(edges));
}