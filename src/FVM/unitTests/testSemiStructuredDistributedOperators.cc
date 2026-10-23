/** @file testSemiStructuredDistributedOperators.cc
 * @brief Native distributed XY/Z field, traversal and conservative FVM integration.
 */
#include "FVM/Operators.hh"
#include "geometry/DirectBatchRange.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshTraverse.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Native = Meshes::SemiStructuredXY_Z;
using Local = Pack::local_ordinal_type;
using Global = Pack::global_ordinal_type;
using Vec = Native::Vec3;
using Scalar = ScalarCellFieldStored<Pack>;
using Vector = VectorCellFieldStored<Pack>;
using Flux = ScalarFaceFieldStored<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

enum class Layout
{
    Automatic,
    GraphXY,
    ZSlabs,
    Hybrid
};
struct Scenario
{
    Layout layout;
    bool reordered, tiny, single_layer;
};
std::vector<Scenario> scenarios(bool include_single_layer = false)
{
    std::vector<Layout> layouts{Layout::Automatic, Layout::GraphXY, Layout::ZSlabs};
    const int ranks = Tpetra::getDefaultComm()->getSize();
    if (ranks >= 4 && ranks % 2 == 0)
        layouts.push_back(Layout::Hybrid);
    std::vector<Scenario> result;
    for (const auto layout : layouts)
        for (const bool single : {false, true})
        {
            if (single && !include_single_layer)
                continue;
            result.push_back({layout, false, false, single});
            result.push_back({layout, true, false, single});
            result.push_back({layout, false, true, single});
        }
    return result;
}

struct Polygon
{
    double area;
    Vec centroid;
};
struct Face
{
    Global owner, neighbor;
    Vec center, area_vector;
};

/** Input polygons and shoelace/edge metrics are independent of handle caches. */
struct Fixture
{
    Arr<Vec> points;
    Arr<Arr<unsigned>> polygons;
    ArrReal z;
    std::vector<Polygon> base;
    SP<Native> geometry;
    SP<const Handle> mesh;
    std::vector<Global> algebraic_cell_geometry, algebraic_face_geometry;
    std::vector<Global> geometry_cell_algebraic, geometry_face_algebraic;

    explicit Fixture(
        bool tiny = false, bool single_layer = false, Layout layout = Layout::Automatic, bool reordered = false)
    {
        Arr<Native::BoundaryEdge> boundaries;
        if (tiny)
        {
            points = {{0, 0, 0}, {1.2, .1, 0}, {1.4, 1.1, 0}, {.1, 1.2, 0}};
            polygons = {{0, 1, 2, 3}};
            z = {0, .7};
        }
        else
        {
            for (unsigned y = 0; y < 4; ++y)
                for (unsigned x = 0; x < 4; ++x)
                {
                    Vec p{double(x) + .18 * y + .04 * x * y, double(y) + .08 * x * x, 0};
                    if (x == 1 && y == 1)
                        p = p + Vec{.13, -.07, 0};
                    points.push_back(p);
                }
            for (unsigned y = 0; y < 3; ++y)
                for (unsigned x = 0; x < 3; ++x)
                {
                    const unsigned a = x + 4 * y, b = a + 1, c = b + 4, d = a + 4;
                    if (x == 0 && y == 0)
                    {
                        polygons.push_back({a, b, d});
                        polygons.push_back({b, c, d});
                    }
                    else
                        polygons.push_back({a, b, c, d});
                }
            for (unsigned i = 0; i < 3; ++i)
            {
                boundaries.push_back({i, i + 1, "south"});
                boundaries.push_back({12 + i, 13 + i, "north"});
                boundaries.push_back({4 * i, 4 * (i + 1), "west"});
                boundaries.push_back({3 + 4 * i, 3 + 4 * (i + 1), "east"});
            }
            z = single_layer ? ArrReal{0, .7} : ArrReal{0, .13, .62, 1.7};
        }
        if (reordered && !tiny)
        {
            // Interleave distant polygons: geometric graph neighborhoods are
            // deliberately different from contiguous canonical ID intervals.
            const auto original = polygons;
            const std::array<std::size_t, 10> order{0, 9, 1, 8, 2, 7, 3, 6, 4, 5};
            for (std::size_t i = 0; i < order.size(); ++i)
                polygons[i] = original[order[i]];
        }
        for (const auto& polygon : polygons)
        {
            double twice_area = 0;
            Vec moment{};
            for (std::size_t i = 0; i < polygon.size(); ++i)
            {
                const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]];
                const double cross = a.x * b.y - b.x * a.y;
                twice_area += cross;
                moment = moment + (a + b) * cross;
            }
            base.push_back({.5 * twice_area, moment / (3 * twice_area)});
        }
        geometry = std::make_shared<Native>(points, polygons, z, boundaries);
        Handle::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        distribution.ghost_layers = 1;
        const auto ranks = std::size_t(Tpetra::getDefaultComm()->getSize());
        if (layout == Layout::GraphXY)
        {
            distribution.xy_partitions = ranks;
            distribution.z_partitions = 1;
        }
        else if (layout == Layout::ZSlabs)
        {
            distribution.xy_partitions = 1;
            distribution.z_partitions = ranks;
        }
        else if (layout == Layout::Hybrid)
        {
            distribution.xy_partitions = 2;
            distribution.z_partitions = ranks / 2;
        }
        mesh = std::make_shared<Handle>(geometry, distribution);
        const auto comm = mesh->owned_cell_map()->getComm();
        const auto mapping = [&](bool cells)
        {
            const auto total = cells ? geometry->num_cells() : geometry->num_faces();
            const auto owned = cells ? mesh->num_owned_cells() : mesh->num_owned_faces();
            const auto map = cells ? mesh->owned_cell_map() : mesh->owned_face_map();
            std::vector<Global> local(total, -1), result(total, -1);
            int valid = 1, all_valid = 0;
            for (std::size_t i = 0; i < owned; ++i)
            {
                const auto algebraic = map->getGlobalElement(Local(i));
                const auto canonical =
                    cells ? mesh->cell_geometry_global_id(Local(i)) : mesh->face_geometry_global_id(Local(i));
                if (algebraic < 0 || std::size_t(algebraic) >= total || canonical < 0 ||
                    std::size_t(canonical) >= total)
                {
                    valid = 0;
                    continue;
                }
                local[std::size_t(algebraic)] = canonical;
            }
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &valid, &all_valid);
            if (!all_valid)
                throw std::runtime_error("Fixture encountered an invalid algebraic/canonical ID.");
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(total), local.data(), result.data());
            if (std::find(result.begin(), result.end(), Global{-1}) != result.end())
                throw std::runtime_error("Fixture algebraic ownership does not cover every canonical entity.");
            return result;
        };
        algebraic_cell_geometry = mapping(true);
        algebraic_face_geometry = mapping(false);
        geometry_cell_algebraic.resize(algebraic_cell_geometry.size());
        geometry_face_algebraic.resize(algebraic_face_geometry.size());
        for (std::size_t i = 0; i < algebraic_cell_geometry.size(); ++i)
            geometry_cell_algebraic[std::size_t(algebraic_cell_geometry[i])] = Global(i);
        for (std::size_t i = 0; i < algebraic_face_geometry.size(); ++i)
            geometry_face_algebraic[std::size_t(algebraic_face_geometry[i])] = Global(i);
    }

    Vec center(Global cell) const
    {
        const auto xy = std::size_t(cell) % base.size(), k = std::size_t(cell) / base.size();
        auto result = base[xy].centroid;
        result.z = .5 * (z[k] + z[k + 1]);
        return result;
    }
    double volume(Global cell) const
    {
        const auto xy = std::size_t(cell) % base.size(), k = std::size_t(cell) / base.size();
        return base[xy].area * (z[k + 1] - z[k]);
    }
    Face face(Global ordinal) const
    {
        // The indexer supplies identity only. Metric arithmetic uses raw inputs.
        const auto id = geometry->indexer().face_id(std::size_t(ordinal));
        if (id.orientation == Native::AXIAL)
        {
            const Global owner = id.ij + base.size() * (id.k == 0 ? 0 : id.k - 1);
            const Global neighbor = id.k == 0 || id.k + 1 == z.size() ? Global{-1} : Global(id.ij + base.size() * id.k);
            auto point = base[id.ij].centroid;
            point.z = z[id.k];
            return {owner, neighbor, point, {0, 0, (id.k == 0 ? -1 : 1) * base[id.ij].area}};
        }
        const auto& edge = geometry->topology().side_face(id.ij);
        const auto a = points[edge.nodes[0]], b = points[edge.nodes[1]];
        const double dz = z[id.k + 1] - z[id.k];
        const Global owner = edge.owner + base.size() * id.k;
        const Global neighbor = edge.neighbor == Native::Topology::invalid_ordinal
                                    ? Global{-1}
                                    : Global(edge.neighbor + base.size() * id.k);
        return {owner, neighbor, {.5 * (a.x + b.x), .5 * (a.y + b.y), .5 * (z[id.k] + z[id.k + 1])},
            {(b.y - a.y) * dz, -(b.x - a.x) * dz, 0}};
    }
    std::vector<Global> cell_faces(Global cell) const
    {
        std::vector<Global> result;
        const auto id = geometry->indexer().cell_id(std::size_t(cell));
        for (const auto f : geometry->topology().cell_faces(id))
            result.push_back(geometry->indexer().face_ordinal(f));
        return result;
    }
};

double analytic(Vec point)
{
    return 2 + 1.3 * point.x - .7 * point.y + .4 * point.z;
}
Vec analytic_vector(Vec point)
{
    return {analytic(point), 2 * point.x - point.z, -point.y + 3 * point.z};
}
double dot(Vec a, Vec b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void expect_vector(Vec actual, Vec expected, double tolerance = 2e-13)
{
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}
double global_sum(double value)
{
    double result = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &value, &result);
    return result;
}
void expect_native(const Fixture& fixture)
{
    EXPECT_FALSE(fixture.mesh->legacy_mesh());
    EXPECT_TRUE(fixture.mesh->visit(
        [&](const auto& provider)
        {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(provider)>, Native>)
                return &provider == fixture.geometry.get();
            else
                return false;
        }));
    EXPECT_EQ(fixture.mesh->connectivity_storage_bytes(), 0U);
}
void initialize(const Fixture& fixture, Scalar& scalar)
{
    for (std::size_t c = 0; c < fixture.mesh->num_owned_cells(); ++c)
        scalar.set_owned_value(Local(c), analytic(fixture.center(fixture.mesh->cell_geometry_global_id(Local(c)))));
    scalar.sync_ghosts();
}
auto matrix_row(const Fixture& fixture, const Pack::matrix_type& matrix, Local row)
{
    Pack::matrix_type::local_inds_host_view_type columns;
    Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(row, columns, values);
    std::map<Global, double> result;
    for (std::size_t i = 0; i < columns.size(); ++i)
        result[fixture.algebraic_cell_geometry.at(std::size_t(matrix.getColMap()->getGlobalElement(columns[i])))] +=
            values[i];
    return result;
}
void expect_algebraic_maps(const Fixture& fixture)
{
    const auto& mesh = *fixture.mesh;
    const auto comm = mesh.owned_cell_map()->getComm();
    const int rank = comm->getRank(), ranks = comm->getSize();
    std::vector<Global> local(2 * ranks), counts(2 * ranks);
    local[2 * rank] = Global(mesh.num_owned_cells());
    local[2 * rank + 1] = Global(mesh.num_owned_faces());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), counts.data());
    Global cell_offset = 0, face_offset = 0;
    for (int r = 0; r < rank; ++r)
    {
        cell_offset += counts[2 * r];
        face_offset += counts[2 * r + 1];
    }
    EXPECT_TRUE(mesh.owned_cell_map()->isContiguous());
    EXPECT_TRUE(mesh.owned_face_map()->isContiguous());
    EXPECT_TRUE(mesh.owned_cell_map()->isOneToOne());
    EXPECT_TRUE(mesh.owned_face_map()->isOneToOne());
    EXPECT_EQ(mesh.owned_cell_map()->getGlobalNumElements(), fixture.geometry->num_cells());
    EXPECT_EQ(mesh.owned_face_map()->getGlobalNumElements(), fixture.geometry->num_faces());
    auto canonical_cells = fixture.algebraic_cell_geometry, canonical_faces = fixture.algebraic_face_geometry;
    std::ranges::sort(canonical_cells);
    std::ranges::sort(canonical_faces);
    for (std::size_t i = 0; i < canonical_cells.size(); ++i)
        EXPECT_EQ(canonical_cells[i], Global(i));
    for (std::size_t i = 0; i < canonical_faces.size(); ++i)
        EXPECT_EQ(canonical_faces[i], Global(i));
    for (std::size_t i = 0; i < mesh.num_local_cells(); ++i)
    {
        const Global algebraic = mesh.overlap_cell_map()->getGlobalElement(Local(i));
        EXPECT_EQ(mesh.cell_global_id(Local(i)), algebraic);
        EXPECT_EQ(fixture.algebraic_cell_geometry.at(std::size_t(algebraic)), mesh.cell_geometry_global_id(Local(i)));
        EXPECT_EQ(mesh.overlap_cell_map()->getLocalElement(algebraic), Local(i));
        if (i < mesh.num_owned_cells())
        {
            EXPECT_EQ(algebraic, cell_offset + Global(i));
            EXPECT_EQ(mesh.owned_cell_map()->getGlobalElement(Local(i)), algebraic);
        }
    }
    for (std::size_t i = 0; i < mesh.num_faces(); ++i)
    {
        const Global algebraic = mesh.overlap_face_map()->getGlobalElement(Local(i));
        EXPECT_EQ(mesh.face_global_id(Local(i)), algebraic);
        EXPECT_EQ(fixture.algebraic_face_geometry.at(std::size_t(algebraic)), mesh.face_geometry_global_id(Local(i)));
        EXPECT_EQ(mesh.overlap_face_map()->getLocalElement(algebraic), Local(i));
        if (i < mesh.num_owned_faces())
        {
            EXPECT_EQ(algebraic, face_offset + Global(i));
            EXPECT_EQ(mesh.owned_face_map()->getGlobalElement(Local(i)), algebraic);
        }
    }
}

std::vector<int> owner_map(const Fixture& fixture)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto total = fixture.geometry->num_cells();
    std::vector<int> local(total, -1), owners(total), local_count(total), counts(total);
    for (std::size_t cell = 0; cell < fixture.mesh->num_owned_cells(); ++cell)
    {
        const auto gid = fixture.mesh->cell_geometry_global_id(Local(cell));
        local[std::size_t(gid)] = comm->getRank();
        local_count[std::size_t(gid)] = 1;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(total), local.data(), owners.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(total), local_count.data(), counts.data());
    for (std::size_t cell = 0; cell < total; ++cell)
    {
        EXPECT_EQ(counts[cell], 1);
        EXPECT_GE(owners[cell], 0);
        EXPECT_LT(owners[cell], comm->getSize());
    }
    return owners;
}

void expect_factorization(const Fixture& fixture, const Scenario& scenario, const std::vector<int>& owners)
{
    const auto& decomposition = fixture.mesh->semi_structured_decomposition();
    const auto px = decomposition.xy_partitions, pz = decomposition.z_partitions;
    const auto ranks = std::size_t(Tpetra::getDefaultComm()->getSize());
    EXPECT_EQ(px * pz, ranks);
    if (scenario.layout == Layout::GraphXY)
    {
        EXPECT_EQ(px, ranks);
        EXPECT_EQ(pz, 1U);
    }
    else if (scenario.layout == Layout::ZSlabs)
    {
        EXPECT_EQ(px, 1U);
        EXPECT_EQ(pz, ranks);
    }
    else if (scenario.layout == Layout::Hybrid)
    {
        EXPECT_EQ(px, 2U);
        EXPECT_EQ(pz, ranks / 2);
    }
    if (px == 0 || pz == 0)
        return;
    const auto nxy = fixture.base.size(), layers = fixture.z.size() - 1;
    for (std::size_t k = 0; k < layers; ++k)
        for (std::size_t xy = 0; xy < nxy; ++xy)
        {
            EXPECT_EQ(std::size_t(owners[xy + k * nxy]) % px, std::size_t(owners[xy]) % px);
            EXPECT_EQ(std::size_t(owners[xy + k * nxy]) / px, std::size_t(owners[k * nxy]) / px);
        }
    std::size_t gapped_xy_sets = 0;
    for (std::size_t rank = 0; rank < ranks; ++rank)
    {
        std::vector<std::size_t> xy_owned;
        for (std::size_t xy = 0; xy < nxy; ++xy)
            if (std::size_t(owners[xy]) % px == rank % px)
                xy_owned.push_back(xy);
        if (xy_owned.size() > 1 && xy_owned.back() - xy_owned.front() + 1 != xy_owned.size())
            ++gapped_xy_sets;
        std::size_t actual = 0, expected = 0;
        for (std::size_t k = 0; k < layers; ++k)
            for (std::size_t xy = 0; xy < nxy; ++xy)
            {
                actual += owners[xy + k * nxy] == int(rank);
                expected += std::size_t(owners[xy]) % px == rank % px && std::size_t(owners[k * nxy]) / px == rank / px;
            }
        EXPECT_EQ(actual, expected);
    }
    if (scenario.reordered && !scenario.tiny && px > 1)
        EXPECT_GT(gapped_xy_sets, 0U);
}

void expect_empty_owners(const Fixture& fixture, bool tiny)
{
    const auto comm = Tpetra::getDefaultComm();
    const double empty = global_sum(fixture.mesh->num_owned_cells() == 0 ? 1 : 0);
    if (tiny)
    {
        EXPECT_EQ(global_sum(double(fixture.mesh->num_owned_cells())), 1);
        EXPECT_EQ(empty, comm->getSize() - 1);
        if (fixture.mesh->num_owned_cells() == 0)
        {
            EXPECT_EQ(fixture.mesh->num_local_cells(), 0U);
            EXPECT_EQ(fixture.mesh->num_faces(), 0U);
        }
    }
}
} // namespace

TEST(SemiStructuredDistributedOperatorsTest, NativeCellVectorAndFaceGhostImportsPreserveCanonicalMetrics)
{
    for (const auto scenario : scenarios())
    {
        const bool tiny = scenario.tiny;
        SCOPED_TRACE(::testing::Message()
                     << "layout=" << int(scenario.layout) << " reordered=" << scenario.reordered << " tiny=" << tiny);
        const Fixture fixture(tiny, false, scenario.layout, scenario.reordered);
        const auto& mesh = *fixture.mesh;
        expect_algebraic_maps(fixture);
        Scalar cell_ids(fixture.mesh, -91, "algebraic_cell_ids");
        Flux face_ids(fixture.mesh, -91, "algebraic_face_ids");
        for (std::size_t i = 0; i < mesh.num_owned_cells(); ++i)
            cell_ids.set_owned_value(Local(i), double(mesh.owned_cell_map()->getGlobalElement(Local(i))) + .5);
        for (std::size_t i = 0; i < mesh.num_owned_faces(); ++i)
            face_ids.set_owned_value(Local(i), double(mesh.owned_face_map()->getGlobalElement(Local(i))) + .25);
        cell_ids.sync_ghosts();
        face_ids.sync_ghosts();
        EXPECT_EQ(cell_ids.owned_read_view().extent(0), mesh.num_owned_cells());
        EXPECT_EQ(cell_ids.local_read_view().extent(0), mesh.num_local_cells());
        EXPECT_EQ(face_ids.owned_read_view().extent(0), mesh.num_owned_faces());
        EXPECT_EQ(face_ids.local_read_view().extent(0), mesh.num_faces());
        for (std::size_t i = 0; i < mesh.num_local_cells(); ++i)
            EXPECT_DOUBLE_EQ(
                cell_ids.local_value(Local(i)), double(mesh.overlap_cell_map()->getGlobalElement(Local(i))) + .5);
        for (std::size_t i = 0; i < mesh.num_faces(); ++i)
            EXPECT_DOUBLE_EQ(
                face_ids.local_value(Local(i)), double(mesh.overlap_face_map()->getGlobalElement(Local(i))) + .25);
        Scalar scalar(fixture.mesh, -91, "analytic");
        Vector vector(fixture.mesh, Vec{-91, -91, -91}, "analytic_vector");
        Flux flux(fixture.mesh, -91, "canonical_flux");
        for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
        {
            const auto point = fixture.center(mesh.cell_geometry_global_id(Local(c)));
            scalar.set_owned_value(Local(c), analytic(point));
            vector.set_owned_value(Local(c), analytic_vector(point));
        }
        for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
            flux.set_owned_value(
                Local(f), dot(fixture.face(mesh.face_geometry_global_id(Local(f))).area_vector, {.2, -.1, .3}));
        scalar.sync_ghosts();
        vector.sync_ghosts();
        flux.sync_ghosts();
        double local_ghosts = 0;
        for (std::size_t c = 0; c < mesh.num_local_cells(); ++c)
        {
            const auto canonical = mesh.cell_geometry_global_id(Local(c));
            expect_vector(mesh.cell_centroid(Local(c)), fixture.center(canonical));
            EXPECT_NEAR(mesh.cell_volume(Local(c)), fixture.volume(canonical), 2e-13);
            EXPECT_NEAR(scalar.local_value(Local(c)), analytic(fixture.center(canonical)), 2e-13);
            expect_vector(vector.local_value(Local(c)), analytic_vector(fixture.center(canonical)));
            local_ghosts += mesh.is_owned_cell(Local(c)) ? 0 : 1;
        }
        for (std::size_t f = 0; f < mesh.num_faces(); ++f)
        {
            const auto raw = fixture.face(mesh.face_geometry_global_id(Local(f)));
            expect_vector(mesh.face_centroid(Local(f)), raw.center);
            expect_vector(mesh.face_area_vector(Local(f)), raw.area_vector);
            EXPECT_NEAR(flux.local_value(Local(f)), dot(raw.area_vector, {.2, -.1, .3}), 2e-13);
        }
        for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
            scalar.set_owned_value(Local(c), scalar.value(Local(c)) + 3);
        scalar.sync_ghosts();
        for (std::size_t c = 0; c < mesh.num_local_cells(); ++c)
            EXPECT_NEAR(scalar.local_value(Local(c)),
                3 + analytic(fixture.center(mesh.cell_geometry_global_id(Local(c)))), 2e-13);
        EXPECT_EQ(global_sum(double(mesh.num_owned_cells())), fixture.geometry->num_cells());
        EXPECT_EQ(global_sum(double(mesh.num_owned_faces())), fixture.geometry->num_faces());
        if (!tiny && Tpetra::getDefaultComm()->getSize() > 1)
            EXPECT_GT(global_sum(local_ghosts), 0);
        // Empty local batches remain part of the declared boundary schema.
        BoundaryConditionSet conditions;
        const Vec prescribed{.25, -.5, .75};
        for (const auto& [batch, local] : mesh.boundary_batches())
            conditions.velocity[mesh.boundary_batch_name(batch)] = {BoundaryConditionType::Dirichlet, prescribed};
        const auto boundary_cache = FVM::cache_velocity_boundary_conditions<Pack>(fixture.mesh, conditions);
        EXPECT_EQ(boundary_cache.value.size(), mesh.boundary_batches().size());
        EXPECT_EQ(boundary_cache.type.size(), mesh.boundary_batches().size());
        for (const auto& [batch, local] : mesh.boundary_batches())
        {
            EXPECT_TRUE(boundary_cache.value.contains(batch));
            EXPECT_TRUE(boundary_cache.type.contains(batch));
            if (boundary_cache.value.contains(batch))
            {
                const auto& values = boundary_cache.value.at(batch);
                EXPECT_EQ(values.size(), local.face_lids.size());
                for (const auto value : values)
                    expect_vector(value, prescribed);
            }
            if (boundary_cache.type.contains(batch))
                EXPECT_EQ(boundary_cache.type.at(batch), BoundaryConditionType::Dirichlet);
        }
        expect_empty_owners(fixture, tiny);
        expect_native(fixture);
    }
}

TEST(SemiStructuredDistributedOperatorsTest, MappedTraverseAndDirectBatchesKeepFieldOrdinalsAndIncidentOwnerSigns)
{
    for (const auto scenario : scenarios(true))
    {
        const bool tiny = scenario.tiny, single_layer = scenario.single_layer;
        SCOPED_TRACE(::testing::Message() << "layout=" << int(scenario.layout) << " reordered=" << scenario.reordered
                                          << " tiny=" << tiny << " single=" << single_layer);
        const Fixture fixture(tiny, single_layer, scenario.layout, scenario.reordered);
        const auto& mesh = *fixture.mesh;
        expect_algebraic_maps(fixture);
        const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
        const auto traversal = plan.acquire();
        if (Tpetra::getDefaultComm()->getSize() > 1)
        {
            EXPECT_FALSE(plan.uses_structured_ranges());
            EXPECT_FALSE(plan.uses_layered_ranges());
        }
        std::vector<Local> cells, faces, local_cells, local_faces;
        traversal.cells().all().apply(
            [&](const auto& cell)
            {
                cells.push_back(cell.id());
                EXPECT_NEAR(cell.volume(), fixture.volume(mesh.cell_geometry_global_id(cell.id())), 2e-13);
                expect_vector(cell.centroid(), fixture.center(mesh.cell_geometry_global_id(cell.id())));
            });
        traversal.cells().all().local().apply([&](const auto& cell) { local_cells.push_back(cell.id()); });
        traversal.faces().all().local().apply([&](const auto& face) { local_faces.push_back(face.id()); });
        double partition_faces = 0, skew_partition_faces = 0;
        traversal.faces().all().apply(
            [&](const auto& face)
            {
                faces.push_back(face.id());
                const auto raw = fixture.face(mesh.face_geometry_global_id(face.id()));
                expect_vector(face.centroid(), raw.center);
                expect_vector(face.normal() * face.area(), raw.area_vector);
                ASSERT_NE(face.owner(), Handle::invalid_local_id());
                EXPECT_EQ(mesh.cell_geometry_global_id(face.owner()), raw.owner);
                if constexpr (std::remove_cvref_t<decltype(face)>::is_interior)
                {
                    ASSERT_NE(face.neighbor(), Handle::invalid_local_id());
                    EXPECT_EQ(mesh.cell_geometry_global_id(face.neighbor()), raw.neighbor);
                    if (!mesh.is_owned_cell(face.neighbor()))
                    {
                        ++partition_faces;
                        const auto delta = fixture.center(raw.neighbor) - fixture.center(raw.owner);
                        if ((delta -
                                raw.area_vector * (dot(delta, raw.area_vector) / dot(raw.area_vector, raw.area_vector)))
                                .norm() > 1e-6)
                            ++skew_partition_faces;
                    }
                }
                else
                    EXPECT_EQ(raw.neighbor, -1);
            });
        EXPECT_EQ(cells.size(), mesh.num_owned_cells());
        EXPECT_EQ(faces.size(), mesh.num_owned_faces());
        EXPECT_EQ(local_cells.size(), mesh.num_local_cells());
        EXPECT_EQ(local_faces.size(), mesh.num_faces());
        for (std::size_t i = 0; i < local_cells.size(); ++i)
            EXPECT_EQ(local_cells[i], Local(i));
        for (std::size_t i = 0; i < local_faces.size(); ++i)
            EXPECT_EQ(local_faces[i], Local(i));
        std::vector<double> metrics;
        for (const auto cell : cells)
        {
            metrics.push_back(fixture.volume(mesh.cell_geometry_global_id(cell)));
            std::vector<Global> actual;
            traversal.cell_faces(cell).all().apply(
                [&](const auto& face)
                {
                    const auto canonical = mesh.face_geometry_global_id(face.id());
                    actual.push_back(canonical);
                    const auto raw = fixture.face(canonical);
                    const int sign = raw.owner == mesh.cell_geometry_global_id(cell) ? 1 : -1;
                    EXPECT_EQ(face.orientation(), sign);
                    expect_vector(face.outward_normal() * face.area(), raw.area_vector * sign);
                });
            EXPECT_EQ(actual, fixture.cell_faces(mesh.cell_geometry_global_id(cell)));
        }
        // A grouped position deliberately differs from its field ordinal.
        // Direct batches must retain the explicit pairing in that order.
        std::reverse(cells.begin(), cells.end());
        std::reverse(metrics.begin(), metrics.end());
        const auto direct = DirectBatchRange<Local, double>::checked(cells, metrics);
        double direct_volume = 0;
        direct.apply(
            [&](Local cell, double volume)
            {
                EXPECT_NEAR(volume, fixture.volume(mesh.cell_geometry_global_id(cell)), 2e-13);
                direct_volume += volume;
            });
        EXPECT_EQ(direct.size(), mesh.num_owned_cells());
        EXPECT_TRUE(direct.subrange(direct.size(), 0).empty());
        double expected_volume = 0;
        for (std::size_t cell = 0; cell < fixture.geometry->num_cells(); ++cell)
            expected_volume += fixture.volume(Global(cell));
        EXPECT_NEAR(global_sum(direct_volume), expected_volume, 2e-12);
        const auto owners = owner_map(fixture);
        expect_factorization(fixture, scenario, owners);
        double expected_partition_faces = 0, expected_skew_partition_faces = 0;
        double expected_xy_cuts = 0, expected_z_cuts = 0;
        for (std::size_t f = 0; f < fixture.geometry->num_faces(); ++f)
        {
            const auto raw = fixture.face(Global(f));
            if (raw.neighbor < 0 || owners[std::size_t(raw.owner)] == owners[std::size_t(raw.neighbor)])
                continue;
            ++expected_partition_faces;
            const auto id = fixture.geometry->indexer().face_id(f);
            (id.orientation == Native::AXIAL ? expected_z_cuts : expected_xy_cuts) += 1;
            const auto delta = fixture.center(raw.neighbor) - fixture.center(raw.owner);
            if ((delta - raw.area_vector * (dot(delta, raw.area_vector) / dot(raw.area_vector, raw.area_vector)))
                    .norm() > 1e-6)
                ++expected_skew_partition_faces;
        }
        EXPECT_EQ(global_sum(partition_faces), expected_partition_faces);
        EXPECT_EQ(global_sum(skew_partition_faces), expected_skew_partition_faces);
        if (!tiny && Tpetra::getDefaultComm()->getSize() > 1)
        {
            const auto& decomposition = mesh.semi_structured_decomposition();
            if (decomposition.xy_partitions > 1)
            {
                EXPECT_GT(expected_xy_cuts, 0);
                EXPECT_GT(expected_skew_partition_faces, 0);
            }
            if (!single_layer && decomposition.z_partitions > 1)
                EXPECT_GT(expected_z_cuts, 0);
        }
        expect_empty_owners(fixture, tiny);
        expect_native(fixture);
    }
}

TEST(SemiStructuredDistributedOperatorsTest, LinearGradientAndNonorthogonalDiffusionSatisfyIndependentAnalyticResidual)
{
    for (const auto scenario : scenarios())
    {
        const bool tiny = scenario.tiny;
        SCOPED_TRACE(::testing::Message()
                     << "layout=" << int(scenario.layout) << " reordered=" << scenario.reordered << " tiny=" << tiny);
        const Fixture fixture(tiny, false, scenario.layout, scenario.reordered);
        const auto& mesh = *fixture.mesh;
        Scalar scalar(fixture.mesh, "linear");
        Vector gradient(fixture.mesh, "gradient");
        Flux zero_flux(fixture.mesh, 0.0, "zero_flux");
        initialize(fixture, scalar);
        auto boundary_value = [&](int batch, std::size_t index)
        {
            const auto local = mesh.boundary_face_batch(batch).face_lids.at(index);
            return analytic(fixture.face(mesh.face_geometry_global_id(local)).center);
        };
        auto boundary = [&](int batch, std::size_t index)
        { return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary_value(batch, index)}; };
        FVM::CellGradientCache<Pack, Handle> cache(fixture.mesh);
        FVM::cell_gradient(scalar, boundary, boundary_value, gradient, cache);
        for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
            expect_vector(gradient.value(Local(c)), {1.3, -.7, .4}, 2e-11);
        EXPECT_EQ(cache.mesh_ptr().get(), fixture.mesh.get());
        if (cache.uses_layered_storage() || cache.uses_structured_storage())
            EXPECT_FALSE(cache.legacy_geometry_materialized());
        else
        {
            // "Legacy geometry" here names the generic cache metric vectors,
            // not a conversion to legacy Mesh. Mapped native caches own only
            // this rank's rows, including no rows on an empty owner.
            EXPECT_EQ(cache.interior_geometry().size(), mesh.num_owned_cells());
            EXPECT_EQ(cache.boundary_geometry().size(), mesh.num_owned_cells());
        }
        expect_native(fixture);
        FVM::TransportGeometryCache<Handle> transport_geometry(mesh);
        if (!tiny)
            EXPECT_GT(global_sum(transport_geometry.has_non_orthogonal_faces() ? 1 : 0), 0);
        for (const auto treatment : {FVM::NonOrthogonalTreatment::Explicit, FVM::NonOrthogonalTreatment::Implicit,
                 FVM::NonOrthogonalTreatment::Hybrid})
        {
            const auto system = FVM::non_orthogonal_transport_system<Pack>(
                scalar, zero_flux, .03, .7, boundary, boundary_value, [](Local) { return 0.; }, treatment,
                treatment == FVM::NonOrthogonalTreatment::Implicit ? nullptr : &scalar, Teuchos::null,
                &transport_geometry);
            const auto rhs = system.rhs->getData();
            for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
            {
                double action = 0;
                for (const auto [column, value] : matrix_row(fixture, *system.matrix, Local(c)))
                    action += value * analytic(fixture.center(column));
                EXPECT_NEAR(action - rhs[c], 0, 2e-10);
            }
            Pack::vector_type solution(mesh.owned_cell_map(), true);
            BelosLinearSolver<Pack> solver;
            LinearSolverOptions options;
            options.tolerance = 1e-12;
            options.max_iterations = 300;
            ASSERT_TRUE(solver.solve(system.matrix, *system.rhs, solution, options));
            const auto values = solution.getData();
            for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
                EXPECT_NEAR(values[c], analytic(fixture.center(mesh.cell_geometry_global_id(Local(c)))), 2e-10);
        }
        expect_empty_owners(fixture, tiny);
        expect_native(fixture);
    }
}

TEST(SemiStructuredDistributedOperatorsTest, PartitionUpwindRowsAndTransportConserveTheIndependentExtensiveInventory)
{
    for (const auto scenario : scenarios())
    {
        const bool tiny = scenario.tiny;
        SCOPED_TRACE(::testing::Message()
                     << "layout=" << int(scenario.layout) << " reordered=" << scenario.reordered << " tiny=" << tiny);
        const Fixture fixture(tiny, false, scenario.layout, scenario.reordered);
        const auto& mesh = *fixture.mesh;
        Scalar old(fixture.mesh, "old");
        Flux flux(fixture.mesh, 0.0, "transport_flux");
        initialize(fixture, old);
        const Vec velocity{.2, -.1, .3};
        for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
        {
            const auto raw = fixture.face(mesh.face_geometry_global_id(Local(f)));
            flux.set_owned_value(Local(f), raw.neighbor < 0 ? 0 : dot(raw.area_vector, velocity));
        }
        flux.sync_ghosts();
        constexpr double dt = .03;
        const auto system = FVM::transport_system<Pack>(
            old, flux, dt, 0., [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; },
            [](int, std::size_t) { return 0.; }, [](Local) { return 0.; });
        for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
        {
            const Global canonical = mesh.cell_geometry_global_id(Local(c));
            std::map<Global, double> expected{{canonical, fixture.volume(canonical) / dt}};
            for (const auto f : fixture.cell_faces(canonical))
            {
                const auto raw = fixture.face(f);
                if (raw.neighbor < 0)
                    continue;
                const double outward = (raw.owner == canonical ? 1 : -1) * dot(raw.area_vector, velocity);
                const auto upwind = outward >= 0 ? canonical : (raw.owner == canonical ? raw.neighbor : raw.owner);
                expected[upwind] += outward;
            }
            const auto actual = matrix_row(fixture, *system.matrix, Local(c));
            for (const auto [column, value] : actual)
                EXPECT_NEAR(value, expected.contains(column) ? expected.at(column) : 0, 2e-12);
            for (const auto [column, value] : expected)
            {
                EXPECT_TRUE(actual.contains(column));
                if (actual.contains(column))
                    EXPECT_NEAR(actual.at(column), value, 2e-12);
            }
            EXPECT_NEAR(
                system.rhs->getData()[c], fixture.volume(canonical) / dt * analytic(fixture.center(canonical)), 2e-12);
        }
        Pack::vector_type solution(mesh.owned_cell_map(), true);
        BelosLinearSolver<Pack> solver;
        LinearSolverOptions options;
        options.tolerance = 1e-12;
        options.max_iterations = 300;
        ASSERT_TRUE(solver.solve(system.matrix, *system.rhs, solution, options));
        const auto values = solution.getData();
        double before = 0, after = 0, change = 0;
        for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
        {
            const auto canonical = mesh.cell_geometry_global_id(Local(c));
            before += fixture.volume(canonical) * analytic(fixture.center(canonical));
            after += fixture.volume(canonical) * values[c];
            change += fixture.volume(canonical) * std::abs(values[c] - old.value(Local(c)));
            EXPECT_GE(values[c], 0);
        }
        EXPECT_NEAR(global_sum(after), global_sum(before), 2e-10);
        if (!tiny)
            EXPECT_GT(global_sum(change), 1e-4);
        expect_empty_owners(fixture, tiny);
        expect_native(fixture);
    }
}
