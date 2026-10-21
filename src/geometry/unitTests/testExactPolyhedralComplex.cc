/** @file testExactPolyhedralComplex.cc
 * @brief Exact octants, canonical nonnested seams, sparse topology and parent recovery.
 */
#include "geometry/ExactPolyhedralComplex.hh"
#include "geometry/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using ID = std::uint64_t;
constexpr ID absent = std::numeric_limits<ID>::max();
const std::vector<std::vector<std::size_t>> cube_faces{
    {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
ConvexGeometryPtr cube()
{
    return ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1});
}
AdaptiveCellKey key(ID root, std::vector<ID> path = {})
{
    return {0, ConvexCellAncestry{root, std::move(path)}};
}
PolyhedralComplexCell tagged(ConvexGeometryPtr geometry, ID root, ID gid)
{
    PolyhedralComplexCell cell{key(root), gid, std::move(geometry), {}};
    for (std::size_t f = 0; f < cell.geometry->num_faces(); ++f)
        cell.surfaces.push_back({{PolyhedralSurfaceKey::Kind::Root, 0, {100 * root + f, {}}, 0}, 1, true, "wall"});
    return cell;
}
std::vector<PolyhedralComplexCell> split(const PolyhedralComplexCell& parent, ID first)
{
    std::vector<PolyhedralComplexCell> cells;
    const auto ancestry = std::get<ConvexCellAncestry>(parent.key.ancestry);
    for (const auto& piece : parent.geometry->split_aabb_midpoints())
    {
        auto path = ancestry.path;
        path.push_back(piece.octant);
        PolyhedralComplexCell cell{key(ancestry.root, std::move(path)), first + cells.size(), piece.geometry, {}};
        for (const auto& origin : piece.face_origins)
        {
            if (origin.kind == ConvexFaceOrigin::Kind::ParentFace)
                cell.surfaces.push_back(parent.surfaces.at(origin.index));
            else
                cell.surfaces.push_back({{PolyhedralSurfaceKey::Kind::Cut, 0, ancestry, origin.index},
                    (piece.octant >> origin.index) & 1 ? -1 : 1, false, {}});
        }
        cells.push_back(std::move(cell));
    }
    return cells;
}
ConvexGeometryPtr skew(double x0, double x1)
{
    std::array<GeometryPoint, 8> vertices;
    for (unsigned i = 0; i < 8; ++i)
    {
        const double x = i & 1 ? x1 : x0, y = (i >> 1) & 1, z = (i >> 2) & 1;
        vertices[i] = {x, y + .5 * x, z + .3 * x};
    }
    return ConvexCellGeometry::make(vertices, cube_faces);
}
std::pair<PolyhedralComplexCell, PolyhedralComplexCell> neighbors()
{
    auto left = tagged(skew(-1, 0), 0, 0), right = tagged(skew(0, 1), 1, 1);
    const PolyhedralSurfaceKey seam{PolyhedralSurfaceKey::Kind::Root, 0, {999, {}}, 0};
    left.surfaces[1] = {seam, 1, false, {}};
    right.surfaces[0] = {seam, -1, false, {}};
    return {left, right};
}
std::vector<ID> seam_faces(const ExactPolyhedralComplex& complex)
{
    std::vector<ID> result;
    for (auto face : complex.face_ids())
        if (complex.face_surface(face).key.creator.root == 999 &&
            complex.face_surface(face).key.kind == PolyhedralSurfaceKey::Kind::Root)
            result.push_back(face);
    return result;
}
TEST(ExactPolyhedralComplexTest, ActualTetrahedronHasFourOccupiedOctantsWithAnalyticVolumes)
{
    const std::array<GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    auto tetra = ConvexCellGeometry::make(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    auto children = tetra->split_aabb_midpoints();
    ASSERT_EQ(children.size(), 4U);
    std::set<unsigned> occupied;
    std::vector<ConvexGeometryPtr> pieces;
    for (const auto& child : children)
    {
        occupied.insert(child.octant);
        const auto expected = child.octant == 0 ? 5.0L / 48 : 1.0L / 48;
        EXPECT_NEAR(child.geometry->moments().volume.value, expected, 1e-18L);
        EXPECT_EQ(child.face_origins.size(), child.geometry->num_faces());
        pieces.push_back(child.geometry);
    }
    EXPECT_EQ(occupied, (std::set<unsigned>{0, 1, 2, 4}));
    EXPECT_NO_THROW(tetra->certify_partition(pieces));
    auto complex = ExactPolyhedralComplex::make(split(tagged(tetra, 0, 0), 0));
    EXPECT_EQ(complex->global_cells(), 4U);
}
TEST(ExactPolyhedralComplexTest, RationalAABBBoundsAndPrismCutsRetainTruePositivePieces)
{
    auto parent = cube()->clip({{3, 0, 0}, {0, 0, 0}, 1}).inside;
    ASSERT_TRUE(parent);
    auto children = parent->split_aabb_midpoints();
    ASSERT_EQ(children.size(), 8U);
    std::vector<ConvexGeometryPtr> pieces;
    for (const auto& child : children)
    {
        EXPECT_NEAR(child.geometry->moments().volume.value, 1.0L / 24, 1e-18L);
        pieces.push_back(child.geometry);
    }
    EXPECT_NO_THROW(parent->certify_partition(pieces));
    const std::array<std::array<double, 2>, 3> base{{{0, 0}, {1, 0}, {0, 1}}};
    auto prism = ConvexCellGeometry::extruded_polygon(base, 0, 1);
    auto cuts = prism->split_aabb_midpoints();
    EXPECT_EQ(cuts.size(), 6U);
    auto tiny = ConvexCellGeometry::box({0, 0, 0}, {1e-200, 1, 1});
    auto retained = tiny->split_aabb_midpoints();
    ASSERT_EQ(retained.size(), 8U);
    for (const auto& child : retained)
        EXPECT_GT(child.geometry->moments().volume.as_double(), 0);
    EXPECT_THROW((void) tiny->split_aabb_midpoints(1e-100), std::invalid_argument);
    EXPECT_GT(tiny->moments().volume.as_double(), 0);
    EXPECT_TRUE(parent->quality_above(0));
    EXPECT_FALSE(parent->quality_above(1));
}
TEST(ExactPolyhedralComplexTest, CoplanarAtomicParentFacesKeepTheirIndividualOrigins)
{
    std::array<GeometryPoint, 8> vertices;
    for (unsigned i = 0; i < 8; ++i)
        vertices[i] = {double(i & 1), double((i >> 1) & 1), double((i >> 2) & 1)};
    auto faces = cube_faces;
    faces.back() = {4, 5, 7};
    faces.push_back({4, 7, 6});
    auto parent = ConvexCellGeometry::make(vertices, faces);
    auto tagged_parent = tagged(parent, 0, 0);
    std::set<std::size_t> origins;
    auto pieces = parent->split_aabb_midpoints();
    for (const auto& piece : pieces)
        for (const auto& origin : piece.face_origins)
            if (origin.kind == ConvexFaceOrigin::Kind::ParentFace && origin.index >= 5)
                origins.insert(origin.index);
    EXPECT_EQ(origins, (std::set<std::size_t>{5, 6}));
    auto complex = ExactPolyhedralComplex::make(split(tagged_parent, 0));
    std::set<ID> top_surface_ids;
    for (auto face : complex->face_ids())
        if (complex->face_surface(face).exterior && complex->face_surface(face).key.creator.root >= 5)
            top_surface_ids.insert(complex->face_surface(face).key.creator.root);
    EXPECT_EQ(top_surface_ids, (std::set<ID>{5, 6}));
}
TEST(ExactPolyhedralComplexTest, UnequalNeighborTracesProduceOneNineTileSeamAndPropagateEdges)
{
    const auto [left, right] = neighbors();
    auto cells = split(left, 0);
    auto right_cells = split(right, cells.size());
    cells.insert(cells.end(), right_cells.begin(), right_cells.end());
    auto complex = ExactPolyhedralComplex::make(cells);
    const auto seam = seam_faces(*complex);
    ASSERT_EQ(seam.size(), 9U);
    double area = 0;
    for (auto face : seam)
    {
        area += complex->face_area(face);
        EXPECT_NE(complex->neighbor_cell(face), absent);
        EXPECT_NE(complex->owner_cell(face), complex->neighbor_cell(face));
    }
    EXPECT_NEAR(area, 1, 2e-15);
    bool propagated = false;
    for (auto node : complex->node_ids())
    {
        auto point = complex->node_coordinates(node);
        if (point.x == 0 && std::abs(point.y - .75) < 1e-15 && point.z == 0)
        {
            unsigned incidence = 0;
            for (auto face : complex->face_ids())
                incidence += std::count(complex->face_nodes(face).begin(), complex->face_nodes(face).end(), node) > 0;
            EXPECT_GE(incidence, 4U);
            propagated = true;
        }
    }
    EXPECT_TRUE(propagated);
    std::reverse(cells.begin(), cells.end());
    auto reordered = ExactPolyhedralComplex::make(cells);
    EXPECT_EQ(complex->global_faces(), reordered->global_faces());
    for (auto face : complex->face_ids())
    {
        EXPECT_EQ(complex->face_identity(face), reordered->face_identity(face));
        EXPECT_EQ(complex->face_nodes(face), reordered->face_nodes(face));
    }
    auto coarse = left;
    coarse.gid = 0;
    auto fine = split(right, 1);
    fine.insert(fine.begin(), coarse);
    auto mixed = ExactPolyhedralComplex::make(fine);
    EXPECT_EQ(seam_faces(*mixed).size(), 4U);
    EXPECT_GT(mixed->cell_faces(0).size(), left.geometry->num_faces());
    EXPECT_TRUE(mixed->cell_geometry(0)->equivalent(*left.geometry));
    EXPECT_EQ(mixed->atomic_cell_geometry(0)->num_faces(), mixed->cell_faces(0).size());
    EXPECT_TRUE(mixed->atomic_cell_geometry(0)->equivalent(*left.geometry));
    EXPECT_EQ(left.geometry->num_faces(), 6U);
    auto restored = ExactPolyhedralComplex::make({left, right});
    EXPECT_EQ(seam_faces(*restored).size(), 1U);
    EXPECT_TRUE(restored->cell_geometry(0)->equivalent(*left.geometry));
}
TEST(ExactPolyhedralComplexTest, TileIdentityIgnoresLoopStartAndRedundantCollinearVertices)
{
    auto normal = tagged(cube(), 0, 0);
    auto base = ExactPolyhedralComplex::make({normal});
    std::vector<GeometryPoint> vertices{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {.5, 0, 0}};
    auto faces = cube_faces;
    faces[2] = {0, 8, 1, 5, 4};
    faces[4] = {0, 2, 3, 1, 8};
    for (auto& face : faces)
        std::rotate(face.begin(), face.begin() + 1, face.end());
    auto extra = tagged(ConvexCellGeometry::make(vertices, faces), 0, 0);
    auto refined = ExactPolyhedralComplex::make({extra});
    ASSERT_EQ(base->global_faces(), refined->global_faces());
    bool larger_loop = false;
    for (auto face : base->face_ids())
    {
        EXPECT_EQ(base->face_identity(face), refined->face_identity(face));
        larger_loop = larger_loop || base->face_nodes(face).size() < refined->face_nodes(face).size();
    }
    EXPECT_TRUE(larger_loop);
    auto [left, right] = neighbors();
    auto a = ExactPolyhedralComplex::make({left, right});
    left.gid = 1;
    right.gid = 0;
    auto b = ExactPolyhedralComplex::make({left, right});
    auto af = seam_faces(*a), bf = seam_faces(*b);
    ASSERT_EQ(af.size(), 1U);
    ASSERT_EQ(bf.size(), 1U);
    EXPECT_EQ(a->face_identity(af[0]), b->face_identity(bf[0]));
}
TEST(ExactPolyhedralComplexTest, SparseCodecRetainsOnlyVisibleGeometryAndMetricNeighbors)
{
    auto [left, right] = neighbors();
    auto complex = ExactPolyhedralComplex::make({left, right});
    const std::array<ID, 1> visible{0};
    auto sparse = ExactPolyhedralComplex::decode(complex->encode(visible));
    EXPECT_EQ(sparse->global_cells(), 2U);
    EXPECT_TRUE(sparse->has_full_cell(0));
    EXPECT_FALSE(sparse->has_full_cell(1));
    EXPECT_TRUE(sparse->cell_faces(1).empty());
    EXPECT_TRUE(sparse->cell_nodes(1).empty());
    EXPECT_EQ(sparse->cell_volume(1), complex->cell_volume(1));
    EXPECT_THROW((void) sparse->cell_geometry(1), std::invalid_argument);
    EXPECT_TRUE(sparse->cell_geometry(0)->equivalent(*left.geometry));
    for (auto face : sparse->face_ids())
        EXPECT_EQ(sparse->face_identity(face), complex->face_identity(face));
    auto empty = ExactPolyhedralComplex::decode(complex->encode(std::span<const ID>{}));
    EXPECT_TRUE(empty->resident_cell_ids().empty());
    EXPECT_TRUE(empty->node_ids().empty());
    EXPECT_EQ(empty->global_nodes(), complex->global_nodes());
    auto packet = complex->encode();
    packet.pop_back();
    EXPECT_THROW((void) ExactPolyhedralComplex::decode(packet), std::invalid_argument);
    packet = complex->encode();
    packet[0] ^= 1;
    EXPECT_THROW((void) ExactPolyhedralComplex::decode(packet), std::invalid_argument);
    packet = complex->encode();
    packet.push_back(0);
    EXPECT_THROW((void) ExactPolyhedralComplex::decode(packet), std::invalid_argument);
}
TEST(ExactPolyhedralComplexTest, OrderedAtomicNativeImportDoesNotGuessConstructedFaceOrder)
{
    ExtrudedAMRGrid grid;
    grid.xy_nodes = {{{.1, .1, 0}}, {{.2, .1, 0}}, {{.2, .2, 0}}, {{.1, .2, 0}}};
    grid.base_cells = {{0, 1, 2, 3}};
    grid.z_edges = {.1, .2};
    auto hierarchy = ExtrudedAMR(grid).adapt({0});
    const auto& native = *hierarchy->geometry();
    std::vector<PolyhedralComplexCell> cells;
    for (ID cell = 0; cell < native.num_cells(); ++cell)
    {
        auto geometry = ConvexCellGeometry::from_native_cell_atomic(native, cell);
        PolyhedralComplexCell record{key(cell), cell, geometry, {}};
        for (auto face : native.cell_faces(cell))
            record.surfaces.push_back(
                {{PolyhedralSurfaceKey::Kind::Root, 0, {face, {}}, 0}, native.owner_cell(face) == cell ? 1 : -1,
                    native.neighbor_cell(face) == absent, native.neighbor_cell(face) == absent ? "wall" : ""});
        ASSERT_EQ(record.surfaces.size(), geometry->num_faces());
        cells.push_back(std::move(record));
    }
    const auto complex = ExactPolyhedralComplex::make(cells);
    EXPECT_EQ(complex->global_faces(), native.num_faces());
    for (ID cell = 0; cell < native.num_cells(); ++cell)
        EXPECT_TRUE(complex->cell_geometry(cell)->equivalent(*hierarchy->cell_geometry(cell)));
}
TEST(ExactPolyhedralComplexTest, InvalidAncestralSeamsAndNativeUnrepresentabilityRejectAtomically)
{
    const auto [left, right] = neighbors();
    auto invalid = right;
    invalid.surfaces[0].orientation = 1;
    EXPECT_THROW((void) ExactPolyhedralComplex::make({left, invalid}), std::invalid_argument);
    invalid = right;
    invalid.surfaces[0].key.creator.root = 12345;
    EXPECT_THROW((void) ExactPolyhedralComplex::make({left, invalid}), std::invalid_argument);
    auto overlap = tagged(ConvexCellGeometry::box({-.5, 0, 0}, {.5, 1, 1}), 1, 1);
    auto origin = tagged(cube(), 0, 0);
    EXPECT_THROW((void) ExactPolyhedralComplex::make({origin, overlap}), std::invalid_argument);
    auto near = tagged(ConvexCellGeometry::box({1e16, 0, 0}, {1e16 + 2, 1, 1}), 0, 0);
    EXPECT_THROW((void) ExactPolyhedralComplex::make({near}), std::invalid_argument);
    EXPECT_GT(left.geometry->moments().volume.value, 0);
    EXPECT_TRUE(ExactPolyhedralComplex::make({left, right})->has_full_cell(0));
}
TEST(ExactPolyhedralComplexTest, SparseDecoderRejectsMissingReverseFullCellIncidence)
{
    auto [left, right] = neighbors();
    auto remote = tagged(ConvexCellGeometry::box({3, 3, 0}, {4, 4, 1}), 2, 2);
    const auto complex = ExactPolyhedralComplex::make({left, right, remote});
    auto packet = complex->encode();
    std::size_t cursor = 0;
    auto word = [&]()
    {
        ID value = 0;
        for (unsigned b = 0; b < 8; ++b)
            value |= ID(static_cast<unsigned char>(packet.at(cursor++))) << (8 * b);
        return value;
    };
    auto skip_words = [&](ID count)
    {
        cursor += count * 8;
        ASSERT_LE(cursor, packet.size());
    };
    auto put = [&](std::size_t offset, ID value)
    {
        for (unsigned b = 0; b < 8; ++b)
            packet.at(offset + b) = static_cast<char>(value >> (8 * b));
    };
    skip_words(5);
    const auto cells = word();
    for (ID cell = 0; cell < cells; ++cell)
    {
        skip_words(3);
        const auto path = word();
        skip_words(path);
        skip_words(4);
        const auto full = word();
        if (full)
        {
            const auto bytes = word();
            cursor += bytes;
            const auto faces = word();
            skip_words(faces);
            const auto nodes = word();
            skip_words(nodes);
        }
    }
    const auto faces = word();
    bool changed = false;
    for (ID face = 0; face < faces; ++face)
    {
        const auto face_id = word();
        skip_words(3);
        const auto path = word();
        skip_words(path);
        skip_words(2);
        const auto exterior_offset = cursor;
        const auto exterior = word();
        const auto length_offset = cursor;
        const auto length = word();
        const auto text_offset = cursor;
        cursor += length;
        const auto owner = word();
        const auto neighbor_offset = cursor;
        const auto neighbor = word();
        const auto nodes = word();
        skip_words(nodes);
        if (complex->face_surface(face_id).key.creator.root == 3 && exterior && owner == 0 && neighbor == absent)
        {
            // Turn the owner's y-upper boundary into an interface to a remote
            // full cell. Both closed cell shells remain unchanged; only the
            // forged reverse incidence is missing, and signed visibility fits.
            put(exterior_offset, 0);
            put(length_offset, 0);
            put(neighbor_offset, 2);
            packet.erase(packet.begin() + text_offset, packet.begin() + text_offset + length);
            changed = true;
            break;
        }
    }
    ASSERT_TRUE(changed);
    try
    {
        (void) ExactPolyhedralComplex::decode(packet);
        FAIL() << "Accepted nonreciprocal full-cell incidence.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("reciprocal"), std::string::npos);
    }
}
TEST(ExactPolyhedralComplexTest, UnnamedExteriorStaysTopologicalExteriorAcrossCodec)
{
    auto input = tagged(cube(), 0, 0);
    for (auto& surface : input.surfaces)
        surface.boundary.clear();
    auto complex = ExactPolyhedralComplex::make({input});
    auto copy = ExactPolyhedralComplex::decode(complex->encode());
    for (auto face : copy->face_ids())
    {
        EXPECT_TRUE(copy->face_surface(face).exterior);
        EXPECT_TRUE(copy->face_surface(face).boundary.empty());
        EXPECT_EQ(copy->neighbor_cell(face), absent);
    }
    auto invalid = input;
    invalid.surfaces[0].exterior = false;
    EXPECT_THROW((void) ExactPolyhedralComplex::make({invalid}), std::invalid_argument);
    auto [left, right] = neighbors();
    left.surfaces[1].boundary = "invented-interior-patch";
    EXPECT_THROW((void) ExactPolyhedralComplex::make({left, right}), std::invalid_argument);
}
} // namespace
