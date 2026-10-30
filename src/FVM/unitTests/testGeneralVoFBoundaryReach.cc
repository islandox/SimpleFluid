/** @file testGeneralVoFBoundaryReach.cc
 * @brief Construction replay and rejection counterexamples for retained reach.
 */
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/swept_volume/FiniteVoFTransport.tcc"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>

namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using namespace SimpleFluid;
namespace local = VoF::general_vof_detail;
namespace sweep = geometry::swept_volume;
constexpr auto mode = ArithmeticMode::Ordinary;
using Geometry = PLIC::ConvexCellGeometry<mode>;
using Point = PLIC::Point<mode>;
using Class = PLIC::OrdinaryBoundaryClass;

Point source_at(const Geometry& source, const std::array<long double, 3>& coordinate)
{
    for (std::size_t f = 0; f < source.num_faces(); ++f)
    {
        const auto face = source.face_geometry(f);
        const auto identities = face->vertex_identities();
        for (std::size_t p = 0; p < face->points().size(); ++p)
            if (face->points()[p] == coordinate)
                return PLIC::source_point<mode>(source, identities[p]);
    }
    throw std::logic_error("Replay point is absent from its original source.");
}

VoF::GeneralVoF::Data retained_data(
    const PLIC::ConvexGeometryPtr<mode>& geometry, std::size_t face, bool physical = true, bool retained_first = true)
{
    VoF::GeneralVoF::Data data;
    data.observation.identity = std::make_shared<const int>(1);
    data.observation.geometry = 7;
    data.supplied_observation = data.observation;
    local::Cell cell;
    cell.key.region = 3;
    cell.gid = 11;
    cell.owner = 0;
    cell.finite_geometry = geometry->dynamic_geometry();
    data.cells.emplace(cell.key, cell);
    auto topology = std::make_shared<VoF::GeneralVoF::Data::Topology>();
    Meshes::AMR::AdaptiveTile tile;
    tile.face_gid = 23;
    tile.identity = {'r', 'e', 'p', 'l', 'a', 'y'};
    tile.geometry = geometry->face_geometry(face);
    tile.first = cell.key;
    tile.first_owner = cell.owner;
    tile.exterior = physical;
    if (physical)
        tile.boundary = "wall";
    else
    {
        auto missing = cell.key;
        missing.region = 99;
        tile.second = missing;
        tile.second_owner = 3;
        if (!retained_first)
        {
            std::swap(tile.first, *tile.second);
            std::swap(tile.first_owner, tile.second_owner);
            tile.geometry = tile.geometry->reversed();
        }
    }
    topology->tiles.push_back(std::move(tile));
    topology->frontier_tile_references.push_back({decltype(topology)::element_type::SurfaceTileOwner::OwnedTiles, 0});
    data.topology = std::move(topology);
    return data;
}

struct Trial
{
    sweep::Work work{};
    VoF::GeneralVoFTransportDiagnostics diagnostics;
    std::shared_ptr<const local::LocalSpatialCache> cache;
    explicit Trial(const VoF::GeneralVoF::Data& data) : cache(local::retained_spatial_cache(data, work, diagnostics))
    {
        work.arithmetic_mode = mode;
    }
    void admit(const VoF::GeneralVoF::Data& data, const sweep::Sweep<mode>& recipe)
    {
        local::prove_retained(data, *cache, cache->frontiers.at(0), recipe, work);
    }
};

sweep::Sweep<mode> point_recipe(Point current, Point departure)
{
    sweep::Sweep<mode> recipe;
    recipe.face = 17;
    recipe.current = {current};
    recipe.departure = {departure};
    recipe.center = current;
    recipe.backcenter = departure;
    return recipe;
}

std::string rejection(Trial& trial, const VoF::GeneralVoF::Data& data, const sweep::Sweep<mode>& recipe)
{
    try
    {
        trial.admit(data, recipe);
        ADD_FAILURE() << "Exterior or stale replay was admitted.";
    }
    catch (const std::invalid_argument& error)
    {
        return error.what();
    }
    return {};
}

TEST(GeneralVoFBoundaryReach, Face17ReplaysOriginalWallConstructionBeforeDonorClipping)
{
    // Minimal native source fan from the original 50-cell fixture. These inputs
    // were captured before reach admission: reconstruct native supports and
    // velocities, rather than attaching provenance to saved departure values.
    Meshes::AMR::ExtrudedAMRGrid grid;
    grid.xy_nodes = {{0, 0, 0}, {.25, 0, 0}, {.5, 0, 0}, {.1, .5, 0}, {.35, .5, 0}, {.6, .5, 0}, {.2, 1, 0},
        {.45, 1, 0}, {.7, 1, 0}};
    grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
    grid.boundaries = {{0, 1, "wall"}, {1, 2, "wall"}, {6, 7, "wall"}, {7, 8, "wall"}, {0, 3, "wall"}, {3, 6, "wall"},
        {2, 5, "wall"}, {5, 8, "wall"}};
    grid.z_edges = {0, .5, 1};
    const Meshes::AMR::ExtrudedAMR native(grid);
    std::vector<PLIC::ConvexGeometryPtr<mode>> sources, finite;
    for (unsigned c = 0; c < 8; ++c)
    {
        // Reorder the native axial-major IDs to the captured distributed
        // source catalog's base-major, then axial order.
        sources.push_back(native.cell_geometry(c / 2 + 4 * (c % 2)));
        finite.push_back(Geometry::from_geometry(*sources.back()));
    }
    const std::array<std::uint64_t, 4> nodes{2, 11, 52, 48};
    const std::array<PLIC::GeometryPoint, 4> velocities{
        {{0.005340562151611777, -0.058947320121338027, 0.0014796064004184261},
            {0.03231757636313036, -0.034235044366371446, 0.00020134044560284232},
            {-0.0039310268269660432, -0.027857764196718336, -3.2582143939839351e-05},
            {-0.0095752013175459966, -0.057372434649726342, 0.00014737937368954833}}};
    const auto face = sources[0]->face_geometry(1);
    const auto ids = face->vertex_identities();
    std::map<std::uint64_t, std::pair<Point, Point>> traces;
    // Source catalog order is base-major, then axial interval.
    const std::array<std::vector<std::pair<unsigned, unsigned>>, 4> wall_faces{
        {{{1, 2}, {0, 2}, {0, 5}, {1, 5}}, {{1, 2}, {0, 2}, {2, 2}, {3, 2}}, {}, {{0, 5}, {1, 5}, {4, 5}, {5, 5}}}};
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const auto current = PLIC::source_point<mode>(sources, ids.at(i));
        std::vector<PLIC::FaceConstraint<mode>> walls;
        for (const auto [cell, wall] : wall_faces[i])
            walls.push_back({finite.at(cell), wall});
        traces.emplace(
            nodes[i], std::pair{current, PLIC::backtrace_constrained<mode>(current, velocities[i], .002, walls)});
    }
    auto data = retained_data(finite[0], 5);
    auto topology = std::make_shared<VoF::GeneralVoF::Data::Topology>();
    topology->tiles = data.topology->tiles;
    topology->frontier_tile_references = data.topology->frontier_tile_references;
    topology->tiles[0].face_gid = 73;
    topology->tiles[0].node_gids = {9, 0, 2, 48};
    data.topology = topology;
    Trial trial(data);
    const auto recipe = sweep::prepare_sweep<mode>(
        17, nodes, traces, {2.88986373642823436633e-07L, 0, mode}, trial.work, 8.33333333333333351685e-17L);
    const auto& departure = traces.at(48).second;
    const auto wall = finite[0]->face_geometry(5);
    const auto& boundary = *trial.cache->frontiers[0].original_boundary;
    ASSERT_TRUE(PLIC::has_any_support(departure.dynamic(), boundary.authorities()));
    EXPECT_EQ(boundary.classify(departure), Class::ConstructionContact);
    EXPECT_NE(boundary.classify(Point(departure.value)), Class::ConstructionContact);
    const auto orientation = PLIC::orientation<mode>(
        {Point(wall->points()[0]), Point(wall->points()[1]), Point(wall->points()[2]), Point(departure.value)})
                                 .value;
    // The saved positive sign belongs to the 64-bit-significand capture. Other
    // scalar formats still exercise construction contact and the unchanged reach
    // gate; their rounded tuples need not fall on the same side of the wall.
    if constexpr (std::numeric_limits<long double>::digits == 64)
        EXPECT_GT(orientation, 0);
    constexpr auto replay_tolerance = 8 * std::numeric_limits<long double>::epsilon();
    EXPECT_LE(std::abs(departure.value[0] - 0.100022802874966634604L), replay_tolerance);
    EXPECT_LE(std::abs(departure.value[1] - 0.500114014374833145272L), replay_tolerance);
    EXPECT_LE(std::abs(departure.value[2] - 0.499999705241252620898L), replay_tolerance);
    EXPECT_LE(std::abs(recipe.backcenter.value[2] - 0.499994860273139090859L), replay_tolerance);
    EXPECT_TRUE(recipe.backcenter.supports().empty());
    ASSERT_NO_THROW(trial.admit(data, recipe));
    const auto record = [](const char* name, long double value)
    {
        std::ostringstream text;
        text << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
        testing::Test::RecordProperty(name, text.str());
    };
    record("coordinate_orientation", orientation);
    const auto raw = PLIC::point_average<mode>(recipe.departure);
    record("raw_cap_z", raw.value[2]);
    record("corrected_cap_z", recipe.backcenter.value[2]);
    record("cap_displacement_over_cell_height", std::abs(recipe.backcenter.value[2] - raw.value[2]) / .5L);
    long double raw_control = 0;
    for (const auto& triangle : sweep::sweep_boundary(nodes, recipe.current, recipe.departure, recipe.center, raw))
        raw_control += sweep::cone_determinant(recipe.center, triangle).value / 6;
    record("control_minus_raw_sweep", recipe.control.value - raw_control);
}

TEST(GeneralVoFBoundaryReach, OriginalSupportCannotAuthorizeAStaleEpochOrOwner)
{
    const auto source = Geometry::box({0, 0, 0}, {1, 1, 1});
    const auto data = retained_data(source, 0);
    Trial trial(data);
    const auto contact = source_at(*source, {0, 0, 0});
    const auto recipe = point_recipe(contact, contact);
    ASSERT_NO_THROW(trial.admit(data, recipe));
    ASSERT_EQ(trial.cache->frontiers[0].original_boundary->classify(contact), Class::ConstructionContact);
    auto stale = data;
    ++stale.observation.geometry;
    EXPECT_FALSE(rejection(trial, stale, recipe).empty());
    stale = data;
    ++stale.supplied_observation.geometry;
    EXPECT_FALSE(rejection(trial, stale, recipe).empty());
    stale = data;
    ++stale.cells.begin()->second.owner;
    EXPECT_FALSE(rejection(trial, stale, recipe).empty());
    stale = data;
    const auto rebuilt = Geometry::box({0, 0, 0}, {1, 1, 1});
    // Equal geometric keys do not authenticate a replacement immutable owner.
    EXPECT_EQ(PLIC::original_support_authority(*source->face_geometry(0)),
        PLIC::original_support_authority(*rebuilt->face_geometry(0)));
    stale.cells.begin()->second.finite_geometry = rebuilt->dynamic_geometry();
    EXPECT_FALSE(rejection(trial, stale, recipe).empty());
}

TEST(GeneralVoFBoundaryReach, MissingDonorHaloAndPhysicalBoundaryHaveDistinctRejections)
{
    const auto source = Geometry::box({0, 0, 0}, {1, 1, 1});
    const auto recipe = point_recipe(Point({.5L, .5L, .5L}), Point({1.25L, .5L, .5L}));
    const auto wall = retained_data(source, 1);
    Trial wall_trial(wall);
    const auto wall_error = rejection(wall_trial, wall, recipe);
    EXPECT_NE(wall_error.find("physical"), std::string::npos) << wall_error;
    for (const bool first : {true, false})
    {
        const auto halo = retained_data(source, 1, false, first);
        Trial halo_trial(halo);
        const auto& frontier = halo_trial.cache->frontiers.at(0);
        EXPECT_EQ(frontier.first, first);
        ASSERT_TRUE(frontier.outside_key);
        EXPECT_EQ(frontier.outside_key->region, 99u);
        EXPECT_EQ(frontier.outside_owner, 3);
        const auto halo_error = rejection(halo_trial, halo, recipe);
        EXPECT_NE(halo_error.find("halo"), std::string::npos) << halo_error;
        EXPECT_NE(halo_error.find("departure"), std::string::npos) << halo_error;
        EXPECT_NE(halo_error, wall_error);
    }
}

TEST(GeneralVoFBoundaryReach, FluxMatchedCapEscapeDropsInheritedWallSupportAndRejects)
{
    const auto source = Geometry::box({0, 0, 0}, {1, 1, 1});
    const auto face = source->face_geometry(4); // z=0, outward normal -z.
    const std::array<std::uint64_t, 4> nodes{0, 1, 2, 3};
    std::map<std::uint64_t, std::pair<Point, Point>> traces;
    std::vector<Point> departure;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const auto p = source_at(*source, face->points().at(i));
        traces.emplace(nodes[i], std::pair{p, p});
        departure.push_back(p);
    }
    const auto data = retained_data(source, 4);
    Trial trial(data);
    const auto& boundary = *trial.cache->frontiers[0].original_boundary;
    const auto raw_center = PLIC::point_average<mode>(departure);
    ASSERT_EQ(boundary.classify(raw_center), Class::ConstructionContact);
    const auto recipe = sweep::prepare_sweep<mode>(17, nodes, traces, {-0.001L, 0, mode}, trial.work, 1e-12L);
    EXPECT_TRUE(recipe.backcenter.supports().empty());
    EXPECT_LT(recipe.backcenter.value[2], 0);
    EXPECT_EQ(boundary.classify(recipe.backcenter), Class::Outside);
    for (const auto& p : recipe.departure)
        EXPECT_EQ(boundary.classify(p), Class::ConstructionContact);
    const auto message = rejection(trial, data, recipe);
    EXPECT_NE(message.find("center"), std::string::npos) << message;
}
} // namespace
