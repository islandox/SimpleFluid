/** @file testLocalProductAMR.cc
 * @brief Local factor ancestry, shared native views and exact face certificates.
 */
#include "geometry/ExactPlanarFace.hh"
#include "geometry/LocalProductAMR.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
ExtrudedAMRGrid grid()
{
    ExtrudedAMRGrid g;
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 3; ++i)
            g.xy_nodes.push_back({double(i), double(j), 0});
    g.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
    g.z_edges = {0, .25, .5, .75, 1};
    g.boundary_aliases = {{"zmin", "floor"}, {"zmax", "roof"}, {"side", "wall"}};
    return g;
}
ProductCellAncestry mark(unsigned base = 0, unsigned interval = 1)
{
    return {{base, 1, {}}, {interval, {}}};
}
std::map<AdaptiveCellKey, ConvexGeometryPtr> cells(const LocalProductAMR& h)
{
    std::map<AdaptiveCellKey, ConvexGeometryPtr> result;
    for (std::size_t r = 0; r < h.runs().size(); ++r)
        for (std::uint64_t c = 0; c < h.runs()[r].region.layout().cells; ++c)
            EXPECT_TRUE(result.emplace(h.cell_key(r, c), h.cell_geometry(r, c)).second);
    return result;
}
void same_cells(const LocalProductAMR& a, const LocalProductAMR& b)
{
    const auto first = cells(a), second = cells(b);
    ASSERT_EQ(first.size(), second.size());
    for (const auto& [key, geometry] : first)
    {
        auto it = second.find(key);
        ASSERT_NE(it, second.end());
        EXPECT_TRUE(geometry->equivalent(*it->second));
    }
}
void covers(const LocalProductAMR& source, const LocalProductAMR& target)
{
    const auto old = cells(source), next = cells(target);
    for (const auto& [key, geometry] : old)
    {
        std::vector<ConvexGeometryPtr> pieces;
        for (const auto& [other, candidate] : next)
            if (auto overlap = geometry->intersect(*candidate))
                pieces.push_back(std::move(overlap));
        EXPECT_NO_THROW(geometry->certify_partition(pieces));
    }
}
TEST(LocalProductAMRTest, SelectedColumnAndIntervalGiveNineteenOrSeventeenCells)
{
    LocalProductAMR source(grid());
    EXPECT_EQ(source.storage_counts().cells, 16U);
    ASSERT_EQ(source.runs().size(), 1U);
    auto xy = source.adapt({mark()});
    auto z = source.adapt({}, {mark()});
    EXPECT_EQ(xy->storage_counts().cells, 19U);
    EXPECT_EQ(z->storage_counts().cells, 17U);
    EXPECT_EQ(xy->runs().size(), 4U);
    EXPECT_EQ(z->runs().size(), 2U);
    covers(source, *xy);
    covers(source, *z);
    auto before = cells(source), after = cells(*xy);
    for (const auto& [key, geometry] : before)
        if (key.ancestry != AdaptiveCellKey{source.cell_key(0, 0).region, mark()}.ancestry)
        {
            auto found = after.find(key);
            ASSERT_NE(found, after.end());
            EXPECT_TRUE(geometry->equivalent(*found->second));
        }
    auto whole = ExtrudedAMR(grid()).adapt({0});
    EXPECT_GE(whole->geometry()->num_cells(), 28U);
    EXPECT_GT(whole->geometry()->num_cells(), xy->storage_counts().cells);
    std::cout << "Local product cells: baseline=16 local_xy=" << xy->storage_counts().cells
              << " local_z=" << z->storage_counts().cells << " whole_base_xy=" << whole->geometry()->num_cells()
              << '\n';
}
TEST(LocalProductAMRTest, FactorOrderCompleteFamiliesAndPatchLabelsRecoverOriginalDomain)
{
    LocalProductAMR source(grid());
    auto xy = source.adapt({mark()});
    std::vector<ProductCellAncestry> refine_z;
    for (const auto& [key, geometry] : cells(*xy))
    {
        const auto p = std::get<ProductCellAncestry>(key.ancestry);
        if (p.base.root == 0 && !p.base.path.empty() && p.axial == mark().axial)
            refine_z.push_back(p);
    }
    ASSERT_EQ(refine_z.size(), 4U);
    auto xyz = xy->adapt({}, refine_z);
    auto z = source.adapt({}, {mark()});
    std::vector<ProductCellAncestry> refine_xy;
    for (const auto& [key, geometry] : cells(*z))
    {
        const auto p = std::get<ProductCellAncestry>(key.ancestry);
        if (p.base == mark().base && p.axial.root == 1 && !p.axial.path.empty())
            refine_xy.push_back(p);
    }
    ASSERT_EQ(refine_xy.size(), 2U);
    auto zxy = z->adapt(refine_xy);
    auto together = source.adapt({mark()}, {mark()});
    EXPECT_EQ(xyz->storage_counts().cells, 23U);
    same_cells(*xyz, *zxy);
    same_cells(*xyz, *together);
    covers(source, *xyz);
    std::vector<ProductCellAncestry> coarsen_xy;
    for (const auto& p : refine_xy)
        coarsen_xy.push_back({mark().base, p.axial});
    auto xy_restored = xyz->adapt({}, {}, coarsen_xy);
    auto restored = xy_restored->adapt({}, {}, {}, {mark()});
    same_cells(source, *restored);
    EXPECT_EQ(restored->runs().size(), 1U);
    std::set<std::string> names;
    for (const auto& run : restored->runs())
        for (auto id : run.region.topology().boundary_batch_ids())
            names.insert(run.region.topology().boundary_batch_name(id));
    EXPECT_TRUE(names.contains("floor"));
    EXPECT_TRUE(names.contains("roof"));
    EXPECT_TRUE(names.contains("wall"));
    auto invalid = mark();
    invalid.base.path = {99};
    const auto identity = xyz->identity();
    EXPECT_THROW((void) xyz->adapt({}, {}, {mark()}), std::invalid_argument);
    EXPECT_THROW((void) xyz->adapt({}, {}, {}, {mark()}), std::invalid_argument);
    EXPECT_THROW((void) xyz->adapt({}, {}, {invalid}), std::invalid_argument);
    EXPECT_EQ(xyz->identity(), identity);
}
TEST(LocalProductAMRTest, RepeatedRunsShareBaseTopologyMetricsAxisValuesAndKeyViews)
{
    LocalProductAMR source(grid());
    auto repeated = source.adapt({mark(0, 1), mark(0, 3)});
    const auto counts = repeated->storage_counts();
    EXPECT_EQ(counts.cells, 22U);
    EXPECT_EQ(counts.runs, 5U);
    EXPECT_EQ(counts.base_templates, 3U);
    EXPECT_EQ(counts.axial_values, 1U);
    bool pair = false;
    for (std::size_t a = 0; a < repeated->runs().size(); ++a)
        for (std::size_t b = 0; b < a; ++b)
        {
            const auto& x = repeated->runs()[a];
            const auto& y = repeated->runs()[b];
            if (x.base_keys.size() != y.base_keys.size() ||
                !std::equal(x.base_keys.begin(), x.base_keys.end(), y.base_keys.begin()))
                continue;
            pair = true;
            EXPECT_EQ(x.region.topology().base_storage_identity(), y.region.topology().base_storage_identity());
            EXPECT_EQ(x.region.geometry().construction()->base_storage_identity(),
                y.region.geometry().construction()->base_storage_identity());
            EXPECT_EQ(x.region.geometry().xy_nodes().data(), y.region.geometry().xy_nodes().data());
            EXPECT_EQ(x.base_keys.storage_identity(), y.base_keys.storage_identity());
            EXPECT_EQ(x.region.geometry().construction()->axis_values_storage_identity(),
                y.region.geometry().construction()->axis_values_storage_identity());
        }
    EXPECT_TRUE(pair);
    EXPECT_GT(counts.seam_descriptors, 0U);
    EXPECT_GT(repeated->storage_report().geometry, 0U);
    std::cout << "Local product shared state: runs=" << counts.runs << " bases=" << counts.base_templates
              << " axes=" << counts.axis_templates << " coordinate_dags=" << counts.axial_values
              << " selection_entries=" << counts.selection_entries
              << " bytes=" << repeated->storage_report().measured_bytes() << '\n';
}
TEST(LocalProductAMRTest, DiscardedCyclesReleaseTemplatesAndMergeAdjacentRuns)
{
    auto source = std::make_shared<const LocalProductAMR>(grid());
    const auto baseline = source->storage_report().measured_bytes();
    const auto initial = source->storage_counts();
    auto current = source;
    for (unsigned cycle = 0; cycle < 4; ++cycle)
    {
        auto fine = current->adapt({mark()});
        std::weak_ptr<const ExactExtrudedConstruction> retired;
        for (const auto& run : fine->runs())
            if (!run.base_keys.front().path.empty())
                retired = run.region.geometry().construction();
        ASSERT_FALSE(retired.expired());
        current = fine->adapt({}, {}, {mark()});
        fine.reset();
        EXPECT_TRUE(retired.expired());
        same_cells(*source, *current);
        EXPECT_EQ(current->runs().size(), 1U);
        EXPECT_EQ(current->storage_counts().point_recipes, initial.point_recipes);
        EXPECT_EQ(current->storage_counts().z_recipes, initial.z_recipes);
        EXPECT_EQ(current->storage_counts().base_templates, 1U);
        EXPECT_EQ(current->storage_counts().axis_templates, 1U);
        EXPECT_LE(current->storage_report().measured_bytes(), baseline + 2048U);
    }
}
TEST(LocalProductAMRTest, SharedCodecBindsPartsAndPreservesRunOffsetWithoutDisplayCopies)
{
    LocalProductAMR source(grid());
    auto local = source.adapt({mark()});
    std::map<const void*, std::shared_ptr<const ExtrudedTopology>> topology;
    std::map<const void*, std::shared_ptr<const ExactExtrudedConstruction>> bases, values;
    bool changed_selection = false, nonzero_offset = false;
    for (const auto& run : local->runs())
    {
        const auto& t = run.region.topology();
        const auto& c = run.region.geometry().construction();
        auto [top, inserted] = topology.try_emplace(t.base_storage_identity());
        if (inserted)
            top->second = ExtrudedTopology::decode_base(t.encode_base());
        auto target = top->second->with_layers(t.indexer().num_layers, t.boundary_aliases());
        auto retained_values = values[c->axis_values_storage_identity()];
        changed_selection = changed_selection ||
                            (retained_values && retained_values->encode_axis_selection() != c->encode_axis_selection());
        nonzero_offset = nonzero_offset || c->first_layer() != 0;
        auto decoded = ExactExtrudedConstruction::decode_shared(target, c->encode_base(), c->encode_axis(),
            c->encode_view(), bases[c->base_storage_identity()], values[c->axis_values_storage_identity()]);
        if (bases[c->base_storage_identity()])
            EXPECT_EQ(decoded->base_storage_identity(), bases[c->base_storage_identity()]->base_storage_identity());
        else
            bases[c->base_storage_identity()] = decoded;
        if (values[c->axis_values_storage_identity()])
            EXPECT_EQ(decoded->axis_values_storage_identity(),
                values[c->axis_values_storage_identity()]->axis_values_storage_identity());
        else
            values[c->axis_values_storage_identity()] = decoded;
        EXPECT_EQ(decoded->first_layer(), c->first_layer());
        EXPECT_EQ(decoded->display_z(), c->display_z());
        for (std::uint64_t cell = 0; cell < run.region.layout().cells; ++cell)
            EXPECT_TRUE(decoded->certified_cell_geometry(cell)->equivalent(*c->certified_cell_geometry(cell)));
        auto geometry = std::make_shared<const ExtrudedGeometry>(target, decoded);
        EXPECT_EQ(geometry->storage_report().geometry, decoded->storage_bytes());
    }
    EXPECT_TRUE(changed_selection);
    EXPECT_TRUE(nonzero_offset);
}
TEST(LocalProductAMRTest, ExactNestedFaceCertificatesRejectGapsCrossedTilesAndWrongOrientation)
{
    const std::array<GeometryPoint, 4> full{{{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}};
    const std::array<GeometryPoint, 4> bottom{{{0, 0, 0}, {0, 1, 0}, {0, 1, .5}, {0, 0, .5}}};
    const std::array<GeometryPoint, 4> top{{{0, 0, .5}, {0, 1, .5}, {0, 1, 1}, {0, 0, 1}}};
    auto coarse = ExactPlanarFace::make(full), low = ExactPlanarFace::make(bottom),
         high = ExactPlanarFace::make(top)->reversed();
    const std::array tiles{low, high};
    EXPECT_NO_THROW(coarse->certify_partition(tiles));
    EXPECT_TRUE(low->oppositely_oriented(*high));
    auto roundtrip = ExactPlanarFace::decode(high->encode());
    EXPECT_TRUE(roundtrip->equivalent(*high));
    EXPECT_FALSE(roundtrip->oppositely_oriented(*high));
    auto gap = top;
    gap[0][2] = gap[1][2] = std::nextafter(.5, 1.);
    const std::array missing{low, ExactPlanarFace::make(gap)};
    EXPECT_THROW(coarse->certify_partition(missing), std::invalid_argument);
    auto half = full;
    half[1][1] = half[2][1] = .5;
    auto strip = ExactPlanarFace::make(half);
    EXPECT_FALSE(strip->contains(*low));
    EXPECT_THROW(strip->certify_partition(tiles), std::invalid_argument);
    const std::array<GeometryPoint, 4> tiny{{{0, 0, 0}, {0, 1e-160, 0}, {0, 1e-160, 1e-160}, {0, 0, 1e-160}}};
    auto microscopic = ExactPlanarFace::make(tiny);
    auto opposite = microscopic->oriented_like({-1e-320, 0, 0});
    EXPECT_TRUE(microscopic->oppositely_oriented(*opposite));
    EXPECT_GT(opposite->area().as_double(), 0);
    auto translated = coarse->translated({1e6, 0, 0});
    EXPECT_TRUE(translated->translated({-1e6, 0, 0})->equivalent(*coarse));
}
TEST(LocalProductAMRTest, RawSharedAxisSelectionChecksPairedMetricRepresentability)
{
    auto topology = std::make_shared<const ExtrudedTopology>(4, Arr<Arr<unsigned>>{{0, 1, 2, 3}}, 1);
    auto points = [](double width)
    {
        return std::vector<ExactPointRecipe>{
            {{0, 0, 0}, {}}, {{width, 0, 0}, {}}, {{width, width, 0}, {}}, {{0, width, 0}, {}}};
    };
    const std::vector<std::size_t> active{0, 1, 2, 3}, axial{0, 1};
    auto tiny_base = ExactExtrudedConstruction::make(topology, points(1e-100), active, {{0, {}}, {1, {}}}, axial);
    auto thin_axis = ExactExtrudedConstruction::make(topology, points(1), active, {{0, {}}, {1e-200, {}}}, axial);
    EXPECT_NE(tiny_base->base_storage_identity(), thin_axis->base_storage_identity());
    EXPECT_GT(tiny_base->cell_volume(0), 0);
    EXPECT_GT(thin_axis->cell_volume(0), 0);
    EXPECT_THROW((void) ExactExtrudedConstruction::shared_axis_selection(topology, tiny_base, thin_axis, axial),
        std::runtime_error);
    EXPECT_NO_THROW((void) ExactExtrudedConstruction::shared_axis_selection(topology, tiny_base, tiny_base, axial));
}
TEST(LocalProductAMRTest, LocalFailuresAreAtomicAndRankDivergenceRejectsCollectively)
{
    auto g = grid();
    g.max_depth = 1;
    LocalProductAMR source(g);
    const auto identity = source.identity();
    auto fine = source.adapt({mark()});
    auto deep = mark();
    deep.base.path = {0};
    EXPECT_THROW((void) fine->adapt({deep}), std::invalid_argument);
    EXPECT_EQ(source.identity(), identity);
    EXPECT_EQ(source.adapt()->identity(), identity);
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() > 1)
    {
        EXPECT_THROW((void) source.adapt(comm->getRank() ? std::vector<ProductCellAncestry>{mark()}
                                                         : std::vector<ProductCellAncestry>{}),
            std::invalid_argument);
        auto invalid = mark();
        invalid.base.root = 999;
        EXPECT_THROW((void) source.adapt({comm->getRank() ? invalid : mark()}), std::invalid_argument);
    }
    EXPECT_EQ(source.storage_counts().cells, 16U);
}
TEST(LocalProductAMRTest, ClosedContactsAdmitFirstSplitAndDetectFurtherUngradedDescendants)
{
    LocalProductAMR source(grid());
    EXPECT_TRUE(source.contact_violations(2).empty());
    auto first = source.adapt({mark()});
    EXPECT_TRUE(first->contact_violations(2).empty());
    EXPECT_TRUE(source.adapt({}, {mark()})->contact_violations(2).empty());
    std::vector<ProductCellAncestry> children;
    for (unsigned child = 0; child < 4; ++child)
    {
        auto key = mark();
        key.base.path.push_back(child);
        children.push_back(key);
    }
    auto second = first->adapt(children);
    const auto violations = second->contact_violations(2);
    ASSERT_FALSE(violations.empty());
    EXPECT_TRUE(std::any_of(
        violations.begin(), violations.end(), [](const auto& v) { return v.dimension == 2 && v.span_exceeded; }));
    EXPECT_TRUE(std::any_of(
        violations.begin(), violations.end(), [](const auto& v) { return v.dimension == 1 && v.span_exceeded; }));
    EXPECT_TRUE(std::any_of(
        violations.begin(), violations.end(), [](const auto& v) { return v.dimension == 0 && v.span_exceeded; }));
    EXPECT_TRUE(first->contact_violations(2).empty());
    EXPECT_THROW((void) source.contact_violations(0.5), std::invalid_argument);
}
TEST(LocalProductAMRTest, AdjacentAxialBranchesEnforceIndependentTraceLimitAndBudget)
{
    auto g = grid();
    g.z_edges = {0, 1};
    LocalProductAMR source(g);
    auto root = mark(0, 0);
    auto first = source.adapt({}, {root});
    auto left = root;
    left.axial.path = {0};
    auto second = first->adapt({}, {left});
    auto middle = root;
    middle.axial.path = {0, 1};
    auto third = second->adapt({}, {middle});
    const auto violations = third->contact_violations(16);
    EXPECT_TRUE(std::any_of(
        violations.begin(), violations.end(), [](const auto& v) { return v.trace_jump_exceeded && !v.span_exceeded; }));
    EXPECT_THROW((void) source.contact_violations(16, 1, 1), std::invalid_argument);
    EXPECT_TRUE(source.contact_violations(16).empty());
    ExtrudedAMR whole(g);
    auto a = whole.adapt({}, {0});
    auto b = a->adapt({}, {0});
    auto c = b->adapt({}, {1});
    const auto whole_violations = c->contact_violations(16);
    EXPECT_TRUE(std::any_of(whole_violations.begin(), whole_violations.end(),
        [](const auto& v) { return v.trace_jump_exceeded && !v.span_exceeded; }));
    EXPECT_THROW((void) whole.contact_violations(16, 1, 1), std::invalid_argument);
    EXPECT_TRUE(whole.contact_violations(16).empty());
}
TEST(LocalProductAMRTest, DyadicEdgeEndpointsRetainTraceDepthWithLoosePhysicalRatio)
{
    auto g = grid();
    g.z_edges = {0, 1};
    LocalProductAMR source(g);
    auto root = mark(0, 0);
    auto first = source.adapt({root});
    auto left = root, right = root;
    left.base.path = {0};
    right.base.path = {1};
    auto second = first->adapt({left, right});
    auto below = root;
    below.base.path = {0, 1};
    auto third = second->adapt({below});
    auto endpoint = root;
    endpoint.base.path = {0, 1, 1};
    auto fourth = third->adapt({endpoint});
    // Opposite sides of the same vertical template edge, below/above its
    // midpoint: these cells meet only at that endpoint in the base plane.
    auto fine = root, coarse = root;
    fine.base.path = {0, 1, 1, 0};
    coarse.base.path = {1, 2};
    const auto catalog = cells(*fourth);
    const auto fine_geometry = catalog.at({g.region_id, fine});
    const auto coarse_geometry = catalog.at({g.region_id, coarse});
    auto bounds = [](const ConvexGeometryPtr& geometry, unsigned axis)
    {
        const auto moments = geometry->moments();
        const auto center = moments.origin[axis] + moments.first_moment[axis].value / moments.volume.value;
        const auto width = geometry->axis_spans()[axis].value;
        return std::pair{center - width / 2, center + width / 2};
    };
    EXPECT_EQ(bounds(fine_geometry, 0).second, .5L);
    EXPECT_EQ(bounds(coarse_geometry, 0).first, .5L);
    EXPECT_EQ(bounds(fine_geometry, 1).second, .25L);
    EXPECT_EQ(bounds(coarse_geometry, 1).first, .25L);
    EXPECT_EQ(bounds(fine_geometry, 0).second - bounds(fine_geometry, 0).first, 1.L / 16);
    EXPECT_EQ(bounds(coarse_geometry, 0).second - bounds(coarse_geometry, 0).first, 1.L / 4);
    const auto violations = fourth->contact_violations(16);
    EXPECT_TRUE(std::any_of(violations.begin(), violations.end(),
        [&](const auto& v)
        {
            return ((v.first == fine && v.second == coarse) || (v.first == coarse && v.second == fine)) &&
                   v.dimension == 1 && v.trace_jump_exceeded && !v.span_exceeded;
        }));
}
TEST(LocalProductAMRTest, ExactFaceBoundsRetainClosedTranslatedTinyContact)
{
    LocalProductAMR source(grid());
    auto cell = source.cell_geometry(0, 0);
    const std::vector<GeometryPoint> on = {{0, 0, 0}, {0, 1, 0}, {0, 1, .25}, {0, 0, .25}};
    auto face = ExactPlanarFace::make(on);
    EXPECT_TRUE(face->cell_bounds_overlap(*cell));
    EXPECT_FALSE(face->translated({-1e-200, 0, 0})->cell_bounds_overlap(*cell));
    auto shifted_grid = grid();
    for (auto& point : shifted_grid.xy_nodes)
        point[0] += 1e6;
    LocalProductAMR shifted(shifted_grid);
    auto translated_cell = shifted.cell_geometry(0, 0);
    EXPECT_TRUE(face->translated({1e6, 0, 0})->cell_bounds_overlap(*translated_cell));
}
} // namespace
