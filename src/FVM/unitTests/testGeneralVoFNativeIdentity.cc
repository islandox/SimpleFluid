/** @file testGeneralVoFNativeIdentity.cc
 * @brief Independent equal-metric shape counterexample and original remap coverage gate.
 */
#include "FVM/details/GeneralVoF/CellCatalog.hh"
#include "FVM/details/GeneralVoF/Retention.hh"
#include "geometry/PLIC/details/CellGeometryEvaluation.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <string_view>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
namespace AMR = SimpleFluid::Meshes::AMR;
namespace detail = SimpleFluid::VoF::general_vof_detail;
constexpr auto mode = ArithmeticMode::Ordinary;
using Catalog = PLIC::CellGeometryCatalog<mode>;
using View = PLIC::CellGeometryView<mode>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

PLIC::GeometryLimits limits()
{
    PLIC::GeometryLimits result;
    result.arithmetic_mode = mode;
    return result;
}
View box_view(PLIC::GeometryPoint lower, PLIC::GeometryPoint upper)
{
    PLIC::CartesianSourceDescription source;
    for (unsigned axis = 0; axis < 3; ++axis)
        source.axes[axis] = {lower[axis], upper[axis]};
    PLIC::CartesianSourceCellSlot cell;
    cell.upper = {1, 1, 1};
    source.cells.push_back(cell);
    return {Catalog::cartesian(PLIC::CartesianCellSource::make(std::move(source), limits()), limits()), 0};
}
AMR::AMRGrid cube_grid()
{
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.origin = {-.5, -.5, -.5};
    grid.extent = {2, 2, 2};
    return grid;
}
AMR::AdaptiveBootstrap detached(const AMR::AMRGrid& grid, std::span<const std::uint64_t> ordinals)
{
    AMR::CartesianAMR mesh(grid);
    auto result = mesh.detach_owned(ordinals, {'n', 'a', 't', 'i', 'v', 'e', '-', 'i', 'd'});
    for (const auto& leaf : result.owned)
        result.kernel->validate(leaf);
    return result;
}
std::shared_ptr<const GeneralVoF> cube_state(ArithmeticMode arithmetic_mode = mode)
{
    const std::array<std::uint64_t, 8> ordinals{0, 1, 2, 3, 4, 5, 6, 7};
    auto source = detached(cube_grid(), ordinals);
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(source.owned), std::move(source.families), source.kernel, Tpetra::getDefaultComm());
    std::vector<GeneralVoF::Key> order;
    for (const auto& leaf : hierarchy->owned_leaves())
        order.push_back(leaf.key);
    GeneralVoFOptions options;
    options.arithmetic_mode = arithmetic_mode;
    options.phase_cleanup.relative_cutoff = 0;
    return std::make_shared<const GeneralVoF>(
        hierarchy, order, std::vector<double>(order.size(), 0), std::vector<double>(order.size(), 1), options);
}
auto identity_plan(const GeneralVoF& state)
{
    return state.hierarchy()->plan_migration(std::vector<int>(state.owned_order().size(), 0));
}
auto bound_candidate(const GeneralVoF& state, const GeneralVoF::Hierarchy::Plan& plan)
{
    const auto raw = state.hierarchy()->apply(plan);
    const auto native = AMR::assemble_adaptive_native(*raw, raw->configuration_descriptor());
    return state.hierarchy()->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
}

TEST(GeneralVoFNativeIdentity, SameVolumeAndCentroidDoNotProveOriginalShape)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused structural proof oracle is serial.";
    // Independent oracle: 2*.5*1 == 1*1*1 == 1; both centers are exactly zero.
    const auto stretched = box_view({-1, -.25, -.5}, {1, .25, .5});
    const auto cube = box_view({-.5, -.5, -.5}, {.5, .5, .5});
    const auto equal_cube = box_view({-.5, -.5, -.5}, {.5, .5, .5});
    EXPECT_EQ(stretched.volume().value, 1.L);
    EXPECT_EQ(cube.volume().value, 1.L);
    EXPECT_EQ(stretched.centroid(), (PLIC::GeometryPoint{0, 0, 0}));
    EXPECT_EQ(cube.centroid(), (PLIC::GeometryPoint{0, 0, 0}));
    EXPECT_FALSE(detail::cell_catalog_detail::same_native_representation(stretched, cube));
    EXPECT_TRUE(detail::cell_catalog_detail::same_native_representation(cube, equal_cube));
    EXPECT_NE(stretched.corner(0), cube.corner(0));
    EXPECT_NE(stretched.source_support(0), cube.source_support(0));
    const std::array<std::uint64_t, 1> ordinal{0};
    const auto a = detached(cube_grid(), ordinal), b = detached(cube_grid(), ordinal);
    auto different_grid = cube_grid();
    different_grid.origin = {-1, -.25, -.5};
    different_grid.extent = {4, 1, 2};
    const auto different = detached(different_grid, ordinal);
    ASSERT_EQ(a.owned.front().key, different.owned.front().key);
    ASSERT_EQ(a.owned.front().revision, different.owned.front().revision);
    EXPECT_TRUE(detail::cell_catalog_detail::same_native_recipe(a.owned.front(), b.owned.front()));
    EXPECT_FALSE(detail::cell_catalog_detail::same_native_recipe(a.owned.front(), different.owned.front()));
}

TEST(GeneralVoFNativeIdentity, EqualAuthenticatedRecipesKeepTheOwnershipRemap)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused structural proof oracle is serial.";
    PLIC::DynamicCellGeometryView surviving;
    std::weak_ptr<const void> source_lifetime;
    PLIC::GeometryPoint expected_centroid{};
    {
        const auto source = cube_state();
        source_lifetime = source->accepted_identity();
        const auto plan = identity_plan(*source);
        const auto target = bound_candidate(*source, plan);
        const auto trial = source->plan_remap(plan, target);
        ASSERT_TRUE(trial.ownership_only());
        ASSERT_EQ(trial.owned_routes().size(), source->owned_order().size());
        const auto candidate = trial.candidate();
        EXPECT_EQ(candidate->gas_volumes(), source->gas_volumes());
        EXPECT_EQ(candidate->carrier_volumes(), source->carrier_volumes());
        const auto& common = trial.source_common_cells();
        ASSERT_EQ(common.size(), source->owned_order().size());
        std::map<GeneralVoF::Key, View> original;
        const auto source_geometry = source->owned_geometry();
        for (std::size_t index = 0; index < source->owned_order().size(); ++index)
            original.emplace(source->owned_order()[index], source_geometry[index]);
        auto expected = source->hierarchy()->owned_leaves().begin();
        for (const auto& record : common)
        {
            ASSERT_NE(expected, source->hierarchy()->owned_leaves().end());
            EXPECT_EQ(record.source_key, expected->key);
            EXPECT_EQ(record.target_key, expected->key);
            ++expected;
            ASSERT_TRUE(record.geometry);
            const auto view = record.geometry.get<mode>();
            EXPECT_TRUE(view.is_procedural());
            EXPECT_EQ(view.volume().value, 1.L);
            const auto expected_ref =
                PLIC::details::NativeCellGeometrySourceAccess::source_cell(original.at(record.source_key));
            const auto actual_ref = PLIC::details::NativeCellGeometrySourceAccess::source_cell(view);
            EXPECT_EQ(actual_ref.source, expected_ref.source);
            EXPECT_EQ(actual_ref.cell_slot, expected_ref.cell_slot);
            EXPECT_EQ(view.centroid(), original.at(record.source_key).centroid());
        }
        surviving = common.front().geometry;
        expected_centroid = surviving.get<mode>().centroid();
        for (const auto view : candidate->owned_geometry())
            EXPECT_TRUE(view.is_procedural());
    }
    EXPECT_TRUE(source_lifetime.expired());
    ASSERT_TRUE(surviving);
    EXPECT_EQ(surviving.get<mode>().volume().value, 1.L);
    EXPECT_EQ(surviving.get<mode>().centroid(), expected_centroid);
}

TEST(GeneralVoFNativeIdentity, BoundedConstructionRetainsOriginalAndSelectedCatalogs)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused catalog mode oracle is serial.";
    const auto source = cube_state(ArithmeticMode::BoundedCertification);
    const auto original = source->owned_geometry();
    const auto selected = source->owned_finite_geometry<ArithmeticMode::BoundedCertification>();
    ASSERT_EQ(original.size(), 8u);
    ASSERT_EQ(selected.size(), original.size());
    for (std::size_t index = 0; index < original.size(); ++index)
    {
        EXPECT_FALSE(original[index].is_procedural());
        EXPECT_FALSE(selected[index].is_procedural());
        EXPECT_EQ(original[index].volume().value, 1.L);
        const auto volume = selected[index].volume();
        EXPECT_LE(volume.lower(), 1.L);
        EXPECT_GE(volume.upper(), 1.L);
        const auto old_owner = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(original[index]);
        const auto selected_owner = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(selected[index]);
        EXPECT_NE(old_owner->storage_identity(), selected_owner->storage_identity());
    }
}

TEST(GeneralVoFNativeIdentity, RegularPublicationAndQueriesRetainNoGeneralCellGeometry)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused accepted-owner closure oracle is serial.";
    const auto source = cube_state();
    for (const auto view : source->owned_geometry())
    {
        ASSERT_TRUE(view.is_procedural());
        EXPECT_EQ(view.volume().value, 1.L);
        static_cast<void>(view.moments());
        static_cast<void>(view.centroid());
        static_cast<void>(view.axis_spans());
        static_cast<void>(view.closed_contact_bounds());
        for (const auto face : view.faces())
        {
            static_cast<void>(face.area());
            static_cast<void>(face.support());
        }
    }
    for (const auto& cell : source->support_cells())
    {
        EXPECT_TRUE(cell.geometry.is_procedural());
        EXPECT_TRUE(cell.finite_geometry.get<mode>().is_procedural());
    }
    const auto accepted = source->accepted_identity();
    const auto owners = accepted.use_count();
    const auto rounding = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
    std::vector<std::byte> memory(8 * 1024 * 1024);
    PLIC::details::retention::ProbeScratch scratch(memory);
    detail::RetentionRoots roots(scratch);
    roots.state(*source);
    const auto report = PLIC::details::retention::inspect_retention(
        roots.roots, roots.ledgers, scratch, {.complete_root_manifest = roots.complete_manifest});
    ASSERT_TRUE(report.complete()) << report.detail;
    ASSERT_TRUE(report.complete_root_manifest);
    std::size_t catalogs = 0, contexts = 0;
    for (const auto& owner : report.owners)
    {
        const std::string_view kind(owner.kind);
        EXPECT_NE(kind, "geometry");
        EXPECT_NE(kind, "topology");
        EXPECT_NE(kind, "plane");
        EXPECT_EQ(kind.find("export"), std::string_view::npos) << kind;
        catalogs += kind == "cell_catalog";
        contexts += kind == "cartesian_source";
    }
    EXPECT_GT(catalogs, 0u);
    EXPECT_GT(contexts, 0u);
    EXPECT_EQ(accepted.use_count(), owners);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
}

TEST(GeneralVoFNativeIdentity, EqualMetricCatalogMismatchStillReachesOriginalRemapCoverageGate)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused structural proof oracle is serial.";
    const auto source = cube_state();
    const auto plan = identity_plan(*source);
    const auto target = bound_candidate(*source, plan);
    const auto* accepted = detail::RetentionAccess::data(*source);
    auto* topology = const_cast<GeneralVoF::Data::Topology*>(accepted->topology.get());
    struct Restore
    {
        GeneralVoF::Data::Topology* topology;
        std::shared_ptr<const Catalog> catalog;
        ~Restore() { topology->cell_geometry = std::move(catalog); }
    } restore{topology, topology->cell_geometry};
    // Deliberate private adversarial injection: keep key/revision/owner/GID,
    // native volume and centroid unchanged, but replace one represented shape.
    // This cannot be issued by the public immutable publication API.
    Catalog::SourceTable sources;
    std::map<const PLIC::CartesianCellSource*, std::size_t> source_slots;
    std::vector<Catalog::Entry> entries(topology->cell_geometry->size());
    for (const auto& [key, cell] : accepted->cells)
    {
        const auto ref =
            PLIC::details::NativeCellGeometrySourceAccess::source_cell(detail::native_cell_geometry(*accepted, cell));
        const auto [found, inserted] = source_slots.emplace(ref.source.get(), sources.size());
        if (inserted)
            sources.push_back(ref.source);
        entries.at(cell.geometry_slot) = PLIC::NativeCellGeometrySlot{found->second, ref.cell_slot};
    }
    const auto stretched = box_view({-1, -.25, -.5}, {1, .25, .5});
    const auto replacement = PLIC::details::NativeCellGeometrySourceAccess::source_cell(stretched);
    const auto& cell = accepted->cells.at(source->owned_order().front());
    const auto original = detail::native_cell_geometry(*accepted, cell);
    ASSERT_EQ(original.centroid(), (PLIC::GeometryPoint{0, 0, 0}));
    ASSERT_EQ(original.volume().value, stretched.volume().value);
    ASSERT_FALSE(detail::cell_catalog_detail::same_native_representation(original, stretched));
    entries.at(cell.geometry_slot) = PLIC::NativeCellGeometrySlot{sources.size(), replacement.cell_slot};
    sources.push_back(replacement.source);
    topology->cell_geometry = Catalog::make(std::move(sources), std::move(entries), limits());
    const auto identity = source->accepted_identity();
    const auto control = source->accepted_control();
    const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
    try
    {
        (void) source->plan_remap(plan, target);
        FAIL() << "Different shapes with equal volume/centroid bypassed remap coverage.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("partition piece escapes its parent"), std::string::npos)
            << "Mismatch must be rejected by the original geometric coverage gate: " << error.what();
    }
    EXPECT_EQ(source->accepted_identity(), identity);
    EXPECT_EQ(source->accepted_control(), control);
    EXPECT_EQ(source->gas_volumes(), gas);
    EXPECT_EQ(source->carrier_volumes(), carrier);
}
} // namespace

namespace
{
TEST(GeneralVoFNativeIdentity, MeshSnapshotSharesActualSourcesAcrossStatesAndSelections)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused mesh snapshot ownership oracle is serial.";
    const auto first = cube_state();
    auto reverse = first->owned_order();
    std::reverse(reverse.begin(), reverse.end());
    GeneralVoFOptions options;
    options.phase_cleanup.relative_cutoff = 0;
    const GeneralVoF second(first->hierarchy(), reverse, std::vector<double>(8, 0), std::vector<double>(8, 1), options);
    std::map<GeneralVoF::Key, PLIC::details::NativeCartesianCellReference> sources;
    const auto first_geometry = first->owned_geometry();
    for (std::size_t i = 0; i < first->owned_order().size(); ++i)
        sources.emplace(
            first->owned_order()[i], PLIC::details::NativeCellGeometrySourceAccess::source_cell(first_geometry[i]));
    const auto second_geometry = second.owned_geometry();
    for (std::size_t i = 0; i < second.owned_order().size(); ++i)
    {
        const auto actual = PLIC::details::NativeCellGeometrySourceAccess::source_cell(second_geometry[i]);
        const auto& expected = sources.at(second.owned_order()[i]);
        EXPECT_EQ(actual.source, expected.source);
        EXPECT_EQ(actual.cell_slot, expected.cell_slot);
    }
    const auto& hierarchy = *first->hierarchy();
    const std::array<const AMR::AdaptiveLeaf*, 2> selected{
        &hierarchy.owned_leaves().at(reverse.front()), &hierarchy.owned_leaves().at(reverse.back())};
    detail::wire::Budget budget{1024 * 1024};
    const auto catalog = detail::cell_catalog_detail::build<mode>(hierarchy, selected, budget);
    for (std::size_t i = 0; i < selected.size(); ++i)
    {
        const auto actual = PLIC::details::NativeCellGeometrySourceAccess::source_cell(View(catalog.ordinary, i));
        EXPECT_EQ(actual.source, sources.at(selected[i]->key).source);
        EXPECT_EQ(actual.cell_slot, sources.at(selected[i]->key).cell_slot);
    }
    const auto independent = cube_state();
    const auto independent_source =
        PLIC::details::NativeCellGeometrySourceAccess::source_cell(independent->owned_geometry()[0]);
    EXPECT_NE(independent_source.source, sources.at(independent->owned_order()[0]).source);
    EXPECT_NE(
        independent_source.source->base_identity(), sources.at(independent->owned_order()[0]).source->base_identity());
}

TEST(GeneralVoFNativeIdentity, MeshSnapshotExpansionIsStableAndOldViewsOutliveHierarchy)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused detached geometry snapshot oracle is serial.";
    View surviving;
    std::weak_ptr<const GeneralVoF::Hierarchy> lifetime;
    {
        const std::array<std::uint64_t, 8> ordinals{0, 1, 2, 3, 4, 5, 6, 7};
        auto input = detached(cube_grid(), ordinals);
        auto hierarchy = std::make_shared<const GeneralVoF::Hierarchy>(
            std::move(input.owned), std::move(input.families), input.kernel, Tpetra::getDefaultComm());
        lifetime = hierarchy;
        const auto* first = &*hierarchy->owned_leaves().begin();
        const std::array<const AMR::AdaptiveLeaf*, 1> subset{first};
        std::size_t used = 0;
        auto initial =
            hierarchy->geometry_snapshot(subset, mode, AMR::GeometrySnapshotPurpose::State, used, 1024 * 1024);
        surviving = View(initial->ordinary_catalog(), *initial->find(*first));
        const auto before = PLIC::details::NativeCellGeometrySourceAccess::source_cell(surviving);
        std::vector<const AMR::AdaptiveLeaf*> full;
        for (const auto& leaf : hierarchy->owned_leaves())
            full.push_back(&leaf);
        used = 0;
        const auto expanded =
            hierarchy->geometry_snapshot(full, mode, AMR::GeometrySnapshotPurpose::State, used, 1024 * 1024);
        const auto after = PLIC::details::NativeCellGeometrySourceAccess::source_cell(
            View(expanded->ordinary_catalog(), *expanded->find(*first)));
        EXPECT_EQ(before.source, after.source);
        EXPECT_EQ(before.cell_slot, after.cell_slot);
        auto stale = *first;
        ++stale.revision;
        const std::array<const AMR::AdaptiveLeaf*, 1> stale_rows{&stale};
        used = 0;
        EXPECT_THROW(
            hierarchy->geometry_snapshot(stale_rows, mode, AMR::GeometrySnapshotPurpose::State, used, 1024 * 1024),
            std::invalid_argument);
        EXPECT_EQ(used, 0u);
        EXPECT_EQ(surviving.volume().value, 1.L);
    }
    EXPECT_TRUE(lifetime.expired());
    EXPECT_EQ(surviving.volume().value, 1.L);
    EXPECT_EQ(surviving.num_faces(), 6u);
}

TEST(GeneralVoFNativeIdentity, MeshSnapshotHotAdmissionHasExactBoundaryAndAtomicFailure)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused snapshot admission oracle is serial.";
    const auto state = cube_state();
    const auto& hierarchy = *state->hierarchy();
    const std::array<const AMR::AdaptiveLeaf*, 1> rows{&*hierarchy.owned_leaves().begin()};
    std::size_t required = 0;
    const auto expected =
        hierarchy.geometry_snapshot(rows, mode, AMR::GeometrySnapshotPurpose::State, required, 1024 * 1024);
    ASSERT_GT(required, 0u);
    std::size_t exact = 0;
    EXPECT_EQ(hierarchy.geometry_snapshot(rows, mode, AMR::GeometrySnapshotPurpose::State, exact, required), expected);
    EXPECT_EQ(exact, required);
    std::size_t short_used = 0;
    EXPECT_THROW(hierarchy.geometry_snapshot(rows, mode, AMR::GeometrySnapshotPurpose::State, short_used, required - 1),
        std::exception);
    EXPECT_GT(short_used, 0u);
    EXPECT_LE(short_used, required - 1);
    std::size_t retry = 0;
    EXPECT_EQ(hierarchy.geometry_snapshot(rows, mode, AMR::GeometrySnapshotPurpose::State, retry, required), expected);
}
} // namespace

namespace
{
TEST(GeneralVoFNativeIdentity, BoundedFallbackEvaluationsKeepIndependentFirstQueryControls)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused lazy fallback evaluation oracle is serial.";
    struct EnvironmentRestore
    {
        std::fenv_t saved{};
        EnvironmentRestore() { std::fegetenv(&saved); }
        ~EnvironmentRestore() { std::fesetenv(&saved); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    auto grid = cube_grid();
    grid.origin = {-.7, .2, -.9};
    grid.extent = {2.3, 5.1, 7.7};
    const std::array<std::uint64_t, 8> ordinals{0, 1, 2, 3, 4, 5, 6, 7};
    auto input = detached(grid, ordinals);
    auto hierarchy = std::make_shared<const GeneralVoF::Hierarchy>(
        std::move(input.owned), std::move(input.families), input.kernel, Tpetra::getDefaultComm());
    std::vector<GeneralVoF::Key> order;
    std::vector<double> carrier;
    for (const auto& leaf : hierarchy->owned_leaves())
    {
        order.push_back(leaf.key);
        carrier.push_back(hierarchy->metrics(leaf).volume);
    }
    GeneralVoFOptions options;
    options.arithmetic_mode = ArithmeticMode::BoundedCertification;
    options.phase_cleanup.relative_cutoff = 0;
    const GeneralVoF first(hierarchy, order, std::vector<double>(8, 0), carrier, options);
    std::vector<const AMR::AdaptiveLeaf*> rows;
    for (const auto& value : hierarchy->owned_leaves())
        rows.push_back(&value);
    std::size_t admitted = 0;
    const auto mesh_snapshot = hierarchy->geometry_snapshot(
        rows, ArithmeticMode::BoundedCertification, AMR::GeometrySnapshotPurpose::State, admitted, 4 * 1024 * 1024);
    // A direct immutable snapshot query may fill a numerical cache after its
    // construction policy was captured. It must not seed the next evaluation.
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    static_cast<void>(View(mesh_snapshot->ordinary_catalog(), 0).moments());
    const PLIC::CellGeometryView<ArithmeticMode::BoundedCertification> bounded_template(
        mesh_snapshot->bounded_catalog(), 0);
    EXPECT_THROW(static_cast<void>(bounded_template.moments()), std::invalid_argument);
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    static_cast<void>(bounded_template.moments());
    const GeneralVoF second(hierarchy, order, std::vector<double>(8, 0), carrier, options);
    const auto& leaf = hierarchy->owned_leaves().at(first.owned_order()[0]);
    auto raw_up = hierarchy->geometry(leaf), raw_down = hierarchy->geometry(leaf);
    auto bounded_up = PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>::from_geometry(*raw_up);
    auto bounded_down = PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>::from_geometry(*raw_down);
    // The original full-catalog consistency check certifies each Ordinary
    // geometry against its reused source under constructor rounding. That
    // check prepares its moments even when both handles name the same input.
    ASSERT_TRUE(raw_up->equivalent(*raw_up));
    ASSERT_TRUE(raw_down->equivalent(*raw_down));
    const auto original_first = first.owned_geometry()[0], original_second = second.owned_geometry()[0];
    const auto selected_first = first.owned_finite_geometry<ArithmeticMode::BoundedCertification>()[0];
    const auto selected_second = second.owned_finite_geometry<ArithmeticMode::BoundedCertification>()[0];
    EXPECT_NE(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(original_first),
        PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(original_second));
    EXPECT_NE(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(selected_first),
        PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(selected_second));
    const auto same_moments = [](const auto& actual, const auto& expected)
    {
        const auto same_value = [](const auto& value, const auto& reference)
        {
            EXPECT_EQ(value.value, reference.value);
            EXPECT_EQ(std::signbit(value.value), std::signbit(reference.value));
            if constexpr (requires { value.absolute_error; })
            {
                EXPECT_EQ(value.absolute_error, reference.absolute_error);
                EXPECT_EQ(std::signbit(value.absolute_error), std::signbit(reference.absolute_error));
            }
        };
        same_value(actual.volume, expected.volume);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            EXPECT_EQ(actual.origin[axis], expected.origin[axis]);
            EXPECT_EQ(std::signbit(actual.origin[axis]), std::signbit(expected.origin[axis]));
            same_value(actual.first_moment[axis], expected.first_moment[axis]);
        }
    };
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    same_moments(original_first.moments(), raw_up->moments());
    EXPECT_THROW(static_cast<void>(selected_first.moments()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bounded_up->moments()), std::invalid_argument);
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    same_moments(original_second.moments(), raw_down->moments());
    EXPECT_THROW(static_cast<void>(selected_second.moments()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bounded_down->moments()), std::invalid_argument);
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    // Bounded arithmetic supports nearest rounding only. Rejected directed
    // queries must not populate a cache; later template priming must not seed
    // either independent evaluation.
    EXPECT_EQ(PLIC::details::cell_evaluation_cache_ready(
                  PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(selected_first)),
        PLIC::details::cell_evaluation_cache_ready(bounded_up));
    EXPECT_EQ(PLIC::details::cell_evaluation_cache_ready(
                  PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(selected_second)),
        PLIC::details::cell_evaluation_cache_ready(bounded_down));
    same_moments(selected_first.moments(), bounded_up->moments());
    same_moments(selected_second.moments(), bounded_down->moments());
}
} // namespace

namespace
{
TEST(GeneralVoFNativeIdentity, MixedPublicationDraftPreservesOriginalGeometryAndCompatibleMeshOwners)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused mixed publication geometry oracle is serial.";
    const AMR::MixedAMR mesh(SimpleFluid::test::mixed_joined_grid());
    std::vector<std::uint64_t> owned;
    for (std::uint64_t cell = 0; cell < mesh.geometry()->num_cells(); ++cell)
        owned.push_back(cell);
    auto input = mesh.detach_owned(owned, {'m', 'i', 'x', 'e', 'd', '-', 's', 'n', 'a', 'p'});
    auto hierarchy = std::make_shared<const GeneralVoF::Hierarchy>(
        std::move(input.owned), std::move(input.families), input.kernel, Tpetra::getDefaultComm());
    std::map<GeneralVoF::Key, PLIC::ConvexGeometryPtr<mode>> initial;
    for (const auto& leaf : hierarchy->owned_leaves())
        initial.emplace(leaf.key, hierarchy->geometry(leaf));
    const auto draft = AMR::adaptive_detail::fetch_compact_adaptive_native_draft(*hierarchy);
    // This fixture's compatible Product rows reused their initial provider
    // geometry in the original constructor. Polyhedral publication rows
    // already own the mapped mesh geometry; applying the provider again would
    // incorrectly map their physical surfaces a second time.
    std::map<GeneralVoF::Key, PLIC::ConvexGeometryPtr<mode>> expected;
    for (const auto& leaf : draft.owned)
    {
        if (leaf.provider == AMR::AdaptiveProviderKind::Product)
        {
            const auto& provider = hierarchy->owned_leaves().at(leaf.key);
            ASSERT_EQ(leaf.revision, provider.revision);
            ASSERT_EQ(leaf.material, provider.material);
            ASSERT_EQ(leaf.domain, provider.domain);
            ASSERT_EQ(leaf.native_cell, provider.native_cell);
            ASSERT_TRUE(leaf.native_recipe);
            ASSERT_TRUE(provider.native_recipe);
            ASSERT_EQ(leaf.native_recipe->kind, provider.native_recipe->kind);
            ASSERT_EQ(leaf.native_recipe->view, provider.native_recipe->view);
            ASSERT_EQ(leaf.native_recipe->parts.size(), provider.native_recipe->parts.size());
            for (std::size_t part = 0; part < leaf.native_recipe->parts.size(); ++part)
            {
                ASSERT_TRUE(leaf.native_recipe->parts[part]);
                ASSERT_TRUE(provider.native_recipe->parts[part]);
                ASSERT_EQ(*leaf.native_recipe->parts[part], *provider.native_recipe->parts[part]);
            }
            expected.emplace(leaf.key, initial.at(leaf.key));
        }
        else
        {
            ASSERT_EQ(leaf.provider, AMR::AdaptiveProviderKind::Polyhedral);
            ASSERT_TRUE(leaf.geometry);
            expected.emplace(leaf.key, leaf.geometry);
        }
        ASSERT_TRUE(initial.at(leaf.key)->equivalent(*expected.at(leaf.key)));
    }
    std::vector<GeneralVoF::Key> order;
    std::vector<double> carrier;
    for (const auto& leaf : hierarchy->owned_leaves())
    {
        order.push_back(leaf.key);
        carrier.push_back(hierarchy->metrics(leaf).volume);
    }
    GeneralVoFOptions options;
    options.phase_cleanup.relative_cutoff = 0;
    const GeneralVoF first(hierarchy, order, std::vector<double>(order.size(), 0), carrier, options);
    const GeneralVoF second(hierarchy, order, std::vector<double>(order.size(), 0), carrier, options);
    const auto* first_data = detail::RetentionAccess::data(first);
    const auto* second_data = detail::RetentionAccess::data(second);
    const auto geometry = first.owned_geometry();
    for (std::size_t cell = 0; cell < first.owned_order().size(); ++cell)
    {
        const auto baseline = View::from_existing(expected.at(first.owned_order()[cell]));
        const auto actual = geometry[cell];
        if (hierarchy->owned_leaves().at(first.owned_order()[cell]).provider == AMR::AdaptiveProviderKind::Polyhedral)
        {
            const auto owner = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(actual);
            const auto& catalog = first_data->topology->geometry_snapshot->ordinary_catalog();
            bool shared_publication_owner = false;
            for (std::size_t slot = 0; slot < catalog->size(); ++slot)
            {
                const View source(catalog, slot);
                if (!source.is_procedural() &&
                    PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(source) == owner)
                    shared_publication_owner = true;
            }
            EXPECT_TRUE(shared_publication_owner);
        }
        EXPECT_EQ(actual.volume().value, baseline.volume().value);
        ASSERT_EQ(actual.vertex_count(), baseline.vertex_count());
        ASSERT_EQ(actual.num_faces(), baseline.num_faces());
        for (std::size_t vertex = 0; vertex < actual.vertex_count(); ++vertex)
            EXPECT_EQ(actual.corner(vertex), baseline.corner(vertex));
        for (std::size_t face = 0; face < actual.num_faces(); ++face)
        {
            EXPECT_EQ(actual.source_support(face), baseline.source_support(face));
            const auto a = actual.ordered_face(face), b = baseline.ordered_face(face);
            ASSERT_EQ(a.size(), b.size());
            for (std::size_t point = 0; point < a.size(); ++point)
                EXPECT_EQ(a[point], b[point]);
        }
    }
    bool checked_compatible_owner = false;
    for (const auto& leaf : draft.owned)
        if (leaf.provider == AMR::AdaptiveProviderKind::Product && !leaf.geometry)
        {
            const auto& first_snapshot = *first_data->topology->geometry_snapshot;
            const auto& second_snapshot = *second_data->topology->geometry_snapshot;
            const auto a = first_snapshot.find(leaf), b = second_snapshot.find(leaf);
            ASSERT_TRUE(a);
            ASSERT_TRUE(b);
            EXPECT_EQ(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                          View(first_snapshot.ordinary_catalog(), *a)),
                PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                    View(second_snapshot.ordinary_catalog(), *b)));
            checked_compatible_owner = true;
        }
    EXPECT_TRUE(checked_compatible_owner);
}
} // namespace

namespace
{
TEST(GeneralVoFNativeIdentity, ConstructorLocalFallbackEvaluationsRequireTheOriginalOperationAuthority)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused constructor-local evaluation authority oracle is serial.";
    struct EnvironmentRestore
    {
        std::fenv_t saved{};
        EnvironmentRestore() { std::fegetenv(&saved); }
        ~EnvironmentRestore() { std::fesetenv(&saved); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    namespace catalog = detail::cell_catalog_detail;
    if (!catalog::construction_controls().supported || !PLIC::certification_environment())
        GTEST_SKIP() << "Named construction controls and Bounded FP certification are required.";
    constexpr auto bounded_mode = ArithmeticMode::BoundedCertification;
    const auto accepted = cube_state(bounded_mode);
    const auto hierarchy = accepted->hierarchy();
    std::vector<const AMR::AdaptiveLeaf*> owned;
    for (const auto& leaf : hierarchy->owned_leaves())
        owned.push_back(&leaf);
    const unsigned operation = 0, other_operation = 0;
    detail::wire::Budget initial_budget{16 * 1024 * 1024};
    const auto initial = catalog::build<bounded_mode>(
        *hierarchy, owned, initial_budget, nullptr, catalog::CatalogPurpose::State, nullptr, nullptr, &operation);
    ASSERT_TRUE(initial.reuse_authority.stable);
    EXPECT_EQ(initial.reuse_authority.operation, &operation);
    EXPECT_EQ(initial.reuse_authority.original_catalog, initial.ordinary.get());
    EXPECT_EQ(initial.reuse_authority.source_snapshot, initial.snapshot.get());
    GeneralVoF::Data reuse;
    auto topology = std::make_shared<GeneralVoF::Data::Topology>();
    reuse.hierarchy = hierarchy;
    reuse.options.arithmetic_mode = bounded_mode;
    reuse.topology = topology;
    topology->geometry_snapshot = initial.snapshot;
    topology->cell_geometry = initial.ordinary;
    topology->ordinary_cell_geometry = initial.finite_ordinary;
    topology->bounded_cell_geometry = initial.bounded;
    for (std::size_t slot = 0; slot < owned.size(); ++slot)
    {
        detail::Cell cell;
        cell.key = owned[slot]->key;
        cell.revision = owned[slot]->revision;
        cell.owner = hierarchy->communicator()->getRank();
        cell.geometry_slot = slot;
        reuse.cells.emplace(cell.key, std::move(cell));
    }
    const auto draft = AMR::adaptive_detail::fetch_compact_adaptive_native_draft(*hierarchy);
    const auto observation = hierarchy->observation();
    std::vector<const AMR::AdaptiveLeaf*> full;
    for (auto leaf = draft.owned.rbegin(); leaf != draft.owned.rend(); ++leaf)
        full.push_back(&*leaf);
    detail::wire::Budget full_budget{16 * 1024 * 1024};
    const auto expanded = catalog::build<bounded_mode>(*hierarchy, full, full_budget, &reuse,
        catalog::CatalogPurpose::State, nullptr, &initial.reuse_authority, &operation, &draft, &observation);
    for (std::size_t slot = 0; slot < full.size(); ++slot)
    {
        const auto previous_slot = reuse.cells.at(full[slot]->key).geometry_slot;
        EXPECT_EQ(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(View(expanded.ordinary, slot)),
            PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(View(initial.ordinary, previous_slot)));
        EXPECT_EQ(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(expanded.view<bounded_mode>(slot)),
            PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                initial.view<bounded_mode>(previous_slot)));
    }
    detail::wire::Budget independent_budget{16 * 1024 * 1024};
    const auto independent = catalog::build<bounded_mode>(*hierarchy, owned, independent_budget, nullptr,
        catalog::CatalogPurpose::State, nullptr, nullptr, &other_operation);
    EXPECT_NE(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(independent.view<bounded_mode>(0)),
        PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(initial.view<bounded_mode>(0)));
    for (const bool wrong_operation : {true, false})
    {
        auto authority = initial.reuse_authority;
        if (!wrong_operation)
            authority.stable = false;
        detail::wire::Budget rejected_budget{16 * 1024 * 1024};
        const auto rejected = catalog::build<bounded_mode>(*hierarchy, owned, rejected_budget, &reuse,
            catalog::CatalogPurpose::State, nullptr, &authority, wrong_operation ? &other_operation : &operation);
        EXPECT_NE(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(rejected.view<bounded_mode>(0)),
            PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(initial.view<bounded_mode>(0)));
    }
    // One accepted initial/full evaluation integrates once. An independent
    // acquisition integrates its own cache, preserving the old constructor's
    // moment work without sharing caller-dependent state across acquisitions.
    PLIC::reset_geometry_statistics();
    static_cast<void>(initial.view<bounded_mode>(0).moments());
    const auto first_moments = PLIC::geometry_statistics().moment_evaluations;
    ASSERT_GT(first_moments, 0u);
    const auto expanded_slot =
        std::find_if(full.begin(), full.end(), [&](const auto* leaf) { return leaf->key == owned[0]->key; }) -
        full.begin();
    static_cast<void>(expanded.view<bounded_mode>(static_cast<std::size_t>(expanded_slot)).moments());
    EXPECT_EQ(PLIC::geometry_statistics().moment_evaluations, first_moments);
    static_cast<void>(independent.view<bounded_mode>(0).moments());
    EXPECT_GT(PLIC::geometry_statistics().moment_evaluations, first_moments);
}
} // namespace

namespace
{
TEST(GeneralVoFNativeIdentity, PrivatePublicationSnapshotAdoptsOriginalOwnerAndRequiresItsWitness)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused private publication ownership oracle is serial.";
    const AMR::MixedAMR mesh(SimpleFluid::test::mixed_joined_grid());
    std::vector<std::uint64_t> owned;
    for (std::uint64_t cell = 0; cell < mesh.geometry()->num_cells(); ++cell)
        owned.push_back(cell);
    auto input = mesh.detach_owned(owned, {'p', 'u', 'b', 'l', 'i', 's', 'h', 'e', 'd'});
    const auto hierarchy = std::make_shared<const GeneralVoF::Hierarchy>(
        std::move(input.owned), std::move(input.families), input.kernel, Tpetra::getDefaultComm());
    const auto draft = AMR::adaptive_detail::fetch_compact_adaptive_native_draft(*hierarchy);
    const auto observation = hierarchy->observation();
    std::vector<const AMR::AdaptiveLeaf*> rows;
    for (const auto& leaf : draft.owned)
        rows.push_back(&leaf);
    std::size_t used = 0;
    EXPECT_THROW(
        (void) hierarchy->geometry_snapshot(rows, mode, AMR::GeometrySnapshotPurpose::State, used, 16 * 1024 * 1024),
        std::invalid_argument);
    detail::wire::Budget missing_witness{16 * 1024 * 1024};
    EXPECT_THROW((void) detail::cell_catalog_detail::build<mode>(*hierarchy, rows, missing_witness, nullptr,
                     detail::cell_catalog_detail::CatalogPurpose::State, nullptr, nullptr, nullptr, &draft, nullptr),
        std::invalid_argument);
    using Access = AMR::adaptive_detail::GeometrySnapshotAccess;
    const auto reject = [&](const AMR::AdaptiveNativeDraft& rejected_draft,
                            std::span<const AMR::AdaptiveLeaf* const> rejected_rows,
                            const AMR::HierarchyObservation& rejected_observation)
    {
        PLIC::GeometryAllocationLedger original, selected;
        std::size_t rejected_used = 0;
        EXPECT_THROW((void) Access::native_draft(*hierarchy, rejected_draft, rejected_rows, rejected_observation, mode,
                         AMR::GeometrySnapshotPurpose::State, rejected_used, 16 * 1024 * 1024, original, selected),
            std::invalid_argument);
    };
    auto stale = observation;
    ++stale.geometry;
    reject(draft, rows, stale);
    auto missing_binding = draft;
    missing_binding.cell_bindings.clear();
    std::vector<const AMR::AdaptiveLeaf*> unbound_rows;
    for (const auto& leaf : missing_binding.owned)
        unbound_rows.push_back(&leaf);
    reject(missing_binding, unbound_rows, observation);
    const auto copied_row = draft.owned.front();
    const std::array<const AMR::AdaptiveLeaf*, 1> outside{&copied_row};
    reject(draft, outside, observation);
    PLIC::GeometryAllocationLedger original, selected;
    used = 0;
    const auto snapshot = Access::native_draft(*hierarchy, draft, rows, observation, mode,
        AMR::GeometrySnapshotPurpose::State, used, 16 * 1024 * 1024, original, selected);
    bool checked_explicit = false;
    for (const auto& leaf : draft.owned)
        if (leaf.provider == AMR::AdaptiveProviderKind::Polyhedral)
        {
            ASSERT_TRUE(leaf.geometry);
            const auto slot = snapshot->find(leaf);
            ASSERT_TRUE(slot);
            EXPECT_EQ(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                          View(snapshot->ordinary_catalog(), *slot)),
                leaf.geometry);
            ASSERT_TRUE(snapshot->finite_ordinary_catalog());
            EXPECT_NE(PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(
                          View(snapshot->finite_ordinary_catalog(), *slot)),
                leaf.geometry);
            const auto binding = std::find_if(draft.cell_bindings.begin(), draft.cell_bindings.end(),
                [&](const auto& value) { return value.key == leaf.key && value.full; });
            ASSERT_NE(binding, draft.cell_bindings.end());
            EXPECT_EQ(snapshot->native_metrics()[*slot].volume, binding->volume);
            EXPECT_EQ(snapshot->native_metrics()[*slot].centroid, binding->centroid);
            checked_explicit = true;
        }
    EXPECT_TRUE(checked_explicit);
    std::size_t cached = 0;
    hierarchy->visit_geometry_snapshots([&](const auto&) { ++cached; });
    EXPECT_EQ(cached, 0u) << "Derived publication representations must not enter the provider-input cache.";
}
} // namespace
