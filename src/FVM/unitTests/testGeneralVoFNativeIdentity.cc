/** @file testGeneralVoFNativeIdentity.cc
 * @brief Independent equal-metric shape counterexample and original remap coverage gate.
 */
#include "FVM/details/GeneralVoF/CellCatalog.hh"
#include "FVM/details/GeneralVoF/Retention.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "utils/testing_environment.hh"
#include <cfenv>
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
    for (const auto& [key, leaf] : hierarchy->owned())
        order.push_back(key);
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
        auto expected = source->hierarchy()->owned().begin();
        for (const auto& record : common)
        {
            ASSERT_NE(expected, source->hierarchy()->owned().end());
            EXPECT_EQ(record.source_key, expected->first);
            EXPECT_EQ(record.target_key, expected->first);
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
