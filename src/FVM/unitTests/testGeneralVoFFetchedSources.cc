/** @file testGeneralVoFFetchedSources.cc
 * @brief Source-only original-owner, immutable lifetime and collective cap probes.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/details/GeneralVoF/CellCatalog.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "utils/testing_environment.hh"
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
namespace AMR = Meshes::AMR;
namespace catalog = general_vof_detail::cell_catalog_detail;
constexpr auto ordinary = ArithmeticMode::Ordinary;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
AMR::AMRGrid grid()
{
    AMR::AMRGrid result;
    result.block_cells = {2, 2, 2};
    result.extent = {2, 2, 2};
    return result;
}
AMR::AdaptiveBootstrap detached(std::span<const std::uint64_t> ordinals)
{
    AMR::CartesianAMR mesh(grid());
    auto result = mesh.detach_owned(ordinals, {'f', 'e', 't', 'c', 'h'});
    for (const auto& leaf : result.owned)
        result.kernel->validate(leaf);
    return result;
}
std::shared_ptr<const GeneralVoF> state(ArithmeticMode mode = ordinary)
{
    const std::array<std::uint64_t, 8> ordinals{0, 1, 2, 3, 4, 5, 6, 7};
    auto input = detached(ordinals);
    auto hierarchy = std::make_shared<const GeneralVoF::Hierarchy>(
        std::move(input.owned), std::move(input.families), input.kernel, Tpetra::getDefaultComm());
    std::vector<GeneralVoF::Key> order;
    for (const auto& [key, leaf] : hierarchy->owned())
        order.push_back(key);
    GeneralVoFOptions options;
    options.arithmetic_mode = mode;
    options.phase_cleanup.relative_cutoff = 0;
    return std::make_shared<const GeneralVoF>(
        hierarchy, order, std::vector<double>(order.size(), 0), std::vector<double>(order.size(), 1), options);
}
class EnvironmentRestore
{
    std::fenv_t environment_{};

public:
    EnvironmentRestore()
    {
        if (std::fegetenv(&environment_) != 0)
            throw std::runtime_error("Cannot capture fenv.");
    }
    ~EnvironmentRestore() { (void) std::fesetenv(&environment_); }
};
template<class T> void same_number(T actual, T expected)
{
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(std::signbit(actual), std::signbit(expected));
}
void same_geometry(const PLIC::CellGeometryView<ordinary>& actual, const PLIC::ConvexGeometryPtr<ordinary>& factory)
{
    const auto expected = PLIC::CellGeometryView<ordinary>::from_existing(factory);
    same_number(actual.volume().value, expected.volume().value);
    EXPECT_EQ(actual.origin(), expected.origin());
    const auto actual_moments = actual.moments(), expected_moments = expected.moments();
    same_number(actual_moments.volume.value, expected_moments.volume.value);
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        same_number(actual_moments.first_moment[axis].value, expected_moments.first_moment[axis].value);
        same_number(actual.centroid()[axis], expected.centroid()[axis]);
    }
    ASSERT_EQ(actual.vertex_count(), expected.vertex_count());
    for (std::size_t vertex = 0; vertex < actual.vertex_count(); ++vertex)
        for (unsigned axis = 0; axis < 3; ++axis)
            same_number(actual.corner(vertex)[axis], expected.corner(vertex)[axis]);
    ASSERT_EQ(actual.num_faces(), expected.num_faces());
    for (std::size_t face = 0; face < actual.num_faces(); ++face)
    {
        const auto a = actual.ordered_face(face), b = expected.ordered_face(face);
        EXPECT_EQ(a.support(), b.support());
        EXPECT_EQ(a.orientation(), b.orientation());
        same_number(a.area().value, b.area().value);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            same_number(a.centroid()[axis], b.centroid()[axis]);
            same_number(a.area_vector()[axis], b.area_vector()[axis]);
        }
    }
}
/** Deliberately labels its leaf Cartesian while overriding both factory facts.
 * Dynamic type, not the public label/recipe, must decide the native shortcut. */
class LabelledCartesianKernel final : public AMR::AdaptiveProviderKernel
{
    PLIC::ConvexGeometryPtr<ordinary> shape_;

public:
    explicit LabelledCartesianKernel(PLIC::ConvexGeometryPtr<ordinary> shape) : shape_(std::move(shape)) {}
    AMR::AdaptiveBytes descriptor() const override { return {'l', 'a', 'b', 'e', 'l'}; }
    void validate(const AMR::AdaptiveLeaf&) const override {}
    PLIC::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf&) const override { return shape_; }
    AMR::AdaptiveCellMetrics metrics(const AMR::AdaptiveLeaf&) const override { return {17, {19, 23, 29}}; }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf&) const override { return {}; }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf&, AMR::AdaptiveSplit) const override
    {
        throw std::invalid_argument("Probe does not split.");
    }
};
TEST(GeneralVoFFetchedSources, SourceOnlyKeepsTwoNativeContextsAndNoSelectedOrMetricStorage)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused private construction probe is serial.";
    const auto accepted = state();
    const std::array<std::uint64_t, 1> first_ordinal{0}, second_ordinal{1};
    // Independently detached recipes must remain independently owned even
    // though both come from the same physical Cartesian block.
    const auto first = detached(first_ordinal), second = detached(second_ordinal);
    ASSERT_NE(first.owned.front().native_recipe, second.owned.front().native_recipe);
    const std::array<const AMR::AdaptiveLeaf*, 2> leaves{&first.owned.front(), &second.owned.front()};
    general_vof_detail::wire::Budget budget{64 * 1024};
    const auto result = catalog::build<ordinary>(
        *accepted->hierarchy(), leaves, budget, nullptr, catalog::CatalogPurpose::OriginalSources);
    ASSERT_TRUE(result.ordinary);
    EXPECT_FALSE(result.finite_ordinary);
    EXPECT_FALSE(result.bounded);
    EXPECT_TRUE(result.native_metrics.empty());
    const PLIC::CellGeometryView<ordinary> a{result.ordinary, 0}, b{result.ordinary, 1};
    EXPECT_TRUE(a.is_procedural());
    EXPECT_TRUE(b.is_procedural());
    EXPECT_EQ(a.volume().value, 1.L);
    EXPECT_EQ(b.volume().value, 1.L);
    const auto a_source = PLIC::details::NativeCellGeometrySourceAccess::source_cell(a);
    const auto b_source = PLIC::details::NativeCellGeometrySourceAccess::source_cell(b);
    EXPECT_NE(a_source.source, b_source.source);
    general_vof_detail::wire::Budget tiny{1};
    EXPECT_THROW((void) catalog::build<ordinary>(
                     *accepted->hierarchy(), leaves, tiny, nullptr, catalog::CatalogPurpose::OriginalSources),
        std::exception);
}
TEST(GeneralVoFFetchedSources, SourceOnlyMatchesActualCartesianFactoryWithoutSelectedImport)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused private construction probe is serial.";
    const auto accepted = state();
    const std::array<std::uint64_t, 1> ordinal{0};
    auto input = detached(ordinal);
    auto& leaf = input.owned.front();
    leaf.geometry = accepted->hierarchy()->geometry(leaf);
    const auto original = leaf.geometry;
    const auto factory_geometry = accepted->hierarchy()->geometry(leaf);
    const std::array<const AMR::AdaptiveLeaf*, 1> leaves{&leaf};
    general_vof_detail::wire::Budget budget{64 * 1024};
    const auto result = catalog::build<ArithmeticMode::BoundedCertification>(
        *accepted->hierarchy(), leaves, budget, nullptr, catalog::CatalogPurpose::OriginalSources);
    EXPECT_FALSE(result.finite_ordinary);
    EXPECT_FALSE(result.bounded);
    EXPECT_TRUE(result.native_metrics.empty());
    const PLIC::CellGeometryView<ordinary> view{result.ordinary, 0};
    EXPECT_FALSE(view.is_procedural());
    const auto owner = PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(view);
    EXPECT_NE(owner, original); // The actual Cartesian factory ignores the cached certificate.
    EXPECT_TRUE(owner->same_native_representation(*factory_geometry));
    EXPECT_EQ(view.volume().value, 1.L);
    general_vof_detail::wire::Budget tiny{1};
    EXPECT_THROW((void) catalog::build<ArithmeticMode::BoundedCertification>(
                     *accepted->hierarchy(), leaves, tiny, nullptr, catalog::CatalogPurpose::OriginalSources),
        std::exception);
}
TEST(GeneralVoFFetchedSources, PublicFetchedViewsKeepCompleteOrderAndIndependentLifetime)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Remote-request and empty-owner cases are separate MPI integration gates.";
    std::vector<PLIC::CellGeometryView<ordinary>> surviving;
    std::vector<std::weak_ptr<const void>> accepted_lifetimes;
    for (const auto mode : {ordinary, ArithmeticMode::BoundedCertification})
    {
        const auto accepted = state(mode);
        accepted_lifetimes.push_back(accepted->accepted_identity());
        const auto fetched = accepted->fetch_native_sources(accepted->owned_order(), 4 * 1024 * 1024);
        const auto& draft = fetched.draft();
        const auto geometry = fetched.geometry();
        ASSERT_EQ(geometry.size(), draft.owned.size() + draft.ghosts.size());
        EXPECT_GT(fetched.admitted_bytes(), 0u);
        EXPECT_LE(fetched.admitted_bytes(), 4u * 1024 * 1024);
        EXPECT_EQ(fetched.fetch_byte_budget(), accepted->hierarchy()->options().byte_budget);
        std::map<GeneralVoF::Key, PLIC::CellGeometryView<ordinary>> accepted_views;
        if (mode == ordinary)
        {
            const auto original = accepted->owned_geometry();
            for (std::size_t i = 0; i < accepted->owned_order().size(); ++i)
                accepted_views.emplace(accepted->owned_order()[i], original[i]);
        }
        std::size_t index = 0;
        for (const auto* rows : {&draft.owned, &draft.ghosts})
            for (const auto& leaf : *rows)
            {
                const auto view = geometry[index++];
                EXPECT_EQ(view.is_procedural(), mode == ordinary);
                EXPECT_EQ(view.volume().value, 1.L);
                if (mode == ordinary && accepted_views.contains(leaf.key))
                {
                    const auto fresh = PLIC::details::NativeCellGeometrySourceAccess::source_cell(view);
                    const auto old =
                        PLIC::details::NativeCellGeometrySourceAccess::source_cell(accepted_views.at(leaf.key));
                    EXPECT_NE(fresh.source, old.source);
                }
                const auto expected = accepted->hierarchy()->metrics(leaf);
                EXPECT_EQ(view.centroid(), expected.centroid);
            }
        surviving.push_back(geometry[0]);
    }
    for (const auto& lifetime : accepted_lifetimes)
        EXPECT_TRUE(lifetime.expired());
    ASSERT_EQ(surviving.size(), 2u);
    for (const auto& view : surviving)
    {
        ASSERT_TRUE(view);
        EXPECT_EQ(view.volume().value, 1.L);
    }
    EXPECT_TRUE(surviving[0].is_procedural());
    EXPECT_FALSE(surviving[1].is_procedural());
}
TEST(GeneralVoFFetchedSources, ActualFactoryBoundsMatchUnderChangedConstructionAndQueryRoundings)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused current-control factory oracle is serial.";
    EnvironmentRestore restore;
    const std::array<int, 4> roundings{FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    for (const auto translated : {false, true})
    {
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
        auto input_grid = grid();
        input_grid.origin = translated ? PLIC::GeometryPoint{0x1p40, -0x1p35, .3} : PLIC::GeometryPoint{-.7, .2, -.9};
        input_grid.extent = {2.3, 5.1, 7.7};
        const std::array<std::uint64_t, 2> ordinals{0, 7};
        AMR::CartesianAMR mesh(input_grid);
        auto input = mesh.detach_owned(ordinals, {'c', 'o', 'n', 't', 'r', 'o', 'l'});
        // This is a private factory oracle. Full boundary/partition closure is
        // unrelated; keep validated detached leaves and the actual kernel while
        // using an empty hierarchy exactly as the custom-provider probe does.
        for (const auto& leaf : input.owned)
            input.kernel->validate(leaf);
        const GeneralVoF::Hierarchy hierarchy({}, {}, input.kernel, Tpetra::getDefaultComm());
        ASSERT_TRUE(hierarchy.has_native_cartesian_geometry());
        std::vector<const AMR::AdaptiveLeaf*> leaves;
        for (const auto& leaf : input.owned)
            leaves.push_back(&leaf);
        for (const auto construction : roundings)
            for (const auto first_query : roundings)
            {
                // Every pair gets fresh raw/view caches. A previous query must
                // never prime the reference or native source for the next pair.
                ASSERT_EQ(std::fesetround(construction), 0);
                std::vector<PLIC::ConvexGeometryPtr<ordinary>> raw_factory, state_original, state_imported;
                std::vector<PLIC::GeometryValue<ordinary>> raw_constructor_volume;
                for (const auto* leaf : leaves)
                {
                    auto raw = hierarchy.geometry(*leaf);
                    raw_constructor_volume.push_back(raw->volume());
                    raw_factory.push_back(std::move(raw)); // No moments()/centroid() query here.
                    // Independent old GeneralVoF State workflow: a distinct
                    // factory result, finite import and native-equivalence
                    // comparison, which primes its moments at construction.
                    auto original = hierarchy.geometry(*leaf);
                    auto imported = PLIC::ConvexCellGeometry<ordinary>::from_geometry(*original);
                    if (!leaf->geometry && std::holds_alternative<AMR::CartesianCellAncestry>(leaf->key.ancestry) &&
                        original->same_native_representation(*imported))
                        original = imported;
                    state_original.push_back(std::move(original));
                    state_imported.push_back(std::move(imported));
                }
                general_vof_detail::wire::Budget source_budget{1024 * 1024}, state_budget{1024 * 1024};
                const auto sources = catalog::build<ordinary>(
                    hierarchy, leaves, source_budget, nullptr, catalog::CatalogPurpose::OriginalSources);
                const auto selected = catalog::build<ordinary>(hierarchy, leaves, state_budget);
                ASSERT_TRUE(sources.native_metrics.empty());
                for (std::size_t i = 0; i < leaves.size(); ++i)
                {
                    const PLIC::CellGeometryView<ordinary> raw{sources.ordinary, i};
                    ASSERT_TRUE(raw.is_procedural());
                    // The constructor volume is distinct from the potentially
                    // first-query-dependent moments volume and remains stored.
                    same_number(raw.volume().value, raw_constructor_volume[i].value);
                    const auto metric = hierarchy.metrics(*leaves[i]);
                    same_number(selected.native_metrics[i].volume, metric.volume);
                    for (unsigned axis = 0; axis < 3; ++axis)
                        same_number(selected.native_metrics[i].centroid[axis], metric.centroid[axis]);
                }
                ASSERT_EQ(std::fesetround(first_query), 0);
                SCOPED_TRACE(::testing::Message() << "translated=" << translated << " construction=" << construction
                                                  << " first_query=" << first_query);
                for (std::size_t i = 0; i < leaves.size(); ++i)
                {
                    const PLIC::CellGeometryView<ordinary> raw{sources.ordinary, i};
                    // This performs the FIRST moment query on each raw owner
                    // and view under first_query, followed by centroid parity.
                    // It intentionally does not prime RawReference earlier to
                    // match the separate State import's moment policy.
                    same_geometry(raw, raw_factory[i]);
                    same_number(raw.volume().value, raw_constructor_volume[i].value);
                    same_number(raw.moments().volume.value, raw_factory[i]->moments().volume.value);
                    // centroid() must preserve the old factory's constructor
                    // volume denominator, even if queried moments differ.
                    for (unsigned axis = 0; axis < 3; ++axis)
                        same_number(raw.centroid()[axis], raw_factory[i]->centroid()[axis]);
                    same_geometry({selected.ordinary, i}, state_original[i]);
                    same_geometry(selected.view<ordinary>(i), state_imported[i]);
                }
            }
    }
}
TEST(GeneralVoFFetchedSources, CustomCartesianLabelUsesOriginalGeometryAndMetricFactories)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Focused actual-kernel capability oracle is serial.";
    const auto shape = PLIC::ConvexCellGeometry<ordinary>::box({-1, -.25, -.5}, {1, .25, .5});
    const auto kernel = std::make_shared<const LabelledCartesianKernel>(shape);
    const GeneralVoF::Hierarchy hierarchy({}, {}, kernel, Tpetra::getDefaultComm());
    EXPECT_FALSE(hierarchy.has_native_cartesian_geometry());
    const std::array<std::uint64_t, 1> ordinal{0};
    const auto input = detached(ordinal);
    const auto& leaf = input.owned.front();
    ASSERT_EQ(leaf.provider, AMR::AdaptiveProviderKind::Cartesian);
    EXPECT_FALSE(hierarchy.native_cartesian_geometry_bounds(leaf));
    const std::array<const AMR::AdaptiveLeaf*, 1> leaves{&leaf};
    general_vof_detail::wire::Budget budget{1024 * 1024};
    const auto result = catalog::build<ordinary>(hierarchy, leaves, budget);
    const PLIC::CellGeometryView<ordinary> view{result.ordinary, 0};
    EXPECT_FALSE(view.is_procedural());
    same_geometry(view, shape);
    EXPECT_EQ(result.native_metrics[0].volume, 17);
    EXPECT_EQ(result.native_metrics[0].centroid, (PLIC::GeometryPoint{19, 23, 29}));
    general_vof_detail::wire::Budget metric_budget{1024 * 1024};
    catalog::NativeMetrics metrics(hierarchy, ordinary, metric_budget);
    EXPECT_EQ(metrics(leaf).volume, 17);
    EXPECT_EQ(metrics(leaf).centroid, (PLIC::GeometryPoint{19, 23, 29}));
}
TEST(GeneralVoFFetchedSources, TinyOperationAdmissionRejectsWithoutChangingAcceptedState)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Rank-asymmetric admission is a separate MPI integration gate.";
    const auto accepted = state();
    const auto identity = accepted->accepted_identity();
    const auto control = accepted->accepted_control();
    const auto gas = accepted->gas_volumes(), carrier = accepted->carrier_volumes();
    EXPECT_THROW((void) accepted->fetch_native_sources(accepted->owned_order(), 1), std::exception);
    EXPECT_EQ(accepted->accepted_identity(), identity);
    EXPECT_EQ(accepted->accepted_control(), control);
    EXPECT_EQ(accepted->gas_volumes(), gas);
    EXPECT_EQ(accepted->carrier_volumes(), carrier);
}
} // namespace
