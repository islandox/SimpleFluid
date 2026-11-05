/** Independent native-view geometry, ownership and admission contracts. */
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/PLIC/details/CellGeometryEvaluation.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
namespace P = SimpleFluid::PLIC;
namespace R = P::details::retention;
using SimpleFluid::ArithmeticMode;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto bounded = ArithmeticMode::BoundedCertification;
constexpr auto exact = ArithmeticMode::Exact;
using Source = P::CartesianCellSource;
using Catalog = P::CellGeometryCatalog<ordinary>;
using View = P::CellGeometryView<ordinary>;
using Range = P::CellGeometryRange<ordinary>;
using Geometry = P::ConvexCellGeometry<ordinary>;
constexpr std::size_t cap = 64 * 1024 * 1024;

P::GeometryLimits limits()
{
    P::GeometryLimits result;
    result.arithmetic_mode = ordinary;
    return result;
}
std::shared_ptr<const Source> source(std::array<double, 3> lower = {0, 0, 0}, std::array<double, 3> upper = {1, 1, 1})
{
    P::CartesianSourceDescription description;
    for (std::size_t axis = 0; axis < 3; ++axis)
        description.axes[axis] = {lower[axis], upper[axis]};
    P::CartesianSourceCellSlot slot;
    slot.upper = {1, 1, 1};
    description.cells.push_back(slot);
    return Source::make(std::move(description), limits());
}
struct Environment
{
    fenv_t saved{};
    Environment() { fegetenv(&saved); }
    ~Environment() { fesetenv(&saved); }
};
template<class V>
concept CanMaterialize =
    requires(const V& view, P::CellGeometryOperationWorkspace& workspace) { P::materialize_cell(view, workspace); };
static_assert(CanMaterialize<View>);
static_assert(CanMaterialize<P::CellGeometryView<bounded>>);
static_assert(!CanMaterialize<P::CellGeometryView<exact>>);
static_assert(!std::is_convertible_v<View, P::ConvexGeometryPtr<ordinary>>);
static_assert(std::ranges::input_range<Range>);
} // namespace

TEST(CellGeometryView, IndependentBoxMeasuresAndOrderedFaces)
{
    const auto catalog = Catalog::cartesian(source({-2, 4, 8}, {6, 6, 12}), limits());
    const View view(catalog, 0);
    ASSERT_TRUE(view.is_procedural());
    EXPECT_EQ(view.volume().value, 64);
    const auto center = view.centroid();
    EXPECT_EQ(center[0], 2);
    EXPECT_EQ(center[1], 5);
    EXPECT_EQ(center[2], 10);
    const auto relative = view.moments({-2, 4, 8});
    const std::array<long double, 3> first{256, 64, 128};
    const std::array<long double, 3> spans{8, 2, 4};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        EXPECT_EQ(relative.first_moment[axis].value, first[axis]);
        EXPECT_EQ(view.axis_spans()[axis].value, spans[axis]);
    }
    const std::array<std::array<std::size_t, 4>, 6> loops{
        {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}}};
    const std::array<long double, 6> areas{8, 8, 32, 32, 16, 16};
    ASSERT_EQ(view.faces().size(), loops.size());
    for (std::size_t face = 0; face < loops.size(); ++face)
    {
        EXPECT_TRUE(std::ranges::equal(view.faces()[face], loops[face]));
        EXPECT_EQ(view.faces()[face].area().value, areas[face]);
    }
    const View unit(Catalog::cartesian(source(), limits()), 0);
    const auto support = unit.support({2, -3, 5}, {.25, .5, .75});
    EXPECT_EQ(support.lower.value, -5.75L);
    EXPECT_EQ(support.upper.value, 4.25L);
}

TEST(CellGeometryView, ValuesFacesIteratorsAndSlicesKeepTheirSourceAlive)
{
    auto original = source();
    std::weak_ptr<const Source> weak = original;
    auto catalog = Catalog::cartesian(original, limits());
    Range range(catalog, 0, 1);
    auto iterator = range.begin();
    auto face = range[0].faces()[0];
    original.reset();
    catalog.reset();
    range = {};
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ((*iterator).volume().value, 1);
    EXPECT_EQ(face.area().value, 1);
    EXPECT_EQ(face.corner(0), (std::array<long double, 3>{0, 0, 0}));
    EXPECT_THROW(static_cast<void>(face.at(4)), P::GeometryError);
    iterator = {};
    EXPECT_FALSE(weak.expired()); // The face still owns the source catalog.
}

TEST(CellGeometryView, EmptyAndRuntimeViewsRejectInvalidQueries)
{
    const View empty;
    EXPECT_FALSE(empty);
    EXPECT_THROW(static_cast<void>(empty.volume()), P::GeometryError);
    const Range range;
    EXPECT_TRUE(range.empty());
    EXPECT_EQ(range.begin(), range.end());
    EXPECT_THROW(static_cast<void>(range.at(0)), P::GeometryError);
    const P::DynamicCellGeometryView runtime;
    EXPECT_FALSE(runtime);
    EXPECT_THROW(static_cast<void>(runtime.arithmetic_mode()), P::GeometryError);
    const P::DynamicCellGeometryView nonempty(View(Catalog::cartesian(source(), limits()), 0));
    EXPECT_EQ(nonempty.arithmetic_mode(), ordinary);
    EXPECT_EQ(nonempty.get<ordinary>().volume().value, 1);
    EXPECT_THROW(static_cast<void>(nonempty.get<bounded>()), P::GeometryError);
}

TEST(CellGeometryView, DistinctEqualSourcesStayDistinctAndReorderingIsFlat)
{
    auto first = source(), equal = source(), large = source({0, 0, 0}, {2, 1, 1});
    auto catalog = Catalog::make({first, equal, first, large},
        {P::NativeCellGeometrySlot{0, 0}, P::NativeCellGeometrySlot{1, 0}, P::NativeCellGeometrySlot{2, 0},
            P::NativeCellGeometrySlot{3, 0}},
        limits());
    EXPECT_EQ(catalog->source_count(), 3u);
    const std::array<std::size_t, 4> order{3, 1, 2, 0};
    auto reordered = Catalog::reordered(catalog, order, limits());
    std::weak_ptr<const Catalog> old = catalog;
    catalog.reset();
    EXPECT_TRUE(old.expired());
    const Range slice(reordered, 0, 2);
    EXPECT_EQ(slice[0].volume().value, 2);
    EXPECT_EQ(slice[1].volume().value, 1);
    EXPECT_THROW((Range(reordered, 3, 2)), P::GeometryError);
    EXPECT_THROW((Range(reordered, std::numeric_limits<std::size_t>::max(), 1)), P::GeometryError);
    const std::array<std::size_t, 4> invalid{0, 1, 2, 2};
    EXPECT_THROW(static_cast<void>(Catalog::reordered(reordered, invalid, limits())), P::GeometryError);
}

TEST(CellGeometryView, CatalogUnionDoesNotConsumeOneCellsShapeLimit)
{
    auto small = limits();
    small.maximum_vertices = 8;
    small.maximum_faces = 6;
    P::CartesianSourceDescription description;
    description.axes[0] = {0, 1};
    description.axes[1] = {0, 1};
    for (unsigned i = 0; i <= 12; ++i)
        description.axes[2].push_back(i);
    for (unsigned i = 0; i < 12; ++i)
    {
        P::CartesianSourceCellSlot slot;
        slot.lower = {0, 0, i};
        slot.upper = {1, 1, i + 1};
        description.cells.push_back(slot);
    }
    const Range range(Catalog::cartesian(Source::make(std::move(description), small), small));
    ASSERT_EQ(range.size(), 12u);
    for (const auto view : range)
        EXPECT_EQ(view.volume().value, 1);
}

TEST(CellGeometryView, ExplicitMaterializationOwnsOutputAndMatchesCanonicalCertificate)
{
    const std::array<double, 3> lower{-.1, .3, -.7}, upper{.8, 1.7, .4};
    const auto expected = Geometry::box({lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]});
    P::ConvexGeometryPtr<ordinary> result;
    {
        const View view(Catalog::cartesian(source(lower, upper), limits()), 0);
        P::CellGeometryOperationWorkspace workspace(limits());
        result = P::materialize_cell(view, workspace);
        EXPECT_GT(workspace.bytes_used(), 0u);
        EXPECT_EQ(result->encode_certificate(), expected->encode_certificate());
        EXPECT_EQ(result->volume().value, view.volume().value);
        EXPECT_EQ(result->volume().binary64_absolute_error(), view.volume().binary64_absolute_error());
    }
    EXPECT_EQ(result->encode_certificate(), expected->encode_certificate());
    P::GeometryAllocationLedger surviving;
    std::size_t used = 0;
    EXPECT_NO_THROW(surviving.retain(result, used, cap));
    EXPECT_GT(used, 0u);
}

TEST(CellGeometryView, SharedCatalogAdmissionIsExactAndFailurePoisonsOnlyItsLedger)
{
    const auto catalog = Catalog::cartesian(source(), limits());
    P::GeometryAllocationLedger measure;
    std::size_t needed = 0;
    measure.retain(catalog, needed, cap);
    const auto owners = measure.owner_count();
    const auto first = needed;
    measure.retain(catalog, needed, cap);
    EXPECT_EQ(needed, first);
    EXPECT_EQ(measure.owner_count(), owners);
    P::GeometryAllocationLedger exact_budget;
    std::size_t exact_used = 0;
    EXPECT_NO_THROW(exact_budget.retain(catalog, exact_used, first));
    EXPECT_EQ(exact_used, first);
    P::GeometryAllocationLedger short_budget;
    std::size_t short_used = 0;
    ASSERT_GT(first, 0u);
    EXPECT_THROW(short_budget.retain(catalog, short_used, first - 1), P::GeometryError);
    EXPECT_THROW(short_budget.retain(catalog, short_used, cap), P::GeometryError);
    EXPECT_EQ(View(catalog, 0).volume().value, 1);
}

TEST(CellGeometryView, BatchMaterializationAdmitsDescriptorsAndUnwindsPartialOutput)
{
    auto first = Geometry::box({0, 0, 0}, {1, 1, 1});
    auto second = Geometry::box({1, 0, 0}, {3, 1, 1});
    std::weak_ptr<const Geometry> first_lifetime = first;
    auto catalog = Catalog::materialized({first, second}, limits());
    Range range(catalog);
    std::size_t required = 0;
    {
        P::CellGeometryOperationWorkspace workspace(limits());
        const auto result = P::materialize_cells(range, workspace);
        ASSERT_EQ(result.size(), 2u);
        EXPECT_EQ(result[0].get(), first.get());
        EXPECT_EQ(result[1].get(), second.get());
        EXPECT_EQ(result[0]->volume().value, 1);
        EXPECT_EQ(result[1]->volume().value, 2);
        required = workspace.bytes_used();
    }
    ASSERT_GT(required, 0u);
    {
        auto exact_limits = limits();
        exact_limits.byte_budget = required;
        P::CellGeometryOperationWorkspace exact_budget(exact_limits);
        EXPECT_EQ(P::materialize_cells(range, exact_budget).size(), 2u);
        EXPECT_EQ(exact_budget.bytes_used(), required);
    }
    {
        auto short_limits = limits();
        short_limits.byte_budget = required - 1;
        P::CellGeometryOperationWorkspace short_budget(short_limits);
        EXPECT_THROW(static_cast<void>(P::materialize_cells(range, short_budget)), P::GeometryError);
        const auto spent = short_budget.bytes_used();
        EXPECT_GT(spent, 0u);
        EXPECT_THROW(static_cast<void>(P::materialize_cell(range[0], short_budget)), P::GeometryError);
        EXPECT_GE(short_budget.bytes_used(), spent);
        range = {};
        catalog.reset();
        first.reset();
        second.reset();
        EXPECT_FALSE(first_lifetime.expired()); // A spent prefix remains pinned by its workspace.
    }
    EXPECT_TRUE(first_lifetime.expired());
}

TEST(CellGeometryView, NativeRetentionHasNoGeneralGeometryOrExportOwners)
{
    const auto catalog = Catalog::cartesian(source(), limits());
    const View view(catalog, 0);
    for (const auto face : view.faces())
    {
        static_cast<void>(face.area());
        static_cast<void>(face.support());
    }
    static_cast<void>(view.moments());
    static_cast<void>(view.closed_contact_bounds());
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(catalog, used, cap);
    const std::array roots{R::root(*catalog)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    std::vector<std::byte> scratch_bytes(1024 * 1024);
    R::ProbeScratch scratch(scratch_bytes);
    const auto report = R::inspect_retention(roots, ledgers, scratch, {1000, 1000, true});
    ASSERT_TRUE(report.complete()) << report.detail;
    for (const auto& owner : report.owners)
    {
        const std::string_view kind(owner.kind);
        EXPECT_NE(kind, "geometry");
        EXPECT_NE(kind, "plane");
        EXPECT_NE(kind, "point_context");
        EXPECT_NE(kind, "topology");
        EXPECT_EQ(kind.find("export"), std::string_view::npos) << kind;
    }
    EXPECT_EQ(std::ranges::count_if(
                  report.owners, [](const auto& owner) { return std::string_view(owner.kind) == "cartesian_source"; }),
        1);
    EXPECT_EQ(std::ranges::count_if(
                  report.owners, [](const auto& owner) { return std::string_view(owner.kind) == "cell_catalog"; }),
        1);
    EXPECT_EQ(report.ledger_only.bytes, 0u);
}

TEST(CellGeometryView, BoundedAndExactViewsKeepExistingOwners)
{
    auto certified = P::ConvexCellGeometry<bounded>::box({0, 0, 0}, {1, 1, 1});
    const auto certified_view = P::CellGeometryView<bounded>::from_existing(certified);
    EXPECT_FALSE(certified_view.is_procedural());
    P::CellGeometryOperationWorkspace workspace;
    EXPECT_EQ(P::materialize_cell(certified_view, workspace).get(), certified.get());
    auto rational = P::ConvexCellGeometry<exact>::box({0, 0, 0}, {1, 1, 1});
    auto exact_view = P::CellGeometryView<exact>::from_existing(rational);
    EXPECT_EQ(exact_view.existing_geometry().get(), rational.get());
    std::weak_ptr<const P::ConvexCellGeometry<exact>> weak = rational;
    rational.reset();
    EXPECT_FALSE(weak.expired());
    exact_view = {};
    EXPECT_TRUE(weak.expired());
}

TEST(CellGeometryView, CachedCoordinatesAndMaterializationRestoreCallerEnvironment)
{
    Environment restore;
    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    const View view(Catalog::cartesian(source({-.1, .3, -.7}, {.8, 1.7, .4}), limits()), 0);
    std::array<std::array<long double, 3>, 8> corners;
    for (std::size_t vertex = 0; vertex < corners.size(); ++vertex)
        corners[vertex] = view.corner(vertex);
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        ASSERT_EQ(fesetround(mode), 0);
        ASSERT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(feraiseexcept(FE_DIVBYZERO), 0);
        for (std::size_t vertex = 0; vertex < corners.size(); ++vertex)
        {
            EXPECT_EQ(view.corner(vertex), corners[vertex]);
            EXPECT_EQ(fegetround(), mode);
            EXPECT_EQ(fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        }
        P::CellGeometryOperationWorkspace workspace(limits());
        const auto materialized = P::materialize_cell(view, workspace);
        EXPECT_EQ(fegetround(), mode);
        EXPECT_EQ(fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        // The first query occurs after the materialization environment has ended.
        const auto actual = materialized->moments();
        const auto expected = view.moments();
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_EQ(actual.first_moment[axis].value, expected.first_moment[axis].value);
            EXPECT_EQ(actual.first_moment[axis].binary64_absolute_error(),
                expected.first_moment[axis].binary64_absolute_error());
        }
    }
}

TEST(CellGeometryView, DeferredClonesShareImmutableBaseAndAdmitOnlyTheirOwnEvaluationState)
{
    auto original = source({-.1, .3, -.7}, {.8, 1.7, .4});
    const auto clone = Source::clone_deferred(original, limits());
    const auto volume = original->volume(0).value;
    EXPECT_EQ(clone->base_identity(), original->base_identity());
    EXPECT_NE(clone.get(), original.get());
    EXPECT_EQ(clone->volume(0).value, original->volume(0).value);
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(original, used, cap);
    const auto payload = ledger.payload_bytes();
    const auto expected = P::details::NativeCellGeometrySourceAccess::exclusive_bytes(*clone);
    ledger.retain(clone, used, cap);
    EXPECT_EQ(ledger.payload_bytes() - payload, expected[0] + expected[1] + expected[2]);
    const auto after = used;
    ledger.retain(clone, used, cap);
    EXPECT_EQ(used, after);
    std::weak_ptr<const Source> weak = original;
    original.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(clone->source_support(0, 0), weak.lock()->source_support(0, 0));
    std::array<R::BorrowedRoot, 1> roots{R::root(*clone)};
    std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    std::vector<std::byte> scratch_bytes(1024 * 1024);
    R::ProbeScratch scratch(scratch_bytes);
    const auto census = R::inspect_retention(roots, ledgers, scratch, {1000, 1000, true});
    ASSERT_TRUE(census.complete()) << census.detail;
    EXPECT_EQ(census.ledger_payload_admission, ledger.payload_bytes());
    ledger = P::GeometryAllocationLedger{};
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(clone->volume(0).value, volume);
}

TEST(CellGeometryView, OriginalEvaluationClonesKeepDirectedLazyMomentsAndShareOnlyImmutableVertices)
{
    Environment restore;
    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    const P::GeometryPoint lower{-.1, .3, -.7}, upper{.8, 1.7, .4};
    const auto original = Geometry::box(lower, upper);
    const auto original_storage = P::details::checked_geometry_storage<ordinary>(
        P::DynamicGeometryAccess::implementation(*original->dynamic_geometry()));
    ASSERT_FALSE(original_storage->moment_cache->ready.load(std::memory_order_acquire));
    ASSERT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(feraiseexcept(FE_DIVBYZERO), 0);
    const auto first = P::details::clone_cell_evaluation<ordinary>(original, false, limits());
    const auto second = P::details::clone_cell_evaluation<ordinary>(original, false, limits());
    EXPECT_EQ(fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    const auto first_storage = P::details::checked_geometry_storage<ordinary>(
        P::DynamicGeometryAccess::implementation(*first->dynamic_geometry()));
    const auto second_storage = P::details::checked_geometry_storage<ordinary>(
        P::DynamicGeometryAccess::implementation(*second->dynamic_geometry()));
    EXPECT_EQ(first_storage->vertices.data(), original_storage->vertices.data());
    EXPECT_EQ(second_storage->vertices.data(), original_storage->vertices.data());
    EXPECT_EQ(first_storage->vertices.owned_capacity(), 0u);
    EXPECT_NE(first_storage->moment_cache, second_storage->moment_cache);
    std::size_t index = 0;
    for (const auto direction : {FE_UPWARD, FE_DOWNWARD})
    {
        ASSERT_EQ(fesetround(FE_TONEAREST), 0);
        const auto oracle = Geometry::box(lower, upper);
        ASSERT_EQ(fesetround(direction), 0);
        const auto expected = oracle->moments();
        const auto actual = (index++ ? second : first)->moments();
        EXPECT_EQ(actual.volume.value, expected.volume.value);
        for (std::size_t axis = 0; axis < 3; ++axis)
            EXPECT_EQ(actual.first_moment[axis].value, expected.first_moment[axis].value);
    }
    EXPECT_FALSE(original_storage->moment_cache->ready.load(std::memory_order_acquire));
    ASSERT_EQ(fesetround(FE_UPWARD), 0);
    static_cast<void>(original->moments());
    ASSERT_TRUE(P::details::cell_evaluation_cache_ready<ordinary>(original));
    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    const auto oracle = Geometry::box(lower, upper);
    const auto after_template_query = P::details::clone_cell_evaluation<ordinary>(original, false, limits());
    ASSERT_EQ(fesetround(FE_DOWNWARD), 0);
    const auto expected = oracle->moments(), actual = after_template_query->moments();
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_EQ(actual.first_moment[axis].value, expected.first_moment[axis].value);
}

template<ArithmeticMode Mode> void imported_evaluation_cache_contract()
{
    const auto original = P::ConvexCellGeometry<ordinary>::box({-.1, .3, -.7}, {.8, 1.7, .4});
    const auto selected = P::ConvexCellGeometry<Mode>::from_geometry(*original);
    const auto selected_storage = P::details::checked_geometry_storage<Mode>(
        P::DynamicGeometryAccess::implementation(*selected->dynamic_geometry()));
    ASSERT_FALSE(selected_storage->moment_cache->ready.load(std::memory_order_acquire));
    ASSERT_FALSE(P::details::cell_evaluation_cache_ready<Mode>(selected));
    P::GeometryLimits selected_limits;
    selected_limits.arithmetic_mode = Mode;
    P::details::Operation work(selected_limits);
    const auto lazy = P::details::clone_cell_evaluation<Mode>(selected, false, selected_limits);
    EXPECT_EQ(work.work.operations, 0u);
    EXPECT_EQ(work.work.pairs, 0u);
    const auto lazy_storage = P::details::checked_geometry_storage<Mode>(
        P::DynamicGeometryAccess::implementation(*lazy->dynamic_geometry()));
    EXPECT_FALSE(lazy_storage->moment_cache->ready.load(std::memory_order_acquire));
    EXPECT_EQ(lazy_storage->vertices.data(), selected_storage->vertices.data());
    static_cast<void>(selected->moments());
    ASSERT_TRUE(selected_storage->moment_cache->ready.load(std::memory_order_acquire));
    ASSERT_TRUE(P::details::cell_evaluation_cache_ready<Mode>(selected));
    const auto before_operations = work.work.operations, before_pairs = work.work.pairs;
    const auto primed = P::details::clone_cell_evaluation<Mode>(selected, true, selected_limits);
    EXPECT_EQ(work.work.operations, before_operations);
    EXPECT_EQ(work.work.pairs, before_pairs);
    const auto primed_storage = P::details::checked_geometry_storage<Mode>(
        P::DynamicGeometryAccess::implementation(*primed->dynamic_geometry()));
    EXPECT_NE(primed_storage->moment_cache, selected_storage->moment_cache);
    EXPECT_TRUE(primed_storage->moment_cache->ready.load(std::memory_order_acquire));
    const auto expected = selected->moments(), actual = primed->moments();
    EXPECT_EQ(actual.volume.value, expected.volume.value);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        EXPECT_EQ(actual.first_moment[axis].value, expected.first_moment[axis].value);
        if constexpr (Mode == bounded)
            EXPECT_EQ(actual.first_moment[axis].absolute_error, expected.first_moment[axis].absolute_error);
    }
    EXPECT_FALSE(lazy_storage->moment_cache->ready.load(std::memory_order_acquire));
    const auto still_lazy = P::details::clone_cell_evaluation<Mode>(selected, false, selected_limits);
    EXPECT_FALSE(P::details::cell_evaluation_cache_ready<Mode>(still_lazy));
}
TEST(CellGeometryView, ImportedEvaluationClonesPreserveCompletedFactsWithoutPrimingLazyCaches)
{
    imported_evaluation_cache_contract<ordinary>();
    if (P::certification_environment())
        imported_evaluation_cache_contract<bounded>();
}

TEST(CellGeometryView, EvaluationCloneAdmissionCountsTheSharedVertexArrayOnceAndFlattensOwners)
{
    const auto original = Geometry::box({0, 0, 0}, {1, 1, 1});
    const auto first = P::details::clone_cell_evaluation<ordinary>(original, false, limits());
    const auto second = P::details::clone_cell_evaluation<ordinary>(first, false, limits());
    const auto first_storage = P::details::checked_geometry_storage<ordinary>(
        P::DynamicGeometryAccess::implementation(*first->dynamic_geometry()));
    const auto second_storage = P::details::checked_geometry_storage<ordinary>(
        P::DynamicGeometryAccess::implementation(*second->dynamic_geometry()));
    EXPECT_EQ(std::get<std::shared_ptr<const P::DynamicConvexGeometry::Impl>>(first_storage->vertex_owner),
        std::get<std::shared_ptr<const P::DynamicConvexGeometry::Impl>>(second_storage->vertex_owner));
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(original, used, cap);
    const auto original_payload = ledger.payload_bytes();
    ledger.retain(first, used, cap);
    const auto evaluation_bytes = sizeof(Geometry) + sizeof(P::DynamicConvexGeometry) +
        sizeof(P::details::GeometryStorage<ordinary>) + sizeof(P::details::StoredMomentCache<ordinary>);
    EXPECT_EQ(ledger.payload_bytes(), original_payload + evaluation_bytes);
    ledger.retain(second, used, cap);
    EXPECT_EQ(ledger.payload_bytes(), original_payload + 2 * evaluation_bytes);
    const std::array roots{R::root(*second)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    std::vector<std::byte> scratch_bytes(1024 * 1024);
    R::ProbeScratch scratch(scratch_bytes);
    const auto report = R::inspect_retention(roots, ledgers, scratch, {1000, 1000, true});
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.ledger_payload_admission, ledger.payload_bytes());
    P::GeometryAllocationLedger exact;
    std::size_t exact_used = 0;
    exact.retain(original, exact_used, used);
    exact.retain(first, exact_used, used);
    EXPECT_NO_THROW(exact.retain(second, exact_used, used));
    EXPECT_EQ(exact_used, used);
    P::GeometryAllocationLedger short_cap;
    std::size_t short_used = 0;
    short_cap.retain(original, short_used, used - 1);
    short_cap.retain(first, short_used, used - 1);
    EXPECT_THROW(short_cap.retain(second, short_used, used - 1), P::GeometryError);
}

TEST(CellGeometryView, NativeEvaluationCloneBorrowsInputBodyWithoutRetainingItsTemporaryImpl)
{
    P::ConvexGeometryPtr<ordinary> evaluation;
    std::weak_ptr<const P::DynamicConvexGeometry::Impl> weak;
    {
        const View native(Catalog::cartesian(source(), limits()), 0);
        P::CellGeometryOperationWorkspace workspace(limits());
        const auto original = P::materialize_cell(native, workspace);
        weak = P::DynamicGeometryAccess::implementation(*original->dynamic_geometry());
        evaluation = P::details::clone_cell_evaluation<ordinary>(original, false, limits());
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(evaluation->volume().value, 1);
    EXPECT_EQ(evaluation->moments().first_moment[0].value, .5L);
    const auto stored = P::details::checked_geometry_storage<ordinary>(
        P::DynamicGeometryAccess::implementation(*evaluation->dynamic_geometry()));
    EXPECT_TRUE(std::holds_alternative<std::shared_ptr<const P::details::NativeGeometryBody>>(stored->vertex_owner));
}
