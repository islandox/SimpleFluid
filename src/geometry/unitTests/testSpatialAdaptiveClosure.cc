/** Independent geometric, slab-halo and cumulative-work closure oracles. */
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Cell = P::ConvexCellGeometry<ordinary>;
using Wide = std::array<long double, 3>;
using Marks = std::vector<std::pair<AMR::AdaptiveCellKey, AMR::AdaptiveSplit>>;
constexpr std::size_t cap64 = 64 * 1024 * 1024;
constexpr std::size_t cap96 = 96 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

Marks mark_set(const std::vector<AMR::AdaptiveMark>& marks)
{
    Marks result;
    for (const auto& mark : marks)
        result.emplace_back(mark.key, mark.split);
    std::sort(result.begin(), result.end());
    return result;
}
std::vector<AMR::AdaptiveBytes> snapshot(std::span<const AMR::AdaptiveLeaf> source)
{
    std::vector<AMR::AdaptiveBytes> result;
    for (const auto& leaf : source)
        result.push_back(AMR::encode_adaptive_leaf(leaf));
    return result;
}
std::vector<AMR::AdaptiveLeaf> owned(std::vector<AMR::AdaptiveLeaf> source)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<AMR::AdaptiveLeaf> result;
    for (std::size_t i = 0; i < source.size(); ++i)
        if (i % comm->getSize() == std::size_t(comm->getRank()))
            result.push_back(std::move(source[i]));
    return result;
}
class ContactKernel final : public AMR::AdaptiveProviderKernel
{
    bool traces_;
    std::array<bool, 3> periodic_;
    bool restricted_;

public:
    explicit ContactKernel(bool traces = false, std::array<bool, 3> periodic = {}, bool restricted = false)
        : traces_(traces), periodic_(periodic), restricted_(restricted)
    {
    }
    AMR::AdaptiveBytes descriptor() const override
    {
        return {'s', 'p', 'a', 't', char(traces_), char(periodic_[0]), char(periodic_[1]), char(periodic_[2]),
            char(restricted_)};
    }
    AMR::AdaptiveContactCapabilities contact_capabilities(const AMR::AdaptiveLeaf&) const override
    {
        if (periodic_[0] || periodic_[1] || periodic_[2])
            return {}; // This fixture promises normalization, not provider hull images.
        return {AMR::AdaptiveContactNormalization::PhysicalIdentity, false, false};
    }
    P::PointIdentity<ordinary> contact_point(
        const AMR::AdaptiveLeaf&, const P::PointIdentity<ordinary>& point) const override
    {
        return P::periodic_point<ordinary>(point, {0, 0, 0}, {4, 4, 4}, periodic_);
    }
    P::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf& leaf) const override { return leaf.geometry; }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override { return leaf.surfaces; }
    AMR::AdaptiveTrace traces(const AMR::AdaptiveLeaf& leaf) const override
    {
        if (!traces_)
            return {leaf.key, {}};
        const auto& ancestry = std::get<AMR::CartesianCellAncestry>(leaf.key.ancestry);
        P::PlanarFacePtr<ordinary> face;
        if (restricted_)
            for (const auto& surface : leaf.surfaces)
                if ((ancestry.index[0] == 0 && surface.geometry->area_vector().x < 0) ||
                    (ancestry.index[0] != 0 && surface.geometry->area_vector().x > 0))
                    face = surface.geometry;
        return {leaf.key, {{{'t'}, ancestry.level, AMR::AdaptiveSplit::Isotropic, std::move(face), true}}};
    }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf&, AMR::AdaptiveSplit) const override
    {
        throw std::invalid_argument("Independent closure fixture has no refinement implementation.");
    }
};
AMR::AdaptiveLeaf box(std::uint64_t id, P::GeometryPoint lower, P::GeometryPoint upper, unsigned depth = 0)
{
    AMR::AdaptiveLeaf leaf;
    leaf.key = {0, AMR::CartesianCellAncestry{depth, {id, 0, 0}}};
    leaf.domain = {'d'};
    leaf.geometry = Cell::box(lower, upper);
    for (std::size_t f = 0; f < leaf.geometry->num_faces(); ++f)
    {
        auto support = AMR::encode_adaptive_key(leaf.key);
        support.push_back(char(f));
        leaf.surfaces.push_back({std::move(support), leaf.geometry->face_geometry(f), 1, true, "wall"});
    }
    return leaf;
}
void compare_with_reference(const std::vector<AMR::AdaptiveLeaf>& source, const AMR::AdaptiveProviderKernel& kernel,
    const Marks& expected, AMR::AdaptiveClosureOptions options = {})
{
    const auto before = snapshot(source);
    const auto comm = Tpetra::getDefaultComm();
    const auto reference = D::legacy_required_adaptive_refinements(source, kernel, comm, options);
    const auto spatial = AMR::required_adaptive_refinements(source, kernel, comm, options);
    EXPECT_EQ(mark_set(reference), expected);
    EXPECT_EQ(mark_set(spatial), expected);
    EXPECT_EQ(mark_set(spatial), mark_set(reference));
    EXPECT_EQ(snapshot(source), before);
}
Marks coarse_on_rank_zero(const AMR::AdaptiveCellKey& coarse)
{
    if (Tpetra::getDefaultComm()->getRank() == 0)
        return {{coarse, AMR::AdaptiveSplit::Isotropic}};
    return {};
}
template<class Operation> void expect_collective_work_rejection(Operation&& operation)
{
    int rejected = 0, work_budget = 0;
    try
    {
        operation();
    }
    catch (const std::invalid_argument& error)
    {
        rejected = 1;
        const std::string_view message = error.what();
        work_budget = message == "Spatial closure cumulative comparison budget exhausted." ||
                      message == "Spatial closure cumulative trace budget exhausted.";
        if (!work_budget)
            EXPECT_EQ(message, "Distributed adaptation rejected on another rank.");
    }
    catch (const std::exception& error)
    {
        rejected = 1;
        ADD_FAILURE() << "Unexpected work-limit rejection: " << error.what();
    }
    int all_rejected = 0, any_work_budget = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &work_budget, &any_work_budget);
    EXPECT_EQ(all_rejected, 1);
    EXPECT_EQ(any_work_budget, 1);
}
std::uint64_t wire_word(std::span<const char> bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < 8)
        throw std::invalid_argument("Test injection did not find a complete wire word.");
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= std::uint64_t(static_cast<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
void overwrite_wire_word(AMR::AdaptiveBytes& bytes, std::size_t offset, std::uint64_t value)
{
    (void) wire_word(bytes, offset);
    for (unsigned i = 0; i < 8; ++i)
        bytes[offset + i] = char(value >> (8 * i));
}

struct SlabFixture
{
    AMR::AdaptiveBootstrap detached;
    unsigned n = 0, width = 0, sides = 0;
};
SlabFixture slab(unsigned n)
{
    const auto comm = Tpetra::getDefaultComm();
    SlabFixture result;
    result.n = n;
    result.width = n / comm->getSize();
    result.sides = unsigned(comm->getRank() > 0) + unsigned(comm->getRank() + 1 < comm->getSize());
    AMR::AMRGrid grid;
    grid.block_cells = {n, n, n};
    grid.extent = {double(n), double(n), double(n)};
    grid.max_level = 0;
    // Only compact detached recipes survive this scope. The native Cartesian
    // provider authenticates the inputs before the halo receives them.
    result.detached = [&]
    {
        AMR::CartesianAMR provider(grid);
        std::vector<std::uint64_t> selection;
        for (std::uint64_t z = 0; z < n; ++z)
            for (std::uint64_t y = 0; y < n; ++y)
                for (std::uint64_t x = comm->getRank() * result.width; x < (comm->getRank() + 1) * result.width; ++x)
                    selection.push_back(x + n * (y + n * z));
        return provider.detach_owned(selection, {'s'});
    }();
    return result;
}
D::ContactIncidenceRow corner_row(const AMR::AdaptiveLeaf& leaf, Wide lower, Wide upper)
{
    D::ContactIncidenceRow row{leaf.key, leaf.revision, leaf.domain, {}};
    row.points.reserve(8);
    for (unsigned mask = 0; mask < 8; ++mask)
    {
        Wide coordinate;
        for (unsigned axis = 0; axis < 3; ++axis)
            coordinate[axis] = mask & (1u << axis) ? upper[axis] : lower[axis];
        const auto point = P::details::GeometryIdentityAccess::point(coordinate);
        row.points.push_back({point, point});
    }
    return row;
}
std::vector<D::ContactIncidenceRow> slab_rows(const SlabFixture& fixture)
{
    std::vector<D::ContactIncidenceRow> rows;
    rows.reserve(fixture.detached.owned.size());
    for (const auto& leaf : fixture.detached.owned)
    {
        const auto& logical = std::get<AMR::CartesianCellAncestry>(leaf.key.ancestry);
        const Wide lower{static_cast<long double>(logical.index[0]), static_cast<long double>(logical.index[1]),
            static_cast<long double>(logical.index[2])};
        // The oracle's unit-spacing integer vertices do not call geometry,
        // contact_point, the spatial index, or legacy incidence authorities.
        rows.push_back(corner_row(leaf, lower, {lower[0] + 1, lower[1] + 1, lower[2] + 1}));
    }
    return rows;
}
std::uint64_t unique_pairs(const SlabFixture& fixture)
{
    const std::uint64_t n = fixture.n, w = fixture.width, transverse = (3 * n - 2) * (3 * n - 2);
    return (transverse * (3 * w - 2) - n * n * w) / 2 + fixture.sides * transverse;
}
std::uint64_t incidence_pairs(const SlabFixture& fixture)
{
    const std::uint64_t n = fixture.n, w = fixture.width, transverse = (4 * n - 2) * (4 * n - 2);
    return (transverse * (4 * w - 2) - 8 * n * n * w) / 2 + fixture.sides * transverse;
}
void check_slab(unsigned n, bool canonical = false, std::size_t byte_budget = cap64)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto fixture = slab(n);
    const auto& leaves = fixture.detached.owned;
    const auto expected_owned = std::size_t(n) * n * fixture.width;
    const auto expected_remote = std::size_t(n) * n * fixture.sides;
    EXPECT_EQ(leaves.size(), expected_owned);
    for (const auto& leaf : leaves)
    {
        EXPECT_FALSE(leaf.geometry);
        EXPECT_TRUE(leaf.surfaces.empty());
        EXPECT_TRUE(leaf.native_recipe);
    }
    AMR::AdaptiveClosureOptions options;
    options.byte_budget = byte_budget;
    D::DistributedAdaptiveContactHalo halo(*fixture.detached.kernel, comm, options, {7, 11, 13, 0});
    if (canonical)
    {
        std::map<AMR::AdaptiveCellKey, const AMR::AdaptiveLeaf*> source;
        for (const auto& leaf : leaves)
            source.emplace(leaf.key, &leaf);
        halo.bootstrap_canonical(leaves, [&](const auto& key) -> const AMR::AdaptiveLeaf& { return *source.at(key); });
    }
    else
        halo.bootstrap(leaves, slab_rows(fixture));
    EXPECT_TRUE(halo.required_refinements().empty());
    const auto statistics = halo.statistics();
    EXPECT_EQ(statistics.owned_records, expected_owned);
    EXPECT_EQ(statistics.remote_records, expected_remote);
    EXPECT_EQ(statistics.record_fetches, expected_remote);
    EXPECT_EQ(statistics.decoded_record_replies, expected_remote);
    EXPECT_EQ(statistics.unique_cell_pairs, unique_pairs(fixture));
    EXPECT_EQ(statistics.logical_contact_pairs, incidence_pairs(fixture));
    EXPECT_EQ(statistics.span_pair_evaluations, unique_pairs(fixture));
    EXPECT_EQ(statistics.trace_pair_evaluations, unique_pairs(fixture));
    EXPECT_EQ(statistics.work, 2 * unique_pairs(fixture));
    EXPECT_LE(statistics.retained_bytes, options.byte_budget);
    EXPECT_LE(statistics.peak_bytes, options.byte_budget);
    EXPECT_TRUE(halo.required_refinements().empty());
    const auto& repeated = halo.statistics();
    EXPECT_EQ(repeated.work, statistics.work);
    EXPECT_EQ(repeated.record_fetches, statistics.record_fetches);
    EXPECT_EQ(repeated.decoded_record_replies, statistics.decoded_record_replies);
    EXPECT_EQ(repeated.query_bytes, statistics.query_bytes);
    EXPECT_EQ(repeated.record_bytes, statistics.record_bytes);
    EXPECT_EQ(repeated.pages, statistics.pages);
    for (const auto& leaf : leaves)
    {
        EXPECT_FALSE(leaf.geometry);
        EXPECT_TRUE(leaf.surfaces.empty());
        EXPECT_EQ(leaf.revision, 0u);
        EXPECT_EQ(leaf.domain, (AMR::AdaptiveBytes{'s'}));
    }
    ::testing::Test::RecordProperty("slab_width", fixture.width);
    ::testing::Test::RecordProperty("remote_records", int(statistics.remote_records));
    ::testing::Test::RecordProperty("peak_bytes", int(statistics.peak_bytes));
    ::testing::Test::RecordProperty("retained_bytes", std::to_string(statistics.retained_bytes));
    ::testing::Test::RecordProperty("cumulative_bytes", std::to_string(statistics.cumulative_bytes));
    ::testing::Test::RecordProperty("owned_records", std::to_string(statistics.owned_records));
    ::testing::Test::RecordProperty("unique_cell_pairs", std::to_string(statistics.unique_cell_pairs));
    struct
    {
        std::size_t count = 0, point_storage_bytes = 0, shared_normalized_rows = 0;
    } rows;
    halo.for_each_incidence_row(
        [&rows](const D::ContactIncidenceRow& row)
        {
            ++rows.count;
            rows.point_storage_bytes += row.points.storage_bytes();
            rows.shared_normalized_rows += row.points.shares_normalized_storage();
        });
    ::testing::Test::RecordProperty("incidence_rows", std::to_string(rows.count));
    ::testing::Test::RecordProperty("incidence_point_storage_bytes", std::to_string(rows.point_storage_bytes));
    ::testing::Test::RecordProperty("incidence_shared_normalized_rows", std::to_string(rows.shared_normalized_rows));
}

TEST(SpatialAdaptiveClosure, CartesianEightCellAxisSlabsHaveExactSurfaceHalos)
{
    const auto size = Tpetra::getDefaultComm()->getSize();
    if (size > 8 || 8 % size)
        GTEST_SKIP() << "This slab oracle requires a rank count dividing eight.";
    check_slab(8);
}
TEST(SpatialAdaptiveClosure, CartesianEightCellCanonicalCacheHasExactSurfaceHalos)
{
    const auto size = Tpetra::getDefaultComm()->getSize();
    if (size > 8 || 8 % size)
        GTEST_SKIP() << "This slab oracle requires a rank count dividing eight.";
    check_slab(8, true);
}
TEST(SpatialAdaptiveClosure, CartesianSixteenCellAxisSlabsHaveExactSurfaceHalos)
{
    const auto size = Tpetra::getDefaultComm()->getSize();
    if (size > 16 || 16 % size)
        GTEST_SKIP() << "This slab oracle requires a rank count dividing sixteen.";
    check_slab(16);
}
// Explicit setup/storage observation only; no physical transport qualification.
TEST(SpatialAdaptiveClosure, CubicN32FourSlabHaloUnder64MiB)
{
    const auto comm = Tpetra::getDefaultComm();
    const int requested = std::getenv("SIMPLEFLUID_SPATIAL_CLOSURE_N32") != nullptr;
    int minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &requested, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &requested, &maximum);
    ASSERT_EQ(minimum, maximum);
    if (!requested)
        GTEST_SKIP() << "Set SIMPLEFLUID_SPATIAL_CLOSURE_N32 for the explicit N32 storage observation.";
    if (comm->getSize() != 4)
        GTEST_SKIP() << "The original N32 storage oracle requires exactly four ranks.";
    // check_slab independently requires 1024, 2048, 2048, 1024 remote records.
    check_slab(32);
}

TEST(SpatialAdaptiveClosure, CubicN16CanonicalCacheStorageObservation)
{
    const auto comm = Tpetra::getDefaultComm();
    const int requested = std::getenv("SIMPLEFLUID_SPATIAL_CLOSURE_N32") != nullptr;
    int minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &requested, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &requested, &maximum);
    ASSERT_EQ(minimum, maximum);
    if (!requested)
        GTEST_SKIP() << "Set SIMPLEFLUID_SPATIAL_CLOSURE_N32 for the explicit N32 storage observation.";
    if (comm->getSize() != 4)
        GTEST_SKIP() << "The canonical storage observation requires exactly four ranks.";
    // Independent N16 halo sizes are 256, 512, 512, 256 records.
    check_slab(16, true);
}

TEST(SpatialAdaptiveClosure, CubicN32CanonicalCacheUnder96MiB)
{
    const auto comm = Tpetra::getDefaultComm();
    const int requested = std::getenv("SIMPLEFLUID_SPATIAL_CLOSURE_N32") != nullptr;
    int minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &requested, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &requested, &maximum);
    ASSERT_EQ(minimum, maximum);
    if (!requested)
        GTEST_SKIP() << "Set SIMPLEFLUID_SPATIAL_CLOSURE_N32 for the explicit N32 storage observation.";
    if (comm->getSize() != 4)
        GTEST_SKIP() << "The original N32 storage oracle requires exactly four ranks.";
    // check_slab independently requires 1024, 2048, 2048, 1024 remote records.
    check_slab(32, true, cap96);
}

TEST(SpatialAdaptiveClosure, IsolatedEdgeVertexAndTJunctionMatchIndependentMarksAndLegacy)
{
    const auto coarse = box(0, {0, 0, 0}, {4, 4, 4});
    const std::array fine{box(1, {4, 4, 0}, {5, 5, 4}), box(1, {4, 4, 4}, {5, 5, 5}), box(1, {4, 4, 1}, {5, 5, 2})};
    for (const auto& leaf : fine)
        compare_with_reference(owned({coarse, leaf}), ContactKernel{}, coarse_on_rank_zero(coarse.key));
}
TEST(SpatialAdaptiveClosure, StrictPhysicalRatioAndTraceOnlyDemandMatchLegacy)
{
    const auto fine = box(1, {2, 2, 2}, {3, 3, 3});
    compare_with_reference(owned({box(0, {0, 0, 0}, {2, 2, 2}), fine}), ContactKernel{}, {});
    if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
    {
        const auto coarse = box(0, {-0x1p-54, 0, 0}, {2, 2, 2});
        compare_with_reference(owned({coarse, fine}), ContactKernel{}, coarse_on_rank_zero(coarse.key));
    }
    const auto shallow = box(0, {0, 0, 0}, {1, 1, 1}, 0);
    AMR::AdaptiveClosureOptions options;
    options.maximum_span_ratio = 100;
    compare_with_reference(owned({shallow, box(1, {1, 1, 1}, {2, 2, 2}, 3)}), ContactKernel{true},
        coarse_on_rank_zero(shallow.key), options);
}
TEST(SpatialAdaptiveClosure, PeriodicFaceAndCornerImagesMatchLegacyWithoutRefinement)
{
    const auto comm = Tpetra::getDefaultComm();
    for (const auto axes : {std::array{true, false, false}, std::array{true, true, true}})
    {
        AMR::AMRGrid grid;
        grid.block_cells = {2, 2, 2};
        grid.periodic = axes;
        const auto detached = [&]
        {
            AMR::CartesianAMR provider(grid);
            std::vector<std::uint64_t> selection;
            for (std::uint64_t cell = 0; cell < 8; ++cell)
                if (cell % comm->getSize() == std::uint64_t(comm->getRank()))
                    selection.push_back(cell);
            return provider.detach_owned(selection, {'p'});
        }();
        compare_with_reference(detached.owned, *detached.kernel, {});
    }
}
TEST(SpatialAdaptiveClosure, PeriodicNormalizationMappingRetainsFaceCornerAndPhysicalTraceDemands)
{
    // Isolated exterior shells make this a normalization-mapping fixture. The
    // native periodic overlay is independently covered by the Cartesian case.
    const auto coarse = box(0, {0, 0, 0}, {1, 1, 1});
    compare_with_reference(owned({coarse, box(1, {3.75, 0, 0}, {4, .25, .25})}),
        ContactKernel{false, {true, false, false}}, coarse_on_rank_zero(coarse.key));
    compare_with_reference(owned({coarse, box(1, {3.75, 3.75, 3.75}, {4, 4, 4})}),
        ContactKernel{false, {true, true, true}}, coarse_on_rank_zero(coarse.key));
    // Equal physical spans remove ratio grading. The shallow trace lives at
    // physical x=0 and the deep trace at x=4; both normalize to authority x=0.
    // Testing the deep face against the normalized point would lose this mark.
    compare_with_reference(owned({coarse, box(1, {3, 0, 0}, {4, 1, 1}, 3)}),
        ContactKernel{true, {true, false, false}, true}, coarse_on_rank_zero(coarse.key));
}
TEST(SpatialAdaptiveClosure, MixedRoutesRetainConnectedDomainSeparation)
{
    const auto pure = std::make_shared<const ContactKernel>();
    std::vector<AMR::DistributedMixedRoute> routes;
    std::vector<AMR::AdaptiveLeaf> cells{
        box(0, {0, 0, 0}, {4, 4, 4}), box(1, {4, 4, 4}, {5, 5, 5}), box(2, {4, 0, 0}, {4.25, .25, .25})};
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        auto& leaf = cells[i];
        const std::uint64_t route = i + 1, region = 100 + i;
        routes.push_back({route, 0, region, AMR::AdaptiveProviderKind::Polyhedral, pure, {}, false});
        AMR::AdaptiveJoinCapsule capsule;
        capsule.route = route;
        for (const auto& face : leaf.surfaces)
            capsule.boundaries.push_back({face.key, face.geometry, "wall"});
        leaf.join_recipe = std::make_shared<const AMR::AdaptiveBytes>(AMR::encode_join_capsule(capsule, cap64));
        leaf.key.region = region;
        if (i == 2)
            leaf.domain = {'x'};
    }
    const auto expected = coarse_on_rank_zero(cells[0].key);
    const auto kernel = AMR::mixed_adaptive_kernel(std::move(routes), cap64);
    compare_with_reference(owned(std::move(cells)), *kernel, expected);
    auto a = box(0, {0, 0, 0}, {4, 4, 4}), b = box(1, {4, 4, 4}, {5, 5, 5});
    b.domain = {'x'};
    compare_with_reference(owned({a, b}), *pure, {});
}

TEST(SpatialAdaptiveClosure, UniquePairWorkLimitIsCumulativeAndLeavesSourceImmutable)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() > 8 || 8 % comm->getSize())
        GTEST_SKIP() << "This slab oracle requires a rank count dividing eight.";
    const auto fixture = slab(8);
    const auto& source = fixture.detached.owned;
    const auto before = snapshot(source);
    const std::uint64_t local_work = 2 * unique_pairs(fixture);
    std::uint64_t required_cap = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_work, &required_cap);
    AMR::AdaptiveClosureOptions options;
    options.contact_budget = required_cap;
    {
        D::DistributedAdaptiveContactHalo exact(*fixture.detached.kernel, comm, options);
        exact.bootstrap(source, slab_rows(fixture));
        EXPECT_TRUE(exact.required_refinements().empty());
        EXPECT_EQ(exact.statistics().work, local_work);
        EXPECT_GT(exact.statistics().logical_contact_pairs, exact.statistics().unique_cell_pairs);
    }
    --options.contact_budget;
    expect_collective_work_rejection(
        [&]
        {
            D::DistributedAdaptiveContactHalo short_budget(*fixture.detached.kernel, comm, options);
            short_budget.bootstrap(source, slab_rows(fixture));
            (void) short_budget.required_refinements();
        });
    EXPECT_EQ(snapshot(source), before);
}
TEST(SpatialAdaptiveClosure, TraceWorkStaticDemandAndInvalidOptionsRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto shallow = box(0, {0, 0, 0}, {1, 1, 1}, 0);
    const auto source = owned({shallow, box(1, {1, 1, 0}, {2, 2, 1}, 3)});
    const auto before = snapshot(source);
    AMR::AdaptiveClosureOptions options;
    // Two shared edge endpoints establish one cell pair and one unrestricted
    // trace comparison. Per-point incidence is not a second numerical charge.
    options.contact_budget = 2;
    EXPECT_EQ(mark_set(AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options)),
        coarse_on_rank_zero(shallow.key));
    options.contact_budget = 1;
    expect_collective_work_rejection(
        [&] { (void) AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options); });
    options = {};
    options.byte_budget = 1;
    EXPECT_THROW(
        (void) AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options), std::invalid_argument);
    options = {};
    options.maximum_span_ratio = .5;
    EXPECT_THROW(
        (void) AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options), std::invalid_argument);
    options = {};
    options.maximum_trace_jump = 2;
    EXPECT_THROW(
        (void) AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options), std::invalid_argument);
    EXPECT_EQ(snapshot(source), before);
    auto fixed = box(0, {0, 0, 0}, {4, 4, 4});
    fixed.provider = AMR::AdaptiveProviderKind::Static;
    const auto static_source = owned({fixed, box(1, {4, 4, 4}, {5, 5, 5})});
    const auto static_before = snapshot(static_source);
    EXPECT_THROW(
        (void) AMR::required_adaptive_refinements(static_source, ContactKernel{}, comm), std::invalid_argument);
    EXPECT_EQ(snapshot(static_source), static_before);
    EXPECT_TRUE(AMR::required_adaptive_refinements({}, ContactKernel{}, comm).empty());
}

TEST(SpatialAdaptiveClosure, RankLocalRowIdentityFailuresRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    const ContactKernel kernel;
    const auto source = owned({box(0, {0, 0, 0}, {1, 1, 1}), box(1, {1, 1, 1}, {2, 2, 2})});
    const auto before = snapshot(source);
    const auto rows = [&]
    {
        std::vector<D::ContactIncidenceRow> result;
        for (const auto& leaf : source)
        {
            const auto id = std::get<AMR::CartesianCellAncestry>(leaf.key.ancestry).index[0];
            const auto low = static_cast<long double>(id);
            result.push_back(corner_row(leaf, {low, low, low}, {low + 1, low + 1, low + 1}));
        }
        return result;
    };
    for (unsigned defect = 0; defect < 5; ++defect)
    {
        SCOPED_TRACE(defect);
        auto invalid = rows();
        if (comm->getRank() == 0)
        {
            if (defect == 0)
                ++invalid.front().revision;
            else if (defect == 1)
                invalid.front().domain = {'x'};
            else if (defect == 2)
                ++invalid.front().key.region;
            else if (defect == 3)
                invalid.push_back(invalid.front());
            else
                invalid.erase(invalid.begin());
        }
        EXPECT_THROW(
            {
                D::DistributedAdaptiveContactHalo halo(kernel, comm, {});
                halo.bootstrap(source, std::move(invalid));
            },
            std::invalid_argument);
        EXPECT_EQ(snapshot(source), before);
    }
    D::DistributedAdaptiveContactHalo recovered(kernel, comm, {});
    recovered.bootstrap(source, rows());
    EXPECT_TRUE(recovered.required_refinements().empty());
}
TEST(SpatialAdaptiveClosure, RankDivergentSessionContextRejectsBeforeBootstrap)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Rank-divergent context requires at least two ranks.";
    const ContactKernel kernel;
    for (unsigned field = 0; field < 4; ++field)
    {
        D::ContactHaloContext context{1, 2, 3, 4};
        if (comm->getRank() == 0)
        {
            if (field == 0)
                ++context.epoch;
            else if (field == 1)
                ++context.topology;
            else if (field == 2)
                ++context.partition;
            else
                ++context.round;
        }
        EXPECT_THROW((void) D::DistributedAdaptiveContactHalo(kernel, comm, {}, context), std::invalid_argument);
    }
    D::DistributedAdaptiveContactHalo recovered(kernel, comm, {}, {1, 2, 3, 4});
    recovered.bootstrap({}, {});
    EXPECT_TRUE(recovered.required_refinements().empty());
}
TEST(SpatialAdaptiveClosure, WireFailuresRejectCollectivelyAndAllowCleanRetry)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Wire-failure rejection requires an actual remote owner.";
    const ContactKernel kernel;
    const double low = comm->getRank(), high = low + 1;
    const std::vector<AMR::AdaptiveLeaf> source{box(comm->getRank(), {low, low, low}, {high, high, high})};
    const auto before = snapshot(source);
    const auto rows = [&] { return std::vector{corner_row(source.front(), {low, low, low}, {high, high, high})}; };
    const D::ContactHaloContext context{7, 11, 13, 17};
    for (unsigned defect = 0; defect < 4; ++defect)
    {
        SCOPED_TRACE(defect);
        int injected = 0, rejected = 0;
        D::ContactHaloBeforeSend corrupt =
            [&](const D::ContactHaloContext&, D::ContactHaloMessageKind kind, auto& packets)
        {
            if (comm->getRank() + 1 != comm->getSize() || injected || packets.empty())
                return;
            auto& bytes = packets.front().bytes;
            // Private protocol v2 has thirteen little-endian uint64 header
            // words. Context epoch is word 3; final completion is word 12.
            constexpr std::size_t header_bytes = 13 * 8;
            if (defect == 0 && kind == D::ContactHaloMessageKind::Fan && wire_word(bytes, 12 * 8) == 1)
                packets.clear();
            else if (defect == 1 && kind == D::ContactHaloMessageKind::Records)
            {
                if (wire_word(bytes, header_bytes) == 0)
                    throw std::invalid_argument("Test injection expected a nonempty record request.");
                // Count, query, length-prefixed full key, revision. Change only
                // the requested revision, keeping framing and key intact.
                const auto key_bytes = wire_word(bytes, header_bytes + 2 * 8);
                if (key_bytes > bytes.size() - std::min(bytes.size(), header_bytes + 3 * 8))
                    throw std::invalid_argument("Test injection encountered a truncated record request key.");
                const auto revision_offset = header_bytes + 3 * 8 + key_bytes;
                overwrite_wire_word(bytes, revision_offset, wire_word(bytes, revision_offset) + 1);
            }
            else if (defect == 2 && kind == D::ContactHaloMessageKind::RecordData)
                overwrite_wire_word(bytes, 3 * 8, context.epoch + 1);
            else if (defect == 3 && kind == D::ContactHaloMessageKind::RecordData)
                bytes.pop_back();
            else
                return;
            ++injected;
        };
        try
        {
            D::DistributedAdaptiveContactHalo halo(kernel, comm, {}, context, std::move(corrupt));
            halo.bootstrap(source, rows());
            (void) halo.required_refinements();
        }
        catch (const std::invalid_argument&)
        {
            rejected = 1;
        }
        catch (const std::exception& error)
        {
            ADD_FAILURE() << "Wrong wire-rejection exception type: " << error.what();
        }
        int all_rejected = 0, total_injected = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &injected, &total_injected);
        EXPECT_EQ(all_rejected, 1);
        EXPECT_EQ(total_injected, 1);
        EXPECT_EQ(snapshot(source), before);
        D::DistributedAdaptiveContactHalo retry(kernel, comm, {}, context);
        retry.bootstrap(source, rows());
        EXPECT_TRUE(retry.required_refinements().empty());
        const auto neighbors = unsigned(comm->getRank() > 0) + unsigned(comm->getRank() + 1 < comm->getSize());
        EXPECT_EQ(retry.statistics().remote_records, neighbors);
        EXPECT_EQ(retry.statistics().record_fetches, neighbors);
        EXPECT_EQ(retry.statistics().decoded_record_replies, neighbors);
        EXPECT_EQ(snapshot(source), before);
    }
}
} // namespace
