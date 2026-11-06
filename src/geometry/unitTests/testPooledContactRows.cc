#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/IncidenceOwnerRows.hh"
#include "geometry/mesh/AMR/PooledCellPoints.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace P = SimpleFluid::PLIC;
using Point = D::ContactPointIdentity;
using Pool = D::IdentityPool<Point>;
constexpr std::size_t ample = 16 * 1024 * 1024;
Point point(long double x, long double y = 0, long double z = 0)
{
    return P::details::GeometryIdentityAccess::point({x, y, z});
}
A::AdaptiveLeaf leaf()
{
    A::AdaptiveLeaf result;
    result.key = {7, A::CartesianCellAncestry{0, {0, 0, 0}}};
    result.domain = {'r', 'o', 'w'};
    result.revision = 19;
    return result;
}
class Kernel final : public A::AdaptiveProviderKernel
{
public:
    enum Mode
    {
        Identity,
        MapTwelveToZero,
        DuplicateTwelveToOne
    };
    explicit Kernel(Mode mode = Identity) : mode_(mode) {}
    mutable std::vector<Point> calls;
    A::AdaptiveBytes descriptor() const override { return {'p', 'o', 'o', 'l'}; }
    P::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry(const A::AdaptiveLeaf&) const override
    {
        throw std::logic_error("Unused geometry.");
    }
    std::vector<A::AdaptiveSurface> surfaces(const A::AdaptiveLeaf&) const override
    {
        throw std::logic_error("Unused surfaces.");
    }
    A::AdaptiveSplitResult split(const A::AdaptiveLeaf&, A::AdaptiveSplit) const override
    {
        throw std::logic_error("Unused split.");
    }
    Point contact_point(const A::AdaptiveLeaf&, const Point& value) const override
    {
        calls.push_back(value);
        if (value == point(12) && mode_ != Identity)
            return point(mode_ == MapTwelveToZero ? 0 : 1);
        return value;
    }

private:
    Mode mode_;
};
// The array buffers belong to the caller, independently of the page directory.
// Destroy them before returning precisely their exclusive retained admission.
struct ArrayOwner
{
    std::shared_ptr<D::Budget> budget;
    D::ContactPointArray value;
    explicit ArrayOwner(std::shared_ptr<D::Budget> b) : budget(std::move(b)) {}
    ~ArrayOwner()
    {
        const auto bytes = value.storage_bytes();
        value = {};
        budget->used -= bytes;
    }
};
struct RowOwner
{
    std::shared_ptr<D::Budget> budget;
    D::ContactIncidenceRow value;
    RowOwner(std::shared_ptr<D::Budget> b, D::ContactIncidenceRow row) : budget(std::move(b)), value(std::move(row)) {}
    ~RowOwner()
    {
        const auto bytes = D::key_capacity(value.key) + value.domain.capacity() + value.points.storage_bytes();
        value = {};
        budget->used -= bytes;
    }
};
void expect_point(const Point& actual, const Point& expected)
{
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(P::encode_identity(actual), P::encode_identity(expected));
}
// These fixtures use physical page batches. Identity reads return values, so
// inspect their captured page references when checking actual record sharing.
const Point* borrowed_stored_point(const D::ContactPointArray& array, std::size_t index, bool normalized = false)
{
    const auto locator = normalized ? array.normalized_locator(index) : array.physical_locator(index);
    if (!locator || !locator->view || locator->source_view)
        throw std::logic_error("Fixture requires a captured physical page reference.");
    return &locator->view->get(locator->reference);
}

TEST(PooledContactRows, PartialPageForksKeepLiteralValuesAndIndependentArenaLifetimes)
{
    for (const auto count : {1u, 15u, 16u, 17u})
    {
        SCOPED_TRACE(count);
        auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto candidate_budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto sibling_budget = std::make_shared<D::Budget>(D::Budget{ample});
        std::weak_ptr<D::Budget> source_weak = source_budget;
        auto source = std::make_unique<Pool>(*source_budget, source_budget);
        std::vector<const Point*> prefix;
        for (unsigned i = 0; i < count; ++i)
        {
            EXPECT_EQ(source->intern(point(i)), i);
            prefix.push_back(&source->get(i));
        }
        auto candidate = std::make_unique<Pool>(*source, *candidate_budget, candidate_budget);
        auto sibling = std::make_unique<Pool>(*source, *sibling_budget, sibling_budget);
        EXPECT_EQ(source->intern(point(1001)), count);
        EXPECT_EQ(candidate->intern(point(2001)), count);
        EXPECT_EQ(sibling->intern(point(3001)), count);
        auto source_view = source->page_view();
        auto candidate_view = candidate->page_view();
        auto sibling_view = sibling->page_view();
        std::vector<Pool::CapturedReference> candidate_prefix;
        for (unsigned i = 0; i < count; ++i)
        {
            candidate_prefix.push_back(candidate_view->capture(*candidate, i));
            EXPECT_EQ(&source->get(i), prefix[i]);
            EXPECT_EQ(&candidate->get(i), prefix[i]);
            EXPECT_EQ(&sibling->get(i), prefix[i]);
        }
        const auto source_tail = source_view->capture(*source, count);
        const auto candidate_tail = candidate_view->capture(*candidate, count);
        const auto sibling_tail = sibling_view->capture(*sibling, count);
        EXPECT_THROW((void) candidate_view->get(candidate_tail), std::invalid_argument);
        source_view->freeze();
        candidate_view->freeze();
        sibling_view->freeze();
        source.reset();
        source_budget.reset();
        EXPECT_FALSE(source_weak.expired());
        expect_point(source_view->get(source_tail), point(1001));
        source_view.reset();
        EXPECT_TRUE(source_weak.expired());
        candidate.reset();
        sibling.reset();
        for (unsigned i = 0; i < count; ++i)
        {
            expect_point(candidate_view->get(candidate_prefix[i]), point(i));
            EXPECT_EQ(&candidate_view->get(candidate_prefix[i]), prefix[i]);
        }
        expect_point(candidate_view->get(candidate_tail), point(2001));
        expect_point(sibling_view->get(sibling_tail), point(3001));
        candidate_view.reset();
        sibling_view.reset();
        EXPECT_EQ(candidate_budget->used, 0u);
        EXPECT_EQ(sibling_budget->used, 0u);
    }
}

TEST(PooledContactRows, PhysicalOrderPrecedesNormalizationAndRowsBorrowPoolIdentities)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::IdentityBudget identities;
        D::IncidencePools pools(*budget, identities, budget);
        D::PooledCellPoints points(*budget, pools);
        const auto precise = std::nextafter(1.L, 2.L);
        const std::array<Point, 4> arrivals{point(12), point(1), point(precise), point(8)};
        for (const auto& value : arrivals)
            points.add(pools.points.intern(value));
        Kernel kernel(Kernel::MapTwelveToZero);
        D::ContactPointBatch batch(pools.points);
        RowOwner row(budget, points.row(leaf(), kernel, batch));
        batch.freeze();
        const std::array<Point, 4> expected_physical{point(1), point(precise), point(8), point(12)};
        const std::array<Point, 4> expected_normalized{point(1), point(precise), point(8), point(0)};
        ASSERT_EQ(kernel.calls.size(), 4u);
        ASSERT_EQ(row.value.points.size(), 4u);
        EXPECT_TRUE(row.value.points.pooled());
        EXPECT_FALSE(row.value.points.shares_normalized_storage());
        for (std::size_t i = 0; i < expected_physical.size(); ++i)
        {
            expect_point(kernel.calls[i], expected_physical[i]);
            const auto actual = row.value.points[i];
            expect_point(actual.physical, expected_physical[i]);
            expect_point(actual.normalized, expected_normalized[i]);
            EXPECT_EQ(borrowed_stored_point(row.value.points, i),
                &pools.points.physical_pool().get(*pools.points.find(expected_physical[i])));
            EXPECT_EQ(borrowed_stored_point(row.value.points, i, true),
                &pools.points.physical_pool().get(*pools.points.find(expected_normalized[i])));
        }
        EXPECT_EQ(row.value.points.storage_bytes(), 8 * sizeof(std::uint32_t));
        EXPECT_EQ(batch.page_count(), 1u); // Capture did not seal a page between normalizations.
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, FullPrecisionCanonicalZeroAndFloatingEnvironmentSurviveFreezeAndRead)
{
    struct Environment
    {
        std::fenv_t saved;
        Environment() { std::fegetenv(&saved); }
        ~Environment() { std::fesetenv(&saved); }
    } environment;
    const auto precise = std::nextafter(1.L, 2.L);
    ASSERT_EQ(static_cast<double>(precise), 1.);
    const auto first = point(1, -0.L, 0.L), second = point(precise, 0.L, -0.L);
    ASSERT_NE(first, second);
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        Pool pool(*budget, budget);
        std::fesetround(FE_UPWARD);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
        const auto a = pool.intern(first), b = pool.intern(second);
        D::ContactPointBatch batch(pool);
        ArrayOwner array(budget);
        array.value = D::ContactPointArray::pooled(batch, 2, *budget);
        const auto ra = batch.capture(a), rb = batch.capture(b);
        array.value.push_reference(ra, ra, *budget);
        array.value.push_reference(rb, rb, *budget);
        batch.freeze();
        EXPECT_TRUE(array.value.shares_normalized_storage());
        EXPECT_EQ(borrowed_stored_point(array.value, 0), borrowed_stored_point(array.value, 0, true));
        EXPECT_EQ(borrowed_stored_point(array.value, 1), &pool.get(b));
        EXPECT_EQ(array.value[0].physical, first);
        EXPECT_EQ(array.value[1].physical, second);
        EXPECT_EQ(std::fegetround(), FE_UPWARD);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
        expect_point(array.value[0].physical, point(1, 0.L, -0.L));
        expect_point(array.value[1].physical, second);
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, ReferencedPagesAndSharedDirectoryOutlivePoolWithoutDuplicatePageCredit)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        auto pool = std::make_unique<Pool>(*budget, budget);
        for (unsigned i = 0; i < 33; ++i)
            (void) pool->intern(point(i));
        D::ContactPointBatch batch(*pool);
        ArrayOwner first(budget), second(budget);
        first.value = D::ContactPointArray::pooled(batch, 1, *budget);
        second.value = D::ContactPointArray::pooled(batch, 1, *budget);
        const auto reference = batch.capture(17);
        first.value.push_reference(reference, reference, *budget);
        const auto after_first = budget->used;
        const auto duplicate = batch.capture(17);
        EXPECT_EQ(budget->used, after_first);
        second.value.push_reference(duplicate, duplicate, *budget);
        EXPECT_EQ(first.value.page_view().get(), second.value.page_view().get());
        batch.freeze();
        ASSERT_EQ(batch.page_count(), 1u);
        const auto retained = budget->used;
        const auto metadata = pool->metadata_bytes();
        pool.reset();
        EXPECT_LT(budget->used, retained - metadata); // The two unreferenced pages were released.
        expect_point(first.value[0].physical, point(17));
        expect_point(second.value[0].physical, point(17));
        EXPECT_EQ(borrowed_stored_point(first.value, 0), borrowed_stored_point(second.value, 0));
        EXPECT_GT(budget->used, first.value.storage_bytes() + second.value.storage_bytes());
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, EveryPoolPaysCreditMetadataAndForkedCreditsRemainDistinctOwners)
{
    auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
    auto candidate_budget = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget source_identities, candidate_identities;
    D::IncidencePools source(*source_budget, source_identities, source_budget);
    (void) source.cells.intern(leaf().key);
    (void) source.domains.intern({'d'});
    (void) source.supports.intern({'s'});
    (void) source.lines.intern({'l'});
    const auto slot = source.points.intern(point(1));
    (void) source.planes.intern(P::details::GeometryIdentityAccess::support_axis(0, 0));
    (void) source.normals.intern(D::IncidencePools::Normal({1, 0, 0}));
    (void) source.images.intern(P::details::GeometryIdentityAccess::displacement({0, 0, 0}));
    D::ContactPointBatch source_batch(source.points);
    const auto reference = source_batch.capture(slot);
    source_batch.freeze();
    const void* source_credit = nullptr;
    std::size_t credit_bytes = 0;
    source_batch.view()->visit_credit_owners(
        [&](const auto& owner, std::size_t bytes)
        {
            ASSERT_EQ(source_credit, nullptr);
            source_credit = owner.get();
            credit_bytes = bytes;
        });
    ASSERT_NE(source_credit, nullptr);
    EXPECT_EQ(source.page_credit_overhead_bytes(), 8 * credit_bytes);
    EXPECT_TRUE(source.points.owns_page_credit(source_credit));
    EXPECT_EQ(source_batch.view()->origin_arena(), source_budget.get());

    D::IncidencePools candidate(source, *candidate_budget, candidate_identities, candidate_budget);
    EXPECT_EQ(candidate.page_credit_overhead_bytes(), 8 * credit_bytes);
    EXPECT_FALSE(candidate.points.owns_page_credit(source_credit));
    D::ContactPointBatch candidate_batch(candidate.points);
    const auto copied = candidate_batch.capture(slot);
    candidate_batch.freeze();
    EXPECT_EQ(&candidate_batch.view()->get(copied), &source_batch.view()->get(reference));
    EXPECT_EQ(candidate_batch.view()->origin_arena(), candidate_budget.get());
    candidate_batch.view()->visit_credit_owners(
        [&](const auto& owner, std::size_t bytes)
        {
            EXPECT_NE(owner.get(), source_credit);
            EXPECT_TRUE(candidate.points.owns_page_credit(owner.get()));
            EXPECT_FALSE(source.points.owns_page_credit(owner.get()));
            EXPECT_EQ(bytes, credit_bytes);
        });
}

TEST(PooledContactRows, NarrowBoundaryWidensExactlyAndRejectsInvalidLookup)
{
    constexpr auto maximum = std::uint64_t{std::numeric_limits<std::uint32_t>::max()};
    EXPECT_FALSE(D::ContactPointArray::reference_uses_wide(maximum - 1));
    EXPECT_FALSE(D::ContactPointArray::reference_uses_wide(maximum));
    EXPECT_TRUE(D::ContactPointArray::reference_uses_wide(maximum + 1));
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        Pool pool(*budget, budget);
        (void) pool.intern(point(4));
        D::ContactPointBatch batch(pool);
        ArrayOwner array(budget);
        array.value = D::ContactPointArray::pooled(batch, 3, *budget);
        array.value.push_reference(Pool::CapturedReference{maximum - 1}, Pool::CapturedReference{maximum - 1}, *budget);
        array.value.push_reference(Pool::CapturedReference{maximum}, Pool::CapturedReference{maximum}, *budget);
        EXPECT_EQ(array.value.storage_bytes(), 3 * sizeof(std::uint32_t));
        array.value.push_reference(Pool::CapturedReference{maximum + 1}, Pool::CapturedReference{maximum + 1}, *budget);
        EXPECT_EQ(array.value.storage_bytes(), 3 * sizeof(std::uint64_t));
        EXPECT_EQ(array.value.size(), 3u);
        batch.freeze();
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_THROW((void) array.value.at(i), std::out_of_range);
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, FailedSecondWideBufferPreservesExistingValuesAndNeverRefundsCumulative)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        Pool pool(*budget, budget);
        const auto a = pool.intern(point(3)), b = pool.intern(point(9));
        D::ContactPointBatch batch(pool);
        ArrayOwner array(budget);
        array.value = D::ContactPointArray::pooled(batch, 2, *budget);
        array.value.push_reference(batch.capture(a), batch.capture(b), *budget);
        const auto before = budget->used, cumulative = budget->cumulative;
        const auto old_bytes = array.value.storage_bytes();
        budget->limit = before + 2 * sizeof(std::uint64_t);
        constexpr auto wide = std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1;
        EXPECT_THROW(array.value.push_reference(Pool::CapturedReference{wide}, Pool::CapturedReference{wide}, *budget),
            std::invalid_argument);
        EXPECT_EQ(budget->used, before);
        EXPECT_GT(budget->cumulative, cumulative);
        EXPECT_EQ(array.value.capacity(), 2u);
        EXPECT_EQ(array.value.size(), 1u);
        EXPECT_EQ(array.value.storage_bytes(), old_bytes);
        batch.freeze();
        expect_point(array.value[0].physical, point(3));
        expect_point(array.value[0].normalized, point(9));
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, ExactObservedPeakAdmitsAndOneByteLessUnwindsEveryExclusiveOwner)
{
    constexpr auto caller_live = 107u;
    const auto exercise = [caller_live](std::size_t limit, bool reject)
    {
        auto budget = std::make_shared<D::Budget>(D::Budget{limit});
        budget->add(caller_live);
        const auto run = [&]
        {
            D::IdentityBudget identities;
            D::IncidencePools pools(*budget, identities, budget);
            D::PooledCellPoints points(*budget, pools);
            for (const auto x : {12, 1, 8})
                points.add(pools.points.intern(point(x)));
            Kernel kernel(Kernel::MapTwelveToZero);
            D::ContactPointBatch batch(pools.points);
            RowOwner row(budget, points.row(leaf(), kernel, batch));
            batch.freeze();
            EXPECT_EQ(row.value.points.size(), 3u);
        };
        if (reject)
            EXPECT_THROW(run(), std::invalid_argument);
        else
            EXPECT_NO_THROW(run());
        EXPECT_EQ(budget->used, caller_live);
        EXPECT_GT(budget->cumulative, caller_live);
        return budget->peak;
    };
    const auto required = exercise(ample, false);
    EXPECT_EQ(exercise(required, false), required);
    EXPECT_LE(exercise(required - 1, true), required - 1);
    RecordProperty("required_peak_bytes", std::to_string(required));
}

TEST(PooledContactRows, LateDuplicateNormalizationUnwindsCandidateAndPreservesAcceptedView)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::IdentityBudget identities;
        D::IncidencePools pools(*budget, identities, budget);
        const auto accepted_slot = pools.points.intern(point(1));
        D::ContactPointBatch accepted(pools.points);
        const auto reference = accepted.capture(accepted_slot);
        accepted.freeze();
        const auto* address = &accepted.view()->get(reference);
        const auto packet = P::encode_identity(accepted.view()->get(reference));
        D::PooledCellPoints points(*budget, pools);
        for (const auto x : {1, 8, 12})
            points.add(pools.points.intern(point(x)));
        Kernel kernel(Kernel::DuplicateTwelveToOne);
        const auto before = budget->used, cumulative = budget->cumulative;
        {
            D::ContactPointBatch candidate(pools.points);
            EXPECT_THROW((void) points.row(leaf(), kernel, candidate), std::invalid_argument);
        }
        EXPECT_EQ(budget->used, before);
        EXPECT_GT(budget->cumulative, cumulative);
        ASSERT_EQ(kernel.calls.size(), 3u);
        expect_point(kernel.calls[0], point(1));
        expect_point(kernel.calls[1], point(8));
        expect_point(kernel.calls[2], point(12));
        EXPECT_EQ(&accepted.view()->get(reference), address);
        EXPECT_EQ(P::encode_identity(accepted.view()->get(reference)), packet);
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, CopiedAncestryPaysItsActualCapacityAndReleasesTheWholeRowCredit)
{
    auto source = leaf();
    source.key.ancestry = A::ConvexCellAncestry{17, {3, 5}};
    auto& path = std::get<A::ConvexCellAncestry>(source.key.ancestry).path;
    path.reserve(64);
    ASSERT_GT(path.capacity(), path.size());
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::IdentityBudget identities;
        D::IncidencePools pools(*budget, identities, budget);
        D::PooledCellPoints points(*budget, pools);
        points.add(pools.points.intern(point(1)));
        points.add(pools.points.intern(point(8)));
        Kernel kernel;
        const auto before = budget->used;
        {
            D::ContactPointBatch batch(pools.points);
            RowOwner row(budget, points.row(source, kernel, batch));
            batch.freeze();
            EXPECT_EQ(row.value.key, source.key);
            const auto& copied = std::get<A::ConvexCellAncestry>(row.value.key.ancestry).path;
            EXPECT_EQ(copied.capacity(), copied.size());
            EXPECT_EQ(D::key_capacity(row.value.key), 2 * sizeof(std::uint64_t));
        }
        EXPECT_EQ(budget->used, before);
    }
    EXPECT_EQ(budget->used, 0u);
}

// Patch handles own their row admissions; their vector allocation is retained
// separately by finish's caller.
struct PatchOwner
{
    std::shared_ptr<D::Budget> budget;
    std::vector<D::IncidenceOwnerRows::Patch> value;
    PatchOwner(std::shared_ptr<D::Budget> b, std::vector<D::IncidenceOwnerRows::Patch> patches)
        : budget(std::move(b)), value(std::move(patches))
    {
    }
    ~PatchOwner()
    {
        const auto bytes = value.capacity() * sizeof(D::IncidenceOwnerRows::Patch);
        std::vector<D::IncidenceOwnerRows::Patch>().swap(value);
        budget->used -= bytes;
    }
};

TEST(PooledContactRows, NoOpFinishAtExactLiveBudgetAllocatesNothingAndNeverLooksUpLeaf)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::IdentityBudget identities;
        D::IncidencePools pools(*budget, identities, budget);
        Kernel kernel;
        D::IncidenceOwnerRows owners(*budget, identities, pools, kernel, 0, budget);
        const auto cell = leaf();
        D::PooledTile tile;
        tile.first_cell = pools.cells.intern(cell.key);
        const std::array<std::size_t, 2> loop{pools.points.intern(point(2)), pools.points.intern(point(5))};
        owners.insert(tile.first_cell);
        owners.add(tile, loop);
        {
            PatchOwner published(
                budget, owners.finish([&](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf& { return cell; }));
            ASSERT_EQ(published.value.size(), 1u);
            expect_point(published.value[0].row->points[0].physical, point(2));
        }
        unsigned lookups = 0;
        const auto forbidden_lookup = [&](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf&
        {
            ++lookups;
            throw std::logic_error("No-op finish must not invoke lookup.");
        };
        owners.begin_delta();
        budget->limit = budget->used;
        auto cumulative = budget->cumulative;
        {
            PatchOwner unchanged(budget, owners.finish(forbidden_lookup));
            EXPECT_TRUE(unchanged.value.empty());
            EXPECT_EQ(unchanged.value.capacity(), 0u);
        }
        EXPECT_EQ(budget->cumulative, cumulative);
        EXPECT_EQ(lookups, 0u);
        // Exercise an explicitly touched cell whose net physical set is equal.
        budget->limit = ample;
        owners.begin_delta();
        owners.remove(tile, loop);
        owners.add(tile, loop);
        budget->limit = budget->used;
        cumulative = budget->cumulative;
        {
            PatchOwner unchanged(budget, owners.finish(forbidden_lookup));
            EXPECT_TRUE(unchanged.value.empty());
            EXPECT_EQ(unchanged.value.capacity(), 0u);
        }
        EXPECT_EQ(budget->cumulative, cumulative);
        EXPECT_EQ(lookups, 0u);
        EXPECT_EQ(kernel.calls.size(), 2u);
    }
    EXPECT_EQ(budget->used, 0u);
}

TEST(PooledContactRows, SecondRowLookupFailureReleasesStagedRowsAndPreservesAcceptedSource)
{
    auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    constexpr std::size_t caller_live = 293;
    budget->add(caller_live);
    D::IdentityBudget source_identities;
    auto source = std::make_unique<D::IncidencePools>(*source_budget, source_identities, source_budget);
    const auto source_slot = source->points.intern(point(1));
    auto accepted = source->points.page_view();
    const auto accepted_reference = accepted->capture(source->points, source_slot);
    accepted->freeze();
    const auto* accepted_address = &accepted->get(accepted_reference);
    const auto accepted_packet = P::encode_identity(accepted->get(accepted_reference));
    Kernel kernel;
    std::size_t cumulative_after_failure = 0;
    {
        D::IdentityBudget identities;
        D::IncidencePools pools(*source, *budget, identities, budget);
        D::IncidenceOwnerRows owners(*budget, identities, pools, kernel, 0, budget);
        const auto first = leaf();
        auto second = leaf();
        second.key = {7, A::CartesianCellAncestry{0, {1, 0, 0}}};
        D::PooledTile first_tile, second_tile;
        first_tile.first_cell = pools.cells.intern(first.key);
        second_tile.first_cell = pools.cells.intern(second.key);
        const std::array<std::size_t, 2> first_loop{pools.points.intern(point(1)), pools.points.intern(point(8))};
        const std::array<std::size_t, 2> second_loop{pools.points.intern(point(12)), pools.points.intern(point(14))};
        owners.insert(first_tile.first_cell);
        owners.insert(second_tile.first_cell);
        owners.add(first_tile, first_loop);
        owners.add(second_tile, second_loop);
        unsigned lookups = 0;
        const auto lookup = [&](const A::AdaptiveCellKey& key) -> const A::AdaptiveLeaf&
        {
            ++lookups;
            if (key == first.key)
                return first;
            EXPECT_EQ(key, second.key);
            // The first normalized row has been staged before the second lookup.
            EXPECT_EQ(kernel.calls.size(), 2u);
            throw std::invalid_argument("Injected second-row leaf lookup failure.");
        };
        EXPECT_THROW((void) owners.finish(lookup), std::invalid_argument);
        EXPECT_EQ(lookups, 2u);
        ASSERT_EQ(kernel.calls.size(), 2u);
        expect_point(kernel.calls[0], point(1));
        expect_point(kernel.calls[1], point(8));
        cumulative_after_failure = budget->cumulative;
    }
    EXPECT_EQ(budget->used, caller_live);
    EXPECT_EQ(budget->cumulative, cumulative_after_failure);
    source.reset();
    source_budget.reset();
    EXPECT_EQ(&accepted->get(accepted_reference), accepted_address);
    expect_point(accepted->get(accepted_reference), point(1));
    EXPECT_EQ(P::encode_identity(accepted->get(accepted_reference)), accepted_packet);
    EXPECT_EQ(budget->used, caller_live);
}

TEST(PooledContactRows, MultiplicityOnlyEditsPersistAcrossForksWithoutPublishingOrBumpingGeneration)
{
    auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget source_identities;
    D::IncidencePools source_pools(*source_budget, source_identities, source_budget);
    Kernel kernel;
    D::IncidenceOwnerRows source(*source_budget, source_identities, source_pools, kernel, 0, source_budget);
    const auto cell = leaf();
    D::PooledTile tile;
    tile.first_cell = source_pools.cells.intern(cell.key);
    const auto first = source_pools.points.intern(point(2));
    const auto second = source_pools.points.intern(point(5));
    const std::array loop{first, second};
    const std::array first_only{first};
    const auto lookup = [&](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf& { return cell; };
    const auto forbidden = [](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf&
    { throw std::logic_error("Multiplicity-only edit must not look up the leaf."); };
    source.insert(tile.first_cell, 10);
    source.add(tile, loop);
    {
        PatchOwner patches(source_budget, source.finish(lookup));
        ASSERT_EQ(patches.value.size(), 1u);
        EXPECT_EQ(patches.value.front().row->incidence_generation, 11u);
    }

    auto second_budget = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget second_identities;
    D::IncidencePools second_pools(source_pools, *second_budget, second_identities, second_budget);
    D::IncidenceOwnerRows candidate(source, *second_budget, second_identities, second_pools, second_budget);
    candidate.begin_delta();
    candidate.add(tile, first_only);
    {
        PatchOwner patches(second_budget, candidate.finish(forbidden));
        EXPECT_TRUE(patches.value.empty());
    }
    EXPECT_GT(candidate.inherited_overlap_metadata_bytes(), 0u);
    const auto before = second_budget->used;
    const auto retired = candidate.inherited_overlap_metadata_bytes();
    candidate.finish_inherited_overlap();
    EXPECT_EQ(second_budget->used, before - retired);

    auto third_budget = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget third_identities;
    D::IncidencePools third_pools(second_pools, *third_budget, third_identities, third_budget);
    D::IncidenceOwnerRows third(candidate, *third_budget, third_identities, third_pools, third_budget);
    third.begin_delta();
    third.remove(tile, first_only);
    {
        PatchOwner patches(third_budget, third.finish(forbidden));
        EXPECT_TRUE(patches.value.empty());
    }
    third.begin_delta();
    third.remove(tile, first_only);
    {
        PatchOwner patches(third_budget, third.finish(lookup));
        ASSERT_EQ(patches.value.size(), 1u);
        EXPECT_EQ(patches.value.front().row->incidence_generation, 12u);
        ASSERT_EQ(patches.value.front().row->points.size(), 1u);
        expect_point(patches.value.front().row->points[0].physical, point(5));
    }
    source.begin_delta();
    source.remove(tile, first_only);
    PatchOwner original(source_budget, source.finish(lookup));
    ASSERT_EQ(original.value.size(), 1u);
    EXPECT_EQ(original.value.front().row->incidence_generation, 12u);
}

TEST(PooledContactRows, OwnerRowsForkSharesActualPagesAndCountsAndCopiesOneBoundedPath)
{
    struct Measure
    {
        std::size_t new_payload, cumulative, copied_pages, copied_bytes;
    };
    const auto run = [](std::size_t count, Measure& measure)
    {
        constexpr auto cap = 16 * ample;
        auto source_budget = std::make_shared<D::Budget>(D::Budget{cap});
        D::IdentityBudget identities;
        D::IncidencePools pools(*source_budget, identities, source_budget, D::IdentityPoolStorage::Persistent);
        Kernel kernel;
        D::IncidenceOwnerRows source(*source_budget, identities, pools, kernel, 0, source_budget);
        auto cell = leaf();
        const std::array loop{pools.points.intern(point(2)), pools.points.intern(point(5))};
        D::PooledTile edited;
        for (std::size_t n = 0; n != count; ++n)
        {
            cell.key = {7, A::CartesianCellAncestry{0, {n, 0, 0}}};
            D::PooledTile tile;
            tile.first_cell = pools.cells.intern(cell.key);
            source.insert(tile.first_cell);
            source.add(tile, loop);
            if (n == 2)
                edited = tile;
        }
        const auto lookup = [&](const A::AdaptiveCellKey& key) -> const A::AdaptiveLeaf&
        {
            cell.key = key;
            return cell;
        };
        {
            PatchOwner initial(source_budget, source.finish(lookup));
            ASSERT_EQ(initial.value.size(), count);
        }
        using Identity = std::pair<D::ContactPayloadRole, std::uintptr_t>;
        const auto identity = [](const D::ContactPayloadAllocation& allocation)
        { return Identity{allocation.identity.role, reinterpret_cast<std::uintptr_t>(allocation.identity.address)}; };
        std::set<Identity> accepted;
        std::size_t counts = 0;
        source.visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& allocation)
            {
                accepted.insert(identity(allocation));
                if (allocation.identity.role == D::ContactPayloadRole::CellPointCounts)
                    ++counts;
            });
        ASSERT_EQ(counts, count);

        auto budget = std::make_shared<D::Budget>(D::Budget{cap});
        D::IdentityBudget candidate_identities;
        D::IncidencePools candidate_pools(pools, *budget, candidate_identities, budget);
        const auto base = budget->cumulative;
        D::IncidenceOwnerRows candidate(source, *budget, candidate_identities, candidate_pools, budget);
        EXPECT_EQ(candidate.persistence_work().copied_cells, 0u);
        EXPECT_EQ(candidate.persistence_work().copied_pages, 0u);
        EXPECT_EQ(candidate.persistence_work().legacy_fork_visits, 0u);
        EXPECT_EQ(budget->cumulative - base, source.metadata_bytes());
        std::size_t fork_private = 0;
        candidate.visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& allocation)
            {
                if (!accepted.contains(identity(allocation)))
                    fork_private += allocation.payload_bytes;
            });
        EXPECT_EQ(fork_private, source.source_overlap_metadata_bytes());

        const auto edit_base = budget->cumulative;
        candidate.begin_delta();
        candidate.add(edited, std::array{loop[0]});
        const auto forbidden = [](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf&
        { throw std::logic_error("Multiplicity-only edit must not look up the leaf."); };
        {
            PatchOwner patches(budget, candidate.finish(forbidden));
            EXPECT_TRUE(patches.value.empty());
        }
        candidate.finish_inherited_overlap();
        EXPECT_EQ(candidate.persistence_work().copied_cells, 1u);
        EXPECT_EQ(candidate.persistence_work().legacy_fork_visits, 0u);
        std::size_t new_payload = 0;
        std::size_t new_counts = 0;
        candidate.visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& allocation)
            {
                if (accepted.contains(identity(allocation)))
                    return;
                new_payload += allocation.payload_bytes;
                if (allocation.identity.role == D::ContactPayloadRole::CellPointCounts)
                    ++new_counts;
            });
        EXPECT_EQ(new_counts, 1u);
        const auto work = candidate.persistence_work();
        measure = {new_payload, budget->cumulative - edit_base, work.copied_pages, work.copied_bytes};
    };
    Measure small{}, large{};
    ASSERT_NO_FATAL_FAILURE(run(4096, small));
    ASSERT_NO_FATAL_FAILURE(run(8192, large));
    EXPECT_GT(small.copied_pages, 0u);
    EXPECT_EQ(large.copied_pages, small.copied_pages);
    EXPECT_LE(large.copied_bytes, small.copied_bytes + 64 * sizeof(void*));
    EXPECT_LE(large.new_payload, small.new_payload + 64 * sizeof(void*));
    EXPECT_LE(large.cumulative, small.cumulative + 64 * sizeof(void*));
}

TEST(PooledContactRows, OwnerRowsOverlapPeakRejectsOneByteLessAndLeavesAcceptedCountsIntact)
{
    auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
    D::IdentityBudget identities;
    D::IncidencePools pools(*source_budget, identities, source_budget);
    Kernel kernel;
    D::IncidenceOwnerRows source(*source_budget, identities, pools, kernel, 0, source_budget);
    const auto cell = leaf();
    D::PooledTile tile;
    tile.first_cell = pools.cells.intern(cell.key);
    const std::array loop{pools.points.intern(point(2)), pools.points.intern(point(5))};
    const auto lookup = [&](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf& { return cell; };
    source.insert(tile.first_cell);
    source.add(tile, loop);
    {
        PatchOwner accepted(source_budget, source.finish(lookup));
        ASSERT_EQ(accepted.value.size(), 1u);
    }
    const auto source_live = source_budget->used;
    const auto run = [&](std::size_t limit, bool reject)
    {
        auto budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::IdentityBudget candidate_identities;
        D::IncidencePools candidate_pools(pools, *budget, candidate_identities, budget);
        const auto baseline = budget->used;
        budget->peak = baseline;
        budget->limit = limit;
        const auto edit = [&]
        {
            D::IncidenceOwnerRows candidate(source, *budget, candidate_identities, candidate_pools, budget);
            candidate.begin_delta();
            candidate.add(tile, std::array{loop[0]});
            PatchOwner patches(budget, candidate.finish(lookup));
            EXPECT_TRUE(patches.value.empty());
            EXPECT_GT(candidate.inherited_overlap_metadata_bytes(), 0u);
            candidate.finish_inherited_overlap();
        };
        if (reject)
            EXPECT_THROW(edit(), std::invalid_argument);
        else
            EXPECT_NO_THROW(edit());
        EXPECT_EQ(budget->used, baseline);
        EXPECT_EQ(source_budget->used, source_live);
        return budget->peak;
    };
    const auto peak = run(ample, false);
    ASSERT_GT(peak, 0u);
    EXPECT_EQ(run(peak, false), peak);
    (void) run(peak - 1, true);
    source.begin_delta();
    source.remove(tile, std::array{loop[0]});
    PatchOwner original(source_budget, source.finish(lookup));
    ASSERT_EQ(original.value.size(), 1u);
    EXPECT_EQ(original.value.front().row->incidence_generation, 2u);
    ASSERT_EQ(original.value.front().row->points.size(), 1u);
    expect_point(original.value.front().row->points[0].physical, point(5));
}

TEST(PooledContactRows, OwnerRowsReleaseSourceArenasWhenTheirLastSharedPathIsRetired)
{
    // Keep the point/cell registry separate to isolate the row-root arena
    // lifetime from other pool pages and external row handles.
    D::Budget pool_budget{ample};
    D::IdentityBudget identities;
    D::IncidencePools pools(pool_budget, identities);
    Kernel kernel;
    const auto cell = leaf();
    const auto slot = pools.cells.intern(cell.key);
    const auto forbidden = [](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf&
    { throw std::logic_error("Empty row must not look up the leaf."); };
    auto first = std::make_shared<D::Budget>(D::Budget{ample});
    std::weak_ptr<D::Budget> first_weak = first;
    auto source = std::make_unique<D::IncidenceOwnerRows>(*first, identities, pools, kernel, 0, first);
    source->insert(slot, 19);
    EXPECT_TRUE(source->finish(forbidden).empty());

    auto second = std::make_shared<D::Budget>(D::Budget{ample});
    std::weak_ptr<D::Budget> second_weak = second;
    auto candidate = std::make_unique<D::IncidenceOwnerRows>(*source, *second, identities, pools, second);
    source.reset();
    first.reset();
    ASSERT_FALSE(first_weak.expired());
    EXPECT_TRUE(candidate->finish(forbidden).empty());
    candidate->finish_inherited_overlap();

    auto third = std::make_shared<D::Budget>(D::Budget{ample});
    {
        D::IncidenceOwnerRows next(*candidate, *third, identities, pools, third);
        candidate.reset();
        second.reset();
        EXPECT_TRUE(second_weak.expired());
        ASSERT_FALSE(first_weak.expired());
        next.erase(slot);
        EXPECT_TRUE(first_weak.expired());
        EXPECT_GT(next.inherited_overlap_metadata_bytes(), 0u);
        EXPECT_TRUE(next.finish(forbidden).empty());
        next.finish_inherited_overlap();
        EXPECT_EQ(third->used, next.source_overlap_metadata_bytes());
    }
    EXPECT_EQ(third->used, 0u);
}

TEST(PooledContactRows, BorrowedArenaOwnerRowsForkCopiesCountsBeforeSourceLifetimeEnds)
{
    D::Budget budget{ample};
    D::IdentityBudget identities;
    D::IncidencePools pools(budget, identities);
    Kernel kernel;
    const auto cell = leaf();
    D::PooledTile tile;
    tile.first_cell = pools.cells.intern(cell.key);
    const std::array loop{pools.points.intern(point(2)), pools.points.intern(point(5))};
    const auto lookup = [&](const A::AdaptiveCellKey&) -> const A::AdaptiveLeaf& { return cell; };
    std::unique_ptr<D::IncidenceOwnerRows> candidate;
    {
        auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::IdentityBudget source_identities;
        D::IncidencePools source_pools(*source_budget, source_identities);
        ASSERT_EQ(source_pools.cells.intern(cell.key), tile.first_cell);
        ASSERT_EQ(source_pools.points.intern(point(2)), loop[0]);
        ASSERT_EQ(source_pools.points.intern(point(5)), loop[1]);
        D::IncidenceOwnerRows source(*source_budget, source_identities, source_pools, kernel, 0);
        source.insert(tile.first_cell, 7);
        source.add(tile, loop);
        {
            PatchOwner patches(source_budget, source.finish(lookup));
            ASSERT_EQ(patches.value.size(), 1u);
        }
        candidate = std::make_unique<D::IncidenceOwnerRows>(source, budget, identities, pools);
        EXPECT_EQ(candidate->persistence_work().legacy_fork_visits, 1u);
    }
    candidate->finish_inherited_overlap();
    candidate->remove(tile, std::array{loop[0]});
    auto keep = std::shared_ptr<D::Budget>(std::shared_ptr<D::Budget>{}, &budget);
    PatchOwner patches(keep, candidate->finish(lookup));
    ASSERT_EQ(patches.value.size(), 1u);
    EXPECT_EQ(patches.value.front().row->incidence_generation, 9u);
    ASSERT_EQ(patches.value.front().row->points.size(), 1u);
    expect_point(patches.value.front().row->points[0].physical, point(5));
}
} // namespace
