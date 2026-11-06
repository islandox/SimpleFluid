#include "geometry/mesh/AMR/ContactCoverIndex.hh"
#include "geometry/mesh/AMR/ContactPointGroups.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
using Point = std::array<long double, 3>;
constexpr std::size_t ample = 16 * 1024 * 1024;
void retire(std::vector<D::ClosedContactCover>& covers, D::Budget& budget)
{
    std::size_t bytes = covers.capacity() * sizeof(D::ClosedContactCover);
    for (const auto& cover : covers)
        bytes += cover.domain.capacity();
    std::vector<D::ClosedContactCover>().swap(covers);
    budget.used -= bytes;
}
bool contains(const A::AdaptiveContactBounds& box, const Point& point)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (point[axis] < box.lower[axis] || box.upper[axis] < point[axis])
            return false;
    return true;
}
TEST(ContactCoverIndex, EmptyAndMixedDomainCoversRetainEveryLiteralPoint)
{
    D::Budget budget{ample};
    {
        D::ContactCoverIndex index(budget, 7);
        const auto empty_bytes = index.storage_bytes();
        index.reserve(0);
        EXPECT_EQ(index.storage_bytes(), empty_bytes);
        EXPECT_EQ(index.point_count(), 0u);
        EXPECT_FALSE(index.aggregate_bounds());
        auto empty = index.ownership_frontier_covers();
        EXPECT_TRUE(empty.empty());
        retire(empty, budget);
        const A::AdaptiveBytes first{'d'}, second{'d', 0};
        const std::array<Point, 4> points{{{0, 0, 0}, {4, 2, 1}, {-2, 3, 5}, {1, -1, 2}}};
        index.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
            (void) index.insert(&points[i], i < 2 ? &first : &second);
        ASSERT_TRUE(index.aggregate_bounds());
        EXPECT_EQ(index.aggregate_bounds()->lower, (Point{-2, -1, 0}));
        EXPECT_EQ(index.aggregate_bounds()->upper, (Point{4, 3, 5}));
        auto covers = index.ownership_frontier_covers(1);
        std::size_t count = 0;
        for (const auto& cover : covers)
        {
            EXPECT_EQ(cover.owner, 7);
            count += cover.images;
        }
        EXPECT_EQ(count, points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            bool found = false;
            for (const auto& cover : covers)
                found |= cover.domain == (i < 2 ? first : second) && contains(cover.bounds, points[i]);
            EXPECT_TRUE(found);
        }
        retire(covers, budget);
        EXPECT_EQ(budget.used, index.storage_bytes());
    }
    EXPECT_EQ(budget.used, 0u);
}
TEST(ContactCoverIndex, ErasureReuseAndCopiedRebindingDoNotRetainSourceOwners)
{
    D::Budget source_budget{ample}, copy_budget{ample};
    const std::array<Point, 4> copied_points{{{0, 0, 0}, {2, 3, 4}, {-1, 2, 7}, {5, 4, 3}}};
    const A::AdaptiveBytes copied_domain{'x'};
    std::unique_ptr<D::ContactCoverIndex> copy;
    std::array<std::size_t, 4> handles{};
    {
        auto source_points = copied_points;
        A::AdaptiveBytes source_domain = copied_domain;
        D::ContactCoverIndex source(source_budget, 0);
        source.reserve(4);
        for (std::size_t i = 0; i < handles.size(); ++i)
            handles[i] = source.insert(&source_points[i], &source_domain);
        copy = std::make_unique<D::ContactCoverIndex>(source, copy_budget);
        for (std::size_t i = 0; i < handles.size(); ++i)
            copy->rebind(handles[i], &copied_points[i], &copied_domain);
        copy->erase(handles[2]);
        EXPECT_EQ(copy->point_count(), 3u);
        EXPECT_EQ(source.point_count(), 4u);
        EXPECT_EQ(source.aggregate_bounds()->lower[0], -1);
        EXPECT_EQ(copy->aggregate_bounds()->lower[0], 0);
    }
    EXPECT_EQ(source_budget.used, 0u);
    const auto restored = copy->insert(&copied_points[2], &copied_domain);
    EXPECT_EQ(copy->point_count(), 4u);
    EXPECT_EQ(copy->aggregate_bounds()->upper, (Point{5, 4, 7}));
    copy->erase(restored);
    for (std::size_t i : {0u, 1u, 3u})
        copy->erase(handles[i]);
    EXPECT_EQ(copy->point_count(), 0u);
    EXPECT_FALSE(copy->aggregate_bounds());
    copy.reset();
    EXPECT_EQ(copy_budget.used, 0u);
}
TEST(ContactCoverIndex, FailedGrowthPreservesPointSetAndAdmittedCapacity)
{
    D::Budget budget{ample};
    const A::AdaptiveBytes domain{'a'};
    const std::array<Point, 3> points{{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}};
    D::ContactCoverIndex index(budget, 0);
    index.reserve(2);
    (void) index.insert(&points[0], &domain);
    (void) index.insert(&points[1], &domain);
    const auto before = budget.used;
    budget.limit = before;
    EXPECT_THROW((void) index.insert(&points[2], &domain), std::invalid_argument);
    EXPECT_EQ(index.point_count(), 2u);
    EXPECT_EQ(index.aggregate_bounds()->upper, points[1]);
    EXPECT_EQ(budget.used, before);
    budget.limit = ample;
    (void) index.insert(&points[2], &domain);
    EXPECT_EQ(index.point_count(), 3u);
    EXPECT_EQ(index.aggregate_bounds()->upper, points[2]);
}
TEST(ContactCoverIndex, BorrowedMetadataUsesLessStorageThanOwningImageEntries)
{
    D::Budget budget{ample};
    const A::AdaptiveBytes domain{'b'};
    std::array<Point, 64> points{};
    for (std::size_t i = 0; i < points.size(); ++i)
        points[i] = {static_cast<long double>(i), 0, 0};
    D::ContactCoverIndex compact(budget, 0);
    compact.reserve(points.size());
    D::ClosedContactIndex owning(ample);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        (void) compact.insert(&points[i], &domain);
        A::AdaptiveContactImage image;
        image.bounds = {points[i], points[i]};
        owning.insert({{0, A::CartesianCellAncestry{0, {i, 0, 0}}}, domain, 0, 0, {image}});
    }
    const auto old_bytes = owning.statistics().retained_bytes + sizeof(owning);
    EXPECT_LT(compact.storage_bytes(), old_bytes);
    RecordProperty("borrowed_point_index_bytes", std::to_string(compact.storage_bytes()));
    RecordProperty("owning_image_index_bytes", std::to_string(old_bytes));
    auto covers = compact.ownership_frontier_covers(8);
    EXPECT_LE(covers.size(), 8u);
    std::size_t count = 0;
    for (const auto& cover : covers)
        count += cover.images;
    EXPECT_EQ(count, points.size());
    retire(covers, budget);
}
TEST(ContactCoverIndex, FiniteExtremeBoundsRestoreRoundingAndExceptionFlags)
{
    struct Environment
    {
        std::fenv_t saved;
        Environment() { std::fegetenv(&saved); }
        ~Environment() { std::fesetenv(&saved); }
    } environment;
    const auto large = std::numeric_limits<long double>::max() / 2;
    const std::array<Point, 3> points{{{-large, -large, -large}, {large, large, large}, {0, 0, 0}}};
    const A::AdaptiveBytes domain{'f'};
    D::Budget budget{ample};
    D::ContactCoverIndex index(budget, 0);
    std::fesetround(FE_UPWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
    for (const auto& point : points)
        (void) index.insert(&point, &domain);
    auto covers = index.ownership_frontier_covers(2);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    retire(covers, budget);
}
} // namespace


TEST(ContactCoverIndex, LogicalBindingsForkPagesAndResolveOnlyInTheCandidate)
{
    struct Facts
    {
        std::vector<Point> coordinates;
        std::array<A::AdaptiveBytes, 2> domains{{{'a'}, {'b'}}};
        D::ContactCoverIndex::GroupResolver resolver() const
        {
            return {this,
                [](const void* pointer, D::ContactGroupId id) -> D::ContactCoverIndex::GroupObservation
                {
                    const auto& facts = *static_cast<const Facts*>(pointer);
                    if (id.generation != 1)
                        throw std::invalid_argument("Unexpected cover binding generation.");
                    return {facts.coordinates.at(id.slot), &facts.domains[id.slot % 2]};
                }};
        }
    };
    for (const std::size_t count : {256u, 512u})
    {
        SCOPED_TRACE(count);
        Facts candidate_facts;
        for (std::size_t i = 0; i < count; ++i)
            candidate_facts.coordinates.push_back({static_cast<long double>(i), 0, 0});
        auto source_budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto candidate_budget = std::make_shared<D::Budget>(D::Budget{ample});
        std::unique_ptr<D::ContactCoverIndex> candidate;
        {
            const Facts original = candidate_facts;
            D::ContactCoverIndex source(source_budget, 0, original.resolver());
            std::size_t first = 0;
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto handle = source.insert(D::ContactGroupId{i, 1});
                if (!i)
                    first = handle;
            }
            const auto source_used = source_budget->used;
            candidate = std::make_unique<D::ContactCoverIndex>(source, candidate_budget, candidate_facts.resolver());
            EXPECT_EQ(candidate->copied_pages(), 0u);
            candidate->erase(first);
            EXPECT_EQ(candidate->point_count(), count - 1);
            EXPECT_EQ(source.point_count(), count);
            EXPECT_LT(candidate->copied_pages(), 24u);
            EXPECT_EQ(source_budget->used, source_used);
            EXPECT_EQ(source.aggregate_bounds()->lower[0], 0);
            EXPECT_EQ(candidate->aggregate_bounds()->lower[0], 1);
            candidate->finish_overlap();
        }
        auto covers = candidate->ownership_frontier_covers(8);
        std::size_t total = 0;
        for (const auto& cover : covers)
            total += cover.images;
        EXPECT_EQ(total, count - 1);
        EXPECT_EQ(candidate->aggregate_bounds()->upper[0], count - 1);
        retire(covers, *candidate_budget);
        EXPECT_EQ(candidate_budget->used, candidate->storage_bytes());
        candidate.reset();
        EXPECT_EQ(candidate_budget->used, 0u);
        EXPECT_EQ(source_budget->used, 0u);
    }
}
