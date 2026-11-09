/** Persistent value lookup shares data owners across long append histories. */
#include "geometry/mesh/AMR/ContactGroupAllocation.hh"
#include "geometry/mesh/AMR/ContactPointValueIndex.hh"
#include "geometry/mesh/AMR/ContactSourcePoints.hh"
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <variant>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Index = D::ContactPointValueIndex;
using Point = Index::Point;
using Component = Index::Component;
using Access = SimpleFluid::PLIC::details::GeometryIdentityAccess;
constexpr std::size_t ample = 64 * 1024 * 1024, width = 8;

struct Block
{
    D::ContactGroupAllocation memory;
    std::size_t first, z_first;
    std::vector<Point> points;
    std::vector<std::array<std::size_t, 3>> selectors;
    std::array<std::vector<Component>, 3> components;
    Block(std::shared_ptr<D::Budget> budget, std::size_t slot)
        : memory(std::move(budget)), first(slot), z_first(slot / 4)
    {
        memory.add(sizeof(Block) + 128);
        memory.reserve(points, width);
        memory.reserve(selectors, width);
        for (unsigned axis = 0; axis < 3; ++axis)
            if (axis == 2 || first == 0)
                memory.reserve(components[axis], 2);
        for (unsigned z = 0; z < 2; ++z)
            for (unsigned y = 0; y < 2; ++y)
                for (unsigned x = 0; x < 2; ++x)
                {
                    points.push_back(Access::point({static_cast<long double>(x), static_cast<long double>(y),
                        static_cast<long double>(z_first + z)}));
                    selectors.push_back({x, y, z_first + z});
                }
        if (first == 0)
        {
            components[0].push_back(Access::point_component(points[0], 0));
            components[0].push_back(Access::point_component(points[1], 0));
            components[1].push_back(Access::point_component(points[0], 1));
            components[1].push_back(Access::point_component(points[2], 1));
        }
        components[2].push_back(Access::point_component(points[0], 2));
        components[2].push_back(Access::point_component(points[4], 2));
    }
    std::size_t first_component(unsigned axis) const noexcept
    {
        return axis == 2 ? z_first : first == 0 ? 0u : 2u;
    }
    const Component* component_at(unsigned axis, std::size_t slot) const noexcept
    {
        const auto first_slot = first_component(axis);
        const auto& values = components[axis];
        return slot >= first_slot && slot - first_slot < values.size() ? &values[slot - first_slot] : nullptr;
    }
};
using Body = std::shared_ptr<const Block>;
struct BlockLess
{
    bool operator()(const Body& a, const Body& b) const noexcept { return a->first < b->first; }
};
using Directory = D::persistent_ordered_detail::OrderedPages<Body, BlockLess>;
struct Values
{
    std::shared_ptr<D::Budget> budget;
    Directory::Ptr blocks;
    Index index;
    Index::Work work;
    std::size_t count = 0;

    Index::Reader reader() const
    {
        return {this, [](const void* self, std::size_t slot, Index::Probe* work)
            { return static_cast<const Values*>(self)->point(slot, work); },
            [](const void* self, unsigned axis, std::size_t slot, Index::Probe* work) -> const Component&
            { return static_cast<const Values*>(self)->component(axis, slot, work); }};
    }
    const Block& block_at(std::size_t point, Index::Probe* work = nullptr) const
    {
        const auto* found = Directory::lower_bound(blocks,
            [&](const Body& block)
            {
                if (work)
                    ++work->directory_comparisons;
                return block->first + width <= point;
            });
        if (!found)
            throw std::out_of_range("Absent point test ordinal.");
        return **found;
    }
    Point point(std::size_t slot, Index::Probe* work = nullptr) const
    {
        const auto& block = block_at(slot, work);
        const auto& selected = block.selectors[slot - block.first];
        const auto select = [&](unsigned axis) -> const Component&
        {
            if (const auto* local = block.component_at(axis, selected[axis]))
                return *local;
            return component(axis, selected[axis], work);
        };
        return Access::point_components({select(0), select(1), select(2)});
    }
    const Component& component(unsigned axis, std::size_t slot, Index::Probe* work = nullptr) const
    {
        const auto* found = Directory::lower_bound(blocks,
            [&](const Body& block)
            {
                if (work)
                    ++work->directory_comparisons;
                return block->first_component(axis) + block->components[axis].size() <= slot;
            });
        const auto* value = found ? (*found)->component_at(axis, slot) : nullptr;
        if (!value)
            throw std::out_of_range("Absent component test ordinal.");
        return *value;
    }
    std::shared_ptr<Values> append(Index::Work* observed = nullptr) const
    {
        auto next = std::make_shared<Values>(*this);
        next->work = {};
        struct Report
        {
            Index::Work& work;
            Index::Work* output;
            ~Report()
            {
                if (output)
                    *output = work;
            }
        } report{next->work, observed};
        const Body body = std::make_shared<Block>(budget, count);
        next->blocks = Directory(budget).edit(blocks, body, &body, false);
        next->count += width;
        std::vector<std::uint64_t> routes;
        D::cartesian_detail::Allocation scratch(*budget);
        if (routing != Index::Routing::Ordered)
        {
            scratch.reserve(routes, width);
            for (std::size_t i = count; i < next->count; ++i)
                routes.push_back(
                    routing == Index::Routing::Canonical ? D::IdentityPoolHasher<Point>{}(next->point(i)) : 7);
        }
        const std::array<std::size_t, 3> first{count ? 2u : 0u, count ? 2u : 0u, count / 4};
        const std::array<std::size_t, 3> end{2, 2, next->count / 4};
        next->index.append(next->reader(), first, end, count, next->count, routes, budget, next->work);
        return next;
    }
    Index::Routing routing = Index::Routing::Canonical;
};

std::shared_ptr<Values> initial(Index::Routing mode = Index::Routing::Canonical)
{
    auto values = std::make_shared<Values>();
    values->budget = std::make_shared<D::Budget>(D::Budget{ample});
    values->routing = mode;
    values->index = Index(mode);
    return values;
}
std::uint64_t route(const Values& values, const Point& point)
{
    return values.routing == Index::Routing::Canonical ? D::IdentityPoolHasher<Point>{}(point)
           : values.routing == Index::Routing::Custom  ? 7
                                                       : 0;
}
using Census = std::map<std::pair<const void*, D::ContactPayloadRole>, std::size_t>;
Census census(const Index& index)
{
    Census result;
    index.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& item)
        {
            const auto [at, added] =
                result.emplace(std::pair{item.identity.address, item.identity.role}, item.payload_bytes);
            if (!added)
                EXPECT_EQ(at->second, item.payload_bytes);
        });
    return result;
}
Census census(const Values& values)
{
    return census(values.index);
}
std::shared_ptr<Values> history(std::size_t blocks, Index::Routing mode = Index::Routing::Canonical)
{
    auto values = initial(mode);
    for (std::size_t step = 0; step < blocks; ++step)
        values = values->append();
    return values;
}
} // namespace

TEST(ContactPointValueIndex, LongHistoryUsesGlobalProbesAndSharesMetadataWithoutSnapshotPins)
{
    struct Work
    {
        std::size_t point_tests = 0, scalar_tests = 0, directory_tests = 0, pages = 0, bytes = 0;
    };
    const auto run = [](std::size_t blocks, Work& work)
    {
        auto old = history(blocks);
        auto current = old->append();
        auto third = current->append();
        const auto before = census(*old), after = census(*third);
        std::size_t shared = 0;
        for (const auto& [key, bytes] : before)
            if (const auto found = after.find(key); found != after.end())
            {
                EXPECT_EQ(found->second, bytes);
                ++shared;
            }
        EXPECT_GT(shared, before.size() / 2);
        Index::Probe probe;
        for (const auto slot : {std::size_t(0), old->count - 1, current->count - 1, third->count - 1})
        {
            const auto point = third->point(slot);
            EXPECT_EQ(point, third->block_at(slot).points[slot % width]);
            EXPECT_EQ(third->index.find(third->reader(), point, route(*third, point), &probe), slot);
        }
        const auto missing = Access::point({2, 2, static_cast<long double>(third->count)});
        EXPECT_FALSE(third->index.find(third->reader(), missing, route(*third, missing), &probe));
        EXPECT_EQ(third->index.find_component(third->reader(), 0, -0.0L, &probe), 0u);
        EXPECT_EQ(third->index.find_component(third->reader(), 2, 0, &probe), 0u);
        EXPECT_EQ(third->index.find_component(third->reader(), 2, third->count / 4 - 1, &probe), third->count / 4 - 1);
        EXPECT_FALSE(third->index.find_component(third->reader(), 2, third->count, &probe));

        // The former representation retained one 16-slot uint32 hash table and
        // one vector variant per eight-point block. All backing bodies and the
        // block directory are shared independently by both representations.
        using Lookup = std::variant<std::vector<std::uint32_t>, std::vector<std::uint64_t>>;
        auto legacy_budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::cartesian_detail::Allocation legacy_memory(*legacy_budget);
        std::vector<Lookup> previous;
        legacy_memory.reserve(previous, blocks + 2);
        previous.resize(blocks + 2);
        for (std::size_t block = 0; block < previous.size(); ++block)
        {
            auto& table = previous[block].template emplace<std::vector<std::uint32_t>>();
            legacy_memory.reserve(table, 16);
            table.resize(16, std::numeric_limits<std::uint32_t>::max());
            for (std::size_t i = 0; i < width; ++i)
            {
                const auto ordinal = block * width + i;
                auto bucket = std::size_t(route(*third, third->point(ordinal))) & 15;
                while (table[bucket] != std::numeric_limits<std::uint32_t>::max())
                    bucket = (bucket + 1) & 15;
                table[bucket] = std::uint32_t(ordinal);
            }
        }
        const auto previous_lookup_bytes = legacy_memory.bytes;
        std::map<const void*, std::size_t> admitted;
        const auto collect = [&](const Values& values)
        {
            values.index.visit_dependencies<false>(
                [&](const auto& owner, const D::Budget*, std::size_t bytes, bool)
                {
                    const auto [at, added] = admitted.emplace(owner.get(), bytes);
                    if (!added)
                        EXPECT_EQ(at->second, bytes);
                });
        };
        collect(*old);
        collect(*current);
        collect(*third);
        // Charge all three complete new snapshot headers. This conservative
        // gate includes Work, index_build_bytes, and padding; it is stricter
        // than comparing only the header-size delta against the former layout.
        std::size_t index_overlap_bytes = 3 * sizeof(D::ContactSourcePointSnapshot);
        for (const auto& [owner, bytes] : admitted)
            index_overlap_bytes += bytes;
        EXPECT_LE(index_overlap_bytes, previous_lookup_bytes);
        std::size_t actual = 0;
        for (const auto& [key, bytes] : after)
            actual += bytes;
        EXPECT_LE(actual, third->index.metadata_bytes());
        EXPECT_GT(actual, third->index.metadata_bytes() / 2);

        const std::weak_ptr<Values> old_controller = old, current_controller = current;
        const std::weak_ptr<const Block> first_body =
            *Directory::lower_bound(old->blocks, [](const Body&) { return false; });
        old.reset();
        current.reset();
        EXPECT_TRUE(old_controller.expired());
        EXPECT_TRUE(current_controller.expired());
        EXPECT_FALSE(first_body.expired());
        EXPECT_EQ(third->index.find(third->reader(), third->point(0), route(*third, third->point(0))), 0u);
        work = {probe.point_comparisons, probe.scalar_comparisons, probe.directory_comparisons,
            third->work.pages.copied_pages, third->work.pages.copied_bytes};
        const auto budget = third->budget;
        third.reset();
        EXPECT_TRUE(first_body.expired());
        EXPECT_EQ(budget->used, 0u);
    };
    Work smaller, larger;
    ASSERT_NO_FATAL_FAILURE(run(1536, smaller));
    ASSERT_NO_FATAL_FAILURE(run(3072, larger));
    EXPECT_LT(larger.point_tests, 128u);
    EXPECT_LT(larger.scalar_tests, 128u);
    EXPECT_LT(larger.directory_tests, 4096u);
    EXPECT_LE(larger.point_tests, 2 * smaller.point_tests);
    EXPECT_LE(larger.scalar_tests, 2 * smaller.scalar_tests);
    EXPECT_LE(larger.directory_tests, 2 * smaller.directory_tests);
    EXPECT_LE(larger.pages, 2 * smaller.pages);
    EXPECT_LE(larger.bytes, 2 * smaller.bytes);
}

TEST(ContactPointValueIndex, OrderedAndConstantHashRoutesPreserveCompleteIdentityAndOrdinals)
{
    for (const auto mode : {Index::Routing::Ordered, Index::Routing::Custom})
    {
        const auto values = history(96, mode);
        for (std::size_t slot = 0; slot < values->count; ++slot)
        {
            const auto point = values->point(slot);
            Index::Probe work;
            EXPECT_EQ(values->index.find(values->reader(), point, route(*values, point), &work), slot);
            EXPECT_LE(work.identity_checks, 1u);
            EXPECT_LT(work.point_comparisons, 32u);
        }
        const auto missing = Access::point({0, 0, static_cast<long double>(values->count)});
        EXPECT_FALSE(values->index.find(values->reader(), missing, route(*values, missing)));
        if (mode == Index::Routing::Custom)
            EXPECT_FALSE(values->index.find(values->reader(), values->point(0), 8));
    }
}

TEST(ContactPointValueIndex, PartialIndexAdmissionFailurePreservesAcceptedRootsAndRecoversAtExactPeak)
{
    auto accepted = history(192);
    const auto before = census(*accepted);
    const auto budget = accepted->budget;
    const auto live = budget->used;
    budget->peak = live;
    {
        const auto candidate = accepted->append();
        EXPECT_GT(candidate->work.pages.copied_pages, 3u);
    }
    const auto required = budget->peak - live;
    ASSERT_GT(required, 1u);
    EXPECT_EQ(budget->used, live);
    budget->limit = live + required - 1;
    Index::Work failed;
    EXPECT_THROW((void) accepted->append(&failed), std::invalid_argument);
    EXPECT_GT(failed.pages.copied_pages, 0u);
    EXPECT_GT(failed.probes.scalar_comparisons, 0u);
    EXPECT_EQ(budget->used, live);
    EXPECT_EQ(census(*accepted), before);
    EXPECT_EQ(accepted->index.find(accepted->reader(), accepted->point(0), route(*accepted, accepted->point(0))), 0u);
    budget->limit = live + required;
    budget->peak = live;
    {
        const auto candidate = accepted->append();
        EXPECT_EQ(budget->peak, budget->limit);
        EXPECT_EQ(candidate->index.find(candidate->reader(), candidate->point(candidate->count - 1),
                      route(*candidate, candidate->point(candidate->count - 1))),
            candidate->count - 1);
    }
    EXPECT_EQ(budget->used, live);
    budget->limit = ample;
    accepted.reset();
    EXPECT_EQ(budget->used, 0u);
}

namespace
{
constexpr std::size_t first_sparse_ordinal = std::numeric_limits<std::uint32_t>::max() - std::size_t{1};
struct SparseValues
{
    std::array<Point, 2> points;
    std::array<Component, 2> components;
    explicit SparseValues(bool duplicate)
        : points{Access::point({0, 0, 0}), Access::point({duplicate ? -0.L : 1.L, 0, 0})},
          components{Access::point_component(points[0], 0), Access::point_component(points[1], 0)}
    {
    }
    Index::Reader reader() const
    {
        return {this, [](const void* context, std::size_t slot, Index::Probe*)
            { return static_cast<const SparseValues*>(context)->points.at(slot - first_sparse_ordinal); },
            [](const void* context, unsigned, std::size_t slot, Index::Probe*) -> const Component&
            { return static_cast<const SparseValues*>(context)->components.at(slot - first_sparse_ordinal); }};
    }
};
} // namespace

TEST(ContactPointValueIndex, WidthPartitionsRejectDuplicatePointsAndKeepDistinctLargeOrdinals)
{
    constexpr std::size_t first = std::numeric_limits<std::uint32_t>::max() - std::size_t{1};
    for (const auto mode : {Index::Routing::Ordered, Index::Routing::Canonical})
        for (const bool duplicate : {false, true})
        {
            SparseValues values(duplicate);
            auto budget = std::make_shared<D::Budget>(D::Budget{ample});
            {
                Index index(mode);
                Index::Work work;
                const std::array<std::size_t, 3> empty{};
                std::vector<std::uint64_t> hashes;
                if (mode == Index::Routing::Canonical)
                    hashes.push_back(D::IdentityPoolHasher<Point>{}(values.points[0]));
                index.append(values.reader(), empty, empty, first, first + 1, hashes, budget, work);
                const auto before = census(index);
                const auto live = budget->used;
                hashes.clear();
                if (mode == Index::Routing::Canonical)
                    hashes.push_back(D::IdentityPoolHasher<Point>{}(values.points[1]));
                const auto append = [&]
                { index.append(values.reader(), empty, empty, first + 1, first + 2, hashes, budget, work); };
                if (duplicate)
                {
                    EXPECT_THROW(append(), std::invalid_argument);
                    EXPECT_EQ(budget->used, live);
                    EXPECT_EQ(census(index), before);
                }
                else
                    EXPECT_NO_THROW(append());
                const auto hash =
                    mode == Index::Routing::Canonical ? D::IdentityPoolHasher<Point>{}(values.points[1]) : 0;
                EXPECT_EQ(index.find(values.reader(), values.points[1], hash), duplicate ? first : first + 1);
            }
            EXPECT_EQ(budget->used, 0u);
            {
                Index index(mode);
                Index::Work work;
                std::vector<std::uint64_t> hashes;
                if (mode == Index::Routing::Canonical)
                    for (const auto& point : values.points)
                        hashes.push_back(D::IdentityPoolHasher<Point>{}(point));
                const auto append = [&]
                { index.append(values.reader(), {}, {}, first, first + 2, hashes, budget, work); };
                if (duplicate)
                {
                    EXPECT_THROW(append(), std::invalid_argument);
                    EXPECT_EQ(budget->used, 0u);
                    EXPECT_TRUE(census(index).empty());
                }
                else
                    EXPECT_NO_THROW(append());
            }
            EXPECT_EQ(budget->used, 0u);
        }
}

TEST(ContactPointValueIndex, WidthPartitionsRejectDuplicateScalarsAndKeepFirstIssuedOrdinal)
{
    constexpr std::size_t first = std::numeric_limits<std::uint32_t>::max() - std::size_t{1};
    for (const auto mode : {Index::Routing::Ordered, Index::Routing::Canonical})
        for (const bool duplicate : {false, true})
        {
            SparseValues values(duplicate);
            auto budget = std::make_shared<D::Budget>(D::Budget{ample});
            {
                Index index(mode);
                Index::Work work;
                index.append(values.reader(), {first, 0, 0}, {first + 1, 0, 0}, 0, 0, {}, budget, work);
                const auto before = census(index);
                const auto live = budget->used;
                const auto append = [&]
                { index.append(values.reader(), {first + 1, 0, 0}, {first + 2, 0, 0}, 0, 0, {}, budget, work); };
                if (duplicate)
                {
                    EXPECT_THROW(append(), std::invalid_argument);
                    EXPECT_EQ(budget->used, live);
                    EXPECT_EQ(census(index), before);
                }
                else
                    EXPECT_NO_THROW(append());
                EXPECT_EQ(index.find_component(values.reader(), 0, values.components[1].value()),
                    duplicate ? first : first + 1);
            }
            EXPECT_EQ(budget->used, 0u);
            {
                Index index(mode);
                Index::Work work;
                const auto append = [&]
                { index.append(values.reader(), {first, 0, 0}, {first + 2, 0, 0}, 0, 0, {}, budget, work); };
                if (duplicate)
                {
                    EXPECT_THROW(append(), std::invalid_argument);
                    EXPECT_EQ(budget->used, 0u);
                    EXPECT_TRUE(census(index).empty());
                }
                else
                    EXPECT_NO_THROW(append());
            }
            EXPECT_EQ(budget->used, 0u);
        }
}
