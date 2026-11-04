/** Literal point-set, transaction-lifetime and sparse-delta closure oracles. */
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Point = P::PointIdentity<ordinary>;
using Wide = std::array<long double, 3>;
using PointPairs = std::vector<std::pair<Point, Point>>; // normalized, physical
using Marks = std::vector<std::pair<AMR::AdaptiveCellKey, AMR::AdaptiveSplit>>;
using Calls = std::array<std::uint64_t, 4>; // geometry, surfaces, traces, normalization
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

AMR::AdaptiveCellKey cell_key(std::uint64_t id)
{
    return {0, AMR::ConvexCellAncestry{id, {}}};
}
AMR::AdaptiveLeaf box(std::uint64_t id, P::GeometryPoint lower, P::GeometryPoint upper)
{
    AMR::AdaptiveLeaf leaf;
    leaf.key = cell_key(id);
    leaf.domain = {'d'};
    leaf.revision = 17;
    leaf.geometry = P::ConvexCellGeometry<ordinary>::box(lower, upper);
    for (std::size_t face = 0; face < leaf.geometry->num_faces(); ++face)
    {
        auto support = AMR::encode_adaptive_key(leaf.key);
        support.push_back(char(face));
        leaf.surfaces.push_back({std::move(support), leaf.geometry->face_geometry(face), 1, true, "wall"});
    }
    return leaf;
}
int other_owner()
{
    return std::min(1, Tpetra::getDefaultComm()->getSize() - 1);
}
std::vector<AMR::AdaptiveLeaf> local(std::initializer_list<std::pair<int, AMR::AdaptiveLeaf>> entries)
{
    std::vector<AMR::AdaptiveLeaf> result;
    for (const auto& [owner, leaf] : entries)
        if (owner == Tpetra::getDefaultComm()->getRank())
            result.push_back(leaf);
    return result;
}
std::vector<AMR::AdaptiveCellKey> inserted_on(int owner, const AMR::AdaptiveCellKey& key)
{
    return Tpetra::getDefaultComm()->getRank() == owner ? std::vector{key} : std::vector<AMR::AdaptiveCellKey>{};
}
std::vector<D::ContactRetiredLeaf> retired_on(int owner, const AMR::AdaptiveCellKey& key, std::uint64_t revision = 17)
{
    return Tpetra::getDefaultComm()->getRank() == owner ? std::vector{D::ContactRetiredLeaf{key, revision}}
                                                        : std::vector<D::ContactRetiredLeaf>{};
}
std::vector<AMR::AdaptiveBytes> encoded(std::span<const AMR::AdaptiveLeaf> leaves)
{
    std::vector<AMR::AdaptiveBytes> result;
    for (const auto& leaf : leaves)
        result.push_back(AMR::encode_adaptive_leaf(leaf));
    return result;
}
class FixtureKernel final : public AMR::AdaptiveProviderKernel
{
    bool periodic_;
    mutable std::map<AMR::AdaptiveCellKey, Calls> calls_;

public:
    std::optional<AMR::AdaptiveCellKey> reject_record;
    explicit FixtureKernel(bool periodic = false) : periodic_(periodic) {}
    AMR::AdaptiveBytes descriptor() const override { return {'i', 'n', 'c', char(periodic_)}; }
    AMR::AdaptiveContactCapabilities contact_capabilities(const AMR::AdaptiveLeaf&) const override
    {
        return {periodic_ ? AMR::AdaptiveContactNormalization::PeriodicCartesian
                          : AMR::AdaptiveContactNormalization::PhysicalIdentity,
            false, false};
    }
    P::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf& leaf) const override
    {
        ++calls_[leaf.key][0];
        return leaf.geometry;
    }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override
    {
        ++calls_[leaf.key][1];
        return leaf.surfaces;
    }
    AMR::AdaptiveTrace traces(const AMR::AdaptiveLeaf& leaf) const override
    {
        ++calls_[leaf.key][2];
        if (reject_record == leaf.key)
            throw std::invalid_argument("Injected contact record failure.");
        return {leaf.key, {}};
    }
    Point contact_point(const AMR::AdaptiveLeaf& leaf, const Point& point) const override
    {
        ++calls_[leaf.key][3];
        return periodic_ ? P::periodic_point<ordinary>(point, {0, 0, 0}, {12, 12, 12}, {true, false, false}) : point;
    }
    std::vector<AMR::AdaptiveContactImage> closed_contact_images(const AMR::AdaptiveLeaf& leaf) const override
    {
        const auto bounds = leaf.geometry->closed_contact_bounds();
        AMR::AdaptiveContactImage image;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            image.bounds.lower[axis] = bounds[axis][0];
            image.bounds.upper[axis] = bounds[axis][1];
        }
        std::vector result{image};
        if (periodic_ && image.bounds.upper[0] == 12)
        {
            image.bounds.lower[0] = image.bounds.upper[0] = 0;
            image.physical_to_contact = P::displacement<ordinary>({12, 0, 0}, {0, 0, 0});
            result.push_back(std::move(image));
        }
        return result;
    }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf&, AMR::AdaptiveSplit) const override
    {
        throw std::invalid_argument("Literal incidence fixture does not split geometry.");
    }
    Calls calls(const AMR::AdaptiveCellKey& key) const
    {
        const auto found = calls_.find(key);
        return found == calls_.end() ? Calls{} : found->second;
    }
    Calls calls() const
    {
        Calls result{};
        for (const auto& [key, value] : calls_)
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] += value[i];
        return result;
    }
};
D::ContactOwnedLeafLookup lookup(const std::vector<AMR::AdaptiveLeaf>& leaves, std::uint64_t* calls = nullptr)
{
    return [&leaves, calls](const AMR::AdaptiveCellKey& key) -> const AMR::AdaptiveLeaf&
    {
        if (calls)
            ++*calls;
        const auto found = std::ranges::find(leaves, key, &AMR::AdaptiveLeaf::key);
        if (found == leaves.end())
            throw std::invalid_argument("Fixture lookup has no current owner.");
        return *found;
    };
}
struct Row
{
    std::uint64_t revision, generation;
    AMR::AdaptiveBytes domain;
    PointPairs points;
    const D::ContactIncidenceRow* address;
    bool operator==(const Row&) const = default;
};
using Rows = std::map<AMR::AdaptiveCellKey, Row>;
Rows rows(const D::DistributedAdaptiveContactHalo& halo)
{
    Rows result;
    halo.for_each_incidence_row(
        [&](const auto& row)
        {
            PointPairs points;
            for (const auto& point : row.points)
                points.emplace_back(point.normalized, point.physical);
            std::sort(points.begin(), points.end());
            EXPECT_TRUE(result
                    .emplace(row.key, Row{row.revision, row.incidence_generation, row.domain, std::move(points), &row})
                    .second);
        });
    return result;
}
PointPairs points(std::vector<Wide> coordinates, bool periodic = false)
{
    PointPairs result;
    for (auto physical : coordinates)
    {
        auto normalized = physical;
        if (periodic && normalized[0] == 12)
            normalized[0] = 0;
        // Literal integer coordinates establish the expected identity. Neither
        // the provider normalization nor an incidence algorithm derives it.
        result.emplace_back(
            P::details::GeometryIdentityAccess::point(normalized), P::details::GeometryIdentityAccess::point(physical));
    }
    std::sort(result.begin(), result.end());
    return result;
}
std::vector<Wide> corners(Wide lower, Wide upper)
{
    std::vector<Wide> result;
    for (unsigned mask = 0; mask < 8; ++mask)
        result.push_back(
            {mask & 1 ? upper[0] : lower[0], mask & 2 ? upper[1] : lower[1], mask & 4 ? upper[2] : lower[2]});
    return result;
}
void expect_row(const Rows& observed, int owner, const AMR::AdaptiveCellKey& key, const PointPairs& expected)
{
    const bool owned = Tpetra::getDefaultComm()->getRank() == owner;
    EXPECT_EQ(observed.contains(key), owned);
    if (owned && observed.contains(key))
    {
        EXPECT_EQ(observed.at(key).revision, 17u);
        EXPECT_EQ(observed.at(key).domain, (AMR::AdaptiveBytes{'d'}));
        EXPECT_EQ(observed.at(key).points, expected);
    }
}
Marks marks(const std::vector<AMR::AdaptiveMark>& values)
{
    Marks result;
    for (const auto& value : values)
        result.emplace_back(value.key, value.split);
    std::sort(result.begin(), result.end());
    return result;
}
Marks expected_marks(std::initializer_list<std::pair<int, AMR::AdaptiveCellKey>> owners)
{
    Marks result;
    for (const auto& [owner, key] : owners)
        if (owner == Tpetra::getDefaultComm()->getRank())
            result.emplace_back(key, AMR::AdaptiveSplit::Isotropic);
    std::sort(result.begin(), result.end());
    return result;
}
template<std::size_t N> std::array<std::uint64_t, N> global_sum(const std::array<std::uint64_t, N>& local_values)
{
    std::array<std::uint64_t, N> result{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, int(N), local_values.data(), result.data());
    return result;
}
template<class Operation> void rejects_collectively(Operation&& operation, std::string_view expected_reason)
{
    int rejected = 0, matching_reason = 0;
    try
    {
        operation();
    }
    catch (const std::invalid_argument& error)
    {
        rejected = 1;
        matching_reason = std::string_view(error.what()).find(expected_reason) != std::string_view::npos;
    }
    catch (const std::exception& error)
    {
        ADD_FAILURE() << "Unexpected rejection type: " << error.what();
    }
    int all_rejected = 0, any_matching = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &matching_reason, &any_matching);
    EXPECT_EQ(all_rejected, 1);
    EXPECT_EQ(any_matching, 1) << expected_reason;
}
D::ContactHaloContext context(std::uint64_t transaction = 101, std::uint64_t round = 0)
{
    return {7, 11, 13, round, transaction};
}

TEST(IncrementalAdaptiveClosure, EdgeInsertionDeletionPreserveRevisionRecordsAndForkIsolation)
{
    const auto comm = Tpetra::getDefaultComm();
    FixtureKernel kernel;
    const auto coarse = box(1, {0, 0, 0}, {4, 4, 4});
    const auto distant = box(2, {20, 20, 20}, {24, 24, 24});
    const auto fine = box(3, {4, 4, 1}, {5, 5, 2});
    auto original = local({{0, coarse}, {0, distant}});
    const auto original_bytes = encoded(original);
    auto source =
        std::make_unique<D::DistributedAdaptiveContactHalo>(kernel, comm, AMR::AdaptiveClosureOptions{}, context());
    source->bootstrap_canonical(original, lookup(original));
    EXPECT_TRUE(source->required_refinements().empty());
    const auto before = rows(*source);
    expect_row(before, 0, coarse.key, points(corners({0, 0, 0}, {4, 4, 4})));
    expect_row(before, 0, distant.key, points(corners({20, 20, 20}, {24, 24, 24})));
    EXPECT_EQ(before.size(), comm->getRank() == 0 ? 2u : 0u);
    const auto initial_empty = global_sum(std::array<std::uint64_t, 1>{original.empty()});
    EXPECT_EQ(initial_empty[0], std::uint64_t(comm->getSize() - 1));

    auto sibling = source->fork_candidate(context(202));
    auto candidate = source->fork_candidate(context(203));
    EXPECT_EQ(rows(*sibling), before);
    EXPECT_EQ(rows(*candidate), before);
    const auto coarse_calls = kernel.calls(coarse.key), distant_calls = kernel.calls(distant.key);
    const auto after_insert = local({{0, coarse}, {0, distant}, {other_owner(), fine}});
    candidate->apply_delta({}, inserted_on(other_owner(), fine.key), context(203, 1), lookup(after_insert));
    EXPECT_EQ(marks(candidate->required_refinements()), expected_marks({{0, coarse.key}}));
    auto enriched = corners({0, 0, 0}, {4, 4, 4});
    enriched.insert(enriched.end(), {{4, 4, 1}, {4, 4, 2}});
    const auto inserted = rows(*candidate);
    expect_row(inserted, 0, coarse.key, points(enriched));
    expect_row(inserted, other_owner(), fine.key, points(corners({4, 4, 1}, {5, 5, 2})));
    if (comm->getRank() == 0 && inserted.contains(coarse.key) && before.contains(coarse.key))
    {
        EXPECT_EQ(inserted.at(coarse.key).generation, before.at(coarse.key).generation + 1);
        EXPECT_NE(inserted.at(coarse.key).address, before.at(coarse.key).address);
        EXPECT_EQ(inserted.at(distant.key), before.at(distant.key));
    }
    EXPECT_EQ(kernel.calls(coarse.key)[0], coarse_calls[0]);
    EXPECT_EQ(kernel.calls(coarse.key)[2], coarse_calls[2]);
    EXPECT_EQ(kernel.calls(distant.key), distant_calls);
    EXPECT_EQ(rows(*source), before);
    EXPECT_EQ(rows(*sibling), before);
    EXPECT_TRUE(source->required_refinements().empty());
    EXPECT_TRUE(sibling->required_refinements().empty());
    EXPECT_EQ(encoded(original), original_bytes);

    // The accepted candidate must own every shared row/layer allocation needed
    // for the inverse delta after both its source and sibling have died.
    source.reset();
    sibling.reset();
    original.clear();
    const auto after_delete = local({{0, coarse}, {0, distant}});
    candidate->apply_delta(retired_on(other_owner(), fine.key), {}, context(203, 2), lookup(after_delete));
    EXPECT_TRUE(candidate->required_refinements().empty());
    const auto deleted = rows(*candidate);
    expect_row(deleted, 0, coarse.key, points(corners({0, 0, 0}, {4, 4, 4})));
    EXPECT_FALSE(deleted.contains(fine.key));
    if (comm->getRank() == 0 && deleted.contains(coarse.key) && inserted.contains(coarse.key))
    {
        EXPECT_EQ(deleted.at(coarse.key).generation, inserted.at(coarse.key).generation + 1);
        EXPECT_EQ(deleted.at(distant.key), before.at(distant.key));
    }
    EXPECT_EQ(kernel.calls(coarse.key)[0], coarse_calls[0]);
    EXPECT_EQ(kernel.calls(coarse.key)[2], coarse_calls[2]);
    EXPECT_EQ(kernel.calls(distant.key), distant_calls);
    EXPECT_EQ(encoded(after_delete), original_bytes);
    EXPECT_EQ(candidate->statistics().full_owned_scans_after_bootstrap, 0u);
}

TEST(IncrementalAdaptiveClosure, RankLocalEarlyAndLateRejectionLeaveAcceptedStateReusable)
{
    const auto comm = Tpetra::getDefaultComm();
    FixtureKernel kernel;
    const auto coarse = box(1, {0, 0, 0}, {4, 4, 4});
    const auto distant = box(2, {20, 20, 20}, {24, 24, 24});
    const auto fine = box(3, {4, 4, 1}, {5, 5, 2});
    const auto original = local({{0, coarse}, {0, distant}});
    const auto prospective = local({{0, coarse}, {0, distant}, {other_owner(), fine}});
    const auto original_bytes = encoded(original), prospective_bytes = encoded(prospective);
    D::DistributedAdaptiveContactHalo accepted(kernel, comm, {}, context());
    accepted.bootstrap_canonical(original, lookup(original));
    EXPECT_TRUE(accepted.required_refinements().empty());
    const auto before = rows(accepted);
    const auto retained = accepted.statistics().retained_bytes;
    for (unsigned defect = 0; defect < 2; ++defect)
    {
        SCOPED_TRACE(defect);
        auto failed = accepted.fork_candidate(context(210 + defect));
        const auto retired = defect == 0 ? retired_on(0, coarse.key, 18) : std::vector<D::ContactRetiredLeaf>{};
        if (defect == 1)
            kernel.reject_record = fine.key;
        rejects_collectively(
            [&]
            {
                failed->apply_delta(
                    retired, inserted_on(other_owner(), fine.key), context(210 + defect, 1), lookup(prospective));
            },
            defect == 0 ? "stale" : "Injected contact record failure");
        kernel.reject_record.reset();
        failed.reset();
        EXPECT_EQ(rows(accepted), before);
        EXPECT_TRUE(accepted.required_refinements().empty());
        EXPECT_EQ(accepted.statistics().retained_bytes, retained);
        EXPECT_EQ(encoded(original), original_bytes);
        EXPECT_EQ(encoded(prospective), prospective_bytes);
    }
    auto retry = accepted.fork_candidate(context(220));
    retry->apply_delta({}, inserted_on(other_owner(), fine.key), context(220, 1), lookup(prospective));
    EXPECT_EQ(marks(retry->required_refinements()), expected_marks({{0, coarse.key}}));
    EXPECT_EQ(rows(accepted), before);
    EXPECT_TRUE(accepted.required_refinements().empty());
}

TEST(IncrementalAdaptiveClosure, SameKeyRevisionMigrationChangesOwnerWithEmptyRanks)
{
    const auto comm = Tpetra::getDefaultComm();
    FixtureKernel kernel;
    const auto coarse = box(1, {0, 0, 0}, {4, 4, 4});
    const auto distant = box(2, {20, 20, 20}, {24, 24, 24});
    const auto fine = box(3, {4, 4, 1}, {5, 5, 2});
    const auto original = local({{0, coarse}, {0, distant}, {other_owner(), fine}});
    const auto original_bytes = encoded(original);
    D::DistributedAdaptiveContactHalo accepted(kernel, comm, {}, context());
    accepted.bootstrap_canonical(original, lookup(original));
    EXPECT_EQ(marks(accepted.required_refinements()), expected_marks({{0, coarse.key}}));
    const auto before = rows(accepted);
    auto migrated = accepted.fork_candidate(context(230));
    const auto moved = local({{other_owner(), coarse}, {0, distant}, {other_owner(), fine}});
    auto moving_context = context(230, 1);
    moving_context.partition = 14;
    migrated->apply_delta(
        retired_on(0, coarse.key), inserted_on(other_owner(), coarse.key), moving_context, lookup(moved));
    EXPECT_EQ(marks(migrated->required_refinements()), expected_marks({{other_owner(), coarse.key}}));
    auto enriched = corners({0, 0, 0}, {4, 4, 4});
    enriched.insert(enriched.end(), {{4, 4, 1}, {4, 4, 2}});
    const auto after = rows(*migrated);
    expect_row(after, other_owner(), coarse.key, points(enriched));
    expect_row(after, other_owner(), fine.key, points(corners({4, 4, 1}, {5, 5, 2})));
    expect_row(after, 0, distant.key, points(corners({20, 20, 20}, {24, 24, 24})));
    if (comm->getRank() == 0 && before.contains(distant.key) && after.contains(distant.key))
        EXPECT_EQ(after.at(distant.key), before.at(distant.key));
    EXPECT_EQ(rows(accepted), before);
    EXPECT_EQ(marks(accepted.required_refinements()), expected_marks({{0, coarse.key}}));
    EXPECT_EQ(encoded(original), original_bytes);
    EXPECT_EQ(
        global_sum(std::array<std::uint64_t, 1>{moved.empty()})[0], std::uint64_t(std::max(0, comm->getSize() - 2)));
    // Reversing the migration exercises removal of an owner's same-revision
    // record and renewed routing to the old owner without changing geometry.
    auto returning_context = context(230, 2);
    returning_context.partition = 15;
    migrated->apply_delta(
        retired_on(other_owner(), coarse.key), inserted_on(0, coarse.key), returning_context, lookup(original));
    EXPECT_EQ(marks(migrated->required_refinements()), expected_marks({{0, coarse.key}}));
    expect_row(rows(*migrated), 0, coarse.key, points(enriched));
}

using DeltaWork = std::array<std::uint64_t, 12>;
DeltaWork delta_work(const D::ContactHaloStatistics& before, const D::ContactHaloStatistics& after,
    const Calls& calls_before, const Calls& calls_after, std::uint64_t lookups)
{
    return global_sum(DeltaWork{after.dirty_supports - before.dirty_supports,
        after.dirty_support_members - before.dirty_support_members, after.dirty_lines - before.dirty_lines,
        after.dirty_subscriptions - before.dirty_subscriptions, after.recomputed_tiles - before.recomputed_tiles,
        after.delta_owned_visits - before.delta_owned_visits,
        after.full_owned_scans_after_bootstrap - before.full_owned_scans_after_bootstrap,
        calls_after[0] - calls_before[0], calls_after[1] - calls_before[1], calls_after[2] - calls_before[2],
        calls_after[3] - calls_before[3], lookups});
}
TEST(IncrementalAdaptiveClosure, DistantComponentsDoNotIncreaseDirtyOrProviderWork)
{
    const auto comm = Tpetra::getDefaultComm();
    std::optional<DeltaWork> insertion_reference, deletion_reference;
    for (const unsigned count : {1, 8, 32})
    {
        SCOPED_TRACE(count);
        FixtureKernel kernel;
        const auto coarse = box(1, {0, 0, 0}, {4, 4, 4});
        const auto fine = box(3, {4, 4, 1}, {5, 5, 2});
        auto original = local({{0, coarse}});
        if (comm->getRank() == 0)
            for (unsigned i = 0; i < count; ++i)
            {
                const double low = 20 + 10 * i;
                original.push_back(box(100 + i, {low, low, low}, {low + 4, low + 4, low + 4}));
            }
        D::DistributedAdaptiveContactHalo accepted(kernel, comm, {}, context());
        accepted.bootstrap_canonical(original, lookup(original));
        EXPECT_TRUE(accepted.required_refinements().empty());
        const auto before_rows = rows(accepted);
        auto candidate = accepted.fork_candidate(context(240));
        auto prospective = original;
        if (comm->getRank() == other_owner())
            prospective.push_back(fine);
        std::uint64_t lookups = 0;
        const auto before = candidate->statistics();
        const auto calls_before = kernel.calls();
        candidate->apply_delta(
            {}, inserted_on(other_owner(), fine.key), context(240, 1), lookup(prospective, &lookups));
        EXPECT_EQ(marks(candidate->required_refinements()), expected_marks({{0, coarse.key}}));
        const auto inserted = delta_work(before, candidate->statistics(), calls_before, kernel.calls(), lookups);
        EXPECT_EQ(inserted[0], 6u);
        EXPECT_EQ(inserted[1], 6u);
        EXPECT_GT(inserted[2], 0u);
        EXPECT_GT(inserted[3], 0u);
        EXPECT_GT(inserted[4], 0u);
        EXPECT_EQ(inserted[5], 1u);
        EXPECT_EQ(inserted[6], 0u);
        if (insertion_reference)
            EXPECT_EQ(inserted, *insertion_reference);
        else
            insertion_reference = inserted;
        const auto after_rows = rows(*candidate);
        for (const auto& [key, row] : before_rows)
            if (key != coarse.key)
            {
                EXPECT_TRUE(after_rows.contains(key));
                if (after_rows.contains(key))
                    EXPECT_EQ(after_rows.at(key), row);
            }
        lookups = 0;
        const auto delete_before = candidate->statistics();
        const auto delete_calls = kernel.calls();
        candidate->apply_delta(retired_on(other_owner(), fine.key), {}, context(240, 2), lookup(original, &lookups));
        EXPECT_TRUE(candidate->required_refinements().empty());
        const auto deleted = delta_work(delete_before, candidate->statistics(), delete_calls, kernel.calls(), lookups);
        EXPECT_GT(deleted[2], 0u);
        EXPECT_EQ(deleted[5], 1u);
        EXPECT_EQ(deleted[6], 0u);
        if (deletion_reference)
            EXPECT_EQ(deleted, *deletion_reference);
        else
            deletion_reference = deleted;
        EXPECT_EQ(rows(accepted), before_rows);
    }
}

TEST(IncrementalAdaptiveClosure, PeriodicImagePropagationAndDeletionRetainPhysicalPoints)
{
    const auto comm = Tpetra::getDefaultComm();
    FixtureKernel kernel(true);
    auto low = box(1, {0, 0, 0}, {4, 4, 4});
    auto high = box(2, {8, 0, 0}, {12, 4, 4});
    const auto fine = box(3, {11, 4, 1}, {12, 5, 2});
    low.provider = high.provider = AMR::AdaptiveProviderKind::Cartesian;
    // Only the two seam faces share an interior support. Its canonical image
    // is x=0; all remaining faces are independent valid exterior supports.
    for (auto& side : low.surfaces)
        if (side.geometry->area_vector().x < 0)
        {
            side.key = {'p'};
            side.orientation = -1;
            side.exterior = false;
            side.boundary.clear();
        }
    for (auto& side : high.surfaces)
        if (side.geometry->area_vector().x > 0)
        {
            side.key = {'p'};
            side.orientation = 1;
            side.exterior = false;
            side.boundary.clear();
            side.authority_image = P::displacement<ordinary>({12, 0, 0}, {0, 0, 0});
        }
    const auto original = local({{0, low}, {other_owner(), high}});
    const auto prospective = local({{0, low}, {other_owner(), high}, {other_owner(), fine}});
    const auto original_bytes = encoded(original);
    D::DistributedAdaptiveContactHalo accepted(kernel, comm, {}, context());
    accepted.bootstrap_canonical(original, lookup(original));
    EXPECT_TRUE(accepted.required_refinements().empty());
    const auto before = rows(accepted);
    expect_row(before, 0, low.key, points(corners({0, 0, 0}, {4, 4, 4}), true));
    expect_row(before, other_owner(), high.key, points(corners({8, 0, 0}, {12, 4, 4}), true));
    auto candidate = accepted.fork_candidate(context(250));
    candidate->apply_delta({}, inserted_on(other_owner(), fine.key), context(250, 1), lookup(prospective));
    EXPECT_EQ(marks(candidate->required_refinements()), expected_marks({{0, low.key}, {other_owner(), high.key}}));
    auto low_points = corners({0, 0, 0}, {4, 4, 4});
    auto high_points = corners({8, 0, 0}, {12, 4, 4});
    low_points.insert(low_points.end(), {{0, 4, 1}, {0, 4, 2}});
    high_points.insert(high_points.end(), {{12, 4, 1}, {12, 4, 2}});
    const auto inserted = rows(*candidate);
    expect_row(inserted, 0, low.key, points(low_points, true));
    expect_row(inserted, other_owner(), high.key, points(high_points, true));
    expect_row(inserted, other_owner(), fine.key, points(corners({11, 4, 1}, {12, 5, 2}), true));
    for (const auto& [key, row] : before)
        if (inserted.contains(key))
            EXPECT_EQ(inserted.at(key).generation, row.generation + 1);
    EXPECT_EQ(rows(accepted), before);
    candidate->apply_delta(retired_on(other_owner(), fine.key), {}, context(250, 2), lookup(original));
    EXPECT_TRUE(candidate->required_refinements().empty());
    const auto deleted = rows(*candidate);
    expect_row(deleted, 0, low.key, points(corners({0, 0, 0}, {4, 4, 4}), true));
    expect_row(deleted, other_owner(), high.key, points(corners({8, 0, 0}, {12, 4, 4}), true));
    EXPECT_FALSE(deleted.contains(fine.key));
    for (const auto& [key, row] : inserted)
        if (deleted.contains(key))
            EXPECT_EQ(deleted.at(key).generation, row.generation + 1);
    EXPECT_EQ(rows(accepted), before);
    EXPECT_EQ(encoded(original), original_bytes);
}
} // namespace
