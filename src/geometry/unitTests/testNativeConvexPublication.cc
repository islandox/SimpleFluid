/** @file testNativeConvexPublication.cc
 * @brief Built-in convex source admission and distributed native publication.
 */
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/unitTests/polyhedral_amr_helpers.hh"
#include "native_constructor_allocation_operators.inc"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <bit>
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>

namespace
{
using namespace SimpleFluid;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
using Hierarchy = A::DistributedAdaptiveHierarchy;
using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
using Admission = geometry_detail::NativeConstructionAdmission;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(NativeConvexAdmission, ExactDecodePeakRejectsOneByteShortAndPreservesAcceptedOwners)
{
    constexpr std::size_t budget = 64 * 1024 * 1024;
    const A::PolyhedralAMR original(test::polyhedral_grid(test::polyhedral_tetrahedron()));
    const std::array<std::uint64_t, 1> owned{0};
    const auto detached = original.detach_owned(owned, {'t', 'e', 't', 'r', 'a'});
    const auto packet = A::encode_adaptive_batch(detached.owned, detached.families, budget);
    const auto make_admission = [](std::size_t bytes)
    {
        auto arena = std::make_shared<D::Budget>(D::Budget{bytes});
        return Admission(geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(
            bytes, 0, D::NativeSourceOwnerStore::mirror(arena)));
    };
    const auto writer = make_admission(budget);
    const auto storage = geometry_detail::native_constructor_storage(writer);
    native_constructor_test::Capture encoding_trace(*writer.state()->used_counter);
    const auto direct = D::encode_adaptive_batch_admitted(detached.owned, detached.families, storage);
    const auto encoded = encoding_trace.finish();
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_EQ(direct, packet);
    EXPECT_GE(encoded.all_live, direct.capacity());

    const auto measured = make_admission(budget);
    const auto initial = measured.counters().used;
    native_constructor_test::Capture trace(*measured.state()->used_counter);
    const auto accepted = D::decode_adaptive_batch_admitted_geometry(packet, budget, measured);
    const auto raw = trace.finish();
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_LE(raw.all_peak, measured.statistics().admission_peak - initial);
    ASSERT_EQ(accepted.leaves.size(), 1u);
    const auto& source = accepted.leaves.front();
    EXPECT_TRUE(geometry_detail::NativeObjectOwnerAccess<Cell>::admitted(source.geometry));
    EXPECT_DOUBLE_EQ(source.geometry->volume().as_double(), 1.0 / 6.0);
    ASSERT_EQ(source.surfaces.size(), 4u);
    for (std::size_t face = 0; face < source.surfaces.size(); ++face)
    {
        EXPECT_TRUE(PLIC::details::NativePlanarOwnerAccess::admitted(source.surfaces[face].geometry));
        EXPECT_EQ(source.surfaces[face].geometry->encode_incidence(),
            detached.owned.front().surfaces[face].geometry->encode_incidence());
    }
    const auto budget_peak = measured.statistics().admission_peak;
    const auto exact = make_admission(budget_peak);
    EXPECT_NO_THROW((void) D::decode_adaptive_batch_admitted_geometry(packet, budget, exact));
    const auto short_admission = make_admission(budget_peak - 1);
    const auto before = short_admission.counters().used;
    const auto accepted_geometry = source.geometry.get();
    EXPECT_THROW((void) D::decode_adaptive_batch_admitted_geometry(packet, budget, short_admission),
        std::invalid_argument);
    EXPECT_EQ(short_admission.counters().used, before);
    EXPECT_EQ(source.geometry.get(), accepted_geometry);
}

TEST(NativeConvexAdmission, OrdinarySplitAdmitsRawPositiveAndEmptyOctants)
{
    const A::PolyhedralAMR original(test::polyhedral_grid(test::polyhedral_tetrahedron()));
    const std::array<std::uint64_t, 1> owned{0};
    const auto detached = original.detach_owned(owned, {'s', 'p', 'l', 'i', 't'});
    const auto& parent = *detached.owned.front().geometry;
    constexpr std::size_t budget = 64 * 1024 * 1024;
    const auto admission = Admission(geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(budget));
    const auto prefix = admission.counters().used;
    struct SplitFacts
    {
        const Cell& parent;
        std::size_t count = 0;
        long double volume = 0;
    } facts{parent};
    native_constructor_test::Capture trace(*admission.state()->used_counter);
    PLIC::details::NativeCellCodecAccess::with_split_profile(admission, &facts, [](void* raw)
    {
        auto& results = *static_cast<SplitFacts*>(raw);
        const auto children = PLIC::details::NativeCellCodecAccess::profiled_octants(results.parent, 0);
        results.count = children.size();
        for (const auto& child : children)
            results.volume += child.geometry->volume().value;
    });
    const auto raw = trace.finish();
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_EQ(raw.all_live, 0u);
    EXPECT_EQ(admission.counters().used, prefix);
    EXPECT_EQ(facts.count, 4u);
    EXPECT_DOUBLE_EQ(static_cast<double>(facts.volume), 1.0 / 6.0);
}

void same_bits(double expected, double actual)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(expected), std::bit_cast<std::uint64_t>(actual));
}
void same_bits(MeshUtils::Vec3 expected, MeshUtils::Vec3 actual)
{
    same_bits(expected.x, actual.x);
    same_bits(expected.y, actual.y);
    same_bits(expected.z, actual.z);
}

void parity(const Hierarchy& h, const A::AdaptiveNativePublicationView& publication)
{
    const auto legacy = A::assemble_adaptive_native(h, h.configuration_descriptor());
    ASSERT_TRUE(legacy.partition);
    ASSERT_TRUE(publication.partition());
    const auto& reference = *legacy.partition->geometry();
    const auto& actual = *publication.partition()->geometry();
    const auto& expected = legacy.partition->plan();
    const auto& plan = publication.partition()->plan();
    EXPECT_EQ(expected.cell_ids(), plan.cell_ids());
    EXPECT_EQ(expected.face_ids(), plan.face_ids());
    EXPECT_EQ(expected.node_ids(), plan.node_ids());
    for (const auto cell : plan.cell_ids())
    {
        EXPECT_EQ(legacy.local_keys.at(cell), publication.local_keys().at(cell));
        same_bits(reference.cell_volume(cell), actual.cell_volume(cell));
        same_bits(reference.cell_centroid(cell), actual.cell_centroid(cell));
    }
    for (const auto face : plan.face_ids())
    {
        EXPECT_EQ(reference.owner_cell(face), actual.owner_cell(face));
        EXPECT_EQ(reference.neighbor_cell(face), actual.neighbor_cell(face));
        EXPECT_EQ(reference.boundary_id(face), actual.boundary_id(face));
        same_bits(reference.face_area(face), actual.face_area(face));
        same_bits(reference.face_centroid(face), actual.face_centroid(face));
        same_bits(reference.face_normal(face), actual.face_normal(face));
    }
    for (const auto node : plan.node_ids())
        same_bits(reference.node_coordinates(node), actual.node_coordinates(node));
}

class ConvexPublication
{
public:
    const Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), ranks = comm->getSize();
    std::shared_ptr<const Hierarchy> make() const
    {
        const A::PolyhedralAMR original(test::polyhedral_grid());
        const std::array<std::uint64_t, 2> roots{0, 1};
        const auto owned = ranks == 1 ? std::span<const std::uint64_t>(roots)
            : rank < 2               ? std::span<const std::uint64_t>(roots).subspan(rank, 1)
                                     : std::span<const std::uint64_t>{};
        auto input = original.detach_owned(owned, {'c', 'o', 'n', 'v', 'e', 'x'});
        return std::make_shared<const Hierarchy>(
            std::move(input.owned), std::move(input.families), std::move(input.kernel), comm);
    }

    A::AdaptiveNativePublicationView publish(const Hierarchy& source) const
    {
        const auto result = A::publish_adaptive_native(source, A::native_publication_schema(source));
        D::collective(*comm, [&]
        {
            const auto current = D::NativePublicationSession::capture(source);
            if (!current.source_coverage || !D::NativePublicationSession::receipt_matches(
                    current, result.partition(), result.mesh()->owned_cell_map()))
                throw std::invalid_argument("Convex fixture requires actual qualified hierarchy publication.");
        });
        return result;
    }
};

TEST(NativeConvexPublication, ProfiledFactoryAdmitsSevenChildRowsAndHeapBoundaryLabels)
{
    const std::string expected_label = "truncated_cube/skew_wall";
    const A::PolyhedralAMR original(test::polyhedral_grid(test::polyhedral_truncated_cube("skew_wall")));
    const std::array<std::uint64_t, 1> ids{0};
    const auto detached = original.detach_owned(ids, {'s', 'k', 'e', 'w'});
    ASSERT_EQ(detached.owned.size(), 1u);
    ASSERT_GT(expected_label.size(), std::string{}.capacity());
    ASSERT_LT(expected_label.size(), 30u);
    ASSERT_EQ(detached.owned.front().surfaces.size(), 7u);
    for (const auto& face : detached.owned.front().surfaces)
    {
        ASSERT_TRUE(face.exterior);
        ASSERT_EQ(face.boundary, expected_label);
    }
    constexpr std::size_t budget = 64 * 1024 * 1024;
    const auto admission = Admission(geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(budget));
    struct Facts
    {
        const std::string& label;
        std::size_t leaves = 0, leaf_extent = 0, children_extent = 0, labelled = 0;
    } facts{expected_label};
    const auto used = admission.counters().used;
    native_constructor_test::Capture trace(*admission.state()->used_counter);
    D::NativeKernelAuthority::profiled_source_split(detached.kernel, detached.owned.front(), A::AdaptiveSplit::Isotropic,
        admission, &facts, [](void* raw, A::AdaptiveSplitResult&& split)
        {
            auto& observed = *static_cast<Facts*>(raw);
            observed.leaves = split.children.size();
            observed.leaf_extent = split.children.capacity();
            observed.children_extent = split.family.children.capacity();
            for (const auto& leaf : split.children)
                for (const auto& face : leaf.surfaces)
                    observed.labelled += face.boundary == observed.label;
        });
    const auto summary = trace.finish();
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_EQ(summary.all_live, 0u);
    EXPECT_EQ(admission.counters().used, used);
    EXPECT_EQ(facts.leaves, 7u);
    EXPECT_EQ(facts.leaf_extent, facts.leaves);
    EXPECT_EQ(facts.children_extent, facts.leaves);
    EXPECT_GT(facts.labelled, 0u);
}

TEST(NativeConvexPublication, ColdAndWarmBindingPreserveOriginalOwnersAfterHierarchyRetires)
{
    const ConvexPublication test;
    auto h = test.make();
    const std::weak_ptr<const A::AdaptiveProviderKernel> policy = D::NativePublicationSession::capture(*h).kernel;
    const std::weak_ptr<const A::AdaptiveLeafLayout> layout = h->leaf_layout();
    const auto cold = test.publish(*h);
    parity(*h, cold);
    auto bound = h->with_partition(cold.partition(), cold.mesh()->owned_cell_map());
    D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
    const auto warm = test.publish(*bound);
    auto rebound = bound->with_partition(warm.partition(), warm.mesh()->owned_cell_map());
    const auto third = test.publish(*rebound);
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, 0u);
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().source_imports, 0u);
    for (const auto& leaf : h->owned_leaves())
    {
        const auto gid = cold.owned_gids().at(leaf.key);
        const auto cold_geometry = cold.partition()->geometry()->certified_sparse_provider_cell_geometry(gid);
        const auto warm_geometry = warm.partition()->geometry()->certified_sparse_provider_cell_geometry(gid);
        EXPECT_EQ(cold_geometry, leaf.geometry);
        EXPECT_FALSE(cold_geometry.owner_before(leaf.geometry) || leaf.geometry.owner_before(cold_geometry));
        EXPECT_EQ(warm_geometry, cold_geometry);
    }
    rebound.reset();
    bound.reset();
    h.reset();
    EXPECT_TRUE(layout.expired());
    EXPECT_TRUE(policy.expired());
    for (const auto& publication : {cold, warm, third})
        for (const auto gid : publication.partition()->plan().owned_cells())
            EXPECT_GT(publication.partition()->geometry()->cell_volume(gid), 0);
}

TEST(NativeConvexPublication, MovingBothCellsToOneRankKeepsCurrentIncidenceAndEmptyOwners)
{
    const ConvexPublication test;
    if (test.ranks == 1)
        GTEST_SKIP() << "Requires the registered MPI communicator.";
    auto h = test.make();
    const auto before = test.publish(*h);
    h = h->with_partition(before.partition(), before.mesh()->owned_cell_map());
    const std::vector<int> destinations(h->owned_leaves().size(), 1);
    const auto plan = h->plan_migration(destinations);
    const std::uint64_t local = plan.native_migration().remote_sent_packets;
    std::uint64_t sent = 0;
    Teuchos::reduceAll(*test.comm, Teuchos::REDUCE_SUM, 1, &local, &sent);
    EXPECT_GT(sent, 0u);
    h = h->apply(plan);
    EXPECT_EQ(h->owned_leaves().size(), test.rank == 1 ? 2u : 0u);
    const auto moved = test.publish(*h);
    parity(*h, moved);
    auto bound = h->with_partition(moved.partition(), moved.mesh()->owned_cell_map());
    EXPECT_EQ(bound->native_partition(), moved.partition());
}

TEST(NativeConvexPublication, FixedSiblingEditKeepsOriginalNeighborAndRechecksIncidence)
{
    const ConvexPublication test;
    auto h = test.make();
    const auto before = test.publish(*h);
    h = h->with_partition(before.partition(), before.mesh()->owned_cell_map());
    const A::AdaptiveCellKey parent{(std::uint64_t{1} << 55) + 19, A::ConvexCellAncestry{0, {}}};
    const A::AdaptiveCellKey sibling{parent.region, A::ConvexCellAncestry{1, {}}};
    const auto proof = [&](const Hierarchy& current)
    {
        const auto input = D::NativePublicationSession::capture(current);
        const std::lock_guard lock(input.cache->mutex);
        const auto* accepted = input.cache->indexes ? input.cache->indexes->find(sibling) : nullptr;
        return accepted ? *accepted : D::NativePublicationIndexRow{};
    };
    const auto original = proof(*h);
    std::vector<A::AdaptiveMark> marks;
    if (h->owned_leaves().contains(parent))
        marks.push_back({parent, A::AdaptiveSplit::Isotropic});
    const auto changed = h->apply(h->plan_adaptation(marks));
    const auto after = test.publish(*changed);
    parity(*changed, after);
    const auto current = proof(*changed);
    if (changed->owned_leaves().contains(sibling))
    {
        ASSERT_TRUE(original.source && original.incidence && current.source && current.incidence);
        EXPECT_EQ(current.source, original.source);
        EXPECT_NE(current.incidence, original.incidence);
        EXPECT_EQ(changed->owned_leaves().at(sibling).geometry, h->owned_leaves().at(sibling).geometry);
    }
    for (const auto& leaf : changed->owned_leaves())
        EXPECT_TRUE(geometry_detail::NativeObjectOwnerAccess<Cell>::admitted(leaf.geometry));
}

TEST(NativeConvexPublication, RemoteSavedFamilyFetchAuthenticatesSevenChildrenAfterMigration)
{
    const ConvexPublication test;
    if (test.ranks == 1)
        GTEST_SKIP() << "Requires the registered MPI communicator.";
    const A::PolyhedralAMR original(test::polyhedral_grid(test::polyhedral_truncated_cube()));
    const std::array<std::uint64_t, 1> root{0};
    const auto owned = test.rank == 0 ? std::span<const std::uint64_t>(root) : std::span<const std::uint64_t>{};
    auto detached = original.detach_owned(owned, {'f', 'a', 'm', 'i', 'l', 'y'});
    auto source = std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), test.comm);
    const auto initial = test.publish(*source);
    source = source->with_partition(initial.partition(), initial.mesh()->owned_cell_map());
    std::vector<A::AdaptiveMark> marks;
    if (test.rank == 0)
        marks.push_back({source->owned_leaves().begin()->key, A::AdaptiveSplit::Isotropic});
    const auto refined = source->apply(source->plan_adaptation(marks));
    EXPECT_EQ(refined->global_cells(), 7u);

    const std::vector<int> destinations(refined->owned_leaves().size(), 1);
    const auto migration = refined->plan_migration(destinations);
    const auto moved = refined->apply(migration);
    const A::AdaptiveCellKey parent{(std::uint64_t{1} << 55) + 19, A::ConvexCellAncestry{0, {}}};
    const auto family_key = A::adaptive_family_key(parent, A::AdaptiveSplit::Isotropic);
    const std::vector<A::AdaptiveBytes> requested =
        test.rank == 0 ? std::vector<A::AdaptiveBytes>{family_key} : std::vector<A::AdaptiveBytes>{};
    const auto fetched = moved->fetch_families(requested);
    EXPECT_EQ(moved->owned_leaves().size(), test.rank == 1 ? 7u : 0u);
    EXPECT_EQ(fetched.size(), requested.size());
    for (const auto& capsule : fetched)
    {
        EXPECT_EQ(capsule.key, family_key);
        EXPECT_EQ(capsule.parent.key, parent);
        EXPECT_EQ(capsule.children.size(), 7u);
        EXPECT_TRUE(geometry_detail::NativeObjectOwnerAccess<Cell>::admitted(capsule.parent.geometry));
        EXPECT_DOUBLE_EQ(capsule.parent.geometry->volume().as_double(), 61.0 / 192.0);
    }
    std::vector<int> sibling_destinations;
    for (std::size_t child = 0; child < moved->owned_leaves().size(); ++child)
        sibling_destinations.push_back(child % 2 == 0 ? 1 : 0);
    const auto split_owners = moved->apply(moved->plan_migration(sibling_destinations));
    EXPECT_EQ(split_owners->owned_leaves().size(), test.rank == 0 ? 3u : test.rank == 1 ? 4u : 0u);
    EXPECT_EQ(split_owners->owned_families().contains(family_key), test.rank == 1);
    const auto reverted = split_owners->apply(split_owners->plan_adaptation({}, requested));
    EXPECT_EQ(reverted->global_cells(), 1u);
    const auto restored = test.publish(*reverted);
    parity(*reverted, restored);
}

TEST(NativeConvexPublication, RankLocalProfileChangeRejectsExplicitIntakeOnEveryRank)
{
    const ConvexPublication test;
    if (test.ranks == 1)
        GTEST_SKIP() << "Requires the registered MPI communicator.";
    const auto source = test.make();
    const A::AdaptiveCellKey parent{(std::uint64_t{1} << 55) + 19, A::ConvexCellAncestry{0, {}}};
    std::vector<A::AdaptiveMark> marks;
    if (source->owned_leaves().contains(parent))
        marks.push_back({parent, A::AdaptiveSplit::Isotropic});
    const auto refined = source->apply(source->plan_adaptation(marks));
    const auto observation = refined->observation();
    const std::vector<A::AdaptiveBytes> requested = test.rank == 0
        ? std::vector<A::AdaptiveBytes>{A::adaptive_family_key(parent, A::AdaptiveSplit::Isotropic)}
        : std::vector<A::AdaptiveBytes>{};
    const std::vector<int> destinations(refined->owned_leaves().size(), 1);
    struct ProfileEnvironment
    {
        std::optional<std::string> original;
        bool changed = false;
        ~ProfileEnvironment()
        {
            if (changed && (original ? setenv("TPETRA_VERBOSE", original->c_str(), 1) : unsetenv("TPETRA_VERBOSE")))
                std::terminate();
        }
    };
    {
        ProfileEnvironment environment;
        D::collective(*test.comm, [&]
        {
            if (test.rank != 1)
                return;
            if (const char* value = std::getenv("TPETRA_VERBOSE"))
                environment.original = value;
            if (setenv("TPETRA_VERBOSE", "0", 1))
                throw std::runtime_error("Cannot configure the rank-local map profile test.");
            environment.changed = true;
        });
        const auto rejects_changed_profile = [](auto operation)
        {
            try
            {
                operation();
                ADD_FAILURE() << "The accepted explicit intake profile changed during the operation.";
            }
            catch (const std::invalid_argument& error)
            {
                EXPECT_STREQ(error.what(), "Accepted native explicit source profile changed on a rank.");
            }
        };
        rejects_changed_profile([&] { (void) refined->fetch_families(requested); });
        rejects_changed_profile([&] { (void) refined->plan_migration(destinations); });
        rejects_changed_profile([&] { (void) refined->plan_adaptation({}, requested); });
        EXPECT_EQ(refined->observation(), observation);
    }
    EXPECT_EQ(refined->fetch_families(requested).size(), requested.size());
}
} // namespace
