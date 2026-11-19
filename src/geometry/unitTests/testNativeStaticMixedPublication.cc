/** Actual detached Static and Mixed factories through native publication and binding. */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/MixedAMR.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <mutex>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

using Hierarchy = A::DistributedAdaptiveHierarchy;
using Publication = A::AdaptiveNativePublicationView;

template<class T> T rank_sum(const Teuchos::Comm<int>& comm, T local)
{
    T total{};
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &total);
    return total;
}

void same_bits(double actual, double expected)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}
void same_vector(const MeshUtils::Vec3& actual, const MeshUtils::Vec3& expected)
{
    same_bits(actual.x, expected.x);
    same_bits(actual.y, expected.y);
    same_bits(actual.z, expected.z);
}

void parity(const Publication& view, const A::AdaptiveNativePublication& full)
{
    const auto& actual = *view.partition()->geometry();
    const auto& expected = *full.partition->geometry();
    const auto& plan = view.partition()->plan();
    EXPECT_EQ(plan.cell_ids(), full.partition->plan().cell_ids());
    EXPECT_EQ(plan.face_ids(), full.partition->plan().face_ids());
    EXPECT_EQ(plan.node_ids(), full.partition->plan().node_ids());
    for (const auto id : plan.cell_ids())
    {
        EXPECT_EQ(view.local_keys().at(id), full.local_keys.at(id));
        EXPECT_TRUE(std::ranges::equal(actual.cell_faces(id), expected.cell_faces(id)));
        EXPECT_TRUE(std::ranges::equal(actual.cell_nodes(id), expected.cell_nodes(id)));
        EXPECT_EQ(actual.native_cell(id).second, expected.native_cell(id).second);
        same_bits(actual.cell_volume(id), expected.cell_volume(id));
        same_vector(actual.cell_centroid(id), expected.cell_centroid(id));
    }
    EXPECT_EQ(actual.boundary_batch_ids(), expected.boundary_batch_ids());
    for (const auto boundary : actual.boundary_batch_ids())
        EXPECT_EQ(actual.boundary_batch_name(boundary), expected.boundary_batch_name(boundary));
    for (const auto face : plan.face_ids())
    {
        EXPECT_TRUE(std::ranges::equal(actual.face_nodes(face), expected.face_nodes(face)));
        EXPECT_EQ(actual.owner_cell(face), expected.owner_cell(face));
        EXPECT_EQ(actual.neighbor_cell(face), expected.neighbor_cell(face));
        EXPECT_EQ(actual.boundary_id(face), expected.boundary_id(face));
        const auto native = actual.native_face(face), full_native = expected.native_face(face);
        EXPECT_EQ(native.face, full_native.face);
        same_bits(actual.native_flux_orientation(native, face), expected.native_flux_orientation(full_native, face));
        same_vector(actual.face_area_vector(face), expected.face_area_vector(face));
        same_vector(actual.face_centroid(face), expected.face_centroid(face));
    }
    for (const auto node : plan.node_ids())
        same_vector(actual.node_coordinates(node), expected.node_coordinates(node));
}

Publication publish(const Hierarchy& h)
{
    auto view = A::publish_adaptive_native(h, A::native_publication_schema(h));
    D::collective(*h.communicator(), [&]
        {
            if (!D::NativePublicationSession::receipt_matches(
                    D::NativePublicationSession::capture(h), view.partition(), view.mesh()->owned_cell_map()))
                throw std::invalid_argument("Fixture requires the actual native publication receipt.");
        });
    return view;
}

std::shared_ptr<const Hierarchy> bind(const Hierarchy& h, const Publication& view)
{
    const auto before = D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions;
    auto bound = h.with_partition(view.partition(), view.mesh()->owned_cell_map());
    EXPECT_EQ(bound->native_partition(), view.partition());
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, before);
    return bound;
}

struct Proofs
{
    std::shared_ptr<const D::NativeCellSourceProof> source;
    std::shared_ptr<const D::NativeCellIncidenceProof> incidence;
};
Proofs proof(const Hierarchy& h, const A::AdaptiveCellKey& key)
{
    auto input = D::NativePublicationSession::capture(h);
    const std::lock_guard lock(input.cache->mutex);
    if (!input.cache->indexes)
        throw std::invalid_argument("Fixture requires a native source cache.");
    if (const auto* found = input.cache->indexes->find(key))
        return {found->source, found->incidence};
    return {};
}

TEST(NativeStaticMixedPublication, LegacyStaticMetricsOrdinalsAndRemoteReuseSurviveMigration)
{
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), ranks = comm->getSize();
    auto mesh = std::make_shared<OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{.1, .3, .5, .7, .9}, {-.2, .3}, {.6, 1.3}}});
    auto original = native_region("legacy_static", mesh);
    std::vector<std::uint64_t> cells;
    for (std::uint64_t cell = 0; cell < 4; ++cell)
        if (rank == (ranks == 1 ? 0 : int(cell / 2)))
            cells.push_back(cell);
    auto input = A::detach_static_participant(original, 71, 8, 3, cells, {'s', 't', 'a', 't', 'i', 'c'});
    auto h = std::make_shared<const Hierarchy>(std::move(input.owned), std::move(input.families), input.kernel, comm);
    const auto first = publish(*h);
    parity(first, A::assemble_adaptive_native(*h, h->configuration_descriptor()));
    auto bound = bind(*h, first);
    const A::AdaptiveCellKey shared{8, A::StaticCellAncestry{71, 2}};
    auto cold = proof(*bound, shared);
    D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
    const auto again = publish(*bound);
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, 0u);
    D::collective(*comm, [&]
        {
            if (rank >= 2)
                return;
            const auto warm = proof(*bound, shared);
            if (!cold.source || !cold.incidence || !warm.source || !warm.incidence)
                throw std::invalid_argument("Fixture requires the accepted shared original cell proofs.");
            EXPECT_EQ(warm.source, cold.source);
            EXPECT_EQ(warm.incidence, cold.incidence);
            EXPECT_EQ(bool(warm.incidence->remote_record()), ranks > 1 && rank == 0);
            EXPECT_EQ(warm.source->native_cell(), 2u);
        });
    cold = {};
    if (ranks > 1)
    {
        std::vector<int> destinations;
        for (const auto& leaf : bound->owned_leaves())
            destinations.push_back(leaf.native_cell == 1 ? 1 : rank);
        const auto migration = bound->plan_migration(destinations);
        EXPECT_GT(rank_sum(*comm, migration.native_migration().remote_sent_packets), 0u);
        bound = bound->apply(migration);
        const A::AdaptiveCellKey moved{8, A::StaticCellAncestry{71, 1}};
        EXPECT_EQ(bound->owned_leaves().contains(moved), rank == 1);
        const auto migrated = publish(*bound);
        parity(migrated, A::assemble_adaptive_native(*bound, bound->configuration_descriptor()));
        bound = bind(*bound, migrated);
        destinations.assign(bound->owned_leaves().size(), 1);
        const auto gather = bound->plan_migration(destinations);
        EXPECT_GT(rank_sum(*comm, gather.native_migration().remote_sent_packets), 0u);
        bound = bound->apply(gather);
        const auto gathered = publish(*bound);
        EXPECT_EQ(gathered.owned_gids().size(), rank == 1 ? 4u : 0u);
        parity(gathered, A::assemble_adaptive_native(*bound, bound->configuration_descriptor()));
    }
    const std::weak_ptr<const A::AdaptiveProviderKernel> weak = input.kernel;
    input = {};
    h.reset();
    bound.reset();
    EXPECT_TRUE(weak.expired());
    for (const auto& view : {first, again})
        for (const auto& leaf : view.owned_gids())
        {
            const auto ordinal = std::get<A::StaticCellAncestry>(leaf.first.ancestry).native_cell;
            const auto& geometry = *view.partition()->geometry();
            EXPECT_EQ(geometry.native_cell(leaf.second).second, ordinal);
            same_bits(geometry.cell_volume(leaf.second), mesh->cell_volume(mesh->cell_id(ordinal)));
        }
}

TEST(NativeStaticMixedPublication, JoinedStaticSourceSurvivesChangedIncidenceFromCartesianCompanion)
{
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), ranks = comm->getSize();
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.region_id = 4;
    grid.material_id = 3;
    auto refinable = std::make_shared<const A::CartesianAMR>(grid);
    auto stationary = std::make_shared<OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{1, 1.5}, {0, .5, 1}, {0, .5, 1}}});
    A::MixedAMRGrid mixed;
    mixed.adaptive = {refinable};
    mixed.stationary = {{native_region("stationary", stationary), 8, 3, 71}};
    mixed.joins = {{0, 1, "xmax", "xmin"}};
    const A::MixedAMR original(mixed);
    std::vector<std::uint64_t> cells;
    const A::CartesianCellAncestry refine_cell{0, {1, 0, 0}};
    const A::AdaptiveCellKey fixed{8, A::StaticCellAncestry{71, 0}};
    A::AdaptiveCellKey cartesian;
    bool found_refine_cell = false, found_fixed_cell = false;
    for (std::uint64_t cell = 0; cell < original.geometry()->num_cells(); ++cell)
    {
        const auto key = original.cell_key(cell);
        if (std::holds_alternative<A::StaticCellAncestry>(key.ancestry))
        {
            found_fixed_cell |= key == fixed;
            if (rank == std::min(1, ranks - 1))
                cells.push_back(cell);
        }
        else
        {
            if (std::get<A::CartesianCellAncestry>(key.ancestry) == refine_cell)
            {
                cartesian = key;
                found_refine_cell = true;
            }
            if (rank == 0)
                cells.push_back(cell);
        }
    }
    ASSERT_TRUE(found_refine_cell);
    ASSERT_TRUE(found_fixed_cell);
    auto input = original.detach_owned(cells, {'m', 'i', 'x', 'e', 'd'});
    const Hierarchy h(std::move(input.owned), std::move(input.families), input.kernel, comm);
    const auto first = publish(h);
    parity(first, A::assemble_adaptive_native(h, h.configuration_descriptor()));
    const auto bound = bind(h, first);
    const auto before = proof(*bound, fixed);
    std::vector<A::AdaptiveMark> marks;
    if (rank == 0)
        marks.push_back({cartesian, A::AdaptiveSplit::Isotropic});
    const auto edited = bound->apply(bound->plan_adaptation(marks));
    std::vector<const A::AdaptiveLeaf*> geometry_rows;
    for (const auto& leaf : edited->owned_leaves())
        geometry_rows.push_back(&leaf);
    std::size_t geometry_bytes = 0;
    const auto warmed = edited->geometry_snapshot(geometry_rows, ArithmeticMode::Ordinary,
        A::GeometrySnapshotPurpose::State, geometry_bytes, edited->options().byte_budget);
    ASSERT_TRUE(warmed);
    for (const auto* leaf : geometry_rows)
        EXPECT_TRUE(warmed->find(*leaf));
    const auto after = publish(*edited);
    parity(after, A::assemble_adaptive_native(*edited, edited->configuration_descriptor()));
    // Cartesian adaptation refines the whole 2-by-2-by-2 companion block.
    EXPECT_EQ(edited->global_cells(), 4u * 4u * 4u + 4u);
    D::collective(*comm, [&]
        {
            if (rank >= 2)
                return;
            const auto current = proof(*edited, fixed);
            if (!before.source || !before.incidence || !current.incidence)
                throw std::invalid_argument("Fixture requires source and incidence for the stationary join.");
            EXPECT_EQ(current.source, before.source);
            EXPECT_NE(current.incidence, before.incidence);
        });
    (void) bind(*edited, after);
    double local_join_area = 0;
    std::uint64_t local_join_faces = 0;
    const auto& geometry = *after.partition()->geometry();
    for (const auto face : after.partition()->plan().owned_faces())
        if (!geometry.is_exterior_face(face) && geometry.face_centroid(face).x == 1)
        {
            local_join_area += geometry.face_area(face);
            ++local_join_faces;
        }
    EXPECT_DOUBLE_EQ(rank_sum(*comm, local_join_area), 1);
    EXPECT_EQ(rank_sum(*comm, local_join_faces), 4u * 4u);
}
} // namespace
