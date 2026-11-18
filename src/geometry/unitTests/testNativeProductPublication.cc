/** Real Product factory publication, repack controls, and distributed ownership. */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedProductKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/NativeFragmentAccess.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <mutex>
#include <numbers>
#include <string>

namespace
{
using namespace SimpleFluid;
namespace M = Meshes;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

A::ExtrudedAMRGrid strip(unsigned layers)
{
    A::ExtrudedAMRGrid grid;
    grid.region_id = 5;
    grid.material_id = 11;
    grid.xy_nodes = {{-.7, .13, 0}, {-.4, .13, 0}, {-.1, .13, 0}, {-.7, .6, 0}, {-.4, .6, 0}, {-.1, .6, 0}};
    grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}};
    grid.boundaries = {{0, 1, "heated lower side of the strip"}, {1, 2, "heated lower side of the strip"}};
    grid.boundary_aliases = {{"zmin", "floor"}, {"zmax", "roof"}, {"side", "wall"},
        {"heated lower side of the strip", "heated boundary of decimal extrusion"}};
    for (unsigned i = 0; i <= layers; ++i) grid.z_edges.push_back(-.37 + .17 * i);
    return grid;
}

std::shared_ptr<const A::DistributedAdaptiveHierarchy> make_hierarchy(
    const A::ExtrudedAMRGrid& grid, bool split_owners = false)
{
    const auto comm = Tpetra::getDefaultComm();
    A::ExtrudedAMR factory(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t ordinal = 0; ordinal < grid.base_cells.size() * (grid.z_edges.size() - 1); ++ordinal)
    {
        const auto key = factory.cell_key(ordinal);
        const auto base = std::get<A::ProductCellAncestry>(key.ancestry).base.root;
        if (comm->getRank() == (split_owners ? int(base % 2) : 0)) owned.push_back(ordinal);
    }
    auto leaves = factory.detach_owned(owned, {'p', 'r', 'o', 'd', 'u', 'c', 't'});
    A::DistributedAdaptiveOptions options;
    options.maximum_span_ratio = 4;
    return std::make_shared<const A::DistributedAdaptiveHierarchy>(
        std::move(leaves.owned), std::move(leaves.families), std::move(leaves.kernel), comm, options);
}

A::AdaptiveCellKey key(std::uint64_t base, std::uint64_t axis)
{
    return {5, A::ProductCellAncestry{{base, 1, {}}, {axis, {}}}};
}

template<class Range> std::vector<std::uint64_t> ids(const Range& range)
{
    return {range.begin(), range.end()};
}

void same_bits(double actual, double expected)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}

void same_vector(MeshUtils::Vec3 actual, MeshUtils::Vec3 expected)
{
    same_bits(actual.x, expected.x);
    same_bits(actual.y, expected.y);
    same_bits(actual.z, expected.z);
}

void compare_with_oracle(const A::AdaptiveNativePublicationView& view, const A::AdaptiveNativePublication& oracle)
{
    const auto& a = *view.partition()->geometry();
    const auto& b = *oracle.partition->geometry();
    const auto& plan = view.partition()->plan();
    ASSERT_EQ(plan.cell_ids(), oracle.partition->plan().cell_ids());
    ASSERT_EQ(plan.face_ids(), oracle.partition->plan().face_ids());
    ASSERT_EQ(plan.node_ids(), oracle.partition->plan().node_ids());
    EXPECT_EQ(a.boundary_batch_ids(), b.boundary_batch_ids());
    for (const auto id : a.boundary_batch_ids()) EXPECT_EQ(a.boundary_batch_name(id), b.boundary_batch_name(id));
    for (const auto c : plan.cell_ids())
    {
        EXPECT_EQ(view.local_keys().at(c), oracle.local_keys.at(c));
        EXPECT_EQ(ids(a.cell_faces(c)), ids(b.cell_faces(c)));
        EXPECT_EQ(ids(a.cell_nodes(c)), ids(b.cell_nodes(c)));
        same_bits(a.cell_volume(c), b.cell_volume(c));
        same_vector(a.cell_centroid(c), b.cell_centroid(c));
        const auto [ar, ac] = a.native_cell(c);
        const auto [br, bc] = b.native_cell(c);
        EXPECT_EQ(ac, bc);
        EXPECT_EQ(std::get<M::ExtrudedRegion>(a.region(ar)).name(), std::get<M::ExtrudedRegion>(b.region(br)).name());
    }
    for (const auto f : plan.face_ids())
    {
        EXPECT_EQ(ids(a.face_nodes(f)), ids(b.face_nodes(f)));
        EXPECT_EQ(a.owner_cell(f), b.owner_cell(f));
        EXPECT_EQ(a.neighbor_cell(f), b.neighbor_cell(f));
        EXPECT_EQ(a.boundary_id(f), b.boundary_id(f));
        same_bits(a.face_area(f), b.face_area(f));
        same_vector(a.face_centroid(f), b.face_centroid(f));
        same_vector(a.face_area_vector(f), b.face_area_vector(f));
        const auto af = a.native_face(f), bf = b.native_face(f);
        EXPECT_EQ(af.face, bf.face);
        same_bits(a.native_flux_orientation(af, f), b.native_flux_orientation(bf, f));
    }
    for (const auto n : plan.node_ids()) same_vector(a.node_coordinates(n), b.node_coordinates(n));
}

struct Publication
{
    A::AdaptiveNativePublicationView view;
    std::shared_ptr<const A::DistributedAdaptiveHierarchy> bound;
};

Publication publish(const A::DistributedAdaptiveHierarchy& hierarchy, bool compare = false)
{
    auto view = A::publish_adaptive_native(hierarchy, A::native_publication_schema(hierarchy));
    const auto& map = view.mesh()->owned_cell_map();
    EXPECT_TRUE(D::NativePublicationSession::receipt_matches(
        D::NativePublicationSession::capture(hierarchy), view.partition(), map));
    auto bound = hierarchy.with_partition(view.partition(), map);
    EXPECT_EQ(bound->native_partition(), view.partition());
    if (compare)
        compare_with_oracle(view, A::assemble_adaptive_native(hierarchy, hierarchy.configuration_descriptor()));
    return {std::move(view), std::move(bound)};
}

D::NativePublicationIndexRow proof(const A::DistributedAdaptiveHierarchy& hierarchy, const A::AdaptiveCellKey& key)
{
    const auto source = D::NativePublicationSession::capture(hierarchy);
    std::lock_guard lock(source.cache->mutex);
    const auto& indexes = source.cache->indexes;
    const auto* row = indexes ? indexes->find(key) : nullptr;
    if (!row || !row->source || !row->incidence)
        throw std::invalid_argument("Product publication did not issue its source and its current incidence proofs.");
    return *row;
}

TEST(NativeProductPublication, DecimalTwelveSidedCellMatchesFullPublicationAndUsesOriginalRoutes)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    A::ExtrudedAMRGrid grid;
    std::vector<unsigned> polygon;
    for (unsigned i = 0; i < 12; ++i)
    {
        const auto angle = 2 * std::numbers::pi * i / 12;
        grid.xy_nodes.push_back({.13 + std::cos(angle), -.29 + std::sin(angle), 0});
        polygon.push_back(i);
    }
    grid.base_cells.push_back(std::move(polygon));
    grid.z_edges = {-.37, .12};
    grid.boundary_aliases = {{"zmin", "floor"}, {"zmax", "roof"}, {"side", "curved prism walls"}};
    const auto source = make_hierarchy(grid);
    const auto actual = publish(*source, true);
    const auto& leaf = *source->owned_leaves().begin();
    const auto verified = proof(*actual.bound, leaf.key);
    EXPECT_EQ(verified.source->faces().size(), 14u);
    const auto warm = publish(*actual.bound);
    EXPECT_EQ(proof(*warm.bound, leaf.key).source, verified.source);
    EXPECT_EQ(proof(*warm.bound, leaf.key).incidence, verified.incidence);
}

TEST(NativeProductPublication, RetainedFactorVerifierRejectsForeignEncodedChildKeys)
{
    const auto comm = Tpetra::getDefaultComm();
    ASSERT_EQ(comm->getSize(), 1);
    auto grid = strip(1);
    grid.base_cells.resize(1);
    grid.boundaries.resize(1);
    const auto root = make_hierarchy(grid);
    const auto kernel = D::NativePublicationSession::capture(*root).kernel;
    const auto& parent = *root->owned_leaves().begin();
    for (const auto kind : {A::AdaptiveSplit::Axial, A::AdaptiveSplit::Base})
    {
        auto split = kernel->split(parent, kind);
        ASSERT_GE(split.children.size(), 2u);
        auto& changed = split.children.front();
        const auto changed_key = changed.key;
        const auto valid_key = split.children.back().key;
        auto sibling = std::get<A::ProductCellAncestry>(valid_key.ancestry);
        auto factors = std::make_shared<A::AdaptiveFactorRecipe>(*changed.factor_recipe);
        auto& capsule = kind == A::AdaptiveSplit::Axial ? factors->axis : factors->base;
        if (kind == A::AdaptiveSplit::Axial) sibling.base = {};
        else sibling.axial = {};
        const auto original = A::encode_adaptive_key({0, sibling});
        const auto foreign = A::encode_adaptive_key({parent.key.region, sibling});
        auto malformed = std::make_shared<A::AdaptiveBytes>(*capsule);
        const auto at = std::search(malformed->begin(), malformed->end(), original.begin(), original.end());
        ASSERT_NE(at, malformed->end());
        ASSERT_EQ(std::search(at + original.size(), malformed->end(), original.begin(), original.end()),
            malformed->end());
        ASSERT_EQ(foreign.size(), original.size());
        std::copy(foreign.begin(), foreign.end(), at);
        capsule = std::move(malformed);
        changed.factor_recipe = std::move(factors);
        EXPECT_NO_THROW((void) kernel->geometry(changed));

        // Install the byte-valid record through genuine Product bootstrap. It
        // still names the same selected cell and exact native fragment.
        const A::DistributedAdaptiveHierarchy source(std::move(split.children), {}, kernel, comm);
        D::NativePublicationSession session(D::NativePublicationSession::capture(source));
        ASSERT_TRUE(session.qualified());
        try
        {
            (void) session.source(source.owned_leaves().at(changed_key));
            ADD_FAILURE() << "The retained-factor verifier accepted a foreign encoded child key.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_STREQ(error.what(), "Product source has a foreign encoded child key.");
        }
        EXPECT_NO_THROW((void) session.source(source.owned_leaves().at(valid_key)));
    }
}

TEST(NativeProductPublication, RawFactorChildrenSurviveFlatBodyRetirementUntilLastExplicitBinding)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    using Editor = PLIC::details::GeometryOwnerGraphEditor;
    constexpr auto cap = 64u * 1024 * 1024;
    std::array<std::weak_ptr<const A::AdaptiveBytes>, 3> weak_parts;
    {
        A::ExtrudedAMR factory(strip(1));
        const std::array<std::uint64_t, 1> cells{0};
        auto detached = factory.detach_owned(cells, {'p'});
        const auto& leaf = detached.owned.front();
        const auto factor = leaf.factor_recipe;
        ASSERT_TRUE(factor);
        const std::array parts{factor->root, factor->base, factor->axis};
        std::ranges::copy(parts, weak_parts.begin());
        geometry_detail::NativeConstructionAdmission admission(
            geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(cap));
        const auto binding = D::NativeSourceBinding::capture(leaf, admission);
        const auto flat_body = [&](Editor& graph, bool retain)
        {
            graph.external(factor, sizeof(A::AdaptiveFactorRecipe), retain);
            for (const auto& part : parts) graph.external(part, sizeof(*part) + part->capacity(), retain);
        };
        std::size_t used = 0;
        Editor initial({}, 0, used, cap);
        flat_body(initial, true);
        D::NativeSourceBinding::graph(binding, initial, true);
        const auto accepted = initial.finish();
        for (const auto& part : parts) EXPECT_EQ(accepted.references(0, Editor::external_kind(), part), 3u);

        Editor retire_body(accepted, 0, used, cap);
        flat_body(retire_body, false);
        const auto retained = retire_body.finish();
        for (const auto& part : parts) EXPECT_EQ(retained.references(0, Editor::external_kind(), part), 2u);
        Editor retire_binding(retained, 0, used, cap);
        EXPECT_NO_THROW(D::NativeSourceBinding::graph(binding, retire_binding, false));
        EXPECT_EQ(retire_binding.finish().owner_count(), 0u);
    }
    for (const auto& part : weak_parts) EXPECT_TRUE(part.expired());
}

TEST(NativeProductPublication, BuiltInRepackPreservesExactFactorAndNativeControlsForMovedOrdinals)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    for (const unsigned layers : {4, 8})
    {
        const auto source = make_hierarchy(strip(layers));
        const auto kernel = D::NativePublicationSession::capture(*source).kernel;
        const auto first = publish(*source);
        std::vector<A::AdaptiveLeaf> selected;
        for (const auto& leaf : source->owned_leaves())
            if (std::get<A::ProductCellAncestry>(leaf.key.ancestry).axial.root != 1) selected.push_back(leaf);
        const auto repacked = kernel->repack(selected);
        auto counter = geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(64 * 1024 * 1024);
        const geometry_detail::NativeConstructionAdmission admission(counter);
        std::size_t moved = 0;
        for (const auto& after : repacked)
        {
            const auto& before = source->owned_leaves().at(after.key);
            const auto old_source = proof(*first.bound, after.key).source;
            EXPECT_EQ(after.factor_recipe, before.factor_recipe);
            EXPECT_TRUE(A::product_native_repack_candidate(before, after));
            ASSERT_EQ(after.native_recipe->parts.size(), 4u);
            for (std::size_t part = 0; part < 4; ++part)
                EXPECT_EQ(after.native_recipe->parts[part], before.native_recipe->parts[part]);
            if (after.native_cell == before.native_cell) continue;
            ++moved;
            const auto view = M::NativeFragmentAccess::materialize_product_view(after.native_recipe,
                *before.native_recipe, *old_source->fragment(), admission);
            const auto& r = std::get<M::ExtrudedRegion>(view->region());
            const auto& old = std::get<M::ExtrudedRegion>(old_source->fragment()->region());
            EXPECT_EQ(r.topology().base_storage_identity(), old.topology().base_storage_identity());
            EXPECT_EQ(r.geometry().construction()->base_storage_identity(), old.geometry().construction()->base_storage_identity());
            EXPECT_EQ(r.geometry().construction()->axis_storage_identity(), old.geometry().construction()->axis_storage_identity());
            EXPECT_EQ(r.geometry().construction()->axis_values_storage_identity(),
                old.geometry().construction()->axis_values_storage_identity());
            const auto expected = kernel->geometry(before);
            EXPECT_TRUE(r.geometry().certified_cell_geometry(after.native_cell)->equivalent(*expected));
            same_bits(r.geometry().cell_volume(after.native_cell), old_source->metrics().volume);
            const auto& centroid = old_source->metrics().centroid;
            same_vector(r.geometry().cell_centroid(after.native_cell), {centroid[0], centroid[1], centroid[2]});
        }
        EXPECT_EQ(moved, 2 * (layers - 2));
    }
}

TEST(NativeProductPublication, FixedAxialEditRetainsUnchangedSourcesAndRevalidatesTheirIncidence)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    std::array<std::size_t, 2> changed_counts{};
    std::array<std::size_t, 2> construction_counts{};
    for (std::size_t size = 0; size < changed_counts.size(); ++size)
    {
        const unsigned layers = 4U << size;
        const auto source = make_hierarchy(strip(layers));
        const auto first = publish(*source);
        const auto neighbor = key(1, 0), far = key(1, layers - 1);
        const auto near_before = proof(*first.bound, neighbor), far_before = proof(*first.bound, far);
        D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
        const auto warm = publish(*first.bound);
        EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, 0u);
        EXPECT_EQ(proof(*warm.bound, far).source, far_before.source);

        const std::array marks{A::AdaptiveMark{key(0, 0), A::AdaptiveSplit::Axial}};
        D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
        const auto changed = warm.bound->apply(warm.bound->plan_adaptation(marks));
        RecordProperty(layers == 4 ? "planning_4_layers_constructions" : "planning_8_layers_constructions",
            std::to_string(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions));
        D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
        const auto next = publish(*changed);
        construction_counts[size] = D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions;
        RecordProperty(layers == 4 ? "fixed_edit_4_layers_constructions" : "fixed_edit_8_layers_constructions",
            std::to_string(construction_counts[size]));
        compare_with_oracle(next.view, A::assemble_adaptive_native(*changed, changed->configuration_descriptor()));
        EXPECT_EQ(proof(*next.bound, neighbor).source, near_before.source);
        EXPECT_NE(proof(*next.bound, neighbor).incidence, near_before.incidence);
        EXPECT_EQ(proof(*next.bound, far).source, far_before.source);
        for (const auto& leaf : next.bound->owned_leaves())
        {
            const auto current = proof(*next.bound, leaf.key).source;
            if (!first.bound->owned_leaves().contains(leaf.key))
                ++changed_counts[size];
            else
                EXPECT_EQ(current, proof(*first.bound, leaf.key).source);
        }
    }
    EXPECT_EQ(changed_counts[0], 2u);
    EXPECT_EQ(changed_counts[1], changed_counts[0]);
    EXPECT_GT(construction_counts[0], 0u);
    EXPECT_EQ(construction_counts[1], construction_counts[0]);
}

TEST(NativeProductPublicationMPI, RemoteReuseChangedIncidenceMigrationAndEmptyOwners)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank();
    ASSERT_TRUE(comm->getSize() == 2 || comm->getSize() == 4);
    const auto hierarchy = make_hierarchy(strip(4), true);
    const auto first = publish(*hierarchy, true);
    const auto warm = publish(*first.bound);
    const auto coarse_neighbor = key(1, 0);
    if (rank < 2)
    {
        EXPECT_EQ(proof(*first.bound, coarse_neighbor).source, proof(*warm.bound, coarse_neighbor).source);
        EXPECT_EQ(proof(*first.bound, coarse_neighbor).incidence, proof(*warm.bound, coarse_neighbor).incidence);
    }

    std::vector<A::AdaptiveMark> marks;
    if (rank == 0) marks.push_back({key(0, 0), A::AdaptiveSplit::Axial});
    const auto adapted = warm.bound->apply(warm.bound->plan_adaptation(marks));
    const auto refined = publish(*adapted, true);
    if (rank < 2)
    {
        EXPECT_EQ(proof(*warm.bound, coarse_neighbor).source, proof(*refined.bound, coarse_neighbor).source);
        EXPECT_NE(proof(*warm.bound, coarse_neighbor).incidence, proof(*refined.bound, coarse_neighbor).incidence);
    }

    const auto moving = key(0, 1);
    std::vector<int> destinations;
    for (const auto& leaf : refined.bound->owned_leaves()) destinations.push_back(leaf.key == moving ? 1 : rank);
    const auto migration = refined.bound->plan_migration(destinations);
    std::uint64_t sent = migration.native_migration().remote_sent_packets, global_sent = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &sent, &global_sent);
    EXPECT_GT(global_sent, 0u);
    const auto migrated = refined.bound->apply(migration);
    const auto moved = publish(*migrated, true);
    const int empty = moved.bound->owned_leaves().empty(), expected_empty = comm->getSize() - 2;
    int global_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &global_empty);
    EXPECT_EQ(global_empty, expected_empty);
    const auto owner = moved.bound->lookup_cells(std::span(&moving, 1));
    ASSERT_EQ(owner.size(), 1u);
    EXPECT_EQ(owner[0].owner, 1);
    EXPECT_EQ(moved.view.owned_gids().contains(moving), rank == 1);
    if (rank == 1) EXPECT_EQ(proof(*moved.bound, moving).source->ticket().issuer, rank);
}

TEST(NativeProductPublicationMPI, IndependentLineagesWithMatchingLocalPlansRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    ASSERT_TRUE(comm->getSize() == 2 || comm->getSize() == 4);
    const auto first = make_hierarchy(strip(2), true);
    const auto second = make_hierarchy(strip(2), true);
    const auto first_plan = first->plan_adaptation({});
    const auto second_plan = second->plan_adaptation({});
    ASSERT_EQ(first_plan.collective_identity(), second_plan.collective_identity());
    const auto& mixed = comm->getRank() == 0 ? *first : *second;
    const auto& mixed_plan = comm->getRank() == 0 ? first_plan : second_plan;
    const auto observation = mixed.observation();
    EXPECT_THROW((void) mixed.apply(mixed_plan), std::invalid_argument);
    EXPECT_THROW((void) mixed.plan_adaptation({}), std::invalid_argument);
    const std::vector<int> destinations(mixed.owned_leaves().size(), comm->getRank());
    EXPECT_THROW((void) mixed.plan_migration(destinations), std::invalid_argument);
    EXPECT_THROW((void) A::publish_adaptive_native(mixed, A::native_publication_schema(mixed)), std::invalid_argument);
    EXPECT_THROW((void) A::publish_adaptive_native(mixed, mixed.configuration_descriptor()), std::invalid_argument);
    EXPECT_EQ(mixed.observation(), observation);
    EXPECT_EQ(first->apply(first_plan)->observation(), first->observation());
    EXPECT_EQ(second->apply(second_plan)->observation(), second->observation());
    EXPECT_NO_THROW((void) publish(*first));
}
} // namespace
