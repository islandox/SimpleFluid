/** Detached canonical pieces compared against the established authority workflow. */
#include "adaptive_storage_test_helpers.hh"
#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/AdaptiveSubfaceSnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/PooledTileCodec.hh"
#include "geometry/mesh/AMR/TileBodyPages.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <set>

namespace
{
using namespace cartesian_incidence_test;
namespace AMR = SimpleFluid::Meshes::AMR;
constexpr auto ordinary = ArithmeticMode::Ordinary;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Comm = Teuchos::RCP<const Teuchos::Comm<int>>;

std::vector<std::uint64_t> selected(std::size_t count, const Comm& comm, bool empty_remote = false)
{
    std::vector<std::uint64_t> rows;
    for (std::uint64_t slot = 0; slot < count; ++slot)
        if (int(empty_remote ? 0 : slot % comm->getSize()) == comm->getRank())
            rows.push_back(slot);
    return rows;
}
AdaptiveBootstrap cartesian(const Comm& comm)
{
    auto config = grid();
    config.periodic = {true, false, true};
    const CartesianAMR mesh(config);
    return mesh.detach_owned(selected(mesh.geometry()->num_cells(), comm), {'c', 'a', 'r', 't'});
}
AdaptiveBootstrap product(const Comm& comm)
{
    const ExtrudedAMR mesh(SimpleFluid::test::mixed_product_grid(true));
    return mesh.detach_owned(selected(mesh.geometry()->num_cells(), comm), {'p', 'r', 'o', 'd'});
}
AdaptiveBootstrap polyhedral(const Comm& comm)
{
    const auto mesh = SimpleFluid::test::mixed_polyhedron()->adapt({0});
    return mesh->detach_owned(selected(mesh->geometry()->num_cells(), comm), {'p', 'o', 'l', 'y'});
}
AdaptiveBootstrap stationary(const Comm& comm)
{
    const CartesianAMR mesh(grid());
    return detach_static_participant(mesh.geometry()->regions().front(), 71, 9, 5,
        selected(mesh.geometry()->num_cells(), comm, true), {'s', 't', 'a', 't'});
}
AdaptiveBootstrap mixed(const Comm& comm)
{
    auto config = SimpleFluid::test::mixed_joined_grid();
    const auto key = std::get<ProductCellAncestry>(config.product->cell_key(0, 0).ancestry);
    config.product = config.product->adapt({key}, {key});
    config.adaptive.front() = std::dynamic_pointer_cast<const PolyhedralAMR>(config.adaptive.front())->adapt({0});
    const MixedAMR mesh(std::move(config));
    return mesh.detach_owned(selected(mesh.geometry()->num_cells(), comm), {'m', 'i', 'x', 'e', 'd'});
}
void release_rows(std::vector<IncidenceRowDelivery>& rows, Budget& budget)
{
    const auto bytes = rows.capacity() * sizeof(IncidenceRowDelivery);
    std::vector<IncidenceRowDelivery>().swap(rows);
    budget.used -= bytes;
}
} // namespace

TEST(AdaptiveSubfaceSnapshot, AuthenticatedPiecesMatchLegacyAuthorityForEveryProvider)
{
    const auto comm = Tpetra::getDefaultComm();
    const std::array fixtures{std::pair{"Cartesian", &cartesian}, std::pair{"Product", &product},
        std::pair{"Polyhedral", &polyhedral}, std::pair{"Static", &stationary}, std::pair{"Mixed", &mixed}};
    for (const auto& [name, factory] : fixtures)
    {
        SCOPED_TRACE(name);
        auto input = factory(comm);
        std::map<AdaptiveCellKey, const AdaptiveLeaf*> lookup;
        for (const auto& row : input.owned)
            lookup.emplace(row.key, &row);
        auto materialized = input.owned;
        for (auto& row : materialized)
            row.surfaces = input.kernel->surfaces(row);
        std::vector<AdaptiveNode> legacy_nodes;
        AdaptiveAuthorityStats legacy_stats;
        const auto old = decide_adaptive_tiles(materialized, comm, ample, 65536, &legacy_stats, &legacy_nodes);
        std::map<AdaptiveCellKey, const AdaptiveLeaf*> publication_rows;
        for (const auto& row : materialized)
            publication_rows.emplace(row.key, &row);
        std::map<AdaptiveBytes, const AdaptiveTile*> expected;
        for (const auto& tile : old)
            EXPECT_TRUE(expected.emplace(tile.identity, &tile).second);

        auto arena = std::make_shared<Budget>(Budget{ample});
        std::shared_ptr<const AdaptiveSubfaceSnapshot> snapshot;
        CanonicalSubfaceView escaped;
        std::optional<ContactPointIdentity> saved_point;
        {
            IdentityBudget identities;
            DistributedAdaptiveIncidenceSession session(*input.kernel, comm, {}, {}, *arena, identities, arena);
            auto rows =
                session.bootstrap(input.owned, [&](const auto& key) -> const AdaptiveLeaf& { return *lookup.at(key); });
            release_rows(rows, *arena);
            snapshot = session.subface_snapshot();
            EXPECT_TRUE(snapshot);
            if (!snapshot)
                continue;
            EXPECT_EQ(snapshot->context(), ContactHaloContext{});
            EXPECT_EQ(snapshot->size(), old.size());
            AdaptiveAuthorityStats snapshot_stats;
            std::vector<AdaptiveNode> numbered_nodes;
            const auto numbered = publish_adaptive_subfaces(
                *snapshot, [&](const auto& key) -> const AdaptiveLeaf& { return *publication_rows.at(key); }, comm,
                ample, 65536, &snapshot_stats, &numbered_nodes);
            EXPECT_EQ(snapshot_stats.global_faces, legacy_stats.global_faces);
            EXPECT_EQ(snapshot_stats.global_nodes, legacy_stats.global_nodes);
            EXPECT_EQ(numbered.size(), old.size());
            for (const auto& tile : numbered)
            {
                const auto match = expected.find(tile.identity);
                EXPECT_NE(match, expected.end());
                if (match == expected.end())
                    continue;
                // Full wire parity covers face GID, owner receipts, native
                // ordinals, both periodic node loops and exact coordinates.
                EXPECT_EQ(encode_adaptive_tile(tile, ample), encode_adaptive_tile(*match->second, ample));
            }
            const auto by_gid = [](const AdaptiveNode& a, const AdaptiveNode& b) { return a.gid < b.gid; };
            std::sort(legacy_nodes.begin(), legacy_nodes.end(), by_gid);
            std::sort(numbered_nodes.begin(), numbered_nodes.end(), by_gid);
            EXPECT_EQ(numbered_nodes.size(), legacy_nodes.size());
            for (std::size_t i = 0; i < std::min(legacy_nodes.size(), numbered_nodes.size()); ++i)
            {
                EXPECT_EQ(numbered_nodes[i].gid, legacy_nodes[i].gid);
                EXPECT_EQ(numbered_nodes[i].identity, legacy_nodes[i].identity);
                EXPECT_EQ(numbered_nodes[i].coordinate_identity, legacy_nodes[i].coordinate_identity);
            }
            snapshot->visit_pieces(
                [&](const auto& view)
                {
                    if (!escaped)
                    {
                        escaped = view;
                        saved_point = view.point(0);
                    }
                    Budget output{ample};
                    IdentityBudget retained;
                    const auto piece = view.materialize(output, retained);
                    const auto found = expected.find(piece.identity);
                    ASSERT_NE(found, expected.end());
                    const auto& reference = *found->second;
                    expected.erase(found);
                    EXPECT_EQ(view.first().key(), reference.first);
                    EXPECT_EQ(view.first().owner(), reference.first_owner);
                    ASSERT_EQ(bool(view.second()), bool(reference.second));
                    if (const auto second = view.second())
                    {
                        EXPECT_EQ(second->key(), *reference.second);
                        EXPECT_EQ(second->owner(), reference.second_owner);
                        if (second->owner() == comm->getRank())
                        {
                            EXPECT_EQ(second->revision(), lookup.at(second->key())->revision);
                            EXPECT_EQ(second->native_face().value_or(std::numeric_limits<std::uint64_t>::max()),
                                reference.second_native_face);
                        }
                        else
                            EXPECT_FALSE(second->native_face());
                    }
                    if (view.first().owner() == comm->getRank())
                    {
                        EXPECT_EQ(view.first().revision(), lookup.at(view.first().key())->revision);
                        EXPECT_EQ(view.first().native_face().value_or(std::numeric_limits<std::uint64_t>::max()),
                            reference.first_native_face);
                    }
                    else
                        EXPECT_FALSE(view.first().native_face());
                    EXPECT_EQ(view.second_image(), reference.second_image_identity);
                    EXPECT_EQ(view.boundary(), reference.boundary);
                    EXPECT_TRUE(std::ranges::equal(view.support(), reference.surface));
                    EXPECT_TRUE(std::ranges::equal(view.domain(), reference.domain));
                    const auto legacy_points = reference.geometry->vertex_identities();
                    const auto restored_points = piece.geometry->vertex_identities();
                    ASSERT_EQ(view.point_count(), legacy_points.size());
                    EXPECT_EQ(restored_points, legacy_points);
                    for (std::size_t slot = 0; slot < view.point_count(); ++slot)
                        EXPECT_EQ(view.point(slot), legacy_points[slot]);
                });
            EXPECT_TRUE(expected.empty());
            for (const auto& leaf : materialized)
                for (std::size_t surface = 0; surface < leaf.surfaces.size(); ++surface)
                {
                    std::size_t count = 0;
                    snapshot->visit_face_pieces(leaf.key, surface,
                        [&](const auto& view)
                        {
                            ++count;
                            const auto incident = view.outward_sign() > 0 ? view.first() : *view.second();
                            EXPECT_EQ(incident.key(), leaf.key);
                            EXPECT_EQ(incident.source_surface(), surface);
                            EXPECT_EQ(incident.owner(), comm->getRank());
                        });
                    EXPECT_GT(count, 0u);
                }
        }
        snapshot.reset();
        arena.reset();
        if (escaped)
            EXPECT_EQ(escaped.point(0), *saved_point);
    }
}

TEST(AdaptiveSubfaceSnapshot, LongBoundaryNameFitsExactPublicationBudget)
{
    const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    auto config = SimpleFluid::test::mixed_product_grid();
    const std::string boundary = "boundary_16chars";
    config.boundary_aliases.emplace("south", boundary);
    const ExtrudedAMR mesh(config);
    auto input = mesh.detach_owned(std::array<std::uint64_t, 1>{0}, {'b'});
    ASSERT_EQ(input.owned.size(), 1u);
    auto& leaf = input.owned.front();
    const auto lookup = [&](const auto& key) -> const AdaptiveLeaf&
    {
        if (key != leaf.key)
            throw std::invalid_argument("Unexpected long-boundary fixture key.");
        return leaf;
    };
    auto arena = std::make_shared<Budget>(Budget{ample});
    IdentityBudget identities;
    DistributedAdaptiveIncidenceSession session(*input.kernel, comm, {}, {}, *arena, identities, arena);
    auto rows = session.bootstrap(input.owned, lookup);
    release_rows(rows, *arena);
    const auto snapshot = session.subface_snapshot();
    ASSERT_TRUE(snapshot);
    leaf.surfaces = input.kernel->surfaces(leaf);
    const auto number = [&](std::size_t cap) { return publish_adaptive_subfaces(*snapshot, lookup, comm, cap, 64); };
    const auto expected = number(ample);
    ASSERT_EQ(expected.size(), 6u);
    EXPECT_EQ(std::count_if(expected.begin(), expected.end(),
                  [&](const AdaptiveTile& tile) { return tile.boundary == boundary; }),
        1);
    std::size_t lower = 1, upper = ample;
    while (lower < upper)
    {
        const auto middle = lower + (upper - lower) / 2;
        try
        {
            (void) number(middle);
            upper = middle;
        }
        catch (const std::invalid_argument&)
        {
            lower = middle + 1;
        }
    }
    ASSERT_GT(lower, 1u);
    EXPECT_THROW((void) number(lower - 1), std::invalid_argument);
    const auto retry = number(lower);
    ASSERT_EQ(retry.size(), expected.size());
    for (std::size_t i = 0; i < retry.size(); ++i)
        EXPECT_EQ(encode_adaptive_tile(retry[i], ample), encode_adaptive_tile(expected[i], ample));
}

TEST(AdaptiveSubfaceSnapshot, RepackedProductUsesCurrentNativeFaceAddressesAfterMigration)
{
    const auto comm = Tpetra::getDefaultComm();
    auto config = SimpleFluid::test::mixed_product_grid();
    config.z_edges = {0, .5, 1};
    const ExtrudedAMR mesh(config);
    auto input = mesh.detach_owned(selected(mesh.geometry()->num_cells(), comm, true), {'r'});
    auto source = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(input.owned), std::move(input.families), std::move(input.kernel), comm);
    std::vector<AdaptiveMark> marks;
    for (const auto& leaf : source->owned_leaves())
        marks.push_back({leaf.key, AdaptiveSplit::Axial});
    const auto refined = source->apply(source->plan_adaptation(marks));
    EXPECT_EQ(refined->global_cells(), 4u);

    const auto completed = AMR::adaptive_detail::subface_snapshot(*refined);
    int stale = 0;
    completed->visit_pieces(
        [&](const auto& piece)
        {
            const auto inspect = [&](const CanonicalSubfaceIncident& incident)
            {
                if (incident.owner() != comm->getRank())
                    return;
                const auto& leaf = refined->owned_leaves().at(incident.key());
                const auto faces = refined->surfaces(leaf);
                if (incident.source_surface() < faces.size() && incident.native_face() &&
                    faces[incident.source_surface()].native_face != *incident.native_face())
                    stale = 1;
            };
            inspect(piece.first());
            if (const auto second = piece.second())
                inspect(*second);
        });
    int any_stale = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &stale, &any_stale);
    EXPECT_EQ(any_stale, 1);

    const auto compare = [&](const AMR::DistributedAdaptiveHierarchy& hierarchy)
    {
        const auto draft = hierarchy.native_draft();
        AdaptiveAuthorityStats stats;
        std::vector<AdaptiveNode> nodes;
        const auto legacy = decide_adaptive_tiles(draft.owned, comm, ample, 65536, &stats, &nodes);
        EXPECT_EQ(draft.global_faces, stats.global_faces);
        EXPECT_EQ(draft.global_nodes, stats.global_nodes);
        EXPECT_EQ(draft.tiles.size(), legacy.size());
        std::map<AdaptiveBytes, const AdaptiveTile*> by_identity;
        for (const auto& tile : legacy)
            by_identity.emplace(tile.identity, &tile);
        for (const auto& tile : draft.tiles)
        {
            const auto match = by_identity.find(tile.identity);
            EXPECT_NE(match, by_identity.end());
            if (match != by_identity.end())
                EXPECT_EQ(encode_adaptive_tile(tile, ample), encode_adaptive_tile(*match->second, ample));
        }
    };
    compare(*refined);
    std::vector<int> destinations(refined->owned_leaves().size(), comm->getRank());
    if (!destinations.empty())
        destinations.back() = comm->getSize() - 1;
    const auto moved = refined->apply(refined->plan_migration(destinations));
    EXPECT_EQ(moved->partition_generation(), refined->partition_generation() + (comm->getSize() > 1));
    if (comm->getSize() == 1)
        EXPECT_EQ(moved->leaf_layout(), refined->leaf_layout());
    compare(*moved);
}

TEST(AdaptiveSubfaceSnapshot, DirtyReplaySharesUnchangedPagesAndPreservesOldViews)
{
    const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    const auto config = grid();
    const auto kernel = cartesian_adaptive_kernel(config);
    const auto initial = leaves(config);
    std::map<AdaptiveCellKey, AdaptiveLeaf> active;
    for (const auto& leaf : initial)
        active.emplace(leaf.key, leaf);
    const auto lookup = [&](const auto& key) -> const AdaptiveLeaf& { return active.at(key); };
    auto source_arena = std::make_shared<Budget>(Budget{ample});
    auto candidate_arena = std::make_shared<Budget>(Budget{ample});
    IdentityBudget source_identities, candidate_identities;
    CanonicalSubfaceView historical;
    std::optional<ContactPointIdentity> saved;
    std::set<const void*> source_pages;
    std::shared_ptr<const AdaptiveSubfaceSnapshot> candidate_snapshot;
    {
        DistributedAdaptiveIncidenceSession source(
            *kernel, comm, {}, {}, *source_arena, source_identities, source_arena);
        auto rows = source.bootstrap(initial, lookup);
        const auto target = initial.front().key;
        const auto old_row =
            std::find_if(rows.begin(), rows.end(), [&](const auto& row) { return row.leaf->key == target; });
        ASSERT_NE(old_row, rows.end());
        const std::array retired{
            IncidenceCellVersion{target, old_row->leaf->revision, old_row->row->incidence_generation}};
        release_rows(rows, *source_arena);
        const auto& old = source.subface_snapshot();
        ASSERT_TRUE(old);
        old->visit_pieces(
            [&](const auto& view)
            {
                source_pages.insert(view.page_identity());
                if (!historical && (view.first().key() == target || (view.second() && view.second()->key() == target)))
                {
                    historical = view;
                    saved = view.point(0);
                }
            });
        ASSERT_TRUE(historical);
        auto candidate = source.fork(*candidate_arena, candidate_identities, candidate_arena);
        auto split = kernel->split(active.at(target), AdaptiveSplit::Isotropic);
        active.erase(target);
        std::vector<AdaptiveCellKey> inserted;
        for (auto& child : split.children)
        {
            inserted.push_back(child.key);
            active.emplace(child.key, std::move(child));
        }
        auto delta = candidate->apply_delta(retired, inserted, ContactHaloContext{0, 1, 0, 1, 1}, lookup);
        release_rows(delta.changed, *candidate_arena);
        candidate_snapshot = candidate->subface_snapshot();
        ASSERT_TRUE(candidate_snapshot);
        std::size_t shared = 0;
        candidate_snapshot->visit_pieces(
            [&](const auto& view) { shared += source_pages.contains(view.page_identity()); });
        EXPECT_GT(shared, 0u);
        EXPECT_NE(candidate_snapshot, old);
        // The accepted source still pins every retired metadata page until
        // the overlap is finished, including pages the candidate replaced.
        const auto retired_overlap = candidate->inherited_overlap_metadata_bytes();
        EXPECT_GT(retired_overlap, 0u);
        const auto before_finish = candidate_arena->used;
        candidate->finish_inherited_overlap();
        EXPECT_EQ(candidate_arena->used + retired_overlap, before_finish);
        EXPECT_EQ(candidate->inherited_overlap_metadata_bytes(), 0u);
        EXPECT_EQ(historical.point(0), *saved);
    }
    source_arena.reset();
    candidate_arena.reset();
    candidate_snapshot.reset();
    EXPECT_EQ(historical.point(0), *saved);
}

TEST(AdaptiveSubfaceSnapshot, DeepSupportMaterializationAdmitsExactPeakAndRejectsOneByteShort)
{
    using Face = PLIC::PlanarFace<ordinary>;
    const std::array<PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    auto face = Face::make(vertices);
    for (unsigned depth = 0; depth < 96; ++depth)
        face = face->translated({1. / 1024, -1. / 2048, 1. / 4096});
    auto arena = std::make_shared<Budget>(Budget{ample});
    IdentityBudget identities;
    IncidencePools pools(*arena, identities, arena);
    TileBodyPages bodies(*arena, arena);
    AdaptiveLeaf leaf;
    leaf.key = {0, StaticCellAncestry{71, 1}};
    leaf.domain = {'d'};
    AdaptiveSurface surface;
    surface.key = {'s'};
    surface.geometry = face;
    surface.exterior = true;
    surface.boundary = "wall";
    leaf.surfaces.push_back(std::move(surface));
    const ReceivedTileRecipe receipt{
        leaf.surfaces.front().key, leaf.domain, {leaf.key, 0, 0, 0, 0}, std::nullopt, face, {}, 0};
    const auto body = bodies.append(freeze_tile(receipt, 0, pools));
    bodies.freeze();
    SubfaceProvenanceBuilder provenance(pools, arena, 0);
    const auto source = provenance.append(receipt, 0, [&](const auto&) -> const AdaptiveLeaf& { return leaf; });
    provenance.freeze();
    SubfacePieceBuilder builder(pools, arena);
    const auto view = builder.append(source, body, {}, nullptr);
    builder.freeze();

    Budget measured{ample};
    IdentityBudget measured_ids;
    const auto expected = view.materialize(measured, measured_ids);
    EXPECT_EQ(expected.geometry->encode_incidence(), face->encode_incidence());
    EXPECT_GE(measured.peak, 6 * face->encoded_incidence_size() + face->encoding_scratch_bytes());
    ASSERT_GT(measured.peak, 0u);
    Budget exact{measured.peak};
    IdentityBudget exact_ids;
    EXPECT_EQ(view.materialize(exact, exact_ids).identity, expected.identity);
    EXPECT_EQ(exact.peak, measured.peak);
    Budget one_short{measured.peak - 1};
    IdentityBudget short_ids;
    EXPECT_THROW((void) view.materialize(one_short, short_ids), std::exception);
}

TEST(AdaptiveSubfaceSnapshot, ThreeGenerationsAdmitSharedPartialPointCreditsExactlyOnce)
{
    const auto config = grid();
    const auto provider = cartesian_adaptive_kernel(config);
    class ExplicitProvider final : public AdaptiveProviderKernel
    {
    public:
        explicit ExplicitProvider(const AdaptiveProviderKernel& value) : inner(value) {}
        AdaptiveBytes descriptor() const override { return inner.descriptor(); }
        PLIC::ConvexGeometryPtr<ordinary> geometry(const AdaptiveLeaf& leaf) const override
        {
            return inner.geometry(leaf);
        }
        std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override { return inner.surfaces(leaf); }
        AdaptiveSplitResult split(const AdaptiveLeaf& leaf, AdaptiveSplit how) const override
        {
            return inner.split(leaf, how);
        }
        AdaptiveTrace traces(const AdaptiveLeaf& leaf) const override { return inner.traces(leaf); }
        AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf& leaf) const override
        {
            return inner.contact_capabilities(leaf);
        }
        std::vector<AdaptiveContactImage> closed_contact_images(const AdaptiveLeaf& leaf) const override
        {
            return inner.closed_contact_images(leaf);
        }
        ContactPointIdentity contact_point(const AdaptiveLeaf& leaf, const ContactPointIdentity& point) const override
        {
            return inner.contact_point(leaf, point);
        }

    private:
        const AdaptiveProviderKernel& inner;
    } explicit_provider(*provider);
    const auto cells = leaves(config);
    const auto lookup = [&](const auto& key) -> const AdaptiveLeaf&
    {
        const auto found = std::find_if(cells.begin(), cells.end(), [&](const auto& leaf) { return leaf.key == key; });
        if (found == cells.end())
            throw std::invalid_argument("Missing explicit test leaf.");
        return *found;
    };
    const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    DistributedAdaptiveContactHalo first(explicit_provider, comm, {}, ContactHaloContext{0, 0, 0, 0, 1});
    first.bootstrap_canonical(cells, lookup);
    EXPECT_TRUE(first.required_refinements().empty());
    ASSERT_TRUE(first.subface_snapshot());
    std::map<const void*, std::size_t> subface_owners;
    first.subface_snapshot()->visit_owners(
        [&](const auto& owner, const Budget*, std::size_t bytes) { subface_owners.emplace(owner.get(), bytes); });
    std::map<const void*, std::size_t> shared;
    bool partial = false;
    first.for_each_borrowed_contact_binding(
        [&](const auto& binding)
        {
            const auto* page = binding.row ? binding.row->points.borrowed_page_view() : nullptr;
            if (page)
                page->visit_borrowed_pages(
                    [&](const auto& record)
                    {
                        if (const auto found = subface_owners.find(record.credit_owner); found != subface_owners.end())
                        {
                            shared.emplace(*found);
                            partial |= record.records.size() < record.record_capacity;
                        }
                    });
        });
    ASSERT_FALSE(shared.empty());
    ASSERT_TRUE(partial);

    auto second = first.fork_candidate(ContactHaloContext{0, 0, 0, 0, 2});
    second->bind_external_retained_bytes(0);
    EXPECT_TRUE(second->required_refinements().empty());
    auto third = second->fork_candidate(ContactHaloContext{0, 0, 0, 0, 3});
    third->bind_external_retained_bytes(0);
    for (const auto* halo : {second.get(), third.get()})
    {
        std::map<const void*, std::vector<std::size_t>> admitted;
        halo->for_each_borrowed_source_admission(
            [&](const void* owner, std::size_t bytes) { admitted[owner].push_back(bytes); });
        for (const auto& [owner, bytes] : shared)
            EXPECT_EQ(admitted[owner], std::vector{bytes});
    }
}

TEST(AdaptiveSubfaceSnapshot, SmallEditsCopyOnlyAffectedPagesForEveryProvider)
{
    namespace fixtures = SimpleFluid::test::adaptive_storage;
    using AllocationKey = std::pair<const void*, ContactPayloadRole>;
    const auto index_allocations = [](const AdaptiveSubfaceSnapshot& snapshot)
    {
        std::map<AllocationKey, std::size_t> result;
        snapshot.visit_borrowed_payload(
            [&](const ContactPayloadAllocation& value)
            {
                using Role = ContactPayloadRole;
                const auto role = value.identity.role;
                if (role == Role::SubfaceIndexPageObject || role == Role::SubfaceIndexChildren ||
                    role == Role::SubfacePieces || role == Role::SubfaceSelectors ||
                    role == Role::SubfacePieceOwnerIndexRows || role == Role::SubfaceProvenanceOwnerIndexRows)
                    result.emplace(AllocationKey{value.identity.address, role}, value.payload_bytes);
            });
        return result;
    };
    struct Work
    {
        std::size_t pages = 0, bytes = 0, retained_pages = 0, retained_bytes = 0, probes = 0;
    };
    for (const auto provider : fixtures::providers)
    {
        SCOPED_TRACE(fixtures::name(provider));
        std::array<Work, 2> work;
        const std::size_t components = provider == fixtures::Provider::Cartesian ? 12 : 24;
        for (std::size_t scale = 0; scale < work.size(); ++scale)
        {
            SCOPED_TRACE(scale);
            fixtures::DisconnectedFixture fixture;
            ASSERT_NO_THROW(fixture = fixtures::make_disconnected(provider, components << scale));
            const auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
            {
                const auto found = std::lower_bound(fixture.owned.begin(), fixture.owned.end(), key,
                    [](const auto& row, const auto& wanted) { return row.key < wanted; });
                if (found == fixture.owned.end() || found->key != key)
                    throw std::invalid_argument("Missing disconnected subface fixture leaf.");
                return *found;
            };
            const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
            auto source_arena = std::make_shared<Budget>(Budget{ample});
            auto next_arena = std::make_shared<Budget>(Budget{ample});
            IdentityBudget source_ids, next_ids;
            DistributedAdaptiveIncidenceSession source(
                *fixture.kernel, comm, {}, ContactHaloContext{}, *source_arena, source_ids, source_arena);
            std::vector<IncidenceRowDelivery> rows;
            ASSERT_NO_THROW(rows = source.bootstrap(fixture.owned, lookup));
            const auto old_row = std::find_if(
                rows.begin(), rows.end(), [&](const auto& row) { return row.leaf->key == fixture.changed; });
            ASSERT_NE(old_row, rows.end());
            const auto old_revision = old_row->leaf->revision;
            const std::array retired{
                IncidenceCellVersion{fixture.changed, old_revision, old_row->row->incidence_generation}};
            release_rows(rows, *source_arena);
            const auto old = source.subface_snapshot();
            ASSERT_TRUE(old);
            const auto old_index = index_allocations(*old);
            CanonicalSubfaceView historical, distant;
            old->visit_face_pieces(fixture.changed, 0, [&](const auto& view) { historical = view; });
            const auto far_key = fixture.owned.back().key;
            old->visit_face_pieces(far_key, 0, [&](const auto& view) { distant = view; });
            ASSERT_TRUE(historical);
            ASSERT_TRUE(distant);
            const auto point = historical.point(0);
            const auto revision_of = [](const auto& view)
            { return view.outward_sign() > 0 ? view.first().revision() : view.second()->revision(); };

            auto next = source.fork(*next_arena, next_ids, next_arena);
            const auto changed = std::find_if(fixture.owned.begin(), fixture.owned.end(),
                [&](const auto& row) { return row.key == fixture.changed; });
            ASSERT_NE(changed, fixture.owned.end());
            ++changed->revision;
            const std::array inserted{fixture.changed};
            IncidenceDelta delta;
            ASSERT_NO_THROW(delta = next->apply_delta(retired, inserted, ContactHaloContext{0, 1, 0, 1, 1}, lookup));
            release_rows(delta.changed, *next_arena);
            const auto current = next->subface_snapshot();
            ASSERT_TRUE(current);
            EXPECT_EQ(current->size(), old->size());
            EXPECT_EQ(revision_of(historical), old_revision);
            EXPECT_EQ(historical.point(0), point);
            std::size_t changed_pieces = 0;
            current->visit_face_pieces(fixture.changed, 0,
                [&](const auto& view)
                {
                    ++changed_pieces;
                    EXPECT_EQ(revision_of(view), old_revision + 1);
                });
            EXPECT_GT(changed_pieces, 0u);
            bool distant_shared = false;
            current->visit_face_pieces(far_key, 0,
                [&](const auto& view) { distant_shared |= view.page_identity() == distant.page_identity(); });
            EXPECT_TRUE(distant_shared);
            const auto next_index = index_allocations(*current);
            for (const auto& [key, bytes] : next_index)
                if (!old_index.contains(key))
                {
                    work[scale].retained_pages += key.second == ContactPayloadRole::SubfaceIndexPageObject;
                    work[scale].retained_bytes += bytes;
                }
            const auto changes = current->index_work();
            work[scale].pages = changes.copied_pages;
            work[scale].bytes = changes.copied_bytes;
            work[scale].probes = changes.owner_probes;
            EXPECT_GT(changes.copied_pages, 0u);
            EXPECT_LT(work[scale].retained_pages,
                std::count_if(next_index.begin(), next_index.end(), [](const auto& value)
                    { return value.first.second == ContactPayloadRole::SubfaceIndexPageObject; }));

            auto third_arena = std::make_shared<Budget>(Budget{ample});
            IdentityBudget third_ids;
            auto third = next->fork(*third_arena, third_ids, third_arena);
            EXPECT_EQ(third->subface_snapshot(), current);
            next.reset();
            EXPECT_EQ(index_allocations(*third->subface_snapshot()), next_index);
        }
        EXPECT_EQ(work[0].pages, work[1].pages);
        EXPECT_EQ(work[0].bytes, work[1].bytes);
        // Physical surviving nodes may be fewer than path copies when
        // several edited selectors share a path. Both are bounded by the
        // same actual copied allocation budget, independent of far cells.
        EXPECT_LE(work[1].retained_pages, work[0].pages);
        EXPECT_LE(work[1].retained_bytes, work[0].bytes);
        EXPECT_LE(work[0].probes, 8 * work[0].pages);
        EXPECT_LE(work[1].probes, 8 * work[0].pages);
    }
}

TEST(AdaptiveSubfaceSnapshot, PersistentSelectorEditAdmitsExactPeakAndRejectsOneByteShort)
{
    using Face = PLIC::PlanarFace<ordinary>;
    const std::array<PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    const auto geometry = Face::make(vertices);
    auto arena = std::make_shared<Budget>(Budget{ample});
    IdentityBudget identities;
    IncidencePools pools(*arena, identities, arena);
    TileBodyPages bodies(*arena, arena);
    AdaptiveLeaf leaf;
    leaf.key = {0, StaticCellAncestry{71, 1}};
    leaf.domain = {'d'};
    AdaptiveSurface surface;
    surface.key = {'s'};
    surface.geometry = geometry;
    surface.exterior = true;
    leaf.surfaces.push_back(std::move(surface));
    ReceivedTileRecipe receipt{
        leaf.surfaces.front().key, leaf.domain, {leaf.key, 0, 0, 0, 0}, std::nullopt, geometry, {}, 0};
    const auto body = bodies.append(freeze_tile(receipt, 0, pools));
    bodies.freeze();
    const auto build = [&]
    {
        SubfaceProvenanceBuilder provenance(pools, arena, 0);
        const auto source =
            provenance.append(receipt, leaf.revision, [&](const auto&) -> const AdaptiveLeaf& { return leaf; });
        provenance.freeze();
        SubfacePieceBuilder builder(pools, arena);
        auto view = builder.append(source, body, {}, nullptr);
        builder.freeze();
        return view;
    };
    const auto before = build();
    ++leaf.revision;
    ++receipt.first.revision;
    const auto after = build();
    const auto controls = geometry_construction_controls();
    const auto accepted = AdaptiveSubfaceSnapshot::seal({}, controls, 0, std::array{before}, *arena, arena);
    const std::array edits{CanonicalSubfaceChange{before, after}};
    const auto derive = [&]
    {
        return AdaptiveSubfaceSnapshot::derive(
            accepted, ContactHaloContext{0, 1, 0, 1, 1}, controls, 0, edits, *arena, arena);
    };
    const auto baseline = arena->used;
    arena->peak = baseline;
    {
        const auto measured = derive();
        EXPECT_GT(measured->index_work().copied_pages, 0u);
        EXPECT_EQ(measured->size(), accepted->size());
    }
    const auto exact_limit = arena->peak;
    ASSERT_EQ(arena->used, baseline);
    ASSERT_GT(exact_limit, baseline);
    arena->limit = exact_limit;
    {
        const auto exact = derive();
        exact->visit_face_pieces(leaf.key, 0, [&](const auto& piece) { EXPECT_EQ(piece.first().revision(), 1u); });
    }
    EXPECT_EQ(arena->used, baseline);
    arena->limit = exact_limit - 1;
    EXPECT_THROW((void) derive(), std::invalid_argument);
    EXPECT_EQ(arena->used, baseline);
    accepted->visit_face_pieces(leaf.key, 0, [&](const auto& piece) { EXPECT_EQ(piece.first().revision(), 0u); });
    EXPECT_EQ(before.first().revision(), 0u);
}
