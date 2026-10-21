/** @file testMixedAMR.cc
 * @brief Mixed-provider assembly, typed adaptation, exact native joins and failure isolation.
 */
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
void closed(const MultiRegionMesh& mesh)
{
    for (std::uint64_t c = 0; c < mesh.num_cells(); ++c)
    {
        MeshUtils::Vec3 sum{};
        double total = 0;
        for (auto f : mesh.cell_faces(c))
        {
            sum = sum + mesh.face_area_vector_outward(f, c);
            total += mesh.face_area(f);
            if (!mesh.is_exterior_face(f))
            {
                auto n = mesh.neighbor_cell(f);
                EXPECT_NE(n, MultiRegionMesh::invalid_cell_id());
                EXPECT_EQ(
                    mesh.face_area_vector_outward(f, mesh.owner_cell(f)).x, -mesh.face_area_vector_outward(f, n).x);
            }
        }
        EXPECT_LE(sum.norm(), 64 * std::numeric_limits<double>::epsilon() * total);
        EXPECT_NO_THROW((void) ConvexCellGeometry::from_native_cell_atomic(mesh, c));
    }
}
TEST(MixedAMRTest, LocalRunsCloseWholeNativeTilesAndRoundTripSharedCatalogs)
{
    auto product = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(test::mixed_product_grid(true));
    auto key = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(product->cell_key(0, 4).ancestry);
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid g;
    g.product = product;
    g.physical_region = 7;
    ::SimpleFluid::Meshes::AMR::MixedAMR root(g);
    auto refined = root.adapt({key});
    EXPECT_GT(refined->geometry()->regions().size(), 1u);
    EXPECT_EQ(refined->geometry()->num_cells(), 19u);
    closed(*refined->geometry());
    bool grouped = false;
    size_t selected = 0;
    for (size_t i = 0; i < refined->geometry()->interfaces().size(); ++i)
        if (const auto* interface =
                std::get_if<MultiProviderNonconformingInterface>(&refined->geometry()->interfaces()[i]))
            for (const auto& participant : interface->fine_patches)
                if (participant.patches.size() > 1)
                    for (const auto& mapping : interface->faces)
                    {
                        std::set<int> boundaries;
                        for (const auto face : mapping.fine_faces)
                            if (face.region == participant.region)
                                for (const auto& patch : participant.patches)
                                    if (std::find(patch.faces.begin(), patch.faces.end(), face.face) !=
                                        patch.faces.end())
                                        boundaries.insert(patch.boundary);
                        if (boundaries.size() > 1)
                        {
                            grouped = true;
                            selected = i;
                        }
                    }
    ASSERT_TRUE(grouped) << "One coarse face needs complete native patches from the same fine provider.";
    auto invalid = refined->geometry()->interfaces();
    auto& descriptor = std::get<MultiProviderNonconformingInterface>(invalid[selected]);
    auto participant = std::find_if(descriptor.fine_patches.begin(), descriptor.fine_patches.end(),
        [](const auto& p) { return p.patches.size() > 1; });
    ASSERT_NE(participant, descriptor.fine_patches.end());
    participant->patches.pop_back();
    EXPECT_THROW(
        (void) MultiRegionMesh(refined->geometry()->regions(), invalid, Teuchos::rcp(new Teuchos::SerialComm<int>())),
        std::invalid_argument);
    invalid = refined->geometry()->interfaces();
    auto& duplicate = std::get<MultiProviderNonconformingInterface>(invalid[selected]);
    duplicate.fine_patches.front().patches.push_back(duplicate.fine_patches.front().patches.front());
    EXPECT_THROW(
        (void) MultiRegionMesh(refined->geometry()->regions(), invalid, Teuchos::rcp(new Teuchos::SerialComm<int>())),
        std::invalid_argument);
    invalid = refined->geometry()->interfaces();
    auto& faces = std::get<MultiProviderNonconformingInterface>(invalid[selected]).faces.front().fine_faces;
    faces.push_back(faces.front());
    EXPECT_THROW(
        (void) MultiRegionMesh(refined->geometry()->regions(), invalid, Teuchos::rcp(new Teuchos::SerialComm<int>())),
        std::invalid_argument);

    std::vector<std::uint64_t> ids(refined->geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), 0);
    const auto decoded = MultiRegionMesh::deserialize_partition(
        refined->geometry()->serialize_partition(ids), Teuchos::rcp(new Teuchos::SerialComm<int>()));
    EXPECT_EQ(decoded->num_cells(), refined->geometry()->num_cells());
    closed(*decoded);
    for (auto c : ids)
        EXPECT_TRUE(ConvexCellGeometry::from_native_cell(*decoded, c)->equivalent(*refined->cell_geometry(c)));
    EXPECT_LE(decoded->storage_report().topology, refined->geometry()->storage_report().topology);
}
TEST(MixedAMRTest, ProductPolyhedralJoinAdaptsWithPersistentPatchIdentityAndPhysicalClosure)
{
    const auto grid = test::mixed_joined_grid(3);
    ::SimpleFluid::Meshes::AMR::MixedAMR root(grid);
    const auto retained = root.storage_breakdown();
    ASSERT_EQ(retained.retained_participants.size(), 2u);
    EXPECT_EQ(retained.retained_participants[0].identity, grid.product.get());
    EXPECT_EQ(retained.retained_participants[1].identity, grid.adaptive.front().get());
    EXPECT_EQ(root.geometry()->num_cells(), 2u);
    const auto original = root.observation();
    const auto k = test::first_product(root);
    auto candidate = root.adapt({k}, {k});
    EXPECT_GT(candidate->geometry()->num_cells(), 9u);
    EXPECT_EQ(root.observation(), original);
    closed(*candidate->geometry());
    bool polyrefined = false;
    for (std::uint64_t c = 0; c < candidate->geometry()->num_cells(); ++c)
    {
        const auto key = candidate->cell_key(c);
        if (auto p = std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&key.ancestry); p && !p->path.empty())
            polyrefined = true;
    }
    EXPECT_TRUE(polyrefined);
    const auto storage = candidate->storage_breakdown();
    EXPECT_GT(storage.assembly_metadata.interfaces, 0u);
    EXPECT_GT(storage.assembly_metadata.indexing, 0u);
    EXPECT_EQ(storage.retained_participants.size(), 2u);
    EXPECT_TRUE(storage.retained_participants[0].includes_complete_planner);
    EXPECT_FALSE(storage.retained_participants[1].includes_complete_planner);
    EXPECT_NE(storage.retained_participants[0].identity, nullptr);
    EXPECT_NE(storage.retained_participants[1].identity, nullptr);
    EXPECT_NE(storage.retained_participants[0].identity, storage.retained_participants[1].identity);
    EXPECT_FALSE(storage.resident_partition_geometry.has_value());
    EXPECT_NO_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        root.observation(), candidate->observation(),
        [&]
        {
            std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> a;
            for (std::uint64_t c = 0; c < root.geometry()->num_cells(); ++c)
                a.push_back({root.cell_key(c), c, root.cell_material(c), root.cell_geometry(c)});
            return a;
        }(),
        [&]
        {
            std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> a;
            for (std::uint64_t c = 0; c < candidate->geometry()->num_cells(); ++c)
                a.push_back({candidate->cell_key(c), c, candidate->cell_material(c), candidate->cell_geometry(c)});
            return a;
        }(),
        root.overlap_relations(*candidate)));
}
TEST(MixedAMRTest, UnsupportedStaticClosureAndBudgetsRejectWithoutChangingSource)
{
    auto g = test::mixed_joined_grid(3);
    auto poly = g.adaptive.front();
    g.adaptive.clear();
    g.stationary.push_back({poly->native_geometry()->regions().front(), 9, 5, 71});
    ::SimpleFluid::Meshes::AMR::MixedAMR source(g);
    const auto before = source.observation();
    auto k = test::first_product(source);
    EXPECT_THROW((void) source.adapt({k}, {k}), std::invalid_argument);
    EXPECT_EQ(source.observation(), before);
    EXPECT_THROW((void) source.adapt({}, {}, {}, {}, {source.cell_key(1)}), std::invalid_argument);
    auto small = test::mixed_joined_grid();
    small.maximum_cells = 2;
    ::SimpleFluid::Meshes::AMR::MixedAMR bounded(small);
    auto mark = test::first_product(bounded);
    EXPECT_THROW((void) bounded.adapt({mark}), std::length_error);
    EXPECT_EQ(bounded.geometry()->num_cells(), 2u);
}
TEST(MixedAMRTest, NativePartitionEpochPlansAndActualEmptyOwners)
{
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid());
    auto plan = source->plan_adaptation();
    EXPECT_EQ(source->apply_adaptation(plan)->observation(), source->observation());
    CompositePartitionOptions options;
    options.policy = CompositePartitionOptions::Policy::ContiguousCanonical;
    options.target_cells_per_fragment = 1;
    auto p = MeshPartitioner<DefaultTpetraTypes>::partition(
        CompositeMeshSource(source->geometry()), options, Tpetra::getDefaultComm());
    auto bound = source->with_partition(std::make_shared<const CompositePartition<>>(p));
    MeshHandle<> handle(*bound->native_partition());
    int local = handle.num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &empty);
    EXPECT_EQ(empty, std::max(0, Tpetra::getDefaultComm()->getSize() - 2));
    EXPECT_EQ(handle.connectivity_storage_bytes(), 0u);
    EXPECT_THROW((void) bound->apply_adaptation(plan), std::invalid_argument);
    const auto product = test::first_product(*bound);
    const auto changed = bound->adapt({product}, {product});
    EXPECT_EQ(changed->observation().partition, bound->observation().partition + 1);
    EXPECT_EQ(changed->observation().topology, bound->observation().topology + 1);
    const auto observed = bound->observation();
    auto retained = std::make_shared<MeshHandle<>>(*bound->native_partition());
    EXPECT_THROW((void) PlanarALEMeshMotion<>(retained), std::exception);
    EXPECT_EQ(bound->observation(), observed);
    EXPECT_TRUE(bound->geometry_unchanged());
}
TEST(MixedAMRTest, ContactDomainsRemainIndependentNativeFluidMeshes)
{
    auto g = ::SimpleFluid::Meshes::AMR::MixedAMRGrid{};
    g.product = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(test::mixed_product_grid());
    auto a = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(g),
         b = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(g);
    ::SimpleFluid::Meshes::AMR::MixedContactDomains contact;
    contact.domains = {a, b};
    contact.contacts = {{0, 1, a->geometry()->boundary_batch_name(*a->geometry()->boundary_batch_ids().begin()),
        b->geometry()->boundary_batch_name(*b->geometry()->boundary_batch_ids().begin())}};
    EXPECT_NO_THROW(contact.validate());
    EXPECT_NE(a->geometry().get(), b->geometry().get());
    EXPECT_EQ(a->geometry()->num_cells(), 1u);
    EXPECT_EQ(b->geometry()->num_cells(), 1u);
}
} // namespace

TEST(MixedAMRTest, CrossedExternalTracesRejectAtomicallyAndQualifiedDescriptorsRejectMissingTiles)
{
    auto g = test::mixed_joined_grid();
    auto right = test::mixed_product_grid();
    for (auto& p : right.xy_nodes)
        p[0] += 1;
    right.region_id = 9;
    right.z_edges = {0, .5, 1};
    g.adaptive = {std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(right)};
    g.joins = {{0, 1, "east", "west"}};
    ::SimpleFluid::Meshes::AMR::MixedAMR source(g);
    const auto before = source.observation();
    const auto k = test::first_product(source);
    EXPECT_THROW((void) source.adapt({k}), std::invalid_argument);
    EXPECT_EQ(source.observation(), before);
    auto interfaces = source.geometry()->interfaces();
    auto* nested = std::get_if<MultiProviderNonconformingInterface>(&interfaces.back());
    ASSERT_NE(nested, nullptr);
    ASSERT_FALSE(nested->faces.empty());
    nested->faces.front().fine_faces.pop_back();
    EXPECT_THROW(
        (void) MultiRegionMesh(source.geometry()->regions(), interfaces, Teuchos::rcp(new Teuchos::SerialComm<int>())),
        std::invalid_argument);
}
TEST(MixedAMRTest, StaticProviderIdentityNativeCellsAndMembershipSurviveCompatibleAssembly)
{
    auto g = test::mixed_joined_grid();
    const auto provider = g.adaptive.front();
    g.adaptive.clear();
    g.stationary.push_back({provider->native_geometry()->regions().front(), 9, 5, 99});
    ::SimpleFluid::Meshes::AMR::MixedAMR root(g);
    const auto key = root.cell_key(1);
    const auto storage =
        std::visit([](const auto& r) { return r.geometry().storage_identity(); }, root.geometry()->regions().back());
    auto k = test::first_product(root);
    auto refined = root.adapt({}, {k});
    EXPECT_EQ(std::visit(
                  [](const auto& r) { return r.geometry().storage_identity(); }, refined->geometry()->regions().back()),
        storage);
    EXPECT_EQ(refined->cell_key(refined->geometry()->num_cells() - 1), key);
    EXPECT_EQ(refined->cell_material(refined->geometry()->num_cells() - 1), 5u);
    EXPECT_EQ(refined->geometry()->native_cell(refined->geometry()->num_cells() - 1).second, 0u);
    closed(*refined->geometry());
}

TEST(MixedAMRTest, PurePolyhedralAndCartesianParticipantsNeedNoUnrelatedProduct)
{
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid polygrid;
    polygrid.adaptive = {test::mixed_polyhedron()};
    ::SimpleFluid::Meshes::AMR::MixedAMR poly(polygrid);
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks split;
    split.participant = 0;
    split.refine = {poly.cell_key(0)};
    auto refined = poly.adapt_participants({split});
    EXPECT_EQ(refined->geometry()->num_cells(), 8u);
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks join;
    join.participant = 0;
    join.coarsen_polyhedral = {poly.cell_key(0)};
    auto restored = refined->adapt_participants({join});
    EXPECT_EQ(restored->geometry()->num_cells(), 1u);
    EXPECT_TRUE(restored->cell_geometry(0)->equivalent(*poly.cell_geometry(0)));
    ::SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.region_id = 31;
    auto cart = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid cartgrid;
    cartgrid.adaptive = {cart};
    ::SimpleFluid::Meshes::AMR::MixedAMR cartesian(cartgrid);
    split.refine = {cartesian.cell_key(0)};
    auto fine = cartesian.adapt_participants({split});
    EXPECT_EQ(fine->geometry()->num_cells(), 64u);
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks coarse;
    coarse.participant = 0;
    coarse.coarsen_cartesian = {cart->blocks().front()};
    auto back = fine->adapt_participants({coarse});
    EXPECT_EQ(back->geometry()->num_cells(), 8u);
    EXPECT_TRUE(back->overlap_relations(cartesian).size() == 8u);
    EXPECT_TRUE(back->overlap_relations(cartesian, {}).empty());
}
TEST(MixedAMRTest, MultipleLocalProductsRetainStablePhysicalJoinsAcrossFactorCoarsening)
{
    auto left = test::mixed_product_grid(), right = left;
    right.region_id = 9;
    for (auto& p : right.xy_nodes)
        p[0] += 1;
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid grid;
    grid.local_products = {std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(left),
        std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(right)};
    grid.joins = {{0, 1, "east", "west"}};
    ::SimpleFluid::Meshes::AMR::MixedAMR root(grid);
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks split;
    split.participant = 0;
    split.refine_z = {std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(root.cell_key(0).ancestry)};
    auto preview = root.plan_participants({split});
    auto fine = root.apply_adaptation(preview);
    EXPECT_EQ(fine->geometry()->num_cells(), 3u);
    closed(*fine->geometry());
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks coarse;
    coarse.participant = 0;
    coarse.coarsen_z = split.refine_z;
    auto back = fine->adapt_participants({coarse});
    EXPECT_EQ(back->geometry()->num_cells(), 2u);
    EXPECT_EQ(back->cell_key(0), root.cell_key(0));
    closed(*back->geometry());
    EXPECT_THROW((void) fine->apply_adaptation(preview), std::invalid_argument);
}
TEST(MixedAMRTest, DuplicatePersistentKeysRejectWithoutRenumberingPhysicalRegions)
{
    ::SimpleFluid::Meshes::AMR::AMRGrid a;
    a.block_cells = {2, 2, 2};
    ::SimpleFluid::Meshes::AMR::AMRGrid b = a;
    b.origin = {1, 0, 0};
    auto first = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(a),
         second = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(b);
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid grid;
    grid.adaptive = {first, second};
    grid.joins = {{0, 1, "xmax", "xmin"}};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(grid), std::invalid_argument);
    const auto& left = std::get<CartesianRegion>(first->geometry()->regions().front());
    const auto& right = std::get<CartesianRegion>(second->geometry()->regions().front());
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid statics;
    statics.stationary = {
        {CartesianRegion("left_static", std::shared_ptr<const RectilinearTopology>(first, &left.topology()),
             std::shared_ptr<const RectilinearGeometry>(first, &left.geometry())),
            0, 0, 0},
        {CartesianRegion("right_static", std::shared_ptr<const RectilinearTopology>(second, &right.topology()),
             std::shared_ptr<const RectilinearGeometry>(second, &right.geometry())),
            0, 0, 0}};
    statics.joins = {{0, 1, "xmax", "xmin"}};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(statics), std::invalid_argument);
}
TEST(MixedAMRTest, RankLocalProductSemanticMismatchAndInvalidMarksRejectCollectively)
{
    if (Tpetra::getDefaultComm()->getSize() < 2)
        GTEST_SKIP() << "Rank-local semantic validation requires MPI.";
    auto rootgrid = test::mixed_product_grid();
    auto first = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(rootgrid);
    ++rootgrid.region_id;
    auto second = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(rootgrid);
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid mismatched;
    mismatched.local_products = {Tpetra::getDefaultComm()->getRank() == 0 ? first : second};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(mismatched), std::invalid_argument);
    ::SimpleFluid::Meshes::AMR::MixedAMR source(test::mixed_joined_grid());
    const auto before = source.observation();
    std::vector<::SimpleFluid::Meshes::AMR::MixedParticipantMarks> marks;
    if (Tpetra::getDefaultComm()->getRank() == 0)
    {
        ::SimpleFluid::Meshes::AMR::MixedParticipantMarks bad;
        bad.participant = 1;
        auto key = source.cell_key(1);
        std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(key.ancestry).path = {99};
        bad.refine = {key};
        marks.push_back(bad);
    }
    EXPECT_THROW((void) source.adapt_participants(marks), std::invalid_argument);
    EXPECT_EQ(source.observation(), before);
}

TEST(MixedAMRTest, FourFineFacesAcrossTwoSharedRunProvidersRetainOneCanonicalFluxEach)
{
    auto g = test::mixed_product_grid();
    g.xy_nodes = {{1, 0, 0}, {2, 0, 0}, {1, .5, 0}, {2, .5, 0}, {1, 1, 0}, {2, 1, 0}};
    g.base_cells = {{0, 1, 3, 2}, {2, 3, 5, 4}};
    g.boundaries = {{0, 1, "south"}, {1, 3, "east"}, {3, 5, "east"}, {5, 4, "north"}, {4, 2, "west"}, {2, 0, "west"}};
    g.z_edges = {0, .5, 1};
    auto product = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(g);
    ASSERT_EQ(product->runs().size(), 1u);
    const auto& full = product->runs().front().region;
    auto anchor = full.geometry().construction();
    auto base = std::shared_ptr<const ExtrudedTopology>(product, &full.topology());
    auto one = base->with_layers(1);
    std::vector<MultiRegionMesh::Region> regions{cartesian_region("coarse", {{{0, 1}, {0, 1}, {0, 1}}})};
    for (size_t z = 0; z < 2; ++z)
    {
        auto view = ExactExtrudedConstruction::shared_view(one, anchor, anchor, z, 1);
        regions.emplace_back(
            ExtrudedRegion("fine_run_" + std::to_string(z), one, std::make_shared<ExtrudedGeometry>(one, view)));
    }
    auto boundary = [&](size_t r, const std::string& name)
    {
        return std::visit(
            [&](const auto& region)
            {
                for (auto b : region.topology().boundary_batch_ids())
                    if (region.topology().boundary_batch_name(b) == name)
                        return b;
                throw std::logic_error("Missing fixture boundary.");
            },
            regions[r]);
    };
    auto faces = [&](size_t r, int b)
    {
        return std::visit(
            [&](const auto& region)
            {
                std::vector<std::uint64_t> result;
                for (std::uint64_t f = 0; f < region.layout().faces; ++f)
                    if (region.topology().boundary_id(f) == b)
                        result.push_back(f);
                return result;
            },
            regions[r]);
    };
    MultiProviderNonconformingInterface side;
    side.coarse_region = 0;
    side.coarse_boundary = boundary(0, "xmax");
    const auto coarse = faces(0, side.coarse_boundary);
    ASSERT_EQ(coarse.size(), 1u);
    side.faces.push_back({coarse.front(), {}});
    for (size_t r = 1; r < 3; ++r)
    {
        const auto b = boundary(r, "west");
        side.fine_patches.push_back({r, b, 0});
        for (auto f : faces(r, b))
            side.faces.front().fine_faces.push_back({r, f});
    }
    ASSERT_EQ(side.faces.front().fine_faces.size(), 4u);
    MultiProviderNonconformingInterface cap;
    cap.coarse_region = 1;
    cap.coarse_boundary = boundary(1, "zmax");
    const auto finecap = boundary(2, "zmin");
    cap.fine_patches = {{2, finecap, 0}};
    const auto lower = faces(1, cap.coarse_boundary), upper = faces(2, finecap);
    ASSERT_EQ(lower.size(), upper.size());
    for (size_t i = 0; i < lower.size(); ++i)
        cap.faces.push_back({lower[i], {{2, upper[i]}}});
    MultiRegionMesh mesh(regions, {side, cap}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    closed(mesh);
    EXPECT_TRUE(mesh.has_qualified_exact_interfaces());
    size_t cross = 0;
    double integrated = 0;
    for (std::uint64_t f = 0; f < mesh.num_faces(); ++f)
        if (!mesh.is_exterior_face(f))
        {
            const auto owner = mesh.owner_cell(f), neighbor = mesh.neighbor_cell(f);
            if (owner == 0 || neighbor == 0)
            {
                ++cross;
                integrated += mesh.face_area(f);
                EXPECT_DOUBLE_EQ(
                    mesh.face_area_vector_outward(f, owner).x, -mesh.face_area_vector_outward(f, neighbor).x);
            }
        }
    EXPECT_EQ(cross, 4u);
    EXPECT_DOUBLE_EQ(integrated, 1);
    std::map<std::uint64_t, double> flux;
    const std::array<double, 4> signed_flux{3, -5, 7, -11};
    size_t next = 0;
    for (std::uint64_t f = 0; f < mesh.num_faces(); ++f)
        if (!mesh.is_exterior_face(f) && (mesh.owner_cell(f) == 0 || mesh.neighbor_cell(f) == 0))
            flux.emplace(f, signed_flux[next++]);
    auto restriction = [&](const MultiRegionMesh& native)
    {
        double coarse = 0, receivers = 0;
        std::map<std::uint64_t, size_t> visits;
        for (std::uint64_t c = 0; c < native.num_cells(); ++c)
            for (auto f : native.cell_faces(c))
                if (auto it = flux.find(f); it != flux.end())
                {
                    const double outward = (native.owner_cell(f) == c ? 1 : -1) * it->second;
                    if (!c)
                        coarse += outward;
                    else
                        receivers += outward;
                    ++visits[f];
                }
        for (const auto& [face, count] : visits)
            EXPECT_EQ(count, 2u);
        EXPECT_EQ(visits.size(), 4u);
        EXPECT_DOUBLE_EQ(coarse, 6);
        EXPECT_DOUBLE_EQ(receivers, -6);
        EXPECT_DOUBLE_EQ(coarse + receivers, 0);
        EXPECT_DOUBLE_EQ(native.restrict_face_flux(
                             {0, side.faces.front().coarse_face}, [&](std::uint64_t face) { return flux.at(face); }),
            coarse);
        return coarse;
    };
    const auto restricted = restriction(mesh);
    std::vector<std::uint64_t> cells(mesh.num_cells());
    std::iota(cells.begin(), cells.end(), 0);
    const auto decoded = MultiRegionMesh::deserialize_partition(
        mesh.serialize_partition(cells), Teuchos::rcp(new Teuchos::SerialComm<int>()));
    closed(*decoded);
    EXPECT_EQ(restriction(*decoded), restricted);
    const auto& a = std::get<ExtrudedRegion>(decoded->regions()[1]);
    const auto& b = std::get<ExtrudedRegion>(decoded->regions()[2]);
    EXPECT_EQ(a.topology().base_storage_identity(), b.topology().base_storage_identity());
    EXPECT_EQ(
        a.geometry().construction()->base_storage_identity(), b.geometry().construction()->base_storage_identity());
    EXPECT_EQ(a.geometry().construction()->axis_values_storage_identity(),
        b.geometry().construction()->axis_values_storage_identity());
}

TEST(MixedAMRTest, IndependentNativePatchCatalogRejectsOmittedFineTilesAndSurvivesSparsePackets)
{
    auto left = cartesian_region("left", {{{0, 1}, {0, .5, 1}, {0, 1}}}),
         right = cartesian_region("right", {{{1, 2}, {0, .5, 1}, {0, 1}}});
    MultiProviderNonconformingInterface join;
    join.coarse_region = 0;
    join.coarse_boundary = 1;
    join.fine_patches = {{1, 0, 0}};
    std::vector<std::uint64_t> coarse, fine;
    for (std::uint64_t f = 0; f < left.layout().faces; ++f)
        if (left.topology().boundary_id(f) == 1)
            coarse.push_back(f);
    for (std::uint64_t f = 0; f < right.layout().faces; ++f)
        if (right.topology().boundary_id(f) == 0)
            fine.push_back(f);
    ASSERT_EQ(coarse.size(), 2u);
    ASSERT_EQ(fine.size(), 2u);
    for (size_t i = 0; i < 2; ++i)
        join.faces.push_back({coarse[i], {{1, fine[i]}}});
    MultiRegionMesh native({left, right}, {join}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    const auto valid = native.serialize_partition({});
    EXPECT_NO_THROW((void) MultiRegionMesh::deserialize_partition(valid, Teuchos::rcp(new Teuchos::SerialComm<int>())));
    auto encode = [](const MultiProviderNonconformingInterface& descriptor)
    {
        std::vector<char> bytes;
        auto append = [&](const auto& value)
        {
            const auto* p = reinterpret_cast<const char*>(&value);
            bytes.insert(bytes.end(), p, p + sizeof(value));
        };
        append(std::uint32_t{3});
        append(descriptor.coarse_region);
        append(descriptor.coarse_boundary);
        append(std::uint8_t(descriptor.partial_coarse_patch));
        append(descriptor.coarse_face_count);
        append(std::uint64_t(descriptor.fine_patches.size()));
        for (const auto& p : descriptor.fine_patches)
        {
            append(p.region);
            append(std::uint64_t(p.patches.size()));
            for (const auto& native : p.patches)
            {
                append(native.boundary);
                append(native.face_count);
                append(std::uint64_t(native.faces.size()));
                for (auto face : native.faces)
                    append(face);
            }
        }
        append(std::uint64_t(descriptor.faces.size()));
        for (const auto& group : descriptor.faces)
        {
            append(group.coarse_face);
            append(std::uint64_t(group.fine_faces.size()));
            for (const auto& f : group.fine_faces)
            {
                append(f.region);
                append(f.face);
            }
        }
        return bytes;
    };
    const auto original = std::get<MultiProviderNonconformingInterface>(native.interfaces().front());
    const auto pattern = encode(original);
    auto forged = original;
    forged.partial_coarse_patch = true;
    forged.faces.pop_back();
    --forged.fine_patches[0].patches[0].face_count;
    forged.fine_patches[0].patches[0].faces.pop_back();
    const auto replacement = encode(forged);
    auto packet = valid;
    const auto found = std::search(packet.begin(), packet.end(), pattern.begin(), pattern.end());
    ASSERT_NE(found, packet.end());
    const auto offset = size_t(found - packet.begin());
    ASSERT_EQ(std::search(found + pattern.size(), packet.end(), pattern.begin(), pattern.end()), packet.end());
    packet.erase(packet.begin() + offset, packet.begin() + offset + pattern.size());
    packet.insert(packet.begin() + offset, replacement.begin(), replacement.end());
    try
    {
        (void) MultiRegionMesh::deserialize_partition(packet, Teuchos::rcp(new Teuchos::SerialComm<int>()));
        FAIL() << "Omitted tiles must not redefine complete native fine patches.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("fine participant"), std::string::npos);
    }
    ::SimpleFluid::Meshes::AMR::MixedAMR mixed(test::mixed_joined_grid());
    const auto empty = mixed.geometry()->serialize_partition({});
    auto shard = MultiRegionMesh::deserialize_partition(empty, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    EXPECT_EQ(shard->serialize_partition({}), empty);
    const auto roundtrip = MultiRegionMesh::deserialize_partition(
        shard->serialize_partition({}), Teuchos::rcp(new Teuchos::SerialComm<int>()));
    EXPECT_EQ(shard->configuration_signature(), roundtrip->configuration_signature());
    EXPECT_NE(shard->configuration_signature().find("native_patch_counts"), std::string::npos);
}

TEST(MixedAMRTest, MicroscopicNativeFluxOrientationDoesNotDependOnSquaredAreasOrRatioMagnitude)
{
    const double scale = 1e-100;
    auto left = cartesian_region("micro_coarse", {{{0, 1}, {0, scale}, {0, scale}}}),
         right = cartesian_region("micro_fine", {{{1, 2}, {0, scale / 2, scale}, {0, scale}}});
    MultiProviderNonconformingInterface join;
    join.coarse_region = 0;
    join.coarse_boundary = 1;
    join.fine_patches = {{1, 0, 0}};
    std::uint64_t coarse = 0;
    std::vector<std::uint64_t> fine;
    for (std::uint64_t f = 0; f < left.layout().faces; ++f)
        if (left.topology().boundary_id(f) == 1)
            coarse = f;
    for (std::uint64_t f = 0; f < right.layout().faces; ++f)
        if (right.topology().boundary_id(f) == 0)
            fine.push_back(f);
    ASSERT_EQ(fine.size(), 2u);
    join.faces = {{coarse, {{1, fine[0]}, {1, fine[1]}}}};
    MultiRegionMesh mesh({left, right}, {join}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    std::map<std::uint64_t, double> flux;
    const std::array<double, 2> values{3.5, -9};
    for (size_t i = 0; i < fine.size(); ++i)
    {
        const RegionFace address{1, fine[i]};
        const auto canonical = mesh.canonical_face(address);
        flux[canonical] = values[i];
        const auto vector = mesh.face_area_vector(canonical);
        EXPECT_EQ(vector.dot(vector), 0);
        EXPECT_GT(mesh.face_area(canonical), 0);
        EXPECT_TRUE(std::isfinite(mesh.face_area(canonical)));
        const auto normal = mesh.face_normal(canonical);
        EXPECT_DOUBLE_EQ(normal.norm(), 1);
        EXPECT_DOUBLE_EQ(mesh.acquire_execution_view().resolve_face(canonical).normal().norm(), 1);
        EXPECT_DOUBLE_EQ(mesh.face_flux_weights(address)[0].coefficient, 1);
        EXPECT_DOUBLE_EQ(mesh.restrict_face_flux(address, [&](std::uint64_t f) { return flux.at(f); }), values[i]);
    }
    for (auto weight : mesh.face_flux_weights({0, coarse}))
        EXPECT_DOUBLE_EQ(weight.coefficient, -.5);
    EXPECT_DOUBLE_EQ(mesh.restrict_face_flux({0, coarse}, [&](std::uint64_t f) { return flux.at(f); }), 5.5);
    // A positive area fraction below binary64 range must not determine an
    // integrated flux's sign or make otherwise valid restriction unavailable.
    auto wide = cartesian_region("wide", {{{0, 1}, {0, 1e100}, {0, 1e100}}}),
         mixed = cartesian_region("mixed_scale", {{{1, 2}, {0, 1e-100, 1e100}, {0, 1e-100, 1e100}}});
    MultiProviderNonconformingInterface extreme;
    extreme.coarse_region = 0;
    extreme.coarse_boundary = 1;
    extreme.fine_patches = {{1, 0, 0}};
    std::uint64_t wc = 0;
    for (std::uint64_t f = 0; f < wide.layout().faces; ++f)
        if (wide.topology().boundary_id(f) == 1)
            wc = f;
    extreme.faces = {{wc, {}}};
    for (std::uint64_t f = 0; f < mixed.layout().faces; ++f)
        if (mixed.topology().boundary_id(f) == 0)
            extreme.faces[0].fine_faces.push_back({1, f});
    MultiRegionMesh ranged({wide, mixed}, {extreme}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    std::map<std::uint64_t, double> arbitrary;
    const std::array<double, 4> amounts{1, -2, 3, -4};
    size_t index = 0, tiny_index = 0;
    for (auto f : ranged.logical_faces({0, wc}))
    {
        arbitrary[f] = amounts[index];
        if (ranged.face_area(f) < 1e-100)
            tiny_index = index;
        ++index;
    }
    ASSERT_EQ(index, 4u);
    EXPECT_DOUBLE_EQ(ranged.restrict_face_flux({0, wc}, [&](std::uint64_t f) { return arbitrary.at(f); }), 2);
    EXPECT_THROW((void) ranged.face_flux_weights({0, wc})[tiny_index], std::overflow_error);
    EXPECT_DOUBLE_EQ(MultiRegionMesh::ResolvedFace{}.area(), 0);
    const MultiRegionMesh::ResolvedFace negative_zero{.area_vector = {-0.0, 0, -0.0}};
    EXPECT_DOUBLE_EQ(negative_zero.area(), 0);
}

TEST(MixedAMRTest, InternalLocalFaceEdgeAndVertexGradationRejectsSecondBoundaryRefinementAtomically)
{
    for (const int dimension : {2, 1, 0})
    {
        auto grid = test::mixed_product_grid(true);
        if (dimension == 1)
            grid.z_edges = {0, 1};
        if (dimension == 0)
            grid.z_edges = {0, 1, 2};
        auto root = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(grid);
        std::vector<::SimpleFluid::Meshes::AMR::ProductCellAncestry> first_marks;
        for (size_t r = 0; r < root->runs().size(); ++r)
            for (std::uint64_t c = 0; c < root->runs()[r].region.layout().cells; ++c)
            {
                const auto key =
                    std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(root->cell_key(r, c).ancestry);
                if (dimension == 2   ? (key.base.root == 0 && key.axial.root == 1)
                    : dimension == 1 ? key.base.root != 3
                                     : !(key.base.root == 3 && key.axial.root == 1))
                    first_marks.push_back(key);
            }
        const auto first = root->adapt(first_marks);
        ASSERT_TRUE(first->contact_violations(2).empty());
        std::optional<::SimpleFluid::Meshes::AMR::ProductCellAncestry> selected;
        for (size_t r = 0; r < first->runs().size(); ++r)
            for (std::uint64_t c = 0; c < first->runs()[r].region.layout().cells; ++c)
            {
                const auto key =
                    std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(first->cell_key(r, c).ancestry);
                const auto center = first->runs()[r].region.geometry().cell_centroid(c);
                if (key.base.root == 0 && key.base.path.size() == 1 && key.axial.root == (dimension == 2 ? 1u : 0u) &&
                    center.x > .5 && (dimension == 2 ? center.y < .5 : center.y > .5))
                    selected = key;
            }
        ASSERT_TRUE(selected.has_value());
        const auto second = first->adapt({*selected});
        const auto violations = second->contact_violations(2);
        ASSERT_FALSE(violations.empty());
        bool expected = false;
        for (const auto& violation : violations)
        {
            EXPECT_LE(violation.dimension, dimension);
            expected = expected || violation.dimension == dimension;
        }
        EXPECT_TRUE(expected);
        ::SimpleFluid::Meshes::AMR::MixedAMRGrid assembled;
        assembled.local_products = {first};
        if (dimension == 0)
        {
            const auto original = first->identity();
            bool witness = false;
            for (const auto& v : violations)
            {
                auto refined = [](const auto& k)
                { return k.base.root == 0 && k.base.path.size() == 2 && k.axial.root == 0; };
                auto coarse = [](const auto& k)
                { return k.base.root == 3 && k.base.path.empty() && k.axial.root == 1; };
                witness =
                    witness || (v.dimension == 0 && v.span_exceeded &&
                                   ((refined(v.first) && coarse(v.second)) || (coarse(v.first) && refined(v.second))));
            }
            EXPECT_TRUE(witness);
            try
            {
                (void) ::SimpleFluid::Meshes::AMR::MixedAMR(assembled);
                FAIL() << "Balanced crossed patch grouping remains unsupported.";
            }
            catch (const std::invalid_argument& error)
            {
                EXPECT_NE(std::string(error.what()).find("crossed or unmatched seam"), std::string::npos);
            }
            EXPECT_EQ(first->identity(), original);
        }
        else
        {
            ::SimpleFluid::Meshes::AMR::MixedAMR source(assembled);
            const auto before = source.observation();
            const auto retained = source.plan_participants({});
            ::SimpleFluid::Meshes::AMR::MixedParticipantMarks mark;
            mark.participant = 0;
            mark.refine_xy = {*selected};
            EXPECT_THROW((void) source.adapt_participants({mark}), std::invalid_argument);
            EXPECT_EQ(source.observation(), before);
            EXPECT_EQ(source.apply_adaptation(retained)->observation(), before);
        }
        assembled.local_products = {second};
        try
        {
            (void) ::SimpleFluid::Meshes::AMR::MixedAMR(assembled);
            FAIL() << "Overgraded candidate must reject before native seam assembly.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("gradation"), std::string::npos);
        }
        if (dimension == 2)
        {
            const auto trace = second->contact_violations(16);
            ASSERT_FALSE(trace.empty());
            EXPECT_TRUE(std::any_of(
                trace.begin(), trace.end(), [](const auto& v) { return v.trace_jump_exceeded && !v.span_exceeded; }));
            assembled.maximum_span_ratio = 16;
            EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(assembled), std::invalid_argument);
        }
    }
}

TEST(MixedAMRTest, ManualCanonicalInterfacesCannotBypassPhysicalContactAdmission)
{
    auto make = [](double upper)
    {
        auto grid = test::mixed_joined_grid(upper);
        const auto stationary = grid.adaptive.front();
        grid.adaptive.clear();
        grid.stationary.push_back({stationary->native_geometry()->regions().front(), 9, 5, 71});
        grid.joins.clear();
        const auto& product = grid.product->runs().front().region;
        const auto& native = std::get<DistributedExplicitRegion>(grid.stationary.front().region);
        int east = -1, west = -1;
        for (auto b : product.topology().boundary_batch_ids())
            if (product.topology().boundary_batch_name(b) == "east")
                east = b;
        for (auto b : native.topology().boundary_batch_ids())
            if (native.topology().boundary_batch_name(b) == "polyhedral/join")
                west = b;
        if (east < 0 || west < 0)
            throw std::logic_error("Missing authoritative manual join label.");
        std::uint64_t coarse = 0;
        std::vector<RegionFace> fine;
        for (std::uint64_t f = 0; f < native.layout().faces; ++f)
            if (native.topology().boundary_id(f) == west)
                coarse = f;
        for (std::uint64_t f = 0; f < product.layout().faces; ++f)
            if (product.topology().boundary_id(f) == east)
                fine.push_back({0, f});
        MultiProviderNonconformingInterface join;
        join.coarse_region = 1;
        join.coarse_boundary = west;
        join.fine_patches = {{0, east, 0}};
        join.faces = {{coarse, fine}};
        grid.interfaces = {join};
        return grid;
    };
    auto bounded = make(3);
    EXPECT_NO_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(bounded));
    auto excessive = make(5);
    const auto identity = excessive.product->identity();
    const auto epoch = std::visit(
        [](const auto& region) { return region.geometry().geometry_epoch(); }, excessive.stationary.front().region);
    try
    {
        (void) ::SimpleFluid::Meshes::AMR::MixedAMR(excessive);
        FAIL() << "Manual canonical joins must not bypass configured gradation.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("static"), std::string::npos);
    }
    EXPECT_EQ(excessive.product->identity(), identity);
    EXPECT_EQ(std::visit([](const auto& region) { return region.geometry().geometry_epoch(); },
                  excessive.stationary.front().region),
        epoch);
}

TEST(MixedAMRTest, AdjacentAxialSiblingTracesKeepOneGenerationLimitWithLoosePhysicalRatio)
{
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(test::mixed_product_grid());
    auto parent = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(root->cell_key(0, 0).ancestry);
    const auto first = root->adapt({}, {parent});
    std::optional<::SimpleFluid::Meshes::AMR::ProductCellAncestry> left;
    for (size_t r = 0; r < first->runs().size(); ++r)
        for (std::uint64_t c = 0; c < first->runs()[r].region.layout().cells; ++c)
        {
            const auto key = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(first->cell_key(r, c).ancestry);
            if (key.axial.path == std::vector<std::uint64_t>{0})
                left = key;
        }
    ASSERT_TRUE(left.has_value());
    const auto second = first->adapt({}, {*left});
    std::optional<::SimpleFluid::Meshes::AMR::ProductCellAncestry> touching;
    for (size_t r = 0; r < second->runs().size(); ++r)
        for (std::uint64_t c = 0; c < second->runs()[r].region.layout().cells; ++c)
        {
            const auto key = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(second->cell_key(r, c).ancestry);
            if (key.axial.path == std::vector<std::uint64_t>{0, 1})
                touching = key;
        }
    ASSERT_TRUE(touching.has_value());
    ASSERT_TRUE(second->contact_violations(16).empty());
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid grid;
    grid.local_products = {second};
    grid.maximum_span_ratio = 16;
    ::SimpleFluid::Meshes::AMR::MixedAMR source(grid);
    const auto before = source.observation();
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks mark;
    mark.participant = 0;
    mark.refine_z = {*touching};
    EXPECT_THROW((void) source.adapt_participants({mark}), std::invalid_argument);
    EXPECT_EQ(source.observation(), before);
    const auto third = second->adapt({}, {*touching});
    const auto violations = third->contact_violations(16);
    ASSERT_FALSE(violations.empty());
    EXPECT_TRUE(std::any_of(violations.begin(), violations.end(),
        [](const auto& v) { return v.dimension == 2 && v.trace_jump_exceeded && !v.span_exceeded; }));
}

TEST(MixedAMRTest, JoinedFaceNeighborhoodDoesNotEnrollUnrelatedGradedInteriorCells)
{
    auto grid = test::mixed_product_grid();
    grid.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}};
    grid.base_cells = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}};
    grid.boundaries = {{0, 1, "south"}, {1, 2, "south"}, {2, 3, "south"}, {3, 7, "east"}, {7, 6, "north"},
        {6, 5, "north"}, {5, 4, "north"}, {4, 0, "west"}};
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(grid);
    std::vector<::SimpleFluid::Meshes::AMR::ProductCellAncestry> marks;
    for (std::uint64_t c = 0; c < 3; ++c)
    {
        const auto key = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(root->cell_key(0, c).ancestry);
        if (key.base.root < 2)
            marks.push_back(key);
    }
    const auto first = root->adapt(marks);
    marks.clear();
    for (size_t r = 0; r < first->runs().size(); ++r)
        for (std::uint64_t c = 0; c < first->runs()[r].region.layout().cells; ++c)
        {
            const auto key = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(first->cell_key(r, c).ancestry);
            if (key.base.root == 0)
                marks.push_back(key);
        }
    const auto graded = first->adapt(marks);
    ASSERT_EQ(graded->storage_counts().cells, 21u);
    ASSERT_TRUE(graded->contact_violations(2).empty());
    const auto stationary = test::mixed_polyhedron(4, 3);
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid assembly;
    assembly.local_products = {graded};
    assembly.stationary.push_back({stationary->geometry()->regions().front(), 9, 5, 71});
    assembly.joins = {{0, 1, "east", "polyhedral/join"}};
    ::SimpleFluid::Meshes::AMR::MixedAMR source(assembly);
    EXPECT_EQ(source.geometry()->num_cells(), 22u);
    closed(*source.geometry());
}

TEST(MixedAMRTest, WholeProductAdapterCannotBypassComparableAxialTraceAdmission)
{
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(test::mixed_product_grid());
    const auto first = root->adapt({}, {0});
    const auto second = first->adapt({}, {0});
    ASSERT_TRUE(second->contact_violations(16).empty());
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid grid;
    grid.adaptive = {second};
    grid.maximum_span_ratio = 16;
    ::SimpleFluid::Meshes::AMR::MixedAMR source(grid);
    const auto before = source.observation();
    std::optional<::SimpleFluid::Meshes::AMR::ProductCellAncestry> selected;
    for (std::uint64_t c = 0; c < source.geometry()->num_cells(); ++c)
    {
        const auto key = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(source.cell_key(c).ancestry);
        if (key.axial.path == std::vector<std::uint64_t>{0, 1})
            selected = key;
    }
    ASSERT_TRUE(selected.has_value());
    ::SimpleFluid::Meshes::AMR::MixedParticipantMarks mark;
    mark.participant = 0;
    mark.refine_z = {*selected};
    EXPECT_THROW((void) source.adapt_participants({mark}), std::invalid_argument);
    EXPECT_EQ(source.observation(), before);
    const auto third = second->adapt({}, {1});
    EXPECT_FALSE(third->contact_violations(16).empty());
    grid.adaptive = {third};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(grid), std::invalid_argument);
}

TEST(MixedAMRTest, InitialLocalZSplitRemainsAdmissibleAndContactProofBudgetIsAtomic)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::LocalProductAMR>(test::mixed_product_grid(true));
    const auto selected = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(root->cell_key(0, 4).ancestry);
    const auto first = root->adapt({}, {selected});
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid grid;
    grid.local_products = {first};
    ::SimpleFluid::Meshes::AMR::MixedAMR admitted(grid);
    EXPECT_EQ(admitted.geometry()->num_cells(), 17u);
    closed(*admitted.geometry());
    const auto identity = root->identity();
    grid.local_products = {root};
    grid.maximum_interface_tiles = 1;
    try
    {
        (void) ::SimpleFluid::Meshes::AMR::MixedAMR(grid);
        FAIL() << "Contact proof budget must reject before publication.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("budget"), std::string::npos);
    }
    EXPECT_EQ(root->identity(), identity);
    EXPECT_EQ(root->storage_counts().cells, 16u);
    EXPECT_EQ(admitted.geometry()->num_cells(), 17u);
}

TEST(MixedAMRTest, BoundAdaptiveSourceMotionCannotBeRevivedByAssemblyOrRollback)
{
    ::SimpleFluid::Meshes::AMR::AMRGrid config;
    config.block_cells = {2, 2, 2};
    const auto canonical = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(config);
    const auto canonical_observation = canonical->observation();
    const auto volume = canonical->geometry()->cell_volume(0);
    auto bind = [&]
    {
        auto partition = std::make_shared<const CompositePartition<>>(MeshPartitioner<DefaultTpetraTypes>::partition(
            CompositeMeshSource(canonical->geometry()), {}, Tpetra::getDefaultComm()));
        return canonical->with_partition(partition);
    };
    const auto moved = bind();
    auto moved_mesh = std::make_shared<MeshHandle<>>(*moved->native_partition());
    PlanarALEMeshMotion<> before_import(moved_mesh);
    const auto original = before_import.snapshot();
    before_import.begin_trial(1.1, .5);
    before_import.accept_trial();
    ASSERT_FALSE(moved->geometry_unchanged());
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid stale;
    stale.adaptive = {moved};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(stale), std::invalid_argument);
    const auto moved_epoch = moved->observation().geometry;
    before_import.restore(original);
    EXPECT_GT(moved->observation().geometry, moved_epoch);
    EXPECT_FALSE(moved->geometry_unchanged());
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::MixedAMR(stale), std::invalid_argument);
    EXPECT_TRUE(canonical->geometry_unchanged());
    EXPECT_EQ(canonical->observation(), canonical_observation);
    EXPECT_EQ(canonical->geometry()->cell_volume(0), volume);
    const auto retained = bind();
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid grid;
    grid.adaptive = {retained};
    ::SimpleFluid::Meshes::AMR::MixedAMR source(grid);
    const auto accepted = source.observation();
    const auto pending = source.plan_participants({});
    auto mesh = std::make_shared<MeshHandle<>>(*retained->native_partition());
    PlanarALEMeshMotion<> motion(mesh);
    const auto checkpoint = motion.snapshot();
    motion.begin_trial(1.1, .5);
    motion.accept_trial();
    EXPECT_FALSE(source.geometry_unchanged());
    EXPECT_NE(source.observation(), accepted);
    EXPECT_EQ(source.observation().geometry, std::numeric_limits<std::uint64_t>::max());
    EXPECT_THROW((void) source.apply_adaptation(pending), std::invalid_argument);
    EXPECT_THROW((void) source.adapt(), std::invalid_argument);
    EXPECT_THROW((void) source.adapt_participants({}), std::invalid_argument);
    EXPECT_THROW((void) source.cell_geometry(0), std::invalid_argument);
    const auto epoch = retained->observation().geometry;
    motion.restore(checkpoint);
    EXPECT_GT(retained->observation().geometry, epoch);
    EXPECT_FALSE(source.geometry_unchanged());
    EXPECT_THROW((void) source.apply_adaptation(pending), std::invalid_argument);
    EXPECT_THROW((void) source.adapt_participants({}), std::invalid_argument);
    EXPECT_EQ(source.geometry()->cell_volume(0), volume);
    EXPECT_TRUE(canonical->geometry_unchanged());
}
