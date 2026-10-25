/** @file testFrozenNative.cc
 * @brief Accepted concave/curved native ownership copies, source release and topology barriers.
 */
#include "fields/FieldStored.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/FrozenNative.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "parallel/unitTests/frozen_native_helpers.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <bit>
#include <gtest/gtest.h>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using namespace SimpleFluid::test;
void consume(const ::SimpleFluid::Meshes::AMR::FrozenNativeModel& model,
    const std::map<std::uint64_t, SimpleFluid::PLIC::GeometryPoint>& expected_normals = {})
{
    auto publication = model.publication();
    EXPECT_TRUE(publication.partition->geometry()->frozen_native());
    EXPECT_EQ(publication.mesh->num_owned_cells(), model.owned().size());
    ScalarCellFieldStored<> field(publication.mesh, "frozen_key");
    using LO = DefaultTpetraTypes::local_ordinal_type;
    auto value = [](const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& key)
    {
        std::uint64_t native = 0;
        if (const auto* cell = std::get_if<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(&key.ancestry))
            native = cell->index[0] + 2 * (cell->index[1] + 2 * cell->index[2]);
        else
            native = std::get<::SimpleFluid::Meshes::AMR::StaticCellAncestry>(key.ancestry).native_cell;
        return double(key.region * 100 + native) + .25;
    };
    for (std::size_t c = 0; c < publication.mesh->num_owned_cells(); ++c)
    {
        auto gid = publication.mesh->cell_global_id(LO(c));
        field.set_owned_value(LO(c), value(publication.local_keys.at(gid)));
    }
    field.sync_ghosts();
    for (std::size_t c = 0; c < publication.mesh->num_cells(); ++c)
    {
        auto gid = publication.mesh->cell_global_id(LO(c));
        EXPECT_EQ(field.local_value(LO(c)), value(publication.local_keys.at(gid)));
    }
    for (auto face : publication.partition->plan().face_ids())
        if (!expected_normals.empty())
        {
            auto actual = publication.partition->geometry()->face_normal(face);
            const auto& expected = expected_normals.at(face);
            for (std::size_t axis = 0; axis < 3; ++axis)
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(actual.component(axis)), std::bit_cast<std::uint64_t>(expected[axis]));
        }
    double volume = 0;
    for (const auto& [key, gid] : publication.owned_gids)
    {
        volume += publication.partition->geometry()->cell_volume(gid);
        EXPECT_THROW(publication.partition->geometry()->certified_sparse_cell_geometry(gid), std::invalid_argument);
        if (key.region == 40)
        {
            EXPECT_EQ(publication.partition->geometry()->cell_volume(gid), 3);
            EXPECT_EQ(publication.partition->geometry()->cell_nodes(gid).size(), 12);
        }
    }
    double total = 0;
    Teuchos::reduceAll(*model.communicator(), Teuchos::REDUCE_SUM, 1, &volume, &total);
    EXPECT_NEAR(total, 7.5, 1e-14);
    if (model.owned().empty())
    {
        EXPECT_EQ(publication.mesh->num_cells(), 0);
        EXPECT_TRUE(publication.partition->geometry()->regions().empty());
    }
}
TEST(FrozenNative, ConcaveCurvedCopiesReleaseAliasedSourceAndMigrateIntoEmptyOwners)
{
    auto comm = Tpetra::getDefaultComm();
    auto source = std::make_shared<AliasHierarchy>(mixed_source());
    std::weak_ptr<AliasHierarchy::Data> old_data = source->data;
    std::weak_ptr<const MultiRegionMesh> old_geometry = source->native_geometry();
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t c = 0; c < source->native_geometry()->num_cells(); ++c)
            owned.push_back(c);
    std::map<std::uint64_t, SimpleFluid::PLIC::GeometryPoint> normals;
    for (std::uint64_t face = 0; face < source->native_geometry()->num_faces(); ++face)
    {
        auto normal = source->native_geometry()->face_normal(face);
        normals.emplace(face, SimpleFluid::PLIC::GeometryPoint{normal.x, normal.y, normal.z});
    }
    auto bootstrap = ::SimpleFluid::Meshes::AMR::detach_frozen_native(
        *source, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'f'}, comm);
    auto root = bootstrap.root;
    source.reset();
    EXPECT_TRUE(old_data.expired());
    EXPECT_TRUE(old_geometry.expired());
    EXPECT_TRUE(root->source_observation().identity.expired());
    auto model = std::make_shared<const ::SimpleFluid::Meshes::AMR::FrozenNativeModel>(std::move(bootstrap), comm);
    consume(*model, normals);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    if (comm->getRank() == 0)
        marks.push_back({std::find_if(model->owned().begin(), model->owned().end(),
                             [](const auto& entry) { return entry.first.region == 42; })
                             ->first,
            ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    EXPECT_THROW(model->plan_adaptation(marks), std::invalid_argument);
    auto no_op = model->plan_adaptation({});
    EXPECT_EQ(model->apply(no_op)->identity(), model->identity());
    std::vector<int> destinations(model->owned().size(), comm->getSize() - 1);
    auto prepared = model->prepare_migration(destinations);
    EXPECT_EQ(model->owned().empty(), comm->getRank() != 0);
    auto plan = model->plan_migration(prepared);
    EXPECT_EQ(plan.local_application_bytes(), prepared.local_application_bytes());
    EXPECT_EQ(plan.application_bytes(), prepared.application_bytes());
    auto moved = model->apply(plan);
    consume(*moved, normals);
    EXPECT_EQ(moved->owned().empty(), comm->getRank() != comm->getSize() - 1);
    if (comm->getSize() > 1)
    {
        EXPECT_GT(plan.application_bytes(), 0);
        ASSERT_TRUE(plan.native_packet_phase_peak_bytes().has_value());
        EXPECT_LE(*plan.native_packet_phase_peak_bytes(), model->options().byte_budget);
        if (comm->getRank() == 0 || comm->getRank() == comm->getSize() - 1)
            EXPECT_GT(*plan.native_packet_phase_peak_bytes(), 0);
        EXPECT_THROW(moved->apply(plan), std::invalid_argument);
        EXPECT_THROW(moved->plan_migration(prepared), std::invalid_argument);
    }
    EXPECT_EQ(model->owned().empty(), comm->getRank() != 0);
}
TEST(FrozenNative, SharedLayerDictionaryHasBoundedNativeTemplateStorage)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::FrozenNativeOptions options;
    options.byte_budget = 8 * 1024 * 1024;
    const auto make = [&](unsigned layers)
    {
        auto topology = std::make_shared<const ExtrudedTopology>(5,
            Arr<Arr<unsigned>>{{4, 0, 1}, {4, 1, 2}, {4, 2, 3}, {4, 3, 0}}, layers,
            Arr<SemiStructMeshTopo::BoundaryEdge>{{0, 1, "wall"}, {1, 2, "wall"}, {2, 3, "wall"}, {3, 0, "wall"}});
        ArrReal z;
        for (unsigned i = 0; i <= layers; ++i)
            z.push_back(double(i) / layers);
        auto region = extruded_region("layers", topology, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {.5, .5, 0}}, z);
        auto mesh = std::make_shared<MultiRegionMesh>(
            std::vector<MultiRegionMesh::Region>{region}, std::vector<MultiRegionMesh::Interface>{});
        AliasHierarchy source(mesh);
        std::vector<std::uint64_t> owned;
        for (std::uint64_t c = 0; c < 4 * layers; ++c)
            if (c % comm->getSize() == std::uint64_t(comm->getRank()))
                owned.push_back(c);
        auto boot = ::SimpleFluid::Meshes::AMR::detach_frozen_native(
            source, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'s'}, comm, options);
        return std::make_shared<const ::SimpleFluid::Meshes::AMR::FrozenNativeModel>(std::move(boot), comm, options);
    };
    auto small = make(16), model = make(64);
    auto before = small->storage(), storage = model->storage();
    EXPECT_EQ(storage.procedural_fragments, 1);
    EXPECT_GT(storage.shared_native_parts, 0);
    EXPECT_LE(storage.shared_native_parts, 4);
    EXPECT_LT(storage.procedural_bytes, 8192);
    EXPECT_LE(storage.procedural_bytes, before.procedural_bytes + 4096);
    EXPECT_EQ(storage.owned_cells, model->owned().size());
    EXPECT_EQ(storage.explicit_bytes, 0);
    EXPECT_EQ(storage.explicit_payload_bytes, 0);
    EXPECT_GT(storage.canonical_routes, before.canonical_routes);
    EXPECT_LE(storage.canonical_routes, 772);
    EXPECT_LT(storage.routing_bytes, 2300 * model->owned().size() + 65536);
    std::vector<int> destinations(model->owned().size(), 0);
    auto plan = model->plan_migration(destinations);
    std::uint64_t moved = comm->getRank() == 0 ? 0 : model->owned().size(), global_moved = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &moved, &global_moved);
    EXPECT_LE(plan.application_bytes(), 1600 * global_moved + 16384);
    auto candidate = model->apply(plan);
    auto after = candidate->storage();
    EXPECT_EQ(after.explicit_bytes, 0);
    if (comm->getRank() == 0)
    {
        EXPECT_EQ(after.procedural_fragments, 1);
        EXPECT_LE(after.procedural_bytes, storage.procedural_bytes + 4096);
        EXPECT_EQ(after.canonical_routes, 772);
    }
}
TEST(FrozenNative, StaleCanonicalResidentMismatchAndSparseSourcesReject)
{
    auto comm = Tpetra::getDefaultComm();
    auto native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0, .5, 1}, {0, .5, 1}, {0, .5, 1}}});
    auto canonical =
        std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("canonical", native)},
            std::vector<MultiRegionMesh::Interface>{});
    auto resident = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{cartesian_region("resident", {{{0, .5, 1}, {0, .5, 1}, {0, .5, 1}}})},
        std::vector<MultiRegionMesh::Interface>{});
    AliasHierarchy stale(canonical, resident);
    *native = OrthogonalCartesian3D(Vec3D<ArrReal>{{{0, .5, 1}, {0, .5, 1}, {0, .75, 1.5}}});
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        owned.push_back(0);
    EXPECT_ANY_THROW(::SimpleFluid::Meshes::AMR::detach_frozen_native(
        stale, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'x'}, comm));
    auto accepted = std::make_shared<AliasHierarchy>(mixed_source());
    owned.clear();
    if (comm->getRank() == 0)
        for (std::uint64_t c = 0; c < accepted->native_geometry()->num_cells(); ++c)
            owned.push_back(c);
    auto bootstrap = ::SimpleFluid::Meshes::AMR::detach_frozen_native(
        *accepted, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'f'}, comm);
    ::SimpleFluid::Meshes::AMR::FrozenNativeModel model(std::move(bootstrap), comm);
    auto publication = model.publication();
    AliasHierarchy sparse(publication.partition->geometry());
    owned.clear();
    for (const auto& [key, gid] : publication.owned_gids)
        owned.push_back(gid);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::detach_frozen_native(
                     sparse, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'x'}, comm),
        std::invalid_argument);
}
TEST(FrozenNative, ForeignOpaqueSourceBundlesAndAffineFramesRejectCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    AliasHierarchy first(mixed_source()), second(mixed_source());
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t c = 0; c < 13; ++c)
            owned.push_back(c);
    auto a = ::SimpleFluid::Meshes::AMR::detach_frozen_native(
        first, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'f'}, comm);
    auto b = ::SimpleFluid::Meshes::AMR::detach_frozen_native(
        second, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'f'}, comm);
    a.root = b.root;
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::FrozenNativeModel(std::move(a), comm), std::invalid_argument);
    auto geometry = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{cartesian_region("affine", {{{0, .5, 1}, {0, .5, 1}, {0, .5, 1}}})},
        std::vector<MultiRegionMesh::Interface>{});
    auto handle = std::make_shared<MeshHandle<>>(geometry);
    PlanarALEMeshMotion<> motion(handle);
    motion.begin_trial(2, 1);
    motion.accept_trial();
    AliasHierarchy mapped(geometry);
    owned.clear();
    if (comm->getRank() == 0)
        owned.push_back(0);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::detach_frozen_native(
                     mapped, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'a'}, comm),
        std::invalid_argument);
}
TEST(FrozenNative, SubnormalFaceAreaKeepsAcceptedDirectDivisionNormal)
{
    auto comm = Tpetra::getDefaultComm();
    auto geometry = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{cartesian_region("tiny", {{{0, 1e150}, {0, 1e-160}, {0, 1e-160}}})},
        std::vector<MultiRegionMesh::Interface>{});
    AliasHierarchy source(geometry);
    std::map<std::uint64_t, SimpleFluid::PLIC::GeometryPoint> expected;
    bool subnormal = false;
    for (std::uint64_t face = 0; face < geometry->num_faces(); ++face)
    {
        auto n = geometry->face_normal(face);
        expected.emplace(face, SimpleFluid::PLIC::GeometryPoint{n.x, n.y, n.z});
        subnormal |= std::fpclassify(geometry->face_area(face)) == FP_SUBNORMAL;
    }
    ASSERT_TRUE(subnormal);
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        owned.push_back(0);
    auto bootstrap = ::SimpleFluid::Meshes::AMR::detach_frozen_native(
        source, owned, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'t'}, comm);
    ::SimpleFluid::Meshes::AMR::FrozenNativeModel model(std::move(bootstrap), comm);
    auto publication = model.publication();
    for (auto face : publication.partition->plan().face_ids())
    {
        auto actual = publication.partition->geometry()->face_normal(face);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_TRUE(std::isfinite(actual.component(axis)));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.component(axis)),
                std::bit_cast<std::uint64_t>(expected.at(face)[axis]));
        }
    }
}
TEST(FrozenNative, CartesianSingletonPeriodicContractRemainsExplicitlyRejected)
{
    ::SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.block_cells = {1, 2, 2};
    grid.periodic[0] = true;
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::CartesianAMR source(grid), std::invalid_argument);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(grid), std::invalid_argument);
}
} // namespace
