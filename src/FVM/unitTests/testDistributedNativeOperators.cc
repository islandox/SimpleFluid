/** @file testDistributedNativeOperators.cc
 * @brief Actual native FVM caches after distributed owner migration/adaptation.
 */
#include "FVM/Operators.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/unitTests/polyhedral_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using LO = DefaultTpetraTypes::local_ordinal_type;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
void expect_gradient_and_flux(const SP<const MeshHandle<>>& mesh, VectorCellFieldStored<>& gradient)
{
    const auto comm = mesh->owned_cell_map()->getComm();
    const MeshUtils::Vec3 expected{2, -.75, .4};
    // FVM writes owned storage; overlap values require an explicit import.
    D::collective(*comm,
        [&]
        {
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                EXPECT_NEAR((gradient.value(LO(c)) - expected).norm(), 0, 2e-10);
        });
    gradient.sync_ghosts();
    D::collective(*comm,
        [&]
        {
            for (std::size_t c = 0; c < mesh->num_cells(); ++c)
                EXPECT_NEAR((gradient.local_value(LO(c)) - expected).norm(), 0, 2e-10);
        });
    ScalarFaceFieldStored<> flux(mesh, "linear_velocity_integrated_flux");
    std::array<double, 2> local{}, global{};
    D::collective(*comm,
        [&]
        {
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            {
                const auto face = LO(f);
                const auto value = mesh->face_centroid(face).dot(mesh->face_area_vector(face));
                flux.set_owned_value(face, value);
                if (mesh->is_exterior_face(face))
                    local[1] += value;
            }
        });
    flux.sync_ghosts();
    D::collective(*comm,
        [&]
        {
            const auto divergence = FVM::cell_divergence_from_fluxes<DefaultTpetraTypes>(*mesh, flux);
            if (divergence.size() != mesh->num_owned_cells())
                throw std::invalid_argument("Native flux divergence must have one row per owned cell.");
            for (std::size_t c = 0; c < divergence.size(); ++c)
            {
                const auto volume = mesh->cell_volume(LO(c));
                local[0] += volume;
                EXPECT_NEAR(divergence[c] * volume, 3 * volume, 2e-10 * volume);
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local.data(), global.data());
    EXPECT_GT(global[0], 0);
    EXPECT_NEAR(global[1], 3 * global[0], 2e-10 * global[0]);
}
void check_native_operators(const SP<const MeshHandle<>>& mesh)
{
    const auto comm = mesh->owned_cell_map()->getComm();
    ScalarCellFieldStored<> scalar(mesh, "native_linear");
    VectorCellFieldStored<> gradient(mesh, "native_gradient");
    auto linear = [](MeshUtils::Vec3 p) { return 1 + 2 * p.x - .75 * p.y + .4 * p.z; };
    D::collective(*comm,
        [&]
        {
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                scalar.set_owned_value(LO(c), linear(mesh->cell_centroid(LO(c))));
        });
    scalar.sync_ghosts();
    D::collective(*comm,
        [&]
        {
            for (std::size_t c = 0; c < mesh->num_cells(); ++c)
                EXPECT_DOUBLE_EQ(scalar.local_value(LO(c)), linear(mesh->cell_centroid(LO(c))));
            auto boundary = [&](int b, std::size_t i)
            { return linear(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids.at(i))); };
            auto condition = [&](int b, std::size_t i)
            { return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary(b, i)}; };
            FVM::CellGradientCache<DefaultTpetraTypes, MeshHandle<>> cache(mesh);
            FVM::cell_gradient(scalar, condition, boundary, gradient, cache);
        });
    expect_gradient_and_flux(mesh, gradient);
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(mesh->has_materialized_indexer());
}
struct Source
{
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> hierarchy;
    ::SimpleFluid::Meshes::AMR::AdaptiveBytes descriptor;
};
Source detached(unsigned family)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::AdaptiveBootstrap bootstrap;
    ::SimpleFluid::Meshes::AMR::AdaptiveBytes descriptor;
    D::collective(*comm,
        [&]
        {
            if (family == 0)
            {
                ::SimpleFluid::Meshes::AMR::AMRGrid grid;
                grid.block_cells = {2, 2, 2};
                grid.max_level = 2;
                auto source = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
                std::weak_ptr<const MultiRegionMesh> weak = source->geometry();
                std::vector<std::uint64_t> owned;
                if (comm->getRank() == 0)
                    for (std::uint64_t i = 0; i < 8; ++i)
                        owned.push_back(i);
                bootstrap = source->detach_owned(owned, {'n', 'a', 't', 'i', 'v', 'e'});
                source.reset();
                EXPECT_TRUE(weak.expired());
            }
            else if (family == 1)
            {
                ::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid grid;
                grid.xy_nodes = {{.1, .1, 0}, {.3, .1, 0}, {.3, .3, 0}, {.1, .3, 0}};
                grid.base_cells = {{0, 1, 2, 3}};
                grid.boundaries = {{0, 1, "wall"}, {1, 2, "wall"}, {2, 3, "wall"}, {3, 0, "wall"}};
                grid.z_edges = {.1, .3, .5};
                auto source = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid);
                std::weak_ptr<const MultiRegionMesh> weak = source->geometry();
                std::vector<std::uint64_t> owned;
                if (comm->getRank() == 0)
                    owned = {0, 1};
                bootstrap = source->detach_owned(owned, {'n', 'a', 't', 'i', 'v', 'e'});
                source.reset();
                EXPECT_TRUE(weak.expired());
            }
            else
            {
                auto source = std::make_shared<::SimpleFluid::Meshes::AMR::PolyhedralAMR>(test::polyhedral_grid());
                std::weak_ptr<const MultiRegionMesh> weak = source->geometry();
                std::vector<std::uint64_t> owned;
                if (comm->getRank() == 0)
                    owned = {0, 1};
                bootstrap = source->detach_owned(owned, {'n', 'a', 't', 'i', 'v', 'e'});
                source.reset();
                EXPECT_TRUE(weak.expired());
            }
            descriptor = bootstrap.kernel->descriptor();
        });
    return {std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
                std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm),
        std::move(descriptor)};
}
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> exercise(Source source, bool refine)
{
    const auto comm = Tpetra::getDefaultComm();
    auto publication = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*source.hierarchy, source.descriptor);
    source.hierarchy = source.hierarchy->with_partition(publication.partition, publication.mesh->owned_cell_map());
    const auto source_observation = source.hierarchy->observation();
    std::vector<int> destinations;
    std::size_t ordinal = 0;
    for (; ordinal < source.hierarchy->owned_leaves().size(); ++ordinal)
        destinations.push_back(int(ordinal % std::min(2, comm->getSize())));
    auto plan = source.hierarchy->plan_migration(destinations);
    auto candidate = source.hierarchy->apply(plan);
    auto moved = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*candidate, source.descriptor);
    auto bound = candidate->with_partition(moved.partition, moved.mesh->owned_cell_map());
    bound = source.hierarchy->apply(plan, bound);
    EXPECT_EQ(source.hierarchy->observation(), source_observation);
    if (refine)
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
        for (const auto& leaf : bound->owned_leaves())
            marks.push_back({leaf.key, leaf.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Product
                                      ? ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial
                                      : ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
        auto adaptation = bound->plan_adaptation(marks);
        auto next = bound->apply(adaptation);
        auto native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*next, source.descriptor);
        auto accepted = next->with_partition(native.partition, native.mesh->owned_cell_map());
        bound = bound->apply(adaptation, accepted);
        moved = std::move(native);
    }
    const auto mesh = moved.mesh;
    check_native_operators(mesh);
    int local_empty = mesh->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    if (comm->getRank() == 0)
        std::cout << "native FVM owned empty ranks=" << empty << " global cells=" << bound->global_cells() << '\n';
    return bound;
}
TEST(DistributedNativeOperators, CartesianActualOwnerMigrationAndRefinementUseFvmGradientCache)
{
    (void) exercise(detached(0), true);
}
TEST(DistributedNativeOperators, QualifiedCartesianPublicationFeedsNativeGradientAndFlux)
{
    namespace A = SimpleFluid::Meshes::AMR;
    const auto source = detached(0);
    const auto& comm = *source.hierarchy->communicator();
    const auto publication = A::publish_adaptive_native(*source.hierarchy, A::native_publication_schema(*source.hierarchy));
    SP<const MeshHandle<>> mesh;
    Teuchos::RCP<const DefaultTpetraTypes::map_type> map;
    D::collective(comm,
        [&]
        {
            mesh = publication.mesh();
            if (!mesh)
                throw std::invalid_argument("Qualified native operator publication has no MeshHandle.");
            map = mesh->owned_cell_map();
            if (publication.key_registry_storage().owns_key_maps ||
                !D::NativePublicationSession::receipt_matches(D::NativePublicationSession::capture(*source.hierarchy),
                    publication.partition(), map))
                throw std::invalid_argument("Native operator fixture requires the qualified Cartesian result.");
        });
    const auto before_bind = D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions;
    const auto bound = source.hierarchy->with_partition(publication.partition(), map);
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, before_bind);
    EXPECT_EQ(bound->native_partition(), publication.partition());
    const auto oracle = A::assemble_adaptive_native(*source.hierarchy, source.descriptor);
    D::collective(comm,
        [&]
        {
            if (!oracle.mesh || !oracle.partition)
                throw std::invalid_argument("Full native verification has no operator mesh or partition.");
            const auto& plan = publication.partition()->plan();
            const auto& reference = oracle.partition->plan();
            if (plan.cell_ids() != reference.cell_ids() || plan.face_ids() != reference.face_ids() ||
                plan.node_ids() != reference.node_ids())
                throw std::invalid_argument("Qualified native operator routes differ from full verification.");
            EXPECT_EQ(bound->global_cells(), 8u);
            EXPECT_EQ(mesh->num_owned_cells(), comm.getRank() == 0 ? 8u : 0u);
            for (std::size_t c = 0; c < mesh->num_cells(); ++c)
            {
                EXPECT_EQ(mesh->cell_centroid(LO(c)), oracle.mesh->cell_centroid(LO(c)));
                EXPECT_DOUBLE_EQ(mesh->cell_volume(LO(c)), oracle.mesh->cell_volume(LO(c)));
            }
            for (std::size_t f = 0; f < mesh->num_faces(); ++f)
            {
                EXPECT_EQ(mesh->face_centroid(LO(f)), oracle.mesh->face_centroid(LO(f)));
                EXPECT_EQ(mesh->face_area_vector(LO(f)), oracle.mesh->face_area_vector(LO(f)));
                EXPECT_EQ(mesh->is_exterior_face(LO(f)), oracle.mesh->is_exterior_face(LO(f)));
            }
        });
    check_native_operators(mesh);
}
TEST(DistributedNativeOperators, ExactDecimalProductRetainsNativeMetricsAndFvmCacheAcrossMigration)
{
    (void) exercise(detached(1), false);
}
TEST(DistributedNativeOperators, CertifiedLocalPolyShardRetainsCanonicalFluxAndFvmCache)
{
    (void) exercise(detached(2), false);
}
TEST(DistributedNativeOperators, TypedOwnedMapBindingRejectsReorderedAndForeignGidsCollectively)
{
    using Pack = DefaultTpetraTypes;
    using GO = Pack::global_ordinal_type;
    auto source = detached(0);
    const auto comm = source.hierarchy->communicator();
    const auto publication = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*source.hierarchy, source.descriptor);
    const auto map = publication.mesh->owned_cell_map();
    auto bound = source.hierarchy->with_partition(publication.partition, map);
    const auto stamp = bound->observation();
    EXPECT_EQ(bound->with_partition(publication.partition, map)->observation(), stamp);
    std::vector<GO> gids;
    for (std::size_t i = 0; i < map->getLocalNumElements(); ++i)
        gids.push_back(map->getGlobalElement(LO(i)));
    if (comm->getRank() == 0)
        std::reverse(gids.begin(), gids.end());
    auto reordered = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::arrayViewFromVector(gids), GO{}, comm));
    EXPECT_THROW(bound->with_partition(publication.partition, reordered), std::invalid_argument);
    D::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                if (gids.empty())
                    throw std::invalid_argument("Foreign map fixture requires a rank-zero owned GID.");
                std::reverse(gids.begin(), gids.end());
                gids.back() = GO(source.hierarchy->global_cells());
            }
        });
    auto foreign = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::arrayViewFromVector(gids), GO{}, comm));
    EXPECT_THROW(bound->with_partition(publication.partition, foreign), std::invalid_argument);
    EXPECT_EQ(bound->observation(), stamp);
}
TEST(DistributedNativeOperators, RepresentableMicroscopicPolyGeometryPublishesNativeAreas)
{
    const auto comm = Tpetra::getDefaultComm();
    constexpr double side = 1e-100;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> owned;
    if (comm->getRank() == 0)
    {
        ::SimpleFluid::Meshes::AMR::AdaptiveLeaf leaf;
        leaf.key = {17, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{(std::uint64_t{1} << 60) + 3, {}}};
        leaf.material = 23;
        leaf.domain = {'t', 'i', 'n', 'y'};
        leaf.provider = ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral;
        leaf.geometry = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box(
            {0, 0, 0}, {side, side, side});
        for (std::size_t face = 0; face < leaf.geometry->num_faces(); ++face)
        {
            ::SimpleFluid::Meshes::AMR::AdaptiveSurface surface;
            surface.key = {'t', static_cast<char>(face)};
            surface.geometry = leaf.geometry->face_geometry(face);
            surface.exterior = true;
            surface.boundary = "tiny_wall";
            surface.native_face = face;
            leaf.surfaces.push_back(std::move(surface));
        }
        owned.push_back(std::move(leaf));
    }
    const auto kernel = ::SimpleFluid::Meshes::AMR::convex_adaptive_kernel(2, 0);
    auto hierarchy = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(owned), std::vector<::SimpleFluid::Meshes::AMR::AdaptiveFamily>{}, kernel, comm);
    const auto publication = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*hierarchy, kernel->descriptor());
    const auto bound = hierarchy->with_partition(publication.partition, publication.mesh->owned_cell_map());
    EXPECT_TRUE(bound->geometry_unchanged());
    for (std::size_t c = 0; c < publication.mesh->num_owned_cells(); ++c)
    {
        const auto volume = publication.mesh->cell_volume(LO(c));
        EXPECT_GT(volume, 0);
        EXPECT_NEAR(volume, 1e-300, 8 * std::numeric_limits<double>::epsilon() * 1e-300);
    }
    for (std::size_t f = 0; f < publication.mesh->num_owned_faces(); ++f)
    {
        const auto area = publication.mesh->face_area(LO(f));
        EXPECT_GT(area, 0);
        EXPECT_NEAR(area, 1e-200, 8 * std::numeric_limits<double>::epsilon() * 1e-200);
        EXPECT_NEAR(publication.mesh->face_normal(LO(f)).norm(), 1, 8 * std::numeric_limits<double>::epsilon());
    }
    EXPECT_EQ(publication.mesh->connectivity_storage_bytes(), 0u);
}
TEST(DistributedNativeOperators, DifferentReceiversRequestPartialProductRunsWithIndependentNativeAddresses)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Different source/receiver recipe selections require MPI.";
    ::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid grid;
    for (unsigned j = 0; j < 2; ++j)
        for (unsigned i = 0; i < 8; ++i)
            grid.xy_nodes.push_back({double(i), double(j), 0});
    for (unsigned i = 0; i < 7; ++i)
        grid.base_cells.push_back({i, i + 1, i + 9, i + 8});
    for (unsigned i = 0; i < 7; ++i)
    {
        grid.boundaries.push_back({i, i + 1, "wall"});
        grid.boundaries.push_back({i + 9, i + 8, "wall"});
    }
    grid.boundaries.push_back({7, 15, "wall"});
    grid.boundaries.push_back({8, 0, "wall"});
    for (unsigned k = 0; k <= 8; ++k)
        grid.z_edges.push_back(double(k) / 8);
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t z = 0; z < 8; ++z)
        for (std::uint64_t base = 0; base < 7; ++base)
        {
            int owner = 0;
            if (base < 2 && z < 4)
                owner = 1;
            if (base >= 5 && z >= 4)
                owner = std::min(2, comm->getSize() - 1);
            if (owner == comm->getRank())
                owned.push_back(z * 7 + base);
        }
    auto bootstrap = source->detach_owned(owned, {'s', 'u', 'b', 's', 'e', 't'});
    source.reset();
    const auto descriptor = bootstrap.kernel->descriptor();
    auto hierarchy = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    const auto publication = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*hierarchy, descriptor);
    const auto bound = hierarchy->with_partition(publication.partition, publication.mesh->owned_cell_map());
    EXPECT_EQ(bound->global_cells(), 56u);
    const auto mesh = publication.mesh;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> requested_full;
    D::collective(*comm,
        [&]
        {
            for (std::size_t c = mesh->num_owned_cells(); c < mesh->num_cells(); ++c)
                requested_full.push_back(publication.local_keys.at(mesh->cell_global_id(LO(c))));
        });
    const auto original_owners = hierarchy->lookup_cells(requested_full);
    std::size_t from_zero = 0, min_z = 8, max_z = 0;
    D::collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < requested_full.size(); ++i)
                if (original_owners.at(i).owner == 0)
                {
                    ++from_zero;
                    const auto z =
                        std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(requested_full[i].ancestry).axial.root;
                    min_z = std::min(min_z, std::size_t(z));
                    max_z = std::max(max_z, std::size_t(z));
                }
        });
    if (comm->getRank() == 1 || (comm->getSize() > 2 && comm->getRank() == 2))
    {
        EXPECT_GT(from_zero, 0u);
        EXPECT_LT(from_zero, 40u);
        if (comm->getSize() > 2)
        {
            if (comm->getRank() == 1)
                EXPECT_EQ(max_z, 4u);
            if (comm->getRank() == 2)
                EXPECT_EQ(min_z, 3u);
        }
    }
    ScalarCellFieldStored<> scalar(mesh, "subset_linear");
    VectorCellFieldStored<> gradient(mesh, "subset_gradient");
    const auto linear = [](MeshUtils::Vec3 p) { return 2 * p.x - .75 * p.y + .4 * p.z; };
    D::collective(*comm,
        [&]
        {
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                scalar.set_owned_value(LO(c), linear(mesh->cell_centroid(LO(c))));
        });
    scalar.sync_ghosts();
    D::collective(*comm,
        [&]
        {
            auto boundary = [&](int b, std::size_t i)
            { return linear(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids.at(i))); };
            auto condition = [&](int b, std::size_t i)
            { return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary(b, i)}; };
            FVM::CellGradientCache<DefaultTpetraTypes, MeshHandle<>> cache(mesh);
            FVM::cell_gradient(scalar, condition, boundary, gradient, cache);
        });
    expect_gradient_and_flux(mesh, gradient);
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(mesh->has_materialized_indexer());
    int local_empty = mesh->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 3));
}
} // namespace
