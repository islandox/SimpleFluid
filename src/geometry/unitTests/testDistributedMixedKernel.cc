/** @file testDistributedMixedKernel.cc
 * @brief Detached original-root join aliases, exact inverse and physical remainder.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <array>
#include <gtest/gtest.h>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
void closed(const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& h)
{
    auto native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(h, h.configuration_descriptor());
    auto bound = h.with_partition(native.partition, native.mesh->owned_cell_map());
    EXPECT_TRUE(bound->geometry_unchanged());
    for (std::size_t i = 0; i < native.mesh->num_owned_cells(); ++i)
    {
        const LO cell = LO(i);
        MeshUtils::Vec3 area{};
        double flux = 0;
        native.mesh->visit_cell_faces(cell,
            [&](LO f)
            {
                auto a = native.mesh->face_area_vector(f);
                if (native.mesh->owner_cell(f) != cell)
                    a = a * (-1);
                area = area + a;
                flux += native.mesh->face_centroid(f).dot(a);
            });
        const auto v = native.mesh->cell_volume(cell);
        EXPECT_NEAR(area.norm(), 0, 2e-12);
        EXPECT_NEAR(flux, 3 * v, 2e-12 * v);
    }
    EXPECT_EQ(native.mesh->connectivity_storage_bytes(), 0u);
}
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> scatter(
    const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& h)
{
    std::vector<int> destination;
    std::size_t i = 0;
    for (const auto& [key, leaf] : h.owned())
        destination.push_back(int((i++ + h.communicator()->getRank() + 1) % h.communicator()->getSize()));
    return h.apply(h.plan_migration(destination));
}
TEST(DistributedMixedKernel, InteriorOnlyDescendantsKeepOriginalJoinAliasesAfterSourceReleaseAndSiblingInverse)
{
    auto comm = Tpetra::getDefaultComm();
    auto g = test::mixed_joined_grid();
    const auto original_product = g.product->cell_geometry(0, 0);
    const auto original_poly =
        std::dynamic_pointer_cast<const ::SimpleFluid::Meshes::AMR::PolyhedralAMR>(g.adaptive[0])->cell_geometry(0);
    // Build an admitted matching uniform source directly. Each original root
    // retains its exact family tree; no intermediate crossed seam is published.
    for (unsigned level = 0; level < 2; ++level)
    {
        std::vector<::SimpleFluid::Meshes::AMR::ProductCellAncestry> product;
        for (std::size_t r = 0; r < g.product->runs().size(); ++r)
            for (std::uint64_t i = 0; i < g.product->runs()[r].region.layout().cells; ++i)
                product.push_back(
                    std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(g.product->cell_key(r, i).ancestry));
        g.product = g.product->adapt(product, product);
        auto p = std::dynamic_pointer_cast<const ::SimpleFluid::Meshes::AMR::PolyhedralAMR>(g.adaptive[0]);
        std::vector<std::uint64_t> ids;
        for (std::uint64_t i = 0; i < p->geometry()->num_cells(); ++i)
            ids.push_back(i);
        g.adaptive[0] = p->adapt(ids);
    }
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(g);
    g.product.reset();
    g.adaptive.clear();
    const auto count = source->geometry()->num_cells();
    ASSERT_GT(count, 2u);
    std::vector<std::uint64_t> owned;
    std::uint64_t interior = 0;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        const auto key = source->cell_key(i);
        const auto p = source->cell_geometry(i)->centroid();
        const bool inner = std::holds_alternative<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry) &&
                           p[0] > .25 && p[0] < .75 && p[1] > .25 && p[1] < .75;
        int owner = inner ? std::min(1, comm->getSize() - 1) : 0;
        if (owner == comm->getRank())
        {
            owned.push_back(i);
            interior += inner;
        }
    }
    auto boot = source->detach_owned(owned, {'i', 'n', 't', 'e', 'r', 'i', 'o', 'r'});
    if (comm->getSize() > 1 && comm->getRank() == 1)
    {
        ASSERT_GT(interior, 0u);
        for (const auto& leaf : boot.owned)
        {
            ASSERT_TRUE(leaf.join_recipe);
            EXPECT_FALSE(
                ::SimpleFluid::Meshes::AMR::decode_join_capsule(*leaf.join_recipe, 64 * 1024 * 1024).rules.empty());
            for (const auto& face : boot.kernel->surfaces(leaf))
                EXPECT_FALSE(face.geometry->centroid().x == 1);
        }
    }
    std::weak_ptr<const MultiRegionMesh> old_geometry = source->geometry();
    std::weak_ptr<::SimpleFluid::Meshes::AMR::MixedAMR> old = source;
    source.reset();
    EXPECT_TRUE(old.expired());
    EXPECT_TRUE(old_geometry.expired());
    auto h = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    EXPECT_EQ(h->global_cells(), count);
    closed(*h);
    h = scatter(*h);
    for (int depth = 1; depth >= 0; --depth)
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> families;
        for (const auto& [key, f] : h->families())
        {
            if (f.split != ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic)
                continue;
            bool wanted = std::visit(
                [&](const auto& p)
                {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, ::SimpleFluid::Meshes::AMR::ProductCellAncestry>)
                        return p.base.path.size() == std::size_t(depth) && p.axial.path.size() == std::size_t(depth);
                    else if constexpr (std::is_same_v<T, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry>)
                        return p.path.size() == std::size_t(depth);
                    else
                        return false;
                },
                f.parent.key.ancestry);
            if (wanted)
                families.push_back(key);
        }
        h = h->apply(h->plan_adaptation({}, families));
        closed(*h);
        h = scatter(*h);
    }
    EXPECT_EQ(h->global_cells(), 2u);
    for (const auto& [key, leaf] : h->owned())
        EXPECT_TRUE(h->geometry(leaf)->equivalent(
            leaf.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Product ? *original_product
                                                                                       : *original_poly));
    closed(*h);
}
TEST(DistributedMixedKernel, PartialNamedPatchOfWholeNativeFacesKeepsPhysicalRemainderThroughMigrationAndInverse)
{
    auto comm = Tpetra::getDefaultComm();
    auto product_grid = test::mixed_product_grid();
    for (auto& p : product_grid.xy_nodes)
    {
        p[0] += 1;
        p[1] *= .5;
    }
    product_grid.z_edges = {0, .5, 1};
    auto product = std::make_shared<const ::SimpleFluid::Meshes::AMR::LocalProductAMR>(product_grid);
    ::SimpleFluid::Meshes::AMR::AMRGrid cart;
    cart.block_cells = {2, 2, 2};
    cart.region_id = 9;
    cart.material_id = 5;
    auto coarse = std::make_shared<const ::SimpleFluid::Meshes::AMR::CartesianAMR>(cart);
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid g;
    g.product = product;
    g.physical_region = 7;
    g.adaptive = {coarse};
    const auto& fine = product->runs().at(0).region;
    std::map<double, std::uint64_t> fine_faces, coarse_faces;
    for (std::uint64_t f = 0; f < fine.layout().faces; ++f)
        if (fine.geometry().face_centroid(f).x == 1)
            fine_faces.emplace(fine.geometry().face_centroid(f).z, f);
    ASSERT_EQ(fine_faces.size(), 2u);
    const auto& c = coarse->geometry()->regions().at(0);
    std::visit(
        [&](const auto& r)
        {
            for (std::uint64_t f = 0; f < r.layout().faces; ++f)
                if (r.topology().boundary_id(f) == 1 && r.geometry().face_centroid(f).y < .5)
                    coarse_faces.emplace(r.geometry().face_centroid(f).z, f);
        },
        c);
    ASSERT_EQ(coarse_faces.size(), fine_faces.size());
    MultiProviderNonconformingInterface join;
    join.coarse_region = 1;
    join.coarse_boundary = 1;
    join.fine_patches = {{0, fine.topology().boundary_id(fine_faces.begin()->second), 0}};
    for (const auto& [z, face] : coarse_faces)
        join.faces.push_back({face, {{0, fine_faces.at(z)}}});
    join.partial_coarse_patch = true;
    g.interfaces = {join};
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(g);
    std::string remainder;
    for (std::uint64_t f = 0; f < source->geometry()->num_faces(); ++f)
        if (source->geometry()->is_exterior_face(f) && source->geometry()->face_centroid(f).x == 1 &&
            source->geometry()->face_centroid(f).y > .5)
            remainder = source->geometry()->boundary_batch_name(source->geometry()->boundary_id(f));
    ASSERT_FALSE(remainder.empty());
    std::vector<std::uint64_t> ids;
    for (std::uint64_t i = 0; i < source->geometry()->num_cells(); ++i)
        if (i % comm->getSize() == std::uint64_t(comm->getRank()))
            ids.push_back(i);
    auto boot = source->detach_owned(ids, {'p', 'a', 'r', 't', 'i', 'a', 'l'});
    std::weak_ptr<const MultiRegionMesh> weak = source->geometry();
    source.reset();
    g.product.reset();
    g.adaptive.clear();
    product.reset();
    coarse.reset();
    EXPECT_TRUE(weak.expired());
    auto h = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    auto check_remainder = [&](const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& value)
    {
        auto native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(value, value.configuration_descriptor());
        double exterior = 0, internal = 0;
        const auto& mesh = *native.partition->geometry();
        for (std::size_t local_face = 0; local_face < native.mesh->num_owned_faces(); ++local_face)
        {
            const auto f = native.mesh->face_global_id(LO(local_face));
            const auto p = mesh.face_centroid(f);
            if (p.x != 1)
                continue;
            if (mesh.is_exterior_face(f))
            {
                if (p.y > .5)
                {
                    EXPECT_EQ(mesh.boundary_batch_name(mesh.boundary_id(f)), remainder);
                    exterior += mesh.face_area(f);
                }
            }
            else
                internal += mesh.face_area(f);
        }
        double local[2]{exterior, internal}, global[2]{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, global);
        EXPECT_DOUBLE_EQ(global[0], .5);
        EXPECT_DOUBLE_EQ(global[1], .5);
    };
    check_remainder(*h);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    for (const auto& [k, l] : h->owned())
        if (l.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Product)
            marks.push_back({k, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base});
    auto refined = h->apply(h->plan_adaptation(marks));
    refined = scatter(*refined);
    closed(*refined);
    check_remainder(*refined);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> inverse;
    for (const auto& [k, f] : refined->families())
        if (f.parent.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Product &&
            f.split == ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base)
            inverse.push_back(k);
    auto restored = refined->apply(refined->plan_adaptation({}, inverse));
    EXPECT_EQ(restored->global_cells(), h->global_cells());
    closed(*restored);
    check_remainder(*restored);
}
TEST(DistributedMixedKernel, RefinedCartesianSourceKeepsDistinctAcceptedDestinationsOnCoarsenedAncestorFaces)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    auto original = std::make_shared<const ::SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
    auto level_one = original->adapt({0});
    auto refined = level_one->adapt({0});
    ::SimpleFluid::Meshes::AMR::MixedAMRGrid mixed;
    mixed.adaptive = {refined};
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(mixed);
    std::map<std::string, std::array<double, 4>> expected;
    for (std::uint64_t f = 0; f < source->geometry()->num_faces(); ++f)
        if (source->geometry()->is_exterior_face(f))
        {
            const auto area = source->geometry()->face_area(f);
            const auto p = source->geometry()->face_centroid(f);
            auto& value = expected[source->geometry()->boundary_batch_name(source->geometry()->boundary_id(f))];
            value[0] += area;
            value[1] += area * p.x;
            value[2] += area * p.y;
            value[3] += area * p.z;
        }
    ASSERT_GT(expected.size(), 6u);
    std::vector<std::uint64_t> ids;
    for (std::uint64_t i = 0; i < source->geometry()->num_cells(); ++i)
        if (i % comm->getSize() == std::uint64_t(comm->getRank()))
            ids.push_back(i);
    auto boot = source->detach_owned(ids, {'b', 'o', 'u', 'n', 'd', 'a', 'r', 'y'});
    std::weak_ptr<const MultiRegionMesh> weak = source->geometry();
    source.reset();
    mixed.adaptive.clear();
    original.reset();
    level_one.reset();
    refined.reset();
    EXPECT_TRUE(weak.expired());
    auto h = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    h = scatter(*h);
    for (int level = 1; level >= 0; --level)
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> restore;
        for (const auto& [key, family] : h->families())
            if (const auto* key =
                    std::get_if<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(&family.parent.key.ancestry);
                key && key->level == unsigned(level))
                restore.push_back(family.key);
        h = h->apply(h->plan_adaptation({}, restore));
        closed(*h);
        h = scatter(*h);
    }
    EXPECT_EQ(h->global_cells(), 8u);
    auto native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(*h, h->configuration_descriptor());
    std::map<std::string, std::array<double, 4>> local;
    const auto& mesh = *native.partition->geometry();
    for (std::size_t i = 0; i < native.mesh->num_owned_faces(); ++i)
    {
        const auto f = native.mesh->face_global_id(LO(i));
        if (mesh.is_exterior_face(f))
        {
            const auto area = mesh.face_area(f);
            const auto p = mesh.face_centroid(f);
            auto& value = local[mesh.boundary_batch_name(mesh.boundary_id(f))];
            value[0] += area;
            value[1] += area * p.x;
            value[2] += area * p.y;
            value[3] += area * p.z;
        }
    }
    for (const auto& [name, measure] : expected)
    {
        const auto value = local[name];
        std::array<double, 4> global{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 4, value.data(), global.data());
        for (unsigned q = 0; q < 4; ++q)
            EXPECT_DOUBLE_EQ(global[q], measure[q]) << name << " moment " << q;
        local.erase(name);
    }
    EXPECT_TRUE(local.empty());
}
TEST(DistributedMixedKernel, ContactClosureKeepsRetainedPolyhedralSupportsSeparateFromMappedAuthorityCopies)
{
    const auto comm = Tpetra::getDefaultComm();
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid());
    auto source = root->adapt({test::first_product(*root)});
    std::vector<std::uint64_t> ids;
    for (std::uint64_t cell = 0; cell < source->geometry()->num_cells(); ++cell)
        if (cell % comm->getSize() == std::uint64_t(comm->getRank()))
            ids.push_back(cell);
    auto boot = source->detach_owned(ids, {'c', 'l', 'o', 's', 'u', 'r', 'e', '_', 's', 'u', 'p', 'p', 'o', 'r', 't'});
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveCellKey,
        std::vector<std::pair<::SimpleFluid::Meshes::AMR::AdaptiveBytes, std::string>>>
        retained;
    std::uint64_t local_poly = 0, global_poly = 0;
    for (const auto& leaf : boot.owned)
        if (leaf.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral)
        {
            ++local_poly;
            EXPECT_FALSE(leaf.surfaces.empty());
            for (const auto& use : leaf.surfaces)
                retained[leaf.key].emplace_back(use.key, use.boundary);
            const auto mapped = boot.kernel->surfaces(leaf);
            EXPECT_EQ(mapped.size(), leaf.surfaces.size());
            bool rewritten = false;
            for (std::size_t f = 0; f < std::min(mapped.size(), leaf.surfaces.size()); ++f)
                rewritten = rewritten || mapped[f].key != leaf.surfaces[f].key;
            EXPECT_TRUE(rewritten);
        }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_poly, &global_poly);
    ASSERT_GT(global_poly, 0u);
    ::SimpleFluid::Meshes::AMR::AdaptiveClosureOptions options;
    // Repeat the query: accepted source support identities and destinations
    // must never become rewritten canonical keys inside retained records.
    for (unsigned repeat = 0; repeat < 2; ++repeat)
    {
        EXPECT_TRUE(
            ::SimpleFluid::Meshes::AMR::required_adaptive_refinements(boot.owned, *boot.kernel, comm, options).empty());
        for (const auto& leaf : boot.owned)
            if (leaf.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral)
            {
                const auto& original = retained.at(leaf.key);
                EXPECT_EQ(original.size(), leaf.surfaces.size());
                for (std::size_t f = 0; f < std::min(original.size(), leaf.surfaces.size()); ++f)
                {
                    EXPECT_EQ(original[f].first, leaf.surfaces[f].key);
                    EXPECT_EQ(original[f].second, leaf.surfaces[f].boundary);
                }
            }
    }
    auto h = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    closed(*h);
}
} // namespace
