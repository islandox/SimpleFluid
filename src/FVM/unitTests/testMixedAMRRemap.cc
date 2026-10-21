/** @file testMixedAMRRemap.cc
 * @brief Sparse conservative native transfer across mixed product and polyhedral hierarchies.
 */
#include "FVM/ConservativeAMRRemap.hh"
#include "fields/FieldStored.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <iostream>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::VoF;
using Handle = MeshHandle<>;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Bound
{
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::MixedAMR> hierarchy;
    SP<Handle> mesh;
    explicit Bound(std::shared_ptr<const ::SimpleFluid::Meshes::AMR::MixedAMR> h) : hierarchy(std::move(h))
    {
        CompositePartitionOptions options;
        options.policy = CompositePartitionOptions::Policy::ContiguousCanonical;
        options.target_cells_per_fragment = 1;
        auto p = MeshPartitioner<DefaultTpetraTypes>::partition(
            CompositeMeshSource(hierarchy->geometry()), options, Tpetra::getDefaultComm());
        hierarchy = hierarchy->with_partition(std::make_shared<const CompositePartition<>>(p));
        mesh = std::make_shared<Handle>(*hierarchy->native_partition());
    }
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> cells() const
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> result;
        for (std::uint64_t c = 0; c < hierarchy->geometry()->num_cells(); ++c)
            result.push_back({hierarchy->cell_key(c), c, hierarchy->cell_material(c), hierarchy->cell_geometry(c)});
        return result;
    }
};
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> plan(const Bound& a, const Bound& b)
{
    return ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(a.hierarchy->observation(),
        b.hierarchy->observation(), a.cells(), b.cells(), a.hierarchy->overlap_relations(*b.hierarchy));
}
AMRRemapPhases phases(const Bound& b)
{
    AMRRemapPhases result;
    const GeometryHalfspace plane{{1, 0, 1}, {0, 0, 0}, 1.25};
    for (size_t c = 0; c < b.mesh->num_owned_cells(); ++c)
    {
        auto gid = b.mesh->cell_global_id(LO(c));
        auto cut = b.hierarchy->cell_geometry(gid)->clip(plane);
        result.gas_volume.push_back(cut.inside ? cut.inside->moments().volume.as_double() : 0);
        result.carrier_volume.push_back(cut.outside ? cut.outside->moments().volume.as_double() : 0);
        result.gas_halfspace.push_back(plane);
    }
    return result;
}
std::vector<ConservedQuantity> amounts(const AMRRemapPhases& p)
{
    std::vector<ConservedQuantity> q{{"species", InventoryCarrier::GasVolume, {}, true},
        {"number", InventoryCarrier::LiquidVolume, {}, true}, {"moles", InventoryCarrier::LiquidVolume, {}, true},
        {"mass", InventoryCarrier::MixtureMass, {}, true}, {"momentum", InventoryCarrier::MixtureMass, {}, false}};
    for (size_t c = 0; c < p.gas_volume.size(); ++c)
    {
        auto g = p.gas_volume[c], l = p.carrier_volume[c], m = 1000 * l + 2 * g;
        q[0].amount.push_back(7 * g);
        q[1].amount.push_back(11 * l);
        q[2].amount.push_back(3 * l);
        q[3].amount.push_back(m);
        q[4].amount.push_back(-.2 * m);
    }
    return q;
}
void native_fields(const Bound& b)
{
    ScalarCellFieldStored<> field(b.mesh, "mixed_gid");
    for (size_t c = 0; c < b.mesh->num_owned_cells(); ++c)
        field.set_owned_value(LO(c), double(b.mesh->cell_global_id(LO(c))) + .5);
    field.sync_ghosts();
    for (size_t c = 0; c < b.mesh->num_cells(); ++c)
        EXPECT_EQ(field.local_value(LO(c)), double(b.mesh->cell_global_id(LO(c))) + .5);
    EXPECT_EQ(b.mesh->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(b.mesh->has_materialized_indexer());
    for (size_t c = 0; c < b.mesh->num_owned_cells(); ++c)
    {
        MeshUtils::Vec3 sum{};
        double area = 0;
        b.mesh->visit_cell_faces(LO(c),
            [&](LO f)
            {
                sum = sum + b.mesh->face_area_vector_outward(f, LO(c));
                area += b.mesh->face_area(f);
            });
        EXPECT_LE(sum.norm(), 64 * std::numeric_limits<double>::epsilon() * area);
    }
}
TEST(MixedAMRRemapTest, ProductPolyhedralTiltedPhaseAmountsAndReverseHaveActualEmptyOwners)
{
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid(3));
    auto k = test::first_product(*root);
    Bound source(root), target(root->adapt({k}, {k}));
    native_fields(source);
    native_fields(target);
    ConservativeAMRRemap forward(source.mesh, target.mesh, plan(source, target)),
        reverse(target.mesh, source.mesh, plan(target, source));
    auto physical = phases(source);
    auto inventory = amounts(physical);
    AMRRemapOutput out, restored;
    EXPECT_NO_THROW((void) forward.apply(physical, inventory, out, 1000, 2));
    for (size_t c = 0; c < out.gas_volume.size(); ++c)
        if (out.inventories[2].amount[c] > 0)
            EXPECT_NEAR(out.inventories[1].amount[c] / out.inventories[2].amount[c], 11. / 3, 2e-14);
    AMRRemapPhases next{out.gas_volume, out.carrier_volume,
        std::vector<GeometryHalfspace>(out.gas_volume.size(), {{1, 0, 1}, {0, 0, 0}, 1.25})};
    EXPECT_NO_THROW((void) reverse.apply(next, out.inventories, restored, 1000, 2));
    for (size_t c = 0; c < physical.gas_volume.size(); ++c)
    {
        EXPECT_NEAR(restored.gas_volume[c], physical.gas_volume[c], 2e-15);
        EXPECT_NEAR(restored.carrier_volume[c], physical.carrier_volume[c], 2e-15);
        for (size_t q = 0; q < inventory.size(); ++q)
            EXPECT_NEAR(restored.inventories[q].amount[c], inventory[q].amount[c], 2e-11);
    }
    int local = source.mesh->num_owned_cells() == 0, empty = 0, owner = target.mesh->num_owned_cells() > 0, owners = 0;
    auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &empty);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &owner, &owners);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    EXPECT_EQ(owners, comm->getSize());
    if (comm->getRank() == 0)
        std::cout << "mixed source/reverse-target empty=" << empty << " refined owners=" << owners << '\n';
}
TEST(MixedAMRRemapTest, RankLocalInvalidPlanesAndMapReplacementPreserveAcceptedOutputs)
{
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid());
    auto k = test::first_product(*root);
    Bound source(root), target(root->adapt({k}, {k}));
    ConservativeAMRRemap remap(source.mesh, target.mesh, plan(source, target));
    auto physical = phases(source);
    auto q = amounts(physical);
    AMRRemapOutput out;
    (void) remap.apply(physical, q, out, 1000, 2);
    const auto accepted = out;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !physical.gas_halfspace.empty())
        physical.gas_halfspace[0].offset = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) remap.apply(physical, q, out, 1000, 2), std::invalid_argument);
    EXPECT_EQ(out.gas_volume, accepted.gas_volume);
    EXPECT_EQ(out.carrier_volume, accepted.carrier_volume);
    Bound replacement(root);
    *source.mesh = *replacement.mesh;
    EXPECT_THROW((void) remap.apply(phases(replacement), q, out, 1000, 2), std::invalid_argument);
    for (size_t i = 0; i < out.inventories.size(); ++i)
        EXPECT_EQ(out.inventories[i].amount, accepted.inventories[i].amount);
}
TEST(MixedAMRRemapTest, ExplicitPureMicroscopicAndSubnormalInventoriesRemainPaired)
{
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid());
    auto k = test::first_product(*root);
    Bound source(root), target(root->adapt({k}, {k}));
    ConservativeAMRRemap remap(source.mesh, target.mesh, plan(source, target));
    for (double tiny : {1e-200, 8 * std::numeric_limits<double>::denorm_min()})
    {
        AMRRemapPhases p;
        for (size_t c = 0; c < source.mesh->num_owned_cells(); ++c)
        {
            p.gas_volume.push_back(tiny);
            p.carrier_volume.push_back(source.mesh->cell_volume(LO(c)));
            p.gas_halfspace.push_back({});
            p.geometry_declaration.push_back(AMRPhaseGeometry::PureCarrier);
        }
        auto q = amounts(p);
        q[1].carrier = InventoryCarrier::GasVolume;
        q[2].carrier = InventoryCarrier::GasVolume;
        for (size_t c = 0; c < p.gas_volume.size(); ++c)
        {
            q[1].amount[c] = tiny;
            q[2].amount[c] = tiny;
        }
        AMRRemapOutput out;
        EXPECT_NO_THROW((void) remap.apply(p, q, out, 1000, 2));
        for (size_t c = 0; c < out.gas_volume.size(); ++c)
            EXPECT_EQ(out.inventories[1].amount[c], out.inventories[2].amount[c]);
    }
}
TEST(MixedAMRRemapTest, RemoteInactiveSiblingMetadataStillAuthenticatesForBothFamilies)
{
    if (Tpetra::getDefaultComm()->getSize() < 2)
        GTEST_SKIP() << "Remote owner checks require MPI.";
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid());
    auto k = test::first_product(*root);
    Bound source(root), target(root->adapt({k}, {k}));
    auto from = source.cells(), to = target.cells();
    const auto relations = source.hierarchy->overlap_relations(*target.hierarchy);
    EXPECT_NO_THROW((void) ConservativeAMRRemap(source.mesh, target.mesh, plan(source, target)));
    if (Tpetra::getDefaultComm()->getRank() == 0)
    {
        for (auto& c : from)
            c.material ^= 1;
        for (auto& c : to)
            c.material ^= 1;
    }
    auto altered = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        source.hierarchy->observation(), target.hierarchy->observation(), from, to, relations);
    EXPECT_THROW((void) ConservativeAMRRemap(source.mesh, target.mesh, altered), std::invalid_argument);
}
TEST(MixedAMRRemapTest, StaticConvexInventoryAndLargeRootIdentitySurviveProductAdaptation)
{
    auto grid = test::mixed_joined_grid();
    const auto stationary = grid.adaptive.front();
    grid.adaptive.clear();
    constexpr std::uint64_t provider_root = (std::uint64_t{1} << 60) + 123;
    grid.stationary.push_back({stationary->native_geometry()->regions().front(), 9, 5, provider_root});
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(grid);
    const auto product = test::first_product(*root);
    auto fine = root->adapt({}, {product});
    const auto source_static = root->cell_key(1), target_static = fine->cell_key(2);
    ASSERT_EQ(source_static, target_static);
    const auto provenance = std::get<::SimpleFluid::Meshes::AMR::StaticCellAncestry>(source_static.ancestry);
    EXPECT_EQ(provenance.provider_root, provider_root);
    EXPECT_EQ(provenance.native_cell, 0u);
    EXPECT_EQ(root->cell_material(1), fine->cell_material(2));
    EXPECT_EQ(root->geometry()->cell_volume(1), fine->geometry()->cell_volume(2));
    Bound source(root), target(fine);
    native_fields(source);
    native_fields(target);
    const auto certified = plan(source, target);
    ConservativeAMRRemap remap(source.mesh, target.mesh, certified);
    const auto physical = phases(source), expected = phases(target);
    const auto inventory = amounts(physical), expected_inventory = amounts(expected);
    AMRRemapOutput output;
    (void) remap.apply(physical, inventory, output, 1000, 2);
    int local_static = 0;
    for (size_t c = 0; c < target.mesh->num_owned_cells(); ++c)
        if (target.mesh->cell_global_id(LO(c)) == 2)
        {
            ++local_static;
            EXPECT_DOUBLE_EQ(target.mesh->cell_volume(LO(c)), root->geometry()->cell_volume(1));
            EXPECT_DOUBLE_EQ(output.gas_volume[c], expected.gas_volume[c]);
            EXPECT_DOUBLE_EQ(output.carrier_volume[c], expected.carrier_volume[c]);
            for (size_t q = 0; q < inventory.size(); ++q)
                EXPECT_DOUBLE_EQ(output.inventories[q].amount[c], expected_inventory[q].amount[c]);
        }
    int static_owners = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_static, &static_owners);
    EXPECT_EQ(static_owners, 1);
    const auto accepted = output;
    const auto source_observation = source.hierarchy->observation(),
               target_observation = target.hierarchy->observation();
    if (Tpetra::getDefaultComm()->getSize() >= 2)
    {
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            for (size_t c = 0; c < source.mesh->num_owned_cells(); ++c)
                EXPECT_NE(source.mesh->cell_global_id(LO(c)), 1);
            for (size_t c = 0; c < target.mesh->num_owned_cells(); ++c)
                EXPECT_NE(target.mesh->cell_global_id(LO(c)), 2);
        }
        for (const bool forge_root : {true, false})
        {
            auto from = source.cells(), to = target.cells();
            if (Tpetra::getDefaultComm()->getRank() == 0)
                for (auto* catalog : {&from, &to})
                    for (auto& cell : *catalog)
                        if (auto* key = std::get_if<::SimpleFluid::Meshes::AMR::StaticCellAncestry>(&cell.key.ancestry))
                        {
                            if (forge_root)
                                key->provider_root ^= std::uint64_t{1} << 61;
                            else
                                cell.material ^= 1;
                        }
            const auto forged = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(source_observation,
                target_observation, from, to, source.hierarchy->overlap_relations(*target.hierarchy));
            EXPECT_THROW((void) ConservativeAMRRemap(source.mesh, target.mesh, forged), std::invalid_argument);
            EXPECT_EQ(output.gas_volume, accepted.gas_volume);
            EXPECT_EQ(output.carrier_volume, accepted.carrier_volume);
            for (size_t q = 0; q < inventory.size(); ++q)
                EXPECT_EQ(output.inventories[q].amount, accepted.inventories[q].amount);
            EXPECT_EQ(source.hierarchy->observation(), source_observation);
            EXPECT_EQ(target.hierarchy->observation(), target_observation);
        }
        // A rejected owner certificate must not invalidate the retained valid remap.
        (void) remap.apply(physical, inventory, output, 1000, 2);
        for (size_t q = 0; q < inventory.size(); ++q)
            EXPECT_EQ(output.inventories[q].amount, accepted.inventories[q].amount);
    }
}
} // namespace
