/** @file testPolyhedralAMRRemap.cc
 * @brief Exact polyhedral native traversal and physical conservative transfer in MPI.
 */
#include "FVM/ConservativeAMRRemap.hh"
#include "fields/MeshToMeshTransfer.hh"
#include "geometry/unitTests/polyhedral_amr_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>

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
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::PolyhedralAMR> hierarchy;
    SP<Handle> mesh;
    explicit Bound(std::shared_ptr<const ::SimpleFluid::Meshes::AMR::PolyhedralAMR> h) : hierarchy(std::move(h))
    {
        CompositePartitionOptions options;
        options.policy = CompositePartitionOptions::Policy::ContiguousCanonical;
        options.target_cells_per_fragment = 1;
        const auto partition =
            std::make_shared<const CompositePartition<>>(MeshPartitioner<DefaultTpetraTypes>::partition(
                CompositeMeshSource(hierarchy->geometry()), options, Tpetra::getDefaultComm()));
        hierarchy = hierarchy->with_partition(partition);
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
AMRRemapPhases phases(const Bound& bound)
{
    AMRRemapPhases result;
    const SimpleFluid::PLIC::GeometryHalfspace plane{{1, 0.3, 0.7}, {0, 0, 0}, 0.85};
    for (std::size_t c = 0; c < bound.mesh->num_owned_cells(); ++c)
    {
        const auto cut = bound.hierarchy->cell_geometry(bound.mesh->cell_global_id(static_cast<LO>(c)))->clip(plane);
        result.gas_volume.push_back(cut.inside ? cut.inside->moments().volume.as_double() : 0);
        result.carrier_volume.push_back(cut.outside ? cut.outside->moments().volume.as_double() : 0);
        result.gas_halfspace.push_back(plane);
    }
    return result;
}
std::vector<ConservedQuantity> quantities(const AMRRemapPhases& phase)
{
    std::vector<ConservedQuantity> result{{"gas", InventoryCarrier::GasVolume, {}, true},
        {"number", InventoryCarrier::LiquidVolume, {}, true}, {"moles", InventoryCarrier::LiquidVolume, {}, true},
        {"mass", InventoryCarrier::MixtureMass, {}, true}, {"momentum", InventoryCarrier::MixtureMass, {}, false}};
    for (std::size_t c = 0; c < phase.gas_volume.size(); ++c)
    {
        const auto g = phase.gas_volume[c], l = phase.carrier_volume[c], mass = 1000 * l + 2 * g;
        result[0].amount.push_back(7 * g);
        result[1].amount.push_back(11 * l);
        result[2].amount.push_back(3 * l);
        result[3].amount.push_back(mass);
        result[4].amount.push_back(-0.2 * mass);
    }
    return result;
}
void traversal(const Bound& bound)
{
    ScalarCellFieldStored<> field(bound.mesh, "keys");
    for (std::size_t c = 0; c < bound.mesh->num_owned_cells(); ++c)
        field.set_owned_value(static_cast<LO>(c), 1 + bound.mesh->cell_global_id(static_cast<LO>(c)));
    field.sync_ghosts();
    for (std::size_t c = 0; c < bound.mesh->num_cells(); ++c)
        EXPECT_EQ(field.local_value(static_cast<LO>(c)), 1 + bound.mesh->cell_global_id(static_cast<LO>(c)));
    for (std::size_t c = 0; c < bound.mesh->num_owned_cells(); ++c)
    {
        MeshUtils::Vec3 area{};
        double flux = 0, total = 0;
        bound.mesh->visit_cell_faces(static_cast<LO>(c),
            [&](LO face)
            {
                const auto outward = bound.mesh->face_area_vector_outward(face, static_cast<LO>(c));
                area = area + outward;
                flux += outward.x + 0.3 * outward.y - 0.2 * outward.z;
                total += bound.mesh->face_area(face);
            });
        EXPECT_LE(area.norm(), 128 * std::numeric_limits<double>::epsilon() * total);
        EXPECT_LE(std::abs(flux), 128 * std::numeric_limits<double>::epsilon() * total);
    }
    EXPECT_EQ(bound.mesh->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(bound.mesh->has_materialized_indexer());
}
} // namespace
TEST(PolyhedralAMRRemapTest, UnequalSkewCutsAndCoarseningKeepNativeFluxesAndPhysicalInventories)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::PolyhedralAMR>(test::polyhedral_grid());
    Bound coarse(root), fine(root->adapt({0, 1}));
    ConservativeAMRRemap forward(coarse.mesh, fine.mesh, plan(coarse, fine)),
        reverse(fine.mesh, coarse.mesh, plan(fine, coarse));
    const auto physical = phases(coarse);
    const auto initial = quantities(physical);
    AMRRemapOutput output, restored;
    (void) forward.apply(physical, initial, output, 1000, 2);
    for (std::size_t c = 0; c < output.gas_volume.size(); ++c)
        if (output.inventories[2].amount[c] > 0)
            EXPECT_NEAR(output.inventories[1].amount[c] / output.inventories[2].amount[c], 11.0 / 3.0, 3e-14);
    AMRRemapPhases next{output.gas_volume, output.carrier_volume,
        std::vector<SimpleFluid::PLIC::GeometryHalfspace>(output.gas_volume.size(), {{1, 0.3, 0.7}, {0, 0, 0}, 0.85})};
    (void) reverse.apply(next, output.inventories, restored, 1000, 2);
    for (std::size_t c = 0; c < restored.gas_volume.size(); ++c)
    {
        EXPECT_NEAR(restored.gas_volume[c], physical.gas_volume[c], 3e-14);
        EXPECT_NEAR(restored.carrier_volume[c], physical.carrier_volume[c], 3e-14);
        for (std::size_t q = 0; q < initial.size(); ++q)
            EXPECT_NEAR(restored.inventories[q].amount[c], initial[q].amount[c], 2e-10);
    }
    Bound neighbor(fine.hierarchy->adapt({}, {root->cell_key(0)}));
    traversal(coarse);
    traversal(fine);
    traversal(neighbor);
    const auto comm = Tpetra::getDefaultComm();
    const int local = coarse.mesh->num_owned_cells() > 0;
    int owners = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &owners);
    EXPECT_EQ(owners, std::min(2, comm->getSize()));
    EXPECT_EQ(comm->getSize() - owners, std::max(0, comm->getSize() - 2));
    if (comm->getRank() == 0)
        std::cout << "polyhedral_remap,ranks=" << comm->getSize() << ",coarse_owners=" << owners
                  << ",coarse_empty_owners=" << comm->getSize() - owners << '\n';
}
TEST(PolyhedralAMRRemapTest, VariableTetrahedralFamiliesRetainExplicitPureSubnormalPairedCarriers)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::PolyhedralAMR>(
        test::polyhedral_grid(test::polyhedral_tetrahedron()));
    Bound coarse(root), fine(root->adapt({0}));
    ConservativeAMRRemap remap(coarse.mesh, fine.mesh, plan(coarse, fine));
    // Resolved tetra children do not have equal bulk weights; the microscopic
    // physical carrier follows those same exact bulk ratios for both moments.
    AMRRemapPhases physical;
    for (std::size_t c = 0; c < coarse.mesh->num_owned_cells(); ++c)
    {
        physical.gas_volume.push_back(1e-310);
        physical.carrier_volume.push_back(1.0 / 6.0);
        physical.gas_halfspace.push_back({});
        physical.geometry_declaration.push_back(AMRPhaseGeometry::PureCarrier);
    }
    const auto n = coarse.mesh->num_owned_cells();
    const std::vector<ConservedQuantity> paired{
        {"number", InventoryCarrier::GasVolume, std::vector<double>(n, 1e300), true},
        {"moles", InventoryCarrier::GasVolume, std::vector<double>(n, 2e-300), true}};
    AMRRemapOutput result;
    (void) remap.apply(physical, paired, result);
    for (std::size_t c = 0; c < result.gas_volume.size(); ++c)
    {
        const auto gid = fine.mesh->cell_global_id(static_cast<LO>(c));
        const auto ratio = fine.hierarchy->cell_geometry(gid)->moments().volume.value / (1.0L / 6.0L);
        EXPECT_GT(result.gas_volume[c], 0);
        EXPECT_NEAR(result.inventories[0].amount[c] / 1e300, static_cast<double>(ratio), 2e-15);
        EXPECT_NEAR(result.inventories[1].amount[c] / 2e-300, static_cast<double>(ratio), 2e-15);
    }
    const auto accepted = result;
    auto excessive = physical;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !excessive.gas_volume.empty())
    {
        excessive.gas_volume[0] = 1e-3;
        excessive.carrier_volume[0] = 1.0 / 6.0 - 1e-3;
    }
    EXPECT_THROW((void) remap.apply(excessive, paired, result), std::invalid_argument);
    EXPECT_EQ(result.gas_volume, accepted.gas_volume);
    for (std::size_t q = 0; q < paired.size(); ++q)
        EXPECT_EQ(result.inventories[q].amount, accepted.inventories[q].amount);
    const auto comm = Tpetra::getDefaultComm();
    const int owned = coarse.mesh->num_owned_cells() > 0;
    int owners = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &owned, &owners);
    EXPECT_EQ(owners, 1);
    int target_local = fine.mesh->num_owned_cells() > 0, target_owners = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &target_local, &target_owners);
    EXPECT_EQ(target_owners, std::min(4, comm->getSize()));
}
TEST(PolyhedralAMRRemapTest, InvalidLocalInventoryAndReplacedNativeMapPreserveOutput)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::PolyhedralAMR>(test::polyhedral_grid());
    Bound coarse(root), fine(root->adapt({0, 1}));
    ConservativeAMRRemap remap(coarse.mesh, fine.mesh, plan(coarse, fine));
    const auto physical = phases(coarse);
    auto initial = quantities(physical);
    AMRRemapOutput result;
    (void) remap.apply(physical, initial, result, 1000, 2);
    const auto accepted = result;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !initial[0].amount.empty())
        initial[0].amount[0] = -1;
    EXPECT_THROW((void) remap.apply(physical, initial, result, 1000, 2), std::invalid_argument);
    EXPECT_EQ(result.gas_volume, accepted.gas_volume);
    EXPECT_EQ(result.carrier_volume, accepted.carrier_volume);
    for (std::size_t q = 0; q < initial.size(); ++q)
        EXPECT_EQ(result.inventories[q].amount, accepted.inventories[q].amount);
    Bound replacement(root);
    *fine.mesh = *replacement.mesh;
    EXPECT_THROW((void) remap.apply(physical, quantities(physical), result, 1000, 2), std::invalid_argument);
    EXPECT_EQ(result.gas_volume, accepted.gas_volume);
}

TEST(PolyhedralAMRRemapTest, RemoteConvexMetadataReplicasCannotBypassExactOwnerChecks)
{
    if (Tpetra::getDefaultComm()->getSize() < 2)
        GTEST_SKIP() << "Remote owner binding needs MPI.";
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::PolyhedralAMR>(test::polyhedral_grid());
    Bound coarse(root), fine(root->adapt({0, 1}));
    EXPECT_NO_THROW((void) ConservativeAMRRemap(coarse.mesh, fine.mesh, plan(coarse, fine)));
    auto from = coarse.cells(), to = fine.cells();
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (auto* cells : {&from, &to})
            for (auto& cell : *cells)
                cell.material ^= std::uint64_t{1} << 63;
    const auto relations = coarse.hierarchy->overlap_relations(*fine.hierarchy);
    const auto bad = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        coarse.hierarchy->observation(), fine.hierarchy->observation(), from, to, relations);
    EXPECT_THROW((void) ConservativeAMRRemap(coarse.mesh, fine.mesh, bad), std::invalid_argument);
}
