/** @file testProductAMRRemap.cc
 * @brief Authoritative product geometry, native field maps and crossed-factor remap.
 */
#include "FVM/ConservativeAMRRemap.hh"
#include "fields/MeshToMeshTransfer.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <map>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::VoF;
using Handle = MeshHandle<>;
using LO = DefaultTpetraTypes::local_ordinal_type;
using Rational = boost::multiprecision::cpp_rational;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid grid()
{
    ::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid result;
    result.xy_nodes = {{0.1, 0.1, 0}, {0.2, 0.1, 0}, {0.2, 0.2, 0}, {0.1, 0.2, 0}};
    result.base_cells = {{0, 1, 2, 3}};
    result.boundaries = {{0, 1, "bottom"}, {1, 2, "right"}, {2, 3, "top"}, {3, 0, "left"}};
    result.z_edges = {0.1, 0.2};
    result.region_id = (std::uint64_t{1} << 60) + 23;
    result.material_id = (std::uint64_t{1} << 61) + 7;
    return result;
}
struct Bound
{
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::ExtrudedAMR> hierarchy;
    SP<Handle> mesh;
    ::SimpleFluid::Meshes::AMR::HierarchyObservation observation;
    explicit Bound(std::shared_ptr<const ::SimpleFluid::Meshes::AMR::ExtrudedAMR> h) : hierarchy(std::move(h))
    {
        CompositePartitionOptions options;
        options.policy = CompositePartitionOptions::Policy::ContiguousCanonical;
        options.target_cells_per_fragment = 1;
        const auto partition = MeshPartitioner<DefaultTpetraTypes>::partition(
            CompositeMeshSource(hierarchy->geometry()), options, Tpetra::getDefaultComm());
        hierarchy = hierarchy->with_partition(std::make_shared<const CompositePartition<>>(partition));
        mesh = std::make_shared<Handle>(*hierarchy->native_partition());
        observation = hierarchy->observation();
    }
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> cells() const
    {
        // Bounded fixture bootstrap only: certificates are queried on demand,
        // never retained as a global B*Z cache by the hierarchy/provider.
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> result;
        for (std::uint64_t c = 0; c < hierarchy->geometry()->num_cells(); ++c)
            result.push_back({hierarchy->cell_key(c), c, hierarchy->cell_material(c), hierarchy->cell_geometry(c)});
        return result;
    }
};
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> plan(const Bound& source, const Bound& target)
{
    const auto relations = source.hierarchy->overlap_relations(*target.hierarchy);
    return ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        source.observation, target.observation, source.cells(), target.cells(), relations);
}
AMRRemapPhases phases(const Bound& source)
{
    AMRRemapPhases result;
    const GeometryHalfspace plane{{1, 0, 1}, {0.1, 0.1, 0.1}, 0.075};
    for (std::size_t c = 0; c < source.mesh->num_owned_cells(); ++c)
    {
        const auto gid = source.mesh->cell_global_id(static_cast<LO>(c));
        const auto cut = source.hierarchy->cell_geometry(gid)->clip(plane);
        result.gas_volume.push_back(cut.inside ? cut.inside->moments().volume.as_double() : 0);
        result.carrier_volume.push_back(cut.outside ? cut.outside->moments().volume.as_double() : 0);
        result.gas_halfspace.push_back(plane);
    }
    return result;
}
std::vector<ConservedQuantity> inventories(const AMRRemapPhases& phase)
{
    std::vector<ConservedQuantity> result{{"gas_species", InventoryCarrier::GasVolume, {}, true},
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
void native_fields(const Bound& bound)
{
    ScalarCellFieldStored<> field(bound.mesh, "native_gid_field");
    for (std::size_t c = 0; c < bound.mesh->num_owned_cells(); ++c)
        field.set_owned_value(static_cast<LO>(c), 1 + bound.mesh->cell_global_id(static_cast<LO>(c)));
    field.sync_ghosts();
    for (std::size_t c = 0; c < bound.mesh->num_cells(); ++c)
        EXPECT_EQ(field.local_value(static_cast<LO>(c)), 1 + bound.mesh->cell_global_id(static_cast<LO>(c)));
    for (std::size_t c = 0; c < bound.mesh->num_owned_cells(); ++c)
    {
        MeshUtils::Vec3 area{};
        double total = 0;
        bound.mesh->visit_cell_faces(static_cast<LO>(c),
            [&](LO face)
            {
                area = area + bound.mesh->face_area_vector_outward(face, static_cast<LO>(c));
                total += bound.mesh->face_area(face);
            });
        EXPECT_LE(area.norm(), 64 * std::numeric_limits<double>::epsilon() * total);
    }
    EXPECT_EQ(bound.mesh->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(bound.mesh->has_materialized_indexer());
}
::SimpleFluid::Meshes::AMR::HierarchyObservation geometry_observation()
{
    const auto token = std::make_shared<const char>(0);
    return {token, token, token, 0, 0, 0, token.get(), Tpetra::getDefaultComm().getRawPtr()};
}
::SimpleFluid::Meshes::AMR::AdaptiveCellKey product_key(
    std::vector<std::uint64_t> base, std::vector<std::uint64_t> axial)
{
    return {31,
        ::SimpleFluid::Meshes::AMR::ProductCellAncestry{::SimpleFluid::Meshes::AMR::BaseCellKey{7, 9, std::move(base)},
            ::SimpleFluid::Meshes::AMR::AxialIntervalKey{13, std::move(axial)}}};
}
} // namespace

TEST(ProductAMRRemapTest, CrossedActiveProductsRejectDespiteDisjointSurrogateShells)
{
    const auto observation = geometry_observation();
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> cells{
        {product_key({}, {0}), 0, 3, ConvexCellGeometry::box({0, 0, 0}, {1, 1, 0.5})},
        {product_key({0}, {}), 1, 3, ConvexCellGeometry::box({0, 0, 0.5}, {1, 1, 1})}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> identities{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, 0, 0},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, 1, 1}};
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
                     observation, observation, cells, cells, identities),
        std::invalid_argument);
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> disjoint{
        {product_key({0}, {0}), 0, 3, ConvexCellGeometry::box({0, 0, 0}, {1, 1, 0.5})},
        {product_key({1}, {0}), 1, 3, ConvexCellGeometry::box({0, 0, 0.5}, {1, 1, 1})}};
    EXPECT_NO_THROW((void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        observation, observation, disjoint, disjoint, identities));
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> from{
        {product_key({}, {0}), 0, 3, ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1})}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> to{
        {product_key({0}, {}), 0, 3, ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1})}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> crossed{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::ProductIntersection, 0, 0}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> wrong{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 0}};
    EXPECT_NO_THROW(
        (void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(observation, observation, from, to, crossed));
    EXPECT_THROW(
        (void) ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(observation, observation, from, to, wrong),
        std::invalid_argument);
}

TEST(ProductAMRRemapTest, DecimalChildrenUseAuthoritativeVolumeAcrossNativePacketsAndSplitOrder)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    const auto xy_z = root->adapt({0})->adapt({}, {0});
    const auto z_xy = root->adapt({}, {0})->adapt({0});
    Bound native(xy_z), other(z_xy);
    ASSERT_EQ(xy_z->geometry()->num_cells(), 8u);
    ASSERT_EQ(z_xy->geometry()->num_cells(), 8u);
    const Rational width = Rational(0.2) - Rational(0.1);
    const double expected = (width * width * width / 8).convert_to<double>();
    bool changed_display = false;
    const auto resident = std::get<Handle::MultiRegionPtr>(native.mesh->variant());
    EXPECT_TRUE(resident->has_exact_construction());
    for (std::uint64_t gid = 0; gid < 8; ++gid)
    {
        EXPECT_EQ(xy_z->cell_key(gid), z_xy->cell_key(gid));
        EXPECT_TRUE(xy_z->cell_geometry(gid)->equivalent(*z_xy->cell_geometry(gid)));
    }
    for (std::size_t c = 0; c < native.mesh->num_owned_cells(); ++c)
    {
        const auto gid = native.mesh->cell_global_id(static_cast<LO>(c));
        EXPECT_EQ(native.mesh->cell_volume(static_cast<LO>(c)), expected);
        EXPECT_TRUE(ConvexCellGeometry::from_native_cell(*resident, gid)->equivalent(*xy_z->cell_geometry(gid)));
        std::array<double, 3> lo{1, 1, 1}, hi{};
        for (const auto node : resident->cell_nodes(gid))
        {
            const auto point = resident->node_coordinates(node);
            const std::array<double, 3> coordinates{point.x, point.y, point.z};
            for (std::size_t a = 0; a < 3; ++a)
            {
                lo[a] = std::min(lo[a], coordinates[a]);
                hi[a] = std::max(hi[a], coordinates[a]);
            }
        }
        const Rational display = (Rational(hi[0]) - Rational(lo[0])) * (Rational(hi[1]) - Rational(lo[1])) *
                                 (Rational(hi[2]) - Rational(lo[2]));
        changed_display = changed_display || display.convert_to<double>() != expected;
    }
    int changed = changed_display, any_changed = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &changed, &any_changed);
    EXPECT_EQ(any_changed, 1);
    native_fields(native);
    native_fields(other);
    ConservativeAMRRemap transfer(native.mesh, other.mesh, plan(native, other));
    AMRRemapOutput output;
    const auto physical = phases(native);
    (void) transfer.apply(physical, inventories(physical), output, 1000, 2);
    EXPECT_EQ(output.gas_volume.size(), other.mesh->num_owned_cells());
}

TEST(ProductAMRRemapTest, CrossedFactorRemapPreservesTiltedPhaseMassMomentumAndPairedInventories)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    Bound source(root->adapt({}, {0})), target(root->adapt({0})), coarse(root);
    const auto forward = plan(source, target), backward = plan(target, source);
    ASSERT_FALSE(forward->records().empty());
    for (const auto& record : forward->records())
        EXPECT_EQ(record.relation, ::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::ProductIntersection);
    ConservativeAMRRemap split(source.mesh, target.mesh, forward), reverse(target.mesh, source.mesh, backward);
    const auto physical = phases(source);
    const auto initial = inventories(physical);
    AMRRemapOutput output, restored;
    (void) split.apply(physical, initial, output, 1000, 2);
    for (std::size_t c = 0; c < output.gas_volume.size(); ++c)
        if (output.inventories[2].amount[c] > 0)
            EXPECT_NEAR(output.inventories[1].amount[c] / output.inventories[2].amount[c], 11.0 / 3.0, 2e-14);
    AMRRemapPhases next{output.gas_volume, output.carrier_volume,
        std::vector<GeometryHalfspace>(output.gas_volume.size(), {{1, 0, 1}, {0.1, 0.1, 0.1}, 0.075})};
    (void) reverse.apply(next, output.inventories, restored, 1000, 2);
    for (std::size_t c = 0; c < restored.gas_volume.size(); ++c)
    {
        EXPECT_NEAR(restored.gas_volume[c], physical.gas_volume[c], 2e-17);
        EXPECT_NEAR(restored.carrier_volume[c], physical.carrier_volume[c], 2e-17);
        for (std::size_t q = 0; q < initial.size(); ++q)
            EXPECT_NEAR(restored.inventories[q].amount[c], initial[q].amount[c], 2e-13);
    }
    // A root snapshot gives an observed empty source owner at every multi-rank size.
    ConservativeAMRRemap refinement(coarse.mesh, target.mesh, plan(coarse, target));
    const auto root_phases = phases(coarse);
    (void) refinement.apply(root_phases, inventories(root_phases), output, 1000, 2);
    int empty = coarse.mesh->num_owned_cells() == 0, source_empty = 0, owned = target.mesh->num_owned_cells() > 0,
        owners = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &source_empty);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &owned, &owners);
    EXPECT_EQ(source_empty, comm->getSize() - 1);
    EXPECT_EQ(owners, std::min(4, comm->getSize()));
    const int reverse_owned = source.mesh->num_owned_cells() > 0, reverse_empty = source.mesh->num_owned_cells() == 0;
    int reverse_owners = 0, reverse_empty_owners = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &reverse_owned, &reverse_owners);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &reverse_empty, &reverse_empty_owners);
    EXPECT_EQ(reverse_owners, std::min(2, comm->getSize()));
    EXPECT_EQ(reverse_empty_owners, comm->getSize() - std::min(2, comm->getSize()));
    if (comm->getRank() == 0)
        std::cout << "product_native_remap,ranks=" << comm->getSize() << ",root_empty_owners=" << source_empty
                  << ",target_owners=" << owners << ",reverse_target_owners=" << reverse_owners
                  << ",reverse_target_empty_owners=" << reverse_empty_owners << '\n';
    native_fields(source);
    native_fields(target);
    native_fields(coarse);
}

TEST(ProductAMRRemapTest, ConstructedNodeHullConsumersRejectAndRankLocalFailuresPreserveOutput)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    Bound source(root->adapt({}, {0})), target(root->adapt({0}));
    for (const auto method : {MeshToMeshTransferMethod::NearestCell, MeshToMeshTransferMethod::InverseDistance,
             MeshToMeshTransferMethod::ConservativeCellAverage})
    {
        MeshToMeshTransferOptions options;
        options.method = method;
        EXPECT_THROW((void) MeshToMeshTransfer<>(source.mesh, target.mesh, options), std::invalid_argument);
    }
    EXPECT_THROW((void) PlanarALEMeshMotion<>(target.mesh), std::exception);
    auto stale = source.observation;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        ++stale.geometry;
    const auto relations = source.hierarchy->overlap_relations(*target.hierarchy);
    const auto stale_plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        stale, target.observation, source.cells(), target.cells(), relations);
    EXPECT_THROW((void) ConservativeAMRRemap(source.mesh, target.mesh, stale_plan), std::invalid_argument);
    ConservativeAMRRemap remap(source.mesh, target.mesh, plan(source, target));
    const auto physical = phases(source);
    const auto quantities = inventories(physical);
    AMRRemapOutput output;
    (void) remap.apply(physical, quantities, output, 1000, 2);
    const auto accepted = output;
    auto bad = physical;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !bad.gas_halfspace.empty())
        bad.gas_halfspace[0].offset = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) remap.apply(bad, quantities, output, 1000, 2), std::invalid_argument);
    EXPECT_EQ(output.gas_volume, accepted.gas_volume);
    EXPECT_EQ(output.carrier_volume, accepted.carrier_volume);
    for (std::size_t q = 0; q < output.inventories.size(); ++q)
        EXPECT_EQ(output.inventories[q].amount, accepted.inventories[q].amount);
    Bound replacement(root);
    *source.mesh = *replacement.mesh;
    EXPECT_THROW((void) remap.apply(physical, quantities, output, 1000, 2), std::invalid_argument);
    EXPECT_EQ(output.gas_volume, accepted.gas_volume);
    EXPECT_EQ(output.carrier_volume, accepted.carrier_volume);
}

TEST(ProductAMRRemapTest, RemoteProductFactorMetadataMustAuthenticateAgainstOwners)
{
    if (Tpetra::getDefaultComm()->getSize() < 2)
        GTEST_SKIP() << "Remote product metadata requires MPI.";
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    Bound source(root->adapt({}, {0})), target(root->adapt({0}));
    const auto valid = plan(source, target);
    EXPECT_NO_THROW((void) ConservativeAMRRemap(source.mesh, target.mesh, valid));
    auto from = source.cells(), to = target.cells();
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (auto* cells : {&from, &to})
            for (auto& cell : *cells)
                std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(cell.key.ancestry).base.root_template ^=
                    std::uint64_t{1} << 63;
    const auto relations = source.hierarchy->overlap_relations(*target.hierarchy);
    const auto altered = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        source.observation, target.observation, from, to, relations);
    EXPECT_THROW((void) ConservativeAMRRemap(source.mesh, target.mesh, altered), std::invalid_argument);
}

TEST(ProductAMRRemapTest, ConstructedPureGeometryRetainsBoundedMicroscopicAndSubnormalPairedInventories)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    Bound source(root), target(root->adapt({0}));
    ConservativeAMRRemap remap(source.mesh, target.mesh, plan(source, target));
    const int owns_rank_zero = Tpetra::getDefaultComm()->getRank() == 0 && source.mesh->num_owned_cells() > 0;
    int mutations = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &owns_rank_zero, &mutations);
    ASSERT_EQ(mutations, 1);
    for (const auto tiny : {1e-200, 1e-310, 4 * std::numeric_limits<double>::denorm_min()})
        for (const bool gas : {true, false})
        {
            AMRRemapPhases physical;
            for (std::size_t c = 0; c < source.mesh->num_owned_cells(); ++c)
            {
                const auto gid = source.mesh->cell_global_id(static_cast<LO>(c));
                const auto full = source.hierarchy->cell_geometry(gid)->moments().volume.as_double();
                physical.gas_volume.push_back(gas ? tiny : full);
                physical.carrier_volume.push_back(gas ? full : tiny);
                physical.gas_halfspace.push_back({});
                physical.geometry_declaration.push_back(
                    gas ? AMRPhaseGeometry::PureCarrier : AMRPhaseGeometry::PureGas);
            }
            const auto kind = gas ? InventoryCarrier::GasVolume : InventoryCarrier::LiquidVolume;
            const auto owned = source.mesh->num_owned_cells();
            const std::vector<ConservedQuantity> paired{{"number", kind, std::vector<double>(owned, 1e300), true},
                {"moles", kind, std::vector<double>(owned, 2e-300), true}};
            AMRRemapOutput output;
            (void) remap.apply(physical, paired, output);
            for (std::size_t c = 0; c < output.gas_volume.size(); ++c)
            {
                EXPECT_DOUBLE_EQ(gas ? output.gas_volume[c] : output.carrier_volume[c], tiny * 0.25);
                EXPECT_DOUBLE_EQ(output.inventories[0].amount[c], 2.5e299);
                EXPECT_DOUBLE_EQ(output.inventories[1].amount[c], 0.5e-300);
            }
            const auto accepted = output;
            auto excessive = physical;
            if (Tpetra::getDefaultComm()->getRank() == 0 && !excessive.gas_volume.empty())
            {
                const auto full = excessive.gas_volume[0] + excessive.carrier_volume[0];
                excessive.gas_volume[0] = gas ? 1e-6 * full : (1 - 1e-6) * full;
                excessive.carrier_volume[0] = gas ? (1 - 1e-6) * full : 1e-6 * full;
            }
            EXPECT_THROW((void) remap.apply(excessive, paired, output), std::invalid_argument);
            EXPECT_EQ(output.gas_volume, accepted.gas_volume);
            EXPECT_EQ(output.carrier_volume, accepted.carrier_volume);
            for (std::size_t q = 0; q < paired.size(); ++q)
                EXPECT_EQ(output.inventories[q].amount, accepted.inventories[q].amount);
        }
}

TEST(ProductAMRRemapTest, EqualAggregateVolumeRoundedDisplaySurrogatesRejectOwnerAuthentication)
{
    const auto root = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    Bound native(root->adapt({0}, {0}));
    EXPECT_NO_THROW((void) ConservativeAMRRemap(native.mesh, native.mesh, plan(native, native)));
    auto display = native.cells();
    Rational total = 0;
    const auto catalog = native.hierarchy->geometry();
    for (auto& cell : display)
    {
        std::array<double, 3> lo{1, 1, 1}, hi{};
        for (const auto node : catalog->cell_nodes(cell.gid))
        {
            const auto point = catalog->node_coordinates(node);
            const std::array<double, 3> coordinates{point.x, point.y, point.z};
            for (std::size_t a = 0; a < 3; ++a)
            {
                lo[a] = std::min(lo[a], coordinates[a]);
                hi[a] = std::max(hi[a], coordinates[a]);
            }
        }
        total += (Rational(hi[0]) - Rational(lo[0])) * (Rational(hi[1]) - Rational(lo[1])) *
                 (Rational(hi[2]) - Rational(lo[2]));
        cell.geometry = ConvexCellGeometry::box(lo, hi);
    }
    const Rational width = Rational(0.2) - Rational(0.1);
    EXPECT_EQ(total, width * width * width);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> identities;
    for (const auto& cell : display)
        identities.push_back(
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, cell.gid, cell.gid});
    const auto altered = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        native.observation, native.observation, display, display, identities);
    EXPECT_THROW((void) ConservativeAMRRemap(native.mesh, native.mesh, altered), std::invalid_argument);
}
