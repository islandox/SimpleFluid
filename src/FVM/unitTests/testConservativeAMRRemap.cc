/** @file testConservativeAMRRemap.cc
 * @brief Actual convex phase cuts and distributed sparse extensive remap contracts.
 */
#include "FVM/ConservativeAMRRemap.hh"
#include "FVM/details/AMRRemapBudget.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
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
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> prism(double low, double high)
{
    const std::vector<SimpleFluid::PLIC::GeometryPoint> points{
        {0, 0, low}, {1, 0, low}, {0, 1, low}, {0, 0, high}, {1, 0, high}, {0, 1, high}};
    return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::make(
        points, {{2, 1, 0}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}});
}
SP<Handle> native_prism(const ArrReal& levels)
{
    auto provider = std::make_shared<SemiStructuredXY_Z>(Arr<MeshUtils::Vec3>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
        Arr<Arr<unsigned>>{{0, 1, 2}}, levels,
        Arr<SemiStructuredXY_Z::BoundaryEdge>{{0, 1, "bottom"}, {1, 2, "slope"}, {2, 0, "left"}});
    auto catalog =
        std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("prism", provider)},
            std::vector<MultiRegionMesh::Interface>{});
    CompositePartitionOptions options;
    options.target_cells_per_fragment = 1;
    auto partition =
        MeshPartitioner<DefaultTpetraTypes>::partition(CompositeMeshSource(catalog), options, Tpetra::getDefaultComm());
    return std::make_shared<Handle>(partition);
}
::SimpleFluid::Meshes::AMR::HierarchyObservation observation(const SP<Handle>& mesh)
{
    auto token = std::make_shared<const char>(0);
    return {token, token, token, 0, 0, mesh->geometry_epoch(), mesh->geometry_identity(),
        Tpetra::getDefaultComm().getRawPtr()};
}
::SimpleFluid::Meshes::AMR::AdaptiveCellKey key(std::vector<std::uint64_t> path)
{
    return {(std::uint64_t{1} << 60) + 17,
        ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{(std::uint64_t{1} << 61) + 41, std::move(path)}};
}
struct Fixture
{
    SP<Handle> parent = native_prism({0, 1}), children = native_prism({0, 0.5, 1});
    ::SimpleFluid::Meshes::AMR::HierarchyObservation before = observation(parent), after = observation(children);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> coarse{{key({}), 0, 3, prism(0, 1)}};
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> fine{
        {key({0}), 0, 3, prism(0, 0.5)}, {key({1}), 1, 3, prism(0.5, 1)}};
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> split, coarsen;
    Fixture()
    {
        const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> forward{
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 0},
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 1}};
        const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> reverse{
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Coarsening, 0, 0},
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Coarsening, 1, 0}};
        split = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(before, after, coarse, fine, forward);
        coarsen = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(after, before, fine, coarse, reverse);
    }
    AMRRemapPhases phases(bool refined) const
    {
        const auto& mesh = refined ? children : parent;
        AMRRemapPhases result;
        const SimpleFluid::PLIC::GeometryHalfspace plane{{1, 0, 1}, {0.75, 0, 0}};
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto gid = mesh->cell_global_id(static_cast<LO>(c));
            const auto geometry = refined ? fine.at(gid).geometry : coarse.front().geometry;
            const auto cut = geometry->clip(plane);
            result.gas_volume.push_back(cut.inside ? cut.inside->moments().volume.as_double() : 0);
            result.carrier_volume.push_back(cut.outside ? cut.outside->moments().volume.as_double() : 0);
            result.gas_halfspace.push_back(plane);
        }
        return result;
    }
};
std::vector<ConservedQuantity> inventories(const AMRRemapPhases& phases)
{
    std::vector<ConservedQuantity> result{{"gas_species", InventoryCarrier::GasVolume, {}, true},
        {"population_number", InventoryCarrier::LiquidVolume, {}, true},
        {"population_moles", InventoryCarrier::LiquidVolume, {}, true},
        {"mass", InventoryCarrier::MixtureMass, {}, true}, {"momentum", InventoryCarrier::MixtureMass, {}, false}};
    for (std::size_t c = 0; c < phases.gas_volume.size(); ++c)
    {
        const auto mass = 1000 * phases.carrier_volume[c] + 2 * phases.gas_volume[c];
        result[0].amount.push_back(7 * phases.gas_volume[c]);
        result[1].amount.push_back(11 * phases.carrier_volume[c]);
        result[2].amount.push_back(3 * phases.carrier_volume[c]);
        result[3].amount.push_back(mass);
        result[4].amount.push_back(-0.2 * mass);
    }
    return result;
}
void expect_equal(const AMRRemapOutput& actual, const AMRRemapOutput& expected)
{
    EXPECT_EQ(actual.gas_volume, expected.gas_volume);
    EXPECT_EQ(actual.carrier_volume, expected.carrier_volume);
    ASSERT_EQ(actual.inventories.size(), expected.inventories.size());
    for (std::size_t q = 0; q < actual.inventories.size(); ++q)
    {
        EXPECT_EQ(actual.inventories[q].name, expected.inventories[q].name);
        EXPECT_EQ(actual.inventories[q].carrier, expected.inventories[q].carrier);
        EXPECT_EQ(actual.inventories[q].nonnegative, expected.inventories[q].nonnegative);
        EXPECT_EQ(actual.inventories[q].amount, expected.inventories[q].amount);
    }
}
} // namespace

TEST(ConservativeAMRRemapTest, TiltedPrismSplitAndReverseCoarseningPreservePhysicalCarriersAndPairedMoments)
{
    Fixture f;
    ConservativeAMRRemap split(f.parent, f.children, f.split), coarsen(f.children, f.parent, f.coarsen);
    auto phases = f.phases(false);
    const auto initial = inventories(phases);
    AMRRemapOutput result;
    const auto report = split.apply(phases, initial, result, 1000, 2);
    EXPECT_NEAR(std::ldexp(report.scaled_before[0], report.binary_exponent[0]), 27.0 / 128.0, 1e-15);
    for (std::size_t c = 0; c < result.gas_volume.size(); ++c)
    {
        const auto gid = f.children->cell_global_id(static_cast<LO>(c));
        // Independent integration of integral_0^h [(3/4-z)-(3/4-z)^2/2] dz.
        const auto expected = gid == 0 ? 35.0 / 192.0 : 11.0 / 384.0;
        EXPECT_NEAR(result.gas_volume[c], expected, 1e-15);
        EXPECT_NEAR(result.carrier_volume[c], 0.25 - expected, 1e-15);
        EXPECT_NEAR(result.inventories[1].amount[c] / result.inventories[2].amount[c], 11.0 / 3.0, 1e-14);
    }
    AMRRemapPhases fine{result.gas_volume, result.carrier_volume,
        std::vector<SimpleFluid::PLIC::GeometryHalfspace>(result.gas_volume.size(), {{1, 0, 1}, {0.75, 0, 0}})};
    AMRRemapOutput restored;
    const auto reverse_report = coarsen.apply(fine, result.inventories, restored, 1000, 2);
    for (std::size_t c = 0; c < restored.gas_volume.size(); ++c)
    {
        EXPECT_NEAR(restored.gas_volume[c], phases.gas_volume[c], 1e-15);
        EXPECT_NEAR(restored.carrier_volume[c], phases.carrier_volume[c], 1e-15);
        for (std::size_t q = 0; q < initial.size(); ++q)
            EXPECT_NEAR(restored.inventories[q].amount[c], initial[q].amount[c], 1e-12);
    }
    const int local_empty = f.parent->num_owned_cells() == 0;
    int empty = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    EXPECT_EQ(empty, ranks - 1);
    if (ranks > 1)
    {
        const int owns_target = f.children->num_owned_cells() > 0;
        int owners = 0;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &owns_target, &owners);
        EXPECT_EQ(owners, 2);
    }
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << "convex_sparse_remap,ranks=" << ranks
                  << ",source_cells=1,target_cells=2,source_empty_owners=" << empty << '\n';
    EXPECT_EQ(f.parent->connectivity_storage_bytes(), 0u);
    EXPECT_EQ(f.children->connectivity_storage_bytes(), 0u);
    EXPECT_EQ(reverse_report.scaled_before.size(), 7u);
}

TEST(ConservativeAMRRemapTest, RoundedPureMicroscopicAndSubnormalPhasesRetainSharedPairedWeights)
{
    Fixture f;
    ConservativeAMRRemap remap(f.parent, f.children, f.split);
    for (const auto tiny : {1e-200, 1e-310, 2 * std::numeric_limits<double>::denorm_min()})
        for (const bool gas : {true, false})
        {
            AMRRemapPhases phases;
            const auto owned = f.parent->num_owned_cells();
            phases.gas_volume.assign(owned, gas ? tiny : 0.5);
            phases.carrier_volume.assign(owned, gas ? 0.5 : tiny);
            phases.gas_halfspace.resize(owned);
            phases.geometry_declaration.assign(owned, gas ? AMRPhaseGeometry::PureCarrier : AMRPhaseGeometry::PureGas);
            const auto kind = gas ? InventoryCarrier::GasVolume : InventoryCarrier::LiquidVolume;
            std::vector<ConservedQuantity> paired{{"number", kind, std::vector<double>(owned, 1e300), true},
                {"moles", kind, std::vector<double>(owned, 2e-300), true}};
            AMRRemapOutput result;
            const auto report = remap.apply(phases, paired, result);
            for (std::size_t c = 0; c < result.gas_volume.size(); ++c)
            {
                EXPECT_DOUBLE_EQ(gas ? result.gas_volume[c] : result.carrier_volume[c], tiny * 0.5);
                EXPECT_DOUBLE_EQ(result.inventories[0].amount[c], 5e299);
                EXPECT_DOUBLE_EQ(result.inventories[1].amount[c], 1e-300);
            }
            const auto budget = gas ? 0u : 1u;
            EXPECT_GT(report.scaled_before[budget], 0);
            EXPECT_TRUE(std::isfinite(report.scaled_absolute_before[budget]));
            EXPECT_DOUBLE_EQ(std::ldexp(report.scaled_before[budget], report.binary_exponent[budget]), tiny);
            if (tiny < std::numeric_limits<double>::min())
                EXPECT_LT(report.binary_exponent[budget], -1022);
            const auto accepted = result;
            auto undeclared = phases;
            undeclared.geometry_declaration.clear();
            undeclared.gas_halfspace.assign(
                owned, SimpleFluid::PLIC::GeometryHalfspace{{0, 0, gas ? 1.0 : -1.0}, {0, 0, 0}});
            EXPECT_THROW((void) remap.apply(undeclared, paired, result), std::invalid_argument);
            expect_equal(result, accepted);
        }
}

TEST(ConservativeAMRRemapTest, ResolvedTinyPhaseKeepsItsActualGeometricLocation)
{
    Fixture f;
    ConservativeAMRRemap remap(f.parent, f.children, f.split);
    for (const bool gas : {true, false})
    {
        AMRRemapPhases phases;
        const auto owned = f.parent->num_owned_cells();
        phases.gas_volume.assign(owned, gas ? 0.5e-200 : 0.5);
        phases.carrier_volume.assign(owned, gas ? 0.5 : 0.5e-200);
        phases.gas_halfspace.assign(
            owned, SimpleFluid::PLIC::GeometryHalfspace{{0, 0, gas ? 1.0 : -1.0}, {0, 0, 1e-200}});
        const auto kind = gas ? InventoryCarrier::GasVolume : InventoryCarrier::LiquidVolume;
        const std::vector<ConservedQuantity> quantities{
            {"retained_species", kind, std::vector<double>(owned, 3.0), true}};
        AMRRemapOutput result;
        (void) remap.apply(phases, quantities, result);
        for (std::size_t c = 0; c < result.gas_volume.size(); ++c)
        {
            const auto gid = f.children->cell_global_id(static_cast<LO>(c));
            EXPECT_DOUBLE_EQ(gas ? result.gas_volume[c] : result.carrier_volume[c], gid == 0 ? 0.5e-200 : 0.0);
            EXPECT_DOUBLE_EQ(result.inventories[0].amount[c], gid == 0 ? 3.0 : 0.0);
        }
    }
}

TEST(ConservativeAMRRemapTest, EqualVolumeShiftedCatalogCannotMasqueradeAsNativeGeometry)
{
    Fixture f;
    EXPECT_NO_THROW((void) ConservativeAMRRemap(f.parent, f.children, f.split));
    auto coarse = f.coarse, fine = f.fine;
    auto shifted = [](double lower, double upper)
    {
        const std::vector<SimpleFluid::PLIC::GeometryPoint> points{
            {2, 0, lower}, {3, 0, lower}, {2, 1, lower}, {2, 0, upper}, {3, 0, upper}, {2, 1, upper}};
        return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::make(
            points, {{2, 1, 0}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}});
    };
    coarse[0].geometry = shifted(0, 1);
    fine[0].geometry = shifted(0, 0.5);
    fine[1].geometry = shifted(0.5, 1);
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 0},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 1}};
    const auto impostor =
        ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(f.before, f.after, coarse, fine, relations);
    EXPECT_THROW((void) ConservativeAMRRemap(f.parent, f.children, impostor), std::invalid_argument);
}

TEST(ConservativeAMRRemapTest, SignedCancellationUsesL1AndRatioWeightsAvoidLargeAmountOverflow)
{
    Fixture f;
    ConservativeAMRRemap coarsen(f.children, f.parent, f.coarsen);
    auto phases = f.phases(true);
    std::vector<ConservedQuantity> quantities{{"signed", InventoryCarrier::BulkVolume, {}, false}};
    for (std::size_t c = 0; c < f.children->num_owned_cells(); ++c)
        quantities[0].amount.push_back(f.children->cell_global_id(static_cast<LO>(c)) == 0 ? 1e308 : -1e308);
    AMRRemapOutput result;
    const auto report = coarsen.apply(phases, quantities, result);
    EXPECT_EQ(report.scaled_before[2], 0);
    EXPECT_EQ(report.scaled_after[2], 0);
    EXPECT_EQ(report.binary_exponent[2], 1024);
    EXPECT_GT(report.scaled_absolute_before[2], 1.0);
    EXPECT_TRUE(std::isfinite(report.scaled_absolute_before[2]));
    for (const auto value : result.inventories[0].amount)
        EXPECT_DOUBLE_EQ(value, 0);
    ConservativeAMRRemap split(f.parent, f.children, f.split);
    const auto coarse = f.phases(false);
    quantities[0].amount.assign(f.parent->num_owned_cells(), 1e308);
    (void) split.apply(coarse, quantities, result);
    for (const auto value : result.inventories[0].amount)
        EXPECT_DOUBLE_EQ(value, 5e307);
}

TEST(ConservativeAMRRemapTest, RankLocalFailuresAndMovedEndpointsPreserveOutput)
{
    Fixture f;
    ConservativeAMRRemap remap(f.parent, f.children, f.split);
    const auto phases = f.phases(false);
    const auto quantities = inventories(phases);
    AMRRemapOutput result;
    (void) remap.apply(phases, quantities, result);
    const auto accepted = result;
    auto bad = phases;
    if (!bad.gas_volume.empty())
        bad.gas_volume[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) remap.apply(bad, quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
    bad = phases;
    if (!bad.gas_volume.empty())
    {
        bad.gas_volume[0] += 0.01;
        bad.carrier_volume[0] -= 0.01;
    }
    EXPECT_THROW((void) remap.apply(bad, quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
    bad = phases;
    if (!bad.gas_halfspace.empty())
        bad.gas_halfspace[0] = {};
    EXPECT_THROW((void) remap.apply(bad, quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
    bad = phases;
    if (!bad.gas_halfspace.empty())
        bad.gas_halfspace[0].offset = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) remap.apply(bad, quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
    auto bad_quantities = quantities;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        bad_quantities[0].amount.push_back(0);
    EXPECT_THROW((void) remap.apply(phases, bad_quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        bad_quantities = quantities;
        if (Tpetra::getDefaultComm()->getRank() == 0)
            bad_quantities[0].name = "bad_species";
        EXPECT_THROW((void) remap.apply(phases, bad_quantities, result), std::invalid_argument);
        expect_equal(result, accepted);
    }
    PlanarALEMeshMotion<> motion(f.children);
    motion.begin_trial(1.1, 0.1);
    EXPECT_THROW((void) remap.apply(phases, quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
    motion.rollback_trial();
    EXPECT_THROW((void) remap.apply(phases, quantities, result), std::invalid_argument);
    expect_equal(result, accepted);
}

TEST(ConservativeAMRRemapTest, LocalCompleteAncestryFamiliesDoNotRequireGlobalGeometryCatalogs)
{
    const auto source = native_prism({0, 0.5, 1}), target = native_prism({0, 0.25, 0.5, 0.75, 1});
    std::set<std::uint64_t> families;
    for (std::size_t c = 0; c < source->num_owned_cells(); ++c)
        families.insert(source->cell_global_id(static_cast<LO>(c)));
    for (std::size_t c = 0; c < target->num_owned_cells(); ++c)
        families.insert(target->cell_global_id(static_cast<LO>(c)) / 2);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> sources, targets;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations;
    for (const auto family : families)
    {
        const auto coarse_key = ::SimpleFluid::Meshes::AMR::AdaptiveCellKey{
            17, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{100 + family, {}}};
        sources.push_back({coarse_key, family, 3, prism(0.5 * family, 0.5 * (family + 1))});
        for (std::uint64_t child = 0; child < 2; ++child)
        {
            const auto gid = 2 * family + child;
            targets.push_back({::SimpleFluid::Meshes::AMR::AdaptiveCellKey{
                                   17, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{100 + family, {child}}},
                gid, 3, prism(0.25 * gid, 0.25 * (gid + 1))});
            relations.push_back({::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, family, gid});
        }
    }
    const auto plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        observation(source), observation(target), sources, targets, relations);
    ConservativeAMRRemap remap(source, target, plan);
    AMRRemapPhases phases{std::vector<double>(source->num_owned_cells(), 0),
        std::vector<double>(source->num_owned_cells(), 0.25),
        std::vector<SimpleFluid::PLIC::GeometryHalfspace>(source->num_owned_cells())};
    std::vector<ConservedQuantity> quantities{{"tracer", InventoryCarrier::BulkVolume, {}, true}};
    for (std::size_t c = 0; c < source->num_owned_cells(); ++c)
        quantities[0].amount.push_back(1 + source->cell_global_id(static_cast<LO>(c)));
    AMRRemapOutput result;
    (void) remap.apply(phases, quantities, result);
    for (std::size_t c = 0; c < result.carrier_volume.size(); ++c)
    {
        const auto gid = target->cell_global_id(static_cast<LO>(c));
        EXPECT_DOUBLE_EQ(result.carrier_volume[c], 0.125);
        EXPECT_DOUBLE_EQ(result.inventories[0].amount[c], 0.5 * (1 + gid / 2));
    }
    const int local_partial = families.size() < 2;
    int partial = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_partial, &partial);
    if (Tpetra::getDefaultComm()->getSize() > 1)
        EXPECT_GT(partial, 0);
}

TEST(ConservativeAMRRemapTest, CompensatedScalarBudgetsRemainAttainableAtSixtyFourRanks)
{
    std::vector<std::array<double, 4>> ranks(64);
    for (std::size_t rank = 0; rank < ranks.size(); ++rank)
        ranks[rank] = {rank % 2 ? -0.5 : 0.5, rank % 2 ? -0.5 : 0.5, 0.5, 0.5};
    const auto sum = SimpleFluid::VoF::detail::sum_rank_budgets(ranks, 128);
    EXPECT_DOUBLE_EQ(sum.values[0], 0);
    EXPECT_DOUBLE_EQ(sum.values[1], 0);
    EXPECT_DOUBLE_EQ(sum.values[2], 32);
    EXPECT_DOUBLE_EQ(sum.values[3], 32);
    EXPECT_LT(sum.arithmetic, 128 * std::numeric_limits<double>::epsilon() * 32);
    for (auto& rank : ranks)
        rank = {};
    const auto zero = SimpleFluid::VoF::detail::sum_rank_budgets(ranks, 128);
    EXPECT_EQ(zero.arithmetic, 0);
    EXPECT_EQ(zero.values, (std::array<double, 4>{}));
}

TEST(ConservativeAMRRemapTest, DoubleAccumulatorProductsNeverSilentlyUnderflowOrOverflow)
{
    using SimpleFluid::VoF::detail::weighted_amount;
    EXPECT_DOUBLE_EQ(weighted_amount(1e308, 0.25, 0.5), 5e307);
    EXPECT_NEAR(weighted_amount(1e300, 1e-300, 1e300) / 1e-300, 1.0, 1e-15);
    EXPECT_DOUBLE_EQ(weighted_amount(-1e308, 0.25, 0.5), -5e307);
    EXPECT_THROW((void) weighted_amount(std::numeric_limits<double>::denorm_min(), 0.25, 1.0), std::invalid_argument);
    EXPECT_THROW((void) weighted_amount(-std::numeric_limits<double>::denorm_min(), 0.25, 1.0), std::invalid_argument);
    EXPECT_DOUBLE_EQ(weighted_amount(0.0, 0.25, 1.0), 0.0);
}

TEST(ConservativeAMRRemapTest, FictitiousRemoteTargetCannotCloseATrueNativeGap)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires a nonowned sibling in the local exact certificate.";
    const double a = 0.5, b = std::nextafter(a, 1.0);
    const std::array<double, 4> axis{0, a, b, 1};
    Arr<MeshUtils::Vec3> points;
    for (const auto y : axis)
        for (const auto x : axis)
            points.push_back({x, y, 0});
    Arr<Arr<unsigned>> triangles;
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 3; ++i)
        {
            if (i == 1 && j == 1)
                continue;
            const auto v00 = 4 * j + i, v10 = v00 + 1, v01 = v00 + 4, v11 = v01 + 1;
            triangles.push_back({v00, v10, v01});
            triangles.push_back({v10, v11, v01});
        }
    std::map<std::pair<unsigned, unsigned>, std::vector<std::pair<unsigned, unsigned>>> edges;
    for (const auto& triangle : triangles)
        for (std::size_t i = 0; i < 3; ++i)
        {
            const auto from = triangle[i], to = triangle[(i + 1) % 3];
            edges[{std::min(from, to), std::max(from, to)}].push_back({from, to});
        }
    Arr<SemiStructuredXY_Z::BoundaryEdge> boundaries;
    for (const auto& [edge, uses] : edges)
        if (uses.size() == 1)
        {
            const auto [from, to] = uses.front();
            const bool hole = from % 4 >= 1 && from % 4 <= 2 && from / 4 >= 1 && from / 4 <= 2 && to % 4 >= 1 &&
                              to % 4 <= 2 && to / 4 >= 1 && to / 4 <= 2;
            boundaries.push_back({from, to, hole ? "hole" : "wall"});
        }
    const auto provider = std::make_shared<SemiStructuredXY_Z>(points, triangles, ArrReal{0, 1}, boundaries);
    const auto catalog = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{native_region("connected_ring", provider)},
        std::vector<MultiRegionMesh::Interface>{});
    CompositePartitionOptions options;
    options.policy = CompositePartitionOptions::Policy::ContiguousCanonical;
    options.target_cells_per_fragment = 1;
    SP<Handle> target;
    ASSERT_NO_THROW(target = std::make_shared<Handle>(
                        MeshPartitioner<DefaultTpetraTypes>::partition(CompositeMeshSource(catalog), options, comm)));
    ASSERT_TRUE(target);
    EXPECT_EQ(target->owned_cell_map()->getGlobalNumElements(), 16u);
    const auto box = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{cartesian_region("box", {{{0, 1}, {0, 1}, {0, 1}}})},
        std::vector<MultiRegionMesh::Interface>{});
    const auto source = std::make_shared<Handle>(
        MeshPartitioner<DefaultTpetraTypes>::partition(CompositeMeshSource(box), options, comm));
    auto triangle_geometry = [&](std::size_t index, bool fill)
    {
        std::vector<SimpleFluid::PLIC::GeometryPoint> vertices;
        for (const auto z : {0.0, 1.0})
            for (const auto node : triangles[index])
            {
                auto x = points[node].x, y = points[node].y;
                if (fill)
                {
                    if (index == 3 && x == b && y == a)
                        y = b;
                    if (index == 8 && x == b && y == a)
                        x = a;
                    if (index == 7 && x == a && y == b)
                        x = b;
                    if (index == 12 && x == a && y == b)
                        y = a;
                }
                vertices.push_back({x, y, z});
            }
        return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::make(
            vertices, {{2, 1, 0}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}});
    };
    const auto owned = target->owned_cell_map();
    const auto lower = !owned->isNodeGlobalElement(3) ? 3u : 8u;
    const auto upper = !owned->isNodeGlobalElement(7) ? 7u : 12u;
    const int remote_candidates = !owned->isNodeGlobalElement(lower) && !owned->isNodeGlobalElement(upper);
    int all_remote = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &remote_candidates, &all_remote);
    ASSERT_EQ(all_remote, 1);
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> coarse{{key({}), 0, 3,
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1})}};
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> fine;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations;
    for (std::uint64_t index = 0; index < triangles.size(); ++index)
    {
        fine.push_back({key({index}), index, 3, triangle_geometry(index, index == lower || index == upper)});
        relations.push_back({::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, index});
    }
    for (std::size_t c = 0; c < target->num_owned_cells(); ++c)
    {
        const auto gid = target->cell_global_id(static_cast<LO>(c));
        EXPECT_TRUE(fine[gid].geometry->equivalent(*triangle_geometry(gid, false)));
    }
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> actual_cells;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> identities;
    for (std::uint64_t index = 0; index < triangles.size(); ++index)
    {
        actual_cells.push_back({key({index}), index, 3, triangle_geometry(index, false)});
        identities.push_back({::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, index, index});
    }
    const auto target_observation = observation(target);
    const auto identity_plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        target_observation, target_observation, actual_cells, actual_cells, identities);
    ASSERT_NO_THROW((void) ConservativeAMRRemap(target, target, identity_plan));
    // Each local certificate fills the tiny true hole by changing only remote
    // triangles; native construction and local exact coverage must first pass.
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> plan;
    try
    {
        plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
            observation(source), observation(target), coarse, fine, relations);
    }
    catch (const std::exception& error)
    {
        ADD_FAILURE() << "Local exact coverage fixture failed: " << error.what();
    }
    const int built = bool(plan);
    int all_built = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &built, &all_built);
    ASSERT_EQ(all_built, 1);
    EXPECT_THROW((void) ConservativeAMRRemap(source, target, plan), std::invalid_argument);
}

TEST(ConservativeAMRRemapTest, RemoteAncestryAndMaterialReplicasMustMatchTheirOwnerBindings)
{
    if (Tpetra::getDefaultComm()->getSize() < 2)
        GTEST_SKIP() << "Requires different catalog owners and replicas.";
    Fixture f;
    EXPECT_NO_THROW((void) ConservativeAMRRemap(f.parent, f.children, f.split));
    for (const bool material : {false, true})
    {
        auto coarse = f.coarse, fine = f.fine;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            for (auto* cells : {&coarse, &fine})
                for (auto& cell : *cells)
                    if (material)
                        cell.material ^= std::uint64_t{1} << 63;
                    else
                        std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(cell.key.ancestry).root ^=
                            std::uint64_t{1} << 63;
        }
        // Local ancestry, material relationships and exact shells still certify.
        // Only sparse owner binding authentication reveals the inconsistent replica.
        const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations{
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 0},
            {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 1}};
        const auto plan =
            ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(f.before, f.after, coarse, fine, relations);
        EXPECT_THROW((void) ConservativeAMRRemap(f.parent, f.children, plan), std::invalid_argument);
    }
}

TEST(ConservativeAMRRemapTest, SparsePlaneOffsetPreservesTranslatedMicroscopicPhaseLocation)
{
    constexpr double origin = 1e6;
    const auto source = native_prism({origin, origin + 1}), target = native_prism({origin, origin + 0.5, origin + 1});
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> coarse{
        {key({}), 0, 3, prism(origin, origin + 1)}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> fine{
        {key({0}), 0, 3, prism(origin, origin + 0.5)}, {key({1}), 1, 3, prism(origin + 0.5, origin + 1)}};
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations{
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 0},
        {::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Refinement, 0, 1}};
    const auto plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
        observation(source), observation(target), coarse, fine, relations);
    ConservativeAMRRemap remap(source, target, plan);
    const auto owned = source->num_owned_cells();
    AMRRemapPhases phases{std::vector<double>(owned, 0.5e-200), std::vector<double>(owned, 0.5),
        std::vector<SimpleFluid::PLIC::GeometryHalfspace>(
            owned, SimpleFluid::PLIC::GeometryHalfspace{{0, 0, 1}, {0, 0, origin}, 1e-200})};
    const std::vector<ConservedQuantity> quantities{
        {"microscopic_species", InventoryCarrier::GasVolume, std::vector<double>(owned, 7.0), true}};
    AMRRemapOutput result;
    (void) remap.apply(phases, quantities, result);
    for (std::size_t c = 0; c < result.gas_volume.size(); ++c)
    {
        const auto gid = target->cell_global_id(static_cast<LO>(c));
        EXPECT_DOUBLE_EQ(result.gas_volume[c], gid == 0 ? 0.5e-200 : 0.0);
        EXPECT_DOUBLE_EQ(result.inventories[0].amount[c], gid == 0 ? 7.0 : 0.0);
    }
}

TEST(ConservativeAMRRemapTest, VariableOwnerCertificateBudgetRejectsOneRankBeforePublication)
{
    Fixture f;
    const auto comm = Tpetra::getDefaultComm();
    EXPECT_ANY_THROW(ConservativeAMRRemap(
        f.parent, f.children, f.split, comm->getRank() == 0 ? std::size_t(1) : std::size_t(64 * 1024 * 1024)));
    ConservativeAMRRemap accepted(f.parent, f.children, f.split);
    auto phase = f.phases(false);
    AMRRemapOutput result;
    EXPECT_NO_THROW((void) accepted.apply(phase, inventories(phase), result, 1000, 2));
    EXPECT_EQ(result.gas_volume.size(), f.children->num_owned_cells());
}

TEST(ConservativeAMRRemapTest, GloballyEmptyNativeDomainsRejectThroughBothConstructorOverloads)
{
    SparseNativeMeshInput input;
    auto geometry = MultiRegionMesh::from_sparse_native(std::move(input), Tpetra::getDefaultComm());
    auto partition = CompositePartition<>::from_sparse_native(geometry, {}, {});
    auto mesh = std::make_shared<Handle>(partition);
    const auto stamp = observation(mesh);
    auto plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(stamp, stamp, {}, {}, {});
    const auto rejects_empty = [&](bool explicit_budget)
    {
        try
        {
            if (explicit_budget)
                (void) ConservativeAMRRemap(mesh, mesh, plan, 1024 * 1024);
            else
                (void) ConservativeAMRRemap(mesh, mesh, plan);
            ADD_FAILURE() << "Globally empty native remap unexpectedly constructed.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("nonempty global source and target domains"), std::string::npos);
        }
    };
    rejects_empty(false);
    rejects_empty(true);
}
