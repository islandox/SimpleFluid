/** @file testCylindricalMeshToMeshTransfer.cc
 *  @brief Annular conservative transfer and coupling inventory contracts.
 */

#include <gtest/gtest.h>

#include "fields/MeshToMeshTransfer.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Transfer = SimpleFluid::MeshToMeshTransfer<Pack>;
using Options = SimpleFluid::MeshToMeshTransferOptions;
using Method = SimpleFluid::MeshToMeshTransferMethod;
using CoverageMode = SimpleFluid::MeshToMeshCoverageMode;
using Quantity = SimpleFluid::MeshToMeshQuantity;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using Motion = SimpleFluid::PlanarALEMeshMotion<Pack>;
using LO = Pack::local_ordinal_type;

constexpr double pi = std::numbers::pi;
constexpr double inner_radius = 0.03815;

double radius(double area_coordinate)
{
    return std::sqrt(inner_radius * inner_radius + 0.01 * area_coordinate);
}

SimpleFluid::SP<Handle> annulus(
    SimpleFluid::ArrReal r = {inner_radius, radius(4.0)},
    SimpleFluid::ArrReal theta = {0.0, pi, 2.0 * pi},
    SimpleFluid::ArrReal z = {0.0, 1.0})
{
    auto geometry = std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            std::move(r), std::move(theta), std::move(z)}});
    return std::make_shared<Handle>(std::move(geometry));
}

Options conservative(CoverageMode mode = CoverageMode::RequireFull)
{
    Options options;
    options.method = Method::ConservativeCellAverage;
    options.coverage_mode = mode;
    return options;
}

LO cell(const Handle& mesh, unsigned i, unsigned j, unsigned k)
{
    using GO = Pack::global_ordinal_type;
    const auto gid = mesh.visit([&](const auto& geometry) -> GO
    {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(geometry)>,
                                     Handle::Cylindrical>)
            return static_cast<GO>(geometry.cell_local_id(
                Handle::Cylindrical::CellID{i, j, k}));
        else
            throw std::logic_error("This test helper requires a cylindrical mesh.");
    });
    return mesh.owned_cell_map()->getLocalElement(gid);
}

double integral(const ScalarField& field)
{
    double result = 0.0;
    for (size_t row = 0; row < field.num_owned_cells(); ++row)
    {
        const auto lid = static_cast<LO>(row);
        result += field.value(lid) * field.mesh().cell_volume(lid);
    }
    return result;
}

void expect_balanced(const SimpleFluid::MeshToMeshConservationReport& report,
                     size_t components = 1)
{
    ASSERT_EQ(report.source_integral.size(), components);
    ASSERT_EQ(report.transferred_integral.size(), components);
    ASSERT_EQ(report.uncovered_source_integral.size(), components);
    ASSERT_EQ(report.target_integral.size(), components);
    ASSERT_EQ(report.conservation_error.size(), components);
    ASSERT_EQ(report.transfer_error.size(), components);
    for (size_t component = 0; component < components; ++component)
    {
        EXPECT_NEAR(report.conservation_error[component], 0.0, 1.0e-10);
        EXPECT_NEAR(report.transfer_error[component], 0.0, 1.0e-10);
        EXPECT_NEAR(report.target_integral[component] +
                        report.uncovered_source_integral[component],
                    report.source_integral[component], 1.0e-10);
    }
}
} // namespace

TEST(CylindricalMeshToMeshTransferTest, ConstantFieldsAndLiteralAnnularVolume)
{
    auto source_mesh = annulus({inner_radius, radius(1.0), radius(4.0)},
                              {0.0, 0.5 * pi, 2.0 * pi}, {0.0, 1.0, 3.0});
    auto target_mesh = annulus({inner_radius, radius(0.5), radius(4.0)},
                              {0.0, pi, 2.0 * pi}, {0.0, 2.0, 3.0});
    ScalarField source(source_mesh, "temperature");
    ScalarField target(target_mesh, "temperature");
    source.put_scalar(473.25);
    Transfer transfer(source_mesh, target_mesh, conservative());
    const auto report = transfer.project(source, target, Quantity::Intensive);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
        EXPECT_NEAR(target.local_value(static_cast<LO>(row)), 473.25, 1.0e-11);

    const double volume = 0.04 * pi * 3.0;
    const auto& coverage = transfer.coverage();
    EXPECT_NEAR(coverage.source_volume, volume, 1.0e-14);
    EXPECT_NEAR(coverage.target_volume, volume, 1.0e-14);
    EXPECT_NEAR(coverage.overlap_volume, volume, 1.0e-14);
    EXPECT_NEAR(coverage.uncovered_source_volume, 0.0, 1.0e-14);
    EXPECT_NEAR(coverage.uncovered_target_volume, 0.0, 1.0e-14);
    ASSERT_EQ(coverage.source_cell_volumes.size(), source.num_owned_cells());
    ASSERT_EQ(coverage.target_cell_volumes.size(), target.num_owned_cells());
    for (size_t row = 0; row < source.num_owned_cells(); ++row)
    {
        EXPECT_NEAR(coverage.source_cell_volumes[row],
                    source_mesh->cell_volume(static_cast<LO>(row)), 1.0e-14);
        EXPECT_NEAR(coverage.source_covered_volumes[row],
                    coverage.source_cell_volumes[row], 1.0e-14);
        EXPECT_NEAR(coverage.source_uncovered_volumes[row], 0.0, 1.0e-14);
    }
    for (size_t row = 0; row < target.num_owned_cells(); ++row)
    {
        EXPECT_NEAR(coverage.target_covered_volumes[row],
                    coverage.target_cell_volumes[row], 1.0e-14);
        EXPECT_NEAR(coverage.target_uncovered_volumes[row], 0.0, 1.0e-14);
    }
    expect_balanced(report);
    EXPECT_NEAR(report.source_integral[0], 473.25 * volume, 1.0e-10);
}

TEST(CylindricalMeshToMeshTransferTest, NonuniformRemapUsesRadialAngularAndAxialOverlaps)
{
    auto source_mesh = annulus({inner_radius, radius(1.0), radius(4.0)},
                              {0.0, 0.5 * pi, 2.0 * pi}, {0.0, 1.0, 3.0});
    auto target_mesh = annulus({inner_radius, radius(0.5), radius(4.0)},
                              {0.0, pi, 2.0 * pi}, {0.0, 2.0, 3.0});
    ScalarField source(source_mesh, "density");
    ScalarField target(target_mesh, "density");
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned i = 0; i < 2; ++i)
                source.set_owned_value(cell(*source_mesh, i, j, k),
                                       2.0 + 3.0 * i + 10.0 * j + 100.0 * k);

    Transfer transfer(source_mesh, target_mesh, conservative());
    const auto report = transfer.project(source, target, Quantity::Intensive);
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned i = 0; i < 2; ++i)
                EXPECT_NEAR(target.value(cell(*target_mesh, i, j, k)),
                            2.0 + 3.0 * (i == 0 ? 0.0 : 6.0 / 7.0) +
                                10.0 * (j == 0 ? 0.5 : 1.0) +
                                100.0 * (k == 0 ? 0.5 : 1.0), 1.0e-11);
    EXPECT_NEAR(integral(source), integral(target), 1.0e-12);
    expect_balanced(report);
}

TEST(CylindricalMeshToMeshTransferTest, ExtensiveInventoriesPreserveJoulesLiquidAndMoles)
{
    auto source_mesh = annulus({inner_radius, radius(4.0)},
                              {0.0, pi, 2.0 * pi}, {0.0, 1.0, 3.0});
    auto target_mesh = annulus({inner_radius, radius(1.0), radius(4.0)},
                              {0.0, 0.5 * pi, 2.0 * pi}, {0.0, 2.0, 3.0});
    Pack::multi_vector_type source(source_mesh->owned_cell_map(), 3);
    Pack::multi_vector_type target(target_mesh->owned_cell_map(), 3);
    const std::array<double, 3> scales{1000.0, 3.0, 0.125};
    // Columns are separate extensive inventories, with no volume scaling at
    // the project call. The density 2 below z=1 and 8 above gives exact values.
    {
        auto values = source.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (unsigned k = 0; k < 2; ++k)
            for (unsigned j = 0; j < 2; ++j)
            {
                const auto lid = cell(*source_mesh, 0, j, k);
                for (size_t component = 0; component < scales.size(); ++component)
                    values(lid, component) = scales[component] *
                        (k == 0 ? 2.0 : 8.0) * source_mesh->cell_volume(lid);
            }
    }
    Transfer transfer(source_mesh, target_mesh, conservative());
    const auto report = transfer.project(source, target, Quantity::Extensive);
    const auto values = target.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned i = 0; i < 2; ++i)
            {
                const auto lid = cell(*target_mesh, i, j, k);
                for (size_t component = 0; component < scales.size(); ++component)
                    EXPECT_NEAR(values(lid, component), scales[component] *
                        (k == 0 ? 5.0 : 8.0) * target_mesh->cell_volume(lid), 1.0e-10);
            }
    expect_balanced(report, scales.size());
    for (size_t component = 0; component < scales.size(); ++component)
    {
        EXPECT_NEAR(report.source_integral[component],
                    scales[component] * 18.0 * 0.04 * pi, 1.0e-10);
        EXPECT_NEAR(report.uncovered_source_integral[component], 0.0, 1.0e-10);
    }
}

TEST(CylindricalMeshToMeshTransferTest, AngularSeamAndShiftedFullAnnulusUsePhysicalAngles)
{
    for (const bool full_annulus : {false, true})
    {
        SCOPED_TRACE(full_annulus);
        const SimpleFluid::ArrReal source_angles = full_annulus
            ? SimpleFluid::ArrReal{0.0, 0.5 * pi, pi, 1.5 * pi, 2.0 * pi}
            : SimpleFluid::ArrReal{1.5 * pi, 2.0 * pi, 2.5 * pi};
        const SimpleFluid::ArrReal target_angles = full_annulus
            ? SimpleFluid::ArrReal{-pi, -0.5 * pi, 0.0, 0.5 * pi, pi}
            : SimpleFluid::ArrReal{-0.5 * pi, 0.0, 0.5 * pi};
        auto source_mesh = annulus({inner_radius, radius(4.0)}, source_angles);
        auto target_mesh = annulus({inner_radius, radius(4.0)}, target_angles);
        ScalarField source(source_mesh, "source");
        ScalarField target(target_mesh, "target");
        const unsigned count = full_annulus ? 4 : 2;
        for (unsigned j = 0; j < count; ++j)
            source.set_owned_value(cell(*source_mesh, 0, j, 0), 3.0 + 6.0 * j);
        Transfer transfer(source_mesh, target_mesh, conservative());
        const auto report = transfer.project(source, target, Quantity::Intensive);
        for (unsigned j = 0; j < count; ++j)
        {
            const unsigned donor = full_annulus ? (j + 2) % 4 : j;
            EXPECT_NEAR(target.value(cell(*target_mesh, 0, j, 0)),
                        3.0 + 6.0 * donor, 1.0e-12);
        }
        expect_balanced(report);
    }
}

TEST(CylindricalMeshToMeshTransferTest, SectorCoverageHasNoImplicitFullAnnulusMultiplier)
{
    auto full = annulus();
    auto sector = annulus({inner_radius, radius(4.0)}, {0.0, 0.5 * pi});
    EXPECT_THROW((Transfer(full, sector, conservative())), std::invalid_argument);
    EXPECT_THROW((Transfer(sector, full, conservative())), std::invalid_argument);

    ScalarField full_field(full, "full_inventory");
    ScalarField sector_field(sector, "sector_inventory");
    full_field.put_scalar(100.0); // Each half-annulus owns 100 joules.
    Transfer restrict(full, sector, conservative(CoverageMode::AllowPartial));
    const auto report = restrict.project(full_field, sector_field, Quantity::Extensive);
    EXPECT_NEAR(sector_field.value(0), 50.0, 1.0e-12);
    EXPECT_NEAR(report.source_integral[0], 200.0, 1.0e-12);
    EXPECT_NEAR(report.transferred_integral[0], 50.0, 1.0e-12);
    EXPECT_NEAR(report.uncovered_source_integral[0], 150.0, 1.0e-12);
    EXPECT_NEAR(restrict.coverage().source_volume, 0.04 * pi, 1.0e-14);
    EXPECT_NEAR(restrict.coverage().target_volume, 0.01 * pi, 1.0e-14);
    EXPECT_NEAR(restrict.coverage().overlap_volume, 0.01 * pi, 1.0e-14);
    EXPECT_NEAR(restrict.coverage().uncovered_source_volume, 0.03 * pi, 1.0e-14);
    expect_balanced(report);

    Transfer expand(sector, full, conservative(CoverageMode::AllowPartial));
    const auto back_report = expand.project(sector_field, full_field, Quantity::Extensive);
    EXPECT_NEAR(full_field.value(cell(*full, 0, 0, 0)), 50.0, 1.0e-12);
    EXPECT_NEAR(full_field.value(cell(*full, 0, 1, 0)), 0.0, 1.0e-12);
    EXPECT_NEAR(expand.coverage().uncovered_target_volume, 0.03 * pi, 1.0e-14);
    expect_balanced(back_report);

    // A partially occupied target cell retains the whole-cell denominator.
    sector_field.put_scalar(8.0);
    const auto intensive_report = expand.project(sector_field, full_field, Quantity::Intensive);
    EXPECT_NEAR(full_field.value(cell(*full, 0, 0, 0)), 4.0, 1.0e-12);
    EXPECT_NEAR(full_field.value(cell(*full, 0, 1, 0)), 0.0, 1.0e-12);
    expect_balanced(intensive_report);
}

TEST(CylindricalMeshToMeshTransferTest, DisjointPartialTransferReportsAllUnmappedInventory)
{
    auto source_mesh = annulus();
    auto target_mesh = annulus({inner_radius, radius(4.0)},
                              {0.0, pi, 2.0 * pi}, {2.0, 3.0});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    source.put_scalar(7.0);
    target.put_scalar(91.0);
    Transfer transfer(source_mesh, target_mesh, conservative(CoverageMode::AllowPartial));
    EXPECT_THROW(transfer.apply(source, target), std::invalid_argument);
    EXPECT_DOUBLE_EQ(target.value(0), 91.0);
    const auto report = transfer.project(source, target, Quantity::Extensive);
    EXPECT_DOUBLE_EQ(report.source_integral[0], 14.0);
    EXPECT_DOUBLE_EQ(report.uncovered_source_integral[0], 14.0);
    EXPECT_DOUBLE_EQ(report.transferred_integral[0], 0.0);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
        EXPECT_DOUBLE_EQ(target.local_value(static_cast<LO>(row)), 0.0);
    const auto& coverage = transfer.coverage();
    EXPECT_DOUBLE_EQ(coverage.overlap_volume, 0.0);
    EXPECT_NEAR(coverage.uncovered_source_volume, 0.04 * pi, 1.0e-14);
    EXPECT_NEAR(coverage.uncovered_target_volume, 0.04 * pi, 1.0e-14);
    expect_balanced(report);
}

TEST(CylindricalMeshToMeshTransferTest, AxialExpansionAndContractionReportPartialTopOverlap)
{
    const double area = 0.04 * pi;
    for (const double height : {3.0, 1.0})
    {
        SCOPED_TRACE(height);
        auto source_mesh = annulus({inner_radius, radius(4.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0});
        auto target_mesh = annulus({inner_radius, radius(4.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0});
        ScalarField source(source_mesh, "density");
        ScalarField target(target_mesh, "density");
        source.put_scalar(8.0);
        Motion motion(source_mesh);
        motion.begin_trial(height, 1.0);
        motion.accept_trial();
        EXPECT_THROW((Transfer(source_mesh, target_mesh, conservative())),
                     std::invalid_argument);
        Transfer transfer(source_mesh, target_mesh, conservative(CoverageMode::AllowPartial));
        const auto report = transfer.project(source, target, Quantity::Intensive);
        for (unsigned j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(target.value(cell(*target_mesh, 0, j, 0)), 8.0, 1.0e-12);
            EXPECT_NEAR(target.value(cell(*target_mesh, 0, j, 1)),
                        height > 2.0 ? 8.0 : 0.0, 1.0e-12);
        }
        const auto& coverage = transfer.coverage();
        EXPECT_NEAR(coverage.source_volume, area * height, 1.0e-14);
        EXPECT_NEAR(coverage.target_volume, 2.0 * area, 1.0e-14);
        EXPECT_NEAR(coverage.overlap_volume, area * std::min(height, 2.0), 1.0e-14);
        EXPECT_NEAR(coverage.uncovered_source_volume,
                    height > 2.0 ? area : 0.0, 1.0e-14);
        EXPECT_NEAR(coverage.uncovered_target_volume,
                    height < 2.0 ? area : 0.0, 1.0e-14);
        EXPECT_NEAR(report.uncovered_source_integral[0],
                    height > 2.0 ? 8.0 * area : 0.0, 1.0e-12);
        expect_balanced(report);
    }
}

TEST(CylindricalMeshToMeshTransferTest, MotionAndRestorationOnEitherMeshRequireNewWeights)
{
    for (const bool move_source : {true, false})
    {
        SCOPED_TRACE(move_source);
        auto source_mesh = annulus({inner_radius, radius(4.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0});
        auto target_mesh = annulus({inner_radius, radius(4.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0});
        ScalarField source(source_mesh, "source");
        ScalarField target(target_mesh, "target");
        source.put_scalar(8.0);
        Transfer original(source_mesh, target_mesh, conservative());
        Motion motion(move_source ? source_mesh : target_mesh);
        motion.begin_trial(3.0, 1.0);
        EXPECT_THROW(original.coverage(), std::runtime_error);
        EXPECT_THROW((void)original.project(source, target, Quantity::Intensive), std::runtime_error);
        Transfer trial(source_mesh, target_mesh, conservative(CoverageMode::AllowPartial));
        expect_balanced(trial.project(source, target, Quantity::Intensive));
        if (!move_source)
            for (unsigned j = 0; j < 2; ++j)
                EXPECT_NEAR(target.value(cell(*target_mesh, 0, j, 1)), 8.0 / 3.0, 1.0e-12);
        motion.rollback_trial();
        EXPECT_THROW(original.coverage(), std::runtime_error);
        EXPECT_THROW((void)original.project(source, target, Quantity::Intensive), std::runtime_error);
        EXPECT_THROW(trial.coverage(), std::runtime_error);
        EXPECT_THROW((void)trial.project(source, target, Quantity::Intensive), std::runtime_error);
        Transfer restored(source_mesh, target_mesh, conservative());
        expect_balanced(restored.project(source, target, Quantity::Intensive));
        for (size_t row = 0; row < target.num_owned_cells(); ++row)
            EXPECT_NEAR(target.value(static_cast<LO>(row)), 8.0, 1.0e-12);
    }
}

TEST(CylindricalMeshToMeshTransferTest, InvalidProjectionInputLeavesTargetUnchanged)
{
    auto mesh = annulus();
    auto other_mesh = annulus();
    ScalarField source(mesh, "source");
    ScalarField target(mesh, "target");
    ScalarField unrelated(other_mesh, "unrelated");
    source.put_scalar(8.0);
    target.put_scalar(91.0);
    Transfer transfer(mesh, mesh, conservative());
    EXPECT_THROW((void)transfer.project(source, target, static_cast<Quantity>(-1)),
                 std::invalid_argument);
    EXPECT_THROW((void)transfer.project(unrelated, target, Quantity::Intensive),
                 std::invalid_argument);
    EXPECT_THROW((void)transfer.project(source, unrelated, Quantity::Extensive),
                 std::invalid_argument);
    source.set_owned_value(0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW((void)transfer.project(source, target, Quantity::Intensive),
                 std::invalid_argument);
    for (size_t row = 0; row < target.num_owned_cells(); ++row)
        EXPECT_DOUBLE_EQ(target.value(static_cast<LO>(row)), 91.0);

    auto options = conservative();
    options.coverage_mode = static_cast<CoverageMode>(-1);
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    Transfer interpolation(mesh, mesh);
    EXPECT_THROW((void)interpolation.project(source, target, Quantity::Intensive),
                 std::invalid_argument);
}

TEST(CylindricalMeshToMeshTransferTest, ReplacingEitherGeometryAtSameEpochInvalidatesWeights)
{
    for (const bool replace_source : {true, false})
    {
        SCOPED_TRACE(replace_source);
        auto source_mesh = annulus({inner_radius, radius(4.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0});
        auto target_mesh = annulus({inner_radius, radius(4.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0});
        ScalarField source(source_mesh, "source");
        ScalarField target(target_mesh, "target");
        source.put_scalar(8.0);
        target.put_scalar(91.0);
        Transfer transfer(source_mesh, target_mesh, conservative());

        auto replacement = annulus({inner_radius, radius(5.0)},
                                  {0.0, pi, 2.0 * pi}, {0.0, 0.75, 2.0});
        const auto changed = replace_source ? source_mesh : target_mesh;
        ASSERT_EQ(changed->geometry_epoch(), replacement->geometry_epoch());
        ASSERT_TRUE(changed->owned_cell_map()->isSameAs(*replacement->owned_cell_map()));
        *changed = *replacement;

        EXPECT_THROW(transfer.coverage(), std::runtime_error);
        EXPECT_THROW((void)transfer.project(source, target, Quantity::Intensive),
                     std::runtime_error);
        for (size_t row = 0; row < target.num_owned_cells(); ++row)
            EXPECT_DOUBLE_EQ(target.value(static_cast<LO>(row)), 91.0);

        Transfer rebuilt(source_mesh, target_mesh, conservative(CoverageMode::AllowPartial));
        expect_balanced(rebuilt.project(source, target, Quantity::Intensive));
    }
}

TEST(CylindricalMeshToMeshTransferTest, ReorderingEitherMapWithSameGeometryInvalidatesWeights)
{
    using Factory = SimpleFluid::MeshReorderingFactory<Pack>;
    for (const bool reorder_source : {true, false})
    {
        SCOPED_TRACE(reorder_source);
        auto geometry = std::make_shared<Handle::Cylindrical>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {inner_radius, radius(4.0)}, {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0}}});
        auto source_mesh = std::make_shared<Handle>(geometry);
        auto target_mesh = std::make_shared<Handle>(geometry);
        ScalarField source(source_mesh, "source");
        ScalarField target(target_mesh, "target");
        source.put_scalar(8.0);
        target.put_scalar(91.0);
        Transfer transfer(source_mesh, target_mesh, conservative());

        const auto reordered = Factory::selected_cells_first(
            std::make_shared<Handle>(geometry),
            [](Pack::global_ordinal_type, const Handle::Vec3& center)
            { return center.z > 1.0; });
        const auto changed = reorder_source ? source_mesh : target_mesh;
        ASSERT_TRUE(reordered.mesh->has_reordered_cells());
        ASSERT_EQ(changed->geometry_identity(), reordered.mesh->geometry_identity());
        ASSERT_EQ(changed->geometry_epoch(), reordered.mesh->geometry_epoch());
        ASSERT_FALSE(changed->owned_cell_map()->isSameAs(*reordered.mesh->owned_cell_map()));
        *changed = *reordered.mesh;

        EXPECT_THROW(transfer.coverage(), std::runtime_error);
        EXPECT_THROW((void)transfer.project(source, target, Quantity::Intensive),
                     std::runtime_error);
        for (size_t row = 0; row < target.num_owned_cells(); ++row)
            EXPECT_DOUBLE_EQ(target.value(static_cast<LO>(row)), 91.0);
    }
}
