/**
 * @file testMeshToMeshTransfer.cc
 * @brief Cell-centered interpolation and conservative projection contracts.
 */

#include <gtest/gtest.h>

#include "fields/MeshToMeshTransfer.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_Array.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Cartesian = Handle::Cartesian;
using Transfer = SimpleFluid::MeshToMeshTransfer<Pack>;
using Options = SimpleFluid::MeshToMeshTransferOptions;
using Method = SimpleFluid::MeshToMeshTransferMethod;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorField = SimpleFluid::VectorCellFieldStored<Pack>;
using TensorField = SimpleFluid::TensorCellFieldStored<Pack>;
using LO = Pack::local_ordinal_type;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SimpleFluid::SP<Handle> make_cartesian(
    SimpleFluid::ArrReal x,
    SimpleFluid::ArrReal y = {0.0, 1.0},
    SimpleFluid::ArrReal z = {0.0, 1.0})
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            std::move(x), std::move(y), std::move(z)}});
    return std::make_shared<Handle>(std::move(geometry));
}

Options with_method(Method method)
{
    Options options;
    options.method = method;
    return options;
}

double integral(const ScalarField& field)
{
    double result = 0.0;
    for (size_t row = 0; row < field.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        result += field.value(cell) * field.mesh().cell_volume(cell);
    }
    return result;
}

} // namespace

TEST(MeshToMeshTransferTest, ConservativeUnequalVolumesWorkInBothDirections)
{
    auto coarse = make_cartesian({0.0, 1.0, 3.0});
    auto fine = make_cartesian({0.0, 0.5, 2.0, 3.0});
    ScalarField coarse_field(coarse, "coarse");
    ScalarField fine_field(fine, "fine");
    coarse_field.set_owned_value(0, 2.0);
    coarse_field.set_owned_value(1, 8.0);
    const auto options = with_method(Method::ConservativeCellAverage);

    Transfer prolong(coarse, fine, options);
    prolong.apply(coarse_field, fine_field);
    EXPECT_DOUBLE_EQ(fine_field.value(0), 2.0);
    EXPECT_NEAR(fine_field.value(1), 6.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(fine_field.value(2), 8.0);
    EXPECT_NEAR(integral(fine_field), integral(coarse_field), 1.0e-13);

    const double fine_integral = integral(fine_field);
    Transfer restrict(fine, coarse, options);
    restrict.apply(fine_field, coarse_field);
    EXPECT_NEAR(coarse_field.value(0), 4.0, 1.0e-14);
    EXPECT_NEAR(coarse_field.value(1), 7.0, 1.0e-14);
    EXPECT_NEAR(integral(coarse_field), fine_integral, 1.0e-13);
}

TEST(MeshToMeshTransferTest, ConservativeOverlapUsesAllThreeDimensions)
{
    auto source_mesh = make_cartesian(
        {0.0, 1.0, 2.0}, {0.0, 1.0, 3.0}, {0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian(
        {0.0, 0.5, 2.0}, {0.0, 2.0, 3.0}, {0.0, 0.5, 2.0});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    for (size_t row = 0; row < source.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto center = source_mesh->cell_centroid(cell);
        source.set_owned_value(cell, center.x + 10.0 * center.y + 100.0 * center.z);
    }

    Transfer transfer(source_mesh, target_mesh,
                      with_method(Method::ConservativeCellAverage));
    transfer.apply(source, target);
    for (size_t row = 0; row < target.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto center = target_mesh->cell_centroid(cell);
        const double x_average = center.x < 0.5 ? 0.5 : 7.0 / 6.0;
        const double y_average = center.y < 2.0 ? 1.25 : 2.0;
        const double z_average = center.z < 0.5 ? 0.5 : 7.0 / 6.0;
        EXPECT_NEAR(target.value(cell),
                    x_average + 10.0 * y_average + 100.0 * z_average,
                    1.0e-12);
    }
    EXPECT_NEAR(integral(target), integral(source), 1.0e-11);
}

TEST(MeshToMeshTransferTest, EveryMethodPreservesConstantsAcrossResolutions)
{
    auto source_mesh = make_cartesian({0.0, 0.25, 1.0, 3.0});
    auto target_mesh = make_cartesian({0.0, 1.0, 2.0, 2.5, 3.0});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    source.put_scalar(-4.25);
    for (const auto method : {Method::NearestCell, Method::InverseDistance,
                              Method::ConservativeCellAverage})
    {
        SCOPED_TRACE(static_cast<int>(method));
        Transfer transfer(source_mesh, target_mesh, with_method(method));
        transfer.apply(source, target);
        for (size_t row = 0; row < target.num_local_cells(); ++row)
        {
            EXPECT_NEAR(target.local_value(static_cast<LO>(row)), -4.25, 1.0e-14);
        }
    }
}

TEST(MeshToMeshTransferTest, DefaultInverseDistanceHasKnownWeightsAndFreshValues)
{
    auto source_mesh = make_cartesian({0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian({0.5, 1.0});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    source.set_owned_value(0, 2.0);
    source.set_owned_value(1, 12.0);

    // Distances 1/4 and 3/4 give inverse-square weights 9/10 and 1/10.
    Transfer transfer(source_mesh, target_mesh);
    transfer.apply(source, target);
    EXPECT_NEAR(target.value(0), 3.0, 1.0e-14);
    EXPECT_NEAR(target.local_value(0), 3.0, 1.0e-14);

    source.set_owned_value(0, -3.0);
    source.set_owned_value(1, 7.0);
    transfer.apply(source, target);
    EXPECT_NEAR(target.value(0), -2.0, 1.0e-14);
    EXPECT_NEAR(target.local_value(0), -2.0, 1.0e-14);
}

TEST(MeshToMeshTransferTest, NearestTiesChooseLowestGlobalIdAndIdwHonorsNeighborCount)
{
    auto source_mesh = make_cartesian({0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian({0.5, 1.5});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    source.set_owned_value(0, 2.0);
    source.set_owned_value(1, 12.0);

    Transfer nearest(source_mesh, target_mesh, with_method(Method::NearestCell));
    nearest.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 2.0);

    Options options;
    options.neighbors = 1;
    Transfer idw(source_mesh, target_mesh, options);
    idw.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 2.0);
}

TEST(MeshToMeshTransferTest, ExactCentroidsPreserveValuesOnDistinctAndIdenticalMeshes)
{
    auto source_mesh = make_cartesian({0.0, 0.25, 1.0, 3.0});
    auto target_mesh = make_cartesian({0.0, 0.25, 1.0, 3.0});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    for (size_t row = 0; row < source.num_owned_cells(); ++row)
    {
        source.set_owned_value(static_cast<LO>(row), 1.0 + 3.0 * row);
    }
    Transfer between(source_mesh, target_mesh);
    between.apply(source, target);
    Transfer inplace(source_mesh, source_mesh);
    inplace.apply(source, source);
    for (size_t row = 0; row < source.num_owned_cells(); ++row)
    {
        EXPECT_DOUBLE_EQ(source.value(static_cast<LO>(row)), 1.0 + 3.0 * row);
        EXPECT_DOUBLE_EQ(target.value(static_cast<LO>(row)), 1.0 + 3.0 * row);
    }
}

TEST(MeshToMeshTransferTest, TransfersAllVectorAndTensorComponents)
{
    auto source_mesh = make_cartesian({0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian({0.5, 1.0});
    VectorField source_vector(source_mesh, "source_vector");
    VectorField target_vector(target_mesh, "target_vector");
    source_vector.set_owned_value(0, {1.0, 2.0, 3.0});
    source_vector.set_owned_value(1, {11.0, 22.0, 33.0});
    TensorField source_tensor(source_mesh, "source_tensor");
    TensorField target_tensor(target_mesh, "target_tensor");
    for (size_t component = 0; component < 9; ++component)
    {
        source_tensor.set_owned_component_value(0, component, component);
        source_tensor.set_owned_component_value(1, component, component + 10.0);
    }

    Transfer transfer(source_mesh, target_mesh);
    transfer.apply(source_vector, target_vector);
    transfer.apply(source_tensor, target_tensor);
    for (size_t component = 0; component < 3; ++component)
    {
        EXPECT_NEAR(target_vector.component_value(0, component),
                    2.0 * (component + 1.0), 1.0e-13);
        EXPECT_NEAR(target_vector.local_component_value(0, component),
                    2.0 * (component + 1.0), 1.0e-13);
    }
    for (size_t component = 0; component < 9; ++component)
    {
        EXPECT_NEAR(target_tensor.component_value(0, component),
                    component + 1.0, 1.0e-13);
        EXPECT_NEAR(target_tensor.local_component_value(0, component),
                    component + 1.0, 1.0e-13);
    }
}

TEST(MeshToMeshTransferTest, RawMultivectorsSupportSeveralColumnsAndValidateLayouts)
{
    auto source_mesh = make_cartesian({0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian({0.5, 1.0});
    Pack::multi_vector_type source(source_mesh->owned_cell_map(), 2);
    Pack::multi_vector_type target(target_mesh->owned_cell_map(), 2);
    {
        auto values = source.getLocalViewHost(Tpetra::Access::OverwriteAll);
        values(0, 0) = 2.0;
        values(1, 0) = 12.0;
        values(0, 1) = 4.0;
        values(1, 1) = -6.0;
    }
    Transfer transfer(source_mesh, target_mesh);
    transfer.apply(source, target);
    {
        const auto values = target.getLocalViewHost(Tpetra::Access::ReadOnly);
        EXPECT_NEAR(values(0, 0), 3.0, 1.0e-14);
        EXPECT_NEAR(values(0, 1), 3.0, 1.0e-14);
    }

    Pack::multi_vector_type wrong_columns(target_mesh->owned_cell_map(), 1);
    Pack::multi_vector_type wrong_target_map(source_mesh->owned_cell_map(), 2);
    Pack::multi_vector_type wrong_source_map(target_mesh->owned_cell_map(), 2);
    EXPECT_THROW(transfer.apply(source, wrong_columns), std::invalid_argument);
    EXPECT_THROW(transfer.apply(source, wrong_target_map), std::invalid_argument);
    EXPECT_THROW(transfer.apply(wrong_source_map, target), std::invalid_argument);
}

TEST(MeshToMeshTransferTest, RejectsNullMeshesAndInvalidOptions)
{
    auto mesh = make_cartesian({0.0, 1.0, 2.0});
    EXPECT_THROW((Transfer(nullptr, mesh)), std::invalid_argument);
    EXPECT_THROW((Transfer(mesh, nullptr)), std::invalid_argument);

    Options options;
    options.neighbors = 0;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options = Options{};
    options.max_distance = -1.0;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options.max_distance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options = Options{};
    options.coverage_tolerance = -1.0;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options.coverage_tolerance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options = Options{};
    options.method = static_cast<Method>(-1);
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
}

TEST(MeshToMeshTransferTest, NonconstantStrideViewsIgnoreUnselectedColumns)
{
    auto source_mesh = make_cartesian({0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian({0.5, 1.0});
    Pack::multi_vector_type source_storage(source_mesh->owned_cell_map(), 3);
    Pack::multi_vector_type target_storage(target_mesh->owned_cell_map(), 3);
    source_storage.putScalar(std::numeric_limits<double>::quiet_NaN());
    target_storage.putScalar(-11.0);
    {
        auto values = source_storage.getLocalViewHost(Tpetra::Access::ReadWrite);
        values(0, 0) = 2.0;
        values(1, 0) = 12.0;
        values(0, 2) = 4.0;
        values(1, 2) = -6.0;
    }
    Teuchos::Array<size_t> columns{0, 2};
    auto source = source_storage.subView(columns());
    auto target = target_storage.subViewNonConst(columns());
    ASSERT_FALSE(source->isConstantStride());
    ASSERT_FALSE(target->isConstantStride());
    Transfer transfer(source_mesh, target_mesh);
    transfer.apply(*source, *target);
    const auto values = target_storage.getLocalViewHost(Tpetra::Access::ReadOnly);
    EXPECT_NEAR(values(0, 0), 3.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(values(0, 1), -11.0);
    EXPECT_NEAR(values(0, 2), 3.0, 1.0e-14);
}

TEST(MeshToMeshTransferTest, InterpolatesNativeUnstructuredAndStkIntoCartesian)
{
    auto unstructured = std::make_shared<Handle>(
        SimpleFluid::test::make_unstructured_hex_line(2));
    auto stk = std::make_shared<Handle>(SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database()));
    auto target_mesh = make_cartesian({0.5, 1.0});
    for (const auto& source_mesh : {unstructured, stk})
    {
        ScalarField source(source_mesh, "source");
        ScalarField target(target_mesh, "target");
        for (size_t row = 0; row < source.num_owned_cells(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            source.set_owned_value(cell,
                source_mesh->cell_centroid(cell).x < 1.0 ? 2.0 : 12.0);
        }
        Transfer transfer(source_mesh, target_mesh);
        transfer.apply(source, target);
        EXPECT_NEAR(target.value(0), 3.0, 1.0e-13);
    }
}

TEST(MeshToMeshTransferTest, ConservativeProjectionRespectsReorderedGeometryIds)
{
    using Factory = SimpleFluid::MeshReorderingFactory<Pack>;
    auto coarse = make_cartesian({0.0, 1.0, 3.0});
    auto fine = make_cartesian({0.0, 0.5, 2.0, 3.0});
    const auto source_layout = Factory::selected_cells_first(std::move(coarse),
        [](Pack::global_ordinal_type, const Handle::Vec3& center)
        { return center.x > 1.0; });
    const auto target_layout = Factory::selected_cells_first(std::move(fine),
        [](Pack::global_ordinal_type, const Handle::Vec3& center)
        { return center.x > 2.0; });
    ASSERT_TRUE(source_layout.mesh->has_reordered_cells());
    ASSERT_TRUE(target_layout.mesh->has_reordered_cells());
    ScalarField source(source_layout.mesh, "source");
    ScalarField target(target_layout.mesh, "target");
    for (size_t row = 0; row < source.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        source.set_owned_value(cell,
            source.mesh().cell_centroid(cell).x < 1.0 ? 2.0 : 8.0);
    }
    Transfer transfer(source_layout.mesh, target_layout.mesh,
                      with_method(Method::ConservativeCellAverage));
    transfer.apply(source, target);
    for (size_t row = 0; row < target.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const double x = target.mesh().cell_centroid(cell).x;
        EXPECT_NEAR(target.value(cell), x < 0.5 ? 2.0 : x < 2.0 ? 6.0 : 8.0,
                    1.0e-14);
    }
    EXPECT_NEAR(integral(target), integral(source), 1.0e-13);
}

TEST(MeshToMeshTransferTest, RejectsWrongFieldMeshEvenWhenMapsMatch)
{
    auto mesh = make_cartesian({0.0, 1.0, 2.0});
    auto other_mesh = make_cartesian({0.0, 1.0, 2.0});
    ASSERT_TRUE(mesh->owned_cell_map()->isSameAs(*other_mesh->owned_cell_map()));
    ScalarField field(mesh, "field");
    ScalarField unrelated(other_mesh, "unrelated");
    Transfer transfer(mesh, mesh);
    EXPECT_THROW(transfer.apply(unrelated, field), std::invalid_argument);
    EXPECT_THROW(transfer.apply(field, unrelated), std::invalid_argument);
}

TEST(MeshToMeshTransferTest, DistanceLimitRejectsTargetsWithoutSourceSupport)
{
    auto source = make_cartesian({0.0, 1.0, 2.0});
    auto target = make_cartesian({0.5, 1.5});
    Options options;
    options.max_distance = 0.49;
    EXPECT_THROW((Transfer(source, target, options)), std::invalid_argument);
    options.method = Method::NearestCell;
    EXPECT_THROW((Transfer(source, target, options)), std::invalid_argument);
    options.max_distance = 0.5;
    EXPECT_NO_THROW((Transfer(source, target, options)));
}

TEST(MeshToMeshTransferTest, ConservativeRequiresCompleteCoverageInBothDirections)
{
    const auto options = with_method(Method::ConservativeCellAverage);
    auto full = make_cartesian({0.0, 1.0, 2.0});
    auto partial = make_cartesian({0.0, 0.5, 1.0});
    auto shifted = make_cartesian({0.5, 1.5, 2.5});
    EXPECT_THROW((Transfer(full, partial, options)), std::invalid_argument);
    EXPECT_THROW((Transfer(partial, full, options)), std::invalid_argument);
    EXPECT_THROW((Transfer(full, shifted, options)), std::invalid_argument);
}

TEST(MeshToMeshTransferTest, CylindricalTransferWorksButMixedBoxProjectionIsRejected)
{
    auto geometry = std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {1.0, 2.0, 3.0}, {0.0, std::numbers::pi}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    ScalarField source(mesh, "source");
    ScalarField target(mesh, "target");
    source.set_owned_value(0, 3.0);
    source.set_owned_value(1, 9.0);
    Transfer interpolation(mesh, mesh);
    interpolation.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 3.0);
    EXPECT_DOUBLE_EQ(target.value(1), 9.0);
    Transfer projection(mesh, mesh, with_method(Method::ConservativeCellAverage));
    projection.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 3.0);
    EXPECT_DOUBLE_EQ(target.value(1), 9.0);
    auto cartesian = make_cartesian({1.0, 2.0, 3.0});
    EXPECT_THROW((Transfer(mesh, cartesian, with_method(Method::ConservativeCellAverage))),
                 std::invalid_argument);
    EXPECT_THROW((Transfer(cartesian, mesh, with_method(Method::ConservativeCellAverage))),
                 std::invalid_argument);
}

TEST(MeshToMeshTransferTest, RejectsNonfiniteValuesBeforeChangingTarget)
{
    auto mesh = make_cartesian({0.0, 1.0, 2.0});
    ScalarField source(mesh, "source");
    ScalarField target(mesh, "target");
    target.put_scalar(19.0);
    Transfer transfer(mesh, mesh);
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity()})
    {
        source.set_owned_value(1, invalid);
        EXPECT_THROW(transfer.apply(source, target), std::invalid_argument);
        EXPECT_DOUBLE_EQ(target.value(0), 19.0);
        EXPECT_DOUBLE_EQ(target.value(1), 19.0);
    }
}

TEST(MeshToMeshTransferTest, MeshMotionInvalidatesPlansUntilRebuilt)
{
    auto source_mesh = make_cartesian({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0});
    auto target_mesh = make_cartesian({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0});
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    source.put_scalar(7.0);
    Transfer before_source_motion(source_mesh, target_mesh);
    SimpleFluid::PlanarALEMeshMotion<Pack> source_motion(source_mesh);
    source_motion.begin_trial(3.0, 1.0);
    EXPECT_THROW(before_source_motion.apply(source, target), std::runtime_error);
    source_motion.rollback_trial();
    EXPECT_THROW(before_source_motion.apply(source, target), std::runtime_error);

    Transfer before_target_motion(source_mesh, target_mesh);
    SimpleFluid::PlanarALEMeshMotion<Pack> target_motion(target_mesh);
    target_motion.begin_trial(3.0, 1.0);
    EXPECT_THROW(before_target_motion.apply(source, target), std::runtime_error);
    target_motion.accept_trial();
    Transfer rebuilt(source_mesh, target_mesh);
    rebuilt.apply(source, target);
    for (size_t row = 0; row < target.num_owned_cells(); ++row)
    {
        EXPECT_NEAR(target.value(static_cast<LO>(row)), 7.0, 1.0e-14);
    }
}
