#include <gtest/gtest.h>

#include "equations/FissionPowerSource.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using SourceType = SimpleFluid::FissionPowerSource<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_unit_box(size_t cells = 3)
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            cells, cells, cells,
            1.0 / static_cast<double>(cells)));
}

SimpleFluid::SP<MeshType> make_cylinder()
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{0.35});
    database->set(
        "domain_type",
        static_cast<int>(
            SimpleFluid::MeshFactory::DomainType::CYLINDER));
    database->set("radius", SimpleFluid::real_t{1.0});
    database->set("height", SimpleFluid::real_t{2.0});
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    return SimpleFluid::test::build_mesh<Pack>(database);
}

TEST(FissionPowerSourceTest, ParsesDisabledConstantAndGaussianOptions)
{
    SimpleFluid::Database empty;
    EXPECT_EQ(
        SimpleFluid::fission_power_source_options_from_database(empty).profile,
        SimpleFluid::FissionPowerProfile::Disabled);

    SimpleFluid::Database constant;
    constant.set("fission_power_mode", std::string{"constant"});
    constant.set(
        "fission_power_density", SimpleFluid::real_t{12.5});
    const auto constant_options =
        SimpleFluid::fission_power_source_options_from_database(
            constant);
    EXPECT_EQ(
        constant_options.profile,
        SimpleFluid::FissionPowerProfile::Constant);
    EXPECT_DOUBLE_EQ(constant_options.power_density, 12.5);

    SimpleFluid::Database gaussian;
    gaussian.set("fission_power_mode", std::string{"GAUSSIAN"});
    gaussian.set("fission_total_power", SimpleFluid::real_t{40.0});
    gaussian.set(
        "fission_center",
        SimpleFluid::ArrReal{1.0, 2.0, 3.0});
    gaussian.set(
        "fission_standard_deviation",
        SimpleFluid::ArrReal{0.5, 1.0, 2.0});
    const auto gaussian_options =
        SimpleFluid::fission_power_source_options_from_database(
            gaussian);
    EXPECT_EQ(
        gaussian_options.profile,
        SimpleFluid::FissionPowerProfile::Gaussian);
    EXPECT_EQ(
        gaussian_options.center,
        (SimpleFluid::vec3<>{1.0, 2.0, 3.0}));
    EXPECT_EQ(
        gaussian_options.standard_deviation,
        (SimpleFluid::vec3<>{0.5, 1.0, 2.0}));
}

TEST(FissionPowerSourceTest, RejectsInvalidDatabaseOptions)
{
    auto expect_invalid =
        [](SimpleFluid::Database database)
    {
        EXPECT_THROW(
            SimpleFluid::fission_power_source_options_from_database(
                database),
            std::invalid_argument);
    };

    SimpleFluid::Database unknown;
    unknown.set("fission_power_mode", std::string{"point"});
    expect_invalid(unknown);

    SimpleFluid::Database missing_constant;
    missing_constant.set(
        "fission_power_mode", std::string{"constant"});
    expect_invalid(missing_constant);

    SimpleFluid::Database wrong_type;
    wrong_type.set(
        "fission_power_mode", std::string{"constant"});
    wrong_type.set(
        "fission_power_density", std::string{"high"});
    expect_invalid(wrong_type);

    SimpleFluid::Database negative;
    negative.set(
        "fission_power_mode", std::string{"constant"});
    negative.set(
        "fission_power_density", SimpleFluid::real_t{-1.0});
    expect_invalid(negative);

    SimpleFluid::Database missing_gaussian;
    missing_gaussian.set(
        "fission_power_mode", std::string{"gaussian"});
    missing_gaussian.set(
        "fission_total_power", SimpleFluid::real_t{1.0});
    expect_invalid(missing_gaussian);

    SimpleFluid::Database malformed_center;
    malformed_center.set(
        "fission_power_mode", std::string{"gaussian"});
    malformed_center.set(
        "fission_total_power", SimpleFluid::real_t{1.0});
    malformed_center.set(
        "fission_center", SimpleFluid::ArrReal{0.0, 0.0});
    malformed_center.set(
        "fission_standard_deviation",
        SimpleFluid::ArrReal{1.0, 1.0, 1.0});
    expect_invalid(malformed_center);

    SimpleFluid::Database invalid_width;
    invalid_width.set(
        "fission_power_mode", std::string{"gaussian"});
    invalid_width.set(
        "fission_total_power", SimpleFluid::real_t{1.0});
    invalid_width.set(
        "fission_center",
        SimpleFluid::ArrReal{0.0, 0.0, 0.0});
    invalid_width.set(
        "fission_standard_deviation",
        SimpleFluid::ArrReal{1.0, 0.0, 1.0});
    expect_invalid(invalid_width);

    SimpleFluid::Database non_finite;
    non_finite.set(
        "fission_power_mode", std::string{"gaussian"});
    non_finite.set(
        "fission_total_power",
        std::numeric_limits<SimpleFluid::real_t>::infinity());
    non_finite.set(
        "fission_center",
        SimpleFluid::ArrReal{0.0, 0.0, 0.0});
    non_finite.set(
        "fission_standard_deviation",
        SimpleFluid::ArrReal{1.0, 1.0, 1.0});
    expect_invalid(non_finite);
}

TEST(FissionPowerSourceTest, ConstantAndGaussianProfilesIntegrateCorrectly)
{
    auto mesh = make_unit_box();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    SourceType source(mesh, registry);

    source.initialize_constant(8.0);
    EXPECT_NEAR(source.integrated_power(), 8.0, 1.0e-12);

    source.initialize_gaussian(
        27.0,
        {0.5, 0.5, 0.5},
        {0.2, 0.35, 0.5});
    EXPECT_NEAR(source.integrated_power(), 27.0, 1.0e-11);

    auto value_at =
        [&](double x, double y, double z)
    {
        for (size_t owned = 0;
             owned < mesh->num_owned_cells();
             ++owned)
        {
            const auto lid =
                static_cast<Pack::local_ordinal_type>(owned);
            const auto centroid = mesh->cell_centroid(lid);
            if (std::abs(centroid.x - x) < 1.0e-12
                && std::abs(centroid.y - y) < 1.0e-12
                && std::abs(centroid.z - z) < 1.0e-12)
            {
                return source.field().value(lid);
            }
        }
        throw std::runtime_error("Requested centroid was not found.");
    };

    const auto center = value_at(0.5, 0.5, 0.5);
    EXPECT_GT(center, value_at(1.0 / 6.0, 0.5, 0.5));
    EXPECT_NEAR(
        value_at(1.0 / 6.0, 0.5, 0.5),
        value_at(5.0 / 6.0, 0.5, 0.5),
        1.0e-12);
    EXPECT_LT(
        value_at(1.0 / 6.0, 0.5, 0.5),
        value_at(0.5, 1.0 / 6.0, 0.5));
    EXPECT_LT(
        value_at(0.5, 1.0 / 6.0, 0.5),
        value_at(0.5, 0.5, 1.0 / 6.0));
}

TEST(FissionPowerSourceTest, GaussianIsAxisymmetricOnCylinder)
{
    auto mesh = make_cylinder();
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    SourceType source(mesh, registry);
    const SimpleFluid::vec3<> center{0.0, 0.0, 1.0};
    const SimpleFluid::vec3<> width{0.3, 0.3, 0.6};
    source.initialize_gaussian(50.0, center, width);

    EXPECT_NEAR(source.integrated_power(), 50.0, 5.0e-11);
    double normalization = -1.0;
    for (size_t owned = 0;
         owned < mesh->num_owned_cells();
         ++owned)
    {
        const auto lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto centroid = mesh->cell_centroid(lid);
        const auto exponent =
            (centroid.x * centroid.x + centroid.y * centroid.y)
                / (width.x * width.x)
          + ((centroid.z - center.z) * (centroid.z - center.z))
                / (width.z * width.z);
        const auto weight = std::exp(-0.5 * exponent);
        const auto ratio = source.field().value(lid) / weight;
        if (normalization < 0.0)
            normalization = ratio;
        EXPECT_NEAR(ratio, normalization, normalization * 1.0e-12);
    }
}

TEST(FissionPowerSourceTest, CopiesAndNormalizesCallerOwnedFields)
{
    auto mesh = make_unit_box(2);
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    SourceType source(mesh, registry);
    FieldType supplied(mesh, "supplied");
    FieldType shape(mesh, "shape");
    for (size_t owned = 0;
         owned < mesh->num_owned_cells();
         ++owned)
    {
        const auto lid =
            static_cast<Pack::local_ordinal_type>(owned);
        supplied.set_owned_value(lid, static_cast<double>(owned + 1));
        shape.set_owned_value(lid, static_cast<double>(owned));
    }
    supplied.sync_ghosts();
    shape.sync_ghosts();

    source.initialize_from_power_density(supplied);
    supplied.put_scalar(0.0);
    EXPECT_DOUBLE_EQ(source.field().value(0), 1.0);

    source.initialize_from_shape(shape, 12.0);
    EXPECT_NEAR(source.integrated_power(), 12.0, 1.0e-12);

    shape.put_scalar(0.0);
    EXPECT_THROW(
        source.initialize_from_shape(shape, 1.0),
        std::invalid_argument);
    EXPECT_NO_THROW(source.initialize_from_shape(shape, 0.0));
    EXPECT_DOUBLE_EQ(source.integrated_power(), 0.0);

    shape.put_scalar(-1.0);
    EXPECT_THROW(
        source.initialize_from_power_density(shape),
        std::invalid_argument);

    auto other_mesh = make_unit_box(2);
    FieldType other(other_mesh, 1.0, "other");
    EXPECT_THROW(
        source.initialize_from_power_density(other),
        std::invalid_argument);
}

TEST(FissionPowerSourceTest, AppliesTimeMultiplierAtUpdateTime)
{
    auto mesh = make_unit_box(1);
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    SourceType source(mesh, registry);
    source.initialize_constant(4.0);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    SimpleFluid::VectorCellField<Pack> velocity(
        mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        2.5, 7, *mesh, temperature, pressure, velocity};

    int calls = 0;
    source.set_time_multiplier(
        [&](const auto& update)
        {
            ++calls;
            EXPECT_DOUBLE_EQ(update.time, 2.5);
            EXPECT_EQ(update.step_index, 7);
            return 3.0;
        });
    registry.update(context);
    EXPECT_EQ(calls, 1);
    EXPECT_NEAR(source.integrated_power(), 12.0, 1.0e-12);

    source.set_time_multiplier(
        [](const auto&) { return -1.0; });
    EXPECT_THROW(registry.update(context), std::invalid_argument);
    source.set_time_multiplier(
        [](const auto&)
        {
            return std::numeric_limits<double>::quiet_NaN();
        });
    EXPECT_THROW(registry.update(context), std::invalid_argument);
}

TEST(FissionPowerSourceTest, SpecializedNameCannotUseGenericOperations)
{
    auto mesh = make_unit_box(1);
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);
    EXPECT_THROW(
        registry.add("qdot_fission", 1.0),
        std::invalid_argument);
    EXPECT_THROW(
        registry.remove("qdot_fission"),
        std::invalid_argument);

    {
        SourceType source(mesh, registry);
        EXPECT_NE(registry.find("qdot_fission"), nullptr);
    }
    EXPECT_EQ(registry.find("qdot_fission"), nullptr);
}

} // namespace
