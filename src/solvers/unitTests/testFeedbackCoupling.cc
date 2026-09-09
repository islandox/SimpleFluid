/**
 * @file testFeedbackCoupling.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for deterministic Phase 20 feedback exchange scaffolding.
 * @version 0.1
 * @date 2026-08-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/FeedbackMap.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <concepts>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using LegacyField = SimpleFluid::CellField<Pack>;
using NativeField = SimpleFluid::ScalarCellFieldStored<Pack>;
using FeedbackCell = SimpleFluid::FeedbackMap::FeedbackCell<Pack>;

/** @brief Distinct pack identity with the same enabled Tpetra value types. */
struct NonDefaultFeedbackPack : Pack
{
};

/** @brief Associate a legacy test field with a non-default pack identity. */
class NonDefaultPackFieldAdapter
{
public:
    using tpetra_type_pack = NonDefaultFeedbackPack;
    using scalar_type = typename tpetra_type_pack::scalar_type;
    using local_ordinal_type = typename tpetra_type_pack::local_ordinal_type;

    explicit NonDefaultPackFieldAdapter(LegacyField& field) : d_field(field) {}

    const auto& mesh() const noexcept { return d_field.mesh(); }

    void set_owned_value(local_ordinal_type cell_lid, scalar_type value) { d_field.set_owned_value(cell_lid, value); }

    void sync_ghosts() { d_field.sync_ghosts(); }

private:
    LegacyField& d_field;
};

static_assert(SimpleFluid::TpetraTypePack<NonDefaultFeedbackPack>);
static_assert(!std::same_as<NonDefaultFeedbackPack, Pack>);
static_assert(requires(NonDefaultPackFieldAdapter& field, const std::vector<double>& values) {
    SimpleFluid::FeedbackMap::import_power_density(field, values);
});

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<SimpleFluid::Mesh<Pack>> make_legacy_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
}

SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>> make_native_mesh()
{
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(std::move(cartesian));
}

} // namespace

/** @brief Registry exports canonical fields in deterministic name order. */
TEST(FeedbackFieldRegistryTest, ExportsCanonicalFieldsDeterministically)
{
    const auto mesh = make_legacy_mesh();
    LegacyField temperature(mesh, 0.0, "temperature");
    LegacyField gas_fraction(mesh, 0.0, "alpha_g");
    LegacyField density(mesh, 0.0, "rhoFeedback");
    LegacyField precursor(mesh, 0.0, "C_1");

    const std::array<double, 2> temperature_values{300.0, 340.0};
    const std::array<double, 2> gas_values{0.1, 0.3};
    const std::array<double, 2> density_values{1000.0, 900.0};
    const std::array<double, 2> precursor_values{2.0, 6.0};
    for (Pack::local_ordinal_type cell = 0; cell < 2; ++cell)
    {
        const auto index = static_cast<size_t>(cell);
        temperature.set_value(cell, temperature_values[index]);
        gas_fraction.set_value(cell, gas_values[index]);
        density.set_value(cell, density_values[index]);
        precursor.set_value(cell, precursor_values[index]);
    }

    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_density_feedback(density);
    registry.register_precursor_group(1, precursor);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    ASSERT_NO_THROW(registry.require_standard_fields(1));

    const std::vector<std::string> expected_names{"C_1", "T_liquid", "alpha_g", "rhoFeedback"};
    EXPECT_EQ(registry.field_names(), expected_names);

    const std::vector<FeedbackCell> feedback_cells{{"whole_core", {0, 1}}, {"inlet_cell", {0}}};
    const auto snapshot = registry.export_snapshot(feedback_cells, 7);
    EXPECT_EQ(snapshot.sequence_index(), 7u);
    EXPECT_EQ(snapshot.feedback_cell_names(), (std::vector<std::string>{"whole_core", "inlet_cell"}));

    for (const auto& [name, field] : std::vector<std::pair<std::string, const LegacyField*>>{
             {"C_1", &precursor}, {"T_liquid", &temperature}, {"alpha_g", &gas_fraction}, {"rhoFeedback", &density}})
    {
        const auto expected = SimpleFluid::FeedbackMap::volume_weighted_average<Pack>(*field, feedback_cells);
        EXPECT_EQ(snapshot.field(name), expected);
    }
    EXPECT_THROW(snapshot.field("missing"), std::out_of_range);
}

/** @brief Optional planar free-surface fields use stable external names. */
TEST(FeedbackFieldRegistryTest, ExportsPlanarFreeSurfaceFields)
{
    const auto mesh = make_legacy_mesh();
    LegacyField liquid_density(mesh, 998.0, "rhoLiquid");
    LegacyField clear_level(mesh, 0.7, "clearLevel");
    LegacyField pool_level(mesh, 0.72, "poolLevel");
    LegacyField headspace_pressure(mesh, 101325.0, "headspacePressure");
    LegacyField pool_occupancy(mesh, 1.0, "poolOccupancy");

    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_pure_liquid_density(liquid_density);
    registry.register_clear_level(clear_level);
    registry.register_pool_level(pool_level);
    registry.register_headspace_pressure(headspace_pressure);
    registry.register_pool_occupancy(pool_occupancy);

    ASSERT_NO_THROW(registry.require_planar_free_surface_fields());
    const std::vector<FeedbackCell> feedback_cells{{"whole_pool", {0, 1}}};
    const auto snapshot = registry.export_snapshot(feedback_cells);
    EXPECT_DOUBLE_EQ(snapshot.field("rhoLiquid").front(), 998.0);
    EXPECT_DOUBLE_EQ(snapshot.field("clearLevel").front(), 0.7);
    EXPECT_DOUBLE_EQ(snapshot.field("poolLevel").front(), 0.72);
    EXPECT_DOUBLE_EQ(snapshot.field("headspacePressure").front(), 101325.0);
    EXPECT_DOUBLE_EQ(snapshot.field("poolOccupancy").front(), 1.0);
}

/** @brief Native FieldStored fields share the same mapping and import API. */
TEST(FeedbackFieldRegistryTest, SupportsNativeFieldStoredAndMeshHandle)
{
    const auto mesh = make_native_mesh();
    NativeField temperature(mesh, 315.0, "temperature");
    NativeField gas_fraction(mesh, 0.02, "alpha_g");
    NativeField density(mesh, 980.0, "rhoFeedback");
    NativeField precursor(mesh, 4.0, "C_1");
    NativeField power(mesh, 0.0, "qdot_fission");

    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);
    registry.register_precursor_group(1, precursor);

    const std::vector<FeedbackCell> feedback_cells{{"core", {0}}};
    const auto snapshot = registry.export_snapshot(feedback_cells);
    ASSERT_EQ(snapshot.field("T_liquid").size(), 1u);
    EXPECT_DOUBLE_EQ(snapshot.field("T_liquid")[0], 315.0);
    EXPECT_DOUBLE_EQ(snapshot.field("alpha_g")[0], 0.02);
    EXPECT_DOUBLE_EQ(snapshot.field("rhoFeedback")[0], 980.0);
    EXPECT_DOUBLE_EQ(snapshot.field("C_1")[0], 4.0);

    SimpleFluid::FeedbackMap::import_power_density<Pack>(power, {25.0});
    EXPECT_DOUBLE_EQ(power.value(0), 25.0);
    EXPECT_THROW(SimpleFluid::FeedbackMap::import_power_density<Pack>(power, {-1.0}), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::FeedbackMap::import_power_density<Pack>(power, {}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(power.value(0), 25.0);
}

/** @brief Snapshot export rejects non-finite mapped field input. */
TEST(FeedbackFieldRegistryTest, RejectsNonFiniteMappedFieldValues)
{
    const auto mesh = make_legacy_mesh();
    LegacyField temperature(mesh, 300.0, "temperature");
    temperature.set_value(1, std::numeric_limits<double>::quiet_NaN());

    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    const std::vector<FeedbackCell> feedback_cells{{"whole_core", {0, 1}}};
    EXPECT_THROW((void) registry.export_snapshot(feedback_cells), std::invalid_argument);

    using Snapshot = SimpleFluid::FeedbackMap::MappedFeedbackSnapshot<Pack>;
    EXPECT_THROW((void) Snapshot(0, {"whole_core"}, {{"T_liquid", {std::numeric_limits<double>::infinity()}}}),
        std::invalid_argument);
}

/** @brief Power import is field-deduced for a non-default pack identity. */
TEST(FeedbackFieldRegistryTest, DeducesNonDefaultPackPowerImport)
{
    const auto mesh = make_legacy_mesh();
    LegacyField power(mesh, 0.0, "qdot_fission");
    NonDefaultPackFieldAdapter adapted_power(power);

    SimpleFluid::FeedbackMap::import_power_density(adapted_power, std::vector<double>{11.0, 13.0});
    EXPECT_DOUBLE_EQ(power.value(0), 11.0);
    EXPECT_DOUBLE_EQ(power.value(1), 13.0);
}

/** @brief Placeholder loop imports power, subcycles TH, and returns feedback. */
TEST(PlaceholderOuterCouplingDriverTest, RunsDeterministicExchangeLoop)
{
    const auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    LegacyField power(mesh, 0.0, "qdot_fission");
    LegacyField temperature(mesh, 0.0, "temperature");
    LegacyField gas_fraction(mesh, 0.1, "alpha_g");
    LegacyField density(mesh, 950.0, "rhoFeedback");
    LegacyField precursor(mesh, 3.0, "C_1");

    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);
    registry.register_gas_fraction(gas_fraction);
    registry.register_density_feedback(density);
    registry.register_precursor_group(1, precursor);

    SimpleFluid::FeedbackMap::PlaceholderOuterCouplingOptions options;
    options.outer_iterations = 3;
    options.thermal_hydraulic_subcycles = 2;
    options.precursor_group_count = 1;
    SimpleFluid::FeedbackMap::PlaceholderOuterCouplingDriver<Pack> driver(registry, {{"core", {0}}}, options);

    size_t thermal_hydraulic_calls = 0;
    size_t neutronics_calls = 0;
    const auto records = driver.run(
        power, {1.0},
        [&](size_t iteration, size_t subcycle)
        {
            EXPECT_LT(iteration, options.outer_iterations);
            EXPECT_LT(subcycle, options.thermal_hydraulic_subcycles);
            temperature.set_value(0, temperature.value(0) + power.value(0));
            ++thermal_hydraulic_calls;
        },
        [&](const auto& feedback)
        {
            EXPECT_EQ(feedback.sequence_index(), neutronics_calls);
            ++neutronics_calls;
            return std::vector<double>{static_cast<double>(neutronics_calls + 1)};
        });

    ASSERT_EQ(records.size(), 3u);
    EXPECT_EQ(thermal_hydraulic_calls, 6u);
    EXPECT_EQ(neutronics_calls, 3u);
    const std::array<double, 3> expected_temperature{2.0, 6.0, 12.0};
    for (size_t iteration = 0; iteration < records.size(); ++iteration)
    {
        EXPECT_EQ(records[iteration].iteration, iteration);
        ASSERT_EQ(records[iteration].power_applied.size(), 1u);
        ASSERT_EQ(records[iteration].power_returned.size(), 1u);
        EXPECT_DOUBLE_EQ(records[iteration].power_applied[0], static_cast<double>(iteration + 1));
        EXPECT_DOUBLE_EQ(records[iteration].power_returned[0], static_cast<double>(iteration + 2));
        EXPECT_DOUBLE_EQ(records[iteration].feedback.field("T_liquid")[0], expected_temperature[iteration]);
        EXPECT_DOUBLE_EQ(records[iteration].feedback.field("C_1")[0], 3.0);
    }
    EXPECT_DOUBLE_EQ(power.value(0), 4.0);
}

/** @brief Driver rejects an incomplete canonical feedback contract. */
TEST(PlaceholderOuterCouplingDriverTest, RequiresCanonicalFeedbackFields)
{
    const auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    LegacyField temperature(mesh, 300.0, "temperature");
    SimpleFluid::FeedbackMap::FeedbackFieldRegistry<Pack> registry(*mesh);
    registry.register_liquid_temperature(temperature);

    SimpleFluid::FeedbackMap::PlaceholderOuterCouplingOptions options;
    options.precursor_group_count = 1;
    EXPECT_THROW((SimpleFluid::FeedbackMap::PlaceholderOuterCouplingDriver<Pack>(registry, {{"core", {0}}}, options)),
        std::invalid_argument);
}
