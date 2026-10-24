/** @file testSpeciesLiquidVolumePacket.cc
 * @brief independent two-cell BE/upwind oracle for pre-source EOS volume packets.
 * Proposed serial/2-rank/4-rank execution, including actual empty owners.
 */
#include "equations/ReactingGasModel.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Pack = DefaultTpetraTypes;
using Grid = MeshHandle<Pack>;
using Model = ReactingGasModel<Pack, Grid>;
using LO = Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr R::SpeciesId a = 19, b = 811;
constexpr double q = .1, dt = .2, volume = .5;
double rho(double x)
{
    return 1000 * (1 + .25 * x * (1 - x));
}
struct Fixture
{
    SP<Grid> grid;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    ReactingGasOptions options;
    std::unique_ptr<Model> model;
    std::unique_ptr<Model::field_type> temperature, pressure;
    std::unique_ptr<Model::velocity_field_type> velocity;
    std::unique_ptr<Model::face_flux_field_type> flux;
    std::unique_ptr<MaterialPropertyFields<Pack, Grid>> material;
    Fixture()
    {
        Grid::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        grid = std::make_shared<Grid>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                          Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{{a, "finite_A", .02, {{"X", 1}}, {R::SpeciesPhase::Liquid}},
                {b, "finite_B", .02, {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
        options.mode = ReactingGasMode::Species;
        options.reference_pressure = 100000;
        options.surface_tension = 0;
        options.min_radius = 1e-10;
        options.max_radius = .1;
        options.solution_transport = ReactingTransportMode::Advective;
        options.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{
                {7, "isomerization", {{a, R::SpeciesPhase::Liquid, -1, {}}, {b, R::SpeciesPhase::Liquid, 1, {}}}, 0}},
            R::EnergyConvention::SensibleEnergy);
        const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
        options.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
            catalog, bounds, [](std::span<const double> x)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, rho(x[0]), 4000, 0, 1e-9, 0, 0, 0}; },
            R::EnergyConvention::SensibleEnergy);
        options.gas_thermodynamics = R::make_ideal_gas_mixture(
            catalog, {{20, 300, 0}, {20, 300, 0}}, bounds, R::EnergyConvention::SensibleEnergy);
        options.species_source = [](const R::ReactionContext& context, double)
        {
            R::ReactionSourceInput source;
            source.integratedExtents = {{7, .5 * context.state.amount(context.catalog, a, R::SpeciesPhase::Liquid)}};
            return source;
        };
        model = std::make_unique<Model>(grid, options);
        temperature = std::make_unique<Model::field_type>(grid, 300., "T");
        pressure = std::make_unique<Model::field_type>(grid, 0., "p");
        velocity = std::make_unique<Model::velocity_field_type>(grid, vec3<double>{}, "U");
        flux = std::make_unique<Model::face_flux_field_type>(grid, 0., "phi");
        BoussinesqModelOptions physics;
        physics.reference_density = physics.density = 1000;
        physics.specific_heat_capacity = 4000;
        physics.dynamic_viscosity = .001;
        physics.thermal_conductivity = 0;
        material = std::make_unique<MaterialPropertyFields<Pack, Grid>>(grid, physics, TimeStepperOptions{});
        std::vector<R::CellSpeciesState> initial;
        std::vector<std::vector<double>> numbers;
        for (std::size_t c = 0; c < grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const double x = grid->cell_centroid(cell).z < .5 ? .8 : .3;
            R::CellSpeciesState state(*catalog);
            state.dispersedPopulations = {{"micro", std::vector<double>(2)}, {"large", std::vector<double>(2)}};
            state.set_amount(*catalog, a, R::SpeciesPhase::Liquid, volume * rho(x) * x / .02);
            state.set_amount(*catalog, b, R::SpeciesPhase::Liquid, volume * rho(x) * (1 - x) / .02);
            initial.push_back(std::move(state));
            numbers.push_back({0, 0});
        }
        model->initialize_species(initial, numbers);
        model->initialize_state(0, *temperature, *pressure, *velocity, *material);
        for (const auto face : flux->owned_face_ids())
        {
            if (grid->is_geometry_exterior_face(face))
                continue;
            const double sign = grid->face_area_vector(face).z > 0 ? 1 : -1;
            flux->set_owned_value(face, sign * q);
        }
        flux->sync_ghosts();
    }
};
TEST(SpeciesLiquidVolumePacketTest, UsesCorrectedUpwindMolesAndPreSourceNonlinearComposition)
{
    Fixture f;
    f.model->advance(dt, dt, *f.temperature, *f.pressure, *f.velocity, *f.flux, *f.material);
    const double expected = q / (1 + q * dt / volume);
    const auto& packet = f.model->transported_liquid_material_volume_flux();
    const auto& receipt = f.model->liquid_transport_volume_receipt();
    EXPECT_TRUE(receipt.complete_carrier);
    EXPECT_EQ(receipt.event, f.model->species_event_receipt());
    EXPECT_EQ(receipt.geometry_epoch, mesh_geometry_epoch(*f.grid));
    int local_faces = 0, faces = 0;
    for (const auto face : packet.owned_face_ids())
    {
        if (f.grid->is_geometry_exterior_face(face))
            EXPECT_DOUBLE_EQ(packet.value(face), 0.);
        else
        {
            ++local_faces;
            const double sign = f.grid->face_area_vector(face).z > 0 ? 1 : -1;
            EXPECT_NEAR(packet.value(face), sign * expected, 2e-13);
            const auto donor = f.grid->cell_centroid(f.grid->owner_cell(face)).z < .5
                                   ? f.grid->owner_cell(face)
                                   : f.grid->opposite_or_periodic_neighbor_cell(face, f.grid->owner_cell(face));
            const auto final_mass = f.model->liquid_mass_density().local_value(donor);
            const auto final_density = f.model->pure_liquid_density().local_value(donor);
            EXPECT_GT(std::abs(packet.value(face) - sign * q * final_mass / final_density), 1e-4);
        }
    }
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &faces);
    EXPECT_EQ(faces, 1);
    int local_empty = f.grid->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (f.grid->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(empty, 2);
}
TEST(SpeciesLiquidVolumePacketTest, PressureThermalRefreshAndSnapshotDoNotInventNewTransport)
{
    Fixture f;
    f.model->advance(dt, dt, *f.temperature, *f.pressure, *f.velocity, *f.flux, *f.material);
    const auto accepted = f.model->snapshot();
    const auto receipt = f.model->species_event_receipt();
    const auto packet_receipt = f.model->liquid_transport_volume_receipt();
    std::vector<double> packet;
    for (const auto face : f.flux->owned_face_ids())
        packet.push_back(f.model->transported_liquid_material_volume_flux().value(face));
    f.model->set_absolute_pressure_offset(110000);
    f.temperature->put_scalar(310);
    f.model->refresh_species_thermodynamics(dt, *f.temperature, *f.pressure, *f.velocity, *f.material);
    EXPECT_EQ(f.model->species_event_receipt(), receipt);
    EXPECT_EQ(f.model->liquid_transport_volume_receipt(), packet_receipt);
    std::size_t index = 0;
    for (const auto face : f.flux->owned_face_ids())
        EXPECT_DOUBLE_EQ(f.model->transported_liquid_material_volume_flux().value(face), packet[index++]);
    f.model->restore(accepted);
    EXPECT_EQ(f.model->species_event_receipt(), receipt);
    EXPECT_EQ(f.model->liquid_transport_volume_receipt(), packet_receipt);

    PlanarALEMeshMotion<Pack> motion(f.grid);
    motion.begin_trial(1.05, .1);
    f.model->refresh_geometry();
    EXPECT_NE(f.model->liquid_transport_volume_receipt().geometry_epoch, mesh_geometry_epoch(*f.grid));
    motion.rollback_trial();
    f.model->restore(accepted);
    // Restoring physical stocks does not bless a packet for a new epoch. The
    // outer ALE replay must recompute it from these stocks at its own T/p guess.
    EXPECT_EQ(f.model->species_event_receipt(), receipt);
    EXPECT_NE(f.model->liquid_transport_volume_receipt().geometry_epoch, mesh_geometry_epoch(*f.grid));
}
} // namespace
