#include "equations/LumpedInnerCavityCoupling.hh"
#include "equations/SolidHeatConductionEquation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <gtest/gtest.h>

using namespace SimpleFluid;

namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
using Solid = SolidSubdomain<Pack>;
using Cells = ScalarCellFieldStored<Pack, MeshType>;
using Faces = ScalarFaceFieldStored<Pack, MeshType>;
using Batch = InnerCavityFieldBatch;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const MeshType> mesh()
{
    auto geometry = std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0.0, 1.0, 3.0, 6.0, 10.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<MeshType>(std::move(geometry));
}

LumpedInnerCavityOptions options()
{
    LumpedInnerCavityOptions o;
    o.saturation_pressure = [](real_t t) { return 3000.0 * std::exp(0.05 * (t - 300.0)); };
    return o;
}

InnerCavitySurface properties(std::string name, bool wet = false)
{
    InnerCavitySurface p;
    p.name = std::move(name);
    p.convection.heat_transfer_coefficient = 2.0;
    if (wet)
    {
        p.moisture_exchange = CavityMoistureExchange::LiquidSurface;
        p.mass_transfer_coefficient = 0.002;
    }
    return p;
}

real_t sum(const MeshType& mesh, real_t local)
{
    real_t global = 0.0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

template<class Field> real_t energy(const Field& field)
{
    real_t local = 0.0, global = 0.0;
    const auto& m = field.mesh();
    for (std::size_t c = 0; c < m.num_owned_cells(); ++c)
        local += 1000.0 * m.cell_volume(static_cast<int>(c)) * field.value(static_cast<int>(c));
    Teuchos::reduceAll(*m.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

template<class Mesh> MaterialPropertyFields<Pack, Mesh> material(SP<const Mesh> m)
{
    BoussinesqModelOptions o;
    o.reference_density = o.density = 1.0;
    o.specific_heat_capacity = 1000.0;
    o.thermal_conductivity = 2.0;
    return {m, o, TimeStepperOptions{}};
}
} // namespace

TEST(LumpedInnerCavityCouplingTest, CellAndFaceFieldsRetainNonlinearFaceExchange)
{
    const auto m = mesh();
    Cells cells(m, 300.0, "cell_temperature");
    Faces faces(m, 300.0, "face_temperature");
    for (std::size_t c = 0; c < m->num_owned_cells(); ++c)
        cells.set_owned_value(c, 300.0 + 2.0 * m->cell_centroid(c).x);
    for (const auto f : faces.owned_face_ids())
        faces.set_value(f, cells.value(m->owner_cell(f)));
    const auto p = properties("liquid", true);
    InnerCavityFieldInput cell_batch(m->owned_cell_map()->getComm(), {Batch::from_field(cells, "ymax", p)});
    InnerCavityFieldInput face_batch(m->owned_cell_map()->getComm(), {Batch::from_field(faces, "ymax", p)});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.005});
    const auto cell_result = cavity.evaluate(cell_batch);
    const auto face_result = cavity.evaluate(face_batch);
    ASSERT_EQ(cell_result.surfaces.size(), 4U);
    EXPECT_DOUBLE_EQ(cell_result.vapor_mass_rate, face_result.vapor_mass_rate);
    EXPECT_DOUBLE_EQ(cell_result.air_energy_rate, face_result.air_energy_rate);
    std::vector<InnerCavitySurface> reference;
    const std::array areas{1.0, 2.0, 3.0, 4.0}, temperatures{301.0, 304.0, 309.0, 316.0};
    for (std::size_t i = 0; i < 4; ++i)
    {
        auto s = p;
        s.name = std::to_string(i);
        s.area = areas[i];
        s.temperature = temperatures[i];
        reference.push_back(s);
    }
    const auto expected = cavity.evaluate(reference);
    EXPECT_NEAR(cell_result.vapor_mass_rate, expected.vapor_mass_rate, 1e-15);
    EXPECT_NEAR(cell_result.air_energy_rate, expected.air_energy_rate, 1e-9);
    real_t local_energy = 0.0;
    for (const auto& f : cell_result.boundaries[0].faces)
        local_energy += f.exchange.energy_rate;
    EXPECT_NEAR(sum(*m, local_energy), cell_result.air_energy_rate, 1e-9);

    const auto trial = cavity.preview(0.1, cell_batch);
    EXPECT_DOUBLE_EQ(cavity.state().temperature, 300.0);
    const auto accepted = cavity.advance(0.1, face_batch);
    EXPECT_DOUBLE_EQ(trial.state.temperature, accepted.state.temperature);
    EXPECT_DOUBLE_EQ(trial.state.humidity_ratio, accepted.state.humidity_ratio);
    cells.put_scalar(320.0);
    EXPECT_NE(cavity.evaluate(cell_batch).air_energy_rate, cavity.evaluate(face_batch).air_energy_rate);
}

TEST(LumpedInnerCavityCouplingTest, CopiedAndMovedBatchesKeepTheirSharedFieldBinding)
{
    const auto m = mesh();
    Cells cells(m, 310.0, "temperature");
    const auto input = [&]
    {
        const auto original = Batch::from_field(cells, "ymax", properties("wall"));
        auto copied = original;
        std::vector<Batch> batches;
        batches.push_back(std::move(copied));
        return InnerCavityFieldInput(m->owned_cell_map()->getComm(), std::move(batches));
    }();
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    const auto initial = cavity.evaluate(input);
    EXPECT_EQ(initial.surfaces.size(), 4U);
    ASSERT_EQ(initial.boundaries.size(), 1U);
    EXPECT_EQ(initial.boundaries[0].mesh.get(), m.get());
    EXPECT_EQ(initial.boundaries[0].batch_id, inner_cavity_batch_id(*m, "ymax"));
    EXPECT_DOUBLE_EQ(initial.air_energy_rate, 200.0);
    cells.put_scalar(320.0);
    EXPECT_DOUBLE_EQ(cavity.evaluate(input).air_energy_rate, 400.0);
}

TEST(LumpedInnerCavityCouplingTest, AreaAverageUsesAreaWeightsAndConservativeRedistribution)
{
    const auto m = mesh();
    Cells cells(m, 300.0, "cell_temperature");
    for (std::size_t c = 0; c < m->num_owned_cells(); ++c)
        cells.set_owned_value(c, 300.0 + 2.0 * m->cell_centroid(c).x);
    auto p = properties("pool", true);
    InnerCavityFieldInput batch(
        m->owned_cell_map()->getComm(), {Batch::from_field(cells, "ymax", p, InnerCavitySampling::AreaAverage)});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.005});
    const auto result = cavity.evaluate(batch);
    ASSERT_EQ(result.surfaces.size(), 1U);
    p.area = 10.0;
    p.temperature = 310.0;
    const std::array reference{p};
    const auto expected = cavity.evaluate(reference);
    EXPECT_DOUBLE_EQ(result.air_energy_rate, expected.air_energy_rate);
    for (const auto& f : result.boundaries[0].faces)
    {
        EXPECT_NEAR(f.exchange.vapor_mass_rate, result.vapor_mass_rate * f.area / 10.0, 1e-15);
        EXPECT_NEAR(f.exchange.energy_rate, result.air_energy_rate * f.area / 10.0, 1e-10);
    }
    InnerCavityFieldInput facewise(m->owned_cell_map()->getComm(), {Batch::from_field(cells, "ymax", p)});
    EXPECT_GT(cavity.evaluate(facewise).vapor_mass_rate, result.vapor_mass_rate);
}

TEST(LumpedInnerCavityCouplingTest, MixedModesAndSubdomainFieldTypesSupportRadiation)
{
    const auto m = mesh();
    auto solid = std::make_shared<const Solid>(mesh());
    Faces pool(m, 320.0, "pool_temperature");
    ScalarCellFieldStored<Pack, Solid> wall(solid, 290.0, "wall_temperature");
    auto a = properties("pool"), b = properties("wall");
    a.convection.heat_transfer_coefficient = b.convection.heat_transfer_coefficient = 0.0;
    // One face on xmin versus a patch average on the independent solid mesh.
    InnerCavityFieldInput batch(m->owned_cell_map()->getComm(),
        {Batch::from_field(pool, "xmin", a), Batch::from_field(wall, "xmax", b, InnerCavitySampling::AreaAverage)});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    const std::array factors{0.0, 1.0, 1.0, 0.0};
    const auto result = cavity.evaluate(batch, factors);
    ASSERT_EQ(result.surfaces.size(), 2U);
    ASSERT_EQ(result.boundaries.size(), 2U);
    const auto expected = 5.670374419e-8 * (std::pow(320.0, 4) - std::pow(290.0, 4));
    EXPECT_NEAR(result.surfaces[0].radiation_heat_rate, expected, 1e-10);
    EXPECT_NEAR(result.surfaces[1].radiation_heat_rate, -expected, 1e-10);
    EXPECT_DOUBLE_EQ(result.air_energy_rate, 0.0);
    FVM::MeshBoundaryCache<Pack, Solid> flux{{}, solid};
    refresh_inner_cavity_boundary_conditions(flux, result);
    real_t local = 0.0;
    for (const auto& [id, values] : flux.value)
        for (std::size_t i = 0; i < values.size(); ++i)
            if (solid->is_owned_face(solid->boundary_face_batch(id).face_lids[i]))
                local += values[i] * solid->face_area(solid->boundary_face_batch(id).face_lids[i]);
    EXPECT_NEAR(sum(*m, local), -expected, 1e-10);
}

TEST(LumpedInnerCavityCouplingTest, SolidRefreshBalancesCavityEnergyAndPreservesUnrelatedPatch)
{
    auto solid = std::make_shared<const Solid>(mesh());
    ScalarCellFieldStored<Pack, Solid> wall(solid, 320.0, "wall_temperature");
    InnerCavityFieldInput batch(
        solid->owned_cell_map()->getComm(), {Batch::from_field(wall, "ymax", properties("wall"))});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    auto mat = material<Solid>(solid);
    BoundaryConditionSet bc;
    bc.temperature["ymax"] = {BoundaryConditionType::Dirichlet, 1000.0}; // Replaced by flux override.
    SolidHeatConductionEquation<Pack> conduction(solid, bc);
    using Cache = SolidHeatConductionEquation<Pack>::boundary_cache_type;
    Cache initial_flux{{}, solid};
    for (const auto& [id, faces] : solid->boundary_batches())
        if (solid->boundary_batch_name(id) == "ymin")
            initial_flux.value[id] = ArrReal(faces.face_lids.size(), 0.0);
    SP<const Cache> flux = std::make_shared<const Cache>(std::move(initial_flux));
    const auto initial = energy(wall) + cavity.internal_energy();
    for (int i = 0; i < 2; ++i)
    {
        const auto trial = cavity.preview(0.1, batch);
        refresh_inner_cavity_boundary_conditions(conduction, flux, trial.exchange);
        EXPECT_EQ(conduction.boundary_heat_flux().get(), flux.get());
        EXPECT_TRUE(conduction.advance(wall, 0.1, mat, wall).converged);
        cavity.set_state(trial.state);
        EXPECT_NEAR(energy(wall) + cavity.internal_energy(), initial, 1e-6);
        for (const auto& [id, faces] : solid->boundary_batches())
            if (solid->boundary_batch_name(id) == "ymin")
                EXPECT_TRUE(flux->value.contains(id));
    }
}

TEST(LumpedInnerCavityCouplingTest, RefreshSelectsHeatOrTotalEnergyAndRollsBackOnInvalidMapping)
{
    const auto m = mesh();
    Faces pool(m, 320.0, "pool_temperature");
    InnerCavityFieldInput batch(
        m->owned_cell_map()->getComm(), {Batch::from_field(pool, "ymax", properties("pool", true))});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.005});
    const auto result = cavity.evaluate(batch);
    FVM::MeshBoundaryCache<Pack, MeshType> heat{{}, m}, total{{}, m};
    refresh_inner_cavity_boundary_conditions(heat, result);
    refresh_inner_cavity_boundary_conditions(total, result, InnerCavityEnergyTransfer::TotalEnergy);
    real_t local_difference = 0.0;
    for (const auto& [id, values] : heat.value)
        for (std::size_t i = 0; i < values.size(); ++i)
            if (m->is_owned_face(m->boundary_face_batch(id).face_lids[i]))
                local_difference +=
                    (total.value.at(id)[i] - values[i]) * m->face_area(m->boundary_face_batch(id).face_lids[i]);
    EXPECT_NEAR(sum(*m, local_difference), result.vapor_mass_rate * options().liquid_cp * (320.0 - 273.15), 1e-9);
    const auto saved = heat.value;
    auto bad = result;
    if (m->owned_cell_map()->getComm()->getRank() == 0)
        bad.boundaries[0].faces[0].area *= 2.0;
    EXPECT_ANY_THROW(refresh_inner_cavity_boundary_conditions(heat, bad));
    EXPECT_EQ(heat.value, saved);
    FVM::MeshBoundaryCache<Pack, MeshType> wrong{{}, mesh()};
    EXPECT_ANY_THROW(refresh_inner_cavity_boundary_conditions(wrong, result));
}

TEST(LumpedInnerCavityCouplingTest, InvalidEntryAndRankLocalSamplingFailurePreserveState)
{
    const auto m = mesh();
    Cells cells(m, 310.0, "cell_temperature");
    const auto comm = m->owned_cell_map()->getComm();
    auto p = properties("wall");
    InnerCavityFieldInput batch(comm, {Batch::from_field(cells, "ymax", p)});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.01});
    if (comm->getRank() == 0)
        cells.set_owned_value(0, std::numeric_limits<real_t>::quiet_NaN());
    EXPECT_ANY_THROW(cavity.advance(1.0, batch));
    EXPECT_DOUBLE_EQ(cavity.state().temperature, 300.0);
    EXPECT_DOUBLE_EQ(cavity.state().humidity_ratio, 0.01);
    cells.put_scalar(310.0);
    InnerCavityFieldInput missing(comm, {Batch::from_field(cells, 999999, p)});
    EXPECT_ANY_THROW(cavity.evaluate(missing));
    auto p2 = p;
    p2.name = "duplicate";
    InnerCavityFieldInput duplicate(comm, {Batch::from_field(cells, "ymax", p), Batch::from_field(cells, "ymax", p2)});
    EXPECT_ANY_THROW(cavity.preview(1.0, duplicate));
    p.convection_coefficient = [rank = comm->getRank()](real_t, real_t)
    {
        if (rank == 0)
            throw std::domain_error("closure failure");
        return 1.0;
    };
    InnerCavityFieldInput invalid_closure(comm, {Batch::from_field(cells, "ymax", p)});
    EXPECT_ANY_THROW(cavity.advance(1.0, invalid_closure));
    EXPECT_DOUBLE_EQ(cavity.state().temperature, 300.0);
}

TEST(LumpedInnerCavityCouplingTest, RankMismatchAndEmptyOwningRankAreHandledCollectively)
{
    const auto m = mesh();
    const auto comm = m->owned_cell_map()->getComm();
    Cells cells(m, 310.0, "cell_temperature");
    auto p = properties("wall");
    InnerCavityFieldInput batch(comm, {Batch::from_field(cells, "xmin", p)});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    const auto result = cavity.evaluate(batch);
    ASSERT_EQ(result.surfaces.size(), 1U);
    EXPECT_DOUBLE_EQ(result.air_energy_rate, 20.0);
    EXPECT_DOUBLE_EQ(sum(*m, result.boundaries[0].faces.size()), 1.0);
    if (comm->getSize() == 1)
        return;
    EXPECT_ANY_THROW(cavity.advance(comm->getRank() == 0 ? 1.0 : 2.0, batch));
    const auto mode = comm->getRank() == 0 ? InnerCavitySampling::Facewise : InnerCavitySampling::AreaAverage;
    InnerCavityFieldInput different(comm, {Batch::from_field(cells, "xmin", p, mode)});
    EXPECT_ANY_THROW(cavity.evaluate(different));
    p.convection_coefficient = [rank = comm->getRank()](real_t, real_t) { return 1.0 + rank; };
    InnerCavityFieldInput different_result(comm, {Batch::from_field(cells, "xmin", p)});
    EXPECT_ANY_THROW(cavity.advance(1.0, different_result));
    EXPECT_DOUBLE_EQ(cavity.state().temperature, 300.0);
}

TEST(LumpedInnerCavityCouplingTest, LegacyScalarFieldsSupportBothLocations)
{
    const auto m = test::build_mesh<Pack>(test::make_box_database(4, 1, 1));
    CellField<Pack> cells(m, 310.0);
    FaceField<Pack> faces(m, 310.0);
    InnerCavityFieldInput batch(m->owned_cell_map()->getComm(),
        {Batch::from_field(cells, "ymin", properties("cells")),
            Batch::from_field(faces, "ymax", properties("faces"), InnerCavitySampling::AreaAverage)});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    const auto result = cavity.evaluate(batch);
    EXPECT_EQ(result.surfaces.size(), 5U);
    EXPECT_DOUBLE_EQ(result.air_energy_rate, 160.0);
}

TEST(LumpedInnerCavityCouplingTest, CongruentContextsAreAcceptedAndReorderedRanksRejected)
{
    const auto m = mesh();
    const auto comm = m->owned_cell_map()->getComm();
    const auto duplicate = comm->duplicate();
    Cells cells(m, 310.0, "temperature");
    const auto id = inner_cavity_batch_id(*m, "ymax");
    InnerCavityFieldInput input(duplicate, {Batch::from_field(cells, id, properties("wall"))});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    EXPECT_DOUBLE_EQ(cavity.evaluate(input).air_energy_rate, 200.0);
    cells.put_scalar(320.0);
    EXPECT_DOUBLE_EQ(cavity.evaluate(input).air_energy_rate, 400.0);
#ifdef HAVE_TEUCHOS_MPI
    if (comm->getSize() > 1)
    {
        const auto reversed = comm->split(0, comm->getSize() - 1 - comm->getRank());
        InnerCavityFieldInput invalid(reversed, {Batch::from_field(cells, id, properties("wall"))});
        EXPECT_ANY_THROW(cavity.evaluate(invalid));
    }
#endif
}

TEST(LumpedInnerCavityCouplingTest, SamplingBorrowsClosuresWithoutPerFaceCopies)
{
    struct Closure
    {
        SP<int> copies;
        explicit Closure(SP<int> count) : copies(std::move(count)) {}
        Closure(const Closure& other) : copies(other.copies) { ++*copies; }
        real_t operator()(real_t, real_t) const { return 2.0; }
    };
    const auto m = mesh();
    Cells cells(m, 310.0, "temperature");
    auto copies = std::make_shared<int>(0);
    auto p = properties("wall");
    p.convection_coefficient = Closure(copies);
    for (const auto mode : {InnerCavitySampling::Facewise, InnerCavitySampling::AreaAverage})
    {
        InnerCavityFieldInput input(m->owned_cell_map()->getComm(), {Batch::from_field(cells, "ymax", p, mode)});
        *copies = 0;
        LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
        EXPECT_DOUBLE_EQ(cavity.evaluate(input).air_energy_rate, 200.0);
        cavity.preview(0.1, input);
        EXPECT_EQ(*copies, 0);
    }
}

TEST(LumpedInnerCavityCouplingTest, SharedFluxSnapshotsSupportRollbackAndRejectStaleSlots)
{
    const auto m = mesh();
    Cells cells(m, 310.0, "temperature");
    TemperatureDiffusionEquation<Pack, MeshType> equation(m, {});
    using Cache = FVM::MeshBoundaryCache<Pack, MeshType>;
    SP<const Cache> flux = std::make_shared<const Cache>(Cache{{}, m});
    InnerCavityFieldInput input(m->owned_cell_map()->getComm(), {Batch::from_field(cells, "ymax", properties("wall"))});
    LumpedInnerCavityModel cavity(options(), {300.0, 0.0});
    refresh_inner_cavity_boundary_conditions(equation, flux, cavity.evaluate(input));
    const auto checkpoint = flux;
    const auto values = checkpoint->value;
    cells.put_scalar(320.0);
    auto exchange = cavity.evaluate(input);
    refresh_inner_cavity_boundary_conditions(equation, flux, exchange);
    EXPECT_NE(flux.get(), checkpoint.get());
    EXPECT_EQ(flux.get(), equation.boundary_heat_flux().get());
    EXPECT_EQ(checkpoint->value, values);
    const auto accepted = flux;
    if (m->owned_cell_map()->getComm()->getRank() == 0)
        exchange.boundaries[0].faces[0].batch_slot = std::numeric_limits<std::size_t>::max();
    EXPECT_ANY_THROW(refresh_inner_cavity_boundary_conditions(equation, flux, exchange));
    EXPECT_EQ(flux.get(), accepted.get());
    EXPECT_EQ(flux.get(), equation.boundary_heat_flux().get());
    equation.set_boundary_heat_flux_snapshot(checkpoint);
    EXPECT_EQ(equation.boundary_heat_flux().get(), checkpoint.get());
    EXPECT_ANY_THROW(equation.set_boundary_heat_flux_snapshot(nullptr));
    EXPECT_EQ(equation.boundary_heat_flux().get(), checkpoint.get());
}

TEST(LumpedInnerCavityCouplingTest, PhysicalTemperatureFluxUpdatesAreTransactionalAndUseConductivityOverride)
{
    const auto m = mesh();
    Cells temperature(m, 320.0, "temperature"), conductivity(m, 4.0, "conductivity");
    Faces carrier(m, 0.0, "carrier_flux");
    auto mat = material(m);
    TemperatureDiffusionEquation<Pack, MeshType> equation(m, {});
    FVM::MeshBoundaryCache<Pack, MeshType> flux{{}, m};
    for (const auto& [id, batch] : m->boundary_batches())
        if (m->boundary_batch_name(id) == "ymax")
            flux.value[id] = ArrReal(batch.face_lids.size(), 10.0);
    equation.set_boundary_heat_flux(flux);
    auto invalid = flux;
    if (m->owned_cell_map()->getComm()->getRank() == 0)
        invalid.value.begin()->second.clear();
    EXPECT_ANY_THROW(equation.set_boundary_heat_flux(invalid));
    const auto initial = energy(temperature);
    LinearSolverOptions linear;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    EXPECT_TRUE(equation
            .advance_physical(
                temperature, carrier, 0.1, mat, temperature, [](int) { return 0.0; },
                FVM::NonOrthogonalTreatment::Implicit, linear, &conductivity)
            .converged);
    EXPECT_NEAR(energy(temperature), initial - 0.1 * 10.0 * 10.0, 1e-6);
    for (auto& [id, values] : flux.value)
        std::fill(values.begin(), values.end(), 20.0);
    equation.set_boundary_heat_flux(flux); // Same support, changed RHS only.
    conductivity.put_scalar(8.0);
    EXPECT_TRUE(equation
            .advance_physical(
                temperature, carrier, 0.2, mat, temperature, [](int) { return 0.0; },
                FVM::NonOrthogonalTreatment::Implicit, linear, &conductivity)
            .converged);
    EXPECT_NEAR(energy(temperature), initial - 50.0, 1e-6);
    EXPECT_ANY_THROW(equation.advance_semi_implicit(temperature, carrier, 0.1, 1.0, temperature));
    conductivity.put_scalar(0.0);
    EXPECT_ANY_THROW(equation.advance_physical(
        temperature, carrier, 0.1, mat, temperature, [](int) { return 0.0; }, FVM::NonOrthogonalTreatment::Implicit, {},
        &conductivity));
    const auto saved = energy(temperature);
    equation.set_boundary_heat_flux({{}, m});
    EXPECT_TRUE(equation
            .advance_physical(
                temperature, carrier, 0.1, mat, temperature, [](int) { return 0.0; },
                FVM::NonOrthogonalTreatment::Implicit)
            .converged);
    EXPECT_NEAR(energy(temperature), saved, 1e-6);
}
