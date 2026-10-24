/**
 * @file testReactingGasModel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief General finite-species, supplied-event and transport/rollback regression tests.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include <memory>

#include "FVM/FaceFlux.hh"
#include "GeneralGasModelTestSupport.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
namespace GeneralGas = SimpleFluid::Testing::GeneralGasModel;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;
using FaceFieldType = SimpleFluid::FaceField<Pack>;
using ReactingModelType = SimpleFluid::ReactingGasModel<Pack>;
using ALEMeshType = SimpleFluid::MeshHandle<Pack>;
using ALEFieldType = SimpleFluid::ScalarCellFieldStored<Pack, ALEMeshType>;
using ALEVelocityFieldType = SimpleFluid::VectorCellFieldStored<Pack, ALEMeshType>;
using ALEFaceFieldType = SimpleFluid::ScalarFaceFieldStored<Pack, ALEMeshType>;
using ALEReactingModelType = SimpleFluid::ReactingGasModel<Pack, ALEMeshType>;
using ALEMotionType = SimpleFluid::PlanarALEMeshMotion<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Build the one-cell mesh used by local reacting_gas checks.
 *
 * @return Assembled single-cell mesh.
 */
SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
}

/** @brief Build a mutable unit-area, one-cell Cartesian column. */
SimpleFluid::SP<ALEMeshType> make_ale_single_cell_column()
{
    auto geometry = std::make_shared<ALEMeshType::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<ALEMeshType>(std::move(geometry));
}

/** @brief Four cells with an explicit inter-region join and shared wall names. */
SimpleFluid::SP<SimpleFluid::Meshes::MultiRegionMesh> make_gas_composite_column()
{
    using Composite = SimpleFluid::Meshes::MultiRegionMesh;
    return std::make_shared<Composite>(
        std::vector<Composite::Region>{SimpleFluid::Meshes::cartesian_region("left", {{{0, .5}, {0, 1}, {0, .5, 1}}}),
            SimpleFluid::Meshes::cartesian_region("right", {{{.5, 1}, {0, 1}, {0, .5, 1}}})},
        std::vector<Composite::Interface>{SimpleFluid::Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}},
        SimpleFluid::Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames);
}

/** @brief Own swept-flux data so a property callback can invalidate only the
 * assembly-time GCL input, after advance() has completed its entry validation.
 */
class MutableGasALEView final : public SimpleFluid::MeshMotionModel
{
public:
    MutableGasALEView(SimpleFluid::SP<ALEMeshType> mesh, ALEMotionType& motion)
        : mesh_(std::move(mesh)), motion_(motion),
          flux_(motion.face_mesh_fluxes().begin(), motion.face_mesh_fluxes().end())
    {
    }
    void begin_trial(SimpleFluid::real_t top, SimpleFluid::real_t dt) override { motion_.begin_trial(top, dt); }
    void accept_trial() override { motion_.accept_trial(); }
    void rollback_trial() override { motion_.rollback_trial(); }
    bool has_active_trial() const noexcept override { return motion_.has_active_trial(); }
    std::string_view mesh_family() const noexcept override { return motion_.mesh_family(); }
    std::span<const SimpleFluid::real_t> old_cell_volumes() const noexcept override
    {
        return motion_.old_cell_volumes();
    }
    std::span<const SimpleFluid::real_t> new_cell_volumes() const noexcept override
    {
        return motion_.new_cell_volumes();
    }
    std::span<const SimpleFluid::real_t> face_mesh_fluxes() const noexcept override { return flux_; }
    const SimpleFluid::MeshMotionDiagnostics& diagnostics() const noexcept override { return motion_.diagnostics(); }
    const SimpleFluid::SP<ALEMeshType>& mesh_ptr() const noexcept { return mesh_; }
    void invalidate_flux() { flux_.front() += 1.0; }

private:
    SimpleFluid::SP<ALEMeshType> mesh_;
    ALEMotionType& motion_;
    std::vector<SimpleFluid::real_t> flux_;
};

thread_local MutableGasALEView* gas_gcl_fault_motion = nullptr;
thread_local bool gas_gcl_fault_armed = false;
thread_local bool gas_gcl_fault_injected = false;

double gas_diffusivity_with_gcl_fault(double)
{
    if (gas_gcl_fault_armed)
    {
        gas_gcl_fault_armed = false;
        gas_gcl_fault_injected = true;
        gas_gcl_fault_motion->invalidate_flux();
    }
    return 1e-8;
}

/**
 * @brief Create water-like material fields for the reacting_gas tests.
 *
 * @param mesh Mesh owning the material fields.
 * @return Initialized water-property fields.
 */
SimpleFluid::MaterialPropertyFields<Pack> make_water_properties(const SimpleFluid::SP<MeshType>& mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1000.0;
    options.density = 1000.0;
    options.specific_heat_capacity = 4200.0;
    options.dynamic_viscosity = 1.0e-3;
    options.thermal_conductivity = 0.6;
    return {mesh, options, time_options};
}

/** @brief Create water-like material fields on a mutable mesh handle. */
SimpleFluid::MaterialPropertyFields<Pack, ALEMeshType> make_ale_water_properties(
    const SimpleFluid::SP<ALEMeshType>& mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1000.0;
    options.density = 1000.0;
    options.specific_heat_capacity = 4200.0;
    options.dynamic_viscosity = 1.0e-3;
    options.thermal_conductivity = 0.6;
    return {mesh, options, time_options};
}

/** General species transport policy retains explicit finite source ownership. */
TEST(ReactingGasModelTest, TransportSolverPolicyDefaultsAndConfigureReset)
{
    auto mesh = make_single_cell_mesh();
    auto options = GeneralGas::options();
    options.transport_solver_tolerance = 3.0e-9;
    ReactingModelType model(mesh, options);
    auto linear = model.transport_linear_solver_options();
    EXPECT_EQ(linear.backend, SimpleFluid::LinearSolverBackend::Gmres);
    EXPECT_EQ(linear.preconditioner, SimpleFluid::LinearPreconditioner::None);
    EXPECT_EQ(linear.max_iterations, 200);
    EXPECT_DOUBLE_EQ(linear.tolerance, options.transport_solver_tolerance);

    linear.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    linear.preconditioner = SimpleFluid::LinearPreconditioner::GaussSeidel;
    linear.max_iterations = 123;
    linear.tolerance = 2.0e-8;
    model.set_transport_linear_solver_options(linear);
    EXPECT_EQ(model.transport_linear_solver_options().backend, linear.backend);
    EXPECT_EQ(model.transport_linear_solver_options().preconditioner, linear.preconditioner);
    EXPECT_EQ(model.transport_linear_solver_options().max_iterations, linear.max_iterations);
    EXPECT_DOUBLE_EQ(model.options().transport_solver_tolerance, linear.tolerance);

    const auto before_configuration = model.snapshot();
    model.configure(options);
    EXPECT_THROW(model.restore(before_configuration), std::invalid_argument);
    EXPECT_EQ(model.transport_linear_solver_options().backend, SimpleFluid::LinearSolverBackend::Gmres);
    EXPECT_EQ(model.transport_linear_solver_options().preconditioner, SimpleFluid::LinearPreconditioner::None);
    EXPECT_EQ(model.transport_linear_solver_options().max_iterations, 200);
    EXPECT_DOUBLE_EQ(model.transport_linear_solver_options().tolerance, options.transport_solver_tolerance);
}

/** @brief Invalid solver changes leave the accepted policy and tolerance intact. */
TEST(ReactingGasModelTest, TransportSolverPolicyRejectsInvalidOptions)
{
    ReactingModelType model(make_single_cell_mesh(), GeneralGas::options());
    const auto original = model.transport_linear_solver_options();
    const auto expect_rejected = [&](auto change)
    {
        auto candidate = original;
        change(candidate);
        EXPECT_THROW(model.set_transport_linear_solver_options(candidate), std::invalid_argument);
        EXPECT_EQ(model.transport_linear_solver_options().backend, original.backend);
        EXPECT_EQ(model.transport_linear_solver_options().preconditioner, original.preconditioner);
        EXPECT_EQ(model.transport_linear_solver_options().max_iterations, original.max_iterations);
        EXPECT_DOUBLE_EQ(model.transport_linear_solver_options().tolerance, original.tolerance);
        EXPECT_DOUBLE_EQ(model.options().transport_solver_tolerance, original.tolerance);
    };
    expect_rejected([](auto& value) { value.max_iterations = 0; });
    expect_rejected([](auto& value) { value.tolerance = 0.0; });
    expect_rejected([](auto& value) { value.tolerance = std::numeric_limits<double>::quiet_NaN(); });
    expect_rejected([](auto& value) { value.backend = static_cast<SimpleFluid::LinearSolverBackend>(-1); });
    expect_rejected([](auto& value) { value.preconditioner = static_cast<SimpleFluid::LinearPreconditioner>(-1); });
    expect_rejected([](auto& value) { value.backend = SimpleFluid::LinearSolverBackend::Cg; });
}

/** @brief GS-preconditioned transport matches the same full-schema update. */
TEST(ReactingGasModelTest, BiCGStabGaussSeidelTransportMatchesDefault)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    const auto options = GeneralGas::options(GeneralGas::populated());
    ReactingModelType reference(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_external");
    VelocityFieldType velocity(mesh, MeshType::Vec3{0.1, 0.0, 0.0}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    for (const auto face_lid : flux.owned_face_ids())
        flux.set_value(face_lid, 0.1 * mesh->face_area_vector(face_lid).x);
    auto material = make_water_properties(mesh);
    reference.initialize_state(0.0, temperature, pressure, velocity, material);
    reference.advance(1.0e-4, 1.0e-4, temperature, pressure, velocity, flux, material);
    ASSERT_EQ(reference.last_statistics().transport_linear.solves, 9);

    for (const auto preconditioner :
        {SimpleFluid::LinearPreconditioner::GaussSeidel, SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel})
    {
        ReactingModelType model(mesh, options);
        auto linear = model.transport_linear_solver_options();
        linear.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
        linear.preconditioner = preconditioner;
        model.set_transport_linear_solver_options(linear);
        model.initialize_state(0.0, temperature, pressure, velocity, material);
        model.advance(1.0e-4, 1.0e-4, temperature, pressure, velocity, flux, material);
        const auto summary = model.last_statistics().transport_linear;
        EXPECT_TRUE(summary.converged);
        EXPECT_EQ(summary.solves, 9);
        EXPECT_GT(summary.iterations, 0);
        EXPECT_LE(summary.achieved_tolerance, linear.tolerance);
        for (const auto& [name, expected_field] : reference.output_fields())
        {
            SCOPED_TRACE(name);
            const auto& actual_field = *model.output_fields().at(name);
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
                const auto expected = expected_field->value(cell_lid);
                EXPECT_NEAR(actual_field.value(cell_lid), expected, std::max(1.0e-12, std::abs(expected) * 1.0e-8));
            }
        }
        const auto accepted = model.snapshot();
        model.advance(2.0e-4, 1.0e-4, temperature, pressure, velocity, flux, material);
        model.restore(accepted);
        EXPECT_EQ(model.last_statistics().transport_linear.solves, summary.solves);
        EXPECT_EQ(model.last_statistics().transport_linear.iterations, summary.iterations);
        EXPECT_DOUBLE_EQ(model.last_statistics().transport_linear.achieved_tolerance, summary.achieved_tolerance);
    }
}

/** @brief Reused graphs/factors follow changing fluxes and timestep after rollback. */
TEST(ReactingGasModelTest, TransportWorkspaceMatchesFreshAssemblyAcrossStages)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    auto options = GeneralGas::options(GeneralGas::populated());
    ReactingModelType model(mesh, options);
    ReactingModelType fresh(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_external");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    auto linear = model.transport_linear_solver_options();
    linear.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    linear.preconditioner = SimpleFluid::LinearPreconditioner::SymmetricGaussSeidel;
    linear.reuse_preconditioner = true;
    model.set_transport_linear_solver_options(linear);
    fresh.set_transport_linear_solver_options(linear);
    model.initialize_state(0.0, temperature, pressure, velocity, material);
    fresh.initialize_state(0.0, temperature, pressure, velocity, material);
    double time = 0.0;
    for (int step = 0; step < 4; ++step)
    {
        SCOPED_TRACE(step);
        const auto accepted = model.snapshot();
        // Force numeric reassembly without replacing physical initialization or its receipts.
        GeneralGas::force_numeric_reassembly(fresh);
        fresh.set_transport_linear_solver_options(linear);
        const double speed = step % 2 == 0 ? 0.1 : -0.2;
        const double dt = (step + 1) * 1.0e-4;
        for (const auto face : flux.owned_face_ids())
            flux.set_value(face, speed * mesh->face_area_vector(face).x);
        const auto advance = [&](auto& candidate)
        { candidate.advance(time + dt, dt, temperature, pressure, velocity, flux, material); };
        advance(model);
        advance(fresh);
        if (step == 2)
        {
            model.restore(accepted);
            advance(model);
        }
        EXPECT_EQ(model.last_statistics().transport_linear.solves, 9);
        EXPECT_TRUE(model.last_statistics().transport_linear.converged);
        for (const auto& [name, expected] : fresh.output_fields())
        {
            SCOPED_TRACE(name);
            const auto& actual = *model.output_fields().at(name);
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                EXPECT_DOUBLE_EQ(actual.value(cell), expected->value(cell));
            }
        }
        EXPECT_DOUBLE_EQ(GeneralGas::step_escape(model), GeneralGas::step_escape(fresh));
        time += dt;
    }
}

/**
 * @brief A supplied gas feed computes EOS volume production without mutating an external void field.
 */
TEST(ReactingGasModelTest, IdealSourceRequiresAndDoesNotAdvanceAuthoritativeVoid)
{
    GeneralGas::Parameters p;
    p.source = [](const GeneralGas::R::ReactionContext& c, double dt)
    {
        GeneralGas::R::ReactionSourceInput input;
        const double n = 2e-7 * 4e6 * dt * c.environment.bulkVolume;
        input.externalTransfers = {
            {"test.explicit_resolved_gas_feed", GeneralGas::one, GeneralGas::Phase::Resolved, n, .002 * n, 0, {}, {}}};
        return input;
    };
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    FieldType external_alpha(f.grid, 0., "unowned_external_alpha");
    f.initialize();
    f.advance(.1, .1);
    const double expected = 2e-7 * 4e6 * GeneralGas::gas_constant * 300 / 1e5;
    EXPECT_NEAR(f.model.source_alpha_reaction().value(0), expected, 1e-14);
    EXPECT_NEAR(f.model.alpha_g().value(0), .1 * expected, 1e-14);
    EXPECT_DOUBLE_EQ(external_alpha.value(0), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).chemical_production, 0);
    EXPECT_NEAR(GeneralGas::history(f.model).external_supply, 2e-7 * 4e6 * .1, 1e-14);
    // The retired enabled/null-mechanism fallback must not recreate this physical source implicitly.
    SimpleFluid::ReactingGasOptions missing;
    missing.mode = SimpleFluid::ReactingGasMode::Species;
    EXPECT_THROW(ReactingModelType(f.grid, missing), std::invalid_argument);
}

/**
 * @brief Ideal-gas source respects configured rate and void-fraction limits.
 */
TEST(ReactingGasModelTest, IdealSourceHonorsAlphaAndRateLimits)
{
    const auto reference = GeneralGas::options();
    const auto gas = reference.gas_thermodynamics;
    const auto catalog = std::make_shared<const GeneralGas::R::SpeciesCatalog>(reference.reaction_mechanism->catalog());
    const auto capacity = [gas, catalog](const GeneralGas::R::ReactionEnvironment& e)
    {
        std::vector<double> x(catalog->species().size());
        x[catalog->index(GeneralGas::one)] = 1;
        const double rho = gas->evaluate(e.temperature, e.absolutePressure, x).density;
        const double limit = .2 * e.bulkVolume;
        double n = limit * rho / .002;
        while (n * .002 / rho > limit)
            n = std::nextafter(n, 0.);
        while (std::nextafter(n, std::numeric_limits<double>::infinity()) * .002 / rho <= limit)
            n = std::nextafter(n, std::numeric_limits<double>::infinity());
        return n;
    };
    GeneralGas::Parameters p;
    p.controls.alpha_max = .2;
    p.source = [capacity](const GeneralGas::R::ReactionContext& c, double dt)
    {
        const double requested = 2e-7 * 1e12 * dt * c.environment.bulkVolume;
        const double rate_limit = .01 * dt * c.environment.bulkVolume * c.environment.absolutePressure /
                                  (GeneralGas::gas_constant * c.environment.temperature);
        const double stock = c.state.amount(c.catalog, GeneralGas::one, GeneralGas::Phase::Resolved);
        const double limit = capacity(c.environment);
        GeneralGas::M::require(stock <= limit, "Source admission received an already overcapacity state.");
        const double accepted = std::min({requested, rate_limit, limit - stock});
        GeneralGas::R::ReactionSourceInput result;
        if (accepted > 0)
            result.externalTransfers = {{"test.supplied_rate_and_phase_admission", GeneralGas::one,
                GeneralGas::Phase::Resolved, accepted, .002 * accepted, 0, {}, {}}};
        return result;
    };
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    f.advance(.1, .1);
    EXPECT_DOUBLE_EQ(f.model.source_alpha_reaction().value(0), .01);
    EXPECT_NEAR(f.model.alpha_g().value(0), .001, 1e-14);
    const double accepted = GeneralGas::step_supply(f.model);
    EXPECT_LT(accepted, 2e-7 * 1e12 * .1);
    EXPECT_GT(accepted, 0);
    GeneralGas::R::ReactionEnvironment e{.1, 300, 100000, 1, 1};
    GeneralGas::R::CellSpeciesState full(*catalog);
    full.dispersedPopulations = {{"micro", std::vector<double>(catalog->species().size())},
        {"large", std::vector<double>(catalog->species().size())}};
    const double n = capacity(e);
    full.set_amount(*catalog, GeneralGas::one, GeneralGas::Phase::Resolved, n);
    full.set_amount(*catalog, GeneralGas::solvent, GeneralGas::Phase::Liquid, 800 / .018);
    f.model.initialize_species(
        std::vector<GeneralGas::R::CellSpeciesState>{full}, std::vector<std::vector<double>>{{0, 0}});
    f.model.initialize_state(.1, f.temperature, f.pressure, f.velocity, f.material);
    f.advance(.2, .1);
    EXPECT_DOUBLE_EQ(GeneralGas::step_supply(f.model), 0);
    EXPECT_DOUBLE_EQ(f.model.source_alpha_reaction().value(0), 0);
    EXPECT_DOUBLE_EQ(f.model.alpha_g().value(0), .2);
}

/**
 * @brief Initial catalogued inventories publish derived void without a fake source.
 */
TEST(ReactingGasModelTest, InitialPopulationsAreReconstructedBeforeFirstKinetics)
{
    GeneralGas::Parameters parameters;
    parameters.dissolved_one = 2.5;
    parameters.number_density[0] = 1e10;
    parameters.molar_one[0] = 1e-3;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), parameters);
    EXPECT_FALSE(f.model.initial_state_initialized());
    f.initialize();
    EXPECT_TRUE(f.model.initial_state_initialized());
    EXPECT_EQ(&f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).mesh(), f.grid.get());
    const double gas_volume = GeneralGas::volume(1e-3, 1e10, 300, 100000, .07);
    ASSERT_GT(gas_volume, 0);
    EXPECT_NEAR(f.model.alpha_g().value(0), gas_volume, 1e-14);
    EXPECT_NEAR(f.model.alpha_l().value(0), 1 - gas_volume, 1e-14);
    const auto& dissolved = f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved);
    EXPECT_NEAR(dissolved.value(0) / f.model.alpha_l().value(0), 2.5, 1e-14);
    EXPECT_NEAR(dissolved.value(0), 2.5 * (1 - gas_volume), 1e-14);
    EXPECT_FALSE(f.model.species_event_receipt().accepted);
    EXPECT_DOUBLE_EQ(f.model.source_alpha_reaction().value(0), 0);
    f.advance(1e-6, 1e-6);
    EXPECT_NEAR(f.model.alpha_g().value(0), gas_volume, 1e-14);
    EXPECT_NEAR(f.model.source_alpha_reaction().value(0), 0, 1e-14);
    EXPECT_DOUBLE_EQ(GeneralGas::step_supply(f.model), 0);
}

/** @brief Per-species inventory accessors keep dissolved and dispersed compartments distinct. */
TEST(ReactingGasModelTest, SubmergedInventoryAccessorsResolveBothBubblePopulations)
{
    GeneralGas::Parameters p;
    p.dissolved_one = 2.5;
    p.number_density = {2e8, 4e6};
    p.molar_one = {3e-6, 5e-6};
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    const auto dissolved = GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dissolved);
    const auto micro = GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro");
    const auto large = GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large");
    EXPECT_DOUBLE_EQ(f.model.global_species_moles(GeneralGas::one), dissolved + micro + large);
    EXPECT_NEAR(micro, 3e-6, 1e-15);
    EXPECT_NEAR(large, 5e-6, 1e-15);
    const auto micro_volume = GeneralGas::volume(micro, 2e8, 300, 100000, .07);
    const auto large_volume = GeneralGas::volume(large, 4e6, 300, 100000, .07);
    EXPECT_NEAR(f.model.global_submerged_bubble_volume(), micro_volume + large_volume, 1e-15);
    EXPECT_NEAR(dissolved, 2.5 * (1 - micro_volume - large_volume), 1e-14);
}

/** @brief Candidate EOS volume responds to pressure and stored temperature. */
TEST(ReactingGasModelTest, CandidateSubmergedBubbleVolumeRespondsToPressureAndTemperature)
{
    GeneralGas::Parameters p;
    p.number_density = {2e8, 4e6};
    p.molar_one = {3e-6, 5e-6};
    const auto initialized_volume = [&](double temperature)
    {
        auto selected = p;
        selected.temperature = temperature;
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), selected);
        EXPECT_THROW(f.model.evaluate_submerged_bubble_volume(p.pressure), std::logic_error);
        f.initialize();
        const auto actual = f.model.evaluate_submerged_bubble_volume(p.pressure);
        const auto expected = GeneralGas::volume(3e-6, 2e8, temperature, p.pressure, p.sigma) +
                              GeneralGas::volume(5e-6, 4e6, temperature, p.pressure, p.sigma);
        EXPECT_NEAR(actual, expected, expected * 1e-11);
        EXPECT_NEAR(actual, f.model.global_submerged_bubble_volume(), actual * 1e-11);
        return std::pair{actual, f.model.evaluate_submerged_bubble_volume(2 * p.pressure)};
    };
    const auto [cold, compressed_cold] = initialized_volume(300);
    const auto [hot, compressed_hot] = initialized_volume(360);
    EXPECT_GT(cold, compressed_cold);
    EXPECT_GT(hot, compressed_hot);
    EXPECT_GT(hot, cold);
}

/** @brief Dissolved H2 is excluded from every submerged bubble-volume API. */
TEST(ReactingGasModelTest, DissolvedHydrogenDoesNotOccupyBubbleVolume)
{
    GeneralGas::Parameters p;
    p.dissolved_one = 100;
    p.dissolved_two = 50;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    EXPECT_GT(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dissolved), 0);
    EXPECT_GT(GeneralGas::compartment(f.model, GeneralGas::two, GeneralGas::Phase::Dissolved), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro"), 0);
    EXPECT_DOUBLE_EQ(f.model.global_submerged_bubble_volume(), 0);
    EXPECT_DOUBLE_EQ(f.model.evaluate_submerged_bubble_volume(p.pressure), 0);
}

/**
 * @brief Named external reservoirs conserve both catalogued gas inventories.
 */
TEST(ReactingGasModelTest, TwoPopulationStepConservesHydrogen)
{
    GeneralGas::Parameters p;
    p.reservoir_mol_per_joule = {2e-7, 3e-7};
    p.nucleate_supplied_gas = true;
    p.independent_heat_rate = 7;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.power.put_scalar(1e5);
    f.initialize();
    EXPECT_EQ(&f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).mesh(), f.grid.get());
    const auto initial = f.model.snapshot();
    constexpr double dt = 1e-6;
    f.advance(dt, dt);
    const auto first = f.model.species_histories();
    for (const auto [species, coefficient] :
        std::array<std::pair<GeneralGas::R::SpeciesId, double>, 2>{{{GeneralGas::one, 2e-7}, {GeneralGas::two, 3e-7}}})
    {
        const double expected = coefficient * 1e5 * dt;
        const auto& history = GeneralGas::history(f.model, species);
        EXPECT_NEAR(history.external_supply, expected, 1e-15);
        EXPECT_DOUBLE_EQ(history.chemical_production, 0);
        EXPECT_NEAR(history.current_moles, expected, 1e-15);
        EXPECT_NEAR(history.inventory_error, 0, 1e-14);
        EXPECT_NEAR(GeneralGas::step_supply(f.model, species), expected, 1e-15);
        EXPECT_GT(GeneralGas::compartment(f.model, species, GeneralGas::Phase::Dispersed, "micro"), 0);
    }
    EXPECT_NEAR(GeneralGas::integral(f.model.thermal_source()), 7, 1e-14);
    const auto accepted = f.model.species_event_receipt();
    EXPECT_EQ(accepted.revision, 1u);
    f.advance(2 * dt, dt);
    EXPECT_EQ(f.model.species_event_receipt().revision, 2u);
    EXPECT_NEAR(GeneralGas::history(f.model).external_supply, 2 * 2e-7 * 1e5 * dt, 1e-15);
    f.model.restore(initial);
    EXPECT_FALSE(f.model.species_event_receipt().accepted);
    EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).external_supply, 0);
    f.advance(dt, dt);
    EXPECT_EQ(f.model.species_histories(), first);
    EXPECT_EQ(f.model.species_event_receipt().revision, accepted.revision);
    EXPECT_GT(f.model.species_event_receipt().publication_nonce, accepted.publication_nonce);
}

/** Both diffusivity providers route high-Sc transfer and reconstructed K_L through the selected law. */
TEST(ReactingGasModelTest, HighScSphericalDiffusionUsesSelectedLawAndConservesHydrogen)
{
    std::array<double, 2> transfers{};
    for (std::size_t external = 0; external < 2; ++external)
    {
        GeneralGas::Parameters p;
        p.temperature = 298.85;
        p.density = 1546.23563;
        p.viscosity = .00243878721;
        p.diffusivity[0] = 4.5e-9;
        p.dissolved_one = 10;
        p.number_density[1] = 1e6;
        p.molar_one[1] = 1e-7;
        p.controls.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::SphericalDiffusion;
        if (external)
            p.diffusivity_one = [](const auto&) { return 5.1886e-9; };
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        const double before = f.model.global_species_moles(GeneralGas::one);
        constexpr double dt = 1e-6;
        const auto observed = GeneralGas::advance_film(f, dt, dt).front();
        transfers[external] = observed.transfer;
        ASSERT_GT(observed.transfer, 0);
        const double D = external ? 5.1886e-9 : 4.5e-9;
        // D/r film flux independently determines r when the supplied Henry/Laplace law is included.
        const double inferred =
            (observed.transfer / (dt * 4 * std::numbers::pi * D * 1e6) + 2 * 1e-5 * p.sigma) / (10 - 1e-5 * p.pressure);
        const double eos_volume =
            1e-7 * GeneralGas::gas_constant * p.temperature / (p.pressure + 2 * p.sigma / inferred);
        const double sphere = 1e6 * 4 * std::numbers::pi * inferred * inferred * inferred / 3;
        EXPECT_LE(std::abs(sphere - eos_volume), (4e-12 + 128 * std::numeric_limits<double>::epsilon()) * eos_volume);
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), before, 1e-12);
        EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-12);
    }
    EXPECT_NEAR(transfers[1] / transfers[0], 5.1886e-9 / 4.5e-9, 128 * std::numeric_limits<double>::epsilon());
}

/** Finite-slip Feng endpoint selection reaches population kinetics and K_L at high Sc. */
TEST(ReactingGasModelTest, HighScFengEndpointsTransferAndConserveAtPrescribedSlip)
{
    std::array<double, 2> coefficients{};
    const std::array cases{std::pair{SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid, .322},
        std::pair{SimpleFluid::BubbleMassTransferMode::FengMichaelidesClean, .435}};
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        const auto [mode, exponent] = cases[index];
        GeneralGas::Parameters p;
        p.temperature = 298.85;
        p.density = 1546.23563;
        p.viscosity = .00243878721;
        p.diffusivity[0] = 4.5e-9;
        p.dissolved_one = 10;
        p.number_density[1] = 1e6;
        p.molar_one[1] = .0002947014976484349;
        p.controls.mass_transfer_mode = mode;
        p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
        p.controls.constant_slip_velocity = .0045;
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        const auto start = f.model.snapshot();
        const double before = f.model.global_species_moles(GeneralGas::one);
        constexpr double dt = 1e-6;
        const auto observed = GeneralGas::advance_film(f, dt, dt).front();
        const double pe = .0045 * observed.radius / 4.5e-9, re = 2 * p.density * .0045 * observed.radius / p.viscosity;
        ASSERT_GE(pe, 100);
        ASSERT_LE(pe, 5000);
        ASSERT_LT(re, 1);
        coefficients[index] = observed.coefficient;
        EXPECT_NEAR(observed.coefficient, 1.49 * std::pow(pe, exponent) * 4.5e-9 / (2 * observed.radius), 1e-12);
        EXPECT_GT(observed.transfer, 0);
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), before, 1e-10);
        EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-10);
        if (index == 0)
        {
            const double full = observed.gas_after;
            f.model.restore(start);
            GeneralGas::advance_film(f, dt / 2, dt / 2);
            GeneralGas::advance_film(f, dt, dt / 2);
            EXPECT_NEAR(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), full,
                std::max(1e-13, .05 * observed.transfer));
            EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), before, 1e-10);
        }
    }
    EXPECT_GT(coefficients[1], coefficients[0]);
}

/** The active one-cell path also uses Feng's selected low-Pe Eq. (4) branch. */
TEST(ReactingGasModelTest, HighScFengLowPecletEndpointsTransferAndConserve)
{
    std::array<double, 2> coefficients{};
    const std::array cases{std::pair{SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid, 1. / 8},
        std::pair{SimpleFluid::BubbleMassTransferMode::FengMichaelidesClean, 1. / 12}};
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        const auto [mode, logarithmic] = cases[index];
        GeneralGas::Parameters p;
        p.temperature = 298.85;
        p.density = 1546.23563;
        p.viscosity = .00243878721;
        p.diffusivity[0] = 4.5e-9;
        p.dissolved_one = 10;
        p.number_density[1] = 1e6;
        p.molar_one[1] = .0002947014976484349;
        p.controls.mass_transfer_mode = mode;
        p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
        p.controls.constant_slip_velocity = 9.375e-6;
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        const double before = f.model.global_species_moles(GeneralGas::one);
        const auto observed = GeneralGas::advance_film(f, 1e-6, 1e-6).front();
        const double pe = 2 * 9.375e-6 * observed.radius / 4.5e-9;
        ASSERT_GT(pe, 0);
        ASSERT_LT(pe, 1);
        EXPECT_NEAR(pe, .5, 1e-8);
        coefficients[index] = observed.coefficient;
        EXPECT_NEAR(observed.coefficient,
            (2 + .5 * pe + logarithmic * pe * pe * std::log(pe)) * 4.5e-9 / (2 * observed.radius), 1e-12);
        EXPECT_GT(observed.transfer, 0);
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), before, 1e-10);
        EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-10);
    }
    EXPECT_GT(coefficients[1], coefficients[0]);
}

/** One selected law must retain the local Pe regime in every active field cell. */
TEST(ReactingGasModelTest, FengFieldSelectionKeepsBothPecletRegimesCellLocal)
{
    GeneralGas::Parameters p;
    p.temperature = 298.85;
    p.density = 1546.23563;
    p.viscosity = .00243878721;
    p.diffusivity_one = [](const auto& c) { return c.environment.temperature < 299 ? 4.5e-9 : 2e-6; };
    p.dissolved_one = 10;
    p.number_density[1] = 1e6;
    p.molar_one[1] = .0002947014976484349;
    p.controls.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = .0045;
    GeneralGas::Fixture<MeshType> f(SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database()), p);
    f.temperature.set_owned_value(1, 300);
    f.temperature.sync_ghosts();
    f.initialize();
    const double before = f.model.global_species_moles(GeneralGas::one);
    const auto values = GeneralGas::advance_film(f, 1e-6, 1e-6);
    ASSERT_EQ(values.size(), 2u);
    for (std::size_t row = 0; row < values.size(); ++row)
    {
        const double D = row == 0 ? 4.5e-9 : 2e-6, r = values[row].radius, pe = 2 * .0045 * r / D;
        EXPECT_LT(2 * p.density * .0045 * r / p.viscosity, 1);
        double expected = 0;
        if (row == 0)
        {
            EXPECT_GE(pe / 2, 100);
            EXPECT_LE(pe / 2, 5000);
            expected = 1.49 * std::pow(pe / 2, .322) * D / (2 * r);
        }
        else
        {
            EXPECT_GT(pe, 0);
            EXPECT_LT(pe, 1);
            expected = (2 + .5 * pe + .125 * pe * pe * std::log(pe)) * D / (2 * r);
        }
        EXPECT_NEAR(values[row].coefficient, expected, expected * 1e-11);
        EXPECT_GT(values[row].transfer, 0);
    }
    EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), before, 1e-10);
    EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-10);
}

TEST(ReactingGasModelTest, SnapshotRestoresEveryInventoryAndCumulativeLedger)
{
    auto p = GeneralGas::populated();
    p.reservoir_mol_per_joule = {2e-7, 3e-7};
    p.nucleate_supplied_gas = true;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.power.put_scalar(1e5);
    f.initialize();
    const auto checkpoint = f.model.snapshot();
    const auto histories = f.model.species_histories();
    const auto receipt = f.model.species_event_receipt();
    std::map<std::string, std::vector<double>> fields;
    for (const auto& [name, field] : f.model.output_fields())
        for (std::size_t row = 0; row < f.grid->num_local_cells(); ++row)
            fields[name].push_back(field->local_value(static_cast<Pack::local_ordinal_type>(row)));
    f.advance(1e-6, 1e-6);
    EXPECT_GT(GeneralGas::history(f.model).external_supply, 0);
    f.model.restore(checkpoint);
    EXPECT_EQ(f.model.species_histories(), histories);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    for (const auto& [name, values] : fields)
        for (std::size_t row = 0; row < values.size(); ++row)
            EXPECT_DOUBLE_EQ(
                f.model.output_fields().at(name)->local_value(static_cast<Pack::local_ordinal_type>(row)), values[row]);
    EXPECT_TRUE(f.model.initial_state_initialized());
}

/** @brief Sheng transport runs directly on MeshHandle/FieldStored storage. */
TEST(ReactingGasModelTest, NativeTwoPopulationStepConservesHydrogen)
{
    GeneralGas::Parameters p;
    p.reservoir_mol_per_joule = {2e-7, 3e-7};
    p.nucleate_supplied_gas = true;
    p.independent_heat_rate = 7;
    GeneralGas::Fixture<ALEMeshType> f(make_ale_single_cell_column(), p);
    f.power.put_scalar(1e5);
    f.initialize();
    EXPECT_EQ(&f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).mesh(), f.grid.get());
    const auto initial = f.model.snapshot();
    constexpr double dt = 1e-6;
    f.advance(dt, dt);
    const auto first = f.model.species_histories();
    for (const auto [species, coefficient] :
        std::array<std::pair<GeneralGas::R::SpeciesId, double>, 2>{{{GeneralGas::one, 2e-7}, {GeneralGas::two, 3e-7}}})
    {
        const double expected = coefficient * 1e5 * dt;
        const auto& history = GeneralGas::history(f.model, species);
        EXPECT_NEAR(history.external_supply, expected, 1e-15);
        EXPECT_DOUBLE_EQ(history.chemical_production, 0);
        EXPECT_NEAR(history.current_moles, expected, 1e-15);
        EXPECT_NEAR(history.inventory_error, 0, 1e-14);
        EXPECT_NEAR(GeneralGas::step_supply(f.model, species), expected, 1e-15);
        EXPECT_GT(GeneralGas::compartment(f.model, species, GeneralGas::Phase::Dispersed, "micro"), 0);
    }
    EXPECT_NEAR(GeneralGas::integral(f.model.thermal_source()), 7, 1e-14);
    const auto accepted = f.model.species_event_receipt();
    EXPECT_EQ(accepted.revision, 1u);
    f.advance(2 * dt, dt);
    EXPECT_EQ(f.model.species_event_receipt().revision, 2u);
    EXPECT_NEAR(GeneralGas::history(f.model).external_supply, 2 * 2e-7 * 1e5 * dt, 1e-15);
    f.model.restore(initial);
    EXPECT_FALSE(f.model.species_event_receipt().accepted);
    EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).external_supply, 0);
    f.advance(dt, dt);
    EXPECT_EQ(f.model.species_histories(), first);
    EXPECT_EQ(f.model.species_event_receipt().revision, accepted.revision);
    EXPECT_GT(f.model.species_event_receipt().publication_nonce, accepted.publication_nonce);
}

/**
 * @brief Microbubble decay follows the analytic lifetime solution.
 */
TEST(ReactingGasModelTest, MicrobubbleDecayIsAnalyticAndConservative)
{
    GeneralGas::Parameters p;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-6;
    p.micro_decay_time = 1e-5;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    constexpr double dt = 5e-6;
    f.advance(dt, dt);
    const double retention = std::exp(-dt / *p.micro_decay_time);
    EXPECT_NEAR(f.model.population_number_field("micro").value(0), 1e10 * retention, 1e10 * 1e-12);
    EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "micro").value(0),
        1e-6 * retention, 1e-16);
    EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).value(0) +
                    f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "micro").value(0),
        1e-6, 1e-15);
    EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).external_supply, 0);
    EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-15);
}

/** Supplied kinetic metadata validates before any numerical event. */
TEST(ReactingGasModelTest, RejectsUnknownKineticsMode)
{
    GeneralGas::Parameters p;
    SimpleFluid::Database retired;
    retired.set("reacting_kinetics_mode", std::string("exactInactive"));
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(retired), std::invalid_argument);
    p.micro_decay_time = 0;
    EXPECT_THROW((void) GeneralGas::options(p), std::invalid_argument);
}

/** @brief Continuous production/decay is exact across small and very large lifetime ratios. */
TEST(ReactingGasModelTest, ExactInactiveMatchesContinuousProductionAndDecay)
{
    for (const double lifetime : {1e-5, 5e9, 1e-300, std::numeric_limits<double>::denorm_min()})
    {
        SCOPED_TRACE(lifetime);
        GeneralGas::Parameters p;
        p.micro_decay_time = lifetime;
        p.number_density[0] = 2e8;
        p.molar_one[0] = 3e-6;
        p.reservoir_mol_per_joule[0] = 2e-7;
        p.nucleate_supplied_gas = true;
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.power.put_scalar(1e5);
        f.initialize();
        const auto checkpoint = f.model.snapshot();
        constexpr double dt = .005;
        if (lifetime == std::numeric_limits<double>::denorm_min())
        {
            // Positive gas survival is below double representation: no count-only ghost phase.
            const auto histories = f.model.species_histories();
            EXPECT_THROW(f.advance(dt, dt), std::exception);
            EXPECT_EQ(f.model.species_histories(), histories);
            f.model.restore(checkpoint);
            EXPECT_DOUBLE_EQ(f.model.population_number_field("micro").value(0), 2e8);
            continue;
        }
        f.advance(dt, dt);
        const long double source = 2e-7L * 1e5L, x = static_cast<long double>(dt) / lifetime;
        const long double retained = source * lifetime * (-std::expm1(-x));
        const long double gas = 3e-6L * std::exp(-x) + retained;
        const long double r = p.nucleation_radius;
        const long double per_bubble = 4 * std::numbers::pi_v<long double> / 3 *
                                       (p.pressure * r * r * r + 2 * p.sigma * r * r) /
                                       (GeneralGas::gas_constant * 300.L);
        const long double count = 2e8L * std::exp(-x) + retained / per_bubble;
        const long double source_decay =
            x < 1e-3L ? source * dt * (x / 2 - x * x / 6 + x * x * x / 24) : source * dt - retained;
        const long double dissolved = 3e-6L * (-std::expm1(-x)) + source_decay;
        EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "micro").value(0), gas,
            std::max(1e-315, static_cast<double>(gas) * 2e-12));
        EXPECT_NEAR(f.model.population_number_field("micro").value(0), count,
            std::max(1e-290, static_cast<double>(count) * 2e-12));
        EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).value(0), dissolved,
            std::max(1e-30, static_cast<double>(dissolved) * 2e-12));
        EXPECT_NEAR(GeneralGas::history(f.model).external_supply, 2e-7 * 1e5 * dt, 1e-15);
    }
}

/** @brief Zero and negative prescribed_heat power retain only conservative initial-population decay. */
TEST(ReactingGasModelTest, ExactInactiveDecaysInitialPopulationWithoutPositivePower)
{
    for (const double power : {0., -1e5})
    {
        GeneralGas::Parameters p;
        p.micro_decay_time = .01;
        p.number_density[0] = 2e8;
        p.molar_one[0] = 3e-6;
        p.reservoir_mol_per_joule[0] = 2e-7;
        p.nucleate_supplied_gas = true;
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.power.put_scalar(power);
        f.initialize();
        f.advance(.005, .005);
        const auto retention = std::exp(-.5);
        EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "micro").value(0),
            3e-6 * retention, 3e-6 * 2e-12);
        EXPECT_NEAR(f.model.population_number_field("micro").value(0), 2e8 * retention, 2e8 * 2e-12);
        EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).external_supply, 0);
    }
}

/** @brief Crossing, saturated and populated-large states preserve the legacy local update exactly. */
TEST(ReactingGasModelTest, ExactInactiveFallsBackAtConversionOrExistingLargePopulation)
{
    // Specialized dispatch is retired. The same supplied conservative operator
    // must handle active conversion and existing large bins without losing categories.
    for (const bool conversion : {false, true})
    {
        auto p = GeneralGas::populated();
        p.micro_decay_time = .01;
        p.micro_conversion_frequency = conversion ? 2 : 0;
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        const auto before_one = f.model.global_species_moles(GeneralGas::one);
        const auto before_two = f.model.global_species_moles(GeneralGas::two);
        f.advance(.001, .001);
        // Conversion and dissolution are internal transfers; prescribed slip
        // still removes the exact material recorded by accepted escape events.
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one) + GeneralGas::step_escape(f.model, GeneralGas::one),
            before_one, 1e-14);
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::two) + GeneralGas::step_escape(f.model, GeneralGas::two),
            before_two, 1e-14);
        EXPECT_GT(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), 0);
        EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-14);
    }
}

/**
 * @brief Micro-to-large conversion transfers number and gas inventory.
 */
TEST(ReactingGasModelTest, MicroToLargeConversionConservesCategories)
{
    GeneralGas::Parameters p;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-6;
    p.molar_two[0] = 2e-6;
    p.micro_conversion_frequency = 10;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    constexpr double dt = 1e-6;
    f.advance(dt, dt);
    const double converted = -std::expm1(-10 * dt);
    for (const auto [id, initial] :
        std::array<std::pair<GeneralGas::R::SpeciesId, double>, 2>{{{GeneralGas::one, 1e-6}, {GeneralGas::two, 2e-6}}})
    {
        EXPECT_NEAR(
            GeneralGas::compartment(f.model, id, GeneralGas::Phase::Dispersed, "large"), initial * converted, 1e-16);
        EXPECT_NEAR(f.model.global_species_moles(id), initial, 1e-15);
        EXPECT_NEAR(GeneralGas::history(f.model, id).inventory_error, 0, 1e-15);
    }
    EXPECT_NEAR(GeneralGas::integral(f.model.population_number_field("large")), 1e10 * converted, 1e10 * 1e-12);
    EXPECT_NEAR(GeneralGas::integral(f.model.population_number_field("micro")) +
                    GeneralGas::integral(f.model.population_number_field("large")),
        1e10, 1e10 * 1e-12);
}

/**
 * @brief Large bubbles grow or dissolve according to local saturation.
 */
TEST(ReactingGasModelTest, LargeBubbleGrowthAndDissolutionFollowSaturation)
{
    GeneralGas::Parameters p;
    p.dissolved_one = 1e3;
    p.number_density[1] = 1e6;
    p.molar_one[1] = 1e-7;
    p.diffusivity[0] = 1e-8;
    p.controls.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::Hughmark;
    GeneralGas::Fixture<MeshType> saturated(make_single_cell_mesh(), p);
    saturated.initialize();
    const auto growth = GeneralGas::advance_film(saturated, 1e-6, 1e-6).front();
    EXPECT_GT(growth.transfer, 0);
    EXPECT_GT(growth.gas_after, 1e-7);
    EXPECT_NEAR(growth.count_after, 1e6, 1e6 * 1e-12);
    p.dissolved_one = 0;
    p.controls.large_dissolution_frequency = 1e5;
    GeneralGas::Fixture<MeshType> undersaturated(make_single_cell_mesh(), p);
    undersaturated.initialize();
    undersaturated.advance(1e-6, 1e-6);
    EXPECT_LT(
        GeneralGas::compartment(undersaturated.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), 1e-7);
    EXPECT_LT(GeneralGas::integral(undersaturated.model.population_number_field("large")), 1e6);
    EXPECT_NEAR(GeneralGas::history(undersaturated.model).inventory_error, 0, 1e-14);
}

/**
 * @brief Prescribed pressure mode interpolates the supplied pressure history.
 */
TEST(ReactingGasModelTest, PrescribedPressureHistoryInterpolates)
{
    GeneralGas::Parameters p;
    p.controls.pressure_mode = SimpleFluid::ReactingPressureMode::PrescribedHistory;
    p.controls.pressure_history_times = {0, 1};
    p.controls.pressure_history_values = {1e5, 2e5};
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.pressure.put_scalar(8e4);
    f.initialize();
    f.advance(.5, .1);
    EXPECT_DOUBLE_EQ(f.model.absolute_pressure().value(0), 1.5e5);
    EXPECT_THROW(f.model.set_absolute_pressure_offset(2e5), std::logic_error);
}

/**
 * @brief Reconstructed pressure uses the solver's gauge variation in Pa.
 */
TEST(ReactingGasModelTest, ReconstructedPressureUsesPascalGaugeVariation)
{
    GeneralGas::Parameters p;
    p.controls.pressure_mode = SimpleFluid::ReactingPressureMode::Reconstructed;
    p.density = 800;
    GeneralGas::Fixture<MeshType> f(SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database()), p);
    ASSERT_EQ(f.grid->num_owned_cells(), 2u);
    f.pressure.set_owned_value(0, -1000);
    f.pressure.set_owned_value(1, 1000);
    f.pressure.sync_ghosts();
    f.initialize();
    f.advance(.1, .1);
    EXPECT_DOUBLE_EQ(f.material.density.value(0), 800);
    EXPECT_NEAR(f.model.absolute_pressure().value(0), 9.9e4, 1e-10);
    EXPECT_NEAR(f.model.absolute_pressure().value(1), 1.01e5, 1e-10);
}

/**
 * @brief The pressure-offset owner accounts for the gauge mean at the top.
 *
 * Reconstructed pressure removes the volume-weighted gauge mean.  Therefore
 * a zero-gauge free surface has the prescribed headspace pressure only when
 * the externally owned offset is `p_headspace + mean(p_gauge)`.
 */
TEST(ReactingGasModelTest, ReconstructedOffsetIncludingGaugeMeanMatchesZeroGaugeTopPressure)
{
    auto geometry = std::make_shared<ALEMeshType::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.25, 1.0}}});
    auto mesh = std::make_shared<ALEMeshType>(std::move(geometry));
    ASSERT_EQ(mesh->num_owned_cells(), 2U);

    auto options = GeneralGas::options();
    options.pressure_mode = SimpleFluid::ReactingPressureMode::Reconstructed;
    ALEReactingModelType model(mesh, options);
    ALEFieldType temperature(mesh, 300.0, "temperature");
    ALEFieldType gauge_pressure(mesh, 0.0, "pressure");
    ALEFieldType power(mesh, 0.0, "qdot_external");
    ALEFieldType authoritative_alpha(mesh, 0.0, "authoritative_alpha_g");
    ALEVelocityFieldType velocity(mesh, ALEMeshType::Vec3{}, "velocity");
    ALEFaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_ale_water_properties(mesh);

    Pack::local_ordinal_type top_cell = 0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        if (mesh->cell_centroid(cell_lid).z > mesh->cell_centroid(top_cell).z)
        {
            top_cell = cell_lid;
        }
    }
    const auto bottom_cell =
        top_cell == Pack::local_ordinal_type{0} ? Pack::local_ordinal_type{1} : Pack::local_ordinal_type{0};
    gauge_pressure.set_owned_value(bottom_cell, 4000.0);
    gauge_pressure.set_owned_value(top_cell, 0.0);
    gauge_pressure.sync_ghosts();

    double pressure_volume = 0.0;
    double volume = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto cell_volume = mesh->cell_volume(cell_lid);
        pressure_volume += gauge_pressure.value(cell_lid) * cell_volume;
        volume += cell_volume;
    }
    const auto gauge_mean = pressure_volume / volume;
    ASSERT_DOUBLE_EQ(gauge_mean, 1000.0);

    constexpr double headspace_pressure = 125000.0;
    model.set_absolute_pressure_offset(headspace_pressure + gauge_mean);
    model.initialize_state(0.0, temperature, gauge_pressure, velocity, material);
    model.advance(0.1, 0.1, temperature, gauge_pressure, velocity, flux, material);

    EXPECT_NEAR(model.absolute_pressure().value(top_cell), headspace_pressure, 1.0e-10);
    EXPECT_NEAR(model.absolute_pressure().value(bottom_cell), headspace_pressure + 4000.0, 1.0e-10);
}

/** @brief Coupled pressure offsets preserve reconstructed gauge variations. */
TEST(ReactingGasModelTest, AbsolutePressureOffsetSetterShiftsPublishedPressure)
{
    GeneralGas::Parameters p;
    p.controls.pressure_mode = SimpleFluid::ReactingPressureMode::Reconstructed;
    GeneralGas::Fixture<MeshType> f(SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database()), p);
    f.pressure.set_owned_value(0, -1000);
    f.pressure.set_owned_value(1, 1000);
    f.pressure.sync_ghosts();
    f.initialize();
    f.advance(.1, .1);
    EXPECT_DOUBLE_EQ(f.model.absolute_pressure_offset(), 1e5);
    const double variation = f.model.absolute_pressure().value(1) - f.model.absolute_pressure().value(0);
    const auto receipt = f.model.species_event_receipt();
    const auto histories = f.model.species_histories();
    f.model.set_absolute_pressure_offset(2e5);
    EXPECT_DOUBLE_EQ(f.model.absolute_pressure_offset(), 2e5);
    EXPECT_NEAR(f.model.absolute_pressure().value(0), 1.99e5, 1e-10);
    EXPECT_NEAR(f.model.absolute_pressure().value(1), 2.01e5, 1e-10);
    EXPECT_DOUBLE_EQ(f.model.absolute_pressure().value(1) - f.model.absolute_pressure().value(0), variation);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    EXPECT_EQ(f.model.species_histories(), histories);
    f.advance(.2, .1);
    EXPECT_NEAR(f.model.absolute_pressure().value(0), 1.99e5, 1e-10);
    EXPECT_NEAR(f.model.absolute_pressure().value(1), 2.01e5, 1e-10);
    EXPECT_THROW(f.model.set_absolute_pressure_offset(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(f.model.set_absolute_pressure_offset(.5), std::invalid_argument);
    auto prescribed = p;
    prescribed.controls.pressure_mode = SimpleFluid::ReactingPressureMode::PrescribedHistory;
    prescribed.controls.pressure_history_times = {0, 1};
    prescribed.controls.pressure_history_values = {1e5, 1e5};
    ReactingModelType external(f.grid, GeneralGas::options(prescribed));
    EXPECT_THROW(external.set_absolute_pressure_offset(2e5), std::logic_error);
    EXPECT_THROW(external.minimum_valid_absolute_pressure_offset(), std::logic_error);
}

/** @brief Accepted offsets rebuild every pressure-dependent bubble field. */
TEST(ReactingGasModelTest, ReconstructedOffsetRefreshesTwoPopulationDiagnostics)
{
    GeneralGas::Parameters p;
    p.temperature = 330;
    p.number_density = {2e8, 4e6};
    p.molar_one = {3e-6, 5e-6};
    p.controls.pressure_mode = SimpleFluid::ReactingPressureMode::Reconstructed;
    p.controls.minimum_absolute_pressure = 5e4;
    p.controls.alpha_max = 5e-5;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = .1;
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database()), p);
    f.pressure.set_owned_value(0, -1000);
    f.pressure.set_owned_value(1, 1000);
    f.pressure.sync_ghosts();
    f.initialize();
    f.advance(.01, .01);
    EXPECT_NEAR(f.model.minimum_valid_absolute_pressure_offset(), 51000, 1e-10);
    EXPECT_THROW(f.model.evaluate_submerged_bubble_volume(50500), std::invalid_argument);
    const auto histories = f.model.species_histories();
    const auto receipt = f.model.species_event_receipt();
    const auto old_volume = f.model.global_submerged_bubble_volume();
    EXPECT_GT(GeneralGas::history(f.model).physical_outflow, 0);
    const auto proposed = f.model.evaluate_submerged_bubble_volume(2e5);
    f.model.set_absolute_pressure_offset(2e5);
    EXPECT_EQ(f.model.species_histories(), histories);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    EXPECT_NEAR(f.model.global_submerged_bubble_volume(), proposed, proposed * 1e-11);
    EXPECT_LT(proposed, old_volume);
    double expected_volume = 0;
    for (std::size_t row = 0; row < f.grid->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const double cv = f.grid->cell_volume(cell);
        for (const auto& bin : f.model.options().population_ids)
        {
            const auto count = f.model.population_number_field(bin).value(cell) * cv;
            const auto n = f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, bin).value(cell) * cv;
            expected_volume += GeneralGas::volume(n, count, 330, f.model.absolute_pressure().value(cell), p.sigma);
        }
        // Reference CV is not a phase-occupancy closure: actual liquid mass is retained on pressure refresh.
        EXPECT_NEAR(f.model.alpha_l().value(cell), f.model.liquid_mass_density().value(cell) / p.density, 1e-13);
    }
    EXPECT_NEAR(proposed, expected_volume, std::abs(expected_volume) * 1e-13);
}

/**
 * @brief ALE-relative top escape is exact and rollback-safe for either motion.
 */
TEST(ReactingGasModelTest, MovingTopRelativeEscapeIsExactForStationaryExpansionAndContraction)
{
    constexpr double dt = .25;
    for (const double target : {1., 1.2, .8})
    {
        SCOPED_TRACE(target);
        auto p = GeneralGas::populated();
        GeneralGas::Fixture<ALEMeshType> f(make_ale_single_cell_column(), p);
        f.initialize();
        const auto checkpoint = f.model.snapshot();
        const auto original_histories = f.model.species_histories();
        const double initial_volume = f.model.global_submerged_bubble_volume();
        std::array<double, 2> initial_moles{
            GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro"),
            GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large")};
        const double initial_dissolved =
            GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dissolved);
        ALEMotionType motion(f.grid);
        Pack::local_ordinal_type top = -1;
        for (const auto& [id, batch] : f.grid->boundary_batches())
            if (f.grid->boundary_batch_name(id) == "zmax")
            {
                ASSERT_EQ(batch.face_lids.size(), 1u);
                top = batch.face_lids.front();
            }
        ASSERT_GE(top, 0);
        EXPECT_DOUBLE_EQ(f.grid->face_area(top), 1);
        const double factor = target / (target + dt * .4);
        const auto run = [&]
        {
            motion.begin_trial(target, dt);
            const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*f.grid, motion);
            EXPECT_DOUBLE_EQ(ale.old_cell_volumes()[0], 1);
            EXPECT_DOUBLE_EQ(ale.new_cell_volumes()[0], target);
            ALEFaceFieldType absolute(f.grid, 0., "absolute"), relative(f.grid, 0., "relative");
            for (const auto face : absolute.owned_face_ids())
                absolute.set_owned_value(face, ale.face_mesh_fluxes()[static_cast<std::size_t>(face)]);
            absolute.sync_ghosts();
            SimpleFluid::FVM::mesh_relative_face_fluxes(absolute, ale, relative);
            EXPECT_DOUBLE_EQ(relative.value(top), 0);
            f.model.refresh_geometry();
            f.model.advance(
                dt, dt, f.temperature, f.pressure, f.velocity, relative, f.material, &ale, SimpleFluid::Dimension::Z);
            for (std::size_t bin = 0; bin < 2; ++bin)
            {
                const auto name = bin == 0 ? "micro" : "large";
                EXPECT_NEAR(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, name),
                    initial_moles[bin] * factor, 2e-18);
                double escaped = 0;
                for (const auto& e : f.model.physical_escape_events())
                    if (e.species == GeneralGas::one && e.population == name)
                        escaped += e.moles;
                EXPECT_NEAR(escaped, initial_moles[bin] * (1 - factor), 2e-18);
                EXPECT_NEAR(GeneralGas::number_escape(f.model, name), p.number_density[bin] * (1 - factor), 1e-2);
            }
            EXPECT_NEAR(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dissolved),
                initial_dissolved, 5e-17);
            const double escaped = GeneralGas::step_escape(f.model, GeneralGas::one, true);
            EXPECT_NEAR(escaped, (initial_moles[0] + initial_moles[1]) * (1 - factor), 3e-18);
            EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 5e-14);
            EXPECT_NEAR(f.model.global_submerged_bubble_volume(), initial_volume * factor,
                std::max(1e-20, std::abs(initial_volume * factor) * 2e-8));
            EXPECT_NEAR(f.model.transported_bubble_slip_volume_flux().value(top), .4 * initial_volume * factor / target,
                std::max(1e-20, std::abs(.4 * initial_volume * factor / target) * 1e-12));
            EXPECT_DOUBLE_EQ(f.model.transported_bubble_carrier_volume_flux().value(top), 0);
            return f.model.species_histories();
        };
        const auto first = run();
        f.model.restore(checkpoint);
        motion.rollback_trial();
        f.model.refresh_geometry();
        EXPECT_EQ(f.model.species_histories(), original_histories);
        EXPECT_DOUBLE_EQ(
            GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro"), initial_moles[0]);
        const auto repeated = run();
        EXPECT_EQ(repeated, first);
        f.model.restore(checkpoint);
        motion.rollback_trial();
        f.model.refresh_geometry();
    }
}

/**
 * @brief Finite-Courant escape closes step and cumulative inventories.
 */
TEST(ReactingGasModelTest, FreeSurfaceEscapeAccumulatesGlobally)
{
    GeneralGas::Parameters p;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-6;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = 10;
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    const auto start = f.model.snapshot();
    constexpr double dt = .1;
    f.advance(dt, dt);
    EXPECT_NEAR(GeneralGas::step_escape(f.model), 5e-7, 1e-14);
    EXPECT_NEAR(GeneralGas::step_escape(f.model, GeneralGas::one, true), 5e-7, 1e-14);
    EXPECT_NEAR(GeneralGas::number_escape(f.model, "micro"), 5e9, 1e-2);
    EXPECT_DOUBLE_EQ(GeneralGas::number_escape(f.model, "large"), 0);
    EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one) + GeneralGas::step_escape(f.model), 1e-6, 1e-14);
    EXPECT_NEAR(
        GeneralGas::integral(f.model.population_number_field("micro")) + GeneralGas::number_escape(f.model, "micro"),
        1e10, 1e-2);
    const auto first = f.model.population_escape_receipt();
    const double first_molar = GeneralGas::step_escape(f.model);
    EXPECT_EQ(first.event, f.model.species_event_receipt());
    f.advance(2 * dt, dt);
    EXPECT_NEAR(GeneralGas::history(f.model).physical_outflow, first_molar + GeneralGas::step_escape(f.model), 1e-14);
    EXPECT_NEAR(GeneralGas::integral(f.model.population_number_field("micro")) + first.events.front().number +
                    GeneralGas::number_escape(f.model, "micro"),
        1e10, 1e-2);
    // Escape is already transport; do not debit it again as local kinetic gas destruction.
    EXPECT_NEAR(f.model.source_alpha_reaction().value(0), 0, 1e-14);
    f.model.restore(start);
    f.advance(dt, dt);
    EXPECT_EQ(f.model.population_escape_receipt().events, first.events);
    EXPECT_GT(f.model.population_escape_receipt().event.publication_nonce, first.event.publication_nonce);
}

/** @brief Dissolved boundary outflow is not reported as escaped bubbles. */
TEST(ReactingGasModelTest, DissolvedBoundaryOutflowIsSeparateFromBubbleEscape)
{
    GeneralGas::Parameters p;
    p.dissolved_one = 1e-3;
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    for (const auto face : f.flux.owned_face_ids())
        if (f.grid->is_boundary_face(face) && f.grid->boundary_batch_name(f.grid->boundary_id(face)) == "zmax")
            f.flux.set_value(face, 1);
    f.advance(.1, .1);
    EXPECT_GT(GeneralGas::step_escape(f.model), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::step_escape(f.model, GeneralGas::one, true), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::number_escape(f.model, "micro"), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::number_escape(f.model, "large"), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).physical_outflow, GeneralGas::step_escape(f.model));
    for (const auto& event : f.model.physical_escape_events())
        if (event.species == GeneralGas::one)
            EXPECT_EQ(event.phase, GeneralGas::Phase::Dissolved);
    EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-14);
}

/** @brief Many cells must not turn summation error into apparent bubble creation. */
TEST(ReactingGasModelTest, LargeCartesianTransportInventoryIsStable)
{
    SimpleFluid::ArrReal x, y, z;
    for (int i = 0; i <= 20; ++i)
        x.push_back(.2 * i / 20);
    for (int i = 0; i <= 10; ++i)
        y.push_back(.02 * i / 10);
    for (int i = 0; i <= 80; ++i)
        z.push_back(.2 * i / 80);
    auto mesh = std::make_shared<ALEMeshType>(
        std::make_shared<ALEMeshType::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{x, y, z}}));
    GeneralGas::Parameters p;
    p.number_density[0] = 1.2345678912345e15;
    p.molar_one[0] = 3.141592653589793e-5;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = .005;
    p.controls.transport_solver_tolerance = 1e-14;
    GeneralGas::Fixture<ALEMeshType> f(mesh, p);
    f.initialize();
    const double expected = p.molar_one[0] * .2 * .02 * .2;
    const double roundoff = 64 * std::numeric_limits<double>::epsilon() * expected;
    EXPECT_NEAR(
        GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro"), expected, roundoff);
    ASSERT_NO_THROW(f.advance(.02, .02));
    EXPECT_NEAR(
        GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro"), expected, roundoff);
    EXPECT_LE(GeneralGas::step_escape(f.model), roundoff);
}

/** @brief Positive escape below the old relative cutoff remains observable. */
TEST(ReactingGasModelTest, SmallPositiveEscapeIsNotDiscarded)
{
    GeneralGas::Parameters p;
    p.dissolved_one = 1e9;
    p.molar_mass_one = 1e-9;
    // Explicit synthetic light tracer preserves the original1e9mol/m3 stress at
    // finite1000kg/m3 solution mass. A hydrogen-weight stock at that concentration was impossible.
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    for (const auto face : f.flux.owned_face_ids())
        if (f.grid->is_boundary_face(face) && f.grid->boundary_batch_name(f.grid->boundary_id(face)) == "zmax")
            f.flux.set_value(face, 1e-13);
    const double before = f.model.global_species_moles(GeneralGas::one);
    f.advance(.1, .1);
    const double escaped = GeneralGas::step_escape(f.model);
    EXPECT_GT(escaped, 0);
    EXPECT_LT(escaped, 64 * std::numeric_limits<double>::epsilon() * before);
    EXPECT_DOUBLE_EQ(GeneralGas::history(f.model).physical_outflow, escaped);
    EXPECT_DOUBLE_EQ(GeneralGas::step_escape(f.model, GeneralGas::one, true), 0);
}

/** @brief Verify escape-rate diagnostics are localized to free-surface cells. */
TEST(ReactingGasModelTest, EscapeRateFieldsAreBoundaryLocalized)
{
    GeneralGas::Parameters p;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-6;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = 1;
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(
        SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(1, 1, 2)), p);
    f.initialize();
    constexpr double dt = .1;
    f.advance(dt, dt);
    const auto& catalog = f.model.options().reaction_mechanism->catalog();
    double molar = 0, number = 0;
    for (std::size_t row = 0; row < f.grid->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const double cv = f.grid->cell_volume(cell);
        const double escaped = -f.model.escape_ledgers()[row].externalChange.amount(
            catalog, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro");
        if (f.grid->cell_centroid(cell).z > 1)
        {
            EXPECT_GT(escaped, 0);
            // Independent owner-oriented top-face N flux: dt*u*A*N_new, area1m2.
            const double face_number = dt * f.model.population_number_field("micro").value(cell);
            EXPECT_GT(face_number, 0);
            number += face_number;
        }
        else
            EXPECT_DOUBLE_EQ(escaped, 0);
        const double rate = escaped / (dt * cv);
        molar += dt * cv * rate;
    }
    EXPECT_NEAR(molar, GeneralGas::step_escape(f.model), 1e-14);
    EXPECT_NEAR(number, GeneralGas::number_escape(f.model, "micro"), 1e-2);
}

/**
 * @brief Celata rise-velocity selection produces free-surface escape.
 */
TEST(ReactingGasModelTest, CelataSelectorDrivesBubbleEscape)
{
    GeneralGas::Parameters p;
    p.number_density[1] = 100;
    p.molar_one[1] = 1e-6;
    const double r = GeneralGas::radius(1e-6, 100, 300, p.pressure, p.sigma);
    p.molar_mass_one = 1.2 * GeneralGas::gas_constant * 300 / (p.pressure + 2 * p.sigma / r);
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::Celata2007;
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    f.advance(1e-6, 1e-6);
    EXPECT_GT(GeneralGas::step_escape(f.model), 0);
    EXPECT_GT(GeneralGas::number_escape(f.model, "large"), 0);
    EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-14);
}

/**
 * @brief Axial bubble transport ignores transverse liquid fluxes.
 */
TEST(ReactingGasModelTest, AxialTransportDropsTransverseLiquidFlux)
{
    GeneralGas::Parameters p;
    p.number_density[1] = 100;
    p.molar_one[1] = 1e-6;
    p.controls.free_surface_patches = {"xmax"};
    GeneralGas::Fixture<MeshType> general(make_single_cell_mesh(), p), axial(make_single_cell_mesh(), p);
    general.velocity.put_scalar(MeshType::Vec3{.1, 0, 0});
    axial.velocity.put_scalar(MeshType::Vec3{.1, 0, 0});
    for (const auto face : general.flux.owned_face_ids())
        general.flux.set_value(face, .1 * general.grid->face_area_vector(face).x);
    // Axial filtering is explicit caller ownership, not an ignored specialized backend selector.
    for (const auto face : axial.flux.owned_face_ids())
    {
        const auto normal = axial.grid->face_area_vector(face);
        axial.flux.set_value(face, normal.z == 0 ? 0 : .1 * normal.x);
    }
    general.initialize();
    axial.initialize();
    general.advance(1e-6, 1e-6);
    axial.advance(1e-6, 1e-6);
    EXPECT_GT(GeneralGas::step_escape(general.model), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::step_escape(axial.model), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::number_escape(axial.model, "large"), 0);
}

/**
 * @brief Population transport conserves global number and H2 without escape.
 */
TEST(ReactingGasModelTest, PopulationTransportConservesInventoryWithoutEscape)
{
    GeneralGas::Parameters p;
    p.number_density[1] = 1e6;
    p.molar_one[1] = 1e-7;
    const double radius = GeneralGas::radius(1e-7, 1e6, 300, p.pressure, p.sigma);
    p.dissolved_one = 1e-5 * (p.pressure + 2 * p.sigma / radius);
    GeneralGas::Fixture<MeshType> f(SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database()), p);
    f.initialize();
    for (const auto face : f.flux.owned_face_ids())
        f.flux.set_value(face, f.grid->is_boundary_face(face) ? 0 : .1 * f.grid->face_area_vector(face).x);
    const double number = GeneralGas::integral(f.model.population_number_field("large"));
    const double moles = GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large");
    const double total = f.model.global_species_moles(GeneralGas::one);
    f.advance(1e-4, 1e-4);
    EXPECT_NEAR(GeneralGas::integral(f.model.population_number_field("large")), number, number * 1e-9);
    EXPECT_NEAR(
        GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), moles, moles * 1e-9);
    EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), total, total * 1e-9);
    EXPECT_DOUBLE_EQ(GeneralGas::step_escape(f.model), 0);
    EXPECT_DOUBLE_EQ(GeneralGas::number_escape(f.model, "large"), 0);
    double change = 0;
    for (std::size_t row = 0; row < f.grid->num_owned_cells(); ++row)
        change = std::max(change,
            std::abs(f.model.population_number_field("large").value(static_cast<Pack::local_ordinal_type>(row)) - 1e6));
    EXPECT_GT(change, 1e6 * 1e-8);
}

/**
 * @brief Higher pressure raises the critical concentration and delays conversion.
 */
TEST(ReactingGasModelTest, HigherPressureRaisesCriticalConcentrationAndDelaysConversion)
{
    constexpr double low = 1e5, high = 5e5;
    GeneralGas::Parameters base;
    base.number_density[0] = 1e8;
    base.molar_one[0] = 1e-8;
    const auto critical = [&](double pressure) { return 1e-5 * (pressure + 2 * base.sigma / base.nucleation_radius); };
    EXPECT_GT(critical(high), critical(low));
    base.dissolved_one = .5 * (critical(low) + critical(high));
    const auto run = [&](double pressure)
    {
        auto p = base;
        p.pressure = pressure;
        p.population = [sigma = p.sigma, rn = p.nucleation_radius](
                           const SimpleFluid::BubbleGasPhysics::MultispeciesPopulationContext& c, double dt)
        {
            const auto& e = c.chemistry.environment;
            const auto& catalog = c.chemistry.catalog;
            const double concentration =
                c.chemistry.state.amount(catalog, GeneralGas::one, GeneralGas::Phase::Dissolved) / e.liquidVolume;
            SimpleFluid::BubbleGasPhysics::MultispeciesPopulationPacket result;
            if (concentration > 1e-5 * (e.absolutePressure + 2 * sigma / rn))
            {
                const double fraction = -std::expm1(-1e-4 * e.absolutePressure * dt);
                const double n =
                    c.chemistry.state.amount(catalog, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro");
                result.transfers = {{GeneralGas::one, GeneralGas::Phase::Dispersed, GeneralGas::Phase::Dispersed,
                    "micro", "large", n * fraction}};
                const double count = c.populations.front().number * fraction;
                result.numberEvents = {{"micro", 0, count}, {"large", count, 0}};
            }
            return result;
        };
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        f.advance(1e-6, 1e-6);
        EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-14);
        return GeneralGas::integral(f.model.population_number_field("large"));
    };
    const double converted_low = run(low), converted_high = run(high);
    EXPECT_GT(converted_low, converted_high);
    EXPECT_GT(converted_low / 1e-6, converted_high / 1e-6);
}

/**
 * @brief Constant and flat prescribed-pressure histories produce the same state.
 */
TEST(ReactingGasModelTest, ConstantAndPrescribedPressureModesRegressTogether)
{
    GeneralGas::Parameters base;
    base.number_density[0] = 1e8;
    base.molar_one[0] = 1e-8;
    base.dissolved_one = 1e3;
    base.micro_decay_time = 1e9;
    base.micro_conversion_frequency = .001;
    const auto run = [&](bool history)
    {
        auto p = base;
        if (history)
        {
            p.controls.pressure_mode = SimpleFluid::ReactingPressureMode::PrescribedHistory;
            p.controls.pressure_history_times = {0, 1};
            p.controls.pressure_history_values = {p.pressure, p.pressure};
        }
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        f.advance(.25, 1e-7);
        return std::array<double, 4>{f.model.absolute_pressure().value(0),
            1e-5 * (f.model.absolute_pressure().value(0) + 2 * p.sigma / p.nucleation_radius),
            f.model.alpha_g().value(0), f.model.global_species_moles(GeneralGas::one)};
    };
    const auto fixed = run(false), prescribed = run(true);
    for (std::size_t i = 0; i < fixed.size(); ++i)
        EXPECT_NEAR(prescribed[i], fixed[i], std::max(1e-12, std::abs(fixed[i]) * 1e-12));
}

/**
 * @brief Supplied pressure evolution stays positive; unsupported inertial ownership rejects explicitly.
 */
TEST(ReactingGasModelTest, ConstantAndInertialPressureModesRunFromSameCase)
{
    GeneralGas::Parameters p;
    p.number_density[1] = 1e6;
    p.molar_one[1] = 1e-7;
    p.dissolved_one = 5;
    GeneralGas::Fixture<MeshType> fixed(make_single_cell_mesh(), p);
    fixed.initialize();
    for (int step = 0; step < 2; ++step)
    {
        fixed.temperature.put_scalar(300 + .5 * step);
        fixed.advance((step + 1) * 1e-5, 1e-5);
        EXPECT_TRUE(std::isfinite(fixed.model.absolute_pressure().value(0)));
        EXPECT_GE(fixed.model.absolute_pressure().value(0), fixed.model.options().minimum_absolute_pressure);
        EXPECT_NEAR(GeneralGas::history(fixed.model).inventory_error, 0, 1e-14);
    }
    // A general species backend cannot silently freeze the retired specialized
    // inertial pressure history. Its physical evolution belongs to a supplied flow/EOS owner.
    SimpleFluid::Database retired;
    retired.set("reacting_pressure_mode", std::string("inertial"));
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(retired), std::invalid_argument);
}

/**
 * @brief More local subcycles move stiff kinetics toward the fine solution.
 */
TEST(ReactingGasModelTest, StiffLocalKineticsConvergesWithSubcycles)
{
    const auto run = [](int resolution)
    {
        constexpr double lifetime = 1e-7;
        GeneralGas::Parameters p;
        p.reactant_concentration = 1e3;
        p.controls.reaction_advance.maximumSubstep = lifetime / resolution;
        p.source = [](const GeneralGas::R::ReactionContext& c, double)
        {
            GeneralGas::R::ReactionRateProvider provider;
            provider.identity = "test.supplied_stiff_finite_reactant";
            provider.schemaRevision = c.catalog.revision();
            provider.schemaFingerprint = c.catalog.fingerprint();
            provider.validityDescription = "Finite A solution, k=1e7/s";
            provider.parameterUnits = "mol/(m3 liquid s)";
            provider.validate = [](const auto& context)
            { GeneralGas::M::require(context.environment.liquidVolume > 0, "Finite reactant needs actual liquid."); };
            provider.evaluate = [](const auto& context)
            {
                return std::vector<GeneralGas::R::ReactionRate>{{GeneralGas::M::conversion,
                    1e7 * context.state.amount(context.catalog, GeneralGas::M::reactant, GeneralGas::Phase::Liquid) /
                        context.environment.liquidVolume,
                    GeneralGas::R::SourceVolumeBasis::LiquidVolume}};
            };
            GeneralGas::R::ReactionSourceInput input;
            input.rates = std::move(provider);
            return input;
        };
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        f.advance(2e-6, 2e-6);
        return std::array<double, 3>{
            GeneralGas::compartment(f.model, GeneralGas::M::reactant, GeneralGas::Phase::Liquid),
            GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dissolved),
            GeneralGas::compartment(f.model, GeneralGas::two, GeneralGas::Phase::Dissolved)};
    };
    const auto coarse = run(1), medium = run(5), fine = run(100);
    const auto error = [&](const auto& value)
    {
        double e = 0;
        for (std::size_t i = 0; i < value.size(); ++i)
            e += std::abs(value[i] - fine[i]);
        return e;
    };
    EXPECT_GT(error(coarse), 0);
    EXPECT_LT(error(medium), error(coarse));
    const double exact = 1e3 * std::exp(-20.);
    EXPECT_NEAR(fine[0], exact, exact * .001);
}

/**
 * @brief Explicit admission retains declined requests outside the finite accepted inventories.
 */
TEST(ReactingGasModelTest, ClippedConcentrationAndVoidBoundsReportExcludedInventory)
{
    constexpr double requested_dissolved = 5, admitted_dissolved = 1, requested_gas = 1, alpha_max = .01, number = 1e12;
    const double radius = std::cbrt(3 * alpha_max / (4 * std::numbers::pi * number));
    // This explicit initial admission law retains rejected requests in its named
    // external reservoir. It never stores an overcapacity phase then clips its display.
    const double capacity_moles = alpha_max * (1e5 + 2 * .07 / radius) / (GeneralGas::gas_constant * 300);
    const double admitted_gas =
        capacity_moles * (1 - 8e-12); // Reserve the declared4e-12 EOS inversion residual, twice.
    GeneralGas::Parameters p;
    p.dissolved_one = admitted_dissolved;
    p.number_density[1] = number;
    p.molar_one[1] = admitted_gas;
    p.controls.alpha_max = alpha_max;
    GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
    f.initialize();
    EXPECT_LE(f.model.alpha_g().value(0), alpha_max);
    EXPECT_GT(f.model.alpha_g().value(0), 0);
    EXPECT_NEAR(f.model.global_submerged_bubble_volume(), f.model.evaluate_submerged_bubble_volume(p.pressure),
        f.model.global_submerged_bubble_volume() * 1e-11);
    const double liquid = f.model.alpha_l().value(0);
    EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).value(0) / liquid,
        admitted_dissolved, 1e-14);
    const double reservoir_dissolved = (requested_dissolved - admitted_dissolved) * liquid;
    const double reservoir_gas = requested_gas - admitted_gas;
    EXPECT_GT(reservoir_dissolved, 0);
    EXPECT_GT(reservoir_gas, 0);
    EXPECT_NEAR(
        GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large") + reservoir_gas,
        requested_gas, 1e-12);
    EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).value(0) + reservoir_dissolved,
        requested_dissolved * liquid, 1e-12);
    auto invalid = p;
    invalid.molar_one[1] = requested_gas;
    GeneralGas::Fixture<MeshType> overcapacity(make_single_cell_mesh(), invalid);
    EXPECT_THROW(overcapacity.initialize(), std::exception);
    EXPECT_FALSE(overcapacity.model.initial_state_initialized());
}

/**
 * @brief All pressure modes keep published state finite and bounded.
 */
TEST(ReactingGasModelTest, StateFieldsRemainFiniteNonNegativeAndBoundedAcrossPressureModes)
{
    for (const auto mode : {SimpleFluid::ReactingPressureMode::Constant,
             SimpleFluid::ReactingPressureMode::PrescribedHistory, SimpleFluid::ReactingPressureMode::Reconstructed})
    {
        auto p = GeneralGas::populated();
        p.controls.pressure_mode = mode;
        if (mode == SimpleFluid::ReactingPressureMode::PrescribedHistory)
        {
            p.controls.pressure_history_times = {0, 1};
            p.controls.pressure_history_values = {1e5, 2e5};
        }
        GeneralGas::Fixture<MeshType> f(make_single_cell_mesh(), p);
        f.initialize();
        for (int step = 0; step < 3; ++step)
        {
            f.temperature.put_scalar(300 + step * .5);
            f.advance((step + 1) * 1e-6, 1e-6);
            for (const auto& [name, field] : f.model.output_fields())
                for (std::size_t row = 0; row < f.grid->num_owned_cells(); ++row)
                    EXPECT_TRUE(std::isfinite(field->value(static_cast<Pack::local_ordinal_type>(row)))) << name;
            const auto state = f.model.cell_species_state(0);
            state.validate(f.model.options().reaction_mechanism->catalog());
            for (const auto value : state.amounts)
                EXPECT_GE(value, 0);
            for (const auto& population : state.dispersedPopulations)
                for (const auto value : population.amounts)
                    EXPECT_GE(value, 0);
            EXPECT_GE(f.model.alpha_g().value(0), 0);
            EXPECT_LE(f.model.alpha_g().value(0), p.controls.alpha_max);
        }
    }
}

/**
 * @brief Solver startup publishes initial bubble void and the scalar mirror.
 */
TEST(ReactingGasModelTest, SolverPublishesInitialShengStateInEitherConfigurationOrder)
{
    GeneralGas::Parameters p;
    p.dissolved_one = 2.5;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-3;
    auto options = GeneralGas::options(p);

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-6;
    time_options.reference_temperature = 300.0;
    auto model_options = SimpleFluid::BoussinesqModelOptions::legacy_defaults(time_options);
    model_options.reference_density = 1000.0;
    model_options.density = 1000.0;
    model_options.specific_heat_capacity = 4200.0;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = 0.6;

    const auto expect_published = [&](const auto& reacting_gas, const auto& scalar_void)
    {
        ASSERT_TRUE(reacting_gas.initial_state_initialized());
        const auto alpha_g = reacting_gas.alpha_g().value(0);
        ASSERT_GT(alpha_g, options.alpha_min);
        EXPECT_NEAR(scalar_void.alpha_g().value(0), alpha_g, 1.0e-14);
        EXPECT_NEAR(scalar_void.alpha_l().value(0), 1.0 - alpha_g, 1.0e-14);
        EXPECT_DOUBLE_EQ(reacting_gas.source_alpha_reaction().value(0), 0.0);
        EXPECT_DOUBLE_EQ(scalar_void.source_alpha_total().value(0), 0.0);
        EXPECT_NEAR(reacting_gas.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).value(0) /
                        reacting_gas.alpha_l().value(0),
            p.dissolved_one, 1.0e-14);
        EXPECT_NEAR(reacting_gas.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).value(0),
            (1.0 - alpha_g) * p.dissolved_one, 1.0e-14);
    };

    {
        auto mesh = make_single_cell_mesh();
        auto solver_shared = std::make_shared<SimpleFluid::BoussinesqSolver<Pack>>(
            mesh, SimpleFluid::BoundaryConditionSet{}, time_options, SimpleFluid::LinearSolverOptions{}, model_options);
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        solver_coupling.initialize_heated_box(300.0, 300.0);
        const auto& reacting_gas = solver_coupling.configure_reacting_gas(options);
        const auto* scalar_void = solver_coupling.find_scalar_void_fraction_model();
        ASSERT_NE(scalar_void, nullptr);
        expect_published(reacting_gas, *scalar_void);
    }

    {
        auto mesh = make_single_cell_mesh();
        auto solver_shared = std::make_shared<SimpleFluid::BoussinesqSolver<Pack>>(
            mesh, SimpleFluid::BoundaryConditionSet{}, time_options, SimpleFluid::LinearSolverOptions{}, model_options);
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        const auto& reacting_gas = solver_coupling.configure_reacting_gas(options);
        ASSERT_FALSE(reacting_gas.initial_state_initialized());

        solver_coupling.initialize_heated_box(300.0, 300.0);

        const auto* scalar_void = solver_coupling.find_scalar_void_fraction_model();
        ASSERT_NE(scalar_void, nullptr);
        expect_published(reacting_gas, *scalar_void);
    }
}

/**
 * @brief First temperature solve uses feedback from reconstructed initial void.
 */
TEST(ReactingGasModelTest, FirstPhysicalSolveUsesReconstructedShengVoidFeedback)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-3;
    time_options.reference_temperature = 300.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = 0.0;
    time_options.gravity_z = 0.0;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 1000.0;
    model_options.density = 1000.0;
    model_options.specific_heat_capacity = 1.0;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = 0.0;
    auto solver_shared = std::make_shared<SimpleFluid::BoussinesqSolver<Pack>>(
        mesh, SimpleFluid::BoundaryConditionSet{}, time_options, SimpleFluid::LinearSolverOptions{}, model_options);
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.initialize_heated_box(300.0, 300.0);

    SimpleFluid::PrescribedHeatSourceOptions prescribed_heat;
    prescribed_heat.profile = SimpleFluid::PrescribedHeatProfile::Constant;
    prescribed_heat.power_density = 0.0;
    solver_coupling.configure_prescribed_heat_source(prescribed_heat);
    constexpr double power_density = 1.0e6;
    solver.add_temperature_source("startup_heating", power_density);

    GeneralGas::Parameters p;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 5;
    auto options = GeneralGas::options(p);
    const auto& reacting_gas = solver_coupling.configure_reacting_gas(options);
    const auto initial_alpha = reacting_gas.alpha_g().value(0);
    ASSERT_GT(initial_alpha, 0.05);

    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::Mixture;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_dynamic_viscosity = 1.0e-3;
    solver_coupling.configure_material_feedback(feedback);

    solver_coupling.step();

    const auto initial_mixture_density =
        feedback.liquid_density * (1.0 - initial_alpha) + feedback.gas_density * initial_alpha;
    const auto expected_temperature = 300.0 + time_options.time_step * power_density /
                                                  (initial_mixture_density * model_options.specific_heat_capacity);
    EXPECT_NEAR(solver.temperature().value(0), expected_temperature, 1.0e-10);
    EXPECT_GT(
        solver.temperature().value(0), 300.0 + time_options.time_step * power_density /
                                                   (model_options.density * model_options.specific_heat_capacity));
}

/**
 * @brief BoussinesqSolver owns, publishes, and removes the optional gas model.
 */
TEST(ReactingGasModelTest, SolverOwnsAndPublishesOptionalModel)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    auto model_options = SimpleFluid::BoussinesqModelOptions::legacy_defaults(time_options);
    auto solver_shared = std::make_shared<SimpleFluid::BoussinesqSolver<Pack>>(
        mesh, SimpleFluid::BoundaryConditionSet{}, time_options, SimpleFluid::LinearSolverOptions{}, model_options);
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_reacting_gas(GeneralGas::options());
    ASSERT_NE(solver_coupling.find_reacting_gas_model(), nullptr);

    const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output =
        std::filesystem::temp_directory_path() / ("SimpleFluid_reacting_gas_" + std::to_string(unique_id) + ".vtu");
    solver_coupling.write_solution_vtu(output.string(),
        SimpleFluid::SolutionOutputOptions{.include_sources = true, .include_reacting_gas_fields = true});

    std::ifstream input(output);
    const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("Name=\"S_alpha_reaction\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"alpha_g\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"p_abs\""), std::string::npos);

    EXPECT_TRUE(solver_coupling.remove_reacting_gas_model());
    EXPECT_EQ(solver_coupling.find_reacting_gas_model(), nullptr);
    std::filesystem::remove(output);
}

/** @brief Composite transport agrees with independently assembled explicit geometry. */
TEST(ReactingGasModelTest, CompositeTransportLeaseMatchesExplicitReference)
{
    auto composite = make_gas_composite_column();
    auto mesh = std::make_shared<ALEMeshType>(composite);
    if (mesh->owned_cell_map()->getComm()->getSize() != 1)
        GTEST_SKIP() << "The independent explicit geometry oracle uses one rank.";
    auto explicit_mesh = std::make_shared<ALEMeshType>(SimpleFluid::test::explicit_reference(*composite));
    const auto options = GeneralGas::options(GeneralGas::populated());
    const auto advance = [&](const SimpleFluid::SP<ALEMeshType>& selected)
    {
        ALEReactingModelType model(selected, options);
        ALEFieldType temperature(selected, 300., "temperature"), pressure(selected, 0., "pressure"),
            power(selected, 0., "power");
        ALEVelocityFieldType velocity(selected, ALEMeshType::Vec3{}, "velocity");
        ALEFaceFieldType flux(selected, 0., "flux");
        for (const auto face : flux.owned_face_ids())
            if (!selected->is_boundary_face(face))
                flux.set_owned_value(face, .05 * selected->face_area_vector(face).x);
        flux.sync_ghosts();
        auto material = make_ale_water_properties(selected);
        model.initialize_state(0., temperature, pressure, velocity, material);
        const double before = model.global_species_moles(GeneralGas::one);
        model.advance(.01, .01, temperature, pressure, velocity, flux, material);
        const auto& statistics = model.last_statistics();
        EXPECT_GE(statistics.transport_linear.solves, 6);
        EXPECT_GT(GeneralGas::step_escape(model, GeneralGas::one, true), 0.);
        EXPECT_NEAR(model.global_species_moles(GeneralGas::one) + GeneralGas::step_escape(model), before, 1e-12);
        EXPECT_NEAR(GeneralGas::history(model).inventory_error, 0., 1e-12);
        EXPECT_NEAR(GeneralGas::history(model).inventory_error, 0., 1e-12);
        std::map<std::string, std::vector<double>> result;
        for (const auto& [name, field] : model.output_fields())
            for (size_t row = 0; row < selected->num_owned_cells(); ++row)
                result[name].push_back(field->value(static_cast<Pack::local_ordinal_type>(row)));
        return result;
    };
    const auto actual = advance(mesh);
    EXPECT_NO_THROW(composite->require_geometry_writable());
    const auto expected = advance(explicit_mesh);
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [name, values] : expected)
    {
        SCOPED_TRACE(name);
        const auto& obtained = actual.at(name);
        ASSERT_EQ(obtained.size(), values.size());
        for (size_t cell = 0; cell < values.size(); ++cell)
            EXPECT_NEAR(obtained[cell], values[cell], std::max(1e-17, std::abs(values[cell]) * 1e-9));
    }
}

/** @brief Composite ALE transport releases its lease on success/failure and
 * retains the assembly-time GCL check after the initial advance preflight.
 */
TEST(ReactingGasModelTest, CompositeTransportLeasePreservesALEReplayAndAssemblyGCL)
{
    for (const double target : {.8, 1.2})
    {
        SCOPED_TRACE(target);
        auto composite = make_gas_composite_column();
        auto mesh = std::make_shared<ALEMeshType>(composite);
        if (mesh->owned_cell_map()->getComm()->getSize() != 1)
            GTEST_SKIP() << "The assembly-time fault fixture uses one rank.";
        gas_gcl_fault_motion = nullptr;
        gas_gcl_fault_armed = false;
        gas_gcl_fault_injected = false;
        auto p = GeneralGas::populated();
        p.diffusivity_two = [](const auto& c) { return gas_diffusivity_with_gcl_fault(c.environment.temperature); };
        auto options = GeneralGas::options(p);
        ALEReactingModelType model(mesh, options);
        ALEFieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
        ALEVelocityFieldType velocity(mesh, ALEMeshType::Vec3{}, "velocity");
        ALEFaceFieldType relative_flux(mesh, 0., "relative_flux");
        auto material = make_ale_water_properties(mesh);
        model.initialize_state(0., temperature, pressure, velocity, material);
        const auto checkpoint = model.snapshot();
        const double before = model.global_species_moles(GeneralGas::one);
        ALEMotionType motion(mesh);
        constexpr double dt = .01;
        const auto fields = [&]
        {
            std::map<std::string, std::vector<double>> result;
            for (const auto& [name, field] : model.output_fields())
                for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
                    result[name].push_back(field->value(static_cast<Pack::local_ordinal_type>(row)));
            return result;
        };
        const auto advance = [&]
        {
            motion.begin_trial(target, dt);
            const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
            model.refresh_geometry();
            model.advance(
                dt, dt, temperature, pressure, velocity, relative_flux, material, &ale, SimpleFluid::Dimension::Z);
            EXPECT_NEAR(model.global_species_moles(GeneralGas::one) + GeneralGas::step_escape(model), before, 1e-12);
            EXPECT_NEAR(GeneralGas::history(model).inventory_error, 0., 1e-12);
            EXPECT_NO_THROW(composite->require_geometry_writable());
            return fields();
        };
        const auto rollback = [&]
        {
            motion.rollback_trial();
            // Public restore must refresh the new rollback epoch itself.
            // Omitting an extra caller refresh exercises standalone restore.
            model.restore(checkpoint);
        };
        const auto expected = advance();
        rollback();
        const auto replay = advance();
        for (const auto& [name, values] : expected)
        {
            SCOPED_TRACE(name);
            for (size_t cell = 0; cell < values.size(); ++cell)
                EXPECT_NEAR(replay.at(name)[cell], values[cell], std::max(1e-17, std::abs(values[cell]) * 1e-11));
        }
        rollback();

        motion.begin_trial(target, dt);
        {
            MutableGasALEView altered(mesh, motion);
            const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, altered);
            gas_gcl_fault_motion = &altered;
            model.refresh_geometry();
            gas_gcl_fault_armed = true;
            std::string message;
            try
            {
                model.advance(
                    dt, dt, temperature, pressure, velocity, relative_flux, material, &ale, SimpleFluid::Dimension::Z);
            }
            catch (const std::invalid_argument& error)
            {
                message = error.what();
            }
            EXPECT_TRUE(gas_gcl_fault_injected);
            EXPECT_NE(message.find("geometric conservation law"), std::string::npos);
            EXPECT_NO_THROW(composite->require_geometry_writable());
            gas_gcl_fault_motion = nullptr;
            gas_gcl_fault_armed = false;
        }
        rollback();
        const auto recovered = advance();
        for (const auto& [name, values] : expected)
        {
            SCOPED_TRACE(name);
            for (size_t cell = 0; cell < values.size(); ++cell)
                EXPECT_NEAR(recovered.at(name)[cell], values[cell], std::max(1e-17, std::abs(values[cell]) * 1e-11));
        }
        rollback();
    }
}

GeneralGas::Parameters zero_ale_transport_options()
{
    GeneralGas::Parameters p;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = .4;
    p.controls.free_surface_patches = {"zmax"};
    return p;
}
/** Cached and independently refreshed numeric operators share only prescribed inputs. */
struct ALEZeroTransportPair
{
    SimpleFluid::SP<ALEMeshType> mesh = std::make_shared<ALEMeshType>(make_gas_composite_column());
    GeneralGas::Parameters parameters;
    ALEFieldType temperature{mesh, 300., "temperature"}, pressure{mesh, 0., "pressure"}, power{mesh, 0., "power"};
    ALEVelocityFieldType velocity{mesh, ALEMeshType::Vec3{}, "velocity"};
    ALEFaceFieldType flux{mesh, 0., "relative_flux"};
    SimpleFluid::MaterialPropertyFields<Pack, ALEMeshType> material = make_ale_water_properties(mesh);
    ALEReactingModelType full, reduced;
    ALEMotionType motion{mesh};
    explicit ALEZeroTransportPair(GeneralGas::Parameters p)
        : parameters(std::move(p)), full(mesh, GeneralGas::options(parameters, &power)),
          reduced(mesh, GeneralGas::options(parameters, &power))
    {
        full.initialize_state(0, temperature, pressure, velocity, material);
        reduced.initialize_state(0, temperature, pressure, velocity, material);
    }
    void advance(double top, double time, double dt)
    {
        motion.begin_trial(top, dt);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
        full.refresh_geometry();
        GeneralGas::force_numeric_reassembly(reduced);
        for (auto* model : {&full, &reduced})
        {
            model->advance(time, dt, temperature, pressure, velocity, flux, material, &ale);
        }
    }
    void compare() const
    {
        for (const auto& [name, field] : full.output_fields())
        {
            const auto* actual = reduced.output_fields().at(name);
            for (std::size_t row = 0; row < mesh->num_local_cells(); ++row)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(row);
                const auto expected = field->local_value(cell);
                EXPECT_NEAR(actual->local_value(cell), expected, std::max(1e-20, std::abs(expected) * 1e-12)) << name;
            }
        }
        for (const auto& h : reduced.species_histories())
        {
            SCOPED_TRACE(h.species);
            if (h.species == GeneralGas::one || h.species == GeneralGas::two)
                EXPECT_NEAR(h.inventory_error, 0, 1e-12);
            else
            {
                // The historical absolute oracle concerns transported gas.
                // Finite solution stocks have no sources or outward carrier
                // flux here and retain their independently initialized amount.
                // Their total is about 5e4 mol, where one double ULP exceeds
                // the gas-only absolute gate; use floating-point equality for
                // this additional solvent check, with every event exactly zero.
                EXPECT_DOUBLE_EQ(h.current_moles, h.initial_moles);
                EXPECT_DOUBLE_EQ(h.chemical_production, 0.0);
                EXPECT_DOUBLE_EQ(h.chemical_consumption, 0.0);
                EXPECT_DOUBLE_EQ(h.external_supply, 0.0);
                EXPECT_DOUBLE_EQ(h.external_withdrawal, 0.0);
                EXPECT_DOUBLE_EQ(h.physical_outflow, 0.0);
            }
        }
        EXPECT_EQ(reduced.population_escape_receipt().events, full.population_escape_receipt().events);
    }
    void rollback(const ALEReactingModelType::StateSnapshot& a, const ALEReactingModelType::StateSnapshot& b)
    {
        motion.rollback_trial();
        full.restore(a);
        reduced.restore(b);
        full.refresh_geometry();
        reduced.refresh_geometry();
    }
};

TEST(ReactingGasModelTest, ALEZeroAuxiliaryMatchesFullAcrossMotionAndReplay)
{
    ALEZeroTransportPair state(zero_ale_transport_options());
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    for (const double top : {.8, 1.2, .8})
    {
        state.advance(top, .001, .001);
        state.compare();
        EXPECT_GT(state.full.last_statistics().transport_linear.solves, 0);
        EXPECT_EQ(state.full.last_statistics().transport_linear.solves,
            state.reduced.last_statistics().transport_linear.solves);
        EXPECT_DOUBLE_EQ(state.reduced.global_species_moles(GeneralGas::one), 0);
        EXPECT_DOUBLE_EQ(GeneralGas::integral(state.reduced.population_number_field("large")), 0);
        state.rollback(a, b);
    }
}

TEST(ReactingGasModelTest, ALEZeroAuxiliaryPreservesSharedMicroEscape)
{
    auto p = zero_ale_transport_options();
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-6;
    ALEZeroTransportPair state(p);
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    for (const double top : {.8, 1.2})
    {
        state.advance(top, .01, .01);
        state.compare();
        EXPECT_GT(GeneralGas::step_escape(state.reduced, GeneralGas::one, true), 0);
        EXPECT_GT(GeneralGas::number_escape(state.reduced, "micro"), 0);
        EXPECT_DOUBLE_EQ(GeneralGas::number_escape(state.reduced, "large"), 0);
        EXPECT_NEAR(GeneralGas::step_escape(state.reduced), GeneralGas::step_escape(state.full), 1e-20);
        state.rollback(a, b);
    }
}

TEST(ReactingGasModelTest, ALEZeroLargeNumberReassemblesPopulatedMoles)
{
    auto p = zero_ale_transport_options();
    p.reservoir_mol_per_joule[0] = 2e-7;
    p.nucleate_supplied_gas = true;
    p.nucleation_population = "large";
    ALEZeroTransportPair state(p);
    double time = 0;
    for (int step = 1; step <= 2; ++step)
    {
        if (step == 2)
            state.power.put_scalar(1000);
        const double dt = step * .001;
        time += dt;
        state.advance(step == 1 ? 1.2 : .8, time, dt);
        state.compare();
        if (step == 1)
            EXPECT_DOUBLE_EQ(
                GeneralGas::compartment(state.reduced, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), 0);
        else
        {
            EXPECT_GT(
                GeneralGas::compartment(state.reduced, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), 0);
            EXPECT_GT(GeneralGas::integral(state.reduced.population_number_field("large")), 0);
        }
        state.motion.accept_trial();
    }
    auto invalid = p;
    invalid.molar_one[1] = 1e-6;
    invalid.number_density[1] = 0;
    EXPECT_THROW(ALEZeroTransportPair malformed(invalid), std::invalid_argument);
}

TEST(ReactingGasModelTest, ALEZeroAuxiliaryRetainsMotionAndPostCallbackGCLGuards)
{
    auto p = zero_ale_transport_options();
    p.diffusivity_two = [](const auto& c) { return gas_diffusivity_with_gcl_fault(c.environment.temperature); };
    gas_gcl_fault_armed = false;
    gas_gcl_fault_motion = nullptr;
    ALEZeroTransportPair state(p);
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    constexpr double dt = .001;
    state.motion.begin_trial(1.2, dt);
    const auto old = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
    const auto advance = [&](const auto& ale, double step = .001)
    {
        state.reduced.advance(
            dt, step, state.temperature, state.pressure, state.velocity, state.flux, state.material, &ale);
    };
    EXPECT_THROW(advance(old, 2 * dt), std::invalid_argument);
    state.motion.rollback_trial();
    EXPECT_THROW(advance(old), std::logic_error);
    state.motion.begin_trial(1.1, dt);
    EXPECT_THROW(advance(old), std::invalid_argument);
    state.rollback(a, b);
    for (const bool after_callback : {false, true})
    {
        state.motion.begin_trial(1.2, dt);
        {
            MutableGasALEView altered(state.mesh, state.motion);
            const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, altered);
            state.reduced.refresh_geometry();
            gas_gcl_fault_motion = &altered;
            gas_gcl_fault_injected = false;
            if (after_callback)
                gas_gcl_fault_armed = true;
            else
                altered.invalidate_flux();
            std::string message;
            try
            {
                advance(ale);
            }
            catch (const std::invalid_argument& e)
            {
                message = e.what();
            }
            EXPECT_NE(message.find("geometric conservation law"), std::string::npos);
            if (after_callback)
                EXPECT_TRUE(gas_gcl_fault_injected);
            EXPECT_EQ(state.reduced.last_statistics().transport_linear.solves, 0);
            gas_gcl_fault_armed = false;
            gas_gcl_fault_motion = nullptr;
        }
        state.rollback(a, b);
    }
    state.advance(1.1, dt, dt);
    state.compare();
    state.rollback(a, b);
}

TEST(ReactingGasModelTest, ALEZeroAuxiliaryRetainsLiquidStorageValidation)
{
    ALEZeroTransportPair state(zero_ale_transport_options());
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    state.motion.begin_trial(1.1, .001);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
    std::array<std::string, 2> messages;
    int index = 0;
    for (auto* model : {&state.full, &state.reduced})
    {
        model->refresh_geometry();
        // Poison actual conserved liquid storage; derived alpha_l is correctly rebuilt from this stock.
        const_cast<ALEFieldType&>(model->species_field(GeneralGas::solvent, GeneralGas::Phase::Liquid))
            .put_scalar(std::numeric_limits<double>::quiet_NaN());
        try
        {
            model->advance(
                .001, .001, state.temperature, state.pressure, state.velocity, state.flux, state.material, &ale);
        }
        catch (const std::exception& error)
        {
            messages[index] = error.what();
        }
        ++index;
    }
    EXPECT_FALSE(messages[0].empty());
    EXPECT_EQ(messages[0], messages[1]);
    state.rollback(a, b);
    state.advance(1.1, .001, .001);
    state.compare();
    state.rollback(a, b);
}

} // namespace
