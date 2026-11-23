/** @file testLowMachMultiRhsTransport.cc
 * @brief Phase 2 regressions for carrier-grouped MultiVector transport and batched conservation repair.
 */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;

constexpr double gas_constant = 8.31446261815324;

testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<Flow::Mesh> make_mesh_1d(ArrReal z_nodes)
{
    Flow::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Flow::Mesh>(
        std::make_shared<Meshes::OrthogonalCartesian3D>(
            Vec3D<ArrReal>{ArrReal{0.0, 1.0}, ArrReal{0.0, 1.0}, std::move(z_nodes)}),
        distribution);
}

void expect_state_and_snapshot_near(
    const Flow::Trial& expected, const Flow::Trial& actual, double tol)
{
    double local_max_mech_pressure = 0.0;
    for (const auto& cell : expected.state().cells)
    {
        local_max_mech_pressure = std::max(local_max_mech_pressure, std::abs(cell.mechanical_pressure));
    }
    double global_max_mech_pressure = 0.0;
    Teuchos::reduceAll(
        *Teuchos::DefaultComm<int>::getComm(), Teuchos::REDUCE_MAX, 1, &local_max_mech_pressure,
        &global_max_mech_pressure);
    const double mech_pressure_scale = std::max(1.0, global_max_mech_pressure);

    EXPECT_NEAR(actual.state().thermodynamic_pressure, expected.state().thermodynamic_pressure,
        tol * std::max(1.0, std::abs(expected.state().thermodynamic_pressure)));
    ASSERT_EQ(actual.state().cells.size(), expected.state().cells.size());
    for (std::size_t c = 0; c < expected.state().cells.size(); ++c)
    {
        const auto& ec = expected.state().cells[c];
        const auto& ac = actual.state().cells[c];
        // Mechanical pressure is a gauge-pinned elliptic perturbation field; scale its tolerance by the
        // domain L_inf mechanical pressure rather than pointwise magnitude near the zero-pressure gauge cell.
        EXPECT_NEAR(ac.mechanical_pressure, ec.mechanical_pressure, tol * mech_pressure_scale);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(ac.velocity.component(axis), ec.velocity.component(axis), tol);
        }
        ASSERT_EQ(ac.phases.size(), ec.phases.size());
        for (std::size_t a = 0; a < ec.phases.size(); ++a)
        {
            const auto& ep = ec.phases[a];
            const auto& ap = ac.phases[a];
            EXPECT_NEAR(ap.internal_energy, ep.internal_energy,
                tol * std::max(1.0, std::abs(ep.internal_energy)));
            EXPECT_NEAR(ap.temperature, ep.temperature, tol * std::max(1.0, std::abs(ep.temperature)));
            ASSERT_EQ(ap.moles.size(), ep.moles.size());
            for (std::size_t s = 0; s < ep.moles.size(); ++s)
            {
                EXPECT_NEAR(ap.moles[s], ep.moles[s], tol * std::max(1.0, std::abs(ep.moles[s])));
            }
            if (ec.phase_numbers[a].has_value())
            {
                ASSERT_TRUE(ac.phase_numbers[a].has_value());
                EXPECT_NEAR(*ac.phase_numbers[a], *ec.phase_numbers[a],
                    tol * std::max(1.0, std::abs(*ec.phase_numbers[a])));
            }
            else
            {
                EXPECT_FALSE(ac.phase_numbers[a].has_value());
            }
        }
    }

    const auto& esnap = expected.transport_snapshot();
    const auto& asnap = actual.transport_snapshot();
    ASSERT_EQ(asnap.cells.size(), esnap.cells.size());
    for (std::size_t c = 0; c < esnap.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(asnap.old_control_volumes[c], esnap.old_control_volumes[c]);
        EXPECT_DOUBLE_EQ(asnap.new_control_volumes[c], esnap.new_control_volumes[c]);
        for (std::size_t a = 0; a < esnap.cells[c].phases.size(); ++a)
        {
            const auto& ei = esnap.cells[c].phases[a];
            const auto& ai = asnap.cells[c].phases[a];
            EXPECT_NEAR(ai.internal_energy, ei.internal_energy,
                tol * std::max(1.0, std::abs(ei.internal_energy)));
            EXPECT_NEAR(ai.advected_material_volume, ei.advected_material_volume, tol);
            ASSERT_EQ(ai.moles.size(), ei.moles.size());
            for (std::size_t s = 0; s < ei.moles.size(); ++s)
            {
                EXPECT_NEAR(ai.moles[s], ei.moles[s], tol * std::max(1.0, std::abs(ei.moles[s])));
            }
            if (ei.population_number.has_value())
            {
                ASSERT_TRUE(ai.population_number.has_value());
                EXPECT_NEAR(*ai.population_number, *ei.population_number,
                    tol * std::max(1.0, std::abs(*ei.population_number)));
            }
        }
    }

    ASSERT_EQ(actual.owned_volume_flux().size(), expected.owned_volume_flux().size());
    for (std::size_t f = 0; f < expected.owned_volume_flux().size(); ++f)
    {
        EXPECT_NEAR(actual.owned_volume_flux()[f], expected.owned_volume_flux()[f], tol);
    }
}

struct MultiCarrierFixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid;
    std::shared_ptr<const R::MixtureThermodynamics> gas;
    Flow::Options options;
    Flow::State initial;

    explicit MultiCarrierFixture(
        ArrReal z_nodes = {0.0, 0.25, 0.5, 0.75, 1.0},
        double energy_reference_shift = 1.0e6)
        : mesh(make_mesh_1d(std::move(z_nodes)))
    {
        const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Resolved, R::SpeciesPhase::Dispersed};
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {101, "vapor", 0.02, {{"A", 1}}, roles},
            {202, "inert", 0.04, {{"B", 1}}, roles}});
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000},
            [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
        gas = R::make_ideal_gas_mixture(
            catalog,
            {{20, 300, 6000 + energy_reference_shift}, {20, 300, 6000 + energy_reference_shift}},
            {250, 600, 10000, 2000000});

        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.tolerance = 1e-12;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
        options.phase_roles = {
            {Flow::PhaseRole::BulkLiquid, "liquid"},
            {Flow::PhaseRole::BulkGas, "bulk_gas"},
            {Flow::PhaseRole::DispersedGas, "micro"}};

        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(mesh->num_owned_cells());
        for (std::size_t c = 0; c < initial.cells.size(); ++c)
        {
            const double vol = mesh->cell_volume(static_cast<LO>(c));
            const double z = mesh->cell_centroid(static_cast<LO>(c)).z;
            const double liquid_mass = 0.7 * vol * 1000.0;
            const double bulk_moles = 100000 * 0.15 * vol / (gas_constant * 300.0);
            const double micro_moles = 100000 * 0.15 * vol / (gas_constant * 300.0);
            const double frac = z < 0.5 ? 0.75 : 0.25;
            initial.cells[c].phases = {
                {liquid, {liquid_mass / 0.02, 0.0}, 100000, liquid_mass * 1e6, 300.0},
                {gas, {frac * bulk_moles, (1.0 - frac) * bulk_moles}, 100000,
                    (6000.0 + energy_reference_shift) * bulk_moles, 300.0},
                {gas, {(1.0 - frac) * micro_moles, frac * micro_moles}, 100000,
                    (6000.0 + energy_reference_shift) * micro_moles, 300.0}};
            initial.cells[c].phase_numbers = {std::nullopt, std::nullopt, 25.0 * micro_moles};
            const bool internal_cell = z > 0.25 && z < 0.75;
            initial.cells[c].drift_velocity = {
                Flow::Vector{0, 0, 0.0},
                Flow::Vector{0, 0, internal_cell ? 0.08 : 0.0},
                Flow::Vector{0, 0, internal_cell ? -0.06 : 0.0}};
        }
    }
};

TEST(LowMachMultiRhsTransportTest, RejectsCarrierIdentitiesThatCollideAfterDefaults)
{
    MultiCarrierFixture f;

    // A generated default must not alias an explicit identity from another phase.
    auto liquid_defaults = f.options;
    liquid_defaults.phase_roles = {
        {Flow::PhaseRole::BulkLiquid, ""},
        {Flow::PhaseRole::BulkLiquid, "liquid_0"}};
    EXPECT_THROW(
        (Flow(f.mesh, {f.liquid, f.liquid}, liquid_defaults)), std::invalid_argument);

    auto gas_defaults = f.options;
    gas_defaults.phase_roles = {
        {Flow::PhaseRole::BulkGas, ""},
        {Flow::PhaseRole::BulkGas, "bulk_gas_0"}};
    EXPECT_THROW(
        (Flow(f.mesh, {f.gas, f.gas}, gas_defaults)), std::invalid_argument);

    // Distinct generated defaults remain valid carrier identities.
    auto distinct_defaults = f.options;
    distinct_defaults.phase_roles = {
        {Flow::PhaseRole::BulkLiquid, ""},
        {Flow::PhaseRole::BulkGas, ""}};
    EXPECT_NO_THROW((Flow(f.mesh, {f.liquid, f.gas}, distinct_defaults)));
}

TEST(LowMachMultiRhsTransportTest, ScalarVsMultiRhsExactParityFixedAndALE)
{
    for (const double shift : {0.0, 1.0e6})
    {
        MultiCarrierFixture f({0.0, 0.25, 0.5, 0.75, 1.0}, shift);
        std::vector<Flow::CellSources> sources(f.initial.cells.size());
        for (std::size_t c = 0; c < sources.size(); ++c)
        {
            sources[c].species_moles = {{0.02, 0.0}, {-0.005, 0.0}, {0.0, 0.002}};
            sources[c].internal_energy = {500.0, -50.0, 25.0};
            sources[c].population_number_changes = {0.0, 0.0, 0.05};
        }

        auto scalar_opts = f.options;
        scalar_opts.use_multi_rhs_transport = false;
        auto multi_opts = f.options;
        multi_opts.use_multi_rhs_transport = true;

        // Fixed grid comparison.
        Flow scalar_solver(f.mesh, {f.liquid, f.gas, f.gas}, scalar_opts);
        Flow multi_solver(f.mesh, {f.liquid, f.gas, f.gas}, multi_opts);
        scalar_solver.initialize(f.initial);
        multi_solver.initialize(f.initial);
        const auto scalar_fixed = scalar_solver.preview(0.05, sources);
        const auto multi_fixed = multi_solver.preview(0.05, sources);
        expect_state_and_snapshot_near(scalar_fixed, multi_fixed, 1e-10);
        EXPECT_EQ(multi_fixed.diagnostics().transport_multi_rhs.operator_groups, 3U);

        // Moving planar ALE comparison.
        auto mesh_s = make_mesh_1d({0.0, 0.25, 0.5, 0.75, 1.0});
        auto mesh_m = make_mesh_1d({0.0, 0.25, 0.5, 0.75, 1.0});
        Flow ale_scalar(mesh_s, {f.liquid, f.gas, f.gas}, scalar_opts);
        Flow ale_multi(mesh_m, {f.liquid, f.gas, f.gas}, multi_opts);
        ale_scalar.initialize(f.initial);
        ale_multi.initialize(f.initial);
        Flow::PlanarMotion motion_s(mesh_s);
        Flow::PlanarMotion motion_m(mesh_m);
        const auto scalar_ale = ale_scalar.preview_ale(0.05, motion_s, 1.02, sources);
        const auto multi_ale = ale_multi.preview_ale(0.05, motion_m, 1.02, sources);
        expect_state_and_snapshot_near(scalar_ale, multi_ale, 1e-10);
        EXPECT_EQ(multi_ale.diagnostics().transport_multi_rhs.operator_groups, 3U);
    }
}

TEST(LowMachMultiRhsTransportTest, CarrierOperatorGroupingCountBatchProofAndBiCGStabMode)
{
    auto mesh = make_mesh_1d({0.0, 0.25, 0.5, 0.75, 1.0});
    const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Resolved, R::SpeciesPhase::Dispersed};
    auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {101, "s0", 0.02, {{"A", 1}}, roles},
        {202, "s1", 0.03, {{"B", 1}}, roles},
        {303, "s2", 0.04, {{"C", 1}}, roles}});
    auto liquid = R::make_linear_compressible_mixture_liquid(
        catalog, {250, 600, 10000, 2000000},
        [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
    auto gas = R::make_ideal_gas_mixture(
        catalog, {{20, 300, 6000}, {20, 300, 6000}, {20, 300, 6000}}, {250, 600, 10000, 2000000});

    // 1 BulkLiquid + 2 BulkGas + 2 DispersedGas = 5 carriers, 22 columns (4+4+4+5+5).
    Flow::Options opts;
    opts.minimum_pressure = 20000;
    opts.maximum_pressure = 1000000;
    opts.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
    opts.linear.backend = LinearSolverBackend::Gmres;
    opts.linear.preconditioner = LinearPreconditioner::SymmetricGaussSeidel;
    opts.linear.tolerance = 1e-12;
    opts.linear.max_batch_width = 0;
    opts.phase_roles = {
        {Flow::PhaseRole::BulkLiquid, "liq"},
        {Flow::PhaseRole::BulkGas, "gas_a"},
        {Flow::PhaseRole::BulkGas, "gas_b"},
        {Flow::PhaseRole::DispersedGas, "micro"},
        {Flow::PhaseRole::DispersedGas, "large"}};

    Flow::State init;
    init.thermodynamic_pressure = 100000;
    init.cells.resize(mesh->num_owned_cells());
    for (std::size_t c = 0; c < init.cells.size(); ++c)
    {
        const double vol = mesh->cell_volume(static_cast<LO>(c));
        const double z = mesh->cell_centroid(static_cast<LO>(c)).z;
        const double lm = 0.6 * vol * 1000.0;
        const double g_moles = 100000 * 0.1 * vol / (gas_constant * 300.0);
        const double f = z < 0.5 ? 0.6 : 0.3;
        init.cells[c].phases = {
            {liquid, {lm / 0.02, 0.0, 0.0}, 100000, lm * 1e6, 300.0},
            {gas, {f * g_moles, (1.0 - f) * 0.5 * g_moles, (1.0 - f) * 0.5 * g_moles}, 100000, 6000 * g_moles, 300.0},
            {gas, {(1.0 - f) * g_moles, 0.5 * f * g_moles, 0.5 * f * g_moles}, 100000, 6000 * g_moles, 300.0},
            {gas, {0.5 * g_moles, 0.3 * g_moles, 0.2 * g_moles}, 100000, 6000 * g_moles, 300.0},
            {gas, {0.2 * g_moles, 0.5 * g_moles, 0.3 * g_moles}, 100000, 6000 * g_moles, 300.0}};
        init.cells[c].phase_numbers = {
            std::nullopt, std::nullopt, std::nullopt, 15.0 * g_moles, 30.0 * g_moles};
        init.cells[c].drift_velocity = {
            Flow::Vector{0, 0, 0.0},
            Flow::Vector{0, 0, 0.04},
            Flow::Vector{0, 0, 0.09},
            Flow::Vector{0, 0, -0.05},
            Flow::Vector{0, 0, -0.11}};
    }

    Flow gmres_solver(mesh, {liquid, gas, gas, gas, gas}, opts);
    gmres_solver.initialize(init);
    std::vector<Flow::CellSources> no_sources(init.cells.size());
    for (auto& cell_src : no_sources)
    {
        cell_src.internal_energy = {400.0, 10.0, -5.0, 5.0, 0.0};
    }
    const auto gmres_trial = gmres_solver.preview(0.05, no_sources);
    const auto& gmres_diag = gmres_trial.diagnostics().transport_multi_rhs;

    EXPECT_EQ(gmres_diag.operator_groups, 5U);
    EXPECT_EQ(gmres_diag.matrix_assemblies, 5U);
    EXPECT_EQ(gmres_diag.preconditioner_builds, 5U);
    EXPECT_EQ(gmres_diag.transported_columns, 22U);
    EXPECT_EQ(gmres_diag.batched_krylov_solves, 5U);
    EXPECT_EQ(gmres_diag.column_serial_solves, 0U);
    EXPECT_EQ(gmres_diag.execution_mode, MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(gmres_diag.max_batch_width, 5U);
    EXPECT_LT(gmres_diag.operator_apply_calls, gmres_diag.operator_column_applies);
    EXPECT_GE(gmres_diag.halo_cell_imports, 5U);
    EXPECT_GE(gmres_diag.halo_face_imports, 5U);
    EXPECT_EQ(gmres_diag.column_consistency_sweeps.size(), 22U);
    EXPECT_GE(gmres_diag.matrix_assembly_seconds, 0.0);
    EXPECT_GE(gmres_diag.preconditioner_setup_seconds, 0.0);
    EXPECT_GE(gmres_diag.linear_solve_seconds, 0.0);
    EXPECT_GE(gmres_diag.pack_unpack_seconds, 0.0);
    EXPECT_GE(gmres_diag.halo_exchange_seconds, 0.0);
    EXPECT_GE(gmres_diag.consistency_repair_seconds, 0.0);

    // BiCGStab must report ColumnSerial honestly while giving identical physical results.
    auto bicg_opts = opts;
    bicg_opts.linear.backend = LinearSolverBackend::BiCGStab;
    bicg_opts.linear.preconditioner = LinearPreconditioner::None;
    Flow bicg_solver(mesh, {liquid, gas, gas, gas, gas}, bicg_opts);
    bicg_solver.initialize(init);
    const auto bicg_trial = bicg_solver.preview(0.05, no_sources);
    const auto& bicg_diag = bicg_trial.diagnostics().transport_multi_rhs;

    EXPECT_EQ(bicg_diag.operator_groups, 5U);
    EXPECT_EQ(bicg_diag.matrix_assemblies, 5U);
    EXPECT_EQ(bicg_diag.preconditioner_builds, 0U);
    EXPECT_EQ(bicg_diag.execution_mode, MultiRhsExecutionMode::ColumnSerial);
    EXPECT_EQ(bicg_diag.column_serial_solves, 5U);
    EXPECT_EQ(bicg_diag.batched_krylov_solves, 0U);
    EXPECT_LE(bicg_diag.max_batch_width, 1U);
    expect_state_and_snapshot_near(gmres_trial, bicg_trial, 1e-10);
}

TEST(LowMachMultiRhsTransportTest, MaxBatchWidthAndSpeciesPermutationInvariance)
{
    auto mesh = make_mesh_1d({0.0, 0.25, 0.5, 0.75, 1.0});
    const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Resolved, R::SpeciesPhase::Dispersed};
    const std::vector<R::SpeciesDefinition> base_defs{
        {101, "main", 0.02, {{"A", 1}}, roles},
        {202, "trace", 0.03, {{"B", 1}}, roles},
        {303, "dup", 0.02, {{"C", 1}}, roles},
        {404, "zero", 0.04, {{"D", 1}}, roles}};

    auto make_problem = [&](const std::vector<std::size_t>& perm, int max_batch_width)
    {
        std::vector<R::SpeciesDefinition> defs;
        defs.reserve(4);
        for (const std::size_t p : perm)
        {
            defs.push_back(base_defs[p]);
        }
        auto catalog = std::make_shared<const R::SpeciesCatalog>(defs);
        auto liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000},
            [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 0.0, 0, 0}; });
        auto gas = R::make_ideal_gas_mixture(
            catalog,
            {{20, 300, 0.0}, {20, 300, 0.0}, {20, 300, 0.0}, {20, 300, 0.0}},
            {250, 600, 10000, 2000000});

        Flow::Options opts;
        opts.minimum_pressure = 20000;
        opts.maximum_pressure = 1000000;
        opts.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
        opts.linear.backend = LinearSolverBackend::Gmres;
        opts.linear.tolerance = 1e-12;
        opts.linear.max_batch_width = max_batch_width;
        opts.phase_roles = {
            {Flow::PhaseRole::BulkLiquid, "liq"},
            {Flow::PhaseRole::BulkGas, "gas"},
            {Flow::PhaseRole::DispersedGas, "micro"}};

        Flow::State init;
        init.thermodynamic_pressure = 100000;
        init.cells.resize(mesh->num_owned_cells());
        for (std::size_t c = 0; c < init.cells.size(); ++c)
        {
            const double vol = mesh->cell_volume(static_cast<LO>(c));
            const double z = mesh->cell_centroid(static_cast<LO>(c)).z;
            const double gas_T = z < 0.5 ? 280.0 : 320.0; // Asymmetric T -> signed zero-sum internal energy!
            const double base_gas = 100000 * 0.14 * vol / (gas_constant * 300.0);
            const double gas_vol_each = base_gas * (1.0 + 1.0e-12) * gas_constant * gas_T / 100000.0;
            const double lm = (vol - 2.0 * gas_vol_each) * 1000.0;
            const double liq_trace_mol = 1.0e-12;
            const double liq_rem_mol = (lm - liq_trace_mol * 0.03) / 0.02;
            const std::array<double, 4> liq_base{0.5 * liq_rem_mol, liq_trace_mol, 0.5 * liq_rem_mol, 0.0};
            const std::array<double, 4> gas_base{
                0.5 * base_gas, 1.0e-12 * base_gas, 0.5 * base_gas, 0.0};
            std::vector<double> l_moles(4), b_moles(4), d_moles(4);
            for (std::size_t k = 0; k < 4; ++k)
            {
                l_moles[k] = liq_base[perm[k]];
                b_moles[k] = gas_base[perm[k]];
                d_moles[k] = gas_base[perm[k]];
            }
            const double g_total = base_gas * (1.0 + 1.0e-12);
            const double signed_energy = g_total * 20.0 * (gas_T - 300.0);
            const bool internal_cell = z > 0.25 && z < 0.75;
            init.cells[c].phases = {
                {liquid, std::move(l_moles), 100000, 0.0, 300.0},
                {gas, std::move(b_moles), 100000, signed_energy, gas_T},
                {gas, std::move(d_moles), 100000, signed_energy, gas_T}};
            init.cells[c].phase_numbers = {std::nullopt, std::nullopt, 10.0 * base_gas};
            init.cells[c].drift_velocity = {
                Flow::Vector{0, 0, 0.0},
                Flow::Vector{0, 0, internal_cell ? 0.07 : 0.0},
                Flow::Vector{0, 0, internal_cell ? -0.05 : 0.0}};
        }
        auto solver = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas}, opts);
        solver->initialize(init);
        return solver;
    };

    const std::vector<std::size_t> id_perm{0, 1, 2, 3};
    const auto base_solver = make_problem(id_perm, 0);
    const auto base_trial = base_solver->preview(0.04);

    // Zero-sum signed InternalEnergy columns (BulkGas & DispersedGas) must be active
    // (Liquid reference energy at 300 K and the 3 exact-zero species columns are 0 -> 12 active of 16).
    EXPECT_EQ(base_trial.diagnostics().transport_multi_rhs.transported_columns, 16U);
    EXPECT_EQ(base_trial.diagnostics().transport_multi_rhs.active_columns, 12U);

    for (const int bw : {1, 2, 4})
    {
        const auto bw_solver = make_problem(id_perm, bw);
        const auto bw_trial = bw_solver->preview(0.04);
        expect_state_and_snapshot_near(base_trial, bw_trial, 1e-12);
    }

    // Reverse species order [3, 2, 1, 0] and map back by permutation.
    const std::vector<std::size_t> rev_perm{3, 2, 1, 0};
    const auto rev_solver = make_problem(rev_perm, 0);
    const auto rev_trial = rev_solver->preview(0.04);
    EXPECT_NEAR(rev_trial.state().thermodynamic_pressure, base_trial.state().thermodynamic_pressure, 1e-8);
    ASSERT_EQ(rev_trial.state().cells.size(), base_trial.state().cells.size());
    for (std::size_t c = 0; c < base_trial.state().cells.size(); ++c)
    {
        for (std::size_t a = 0; a < 3; ++a)
        {
            const auto& ep = base_trial.state().cells[c].phases[a];
            const auto& ap = rev_trial.state().cells[c].phases[a];
            EXPECT_NEAR(ap.internal_energy, ep.internal_energy, 1e-11 * std::max(1.0, std::abs(ep.internal_energy)));
            EXPECT_NEAR(ap.temperature, ep.temperature, 1e-11);
            for (std::size_t k = 0; k < 4; ++k)
            {
                EXPECT_NEAR(ap.moles[k], ep.moles[rev_perm[k]], 1e-11 * std::max(1.0, std::abs(ep.moles[rev_perm[k]])));
            }
        }
        if (base_trial.state().cells[c].phase_numbers[2].has_value())
        {
            EXPECT_NEAR(*rev_trial.state().cells[c].phase_numbers[2],
                *base_trial.state().cells[c].phase_numbers[2], 1e-11);
        }
    }
}

TEST(LowMachMultiRhsTransportTest, VanishedPopulationAndSourceNucleationInvariance)
{
    MultiCarrierFixture f({0.0, 0.25, 0.5, 0.75, 1.0});
    f.options.linear.preconditioner = LinearPreconditioner::SymmetricGaussSeidel;

    // Zero out DispersedGas completely everywhere before preview; BulkGas fills the full 0.30 gas volume.
    for (std::size_t c = 0; c < f.initial.cells.size(); ++c)
    {
        auto& cell = f.initial.cells[c];
        const double vol = f.mesh->cell_volume(static_cast<LO>(c));
        const double z = f.mesh->cell_centroid(static_cast<LO>(c)).z;
        const double frac = z < 0.5 ? 0.75 : 0.25;
        const double bulk_moles = 100000 * 0.30 * vol / (gas_constant * 300.0);
        cell.phases[1].moles = {frac * bulk_moles, (1.0 - frac) * bulk_moles};
        cell.phases[1].internal_energy = (6000.0 + 1.0e6) * bulk_moles;
        cell.phases[2].moles = {0.0, 0.0};
        cell.phases[2].internal_energy = 0.0;
        cell.phases[2].temperature = 0.0;
        cell.phase_numbers[2] = 0.0;
    }

    Flow solver(f.mesh, {f.liquid, f.gas, f.gas}, f.options);
    solver.initialize(f.initial);
    const auto empty_trial = solver.preview(0.005);
    // Empty DispersedGas skips Krylov cleanly -> preconditioner builds only for active Liquid + BulkGas.
    EXPECT_EQ(empty_trial.diagnostics().transport_multi_rhs.operator_groups, 3U);
    EXPECT_EQ(empty_trial.diagnostics().transport_multi_rhs.preconditioner_builds, 2U);
    for (const auto& cell : empty_trial.state().cells)
    {
        EXPECT_DOUBLE_EQ(cell.phases[2].moles[0], 0.0);
        EXPECT_DOUBLE_EQ(cell.phases[2].moles[1], 0.0);
        EXPECT_DOUBLE_EQ(cell.phases[2].internal_energy, 0.0);
        EXPECT_DOUBLE_EQ(*cell.phase_numbers[2], 0.0);
    }

    // Now nucleate fresh dispersed population via SourceStageProvider: verify scalar vs multi-RHS match.
    const auto fingerprint = f.catalog->fingerprint();
    auto provider = std::make_shared<Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "nucleation", 1, fingerprint,
        [](const Flow::SourceStageContext& ctx)
        {
            Flow::SourceStageResult res;
            res.sources.resize(ctx.transport.cells.size());
            for (std::size_t c = 0; c < ctx.transport.cells.size(); ++c)
            {
                const double n_nuc = 0.005;
                const double molar_enthalpy = 6000.0 + 1.0e6 + gas_constant * 300.0;
                res.sources[c].species_moles = {{0.0, 0.0}, {-n_nuc, 0.0}, {n_nuc, 0.0}};
                res.sources[c].internal_energy = {0.0, -molar_enthalpy * n_nuc, molar_enthalpy * n_nuc};
                res.sources[c].population_number_changes = {0.0, 0.0, 5.0 * n_nuc};
            }
            return res;
        },
        0.0, 0});

    auto scalar_opts = f.options;
    scalar_opts.use_multi_rhs_transport = false;
    Flow scalar_solver(f.mesh, {f.liquid, f.gas, f.gas}, scalar_opts);
    scalar_solver.initialize(f.initial);
    const auto scalar_nuc = scalar_solver.preview_with_sources(0.005, provider);
    const auto multi_nuc = solver.preview_with_sources(0.005, provider);
    expect_state_and_snapshot_near(scalar_nuc, multi_nuc, 1e-10);
    for (const auto& cell : multi_nuc.state().cells)
    {
        EXPECT_GT(cell.phases[2].moles[0], 0.0);
        EXPECT_GT(*cell.phase_numbers[2], 0.0);
        EXPECT_GT(cell.phases[2].temperature, 0.0);
    }
    EXPECT_NO_THROW(solver.commit(multi_nuc));
}

TEST(LowMachMultiRhsTransportTest, ShiftedEnergyReferenceAndUlpPerturbationInvariance)
{
    auto run_nucleation = [](double shift, int top_cell_ulps)
    {
        MultiCarrierFixture f({0.0, 0.25, 0.5, 0.75, 1.0}, shift);
        f.options.linear.preconditioner = LinearPreconditioner::SymmetricGaussSeidel;
        f.options.linear.tolerance = 1e-14;
        for (std::size_t c = 0; c < f.initial.cells.size(); ++c)
        {
            auto& cell = f.initial.cells[c];
            const double vol = f.mesh->cell_volume(static_cast<LO>(c));
            const double z = f.mesh->cell_centroid(static_cast<LO>(c)).z;
            const double frac = z < 0.5 ? 0.75 : 0.25;
            const double bulk_moles = 100000 * 0.30 * vol / (gas_constant * 300.0);
            cell.phases[1].moles = {frac * bulk_moles, (1.0 - frac) * bulk_moles};
            cell.phases[1].internal_energy = (6000.0 + shift) * bulk_moles;
            cell.phases[2].moles = {0.0, 0.0};
            cell.phases[2].internal_energy = 0.0;
            cell.phases[2].temperature = 0.0;
            cell.phase_numbers[2] = 0.0;
        }
        Flow solver(f.mesh, {f.liquid, f.gas, f.gas}, f.options);
        solver.initialize(f.initial);

        const auto mesh = f.mesh;
        const auto fingerprint = f.catalog->fingerprint();
        auto provider = std::make_shared<Flow::SourceStageProvider>(Flow::SourceStageProvider{
            "nucleation", 1, fingerprint,
            [mesh, shift, top_cell_ulps](const Flow::SourceStageContext& ctx)
            {
                Flow::SourceStageResult res;
                res.sources.resize(ctx.transport.cells.size());
                for (std::size_t c = 0; c < ctx.transport.cells.size(); ++c)
                {
                    const double n_nuc = 0.005;
                    const double molar_enthalpy = 6000.0 + shift + gas_constant * 300.0;
                    double bulk_e_src = -molar_enthalpy * n_nuc;
                    if (top_cell_ulps != 0 && mesh->cell_centroid(static_cast<LO>(c)).z > 0.75)
                    {
                        const double scale_e = ctx.transport.cells[c].phases[1].internal_energy;
                        const double ulp =
                            std::nextafter(scale_e, std::numeric_limits<double>::infinity()) - scale_e;
                        bulk_e_src += static_cast<double>(top_cell_ulps) * ulp;
                    }
                    res.sources[c].species_moles = {{0.0, 0.0}, {-n_nuc, 0.0}, {n_nuc, 0.0}};
                    res.sources[c].internal_energy = {0.0, bulk_e_src, molar_enthalpy * n_nuc};
                    res.sources[c].population_number_changes = {0.0, 0.0, 5.0 * n_nuc};
                }
                return res;
            },
            0.0, 0});
        return solver.preview_with_sources(0.005, provider);
    };

    // Caloric inversion must remain invariant to a 1e6 J/mol energy reference shift without
    // premature 32*eps*scale bisection termination (which previously caused >1e-7 Pa drift).
    const auto shift0 = run_nucleation(0.0, 0);
    const auto shift1e6 = run_nucleation(1.0e6, 0);
    EXPECT_NEAR(shift1e6.state().thermodynamic_pressure, shift0.state().thermodynamic_pressure, 1.0e-7);
    ASSERT_EQ(shift1e6.state().cells.size(), shift0.state().cells.size());
    for (std::size_t c = 0; c < shift0.state().cells.size(); ++c)
    {
        EXPECT_NEAR(shift1e6.state().cells[c].mechanical_pressure, shift0.state().cells[c].mechanical_pressure,
            1.0e-8);
        EXPECT_NEAR(shift1e6.state().cells[c].phases[1].temperature, shift0.state().cells[c].phases[1].temperature,
            5.0e-11);
    }

    // A 1-ULP BulkGas energy perturbation under a 1e6 J/mol shift must not trigger a discrete
    // bisection-step jump in recovered temperature or mechanical pressure.
    for (const int ulps : {-1, 1})
    {
        SCOPED_TRACE(ulps);
        const auto perturbed = run_nucleation(1.0e6, ulps);
        ASSERT_EQ(perturbed.state().cells.size(), shift1e6.state().cells.size());
        for (std::size_t c = 0; c < shift1e6.state().cells.size(); ++c)
        {
            EXPECT_NEAR(perturbed.state().cells[c].mechanical_pressure,
                shift1e6.state().cells[c].mechanical_pressure, 1.2e-8);
            EXPECT_NEAR(perturbed.state().cells[c].phases[1].temperature,
                shift1e6.state().cells[c].phases[1].temperature, 3.0e-11);
        }
    }
}

TEST(LowMachMultiRhsTransportTest, MpiEmptyOwnerCrossPartitionAleAndRollback)
{
    // 1) Empty-owner rank test (1 cell on >= 2 MPI ranks).
    {
        MultiCarrierFixture empty_owner({0.0, 1.0});
        auto scalar_opts = empty_owner.options;
        scalar_opts.use_multi_rhs_transport = false;
        Flow s_solver(empty_owner.mesh, {empty_owner.liquid, empty_owner.gas, empty_owner.gas}, scalar_opts);
        Flow m_solver(empty_owner.mesh, {empty_owner.liquid, empty_owner.gas, empty_owner.gas}, empty_owner.options);
        s_solver.initialize(empty_owner.initial);
        m_solver.initialize(empty_owner.initial);
        expect_state_and_snapshot_near(s_solver.preview(0.05), m_solver.preview(0.05), 1e-12);
    }

    // 2) Cross-partition ALE with rank-0-only trace species on distributed mesh.
    {
        MultiCarrierFixture ale_s({0.0, 0.25, 0.5, 0.75, 1.0});
        MultiCarrierFixture ale_m({0.0, 0.25, 0.5, 0.75, 1.0});
        for (std::size_t c = 0; c < ale_s.initial.cells.size(); ++c)
        {
            const double z = ale_s.mesh->cell_centroid(static_cast<LO>(c)).z;
            const double vol = ale_s.mesh->cell_volume(static_cast<LO>(c));
            const double micro_moles = 100000 * 0.15 * vol / (gas_constant * 300.0);
            const double trace = z < 0.25 ? 1.0e-10 : 0.0;
            ale_s.initial.cells[c].phases[2].moles = {micro_moles - trace, trace};
            ale_m.initial.cells[c].phases[2].moles = {micro_moles - trace, trace};
        }
        auto scalar_opts = ale_s.options;
        scalar_opts.use_multi_rhs_transport = false;
        Flow s_solver(ale_s.mesh, {ale_s.liquid, ale_s.gas, ale_s.gas}, scalar_opts);
        Flow m_solver(ale_m.mesh, {ale_m.liquid, ale_m.gas, ale_m.gas}, ale_m.options);
        s_solver.initialize(ale_s.initial);
        m_solver.initialize(ale_m.initial);
        Flow::PlanarMotion motion_s(ale_s.mesh);
        Flow::PlanarMotion motion_m(ale_m.mesh);
        const auto t_s = s_solver.preview_ale(0.05, motion_s, 1.03);
        const auto t_m = m_solver.preview_ale(0.05, motion_m, 1.03);
        expect_state_and_snapshot_near(t_s, t_m, 1e-10);
        // The generic helper floors at 1e-10, exactly the trace scale. Assert the
        // trace with a scale-relative tolerance so losing it entirely cannot pass.
        for (std::size_t c = 0; c < t_s.state().cells.size(); ++c)
        {
            const double expected_state = t_s.state().cells[c].phases[2].moles[1];
            const double actual_state = t_m.state().cells[c].phases[2].moles[1];
            EXPECT_NEAR(actual_state, expected_state, 1.0e-4 * std::abs(expected_state) + 1.0e-18);
            const double expected_snapshot = t_s.transport_snapshot().cells[c].phases[2].moles[1];
            const double actual_snapshot = t_m.transport_snapshot().cells[c].phases[2].moles[1];
            EXPECT_NEAR(actual_snapshot, expected_snapshot, 1.0e-4 * std::abs(expected_snapshot) + 1.0e-18);
        }
    }

    // 3) Single-column failure atomic rollback & checkpoint replay.
    {
        auto mesh = make_mesh_1d({0.0, 0.25, 0.5, 0.75, 1.0});
        const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Resolved};
        auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {10, "vapor", 0.018, {{"A", 1}}, roles}, {29, "air", 0.029, {{"B", 1}}, roles}});
        auto liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000},
            [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
        auto gas = R::make_ideal_gas_mixture(catalog, {{20, 300, 6000}, {20, 300, 6000}}, {250, 600, 10000, 2000000});

        Flow::Options tight;
        tight.minimum_pressure = 20000;
        tight.maximum_pressure = 1000000;
        tight.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
        tight.linear.tolerance = 1e-13;
        tight.linear.max_iterations = 1; // Forces failure when repair or Krylov exceeds 1 iteration.
        tight.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liq"}, {Flow::PhaseRole::BulkGas, "gas"}};

        Flow::State init;
        init.thermodynamic_pressure = 100000;
        init.cells.resize(mesh->num_owned_cells());
        for (std::size_t c = 0; c < init.cells.size(); ++c)
        {
            const double vol = mesh->cell_volume(static_cast<LO>(c));
            const double z = mesh->cell_centroid(static_cast<LO>(c)).z;
            const double alpha_g = z < 0.25 ? 0.2 : 0.0;
            const double lm = (1.0 - alpha_g) * vol * 1000.0;
            const double g = alpha_g > 0.0 ? 100000 * alpha_g * vol / (gas_constant * 300.0) : 0.0;
            init.cells[c].phases = {
                {liquid, {lm / 0.018, 0.0}, 100000, lm * 1e6, 300.0},
                {gas, {g, 0.0}, 100000, g * 6000.0, g > 0.0 ? 300.0 : 0.0}};
            init.cells[c].drift_velocity = {
                Flow::Vector{}, Flow::Vector{0, 0, z > 0.25 && z < 0.75 ? 0.2 : 0.0}};
        }

        Flow failing(mesh, {liquid, gas}, tight);
        failing.initialize(init);
        const auto cp = failing.checkpoint();
        const auto saved_state = failing.state();
        EXPECT_THROW((void) failing.preview(0.05), std::runtime_error);
        EXPECT_EQ(failing.state().revision, saved_state.revision);
        EXPECT_DOUBLE_EQ(failing.state().thermodynamic_pressure, saved_state.thermodynamic_pressure);
        failing.restore(cp);

        auto good = tight;
        good.linear.max_iterations = 200;
        Flow ref_solver(mesh, {liquid, gas}, good);
        ref_solver.initialize(init);
        const auto cp_ok = ref_solver.checkpoint();
        const auto ref_trial = ref_solver.preview(0.05);
        ref_solver.restore(cp_ok);
        const auto replay_same = ref_solver.preview(0.05);
        expect_state_and_snapshot_near(ref_trial, replay_same, 0.0);

        Flow replayed_solver(mesh, {liquid, gas}, good);
        replayed_solver.initialize(init);
        const auto replayed_trial = replayed_solver.preview(0.05);
        expect_state_and_snapshot_near(ref_trial, replayed_trial, 1e-13);
    }
}

} // namespace
