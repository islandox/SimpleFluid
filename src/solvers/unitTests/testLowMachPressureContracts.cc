/** @file testLowMachPressureContracts.cc
 * @brief Independent material-volume projection, anchoring and collective transaction contracts.
 */
#include "equations/LowMachPressureEquation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace
{
namespace S = SimpleFluid;
using Pressure = S::Reacting::LowMachPressureEquation;
using Mesh = Pressure::Mesh;
using LO = Pressure::Pack::local_ordinal_type;
using Vector = Pressure::VectorField::vec_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Published
{
    std::vector<double> pressure, carrier;
    std::vector<Vector> velocity;
};

struct Fixture
{
    std::shared_ptr<Mesh> grid;
    Pressure equation;
    Pressure::ScalarField rho, mechanical;
    Pressure::VectorField velocity;
    Pressure::FaceField theta, slip, predictor, corrected;
    std::vector<double> target, chi;
    int bad_rank = 0;
    bool gauge_lower = false;

    static std::shared_ptr<Mesh> mesh()
    {
        Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        // Two cells of volume 2 m3, unit face area, and center separation 2 m.
        return std::make_shared<Mesh>(std::make_shared<S::Meshes::OrthogonalCartesian3D>(S::Vec3D<S::ArrReal>{
                                          S::ArrReal{0., 1.}, S::ArrReal{0., 1.}, S::ArrReal{0., 2., 4.}}),
            distribution);
    }
    Fixture()
        : grid(mesh()), equation(grid), rho(grid, 2.0, "rho"), mechanical(grid, 0.0, "mechanical"),
          velocity(grid, Vector{}, "velocity"), theta(grid, 0.0, "theta"), slip(grid, 0.0, "slip_material"),
          predictor(grid, 0.0, "predictor_carrier"), corrected(grid, 7.0, "corrected_carrier"),
          target(grid->num_owned_cells()), chi(grid->num_owned_cells(), 0.0)
    {
        const auto comm = grid->owned_cell_map()->getComm();
        const int local_empty = grid->num_owned_cells() == 0 ? 1 : 0;
        int empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
        const int local_cells = static_cast<int>(grid->num_owned_cells());
        int global_cells = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
        EXPECT_EQ(global_cells, 2);
        const int empty_candidate = local_empty ? comm->getRank() : -1;
        int empty_rank = -1;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &empty_candidate, &empty_rank);
        bad_rank = empty_rank >= 0 ? empty_rank : comm->getSize() - 1;
        if (comm->getSize() == 4)
        {
            EXPECT_EQ(empty, 2);
            if (comm->getRank() == bad_rank)
                EXPECT_EQ(grid->num_owned_cells(), 0U);
        }
        // The gauge is selected by the algebraic map; physical expectations use its centroid.
        const auto gauge = grid->owned_cell_map()->getMinAllGlobalIndex();
        int local_gauge_lower = 0, local_gauge_count = 0, global_gauge_lower = 0, global_gauge_count = 0;
        for (std::size_t c = 0; c < target.size(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            EXPECT_DOUBLE_EQ(grid->cell_volume(cell), 2.);
            if (grid->cell_global_id(cell) == gauge)
            {
                local_gauge_lower = lower(cell) ? 1 : 0;
                local_gauge_count++;
            }
        }
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_gauge_lower, &global_gauge_lower);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_gauge_count, &global_gauge_count);
        EXPECT_EQ(global_gauge_count, 1);
        gauge_lower = global_gauge_lower != 0;
        rho.sync_ghosts();
        configure(.5, .2, .1);
        reset_outputs();
    }
    bool lower(LO cell) const { return grid->cell_centroid(cell).z < 2.; }
    double orientation(LO face) const
    {
        const auto owner = grid->owner_cell(face), neighbor = grid->neighbor_cell(face);
        EXPECT_GT(
            grid->face_area_vector(face).z * (grid->cell_centroid(neighbor).z - grid->cell_centroid(owner).z), 0.);
        return lower(owner) ? 1. : -1.;
    }
    S::LinearSolverOptions linear() const
    {
        S::LinearSolverOptions result;
        result.tolerance = 1e-13;
        result.verbosity = 0;
        result.preconditioner = S::LinearPreconditioner::None;
        return result;
    }
    void configure(double fraction, double slip_z, double carrier_z)
    {
        for (std::size_t c = 0; c < target.size(); ++c)
            target[c] = lower(static_cast<LO>(c)) ? 1. : -1.;
        for (const auto face : theta.owned_face_ids())
        {
            const bool interior = !grid->is_geometry_exterior_face(face);
            const double sign = interior ? orientation(face) : 0.;
            theta.set_owned_value(face, interior ? fraction : 0.);
            slip.set_owned_value(face, slip_z * sign);
            predictor.set_owned_value(face, carrier_z * sign);
            if (interior)
            {
                EXPECT_DOUBLE_EQ(grid->face_area(face), 1.);
                EXPECT_DOUBLE_EQ(grid->face_cell_center_distance(face), 2.);
            }
        }
        theta.sync_ghosts();
        slip.sync_ghosts();
        predictor.sync_ghosts();
    }
    void reset_outputs(bool sentinels = false)
    {
        for (std::size_t c = 0; c < target.size(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            mechanical.set_owned_value(cell, sentinels ? 11. : 0.);
            velocity.set_owned_value(cell, sentinels ? Vector{7., 8., 9.} : Vector{});
        }
        corrected.owned_data().putScalar(sentinels ? 13. : 7.);
        mechanical.sync_ghosts();
        velocity.sync_ghosts();
        corrected.sync_ghosts();
    }
    Published snapshot() const
    {
        Published result;
        for (std::size_t c = 0; c < target.size(); ++c)
        {
            result.pressure.push_back(mechanical.value(static_cast<LO>(c)));
            result.velocity.push_back(velocity.value(static_cast<LO>(c)));
        }
        for (const auto face : corrected.owned_face_ids())
            result.carrier.push_back(corrected.value(face));
        return result;
    }
    void unchanged(const Published& saved) const
    {
        for (std::size_t c = 0; c < target.size(); ++c)
        {
            EXPECT_DOUBLE_EQ(mechanical.value(static_cast<LO>(c)), saved.pressure[c]);
            for (std::size_t axis = 0; axis < 3; ++axis)
                EXPECT_DOUBLE_EQ(velocity.value(static_cast<LO>(c)).component(axis), saved.velocity[c].component(axis));
        }
        std::size_t row = 0;
        for (const auto face : corrected.owned_face_ids())
            EXPECT_DOUBLE_EQ(corrected.value(face), saved.carrier[row++]);
    }
    Pressure::Result project(const S::LinearSolverOptions& controls)
    {
        return equation.project_material_with_eos(
            rho, 1., target, chi, theta, slip, predictor, corrected, mechanical, velocity, controls, 1e-11);
    }
    void nonunit_oracle(const Pressure::Result& result, double fraction) const
    {
        // Independently: q_material=1 m3/s, q_carrier=(1-.2)/theta.
        // Bare pressure mobility is .25 m3/(Pa*s), so p_lower-p_upper=4*(q_carrier-.1).
        const double carrier = fraction == .5 ? 1.6 : .4;
        const double pressure_difference = fraction == .5 ? 6. : 1.2;
        EXPECT_TRUE(result.linear.converged);
        EXPECT_NEAR(result.thermodynamic_pressure_increment, 0., 1e-12);
        EXPECT_LE(result.maximum_continuity_residual, 1e-11);
        EXPECT_LE(result.maximum_linearized_continuity_residual, 1e-11);
        const double gauge_value = gauge_lower ? pressure_difference : 0.;
        for (std::size_t c = 0; c < target.size(); ++c)
            EXPECT_NEAR(mechanical.value(static_cast<LO>(c)),
                (lower(static_cast<LO>(c)) ? pressure_difference : 0.) - gauge_value, 2e-11);
        for (const auto face : corrected.owned_face_ids())
        {
            if (grid->is_geometry_exterior_face(face))
                EXPECT_DOUBLE_EQ(corrected.value(face), 0.);
            else
            {
                const double sign = orientation(face);
                EXPECT_NEAR(corrected.value(face), sign * carrier, 2e-12);
                EXPECT_NEAR(fraction * corrected.value(face) + slip.value(face), sign, 2e-12);
            }
        }
    }
    void zero_theta_oracle(const Pressure::Result& result) const
    {
        // Each isolated row solves chi_i*dp_i/dt=target_i: dp_lower=2 Pa, dp_upper=-4 Pa.
        // The material flux cannot respond, but bare carrier mobility yields .1+.25*(2-(-4))=1.6.
        const double gauge_increment = gauge_lower ? 2. : -4.;
        EXPECT_TRUE(result.linear.converged);
        EXPECT_NEAR(result.thermodynamic_pressure_increment, gauge_increment, 2e-11);
        EXPECT_NEAR(result.maximum_continuity_residual, 1., 2e-12);
        EXPECT_LE(result.maximum_linearized_continuity_residual, 1e-11);
        double local_velocity_change = 0.;
        for (std::size_t c = 0; c < target.size(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            EXPECT_NEAR(
                mechanical.value(cell) + result.thermodynamic_pressure_increment, lower(cell) ? 2. : -4., 2e-11);
            const auto u = velocity.value(cell);
            EXPECT_TRUE(std::isfinite(u.z));
            // Closed-Neumann boundary-aware least squares has Nzz=2^2+1^2=5
            // and rhs_z=2*(-6)=-12, hence grad_z=-2.4 and delta_u_z=+1.2.
            EXPECT_NEAR(u.z, 1.2, 2e-11);
            EXPECT_NEAR(u.x, 0., 1e-12);
            EXPECT_NEAR(u.y, 0., 1e-12);
            local_velocity_change += std::abs(u.z);
        }
        double global_velocity_change = 0.;
        Teuchos::reduceAll(*grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_velocity_change,
            &global_velocity_change);
        EXPECT_GT(global_velocity_change, 0.); // No theta multiplier is allowed in the momentum correction.
        for (const auto face : corrected.owned_face_ids())
        {
            if (grid->is_geometry_exterior_face(face))
                EXPECT_DOUBLE_EQ(corrected.value(face), 0.);
            else
            {
                EXPECT_NEAR(corrected.value(face), orientation(face) * 1.6, 2e-12);
                EXPECT_DOUBLE_EQ(theta.value(face) * corrected.value(face) + slip.value(face), 0.);
            }
        }
    }
};

TEST(LowMachPressureContractsTest, NonunitMaterialFractionsPreserveCarrierFluxAndPhysicalGaugeOracle)
{
    Fixture fixture;
    for (const double fraction : {.5, 2.})
    {
        SCOPED_TRACE(fraction);
        fixture.configure(fraction, .2, .1);
        fixture.reset_outputs();
        fixture.nonunit_oracle(fixture.project(fixture.linear()), fraction);
    }
}

TEST(LowMachPressureContractsTest, ZeroThetaWithPositiveSusceptibilityKeepsBareCarrierAndPressureResponse)
{
    Fixture fixture;
    fixture.configure(0., 0., .1);
    for (std::size_t c = 0; c < fixture.chi.size(); ++c)
        fixture.chi[c] = fixture.lower(static_cast<LO>(c)) ? .5 : .25;
    fixture.zero_theta_oracle(fixture.project(fixture.linear()));
}

TEST(LowMachPressureContractsTest, ZeroThetaAndAnyUnanchoredCellRejectAtomicallyAndValidStateReplays)
{
    Fixture fixture;
    fixture.configure(0., 0., .1);
    for (int pattern = 0; pattern < 3; ++pattern)
        for (const bool incompatible_components : {false, true})
        {
            SCOPED_TRACE(pattern);
            SCOPED_TRACE(incompatible_components);
            for (std::size_t c = 0; c < fixture.chi.size(); ++c)
            {
                const bool lower = fixture.lower(static_cast<LO>(c));
                fixture.chi[c] = pattern == 0 || (pattern == 1 ? lower : !lower) ? 0. : .5;
                // Global compatibility can be zero while both disconnected components are incompatible.
                fixture.target[c] = incompatible_components ? (lower ? 1. : -1.) : 0.;
            }
            fixture.reset_outputs(true);
            const auto saved = fixture.snapshot();
            EXPECT_THROW((void) fixture.project(fixture.linear()), std::exception);
            fixture.unchanged(saved);
        }
    fixture.configure(0., 0., .1);
    for (std::size_t c = 0; c < fixture.chi.size(); ++c)
        fixture.chi[c] = fixture.lower(static_cast<LO>(c)) ? .5 : .25;
    fixture.reset_outputs();
    fixture.zero_theta_oracle(fixture.project(fixture.linear()));
}

TEST(LowMachPressureContractsTest, OneRankMalformedLinearPoliciesPreserveAllFieldsAndAllowValidReplay)
{
    Fixture fixture;
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    for (int choice = 0; choice < 4; ++choice)
    {
        SCOPED_TRACE(choice);
        auto selected = fixture.linear();
        if (comm->getRank() == fixture.bad_rank)
        {
            if (choice == 0)
                selected.max_iterations = 0;
            if (choice == 1)
                selected.tolerance = std::numeric_limits<double>::quiet_NaN();
            if (choice == 2)
                selected.backend = static_cast<S::LinearSolverBackend>(-1);
            if (choice == 3)
                selected.preconditioner = static_cast<S::LinearPreconditioner>(-1);
        }
        fixture.reset_outputs(true);
        const auto saved = fixture.snapshot();
        EXPECT_THROW((void) fixture.project(selected), std::exception);
        fixture.unchanged(saved);
        fixture.reset_outputs();
        fixture.nonunit_oracle(fixture.project(fixture.linear()), .5);
    }
}

TEST(LowMachPressureContractsTest, AllSixDivergentLinearControlsRejectIncludingAnEmptyOwnerAndReplay)
{
    Fixture fixture;
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Rank-divergent controls require MPI; malformed controls execute in serial.";
    for (int choice = 0; choice < 6; ++choice)
    {
        SCOPED_TRACE(choice);
        auto selected = fixture.linear();
        if (comm->getRank() == fixture.bad_rank)
        {
            if (choice == 0)
                ++selected.max_iterations;
            if (choice == 1)
                selected.tolerance *= 10.;
            if (choice == 2)
                selected.backend = S::LinearSolverBackend::BiCGStab;
            if (choice == 3)
                selected.preconditioner = S::LinearPreconditioner::Jacobi;
            if (choice == 4)
            {
                selected.verbosity = Belos::Warnings; // Errors is zero, identical to the fixture baseline.
                EXPECT_NE(selected.verbosity, fixture.linear().verbosity);
            }
            if (choice == 5)
                selected.reuse_preconditioner = true;
        }
        fixture.reset_outputs(true);
        const auto saved = fixture.snapshot();
        EXPECT_THROW((void) fixture.project(selected), std::exception);
        fixture.unchanged(saved);
        fixture.reset_outputs();
        fixture.nonunit_oracle(fixture.project(fixture.linear()), .5);
    }
}
} // namespace
