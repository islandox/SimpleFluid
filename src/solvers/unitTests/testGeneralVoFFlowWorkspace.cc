/** @file testGeneralVoFFlowWorkspace.cc */
#include "general_vof_flow_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
TEST(GeneralVoFFlowWorkspaceTest, OrdinaryProjectionKeepsPhysicalThresholdWithoutIntervalAdmission)
{
    const auto mesh = skew_native();
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh);
    EXPECT_EQ(flow.arithmetic_mode(), ArithmeticMode::Ordinary);
    std::vector<VoF::Vector> velocity(n);
    std::vector<double> face(mesh->num_faces()), rho(n, 1000), fraction(n), curvature(n), pressure;
    std::vector<long double> mass(n), uncertainty(n);
    for (std::size_t c = 0; c < n; ++c)
    {
        mass[c] = static_cast<long double>(rho[c]) * mesh->cell_volume(LO(c));
        uncertainty[c] = 2 * mass[c]; // Valid metadata, intentionally no positive interval lower bound.
    }
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto x = mesh->face_centroid(LO(f));
            face[f] = .02 * std::sin(x.x + 2 * x.y + 3 * x.z);
        }
    face = flow.faces(face);
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 1e-11;
    options.pressure_relative_tolerance = 0;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance provenance;
    ASSERT_NO_THROW(flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics,
        &provenance, std::span<const long double>(mass), nullptr, std::span<const long double>(uncertainty)));
    EXPECT_LE(diagnostics.maximum_divergence, .01 * options.pressure_absolute_tolerance);
    EXPECT_GT(diagnostics.pressure_iterations, 0);
    EXPECT_EQ(provenance.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_EQ(provenance.closure_storage_bytes, 0u);
    for (auto value : provenance.face_radius)
        EXPECT_EQ(value, 0);
    for (auto value : provenance.cell_radius)
        EXPECT_EQ(value, 0);
    for (const auto& row : provenance.velocity_error)
        for (auto value : row)
            EXPECT_EQ(value, 0);
    const auto old_face = face, old_pressure = pressure;
    const auto old_velocity = velocity;
    provenance.arithmetic_mode = ArithmeticMode::BoundedCertification;
    EXPECT_THROW(
        flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics, &provenance),
        std::runtime_error);
    EXPECT_EQ(face, old_face);
    EXPECT_EQ(pressure, old_pressure);
    EXPECT_EQ(velocity, old_velocity);
    provenance.arithmetic_mode = ArithmeticMode::Ordinary;
    provenance.byte_budget = 1;
    EXPECT_THROW(
        flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics, &provenance),
        std::runtime_error);
    EXPECT_EQ(face, old_face);
    EXPECT_EQ(pressure, old_pressure);
    EXPECT_EQ(velocity, old_velocity);
}

TEST(GeneralVoFFlowWorkspaceTest, ArithmeticModeMismatchRejectsIncludingEmptyOwners)
{
    const auto mesh = skew_native();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
    {
        const auto mode =
            comm->getRank() + 1 == comm->getSize() ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
        EXPECT_THROW((void) detail::GeneralVoFFlowWorkspace(mesh, mode), std::runtime_error);
    }
    else
        EXPECT_THROW((void) detail::GeneralVoFFlowWorkspace(mesh, static_cast<ArithmeticMode>(99)), std::runtime_error);
    detail::GeneralVoFFlowWorkspace recovered(mesh);
    EXPECT_EQ(recovered.arithmetic_mode(), ArithmeticMode::Ordinary);
}

TEST(GeneralVoFFlowWorkspaceTest, FiniteProjectionKeepsOriginalPhysicalThresholdWithObservedEmptyOwners)
{
    const auto mesh = skew_native();
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(
        mesh, ArithmeticMode::BoundedCertification, std::vector<VoF::Vector>(n, {1.3, 1, 1}));
    std::vector<VoF::Vector> velocity(n);
    std::vector<double> face(mesh->num_faces()), rho(n, 1000), fraction(n), curvature(n), pressure;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto x = mesh->face_centroid(LO(f));
            face[f] = .02 * std::sin(x.x + 2 * x.y + 3 * x.z);
        }
    face = flow.faces(face);
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 1e-11;
    options.pressure_relative_tolerance = 0;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance provenance;
    std::vector<long double> mass(n), mass_error(n);
    for (std::size_t c = 0; c < n; ++c)
    {
        mass[c] = static_cast<long double>(rho[c]) * mesh->cell_volume(LO(c));
        mass_error[c] = 1e-10L * mass[c];
    }
    ASSERT_NO_THROW(flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics,
        &provenance, std::span<const long double>(mass), nullptr, std::span<const long double>(mass_error)));
    EXPECT_LE(diagnostics.maximum_divergence, .01 * options.pressure_absolute_tolerance);
    EXPECT_GT(diagnostics.pressure_iterations, 0);
    EXPECT_EQ(provenance.face_rate.size(), mesh->num_owned_faces());
    EXPECT_EQ(provenance.face_radius.size(), mesh->num_owned_faces());
    EXPECT_EQ(provenance.cell_radius.size(), n);
    ASSERT_EQ(provenance.velocity_error.size(), n);
    int local_nonzero_error = 0, nonzero_error = 0;
    for (const auto& row : provenance.velocity_error)
        for (auto error : row)
        {
            EXPECT_TRUE(std::isfinite(error));
            EXPECT_GE(error, 0);
            local_nonzero_error += error > 0;
        }
    Teuchos::reduceAll(
        *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_nonzero_error, &nonzero_error);
    EXPECT_GT(nonzero_error, 0);
    for (std::size_t f = 0; f < provenance.face_rate.size(); ++f)
    {
        EXPECT_TRUE(std::isfinite(provenance.face_rate[f]));
        EXPECT_GE(provenance.face_radius[f], 0);
        if (mesh->is_geometry_exterior_face(LO(f)))
            EXPECT_EQ(provenance.face_rate[f], 0);
    }
    const int local_empty = n == 0;
    int empty = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, mesh->owned_cell_map()->getComm()->getSize() - 1);

    // Exhaustion is independent of the observed pressure defect. Rejection
    // occurs at preflight, before any candidate field can be changed.
    const auto before_face = face, before_pressure = pressure;
    const auto before_velocity = velocity;
    provenance.byte_budget = 1;
    EXPECT_THROW(
        flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics, &provenance),
        std::runtime_error);
    EXPECT_EQ(face, before_face);
    EXPECT_EQ(pressure, before_pressure);
    EXPECT_EQ(velocity, before_velocity);
    provenance.byte_budget = 64 * 1024 * 1024;
    mass_error = mass;
    EXPECT_THROW(
        flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics, &provenance,
            std::span<const long double>(mass), nullptr, std::span<const long double>(mass_error)),
        std::runtime_error);
    EXPECT_EQ(face, before_face);
    EXPECT_EQ(pressure, before_pressure);
    EXPECT_EQ(velocity, before_velocity);
}
} // namespace

namespace
{
TEST(GeneralVoFFlowWorkspaceTest, FiniteExpressionBoundsRetainCanceledOperandsAndCachedCoefficientError)
{
    namespace arithmetic = detail::vof_projection_fp;
    namespace fp = VoF::finite_detail;
    const auto large = std::ldexp(1., 53);
    // The old post-cancellation magnitude is zero even though this sequence
    // has discarded one. A cached binary64 intermediate must retain it.
    const auto rounded = arithmetic::binary64(fp::add({large, 0}, {1, 0}));
    const auto canceled = fp::subtract(rounded, {large, 0});
    EXPECT_EQ(canceled.value, 0);
    EXPECT_GE(canceled.absolute_error, 1);
    const auto wide = arithmetic::dot({large, 1, -large}, {1, 1, 1}, ArithmeticMode::BoundedCertification);
    EXPECT_LE(std::abs(wide.value - 1.L), wide.absolute_error);
    EXPECT_EQ(wide.value, 1);

    const auto beta = arithmetic::binary64(fp::divide({1, 0}, {3, 0}));
    // The nearest binary64 value of 1/3 is below it by 2^-54 / 3.
    // Its upward FP enclosure provides an independent analytic error bound.
    const auto difference_upper = std::nextafter(std::ldexp(1.L, -54) / 3,
        std::numeric_limits<long double>::infinity());
    EXPECT_GE(beta.absolute_error, difference_upper);
    EXPECT_GT(beta.absolute_error, 0);
    // Reusing a cached coefficient must preserve its source error; multiplying
    // by exact zero, in contrast, proves an exact zero contribution.
    const auto reused = fp::multiply(beta, {8, 0});
    EXPECT_GE(reused.absolute_error, beta.absolute_error * 8);
    EXPECT_EQ(fp::multiply(beta, {0, 0}).absolute_error, 0);
    EXPECT_THROW(arithmetic::binary64({std::numeric_limits<long double>::max(), 0}), std::invalid_argument);
}
TEST(GeneralVoFFlowWorkspaceTest, FiniteConstantCurvatureRetainsPressureWallForceAndMassError)
{
    const auto mesh = skew_native();
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(
        mesh, ArithmeticMode::BoundedCertification, std::vector<VoF::Vector>(n, {1.3, 1, 1}));
    std::vector<double> rho(n, 1), fraction(n), curvature(n, 1), pressure, face(mesh->num_faces());
    std::vector<VoF::Vector> velocity(n);
    std::vector<long double> mass(n), mass_error(n);
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        fraction[c] = .2 + .1 * x.x + .05 * x.y - .02 * x.z;
        mass[c] = mesh->cell_volume(LO(c));
        mass_error[c] = mass[c] * 1e-12L;
    }
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 1e-13;
    options.pressure_relative_tolerance = 1e-12;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance provenance;
    ASSERT_NO_THROW(flow.project(face, velocity, pressure, rho, fraction, curvature, .01, 1, options, diagnostics,
        &provenance, std::span<const long double>(mass), nullptr, std::span<const long double>(mass_error)));
    const auto gradient = flow.gradient(pressure);
    for (std::size_t c = 0; c < n; ++c)
    {
        EXPECT_NEAR(gradient[c][0], .1, 2e-10);
        EXPECT_NEAR(gradient[c][1], .05, 2e-10);
        EXPECT_NEAR(gradient[c][2], -.02, 2e-10);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(velocity[c][axis], 0, 2e-11);
            EXPECT_TRUE(std::isfinite(provenance.velocity_error[c][axis]));
            EXPECT_GE(provenance.velocity_error[c][axis], 0);
        }
    }
    for (double value : face)
        EXPECT_NEAR(value, 0, 2e-11);
    EXPECT_LT(diagnostics.maximum_divergence, 1e-12);
}
} // namespace

namespace
{
TEST(GeneralVoFFlowWorkspaceTest, StaticProjectionArithmeticPreservesModeAndNarrowingEvidence)
{
    namespace arithmetic = detail::vof_projection_fp;
    namespace fp = VoF::finite_detail;
    using Ordinary = fp::Arithmetic<ArithmeticMode::Ordinary>;
    using Certified = fp::Arithmetic<ArithmeticMode::BoundedCertification>;
    const auto ordinary = arithmetic::binary64<ArithmeticMode::Ordinary>(
        Ordinary::divide(Ordinary::make_value(1), Ordinary::make_value(3)));
    const auto certified = arithmetic::binary64<ArithmeticMode::BoundedCertification>(
        Certified::divide(Certified::make_value(1), Certified::make_value(3)));
    EXPECT_EQ(ordinary.value, certified.value);
    EXPECT_EQ(ordinary.absolute_error, 0);
    EXPECT_GT(certified.absolute_error, 0);
    EXPECT_EQ(ordinary.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_EQ(certified.arithmetic_mode, ArithmeticMode::BoundedCertification);
    EXPECT_THROW((arithmetic::binary64<ArithmeticMode::Ordinary>(Certified::make_value(1, .125L))),
        std::invalid_argument);
    EXPECT_THROW((arithmetic::binary64<ArithmeticMode::BoundedCertification>(Ordinary::make_value(1))),
        std::invalid_argument);

    const auto large = std::ldexp(1., 53);
    const std::array<double, 3> left{large, 1, -large}, right{1, 1, 1};
    const auto represented = arithmetic::dot<ArithmeticMode::Ordinary>(left, right);
    const auto bounded = arithmetic::dot<ArithmeticMode::BoundedCertification>(left, right);
    EXPECT_EQ(represented.value, bounded.value);
    EXPECT_EQ(represented.absolute_error, 0);
    EXPECT_EQ(bounded.value, 1);
    EXPECT_LE(std::abs(bounded.value - 1), bounded.absolute_error);
    const auto adapter = arithmetic::dot(left, right, ArithmeticMode::BoundedCertification);
    EXPECT_EQ(adapter.value, bounded.value);
    EXPECT_EQ(adapter.absolute_error, bounded.absolute_error);
}
} // namespace
