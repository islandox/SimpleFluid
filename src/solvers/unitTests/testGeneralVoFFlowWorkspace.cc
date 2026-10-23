/** @file testGeneralVoFFlowWorkspace.cc */
#include "general_vof_flow_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
TEST(GeneralVoFFlowWorkspaceTest, FiniteProjectionKeepsOriginalPhysicalThresholdWithObservedEmptyOwners)
{
    const auto mesh = skew_native();
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh, true, std::vector<VoF::Vector>(n, {1.3, 1, 1}));
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


} // namespace
