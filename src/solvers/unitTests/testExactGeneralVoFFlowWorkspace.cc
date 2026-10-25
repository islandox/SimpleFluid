/** @file testExactGeneralVoFFlowWorkspace.cc */
#include "general_vof_flow_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
TEST(ExactGeneralVoFFlowWorkspaceTest, SingleExtrudedLayerRetainsConstantAndInPlaneAffineGradients)
{
    const auto mesh = skew_native(1);
    EXPECT_EQ(mesh->num_owned_cells() == 0, Tpetra::getDefaultComm()->getRank() != 0);
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<double> field(mesh->num_owned_cells(), 7);
    for (const auto& g : flow.gradient(field))
        for (double component : g)
            EXPECT_EQ(component, 0);
    for (std::size_t c = 0; c < field.size(); ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        field[c] = 2 * x.x - .75 * x.y + 3;
    }
    for (const auto& g : flow.gradient(field))
    {
        EXPECT_NEAR(g[0], 2, 2e-12);
        EXPECT_NEAR(g[1], -.75, 2e-12);
        EXPECT_EQ(g[2], 0);
    }
}

TEST(ExactGeneralVoFFlowWorkspaceTest, IsolatedBoundaryTetrahedronConstantIsExactAndNonfiniteIsRejectedCollectively)
{
    const auto mesh = boundary_polyhedron(false);
    EXPECT_EQ(mesh->num_owned_cells() == 0, Tpetra::getDefaultComm()->getRank() != 0);
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<double> scalar(mesh->num_owned_cells(), 3);
    for (const auto& g : flow.gradient(scalar))
        for (double component : g)
            EXPECT_EQ(component, 0);
    for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        std::fill(scalar.begin(), scalar.end(), value);
        EXPECT_THROW((void) flow.gradient(scalar), std::runtime_error);
    }
}

TEST(ExactGeneralVoFFlowWorkspaceTest, UnresolvedNonconstantBoundaryPolyhedralStencilRemainsRejected)
{
    const auto mesh = boundary_polyhedron(true);
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<double> scalar(mesh->num_owned_cells());
    for (std::size_t c = 0; c < scalar.size(); ++c)
        scalar[c] = mesh->cell_centroid(LO(c)).x;
    EXPECT_THROW((void) flow.gradient(scalar), std::runtime_error);
}

TEST(ExactGeneralVoFFlowWorkspaceTest, AffinePhysicalGradientAndCurvatureOnActualSkewNativeMesh)
{
    const auto mesh = skew_native();
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<double> scalar(mesh->num_owned_cells());
    std::vector<VoF::Vector> normal(scalar.size());
    for (std::size_t c = 0; c < scalar.size(); ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        scalar[c] = 2 * x.x - .75 * x.y + .4 * x.z;
        normal[c] = {1, 0, 0};
    }
    const auto grad = flow.gradient(scalar);
    const auto k = flow.curvature(normal);
    for (std::size_t c = 0; c < scalar.size(); ++c)
    {
        EXPECT_NEAR(grad[c][0], 2, 2e-12);
        EXPECT_NEAR(grad[c][1], -.75, 2e-12);
        EXPECT_NEAR(grad[c][2], .4, 2e-12);
        EXPECT_NEAR(k[c], 0, 2e-12);
        EXPECT_GT(flow.physical_spans()[c][0], 1);
        EXPECT_NEAR(flow.physical_spans()[c][1], 1, 2e-12);
    }
}

TEST(ExactGeneralVoFFlowWorkspaceTest, NonorthogonalProjectionChecksPhysicalDivergenceAndViscousRigidMode)
{
    const auto mesh = skew_native();
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    const auto n = mesh->num_owned_cells();
    std::vector<VoF::Vector> velocity(n, {.2, -.1, .3});
    const auto rigid = flow.viscous_force(velocity, std::vector<double>(n, 1));
    for (const auto& force : rigid)
        for (double value : force)
            EXPECT_NEAR(value, 0, 1e-12);
    std::vector<double> face(mesh->num_faces()), rho(n, 1000), fraction(n), kappa(n), pressure;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto x = mesh->face_centroid(LO(f));
            face[f] = .02 * std::sin(x.x + 2 * x.y + 3 * x.z);
        }
    face = flow.faces(face);
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 1e-11;
    VoFStepDiagnostics diagnostic;
    EXPECT_NO_THROW(detail::Exact::GeneralVoFFlowWorkspace::project(
        flow, face, velocity, pressure, rho, fraction, kappa, .01, 0, options, diagnostic));
    EXPECT_LT(diagnostic.maximum_divergence, 2e-12);
    EXPECT_GT(diagnostic.pressure_iterations, 0);
    EXPECT_GT(flow.stable_timestep(face, 1000, 1, 0, options), 0);
}

TEST(ExactGeneralVoFFlowWorkspaceTest, ProjectionArithmeticRequiresOriginalDefectCorrections)
{
    const auto mesh = skew_native();
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    const auto n = mesh->num_owned_cells();
    std::vector<double> face(mesh->num_faces()), rho(n, 1000), fraction(n), curvature(n), pressure;
    std::vector<VoF::Vector> velocity(n);
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto x = mesh->face_centroid(LO(f));
            face[f] = .02 * std::sin(x.x + 2 * x.y + 3 * x.z);
        }
    face = flow.faces(face);
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 1e-11;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFProjectionProvenance provenance;
    EXPECT_NO_THROW(detail::Exact::GeneralVoFFlowWorkspace::project(
        flow, face, velocity, pressure, rho, fraction, curvature, .01, 0, options, diagnostics, &provenance));
    EXPECT_EQ(provenance.face_velocity.size(), mesh->num_owned_faces());
    EXPECT_LT(diagnostics.maximum_divergence, 2e-12);
    EXPECT_GT(diagnostics.pressure_iterations, 0);
    for (std::size_t f = 0; f < provenance.face_velocity.size(); ++f)
        EXPECT_EQ(provenance.face_velocity[f].value, face[f]);
}

TEST(ExactGeneralVoFFlowWorkspaceTest, ConstantCurvaturePressureAndWallCapillaryReactionBalance)
{
    const auto mesh = skew_native();
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    const auto n = mesh->num_owned_cells();
    std::vector<double> rho(n, 1), fraction(n), curvature(n, 1), pressure, face(mesh->num_faces());
    std::vector<VoF::Vector> velocity(n);
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        fraction[c] = .2 + .1 * x.x + .05 * x.y - .02 * x.z;
    }
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 1e-13;
    options.pressure_relative_tolerance = 1e-12;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFProjectionProvenance provenance;
    std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> mass(n);
    const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&mesh->variant());
    ASSERT_TRUE(native && *native);
    for (std::size_t c = 0; c < n; ++c)
        mass[c] = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(
            *SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>::from_native_cell(
                **native, mesh->cell_geometry_global_id(LO(c))));
    auto inconsistent = mass;
    if (!inconsistent.empty())
        inconsistent[0] = inconsistent[0].scaled(2);
    EXPECT_THROW(
        detail::Exact::GeneralVoFFlowWorkspace::project(flow, face, velocity, pressure, rho, fraction, curvature, .01,
            1, options, diagnostics, &provenance,
            std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(inconsistent)),
        std::runtime_error);
    ASSERT_NO_THROW(detail::Exact::GeneralVoFFlowWorkspace::project(flow, face, velocity, pressure, rho, fraction,
        curvature, .01, 1, options, diagnostics, &provenance,
        std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(mass)));
    const auto gradient = flow.gradient(pressure);
    for (std::size_t c = 0; c < n; ++c)
    {
        EXPECT_NEAR(gradient[c][0], .1, 2e-10);
        EXPECT_NEAR(gradient[c][1], .05, 2e-10);
        EXPECT_NEAR(gradient[c][2], -.02, 2e-10);
        for (double u : velocity[c])
            EXPECT_NEAR(u, 0, 2e-11);
    }
    for (double u : face)
        EXPECT_NEAR(u, 0, 2e-11);
    EXPECT_LT(diagnostics.maximum_divergence, 1e-12);
}

TEST(ExactGeneralVoFFlowWorkspaceTest, IndependentInteriorPolynomialRecoversFullSymmetricStressDivergence)
{
    AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {6, 6, 6};
    grid.extent = {6, 6, 6};
    auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    std::vector<std::uint64_t> owned;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t c = 0; c < 216; ++c)
            owned.push_back(c);
    auto bootstrap = provider->detach_owned(owned, {'p', 'o', 'l', 'y', 's', 't', 'r', 'e', 's', 's'});
    const auto schema = bootstrap.kernel->descriptor();
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    const auto mesh = AMR::assemble_adaptive_native(*hierarchy, schema).mesh;
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<VoF::Vector> velocity(mesh->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        velocity[c] = {x.x * x.x, -2 * x.x * x.y, 0};
    }
    const auto force = flow.viscous_force(velocity, std::vector<double>(velocity.size(), .125));
    double local_checked = 0, checked = 0;
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        if (std::abs(x.x - 2.5) < 1e-12 && std::abs(x.y - 2.5) < 1e-12 && std::abs(x.z - 2.5) < 1e-12)
        {
            // div(mu*(grad u+grad u^T))=(2mu,0,0). The transpose
            // contribution is essential: dropping it gives4mu instead.
            EXPECT_NEAR(force[c][0], .25, 5e-13);
            EXPECT_NEAR(force[c][1], 0, 5e-13);
            EXPECT_NEAR(force[c][2], 0, 5e-13);
            local_checked += 1;
        }
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_checked, &checked);
    EXPECT_EQ(checked, 1);
}

TEST(ExactGeneralVoFFlowWorkspaceTest, PrimaryRatePreservesSubUlpPressureCorrectionsAndRecovers)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    const auto mesh = skew_native();
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    const auto n = mesh->num_owned_cells();
    std::vector<double> initial(mesh->num_faces()), rho(n, 1), fraction(n), curvature(n), pressure;
    std::vector<VoF::Vector> velocity(n);
    detail::Exact::GeneralVoFProjectionClosureTopology topology;
    const auto identity = [](const auto& x, char kind)
    {
        std::vector<char> result{kind};
        for (double value : {x.x, x.y, x.z})
        {
            const auto bits = std::bit_cast<std::uint64_t>(value);
            for (unsigned b = 0; b < 8; ++b)
                result.push_back(char(bits >> (8 * b)));
        }
        return result;
    };
    {
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
            topology.cells.push_back({mesh->cell_global_id(LO(c)), identity(mesh->cell_centroid(LO(c)), 'c'), {}});
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            if (!mesh->is_geometry_exterior_face(LO(f)))
            {
                const auto x = mesh->face_centroid(LO(f));
                initial[f] = 1e-6 * std::sin(x.x + 2 * x.y + 3 * x.z);
                topology.faces.push_back({mesh->face_global_id(LO(f)), mesh->cell_global_id(mesh->owner_cell(LO(f))),
                    mesh->cell_global_id(mesh->neighbor_cell(LO(f))), identity(x, 'f'), {}, {}});
            }
    }
    initial = flow.faces(initial);
    VoFSolverOptions options;
    options.max_pressure_iterations = 1;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFProjectionProvenance provenance;
    provenance.face_rate = {M::from_value(17)};
    auto face = initial;
    EXPECT_THROW(detail::Exact::GeneralVoFFlowWorkspace::project(flow, face, velocity, pressure, rho, fraction,
                     curvature, .002, 0, options, diagnostics, &provenance, std::nullopt, &topology),
        std::runtime_error);
    ASSERT_EQ(provenance.face_rate.size(), 1u);
    EXPECT_EQ(provenance.face_rate[0].compare(M::from_value(17)), 0);
    face = initial;
    velocity.assign(n, {});
    options.max_pressure_iterations = 3000;
    ASSERT_NO_THROW(detail::Exact::GeneralVoFFlowWorkspace::project(flow, face, velocity, pressure, rho, fraction,
        curvature, .002, 0, options, diagnostics, &provenance, std::nullopt, &topology));
    EXPECT_LE(diagnostics.maximum_divergence, options.divergence_tolerance);
    EXPECT_GT(diagnostics.pressure_iterations, 1);
    int local_retained = 0, retained = 0;
    const auto lease = mesh->acquire_execution_view();
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        local_retained += provenance.face_rate[f].compare(M::from_value(face[f]).scaled(mesh->face_area(LO(f)))) != 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_retained, &retained);
    EXPECT_GT(retained, 0); // The authoritative iterate retains sub-ulp pressure work.
}

TEST(ExactGeneralVoFFlowWorkspaceTest, ZeroRadiusOrthogonalPressureUsesSameExactRateAndWallImpulse)
{
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {1, 1, 1};
    auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    std::vector<std::uint64_t> owned;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t c = 0; c < 8; ++c)
            owned.push_back(c);
    auto bootstrap = provider->detach_owned(owned, {'e', 'x', 'a', 'c', 't', 'w', 'a', 'l', 'l'});
    const auto schema = bootstrap.kernel->descriptor();
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    const auto mesh = AMR::assemble_adaptive_native(*hierarchy, schema).mesh;
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    const auto n = mesh->num_owned_cells();
    std::vector<double> face(mesh->num_faces()), rho(n, 1), fraction(n), curvature(n), pressure;
    std::vector<VoF::Vector> velocity(n, {1, 0, 0});
    detail::Exact::GeneralVoFProjectionClosureTopology topology;
    {
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
            topology.cells.push_back({mesh->cell_global_id(LO(c)), {char('c'), char(c)}, {}});
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            if (!mesh->is_geometry_exterior_face(LO(f)))
            {
                face[f] = mesh->face_normal(LO(f)).x;
                topology.faces.push_back({mesh->face_global_id(LO(f)), mesh->cell_global_id(mesh->owner_cell(LO(f))),
                    mesh->cell_global_id(mesh->neighbor_cell(LO(f))), {char('f'), char(f)}, {}, {}});
            }
    }
    face = flow.faces(face);
    VoFSolverOptions options;
    options.max_pressure_iterations = 16;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFProjectionProvenance provenance;
    ASSERT_NO_THROW(detail::Exact::GeneralVoFFlowWorkspace::project(flow, face, velocity, pressure, rho, fraction,
        curvature, 1, 0, options, diagnostics, &provenance, std::nullopt, &topology));
    EXPECT_GT(diagnostics.pressure_iterations, 0);
    EXPECT_EQ(diagnostics.maximum_divergence, 0);
    for (const auto& rate : provenance.face_rate)
        EXPECT_EQ(rate.sign(), 0);
    for (std::size_t c = 0; c < n; ++c)
    {
        EXPECT_NEAR(velocity[c][0], 0, 5e-15);
        EXPECT_NEAR(velocity[c][1], 0, 5e-15);
        EXPECT_NEAR(velocity[c][2], 0, 5e-15);
        // p=x plus an arbitrary constant: test the physical difference using
        // native x-centers instead of relying on a particular gauge value.
        EXPECT_NEAR(pressure[c] - pressure[0], mesh->cell_centroid(LO(c)).x - mesh->cell_centroid(0).x, 5e-15);
    }
}

TEST(ExactGeneralVoFFlowWorkspaceTest, PreparedGradientRetainsNonlinearResultsAfterRejectedSamplesAndNewWorkspaces)
{
    const auto mesh = skew_native();
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<double> scalar(mesh->num_owned_cells());
    for (std::size_t c = 0; c < scalar.size(); ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        scalar[c] = std::sin(x.x) + x.y * x.z + .25 * x.x * x.x;
    }
    const auto first = flow.gradient(scalar);
    const auto reference = uncached_gradient(*mesh, scalar, flow.cells(scalar), flow.physical_spans());
    auto rejected = scalar;
    if (!rejected.empty())
        rejected.front() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) flow.gradient(rejected), std::runtime_error);
    const auto again = flow.gradient(scalar);
    detail::GeneralVoFFlowWorkspace fresh(mesh);
    const auto rebuilt = fresh.gradient(scalar);
    ASSERT_EQ(first.size(), again.size());
    for (std::size_t c = 0; c < first.size(); ++c)
        for (unsigned a = 0; a < 3; ++a)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(first[c][a]), std::bit_cast<std::uint64_t>(reference[c][a]));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(first[c][a]), std::bit_cast<std::uint64_t>(again[c][a]));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(first[c][a]), std::bit_cast<std::uint64_t>(rebuilt[c][a]));
        }
}

TEST(ExactGeneralVoFFlowWorkspaceTest, GeometryReplacementRejectsAllRetainedGeometryAndFreshWorkspaceRecovers)
{
    using namespace Meshes;
    auto native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}});
    const auto composite =
        std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("moving", native)},
            std::vector<MultiRegionMesh::Interface>{});
    const auto mesh = std::make_shared<MeshHandle<>>(composite);
    detail::GeneralVoFFlowWorkspace flow(mesh, ArithmeticMode::Exact);
    std::vector<double> scalar(mesh->num_owned_cells(), 3), face(mesh->num_faces());
    EXPECT_NO_THROW((void) flow.gradient(scalar));
    if (Tpetra::getDefaultComm()->getRank() == 0)
        *native = OrthogonalCartesian3D(Vec3D<ArrReal>{{{0, 1, 2}, {0, 1, 2}, {0, 2, 4}}});
    EXPECT_THROW((void) flow.gradient(scalar), std::runtime_error);
    EXPECT_THROW((void) flow.cells(scalar), std::runtime_error);
    EXPECT_THROW((void) flow.faces(face), std::runtime_error);
    if (Tpetra::getDefaultComm()->getRank() == 0)
        EXPECT_THROW((void) flow.physical_spans(), std::runtime_error);
    else
        EXPECT_NO_THROW((void) flow.physical_spans());
    EXPECT_THROW((void) flow.stable_timestep(face, 1, 0, 0, VoFSolverOptions{}), std::runtime_error);
    if (Tpetra::getDefaultComm()->getRank() != 0)
        *native = OrthogonalCartesian3D(Vec3D<ArrReal>{{{0, 1, 2}, {0, 1, 2}, {0, 2, 4}}});
    const auto replacement =
        std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("replacement", native)},
            std::vector<MultiRegionMesh::Interface>{});
    const auto target = std::make_shared<MeshHandle<>>(replacement);
    detail::GeneralVoFFlowWorkspace rebuilt(target, ArithmeticMode::Exact);
    for (const auto& g : rebuilt.gradient(std::vector<double>(target->num_owned_cells(), 3)))
        for (double component : g)
            EXPECT_EQ(component, 0);
}
} // namespace
