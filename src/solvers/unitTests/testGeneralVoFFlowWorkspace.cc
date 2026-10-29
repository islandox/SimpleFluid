/** @file testGeneralVoFFlowWorkspace.cc */
#include "general_vof_flow_test_helpers.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include <boost/multiprecision/cpp_int.hpp>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
SP<const MeshHandle<>> axial_slab_native(unsigned normal_axis, unsigned rows = 2)
{
    using U = Meshes::UnstructuredMesh;
    Arr<MeshUtils::Vec3> nodes;
    for (unsigned z = 0; z < 2; ++z)
        for (unsigned y = 0; y <= rows; ++y)
            for (unsigned x = 0; x < 3; ++x)
            {
                const std::array<double, 3> original{double(x), double(y), double(z)};
                std::array<double, 3> rotated{};
                for (unsigned a = 0; a < 3; ++a)
                    rotated[(a + normal_axis + 1) % 3] = original[a];
                nodes.push_back({rotated[0], rotated[1], rotated[2]});
            }
    Arr<U::CellDefinition> cells;
    const auto plane_nodes = U::NodeID(3 * (rows + 1));
    for (unsigned y = 0; y < rows; ++y)
        for (unsigned x = 0; x < 2; ++x)
        {
            U::CellDefinition cell;
            cell.type = MeshUtils::CellType::HEXAHEDRON;
            const auto lo = U::NodeID(y * 3 + x);
            cell.node_ids = {lo, lo + 1, lo + 4, lo + 3, lo + plane_nodes, lo + plane_nodes + 1,
                lo + plane_nodes + 4, lo + plane_nodes + 3};
            cells.push_back(std::move(cell));
        }
    const U untagged(nodes, cells);
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (std::size_t f = 0; f < untagged.num_faces(); ++f)
        if (untagged.is_exterior_face(f))
            boundaries.push_back({untagged.face_nodes(f), 0, "wall"});
    const auto native = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{
            Meshes::native_region("slab", std::make_shared<U>(nodes, cells, boundaries))},
        std::vector<Meshes::MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    AMR::PolyhedralAMR source(test::polyhedral_grid(native));
    std::vector<std::uint64_t> owned;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t c = 0; c < 2 * rows; ++c)
            owned.push_back(c);
    auto bootstrap = source.detach_owned(owned, {'s', 'l', 'a', 'b'});
    const auto schema = bootstrap.kernel->descriptor();
    const AMR::DistributedAdaptiveHierarchy hierarchy(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    return AMR::assemble_adaptive_native(hierarchy, schema).mesh;
}

template<ArithmeticMode Mode> void missing_axis_wall_pressure_contract()
{
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        constexpr auto mode = Mode;
        SCOPED_TRACE(::testing::Message() << axis << ',' << int(mode));
        const auto mesh = axial_slab_native(axis);
        detail::GeneralVoFFlowWorkspace flow(mesh, mode);
        const auto n = mesh->num_owned_cells();
        std::vector<VoF::Vector> velocity(n);
        for (auto& value : velocity)
            value[axis] = .125;
        std::vector<double> faces(mesh->num_faces()), density(n, 3), fraction(n), curvature(n), pressure;
        std::vector<long double> mass(n);
        for (std::size_t c = 0; c < n; ++c)
            mass[c] = 7.L * mesh->cell_volume(LO(c));
        VoFSolverOptions options;
        VoFStepDiagnostics diagnostics;
        detail::GeneralVoFFiniteProjectionProvenance evidence;
        flow.project(faces, velocity, pressure, density, fraction, curvature, .125, 0, options, diagnostics, &evidence,
            std::span<const long double>(mass));
        ASSERT_EQ(evidence.wall_pressure_offset.size(), mesh->num_owned_faces());
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        {
            const auto face = LO(f);
            const auto normal = mesh->face_normal(face);
            const std::array<double, 3> components{normal.x, normal.y, normal.z};
            const auto displacement = mesh->face_center_vector(face, mesh->owner_cell(face));
            const std::array<double, 3> offset{displacement.x, displacement.y, displacement.z};
            const auto expected = mesh->is_geometry_exterior_face(face) && components[axis] != 0 ? 7 * offset[axis] : 0;
            EXPECT_NEAR(evidence.wall_pressure_offset[f], expected, 1e-12);
            EXPECT_EQ(faces[f], 0);
        }
        for (const auto& value : velocity)
            for (const auto component : value)
                EXPECT_NEAR(component, 0, 1e-12);
        EXPECT_LE(evidence.maximum_wall_normal_residual, options.divergence_tolerance);
        EXPECT_EQ(diagnostics.maximum_divergence, 0);
        const int local_empty = n == 0;
        int empty = 0;
        const auto comm = mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(empty, comm->getSize() - 1);
    }
}

TEST(GeneralVoFFlowWorkspaceTest, OrdinaryMissingAxisWallPressureBalancesPhysicalMassAndInitialNormalMotion)
{
    missing_axis_wall_pressure_contract<ArithmeticMode::Ordinary>();
}

TEST(GeneralVoFFlowWorkspaceTest, CertifiedMissingAxisWallPressureBalancesPhysicalMassAndInitialNormalMotion)
{
    const auto comm = Tpetra::getDefaultComm();
    const int local_supported = PLIC::certification_environment();
    int supported = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_supported, &supported);
    if (!supported)
        GTEST_SKIP() << "Bounded certification requires its supported FP environment on every rank.";
    missing_axis_wall_pressure_contract<ArithmeticMode::BoundedCertification>();
}

TEST(GeneralVoFFlowWorkspaceTest, SingleLayerPressureReactionPreservesResolvedTangentialProjection)
{
    const auto mesh = skew_native(1);
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace baseline(mesh), with_normal(mesh);
    std::vector<VoF::Vector> tangent(n), velocity;
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto center = mesh->cell_centroid(LO(c));
        tangent[c] = {.01 * (center.y - 1.5), -.01 * (center.x - 1.95), 0};
    }
    velocity = tangent;
    for (auto& value : velocity)
        value[2] = .125;
    auto base_face = baseline.interpolate(tangent), face = base_face;
    std::vector<double> rho(n, 1), fraction(n), curvature(n), base_pressure, pressure;
    VoFSolverOptions options;
    VoFStepDiagnostics base_diagnostics, diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance evidence;
    baseline.project(base_face, tangent, base_pressure, rho, fraction, curvature, .125, 0, options, base_diagnostics);
    with_normal.project(face, velocity, pressure, rho, fraction, curvature, .125, 0, options, diagnostics, &evidence);
    for (std::size_t c = 0; c < n; ++c)
    {
        EXPECT_NEAR(velocity[c][0], tangent[c][0], 1e-12);
        EXPECT_NEAR(velocity[c][1], tangent[c][1], 1e-12);
        EXPECT_NEAR(velocity[c][2], 0, 1e-12);
        EXPECT_NEAR(pressure[c], base_pressure[c], 1e-12);
    }
    EXPECT_EQ(face, base_face);
    EXPECT_LE(evidence.maximum_wall_normal_residual, options.divergence_tolerance);
}

TEST(GeneralVoFFlowWorkspaceTest, SampledThreeDimensionalProjectionHasNoAffineWallLift)
{
    const auto mesh = skew_native();
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh);
    std::vector<VoF::Vector> velocity(n, {0, 0, .125});
    auto face = flow.interpolate(velocity);
    std::vector<double> rho(n, 1), fraction(n), curvature(n), pressure;
    VoFSolverOptions options;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance evidence;
    flow.project(face, velocity, pressure, rho, fraction, curvature, .125, 0, options, diagnostics, &evidence);
    EXPECT_TRUE(evidence.wall_pressure_offset.empty());
    EXPECT_EQ(evidence.maximum_wall_normal_residual, 0);
    EXPECT_LE(diagnostics.maximum_divergence, options.divergence_tolerance);
}

TEST(GeneralVoFFlowWorkspaceTest, UnsupportedUnsampledNormalMotionRejectsWithoutChangingCandidates)
{
    const auto mesh = boundary_polyhedron(false);
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh);
    std::vector<VoF::Vector> velocity(n, {.125, 0, 0});
    std::vector<double> face(mesh->num_faces()), rho(n, 1), fraction(n), curvature(n), pressure(n, 17);
    const auto old_velocity = velocity;
    const auto old_face = face, old_pressure = pressure;
    VoFSolverOptions options;
    VoFStepDiagnostics diagnostics;
    EXPECT_THROW(flow.project(face, velocity, pressure, rho, fraction, curvature, .125, 0, options, diagnostics),
        std::runtime_error);
    EXPECT_EQ(velocity, old_velocity);
    EXPECT_EQ(face, old_face);
    EXPECT_EQ(pressure, old_pressure);
}

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

TEST(GeneralVoFFlowWorkspaceTest, OrdinaryAlgebraicReadinessRefinesBeyondLoosePhysicalGate)
{
    using Q = boost::multiprecision::cpp_rational;
    const auto mesh = axial_slab_native(2);
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh);
    std::vector<VoF::Vector> initial_velocity(n);
    std::vector<double> initial_face(mesh->num_faces()), rho(n, 1000), fraction(n), curvature(n);
    // An exact clockwise circulation has rate 1/8 on each of four edges.
    // The bottom edge additionally carries a known, nonzero continuity defect.
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto x = mesh->face_centroid(LO(f)), normal = mesh->face_normal(LO(f));
            initial_face[f] =
                normal.x != 0 ? normal.x * (x.y < 1 ? .125 + .01 : -.125) : normal.y * (x.x > 1 ? .125 : -.125);
        }
    initial_face = flow.faces(initial_face);
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 10;
    options.pressure_relative_tolerance = 0;
    options.divergence_tolerance = 1;
    options.max_courant = .25;
    auto physical_face = initial_face;
    auto physical_velocity = initial_velocity;
    std::vector<double> physical_pressure;
    VoFStepDiagnostics physical_diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance physical;
    ASSERT_NO_THROW(flow.project(physical_face, physical_velocity, physical_pressure, rho, fraction, curvature, .125, 0,
        options, physical_diagnostics, &physical));
    EXPECT_EQ(physical_diagnostics.pressure_iterations, 0);
    EXPECT_GT(physical_diagnostics.maximum_divergence, .009);

    auto face = initial_face;
    auto velocity = initial_velocity;
    std::vector<double> pressure;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance ready;
    constexpr double tau = 8e-15, courant = .25;
    ready.transport_readiness = detail::GeneralVoFFiniteProjectionProvenance::TransportReadiness{tau, courant};
    {
        auto limited_options = options;
        limited_options.max_pressure_iterations = 1;
        auto limited_face = initial_face;
        auto limited_velocity = initial_velocity;
        std::vector<double> limited_pressure(n, 17);
        const auto old_pressure = limited_pressure;
        VoFStepDiagnostics limited_diagnostics;
        auto limited_evidence = ready;
        try
        {
            flow.project(limited_face, limited_velocity, limited_pressure, rho, fraction, curvature, .125, 0,
                limited_options, limited_diagnostics, &limited_evidence);
            FAIL() << "A single pressure iteration cannot resolve the square-graph defect.";
        }
        catch (const std::runtime_error& error)
        {
            EXPECT_NE(std::string(error.what()).find("residual="), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("transport_readiness_ratio="), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("transport_step_defect="), std::string::npos);
        }
        EXPECT_LE(limited_diagnostics.pressure_iterations, limited_options.max_pressure_iterations);
        EXPECT_EQ(limited_face, initial_face);
        EXPECT_EQ(limited_velocity, initial_velocity);
        EXPECT_EQ(limited_pressure, old_pressure);
        EXPECT_TRUE(limited_evidence.face_rate.empty());
    }
    ASSERT_NO_THROW(
        flow.project(face, velocity, pressure, rho, fraction, curvature, .125, 0, options, diagnostics, &ready));
    EXPECT_GT(diagnostics.pressure_iterations, physical_diagnostics.pressure_iterations);
    EXPECT_LE(diagnostics.maximum_divergence, options.divergence_tolerance);
    EXPECT_EQ(ready.closure_storage_bytes, 0u);
    for (const auto radius : ready.face_radius)
        EXPECT_EQ(radius, 0);
    Q global_mass_change = 0, global_momentum_change = 0;
    int local_old_failure = 0, old_failure = 0;
    const auto lease = mesh->acquire_execution_view();
    for (std::size_t c = 0; c < n; ++c)
    {
        Q net = 0, incident = 0, old_net = 0;
        for (auto f : mesh->faces(LO(c)))
        {
            EXPECT_LT(std::size_t(f), ready.face_rate.size());
            if (std::size_t(f) >= ready.face_rate.size())
                continue;
            const Q rate(ready.face_rate[f]);
            const auto sign = mesh->owner_cell(f) == LO(c) ? 1 : -1;
            net += sign * rate;
            old_net += sign * Q(physical.face_rate[f]);
            incident += abs(rate);
        }
        local_old_failure += abs(old_net) > Q(tau) * Q(mesh->cell_volume(LO(c)));
        const Q cell_volume(mesh->cell_volume(LO(c)));
        EXPECT_TRUE(abs(net) * (2 * Q(courant)) <= Q(tau) * incident ||
                    abs(net) * Q(.125) <= Q(tau) * cell_volume / 2);
        EXPECT_TRUE(incident > 0);
        if (incident == 0)
            continue;
        // Independently combine future CFL and canonical continuity bounds,
        // rather than reuse the projection step or its residual diagnostic.
        Q admitted_dt = Q(courant) * cell_volume / incident;
        if (net != 0)
        {
            const Q continuity_dt = Q(tau) * cell_volume / (2 * abs(net));
            if (continuity_dt < admitted_dt)
                admitted_dt = continuity_dt;
        }
        EXPECT_TRUE(abs(net) * admitted_dt <= Q(tau) * cell_volume / 2);
        // For constant density and constant carried velocity, paired canonical
        // streams conserve mass and momentum exactly over this closed graph.
        global_mass_change -= Q(.125) * Q(1000) * net;
        global_momentum_change -= Q(.125) * Q(1000) * Q(.375) * net;
    }
    EXPECT_EQ(global_mass_change, Q(0));
    EXPECT_EQ(global_momentum_change, Q(0));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_old_failure, &old_failure);
    EXPECT_GT(old_failure, 0);
    const int local_empty = n == 0;
    int empty = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, mesh->owned_cell_map()->getComm()->getSize() - 1);
}

TEST(GeneralVoFFlowWorkspaceTest, OrdinaryAlgebraicReadinessAcceptsZeroCirculationTree)
{
    // On a closed two-cell tree, continuity uniquely requires zero on its
    // single interior edge. Non-dyadic inputs exercise pressure-rounding tails;
    // physical acceptance must not demand an exactly represented zero rate.
    const auto mesh = axial_slab_native(2, 1);
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh);
    std::vector<VoF::Vector> velocity(n);
    std::vector<double> face(mesh->num_faces()), rho(n, 3), fraction(n), curvature(n), pressure;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
            face[f] = .1 * mesh->face_normal(LO(f)).x;
    face = flow.faces(face);
    VoFSolverOptions options;
    options.pressure_absolute_tolerance = 10;
    options.pressure_relative_tolerance = 0;
    options.divergence_tolerance = 1;
    options.max_courant = .25;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance evidence;
    constexpr double tau = 8e-15, dt = .13;
    evidence.transport_readiness = detail::GeneralVoFFiniteProjectionProvenance::TransportReadiness{
        tau, options.max_courant};
    bool accepted = true;
    try
    {
        flow.project(face, velocity, pressure, rho, fraction, curvature, dt, 0, options, diagnostics, &evidence);
    }
    catch (const std::runtime_error& error)
    {
        accepted = false;
        ADD_FAILURE() << "Zero-circulation tree must admit a physically resolved projection: " << error.what();
    }
    if (accepted)
    {
        EXPECT_GT(diagnostics.pressure_iterations, 0);
        EXPECT_EQ(evidence.closure_storage_bytes, 0u);
        EXPECT_LE(diagnostics.maximum_divergence * dt, tau / 2);
        for (const auto rate : evidence.face_rate)
        {
            EXPECT_LE(std::abs(rate) * dt, tau / 2);
            if (rate != 0)
            {
                using Q = boost::multiprecision::cpp_rational;
                const Q continuity_dt = Q(tau) / (2 * abs(Q(rate)));
                const Q courant_dt = Q(options.max_courant) / abs(Q(rate));
                EXPECT_LT(continuity_dt, courant_dt);
                EXPECT_GE(continuity_dt, Q(dt));
                EXPECT_LE(abs(Q(rate)) * continuity_dt, Q(tau) / 2);
            }
        }
    }
    const int local_empty = n == 0;
    int empty = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, mesh->owned_cell_map()->getComm()->getSize() - 1);
}

TEST(GeneralVoFFlowWorkspaceTest, OrdinaryAlgebraicReadinessRejectsRankLocalRequestsAndControls)
{
    const auto mesh = axial_slab_native(2);
    const auto comm = mesh->owned_cell_map()->getComm();
    const auto n = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace flow(mesh);
    std::vector<VoF::Vector> velocity(n);
    std::vector<double> face(mesh->num_faces()), rho(n, 1), fraction(n), curvature(n), pressure(n, 17);
    const auto old_velocity = velocity;
    const auto old_face = face, old_pressure = pressure;
    VoFSolverOptions options;
    VoFStepDiagnostics diagnostics;
    detail::GeneralVoFFiniteProjectionProvenance evidence;
    using Request = detail::GeneralVoFFiniteProjectionProvenance::TransportReadiness;
    for (unsigned mismatch = 0; mismatch < 3; ++mismatch)
    {
        evidence.transport_readiness = Request{8e-15, .25};
        if (comm->getSize() == 1)
        {
            if (mismatch == 0)
                evidence.transport_readiness->volume_tolerance = 0;
            else if (mismatch == 1)
                evidence.transport_readiness->max_courant = -1;
            else
                evidence.transport_readiness->max_courant = std::numeric_limits<double>::quiet_NaN();
        }
        else if (comm->getRank() + 1 == comm->getSize())
        {
            if (mismatch == 0)
                evidence.transport_readiness.reset();
            else if (mismatch == 1)
                evidence.transport_readiness->volume_tolerance *= 2;
            else
                evidence.transport_readiness->max_courant *= 2;
        }
        EXPECT_THROW(
            flow.project(face, velocity, pressure, rho, fraction, curvature, .125, 0, options, diagnostics, &evidence),
            std::runtime_error);
        EXPECT_EQ(face, old_face);
        EXPECT_EQ(velocity, old_velocity);
        EXPECT_EQ(pressure, old_pressure);
    }
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

namespace
{
SP<const MeshHandle<>> capillary_dependency_native(bool skew)
{
    // Five columns separate a constant cell adjacent to a nonconstant stencil
    // (column 2) from a cell whose entire two-ring neighborhood is constant (1).
    AMR::ExtrudedAMRGrid grid;
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned i = 0; i < 6; ++i)
            grid.xy_nodes.push_back({double(i) + (skew ? .3 * j : 0), double(j), 0});
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 5; ++i)
        {
            const auto a = j * 6 + i;
            grid.base_cells.push_back({a, a + 1, a + 7, a + 6});
        }
    for (unsigned i = 0; i < 5; ++i)
    {
        grid.boundaries.push_back({i, i + 1, "wall"});
        grid.boundaries.push_back({18 + i, 19 + i, "wall"});
    }
    for (unsigned j = 0; j < 3; ++j)
    {
        grid.boundaries.push_back({6 * j, 6 * (j + 1), "wall"});
        grid.boundaries.push_back({6 * j + 5, 6 * (j + 1) + 5, "wall"});
    }
    grid.z_edges = {0, 1, 2, 3};
    AMR::ExtrudedAMR source(grid);
    std::vector<std::uint64_t> owned;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t c = 0; c < 45; ++c)
            owned.push_back(c);
    auto bootstrap = source.detach_owned(owned, {'c', 'a', 'p', 'i', 'l', 'l', 'a', 'r', 'y'});
    const auto schema = bootstrap.kernel->descriptor();
    const AMR::DistributedAdaptiveHierarchy hierarchy(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    return AMR::assemble_adaptive_native(hierarchy, schema).mesh;
}

TEST(GeneralVoFFlowWorkspaceTest, CapillaryExtensionMatchesOrthogonalAndSkewSupportIncludingEmptyOwners)
{
    constexpr std::size_t budget = 64 * 1024 * 1024;
    for (const bool skew : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "skew=" << skew);
        const auto mesh = capillary_dependency_native(skew);
        detail::GeneralVoFFlowWorkspace flow(mesh);
        std::vector<double> fraction(mesh->num_owned_cells());
        std::vector<unsigned char> expected(fraction.size());
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            const auto center = mesh->cell_centroid(LO(c));
            const double logical_x = center.x - (skew ? .3 * center.y : 0);
            fraction[c] = logical_x > 4 ? 1 : 0;
            // Orthogonal reconstructions have no neighbor gradient correction.
            // Sheared column faces transmit that correction one column farther.
            expected[c] = logical_x > (skew ? 2 : 3);
        }
        const auto original = fraction;
        EXPECT_EQ(flow.capillary_extension_required(fraction, budget), expected);
        EXPECT_EQ(fraction, original);
        for (auto& value : fraction)
            value = 1 - value;
        EXPECT_EQ(flow.capillary_extension_required(fraction, budget), expected);
        const int local_empty = fraction.empty();
        int empty = 0;
        const auto comm = mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(empty, comm->getSize() - 1);
    }
}

TEST(GeneralVoFFlowWorkspaceTest, CapillaryExtensionRetainsTracesAndRejectsInvalidInputsCollectively)
{
    constexpr std::size_t budget = 64 * 1024 * 1024;
    const auto mesh = skew_native(1);
    detail::GeneralVoFFlowWorkspace flow(mesh);
    const auto count = mesh->num_owned_cells();
    for (const double value : {0., 1.})
        EXPECT_EQ(flow.capillary_extension_required(std::vector<double>(count, value), budget),
            std::vector<unsigned char>(count, 0));
    for (const double trace : {std::numeric_limits<double>::denorm_min(), 1e-25, .5})
    {
        const std::vector<double> fraction(count, trace);
        EXPECT_EQ(flow.capillary_extension_required(fraction, budget), std::vector<unsigned char>(count, 1));
        for (auto value : fraction)
            EXPECT_EQ(value, trace);
    }
    std::vector<double> fraction(count);
    EXPECT_THROW((void) flow.capillary_extension_required(fraction, 1), std::runtime_error);
    auto invalid_shape = fraction;
    if (!invalid_shape.empty())
        invalid_shape.pop_back();
    EXPECT_THROW((void) flow.capillary_extension_required(invalid_shape, budget), std::runtime_error);
    for (const double invalid :
        {-1., 1.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto bad = fraction;
        if (!bad.empty())
            bad[0] = invalid;
        EXPECT_THROW((void) flow.capillary_extension_required(bad, budget), std::runtime_error);
    }
    detail::GeneralVoFFlowWorkspace bounded(mesh, ArithmeticMode::BoundedCertification);
    EXPECT_THROW((void) bounded.capillary_extension_required(fraction, budget), std::runtime_error);
    EXPECT_EQ(flow.capillary_extension_required(fraction, budget), std::vector<unsigned char>(count, 0));
}

TEST(GeneralVoFFlowWorkspaceTest, CapillaryExtensionInactiveCurvatureLeavesPhysicalProjectionIdentical)
{
    constexpr std::size_t budget = 64 * 1024 * 1024;
    const auto mesh = capillary_dependency_native(false);
    const auto count = mesh->num_owned_cells();
    detail::GeneralVoFFlowWorkspace baseline(mesh), changed(mesh);
    std::vector<double> fraction(count), density(count), curvature(count, 1), changed_curvature;
    for (std::size_t c = 0; c < count; ++c)
    {
        const auto center = mesh->cell_centroid(LO(c));
        fraction[c] = center.x > 4 ? 1 : 0;
        density[c] = 1 + center.x;
    }
    const auto required = baseline.capillary_extension_required(fraction, budget);
    changed_curvature = curvature;
    int local_changed = 0, global_changed = 0;
    for (std::size_t c = 0; c < count; ++c)
    {
        const auto center = mesh->cell_centroid(LO(c));
        // This column is locally constant but adjacent to a cell with a
        // nonconstant fraction stencil. Its orthogonal operator still has no
        // coefficient for this curvature, even with unequal adjacent densities.
        if (center.x > 2 && center.x < 3)
        {
            EXPECT_EQ(required[c], 0);
            changed_curvature[c] = 37;
            ++local_changed;
        }
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_changed, &global_changed);
    EXPECT_EQ(global_changed, 9);
    std::vector<double> base_face(mesh->num_faces()), changed_face = base_face, base_pressure, changed_pressure;
    std::vector<VoF::Vector> base_velocity(count), changed_velocity = base_velocity;
    VoFSolverOptions options;
    VoFStepDiagnostics base_diagnostics, changed_diagnostics;
    ASSERT_NO_THROW(baseline.project(
        base_face, base_velocity, base_pressure, density, fraction, curvature, .01, .05, options, base_diagnostics));
    ASSERT_NO_THROW(changed.project(changed_face, changed_velocity, changed_pressure, density, fraction,
        changed_curvature, .01, .05, options, changed_diagnostics));
    EXPECT_EQ(changed_face, base_face);
    EXPECT_EQ(changed_pressure, base_pressure);
    EXPECT_EQ(changed_velocity, base_velocity);
    EXPECT_EQ(changed_diagnostics.maximum_divergence, base_diagnostics.maximum_divergence);
}
} // namespace
