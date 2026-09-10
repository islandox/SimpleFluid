#include <gtest/gtest.h>
#include "equations/turbulence/TurbulenceModel.hh"
#include "equations/CollectiveValidation.hh"
#include "FVM/CellOperators.hh"
#include "FVM/VectorLaplacian.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "parallel/MeshPartitioner.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Native = MeshHandle<Pack>;
using Legacy = Mesh<Pack>;
auto* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const Native> native_mesh(size_t n = 4)
{
    ArrReal edges(n + 1);
    for (size_t i = 0; i <= n; ++i) edges[i] = real_t(i) / n;
    return std::make_shared<Native>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{edges, edges, edges}}));
}

TurbulenceModelOptions options(bool enabled = true)
{
    TurbulenceModelOptions o;
    o.model = TurbulenceModelType::SSTKOmegaSAS;
    o.sas.enabled = enabled; o.sas.diagnostics = true;
    o.initial_wall_distance = 0.25;
    o.initial_turbulent_kinetic_energy = 0.2;
    o.initial_specific_dissipation_rate = 2;
    return o;
}

BoundaryConditionSet prescribed_boundaries()
{
    BoundaryConditionSet bc;
    for (auto name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        bc.velocity[name] = {BoundaryConditionType::Dirichlet, {}};
    return bc;
}

template<class M> BoundaryConditionSet prescribed_boundaries(const M& mesh)
{
    BoundaryConditionSet result;
    for (const auto& [batch, faces] : mesh.boundary_batches())
        result.velocity[mesh.boundary_batch_name(batch)] = {BoundaryConditionType::Dirichlet, {}};
    return result;
}

template<class M> MaterialPropertyFields<Pack, M> material(SP<const M> mesh, double rho = 1)
{
    BoussinesqModelOptions o;
    o.reference_density = rho; o.density = rho; o.dynamic_viscosity = rho*1e-3;
    o.specific_heat_capacity = 1; o.thermal_conductivity = 0.01;
    return {mesh, o, TimeStepperOptions{}};
}

template<class M> void model_contract(SP<const M> mesh)
{
    using Traits = MeshFieldTraits<Pack, M>;
    using Model = TurbulenceModel<Pack, M>;
    auto mat = material(mesh);
    auto bc = prescribed_boundaries(*mesh);
    Model sas(mesh, bc), parent(mesh, bc), disabled(mesh, bc);
    auto config = options();
    sas.configure(config, mat, 1);
    disabled.configure(options(false), mat, 1);
    config.model = TurbulenceModelType::SSTKOmega;
    parent.configure(config, mat, 1);
    typename Traits::vector_cell_type u(mesh, "velocity");
    typename Traits::scalar_face_type flux(mesh, 0., "flux");
    auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, bc);
    for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
    {
        const auto y = mesh->cell_centroid(i).y;
        u.set_owned_value(i, {y*y, 0, 0});
    }
    u.sync_ghosts();
    for (const auto& [batch, faces] : mesh->boundary_batches())
        for (size_t i = 0; i < faces.face_lids.size(); ++i)
        {
            const auto y = mesh->face_centroid(faces.face_lids[i]).y;
            cache.value.at(batch)[i] = {y*y, 0, 0};
        }
    auto advance = [&](Model& model, double dt = 0.001, const LinearSolverOptions& linear = LinearSolverOptions{})
    {
        return model.advance(u, flux, cache, dt, mat, 1, FVM::NonOrthogonalTreatment::Explicit, linear);
    };
    for (int step = 0; step < 2; ++step)
    {
        ASSERT_TRUE(advance(parent).converged);
        ASSERT_TRUE(advance(disabled).converged);
        for (auto name : {"k", "omega", "nu_t", "mu_eff", "lambda_eff"})
            for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
                EXPECT_DOUBLE_EQ(parent.output_fields().at(name)->value(i), disabled.output_fields().at(name)->value(i));
    }
    EXPECT_EQ(disabled.output_fields().size(), 6U);
    EXPECT_FALSE(sas.sas_statistics().valid);
    ASSERT_TRUE(advance(sas).converged);
    EXPECT_TRUE(sas.sas_statistics().valid);
    EXPECT_GT(sas.sas_statistics().active_cell_fraction, 0.5);
    EXPECT_GT(sas.sas_statistics().max_source, 0.01); // Cannot pass with a zero implementation.
    EXPECT_EQ(sas.output_fields().size(), 17U);
    const auto accepted = sas.snapshot();
    const auto saved_stats = sas.sas_statistics();
    std::vector<double> k, q;
    for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
    {
        k.push_back(sas.turbulent_kinetic_energy().value(i));
        q.push_back(sas.output_fields().at("sas_Q_applied")->value(i));
    }
    EXPECT_ANY_THROW(advance(sas, 0));
    EXPECT_ANY_THROW(sas.validate_time_mode(false));
    // Force transport rejection after closure and diagnostic staging.
    LinearSolverOptions bad; bad.max_iterations = 0;
    EXPECT_ANY_THROW(advance(sas, 0.1, bad));
    EXPECT_DOUBLE_EQ(sas.sas_statistics().max_source, saved_stats.max_source);
    for (size_t i = 0; i < k.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(sas.turbulent_kinetic_energy().value(i), k[i]);
        EXPECT_DOUBLE_EQ(sas.output_fields().at("sas_Q_applied")->value(i), q[i]);
    }
    ASSERT_TRUE(advance(sas).converged);
    sas.restore(accepted);
    for (size_t i = 0; i < k.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(sas.turbulent_kinetic_energy().value(i), k[i]);
        EXPECT_DOUBLE_EQ(sas.output_fields().at("sas_Q_applied")->value(i), q[i]);
    }
    EXPECT_DOUBLE_EQ(sas.sas_statistics().max_source, saved_stats.max_source);
    // A field-only restart has no evidence of a previously assembled source.
    sas.restore_transported_state(sas.turbulent_kinetic_energy(), *sas.specific_dissipation_rate(),
        sas.turbulent_kinematic_viscosity(), u, mat, 1);
    EXPECT_FALSE(sas.sas_statistics().valid);
    auto no_diagnostics = options(); no_diagnostics.sas.diagnostics = false;
    sas.configure(no_diagnostics, mat, 1);
    EXPECT_EQ(sas.output_fields().size(), 6U);
    ASSERT_TRUE(advance(sas).converged);
    EXPECT_GT(sas.sas_statistics().max_source, .01);
    EXPECT_ANY_THROW(sas.restore(accepted));
    sas.disable();
    EXPECT_TRUE(sas.output_fields().empty());
}

TEST(SSTSASModelTest, NativeTransactionsActivationAndDisabledEquivalence) { model_contract(native_mesh()); }
TEST(SSTSASModelTest, LegacyTransactionsActivationAndDisabledEquivalence)
{
    model_contract<Legacy>(test::build_mesh<Pack>(test::make_box_database(4, 4, 4, 0.25)));
}

TEST(SSTSASModelTest, RankLocalInvalidStateIsCollective)
{
    auto mesh = native_mesh(); auto mat = material(mesh);
    const auto bc = prescribed_boundaries();
    TurbulenceModel<Pack, Native> model(mesh, bc); model.configure(options(), mat, 1);
    VectorCellFieldStored<Pack> u(mesh, "u"); ScalarFaceFieldStored<Pack> flux(mesh, 0., "flux");
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, bc);
    if (mesh->owned_cell_map()->getComm()->getRank() == 0)
        u.set_owned_value(0, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
    u.sync_ghosts();
    EXPECT_ANY_THROW(model.advance(u, flux, cache, .001, mat, 1, FVM::NonOrthogonalTreatment::Explicit));
    EXPECT_FALSE(model.sas_statistics().valid);
    for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        EXPECT_DOUBLE_EQ(model.turbulent_kinetic_energy().value(i), .2);
    auto invalid = options();
    if (mesh->owned_cell_map()->getComm()->getRank() == 0) invalid.sas.cs = 0;
    EXPECT_ANY_THROW(model.configure(invalid, mat, 1));
}

TEST(SSTSASModelTest, OneCellSourceUnitsVolumeAndDensity)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "One-cell serial assembly fixture";
    for (const auto size : {0.5, 2.0})
        for (const auto rho : {1., 1000.})
          for (const double dt : {.001, 1.0})
        {
            SP<const Legacy> mesh = test::build_mesh<Pack>(test::make_box_database(1, 1, 1, size));
            auto mat = material(mesh, rho); auto bc = prescribed_boundaries();
            TurbulenceModel<Pack> model(mesh, bc); model.configure(options(), mat, rho);
            VectorCellField<Pack> u(mesh, "u"); u.set_owned_value(0, {0.25, 0, 0}); u.sync_ghosts();
            FaceField<Pack> flux(mesh, 0., "flux");
            auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, bc);
            for (const auto& [batch, faces] : mesh->boundary_batches())
                for (size_t i = 0; i < faces.face_lids.size(); ++i)
                {
                    const auto y = mesh->face_centroid(faces.face_lids[i]).y / size;
                    cache.value.at(batch)[i] = {y*y, 0, 0};
                }
            model.advance(u, flux, cache, dt, mat, rho, FVM::NonOrthogonalTreatment::Explicit);
            const SSTKOmegaEquation parent;
            const auto f1 = parent.blending_function_1({.2, 2}, {.001, .25, 0, 1/size});
            const auto blend = parent.blended_coefficients(f1);
            const auto q = SSTSASSource{}.evaluate({{.2, 2}, 1/(size*size), 1/(size*size),
                0, 0, size, dt, .09, blend.beta, blend.gamma, .41}).Q_applied;
            EXPECT_GT(q, 0);
            EXPECT_EQ(model.sas_statistics().cap_active_cell_fraction, size == .5 && dt == 1.0 ? 1. : 0.);
            EXPECT_NEAR(model.output_fields().at("sas_Q_applied")->value(0), q, 1e-13);
            const double expected = (2 + dt*(blend.gamma/(size*size)+q))/(1+dt*blend.beta*2);
            EXPECT_NEAR(model.specific_dissipation_rate()->value(0), expected, 2e-12);
        }
}

template<class M> void derivative_contract(SP<const M> mesh)
{
    using Traits = MeshFieldTraits<Pack, M>;
    typename Traits::vector_cell_type u(mesh, "u"), lap(mesh, "lap"), kg(mesh, "kg");
    typename Traits::tensor_cell_type gradient(mesh, "grad");
    typename Traits::scalar_cell_type scalar(mesh, "k");
    FVM::CellGradientCache<Pack, M> gradient_cache(mesh);
    FVM::TransportGeometryCache<M> geometry(*mesh);
    auto boundary_type = [](int, size_t) { return BoundaryConditionType::Dirichlet; };
    // Both affine shear and rigid rotation, with a velocity offset and a 90-degree coordinate rotation.
    for (int mode = 0; mode < 4; ++mode)
    {
        auto value = [mode](auto p) -> typename Traits::vector_cell_type::vec_type
        {
            if (mode == 0) return {3., -2., 7.};
            if (mode == 1) return {p.y+3, -2., 7.};
            if (mode == 2) return {-p.y+3, p.x-2, 7.};
            return {3., p.x-2, 7.};
        };
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            u.set_owned_value(i, value(mesh->cell_centroid(i)));
            const auto p = mesh->cell_centroid(i); scalar.set_owned_value(i, 4 + p.x + 2*p.y - p.z);
        }
        u.sync_ghosts(); scalar.sync_ghosts();
        auto boundary_value = [&](int b, size_t i) { return value(mesh->face_centroid(mesh->boundary_batches().at(b).face_lids[i])); };
        FVM::cell_gradient(u, boundary_value, gradient, gradient_cache); gradient.sync_ghosts();
        collective_detail::collective_local_validation(*mesh, "Manufactured vector Laplacian", [&]
        { FVM::unit_vector_laplacian(u, gradient, lap, geometry, boundary_value, boundary_type); });
        auto scalar_bc = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 0}; };
        auto scalar_value = [&](int b, size_t i)
        { const auto p=mesh->face_centroid(mesh->boundary_batches().at(b).face_lids[i]); return 4+p.x+2*p.y-p.z; };
        FVM::cell_gradient(scalar, scalar_bc, scalar_value, kg, gradient_cache);
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            const auto v = lap.value(i); EXPECT_LT(std::hypot(v.x, v.y, v.z), 2e-11);
            const auto g = kg.value(i); EXPECT_NEAR(g.x, 1, 1e-12); EXPECT_NEAR(g.y, 2, 1e-12); EXPECT_NEAR(g.z, -1, 1e-12);
        }
    }
    // Known vector Laplacian, not Laplacian(speed) or a Hessian norm.
    auto value = [](auto p) -> typename Traits::vector_cell_type::vec_type { return {p.y*p.y, 2*p.z*p.z, -3*p.x*p.x}; };
    for (size_t i = 0; i < mesh->num_owned_cells(); ++i) u.set_owned_value(i, value(mesh->cell_centroid(i)));
    u.sync_ghosts();
    auto boundary_value = [&](int b, size_t i) { return value(mesh->face_centroid(mesh->boundary_batches().at(b).face_lids[i])); };
    FVM::cell_gradient(u, boundary_value, gradient, gradient_cache); gradient.sync_ghosts();
    collective_detail::collective_local_validation(*mesh, "Quadratic vector Laplacian", [&]
    { FVM::unit_vector_laplacian(u, gradient, lap, geometry, boundary_value, boundary_type); });
    // Orthogonal interior cells have exact quadratic differences. Boundary flux is first order.
    if (!geometry.has_non_orthogonal_faces())
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            bool interior = true;
            for (auto f : mesh->faces(i)) if (mesh->is_boundary_face(f)) interior = false;
            if (!interior) continue;
            auto v = lap.value(i); EXPECT_NEAR(v.x, 2, 1e-12); EXPECT_NEAR(v.y, 4, 1e-12); EXPECT_NEAR(v.z, -6, 1e-12);
        }
}

TEST(SSTSASDerivativesTest, NativeOrthogonalPartitionFaces) { derivative_contract(native_mesh()); }
TEST(SSTSASDerivativesTest, LegacyOrthogonalPartitionFaces)
{ derivative_contract<Legacy>(test::build_mesh<Pack>(test::make_box_database(4,4,4,.25))); }
TEST(SSTSASDerivativesTest, SkewedPrismAffineAndScalarGradients)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Serial skew fixture";
    auto mesh = test::make_skewed_prism_mesh<Pack>();
    derivative_contract<Legacy>(mesh);
    derivative_contract<Native>(std::make_shared<Native>(mesh));
}
} // namespace

TEST(SSTSASModelTest, AutomaticDistanceAndResolvedSSTCoefficients)
{
    auto mesh = native_mesh(); auto mat = material(mesh);
    BoundaryConditionSet bc;
    for (auto name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        bc.velocity[name] = {BoundaryConditionType::NoSlip, {}};
    auto config = options(); config.initial_wall_distance.reset();
    config.wall_treatment = TurbulenceWallTreatmentType::ResolvedLowReSST;
    config.wall_options.boundary_names = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    config.wall_options.kappa = .39; config.wall_options.sst_beta_1 = .08;
    TurbulenceModel<Pack, Native> model(mesh, bc);
    ASSERT_NO_THROW(model.configure(config, mat, 1));
    EXPECT_TRUE(model.output_fields().contains("wall_y_plus"));
    for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        EXPECT_GT(model.output_fields().at("wall_distance")->value(i), 0);
    VectorCellFieldStored<Pack> u(mesh, "u"); ScalarFaceFieldStored<Pack> flux(mesh, 0., "flux");
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, bc);
    EXPECT_TRUE(model.advance(u, flux, cache, .001, mat, 1, FVM::NonOrthogonalTreatment::Explicit).converged);
    EXPECT_DOUBLE_EQ(model.sas_statistics().max_source, 0);
}

TEST(SSTSASModelTest, NonzeroSourceIsInvariantUnderOffsetsAndCoordinateRotation)
{
    auto mesh = native_mesh(); auto mat = material(mesh); auto bc = prescribed_boundaries();
    ScalarFaceFieldStored<Pack> flux(mesh, 0., "flux");
    double reference = 0;
    for (int mode = 0; mode < 3; ++mode)
    {
        TurbulenceModel<Pack, Native> model(mesh, bc); model.configure(options(), mat, 1);
        VectorCellFieldStored<Pack> u(mesh, "u");
        auto value = [mode](auto p) -> vec3<double>
        {
            if (mode == 0) return {p.y*p.y, 0, 0};
            if (mode == 1) return {p.y*p.y + 4, -7, 2};
            return {0, (1-p.x)*(1-p.x), 0};
        };
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i) u.set_owned_value(i, value(mesh->cell_centroid(i)));
        u.sync_ghosts();
        for (const auto face : flux.owned_face_ids())
            flux.set_owned_value(face, value(mesh->face_centroid(face)).dot(
                mesh->face_area_vector_outward(face, mesh->owner_cell(face))));
        flux.sync_ghosts();
        auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, bc);
        for (const auto& [batch, faces] : mesh->boundary_batches())
            for (size_t i = 0; i < faces.face_lids.size(); ++i) cache.value.at(batch)[i] = value(mesh->face_centroid(faces.face_lids[i]));
        model.advance(u, flux, cache, .001, mat, 1, FVM::NonOrthogonalTreatment::Explicit);
        const auto result = model.sas_statistics().volume_integrated_source;
        EXPECT_GT(result, .01);
        if (mode == 0) reference = result;
        else EXPECT_NEAR(result, reference, 1e-12);
    }
}

TEST(SSTSASDerivativesTest, SkewedQuadraticCurvatureRefines)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Serial manufactured refinement";
    double previous = 0;
    for (const size_t n : {4, 8})
    {
        SP<const Legacy> mesh = test::make_skewed_prism_mesh<Pack>(n, n, n);
        VectorCellField<Pack> u(mesh, "u"), lap(mesh, "lap"); TensorCellField<Pack> g(mesh, "g");
        auto value = [](auto p) -> vec3<double> { return {p.y*p.y, 2*p.z*p.z, -3*p.x*p.x}; };
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i) u.set_owned_value(i, value(mesh->cell_centroid(i)));
        u.sync_ghosts();
        auto boundary = [&](int b, size_t i) { return value(mesh->face_centroid(mesh->boundary_batches().at(b).face_lids[i])); };
        FVM::CellGradientCache<Pack> gc(mesh); FVM::TransportGeometryCache<Legacy> geometry(*mesh);
        FVM::cell_gradient(u, boundary, g, gc); g.sync_ghosts();
        FVM::unit_vector_laplacian(u, g, lap, geometry, boundary, [](int, size_t) { return BoundaryConditionType::Dirichlet; });
        double error = 0, volume = 0;
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            const auto l = lap.value(i); const auto v = mesh->cell_volume(i);
            error += v*std::hypot(l.x-2, l.y-4, l.z+6); volume += v;
        }
        error /= volume;
        std::cout << "SAS manufactured skew n=" << n << " L1=" << error << '\n';
        if (n == 8) EXPECT_LT(error, previous * .8);
        previous = error;
    }
}

TEST(SSTSASDerivativesTest, NativePartitionedUnstructuredSkewedFaces)
{
    using Unstructured = Meshes::UnstructuredMesh;
    using Partitioned = Meshes::PartitionedMesh<Unstructured, Pack>;
    const auto original = test::make_unstructured_hex_line(8, .125);
    auto nodes = original->nodes();
    for (auto& p : nodes) p.x += .3*p.y;
    Arr<Unstructured::CellDefinition> cells;
    for (size_t i = 0; i < original->num_cells(); ++i)
    {
        auto cell = original->cell_id(i); cells.push_back({original->cell_type(cell), original->cell_nodes(cell)});
    }
    Arr<Unstructured::BoundaryFaceDefinition> boundaries;
    for (const auto& [batch, faces] : original->boundary_batches())
        for (auto f : faces.face_lids) boundaries.push_back({original->face_nodes(f), batch, original->boundary_names().at(batch)});
    auto geometry = std::make_shared<Unstructured>(nodes, cells, boundaries);
    const auto comm = Tpetra::getDefaultComm();
    auto partition = MeshPartitioner<Pack>::partition(*geometry, comm);
    auto distributed = std::make_shared<Partitioned>(geometry, std::move(partition.indexer), comm);
    SP<const Native> mesh = std::make_shared<Native>(distributed);
    ASSERT_FALSE(mesh->legacy_mesh());
    derivative_contract(mesh);
    model_contract(mesh);
}

TEST(SSTSASModelTest, RejectsUnverifiedPeriodicAndStaleGeometry)
{
    auto cartesian = std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{{
        {0., .25, .5, .75, 1.}, {0., .25, .5, .75, 1.}, {0., .25, .5, .75, 1.}}});
    auto mutable_mesh = std::make_shared<Native>(cartesian);
    SP<const Native> mesh = mutable_mesh; auto mat = material(mesh);
    auto bc = prescribed_boundaries(); bc.velocity["xmin"].type = BoundaryConditionType::Periodic;
    TurbulenceModel<Pack, Native> slip(mesh, bc);
    EXPECT_THROW(slip.configure(options(), mat, 1), std::invalid_argument);
    TurbulenceModel<Pack, Native> model(mesh, {}); model.configure(options(), mat, 1);
    const auto snapshot = model.snapshot();
    VectorCellFieldStored<Pack> u(mesh, "u"); ScalarFaceFieldStored<Pack> flux(mesh, 0., "flux");
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, BoundaryConditionSet{});
    PlanarALEMeshMotion<Pack> motion(mutable_mesh);
    motion.begin_trial(1.1, .001);
    EXPECT_ANY_THROW(model.advance(u, flux, cache, .001, mat, 1, FVM::NonOrthogonalTreatment::Explicit));
    EXPECT_ANY_THROW(model.restore(snapshot));
    EXPECT_FALSE(model.sas_statistics().valid);
    motion.rollback_trial();


}

namespace
{
SP<const Native> cylindrical_mesh(size_t n, bool closed)
{
    ArrReal radial(n+1), axial(n+1), theta(4*n+1);
    for (size_t i=0; i<=n; ++i) { radial[i]=1.+double(i)/n; axial[i]=double(i)/n; }
    const double span=closed ? 2*std::acos(-1.) : 1.5;
    for (size_t i=0; i<theta.size(); ++i) theta[i]=span*i/(theta.size()-1);
    return std::make_shared<Native>(std::make_shared<Meshes::OrthogonalCylindrial3D>(Vec3D<ArrReal>{{radial,theta,axial}}));
}

template<class M> double quadratic_laplacian_error(SP<const M> mesh)
{
    using Traits=MeshFieldTraits<Pack,M>;
    typename Traits::vector_cell_type u(mesh,"quadratic_u"), lap(mesh,"quadratic_lap");
    typename Traits::tensor_cell_type grad(mesh,"quadratic_grad");
    auto value=[](auto p)->vec3<double> { return {p.y*p.y,2*p.z*p.z,-3*p.x*p.x}; };
    for (size_t i=0; i<mesh->num_owned_cells(); ++i) u.set_owned_value(i,value(mesh->cell_centroid(i)));
    u.sync_ghosts();
    auto boundary=[&](int b,size_t i) { return value(mesh->face_centroid(mesh->boundary_batches().at(b).face_lids[i])); };
    FVM::CellGradientCache<Pack,M> gradient_cache(mesh);
    FVM::TransportGeometryCache<M> geometry(*mesh);
    FVM::cell_gradient(u,boundary,grad,gradient_cache); grad.sync_ghosts();
    collective_detail::collective_local_validation(*mesh,"Manufactured SAS curvature",[&]
    { FVM::unit_vector_laplacian(u,grad,lap,geometry,boundary,[](int,size_t) { return BoundaryConditionType::Dirichlet; }); });
    std::array<double,2> local{},total{};
    for (size_t i=0; i<mesh->num_owned_cells(); ++i)
    {
        const auto v=lap.value(i); const auto volume=mesh->cell_volume(i);
        local[0]+=volume*std::hypot(v.x-2,v.y-4,v.z+6); local[1]+=volume;
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(),Teuchos::REDUCE_SUM,2,local.data(),total.data());
    return total[0]/total[1];
}
}

TEST(SSTSASDerivativesTest, CylindricalSectorsAndClosedAnnuli)
{
    for (bool closed : {false,true})
    {
        auto mesh=cylindrical_mesh(4,closed);
        ASSERT_FALSE(mesh->legacy_mesh());
        derivative_contract(mesh); // Uniform, affine, rotation and scalar gradient checks.
        const auto coarse=quadratic_laplacian_error(mesh);
        const auto fine=quadratic_laplacian_error(cylindrical_mesh(8,closed));
        std::cout<<"SAS cylindrical closed="<<closed<<" L1="<<coarse<<" -> "<<fine<<'\n';
        EXPECT_LT(fine,.8*coarse);
        model_contract(mesh); // Nonzero source, disabled equivalence, snapshots and restart.
    }
}

namespace
{
SP<const Native> semi_structured_mesh(size_t n)
{
    using Semi=Meshes::SemiStructuredXY_Z;
    Arr<Semi::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    ArrReal z(n+1);
    for (size_t j=0; j<=n; ++j)
        for (size_t i=0; i<=n; ++i) nodes.push_back({(i+.3*j)/n,double(j)/n,0});
    for (size_t j=0; j<n; ++j)
        for (size_t i=0; i<n; ++i)
        {
            const auto a=static_cast<unsigned>(i+(n+1)*j), b=a+1, c=a+static_cast<unsigned>(n+1), d=c+1;
            cells.push_back({a,b,d}); cells.push_back({a,d,c});
        }
    for (size_t i=0; i<=n; ++i) z[i]=double(i)/n;
    return std::make_shared<Native>(std::make_shared<Semi>(nodes,cells,z));
}
}

TEST(SSTSASDerivativesTest, SemiStructuredSkewedExtrusion)
{
    if (Tpetra::getDefaultComm()->getSize()!=1) GTEST_SKIP()<<"MeshHandle semi-structured ownership remains serial-only";
    const auto mesh=semi_structured_mesh(4);
    ASSERT_FALSE(mesh->legacy_mesh());
    derivative_contract(mesh);
    const auto coarse=quadratic_laplacian_error(mesh);
    const auto fine=quadratic_laplacian_error(semi_structured_mesh(8));
    std::cout<<"SAS semi-structured L1="<<coarse<<" -> "<<fine<<'\n';
    EXPECT_LT(fine,.8*coarse);
    model_contract(mesh);
}

namespace
{
template<class M> void slip_derivative_contract(SP<const M> mesh)
{
    using Traits=MeshFieldTraits<Pack,M>;
    auto bc=prescribed_boundaries(*mesh); bc.velocity["zmin"]={BoundaryConditionType::Slip,{}};
    auto mat=material(mesh);
    for (int mode=0; mode<3; ++mode)
    {
        TurbulenceModel<Pack,M> model(mesh,bc); model.configure(options(),mat,1);
        typename Traits::vector_cell_type u(mesh,"slip_u");
        typename Traits::scalar_face_type phi(mesh,0.,"slip_phi");
        auto value=[mode](auto p)->vec3<double>
        { return mode==0 ? vec3<double>{p.y+3,0,0} : mode==1 ? vec3<double>{0,0,p.z} : vec3<double>{p.y*p.y,0,0}; };
        for (size_t i=0; i<mesh->num_owned_cells(); ++i) u.set_owned_value(i,value(mesh->cell_centroid(i)));
        u.sync_ghosts();
        auto cache=FVM::cache_velocity_boundary_conditions<Pack>(mesh,bc);
        for (const auto& [batch,faces] : mesh->boundary_batches())
            for (size_t i=0; i<faces.face_lids.size(); ++i)
                cache.value.at(batch)[i]=value(mesh->face_centroid(faces.face_lids[i]));
        EXPECT_TRUE(model.advance(u,phi,cache,.001,mat,1,FVM::NonOrthogonalTreatment::Explicit).converged);
        if (mode<2) EXPECT_LT(model.sas_statistics().max_source,1e-20);
        else EXPECT_GT(model.sas_statistics().max_source,1e-4);
    }
}
}

TEST(SSTSASDerivativesTest, SlipMixedConditionPreservesAffineShearAndNormalFlux)
{
    slip_derivative_contract(native_mesh());
    slip_derivative_contract<Legacy>(test::build_mesh<Pack>(test::make_box_database(4,4,4,.25)));
    if (Tpetra::getDefaultComm()->getSize()==1)
    {
        auto skew=test::make_skewed_prism_mesh<Pack>();
        slip_derivative_contract<Legacy>(skew);
        slip_derivative_contract<Native>(std::make_shared<Native>(skew));
    }
}
