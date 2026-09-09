/** @file testCoupledBlockOperator.cc @brief Composite algebra, ownership and distributed solves. */
#include "FVM/FaceFlux.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/TimeStepperOptions.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "solvers/CoupledPressureVelocitySolver.hh"
#include "utils/testing_environment.hh"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using MV = Pack::multi_vector_type;
using Matrix = Pack::matrix_type;
using Map = Pack::map_type;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Handle = MeshHandle<Pack>;
using Solver = CoupledPressureVelocitySolver<Pack, Handle>;
using Backend = CoupledOperatorBackend;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void fill(MV& x)
{
    for (size_t j = 0; j < x.getNumVectors(); ++j)
    {
        auto column = x.getVectorNonConst(j);
        auto data = column->getDataNonConst();
        for (size_t row = 0; row < data.size(); ++row)
            data[row] = std::sin(0.13 * (x.getMap()->getGlobalElement(static_cast<LO>(row)) + 1) + 0.7 * j);
    }
}

void expect_near(const MV& a, const MV& b, double scale = 1.0)
{
    ASSERT_EQ(a.getNumVectors(), b.getNumVectors());
    for (size_t j = 0; j < a.getNumVectors(); ++j)
    {
        const auto av = a.getVector(j)->getData(), bv = b.getVector(j)->getData();
        for (size_t row = 0; row < av.size(); ++row)
            EXPECT_NEAR(av[row], bv[row],
                512 * std::numeric_limits<double>::epsilon() *
                    std::max({1.0, scale, std::abs(av[row]), std::abs(bv[row])}));
    }
}

struct AlgebraFixture
{
    Teuchos::RCP<const Map> cells, coupled;
    Teuchos::RCP<Matrix> momentum, stabilization, assembled;
    std::array<Teuchos::RCP<Matrix>, 3> gradient, divergence;
    std::unique_ptr<CoupledBlockOperator<Pack>> composite;

    explicit AlgebraFixture(bool pin)
    {
        const auto comm = Tpetra::getDefaultComm();
        const int rank = comm->getRank(), ranks = comm->getSize();
        // Noncontiguous global IDs and reversed rank ownership: gauge on last rank.
        Teuchos::Array<GO> ids{10 + 4 * (ranks - rank - 1), 12 + 4 * (ranks - rank - 1)};
        cells = Teuchos::rcp(new Map(2 * ranks, ids(), 0, comm));
        Teuchos::Array<GO> block_ids;
        for (auto gid : ids)
            for (int c = 0; c < 4; ++c)
                block_ids.push_back(4 * gid + c);
        coupled = Teuchos::rcp(new Map(8 * ranks, block_ids(), 0, comm));
        auto make = [&](double diagonal, double neighbor)
        {
            auto m = Teuchos::rcp(new Matrix(cells, 2));
            for (auto gid : ids)
            {
                GO next = 10 + 2 * (((gid - 10) / 2 + 1) % (2 * ranks));
                Teuchos::Array<GO> cols{gid, next};
                Teuchos::Array<double> vals{diagonal, neighbor};
                m->insertGlobalValues(gid, cols(), vals());
            }
            m->fillComplete();
            return m;
        };
        momentum = make(3.0, -0.7);
        stabilization = make(0.31, -0.2);
        for (size_t c = 0; c < 3; ++c)
        {
            gradient[c] = make(0.13 * (c + 1), -0.09);
            divergence[c] = make(-0.17, 0.23 * (c + 1));
        }
        const auto gauge = pin ? std::optional<GO>{10} : std::optional<GO>{};
        composite =
            std::make_unique<CoupledBlockOperator<Pack>>(coupled, momentum, gradient, divergence, stabilization, gauge);
        assembled = Teuchos::rcp(new Matrix(coupled, 12));
        for (size_t row = 0; row < ids.size(); ++row)
        {
            auto add = [&](const Matrix& m, int equation, int unknown)
            {
                Matrix::local_inds_host_view_type cols;
                Matrix::values_host_view_type vals;
                m.getLocalRowView(static_cast<LO>(row), cols, vals);
                Teuchos::Array<GO> indices;
                Teuchos::Array<double> values;
                for (size_t k = 0; k < cols.extent(0); ++k)
                {
                    indices.push_back(4 * m.getColMap()->getGlobalElement(cols(k)) + unknown);
                    values.push_back(vals(k));
                }
                assembled->insertGlobalValues(4 * ids[row] + equation, indices(), values());
            };
            for (int c = 0; c < 3; ++c)
            {
                add(*momentum, c, c);
                add(*gradient[c], c, 3);
            }
            if (pin && ids[row] == 10)
            {
                Teuchos::Array<GO> cols{4 * ids[row] + 3};
                Teuchos::Array<double> vals{1.0};
                assembled->insertGlobalValues(cols[0], cols(), vals());
            }
            else
            {
                for (int c = 0; c < 3; ++c)
                    add(*divergence[c], 3, c);
                add(*stabilization, 3, 3);
            }
        }
        assembled->fillComplete();
    }
};

TEST(CoupledBlockOperatorTest, DistributedApplicationContract)
{
    for (bool gauge : {false, true})
    {
        AlgebraFixture f(gauge);
        MV x(f.coupled, 3), y(f.coupled, 3), reference(f.coupled, 3);
        for (int block = -1; block < 4; ++block)
        {
            fill(x);
            if (block >= 0)
            {
                auto v = x.getLocalViewHost(Tpetra::Access::ReadWrite);
                for (size_t row = 0; row < x.getLocalLength(); ++row)
                    if (row % 4 != static_cast<size_t>(block))
                        for (size_t j = 0; j < 3; ++j)
                            v(row, j) = 0.0;
            }
            y.putScalar(std::numeric_limits<double>::quiet_NaN());
            f.composite->apply(x, y);
            f.assembled->apply(x, reference);
            expect_near(y, reference);
        }
        fill(x);
        fill(y);
        reference.assign(y);
        f.composite->apply(x, y, Teuchos::NO_TRANS, 1.7, -0.3);
        f.assembled->apply(x, reference, Teuchos::NO_TRANS, 1.7, -0.3);
        expect_near(y, reference);
        f.assembled->apply(x, reference);
        f.composite->apply(x, x);
        expect_near(x, reference);
        x.putScalar(0.0);
        f.composite->apply(x, y);
        EXPECT_EQ(y.getVector(0)->normInf(), 0.0);
        x.putScalar(std::numeric_limits<double>::quiet_NaN());
        y.putScalar(std::numeric_limits<double>::quiet_NaN());
        f.composite->apply(x, y, Teuchos::NO_TRANS, 0.0, 0.0);
        EXPECT_EQ(y.getVector(0)->normInf(), 0.0);
        y.putScalar(2.0);
        f.composite->apply(x, y, Teuchos::NO_TRANS, 0.0, 3.0);
        EXPECT_EQ(y.getVector(0)->normInf(), 6.0);
        EXPECT_FALSE(f.composite->hasTransposeApply());
        EXPECT_THROW(f.composite->apply(x, y, Teuchos::TRANS), std::invalid_argument);
        EXPECT_THROW(f.composite->apply(x, y, Teuchos::CONJ_TRANS), std::invalid_argument);
        MV wrong_columns(f.coupled, 2), wrong_map(f.cells, 3);
        EXPECT_THROW(f.composite->apply(x, wrong_columns), std::invalid_argument);
        EXPECT_THROW(f.composite->apply(x, wrong_map), std::invalid_argument);
        EXPECT_THROW(f.composite->apply(wrong_map, y), std::invalid_argument);
        const auto allocations = f.composite->scratch_allocations();
        fill(x);
        for (int repeat = 0; repeat < 5; ++repeat)
            f.composite->apply(x, y);
        EXPECT_EQ(f.composite->scratch_allocations(), allocations);
        EXPECT_EQ(f.composite->scratch_payload_bytes(), 10 * 2 * 3 * sizeof(double));

        // Nonconstant-stride, overlapping input/output selections.
        MV parent(f.coupled, 5);
        fill(parent);
        Teuchos::Array<size_t> in_columns{4, 0, 2}, out_columns{2, 4, 1};
        auto in = parent.subView(in_columns());
        auto out = parent.subViewNonConst(out_columns());
        MV saved_in(*in, Teuchos::Copy), saved_out(*out, Teuchos::Copy);
        f.assembled->apply(saved_in, saved_out, Teuchos::NO_TRANS, 0.9, -0.4);
        f.composite->apply(*in, *out, Teuchos::NO_TRANS, 0.9, -0.4);
        expect_near(*out, saved_out);
        f.assembled->apply(saved_in, reference);
        in = Teuchos::null;
        out->putScalar(std::numeric_limits<double>::quiet_NaN());
        f.composite->apply(saved_in, *out);
        expect_near(*out, reference);
    }
}

SP<const Handle> make_mesh(int kind)
{
    if (kind == 0)
        return std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
            Vec3D<ArrReal>{{{0., .25, .5, .75, 1.}, {0., .5, 1.}, {0., .5, 1.}}}));
    if (kind == 1)
        return std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCylindrial3D>(
            Vec3D<ArrReal>{{{.5, .75, 1.}, {0., .4, .8}, {0., .25, .5, .75, 1.}}}));
    auto native = test::make_unstructured_hex_line(16, 0.0625);
    using Partitioned = Meshes::PartitionedMesh<Meshes::UnstructuredMesh, Pack>;
    const auto comm = Tpetra::getDefaultComm();
    auto partition = MeshPartitioner<Pack>::partition(*native, comm);
    return std::make_shared<Handle>(std::make_shared<Partitioned>(native, std::move(partition.indexer), comm));
}

class CompositeMeshTest : public testing::TestWithParam<int>
{
};
TEST_P(CompositeMeshTest, ReferenceSolveAndRetainedGenerations)
{
    const auto mesh = make_mesh(GetParam());
    VectorCellFieldStored<Pack> velocity(mesh, vec3<double>{0.1, 0.2, -0.1}, "u");
    ScalarCellFieldStored<Pack> pressure(mesh, 0.0, "p");
    ScalarFaceFieldStored<Pack> flux(mesh, 0.0, "phi");
    BoundaryConditionSet boundaries;
    const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
    IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
    Solver solver(mesh);
    TimeStepperOptions options;
    options.time_step = .02;
    auto reference = solver.assemble(equation, velocity, pressure, flux, boundary, boundaries, options, 7.0);
    options.coupled_operator_backend = Backend::BlockComposite;
    auto composite = solver.assemble(equation, velocity, pressure, flux, boundary, boundaries, options, 7.0);
    ASSERT_TRUE(composite.matrix.is_null());
    EXPECT_TRUE(composite.overlap_map.is_null());
    ASSERT_FALSE(composite.linear_operator.is_null());
    expect_near(*reference.rhs, *composite.rhs);
    MV x(composite.map, 2), a(composite.map, 2), b(composite.map, 2);
    fill(x);
    reference.matrix->apply(x, a);
    composite.linear_operator->apply(x, b);
    expect_near(a, b, 100.0);
    LinearSolverOptions linear;
    linear.tolerance = 1e-10;
    linear.max_iterations = 200;
    EXPECT_TRUE(solver.solve(reference, velocity, pressure, linear).converged);
    MV reference_velocity(velocity.owned_data(), Teuchos::Copy);
    MV reference_pressure(pressure.owned_data(), Teuchos::Copy);
    velocity.owned_data().putScalar(0.0);
    pressure.owned_data().putScalar(0.0);
    EXPECT_TRUE(solver.solve(composite, velocity, pressure, linear).converged);
    // Linear convergence tolerance controls solution comparisons.
    MV delta(velocity.owned_data(), Teuchos::Copy);
    delta.update(-1., reference_velocity, 1.);
    EXPECT_LT(delta.getVector(0)->normInf(), 2e-8);
    MV dp(pressure.owned_data(), Teuchos::Copy);
    dp.update(-1., reference_pressure, 1.);
    EXPECT_LT(dp.getVector(0)->normInf(), 2e-8);

    MV saved_rhs(*composite.rhs, Teuchos::Copy);
    const auto* old_momentum = composite.momentum.getRawPtr();
    options.time_step *= 2;
    auto updated = solver.assemble(equation, velocity, pressure, flux, boundary, boundaries, options, 7.0);
    EXPECT_NE(updated.momentum.getRawPtr(), old_momentum);
    composite.linear_operator->apply(x, b);
    expect_near(a, b, 100.0);
    expect_near(saved_rhs, *composite.rhs);
    // Retain only the operator across cache clearing and an equation update.
    auto retained = composite.linear_operator;
    composite = {};
    solver.clear_cache();
    options.time_step *= 2;
    updated = solver.assemble(equation, velocity, pressure, flux, boundary, boundaries, options, 7.0);
    retained->apply(x, b);
    expect_near(a, b, 100.0);
}
TEST(CoupledBlockOperatorTest, ProductionReuseAndStorageRelease)
{
    const auto mesh = make_mesh(0);
    VectorCellFieldStored<Pack> u(mesh, vec3<double>{.1, .2, -.1}, "u");
    ScalarCellFieldStored<Pack> p(mesh, 0., "p");
    ScalarFaceFieldStored<Pack> flux(mesh, 0., "phi");
    BoundaryConditionSet boundaries;
    const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
    IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
    Solver solver(mesh);
    TimeStepperOptions options;
    options.coupled_operator_backend = Backend::BlockComposite;
    LinearSolverOptions linear;
    linear.tolerance = 1e-10;
    linear.max_iterations = 200;
    CoupledStorageStatistics warm;
    for (int step = 0; step < 4; ++step)
    {
        options.time_step *= 1.1;
        auto system = solver.assemble(equation, u, p, flux, boundary, boundaries, options);
        ASSERT_TRUE(system.matrix.is_null());
        EXPECT_TRUE(solver.solve(system, u, p, linear).converged);
        auto storage = solver.storage_statistics();
        if (step == 0)
            warm = storage;
        else
        {
            EXPECT_EQ(storage.live_matrices, warm.live_matrices);
            EXPECT_EQ(storage.value_bytes, warm.value_bytes);
            EXPECT_EQ(storage.graph_bytes, warm.graph_bytes);
            EXPECT_EQ(storage.composite_scratch_bytes, warm.composite_scratch_bytes);
            EXPECT_EQ(storage.composite_scratch_allocations, warm.composite_scratch_allocations);
        }
    }
    EXPECT_EQ(solver.cache_statistics().coupled_matrix_builds, 0U);
    EXPECT_EQ(solver.cache_statistics().composite_operator_builds, 1U);
    EXPECT_EQ(solver.cache_statistics().preconditioner_builds, 1U);
    EXPECT_EQ(solver.cache_statistics().preconditioner_numeric_reuses, 3U);
    EXPECT_EQ(solver.cache_statistics().belos_solver_builds, 1U);
    solver.clear_cache();
    const auto cleared = solver.storage_statistics();
    EXPECT_EQ(cleared.live_matrices, 1U); // equation still owns momentum
    EXPECT_EQ(cleared.composite_scratch_bytes, 0U);
    EXPECT_LT(cleared.value_bytes + cleared.graph_bytes, warm.value_bytes + warm.graph_bytes);
}

TEST(CoupledBlockOperatorTest, BoundaryAffineAndFixedFluxEquivalence)
{
    const auto mesh = make_mesh(0);
    VectorCellFieldStored<Pack> u(mesh, vec3<double>{.1, -.2, .3}, "u");
    ScalarCellFieldStored<Pack> p(mesh, 5., "p");
    ScalarFaceFieldStored<Pack> flux(mesh, 0., "phi");
    IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
    Solver solver(mesh);
    TimeStepperOptions options;
    for (int kind = 0; kind < 3; ++kind)
    {
        BoundaryConditionSet boundaries;
        boundaries.pressure["zmax"] = {BoundaryConditionType::Dirichlet, 17.};
        boundaries.pressure["zmin"] = {BoundaryConditionType::Neumann, 2.};
        boundaries.velocity["zmax"] = {BoundaryConditionType::Neumann, {}};
        if (kind == 2)
        {
            boundaries.velocity["zmax"] = {BoundaryConditionType::Dirichlet, {0., 0., .01}};
            solver.set_fixed_boundary_flux_provider({"zmax"}, [](int, size_t, LO) { return .001; }, 2);
        }
        else if (kind == 0)
            boundaries.pressure["zmax"] = {BoundaryConditionType::Neumann, 3.};
        const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
        options.coupled_operator_backend = Backend::Assembled;
        const auto reference = solver.assemble(equation, u, p, flux, boundary, boundaries, options, 13.);
        options.coupled_operator_backend = Backend::BlockComposite;
        const auto composite = solver.assemble(equation, u, p, flux, boundary, boundaries, options, 13.);
        EXPECT_EQ(reference.pressure_gauge_gid, composite.pressure_gauge_gid);
        EXPECT_EQ(composite.pressure_gauge_gid.has_value(), kind != 1);
        expect_near(*reference.rhs, *composite.rhs);
        MV x(composite.map, 2), a(composite.map, 2), b(composite.map, 2);
        fill(x);
        reference.matrix->apply(x, a);
        composite.linear_operator->apply(x, b);
        expect_near(a, b, 1000.);
        x.putScalar(0.);
        composite.linear_operator->apply(x, b);
        EXPECT_EQ(b.getVector(0)->normInf(), 0.);
    }
    EXPECT_THROW(coupled_operator_backend_from_string("automatic"), std::invalid_argument);
}

TEST_P(CompositeMeshTest, StreamedProductsPreserveNumericsAndReleaseStorage)
{
    const auto mesh = make_mesh(GetParam());
    VectorCellFieldStored<Pack> u(mesh, vec3<double>{.1, .2, -.1}, "u");
    ScalarCellFieldStored<Pack> p(mesh, 0., "p");
    ScalarFaceFieldStored<Pack> flux(mesh, 0., "phi");
    BoundaryConditionSet boundaries;
    const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
    IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
    Solver cached(mesh), streamed(mesh);
    TimeStepperOptions options;
    options.coupled_operator_backend = Backend::BlockComposite;
    auto reference = cached.assemble(equation, u, p, flux, boundary, boundaries, options);
    const auto cached_storage = cached.storage_statistics();
    options.coupled_workspace_policy = CoupledWorkspacePolicy::StreamedProducts;
    auto system = streamed.assemble(equation, u, p, flux, boundary, boundaries, options);
    const auto streamed_storage = streamed.storage_statistics();
    EXPECT_EQ(cached_storage.live_matrices - streamed_storage.live_matrices, 9U);
    EXPECT_LT(streamed_storage.value_bytes + streamed_storage.graph_bytes,
        cached_storage.value_bytes + cached_storage.graph_bytes);
    expect_near(*system.rhs, *reference.rhs);
    MV x(mesh->owned_cell_map(), 2), a(mesh->owned_cell_map(), 2), b(mesh->owned_cell_map(), 2);
    fill(x);
    reference.pressure_stabilization->apply(x, a);
    system.pressure_stabilization->apply(x, b);
    expect_near(a, b);
    reference.schur->apply(x, a);
    system.schur->apply(x, b);
    expect_near(a, b);
    LinearSolverOptions linear;
    linear.tolerance = 1e-10;
    linear.max_iterations = 200;
    ASSERT_TRUE(streamed.solve(system, u, p, linear).converged);
    const auto before = streamed.storage_statistics();
    system = {};
    options.time_step *= 1.2;
    system = streamed.assemble(equation, u, p, flux, boundary, boundaries, options);
    ASSERT_TRUE(streamed.solve(system, u, p, linear).converged);
    EXPECT_EQ(streamed.storage_statistics().live_matrices, before.live_matrices);
    EXPECT_EQ(streamed.cache_statistics().coupled_matrix_builds, 0U);
    EXPECT_EQ(streamed.cache_statistics().matrix_graph_reuses, 1U);
    EXPECT_EQ(streamed.cache_statistics().streamed_product_peak, 1U);
    EXPECT_EQ(streamed.cache_statistics().streamed_products_live, 0U);
    // Switching workspace policy does not change the true operator backend.
    system = {};
    options.coupled_workspace_policy = CoupledWorkspacePolicy::CachedProducts;
    system = streamed.assemble(equation, u, p, flux, boundary, boundaries, options);
    EXPECT_EQ(streamed.storage_statistics().live_matrices, before.live_matrices + 9);
    system = {};
    options.coupled_workspace_policy = CoupledWorkspacePolicy::StreamedProducts;
    system = streamed.assemble(equation, u, p, flux, boundary, boundaries, options);
    EXPECT_EQ(streamed.storage_statistics().live_matrices, before.live_matrices);
    EXPECT_THROW(coupled_workspace_policy_from_string("local_only"), std::invalid_argument);
}

TEST(CoupledBlockOperatorTest, NumericUpdatesMatchFreshAssemblyAtScale)
{
    Vec3D<ArrReal> edges;
    for (int axis = 0; axis < 3; ++axis)
        for (int i = 0; i <= 8; ++i)
            edges[axis].push_back(i / 8.0);
    SP<const Handle> mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(edges));
    BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        boundaries.velocity[name] = {BoundaryConditionType::NoSlip, {}};
    const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
    LinearSolverOptions linear;
    linear.tolerance = 1e-9;
    linear.max_iterations = 400;
    for (const auto backend : {Backend::Assembled, Backend::BlockComposite})
        for (const auto workspace : {CoupledWorkspacePolicy::CachedProducts, CoupledWorkspacePolicy::StreamedProducts})
        {
            SCOPED_TRACE(std::string(to_string(backend)) + "/" + std::string(to_string(workspace)));
            Solver reused(mesh), fresh(mesh);
            fresh.set_rebuild_policy(CoupledRebuildPolicy::Always);
            IncompressibleMomentumEquation<Pack, Handle> equation(mesh), reference_equation(mesh);
            VectorCellFieldStored<Pack> u(mesh, vec3<double>{}, "u"), reference_u(mesh, vec3<double>{}, "reference_u");
            ScalarCellFieldStored<Pack> p(mesh, 0., "p"), reference_p(mesh, 0., "reference_p");
            ScalarFaceFieldStored<Pack> flux(mesh, 0., "phi");
            TimeStepperOptions options;
            options.time_step = .01;
            options.non_orthogonal_treatment = FVM::NonOrthogonalTreatment::Explicit;
            options.coupled_operator_backend = backend;
            options.coupled_workspace_policy = workspace;
            for (int step = 0; step < 4; ++step)
            {
                SCOPED_TRACE(step);
                for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
                {
                    u.set_owned_value(static_cast<LO>(cell), {.1, .2, -.1});
                    reference_u.set_owned_value(static_cast<LO>(cell), {.1, .2, -.1});
                }
                p.owned_data().putScalar(0.);
                reference_p.owned_data().putScalar(0.);
                const auto system = reused.assemble(equation, u, p, flux, boundary, boundaries, options);
                auto reference_options = options;
                reference_options.coupled_operator_backend = Backend::Assembled;
                reference_options.coupled_workspace_policy = CoupledWorkspacePolicy::CachedProducts;
                const auto reference = fresh.assemble(
                    reference_equation, reference_u, reference_p, flux, boundary, boundaries, reference_options);
                expect_near(*system.rhs, *reference.rhs);
                MV x(system.map, 1), a(system.map, 1), b(system.map, 1);
                fill(x);
                system.linear_operator->apply(x, a);
                reference.linear_operator->apply(x, b);
                expect_near(a, b);
                MV q(mesh->owned_cell_map(), 1), ca(mesh->owned_cell_map(), 1), cb(mesh->owned_cell_map(), 1);
                fill(q);
                system.pressure_stabilization->apply(q, ca);
                reference.pressure_stabilization->apply(q, cb);
                expect_near(ca, cb);
                system.schur->apply(q, ca);
                reference.schur->apply(q, cb);
                expect_near(ca, cb);
                const auto ref_result = fresh.solve(reference, reference_u, reference_p, linear);
                const auto result = reused.solve(system, u, p, linear);
                ASSERT_TRUE(ref_result.converged) << "fresh iterations=" << ref_result.iterations;
                ASSERT_TRUE(result.converged)
                    << "reused iterations=" << result.iterations << ", fresh iterations=" << ref_result.iterations;
                MV delta(u.owned_data(), Teuchos::Copy);
                delta.update(-1., reference_u.owned_data(), 1.);
                for (size_t component = 0; component < 3; ++component)
                    EXPECT_LT(delta.getVector(component)->normInf(), 1e-8);
                options.time_step *= 1.1;
            }
        }
}

TEST_P(CompositeMeshTest, GradientSchemeRefreshPreservesAffineFluxBalance)
{
    const auto mesh = make_mesh(GetParam());
    VectorCellFieldStored<Pack> u(mesh, vec3<double>{}, "u");
    ScalarCellFieldStored<Pack> p(mesh, 0., "p");
    ScalarFaceFieldStored<Pack> flux(mesh, 0., "phi");
    BoundaryConditionSet boundaries;
    for (const auto& [batch, faces] : mesh->boundary_batches())
    {
        const auto name = mesh->boundary_batch_name(batch);
        boundaries.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        boundaries.pressure[name] = {BoundaryConditionType::Neumann, 2.};
    }
    boundaries.velocity["zmax"] = {BoundaryConditionType::Neumann, {}};
    boundaries.pressure["zmax"] = {BoundaryConditionType::Dirichlet, 17.};
    const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
    FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Handle> face_workspace(mesh);
    IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
    const VolumeContinuityTarget<Pack, Handle> target(mesh);
    constexpr double density = 13.;
    for (const auto backend : {Backend::Assembled, Backend::BlockComposite})
        for (const auto policy : {CoupledWorkspacePolicy::CachedProducts, CoupledWorkspacePolicy::StreamedProducts})
        {
            Solver solver(mesh);
            TimeStepperOptions options;
            options.coupled_operator_backend = backend;
            options.coupled_workspace_policy = policy;
            Solver::system_type old;
            std::unique_ptr<MV> trial, previous_action;
            for (const auto scheme : {FVM::CellGradientScheme::LeastSquares, FVM::CellGradientScheme::GaussLinear,
                     FVM::CellGradientScheme::LeastSquares})
            {
                options.pressure_gradient_scheme = scheme;
                const auto system =
                    solver.assemble(equation, u, p, flux, boundary, boundaries, options, target, density);
                if (!trial)
                {
                    trial = std::make_unique<MV>(system.map, 1);
                    previous_action = std::make_unique<MV>(system.map, 1);
                    trial->putScalar(0.);
                    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
                    {
                        const auto gid = mesh->owned_cell_map()->getGlobalElement(static_cast<LO>(cell));
                        const auto q = std::sin(.3 * (gid + 1));
                        trial->replaceLocalValue(4 * cell + 3, 0, q);
                        p.set_owned_value(static_cast<LO>(cell), density * q);
                    }
                    p.sync_ghosts();
                }
                if (!old.linear_operator.is_null())
                {
                    MV retained(system.map, 1);
                    old.linear_operator->apply(*trial, retained);
                    expect_near(retained, *previous_action);
                }
                MV residual(system.map, 1);
                system.linear_operator->apply(*trial, *previous_action);
                residual.assign(*previous_action);
                residual.update(-1., *system.rhs, 1.);
                FVM::pressure_weighted_face_fluxes(
                    u, p, options.time_step / density, boundary, boundaries.pressure, face_workspace, flux, scheme);
                const auto face_data = flux.owned_read_view();
                const auto rows = residual.getLocalViewHost(Tpetra::Access::ReadOnly);
                for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
                {
                    const auto balance = FVM::cell_flux_balance<Pack>(*mesh, flux, face_data, static_cast<LO>(cell));
                    EXPECT_NEAR(rows(4 * cell + 3, 0), balance,
                        512 * std::numeric_limits<double>::epsilon() * std::max(1., std::abs(balance)));
                }
                old = system;
            }
            EXPECT_EQ(solver.cache_statistics().static_geometry_builds, 3U);
        }
}

INSTANTIATE_TEST_SUITE_P(Native, CompositeMeshTest, testing::Values(0, 1, 2));
} // namespace
