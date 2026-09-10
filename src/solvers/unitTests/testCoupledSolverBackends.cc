/** @file testCoupledSolverBackends.cc @brief End-to-end qualification of coupled backend choices. */
#include "FVM/FaceFlux.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "utils/testing_environment.hh"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <tuple>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using LO = Pack::local_ordinal_type;
using Backend = CoupledOperatorBackend;
using Workspace = CoupledWorkspacePolicy;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Selection
{
    Backend backend;
    Workspace workspace;
};
const std::array<Selection, 4> selections{
    {{Backend::Assembled, Workspace::CachedProducts}, {Backend::Assembled, Workspace::StreamedProducts},
        {Backend::BlockComposite, Workspace::CachedProducts}, {Backend::BlockComposite, Workspace::StreamedProducts}}};

/** Inspect the production registry rather than assembling a second diagnostic operator. */
template<class Base> class Inspectable : public Base
{
public:
    using Base::Base;
    CoupledPressureVelocityCacheStatistics coupled_statistics()
    {
        auto& problem = this->FluidSolver<Pack>::d_problem;
        using Native = CoupledPressureVelocitySolver<Pack, Handle>;
        for (const auto* name :
            {"boussinesq_coupled_pressure_velocity_solver", "isothermal_coupled_pressure_velocity_solver"})
            if (problem.contains(name))
            {
                auto result = problem.template object<Native>(name).cache_statistics();
                // Legacy physical equations assemble on the canonical handle,
                // while the compatibility solver owns the Belos solve state.
                const auto& solve =
                    problem.template object<CoupledPressureVelocitySolver<Pack>>("coupled_pressure_velocity_solver")
                        .cache_statistics();
                result.belos_solver_reuses = solve.belos_solver_reuses;
                return result;
            }
        if (this->FluidSolver<Pack>::uses_legacy_backend())
            return problem.template object<CoupledPressureVelocitySolver<Pack>>("coupled_pressure_velocity_solver")
                .cache_statistics();
        return problem.template object<Native>("coupled_pressure_velocity_solver").cache_statistics();
    }
    void select(Selection choice)
    {
        auto& options = this->FluidSolver<Pack>::d_problem.time_options();
        options.coupled_operator_backend = choice.backend;
        options.coupled_workspace_policy = choice.workspace;
    }
};

enum class Family
{
    Fluid,
    Isothermal,
    Boussinesq,
    PhysicalBoussinesq
};

SP<const Handle> native_mesh(int kind)
{
    if (kind == 0)
        return std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
            Vec3D<ArrReal>{{{0., .25, .5, .75, 1.}, {0., .25, .5, .75, 1.}, {0., .25, .5, .75, 1.}}}));
    if (kind == 1)
        return std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCylindrial3D>(
            Vec3D<ArrReal>{{{1., 1.25, 1.5, 1.75, 2.}, {0., .25, .5, .75, 1.}, {0., .25, .5, .75, 1.}}}));
    if (kind == 4)
        return std::make_shared<Handle>(
            std::make_shared<Meshes::OrthogonalCylindrial3D>(Vec3D<ArrReal>{{{1., 1.25, 1.5, 1.75, 2.},
                {0., .5 * std::numbers::pi, std::numbers::pi, 1.5 * std::numbers::pi, 2. * std::numbers::pi},
                {0., .25, .5, .75, 1.}}}));
    if (kind == 2)
    {
        auto native = test::make_unstructured_hex_line(16, .0625);
        const auto comm = Tpetra::getDefaultComm();
        if (comm->getSize() == 1)
            return std::make_shared<Handle>(native);
        using Partitioned = Meshes::PartitionedMesh<Meshes::UnstructuredMesh, Pack>;
        auto partition = MeshPartitioner<Pack>::partition(*native, comm);
        return std::make_shared<Handle>(std::make_shared<Partitioned>(native, std::move(partition.indexer), comm));
    }
    using Semi = Meshes::SemiStructuredXY_Z;
    auto native =
        std::make_shared<Semi>(Arr<Semi::Vec3>{{0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.}, {0., 1., 0.}, {.35, .40, 0.}},
            Arr<Arr<unsigned>>{{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}}, ArrReal{0., .25, .5, .75, 1.});
    return std::make_shared<Handle>(native);
}

std::vector<double> values(const Pack::multi_vector_type& field)
{
    const auto data = field.getLocalViewHost(Tpetra::Access::ReadOnly);
    std::vector<double> result;
    for (size_t cell = 0; cell < data.extent(0); ++cell)
        for (size_t component = 0; component < data.extent(1); ++component)
            result.push_back(data(cell, component));
    return result;
}

struct Frame
{
    std::vector<double> velocity, pressure, flux, temperature;
};
struct Run
{
    std::vector<Frame> frames;
    CoupledPressureVelocityCacheStatistics statistics;
};

template<class Base, class Mesh, class... Args>
Run advance(SP<const Mesh> mesh, BoundaryConditionSet boundaries, TimeStepperOptions options,
    LinearSolverOptions linear, Selection selection, bool switch_backends, Args... args)
{
    options.coupled_operator_backend = selection.backend;
    options.coupled_workspace_policy = selection.workspace;
    Inspectable<Base> solver(mesh, boundaries, options, linear, args...);
    if constexpr (std::is_same_v<Base, BoussinesqSolver<Pack>>)
        solver.initialize_linear_temperature({0., 0., 1.}, 300.2, 300.);
    const auto& fields_mesh = solver.velocity().mesh();
    for (size_t cell = 0; cell < fields_mesh.num_owned_cells(); ++cell)
    {
        const auto center = fields_mesh.cell_centroid(static_cast<LO>(cell));
        solver.velocity().set_owned_value(
            static_cast<LO>(cell), {.02 * std::sin(center.y), -.01 * std::cos(center.x), .005 * std::sin(center.z)});
    }
    solver.velocity().sync_ghosts();
    Run run;
    for (int step = 0; step < 4; ++step)
    {
        if (switch_backends)
            solver.select(selections[step]);
        solver.set_time_step(options.time_step * (1. + .1 * step));
        solver.step();
        if constexpr (std::is_same_v<Mesh, Handle>)
        {
            if (std::holds_alternative<Handle::MultiRegionPtr>(mesh->variant()))
            {
                EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
                EXPECT_FALSE(mesh->has_materialized_connectivity());
                EXPECT_FALSE(mesh->has_materialized_indexer());
            }
        }
        EXPECT_EQ(solver.step_index(), step + 1);
        EXPECT_TRUE(solver.last_step_statistics().converged);
        EXPECT_LT(solver.last_volume_continuity_residuals().maximum, 1e-8);
        const auto& corrected_flux = solver.pressure_corrected_face_fluxes();
        const auto flux_data = corrected_flux.owned_read_view();
        for (size_t cell = 0; cell < fields_mesh.num_owned_cells(); ++cell)
            EXPECT_LT(
                std::abs(FVM::cell_flux_balance<Pack>(fields_mesh, corrected_flux, flux_data, static_cast<LO>(cell))),
                1e-8);
        Frame frame{values(solver.velocity().owned_data()), values(solver.pressure().owned_data()),
            values(solver.pressure_corrected_face_fluxes().owned_data()), {}};
        if constexpr (std::is_same_v<Base, BoussinesqSolver<Pack>>)
            frame.temperature = values(solver.temperature().owned_data());
        run.frames.push_back(std::move(frame));
    }
    run.statistics = solver.coupled_statistics();
    return run;
}

void compare(const Run& reference, const Run& candidate)
{
    ASSERT_EQ(reference.frames.size(), candidate.frames.size());
    for (size_t step = 0; step < reference.frames.size(); ++step)
    {
        SCOPED_TRACE(step);
        const auto& a = reference.frames[step];
        const auto& b = candidate.frames[step];
        for (auto fields : {std::pair{&a.velocity, &b.velocity}, {&a.pressure, &b.pressure}, {&a.flux, &b.flux},
                 {&a.temperature, &b.temperature}})
        {
            ASSERT_EQ(fields.first->size(), fields.second->size());
            for (size_t i = 0; i < fields.first->size(); ++i)
                EXPECT_NEAR((*fields.first)[i], (*fields.second)[i], 2e-8);
        }
    }
}

template<class Mesh>
void exercise(SP<const Mesh> mesh, Family family, bool outlet, FVM::NonOrthogonalTreatment treatment,
    FVM::CellGradientScheme gradient = FVM::CellGradientScheme::LeastSquares)
{
    BoundaryConditionSet boundaries;
    for (const auto& [batch, faces] : mesh->boundary_batches())
    {
        const auto name = mesh->boundary_batch_name(batch);
        boundaries.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        boundaries.pressure[name] = {BoundaryConditionType::Neumann, 0.};
        boundaries.temperature[name] = {BoundaryConditionType::Neumann, 0.};
    }
    if (outlet)
    {
        boundaries.velocity["xmin"] = {BoundaryConditionType::Dirichlet, {.03, 0., 0.}};
        boundaries.velocity["xmax"] = {BoundaryConditionType::Neumann, {}};
        boundaries.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 17.};
        boundaries.temperature["xmin"] = {BoundaryConditionType::Dirichlet, 300.};
    }
    TimeStepperOptions options;
    options.pressure_velocity_coupling = PressureVelocityCoupling::CoupledKrylov;
    options.time_step = .002;
    options.kinematic_viscosity = .01;
    options.thermal_diffusivity = .01;
    options.reference_temperature = 300.;
    options.gravity_z = -.1;
    options.non_orthogonal_treatment = treatment;
    options.pressure_gradient_scheme = gradient;
    LinearSolverOptions linear;
    linear.tolerance = 1e-11;
    linear.max_iterations = 400;
    auto run = [&](Selection selection, bool switching)
    {
        switch (family)
        {
            case Family::Fluid:
                return advance<FluidSolver<Pack>>(mesh, boundaries, options, linear, selection, switching);
            case Family::Isothermal:
                return advance<IncompressibleIsothermalSolver<Pack>>(
                    mesh, boundaries, options, linear, selection, switching, 7.);
            case Family::Boussinesq:
                return advance<BoussinesqSolver<Pack>>(mesh, boundaries, options, linear, selection, switching);
            case Family::PhysicalBoussinesq:
                BoussinesqModelOptions model;
                model.reference_density = 7.;
                model.density = 7.;
                model.specific_heat_capacity = 4.;
                model.dynamic_viscosity = .07;
                model.thermal_conductivity = .28;
                return advance<BoussinesqSolver<Pack>>(mesh, boundaries, options, linear, selection, switching, model);
        }
        throw std::logic_error("unknown family");
    };
    const auto reference = run(selections[0], false);
    for (const auto selection : selections)
    {
        SCOPED_TRACE(std::string(to_string(selection.backend)) + "/" + std::string(to_string(selection.workspace)));
        const auto candidate = run(selection, false);
        compare(reference, candidate);
        EXPECT_GT(candidate.statistics.belos_solver_reuses, 0U);
        if (selection.backend == Backend::BlockComposite)
        {
            EXPECT_EQ(candidate.statistics.coupled_matrix_builds, 0U);
            EXPECT_GT(candidate.statistics.composite_operator_builds, 0U);
        }
        if (selection.workspace == Workspace::StreamedProducts)
        {
            EXPECT_EQ(candidate.statistics.streamed_products_live, 0U);
            EXPECT_EQ(candidate.statistics.streamed_product_peak, 1U);
        }
    }
    compare(reference, run(selections[0], true));
}

class CoupledSolverBackendsTest : public testing::TestWithParam<Family>
{
};
TEST_P(CoupledSolverBackendsTest, NativeCartesianAndCylindrical)
{
    for (int kind : {0, 1})
    {
        SCOPED_TRACE(kind);
        exercise(native_mesh(kind), GetParam(), false, FVM::NonOrthogonalTreatment::Implicit);
    }
}
TEST_P(CoupledSolverBackendsTest, NativeUnstructuredPressureOutlet)
{
    exercise(native_mesh(2), GetParam(), true, FVM::NonOrthogonalTreatment::Implicit);
}
TEST_P(CoupledSolverBackendsTest, ConfiguredGaussPressureGradient)
{
    for (int kind : {0, 1, 2})
        exercise(native_mesh(kind), GetParam(), kind == 2, FVM::NonOrthogonalTreatment::Implicit,
            FVM::CellGradientScheme::GaussLinear);
}
TEST_P(CoupledSolverBackendsTest, LegacyCartesianPressureOutlet)
{
    auto db = std::make_shared<Database>(*test::make_2x2x2_database());
    db->set("X", ArrReal{0., .25, .5, .75, 1.});
    db->set("Y", ArrReal{0., .25, .5, .75, 1.});
    db->set("Z", ArrReal{0., .25, .5, .75, 1.});
    SP<const Mesh<Pack>> mesh = test::build_mesh<Pack>(db);
    exercise(mesh, GetParam(), true, FVM::NonOrthogonalTreatment::Explicit);
    exercise(mesh, GetParam(), true, FVM::NonOrthogonalTreatment::Implicit, FVM::CellGradientScheme::GaussLinear);
}
TEST_P(CoupledSolverBackendsTest, NativePeriodicCylindrical)
{
    const auto mesh = native_mesh(4);
    for (const auto gradient : {FVM::CellGradientScheme::LeastSquares, FVM::CellGradientScheme::GaussLinear})
    {
        SCOPED_TRACE(static_cast<int>(gradient));
        exercise(mesh, GetParam(), false, FVM::NonOrthogonalTreatment::Implicit, gradient);
    }
}
TEST_P(CoupledSolverBackendsTest, LegacyCurvedGeometry)
{
    for (const auto domain : {MeshFactory::DomainType::CYLINDER, MeshFactory::DomainType::SPHERE})
    {
        SCOPED_TRACE(static_cast<int>(domain));
        auto db = std::make_shared<Database>();
        db->set("dimension", 3);
        db->set("mesh_size", real_t{1.});
        db->set("domain_type", static_cast<int>(domain));
        db->set("radius", real_t{1.});
        db->set("height", real_t{2.});
        db->set("domain_exterior_face_types", domain == MeshFactory::DomainType::CYLINDER
                                                  ? ArrString{"radial", "zmin", "zmax"}
                                                  : ArrString{"lower_surface", "upper_surface"});
        SP<const Mesh<Pack>> mesh = test::build_mesh<Pack>(db);
        exercise(mesh, GetParam(), false, FVM::NonOrthogonalTreatment::Implicit);
    }
}
TEST_P(CoupledSolverBackendsTest, SerialSemiStructuredNonOrthogonal)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "SemiStructuredXY_Z is serial-only.";
    for (const auto treatment : {FVM::NonOrthogonalTreatment::Explicit, FVM::NonOrthogonalTreatment::Implicit,
             FVM::NonOrthogonalTreatment::Hybrid})
    {
        SCOPED_TRACE(static_cast<int>(treatment));
        exercise(native_mesh(3), GetParam(), false, treatment);
    }
}
/** Compare each backend/policy and in-place switching across Cartesian seams. */
TEST_P(CoupledSolverBackendsTest, MultiRegionCartesian)
{
    SP<const Handle> mesh = std::make_shared<Handle>(test::two_regions());
    for (const auto gradient : {FVM::CellGradientScheme::LeastSquares, FVM::CellGradientScheme::GaussLinear})
    {
        SCOPED_TRACE(static_cast<int>(gradient));
        exercise(mesh, GetParam(), false, FVM::NonOrthogonalTreatment::Implicit, gradient);
    }
}

/** Keep mixed HEX_8/prism-side incidence compact through non-orthogonal solves. */
TEST_P(CoupledSolverBackendsTest, MultiRegionMixedNonOrthogonal)
{
    SP<const Handle> mesh = std::make_shared<Handle>(test::mixed_regions());
    for (const auto treatment : {FVM::NonOrthogonalTreatment::Explicit, FVM::NonOrthogonalTreatment::Implicit,
             FVM::NonOrthogonalTreatment::Hybrid})
    {
        SCOPED_TRACE(static_cast<int>(treatment));
        for (const auto gradient : {FVM::CellGradientScheme::LeastSquares, FVM::CellGradientScheme::GaussLinear})
        {
            SCOPED_TRACE(static_cast<int>(gradient));
            exercise(mesh, GetParam(), false, treatment, gradient);
        }
    }
}
TEST_P(CoupledSolverBackendsTest, MultiRegionExtendedFamilies)
{
    for(const auto& geometry:{test::coarse_fine_regions(),test::periodic_regions(),test::cylindrical_regions(),test::independent_extruded_regions()})
    {
        SP<const Handle> mesh=std::make_shared<Handle>(geometry);
        exercise(mesh,GetParam(),false,FVM::NonOrthogonalTreatment::Implicit);
    }
}
INSTANTIATE_TEST_SUITE_P(SupportedSolvers, CoupledSolverBackendsTest,
    testing::Values(Family::Fluid, Family::Isothermal, Family::Boussinesq, Family::PhysicalBoussinesq),
    [](const testing::TestParamInfo<Family>& p)
    {
        switch (p.param)
        {
            case Family::Fluid:
                return "Fluid";
            case Family::Isothermal:
                return "Isothermal";
            case Family::Boussinesq:
                return "Boussinesq";
            case Family::PhysicalBoussinesq:
                return "PhysicalBoussinesq";
        }
        return "Unknown";
    });
} // namespace
