/** Analytic closed-piston and portable planar QBMM restart tests. */
#include "QbmmRestartTestSupport.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace RestartIO = Testing::QbmmRestart;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using Shape = R::QbmmShapeState;
constexpr double p0 = 100000, t0 = 300, gas_constant = 8.31446261815324, cv = 20;
constexpr double molar_mass = .028, mass_scale = .01, dt = .1;
constexpr char target_identity[] = "Q7a prescribed continuation";
constexpr unsigned target_revision = 5;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

enum class Geometry
{
    CartesianX,
    CartesianZ,
    PeriodicCartesianZ,
    CylindricalZ,
    SemiStructuredZ
};

std::shared_ptr<Flow::Mesh> make_mesh(Geometry geometry, const ArrReal& edges)
{
    Flow::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    if (geometry == Geometry::SemiStructuredZ)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::SemiStructuredXY_Z>(
                Arr<Meshes::SemiStructuredXY_Z::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}},
                Arr<Arr<unsigned>>{{0, 1, 2, 3}}, edges),
            distribution);
    if (geometry == Geometry::CylindricalZ)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1, 2}, ArrReal{0, 1}, edges}),
            distribution);
    const auto axes = geometry == Geometry::CartesianX ? Vec3D<ArrReal>{edges, ArrReal{0, 1}, ArrReal{0, 1}}
                                                       : Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, edges};
    return std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cartesian>(
                                            axes, Vec3D<bool>{false, false, geometry == Geometry::PeriodicCartesianZ}),
        distribution);
}

struct Piston
{
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    int eos_calls = 0;
    std::function<double(double)> viscosity_at_pressure;
    std::shared_ptr<const R::MixtureThermodynamics> gas;
    Flow::Options options;
    Flow flow;
    Flow::PlanarMotion motion;

    explicit Piston(Geometry geometry, const ArrReal& edges = {0, 1}, PlanarALEMeshMotionOptions motion_options = {},
        double viscosity = 0, R::QbmmClosureVersions policies = {})
        : mesh(make_mesh(geometry, edges)),
          catalog(std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
              {47, "finite gas", molar_mass, {{"G", 1}}, {R::SpeciesPhase::Dispersed}}})),
          viscosity_at_pressure([viscosity](double) { return viscosity; }),
          gas(R::make_ideal_gas_mixture(catalog, {{cv, t0, -4000}}, {250, 600, 10000, 2000000},
              R::EnergyConvention::IncludesFormationEnergy, gas_constant,
              [this](double, double pressure, std::span<const double>)
              {
                  ++eos_calls;
                  return std::pair{viscosity_at_pressure(pressure), 0.};
              })),
          options(make_options(catalog, std::move(policies))), flow(mesh, {gas}, options),
          motion(mesh,
              [&]
              {
                  if (geometry == Geometry::CartesianX)
                      motion_options.axis = Dimension::X;
                  return motion_options;
              }())
    {
    }

    static Flow::Options make_options(
        const std::shared_ptr<const R::SpeciesCatalog>& catalog, R::QbmmClosureVersions policies)
    {
        Flow::Options options;
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.phase_roles = {{Flow::PhaseRole::DispersedGas, "piston bubbles"}};
        options.qbmm_families = {
            std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "piston bubbles",
                options.phase_roles[0], mass_scale, std::vector<R::QbmmMassFraction>{{47, 1}}, std::move(policies))};
        return options;
    }

    void initialize()
    {
        Flow::State state;
        state.thermodynamic_pressure = p0;
        state.qbmm_families = {Flow::qbmm_family_stamp(*options.qbmm_families[0])};
        state.cells.resize(mesh->num_owned_cells());
        for (size_t i = 0; i < state.cells.size(); ++i)
        {
            const double volume = mesh->cell_volume(static_cast<LO>(i));
            const double n = p0 * volume / (gas_constant * t0), w = n * molar_mass / (5 * mass_scale);
            auto& cell = state.cells[i];
            cell.phases = {{gas, {n}, p0, -4000 * n, t0}};
            cell.phase_numbers = {3 * w};
            cell.qbmm_shape = {Shape{11 * w, 29 * w}};
        }
        flow.initialize(std::move(state));
    }

    double sum(double x) const
    {
        double total = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &x, &total);
        return total;
    }

    double maximum(double x) const
    {
        double max = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &x, &max);
        return max;
    }

    void advance(double target)
    {
        const auto trial = flow.preview_ale(dt, motion, target);
        flow.commit_ale(trial, motion);
    }

    std::vector<std::byte> archive() const
    {
        return flow.serialize_qbmm_restart(motion, target_identity, target_revision);
    }

    void restore(std::span<const std::byte> bytes)
    {
        flow.restore_qbmm_restart(bytes, motion, target_identity, target_revision);
    }
};

void expect_same_endpoint(const Piston& a, const Piston& b);

void expect_restored_motion_frame(const Piston& f, double surface, std::span<const double> previous_absolute_flux)
{
    EXPECT_FALSE(f.motion.has_active_trial());
    double local_wall_flux = 0;
    for (size_t i = 0; i < f.mesh->num_owned_faces(); ++i)
    {
        const auto face = static_cast<LO>(i);
        if (f.mesh->is_geometry_exterior_face(face))
            local_wall_flux += std::abs(f.flow.volume_flux().owned_value(face));
    }
    EXPECT_GT(f.sum(local_wall_flux), 0.);

    const auto& diagnostics = f.motion.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.old_surface_elevation, surface);
    EXPECT_DOUBLE_EQ(diagnostics.new_surface_elevation, surface);
    EXPECT_DOUBLE_EQ(diagnostics.time_step, 0.);
    const auto old_volumes = f.motion.old_cell_volumes(), new_volumes = f.motion.new_cell_volumes();
    ASSERT_EQ(old_volumes.size(), f.mesh->num_local_cells());
    ASSERT_EQ(new_volumes.size(), f.mesh->num_local_cells());
    for (size_t c = 0; c < old_volumes.size(); ++c)
    {
        const double actual = f.mesh->cell_volume(static_cast<LO>(c));
        EXPECT_DOUBLE_EQ(old_volumes[c], actual);
        EXPECT_DOUBLE_EQ(new_volumes[c], actual);
    }
    const auto mesh_sweeps = f.motion.face_mesh_fluxes();
    ASSERT_EQ(mesh_sweeps.size(), f.mesh->num_faces());
    for (const double sweep : mesh_sweeps)
        EXPECT_DOUBLE_EQ(sweep, 0.);

    // Mesh motion is stationary after restore while solver flux retains the accepted interval.
    ASSERT_EQ(previous_absolute_flux.size(), f.mesh->num_owned_faces());
    for (size_t i = 0; i < previous_absolute_flux.size(); ++i)
        EXPECT_DOUBLE_EQ(f.flow.volume_flux().owned_value(static_cast<LO>(i)), previous_absolute_flux[i]);
}

TEST(LowMachQbmmPlanarALETest, ClosedPistonPreservesExtensiveMomentsAndMatchesPressureTemperatureAndWork)
{
    double pressure_error = 0, temperature_error = 0, work_error = 0;
    // Evaluating accepted stocks on trial geometry (or the reverse at the endpoint)
    // produces an O(1%) volume-fraction error and must trip this hard transport gate.
    R::QbmmClosureVersions policies;
    policies.laws = {{"occupied piston gas", 1, R::QbmmLawKind::TransportProperty, "gas volume / bulk volume",
        "dimensionless", R::QbmmApplicabilityGate::RejectOutside, true,
        {{R::QbmmRangeVariable::GasVolumeFraction, .999999, 1.000001, {}}}}};
    for (const auto geometry :
        {Geometry::CartesianX, Geometry::CartesianZ, Geometry::CylindricalZ, Geometry::SemiStructuredZ})
        for (const double ratio : {.99, 1.01})
        {
            SCOPED_TRACE(static_cast<int>(geometry));
            SCOPED_TRACE(ratio);
            Piston f(geometry, {0, 1}, {}, 0., policies);
            f.initialize();
            const auto& accepted = f.flow.state();
            const auto initial = f.flow.qbmm_report();
            const auto trial = f.flow.preview_ale(dt, f.motion, ratio);
            const auto report = f.flow.qbmm_report(trial);
            EXPECT_EQ(f.flow.state().time, 0);
            EXPECT_FALSE(f.motion.has_active_trial());
            EXPECT_EQ(
                f.sum(f.mesh->num_owned_cells() == 0 ? 1 : 0), f.mesh->owned_cell_map()->getComm()->getSize() - 1);
            EXPECT_EQ(f.sum(static_cast<double>(f.mesh->num_owned_cells())), 1);
            ASSERT_TRUE(report.interval);
            EXPECT_EQ(report.provenance.geometry_epoch, trial.transport_snapshot().new_geometry_epoch);

            // n cv (T1 - T0) = -p1 (V1 - V0), with p1 V1 = n R T1.
            const double p1 = p0 / (ratio + (gas_constant / cv) * (ratio - 1));
            const double t1 = t0 * ratio * p1 / p0;
            pressure_error = std::max(pressure_error, std::abs(trial.state().thermodynamic_pressure - p1));
            EXPECT_NEAR(trial.state().thermodynamic_pressure, p1, 1e-5);
            double local_work = 0, local_temperature_error = 0, local_sweep = 0;
            for (size_t i = 0; i < accepted.cells.size(); ++i)
            {
                const auto& previous = *initial.cells[i].families[0];
                const auto& endpoint = *report.cells[i].families[0];
                ASSERT_EQ(endpoint.applicability.size(), 1u);
                EXPECT_TRUE(endpoint.applicability.front().gate_passed);
                const auto& cell = trial.state().cells[i];
                const auto& transported = trial.transport_snapshot().cells[i].phases[0];
                const double v0 = initial.cells[i].bulk_volume_m3, v1 = v0 * ratio;
                const double n = p0 * v0 / (gas_constant * t0), w = n * molar_mass / (5 * mass_scale);
                const std::array moments{3 * w, 5 * w, 11 * w, 29 * w};
                const double work = -p1 * (v1 - v0);
                local_work += work;
                local_temperature_error = std::max(local_temperature_error, std::abs(cell.phases[0].temperature - t1));
                EXPECT_NEAR(cell.phases[0].temperature, t1, 1e-7);
                EXPECT_NEAR(cell.phases[0].internal_energy - accepted.cells[i].phases[0].internal_energy, work, 2e-6);
                EXPECT_DOUBLE_EQ(trial.transport_snapshot().old_control_volumes[i], v0);
                EXPECT_DOUBLE_EQ(transported.advected_material_volume, v0);
                EXPECT_NEAR(report.cells[i].bulk_volume_m3, v1, 1e-13 * v1);
                EXPECT_NEAR(endpoint.phase_volume_m3, v1, f.options.volume_relative_tolerance * v1);
                EXPECT_NEAR(endpoint.bulk_concentration_mol_per_m3[0], n / v1, 1e-12 * n / v1);
                ASSERT_TRUE(endpoint.phase_concentration_mol_per_m3);
                EXPECT_NEAR((*endpoint.phase_concentration_mol_per_m3)[0], n / v1, 3e-8 * n / v1);
                EXPECT_NE(endpoint.bulk_concentration_mol_per_m3, previous.bulk_concentration_mol_per_m3);
                EXPECT_EQ(endpoint.quadrature.status, R::InversionStatus::TwoNodes);
                const std::array<double, 2> weight_fractions{2. / 3, 1. / 3}, normalized_mass{1, 3};
                for (size_t node = 0; node < weight_fractions.size(); ++node)
                {
                    EXPECT_NEAR(
                        endpoint.quadrature.atoms[node].count / endpoint.moments[0], weight_fractions[node], 2e-13);
                    EXPECT_NEAR(endpoint.quadrature.atoms[node].mass_over_scale, normalized_mass[node], 2e-13);
                }
                ASSERT_TRUE(endpoint.geometry);
                ASSERT_TRUE(previous.geometry);
                EXPECT_NEAR(endpoint.geometry->diameter_m[1] / endpoint.geometry->diameter_m[0], std::cbrt(3.), 2e-13);
                EXPECT_NEAR(
                    endpoint.geometry->diameter_m[0] / previous.geometry->diameter_m[0], std::cbrt(ratio), 3e-8);
                for (size_t k = 0; k < moments.size(); ++k)
                {
                    const double density = moments[k] * std::pow(mass_scale, k) / v1;
                    EXPECT_NEAR(endpoint.moments[k], moments[k], 2e-12 * moments[k]);
                    EXPECT_NEAR(endpoint.mass_moment_density_si[k], density, 2e-12 * density);
                }
                EXPECT_NEAR(transported.internal_energy, -4000 * n, 1e-9);
            }
            for (size_t i = 0; i < trial.owned_phase_transport().size(); ++i)
            {
                const auto& face = trial.owned_phase_transport()[i];
                EXPECT_TRUE(f.mesh->is_geometry_exterior_face(static_cast<LO>(i)));
                EXPECT_DOUBLE_EQ(trial.owned_volume_flux()[i], face.mesh_volume_flux);
                EXPECT_DOUBLE_EQ(face.relative_phase_volume_flux[0], 0);
                EXPECT_DOUBLE_EQ(face.qbmm_flux[0]->derived_mass_flux, 0);
                local_sweep += std::abs(face.mesh_volume_flux);
            }
            EXPECT_GT(f.sum(local_sweep), 0);
            temperature_error = std::max(temperature_error, f.maximum(local_temperature_error));
            const double work = f.sum(local_work);
            work_error = std::max(work_error, std::abs(trial.diagnostics().pressure_work - work));
            EXPECT_NEAR(trial.diagnostics().pressure_work, work, 2e-6);
            const int calls = f.eos_calls;
            f.flow.commit_ale(trial, f.motion);
            EXPECT_EQ(f.eos_calls, calls);
            EXPECT_EQ(f.flow.state().time, dt);
            const auto committed = f.flow.qbmm_report();
            EXPECT_EQ(committed.receipt_history, Flow::QbmmReceiptHistory::Available);
            for (size_t i = 0; i < report.cells.size(); ++i)
                EXPECT_EQ(committed.cells[i].bulk_volume_m3, report.cells[i].bulk_volume_m3);
            Piston restored(geometry, {0, 1}, {}, 0., policies);
            restored.restore(f.archive());
            expect_same_endpoint(f, restored);
            expect_restored_motion_frame(restored, ratio, trial.owned_volume_flux());
        }
    const auto record = [](const char* name, double value)
    {
        std::ostringstream out;
        out << std::setprecision(17) << value;
        testing::Test::RecordProperty(name, out.str());
    };
    record("maximum_piston_pressure_error_pa", pressure_error);
    record("maximum_piston_temperature_error_k", temperature_error);
    record("maximum_piston_work_error_j", work_error);
}

TEST(LowMachQbmmPlanarALETest, PeriodicTopologyAndViscosityRejectWithoutPublishing)
{
    for (const auto geometry : {Geometry::PeriodicCartesianZ, Geometry::CartesianZ})
    {
        SCOPED_TRACE(static_cast<int>(geometry));
        Piston f(geometry, {0, .5, 1}, {}, geometry == Geometry::CartesianZ ? .01 : 0.);
        f.initialize();
        const auto retained = f.flow.preview(dt);
        const auto before = f.flow.serialize_qbmm_restart();
        EXPECT_THROW((void) f.flow.preview_ale(dt, f.motion, 1.01), std::exception);
        EXPECT_FALSE(f.motion.has_active_trial());
        EXPECT_EQ(f.flow.serialize_qbmm_restart(), before);
        f.flow.commit(retained);
        EXPECT_EQ(f.flow.state().time, dt);
    }
}

TEST(LowMachQbmmPlanarALETest, ReusedMotionAddressRejectsEarlierAuthorityOnTheSameMesh)
{
    Piston reference(Geometry::CartesianZ);
    reference.initialize();
    auto mesh = make_mesh(Geometry::CartesianZ, {0, 1});
    Flow flow(mesh, {reference.gas}, reference.options);
    flow.initialize(reference.flow.state());
    std::optional<Flow::PlanarMotion> motion;
    motion.emplace(mesh);
    const auto* address = std::addressof(*motion);
    const auto checkpoint = flow.checkpoint(*motion);
    const auto earlier = flow.preview_ale(dt, *motion, 1.001);
    motion.reset();
    motion.emplace(mesh);
    EXPECT_EQ(std::addressof(*motion), address);
    EXPECT_THROW(flow.commit_ale(earlier, *motion), std::exception);
    EXPECT_THROW(flow.restore(checkpoint, *motion), std::exception);
    EXPECT_EQ(flow.state().time, 0.);
    EXPECT_FALSE(motion->has_active_trial());

    const auto replacement = flow.preview_ale(dt, *motion, .999);
    flow.commit_ale(replacement, *motion);
    EXPECT_EQ(flow.state().time, dt);
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<LO>(c)), .999);
}

TEST(LowMachQbmmPlanarALETest, ReusedSolverAddressRejectsEarlierCertificatesWithSameMotion)
{
    Piston reference(Geometry::CartesianZ);
    reference.initialize();
    for (const bool restore : {false, true})
    {
        SCOPED_TRACE(restore);
        auto mesh = make_mesh(Geometry::CartesianZ, {0, 1});
        Flow::PlanarMotion motion(mesh);
        std::optional<Flow> solver;
        solver.emplace(mesh, std::vector{reference.gas}, reference.options);
        solver->initialize(reference.flow.state());
        const auto* address = std::addressof(*solver);
        const auto checkpoint = solver->checkpoint(motion);
        const auto earlier = solver->preview_ale(dt, motion, 1.001);
        solver.reset();
        solver.emplace(mesh, std::vector{reference.gas}, reference.options);
        solver->initialize(reference.flow.state());
        EXPECT_EQ(std::addressof(*solver), address);
        EXPECT_THROW((void) solver->qbmm_report(earlier), std::exception);
        if (restore)
            EXPECT_THROW(solver->restore(checkpoint, motion), std::exception);
        else
            EXPECT_THROW(solver->commit_ale(earlier, motion), std::exception);
        EXPECT_EQ(solver->state().time, 0.);
        EXPECT_FALSE(motion.has_active_trial());
        const auto replacement = solver->preview_ale(dt, motion, .999);
        EXPECT_NO_THROW((void) solver->qbmm_report(replacement));
        solver->commit_ale(replacement, motion);
        EXPECT_EQ(solver->state().time, dt);
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
            EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<LO>(c)), .999);
    }
}

void expect_same_endpoint(const Piston& a, const Piston& b)
{
    const auto& left = a.flow.state();
    const auto& right = b.flow.state();
    EXPECT_EQ(left.time, right.time);
    EXPECT_EQ(left.revision, right.revision);
    EXPECT_DOUBLE_EQ(left.thermodynamic_pressure, right.thermodynamic_pressure);
    EXPECT_EQ(left.qbmm_families, right.qbmm_families);
    ASSERT_EQ(left.cells.size(), right.cells.size());
    for (size_t i = 0; i < left.cells.size(); ++i)
    {
        const auto& l = left.cells[i];
        const auto& r = right.cells[i];
        const LO cell = static_cast<LO>(i);
        EXPECT_DOUBLE_EQ(a.mesh->cell_volume(cell), b.mesh->cell_volume(cell));
        EXPECT_EQ(a.mesh->cell_centroid(cell), b.mesh->cell_centroid(cell));
        EXPECT_DOUBLE_EQ(l.mechanical_pressure, r.mechanical_pressure);
        EXPECT_EQ(l.phase_numbers, r.phase_numbers);
        EXPECT_EQ(l.qbmm_shape, r.qbmm_shape);
        EXPECT_EQ(l.velocity, r.velocity);
        EXPECT_EQ(l.phases[0].moles, r.phases[0].moles);
        EXPECT_DOUBLE_EQ(l.phases[0].internal_energy, r.phases[0].internal_energy);
        EXPECT_DOUBLE_EQ(l.phases[0].temperature, r.phases[0].temperature);
        EXPECT_DOUBLE_EQ(l.phases[0].absolute_pressure, r.phases[0].absolute_pressure);
    }
    ASSERT_EQ(a.mesh->num_faces(), b.mesh->num_faces());
    for (size_t i = 0; i < a.mesh->num_faces(); ++i)
        EXPECT_DOUBLE_EQ(
            a.flow.volume_flux().local_value(static_cast<LO>(i)), b.flow.volume_flux().local_value(static_cast<LO>(i)));
}

TEST(LowMachQbmmPlanarALETest, RankLocalViscousRestartCandidateRejectsWithoutConsumingEarlierTrial)
{
    constexpr double archived_target = 1.002, accepted_target = .999, next_target = 1.001;
    Piston archived(Geometry::CartesianZ);
    archived.initialize();
    archived.advance(archived_target);
    const auto incoming = archived.archive();
    Piston receiver(Geometry::CartesianZ);
    receiver.initialize();
    receiver.advance(accepted_target);
    const auto accepted = receiver.archive();
    const auto earlier = receiver.flow.preview_ale(dt, receiver.motion, next_target);
    const auto earlier_report = receiver.flow.qbmm_report(earlier);
    Piston expected(Geometry::CartesianZ);
    expected.restore(accepted);

    const double archived_pressure = archived.flow.state().thermodynamic_pressure;
    const double receiver_pressure = receiver.flow.state().thermodynamic_pressure;
    ASSERT_LT(archived_pressure, receiver_pressure);
    const double threshold = .5 * (archived_pressure + receiver_pressure);
    const bool owns_cell = receiver.mesh->num_owned_cells() != 0;
    EXPECT_EQ(receiver.sum(owns_cell ? 1. : 0.), 1.);
    int positive_viscosity_evaluations = 0;
    auto original_viscosity = receiver.viscosity_at_pressure;
    receiver.viscosity_at_pressure = [&](double pressure)
    {
        // Only the real owner can reject, and only when its decoded candidate is sampled.
        if (!owns_cell || pressure >= threshold)
            return 0.;
        ++positive_viscosity_evaluations;
        return .01;
    };
    EXPECT_THROW(receiver.restore(incoming), std::exception);
    receiver.viscosity_at_pressure = std::move(original_viscosity);
    EXPECT_GT(receiver.sum(positive_viscosity_evaluations), 0);
    EXPECT_FALSE(receiver.motion.has_active_trial());
    EXPECT_EQ(receiver.archive(), accepted);
    expect_same_endpoint(receiver, expected);
    const auto retained = receiver.flow.qbmm_report(earlier);
    EXPECT_EQ(retained.provenance.preview_id, earlier_report.provenance.preview_id);
    EXPECT_EQ(retained.provenance.geometry_epoch, earlier_report.provenance.geometry_epoch);
    receiver.flow.commit_ale(earlier, receiver.motion);
    expected.advance(next_target);
    expect_same_endpoint(receiver, expected);
}

TEST(LowMachQbmmPlanarALETest, ChecksumValidWallFluxMismatchPreservesAcceptedEndpointAndEarlierTrial)
{
    constexpr double archived_target = 1.002, accepted_target = .999, next_target = 1.001;
    Piston archived(Geometry::CartesianZ);
    archived.initialize();
    archived.advance(archived_target);
    const auto valid = archived.archive();
    constexpr auto word = sizeof(std::uint64_t);
    // The v6 header has 20 words. Its single family stamp has four words.
    // Each owned cell has four carrier words, six thermal/drift words, one
    // species word, two number words and three shape words in this fixture.
    const auto flux_begin = word * (24 + 16 * archived.mesh->num_owned_cells());
    ASSERT_GE(valid.size(), flux_begin + word * (archived.mesh->num_owned_faces() + 2));
    ASSERT_EQ(valid[word], std::byte{6});
    Piston receiver(Geometry::CartesianZ);
    receiver.initialize();
    receiver.advance(accepted_target);
    const auto checkpoint = receiver.flow.checkpoint(receiver.motion);
    const auto accepted_bytes = receiver.archive();
    Piston accepted(Geometry::CartesianZ);
    accepted.restore(accepted_bytes);
    // Re-encoding a portable endpoint preserves the evidence for its nonzero accepted sweep.
    Piston reencoded(Geometry::CartesianZ);
    reencoded.restore(accepted.archive());
    expect_same_endpoint(accepted, reencoded);
    const auto earlier = receiver.flow.preview_ale(dt, receiver.motion, next_target);
    const auto earlier_report = receiver.flow.qbmm_report(earlier);
    const auto comm = archived.mesh->owned_cell_map()->getComm();
    for (const int normal_z : {-1, 0, 1})
    {
        SCOPED_TRACE(normal_z);
        // Alter one owned exterior receipt: stationary bottom, stationary side, or moving top.
        size_t face_index = archived.mesh->num_owned_faces();
        for (size_t f = 0; f < archived.mesh->num_owned_faces(); ++f)
        {
            const auto face = static_cast<LO>(f);
            if (!archived.mesh->is_geometry_exterior_face(face))
                continue;
            const double z = archived.mesh->face_area_vector(face).z;
            if ((z > 0) - (z < 0) == normal_z)
            {
                face_index = f;
                break;
            }
        }
        const int candidate = face_index < archived.mesh->num_owned_faces() ? comm->getRank() : comm->getSize();
        int owner = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &candidate, &owner);
        ASSERT_LT(owner, comm->getSize());
        auto damaged = valid;
        damaged.resize(damaged.size() - 2 * word);
        if (comm->getRank() == owner)
        {
            const double magnitude = archived.flow.volume_flux().owned_value(static_cast<LO>(face_index));
            if (normal_z == 1)
                EXPECT_NEAR(magnitude, (archived_target - 1.) / dt, 1e-14);
            else
                EXPECT_EQ(magnitude, 0.);
            const auto position = damaged.begin() + flux_begin + word * face_index;
            std::vector<std::byte> encoded;
            RestartIO::append_restart_word(encoded, std::bit_cast<std::uint64_t>(magnitude));
            EXPECT_TRUE(std::equal(encoded.begin(), encoded.end(), position));
            encoded.clear();
            RestartIO::append_restart_word(encoded, std::bit_cast<std::uint64_t>(magnitude + .125));
            std::copy(encoded.begin(), encoded.end(), position);
        }
        RestartIO::append_restart_checksums(*comm, damaged);
        EXPECT_THROW(receiver.restore(damaged), std::exception);
        EXPECT_FALSE(receiver.motion.has_active_trial());
        EXPECT_EQ(receiver.archive(), accepted_bytes);
        expect_same_endpoint(receiver, accepted);
        const auto report = receiver.flow.qbmm_report(earlier);
        EXPECT_EQ(report.provenance.preview_id, earlier_report.provenance.preview_id);
        EXPECT_EQ(report.provenance.geometry_epoch, earlier_report.provenance.geometry_epoch);
        for (size_t i = 0; i < report.cells.size(); ++i)
            EXPECT_EQ(report.cells[i].bulk_volume_m3, earlier_report.cells[i].bulk_volume_m3);
    }
    receiver.flow.commit_ale(earlier, receiver.motion);
    reencoded.advance(next_target);
    expect_same_endpoint(receiver, reencoded);
    receiver.flow.restore(checkpoint, receiver.motion);
    // Checkpoint replay supplies the same sweep provenance to a new portable image.
    reencoded.restore(receiver.archive());
    expect_same_endpoint(receiver, accepted);
    expect_same_endpoint(receiver, reencoded);
}

TEST(LowMachQbmmPlanarALETest, PortableRestartContinuesFromOriginalReference)
{
    const char* writer = std::getenv("SIMPLEFLUID_Q7A_RESTART_WRITE");
    const char* reader = std::getenv("SIMPLEFLUID_Q7A_RESTART_READ");
    ASSERT_FALSE(writer && reader);
    constexpr double first_target = 1.002, next_target = .997;
    const ArrReal reference{0, .17, .55, 1};
    PlanarALEMeshMotionOptions options;
    options.deformation_start_elevation = .17;
    options.maximum_level_change = .02;
    Piston baseline(Geometry::CartesianZ, reference, options);
    const auto comm = baseline.mesh->owned_cell_map()->getComm();
    if (writer)
    {
        baseline.initialize();
        baseline.advance(first_target);
        RestartIO::write_restart(*comm, writer, ".first", baseline.archive());
        baseline.advance(next_target);
        RestartIO::write_restart(*comm, writer, ".next", baseline.archive());
        return;
    }
    if (reader)
    {
        baseline.restore(RestartIO::read_restart(*comm, reader, ".next"));
        Piston fresh(Geometry::CartesianZ, reference, options);
        fresh.restore(RestartIO::read_restart(*comm, reader, ".first"));
        fresh.advance(next_target);
        expect_same_endpoint(baseline, fresh);
        return;
    }

    baseline.initialize();
    Flow::Trial stale = baseline.flow.preview_ale(dt, baseline.motion, first_target);
    Piston different_authority(Geometry::CartesianZ, reference, options);
    EXPECT_THROW(baseline.flow.commit_ale(stale, different_authority.motion), std::exception);
    EXPECT_THROW(baseline.flow.commit(stale), std::exception);
    const auto other = baseline.flow.preview_ale(dt, baseline.motion, next_target);
    if (comm->getSize() > 1)
    {
        const auto& mixed = comm->getRank() == 0 ? stale : other;
        EXPECT_THROW(baseline.flow.commit_ale(mixed, baseline.motion), std::exception);
        EXPECT_THROW((void) baseline.flow.qbmm_report(mixed), std::exception);
    }
    EXPECT_EQ(baseline.flow.state().time, 0);
    baseline.flow.commit_ale(stale, baseline.motion);
    const auto first = baseline.archive();
    const auto saved = baseline.flow.checkpoint(baseline.motion);
    const auto first_report = baseline.flow.qbmm_report();
    EXPECT_EQ(first_report.receipt_history, Flow::QbmmReceiptHistory::Available);

    Piston fresh(Geometry::CartesianZ, reference, options);
    fresh.restore(first);
    expect_same_endpoint(baseline, fresh);
    expect_restored_motion_frame(fresh, first_target, stale.owned_volume_flux());
    const auto report = fresh.flow.qbmm_report();
    EXPECT_EQ(report.provenance.origin, Flow::QbmmStateOrigin::PortableRestart);
    EXPECT_EQ(report.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    EXPECT_FALSE(report.interval);
    auto damaged = first;
    ASSERT_FALSE(damaged.empty());
    if (comm->getRank() == 0)
        damaged.pop_back();
    EXPECT_THROW(fresh.restore(damaged), std::exception);
    EXPECT_THROW(fresh.flow.restore_qbmm_restart(first, fresh.motion, "other target", target_revision), std::exception);
    EXPECT_THROW(
        fresh.flow.restore_qbmm_restart(first, fresh.motion, target_identity, target_revision + 1), std::exception);
    expect_same_endpoint(baseline, fresh);
    Piston wrong_reference(Geometry::CartesianZ, {0, .17, .56, 1}, options);
    EXPECT_THROW(wrong_reference.restore(first), std::exception);
    EXPECT_FALSE(wrong_reference.flow.initialized());
    auto changed_options = options;
    changed_options.deformation_start_elevation = .16;
    Piston wrong_options(Geometry::CartesianZ, reference, changed_options);
    EXPECT_THROW(wrong_options.restore(first), std::exception);
    EXPECT_FALSE(wrong_options.flow.initialized());

    const auto next = baseline.flow.preview_ale(dt, baseline.motion, next_target);
    baseline.flow.commit_ale(next, baseline.motion);
    fresh.advance(next_target);
    expect_same_endpoint(baseline, fresh);
    baseline.flow.restore(saved, baseline.motion);
    EXPECT_EQ(baseline.flow.state().time, dt);
    expect_restored_motion_frame(baseline, first_target, stale.owned_volume_flux());
    const auto restored_report = baseline.flow.qbmm_report();
    EXPECT_EQ(restored_report.receipt_history, Flow::QbmmReceiptHistory::Available);
    ASSERT_TRUE(restored_report.interval);
    ASSERT_TRUE(first_report.interval);
    EXPECT_EQ(restored_report.interval->preview_id, first_report.interval->preview_id);
    EXPECT_THROW(baseline.flow.commit_ale(next, baseline.motion), std::exception);
    baseline.advance(next_target);
    expect_same_endpoint(baseline, fresh);
}
} // namespace
