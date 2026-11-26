/** Shared forced planar ALE hydrostatic and caloric piston tests. */
#pragma once

#include "QbmmCylindricalTestMesh.hh"
#include "QbmmKineticTestSupport.hh"
#include "solvers/ReactingMixtureCoupler.hh"

#include <cmath>
#include <cstdlib>
#include <numbers>
#include <numeric>

namespace SimpleFluid::Testing::QbmmPlanarGravity
{
using namespace SimpleFluid::Testing::QbmmKinetic;
using Coupler = R::ReactingMixtureCoupler;
using Velocity = Model::Velocity;
using Vec = Flow::Vector;
using Limiter = Model::LinearSlopeLimiter;

enum class Geometry
{
    Cartesian,
    Sector,
    Extruded,
    CartesianX,
    CartesianY,
    Disk,
    Annulus
};

constexpr double p0 = 64., t0 = 8., rho_liquid = 16., rho_gas = 8., cv = 2., dt = 1. / 64.;
// Molecular mass and gas constant are both one in SI units, so mass equals mole amount.
constexpr double scale = .125;

inline Dimension column_axis(Geometry kind)
{
    if (kind == Geometry::CartesianX)
        return Dimension::X;
    return kind == Geometry::CartesianY ? Dimension::Y : Dimension::Z;
}

inline unsigned cross_section_cells(Geometry kind)
{
    if (kind == Geometry::Disk || kind == Geometry::Annulus)
        return QbmmCylinder::wedges(QbmmCylinder::Geometry::PolygonalDisk);
    return 1;
}

// Total analytic area of an annular sector or regular polygonal cross section.
inline double cross_section_area(Geometry kind)
{
    if (kind == Geometry::Sector)
        return 1.5;
    if (kind == Geometry::Disk || kind == Geometry::Annulus)
    {
        const unsigned n = cross_section_cells(kind);
        const double radii_squared_difference = kind == Geometry::Disk ? 4. : 3.;
        return .5 * n * radii_squared_difference * std::sin(2. * std::numbers::pi / n);
    }
    return 1.;
}

inline SP<Flow::Mesh> column_mesh(Geometry kind, size_t layers)
{
    const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    for (const char* key : {"SIMPLEFLUID_Q8E_RANKS", "SIMPLEFLUID_Q8F_RANKS"})
        if (const char* expected = std::getenv(key))
            EXPECT_EQ(ranks, std::stoul(expected));
    testing::Test::RecordProperty("mpi_ranks", int(ranks));
    ArrReal z;
    for (size_t j = 0; j <= layers; ++j)
        z.push_back(double(j) / layers);
    Flow::Mesh::DistributionOptions partition{.allow_empty_partitions = true};
    if (kind == Geometry::Disk || kind == Geometry::Annulus)
        return QbmmCylinder::mesh(
            kind == Geometry::Disk ? QbmmCylinder::Geometry::PolygonalDisk : QbmmCylinder::Geometry::PolygonalAnnulus,
            1, z);
    if (kind == Geometry::Extruded)
    {
        using Semi = Meshes::SemiStructuredXY_Z;
        partition.xy_partitions = 1;
        partition.z_partitions = ranks;
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Semi>(
                Arr<Semi::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}, Arr<Arr<unsigned>>{{0, 1, 2, 3}}, z),
            partition);
    }
    const auto axis = column_axis(kind);
    std::array<size_t, 3> dimensions{1, 1, 1};
    dimensions[axis] = ranks;
    partition.orthogonal_partitions = dimensions;
    if (kind == Geometry::Sector)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1, 2}, {0, 1}, z}), partition);
    Vec3D<ArrReal> edges{ArrReal{0., 1.}, ArrReal{0., 1.}, ArrReal{0., 1.}};
    edges[axis] = z;
    return std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cartesian>(edges), partition);
}

inline double gas_fraction(size_t j)
{
    return (j + 1.) / 8.;
}
inline double layer_density(size_t j)
{
    return (1. - gas_fraction(j)) * rho_liquid + gas_fraction(j) * rho_gas;
}

struct Column
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<Coupler> controller;
    std::vector<double> pressure;
    Dimension axis;
    double target = 1.;

    Column(Geometry kind, Velocity velocity, size_t layers, std::optional<double> gravity, bool moving,
        bool count_only = false, double absolute_linear_tolerance = 0., std::optional<Limiter> limiter = {},
        double liquid_conductance = 0.)
        : mesh(column_mesh(kind, layers)), pressure(layers, p0), axis(column_axis(kind))
    {
        using Phase = R::SpeciesPhase;
        auto species = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "column material", 1., {{"G", 1}}, {Phase::Liquid, Phase::Dispersed}}});
        const R::ThermodynamicBounds bounds{2., 16., 16., 128.};
        auto liquid = R::make_linear_compressible_mixture_liquid(species, bounds,
            [](auto) { return R::LinearCompressibleMixtureLiquid{t0, p0, rho_liquid, cv, 0., 0., cv * t0, 0., 0.}; });
        auto gas = R::make_ideal_gas_mixture(
            species, {{cv, t0, cv * t0}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
        Flow::Options options;
        options.minimum_pressure = bounds.minimum_pressure;
        options.maximum_pressure = bounds.maximum_pressure;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.linear.absolute_tolerance = absolute_linear_tolerance;
        options.qbmm_model.velocity = velocity;
        if (limiter)
        {
            options.qbmm_model.family_transport = Model::FamilyTransport::RealizableLinearSSPRK2;
            options.qbmm_model.linear_slope_limiter = *limiter;
        }
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "bubbles"}};
        options.qbmm_eos_law_identities = {R::ThermalLawIdentity{"column liquid", 1, species->fingerprint()}, {}};
        Flow::State state;
        if (!count_only)
        {
            options.qbmm_families = {
                nullptr, std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), "bubbles",
                             options.phase_roles[1], scale, std::vector<R::QbmmMassFraction>{{47, 1.}})};
            state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*options.qbmm_families[1])};
        }
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas}, options);
        for (size_t j = 1; j < layers; ++j)
            pressure[j] =
                pressure[j - 1] + .5 * (layer_density(j - 1) + layer_density(j)) * gravity.value_or(0.) / layers;
        state.thermodynamic_pressure = std::accumulate(pressure.begin(), pressure.end(), 0.) / layers;
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            const size_t j = static_cast<size_t>(mesh->cell_centroid(static_cast<LO>(i)).component(axis) * layers);
            const double v = mesh->cell_volume(static_cast<LO>(i)), p = pressure[j], t = p / rho_gas;
            const double ng = gas_fraction(j) * v * rho_gas, nl = (1. - gas_fraction(j)) * v * rho_liquid;
            auto& cell = state.cells.emplace_back();
            cell.mechanical_pressure = p - state.thermodynamic_pressure;
            cell.phases = {{liquid, {nl}, p, nl * cv * t, t}, {gas, {ng}, p, ng * cv * t, t}};
            // Weights 2w and w at dimensionless masses 1 and 3.
            const double w = ng / (5. * scale);
            cell.phase_numbers = {std::nullopt, 3. * w};
            if (!count_only)
                cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{11. * w, 29. * w}};
        }
        flow->initialize(std::move(state));
        R::ReactingMixtureOptions controls;
        controls.mechanism = std::make_shared<const R::ReactionMechanism>(
            species, std::vector<R::ReactionDefinition>{}, liquid->energy_convention());
        controls.phases = {{Phase::Liquid, {}, {}}, {Phase::Dispersed, "bubbles", {}}};
        controls.source_identity = "inert column";
        controls.source = [](const auto&, double) { return R::ReactionSourceInput{}; };
        if (liquid_conductance > 0.)
        {
            const auto fingerprint = species->fingerprint();
            controls.face_transport = {{"liquid conduction", 1, fingerprint, 0, std::nullopt,
                R::MixtureFourierLaw{"geometry-scaled contact conductance", 1, fingerprint,
                    [this, layers, liquid_conductance](const auto& face)
                    {
                        EXPECT_NEAR(face.center_distance, target / layers, 2e-14);
                        return liquid_conductance / (layers * face.center_distance);
                    }}}};
        }
        if (gravity)
        {
            Vec acceleration{};
            acceleration.component(axis) = *gravity;
            controls.body_acceleration = acceleration;
        }
        if (moving)
            controls.planar_motion =
                std::make_shared<const R::MixturePlanarMotionProvider>(R::MixturePlanarMotionProvider{
                    mesh, {.axis = axis}, "prescribed column end", 1, [this](double, double) { return target; }});
        controller = std::make_unique<Coupler>(flow, std::move(controls));
    }

    double sum(double local) const { return reduce(*mesh, Teuchos::REDUCE_SUM, local); }
};

inline double mass(const Flow::CellState& c)
{
    return c.phases[0].moles[0] + c.phases[1].moles[0];
}

inline void metric(const std::string& name, double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    testing::Test::RecordProperty(name, stream.str());
}

inline void check_impulse(
    const Column& column, const Coupler::Trial& trial, double gravity, bool stationary, const std::string& label = {})
{
    const auto& diagnostics = trial.diagnostics().flow;
    Vec acceleration{};
    acceleration.component(column.axis) = gravity;
    EXPECT_EQ(diagnostics.body_acceleration_m_per_s2, acceleration);
    ASSERT_TRUE(diagnostics.impulse_parts);
    double local_mass = 0.;
    Vec local_delta{};
    for (size_t i = 0; i < trial.state().cells.size(); ++i)
    {
        const auto& before = column.flow->state().cells[i];
        const auto& after = trial.state().cells[i];
        local_mass += mass(before);
        local_delta = local_delta + after.velocity * mass(after) - before.velocity * mass(before);
    }
    const Vec body = acceleration * (column.sum(local_mass) * dt);
    double body_error = 0., momentum_error = 0.;
    for (size_t k = 0; k < 3; ++k)
    {
        EXPECT_NEAR(diagnostics.impulse_parts->body.component(k), body.component(k), 3e-13);
        EXPECT_EQ(diagnostics.impulse_parts->other.component(k), 0.);
        EXPECT_NEAR(diagnostics.external_impulse.component(k), body.component(k), 3e-13);
        const double observed = column.sum(local_delta.component(k));
        const double impulse = body.component(k) + diagnostics.pressure_impulse.component(k);
        EXPECT_NEAR(observed, impulse, 1e-10);
        body_error = std::max(body_error, std::abs(diagnostics.impulse_parts->body.component(k) - body.component(k)));
        momentum_error = std::max(momentum_error, std::abs(observed - impulse));
        if (stationary)
            EXPECT_NEAR(diagnostics.pressure_impulse.component(k), -body.component(k), 1e-10);
    }
    if (!label.empty())
    {
        metric(label + "body_impulse_defect_kg_m_s", body_error);
        metric(label + "momentum_defect_kg_m_s", momentum_error);
        if (stationary)
            metric(label + "pressure_impulse_defect_kg_m_s", (diagnostics.pressure_impulse + body).norm());
    }
}

inline void check_stocks(const Flow::CellState& endpoint, const Flow::CellState& initial)
{
    for (size_t p = 0; p < initial.phases.size(); ++p)
        EXPECT_NEAR(endpoint.phases[p].moles[0], initial.phases[p].moles[0], 2e-10);
    EXPECT_NEAR(endpoint.phase_numbers[1].value(), initial.phase_numbers[1].value(), 2e-10);
    if (!initial.qbmm_shape.empty())
    {
        EXPECT_NEAR(endpoint.qbmm_shape[1]->H2, initial.qbmm_shape[1]->H2, 2e-9);
        EXPECT_NEAR(endpoint.qbmm_shape[1]->H3, initial.qbmm_shape[1]->H3, 2e-9);
    }
}

inline void hydrostatic(Geometry kind, Velocity velocity, double gravity, std::optional<Limiter> limiter = {})
{
    SCOPED_TRACE(gravity);
    constexpr size_t layers = 4;
    Column fixed(kind, velocity, layers, gravity, false, false, 0., limiter);
    Column moving(kind, velocity, layers, gravity, true, false, 0., limiter);
    const auto reference = fixed.flow->state();
    const auto across = cross_section_cells(kind);
    EXPECT_EQ(moving.sum(moving.mesh->num_owned_cells()), layers * across);
    EXPECT_EQ(moving.sum(moving.mesh->num_owned_cells() == 0), 0.);
    const std::string sign = gravity < 0. ? "down_" : gravity > 0. ? "up_" : "zero_";
    double speed = 0., rate = 0., edge = 0.;
    for (int step = 0; step < 2; ++step)
    {
        auto a = fixed.controller->preview(dt);
        auto b = moving.controller->preview(dt);
        check_impulse(moving, b, gravity, true, sign + "step" + std::to_string(step + 1) + "_");
        check_impulse(fixed, a, gravity, true);
        fixed.controller->commit(a);
        moving.controller->commit(b);
        Flow::Traits::scalar_cell_type pressure(moving.mesh, 0., "column_pressure"),
            density(moving.mesh, 0., "column_density");
        for (size_t i = 0; i < reference.cells.size(); ++i)
        {
            const auto id = static_cast<LO>(i);
            const auto j = static_cast<size_t>(moving.mesh->cell_centroid(id).component(moving.axis) * layers);
            const auto& cell = moving.flow->state().cells[i];
            const auto& stationary = fixed.flow->state().cells[i];
            const double p = moving.flow->state().thermodynamic_pressure + cell.mechanical_pressure;
            const double rho = mass(cell) / moving.mesh->cell_volume(id);
            pressure.set_owned_value(id, p);
            density.set_owned_value(id, rho);
            EXPECT_NEAR(p, fixed.flow->state().thermodynamic_pressure + stationary.mechanical_pressure, 1e-6);
            EXPECT_NEAR(p, moving.pressure[j], 1e-6);
            EXPECT_NEAR(rho, layer_density(j), 1e-9);
            speed = std::max(speed, cell.velocity.norm());
            EXPECT_LT(cell.velocity.norm(), 1e-9);
            EXPECT_LT(stationary.velocity.norm(), 1e-9);
            check_stocks(cell, reference.cells[i]);
            for (size_t p = 0; p < cell.phases.size(); ++p)
                EXPECT_NEAR(cell.phases[p].internal_energy, reference.cells[i].phases[p].internal_energy, 1e-8);
        }
        pressure.sync_ghosts();
        density.sync_ghosts();
        for (const auto face : moving.flow->volume_flux().owned_face_ids())
        {
            const double absolute_rate = std::abs(moving.flow->volume_flux().owned_value(face));
            rate = std::max(rate, absolute_rate);
            EXPECT_LT(absolute_rate, 1e-10);
            EXPECT_LT(std::abs(fixed.flow->volume_flux().owned_value(face)), 1e-10);
            const auto& receipt = b.owned_phase_transport()[moving.flow->volume_flux().owned_row(face)];
            EXPECT_EQ(receipt.mesh_volume_flux, 0.);
            for (double relative : receipt.relative_phase_volume_flux)
                EXPECT_LT(std::abs(relative), 1e-10);
            if (moving.mesh->is_interior_face(face))
            {
                const auto l = moving.mesh->owner_cell(face), r = moving.mesh->neighbor_cell(face);
                const double dx =
                    (moving.mesh->cell_centroid(r) - moving.mesh->cell_centroid(l)).component(moving.axis);
                const double head = gravity * .5 * (density.local_value(l) + density.local_value(r)) * dx;
                const double defect = std::abs(pressure.local_value(r) - pressure.local_value(l) - head);
                edge = std::max(edge, defect);
                EXPECT_LT(defect, 1e-6);
            }
        }
        EXPECT_NEAR(b.diagnostics().flow.pressure_work, 0., 2e-9);
    }
    metric(sign + "max_speed_m_s", reduce(*moving.mesh, Teuchos::REDUCE_MAX, speed));
    metric(sign + "max_face_rate_m3_s", reduce(*moving.mesh, Teuchos::REDUCE_MAX, rate));
    metric(sign + "max_pressure_edge_defect_Pa", reduce(*moving.mesh, Teuchos::REDUCE_MAX, edge));
}

inline void piston(Geometry kind, Velocity velocity, std::optional<Limiter> limiter = {})
{
    constexpr double gravity = -.75;
    Column column(kind, velocity, 1, gravity, true, false, 0., limiter);
    const auto original = column.flow->state();
    const int ranks = column.mesh->owned_cell_map()->getComm()->getSize();
    const auto across = cross_section_cells(kind);
    EXPECT_EQ(column.sum(column.mesh->num_owned_cells()), across);
    EXPECT_EQ(column.sum(column.mesh->num_owned_cells() == 0), std::max(0, ranks - int(across)));
    // At fixed column base, V = base area * target. Each interval uses BE pressure work
    // C (T1-T0) = -p1 (V1-V0), ideal gas p1*(V1-Vliquid) = n_g*T1.
    const double area = cross_section_area(kind);
    const double liquid_v = (1. - gas_fraction(0)) * area, gas_n = gas_fraction(0) * area * rho_gas;
    const double heat_capacity = cv * (gas_n + rho_liquid * liquid_v);
    double previous_volume = area, previous_temperature = t0;
    for (const double height : {1. - 1. / 128., 1. + 1. / 128.})
    {
        SCOPED_TRACE(height);
        column.target = height;
        const double v1 = area * height, dv = v1 - previous_volume;
        const double t1 = heat_capacity * previous_temperature / (heat_capacity + gas_n * dv / (v1 - liquid_v));
        const double p1 = gas_n * t1 / (v1 - liquid_v), work = -p1 * dv;
        const auto trial = column.controller->preview(dt);
        const auto& d = trial.diagnostics().flow;
        const std::string step = dv < 0. ? "compression_" : "expansion_";
        check_impulse(column, trial, gravity, false, step);
        double local_delta_u = 0., local_sweep = 0.;
        for (size_t i = 0; i < trial.state().cells.size(); ++i)
        {
            const auto& cell = trial.state().cells[i];
            check_stocks(cell, original.cells[i]);
            EXPECT_NEAR(trial.transport_snapshot().old_control_volumes[i], previous_volume / across, 1e-13);
            EXPECT_NEAR(trial.transport_snapshot().new_control_volumes[i], v1 / across, 1e-13);
            EXPECT_NEAR(trial.state().thermodynamic_pressure + cell.mechanical_pressure, p1, 1e-6);
            for (size_t p = 0; p < cell.phases.size(); ++p)
            {
                EXPECT_NEAR(cell.phases[p].temperature, t1, 2e-7);
                local_delta_u +=
                    cell.phases[p].internal_energy - column.flow->state().cells[i].phases[p].internal_energy;
            }
        }
        for (const auto id : column.flow->volume_flux().owned_face_ids())
        {
            const auto& face = trial.owned_phase_transport()[column.flow->volume_flux().owned_row(id)];
            if (!column.mesh->is_geometry_exterior_face(id))
                continue;
            EXPECT_EQ(face.carrier_volume_flux, face.mesh_volume_flux);
            for (double q : face.relative_phase_volume_flux)
                EXPECT_EQ(q, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->derived_mass_flux, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->population_number_flux, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->shape_flux, R::QbmmShapeState{});
            EXPECT_EQ(face.qbmm_flux[1]->internal_energy_flux, 0.);
            const double normal = column.mesh->face_normal(id).component(column.axis);
            const double expected_sweep = normal > .5 ? dv / across : 0.;
            EXPECT_NEAR(face.mesh_volume_flux * dt, expected_sweep, 1e-13);
            local_sweep += face.mesh_volume_flux;
        }
        const double sweep = column.sum(local_sweep) * dt, delta_u = column.sum(local_delta_u);
        metric(step + "gcl_defect_m3", std::abs(sweep - dv));
        metric(step + "energy_defect_J", std::abs(delta_u - work));
        metric(step + "work_defect_J", std::abs(d.pressure_work - work));
        EXPECT_NEAR(sweep, dv, 1e-13);
        EXPECT_NEAR(delta_u, work, 2e-7);
        EXPECT_NEAR(d.pressure_work, work, 2e-7);
        EXPECT_LE(d.maximum_continuity_residual, column.flow->options().continuity_tolerance);
        EXPECT_LE(d.maximum_occupancy_error, column.flow->options().volume_relative_tolerance);
        column.controller->commit(trial);
        previous_volume = v1;
        previous_temperature = t1;
    }
}

} // namespace SimpleFluid::Testing::QbmmPlanarGravity
