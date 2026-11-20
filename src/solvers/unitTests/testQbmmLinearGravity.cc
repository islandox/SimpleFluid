/** Fixed-grid hydrostatic and stage-packet oracles for forced shared-carrier SSPRK2. */
#include "QbmmKineticTestSupport.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <numeric>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
using Coupler = R::ReactingMixtureCoupler;
using Vec = Flow::Vector;
using Eos = std::shared_ptr<const R::MixtureThermodynamics>;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

enum class Geometry
{
    Cartesian,
    Sector,
    Annulus,
    PolygonDisk,
    PolygonAnnulus
};
enum class Profile
{
    ConstantDensity,
    Stratified,
    Circulation
};
constexpr double pressure0 = 64., temperature0 = 8., liquid_density = 16., gas_mass_scale = .0625;
constexpr double hydro_dt = 1. / 64., replay_cfl = .008;

SP<Flow::Mesh> grid_for(Geometry kind, const ArrReal& z)
{
    const Flow::Mesh::DistributionOptions distribution{.allow_empty_partitions = true};
    if (kind == Geometry::Cartesian)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{ArrReal{0., 1., 2.}, {0., 1., 2.}, z}),
            distribution);

    constexpr unsigned sectors = 8;
    ArrReal angles;
    for (unsigned j = 0; j <= sectors; ++j)
        angles.push_back((kind == Geometry::Sector ? 1. : 2. * std::numbers::pi) * j / sectors);
    if (kind == Geometry::Sector || kind == Geometry::Annulus)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1., 1.5, 2.}, angles, z}), distribution);

    using Semi = Meshes::SemiStructuredXY_Z;
    Arr<Semi::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    const bool disk = kind == Geometry::PolygonDisk;
    if (disk)
        nodes.push_back({0., 0., 0.});
    const auto ring = [&](double radius)
    {
        for (unsigned j = 0; j < sectors; ++j)
            nodes.push_back({radius * std::cos(angles[j]), radius * std::sin(angles[j]), 0.});
    };
    if (!disk)
        ring(1.);
    ring(2.);
    for (unsigned j = 0; j < sectors; ++j)
    {
        const unsigned next = (j + 1) % sectors;
        if (disk)
            cells.push_back({0, 1 + j, 1 + next});
        else
            cells.push_back({j, sectors + j, sectors + next, next});
    }
    return std::make_shared<Flow::Mesh>(std::make_shared<Semi>(nodes, cells, z), distribution);
}

struct Column
{
    Profile profile;
    std::optional<Vec> gravity;
    bool head;
    ArrReal z;
    std::vector<double> pressure;

    Column(Profile profile, std::optional<Vec> gravity, bool head, ArrReal z)
        : profile(profile), gravity(gravity), head(head), z(std::move(z)), pressure(this->z.size() - 1, pressure0)
    {
        if (profile == Profile::ConstantDensity || !head)
            return;
        // At fixed T the actual ideal-gas density is p/T. Solve the trapezoidal edge
        // equation for the next unknown p; its density changes with the new pressure.
        for (size_t j = 1; j < pressure.size(); ++j)
        {
            const double half_gdz = .25 * gravity.value_or(Vec{}).z * (this->z[j + 1] - this->z[j - 1]);
            const double rho_left = density(j - 1, pressure[j - 1]);
            pressure[j] = (pressure[j - 1] + half_gdz * (rho_left + (1. - alpha(j)) * liquid_density)) /
                          (1. - half_gdz * alpha(j) / temperature0);
        }
    }

    size_t layer(double height) const
    {
        return static_cast<size_t>(std::upper_bound(z.begin(), z.end(), height) - z.begin() - 1);
    }
    double alpha(size_t j) const { return profile == Profile::Stratified ? (j + 1.) / 8. : .5; }
    double temperature(double p) const { return profile == Profile::ConstantDensity ? p / 8. : temperature0; }
    double density(size_t j, double p) const
    {
        return (1. - alpha(j)) * liquid_density + alpha(j) * p / temperature(p);
    }
    double absolute_pressure(Vec point) const
    {
        if (head && profile == Profile::ConstantDensity)
            return pressure0 + 12. * gravity.value_or(Vec{}).dot(point);
        return pressure.at(layer(point.z));
    }
};

struct SourceObservation
{
    int calls = 0;
    double pressure = 0;
    Vec velocity{};
};

struct Fixture
{
    SP<Flow::Mesh> grid;
    Column column;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<Coupler> controller;
    std::vector<SourceObservation> observations;

    Fixture(SP<Flow::Mesh> mesh, Column setup, double angular_velocity = 0.)
        : grid(std::move(mesh)), column(std::move(setup)), observations(grid->num_owned_cells())
    {
        const int ranks = grid->owned_cell_map()->getComm()->getSize();
        if (const char* expected = std::getenv("SIMPLEFLUID_Q8C_EXPECTED_RANKS"))
            EXPECT_EQ(ranks, std::stoi(expected));
        ::testing::Test::RecordProperty("mpi_ranks", ranks);
        auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "column material", 1., {{"M", 1}}, {R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed}}});
        const R::ThermodynamicBounds bounds{2., 16., 16., 128.};
        auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds,
            [](auto)
            {
                return R::LinearCompressibleMixtureLiquid{
                    temperature0, pressure0, liquid_density, 2., 0., 0., 16., 0., 0.};
            });
        auto gas = R::make_ideal_gas_mixture(
            catalog, {{2., temperature0, 16.}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
        Flow::Options opts;
        opts.minimum_pressure = bounds.minimum_pressure;
        opts.maximum_pressure = bounds.maximum_pressure;
        opts.linear.preconditioner = LinearPreconditioner::None;
        opts.linear.tolerance = 1e-12;
        opts.qbmm_model.family_transport = Model::FamilyTransport::RealizableLinearSSPRK2;
        const bool circulating = column.profile == Profile::Circulation;
        if (circulating)
            opts.qbmm_model.linear_cfl = replay_cfl;
        opts.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solvent"}};
        opts.qbmm_families = {nullptr};
        opts.qbmm_eos_law_identities = {
            R::ThermalLawIdentity{"linear gravity constant liquid", 1, catalog->fingerprint()}};
        std::vector<Eos> providers{liquid};
        R::ReactingMixtureOptions controls;
        controls.mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, liquid->energy_convention());
        controls.phases = {{R::SpeciesPhase::Liquid, {}, {}}};
        Flow::State initial;
        initial.qbmm_families = {std::nullopt};
        for (size_t a = 1; a <= (circulating ? 2u : 1u); ++a)
        {
            const std::string name = "column family " + std::to_string(a);
            opts.phase_roles.push_back({Flow::PhaseRole::DispersedGas, name});
            opts.qbmm_families.push_back(
                std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), name,
                    opts.phase_roles.back(), gas_mass_scale, std::vector<R::QbmmMassFraction>{{47, 1.}}));
            opts.qbmm_eos_law_identities.push_back(std::nullopt);
            initial.qbmm_families.push_back(Flow::qbmm_family_stamp(*opts.qbmm_families.back()));
            providers.push_back(gas);
            controls.phases.push_back({R::SpeciesPhase::Dispersed, name, {}});
        }
        flow = std::make_shared<Flow>(grid, providers, opts);
        double p_volume = 0., volume_sum = 0.;
        for (size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            const auto id = static_cast<LO>(row);
            const auto volume = grid->cell_volume(id);
            p_volume += volume * column.absolute_pressure(grid->cell_centroid(id));
            volume_sum += volume;
        }
        initial.thermodynamic_pressure = sum(p_volume) / sum(volume_sum);
        initial.cells.resize(grid->num_owned_cells());
        for (size_t row = 0; row < initial.cells.size(); ++row)
        {
            const auto id = static_cast<LO>(row);
            const auto x = grid->cell_centroid(id);
            const double p = column.absolute_pressure(x), t = column.temperature(p);
            const double volume = grid->cell_volume(id), alpha = column.alpha(column.layer(x.z));
            const double share = circulating ? .5 + .125 * (x.x - 1.) + .25 * (x.y - 1.) : 1.;
            auto& cell = initial.cells[row];
            cell.mechanical_pressure = p - initial.thermodynamic_pressure;
            if (circulating)
                cell.velocity = {1. - x.y, x.x - 1., 0.};
            else if (angular_velocity != 0.)
                cell.velocity = {-angular_velocity * x.y, angular_velocity * x.x, 0.};
            for (size_t a = 0; a < providers.size(); ++a)
            {
                const double fraction = a == 0 ? 1. - alpha : alpha * (a == 1 ? share : 1. - share);
                const auto point = providers[a]->evaluate(t, p, std::array{1.});
                const double mass = fraction * volume * point.density;
                cell.phases.push_back({providers[a], {mass}, p, mass * point.internal_energy, t});
                if (!a)
                {
                    cell.phase_numbers.push_back(std::nullopt);
                    cell.qbmm_shape.push_back(std::nullopt);
                    continue;
                }
                // Two strictly occupied support points xi=1,3. Their number and shape
                // change with the family stock, independently of its complementary family.
                const double coarse = mass / (4. * gas_mass_scale), fine = coarse;
                cell.phase_numbers.push_back(fine + coarse);
                cell.qbmm_shape.push_back(R::QbmmShapeState{fine + 9. * coarse, fine + 27. * coarse});
            }
        }
        flow->initialize(std::move(initial));
        controls.source_identity = "inert observed linear gravity source";
        controls.source = [this](const R::CoupledReactionContext& context, double)
        {
            auto& s = observations.at(context.owned_cell);
            ++s.calls;
            s.pressure = context.thermodynamic_trial.phases.front().absolute_pressure;
            s.velocity = context.thermodynamic_trial.velocity;
            return R::ReactionSourceInput{};
        };
        controls.body_acceleration = column.gravity;
        controller = std::make_unique<Coupler>(flow, std::move(controls));
    }

    double sum(double local) const { return reduce(*grid, Teuchos::REDUCE_SUM, local); }
    double maximum(double local) const { return reduce(*grid, Teuchos::REDUCE_MAX, local); }
    void metric(const std::string& name, double value) const
    {
        std::ostringstream s;
        s << std::setprecision(17) << value;
        ::testing::Test::RecordProperty(name, s.str());
        if (grid->owned_cell_map()->getComm()->getRank() == 0)
            std::cout << "Q8c " << name << '=' << s.str() << '\n';
    }

    void check_body(const Coupler::Trial& trial, double dt) const
    {
        double local = 0., energy_change = 0.;
        for (size_t row = 0; row < trial.state().cells.size(); ++row)
        {
            const auto& cell = trial.state().cells[row];
            for (size_t a = 0; a < cell.phases.size(); ++a)
            {
                local += cell.phases[a].moles.at(0); // The fixture species is 1 kg/mol in every compartment.
                energy_change += cell.phases[a].internal_energy - flow->state().cells[row].phases[a].internal_energy;
            }
        }
        const auto& d = trial.diagnostics().flow;
        const Vec expected = column.gravity.value_or(Vec{}) * (sum(local) * dt);
        EXPECT_EQ(d.body_acceleration_m_per_s2, column.gravity);
        EXPECT_EQ(d.impulse_parts.has_value(), column.gravity.has_value());
        for (size_t k = 0; k < 3; ++k)
        {
            EXPECT_NEAR(d.external_impulse.component(k), expected.component(k), 1e-11);
            if (d.impulse_parts)
            {
                EXPECT_NEAR(d.impulse_parts->body.component(k), expected.component(k), 1e-11);
                EXPECT_EQ(d.impulse_parts->other.component(k), 0.);
            }
        }
        EXPECT_LE(d.maximum_occupancy_error, Flow::Options{}.volume_relative_tolerance);
        EXPECT_LE(d.maximum_continuity_residual, Flow::Options{}.continuity_tolerance);
        EXPECT_NEAR(sum(energy_change), d.pressure_work, 2e-9 + d.internal_energy_roundoff_bound);
    }
};

void equilibrium(const SP<Flow::Mesh>& grid, const Column& column, const std::string& label)
{
    SCOPED_TRACE(label);
    Fixture f(grid, column);
    const auto initial = f.flow->state();
    const auto g = column.gravity.value_or(Vec{});
    double speed = 0., face_rate = 0., edge = 0.;
    for (int step = 0; step < 3; ++step)
    {
        auto trial = f.controller->preview(hydro_dt);
        f.check_body(trial, hydro_dt);
        f.controller->commit(trial);
        Flow::Traits::scalar_cell_type pressure(grid, 0., "observed_pressure"), density(grid, 0., "inventory_density");
        for (size_t row = 0; row < initial.cells.size(); ++row)
        {
            const auto id = static_cast<LO>(row);
            const auto x = grid->cell_centroid(id);
            const auto& cell = f.flow->state().cells[row];
            const double p = f.flow->state().thermodynamic_pressure + cell.mechanical_pressure;
            const double rho = (cell.phases[0].moles[0] + cell.phases[1].moles[0]) / grid->cell_volume(id);
            EXPECT_NEAR(p, column.absolute_pressure(x), 1e-6);
            EXPECT_NEAR(rho, column.density(column.layer(x.z), column.absolute_pressure(x)), 1e-9);
            EXPECT_NEAR(*cell.phase_numbers[1], *initial.cells[row].phase_numbers[1], 1e-9);
            EXPECT_NEAR(cell.qbmm_shape[1]->H2, initial.cells[row].qbmm_shape[1]->H2, 1e-8);
            EXPECT_NEAR(cell.qbmm_shape[1]->H3, initial.cells[row].qbmm_shape[1]->H3, 1e-8);
            speed = std::max(speed, cell.velocity.norm());
            pressure.set_owned_value(id, cell.mechanical_pressure);
            density.set_owned_value(id, rho);
        }
        pressure.sync_ghosts();
        density.sync_ghosts();
        for (const auto face : f.flow->volume_flux().owned_face_ids())
        {
            const auto rate = f.flow->volume_flux().owned_value(face);
            face_rate = std::max(face_rate, std::abs(rate));
            if (grid->is_geometry_exterior_face(face))
            {
                EXPECT_EQ(rate, 0.);
                continue;
            }
            const auto l = grid->owner_cell(face), r = grid->neighbor_cell(face);
            const auto dx = grid->cell_centroid(r) - grid->cell_centroid(l);
            const double head = .5 * (density.local_value(l) + density.local_value(r)) * g.dot(dx);
            edge = std::max(edge, std::abs(pressure.local_value(r) - pressure.local_value(l) - head));
        }
    }
    speed = f.maximum(speed);
    face_rate = f.maximum(face_rate);
    edge = f.maximum(edge);
    f.metric(label + "_maximum_speed_m_s", speed);
    f.metric(label + "_maximum_rate_m3_s", face_rate);
    f.metric(label + "_maximum_edge_defect_Pa", edge);
    EXPECT_LT(speed, 1e-9);
    EXPECT_LT(face_rate, 1e-10);
    EXPECT_LT(edge, 1e-6);
}

void qualify_column(Geometry kind)
{
    const ArrReal z{0., .25, .5, .75, 1.};
    const auto grid = grid_for(kind, z);
    if (kind != Geometry::Cartesian)
    {
        int seam = 0, wall = 0;
        for (size_t row = 0; row < grid->num_owned_faces(); ++row)
        {
            const auto face = static_cast<LO>(row);
            const auto x = grid->face_centroid(face);
            if (x.x > 0. && std::abs(x.y) < 1e-12 && std::abs(grid->face_normal(face).y) > .99)
            {
                if (grid->is_geometry_exterior_face(face))
                    ++wall;
                else
                    ++seam;
            }
        }
        seam = reduce(*grid, Teuchos::REDUCE_SUM, seam);
        wall = reduce(*grid, Teuchos::REDUCE_SUM, wall);
        // The +x radial half-plane is a wall in the sector, and an interior
        // neighbour connection in each full-turn native or polygonal mesh.
        EXPECT_EQ(wall, kind == Geometry::Sector ? 8 : 0);
        EXPECT_EQ(seam, kind == Geometry::Sector ? 0 : (kind == Geometry::Annulus ? 8 : 4));
        ::testing::Test::RecordProperty("positive_x_interior_seam_faces", seam);
        ::testing::Test::RecordProperty("positive_x_sector_wall_faces", wall);
    }
    equilibrium(grid, {Profile::Stratified, Vec{0., 0., -.981}, true, z}, "stratified");
    equilibrium(grid, {Profile::Stratified, Vec{0., 0., .981}, true, z}, "reversed");
    equilibrium(grid, {Profile::ConstantDensity, Vec{.5, -.25, -1.}, true, z}, "oblique");
    equilibrium(grid, {Profile::Stratified, std::nullopt, false, z}, "absent_gravity");
    for (double acceleration : {-.981, .981})
    {
        Fixture f(grid, {Profile::Stratified, Vec{0., 0., acceleration}, false, z});
        const auto trial = f.controller->preview(hydro_dt);
        f.check_body(trial, hydro_dt);
        double momentum = 0., head = 0., layer_volume = 0.;
        for (size_t row = 0; row < trial.state().cells.size(); ++row)
        {
            const auto id = static_cast<LO>(row);
            const auto& cell = trial.state().cells[row];
            const double volume = grid->cell_volume(id);
            const auto j = f.column.layer(grid->cell_centroid(id).z);
            momentum += (cell.phases[0].moles[0] + cell.phases[1].moles[0]) * cell.velocity.z;
            head += ((j == 3) - (j == 0)) * cell.mechanical_pressure * volume;
            layer_volume += (j == 0) * volume;
        }
        head = f.sum(head) / f.sum(layer_volume);
        momentum = f.sum(momentum);
        EXPECT_GT(acceleration * momentum, 1e-8);
        EXPECT_GT(acceleration * head, 1e-6);
        const std::string label = acceleration > 0. ? "flat_positive" : "flat_negative";
        f.metric(label + "_vertical_momentum_kg_m_s", momentum);
        f.metric(label + "_top_minus_bottom_Pa", head);
    }
}

TEST(QbmmLinearGravity, CartesianColumn)
{
    qualify_column(Geometry::Cartesian);
}
TEST(QbmmLinearGravity, WalledCylindricalSector)
{
    qualify_column(Geometry::Sector);
}
TEST(QbmmLinearGravity, AzimuthalSeamAnnulus)
{
    qualify_column(Geometry::Annulus);
}
TEST(QbmmLinearGravity, RotatingFamilyCrossesTheAzimuthalSeam)
{
    const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    ArrReal angles;
    for (unsigned j = 0; j <= 8; ++j)
        angles.push_back(j * std::numbers::pi / 4);
    const ArrReal heights{0., .5, 1.};
    const auto native = std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1., 2.}, angles, heights});
    const auto grid = std::make_shared<Flow::Mesh>(
        native, Flow::Mesh::DistributionOptions{.orthogonal_partitions = std::array<size_t, 3>{1, ranks, 1}});
    Fixture f(grid, {Profile::ConstantDensity, Vec{0., 0., -1.}, true, heights}, .25);
    const auto trial = f.controller->preview(hydro_dt);
    f.check_body(trial, hydro_dt);
    int seam = 0, interrank = 0;
    for (const auto face : f.flow->volume_flux().owned_face_ids())
    {
        const auto id = native->face_id(grid->face_geometry_global_id(face));
        if (id.orientation != Flow::Mesh::Cylindrical::THETA_FACE || id.j != 0)
            continue;
        const auto owner = grid->owner_cell(face), neighbor = grid->neighbor_cell(face);
        EXPECT_LT(grid->cell_centroid(owner).y, 0.);
        EXPECT_GT(grid->cell_centroid(neighbor).y, 0.);
        ++seam;
        interrank += !grid->is_owned_cell(neighbor) || !grid->is_owned_cell(owner);
        const auto& receipt = trial.owned_phase_transport()[f.flow->volume_flux().owned_row(face)];
        const auto& packet = receipt.qbmm_flux.at(1).value();
        // Axisymmetric velocity and inventory in each layer. The seam transports the same
        // two positive populations as the other angle faces. M/V = alpha*rho_g = 4.
        EXPECT_GT(receipt.carrier_volume_flux, .05);
        EXPECT_NEAR(packet.derived_mass_flux, 4. * receipt.carrier_volume_flux, 1e-10);
        EXPECT_NEAR(packet.species_molar_flux[0], packet.derived_mass_flux, 1e-12);
        EXPECT_NEAR(packet.population_number_flux, packet.derived_mass_flux / (2 * gas_mass_scale), 1e-11);
        EXPECT_NEAR(packet.kinetic->forward.integrated_mass, hydro_dt * packet.derived_mass_flux, 1e-12);
        EXPECT_EQ(packet.kinetic->backward.integrated_mass, 0.);
    }
    EXPECT_EQ(f.sum(seam), 2);
    EXPECT_EQ(f.sum(interrank), ranks == 1 ? 0 : 2);
    f.controller->commit(trial);
}

TEST(QbmmLinearGravity, OrthogonalPolygonDiskThroughAxis)
{
    qualify_column(Geometry::PolygonDisk);
}
TEST(QbmmLinearGravity, OrthogonalPolygonAnnulus)
{
    qualify_column(Geometry::PolygonAnnulus);
}

TEST(QbmmLinearGravity, SuppliedZeroAccelerationWithEmptyOwners)
{
    const auto grid = mesh({0., 1.});
    record_ownership(*grid, 1);
    equilibrium(grid, {Profile::ConstantDensity, Vec{}, true, {0., 1.}}, "supplied_zero");
}

// On this eight-cell Cartesian grid there is one neighbour per coordinate direction.
// Its centered linear reconstruction at a common face is the arithmetic mean of the
// two stocks (unit volumes). The strictly positive two-node proportions and common
// specific U ensure both this trace and the retained mixture remain admissible.
using EightCells = std::array<double, 8>;
struct Edge
{
    size_t low, high;
    double rate = 0.;
    std::array<double, 2> mass_packets{}, volume_derivative{};
};

template<size_t N> std::array<double, N> gather(const Flow::Mesh& grid, const std::array<double, N>& local)
{
    std::array<double, N> result{};
    Teuchos::reduceAll(*grid.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, int(N), local.data(), result.data());
    return result;
}

TEST(QbmmLinearGravity, TwoStagePacketsAndSingleImpulseDuringBalancedCirculation)
{
    const ArrReal edges{0., 1., 2.};
    const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    const auto grid =
        std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{edges, edges, edges}),
            Flow::Mesh::DistributionOptions{
                .orthogonal_partitions = std::array<size_t, 3>{ranks == 1 ? 1u : 2u, ranks == 4 ? 2u : 1u, 1u}});
    record_ownership(*grid, 8);
    Fixture f(grid, {Profile::Circulation, Vec{0., 0., -1.}, true, edges});
    constexpr double dt = .0625;
    const auto trial = f.controller->preview(dt);
    f.check_body(trial, dt);
    const auto& d = trial.diagnostics().flow;
    EXPECT_EQ(d.source_evaluations, d.correctors);
    EXPECT_GE(d.source_evaluations, 2);
    EXPECT_LE(d.maximum_linear_stage_residual, 1.);
    EXPECT_LE(d.maximum_linear_step_residual, 1.);
    EXPECT_EQ(d.linear_stages, 2u * d.linear_substeps);
    const auto report = f.controller->qbmm_report(trial);
    ASSERT_EQ(f.sum(bool(report.interval)), ranks);
    EXPECT_EQ(report.interval->source_stage.corrector + 1, d.source_evaluations);

    const auto index = [&](LO cell)
    {
        const auto x = grid->cell_centroid(cell);
        return size_t(x.x) + 2u * size_t(x.y) + 4u * size_t(x.z);
    };
    EightCells local_p{};
    std::array<EightCells, 3> velocity{};
    std::array<EightCells, 2> stocks{};
    for (size_t row = 0; row < grid->num_owned_cells(); ++row)
    {
        const size_t i = index(static_cast<LO>(row));
        EXPECT_EQ(f.observations[row].calls, d.source_evaluations);
        local_p[i] = f.observations[row].pressure;
        for (size_t k = 0; k < 3; ++k)
            velocity[k][i] = f.observations[row].velocity.component(k);
        for (size_t a = 0; a < 2; ++a)
            stocks[a][i] = f.flow->state().cells[row].phases[a + 1].moles[0];
    }
    const auto pressure = gather(*grid, local_p);
    for (auto& component : velocity)
        component = gather(*grid, component);
    for (auto& stock : stocks)
        stock = gather(*grid, stock);

    std::vector<Edge> faces;
    for (size_t i = 0; i < 8; ++i)
        for (size_t stride : {1u, 2u, 4u})
            if (!(i & stride))
                faces.push_back({i, i + stride});
    auto find_face = [&](LO face)
    {
        const auto l = index(grid->owner_cell(face)), r = index(grid->neighbor_cell(face));
        return std::ranges::find_if(
            faces, [&](const Edge& e) { return e.low == std::min(l, r) && e.high == std::max(l, r); });
    };
    std::array<double, 12> local_rates{};
    double partition_transfers = 0.;
    for (const auto face : f.flow->volume_flux().owned_face_ids())
    {
        if (grid->is_geometry_exterior_face(face))
            continue;
        const auto e = find_face(face);
        const auto rate = trial.owned_phase_transport()[f.flow->volume_flux().owned_row(face)].carrier_volume_flux;
        local_rates.at(e - faces.begin()) = e->low == index(grid->owner_cell(face)) ? rate : -rate;
        if (!grid->is_owned_cell(grid->neighbor_cell(face)) && std::abs(rate) > .1)
            ++partition_transfers;
    }
    const auto rates = gather(*grid, local_rates);
    EightCells outgoing{};
    for (size_t k = 0; k < faces.size(); ++k)
    {
        auto& e = faces[k];
        e.rate = rates[k];
        outgoing[e.rate > 0. ? e.low : e.high] += std::abs(e.rate);
        if (e.high - e.low == 4)
            EXPECT_NEAR(e.rate, 0., 1e-9);
        else
            EXPECT_GT(std::abs(e.rate), .1);
    }
    const double shared = f.sum(partition_transfers);
    if (ranks > 1)
        EXPECT_GT(shared, 0.);
    f.metric("active_partition_faces", shared);
    // Compute the common CFL-limited intervals from the frozen face rates.
    const double limit = replay_cfl / *std::ranges::max_element(outgoing);
    ASSERT_GT(limit, 0.);
    ASSERT_TRUE(std::isfinite(limit));
    const int substeps = static_cast<int>(std::ceil(dt / limit));
    ASSERT_EQ(substeps, d.linear_substeps);
    ASSERT_GT(substeps, 1);
    double stage_separation = 0.;
    for (int sub = 0; sub < substeps; ++sub)
    {
        const double h = std::min(limit, dt - sub * limit);
        for (size_t a = 0; a < 2; ++a)
        {
            const auto start = stocks[a];
            std::array<double, 12> first_trace{};
            for (int stage = 0; stage < 2; ++stage)
            {
                const auto input = stocks[a];
                for (size_t k = 0; k < faces.size(); ++k)
                {
                    auto& e = faces[k];
                    const double trace = .5 * (input[e.low] + input[e.high]);
                    const double mass = h * e.rate * trace;
                    stocks[a][e.low] -= mass;
                    stocks[a][e.high] += mass;
                    e.mass_packets[a] += .5 * mass;
                    // For this gas u=2T at R=1. Both stages advect U=16M so T=8.
                    const double specific_volume =
                        e.rate > 0.
                            ? 8. / pressure[e.low]
                            : (e.rate < 0. ? 8. / pressure[e.high] : 4. / pressure[e.low] + 4. / pressure[e.high]);
                    e.volume_derivative[a] += .5 * h * trace * specific_volume;
                    if (stage == 0)
                        first_trace[k] = trace;
                    else
                        stage_separation = std::max(stage_separation, std::abs(e.rate * (trace - first_trace[k])));
                }
            }
            for (size_t i = 0; i < start.size(); ++i)
                stocks[a][i] = .5 * (start[i] + stocks[a][i]);
        }
    }
    EXPECT_GT(stage_separation, 1e-4);
    double mass_defect = 0.;
    for (const auto face : f.flow->volume_flux().owned_face_ids())
    {
        const auto& receipt = trial.owned_phase_transport()[f.flow->volume_flux().owned_row(face)];
        const bool wall = grid->is_geometry_exterior_face(face);
        for (size_t a = 0; a < 2; ++a)
        {
            const auto& packet = receipt.qbmm_flux.at(a + 1).value();
            double expected_rate = 0., theta = 0.;
            size_t donor = 0;
            if (!wall)
            {
                const auto& e = *find_face(face);
                const double orientation = e.low == index(grid->owner_cell(face)) ? 1. : -1.;
                expected_rate = orientation * e.mass_packets[a] / dt;
                theta = e.volume_derivative[a] / dt;
                donor = e.rate > 0. ? e.low : e.high;
            }
            EXPECT_NEAR(packet.derived_mass_flux, expected_rate, 1e-11);
            EXPECT_NEAR(packet.species_molar_flux.at(0), expected_rate, 1e-11);
            EXPECT_NEAR(packet.internal_energy_flux, 16. * expected_rate, 2e-10);
            EXPECT_NEAR(packet.material_volume_flux, receipt.carrier_volume_flux * theta, 1e-11);
            EXPECT_NEAR(packet.kinetic->theta, theta, 1e-11);
            const double weight = expected_rate / (4. * gas_mass_scale);
            EXPECT_NEAR(packet.population_number_flux, 2. * weight, 1e-10);
            EXPECT_NEAR(packet.shape_flux.H2, 10. * weight, 1e-9);
            EXPECT_NEAR(packet.shape_flux.H3, 28. * weight, 1e-9);
            const auto& directional = expected_rate >= 0. ? packet.kinetic->forward : packet.kinetic->backward;
            EXPECT_NEAR(directional.integrated_mass, expected_rate * dt, 1e-12);
            for (size_t k = 0; k < 3; ++k)
                EXPECT_NEAR(directional.carrier_momentum_flux.component(k), expected_rate * velocity[k][donor], 1e-11);
            mass_defect = std::max(mass_defect, std::abs(packet.derived_mass_flux - expected_rate));
        }
    }
    for (size_t row = 0; row < grid->num_owned_cells(); ++row)
    {
        const auto i = index(static_cast<LO>(row));
        const auto& cell = trial.state().cells[row];
        const auto& transported = trial.transport_snapshot().cells[row];
        double total_mass = cell.phases[0].moles[0];
        for (size_t a = 0; a < 2; ++a)
        {
            EXPECT_NEAR(transported.phases[a + 1].moles[0], stocks[a][i], 1e-11);
            EXPECT_NEAR(transported.phases[a + 1].internal_energy, 16. * stocks[a][i], 2e-10);
            EXPECT_NEAR(cell.phases[a + 1].moles[0], stocks[a][i], 1e-11);
            const double weight = stocks[a][i] / (4. * gas_mass_scale);
            EXPECT_NEAR(*cell.phase_numbers[a + 1], 2. * weight, 1e-10);
            EXPECT_NEAR(cell.qbmm_shape[a + 1]->H2, 10. * weight, 1e-9);
            EXPECT_NEAR(cell.qbmm_shape[a + 1]->H3, 28. * weight, 1e-9);
            total_mass += stocks[a][i];
        }
        EXPECT_NEAR(cell.velocity.z, 0., 1e-9);
        const auto& gross = report.interval->local_gross_events.at(row);
        EXPECT_EQ(std::count_if(gross.begin(), gross.end(), [](double amount) { return amount != 0.; }), 1);
        EXPECT_NEAR(std::accumulate(gross.begin(), gross.end(), 0.), total_mass * dt, 1e-11);
    }
    f.metric("linear_substeps", d.linear_substeps);
    f.metric("linear_stages", d.linear_stages);
    f.metric("source_evaluations", d.source_evaluations);
    f.metric("maximum_stage_rate_separation_kg_s", stage_separation);
    f.metric("maximum_mass_rate_replay_error_kg_s", f.maximum(mass_defect));
    f.metric("maximum_stage_residual_normalized", d.maximum_linear_stage_residual);
    f.metric("maximum_step_residual_normalized", d.maximum_linear_step_residual);
    f.metric("body_impulse_z_kg_m_s", d.impulse_parts.value().body.z);
    f.controller->commit(trial);
    EXPECT_EQ(f.flow->state().time, dt);
    for (const auto& observation : f.observations)
        EXPECT_EQ(observation.calls, d.source_evaluations);
}
} // namespace
