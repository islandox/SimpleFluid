/** Exact controls for the same synchronized kinetic operator used by the native corrector. */
#include "QbmmCylindricalTestMesh.hh"
#include "QbmmKineticTestSupport.hh"
#include "solvers/details/LowMachQbmmKineticTransport.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <iostream>
#include <limits>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
namespace Kernel = R::qbmm_kinetic_detail;
using Range = R::QbmmRangeVariable;
using Gate = R::QbmmApplicabilityGate;
using Status = R::QbmmRangeStatus;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Operator
{
    SP<Flow::Mesh> grid;
    std::shared_ptr<const R::SpeciesCatalog> species = catalog();
    R::ThermodynamicBounds bounds;
    std::shared_ptr<const R::MixtureThermodynamics> gas;
    Flow::State accepted;
    Flow::Traits::scalar_face_type carrier;
    std::vector<std::vector<double>> occupied_volume;

    explicit Operator(SP<Flow::Mesh> mesh, double pressure = 4., double e_at_one = 1.)
        : grid(std::move(mesh)), bounds{.125, 8., pressure / 2., pressure * 1.5},
          gas(R::make_ideal_gas_mixture(
              species, {{2., 1., e_at_one}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.)),
          carrier(grid, 0., "kinetic_carrier_rate"), occupied_volume(grid->num_owned_cells(), std::vector<double>(1))
    {
        accepted.thermodynamic_pressure = pressure;
        accepted.cells.resize(grid->num_owned_cells());
    }

    void fill(const Flow::Options& opts, const std::function<R::QbmmMomentBlock(Flow::Vector)>& state,
        const std::function<double(Flow::Vector)>& temperature, double e_at_one = 1.)
    {
        accepted.qbmm_families = {Flow::qbmm_family_stamp(*opts.qbmm_families[0])};
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            const auto position = grid->cell_centroid(static_cast<LO>(c));
            const auto h = state(position);
            const double t = temperature(position), p = accepted.thermodynamic_pressure;
            set_stock(accepted.cells[c], gas, h, p, t, e_at_one + 2. * (t - 1.));
            occupied_volume[c][0] = h[1] / (p / t);
        }
    }

    Kernel::TransportResult advance(const Flow::Options& opts, Flow::State& candidate, double dt)
    {
        candidate = accepted;
        carrier.sync_ghosts();
        return Kernel::transport(grid, opts, accepted, accepted, candidate, carrier, occupied_volume, dt, 0);
    }

    template<class Check> void interior_flux(const Kernel::TransportResult& result, Check check)
    {
        int local_faces = 0, local_partition_faces = 0;
        for (const auto face : carrier.owned_face_ids())
        {
            const auto& packet = result.owned_flux.at(0).at(carrier.owned_row(face));
            EXPECT_TRUE(packet.kinetic.has_value());
            if (grid->is_interior_face(face))
            {
                ++local_faces;
                if (!grid->is_owned_cell(grid->owner_cell(face)) || !grid->is_owned_cell(grid->neighbor_cell(face)))
                    ++local_partition_faces;
                if (packet.kinetic)
                    check(face, packet, grid->face_area_vector(face));
            }
            else
            {
                EXPECT_EQ(packet.population_number_flux, 0.);
                EXPECT_EQ(packet.derived_mass_flux, 0.);
                EXPECT_EQ(packet.material_volume_flux, 0.);
            }
        }
        const int partition_faces = reduce(*grid, Teuchos::REDUCE_SUM, local_partition_faces);
        EXPECT_GT(reduce(*grid, Teuchos::REDUCE_SUM, local_faces), 0);
        if (grid->owned_cell_map()->getComm()->getSize() > 1)
            EXPECT_GT(partition_faces, 0);
        ::testing::Test::RecordProperty("partition_faces", partition_faces);
    }
};

std::array<double, 3> directional_slip(const R::QbmmNodeSlipInput& in)
{
    return {in.location.value().centroid[0] < 1. ? .25 : -.25, 0., 0.};
}

void check_moments(const R::QbmmMomentBlock& actual, const R::QbmmMomentBlock& expected, double& max_error)
{
    for (size_t k = 0; k < actual.size(); ++k)
    {
        SCOPED_TRACE(k);
        EXPECT_NEAR(actual[k], expected[k], 2e-13);
        max_error = std::max(max_error, std::abs(actual[k] - expected[k]));
    }
}

TEST(LowMachQbmmKineticTransportTest, TwoSizeColumnHasAnalyticUpwardMomentRates)
{
    // Distinguish the face area (2 m²) from the donor volume (0.5 m³).
    const auto grid = std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{{{0., 2.}, {0., 1.}, {0., .25, .5}}}),
        Flow::Mesh::DistributionOptions{.allow_empty_partitions = true,
            .orthogonal_partitions =
                std::array<size_t, 3>{1, 1, static_cast<size_t>(Tpetra::getDefaultComm()->getSize())}});
    Operator op(grid, 16.);
    record_ownership(*grid, 2);
    constexpr double s = .125, dt = .25, area_over_volume = 4.;
    const auto opts = options(
        op.species, [](const R::QbmmNodeSlipInput& atom)
        { return std::array{0., 0., atom.mass_kg_per_bubble < 2. ? s : 2. * s}; }, op.bounds);
    op.fill(
        opts, [](auto x) { return x.z < .25 ? R::QbmmMomentBlock{3., 5., 11., 29.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; });
    Flow::State endpoint;
    const auto result = op.advance(opts, endpoint, dt);
    EXPECT_EQ(result.substeps, 1);
    constexpr R::QbmmMomentBlock coefficients{4., 8., 20., 56.};
    double error = 0.;
    op.interior_flux(result,
        [&](auto face, const auto& p, auto a)
        {
            EXPECT_EQ(op.carrier.owned_value(face), 0.);
            const double sign = std::copysign(1., a.z);
            const R::QbmmMomentBlock rates{
                p.population_number_flux, p.derived_mass_flux, p.shape_flux.H2, p.shape_flux.H3};
            for (size_t k = 0; k < rates.size(); ++k)
            {
                const double expected = coefficients[k] * s * area_over_volume;
                EXPECT_NEAR(sign * rates[k], expected, 2e-13);
                error = std::max(error, std::abs(sign * rates[k] - expected));
            }
        });
    record_maximum(*grid, "maximum_analytic_moment_rate_error", error);
}

TEST(QbmmCylindricalKineticMetric, AnalyticMomentRatesAndSideMetrics)
{
    namespace C = SimpleFluid::Testing::QbmmCylinder;
    constexpr double dz = .25, speed = .125;
    for (const auto kind : C::geometries)
    {
        const std::string prefix = C::names[static_cast<size_t>(kind)];
        SCOPED_TRACE(prefix);
        const auto grid = C::mesh(kind, 2, {0., dz, 2. * dz});
        const double angle = C::span(kind) / C::wedges(kind);
        const double shape_factor = C::native(kind) ? angle : std::sin(angle);
        const double r0 = C::inner_radius(kind), middle = .5 * (r0 + 2.);
        // Sector centroid lies inward of the circular midpoint. Polygon centroids
        // have an additional chord contraction. Classify rings by their known bisector.
        const double centroid_scale = C::native(kind) ? std::sin(.5 * angle) / (.5 * angle) : std::cos(.5 * angle);
        // Centroid radius follows from the first moment of an annular wedge.
        // A polygonal strip is the difference of two triangles on the same bisector.
        const auto centroid_radius = [&](double low, double high)
        { return centroid_scale * (2. / 3.) * (high * high + high * low + low * low) / (high + low); };
        const auto cap_area = [&](Flow::Vector center)
        {
            const bool inner_ring = std::hypot(center.x, center.y) < middle * centroid_scale;
            const double low = inner_ring ? r0 : middle, high = inner_ring ? middle : 2.;
            return .5 * shape_factor * (high * high - low * low);
        };
        Operator op(grid, 128.);
        const auto opts = options(
            op.species, [](const auto& node)
            { return std::array{0., 0., node.mass_kg_per_bubble < 2. ? speed : 2. * speed}; }, op.bounds);
        op.fill(
            opts, [](auto x) { return x.z < dz ? R::QbmmMomentBlock{3., 5., 11., 29.} : R::QbmmMomentBlock{}; },
            [](auto) { return 1.; });
        for (size_t c = 0; c < grid->num_owned_cells(); ++c)
        {
            const auto id = static_cast<LO>(c);
            EXPECT_NEAR(grid->cell_volume(id), dz * cap_area(grid->cell_centroid(id)), 2e-14);
        }
        Flow::State endpoint;
        const auto result = op.advance(opts, endpoint, .125);
        EXPECT_EQ(result.substeps, 1);
        int axial = 0, radial = 0, angular = 0;
        double max_error = 0., max_distance_error = 0.;
        op.interior_flux(result,
            [&](auto face, const auto& packet, auto a)
            {
                const auto center = grid->face_centroid(face);
                const R::QbmmMomentBlock rates{packet.population_number_flux, packet.derived_mass_flux,
                    packet.shape_flux.H2, packet.shape_flux.H3};
                constexpr R::QbmmMomentBlock coefficients{4., 8., 20., 56.};
                double signed_area = 0., volume = 1., distance = dz;
                const auto separation =
                    grid->cell_centroid(grid->neighbor_cell(face)) - grid->cell_centroid(grid->owner_cell(face));
                EXPECT_GT(a.dot(separation), 0.);
                if (center.z == dz)
                {
                    EXPECT_EQ(a.x, 0.);
                    EXPECT_EQ(a.y, 0.);
                    ++axial;
                    const double area = cap_area(center);
                    EXPECT_NEAR(std::abs(a.z), area, 2e-14);
                    signed_area = std::copysign(area, a.z);
                    volume = area * dz;
                }
                else
                {
                    EXPECT_EQ(a.z, 0.);
                    const double radius = std::hypot(center.x, center.y);
                    const double alignment = (a.x * center.x + a.y * center.y) / (radius * grid->face_area(face));
                    if (std::abs(alignment) > .5)
                    {
                        ++radial;
                        distance = centroid_radius(middle, 2.) - centroid_radius(r0, middle);
                        const double width = C::native(kind) ? angle : 2. * std::sin(.5 * angle);
                        EXPECT_NEAR(grid->face_area(face), middle * width * dz, 2e-14);
                        EXPECT_NEAR(std::abs(alignment), 1., 2e-14);
                    }
                    else
                    {
                        ++angular;
                        const double rc = radius < middle ? centroid_radius(r0, middle) : centroid_radius(middle, 2.);
                        distance = 2. * rc * std::sin(.5 * angle);
                        EXPECT_NEAR(alignment, 0., 2e-14);
                        EXPECT_NEAR(grid->face_area(face), (2. - middle) * dz, 2e-14);
                    }
                }
                const double measured = grid->face_cell_center_distance(face);
                EXPECT_NEAR(measured, distance, 2e-14);
                max_distance_error = std::max(max_distance_error, std::abs(measured - distance));
                for (size_t k = 0; k < rates.size(); ++k)
                {
                    const double expected = coefficients[k] * speed * signed_area / volume;
                    EXPECT_NEAR(rates[k], expected, 2e-13);
                    max_error = std::max(max_error, std::abs(rates[k] - expected));
                }
            });
        EXPECT_EQ(reduce(*grid, Teuchos::REDUCE_SUM, axial), 2 * C::wedges(kind));
        EXPECT_GT(reduce(*grid, Teuchos::REDUCE_SUM, angular), 0);
        EXPECT_EQ(reduce(*grid, Teuchos::REDUCE_SUM, radial), 2 * C::wedges(kind));
        const auto metric_error = reduce(*grid, Teuchos::REDUCE_MAX, max_distance_error);
        std::ostringstream distance_value;
        distance_value << std::setprecision(17) << metric_error;
        ::testing::Test::RecordProperty(prefix + "_maximum_pressure_distance_error_m", distance_value.str());
        if (grid->owned_cell_map()->getComm()->getRank() == 0)
            std::cout << "Q8d " << prefix << "_maximum_pressure_distance_error_m=" << distance_value.str() << '\n';
        const auto error = reduce(*grid, Teuchos::REDUCE_MAX, max_error);
        const auto key = prefix + "_maximum_moment_rate_error";
        std::ostringstream formatted;
        formatted << std::setprecision(17) << error;
        ::testing::Test::RecordProperty(key, formatted.str());
        if (grid->owned_cell_map()->getComm()->getRank() == 0)
            std::cout << "Q8d " << key << '=' << formatted.str() << '\n';
    }
}

TEST(LowMachQbmmKineticTransportTest, CylindricalUnresolvedOffsetRejectsOnlyWhenMaterialCrosses)
{
    // A single radial shell with two equal angular cells is orthogonal. Their
    // only interior displacement is tangential, while the face centroid has a radial offset.
    const auto grid = std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1., 2.}, {0., .25, .5}, {0., 1.}}),
        Flow::Mesh::DistributionOptions{.allow_empty_partitions = true});
    Operator op(grid);
    const auto opts = options(op.species, [](const auto&) { return std::array{0., 0., .125}; }, op.bounds);
    op.fill(opts, [](auto) { return R::QbmmMomentBlock{2., 2., 2., 2.}; }, [](auto) { return 1.; });
    Flow::State endpoint;
    EXPECT_NO_THROW((void) op.advance(opts, endpoint, .01));
    for (const double rate : {.01, -.01})
    {
        for (const auto face : op.carrier.owned_face_ids())
            if (grid->is_interior_face(face))
                op.carrier.set_owned_value(face, rate);
        EXPECT_THROW((void) op.advance(opts, endpoint, .01), std::exception);
    }
}

TEST(LowMachQbmmKineticTransportTest, CancellingKineticFamiliesRetainExactSpeciesWeightedMassRemainder)
{
    constexpr double width = 0x1p25, pressure = 8.;
    const auto grid = mesh({0., width, 2. * width});
    record_ownership(*grid, 2);
    constexpr std::array counts{0x1p27 + 1., 0x1p27}, molar_mass{0x1p27 + 1., 0x1p27 + 2.};
    const auto species = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {47, "forward species", molar_mass[0], {{"G", 1}}, {R::SpeciesPhase::Dispersed}},
        {48, "backward species", molar_mass[1], {{"G", 1}}, {R::SpeciesPhase::Dispersed}}});
    const R::ThermodynamicBounds bounds{.5, 2., 4., 12.};
    const auto gas = R::make_ideal_gas_mixture(species,
        {{2. * molar_mass[0], 1., molar_mass[0]}, {2. * molar_mass[1], 1., molar_mass[1]}}, bounds,
        R::EnergyConvention::IncludesFormationEnergy, 1.);
    Flow::Options opts;
    opts.qbmm_model.velocity = Model::Velocity::NodeDependentSlip;
    opts.qbmm_model.family_transport = Model::FamilyTransport::KineticEuler;
    Flow::State accepted;
    accepted.thermodynamic_pressure = pressure;
    for (size_t a = 0; a < counts.size(); ++a)
    {
        const std::string identity = a == 0 ? "rightgoing bubbles" : "leftgoing bubbles";
        const R::QbmmLawMetadata meta{identity, 1, R::QbmmLawKind::NodeSlip, "physical atom transport", "m/s"};
        opts.phase_roles.push_back({Flow::PhaseRole::DispersedGas, identity});
        R::QbmmClosureVersions versions;
        versions.laws = {meta};
        const auto family = std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), identity,
            opts.phase_roles.back(), molar_mass[a],
            std::vector<R::QbmmMassFraction>{{47, double(a == 0)}, {48, double(a == 1)}}, std::move(versions));
        opts.qbmm_families.push_back(family);
        opts.qbmm_node_slip_laws.push_back(std::make_shared<const R::QbmmNodeSlipLaw>(
            R::QbmmNodeSlipLaw{meta, [a](const auto&) { return std::array{a == 0 ? 1. : -1., 0., 0.}; }}));
        accepted.qbmm_families.push_back(Flow::qbmm_family_stamp(*family));
    }
    accepted.cells.resize(grid->num_owned_cells());
    std::vector volumes(grid->num_owned_cells(), std::vector<double>(counts.size()));
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        auto& cell = accepted.cells[c];
        const size_t donor = grid->cell_centroid(static_cast<LO>(c)).x < width ? 0 : 1;
        for (size_t a = 0; a < counts.size(); ++a)
        {
            const double count = a == donor ? counts[a] : 0.;
            std::vector<double> moles(counts.size());
            moles[a] = count;
            cell.phases.push_back({gas, std::move(moles), pressure, count * molar_mass[a], count == 0. ? 0. : 1.});
            cell.phase_numbers.push_back(count);
            cell.qbmm_shape.push_back(R::QbmmShapeState{count, count});
            volumes[c][a] = count / pressure;
        }
    }
    auto candidate = accepted;
    Flow::Traits::scalar_face_type carrier(grid, 0., "zero_carrier");
    carrier.sync_ghosts();
    const auto result = Kernel::transport(grid, opts, accepted, accepted, candidate, carrier, volumes, width, 0);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_EQ(result.maximum_cfl, 1.);
    int local_remote = 0;
    for (const auto face : carrier.owned_face_ids())
        if (grid->is_interior_face(face))
        {
            if (!grid->is_owned_cell(grid->owner_cell(face)) || !grid->is_owned_cell(grid->neighbor_cell(face)))
                ++local_remote;
            const size_t f = carrier.owned_row(face);
            const auto& integral = result.integrated_faces[f];
            const double remainder = integral.net_mass;
            const double sign = grid->face_area_vector(face).x;
            // A=(2^27+1)^2 and B=2^27(2^27+2) round to the same binary64 mass, while A-B=1 exactly.
            EXPECT_EQ(result.owned_flux[0][f].derived_mass_flux + result.owned_flux[1][f].derived_mass_flux, 0.);
            EXPECT_EQ(sign * remainder, 1.);
            EXPECT_EQ(sign * remainder / width, 1. / width);
        }
    const int remote = reduce(*grid, Teuchos::REDUCE_SUM, local_remote);
    EXPECT_EQ(remote, int(grid->owned_cell_map()->getComm()->getSize() > 1));
    RecordProperty("partition_faces", remote);
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const size_t receiver = grid->cell_centroid(static_cast<LO>(c)).x < width ? 1 : 0;
        EXPECT_EQ(candidate.cells[c].phases[receiver].moles[receiver], counts[receiver]);
        EXPECT_EQ(candidate.cells[c].phases[1 - receiver].moles[1 - receiver], 0.);
    }
}

TEST(LowMachQbmmKineticTransportTest, ReservationRoundsDownWhenTheSweptProductTiesAtTheCap)
{
    constexpr double rate = 1. + 0x1p-32, width = 1. + 0x1p-31;
    Operator op(mesh({0., width, 2. * width}), 8.);
    const auto opts = options(op.species, [](const auto&) { return std::array{0., 0., 0.}; }, op.bounds, 1., 2);
    op.fill(
        opts, [](auto x) { return x.x < width ? R::QbmmMomentBlock{1., 1., 1., 1.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; });
    for (const auto face : op.carrier.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.carrier.set_owned_value(face, rate * op.grid->face_area_vector(face).x);

    // The exact sweep for dt=rate is width+2^-64 even though the product can round to width.
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, rate);
    const double first = std::nextafter(rate, 0.);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{first, rate - first}));
    EXPECT_LE(result.maximum_cfl, 1.);
}

TEST(LowMachQbmmKineticTransportTest, SubnormalReservationRateChoosesTheLargestSafeDuration)
{
    // The donor stock is deliberately large so even the smallest swept fraction has
    // a material-volume packet that passes the transport representation gate.
    Operator op(mesh({0., 4., 8.}), 8.);
    record_ownership(*op.grid, 2);
    constexpr double floor = std::numeric_limits<double>::denorm_min(), count = 0x1p100;
    const auto opts = options(
        op.species, [](const R::QbmmNodeSlipInput& in)
        { return std::array{in.location.value().centroid[0] < 4. ? 5. * floor : -5. * floor, 0., 0.}; }, op.bounds,
        floor, 2);
    op.fill(opts, [](auto) { return R::QbmmMomentBlock{count, count, count, count}; }, [](auto) { return 1.; });
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, 1.);
    const double safe_h = std::nextafter(.8, 0.);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{safe_h, 1. - safe_h}));
    EXPECT_EQ(result.maximum_cfl, floor);
    EXPECT_LE(std::fma(5., safe_h, -4.), 0.);
    EXPECT_GT(std::fma(5., std::nextafter(safe_h, 1.), -4.), 0.);
    op.interior_flux(result,
        [&](auto, const auto& p, const auto&)
        {
            EXPECT_EQ(p.population_number_flux, 0.);
            EXPECT_DOUBLE_EQ(p.kinetic->forward.integrated_count, 5. * 0x1p-976);
            EXPECT_DOUBLE_EQ(p.kinetic->backward.integrated_count, -5. * 0x1p-976);
        });
    for (const auto& cell : candidate.cells)
        EXPECT_EQ(stock(cell), (R::QbmmMomentBlock{count, count, count, count}));
    record_maximum(*op.grid, "first_subcycle_duration_s", safe_h);
}

TEST(LowMachQbmmKineticTransportTest, CounterflowHasDistinctMomentEnergyVolumeAndMomentumPackets)
{
    Operator op(mesh({0., 1., 2.}));
    record_ownership(*op.grid, 2);
    const auto opts = options(op.species, directional_slip, op.bounds);
    op.fill(
        opts, [](auto x) { return x.x < 1. ? R::QbmmMomentBlock{2, 2, 2, 2} : R::QbmmMomentBlock{1, 2, 4, 8}; },
        [](auto x) { return x.x < 1. ? 2. : 1.; });
    for (size_t c = 0; c < op.accepted.cells.size(); ++c)
        op.accepted.cells[c].velocity.y = op.grid->cell_centroid(static_cast<LO>(c)).x < 1. ? 2. : -1.;
    constexpr double dt = .25;
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, dt);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{dt}));
    EXPECT_DOUBLE_EQ(result.maximum_cfl, 1. / 16.);
    double max_error = 0.;
    op.interior_flux(result,
        [&](auto, const auto& packet, const auto& normal)
        {
            const double orientation = normal.x;
            const auto& kinetic = *packet.kinetic;
            const auto& rightgoing = orientation > 0 ? kinetic.forward : kinetic.backward;
            const auto& leftgoing = orientation > 0 ? kinetic.backward : kinetic.forward;
            R::QbmmMomentBlock forward{}, backward{};
            for (size_t k = 0; k < forward.size(); ++k)
            {
                forward[k] = orientation * rightgoing.moment_flux[k];
                backward[k] = orientation * leftgoing.moment_flux[k];
            }
            check_moments(forward, {.5, .5, .5, .5}, max_error);
            check_moments(backward, {-.25, -.5, -1., -2.}, max_error);
            EXPECT_DOUBLE_EQ(orientation * rightgoing.species_molar_flux.at(0), .5);
            EXPECT_DOUBLE_EQ(orientation * leftgoing.species_molar_flux.at(0), -.5);
            EXPECT_DOUBLE_EQ(orientation * rightgoing.integrated_count, .125);
            EXPECT_DOUBLE_EQ(orientation * leftgoing.integrated_count, -.0625);
            EXPECT_DOUBLE_EQ(orientation * rightgoing.integrated_mass, .125);
            EXPECT_DOUBLE_EQ(orientation * leftgoing.integrated_mass, -.125);
            EXPECT_DOUBLE_EQ(orientation * rightgoing.internal_energy_flux, 1.5);
            EXPECT_DOUBLE_EQ(orientation * leftgoing.internal_energy_flux, -.5);
            EXPECT_DOUBLE_EQ(orientation * rightgoing.material_volume_flux, .25);
            EXPECT_DOUBLE_EQ(orientation * leftgoing.material_volume_flux, -.125);
            EXPECT_DOUBLE_EQ(kinetic.theta, 1.5);
            EXPECT_DOUBLE_EQ(orientation * kinetic.slip_volume_flux, .125);
            EXPECT_DOUBLE_EQ(orientation * packet.material_volume_flux, .125);
            EXPECT_DOUBLE_EQ(orientation * packet.population_number_flux, .25);
            EXPECT_DOUBLE_EQ(orientation * packet.shape_flux.H2, -.5);
            EXPECT_DOUBLE_EQ(orientation * packet.shape_flux.H3, -1.5);
            EXPECT_DOUBLE_EQ(packet.derived_mass_flux, 0.);
            EXPECT_DOUBLE_EQ(orientation * (rightgoing.drift_momentum_flux.x + leftgoing.drift_momentum_flux.x), .25);
            EXPECT_DOUBLE_EQ(
                orientation * (rightgoing.carrier_momentum_flux.y + leftgoing.carrier_momentum_flux.y), 1.5);
        });
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const bool left = op.grid->cell_centroid(static_cast<LO>(c)).x < 1.;
        check_moments(stock(candidate.cells[c]),
            left ? R::QbmmMomentBlock{31. / 16., 2., 17. / 8., 19. / 8.}
                 : R::QbmmMomentBlock{17. / 16., 2., 31. / 8., 61. / 8.},
            max_error);
        EXPECT_DOUBLE_EQ(candidate.cells[c].phases[0].internal_energy, left ? 5.75 : 2.25);
        EXPECT_DOUBLE_EQ(result.advected_material_volumes[c][0], left ? 31. / 32. : 17. / 32.);
    }
    record_maximum(*op.grid, "maximum_moment_error", max_error);
}

TEST(LowMachQbmmKineticTransportTest, TiedBranchContributesHalfOfItsPressureDerivative)
{
    Operator op(mesh({0., 1., 2.}));
    const auto opts = options(
        op.species, [](const R::QbmmNodeSlipInput& in)
        { return std::array{in.location.value().centroid[0] < 1. ? -.25 : -.5, 0., 0.}; }, op.bounds);
    op.fill(
        opts, [](auto x) { return x.x < 1. ? R::QbmmMomentBlock{2, 2, 2, 2} : R::QbmmMomentBlock{1, 2, 4, 8}; },
        [](auto x) { return x.x < 1. ? 2. : 1.; });
    for (const auto f : op.carrier.owned_face_ids())
        if (op.grid->is_interior_face(f))
            op.carrier.set_owned_value(f, .25 * op.grid->face_area_vector(f).x);
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, .25);
    op.interior_flux(result,
        [&](auto face, const auto& packet, const auto& normal)
        {
            const auto& kinetic = *packet.kinetic;
            const auto& tied = normal.x > 0 ? kinetic.forward : kinetic.backward;
            EXPECT_EQ(tied.moment_flux, R::QbmmMomentBlock{});
            EXPECT_DOUBLE_EQ(kinetic.theta, 1.);
            EXPECT_DOUBLE_EQ(normal.x * kinetic.slip_volume_flux, -.375);
            EXPECT_DOUBLE_EQ(normal.x * packet.material_volume_flux, -.125);
            EXPECT_DOUBLE_EQ(
                packet.material_volume_flux, kinetic.theta * op.carrier.value(face) + kinetic.slip_volume_flux);
        });
}

TEST(LowMachQbmmKineticTransportTest, ProjectedSlipCancellationPreservesTheActiveBranch)
{
    constexpr double area = 1. - 0x1p-52, slip = 1. + 0x1p-52, dt = 0x1p102;
    Operator op(mesh({0., 1., 2.}, {0., area}), 8.);
    record_ownership(*op.grid, 2);
    const auto opts = options(op.species, [](const auto&) { return std::array{slip, 0., 0.}; }, op.bounds);
    op.fill(
        opts, [](auto x) { return x.x > 1. ? R::QbmmMomentBlock{1., 1., 1., 1.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; });
    for (const auto face : op.carrier.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.carrier.set_owned_value(face, -std::copysign(1., op.grid->face_area_vector(face).x));
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, dt);
    // Algebra gives (1 - a)(1 + a) - 1 = -a^2. The donor is the right cell,
    // and exactly one fourth of a unit bulk volume traverses the face.
    const double count = .25 / area;
    EXPECT_DOUBLE_EQ(result.maximum_cfl, count);
    op.interior_flux(result,
        [&](auto face, const auto& packet, const auto& normal)
        {
            const auto& kinetic = packet.kinetic.value();
            const double orientation = std::copysign(1., normal.x);
            EXPECT_DOUBLE_EQ(orientation * packet.population_number_flux, -count / dt);
            EXPECT_DOUBLE_EQ(kinetic.theta, 1. / (8. * area));
            EXPECT_DOUBLE_EQ(orientation * packet.material_volume_flux, -count / (8. * dt));
            EXPECT_DOUBLE_EQ(
                packet.material_volume_flux, std::fma(kinetic.theta, op.carrier.value(face), kinetic.slip_volume_flux) +
                                                 kinetic.pressure_rounding_flux);
        });
}

TEST(LowMachQbmmKineticTransportTest, SubnormalVolumeAndPressureCoefficientRemainRepresented)
{
    Operator op(mesh({0., 1., 2.}), 8.);
    const auto opts = options(op.species, [](const auto&) { return std::array{0., 0., 0.}; }, op.bounds);
    op.fill(
        opts,
        [](auto x)
        {
            // A small stock at ordinary density 8 gives an exactly subnormal physical phase volume.
            constexpr double atom_count = 0x1p-1020;
            return x.x < 1. ? R::QbmmMomentBlock{atom_count, atom_count, atom_count, atom_count} : R::QbmmMomentBlock{};
        },
        [](auto) { return 1.; });
    for (const auto f : op.carrier.owned_face_ids())
        if (op.grid->is_interior_face(f))
            op.carrier.set_owned_value(f, .5 * op.grid->face_area_vector(f).x);
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, .5);
    op.interior_flux(result,
        [&](auto face, const auto& packet, const auto& normal)
        {
            EXPECT_EQ(std::fpclassify(packet.kinetic->theta), FP_SUBNORMAL);
            EXPECT_EQ(std::fpclassify(packet.material_volume_flux), FP_SUBNORMAL);
            EXPECT_EQ(packet.kinetic->theta, 0x1p-1023);
            EXPECT_EQ(normal.x * packet.material_volume_flux, 0x1p-1024);
            EXPECT_EQ(packet.material_volume_flux, packet.kinetic->theta * op.carrier.value(face));
            EXPECT_EQ(packet.kinetic->pressure_rounding_flux, 0.);
        });
}

TEST(LowMachQbmmKineticTransportTest, SubnormalThetaRoundingKeepsPhysicalSlipAndVolumeFluxDistinct)
{
    constexpr double width = 0x1p54, rate = 0x1p10, dt = 0x1p43;
    constexpr double floor = std::numeric_limits<double>::denorm_min();
    constexpr double quanta = 0x1p46 + 1.;
    Operator op(mesh({0., width, 2. * width}), 8.);
    const auto opts = options(op.species, [](const auto&) { return std::array{0., 0., 0.}; }, op.bounds);
    op.fill(
        opts,
        [](auto x)
        {
            // Leave enough subnormal quanta to represent the donor's diameter
            // moments at the native geometry tolerance while retaining a tie in theta.
            constexpr double count = (quanta + .5) * 0x1p-1017;
            return x.x < width ? R::QbmmMomentBlock{count, count, count, count} : R::QbmmMomentBlock{};
        },
        [](auto) { return 1.; });
    for (const auto f : op.carrier.owned_face_ids())
        if (op.grid->is_interior_face(f))
            op.carrier.set_owned_value(f, rate * op.grid->face_area_vector(f).x);
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, dt);
    EXPECT_EQ(result.substeps, 1);
    op.interior_flux(result,
        [&](auto face, const auto& p, const auto& normal)
        {
            const auto& k = p.kinetic.value();
            // True theta=(quanta+0.5)*floor ties upward to an even number of quanta.
            EXPECT_EQ(k.theta, (quanta + 1.) * floor);
            EXPECT_EQ(k.slip_volume_flux, 0.);
            EXPECT_EQ(normal.x * k.pressure_rounding_flux, -512. * floor);
            EXPECT_EQ(normal.x * p.material_volume_flux, (quanta + .5) * rate * floor);
            EXPECT_EQ(p.material_volume_flux, k.theta * op.carrier.value(face) + k.pressure_rounding_flux);
        });
}

TEST(LowMachQbmmKineticTransportTest, SubnormalSubstepSurvivesAnOutgoingRateBeyondBinary64)
{
    constexpr double width = 0x1p-20, rate = 0x1p1010, atom_count = 0x1p-32;
    constexpr double h = 0x1p-1031, dt = 2. * h;
    Operator op(mesh({0., width, 2. * width}), 8.);
    record_ownership(*op.grid, 2);
    const auto opts = options(op.species, [](const auto&) { return std::array{0., 0., 0.}; }, op.bounds, .5, 2);
    op.fill(
        opts,
        [](auto x)
        {
            return x.x < width ? R::QbmmMomentBlock{atom_count, atom_count, atom_count, atom_count}
                               : R::QbmmMomentBlock{};
        },
        [](auto) { return 1.; });
    for (const auto face : op.carrier.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.carrier.set_owned_value(face, rate * op.grid->face_area_vector(face).x);
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, dt);
    EXPECT_EQ(std::fpclassify(h), FP_SUBNORMAL);
    EXPECT_EQ(result.substep_durations, (std::vector{h, h}));
    EXPECT_EQ(result.maximum_cfl, .5);
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const bool donor = op.grid->cell_centroid(static_cast<LO>(c)).x < width;
        const double expected = (donor ? .25 : .75) * atom_count;
        EXPECT_EQ(stock(candidate.cells[c]), (R::QbmmMomentBlock{expected, expected, expected, expected}));
    }
}

TEST(LowMachQbmmKineticTransportTest, TwoSizesCrossPartitionFacesIntoVacuumWithDistinctKnownWeights)
{
    Operator op(mesh({0., 1., 2., 3.}), 8., -3.);
    record_ownership(*op.grid, 3);
    const auto opts = options(op.species,
        [](const auto& atom) { return std::array{atom.mass_kg_per_bubble / 4., 0., 0.}; }, op.bounds, .25, 2,
        Gate::ReportOnly, {{Range::NodeMassKgPerBubble, .5, 1.5, {}}, {Range::TimeStepS, .5, .5, {}}});
    op.fill(
        opts, [](auto x) { return x.x < 1. ? R::QbmmMomentBlock{3., 4., 6., 10.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; }, -3.);
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, 1.);
    EXPECT_EQ(result.substeps, 2);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{.5, .5}));
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .25);
    constexpr std::array small{49. / 32., 7. / 16., 1. / 32.};
    constexpr std::array large{9. / 16., 3. / 8., 1. / 16.};
    double max_error = 0.;
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const auto i = static_cast<size_t>(op.grid->cell_centroid(static_cast<LO>(c)).x);
        const auto expected = moments(std::array<R::QbmmAtom, 2>{{{small[i], 1.}, {large[i], 2.}}});
        check_moments(stock(candidate.cells[c]), expected, max_error);
        EXPECT_NEAR(candidate.cells[c].phases[0].internal_energy, -3. * expected[1], 2e-13);
        EXPECT_NEAR(result.advected_material_volumes[c][0], expected[1] / 8., 2e-13);
    }
    int flowing_partition_faces = 0;
    op.interior_flux(result,
        [&](auto face, const auto& packet, const auto& normal)
        {
            EXPECT_GT(packet.population_number_flux * normal.x, 0.);
            if (!op.grid->is_owned_cell(op.grid->owner_cell(face)) ||
                !op.grid->is_owned_cell(op.grid->neighbor_cell(face)))
                ++flowing_partition_faces;
        });
    EXPECT_GE(reduce(*op.grid, Teuchos::REDUCE_SUM, flowing_partition_faces),
        op.grid->owned_cell_map()->getComm()->getSize() > 1 ? 1 : 0);
    int small_receipts = 0, large_receipts = 0;
    for (const auto& receipt : result.law_receipts)
    {
        EXPECT_EQ(receipt.duration, .5);
        const bool big = receipt.mass_kg_per_bubble > 1.5;
        if (big)
            ++large_receipts;
        else
            ++small_receipts;
        EXPECT_TRUE(receipt.applicability.gate_passed);
        EXPECT_EQ(receipt.applicability.applicable, !big);
        EXPECT_EQ(receipt.applicability.ranges.size(), 2);
        if (receipt.applicability.ranges.size() != 2)
            continue;
        EXPECT_EQ(receipt.applicability.ranges[0].status, big ? Status::OutsideRange : Status::WithinRange);
        EXPECT_EQ(receipt.applicability.ranges[1].status, Status::WithinRange);
        EXPECT_EQ(receipt.applicability.ranges[1].observed_minimum, .5);
    }
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, small_receipts), 3);
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, large_receipts), 3);
    const auto rejected = options(op.species, opts.qbmm_node_slip_laws[0]->evaluate, op.bounds, .25, 2,
        Gate::RejectOutside, {{Range::NodeMassKgPerBubble, .5, 1.5, {}}});
    op.accepted.qbmm_families = {Flow::qbmm_family_stamp(*rejected.qbmm_families[0])};
    EXPECT_THROW((void) op.advance(rejected, candidate, 1.), std::exception);
    record_maximum(*op.grid, "maximum_differential_moment_error", max_error);
}

TEST(LowMachQbmmKineticTransportTest, NewlyOccupiedFastReceiverRecomputesCflAndAppliesItsActualDurationGate)
{
    Operator op(mesh({0., 1., 2., 3.}));
    const Law velocity = [](const R::QbmmNodeSlipInput& in)
    {
        const double x = in.location.value().centroid[0];
        return std::array{x < 1. ? 1. : (x < 2. ? 4. : 0.), 0., 0.};
    };
    const auto opts =
        options(op.species, velocity, op.bounds, .5, 3, Gate::ReportOnly, {{Range::TimeStepS, .25, 1., {}}});
    op.fill(
        opts, [](auto x) { return x.x < 1. ? R::QbmmMomentBlock{1., 1., 1., 1.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; });
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, .75);
    EXPECT_EQ(result.substeps, 3);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{.5, .125, .125}));
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .5);
    constexpr std::array expected{49. / 128., 27. / 128., 52. / 128.};
    double max_error = 0.;
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const double count = expected[static_cast<size_t>(op.grid->cell_centroid(static_cast<LO>(c)).x)];
        check_moments(stock(candidate.cells[c]), {count, count, count, count}, max_error);
    }
    int faster = 0;
    for (const auto& receipt : result.law_receipts)
        if (receipt.slip_velocity.x == 4.)
        {
            ++faster;
            EXPECT_GT(receipt.substep, 0U);
            EXPECT_EQ(receipt.duration, .125);
            EXPECT_FALSE(receipt.applicability.applicable);
            EXPECT_TRUE(receipt.applicability.gate_passed);
        }
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, faster), 2);
    const auto rejected =
        options(op.species, velocity, op.bounds, .5, 3, Gate::RejectOutside, {{Range::TimeStepS, .25, 1., {}}});
    op.accepted.qbmm_families = {Flow::qbmm_family_stamp(*rejected.qbmm_families[0])};
    EXPECT_THROW((void) op.advance(rejected, candidate, .75), std::exception);
    auto capped = opts;
    capped.qbmm_model.maximum_kinetic_substeps = 2;
    op.accepted.qbmm_families = {Flow::qbmm_family_stamp(*capped.qbmm_families[0])};
    EXPECT_THROW((void) op.advance(capped, candidate, .75), std::exception);
    record_maximum(*op.grid, "maximum_fast_receiver_moment_error", max_error);
}

TEST(LowMachQbmmKineticTransportTest, SimultaneousTwoFaceReservationEvacuatesTheDonorExactly)
{
    Operator op(mesh({0., 1., 2.}, {0., 1., 2.}));
    record_ownership(*op.grid, 4);
    const auto opts = options(
        op.species,
        [](const R::QbmmNodeSlipInput& in)
        {
            const auto& x = in.location.value().centroid;
            return std::array{x[0] < 1. ? 1. : 0., x[1] < 1. ? 1. : 0., 0.};
        },
        op.bounds);
    op.fill(
        opts, [](auto x) { return x.x < 1. && x.y < 1. ? R::QbmmMomentBlock{1., 1., 1., 1.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; });
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, .5);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_DOUBLE_EQ(result.maximum_cfl, 1.);
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const auto x = op.grid->cell_centroid(static_cast<LO>(c));
        const double expected = (x.x < 1.) != (x.y < 1.) ? .5 : 0.;
        EXPECT_EQ(stock(candidate.cells[c]), (R::QbmmMomentBlock{expected, expected, expected, expected}));
        EXPECT_EQ(candidate.cells[c].phases[0].internal_energy, expected);
        EXPECT_EQ(result.advected_material_volumes[c][0], expected / 4.);
    }
    int local_streams = 0;
    op.interior_flux(result,
        [&](auto, const auto& packet, const auto&)
        {
            if (packet.population_number_flux != 0.)
            {
                ++local_streams;
                EXPECT_DOUBLE_EQ(std::abs(packet.population_number_flux), 1.);
            }
        });
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, local_streams), 2);
}

TEST(LowMachQbmmKineticTransportTest, ThreeSupportCompressionCreatesAFasterSecondSubstepDonor)
{
    Operator op(mesh({0., 1., 2., 3.}));
    record_ownership(*op.grid, 3);
    const auto opts = options(
        op.species,
        [](const R::QbmmNodeSlipInput& atom)
        {
            const double x = atom.location.value().centroid[0], xi = atom.mass_kg_per_bubble;
            return std::array{x < 1. ? 1. : (x < 2. && xi > 3. && xi < 4. ? 4. : 0.), 0., 0.};
        },
        op.bounds, .5, 2);
    op.fill(
        opts,
        [](auto x)
        {
            if (x.x < 1.)
                return R::QbmmMomentBlock{2., 2., 2., 2.};
            if (x.x < 2.)
                return R::QbmmMomentBlock{2., 6., 20., 72.};
            return R::QbmmMomentBlock{};
        },
        [](auto) { return 1.; });
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, .625);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{.5, .125}));
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .5);

    // After h=.5 the middle cell has unit counts at 1, 2 and 4.
    // Its monic degree-two moment-orthogonal polynomial is
    // x^2 - (36/7) x + 5. Only the upper root is in (3,4).
    const double root_gap = std::sqrt(79.);
    const double low = (18. - root_gap) / 7., high = (18. + root_gap) / 7.;
    const double fast_count = (7. - 3. * low) / (high - low);
    const auto fast_packet = moments(std::array{R::QbmmAtom{.5 * fast_count, high}});
    double max_error = 0.;
    int changed_support = 0;
    for (const auto& receipt : result.law_receipts)
        if (receipt.substep == 1 && receipt.slip_velocity.x == 4.)
        {
            EXPECT_NEAR(receipt.mass_kg_per_bubble, high, 2e-13);
            EXPECT_DOUBLE_EQ(receipt.duration, .125);
            ++changed_support;
        }
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, changed_support), 1);
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const double x = op.grid->cell_centroid(static_cast<LO>(c)).x;
        auto expected = x < 1. ? R::QbmmMomentBlock{.875, .875, .875, .875} : fast_packet;
        if (x >= 1. && x < 2.)
        {
            expected = {3.125, 7.125, 21.125, 73.125};
            for (size_t k = 0; k < expected.size(); ++k)
                expected[k] -= fast_packet[k];
        }
        check_moments(stock(candidate.cells[c]), expected, max_error);
    }
    record_maximum(*op.grid, "mixed_three_support_maximum_error", max_error);
}

TEST(LowMachQbmmKineticTransportTest, SubstepsCoverTheBinary64IntervalExactly)
{
    Operator op(mesh({0., 1., 2.}));
    const auto opts = options(
        op.species, [](const R::QbmmNodeSlipInput& atom)
        { return std::array{atom.location.value().centroid[0] < 1. && atom.count > .9 ? 10. : 0., 0., 0.}; }, op.bounds,
        .5, 2);
    op.fill(
        opts, [](auto x) { return x.x < 1. ? R::QbmmMomentBlock{1., 1., 1., 1.} : R::QbmmMomentBlock{}; },
        [](auto) { return 1.; });
    Flow::State candidate;
    const auto result = op.advance(opts, candidate, 1.);
    const double endpoint = std::nextafter(.95, 1.);
    ASSERT_EQ(result.substep_durations.size(), 2);
    EXPECT_EQ(result.substep_durations[0], 1. - endpoint);
    EXPECT_EQ(result.substep_durations[1], endpoint);
    EXPECT_LE(result.substep_durations[0], .05);
    EXPECT_LE(result.maximum_cfl, .5);
}

TEST(LowMachQbmmKineticTransportTest, UnrepresentedNetWithdrawalRejectsOnlyWhenStreamsFailToCancel)
{
    Operator op(mesh({0., 1., 2.}));
    record_ownership(*op.grid, 2);
    const auto slip = [](double right_speed)
    {
        return [right_speed](const R::QbmmNodeSlipInput& atom)
        { return std::array{atom.location.value().centroid[0] < 1. ? 0x1p-60 : -right_speed, 0., 0.}; };
    };
    const auto equal = options(op.species, slip(0x1p-60), op.bounds);
    op.fill(equal, [](auto) { return R::QbmmMomentBlock{1., 1., 1., 1.}; }, [](auto) { return 1.; });
    Flow::State candidate;
    bool owns_debit = false;
    for (size_t c = 0; c < op.accepted.cells.size(); ++c)
        owns_debit |= op.grid->cell_centroid(static_cast<LO>(c)).x < 1.;
    for (const double right_speed : {0x1p-61, std::nextafter(0x1p-60, 0.)})
    {
        const auto unequal = options(op.species, slip(right_speed), op.bounds);
        std::string failure;
        try
        {
            (void) op.advance(unequal, candidate, 1.);
        }
        catch (const std::exception& error)
        {
            failure = error.what();
        }
        EXPECT_EQ(failure, owns_debit ? "QBMM kinetic net transfer is too small to update its authoritative stock."
                                      : "QBMM kinetic constructive compression and balances failed on another rank.");
    }
    const auto result = op.advance(equal, candidate, 1.);
    op.interior_flux(result,
        [](auto, const auto& packet, const auto&)
        {
            EXPECT_EQ(packet.kinetic->forward.integrated_count, 0x1p-60);
            EXPECT_EQ(packet.kinetic->backward.integrated_count, -0x1p-60);
            EXPECT_EQ(packet.population_number_flux, 0.);
        });
    for (size_t c = 0; c < candidate.cells.size(); ++c)
        EXPECT_EQ(stock(candidate.cells[c]), stock(op.accepted.cells[c]));
}
} // namespace
