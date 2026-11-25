/** Independent finite-volume oracles for the realizable linear SSPRK2 transport. */
#include "QbmmKineticTestSupport.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/details/LowMachQbmmLinearTransport.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
namespace Kernel = R::qbmm_linear_detail;
using Vec = Flow::Vector;
using Stock = std::array<double, 8>;
using Weights = std::array<double, 2>;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double pressure = 8.;
constexpr size_t number = 0, mass = 1, h2 = 2, h3 = 3, species_a = 4, species_b = 5, energy = 6, work = 7;
constexpr std::array<std::string_view, 8> channel_names{"H0", "H1", "H2", "H3", "species_A", "species_B", "U", "W"};

Stock stock_from_atoms(double w1, double w2, double internal_energy)
{
    const double m = w1 + 2. * w2;
    const double a = .25 * m, b = .375 * m;
    // Both species have Cv = 2 J/(mol K), T_ref = 2 K, and R = 1 J/(mol K).
    return {w1 + w2, m, w1 + 4. * w2, w1 + 8. * w2, a, b, internal_energy,
        (2. * (a + b) + .5 * internal_energy) / pressure};
}

Stock stock_from_weights(const Weights& weights)
{
    return stock_from_atoms(weights[0], weights[1], -.25 * (weights[0] + 2. * weights[1]));
}

struct Problem
{
    SP<Flow::Mesh> grid;
    std::shared_ptr<const R::SpeciesCatalog> species;
    std::shared_ptr<const R::MixtureThermodynamics> gas;
    Flow::Options options;
    Flow::Traits::scalar_face_type flux;
    Flow::State state;
    std::vector<std::vector<double>> volumes;

    explicit Problem(SP<Flow::Mesh> mesh)
        : grid(std::move(mesh)), species(std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
                                     {47, "gas A", 1., {{"A", 1}}, {R::SpeciesPhase::Dispersed}},
                                     {48, "gas B", 2., {{"B", 1}}, {R::SpeciesPhase::Dispersed}}})),
          gas(R::make_ideal_gas_mixture(species, {{2., 2., 0.}, {2., 2., 0.}}, {.5, 4., 4., 12.},
              R::EnergyConvention::IncludesFormationEnergy, 1.)),
          flux(grid, 0., "prescribed_closed_carrier_flux"), volumes(grid->num_owned_cells(), std::vector<double>(1))
    {
        const int ranks = grid->owned_cell_map()->getComm()->getSize();
        if (const char* expected = std::getenv("SIMPLEFLUID_Q8C_EXPECTED_RANKS"))
            EXPECT_EQ(ranks, std::stoi(expected));
        options.qbmm_model.family_transport = Model::FamilyTransport::RealizableLinearSSPRK2;
        options.minimum_pressure = 4.;
        options.maximum_pressure = 12.;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.phase_roles = {{Flow::PhaseRole::DispersedGas, "linear advection bubbles"}};
        const auto family =
            std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), "linear advection bubbles",
                options.phase_roles[0], 1., std::vector<R::QbmmMassFraction>{{47, .25}, {48, .75}});
        options.qbmm_families = {family};
        state.thermodynamic_pressure = pressure;
        state.qbmm_families = {Flow::qbmm_family_stamp(*family)};
        state.cells.resize(grid->num_owned_cells());
    }

    static double temperature(double a, double b, double u) { return a + b > 0. ? 2. + u / (2. * (a + b)) : 0.; }

    void set(LO cell, const Stock& stock)
    {
        auto& value = state.cells[cell];
        value.phases = {{gas, {stock[species_a], stock[species_b]}, pressure, stock[energy],
            temperature(stock[species_a], stock[species_b], stock[energy])}};
        value.phase_numbers = {stock[number]};
        value.qbmm_shape = {R::QbmmShapeState{stock[h2], stock[h3]}};
        volumes[cell][0] = stock[work];
    }

    Stock get(LO cell) const
    {
        const auto& value = state.cells[cell];
        const auto& thermal = value.phases[0];
        return {*value.phase_numbers[0], thermal.moles[0] + 2. * thermal.moles[1], value.qbmm_shape[0]->H2,
            value.qbmm_shape[0]->H3, thermal.moles[0], thermal.moles[1], thermal.internal_energy, volumes[cell][0]};
    }

    Kernel::TransportResult advance(double dt, Kernel::TraceLimiter limiter = Kernel::TraceLimiter::Outgoing)
    {
        auto candidate = state;
        auto result = Kernel::transport(grid, options, state, state, candidate, flux, volumes, dt, 0, limiter);
        state = std::move(candidate);
        volumes = result.advected_material_volumes;
        for (auto& cell : state.cells)
        {
            auto& phase = cell.phases[0];
            phase.temperature = temperature(phase.moles[0], phase.moles[1], phase.internal_energy);
        }
        return result;
    }

    Stock totals() const
    {
        Stock local{}, global{};
        for (size_t row = 0; row < state.cells.size(); ++row)
        {
            const auto stock = get(static_cast<LO>(row));
            for (size_t k = 0; k < stock.size(); ++k)
                local[k] += stock[k];
        }
        Teuchos::reduceAll(
            *grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
        return global;
    }
};

Stock rates(const Flow::QbmmFaceFlux& packet)
{
    return {packet.population_number_flux, packet.derived_mass_flux, packet.shape_flux.H2, packet.shape_flux.H3,
        packet.species_molar_flux.at(0), packet.species_molar_flux.at(1), packet.internal_energy_flux,
        packet.material_volume_flux};
}

void check_stock(const Stock& actual, const Stock& expected, double tolerance)
{
    for (size_t k = 0; k < actual.size(); ++k)
    {
        SCOPED_TRACE(channel_names[k]);
        EXPECT_NEAR(actual[k], expected[k], tolerance * std::max(1., std::abs(expected[k])));
    }
}

enum class Geometry
{
    Cartesian,
    Cylindrical,
    PolarExtrusion
};
constexpr std::array mesh_families{Geometry::Cartesian, Geometry::Cylindrical, Geometry::PolarExtrusion};
constexpr std::array<std::string_view, 3> family_names{"Cartesian", "Cylindrical", "PolarExtrusion"};

ArrReal edges(unsigned n, double low, double high, bool graded = false)
{
    ArrReal result;
    for (unsigned i = 0; i <= n; ++i)
    {
        const double q = double(i) / n;
        result.push_back(std::lerp(low, high, graded ? q + .22 * q * (1. - q) : q));
    }
    return result;
}

struct Plane
{
    Geometry kind;
    unsigned n;
    ArrReal a, b;

    Plane(Geometry family, unsigned count, bool graded = false)
        : kind(family), n(count), a(edges(n, 1., 2., graded)),
          b(edges(n, kind == Geometry::Cartesian ? -.5 : -1. / 3., kind == Geometry::Cartesian ? .5 : 1. / 3.,
              graded && kind == Geometry::Cartesian))
    {
    }

    Vec point(double u, double v) const
    {
        if (kind == Geometry::Cartesian)
            return {u, v, 0.};
        return {u * std::cos(v), u * std::sin(v), 0.};
    }

    Vec vertex(unsigned i, unsigned j) const { return point(a[i], b[j]); }

    SP<Flow::Mesh> mesh() const
    {
        Flow::Mesh::DistributionOptions options{.allow_empty_partitions = true};
        if (kind == Geometry::Cartesian)
            return std::make_shared<Flow::Mesh>(
                std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{a, b, {0., 1.}}), options);
        if (kind == Geometry::Cylindrical)
            return std::make_shared<Flow::Mesh>(
                std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{a, b, {0., 1.}}), options);
        using Semi = Meshes::SemiStructuredXY_Z;
        Arr<Semi::Vec3> nodes;
        Arr<Arr<unsigned>> cells;
        for (unsigned j = 0; j <= n; ++j)
            for (unsigned i = 0; i <= n; ++i)
                nodes.push_back(vertex(i, j));
        for (unsigned j = 0; j < n; ++j)
            for (unsigned i = 0; i < n; ++i)
            {
                const auto low = i + j * (n + 1), high = low + n + 1;
                cells.push_back({low, low + 1, high + 1, high});
            }
        return std::make_shared<Flow::Mesh>(std::make_shared<Semi>(nodes, cells, ArrReal{0., 1.}), options);
    }

    std::array<unsigned, 2> cell_index(const Flow::Mesh& grid, LO cell) const
    {
        const auto id = static_cast<unsigned>(grid.cell_geometry_global_id(cell));
        return {id % n, id / n};
    }
};

// Tensor-product Gauss-Legendre cubature of the manufactured solution over physical FV cells.
// Native cylindrical cells use the exact polar Jacobian. Extruded cells use their straight-edged quad map.
constexpr std::array gauss_points{.0198550717512319, .101666761293187, .237233795041836, .408282678752175,
    .591717321247825, .762766204958164, .898333238706813, .980144928248768};
constexpr std::array gauss_weights{.0506142681451882, .111190517226687, .156853322938944, .181341891689181,
    .181341891689181, .156853322938944, .111190517226687, .0506142681451882};

template<class Evaluate>
Stock integrate(const Plane& plane, unsigned i, unsigned j, const Evaluate& evaluate, double& volume)
{
    const auto p00 = plane.vertex(i, j), p10 = plane.vertex(i + 1, j);
    const auto p01 = plane.vertex(i, j + 1), p11 = plane.vertex(i + 1, j + 1);
    const double da = plane.a[i + 1] - plane.a[i], db = plane.b[j + 1] - plane.b[j];
    Stock result{};
    volume = 0.;
    for (size_t k = 0; k < gauss_points.size(); ++k)
        for (size_t l = 0; l < gauss_points.size(); ++l)
        {
            const double u = gauss_points[k], v = gauss_points[l];
            double jacobian;
            Vec point;
            if (plane.kind == Geometry::PolarExtrusion)
            {
                const Vec du = (p10 - p00) * (1. - v) + (p11 - p01) * v;
                const Vec dv = (p01 - p00) * (1. - u) + (p11 - p10) * u;
                jacobian = du.x * dv.y - du.y * dv.x;
                point = (p00 + (p10 - p00) * u) * (1. - v) + (p01 + (p11 - p01) * u) * v;
            }
            else
            {
                const double a = plane.a[i] + da * u;
                point = plane.point(a, plane.b[j] + db * v);
                jacobian = da * db * (plane.kind == Geometry::Cylindrical ? a : 1.);
            }
            const double dV = gauss_weights[k] * gauss_weights[l] * jacobian;
            const auto value = evaluate(point);
            volume += dV;
            for (size_t m = 0; m < result.size(); ++m)
                result[m] += dV * value[m];
        }
    return result;
}

constexpr double center_x = 1.5, core_radius = .32, outer_radius = .40, profile_radius = .30;

double streamfunction(const Vec& point)
{
    const double x = point.x - center_x, y = point.y;
    const double s = x * x + y * y;
    constexpr double core = core_radius * core_radius, outer = outer_radius * outer_radius, span = outer - core;
    const double q = std::clamp((s - core) / span, 0., 1.);
    // d(2 psi)/d(r^2) is one in the core, then smoothly decreases to zero.
    // Subtract the exterior constant before differencing to make boundary flux exactly zero.
    return s <= core ? .5 * (s - core - .5 * span) : .5 * span * (q - q * q * q + .5 * q * q * q * q - .5);
}

template<class Stream> void set_stream_flux(Problem& op, const Plane& plane, const Stream& psi)
{
    for (const auto face : op.flux.owned_face_ids())
    {
        if (!op.grid->is_interior_face(face))
            continue;
        const auto [i, j] = plane.cell_index(*op.grid, op.grid->owner_cell(face));
        const auto [k, l] = plane.cell_index(*op.grid, op.grid->neighbor_cell(face));
        // The outward flux of (psi_y, -psi_x) is the streamfunction difference
        // along the owner's counterclockwise edge, including when the edge is a cylindrical arc.
        Vec begin, end;
        if (i < k)
        {
            begin = plane.vertex(i + 1, j);
            end = plane.vertex(i + 1, j + 1);
        }
        else if (i > k)
        {
            begin = plane.vertex(i, j + 1);
            end = plane.vertex(i, j);
        }
        else if (j < l)
        {
            begin = plane.vertex(i + 1, j + 1);
            end = plane.vertex(i, j + 1);
        }
        else
        {
            begin = plane.vertex(i, j);
            end = plane.vertex(i + 1, j);
        }
        op.flux.set_owned_value(face, psi(end) - psi(begin));
    }
    op.flux.sync_ghosts();
}

Stock exact_density(const Vec& point, double t)
{
    const double u = (point.x - center_x) / profile_radius, v = point.y / profile_radius;
    const double x = std::cos(t) * u - std::sin(t) * v, y = std::sin(t) * u + std::cos(t) * v;
    const double q = std::max(0., 1. - (u * u + v * v)), smooth = q * q * q * q;
    // The varying parts are supported inside the unit-angular-speed clockwise vortex.
    return stock_from_atoms(.37 + smooth * (.22 + .34 * x + .16 * y + .36 * x * y),
        .22 + smooth * (.11 - .25 * x + .31 * y + .24 * (x * x - y * y)), smooth * (.24 * x + .43 * y + .31 * x * y));
}

using StateSnapshot = std::vector<Stock>;

Stock l1(const Flow::Mesh& grid, const StateSnapshot& a, const StateSnapshot& b)
{
    Stock local{}, global{};
    double local_volume = 0.;
    for (size_t row = 0; row < a.size(); ++row)
    {
        const double bulk = grid.cell_volume(static_cast<LO>(row));
        local_volume += bulk;
        for (size_t k = 0; k < local.size(); ++k)
            local[k] += bulk * std::abs((a[row][k] - b[row][k]) / bulk);
    }
    Teuchos::reduceAll(
        *grid.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    const double total_volume = reduce(grid, Teuchos::REDUCE_SUM, local_volume);
    for (auto& value : global)
        value /= total_volume;
    return global;
}

StateSnapshot march(Problem& op, const StateSnapshot& initial, double time, int steps)
{
    for (size_t row = 0; row < initial.size(); ++row)
        op.set(static_cast<LO>(row), initial[row]);
    for (int stage = 0; stage < steps; ++stage)
    {
        const auto result = op.advance(time / steps);
        EXPECT_EQ(result.substeps, 1);
        EXPECT_GT(result.maximum_cfl, 0.);
        EXPECT_LE(result.maximum_cfl, op.options.qbmm_model.linear_cfl);
    }
    StateSnapshot snapshot(initial.size());
    for (size_t row = 0; row < snapshot.size(); ++row)
        snapshot[row] = op.get(static_cast<LO>(row));
    return snapshot;
}

TEST(LowMachQbmmLinearTransportTest, RejectsUnresolvedOutgoingCylindricalFaceOffset)
{
    // One radial cell and two unequal angular sectors supply only a
    // single displacement direction. The interior angular-face centroid
    // does not lie on their center-to-center line.
    auto mesh = std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{
                                                 ArrReal{1., 2.}, ArrReal{0., .2, .55}, ArrReal{0., 1.}}),
        Flow::Mesh::DistributionOptions{.allow_empty_partitions = true});
    Problem op(mesh);
    for (size_t c = 0; c < op.state.cells.size(); ++c)
        op.set(static_cast<LO>(c), stock_from_weights({.1, .1}));
    for (const auto face : op.flux.owned_face_ids())
        if (mesh->is_interior_face(face))
            op.flux.set_owned_value(face, .01);
    op.flux.sync_ghosts();
    std::string failure;
    try
    {
        (void) op.advance(.01);
    }
    catch (const std::exception& error)
    {
        failure = error.what();
    }
    // The geometry error originates on the upwind owner and is synchronized.
    EXPECT_TRUE(failure.find("cannot reproduce this outgoing face offset") != std::string::npos ||
                failure.find("QBMM linear transport setup failed on another rank") != std::string::npos)
        << failure;
}

TEST(LowMachQbmmLinearTransportTest, SmoothClosedVortexConvergesForEveryChannel)
{
    constexpr double time = .01;
    constexpr std::array resolution{16U, 32U, 64U};
    for (const auto kind : mesh_families)
    {
        const auto name = family_names[static_cast<size_t>(kind)];
        SCOPED_TRACE(name);
        std::array<Stock, resolution.size()> errors{};
        Stock time_coarse{}, time_fine{};
        for (size_t level = 0; level < resolution.size(); ++level)
        {
            const Plane plane(kind, resolution[level]);
            Problem op(plane.mesh());
            set_stream_flux(op, plane, streamfunction);
            StateSnapshot initial(op.state.cells.size()), exact(initial.size());
            for (size_t row = 0; row < initial.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                double divergence = 0.;
                for (const auto face : op.grid->faces(cell))
                    divergence += (op.grid->owner_cell(face) == cell ? 1. : -1.) * op.flux.local_value(face);
                EXPECT_NEAR(divergence, 0., 1e-16);
                const auto [i, j] = plane.cell_index(*op.grid, cell);
                double volume;
                initial[row] = integrate(plane, i, j, [](Vec point) { return exact_density(point, 0.); }, volume);
                // The polygon implementation computes area by differencing coordinate products.
                EXPECT_NEAR(volume, op.grid->cell_volume(cell), 4.e-14 * volume + 8.e-16);
                exact[row] = integrate(plane, i, j, [](Vec point) { return exact_density(point, time); }, volume);
            }
            const auto full = march(op, initial, time, 1);
            errors[level] = l1(*op.grid, full, exact);
            if (level + 1 == resolution.size())
            {
                const auto half = march(op, initial, time, 2);
                const auto quarter = march(op, initial, time, 4);
                time_coarse = l1(*op.grid, full, half);
                time_fine = l1(*op.grid, half, quarter);
            }
        }
        for (size_t channel = 0; channel < errors.front().size(); ++channel)
        {
            SCOPED_TRACE(channel_names[channel]);
            EXPECT_GT(errors.back()[channel], 0.);
            const double first_order = std::log2(errors[0][channel] / errors[1][channel]);
            const double second_order = std::log2(errors[1][channel] / errors[2][channel]);
            EXPECT_GE(first_order, 1.8);
            EXPECT_GE(second_order, 1.8);
            EXPECT_LT(time_coarse[channel], .05 * errors.back()[channel]);
            EXPECT_LT(time_fine[channel], .05 * errors.back()[channel]);
            const std::string key = std::string(name) + '_' + std::string(channel_names[channel]);
            std::ostringstream values;
            values << std::setprecision(9) << errors[0][channel] << ' ' << errors[1][channel] << ' '
                   << errors[2][channel] << " orders " << first_order << ' ' << second_order << " temporal differences "
                   << time_coarse[channel] << ' ' << time_fine[channel] << " temporal_to_spatial "
                   << time_coarse[channel] / errors.back()[channel] << ' '
                   << time_fine[channel] / errors.back()[channel];
            RecordProperty(key, values.str());
            if (Tpetra::getDefaultComm()->getRank() == 0)
                std::cout << key << ' ' << values.str() << '\n';
        }
    }
}

void tvd_metric(const Flow::Mesh& grid, const std::string& key, double value)
{
    testing::Test::RecordProperty(key, value);
    if (grid.owned_cell_map()->getComm()->getRank() == 0)
        std::cout << "TvdMc " << key << '=' << std::setprecision(14) << value << '\n';
}

TEST(LowMachQbmmLinearTransportTest, TvdMcSmoothInteriorConvergesForEveryChannel)
{
    // Constant physical velocity translates the exact solution. Integrate over the central half of
    // each coordinate interval, where the two Euler stages cannot encounter a closed boundary.
    constexpr double vx = .25, vy = .125, dt = .001;
    const auto profile = [](Vec p, double t)
    {
        const double x = p.x - vx * t, y = p.y - vy * t;
        const double w1 = .125 * std::exp(1.25 * x + .875 * y), w2 = .25 * std::exp(.875 * x + 1.125 * y);
        // Manufacture a smooth, signed specific energy independently of the number and mass profiles.
        const double specific_energy = -.25 + .0625 * x + .09375 * y;
        return stock_from_atoms(w1, w2, specific_energy * (w1 + 2. * w2));
    };
    for (const auto kind : mesh_families)
    {
        const auto name = family_names[static_cast<size_t>(kind)];
        SCOPED_TRACE(name);
        Stock previous{};
        for (unsigned n : {16U, 32U, 64U})
        {
            const Plane plane(kind, n);
            Problem op(plane.mesh());
            op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
            set_stream_flux(op, plane, [](Vec p) { return vx * p.y - vy * p.x; });
            StateSnapshot exact(op.state.cells.size());
            for (size_t row = 0; row < exact.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                const auto [i, j] = plane.cell_index(*op.grid, cell);
                double v;
                op.set(cell, integrate(plane, i, j, [&](Vec p) { return profile(p, 0.); }, v));
                exact[row] = integrate(plane, i, j, [&](Vec p) { return profile(p, dt); }, v);
            }
            const auto result = op.advance(dt);
            EXPECT_EQ(result.substeps, 1);
            EXPECT_EQ(result.upstream_returned_packets, 0u);
            Stock local{};
            double local_volume = 0.;
            for (size_t row = 0; row < exact.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                const auto [i, j] = plane.cell_index(*op.grid, cell);
                if (i < n / 4 || j < n / 4 || i >= 3 * n / 4 || j >= 3 * n / 4)
                    continue;
                local_volume += op.grid->cell_volume(cell);
                const auto actual = op.get(cell);
                for (size_t k = 0; k < local.size(); ++k)
                    local[k] += std::abs(actual[k] - exact[row][k]);
            }
            const double volume = reduce(*op.grid, Teuchos::REDUCE_SUM, local_volume);
            for (size_t k = 0; k < local.size(); ++k)
            {
                SCOPED_TRACE(channel_names[k]);
                const double error = reduce(*op.grid, Teuchos::REDUCE_SUM, local[k]) / volume;
                EXPECT_GT(error, 0.);
                if (n > 16)
                {
                    const double order = std::log2(previous[k] / error);
                    EXPECT_GT(order, 1.8);
                    tvd_metric(*op.grid,
                        std::string(name) + "_" + std::string(channel_names[k]) + "_order_" + std::to_string(n), order);
                }
                previous[k] = error;
            }
        }
    }
}

TEST(LowMachQbmmLinearTransportTest, TvdMcStepPreservesBoundsAndStationaryBystanders)
{
    // A divergence-free, closed vortex crosses a positive step. Scalar extrema are independent
    // bounds for every density because both sides have the same composition and specific energy.
    constexpr unsigned n = 12;
    const auto low = stock_from_weights({.125, .25}), high = stock_from_weights({.5, 1.});
    for (const auto kind : mesh_families)
        for (const bool stretched : {false, true})
        {
            SCOPED_TRACE(family_names[static_cast<size_t>(kind)]);
            SCOPED_TRACE(stretched);
            const Plane plane(kind, n, stretched);
            Problem op(plane.mesh());
            op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
            set_stream_flux(op, plane, streamfunction);
            StateSnapshot initial(op.state.cells.size());
            std::vector<bool> stationary(initial.size());
            for (size_t row = 0; row < initial.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                const auto [i, j] = plane.cell_index(*op.grid, cell);
                const double bulk = op.grid->cell_volume(cell);
                double divergence = 0.;
                for (const auto face : op.grid->faces(cell))
                    divergence += (op.grid->owner_cell(face) == cell ? 1. : -1.) * op.flux.local_value(face);
                EXPECT_NEAR(divergence, 0., 1e-16) << i << ',' << j;
                // Put bystanders behind a full stationary ring so their distinct
                // densities cannot enter any transporting cell's reconstruction stencil.
                const auto inactive = [&](unsigned u, unsigned v)
                {
                    const double first = streamfunction(plane.vertex(u, v));
                    return streamfunction(plane.vertex(u + 1, v)) == first &&
                           streamfunction(plane.vertex(u, v + 1)) == first &&
                           streamfunction(plane.vertex(u + 1, v + 1)) == first;
                };
                stationary[row] = inactive(i, j) && (i == 0 || inactive(i - 1, j)) && (j == 0 || inactive(i, j - 1)) &&
                                  (i + 1 == n || inactive(i + 1, j)) && (j + 1 == n || inactive(i, j + 1));
                if (stationary[row])
                    initial[row] = (i + j) % 2 ? Stock{} : stock_from_weights({.125, 0.});
                else
                {
                    initial[row] = i < n / 2 ? low : high;
                    for (auto& value : initial[row])
                        value *= bulk;
                }
                op.set(cell, initial[row]);
            }
            const auto before = op.totals();
            const auto result = op.advance(.02);
            EXPECT_EQ(result.upstream_returned_packets, 0u);
            check_stock(op.totals(), before, 2e-13);
            int moved = 0, vacuum = 0, singleton = 0;
            double maximum_violation = 0.;
            for (size_t row = 0; row < initial.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                const auto actual = op.get(cell);
                if (stationary[row])
                {
                    EXPECT_EQ(actual, initial[row]);
                    vacuum += actual[number] == 0.;
                    singleton += actual[number] != 0.;
                    continue;
                }
                for (size_t k = 0; k < actual.size(); ++k)
                {
                    SCOPED_TRACE(channel_names[k]);
                    const double density = actual[k] / op.grid->cell_volume(cell);
                    maximum_violation = std::max(
                        {maximum_violation, std::min(low[k], high[k]) - density, density - std::max(low[k], high[k])});
                    EXPECT_GE(density, std::min(low[k], high[k]) - 3e-13);
                    EXPECT_LE(density, std::max(low[k], high[k]) + 3e-13);
                }
                moved += std::abs(actual[number] - initial[row][number]) > 1e-8;
            }
            const auto key = std::string(family_names[static_cast<size_t>(kind)]) +
                             (stretched ? "_stretched_step" : "_uniform_step");
            const auto changed = reduce(*op.grid, Teuchos::REDUCE_SUM, moved);
            tvd_metric(
                *op.grid, key + "_maximum_bound_violation", reduce(*op.grid, Teuchos::REDUCE_MAX, maximum_violation));
            tvd_metric(*op.grid, key + "_changed_cells", changed);
            EXPECT_GT(changed, 0);
            EXPECT_GT(reduce(*op.grid, Teuchos::REDUCE_SUM, vacuum), 0);
            EXPECT_GT(reduce(*op.grid, Teuchos::REDUCE_SUM, singleton), 0);
        }
}

TEST(LowMachQbmmLinearTransportTest, TvdMcMatchesExactTwoStageRampAndRetainedMinimum)
{
    for (auto traces : {Kernel::TraceLimiter::AllIncident, Kernel::TraceLimiter::Outgoing})
        for (const bool one_sided : {false, true})
        {
            // On the interior ramp the MC bound is the upstream jump at both stages:
            // (.25, 1, 4.5) gives trace 1.75; the intermediate (.25, .78125, 4.71875)
            // gives trace 1.3125. On the one-sided wall stencil (1, 8) the donor
            // is the stencil minimum. Keeping its normalized retained density at
            // that minimum requires its mean trace in each stage: 1, then .875.
            const ArrReal initial = one_sided ? ArrReal{1., 8.} : ArrReal{.25, 1., 4.5};
            const auto donor = initial.size() - 2, receiver = donor + 1;
            Problem op(mesh(edges(initial.size(), 0., initial.size())));
            op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
            // Equal molar amounts give molecular mass 1.5; U=-2w yields T=1.5
            // exactly at both stages, so the ideal-gas density is exactly 8.
            const auto family = std::make_shared<const R::QbmmPopulationSchema>(op.species, op.species->fingerprint(),
                "linear advection bubbles", op.options.phase_roles[0], 1.,
                std::vector<R::QbmmMassFraction>{{47, 1. / 3.}, {48, 2. / 3.}});
            op.options.qbmm_families = {family};
            op.state.qbmm_families = {Flow::qbmm_family_stamp(*family)};
            const auto ramp_stock = [](double w) -> Stock
            { return {2. * w, 3. * w, 5. * w, 9. * w, w, w, -2. * w, .375 * w}; };
            record_ownership(*op.grid, int(initial.size()));
            for (size_t row = 0; row < op.state.cells.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                const double w = initial[op.grid->cell_geometry_global_id(cell)];
                op.set(cell, ramp_stock(w));
            }
            for (auto face : op.flux.owned_face_ids())
                if (op.grid->is_interior_face(face) && op.grid->face_centroid(face).x == double(receiver))
                    op.flux.set_owned_value(face, .125 * op.grid->face_normal(face).x);
            op.flux.sync_ghosts();
            const double transported = .0625 * (one_sided ? 1. + .875 : 1.75 + 1.3125);
            const auto result = op.advance(1., traces);
            EXPECT_EQ(result.substeps, 1);
            EXPECT_EQ(result.upstream_returned_packets, 0u);
            for (size_t row = 0; row < op.state.cells.size(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                const auto i = static_cast<size_t>(op.grid->cell_geometry_global_id(cell));
                const double w = initial[i] + transported * (int(i == receiver) - int(i == donor));
                if (one_sided)
                    check_stock(op.get(cell), ramp_stock(w), 2e-14);
                else
                    EXPECT_EQ(op.get(cell), ramp_stock(w));
            }
            for (auto face : op.flux.owned_face_ids())
            {
                const auto& packet = result.owned_flux.at(0).at(op.flux.owned_row(face));
                const double w = op.flux.owned_value(face) * 8. * transported;
                if (one_sided)
                    check_stock(rates(packet), ramp_stock(w), 2e-14);
                else
                    EXPECT_EQ(rates(packet), ramp_stock(w));
            }
        }
}

TEST(LowMachQbmmLinearTransportTest, TvdMcShapeOrSignedEnergyExtremumLimitsEveryChannel)
{
    for (int energy_sign : {0, -1, 1})
    {
        Problem op(mesh({0., 1., 2., 3.}));
        op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
        // The first case has H0=(3,7,10.5), H1=(5,8,12), H2=(9,10,15), but H3=(17,14,21).
        // In the other cases every moment is monotone; only the signed specific energy
        // has an extremum. In each case the central family must ship its entire mean block.
        std::array initial{stock_from_weights({1., 2.}), stock_from_weights({6., 1.}), stock_from_weights({9., 1.5})};
        if (energy_sign)
            for (size_t i = 0; i < initial.size(); ++i)
            {
                const double w = 1. + i;
                constexpr std::array specific{-.125, .125, .0625};
                initial[i] = stock_from_atoms(w, w, energy_sign * specific[i] * 3. * w);
            }
        for (size_t row = 0; row < op.state.cells.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            op.set(cell, initial[op.grid->cell_geometry_global_id(cell)]);
        }
        for (auto face : op.flux.owned_face_ids())
            if (op.grid->is_interior_face(face) && op.grid->face_centroid(face).x == 2.)
                op.flux.set_owned_value(face, .125 * op.grid->face_normal(face).x);
        op.flux.sync_ghosts();
        const auto result = op.advance(1.);
        EXPECT_EQ(result.substeps, 1);
        for (auto face : op.flux.owned_face_ids())
        {
            auto expected = initial[1];
            // A=.125*M, B=.125*(M-A). SSPRK2 ships their arithmetic mean.
            for (auto& value : expected)
                value *= op.flux.owned_value(face) * (1. - .0625);
            check_stock(rates(result.owned_flux.at(0).at(op.flux.owned_row(face))), expected, 2e-13);
        }
    }
}

TEST(LowMachQbmmLinearTransportTest, TvdMcBoundsTheReconstructedSpecificEnergyQuotient)
{
    for (const double sign : {-1., 1.})
    {
        Problem op(mesh({0., 1., 2., 3.}));
        op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
        constexpr std::array specific{0., .125, 1.125};
        for (size_t row = 0; row < op.state.cells.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto i = static_cast<size_t>(op.grid->cell_geometry_global_id(cell));
            const double m = i + 1.;
            op.set(cell, stock_from_atoms(.5 * m, .25 * m, sign * m * specific[i]));
        }
        for (auto face : op.flux.owned_face_ids())
            if (op.grid->is_interior_face(face) && op.grid->face_centroid(face).x == 2.)
                op.flux.set_owned_value(face, .25 * op.grid->face_normal(face).x);
        op.flux.sync_ghosts();
        // Changes this small leave both stages' represented input exactly unchanged.
        // The MC increment in |U/M| is min(.125,(.125+1)/4,1)=.125.
        // For the shared factor f, (.25 + .84375*f)/(2 + .5*f) = .25 yields f=8/23.
        const auto result = op.advance(std::ldexp(1., -58));
        const double m = 2. + .5 * (8. / 23.);
        for (auto face : op.flux.owned_face_ids())
        {
            const double phi = op.flux.owned_value(face);
            if (phi == 0.)
                continue;
            const auto& packet = result.owned_flux[0][op.flux.owned_row(face)];
            EXPECT_NEAR(packet.internal_energy_flux / packet.derived_mass_flux, sign * .25, 2e-14);
            const auto expected = stock_from_atoms(phi * .5 * m, phi * .25 * m, phi * sign * .25 * m);
            check_stock(rates(packet), expected, 2e-13);
        }
    }
}

TEST(LowMachQbmmLinearTransportTest, TvdMcReturnsEntireTinyPacketExactlyInEitherDirection)
{
    // A dyadic donor has nonzero amounts in every channel. With one face and two SSPRK
    // stages the exact answer on every partition is two returns and an unchanged donor.
    const double weight = std::ldexp(1., -675);
    const Stock seed = stock_from_weights({weight, 2. * weight});
    for (const int donor : {0, 1})
    {
        Problem op(mesh({0., 1., 2.}));
        op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
        record_ownership(*op.grid, 2);
        for (size_t row = 0; row < op.state.cells.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            op.set(cell, op.grid->cell_geometry_global_id(cell) == donor ? seed : Stock{});
            op.state.cells[row].velocity = {.125, -.5, 1.};
        }
        int remote = 0;
        for (auto face : op.flux.owned_face_ids())
            if (op.grid->is_interior_face(face))
            {
                const auto owner = op.grid->owner_cell(face), neighbor = op.grid->neighbor_cell(face);
                op.flux.set_owned_value(face, op.grid->cell_geometry_global_id(owner) == donor ? .25 : -.25);
                remote += !op.grid->is_owned_cell(owner) || !op.grid->is_owned_cell(neighbor);
            }
        op.flux.sync_ghosts();
        EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, remote), int(Tpetra::getDefaultComm()->getSize() > 1));
        const auto result = op.advance(1.);
        EXPECT_EQ(result.substeps, 1);
        EXPECT_EQ(result.upstream_returned_packets, 2u);
        const auto label = "donor_" + std::to_string(donor);
        tvd_metric(*op.grid, label + "_returned_packets", result.upstream_returned_packets);
        tvd_metric(*op.grid, label + "_ranks", op.grid->owned_cell_map()->getComm()->getSize());
        tvd_metric(
            *op.grid, label + "_empty_ranks", reduce(*op.grid, Teuchos::REDUCE_SUM, int(op.state.cells.empty())));
        EXPECT_EQ(op.totals(), seed);
        for (size_t row = 0; row < op.state.cells.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            EXPECT_EQ(op.get(cell), op.grid->cell_geometry_global_id(cell) == donor ? seed : Stock{});
        }
        for (const auto& packet : result.owned_flux.at(0))
        {
            EXPECT_EQ(rates(packet), Stock{});
            ASSERT_TRUE(packet.kinetic.has_value());
            Flow::QbmmDirectionalFlux empty;
            empty.species_molar_flux = {0., 0.};
            EXPECT_EQ(packet.kinetic->forward, empty);
            EXPECT_EQ(packet.kinetic->backward, empty);
            EXPECT_EQ(packet.kinetic->theta, 0.);
            EXPECT_EQ(packet.kinetic->pressure_rounding_flux, 0.);
        }
    }
}

TEST(LowMachQbmmLinearTransportTest, TvdMcReturnsPacketWhenOnlyHalfItsThirdMomentUnderflows)
{
    // The stage transfer H3=2^-1074 is representable. Its half-stage receipt
    // is not. Count, both species, mass, signed energy, volume, and H2 are normal.
    const double m = std::ldexp(1., -330), dt = std::ldexp(1., -84);
    const Stock seed{1., m, std::ldexp(1., -660), std::ldexp(1., -990), .25 * m, .375 * m, -.25 * m, .140625 * m};
    ASSERT_EQ(seed[h3] * dt, std::numeric_limits<double>::denorm_min());
    ASSERT_EQ(.5 * (seed[h3] * dt), 0.);
    ASSERT_GT(seed[species_a] * dt, 1e-200);
    Problem op(mesh({0., 1., 2.}));
    op.options.qbmm_model.linear_slope_limiter = Model::LinearSlopeLimiter::TvdMc;
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        // Identical neighbors give a flat face trace, so it is the half stage
        // receipt alone that falls below the representable positive range.
        op.set(cell, seed);
    }
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.flux.set_owned_value(face, op.grid->face_normal(face).x);
    op.flux.sync_ghosts();
    const auto result = op.advance(dt);
    EXPECT_EQ(result.upstream_returned_packets, 2u);
    for (size_t row = 0; row < op.state.cells.size(); ++row)
        EXPECT_EQ(op.get(static_cast<LO>(row)), seed);
    for (const auto& packet : result.owned_flux.at(0))
        EXPECT_EQ(rates(packet), Stock{});
}

void check_vacuum_front(Model::LinearSlopeLimiter limiter)
{
    constexpr unsigned n = 12;
    constexpr double time = .02;
    for (const auto kind : mesh_families)
    {
        const auto name = family_names[static_cast<size_t>(kind)];
        SCOPED_TRACE(name);
        const Plane plane(kind, n, true);
        Problem op(plane.mesh());
        op.options.qbmm_model.linear_slope_limiter = limiter;
        set_stream_flux(op, plane, streamfunction);
        StateSnapshot initial(op.state.cells.size());
        int local_rank_one = 0, local_vacuum = 0, local_wetted = 0, local_remote = 0;
        double local_min_volume = std::numeric_limits<double>::infinity(), local_max_volume = 0.;
        for (size_t row = 0; row < initial.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto x = op.grid->cell_centroid(cell);
            const auto bulk = op.grid->cell_volume(cell);
            local_min_volume = std::min(local_min_volume, bulk);
            local_max_volume = std::max(local_max_volume, bulk);
            if (x.x < center_x)
                ++local_vacuum;
            else
            {
                // Use dyadic extensive weights so species-derived mass and rank-one moments agree exactly.
                const double w1 = std::ldexp(std::round(std::ldexp(bulk * (.25 + .0625 * x.x), 24)), -24);
                const double w2 = x.y > 0. ? std::ldexp(std::round(std::ldexp(.125 * bulk, 24)), -24) : 0.;
                local_rank_one += int(w2 == 0.);
                initial[row] = stock_from_atoms(w1, w2, .5 * x.y * (w1 + 2. * w2));
                const auto& h = initial[row];
                EXPECT_EQ(R::invert_qbmm_moments({h[number], h[species_a] + 2. * h[species_b], h[h2], h[h3]}).status,
                    w2 == 0. ? R::InversionStatus::RankOne : R::InversionStatus::TwoNodes);
            }
            op.set(cell, initial[row]);
        }
        const auto before = op.totals();
        const auto result = op.advance(time);
        EXPECT_EQ(result.substeps, 1);
        EXPECT_GT(result.maximum_cfl, 0.);
        EXPECT_LE(result.maximum_cfl, op.options.qbmm_model.linear_cfl);
        check_stock(op.totals(), before, 2e-13);

        for (const auto face : op.flux.owned_face_ids())
        {
            const auto& packet = result.owned_flux.at(0).at(op.flux.owned_row(face));
            EXPECT_TRUE(packet.kinetic.has_value());
            if (packet.kinetic)
            {
                const auto phi = op.flux.owned_value(face);
                const auto& forward = packet.kinetic->forward;
                const auto& backward = packet.kinetic->backward;
                EXPECT_GE(forward.mass_flux, 0.);
                EXPECT_LE(backward.mass_flux, 0.);
                if (phi <= 0.)
                    EXPECT_EQ(forward.moment_flux, R::QbmmMomentBlock{});
                if (phi >= 0.)
                    EXPECT_EQ(backward.moment_flux, R::QbmmMomentBlock{});
            }
            if (!op.grid->is_interior_face(face))
                EXPECT_EQ(rates(packet), Stock{});
            else if (packet.population_number_flux != 0. && (!op.grid->is_owned_cell(op.grid->owner_cell(face)) ||
                                                                !op.grid->is_owned_cell(op.grid->neighbor_cell(face))))
                ++local_remote;
        }
        int local_positive_energy = 0, local_negative_energy = 0, local_static_vacuum = 0;
        int local_wet_rank_one = 0, local_rank_one_to_two = 0;
        for (size_t row = 0; row < initial.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto actual = op.get(cell);
            const auto admissibility = R::invert_qbmm_moments({actual[number], actual[mass], actual[h2], actual[h3]});
            EXPECT_TRUE(admissibility.valid()) << R::qbmm_inversion_status_name(admissibility.status);
            local_wetted += int(initial[row][number] == 0. && actual[number] > 0.);
            local_wet_rank_one +=
                int(initial[row][number] == 0. && admissibility.status == R::InversionStatus::RankOne);
            local_rank_one_to_two += int(initial[row][number] > 0. && initial[row][h2] == initial[row][number] &&
                                         admissibility.status == R::InversionStatus::TwoNodes);
            local_positive_energy += int(actual[energy] > 0.);
            local_negative_energy += int(actual[energy] < 0.);
            if (actual[number] == 0.)
                EXPECT_EQ(actual, Stock{});
            else
            {
                EXPECT_GT(actual[species_a], 0.);
                EXPECT_GT(actual[species_b], 0.);
                EXPECT_GT(actual[work], 0.);
                EXPECT_NEAR(actual[species_a] / actual[mass], .25, 8 * std::numeric_limits<double>::epsilon());
                EXPECT_NEAR(2. * actual[species_b] / actual[mass], .75, 8 * std::numeric_limits<double>::epsilon());
            }

            bool zero_carrier = true;
            for (const auto face : op.grid->faces(cell))
                zero_carrier = zero_carrier && op.flux.local_value(face) == 0.;
            if (initial[row][number] == 0. && zero_carrier)
            {
                EXPECT_EQ(actual, Stock{});
                ++local_static_vacuum;
            }

            Stock predicted = initial[row];
            bool all_faces_owned = true;
            for (const auto face : op.grid->faces(cell))
            {
                if (!op.flux.is_owned(face))
                {
                    all_faces_owned = false;
                    break;
                }
                const double orientation = op.grid->owner_cell(face) == cell ? 1. : -1.;
                const auto packet_stock = rates(result.owned_flux.at(0).at(op.flux.owned_row(face)));
                for (size_t channel = 0; channel < predicted.size(); ++channel)
                    predicted[channel] -= orientation * time * packet_stock[channel];
            }
            if (all_faces_owned)
                check_stock(actual, predicted, 2e-13);
        }
        auto count = [&](int value) { return reduce(*op.grid, Teuchos::REDUCE_SUM, value); };
        const int wet = count(local_wetted), remote = count(local_remote);
        EXPECT_GT(count(local_vacuum), 0);
        EXPECT_GT(count(local_rank_one), 0);
        EXPECT_GT(wet, 0);
        const int static_vacuum = count(local_static_vacuum), wet_rank_one = count(local_wet_rank_one);
        const int rank_one_to_two = count(local_rank_one_to_two);
        EXPECT_GT(static_vacuum, 0);
        EXPECT_GT(wet_rank_one, 0);
        EXPECT_GT(rank_one_to_two, 0);
        EXPECT_GT(count(local_positive_energy), 0);
        EXPECT_GT(count(local_negative_energy), 0);
        EXPECT_GT(reduce(*op.grid, Teuchos::REDUCE_MAX, local_max_volume),
            1.1 * reduce(*op.grid, Teuchos::REDUCE_MIN, local_min_volume));
        if (op.grid->owned_cell_map()->getComm()->getSize() > 1)
            EXPECT_GT(remote, 0);
        testing::Test::RecordProperty(std::string(name) + "_newly_wet_cells", wet);
        testing::Test::RecordProperty(std::string(name) + "_exact_static_vacuum_cells", static_vacuum);
        testing::Test::RecordProperty(std::string(name) + "_vacuum_to_rank_one_cells", wet_rank_one);
        testing::Test::RecordProperty(std::string(name) + "_rank_one_to_two_cells", rank_one_to_two);
        testing::Test::RecordProperty(std::string(name) + "_active_partition_faces", remote);
    }
}

TEST(LowMachQbmmLinearTransportTest, VolumeWeightedVacuumFrontPreservesStocks)
{
    check_vacuum_front(Model::LinearSlopeLimiter::RealizabilityOnly);
}

TEST(LowMachQbmmLinearTransportTest, TvdMcVacuumFrontPreservesStocksAndWetsRealizableSingletons)
{
    check_vacuum_front(Model::LinearSlopeLimiter::TvdMc);
}

TEST(LowMachQbmmLinearTransportTest, CombinedOutgoingTracesShareTheDonorLimiter)
{
    Problem op(mesh({0., 1., 2., 3.}, {0., 1., 2., 3.}));
    record_ownership(*op.grid, 9);
    constexpr size_t center = 4, east = 5, north = 7;
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto id = op.grid->cell_geometry_global_id(cell);
        op.set(cell, stock_from_weights({id == east || id == north ? 7. : 1., 1.}));
    }
    int local_outgoing = 0;
    for (const auto face : op.flux.owned_face_ids())
    {
        if (!op.grid->is_interior_face(face))
            continue;
        const auto owner = op.grid->cell_geometry_global_id(op.grid->owner_cell(face));
        const auto neighbor = op.grid->cell_geometry_global_id(op.grid->neighbor_cell(face));
        if (owner == center && (neighbor == east || neighbor == north))
        {
            ++local_outgoing;
            op.flux.set_owned_value(face, .25);
        }
    }
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, local_outgoing), 2);
    op.flux.sync_ghosts();

    // For either outgoing face, the first weight's unlimited increment is (7-1)/4=1.5.
    // Either face alone could use its full trace. Both full traces spend 1.25 from stock 1.
    // Halving both slopes leaves stage-one weights (1/8, 1/2) at the center.
    constexpr Weights stage1_shipment{.25 * (1. + .5 * 1.5), .25};
    constexpr Weights intermediate{1. - 2. * stage1_shipment[0], 1. - 2. * stage1_shipment[1]};
    // The changed east and north means make the next increments (103/64, 1/16).
    // The first weight has a positive remainder at lambda=1/16 and a negative one at lambda=1/8.
    constexpr double lambda2 = 1. / 16.;
    constexpr Weights stage2_shipment{.25 * (intermediate[0] + lambda2 * (7. + stage1_shipment[0] - 1.) / 4.),
        .25 * (intermediate[1] + lambda2 * stage1_shipment[1] / 4.)};
    constexpr Weights average{
        .5 * (stage1_shipment[0] + stage2_shipment[0]), .5 * (stage1_shipment[1] + stage2_shipment[1])};

    const auto result = op.advance(1.);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .5);
    for (const auto face : op.flux.owned_face_ids())
    {
        const auto& packet = result.owned_flux.at(0).at(op.flux.owned_row(face));
        check_stock(rates(packet), op.flux.owned_value(face) != 0. ? stock_from_weights(average) : Stock{}, 2e-13);
    }
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto id = op.grid->cell_geometry_global_id(cell);
        Weights expected{1., 1.};
        if (id == center)
            expected = {1. - 2. * average[0], 1. - 2. * average[1]};
        if (id == east || id == north)
            expected = {7. + average[0], 1. + average[1]};
        check_stock(op.get(cell), stock_from_weights(expected), 2e-13);
    }
}

TEST(LowMachQbmmLinearTransportTest, ForcedCommonTraceFactorKeepsItsConstraintsAcrossASideFlowReversal)
{
    constexpr auto fixed = Kernel::TraceLimiter::AllIncident;
    constexpr auto outgoing = Kernel::TraceLimiter::Outgoing;
    constexpr LO center = 4, east = 5, north = 7, south = 1;
    constexpr double main_rate = 1. / 8., tiny_rate = 1. / (1u << 28);
    const auto replay = [&](Kernel::TraceLimiter limiter, double side_rate)
    {
        Problem op(mesh({0., 1., 2., 3.}, {0., 1., 2., 3.}));
        record_ownership(*op.grid, 9);
        for (size_t row = 0; row < op.state.cells.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto id = op.grid->cell_geometry_global_id(cell);
            op.set(cell, stock_from_weights({id == north ? 7. : id == east ? 3. : 1., 1.}));
        }
        for (const auto face : op.flux.owned_face_ids())
        {
            if (!op.grid->is_interior_face(face))
                continue;
            const auto owner = op.grid->cell_geometry_global_id(op.grid->owner_cell(face));
            const auto neighbor = op.grid->cell_geometry_global_id(op.grid->neighbor_cell(face));
            if (owner != center && neighbor != center)
                continue;
            const auto other = owner == center ? neighbor : owner;
            if (other == east || other == south)
                op.flux.set_owned_value(face, (owner == center ? 1. : -1.) * (other == east ? main_rate : side_rate));
        }
        op.flux.sync_ghosts();

        // The south trace's first atom has weight 1-1.5*lambda in stage one.
        // Both stages admit lambda=1/2 there. The east-only legacy donor admits lambda=1.
        const double lambda = limiter == fixed || side_rate > 0. ? .5 : 1.;
        const auto stage1_east = Weights{main_rate * (1. + lambda / 2.), main_rate};
        const auto stage1_south = Weights{side_rate * (side_rate > 0. ? .25 : 1.), side_rate};
        Weights average{};
        for (size_t atom = 0; atom < average.size(); ++atom)
        {
            const double east_stock = (atom == 0 ? 3. : 1.) + stage1_east[atom];
            const double center_stock = 1. - stage1_east[atom] - stage1_south[atom];
            const double stage2_east = main_rate * (center_stock + lambda * (east_stock - 1.) / 4.);
            average[atom] = (stage1_east[atom] + stage2_east) * .5;
        }

        const auto result = op.advance(1., limiter);
        EXPECT_EQ(result.substeps, 1);
        EXPECT_DOUBLE_EQ(result.maximum_cfl, main_rate + std::max(0., side_rate));
        Stock local_main{};
        for (const auto face : op.flux.owned_face_ids())
        {
            const double rate = op.flux.owned_value(face);
            if (std::abs(rate) != main_rate)
                continue;
            local_main = rates(result.owned_flux.at(0).at(op.flux.owned_row(face)));
            for (auto& amount : local_main)
                amount = std::copysign(1., rate) * amount;
        }
        Stock main{};
        Teuchos::reduceAll(*op.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, int(main.size()),
            local_main.data(), main.data());
        check_stock(main, stock_from_weights(average), 2e-13);
        return main[h3];
    };

    const double fixed_outward = replay(fixed, tiny_rate);
    const double fixed_inward = replay(fixed, -tiny_rate);
    EXPECT_NEAR(fixed_outward - fixed_inward, -69. / 64. * tiny_rate, 2e-13);
    EXPECT_EQ(replay(outgoing, tiny_rate), fixed_outward);
    EXPECT_GT(replay(outgoing, -tiny_rate) - fixed_inward, .01);
}

TEST(LowMachQbmmLinearTransportTest, SignedEnergyRemainderLimitsTheWholeBlockInBothStages)
{
    Problem op(mesh({0., 1., 2.}));
    // The right cell has T=0.4 K with this caloric reference.
    op.gas = R::make_ideal_gas_mixture(op.species, {{2., 2., 0.}, {2., 2., 0.}}, {.125, 4., 4., 12.},
        R::EnergyConvention::IncludesFormationEnergy, 1.);
    const auto left = stock_from_atoms(2., 1., -4.);
    const auto right = stock_from_atoms(1., 1.5, -8.);
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        op.set(cell, op.grid->cell_geometry_global_id(cell) == 0 ? left : right);
    }
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.flux.set_owned_value(face, .25);
    op.flux.sync_ghosts();

    // Both full traces have weights (1.5, 1.25) and specific energy -1.5.
    // At stage one the unlimited remainder has positive weights (1.625, .6875)
    // and specific energy -2.5/3 > -1. The upwind first stage leaves weights
    // (1.5, .75) and e=-1 in the donor. The receiver has e=-1.8.
    // At stage two the unlimited remainder has positive weights (1.125, .4375)
    // and e=-1.5/2 > -1. Only the energy certificate restricts the slope.
    // For stage factors l1 and l2, the actual mean H3 rate is
    // 35/16 + 9*l1/64 + (1/2 + 3*l1/64)*l2, so its upwind value
    // requires both nonnegative factors to vanish at the certificate precision.
    const auto expected_rate = stock_from_atoms(7. / 16., 7. / 32., -7. / 8.);
    const auto result = op.advance(1.);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .25);
    int local_flows = 0;
    for (const auto face : op.flux.owned_face_ids())
    {
        const auto& packet = result.owned_flux.at(0).at(op.flux.owned_row(face));
        if (!op.grid->is_interior_face(face))
        {
            EXPECT_EQ(rates(packet), Stock{});
            continue;
        }
        ++local_flows;
        check_stock(rates(packet), expected_rate, 2e-13);
        EXPECT_LT(packet.internal_energy_flux, 0.);
        EXPECT_GT(packet.derived_mass_flux, 0.);
        EXPECT_NEAR(packet.internal_energy_flux / packet.derived_mass_flux, -1., 2e-13);
    }
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, local_flows), 1);
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const bool donor = op.grid->cell_geometry_global_id(cell) == 0;
        auto expected = donor ? left : right;
        for (size_t channel = 0; channel < expected.size(); ++channel)
            expected[channel] += donor ? -expected_rate[channel] : expected_rate[channel];
        check_stock(op.get(cell), expected, 2e-13);
    }
}

struct LineStage
{
    std::vector<Weights> cells, face_rates;
};

LineStage reference_line_stage(const std::vector<Weights>& initial)
{
    const auto n = initial.size();
    LineStage stage{initial, std::vector<Weights>(n + 1)};
    constexpr double velocity = -.25;
    for (size_t face = 1; face < n; ++face)
    {
        // Physical x-face number equals the source column of its right (upwind) cell.
        const auto right = std::min(face + 1, n - 1), left = face - 1;
        for (size_t k = 0; k < initial[face].size(); ++k)
        {
            const auto slope = (initial[right][k] - initial[left][k]) / double(right - left);
            const auto rate = velocity * (initial[face][k] - .5 * slope);
            stage.face_rates[face][k] = rate;
            stage.cells[left][k] -= rate;
            stage.cells[face][k] += rate;
        }
    }
    return stage;
}

TEST(LowMachQbmmLinearTransportTest, RemoteUpwindTraceMatchesIndependentStageOracle)
{
    constexpr unsigned nx = 12, ny = 3;
    const size_t ranks = Tpetra::getDefaultComm()->getSize();
    Problem op(std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{edges(nx, 0., nx), edges(ny, 0., ny), {0., 1.}}),
        Flow::Mesh::DistributionOptions{.orthogonal_partitions = std::array<size_t, 3>{ranks, 1, 1}}));
    record_ownership(*op.grid, int(nx * ny));

    std::vector<Weights> initial(nx);
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = {1. + .25 * i, 2. - .0625 * i};
    const auto first = reference_line_stage(initial);
    const auto second = reference_line_stage(first.cells);
    const auto donor_velocity = [](size_t column) -> Vec
    { return {.125 + .0625 * column, -.25 - .03125 * column, .5 + .015625 * column}; };
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const size_t column = op.grid->cell_geometry_global_id(cell) % nx;
        op.set(cell, stock_from_weights(initial[column]));
        op.state.cells[cell].velocity = donor_velocity(column);
    }
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.flux.set_owned_value(face, -.25 * op.grid->face_area_vector(face).x);
    op.flux.sync_ghosts();

    const auto result = op.advance(1.);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .25);
    int local_remote = 0;
    double local_remote_trace_increment = 0.;
    for (const auto face : op.flux.owned_face_ids())
    {
        const auto& packet = result.owned_flux.at(0).at(op.flux.owned_row(face));
        if (op.flux.owned_value(face) == 0.)
        {
            EXPECT_EQ(rates(packet), Stock{});
            continue;
        }
        EXPECT_LT(op.flux.owned_value(face), 0.);
        const auto donor = op.grid->neighbor_cell(face);
        const size_t column = op.grid->cell_geometry_global_id(donor) % nx;
        const Weights average{.5 * (first.face_rates[column][0] + second.face_rates[column][0]),
            .5 * (first.face_rates[column][1] + second.face_rates[column][1])};
        const auto expected = stock_from_weights(average);
        check_stock(rates(packet), expected, 2e-13);
        EXPECT_TRUE(packet.kinetic.has_value());
        if (packet.kinetic)
        {
            EXPECT_EQ(packet.kinetic->forward.carrier_momentum_flux, Vec{});
            const auto actual_momentum = packet.kinetic->backward.carrier_momentum_flux;
            const auto expected_momentum = donor_velocity(column) * expected[mass];
            for (size_t axis = 0; axis < 3; ++axis)
                EXPECT_NEAR(actual_momentum.component(axis), expected_momentum.component(axis), 2e-13);
            EXPECT_NEAR(packet.kinetic->theta, expected[work] / -.25, 2e-13);
        }
        if (!op.grid->is_owned_cell(donor))
        {
            ++local_remote;
            const double centered_mass_rate = -.125 * (initial[column][0] + 2. * initial[column][1] +
                                                          first.cells[column][0] + 2. * first.cells[column][1]);
            const double trace_increment = std::abs(packet.derived_mass_flux - centered_mass_rate);
            EXPECT_GT(trace_increment, .01);
            local_remote_trace_increment = std::max(local_remote_trace_increment, trace_increment);
        }
    }
    const int remote = reduce(*op.grid, Teuchos::REDUCE_SUM, local_remote);
    EXPECT_EQ(remote, int((ranks - 1) * ny));
    RecordProperty("remote_upwind_faces", remote);
    record_maximum(*op.grid, "remote_upwind_maximum_trace_increment_kg_per_s", local_remote_trace_increment);

    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const size_t column = op.grid->cell_geometry_global_id(cell) % nx;
        const Weights endpoint{
            .5 * (initial[column][0] + second.cells[column][0]), .5 * (initial[column][1] + second.cells[column][1])};
        check_stock(op.get(cell), stock_from_weights(endpoint), 2e-13);
    }
}

TEST(LowMachQbmmLinearTransportTest, EmptyOwnersParticipateWithAOneCellMesh)
{
    for (const auto kind : mesh_families)
    {
        SCOPED_TRACE(family_names[static_cast<size_t>(kind)]);
        const Plane plane(kind, 1);
        Problem op(plane.mesh());
        record_ownership(*op.grid, 1);
        const auto expected = stock_from_weights({.25, .5});
        for (size_t row = 0; row < op.state.cells.size(); ++row)
            op.set(static_cast<LO>(row), expected);
        op.flux.sync_ghosts();
        const auto result = op.advance(.125);
        EXPECT_EQ(result.maximum_cfl, 0.);
        EXPECT_EQ(op.totals(), expected);
        EXPECT_EQ(result.owned_flux.at(0).size(), op.flux.owned_face_ids().size());
        for (const auto& packet : result.owned_flux.at(0))
            EXPECT_EQ(rates(packet), Stock{});
    }
}

TEST(LowMachQbmmLinearTransportTest, ZeroFluxPreservesSignedSubnormalEnergy)
{
    // A one-cell, closed operator has no EOS face evaluation. Its stock
    // certificate still has to retain the sign of U, even when U/M cannot
    // be stored by a binary64 EOS specific-energy field.
    constexpr double least = std::numeric_limits<double>::denorm_min();
    for (const double signed_energy : {-least, least})
    {
        Problem op(mesh({0., 1.}));
        const auto expected = stock_from_atoms(2., 0., signed_energy);
        EXPECT_EQ(expected[energy] / expected[mass], 0.);
        for (size_t row = 0; row < op.state.cells.size(); ++row)
            op.set(static_cast<LO>(row), expected);
        op.flux.sync_ghosts();
        const auto result = op.advance(.125);
        EXPECT_EQ(op.totals(), expected);
        for (size_t row = 0; row < op.state.cells.size(); ++row)
            EXPECT_EQ(op.get(static_cast<LO>(row))[energy], signed_energy);
        for (const auto& packet : result.owned_flux.at(0))
            EXPECT_EQ(rates(packet), Stock{});
    }
}

TEST(LowMachQbmmLinearTransportTest, BothSignedSpecificEnergyBoundsConstrainTheOutgoingStageTrace)
{
    for (const double sign : {-1., 1.})
    {
        Problem op(mesh({0., 1., 2., 3.}));
        constexpr std::array energy_profile{-4., -2., -3.};
        for (size_t row = 0; row < op.state.cells.size(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto id = op.grid->cell_geometry_global_id(cell);
            op.set(cell, stock_from_atoms(2., 1., sign * energy_profile.at(id)));
        }
        for (const auto face : op.flux.owned_face_ids())
            if (op.grid->face_centroid(face).x == 2.)
                op.flux.set_owned_value(face, .25 * op.grid->face_normal(face).x);
        op.flux.sync_ghosts();
        const auto before = op.totals();

        // The central donor sits at the upper bound for negative energy
        // and at the lower bound after reflection. The unlimited slope
        // crosses that bound at each stage. The limited trace retains
        // e=-1/2 (or its reflection) and the outgoing masses are 1 and 3/4.
        const auto result = op.advance(1.);
        EXPECT_NEAR(op.totals()[energy], before[energy], 2e-13);
        for (const auto face : op.flux.owned_face_ids())
            if (op.grid->face_centroid(face).x == 2.)
            {
                const auto& receipt = result.owned_flux.at(0).at(op.flux.owned_row(face));
                const double orientation = op.grid->face_normal(face).x;
                EXPECT_NEAR(orientation * receipt.internal_energy_flux, -.4375 * sign, 2e-13);
                EXPECT_NEAR(orientation * receipt.derived_mass_flux, .875, 2e-13);
            }
    }
}

TEST(LowMachQbmmLinearTransportTest, EqualSpecificEnergyBoundsAcceptRoundedZeroSlopeRetention)
{
    Problem op(mesh({0., 1., 2.}));
    const auto family = std::make_shared<const R::QbmmPopulationSchema>(op.species, op.species->fingerprint(),
        "linear advection bubbles", op.options.phase_roles[0], 1.,
        std::vector<R::QbmmMassFraction>{{47, 1.}, {48, 0.}});
    op.options.qbmm_families = {family};
    op.state.qbmm_families = {Flow::qbmm_family_stamp(*family)};
    constexpr double mass_per_cell = 1.1, energy_per_cell = -.2;
    constexpr Stock initial{mass_per_cell, mass_per_cell, mass_per_cell, mass_per_cell, mass_per_cell, 0.,
        energy_per_cell, (2. * mass_per_cell + .5 * energy_per_cell) / pressure};
    for (size_t row = 0; row < op.state.cells.size(); ++row)
        op.set(static_cast<LO>(row), initial);
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.flux.set_owned_value(face, .25);
    op.flux.sync_ghosts();
    const auto before = op.totals();

    // The first stage has no slope. Independent representation of its
    // retained U and material mass shifts the ratio by a fraction of one
    // ulp despite exact equality of both input stencil bounds.
    const auto result = op.advance(1.);
    check_stock(op.totals(), before, 2e-15);
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
        {
            const auto& receipt = result.owned_flux.at(0).at(op.flux.owned_row(face));
            EXPECT_NEAR(receipt.internal_energy_flux, .25 * energy_per_cell, 2e-15);
            EXPECT_NEAR(receipt.derived_mass_flux, .25 * mass_per_cell, 2e-15);
        }
}

TEST(LowMachQbmmLinearTransportTest, ZeroRatePreservesCanonicalMonodisperseStocks)
{
    Problem op(mesh({0., 7.}));
    const auto family = std::make_shared<const R::QbmmPopulationSchema>(op.species, op.species->fingerprint(),
        "linear advection bubbles", op.options.phase_roles[0], 1.,
        std::vector<R::QbmmMassFraction>{{47, 1.}, {48, 0.}});
    op.options.qbmm_families = {family};
    op.state.qbmm_families = {Flow::qbmm_family_stamp(*family)};
    record_ownership(*op.grid, 1);

    // This singleton's canonical extensive block is realizable. Dividing
    // its entries independently by the bulk volume rounds it off the cone.
    constexpr R::QbmmMomentBlock moments{.3, .8999999999999999, 2.6999999999999997, 8.1};
    static constexpr Stock expected{
        moments[0], moments[1], moments[2], moments[3], moments[1], 0., 0., 2. * moments[1] / pressure};
    EXPECT_EQ(R::invert_qbmm_moments(moments).status, R::InversionStatus::RankOne);
    for (size_t row = 0; row < op.state.cells.size(); ++row)
        op.set(static_cast<LO>(row), expected);
    op.flux.sync_ghosts();
    const auto result = op.advance(.125);
    EXPECT_EQ(result.maximum_cfl, 0.);
    EXPECT_EQ(op.totals(), expected);
    for (const auto& packet : result.owned_flux.at(0))
        EXPECT_EQ(rates(packet), Stock{});
}

TEST(LowMachQbmmLinearTransportTest, OutgoingSingletonKeepsItsCertifiedPositiveTraceAcrossBulkScaling)
{
    Problem op(mesh({0., 7., 14., 21.}));
    const auto family = std::make_shared<const R::QbmmPopulationSchema>(op.species, op.species->fingerprint(),
        "linear advection bubbles", op.options.phase_roles[0], 1.,
        std::vector<R::QbmmMassFraction>{{47, 1.}, {48, 0.}});
    op.options.qbmm_families = {family};
    op.state.qbmm_families = {Flow::qbmm_family_stamp(*family)};

    constexpr R::QbmmMomentBlock moments{.3, .8999999999999999, 2.6999999999999997, 8.1};
    EXPECT_EQ(R::invert_qbmm_moments(moments).status, R::InversionStatus::RankOne);
    EXPECT_FALSE(R::invert_qbmm_moments({moments[0] / 7., moments[1] / 7., moments[2] / 7., moments[3] / 7.}).valid());
    constexpr Stock singleton{
        moments[0], moments[1], moments[2], moments[3], moments[1], 0., 0., 2. * moments[1] / pressure};
    constexpr Stock two_nodes{2., 3., 5., 9., 3., 0., 0., 6. / pressure};
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        op.set(cell, op.grid->cell_geometry_global_id(cell) == 1 ? singleton : two_nodes);
    }
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->face_centroid(face).x == 14.)
            op.flux.set_owned_value(face, op.grid->face_normal(face).x);
    op.flux.sync_ghosts();
    const auto before = op.totals();

    // The symmetric neighbors give zero initial slope. FE1 adds c times
    // the singleton to the right cell and removes it from the donor.
    // The next face trace is the remaining donor plus one quarter of
    // that shipment from the centered gradient on the two neighbors.
    constexpr double c = 1. / 7., integrated_factor = c * (1. - .375 * c);
    auto expected = singleton;
    for (auto& amount : expected)
        amount *= integrated_factor;
    const auto result = op.advance(1.);
    EXPECT_EQ(result.substeps, 1);
    EXPECT_DOUBLE_EQ(result.maximum_cfl, c);
    check_stock(op.totals(), before, 2e-13);
    for (const auto face : op.flux.owned_face_ids())
    {
        auto actual = rates(result.owned_flux[0][op.flux.owned_row(face)]);
        if (op.grid->face_centroid(face).x == 14.)
        {
            for (auto& rate : actual)
                rate *= op.grid->face_normal(face).x;
            check_stock(actual, expected, 2e-13);
        }
        else
            EXPECT_EQ(actual, Stock{});
    }
}

TEST(LowMachQbmmLinearTransportTest, AlignedSubcyclesKeepTheIntegratedTwoCellReceipt)
{
    Problem op(mesh({0., 1., 2.}));
    op.options.qbmm_model.linear_cfl = .25;
    op.options.qbmm_model.maximum_linear_substeps = 2;
    const Vec left_velocity{.125, -.25, .5};
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const bool left = op.grid->cell_geometry_global_id(cell) == 0;
        op.set(cell, stock_from_weights(left ? Weights{4., 2.} : Weights{2., 1.}));
        op.state.cells[cell].velocity = left ? left_velocity : Vec{.75, .25, -.5};
    }
    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.flux.set_owned_value(face, .5);
    op.flux.sync_ghosts();

    // Every stage sees the same midpoint trace (3, 1.5) because the adjacent
    // weight sums are invariant. It transfers (.75, .375) per half interval.
    const auto result = op.advance(1.);
    EXPECT_EQ(result.substeps, 2);
    EXPECT_EQ(result.substep_durations, (std::vector<double>{.5, .5}));
    EXPECT_DOUBLE_EQ(result.maximum_cfl, .25);
    EXPECT_LE(result.maximum_stage_residual, 2e-13);
    EXPECT_LE(result.maximum_step_residual, 2e-13);
    for (const auto face : op.flux.owned_face_ids())
    {
        const auto& packet = result.owned_flux.at(0).at(op.flux.owned_row(face));
        check_stock(rates(packet), op.grid->is_interior_face(face) ? stock_from_weights({1.5, .75}) : Stock{}, 2e-13);
        if (op.grid->is_interior_face(face))
        {
            EXPECT_TRUE(packet.kinetic.has_value());
            if (packet.kinetic)
            {
                EXPECT_DOUBLE_EQ(packet.kinetic->forward.integrated_mass, 3.);
                EXPECT_EQ(packet.kinetic->forward.carrier_momentum_flux, left_velocity * 3.);
                EXPECT_EQ(packet.kinetic->backward.carrier_momentum_flux, Vec{});
            }
        }
    }
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const bool left = op.grid->cell_geometry_global_id(cell) == 0;
        check_stock(op.get(cell), stock_from_weights(left ? Weights{2.5, 1.25} : Weights{3.5, 1.75}), 2e-13);
    }
    const auto before = op.totals();
    op.options.qbmm_model.maximum_linear_substeps = 1;
    EXPECT_THROW((void) op.advance(1.), std::exception);
    EXPECT_EQ(op.totals(), before);

    for (const auto face : op.flux.owned_face_ids())
        if (op.grid->is_interior_face(face))
            op.flux.set_owned_value(face, 1.);
    op.flux.sync_ghosts();
    op.options.qbmm_model.linear_cfl = std::numeric_limits<double>::denorm_min();
    try
    {
        (void) op.advance(1.);
        ADD_FAILURE() << "Expected rejection before a stage without representable progress.";
    }
    catch (const std::exception& error)
    {
        EXPECT_STREQ(error.what(), "QBMM linear subcycle has no representable time progress.");
    }
    EXPECT_EQ(op.totals(), before);
}

TEST(LowMachQbmmLinearTransportTest, TiedPressureTraceUsesThePhysicalFaceOnUnequalCells)
{
    Problem op(mesh({0., 1., 4.}));
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const double x = op.grid->cell_centroid(cell).x, bulk = op.grid->cell_volume(cell);
        op.set(cell, stock_from_weights({bulk * (.5 + .5 * x), bulk * (.25 + .25 * x)}));
    }
    op.flux.sync_ghosts();
    const auto initial = op.totals();
    const auto result = op.advance(.5);
    EXPECT_EQ(result.maximum_cfl, 0.);
    EXPECT_EQ(op.totals(), initial);

    int local_ties = 0;
    for (const auto face : op.flux.owned_face_ids())
    {
        const auto row = op.flux.owned_row(face);
        const auto& packet = result.owned_flux.at(0).at(row);
        EXPECT_EQ(rates(packet), Stock{});
        if (op.grid->is_interior_face(face))
        {
            ++local_ties;
            // At x=1 both reconstructed traces have weights (1, 1/2).
            // The center traces average to the affine profile at x=1.5 instead.
            constexpr double theta = (1.25 * 2. - .125 * 2.) / pressure;
            EXPECT_TRUE(packet.kinetic.has_value());
            if (packet.kinetic)
            {
                EXPECT_NEAR(packet.kinetic->theta, theta, 2e-15);
                EXPECT_EQ(packet.kinetic->slip_volume_flux, 0.);
            }
            EXPECT_NEAR(result.integrated_faces.at(row).theta.rounded(.5), theta, 2e-15);
        }
    }
    EXPECT_EQ(reduce(*op.grid, Teuchos::REDUCE_SUM, local_ties), 1);
}

void check_stationary_native_step(const SP<Flow::Mesh>& grid)
{
    Problem op(grid);
    for (size_t row = 0; row < op.state.cells.size(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const double volume = grid->cell_volume(cell);
        // At p=8 and T=2 the ideal gas occupies exactly its own bulk volume.
        op.set(cell, stock_from_atoms(4.4 * volume, volume, 0.));
    }
    Flow flow(grid, {op.gas}, op.options);
    flow.initialize(op.state);
    constexpr double dt = .005;
    const auto trial = flow.preview(dt);
    const auto report = flow.qbmm_report(trial);
    EXPECT_EQ(report.family_transport, Model::FamilyTransport::RealizableLinearSSPRK2);
    EXPECT_EQ(report.linear_substeps, 1);
    EXPECT_EQ(report.linear_stages, 2);
    EXPECT_EQ(report.maximum_linear_cfl, 0.);
    EXPECT_TRUE(report.interval);
    for (const auto& face : trial.owned_phase_transport())
    {
        EXPECT_EQ(face.qbmm_flux.size(), 1);
        if (face.qbmm_flux.size() == 1)
        {
            EXPECT_TRUE(face.qbmm_flux[0]);
            if (face.qbmm_flux[0])
                EXPECT_EQ(rates(*face.qbmm_flux[0]), Stock{});
        }
    }
    flow.commit(trial);
    EXPECT_DOUBLE_EQ(flow.state().time, dt);
    EXPECT_EQ(flow.qbmm_report().linear_stages, 2);
    for (size_t row = 0; row < flow.state().cells.size(); ++row)
    {
        const auto& actual = flow.state().cells[row];
        const auto& initial = op.state.cells[row];
        EXPECT_EQ(actual.phase_numbers, initial.phase_numbers);
        EXPECT_EQ(actual.qbmm_shape, initial.qbmm_shape);
        EXPECT_EQ(actual.phases[0].moles, initial.phases[0].moles);
    }
}

TEST(LowMachQbmmLinearTransportTest, NativeGeometryAdmissionAndStationaryCommit)
{
    for (const auto kind : mesh_families)
    {
        SCOPED_TRACE(family_names[static_cast<size_t>(kind)]);
        check_stationary_native_step(Plane(kind, 4).mesh());
    }
    const auto annulus = std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cylindrical>(
        Vec3D<ArrReal>{ArrReal{1., 1.4, 2.}, edges(8, 0., 2. * std::numbers::pi), ArrReal{0., 1.}}));
    check_stationary_native_step(annulus);

    // Equal sectors meet at the axis and mirror their centroids across each radial face.
    using Semi = Meshes::SemiStructuredXY_Z;
    constexpr unsigned sectors = 8;
    Arr<Semi::Vec3> disk_nodes{{0., 0., 0.}};
    Arr<Arr<unsigned>> disk_cells;
    for (unsigned sector = 0; sector < sectors; ++sector)
    {
        const double theta = 2. * std::numbers::pi * sector / sectors;
        disk_nodes.push_back({std::cos(theta), std::sin(theta), 0.});
        disk_cells.push_back({0, sector + 1, (sector + 1) % sectors + 1});
    }
    const auto disk = std::make_shared<Flow::Mesh>(std::make_shared<Semi>(disk_nodes, disk_cells, ArrReal{0., 1.}));
    check_stationary_native_step(disk);

    Plane skewed(Geometry::PolarExtrusion, 3);
    skewed.b[1] += .06;
    Problem unsupported(skewed.mesh());
    EXPECT_THROW(Flow(unsupported.grid, {unsupported.gas}, unsupported.options), std::exception);
}
} // namespace
