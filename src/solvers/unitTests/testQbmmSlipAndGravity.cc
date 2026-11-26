/** Q6 supplied-slip and fixed-grid gravity qualifications. Synthetic SI cases, not empirical slip validation. */
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using LO = Flow::Pack::local_ordinal_type;
using Velocity = R::QbmmModelOptions::Velocity;
static_assert(std::is_same_v<decltype(Flow::PhaseFaceTransport::momentum_mass_flux), double>);
static_assert(std::is_same_v<decltype(Flow::PhaseFaceTransport::drift_momentum_flux), vec3<double>>);

testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr double gas_constant = 8.31446261815324, temperature = 300, pressure = 100000;
constexpr double liquid_density = 1000, gas_energy = 6000, dt = 1. / 128;
constexpr std::array<double, 2> molar_mass{.02, .04};
constexpr size_t cells = 4, channels = 7; // nA, nB, H0, species-derived H1, H2, H3, phase U.

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void record(const std::string& name, double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    testing::Test::RecordProperty(name, out.str());
}

double mass(const R::PhaseThermalState& phase)
{
    return std::inner_product(phase.moles.begin(), phase.moles.end(), molar_mass.begin(), 0.);
}

double mass(const Flow::CellState& cell)
{
    double total = 0;
    for (const auto& phase : cell.phases)
        total += mass(phase);
    return total;
}

enum class Layout
{
    Loop,
    Column,
    EmptyColumn
};

struct Specification
{
    Layout layout = Layout::Loop;
    bool polydisperse = true;
    Velocity mode = Velocity::SuppliedPhaseSlip;
    bool phase_slip = true, carrier_motion = false;
    bool stratified = false, single_phase_gas = false;
    std::optional<double> hydrostatic_acceleration;
    bool planar_motion = false;
};

struct Fixture
{
    Specification spec;
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos;
    Flow::Options options;
    std::shared_ptr<Flow> flow;
    int source_calls = 0;

    explicit Fixture(Specification input = {}) : spec(input)
    {
        const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
        Flow::Mesh::DistributionOptions partition;
        partition.allow_empty_partitions = true;
        // Explicit custom tensor partition: the only X cell belongs to rank zero.
        // Four Z cells still exercise a real column; MPI2 AND MPI4 have size-1 empty owners.
        if (spec.layout == Layout::EmptyColumn)
            partition.orthogonal_partitions = std::array<size_t, 3>{ranks, 1, 1};
        else if (spec.layout == Layout::Column)
            partition.orthogonal_partitions = std::array<size_t, 3>{1, 1, ranks};
        else
            partition.orthogonal_partitions =
                ranks == 4 ? std::array<size_t, 3>{2, 1, 2} : std::array<size_t, 3>{ranks, 1, 1};
        const ArrReal x = spec.layout == Layout::Loop ? ArrReal{0, .5, 1} : ArrReal{0, 1};
        const ArrReal z = spec.layout == Layout::Loop ? ArrReal{0, .5, 1} : ArrReal{0, .25, .5, .75, 1};
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{x, ArrReal{0, 1}, z}), partition);
        const std::vector phases{R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed, R::SpeciesPhase::Resolved};
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {11, "A", molar_mass[0], {{"A", 1}}, phases}, {29, "B", molar_mass[1], {{"B", 1}}, phases}});
        const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
        auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, pressure, liquid_density, 4000, 0, 0, 0, 0, 0}; });
        auto gas = R::make_ideal_gas_mixture(catalog, {{20, 300, gas_energy}, {20, 300, gas_energy}}, bounds);
        eos = spec.polydisperse ? std::vector{liquid, gas, gas} : std::vector{spec.single_phase_gas ? gas : liquid};
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.qbmm_model.velocity = spec.mode;
        options.phase_roles = {{spec.single_phase_gas ? Flow::PhaseRole::BulkGas : Flow::PhaseRole::BulkLiquid,
            spec.single_phase_gas ? "gas" : "liquid"}};
        if (spec.polydisperse)
        {
            options.phase_roles.push_back({Flow::PhaseRole::DispersedGas, "fine"});
            options.phase_roles.push_back({Flow::PhaseRole::DispersedGas, "coarse"});
            options.qbmm_families = {nullptr,
                std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "fine",
                    options.phase_roles[1], .001, std::vector<R::QbmmMassFraction>{{11, 1}, {29, 0}}),
                std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "coarse",
                    options.phase_roles[2], .003, std::vector<R::QbmmMassFraction>{{11, 0}, {29, 1}})};
        }
        flow = make_flow(options);
        flow->initialize(initial());
    }

    std::shared_ptr<Flow> make_flow(const Flow::Options& controls) const
    {
        return std::make_shared<Flow>(mesh, eos, controls);
    }

    double sum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    }

    double maximum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
        return global;
    }

    std::vector<double> sum(const std::vector<double>& local) const
    {
        std::vector<double> global(local.size());
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local.size()),
            local.data(), global.data());
        return global;
    }

    size_t index(LO cell) const
    {
        const auto point = mesh->cell_centroid(cell);
        return spec.layout == Layout::Loop ? (point.x > .5 ? 1 : 0) + (point.z > .5 ? 2 : 0)
                                           : static_cast<size_t>(4 * point.z);
    }

    Flow::Vector slip(size_t phase, LO cell) const
    {
        if (!spec.phase_slip || phase == 0)
            return {};
        const double sign = phase == 1 ? 1 : -1;
        const auto point = mesh->cell_centroid(cell);
        if (spec.planar_motion && (point.z < .25 || point.z > .75))
            return {};
        return spec.layout == Layout::Loop ? Flow::Vector{sign * (.5 - point.z), 0, sign * (point.x < .5 ? -.25 : .25)}
                                           : Flow::Vector{0, 0, sign * .125};
    }

    double column_density() const { return spec.polydisperse ? 750 : liquid_density; }

    std::array<double, cells> column_pressures(double gravity) const
    {
        // Independent finite-volume hydrostatics: p[j+1]-p[j] = g*h*(rho[j]+rho[j+1])/2.
        // For rho=a+b*p this recurrence is affine in the unknown bottom pressure.
        // A pure isothermal gas reduces to p[j+1]=p[j]*(1-beta)/(1+beta).
        std::array<long double, cells> a{}, b{}, slope{}, intercept{};
        for (size_t j = 0; j < cells; ++j)
        {
            const long double alpha = .1L * (j + 1);
            a[j] = spec.single_phase_gas ? 0 : spec.stratified ? (1 - alpha) * liquid_density : column_density();
            b[j] = spec.single_phase_gas ? molar_mass[0] / (gas_constant * temperature)
                   : spec.stratified     ? alpha * .03L / (gas_constant * temperature)
                                         : 0;
        }
        const long double half_g_h = gravity / 8.;
        slope[0] = 1;
        for (size_t j = 0; j + 1 < cells; ++j)
        {
            const auto denominator = 1 - half_g_h * b[j + 1];
            slope[j + 1] = slope[j] * (1 + half_g_h * b[j]) / denominator;
            intercept[j + 1] = (intercept[j] * (1 + half_g_h * b[j]) + half_g_h * (a[j] + a[j + 1])) / denominator;
        }
        const auto bottom = (cells * pressure - std::accumulate(intercept.begin(), intercept.end(), 0.L)) /
                            std::accumulate(slope.begin(), slope.end(), 0.L);
        std::array<double, cells> result{};
        for (size_t j = 0; j < cells; ++j)
            result[j] = static_cast<double>(slope[j] * bottom + intercept[j]);
        return result;
    }

    Flow::State initial() const
    {
        Flow::State state;
        state.thermodynamic_pressure = pressure;
        if (spec.polydisperse)
            state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*options.qbmm_families[1]),
                Flow::qbmm_family_stamp(*options.qbmm_families[2])};
        state.cells.resize(mesh->num_owned_cells());
        const auto column_p = column_pressures(spec.hydrostatic_acceleration.value_or(0));
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const LO cell_id = static_cast<LO>(c);
            const auto point = mesh->cell_centroid(cell_id);
            const double volume = mesh->cell_volume(cell_id);
            auto& cell = state.cells[c];
            const double p = spec.layout == Layout::Loop ? pressure : column_p[index(cell_id)];
            cell.mechanical_pressure = p - pressure;
            cell.phases.resize(eos.size());
            cell.phase_numbers.resize(eos.size());
            cell.drift_velocity.resize(eos.size());
            if (spec.polydisperse)
                cell.qbmm_shape.resize(eos.size());
            double gas_mass = 0, gas_volume = 0;
            for (size_t a = 1; a < eos.size(); ++a)
            {
                double n = 0;
                if (spec.layout == Layout::Loop)
                {
                    const double alpha = a == 1 ? (point.x < .5 ? .32 : .08) : (point.x < .5 ? .08 : .24);
                    n = alpha * volume * p / (gas_constant * temperature);
                }
                else
                {
                    // Equal gas mole amounts give Wbar=.03 kg/mol. Solve occupancy AND
                    // constant bulk density analytically, rather than pretending gas volume is fixed.
                    const double total = spec.stratified
                                             ? .1 * (index(cell_id) + 1) * volume * p / (gas_constant * temperature)
                                             : volume * (liquid_density - column_density()) /
                                                   (liquid_density * gas_constant * temperature / p - .03);
                    n = total / 2;
                }
                std::vector<double> moles(2);
                moles[a - 1] = n;
                cell.phases[a] = {eos[a], moles, p, gas_energy * n, temperature};
                const double phase_mass = molar_mass[a - 1] * n;
                gas_mass += phase_mass;
                gas_volume += n * gas_constant * temperature / p;
                // Independent two-atom measure: weights 2w,3w at x*,4x*.
                const double w = phase_mass / (14 * options.qbmm_families[a]->mass_scale_kg());
                cell.phase_numbers[a] = 5 * w;
                cell.qbmm_shape[a] = R::QbmmShapeState{50 * w, 194 * w};
                cell.drift_velocity[a] = slip(a, cell_id);
            }
            const double liquid_mass = spec.layout == Layout::Loop || spec.stratified
                                           ? liquid_density * (volume - gas_volume)
                                           : column_density() * volume - gas_mass;
            if (spec.single_phase_gas)
            {
                const double n = p * volume / (gas_constant * temperature);
                cell.phases[0] = {eos[0], {n, 0}, p, gas_energy * n, temperature};
            }
            else
                cell.phases[0] = {eos[0], {liquid_mass / molar_mass[0], 0}, p, 0, temperature};
            if (spec.carrier_motion)
                cell.velocity = Flow::Vector{(.5 - point.z) / 8, 0, (point.x < .5 ? -.25 : .25) / 8};
        }
        return state;
    }

    R::ReactingMixtureOptions controller_options(std::optional<Flow::Vector> acceleration = std::nullopt)
    {
        R::ReactingMixtureOptions controls;
        controls.mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, eos[0]->energy_convention());
        controls.phases = {{spec.single_phase_gas ? R::SpeciesPhase::Resolved : R::SpeciesPhase::Liquid, {}, {}}};
        if (spec.polydisperse)
        {
            controls.phases.push_back({R::SpeciesPhase::Dispersed, "fine", {}});
            controls.phases.push_back({R::SpeciesPhase::Dispersed, "coarse", {}});
        }
        controls.source_identity = "Q6 no chemistry";
        controls.source = [this](const auto&, double)
        {
            ++source_calls;
            return R::ReactionSourceInput{};
        };
        controls.body_acceleration = acceleration;
        return controls;
    }

    void check_ownership() const
    {
        const auto ranks = mesh->owned_cell_map()->getComm()->getSize();
        const double empty = sum(mesh->num_owned_cells() == 0 ? 1 : 0);
        EXPECT_EQ(sum(static_cast<double>(mesh->num_owned_cells())), cells);
        if (spec.layout == Layout::EmptyColumn)
        {
            EXPECT_EQ(empty, ranks - 1);
            EXPECT_EQ(mesh->num_owned_cells(), mesh->owned_cell_map()->getComm()->getRank() == 0 ? cells : 0);
        }
        else if (ranks == 2 || ranks == 4)
            EXPECT_EQ(empty, 0);
        record("mpi_ranks", ranks);
        record("empty_owners", empty);
    }
};

/** Capture public native fields including overlap values, not just the canonical State. */
struct Endpoint
{
    Flow::State state;
    std::vector<double> rho, flux;
    std::vector<Flow::Vector> velocity;

    explicit Endpoint(const Flow& flow) : state(flow.state())
    {
        for (size_t c = 0; c < flow.mesh()->num_local_cells(); ++c)
        {
            rho.push_back(flow.density().local_value(static_cast<LO>(c)));
            velocity.push_back(flow.velocity().local_value(static_cast<LO>(c)));
        }
        for (size_t f = 0; f < flow.mesh()->num_faces(); ++f)
            flux.push_back(flow.volume_flux().local_value(static_cast<LO>(f)));
    }

    void expect_same(const Flow& flow) const
    {
        const auto& other = flow.state();
        EXPECT_EQ(state.time, other.time);
        EXPECT_EQ(state.revision, other.revision);
        EXPECT_EQ(state.thermodynamic_pressure, other.thermodynamic_pressure);
        EXPECT_EQ(state.qbmm_families, other.qbmm_families);
        EXPECT_EQ(state.cells.size(), other.cells.size());
        for (size_t c = 0; c < std::min(state.cells.size(), other.cells.size()); ++c)
        {
            const auto& a = state.cells[c];
            const auto& b = other.cells[c];
            EXPECT_EQ(a.mechanical_pressure, b.mechanical_pressure);
            EXPECT_EQ(a.phase_numbers, b.phase_numbers);
            EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
            EXPECT_EQ(a.phases.size(), b.phases.size());
            EXPECT_EQ(a.drift_velocity.size(), b.drift_velocity.size());
            for (size_t axis = 0; axis < 3; ++axis)
            {
                EXPECT_EQ(a.velocity.component(axis), b.velocity.component(axis));
                for (size_t p = 0; p < std::min(a.drift_velocity.size(), b.drift_velocity.size()); ++p)
                    EXPECT_EQ(a.drift_velocity[p].component(axis), b.drift_velocity[p].component(axis));
            }
            for (size_t p = 0; p < std::min(a.phases.size(), b.phases.size()); ++p)
            {
                EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
                EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
                EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
                EXPECT_EQ(a.phases[p].absolute_pressure, b.phases[p].absolute_pressure);
            }
        }
        EXPECT_EQ(rho.size(), flow.mesh()->num_local_cells());
        EXPECT_EQ(flux.size(), flow.mesh()->num_faces());
        for (size_t c = 0; c < std::min(rho.size(), flow.mesh()->num_local_cells()); ++c)
        {
            EXPECT_EQ(rho[c], flow.density().local_value(static_cast<LO>(c)));
            const auto now = flow.velocity().local_value(static_cast<LO>(c));
            for (size_t axis = 0; axis < 3; ++axis)
                EXPECT_EQ(velocity[c].component(axis), now.component(axis));
        }
        for (size_t f = 0; f < std::min(flux.size(), flow.mesh()->num_faces()); ++f)
            EXPECT_EQ(flux[f], flow.volume_flux().local_value(static_cast<LO>(f)));
    }
};

std::array<double, channels> values(
    const std::vector<double>& moles, double number, const R::QbmmShapeState& shape, double energy, double scale)
{
    return {moles[0], moles[1], number, (molar_mass[0] * moles[0] + molar_mass[1] * moles[1]) / scale, shape.H2,
        shape.H3, energy};
}

// Independent dense long-double elimination of the four-cell backward-Euler equations.
std::vector<long double> solve(std::vector<long double> matrix, std::vector<long double> rhs)
{
    const auto n = rhs.size();
    for (size_t p = 0; p < n; ++p)
    {
        require(matrix[p * n + p] > 0, "Independent transport oracle has no positive pivot.");
        const auto diagonal = matrix[p * n + p];
        for (size_t j = p; j < n; ++j)
            matrix[p * n + j] /= diagonal;
        rhs[p] /= diagonal;
        for (size_t i = 0; i < n; ++i)
            if (i != p)
            {
                const auto multiplier = matrix[i * n + p];
                for (size_t j = p; j < n; ++j)
                    matrix[i * n + j] -= multiplier * matrix[p * n + j];
                rhs[i] -= multiplier * rhs[p];
            }
    }
    return rhs;
}

void near_relative(double actual, double expected, double tolerance = 2e-10)
{
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_LE(std::abs(actual - expected), tolerance * std::abs(expected)); // Exact zero stays exact zero.
}

void audit_transport(const Fixture& f, const Flow::Trial& trial, bool require_partition_counterflow)
{
    const auto& before = f.flow->state();
    const auto& snapshot = trial.transport_snapshot();
    const auto receipts = trial.owned_phase_transport();
    collective_detail::collective_local_validation(*f.mesh, "Q6 oracle layout",
        [&]
        {
            require(snapshot.cells.size() == before.cells.size() && receipts.size() == f.mesh->num_owned_faces(),
                "Q6 transport snapshot or face layout differs.");
            for (size_t c = 0; c < before.cells.size(); ++c)
                require(snapshot.cells[c].phases.size() == 3, "Q6 transport phase layout differs.");
            for (const auto& receipt : receipts)
                require(receipt.qbmm_flux.size() == 3 && receipt.qbmm_flux[1] && receipt.qbmm_flux[2] &&
                            receipt.relative_phase_volume_flux.size() == 3 && receipt.material_fraction.size() == 3,
                    "Q6 phase flux layout differs.");
        });
    double crossing = 0, counterflow = 0, fraction_departure = 0, maximum_oracle_error = 0;
    for (size_t p = 1; p < 3; ++p)
    {
        const auto scale = f.options.qbmm_families[p]->mass_scale_kg();
        std::vector<double> old_local(cells * channels), now_local(cells * channels), q_local(cells * cells);
        std::vector<double> packets_local(cells * cells * channels), volumes_local(cells);
        std::vector<double> old_volume_local(cells), adv_volume_local(cells), volume_flux_local(cells * cells);
        for (size_t c = 0; c < before.cells.size(); ++c)
        {
            const auto i = f.index(static_cast<LO>(c));
            const auto& a = before.cells[c];
            const auto& b = snapshot.cells[c].phases[p];
            const auto first =
                values(a.phases[p].moles, *a.phase_numbers[p], *a.qbmm_shape[p], a.phases[p].internal_energy, scale);
            const auto last = values(b.moles, *b.population_number, *b.qbmm_shape, b.internal_energy, scale);
            std::copy(first.begin(), first.end(), old_local.begin() + i * channels);
            std::copy(last.begin(), last.end(), now_local.begin() + i * channels);
            volumes_local[i] = snapshot.new_control_volumes[c];
            old_volume_local[i] = std::accumulate(a.phases[p].moles.begin(), a.phases[p].moles.end(), 0.) *
                                  gas_constant * a.phases[p].temperature / a.phases[p].absolute_pressure;
            adv_volume_local[i] = b.advected_material_volume;
        }
        for (size_t face = 0; face < receipts.size(); ++face)
        {
            const auto id = static_cast<LO>(face);
            const auto& receipt = receipts[face];
            const auto& packet = *receipt.qbmm_flux[p];
            const auto flux = values(packet.species_molar_flux, packet.population_number_flux, packet.shape_flux,
                packet.internal_energy_flux, scale);
            const auto q = receipt.relative_phase_volume_flux[p];
            if (!f.spec.planar_motion)
                EXPECT_EQ(receipt.mesh_volume_flux, 0);
            EXPECT_DOUBLE_EQ(receipt.phase_volume_flux[p] - receipt.mesh_volume_flux, q);
            near_relative(packet.derived_mass_flux, molar_mass[0] * flux[0] + molar_mass[1] * flux[1]);
            EXPECT_EQ(packet.material_volume_flux, q * receipt.material_fraction[p]);
            if (f.mesh->is_geometry_exterior_face(id))
            {
                EXPECT_EQ(q, 0);
                EXPECT_EQ(packet.material_volume_flux, 0);
                EXPECT_EQ(packet.derived_mass_flux, 0);
                for (const auto value : flux)
                    EXPECT_EQ(value, 0);
                for (size_t axis = 0; axis < 3; ++axis)
                    EXPECT_EQ(packet.derived_drift_momentum_flux.component(axis), 0);
                continue;
            }
            const auto owner = f.mesh->owner_cell(id), neighbor = f.mesh->neighbor_cell(id);
            const auto i = f.index(owner), j = f.index(neighbor);
            const auto donor = q >= 0 ? owner : neighbor;
            const auto slip = (f.slip(p, owner) + f.slip(p, neighbor)) * .5;
            near_relative(
                q, receipt.carrier_volume_flux + slip.dot(f.mesh->face_area_vector(id)) - receipt.mesh_volume_flux);
            const auto donor_slip = f.slip(p, donor);
            for (size_t axis = 0; axis < 3; ++axis)
                near_relative(packet.derived_drift_momentum_flux.component(axis),
                    (molar_mass[0] * flux[0] + molar_mass[1] * flux[1]) * donor_slip.component(axis));
            q_local[i * cells + j] += q;
            volume_flux_local[i * cells + j] += packet.material_volume_flux;
            for (size_t k = 0; k < channels; ++k)
                packets_local[(i * cells + j) * channels + k] += flux[k];
            if (p == 1 && receipt.relative_phase_volume_flux[1] * receipt.relative_phase_volume_flux[2] < 0)
            {
                ++counterflow;
                fraction_departure = std::max(fraction_departure,
                    std::abs(
                        std::accumulate(receipt.material_fraction.begin(), receipt.material_fraction.end(), 0.) - 1));
                if (static_cast<size_t>(owner) >= f.mesh->num_owned_cells() ||
                    static_cast<size_t>(neighbor) >= f.mesh->num_owned_cells())
                    ++crossing;
            }
        }
        const auto old = f.sum(old_local), now = f.sum(now_local), q = f.sum(q_local);
        const auto packets = f.sum(packets_local), volumes = f.sum(volumes_local);
        const auto old_volume = f.sum(old_volume_local), adv_volume = f.sum(adv_volume_local);
        const auto volume_flux = f.sum(volume_flux_local);
        std::vector<long double> matrix(cells * cells);
        for (size_t i = 0; i < cells; ++i)
            matrix[i * cells + i] = 1;
        for (size_t i = 0; i < cells; ++i)
            for (size_t j = 0; j < cells; ++j)
            {
                const auto rate = q[i * cells + j];
                const auto donor = rate >= 0 ? i : j;
                matrix[i * cells + donor] += static_cast<long double>(dt) * rate / volumes[donor];
                matrix[j * cells + donor] -= static_cast<long double>(dt) * rate / volumes[donor];
                for (size_t k = 0; k < channels; ++k)
                    near_relative(
                        packets[(i * cells + j) * channels + k], rate * now[donor * channels + k] / volumes[donor]);
            }
        for (size_t k = 0; k < channels; ++k)
        {
            std::vector<long double> rhs(cells);
            double total_old = 0, total_now = 0;
            for (size_t i = 0; i < cells; ++i)
                rhs[i] = old[i * channels + k];
            const auto expected = solve(matrix, rhs);
            for (size_t i = 0; i < cells; ++i)
            {
                const auto actual = now[i * channels + k];
                near_relative(actual, static_cast<double>(expected[i]));
                if (expected[i] != 0)
                    maximum_oracle_error = std::max(
                        maximum_oracle_error, static_cast<double>(std::abs((actual - expected[i]) / expected[i])));
                total_old += old[i * channels + k];
                total_now += actual;
            }
            near_relative(total_now, total_old);
        }
        for (size_t i = 0; i < cells; ++i)
        {
            double outgoing = 0;
            for (size_t j = 0; j < cells; ++j)
                outgoing += volume_flux[i * cells + j] - volume_flux[j * cells + i];
            near_relative(adv_volume[i], old_volume[i] - dt * outgoing);
        }
    }
    std::vector<double> local_liquid_density(cells);
    for (size_t c = 0; c < snapshot.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto& liquid = snapshot.cells[c].phases[0].moles;
        local_liquid_density[f.index(cell)] =
            std::inner_product(liquid.begin(), liquid.end(), molar_mass.begin(), 0.) / snapshot.new_control_volumes[c];
    }
    const auto liquid_density = f.sum(local_liquid_density);
    for (size_t face = 0; face < receipts.size(); ++face)
    {
        const auto id = static_cast<LO>(face);
        const auto& receipt = receipts[face];
        if (f.mesh->is_geometry_exterior_face(id))
        {
            EXPECT_EQ(receipt.momentum_mass_flux, 0);
            EXPECT_EQ(receipt.drift_momentum_flux, Flow::Vector{});
            continue;
        }
        const auto owner = f.mesh->owner_cell(id), neighbor = f.mesh->neighbor_cell(id);
        const double liquid_rate = receipt.relative_phase_volume_flux[0];
        const auto liquid_donor = liquid_rate >= 0 ? owner : neighbor;
        double expected_mass = liquid_rate * liquid_density[f.index(liquid_donor)];
        double gross_mass = std::abs(expected_mass);
        Flow::Vector expected_drift{}, gross_drift{};
        for (size_t p = 1; p < 3; ++p)
        {
            const auto& species_rate = receipt.qbmm_flux[p]->species_molar_flux;
            const double mass_rate =
                std::inner_product(species_rate.begin(), species_rate.end(), molar_mass.begin(), 0.);
            const auto donor = receipt.relative_phase_volume_flux[p] >= 0 ? owner : neighbor;
            const auto drift_rate = f.slip(p, donor) * mass_rate;
            expected_mass += mass_rate;
            gross_mass += std::abs(mass_rate);
            expected_drift = expected_drift + drift_rate;
            for (size_t axis = 0; axis < 3; ++axis)
                gross_drift.component(axis) += std::abs(drift_rate.component(axis));
        }
        constexpr double roundoff = 16 * std::numeric_limits<double>::epsilon();
        EXPECT_NEAR(receipt.momentum_mass_flux, expected_mass, roundoff * gross_mass);
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(receipt.drift_momentum_flux.component(axis), expected_drift.component(axis),
                roundoff * gross_drift.component(axis));
    }
    counterflow = f.sum(counterflow);
    crossing = f.sum(crossing);
    fraction_departure = f.maximum(fraction_departure);
    if (require_partition_counterflow)
    {
        EXPECT_GT(counterflow, 0);
        EXPECT_GT(fraction_departure, .01); // A normalized donor-fraction implementation must fail.
        if (f.mesh->owned_cell_map()->getComm()->getSize() > 1)
            EXPECT_GT(crossing, 0);
    }
    record("counterflow_faces", counterflow);
    record("partition_counterflow_faces", crossing);
    record("maximum_donor_fraction_sum_departure", fraction_departure);
    record("maximum_transport_oracle_relative_error", maximum_oracle_error);
    EXPECT_LE(trial.diagnostics().maximum_shape_transport_residual, 1);
    EXPECT_LE(trial.diagnostics().maximum_population_transport_residual, 1);
    const auto report = f.flow->qbmm_report(trial);
    for (const auto& cell : report.cells)
        for (size_t p = 1; p < 3; ++p)
            EXPECT_EQ(cell.families[p]->quadrature.status, R::InversionStatus::TwoNodes);
}

TEST(QbmmSlipAndGravity, CounterflowAcrossPartitionsMatchesIndependentTransportAndMomentumPackets)
{
    Fixture f;
    f.check_ownership();
    for (int step = 0; step < 2; ++step)
    {
        Endpoint accepted(*f.flow);
        const auto trial = f.flow->preview(dt);
        accepted.expect_same(*f.flow);
        audit_transport(f, trial, true);
        f.flow->commit(trial);
    }
}

TEST(QbmmSlipAndGravity, MovingPartitionCounterflowClosesBackwardEulerReceiptsWorkAndContinuity)
{
    const Specification spec{.layout = Layout::Column, .stratified = true, .planar_motion = true};
    Fixture f(spec);
    f.check_ownership();
    Flow::PlanarMotion motion(f.mesh);
    const auto audit = [&](const Fixture& fixture, const Flow::Trial& trial)
    {
        audit_transport(fixture, trial, true);
        const auto& snapshot = trial.transport_snapshot();
        const auto& accepted = fixture.flow->state();
        const auto report = fixture.flow->qbmm_report(trial);
        Flow::Traits::scalar_face_type corrected(fixture.mesh, 0., "checked_relative_material_flux");
        double moving_counterflow = 0;
        for (size_t f = 0; f < trial.owned_phase_transport().size(); ++f)
        {
            const auto& face = trial.owned_phase_transport()[f];
            const LO id = static_cast<LO>(f);
            if (fixture.mesh->is_geometry_exterior_face(id))
            {
                EXPECT_DOUBLE_EQ(trial.owned_volume_flux()[f], face.mesh_volume_flux);
                for (const auto relative : face.relative_phase_volume_flux)
                    EXPECT_EQ(relative, 0);
            }
            else if (face.relative_phase_volume_flux[1] * face.relative_phase_volume_flux[2] < 0 &&
                     face.mesh_volume_flux != 0)
            {
                const double theta = std::accumulate(face.material_fraction.begin(), face.material_fraction.end(), 0.);
                if (std::abs(theta - 1) > .01 &&
                    (fixture.mesh->owned_cell_map()->getComm()->getSize() == 1 ||
                        static_cast<size_t>(fixture.mesh->owner_cell(id)) >= fixture.mesh->num_owned_cells() ||
                        static_cast<size_t>(fixture.mesh->neighbor_cell(id)) >= fixture.mesh->num_owned_cells()))
                    ++moving_counterflow;
            }
            double flux = 0;
            for (size_t p = 0; p < face.material_fraction.size(); ++p)
                flux += face.material_fraction[p] * (trial.owned_volume_flux()[f] - face.mesh_volume_flux +
                                                        face.phase_volume_flux[p] - face.carrier_volume_flux);
            corrected.set_owned_value(id, flux);
        }
        corrected.sync_ghosts();
        EXPECT_GT(fixture.sum(moving_counterflow), 0);
        double local_work = 0, local_energy = 0, local_continuity_error = 0, local_drift = 0;
        for (size_t c = 0; c < snapshot.cells.size(); ++c)
        {
            const auto& old = accepted.cells[c];
            const auto& next = trial.state().cells[c];
            const auto volume = [](const R::PhaseThermalState& phase, size_t p)
            {
                return p == 0 ? mass(phase) / liquid_density
                              : std::accumulate(phase.moles.begin(), phase.moles.end(), 0.) * gas_constant *
                                    phase.temperature / phase.absolute_pressure;
            };
            double initial_volume = 0, work_volume = 0, final_volume = 0;
            Flow::Vector drift_momentum{};
            for (size_t p = 0; p < next.phases.size(); ++p)
            {
                const double advected = snapshot.cells[c].phases[p].advected_material_volume;
                const double final = volume(next.phases[p], p);
                initial_volume += volume(old.phases[p], p);
                work_volume += advected;
                final_volume += final;
                local_work -= next.phases[p].absolute_pressure * (final - advected);
                local_energy += next.phases[p].internal_energy - old.phases[p].internal_energy;
                drift_momentum = drift_momentum + next.drift_velocity[p] * mass(next.phases[p]);
            }
            const LO cell = static_cast<LO>(c);
            double divergence = 0;
            for (size_t f = 0; f < fixture.mesh->num_faces(); ++f)
            {
                const LO id = static_cast<LO>(f);
                if (fixture.mesh->owner_cell(id) == cell)
                    divergence += corrected.local_value(id);
                else if (!fixture.mesh->is_geometry_exterior_face(id) && fixture.mesh->neighbor_cell(id) == cell)
                    divergence -= corrected.local_value(id);
            }
            const double v1 = snapshot.new_control_volumes[c];
            EXPECT_NEAR(final_volume, v1, fixture.options.volume_relative_tolerance * v1);
            const double expected_divergence = (final_volume - work_volume + initial_volume - v1) / dt;
            local_continuity_error = std::max(local_continuity_error, std::abs(divergence - expected_divergence));
            EXPECT_NEAR(divergence, expected_divergence, fixture.options.continuity_tolerance);
            const auto barycentric = next.velocity + drift_momentum / mass(next);
            local_drift = std::max(local_drift, std::abs(barycentric.z - next.velocity.z));
            for (size_t axis = 0; axis < 3; ++axis)
            {
                EXPECT_DOUBLE_EQ(report.cells[c].carrier_velocity.component(axis), next.velocity.component(axis));
                EXPECT_NEAR(report.cells[c].barycentric_velocity.component(axis), barycentric.component(axis), 1e-14);
            }
        }
        EXPECT_GT(fixture.maximum(local_drift), 0);
        const double work = fixture.sum(local_work);
        EXPECT_NEAR(trial.diagnostics().pressure_work, work, 1e-8);
        EXPECT_NEAR(fixture.sum(local_energy), work, trial.diagnostics().internal_energy_roundoff_bound + 1e-7);
        record("moving_material_continuity_error_m3_s", fixture.maximum(local_continuity_error));
        record("moving_counterflow_work_error_j", std::abs(trial.diagnostics().pressure_work - work));
    };

    const auto trial = f.flow->preview_ale(dt, motion, 1.00001);
    audit(f, trial);
    f.flow->commit_ale(trial, motion);
    const auto bytes = f.flow->serialize_qbmm_restart(motion);
    Fixture restarted(spec);
    Flow::PlanarMotion fresh_motion(restarted.mesh);
    restarted.flow->restore_qbmm_restart(bytes, fresh_motion);
    Endpoint(*f.flow).expect_same(*restarted.flow);
    const auto second = f.flow->preview_ale(dt, motion, .99999);
    const auto resumed = restarted.flow->preview_ale(dt, fresh_motion, .99999);
    audit(f, second);
    audit(restarted, resumed);
    f.flow->commit_ale(second, motion);
    restarted.flow->commit_ale(resumed, fresh_motion);
    Endpoint(*f.flow).expect_same(*restarted.flow);
}

TEST(QbmmSlipAndGravity, PlanarMotionRejectsWallNormalSlipWithAndWithoutSuppliedGravity)
{
    Fixture f({.layout = Layout::EmptyColumn});
    f.check_ownership();
    const Endpoint accepted(*f.flow);
    for (const Flow::Vector acceleration : {Flow::Vector{}, Flow::Vector{0, 0, -9.81}})
    {
        auto controls = f.controller_options(acceleration);
        int targets = 0;
        controls.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
            R::MixturePlanarMotionProvider{f.mesh, {}, "closed-wall slip test", 1, [&](double, double)
                {
                    ++targets;
                    return 1.01;
                }});
        Coupler controller(f.flow, controls);
        EXPECT_THROW((void) controller.preview(dt), std::exception);
        EXPECT_EQ(targets, 1);
        EXPECT_EQ(f.source_calls, 0);
        accepted.expect_same(*f.flow);
    }
    // The fixed path admits its native phase slip. In the planar path the
    // same axial drift has a nonzero normal component at both closed end walls.
    const auto earlier = f.flow->preview(dt);
    Flow::PlanarMotion motion(f.mesh);
    EXPECT_THROW((void) f.flow->preview_ale(dt, motion, 1.01), std::exception);
    EXPECT_FALSE(motion.has_active_trial());
    accepted.expect_same(*f.flow);
    f.flow->commit(earlier);
    EXPECT_EQ(f.flow->state().time, dt);
}

TEST(QbmmSlipAndGravity, FixedControllerProvenanceRejectsMovingCheckpointAndPortableEncoding)
{
    const std::array<std::optional<Flow::Vector>, 3> accelerations{
        std::nullopt, Flow::Vector{}, Flow::Vector{0, 0, -1}};
    for (const auto& acceleration : accelerations)
    {
        SCOPED_TRACE(acceleration ? acceleration->z : 1.);
        Fixture f({.layout = Layout::EmptyColumn,
            .phase_slip = false,
            .stratified = true,
            .hydrostatic_acceleration = acceleration ? std::optional<double>(acceleration->z) : std::nullopt});
        f.check_ownership();
        Flow::PlanarMotion unrelated(f.mesh);
        const auto reject_moving_capture = [&]
        {
            const Endpoint accepted(*f.flow);
            const auto archive = f.flow->serialize_qbmm_restart();
            const auto epoch = f.mesh->geometry_epoch();
            const int calls = f.source_calls;
            EXPECT_THROW((void) f.flow->checkpoint(unrelated), std::exception);
            EXPECT_THROW((void) f.flow->serialize_qbmm_restart(unrelated), std::exception);
            EXPECT_FALSE(unrelated.has_active_trial());
            EXPECT_EQ(f.mesh->geometry_epoch(), epoch);
            EXPECT_EQ(f.source_calls, calls);
            accepted.expect_same(*f.flow);
            EXPECT_EQ(f.flow->serialize_qbmm_restart(), archive);
        };
        {
            SCOPED_TRACE("attached fixed controller");
            Coupler fixed(f.flow, f.controller_options(acceleration));
            reject_moving_capture();
            const auto earlier = fixed.preview(dt);
            reject_moving_capture();
            fixed.commit(earlier);
            EXPECT_EQ(f.flow->state().time, dt);
            if (acceleration && acceleration->z != 0.)
                EXPECT_LT(fixed.diagnostics().flow.external_impulse.z, 0.);
        }
        {
            SCOPED_TRACE("accepted fixed provenance after controller release");
            EXPECT_TRUE(f.flow->qbmm_report().provenance.controller_configuration_signature);
            reject_moving_capture();
            Coupler fixed(f.flow, f.controller_options(acceleration));
            fixed.advance(dt);
            EXPECT_EQ(f.flow->state().time, 2 * dt);
        }
    }
}

TEST(QbmmSlipAndGravity, ExplicitZeroSlipMatchesSharedCarrierIncludingNativeFields)
{
    Fixture f({.mode = Velocity::SharedCarrier, .phase_slip = false, .carrier_motion = true});
    f.check_ownership();
    auto controls = f.options;
    controls.qbmm_model.velocity = Velocity::SuppliedPhaseSlip;
    auto supplied = f.make_flow(controls);
    supplied->initialize(f.flow->state());
    double transported = 0;
    for (int step = 0; step < 3; ++step)
    {
        const auto shared_trial = f.flow->preview(dt), supplied_trial = supplied->preview(dt);
        const auto a = shared_trial.owned_phase_transport(), b = supplied_trial.owned_phase_transport();
        EXPECT_EQ(a.size(), b.size());
        for (size_t face = 0; face < std::min(a.size(), b.size()); ++face)
        {
            EXPECT_EQ(a[face], b[face]);
            transported += std::abs(a[face].qbmm_flux[1]->population_number_flux);
        }
        f.flow->commit(shared_trial);
        supplied->commit(supplied_trial);
        Endpoint(*f.flow).expect_same(*supplied);
    }
    record("zero_slip_population_turnover_per_s", f.sum(transported));
    EXPECT_GT(f.sum(transported), 0);
}

TEST(QbmmSlipAndGravity, CounterflowCheckpointAndPortableRestartContinueTheSameTrajectory)
{
    Fixture f;
    f.check_ownership();
    f.flow->advance(dt);
    const auto save = f.flow->checkpoint();
    const auto archive = f.flow->serialize_qbmm_restart();
    const Endpoint midpoint(*f.flow);
    f.flow->advance(dt);
    f.flow->advance(dt);
    const Endpoint final(*f.flow);
    f.flow->restore(save);
    midpoint.expect_same(*f.flow);
    f.flow->advance(dt);
    f.flow->advance(dt);
    final.expect_same(*f.flow);
    auto fresh = f.make_flow(f.options);
    fresh->restore_qbmm_restart(archive);
    midpoint.expect_same(*fresh);
    fresh->advance(dt);
    fresh->advance(dt);
    final.expect_same(*fresh);
    record("continued_time_s", fresh->state().time);
    auto incompatible = f.options;
    incompatible.qbmm_model.velocity = Velocity::SharedCarrier;
    auto wrong_mode = f.make_flow(incompatible);
    auto zero_slip = f.initial();
    for (auto& cell : zero_slip.cells)
        std::fill(cell.drift_velocity.begin(), cell.drift_velocity.end(), Flow::Vector{});
    wrong_mode->initialize(std::move(zero_slip));
    const Endpoint rejected(*wrong_mode);
    EXPECT_THROW(wrong_mode->restore_qbmm_restart(archive), std::exception);
    rejected.expect_same(*wrong_mode);
}

TEST(QbmmSlipAndGravity, ActualEmptyOwnersParticipateInSlipTransportCheckpointAndRestart)
{
    Fixture f({.layout = Layout::EmptyColumn});
    f.check_ownership();
    const Endpoint accepted(*f.flow);
    const auto trial = f.flow->preview(dt);
    accepted.expect_same(*f.flow);
    audit_transport(f, trial, false);
    f.flow->commit(trial);
    const auto save = f.flow->checkpoint();
    const auto archive = f.flow->serialize_qbmm_restart();
    f.flow->advance(dt);
    const Endpoint final(*f.flow);
    f.flow->restore(save);
    f.flow->advance(dt);
    final.expect_same(*f.flow);
    auto resumed = f.make_flow(f.options);
    resumed->restore_qbmm_restart(archive);
    resumed->advance(dt);
    final.expect_same(*resumed);
    const auto report = resumed->qbmm_report();
    EXPECT_EQ(report.cells.size(), f.mesh->num_owned_cells());
    for (size_t p = 1; p < 3; ++p)
        EXPECT_EQ(report.families[p]->inversion_counts[static_cast<size_t>(R::InversionStatus::TwoNodes)], cells);
}

TEST(QbmmSlipAndGravity, UnsupportedCapabilitiesAndDivergentSlipModeRejectWithActualEmptyOwners)
{
    Fixture f({.layout = Layout::EmptyColumn, .phase_slip = false});
    f.check_ownership();
    const Endpoint accepted(*f.flow);
    const auto retained = f.flow->preview(dt);
    using Model = R::QbmmModelOptions;
    struct Request
    {
        const char* name;
        void (*select)(Model&);
    };
    const Request requests[]{
        {"node slip with native implicit transport", [](Model& m) { m.velocity = Velocity::NodeDependentSlip; }},
        {"independent momentum", [](Model& m) { m.velocity = Velocity::IndependentPhaseMomentum; }},
        {"BDF2", [](Model& m) { m.transport.time = FVM::ScalarTimeScheme::BDF2; }},
        {"linear upwind", [](Model& m) { m.transport.convection = FVM::ScalarConvectionScheme::BoundedLinearUpwind; }},
        {"variable composition", [](Model& m) { m.variable_composition = true; }},
        {"cross-family coalescence", [](Model& m) { m.cross_family_coalescence = true; }},
        {"bulk-gas conversion", [](Model& m) { m.bulk_gas_conversion = true; }},
        {"capillarity", [](Model& m) { m.capillarity = true; }},
        {"node temperatures", [](Model& m) { m.independent_node_temperatures = true; }}};
    const auto comm = f.mesh->owned_cell_map()->getComm();
    for (const auto& request : requests)
        for (const bool uniform : {true, false})
        {
            SCOPED_TRACE(request.name);
            SCOPED_TRACE(uniform);
            auto controls = f.options;
            if (uniform || comm->getRank() == comm->getSize() - 1)
                request.select(controls.qbmm_model);
            EXPECT_THROW((void) f.make_flow(controls), std::exception);
            accepted.expect_same(*f.flow);
        }
    if (comm->getSize() > 1)
    {
        auto divergent = f.options;
        // Both values are supported individually; only an actual empty owner changes selection.
        if (comm->getRank() == comm->getSize() - 1)
            divergent.qbmm_model.velocity = Velocity::SharedCarrier;
        EXPECT_THROW((void) f.make_flow(divergent), std::exception);
        accepted.expect_same(*f.flow);
    }
    f.flow->commit(retained);
    EXPECT_EQ(f.flow->state().time, dt);
}

void reject_bad_slip(Layout layout)
{
    Fixture f({.layout = layout});
    f.check_ownership();
    const Endpoint accepted(*f.flow);
    const auto retained = f.flow->preview(dt);
    // An owned field value may vary spatially; only nonfinite/undeclared drift is invalid.
    // Rank zero is a real owner in both fixtures. Every other rank must reach the same rejection.
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        auto state = f.flow->state();
        if (f.mesh->owned_cell_map()->getComm()->getRank() == 0)
            state.cells[0].drift_velocity[1].x = invalid;
        EXPECT_THROW(f.flow->initialize(std::move(state)), std::exception);
        accepted.expect_same(*f.flow);
    }
    auto shared_controls = f.options;
    shared_controls.qbmm_model.velocity = Velocity::SharedCarrier;
    auto shared = f.make_flow(shared_controls);
    auto zero = f.initial();
    for (auto& cell : zero.cells)
        std::fill(cell.drift_velocity.begin(), cell.drift_velocity.end(), Flow::Vector{});
    shared->initialize(zero);
    const Endpoint shared_accepted(*shared);
    if (f.mesh->owned_cell_map()->getComm()->getRank() == 0)
        zero.cells[0].drift_velocity[1].z = .125;
    EXPECT_THROW(shared->initialize(std::move(zero)), std::exception);
    shared_accepted.expect_same(*shared);
    f.flow->commit(retained);
    EXPECT_EQ(f.flow->state().time, dt);
}

TEST(QbmmSlipAndGravity, RankLocalInvalidSlipKeepsAcceptedStateFieldsAndEarlierPreview)
{
    reject_bad_slip(Layout::Loop);
}

TEST(QbmmSlipAndGravity, RankLocalInvalidSlipRejectsCollectivelyWithActualEmptyOwners)
{
    reject_bad_slip(Layout::EmptyColumn);
}

std::vector<double> invariants(const Fixture& f, const Flow::State& state)
{
    std::vector<double> local(9);
    for (const auto& cell : state.cells)
    {
        local[0] += mass(cell);
        for (const auto& phase : cell.phases)
            for (size_t s = 0; s < 2; ++s)
                local[1 + s] += phase.moles[s];
        for (size_t p = 1; p < cell.phases.size(); ++p)
        {
            local[2 + p] += *cell.phase_numbers[p];
            local[4 + p] += cell.qbmm_shape[p]->H2;
            local[6 + p] += cell.qbmm_shape[p]->H3;
        }
    }
    return f.sum(local);
}

void check_budgets(const Fixture& f, const Flow::Diagnostics& diagnostics)
{
    EXPECT_LE(diagnostics.maximum_occupancy_error, f.options.volume_relative_tolerance);
    EXPECT_LE(diagnostics.maximum_continuity_residual, f.options.continuity_tolerance);
    EXPECT_LE(diagnostics.maximum_energy_transport_residual, 1);
    EXPECT_LE(diagnostics.maximum_species_transport_residual, 1);
    EXPECT_LE(diagnostics.maximum_population_transport_residual, 1);
    EXPECT_LE(diagnostics.maximum_shape_transport_residual, 1);
}

void check_body_impulse(const Fixture& f, const Flow::State& before, const Flow::State& after,
    const Flow::Diagnostics& diagnostics, double gravity)
{
    double local_mass = 0, local_old_momentum = 0, local_new_momentum = 0;
    for (size_t c = 0; c < before.cells.size(); ++c)
    {
        local_mass += mass(before.cells[c]);
        local_old_momentum += mass(before.cells[c]) * before.cells[c].velocity.z;
        local_new_momentum += mass(after.cells[c]) * after.cells[c].velocity.z;
    }
    // No chemistry or material sources: global transported mass is the independently known old mass.
    const auto expected = f.sum(local_mass) * gravity * dt;
    near_relative(diagnostics.external_impulse.z, expected, 2e-11);
    EXPECT_EQ(diagnostics.external_impulse.x, 0);
    EXPECT_EQ(diagnostics.external_impulse.y, 0);
    const double delta = f.sum(local_new_momentum - local_old_momentum);
    EXPECT_NEAR(delta, diagnostics.external_impulse.z + diagnostics.pressure_impulse.z, 2e-10 * std::abs(expected));
    record("body_impulse_kg_m_per_s", diagnostics.external_impulse.z);
    record("pressure_impulse_kg_m_per_s", diagnostics.pressure_impulse.z);
    record(
        "momentum_balance_error_kg_m_per_s", delta - diagnostics.external_impulse.z - diagnostics.pressure_impulse.z);
}

struct ColumnMeasures
{
    double maximum_speed = 0, maximum_flux = 0, maximum_pressure_error = 0, maximum_density_error = 0;
    std::array<double, cells> p{}, rho{};
};

ColumnMeasures measure_column(const Fixture& f, const Flow::State& reference)
{
    ColumnMeasures result;
    std::vector<double> local_p(cells), local_rho(cells);
    for (size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
    {
        const auto id = static_cast<LO>(c);
        const auto i = f.index(id);
        const auto& cell = f.flow->state().cells[c];
        local_p[i] = f.flow->state().thermodynamic_pressure + cell.mechanical_pressure;
        local_rho[i] = mass(cell) / f.mesh->cell_volume(id);
        result.maximum_pressure_error = std::max(result.maximum_pressure_error,
            std::abs(local_p[i] - reference.thermodynamic_pressure - reference.cells[c].mechanical_pressure));
        result.maximum_density_error =
            std::max(result.maximum_density_error, std::abs(local_rho[i] - f.flow->density().owned_value(id)));
        const auto published = f.flow->velocity().owned_value(id);
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_EQ(published.component(axis), cell.velocity.component(axis));
            result.maximum_speed = std::max(result.maximum_speed, std::abs(published.component(axis)));
        }
    }
    for (size_t face = 0; face < f.mesh->num_owned_faces(); ++face)
    {
        const auto id = static_cast<LO>(face);
        const auto flux = f.flow->volume_flux().owned_value(id);
        EXPECT_TRUE(std::isfinite(flux));
        result.maximum_flux = std::max(result.maximum_flux, std::abs(flux));
        if (f.mesh->is_geometry_exterior_face(id))
            EXPECT_EQ(flux, 0);
    }
    const auto p = f.sum(local_p), rho = f.sum(local_rho);
    std::copy(p.begin(), p.end(), result.p.begin());
    std::copy(rho.begin(), rho.end(), result.rho.begin());
    result.maximum_speed = f.maximum(result.maximum_speed);
    result.maximum_flux = f.maximum(result.maximum_flux);
    result.maximum_pressure_error = f.maximum(result.maximum_pressure_error);
    result.maximum_density_error = f.maximum(result.maximum_density_error);
    record("maximum_speed_m_per_s", result.maximum_speed);
    record("maximum_carrier_flux_m3_per_s", result.maximum_flux);
    record("maximum_absolute_pressure_drift_pa", result.maximum_pressure_error);
    record("maximum_published_density_error_kg_per_m3", result.maximum_density_error);
    record("bottom_to_top_pressure_difference_pa", p.back() - p.front());
    record("bottom_density_kg_per_m3", rho.front());
    record("top_density_kg_per_m3", rho.back());
    return result;
}

void hydrostatic_column(Specification spec)
{
    const double gravity = *spec.hydrostatic_acceleration;
    Fixture f(spec);
    f.check_ownership();
    const auto initial = f.flow->state();
    const auto inventory = invariants(f, initial);
    const auto expected_pressures = f.column_pressures(gravity);
    const double expected_pressure_impulse = -inventory[0] * gravity * dt;
    const auto check_equilibrium_impulse = [&](const Flow::Diagnostics& diagnostics)
    {
        const double tolerance = 2e-10 * std::abs(expected_pressure_impulse);
        EXPECT_NEAR(diagnostics.pressure_impulse.x, 0, tolerance);
        EXPECT_NEAR(diagnostics.pressure_impulse.y, 0, tolerance);
        EXPECT_NEAR(diagnostics.pressure_impulse.z, expected_pressure_impulse, tolerance);
    };
    // The reacting controller requires a liquid compartment. Exercise the native force
    // path directly for the pure-gas oracle; liquid and QBMM cases use the public gravity API.
    std::unique_ptr<Coupler> controller;
    std::shared_ptr<const Flow::SourceStageProvider> native_gravity;
    if (!spec.single_phase_gas)
        controller = std::make_unique<Coupler>(f.flow, f.controller_options(Flow::Vector{0, 0, gravity}));
    else
        native_gravity = std::make_shared<const Flow::SourceStageProvider>(
            Flow::SourceStageProvider{"Q6 native isothermal gas gravity", 1, f.catalog->fingerprint(),
                [gravity](const Flow::SourceStageContext& context)
                {
                    Flow::SourceStageResult result;
                    result.sources.resize(context.transport.cells.size());
                    for (size_t c = 0; c < result.sources.size(); ++c)
                    {
                        double physical_mass = 0;
                        for (const auto& phase : context.transport.cells[c].phases)
                            for (size_t s = 0; s < molar_mass.size(); ++s)
                                physical_mass += molar_mass[s] * phase.moles[s];
                        result.sources[c].momentum =
                            Flow::Vector{0, 0, physical_mass * gravity * context.transport.time_step};
                    }
                    return result;
                },
                0, 0, Flow::Vector{0, 0, gravity}});
    ColumnMeasures measured;
    for (int step = 0; step < 3; ++step)
    {
        SCOPED_TRACE(step);
        const Endpoint accepted(*f.flow);
        const auto before = f.flow->state();
        if (controller)
        {
            const auto trial = controller->preview(dt);
            accepted.expect_same(*f.flow);
            check_budgets(f, trial.diagnostics().flow);
            check_body_impulse(f, before, trial.state(), trial.diagnostics().flow, gravity);
            check_equilibrium_impulse(trial.diagnostics().flow);
            const auto calls = f.source_calls;
            controller->commit(trial);
            EXPECT_EQ(f.source_calls, calls);
        }
        else
        {
            const auto trial = f.flow->preview_with_sources(dt, native_gravity);
            accepted.expect_same(*f.flow);
            check_budgets(f, trial.diagnostics());
            check_body_impulse(f, before, trial.state(), trial.diagnostics(), gravity);
            check_equilibrium_impulse(trial.diagnostics());
            f.flow->commit(trial);
        }
        measured = measure_column(f, initial);
        EXPECT_LT(measured.maximum_speed, 1e-9);
        EXPECT_LT(measured.maximum_flux, 1e-10);
        for (size_t j = 0; j < cells; ++j)
            EXPECT_NEAR(measured.p[j], expected_pressures[j], 1e-6) << "column cell " << j;
        EXPECT_LT(measured.maximum_density_error, 1e-10);
        const auto now = invariants(f, f.flow->state());
        for (size_t k = 0; k < inventory.size(); ++k)
            near_relative(now[k], inventory[k]);
    }
    double balance_error = 0;
    for (size_t j = 0; j + 1 < cells; ++j)
    {
        const double hydrostatic_drop = gravity * .25 * (measured.rho[j] + measured.rho[j + 1]) / 2;
        balance_error = std::max(balance_error, std::abs(measured.p[j + 1] - measured.p[j] - hydrostatic_drop));
    }
    record("maximum_discrete_hydrostatic_residual_pa", balance_error);
    EXPECT_LT(balance_error, 1e-6);
    if (spec.stratified || spec.single_phase_gas)
    {
        EXPECT_GT(measured.rho.front(), measured.rho.back());
        EXPECT_GT(measured.rho.front() - measured.rho.back(), spec.polydisperse ? 200 : 1e-3);
    }
    if (spec.polydisperse)
    {
        const auto report = controller->qbmm_report();
        for (const auto& cell : report.cells)
            for (size_t p = 1; p < 3; ++p)
            {
                EXPECT_EQ(cell.families[p]->quadrature.status, R::InversionStatus::TwoNodes);
                EXPECT_GT(cell.families[p]->phase_volume_m3, 0);
            }
    }
}

TEST(QbmmSlipAndGravity, SinglePhaseClosedLiquidColumnPreservesGenuineHydrostaticRest)
{
    hydrostatic_column(
        {.layout = Layout::Column, .polydisperse = false, .phase_slip = false, .hydrostatic_acceleration = -1});
}

TEST(QbmmSlipAndGravity, HomogeneousPolydisperseClosedColumnPreservesGenuineHydrostaticRest)
{
    hydrostatic_column({.layout = Layout::Column, .phase_slip = false, .hydrostatic_acceleration = -1});
}

TEST(QbmmSlipAndGravity, IsothermalGasColumnWithDifferentLocalDensitiesPreservesHydrostaticRest)
{
    hydrostatic_column({.layout = Layout::Column,
        .polydisperse = false,
        .phase_slip = false,
        .single_phase_gas = true,
        .hydrostatic_acceleration = -512});
}

TEST(QbmmSlipAndGravity, StratifiedPolydisperseColumnUsesLocalAbsolutePressureAndDensity)
{
    hydrostatic_column(
        {.layout = Layout::Column, .phase_slip = false, .stratified = true, .hydrostatic_acceleration = -1});
}

TEST(QbmmSlipAndGravity, StratifiedHydrostaticColumnIncludesActualEmptyOwners)
{
    hydrostatic_column(
        {.layout = Layout::EmptyColumn, .phase_slip = false, .stratified = true, .hydrostatic_acceleration = -1});
}

void gravity_from_flat_pressure(bool polydisperse)
{
    Fixture f({.layout = Layout::Column, .polydisperse = polydisperse, .phase_slip = false});
    f.check_ownership();
    const auto initial = f.flow->state();
    const auto inventory = invariants(f, initial);
    // The unforced identical column must remain flat. This detects gravity that is silently ignored.
    {
        Coupler unforced(f.flow, f.controller_options());
        unforced.advance(dt);
        EXPECT_EQ(unforced.diagnostics().flow.external_impulse.z, 0);
        const auto quiet = measure_column(f, initial);
        EXPECT_LT(quiet.maximum_pressure_error, 1e-7);
    }
    f.flow->initialize(initial);
    Coupler forced(f.flow, f.controller_options(Flow::Vector{0, 0, -1}));
    for (int step = 0; step < 4; ++step)
    {
        const auto before = f.flow->state();
        const Endpoint accepted(*f.flow);
        const auto trial = forced.preview(dt);
        accepted.expect_same(*f.flow);
        check_budgets(f, trial.diagnostics().flow);
        check_body_impulse(f, before, trial.state(), trial.diagnostics().flow, -1);
        const auto calls = f.source_calls;
        forced.commit(trial);
        EXPECT_EQ(f.source_calls, calls);
    }
    const auto measured = measure_column(f, initial);
    EXPECT_GT(measured.maximum_pressure_error, 1);
    EXPECT_LT(measured.p.back() - measured.p.front(), -1);
    const auto now = invariants(f, f.flow->state());
    for (size_t k = 0; k < inventory.size(); ++k)
        near_relative(now[k], inventory[k]);
}

TEST(QbmmSlipAndGravity, SinglePhaseGravityActuallyBuildsPressureAndUsesPhysicalMassImpulse)
{
    gravity_from_flat_pressure(false);
}

TEST(QbmmSlipAndGravity, PolydisperseGravityActuallyBuildsPressureAndUsesPhysicalMassImpulse)
{
    gravity_from_flat_pressure(true);
}

TEST(QbmmSlipAndGravity, InvalidAndRankDivergentGravityRejectWithoutChangingFieldsOrTakingControllerLease)
{
    for (const bool poly : {false, true})
    {
        Fixture f({.layout = Layout::EmptyColumn, .polydisperse = poly, .phase_slip = false});
        f.check_ownership();
        const auto comm = f.mesh->owned_cell_map()->getComm();
        const Endpoint accepted(*f.flow);
        const auto retained = f.flow->preview(dt);
        for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        {
            auto controls = f.controller_options(Flow::Vector{0, 0, -1});
            if (comm->getRank() == comm->getSize() - 1)
                controls.body_acceleration = Flow::Vector{invalid, 0, -1};
            EXPECT_THROW((void) Coupler(f.flow, controls), std::exception);
            accepted.expect_same(*f.flow);
        }
        if (comm->getSize() > 1)
            for (const bool missing : {false, true})
            {
                auto controls = f.controller_options(Flow::Vector{0, 0, -1});
                if (comm->getRank() == comm->getSize() - 1)
                {
                    if (missing)
                        controls.body_acceleration.reset();
                    else
                        controls.body_acceleration = Flow::Vector{0, 0, -2};
                }
                EXPECT_THROW((void) Coupler(f.flow, controls), std::exception);
                accepted.expect_same(*f.flow);
            }
        EXPECT_EQ(f.source_calls, 0);
        // Failed constructor attempts must leave an earlier native trial usable.
        f.flow->commit(retained);
        Coupler valid(f.flow, f.controller_options(Flow::Vector{0, 0, -1}));
        const auto before = f.flow->state();
        const auto trial = valid.preview(dt);
        check_body_impulse(f, before, trial.state(), trial.diagnostics().flow, -1);
        valid.commit(trial);
    }
}

TEST(QbmmSlipAndGravity, GravityCheckpointAndRestartBindAccelerationAndContinueWithActualEmptyOwners)
{
    Fixture f({.layout = Layout::EmptyColumn, .phase_slip = false, .stratified = true, .hydrostatic_acceleration = -1});
    f.check_ownership();
    Coupler controller(f.flow, f.controller_options(Flow::Vector{0, 0, -1}));
    controller.advance(dt);
    const auto save = controller.checkpoint();
    const auto archive = f.flow->serialize_qbmm_restart();
    const Endpoint midpoint(*f.flow);
    controller.advance(dt);
    controller.advance(dt);
    const Endpoint final(*f.flow);
    const auto calls = f.source_calls;
    controller.restore(save);
    EXPECT_EQ(f.source_calls, calls);
    midpoint.expect_same(*f.flow);
    controller.advance(dt);
    controller.advance(dt);
    final.expect_same(*f.flow);

    Fixture fresh(f.spec);
    fresh.flow->restore_qbmm_restart(archive);
    midpoint.expect_same(*fresh.flow);
    const auto comm = fresh.mesh->owned_cell_map()->getComm();
    auto wrong = fresh.controller_options(Flow::Vector{0, 0, -1});
    // On MPI only an empty owner differs; on serial this is a uniformly wrong restart signature.
    if (comm->getRank() == comm->getSize() - 1)
        wrong.body_acceleration = Flow::Vector{0, 0, -2};
    EXPECT_THROW((void) Coupler(fresh.flow, wrong), std::exception);
    midpoint.expect_same(*fresh.flow);
    EXPECT_EQ(fresh.source_calls, 0);
    Coupler resumed(fresh.flow, fresh.controller_options(Flow::Vector{0, 0, -1}));
    resumed.advance(dt);
    resumed.advance(dt);
    final.expect_same(*fresh.flow);
    record("continued_time_s", fresh.flow->state().time);
}
} // namespace
