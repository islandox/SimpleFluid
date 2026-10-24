/** @file LowMachMixtureSolver.cc
 * @brief Conservative native mixture transport, thermodynamic closure and immutable advancement.
 */
#include "solvers/LowMachMixtureSolver.hh"
#include "FVM/CellOperators.hh"
#include "FVM/TransportSystem.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/EquationValidation.hh"
#include "equations/VariableDensityMomentumEquation.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace SimpleFluid::Reacting
{
namespace
{
using Solver = LowMachMixtureSolver;
using LO = Solver::Pack::local_ordinal_type;
using ScalarField = Solver::Traits::scalar_cell_type;
using VectorField = Solver::Traits::vector_cell_type;
using FaceField = Solver::Traits::scalar_face_type;
void check(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
double phase_mass(const PhaseThermalState& phase)
{
    double mass = 0;
    for (size_t s = 0; s < phase.moles.size(); ++s)
        mass += phase.moles[s] * phase.thermodynamics->catalog().species()[s].molarMass;
    return mass;
}
std::vector<double> fractions(const PhaseThermalState& phase)
{
    const double n = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.0);
    auto x = phase.moles;
    for (double& value : x)
        value = n > 0 ? value / n : 0;
    return x;
}
double energy_sum(const Solver::CellState& cell)
{
    double energy = 0;
    for (const auto& phase : cell.phases)
        energy += phase.internal_energy;
    return energy;
}
class CaloricRangeError final : public std::domain_error
{
public:
    using std::domain_error::domain_error;
};
struct CaloricCell
{
    Solver::CellState cell;
    std::vector<double> volumes;
    double volume = 0, derivative = 0, work = 0, roundoff = 0, work_magnitude = 0, pressure_operands = 0;
};
// U_new+p_new*V_new = U_adv+source+p_new*V_adv owns actual phase-volume work once.
CaloricCell recover_enthalpy(const Solver::CellState& transported, std::span<const double> advected_volumes, double p0,
    const Solver::Options& options)
{
    CaloricCell result{transported, std::vector<double>(transported.phases.size()), 0, 0, 0};
    const size_t count = transported.phases.size();
    std::vector<std::vector<double>> x(count);
    std::vector<double> mass(count), targets(count);
    double lower = 0, upper = std::numeric_limits<double>::infinity(), total_target = 0;
    for (size_t a = 0; a < count; ++a)
    {
        auto& phase = result.cell.phases[a];
        phase.absolute_pressure = p0 + transported.mechanical_pressure;
        x[a] = fractions(phase);
        mass[a] = phase_mass(phase);
        targets[a] = phase.internal_energy + phase.absolute_pressure * advected_volumes[a];
        result.pressure_operands += std::abs(phase.absolute_pressure * advected_volumes[a]);
        total_target += targets[a];
        if (mass[a] > 0)
        {
            lower = std::max(lower, phase.thermodynamics->bounds().minimum_temperature);
            upper = std::min(upper, phase.thermodynamics->bounds().maximum_temperature);
        }
        else
        {
            const double bound =
                32 * std::numeric_limits<double>::epsilon() *
                (std::abs(phase.internal_energy) + std::abs(phase.absolute_pressure * advected_volumes[a]));
            check(std::abs(targets[a]) <= bound, "Empty phase has unallocated physical energy/volume work.");
            result.roundoff -= targets[a];
            total_target -= targets[a];
            result.work += phase.absolute_pressure * advected_volumes[a];
            result.work_magnitude += std::abs(phase.absolute_pressure * advected_volumes[a]);
        }
    }
    auto invert = [&](size_t selected, double target, double lo, double hi)
    {
        auto enthalpy = [&](double t)
        {
            double value = 0;
            for (size_t a = 0; a < count; ++a)
                if (mass[a] > 0 && (selected == count || selected == a))
                    value += mass[a] * result.cell.phases[a]
                                           .thermodynamics->evaluate(t, result.cell.phases[a].absolute_pressure, x[a])
                                           .enthalpy;
            return value;
        };
        check(lo <= hi && std::isfinite(target), "Low-Mach enthalpy target or common caloric interval is invalid.");
        const double minimum = enthalpy(lo), maximum = enthalpy(hi);
        check(std::isfinite(minimum) && std::isfinite(maximum), "Low-Mach enthalpy bounds are nonfinite.");
        if (target < minimum || target > maximum)
            throw CaloricRangeError("Low-Mach phase enthalpy is outside supplied caloric validity bounds.");
        for (int iteration = 0; iteration < 180; ++iteration)
        {
            const double mid = std::midpoint(lo, hi), value = enthalpy(mid);
            const double scale = std::max({std::numeric_limits<double>::min(), std::abs(value), std::abs(target)});
            if (std::abs(value - target) <= 32 * std::numeric_limits<double>::epsilon() * scale || mid == lo ||
                mid == hi)
                return mid;
            if (value < target)
                lo = mid;
            else
                hi = mid;
        }
        throw std::runtime_error("Low-Mach phase enthalpy inversion did not converge.");
    };
    const double common =
        options.thermal_mode == PhaseThermalMode::Equilibrium ? invert(count, total_target, lower, upper) : 0;
    double h_t = 0, h_p = 0, v_t = 0, v_p = 0, adv_volume = 0;
    for (size_t a = 0; a < count; ++a)
    {
        auto& phase = result.cell.phases[a];
        if (mass[a] == 0)
        {
            phase.temperature = 0;
            phase.internal_energy = 0;
            continue;
        }
        const auto& bounds = phase.thermodynamics->bounds();
        phase.temperature = options.thermal_mode == PhaseThermalMode::Equilibrium
                                ? common
                                : invert(a, targets[a], bounds.minimum_temperature, bounds.maximum_temperature);
        const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, x[a]);
        const double rho2 = point.density * point.density;
        const double vt = -mass[a] * point.density_temperature / rho2;
        const double vp = -mass[a] * point.density_pressure / rho2;
        const double ht =
            mass[a] * (point.internal_energy_temperature - phase.absolute_pressure * point.density_temperature / rho2);
        const double hp = mass[a] * (1 / point.density + phase.temperature * point.density_temperature / rho2);
        check(std::isfinite(ht) && ht > 0 && std::isfinite(hp) && std::isfinite(vt) && std::isfinite(vp),
            "Low-Mach phase caloric/pressure derivatives are invalid.");
        result.volumes[a] = mass[a] / point.density;
        result.pressure_operands += std::abs(phase.absolute_pressure * result.volumes[a]);
        phase.internal_energy = mass[a] * point.internal_energy;
        result.volume += result.volumes[a];
        const double phase_work = -phase.absolute_pressure * (result.volumes[a] - advected_volumes[a]);
        result.work += phase_work;
        result.work_magnitude += std::abs(phase_work);
        h_t += ht;
        h_p += hp;
        v_t += vt;
        v_p += vp;
        adv_volume += advected_volumes[a];
        if (options.thermal_mode == PhaseThermalMode::SeparatePhaseTemperatures)
            result.derivative += vp + vt * (advected_volumes[a] - hp) / ht;
    }
    if (options.thermal_mode == PhaseThermalMode::Equilibrium)
        result.derivative = v_p + v_t * (adv_volume - h_p) / h_t;
    check(std::isfinite(result.derivative), "Low-Mach mixture pressure susceptibility is nonfinite.");
    return result;
}
} // namespace

struct LowMachMixtureSolver::Implementation
{
    SP<const Mesh> mesh;
    std::vector<std::shared_ptr<const MixtureThermodynamics>> eos;
    Options options;
    State accepted;
    Diagnostics diagnostics;
    bool initialized = false, controller_active = false;
    std::uint64_t trial_nonce = 0, controller_id = 0, source_invocation = 0;
    std::uint64_t epoch = 0;
    ScalarField rho, old_rho, mu, mechanical_pressure, old_scalar, scalar, unit, zero, published_rho;
    VectorField u, old_u, pressure_gradient, published_u;
    FaceField flux, trial_flux, mass_flux;
    FVM::TransportGeometryCache<Mesh> geometry;
    FVM::CellGradientCache<Pack, Mesh> gradient_cache;
    BelosLinearSolver<Pack> transport_solver;
    VariableDensityMomentumEquation<Pack, Mesh> momentum;
    LowMachPressureEquation pressure;
    Implementation(
        SP<const Mesh> grid, std::vector<std::shared_ptr<const MixtureThermodynamics>> providers, Options controls)
        : mesh(EquationValidation::require_non_null_mesh(std::move(grid), "LowMachMixtureSolver")),
          eos(std::move(providers)), options(std::move(controls)), rho(mesh, "mixture_density"),
          old_rho(mesh, "accepted_density"), mu(mesh, "mixture_dynamic_viscosity"),
          mechanical_pressure(mesh, "mechanical_pressure"), old_scalar(mesh, "mixture_transport_old"),
          scalar(mesh, "mixture_transport_trial"), unit(mesh, 1.0, "mixture_unit_weight"),
          zero(mesh, 0.0, "mixture_no_diffusion"), published_rho(mesh, "accepted_mixture_density"),
          u(mesh, "continuous_velocity"), old_u(mesh, "accepted_continuous_velocity"),
          pressure_gradient(mesh, "mechanical_pressure_gradient"), published_u(mesh, "accepted_velocity"),
          flux(mesh, 0.0, "accepted_volume_flux"), trial_flux(mesh, 0.0, "trial_volume_flux"),
          mass_flux(mesh, 0.0, "actual_mass_flux"), geometry(*mesh), gradient_cache(mesh), momentum(mesh),
          pressure(mesh)
    {
        collective_detail::collective_local_validation(*mesh, "Low-Mach mixture options",
            [&]
            {
                check(std::isfinite(options.minimum_pressure) && std::isfinite(options.maximum_pressure) &&
                          std::isfinite(options.volume_relative_tolerance) &&
                          std::isfinite(options.conservation_relative_tolerance) &&
                          std::isfinite(options.continuity_tolerance) && options.continuity_tolerance > 0,
                    "Low-Mach physical tolerances and pressure bounds must be finite and positive.");
                check(std::isfinite(options.source_relative_tolerance) && options.source_relative_tolerance > 0 &&
                          std::isfinite(options.thermal_relative_tolerance) && options.thermal_relative_tolerance > 0,
                    "Source/thermal corrector tolerances must be finite and positive.");
                check(!eos.empty() && options.minimum_pressure > 0 &&
                          options.maximum_pressure > options.minimum_pressure && options.maximum_correctors > 0 &&
                          options.maximum_pressure_iterations > 0 && options.volume_relative_tolerance > 0 &&
                          options.conservation_relative_tolerance > 0,
                    "Low-Mach mixture options/providers are incomplete.");
                check(options.maximum_caloric_seed_evaluations > 0 && options.maximum_caloric_seed_evaluations <= 4096,
                    "Caloric seed-search budget must be between 1 and 4096.");
                if (options.caloric_domain_provider)
                    check(!options.caloric_domain_provider->identity.empty() &&
                              options.caloric_domain_provider->identity.size() <=
                                  static_cast<size_t>(std::numeric_limits<int>::max()) &&
                              options.caloric_domain_provider->revision > 0 &&
                              bool(options.caloric_domain_provider->evaluate),
                        "Caloric domain provider identity/revision/callback is invalid.");
                check(options.thermal_mode == PhaseThermalMode::Equilibrium ||
                          options.thermal_mode == PhaseThermalMode::SeparatePhaseTemperatures,
                    "Unknown low-Mach phase thermal mode.");
                check(eos.front() != nullptr, "Null mixture EOS.");
                for (const auto& provider : eos)
                    check(provider && &provider->catalog() == &eos.front()->catalog() &&
                              provider->energy_convention() == eos.front()->energy_convention(),
                        "Every low-Mach phase must share catalog and caloric formation convention.");
                if (options.phase_roles.empty())
                    for (const auto& provider : eos)
                        options.phase_roles.push_back(
                            {provider->kind() == EOSKind::Liquid ? PhaseRole::BulkLiquid : PhaseRole::BulkGas, {}});
                check(options.phase_roles.size() == eos.size(), "Physical phase-role schema must match EOS phases.");
                std::set<std::string> identities;
                for (size_t a = 0; a < eos.size(); ++a)
                {
                    const auto& role = options.phase_roles[a];
                    check(role.role == PhaseRole::BulkLiquid || role.role == PhaseRole::BulkGas ||
                              role.role == PhaseRole::DispersedGas,
                        "Unknown physical phase role.");
                    check((role.role == PhaseRole::BulkLiquid) == (eos[a]->kind() == EOSKind::Liquid),
                        "Physical phase role/EOS kind mismatch.");
                    check(role.identity.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                              (role.role != PhaseRole::DispersedGas || !role.identity.empty()) &&
                              (role.identity.empty() || identities.insert(role.identity).second),
                        "Physical phase identities must be unique and dispersed identities explicit.");
                }
                for (size_t f = 0; f < mesh->num_faces(); ++f)
                {
                    const auto face = static_cast<LO>(f);
                    if (mesh->is_geometry_exterior_face(face))
                        continue;
                    const auto owner = mesh->owner_cell(face), neighbor = mesh->neighbor_cell(face);
                    const auto owned = [&](LO cell)
                    { return cell >= 0 && static_cast<size_t>(cell) < mesh->num_owned_cells(); };
                    if (!owned(owner) && !owned(neighbor))
                        continue;
                    check(owner >= 0 && neighbor >= 0, "An owned low-Mach incidence lacks its adjacent halo cell.");
                    const auto displacement = mesh->cell_center_vector(face, owner);
                    const auto normal = mesh->face_normal(face);
                    check(std::abs(displacement.dot(normal)) >= (1 - 1e-10) * displacement.norm(),
                        "Low-Mach mixture currently requires orthogonal mapped face geometry.");
                }
            });
        collective_detail::require_uniform_value(*mesh, options.minimum_pressure, "Low-Mach minimum pressure");
        collective_detail::require_uniform_value(*mesh, options.maximum_pressure, "Low-Mach maximum pressure");
        collective_detail::require_uniform_value(*mesh, options.maximum_correctors, "Low-Mach maximum correctors");
        collective_detail::require_uniform_value(
            *mesh, options.maximum_pressure_iterations, "Low-Mach pressure iterations");
        collective_detail::require_uniform_value(*mesh, options.volume_relative_tolerance, "Low-Mach volume gate");
        collective_detail::require_uniform_value(*mesh, options.continuity_tolerance, "Low-Mach continuity gate");
        collective_detail::require_uniform_value(
            *mesh, options.conservation_relative_tolerance, "Low-Mach conservation gate");
        collective_detail::require_uniform_value(*mesh, options.source_relative_tolerance, "Low-Mach source gate");
        collective_detail::require_uniform_value(*mesh, options.thermal_relative_tolerance, "Low-Mach thermal gate");
        collective_detail::require_uniform_value(
            *mesh, options.maximum_caloric_seed_evaluations, "Caloric seed-search budget");
        collective_detail::require_uniform_value(
            *mesh, int(bool(options.caloric_domain_provider)), "Caloric domain provider presence");
        if (options.caloric_domain_provider)
        {
            const auto& provider = *options.caloric_domain_provider;
            collective_detail::require_uniform_value(*mesh, provider.identity.size(), "Caloric domain identity length");
            collective_detail::require_uniform_value(*mesh, provider.revision, "Caloric domain provider revision");
            std::string identity;
            collective_detail::collective_local_validation(
                *mesh, "Caloric domain metadata preparation", [&] { identity = provider.identity; });
            Teuchos::broadcast(
                *mesh->owned_cell_map()->getComm(), 0, static_cast<int>(identity.size()), identity.data());
            collective_detail::collective_local_validation(*mesh, "Caloric domain metadata", [&]
                { check(identity == provider.identity, "Caloric domain provider identity differs across ranks."); });
        }
        collective_detail::require_uniform_value(*mesh, eos.size(), "Low-Mach phase count");
        for (const auto& role : options.phase_roles)
        {
            collective_detail::require_uniform_value(*mesh, int(role.role), "Physical phase role");
            collective_detail::require_uniform_value(*mesh, role.identity.size(), "Physical phase identity length");
            std::string identity;
            collective_detail::collective_local_validation(
                *mesh, "Physical phase identity preparation", [&] { identity = role.identity; });
            Teuchos::broadcast(
                *mesh->owned_cell_map()->getComm(), 0, static_cast<int>(identity.size()), identity.data());
            collective_detail::collective_local_validation(*mesh, "Physical phase identity",
                [&] { check(identity == role.identity, "Physical phase identity differs across ranks."); });
        }
        collective_detail::require_uniform_value(
            *mesh, eos.front()->catalog().revision(), "Low-Mach species schema revision");
        collective_detail::require_uniform_value(
            *mesh, eos.front()->catalog().species().size(), "Low-Mach species count");
        collective_detail::require_uniform_value(
            *mesh, static_cast<int>(options.thermal_mode), "Low-Mach thermal mode");
        collective_detail::require_uniform_value(
            *mesh, eos.front()->catalog().fingerprint(), "Low-Mach species schema fingerprint");
        collective_detail::require_uniform_value(
            *mesh, int(bool(options.mixture_dynamic_viscosity)), "Low-Mach viscosity closure presence");
        for (const auto& provider : eos)
        {
            collective_detail::require_uniform_value(
                *mesh, static_cast<int>(provider->kind()), "Low-Mach phase EOS kind");
            collective_detail::require_uniform_value(
                *mesh, static_cast<int>(provider->energy_convention()), "Low-Mach caloric convention");
            collective_detail::require_uniform_value(*mesh, static_cast<int>(provider->enthalpy_pressure_dependence()),
                "Immutable EOS enthalpy-pressure capability");
            const auto& bounds = provider->bounds();
            collective_detail::require_uniform_value(
                *mesh, bounds.minimum_temperature, "Low-Mach EOS minimum temperature");
            collective_detail::require_uniform_value(
                *mesh, bounds.maximum_temperature, "Low-Mach EOS maximum temperature");
            collective_detail::require_uniform_value(*mesh, bounds.minimum_pressure, "Low-Mach EOS minimum pressure");
            collective_detail::require_uniform_value(*mesh, bounds.maximum_pressure, "Low-Mach EOS maximum pressure");
        }
        epoch = mesh_geometry_epoch(*mesh);
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
    void publish_fields(const State& state, bool publish = false)
    {
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            double mass = 0;
            for (const auto& phase : state.cells[c].phases)
                mass += phase_mass(phase);
            rho.set_owned_value(static_cast<LO>(c), mass / mesh->cell_volume(static_cast<LO>(c)));
            u.set_owned_value(static_cast<LO>(c), state.cells[c].velocity);
            mechanical_pressure.set_owned_value(static_cast<LO>(c), state.cells[c].mechanical_pressure);
        }
        rho.sync_ghosts();
        u.sync_ghosts();
        mechanical_pressure.sync_ghosts();
        if (publish)
        {
            published_rho.owned_data().update(1.0, rho.owned_data(), 0.0);
            published_rho.sync_ghosts();
            published_u.owned_data().update(1.0, u.owned_data(), 0.0);
            published_u.sync_ghosts();
        }
    }
    void transport(std::span<const double> old_amount, const FaceField& phase_flux, double dt,
        std::vector<double>& new_amount, Diagnostics& statistics)
    {
        for (size_t c = 0; c < old_amount.size(); ++c)
            old_scalar.set_owned_value(static_cast<LO>(c), old_amount[c] / mesh->cell_volume(static_cast<LO>(c)));
        old_scalar.sync_ghosts();
        scalar.owned_data().update(1.0, old_scalar.owned_data(), 0.0);
        auto system = FVM::weighted_scalar_transport_system<Pack>(
            FVM::MeshWeightedScalarTransportRequest<Pack, Mesh>{.old_values = old_scalar,
                .face_fluxes = phase_flux,
                .time_step = dt,
                .storage_weight = unit,
                .advection_weight = unit,
                .diffusivity = zero,
                .boundary_condition = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; },
                .boundary_value = [](int, size_t) { return 0.0; },
                .source = [](LO) { return 0.0; },
                .treatment = FVM::NonOrthogonalTreatment::Implicit,
                .geometry_cache = &geometry});
        transport_solver.notify_operator_values_changed();
        const auto result =
            transport_solver.solve_with_statistics(system.matrix, *system.rhs, scalar.owned_data(), options.linear);
        statistics.transport_linear.add(result);
        if (!result.converged)
            throw std::runtime_error("Low-Mach phase transport failed true linear residual gate.");
        new_amount.resize(old_amount.size());
        for (size_t c = 0; c < old_amount.size(); ++c)
            new_amount[c] = scalar.value(static_cast<LO>(c)) * mesh->cell_volume(static_cast<LO>(c));
    }
};
LowMachMixtureSolver::LowMachMixtureSolver(
    SP<const Mesh> mesh, std::vector<std::shared_ptr<const MixtureThermodynamics>> providers, Options options)
    : d(std::make_unique<Implementation>(std::move(mesh), std::move(providers), std::move(options)))
{
}
LowMachMixtureSolver::~LowMachMixtureSolver() = default;
bool LowMachMixtureSolver::initialized() const noexcept
{
    return d->initialized;
}
const SP<const LowMachMixtureSolver::Mesh>& LowMachMixtureSolver::mesh() const noexcept
{
    return d->mesh;
}
const LowMachMixtureSolver::Options& LowMachMixtureSolver::options() const noexcept
{
    return d->options;
}
const LowMachMixtureSolver::State& LowMachMixtureSolver::state() const noexcept
{
    return d->accepted;
}
const LowMachMixtureSolver::Diagnostics& LowMachMixtureSolver::diagnostics() const noexcept
{
    return d->diagnostics;
}
const LowMachMixtureSolver::Traits::scalar_face_type& LowMachMixtureSolver::volume_flux() const noexcept
{
    return d->flux;
}
const LowMachMixtureSolver::Traits::scalar_cell_type& LowMachMixtureSolver::density() const noexcept
{
    return d->published_rho;
}
const LowMachMixtureSolver::Traits::vector_cell_type& LowMachMixtureSolver::velocity() const noexcept
{
    return d->published_u;
}

void LowMachMixtureSolver::initialize(State candidate, const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    collective_detail::require_uniform_value(mesh, candidate.thermodynamic_pressure, "Initial thermodynamic pressure");
    collective_detail::require_uniform_value(mesh, candidate.time, "Initial low-Mach time");
    collective_detail::require_uniform_value(mesh, candidate.revision, "Initial low-Mach revision");
    collective_detail::collective_local_validation(mesh, "Low-Mach initial physical state",
        [&]
        {
            check(d->epoch == mesh_geometry_epoch(mesh) && candidate.cells.size() == mesh.num_owned_cells() &&
                      std::isfinite(candidate.time) &&
                      candidate.thermodynamic_pressure >= d->options.minimum_pressure &&
                      candidate.thermodynamic_pressure <= d->options.maximum_pressure,
                "Low-Mach initial state/map/pressure is invalid.");
            for (size_t c = 0; c < candidate.cells.size(); ++c)
            {
                auto& cell = candidate.cells[c];
                check(cell.phases.size() == d->eos.size() && std::isfinite(cell.mechanical_pressure),
                    "Initial phase topology or mechanical pressure is invalid.");
                if (cell.drift_velocity.empty())
                    cell.drift_velocity.resize(d->eos.size());
                check(cell.drift_velocity.size() == d->eos.size(), "Phase drift topology is invalid.");
                for (size_t a = 0; a < cell.phases.size(); ++a)
                {
                    cell.phases[a].thermodynamics = d->eos[a];
                    cell.phases[a].absolute_pressure = candidate.thermodynamic_pressure + cell.mechanical_pressure;
                    for (size_t component = 0; component < 3; ++component)
                        check(std::isfinite(cell.velocity.component(component)) &&
                                  std::isfinite(cell.drift_velocity[a].component(component)),
                            "Nonfinite initial phase velocity.");
                }
                if (cell.phase_numbers.empty())
                {
                    check(std::none_of(d->options.phase_roles.begin(), d->options.phase_roles.end(),
                              [](const auto& role) { return role.role == PhaseRole::DispersedGas; }),
                        "Declared dispersed phases require explicit initialized numbers.");
                    cell.phase_numbers.resize(d->eos.size());
                }
                source_stage_detail::validate_phase_numbers(cell, d->options.phase_roles);
                PhaseThermalRecovery recovered;
                recovered.phases = cell.phases;
                recovered.volumes.resize(cell.phases.size());
                bool supplied_temperatures_consistent = true;
                double common_temperature = 0;
                for (size_t a = 0; a < cell.phases.size(); ++a)
                {
                    const auto& phase = cell.phases[a];
                    check(phase.moles.size() == d->eos[a]->catalog().species().size() &&
                              std::isfinite(phase.internal_energy) &&
                              std::all_of(phase.moles.begin(), phase.moles.end(),
                                  [](double n) { return std::isfinite(n) && n >= 0; }),
                        "Invalid initial phase inventory/energy.");
                    const double mass = phase_mass(phase);
                    if (mass == 0)
                    {
                        check(phase.temperature == 0 && phase.internal_energy == 0,
                            "Empty initial phase cannot contain temperature or energy.");
                        continue;
                    }
                    if (!(std::isfinite(phase.temperature) && phase.temperature > 0))
                    {
                        supplied_temperatures_consistent = false;
                        continue;
                    }
                    const auto point =
                        phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase));
                    const double expected = mass * point.internal_energy;
                    const double caloric = mass * phase.temperature * point.internal_energy_temperature;
                    const double roundoff = 64 * std::numeric_limits<double>::epsilon() *
                                            (std::abs(expected) + std::abs(phase.internal_energy) +
                                                phase.absolute_pressure * mass / point.density);
                    check(roundoff <= 0.25 * d->options.conservation_relative_tolerance * caloric,
                        "Initial energy reference is too poorly conditioned for requested caloric accuracy.");
                    supplied_temperatures_consistent =
                        supplied_temperatures_consistent && std::abs(expected - phase.internal_energy) <= roundoff;
                    if (d->options.thermal_mode == PhaseThermalMode::Equilibrium)
                    {
                        if (common_temperature == 0)
                            common_temperature = phase.temperature;
                        supplied_temperatures_consistent =
                            supplied_temperatures_consistent &&
                            std::abs(common_temperature - phase.temperature) <=
                                32 * std::numeric_limits<double>::epsilon() * common_temperature;
                    }
                    recovered.volumes[a] = mass / point.density;
                    recovered.total_volume += recovered.volumes[a];
                }
                if (!supplied_temperatures_consistent)
                    recovered = recover_phase_temperatures(cell.phases, energy_sum(cell), d->options.thermal_mode);
                check(std::abs(recovered.total_volume - mesh.cell_volume(static_cast<LO>(c))) <=
                          d->options.volume_relative_tolerance * mesh.cell_volume(static_cast<LO>(c)),
                    "Initial phase EOS volumes do not fill the physical cell.");
                cell.phases = recovered.phases;
            }
        });
    d->publish_fields(candidate);
    FaceField initial_flux(d->mesh, 0.0, "validated_initial_low_mach_flux");
    collective_detail::collective_local_validation(mesh, "Low-Mach initial face flux",
        [&]
        {
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                double value = 0;
                if (!mesh.is_geometry_exterior_face(face))
                {
                    const LO owner = mesh.owner_cell(face),
                             neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                    value = (d->u.local_value(owner) * 0.5 + d->u.local_value(neighbor) * 0.5)
                                .dot(mesh.face_area_vector(face));
                }
                check(std::isfinite(value), "Initial low-Mach volume flux is nonfinite.");
                initial_flux.set_owned_value(face, value);
            }
        });
    initial_flux.sync_ghosts();
    d->publish_fields(candidate, true);
    d->flux.owned_data().update(1.0, initial_flux.owned_data(), 0.0);
    d->flux.sync_ghosts();
    ++d->trial_nonce;
    d->accepted = std::move(candidate);
    d->epoch = mesh_geometry_epoch(mesh);
    d->diagnostics = {};
    d->initialized = true;
}
LowMachMixtureSolver::Checkpoint LowMachMixtureSolver::checkpoint() const
{
    check(d->initialized, "Low-Mach solver is not initialized.");
    Checkpoint saved;
    saved.state = d->accepted;
    saved.owner = this;
    saved.geometry_epoch = d->epoch;
    saved.diagnostics = d->diagnostics;
    saved.face_flux.resize(d->mesh->num_owned_faces());
    for (size_t f = 0; f < saved.face_flux.size(); ++f)
        saved.face_flux[f] = d->flux.local_value(static_cast<LO>(f));
    return saved;
}
void LowMachMixtureSolver::restore(const Checkpoint& saved, const ControllerLease* controller)
{
    require_controller(controller);
    State candidate;
    Diagnostics statistics;
    std::vector<double> face_flux;
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach checkpoint",
        [&]
        {
            check(saved.owner == this && saved.geometry_epoch == mesh_geometry_epoch(*d->mesh) &&
                      saved.state.cells.size() == d->mesh->num_owned_cells() &&
                      saved.face_flux.size() == d->mesh->num_owned_faces(),
                "Low-Mach checkpoint belongs to a foreign solver, stale geometry or different maps.");
            candidate = saved.state;
            statistics = saved.diagnostics;
            face_flux = saved.face_flux;
        });
    // A private checkpoint already passed physical acceptance. Restoring must
    // neither call constitutive providers nor perturb its caloric state.
    d->publish_fields(candidate, true);
    for (size_t f = 0; f < face_flux.size(); ++f)
        d->flux.set_owned_value(static_cast<LO>(f), face_flux[f]);
    d->flux.sync_ghosts();
    d->accepted = std::move(candidate);
    d->diagnostics = std::move(statistics);
    d->epoch = saved.geometry_epoch;
    d->initialized = true;
    ++d->trial_nonce;
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview(
    double dt, const std::vector<CellSources>& sources, const ControllerLease* controller)
{
    return preview_impl(dt, sources, nullptr, controller);
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_with_sources(
    double dt, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    collective_detail::collective_local_validation(*d->mesh, "Source provider ownership",
        [&] { check(bool(provider), "preview_with_sources requires an immutable shared provider."); });
    return preview_impl(dt, {}, std::move(provider), controller);
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_impl(double dt, const std::vector<CellSources>& fixed_sources,
    std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    std::vector<CellSources> sources;
    const auto& mesh = *d->mesh;
    require_controller(controller);
    collective_detail::require_uniform_value(mesh, dt, "Low-Mach mixture timestep");
    collective_detail::collective_local_validation(mesh, "Low-Mach mixture step inputs",
        [&]
        {
            sources = fixed_sources;
            check(d->initialized && d->epoch == mesh_geometry_epoch(mesh) && std::isfinite(dt) && dt > 0,
                "Low-Mach step requires initialized current fixed geometry and a positive timestep.");
            check(d->accepted.revision != std::numeric_limits<std::uint64_t>::max() &&
                      std::isfinite(d->accepted.time + dt) && d->accepted.time + dt > d->accepted.time,
                "Low-Mach next clock/revision cannot be represented without overflow or lost advancement.");
            check(sources.empty() || sources.size() == mesh.num_owned_cells(), "Low-Mach source map differs.");
            source_stage_detail::validate_number_packets(sources, mesh.num_owned_cells(), d->options.phase_roles);
            for (const auto& source : sources)
            {
                check((source.species_moles.empty() || source.species_moles.size() == d->eos.size()) &&
                          (source.internal_energy.empty() || source.internal_energy.size() == d->eos.size()),
                    "Low-Mach source phase topology differs.");
                for (const auto& species : source.species_moles)
                {
                    check(species.size() == d->eos.front()->catalog().species().size(),
                        "Low-Mach source species topology differs.");
                    for (double value : species)
                        check(std::isfinite(value), "Nonfinite integrated species source.");
                }
                for (double value : source.internal_energy)
                    check(std::isfinite(value), "Nonfinite phase energy source.");
                for (size_t component = 0; component < 3; ++component)
                    check(
                        std::isfinite(source.momentum.component(component)), "Nonfinite integrated physical momentum.");
            }
        });
    collective_detail::require_uniform_value(mesh, int(bool(provider)), "Collective source-stage selection");
    SourceStageMetadata stage;
    auto source_options = d->options;
    int maximum_source_evaluations = d->options.maximum_correctors;
    if (provider)
    {
        collective_detail::collective_local_validation(mesh, "Immutable source-stage metadata",
            [&]
            {
                check(!provider->identity.empty() &&
                          provider->identity.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                          provider->revision > 0 &&
                          provider->schema_fingerprint == d->eos.front()->catalog().fingerprint() &&
                          bool(provider->evaluate) && d->options.maximum_correctors >= 2 &&
                          std::isfinite(provider->relative_tolerance) && provider->relative_tolerance >= 0 &&
                          (provider->maximum_evaluations == 0 || provider->maximum_evaluations >= 2),
                    "Source-stage identity/revision/schema/callback or corrector budget is invalid.");
                check(d->source_invocation != std::numeric_limits<std::uint64_t>::max(),
                    "Source invocation identity exhausted.");
            });
        collective_detail::require_uniform_value(mesh, provider->identity.size(), "Source identity length");
        collective_detail::require_uniform_value(mesh, provider->revision, "Source provider revision");
        collective_detail::require_uniform_value(mesh, provider->schema_fingerprint, "Source provider schema");
        collective_detail::require_uniform_value(mesh, provider->relative_tolerance, "Source provider relative gate");
        collective_detail::require_uniform_value(
            mesh, provider->maximum_evaluations, "Source provider evaluation limit");
        if (provider->relative_tolerance > 0)
            source_options.source_relative_tolerance = provider->relative_tolerance;
        if (provider->maximum_evaluations > 0)
            maximum_source_evaluations = std::min(maximum_source_evaluations, provider->maximum_evaluations);
        collective_detail::require_uniform_value(mesh, d->source_invocation, "Source invocation generation");
        std::string identity;
        collective_detail::collective_local_validation(
            mesh, "Source identity preparation", [&] { identity = provider->identity; });
        Teuchos::broadcast(*mesh.owned_cell_map()->getComm(), 0, static_cast<int>(identity.size()), identity.data());
        collective_detail::collective_local_validation(mesh, "Source identity text",
            [&] { check(identity == provider->identity, "Source provider identity differs across ranks."); });
        collective_detail::collective_local_validation(mesh, "Source receipt metadata",
            [&]
            {
                stage.provider_identity = provider->identity;
                stage.provider_revision = provider->revision;
            });
        // Receipt identity allocation is private. It NEVER changes the accepted
        // trial nonce, so previous same-generation Trials remain committable.
        stage.invocation = ++d->source_invocation;
    }
    const auto accepted = d->accepted;
    const size_t nc = accepted.cells.size(), np = d->eos.size(), ns = d->eos.front()->catalog().species().size();
    TransportSnapshot last_transport;
    std::vector<PhaseFaceTransport> last_phase_transport;
    std::optional<TransportSnapshot> previous_transport;
    std::vector<CellSources> previous_sources;
    std::vector<std::vector<double>> gross_events, previous_events;
    std::optional<bool> event_channel;
    d->publish_fields(accepted);
    d->old_rho.owned_data().update(1.0, d->rho.owned_data(), 0.0);
    d->old_rho.sync_ghosts();
    d->old_u.owned_data().update(1.0, d->u.owned_data(), 0.0);
    d->old_u.sync_ghosts();
    d->trial_flux.owned_data().update(1.0, d->flux.owned_data(), 0.0);
    d->trial_flux.sync_ghosts();
    std::vector<std::vector<double>> old_volumes(nc);
    collective_detail::collective_local_validation(mesh, "Accepted low-Mach caloric state",
        [&]
        {
            for (size_t c = 0; c < nc; ++c)
            {
                old_volumes[c].resize(np);
                for (size_t a = 0; a < np; ++a)
                {
                    const auto& phase = accepted.cells[c].phases[a];
                    const double mass = phase_mass(phase);
                    if (mass > 0)
                        old_volumes[c][a] =
                            mass /
                            phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase))
                                .density;
                }
            }
        });
    Diagnostics result;
    State trial = accepted;
    State thermodynamic_guess = accepted;
    Vector pressure_impulse{};
    double caloric_transport_magnitude = 0, energy_transport_operand_magnitude = 0, phase_work_magnitude = 0,
           pressure_operand_magnitude = 0;
    try
    {
        for (int iteration = 0; iteration < d->options.maximum_correctors; ++iteration)
        {
            trial = accepted;
            result = {};
            result.correctors = iteration + 1;
            pressure_impulse = {};
            double local_caloric_transport = 0, local_energy_transport = 0;
            std::vector<std::vector<double>> adv_volumes(nc, std::vector<double>(np));
            std::vector<std::unique_ptr<ScalarField>> transported_species(np * ns);
            std::vector<std::unique_ptr<FaceField>> material_face_fraction(np);
            std::vector<std::unique_ptr<VectorField>> drift(np);
            std::vector<std::unique_ptr<FaceField>> phase_flux(np);
            std::vector<double> before(nc), after;
            for (size_t a = 0; a < np; ++a)
            {
                drift[a] = std::make_unique<VectorField>(d->mesh, "phase_drift");
                phase_flux[a] = std::make_unique<FaceField>(d->mesh, 0.0, "phase_carrier_slip_flux");
                for (size_t c = 0; c < nc; ++c)
                    drift[a]->set_owned_value(static_cast<LO>(c), accepted.cells[c].drift_velocity[a]);
                drift[a]->sync_ghosts();
                for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                {
                    const auto face = static_cast<LO>(f);
                    if (mesh.is_boundary_face(face))
                        continue;
                    const auto owner = mesh.owner_cell(face),
                               neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                    const auto slip = (drift[a]->local_value(owner) + drift[a]->local_value(neighbor)) * 0.5;
                    phase_flux[a]->set_owned_value(
                        face, d->trial_flux.local_value(face) + slip.dot(mesh.face_area_vector(face)));
                }
                phase_flux[a]->sync_ghosts();
                if (d->options.phase_roles[a].role == PhaseRole::DispersedGas)
                {
                    collective_detail::collective_local_validation(mesh, "Population transport input",
                        [&]
                        {
                            for (size_t c = 0; c < nc; ++c)
                            {
                                before[c] = *accepted.cells[c].phase_numbers[a];
                                check(std::isfinite(before[c] / mesh.cell_volume(static_cast<LO>(c))),
                                    "Population number concentration is not representable.");
                            }
                        });
                    d->transport(before, *phase_flux[a], dt, after, result);

                    ScalarField number_concentration(d->mesh, 0.0, "transported_population_number");
                    FaceField number_flux(d->mesh, 0.0, "conservative_population_number_flux");
                    collective_detail::collective_local_validation(mesh, "Transported population numbers",
                        [&]
                        {
                            for (size_t c = 0; c < nc; ++c)
                            {
                                check(std::isfinite(after[c]) && after[c] >= 0,
                                    "Population transport produced a negative/nonfinite number.");
                                trial.cells[c].phase_numbers[a] = after[c];
                                const double concentration = after[c] / mesh.cell_volume(static_cast<LO>(c));
                                check(std::isfinite(concentration),
                                    "Transported population concentration is not representable.");
                                number_concentration.set_owned_value(static_cast<LO>(c), concentration);
                            }
                        });
                    number_concentration.sync_ghosts();
                    collective_detail::collective_local_validation(mesh, "Canonical population number flux",
                        [&]
                        {
                            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                            {
                                const LO face = static_cast<LO>(f);
                                if (mesh.is_geometry_exterior_face(face))
                                    continue;
                                const double phi = phase_flux[a]->local_value(face);
                                const LO owner = mesh.owner_cell(face),
                                         donor =
                                             phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                                const double flux = phi * number_concentration.local_value(donor);
                                check(std::isfinite(flux), "Population number face flux overflow.");
                                number_flux.set_owned_value(face, flux);
                            }
                        });
                    number_flux.sync_ghosts();
                    double local_number_error = 0;
                    collective_detail::collective_local_validation(mesh, "Physical local population balance",
                        [&]
                        {
                            for (size_t c = 0; c < nc; ++c)
                            {
                                const long double flux = static_cast<long double>(dt) *
                                                         FVM::cell_flux_balance(mesh, number_flux, static_cast<LO>(c));
                                const long double residual = static_cast<long double>(after[c]) - before[c] + flux;
                                local_number_error = std::max(local_number_error,
                                    source_stage_detail::normalized(residual,
                                        std::abs(static_cast<long double>(after[c])) +
                                            std::abs(static_cast<long double>(before[c])) + std::abs(flux),
                                        d->options.conservation_relative_tolerance));
                            }
                        });
                    const double count_error = d->maximum(local_number_error);
                    result.maximum_population_transport_residual =
                        std::max(result.maximum_population_transport_residual, count_error);
                    check(count_error <= 1, "Population transport fails its physical local count-balance gate.");
                }
                for (size_t s = 0; s < ns; ++s)
                {
                    for (size_t c = 0; c < nc; ++c)
                        before[c] = accepted.cells[c].phases[a].moles[s];
                    d->transport(before, *phase_flux[a], dt, after, result);
                    auto field = std::make_unique<ScalarField>(d->mesh, "transported_species_concentration");
                    for (size_t c = 0; c < nc; ++c)
                    {
                        trial.cells[c].phases[a].moles[s] = after[c];
                        field->set_owned_value(static_cast<LO>(c), after[c] / mesh.cell_volume(static_cast<LO>(c)));
                    }
                    field->sync_ghosts();
                    transported_species[a * ns + s] = std::move(field);
                }
                for (size_t c = 0; c < nc; ++c)
                    before[c] = accepted.cells[c].phases[a].internal_energy;
                d->transport(before, *phase_flux[a], dt, after, result);
                for (size_t c = 0; c < nc; ++c)
                    trial.cells[c].phases[a].internal_energy = after[c];
                ScalarField transported_energy_density(d->mesh, "transported_energy_operand");
                ScalarField caloric_carrier_density(d->mesh, "caloric_transport_characteristic");
                collective_detail::collective_local_validation(mesh, "Low-Mach caloric transport scales",
                    [&]
                    {
                        for (size_t c = 0; c < nc; ++c)
                        {
                            const LO cell = static_cast<LO>(c);
                            transported_energy_density.set_owned_value(cell, after[c] / mesh.cell_volume(cell));
                            const auto& old_phase = accepted.cells[c].phases[a];
                            const double mass = phase_mass(old_phase);
                            double caloric = 0;
                            if (mass > 0)
                                caloric = mass * old_phase.temperature *
                                          old_phase.thermodynamics
                                              ->evaluate(old_phase.temperature, old_phase.absolute_pressure,
                                                  fractions(old_phase))
                                              .internal_energy_temperature;
                            caloric_carrier_density.set_owned_value(cell, caloric / mesh.cell_volume(cell));
                        }
                    });
                transported_energy_density.sync_ghosts();
                caloric_carrier_density.sync_ghosts();
                for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                {
                    const LO face = static_cast<LO>(f);
                    if (mesh.is_boundary_face(face))
                        continue;
                    const double phi = phase_flux[a]->local_value(face);
                    const LO owner = mesh.owner_cell(face);
                    const LO donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                    local_energy_transport += dt * std::abs(phi * transported_energy_density.local_value(donor));
                    local_caloric_transport += dt * std::abs(phi * caloric_carrier_density.local_value(donor));
                }
                // Reconstruct material-volume packets from the same transported species and upwind trial EOS.
                // A separately advected passive volume label would introduce a CFL-squared occupancy defect.
                ScalarField phase_temperature(d->mesh, "phase_volume_trace_temperature");
                ScalarField phase_pressure(d->mesh, "phase_volume_trace_pressure");
                material_face_fraction[a] = std::make_unique<FaceField>(d->mesh, 0.0, "phase_material_face_fraction");
                FaceField material_volume_flux(d->mesh, 0.0, "phase_material_volume_flux");
                collective_detail::collective_local_validation(mesh, "Phase material EOS traces",
                    [&]
                    {
                        for (size_t c = 0; c < nc; ++c)
                        {
                            const LO cell = static_cast<LO>(c);
                            const auto& guess_phase = thermodynamic_guess.cells[c].phases[a];
                            const double pressure = thermodynamic_guess.thermodynamic_pressure +
                                                    thermodynamic_guess.cells[c].mechanical_pressure;
                            double temperature = guess_phase.temperature;
                            if (!(temperature > 0) && phase_mass(trial.cells[c].phases[a]) > 0)
                            {
                                auto arriving = trial.cells[c].phases[a];
                                arriving.absolute_pressure = pressure;
                                temperature = recover_phase_temperatures(std::vector{arriving},
                                    arriving.internal_energy, PhaseThermalMode::SeparatePhaseTemperatures)
                                                  .phases[0]
                                                  .temperature;
                            }
                            phase_temperature.set_owned_value(cell, temperature);
                            phase_pressure.set_owned_value(cell, pressure);
                        }
                    });
                phase_temperature.sync_ghosts();
                phase_pressure.sync_ghosts();
                collective_detail::collective_local_validation(mesh, "Canonical phase material-volume packets",
                    [&]
                    {
                        for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                        {
                            const LO face = static_cast<LO>(f);
                            if (mesh.is_boundary_face(face))
                                continue;
                            const double phi = phase_flux[a]->local_value(face);
                            const LO owner = mesh.owner_cell(face);
                            const LO donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                            std::vector<double> x(ns);
                            double amount = 0, transported_mass_density = 0;
                            for (size_t s = 0; s < ns; ++s)
                            {
                                x[s] = transported_species[a * ns + s]->local_value(donor);
                                amount += x[s];
                                transported_mass_density += x[s] * d->eos[a]->catalog().species()[s].molarMass;
                            }
                            if (amount == 0)
                                continue;
                            for (double& value : x)
                                value /= amount;
                            const auto point = d->eos[a]->evaluate(
                                phase_temperature.local_value(donor), phase_pressure.local_value(donor), x);
                            const double fraction = transported_mass_density / point.density;
                            material_face_fraction[a]->set_owned_value(face, fraction);
                            material_volume_flux.set_owned_value(face, phi * fraction);
                        }
                    });
                material_face_fraction[a]->sync_ghosts();
                material_volume_flux.sync_ghosts();
                for (size_t c = 0; c < nc; ++c)
                    adv_volumes[c][a] =
                        old_volumes[c][a] - dt * FVM::cell_flux_balance(mesh, material_volume_flux, static_cast<LO>(c));
            }
            caloric_transport_magnitude = d->sum(local_caloric_transport);
            energy_transport_operand_magnitude = d->sum(local_energy_transport);
            stage.corrector = static_cast<size_t>(iteration);
            collective_detail::collective_local_validation(mesh, "Immutable transport snapshot",
                [&]
                {
                    for (const auto& cell : trial.cells)
                        source_stage_detail::validate_phase_numbers(cell, d->options.phase_roles);
                    last_transport = {};
                    last_transport.cells.resize(nc);
                    last_transport.old_control_volumes.resize(nc);
                    last_transport.new_control_volumes.resize(nc);
                    last_transport.accepted_revision = accepted.revision;
                    last_transport.accepted_generation = d->trial_nonce;
                    last_transport.old_geometry_epoch = d->epoch;
                    last_transport.new_geometry_epoch = d->epoch;
                    last_transport.schema_fingerprint = d->eos.front()->catalog().fingerprint();
                    last_transport.accepted_time = accepted.time;
                    last_transport.time_step = dt;
                    last_transport.stage = stage;
                    last_phase_transport.assign(mesh.num_owned_faces(), {});
                    for (size_t f = 0; f < last_phase_transport.size(); ++f)
                    {
                        const LO face = static_cast<LO>(f);
                        auto& packet = last_phase_transport[f];
                        packet.carrier_volume_flux = d->trial_flux.local_value(face);
                        packet.phase_volume_flux.resize(np);
                        packet.material_fraction.resize(np);
                        check(std::isfinite(packet.carrier_volume_flux), "Nonfinite stage carrier flux receipt.");
                        for (size_t a = 0; a < np; ++a)
                        {
                            packet.phase_volume_flux[a] = phase_flux[a]->local_value(face);
                            packet.material_fraction[a] = material_face_fraction[a]->local_value(face);
                            check(std::isfinite(packet.phase_volume_flux[a]) &&
                                      std::isfinite(packet.material_fraction[a]) && packet.material_fraction[a] >= 0,
                                "Invalid exact phase-face transport receipt.");
                        }
                    }
                    for (size_t c = 0; c < nc; ++c)
                    {
                        last_transport.old_control_volumes[c] = mesh.cell_volume(static_cast<LO>(c));
                        last_transport.new_control_volumes[c] = last_transport.old_control_volumes[c];
                        last_transport.cells[c].phases.resize(np);
                        for (size_t a = 0; a < np; ++a)
                        {
                            auto& inventory = last_transport.cells[c].phases[a];
                            inventory.moles = trial.cells[c].phases[a].moles;
                            inventory.internal_energy = trial.cells[c].phases[a].internal_energy;
                            inventory.advected_material_volume = adv_volumes[c][a];
                            inventory.population_number = trial.cells[c].phase_numbers[a];
                        }
                    }
                });
            if (provider)
            {
                SourceStageResult evaluated;
                std::exception_ptr callback_error;
                try
                {
                    evaluated = provider->evaluate(SourceStageContext{last_transport, thermodynamic_guess, mesh});
                }
                catch (...)
                {
                    callback_error = std::current_exception();
                }
                collective_detail::collective_local_validation(mesh, "Source-stage returned packets",
                    [&]
                    {
                        if (callback_error)
                            std::rethrow_exception(callback_error);
                        source_stage_detail::validate_packets(evaluated.sources, nc, np, ns);
                        source_stage_detail::validate_events(evaluated.gross_events, nc);
                        source_stage_detail::validate_number_packets(evaluated.sources, nc, d->options.phase_roles);
                    });
                int local_events = !evaluated.gross_events.empty(), global_events = 0;
                Teuchos::reduceAll(
                    *mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_events, &global_events);
                collective_detail::collective_local_validation(mesh, "Source-stage event schema",
                    [&]
                    {
                        check(!global_events || evaluated.gross_events.size() == nc,
                            "Enabled gross-event channel requires one event vector per owned cell.");
                        check(!event_channel || *event_channel == bool(global_events),
                            "Source stage changed its gross-event channel between correctors.");
                        if (previous_transport)
                            source_stage_detail::validate_event_shape(evaluated.gross_events, previous_events);
                    });
                event_channel = bool(global_events);
                sources = std::move(evaluated.sources);
                gross_events = std::move(evaluated.gross_events);
                double event_residual = global_events ? std::numeric_limits<double>::infinity() : 0;
                double source_residual = std::numeric_limits<double>::infinity();
                double transport_residual = std::numeric_limits<double>::infinity();
                if (previous_transport)
                {
                    collective_detail::collective_local_validation(mesh, "Source/transport convergence",
                        [&]
                        {
                            event_residual = source_stage_detail::event_error(
                                gross_events, previous_events, source_options.source_relative_tolerance);
                            source_residual = source_stage_detail::source_error(sources, previous_sources,
                                last_transport, thermodynamic_guess, accepted, source_options);
                            transport_residual = source_stage_detail::transport_error(
                                last_transport, *previous_transport, thermodynamic_guess, accepted, d->options);
                        });
                }
                result.source_evaluations = iteration + 1;
                result.maximum_event_residual = d->maximum(event_residual);
                result.maximum_source_residual = d->maximum(source_residual);
                result.maximum_transport_residual = d->maximum(transport_residual);
                collective_detail::collective_local_validation(mesh, "Retained source corrector receipt",
                    [&]
                    {
                        previous_transport = last_transport;
                        previous_sources = sources;
                        previous_events = gross_events;
                    });
            }
            collective_detail::collective_local_validation(mesh, "Low-Mach integrated local sources",
                [&]
                {
                    for (size_t c = 0; c < nc; ++c)
                        for (size_t a = 0; a < np; ++a)
                        {
                            auto& phase = trial.cells[c].phases[a];
                            for (size_t s = 0; s < ns; ++s)
                            {
                                if (!sources.empty() && !sources[c].species_moles.empty())
                                    phase.moles[s] += sources[c].species_moles[a][s];
                                check(std::isfinite(phase.moles[s]) && phase.moles[s] >= 0,
                                    "Low-Mach source/transport overdraws phase species.");
                            }
                            if (!sources.empty() && !sources[c].population_number_changes.empty() &&
                                trial.cells[c].phase_numbers[a])
                                *trial.cells[c].phase_numbers[a] += sources[c].population_number_changes[a];
                            if (!sources.empty() && !sources[c].internal_energy.empty())
                                phase.internal_energy += sources[c].internal_energy[a];
                        }
                });
            collective_detail::collective_local_validation(mesh, "Joint phase mole/number source admissibility",
                [&]
                {
                    for (const auto& cell : trial.cells)
                        source_stage_detail::validate_phase_numbers(cell, d->options.phase_roles);
                });
            std::vector<CaloricCell> thermal(nc);
            double local_domain_volume = 0;
            double local_lower = d->options.minimum_pressure, local_upper = d->options.maximum_pressure;
            collective_detail::collective_local_validation(mesh, "Low-Mach pressure validity interval",
                [&]
                {
                    for (size_t c = 0; c < nc; ++c)
                    {
                        local_domain_volume += mesh.cell_volume(static_cast<LO>(c));
                        trial.cells[c].mechanical_pressure = d->mechanical_pressure.local_value(static_cast<LO>(c));
                        for (const auto& phase : trial.cells[c].phases)
                            if (phase_mass(phase) > 0)
                            {
                                local_lower = std::max(local_lower, phase.thermodynamics->bounds().minimum_pressure -
                                                                        trial.cells[c].mechanical_pressure);
                                local_upper = std::min(local_upper, phase.thermodynamics->bounds().maximum_pressure -
                                                                        trial.cells[c].mechanical_pressure);
                            }
                    }
                });
            double lower = 0, upper = 0;
            Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_lower, &lower);
            Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_upper, &upper);
            check(std::isfinite(lower) && std::isfinite(upper) && lower <= upper,
                "Low-Mach phases have no common supplied pressure interval.");
            double p0 = std::clamp(thermodynamic_guess.thermodynamic_pressure, lower, upper);
            double last_feasible_pressure = p0;
            bool have_feasible_pressure = false;
            const double domain_volume = d->sum(local_domain_volume);
            // Only an actual initial CaloricRangeError enters discovery. Source
            // callbacks are NOT evaluated during these inner pressure trials.
            const auto discover_seed = [&](const std::exception_ptr& initial_error)
            {
                double local_cert_lower = lower, local_cert_upper = upper;
                double local_search_lower = lower, local_search_upper = upper;
                int local_empty = 0;
                std::vector<unsigned char> supplied_certification;
                collective_detail::collective_local_validation(mesh, "Local caloric domain discovery",
                    [&]
                    {
                        supplied_certification.assign(nc, 0);
                        for (size_t c = 0; c < nc; ++c)
                        {
                            std::vector<double> offsets(np, trial.cells[c].mechanical_pressure);
                            const CaloricPressureQuery query{trial.cells[c].phases, adv_volumes[c], offsets,
                                {lower, upper}, d->options.thermal_mode};
                            auto domain = exact_caloric_pressure_domain(query);
                            bool supplied = false;
                            if (domain.knowledge == CaloricDomainKnowledge::Unresolved &&
                                d->options.caloric_domain_provider)
                            {
                                // MPI-free supplied hook, fenced before global interval work.
                                domain = d->options.caloric_domain_provider->evaluate(query);
                                supplied = true;
                            }
                            check(domain.knowledge == CaloricDomainKnowledge::CertifiedConnected ||
                                      domain.knowledge == CaloricDomainKnowledge::CertifiedEmpty ||
                                      domain.knowledge == CaloricDomainKnowledge::Unresolved,
                                "Invalid supplied caloric domain knowledge.");
                            if (domain.knowledge == CaloricDomainKnowledge::CertifiedEmpty)
                            {
                                check(!domain.interval, "A certified empty caloric domain cannot carry an interval.");
                                local_empty = 1;
                                continue;
                            }
                            check(domain.knowledge != CaloricDomainKnowledge::CertifiedConnected ||
                                      domain.interval.has_value(),
                                "A connected caloric certificate requires an interval.");
                            if (!domain.interval)
                                continue;
                            const auto interval = *domain.interval;
                            check(std::isfinite(interval.lower) && std::isfinite(interval.upper) &&
                                      interval.lower >= lower && interval.upper <= upper &&
                                      interval.lower <= interval.upper,
                                "Supplied caloric interval is outside query bounds.");
                            local_search_lower = std::max(local_search_lower, interval.lower);
                            local_search_upper = std::min(local_search_upper, interval.upper);
                            if (domain.knowledge == CaloricDomainKnowledge::CertifiedConnected)
                            {
                                local_cert_lower = std::max(local_cert_lower, interval.lower);
                                local_cert_upper = std::min(local_cert_upper, interval.upper);
                                supplied_certification[c] = supplied ? 1 : 0;
                            }
                        }
                    });
                int certified_empty = 0;
                double cert_lower = 0, cert_upper = 0, search_lower = 0, search_upper = 0;
                const auto comm = mesh.owned_cell_map()->getComm();
                Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_empty, &certified_empty);
                Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_cert_lower, &cert_lower);
                Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_cert_upper, &cert_upper);
                Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_search_lower, &search_lower);
                Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_search_upper, &search_upper);
                if (certified_empty || cert_lower > cert_upper)
                {
                    // Preserve the actual initial caloric error on its originating
                    // rank; peers receive the ordinary collective error context.
                    collective_detail::collective_local_validation(mesh, "Initial caloric domain certified empty",
                        [&]
                        {
                            if (initial_error)
                                std::rethrow_exception(initial_error);
                        });
                    throw CaloricRangeError("Initial caloric domain certified empty without a retained range error.");
                }
                lower = cert_lower;
                upper = cert_upper;
                if (search_lower > search_upper)
                    throw CaloricDomainUnresolved(
                        "Supplied uncertified caloric search brackets have no common interval.");
                std::vector<CaloricPressureInterval> pending;
                collective_detail::collective_local_validation(
                    mesh, "Caloric seed-search preparation", [&] { pending.push_back({search_lower, search_upper}); });
                for (int attempt = 0; attempt < d->options.maximum_caloric_seed_evaluations; ++attempt)
                {
                    double candidate = 0;
                    int candidate_available = 1;
                    collective_detail::collective_local_validation(mesh, "Caloric seed candidate",
                        [&]
                        {
                            if (attempt == 1)
                            {
                                candidate = search_lower;
                                return;
                            }
                            if (attempt == 2)
                            {
                                candidate = search_upper;
                                return;
                            }
                            if (pending.empty())
                            {
                                candidate_available = 0;
                                return;
                            }
                            const auto next = std::max_element(pending.begin(), pending.end(),
                                [](const auto& a, const auto& b) { return a.upper - a.lower < b.upper - b.lower; });
                            const auto interval = *next;
                            pending.erase(next);
                            candidate = std::midpoint(interval.lower, interval.upper);
                            if (candidate != interval.lower && candidate != interval.upper)
                            {
                                pending.push_back({interval.lower, candidate});
                                pending.push_back({candidate, interval.upper});
                            }
                        });
                    if (!candidate_available)
                        break;
                    int local_outside = 0;
                    collective_detail::collective_local_validation(mesh, "Caloric seed evaluation",
                        [&]
                        {
                            for (size_t c = 0; c < nc; ++c)
                                try
                                {
                                    thermal[c] =
                                        recover_enthalpy(trial.cells[c], adv_volumes[c], candidate, d->options);
                                }
                                catch (const CaloricRangeError&)
                                {
                                    check(!supplied_certification[c],
                                        "Supplied connected caloric certificate contradicts actual EOS recovery.");
                                    local_outside = 1;
                                }
                        });
                    int outside = 0;
                    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_outside, &outside);
                    if (!outside)
                        return candidate;
                }
                // Finite samples cannot rule out a narrow/disconnected branch.
                throw CaloricDomainUnresolved(
                    "Initial caloric domain unresolved within the bounded seed-search budget.");
            };
            for (int pressure_iteration = 0;; ++pressure_iteration)
            {
                double local_volume = 0, local_derivative = 0;
                int local_outside_caloric_range = 0;
                std::exception_ptr local_caloric_error;
                collective_detail::collective_local_validation(mesh, "Low-Mach p0 caloric closure",
                    [&]
                    {
                        for (size_t c = 0; c < nc; ++c)
                        {
                            try
                            {
                                thermal[c] = recover_enthalpy(trial.cells[c], adv_volumes[c], p0, d->options);
                                local_volume += thermal[c].volume;
                                local_derivative += thermal[c].derivative;
                            }
                            catch (const CaloricRangeError&)
                            {
                                // Only this known validity-range condition is a
                                // search signal. Provider/derivative errors escape.
                                local_outside_caloric_range = 1;
                                if (!local_caloric_error)
                                    local_caloric_error = std::current_exception();
                            }
                        }
                    });
                int outside_caloric_range = 0;
                Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
                    &local_outside_caloric_range, &outside_caloric_range);
                if (outside_caloric_range)
                {
                    if (!have_feasible_pressure)
                    {
                        p0 = discover_seed(local_caloric_error);
                        continue;
                    }
                    check(pressure_iteration < d->options.maximum_pressure_iterations,
                        "Low-Mach caloric pressure backtracking did not converge.");
                    const double next = std::midpoint(p0, last_feasible_pressure);
                    check(next != p0 && next != last_feasible_pressure,
                        "Low-Mach caloric pressure interval cannot be resolved.");
                    p0 = next;
                    continue;
                }
                last_feasible_pressure = p0;
                have_feasible_pressure = true;
                const double residual = d->sum(local_volume) - domain_volume;
                if (std::abs(residual) <= std::min(d->options.volume_relative_tolerance * domain_volume,
                                              0.1 * d->options.continuity_tolerance * dt))
                    break;
                check(pressure_iteration < d->options.maximum_pressure_iterations,
                    "Uniform low-Mach pressure closure did not converge.");
                const double derivative = d->sum(local_derivative);
                check(std::isfinite(derivative) && derivative < 0,
                    "Low-Mach mixture has unstable pressure susceptibility.");
                if (residual > 0)
                    lower = std::max(lower, p0);
                else
                    upper = std::min(upper, p0);
                double next = p0 - residual / derivative;
                if (!std::isfinite(next) || next <= lower || next >= upper)
                    next = std::midpoint(lower, upper);
                check(next != p0 && next >= lower && next <= upper,
                    "Low-Mach bounded pressure interval cannot resolve physical volume closure.");
                p0 = next;
            }
            trial.thermodynamic_pressure = p0;
            std::vector<double> target(nc), volume_susceptibility(nc);
            FaceField slip_volume(d->mesh, 0.0, "phase_slip_material_volume");
            FaceField carrier_fraction(d->mesh, 0.0, "phase_upwind_carrier_fraction");
            std::array<std::unique_ptr<FaceField>, 3> drift_momentum;
            for (auto& field : drift_momentum)
                field = std::make_unique<FaceField>(d->mesh, 0.0, "phase_slip_momentum_flux");
            collective_detail::collective_local_validation(mesh, "Phase mass/material momentum packets",
                [&]
                {
                    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                    {
                        const LO face = static_cast<LO>(f);
                        if (mesh.is_boundary_face(face))
                        {
                            d->mass_flux.set_owned_value(face, 0);
                            continue;
                        }
                        const LO owner = mesh.owner_cell(face),
                                 neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                        double mass_flux = 0, volume_slip = 0, theta = 0;
                        Vector drift_flux{};
                        for (size_t a = 0; a < np; ++a)
                        {
                            const double phi = phase_flux[a]->local_value(face);
                            const LO donor = phi >= 0 ? owner : neighbor;
                            double density = 0;
                            for (size_t s = 0; s < ns; ++s)
                                density += transported_species[a * ns + s]->local_value(donor) *
                                           d->eos[a]->catalog().species()[s].molarMass;
                            const double mf = phi * density;
                            mass_flux += mf;
                            drift_flux = drift_flux + drift[a]->local_value(donor) * mf;
                            const double slip = phi - d->trial_flux.local_value(face);
                            const double phase_fraction = material_face_fraction[a]->local_value(face);
                            theta += phase_fraction;
                            volume_slip += slip * phase_fraction;
                        }
                        check(std::isfinite(mass_flux) && std::isfinite(volume_slip) && std::isfinite(theta) &&
                                  theta >= 0,
                            "Phase mass/material flux packet overflow.");
                        d->mass_flux.set_owned_value(face, mass_flux);
                        slip_volume.set_owned_value(face, volume_slip);
                        carrier_fraction.set_owned_value(face, theta);
                        for (size_t component = 0; component < 3; ++component)
                            drift_momentum[component]->set_owned_value(face, drift_flux.component(component));
                    }
                });
            d->mass_flux.sync_ghosts();
            slip_volume.sync_ghosts();
            carrier_fraction.sync_ghosts();
            for (auto& field : drift_momentum)
                field->sync_ghosts();
            double occupancy = 0, local_work = 0, local_roundoff = 0, local_work_magnitude = 0,
                   local_pressure_operands = 0;
            collective_detail::collective_local_validation(mesh, "Low-Mach EOS transport candidate",
                [&]
                {
                    for (size_t c = 0; c < nc; ++c)
                    {
                        const LO cell = static_cast<LO>(c);
                        const double volume = mesh.cell_volume(cell);
                        trial.cells[c] = thermal[c].cell;
                        const double advected_volume =
                            std::accumulate(adv_volumes[c].begin(), adv_volumes[c].end(), 0.0);
                        volume_susceptibility[c] = -thermal[c].derivative;
                        check(std::isfinite(volume_susceptibility[c]) && volume_susceptibility[c] >= 0,
                            "Low-Mach local EOS volume susceptibility is not stable and nonnegative.");
                        target[c] = (thermal[c].volume - advected_volume) / dt;
                        occupancy = std::max(occupancy, std::abs(thermal[c].volume - volume) / volume);
                        local_work += thermal[c].work;
                        local_work_magnitude += thermal[c].work_magnitude;
                        local_pressure_operands += thermal[c].pressure_operands;
                        local_roundoff += thermal[c].roundoff;
                        double mass = 0;
                        for (const auto& phase : trial.cells[c].phases)
                            mass += phase_mass(phase);
                        d->rho.set_owned_value(cell, mass / volume);
                        double viscosity = 0;
                        if (d->options.mixture_dynamic_viscosity)
                            viscosity = d->options.mixture_dynamic_viscosity(trial.cells[c].phases, thermal[c].volumes);
                        else
                        {
                            size_t populated = 0;
                            for (size_t a = 0; a < np; ++a)
                                if (thermal[c].volumes[a] > 0)
                                {
                                    ++populated;
                                    const auto& phase = trial.cells[c].phases[a];
                                    const double value =
                                        phase.thermodynamics
                                            ->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase))
                                            .dynamic_viscosity;
                                    if (populated == 1)
                                        viscosity = value;
                                    else
                                        check(value == 0 && viscosity == 0,
                                            "Multiphase viscous flow requires a supplied mixture-viscosity closure.");
                                }
                        }
                        check(std::isfinite(viscosity) && viscosity >= 0, "Invalid supplied mixture viscosity.");
                        d->mu.set_owned_value(cell, viscosity);
                    }
                });
            d->rho.sync_ghosts();
            d->mu.sync_ghosts();
            result.pressure_work = d->sum(local_work);
            phase_work_magnitude = d->sum(local_work_magnitude);
            pressure_operand_magnitude = d->sum(local_pressure_operands);
            result.empty_phase_energy_roundoff = d->sum(local_roundoff);
            FVM::cell_gradient(d->mechanical_pressure, BoundaryConditionMap{}, d->pressure_gradient, d->gradient_cache,
                FVM::CellGradientScheme::LeastSquares);
            d->pressure_gradient.sync_ghosts();
            result.momentum_linear = d->momentum.advance(
                d->old_u, d->old_rho, d->rho, d->mass_flux, d->mu, dt, d->u,
                [&](LO cell)
                {
                    Vector old_drift{}, new_drift{};
                    for (size_t a = 0; a < np; ++a)
                    {
                        old_drift = old_drift +
                                    accepted.cells[cell].drift_velocity[a] * phase_mass(accepted.cells[cell].phases[a]);
                        new_drift =
                            new_drift + trial.cells[cell].drift_velocity[a] * phase_mass(trial.cells[cell].phases[a]);
                    }
                    Vector force = d->pressure_gradient.local_value(cell) * -1.0 -
                                   (new_drift - old_drift) / (dt * mesh.cell_volume(cell));
                    if (!sources.empty())
                        force = force + sources[cell].momentum / (dt * mesh.cell_volume(cell));
                    for (size_t component = 0; component < 3; ++component)
                        force.component(component) -=
                            FVM::cell_flux_balance(mesh, *drift_momentum[component], cell) / mesh.cell_volume(cell);
                    return force;
                },
                {}, {}, d->options.linear);
            FaceField predictor(d->mesh, 0.0, "low_mach_velocity_predictor_flux");
            std::vector<double> previous_flux(mesh.num_owned_faces());
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                previous_flux[f] = d->trial_flux.local_value(face);
                if (mesh.is_boundary_face(face))
                    continue;
                const LO owner = mesh.owner_cell(face), neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                auto velocity = (d->u.local_value(owner) + d->u.local_value(neighbor)) * 0.5;
                velocity = velocity + (d->pressure_gradient.local_value(owner) / d->rho.local_value(owner) +
                                          d->pressure_gradient.local_value(neighbor) / d->rho.local_value(neighbor)) *
                                          (0.5 * dt);
                const double pressure_flux =
                    dt * 2 / (d->rho.local_value(owner) + d->rho.local_value(neighbor)) * mesh.face_area(face) /
                    mesh.face_cell_center_distance(face) *
                    (d->mechanical_pressure.local_value(neighbor) - d->mechanical_pressure.local_value(owner));
                predictor.set_owned_value(face, velocity.dot(mesh.face_area_vector(face)) - pressure_flux);
            }
            predictor.sync_ghosts();
            std::vector<Vector> predicted_velocity(nc);
            for (size_t c = 0; c < nc; ++c)
            {
                predicted_velocity[c] = d->u.local_value(static_cast<LO>(c));
                pressure_impulse = pressure_impulse - d->pressure_gradient.local_value(static_cast<LO>(c)) *
                                                          (dt * mesh.cell_volume(static_cast<LO>(c)));
            }
            const auto projected = d->pressure.project_material_with_eos(d->rho, dt, target, volume_susceptibility,
                carrier_fraction, slip_volume, predictor, d->trial_flux, d->mechanical_pressure, d->u,
                d->options.linear, d->options.continuity_tolerance);
            trial.thermodynamic_pressure += projected.thermodynamic_pressure_increment;
            check(std::isfinite(trial.thermodynamic_pressure), "Uniform pressure gauge increment overflow.");
            result.maximum_continuity_residual = projected.maximum_continuity_residual;
            result.maximum_occupancy_error = d->maximum(occupancy);
            double flux_change = 0, smallest_volume = std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < nc; ++c)
            {
                const LO cell = static_cast<LO>(c);
                trial.cells[c].velocity = d->u.local_value(cell);
                trial.cells[c].mechanical_pressure = d->mechanical_pressure.local_value(cell);
                pressure_impulse = pressure_impulse + (d->u.local_value(cell) - predicted_velocity[c]) *
                                                          (d->rho.local_value(cell) * mesh.cell_volume(cell));
                smallest_volume = std::min(smallest_volume, mesh.cell_volume(cell));
            }
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                flux_change =
                    std::max(flux_change, std::abs(d->trial_flux.local_value(static_cast<LO>(f)) - previous_flux[f]));
            double global_smallest_volume = 0;
            Teuchos::reduceAll(
                *mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &smallest_volume, &global_smallest_volume);
            flux_change = d->maximum(flux_change) * dt / global_smallest_volume;
            double thermal_residual = 0;
            collective_detail::collective_local_validation(mesh, "Pressure/thermal corrector convergence",
                [&] { thermal_residual = source_stage_detail::thermal_error(trial, thermodynamic_guess, d->options); });
            result.maximum_thermal_residual = d->maximum(thermal_residual);
            thermodynamic_guess = trial;
            if (result.maximum_occupancy_error <= d->options.volume_relative_tolerance &&
                flux_change <= d->options.volume_relative_tolerance &&
                result.maximum_continuity_residual <=
                    d->options.continuity_tolerance * std::max(1.0, projected.continuity_scale) &&
                result.maximum_thermal_residual <= 1 &&
                (!provider ||
                    (iteration >= 1 && result.maximum_source_residual <= 1 && result.maximum_transport_residual <= 1 &&
                        result.maximum_thermal_residual <= 1 && result.maximum_event_residual <= 1)))
            {
                collective_detail::collective_local_validation(mesh, "Accepted low-Mach uniform pressure",
                    [&]
                    {
                        check(trial.thermodynamic_pressure >= d->options.minimum_pressure &&
                                  trial.thermodynamic_pressure <= d->options.maximum_pressure,
                            "Accepted uniform pressure after mechanical gauge transfer is outside its declared "
                            "bracket.");
                    });
                break;
            }
            if (provider && iteration + 1 >= maximum_source_evaluations)
                throw std::runtime_error(
                    "Low-Mach source/event/transport/thermal correctors exhausted the provider evaluation limit.");
            if (iteration + 1 >= d->options.maximum_correctors)
            {
                std::ostringstream message;
                message.precision(17);
                message << "Low-Mach material-volume/pressure correctors did not converge: occupancy="
                        << result.maximum_occupancy_error << ", normalized flux change=" << flux_change;
                throw std::runtime_error(message.str());
            }
        }
        double old_mass = 0, new_mass = 0, external_mass = 0;
        detail::CompensatedSum<long double> old_energy, new_energy, external_energy;
        detail::CompensatedSum<long double> caloric_storage_scale, caloric_source_scale, energy_operand_scale;
        std::vector<double> old_species(ns), new_species(ns), source_species(ns);
        Vector old_momentum{}, new_momentum{}, external_momentum{}, momentum_operand_scale{};
        collective_detail::collective_local_validation(mesh, "Low-Mach physical caloric ledger",
            [&]
            {
                for (size_t c = 0; c < nc; ++c)
                {
                    double accepted_mass = 0, candidate_mass = 0;
                    detail::CompensatedSum<long double> cell_source_caloric;
                    for (size_t a = 0; a < np; ++a)
                    {
                        const double ma = phase_mass(accepted.cells[c].phases[a]),
                                     mn = phase_mass(trial.cells[c].phases[a]);
                        for (size_t component = 0; component < 3; ++component)
                            momentum_operand_scale.component(component) +=
                                ma * (std::abs(accepted.cells[c].velocity.component(component)) +
                                         std::abs(accepted.cells[c].drift_velocity[a].component(component))) +
                                mn * (std::abs(trial.cells[c].velocity.component(component)) +
                                         std::abs(trial.cells[c].drift_velocity[a].component(component)));
                        old_mass += ma;
                        new_mass += mn;
                        accepted_mass += ma;
                        candidate_mass += mn;
                        old_momentum = old_momentum + accepted.cells[c].drift_velocity[a] * ma;
                        new_momentum = new_momentum + trial.cells[c].drift_velocity[a] * mn;
                        const auto& old_phase = accepted.cells[c].phases[a];
                        const auto& new_phase = trial.cells[c].phases[a];
                        if (ma > 0)
                            caloric_storage_scale +=
                                ma * old_phase.temperature *
                                old_phase.thermodynamics
                                    ->evaluate(old_phase.temperature, old_phase.absolute_pressure, fractions(old_phase))
                                    .internal_energy_temperature;
                        if (mn > 0)
                            caloric_storage_scale +=
                                mn * new_phase.temperature *
                                new_phase.thermodynamics
                                    ->evaluate(new_phase.temperature, new_phase.absolute_pressure, fractions(new_phase))
                                    .internal_energy_temperature;
                        const double source_energy =
                            !sources.empty() && !sources[c].internal_energy.empty() ? sources[c].internal_energy[a] : 0;
                        double source_mass = 0;
                        if (!sources.empty() && !sources[c].species_moles.empty())
                            for (size_t species = 0; species < ns; ++species)
                                source_mass += sources[c].species_moles[a][species] *
                                               d->eos[a]->catalog().species()[species].molarMass;
                        const double reference =
                            ma > 0 ? old_phase.internal_energy / ma : (mn > 0 ? new_phase.internal_energy / mn : 0);
                        cell_source_caloric +=
                            static_cast<long double>(source_energy) - static_cast<long double>(source_mass) * reference;
                        energy_operand_scale += std::abs(old_phase.internal_energy) +
                                                std::abs(new_phase.internal_energy) + std::abs(source_energy) +
                                                std::abs(source_mass * reference);
                        old_energy += accepted.cells[c].phases[a].internal_energy;
                        new_energy += trial.cells[c].phases[a].internal_energy;
                        if (!sources.empty() && !sources[c].internal_energy.empty())
                            external_energy += sources[c].internal_energy[a];
                        for (size_t s = 0; s < ns; ++s)
                        {
                            old_species[s] += accepted.cells[c].phases[a].moles[s];
                            new_species[s] += trial.cells[c].phases[a].moles[s];
                            if (!sources.empty() && !sources[c].species_moles.empty())
                            {
                                source_species[s] += sources[c].species_moles[a][s];
                                external_mass +=
                                    sources[c].species_moles[a][s] * d->eos[a]->catalog().species()[s].molarMass;
                            }
                        }
                    }
                    caloric_source_scale += std::abs(cell_source_caloric.value());
                    old_momentum = old_momentum + accepted.cells[c].velocity * accepted_mass;
                    new_momentum = new_momentum + trial.cells[c].velocity * candidate_mass;
                    if (!sources.empty())
                        external_momentum = external_momentum + sources[c].momentum;
                }
            });
        const double old_mass_global = d->sum(old_mass), new_mass_global = d->sum(new_mass);
        result.mass_error = new_mass_global - old_mass_global - d->sum(external_mass);
        const double energy_global = d->sum(static_cast<double>(old_energy.value()));
        const double new_energy_global = d->sum(static_cast<double>(new_energy.value()));
        result.internal_energy_error = new_energy_global - energy_global -
                                       d->sum(static_cast<double>(external_energy.value())) - result.pressure_work -
                                       result.empty_phase_energy_roundoff;
        check(std::abs(result.mass_error) <=
                  d->options.conservation_relative_tolerance * std::max(old_mass_global, new_mass_global),
            "Low-Mach physical mass budget failed.");
        result.caloric_energy_scale =
            d->sum(static_cast<double>(caloric_storage_scale.value() + caloric_source_scale.value())) +
            caloric_transport_magnitude + phase_work_magnitude;
        const double operands = d->sum(static_cast<double>(energy_operand_scale.value())) +
                                energy_transport_operand_magnitude + phase_work_magnitude +
                                std::abs(result.empty_phase_energy_roundoff);
        const double ranks = mesh.owned_cell_map()->getComm()->getSize();
        result.internal_energy_roundoff_bound = (64 + 2 * ranks) * std::numeric_limits<double>::epsilon() * operands;
        const double physical_budget = d->options.conservation_relative_tolerance * result.caloric_energy_scale;
        check(std::isfinite(physical_budget) && physical_budget > 0 &&
                  std::isfinite(result.internal_energy_roundoff_bound),
            "Low-Mach caloric energy scales are invalid.");
        check(result.internal_energy_roundoff_bound <= 0.25 * physical_budget,
            "Low-Mach energy reference is too poorly conditioned for the requested caloric accuracy.");
        check(std::abs(result.internal_energy_error) <= physical_budget + result.internal_energy_roundoff_bound,
            "Low-Mach physical internal-energy/work budget failed.");

        result.population_number_errors.assign(np, 0);
        for (size_t a = 0; a < np; ++a)
            if (d->options.phase_roles[a].role == PhaseRole::DispersedGas)
            {
                detail::CompensatedSum<long double> old_count, new_count, count_source, signed_residual;
                collective_detail::collective_local_validation(mesh, "Population number budget operands",
                    [&]
                    {
                        for (size_t c = 0; c < nc; ++c)
                        {
                            const double old_number = *accepted.cells[c].phase_numbers[a];
                            const double new_number = *trial.cells[c].phase_numbers[a];
                            const double change = source_stage_detail::number_packet(sources, c, a);
                            old_count += old_number;
                            new_count += new_number;
                            count_source += change;
                            // Retain small-cell cancellations before any positive total is narrowed or reduced.
                            detail::CompensatedSum<long double> cell_residual;
                            cell_residual += new_number;
                            cell_residual += -static_cast<long double>(old_number);
                            cell_residual += -static_cast<long double>(change);
                            signed_residual += cell_residual.value();
                        }
                        check(std::isfinite(old_count.value()) && std::isfinite(new_count.value()) &&
                                  std::isfinite(count_source.value()) && std::isfinite(signed_residual.value()),
                            "Population number budget overflow.");
                    });
                const double before = d->sum(static_cast<double>(old_count.value())),
                             after = d->sum(static_cast<double>(new_count.value())),
                             change = d->sum(static_cast<double>(count_source.value()));
                const double error = d->sum(static_cast<double>(signed_residual.value()));
                check(std::isfinite(before) && std::isfinite(after) && std::isfinite(change) && std::isfinite(error),
                    "Global population number budget overflow.");
                result.population_number_errors[a] = error;
                check(std::abs(error) <=
                          d->options.conservation_relative_tolerance * std::max({before, after, std::abs(change)}),
                    "Low-Mach physical population number budget failed.");
            }
        result.species_errors.resize(ns);
        for (size_t s = 0; s < ns; ++s)
        {
            const double old_n = d->sum(old_species[s]), new_n = d->sum(new_species[s]),
                         delta = d->sum(source_species[s]);
            result.species_errors[s] = new_n - old_n - delta;
            check(std::abs(result.species_errors[s]) <=
                      d->options.conservation_relative_tolerance *
                          std::max({old_n, new_n, std::abs(delta), std::numeric_limits<double>::min()}),
                "Low-Mach physical species budget failed.");
        }
        for (size_t component = 0; component < 3; ++component)
        {
            const double old_p = d->sum(old_momentum.component(component)),
                         new_p = d->sum(new_momentum.component(component));
            const double external_p = d->sum(external_momentum.component(component));
            const double pressure_p = d->sum(pressure_impulse.component(component));
            result.pressure_impulse.component(component) = pressure_p;
            result.external_impulse.component(component) = external_p;
            result.momentum_error.component(component) = new_p - old_p - external_p - pressure_p;
            check(std::abs(result.momentum_error.component(component)) <=
                      d->options.conservation_relative_tolerance *
                          std::max({std::abs(old_p), std::abs(new_p), std::abs(external_p), std::abs(pressure_p),
                              d->sum(momentum_operand_scale.component(component)), std::numeric_limits<double>::min()}),
                "Low-Mach physical momentum budget failed.");
        }
        trial.time += dt;
        ++trial.revision;
        Trial staged;
        staged.candidate = std::move(trial);
        staged.statistics = result;
        staged.transport = std::move(last_transport);
        staged.sources = std::move(sources);
        staged.gross_events = std::move(gross_events);
        staged.provider = std::move(provider);
        staged.owner = this;
        staged.nonce = d->trial_nonce;
        staged.base_revision = accepted.revision;
        staged.geometry_epoch = d->epoch;
        staged.phase_transport = std::move(last_phase_transport);
        staged.face_flux.resize(mesh.num_owned_faces());
        for (size_t f = 0; f < staged.face_flux.size(); ++f)
            staged.face_flux[f] = d->trial_flux.local_value(static_cast<LO>(f));
        return staged;
    }
    catch (...)
    {
        d->publish_fields(accepted);
        throw;
    }
}
void LowMachMixtureSolver::require_controller(const ControllerLease* controller) const
{
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach controller lease",
        [&]
        {
            check(!d->controller_active ? controller == nullptr
                                        : controller && controller->owner == this && controller->id == d->controller_id,
                "Low-Mach advancement belongs exclusively to its active external controller.");
            check(d->trial_nonce != std::numeric_limits<std::uint64_t>::max(), "Low-Mach trial revision exhausted.");
        });
    collective_detail::require_uniform_value(*d->mesh, int(d->controller_active), "Low-Mach controller ownership");
    collective_detail::require_uniform_value(*d->mesh, d->controller_id, "Low-Mach controller identity");
    collective_detail::require_uniform_value(*d->mesh, d->trial_nonce, "Low-Mach accepted trial generation");
}
LowMachMixtureSolver::ControllerLease LowMachMixtureSolver::acquire_controller()
{
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach acquire controller",
        [&]
        {
            check(!d->controller_active && d->controller_id != std::numeric_limits<std::uint64_t>::max() &&
                      d->trial_nonce != std::numeric_limits<std::uint64_t>::max(),
                "Low-Mach solver already has an external controller or exhausted its lease counter.");
        });
    d->controller_active = true;
    ++d->controller_id;
    ++d->trial_nonce;
    return ControllerLease(this, d->controller_id);
}
void LowMachMixtureSolver::release_controller(std::uint64_t id) noexcept
{
    if (d->controller_active && id == d->controller_id)
    {
        d->controller_active = false;
        if (d->trial_nonce != std::numeric_limits<std::uint64_t>::max())
            ++d->trial_nonce;
    }
}
LowMachMixtureSolver::ControllerLease::ControllerLease(ControllerLease&& other) noexcept
    : owner(std::exchange(other.owner, nullptr)), id(other.id)
{
}
LowMachMixtureSolver::ControllerLease& LowMachMixtureSolver::ControllerLease::operator=(
    ControllerLease&& other) noexcept
{
    if (this != &other)
    {
        if (owner)
            owner->release_controller(id);
        owner = std::exchange(other.owner, nullptr);
        id = other.id;
    }
    return *this;
}
LowMachMixtureSolver::ControllerLease::~ControllerLease()
{
    if (owner)
        owner->release_controller(id);
}
void LowMachMixtureSolver::commit(const Trial& trial, const ControllerLease* controller)
{
    require_controller(controller);
    State candidate;
    Diagnostics statistics;
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach immutable trial commit",
        [&]
        {
            check(trial.owner == this && trial.nonce == d->trial_nonce && trial.base_revision == d->accepted.revision &&
                      trial.geometry_epoch == d->epoch && d->epoch == mesh_geometry_epoch(*d->mesh),
                "Low-Mach trial is foreign, already committed, superseded or belongs to stale geometry.");
            candidate = trial.candidate;
            statistics = trial.statistics;
        });
    d->publish_fields(candidate, true);
    for (size_t f = 0; f < trial.face_flux.size(); ++f)
        d->flux.set_owned_value(static_cast<LO>(f), trial.face_flux[f]);
    d->flux.sync_ghosts();
    d->accepted = std::move(candidate);
    d->diagnostics = std::move(statistics);
    ++d->trial_nonce;
}
LowMachMixtureSolver::Diagnostics LowMachMixtureSolver::advance(double dt, const std::vector<CellSources>& sources)
{
    const auto staged = preview(dt, sources);
    commit(staged);
    return d->diagnostics;
}
} // namespace SimpleFluid::Reacting
