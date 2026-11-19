/** @file LowMachMixtureSolver.cc
 * @brief Conservative native mixture transport, thermodynamic closure and immutable advancement.
 */
#include "solvers/LowMachMixtureSolver.hh"
#include "FVM/CarrierTransportPlan.hh"
#include "FVM/CellOperators.hh"
#include "FVM/TransportSystem.hh"
#include "FVM/TransportSystemFactory.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/EquationValidation.hh"
#include "equations/VariableDensityMomentumEquation.hh"
#include "solvers/details/LowMachConditionedThermal.hh"
#include "solvers/details/LowMachQbmmControllerRestart.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"
#include "solvers/details/LowMachQbmmState.hh"
#include "solvers/details/LowMachQbmmTransport.hh"
#include "solvers/details/LowMachScaledFaceSum.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include "solvers/details/LowMachTransportConsistency.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

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
/** Resolve the default identity exactly once so every rank sees the same carrier key. */
FVM::CarrierId resolved_carrier_identity(const Solver::PhaseRoleDescriptor& role, std::size_t phase_index)
{
    switch (role.role)
    {
        case Solver::PhaseRole::BulkLiquid:
            return FVM::CarrierId{
                FVM::CarrierKind::Liquid,
                role.identity.empty() ? "liquid_" + std::to_string(phase_index) : role.identity};
        case Solver::PhaseRole::BulkGas:
            return FVM::CarrierId{
                FVM::CarrierKind::BulkGas,
                role.identity.empty() ? "bulk_gas_" + std::to_string(phase_index) : role.identity};
        case Solver::PhaseRole::DispersedGas:
            return FVM::CarrierId{FVM::CarrierKind::Population, role.identity};
    }
    throw std::logic_error("Unknown Low-Mach phase role in carrier identity resolution.");
}
std::optional<std::array<double, 3>> acceleration_components(const std::optional<Solver::Vector>& acceleration)
{
    if (!acceleration)
        return std::nullopt;
    return std::array<double, 3>{acceleration->x, acceleration->y, acceleration->z};
}

using FaceSum = qbmm_face_detail::ScaledFaceSum;

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

double effective_mixture_viscosity(const Solver::Options& options, std::span<const PhaseThermalState> phases,
    std::span<const double> volumes, bool inviscid_required)
{
    double viscosity = 0;
    if (options.mixture_dynamic_viscosity)
        viscosity = options.mixture_dynamic_viscosity(phases, volumes);
    else
    {
        size_t populated = 0;
        for (size_t a = 0; a < phases.size(); ++a)
            if (volumes[a] > 0)
            {
                const auto& phase = phases[a];
                const auto point =
                    phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase));
                if (++populated == 1)
                    viscosity = point.dynamic_viscosity;
                else
                    check(point.dynamic_viscosity == 0 && viscosity == 0,
                        "Multiphase viscous flow requires a supplied mixture-viscosity closure.");
            }
    }
    check(std::isfinite(viscosity) && viscosity >= 0, "Invalid supplied mixture viscosity.");
    if (inviscid_required)
        check(viscosity == 0., "Prescribed closed ALE requires an inviscid/slip wall policy.");
    return viscosity;
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
    std::optional<PressureConditionedThermalEvaluation> conditioned;
};
// U_new+p_new*V_new = U_adv+source+p_new*V_adv owns actual phase-volume work once.
CaloricCell recover_enthalpy(const Solver::CellState& transported, std::span<const double> advected_volumes, double p0,
    const Solver::Options& options, const FrozenInterphasePlan* conditioned_plan = nullptr,
    std::span<const double> advected_energy = {})
{
    CaloricCell result{transported, std::vector<double>(transported.phases.size()), 0, 0, 0};
    if (conditioned_plan)
    {
        PressureConditionedThermalEvaluation evaluated;
        try
        {
            evaluated = evaluate_pressure_conditioned_thermal(*conditioned_plan, transported.phases, advected_energy,
                advected_volumes, p0 + transported.mechanical_pressure, options.thermal_mode,
                {options.conservation_relative_tolerance, options.thermal_relative_tolerance});
        }
        catch (const ThermalCaloricRangeError& error)
        {
            throw CaloricRangeError(error.what());
        }
        result.cell.phases = evaluated.phases;
        result.volumes = evaluated.volumes;
        result.volume = evaluated.total_volume;
        result.derivative = evaluated.volume_pressure_derivative;
        result.work = evaluated.pressure_work;
        result.roundoff = evaluated.empty_phase_energy_roundoff;
        result.work_magnitude = evaluated.work_magnitude;
        result.pressure_operands = evaluated.pressure_operands;
        result.conditioned = std::move(evaluated);
        return result;
    }
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
        if (target == minimum)
            return lo;
        if (target == maximum)
            return hi;
        // Resolve the monotone caloric law to its representable temperature bracket rather than stopping
        // early on a reference-shifted energy roundoff band.
        for (int iteration = 0; iteration < 180; ++iteration)
        {
            const double mid = std::midpoint(lo, hi);
            if (mid == lo || mid == hi)
                return mid;
            const double value = enthalpy(mid);
            check(std::isfinite(value), "Low-Mach enthalpy inversion returned nonfinite enthalpy.");
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

void LowMachMixtureSolver::require_uniform_advancement(const Mesh& mesh, AdvancementOperation operation)
{
    collective_detail::require_uniform_value(mesh, static_cast<int>(operation), "Low-Mach advancement operation");
}

struct SIMPLEFLUID_SOLVERS_LOCAL PlanarWallInterval
{
    double old_surface = 0, new_surface = 0, time_step = 0;
};

struct LowMachMixtureSolver::ALEGeometryCertificate
{
    struct OwnerIdentity
    {
    };
    std::shared_ptr<const OwnerIdentity> owner;
    PlanarMotion::Authority motion_identity;
    const void* geometry_identity = nullptr;
    PlanarMotion::StateSnapshot source;
    std::uint64_t logical_geometry = 0;
    std::uint64_t old_geometry_epoch = 0, new_geometry_epoch = 0;
    double target_surface = 0, time_step = 0;
    std::vector<double> old_volumes, new_volumes, swept_flux;
    std::vector<Pack::global_ordinal_type> cell_ids, face_ids, cell_geometry_ids, face_geometry_ids;

    void validate_maps(const Mesh& mesh) const
    {
        check(geometry_identity == mesh.geometry_identity() && cell_ids.size() == mesh.num_local_cells() &&
                  cell_geometry_ids.size() == cell_ids.size() && face_ids.size() == mesh.num_faces() &&
                  face_geometry_ids.size() == face_ids.size(),
            "Planar geometry certificate maps differ from the live geometry.");
        for (size_t c = 0; c < cell_ids.size(); ++c)
        {
            const LO cell = static_cast<LO>(c);
            check(
                cell_ids[c] == mesh.cell_global_id(cell) && cell_geometry_ids[c] == mesh.cell_geometry_global_id(cell),
                "Planar cell maps changed.");
        }
        for (size_t f = 0; f < face_ids.size(); ++f)
        {
            const LO face = static_cast<LO>(f);
            check(
                face_ids[f] == mesh.face_global_id(face) && face_geometry_ids[f] == mesh.face_geometry_global_id(face),
                "Planar face maps changed.");
        }
    }
};

struct LowMachMixtureSolver::QbmmPublication
{
    QbmmStateOrigin origin = QbmmStateOrigin::Corrector;
    QbmmReceiptHistory history = QbmmReceiptHistory::NoInterval;
    std::optional<std::uint64_t> archived_generation;
    std::uint64_t shape_diagnostics_version = 3; ///< v2 records shape sources, v3 adds shape transport.
    std::shared_ptr<const QbmmIntervalReport> interval;
    std::optional<QbmmBalances> balances;
    std::optional<PlanarWallInterval> planar_wall_interval;
};

struct LowMachMixtureSolver::Implementation
{
    SP<const Mesh> mesh;
    std::shared_ptr<const ALEGeometryCertificate::OwnerIdentity> planar_owner;
    std::vector<std::shared_ptr<const MixtureThermodynamics>> eos;
    Options options;
    State accepted;
    Diagnostics diagnostics;
    std::shared_ptr<const QbmmPublication> qbmm_publication;
    std::optional<std::uint64_t> active_qbmm_controller, accepted_qbmm_controller;
    PlanarMotion::Authority active_controller_motion; ///< With a controller stamp, empty denotes fixed geometry.
    std::optional<std::array<double, 3>> active_body_acceleration, accepted_count_only_body_acceleration;
    bool count_only_force_provenance_recorded = false;
    bool needs_restart_controller = false;
    bool qbmm_legacy_default_only = false, active_legacy_compatible = true;
    bool initialized = false, controller_active = false;
    std::optional<std::uint64_t> checkpoint_controller() const noexcept
    {
        return active_qbmm_controller ? active_qbmm_controller : accepted_qbmm_controller;
    }
    void validate_planar_controller_authority(const PlanarMotion& motion) const
    {
        check(!active_qbmm_controller || active_controller_motion == motion.d_authority,
            "The reacting controller is bound to different geometry or motion authority.");
        check(active_qbmm_controller || !accepted_qbmm_controller || accepted_planar_motion == motion.d_authority,
            "Fixed reacting QBMM history requires a configured planar controller before moving replay.");
    }
    void validate_planar_checkpoint_provenance(const PlanarMotion& motion) const
    {
        check(options.qbmm_families.empty() || diagnostics.correctors == 0 ||
                  (accepted_planar_motion == motion.d_authority && accepted_qbmm_controller == checkpoint_controller()),
            "Planar QBMM checkpoint requires an interval accepted by its matching motion and source authority.");
    }
    void clear_initialization_provenance() noexcept
    {
        accepted_qbmm_controller.reset();
        accepted_count_only_body_acceleration.reset();
        count_only_force_provenance_recorded = false;
        needs_restart_controller = false;
        qbmm_legacy_default_only = false;
    }
    std::uint64_t trial_nonce = 0, controller_id = 0, source_invocation = 0;
    std::uint64_t preview_sequence = 0;
    std::uint64_t epoch = 0, logical_geometry = 0;
    PlanarMotion::Authority accepted_planar_motion; // QBMM only; compared with explicit live authority.
    ScalarField rho, old_rho, mu, mechanical_pressure, old_scalar, scalar, unit, zero, published_rho;
    VectorField u, old_u, pressure_gradient, published_u;
    FaceField flux, trial_flux, mass_flux;
    FVM::TransportGeometryCache<Mesh> geometry;
    FVM::CellGradientCache<Pack, Mesh> gradient_cache;
    BelosLinearSolver<Pack> transport_solver;
    std::vector<transport_consistency_detail::BatchedWorkspace> transport_workspaces;
    std::vector<FVM::CarrierMultiVectorBuffer<Pack>> carrier_buffers;
    VariableDensityMomentumEquation<Pack, Mesh> momentum;
    std::optional<LowMachPressureEquation> pressure;
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
          pressure(std::in_place, mesh)
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
                std::set<std::string> explicit_identities;
                std::set<FVM::CarrierId> carriers;
                for (size_t a = 0; a < eos.size(); ++a)
                {
                    const auto& role = options.phase_roles[a];
                    check(role.role == PhaseRole::BulkLiquid || role.role == PhaseRole::BulkGas ||
                              role.role == PhaseRole::DispersedGas,
                        "Unknown physical phase role.");
                    check((role.role == PhaseRole::BulkLiquid) == (eos[a]->kind() == EOSKind::Liquid),
                        "Physical phase role/EOS kind mismatch.");
                    const auto carrier = resolved_carrier_identity(role, a);
                    check(role.identity.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                              (role.role != PhaseRole::DispersedGas || !role.identity.empty()) &&
                              (role.identity.empty() || explicit_identities.insert(role.identity).second) &&
                              carriers.insert(carrier).second,
                        "Physical phase identities must resolve to unique carrier identities; "
                        "dispersed identities must be explicit.");
                }
                qbmm_detail::validate_configuration(options, eos);
                planar_owner = std::make_shared<const ALEGeometryCertificate::OwnerIdentity>();
                if (!options.qbmm_families.empty())
                    BelosLinearSolver<Pack>::validate_options(options.linear);
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
        collective_detail::require_uniform_value(
            *mesh, static_cast<int>(options.use_multi_rhs_transport), "Low-Mach multi-RHS transport selection");
        collective_detail::require_uniform_value(
            *mesh, options.linear.max_batch_width, "Low-Mach linear max batch width");
        collective_detail::require_uniform_value(
            *mesh, static_cast<int>(options.linear.multi_rhs_policy), "Low-Mach linear multi-RHS policy");
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
        collective_detail::require_uniform_value(*mesh, options.qbmm_families.size(), "Native QBMM phase layout");
        if (!options.qbmm_families.empty())
        {
            collective_detail::require_uniform_value(
                *mesh, int(options.qbmm_model.velocity), "Native QBMM velocity mode");
            const auto& linear = options.linear;
            collective_detail::require_uniform_value(
                *mesh, linear.max_iterations, "Native QBMM transport iteration budget");
            collective_detail::require_uniform_value(*mesh, linear.tolerance, "Native QBMM transport linear tolerance");
            collective_detail::require_uniform_value(*mesh, linear.verbosity, "Native QBMM transport linear verbosity");
            collective_detail::require_uniform_value(
                *mesh, int(linear.backend), "Native QBMM transport linear backend");
            collective_detail::require_uniform_value(
                *mesh, int(linear.preconditioner), "Native QBMM transport preconditioner");
            collective_detail::require_uniform_value(
                *mesh, int(linear.reuse_preconditioner), "Native QBMM transport preconditioner reuse");
        }
        for (const auto& family : options.qbmm_families)
        {
            collective_detail::require_uniform_value(*mesh, int(bool(family)), "Native QBMM phase binding");
            if (family)
            {
                const auto stamp = qbmm_family_stamp(*family);
                collective_detail::require_uniform_value(*mesh, stamp.catalog_fingerprint, "Native QBMM catalog");
                collective_detail::require_uniform_value(*mesh, stamp.family_fingerprint, "Native QBMM family");
                collective_detail::require_uniform_value(*mesh, stamp.mass_scale_kg, "Native QBMM mass scale");
            }
        }
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
    void refresh_numeric_geometry()
    {
        geometry.refresh();
        gradient_cache.refresh();
        momentum.refresh_geometry();
        // Inline storage removes a rank-local outer allocation before the pressure
        // constructor enters its field/cache communication.
        pressure.emplace(mesh);
        epoch = mesh_geometry_epoch(*mesh);
    }
    void balanced_pressure_gradient(const Vector& acceleration)
    {
        collective_detail::collective_local_validation(*mesh, "Balanced native body-force pressure gradient",
            [&]
            {
                for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                {
                    const LO cell = static_cast<LO>(c);
                    const double density = rho.owned_value(cell), pressure = mechanical_pressure.owned_value(cell);
                    std::array<std::array<real_t, 3>, 3> normal{};
                    Vector rhs{};
                    // Use the same unweighted interior LS geometry and periodic image vectors as native gradients.
                    FVM::detail::visit_gradient_geometry_samples(
                        *mesh, cell,
                        [&](const auto& sample)
                        {
                            const auto& displacement = sample.direction;
                            const double mean_density = .5 * density + .5 * rho.local_value(sample.other_lid);
                            const double residual = mechanical_pressure.local_value(sample.other_lid) - pressure -
                                                    mean_density * acceleration.dot(displacement);
                            check(std::isfinite(residual), "Balanced native pressure edge residual is nonfinite.");
                            rhs = rhs + displacement * residual;
                            for (size_t row = 0; row < 3; ++row)
                                for (size_t column = 0; column < 3; ++column)
                                    normal[row][column] += displacement.component(row) * displacement.component(column);
                        },
                        false);
                    const auto gradient = acceleration * density + FVM::detail::solve_3x3(normal, rhs);
                    for (size_t axis = 0; axis < 3; ++axis)
                        check(
                            std::isfinite(gradient.component(axis)), "Balanced native pressure gradient is nonfinite.");
                    pressure_gradient.set_owned_value(cell, gradient);
                }
            });
    }
    void publish_fields(const State& state, bool publish = false, std::span<const double> control_volumes = {})
    {
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            double mass = 0;
            for (const auto& phase : state.cells[c].phases)
                mass += phase_mass(phase);
            rho.set_owned_value(static_cast<LO>(c),
                mass / (control_volumes.empty() ? mesh->cell_volume(static_cast<LO>(c)) : control_volumes[c]));
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
    void publish_fixed(State&& state, Diagnostics&& statistics, std::span<const double> face_flux,
        std::uint64_t generation, std::shared_ptr<const QbmmPublication> publication = {})
    {
        if (options.qbmm_families.empty())
        {
            publish_fields(state, true);
            for (size_t f = 0; f < face_flux.size(); ++f)
                flux.set_owned_value(static_cast<LO>(f), face_flux[f]);
            flux.sync_ghosts();
        }
        else
        {
            stage_publication_fields(state, face_flux);
            publish_staged_fields();
        }
        accepted = std::move(state);
        diagnostics = std::move(statistics);
        qbmm_publication = std::move(publication);
        accepted_qbmm_controller = active_qbmm_controller;
        needs_restart_controller = false;
        initialized = true;
        trial_nonce = generation;
    }
    void stage_publication_fields(const State& state, std::span<const double> face_flux)
    {
        // These corrector workspaces are overwritten before the next preview reads them.
        // Keep accepted fields and their public objects intact until all imports succeed.
        collective_detail::collective_local_validation(*mesh, "Low-Mach field staging",
            [&]
            {
                check(state.cells.size() == mesh->num_owned_cells() && face_flux.size() == mesh->num_owned_faces(),
                    "Low-Mach publication does not match the owned cell and face maps.");
                for (size_t c = 0; c < state.cells.size(); ++c)
                {
                    const auto cell = static_cast<LO>(c);
                    const auto& candidate = state.cells[c];
                    double mass = 0;
                    for (const auto& phase : candidate.phases)
                        mass += phase_mass(phase);
                    const double density = mass / mesh->cell_volume(cell);
                    check(std::isfinite(density) && density > 0, "Low-Mach publication density is invalid.");
                    old_rho.set_owned_value(cell, density);
                    mu.set_owned_value(cell, density);
                    old_u.set_owned_value(cell, candidate.velocity);
                    pressure_gradient.set_owned_value(cell, candidate.velocity);
                    old_scalar.set_owned_value(cell, candidate.mechanical_pressure);
                }
                for (size_t f = 0; f < face_flux.size(); ++f)
                {
                    check(std::isfinite(face_flux[f]), "Low-Mach publication face flux is invalid.");
                    trial_flux.set_owned_value(static_cast<LO>(f), face_flux[f]);
                }
            });
        old_rho.sync_ghosts();
        mu.sync_ghosts();
        old_u.sync_ghosts();
        pressure_gradient.sync_ghosts();
        old_scalar.sync_ghosts();
        trial_flux.sync_ghosts();
    }
    void publish_staged_fields() noexcept
    {
        const auto swap_storage = [](auto& accepted_field, auto& staged_field)
        {
            accepted_field.owned_data().swap(staged_field.owned_data());
            accepted_field.overlap_data().swap(staged_field.overlap_data());
        };
        swap_storage(rho, old_rho);
        swap_storage(published_rho, mu);
        swap_storage(u, old_u);
        swap_storage(published_u, pressure_gradient);
        swap_storage(mechanical_pressure, old_scalar);
        swap_storage(flux, trial_flux);
    }
    FVM::TransportSystem<Pack> assemble_phase_transport(
        const FaceField& phase_flux, double dt, const FVM::ALEControlVolumeState* ale = nullptr)
    {
        using Factory = FVM::TransportSystemFactory<Pack, Mesh>;
        return Factory::assemble(typename Factory::WeightedScalarTerms{.history = {old_scalar, dt},
                                     .face_fluxes = phase_flux,
                                     .coefficients = {unit, unit, zero},
                                     .sources = {.source = [](LO) { return 0.0; }}},
            typename Factory::WeightedScalarBoundary{
                .condition = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; },
                .value = [](int, size_t) { return 0.0; }},
            typename Factory::WeightedScalarOptions{.discretization = {.time = FVM::ScalarTimeScheme::BackwardEuler,
                                                        .convection = FVM::ScalarConvectionScheme::Upwind},
                .correction = {FVM::NonOrthogonalTreatment::Implicit, nullptr}},
            typename Factory::WeightedScalarContext{.geometry_cache = &geometry, .ale = ale});
    }
    std::vector<QbmmFaceFlux> transport_qbmm(const State& previous, State& candidate, size_t phase,
        const FaceField& phase_flux, double dt, Diagnostics& statistics, const FVM::ALEControlVolumeState* ale)
    {
        const size_t species = eos[phase]->catalog().species().size(), cells = previous.cells.size();
        std::vector<std::vector<double>> old_amount, amount;
        collective_detail::collective_local_validation(*mesh, "Native QBMM block assembly input",
            [&]
            {
                old_amount.assign(species + 4, std::vector<double>(cells));
                for (size_t c = 0; c < cells; ++c)
                {
                    const auto& cell = previous.cells[c];
                    for (size_t s = 0; s < species; ++s)
                        old_amount[s][c] = cell.phases[phase].moles[s];
                    old_amount[species][c] = *cell.phase_numbers[phase];
                    old_amount[species + 1][c] = cell.qbmm_shape[phase]->H2;
                    old_amount[species + 2][c] = cell.qbmm_shape[phase]->H3;
                    old_amount[species + 3][c] = cell.phases[phase].internal_energy;
                    const double volume = ale ? ale->old_cell_volumes()[c] : mesh->cell_volume(static_cast<LO>(c));
                    const double trace =
                        transport_consistency_detail::transport_concentration(old_amount[species][c], volume);
                    old_scalar.set_owned_value(static_cast<LO>(c), trace);
                }
            });
        old_scalar.sync_ghosts();
        // One native concrete operator owns the complete family, including signed phase energy.
        const auto system = assemble_phase_transport(phase_flux, dt, ale);
        auto block = transport_consistency_detail::enforce_block(mesh, old_amount, phase_flux, dt, amount,
            *system.matrix, *system.rhs, options.linear, options.conservation_relative_tolerance, species, ale);
        std::array<long double, 2> local_h1{}, global_h1{};
        collective_detail::collective_local_validation(*mesh, "Native represented QBMM transport block",
            [&]
            {
                const auto& family = *options.qbmm_families[phase];
                detail::CompensatedSum<long double> defect, magnitude;
                for (size_t c = 0; c < cells; ++c)
                {
                    auto& cell = candidate.cells[c];
                    for (size_t s = 0; s < species; ++s)
                        cell.phases[phase].moles[s] = amount[s][c];
                    cell.phase_numbers[phase] = amount[species][c];
                    cell.qbmm_shape[phase] = QbmmShapeState{amount[species + 1][c], amount[species + 2][c]};
                    cell.phases[phase].internal_energy = amount[species + 3][c];
                    const long double old_h1 = family.scaled_mass(previous.cells[c].phases[phase].moles);
                    const long double new_h1 = family.scaled_mass(cell.phases[phase].moles);
                    defect += new_h1 - old_h1;
                    magnitude += old_h1;
                    magnitude += new_h1;
                }
                // No source/thermodynamic callback may see a split or unrealizable block.
                qbmm_detail::validate_transport(options, candidate);
                std::vector<double> normalized_species;
                for (const auto& packet : block.owned_flux)
                {
                    double maximum_rate = 0;
                    for (const double rate : packet.species_molar_flux)
                        maximum_rate = std::max(maximum_rate, std::abs(rate));
                    bool absent;
                    if (std::fpclassify(maximum_rate) == FP_SUBNORMAL)
                    {
                        // Move the common subnormal exponent out of the rates before
                        // the schema multiplies them by their species molar masses.
                        int mass_exponent = std::numeric_limits<int>::min();
                        for (size_t s = 0; s < species; ++s)
                            if (const double rate = packet.species_molar_flux[s]; rate != 0)
                                mass_exponent = std::max(mass_exponent,
                                    std::ilogb(rate) + std::ilogb(family.catalog().species()[s].molarMass) + 2);
                        mass_exponent += static_cast<int>(std::bit_width(species - 1));
                        const int h1_exponent = mass_exponent - std::ilogb(family.mass_scale_kg());
                        const int headroom =
                            std::numeric_limits<double>::max_exponent - 1 - std::max(mass_exponent, h1_exponent);
                        const int shift = std::clamp(headroom, 0, -1 - std::ilogb(maximum_rate));
                        normalized_species.resize(species);
                        for (size_t s = 0; s < species; ++s)
                            normalized_species[s] = std::scalbn(packet.species_molar_flux[s], shift);
                        const double scaled_h1 = family.scaled_mass_change(normalized_species);
                        const double h1 = std::scalbn(scaled_h1, -shift);
                        // Preserve the schema's represented-H1 gate at the physical scale.
                        check(scaled_h1 == 0 || (h1 != 0 && std::abs(std::scalbn(h1, shift) / scaled_h1 - 1.) <=
                                                                64 * std::numeric_limits<double>::epsilon()),
                            "QBMM face H1 cannot be represented within the roundoff budget.");
                        absent = h1 == 0;
                    }
                    else
                        absent = family.scaled_mass_change(packet.species_molar_flux) == 0;
                    check(absent == (packet.population_number_flux == 0) && absent == (packet.shape_flux.H2 == 0) &&
                              absent == (packet.shape_flux.H3 == 0),
                        "QBMM canonical material and moment face packet presence differs.");
                }
                local_h1 = {defect.value(), magnitude.value()};
            });
        const auto comm = mesh->owned_cell_map()->getComm();
        transport_consistency_detail::sum_qbmm_budget(*comm, local_h1, global_h1);
        check(source_stage_detail::normalized(global_h1[0], global_h1[1],
                  transport_consistency_detail::qbmm_closed_budget_tolerance(comm->getSize())) <= 1,
            "QBMM transport fails its closed derived H1 budget.");
        statistics.transport_consistency_refinements += block.sweeps;
        for (size_t q = 0; q < block.components.size(); ++q)
        {
            const auto& component = block.components[q];
            statistics.transport_linear.add(component.linear);
            statistics.maximum_initial_transport_residual =
                std::max(statistics.maximum_initial_transport_residual, component.initial_residual);
            double* residual = q < species        ? &statistics.maximum_species_transport_residual
                               : q == species     ? &statistics.maximum_population_transport_residual
                               : q == species + 3 ? &statistics.maximum_energy_transport_residual
                                                  : &statistics.maximum_shape_transport_residual;
            *residual = std::max(*residual, component.final_residual);
        }
        return std::move(block.owned_flux);
    }
    void transport(std::span<const double> old_amount, const FaceField& phase_flux, double dt,
        std::vector<double>& new_amount, Diagnostics& statistics, transport_consistency_detail::Quantity quantity,
        const FVM::ALEControlVolumeState* ale = nullptr)
    {
        collective_detail::collective_local_validation(*mesh, "Accepted low-Mach transport concentration",
            [&]
            {
                for (size_t c = 0; c < old_amount.size(); ++c)
                {
                    const double value =
                        old_amount[c] / (ale ? ale->old_cell_volumes()[c] : mesh->cell_volume(static_cast<LO>(c)));
                    check(std::isfinite(old_amount[c]) && std::isfinite(value) && (old_amount[c] == 0 || value != 0) &&
                              (quantity == transport_consistency_detail::Quantity::Energy || old_amount[c] >= 0),
                        "Accepted phase transport inventory/concentration is invalid or unrepresentable.");
                    old_scalar.set_owned_value(static_cast<LO>(c), value);
                }
            });
        old_scalar.sync_ghosts();
        scalar.owned_data().update(1.0, old_scalar.owned_data(), 0.0);
        auto system = assemble_phase_transport(phase_flux, dt, ale);
        transport_solver.notify_operator_values_changed();
        const auto result =
            transport_solver.solve_with_statistics(system.matrix, *system.rhs, scalar.owned_data(), options.linear);
        if (!result.converged)
            throw std::runtime_error("Low-Mach phase transport failed true linear residual gate.");
        const auto consistent =
            transport_consistency_detail::enforce(mesh, old_amount, phase_flux, dt, scalar, new_amount, *system.matrix,
                *system.rhs, result, options.linear, options.conservation_relative_tolerance, quantity, ale);
        statistics.transport_linear.add(consistent.linear);
        statistics.transport_consistency_refinements += consistent.refinements;
        statistics.maximum_initial_transport_residual =
            std::max(statistics.maximum_initial_transport_residual, consistent.initial_residual);
        if (quantity == transport_consistency_detail::Quantity::Species)
            statistics.maximum_species_transport_residual =
                std::max(statistics.maximum_species_transport_residual, consistent.final_residual);
        else if (quantity == transport_consistency_detail::Quantity::Energy)
            statistics.maximum_energy_transport_residual =
                std::max(statistics.maximum_energy_transport_residual, consistent.final_residual);
        else
            statistics.maximum_population_transport_residual =
                std::max(statistics.maximum_population_transport_residual, consistent.final_residual);
    }
    struct CarrierSpeciesSlot
    {
        std::size_t group_index = 0;
        std::size_t column_index = 0;
    };
    void transport_all_carriers(const State& accepted_state, State& trial_state,
        const std::vector<std::unique_ptr<FaceField>>& phase_fluxes, double dt, int corrector_iteration,
        Diagnostics& statistics, std::vector<CarrierSpeciesSlot>& species_slots, double& local_caloric_transport,
        double& local_energy_transport, const FVM::ALEControlVolumeState* ale)
    {
        using Factory = FVM::TransportSystemFactory<Pack, Mesh>;
        using Quantity = transport_consistency_detail::Quantity;
        const std::size_t nc = accepted_state.cells.size();
        const std::size_t np = eos.size();
        const std::size_t ns = eos.front()->catalog().species().size();
        const auto& catalog = eos.front()->catalog();
        const auto comm = mesh->owned_cell_map()->getComm();

        auto carrier_for_phase = [&](std::size_t phase_index) -> FVM::CarrierId
        { return resolved_carrier_identity(options.phase_roles[phase_index], phase_index); };

        std::vector<FVM::PlannedColumnEntry> entries;
        entries.reserve(np * (ns + 2));
        for (std::size_t a = 0; a < np; ++a)
        {
            // Bound families own their moments, species, energy and canonical face flux together.
            if (!options.qbmm_families.empty() && options.qbmm_families[a])
                continue;
            const auto carrier = carrier_for_phase(a);
            FVM::TransportOperatorKey key;
            key.mesh_identity = mesh.get();
            key.geometry_epoch = epoch;
            key.has_ale = (ale != nullptr);
            key.ale_old_geometry_epoch = ale ? ale->old_geometry_epoch() : 0;
            key.ale_new_geometry_epoch = ale ? ale->new_geometry_epoch() : 0;
            key.time_step = dt;
            key.discretization = {.time = FVM::ScalarTimeScheme::BackwardEuler,
                .convection = FVM::ScalarConvectionScheme::Upwind};
            key.non_orthogonal_treatment = FVM::NonOrthogonalTreatment::Implicit;
            key.storage_weight = {&unit, 1};
            key.advection_flux = {phase_fluxes[a].get(), static_cast<std::uint64_t>(corrector_iteration + 1)};
            key.advection_weight = {&unit, 1};
            key.diffusivity = {&zero, 1};

            if (options.phase_roles[a].role == PhaseRole::DispersedGas)
            {
                entries.push_back(FVM::PlannedColumnEntry{
                    FVM::TransportColumn{carrier, FVM::QuantityKind::PopulationNumber, std::nullopt,
                        carrier.identity, {.allow_negative = false}, a},
                    key});
            }
            for (std::size_t s = 0; s < ns; ++s)
            {
                entries.push_back(FVM::PlannedColumnEntry{
                    FVM::TransportColumn{carrier, FVM::QuantityKind::SpeciesMoles, catalog.species()[s].id, {},
                        {.allow_negative = false}, a},
                    key});
            }
            entries.push_back(FVM::PlannedColumnEntry{
                FVM::TransportColumn{carrier, FVM::QuantityKind::InternalEnergy, std::nullopt, {},
                    {.allow_negative = true}, a},
                key});
        }

        const auto plan = FVM::build_carrier_transport_plan(
            entries, FVM::CarrierTransportPlanOptions{.allow_cross_carrier_sharing = false});
        transport_workspaces.resize(plan.operator_groups.size());
        carrier_buffers.resize(plan.operator_groups.size());
        species_slots.assign(np * ns, {});

        auto& multi_diag = statistics.transport_multi_rhs;
        multi_diag = {};
        multi_diag.operator_groups = plan.operator_groups.size();
        multi_diag.transported_columns = plan.columns.size();

        std::unordered_map<std::uint64_t, std::size_t> species_index_by_id;
        species_index_by_id.reserve(ns);
        for (std::size_t s = 0; s < ns; ++s)
        {
            species_index_by_id[catalog.species()[s].id] = s;
        }

        const auto old_volume = [&](std::size_t c)
        { return ale ? ale->old_cell_volumes()[c] : mesh->cell_volume(static_cast<LO>(c)); };

        for (std::size_t g = 0; g < plan.operator_groups.size(); ++g)
        {
            const auto& group = plan.operator_groups[g];
            const std::size_t a = plan.columns[group.plan_column_indices.front()].source_index;
            const std::size_t ncols = group.plan_column_indices.size();

            std::vector<std::vector<double>> old_amounts(ncols, std::vector<double>(nc, 0.0));
            std::vector<Quantity> quantities(ncols);
            std::vector<std::size_t> local_species_indices(ncols, ns);
            std::vector<double> local_max_abs(ncols, 0.0);
            std::vector<double> global_max_abs(ncols, 0.0);
            std::vector<bool> active_mask(ncols, false);

            const auto pack_start = std::chrono::steady_clock::now();
            auto& buffer_mv = carrier_buffers[g].ensure(mesh->owned_cell_map(), ncols);
            collective_detail::collective_local_validation(*mesh, "Accepted low-Mach batched transport concentration",
                [&]
                {
                    for (std::size_t k = 0; k < ncols; ++k)
                    {
                        const auto& col_meta = plan.columns[group.plan_column_indices[k]];
                        check(col_meta.source_index == a,
                            "Low-Mach cross-carrier operator sharing is disabled.");
                        auto sol_col = buffer_mv.getDataNonConst(k);
                        if (col_meta.quantity == FVM::QuantityKind::PopulationNumber)
                        {
                            quantities[k] = Quantity::Population;
                            for (std::size_t c = 0; c < nc; ++c)
                            {
                                const double val = *accepted_state.cells[c].phase_numbers[a];
                                const double conc = val / old_volume(c);
                                check(std::isfinite(val) && std::isfinite(conc) && (val == 0 || conc != 0) && val >= 0,
                                    "Population number concentration is not representable.");
                                old_amounts[k][c] = val;
                                sol_col[c] = conc;
                                local_max_abs[k] = std::max(local_max_abs[k], std::abs(val));
                            }
                        }
                        else if (col_meta.quantity == FVM::QuantityKind::SpeciesMoles)
                        {
                            quantities[k] = Quantity::Species;
                            const std::size_t s = species_index_by_id.at(*col_meta.species_id);
                            local_species_indices[k] = s;
                            for (std::size_t c = 0; c < nc; ++c)
                            {
                                const double val = accepted_state.cells[c].phases[a].moles[s];
                                const double conc = val / old_volume(c);
                                check(std::isfinite(val) && std::isfinite(conc) && (val == 0 || conc != 0) && val >= 0,
                                    "Accepted phase transport inventory/concentration is invalid or unrepresentable.");
                                old_amounts[k][c] = val;
                                sol_col[c] = conc;
                                local_max_abs[k] = std::max(local_max_abs[k], std::abs(val));
                            }
                        }
                        else
                        {
                            quantities[k] = Quantity::Energy;
                            for (std::size_t c = 0; c < nc; ++c)
                            {
                                const double val = accepted_state.cells[c].phases[a].internal_energy;
                                const double conc = val / old_volume(c);
                                check(std::isfinite(val) && std::isfinite(conc) && (val == 0 || conc != 0),
                                    "Accepted phase transport inventory/concentration is invalid or unrepresentable.");
                                old_amounts[k][c] = val;
                                sol_col[c] = conc;
                                local_max_abs[k] = std::max(local_max_abs[k], std::abs(val));
                            }
                        }
                    }
                });
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(ncols),
                local_max_abs.data(), global_max_abs.data());
            for (std::size_t k = 0; k < ncols; ++k)
            {
                active_mask[k] = global_max_abs[k] > 0.0;
                if (active_mask[k])
                {
                    ++multi_diag.active_columns;
                }
            }
            multi_diag.pack_unpack_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - pack_start).count();

            std::vector<FVM::MultiColumnScalarInput<Pack, Mesh>> mc_cols(ncols);
            std::vector<std::span<const double>> old_spans(ncols);
            for (std::size_t k = 0; k < ncols; ++k)
            {
                old_spans[k] = old_amounts[k];
                mc_cols[k] = FVM::MultiColumnScalarInput<Pack, Mesh>{
                    .extensive_old_inventories = old_spans[k],
                    .source = [](LO) { return 0.0; },
                    .boundary_value = [](int, std::size_t) { return 0.0; },
                    .neumann_flux = [](int, std::size_t) { return 0.0; },
                    .allow_negative = (quantities[k] == Quantity::Energy)};
            }

            const auto asm_start = std::chrono::steady_clock::now();
            auto mc_sys = Factory::assemble_multi_column(
                typename Factory::MultiColumnWeightedTerms{
                    .time_step = dt,
                    .face_fluxes = *phase_fluxes[a],
                    .coefficients = {unit, unit, zero},
                    .columns = std::move(mc_cols)},
                typename Factory::MultiColumnWeightedBoundary{
                    .condition = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; }},
                typename Factory::WeightedScalarOptions{
                    .discretization = {.time = FVM::ScalarTimeScheme::BackwardEuler,
                        .convection = FVM::ScalarConvectionScheme::Upwind},
                    .correction = {FVM::NonOrthogonalTreatment::Implicit, nullptr}},
                typename Factory::MultiColumnWeightedContext{.geometry_cache = &geometry, .ale = ale});
            multi_diag.matrix_assembly_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - asm_start).count();
            ++multi_diag.matrix_assemblies;

            transport_solver.notify_operator_values_changed();
            MultiRhsSolveOptions multi_opts;
            multi_opts.solver = options.linear;
            multi_opts.active_columns = active_mask;
            const auto linear_result = transport_solver.solve_multi_rhs_with_statistics(
                mc_sys.matrix, *mc_sys.rhs, buffer_mv, multi_opts);
            if (!linear_result.converged)
            {
                throw std::runtime_error("Low-Mach phase transport failed true linear residual gate.");
            }

            multi_diag.preconditioner_builds += linear_result.preconditioner_builds;
            multi_diag.operator_apply_calls += linear_result.operator_apply_calls;
            multi_diag.operator_column_applies += linear_result.operator_column_applies;
            multi_diag.max_batch_width = std::max(multi_diag.max_batch_width, linear_result.max_batch_width);
            multi_diag.batch_krylov_iterations += linear_result.batch_krylov_iterations;
            multi_diag.column_serial_iterations += linear_result.column_serial_iterations;
            multi_diag.preconditioner_setup_seconds += linear_result.timings.preconditioner_setup_seconds;
            multi_diag.linear_solve_seconds += linear_result.timings.solve_seconds;
            multi_diag.pack_unpack_seconds += linear_result.timings.pack_unpack_seconds;
            switch (linear_result.execution_mode)
            {
                case MultiRhsExecutionMode::BatchedKrylov:
                    ++multi_diag.batched_krylov_solves;
                    break;
                case MultiRhsExecutionMode::ColumnSerial:
                    ++multi_diag.column_serial_solves;
                    break;
                case MultiRhsExecutionMode::SingleRhs:
                    ++multi_diag.single_rhs_solves;
                    break;
            }

            std::vector<std::vector<double>> new_amounts;
            const auto consistent = transport_consistency_detail::enforce_batch(
                mesh, old_spans, quantities, active_mask, *phase_fluxes[a], dt, buffer_mv, new_amounts,
                *mc_sys.matrix, *mc_sys.rhs, linear_result, options.linear,
                options.conservation_relative_tolerance, ale, transport_workspaces[g]);

            statistics.transport_linear.add(consistent.linear);
            statistics.transport_consistency_refinements += consistent.total_refinements;
            statistics.maximum_initial_transport_residual =
                std::max(statistics.maximum_initial_transport_residual, consistent.initial_residual);
            multi_diag.halo_cell_imports += consistent.halo_cell_imports;
            multi_diag.halo_face_imports += consistent.halo_face_imports;
            multi_diag.halo_exchange_seconds += consistent.halo_exchange_seconds;
            multi_diag.consistency_repair_seconds += consistent.consistency_repair_seconds;

            const auto unpack_start = std::chrono::steady_clock::now();
            for (std::size_t k = 0; k < ncols; ++k)
            {
                multi_diag.column_consistency_sweeps.push_back(consistent.column_refinements[k]);
                if (quantities[k] == Quantity::Species)
                {
                    statistics.maximum_species_transport_residual = std::max(
                        statistics.maximum_species_transport_residual, consistent.column_final_residuals[k]);
                    const std::size_t s = local_species_indices[k];
                    species_slots[a * ns + s] = {g, k};
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        trial_state.cells[c].phases[a].moles[s] = new_amounts[k][c];
                    }
                }
                else if (quantities[k] == Quantity::Population)
                {
                    statistics.maximum_population_transport_residual = std::max(
                        statistics.maximum_population_transport_residual, consistent.column_final_residuals[k]);
                    collective_detail::collective_local_validation(*mesh, "Transported population numbers",
                        [&]
                        {
                            for (std::size_t c = 0; c < nc; ++c)
                            {
                                check(std::isfinite(new_amounts[k][c]) && new_amounts[k][c] >= 0,
                                    "Population transport produced a negative/nonfinite number.");
                                trial_state.cells[c].phase_numbers[a] = new_amounts[k][c];
                                const double concentration = new_amounts[k][c] / mesh->cell_volume(static_cast<LO>(c));
                                check(std::isfinite(concentration),
                                    "Transported population concentration is not representable.");
                            }
                        });
                }
                else
                {
                    statistics.maximum_energy_transport_residual = std::max(
                        statistics.maximum_energy_transport_residual, consistent.column_final_residuals[k]);
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        trial_state.cells[c].phases[a].internal_energy = new_amounts[k][c];
                    }
                    ScalarField caloric_carrier_density(mesh, "caloric_transport_characteristic");
                    collective_detail::collective_local_validation(*mesh, "Low-Mach caloric transport scales",
                        [&]
                        {
                            for (std::size_t c = 0; c < nc; ++c)
                            {
                                const LO cell = static_cast<LO>(c);
                                const auto& old_phase = accepted_state.cells[c].phases[a];
                                const double mass = phase_mass(old_phase);
                                double caloric = 0;
                                if (mass > 0)
                                {
                                    caloric = mass * old_phase.temperature *
                                              old_phase.thermodynamics
                                                  ->evaluate(old_phase.temperature, old_phase.absolute_pressure,
                                                      fractions(old_phase))
                                                  .internal_energy_temperature;
                                }
                                caloric_carrier_density.set_owned_value(
                                    cell, caloric / (ale ? ale->old_cell_volumes()[c] : mesh->cell_volume(cell)));
                            }
                        });
                    caloric_carrier_density.sync_ghosts();
                    const auto energy_overlap = transport_workspaces[g].conc_overlap->getData(k);
                    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                    {
                        const LO face = static_cast<LO>(f);
                        if (mesh->is_boundary_face(face))
                        {
                            continue;
                        }
                        const double phi = phase_fluxes[a]->local_value(face);
                        const LO owner = mesh->owner_cell(face);
                        const LO donor = phi >= 0 ? owner : mesh->opposite_or_periodic_neighbor_cell(face, owner);
                        local_energy_transport += dt * std::abs(phi * energy_overlap[donor]);
                        local_caloric_transport += dt * std::abs(phi * caloric_carrier_density.local_value(donor));
                    }
                }
            }
            multi_diag.pack_unpack_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - unpack_start).count();
        }
        multi_diag.execution_mode = multi_diag.batched_krylov_solves > 0
                                        ? MultiRhsExecutionMode::BatchedKrylov
                                        : (multi_diag.column_serial_solves > 0 ? MultiRhsExecutionMode::ColumnSerial
                                                                               : MultiRhsExecutionMode::SingleRhs);
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
            qbmm_detail::validate_stamps(d->options, candidate);
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
                    const auto& supplied_eos = cell.phases[a].thermodynamics;
                    if (!d->options.qbmm_families.empty() && d->options.qbmm_families[a] && supplied_eos)
                        check(supplied_eos->catalog().fingerprint() == d->eos[a]->catalog().fingerprint(),
                            "Initial native QBMM material refers to a foreign EOS catalog.");
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
                (void) qbmm_detail::validate_cell(d->options, cell);
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
    const auto face_rate = [&](LO face, const VectorField& velocity)
    {
        if (mesh.is_geometry_exterior_face(face))
            return 0.0;
        const LO owner = mesh.owner_cell(face), neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
        const double value =
            (velocity.local_value(owner) * 0.5 + velocity.local_value(neighbor) * 0.5).dot(mesh.face_area_vector(face));
        check(std::isfinite(value), "Initial low-Mach volume flux is nonfinite.");
        return value;
    };
    if (!d->options.qbmm_families.empty())
    {
        std::vector<double> face_flux;
        collective_detail::collective_local_validation(mesh, "Initial QBMM velocity and face storage",
            [&]
            {
                face_flux.resize(mesh.num_owned_faces());
                for (size_t c = 0; c < candidate.cells.size(); ++c)
                    d->old_u.set_owned_value(static_cast<LO>(c), candidate.cells[c].velocity);
            });
        d->old_u.sync_ghosts();
        collective_detail::collective_local_validation(mesh, "Initial QBMM face flux",
            [&]
            {
                for (size_t f = 0; f < face_flux.size(); ++f)
                    face_flux[f] = face_rate(static_cast<LO>(f), d->old_u);
            });
        d->publish_fixed(std::move(candidate), {}, face_flux, d->trial_nonce + 1);
        d->clear_initialization_provenance();
        return;
    }
    d->publish_fields(candidate);
    FaceField initial_flux(d->mesh, 0.0, "validated_initial_low_mach_flux");
    collective_detail::collective_local_validation(mesh, "Low-Mach initial face flux",
        [&]
        {
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const auto face = static_cast<LO>(f);
                initial_flux.set_owned_value(face, face_rate(face, d->u));
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
    d->clear_initialization_provenance();
}
LowMachMixtureSolver::Checkpoint LowMachMixtureSolver::checkpoint() const
{
    check(!d->accepted_planar_motion && !d->active_controller_motion,
        "Moving checkpoint requires its exact live motion authority.");
    return checkpoint_state();
}

LowMachMixtureSolver::Checkpoint LowMachMixtureSolver::checkpoint_state() const
{
    check(d->initialized, "Low-Mach solver is not initialized.");
    Checkpoint saved;
    saved.state = d->accepted;
    saved.owner = this;
    saved.geometry_epoch = d->epoch;
    saved.logical_geometry = d->logical_geometry;
    saved.generation = d->trial_nonce;
    saved.diagnostics = d->diagnostics;
    saved.qbmm_publication = d->qbmm_publication;
    saved.qbmm_controller_signature = d->checkpoint_controller();
    // A fresh state uses the attached controller's configuration. Accepted and
    // restored states retain their recorded acceleration, including its absence.
    if (d->options.qbmm_families.empty())
        saved.count_only_body_acceleration = d->count_only_force_provenance_recorded
                                                 ? d->accepted_count_only_body_acceleration
                                                 : d->active_body_acceleration;
    saved.qbmm_legacy_default_only = d->qbmm_legacy_default_only;
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
    std::shared_ptr<const QbmmPublication> publication;
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach checkpoint",
        [&]
        {
            check(saved.owner == this && !saved.ale_geometry && !d->accepted_planar_motion &&
                      !d->active_controller_motion && saved.logical_geometry == d->logical_geometry &&
                      d->epoch == mesh_geometry_epoch(*d->mesh) &&
                      (d->options.qbmm_families.empty() || saved.geometry_epoch == d->epoch) &&
                      saved.state.cells.size() == d->mesh->num_owned_cells() &&
                      saved.face_flux.size() == d->mesh->num_owned_faces(),
                "Low-Mach checkpoint belongs to a foreign solver, stale geometry or different maps.");
            check(!d->controller_active || d->active_qbmm_controller == saved.qbmm_controller_signature,
                "QBMM checkpoint belongs to a different coupled source configuration.");
            check(!saved.qbmm_legacy_default_only || !d->controller_active || d->active_legacy_compatible,
                "Legacy QBMM checkpoint requires default force and constitutive-law configuration.");
            qbmm_detail::validate_state(d->options, saved.state);
            candidate = saved.state;
            statistics = saved.diagnostics;
            if (!d->options.qbmm_families.empty())
            {
                auto copy = saved.qbmm_publication ? std::make_shared<QbmmPublication>(*saved.qbmm_publication)
                                                   : std::make_shared<QbmmPublication>();
                copy->origin = QbmmStateOrigin::Checkpoint;
                publication = std::move(copy);
            }
        });
    if (!d->options.qbmm_families.empty())
    {
        collective_detail::require_uniform_value(*d->mesh, saved.generation, "QBMM checkpoint generation");
        collective_detail::require_uniform_value(
            *d->mesh, candidate.thermodynamic_pressure, "QBMM checkpoint pressure");
        collective_detail::require_uniform_value(*d->mesh, candidate.time, "QBMM checkpoint time");
        collective_detail::require_uniform_value(*d->mesh, candidate.revision, "QBMM checkpoint revision");
        collective_detail::require_uniform_value(
            *d->mesh, int(saved.qbmm_controller_signature.has_value()), "QBMM checkpoint controller presence");
        collective_detail::require_uniform_value(
            *d->mesh, saved.qbmm_controller_signature.value_or(0), "QBMM checkpoint controller signature");
        collective_detail::require_uniform_value(
            *d->mesh, int(saved.qbmm_legacy_default_only), "QBMM checkpoint legacy configuration restriction");
    }
    // The physical checkpoint was accepted by this owner. Preserve its caloric state exactly.
    d->publish_fixed(
        std::move(candidate), std::move(statistics), saved.face_flux, d->trial_nonce + 1, std::move(publication));
    d->accepted_qbmm_controller = saved.qbmm_controller_signature;
    d->accepted_count_only_body_acceleration = saved.count_only_body_acceleration;
    d->count_only_force_provenance_recorded = d->options.qbmm_families.empty();
    d->needs_restart_controller = saved.qbmm_controller_signature.has_value();
    d->qbmm_legacy_default_only = saved.qbmm_legacy_default_only;
}

#include "solvers/details/LowMachPlanarALE.inc"
#include "solvers/details/LowMachQbmmReport.inc"
#include "solvers/details/LowMachQbmmRestart.inc"

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview(
    double dt, const std::vector<CellSources>& sources, const ControllerLease* controller)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::Preview);
    return preview_impl(dt, sources, nullptr, controller);
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_with_sources(
    double dt, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::PreviewWithSources);
    collective_detail::collective_local_validation(*d->mesh, "Source provider ownership",
        [&] { check(bool(provider), "preview_with_sources requires an immutable shared provider."); });
    return preview_impl(dt, {}, std::move(provider), controller);
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_with_qbmm_sources(
    double dt, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::PreviewWithQbmmSources);
    return preview_impl(dt, {}, std::move(provider), controller, nullptr, true);
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_impl(double dt, const std::vector<CellSources>& fixed_sources,
    std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller,
    const FVM::ALEControlVolumeState* ale, bool coherent_qbmm)
{
    std::vector<CellSources> sources;
    const auto& mesh = *d->mesh;
    const bool has_qbmm = !d->options.qbmm_families.empty();
    require_controller(controller);
    collective_detail::require_uniform_value(mesh, dt, "Low-Mach mixture timestep");
    collective_detail::require_uniform_value(mesh, int(ale != nullptr), "Low-Mach ALE transport selection");
    if (has_qbmm)
        collective_detail::require_uniform_value(mesh, int(coherent_qbmm), "Native QBMM source authority");
    if (ale)
        ale->validate(mesh, dt);
    collective_detail::collective_local_validation(mesh, "Low-Mach mixture step inputs",
        [&]
        {
            check(!coherent_qbmm || (has_qbmm && controller && controller->reacting_source_authority && provider),
                "Coherent QBMM sources require the reacting controller and an immutable source provider.");
            check(!has_qbmm || !provider || coherent_qbmm,
                "Native QBMM source providers require the coherent reacting controller.");
            check(!has_qbmm || ale || !d->accepted_planar_motion,
                "Accepted moving QBMM state requires a planar preview with live motion authority.");
            check(!(has_qbmm || ale) || d->preview_sequence != std::numeric_limits<std::uint64_t>::max(),
                "Low-Mach preview identity exhausted.");
            check(!d->needs_restart_controller || (coherent_qbmm && d->active_qbmm_controller &&
                                                      d->active_qbmm_controller == d->accepted_qbmm_controller),
                "QBMM restart requires the archived reacting controller before advancement.");
            sources = fixed_sources;
            check(d->initialized && d->epoch == mesh_geometry_epoch(mesh) && std::isfinite(dt) && dt > 0,
                "Low-Mach step requires initialized current geometry and a positive timestep.");
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
            qbmm_detail::validate_sources(d->options, sources);
            // Check the actual interval inputs before transport imports or source callbacks.
            // Initialization and restore have no trial duration and only validate physical state and schema.
            for (size_t c = 0; c < d->accepted.cells.size(); ++c)
            {
                qbmm_native_law_detail::validate_drift(d->accepted.cells[c]);
                const double volume = ale ? ale->old_cell_volumes()[c] : mesh.cell_volume(static_cast<LO>(c));
                qbmm_native_law_detail::validate_transport_gates(d->options, d->accepted.cells[c], volume, dt);
            }
        });
    collective_detail::require_uniform_value(mesh, int(bool(provider)), "Collective source-stage selection");
    // Distinguish coexisting previews even if their clocks and revisions coincide.
    const auto preview_id = has_qbmm || ale ? ++d->preview_sequence : 0;
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
                if (provider->body_acceleration)
                {
                    check(!ale, "Balanced body acceleration requires fixed native geometry.");
                    for (size_t axis = 0; axis < 3; ++axis)
                        check(std::isfinite(provider->body_acceleration->component(axis)),
                            "Source-stage body acceleration must be finite.");
                }
            });
        collective_detail::require_uniform_value(mesh, provider->identity.size(), "Source identity length");
        collective_detail::require_uniform_value(mesh, provider->revision, "Source provider revision");
        collective_detail::require_uniform_value(mesh, provider->schema_fingerprint, "Source provider schema");
        collective_detail::require_uniform_value(mesh, provider->relative_tolerance, "Source provider relative gate");
        collective_detail::require_uniform_value(
            mesh, provider->maximum_evaluations, "Source provider evaluation limit");
        collective_detail::require_uniform_value(
            mesh, int(provider->body_acceleration.has_value()), "Source provider body-acceleration presence");
        if (provider->body_acceleration)
            for (size_t axis = 0; axis < 3; ++axis)
                collective_detail::require_uniform_value(
                    mesh, provider->body_acceleration->component(axis), "Source provider body acceleration");
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
    const auto& qbmm_families = d->options.qbmm_families;
    TransportSnapshot last_transport;
    std::vector<PhaseFaceTransport> last_phase_transport;
    std::optional<TransportSnapshot> previous_transport;
    std::vector<CellSources> previous_sources;
    std::vector<std::vector<double>> gross_events, previous_events;
    FourierGrossResolutions previous_fourier_resolution;
    QbmmSourceCells qbmm_cells;
    std::optional<bool> event_channel, conditioned_channel;
    std::shared_ptr<const PressureConditionedThermalBatch> thermal_batch, previous_thermal_batch;
    std::vector<PressureConditionedThermalEvaluation> last_thermal_evaluations, previous_thermal_evaluations;
    std::optional<TransportSnapshot> previous_conditioned_transport;
    std::optional<std::string> conditioned_signature;
    std::vector<std::vector<double>> advected_energy;
    d->publish_fields(accepted, false, ale ? ale->old_cell_volumes() : std::span<const double>{});
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
            const auto is_qbmm_bound = [&](size_t a) { return !qbmm_families.empty() && qbmm_families[a]; };
            std::vector<std::unique_ptr<ScalarField>> transported_species(
                d->options.use_multi_rhs_transport && qbmm_families.empty() ? 0 : np * ns);
            std::vector<Implementation::CarrierSpeciesSlot> species_slots;
            const auto species_donor_concentration = [&](size_t a, size_t s, LO donor) -> double
            {
                if (d->options.use_multi_rhs_transport && !is_qbmm_bound(a))
                {
                    const auto& slot = species_slots[a * ns + s];
                    return d->transport_workspaces[slot.group_index].conc_overlap->getData(slot.column_index)[donor];
                }
                return transported_species[a * ns + s]->local_value(donor);
            };
            std::vector<std::unique_ptr<FaceField>> material_face_fraction(np);
            std::vector<std::vector<double>> small_fraction_volume_flux(np);
            std::vector<std::unique_ptr<VectorField>> drift(np);
            std::vector<std::unique_ptr<FaceField>> phase_flux(np);
            std::vector<std::vector<QbmmFaceFlux>> qbmm_face_flux(np);
            std::vector<double> before(nc), after;
            if (ale)
            {
                for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                    if (mesh.is_geometry_exterior_face(static_cast<LO>(f)))
                        d->trial_flux.set_owned_value(static_cast<LO>(f), ale->face_mesh_fluxes()[f]);
                d->trial_flux.sync_ghosts();
            }
            for (size_t a = 0; a < np; ++a)
            {
                drift[a] = std::make_unique<VectorField>(d->mesh, "phase_drift");
                phase_flux[a] = std::make_unique<FaceField>(d->mesh, 0.0, "phase_carrier_slip_flux");
                for (size_t c = 0; c < nc; ++c)
                    drift[a]->set_owned_value(static_cast<LO>(c), accepted.cells[c].drift_velocity[a]);
                drift[a]->sync_ghosts();
                collective_detail::collective_local_validation(mesh, "Supplied phase-slip face velocities",
                    [&]
                    {
                        for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                        {
                            const auto face = static_cast<LO>(f);
                            if (mesh.is_boundary_face(face))
                            {
                                // Impermeable moving wall: relative phase advection is exactly zero.
                                phase_flux[a]->set_owned_value(face, 0.);
                                continue;
                            }
                            const auto owner = mesh.owner_cell(face),
                                       neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                            Vector slip;
                            for (size_t axis = 0; axis < 3; ++axis)
                                slip.component(axis) = std::midpoint(drift[a]->local_value(owner).component(axis),
                                    drift[a]->local_value(neighbor).component(axis));
                            const double phi = d->trial_flux.local_value(face) -
                                               (ale ? ale->face_mesh_fluxes()[f] : 0.) +
                                               slip.dot(mesh.face_area_vector(face));
                            check(std::isfinite(phi), "Supplied phase-slip face flux is not representable.");
                            phase_flux[a]->set_owned_value(face, phi);
                        }
                    });
                phase_flux[a]->sync_ghosts();
            }
            if (d->options.use_multi_rhs_transport)
            {
                d->transport_all_carriers(accepted, trial, phase_flux, dt, iteration, result, species_slots,
                    local_caloric_transport, local_energy_transport, ale);
            }
            for (size_t a = 0; a < np; ++a)
            {
                const bool qbmm_bound = is_qbmm_bound(a);
                if (qbmm_bound || !d->options.use_multi_rhs_transport)
                {
                    if (qbmm_bound)
                        qbmm_face_flux[a] = d->transport_qbmm(accepted, trial, a, *phase_flux[a], dt, result, ale);
                    else if (d->options.phase_roles[a].role == PhaseRole::DispersedGas)
                    {
                        collective_detail::collective_local_validation(mesh, "Population transport input",
                            [&]
                            {
                                for (size_t c = 0; c < nc; ++c)
                                {
                                    before[c] = *accepted.cells[c].phase_numbers[a];
                                    check(std::isfinite(before[c] / (ale ? ale->old_cell_volumes()[c]
                                                                         : mesh.cell_volume(static_cast<LO>(c)))),
                                        "Population number concentration is not representable.");
                                }
                            });
                        d->transport(before, *phase_flux[a], dt, after, result,
                            transport_consistency_detail::Quantity::Population, ale);

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
                        std::optional<LO> worst_number_cell;
                        std::array<long double, 3> worst_number_operands{};
                        collective_detail::collective_local_validation(mesh, "Physical local population balance",
                            [&]
                            {
                                for (size_t c = 0; c < nc; ++c)
                                {
                                    const long double flux = static_cast<long double>(dt) *
                                                             FVM::cell_flux_balance(mesh, number_flux, static_cast<LO>(c));
                                    const long double residual = static_cast<long double>(after[c]) - before[c] + flux;
                                    const double error = source_stage_detail::normalized(residual,
                                        std::abs(static_cast<long double>(after[c])) +
                                            std::abs(static_cast<long double>(before[c])) + std::abs(flux),
                                        d->options.conservation_relative_tolerance);
                                    if (error > local_number_error)
                                    {
                                        local_number_error = error;
                                        worst_number_cell = static_cast<LO>(c);
                                        worst_number_operands = {before[c], after[c], flux};
                                    }
                                }
                            });
                        const double count_error = d->maximum(local_number_error);
                        result.maximum_population_transport_residual =
                            std::max(result.maximum_population_transport_residual, count_error);
                        if (count_error > 1)
                        {
                            std::ostringstream message;
                            message.precision(17);
                            message << "Population transport fails its physical local count-balance gate: normalized="
                                    << count_error << ", local=" << local_number_error;
                            if (worst_number_cell)
                                message << ", geometry cell=" << mesh.cell_geometry_global_id(*worst_number_cell)
                                        << ", phase=" << a << ", before=" << worst_number_operands[0]
                                        << ", after=" << worst_number_operands[1]
                                        << ", integrated outgoing=" << worst_number_operands[2];
                            throw std::runtime_error(message.str());
                        }
                    }
                    for (size_t s = 0; s < ns; ++s)
                    {
                        if (!qbmm_bound)
                        {
                            for (size_t c = 0; c < nc; ++c)
                                before[c] = accepted.cells[c].phases[a].moles[s];
                            d->transport(before, *phase_flux[a], dt, after, result,
                                transport_consistency_detail::Quantity::Species, ale);
                            for (size_t c = 0; c < nc; ++c)
                                trial.cells[c].phases[a].moles[s] = after[c];
                        }
                        auto field = std::make_unique<ScalarField>(d->mesh, "transported_species_concentration");
                        for (size_t c = 0; c < nc; ++c)
                            field->set_owned_value(static_cast<LO>(c),
                                trial.cells[c].phases[a].moles[s] / mesh.cell_volume(static_cast<LO>(c)));
                        field->sync_ghosts();
                        transported_species[a * ns + s] = std::move(field);
                    }
                    std::optional<ScalarField> transported_energy_density;
                    if (!qbmm_bound)
                    {
                        for (size_t c = 0; c < nc; ++c)
                            before[c] = accepted.cells[c].phases[a].internal_energy;
                        d->transport(before, *phase_flux[a], dt, after, result,
                            transport_consistency_detail::Quantity::Energy, ale);
                        for (size_t c = 0; c < nc; ++c)
                            trial.cells[c].phases[a].internal_energy = after[c];
                        transported_energy_density.emplace(d->mesh, "transported_energy_operand");
                    }
                    ScalarField caloric_carrier_density(d->mesh, "caloric_transport_characteristic");
                    collective_detail::collective_local_validation(mesh, "Low-Mach caloric transport scales",
                        [&]
                        {
                            for (size_t c = 0; c < nc; ++c)
                            {
                                const LO cell = static_cast<LO>(c);
                                if (transported_energy_density)
                                    transported_energy_density->set_owned_value(cell, after[c] / mesh.cell_volume(cell));
                                const auto& old_phase = accepted.cells[c].phases[a];
                                const double mass = phase_mass(old_phase);
                                double caloric = 0;
                                if (mass > 0)
                                    caloric = mass * old_phase.temperature *
                                              old_phase.thermodynamics
                                                  ->evaluate(old_phase.temperature, old_phase.absolute_pressure,
                                                      fractions(old_phase))
                                                  .internal_energy_temperature;
                                caloric_carrier_density.set_owned_value(
                                    cell, caloric / (ale ? ale->old_cell_volumes()[c] : mesh.cell_volume(cell)));
                            }
                        });
                    if (transported_energy_density)
                        transported_energy_density->sync_ghosts();
                    caloric_carrier_density.sync_ghosts();
                    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                    {
                        const LO face = static_cast<LO>(f);
                        if (mesh.is_boundary_face(face))
                            continue;
                        const double phi = phase_flux[a]->local_value(face);
                        const LO owner = mesh.owner_cell(face);
                        const LO donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                        const double energy_flux = qbmm_bound ? qbmm_face_flux[a][f].internal_energy_flux
                                                              : phi * transported_energy_density->local_value(donor);
                        local_energy_transport += dt * std::abs(energy_flux);
                        local_caloric_transport += dt * std::abs(phi * caloric_carrier_density.local_value(donor));
                    }
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
                            if (mesh.is_boundary_face(face) && !ale)
                                continue;
                            const double phi = phase_flux[a]->local_value(face);
                            const LO owner = mesh.owner_cell(face);
                            const LO donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                            std::vector<double> x(ns);
                            double amount = 0;
                            FaceSum transported_mass_density;
                            for (size_t s = 0; s < ns; ++s)
                            {
                                x[s] = species_donor_concentration(a, s, donor);
                                amount += x[s];
                                transported_mass_density.add_product(x[s], d->eos[a]->catalog().species()[s].molarMass);
                            }
                            if (amount == 0)
                                continue;
                            if (!std::isfinite(amount))
                            {
                                const double scale = *std::max_element(x.begin(), x.end());
                                for (double& value : x)
                                    value /= scale;
                                amount = std::accumulate(x.begin(), x.end(), 0.);
                            }
                            for (double& value : x)
                                value /= amount;
                            const auto point = d->eos[a]->evaluate(
                                phase_temperature.local_value(donor), phase_pressure.local_value(donor), x);
                            const auto [fraction, volume_flux] =
                                qbmm_face_detail::material_volume_rates(transported_mass_density, point.density, phi);
                            check(std::isfinite(fraction) && fraction >= 0 && std::isfinite(volume_flux),
                                "Phase material-volume packet is not representable.");
                            material_face_fraction[a]->set_owned_value(face, fraction);
                            material_volume_flux.set_owned_value(face, volume_flux);
                            if (!std::isnormal(fraction) && volume_flux != 0)
                            {
                                auto& rates = small_fraction_volume_flux[a];
                                if (rates.empty())
                                    rates.resize(mesh.num_owned_faces());
                                rates[f] = volume_flux;
                            }
                            if (qbmm_bound)
                                qbmm_face_flux[a][f].material_volume_flux = volume_flux;
                        }
                    });
                material_face_fraction[a]->sync_ghosts();
                material_volume_flux.sync_ghosts();
                collective_detail::collective_local_validation(mesh, "Phase transport stock/material-volume support",
                    [&]
                    {
                        for (size_t c = 0; c < nc; ++c)
                        {
                            adv_volumes[c][a] = old_volumes[c][a] - dt * FVM::cell_flux_balance(mesh,
                                                                             material_volume_flux, static_cast<LO>(c));
                            const auto& transported = trial.cells[c].phases[a];
                            check(phase_mass(transported) > 0 ||
                                      (transported.internal_energy == 0 && adv_volumes[c][a] == 0),
                                "Locally empty phase has inconsistent canonical transport energy/volume packets.");
                        }
                    });
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
                    advected_energy.resize(nc);
                    last_transport.old_control_volumes.resize(nc);
                    last_transport.new_control_volumes.resize(nc);
                    last_transport.accepted_revision = accepted.revision;
                    last_transport.accepted_generation = d->trial_nonce;
                    last_transport.old_geometry_epoch = ale ? ale->old_geometry_epoch() : d->epoch;
                    last_transport.new_geometry_epoch = ale ? ale->new_geometry_epoch() : d->epoch;
                    last_transport.schema_fingerprint = d->eos.front()->catalog().fingerprint();
                    last_transport.qbmm_families = accepted.qbmm_families;
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
                        packet.mesh_volume_flux = ale ? ale->face_mesh_fluxes()[f] : 0.;
                        packet.relative_phase_volume_flux.resize(np);
                        if (!qbmm_families.empty())
                            packet.qbmm_flux.resize(np);
                        check(std::isfinite(packet.carrier_volume_flux), "Nonfinite stage carrier flux receipt.");
                        for (size_t a = 0; a < np; ++a)
                        {
                            if (!qbmm_families.empty() && qbmm_families[a])
                                packet.qbmm_flux[a] = qbmm_face_flux[a][f];
                            packet.relative_phase_volume_flux[a] = phase_flux[a]->local_value(face);
                            packet.phase_volume_flux[a] =
                                packet.relative_phase_volume_flux[a] + packet.mesh_volume_flux;
                            packet.material_fraction[a] = material_face_fraction[a]->local_value(face);
                            check(std::isfinite(packet.phase_volume_flux[a]) &&
                                      std::isfinite(packet.material_fraction[a]) && packet.material_fraction[a] >= 0,
                                "Invalid exact phase-face transport receipt.");
                        }
                    }
                    for (size_t c = 0; c < nc; ++c)
                    {
                        last_transport.old_control_volumes[c] =
                            ale ? ale->old_cell_volumes()[c] : mesh.cell_volume(static_cast<LO>(c));
                        last_transport.new_control_volumes[c] =
                            ale ? ale->new_cell_volumes()[c] : last_transport.old_control_volumes[c];
                        last_transport.cells[c].phases.resize(np);
                        advected_energy[c].resize(np);
                        for (size_t a = 0; a < np; ++a)
                        {
                            auto& inventory = last_transport.cells[c].phases[a];
                            inventory.moles = trial.cells[c].phases[a].moles;
                            inventory.internal_energy = trial.cells[c].phases[a].internal_energy;
                            advected_energy[c][a] = inventory.internal_energy;
                            inventory.advected_material_volume = adv_volumes[c][a];
                            inventory.population_number = trial.cells[c].phase_numbers[a];
                            if (!qbmm_families.empty())
                                inventory.qbmm_shape = trial.cells[c].qbmm_shape[a];
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
                        check(evaluated.fourier_gross_resolution.empty() ||
                                  (controller && controller->reacting_source_authority),
                            "Fourier gross resolution requires the private reacting-controller lease.");
                        source_stage_detail::validate_fourier_resolution(
                            evaluated.fourier_gross_resolution, evaluated.gross_events);
                        source_stage_detail::validate_number_packets(evaluated.sources, nc, d->options.phase_roles);
                        qbmm_detail::validate_sources(d->options, evaluated.sources, coherent_qbmm);
                        if (coherent_qbmm)
                            qbmm_detail::validate_source_receipts(d->options, last_transport, evaluated);
                        else
                            check(
                                evaluated.qbmm_cells.empty(), "QBMM source receipts require the coherent controller.");
                    });
                collective_detail::require_uniform_value(
                    mesh, int(bool(evaluated.thermal_receipt)), "Conditioned thermal batch presence");
                std::string thermal_signature;
                collective_detail::collective_local_validation(mesh, "Conditioned thermal source receipt",
                    [&]
                    {
                        check(!conditioned_channel || *conditioned_channel == bool(evaluated.thermal_receipt),
                            "Source changed conditioned thermal ownership within one preview.");
                        if (evaluated.thermal_receipt)
                        {
                            check(!thermal_batch || thermal_batch->configuration.get() ==
                                                        evaluated.thermal_receipt->configuration.get(),
                                "Conditioned source replaced its configured law object within one preview.");
                            conditioned_thermal_detail::validate_batch(*evaluated.thermal_receipt,
                                SourceStageContext{last_transport, thermodynamic_guess, mesh}, evaluated.sources);
                            thermal_signature =
                                conditioned_thermal_detail::signature(*evaluated.thermal_receipt->configuration);
                            check(thermal_signature.size() <= size_t(std::numeric_limits<int>::max()),
                                "Conditioned law schema is too large.");
                            check(!conditioned_signature || *conditioned_signature == thermal_signature,
                                "Conditioned law/slot metadata changed within one preview.");
                            last_thermal_evaluations.resize(nc);
                        }
                    });
                if (evaluated.thermal_receipt)
                {
                    collective_detail::require_uniform_value(
                        mesh, thermal_signature.size(), "Conditioned law signature length");
                    std::string rank_zero_signature;
                    collective_detail::collective_local_validation(
                        mesh, "Conditioned law signature buffer", [&] { rank_zero_signature = thermal_signature; });
                    Teuchos::broadcast(*mesh.owned_cell_map()->getComm(), 0,
                        static_cast<int>(rank_zero_signature.size()), rank_zero_signature.data());
                    collective_detail::collective_local_validation(mesh, "Conditioned law signature equality",
                        [&]
                        {
                            check(rank_zero_signature == thermal_signature,
                                "Conditioned configured laws/slots differ across ranks.");
                            conditioned_signature = thermal_signature;
                        });
                }
                conditioned_channel = bool(evaluated.thermal_receipt);
                thermal_batch = std::move(evaluated.thermal_receipt);
                int local_events = !evaluated.gross_events.empty(), global_events = 0;
                Teuchos::reduceAll(
                    *mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_events, &global_events);
                size_t qbmm_event_width = 0;
                if (coherent_qbmm)
                {
                    const size_t local_width = local_events ? evaluated.gross_events.front().size() : 0;
                    Teuchos::reduceAll(
                        *mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_width, &qbmm_event_width);
                }
                collective_detail::collective_local_validation(mesh, "Source-stage event schema",
                    [&]
                    {
                        check(!global_events || evaluated.gross_events.size() == nc,
                            "Enabled gross-event channel requires one event vector per owned cell.");
                        check(!event_channel || *event_channel == bool(global_events),
                            "Source stage changed its gross-event channel between correctors.");
                        if (coherent_qbmm)
                            for (const auto& row : evaluated.gross_events)
                                check(row.size() == qbmm_event_width, "QBMM gross-event schema differs across owners.");
                        if (previous_transport)
                            source_stage_detail::validate_event_shape(evaluated.gross_events, previous_events);
                    });
                event_channel = bool(global_events);
                sources = std::move(evaluated.sources);
                gross_events = std::move(evaluated.gross_events);
                double event_residual = global_events ? std::numeric_limits<double>::infinity() : 0;
                double fourier_uncertainty = 0;
                double source_residual = std::numeric_limits<double>::infinity();
                double transport_residual = std::numeric_limits<double>::infinity();
                double shape_residual = coherent_qbmm ? std::numeric_limits<double>::infinity() : 0;
                if (previous_transport)
                {
                    collective_detail::collective_local_validation(mesh, "Source/transport convergence",
                        [&]
                        {
                            const auto comparison = source_stage_detail::event_error(gross_events, previous_events,
                                source_options.source_relative_tolerance, evaluated.fourier_gross_resolution,
                                previous_fourier_resolution);
                            event_residual = comparison.residual;
                            fourier_uncertainty = comparison.fourier_uncertainty;
                            source_residual = source_stage_detail::source_error(sources, previous_sources,
                                last_transport, thermodynamic_guess, accepted, source_options);
                            transport_residual = source_stage_detail::transport_error(
                                last_transport, *previous_transport, thermodynamic_guess, accepted, d->options);
                            if (coherent_qbmm)
                                shape_residual = source_stage_detail::shape_source_error(sources, previous_sources,
                                    evaluated.qbmm_cells, qbmm_cells, last_transport,
                                    source_options.source_relative_tolerance);
                        });
                }
                qbmm_cells = std::move(evaluated.qbmm_cells);
                result.source_evaluations = iteration + 1;
                const std::array local_event_comparison{event_residual, fourier_uncertainty};
                std::array<double, 2> global_event_comparison;
                Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
                    int(local_event_comparison.size()), local_event_comparison.data(), global_event_comparison.data());
                result.maximum_event_residual = global_event_comparison[0];
                if (previous_transport)
                    result.maximum_fourier_gross_comparison_uncertainty_j = global_event_comparison[1];
                result.maximum_source_residual = d->maximum(source_residual);
                result.maximum_transport_residual = d->maximum(transport_residual);
                if (coherent_qbmm)
                    result.maximum_shape_source_residual = d->maximum(shape_residual);
                collective_detail::collective_local_validation(mesh, "Retained source corrector receipt",
                    [&]
                    {
                        previous_transport = last_transport;
                        previous_sources = sources;
                        previous_events = gross_events;
                        previous_fourier_resolution = std::move(evaluated.fourier_gross_resolution);
                    });
            }
            collective_detail::collective_local_validation(mesh, "Low-Mach integrated local sources",
                [&]
                {
                    for (size_t c = 0; c < nc; ++c)
                        for (size_t a = 0; a < np; ++a)
                        {
                            const bool qbmm_bound = !qbmm_families.empty() && qbmm_families[a];
                            auto& cell = trial.cells[c];
                            auto& phase = cell.phases[a];
                            if (coherent_qbmm && qbmm_bound)
                            {
                                const auto& endpoint = *qbmm_cells[c][a];
                                phase.moles = endpoint.final_moles;
                                cell.phase_numbers[a] = endpoint.final_count;
                                cell.qbmm_shape[a] = endpoint.final_shape;
                            }
                            for (size_t s = 0; s < ns; ++s)
                            {
                                if (!qbmm_bound && !sources.empty() && !sources[c].species_moles.empty())
                                    phase.moles[s] += sources[c].species_moles[a][s];
                                check(std::isfinite(phase.moles[s]) && phase.moles[s] >= 0,
                                    "Low-Mach source/transport overdraws phase species.");
                            }
                            if (!qbmm_bound && !sources.empty() && !sources[c].population_number_changes.empty() &&
                                cell.phase_numbers[a])
                                *cell.phase_numbers[a] += sources[c].population_number_changes[a];
                            if (!sources.empty() && !sources[c].internal_energy.empty())
                                phase.internal_energy += sources[c].internal_energy[a];
                        }
                });
            collective_detail::collective_local_validation(mesh, "Joint phase mole/number source admissibility",
                [&]
                {
                    for (const auto& cell : trial.cells)
                        source_stage_detail::validate_phase_numbers(cell, d->options.phase_roles);
                    if (coherent_qbmm)
                        qbmm_detail::validate_state(d->options, trial);
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
                        if (thermal_batch)
                        {
                            const auto domain =
                                interphase_pressure_bounds(thermal_batch->cells[c], trial.cells[c].phases);
                            local_lower = std::max(local_lower, domain.lower - trial.cells[c].mechanical_pressure);
                            local_upper = std::min(local_upper, domain.upper - trial.cells[c].mechanical_pressure);
                        }
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
            bool saw_positive = false, saw_negative = false;
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
                            // Composite anchors/partial h are NOT certified affine by the final EOS kind/capability.
                            auto domain =
                                thermal_batch ? CaloricPressureDomain{} : exact_caloric_pressure_domain(query);
                            bool supplied = false;
                            if (thermal_batch && thermal_batch->configuration->caloric_domain)
                            {
                                const ConditionedCaloricQuery whole{thermal_batch->cells[c], trial.cells[c].phases,
                                    advected_energy[c], adv_volumes[c], {lower, upper},
                                    trial.cells[c].mechanical_pressure, d->options.thermal_mode};
                                domain = thermal_batch->configuration->caloric_domain->evaluate(whole);
                                supplied = true;
                            }
                            if (!thermal_batch && domain.knowledge == CaloricDomainKnowledge::Unresolved &&
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
                                    thermal[c] = recover_enthalpy(trial.cells[c], adv_volumes[c], candidate, d->options,
                                        thermal_batch ? &thermal_batch->cells[c] : nullptr, advected_energy[c]);
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
                                thermal[c] = recover_enthalpy(trial.cells[c], adv_volumes[c], p0, d->options,
                                    thermal_batch ? &thermal_batch->cells[c] : nullptr, advected_energy[c]);
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
                {
                    saw_positive = true;
                    lower = std::max(lower, p0);
                }
                else if (residual < 0)
                {
                    saw_negative = true;
                    upper = std::min(upper, p0);
                }
                double next = p0 - residual / derivative;
                if (!std::isfinite(next) || next <= lower || next >= upper)
                    next = std::midpoint(lower, upper);
                if (next == p0 && saw_positive && saw_negative &&
                    std::abs(residual) <= d->options.volume_relative_tolerance * domain_volume)
                    break;
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
            const bool qbmm_enabled = !qbmm_families.empty();
            collective_detail::collective_local_validation(mesh, "Phase mass/material momentum packets",
                [&]
                {
                    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                    {
                        const LO face = static_cast<LO>(f);
                        if (mesh.is_boundary_face(face) && !ale)
                        {
                            d->mass_flux.set_owned_value(face, 0);
                            continue;
                        }
                        const LO owner = mesh.owner_cell(face),
                                 neighbor = mesh.is_boundary_face(face)
                                                ? owner
                                                : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                        auto& receipt = last_phase_transport[f];
                        double volume_slip = 0, theta = 0;
                        const double relative_carrier =
                            d->trial_flux.local_value(face) - (ale ? ale->face_mesh_fluxes()[f] : 0.);
                        Vector drift_flux{};
                        FaceSum mixture_mass_rate;
                        std::array<FaceSum, 3> mixture_drift_rate;
                        for (size_t a = 0; a < np; ++a)
                        {
                            const double phi = phase_flux[a]->local_value(face);
                            const LO donor = phi >= 0 ? owner : neighbor;
                            const auto donor_drift = drift[a]->local_value(donor);
                            const auto& species = d->eos[a]->catalog().species();
                            auto* packet = qbmm_enabled && receipt.qbmm_flux[a] ? &*receipt.qbmm_flux[a] : nullptr;
                            FaceSum phase_mass_rate;
                            if (packet)
                            {
                                for (size_t s = 0; s < ns; ++s)
                                    phase_mass_rate.add_product(packet->species_molar_flux[s], species[s].molarMass);
                                packet->derived_mass_flux = phase_mass_rate.rounded();
                                check(std::isfinite(packet->derived_mass_flux),
                                    "QBMM derived physical face packet overflow.");
                            }
                            else if (phi != 0)
                                for (size_t s = 0; s < ns; ++s)
                                    phase_mass_rate.add_product(
                                        phi, species_donor_concentration(a, s, donor), species[s].molarMass);

                            // Weight by donor drift and sum signed phase contributions before rounding
                            // to the face fields. The phase receipts store their own rounded values.
                            mixture_mass_rate.add_weighted(phase_mass_rate, 1.);
                            for (size_t axis = 0; axis < 3; ++axis)
                            {
                                const double drift_component = donor_drift.component(axis);
                                if (drift_component == 0)
                                    continue;
                                mixture_drift_rate[axis].add_weighted(phase_mass_rate, drift_component);
                                if (packet)
                                {
                                    FaceSum phase_drift_rate;
                                    phase_drift_rate.add_weighted(phase_mass_rate, drift_component);
                                    auto& component = packet->derived_drift_momentum_flux.component(axis);
                                    component = phase_drift_rate.rounded();
                                    check(std::isfinite(component), "QBMM drift momentum packet is not representable.");
                                }
                            }
                            const double phase_fraction = material_face_fraction[a]->local_value(face);
                            theta += phase_fraction;
                            if (std::isnormal(phase_fraction))
                                volume_slip += (phi - relative_carrier) * phase_fraction;
                            else
                            {
                                // Keep the directly rounded material rate in the pressure split.
                                const auto& rates = small_fraction_volume_flux[a];
                                volume_slip +=
                                    std::fma(-relative_carrier, phase_fraction, rates.empty() ? 0. : rates[f]);
                            }
                        }
                        const double mass_flux = mixture_mass_rate.rounded();
                        for (size_t axis = 0; axis < 3; ++axis)
                            drift_flux.component(axis) = mixture_drift_rate[axis].rounded();
                        check(std::isfinite(mass_flux) && std::isfinite(volume_slip) && std::isfinite(theta) &&
                                  theta >= 0,
                            "Phase mass/material flux packet overflow.");
                        d->mass_flux.set_owned_value(face, mass_flux);
                        receipt.momentum_mass_flux = mass_flux;
                        slip_volume.set_owned_value(face, volume_slip);
                        carrier_fraction.set_owned_value(face, theta);
                        for (size_t component = 0; component < 3; ++component)
                        {
                            check(std::isfinite(drift_flux.component(component)),
                                "Phase physical drift-momentum flux overflow.");
                            drift_momentum[component]->set_owned_value(face, drift_flux.component(component));
                        }
                        receipt.drift_momentum_flux = drift_flux;
                    }
                });
            d->mass_flux.sync_ghosts();
            slip_volume.sync_ghosts();
            carrier_fraction.sync_ghosts();
            for (auto& field : drift_momentum)
                field->sync_ghosts();
            FaceField theta_mesh_correction(d->mesh, 0., "theta_dependent_mesh_volume_correction");
            if (ale)
            {
                for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                    theta_mesh_correction.set_owned_value(static_cast<LO>(f),
                        carrier_fraction.local_value(static_cast<LO>(f)) * ale->face_mesh_fluxes()[f]);
                theta_mesh_correction.sync_ghosts();
            }
            double occupancy = 0, local_work = 0, local_roundoff = 0, local_work_magnitude = 0,
                   local_pressure_operands = 0, local_raw_roundoff = 0;
            collective_detail::collective_local_validation(mesh, "Low-Mach EOS transport candidate",
                [&]
                {
                    for (size_t c = 0; c < nc; ++c)
                    {
                        const LO cell = static_cast<LO>(c);
                        const double volume = mesh.cell_volume(cell);
                        trial.cells[c] = thermal[c].cell;
                        if (thermal_batch)
                        {
                            check(thermal[c].conditioned.has_value(), "Missing final conditioned caloric evaluation.");
                            last_thermal_evaluations[c] = *thermal[c].conditioned;
                            local_raw_roundoff += thermal[c].conditioned->raw_channel_roundoff_bound;
                        }
                        const double advected_volume =
                            std::accumulate(adv_volumes[c].begin(), adv_volumes[c].end(), 0.0);
                        volume_susceptibility[c] = -thermal[c].derivative;
                        check(std::isfinite(volume_susceptibility[c]) && volume_susceptibility[c] >= 0,
                            "Low-Mach local EOS volume susceptibility is not stable and nonnegative.");
                        target[c] = (thermal[c].volume - advected_volume) / dt;
                        if (ale)
                        {
                            const double old_material =
                                std::accumulate(old_volumes[c].begin(), old_volumes[c].end(), 0.);
                            // Unsimplified identity retains actual accepted occupancy/GCL operands.
                            // With accepted occupancy and exact GCL this is S + D[(theta-1)*w].
                            target[c] += (old_material - volume) / dt +
                                         FVM::cell_flux_balance(mesh, theta_mesh_correction, cell);
                        }
                        occupancy = std::max(occupancy, std::abs(thermal[c].volume - volume) / volume);
                        local_work += thermal[c].work;
                        local_work_magnitude += thermal[c].work_magnitude;
                        local_pressure_operands += thermal[c].pressure_operands;
                        local_roundoff += thermal[c].roundoff;
                        double mass = 0;
                        for (const auto& phase : trial.cells[c].phases)
                            mass += phase_mass(phase);
                        d->rho.set_owned_value(cell, mass / volume);
                        qbmm_native_law_detail::validate_transport_gates(
                            d->options, trial.cells[c], last_transport.new_control_volumes[c], dt, thermal[c].volumes);
                    }
                });
            // All ranks finish checking the recovered state before any rank calls the viscosity law.
            collective_detail::collective_local_validation(mesh, "Low-Mach viscosity evaluation",
                [&]
                {
                    for (size_t c = 0; c < nc; ++c)
                    {
                        const LO cell = static_cast<LO>(c);
                        d->mu.set_owned_value(cell, effective_mixture_viscosity(d->options, trial.cells[c].phases,
                                                        thermal[c].volumes, ale != nullptr));
                    }
                });
            d->rho.sync_ghosts();
            d->mu.sync_ghosts();
            result.pressure_work = d->sum(local_work);
            phase_work_magnitude = d->sum(local_work_magnitude);
            pressure_operand_magnitude = d->sum(local_pressure_operands);
            result.empty_phase_energy_roundoff = d->sum(local_roundoff);
            result.conditioned_raw_energy_roundoff_bound = d->sum(local_raw_roundoff);
            const bool body_forced = provider && provider->body_acceleration &&
                                     (provider->body_acceleration->x != 0 || provider->body_acceleration->y != 0 ||
                                         provider->body_acceleration->z != 0);
            if (body_forced)
                d->balanced_pressure_gradient(*provider->body_acceleration);
            else
                FVM::cell_gradient(d->mechanical_pressure, BoundaryConditionMap{}, d->pressure_gradient,
                    d->gradient_cache, FVM::CellGradientScheme::LeastSquares);
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
                {}, {}, d->options.linear, FVM::NonOrthogonalTreatment::Implicit, ale);
            FaceField predictor(d->mesh, 0.0, "low_mach_velocity_predictor_flux");
            std::vector<double> previous_flux(mesh.num_owned_faces());
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                previous_flux[f] = d->trial_flux.local_value(face);
                if (mesh.is_boundary_face(face))
                {
                    predictor.set_owned_value(face, ale ? ale->face_mesh_fluxes()[f] : 0.);
                    continue;
                }
                const LO owner = mesh.owner_cell(face), neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                auto velocity = (d->u.local_value(owner) + d->u.local_value(neighbor)) * 0.5;
                velocity = velocity + (d->pressure_gradient.local_value(owner) / d->rho.local_value(owner) +
                                          d->pressure_gradient.local_value(neighbor) / d->rho.local_value(neighbor)) *
                                          (0.5 * dt);
                const double mean_density = .5 * d->rho.local_value(owner) + .5 * d->rho.local_value(neighbor);
                const double pressure_flux =
                    dt / mean_density * mesh.face_area(face) / mesh.face_cell_center_distance(face) *
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
            const auto projected =
                ale ? d->pressure->project_material_ale_with_eos(d->rho, dt, target, volume_susceptibility,
                          carrier_fraction, slip_volume, predictor, d->trial_flux, d->mechanical_pressure, d->u, *ale,
                          d->options.linear, d->options.continuity_tolerance)
                    : d->pressure->project_material_with_eos(d->rho, dt, target, volume_susceptibility,
                          carrier_fraction, slip_volume, predictor, d->trial_flux, d->mechanical_pressure, d->u,
                          d->options.linear, d->options.continuity_tolerance,
                          body_forced ? LowMachPressureEquation::CellPressureGradient::Interior
                                      : LowMachPressureEquation::CellPressureGradient::HomogeneousNeumann);
            trial.thermodynamic_pressure += projected.thermodynamic_pressure_increment;
            check(std::isfinite(trial.thermodynamic_pressure), "Uniform pressure gauge increment overflow.");
            result.pressure_linear = projected.linear;
            result.pressure_linear_rhs_norm_floor = projected.linear_rhs_norm_floor;
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
            if (thermal_batch)
            {
                double local_conditioned = std::numeric_limits<double>::infinity();
                collective_detail::collective_local_validation(mesh, "Actual conditioned energy/plan convergence",
                    [&]
                    {
                        if (previous_thermal_batch && previous_conditioned_transport)
                            local_conditioned =
                                conditioned_thermal_detail::convergence(*thermal_batch, *previous_thermal_batch,
                                    last_thermal_evaluations, previous_thermal_evaluations, last_transport,
                                    *previous_conditioned_transport, source_options.source_relative_tolerance);
                        previous_thermal_batch = thermal_batch;
                        previous_thermal_evaluations = last_thermal_evaluations;
                        previous_conditioned_transport = last_transport;
                    });
                result.maximum_conditioned_thermal_residual = d->maximum(local_conditioned);
                result.maximum_source_residual =
                    std::max(result.maximum_source_residual, result.maximum_conditioned_thermal_residual);
            }
            thermodynamic_guess = trial;
            if (result.maximum_occupancy_error <= d->options.volume_relative_tolerance &&
                flux_change <= d->options.volume_relative_tolerance &&
                result.maximum_continuity_residual <=
                    d->options.continuity_tolerance * std::max(1.0, projected.continuity_scale) &&
                result.maximum_thermal_residual <= 1 && result.maximum_conditioned_thermal_residual <= 1 &&
                (!provider || (iteration >= 1 && result.maximum_source_residual <= 1 &&
                                  result.maximum_transport_residual <= 1 && result.maximum_thermal_residual <= 1 &&
                                  result.maximum_event_residual <= 1 && result.maximum_shape_source_residual <= 1)))
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
            {
                std::ostringstream message;
                message << "Low-Mach source corrector exhausted " << maximum_source_evaluations
                        << " evaluations. Residuals relative to acceptance: occupancy="
                        << result.maximum_occupancy_error / d->options.volume_relative_tolerance
                        << ", flux=" << flux_change / d->options.volume_relative_tolerance << ", continuity="
                        << result.maximum_continuity_residual /
                               (d->options.continuity_tolerance * std::max(1.0, projected.continuity_scale))
                        << ", source=" << result.maximum_source_residual
                        << ", transport=" << result.maximum_transport_residual
                        << ", thermal=" << result.maximum_thermal_residual
                        << ", conditioned=" << result.maximum_conditioned_thermal_residual
                        << ", gross=" << result.maximum_event_residual
                        << ", shape=" << result.maximum_shape_source_residual << ", Fourier operand uncertainty [J]="
                        << result.maximum_fourier_gross_comparison_uncertainty_j.value_or(0.);
                throw std::runtime_error(message.str());
            }
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
        std::vector<double> old_species, new_species, source_species;
        Vector old_momentum{}, new_momentum{}, external_momentum{}, momentum_operand_scale{};
        collective_detail::collective_local_validation(mesh, "Low-Mach physical caloric ledger",
            [&]
            {
                old_species.resize(ns);
                new_species.resize(ns);
                source_species.resize(ns);
                result.population_number_errors.assign(np, 0);
                result.species_errors.resize(ns);
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
                        const double source_energy = thermal_batch
                                                         ? last_thermal_evaluations[c].applied_phase_energy[a]
                                                         : (!sources.empty() && !sources[c].internal_energy.empty()
                                                                   ? sources[c].internal_energy[a]
                                                                   : 0);
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
                        external_energy += source_energy;
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
        result.internal_energy_roundoff_bound = (64 + 2 * ranks) * std::numeric_limits<double>::epsilon() * operands +
                                                result.conditioned_raw_energy_roundoff_bound;
        const double physical_budget = d->options.conservation_relative_tolerance * result.caloric_energy_scale;
        check(std::isfinite(physical_budget) && physical_budget > 0 &&
                  std::isfinite(result.internal_energy_roundoff_bound),
            "Low-Mach caloric energy scales are invalid.");
        check(result.internal_energy_roundoff_bound <= 0.25 * physical_budget,
            "Low-Mach energy reference is too poorly conditioned for the requested caloric accuracy.");
        check(std::abs(result.internal_energy_error) <= physical_budget + result.internal_energy_roundoff_bound,
            "Low-Mach physical internal-energy/work budget failed.");

        for (size_t a = 0; a < qbmm_families.size(); ++a)
            if (qbmm_families[a])
            {
                std::array<long double, 4> local{}, global{};
                collective_detail::collective_local_validation(mesh, "QBMM final physical shape-budget operands",
                    [&]
                    {
                        std::array<detail::CompensatedSum<long double>, 2> defect, magnitude;
                        for (size_t c = 0; c < nc; ++c)
                        {
                            const auto old_shape = *accepted.cells[c].qbmm_shape[a];
                            const auto new_shape = *trial.cells[c].qbmm_shape[a];
                            const auto source_shape = source_stage_detail::shape_packet(sources, c, a);
                            const std::array<double, 2> old{old_shape.H2, old_shape.H3},
                                now{new_shape.H2, new_shape.H3}, delta{source_shape.H2, source_shape.H3};
                            for (size_t k = 0; k < 2; ++k)
                            {
                                detail::CompensatedSum<long double> cell_defect;
                                cell_defect += now[k];
                                cell_defect += -static_cast<long double>(old[k]);
                                cell_defect += -static_cast<long double>(delta[k]);
                                defect[k] += cell_defect.value();
                                magnitude[k] += std::abs(static_cast<long double>(now[k]));
                                magnitude[k] += std::abs(static_cast<long double>(old[k]));
                                magnitude[k] += std::abs(static_cast<long double>(delta[k]));
                            }
                        }
                        for (size_t k = 0; k < 2; ++k)
                        {
                            local[k] = defect[k].value();
                            local[2 + k] = magnitude[k].value();
                            check(std::isfinite(local[k]) && std::isfinite(local[2 + k]),
                                "QBMM final physical shape-budget overflow.");
                        }
                    });
                transport_consistency_detail::sum_qbmm_budget(*mesh.owned_cell_map()->getComm(), local, global);
                for (size_t k = 0; k < 2; ++k)
                    check(source_stage_detail::normalized(
                              global[k], global[2 + k], d->options.conservation_relative_tolerance) <= 1,
                        "Low-Mach physical QBMM shape/source budget failed.");
            }
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
        collective_detail::collective_local_validation(mesh, "Low-Mach final trial snapshot",
            [&]
            {
                // Final physical endpoint only: no pressure-root sampling and no replay
                // of the source/force packet. A rank-local gate failure reaches all owners.
                for (size_t c = 0; c < trial.cells.size(); ++c)
                {
                    qbmm_native_law_detail::validate_drift(trial.cells[c]);
                    qbmm_native_law_detail::validate_transport_gates(d->options, trial.cells[c],
                        ale ? ale->new_cell_volumes()[c] : mesh.cell_volume(static_cast<LO>(c)), dt);
                }
                staged.face_flux.resize(mesh.num_owned_faces());
                for (size_t f = 0; f < staged.face_flux.size(); ++f)
                    staged.face_flux[f] = d->trial_flux.local_value(static_cast<LO>(f));
                if (!qbmm_families.empty())
                {
                    qbmm_detail::validate_stamps(d->options, trial);
                    staged.qbmm_validation.reserve(nc);
                    for (const auto& cell : trial.cells)
                        staged.qbmm_validation.push_back(qbmm_detail::validate_cell(d->options, cell));
                }
            });
        staged.candidate = std::move(trial);
        staged.preview_id = preview_id;
        staged.statistics = std::move(result);
        staged.transport = std::move(last_transport);
        staged.sources = std::move(sources);
        staged.gross_events = std::move(gross_events);
        staged.qbmm_cells = std::move(qbmm_cells);
        staged.provider = std::move(provider);
        staged.thermal_receipt = std::move(thermal_batch);
        staged.thermal_evaluations = std::move(last_thermal_evaluations);
        staged.owner = this;
        staged.nonce = d->trial_nonce;
        staged.base_revision = accepted.revision;
        staged.geometry_epoch = d->epoch;
        staged.logical_geometry = d->logical_geometry;
        staged.phase_transport = std::move(last_phase_transport);
        if (!qbmm_families.empty() && !coherent_qbmm)
            finalize_qbmm_balances(staged);
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
    return acquire_controller_for({});
}
LowMachMixtureSolver::ControllerLease LowMachMixtureSolver::acquire_controller_for(std::string_view signature,
    bool legacy_compatible, std::optional<Vector> body_acceleration, const PlanarMotion* motion)
{
    std::optional<std::uint64_t> binding;
    if (!d->options.qbmm_families.empty() && !signature.empty())
    {
        RestartHash hash;
        hash.string(signature);
        binding = hash.value;
    }
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach acquire controller",
        [&]
        {
            check(!motion || (motion->mesh_ptr().get() == d->mesh.get() && !body_acceleration),
                "Planar controller requires its exact mutable solver mesh and zero supplied body force.");
            check(signature.empty() || !d->accepted_planar_motion ||
                      (motion && d->accepted_planar_motion == motion->d_authority),
                "An accepted moving QBMM state requires its live planar controller authority.");
            check(!d->needs_restart_controller || binding == d->accepted_qbmm_controller,
                "QBMM restart requires matching reaction, population and transfer provider metadata.");
            check(!d->qbmm_legacy_default_only || legacy_compatible,
                "Legacy QBMM restart requires default force and constitutive-law configuration.");
            check(!d->controller_active && d->controller_id != std::numeric_limits<std::uint64_t>::max() &&
                      d->trial_nonce != std::numeric_limits<std::uint64_t>::max(),
                "Low-Mach solver already has an external controller or exhausted its lease counter.");
        });
    collective_detail::require_uniform_value(*d->mesh, int(binding.has_value()), "QBMM controller binding presence");
    collective_detail::require_uniform_value(*d->mesh, binding.value_or(0), "QBMM controller binding signature");
    collective_detail::require_uniform_value(*d->mesh, int(legacy_compatible), "QBMM controller legacy compatibility");
    collective_detail::require_uniform_value(
        *d->mesh, int(d->qbmm_legacy_default_only), "QBMM legacy restart configuration restriction");
    d->active_qbmm_controller = binding;
    d->active_controller_motion = binding && motion ? motion->d_authority : PlanarMotion::Authority{};
    d->active_body_acceleration = acceleration_components(body_acceleration);
    d->active_legacy_compatible = legacy_compatible;
    d->controller_active = true;
    ++d->controller_id;
    ++d->trial_nonce;
    // Public lease acquisition passes an empty signature. The reacting
    // controller supplies its configuration signature even for count-only flow.
    return ControllerLease(this, d->controller_id, !signature.empty());
}
void LowMachMixtureSolver::release_controller(std::uint64_t id) noexcept
{
    if (d->controller_active && id == d->controller_id)
    {
        d->controller_active = false;
        d->active_qbmm_controller.reset();
        d->active_controller_motion.reset();
        d->active_body_acceleration.reset();
        if (d->trial_nonce != std::numeric_limits<std::uint64_t>::max())
            ++d->trial_nonce;
    }
}
LowMachMixtureSolver::ControllerLease::ControllerLease(ControllerLease&& other) noexcept
    : owner(std::exchange(other.owner, nullptr)), id(other.id),
      reacting_source_authority(other.reacting_source_authority)
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
        reacting_source_authority = other.reacting_source_authority;
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
    require_uniform_advancement(*d->mesh, AdvancementOperation::Commit);
    require_controller(controller);
    State candidate;
    Diagnostics statistics;
    std::shared_ptr<const QbmmPublication> publication;
    collective_detail::collective_local_validation(*d->mesh, "Low-Mach immutable trial commit",
        [&]
        {
            check(trial.owner == this && !trial.ale_geometry && trial.nonce == d->trial_nonce &&
                      trial.base_revision == d->accepted.revision && trial.logical_geometry == d->logical_geometry &&
                      d->epoch == mesh_geometry_epoch(*d->mesh),
                "Low-Mach trial is foreign, already committed, superseded or belongs to stale geometry.");
            qbmm_detail::validate_state(d->options, trial.candidate);
            candidate = trial.candidate;
            statistics = trial.statistics;
            publication = prepare_qbmm_publication(trial);
        });
    if (!d->options.qbmm_families.empty())
    {
        collective_detail::require_uniform_value(*d->mesh, trial.preview_id, "QBMM collective trial selection");
    }
    d->publish_fixed(
        std::move(candidate), std::move(statistics), trial.face_flux, d->trial_nonce + 1, std::move(publication));
    if (d->options.qbmm_families.empty())
    {
        d->accepted_count_only_body_acceleration =
            trial.provider ? acceleration_components(trial.provider->body_acceleration) : std::nullopt;
        d->count_only_force_provenance_recorded = true;
    }
}
LowMachMixtureSolver::Diagnostics LowMachMixtureSolver::advance(double dt, const std::vector<CellSources>& sources)
{
    const auto staged = preview(dt, sources);
    commit(staged);
    return d->diagnostics;
}
} // namespace SimpleFluid::Reacting
