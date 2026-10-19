/** @file ReactingVoFCoupler.cc
 * @brief Conservative reactions, interface exchange and accepted-only gas ledgers.
 */
#include "solvers/ReactingVoFCoupler.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
namespace
{
using LO = DefaultTpetraTypes::local_ordinal_type;
using Vector = VoF::Vector;
using State = VoF::LowMachState;
using Sources = VoF::LowMachSources;
namespace Physics = BubbleGasPhysics;

void require_all(bool valid, const char* message)
{
    int value = valid ? 1 : 0, minimum = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &value, &minimum);
    if (!minimum)
        throw std::runtime_error(message);
}
template<class Work> void collective(Work&& work, const char* message)
{
    std::exception_ptr error;
    try
    {
        work();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = error ? 1 : 0, any = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &any);
    if (any)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error(message);
    }
}
double reduce(double value, Teuchos::EReductionType op)
{
    double result = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), op, 1, &value, &result);
    return result;
}
double total(const std::vector<double>& values)
{
    detail::CompensatedSum<double> sum;
    for (double x : values)
        sum += x;
    return reduce(sum.value(), Teuchos::REDUCE_SUM);
}
void identical(double x, const char* message)
{
    require_all(std::isfinite(x), message);
    require_all(reduce(x, Teuchos::REDUCE_MIN) == reduce(x, Teuchos::REDUCE_MAX), message);
}
void identical(const std::string& value)
{
    require_all(value.size() < 1000000, "Reacting VoF metadata is too long.");
    const auto comm = Tpetra::getDefaultComm();
    int size = comm->getRank() == 0 ? static_cast<int>(value.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &size);
    std::string reference = comm->getRank() == 0 ? value : std::string(size, '\0');
    if (size)
        Teuchos::broadcast(*comm, 0, size, reference.data());
    require_all(reference == value, "Reacting VoF model metadata differ across ranks.");
}
double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector vector(const MeshUtils::Vec3& x)
{
    return {x.x, x.y, x.z};
}
double gas_enthalpy(const VoF::IdealGasSpecies& gas, double T, double R)
{
    return gas.reference_internal_energy + gas.molar_heat_capacity_v * (T - gas.reference_temperature) + R * T;
}
double surface_tension(const ReactingGasOptions& options, double temperature)
{
    // The hybrid equations also support the explicitly selected zero-capillarity
    // limit. Leave the legacy correlation's positive-sigma contract intact.
    if (options.surface_tension_mode == SurfaceTensionMode::Constant && options.surface_tension == 0)
        return 0;
    return Physics::surface_tension(options, temperature);
}
double hydrogen_total(const State& state)
{
    detail::CompensatedSum<double> sum;
    for (std::size_t c = 0; c < state.dissolved_hydrogen_moles.size(); ++c)
    {
        sum += state.dissolved_hydrogen_moles[c];
        sum += state.resolved_hydrogen_moles[c];
        for (const auto& population : state.dispersed)
            sum += population.hydrogen_moles[c];
    }
    return reduce(sum.value(), Teuchos::REDUCE_SUM);
}
double relative_change(double a, double b)
{
    return std::abs(a - b) / std::max({std::abs(a), std::abs(b), std::numeric_limits<double>::min()});
}

struct Properties
{
    std::vector<double> volume, liquid_volume, carrier_volume, gas_volume, temperature, pressure, gas_pressure;
    std::vector<double> density, viscosity, diffusivity, liquid_enthalpy, hydrogen_enthalpy, inert_enthalpy;
};

/** Shared event amounts survive rounding when added to stored inventories. */
struct ReactionLedger
{
    using Sum = detail::CompensatedSum<double>;
    std::vector<Sum> resolved, dissolved, liquid_moles;
    std::vector<std::vector<Sum>> number, moles;
    ReactionLedger(std::size_t cells, std::size_t populations)
        : resolved(cells), dissolved(cells), liquid_moles(cells), number(populations, std::vector<Sum>(cells)),
          moles(populations, std::vector<Sum>(cells))
    {
    }
};
} // namespace

struct ReactingVoFCoupler::Impl
{
    SP<AdaptiveVoFSolver> flow;
    ReactingVoFOptions options;
    std::shared_ptr<VoF::AdaptiveVoF> geometry;
    ReactingVoFStatistics accepted;
    std::optional<std::pair<Checkpoint, ReactingVoFStatistics>> saved;
    std::uint64_t revision = 0;
    std::size_t micro = 0;
    std::vector<std::size_t> upper;

    struct Candidate
    {
        const Impl* owner;
        std::uint64_t revision;
        AdaptiveVoFSolver::LowMachTrial fluid;
        ReactingVoFStatistics stats;
    };

    Impl(SP<AdaptiveVoFSolver> solver, ReactingVoFOptions configuration)
        : flow(std::move(solver)), options(std::move(configuration))
    {
        require_all(bool(flow), "Reacting VoF coupler requires shared solver ownership.");
        require_all(flow->low_mach_enabled(), "Reacting VoF coupler requires initialized low-Mach state.");
        validate_options();
        refresh_geometry();
        const auto& state = flow->low_mach_state();
        const auto find_population = [&](const std::string& name)
        {
            const auto it = std::find_if(
                state.dispersed.begin(), state.dispersed.end(), [&](const auto& p) { return p.name == name; });
            if (it == state.dispersed.end())
                throw std::invalid_argument("Missing reacting population: " + name);
            return static_cast<std::size_t>(it - state.dispersed.begin());
        };
        collective(
            [&]
            {
                const bool needs_micro =
                    options.radiolysis.mode != ReactingGasMode::Disabled ||
                    (!options.upper_populations.empty() &&
                        options.population_step.underflow == Physics::BubbleUnderflowPolicy::ReturnToMicro);
                if (needs_micro)
                    micro = find_population(options.micro_population);
                for (const auto& name : options.upper_populations)
                    upper.push_back(find_population(name));
                if (upper.size() != options.upper_pivots.mol_per_bubble.size())
                    throw std::invalid_argument("Upper population names and molar pivots must match.");
                auto sorted = upper;
                std::sort(sorted.begin(), sorted.end());
                if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() ||
                    (needs_micro && std::find(upper.begin(), upper.end(), micro) != upper.end()))
                    throw std::invalid_argument("Micro and upper population identities must be distinct.");
            },
            "Reacting population registration failed on another rank.");
        accepted.time = flow->time();
    }

    void validate_options()
    {
        const auto& r = options.radiolysis;
        for (double value : {double(r.mode), r.hydrogen_yield_mol_per_j, r.gas_release_efficiency, r.henry_coefficient,
                 r.microbubble_lifetime, r.micro_to_large_conversion_coefficient, double(r.max_subcycles),
                 double(r.heaviside_mode), r.smooth_heaviside_width, double(r.rise_velocity_mode),
                 r.constant_slip_velocity, double(r.surface_tension_mode), r.surface_tension,
                 double(r.diffusivity_mode), r.hydrogen_diffusivity, double(r.nucleation_radius_mode),
                 r.nucleation_radius, r.gas_constant, r.min_radius, r.max_radius, double(r.max_radius_iterations),
                 r.rise_velocity_tolerance, double(r.max_rise_velocity_iterations), double(r.kinetics_mode),
                 double(options.population_step.underflow), double(options.maximum_correctors),
                 options.relative_tolerance, options.inventory_relative_tolerance,
                 options.dissolved_diffusion ? 1.0 : 0.0})
            identical(value, "Reacting VoF options must be finite and identical across ranks.");
        identical(options.micro_population);
        identical(double(options.upper_populations.size()), "Upper population counts differ across ranks.");
        for (const auto& name : options.upper_populations)
            identical(name);
        identical(double(options.upper_pivots.mol_per_bubble.size()), "Molar grids differ across ranks.");
        for (double x : options.upper_pivots.mol_per_bubble)
            identical(x, "Molar grids differ across ranks.");
        identical(options.upper_kernels.model_name);
        identical(options.upper_kernels.validity_description);
        for (bool flag : {bool(options.upper_kernels.validate), bool(options.upper_kernels.growth_rate),
                 bool(options.upper_kernels.coalescence_kernel), bool(options.upper_kernels.breakup_frequency),
                 bool(options.upper_kernels.breakup_daughters), options.resolved_transfer.enabled,
                 options.capture.enabled, options.headspace.enabled, bool(options.resolved_transfer.thickness),
                 bool(options.resolved_transfer.validate), bool(options.capture.efficiency),
                 bool(options.capture.validate), bool(options.headspace.molar_rate), bool(options.headspace.validate),
                 bool(r.surface_tension_correlation), bool(r.nucleation_radius_correlation),
                 bool(r.hydrogen_diffusivity_correlation)})
            identical(flag ? 1 : 0, "Reacting VoF model selections differ across ranks.");
        for (const auto* text : {&options.resolved_transfer.model_name, &options.resolved_transfer.validity_description,
                 &options.capture.model_name, &options.capture.validity_description, &options.headspace.model_name,
                 &options.headspace.validity_description})
            identical(*text);
        identical(double(options.headspace.regions.size()), "Headspace regions differ across ranks.");
        for (const auto& b : options.headspace.regions)
            for (std::size_t a = 0; a < 3; ++a)
            {
                identical(b.lower[a], "Headspace bounds differ across ranks.");
                identical(b.upper[a], "Headspace bounds differ across ranks.");
                require_all(b.upper[a] >= b.lower[a], "Headspace bounds must be ordered.");
            }
        require_all((r.mode == ReactingGasMode::Disabled || r.mode == ReactingGasMode::Sheng2024TwoPopulation) &&
                        r.hydrogen_yield_mol_per_j >= 0 && r.gas_release_efficiency >= 0 &&
                        r.gas_release_efficiency <= 1 && options.maximum_correctors >= 2 &&
                        options.relative_tolerance > 0 && options.relative_tolerance < 1 &&
                        options.inventory_relative_tolerance > 0 && options.inventory_relative_tolerance < 1,
            "Invalid reacting VoF model or corrector controls.");
        require_all(r.min_radius > 0 && r.max_radius > r.min_radius && r.max_radius_iterations > 0 &&
                        r.rise_velocity_tolerance > 0 && r.max_rise_velocity_iterations > 0 &&
                        r.constant_slip_velocity >= 0 && r.gas_constant == flow->low_mach_options().gas_constant &&
                        (r.rise_velocity_mode == BubbleRiseVelocityMode::ZeroSlip ||
                            r.rise_velocity_mode == BubbleRiseVelocityMode::ConstantSlip ||
                            r.rise_velocity_mode == BubbleRiseVelocityMode::Celata2007),
            "Hybrid bubble geometry/slip controls must be valid and use the flow gas constant.");
        require_all(options.population_step.underflow == Physics::BubbleUnderflowPolicy::Reject ||
                        options.population_step.underflow == Physics::BubbleUnderflowPolicy::ReturnToMicro,
            "Invalid upper-population underflow policy.");
        for (std::size_t i = 0; i < options.upper_pivots.mol_per_bubble.size(); ++i)
            require_all(
                options.upper_pivots.mol_per_bubble[i] > 0 &&
                    (i == 0 || options.upper_pivots.mol_per_bubble[i] > options.upper_pivots.mol_per_bubble[i - 1]),
                "Upper molar pivots must be positive and strictly increasing.");
        const bool upper_rates =
            bool(options.upper_kernels.growth_rate) || bool(options.upper_kernels.coalescence_kernel) ||
            bool(options.upper_kernels.breakup_frequency) || bool(options.upper_kernels.breakup_daughters);
        require_all(!upper_rates || (!options.upper_populations.empty() && bool(options.upper_kernels.validate) &&
                                        !options.upper_kernels.model_name.empty() &&
                                        !options.upper_kernels.validity_description.empty()),
            "Enabled population rates require populated bins and a named validated supplied model.");
        if (r.mode != ReactingGasMode::Disabled)
            require_all(r.microbubble_lifetime > 0 && r.max_subcycles > 0 && r.henry_coefficient > 0 &&
                            r.micro_to_large_conversion_coefficient >= 0 &&
                            (r.micro_to_large_conversion_coefficient == 0 || !options.upper_populations.empty()) &&
                            r.kinetics_mode == ReactingKineticsMode::LegacySubcycled,
                "Active microbubble kinetics requires positive physical coefficients and a conversion destination.");
        if (options.resolved_transfer.enabled)
            require_all(!options.resolved_transfer.model_name.empty() &&
                            !options.resolved_transfer.validity_description.empty() &&
                            bool(options.resolved_transfer.thickness) && bool(options.resolved_transfer.validate) &&
                            r.henry_coefficient > 0,
                "Resolved transfer requires a named, validated supplied film and positive Henry coefficient.");
        if (options.capture.enabled)
            require_all(!options.capture.model_name.empty() && !options.capture.validity_description.empty() &&
                            bool(options.capture.efficiency) && bool(options.capture.validate),
                "Capture requires a named supplied efficiency and physical-domain validator.");
        if (options.headspace.enabled)
            require_all(!options.headspace.regions.empty() && !options.headspace.model_name.empty() &&
                            !options.headspace.validity_description.empty() && bool(options.headspace.molar_rate) &&
                            bool(options.headspace.validate),
                "Venting requires explicit headspace ownership and a named validated outflow law.");
        if (options.dissolved_diffusion || options.resolved_transfer.enabled || options.upper_kernels.growth_rate)
            require_all(r.diffusivity_mode == HydrogenDiffusivityMode::External ||
                            (std::isfinite(r.hydrogen_diffusivity) && r.hydrogen_diffusivity > 0),
                "Dissolved transport or upper growth requires supplied positive molecular diffusivity.");
    }

    void refresh_geometry()
    {
        if (geometry && geometry->hierarchy().get() == flow->hierarchy().get())
            return;
        auto next = std::make_shared<VoF::AdaptiveVoF>(flow->hierarchy());
        require_all(next->mesh()->owned_cell_map()->isSameAs(*flow->mesh()->owned_cell_map()),
            "Coupler and flow native cell ownership differ.");
        geometry.swap(next);
    }

    Properties properties(const State& state) const
    {
        const auto n = flow->mesh()->num_owned_cells();
        Properties p;
        for (auto* values : {&p.volume, &p.liquid_volume, &p.carrier_volume, &p.gas_volume, &p.temperature, &p.pressure,
                 &p.gas_pressure, &p.density, &p.viscosity, &p.diffusivity, &p.liquid_enthalpy, &p.hydrogen_enthalpy,
                 &p.inert_enthalpy})
            values->resize(n);
        collective(
            [&]
            {
                const auto execution = flow->mesh()->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    p.volume[c] = flow->mesh()->cell_volume(static_cast<LO>(c));
                    p.temperature[c] = state.temperature.at(c);
                    p.pressure[c] = state.liquid_absolute_pressure.at(c);
                    p.gas_pressure[c] = state.resolved_absolute_pressure.at(c);
                    const auto liquid = flow->low_mach_options().liquid->evaluate(p.temperature[c], p.pressure[c]);
                    p.density[c] = liquid.density;
                    p.viscosity[c] = liquid.dynamic_viscosity;
                    p.liquid_enthalpy[c] = liquid.enthalpy;
                    p.liquid_volume[c] = state.liquid_fraction.at(c) * p.volume[c];
                    p.carrier_volume[c] = (state.liquid_fraction.at(c) + state.dispersed_fraction.at(c)) * p.volume[c];
                    p.gas_volume[c] = state.resolved_fraction.at(c) * p.volume[c];
                    const auto& r = options.radiolysis;
                    if (r.diffusivity_mode == HydrogenDiffusivityMode::External || r.hydrogen_diffusivity > 0)
                        p.diffusivity[c] = Physics::hydrogen_diffusivity(r, p.temperature[c]);
                    const double sigma = surface_tension(r, p.temperature[c]);
                    if (std::abs(sigma - flow->material().surface_tension) > 1e-12 * std::max(1.0, std::abs(sigma)))
                        throw std::invalid_argument("Hybrid resolved capillarity requires the configured constant "
                                                    "sigma; Marangoni stresses are not enabled.");
                    const auto& thermo = flow->low_mach_options();
                    p.hydrogen_enthalpy[c] = gas_enthalpy(thermo.hydrogen, p.temperature[c], thermo.gas_constant);
                    p.inert_enthalpy[c] = gas_enthalpy(thermo.inert, p.temperature[c], thermo.gas_constant);
                }
            },
            "Reacting material evaluation failed on another rank.");
        return p;
    }

    std::vector<Vector> gradients(
        const std::vector<double>& owned, const std::vector<double>& carrier, const std::vector<double>& aperture) const
    {
        const auto& mesh = *geometry->mesh();
        const auto field = geometry->exchange_cells(owned), host = geometry->exchange_cells(carrier);
        std::vector<Vector> result(owned.size());
        collective(
            [&]
            {
                const auto execution = mesh.acquire_execution_view();
                for (std::size_t c = 0; c < owned.size(); ++c)
                {
                    double A[3][4]{};
                    double lo = owned[c], hi = owned[c];
                    mesh.visit_cell_faces(static_cast<LO>(c),
                        [&](LO face)
                        {
                            if (mesh.is_geometry_exterior_face(face) || aperture[face] == 0)
                                return;
                            const auto other = mesh.opposite_cell(face, static_cast<LO>(c));
                            if (host[other] <= 0)
                                return;
                            const auto d = vector(mesh.cell_center_vector(face, static_cast<LO>(c)));
                            const double weight = 1.0 / dot(d, d), difference = field[other] - owned[c];
                            lo = std::min(lo, field[other]);
                            hi = std::max(hi, field[other]);
                            for (std::size_t a = 0; a < 3; ++a)
                            {
                                A[a][3] += weight * d[a] * difference;
                                for (std::size_t b = 0; b < 3; ++b)
                                    A[a][b] += weight * d[a] * d[b];
                            }
                        });
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        std::size_t pivot = a;
                        for (std::size_t b = a + 1; b < 3; ++b)
                            if (std::abs(A[b][a]) > std::abs(A[pivot][a]))
                                pivot = b;
                        if (std::abs(A[pivot][a]) < 1e-13)
                            continue;
                        for (std::size_t b = 0; b < 4; ++b)
                            std::swap(A[a][b], A[pivot][b]);
                        const double diagonal = A[a][a];
                        for (std::size_t b = a; b < 4; ++b)
                            A[a][b] /= diagonal;
                        for (std::size_t row = 0; row < 3; ++row)
                            if (row != a)
                            {
                                const double factor = A[row][a];
                                for (std::size_t b = a; b < 4; ++b)
                                    A[row][b] -= factor * A[a][b];
                            }
                    }
                    for (std::size_t a = 0; a < 3; ++a)
                        result[c][a] = std::abs(A[a][a]) > 1e-13 ? A[a][3] : 0.0;
                    double limiter = 1.0;
                    mesh.visit_cell_faces(static_cast<LO>(c),
                        [&](LO face)
                        {
                            if (mesh.is_geometry_exterior_face(face) || aperture[face] == 0)
                                return;
                            const double increment =
                                dot(result[c], vector(mesh.face_center_vector(face, static_cast<LO>(c))));
                            if (increment > 0)
                                limiter = std::min(limiter, (hi - owned[c]) / increment);
                            else if (increment < 0)
                                limiter = std::min(limiter, (lo - owned[c]) / increment);
                        });
                    for (double& x : result[c])
                        x *= limiter;
                }
            },
            "Dissolved-gradient reconstruction failed on another rank.");
        std::vector<Vector> halo(mesh.num_local_cells());
        for (std::size_t a = 0; a < 3; ++a)
        {
            std::vector<double> x(owned.size());
            for (std::size_t c = 0; c < x.size(); ++c)
                x[c] = result[c][a];
            const auto values = geometry->exchange_cells(x);
            for (std::size_t c = 0; c < values.size(); ++c)
                halo[c][a] = values[c];
        }
        return halo;
    }

    /** Dilute counterdiffusion: H2 moves within liquid with zero mixture-mass flux. */
    std::vector<double> diffuse(State& working, const Properties& p, const State& guess, double dt) const
    {
        std::vector<double> change(p.volume.size(), 0.0);
        if (!options.dissolved_diffusion)
            return change;
        std::vector<double> C(p.volume.size()), aqueous(p.volume.size());
        bool valid = true;
        for (std::size_t c = 0; c < C.size(); ++c)
        {
            if (p.liquid_volume[c] > 0)
                C[c] = working.dissolved_hydrogen_moles[c] / p.liquid_volume[c];
            else if (working.dissolved_hydrogen_moles[c] != 0)
                valid = false;
            if (p.carrier_volume[c] > 0)
                aqueous[c] = p.liquid_volume[c] / p.carrier_volume[c];
        }
        require_all(valid, "A dissolved inventory has no true aqueous carrier.");
        const auto aperture = geometry->carrier_face_aperture(guess.resolved_fraction);
        const auto concentration = geometry->exchange_cells(C);
        const auto fraction = geometry->exchange_cells(aqueous);
        const auto diffusivity = geometry->exchange_cells(p.diffusivity);
        const auto grad = gradients(C, p.liquid_volume, aperture);
        const auto& mesh = *geometry->mesh();
        std::vector<double> flux(mesh.num_faces(), 0.0);
        collective(
            [&]
            {
                const auto execution = mesh.acquire_execution_view();
                for (std::size_t id = 0; id < mesh.num_owned_faces(); ++id)
                {
                    const auto face = static_cast<LO>(id);
                    if (mesh.is_geometry_exterior_face(face) || aperture[id] == 0)
                        continue;
                    const auto l = mesh.owner_cell(face), r = mesh.neighbor_cell(face);
                    if (fraction[l] == 0 || fraction[r] == 0)
                        continue;
                    const auto normal = vector(mesh.face_normal(face));
                    auto dl = vector(mesh.face_center_vector(face, l)), dr = vector(mesh.face_center_vector(face, r));
                    const double nl = dot(dl, normal), nr = dot(dr, normal);
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        dl[a] -= nl * normal[a];
                        dr[a] -= nr * normal[a];
                    }
                    // The aqueous diffusion coefficients are in series across
                    // the actual one-sided distances. This also retains the
                    // unequal coarse/fine widths and discontinuous D or alpha.
                    const double resistance =
                        std::abs(nl) / (fraction[l] * diffusivity[l]) + std::abs(nr) / (fraction[r] * diffusivity[r]);
                    flux[id] = dt * aperture[id] * mesh.face_area(face) *
                               (concentration[l] + dot(grad[l], dl) - concentration[r] - dot(grad[r], dr)) / resistance;
                }
            },
            "Dissolved diffusive flux construction failed on another rank.");
        flux = geometry->exchange_faces(flux);
        collective(
            [&]
            {
                const auto execution = mesh.acquire_execution_view();
                for (std::size_t c = 0; c < change.size(); ++c)
                {
                    double outgoing = 0;
                    mesh.visit_cell_faces(static_cast<LO>(c),
                        [&](LO f)
                        {
                            const double out = (mesh.owner_cell(f) == static_cast<LO>(c) ? 1 : -1) * flux[f];
                            change[c] -= out;
                            outgoing += std::max(out, 0.0);
                        });
                    valid = valid && outgoing <= working.dissolved_hydrogen_moles[c];
                }
            },
            "Dissolved diffusive donor accounting failed on another rank.");
        require_all(valid, "Dissolved diffusion exhausted its donor; reduce the physical timestep.");
        for (std::size_t c = 0; c < change.size(); ++c)
            working.dissolved_hydrogen_moles[c] += change[c];
        return change;
    }

    /** Apply one uniquely owned patch transfer to both physical reservoirs. */
    void add_exchange(const std::vector<VoF::InterfacePatch>& patches, const std::vector<double>& enthalpy,
        const std::vector<Vector>& momentum, Sources& source) const
    {
        const auto eg = geometry->scatter_patch_amounts(patches, enthalpy, VoF::InterfaceSide::Gas);
        const auto el = geometry->scatter_patch_amounts(patches, enthalpy, VoF::InterfaceSide::Carrier);
        for (std::size_t c = 0; c < eg.size(); ++c)
        {
            source.enthalpy_transfer[c] += eg[c] - el[c];
        }
        for (std::size_t a = 0; a < 3; ++a)
        {
            std::vector<double> values(momentum.size());
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = momentum[i][a];
            const auto mg = geometry->scatter_patch_amounts(patches, values, VoF::InterfaceSide::Gas);
            const auto ml = geometry->scatter_patch_amounts(patches, values, VoF::InterfaceSide::Carrier);
            for (std::size_t c = 0; c < eg.size(); ++c)
                source.momentum_transfer[c][a] += mg[c] - ml[c];
        }
    }

    std::vector<Vector> sample_vectors(const std::vector<VoF::InterfacePatch>& patches,
        const std::vector<Vector>& values, VoF::InterfaceSide side) const
    {
        std::vector<Vector> result(patches.size());
        for (std::size_t a = 0; a < 3; ++a)
        {
            std::vector<double> scalar(values.size());
            for (std::size_t c = 0; c < values.size(); ++c)
                scalar[c] = values[c][a];
            const auto sampled = geometry->sample_patch_values(patches, scalar, side);
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i][a] = sampled[i];
        }
        return result;
    }

    std::vector<double> interface_transfer(State& working, const Properties& p,
        const std::vector<VoF::InterfacePatch>& patches, const std::vector<Vector>& velocity, double dt,
        Sources& source, ReactionLedger& ledger, ReactingVoFStatistics& stats) const
    {
        std::vector<double> amount(patches.size(), 0.0);
        if (!options.resolved_transfer.enabled)
            return amount;
        const auto T = geometry->sample_patch_values(patches, p.temperature, VoF::InterfaceSide::Carrier);
        const auto pl = geometry->sample_patch_values(patches, p.pressure, VoF::InterfaceSide::Carrier);
        const auto pg = geometry->sample_patch_values(patches, p.gas_pressure, VoF::InterfaceSide::Gas);
        const auto D = geometry->sample_patch_values(patches, p.diffusivity, VoF::InterfaceSide::Carrier);
        const auto vl = geometry->sample_patch_values(patches, p.liquid_volume, VoF::InterfaceSide::Carrier);
        const auto I =
            geometry->sample_patch_values(patches, working.dissolved_hydrogen_moles, VoF::InterfaceSide::Carrier);
        const auto H = geometry->sample_patch_values(patches, working.resolved_hydrogen_moles, VoF::InterfaceSide::Gas);
        const auto inert =
            geometry->sample_patch_values(patches, working.resolved_inert_moles, VoF::InterfaceSide::Gas);
        const auto hl = geometry->sample_patch_values(patches, p.liquid_enthalpy, VoF::InterfaceSide::Carrier);
        const auto hg = geometry->sample_patch_values(patches, p.hydrogen_enthalpy, VoF::InterfaceSide::Gas);
        const auto ul = sample_vectors(patches, velocity, VoF::InterfaceSide::Carrier);
        const auto ug = sample_vectors(patches, velocity, VoF::InterfaceSide::Gas);
        std::vector<double> energy(patches.size()), out_l(patches.size()), out_g(patches.size());
        std::vector<Vector> moment(patches.size());
        const double W = flow->low_mach_options().hydrogen.molar_mass;
        collective(
            [&]
            {
                for (std::size_t i = 0; i < patches.size(); ++i)
                {
                    if (vl[i] <= 0 || H[i] + inert[i] <= 0)
                        continue;
                    const auto& patch = patches[i];
                    const double delta = options.resolved_transfer.thickness(patch.centroid, T[i], pl[i]);
                    if (!std::isfinite(delta) || delta <= 0)
                        throw std::domain_error("Film thickness must be supplied, finite and positive.");
                    options.resolved_transfer.validate(patch, delta, T[i], pl[i]);
                    const double length = geometry->film_length(patch, delta);
                    if (!(length > 0))
                        throw std::runtime_error("The interface has no finite aqueous sampling distance.");
                    const double concentration = I[i] / vl[i], partial = pg[i] * H[i] / (H[i] + inert[i]);
                    amount[i] = dt * patch.area * D[i] / length *
                                (concentration - options.radiolysis.henry_coefficient * partial);
                    if (!std::isfinite(amount[i]))
                        throw std::runtime_error("Interfacial hydrogen transfer overflowed.");
                    out_l[i] = std::max(amount[i], 0.0);
                    out_g[i] = std::max(-amount[i], 0.0);
                    energy[i] = amount[i] * (amount[i] >= 0 ? W * hl[i] : hg[i]);
                    for (std::size_t a = 0; a < 3; ++a)
                        moment[i][a] = W * amount[i] * (amount[i] >= 0 ? ul[i][a] : ug[i][a]);
                }
            },
            "Resolved film evaluation failed on another rank.");
        const auto required_l = geometry->scatter_patch_amounts(patches, out_l, VoF::InterfaceSide::Carrier);
        const auto required_g = geometry->scatter_patch_amounts(patches, out_g, VoF::InterfaceSide::Gas);
        bool valid = true;
        for (std::size_t c = 0; c < p.volume.size(); ++c)
            valid = valid && required_l[c] <= working.dissolved_hydrogen_moles[c] &&
                    required_g[c] <= working.resolved_hydrogen_moles[c];
        require_all(valid, "Resolved transfer exhausted a donor; reduce the physical reaction substep.");
        const auto dg = geometry->scatter_patch_amounts(patches, amount, VoF::InterfaceSide::Gas);
        const auto dl = geometry->scatter_patch_amounts(patches, amount, VoF::InterfaceSide::Carrier);
        for (std::size_t c = 0; c < p.volume.size(); ++c)
        {
            working.dissolved_hydrogen_moles[c] -= dl[c];
            working.resolved_hydrogen_moles[c] += dg[c];
            ledger.resolved[c] += dg[c];
            ledger.dissolved[c] += -dl[c];
            ledger.liquid_moles[c] += -dl[c];
        }
        add_exchange(patches, energy, moment, source);
        stats.dissolved_to_resolved += total(required_l);
        stats.resolved_to_dissolved += total(required_g);
        return amount;
    }

    void capture(State& working, const Properties& p, const std::vector<VoF::InterfacePatch>& patches,
        const std::vector<Vector>& velocity, const std::vector<double>& stefan_moles, double dt, Sources& source,
        ReactionLedger& ledger, ReactingVoFStatistics& stats) const
    {
        if (!options.capture.enabled)
            return;
        const auto carrier = geometry->sample_patch_values(patches, p.carrier_volume, VoF::InterfaceSide::Carrier);
        const auto rho = geometry->sample_patch_values(patches, p.density, VoF::InterfaceSide::Carrier);
        const auto T = geometry->sample_patch_values(patches, p.temperature, VoF::InterfaceSide::Carrier);
        const auto h = geometry->sample_patch_values(patches, p.hydrogen_enthalpy, VoF::InterfaceSide::Carrier);
        const auto u = sample_vectors(patches, velocity, VoF::InterfaceSide::Carrier);
        const double W = flow->low_mach_options().hydrogen.molar_mass;
        for (std::size_t k = 0; k < working.dispersed.size(); ++k)
        {
            auto& population = working.dispersed[k];
            const auto N = geometry->sample_patch_values(patches, population.number, VoF::InterfaceSide::Carrier);
            const auto M =
                geometry->sample_patch_values(patches, population.hydrogen_moles, VoF::InterfaceSide::Carrier);
            const auto radius = geometry->sample_patch_values(patches, population.radius, VoF::InterfaceSide::Carrier);
            const auto slip = sample_vectors(patches, population.drift_velocity, VoF::InterfaceSide::Carrier);
            std::vector<double> number(patches.size()), amount(patches.size()), energy(patches.size());
            std::vector<Vector> moment(patches.size());
            collective(
                [&]
                {
                    for (std::size_t i = 0; i < patches.size(); ++i)
                    {
                        if (carrier[i] <= 0 || N[i] <= 0 || M[i] <= 0)
                            continue;
                        const double approach = std::max(0.0,
                            W * stefan_moles[i] / (dt * patches[i].area * rho[i]) - dot(slip[i], patches[i].normal));
                        options.capture.validate(k, radius[i], approach, T[i]);
                        const double efficiency = options.capture.efficiency(k, radius[i], approach, T[i]);
                        if (!std::isfinite(efficiency) || efficiency < 0 || efficiency > 1)
                            throw std::domain_error("Capture efficiency must lie in [0,1].");
                        const double fraction = dt * patches[i].area * approach * efficiency / carrier[i];
                        number[i] = fraction * N[i];
                        amount[i] = fraction * M[i];
                        energy[i] = amount[i] * h[i];
                        for (std::size_t a = 0; a < 3; ++a)
                            moment[i][a] = amount[i] * W * (u[i][a] + slip[i][a]);
                    }
                },
                "Capture evaluation failed on another rank.");
            const auto dn = geometry->scatter_patch_amounts(patches, number, VoF::InterfaceSide::Carrier);
            const auto dm = geometry->scatter_patch_amounts(patches, amount, VoF::InterfaceSide::Carrier);
            const auto gas = geometry->scatter_patch_amounts(patches, amount, VoF::InterfaceSide::Gas);
            bool valid = true;
            for (std::size_t c = 0; c < p.volume.size(); ++c)
                valid = valid && dn[c] <= population.number[c] && dm[c] <= population.hydrogen_moles[c];
            require_all(valid, "Capture exhausted an incident population; reduce the physical timestep.");
            for (std::size_t c = 0; c < p.volume.size(); ++c)
            {
                population.number[c] -= dn[c];
                population.hydrogen_moles[c] -= dm[c];
                working.resolved_hydrogen_moles[c] += gas[c];
                ledger.number[k][c] += -dn[c];
                ledger.moles[k][c] += -dm[c];
                ledger.resolved[c] += gas[c];
                source.captured_hydrogen[c] += gas[c];
            }
            add_exchange(patches, energy, moment, source);
            stats.captured_number += total(dn);
            stats.hydrogen_captured += total(dm);
        }
    }

    void vent(const State& working, const Properties& p, const State& guess, const std::vector<Vector>& velocity,
        double dt, Sources& source, ReactingVoFStatistics& stats) const
    {
        if (!options.headspace.enabled)
            return;
        std::vector<bool> eligible(p.volume.size(), false);
        detail::CompensatedSum<double> n, h2, pn, tn;
        collective(
            [&]
            {
                const auto execution = flow->mesh()->acquire_execution_view();
                for (std::size_t c = 0; c < p.volume.size(); ++c)
                {
                    if (guess.resolved_fraction[c] < 1.0 - 2e-12)
                        continue;
                    const auto x = vector(flow->mesh()->cell_centroid(static_cast<LO>(c)));
                    for (const auto& box : options.headspace.regions)
                    {
                        bool inside = true;
                        for (std::size_t a = 0; a < 3; ++a)
                            inside = inside && x[a] >= box.lower[a] && x[a] <= box.upper[a];
                        eligible[c] = eligible[c] || inside;
                    }
                    if (!eligible[c])
                        continue;
                    const double moles = working.resolved_hydrogen_moles[c] + working.resolved_inert_moles[c];
                    n += moles;
                    h2 += working.resolved_hydrogen_moles[c];
                    pn += moles * p.gas_pressure[c];
                    tn += moles * p.temperature[c];
                }
            },
            "Headspace ownership evaluation failed on another rank.");
        const double available = reduce(n.value(), Teuchos::REDUCE_SUM);
        if (available == 0)
            return;
        const double P = reduce(pn.value(), Teuchos::REDUCE_SUM) / available,
                     T = reduce(tn.value(), Teuchos::REDUCE_SUM) / available;
        const double y = reduce(h2.value(), Teuchos::REDUCE_SUM) / available;
        double removed = 0;
        collective(
            [&]
            {
                options.headspace.validate(P, T, y);
                removed = dt * options.headspace.molar_rate(P, T, y);
                if (!std::isfinite(removed) || removed < 0 || removed > available)
                    throw std::runtime_error("Headspace outflow is invalid or exceeds its available gas.");
            },
            "Headspace vent law failed on another rank.");
        identical(removed, "Headspace vent laws disagree across ranks.");
        const double fraction = removed / available;
        for (std::size_t c = 0; c < p.volume.size(); ++c)
            if (eligible[c])
            {
                const double H = fraction * working.resolved_hydrogen_moles[c],
                             I = fraction * working.resolved_inert_moles[c];
                source.vent_hydrogen_moles[c] = H;
                source.vent_inert_moles[c] = I;
                source.vent_enthalpy[c] = H * p.hydrogen_enthalpy[c] + I * p.inert_enthalpy[c];
                const double mass =
                    H * flow->low_mach_options().hydrogen.molar_mass + I * flow->low_mach_options().inert.molar_mass;
                for (std::size_t a = 0; a < 3; ++a)
                    source.vent_momentum[c][a] = mass * velocity[c][a];
            }
        stats.hydrogen_escaped = total(source.vent_hydrogen_moles);
        stats.escaped_enthalpy = total(source.vent_enthalpy);
    }

    /** Derive physical particle geometry from the new moments, before capture. */
    void update_dispersed_geometry(State& working, const Properties& p) const
    {
        const auto& r = options.radiolysis;
        const auto& thermodynamics = flow->low_mach_options();
        const auto& gravity = flow->options().gravity;
        const double acceleration = std::sqrt(dot(gravity, gravity));
        const double minimum = std::max(r.min_radius, thermodynamics.minimum_bubble_radius);
        const double maximum = std::min(r.max_radius, thermodynamics.maximum_bubble_radius);
        collective(
            [&]
            {
                for (auto& population : working.dispersed)
                {
                    population.radius.assign(p.volume.size(), 0.0);
                    population.volume.assign(p.volume.size(), 0.0);
                    population.drift_velocity.assign(p.volume.size(), Vector{});
                    for (std::size_t c = 0; c < p.volume.size(); ++c)
                    {
                        const double number = population.number[c], moles = population.hydrogen_moles[c];
                        if (!std::isfinite(number) || !std::isfinite(moles) || number < 0 || moles < 0 ||
                            ((number == 0) != (moles == 0)))
                            throw std::runtime_error(
                                "A dispersed population must carry both positive number and moles, or neither.");
                        if (number == 0)
                            continue;
                        const double sigma = surface_tension(r, p.temperature[c]);
                        const auto radius = Physics::solve_bubble_radius(moles / number, p.pressure[c], sigma,
                            thermodynamics.gas_constant, p.temperature[c], minimum, maximum, r.max_radius_iterations);
                        if (!radius.converged || !(radius.radius > 0))
                            throw std::domain_error(
                                "The updated dispersed moments have no radius inside the configured physical domain.");
                        const double bubble_volume = 4.0 * std::numbers::pi / 3.0 * std::pow(radius.radius, 3);
                        population.radius[c] = radius.radius;
                        population.volume[c] = number * bubble_volume;
                        if (!std::isfinite(population.volume[c]) || population.volume[c] <= 0)
                            throw std::runtime_error("The updated dispersed volume is not representable.");
                        double speed = 0;
                        if (r.rise_velocity_mode == BubbleRiseVelocityMode::ConstantSlip)
                            speed = r.constant_slip_velocity;
                        else if (r.rise_velocity_mode == BubbleRiseVelocityMode::Celata2007)
                        {
                            const double gas_density =
                                (moles / number) * thermodynamics.hydrogen.molar_mass / bubble_volume;
                            const auto rise = Physics::celata2007_bubble_rise_velocity(radius.radius, p.density[c],
                                gas_density, p.viscosity[c], sigma, acceleration, r.max_rise_velocity_iterations,
                                r.rise_velocity_tolerance);
                            if (!rise.converged || !rise.within_experimental_range)
                                throw std::domain_error(
                                    "Celata slip is unconverged or outside its declared experimental domain.");
                            speed = rise.velocity;
                        }
                        if (speed != 0)
                        {
                            if (!(acceleration > 0))
                                throw std::domain_error("A nonzero rise speed requires a physical gravity direction.");
                            for (std::size_t a = 0; a < 3; ++a)
                                population.drift_velocity[c][a] = -speed * gravity[a] / acceleration;
                        }
                    }
                }
            },
            "Dispersed radius/slip evaluation failed on another rank.");
    }

    Sources reactions(double dt, const ReactingVoFInput& input, const State& base, const State& guess,
        const std::vector<Vector>& velocity, ReactingVoFStatistics& stats) const
    {
        const std::size_t n = base.liquid_mass.size();
        Sources source;
        for (auto* values :
            {&source.liquid_mass_change, &source.resolved_hydrogen_change, &source.resolved_inert_change,
                &source.dissolved_hydrogen_change, &source.donor_hydrogen_change, &source.net_thermal_energy,
                &source.radiolysis_energy, &source.enthalpy_transfer, &source.vent_hydrogen_moles,
                &source.vent_inert_moles, &source.vent_enthalpy, &source.produced_hydrogen, &source.captured_hydrogen})
            values->assign(n, 0.0);
        source.momentum_transfer.resize(n);
        source.vent_momentum.resize(n);
        for (const auto& population : base.dispersed)
            source.dispersed.push_back({population.name, std::vector<double>(n), std::vector<double>(n), {}});
        if (!input.net_thermal_energy.empty())
            source.net_thermal_energy = input.net_thermal_energy;
        if (!input.absorbed_radiation_energy.empty())
            source.radiolysis_energy = input.absorbed_radiation_energy;
        stats.thermal_input = total(source.net_thermal_energy);
        const auto p = properties(guess);
        State working = base;
        ReactionLedger ledger(n, base.dispersed.size());
        const auto diffusion_change = diffuse(working, p, base, dt);
        for (std::size_t c = 0; c < n; ++c)
            ledger.dissolved[c] += diffusion_change[c];
        const auto patches = (options.resolved_transfer.enabled || options.capture.enabled)
                                 ? geometry->interface_patches(base.resolved_fraction)
                                 : std::vector<VoF::InterfacePatch>{};
        const auto& r = options.radiolysis;
        const bool active = r.mode != ReactingGasMode::Disabled;
        int substeps = 1;
        if (active)
        {
            const double count = std::ceil((dt / r.microbubble_lifetime) / 0.2);
            require_all(std::isfinite(count) && count <= r.max_subcycles,
                "Microbubble kinetics requires more than the configured substep budget; reduce the timestep.");
            substeps = std::max(1, static_cast<int>(count));
        }
        stats.reaction_substeps = substeps;
        const double duration = dt / substeps;
        const double decay = active ? -std::expm1(-duration / r.microbubble_lifetime) : 0;
        collective(
            [&]
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    if (active)
                    {
                        source.produced_hydrogen[c] =
                            r.gas_release_efficiency * r.hydrogen_yield_mol_per_j * source.radiolysis_energy[c];
                        if (!std::isfinite(source.produced_hydrogen[c]) ||
                            source.produced_hydrogen[c] > base.donor_hydrogen_moles[c])
                            throw std::runtime_error("Reacting production exhausts its finite chemical donor.");
                        if (source.produced_hydrogen[c] > 0 && !(p.liquid_volume[c] > 0))
                            throw std::runtime_error("Liquid-absorbed radiolysis requires a true aqueous carrier.");
                        source.donor_hydrogen_change[c] = -source.produced_hydrogen[c];
                        ledger.liquid_moles[c] += -source.produced_hydrogen[c];
                    }
                    // Number is the canonical fixed-pivot variable. Only transport
                    // roundoff may be reconciled with its paired extensive moles.
                    for (std::size_t i = 0; i < upper.size(); ++i)
                    {
                        auto& population = working.dispersed[upper[i]];
                        const double moles = options.upper_pivots.mol_per_bubble[i] * population.number[c];
                        const double scale =
                            std::max({moles, population.hydrogen_moles[c], std::numeric_limits<double>::min()});
                        if (!std::isfinite(moles) || std::abs(moles - population.hydrogen_moles[c]) >
                                                         256 * std::numeric_limits<double>::epsilon() * scale)
                            throw std::runtime_error(
                                "Transported upper-bin number and molar moments disagree with their fixed pivot.");
                        population.hydrogen_moles[c] = moles;
                    }
                }
            },
            "Reacting source/pivot preflight failed on another rank.");
        for (int cycle = 0; cycle < substeps; ++cycle)
        {
            collective(
                [&]
                {
                    for (std::size_t c = 0; c < n; ++c)
                    {
                        const double volume = p.volume[c], aqueous = p.liquid_volume[c] / volume;
                        Physics::BubblePopulationBirth births;
                        if (active && aqueous > 0)
                        {
                            const double sigma = surface_tension(r, p.temperature[c]);
                            const double radius = Physics::nucleation_radius(r, p.temperature[c], p.pressure[c]);
                            if (radius < r.min_radius || radius > r.max_radius)
                                throw std::domain_error("Nucleation radius is outside the configured physical domain.");
                            const double nucleation =
                                4.0 * std::numbers::pi / 3.0 *
                                (p.pressure[c] * radius * radius * radius + 2 * sigma * radius * radius) /
                                (r.gas_constant * p.temperature[c]);
                            auto& population = working.dispersed[micro];
                            const Physics::MicrobubbleSubstep step{.duration = duration,
                                .bulk_production_rate = source.produced_hydrogen[c] / volume / dt,
                                .liquid_fraction = aqueous,
                                .nucleation_moles = nucleation,
                                .critical_concentration = Physics::henry_equilibrium_concentration(
                                    r.henry_coefficient, p.pressure[c], sigma, radius),
                                .micro_decay_fraction = decay,
                                .micro_retention = 1 - decay,
                                .conversion_pressure_factor = r.micro_to_large_conversion_coefficient * p.pressure[c],
                                .heaviside_mode = r.heaviside_mode,
                                .smooth_heaviside_width = r.smooth_heaviside_width};
                            const auto result = Physics::advance_microbubble_substep(
                                {working.dissolved_hydrogen_moles[c] / volume, population.number[c] / volume,
                                    population.hydrogen_moles[c] / volume},
                                step);
                            if ((result.state.number_density == 0) != (result.state.molar_inventory == 0) ||
                                (result.converted.number_density == 0) != (result.converted.molar_inventory == 0))
                                throw std::domain_error(
                                    "Legacy microbubble conversion produced unpaired number/moles: the returned or "
                                    "transported micro moments are outside this conversion substep's admissible "
                                    "domain.");
                            working.dissolved_hydrogen_moles[c] = result.state.dissolved_inventory * volume;
                            population.number[c] = result.state.number_density * volume;
                            population.hydrogen_moles[c] = result.state.molar_inventory * volume;
                            ledger.number[micro][c] += result.number_change * volume;
                            ledger.moles[micro][c] += result.molar_change * volume;
                            ledger.dissolved[c] += result.dissolved_moles * volume;
                            ledger.liquid_moles[c] += result.dissolved_moles * volume;
                            births = result.converted;
                            // The entire radiation debit was checked before any
                            // substep; compute its endpoint from that same stock.
                            const double produced =
                                (cycle + 1 == substeps)
                                    ? source.produced_hydrogen[c]
                                    : source.produced_hydrogen[c] * (static_cast<double>(cycle + 1) / substeps);
                            working.donor_hydrogen_moles[c] = base.donor_hydrogen_moles[c] - produced;
                        }
                        else if (active &&
                                 (working.dissolved_hydrogen_moles[c] != 0 || working.dispersed[micro].number[c] != 0 ||
                                     working.dispersed[micro].hydrogen_moles[c] != 0))
                            throw std::runtime_error("Microbubble kinetics has inventory without true aqueous liquid.");
                        if (upper.empty())
                        {
                            if (births.number_density != 0 || births.molar_inventory != 0)
                                throw std::runtime_error("Microbubble conversion has no upper-population destination.");
                            continue;
                        }
                        if (aqueous <= 0)
                        {
                            bool empty = working.dissolved_hydrogen_moles[c] == 0;
                            for (std::size_t index : upper)
                                empty = empty && working.dispersed[index].number[c] == 0 &&
                                        working.dispersed[index].hydrogen_moles[c] == 0;
                            if (!empty)
                                throw std::runtime_error(
                                    "Upper-population kinetics has inventory without aqueous liquid.");
                            continue;
                        }
                        Physics::UpperBubblePopulation initial;
                        initial.dissolved_inventory = working.dissolved_hydrogen_moles[c] / volume;
                        for (std::size_t index : upper)
                            initial.number_density.push_back(working.dispersed[index].number[c] / volume);
                        const Physics::BubblePopulationEnvironment environment{.time = flow->time() + cycle * duration,
                            .temperature = p.temperature[c],
                            .liquid_pressure = p.pressure[c],
                            .liquid_fraction = aqueous,
                            .carrier_fraction = p.carrier_volume[c] / volume,
                            .liquid_density = p.density[c],
                            .dynamic_viscosity = p.viscosity[c],
                            .surface_tension = surface_tension(r, p.temperature[c]),
                            .hydrogen_diffusivity = p.diffusivity[c],
                            .gas_constant = r.gas_constant};
                        const auto result = Physics::advance_upper_bubble_population(options.upper_pivots, initial,
                            environment, duration, options.upper_kernels, {births}, options.population_step);
                        working.dissolved_hydrogen_moles[c] = result.state.dissolved_inventory * volume;
                        ledger.dissolved[c] += result.gas_to_dissolved_moles * volume;
                        ledger.dissolved[c] += -result.dissolved_to_gas_moles * volume;
                        ledger.liquid_moles[c] += result.gas_to_dissolved_moles * volume;
                        ledger.liquid_moles[c] += -result.dissolved_to_gas_moles * volume;
                        for (std::size_t i = 0; i < upper.size(); ++i)
                        {
                            auto& population = working.dispersed[upper[i]];
                            population.number[c] = result.state.number_density[i] * volume;
                            population.hydrogen_moles[c] =
                                options.upper_pivots.mol_per_bubble[i] * population.number[c];
                            ledger.number[upper[i]][c] += result.number_change[i] * volume;
                            ledger.moles[upper[i]][c] += result.hydrogen_change[i] * volume;
                        }
                        // Underflow enters micro only after this source substep;
                        // it cannot decay/convert a second time in the same cycle.
                        if (result.returned_to_micro.number_density != 0 ||
                            result.returned_to_micro.molar_inventory != 0)
                        {
                            auto& population = working.dispersed[micro];
                            population.number[c] += result.returned_to_micro.number_density * volume;
                            population.hydrogen_moles[c] += result.returned_to_micro.molar_inventory * volume;
                            ledger.number[micro][c] += result.returned_to_micro.number_density * volume;
                            ledger.moles[micro][c] += result.returned_to_micro.molar_inventory * volume;
                        }
                    }
                },
                "Reacting population substep failed on another rank.");
            update_dispersed_geometry(working, p);
            const auto transfer = interface_transfer(working, p, patches, velocity, duration, source, ledger, stats);
            capture(working, p, patches, velocity, transfer, duration, source, ledger, stats);
        }
        vent(working, p, base, velocity, dt, source, stats);
        const double molar_mass = flow->low_mach_options().hydrogen.molar_mass;
        for (std::size_t c = 0; c < n; ++c)
        {
            source.resolved_hydrogen_change[c] = ledger.resolved[c].value();
            source.dissolved_hydrogen_change[c] = ledger.dissolved[c].value();
            // Chemical liquid mass uses shared molecular events directly;
            // dilute counterdiffusion has no mixture-mass contribution.
            source.liquid_mass_change[c] = molar_mass * ledger.liquid_moles[c].value();
            for (std::size_t k = 0; k < working.dispersed.size(); ++k)
            {
                source.dispersed[k].number_change[c] = ledger.number[k][c].value();
                source.dispersed[k].hydrogen_change[c] = ledger.moles[k][c].value();
            }
        }
        for (std::size_t k = 0; k < working.dispersed.size(); ++k)
            source.dispersed[k].drift_velocity = working.dispersed[k].drift_velocity;
        stats.hydrogen_produced = total(source.produced_hydrogen);
        return source;
    }

    double thermo_change(const State& a, const State& b) const
    {
        double difference = relative_change(a.thermodynamic_pressure, b.thermodynamic_pressure);
        for (std::size_t c = 0; c < a.temperature.size(); ++c)
        {
            for (const auto fields : {std::pair{&a.temperature, &b.temperature},
                     std::pair{&a.liquid_absolute_pressure, &b.liquid_absolute_pressure},
                     std::pair{&a.resolved_absolute_pressure, &b.resolved_absolute_pressure}})
                difference = std::max(difference, relative_change((*fields.first)[c], (*fields.second)[c]));
            difference = std::max({difference, std::abs(a.resolved_fraction[c] - b.resolved_fraction[c]),
                std::abs(a.liquid_fraction[c] - b.liquid_fraction[c])});
        }
        return reduce(difference, Teuchos::REDUCE_MAX);
    }

    double source_change(const Sources& a, const Sources& b, const State& candidate, double dt,
        std::string* failure_detail = nullptr) const
    {
        double difference = 0;
        double worst_before = 0, worst_after = 0;
        std::string worst_name = "none";
        const auto update_metric = [&](double change, double x, double y, const std::string& name)
        {
            if (change > difference)
            {
                difference = change;
                worst_before = x;
                worst_after = y;
                if (failure_detail)
                    worst_name = name;
            }
        };
        const auto update = [&](double x, double y, const std::string& name)
        { update_metric(relative_change(x, y) / options.relative_tolerance, x, y, name); };
        const auto scalars = [&](const std::vector<double>& x, const std::vector<double>& y, const std::string& name)
        {
            for (std::size_t c = 0; c < x.size(); ++c)
                update(x[c], y[c], failure_detail ? name + "[" + std::to_string(c) + "]" : name);
        };
        const auto vectors = [&](const std::vector<Vector>& x, const std::vector<Vector>& y, const std::string& name)
        {
            for (std::size_t c = 0; c < x.size(); ++c)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    update(x[c][axis], y[c][axis],
                        failure_detail ? name + "[" + std::to_string(c) + "][" + std::to_string(axis) + "]" : name);
        };
        const auto impulses = [&](const std::vector<Vector>& x, const std::vector<Vector>& y, const std::string& name)
        {
            const auto& thermo = flow->low_mach_options();
            for (std::size_t c = 0; c < x.size(); ++c)
            {
                double mass = candidate.liquid_mass[c] +
                              thermo.hydrogen.molar_mass * candidate.resolved_hydrogen_moles[c] +
                              thermo.inert.molar_mass * candidate.resolved_inert_moles[c];
                for (const auto& population : candidate.dispersed)
                    mass += thermo.hydrogen.molar_mass * population.hydrogen_moles[c];
                const auto spacing = geometry->hierarchy()->block_spacing(
                    geometry->hierarchy()->cell_block(geometry->mesh()->cell_global_id(static_cast<LO>(c))));
                const double width = *std::min_element(spacing.begin(), spacing.end());
                // A near-zero impulse has no useful relative scale. Measure
                // its actual mixture-velocity effect with the same dt*du/h
                // convergence criterion used by the flow corrector.
                const double delta = std::hypot(x[c][0] - y[c][0], x[c][1] - y[c][1], x[c][2] - y[c][2]);
                const double effect = (dt / width) * (delta / mass);
                update_metric(effect / thermo.volume_relative_tolerance, std::hypot(x[c][0], x[c][1], x[c][2]),
                    std::hypot(y[c][0], y[c][1], y[c][2]),
                    failure_detail ? name + "[" + std::to_string(c) + "] impulse displacement" : name);
            }
        };
#define SF_COMPARE_SOURCE(field) scalars(a.field, b.field, #field)
        SF_COMPARE_SOURCE(liquid_mass_change);
        SF_COMPARE_SOURCE(resolved_hydrogen_change);
        SF_COMPARE_SOURCE(resolved_inert_change);
        SF_COMPARE_SOURCE(dissolved_hydrogen_change);
        SF_COMPARE_SOURCE(donor_hydrogen_change);
        SF_COMPARE_SOURCE(enthalpy_transfer);
        SF_COMPARE_SOURCE(vent_hydrogen_moles);
        SF_COMPARE_SOURCE(vent_inert_moles);
        SF_COMPARE_SOURCE(vent_enthalpy);
        SF_COMPARE_SOURCE(captured_hydrogen);
#undef SF_COMPARE_SOURCE
        impulses(a.momentum_transfer, b.momentum_transfer, "momentum_transfer");
        impulses(a.vent_momentum, b.vent_momentum, "vent_momentum");
        for (std::size_t k = 0; k < a.dispersed.size(); ++k)
        {
            const auto name = "population " + a.dispersed[k].name;
            scalars(a.dispersed[k].number_change, b.dispersed[k].number_change, name + " number_change");
            scalars(a.dispersed[k].hydrogen_change, b.dispersed[k].hydrogen_change, name + " hydrogen_change");
            vectors(a.dispersed[k].drift_velocity, b.dispersed[k].drift_velocity, name + " drift_velocity");
        }
        const double maximum = reduce(difference, Teuchos::REDUCE_MAX);
        if (failure_detail)
        {
            const auto comm = Tpetra::getDefaultComm();
            const int local = difference == maximum ? comm->getRank() : comm->getSize();
            int owner = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &owner);
            int length = comm->getRank() == owner ? static_cast<int>(worst_name.size()) : 0;
            Teuchos::broadcast(*comm, owner, 1, &length);
            worst_name.resize(static_cast<std::size_t>(length));
            if (length)
                Teuchos::broadcast(*comm, owner, length, worst_name.data());
            std::array<double, 2> values{worst_before, worst_after};
            Teuchos::broadcast(*comm, owner, 2, values.data());
            std::ostringstream detail;
            detail << std::setprecision(17) << worst_name << " on rank " << owner << ", before=" << values[0]
                   << ", after=" << values[1];
            *failure_detail = detail.str();
        }
        return maximum;
    }
};

ReactingVoFCoupler::ReactingVoFCoupler(SP<AdaptiveVoFSolver> solver, ReactingVoFOptions options)
    : impl_(std::make_unique<Impl>(std::move(solver), std::move(options)))
{
}
ReactingVoFCoupler::~ReactingVoFCoupler() = default;
ReactingVoFCoupler::ReactingVoFCoupler(ReactingVoFCoupler&&) noexcept = default;
ReactingVoFCoupler& ReactingVoFCoupler::operator=(ReactingVoFCoupler&&) noexcept = default;

ReactingVoFCoupler::Trial::Trial(std::shared_ptr<const void> candidate) : candidate_(std::move(candidate)) {}
const VoF::LowMachState& ReactingVoFCoupler::Trial::state() const
{
    if (!candidate_)
        throw std::logic_error("Reacting VoF trial is empty.");
    return std::static_pointer_cast<const Impl::Candidate>(candidate_)->fluid.state();
}
const ReactingVoFStatistics& ReactingVoFCoupler::Trial::statistics() const
{
    if (!candidate_)
        throw std::logic_error("Reacting VoF trial is empty.");
    return std::static_pointer_cast<const Impl::Candidate>(candidate_)->stats;
}

ReactingVoFCoupler::Trial ReactingVoFCoupler::preview(double dt, const ReactingVoFInput& input)
{
    identical(dt, "Reacting VoF timesteps must be finite and identical across ranks.");
    require_all(dt > 0, "Reacting VoF requires a positive timestep.");
    impl_->refresh_geometry();
    const State base = impl_->flow->low_mach_state();
    const auto n = base.liquid_mass.size();
    bool valid = (input.absorbed_radiation_energy.empty() || input.absorbed_radiation_energy.size() == n) &&
                 (input.net_thermal_energy.empty() || input.net_thermal_energy.size() == n);
    for (double value : input.absorbed_radiation_energy)
        valid = valid && std::isfinite(value) && value >= 0;
    for (double value : input.net_thermal_energy)
        valid = valid && std::isfinite(value);
    require_all(
        valid, "Reacting VoF input must contain finite owned-cell energy; absorbed radiation is nonnegative.");
    const double before = hydrogen_total(base), donor_before = total(base.donor_hydrogen_moles);
    State guess = base;
    auto velocity = impl_->flow->velocity();
    std::optional<Sources> previous;
    double last_thermodynamic_change = 0, last_source_change = 0;
    std::string source_failure_detail;
    for (int iteration = 0; iteration < impl_->options.maximum_correctors; ++iteration)
    {
        ReactingVoFStatistics stats;
        auto sources = impl_->reactions(dt, input, base, guess, velocity, stats);
        auto fluid = impl_->flow->preview_low_mach_step(dt, sources, &guess);
        const double thermodynamic_change = impl_->thermo_change(guess, fluid.state());
        const double source_change =
            previous ? impl_->source_change(*previous, sources, fluid.state(), dt,
                           iteration + 1 == impl_->options.maximum_correctors ? &source_failure_detail : nullptr)
                     : std::numeric_limits<double>::infinity();
        last_thermodynamic_change = thermodynamic_change;
        last_source_change = source_change;
        if (thermodynamic_change <= impl_->options.relative_tolerance && source_change <= 1.0)
        {
            stats.time = impl_->flow->time() + dt;
            stats.correctors = iteration + 1;
            stats.hydrogen_before = before;
            stats.hydrogen_after = hydrogen_total(fluid.state());
            const double donor_after = total(fluid.state().donor_hydrogen_moles);
            stats.donor_consumed = donor_before - donor_after;
            detail::CompensatedSum<double> hydrogen_error, donor_error;
            hydrogen_error += stats.hydrogen_after;
            hydrogen_error += -before;
            hydrogen_error += -stats.hydrogen_produced;
            hydrogen_error += stats.hydrogen_escaped;
            donor_error += donor_before;
            donor_error += -donor_after;
            donor_error += -stats.hydrogen_produced;
            stats.hydrogen_error = hydrogen_error.value();
            stats.donor_error = donor_error.value();
            const double hydrogen_scale =
                std::max(before + stats.hydrogen_after + stats.hydrogen_produced + stats.hydrogen_escaped,
                    std::numeric_limits<double>::min());
            const double donor_scale =
                std::max(donor_before + donor_after + stats.hydrogen_produced, std::numeric_limits<double>::min());
            require_all(
                std::abs(stats.hydrogen_error) <= impl_->options.inventory_relative_tolerance * hydrogen_scale &&
                    std::abs(stats.donor_error) <= impl_->options.inventory_relative_tolerance * donor_scale,
                "Reacting VoF failed its hydrogen/chemical-donor conservation gate.");
            stats.mass_error = fluid.diagnostics().mass_error;
            stats.energy_error = fluid.diagnostics().energy_error;
            stats.cumulative_produced = impl_->accepted.cumulative_produced + stats.hydrogen_produced;
            stats.cumulative_captured = impl_->accepted.cumulative_captured + stats.hydrogen_captured;
            stats.cumulative_escaped = impl_->accepted.cumulative_escaped + stats.hydrogen_escaped;
            auto candidate = std::make_shared<const Impl::Candidate>(
                Impl::Candidate{impl_.get(), impl_->revision, std::move(fluid), stats});
            return Trial(std::move(candidate));
        }
        guess = fluid.state();
        velocity = fluid.velocity();
        previous = std::move(sources);
    }
    std::ostringstream message;
    message << std::setprecision(17)
            << "Reacting VoF source/thermodynamic correctors did not converge; no state was published. "
            << "thermodynamic residual=" << last_thermodynamic_change
            << ", normalized source residual=" << last_source_change << "; " << source_failure_detail;
    throw std::runtime_error(message.str());
}

void ReactingVoFCoupler::commit(const Trial& trial)
{
    const auto candidate = std::static_pointer_cast<const Impl::Candidate>(trial.candidate_);
    require_all(candidate && candidate->owner == impl_.get() && candidate->revision == impl_->revision &&
                    impl_->revision < std::numeric_limits<std::uint64_t>::max(),
        "Reacting VoF trial is foreign, stale or already committed.");
    impl_->flow->commit_low_mach_trial(candidate->fluid);
    impl_->accepted = candidate->stats;
    ++impl_->revision;
}
ReactingVoFStatistics ReactingVoFCoupler::step(double dt, const ReactingVoFInput& input)
{
    const auto trial = preview(dt, input);
    commit(trial);
    return impl_->accepted;
}
ReactingVoFCoupler::Checkpoint ReactingVoFCoupler::checkpoint()
{
    require_all(!impl_->saved, "Reacting VoF already has an active checkpoint.");
    const auto token = impl_->flow->checkpoint();
    impl_->saved = std::pair{token, impl_->accepted};
    return token;
}
void ReactingVoFCoupler::restore(Checkpoint token)
{
    require_all(
        impl_->saved && impl_->saved->first == token && impl_->revision < std::numeric_limits<std::uint64_t>::max(),
        "Reacting VoF checkpoint token is not active or its revision counter is exhausted.");
    impl_->flow->restore(token);
    impl_->accepted = impl_->saved->second;
    ++impl_->revision;
    impl_->geometry.reset();
}
void ReactingVoFCoupler::release_checkpoint(Checkpoint token)
{
    require_all(impl_->saved && impl_->saved->first == token, "Reacting VoF checkpoint token is not active.");
    impl_->flow->release_checkpoint(token);
    impl_->saved.reset();
}
void ReactingVoFCoupler::regrid(const VoF::RefinementPlan& plan)
{
    require_all(
        impl_->revision < std::numeric_limits<std::uint64_t>::max(), "Reacting VoF revision counter is exhausted.");
    impl_->flow->regrid(plan);
    impl_->geometry.reset();
    ++impl_->revision;
}
const SP<AdaptiveVoFSolver>& ReactingVoFCoupler::solver() const noexcept
{
    return impl_->flow;
}
const ReactingVoFOptions& ReactingVoFCoupler::options() const noexcept
{
    return impl_->options;
}
const ReactingVoFStatistics& ReactingVoFCoupler::statistics() const noexcept
{
    return impl_->accepted;
}

} // namespace SimpleFluid
