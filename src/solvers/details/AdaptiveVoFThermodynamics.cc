/** @file AdaptiveVoFThermodynamics.cc
 * @brief Adaptive phase thermodynamics, material closure and stability.
 */
#include "solvers/details/AdaptiveVoFContext.hh"
namespace SimpleFluid::detail::adaptive_vof
{
double AdaptiveVoFContext::gas_e(const VoF::IdealGasSpecies& gas, double temperature) const
{
    return gas.reference_internal_energy + gas.molar_heat_capacity_v * (temperature - gas.reference_temperature);
}

double AdaptiveVoFContext::low_mass(const VoF::LowMachState& s, std::size_t c) const
{
    double value = s.liquid_mass[c] + low_options->hydrogen.molar_mass * s.resolved_hydrogen_moles[c] +
                   low_options->inert.molar_mass * s.resolved_inert_moles[c];
    for (const auto& population : s.dispersed)
        value += low_options->hydrogen.molar_mass * population.hydrogen_moles[c];
    return value;
}

Vector AdaptiveVoFContext::drift_momentum(const VoF::LowMachState& s, std::size_t c) const
{
    Vector value{};
    for (const auto& population : s.dispersed)
        for (std::size_t a = 0; a < 3; ++a)
            value[a] +=
                low_options->hydrogen.molar_mass * population.hydrogen_moles[c] * population.drift_velocity[c][a];
    return value;
}

Vector AdaptiveVoFContext::low_drift_momentum_l1(const VoF::LowMachState& state) const
{
    std::array<detail::CompensatedSum<>, 3> local;
    for (const auto& population : state.dispersed)
        for (std::size_t c = 0; c < owned(); ++c)
            for (std::size_t a = 0; a < 3; ++a)
                local[a] += low_options->hydrogen.molar_mass * population.hydrogen_moles[c] *
                            std::abs(population.drift_velocity[c][a]);
    Vector result{};
    for (std::size_t a = 0; a < 3; ++a)
        result[a] = sum(static_cast<double>(local[a].value()));
    return result;
}

double AdaptiveVoFContext::low_momentum_roundoff_factor(const State& state) const
{
    // gamma_n bounds accumulation and the elementary multiply/divide/add
    // updates. Count actual pressure-face contributions, transport/viscous
    // faces, drift terms, reconstruction and a conservative rank-count bound
    // for the final reduction. Canonical face operands are shared by both cells.
    const double count = state.low_momentum_operation_bound + mesh->owned_cell_map()->getComm()->getSize() + 16;
    const double product = count * std::numeric_limits<double>::epsilon();
    require_all(product < 1, "Low-Mach momentum assembly exceeds its representable roundoff bound.");
    return product / (1 - product);
}

void AdaptiveVoFContext::validate_low_options(const VoF::LowMachOptions& o) const
{
    require_all(bool(o.liquid), "Low-Mach mode requires an immutable liquid caloric EOS.");
    const auto& b = o.liquid->bounds();
    for (double value : {b.minimum_temperature, b.maximum_temperature, b.minimum_pressure, b.maximum_pressure,
             o.gas_constant, o.minimum_bubble_radius, o.maximum_bubble_radius, o.volume_relative_tolerance,
             o.energy_relative_tolerance, o.inventory_relative_tolerance, o.relaxation,
             static_cast<double>(o.maximum_correctors), static_cast<double>(o.maximum_thermodynamic_iterations)})
        identical(value, "Low-Mach thermodynamic controls differ across ranks or are nonfinite.");
    for (const auto* gas : {&o.hydrogen, &o.inert})
        for (double value : {gas->molar_mass, gas->molar_heat_capacity_v, gas->reference_temperature,
                 gas->reference_internal_energy, gas->dynamic_viscosity, gas->thermal_conductivity})
            identical(value, "Low-Mach ideal-gas coefficients differ across ranks or are nonfinite.");
    require_all(o.gas_constant > 0 && o.minimum_bubble_radius > 0 &&
                    o.maximum_bubble_radius > o.minimum_bubble_radius && o.maximum_correctors > 0 &&
                    o.maximum_thermodynamic_iterations > 0 && o.volume_relative_tolerance > 0 &&
                    o.energy_relative_tolerance > 0 && o.inventory_relative_tolerance > 0 && o.relaxation > 0 &&
                    o.relaxation <= 1 && o.hydrogen.molar_mass > 0 && o.inert.molar_mass > 0 &&
                    o.hydrogen.molar_heat_capacity_v > 0 && o.inert.molar_heat_capacity_v > 0 &&
                    o.hydrogen.reference_temperature > 0 && o.inert.reference_temperature > 0 &&
                    o.hydrogen.dynamic_viscosity >= 0 && o.inert.dynamic_viscosity >= 0 &&
                    o.hydrogen.thermal_conductivity >= 0 && o.inert.thermal_conductivity >= 0,
        "Low-Mach thermodynamic controls or ideal-gas properties are invalid.");
}

std::vector<double> AdaptiveVoFContext::low_values(const std::vector<double>& input, bool nonnegative, const char* message) const
{
    bool valid = input.empty() || input.size() == owned();
    for (double v : input)
        valid = valid && std::isfinite(v) && (!nonnegative || v >= 0);
    require_all(valid, message);
    return input.empty() ? std::vector<double>(owned(), 0) : input;
}

std::vector<Vector> AdaptiveVoFContext::low_vectors(const std::vector<Vector>& input, const char* message) const
{
    bool valid = input.empty() || input.size() == owned();
    for (const auto& v : input)
        for (double x : v)
            valid = valid && std::isfinite(x);
    require_all(valid, message);
    return input.empty() ? std::vector<Vector>(owned()) : input;
}

void AdaptiveVoFContext::validate_low_inventory(const VoF::LowMachState& s) const
{
    bool valid = true;
    for (const auto* values : {&s.liquid_mass, &s.resolved_hydrogen_moles, &s.resolved_inert_moles,
             &s.dissolved_hydrogen_moles, &s.donor_hydrogen_moles})
        valid = valid && values->size() == owned() &&
                std::all_of(values->begin(), values->end(), [](double v) { return std::isfinite(v) && v >= 0; });
    valid = valid && s.internal_energy.size() == owned() &&
            std::all_of(s.internal_energy.begin(), s.internal_energy.end(), [](double v) { return std::isfinite(v); });
    require_all(valid, "Low-Mach phase mass, species, or thermal inventory is invalid.");
    identical(static_cast<double>(s.dispersed.size()), "Low-Mach population counts differ across ranks.");
    for (const auto& p : s.dispersed)
    {
        identical_name(p.name);
        valid = !p.name.empty() && p.number.size() == owned() && p.hydrogen_moles.size() == owned() &&
                p.drift_velocity.size() == owned();
        if (valid)
            for (std::size_t c = 0; c < owned(); ++c)
            {
                valid = valid && std::isfinite(p.number[c]) && std::isfinite(p.hydrogen_moles[c]) && p.number[c] >= 0 &&
                        p.hydrogen_moles[c] >= 0 && ((p.number[c] == 0) == (p.hydrogen_moles[c] == 0));
                for (double w : p.drift_velocity[c])
                    valid = valid && std::isfinite(w);
            }
        require_all(valid, "Low-Mach populations require paired positive number/moles and finite supplied drift.");
    }
    for (std::size_t c = 0; c < owned(); ++c)
    {
        const double hydrogen_mass =
            low_options->hydrogen.molar_mass * (s.dissolved_hydrogen_moles[c] + s.donor_hydrogen_moles[c]);
        valid = valid && hydrogen_mass <= s.liquid_mass[c] + 64 * std::numeric_limits<double>::epsilon() *
                                                                 std::max(s.liquid_mass[c], hydrogen_mass);
    }
    require_all(valid, "Low-Mach chemically bound/dissolved hydrogen exceeds the actual liquid mass.");
}

AdaptiveVoFContext::LowTotals AdaptiveVoFContext::low_totals(const VoF::LowMachState& s) const
{
    std::array<detail::CompensatedSum<>, 11> local;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        double hydrogen = s.donor_hydrogen_moles[c] + s.dissolved_hydrogen_moles[c] + s.resolved_hydrogen_moles[c];
        for (const auto& p : s.dispersed)
            hydrogen += p.hydrogen_moles[c];
        local[0] += low_mass(s, c);
        local[1] += hydrogen;
        local[2] += s.resolved_inert_moles[c];
        local[3] += s.internal_energy[c];
        local[4] += std::abs(s.internal_energy[c]);
        for (std::size_t a = 0; a < 3; ++a)
        {
            local[5 + a] += s.momentum[c][a];
            local[8 + a] += std::abs(s.momentum[c][a]);
        }
    }
    std::array<double, 11> global{};
    for (std::size_t i = 0; i < global.size(); ++i)
        global[i] = sum(static_cast<double>(local[i].value()));
    return {global[0], global[1], global[2], global[3], global[4], {global[5], global[6], global[7]},
        {global[8], global[9], global[10]}};
}

std::optional<AdaptiveVoFContext::ThermoEvaluation> AdaptiveVoFContext::evaluate_low(const VoF::LowMachState& s, const std::vector<double>& mechanical,
    const std::vector<double>& f, const std::vector<double>& k, double p0) const
{
    ThermoEvaluation result;
    for (auto* values : {&result.temperature, &result.rho_l, &result.rho_r, &result.p_l, &result.p_r, &result.partial,
             &result.gas, &result.liquid, &result.dispersed})
        values->resize(owned());
    result.bubble_volume.resize(s.dispersed.size(), std::vector<double>(owned()));
    result.bubble_radius = result.bubble_volume;
    result.bubble_pressure = result.bubble_volume;
    bool valid = true;
    detail::CompensatedSum<> total;
    const auto& o = *low_options;
    const auto& bounds = o.liquid->bounds();
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double jump = material.surface_tension * k[c];
            const double pl = p0 + mechanical[c] - f[c] * jump, pr = p0 + mechanical[c] + (1 - f[c]) * jump;
            if (pl < bounds.minimum_pressure || pl > bounds.maximum_pressure || pr < bounds.minimum_pressure ||
                pr > bounds.maximum_pressure)
                throw std::out_of_range("Phase absolute pressure is outside the EOS bounds.");
            double nh = s.resolved_hydrogen_moles[c];
            for (const auto& population : s.dispersed)
                nh += population.hydrogen_moles[c];
            const double ni = s.resolved_inert_moles[c];
            const auto energy = [&](double temperature)
            {
                const auto liquid = o.liquid->evaluate(temperature, pl);
                return std::pair{s.liquid_mass[c] * liquid.internal_energy + nh * gas_e(o.hydrogen, temperature) +
                                     ni * gas_e(o.inert, temperature),
                    s.liquid_mass[c] * liquid.internal_energy_temperature + nh * o.hydrogen.molar_heat_capacity_v +
                        ni * o.inert.molar_heat_capacity_v};
            };
            double lower = bounds.minimum_temperature, upper = bounds.maximum_temperature;
            const auto lower_value = energy(lower), upper_value = energy(upper);
            const double target = s.internal_energy[c];
            const double roundoff = 128 * std::numeric_limits<double>::epsilon() *
                                    std::max({1e-30, std::abs(target), std::abs(lower_value.first),
                                        std::abs(upper_value.first), upper_value.second * upper});
            if (target < lower_value.first - roundoff || target > upper_value.first + roundoff ||
                !(lower_value.second > 0))
                throw std::out_of_range("Thermal inventory has no temperature in the EOS range.");
            double temperature =
                s.temperature.size() == owned() ? std::clamp(s.temperature[c], lower, upper) : 0.5 * (lower + upper);
            bool converged = false;
            for (int iteration = 0; iteration < o.maximum_thermodynamic_iterations; ++iteration)
            {
                const auto [value, derivative] = energy(temperature);
                if (std::abs(value - target) <= roundoff)
                {
                    converged = true;
                    break;
                }
                if (value > target)
                    upper = temperature;
                else
                    lower = temperature;
                const double proposed = temperature + (target - value) / derivative;
                temperature = proposed > lower && proposed < upper ? proposed : 0.5 * (lower + upper);
            }
            if (!converged)
                throw std::runtime_error("Low-Mach caloric inversion did not converge.");
            const auto liquid = o.liquid->evaluate(temperature, pl);
            const double resolved_n = s.resolved_hydrogen_moles[c] + s.resolved_inert_moles[c];
            const double vg = resolved_n * o.gas_constant * temperature / pr;
            const double ml = s.liquid_mass[c], vl = ml / liquid.density;
            double vd = 0;
            for (std::size_t q = 0; q < s.dispersed.size(); ++q)
            {
                const auto& population = s.dispersed[q];
                result.bubble_pressure[q][c] = pl;
                if (population.number[c] == 0)
                    continue;
                const auto radius = BubbleGasPhysics::solve_bubble_radius(
                    population.hydrogen_moles[c] / population.number[c], pl, material.surface_tension, o.gas_constant,
                    temperature, o.minimum_bubble_radius, o.maximum_bubble_radius, o.maximum_thermodynamic_iterations,
                    4 * std::numeric_limits<double>::epsilon());
                if (!radius.converged)
                    throw std::runtime_error("Dispersed Laplace/EOS radius did not converge.");
                result.bubble_radius[q][c] = radius.radius;
                result.bubble_pressure[q][c] = pl + 2 * material.surface_tension / radius.radius;
                result.bubble_volume[q][c] =
                    population.number[c] * 4 * std::numbers::pi / 3 * radius.radius * radius.radius * radius.radius;
                vd += result.bubble_volume[q][c];
            }
            const double molecular_mass = resolved_n > 0 ? (o.hydrogen.molar_mass * s.resolved_hydrogen_moles[c] +
                                                               o.inert.molar_mass * s.resolved_inert_moles[c]) /
                                                               resolved_n
                                                         : o.hydrogen.molar_mass;
            result.temperature[c] = temperature;
            result.rho_l[c] = liquid.density;
            result.rho_r[c] = molecular_mass * pr / (o.gas_constant * temperature);
            result.p_l[c] = pl;
            result.p_r[c] = pr;
            result.partial[c] = resolved_n > 0 ? pr * s.resolved_hydrogen_moles[c] / resolved_n : 0;
            result.gas[c] = vg;
            result.liquid[c] = vl;
            result.dispersed[c] = vd;
            total += static_cast<long double>(vg) + vl + vd;
        }
    }
    catch (...)
    {
        valid = false;
    }
    int local_valid = valid ? 1 : 0, all_valid = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_valid, &all_valid);
    if (!all_valid)
        return std::nullopt;
    result.material_volume = sum(static_cast<double>(total.value()));
    return result;
}

void AdaptiveVoFContext::assign_thermo(VoF::LowMachState& s, ThermoEvaluation evaluation, double p0) const
{
    s.pending_representation_relaxation = false;
    s.thermodynamic_pressure = p0;
    s.temperature = std::move(evaluation.temperature);
    s.liquid_density = std::move(evaluation.rho_l);
    s.resolved_density = std::move(evaluation.rho_r);
    s.liquid_absolute_pressure = std::move(evaluation.p_l);
    s.resolved_absolute_pressure = std::move(evaluation.p_r);
    s.resolved_hydrogen_partial_pressure = std::move(evaluation.partial);
    s.resolved_fraction.resize(owned());
    s.liquid_fraction.resize(owned());
    s.dispersed_fraction.resize(owned());
    for (std::size_t c = 0; c < owned(); ++c)
    {
        s.resolved_fraction[c] = evaluation.gas[c] / volume[c];
        s.liquid_fraction[c] = evaluation.liquid[c] / volume[c];
        s.dispersed_fraction[c] = evaluation.dispersed[c] / volume[c];
    }
    for (std::size_t q = 0; q < s.dispersed.size(); ++q)
    {
        s.dispersed[q].volume = std::move(evaluation.bubble_volume[q]);
        s.dispersed[q].radius = std::move(evaluation.bubble_radius[q]);
        s.dispersed[q].absolute_pressure = std::move(evaluation.bubble_pressure[q]);
    }
}

AdaptiveVoFContext::PhasePressureSusceptibility AdaptiveVoFContext::low_pressure_susceptibility(const VoF::LowMachState& state) const
{
    PhasePressureSusceptibility response{
        std::vector<double>(owned()), std::vector<double>(owned()), std::vector<double>(owned())};
    const auto& properties = *low_options;
    const auto& bounds = properties.liquid->bounds();
    const double difference_scale = std::cbrt(std::numeric_limits<double>::epsilon());
    bool valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double T = state.temperature[c], pl = state.liquid_absolute_pressure[c];
            const double pr = state.resolved_absolute_pressure[c], rho = state.liquid_density[c];
            const auto liquid = properties.liquid->evaluate(T, pl);
            const double plo = std::max(bounds.minimum_pressure, pl - difference_scale * pl);
            const double phi = std::min(bounds.maximum_pressure, pl + difference_scale * pl);
            const double tlo = std::max(bounds.minimum_temperature, T - difference_scale * T);
            const double thi = std::min(bounds.maximum_temperature, T + difference_scale * T);
            const double rho_p =
                (properties.liquid->evaluate(T, phi).density - properties.liquid->evaluate(T, plo).density) /
                (phi - plo);
            const double rho_t =
                (properties.liquid->evaluate(thi, pl).density - properties.liquid->evaluate(tlo, pl).density) /
                (thi - tlo);
            double capacity = state.liquid_mass[c] * liquid.internal_energy_temperature +
                              state.resolved_hydrogen_moles[c] * properties.hydrogen.molar_heat_capacity_v +
                              state.resolved_inert_moles[c] * properties.inert.molar_heat_capacity_v;
            for (const auto& population : state.dispersed)
                capacity += population.hydrogen_moles[c] * properties.hydrogen.molar_heat_capacity_v;
            // Gibbs consistency gives e_p=-T*v_T-p*v_p. Deriving it from
            // density partials avoids cancellation of an arbitrary caloric
            // reference in two large internal-energy values.
            const double energy_pressure = (T * rho_t + pl * rho_p) / (rho * rho);
            const double temperature_pressure = -state.liquid_mass[c] * energy_pressure / capacity;
            response.gas[c] = state.resolved_fraction[c] * volume[c] * (1 / pr - temperature_pressure / T);
            response.carrier[c] = state.liquid_mass[c] * (rho_p + rho_t * temperature_pressure) / (rho * rho);
            for (const auto& population : state.dispersed)
            {
                if (population.number[c] == 0)
                    continue;
                const double r = population.radius[c], denominator = 3 * pl * r + 4 * material.surface_tension;
                response.carrier[c] +=
                    population.volume[c] * (3 * r / denominator - 3 * (pl * r + 2 * material.surface_tension) /
                                                                      denominator * temperature_pressure / T);
            }
            response.total[c] = response.gas[c] + response.carrier[c];
            valid = valid && std::isfinite(response.gas[c]) && std::isfinite(response.carrier[c]) &&
                    std::isfinite(response.total[c]) && response.total[c] >= 0 && capacity > 0;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach fixed-energy EOS volume susceptibility is invalid or mechanically unstable.");
    return response;
}

double AdaptiveVoFContext::domain_volume() const
{
    detail::CompensatedSum<> local;
    for (std::size_t c = 0; c < owned(); ++c)
        local += volume[c];
    return sum(static_cast<double>(local.value()));
}

void AdaptiveVoFContext::recover_closed_thermo(VoF::LowMachState& s, const std::vector<double>& mechanical, const std::vector<double>& f,
    const std::vector<double>& k) const
{
    validate_low_inventory(s);
    const auto& o = *low_options;
    const auto& b = o.liquid->bounds();
    double lower = b.minimum_pressure, upper = b.maximum_pressure;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        const double jump = material.surface_tension * k[c];
        lower = std::max({lower, b.minimum_pressure - mechanical[c] + f[c] * jump,
            b.minimum_pressure - mechanical[c] - (1 - f[c]) * jump});
        upper = std::min({upper, b.maximum_pressure - mechanical[c] + f[c] * jump,
            b.maximum_pressure - mechanical[c] - (1 - f[c]) * jump});
    }
    lower = maximum(lower);
    double minus_upper = maximum(-upper);
    upper = -minus_upper;
    if (!(lower < upper))
    {
        double local_minimum = std::numeric_limits<double>::infinity();
        double local_maximum = -std::numeric_limits<double>::infinity();
        for (double value : mechanical)
        {
            local_minimum = std::min(local_minimum, value);
            local_maximum = std::max(local_maximum, value);
        }
        std::ostringstream message;
        message << std::setprecision(17)
                << "Closed-vessel EOS has no admissible absolute-pressure interval: P0 interval=[" << lower << ','
                << upper << "] mechanical pressure range=[" << -maximum(-local_minimum) << ',' << maximum(local_maximum)
                << "] Pa.";
        throw std::runtime_error(message.str());
    }
    const double target = domain_volume();
    const double tolerance = 8 * std::numeric_limits<double>::epsilon() * target;
    double p = std::clamp(s.thermodynamic_pressure, lower, upper);
    auto value = evaluate_low(s, mechanical, f, k, p);
    require_all(bool(value), "Low-Mach temperature/pressure inventory is outside its declared caloric range.");
    double residual = value->material_volume - target;
    if (std::abs(residual) <= tolerance)
    {
        assign_thermo(s, std::move(*value), p);
        return;
    }
    double a = p, fa = residual, z = p, fz = residual;
    auto best = *value;
    double best_pressure = p, best_error = std::abs(residual);
    double width = std::max(1.0, 0.1 * std::abs(p));
    bool bracketed = false;
    for (int iteration = 0; iteration < o.maximum_thermodynamic_iterations; ++iteration)
    {
        double next = std::clamp(p + (residual > 0 ? width : -width), lower, upper);
        if (next == p)
            break;
        auto trial = evaluate_low(s, mechanical, f, k, next);
        if (!trial)
        {
            width *= 0.5;
            continue;
        }
        const double error = trial->material_volume - target;
        if (std::abs(error) < best_error)
        {
            best = *trial;
            best_error = std::abs(error);
            best_pressure = next;
        }
        if (error * residual <= 0)
        {
            if (next < p)
            {
                a = next;
                fa = error;
                z = p;
                fz = residual;
            }
            else
            {
                a = p;
                fa = residual;
                z = next;
                fz = error;
            }
            bracketed = true;
            break;
        }
        p = next;
        residual = error;
        width *= 2;
    }
    require_all(bracketed, "Closed fixed-vessel material volume has no pressure root in the supplied EOS range.");
    for (int iteration = 0; iteration < o.maximum_thermodynamic_iterations && best_error > tolerance; ++iteration)
    {
        const double middle = 0.5 * (a + z);
        if (middle == a || middle == z)
            break;
        auto trial = evaluate_low(s, mechanical, f, k, middle);
        require_all(bool(trial), "Closed-vessel caloric EOS lost its valid pressure bracket.");
        const double error = trial->material_volume - target;
        if (std::abs(error) < best_error)
        {
            best = *trial;
            best_error = std::abs(error);
            best_pressure = middle;
        }
        if (error * fa > 0)
        {
            a = middle;
            fa = error;
        }
        else
        {
            z = middle;
            fz = error;
        }
    }
    require_all(best_error <= 64 * std::numeric_limits<double>::epsilon() * target,
        "Closed-vessel pressure root did not reach representable material-volume closure.");
    assign_thermo(s, std::move(best), best_pressure);
}

void AdaptiveVoFContext::normalize_low_pressure(State& s) const
{
    detail::CompensatedSum<> local;
    for (std::size_t c = 0; c < owned(); ++c)
        local += static_cast<long double>(s.p[c]) * volume[c];
    const double mean = sum(static_cast<double>(local.value())) / domain_volume();
    for (double& p : s.p)
        p -= mean;
    s.low->thermodynamic_pressure += mean;
}

std::vector<double> AdaptiveVoFContext::low_viscosity(const State& s) const
{
    std::vector<double> result(owned());
    bool valid = true;
    try
    {
        const auto& q = *s.low;
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto liquid = low_options->liquid->evaluate(q.temperature[c], q.liquid_absolute_pressure[c]);
            const double n = q.resolved_hydrogen_moles[c] + q.resolved_inert_moles[c];
            const double x = n > 0 ? q.resolved_hydrogen_moles[c] / n : 1;
            result[c] =
                q.liquid_fraction[c] * liquid.dynamic_viscosity +
                q.dispersed_fraction[c] * low_options->hydrogen.dynamic_viscosity +
                s.f[c] * (x * low_options->hydrogen.dynamic_viscosity + (1 - x) * low_options->inert.dynamic_viscosity);
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach viscosity evaluation failed collectively.");
    return result;
}

double AdaptiveVoFContext::low_conductivity(const State& state, std::size_t cell, const VoF::LiquidThermodynamicPoint& liquid) const
{
    const auto& thermo = *state.low;
    const double moles = thermo.resolved_hydrogen_moles[cell] + thermo.resolved_inert_moles[cell];
    const double hydrogen_fraction = moles > 0 ? thermo.resolved_hydrogen_moles[cell] / moles : 1;
    return thermo.liquid_fraction[cell] * liquid.thermal_conductivity +
           thermo.dispersed_fraction[cell] * low_options->hydrogen.thermal_conductivity +
           state.f[cell] * (hydrogen_fraction * low_options->hydrogen.thermal_conductivity +
                               (1 - hydrogen_fraction) * low_options->inert.thermal_conductivity);
}

double AdaptiveVoFContext::low_thermal_iteration_error(const State& candidate, const State& guess) const
{
    const auto& next = *candidate.low;
    const auto& previous = *guess.low;
    double error = 0;
    bool valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto liquid = low_options->liquid->evaluate(next.temperature[c], next.liquid_absolute_pressure[c]);
            const auto old_liquid =
                low_options->liquid->evaluate(previous.temperature[c], previous.liquid_absolute_pressure[c]);
            const double temperature_error = std::abs(next.temperature[c] - previous.temperature[c]) /
                                             std::max(next.temperature[c], previous.temperature[c]);
            double specific_energy_error = 0;
            if (next.liquid_mass[c] > 0 || previous.liquid_mass[c] > 0)
            {
                // A caloric scale avoids masking a flux error with an arbitrary
                // internal-energy reference. Ideal-gas specific energy is
                // affine in T, already controlled by temperature_error.
                const double scale = std::max(liquid.internal_energy_temperature * next.temperature[c],
                    old_liquid.internal_energy_temperature * previous.temperature[c]);
                valid = valid && std::isfinite(scale) && scale > 0;
                specific_energy_error = std::abs(liquid.internal_energy - old_liquid.internal_energy) / scale;
            }
            const double conductivity = low_conductivity(candidate, c, liquid);
            const double old_conductivity = low_conductivity(guess, c, old_liquid);
            // Use the same pure-phase scale as the explicit thermal stability
            // bound, so a numerically tiny occupied phase does not turn a
            // negligible conductivity change into an order-one residual.
            const double conductivity_scale = std::max({liquid.thermal_conductivity, old_liquid.thermal_conductivity,
                low_options->hydrogen.thermal_conductivity, low_options->inert.thermal_conductivity});
            const double conductivity_error =
                conductivity_scale > 0 ? std::abs(conductivity - old_conductivity) / conductivity_scale : 0;
            valid = valid && std::isfinite(temperature_error) && std::isfinite(specific_energy_error) &&
                    std::isfinite(conductivity) && std::isfinite(old_conductivity) && std::isfinite(conductivity_error);
            error = std::max({error, temperature_error, specific_energy_error, conductivity_error});
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach thermal fixed-point property evaluation failed collectively.");
    return maximum(error);
}

Vector AdaptiveVoFContext::population_face_drift(const Face& face, const std::vector<Vector>& drift,
    const std::vector<double>& number) const
{
    if (face.boundary)
        return drift[face.owner];
    const bool left = number[face.owner] > 0, right = number[face.neighbor] > 0;
    if (left != right)
        return drift[left ? face.owner : face.neighbor];
    Vector value{};
    for (std::size_t a = 0; a < 3; ++a)
        value[a] = (face.neighbor_distance * drift[face.owner][a] +
                       face.owner_distance * drift[face.neighbor][a]) / face.distance;
    return value;
}

std::vector<double> AdaptiveVoFContext::population_closure_number(const VoF::DispersedPopulation& accepted,
    const VoF::PopulationSource& source) const
{
    std::vector<double> number(owned());
    bool valid = true;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        number[c] = accepted.number[c] + source.number_change[c];
        const double moles = accepted.hydrogen_moles[c] + source.hydrogen_change[c];
        valid = valid && std::isfinite(number[c]) && std::isfinite(moles) && number[c] >= 0 && moles >= 0 &&
                ((number[c] == 0) == (moles == 0));
    }
    require_all(valid, "Low-Mach population closure requires paired nonnegative post-source moments.");
    return number;
}

double AdaptiveVoFContext::stable_low_timestep(const State& s) const
{
    const auto& q = *s.low;
    // Use the same distance-interpolated halo drift as the transported
    // canonical subface flux. A coarse neighbor's drift can otherwise exceed
    // the fine cell's own value while crossing its smaller cell width.
    std::vector<double> face_drift_speed(faces.size());
    for (const auto& population : q.dispersed)
    {
        const auto drift = exchange_vectors(population.drift_velocity);
        const auto number = geometry->exchange_cells(population.number);
        for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
        {
            const auto& face = faces[id];
            const auto velocity = population_face_drift(face, drift, number);
            face_drift_speed[id] = std::max(face_drift_speed[id], std::abs(dot(velocity, face.normal)));
        }
    }
    if (!q.dispersed.empty())
        face_drift_speed = geometry->exchange_faces(face_drift_speed);
    double advective = 0, diffusion = 0, capillary = 0;
    bool valid = true;
    try
    {
        with_geometry(
            [&]
            {
                for (std::size_t c = 0; c < owned(); ++c)
                {
                    double rate = 0, inverse_h2 = 0, h = std::numeric_limits<double>::infinity();
                    Vector maximum_speed{};
                    for (const auto id : mesh->faces(static_cast<LO>(c)))
                        for (std::size_t a = 0; a < 3; ++a)
                            if (std::abs(faces[id].normal[a]) > 0.5)
                                maximum_speed[a] =
                                    std::max(maximum_speed[a], std::abs(s.face_u[id]) + face_drift_speed[id]);
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        rate += maximum_speed[a] / spacing[c][a];
                        inverse_h2 += 1 / (spacing[c][a] * spacing[c][a]);
                        h = std::min(h, spacing[c][a]);
                    }
                    advective = std::max(advective, rate);
                    const auto liquid = low_options->liquid->evaluate(q.temperature[c], q.liquid_absolute_pressure[c]);
                    const double maximum_mu = std::max({liquid.dynamic_viscosity,
                        low_options->hydrogen.dynamic_viscosity, low_options->inert.dynamic_viscosity});
                    const double minimum_rho = std::min(q.liquid_density[c], q.resolved_density[c]);
                    double heat_capacity = q.liquid_mass[c] * liquid.internal_energy_temperature +
                                           q.resolved_hydrogen_moles[c] * low_options->hydrogen.molar_heat_capacity_v +
                                           q.resolved_inert_moles[c] * low_options->inert.molar_heat_capacity_v;
                    for (const auto& population : q.dispersed)
                        heat_capacity += population.hydrogen_moles[c] * low_options->hydrogen.molar_heat_capacity_v;
                    const double conductivity = std::max({liquid.thermal_conductivity,
                        low_options->hydrogen.thermal_conductivity, low_options->inert.thermal_conductivity});
                    diffusion = std::max(diffusion,
                        std::max(8 * maximum_mu / minimum_rho, 4 * conductivity * volume[c] / heat_capacity) *
                            inverse_h2);
                    if (material.surface_tension > 0)
                        capillary =
                            std::max(capillary, std::sqrt(2 * std::numbers::pi * material.surface_tension /
                                                          ((q.liquid_density[c] + q.resolved_density[c]) * h * h * h)));
                }
            });
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach timestep property evaluation failed.");
    const double a = maximum(advective), d = maximum(diffusion), c = maximum(capillary);
    double dt = std::numeric_limits<double>::infinity();
    if (a > 0)
        dt = std::min(dt, options.max_courant / a);
    if (d > 0)
        dt = std::min(dt, options.viscous_safety / d);
    if (c > 0)
        dt = std::min(dt, options.capillary_safety / c);
    return dt;
}

void AdaptiveVoFContext::budget_low(State& s) const
{
    validate_fraction(s.f);
    validate_low_inventory(*s.low);
    auto& q = *s.low;
    bool valid = q.momentum.size() == owned() && s.u.size() == owned() && s.p.size() == owned();
    double occupancy = 0, speed = 0;
    detail::CompensatedSum<> gas_volume, liquid_mass, gas_mass, kinetic, material_volume;
    if (valid)
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double unresolved_mass = low_mass(q, c), carrier_gas = q.dispersed_fraction[c] * volume[c];
            gas_volume += s.f[c] * volume[c] + carrier_gas;
            liquid_mass += q.liquid_mass[c];
            gas_mass += unresolved_mass - q.liquid_mass[c];
            const double w = (q.resolved_fraction[c] + q.dispersed_fraction[c] + q.liquid_fraction[c]) * volume[c];
            material_volume += w;
            occupancy = std::max({occupancy, std::abs(w / volume[c] - 1), std::abs(q.resolved_fraction[c] - s.f[c])});
            double base_speed = dot(s.u[c], s.u[c]);
            double dispersed_mass = 0;
            for (const auto& population : q.dispersed)
            {
                const double mass = population.hydrogen_moles[c] * low_options->hydrogen.molar_mass;
                dispersed_mass += mass;
                Vector velocity = s.u[c];
                for (std::size_t a = 0; a < 3; ++a)
                    velocity[a] += population.drift_velocity[c][a];
                kinetic += 0.5L * mass * dot(velocity, velocity);
                speed = std::max(speed, std::sqrt(dot(velocity, velocity)));
            }
            kinetic += 0.5L * (unresolved_mass - dispersed_mass) * base_speed;
            speed = std::max(speed, std::sqrt(base_speed));
            for (double v : s.u[c])
                valid = valid && std::isfinite(v);
            for (double v : q.momentum[c])
                valid = valid && std::isfinite(v);
            valid = valid && std::isfinite(s.p[c]) && std::isfinite(q.temperature[c]);
        }
    require_all(valid, "Low-Mach accepted state has invalid momentum, temperature, or storage.");
    const auto totals = low_totals(q);
    s.diagnostics.gas_volume = sum(static_cast<double>(gas_volume.value()));
    s.diagnostics.gas_mass = sum(static_cast<double>(gas_mass.value()));
    s.diagnostics.liquid_mass = sum(static_cast<double>(liquid_mass.value()));
    s.diagnostics.kinetic_energy = sum(static_cast<double>(kinetic.value()));
    s.diagnostics.momentum = totals.momentum;
    s.diagnostics.maximum_velocity = maximum(speed);
    s.face_u = checked_faces(s.face_u);
    s.diagnostics.maximum_divergence = max_abs(divergence(s.face_u));
    s.low_diagnostics.maximum_occupancy_error = maximum(occupancy);
    s.low_diagnostics.global_volume_error = sum(static_cast<double>(material_volume.value())) - domain_volume();
    s.low_diagnostics.total_mass = totals.mass;
    s.low_diagnostics.hydrogen_inventory = totals.hydrogen;
    s.low_diagnostics.inert_inventory = totals.inert;
    s.low_diagnostics.internal_energy = totals.energy;
}

} // namespace SimpleFluid::detail::adaptive_vof
