// Private compiled members of AdaptiveVoFSolver::Impl::Context.

double gas_e(const VoF::IdealGasSpecies& gas, double temperature) const
{
    return gas.reference_internal_energy + gas.molar_heat_capacity_v * (temperature - gas.reference_temperature);
}

double low_mass(const VoF::LowMachState& s, std::size_t c) const
{
    double value = s.liquid_mass[c] + low_options->hydrogen.molar_mass * s.resolved_hydrogen_moles[c] +
                   low_options->inert.molar_mass * s.resolved_inert_moles[c];
    for (const auto& population : s.dispersed)
        value += low_options->hydrogen.molar_mass * population.hydrogen_moles[c];
    return value;
}

Vector drift_momentum(const VoF::LowMachState& s, std::size_t c) const
{
    Vector value{};
    for (const auto& population : s.dispersed)
        for (std::size_t a = 0; a < 3; ++a)
            value[a] +=
                low_options->hydrogen.molar_mass * population.hydrogen_moles[c] * population.drift_velocity[c][a];
    return value;
}

Vector low_drift_momentum_l1(const VoF::LowMachState& state) const
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

double low_momentum_roundoff_factor(const State& state) const
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

void validate_low_options(const VoF::LowMachOptions& o) const
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

std::vector<double> low_values(const std::vector<double>& input, bool nonnegative, const char* message) const
{
    bool valid = input.empty() || input.size() == owned();
    for (double v : input)
        valid = valid && std::isfinite(v) && (!nonnegative || v >= 0);
    require_all(valid, message);
    return input.empty() ? std::vector<double>(owned(), 0) : input;
}

std::vector<Vector> low_vectors(const std::vector<Vector>& input, const char* message) const
{
    bool valid = input.empty() || input.size() == owned();
    for (const auto& v : input)
        for (double x : v)
            valid = valid && std::isfinite(x);
    require_all(valid, message);
    return input.empty() ? std::vector<Vector>(owned()) : input;
}

void validate_low_inventory(const VoF::LowMachState& s) const
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

struct LowTotals
{
    double mass = 0, hydrogen = 0, inert = 0, energy = 0, energy_l1 = 0;
    Vector momentum{}, momentum_l1{};
};

LowTotals low_totals(const VoF::LowMachState& s) const
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

struct ThermoEvaluation
{
    std::vector<double> temperature, rho_l, rho_r, p_l, p_r, partial, gas, liquid, dispersed;
    std::vector<std::vector<double>> bubble_volume, bubble_radius, bubble_pressure;
    double material_volume = 0;
};

std::optional<ThermoEvaluation> evaluate_low(const VoF::LowMachState& s, const std::vector<double>& mechanical,
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
                const auto radius = RadiolyticGasPhysics::solve_bubble_radius(
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

void assign_thermo(VoF::LowMachState& s, ThermoEvaluation evaluation, double p0) const
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

struct PhasePressureSusceptibility
{
    std::vector<double> gas, carrier, total; // -dV_phase/dpi at fixed extensive inventories and internal energy.
};

PhasePressureSusceptibility low_pressure_susceptibility(const VoF::LowMachState& state) const
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

double domain_volume() const
{
    detail::CompensatedSum<> local;
    for (std::size_t c = 0; c < owned(); ++c)
        local += volume[c];
    return sum(static_cast<double>(local.value()));
}

void recover_closed_thermo(VoF::LowMachState& s, const std::vector<double>& mechanical, const std::vector<double>& f,
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

void normalize_low_pressure(State& s) const
{
    detail::CompensatedSum<> local;
    for (std::size_t c = 0; c < owned(); ++c)
        local += static_cast<long double>(s.p[c]) * volume[c];
    const double mean = sum(static_cast<double>(local.value())) / domain_volume();
    for (double& p : s.p)
        p -= mean;
    s.low->thermodynamic_pressure += mean;
}

std::vector<double> low_viscosity(const State& s) const
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

double low_conductivity(const State& state, std::size_t cell, const VoF::LiquidThermodynamicPoint& liquid) const
{
    const auto& thermo = *state.low;
    const double moles = thermo.resolved_hydrogen_moles[cell] + thermo.resolved_inert_moles[cell];
    const double hydrogen_fraction = moles > 0 ? thermo.resolved_hydrogen_moles[cell] / moles : 1;
    return thermo.liquid_fraction[cell] * liquid.thermal_conductivity +
           thermo.dispersed_fraction[cell] * low_options->hydrogen.thermal_conductivity +
           state.f[cell] * (hydrogen_fraction * low_options->hydrogen.thermal_conductivity +
                               (1 - hydrogen_fraction) * low_options->inert.thermal_conductivity);
}

double low_thermal_iteration_error(const State& candidate, const State& guess) const
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

/** No fictitious radius is assigned to an empty cell. At a particle front
 * extend the populated-side evaluated closure; otherwise distance interpolate.
 * Constant-slip cell closures extend throughout the available liquid carrier.
 */
Vector population_face_drift(const Face& face, const std::vector<Vector>& drift,
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

double stable_low_timestep(const State& s) const
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

void budget_low(State& s) const
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

VoF::LowMachSources normalize_low_sources(const State& old, const VoF::LowMachSources& input) const
{
    auto result = input;
#define SF_LOW_SOURCE(name, positive)                                                                                  \
    result.name = low_values(input.name, positive, "Low-Mach source " #name " has invalid owned values.")
    SF_LOW_SOURCE(liquid_mass_change, false);
    SF_LOW_SOURCE(resolved_hydrogen_change, false);
    SF_LOW_SOURCE(resolved_inert_change, false);
    SF_LOW_SOURCE(dissolved_hydrogen_change, false);
    SF_LOW_SOURCE(donor_hydrogen_change, false);
    SF_LOW_SOURCE(net_thermal_energy, false);
    SF_LOW_SOURCE(radiolysis_energy, true);
    SF_LOW_SOURCE(enthalpy_transfer, false);
    SF_LOW_SOURCE(vent_hydrogen_moles, true);
    SF_LOW_SOURCE(vent_inert_moles, true);
    SF_LOW_SOURCE(vent_enthalpy, false);
    SF_LOW_SOURCE(produced_hydrogen, true);
    SF_LOW_SOURCE(captured_hydrogen, true);
#undef SF_LOW_SOURCE
    result.momentum_transfer = low_vectors(input.momentum_transfer, "Low-Mach internal momentum transfer is invalid.");
    result.vent_momentum = low_vectors(input.vent_momentum, "Low-Mach vent momentum is invalid.");
    const auto sigma = low_values(input.surface_tension, false, "Low-Mach surface-tension input is invalid.");
    require_all(input.surface_tension.empty() ||
                    std::all_of(sigma.begin(), sigma.end(), [&](double v) { return v == material.surface_tension; }),
        "This low-Mach baseline requires the configured constant surface tension; nonuniform sigma needs Marangoni "
        "stresses.");
    identical(static_cast<double>(input.dispersed.size()), "Low-Mach source population lists differ across ranks.");
    result.dispersed.clear();
    for (const auto& population : old.low->dispersed)
        result.dispersed.push_back(
            {population.name, std::vector<double>(owned()), std::vector<double>(owned()), population.drift_velocity});
    std::vector<std::string> names;
    for (const auto& source : input.dispersed)
    {
        identical_name(source.name);
        const auto found = std::find_if(
            result.dispersed.begin(), result.dispersed.end(), [&](const auto& p) { return p.name == source.name; });
        require_all(
            found != result.dispersed.end() && std::find(names.begin(), names.end(), source.name) == names.end(),
            "Low-Mach source refers to an unknown or repeated population.");
        names.push_back(source.name);
        found->number_change = low_values(source.number_change, false, "Low-Mach population number change is invalid.");
        found->hydrogen_change =
            low_values(source.hydrogen_change, false, "Low-Mach population hydrogen change is invalid.");
        found->drift_velocity =
            low_vectors(source.drift_velocity.empty() ? found->drift_velocity : source.drift_velocity,
                "Low-Mach population drift is invalid.");
    }
    std::array<detail::CompensatedSum<>, 10> signed_sum, absolute_sum;
    bool local_valid = true;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        double gas_change = result.resolved_hydrogen_change[c];
        double gas_change_l1 = std::abs(result.resolved_hydrogen_change[c]);
        for (const auto& p : result.dispersed)
        {
            gas_change += p.hydrogen_change[c];
            gas_change_l1 += std::abs(p.hydrogen_change[c]);
        }
        const double mass = result.liquid_mass_change[c] + low_options->hydrogen.molar_mass * gas_change +
                            low_options->inert.molar_mass * result.resolved_inert_change[c];
        const double hydrogen = gas_change + result.dissolved_hydrogen_change[c] + result.donor_hydrogen_change[c];
        const std::array<double, 10> residual{mass, hydrogen, result.resolved_inert_change[c],
            result.enthalpy_transfer[c], result.momentum_transfer[c][0], result.momentum_transfer[c][1],
            result.momentum_transfer[c][2], result.produced_hydrogen[c] + result.donor_hydrogen_change[c], 0, 0};
        for (std::size_t i = 0; i < residual.size(); ++i)
        {
            signed_sum[i] += residual[i];
            absolute_sum[i] += std::abs(residual[i]);
        }
        absolute_sum[0] += std::abs(result.liquid_mass_change[c]) + low_options->hydrogen.molar_mass * gas_change_l1 +
                           low_options->inert.molar_mass * std::abs(result.resolved_inert_change[c]);
        absolute_sum[1] +=
            gas_change_l1 + std::abs(result.donor_hydrogen_change[c]) + std::abs(result.dissolved_hydrogen_change[c]);
        absolute_sum[7] += result.produced_hydrogen[c] + std::abs(result.donor_hydrogen_change[c]);
        if (result.vent_hydrogen_moles[c] > 0 || result.vent_inert_moles[c] > 0)
            local_valid = local_valid && old.f[c] >= 1 - 2e-12;
    }
    require_all(local_valid, "Low-Mach vent withdrawal is supported only in designated fully resolved-gas cells.");
    for (std::size_t i = 0; i < 8; ++i)
    {
        const double residual = sum(static_cast<double>(signed_sum[i].value())),
                     scale = sum(static_cast<double>(absolute_sum[i].value()));
        require_all(std::abs(residual) <= low_options->inventory_relative_tolerance * std::max(scale, 1e-30),
            "Low-Mach internal source packet violates global mass, H2, inert, momentum, enthalpy, or production "
            "accounting.");
    }
    return result;
}

struct LowTransportScalar
{
    std::vector<double> amount, flux, trace;
};

LowTransportScalar transport_low_scalar(const std::vector<double>& old, const std::vector<double>& source,
    const std::vector<double>& capacity, const std::vector<double>& carrier_flux, bool positive)
{
    if (max_abs(carrier_flux) == 0)
    {
        LowTransportScalar result{old, std::vector<double>(faces.size()), std::vector<double>(faces.size())};
        bool valid = true;
        std::vector<double> concentration(owned());
        for (std::size_t c = 0; c < owned(); ++c)
        {
            result.amount[c] += source[c];
            valid = valid && std::isfinite(result.amount[c]) && (!positive || result.amount[c] >= 0) &&
                    (capacity[c] > 0 || result.amount[c] == 0);
            concentration[c] = capacity[c] > 0 ? result.amount[c] / capacity[c] : 0;
        }
        require_all(valid, "A low-Mach local source exhausts its inventory or has no geometric carrier.");
        concentration = geometry->exchange_cells(concentration);
        for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
            if (!faces[id].boundary)
                result.trace[id] = 0.5 * (concentration[faces[id].owner] + concentration[faces[id].neighbor]);
        result.trace = geometry->exchange_faces(result.trace);
        return result;
    }
    auto matrix = Teuchos::rcp(new Pack::matrix_type(mesh->owned_cell_map(), row_capacity));
    Pack::vector_type rhs(mesh->owned_cell_map(), true), solution(mesh->owned_cell_map(), true);
    bool valid = true;
    double local_diagonal_min = std::numeric_limits<double>::infinity(), local_diagonal_max = 0;
    std::string absent_carrier_detail;
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                const auto gid = mesh->cell_global_id(static_cast<LO>(c));
                double diagonal = capacity[c];
                std::map<GO, double> row;
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[id];
                    if (f.boundary)
                        continue;
                    const double outgoing = carrier_flux[id] * (f.owner == static_cast<LO>(c) ? 1 : -1);
                    if (outgoing >= 0)
                        diagonal += outgoing;
                    else
                        row[mesh->cell_global_id(f.owner == static_cast<LO>(c) ? f.neighbor : f.owner)] += outgoing;
                }
                if (diagonal == 0 && row.empty())
                {
                    valid = valid && old[c] + source[c] == 0;
                    diagonal = 1;
                }
                if (diagonal == 0 && absent_carrier_detail.empty())
                {
                    std::ostringstream message;
                    message << std::setprecision(17) << " gid=" << gid << " capacity=" << capacity[c]
                            << " accepted=" << old[c] << " source=" << source[c] << " incident phase volumes=";
                    for (const auto id : mesh->faces(static_cast<LO>(c)))
                        message << (faces[id].owner == static_cast<LO>(c) ? 1 : -1) * carrier_flux[id] << ',';
                    absent_carrier_detail = message.str();
                }
                row[gid] += diagonal;
                local_diagonal_min = std::min(local_diagonal_min, diagonal);
                local_diagonal_max = std::max(local_diagonal_max, diagonal);
                rhs.replaceLocalValue(static_cast<LO>(c), old[c] + source[c]);
                Teuchos::Array<GO> columns;
                Teuchos::Array<double> values;
                for (const auto& [column, value] : row)
                {
                    columns.push_back(column);
                    values.push_back(value);
                }
                matrix->insertGlobalValues(gid, columns(), values());
            }
        });
    require_all(valid, "A low-Mach source creates inventory without a geometric carrier or transport path.");
    if (maximum(absent_carrier_detail.empty() ? 0.0 : 1.0) > 0)
        throw std::runtime_error(
            "Low-Mach phase transport has a zero-storage incoming-only row:" + absent_carrier_detail);
    matrix->fillComplete();
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Gmres;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.tolerance = 1e-12;
    linear.max_iterations = options.max_pressure_iterations;
    linear.verbosity = 0;
    const auto statistics = inventory_solver.solve_from_zero_with_statistics(
        Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), rhs, solution, linear);
    if (!statistics.converged)
    {
        const double minimum_diagonal = -maximum(-local_diagonal_min), maximum_diagonal = maximum(local_diagonal_max);
        std::ostringstream message;
        message << std::setprecision(17) << "Low-Mach conservative phase transport did not converge: diagonal=["
                << minimum_diagonal << ',' << maximum_diagonal << "] max capacity=" << max_abs(capacity)
                << " max integrated carrier flux=" << max_abs(carrier_flux)
                << " relative residual=" << statistics.achieved_tolerance << '.';
        throw std::runtime_error(message.str());
    }
    const auto values = solution.getData();
    std::vector<double> concentration(values.begin(), values.end());
    LowTransportScalar result{
        std::vector<double>(owned()), std::vector<double>(faces.size()), std::vector<double>(faces.size())};
    for (std::size_t c = 0; c < owned(); ++c)
    {
        result.amount[c] = capacity[c] * concentration[c];
        valid = valid && std::isfinite(result.amount[c]) && (!positive || result.amount[c] >= 0);
    }
    require_all(valid, "Low-Mach phase transport lost inventory positivity or finiteness.");
    concentration = geometry->exchange_cells(concentration);
    for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
    {
        const auto& f = faces[id];
        if (f.boundary)
            continue;
        const auto donor = carrier_flux[id] >= 0 ? f.owner : f.neighbor;
        result.trace[id] =
            carrier_flux[id] == 0 ? 0.5 * (concentration[f.owner] + concentration[f.neighbor]) : concentration[donor];
        result.flux[id] = carrier_flux[id] * result.trace[id];
    }
    result.flux = geometry->exchange_faces(result.flux);
    result.trace = geometry->exchange_faces(result.trace);
    return result;
}

struct LowFluxWork
{
    VoF::LowMachTransport published;
    std::vector<double> liquid_volume, gas_volume;
    std::vector<std::vector<double>> dispersed_volume;
    std::vector<std::vector<double>> population_number_trace, population_molar_trace, relative_population_moles;
    std::vector<double> energy_rhs;
    std::vector<Vector> momentum_rhs;
};

LowFluxWork transport_low(const State& old, State& candidate, const State& guess,
    const VoF::AdaptiveAdvection& geometric, const VoF::PhaseVolumeIncrement& increment,
    const VoF::LowMachSources& source, double dt, int corrector)
{
    const auto& accepted = *old.low;
    auto& next = *candidate.low;
    const auto& thermo = *guess.low;
    LowFluxWork work;
    auto& out = work.published;
    out.resolved_volume = geometric.gas_volume;
    out.carrier_volume = geometric.liquid_volume;
    out.time_step = dt;
    std::vector<double> gas_capacity(owned()), carrier_capacity(owned());
    bool capacity_valid = true;
    std::string capacity_failure;
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                detail::CompensatedSum<> gas, carrier, residual, face_gas, face_carrier;
                // The thermodynamic inventories retain a carrier smaller than one
                // ulp beside f=1. Reconstructing it again as (1-f)*V would erase
                // real mass on the next step. These are conservative phase storage
                // amounts, never a matrix diagonal floor.
                gas += static_cast<long double>(accepted.resolved_fraction[c]) * volume[c];
                carrier += static_cast<long double>(accepted.liquid_fraction[c]) * volume[c];
                carrier += static_cast<long double>(accepted.dispersed_fraction[c]) * volume[c];
                gas += increment.gas[c];
                carrier += increment.carrier[c];
                residual += -increment.gas[c];
                residual += -increment.carrier[c];
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const double sign = faces[id].owner == static_cast<LO>(c) ? 1 : -1;
                    gas += -sign * out.resolved_volume[id];
                    carrier += -sign * out.carrier_volume[id];
                    residual += sign * out.resolved_volume[id];
                    residual += sign * out.carrier_volume[id];
                    face_gas += sign * out.resolved_volume[id];
                    face_carrier += sign * out.carrier_volume[id];
                }
                // The same frozen Weymouth-Yue residual as the geometric fraction.
                if (old.f[c] >= 0.5)
                    gas += residual.value();
                else
                    carrier += residual.value();
                gas_capacity[c] = static_cast<double>(gas.value());
                carrier_capacity[c] = static_cast<double>(carrier.value());
                const double tolerance =
                    (low_options->volume_relative_tolerance + 512 * std::numeric_limits<double>::epsilon()) * volume[c];
                const bool valid_cell = gas_capacity[c] >= 0 && carrier_capacity[c] >= 0 &&
                                        std::abs(gas_capacity[c] - candidate.f[c] * volume[c]) <= tolerance &&
                                        std::abs(carrier_capacity[c] - (1 - candidate.f[c]) * volume[c]) <= tolerance;
                capacity_valid = capacity_valid && valid_cell;
                if (!valid_cell && capacity_failure.empty())
                {
                    std::ostringstream message;
                    message << std::setprecision(std::numeric_limits<long double>::max_digits10)
                            << " corrector=" << corrector << " gid=" << mesh->cell_global_id(static_cast<LO>(c))
                            << " V=" << volume[c] << " accepted_f=" << old.f[c] << " candidate_f=" << candidate.f[c]
                            << " accepted_gas=" << static_cast<long double>(accepted.resolved_fraction[c]) * volume[c]
                            << " accepted_carrier="
                            << (static_cast<long double>(accepted.liquid_fraction[c]) +
                                   accepted.dispersed_fraction[c]) *
                                   volume[c]
                            << " increment_gas=" << increment.gas[c] << " increment_carrier=" << increment.carrier[c]
                            << " outward_gas=" << face_gas.value() << " outward_carrier=" << face_carrier.value()
                            << " frozen_WY_residual=" << residual.value() << " gas_capacity=" << gas.value()
                            << " carrier_capacity=" << carrier.value() << " native_gas=" << candidate.f[c] * volume[c]
                            << " native_carrier=" << (1 - candidate.f[c]) * volume[c]
                            << " consistency_bound=" << tolerance << " m3.";
                    capacity_failure = message.str();
                }
            }
        });
    if (maximum(capacity_valid ? 0.0 : 1.0) > 0)
        throw std::runtime_error("Low-Mach conservative phase capacities are negative or disagree with the geometric "
                                 "volume contract:" +
                                 capacity_failure);
    auto liquid = transport_low_scalar(
        accepted.liquid_mass, source.liquid_mass_change, carrier_capacity, out.carrier_volume, true);
    auto h_source = source.resolved_hydrogen_change, i_source = source.resolved_inert_change;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        h_source[c] -= source.vent_hydrogen_moles[c];
        i_source[c] -= source.vent_inert_moles[c];
    }
    auto hydrogen =
        transport_low_scalar(accepted.resolved_hydrogen_moles, h_source, gas_capacity, out.resolved_volume, true);
    auto inert = transport_low_scalar(accepted.resolved_inert_moles, i_source, gas_capacity, out.resolved_volume, true);
    auto dissolved = transport_low_scalar(accepted.dissolved_hydrogen_moles, source.dissolved_hydrogen_change,
        carrier_capacity, out.carrier_volume, true);
    auto donor = transport_low_scalar(
        accepted.donor_hydrogen_moles, source.donor_hydrogen_change, carrier_capacity, out.carrier_volume, true);
    next.liquid_mass = std::move(liquid.amount);
    next.resolved_hydrogen_moles = std::move(hydrogen.amount);
    next.resolved_inert_moles = std::move(inert.amount);
    next.dissolved_hydrogen_moles = std::move(dissolved.amount);
    next.donor_hydrogen_moles = std::move(donor.amount);
    out.liquid_mass = std::move(liquid.flux);
    out.resolved_hydrogen = std::move(hydrogen.flux);
    out.resolved_inert = std::move(inert.flux);
    out.dissolved_hydrogen = std::move(dissolved.flux);
    out.donor_hydrogen = std::move(donor.flux);
    const auto aperture = geometry->carrier_face_aperture(old.f);
    work.population_number_trace.resize(accepted.dispersed.size());
    work.population_molar_trace.resize(accepted.dispersed.size());
    work.relative_population_moles.resize(accepted.dispersed.size());
    out.dispersed_number.resize(accepted.dispersed.size());
    out.dispersed_hydrogen.resize(accepted.dispersed.size());
    out.dispersed_slip_volume.resize(accepted.dispersed.size());
    next.dispersed = accepted.dispersed;
    for (std::size_t q = 0; q < accepted.dispersed.size(); ++q)
    {
        next.dispersed[q].drift_velocity = source.dispersed[q].drift_velocity;
        const auto drift = exchange_vectors(source.dispersed[q].drift_velocity);
        auto supported_number = accepted.dispersed[q].number;
        for (std::size_t c = 0; c < owned(); ++c)
            supported_number[c] += std::max(0.0, source.dispersed[q].number_change[c]);
        const auto number_halo = geometry->exchange_cells(supported_number);
        std::vector<double> slip(faces.size()), carrier = out.carrier_volume;
        for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
        {
            const auto& f = faces[id];
            if (f.boundary)
                continue;
            const auto w = population_face_drift(f, drift, number_halo);
            slip[id] = dt * f.area * aperture[id] * dot(w, f.normal);
        }
        slip = geometry->exchange_faces(slip);
        for (std::size_t id = 0; id < faces.size(); ++id)
            carrier[id] += slip[id];
        auto number = transport_low_scalar(
            accepted.dispersed[q].number, source.dispersed[q].number_change, carrier_capacity, carrier, true);
        auto moles = transport_low_scalar(
            accepted.dispersed[q].hydrogen_moles, source.dispersed[q].hydrogen_change, carrier_capacity, carrier, true);
        next.dispersed[q].number = std::move(number.amount);
        next.dispersed[q].hydrogen_moles = std::move(moles.amount);
        out.dispersed_number[q] = std::move(number.flux);
        out.dispersed_hydrogen[q] = std::move(moles.flux);
        work.relative_population_moles[q].resize(faces.size());
        for (std::size_t id = 0; id < faces.size(); ++id)
            work.relative_population_moles[q][id] = slip[id] * moles.trace[id];
        work.population_number_trace[q] = std::move(number.trace);
        work.population_molar_trace[q] = std::move(moles.trace);
    }
    const auto temperature = geometry->exchange_cells(thermo.temperature);
    const auto velocity = exchange_vectors(old.u);
    std::vector<double> liquid_e(owned()), conductivity(owned());
    bool valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto liquid_point =
                low_options->liquid->evaluate(thermo.temperature[c], thermo.liquid_absolute_pressure[c]);
            liquid_e[c] = liquid_point.internal_energy;
            conductivity[c] = low_conductivity(guess, c, liquid_point);
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach thermal transport property evaluation failed.");
    liquid_e = geometry->exchange_cells(liquid_e);
    conductivity = geometry->exchange_cells(conductivity);
    const auto grad_t = gradients(temperature);
    std::vector<double> heat_flux(faces.size());
    out.internal_energy.assign(faces.size(), 0);
    out.momentum.assign(faces.size(), Vector{});
    std::vector<std::vector<Vector>> drift;
    for (std::size_t q = 0; q < next.dispersed.size(); ++q)
        drift.push_back(exchange_vectors(source.dispersed[q].drift_velocity));
    for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
    {
        const auto& f = faces[id];
        if (f.boundary)
            continue;
        const auto cl = out.liquid_mass[id] >= 0 ? f.owner : f.neighbor;
        const auto ch = out.resolved_hydrogen[id] >= 0 ? f.owner : f.neighbor;
        const auto ci = out.resolved_inert[id] >= 0 ? f.owner : f.neighbor;
        out.internal_energy[id] = out.liquid_mass[id] * liquid_e[cl] +
                                  out.resolved_hydrogen[id] * gas_e(low_options->hydrogen, temperature[ch]) +
                                  out.resolved_inert[id] * gas_e(low_options->inert, temperature[ci]);
        for (std::size_t a = 0; a < 3; ++a)
            out.momentum[id][a] = out.liquid_mass[id] * velocity[cl][a] +
                                  low_options->hydrogen.molar_mass * out.resolved_hydrogen[id] * velocity[ch][a] +
                                  low_options->inert.molar_mass * out.resolved_inert[id] * velocity[ci][a];
        for (std::size_t q = 0; q < next.dispersed.size(); ++q)
        {
            const auto cd = out.dispersed_hydrogen[q][id] >= 0 ? f.owner : f.neighbor;
            out.internal_energy[id] += out.dispersed_hydrogen[q][id] * gas_e(low_options->hydrogen, temperature[cd]);
            for (std::size_t a = 0; a < 3; ++a)
                out.momentum[id][a] += low_options->hydrogen.molar_mass * out.dispersed_hydrogen[q][id] *
                                       (velocity[cd][a] + drift[q][cd][a]);
        }
        const double ko = conductivity[f.owner], kn = conductivity[f.neighbor];
        const double kface = ko > 0 && kn > 0 ? f.distance / (f.owner_distance / ko + f.neighbor_distance / kn) : 0;
        const double left = temperature[f.owner] + dot(grad_t[f.owner], f.owner_tangent);
        const double right = temperature[f.neighbor] + dot(grad_t[f.neighbor], f.neighbor_tangent);
        heat_flux[id] = -dt * kface * f.area * (right - left) / f.distance;
    }
    out.internal_energy = geometry->exchange_faces(out.internal_energy);
    heat_flux = geometry->exchange_faces(heat_flux);
    out.conductive_energy = heat_flux;
    for (std::size_t a = 0; a < 3; ++a)
    {
        std::vector<double> component(faces.size());
        for (std::size_t id = 0; id < faces.size(); ++id)
            component[id] = out.momentum[id][a];
        component = geometry->exchange_faces(component);
        for (std::size_t id = 0; id < faces.size(); ++id)
            out.momentum[id][a] = component[id];
    }
    work.energy_rhs = accepted.internal_energy;
    work.momentum_rhs = accepted.momentum;
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                work.energy_rhs[c] +=
                    source.net_thermal_energy[c] + source.enthalpy_transfer[c] - source.vent_enthalpy[c];
                for (std::size_t a = 0; a < 3; ++a)
                    work.momentum_rhs[c][a] += source.momentum_transfer[c][a] - source.vent_momentum[c][a];
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const double sign = faces[id].owner == static_cast<LO>(c) ? 1 : -1;
                    work.energy_rhs[c] -= sign * (out.internal_energy[id] + heat_flux[id]);
                    for (std::size_t a = 0; a < 3; ++a)
                        work.momentum_rhs[c][a] -= sign * out.momentum[id][a];
                }
            }
        });
    return work;
}

void recover_low_volume_fluxes(const State& candidate, LowFluxWork& work) const
{
    const auto& state = *candidate.low;
    auto& flux = work.published;
    const auto temperature = geometry->exchange_cells(state.temperature);
    const auto liquid_density = geometry->exchange_cells(state.liquid_density);
    const auto gas_pressure = geometry->exchange_cells(state.resolved_absolute_pressure);
    const auto liquid_pressure = geometry->exchange_cells(state.liquid_absolute_pressure);
    work.liquid_volume.assign(faces.size(), 0);
    work.gas_volume.assign(faces.size(), 0);
    work.dispersed_volume.assign(state.dispersed.size(), std::vector<double>(faces.size()));
    flux.dispersed_slip_volume.assign(state.dispersed.size(), std::vector<double>(faces.size()));
    bool valid = true;
    try
    {
        for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
        {
            const auto& face = faces[id];
            if (face.boundary)
                continue;
            const auto liquid_donor = flux.liquid_mass[id] >= 0 ? face.owner : face.neighbor;
            const auto hydrogen_donor = flux.resolved_hydrogen[id] >= 0 ? face.owner : face.neighbor;
            const auto inert_donor = flux.resolved_inert[id] >= 0 ? face.owner : face.neighbor;
            work.liquid_volume[id] = flux.liquid_mass[id] / liquid_density[liquid_donor];
            work.gas_volume[id] = flux.resolved_hydrogen[id] * low_options->gas_constant * temperature[hydrogen_donor] /
                                      gas_pressure[hydrogen_donor] +
                                  flux.resolved_inert[id] * low_options->gas_constant * temperature[inert_donor] /
                                      gas_pressure[inert_donor];
            valid = valid && std::isfinite(work.liquid_volume[id]) && std::isfinite(work.gas_volume[id]);
            for (std::size_t population = 0; population < state.dispersed.size(); ++population)
            {
                const double number = work.population_number_trace[population][id];
                const double moles = work.population_molar_trace[population][id];
                if (!std::isfinite(number) || !std::isfinite(moles) || !(number >= 0) || !(moles >= 0) ||
                    ((number == 0) != (moles == 0)))
                    throw std::runtime_error("A transported bubble trace has unpaired or negative moments.");
                if (moles == 0)
                    continue;
                const double transported = flux.dispersed_hydrogen[population][id];
                const auto donor = transported >= 0 ? face.owner : face.neighbor;
                // At exact carrier/slip cancellation the scalar solve supplies
                // centered number/molar traces. Use the same centered thermal
                // state, retaining a nonzero relative-volume flux.
                const double T = transported == 0 ? 0.5 * (temperature[face.owner] + temperature[face.neighbor])
                                                  : temperature[donor];
                const double pl = transported == 0
                                      ? 0.5 * (liquid_pressure[face.owner] + liquid_pressure[face.neighbor])
                                      : liquid_pressure[donor];
                double pressure = pl;
                if (material.surface_tension > 0)
                {
                    const auto radius = RadiolyticGasPhysics::solve_bubble_radius(moles / number, pl,
                        material.surface_tension, low_options->gas_constant, T, low_options->minimum_bubble_radius,
                        low_options->maximum_bubble_radius, low_options->maximum_thermodynamic_iterations,
                        4 * std::numeric_limits<double>::epsilon());
                    if (!radius.converged)
                        throw std::runtime_error("Transported bubble trace has no valid Laplace/EOS radius.");
                    pressure += 2 * material.surface_tension / radius.radius;
                }
                const double molar_volume = low_options->gas_constant * T / pressure;
                work.dispersed_volume[population][id] = transported * molar_volume;
                flux.dispersed_slip_volume[population][id] =
                    work.relative_population_moles[population][id] * molar_volume;
                valid = valid && std::isfinite(work.dispersed_volume[population][id]) &&
                        std::isfinite(flux.dispersed_slip_volume[population][id]);
            }
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach candidate EOS could not reconstruct the transported phase-volume ledger.");
    work.liquid_volume = geometry->exchange_faces(work.liquid_volume);
    work.gas_volume = geometry->exchange_faces(work.gas_volume);
    for (std::size_t population = 0; population < state.dispersed.size(); ++population)
    {
        work.dispersed_volume[population] = geometry->exchange_faces(work.dispersed_volume[population]);
        flux.dispersed_slip_volume[population] = geometry->exchange_faces(flux.dispersed_slip_volume[population]);
    }
}

State initialize_low(const VoF::LowMachInitialState& initial)
{
    identical(initial.thermodynamic_pressure, "Low-Mach initial P0 must be finite and identical across ranks.");
    validate_fraction(initial.resolved_fraction);
    State s;
    s.f = initial.resolved_fraction;
    s.p.assign(owned(), 0);
    s.k = geometry->curvature(s.f);
    s.u = low_vectors(initial.velocity, "Low-Mach initial velocity is invalid.");
    s.face_u = interpolate(s.u);
    s.low.emplace();
    auto& q = *s.low;
    q.thermodynamic_pressure = initial.thermodynamic_pressure;
    q.temperature = low_values(initial.temperature, true, "Low-Mach initial temperature is invalid.");
    require_all(initial.temperature.size() == owned(), "Low-Mach initialization requires an owned temperature array.");
    const auto xh =
        low_values(initial.resolved_hydrogen_mole_fraction.empty() ? std::vector<double>(owned(), 1)
                                                                   : initial.resolved_hydrogen_mole_fraction,
            true, "Low-Mach gas composition is invalid.");
    require_all(std::all_of(xh.begin(), xh.end(), [](double x) { return x <= 1; }),
        "Resolved gas H2 mole fraction must lie in [0,1].");
    q.dissolved_hydrogen_moles = low_values(initial.dissolved_hydrogen_moles, true, "Initial dissolved H2 is invalid.");
    q.donor_hydrogen_moles = low_values(initial.donor_hydrogen_moles, true, "Initial donor H2 stock is invalid.");
    identical(static_cast<double>(initial.dispersed.size()), "Initial low-Mach populations differ across ranks.");
    std::vector<std::string> names;
    for (const auto& supplied : initial.dispersed)
    {
        identical_name(supplied.name);
        require_all(!supplied.name.empty() && std::find(names.begin(), names.end(), supplied.name) == names.end(),
            "Low-Mach population names must be nonempty and unique.");
        names.push_back(supplied.name);
        VoF::DispersedPopulation p;
        p.name = supplied.name;
        p.number = low_values(supplied.number, true, "Initial population number is invalid.");
        p.hydrogen_moles = low_values(supplied.hydrogen_moles, true, "Initial population hydrogen is invalid.");
        p.drift_velocity = low_vectors(supplied.drift_velocity, "Initial population drift is invalid.");
        p.volume.resize(owned());
        p.radius.resize(owned());
        q.dispersed.push_back(std::move(p));
    }
    for (auto* field : {&q.liquid_mass, &q.resolved_hydrogen_moles, &q.resolved_inert_moles, &q.internal_energy})
        field->resize(owned());
    q.momentum.resize(owned());
    // Obtain the mechanical capillary/gauge variation before assigning
    // thermodynamic gas moles. This does not advance physical time.
    State pressure_seed;
    pressure_seed.f = s.f;
    pressure_seed.k = s.k;
    pressure_seed.p = s.p;
    pressure_seed.u.assign(owned(), Vector{});
    pressure_seed.face_u.assign(faces.size(), 0);
    std::vector<double> reference_rho(owned());
    bool valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto liquid = low_options->liquid->evaluate(q.temperature[c], q.thermodynamic_pressure);
            const double molecular_mass =
                xh[c] * low_options->hydrogen.molar_mass + (1 - xh[c]) * low_options->inert.molar_mass;
            const double gas_density =
                molecular_mass * q.thermodynamic_pressure / (low_options->gas_constant * q.temperature[c]);
            reference_rho[c] = (1 - s.f[c]) * liquid.density + s.f[c] * gas_density;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach initial state is outside the EOS validity range.");
    if (material.surface_tension > 0)
    {
        project(pressure_seed, reference_rho, 1, std::vector<Vector>(owned()), true, BoundaryProjection::InitialFace);
        s.p = pressure_seed.p;
    }
    valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double pl = q.thermodynamic_pressure + s.p[c] - s.f[c] * material.surface_tension * s.k[c];
            const double pr = q.thermodynamic_pressure + s.p[c] + (1 - s.f[c]) * material.surface_tension * s.k[c];
            const auto liquid = low_options->liquid->evaluate(q.temperature[c], pl);
            double vd = 0, nd = 0;
            for (auto& population : q.dispersed)
            {
                if ((population.number[c] == 0) != (population.hydrogen_moles[c] == 0))
                    throw std::invalid_argument("Unpaired initial population.");
                if (population.number[c] == 0)
                    continue;
                const auto root =
                    RadiolyticGasPhysics::solve_bubble_radius(population.hydrogen_moles[c] / population.number[c], pl,
                        material.surface_tension, low_options->gas_constant, q.temperature[c],
                        low_options->minimum_bubble_radius, low_options->maximum_bubble_radius,
                        low_options->maximum_thermodynamic_iterations, 4 * std::numeric_limits<double>::epsilon());
                if (!root.converged)
                    throw std::runtime_error("Initial bubble radius did not converge.");
                population.radius[c] = root.radius;
                population.volume[c] =
                    population.number[c] * 4 * std::numbers::pi / 3 * root.radius * root.radius * root.radius;
                vd += population.volume[c];
                nd += population.hydrogen_moles[c];
            }
            const double vl = (1 - s.f[c]) * volume[c] - vd;
            if (vl < 0 || pr <= 0)
                throw std::invalid_argument("Initial low-Mach phase occupancy or pressure is invalid.");
            q.liquid_mass[c] = vl * liquid.density;
            const double nr = s.f[c] * volume[c] * pr / (low_options->gas_constant * q.temperature[c]);
            q.resolved_hydrogen_moles[c] = nr * xh[c];
            q.resolved_inert_moles[c] = nr * (1 - xh[c]);
            q.internal_energy[c] =
                q.liquid_mass[c] * liquid.internal_energy +
                (q.resolved_hydrogen_moles[c] + nd) * gas_e(low_options->hydrogen, q.temperature[c]) +
                q.resolved_inert_moles[c] * gas_e(low_options->inert, q.temperature[c]);
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach initial phase inventories cannot fill the supplied geometry.");
    validate_low_inventory(q);
    auto evaluation = evaluate_low(q, s.p, s.f, s.k, q.thermodynamic_pressure);
    require_all(bool(evaluation), "Low-Mach initial caloric/EOS reconstruction failed.");
    assign_thermo(q, std::move(*evaluation), q.thermodynamic_pressure);
    normalize_low_pressure(s);
    // Initial prescribed velocity is projected independently; retain
    // the capillary/thermodynamic pressure established above.
    const auto physical_pressure = s.p;
    project(s, density(s), 1, std::vector<Vector>(owned()), false, BoundaryProjection::InitialCell);
    s.p = physical_pressure;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        q.momentum[c] = drift_momentum(q, c);
        for (std::size_t a = 0; a < 3; ++a)
            q.momentum[c][a] += low_mass(q, c) * s.u[c][a];
    }
    budget_low(s);
    require_all(s.low_diagnostics.maximum_occupancy_error <= low_options->volume_relative_tolerance,
        "Low-Mach initial EOS state does not satisfy local material-volume closure.");
    return s;
}

State advance_low(const State& old, double dt, const VoF::LowMachSources& input, const VoF::LowMachState* hint)
{
    identical(dt, "Low-Mach timestep must be finite and identical across ranks.");
    require_all(dt > 0 && dt <= stable_low_timestep(old) * (1 + 8 * std::numeric_limits<double>::epsilon()),
        "Low-Mach timestep exceeds the transport, capillary, viscous, or thermal limit.");
    const auto source = normalize_low_sources(old, input);
    State transport_reference = old;
    for (std::size_t population = 0; population < source.dispersed.size(); ++population)
    {
        auto& reference = transport_reference.low->dispersed[population];
        reference.drift_velocity = source.dispersed[population].drift_velocity;
        for (std::size_t c = 0; c < owned(); ++c)
            reference.number[c] += std::max(0.0, source.dispersed[population].number_change[c]);
    }
    require_all(dt <= stable_low_timestep(transport_reference) * (1 + 8 * std::numeric_limits<double>::epsilon()),
        "Low-Mach timestep exceeds the advective bound for the supplied population drift.");
    const auto initial_totals = low_totals(*old.low);
    State guess = old;
    identical(hint ? 1.0 : 0.0, "Low-Mach thermodynamic guess presence differs across ranks.");
    if (hint)
    {
        validate_low_inventory(*hint);
        bool valid = hint->dispersed.size() == old.low->dispersed.size() && hint->momentum.size() == owned();
        for (const auto* field : {&hint->temperature, &hint->liquid_absolute_pressure,
                 &hint->resolved_absolute_pressure, &hint->liquid_density, &hint->resolved_density})
            valid = valid && field->size() == owned() &&
                    std::all_of(field->begin(), field->end(), [](double v) { return std::isfinite(v) && v > 0; });
        for (const auto* field : {&hint->resolved_fraction, &hint->dispersed_fraction, &hint->liquid_fraction,
                 &hint->resolved_hydrogen_partial_pressure})
            valid = valid && field->size() == owned() &&
                    std::all_of(field->begin(), field->end(), [](double v) { return std::isfinite(v) && v >= 0; });
        if (valid)
            for (std::size_t p = 0; p < hint->dispersed.size(); ++p)
            {
                const auto& population = hint->dispersed[p];
                valid = valid && population.name == old.low->dispersed[p].name;
                for (const auto* field : {&population.volume, &population.radius})
                    valid =
                        valid && field->size() == owned() &&
                        std::all_of(field->begin(), field->end(), [](double v) { return std::isfinite(v) && v >= 0; });
            }
        for (const auto& momentum : hint->momentum)
            for (double value : momentum)
                valid = valid && std::isfinite(value);
        require_all(valid, "Low-Mach thermodynamic guess has incompatible or invalid derived fields.");
        identical(hint->thermodynamic_pressure, "Low-Mach guess thermodynamic pressure differs across ranks.");
        require_all(hint->thermodynamic_pressure > 0, "Low-Mach guess thermodynamic pressure must be positive.");
        guess.low = *hint;
    }
    std::vector<double> velocity = old.face_u, pressure_work(owned(), 0);
    VoF::PhaseVolumeIncrement increment{std::vector<double>(owned()), std::vector<double>(owned())};
    const auto old_div = divergence(velocity);
    for (std::size_t c = 0; c < owned(); ++c)
    {
        increment.gas[c] = old.f[c] * old_div[c] * volume[c] * dt;
        increment.carrier[c] = (1 - old.f[c]) * old_div[c] * volume[c] * dt;
    }
    Vector viscous_traction_scale{};
    const auto viscosity = viscous_force(old, &viscous_traction_scale);
    const double total_volume = domain_volume();
    const auto& control = *low_options;
    std::array<double, 6> last_error{};
    for (int iteration = 0; iteration < control.maximum_correctors; ++iteration)
    {
        auto geometric = geometry->advect(old.f, velocity, dt, increment, old.steps % 3);
        State next = old;
        next.f = geometric.fraction;
        next.face_u = velocity;
        next.p = guess.p;
        next.steps = old.steps + 1;
        next.diagnostics.time = old.diagnostics.time + dt;
        require_all(std::isfinite(next.diagnostics.time) && next.diagnostics.time > old.diagnostics.time,
            "Low-Mach timestep does not advance finite time.");
        next.k = material.surface_tension > 0 ? geometry->curvature(next.f) : old.k;
        auto flux = transport_low(old, next, guess, geometric, increment, source, dt, iteration + 1);
        auto& thermo = *next.low;
        thermo.temperature = guess.low->temperature;
        thermo.thermodynamic_pressure = guess.low->thermodynamic_pressure;
        for (std::size_t c = 0; c < owned(); ++c)
            thermo.internal_energy[c] = flux.energy_rhs[c] + pressure_work[c];
        try
        {
            recover_closed_thermo(thermo, next.p, next.f, next.k);
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error("Low-Mach corrector " + std::to_string(iteration + 1) + ": " + error.what());
        }
        // The EOS evaluation and the gauge shift refer to this same
        // pressure field, preserving every absolute phase pressure.
        normalize_low_pressure(next);
        // Volume ledgers use the recovered candidate EOS and transported face
        // moments, never cached densities or molar volumes from a numerical hint.
        recover_low_volume_fluxes(next, flux);
        const auto eos_pressure = next.p;
        std::vector<double> qg(owned()), qc(owned()), new_work(owned()), raw_target(owned()), target(owned());
        double local_scale = 0;
        with_geometry(
            [&]
            {
                for (std::size_t c = 0; c < owned(); ++c)
                {
                    double ql = (thermo.liquid_fraction[c] - old.low->liquid_fraction[c]) * volume[c];
                    qg[c] = (thermo.resolved_fraction[c] - old.low->resolved_fraction[c]) * volume[c];
                    std::vector<double> qd(thermo.dispersed.size());
                    for (std::size_t q = 0; q < qd.size(); ++q)
                        qd[q] = thermo.dispersed[q].volume[c] - old.low->dispersed[q].volume[c];
                    double slip = 0;
                    for (const auto id : mesh->faces(static_cast<LO>(c)))
                    {
                        const double sign = faces[id].owner == static_cast<LO>(c) ? 1 : -1;
                        qg[c] += sign * flux.gas_volume[id];
                        ql += sign * flux.liquid_volume[id];
                        for (std::size_t q = 0; q < qd.size(); ++q)
                        {
                            qd[q] += sign * flux.dispersed_volume[q][id];
                            slip += sign * flux.published.dispersed_slip_volume[q][id];
                        }
                    }
                    qc[c] = ql - slip;
                    new_work[c] =
                        -0.5 * (thermo.liquid_absolute_pressure[c] + old.low->liquid_absolute_pressure[c]) * ql -
                        0.5 * (thermo.resolved_absolute_pressure[c] + old.low->resolved_absolute_pressure[c]) * qg[c];
                    for (std::size_t q = 0; q < qd.size(); ++q)
                    {
                        qc[c] += qd[q];
                        const double pb = thermo.dispersed[q].absolute_pressure[c];
                        const double old_pb =
                            old.low->dispersed[q].volume[c] > 0 ? old.low->dispersed[q].absolute_pressure[c] : pb;
                        new_work[c] -= 0.5 * (pb + old_pb) * qd[q];
                    }
                    raw_target[c] = qg[c] + qc[c];
                    local_scale += std::abs(qg[c]) + std::abs(qc[c]);
                }
            });
        detail::CompensatedSum<> compatibility_sum;
        for (double q : raw_target)
            compatibility_sum += q;
        const double compatibility = sum(static_cast<double>(compatibility_sum.value()));
        const double compatibility_bound =
            256 * std::numeric_limits<double>::epsilon() * (total_volume + sum(local_scale));
        require_all(std::abs(compatibility) <= compatibility_bound,
            "Low-Mach closed-vessel source is physically incompatible after the thermodynamic pressure solve.");
        for (std::size_t c = 0; c < owned(); ++c)
            target[c] = (raw_target[c] - compatibility * volume[c] / total_volume) / dt;
        std::vector<Vector> change(owned());
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double mass = low_mass(thermo, c);
            const auto relative = drift_momentum(thermo, c);
            thermo.momentum[c] = flux.momentum_rhs[c];
            for (std::size_t a = 0; a < 3; ++a)
            {
                thermo.momentum[c][a] += dt * (viscosity[c][a] + mass * options.gravity[a]);
                next.u[c][a] = (thermo.momentum[c][a] - relative[a]) / mass;
                change[c][a] = next.u[c][a] - old.u[c][a];
            }
        }
        next.face_u = old.face_u;
        const auto predicted = interpolate(change);
        for (std::size_t id = 0; id < faces.size(); ++id)
            next.face_u[id] += predicted[id];
        const auto susceptibility = low_pressure_susceptibility(thermo);
        project(next, density(next), dt, change, material.surface_tension > 0, BoundaryProjection::Step, &target,
            &susceptibility.total, &eos_pressure);
        std::vector<double> linearized_gas(owned()), linearized_carrier(owned());
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double pressure_change = next.p[c] - eos_pressure[c];
            const double numerical_remainder = compatibility * volume[c] / total_volume;
            // These are only the next nonlinear iterate's geometric sources.
            // The accepted inventories and final diagnostic target retain the
            // unmodified nonlinear EOS phase increments qg/qc.
            linearized_gas[c] = qg[c] - susceptibility.gas[c] * pressure_change - next.f[c] * numerical_remainder;
            linearized_carrier[c] =
                qc[c] - susceptibility.carrier[c] * pressure_change - (1 - next.f[c]) * numerical_remainder;
        }
        detail::CompensatedSum<> pressure_mean;
        for (std::size_t c = 0; c < owned(); ++c)
            pressure_mean += static_cast<long double>(next.p[c]) * volume[c];
        const double mean = sum(static_cast<double>(pressure_mean.value())) / total_volume;
        for (double& p : next.p)
            p -= mean;
        double local_error = 0, energy_change = 0, mechanical_change = 0, velocity_change = 0;
        with_geometry(
            [&]
            {
                for (std::size_t c = 0; c < owned(); ++c)
                {
                    local_error = std::max({local_error,
                        std::abs(
                            thermo.resolved_fraction[c] + thermo.liquid_fraction[c] + thermo.dispersed_fraction[c] - 1),
                        std::abs(thermo.resolved_fraction[c] - next.f[c]),
                        std::abs(qg[c] - increment.gas[c]) / volume[c],
                        std::abs(qc[c] - increment.carrier[c]) / volume[c]});
                    const double energy_scale = std::max({1.0, std::abs(thermo.internal_energy[c]),
                        std::abs(flux.energy_rhs[c]), std::abs(new_work[c])});
                    energy_change = std::max(energy_change, std::abs(new_work[c] - pressure_work[c]) / energy_scale);
                    mechanical_change = std::max(mechanical_change,
                        std::abs(next.p[c] - eos_pressure[c]) / std::max(1.0, thermo.thermodynamic_pressure));
                    const auto relative = drift_momentum(thermo, c);
                    for (std::size_t a = 0; a < 3; ++a)
                        thermo.momentum[c][a] = low_mass(thermo, c) * next.u[c][a] + relative[a];
                    for (const auto id : mesh->faces(static_cast<LO>(c)))
                        velocity_change =
                            std::max(velocity_change, dt * std::abs(next.face_u[id] - velocity[id]) /
                                                          *std::min_element(spacing[c].begin(), spacing[c].end()));
                }
            });
        local_error = maximum(local_error);
        energy_change = maximum(energy_change);
        mechanical_change = maximum(mechanical_change);
        velocity_change = maximum(velocity_change);
        const double thermal_change = low_thermal_iteration_error(next, guess);
        const auto raw_divergence = divergence(next.face_u);
        double raw_continuity_error = 0;
        for (std::size_t c = 0; c < owned(); ++c)
            raw_continuity_error =
                std::max(raw_continuity_error, std::abs(raw_divergence[c] - raw_target[c] / (dt * volume[c])));
        raw_continuity_error = maximum(raw_continuity_error);
        last_error = {
            local_error, energy_change, mechanical_change, velocity_change, raw_continuity_error, thermal_change};
        if (local_error <= control.volume_relative_tolerance && energy_change <= control.energy_relative_tolerance &&
            thermal_change <= control.energy_relative_tolerance &&
            mechanical_change <= control.volume_relative_tolerance &&
            velocity_change <= control.volume_relative_tolerance &&
            raw_continuity_error <= options.divergence_tolerance && dt * raw_continuity_error <= 1e-10)
        {
            // Publish the pressure used by the converged caloric EOS;
            // its difference from the final mechanical correction was gated above.
            next.p = eos_pressure;
            if (material.surface_tension == 0)
                next.k = geometry->curvature(next.f);
            next.low_flux = std::move(flux.published);
            next.low_flux.gas_volume_increment = qg;
            next.low_flux.carrier_volume_increment = qc;
            next.low_diagnostics = {};
            next.low_diagnostics.correctors = iteration + 1;
            next.low_diagnostics.raw_compatibility_residual = compatibility;
            next.low_diagnostics.compatibility_roundoff_correction = compatibility;
            next.low_diagnostics.maximum_continuity_residual = raw_continuity_error;
            next.diagnostics.pressure_residual = raw_continuity_error / dt;
            require_all(next.low_diagnostics.maximum_continuity_residual <= options.divergence_tolerance &&
                            dt * next.low_diagnostics.maximum_continuity_residual <= 1e-10,
                "Low-Mach projected flux fails the original unmodified physical volume target.");
            auto total = [&](const auto& values)
            {
                detail::CompensatedSum<> x;
                for (double v : values)
                    x += v;
                return sum(static_cast<double>(x.value()));
            };
            next.low_diagnostics.pressure_work = total(new_work);
            next.low_diagnostics.net_thermal_energy = total(source.net_thermal_energy);
            next.low_diagnostics.radiolysis_energy = total(source.radiolysis_energy);
            next.low_diagnostics.produced_hydrogen = total(source.produced_hydrogen);
            next.low_diagnostics.captured_hydrogen = total(source.captured_hydrogen);
            next.low_diagnostics.escaped_hydrogen = total(source.vent_hydrogen_moles);
            next.low_diagnostics.escaped_inert = total(source.vent_inert_moles);
            next.low_diagnostics.escaped_enthalpy = total(source.vent_enthalpy);
            budget_low(next);
            const auto final = low_totals(thermo);
            auto& d = next.low_diagnostics;
            d.mass_error = final.mass - initial_totals.mass + low_options->hydrogen.molar_mass * d.escaped_hydrogen +
                           low_options->inert.molar_mass * d.escaped_inert;
            d.hydrogen_error = final.hydrogen - initial_totals.hydrogen + d.escaped_hydrogen;
            d.energy_error =
                final.energy - initial_totals.energy - d.net_thermal_energy + d.escaped_enthalpy - d.pressure_work;
            require_all(std::abs(d.mass_error) <= control.inventory_relative_tolerance *
                                                      std::max(initial_totals.mass + final.mass, 1e-30) &&
                            std::abs(d.hydrogen_error) <=
                                control.inventory_relative_tolerance *
                                    std::max(initial_totals.hydrogen + final.hydrogen + d.escaped_hydrogen, 1e-30) &&
                            std::abs(final.inert - initial_totals.inert + d.escaped_inert) <=
                                control.inventory_relative_tolerance *
                                    std::max(initial_totals.inert + final.inert + d.escaped_inert, 1e-30),
                "Low-Mach accepted trial failed mass/H2/inert conservation.");
            require_all(std::abs(d.energy_error) <= 8 * control.energy_relative_tolerance *
                                                        std::max({1.0, initial_totals.energy_l1 + final.energy_l1,
                                                            std::abs(d.net_thermal_energy) + std::abs(d.pressure_work) +
                                                                std::abs(d.escaped_enthalpy)}),
                "Low-Mach accepted trial failed thermal/pressure-work conservation.");
            if (material.surface_tension == 0)
            {
                const auto initial_drift = low_drift_momentum_l1(*old.low), final_drift = low_drift_momentum_l1(thermo);
                const double roundoff_factor = low_momentum_roundoff_factor(next);
                for (std::size_t a = 0; a < 3; ++a)
                    if (geometry->hierarchy()->config().periodic[a])
                    {
                        detail::CompensatedSum<> escaped, source_scale, transported;
                        for (std::size_t c = 0; c < owned(); ++c)
                        {
                            escaped += source.vent_momentum[c][a];
                            source_scale +=
                                std::abs(source.vent_momentum[c][a]) + std::abs(source.momentum_transfer[c][a]);
                        }
                        for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
                            transported += std::abs(next.low_flux.momentum[id][a]);
                        const double vent_impulse = sum(static_cast<double>(escaped.value()));
                        const double transport_scale = sum(static_cast<double>(transported.value()));
                        const double applied_source_scale =
                            sum(static_cast<double>(source_scale.value())) + dt * viscous_traction_scale[a];
                        const double gravity_impulse = dt * final.mass * options.gravity[a];
                        const double momentum_error =
                            final.momentum[a] - initial_totals.momentum[a] + vent_impulse - gravity_impulse;
                        const double momentum_scale = initial_totals.momentum_l1[a] + final.momentum_l1[a] +
                                                      std::abs(vent_impulse) + std::abs(gravity_impulse);
                        const double roundoff_bound =
                            roundoff_factor *
                            (momentum_scale + next.low_pressure_impulse_scale[a] + 2 * transport_scale +
                                applied_source_scale + 2 * (initial_drift[a] + final_drift[a]));
                        d.momentum_error[a] = momentum_error;
                        d.momentum_roundoff_bound[a] = roundoff_bound;
                        require_all(std::isfinite(momentum_error) && std::isfinite(momentum_scale) &&
                                        std::isfinite(roundoff_bound),
                            "Low-Mach momentum ledger or roundoff bound is nonfinite.");
                        if (std::abs(momentum_error) >
                            control.inventory_relative_tolerance * std::max(momentum_scale, 1e-30) + roundoff_bound)
                        {
                            std::ostringstream message;
                            message << std::setprecision(17)
                                    << "Low-Mach periodic physical momentum gate failed: axis=" << a
                                    << " before=" << initial_totals.momentum[a] << " after=" << final.momentum[a]
                                    << " vent=" << vent_impulse << " gravity=" << gravity_impulse
                                    << " L1=" << momentum_scale
                                    << " pressure impulse scale=" << next.low_pressure_impulse_scale[a]
                                    << " transport impulse scale=" << transport_scale
                                    << " roundoff bound=" << roundoff_bound << " residual=" << momentum_error
                                    << " kg m/s.";
                            throw std::runtime_error(message.str());
                        }
                    }
            }
            next.inventories = transport_inventories(old, next, geometric);
            require_all(dt <= stable_low_timestep(next) * (1 + 8 * std::numeric_limits<double>::epsilon()),
                "Low-Mach trial material properties or velocity invalidate its explicit timestep bound.");
            return next;
        }
        const double relaxation = control.relaxation;
        for (std::size_t id = 0; id < faces.size(); ++id)
            velocity[id] = (1 - relaxation) * velocity[id] + relaxation * next.face_u[id];
        for (std::size_t c = 0; c < owned(); ++c)
        {
            increment.gas[c] = (1 - relaxation) * increment.gas[c] + relaxation * linearized_gas[c];
            increment.carrier[c] = (1 - relaxation) * increment.carrier[c] + relaxation * linearized_carrier[c];
            pressure_work[c] = (1 - relaxation) * pressure_work[c] + relaxation * new_work[c];
            // The mechanical pressure participates in the compressible EOS
            // feedback as well as the volume/energy iteration. Apply the same
            // coupled relaxation to this unknown, preserving its zero mean.
            next.p[c] = (1 - relaxation) * guess.p[c] + relaxation * next.p[c];
        }
        next.low->thermodynamic_pressure += relaxation * mean;
        guess = std::move(next);
    }
    std::ostringstream message;
    message << std::setprecision(17)
            << "Low-Mach correctors did not converge: relative occupancy/increment=" << last_error[0]
            << ", energy=" << last_error[1] << ", pressure=" << last_error[2]
            << ", integrated face-velocity change=" << last_error[3]
            << ", original continuity residual=" << last_error[4] << " 1/s"
            << ", thermal state/coefficient change=" << last_error[5] << "; accepted state preserved.";
    throw std::runtime_error(message.str());
}

State remap_low(const State& old, Context& target) const
{
    target.low_options = low_options;
    const auto& q = *old.low;
    auto quantities = old.inventories;
    std::vector<unsigned char> absent_gas(owned()), absent_carrier(owned());
    bool valid_remainder = true;
    double local_absent_gas = 0, local_absent_carrier = 0;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        const double vg = q.resolved_fraction[c] * volume[c];
        const double vc = (q.liquid_fraction[c] + q.dispersed_fraction[c]) * volume[c];
        const double bound =
            (low_options->volume_relative_tolerance + 512 * std::numeric_limits<double>::epsilon()) * volume[c];
        absent_gas[c] = old.f[c] <= 0 && vg > 0;
        absent_carrier[c] = old.f[c] >= 1 && vc > 0;
        valid_remainder = valid_remainder && (!absent_gas[c] || vg <= bound) && (!absent_carrier[c] || vc <= bound);
        local_absent_gas = std::max(local_absent_gas, static_cast<double>(absent_gas[c]));
        local_absent_carrier = std::max(local_absent_carrier, static_cast<double>(absent_carrier[c]));
    }
    require_all(valid_remainder, "Low-Mach regrid has a material phase hidden by an inconsistent geometric fraction.");
    const bool retain_gas = maximum(local_absent_gas) > 0, retain_carrier = maximum(local_absent_carrier) > 0;
    std::vector<std::pair<std::size_t, std::size_t>> phase_remainders;
    auto add = [&](std::string name, Carrier carrier, std::vector<double> amount, bool positive = false)
    {
        const auto* absent = carrier == Carrier::GasVolume && retain_gas          ? &absent_gas
                             : carrier == Carrier::LiquidVolume && retain_carrier ? &absent_carrier
                                                                                  : nullptr;
        const auto index = quantities.size();
        if (absent)
        {
            std::vector<double> remainder(owned());
            for (std::size_t c = 0; c < owned(); ++c)
                if ((*absent)[c])
                {
                    remainder[c] = amount[c];
                    amount[c] = 0;
                }
            const auto remainder_name = name + "_retained_phase";
            quantities.push_back({std::move(name), carrier, std::move(amount), positive});
            // The native fraction cannot locate this bounded retained phase
            // inside its cell. Transfer all of its paired inventories by the
            // same cell overlap, then recombine; invent no interface or sink.
            quantities.push_back({remainder_name, Carrier::BulkVolume, std::move(remainder), positive});
            phase_remainders.emplace_back(index, index + 1);
            return index;
        }
        quantities.push_back({std::move(name), carrier, std::move(amount), positive});
        return index;
    };
    const auto ml = add("__lm_liquid", Carrier::LiquidVolume, q.liquid_mass, true);
    const auto nh = add("__lm_resolved_h", Carrier::GasVolume, q.resolved_hydrogen_moles, true);
    const auto ni = add("__lm_resolved_i", Carrier::GasVolume, q.resolved_inert_moles, true);
    const auto nd = add("__lm_dissolved", Carrier::LiquidVolume, q.dissolved_hydrogen_moles, true);
    const auto nb = add("__lm_bound", Carrier::LiquidVolume, q.donor_hydrogen_moles, true);
    std::vector<double> ec(owned()), er(owned()), remainder(owned()), pl(owned()), pr(owned()), temperature(owned()),
        target_rate(owned());
    std::array<std::vector<double>, 3> mc, mr, merr;
    for (std::size_t a = 0; a < 3; ++a)
    {
        mc[a].resize(owned());
        mr[a].resize(owned());
        merr[a].resize(owned());
    }
    bool valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto liquid = low_options->liquid->evaluate(q.temperature[c], q.liquid_absolute_pressure[c]);
            ec[c] = q.liquid_mass[c] * liquid.internal_energy;
            er[c] = q.resolved_hydrogen_moles[c] * gas_e(low_options->hydrogen, q.temperature[c]) +
                    q.resolved_inert_moles[c] * gas_e(low_options->inert, q.temperature[c]);
            const double resolved_mass = low_options->hydrogen.molar_mass * q.resolved_hydrogen_moles[c] +
                                         low_options->inert.molar_mass * q.resolved_inert_moles[c];
            for (const auto& population : q.dispersed)
                ec[c] += population.hydrogen_moles[c] * gas_e(low_options->hydrogen, q.temperature[c]);
            remainder[c] = q.internal_energy[c] - ec[c] - er[c];
            const auto relative = drift_momentum(q, c);
            for (std::size_t a = 0; a < 3; ++a)
            {
                mr[a][c] = resolved_mass * old.u[c][a];
                mc[a][c] = (low_mass(q, c) - resolved_mass) * old.u[c][a] + relative[a];
                merr[a][c] = q.momentum[c][a] - mr[a][c] - mc[a][c];
            }
            pl[c] = (q.liquid_absolute_pressure[c] - q.thermodynamic_pressure) * (1 - old.f[c]) * volume[c];
            pr[c] = (q.resolved_absolute_pressure[c] - q.thermodynamic_pressure) * old.f[c] * volume[c];
            temperature[c] = q.temperature[c] * volume[c];
            if (old.low_flux.time_step > 0)
                target_rate[c] = (old.low_flux.gas_volume_increment[c] + old.low_flux.carrier_volume_increment[c]) /
                                 old.low_flux.time_step;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Low-Mach phase-energy remap property evaluation failed.");
    const auto ec_id = add("__lm_ec", Carrier::LiquidVolume, std::move(ec));
    const auto er_id = add("__lm_er", Carrier::GasVolume, std::move(er));
    const auto e_rem = add("__lm_e_roundoff", Carrier::BulkVolume, std::move(remainder));
    const auto pl_id = add("__lm_pl", Carrier::LiquidVolume, std::move(pl));
    const auto pr_id = add("__lm_pr", Carrier::GasVolume, std::move(pr));
    const auto t_id = add("__lm_t_guess", Carrier::BulkVolume, std::move(temperature));
    const auto target_id = add("__lm_target", Carrier::BulkVolume, std::move(target_rate));
    std::array<std::size_t, 3> mc_id{}, mr_id{}, me_id{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        mc_id[a] = add("__lm_mc" + std::to_string(a), Carrier::LiquidVolume, std::move(mc[a]));
        mr_id[a] = add("__lm_mr" + std::to_string(a), Carrier::GasVolume, std::move(mr[a]));
        me_id[a] = add("__lm_me" + std::to_string(a), Carrier::BulkVolume, std::move(merr[a]));
    }
    struct PopulationIndices
    {
        std::size_t number, moles;
        std::array<std::size_t, 3> drift;
    };
    std::vector<PopulationIndices> population_ids;
    for (std::size_t p = 0; p < q.dispersed.size(); ++p)
    {
        const auto& population = q.dispersed[p];
        const auto prefix = "__lm_pop" + std::to_string(p);
        PopulationIndices ids{add(prefix + "n", Carrier::LiquidVolume, population.number, true),
            add(prefix + "h", Carrier::LiquidVolume, population.hydrogen_moles, true), {}};
        for (std::size_t a = 0; a < 3; ++a)
        {
            std::vector<double> amount(owned());
            for (std::size_t c = 0; c < owned(); ++c)
                amount[c] = population.hydrogen_moles[c] * population.drift_velocity[c][a];
            ids.drift[a] = add(prefix + "w" + std::to_string(a), Carrier::LiquidVolume, std::move(amount));
        }
        population_ids.push_back(ids);
    }
    auto mapped = geometry->remap(*target.geometry, old.f, quantities, material.liquid_density, material.gas_density);
    for (const auto [main, remainder] : phase_remainders)
        for (std::size_t c = 0; c < target.owned(); ++c)
        {
            detail::CompensatedSum<> value;
            value += mapped.inventories[main].amount[c];
            value += mapped.inventories[remainder].amount[c];
            mapped.inventories[main].amount[c] = static_cast<double>(value.value());
        }
    auto values = [&](std::size_t index) -> const std::vector<double>& { return mapped.inventories[index].amount; };
    State next;
    next.f = std::move(mapped.fraction);
    next.low.emplace();
    auto& s = *next.low;
    s.thermodynamic_pressure = q.thermodynamic_pressure;
    s.liquid_mass = values(ml);
    s.resolved_hydrogen_moles = values(nh);
    s.resolved_inert_moles = values(ni);
    s.dissolved_hydrogen_moles = values(nd);
    s.donor_hydrogen_moles = values(nb);
    s.internal_energy.resize(target.owned());
    s.momentum.resize(target.owned());
    s.temperature.resize(target.owned());
    next.p.resize(target.owned());
    next.u.resize(target.owned());
    next.k = target.geometry->curvature(next.f);
    std::vector<double> history_curvature(target.owned());
    for (std::size_t c = 0; c < target.owned(); ++c)
    {
        const double vg = next.f[c] * target.volume[c], vc = (1 - next.f[c]) * target.volume[c];
        double liquid_pressure = vc > 0 ? values(pl_id)[c] / vc : 0, gas_pressure = vg > 0 ? values(pr_id)[c] / vg : 0;
        if (vc == 0)
            liquid_pressure = gas_pressure;
        if (vg == 0)
            gas_pressure = liquid_pressure;
        next.p[c] = (values(pl_id)[c] + values(pr_id)[c]) / target.volume[c];
        history_curvature[c] =
            material.surface_tension > 0 ? (gas_pressure - liquid_pressure) / material.surface_tension : 0;
        s.temperature[c] = values(t_id)[c] / target.volume[c];
        s.internal_energy[c] = values(ec_id)[c] + values(er_id)[c] + values(e_rem)[c];
        for (std::size_t a = 0; a < 3; ++a)
            s.momentum[c][a] = values(mc_id[a])[c] + values(mr_id[a])[c] + values(me_id[a])[c];
    }
    for (std::size_t p = 0; p < q.dispersed.size(); ++p)
    {
        VoF::DispersedPopulation population;
        population.name = q.dispersed[p].name;
        population.number = values(population_ids[p].number);
        population.hydrogen_moles = values(population_ids[p].moles);
        population.drift_velocity.resize(target.owned());
        for (std::size_t c = 0; c < target.owned(); ++c)
            if (population.hydrogen_moles[c] > 0)
                for (std::size_t a = 0; a < 3; ++a)
                    population.drift_velocity[c][a] =
                        values(population_ids[p].drift[a])[c] / population.hydrogen_moles[c];
        s.dispersed.push_back(std::move(population));
    }
    // Transfer one-sided pressure history. Refinement is not a virtual
    // physical capillary timestep; the next step uses new curvature.
    target.recover_closed_thermo(s, next.p, next.f, history_curvature);
    target.normalize_low_pressure(next);
    const auto pressure_history = next.p;
    for (std::size_t c = 0; c < target.owned(); ++c)
    {
        const auto relative = target.drift_momentum(s, c);
        for (std::size_t a = 0; a < 3; ++a)
            next.u[c][a] = (s.momentum[c][a] - relative[a]) / target.low_mass(s, c);
    }
    next.face_u = target.interpolate(next.u);
    auto continuity = values(target_id);
    const auto raw_continuity = continuity;
    detail::CompensatedSum<> rate_sum, rate_l1;
    for (double rate : continuity)
    {
        rate_sum += rate;
        rate_l1 += std::abs(rate);
    }
    const double correction = sum(static_cast<double>(rate_sum.value())), new_volume = target.domain_volume();
    const double history_dt = old.low_flux.time_step > 0 ? old.low_flux.time_step : 1;
    const double roundoff_bound = 256 * std::numeric_limits<double>::epsilon() *
                                  (new_volume + history_dt * sum(static_cast<double>(rate_l1.value())));
    require_all(std::abs(correction) * history_dt <= roundoff_bound,
        "Low-Mach remapped volume target has a material closed-vessel incompatibility.");
    for (std::size_t c = 0; c < target.owned(); ++c)
        continuity[c] -= correction * target.volume[c] / new_volume;
    target.project(next, target.density(next), 1, std::vector<Vector>(target.owned()), false,
        BoundaryProjection::InitialCell, &continuity);
    next.p = pressure_history;
    for (std::size_t c = 0; c < target.owned(); ++c)
    {
        const auto relative = target.drift_momentum(s, c);
        for (std::size_t a = 0; a < 3; ++a)
            s.momentum[c][a] = target.low_mass(s, c) * next.u[c][a] + relative[a];
    }
    next.steps = old.steps;
    next.diagnostics.time = old.diagnostics.time;
    next.inventories.assign(
        mapped.inventories.begin(), mapped.inventories.begin() + static_cast<std::ptrdiff_t>(old.inventories.size()));
    next.low_flux.time_step = history_dt;
    next.low_flux.gas_volume_increment.assign(target.owned(), 0);
    next.low_flux.carrier_volume_increment.resize(target.owned());
    for (std::size_t c = 0; c < target.owned(); ++c)
        next.low_flux.carrier_volume_increment[c] = raw_continuity[c] * history_dt;
    next.low_diagnostics.raw_compatibility_residual = correction * history_dt;
    next.low_diagnostics.compatibility_roundoff_correction = correction * history_dt;
    const auto actual_divergence = target.divergence(next.face_u);
    double raw_residual = 0;
    for (std::size_t c = 0; c < target.owned(); ++c)
        raw_residual = std::max(raw_residual, std::abs(actual_divergence[c] - raw_continuity[c] / target.volume[c]));
    next.low_diagnostics.maximum_continuity_residual = maximum(raw_residual);
    require_all(next.low_diagnostics.maximum_continuity_residual <= options.divergence_tolerance &&
                    history_dt * next.low_diagnostics.maximum_continuity_residual <= 1e-10,
        "Low-Mach regrid projection fails the original physical volume target.");
    target.budget_low(next);
    require_all(next.low_diagnostics.maximum_occupancy_error <= low_options->volume_relative_tolerance,
        "Thermodynamic AMR remap cannot represent the locally mixed EOS state at fixed volume; accepted mesh "
        "retained.");
    const auto before = low_totals(q), after = target.low_totals(s);
    require_all(std::abs(after.mass - before.mass) <=
                        low_options->inventory_relative_tolerance * std::max(before.mass + after.mass, 1e-30) &&
                    std::abs(after.hydrogen - before.hydrogen) <=
                        low_options->inventory_relative_tolerance * std::max(before.hydrogen + after.hydrogen, 1e-30) &&
                    std::abs(after.inert - before.inert) <=
                        low_options->inventory_relative_tolerance * std::max(before.inert + after.inert, 1e-30) &&
                    std::abs(after.energy - before.energy) <=
                        low_options->energy_relative_tolerance * std::max(before.energy_l1 + after.energy_l1, 1e-30),
        "Low-Mach AMR remap failed phase mass, species, or thermal conservation.");
    const auto before_drift = low_drift_momentum_l1(q), after_drift = target.low_drift_momentum_l1(s);
    const double momentum_roundoff_factor = target.low_momentum_roundoff_factor(next);
    for (std::size_t a = 0; a < 3; ++a)
        if (geometry->hierarchy()->config().periodic[a])
        {
            const double residual = after.momentum[a] - before.momentum[a];
            const double scale = before.momentum_l1[a] + after.momentum_l1[a];
            const double roundoff = momentum_roundoff_factor * (scale + next.low_pressure_impulse_scale[a] +
                                                                   2 * (before_drift[a] + after_drift[a]));
            next.low_diagnostics.momentum_error[a] = residual;
            next.low_diagnostics.momentum_roundoff_bound[a] = roundoff;
            require_all(
                std::isfinite(residual) && std::isfinite(roundoff) &&
                    std::abs(residual) <= low_options->inventory_relative_tolerance * std::max(scale, 1e-30) + roundoff,
                "Low-Mach AMR remap failed periodic physical momentum conservation.");
        }
    return next;
}
