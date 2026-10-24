/** @file AdaptiveVoFLowMach.cc
 * @brief Compiled adaptive VoF low operations.
 */
#include "solvers/details/AdaptiveVoFContext.hh"
namespace SimpleFluid::detail::adaptive_vof
{
AdaptiveVoFContext::State AdaptiveVoFContext::initialize_low(const VoF::LowMachInitialState& initial)
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
                    BubbleGasPhysics::solve_bubble_radius(population.hydrogen_moles[c] / population.number[c], pl,
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

AdaptiveVoFContext::State AdaptiveVoFContext::advance_low(const State& old, double dt, const VoF::LowMachSources& input, const VoF::LowMachState* hint)
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
        reference.number = population_closure_number(old.low->dispersed[population], source.dispersed[population]);
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
            next.low_diagnostics.reacting_gas_energy = total(source.reacting_gas_energy);
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

AdaptiveVoFContext::State AdaptiveVoFContext::remap_low(const State& old, AdaptiveVoFContext& target) const
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

} // namespace SimpleFluid::detail::adaptive_vof
