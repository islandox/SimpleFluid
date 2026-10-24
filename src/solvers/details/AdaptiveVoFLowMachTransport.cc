/** @file AdaptiveVoFLowMachTransport.cc
 * @brief Conservative low-Mach sources and phase transport.
 */
#include "solvers/details/AdaptiveVoFContext.hh"
namespace SimpleFluid::detail::adaptive_vof
{
VoF::LowMachSources AdaptiveVoFContext::normalize_low_sources(const State& old, const VoF::LowMachSources& input) const
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
    SF_LOW_SOURCE(reacting_gas_energy, true);
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

AdaptiveVoFContext::LowTransportScalar AdaptiveVoFContext::transport_low_scalar(const std::vector<double>& old, const std::vector<double>& source,
    const std::vector<double>& capacity, const std::vector<double>& carrier_flux, bool positive)
{
    return conservative_transport.solve(old, source, capacity, carrier_flux, positive, options.max_pressure_iterations);
}

AdaptiveVoFContext::LowFluxWork AdaptiveVoFContext::transport_low(const State& old, State& candidate, const State& guess,
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
    // Adjacent RHS groups have exactly the same capacity and carrier flux.
    // Reuse their numerical matrix/preconditioner; sources and acceptance are
    // still evaluated separately for each physical quantity.
    auto liquid = transport_low_scalar(
        accepted.liquid_mass, source.liquid_mass_change, carrier_capacity, out.carrier_volume, true);
    auto dissolved = transport_low_scalar(accepted.dissolved_hydrogen_moles, source.dissolved_hydrogen_change,
        carrier_capacity, out.carrier_volume, true);
    auto donor = transport_low_scalar(
        accepted.donor_hydrogen_moles, source.donor_hydrogen_change, carrier_capacity, out.carrier_volume, true);
    auto h_source = source.resolved_hydrogen_change, i_source = source.resolved_inert_change;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        h_source[c] -= source.vent_hydrogen_moles[c];
        i_source[c] -= source.vent_inert_moles[c];
    }
    auto hydrogen =
        transport_low_scalar(accepted.resolved_hydrogen_moles, h_source, gas_capacity, out.resolved_volume, true);
    auto inert = transport_low_scalar(accepted.resolved_inert_moles, i_source, gas_capacity, out.resolved_volume, true);
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
        const auto supported_number = population_closure_number(accepted.dispersed[q], source.dispersed[q]);
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
        for (std::size_t q = 0; q < next.dispersed.size(); ++q)
        {
            const auto cd = out.dispersed_hydrogen[q][id] >= 0 ? f.owner : f.neighbor;
            out.internal_energy[id] += out.dispersed_hydrogen[q][id] * gas_e(low_options->hydrogen, temperature[cd]);
            for (std::size_t a = 0; a < 3; ++a)
                out.momentum[id][a] += low_options->hydrogen.molar_mass * out.dispersed_hydrogen[q][id] *
                                       drift[q][cd][a];
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
    out.momentum = geometry->exchange_face_vectors(out.momentum);
    work.energy_rhs = accepted.internal_energy;
    work.momentum_rhs = accepted.momentum;
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                work.energy_rhs[c] +=
                    source.net_thermal_energy[c] + source.enthalpy_transfer[c] - source.vent_enthalpy[c];
                const auto relative = drift_momentum(next, c);
                for (std::size_t a = 0; a < 3; ++a)
                    work.momentum_rhs[c][a] += source.momentum_transfer[c][a] - source.vent_momentum[c][a] - relative[a];
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const double sign = faces[id].owner == static_cast<LO>(c) ? 1 : -1;
                    work.energy_rhs[c] -= sign * (out.internal_energy[id] + heat_flux[id]);
                    for (std::size_t a = 0; a < 3; ++a)
                        work.momentum_rhs[c][a] -= sign * out.momentum[id][a];
                }
            }
        });
    // Species equations define the mass ledger. Transport the common velocity
    // implicitly on that exact ledger, then restore the physical drift momentum.
    // In the constant-density source-free limit this is the same M-matrix
    // transport as the incompressible solver, independent of sweep order.
    std::vector<double> mass_flux(faces.size()), capacity(owned()), zero(owned());
    for (std::size_t c = 0; c < owned(); ++c)
        capacity[c] = low_mass(next, c);
    for (std::size_t id = 0; id < faces.size(); ++id)
    {
        mass_flux[id] = out.liquid_mass[id] + low_options->hydrogen.molar_mass * out.resolved_hydrogen[id] +
                        low_options->inert.molar_mass * out.resolved_inert[id];
        for (const auto& population_flux : out.dispersed_hydrogen)
            mass_flux[id] += low_options->hydrogen.molar_mass * population_flux[id];
    }
    for (std::size_t a = 0; a < 3; ++a)
    {
        std::vector<double> rhs(owned());
        for (std::size_t c = 0; c < owned(); ++c)
            rhs[c] = work.momentum_rhs[c][a];
        const auto carried = conservative_transport.solve(rhs, zero, capacity, mass_flux, false,
            options.max_pressure_iterations, 1e-14);
        for (std::size_t id = 0; id < faces.size(); ++id)
            out.momentum[id][a] += carried.flux[id];
    }
    // Publish the conservative face-ledger update; iterative residuals never
    // become a nonconservative inventory correction.
    work.momentum_rhs = accepted.momentum;
    with_geometry([&]
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            for (std::size_t a = 0; a < 3; ++a)
                work.momentum_rhs[c][a] += source.momentum_transfer[c][a] - source.vent_momentum[c][a];
            for (const auto id : mesh->faces(static_cast<LO>(c)))
            {
                const double sign = faces[id].owner == static_cast<LO>(c) ? 1 : -1;
                for (std::size_t a = 0; a < 3; ++a)
                    work.momentum_rhs[c][a] -= sign * out.momentum[id][a];
            }
        }
    });
    return work;
}

void AdaptiveVoFContext::recover_low_volume_fluxes(const State& candidate, LowFluxWork& work) const
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
                    const auto radius = BubbleGasPhysics::solve_bubble_radius(moles / number, pl,
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

} // namespace SimpleFluid::detail::adaptive_vof
