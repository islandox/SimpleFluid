// Private compiled members of AdaptiveVoFSolver::Impl::Context.
// Every sparse parcel is owned once. Only its source and destination packets
// are imported; neither a global solution nor a global parcel list is formed.
struct RepresentationPacket
{
    enum Quantity : std::size_t
    {
        Volume,
        LiquidMass,
        Hydrogen,
        Inert,
        Number,
        Dissolved,
        Donor,
        Energy,
        MomentumX,
        MomentumY,
        MomentumZ,
        PressureVolume,
        Width
    };
    using Data = std::vector<std::vector<double>>; // phase-major columns, native owned/sparse rows
    static std::size_t column(std::size_t phase, std::size_t quantity) { return phase * Width + quantity; }
};

struct RepresentationExchange
{
    Teuchos::RCP<const Pack::map_type> source, target;
    std::unique_ptr<Pack::import_type> importer;
    RepresentationExchange(Teuchos::RCP<const Pack::map_type> owned_map, const std::set<GO>& ids)
        : source(std::move(owned_map))
    {
        std::vector<GO> list(ids.begin(), ids.end());
        target = Teuchos::rcp(new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            Teuchos::arrayViewFromVector(list), GO{}, source->getComm()));
        importer = std::make_unique<Pack::import_type>(source, target);
    }
    RepresentationPacket::Data transfer(const RepresentationPacket::Data& input, bool reverse) const
    {
        const auto from_map = reverse ? target : source, to_map = reverse ? source : target;
        Pack::multi_vector_type from(from_map, input.size(), true), to(to_map, input.size(), true);
        for (std::size_t q = 0; q < input.size(); ++q)
        {
            auto values = from.getDataNonConst(q);
            std::copy(input[q].begin(), input[q].end(), values.begin());
        }
        if (reverse)
            to.doExport(from, *importer, Tpetra::ADD);
        else
            to.doImport(from, *importer, Tpetra::INSERT);
        RepresentationPacket::Data result(input.size());
        for (std::size_t q = 0; q < input.size(); ++q)
        {
            const auto values = to.getData(q);
            result[q].assign(values.begin(), values.end());
        }
        return result;
    }
};

RepresentationPacket::Data representation_packets(const State& state) const
{
    using P = RepresentationPacket;
    const auto& q = *state.low;
    const auto phases = q.dispersed.size() + 2;
    P::Data result(phases * P::Width, std::vector<double>(owned()));
    bool valid = true;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto at = [&](std::size_t phase, std::size_t quantity) -> double&
            { return result[P::column(phase, quantity)][c]; };
            const auto liquid = low_options->liquid->evaluate(q.temperature[c], q.liquid_absolute_pressure[c]);
            at(0, P::Volume) = q.liquid_fraction[c] * volume[c];
            at(0, P::LiquidMass) = q.liquid_mass[c];
            at(0, P::Dissolved) = q.dissolved_hydrogen_moles[c];
            at(0, P::Donor) = q.donor_hydrogen_moles[c];
            at(0, P::Energy) = q.liquid_mass[c] * liquid.internal_energy;
            at(0, P::PressureVolume) = q.liquid_absolute_pressure[c] * at(0, P::Volume);
            at(1, P::Volume) = q.resolved_fraction[c] * volume[c];
            at(1, P::Hydrogen) = q.resolved_hydrogen_moles[c];
            at(1, P::Inert) = q.resolved_inert_moles[c];
            at(1, P::Energy) = q.resolved_hydrogen_moles[c] * gas_e(low_options->hydrogen, q.temperature[c]) +
                               q.resolved_inert_moles[c] * gas_e(low_options->inert, q.temperature[c]);
            at(1, P::PressureVolume) = q.resolved_absolute_pressure[c] * at(1, P::Volume);
            std::vector<double> masses(phases), capacities(phases);
            masses[0] = q.liquid_mass[c];
            masses[1] =
                low_options->hydrogen.molar_mass * at(1, P::Hydrogen) + low_options->inert.molar_mass * at(1, P::Inert);
            capacities[0] = q.liquid_mass[c] * liquid.internal_energy_temperature;
            capacities[1] = at(1, P::Hydrogen) * low_options->hydrogen.molar_heat_capacity_v +
                            at(1, P::Inert) * low_options->inert.molar_heat_capacity_v;
            for (std::size_t p = 0; p < q.dispersed.size(); ++p)
            {
                const auto& population = q.dispersed[p];
                const auto phase = p + 2;
                at(phase, P::Volume) = population.volume[c];
                at(phase, P::Hydrogen) = population.hydrogen_moles[c];
                at(phase, P::Number) = population.number[c];
                at(phase, P::Energy) = population.hydrogen_moles[c] * gas_e(low_options->hydrogen, q.temperature[c]);
                at(phase, P::PressureVolume) = population.absolute_pressure[c] * population.volume[c];
                masses[phase] = low_options->hydrogen.molar_mass * population.hydrogen_moles[c];
                capacities[phase] = low_options->hydrogen.molar_heat_capacity_v * population.hydrogen_moles[c];
            }
            detail::CompensatedSum<> energy;
            std::array<detail::CompensatedSum<>, 3> momentum;
            for (std::size_t phase = 0; phase < phases; ++phase)
            {
                energy += at(phase, P::Energy);
                for (std::size_t a = 0; a < 3; ++a)
                {
                    const double velocity =
                        state.u[c][a] + (phase >= 2 ? q.dispersed[phase - 2].drift_velocity[c][a] : 0);
                    at(phase, P::MomentumX + a) = masses[phase] * velocity;
                    momentum[a] += at(phase, P::MomentumX + a);
                }
            }
            // Preserve authoritative extensive inventories, including their
            // bounded arithmetic remainder, in an actual populated phase.
            const auto thermal_phase =
                static_cast<std::size_t>(std::max_element(capacities.begin(), capacities.end()) - capacities.begin());
            const auto massive_phase =
                static_cast<std::size_t>(std::max_element(masses.begin(), masses.end()) - masses.begin());
            at(thermal_phase, P::Energy) += q.internal_energy[c] - static_cast<double>(energy.value());
            for (std::size_t a = 0; a < 3; ++a)
                at(massive_phase, P::MomentumX + a) += q.momentum[c][a] - static_cast<double>(momentum[a].value());
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Representation conversion could not construct accepted phase packets.");
    return result;
}

State change_representation(const State& old, const VoF::RepresentationChangeRequest& request,
    VoF::RepresentationDiagnostics& diagnostics) const
{
    using P = RepresentationPacket;
    const auto require_packet_state = [&](bool valid, const std::string& detail, const char* fallback)
    {
        const auto comm = mesh->owned_cell_map()->getComm();
        const int local = valid ? comm->getSize() : comm->getRank();
        int witness = comm->getSize();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &witness);
        if (witness == comm->getSize())
            return;
        std::string message = comm->getRank() == witness ? (detail.empty() ? fallback : detail) : std::string{};
        int length = static_cast<int>(message.size());
        Teuchos::broadcast(*comm, witness, 1, &length);
        message.resize(static_cast<std::size_t>(length));
        Teuchos::broadcast(*comm, witness, length, message.data());
        throw std::runtime_error(message);
    };
    const auto donor_failure =
        [&](const char* carrier, std::size_t cell, int population, double capacity, double withdrawn)
    {
        std::ostringstream message;
        message << std::setprecision(17) << "Representation " << carrier
                << " donor capacity failure: gid=" << mesh->cell_global_id(static_cast<LO>(cell))
                << ", population=" << population << ", capacity_m3=" << capacity
                << ", aggregate_withdrawn_m3=" << withdrawn << ", absolute_excess_m3=" << withdrawn - capacity
                << ", relative_excess="
                << (capacity != 0 ? (withdrawn - capacity) / capacity
                                  : (withdrawn == 0 ? 0 : std::numeric_limits<double>::infinity()))
                << ", cause="
                << (!std::isfinite(capacity) || !std::isfinite(withdrawn) ? "nonfinite capacity/debit"
                       : withdrawn < 0                                    ? "negative debit"
                                                                          : "overdraw")
                << "; population -1 denotes resolved gas, -2 denotes complete residual carrier.";
        return message.str();
    };
    const auto events = sum(static_cast<double>(request.gas_parcels.size() + request.carrier_parcels.size()));
    if (events == 0)
        return old;
    require_all(!old.low->pending_representation_relaxation,
        "Representation conversion requires the previous conversion's physical relaxation first.");
    require_all(old.inventories.empty(),
        "Representation conversion does not define destinations for generic registered passive inventories.");
    const auto populations = old.low->dispersed.size(), phases = populations + 2, columns = phases * P::Width;
    const auto source_map = mesh->owned_cell_map();
    const auto count = source_map->getGlobalNumElements();
    std::set<GO> ids;
    bool valid = true;
    const auto valid_cell = [&](std::uint64_t id)
    { return id < count && id <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max()); };
    const auto valid_population = [&](int population)
    { return population == -1 || (population >= 0 && static_cast<std::size_t>(population) < populations); };
    for (const auto& parcel : request.gas_parcels)
    {
        const bool good = valid_cell(parcel.source_cell) && valid_cell(parcel.target_cell) &&
                          valid_population(parcel.source_population) && valid_population(parcel.target_population) &&
                          ((parcel.source_population == -1) != (parcel.target_population == -1)) &&
                          std::isfinite(parcel.volume) && parcel.volume > 0 && std::isfinite(parcel.target_number) &&
                          (parcel.target_population == -1 ? parcel.target_number == 0 : parcel.target_number > 0);
        valid = valid && good;
        if (good)
        {
            ids.insert(static_cast<GO>(parcel.source_cell));
            ids.insert(static_cast<GO>(parcel.target_cell));
        }
    }
    for (const auto& parcel : request.carrier_parcels)
    {
        const bool good = valid_cell(parcel.source_cell) && valid_cell(parcel.target_cell) &&
                          std::isfinite(parcel.volume) && parcel.volume > 0;
        valid = valid && good;
        if (good)
        {
            ids.insert(static_cast<GO>(parcel.source_cell));
            ids.insert(static_cast<GO>(parcel.target_cell));
        }
    }
    require_all(
        valid, "Representation parcels have invalid global cells, populations, amounts, or conversion directions.");
    RepresentationExchange exchange(source_map, ids);
    const auto sparse_size = exchange.target->getLocalNumElements();
    auto packets = representation_packets(old);
    std::array<detail::CompensatedSum<>, 4> packet_operand_l1;
    const auto count_operands = [&](const P::Data& values)
    {
        for (std::size_t phase = 0; phase < phases; ++phase)
            for (std::size_t c = 0; c < owned(); ++c)
            {
                packet_operand_l1[0] += std::abs(values[P::column(phase, P::Energy)][c]);
                for (std::size_t a = 0; a < 3; ++a)
                    packet_operand_l1[a + 1] += std::abs(values[P::column(phase, P::MomentumX + a)][c]);
            }
    };
    count_operands(packets);
    const auto imported = exchange.transfer(packets, false);
    P::Data debit(phases, std::vector<double>(sparse_size));
    std::vector<std::vector<detail::CompensatedSum<>>> gas_debit_sum(
        phases, std::vector<detail::CompensatedSum<>>(sparse_size));
    auto gas_deposit_volume_sum = gas_debit_sum;
    P::Data deposit(columns, std::vector<double>(sparse_size));
    std::array<detail::CompensatedSum<>, 7> event_sum;
    const auto index = [&](std::uint64_t id)
    { return static_cast<std::size_t>(exchange.target->getLocalElement(static_cast<GO>(id))); };
    const auto phase_index = [](int population) { return static_cast<std::size_t>(population + 2); };
    for (const auto& parcel : request.gas_parcels)
    {
        const auto source = index(parcel.source_cell), target = index(parcel.target_cell);
        const auto from = phase_index(parcel.source_population), to = phase_index(parcel.target_population);
        const double capacity = imported[P::column(from, P::Volume)][source];
        valid = valid && capacity > 0 && (from != 1 || imported[P::column(from, P::Inert)][source] == 0);
        if (!(capacity > 0))
            continue;
        const double fraction = parcel.volume / capacity;
        gas_debit_sum[from][source] += parcel.volume;
        gas_deposit_volume_sum[to][target] += parcel.volume;
        for (std::size_t q = 0; q < P::Width; ++q)
            if (q != P::Number && q != P::Volume)
                deposit[P::column(to, q)][target] += fraction * imported[P::column(from, q)][source];
        if (to >= 2)
            deposit[P::column(to, P::Number)][target] += parcel.target_number;
        if (to == 1)
        {
            event_sum[0] += parcel.volume;
            event_sum[2] += fraction * imported[P::column(from, P::Hydrogen)][source];
            event_sum[4] += fraction * imported[P::column(from, P::Number)][source];
        }
        else
        {
            event_sum[1] += parcel.volume;
            event_sum[3] += fraction * imported[P::column(from, P::Hydrogen)][source];
            event_sum[5] += parcel.target_number;
        }
    }
    require_all(valid, "Representation gas parcels require populated donors and pure-H2 resolved donors.");
    for (std::size_t phase = 0; phase < phases; ++phase)
        for (std::size_t c = 0; c < sparse_size; ++c)
        {
            debit[phase][c] = static_cast<double>(gas_debit_sum[phase][c].value());
            deposit[P::column(phase, P::Volume)][c] = static_cast<double>(gas_deposit_volume_sum[phase][c].value());
        }
    const auto withdrawn = exchange.transfer(debit, true);
    std::string gas_capacity_failure;
    for (std::size_t c = 0; c < owned(); ++c)
        for (std::size_t phase = 1; phase < phases; ++phase)
        {
            const double capacity = packets[P::column(phase, P::Volume)][c], amount = withdrawn[phase][c];
            const bool donor_valid = amount <= capacity && amount >= 0;
            valid = valid && donor_valid;
            if (!donor_valid && gas_capacity_failure.empty())
                gas_capacity_failure = donor_failure("gas", c, static_cast<int>(phase) - 2, capacity, amount);
            if (amount <= capacity && amount > 0)
            {
                const double fraction = (capacity - amount) / capacity;
                for (std::size_t q = 0; q < P::Width; ++q)
                    packets[P::column(phase, q)][c] *= fraction;
                // Volume is the integrated parcel ledger itself, not a second
                // multiply/divide reconstruction of the retained quota.
                packets[P::column(phase, P::Volume)][c] = capacity - amount;
            }
        }
    require_packet_state(valid, gas_capacity_failure, "Representation gas parcels overdraw an accepted donor phase.");
    // Carrier packets are deliberately sampled only now, after every gas debit
    // and before any converted gas deposit, including all unselected classes.
    const auto remaining = exchange.transfer(packets, false);
    P::Data carrier_debit(1, std::vector<double>(sparse_size));
    std::vector<detail::CompensatedSum<>> carrier_debit_sum(sparse_size);
    for (const auto& parcel : request.carrier_parcels)
    {
        const auto source = index(parcel.source_cell), target = index(parcel.target_cell);
        detail::CompensatedSum<> capacity;
        for (std::size_t phase = 0; phase < phases; ++phase)
            if (phase != 1)
                capacity += remaining[P::column(phase, P::Volume)][source];
        const double source_volume = static_cast<double>(capacity.value());
        valid = valid && source_volume > 0;
        if (!(source_volume > 0))
            continue;
        carrier_debit_sum[source] += parcel.volume;
        const double fraction = parcel.volume / source_volume;
        for (std::size_t phase = 0; phase < phases; ++phase)
            if (phase != 1)
                for (std::size_t q = 0; q < P::Width; ++q)
                    deposit[P::column(phase, q)][target] += fraction * remaining[P::column(phase, q)][source];
        event_sum[6] += parcel.volume;
    }
    require_all(valid, "Representation carrier parcels require populated residual carrier donors.");
    for (std::size_t c = 0; c < sparse_size; ++c)
        carrier_debit[0][c] = static_cast<double>(carrier_debit_sum[c].value());
    const auto carrier_withdrawn = exchange.transfer(carrier_debit, true);
    const auto received = exchange.transfer(deposit, true);
    bool carrier_valid = true, packet_valid = true;
    std::string carrier_capacity_failure, packet_failure;
    constexpr std::array<const char*, P::Width> packet_names{"volume", "liquid mass", "hydrogen", "inert", "number",
        "dissolved hydrogen", "donor hydrogen", "internal energy", "x momentum", "y momentum", "z momentum",
        "pressure-volume"};
    for (std::size_t c = 0; c < owned(); ++c)
    {
        detail::CompensatedSum<> capacity;
        for (std::size_t phase = 0; phase < phases; ++phase)
            if (phase != 1)
                capacity += packets[P::column(phase, P::Volume)][c];
        const double source_volume = static_cast<double>(capacity.value()), amount = carrier_withdrawn[0][c];
        const bool donor_valid = amount >= 0 && amount <= source_volume;
        carrier_valid = carrier_valid && donor_valid;
        if (!donor_valid && carrier_capacity_failure.empty())
            carrier_capacity_failure = donor_failure("residual carrier", c, -2, source_volume, amount);
        const double fraction = amount > 0 && amount <= source_volume ? (source_volume - amount) / source_volume : 1;
        for (std::size_t phase = 0; phase < phases; ++phase)
            for (std::size_t q = 0; q < P::Width; ++q)
            {
                auto& value = packets[P::column(phase, q)][c];
                if (phase != 1)
                    value *= fraction;
                value += received[P::column(phase, q)][c];
                const bool field_valid = std::isfinite(value) && (q >= P::Energy || value >= 0);
                packet_valid = packet_valid && field_valid;
                if (!field_valid && packet_failure.empty())
                {
                    std::ostringstream message;
                    message << std::setprecision(17)
                            << "Representation packet field failure: gid=" << mesh->cell_global_id(static_cast<LO>(c))
                            << ", phase=" << phase << ", field=" << packet_names[q] << ", value=" << value
                            << ", received=" << received[P::column(phase, q)][c] << ", retained_fraction=" << fraction
                            << ", cause="
                            << (std::isfinite(value) ? "negative nonnegative inventory" : "nonfinite inventory")
                            << "; phase 0 denotes liquid, 1 resolved gas, 2+population dispersed gas.";
                    packet_failure = message.str();
                }
            }
    }
    require_packet_state(
        carrier_valid, carrier_capacity_failure, "Representation carrier parcels overdraw a residual carrier.");
    require_packet_state(packet_valid, packet_failure, "Representation transfer produced invalid packet inventories.");
    count_operands(packets);
    State next = old;
    auto& state = *next.low;
    const auto& properties = *low_options;
    const auto& bounds = properties.liquid->bounds();
    double volume_error = 0, thermal_error = 0;
    try
    {
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto at = [&](std::size_t phase, std::size_t quantity)
            { return packets[P::column(phase, quantity)][c]; };
            std::vector<double> pressures(phases), masses(phases);
            detail::CompensatedSum<> energy, occupancy;
            std::array<detail::CompensatedSum<>, 3> momentum;
            for (std::size_t phase = 0; phase < phases; ++phase)
            {
                const double phase_volume = at(phase, P::Volume);
                pressures[phase] = phase_volume > 0 ? at(phase, P::PressureVolume) / phase_volume
                                                    : (phase == 1 ? old.low->resolved_absolute_pressure[c]
                                                                  : old.low->liquid_absolute_pressure[c]);
                // The liquid evaluator's domain constrains liquid/resolved phase
                // reconstruction, as in evaluate_low. An ideal dispersed gas
                // may exceed that pressure through its valid Laplace jump.
                const bool pressure_valid = std::isfinite(pressures[phase]) && pressures[phase] > 0 &&
                                            (phase >= 2 || (pressures[phase] >= bounds.minimum_pressure &&
                                                               pressures[phase] <= bounds.maximum_pressure));
                if (!pressure_valid)
                    throw std::runtime_error("Carried phase pressure is outside its EOS domain.");
                masses[phase] = at(phase, P::LiquidMass) + properties.hydrogen.molar_mass * at(phase, P::Hydrogen) +
                                properties.inert.molar_mass * at(phase, P::Inert);
                energy += at(phase, P::Energy);
                occupancy += phase_volume;
                for (std::size_t a = 0; a < 3; ++a)
                    momentum[a] += at(phase, P::MomentumX + a);
            }
            const double target_energy = static_cast<double>(energy.value());
            const auto thermal = [&](double T)
            {
                const auto liquid = properties.liquid->evaluate(T, pressures[0]);
                detail::CompensatedSum<> value, capacity;
                value += at(0, P::LiquidMass) * liquid.internal_energy;
                capacity += at(0, P::LiquidMass) * liquid.internal_energy_temperature;
                for (std::size_t phase = 1; phase < phases; ++phase)
                {
                    value += at(phase, P::Hydrogen) * gas_e(properties.hydrogen, T) +
                             at(phase, P::Inert) * gas_e(properties.inert, T);
                    capacity += at(phase, P::Hydrogen) * properties.hydrogen.molar_heat_capacity_v +
                                at(phase, P::Inert) * properties.inert.molar_heat_capacity_v;
                }
                return std::pair{static_cast<double>(value.value()), static_cast<double>(capacity.value())};
            };
            double lower = bounds.minimum_temperature, upper = bounds.maximum_temperature;
            double T = std::clamp(old.low->temperature[c], lower, upper);
            bool converged = false;
            for (int iteration = 0; iteration < properties.maximum_thermodynamic_iterations; ++iteration)
            {
                const auto [value, capacity] = thermal(T);
                const double tolerance = 128 * std::numeric_limits<double>::epsilon() *
                                         std::max({std::abs(target_energy), std::abs(value), capacity * T, 1e-300});
                if (std::abs(value - target_energy) <= tolerance)
                {
                    converged = true;
                    break;
                }
                if (value > target_energy)
                    upper = T;
                else
                    lower = T;
                const double proposed = T + (target_energy - value) / capacity;
                T = proposed > lower && proposed < upper ? proposed : 0.5 * (lower + upper);
            }
            if (!converged)
                throw std::runtime_error("Carried phase energy has no admissible common temperature.");
            const auto liquid = properties.liquid->evaluate(T, pressures[0]);
            for (std::size_t phase = 0; phase < phases; ++phase)
            {
                const double expected_energy = phase == 0 ? masses[0] * liquid.internal_energy
                                                          : at(phase, P::Hydrogen) * gas_e(properties.hydrogen, T) +
                                                                at(phase, P::Inert) * gas_e(properties.inert, T);
                const double capacity = phase == 0
                                            ? masses[0] * liquid.internal_energy_temperature
                                            : at(phase, P::Hydrogen) * properties.hydrogen.molar_heat_capacity_v +
                                                  at(phase, P::Inert) * properties.inert.molar_heat_capacity_v;
                const double scale = std::max(capacity * T, 1e-300);
                thermal_error = std::max(thermal_error, std::abs(expected_energy - at(phase, P::Energy)) / scale);
                const double eos_volume = phase == 0 ? masses[0] / liquid.density
                                                     : (at(phase, P::Hydrogen) + at(phase, P::Inert)) *
                                                           properties.gas_constant * T / pressures[phase];
                const double phase_volume = at(phase, P::Volume);
                const double phase_scale = std::max(eos_volume, phase_volume);
                if (phase_scale > 0)
                    volume_error = std::max(volume_error, std::abs(eos_volume - phase_volume) / phase_scale);
            }
            volume_error = std::max(volume_error, std::abs(static_cast<double>(occupancy.value()) / volume[c] - 1));
            state.liquid_mass[c] = at(0, P::LiquidMass);
            state.dissolved_hydrogen_moles[c] = at(0, P::Dissolved);
            state.donor_hydrogen_moles[c] = at(0, P::Donor);
            state.resolved_hydrogen_moles[c] = at(1, P::Hydrogen);
            state.resolved_inert_moles[c] = at(1, P::Inert);
            state.internal_energy[c] = target_energy;
            state.temperature[c] = T;
            state.liquid_density[c] = liquid.density;
            state.liquid_absolute_pressure[c] = pressures[0];
            state.resolved_absolute_pressure[c] = pressures[1];
            state.resolved_fraction[c] = at(1, P::Volume) / volume[c];
            next.f[c] = state.resolved_fraction[c];
            state.liquid_fraction[c] = at(0, P::Volume) / volume[c];
            state.dispersed_fraction[c] = 0;
            const double resolved_n = at(1, P::Hydrogen) + at(1, P::Inert);
            state.resolved_density[c] = (resolved_n > 0 ? masses[1] / resolved_n : properties.hydrogen.molar_mass) *
                                        pressures[1] / (properties.gas_constant * T);
            state.resolved_hydrogen_partial_pressure[c] =
                resolved_n > 0 ? pressures[1] * at(1, P::Hydrogen) / resolved_n : 0;
            const double continuous_mass = masses[0] + masses[1];
            double total_mass = continuous_mass;
            for (std::size_t phase = 2; phase < phases; ++phase)
                total_mass += masses[phase];
            for (std::size_t a = 0; a < 3; ++a)
            {
                state.momentum[c][a] = static_cast<double>(momentum[a].value());
                next.u[c][a] = continuous_mass > 0
                                   ? (at(0, P::MomentumX + a) + at(1, P::MomentumX + a)) / continuous_mass
                                   : state.momentum[c][a] / total_mass;
            }
            for (std::size_t p = 0; p < populations; ++p)
            {
                auto& population = state.dispersed[p];
                const auto phase = p + 2;
                population.number[c] = at(phase, P::Number);
                population.hydrogen_moles[c] = at(phase, P::Hydrogen);
                population.volume[c] = at(phase, P::Volume);
                population.radius[c] =
                    population.number[c] > 0
                        ? std::cbrt(3 * population.volume[c] / (4 * std::numbers::pi * population.number[c]))
                        : 0;
                if (population.number[c] > 0 && !(population.radius[c] >= properties.minimum_bubble_radius &&
                                                    population.radius[c] <= properties.maximum_bubble_radius))
                    throw std::runtime_error("Carried population radius is outside the EOS domain.");
                population.absolute_pressure[c] = pressures[phase];
                state.dispersed_fraction[c] += population.volume[c] / volume[c];
                for (std::size_t a = 0; a < 3; ++a)
                    population.drift_velocity[c][a] =
                        masses[phase] > 0 ? at(phase, P::MomentumX + a) / masses[phase] - next.u[c][a] : 0;
            }
            bool carrier_absent = at(0, P::Volume) == 0;
            for (std::size_t phase = 2; phase < phases; ++phase)
                carrier_absent = carrier_absent && at(phase, P::Volume) == 0;
            // Classify exact material absence independently of a bounded EOS
            // occupancy remainder. Keep every actual phase volume untouched;
            // a single-gas cell must not acquire a fictitious PLIC carrier film.
            if (at(1, P::Volume) == 0)
                next.f[c] = 0;
            else if (carrier_absent)
                next.f[c] = 1;
            next.p[c] = (1 - next.f[c]) * pressures[0] + next.f[c] * pressures[1] - state.thermodynamic_pressure;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require_all(valid, "Representation phase histories have no admissible common-temperature EOS state.");
    validate_low_inventory(state);
    for (std::size_t c = 0; c < owned(); ++c)
    {
        for (double value : {state.temperature[c], state.liquid_density[c], state.resolved_density[c],
                 state.liquid_absolute_pressure[c], state.resolved_absolute_pressure[c]})
            valid = valid && std::isfinite(value) && value > 0;
        for (double value : {state.resolved_fraction[c], state.liquid_fraction[c], state.dispersed_fraction[c],
                 state.resolved_hydrogen_partial_pressure[c]})
            valid = valid && std::isfinite(value) && value >= 0;
        for (const auto& population : state.dispersed)
            valid = valid && std::isfinite(population.volume[c]) && population.volume[c] >= 0 &&
                    std::isfinite(population.radius[c]) && population.radius[c] >= 0 &&
                    std::isfinite(population.absolute_pressure[c]) && population.absolute_pressure[c] > 0;
    }
    require_all(valid, "Representation conversion produced invalid derived phase fields.");
    diagnostics.maximum_volume_error = maximum(volume_error);
    require_all(diagnostics.maximum_volume_error <= properties.volume_relative_tolerance &&
                    maximum(thermal_error) <= properties.energy_relative_tolerance,
        "Representation parcels violate local volume closure or carried phase-temperature compatibility.");
    state.pending_representation_relaxation = true;
    next.k = geometry->curvature(next.f);
    normalize_low_pressure(next);
    next.face_u = interpolate(next.u);
    next.low_flux = {};
    next.low_diagnostics = {};
    budget_low(next);
    diagnostics.maximum_face_divergence = next.diagnostics.maximum_divergence;
    const auto before = low_totals(*old.low), after = low_totals(state);
    diagnostics.mass_error = after.mass - before.mass;
    diagnostics.hydrogen_error = after.hydrogen - before.hydrogen;
    diagnostics.inert_error = after.inert - before.inert;
    diagnostics.internal_energy_error = after.energy - before.energy;
    // Bound the actual arithmetic packet path, including sparse owner sums,
    // phase recombination and compensated local/MPI budget reductions. Phase
    // operand norms retain cancellation hidden by a cell's signed total U/M.
    const double operation_count = 32 + 8 * phases + 8 * events + 4 * count + source_map->getComm()->getSize();
    const double product = operation_count * std::numeric_limits<double>::epsilon();
    require_all(product < 1, "Representation packet arithmetic exceeds its roundoff-bound domain.");
    const double gamma = product / (1 - product);
    diagnostics.internal_energy_roundoff_bound = gamma * sum(static_cast<double>(packet_operand_l1[0].value()));
    for (std::size_t a = 0; a < 3; ++a)
        diagnostics.momentum_roundoff_bound[a] = gamma * sum(static_cast<double>(packet_operand_l1[a + 1].value()));
    const auto conserved = [&](double error, double scale)
    { return std::abs(error) <= properties.inventory_relative_tolerance * std::max(scale, 1e-300); };
    require_all(conserved(diagnostics.mass_error, before.mass + after.mass) &&
                    conserved(diagnostics.hydrogen_error, before.hydrogen + after.hydrogen) &&
                    conserved(diagnostics.inert_error, before.inert + after.inert) &&
                    std::abs(diagnostics.internal_energy_error) <=
                        properties.inventory_relative_tolerance * (before.energy_l1 + after.energy_l1) +
                            diagnostics.internal_energy_roundoff_bound,
        "Representation conversion failed a mass, hydrogen, inert, or internal-energy budget.");
    for (std::size_t a = 0; a < 3; ++a)
    {
        diagnostics.momentum_error[a] = after.momentum[a] - before.momentum[a];
        require_all(std::abs(diagnostics.momentum_error[a]) <=
                        properties.inventory_relative_tolerance * (before.momentum_l1[a] + after.momentum_l1[a]) +
                            diagnostics.momentum_roundoff_bound[a],
            "Representation conversion failed its physical-momentum budget.");
    }
    const auto total = [&](const std::vector<double>& values)
    {
        detail::CompensatedSum<> local;
        for (double value : values)
            local += value;
        return sum(static_cast<double>(local.value()));
    };
    for (const auto field : {&VoF::LowMachState::liquid_mass, &VoF::LowMachState::dissolved_hydrogen_moles,
             &VoF::LowMachState::donor_hydrogen_moles})
    {
        const double a = total((*old.low).*field), b = total(state.*field);
        require_all(conserved(b - a, a + b), "Representation conversion altered liquid, dissolved, or donor stock.");
    }
    diagnostics.gas_volume_promoted = sum(static_cast<double>(event_sum[0].value()));
    diagnostics.gas_volume_demoted = sum(static_cast<double>(event_sum[1].value()));
    diagnostics.hydrogen_promoted = sum(static_cast<double>(event_sum[2].value()));
    diagnostics.hydrogen_demoted = sum(static_cast<double>(event_sum[3].value()));
    diagnostics.dispersed_number_removed = sum(static_cast<double>(event_sum[4].value()));
    diagnostics.dispersed_number_added = sum(static_cast<double>(event_sum[5].value()));
    diagnostics.carrier_volume_moved = sum(static_cast<double>(event_sum[6].value()));
    std::array<detail::CompensatedSum<>, 4> representation_totals;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        representation_totals[0] += old.low->resolved_fraction[c] * volume[c];
        representation_totals[1] += state.resolved_fraction[c] * volume[c];
        for (std::size_t p = 0; p < populations; ++p)
        {
            representation_totals[2] += old.low->dispersed[p].number[c];
            representation_totals[3] += state.dispersed[p].number[c];
        }
    }
    std::array<double, 4> global_totals;
    for (std::size_t q = 0; q < global_totals.size(); ++q)
        global_totals[q] = sum(static_cast<double>(representation_totals[q].value()));
    require_all(
        conserved(
            global_totals[1] - global_totals[0] - diagnostics.gas_volume_promoted + diagnostics.gas_volume_demoted,
            global_totals[0] + global_totals[1] + diagnostics.gas_volume_promoted + diagnostics.gas_volume_demoted) &&
            conserved(global_totals[3] - global_totals[2] + diagnostics.dispersed_number_removed -
                          diagnostics.dispersed_number_added,
                global_totals[2] + global_totals[3] + diagnostics.dispersed_number_removed +
                    diagnostics.dispersed_number_added),
        "Representation conversion failed its resolved-volume or dispersed-number parcel ledger.");
    detail::CompensatedSum<> resolved_area, dispersed_area;
    for (const auto& patch : geometry->interface_patches(next.f))
        resolved_area += patch.area;
    for (const auto& patch : geometry->interface_patches(old.f))
        resolved_area += -patch.area;
    for (std::size_t p = 0; p < populations; ++p)
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const auto& a = old.low->dispersed[p];
            const auto& b = state.dispersed[p];
            dispersed_area += 4 * std::numbers::pi *
                              (b.number[c] * b.radius[c] * b.radius[c] - a.number[c] * a.radius[c] * a.radius[c]);
        }
    diagnostics.resolved_area_change = sum(static_cast<double>(resolved_area.value()));
    diagnostics.dispersed_area_change = sum(static_cast<double>(dispersed_area.value()));
    diagnostics.surface_energy_change =
        material.surface_tension * (diagnostics.resolved_area_change + diagnostics.dispersed_area_change);
    return next;
}
