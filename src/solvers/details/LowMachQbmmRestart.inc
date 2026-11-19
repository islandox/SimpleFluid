/** @file LowMachQbmmRestart.inc
 * @brief Compiled implementation fragment for the versioned fixed-mesh QBMM restart.
 */
namespace
{
// The archive uses little-endian 64-bit words and IEEE binary64. Its dimensions
// come from the receiving mesh and schema. Only strings used in schema hashing
// have variable byte lengths. The species, phase and element layouts bound the diagnostic records.
// Versions 2 and 3 add shape residuals. Version 4 records controller provenance and physical balances.
// Version 5 adds the exact admitted transport/velocity model stamp and the
// retained legacy default-only restriction after the controller stamp. A known
// Fourier uncertainty occupies one optional trailing word before the checksums.
constexpr std::uint64_t qbmm_restart_magic = 0x5453524d4d425153ULL;
constexpr std::uint64_t qbmm_restart_version = 5;
constexpr size_t restart_word_size = sizeof(std::uint64_t);
static_assert(sizeof(double) == restart_word_size && std::numeric_limits<double>::is_iec559);

void write_word(std::vector<std::byte>& bytes, std::uint64_t value)
{
    for (size_t i = 0; i < restart_word_size; ++i)
    {
        bytes.push_back(static_cast<std::byte>(value & 0xff));
        value >>= 8;
    }
}

class RestartWriter
{
public:
    std::vector<std::byte> bytes;

    template<class... Values> void operator()(const Values&... values) { (write(values), ...); }

    void diagnostic_vector(const std::vector<double>& values, size_t)
    {
        write(static_cast<std::uint64_t>(values.size()));
        for (const double value : values)
            write(value);
    }

private:
    void write(std::uint64_t value) { write_word(bytes, value); }
    void write(int value) { write(static_cast<std::uint64_t>(value)); }
    void write(bool value) { write(static_cast<std::uint64_t>(value)); }
    void write(double value) { write(std::bit_cast<std::uint64_t>(value)); }
    void write(const Solver::Vector& value) { (*this)(value.x, value.y, value.z); }
};

class RestartReader
{
public:
    explicit RestartReader(std::span<const std::byte> bytes) : d_remaining(bytes) {}

    template<class... Values> void operator()(Values&... values) { (read(values), ...); }

    void expect(std::uint64_t expected, const char* message)
    {
        std::uint64_t value = 0;
        read(value);
        check(value == expected, message);
    }

    [[nodiscard]] size_t remaining_size() const noexcept { return d_remaining.size(); }

    void diagnostic_vector(std::vector<double>& values, size_t dimension)
    {
        std::uint64_t count = 0;
        read(count);
        check((count == 0 || count == dimension) && count <= remaining_size() / restart_word_size,
            "QBMM restart diagnostics dimensions are invalid.");
        values.resize(static_cast<size_t>(count));
        for (double& value : values)
            read(value);
    }

private:
    std::span<const std::byte> d_remaining;

    void read(std::uint64_t& value)
    {
        check(remaining_size() >= restart_word_size, "Truncated QBMM restart word.");
        value = 0;
        for (size_t i = 0; i < restart_word_size; ++i)
            value |= std::to_integer<std::uint64_t>(d_remaining[i]) << (8 * i);
        d_remaining = d_remaining.subspan(restart_word_size);
    }
    void read(int& value)
    {
        std::uint64_t word = 0;
        read(word);
        check(word <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            "QBMM restart diagnostic counter is out of range.");
        value = static_cast<int>(word);
    }
    void read(bool& value)
    {
        std::uint64_t word = 0;
        read(word);
        check(word <= 1, "Invalid QBMM restart presence flag.");
        value = word != 0;
    }
    void read(double& value)
    {
        std::uint64_t word = 0;
        read(word);
        value = std::bit_cast<double>(word);
        check(std::isfinite(value), "Nonfinite QBMM restart value.");
    }
    void read(Solver::Vector& value) { (*this)(value.x, value.y, value.z); }
};

struct RestartHash
{
    std::uint64_t value = 14695981039346656037ULL;

    void byte(unsigned char input) { value = (value ^ input) * 1099511628211ULL; }
    void word(std::uint64_t input)
    {
        for (size_t i = 0; i < restart_word_size; ++i)
        {
            byte(static_cast<unsigned char>(input & 0xff));
            input >>= 8;
        }
    }
    void number(double input) { word(std::bit_cast<std::uint64_t>(input)); }
    void vector(const Solver::Vector& input)
    {
        number(input.x);
        number(input.y);
        number(input.z);
    }
    void string(std::string_view input)
    {
        word(input.size());
        for (const unsigned char item : input)
            byte(item);
    }
};

std::uint64_t restart_checksum(std::span<const std::byte> payload)
{
    RestartHash hash;
    for (const auto byte : payload)
        hash.byte(std::to_integer<unsigned char>(byte));
    return hash.value;
}

std::uint64_t restart_partition_fingerprint(const Solver::Mesh& mesh)
{
    const auto lease = mesh.visit(
        [](const auto& native) -> std::optional<GeometryExecutionGuard::ReadLease>
        {
            if constexpr (requires { native.acquire_geometry_read(); })
                return native.acquire_geometry_read();
            return {};
        });
    const auto comm = mesh.owned_cell_map()->getComm();
    RestartHash hash;
    hash.word(static_cast<std::uint64_t>(comm->getSize()));
    hash.word(static_cast<std::uint64_t>(comm->getRank()));
    hash.word(mesh.owned_cell_map()->getGlobalNumElements());
    hash.word(mesh.owned_face_map()->getGlobalNumElements());
    hash.word(mesh.num_owned_cells());
    hash.word(mesh.num_local_cells());
    hash.word(mesh.num_owned_faces());
    hash.word(mesh.num_faces());
    hash.word(mesh.variant().index());
    mesh.visit(
        [&](const auto& native)
        {
            if constexpr (requires {
                              native.cell_edges()[0];
                              native.indexer().periodic_dimensions[0];
                          })
            {
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    hash.word(native.indexer().periodic_dimensions[axis]);
                    const auto& edges = native.cell_edges()[axis];
                    hash.word(edges.size());
                    for (const double edge : edges)
                        hash.number(edge);
                }
            }
            else if constexpr (requires {
                                   native.xy_nodes();
                                   native.xy_cell_nodes();
                                   native.z_edges();
                               })
            {
                hash.word(native.indexer().axial_periodic);
                hash.word(native.xy_nodes().size());
                for (const auto& node : native.xy_nodes())
                    hash.vector(node);
                hash.word(native.xy_cell_nodes().size());
                for (const auto& nodes : native.xy_cell_nodes())
                {
                    hash.word(nodes.size());
                    for (const auto node : nodes)
                        hash.word(node);
                }
                hash.word(native.z_edges().size());
                for (const double edge : native.z_edges())
                    hash.number(edge);
            }
        });
    for (size_t c = 0; c < mesh.num_local_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        hash.word(static_cast<std::uint64_t>(mesh.cell_global_id(cell)));
        hash.word(static_cast<std::uint64_t>(mesh.cell_geometry_global_id(cell)));
        hash.number(mesh.cell_volume(cell));
        hash.vector(mesh.cell_centroid(cell));
        mesh.visit_cell_faces(cell,
            [&](LO face, auto native_face, auto, const auto& native)
            {
                hash.word(static_cast<std::uint64_t>(face));
                if constexpr (requires { native.face_nodes(native_face); })
                {
                    const auto& nodes = native.face_nodes(native_face);
                    hash.word(nodes.size());
                    for (const auto node : nodes)
                    {
                        hash.word(static_cast<std::uint64_t>(native.node_local_id(node)));
                        hash.vector(native.node_coordinates(node));
                    }
                }
            });
        hash.word(std::numeric_limits<std::uint64_t>::max()); // End of ordered incidence row.
    }
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const auto owner = mesh.owner_cell(face);
        hash.word(static_cast<std::uint64_t>(mesh.face_global_id(face)));
        hash.word(static_cast<std::uint64_t>(mesh.face_geometry_global_id(face)));
        hash.word(static_cast<std::uint64_t>(owner));
        // Native periodic links are represented by the mapped neighbor.
        hash.word(static_cast<std::uint64_t>(mesh.neighbor_cell(face)));
        hash.vector(mesh.face_centroid(face));
        hash.vector(mesh.face_area_vector(face));
        hash.word(mesh.is_geometry_exterior_face(face));
        hash.word(mesh.is_boundary_face(face));
        if (mesh.is_boundary_face(face))
            hash.word(static_cast<std::uint64_t>(mesh.boundary_id(face)));
    }
    return hash.value;
}

std::uint64_t restart_schema_fingerprint(
    const Solver::Options& options, std::span<const std::shared_ptr<const MixtureThermodynamics>> eos)
{
    RestartHash hash;
    hash.word(eos.size());
    hash.word(static_cast<std::uint64_t>(options.thermal_mode));
    for (size_t phase = 0; phase < eos.size(); ++phase)
    {
        const auto& role = options.phase_roles[phase];
        const auto& provider = *eos[phase];
        const auto& bounds = provider.bounds();
        hash.word(static_cast<std::uint64_t>(role.role));
        hash.string(role.identity);
        hash.word(static_cast<std::uint64_t>(provider.kind()));
        hash.word(static_cast<std::uint64_t>(provider.energy_convention()));
        hash.word(static_cast<std::uint64_t>(provider.enthalpy_pressure_dependence()));
        hash.number(bounds.minimum_temperature);
        hash.number(bounds.maximum_temperature);
        hash.number(bounds.minimum_pressure);
        hash.number(bounds.maximum_pressure);
    }
    return hash.value;
}

std::uint64_t restart_model_code(const Solver::Options& options)
{
    const auto& model = options.qbmm_model;
    check(model.transport.mode == FVM::AssemblyMode::Transient &&
              model.transport.time == FVM::ScalarTimeScheme::BackwardEuler &&
              model.transport.convection == FVM::ScalarConvectionScheme::Upwind && !model.variable_composition &&
              !model.cross_family_coalescence && !model.bulk_gas_conversion && !model.capillarity &&
              !model.independent_node_temperatures,
        "QBMM restart requires the admitted fixed-composition backward-Euler upwind model.");
    // Explicit wire codes: enum declaration order is not an archive contract.
    switch (model.velocity)
    {
        case QbmmModelOptions::Velocity::SharedCarrier:
            return 0;
        case QbmmModelOptions::Velocity::SuppliedPhaseSlip:
            return 1;
        default:
            throw std::invalid_argument("QBMM restart has an unsupported velocity model.");
    }
}

void validate_legacy_restart_model(const Solver::Options& options)
{
    check(restart_model_code(options) == 0, "QBMM restart versions 1-4 require the zero-slip shared-carrier model.");
    for (const auto& family : options.qbmm_families)
        check(!family || family->closure_versions().laws.empty(),
            "QBMM restart versions 1-4 have no supplied constitutive-law metadata.");
}

void require_uniform_restart_configuration(const Solver::Mesh& mesh, const Solver::Options& options,
    std::span<const std::shared_ptr<const MixtureThermodynamics>> eos, const Solver::State& state,
    std::optional<std::uint64_t> controller, std::uint64_t model, bool legacy_default_only = false)
{
    // This is schema-only work, including on owners without cells. Drift is
    // deliberately excluded: SuppliedPhaseSlip is a per-cell canonical field.
    RestartHash hash;
    hash.word(restart_schema_fingerprint(options, eos));
    hash.word(eos.front()->catalog().fingerprint());
    hash.word(model);
    hash.word(legacy_default_only);
    hash.word(state.qbmm_families.size());
    for (const auto& stamp : state.qbmm_families)
    {
        hash.word(stamp.has_value());
        if (stamp)
        {
            hash.word(stamp->catalog_fingerprint);
            hash.word(stamp->family_fingerprint);
            hash.number(stamp->mass_scale_kg);
        }
    }
    collective_detail::require_uniform_value(mesh, hash.value, "QBMM restart model and family schema");
    collective_detail::require_uniform_value(mesh, int(controller.has_value()), "QBMM restart controller presence");
    collective_detail::require_uniform_value(mesh, controller.value_or(0), "QBMM restart controller signature");
}

template<class Archive, class Diagnostics>
void restart_diagnostics(Archive& io, Diagnostics& value, size_t phases, size_t species, std::uint64_t version)
{
    io(value.correctors, value.source_evaluations, value.maximum_source_residual, value.maximum_transport_residual,
        value.maximum_thermal_residual, value.maximum_event_residual, value.maximum_conditioned_thermal_residual,
        value.conditioned_raw_energy_roundoff_bound, value.maximum_occupancy_error, value.maximum_continuity_residual,
        value.mass_error, value.internal_energy_error, value.pressure_work, value.caloric_energy_scale,
        value.internal_energy_roundoff_bound, value.empty_phase_energy_roundoff, value.momentum_error,
        value.pressure_impulse, value.external_impulse);
    io.diagnostic_vector(value.species_errors, species);
    io.diagnostic_vector(value.population_number_errors, phases);
    io(value.maximum_population_transport_residual, value.maximum_initial_transport_residual,
        value.maximum_species_transport_residual, value.maximum_energy_transport_residual,
        value.transport_consistency_refinements, value.transport_linear.converged, value.transport_linear.solves,
        value.transport_linear.iterations, value.transport_linear.achieved_tolerance, value.momentum_linear.converged,
        value.momentum_linear.solves, value.momentum_linear.iterations, value.momentum_linear.achieved_tolerance,
        value.pressure_linear.converged, value.pressure_linear.iterations, value.pressure_linear.achieved_tolerance,
        value.pressure_linear.rhs_norm, value.pressure_linear_rhs_norm_floor);
    if (version >= 2)
        io(value.maximum_shape_source_residual);
    if (version >= 3)
        io(value.maximum_shape_transport_residual);
}

void validate_restart_fourier_uncertainty(const Solver::Diagnostics& statistics)
{
    if (const auto value = statistics.maximum_fourier_gross_comparison_uncertainty_j)
        check(statistics.source_evaluations >= 2 && statistics.source_evaluations == statistics.correctors &&
                  std::isfinite(*value) && *value >= 0,
            "QBMM restart Fourier gross comparison uncertainty needs at least two source corrector evaluations "
            "and a finite, nonnegative value.");
}

void require_uniform_restart_fourier_uncertainty(const Solver::Mesh& mesh, const Solver::Diagnostics& statistics)
{
    const auto uncertainty = statistics.maximum_fourier_gross_comparison_uncertainty_j;
    collective_detail::require_uniform_value(
        mesh, uncertainty.value_or(-1.), "QBMM restart Fourier gross comparison uncertainty");
    if (uncertainty)
        collective_detail::require_uniform_value(
            mesh, statistics.source_evaluations, "QBMM restart Fourier source evaluation count");
}

template<class Archive>
void restart_qbmm_balances(Archive& io, std::optional<Solver::QbmmBalances>& balances,
    const Solver::Diagnostics& statistics, const SpeciesCatalog& catalog)
{
    constexpr bool reading = std::is_same_v<Archive, RestartReader>;
    using Basis = Solver::QbmmMaterialBalanceBasis;
    bool present = balances.has_value();
    io(present);
    check(present == (statistics.correctors > 0), "QBMM restart balance presence differs from solver diagnostics.");
    if (!present)
        return;
    if constexpr (reading)
        balances = qbmm_balance_diagnostics(statistics);
    auto& value = balances.value();
    auto basis = static_cast<std::uint64_t>(value.material_basis);
    io(basis, value.mass_defect_kg);
    check(basis <= static_cast<std::uint64_t>(Basis::ArchivedNativeDefect), "Invalid QBMM restart balance basis.");
    value.material_basis = static_cast<Basis>(basis);
    const auto species = catalog.species().size();
    io.diagnostic_vector(value.external_species_moles, species);
    const bool legacy = value.material_basis == Basis::ArchivedNativeDefect;
    check(
        value.external_species_moles.size() == (legacy ? 0 : species), "Invalid QBMM restart external balance layout.");
    if (legacy)
        return;
    if constexpr (reading)
        value.element_defects_mol = qbmm_declared_elements(catalog);
    for (auto& entry : value.element_defects_mol)
        io(entry.second);
}

void validate_restart_physics(const Solver::Mesh& mesh, const Solver::Options& options, const Solver::State& state,
    std::span<const double> face_flux)
{
    constexpr double roundoff_factor = 64 * std::numeric_limits<double>::epsilon();
    check(state.thermodynamic_pressure >= options.minimum_pressure &&
              state.thermodynamic_pressure <= options.maximum_pressure,
        "QBMM restart uniform pressure is outside the configured bracket.");
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto& cell = state.cells[c];
        source_stage_detail::validate_phase_numbers(cell, options.phase_roles);
        const long double absolute_pressure =
            static_cast<long double>(state.thermodynamic_pressure) + cell.mechanical_pressure;
        check(absolute_pressure > 0 && absolute_pressure <= std::numeric_limits<double>::max(),
            "QBMM restart absolute pressure is nonphysical.");
        double common_temperature = 0, total_volume = 0, total_mass = 0;
        for (const auto& phase : cell.phases)
        {
            check(std::all_of(phase.moles.begin(), phase.moles.end(), [](double moles) { return moles >= 0; }),
                "QBMM restart contains negative species moles.");
            check(phase.absolute_pressure > 0 &&
                      std::abs(static_cast<long double>(phase.absolute_pressure) - absolute_pressure) <=
                          std::max(roundoff_factor, options.thermal_relative_tolerance) * absolute_pressure,
                "QBMM restart phase pressure disagrees with accepted mechanical and thermodynamic pressure.");
            const double mass = phase_mass(phase);
            check(std::isfinite(mass), "QBMM restart phase mass is unrepresentable.");
            if (mass == 0)
            {
                check(phase.internal_energy == 0 && phase.temperature == 0 &&
                          std::all_of(phase.moles.begin(), phase.moles.end(), [](double moles) { return moles == 0; }),
                    "QBMM restart contains energy, temperature or unrepresentable material in an empty phase.");
                continue;
            }
            const auto point =
                phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase));
            const double volume = mass / point.density;
            const double energy = mass * point.internal_energy;
            const long double caloric =
                static_cast<long double>(mass) * phase.temperature * point.internal_energy_temperature;
            const long double roundoff =
                roundoff_factor * (std::abs(static_cast<long double>(energy)) + std::abs(phase.internal_energy) +
                                      static_cast<long double>(phase.absolute_pressure) * volume);
            check(std::isfinite(volume) && volume > 0 && std::isfinite(energy) && std::isfinite(caloric) &&
                      std::abs(static_cast<long double>(energy) - phase.internal_energy) <=
                          std::min(roundoff, 0.25 * options.conservation_relative_tolerance * caloric),
                "QBMM restart energy is inconsistent with the supplied caloric model.");
            if (common_temperature == 0)
                common_temperature = phase.temperature;
            check(std::abs(common_temperature - phase.temperature) <= roundoff_factor * common_temperature,
                "QBMM restart phases do not share a common temperature.");
            total_volume += volume;
            total_mass += mass;
        }
        const double volume = mesh.cell_volume(static_cast<LO>(c));
        check(std::isfinite(total_mass) && total_mass > 0 &&
                  std::abs(total_volume - volume) <= options.volume_relative_tolerance * volume,
            "QBMM restart phase EOS volumes do not fill the receiving control volume.");
        for (size_t component = 0; component < 3; ++component)
            check(std::isfinite(total_mass * cell.velocity.component(component)),
                "QBMM restart momentum is unrepresentable.");
    }
    for (size_t f = 0; f < face_flux.size(); ++f)
        check(std::isfinite(face_flux[f]) && (!mesh.is_geometry_exterior_face(static_cast<LO>(f)) || face_flux[f] == 0),
            "QBMM restart face flux is nonfinite or flows through a closed boundary.");
}

void require_uniform_restart_state(const Solver::Mesh& mesh, const Solver::State& state, std::uint64_t generation)
{
    collective_detail::require_uniform_value(mesh, state.thermodynamic_pressure, "QBMM restart pressure");
    collective_detail::require_uniform_value(mesh, state.time, "QBMM restart time");
    collective_detail::require_uniform_value(mesh, state.revision, "QBMM restart revision");
    collective_detail::require_uniform_value(mesh, generation, "QBMM restart generation");
}
} // namespace

std::vector<std::byte> LowMachMixtureSolver::serialize_qbmm_restart() const
{
    const auto& mesh = *d->mesh;
    const auto& state = d->accepted;
    RestartWriter writer;
    std::uint64_t checksum = 0, model = 0;
    const auto signature = d->checkpoint_controller();
    collective_detail::collective_local_validation(mesh, "QBMM restart encoding",
        [&]
        {
            check(d->initialized && !d->options.qbmm_families.empty() && d->logical_geometry == 0 &&
                      d->epoch == mesh_geometry_epoch(mesh),
                "QBMM restart requires initialized QBMM state on unchanged fixed geometry.");
            check(d->trial_nonce != 0 && d->trial_nonce != std::numeric_limits<std::uint64_t>::max(),
                "QBMM restart generation is invalid or exhausted.");
            qbmm_detail::validate_state(d->options, state);
            model = restart_model_code(d->options);
            const auto& catalog = d->eos.front()->catalog();
            writer(qbmm_restart_magic, qbmm_restart_version, restart_partition_fingerprint(mesh),
                restart_schema_fingerprint(d->options, d->eos), catalog.fingerprint(),
                static_cast<std::uint64_t>(d->eos.size()), static_cast<std::uint64_t>(catalog.species().size()),
                static_cast<std::uint64_t>(mesh.num_owned_cells()), static_cast<std::uint64_t>(mesh.num_owned_faces()),
                state.thermodynamic_pressure, state.time, state.revision, d->trial_nonce);
            writer(signature.has_value(), signature.value_or(0));
            writer(model, d->qbmm_legacy_default_only);
            for (const auto& stamp : state.qbmm_families)
            {
                writer(stamp.has_value());
                if (stamp)
                    writer(stamp->catalog_fingerprint, stamp->family_fingerprint, stamp->mass_scale_kg);
            }
            for (const auto& cell : state.cells)
            {
                writer(cell.velocity, cell.mechanical_pressure);
                for (size_t a = 0; a < d->eos.size(); ++a)
                {
                    const auto& phase = cell.phases[a];
                    writer(phase.absolute_pressure, phase.internal_energy, phase.temperature, cell.drift_velocity[a]);
                    for (const double moles : phase.moles)
                        writer(moles);
                    const auto& number = cell.phase_numbers[a];
                    const auto& shape = cell.qbmm_shape[a];
                    writer(number.has_value());
                    if (number)
                        writer(*number);
                    writer(shape.has_value());
                    if (shape)
                        writer(shape->H2, shape->H3);
                }
            }
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                writer(d->flux.owned_value(static_cast<LO>(f)));
            restart_diagnostics(writer, d->diagnostics, d->eos.size(), catalog.species().size(), qbmm_restart_version);
            auto balances = d->qbmm_publication ? d->qbmm_publication->balances : std::nullopt;
            restart_qbmm_balances(writer, balances, d->diagnostics, catalog);
            if (balances)
                writer(d->qbmm_publication->shape_diagnostics_version);
            validate_restart_fourier_uncertainty(d->diagnostics);
            if (const auto uncertainty = d->diagnostics.maximum_fourier_gross_comparison_uncertainty_j)
                writer(*uncertainty);
            checksum = restart_checksum(writer.bytes);
        });
    require_uniform_restart_state(mesh, state, d->trial_nonce);
    require_uniform_restart_configuration(
        mesh, d->options, d->eos, state, signature, model, d->qbmm_legacy_default_only);
    require_uniform_restart_fourier_uncertainty(mesh, d->diagnostics);
    // The local digest includes the rank and its maps; the collective digest
    // also rejects shards accidentally mixed between different saved images.
    std::uint64_t image_checksum = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &checksum, &image_checksum);
    collective_detail::collective_local_validation(
        mesh, "QBMM restart checksum encoding", [&] { writer(checksum, image_checksum); });
    return std::move(writer.bytes);
}

void LowMachMixtureSolver::restore_qbmm_restart(std::span<const std::byte> payload, const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    State candidate;
    Diagnostics statistics;
    std::vector<double> face_flux;
    std::optional<std::uint64_t> controller_signature;
    std::shared_ptr<const QbmmPublication> publication;
    std::uint64_t version = 0, model = 0, saved_generation = 0, checksum = 0, image_checksum = 0;
    bool legacy_default_only = false;
    collective_detail::collective_local_validation(mesh, "QBMM restart decoding",
        [&]
        {
            check(
                !d->options.qbmm_families.empty() && d->logical_geometry == 0 && d->epoch == mesh_geometry_epoch(mesh),
                "QBMM restart target requires QBMM configuration on unchanged fixed geometry.");
            check(payload.size() >= 2 * restart_word_size, "Truncated QBMM restart checksum.");
            RestartReader end(payload.last(2 * restart_word_size));
            end(checksum, image_checksum);
            const auto body = payload.first(payload.size() - 2 * restart_word_size);
            check(restart_checksum(body) == checksum, "QBMM restart data checksum differs.");
            RestartReader reader(body);
            reader.expect(qbmm_restart_magic, "Invalid QBMM restart signature.");
            reader(version);
            check(version >= 1 && version <= qbmm_restart_version, "Unsupported QBMM restart version.");
            reader.expect(restart_partition_fingerprint(mesh), "QBMM restart mesh geometry or partition differs.");
            reader.expect(restart_schema_fingerprint(d->options, d->eos), "QBMM restart phase or EOS schema differs.");
            const auto& catalog = d->eos.front()->catalog();
            const auto phases = d->eos.size(), species = catalog.species().size();
            reader.expect(catalog.fingerprint(), "QBMM restart ordered species catalog differs.");
            reader.expect(phases, "QBMM restart phase count differs.");
            reader.expect(species, "QBMM restart species count differs.");
            reader.expect(mesh.num_owned_cells(), "QBMM restart owned cell count differs.");
            reader.expect(mesh.num_owned_faces(), "QBMM restart owned face count differs.");
            reader(candidate.thermodynamic_pressure, candidate.time, candidate.revision, saved_generation);
            check(saved_generation != 0 && saved_generation != std::numeric_limits<std::uint64_t>::max(),
                "QBMM restart generation is invalid or exhausted.");
            if (version >= 4)
            {
                bool coupled = false;
                std::uint64_t signature = 0;
                reader(coupled, signature);
                check(coupled || signature == 0, "Invalid QBMM restart controller stamp.");
                if (coupled)
                    controller_signature = signature;
            }
            // Old images have no model stamp. In particular, a numerically zero
            // supplied drift does not make SuppliedPhaseSlip a legacy model.
            model = restart_model_code(d->options);
            if (version >= 5)
            {
                reader.expect(model, "QBMM restart transport or velocity model differs.");
                reader(legacy_default_only);
            }
            else
                legacy_default_only = true;
            if (legacy_default_only)
                validate_legacy_restart_model(d->options);
            check(!legacy_default_only || !d->controller_active || d->active_legacy_compatible,
                "Legacy QBMM restart requires default force and constitutive-law configuration.");
            check(!d->controller_active || d->active_qbmm_controller == controller_signature,
                "QBMM restart target has a different or unrecorded coupled source configuration.");
            candidate.qbmm_families.resize(phases);
            for (auto& stamp : candidate.qbmm_families)
            {
                bool present = false;
                reader(present);
                if (present)
                {
                    stamp.emplace();
                    reader(stamp->catalog_fingerprint, stamp->family_fingerprint, stamp->mass_scale_kg);
                }
            }
            const auto remaining_words = reader.remaining_size() / restart_word_size;
            const auto cells = mesh.num_owned_cells(), faces = mesh.num_owned_faces();
            check(faces <= remaining_words, "QBMM restart lacks face history.");
            if (cells != 0)
            {
                const auto cell_words = (remaining_words - faces) / cells;
                check(cell_words >= 4, "QBMM restart lacks cell velocity or mechanical pressure.");
                const auto phase_words = (cell_words - 4) / phases;
                check(phase_words >= 8 && phase_words - 8 >= species, "QBMM restart lacks phase inventory.");
            }
            candidate.cells.resize(cells);
            for (auto& cell : candidate.cells)
            {
                reader(cell.velocity, cell.mechanical_pressure);
                cell.phases.resize(phases);
                cell.drift_velocity.resize(phases);
                cell.phase_numbers.resize(phases);
                cell.qbmm_shape.resize(phases);
                for (size_t a = 0; a < phases; ++a)
                {
                    auto& phase = cell.phases[a];
                    phase.thermodynamics = d->eos[a];
                    phase.moles.resize(species);
                    reader(phase.absolute_pressure, phase.internal_energy, phase.temperature, cell.drift_velocity[a]);
                    for (double& moles : phase.moles)
                        reader(moles);
                    bool present = false;
                    reader(present);
                    if (present)
                        reader(cell.phase_numbers[a].emplace());
                    reader(present);
                    if (present)
                    {
                        auto& shape = cell.qbmm_shape[a].emplace();
                        reader(shape.H2, shape.H3);
                    }
                }
            }
            face_flux.resize(faces);
            for (double& rate : face_flux)
                reader(rate);
            restart_diagnostics(reader, statistics, phases, species, version);
            auto metadata = std::make_shared<QbmmPublication>();
            metadata->origin = QbmmStateOrigin::PortableRestart;
            metadata->archived_generation = saved_generation;
            if (version >= 4)
            {
                restart_qbmm_balances(reader, metadata->balances, statistics, catalog);
                if (metadata->balances)
                {
                    reader(metadata->shape_diagnostics_version);
                    check(metadata->shape_diagnostics_version >= 1 && metadata->shape_diagnostics_version <= 3,
                        "Invalid QBMM restart shape diagnostic provenance.");
                }
            }
            else
            {
                metadata->balances = qbmm_archived_native_balances(statistics);
                metadata->shape_diagnostics_version = version;
            }
            if (statistics.correctors > 0)
                metadata->history = QbmmReceiptHistory::UnavailableAfterPortableRestart;
            publication = std::move(metadata);
            if (version == 5 && reader.remaining_size() == restart_word_size)
                reader(statistics.maximum_fourier_gross_comparison_uncertainty_j.emplace());
            check(reader.remaining_size() == 0, "Unexpected trailing QBMM restart data.");
            validate_restart_fourier_uncertainty(statistics);
            qbmm_detail::validate_state(d->options, candidate);
        });
    require_uniform_restart_state(mesh, candidate, saved_generation);
    require_uniform_restart_configuration(
        mesh, d->options, d->eos, candidate, controller_signature, model, legacy_default_only);
    collective_detail::require_uniform_value(mesh, version, "QBMM restart archive version");
    require_uniform_restart_fourier_uncertainty(mesh, statistics);
    collective_detail::require_uniform_value(mesh, image_checksum, "QBMM restart image checksum");
    std::uint64_t actual_image_checksum = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &checksum, &actual_image_checksum);
    check(actual_image_checksum == image_checksum, "QBMM restart ranks do not belong to the same saved image.");
    collective_detail::collective_local_validation(
        mesh, "QBMM restart physical state", [&] { validate_restart_physics(mesh, d->options, candidate, face_flux); });
    d->publish_fixed(std::move(candidate), std::move(statistics), face_flux,
        std::max(d->trial_nonce, saved_generation) + 1, std::move(publication));
    d->accepted_qbmm_controller = controller_signature;
    d->needs_restart_controller = controller_signature.has_value();
    d->qbmm_legacy_default_only = legacy_default_only;
}

void LowMachMixtureSolver::restore_qbmm_checkpoint(const LowMachMixtureSolver& previous, const Checkpoint& saved,
    const QbmmInitialization& initialization, const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    const auto& prior = *previous.d;
    State candidate;
    Diagnostics statistics;
    std::shared_ptr<const QbmmPublication> publication;
    std::uint64_t model = 0;
    collective_detail::collective_local_validation(mesh, "Count-only checkpoint migration authority",
        [&]
        {
            check(prior.initialized && prior.options.qbmm_families.empty() && !d->options.qbmm_families.empty() &&
                      saved.owner == &previous && !saved.ale_geometry && d->logical_geometry == 0 &&
                      prior.logical_geometry == d->logical_geometry && saved.logical_geometry == prior.logical_geometry,
                "QBMM migration requires an owner-bound, count-only fixed-mesh checkpoint.");
            check(mesh.geometry_identity() == prior.mesh->geometry_identity() &&
                      d->epoch == mesh_geometry_epoch(mesh) && saved.geometry_epoch == d->epoch &&
                      prior.epoch == d->epoch && saved.state.cells.size() == mesh.num_owned_cells() &&
                      saved.face_flux.size() == mesh.num_owned_faces(),
                "QBMM migration requires unchanged native geometry and owned storage.");
            check(prior.mesh.get() == d->mesh.get() ||
                      restart_partition_fingerprint(mesh) == restart_partition_fingerprint(*prior.mesh),
                "QBMM migration requires the same native partition and local ordering.");
            check(d->eos.front()->catalog().fingerprint() == prior.eos.front()->catalog().fingerprint() &&
                      restart_schema_fingerprint(d->options, d->eos) ==
                          restart_schema_fingerprint(prior.options, prior.eos),
                "QBMM migration requires matching native catalogs, physical roles and EOS schemas.");
            model = restart_model_code(d->options);
            check(saved.count_only_body_acceleration == d->active_body_acceleration,
                "QBMM migration requires the same physical body acceleration, including its opt-in state.");
            check(saved.generation != 0 && saved.generation <= prior.trial_nonce &&
                      saved.generation != std::numeric_limits<std::uint64_t>::max() &&
                      std::isfinite(saved.state.time) &&
                      saved.state.thermodynamic_pressure >= d->options.minimum_pressure &&
                      saved.state.thermodynamic_pressure <= d->options.maximum_pressure,
                "QBMM migration checkpoint time, pressure or generation is invalid.");
            candidate = saved.state;
            statistics = saved.diagnostics;
        });
    require_uniform_restart_state(mesh, candidate, saved.generation);
    collective_detail::require_uniform_value(mesh, model, "QBMM migration velocity model");
    supply_qbmm_shape(candidate, initialization);
    collective_detail::collective_local_validation(mesh, "Migrated QBMM checkpoint physical state",
        [&]
        {
            for (auto& cell : candidate.cells)
                for (size_t a = 0; a < d->eos.size(); ++a)
                {
                    check(cell.phases[a].thermodynamics == prior.eos[a],
                        "QBMM migration checkpoint contains a foreign EOS provider.");
                    cell.phases[a].thermodynamics = d->eos[a];
                }
            qbmm_detail::validate_state(d->options, candidate);
            // Validate the receiving pressure/tolerance contract even with shared
            // EOS objects. Compatible replacements retain the original migration
            // path; EOS evaluation invokes no kinetics or force callback.
            validate_restart_physics(mesh, d->options, candidate, saved.face_flux);
            auto metadata = std::make_shared<QbmmPublication>();
            metadata->origin = QbmmStateOrigin::CountOnlyMigration;
            metadata->balances = qbmm_archived_native_balances(statistics);
            metadata->shape_diagnostics_version = 1;
            if (metadata->balances)
                metadata->history = QbmmReceiptHistory::UnavailableAfterCountOnlyMigration;
            publication = std::move(metadata);
        });
    require_uniform_restart_configuration(mesh, d->options, d->eos, candidate, d->active_qbmm_controller, model);
    d->publish_fixed(std::move(candidate), std::move(statistics), saved.face_flux,
        std::max(d->trial_nonce, saved.generation) + 1, std::move(publication));
    d->qbmm_legacy_default_only = false;
}
