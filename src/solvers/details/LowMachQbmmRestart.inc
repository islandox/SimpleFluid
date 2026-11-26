/** @file LowMachQbmmRestart.inc
 * @brief Compiled implementation fragment for versioned QBMM restart.
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
// Version 6 adds prescribed planar motion: original reference/options/target
// binding, accepted surface and geometry, the last physical motion interval and absolute wall flux,
// and accepted aggregate controller diagnostics when available.
// Version 7 is fixed Cartesian kinetic transport. It binds the substep policy
// and node laws, and retains endpoint node velocities, drift and interval diagnostics.
// Version 8 binds the realizable linear SSPRK2 policy and retains interval and
// controller diagnostics. Only the accepted step boundary is archived.
// Version 9 tags forced Cartesian kinetic, forced SSPRK2, axial kinetic, TVD-MC SSPRK2, and planar SSPRK2 payloads.
// The axial and TVD-MC tags distinguish absent acceleration from supplied zero acceleration.
// The planar SSPRK2 tag binds moving-stage and advancement controls before the planar geometry record.
// Version 10 adds fixed shared-carrier implicit BulkGas pool conversion and its accepted directional totals.
// Its new tags extend the implicit pool to planar motion, fixed forcing and fixed unforced supplied slip.
constexpr std::uint64_t qbmm_restart_magic = 0x5453524d4d425153ULL;
constexpr std::uint64_t qbmm_fixed_restart_version = 5;
constexpr std::uint64_t qbmm_planar_restart_version = 6;
constexpr std::uint64_t qbmm_kinetic_restart_version = 7;
constexpr std::uint64_t qbmm_linear_restart_version = 8;
constexpr std::uint64_t qbmm_tagged_restart_version = 9;
constexpr std::uint64_t qbmm_pool_restart_version = 10;
constexpr std::uint64_t qbmm_pool_restart_policy_revision = 1;

enum class RestartPayload : std::uint64_t
{
    FixedImplicit = 1,
    PlanarImplicit = 2,
    FixedKinetic = 3,
    FixedLinear = 4,
    FixedAxialKinetic = 5,
    FixedLinearTvdMc = 6,
    PlanarLinear = 7,
    FixedImplicitBulkGasPool = 8,
    PlanarImplicitBulkGasPool = 9,
    ForcedImplicitBulkGasPool = 10,
    FixedSlipImplicitBulkGasPool = 11
};

bool is_kinetic_payload(RestartPayload payload)
{
    return payload == RestartPayload::FixedKinetic || payload == RestartPayload::FixedAxialKinetic;
}

bool is_linear_payload(RestartPayload payload)
{
    return payload == RestartPayload::FixedLinear || payload == RestartPayload::FixedLinearTvdMc ||
           payload == RestartPayload::PlanarLinear;
}

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

    void string(std::string_view value)
    {
        write(static_cast<std::uint64_t>(value.size()));
        for (const unsigned char byte : value)
            bytes.push_back(static_cast<std::byte>(byte));
    }

    void diagnostic_vector(const std::vector<double>& values, std::optional<size_t>)
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

    void expect_bytes(std::span<const std::byte> expected, const char* message)
    {
        check(expected.size() <= remaining_size() && std::ranges::equal(d_remaining.first(expected.size()), expected),
            message);
        d_remaining = d_remaining.subspan(expected.size());
    }

    [[nodiscard]] size_t remaining_size() const noexcept { return d_remaining.size(); }

    void diagnostic_vector(std::vector<double>& values, std::optional<size_t> dimension)
    {
        std::uint64_t count = 0;
        read(count);
        check((!dimension || count == 0 || count == *dimension) && count <= remaining_size() / restart_word_size,
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

// Versions through 8 imply a payload. New versions carry a wire tag before the partition stamp.
RestartPayload decode_restart_payload(std::uint64_t version, RestartReader& reader)
{
    switch (version)
    {
        case 1:
        case 2:
        case 3:
        case 4:
        case qbmm_fixed_restart_version:
            return RestartPayload::FixedImplicit;
        case qbmm_planar_restart_version:
            return RestartPayload::PlanarImplicit;
        case qbmm_kinetic_restart_version:
            return RestartPayload::FixedKinetic;
        case qbmm_linear_restart_version:
            return RestartPayload::FixedLinear;
        case qbmm_tagged_restart_version: {
            std::uint64_t tag = 0;
            reader(tag);
            check(tag == static_cast<std::uint64_t>(RestartPayload::FixedKinetic) ||
                      tag == static_cast<std::uint64_t>(RestartPayload::FixedLinear) ||
                      tag == static_cast<std::uint64_t>(RestartPayload::FixedAxialKinetic) ||
                      tag == static_cast<std::uint64_t>(RestartPayload::FixedLinearTvdMc) ||
                      tag == static_cast<std::uint64_t>(RestartPayload::PlanarLinear),
                "Unknown v9 QBMM payload type.");
            return static_cast<RestartPayload>(tag);
        }
        case qbmm_pool_restart_version: {
            std::uint64_t tag = 0;
            reader(tag);
            check(tag >= static_cast<std::uint64_t>(RestartPayload::FixedImplicitBulkGasPool) &&
                      tag <= static_cast<std::uint64_t>(RestartPayload::FixedSlipImplicitBulkGasPool),
                "Unknown v10 QBMM payload type.");
            return static_cast<RestartPayload>(tag);
        }
        default:
            throw std::invalid_argument("Unknown QBMM restart version.");
    }
}

RestartPayload configured_restart_payload(
    const Solver::Options& options, const Solver::Mesh& mesh, bool motion, bool force)
{
    if (options.qbmm_model.bulk_gas_conversion)
    {
        if (motion)
            return RestartPayload::PlanarImplicitBulkGasPool;
        if (force)
            return RestartPayload::ForcedImplicitBulkGasPool;
        if (options.qbmm_model.velocity == QbmmModelOptions::Velocity::SuppliedPhaseSlip)
            return RestartPayload::FixedSlipImplicitBulkGasPool;
        return RestartPayload::FixedImplicitBulkGasPool;
    }
    if (motion)
        return uses_qbmm_linear_transport(options.qbmm_model) ? RestartPayload::PlanarLinear
                                                             : RestartPayload::PlanarImplicit;
    if (qbmm_detail::uses_axial_kinetic_transport(options, mesh))
        return RestartPayload::FixedAxialKinetic;
    if (uses_qbmm_kinetic_transport(options.qbmm_model))
        return RestartPayload::FixedKinetic;
    if (uses_qbmm_linear_transport(options.qbmm_model))
        return options.qbmm_model.linear_slope_limiter == QbmmModelOptions::LinearSlopeLimiter::TvdMc
                   ? RestartPayload::FixedLinearTvdMc
                   : RestartPayload::FixedLinear;
    return RestartPayload::FixedImplicit;
}

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

std::uint64_t restart_ordered_maps_fingerprint(const Solver::Mesh& mesh)
{
    RestartHash hash;
    const auto comm = mesh.owned_cell_map()->getComm();
    hash.word(static_cast<std::uint64_t>(comm->getSize()));
    hash.word(static_cast<std::uint64_t>(comm->getRank()));
    for (const auto& map : {mesh.owned_cell_map(), mesh.overlap_cell_map(), mesh.owned_face_map(),
             mesh.overlap_face_map(), mesh.boundary_face_map()})
    {
        hash.word(map->getGlobalNumElements());
        hash.word(static_cast<std::uint64_t>(map->getIndexBase()));
        const auto ids = map->getLocalElementList();
        hash.word(ids.size());
        for (const auto id : ids)
            hash.word(static_cast<std::uint64_t>(id));
    }
    return hash.value;
}

std::uint64_t restart_planar_configuration_fingerprint(
    const Solver::PlanarMotion& motion, std::string_view target_identity, std::uint64_t target_revision)
{
    check(!target_identity.empty() && target_revision != 0,
        "Planar QBMM restart requires an immutable target-law identity and positive revision.");
    const auto& options = motion.options();
    RestartHash hash;
    hash.string(target_identity);
    hash.word(target_revision);
    hash.string(motion.mesh_family());
    hash.word(static_cast<std::uint64_t>(options.axis));
    for (const auto& edges : motion.reference_geometry_edges())
    {
        hash.word(edges.size());
        for (const auto edge : edges)
            hash.number(edge);
    }
    for (const auto& option : {options.deformation_start_elevation, options.maximum_level_change,
             options.quality_limits.maximum_growth_ratio, options.quality_limits.maximum_non_orthogonality_degrees,
             options.quality_limits.maximum_skewness, options.quality_limits.maximum_aspect_ratio})
    {
        hash.word(option.has_value());
        if (option)
            hash.number(*option);
    }
    hash.number(options.gcl_absolute_tolerance);
    hash.number(options.gcl_relative_tolerance);
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
              !model.cross_family_coalescence && !model.capillarity &&
              !model.independent_node_temperatures,
        "QBMM restart requires the admitted fixed-composition backward-Euler upwind model.");
    check(!model.bulk_gas_conversion || (model.family_transport == QbmmModelOptions::FamilyTransport::NativeImplicit &&
                                            (model.velocity == QbmmModelOptions::Velocity::SharedCarrier ||
                                                model.velocity == QbmmModelOptions::Velocity::SuppliedPhaseSlip)),
        "QBMM pool restart requires native implicit transport with shared-carrier or supplied-slip velocity.");
    // Explicit wire codes: enum declaration order is not an archive contract.
    switch (model.velocity)
    {
        case QbmmModelOptions::Velocity::SharedCarrier:
            return uses_qbmm_linear_transport(model) ? 3 : 0;
        case QbmmModelOptions::Velocity::SuppliedPhaseSlip:
            return 1;
        case QbmmModelOptions::Velocity::NodeDependentSlip:
            check(uses_qbmm_kinetic_transport(model), "QBMM restart node slip requires the kinetic integrator.");
            return 2;
        default:
            throw std::invalid_argument("QBMM restart has an unsupported velocity model.");
    }
}

std::uint64_t restart_kinetic_policy_fingerprint(const Solver::Options& options)
{
    RestartHash hash;
    hash.word(1); // Kinetic Euler and exact-zero symmetric donor derivative policy.
    hash.number(options.qbmm_model.kinetic_cfl);
    hash.word(options.qbmm_model.maximum_kinetic_substeps);
    hash.word(options.qbmm_node_slip_laws.size());
    for (const auto& law : options.qbmm_node_slip_laws)
    {
        hash.word(bool(law));
        if (law)
            hash.string(qbmm_node_slip_signature(*law));
    }
    return hash.value;
}

std::uint64_t restart_advance_controls(const Solver::Options& options)
{
    // Preserve the legacy physical advancement fingerprint across scalar and
    // grouped execution. The multi-RHS policy and width tune the same operator.
    RestartHash hash;
    hash.number(options.minimum_pressure);
    hash.number(options.maximum_pressure);
    hash.number(options.volume_relative_tolerance);
    hash.number(options.conservation_relative_tolerance);
    hash.number(options.continuity_tolerance);
    hash.number(options.source_relative_tolerance);
    hash.number(options.thermal_relative_tolerance);
    hash.word(options.maximum_correctors);
    hash.word(options.maximum_pressure_iterations);
    hash.word(options.maximum_caloric_seed_evaluations);
    hash.word(bool(options.caloric_domain_provider));
    if (options.caloric_domain_provider)
    {
        hash.string(options.caloric_domain_provider->identity);
        hash.word(options.caloric_domain_provider->revision);
    }
    hash.word(bool(options.mixture_dynamic_viscosity));
    hash.word(static_cast<std::uint64_t>(options.linear.backend));
    hash.word(static_cast<std::uint64_t>(options.linear.preconditioner));
    hash.word(options.linear.reuse_preconditioner);
    hash.word(options.linear.max_iterations);
    hash.number(options.linear.tolerance);
    return hash.value;
}

std::uint64_t restart_physical_fingerprint(const Solver::Options& options,
    std::span<const std::shared_ptr<const MixtureThermodynamics>> eos, std::string_view domain, bool require_identity)
{
    RestartHash hash;
    hash.string(domain);
    hash.word(restart_schema_fingerprint(options, eos));
    hash.word(eos.front()->catalog().fingerprint());
    const std::optional<ThermalLawIdentity> no_identity;
    const auto append_identity = [&](const std::optional<ThermalLawIdentity>& identity)
    {
        hash.word(identity.has_value());
        if (identity)
        {
            hash.string(identity->identity);
            hash.word(identity->revision);
            hash.word(identity->schema_fingerprint);
        }
    };
    for (size_t phase = 0; phase < eos.size(); ++phase)
    {
        const auto ideal = eos[phase]->builtin_ideal_gas_identity();
        const auto& supplied =
            options.qbmm_eos_law_identities.empty() ? no_identity : options.qbmm_eos_law_identities[phase];
        check(!require_identity || (ideal && !ideal->has_supplied_transport) || supplied,
            "Tagged QBMM restart requires a declared identity for each supplied EOS or EOS transport law.");
        hash.word(ideal.has_value());
        if (ideal)
        {
            hash.number(ideal->gas_constant);
            hash.word(ideal->has_supplied_transport);
            hash.word(ideal->caloric.size());
            for (const auto& species : ideal->caloric)
            {
                hash.number(species.molar_heat_capacity_v);
                hash.number(species.reference_temperature);
                hash.number(species.reference_internal_energy);
            }
        }
        append_identity(supplied);
    }
    check(!require_identity || !options.mixture_dynamic_viscosity || options.qbmm_mixture_viscosity_identity,
        "Tagged QBMM restart requires a declared identity for its supplied mixture viscosity law.");
    append_identity(options.qbmm_mixture_viscosity_identity);
    return hash.value;
}

RestartWriter restart_pool_configuration(const Solver::Options& options,
    std::span<const std::shared_ptr<const MixtureThermodynamics>> eos, RestartPayload type)
{
    check(bool(options.qbmm_bulk_gas_pool), "QBMM pool restart requires its BulkGas phase binding.");
    const auto& pool = *options.qbmm_bulk_gas_pool;
    const auto species = pool.catalog().species();
    RestartWriter header;
    header(qbmm_pool_restart_policy_revision, restart_advance_controls(options),
        restart_physical_fingerprint(
            options, eos, "implicit-pool-physical-providers-v1", type == RestartPayload::ForcedImplicitBulkGasPool),
        static_cast<std::uint64_t>(pool.phase_index()), static_cast<std::uint64_t>(pool.direction()),
        pool.fingerprint());
    header.string(pool.phase_identity());
    header(static_cast<std::uint64_t>(species.size()));
    for (const auto& member : species)
        header(static_cast<std::uint64_t>(member.id), pool.permits(member.id));
    header(static_cast<std::uint64_t>(options.qbmm_families.size()));
    for (const auto& family : options.qbmm_families)
    {
        header(bool(family));
        if (!family)
            continue;
        const auto& versions = family->closure_versions();
        header(versions.inversion, versions.thermodynamics, versions.transport, versions.source);
        header.string(versions.source_identity);
        header(static_cast<std::uint64_t>(versions.laws.size()));
        for (const auto& law : versions.laws)
            header.string(qbmm_law_signature(law));
    }
    return header;
}

std::uint64_t restart_forced_physical_fingerprint(const Solver::Options& options,
    std::span<const std::shared_ptr<const MixtureThermodynamics>> eos, QbmmModelOptions::FamilyTransport transport,
    bool axial_kinetic)
{
    const auto domain =
        axial_kinetic                                                    ? "axial-kinetic-physical-providers-v1"
        : transport == QbmmModelOptions::FamilyTransport::KineticEuler   ? "forced-kinetic-physical-providers-v1"
        : transport == QbmmModelOptions::FamilyTransport::NativeImplicit ? "implicit-pool-physical-providers-v1"
                                                                         : "forced-linear-physical-providers-v1";
    return restart_physical_fingerprint(options, eos, domain, true);
}

std::uint64_t restart_force_binding(
    std::uint64_t signature, const Solver::Vector& acceleration, std::uint64_t physical_fingerprint)
{
    RestartHash hash;
    hash.word(signature);
    hash.vector(acceleration);
    hash.word(physical_fingerprint);
    return hash.value;
}

std::uint64_t restart_axial_binding(
    std::optional<std::uint64_t> signature, std::optional<Solver::Vector> force, std::uint64_t physical_fingerprint)
{
    RestartHash hash;
    hash.word(signature.has_value());
    hash.word(signature.value_or(0));
    hash.word(force.has_value());
    if (force)
        hash.vector(*force);
    hash.word(physical_fingerprint);
    return hash.value;
}

void validate_legacy_restart_model(const Solver::Options& options)
{
    check(!options.qbmm_model.bulk_gas_conversion && restart_model_code(options) == 0,
        "QBMM restart versions 1-4 require the zero-slip shared-carrier model.");
    for (const auto& family : options.qbmm_families)
        check(!family || family->closure_versions().laws.empty(),
            "QBMM restart versions 1-4 have no supplied constitutive-law metadata.");
}

void require_uniform_restart_configuration(const Solver::Mesh& mesh, const Solver::Options& options,
    std::span<const std::shared_ptr<const MixtureThermodynamics>> eos, const Solver::State& state,
    std::optional<std::uint64_t> controller, std::uint64_t model, bool legacy_default_only = false)
{
    RestartHash hash;
    collective_detail::collective_local_validation(mesh, "QBMM restart fingerprint preparation",
        [&]
        {
            hash.word(restart_schema_fingerprint(options, eos));
            hash.word(eos.front()->catalog().fingerprint());
            hash.word(model);
            if (uses_qbmm_kinetic_transport(options.qbmm_model))
                hash.word(restart_kinetic_policy_fingerprint(options));
            if (uses_qbmm_linear_transport(options.qbmm_model))
            {
                hash.word(qbmm_linear_transport_revision);
                hash.number(options.qbmm_model.linear_cfl);
                hash.word(options.qbmm_model.maximum_linear_substeps);
                hash.word(static_cast<std::uint64_t>(options.qbmm_model.linear_slope_limiter));
            }
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
        });
    collective_detail::require_uniform_value(mesh, hash.value, "QBMM restart model and family schema");
    collective_detail::require_uniform_value(mesh, int(controller.has_value()), "QBMM restart controller presence");
    collective_detail::require_uniform_value(mesh, controller.value_or(0), "QBMM restart controller signature");
}

template<class Archive, class Diagnostics>
void restart_diagnostics(
    Archive& io, Diagnostics& value, size_t phases, size_t species, std::uint64_t version, RestartPayload type)
{
    // Keep the existing portable summaries. The multi-RHS and family halo
    // instrumentation stays in live diagnostics.
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
    if (is_kinetic_payload(type))
        io(value.kinetic_substeps, value.maximum_kinetic_cfl);
    if (is_linear_payload(type))
        io(value.linear_substeps, value.linear_stages, value.maximum_linear_cfl, value.maximum_linear_stage_residual,
            value.maximum_linear_step_residual);
    if (type == RestartPayload::FixedLinearTvdMc || type == RestartPayload::PlanarLinear)
        io(value.linear_upstream_returned_packets);
}

void validate_restart_kinetic_diagnostics(const Solver::Options& options, const Solver::Diagnostics& statistics)
{
    check(statistics.kinetic_substeps >= 0 &&
              statistics.kinetic_substeps <= options.qbmm_model.maximum_kinetic_substeps &&
              std::isfinite(statistics.maximum_kinetic_cfl) && statistics.maximum_kinetic_cfl >= 0 &&
              statistics.maximum_kinetic_cfl <= options.qbmm_model.kinetic_cfl &&
              (statistics.kinetic_substeps > 0 ? statistics.correctors > 0 : statistics.maximum_kinetic_cfl == 0),
        "QBMM restart kinetic interval diagnostics disagree with the configured substep policy.");
}

template<class Archive, class Diagnostics, class Interval>
void restart_forced_family_impulses(Archive& io, Diagnostics& stats, Interval& interval, bool has_force)
{
    bool present = stats.impulse_parts.has_value();
    io(present);
    check(!present || has_force, "QBMM restart body impulse requires supplied acceleration.");
    if (!present)
        return;
    if constexpr (std::is_same_v<Archive, RestartReader>)
    {
        stats.impulse_parts.emplace();
        interval.emplace();
    }
    check(interval.has_value(), "QBMM restart body impulse requires interval provenance.");
    auto& parts = *stats.impulse_parts;
    io(parts.body, parts.other, interval->old_time, interval->time_step, interval->local_body, interval->local_other,
        interval->local_external);
}

void validate_forced_family_impulses(const Solver::Vector& acceleration, const Solver::Diagnostics& stats,
    const Solver::State& state, const std::optional<ForcedFamilyInterval>& interval)
{
    const bool advanced = stats.family_transport == QbmmModelOptions::FamilyTransport::NativeImplicit
                              ? stats.correctors > 0
                              : stats.kinetic_substeps > 0 || stats.linear_substeps > 0;
    check(stats.impulse_parts.has_value() == advanced && interval.has_value() == stats.impulse_parts.has_value(),
        "Forced QBMM impulse diagnostics disagree with interval history.");
    if (!interval)
        return;
    check(stats.source_evaluations == stats.correctors && stats.correctors >= 2,
        "Forced QBMM interval requires a source evaluation for each accepted corrector.");
    check(std::isfinite(interval->old_time) && std::isfinite(interval->time_step) && interval->time_step > 0 &&
              state.time > interval->old_time && state.time == interval->old_time + interval->time_step,
        "Forced QBMM impulse interval disagrees with the accepted clock.");

    // Reconstruct the local contribution independently from accepted species stocks. The source-stage
    // replay admits a relative endpoint discrepancy of 64 eps times the sum of both endpoint magnitudes.
    // Allow that discrepancy and the rounding of each cell impulse and the final local sum.
    FaceSum mass;
    for (const auto& cell : state.cells)
        for (const auto& phase : cell.phases)
        {
            const auto& species = phase.thermodynamics->catalog().species();
            for (size_t s = 0; s < phase.moles.size(); ++s)
                mass.add_product(phase.moles[s], species[s].molarMass);
        }
    constexpr double eps = std::numeric_limits<double>::epsilon();
    constexpr double quantum = std::numeric_limits<double>::denorm_min();
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const double g = acceleration.component(axis), body = interval->local_body.component(axis);
        const double other = interval->local_other.component(axis), external = interval->local_external.component(axis);
        const double global_body = stats.impulse_parts->body.component(axis);
        const double global_other = stats.impulse_parts->other.component(axis);
        const double global_external = stats.external_impulse.component(axis);
        check(std::isfinite(body) && std::isfinite(other) && std::isfinite(external) && std::isfinite(global_body) &&
                  std::isfinite(global_other) && std::isfinite(global_external),
            "Forced QBMM impulse diagnostics are nonfinite.");
        if (state.cells.empty())
            check(body == 0. && other == 0. && external == 0., "Empty owner has a forced QBMM impulse.");
        if (g == 0.)
        {
            check(body == 0. && global_body == 0. && other == external && global_other == global_external,
                "Forced QBMM impulse is inconsistent with an unforced component.");
            continue;
        }
        check((body == 0. || std::signbit(body) == std::signbit(g)) &&
                  (global_body == 0. || std::signbit(global_body) == std::signbit(g)),
            "Forced QBMM body impulse points against its acceleration.");
        FaceSum force, body_error;
        force.add_weighted(mass, g);
        body_error.add_weighted(force, interval->time_step);
        body_error.add_product(-body, 1.);
        const double body_bound = 256 * eps * std::abs(body) + quantum * (1. + static_cast<double>(state.cells.size()));
        check(std::abs(body_error.rounded()) <= body_bound,
            "Forced QBMM body impulse disagrees with the accepted material mass and interval.");

        FaceSum split_error;
        split_error.add_product(external, 1.);
        split_error.add_product(-body, 1.);
        split_error.add_product(-other, 1.);
        const double split_bound =
            eps * std::abs(body) + eps * std::abs(other) + eps * std::abs(external) + 3 * quantum;
        check(std::abs(split_error.rounded()) <= split_bound,
            "Forced QBMM rank-local external impulse disagrees with its body and other operands.");
    }
}

void require_uniform_forced_family_history(const Solver::Mesh& mesh, const Solver::Vector& force,
    const Solver::Diagnostics& stats, const std::optional<ForcedFamilyInterval>& interval)
{
    RestartHash hash;
    hash.vector(force);
    hash.vector(stats.external_impulse);
    hash.vector(stats.pressure_impulse);
    hash.word(stats.impulse_parts.has_value());
    if (const auto& parts = stats.impulse_parts)
    {
        hash.vector(parts->body);
        hash.vector(parts->other);
        hash.number(interval->old_time);
        hash.number(interval->time_step);
    }
    collective_detail::require_uniform_value(mesh, hash.value, "Forced QBMM restart impulse history");
    if (!interval)
        return;

    // The published totals came from three independent reductions. Normalize each component by a
    // power of two and bound the discrepancy using the sum of rank-local magnitudes. This allows
    // cancellation and different MPI summation orders without using the possibly tiny net impulse.
    std::array<double, 9> local{}, saved{}, magnitudes{}, maximum{};
    for (size_t axis = 0; axis < 3; ++axis)
    {
        local[axis] = interval->local_body.component(axis);
        local[3 + axis] = interval->local_other.component(axis);
        local[6 + axis] = interval->local_external.component(axis);
        saved[axis] = stats.impulse_parts->body.component(axis);
        saved[3 + axis] = stats.impulse_parts->other.component(axis);
        saved[6 + axis] = stats.external_impulse.component(axis);
    }
    const auto& comm = *mesh.owned_cell_map()->getComm();
    if (comm.getSize() == 1)
    {
        check(local == saved, "Forced QBMM archived impulse totals differ from their local operands.");
        return;
    }
    std::ranges::transform(local, magnitudes.begin(), [](double value) { return std::abs(value); });
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, static_cast<int>(local.size()), magnitudes.data(), maximum.data());
    std::array<int, 9> exponents{};
    std::array<double, 18> scaled_local{}, scaled_sum{};
    for (size_t k = 0; k < local.size(); ++k)
    {
        std::frexp(maximum[k], &exponents[k]);
        scaled_local[k] = std::scalbn(local[k], -exponents[k]);
        scaled_local[local.size() + k] = std::abs(scaled_local[k]);
    }
    Teuchos::reduceAll(
        comm, Teuchos::REDUCE_SUM, static_cast<int>(scaled_local.size()), scaled_local.data(), scaled_sum.data());
    for (size_t k = 0; k < local.size(); ++k)
    {
        const double reported = std::scalbn(saved[k], -exponents[k]);
        const double allowance =
            4. * comm.getSize() * std::numeric_limits<double>::epsilon() * scaled_sum[local.size() + k];
        check(std::abs(reported - scaled_sum[k]) <= allowance,
            "Forced QBMM archived impulse totals differ from their rank-local operands.");
    }
}

void require_uniform_restart_kinetic_diagnostics(const Solver::Mesh& mesh, const Solver::Diagnostics& statistics)
{
    collective_detail::require_uniform_value(mesh, statistics.kinetic_substeps, "QBMM restart kinetic substeps");
    collective_detail::require_uniform_value(mesh, statistics.maximum_kinetic_cfl, "QBMM restart maximum kinetic CFL");
}

void validate_restart_linear_diagnostics(const QbmmModelOptions& model, const Solver::Diagnostics& statistics)
{
    const auto in_gate = [](double value, double limit)
    { return std::isfinite(value) && value >= 0 && value <= limit; };
    check(statistics.linear_substeps >= 0 && statistics.linear_substeps <= model.maximum_linear_substeps &&
              statistics.linear_stages == 2 * static_cast<std::uint64_t>(statistics.linear_substeps) &&
              in_gate(statistics.maximum_linear_cfl, model.linear_cfl) &&
              in_gate(statistics.maximum_linear_stage_residual, 1.) &&
              in_gate(statistics.maximum_linear_step_residual, 1.) &&
              (statistics.linear_upstream_returned_packets == 0 ||
                  (model.linear_slope_limiter == QbmmModelOptions::LinearSlopeLimiter::TvdMc &&
                      statistics.linear_stages > 0)) &&
              (statistics.linear_substeps > 0
                      ? statistics.correctors > 0
                      : statistics.maximum_linear_cfl == 0 && statistics.maximum_linear_stage_residual == 0 &&
                            statistics.maximum_linear_step_residual == 0),
        "QBMM restart realizable linear diagnostics disagree with the configured substep policy.");
}

void require_uniform_restart_linear_diagnostics(const Solver::Mesh& mesh, const Solver::Diagnostics& statistics)
{
    collective_detail::require_uniform_value(mesh, statistics.linear_substeps, "QBMM restart linear substeps");
    collective_detail::require_uniform_value(mesh, statistics.linear_stages, "QBMM restart linear stages");
    collective_detail::require_uniform_value(
        mesh, statistics.linear_upstream_returned_packets, "QBMM restart linear upstream returns");
    collective_detail::require_uniform_value(mesh, statistics.maximum_linear_cfl, "QBMM restart maximum linear CFL");
    collective_detail::require_uniform_value(
        mesh, statistics.maximum_linear_stage_residual, "QBMM restart maximum linear stage residual");
    collective_detail::require_uniform_value(
        mesh, statistics.maximum_linear_step_residual, "QBMM restart maximum linear step residual");
}

template<class Archive, class Diagnostics>
void restart_controller_diagnostics(
    Archive& io, Diagnostics& value, size_t phases, std::optional<size_t> reactions, int flow_correctors)
{
    io(value.correctors, value.chemical_heat, value.external_heat, value.transfer_energy, value.external_mass,
        value.net_thermal_energy);
    check(value.correctors > 0 && value.correctors == flow_correctors,
        "QBMM restart reacting and native corrector counts differ.");
    for (auto* extents : {&value.reaction_extents, &value.forward_reaction_extents, &value.reverse_reaction_extents})
    {
        io.diagnostic_vector(*extents, reactions);
        if (!reactions)
            reactions = extents->size();
        check(extents->size() == *reactions, "QBMM restart controller reaction diagnostic dimensions differ.");
    }
    for (auto* numbers : {&value.population_births, &value.population_deaths, &value.population_number_changes})
    {
        io.diagnostic_vector(*numbers, phases);
        check(numbers->size() == phases, "QBMM restart controller population diagnostic dimensions differ.");
    }
}

template<class Archive, class Diagnostics>
void restart_pool_diagnostics(Archive& io, Diagnostics& value, const QbmmBulkGasPoolBinding& pool)
{
    constexpr bool reading = std::is_same_v<Archive, RestartReader>;
    const auto species = pool.catalog().species();
    bool present = value.pool_conversion.has_value();
    io(present);
    check(present == (value.correctors > 0), "QBMM pool restart diagnostics disagree with accepted correctors.");
    if (!present)
        return;
    if constexpr (reading)
        value.pool_conversion.emplace();
    auto& diagnostics = *value.pool_conversion;
    io(diagnostics.law_evaluations);
    for (size_t i = 0; i < diagnostics.directions.size(); ++i)
    {
        const auto direction = i == 0 ? QbmmPoolConversionDirection::DispersedToPool
                                      : QbmmPoolConversionDirection::PoolToDispersed;
        const bool allowed = pool.allows(direction);
        auto& totals = diagnostics.directions[i];
        io(totals.count, totals.mass_kg, totals.H2, totals.H3, totals.carried_enthalpy_j);
        io.diagnostic_vector(totals.species_moles, species.size());
        check(totals.species_moles.size() == species.size() && std::isfinite(totals.carried_enthalpy_j) &&
                  (allowed || totals.carried_enthalpy_j == 0.),
            "QBMM pool restart diagnostics have invalid species dimensions or carried enthalpy.");
        for (const double quantity : {totals.count, totals.mass_kg, totals.H2, totals.H3})
            check(std::isfinite(quantity) && quantity >= 0. && (allowed || quantity == 0.),
                "QBMM pool restart contains an invalid directional total.");
        FaceSum mass;
        for (size_t k = 0; k < species.size(); ++k)
        {
            const double moles = totals.species_moles[k];
            check(std::isfinite(moles) && moles >= 0. && ((allowed && pool.permits(species[k].id)) || moles == 0.),
                "QBMM pool restart contains an invalid directional species total.");
            mass.add_product(moles, species[k].molarMass);
        }
        const auto expected_mass = mass.rounded();
        const bool active = totals.count > 0;
        check(active == (expected_mass > 0) && active == (totals.mass_kg > 0) &&
                  active == (totals.H2 > 0) && active == (totals.H3 > 0) &&
                  (active ? diagnostics.law_evaluations > 0 : totals.carried_enthalpy_j == 0),
            "QBMM pool restart activity differs from its directional quantities or law provenance.");
        qbmm_detail::validate_raw_value(totals.mass_kg, expected_mass,
            "QBMM pool restart mass disagrees with its catalogued species.");
    }
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

struct RestartPlanarWalls
{
    size_t axis;
    double old_cap, time_step;
};

void validate_restart_physics(const Solver::Mesh& mesh, const Solver::Options& options, Solver::State& state,
    std::span<const double> face_flux, std::optional<RestartPlanarWalls> planar = {}, bool initialize_kinetic = false)
{
    constexpr double roundoff_factor = 64 * std::numeric_limits<double>::epsilon();
    const bool retained_kinetic = uses_qbmm_kinetic_transport(options.qbmm_model) && !initialize_kinetic;
    check(state.thermodynamic_pressure >= options.minimum_pressure &&
              state.thermodynamic_pressure <= options.maximum_pressure,
        "QBMM restart uniform pressure is outside the configured bracket.");
    std::vector<double> phase_volumes;
    if (planar || initialize_kinetic)
        phase_volumes.resize(options.phase_roles.size());
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        auto& cell = state.cells[c];
        source_stage_detail::validate_phase_numbers(cell, options.phase_roles);
        check(!retained_kinetic || cell.qbmm_kinetic_eos_volumes.size() == cell.phases.size(),
            "Kinetic QBMM restart EOS traces differ from the physical phase layout.");
        const long double absolute_pressure =
            static_cast<long double>(state.thermodynamic_pressure) + cell.mechanical_pressure;
        check(absolute_pressure > 0 && absolute_pressure <= std::numeric_limits<double>::max(),
            "QBMM restart absolute pressure is nonphysical.");
        double common_temperature = 0, total_volume = 0, total_mass = 0;
        for (size_t a = 0; a < cell.phases.size(); ++a)
        {
            const auto& phase = cell.phases[a];
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
                check(!retained_kinetic || cell.qbmm_kinetic_eos_volumes[a] == 0.,
                    "An empty kinetic QBMM restart phase must have zero retained EOS volume.");
                if (planar || initialize_kinetic)
                    phase_volumes[a] = 0.;
                continue;
            }
            const auto point =
                phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions(phase));
            const double volume = mass / point.density;
            if (retained_kinetic)
            {
                const double retained = cell.qbmm_kinetic_eos_volumes[a];
                check(std::isfinite(retained) && retained > 0 &&
                          std::abs(retained - volume) / volume <= options.volume_relative_tolerance,
                    "Kinetic QBMM restart retained volume is inconsistent with the occupied EOS state.");
            }
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
            if (planar || initialize_kinetic)
                phase_volumes[a] = volume;
        }
        const double volume = mesh.cell_volume(static_cast<LO>(c));
        check(std::isfinite(total_mass) && total_mass > 0 &&
                  std::abs(total_volume - volume) <= options.volume_relative_tolerance * volume,
            "QBMM restart phase EOS volumes do not fill the receiving control volume.");
        for (size_t component = 0; component < 3; ++component)
            check(std::isfinite(total_mass * cell.velocity.component(component)),
                "QBMM restart momentum is unrepresentable.");
        if (planar)
            (void) effective_mixture_viscosity(options, cell.phases, phase_volumes, true);
        if (initialize_kinetic)
            qbmm_native_law_detail::evaluate_kinetic_closure(options, mesh, cell, volume, phase_volumes,
                qbmm_native_law_detail::kinetic_location(mesh, static_cast<LO>(c)));
        if (retained_kinetic)
            qbmm_native_law_detail::validate_archived_node_slip_gates(options, cell, volume);
    }
    for (size_t f = 0; f < face_flux.size(); ++f)
    {
        check(std::isfinite(face_flux[f]), "QBMM restart face flux is nonfinite.");
        const LO face = static_cast<LO>(f);
        if (!mesh.is_geometry_exterior_face(face))
            continue;
        if (planar)
        {
            const auto area = mesh.face_area_vector(face);
            double expected = 0.;
            if (planar->time_step > 0. && area.component(planar->axis) > 0.)
            {
                // The outward upper cap moves along its selected axis. Reuse the native
                // represented centroid, oriented area and sweep evaluation order.
                const auto current = mesh.face_centroid(face);
                auto previous = current;
                previous.component(planar->axis) = planar->old_cap;
                expected = (current - previous).dot(area) / planar->time_step;
            }
            check(std::isfinite(expected) && face_flux[f] == expected,
                "QBMM restart exterior flux differs from the accepted planar wall sweep.");
            for (const auto& drift : state.cells[mesh.owner_cell(face)].drift_velocity)
                check(drift.dot(area) == 0., "Planar QBMM restart requires zero normal phase drift at moving walls.");
        }
        else
            check(face_flux[f] == 0., "QBMM restart face flux flows through a fixed closed wall.");
    }
}

void require_uniform_restart_wall_interval(const Solver::Mesh& mesh, const std::optional<PlanarWallInterval>& interval)
{
    collective_detail::require_uniform_value(mesh, int(interval.has_value()), "QBMM restart wall interval presence");
    if (interval)
    {
        collective_detail::require_uniform_value(
            mesh, interval->old_surface, "QBMM restart previous surface elevation");
        collective_detail::require_uniform_value(mesh, interval->time_step, "QBMM restart wall interval duration");
    }
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
    return serialize_qbmm_restart_impl(nullptr, {}, 0);
}

std::vector<std::byte> LowMachMixtureSolver::serialize_qbmm_restart(
    const PlanarMotion& motion, std::string_view target_identity, std::uint64_t target_revision) const
{
    return serialize_qbmm_restart_impl(&motion, target_identity, target_revision);
}

std::vector<std::byte> LowMachMixtureSolver::serialize_qbmm_restart_impl(const PlanarMotion* motion,
    std::string_view target_identity, std::uint64_t target_revision,
    const ControllerRestartDiagnostics* controller_diagnostics) const
{
    const auto& mesh = *d->mesh;
    const auto& state = d->accepted;
    const bool kinetic = uses_qbmm_kinetic_transport(d->options.qbmm_model);
    const bool linear = uses_qbmm_linear_transport(d->options.qbmm_model);
    const bool explicit_family = kinetic || linear;
    const bool pool_conversion = d->options.qbmm_model.bulk_gas_conversion;
    const auto force = d->controller_active && !d->has_forced_family_provenance() &&
                               !(pool_conversion && d->explicit_family_force_recorded)
                           ? acceleration_vector(d->active_body_acceleration)
                           : d->diagnostics.body_acceleration_m_per_s2;
    const auto type = configured_restart_payload(d->options, mesh, motion != nullptr, force.has_value());
    const bool forced_pool = type == RestartPayload::ForcedImplicitBulkGasPool;
    const bool axial_kinetic = type == RestartPayload::FixedAxialKinetic;
    const bool planar_linear = type == RestartPayload::PlanarLinear;
    const bool fixed_tvd_mc = type == RestartPayload::FixedLinearTvdMc;
    const bool tvd_mc = d->options.qbmm_model.linear_slope_limiter == QbmmModelOptions::LinearSlopeLimiter::TvdMc;
    const bool retains_linear_controller = fixed_tvd_mc || planar_linear;
    const bool tagged = axial_kinetic || retains_linear_controller || (explicit_family && force);
    const auto version = pool_conversion ? qbmm_pool_restart_version
                         : tagged        ? qbmm_tagged_restart_version
                         : motion        ? qbmm_planar_restart_version
                         : kinetic       ? qbmm_kinetic_restart_version
                         : linear        ? qbmm_linear_restart_version
                                         : qbmm_fixed_restart_version;
    collective_detail::require_uniform_value(mesh, version, "QBMM restart encoding version");
    collective_detail::require_uniform_value(mesh, static_cast<std::uint64_t>(type), "QBMM restart payload type");
    RestartWriter writer;
    std::uint64_t checksum = 0, model = 0, motion_configuration = 0, controller_history = 0;
    std::uint64_t physical_fingerprint = 0;
    std::uint64_t axial_binding = 0, planar_linear_controls = 0, pool_configuration = 0;
    double surface = 0;
    const bool accepted_controller_only = (axial_kinetic && d->diagnostics.kinetic_substeps > 0) ||
                                          (retains_linear_controller && d->diagnostics.linear_substeps > 0) ||
                                          (pool_conversion && d->diagnostics.correctors > 0);
    const auto signature = accepted_controller_only ? d->accepted_qbmm_controller : d->checkpoint_controller();
    const auto wall_interval = d->qbmm_publication ? d->qbmm_publication->planar_wall_interval : std::nullopt;
    const auto forced_interval = d->qbmm_publication ? d->qbmm_publication->forced_family_interval : std::nullopt;
    if (motion)
    {
        collective_detail::collective_local_validation(mesh, "QBMM restart motion authority",
            [&]
            {
                d->validate_planar_controller_authority(*motion);
                d->validate_planar_checkpoint_provenance(*motion);
                check(motion->mesh_ptr().get() == d->mesh.get() && !motion->has_active_trial() &&
                          (!d->accepted_planar_motion || d->accepted_planar_motion == motion->d_authority) &&
                          !d->active_body_acceleration && (!explicit_family || planar_linear),
                    "Planar QBMM restart requires its live idle motion authority and zero supplied body force.");
                motion_configuration =
                    restart_planar_configuration_fingerprint(*motion, target_identity, target_revision);
            });
        collective_detail::require_uniform_value(mesh, motion_configuration, "QBMM restart planar configuration");
        surface = motion->snapshot().surface_elevation();
    }
    collective_detail::collective_local_validation(mesh, "QBMM restart encoding",
        [&]
        {
            check(d->initialized && !d->options.qbmm_families.empty() &&
                      (motion || (d->logical_geometry == 0 && !d->active_controller_motion)) &&
                      d->epoch == mesh_geometry_epoch(mesh),
                "QBMM restart requires initialized QBMM state on its current accepted geometry.");
            check(d->trial_nonce != 0 && d->trial_nonce != std::numeric_limits<std::uint64_t>::max(),
                "QBMM restart generation is invalid or exhausted.");
            qbmm_detail::validate_state(d->options, state);
            model = restart_model_code(d->options);
            const auto& catalog = d->eos.front()->catalog();
            qbmm_detail::validate_capability(d->options, mesh, motion != nullptr, force);
            writer(qbmm_restart_magic, version);
            if (pool_conversion)
            {
                check(!d->qbmm_legacy_default_only && (!wall_interval || motion) &&
                          (forced_pool ? (signature && force)
                                       : (!force && !d->diagnostics.body_acceleration_m_per_s2 &&
                                             !d->diagnostics.impulse_parts && !forced_interval)),
                    "QBMM pool restart geometry, acceleration and controller provenance differ.");
                writer(static_cast<std::uint64_t>(type));
                const auto header = restart_pool_configuration(d->options, d->eos, type);
                pool_configuration = restart_checksum(header.bytes);
                writer.bytes.insert(writer.bytes.end(), header.bytes.begin(), header.bytes.end());
                if (forced_pool)
                {
                    validate_forced_family_impulses(*force, d->diagnostics, state, forced_interval);
                    writer(*force);
                    restart_forced_family_impulses(writer, d->diagnostics, forced_interval, true);
                }
            }
            else if (planar_linear)
            {
                check(!force && !d->diagnostics.impulse_parts && !forced_interval,
                    "Planar linear QBMM restart requires absent body acceleration and impulse history.");
                planar_linear_controls = restart_advance_controls(d->options);
                writer(static_cast<std::uint64_t>(type), qbmm_planar_linear_transport_revision, planar_linear_controls);
            }
            else if (tagged)
            {
                check(!force || signature, "Forced explicit QBMM restart requires a bound reacting controller.");
                if (force || axial_kinetic)
                    physical_fingerprint = restart_forced_physical_fingerprint(
                        d->options, d->eos, d->options.qbmm_model.family_transport, axial_kinetic);
                if (force)
                    validate_forced_family_impulses(*force, d->diagnostics, state, forced_interval);
                else
                    check(!d->diagnostics.impulse_parts && !forced_interval,
                        "QBMM restart body impulse requires supplied acceleration.");
                writer(static_cast<std::uint64_t>(type), force.has_value(), force.value_or(Vector{}),
                    kinetic ? balanced_kinetic_pressure_revision : balanced_linear_pressure_revision,
                    kinetic ? forced_kinetic_thermal_revision : forced_linear_thermal_revision,
                    restart_advance_controls(d->options), physical_fingerprint);
                restart_forced_family_impulses(writer, d->diagnostics, forced_interval, force.has_value());
            }
            writer(restart_partition_fingerprint(mesh));
            if (axial_kinetic)
                writer(restart_ordered_maps_fingerprint(mesh));
            if (motion)
                writer(motion_configuration, restart_ordered_maps_fingerprint(mesh), surface);
            writer(restart_schema_fingerprint(d->options, d->eos), catalog.fingerprint(),
                static_cast<std::uint64_t>(d->eos.size()), static_cast<std::uint64_t>(catalog.species().size()),
                static_cast<std::uint64_t>(mesh.num_owned_cells()), static_cast<std::uint64_t>(mesh.num_owned_faces()),
                state.thermodynamic_pressure, state.time, state.revision, d->trial_nonce);
            writer(signature.has_value(), signature.value_or(0));
            if (axial_kinetic)
            {
                axial_binding = restart_axial_binding(signature, force, physical_fingerprint);
                writer(axial_binding);
            }
            else if ((tagged || forced_pool) && force)
                writer(
                    restart_force_binding(*signature, *force, forced_pool ? pool_configuration : physical_fingerprint));
            writer(model, d->qbmm_legacy_default_only);
            if (kinetic)
                writer(restart_kinetic_policy_fingerprint(d->options));
            if (linear)
                writer(tvd_mc ? qbmm_linear_tvd_mc_revision : qbmm_linear_transport_revision,
                    d->options.qbmm_model.linear_cfl,
                    d->options.qbmm_model.maximum_linear_substeps);
            if (retains_linear_controller)
                writer(static_cast<std::uint64_t>(d->options.qbmm_model.linear_slope_limiter));
            for (const auto& stamp : state.qbmm_families)
            {
                writer(stamp.has_value());
                if (stamp)
                    writer(stamp->catalog_fingerprint, stamp->family_fingerprint, stamp->mass_scale_kg);
            }
            for (const auto& cell : state.cells)
            {
                qbmm_native_law_detail::validate_kinetic_closure(d->options, mesh, cell);
                check(!kinetic || cell.qbmm_kinetic_eos_volumes.size() == cell.phases.size(),
                    "Kinetic QBMM restart EOS traces differ from the physical phase layout.");
                writer(cell.velocity, cell.mechanical_pressure);
                for (size_t a = 0; a < d->eos.size(); ++a)
                {
                    const auto& phase = cell.phases[a];
                    writer(phase.absolute_pressure, phase.internal_energy, phase.temperature, cell.drift_velocity[a]);
                    if (kinetic)
                        writer(cell.qbmm_kinetic_eos_volumes[a]);
                    for (const double moles : phase.moles)
                        writer(moles);
                    const auto& number = cell.phase_numbers[a];
                    const auto& shape = cell.qbmm_shape[a];
                    writer(number.has_value());
                    if (number)
                        writer(*number);
                    writer(shape.has_value());
                    if (shape)
                    {
                        writer(shape->H2, shape->H3);
                        if (kinetic)
                        {
                            const auto& closure = *cell.qbmm_kinetic_closure[a];
                            writer(static_cast<std::uint64_t>(closure.active_atoms), closure.node_velocities[0],
                                closure.node_velocities[1], closure.drift_momentum);
                        }
                    }
                }
            }
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                writer(d->flux.owned_value(static_cast<LO>(f)));
            if (motion)
            {
                check(wall_interval.has_value() == (d->diagnostics.correctors > 0),
                    "QBMM restart wall interval differs from the accepted planar correctors.");
                check(!planar_linear || wall_interval.has_value() == (d->diagnostics.linear_substeps > 0),
                    "QBMM restart wall interval differs from the accepted planar SSPRK2 interval.");
                writer(wall_interval.has_value());
                if (wall_interval)
                {
                    check(wall_interval->new_surface == surface,
                        "QBMM restart last wall interval ends on different accepted geometry.");
                    writer(wall_interval->old_surface, wall_interval->time_step);
                }
            }
            const auto summary_start = writer.bytes.size();
            restart_diagnostics(writer, d->diagnostics, d->eos.size(), catalog.species().size(), version, type);
            auto balances = d->qbmm_publication ? d->qbmm_publication->balances : std::nullopt;
            restart_qbmm_balances(writer, balances, d->diagnostics, catalog);
            if (balances)
                writer(d->qbmm_publication->shape_diagnostics_version);
            if (motion || explicit_family || pool_conversion)
            {
                const bool q9c_pool = pool_conversion && type != RestartPayload::FixedImplicitBulkGasPool;
                const auto start = q9c_pool ? summary_start : writer.bytes.size();
                const auto* live = controller_diagnostics ? controller_diagnostics->saved : nullptr;
                const auto* retained = (explicit_family || pool_conversion) && d->qbmm_publication
                                           ? d->qbmm_publication->controller_diagnostics.get()
                                           : nullptr;
                const auto* saved = live && !tagged && !pool_conversion ? live : retained;
                const bool has_history = saved && saved->correctors > 0;
                check(!force || !tagged || has_history == forced_interval.has_value(),
                    "Forced explicit QBMM restart controller history differs from its impulse interval.");
                check(!axial_kinetic || has_history == (signature.has_value() && d->diagnostics.kinetic_substeps > 0),
                    "Axial kinetic QBMM restart controller history differs from its accepted controller provenance.");
                check(!retains_linear_controller ||
                          has_history == (signature.has_value() && d->diagnostics.linear_substeps > 0),
                    "Tagged linear QBMM restart controller history differs from its accepted controller provenance.");
                if (pool_conversion)
                {
                    check(has_history == (d->diagnostics.correctors > 0) && (!has_history || signature.has_value()),
                        "QBMM pool restart controller history differs from its accepted controller provenance.");
                    check(!has_history || saved->flow.pool_conversion == d->diagnostics.pool_conversion,
                        "QBMM pool restart controller and solver directional diagnostics differ.");
                }
                writer(has_history);
                if (has_history)
                    restart_controller_diagnostics(writer, *saved, d->eos.size(),
                        saved == live ? controller_diagnostics->reaction_count : saved->reaction_extents.size(),
                        d->diagnostics.correctors);
                if (pool_conversion)
                    restart_pool_diagnostics(writer, d->diagnostics, *d->options.qbmm_bulk_gas_pool);
                controller_history = restart_checksum(std::span<const std::byte>{writer.bytes}.subspan(start));
            }
            if (kinetic)
                validate_restart_kinetic_diagnostics(d->options, d->diagnostics);
            if (linear)
                validate_restart_linear_diagnostics(d->options.qbmm_model, d->diagnostics);
            validate_restart_fourier_uncertainty(d->diagnostics);
            if (const auto uncertainty = d->diagnostics.maximum_fourier_gross_comparison_uncertainty_j)
                writer(*uncertainty);
            checksum = restart_checksum(writer.bytes);
        });
    require_uniform_restart_state(mesh, state, d->trial_nonce);
    require_uniform_restart_configuration(
        mesh, d->options, d->eos, state, signature, model, d->qbmm_legacy_default_only);
    if (motion)
        require_uniform_restart_wall_interval(mesh, wall_interval);
    if (motion || explicit_family || pool_conversion)
    {
        collective_detail::require_uniform_value(mesh, controller_history, "QBMM restart controller diagnostics");
    }
    if (pool_conversion)
    {
        collective_detail::require_uniform_value(mesh, pool_configuration, "QBMM restart BulkGas pool configuration");
        if (forced_pool)
            require_uniform_forced_family_history(mesh, *force, d->diagnostics, forced_interval);
    }
    else if (planar_linear)
        collective_detail::require_uniform_value(mesh, planar_linear_controls, "Planar linear QBMM advancement controls");
    else if (tagged)
    {
        collective_detail::require_uniform_value(mesh, int(force.has_value()), "Tagged QBMM acceleration presence");
        collective_detail::require_uniform_value(mesh, physical_fingerprint, "Tagged QBMM physical provider binding");
        if (axial_kinetic)
            collective_detail::require_uniform_value(mesh, axial_binding, "Axial kinetic QBMM controller binding");
        if (force)
            require_uniform_forced_family_history(mesh, *force, d->diagnostics, forced_interval);
    }
    if (kinetic)
        require_uniform_restart_kinetic_diagnostics(mesh, d->diagnostics);
    if (linear)
        require_uniform_restart_linear_diagnostics(mesh, d->diagnostics);
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
    restore_qbmm_restart_impl(payload, nullptr, {}, 0, controller);
}

void LowMachMixtureSolver::restore_qbmm_restart(std::span<const std::byte> payload, PlanarMotion& motion,
    std::string_view target_identity, std::uint64_t target_revision, const ControllerLease* controller)
{
    restore_qbmm_restart_impl(payload, &motion, target_identity, target_revision, controller);
}

void LowMachMixtureSolver::restore_qbmm_restart_impl(std::span<const std::byte> payload, PlanarMotion* motion,
    std::string_view target_identity, std::uint64_t target_revision, const ControllerLease* controller,
    const ControllerRestartDiagnostics* controller_diagnostics)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    const bool kinetic = uses_qbmm_kinetic_transport(d->options.qbmm_model);
    const bool linear = uses_qbmm_linear_transport(d->options.qbmm_model);
    const bool explicit_family = kinetic || linear;
    const auto configured_type =
        configured_restart_payload(d->options, mesh, motion != nullptr, d->active_body_acceleration.has_value());
    const bool pool_conversion = d->options.qbmm_model.bulk_gas_conversion;
    const bool forced_pool = configured_type == RestartPayload::ForcedImplicitBulkGasPool;
    const bool axial_kinetic = configured_type == RestartPayload::FixedAxialKinetic;
    const bool planar_linear = configured_type == RestartPayload::PlanarLinear;
    const bool fixed_tvd_mc = configured_type == RestartPayload::FixedLinearTvdMc;
    const bool tvd_mc = d->options.qbmm_model.linear_slope_limiter == QbmmModelOptions::LinearSlopeLimiter::TvdMc;
    const bool retains_linear_controller = fixed_tvd_mc || planar_linear;
    State candidate;
    Diagnostics statistics;
    statistics.family_transport = d->options.qbmm_model.family_transport;
    std::vector<double> face_flux;
    std::optional<std::uint64_t> controller_signature;
    std::shared_ptr<const QbmmPublication> publication;
    std::uint64_t version = 0, model = 0, saved_generation = 0, checksum = 0, image_checksum = 0;
    std::uint64_t accepted_geometry = 0, motion_configuration = 0, controller_history = 0;
    std::uint64_t physical_fingerprint = 0;
    std::uint64_t axial_binding = 0, pool_configuration = 0;
    double surface = 0, old_cap = 0;
    std::optional<PlanarWallInterval> wall_interval;
    std::optional<ForcedFamilyInterval> forced_interval;
    bool legacy_default_only = false;
    collective_detail::require_uniform_value(mesh, int(motion != nullptr), "QBMM restart receiving geometry mode");
    collective_detail::collective_local_validation(mesh, "QBMM restart decoding",
        [&]
        {
            check(d->trial_nonce != std::numeric_limits<std::uint64_t>::max() && !d->options.qbmm_families.empty() &&
                      (motion || (d->logical_geometry == 0 && !d->active_controller_motion)) &&
                      d->epoch == mesh_geometry_epoch(mesh),
                "QBMM restart target requires QBMM configuration on its current accepted geometry.");
            if (motion)
            {
                d->validate_planar_controller_authority(*motion);
                check(motion->mesh_ptr().get() == d->mesh.get() && !motion->has_active_trial() &&
                          (!d->accepted_planar_motion || d->accepted_planar_motion == motion->d_authority) &&
                          !d->active_body_acceleration && (!explicit_family || planar_linear) &&
                          d->logical_geometry != std::numeric_limits<std::uint64_t>::max(),
                    "Planar QBMM restart requires its live idle motion authority and zero supplied body force.");
                motion_configuration =
                    restart_planar_configuration_fingerprint(*motion, target_identity, target_revision);
            }
            check(payload.size() >= 2 * restart_word_size, "Truncated QBMM restart checksum.");
            RestartReader end(payload.last(2 * restart_word_size));
            end(checksum, image_checksum);
            const auto body = payload.first(payload.size() - 2 * restart_word_size);
            check(restart_checksum(body) == checksum, "QBMM restart data checksum differs.");
            RestartReader reader(body);
            reader.expect(qbmm_restart_magic, "Invalid QBMM restart signature.");
            reader(version);
            const auto type = decode_restart_payload(version, reader);
            check(!is_linear_payload(type) || !linear || (type == RestartPayload::PlanarLinear) != planar_linear ||
                      type == configured_type,
                "QBMM restart linear slope limiter differs.");
            check(type == configured_type,
                "QBMM restart payload is incompatible with the receiving geometry or transport mode.");
            if (pool_conversion)
            {
                const auto header = restart_pool_configuration(d->options, d->eos, type);
                reader.expect_bytes(
                    header.bytes, "QBMM restart BulkGas pool, closure laws or advancement controls differ.");
                pool_configuration = restart_checksum(header.bytes);
                if (forced_pool)
                {
                    reader(statistics.body_acceleration_m_per_s2.emplace());
                    restart_forced_family_impulses(reader, statistics, forced_interval, true);
                }
            }
            else if (planar_linear)
            {
                reader.expect(qbmm_planar_linear_transport_revision, "QBMM restart planar linear algorithm differs.");
                reader.expect(restart_advance_controls(d->options), "QBMM restart advancement controls differ.");
            }
            else if (version == qbmm_tagged_restart_version)
            {
                bool has_force = false;
                Vector force{};
                reader(has_force, force);
                check(has_force || ((axial_kinetic || fixed_tvd_mc) && force == Vector{}),
                    "QBMM restart has an invalid supplied acceleration record.");
                if (has_force)
                    statistics.body_acceleration_m_per_s2 = force;
                reader.expect(kinetic ? balanced_kinetic_pressure_revision : balanced_linear_pressure_revision,
                    "QBMM restart pressure algorithm differs.");
                reader.expect(kinetic ? forced_kinetic_thermal_revision : forced_linear_thermal_revision,
                    "QBMM restart thermal algorithm differs.");
                reader.expect(restart_advance_controls(d->options), "QBMM restart advancement controls differ.");
                if (has_force || axial_kinetic)
                    physical_fingerprint = restart_forced_physical_fingerprint(
                        d->options, d->eos, d->options.qbmm_model.family_transport, axial_kinetic);
                reader.expect(physical_fingerprint, "Tagged QBMM restart physical provider binding differs.");
                restart_forced_family_impulses(reader, statistics, forced_interval, has_force);
            }
            if ((explicit_family || pool_conversion) && d->explicit_family_force_recorded)
                check(d->diagnostics.body_acceleration_m_per_s2 == statistics.body_acceleration_m_per_s2,
                    "QBMM restart body acceleration differs from the accepted configuration.");
            if ((explicit_family || pool_conversion) && d->controller_active)
                check(d->active_body_acceleration == acceleration_components(statistics.body_acceleration_m_per_s2),
                    "QBMM restart body acceleration differs from the receiving controller.");
            qbmm_detail::validate_capability(d->options, mesh, motion != nullptr,
                explicit_family || pool_conversion ? statistics.body_acceleration_m_per_s2
                                                   : acceleration_vector(d->active_body_acceleration));
            if (statistics.body_acceleration_m_per_s2)
                d->validate_forced_linear_geometry();
            if (motion)
            {
                reader(accepted_geometry);
                reader.expect(motion_configuration, "QBMM restart reference geometry, options or target law differs.");
                reader.expect(restart_ordered_maps_fingerprint(mesh), "QBMM restart ordered native maps differ.");
                reader(surface);
            }
            else
                reader.expect(restart_partition_fingerprint(mesh), "QBMM restart mesh geometry or partition differs.");
            if (axial_kinetic)
                reader.expect(restart_ordered_maps_fingerprint(mesh), "QBMM restart ordered native maps differ.");
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
            if (pool_conversion)
                check(!d->accepted_qbmm_controller || controller_signature == d->accepted_qbmm_controller,
                    "QBMM pool restart controller signature differs from the accepted configuration.");
            if (axial_kinetic || fixed_tvd_mc)
            {
                const auto& force = statistics.body_acceleration_m_per_s2;
                check(!force || controller_signature,
                    "Forced QBMM restart lacks its controller signature.");
                // require_controller has authenticated the lease. Only the controller that owns the
                // accepted history can explicitly replace it with a compatible standalone state.
                const bool standalone_rollback = !controller_signature && !force && controller &&
                                                 controller->reacting_source_authority && d->active_qbmm_controller &&
                                                 d->active_qbmm_controller == d->accepted_qbmm_controller;
                check(!d->accepted_qbmm_controller || controller_signature == d->accepted_qbmm_controller ||
                          standalone_rollback,
                    "QBMM restart controller signature differs from the accepted configuration.");
            }
            if (axial_kinetic)
            {
                const auto& force = statistics.body_acceleration_m_per_s2;
                axial_binding = restart_axial_binding(controller_signature, force, physical_fingerprint);
                reader.expect(axial_binding, "Axial kinetic QBMM restart force and controller binding differs.");
            }
            else if (statistics.body_acceleration_m_per_s2)
            {
                check(controller_signature.has_value(), "Forced QBMM restart lacks its controller signature.");
                check(!d->has_forced_family_provenance() || controller_signature == d->accepted_qbmm_controller,
                    "QBMM restart controller signature differs from the accepted forced explicit configuration.");
                reader.expect(restart_force_binding(*controller_signature, *statistics.body_acceleration_m_per_s2,
                                  forced_pool ? pool_configuration : physical_fingerprint),
                    "QBMM restart force and controller binding differ.");
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
            if (kinetic)
                reader.expect(restart_kinetic_policy_fingerprint(d->options),
                    "QBMM restart kinetic CFL, substep cap or ordered node-slip law differs.");
            if (linear)
            {
                reader.expect(tvd_mc ? qbmm_linear_tvd_mc_revision : qbmm_linear_transport_revision,
                    "QBMM restart linear transport revision differs.");
                double cfl = 0.;
                int substeps = 0;
                reader(cfl, substeps);
                const auto& linear_model = d->options.qbmm_model;
                check(cfl == linear_model.linear_cfl && substeps == linear_model.maximum_linear_substeps,
                    "QBMM restart linear CFL or substep cap differs.");
                if (retains_linear_controller)
                    reader.expect(static_cast<std::uint64_t>(linear_model.linear_slope_limiter),
                        "QBMM restart linear slope limiter differs.");
            }
            check(!legacy_default_only || !d->controller_active || d->active_legacy_compatible,
                "Legacy QBMM restart requires default force and constitutive-law configuration.");
            check((!motion && !d->controller_active) || ((axial_kinetic || fixed_tvd_mc) && !controller_signature) ||
                      d->active_qbmm_controller == controller_signature,
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
                if (kinetic)
                {
                    cell.qbmm_kinetic_closure.resize(phases);
                    cell.qbmm_kinetic_eos_volumes.resize(phases);
                }
                for (size_t a = 0; a < phases; ++a)
                {
                    auto& phase = cell.phases[a];
                    phase.thermodynamics = d->eos[a];
                    phase.moles.resize(species);
                    reader(phase.absolute_pressure, phase.internal_energy, phase.temperature, cell.drift_velocity[a]);
                    if (kinetic)
                        reader(cell.qbmm_kinetic_eos_volumes[a]);
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
                        if (kinetic)
                        {
                            auto& closure = cell.qbmm_kinetic_closure[a].emplace();
                            std::uint64_t active_atoms = 0;
                            reader(active_atoms, closure.node_velocities[0], closure.node_velocities[1],
                                closure.drift_momentum);
                            check(active_atoms <= closure.node_velocities.size(), "Invalid QBMM restart atom count.");
                            closure.active_atoms = static_cast<std::size_t>(active_atoms);
                        }
                    }
                }
            }
            face_flux.resize(faces);
            for (double& rate : face_flux)
                reader(rate);
            if (motion)
            {
                bool has_interval = false;
                reader(has_interval);
                if (has_interval)
                {
                    auto& interval = wall_interval.emplace();
                    interval.new_surface = surface;
                    reader(interval.old_surface, interval.time_step);
                    const auto displacement = surface - interval.old_surface;
                    const auto limit = motion->options().maximum_level_change;
                    check(interval.time_step > 0. && std::isfinite(displacement) &&
                              (!limit || std::abs(displacement) <= *limit),
                        "QBMM restart wall interval has an invalid duration or displacement.");
                    const auto edges = motion->candidate_axis_edges(interval.old_surface);
                    check(!edges.empty(), "QBMM restart lacks reference edges for the last wall interval.");
                    for (size_t i = 0; i < edges.size(); ++i)
                        check(std::isfinite(edges[i]) && (i == 0 || edges[i] > edges[i - 1]),
                            "QBMM restart previous wall geometry has nonfinite or unordered edges.");
                    old_cap = edges.back();
                }
            }
            const auto summary_remaining = reader.remaining_size();
            restart_diagnostics(reader, statistics, phases, species, version, type);
            if (statistics.body_acceleration_m_per_s2)
                validate_forced_family_impulses(
                    *statistics.body_acceleration_m_per_s2, statistics, candidate, forced_interval);
            auto metadata = std::make_shared<QbmmPublication>();
            metadata->origin = QbmmStateOrigin::PortableRestart;
            metadata->archived_generation = saved_generation;
            metadata->planar_wall_interval = wall_interval;
            metadata->forced_family_interval = forced_interval;
            check(!motion || wall_interval.has_value() == (statistics.correctors > 0),
                "QBMM restart wall interval differs from the accepted planar correctors.");
            check(!planar_linear || wall_interval.has_value() == (statistics.linear_substeps > 0),
                "QBMM restart wall interval differs from the accepted planar SSPRK2 interval.");
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
            auto* restored = controller_diagnostics ? controller_diagnostics->restored : nullptr;
            std::shared_ptr<ReactingMixtureCoupler::Diagnostics> archived_controller;
            if (motion || explicit_family || pool_conversion)
            {
                const bool q9c_pool = pool_conversion && type != RestartPayload::FixedImplicitBulkGasPool;
                const auto before = q9c_pool ? summary_remaining : reader.remaining_size();
                bool has_history = false;
                reader(has_history);
                check(!statistics.body_acceleration_m_per_s2 || has_history == forced_interval.has_value(),
                    "Forced explicit QBMM restart controller history differs from its impulse interval.");
                check(!axial_kinetic ||
                          has_history == (controller_signature.has_value() && statistics.kinetic_substeps > 0),
                    "Axial kinetic QBMM restart controller history differs from its accepted controller provenance.");
                check(!retains_linear_controller ||
                          has_history == (controller_signature.has_value() && statistics.linear_substeps > 0),
                    "Tagged linear QBMM restart controller history differs from its accepted controller provenance.");
                check(!pool_conversion || (has_history == (statistics.correctors > 0) &&
                                              (!has_history || controller_signature.has_value())),
                    "QBMM pool restart controller history differs from its accepted controller provenance.");
                if (has_history)
                {
                    check(controller_signature && (explicit_family || pool_conversion || restored),
                        "QBMM restart controller diagnostics require matching reacting authority.");
                    std::optional<size_t> expected_reactions;
                    if (controller_diagnostics)
                        expected_reactions = controller_diagnostics->reaction_count;
                    if (explicit_family || pool_conversion)
                        archived_controller = std::make_shared<ReactingMixtureCoupler::Diagnostics>();
                    auto& diagnostics = archived_controller ? *archived_controller : *restored;
                    restart_controller_diagnostics(
                        reader, diagnostics, phases, expected_reactions, statistics.correctors);
                }
                if (pool_conversion)
                    restart_pool_diagnostics(reader, statistics, *d->options.qbmm_bulk_gas_pool);
                controller_history =
                    restart_checksum(body.subspan(body.size() - before, before - reader.remaining_size()));
            }
            if (version >= 5 && reader.remaining_size() == restart_word_size)
                reader(statistics.maximum_fourier_gross_comparison_uncertainty_j.emplace());
            check(reader.remaining_size() == 0, "Unexpected trailing QBMM restart data.");
            validate_restart_fourier_uncertainty(statistics);
            qbmm_detail::validate_state(d->options, candidate);
            if (pool_conversion)
            {
                const bool has_chemistry = controller_diagnostics && controller_diagnostics->validate_pool_state;
                check(has_chemistry || (!d->controller_active && statistics.correctors == 0),
                    "Coupled QBMM pool restart requires its receiving chemistry authority.");
                if (has_chemistry)
                    controller_diagnostics->validate_pool_state(controller_diagnostics->source_context, candidate);
            }
            if (kinetic)
            {
                validate_restart_kinetic_diagnostics(d->options, statistics);
                for (const auto& cell : candidate.cells)
                    qbmm_native_law_detail::validate_kinetic_closure(d->options, mesh, cell);
            }
            if (linear)
                validate_restart_linear_diagnostics(d->options.qbmm_model, statistics);
            if (archived_controller)
            {
                archived_controller->flow = statistics;
                if (restored)
                    *restored = *archived_controller;
                metadata->controller_diagnostics = std::move(archived_controller);
            }
            else if (restored)
                restored->flow = statistics;
            publication = std::move(metadata);
        });
    require_uniform_restart_state(mesh, candidate, saved_generation);
    require_uniform_restart_configuration(
        mesh, d->options, d->eos, candidate, controller_signature, model, legacy_default_only);
    collective_detail::require_uniform_value(mesh, version, "QBMM restart archive version");
    collective_detail::require_uniform_value(
        mesh, static_cast<std::uint64_t>(configured_type), "QBMM restart payload type");
    if (pool_conversion)
    {
        collective_detail::require_uniform_value(mesh, pool_configuration, "QBMM restart BulkGas pool configuration");
        if (forced_pool)
            require_uniform_forced_family_history(
                mesh, *statistics.body_acceleration_m_per_s2, statistics, forced_interval);
    }
    if (version == qbmm_tagged_restart_version && !planar_linear)
    {
        collective_detail::require_uniform_value(mesh, int(statistics.body_acceleration_m_per_s2.has_value()),
            "Tagged QBMM acceleration presence");
        collective_detail::require_uniform_value(mesh, physical_fingerprint, "Tagged QBMM physical provider binding");
        if (axial_kinetic)
            collective_detail::require_uniform_value(mesh, axial_binding, "Axial kinetic QBMM controller binding");
        if (statistics.body_acceleration_m_per_s2)
            require_uniform_forced_family_history(
                mesh, *statistics.body_acceleration_m_per_s2, statistics, forced_interval);
    }
    require_uniform_restart_fourier_uncertainty(mesh, statistics);
    collective_detail::require_uniform_value(mesh, image_checksum, "QBMM restart image checksum");
    std::uint64_t actual_image_checksum = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &checksum, &actual_image_checksum);
    check(actual_image_checksum == image_checksum, "QBMM restart ranks do not belong to the same saved image.");
    const auto generation = std::max(d->trial_nonce, saved_generation) + 1;
    if (motion || explicit_family || pool_conversion)
        collective_detail::require_uniform_value(mesh, controller_history, "QBMM restart controller diagnostics");
    if (kinetic)
        require_uniform_restart_kinetic_diagnostics(mesh, statistics);
    if (linear)
        require_uniform_restart_linear_diagnostics(mesh, statistics);
    if (motion)
    {
        collective_detail::require_uniform_value(mesh, motion_configuration, "QBMM restart planar configuration");
        collective_detail::require_uniform_value(mesh, surface, "QBMM restart accepted surface elevation");
        require_uniform_restart_wall_interval(mesh, wall_interval);
        auto accepted_frame = capture_planar_motion_frame(*motion);
        try
        {
            motion->stage_restart_geometry(surface, d->preserves_planar_trial());
            collective_detail::collective_local_validation(mesh, "Planar QBMM restart physical state",
                [&]
                {
                    check(accepted_geometry == restart_partition_fingerprint(mesh),
                        "QBMM restart accepted planar geometry or ordered partition differs.");
                    const RestartPlanarWalls walls{static_cast<size_t>(motion->options().axis), old_cap,
                        wall_interval ? wall_interval->time_step : 0.};
                    validate_restart_physics(mesh, d->options, candidate, face_flux, walls);
                });
            d->refresh_numeric_geometry();
            d->stage_publication_fields(candidate, face_flux);
            motion->accept_trial();
        }
        catch (...)
        {
            const auto failure = std::current_exception();
            rollback_planar_geometry(*motion, accepted_frame);
            std::rethrow_exception(failure);
        }
        d->publish_staged_fields();
        d->accepted = std::move(candidate);
        d->diagnostics = std::move(statistics);
        d->qbmm_publication = std::move(publication);
        d->initialized = true;
        d->trial_nonce = generation;
        d->accepted_planar_motion = motion->d_authority;
        ++d->logical_geometry;
    }
    else
    {
        collective_detail::collective_local_validation(mesh, "QBMM restart physical state",
            [&] { validate_restart_physics(mesh, d->options, candidate, face_flux); });
        d->publish_fixed(std::move(candidate), std::move(statistics), face_flux, generation, std::move(publication));
    }
    d->explicit_family_force_recorded = explicit_family || pool_conversion;
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
            check(!d->options.qbmm_model.bulk_gas_conversion,
                "QBMM pool conversion requires initialization or a v10 restart.");
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
            const auto transport = d->options.qbmm_model.family_transport;
            const bool axial_kinetic = qbmm_detail::uses_axial_kinetic_transport(d->options, mesh);
            if (axial_kinetic ||
                (saved.count_only_body_acceleration && uses_qbmm_explicit_transport(d->options.qbmm_model)))
                check(restart_forced_physical_fingerprint(d->options, d->eos, transport, axial_kinetic) ==
                          restart_forced_physical_fingerprint(prior.options, prior.eos, transport, axial_kinetic),
                    "Explicit QBMM migration requires matching physical provider bindings.");
            qbmm_detail::validate_capability(
                d->options, mesh, false, acceleration_vector(saved.count_only_body_acceleration));
            check(saved.generation != 0 && saved.generation <= prior.trial_nonce &&
                      saved.generation != std::numeric_limits<std::uint64_t>::max() &&
                      std::isfinite(saved.state.time) &&
                      saved.state.thermodynamic_pressure >= d->options.minimum_pressure &&
                      saved.state.thermodynamic_pressure <= d->options.maximum_pressure,
                "QBMM migration checkpoint time, pressure or generation is invalid.");
            candidate = saved.state;
            statistics = saved.diagnostics;
            statistics.body_acceleration_m_per_s2 = acceleration_vector(saved.count_only_body_acceleration);
            statistics.impulse_parts.reset(); // The migrated stock has no accepted QBMM interval.
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
            // A new kinetic integrator needs its initial endpoint closure on the migrated moments.
            validate_restart_physics(
                mesh, d->options, candidate, saved.face_flux, {}, uses_qbmm_kinetic_transport(d->options.qbmm_model));
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
    d->explicit_family_force_recorded = uses_qbmm_explicit_transport(d->options.qbmm_model);
    d->qbmm_legacy_default_only = false;
}
