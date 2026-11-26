/** @file LowMachQbmmState.hh
 * @brief Private validation of native QBMM phase bindings and canonical stock.
 */
#pragma once
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/details/LowMachQbmmExplicitTransport.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace SimpleFluid::Reacting::qbmm_detail
{
using Flow = LowMachMixtureSolver;

SIMPLEFLUID_SOLVERS_LOCAL inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_configuration(
    const Flow::Options& options, std::span<const std::shared_ptr<const MixtureThermodynamics>> eos)
{
    const auto catalog = eos.front()->catalog().fingerprint();
    const auto validate_identity = [catalog](const ThermalLawIdentity& law)
    {
        require(!law.identity.empty() && law.revision > 0 && law.schema_fingerprint == catalog,
            "QBMM physical law identity needs an immutable name, positive revision and matching catalog.");
    };
    require(options.qbmm_eos_law_identities.empty() || options.qbmm_eos_law_identities.size() == eos.size(),
        "QBMM EOS law identities must use native physical phase order.");
    for (const auto& law : options.qbmm_eos_law_identities)
        if (law)
            validate_identity(*law);
    if (options.qbmm_mixture_viscosity_identity)
    {
        require(bool(options.mixture_dynamic_viscosity), "QBMM viscosity identity requires a supplied mixture law.");
        validate_identity(*options.qbmm_mixture_viscosity_identity);
    }
    const auto& families = options.qbmm_families;
    const auto& model = options.qbmm_model;
    const bool kinetic = uses_qbmm_kinetic_transport(model);
    const bool linear = uses_qbmm_linear_transport(model);
    require(model.transport.mode == FVM::AssemblyMode::Transient &&
                model.transport.time == FVM::ScalarTimeScheme::BackwardEuler &&
                model.transport.convection == FVM::ScalarConvectionScheme::Upwind,
        "Native QBMM transport requires transient backward Euler and first-order upwind.");
    const auto& node_laws = options.qbmm_node_slip_laws;
    require(kinetic ? !families.empty() && node_laws.size() == families.size() && std::isfinite(model.kinetic_cfl) &&
                          model.kinetic_cfl > 0 && model.kinetic_cfl <= 1 && model.maximum_kinetic_substeps > 0
                    : node_laws.empty(),
        "Kinetic QBMM requires an EOS-indexed node-law layout, a CFL in (0, 1], and a positive substep cap.");
    require(!linear || (!families.empty() && std::isfinite(model.linear_cfl) && model.linear_cfl > 0 &&
                           model.linear_cfl <= 0.5 && model.maximum_linear_substeps > 0),
        "Realizable linear QBMM requires a bound family, a CFL in (0, 0.5], and a positive substep cap.");
    if (families.empty())
        return;
    require(families.size() == eos.size() && options.phase_roles.size() == eos.size(),
        "QBMM family bindings must match the native EOS phase layout.");
    require(
        options.thermal_mode == PhaseThermalMode::Equilibrium, "Native QBMM requires equilibrium phase temperatures.");
    require(std::ranges::any_of(families, [](const auto& family) { return bool(family); }),
        "A native QBMM phase layout requires at least one bound family.");
    for (size_t a = 0; a < families.size(); ++a)
    {
        const auto& family = families[a];
        require(!kinetic || bool(node_laws[a]) == bool(family),
            "Kinetic QBMM requires one node-slip law per bound family.");
        if (!family)
            continue;
        const auto& role = options.phase_roles[a];
        require(role.role == Flow::PhaseRole::DispersedGas && role.identity == family->population() && eos[a] &&
                    eos[a]->kind() == EOSKind::Gas &&
                    eos[a]->catalog().fingerprint() == family->catalog().fingerprint(),
            "A native QBMM family must bind to its named dispersed-gas EOS phase and ordered catalog.");
        const auto& declared = family->closure_versions().laws;
        require(std::ranges::none_of(declared, qbmm_uses_bulk_gas_pool),
            "QBMM BulkGas pool law contracts are available only in the pure event kernel.");
        const auto node_law_count = std::ranges::count(declared, QbmmLawKind::NodeSlip, &QbmmLawMetadata::kind);
        require(kinetic ? node_law_count == 1 && node_laws[a]->metadata.kind == QbmmLawKind::NodeSlip &&
                              family->declares_law(node_laws[a]->metadata) && bool(node_laws[a]->evaluate) &&
                              std::ranges::none_of(
                                  declared, [](const auto& law) { return law.kind == QbmmLawKind::PhaseSlip; })
                        : node_law_count == 0,
            "A kinetic family requires a single matching node-slip contract and evaluator.");
    }
}

/** Check the admitted physics, topology and transport combinations on every owner.
 * Construction can precede selection of a reacting controller. Its immutable node-law dependencies must still agree.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_capability(const Flow::Options& options, const Flow::Mesh& mesh,
    bool moving, std::optional<Flow::Vector> force, bool force_known = true)
{
    const auto& model = options.qbmm_model;
    const bool kinetic = uses_qbmm_kinetic_transport(model), linear = uses_qbmm_linear_transport(model);
    using LinearSlopeLimiter = QbmmModelOptions::LinearSlopeLimiter;
    require(model.linear_slope_limiter == LinearSlopeLimiter::RealizabilityOnly ||
                (linear && model.linear_slope_limiter == LinearSlopeLimiter::TvdMc),
        "QBMM requires a valid linear slope limiter, and TVD-MC requires RealizableLinearSSPRK2.");
    const bool axial_kinetic = uses_axial_kinetic_transport(options, mesh);
    using Velocity = QbmmModelOptions::Velocity;
    require((kinetic && model.velocity == Velocity::NodeDependentSlip) ||
                (linear && model.velocity == Velocity::SharedCarrier) ||
                (model.family_transport == QbmmModelOptions::FamilyTransport::NativeImplicit &&
                    (model.velocity == Velocity::SharedCarrier || model.velocity == Velocity::SuppliedPhaseSlip)),
        "Native QBMM requires a compatible family integrator and velocity mode.");
    require(!model.bulk_gas_conversion, "Runtime QBMM BulkGas conversion is deferred to Phase Q9b.");
    require(!model.variable_composition && !model.cross_family_coalescence,
        "Native QBMM requires fixed family composition and within-family population events.");
    require(!model.capillarity && !model.independent_node_temperatures,
        "Native QBMM requires common pressure and temperature with zero surface tension.");
    const auto validate_force = [axial_kinetic](const Flow::Vector& acceleration)
    {
        require(std::isfinite(acceleration.x) && std::isfinite(acceleration.y) && std::isfinite(acceleration.z),
            "Supplied body acceleration must be finite in every component.");
        if (axial_kinetic)
            require_axial(acceleration, "Cylindrical kinetic QBMM requires axial body acceleration.");
    };
    if (force)
        validate_force(*force);
    if (moving)
    {
        require(!kinetic && !force, "Planar ALE excludes kinetic transport and supplied body acceleration.");
        mesh.visit(
            [](const auto& native)
            {
                using Native = std::remove_cvref_t<decltype(native)>;
                require(std::is_same_v<Native, Flow::Mesh::Cartesian> ||
                            std::is_same_v<Native, Flow::Mesh::Cylindrical> ||
                            std::is_same_v<Native, Flow::Mesh::SemiStructured>,
                    "Low-Mach planar ALE requires native orthogonal mapped geometry.");
                if constexpr (requires { native.indexer().periodic_dimensions[0]; })
                    for (size_t axis = 0; axis < 3; ++axis)
                        require(!native.indexer().periodic_dimensions[axis], "Low-Mach planar ALE requires closed topology.");
                if constexpr (requires { native.indexer().axial_periodic; })
                    require(!native.indexer().axial_periodic, "Low-Mach planar ALE requires closed axial topology.");
            });
    }
    std::optional<Flow::Vector> declared_force;
    for (const auto& law : options.qbmm_node_slip_laws)
        if (law && law->required_body_acceleration_m_per_s2)
        {
            const auto& dependency = law->required_body_acceleration_m_per_s2;
            validate_force(*dependency);
            require((!declared_force || declared_force == dependency) && (!force_known || force == dependency),
                "QBMM node-slip acceleration dependencies must match the supplied body acceleration.");
            declared_force = dependency;
        }
    if (!kinetic && !linear)
        return;
    mesh.visit(
        [axial_kinetic, moving](const auto& native)
        {
            using Native = std::remove_cvref_t<decltype(native)>;
            if constexpr (std::is_same_v<Native, Flow::Mesh::Cartesian>)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                    require(!native.indexer().periodic_dimensions[axis],
                        "Explicit QBMM requires closed Cartesian topology.");
            }
            else if constexpr (std::is_same_v<Native, Flow::Mesh::Cylindrical>)
            {
                require(!native.indexer().periodic_dimensions[Native::R] &&
                            !native.indexer().periodic_dimensions[Native::AXIAL],
                    "Explicit cylindrical QBMM requires closed radial and axial topology.");
            }
            else if constexpr (std::is_same_v<Native, Flow::Mesh::SemiStructured>)
            {
                require(
                    !native.indexer().axial_periodic, "Semi-structured explicit QBMM requires closed axial topology.");
                if (axial_kinetic || moving)
                    qbmm_explicit_detail::validate_cylindrical_base(native);
            }
            else
                require(false, "Explicit QBMM requires a supported native orthogonal mesh.");
        });
    if (axial_kinetic)
    {
        qbmm_explicit_detail::validate_axial_kinetic_geometry(mesh);
        return;
    }
    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<Flow::Pack::local_ordinal_type>(f);
        require(!mesh.is_geometry_exterior_face(face) || mesh.is_boundary_face(face),
            "Explicit QBMM requires physical walls at geometric exterior faces.");
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_stamps(const Flow::Options& options, const Flow::State& state)
{
    require(state.qbmm_families.size() == options.qbmm_families.size(),
        "Native QBMM family stamps must match the configured phase layout.");
    for (size_t a = 0; a < options.qbmm_families.size(); ++a)
    {
        const auto& family = options.qbmm_families[a];
        const auto& stamp = state.qbmm_families[a];
        require(family ? stamp && *stamp == Flow::qbmm_family_stamp(*family) : !stamp,
            "Native QBMM catalog, family, or mass-scale stamp differs from the configured phase.");
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline std::vector<std::optional<InversionStatus>> validate_cell(
    const Flow::Options& options, const Flow::CellState& cell)
{
    const auto& families = options.qbmm_families;
    require(
        cell.qbmm_shape.size() == families.size(), "Native QBMM shape entries must match the configured phase layout.");
    if (families.empty())
        return {};
    require(cell.phases.size() == families.size() && cell.phase_numbers.size() == families.size() &&
                cell.drift_velocity.size() == families.size(),
        "Native QBMM cell phase, population number, or velocity layout differs.");
    std::vector<std::optional<InversionStatus>> statuses(families.size());
    for (size_t a = 0; a < families.size(); ++a)
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const double drift = cell.drift_velocity[a].component(axis);
            require(std::isfinite(drift), "Native QBMM supplied phase slip must be finite.");
            require(drift == 0 || options.qbmm_model.velocity == QbmmModelOptions::Velocity::SuppliedPhaseSlip ||
                        (uses_qbmm_kinetic_transport(options.qbmm_model) && !families[a]),
                "Native QBMM nonzero phase slip requires the explicit SuppliedPhaseSlip velocity mode.");
        }
        const auto& family = families[a];
        const auto& shape = cell.qbmm_shape[a];
        require(bool(shape) == bool(family), "Native QBMM requires shape precisely for each bound family.");
        if (!family)
            continue;
        const auto& number = cell.phase_numbers[a];
        const auto& phase = cell.phases[a];
        require(number && phase.thermodynamics && phase.thermodynamics->kind() == EOSKind::Gas,
            "A native QBMM family requires an engaged native population number and gas EOS.");
        const auto quadrature = invert_qbmm_moments(
            qbmm_cell_moments(*family, {options.phase_roles[a].identity, phase.thermodynamics->catalog().fingerprint(),
                                           phase.moles, *number, *shape, family->fingerprint()}));
        if (!quadrature.valid())
            throw std::invalid_argument("Native QBMM family '" + family->population() + "' failed inversion (" +
                                        qbmm_inversion_status_name(quadrature.status) + ").");
        statuses[a] = quadrature.status;
    }
    return statuses;
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_state(const Flow::Options& options, const Flow::State& state)
{
    validate_stamps(options, state);
    for (const auto& cell : state.cells)
        (void) validate_cell(options, cell);
}

/** The transport owner invokes this before publishing a snapshot or evaluating sources.
 * Kept distinct from scalar balance checks: nonnegative scalars and globally
 * conserved totals do not establish fixed composition or moment realizability.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_transport(const Flow::Options& options, const Flow::State& state)
{
    validate_state(options, state);
}

// The caller first validates the native species and count packet layouts.
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_sources(
    const Flow::Options& options, std::span<const Flow::CellSources> sources, bool coherent = false)
{
    const auto& families = options.qbmm_families;
    for (const auto& source : sources)
    {
        require(source.qbmm_shape_changes.empty() || source.qbmm_shape_changes.size() == families.size(),
            "Native QBMM shape-source layout differs from the configured family layout.");
        for (size_t a = 0; a < families.size(); ++a)
        {
            if (!source.qbmm_shape_changes.empty())
            {
                const auto& shape = source.qbmm_shape_changes[a];
                require(!shape || (families[a] && std::isfinite(shape->H2) && std::isfinite(shape->H3) &&
                                      (coherent || (shape->H2 == 0 && shape->H3 == 0))),
                    "Native QBMM shape sources require a bound phase and a coherent source receipt for changes.");
            }
            if (!families[a] || coherent)
                continue;
            require(source.species_moles.empty() ||
                        std::ranges::all_of(source.species_moles[a], [](double change) { return change == 0; }),
                "Native QBMM currently requires zero material sources on bound phases.");
            require(source.population_number_changes.empty() || source.population_number_changes[a] == 0,
                "Native QBMM currently requires zero population number sources on bound phases.");
        }
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_raw_value(long double actual, long double expected, const char* message)
{
    require(std::isfinite(actual) && std::isfinite(expected) && (actual == 0) == (expected == 0) &&
                std::abs(actual - expected) <=
                    qbmm_reconstruction_tolerance * std::max(std::abs(actual), std::abs(expected)),
        message);
}

using SourceReceiptSum = SimpleFluid::detail::CompensatedSum<long double>;

struct SourceReceiptEndpoint
{
    std::span<const double> moles;
    double count;
    QbmmShapeState shape;
};

struct SourceReceiptGross
{
    SourceReceiptSum births, deaths, coalescence_pairs, breakup_parents, incoming_mass, outgoing_mass;
    SourceReceiptSum produced_H2, consumed_H2, produced_H3, consumed_H3;
    std::array<SourceReceiptSum, 8> reservoir_activity{};

    void add(const QbmmEventReceipt& event, const SpeciesCatalog& catalog)
    {
        for (const auto& allocation : event.transfers)
        {
            (allocation.into_family ? incoming_mass : outgoing_mass) +=
                source_stage_detail::mass(allocation.transfer.moles, catalog);
            if (const auto* reservoir = std::get_if<QbmmReservoirExchange>(&allocation.transfer.counterpart))
            {
                const std::array channels{
                    reservoir->energy_j, reservoir->momentum[0], reservoir->momentum[1], reservoir->momentum[2]};
                for (size_t i = 0; i < channels.size(); ++i)
                    reservoir_activity[2 * i + (channels[i] < 0)] += std::abs(channels[i]);
            }
        }
        switch (event.kind)
        {
            case QbmmEventKind::Birth:
                births += event.gross_created[0];
                break;
            case QbmmEventKind::Disappearance:
                deaths += event.gross_consumed[0];
                break;
            case QbmmEventKind::Coalescence:
                coalescence_pairs += event.gross_created[0];
                break;
            case QbmmEventKind::Breakup:
                breakup_parents += event.gross_consumed[0];
                break;
            case QbmmEventKind::Growth:
            case QbmmEventKind::Shrinkage:
                break;
        }
        produced_H2 += event.gross_created[2];
        consumed_H2 += event.gross_consumed[2];
        produced_H3 += event.gross_created[3];
        consumed_H3 += event.gross_consumed[3];
    }

    void append(std::vector<double>& row) const
    {
        const auto append_gross = [&](const SourceReceiptSum& sum)
        {
            const double narrowed = static_cast<double>(sum.value());
            require(narrowed >= 0, "QBMM gross activity must be nonnegative.");
            validate_raw_value(narrowed, sum.value(), "QBMM gross activity is not representable.");
            row.push_back(narrowed);
        };
        for (const auto& sum : {births, deaths, coalescence_pairs, breakup_parents, incoming_mass, outgoing_mass,
                 produced_H2, consumed_H2, produced_H3, consumed_H3})
            append_gross(sum);
        for (const auto& sum : reservoir_activity)
            append_gross(sum);
    }
};

struct SourceReceiptReplay
{
    SourceReceiptGross gross;
    std::vector<SourceReceiptSum> species_changes, species_stocks;
    std::array<SourceReceiptSum, 2> event_shape_change;

    explicit SourceReceiptReplay(std::span<const double> moles)
        : species_changes(moles.size()), species_stocks(moles.size())
    {
        for (size_t s = 0; s < moles.size(); ++s)
            species_stocks[s] += moles[s];
    }
};

SIMPLEFLUID_SOLVERS_LOCAL inline QbmmEventMoments source_atom_moments(long double count, double size)
{
    QbmmEventMoments moments{count};
    for (size_t i = 1; i < moments.size(); ++i)
        moments[i] = moments[i - 1] * size;
    return moments;
}

/** Reconstruct the gross moments and signed shape evidence used across accepted stages. */
SIMPLEFLUID_SOLVERS_LOCAL inline void replay_source_event_moments(
    const QbmmEventReceipt& event, std::array<SourceReceiptSum, 2>& changes)
{
    std::array<SourceReceiptSum, 4> consumed, created, signed_moments;
    const bool use_atoms = event.kind == QbmmEventKind::Birth || event.kind == QbmmEventKind::Breakup;
    for (const auto& donor : event.donors)
    {
        const auto moments = source_atom_moments(donor.atom.count, donor.atom.mass_over_scale);
        for (size_t k = 0; k < moments.size(); ++k)
        {
            consumed[k] += moments[k];
            signed_moments[k] += -moments[k];
        }
        if (use_atoms)
            for (size_t k = 0; k < changes.size(); ++k)
                changes[k] += -moments[k + 2];
    }
    for (const auto& product : event.products)
    {
        const auto moments = source_atom_moments(product.count, product.mass_over_scale);
        for (size_t k = 0; k < moments.size(); ++k)
        {
            created[k] += moments[k];
            signed_moments[k] += moments[k];
        }
        if (use_atoms)
            for (size_t k = 0; k < changes.size(); ++k)
                changes[k] += moments[k + 2];
    }
    for (size_t k = 0; k < signed_moments.size(); ++k)
    {
        require(consumed[k].value() >= 0 && created[k].value() >= 0, "QBMM gross atom moments must be nonnegative.");
        validate_raw_value(event.gross_consumed[k], consumed[k].value(), "QBMM gross donor moments differ.");
        validate_raw_value(event.gross_created[k], created[k].value(), "QBMM gross product moments differ.");
        const auto scale = std::max(consumed[k].value(), created[k].value());
        require(std::isfinite(event.moment_change[k]) && std::abs(event.moment_change[k] - signed_moments[k].value()) <=
                                                             qbmm_reconstruction_tolerance * scale,
            "QBMM signed event moments disagree with the supplied atom allocations.");
    }
    if (use_atoms)
        return;

    std::array<long double, 2> shape_change{signed_moments[2].value(), signed_moments[3].value()};
    if (event.kind == QbmmEventKind::Growth || event.kind == QbmmEventKind::Shrinkage)
    {
        require(event.donors.size() == 1 && event.products.size() == 1,
            "QBMM resizing requires one donor and one product.");
        const auto& donor = event.donors.front().atom;
        const auto& product = event.products.front();
        const long double a = donor.mass_over_scale, b = product.mass_over_scale;
        require(donor.count == product.count && (event.kind == QbmmEventKind::Growth ? b > a : b < a),
            "QBMM resizing must preserve count and change size in the specified direction.");
        const long double dm = (b - a) * donor.count;
        const auto left = dm * a, right = dm * b;
        shape_change = {left + right, left * a + left * b + right * b};
    }
    else if (event.kind == QbmmEventKind::Coalescence)
    {
        require(event.donors.size() == 2 && event.products.size() == 1,
            "QBMM coalescence requires two donors and one product.");
        const auto& first = event.donors.front().atom;
        const auto& second = event.donors.back().atom;
        const auto& product = event.products.front();
        const long double a = first.mass_over_scale, b = second.mass_over_scale;
        require(first.count == product.count && second.count == product.count &&
                    static_cast<double>(a + b) == product.mass_over_scale,
            "QBMM coalescence must preserve pair count and combine both donor masses.");
        const long double cross = (first.count * a) * b;
        shape_change = {2 * cross, 3 * (cross * a + cross * b)};
    }
    for (size_t k = 0; k < changes.size(); ++k)
    {
        validate_raw_value(event.moment_change[k + 2], shape_change[k], "QBMM signed event shape change differs.");
        changes[k] += shape_change[k];
    }
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_source_transfer(
    const QbmmMaterialTransfer& transfer, const QbmmPopulationSchema& family)
{
    if (const auto* phase = std::get_if<QbmmPhaseExchange>(&transfer.counterpart))
    {
        require(static_cast<size_t>(phase->phase) < static_cast<size_t>(SpeciesPhase::Count),
            "QBMM receipt specifies an invalid phase counterpart.");
        require(!phase->resolved_pool, "Runtime QBMM receipts cannot bind a BulkGas pool.");
        const auto& catalog = family.catalog();
        for (size_t s = 0; s < transfer.moles.size(); ++s)
            require(transfer.moles[s] == 0 || catalog.allows(catalog.species()[s].id, phase->phase),
                "QBMM receipt species is disallowed in the phase counterpart.");
    }
    else
    {
        const auto& reservoir = std::get<QbmmReservoirExchange>(transfer.counterpart);
        require(!reservoir.reservoir.empty() && std::isfinite(reservoir.energy_j) &&
                    std::ranges::all_of(reservoir.momentum, [](double value) { return std::isfinite(value); }),
            "QBMM receipt reservoir, energy or impulse is invalid.");
    }
}

/** Replay one event batch against its own original atoms and material donor stocks. */
SIMPLEFLUID_SOLVERS_LOCAL inline SourceReceiptSum replay_source_substep(const QbmmPopulationSchema& family,
    std::uint64_t catalog, SourceReceiptEndpoint initial, SourceReceiptEndpoint final,
    std::span<const QbmmEventReceipt> events, std::uint64_t substep, double duration, SourceReceiptReplay& replay,
    bool substepped)
{
    if (events.empty())
    {
        require(std::ranges::equal(final.moles, initial.moles) && final.count == initial.count &&
                    final.shape == initial.shape,
            "QBMM source changed a family without a constructive event receipt.");
        return {};
    }

    const auto pre = invert_qbmm_moments(qbmm_cell_moments(
        family, {family.population(), catalog, initial.moles, initial.count, initial.shape, family.fingerprint()}));
    require(pre.valid(), "QBMM source receipt predecessor has no supported quadrature.");
    std::array<SourceReceiptSum, 2> remaining;
    std::array<SourceReceiptSum, 2> shape_changes;
    SourceReceiptSum count, represented_count, H1, H2, H3;
    if (substepped)
        count += initial.count;
    auto margins = substepped ? replay.species_stocks : std::vector<SourceReceiptSum>{};
    for (size_t i = 0; i < pre.active_atoms; ++i)
        remaining[i] += pre.atoms[i].count;
    for (const auto& event : events)
    {
        require(event.population == family.population() && event.family_fingerprint == family.fingerprint() &&
                    event.catalog_fingerprint == catalog && !event.provider.empty() && event.provider_revision > 0 &&
                    event.substep == substep && event.duration == duration,
            "QBMM event provenance differs from the family and source substep.");
        require(event.kind >= QbmmEventKind::Birth && event.kind <= QbmmEventKind::Breakup &&
                    (!event.donors.empty() || !event.products.empty()),
            "QBMM event receipt requires a supported event kind and a constructive atom allocation.");
        for (const auto& donor : event.donors)
        {
            require(donor.index < pre.active_atoms &&
                        donor.atom.mass_over_scale == pre.atoms[donor.index].mass_over_scale &&
                        std::isfinite(donor.atom.count) && donor.atom.count > 0,
                "QBMM event donor differs from the substep predecessor quadrature.");
            remaining[donor.index] += -static_cast<long double>(donor.atom.count);
            require(remaining[donor.index].value() >= 0, "QBMM source overdraws an original donor count.");
            count += -static_cast<long double>(donor.atom.count);
        }
        for (const auto& product : event.products)
        {
            require(std::isfinite(product.count) && product.count > 0 && std::isfinite(product.mass_over_scale) &&
                        product.mass_over_scale > 0,
                "QBMM event product requires finite positive count and size.");
            count += product.count;
            if (!substepped)
                represented_count += product.count;
            const auto mass = static_cast<long double>(product.count) * product.mass_over_scale;
            H1 += mass;
            const auto sized = mass * product.mass_over_scale;
            H2 += sized;
            H3 += sized * product.mass_over_scale;
        }
        SourceReceiptSum material_change;
        for (const auto& allocation : event.transfers)
        {
            const auto& moles = allocation.transfer.moles;
            require(moles.size() == initial.moles.size(), "QBMM event material catalog layout differs.");
            for (size_t s = 0; s < moles.size(); ++s)
            {
                require(std::isfinite(moles[s]) && moles[s] >= 0, "QBMM event material transfer is invalid.");
                const long double change = allocation.into_family ? moles[s] : -static_cast<long double>(moles[s]);
                replay.species_changes[s] += change;
                if (substepped)
                {
                    replay.species_stocks[s] += change;
                    if (change < 0)
                    {
                        margins[s] += change;
                        require(std::isfinite(margins[s].value()) && margins[s].value() >= 0,
                            "QBMM source overdraws an original family species donor.");
                    }
                }
            }
            if (substepped)
            {
                validate_source_transfer(allocation.transfer, family);
                const auto mass = family.scaled_mass(moles);
                require(mass > 0, "QBMM event material transfer must have positive mass.");
                material_change += allocation.into_family ? mass : -static_cast<long double>(mass);
            }
        }
        if (substepped)
        {
            replay_source_event_moments(event, shape_changes);
            validate_raw_value(event.material_mass_change_over_scale, material_change.value(),
                "QBMM event material mass differs from the signed raw moles.");
            validate_raw_value(event.moment_change[1], material_change.value(),
                "QBMM event atom mass change differs from the signed raw moles.");
        }
        replay.gross.add(event, family.catalog());
    }
    for (size_t i = 0; i < pre.active_atoms; ++i)
    {
        const long double size = pre.atoms[i].mass_over_scale;
        represented_count += substepped ? static_cast<double>(remaining[i].value()) : remaining[i].value();
        const auto mass = remaining[i].value() * size;
        H1 += mass;
        const auto sized = mass * size;
        H2 += sized;
        H3 += sized * size;
    }

    validate_raw_value(final.shape.H2, H2.value(), "QBMM constructive H2 disagrees with raw atoms.");
    validate_raw_value(final.shape.H3, H3.value(), "QBMM constructive H3 disagrees with raw atoms.");
    if (substepped)
    {
        // The Q1 kernel sums the rounded survivor weights before adding products.
        for (const auto& event : events)
            for (const auto& product : event.products)
                represented_count += product.count;
        for (size_t s = 0; s < final.moles.size(); ++s)
        {
            const auto stock = replay.species_stocks[s].value();
            require(std::isfinite(stock) && stock >= 0 && static_cast<double>(stock) == final.moles[s] &&
                        (stock == 0) == (final.moles[s] == 0),
                "QBMM substep species endpoint differs from the sequential raw transfers.");
        }
        validate_raw_value(
            family.scaled_mass(final.moles), H1.value(), "QBMM constructive H1 disagrees with raw atoms.");
        require(invert_qbmm_moments(qbmm_cell_moments(family, {family.population(), catalog, final.moles, final.count,
                                                                  final.shape, family.fingerprint()}))
                    .valid(),
            "QBMM source receipt endpoint has no supported quadrature.");
        for (size_t k = 0; k < shape_changes.size(); ++k)
            replay.event_shape_change[k].append(shape_changes[k]);
    }
    const bool depleted = represented_count.value() < .5L * initial.count;
    if (substepped)
    {
        // Keep the compensated accumulator through the subtraction, including changes below the stored count's ULP.
        if (depleted)
            count = represented_count;
        validate_raw_value(final.count, count.value(), "QBMM constructive count disagrees with raw atoms.");
        count += -static_cast<long double>(initial.count);
        return count;
    }
    if (depleted)
    {
        // Replay a large depletion from its small positive remainder, as in the Q1 kernel.
        validate_raw_value(final.count, represented_count.value(), "QBMM remaining count disagrees with raw products.");
        represented_count += -static_cast<long double>(initial.count);
        return represented_count;
    }
    const auto change = count;
    count += initial.count;
    validate_raw_value(final.count, count.value(), "QBMM constructive count disagrees with raw atoms.");
    return change;
}

/** Check the constructive Q1 endpoints against the transport predecessor and integrated packets. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_source_receipt(const QbmmPopulationSchema& family,
    const Flow::TransportSnapshot& transport, Flow::SourceStageResult& result, size_t c, size_t a)
{
    using source_stage_detail::number_packet;
    using source_stage_detail::shape_packet;
    using source_stage_detail::species_packet;
    using source_stage_detail::validate_source_endpoint;
    const auto& receipt = *result.qbmm_cells[c][a];
    const auto& transported = transport.cells[c].phases[a];
    const auto species = transported.moles.size();
    require(transport.qbmm_families[a] == Flow::qbmm_family_stamp(family) && transported.population_number &&
                transported.qbmm_shape && receipt.family_fingerprint == family.fingerprint() &&
                receipt.final_moles.size() == species,
        "QBMM source receipt family or inventory layout differs from the transport snapshot.");
    const SourceReceiptEndpoint initial{transported.moles, *transported.population_number, *transported.qbmm_shape};
    const SourceReceiptEndpoint final{receipt.final_moles, receipt.final_count, receipt.final_shape};
    const auto number_change = number_packet(result.sources, c, a);
    const auto shape_change = shape_packet(result.sources, c, a);
    validate_source_endpoint(initial.count, number_change, final.count);
    validate_source_endpoint(initial.shape.H2, shape_change.H2, final.shape.H2);
    validate_source_endpoint(initial.shape.H3, shape_change.H3, final.shape.H3);
    validate_raw_value(shape_change.H2, static_cast<long double>(final.shape.H2) - initial.shape.H2,
        "QBMM signed H2 packet disagrees with the constructive delta.");
    validate_raw_value(shape_change.H3, static_cast<long double>(final.shape.H3) - initial.shape.H3,
        "QBMM signed H3 packet disagrees with the constructive delta.");

    SourceReceiptReplay replay(initial.moles);
    if (receipt.substeps.empty())
    {
        const auto change = replay_source_substep(family, transport.schema_fingerprint, initial, final, receipt.events,
            0, transport.time_step, replay, false);
        validate_raw_value(number_change, change.value(),
            "QBMM signed count packet disagrees with the raw event receipt.");
    }
    else
    {
        const std::span<const QbmmEventReceipt> events(receipt.events);
        auto previous = initial;
        SourceReceiptSum count_change;
        size_t begin = 0;
        double elapsed = 0;
        for (size_t i = 0; i < receipt.substeps.size(); ++i)
        {
            const auto& stage = receipt.substeps[i];
            const double end = transport.time_step * (double(i + 1) / double(receipt.substeps.size()));
            require(stage.substep == i && std::isfinite(stage.duration) && stage.duration > 0 &&
                        stage.duration == end - elapsed && stage.event_end >= begin &&
                        stage.event_end <= events.size() && stage.final_moles.size() == species,
                "QBMM source substep chronology, duration or inventory layout differs.");
            const SourceReceiptEndpoint endpoint{stage.final_moles, stage.final_count, stage.final_shape};
            count_change.append(replay_source_substep(family, transport.schema_fingerprint, previous, endpoint,
                events.subspan(begin, stage.event_end - begin), stage.substep, stage.duration, replay, true));
            previous = endpoint;
            elapsed = end;
            begin = stage.event_end;
        }
        require(elapsed == transport.time_step && begin == events.size() &&
                    std::ranges::equal(previous.moles, final.moles) && previous.count == final.count &&
                    previous.shape == final.shape,
            "QBMM source substeps do not cover the interval and reproduce its endpoint.");
        validate_raw_value(number_change, count_change.value(),
            "QBMM signed count packet disagrees with the chronological event receipts.");
        for (size_t k = 0; k < replay.event_shape_change.size(); ++k)
            validate_raw_value(receipt.event_shape_change[k], replay.event_shape_change[k].value(),
                "QBMM signed event shape change differs from the chronological event receipts.");
    }
    for (size_t s = 0; s < species; ++s)
    {
        auto change = replay.species_changes[s];
        const auto packet = species_packet(result.sources, c, a, s);
        validate_source_endpoint(initial.moles[s], packet, final.moles[s]);
        validate_raw_value(packet, change.value(), "QBMM signed species packet disagrees with raw transfers.");
        if (receipt.substeps.empty())
        {
            change += initial.moles[s];
            validate_raw_value(
                final.moles[s], change.value(), "QBMM constructive species disagrees with raw transfers.");
        }
    }
    // Per-family gross layout stays fixed through rank changes, empty states and changes of event kind.
    replay.gross.append(result.gross_events[c]);
}

SIMPLEFLUID_SOLVERS_LOCAL inline void validate_source_receipts(
    const Flow::Options& options, const Flow::TransportSnapshot& transport, Flow::SourceStageResult& result)
{
    const auto& families = options.qbmm_families;
    const auto cells = transport.cells.size();
    require(result.qbmm_cells.size() == cells, "QBMM source receipt owned-cell layout differs.");
    require(transport.qbmm_families.size() == families.size(), "QBMM transport phase layout differs.");
    if (result.gross_events.empty())
        result.gross_events.resize(cells);
    if (cells > 0 && result.gross_channel_metadata.empty())
        result.gross_channel_metadata.resize(result.gross_events.front().size());
    constexpr std::array directions{1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
    static_assert(directions.size() == Flow::QbmmSourceReceipt::gross_channels);
    for (size_t a = 0; a < families.size(); ++a)
        if (families[a])
            for (const auto direction : directions)
                result.gross_channel_metadata.push_back({Flow::GrossResidualStage::QbmmPopulation, a, direction});
    for (size_t c = 0; c < cells; ++c)
    {
        require(result.qbmm_cells[c].size() == families.size(), "QBMM source receipt phase layout differs.");
        std::optional<size_t> substeps;
        for (size_t a = 0; a < families.size(); ++a)
        {
            const auto& family = families[a];
            const auto& receipt = result.qbmm_cells[c][a];
            require(bool(family) == bool(receipt), "QBMM source receipts must match the bound families.");
            if (family)
            {
                require(!substeps || *substeps == receipt->substeps.size(),
                    "QBMM source substep schedule differs across families in this cell.");
                substeps = receipt->substeps.size();
                validate_source_receipt(*family, transport, result, c, a);
            }
        }
    }
}
} // namespace SimpleFluid::Reacting::qbmm_detail
