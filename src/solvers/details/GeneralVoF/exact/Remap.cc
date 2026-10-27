/** @file Remap.cc
 * @brief Compiled general VoF Remap operation.
 */
#include "solvers/details/GeneralVoF/Remap.hh"
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "solvers/details/GeneralVoF/exact/Reduction.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
PreparedRemap Remap<ArithmeticMode::Exact>::prepare(const Resources& resources, const State& old, bool migration,
    std::span<const int> destinations, std::span<const Meshes::AMR::AdaptiveMark> refine,
    std::span<const Meshes::AMR::AdaptiveBytes> coarsen)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& volume = resources.volume;
    const Execution execution(*resources.mesh);
    using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
    const auto hierarchy = old.phases->hierarchy();
    auto source = old.phases->export_transfer_source(!migration);
    std::vector<VoF::ExactTransferChannel> schema = {{"momentum_x", VoF::ExactTransferCarrier::MixtureMass, false},
        {"momentum_y", VoF::ExactTransferCarrier::MixtureMass, false},
        {"momentum_z", VoF::ExactTransferCarrier::MixtureMass, false},
        {"pressure_volume", VoF::ExactTransferCarrier::BulkVolume, false}};
    std::vector<VoF::ExactTransferOwnedChannels> channels;
    const auto source_geometry = old.phases->owned_geometry();
    std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> source_inventory;
    execution.collective(
        [&]
        {
            source_inventory = old.phases->inventory_measures();
            for (std::size_t c = 0; c < old.phases->owned_order().size(); ++c)
            {
                const auto& gas = source_inventory[c].gas;
                const auto& carrier = source_inventory[c].carrier;
                VoF::ExactTransferOwnedChannels item;
                item.key = old.phases->owned_order()[c];
                item.phase_mass = VoF::ExactTransferPhaseMass{
                    gas.scaled(material.gas_density), carrier.scaled(material.liquid_density)};
                const auto mass = item.phase_mass->gas + item.phase_mass->carrier;
                for (unsigned a = 0; a < 3; ++a)
                    item.amounts.push_back(mass.scaled(old.velocity[c][a]));
                item.amounts.push_back(M::volume(*source_geometry[c]).scaled(old.pressure[c]));
                channels.push_back(std::move(item));
            }
        });
    VoF::ExactPhaseTransferOptions transfer_options;
    transfer_options.byte_budget = options.byte_budget;
    source = source.with_channels(std::move(schema), std::move(channels), transfer_options);
    std::vector<int> ordered_destinations;
    if (migration)
    {
        execution.require(destinations.size() == old.phases->owned_order().size(),
            "General VoF repartition destination count differs.");
        execution.collective(
            [&]
            {
                std::map<VoF::GeneralVoF::Key, int> requested;
                for (std::size_t i = 0; i < destinations.size(); ++i)
                    requested.emplace(old.phases->owned_order()[i], destinations[i]);
                for (const auto& [key, leaf] : hierarchy->owned())
                    ordered_destinations.push_back(requested.at(key));
            });
    }
    const auto plan =
        migration ? hierarchy->plan_migration(ordered_destinations) : hierarchy->plan_adaptation(refine, coarsen);
    const auto raw_target = hierarchy->apply(plan);
    const auto native_target =
        Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
    const auto bound_target = raw_target->with_partition(native_target.partition, native_target.mesh->owned_cell_map());
    const auto target = hierarchy->apply(plan, bound_target);
    const auto transferred = VoF::ExactPhaseTransfer::prepare(source, plan, target, transfer_options);
    const auto candidate_phase = old.phases->adopt_transfer(transferred);
    execution.require(candidate_phase->inventory_policy() == old.phases->inventory_policy(),
        "General VoF remap changed inventory authority policy.");
    auto candidate = std::make_unique<Resources>(candidate_phase, material, options);
    const Execution target_execution(*candidate->mesh);
    auto state = std::make_shared<State>();
    state->phases = candidate_phase;
    {
        auto storage = PhaseStorage<ArithmeticMode::Exact>::evaluate(
            *(*candidate).mesh, *state->phases, (*candidate).material, (*candidate).options.byte_budget);
        state->fraction = std::move(storage.fraction);
        state->density = std::move(storage.density);
        state->physical_mass = std::move(storage.mass);
    }
    state->velocity.resize(candidate->volume.size());
    std::vector<double> mapped_pressure(candidate->volume.size());
    std::map<VoF::GeneralVoF::Key, const VoF::ExactPhaseTransferCapsule*> result;
    for (const auto& item : transferred.owned())
        result.emplace(item.key, &item);
    const auto target_geometry = candidate_phase->owned_geometry();
    target_execution.collective(
        [&]
        {
            for (std::size_t c = 0; c < candidate_phase->owned_order().size(); ++c)
            {
                const auto& item = *result.at(candidate_phase->owned_order()[c]);
                if (item.amounts.size() != 4 || !item.phase_mass)
                    throw std::runtime_error("Exact general flow transfer schema differs.");
                const auto mass = state->physical_mass[c];
                if (!std::isfinite(item.roundoff.gas) || !std::isfinite(item.roundoff.carrier) ||
                    item.roundoff.gas < 0 || item.roundoff.carrier < 0)
                    throw std::runtime_error("Remapped phase publication bound is invalid.");
                const auto mass_difference = item.phase_mass->gas + item.phase_mass->carrier - mass;
                const auto mass_error = mass_difference.sign() < 0 ? mass_difference.scaled(-1) : mass_difference;
                // Ownership copies perform no new publication. Their
                // historical phase metadata cannot admit a mass mismatch.
                const bool exact_policy =
                    candidate_phase->inventory_policy() == VoF::PhaseInventoryPolicy::ExperimentalExact;
                const auto mass_bound = migration || exact_policy
                                            ? M{}
                                            : M::from_value(item.roundoff.gas).scaled(material.gas_density) +
                                                  M::from_value(item.roundoff.carrier).scaled(material.liquid_density);
                if (mass_error.compare(mass_bound) > 0)
                    throw std::runtime_error("Remapped phase mass exceeds its finite publication bound.");
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto value = item.amounts[a].divided_by(mass, options.byte_budget);
                    state->velocity[c][a] = value.sign() ? value.certified().as_double() : 0;
                }
                const auto pressure = item.amounts[3].divided_by(M::volume(*target_geometry[c]), options.byte_budget);
                mapped_pressure[c] = pressure.sign() ? pressure.certified().as_double() : 0;
            }
        });
    state->curvature = Curvature<ArithmeticMode::Exact>::assemble(*(*candidate).mesh, (*candidate).flow,
        *state->phases, (*candidate).material.surface_tension, (*candidate).options.byte_budget,
        (*candidate).options.capillary, &state->capillary);
    // Cell momentum/pV and face q are independent accepted degrees of
    // freedom. A fixed-clock handoff must not apply a physical dt=1 impulse.
    using Domain = detail::GeneralVoFStorageDomain;
    detail::GeneralVoFExactStorage handoff_storage(options.byte_budget);
    std::size_t face_budget = 0, certificate_budget = 0, scalar_budget = 0;
    std::vector<VoF::ExactOwnedFaceRate> old_rates;
    std::vector<VoF::ExactOwnedCellVelocity> seeds;
    target_execution.collective(
        [&]
        {
            const auto source_faces = mesh->num_owned_faces(), target_faces = candidate->mesh->num_owned_faces();
            if (old.projection_arithmetic.face_rate.size() != source_faces)
                throw std::runtime_error("Missing authoritative source face rates for fixed-clock handoff.");
            handoff_storage.capacity(Domain::Input, old.projection_arithmetic.face_rate.capacity(), sizeof(M));
            for (const auto& q : old.projection_arithmetic.face_rate)
                handoff_storage.measure(Domain::Input, q);
            handoff_storage.capacity(Domain::Scratch, source_faces, sizeof(VoF::ExactOwnedFaceRate));
            handoff_storage.capacity(
                Domain::Scratch, migration ? 0 : old.velocity.size(), sizeof(VoF::ExactOwnedCellVelocity));
            if (!migration)
                for (const auto& key : old.phases->owned_order())
                    handoff_storage.capacity(Domain::Scratch, Meshes::AMR::encode_adaptive_key(key).size(), 1);
            handoff_storage.capacity(Domain::Result, target_faces, sizeof(M) + sizeof(detail::GeneralVoFRoundedValue));
            handoff_storage.capacity(Domain::Scratch, target_faces, sizeof(M) + sizeof(unsigned char));
            handoff_storage.capacity(Domain::Result, candidate->mesh->num_faces(), sizeof(double));
            // Preserve one admitted half for the face-transfer result throughout
            // validation; the other covers caller views and exact closure proof.
            face_budget = handoff_storage.remaining() / 2;
            handoff_storage.capacity(Domain::Scratch, face_budget, 1);
            certificate_budget = handoff_storage.remaining() / 2;
            handoff_storage.capacity(Domain::Scratch, certificate_budget, 1);
            if (target_faces > (std::numeric_limits<std::size_t>::max() - 32) / 4)
                throw std::runtime_error("Fixed-clock face handoff scalar count overflow.");
            const auto slots = 4 * target_faces + 32;
            scalar_budget = handoff_storage.remaining() / slots;
            if (face_budget < 1024 || certificate_budget < 1024 || scalar_budget < 1024)
                throw std::runtime_error("Fixed-clock face handoff cumulative reserve exhausted.");
            handoff_storage.capacity(Domain::Scratch, slots, scalar_budget);
            old_rates.reserve(source_faces);
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t f = 0; f < source_faces; ++f)
                old_rates.push_back(
                    {std::uint64_t(mesh->face_global_id(LO(f))), old.projection_arithmetic.face_rate[f]});
            if (!migration)
            {
                seeds.reserve(old.velocity.size());
                for (std::size_t c = 0; c < old.velocity.size(); ++c)
                    seeds.push_back({old.phases->owned_order()[c], old.velocity[c]});
            }
        });
    VoF::ExactFaceTransferOptions face_options;
    face_options.byte_budget = face_budget;
    const auto transferred_faces =
        VoF::ExactFaceTransfer::prepare(*old.phases, *candidate_phase, transferred, old_rates, seeds, face_options);
    transferred_faces.validate_observations(*old.phases, *candidate_phase);
    std::vector<M> zero_radius;
    target_execution.collective(
        [&]
        {
            const auto count = candidate->mesh->num_owned_faces();
            state->projection_arithmetic.byte_budget = options.byte_budget;
            state->projection_arithmetic.face_rate.resize(count);
            state->projection_arithmetic.face_velocity.resize(count);
            state->face_velocity.resize(candidate->mesh->num_faces());
            zero_radius.resize(count);
            std::vector<unsigned char> seen(count);
            detail::GeneralVoFProjectionArithmetic arithmetic(scalar_budget);
            if (transferred_faces.owned_rates().size() != count)
                throw std::runtime_error("Fixed-clock face handoff has incomplete actual target coverage.");
            const auto lease = candidate->mesh->acquire_execution_view();
            for (const auto& item : transferred_faces.owned_rates())
            {
                const auto f = candidate->mesh->owned_face_map()->getLocalElement(GO(item.face));
                if (f < 0 || seen[f]++)
                    throw std::runtime_error("Foreign/duplicate target face in fixed-clock handoff.");
                if (candidate->mesh->is_geometry_exterior_face(f) && item.rate.sign())
                    throw std::runtime_error("Fixed-clock handoff changes a closed physical wall rate.");
                const auto view = detail::vof_exact::leased_divide(
                    item.rate, M::from_value(candidate->mesh->face_area(f)), scalar_budget)
                                      .enclosure();
                const double velocity = static_cast<double>(view.value);
                if (!std::isfinite(velocity))
                    throw std::runtime_error("Fixed-clock face diagnostic overflows.");
                state->projection_arithmetic.face_rate[f] = item.rate;
                // This leaf describes the diagnostic view only. Radius zero
                // belongs to retained EXACT q, not to a rounded q/area product.
                state->projection_arithmetic.face_velocity[f] = arithmetic.leaf(velocity);
                state->face_velocity[f] = velocity;
            }
            state->pressure = std::move(mapped_pressure);
        });
    const auto proof =
        detail::Exact::certify_general_vof_projection_rate(candidate->mesh, state->projection_arithmetic.face_velocity,
            certificate_budget, &state->projection_arithmetic.face_rate, &zero_radius);
    target_execution.require(
        proof.arithmetic_admitted, "Transferred exact face rates do not close the actual target topology.");
    target_execution.collective(
        [&]
        {
            for (const auto& q : proof.owned_net_rate)
                if (q.sign())
                    throw std::runtime_error("Fixed-clock face handoff has nonzero exact recipient divergence.");
        });
    state->face_velocity = candidate->flow.faces(state->face_velocity);
    state->diagnostics = old.diagnostics;
    state->momentum_transport = old.momentum_transport;
    state->diagnostics.time = old.diagnostics.time;
    state->diagnostics.maximum_divergence = 0; // Proven on actual target faces.
    state->diagnostics = Diagnostics<ArithmeticMode::Exact>::evaluate(*(*candidate).mesh, *state->phases,
        (*candidate).material, state->physical_mass, state->velocity, state->density, (*candidate).options.byte_budget,
        state->diagnostics);
    // The opaque remap proves exact channel columns. Final represented
    // velocity recovery retains its independent physical momentum gate.
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<M> before, after, reference;
        execution.collective(
            [&]
            {
                for (std::size_t c = 0; c < old.velocity.size(); ++c)
                {
                    const auto value = old.physical_mass[c].scaled(old.velocity[c][a]);
                    before.push_back(value);
                    reference.push_back(value.sign() < 0 ? value.scaled(-1) : value);
                }
                for (std::size_t c = 0; c < state->velocity.size(); ++c)
                {
                    const auto value = state->physical_mass[c].scaled(state->velocity[c][a]);
                    after.push_back(value);
                    reference.push_back(value.sign() < 0 ? value.scaled(-1) : value);
                }
            });
        const auto difference = exact::global_sum(*mesh, after, options.byte_budget) -
                                exact::global_sum(*mesh, before, options.byte_budget);
        const auto magnitude = difference.sign() < 0 ? difference.scaled(-1) : difference;
        const auto limit =
            exact::global_sum(*mesh, reference, options.byte_budget).scaled(options.momentum_relative_tolerance);
        // This is a residual-vs-L1 bound, not a value-vs-reference relative
        // error. An exactly zero L1 reference requires an exact zero residual.
        execution.require(magnitude.compare(limit) <= 0,
            "General VoF fixed-clock remap changed physical phase-mass-weighted momentum.");
    }
    transferred_faces.validate_observations(*old.phases, *candidate_phase);
    return {std::move(candidate), std::move(state)};
}

} // namespace SimpleFluid::detail::general_vof
