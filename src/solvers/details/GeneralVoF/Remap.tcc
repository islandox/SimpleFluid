/** @file Remap.tcc
 * @brief Compiled general VoF Remap operation.
 */
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "solvers/details/GeneralVoF/AdaptedFaceTransfer.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/OwnedFaceTransfer.hh"
#include "solvers/details/GeneralVoF/OwnedMigration.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "solvers/details/GeneralVoF/Remap.hh"
#include <algorithm>
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
template<ArithmeticMode Mode>
PreparedRemap Remap<Mode>::prepare(const Resources& resources, const State& old, bool migration,
    std::span<const int> destinations, std::span<const Meshes::AMR::AdaptiveMark> refine,
    std::span<const Meshes::AMR::AdaptiveBytes> coarsen)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& volume = resources.volume;
    const auto arithmetic_mode_ = resources.arithmetic_mode;
    const Execution execution(*resources.mesh);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    namespace fp = VoF::finite_detail;
    const auto hierarchy = old.phases->hierarchy();
    std::vector<int> ordered;
    if (migration)
    {
        execution.require(
            destinations.size() == old.phases->owned_order().size(), "Finite migration destination shape mismatch.");
        execution.collective(
            [&]
            {
                std::map<VoF::GeneralVoF::Key, int> requested;
                for (std::size_t i = 0; i < destinations.size(); ++i)
                    requested.emplace(old.phases->owned_order()[i], destinations[i]);
                for (const auto& [key, leaf] : hierarchy->owned())
                    ordered.push_back(requested.at(key));
            });
    }
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (migration)
        {
            bool moved = false;
            for (int destination : ordered)
                moved = moved || destination != hierarchy->communicator()->getRank();
            if (execution.maximum(moved ? 1. : 0.) == 0)
            {
                execution.require(old.phases->accepted_unchanged() && hierarchy->geometry_unchanged(),
                    "Identity ownership migration has stale accepted geometry or source publication.");
                return {nullptr, nullptr, true};
            }
            const auto plan = hierarchy->plan_migration(ordered);
            return prepare_owned_migration(resources, old, plan);
        }
    const auto plan = migration ? hierarchy->plan_migration(ordered) : hierarchy->plan_adaptation(refine, coarsen);
    const auto raw_target = hierarchy->apply(plan);
    const auto native = Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
    const auto bound = raw_target->with_partition(native.partition, native.mesh->owned_cell_map());
    const auto target = hierarchy->apply(plan, bound);
    VoF::GeneralVoFFiniteRemapOptions remap_options;
    remap_options.byte_budget = options.byte_budget;
    remap_options.phase_density = {material.gas_density, material.liquid_density};
    remap_options.channels = {{"momentum_x", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
        {"momentum_y", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
        {"momentum_z", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
        {"pressure_volume", VoF::GeneralVoFFiniteRemapCarrier::BulkVolume, false}};
    std::vector<VoF::GeneralVoFFiniteOwnedChannels> channels;
    execution.collective(
        [&]
        {
            const auto geometry = old.phases->owned_finite_geometry<Mode>();
            for (std::size_t c = 0; c < old.velocity.size(); ++c)
            {
                VoF::GeneralVoFFiniteOwnedChannels item;
                item.key = old.phases->owned_order()[c];
                for (unsigned a = 0; a < 3; ++a)
                    item.amounts.push_back(A::multiply(A::make_value(old.finite_mass[c], old.finite_mass_error[c]),
                        A::make_value(old.velocity[c][a], 0)));
                item.amounts.push_back(
                    A::multiply(SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume()),
                        A::make_value(old.pressure[c], 0)));
                channels.push_back(std::move(item));
            }
        });
    const auto trial = old.phases->plan_remap(plan, target, channels, remap_options);
    trial.validate_observations();
    const auto phases = trial.candidate();
    execution.require(phases->arithmetic_mode() == arithmetic_mode_, "General VoF remap changed arithmetic mode.");
    auto candidate = std::make_unique<Resources>(phases, material, options);
    const Execution target_execution(*candidate->mesh);
    auto state = std::make_shared<State>();
    state->phases = phases;
    {
        auto storage = PhaseStorage<Mode>::evaluate(*(*candidate).mesh, *state->phases, (*candidate).material);
        state->fraction = std::move(storage.fraction);
        state->density = std::move(storage.density);
        state->finite_mass = std::move(storage.mass);
        state->finite_mass_error = std::move(storage.mass_error);
        state->physical_mass.clear();
    }
    state->velocity.resize(candidate->volume.size());
    state->pressure.resize(candidate->volume.size());
    std::map<VoF::GeneralVoF::Key, const VoF::GeneralVoFFiniteOwnedChannels*> result;
    for (const auto& item : trial.owned_channels())
        result.emplace(item.key, &item);
    target_execution.collective(
        [&]
        {
            const auto geometry = phases->owned_finite_geometry<Mode>();
            for (std::size_t c = 0; c < candidate->volume.size(); ++c)
            {
                const auto& item = *result.at(phases->owned_order()[c]);
                if (item.amounts.size() != 4)
                    throw std::runtime_error("Finite remap channel schema differs.");
                for (unsigned a = 0; a < 3; ++a)
                    state->velocity[c][a] = A::as_double(
                        A::divide(item.amounts[a], A::make_value(state->finite_mass[c], state->finite_mass_error[c])));
                state->pressure[c] = A::as_double(A::divide(
                    item.amounts[3], SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume())));
            }
        });
    state->curvature = Curvature<Mode>::assemble(*(*candidate).mesh, (*candidate).flow, *state->phases,
        (*candidate).material.surface_tension, (*candidate).options.byte_budget, (*candidate).options.capillary,
        &state->capillary);
    const FaceTransferSource face_source{
        *mesh, *old.phases, old.velocity, old.finite_projection, flow, arithmetic_mode_};
    const FaceTransferTarget face_target{
        *candidate->mesh, *state->phases, candidate->volume, candidate->flow, candidate->arithmetic_mode};
    const FaceTransferControls face_controls{options.byte_budget, options.flow.divergence_tolerance};
    auto handoff = trial.ownership_only()
                       ? transfer_owned_faces<Mode>(face_source, face_target, face_controls)
                       : transfer_adapted_faces<Mode>(face_source, face_target, face_controls, trial);
    state->face_velocity = std::move(handoff.face_velocity);
    state->finite_projection = std::move(handoff.projection);
    const auto handoff_divergence = handoff.divergence;
    state->diagnostics = old.diagnostics;
    state->diagnostics.time = old.diagnostics.time;
    state->momentum_transport = old.momentum_transport;
    state->diagnostics.maximum_divergence = handoff_divergence;
    state->diagnostics = Diagnostics<Mode>::evaluate(*(*candidate).mesh, *state->phases, (*candidate).material,
        state->finite_mass, state->velocity, state->diagnostics);
    // Recovering a finite velocity must preserve the original physical
    // phase-mass-weighted momentum gate independently of channel transfer.
    for (unsigned a = 0; a < 3; ++a)
    {
        VoF::finite_detail::SumFor<Mode> change(arithmetic_mode_), reference(arithmetic_mode_);
        for (std::size_t c = 0; c < old.velocity.size(); ++c)
        {
            const auto amount = A::multiply(
                A::make_value(old.finite_mass[c], old.finite_mass_error[c]), A::make_value(old.velocity[c][a], 0));
            change.add(A::negate(amount));
            reference.add(A::make_value(std::abs(amount.value), amount.absolute_error));
        }
        for (std::size_t c = 0; c < state->velocity.size(); ++c)
        {
            const auto amount = A::multiply(A::make_value(state->finite_mass[c], state->finite_mass_error[c]),
                A::make_value(state->velocity[c][a], 0));
            change.add(amount);
            reference.add(A::make_value(std::abs(amount.value), amount.absolute_error));
        }
        const auto residual = change.value(), l1 = reference.value();
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        if (l1.value != 0)
            (void) std::frexp(l1.value + (bounded_mode ? l1.absolute_error : 0), &local_exponent);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
        const auto center = residual.value / scale;
        const auto narrowed_center = static_cast<double>(center);
        const auto reference_difference =
            !bounded_mode || l1.absolute_error == 0
                ? l1.value
                : std::nextafter(l1.value - l1.absolute_error, -std::numeric_limits<long double>::infinity());
        const auto scaled_reference =
            A::divide(A::make_value(std::max(0.L, reference_difference), 0), A::make_value(scale, 0));
        const auto lower_reference = std::max(0.L, A::lower(scaled_reference));
        auto narrowed_reference = static_cast<double>(lower_reference);
        if (bounded_mode && static_cast<long double>(narrowed_reference) > lower_reference)
            narrowed_reference = std::nextafter(narrowed_reference, 0.);
        double local[4] = {narrowed_center,
            bounded_mode
                ? fp::upper_double(fp::bound_add(residual.absolute_error / scale, std::abs(center - narrowed_center)))
                : 0,
            narrowed_reference, bounded_mode ? fp::upper_double(std::abs(center)) : double(std::abs(center))},
               global[4]{};
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
        const auto u = bounded_mode ? (8.L * mesh->owned_cell_map()->getComm()->getSize() + 16) *
                                          std::numeric_limits<double>::epsilon()
                                    : 0;
        const auto error = bounded_mode ? (global[1] + u * (global[3] + global[1])) / (1 - u) : 0;
        execution.require(std::abs(global[0]) + error <= options.momentum_relative_tolerance * global[2] * (1 - u),
            "Finite fixed-clock remap changed physical phase-mass-weighted momentum.");
    }
    trial.validate_observations();
    return {std::move(candidate), std::move(state)};
}

} // namespace SimpleFluid::detail::general_vof
