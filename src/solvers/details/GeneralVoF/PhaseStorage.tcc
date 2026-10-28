/** @file PhaseStorage.tcc
 * @brief Compiled physical storage and phase-publication checks.
 */
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "utils/CompensatedSum.hh"
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode>
typename PhaseStorage<Mode>::Result PhaseStorage<Mode>::evaluate(
    const MeshHandle<>& mesh_view, const VoF::GeneralVoF& authority, const VoFMaterial& material)
{
    const auto owned_count = mesh_view.num_owned_cells();
    const Execution execution(mesh_view);
    Result result;
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    namespace fp = VoF::finite_detail;
    const auto& phases = authority.phase_volumes();
    const auto geometry = authority.owned_finite_geometry<Mode>();
    const auto fraction = authority.geometric_gas_fractions();
    execution.collective(
        [&]
        {
            if (phases.size() != owned_count || geometry.size() != owned_count)
                throw std::runtime_error("Finite VoF phase/native dimensions differ.");
            result.fraction = fraction;
            result.density.resize(owned_count);
            result.mass.resize(owned_count);
            result.mass_error.resize(owned_count);
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                const auto mass =
                    A::add(A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_density, 0)),
                        A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_density, 0)));
                const auto density = A::divide(mass, SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume()));
                result.mass[c] = mass.value;
                result.mass_error[c] = mass.absolute_error;
                result.density[c] = A::as_double(density);
                if (!(mass.value > (bounded_mode ? mass.absolute_error : 0)) || !std::isfinite(result.density[c]) ||
                    !(result.density[c] > 0) || result.fraction[c] < 0 || result.fraction[c] > 1)
                    throw std::runtime_error("Finite VoF physical mixture storage is unresolved.");
            }
        });
    return result;
}

template<ArithmeticMode Mode>
void PhaseStorage<Mode>::validate(const MeshHandle<>& mesh_view, const VoF::GeneralVoF& source,
    const VoF::GeneralVoFPreparedTransport& prepared, const VoF::GeneralVoF& candidate)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    const auto* mesh = &mesh_view;
    const auto owned_count = mesh_view.num_owned_cells();
    const Execution execution(mesh_view);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    namespace fp = VoF::finite_detail;
    execution.collective(
        [&]
        {
            if (candidate.arithmetic_mode() != Mode || candidate.arithmetic_mode() != source.arithmetic_mode())
                throw std::runtime_error("Finite VoF candidate changed inventory policy.");
            const auto same_cleanup = [](const auto& first, const auto& second)
            {
                return first.signed_volume == second.signed_volume && first.absolute_volume == second.absolute_volume &&
                       first.largest_absolute_volume == second.largest_absolute_volume &&
                       first.cleaned_cells == second.cleaned_cells && first.policy_bound == second.policy_bound &&
                       first.policy.relative_cutoff == second.policy.relative_cutoff &&
                       first.policy.maximum_absolute_volume == second.policy.maximum_absolute_volume;
            };
            if (!same_cleanup(prepared.old_phase_cleanup, source.phase_cleanup_state()) ||
                !same_cleanup(prepared.new_phase_cleanup, candidate.phase_cleanup_state()))
                throw std::runtime_error("Finite phase cleanup cumulative publication differs.");
            if (candidate.accepted_identity() == source.accepted_identity())
            {
                if (!prepared.phase_cleanup.empty())
                    throw std::runtime_error("Unchanged finite phase publication repeats cleanup.");
                return;
            }
            const auto latest_cleanup = candidate.latest_phase_cleanup();
            if (latest_cleanup.size() != prepared.phase_cleanup.size())
                throw std::runtime_error("Finite phase cleanup ledger is incomplete.");
            for (std::size_t c = 0; c < latest_cleanup.size(); ++c)
            {
                const auto& first = latest_cleanup[c];
                const auto& second = prepared.phase_cleanup[c];
                if (first.key != second.key || first.gid != second.gid || first.before != second.before ||
                    first.after != second.after || first.volume != second.volume)
                    throw std::runtime_error("Finite phase cleanup ledger differs from its authority.");
            }
            std::vector<VoF::finite_detail::SumFor<Mode>> gas(owned_count, VoF::finite_detail::SumFor<Mode>(Mode)),
                carrier(owned_count, VoF::finite_detail::SumFor<Mode>(Mode));
            std::vector<CompensatedSum<>> gas_magnitude(owned_count), carrier_magnitude(owned_count);
            std::vector<std::size_t> publication_terms(owned_count);
            const bool signed_family = prepared.family == VoF::GeneralVoFTransportFamily::InterfaceLocalSwept;
            if (signed_family && Mode != ArithmeticMode::Ordinary)
                throw std::runtime_error("Signed phase publication requires Ordinary arithmetic.");
            const auto add_receipt = [&](const auto& route)
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                if (row < 0)
                    throw std::runtime_error("Finite phase publication has foreign recipient.");
                gas[row].add(route.amount.gas);
                carrier[row].add(route.amount.carrier);
                gas_magnitude[row] += std::abs(route.amount.gas.value);
                carrier_magnitude[row] += std::abs(route.amount.carrier.value);
                ++publication_terms[row];
            };
            if (signed_family)
                for (const auto& receipt : prepared.signed_receipts)
                    add_receipt(receipt);
            else
                for (const auto& route : prepared.positive_transfers)
                    add_receipt(route);
            const auto& phases = candidate.phase_volumes();
            const auto& bounds = prepared.phase_roundoff;
            if (phases.size() != owned_count || bounds.size() != owned_count)
                throw std::runtime_error("Finite phase publication has incompatible shape.");
            auto raw_phases = phases;
            std::set<GO> cleaned;
            for (const auto& delta : prepared.phase_cleanup)
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(delta.gid));
                if (Mode != ArithmeticMode::Ordinary || row < 0 || !cleaned.insert(GO(delta.gid)).second ||
                    delta.key != candidate.owned_order()[row] || delta.after[0] != phases[row].gas ||
                    delta.after[1] != phases[row].carrier ||
                    delta.volume[0] != static_cast<long double>(delta.after[0]) - delta.before[0] ||
                    delta.volume[1] != static_cast<long double>(delta.after[1]) - delta.before[1])
                    throw std::runtime_error("Finite phase cleanup differs from its declared publication.");
                raw_phases[row].gas = delta.before[0];
                raw_phases[row].carrier = delta.before[1];
            }
            for (std::size_t c = 0; c < owned_count; ++c)
                for (const auto& item :
                    {std::tuple{raw_phases[c].gas, bounds[c].gas, gas[c].value(), gas_magnitude[c].value()},
                        std::tuple{raw_phases[c].carrier, bounds[c].carrier, carrier[c].value(),
                            carrier_magnitude[c].value()}})
                {
                    const auto expected = std::get<2>(item);
                    const auto difference = std::abs(static_cast<long double>(std::get<0>(item)) - expected.value);
                    // Two independently ordered sums of the same n represented
                    // terms differ by at most twice the standard gamma bound.
                    // This allowance uses count/magnitude/precision, not defect.
                    const auto u = (publication_terms[c] + 4.L) * std::numeric_limits<long double>::epsilon();
                    if (!(u < .25L))
                        throw std::runtime_error("Finite phase publication reduction bound exhausted.");
                    const auto sum_roundoff = bounded_mode ? 0.L : 2 * u / (1 - u) * std::get<3>(item);
                    if (!std::isfinite(std::get<1>(item)) || std::get<1>(item) < 0 ||
                        difference + (bounded_mode ? expected.absolute_error : 0) > std::get<1>(item) + sum_roundoff)
                        throw std::runtime_error("Finite phase publication exceeds its own bounded ledger.");
                }
        });
}
} // namespace SimpleFluid::detail::general_vof
