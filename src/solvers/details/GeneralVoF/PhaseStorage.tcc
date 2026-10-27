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
            if (candidate.accepted_identity() == source.accepted_identity())
                return;
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
            for (std::size_t c = 0; c < owned_count; ++c)
                for (const auto& item :
                    {std::tuple{phases[c].gas, bounds[c].gas, gas[c].value(), gas_magnitude[c].value()},
                        std::tuple{
                            phases[c].carrier, bounds[c].carrier, carrier[c].value(), carrier_magnitude[c].value()}})
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
