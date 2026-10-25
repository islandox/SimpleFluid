/** @file PhaseStorage.cc
 * @brief Optional rational phase storage and publication validation.
 */
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <cmath>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof
{
PhaseStorage<ArithmeticMode::Exact>::Result PhaseStorage<ArithmeticMode::Exact>::evaluate(
    const MeshHandle<>& mesh_view, const VoF::GeneralVoF& source, const VoFMaterial& material, std::size_t byte_budget)
{
    const auto owned_count = mesh_view.num_owned_cells();
    const Execution execution(mesh_view);
    Result result;
    const auto& gas = source.gas_volumes();
    const auto& carrier = source.carrier_volumes();
    const auto geometric_fraction = source.geometric_gas_fractions();
    std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> authority;
    execution.require(gas.size() == owned_count && carrier.size() == owned_count,
        "General VoF phase/native storage dimensions differ.");
    std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> geometry;
    execution.collective(
        [&]
        {
            authority = source.inventory_measures();
            geometry = source.owned_geometry();
            result.fraction.resize(owned_count);
            result.density.resize(owned_count);
            result.mass.resize(owned_count);
        });
    execution.collective(
        [&]
        {
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                result.fraction[c] = geometric_fraction[c];
                result.mass[c] = authority[c].gas.scaled(material.gas_density) +
                                 authority[c].carrier.scaled(material.liquid_density);
                const auto& mass = result.mass[c];
                const auto density = mass.divided_by(
                    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(*geometry[c]),
                    byte_budget);
                result.density[c] = density.sign() ? density.certified().as_double() : 0;
                if (mass.sign() <= 0 || !std::isfinite(result.density[c]) || !(result.density[c] > 0) ||
                    !std::isfinite(result.fraction[c]) || result.fraction[c] < 0 || result.fraction[c] > 1)
                    throw std::runtime_error("General VoF final extensive mixture mass/fraction is invalid.");
            }
        });
    return result;
}
void PhaseStorage<ArithmeticMode::Exact>::validate(const MeshHandle<>& mesh_view, const VoF::GeneralVoF& source,
    const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate, std::size_t byte_budget)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    const auto* mesh = &mesh_view;
    const auto owned_count = mesh_view.num_owned_cells();
    const Execution execution(mesh_view);
    execution.collective(
        [&]
        {
            using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
            using Domain = detail::GeneralVoFStorageDomain;
            detail::GeneralVoFExactStorage storage(byte_budget);
            // Only finite centers/bounds enter these small interval sums.
            // Keep their pending values and one operation disjoint from
            // the variable-size recipient aggregation below.
            storage.capacity(Domain::Scratch, 16 * 1024, 1);
            const auto n = owned_count;
            storage.capacity(Domain::Scratch, n, 4 * sizeof(M));
            M zero;
            storage.measure(Domain::Input, zero);
            for (const auto& route : trial.transfers())
            {
                storage.measure(Domain::Input, route.exact.gas);
                storage.measure(Domain::Input, route.exact.carrier);
            }
            std::vector<M> gas(n, zero), carrier(n, zero);
            if (gas.capacity() > 2 * n || carrier.capacity() > 2 * n)
                throw std::runtime_error("General VoF publication vector capacity exceeds admission.");
            for (std::size_t c = 0; c < n; ++c)
            {
                storage.measure(Domain::Scratch, gas[c]);
                storage.measure(Domain::Scratch, carrier[c]);
            }
            const auto add = [&](M& value, const M& part)
            {
                const auto remaining = storage.remaining();
                if (remaining <= 2048)
                    throw std::runtime_error("General VoF finite publication proof exhausted its operation lease.");
                auto sum = detail::vof_exact::add(value, part, remaining - 1024);
                storage.measure(Domain::Scratch, sum);
                storage.release(Domain::Scratch, value);
                value = std::move(sum);
            };
            for (const auto& route : trial.transfers())
            {
                const auto local = mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                if (local < 0)
                    throw std::runtime_error("General VoF publication proof has a foreign recipient.");
                add(gas[local], route.exact.gas);
                add(carrier[local], route.exact.carrier);
            }
            if (candidate.inventory_policy() != source.inventory_policy())
                throw std::runtime_error("General VoF trial changed inventory authority policy.");
            if (candidate.inventory_policy() == VoF::PhaseInventoryPolicy::ExperimentalExact)
            {
                const auto& exact = candidate.experimental_phases();
                if (exact.size() != n)
                    throw std::runtime_error("Experimental exact phase publication shape differs.");
                for (std::size_t c = 0; c < n; ++c)
                    if (gas[c].compare(exact[c].gas) || carrier[c].compare(exact[c].carrier))
                        throw std::runtime_error("Experimental exact phase publication differs from its routes.");
                return;
            }
            const auto& phases = candidate.phase_volumes();
            // Identity trials retain the accepted object and its historical
            // publication metadata; only THIS trial's bounds may admit error.
            const auto& bounds = trial.phase_roundoff_bounds();
            if (phases.size() != n || bounds.size() != n)
                throw std::runtime_error("General VoF finite publication bound shape differs.");
            const auto check = [&](double value, double bound, const M& expected)
            {
                if (!std::isfinite(value) || !std::isfinite(bound) || bound < 0)
                    throw std::runtime_error("General VoF finite publication bound is invalid.");
                const auto center = M::from_value(value);
                const auto lower = detail::vof_exact::add(center, M::from_value(-bound), 8192);
                const auto upper = detail::vof_exact::add(center, M::from_value(bound), 8192);
                if (expected.compare(lower) < 0 || expected.compare(upper) > 0)
                    throw std::runtime_error("General VoF phase publication exceeds its own conversion bound.");
            };
            for (std::size_t c = 0; c < n; ++c)
            {
                check(phases[c].gas, bounds[c].gas, gas[c]);
                check(phases[c].carrier, bounds[c].carrier, carrier[c]);
            }
            // Positive densities give |Mnew-Mrouted| <= rhoG*boundG+
            // rhoC*boundC. This is arithmetic provenance only: momentum
            // still solves its original equation with represented Mnew.
        });
}
} // namespace SimpleFluid::detail::general_vof
