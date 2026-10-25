/** @file Diagnostics.cc
 * @brief Exact physical inventory and velocity diagnostics.
 */
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/exact/Reduction.hh"
#include <algorithm>
#include <array>
#include <cmath>

namespace SimpleFluid::detail::general_vof
{
VoFStepDiagnostics Diagnostics<ArithmeticMode::Exact>::evaluate(const MeshHandle<>& mesh_view,
    const VoF::GeneralVoF& source, const VoFMaterial& material,
    std::span<const PLIC::GeometryMeasure<ArithmeticMode::Exact>> masses, std::span<const VoF::Vector> velocities,
    std::span<const double> densities, std::size_t byte_budget, VoFStepDiagnostics diagnostics)
{
    using Measure = PLIC::GeometryMeasure<ArithmeticMode::Exact>;
    const auto owned_count = mesh_view.num_owned_cells();
    const Execution execution(mesh_view);
    const auto dot = [](const VoF::Vector& a, const VoF::Vector& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    std::array<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>, 7> local;
    std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> authority;
    bool valid = true;
    execution.collective(
        [&]
        {
            authority = source.inventory_measures();
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                const auto& gas = authority[c].gas;
                const auto& carrier = authority[c].carrier;
                const auto& mass = masses[c];
                const auto& velocity = velocities[c];
                local[0] = detail::vof_exact::add(local[0], gas, byte_budget);
                local[1] = detail::vof_exact::add(local[1], gas.scaled(material.gas_density), byte_budget);
                local[2] = detail::vof_exact::add(local[2], carrier.scaled(material.liquid_density), byte_budget);
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto momentum = mass.scaled(velocity[a]);
                    local[3 + a] = detail::vof_exact::add(local[3 + a], momentum, byte_budget);
                    valid = valid && std::isfinite(velocity[a]);
                }
                local[6] = detail::vof_exact::add(local[6], mass.scaled(.5 * dot(velocity, velocity)), byte_budget);
                valid = valid && std::isfinite(densities[c]) && densities[c] > 0;
            }
        });
    execution.require(valid, "General VoF accepted candidate has nonfinite velocity/storage.");
    std::array<double, 7> global{};
    for (unsigned a = 0; a < 7; ++a)
    {
        const auto sum = exact::global_sum(mesh_view, std::span<const Measure>(&local[a], 1), byte_budget);
        execution.collective([&] { global[a] = sum.sign() ? sum.certified().as_double() : 0; });
    }
    diagnostics.gas_volume = global[0];
    diagnostics.gas_mass = global[1];
    diagnostics.liquid_mass = global[2];
    for (unsigned a = 0; a < 3; ++a)
        diagnostics.momentum[a] = global[3 + a];
    diagnostics.kinetic_energy = global[6];
    double speed = 0;
    for (const auto& velocity : velocities)
        speed = std::max(speed, std::sqrt(dot(velocity, velocity)));
    diagnostics.maximum_velocity = execution.maximum(speed);
    execution.require(std::all_of(global.begin(), global.end(), [](double value) { return std::isfinite(value); }),
        "General VoF global accepted budget is nonfinite.");
    return diagnostics;
}
} // namespace SimpleFluid::detail::general_vof
