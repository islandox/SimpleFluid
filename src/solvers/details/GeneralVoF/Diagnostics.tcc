/** @file Diagnostics.tcc
 * @brief Compiled scaled reductions for physical General VoF diagnostics.
 */
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode>
VoFStepDiagnostics Diagnostics<Mode>::evaluate(const MeshHandle<>& mesh_view, const VoF::GeneralVoF& authority,
    const VoFMaterial& material, std::span<const long double> mass, std::span<const VoF::Vector> velocity,
    VoFStepDiagnostics diagnostics)
{
    const auto* mesh = &mesh_view;
    const auto owned_count = mesh_view.num_owned_cells();
    const Execution execution(mesh_view);
    const auto dot = [](const VoF::Vector& a, const VoF::Vector& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    using A = VoF::finite_detail::Arithmetic<Mode>;
    namespace fp = VoF::finite_detail;
    std::array<VoF::finite_detail::SumFor<Mode>, 7> sums;
    for (auto& sum : sums)
        sum = VoF::finite_detail::SumFor<Mode>(Mode);
    execution.collective(
        [&]
        {
            const auto& phases = authority.phase_volumes();
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                sums[0].add(A::make_value(phases[c].gas, 0));
                sums[1].add(A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_density, 0)));
                sums[2].add(
                    A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_density, 0)));
                for (unsigned a = 0; a < 3; ++a)
                    sums[3 + a].add(A::multiply(A::make_value(mass[c], 0), A::make_value(velocity[c][a], 0)));
                const auto square = static_cast<long double>(velocity[c][0]) * velocity[c][0] +
                                    static_cast<long double>(velocity[c][1]) * velocity[c][1] +
                                    static_cast<long double>(velocity[c][2]) * velocity[c][2];
                sums[6].add(A::multiply(A::make_value(mass[c], 0), A::make_value(square / 2, 0)));
            }
        });
    std::array<double, 7> global{};
    for (unsigned a = 0; a < 7; ++a)
    {
        const auto local = sums[a].value();
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        if (local.value != 0)
            (void) std::frexp(std::abs(local.value), &local_exponent);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
        const auto normalized = static_cast<double>(local.value / scale);
        double sum = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &normalized, &sum);
        execution.collective(
            [&]
            {
                const auto value = static_cast<long double>(sum) * scale;
                global[a] = static_cast<double>(value);
                if (!std::isfinite(global[a]) || (value != 0 && global[a] == 0))
                    throw std::runtime_error("Finite VoF global physical diagnostic is outside representable range.");
            });
    }
    diagnostics.gas_volume = global[0];
    diagnostics.gas_mass = global[1];
    diagnostics.liquid_mass = global[2];
    for (unsigned a = 0; a < 3; ++a)
        diagnostics.momentum[a] = global[3 + a];
    diagnostics.kinetic_energy = global[6];
    double speed = 0;
    for (const auto& item : velocity)
        speed = std::max(speed, std::sqrt(dot(item, item)));
    diagnostics.maximum_velocity = execution.maximum(speed);
    return diagnostics;
}
} // namespace SimpleFluid::detail::general_vof
