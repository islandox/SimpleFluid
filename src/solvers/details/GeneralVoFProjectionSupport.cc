/** @file GeneralVoFProjectionSupport.cc
 * @brief Strict floating-point projection reductions and local occupancy stages.
 */
#include "solvers/details/GeneralVoFProjectionSupport.hh"
#include "FVM/details/GeneralVoF/OccupancyAdmission.hh"
#include "solvers/details/GeneralVoF/FlowTrace.hh"
#include "utils/CompensatedSum.hh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace SimpleFluid::detail::projection_support
{
long double wide_maximum(const Teuchos::Comm<int>& comm, long double local)
{
    int exponent = std::numeric_limits<int>::min(), global_exponent = 0;
    double mantissa = 0, global_mantissa = 0;
    if (local > 0)
    {
        const auto wide = std::frexp(local, &exponent);
        mantissa = static_cast<double>(wide);
        if (static_cast<long double>(mantissa) < wide)
            mantissa = std::nextafter(mantissa, std::numeric_limits<double>::infinity());
    }
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &exponent, &global_exponent);
    if (exponent != global_exponent)
        mantissa = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &mantissa, &global_mantissa);
    return global_mantissa > 0 ? std::ldexp(static_cast<long double>(global_mantissa), global_exponent) : 0.L;
}
std::array<long double, 2> wide_sums(const Teuchos::Comm<int>& comm, const std::array<long double, 2>& local)
{
    std::array<double, 6> send{};
    std::vector<double> received;
    arithmetic_guard(
        comm,
        [&]
        {
            received.resize(6 * std::size_t(comm.getSize()));
            for (unsigned a = 0; a < 2; ++a)
            {
                if (!std::isfinite(local[a]))
                    throw std::runtime_error("Nonfinite occupancy target reduction.");
                int exponent = 0;
                const auto mantissa = std::frexp(local[a], &exponent);
                send[3 * a] = double(mantissa);
                send[3 * a + 1] = double(mantissa - send[3 * a]);
                send[3 * a + 2] = exponent;
                if (static_cast<long double>(send[3 * a]) + send[3 * a + 1] != mantissa)
                    throw std::runtime_error("Occupancy target reduction loses represented precision.");
            }
        },
        "Occupancy target reduction exceeds its admitted representation.");
    Teuchos::gatherAll(comm, 6, send.data(), 6 * comm.getSize(), received.data());
    std::array<CompensatedSum<long double>, 2> sums;
    arithmetic_guard(
        comm,
        [&]
        {
            for (int rank = 0; rank < comm.getSize(); ++rank)
                for (unsigned a = 0; a < 2; ++a)
                {
                    const auto i = 6 * std::size_t(rank) + 3 * a;
                    sums[a] +=
                        std::ldexp(static_cast<long double>(received[i]) + received[i + 1], int(received[i + 2]));
                }
        },
        "Occupancy target reduction exceeds its arithmetic range.");
    return std::array<long double, 2>{sums[0].value(), sums[1].value()};
}
std::array<long double, 2> occupancy_capacities(const OccupancyTargetInput& input, OccupancyTargetResult& result)
{
    long double local_capacity = 0, local_floor = 0;
    for (std::size_t c = 0; c < input.cell_volumes.size(); ++c)
    {
        // Half of each original allowance remains for solve error.
        const auto capacity =
            .5L *
            std::min({input.volume_tolerance * input.incident_rates[c] / (2 * input.max_courant),
                input.physical_limit * input.cell_volumes[c],
                2 * VoF::algebraic_physical_occupancy_relative_limit * input.incident_rates[c] / input.max_courant});
        result.target[c] = capacity;
        local_capacity = std::max(local_capacity, capacity);
        local_floor = std::max(local_floor,
            (32.L + 8.L * input.mesh.faces(DefaultTpetraTypes::local_ordinal_type(c)).size()) *
                std::numeric_limits<long double>::epsilon() * input.incident_rates[c] / input.cell_volumes[c]);
    }
    return {local_capacity, local_floor};
}
std::array<long double, 2> occupancy_weights(
    const OccupancyTargetInput& input, const OccupancyTargetResult& result, long double capacity_scale)
{
    CompensatedSum<long double> weights, weighted_discrepancy;
    for (std::size_t c = 0; c < input.cell_volumes.size(); ++c)
    {
        const auto weight = result.target[c] / capacity_scale;
        const auto relative = input.signed_discrepancy[c] /
                              (VoF::algebraic_physical_occupancy_relative_limit * input.inventory_volumes[c]);
        weights += weight;
        weighted_discrepancy += weight * relative;
    }
    return {weights.value(), weighted_discrepancy.value()};
}
OccupancyTargetLocalTotals occupancy_targets(
    const OccupancyTargetInput& input, OccupancyTargetResult& result, long double mean)
{
    CompensatedSum<long double> target_sum, capacity_sum;
    long double local_target = 0;
    for (std::size_t c = 0; c < input.cell_volumes.size(); ++c)
    {
        const auto capacity = result.target[c];
        const auto relative = input.signed_discrepancy[c] /
                              (VoF::algebraic_physical_occupancy_relative_limit * input.inventory_volumes[c]);
        result.target[c] = capacity * ((relative - mean) / 2);
        target_sum += result.target[c];
        capacity_sum += capacity;
        local_target = std::max(local_target, std::abs(result.target[c]) / input.cell_volumes[c]);
        general_vof::flow_trace(input.mesh.owned_cell_map()->getComm()->getRank(), "occupancy_pressure_target",
            [&](auto& out)
            {
                out << ",\"cell\":" << input.mesh.cell_global_id(DefaultTpetraTypes::local_ordinal_type(c))
                    << ",\"signed_occupancy\":" << input.signed_discrepancy[c] << ",\"capacity\":" << capacity
                    << ",\"target_net_rate\":" << result.target[c] << ",\"normalized_mean\":" << mean;
            });
    }
    return {{target_sum.value(), capacity_sum.value()}, local_target};
}
} // namespace SimpleFluid::detail::projection_support
