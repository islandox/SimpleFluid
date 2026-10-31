/** @file GeneralVoFProjectionSupport.hh
 * @brief Private mode-independent projection reductions and occupancy stages.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/MeshHandle.hh"
#include <Teuchos_Comm.hpp>
#include <array>
#include <span>
#include <vector>

namespace SimpleFluid::detail::projection_support
{
/** Positive finite maxima retain the original outward binary64 mantissa wire. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL long double wide_maximum(const Teuchos::Comm<int>& comm, long double local);
/** Rank-major compensated reconstruction of two represented wide sums. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::array<long double, 2> wide_sums(
    const Teuchos::Comm<int>& comm, const std::array<long double, 2>& local);
/** Catch local work failures before the single collective acceptance reduction. */
template<class Work>
SIMPLEFLUID_SOLVERS_LOCAL void arithmetic_guard(const Teuchos::Comm<int>& comm, Work&& work, const char* message);

/** Borrowed owned-cell data. The coordinator owns shape and readiness admission. */
struct OccupancyTargetInput
{
    const MeshHandle<>& mesh;
    std::span<const double> cell_volumes;
    std::span<const long double> incident_rates, signed_discrepancy, inventory_volumes;
    double volume_tolerance, max_courant;
    long double physical_limit;
};
/** Capacities are overwritten by targets in the coordinator's existing allocation. */
struct OccupancyTargetResult
{
    std::vector<long double> target;
    long double maximum = 0, target_limit = 0, compatibility_residual = 0, compatibility_bound = 0;
};
struct OccupancyTargetLocalTotals
{
    std::array<long double, 2> sums{};
    long double maximum = 0;
};
/** Local stages perform no collectives and reuse existing numerical scratch. The coordinator retains guards,
 * execution leases, reductions and acceptance gates in their original order. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::array<long double, 2> occupancy_capacities(
    const OccupancyTargetInput& input, OccupancyTargetResult& result);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::array<long double, 2> occupancy_weights(
    const OccupancyTargetInput& input, const OccupancyTargetResult& result, long double capacity_scale);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL OccupancyTargetLocalTotals occupancy_targets(
    const OccupancyTargetInput& input, OccupancyTargetResult& result, long double mean);
} // namespace SimpleFluid::detail::projection_support

#include "solvers/details/GeneralVoFProjectionSupport.ipp"
