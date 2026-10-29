/** @file BalanceForecast.hh
 * @brief Private explicit-destination seam for independent work-model checks.
 */
#pragma once
#include "FVM/GeneralVoFBalance.hh"
#include "FVM/GeneralVoFWork.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
struct GeneralVoFBalanceForecast
{
    GeneralVoFBalanceLoad measured_before, forecast_proposed;
    double local_core_before = 0, local_receiver_before = 0, local_core_proposed = 0, local_receiver_proposed = 0;
    bool incomplete_receiver_model = false;
};
/** The caller authenticates source identity. This helper independently validates
 * sorted complete owned rows and sparse dependency/receiver agreement. Destinations
 * follow sample.cells in increasing key order. No migration or geometry is performed. */
[[nodiscard]] SIMPLEFLUID_FVM_LOCAL GeneralVoFBalanceForecast forecast_partition_work(const GeneralVoFWorkSample&,
    std::span<const int> destinations, Teuchos::RCP<const Teuchos::Comm<int>>, GeneralVoFBalanceOptions = {});
} // namespace SimpleFluid::VoF::general_vof_detail
