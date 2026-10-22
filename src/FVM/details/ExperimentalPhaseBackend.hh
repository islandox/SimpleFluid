/** @file ExperimentalPhaseBackend.hh
 * @brief Private opt-in exact accepted-inventory backend, deliberately unqualified.
 */
#pragma once
#include "FVM/ExperimentalPhaseInventory.hh"
#include "SimpleFluidExport.hh"
#include <memory>
#include <span>
#include <vector>
namespace SimpleFluid::VoF::detail
{
struct ExperimentalPhaseView
{
    PhaseVolumes phases;
    PhaseError error;
    PhaseRoundoff roundoff;
};
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT std::shared_ptr<const ExperimentalPhaseInventory> experimental_initial(
    const Meshes::ConvexCellGeometry&, double native_volume, double gas, double carrier, std::size_t byte_budget);
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT std::shared_ptr<const ExperimentalPhaseInventory> experimental_advance(
    const ExperimentalPhaseInventory&, Meshes::ExactPhaseMeasures, const Meshes::ConvexCellGeometry&,
    double native_volume, std::size_t byte_budget);
SIMPLEFLUID_FVM_EXPORT void validate_experimental(const ExperimentalPhaseInventory&, const Meshes::ConvexCellGeometry&,
    double native_volume, std::size_t byte_budget);
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT std::vector<char> encode_experimental(
    const ExperimentalPhaseInventory&, std::size_t byte_budget);
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT std::shared_ptr<const ExperimentalPhaseInventory> decode_experimental(
    std::span<const char>, std::size_t byte_budget);
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT ExperimentalPhaseView experimental_view(
    const ExperimentalPhaseInventory&, double native_volume);
} // namespace SimpleFluid::VoF::detail
