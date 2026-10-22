/** @file ExactPhaseAccumulator.hh
 * @brief Bounded donor-linear accumulation of transient geometric phase measures.
 */
#pragma once
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
namespace SimpleFluid::VoF::detail
{
/** Bounded raw geometric sum before ONE donor-linear relabelling. Never use
 * this for independent physical inventories or occupancy-defect self records. */
class SIMPLEFLUID_FVM_EXPORT ExactPhaseAccumulator
{
public:
    explicit ExactPhaseAccumulator(ExactLiveStorage&);
    ExactPhaseAccumulator(const ExactPhaseAccumulator&) = delete;
    void add(const Meshes::ExactPhaseMeasures&, int sign = 1);
    [[nodiscard]] Meshes::ExactPhaseMeasures apply(const ExactPhaseLabelMap&) const;
    [[nodiscard]] const Meshes::ExactPhaseMeasures& raw() const noexcept { return raw_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    ExactLiveStorage& live_;
    Meshes::ExactPhaseMeasures raw_;
    ExactLiveStorage::Scope scope_;
    std::size_t count_ = 0;
};
} // namespace SimpleFluid::VoF::detail
