/** @file PhaseStorage.hh
 * @brief Physical mixture storage and publication validation for a selected arithmetic mode.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "SimpleFluidExport.hh"
#include "solvers/VoFSolver.hh"
#include <vector>

namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode> struct SIMPLEFLUID_SOLVERS_LOCAL PhaseStorage
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    struct Result
    {
        std::vector<double> fraction, density;
        std::vector<long double> mass, mass_error;
    };
    /** Collective. Derives storage from the selected physical phase authority. */
    [[nodiscard]] static Result evaluate(const MeshHandle<>&, const VoF::GeneralVoF&, const VoFMaterial&);
    /** Collective. Checks represented publication against its own transfer ledger. */
    static void validate(const MeshHandle<>&, const VoF::GeneralVoF& source, const VoF::GeneralVoFPreparedTransport&,
        const VoF::GeneralVoF& candidate);
};
template<> struct SIMPLEFLUID_SOLVERS_LOCAL PhaseStorage<ArithmeticMode::Exact>
{
    struct Result
    {
        std::vector<double> fraction, density;
        std::vector<PLIC::GeometryMeasure<ArithmeticMode::Exact>> mass;
    };
    [[nodiscard]] static Result evaluate(
        const MeshHandle<>&, const VoF::GeneralVoF&, const VoFMaterial&, std::size_t byte_budget);
    static void validate(const MeshHandle<>&, const VoF::GeneralVoF& source, const VoF::GeneralVoF::Trial&,
        const VoF::GeneralVoF& candidate, std::size_t byte_budget);
};
extern template struct PhaseStorage<ArithmeticMode::Ordinary>;
extern template struct PhaseStorage<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
