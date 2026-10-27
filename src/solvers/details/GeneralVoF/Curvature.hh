/** @file Curvature.hh
 * @brief Distributed surface assembly for the physical PLIC curvature kernel.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "SimpleFluidExport.hh"
#include <vector>

namespace SimpleFluid::detail
{
class GeneralVoFFlowWorkspace;
}
namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode> struct SIMPLEFLUID_SOLVERS_LOCAL Curvature
{
    /** Collective. Fits mixed sections in retained stencil order, then extends
     * their curvature to admitted neighboring pure cells. */
    [[nodiscard]] static std::vector<double> assemble(const MeshHandle<>& mesh,
        const GeneralVoFFlowWorkspace& workspace, const VoF::GeneralVoF& phases, double surface_tension,
        std::size_t byte_budget, const VoF::GeneralVoFSurfaceOptions& = {},
        VoF::GeneralVoFCapillaryDiagnostics* = nullptr);
};
extern template struct Curvature<ArithmeticMode::Ordinary>;
extern template struct Curvature<ArithmeticMode::BoundedCertification>;
extern template struct Curvature<ArithmeticMode::Exact>;
} // namespace SimpleFluid::detail::general_vof
