#pragma once

#include "equations/TimeStepperOptions.hh"
#include <array>
#include <span>
#include <string>

namespace SimpleFluid::test
{
struct CoupledBackendSelection
{
    CoupledOperatorBackend backend = CoupledOperatorBackend::Assembled;
    CoupledWorkspacePolicy workspace = CoupledWorkspacePolicy::CachedProducts;
};
inline constexpr std::array<CoupledBackendSelection, 4> coupled_backend_selections{
    {{CoupledOperatorBackend::Assembled, CoupledWorkspacePolicy::CachedProducts},
        {CoupledOperatorBackend::Assembled, CoupledWorkspacePolicy::StreamedProducts},
        {CoupledOperatorBackend::BlockComposite, CoupledWorkspacePolicy::CachedProducts},
        {CoupledOperatorBackend::BlockComposite, CoupledWorkspacePolicy::StreamedProducts}}};

inline std::span<const CoupledBackendSelection> backends_for(PressureVelocityCoupling coupling)
{
    return std::span(coupled_backend_selections).first(coupling == PressureVelocityCoupling::CoupledKrylov ? 4 : 1);
}
inline TimeStepperOptions with_coupled_backend(TimeStepperOptions options, CoupledBackendSelection selection)
{
    options.coupled_operator_backend = selection.backend;
    options.coupled_workspace_policy = selection.workspace;
    return options;
}
inline std::string backend_name(CoupledBackendSelection selection)
{
    return std::string(to_string(selection.backend)) + "/" + std::string(to_string(selection.workspace));
}
} // namespace SimpleFluid::test
