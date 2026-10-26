/** @file Execution.hh
 * @brief Borrowed native maps and collective failure boundaries for General VoF operations.
 */
#pragma once
#include "FVM/details/VoFFailure.hh"
#include "SimpleFluidExport.hh"
#include "geometry/MeshHandle.hh"
#include <Teuchos_CommHelpers.hpp>
#include <exception>
#include <span>
#include <utility>
#include <vector>

namespace SimpleFluid::detail::general_vof
{
class SIMPLEFLUID_SOLVERS_LOCAL Execution
{
public:
    explicit Execution(const MeshHandle<>& mesh) : mesh_(mesh) {}
    void require(bool condition, const char* message) const;
    template<class Work> void collective(Work&& work, const char* stage = "general_vof_solver") const;
    [[nodiscard]] double maximum(double value) const;
    [[nodiscard]] std::vector<double> import_owned(
        std::span<const double> owned, const Teuchos::RCP<const DefaultTpetraTypes::map_type>& target) const;

private:
    const MeshHandle<>& mesh_;
};
} // namespace SimpleFluid::detail::general_vof
#include "solvers/details/GeneralVoF/Execution.ipp"
