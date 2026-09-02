/**
 * @file MomentumSolveException.hh
 * @brief Typed failure reported by retryable momentum solves.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief A momentum solve rejected for nonconvergence without publishing state.
 *
 * This exception is reserved for linear-solver nonconvergence after the
 * momentum equation has kept the accepted velocity unchanged. Callers may
 * therefore reduce the time step and retry. Other assembly, validation, and
 * non-finite-result failures use their original exception types.
 *
 * The out-of-line destructor is the key function for stable RTTI ownership
 * across the SimpleFluid shared-library boundary.
 */
class SIMPLEFLUID_EQUATIONS_EXPORT RetryableMomentumNonconvergence final : public std::runtime_error
{
public:
    explicit RetryableMomentumNonconvergence(const char* message);
    ~RetryableMomentumNonconvergence() noexcept override;
};

} // namespace SimpleFluid
