/** @file GeometryError.hh
 * @brief Typed finite geometry failures; only unresolved arithmetic is a recoverable trial.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include <stdexcept>
#include <string>

namespace SimpleFluid::PLIC
{
enum class GeometryErrorCode
{
    InvalidInput,
    InvalidTopology,
    ProvenanceMismatch,
    ResourceLimit,
    UnresolvedArithmetic
};
class SIMPLEFLUID_PUBLIC_TYPE GeometryError : public std::invalid_argument
{
public:
    GeometryError(GeometryErrorCode code, const std::string& message) : std::invalid_argument(message), code_(code) {}
    [[nodiscard]] GeometryErrorCode code() const noexcept { return code_; }
    [[nodiscard]] bool recoverable_trial() const noexcept { return code_ == GeometryErrorCode::UnresolvedArithmetic; }

private:
    GeometryErrorCode code_;
};
} // namespace SimpleFluid::PLIC
