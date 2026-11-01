/** @file ArithmeticMode.hh
 * @brief Explicit numerical arithmetic selection; modes never fall back.
 */
#pragma once

#include <cstdint>

namespace SimpleFluid
{
using std::uint8_t;
enum class ArithmeticMode : uint8_t
{
    Ordinary,
    BoundedCertification,
    Exact
};
} // namespace SimpleFluid
