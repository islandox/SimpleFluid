/** @file ArithmeticMode.hh
 * @brief Explicit numerical arithmetic selection; modes never fall back.
 */
#pragma once

namespace SimpleFluid
{
enum class ArithmeticMode
{
    Ordinary,
    BoundedCertification,
    Exact
};
} // namespace SimpleFluid
