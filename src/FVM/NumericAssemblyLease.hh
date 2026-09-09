#pragma once

namespace SimpleFluid::FVM
{
/**
 * A coupled generation protects its momentum coefficients across equation calls.
 * The coupled cache may release protection only when no caller retains that
 * generation. This is sequential-use metadata, never a numerical coefficient.
 */
struct NumericAssemblyLease
{
    bool protected_generation = true;
};
} // namespace SimpleFluid::FVM
