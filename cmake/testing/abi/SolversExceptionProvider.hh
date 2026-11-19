/**
 * @file SolversExceptionProvider.hh
 * @brief Test-only producer API for typed caloric exceptions across shared libraries.
 */
#pragma once

#include <stdexcept>
#include <typeinfo>

namespace SimpleFluid::abi_fixture
{
enum class CaloricExceptionKind
{
    Domain,
    Range
};

/** Throw the requested public exception with a stable test message. */
[[noreturn, gnu::visibility("default")]] void throw_caloric_exception(CaloricExceptionKind kind);
/** Transfer ownership of a derived exception through its standard base. */
[[nodiscard, gnu::visibility("default")]] std::domain_error* make_caloric_exception(CaloricExceptionKind kind);
/** Expose the producer's RTTI identity for comparison with its consumer. */
[[nodiscard, gnu::visibility("default")]] const std::type_info& caloric_exception_type(CaloricExceptionKind kind);
} // namespace SimpleFluid::abi_fixture
