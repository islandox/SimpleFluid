/**
 * @file SimpleFluidExport.hh
 * @brief Shared-library visibility annotations for the public binary API.
 */
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(SimpleFluidEquations_EXPORTS)
#define SIMPLEFLUID_EQUATIONS_EXPORT __declspec(dllexport)
#else
#define SIMPLEFLUID_EQUATIONS_EXPORT __declspec(dllimport)
#endif
#if defined(SimpleFluidSolvers_EXPORTS)
#define SIMPLEFLUID_SOLVERS_EXPORT __declspec(dllexport)
#else
#define SIMPLEFLUID_SOLVERS_EXPORT __declspec(dllimport)
#endif
#define SIMPLEFLUID_EQUATIONS_LOCAL
#define SIMPLEFLUID_SOLVERS_LOCAL
#define SIMPLEFLUID_LOCAL
#define SIMPLEFLUID_PUBLIC_TYPE
#elif defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_EQUATIONS_EXPORT [[gnu::visibility("default")]]
#define SIMPLEFLUID_SOLVERS_EXPORT [[gnu::visibility("default")]]
#define SIMPLEFLUID_EQUATIONS_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_SOLVERS_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_PUBLIC_TYPE [[gnu::visibility("default")]]
#else
#define SIMPLEFLUID_EQUATIONS_EXPORT
#define SIMPLEFLUID_SOLVERS_EXPORT
#define SIMPLEFLUID_EQUATIONS_LOCAL
#define SIMPLEFLUID_SOLVERS_LOCAL
#define SIMPLEFLUID_LOCAL
#define SIMPLEFLUID_PUBLIC_TYPE
#endif
