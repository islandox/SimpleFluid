/**
 * @file SimpleFluidExport.hh
 * @brief Shared-library visibility annotations for the public binary API.
 */
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(SimpleFluidFVM_EXPORTS)
#define SIMPLEFLUID_FVM_EXPORT __declspec(dllexport)
#else
#define SIMPLEFLUID_FVM_EXPORT __declspec(dllimport)
#endif
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
#define SIMPLEFLUID_FVM_LOCAL
#define SIMPLEFLUID_EQUATIONS_LOCAL
#define SIMPLEFLUID_SOLVERS_LOCAL
#define SIMPLEFLUID_LOCAL
#define SIMPLEFLUID_PUBLIC_TYPE
#elif defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_FVM_EXPORT [[gnu::visibility("default")]]
#define SIMPLEFLUID_EQUATIONS_EXPORT [[gnu::visibility("default")]]
#define SIMPLEFLUID_SOLVERS_EXPORT [[gnu::visibility("default")]]
#define SIMPLEFLUID_FVM_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_EQUATIONS_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_SOLVERS_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_LOCAL [[gnu::visibility("hidden")]]
#define SIMPLEFLUID_PUBLIC_TYPE [[gnu::visibility("default")]]
#else
#define SIMPLEFLUID_FVM_EXPORT
#define SIMPLEFLUID_EQUATIONS_EXPORT
#define SIMPLEFLUID_SOLVERS_EXPORT
#define SIMPLEFLUID_FVM_LOCAL
#define SIMPLEFLUID_EQUATIONS_LOCAL
#define SIMPLEFLUID_SOLVERS_LOCAL
#define SIMPLEFLUID_LOCAL
#define SIMPLEFLUID_PUBLIC_TYPE
#endif

// Mesh remains a real compatibility class; the native path uses MeshHandle.
#if defined(SIMPLEFLUID_DISABLE_LEGACY_MESH_DEPRECATION)
#define SIMPLEFLUID_LEGACY_MESH_DEPRECATED
#else
#define SIMPLEFLUID_LEGACY_MESH_DEPRECATED                                                                             \
    [[deprecated("Use MeshHandle and MeshFactory::build_handle() for native mesh execution.")]]
#endif

// Scope intentional compatibility declarations without disabling diagnostics
// for client code which explicitly names the deprecated class/API.
#if defined(_MSC_VER)
#define SIMPLEFLUID_LEGACY_API_BEGIN __pragma(warning(push)) __pragma(warning(disable : 4996))
#define SIMPLEFLUID_LEGACY_API_END __pragma(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_LEGACY_API_BEGIN                                                                                   \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define SIMPLEFLUID_LEGACY_API_END _Pragma("GCC diagnostic pop")
#else
#define SIMPLEFLUID_LEGACY_API_BEGIN
#define SIMPLEFLUID_LEGACY_API_END
#endif
