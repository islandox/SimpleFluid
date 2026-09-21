/** @file MaterialCAPI.h
 * @brief Versioned, allocation-free C boundary for material property providers.
 *
 * This header is usable from C and C++. The provider owns its code, name and
 * context; all three must remain valid throughout every consumer's lifetime.
 * No C++ object layout, exception, allocator, or ownership crosses this ABI.
 */
#ifndef SIMPLEFLUID_MATERIAL_C_API_H
#define SIMPLEFLUID_MATERIAL_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SF_MATERIAL_ABI_VERSION_1 UINT32_C(1)

typedef enum sf_material_status_v1
{
    SF_MATERIAL_SUCCESS = 0,
    SF_MATERIAL_INVALID_ARGUMENT = 1,
    SF_MATERIAL_UNSUPPORTED_ABI = 2,
    SF_MATERIAL_EVALUATION_FAILED = 3
} sf_material_status_v1;

/** Density is affine in temperature at fixed pressure and composition. */
#define SF_MATERIAL_CAP_AFFINE_DENSITY_TEMPERATURE UINT64_C(1)
/** cp, viscosity and conductivity are temperature-independent at fixed
 * pressure and composition. */
#define SF_MATERIAL_CAP_CONSTANT_CP_MU_K UINT64_C(2)

typedef struct sf_material_input_v1
{
    double temperature_kelvin;       /**< Absolute temperature, K; finite and >0. */
    double absolute_pressure_pascal; /**< Absolute pressure, Pa; finite and >0. */
    /** Provider-defined composition schema, ordering and SI units. The
     * provider must document these; the generic ABI implies no normalization.
     * Borrowed for this synchronous call; required when composition_count>0. */
    const double* composition;
    size_t composition_count;
} sf_material_input_v1;

typedef struct sf_material_properties_v1
{
    double density;                        /**< kg/m^3; finite and >0. */
    double specific_heat_capacity;         /**< J/(kg K); finite and >0. */
    double dynamic_viscosity;              /**< Pa s; finite and >=0. */
    double thermal_conductivity;           /**< W/(m K); finite and >0. */
    /** Partial derivative d(rho)/dT at fixed absolute pressure and
     * composition, kg/(m^3 K); finite. */
    double density_temperature_derivative;
} sf_material_properties_v1;

/**
 * Evaluate a synchronous batch. Implementations must be reentrant/thread-safe
 * and must never let exceptions cross this boundary. No pointer may be retained
 * from inputs, outputs, composition arrays, or the error buffer.
 *
 * count==0 succeeds and permits NULL inputs/outputs. Otherwise both arrays
 * contain count records. Each input's composition_count equals the provider's
 * declared count. On success every output field is initialized. On failure
 * outputs may be partially written and must be discarded by the caller.
 *
 * error may be NULL only when error_capacity==0. When capacity is positive,
 * write a bounded NUL-terminated diagnostic (an empty string on success).
 * Return one of sf_material_status_v1; no allocation crosses the ABI.
 */
typedef int (*sf_material_evaluate_v1)(const void* context, size_t count,
    const sf_material_input_v1* inputs, sf_material_properties_v1* outputs,
    char* error, size_t error_capacity);

/**
 * A borrowed provider description. Set abi_version=SF_MATERIAL_ABI_VERSION_1
 * and struct_size=sizeof(sf_material_provider_v1). Version-1 consumers accept
 * a larger struct_size for appended extensions, but read only this prefix.
 * name is a nonempty NUL-terminated string; context may be NULL. Context data
 * shared between concurrent calls must be immutable or internally synchronized.
 * Capability bits are promises by the provider, not requests to modify physics.
 */
typedef struct sf_material_provider_v1
{
    uint32_t abi_version;
    size_t struct_size;
    const char* name;
    size_t composition_count;
    uint64_t capabilities;
    const void* context;
    sf_material_evaluate_v1 evaluate;
} sf_material_provider_v1;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SIMPLEFLUID_MATERIAL_C_API_H */
