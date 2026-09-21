# External material providers

[`MaterialCAPI.h`](../src/materials/MaterialCAPI.h) defines a versioned C ABI
for synchronous material-property batches. The header-only C++ consumer,
[`MaterialProvider.hh`](../src/materials/MaterialProvider.hh), validates the
contract and results. Link C++ consumers to `SimpleFluid::Materials`; this
target is always available and has no IF97 dependency. The existing
`SIMPLEFLUID_ENABLE_IF97` option remains independent and defaults to `OFF`.

## Provider contract

Publish an `sf_material_provider_v1` with
`abi_version = SF_MATERIAL_ABI_VERSION_1`,
`struct_size = sizeof(sf_material_provider_v1)`, a nonempty name, the required
composition count, capability bits, an optional context pointer, and an
`sf_material_evaluate_v1` callback. Consumers accept a larger version-1
descriptor and read only the known prefix.

Each `sf_material_input_v1` contains positive finite temperature in kelvin,
positive finite **absolute** pressure in pascal, and a borrowed composition
array. Its count must match the descriptor. The provider documents the
composition ordering, meaning, SI units, and valid domain; the ABI does not
normalize or infer concentrations or fractions.

Each `sf_material_properties_v1` contains:

| Field | SI units | Required values |
| --- | --- | --- |
| `density` | kg/m³ | Finite, positive |
| `specific_heat_capacity` | J/(kg K) | Finite, positive |
| `dynamic_viscosity` | Pa s | Finite, nonnegative |
| `thermal_conductivity` | W/(m K) | Finite, positive |
| `density_temperature_derivative` | kg/(m³ K) | Finite; evaluated at fixed pressure and composition |

The callback returns `SF_MATERIAL_SUCCESS` (0),
`SF_MATERIAL_INVALID_ARGUMENT` (1), `SF_MATERIAL_UNSUPPORTED_ABI` (2), or
`SF_MATERIAL_EVALUATION_FAILED` (3). It initializes every output field on
success. Failure may leave partial outputs, which callers must discard.
An empty batch succeeds with null input/output arrays. A positive error-buffer
capacity requires a bounded, NUL-terminated diagnostic; capacity zero permits
a null buffer.

Callbacks must be thread-safe and must not throw across the C boundary.
Input, output, composition, and diagnostic pointers are borrowed only for the
call. The consumer copies descriptor fields but borrows the provider's name,
context, and callback code; these must outlive the consumer. The context may
be null and shared mutable context needs provider-side synchronization.

`SF_MATERIAL_CAP_AFFINE_DENSITY_TEMPERATURE` promises affine density in
temperature. `SF_MATERIAL_CAP_CONSTANT_CP_MU_K` promises temperature-independent
heat capacity, viscosity, and conductivity. Both promises apply at fixed
pressure and composition; neither changes solver behavior automatically.

## C++ consumption

The application explicitly obtains a descriptor from its linked provider.
For example, this provider declares zero composition components:

```cpp
#include "materials/MaterialProvider.hh"
#include <array>

extern "C" const sf_material_provider_v1* example_liquid_provider_v1();

double reference_expansion()
{
    const auto* descriptor = example_liquid_provider_v1();
    if (!descriptor) throw std::runtime_error("Material provider unavailable");
    const SimpleFluid::MaterialProvider material(*descriptor);
    const std::array<sf_material_input_v1, 2> states{{
        {300.0, 101325.0, nullptr, 0},
        {310.0, 101325.0, nullptr, 0}}};
    std::array<sf_material_properties_v1, 2> properties{};
    material.evaluate(states, properties);
    return SimpleFluid::MaterialProvider::beta(properties[0]);
}
```

```cmake
target_link_libraries(app PRIVATE SimpleFluid::Materials example_material)
```

Batch evaluation stages owned temporary results and publishes them only after
every input and output passes validation. Empty batches skip the callback.
Single-state evaluation and an overload returning a result vector are also
available. Invalid descriptors, inputs, or batch shapes throw
`std::invalid_argument`; callback failures and invalid results throw
`std::runtime_error` with the provider name. `beta(properties)` computes
`-density_temperature_derivative / density` in K⁻¹ and retains its sign.

The example provider can be statically linked. The API performs no implicit
`dlopen`, provider discovery, solver callback registration, or material-field
update; the application owns those integration decisions.
