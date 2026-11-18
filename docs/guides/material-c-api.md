# External Material Provider C ABI

The `MaterialCAPI` defines a versioned C ABI for synchronous, batch-based material property evaluation, enabling integration with external thermodynamic databases and constitutive models.

## ABI Specification

- **Versioning:** Providers must publish a descriptor containing `SF_MATERIAL_ABI_VERSION_1`. Consumers rely on `struct_size` to maintain backward compatibility, reading only the fields known to the consumer version.
- **Batched Interface:** The ABI utilizes a batch-evaluation callback (`sf_material_evaluate`), accepting arrays of input states (temperature, absolute pressure, composition) and outputting arrays of material properties (density, $c_p$, viscosity, thermal conductivity).
- **Thread Safety:** The provider callback must be thread-safe and must not propagate C++ exceptions across the ABI boundary.

## Provider Obligations

- **Units & Context:** Providers are responsible for defining the ordering, SI units, and valid domains of input composition arrays. The ABI acts as a raw data container; it does not perform unit conversion or composition inference.
- **Error Handling:** The callback must return defined `SF_MATERIAL_*` status codes. On failure, partial outputs must be discarded by the consumer. Diagnostic messages must be provided as NUL-terminated strings within the allocated capacity.

## C++ Consumption Contract

- **Wrapper Utility:** C++ consumers utilize `MaterialProvider` to wrap the C ABI. This utility enforces input/output validation, batch handling, and the translation of ABI error codes into C++ exceptions (`std::invalid_argument`, `std::runtime_error`).
- **Resource Ownership:** The consumer copies descriptor fields but borrows the provider's context, name, and callback function pointer. These objects must outlive the `MaterialProvider` wrapper instance.
- **Integration:** The API is designed for static linking. It performs no implicit discovery (no `dlopen`), providing full control to the application over material-provider registration and lifecycle management.
