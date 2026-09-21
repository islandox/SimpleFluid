/* This translation unit deliberately compiles as C, without C++ headers. */
#include "materials/MaterialCAPI.h"

_Static_assert(SF_MATERIAL_ABI_VERSION_1 == 1, "version-1 ABI identifier");
_Static_assert(SF_MATERIAL_SUCCESS == 0 && SF_MATERIAL_INVALID_ARGUMENT == 1
    && SF_MATERIAL_UNSUPPORTED_ABI == 2 && SF_MATERIAL_EVALUATION_FAILED == 3,
    "stable callback status values");

static int constant_material(const void* context, size_t count,
    const sf_material_input_v1* inputs, sf_material_properties_v1* outputs,
    char* error, size_t error_capacity)
{
    size_t i;
    (void)context;
    if (error_capacity && error) error[0] = '\0';
    if ((error_capacity && !error) || (count && (!inputs || !outputs)))
        return SF_MATERIAL_INVALID_ARGUMENT;
    for (i = 0; i < count; ++i)
    {
        const sf_material_properties_v1 properties = {1000, 4180, 0.001, 0.6, -0.3};
        outputs[i] = properties;
    }
    return SF_MATERIAL_SUCCESS;
}

int main(void)
{
    const sf_material_provider_v1 provider = {
        SF_MATERIAL_ABI_VERSION_1, sizeof(sf_material_provider_v1), "C fixture", 0,
        SF_MATERIAL_CAP_AFFINE_DENSITY_TEMPERATURE | SF_MATERIAL_CAP_CONSTANT_CP_MU_K,
        NULL, constant_material};
    const sf_material_input_v1 input = {300, 101325, NULL, 0};
    sf_material_properties_v1 output = {0, 0, 0, 0, 0};
    char error[32] = {0};
    if (provider.evaluate(provider.context, 0, NULL, NULL, error, sizeof(error)) != SF_MATERIAL_SUCCESS)
        return 1;
    if (provider.evaluate(provider.context, 1, &input, &output, error, sizeof(error)) != SF_MATERIAL_SUCCESS)
        return 2;
    if (output.density != 1000 || output.density_temperature_derivative != -0.3 || error[0] != '\0')
        return 3;
    return 0;
}
