/** @file MaterialProvider.hh
 * @brief Header-only validated consumer of the material C ABI.
 */
#pragma once

#include "materials/MaterialCAPI.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid
{
/**
 * @brief Borrow a material provider while staging and validating its results.
 *
 * The descriptor's fields are copied. Its name, context and callback code
 * remain borrowed and must outlive this object. Evaluations use only local
 * temporary storage, so concurrent calls are safe when the provider honors
 * the ABI's thread-safety contract. No solver or material field is updated.
 */
class MaterialProvider
{
public:
    using Input = sf_material_input_v1;
    using Properties = sf_material_properties_v1;

    explicit MaterialProvider(const sf_material_provider_v1& descriptor)
    {
        const auto fail = [&](std::string_view reason)
        { throw std::invalid_argument(descriptor_label(descriptor) + std::string(reason)); };
        if (descriptor.abi_version != SF_MATERIAL_ABI_VERSION_1)
            fail("unsupported material ABI version " + std::to_string(descriptor.abi_version));
        if (descriptor.struct_size < sizeof(sf_material_provider_v1))
            fail("descriptor is shorter than the version-1 ABI prefix");
        if (!descriptor.name || descriptor.name[0] == '\0') fail("name must be nonempty");
        if (!descriptor.evaluate) fail("evaluation callback is null");
        d_provider = descriptor;
    }

    std::string_view name() const noexcept { return d_provider.name; }
    size_t composition_count() const noexcept { return d_provider.composition_count; }
    uint64_t capabilities() const noexcept { return d_provider.capabilities; }
    bool has_capability(uint64_t capability) const noexcept
    { return (d_provider.capabilities & capability) == capability; }
    const sf_material_provider_v1& descriptor() const noexcept { return d_provider; }

    /** @brief Evaluate and validate one material state. */
    Properties evaluate(const Input& input) const
    {
        auto result = evaluate(std::span<const Input>(&input, 1));
        return result.front();
    }

    /** @brief Return a fully validated batch; an empty batch skips the callback. */
    std::vector<Properties> evaluate(std::span<const Input> inputs) const
    {
        if (!inputs.empty() && !inputs.data()) invalid("input array is null");
        for (size_t i = 0; i < inputs.size(); ++i) validate_input(inputs[i], i);
        if (inputs.empty()) return {};
        const auto missing = std::numeric_limits<double>::quiet_NaN();
        std::vector<Properties> staged(inputs.size(), Properties{missing, missing, missing, missing, missing});
        std::array<char, 512> error{};
        int status;
        try
        {
            status = d_provider.evaluate(d_provider.context, inputs.size(), inputs.data(),
                                         staged.data(), error.data(), error.size());
        }
        catch (const std::exception& exception)
        {
            throw std::runtime_error(label() + "callback violated the no-exception contract: " + exception.what());
        }
        catch (...)
        {
            throw std::runtime_error(label() + "callback violated the no-exception contract");
        }
        // Even a nonconforming provider cannot make diagnostic reading escape
        // this buffer. Its outputs remain staged until all checks complete.
        error.back() = '\0';
        if (status != SF_MATERIAL_SUCCESS)
            throw std::runtime_error(label() + "evaluation failed with status " + std::to_string(status)
                + (error.front() ? ": " + std::string(error.data()) : std::string{}));
        for (size_t i = 0; i < staged.size(); ++i) validate_properties(staged[i], i);
        return staged;
    }

    /** @brief Publish a batch only after every result is valid. */
    void evaluate(std::span<const Input> inputs, std::span<Properties> outputs) const
    {
        if (inputs.size() != outputs.size()) invalid("input and output counts differ");
        if (!outputs.empty() && !outputs.data()) invalid("output array is null");
        const auto staged = evaluate(inputs);
        std::copy(staged.begin(), staged.end(), outputs.begin());
    }

    /** @brief Isobaric expansion coefficient -(d rho/dT)/rho, in 1/K. */
    static double beta(const Properties& properties)
    {
        if (!(properties.density > 0) || !std::isfinite(properties.density)
            || !std::isfinite(properties.density_temperature_derivative))
            throw std::invalid_argument("Material expansion requires positive finite density and a finite temperature derivative");
        const auto value = -properties.density_temperature_derivative / properties.density;
        if (!std::isfinite(value))
            throw std::invalid_argument("Material expansion coefficient is not finite");
        return value;
    }

private:
    static std::string descriptor_label(const sf_material_provider_v1& descriptor)
    {
        const bool has_name = descriptor.struct_size >= offsetof(sf_material_provider_v1, name) + sizeof(descriptor.name)
                           && descriptor.name && descriptor.name[0];
        return "Material provider '" + std::string(has_name ? descriptor.name : "<unnamed>") + "': ";
    }
    std::string label() const { return "Material provider '" + std::string(name()) + "': "; }
    [[noreturn]] void invalid(std::string message) const
    { throw std::invalid_argument(label() + std::move(message)); }
    void validate_input(const Input& input, size_t index) const
    {
        const auto prefix = "input " + std::to_string(index) + ": ";
        if (!(input.temperature_kelvin > 0) || !std::isfinite(input.temperature_kelvin))
            invalid(prefix + "temperature must be finite and positive in kelvin");
        if (!(input.absolute_pressure_pascal > 0) || !std::isfinite(input.absolute_pressure_pascal))
            invalid(prefix + "absolute pressure must be finite and positive in pascal");
        if (input.composition_count != d_provider.composition_count)
            invalid(prefix + "composition count does not match provider schema");
        if (input.composition_count && !input.composition) invalid(prefix + "composition array is null");
        for (size_t i = 0; i < input.composition_count; ++i)
            if (!std::isfinite(input.composition[i])) invalid(prefix + "composition contains a non-finite value");
    }
    void validate_properties(const Properties& p, size_t index) const
    {
        const auto fail = [&](std::string_view property)
        { throw std::runtime_error(label() + "output " + std::to_string(index) + ": invalid " + std::string(property)); };
        if (!(p.density > 0) || !std::isfinite(p.density)) fail("density");
        if (!(p.specific_heat_capacity > 0) || !std::isfinite(p.specific_heat_capacity)) fail("specific heat capacity");
        if (!(p.dynamic_viscosity >= 0) || !std::isfinite(p.dynamic_viscosity)) fail("dynamic viscosity");
        if (!(p.thermal_conductivity > 0) || !std::isfinite(p.thermal_conductivity)) fail("thermal conductivity");
        if (!std::isfinite(p.density_temperature_derivative)) fail("density temperature derivative");
    }
    sf_material_provider_v1 d_provider{};
};
} // namespace SimpleFluid
