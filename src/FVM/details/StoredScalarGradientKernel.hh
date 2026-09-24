#pragma once

namespace SimpleFluid::FVM::detail
{

// Applies either an interior difference stencil or an affine stencil. The
// caller owns compatibility checks and any required overlap synchronization.
template<class Vec, class Stencil, class ReadView>
Vec apply_stored_scalar_gradient(const Stencil& stencil, const ReadView& values)
{
    Vec gradient{};
    if constexpr (requires { stencil.constant; })
        gradient = stencil.constant;
    const auto& entries = [&]() -> const auto&
    {
        if constexpr (requires { stencil.entries; })
            return stencil.entries;
        else
            return stencil;
    }();
    for (const auto& entry : entries)
        gradient = gradient + entry.coefficient * values(entry.cell_lid, 0);
    return gradient;
}

} // namespace SimpleFluid::FVM::detail
