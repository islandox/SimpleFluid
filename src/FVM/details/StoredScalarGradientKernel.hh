/** @file StoredScalarGradientKernel.hh
 * @brief Applies cached scalar-gradient stencils to field values.
 */
#pragma once

namespace SimpleFluid::FVM::detail
{

// Applies either an interior difference stencil or an affine stencil in
// difference form relative to the owning cell's own value. Interior
// stencils retain a self entry with the negated basis sums; its difference
// is identically zero and is skipped, so constant fields reproduce a
// bit-exact zero gradient. Unfolded Dirichlet-like boundary terms act on
// differences relative to the own value as well, while Neumann increments
// stay folded in the affine constant. The caller owns compatibility checks
// and any required overlap synchronization.
template<class Vec, class Stencil, class ReadView, class LocalOrdinal>
Vec apply_stored_scalar_gradient(
    const Stencil& stencil, const ReadView& values, LocalOrdinal own_lid)
{
    Vec gradient{};
    if constexpr (requires { stencil.constant; })
        gradient = stencil.constant;
    const auto own_value = values(own_lid, 0);
    const auto& entries = [&]() -> const auto&
    {
        if constexpr (requires { stencil.entries; })
            return stencil.entries;
        else
            return stencil;
    }();
    for (const auto& entry : entries)
    {
        if (entry.cell_lid == own_lid)
            continue;
        gradient = gradient + entry.coefficient * (values(entry.cell_lid, 0) - own_value);
    }
    if constexpr (requires { stencil.boundary_terms; })
    {
        for (const auto& term : stencil.boundary_terms)
            gradient = gradient + term.basis * (term.value - own_value);
    }
    return gradient;
}

} // namespace SimpleFluid::FVM::detail
