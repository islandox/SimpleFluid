/**
 * @file EquationOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Typed operator terms used by generic finite-volume equations.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "FVM/NonOrthogonalTreatment.hh"
#include "dataclass/TpetraTypes.hh"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace SimpleFluid::FVM
{

/**
 * @brief Implicit first-order time derivative term.
 *
 * The old-value provider returns a component value for a mesh-local cell.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
struct TransientOperator
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    scalar_type time_step = 0.0;
    std::function<scalar_type(local_ordinal_type, size_t)> old_value;
};

/**
 * @brief Upwind convection term driven by an oriented face flux.
 *
 * Fluxes are positive in the owner-to-neighbor direction.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
struct ConvectionOperator
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    std::function<scalar_type(local_ordinal_type)> face_flux;
};

/** @brief Isotropic diffusion term and non-orthogonal treatment. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
struct DiffusionOperator
{
    using scalar_type = typename Pack::scalar_type;

    scalar_type diffusivity = 0.0;
    NonOrthogonalTreatment non_orthogonal_treatment =
        NonOrthogonalTreatment::Explicit;
};

/** @brief Cell-volume source term for scalar or vector equations. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
struct SourceOperator
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    std::function<scalar_type(local_ordinal_type, size_t)> value;
};

/** @brief Coupling term representing the gradient of another field. */
struct GradientOperator
{
    std::string source_field;
};

/** @brief Coupling term representing the divergence of another field. */
struct DivergenceOperator
{
    std::string source_field;
};

template<TpetraTypePack Pack = DefaultTpetraTypes>
using OperatorVariant = std::variant<
    TransientOperator<Pack>,
    ConvectionOperator<Pack>,
    DiffusionOperator<Pack>,
    SourceOperator<Pack>,
    GradientOperator,
    DivergenceOperator>;

/**
 * @brief Validate an operator before adding it to an equation.
 * @param term Operator configuration to validate.
 * @param components Number of components in the equation unknown.
 * @throws std::invalid_argument If a coefficient or provider is invalid.
 */
template<TpetraTypePack Pack>
void validate_operator(const OperatorVariant<Pack>& term,
                       size_t components)
{
    std::visit(
        [&](const auto& value)
        {
            using Term = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Term, TransientOperator<Pack>>)
            {
                if (value.time_step <= typename Pack::scalar_type{})
                {
                    throw std::invalid_argument(
                        "TransientOperator requires a positive time step.");
                }
                if (!value.old_value)
                {
                    throw std::invalid_argument(
                        "TransientOperator requires an old-value provider.");
                }
            }
            else if constexpr (std::is_same_v<Term, ConvectionOperator<Pack>>)
            {
                if (!value.face_flux)
                {
                    throw std::invalid_argument(
                        "ConvectionOperator requires a face-flux provider.");
                }
            }
            else if constexpr (std::is_same_v<Term, DiffusionOperator<Pack>>)
            {
                if (value.diffusivity < typename Pack::scalar_type{})
                {
                    throw std::invalid_argument(
                        "DiffusionOperator requires non-negative diffusivity.");
                }
            }
            else if constexpr (std::is_same_v<Term, SourceOperator<Pack>>)
            {
                if (!value.value)
                {
                    throw std::invalid_argument(
                        "SourceOperator requires a value provider.");
                }
            }
            else
            {
                if (value.source_field.empty())
                {
                    throw std::invalid_argument(
                        "Coupling operator requires a source field name.");
                }
                if (components == 0)
                {
                    throw std::invalid_argument(
                        "Equation has no field components.");
                }
            }
        },
        term);
}

} // namespace SimpleFluid::FVM
