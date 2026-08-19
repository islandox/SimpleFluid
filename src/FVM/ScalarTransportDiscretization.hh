/**
 * @file ScalarTransportDiscretization.hh
 * @brief Shared temporal and convective policies for scalar transport.
 */

#pragma once

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace SimpleFluid::FVM
{

/** @brief Temporal discretization used by an opt-in scalar transport step. */
enum class ScalarTimeScheme
{
    BackwardEuler,
    BDF2
};

/** @brief Convective reconstruction used by an opt-in scalar transport step. */
enum class ScalarConvectionScheme
{
    Upwind,
    BoundedLinearUpwind
};

/**
 * @brief Scalar transport discretization selection.
 *
 * Defaults reproduce the historical backward-Euler, first-order-upwind
 * assembly exactly. BDF2 requires a second, older solution field. Bounded
 * linear upwind retains the implicit upwind matrix and adds a limited,
 * conservative deferred correction to the right-hand side.
 */
struct ScalarTransportDiscretization
{
    ScalarTimeScheme time = ScalarTimeScheme::BackwardEuler;
    ScalarConvectionScheme convection = ScalarConvectionScheme::Upwind;
};

namespace detail
{

/** @brief Dimensionless coefficients for one scalar transient derivative. */
template<class Scalar> struct ScalarTransientCoefficients
{
    Scalar diagonal{};
    Scalar previous{};
    Scalar older{};
};

/** @brief Return backward-Euler or constant-step BDF2 coefficients. */
template<class Scalar>
constexpr ScalarTransientCoefficients<Scalar> scalar_transient_coefficients(ScalarTimeScheme scheme)
{
    switch (scheme)
    {
        case ScalarTimeScheme::BackwardEuler:
            return {Scalar{1}, Scalar{1}, Scalar{0}};
        case ScalarTimeScheme::BDF2:
            return {Scalar{3} / Scalar{2}, Scalar{2}, -Scalar{1} / Scalar{2}};
    }
    throw std::invalid_argument("Unknown scalar transport time scheme.");
}

/** @brief Stable integer code used by collective policy validation. */
constexpr int scalar_time_scheme_code(ScalarTimeScheme scheme) noexcept
{
    switch (scheme)
    {
        case ScalarTimeScheme::BackwardEuler:
            return 0;
        case ScalarTimeScheme::BDF2:
            return 1;
    }
    return 2;
}

/** @brief Stable integer code used by collective policy validation. */
constexpr int scalar_convection_scheme_code(ScalarConvectionScheme scheme) noexcept
{
    switch (scheme)
    {
        case ScalarConvectionScheme::Upwind:
            return 0;
        case ScalarConvectionScheme::BoundedLinearUpwind:
            return 1;
    }
    return 2;
}

/**
 * @brief Collectively validate a scalar policy and older-field selection.
 *
 * @param mesh Mesh providing the communicator.
 * @param discretization Local policy selection.
 * @param older_field_state Zero for absent, one for compatible, two for a
 *        field on another mesh.
 * @param context Assembly routine name used in diagnostics.
 */
template<class MeshType>
void validate_scalar_transport_discretization(
    const MeshType& mesh, ScalarTransportDiscretization discretization, int older_field_state, std::string_view context)
{
    const auto time_state = scalar_time_scheme_code(discretization.time);
    const auto convection_state = scalar_convection_scheme_code(discretization.convection);
    const std::array<int, 6> local_state{
        time_state, -time_state, convection_state, -convection_state, older_field_state, -older_field_state};
    auto global_state = local_state;
    const auto communicator = mesh.owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
    {
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_state.size()), local_state.data(),
            global_state.data());
    }

    const auto prefix = std::string(context);
    if (global_state[0] == 2)
    {
        throw std::invalid_argument(prefix + " received an unknown scalar time scheme.");
    }
    if (-global_state[1] != global_state[0])
    {
        throw std::invalid_argument(prefix + " requires every rank to use the same scalar "
                                             "time scheme.");
    }
    if (global_state[2] == 2)
    {
        throw std::invalid_argument(prefix + " received an unknown scalar convection scheme.");
    }
    if (-global_state[3] != global_state[2])
    {
        throw std::invalid_argument(prefix + " requires every rank to use the same scalar "
                                             "convection scheme.");
    }
    if (-global_state[5] != global_state[4])
    {
        throw std::invalid_argument(prefix + " requires every rank to select the same category "
                                             "of older field.");
    }
    if (global_state[4] == 2)
    {
        throw std::invalid_argument(prefix + " requires the older field on the transported-field "
                                             "mesh.");
    }
    if (global_state[0] == 1 && global_state[4] != 1)
    {
        throw std::invalid_argument(prefix + " requires an older field for BDF2.");
    }
}

/**
 * @brief Reconstruct and locally bound a linear-upwind face value.
 *
 * The unlimited value is extrapolated from the upwind cell center and then
 * clipped to the interval spanned by the two adjacent lagged cell values.
 */
template<class Scalar, class Vec>
Scalar bounded_linear_upwind_face_value(
    Scalar upwind_value, Scalar downwind_value, const Vec& upwind_gradient, const Vec& upwind_to_face)
{
    const auto unlimited = upwind_value + upwind_gradient.dot(upwind_to_face);
    return std::clamp(unlimited, std::min(upwind_value, downwind_value), std::max(upwind_value, downwind_value));
}

/** @brief Move a deferred convective face correction to the RHS. */
template<class Scalar>
constexpr Scalar deferred_convection_rhs_correction(
    Scalar outward_flux, Scalar upwind_weight, Scalar upwind_value, Scalar reconstructed_face_value) noexcept
{
    return -outward_flux * upwind_weight * (reconstructed_face_value - upwind_value);
}

} // namespace detail
} // namespace SimpleFluid::FVM
