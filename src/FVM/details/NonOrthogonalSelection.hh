/** @file NonOrthogonalSelection.hh
 * @brief Resolves non-orthogonal correction selectors for transport assembly.
 */
#pragma once

#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/TransportValidation.hh"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace SimpleFluid::FVM::detail
{

template<class Scalar> struct NonOrthogonalWeights
{
    Scalar implicit{};
    Scalar explicit_{};
};

inline constexpr int non_orthogonal_treatment_code(NonOrthogonalTreatment treatment) noexcept
{
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            return 0;
        case NonOrthogonalTreatment::Implicit:
            return 1;
        case NonOrthogonalTreatment::Hybrid:
            return 2;
    }
    return 3;
}

inline std::array<int, 4> non_orthogonal_selection_state(
    NonOrthogonalTreatment treatment, int correction_state) noexcept
{
    const auto code = non_orthogonal_treatment_code(treatment);
    return {code, -code, correction_state, -correction_state};
}

inline void report_non_orthogonal_selection_state(const std::array<int, 4>& state, std::string_view context)
{
    const auto prefix = std::string(context);
    if (state[0] == 3)
        throw std::invalid_argument(prefix + " received an unknown non-orthogonal treatment.");
    if (state[0] != -state[1])
        throw std::invalid_argument(prefix + " requires every rank to use the same non-orthogonal treatment.");
    if (state[2] != -state[3])
        throw std::invalid_argument(prefix + " requires every rank to select the same category of correction field.");
    if (state[2] == 2)
        throw std::invalid_argument(prefix + " requires the correction field on the transported-field mesh.");
}

template<class Scalar> NonOrthogonalWeights<Scalar> non_orthogonal_weights(NonOrthogonalTreatment treatment)
{
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            return {Scalar{}, Scalar{1}};
        case NonOrthogonalTreatment::Implicit:
            return {Scalar{1}, Scalar{}};
        case NonOrthogonalTreatment::Hybrid:
            return {Scalar{0.5}, Scalar{0.5}};
    }
    throw std::invalid_argument("Unknown non-orthogonal treatment.");
}

template<class Scalar, class MeshType>
NonOrthogonalWeights<Scalar> validate_non_orthogonal_selection(
    const MeshType& mesh, NonOrthogonalTreatment treatment, int correction_state, std::string_view context)
{
    const auto state =
        reduce_transport_validation_state(mesh, non_orthogonal_selection_state(treatment, correction_state));
    report_non_orthogonal_selection_state(state, context);
    return non_orthogonal_weights<Scalar>(treatment);
}

} // namespace SimpleFluid::FVM::detail
