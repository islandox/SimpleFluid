/**
 * @file FVM/NonOrthogonalTreatment.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Runtime switch for non-orthogonal diffusion treatments.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace SimpleFluid::FVM
{

/**
 * @brief Available treatments for non-orthogonal diffusion terms.
 */
enum class NonOrthogonalTreatment
{
    Explicit,
    Implicit,
    Hybrid
};

/**
 * @brief Parse the runtime non-orthogonal treatment switch.
 *
 * Accepted values match the input-file spelling:
 * `explicit`, `implicit`, and `hybrid`.
 */
inline NonOrthogonalTreatment
non_orthogonal_treatment_from_string(std::string_view value)
{
    if (value == "explicit")
    {
        return NonOrthogonalTreatment::Explicit;
    }
    if (value == "implicit")
    {
        return NonOrthogonalTreatment::Implicit;
    }
    if (value == "hybrid")
    {
        return NonOrthogonalTreatment::Hybrid;
    }

    throw std::invalid_argument(
        "Unknown nonOrthogonalTreatment value: " + std::string(value));
}

/**
 * @brief Return the input-file spelling for a non-orthogonal treatment.
 */
inline std::string_view
to_string(NonOrthogonalTreatment treatment)
{
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit: return "explicit";
        case NonOrthogonalTreatment::Implicit: return "implicit";
        case NonOrthogonalTreatment::Hybrid:   return "hybrid";
    }

    throw std::invalid_argument("Unknown NonOrthogonalTreatment value.");
}

} // namespace SimpleFluid::FVM
