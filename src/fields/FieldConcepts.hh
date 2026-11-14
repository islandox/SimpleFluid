/** @file FieldConcepts.hh
 * @brief Storage-free field locations and descriptor contract.
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <string>
#include <type_traits>

namespace SimpleFluid
{
/** @brief Location tag for cell-centered fields. */
struct CellLocation
{
};
/** @brief Location tag for face-centered fields. */
struct FaceLocation
{
};
/** @brief Location tag for values stored only on boundary faces. */
struct BoundaryFaceLocation
{
};

/**
 * @brief Named, identified descriptor at one supported field location.
 *
 * No inheritance, storage, default constructor or string constructor is
 * required. FieldStored separately checks scalar compatibility with its pack.
 * name() must return a stable reference, as storage forwards that reference.
 */
template<class F>
concept FieldType =
    requires {
        typename F::value_type;
        typename F::location_type;
    } && std::is_object_v<typename F::value_type> &&
    (std::same_as<typename F::location_type, CellLocation> || std::same_as<typename F::location_type, FaceLocation> ||
        std::same_as<typename F::location_type, BoundaryFaceLocation>) &&
    requires(const F& field) {
        { field.id() } -> std::convertible_to<size_t>;
        { field.name() } -> std::convertible_to<const std::string&>;
        requires std::is_lvalue_reference_v<decltype(field.name())>;
        requires std::derived_from<std::remove_cvref_t<decltype(field.name())>, std::string>;
    };
} // namespace SimpleFluid
