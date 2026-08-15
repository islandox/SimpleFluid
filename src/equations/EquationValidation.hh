/**
 * @file EquationValidation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared validation helpers for equation classes.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace SimpleFluid::EquationValidation
{

/**
 * @brief Assert that a mesh pointer is non-null and return it.
 *
 * @tparam MeshPtr Type of the mesh pointer.
 * @param mesh The mesh pointer to validate.
 * @param class_name Name of the calling class for error messages.
 * @return The validated non-null @p mesh pointer.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<class MeshPtr>
MeshPtr require_non_null_mesh(MeshPtr mesh, const char* class_name)
{
    if (!mesh)
    {
        throw std::invalid_argument(std::string(class_name)
                                  + " requires a non-null mesh.");
    }

    return mesh;
}

/**
 * @brief Verify that a field is associated with the expected mesh.
 *
 * @tparam Mesh Type of the mesh.
 * @tparam Field Type of the field to check.
 * @param expected The expected mesh reference.
 * @param field The field whose mesh association is checked.
 * @param class_name Name of the calling class for error messages.
 * @throws std::invalid_argument if the field's mesh does not match @p expected.
 */
template<class Mesh, class Field>
void require_mesh_match(const Mesh& expected,
                        const Field& field,
                        const char* class_name)
{
    if (&field.mesh() != &expected)
    {
        throw std::invalid_argument(std::string(class_name)
                                  + " field mesh mismatch.");
    }
}

/**
 * @brief Assert that a scalar value is non-negative.
 *
 * @tparam Scalar The scalar type.
 * @param value The value to check.
 * @param parameter_name Name of the parameter for error messages.
 * @param class_name Name of the calling class for error messages.
 * @throws std::invalid_argument if @p value is negative.
 */
template<class Scalar>
void require_non_negative(Scalar value,
                          const char* parameter_name,
                          const char* class_name)
{
    if (value < Scalar{})
    {
        throw std::invalid_argument(std::string(class_name)
                                  + " requires non-negative "
                                  + parameter_name + ".");
    }
}

/**
 * @brief Assert that a cache container holds at least the required number
 *        of elements.
 *
 * @param cache_size Current size of the cache.
 * @param required_size Minimum required size.
 */
inline void assert_sufficient_cache_size(size_t cache_size,
                                         size_t required_size)
{
    assert(cache_size >= required_size);
    (void)cache_size;
    (void)required_size;
}

} // namespace SimpleFluid::EquationValidation
