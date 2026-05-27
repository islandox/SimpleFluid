/**
 * @file EquationValidation.hh
 * @brief Shared validation helpers for equation classes.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace SimpleFluid::EquationValidation
{

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

inline void assert_sufficient_cache_size(std::size_t cache_size,
                                         std::size_t required_size)
{
    assert(cache_size >= required_size);
    (void)cache_size;
    (void)required_size;
}

} // namespace SimpleFluid::EquationValidation
