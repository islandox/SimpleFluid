/** @file StoredMeshIdentity.hh
 * @brief Records stable mesh identity for reusable stored operators.
 */
#pragma once

#include <stdexcept>
#include <string>

namespace SimpleFluid::FVM::detail
{

template<class InputField, class OutputField>
void require_same_stored_mesh(const InputField& input, const OutputField& output, const char* operation)
{
    if (input.mesh_ptr().get() != output.mesh_ptr().get())
        throw std::invalid_argument(std::string(operation) + " requires input and output fields on one mesh.");
}

} // namespace SimpleFluid::FVM::detail
