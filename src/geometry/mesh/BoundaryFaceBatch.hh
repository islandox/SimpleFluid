/**
 * @file BoundaryFaceBatch.hh
 * @brief Boundary-face batch data shared by mesh implementations.
 */

#pragma once

#include <vector>

namespace SimpleFluid::Meshes
{

template <class FaceID>
struct BoundaryFaceBatch
{
    int id = -1;
    std::vector<FaceID> face_lids;
};

} // namespace SimpleFluid::Meshes
