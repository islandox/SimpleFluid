/**
 * @file BoundaryFacePatch.hh
 * @brief Boundary-face patch data shared by mesh implementations.
 */

#pragma once

#include <vector>

namespace SimpleFluid::Meshes
{

template <class FaceID>
struct BoundaryFacePatch
{
    int id = -1;
    std::vector<FaceID> face_lids;
};

} // namespace SimpleFluid::Meshes
