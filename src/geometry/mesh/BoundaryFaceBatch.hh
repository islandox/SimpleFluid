/**
 * @file BoundaryFaceBatch.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Boundary-face batch data shared by mesh implementations.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Faces associated with one physical boundary identifier.
 * @tparam FaceID Mesh-specific face identifier type.
 */
template <class FaceID>
struct BoundaryFaceBatch
{
    int id = -1;
    std::vector<FaceID> face_lids;
};

} // namespace SimpleFluid::Meshes
