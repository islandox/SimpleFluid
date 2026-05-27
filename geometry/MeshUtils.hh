/**
 * @file MeshUtils.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "dataclass/vec3.hh"

namespace SimpleFluid
{

namespace MeshUtils
{
enum class CellType : uint8_t
{
    INVALID = 0,
    TETRAHEDRON = 1,
    HEXAHEDRON = 2,
    TRIPRISM = 3
};

enum class FaceType : uint8_t
{
    INVALID = 0,
    QUAD = 1,
    TRIANGLE = 2
};

using Vec3 = vec3<real_t>;
} // namespace MeshUtils

} // namespace SimpleFluid
