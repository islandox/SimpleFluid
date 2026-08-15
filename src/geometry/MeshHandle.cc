/**
 * @file MeshHandle.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for MeshHandle.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "MeshHandle.hh"
#include "MeshHandle.tcc"

namespace SimpleFluid
{
template class MeshHandle<DefaultTpetraTypes>;
}
