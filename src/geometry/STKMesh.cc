/**
 * @file STKMesh.cc
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for the STKMesh class.
 * @version 0.1
 * @date 2026-05-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "STKMesh.hh"
#include "STKMesh.tcc"

namespace SimpleFluid
{
template class STKMesh<DefaultTpetraTypes>;
}