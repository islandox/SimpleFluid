/** @file ALEFixtures.hh
 * @brief Custom mesh shared with its private ALE validation provider.
 */
#pragma once

#include "geometry/MeshHandle.hh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace SimpleFluid::test::ale
{
using Handle = MeshHandle<>;

class ThrowingHandle : public Handle
{
public:
    using Handle::Handle;
    bool fail_epoch = false, fail_faces = false, fail_owner = false;
    mutable size_t face_traversals = 0;

    std::uint64_t geometry_epoch() const;

    CellFaceRange faces(local_ordinal_type cell) const;
    local_ordinal_type owner_cell(local_ordinal_type face) const;
};
} // namespace SimpleFluid::test::ale

#include "ALEFixtures.ipp"
