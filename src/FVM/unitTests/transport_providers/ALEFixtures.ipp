/** @file ALEFixtures.ipp
 * @brief Inline mesh queries with injected geometry and traversal failures.
 */
#pragma once

namespace SimpleFluid::test::ale
{
inline std::uint64_t ThrowingHandle::geometry_epoch() const
{
    if (fail_epoch) throw std::runtime_error("Injected rank-local static geometry failure.");
    return Handle::geometry_epoch();
}

inline ThrowingHandle::CellFaceRange ThrowingHandle::faces(local_ordinal_type cell) const
{
    ++face_traversals;
    if (fail_faces) throw std::runtime_error("Injected rank-local face traversal failure.");
    return Handle::faces(cell);
}

inline ThrowingHandle::local_ordinal_type ThrowingHandle::owner_cell(local_ordinal_type face) const
{
    if (fail_owner) throw std::runtime_error("Injected rank-local owner lookup failure.");
    return Handle::owner_cell(face);
}
} // namespace SimpleFluid::test::ale
