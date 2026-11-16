#pragma once
#include "AdaptiveHierarchy.hh"
#include "ContactPointArray.hh"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** A row borrowing an immutable accepted snapshot. Key and domain must refer
 * into snapshot; points independently retain that same pin so copying the point
 * reader out of the row also preserves its lifetime. Construction copies shared
 * ownership without allocating a control block or materializing row storage. */
struct ContactIncidenceRowView
{
    std::shared_ptr<const void> snapshot;
    const AdaptiveCellKey* key = nullptr;
    std::uint64_t revision = 0;
    std::span<const char> domain;
    ContactPointArray points;
    std::uint64_t incidence_generation = 0;
};

/** Indexed immutable rows pinned before traversal. Readers must return borrows
 * into the supplied pin and retain it in both the row and its generated points.
 * Neither range construction nor row access allocates. A later publication must
 * replace the accepted snapshot rather than mutate the object pinned here. */
class ContactIncidenceRowsView
{
public:
    using Reader = ContactIncidenceRowView (*)(const std::shared_ptr<const void>&, std::size_t);

    ContactIncidenceRowsView() = default;
    ContactIncidenceRowsView(std::shared_ptr<const void> snapshot, std::size_t count, Reader read)
        : snapshot_(std::move(snapshot)), count_(count), read_(read)
    {
        if (count && (!snapshot_ || !read_))
            throw std::invalid_argument("Contact row views require an immutable snapshot and reader.");
    }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
    [[nodiscard]] ContactIncidenceRowView row(std::size_t index) const
    {
        if (index >= count_)
            throw std::out_of_range("Contact incidence row index is out of range.");
        return read_(snapshot_, index);
    }

private:
    std::shared_ptr<const void> snapshot_;
    std::size_t count_ = 0;
    Reader read_ = nullptr;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
