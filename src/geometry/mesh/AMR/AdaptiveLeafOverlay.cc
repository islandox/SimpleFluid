/** @file AdaptiveLeafOverlay.cc
 * @brief Budgeted sparse owner-local edits before AMR publication.
 */
#include "geometry/mesh/AMR/AdaptiveLeafOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"

#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
std::size_t key_buffers(const AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& ancestry) -> std::size_t
        {
            if constexpr (requires { ancestry.path; })
                return ancestry.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (requires { ancestry.base.path; })
                return (ancestry.base.path.capacity() + ancestry.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
} // namespace

AdaptiveLeafOverlay::AdaptiveLeafOverlay(AdaptiveLeafPages source, Budget& arena)
    : source_(std::move(source)), arena_(arena), size_(source_.size())
{
}

AdaptiveLeafOverlay::~AdaptiveLeafOverlay()
{
    edits_.clear();
    arena_.used -= charged_;
}

const AdaptiveLeaf* AdaptiveLeafOverlay::find(const AdaptiveCellKey& key) const
{
    if (const auto edit = edits_.find(key); edit != edits_.end())
        return edit->second.leaf ? &*edit->second.leaf : nullptr;
    return source_.find(key);
}

const AdaptiveLeaf& AdaptiveLeafOverlay::at(const AdaptiveCellKey& key) const
{
    if (const auto* row = find(key))
        return *row;
    throw std::out_of_range("Adaptive candidate leaf key is not locally owned.");
}

AdaptiveLeafOverlay::Edits::iterator AdaptiveLeafOverlay::append(
    const AdaptiveCellKey& key, std::optional<std::uint64_t> source_revision)
{
    const auto capacity = key_buffers(key);
    const auto bytes = sizeof(Edits::value_type) + 4 * sizeof(void*) + capacity;
    arena_.add(bytes);
    try
    {
        auto [position, inserted] = edits_.emplace(key, Pending{source_revision, {}, bytes});
        if (!inserted)
            throw std::logic_error("Duplicate adaptive leaf edit.");
        if (key_buffers(position->first) > capacity)
        {
            edits_.erase(position);
            throw std::invalid_argument("Adaptive overlay key capacity exceeds preadmission.");
        }
        charged_ += bytes;
        return position;
    }
    catch (...)
    {
        arena_.used -= bytes;
        throw;
    }
}

void AdaptiveLeafOverlay::discard(Edits::iterator position) noexcept
{
    const auto bytes = position->second.bytes;
    edits_.erase(position);
    charged_ -= bytes;
    arena_.used -= bytes;
}

bool AdaptiveLeafOverlay::insert(AdaptiveLeaf&& leaf)
{
    auto position = edits_.find(leaf.key);
    if (position != edits_.end() ? bool(position->second.leaf) : source_.contains(leaf.key))
        return false;
    if (size_ == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Adaptive candidate leaf count overflows.");
    if (position == edits_.end())
        position = append(leaf.key, {});
    try
    {
        position->second.leaf.emplace(std::move(leaf));
    }
    catch (...)
    {
        if (!position->second.source_revision)
            discard(position);
        throw;
    }
    ++size_;
    return true;
}

bool AdaptiveLeafOverlay::retire(const AdaptiveCellKey& key)
{
    if (const auto edit = edits_.find(key); edit != edits_.end())
    {
        if (!edit->second.leaf)
            return false;
        if (edit->second.source_revision)
            edit->second.leaf.reset();
        else
            discard(edit);
    }
    else if (const auto* row = source_.find(key))
        (void) append(key, row->revision);
    else
        return false;
    --size_;
    return true;
}

void AdaptiveLeafOverlay::replace_active(AdaptiveLeaf&& leaf)
{
    auto edit = edits_.find(leaf.key);
    const auto* row = edit == edits_.end() ? source_.find(leaf.key)
                                          : (edit->second.leaf ? &*edit->second.leaf : nullptr);
    if (!row || row->revision != leaf.revision)
        throw std::invalid_argument("Normalized adaptive leaf differs from candidate key or revision.");
    if (edit == edits_.end())
        edit = append(leaf.key, row->revision);
    edit->second.leaf.emplace(std::move(leaf));
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
