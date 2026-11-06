/** @file AdaptiveLeafOverlay.hh
 * @brief Candidate-only edits over an immutable ordered AMR leaf tree.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"

#include <map>
#include <optional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Sparse candidate rows and retirements. Read through to the pinned source
 * tree and keep the source tree unchanged throughout closure. The caller
 * admits incoming row bodies and keeps the metadata arena alive. */
class AdaptiveLeafOverlay
{
public:
    AdaptiveLeafOverlay(AdaptiveLeafPages source, Budget& arena);
    ~AdaptiveLeafOverlay();
    AdaptiveLeafOverlay(const AdaptiveLeafOverlay&) = delete;
    AdaptiveLeafOverlay& operator=(const AdaptiveLeafOverlay&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t edit_count() const noexcept { return edits_.size(); }
    [[nodiscard]] const AdaptiveLeaf* find(const AdaptiveCellKey&) const;
    [[nodiscard]] const AdaptiveLeaf& at(const AdaptiveCellKey&) const;
    [[nodiscard]] bool contains(const AdaptiveCellKey& key) const { return find(key) != nullptr; }

    /** Activate an absent key; a retired source key becomes a replacement. */
    bool insert(AdaptiveLeaf&&);
    /** Retire an active key; cancelling a new insertion leaves no edit. */
    bool retire(const AdaptiveCellKey&);
    /** Replace an active record with its normalized representation. */
    void replace_active(AdaptiveLeaf&&);

    template<class Visitor> void visit_owned(Visitor&& visitor) const
    {
        auto row = source_.begin();
        const auto end = source_.end();
        auto edit = edits_.begin();
        while (row != end || edit != edits_.end())
        {
            if (edit == edits_.end() || (row != end && row->key < edit->first))
            {
                visitor(*row++);
                continue;
            }
            if (row != end && row->key == edit->first)
                ++row;
            if (edit->second.leaf)
                visitor(*edit->second.leaf);
            ++edit;
        }
    }

    template<class Visitor> void visit_edits(Visitor&& visitor) const
    {
        for (const auto& [key, edit] : edits_)
            visitor(AdaptiveLeafPages::EditView{key, edit.source_revision, edit.leaf ? &*edit.leaf : nullptr});
    }

    [[nodiscard]] const AdaptiveLeafPages& source() const noexcept { return source_; }

private:
    struct Pending
    {
        std::optional<std::uint64_t> source_revision;
        std::optional<AdaptiveLeaf> leaf;
        std::size_t bytes;
    };
    using Edits = std::map<AdaptiveCellKey, Pending>;

    [[nodiscard]] Edits::iterator append(const AdaptiveCellKey&, std::optional<std::uint64_t>);
    void discard(Edits::iterator) noexcept;

    AdaptiveLeafPages source_;
    Budget& arena_;
    Edits edits_;
    std::size_t size_, charged_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
