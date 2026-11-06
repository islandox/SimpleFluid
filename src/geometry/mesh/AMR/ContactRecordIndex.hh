/** @file ContactRecordIndex.hh
 * @brief Exact typed-key lookups over persistent contact record slots.
 */
#pragma once
#include "ContactSlotIndex.hh"
#include "geometry/mesh/AMR/AdaptiveHierarchy.hh"
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class ContactRecordIndex
{
public:
    explicit ContactRecordIndex(std::shared_ptr<Budget> arena) : index_(std::move(arena)) {}
    ContactRecordIndex(const ContactRecordIndex& source, std::shared_ptr<Budget> arena)
        : index_(source.index_, std::move(arena)) {}
    template<class KeyAt>
    [[nodiscard]] std::optional<std::size_t> find(const AdaptiveCellKey& key, const KeyAt& at) const
    {
        return index_.find(hash(key), [&](std::size_t slot) { return at(slot) == key; });
    }
    template<class KeyAt> void insert(std::size_t id, const KeyAt& at)
    {
        index_.insert(id, [&](std::size_t slot) { return hash(at(slot)); });
    }
    void erase(const AdaptiveCellKey& key, std::size_t id) { index_.erase(hash(key), id); }
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return index_.storage_bytes(); }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return index_.overlap_bytes(); }
    [[nodiscard]] std::size_t copied_pages() const noexcept { return index_.copied_pages(); }
    [[nodiscard]] std::size_t split_records() const noexcept { return index_.split_records(); }
    void finish_overlap() noexcept { index_.finish_overlap(); }
    template<class Visit> void visit_pages(Visit&& visit) const { index_.visit_pages(visit); }

private:
    static std::uint64_t hash(const AdaptiveCellKey& key)
    {
        std::uint64_t seed = 0;
        const auto word = [&](std::uint64_t value)
        {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            seed ^= (value ^ (value >> 31)) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        const auto path = [&](const auto& values)
        {
            word(values.size());
            for (const auto value : values)
                word(value);
        };
        word(key.region);
        word(key.ancestry.index());
        std::visit([&](const auto& ancestry)
            {
                using Ancestry = std::decay_t<decltype(ancestry)>;
                if constexpr (std::is_same_v<Ancestry, CartesianCellAncestry>)
                {
                    word(ancestry.level);
                    for (const auto index : ancestry.index)
                        word(index);
                }
                else if constexpr (std::is_same_v<Ancestry, ConvexCellAncestry>)
                {
                    word(ancestry.root);
                    path(ancestry.path);
                }
                else if constexpr (std::is_same_v<Ancestry, ProductCellAncestry>)
                {
                    word(ancestry.base.root);
                    word(ancestry.base.root_template);
                    path(ancestry.base.path);
                    word(ancestry.axial.root);
                    path(ancestry.axial.path);
                }
                else
                {
                    word(ancestry.provider_root);
                    word(ancestry.native_cell);
                }
            }, key.ancestry);
        return seed;
    }

    ContactSlotIndex index_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
