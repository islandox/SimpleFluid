#pragma once

#include "IdentityPool.hh"
#include "IncidenceSupportKey.hh"
#include "PersistentSlotValues.hh"
#include <limits>
#include <optional>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Exact spelling of Cartesian support keys. Packed slots are private selectors.
 * Complete bytes remain the authority for routing, ordering and exchange. */
class CompactSupportDirectory
{
public:
    using Slot = std::size_t;
    using LiteralPool = IdentityPool<AdaptiveBytes>;

private:
    static constexpr std::uint32_t compact_tag = std::uint32_t{1} << 31;
    static constexpr std::uint64_t magic = 0x5346434152543631ULL;
    static constexpr std::size_t fixed_extent = 64, suffix_extent = 40;
    static constexpr unsigned presence_width = std::numeric_limits<std::uint64_t>::digits;

    [[nodiscard]] static bool tagged(Slot slot) noexcept
    {
        return slot >= compact_tag && slot <= std::numeric_limits<std::uint32_t>::max();
    }

public:
    class Frame
    {
    public:
        [[nodiscard]] IncidenceSupportKey expand(Slot slot) const
        {
            if (!size_ || !tagged(slot) || ((slot >> 24) & 3) == 3)
                throw std::invalid_argument("Invalid compact incidence support selector.");
            IncidenceSupportKey result;
            result.inline_size_ = size_ + suffix_extent;
            std::copy_n(prefix_.begin(), size_, result.inline_.begin());
            result.inline_[size_] = static_cast<char>((slot >> 24) & 3);
            result.inline_[size_ + 8] = static_cast<char>((slot >> 26) & 31);
            result.inline_[size_ + 16] = static_cast<char>((slot >> 16) & 255);
            result.inline_[size_ + 24] = static_cast<char>((slot >> 8) & 255);
            result.inline_[size_ + 32] = static_cast<char>(slot & 255);
            return result;
        }

    private:
        friend class CompactSupportDirectory;
        std::array<char, fixed_extent - suffix_extent + IncidenceSupportKey::max_domain> prefix_{};
        std::size_t size_ = 0;
    };

    CompactSupportDirectory(
        Budget& budget, std::shared_ptr<Budget> keeper = {}, IdentityPoolStorage storage = IdentityPoolStorage::Flat)
        : literal_(budget, keeper, {}, {}, storage), presence_(budget, keeper), budget_(&budget),
          enabled_(literal_.persistent() && std::numeric_limits<Slot>::digits > 32)
    {
    }
    CompactSupportDirectory(const CompactSupportDirectory& source, Budget& budget, std::shared_ptr<Budget> keeper = {})
        : literal_(source.literal_, budget, keeper), presence_(source.presence_, budget, keeper), budget_(&budget),
          frame_(source.frame_), compact_count_(source.compact_count_), enabled_(source.enabled_)
    {
    }

    [[nodiscard]] const Frame& frame() const noexcept { return frame_; }
    [[nodiscard]] bool is_compact(Slot slot) const noexcept { return enabled_ && tagged(slot); }
    [[nodiscard]] const LiteralPool& literal_pool() const noexcept { return literal_; }
    [[nodiscard]] std::shared_ptr<LiteralPool::PageView> literal_page_view() { return literal_.page_view(); }
    [[nodiscard]] Slot literal_slot(Slot slot) const
    {
        if (is_compact(slot))
            throw std::invalid_argument("Compact incidence support has no literal pool slot.");
        return enabled_ && slot >= compact_tag ? slot - compact_tag : slot;
    }
    [[nodiscard]] IncidenceSupportKey get(Slot slot) const
    {
        if (!is_compact(slot))
            return IncidenceSupportKey(literal_.get(literal_slot(slot)));
        if (!present(slot))
            throw std::invalid_argument("Compact incidence support selector was not interned.");
        return frame_.expand(slot);
    }

    [[nodiscard]] std::optional<Slot> find(const AdaptiveBytes& key) const { return find_borrowed(key); }
    [[nodiscard]] std::optional<Slot> find_bytes(std::span<const char> key) const { return find_borrowed(key); }
    template<class View> [[nodiscard]] std::optional<Slot> find_borrowed(const View& key) const
    {
        if (const auto compact = pack(key))
            return present(*compact) ? compact : std::nullopt;
        if (const auto literal = literal_.find_borrowed(key))
            return encode_literal(*literal);
        return std::nullopt;
    }
    [[nodiscard]] Slot intern(const AdaptiveBytes& key) { return intern_literal_or_compact(key); }
    [[nodiscard]] Slot intern(AdaptiveBytes&& key) { return intern_literal_or_compact(std::move(key)); }
    [[nodiscard]] Slot intern_bytes(std::span<const char> key)
    {
        if (const auto compact = pack(key))
            return remember(*compact, key);
        if (const auto literal = literal_.find_borrowed(key))
            return encode_literal(*literal);
        const auto expected = encode_literal(literal_.size());
        struct Scratch
        {
            Budget& budget;
            std::size_t bytes;
            ~Scratch() { budget.used -= bytes; }
        };
        budget_->add(key.size());
        const Scratch scratch{*budget_, key.size()};
        AdaptiveBytes owned;
        owned.reserve(key.size());
        if (owned.capacity() != key.size())
            throw std::invalid_argument("Incidence support scratch capacity differs from admission.");
        owned.insert(owned.end(), key.begin(), key.end());
        (void) literal_.intern(std::move(owned));
        return expected;
    }

    [[nodiscard]] std::size_t size() const noexcept { return literal_.size() + compact_count_; }
    [[nodiscard]] std::uint64_t scope_id() const noexcept { return literal_.scope_id(); }
    [[nodiscard]] std::size_t source_only_metadata_bytes() const noexcept
    {
        return literal_.source_only_metadata_bytes();
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return literal_.overlap_bytes() + presence_.overlap_bytes();
    }
    void finish_overlap() noexcept
    {
        literal_.finish_overlap();
        presence_.finish_overlap();
    }
    [[nodiscard]] std::size_t page_credit_overhead_bytes() const noexcept
    {
        return literal_.page_credit_overhead_bytes();
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return literal_.metadata_bytes() + presence_.metadata_bytes();
    }
    [[nodiscard]] std::size_t retained_bytes() const noexcept
    {
        return literal_.retained_bytes() + presence_.metadata_bytes();
    }
    template<class Visit> void visit_borrowed_payload(Visit&& visit) const
    {
        literal_.visit_borrowed_payload(visit);
        presence_.visit_borrowed_payload(visit, ContactPayloadRole::IdentityPersistentIndex);
    }

private:
    template<class View> [[nodiscard]] std::optional<Slot> pack(const View& key) const
    {
        if (!enabled_ || key.size() < fixed_extent || key.size() > fixed_extent + IncidenceSupportKey::max_domain)
            return std::nullopt;
        const auto prefix = key.size() - suffix_extent;
        if (frame_.size_)
        {
            if (prefix != frame_.size_)
                return std::nullopt;
            for (std::size_t index = 0; index < prefix; ++index)
                if (static_cast<char>(key[index]) != frame_.prefix_[index])
                    return std::nullopt;
        }
        else
        {
            const auto domain_size = static_cast<std::uint64_t>(key.size() - fixed_extent);
            for (std::size_t byte = 0; byte < 8; ++byte)
                if (static_cast<std::uint8_t>(key[byte]) != ((magic >> (8 * byte)) & 255) ||
                    static_cast<std::uint8_t>(key[8 + byte]) != ((domain_size >> (8 * byte)) & 255))
                    return std::nullopt;
        }
        std::array<std::uint32_t, 5> values{};
        for (std::size_t field = 0; field < values.size(); ++field)
        {
            const auto offset = prefix + 8 * field;
            values[field] = static_cast<std::uint8_t>(key[offset]);
            for (std::size_t byte = 1; byte < 8; ++byte)
                if (key[offset + byte] != 0)
                    return std::nullopt;
        }
        if (values[0] > 2 || values[1] > 31)
            return std::nullopt;
        return compact_tag | (values[1] << 26) | (values[0] << 24) | (values[2] << 16) | (values[3] << 8) | values[4];
    }
    [[nodiscard]] static std::uint64_t bit(Slot slot) noexcept { return std::uint64_t{1} << (slot % presence_width); }
    [[nodiscard]] bool present(Slot slot) const noexcept
    {
        const auto* mask = presence_.find(slot / presence_width);
        return mask && (*mask & bit(slot));
    }
    template<class View> [[nodiscard]] Slot remember(Slot compact, const View& key)
    {
        const auto group = compact / presence_width;
        const auto* old = presence_.find(group);
        const auto mask = bit(compact);
        if (old && (*old & mask))
            return compact;
        presence_.set(group, old ? *old | mask : mask);
        if (!frame_.size_)
        {
            frame_.size_ = key.size() - suffix_extent;
            std::copy_n(key.begin(), frame_.size_, frame_.prefix_.begin());
        }
        ++compact_count_;
        return compact;
    }
    [[nodiscard]] Slot encode_literal(Slot slot) const
    {
        if (!enabled_ || slot < compact_tag)
            return slot;
        if (slot > std::numeric_limits<Slot>::max() - compact_tag)
            throw std::invalid_argument("Incidence literal support slot capacity exhausted.");
        return slot + compact_tag;
    }
    template<class Bytes> [[nodiscard]] Slot intern_literal_or_compact(Bytes&& key)
    {
        if (const auto compact = pack(key))
            return remember(*compact, key);
        if (const auto literal = literal_.find(key))
            return encode_literal(*literal);
        const auto expected = encode_literal(literal_.size());
        (void) literal_.intern(std::forward<Bytes>(key));
        return expected;
    }

    LiteralPool literal_;
    PersistentSlotValues<std::uint64_t> presence_;
    Budget* budget_;
    Frame frame_;
    std::size_t compact_count_ = 0;
    bool enabled_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
