#pragma once

#include "DistributedAdaptiveTypes.hh"
#include <algorithm>
#include <array>
#include <compare>
#include <span>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** A contiguous key value. Packed keys expand into the value itself. Literal keys
 * borrow the immutable pool body or captured page supplied by the caller. */
class IncidenceSupportKey
{
public:
    IncidenceSupportKey() = default;
    explicit IncidenceSupportKey(std::span<const char> bytes) noexcept : literal_(bytes) {}

    [[nodiscard]] const char* data() const noexcept { return inline_size_ ? inline_.data() : literal_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return inline_size_ ? inline_size_ : literal_.size(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] const char* begin() const noexcept { return data(); }
    [[nodiscard]] const char* end() const noexcept { return data() + size(); }
    [[nodiscard]] char operator[](std::size_t index) const noexcept { return data()[index]; }
    [[nodiscard]] std::span<const char> bytes() const& noexcept { return {data(), size()}; }
    std::span<const char> bytes() const&& = delete;

    [[nodiscard]] bool operator==(const IncidenceSupportKey& other) const noexcept
    {
        return std::ranges::equal(*this, other);
    }
    [[nodiscard]] bool operator==(const AdaptiveBytes& other) const noexcept
    {
        return std::ranges::equal(*this, other);
    }
    [[nodiscard]] std::strong_ordering operator<=>(const IncidenceSupportKey& other) const noexcept
    {
        return std::lexicographical_compare_three_way(begin(), end(), other.begin(), other.end());
    }
    [[nodiscard]] std::strong_ordering operator<=>(const AdaptiveBytes& other) const noexcept
    {
        return std::lexicographical_compare_three_way(begin(), end(), other.begin(), other.end());
    }

private:
    friend class CompactSupportDirectory;
    static constexpr std::size_t max_domain = 32;
    std::array<char, 64 + max_domain> inline_{};
    std::span<const char> literal_;
    std::size_t inline_size_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
