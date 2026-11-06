/** @file SmallMembership.hh
 * @brief Inline small membership ranges with immutable shared overflow storage. */
#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
template<class T> class SmallMembership
{
public:
    using Spill = std::vector<T>;
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return !size_; }
    [[nodiscard]] const T* begin() const noexcept { return spill_ ? spill_->data() : inline_.data(); }
    [[nodiscard]] const T* end() const noexcept { return begin() + size_; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return begin()[i]; }
    [[nodiscard]] const std::shared_ptr<const Spill>& spill() const noexcept { return spill_; }

    template<class Get> static SmallMembership make(std::size_t count, Get get)
    {
        return make(count, get, [] { return std::make_shared<Spill>(); });
    }
    template<class Get, class CreateSpill>
    static SmallMembership make(std::size_t count, Get get, CreateSpill create_spill)
    {
        SmallMembership result;
        result.size_ = count;
        if (count <= result.inline_.size())
        {
            for (std::size_t i = 0; i < count; ++i)
                result.inline_[i] = get(i);
        }
        else
        {
            auto values = create_spill();
            values->reserve(count);
            if (values->capacity() != count)
                throw std::invalid_argument("Membership capacity differs from admission.");
            for (std::size_t i = 0; i < count; ++i)
                values->push_back(get(i));
            result.spill_ = std::move(values);
        }
        return result;
    }

private:
    std::array<T, 2> inline_{};
    std::shared_ptr<const Spill> spill_;
    std::size_t size_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
