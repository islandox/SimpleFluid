#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** One arena's admission. Shared immutable owners have one such credit per arena. */
class ContactGroupAllocation
{
public:
    explicit ContactGroupAllocation(std::shared_ptr<Budget> arena) : arena_(std::move(arena))
    {
        if (!arena_)
            throw std::invalid_argument("Contact storage requires a shared arena.");
    }
    ~ContactGroupAllocation() { arena_->used -= bytes_; }
    ContactGroupAllocation(ContactGroupAllocation&& other) noexcept
        : arena_(other.arena_), bytes_(std::exchange(other.bytes_, 0)) {}
    ContactGroupAllocation(const ContactGroupAllocation&) = delete;
    ContactGroupAllocation& operator=(const ContactGroupAllocation&) = delete;
    void add(std::size_t bytes)
    {
        if (bytes > std::numeric_limits<std::size_t>::max() - bytes_)
            throw std::invalid_argument("Contact storage admission overflows.");
        arena_->add(bytes);
        bytes_ += bytes;
    }
    void release(std::size_t bytes) noexcept { arena_->used -= bytes; bytes_ -= bytes; }
    void transfer_from(ContactGroupAllocation& source, std::size_t bytes)
    {
        if (arena_.get() != source.arena_.get() || bytes > source.bytes_ ||
            bytes > std::numeric_limits<std::size_t>::max() - bytes_)
            throw std::invalid_argument("Invalid contact storage admission transfer.");
        source.bytes_ -= bytes;
        bytes_ += bytes;
    }
    std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const void* identity() const noexcept { return &bytes_; }
    Budget& arena() const noexcept { return *arena_; }
    const std::shared_ptr<Budget>& keeper() const noexcept { return arena_; }
    static std::size_t product(std::size_t count, std::size_t size)
    {
        if (size && count > std::numeric_limits<std::size_t>::max() / size)
            throw std::invalid_argument("Contact storage capacity overflows.");
        return count * size;
    }
    template<class T> void reserve(std::vector<T>& values, std::size_t count)
    {
        if (count <= values.capacity())
            return;
        const auto next = product(count, sizeof(T)), old = product(values.capacity(), sizeof(T));
        add(next);
        try
        {
            std::vector<T> replacement;
            replacement.reserve(count);
            if (replacement.capacity() != count)
                throw std::invalid_argument("Contact storage capacity differs from admission.");
            replacement.assign(values.begin(), values.end());
            values.swap(replacement);
        }
        catch (...)
        {
            release(next);
            throw;
        }
        release(old);
    }
    template<class T> void append(std::vector<T>& values)
    {
        if (values.size() < values.capacity())
            return;
        const auto old = values.capacity();
        if (old > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Contact storage capacity overflows.");
        reserve(values, old ? 2 * old : 1);
    }
private:
    std::shared_ptr<Budget> arena_;
    std::size_t bytes_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
