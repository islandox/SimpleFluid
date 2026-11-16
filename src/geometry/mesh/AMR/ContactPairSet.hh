/** @file ContactPairSet.hh
 * @brief Exact, budgeted contact deduplication with ascending pair traversal.
 */
#pragma once
#include "utils/export_helper.hh"
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget;
/** Construction-only exact set. The budget must outlive this object and every
 * sorted view it produces. Compact storage is used only for actual IDs fitting
 * uint32_t; larger identities retain their complete size_t values. No geometry
 * owners or borrowed references survive insertion. */
class SIMPLEFLUID_LOCAL ContactPairSet
{
public:
    using Pair = std::pair<std::size_t, std::size_t>;
    class SIMPLEFLUID_LOCAL SortedPairs
    {
    public:
        class const_iterator
        {
        public:
            using value_type = Pair;
            using reference = Pair;
            using pointer = void;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;
            const_iterator() = default;
            Pair operator*() const noexcept;
            const_iterator& operator++() noexcept
            {
                ++index_;
                return *this;
            }
            const_iterator operator++(int) noexcept
            {
                auto previous = *this;
                ++*this;
                return previous;
            }
            bool operator==(const const_iterator&) const = default;

        private:
            friend class SortedPairs;
            const_iterator(const SortedPairs* owner, std::size_t index) : owner_(owner), index_(index) {}
            const SortedPairs* owner_ = nullptr;
            std::size_t index_ = 0;
        };
        SortedPairs() = default;
        ~SortedPairs();
        SortedPairs(SortedPairs&&) noexcept;
        SortedPairs& operator=(SortedPairs&&) noexcept;
        SortedPairs(const SortedPairs&) = delete;
        SortedPairs& operator=(const SortedPairs&) = delete;
        [[nodiscard]] std::size_t size() const noexcept { return size_; }
        [[nodiscard]] std::size_t storage_bytes() const noexcept { return charged_; }
        [[nodiscard]] const_iterator begin() const noexcept { return {this, 0}; }
        [[nodiscard]] const_iterator end() const noexcept { return {this, size_}; }
        /** Destroy buffers before returning their live credit. Cumulative
         * admission remains monotone. Iterators are invalidated. */
        void clear() noexcept;

    private:
        friend class ContactPairSet;
        Budget* budget_ = nullptr;
        std::size_t size_ = 0, charged_ = 0;
        std::unique_ptr<std::uint64_t[]> compact_;
        std::unique_ptr<Pair[]> wide_;
    };

    explicit ContactPairSet(Budget&) noexcept;
    ~ContactPairSet();
    ContactPairSet(const ContactPairSet&) = delete;
    ContactPairSet& operator=(const ContactPairSet&) = delete;
    /** Requires first < second. Returns false for an exact duplicate. Failed
     * admission/allocation preserves all existing pairs and their storage. */
    bool insert(Pair);
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return charged_; }
    /** Pre-admit the exact sorted buffer while the table is live. After its
     * complete ascending construction, retire the table and empty the set.
     * Failed admission/allocation leaves the set unchanged. */
    [[nodiscard]] SortedPairs take_sorted();

private:
    void rebuild(std::size_t capacity, bool wide);
    void clear() noexcept;
    Budget* budget_;
    std::size_t size_ = 0, capacity_ = 0, charged_ = 0;
    std::unique_ptr<std::uint64_t[]> compact_;
    std::unique_ptr<Pair[]> wide_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
