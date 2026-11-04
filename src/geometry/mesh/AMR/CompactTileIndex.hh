#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Candidate-private metadata. Published descriptors are sorted by the complete
 * support/emission key. A batch keeps them immutable until one admitted merge;
 * entry addresses survive index growth. Entry pages have sixteen slots and a
 * single occupancy bitmap. Dead slots become reusable after publication only.
 * The optional conservative slot ceiling is for synthetic width-boundary tests;
 * production retains the full uint32 range. No wire/global identity is narrowed. */
template<class Entry, std::size_t CompactSlotLimit = std::numeric_limits<std::uint32_t>::max()> class CompactTileIndex
{
public:
    using Key = std::pair<std::size_t, std::uint64_t>;
    struct CompactDescriptor
    {
        std::uint64_t emission = 0;
        std::uint32_t support = 0, slot = 0;
    };
    struct WideDescriptor
    {
        Key key{};
        std::size_t slot = 0;
    };
    explicit CompactTileIndex(Budget&, std::shared_ptr<Budget> keeper = {});
    ~CompactTileIndex();
    CompactTileIndex(const CompactTileIndex&) = delete;
    CompactTileIndex& operator=(const CompactTileIndex&) = delete;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool wide() const noexcept { return wide_mode_; }
    [[nodiscard]] Key key_at(std::size_t) const noexcept;
    [[nodiscard]] Entry& value_at(std::size_t) noexcept;
    [[nodiscard]] const Entry& value_at(std::size_t) const noexcept;
    [[nodiscard]] std::size_t lower_bound(Key) const noexcept;
    /** Borrow only the current publication. Addresses remain valid until that
     * support is retired by a successful batch; failed publication preserves
     * them. Staged insertions remain invisible to all queries. */
    [[nodiscard]] Entry* find(Key) noexcept;
    [[nodiscard]] const Entry* find(Key) const noexcept;
    [[nodiscard]] Entry& at(Key);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t last_merge_comparisons() const noexcept { return merge_comparisons_; }
    /** Clone one live metadata record per source key; the transform explicitly
     * rebinds body handles into the candidate's already-forked body arena. */
    template<class Transform> void clone_from(const CompactTileIndex&, Transform&&);

private:
    static_assert(std::is_nothrow_move_constructible_v<Entry> && std::is_nothrow_destructible_v<Entry>);
    static_assert(CompactSlotLimit <= std::numeric_limits<std::uint32_t>::max());
    static constexpr std::size_t page_size = 16, absent = std::numeric_limits<std::size_t>::max();
    template<class T> class Array
    {
    public:
        explicit Array(Budget& b) : budget_(&b) {}
        ~Array();
        Array(const Array&) = delete;
        Array& operator=(const Array&) = delete;
        void reserve(std::size_t);
        void clear() noexcept;
        void append(T);
        void swap(Array&) noexcept;
        [[nodiscard]] std::size_t bytes() const noexcept { return capacity * sizeof(T); }
        T& operator[](std::size_t i) noexcept { return data[i]; }
        const T& operator[](std::size_t i) const noexcept { return data[i]; }
        struct DestroyArray
        {
            std::size_t count = 0;
            void operator()(T*) const noexcept;
        };
        std::unique_ptr<T[], DestroyArray> data;
        std::size_t count = 0, capacity = 0;

    private:
        Budget* budget_;
    };
    union Cell
    {
        Entry value;
        std::size_t next;
        Cell() : next(absent) {}
        ~Cell() {}
    };
    struct Page
    {
        std::array<Cell, page_size> cells;
        std::uint16_t occupied = 0;
        std::size_t directory_slot = 0;
        ~Page();
    };
    [[nodiscard]] std::size_t slot_at(std::size_t) const noexcept;
    [[nodiscard]] Entry& entry(std::size_t) noexcept;
    [[nodiscard]] const Entry& entry(std::size_t) const noexcept;
    std::size_t insert_value(Entry);
    void retire_value(std::size_t) noexcept;
    void reap_empty_pages() noexcept;
    Budget* budget_;
    std::shared_ptr<Budget> keeper_;
    Array<std::unique_ptr<Page>> pages_;
    Array<CompactDescriptor> compact_;
    Array<WideDescriptor> wide_;
    std::size_t free_ = absent, first_empty_page_ = 0, page_bytes_ = 0, merge_comparisons_ = 0;
    bool wide_mode_ = false, batch_live_ = false;

public:
    class Batch
    {
    public:
        explicit Batch(CompactTileIndex&);
        ~Batch();
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;
        void replace_support(std::size_t);
        void insert(Key, Entry);
        void publish(std::size_t tile_limit = std::numeric_limits<std::size_t>::max());

    private:
        CompactTileIndex* owner_;
        Array<std::size_t> retired_;
        Array<WideDescriptor> staged_;
        bool published_ = false;
    };
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
#include "CompactTileIndex.ipp"
