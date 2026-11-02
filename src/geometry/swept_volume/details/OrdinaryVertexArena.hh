#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include <algorithm>
#include <array>
#include <cassert>
#include <compare>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace SimpleFluid::geometry::swept_volume::details
{
/** Private dense storage. Slot identities survive swap-last compaction;
 * numerical and construction identities are independent of the allocation slot.
 */
template<class Support, class Work> class OrdinaryVertexArena
{
public:
    using Weights = std::array<long double, 4>;
    struct SlotId
    {
        std::size_t value = std::numeric_limits<std::size_t>::max();
        auto operator<=>(const SlotId&) const = default;
    };
    struct Record
    {
        Weights weights{}, errors{};
        std::vector<const Support*> supports;
        const void* source = nullptr;
        const void* chart = nullptr;
        std::size_t construction = 0, evidence_version = 0;
        SlotId owning_slot;
        Record(const Weights& w, const Weights& e, std::vector<const Support*>&& incidence, const void* src,
            const void* frame, std::size_t identity, std::size_t version, SlotId slot) noexcept
            : weights(w), errors(e), supports(std::move(incidence)), source(src), chart(frame), construction(identity),
              evidence_version(version), owning_slot(slot)
        {
        }
    };
    class Handle
    {
    public:
        Handle() = default;
        Handle(const Handle& other) : arena_(other.arena_), slot_(other.slot_)
        {
            if (arena_)
            {
                ++arena_->slot(slot_).references;
                ++arena_->work_.stats.vertex_handle_copies;
            }
        }
        Handle(Handle&& other) noexcept : arena_(std::exchange(other.arena_, nullptr)), slot_(other.slot_) {}
        Handle& operator=(Handle other) noexcept
        {
            swap(other);
            return *this;
        }
        ~Handle()
        {
            if (arena_)
                arena_->release(slot_);
        }
        void swap(Handle& other) noexcept
        {
            std::swap(arena_, other.arena_);
            std::swap(slot_, other.slot_);
        }
        const Record& get() const { return *arena_->slot(slot_).record; }
        SlotId slot_id() const { return slot_; }

    private:
        friend class OrdinaryVertexArena;
        Handle(OrdinaryVertexArena& arena, SlotId id) : arena_(&arena), slot_(id) {}
        OrdinaryVertexArena* arena_ = nullptr;
        SlotId slot_;
    };

    explicit OrdinaryVertexArena(Work& work) : work_(work)
    {
        work_.charge(sizeof(*this));
        observe();
    }
    OrdinaryVertexArena(const OrdinaryVertexArena&) = delete;
    ~OrdinaryVertexArena()
    {
        assert(live_ == 0);
        work_.live_arena_bytes -= observed_arena_;
        work_.live_slot_bytes -= observed_slots_;
        work_.live_incidence_bytes -= observed_incidence_;
        work_.stats.live_vertex_records = 0;
        work_.stats.live_vertex_record_capacity = 0;
        work_.stats.live_vertex_slot_capacity = 0;
        work_.observe_geometry();
    }
    Handle create(const Weights& weights, const Weights& errors, const std::vector<const Support*>& supports,
        const void* source, const void* chart)
    {
        return create_span(weights, errors, supports, source, chart);
    }
    Handle create_span(const Weights& weights, const Weights& errors, std::span<const Support* const> supports,
        const void* source, const void* chart)
    {
        if (next_construction_ == std::numeric_limits<std::size_t>::max())
            counter_limit();
        auto result = make(weights, errors, supports, source, chart, next_construction_, false);
        ++next_construction_;
        return result;
    }
    Handle extend(const Handle& original, const Support* support)
    {
        // Copy numerical/source evidence before any allocation or compaction.
        const auto weights = original.get().weights, errors = original.get().errors;
        const auto source = original.get().source, chart = original.get().chart;
        const auto construction = original.get().construction;
        const auto& old = original.get().supports;
        work_.charge((old.size() + 1) * sizeof(const Support*));
        std::vector<const Support*> incidence;
        incidence.reserve(old.size() + 1);
        work_.charge((incidence.capacity() - old.size() - 1) * sizeof(const Support*));
        incidence.insert(incidence.end(), old.begin(), old.end());
        incidence.push_back(support);
        IncidenceScratch scratch(work_, incidence.capacity() * sizeof(const Support*));
        return make(weights, errors, incidence, source, chart, construction, true);
    }
    /** Consume the target's old evidence version. Shared versions stay immutable;
     * an exclusively owned record can replace its complete incidence in place.
     * Success invalidates borrowed Record/support views of target. Failure
     * preserves target, its version, and every surviving handle. */
    void replace_with_extended_evidence(Handle& target, const Support* support)
    {
        assert(target.arena_ == this);
        auto& metadata = slot(target.slot_);
        if (metadata.references > 1)
        {
            target = extend(target, support);
            return;
        }
        admit_version(true);
        auto& record = *metadata.record;
        const auto required = record.supports.size() + 1;
        work_.charge(sizeof(Record) + required * sizeof(const Support*));
        std::vector<const Support*> incidence;
        incidence.reserve(required);
        IncidenceScratch scratch(work_, incidence.capacity() * sizeof(const Support*));
        work_.charge((incidence.capacity() - required) * sizeof(const Support*));
        incidence.insert(incidence.end(), record.supports.begin(), record.supports.end());
        incidence.push_back(support);
        const auto old_bytes = record.supports.capacity() * sizeof(const Support*);
        const auto new_bytes = incidence.capacity() * sizeof(const Support*);
        // Every admission/allocation has succeeded. Preserve the numerical,
        // source/chart, construction and owning-slot identities at commit.
        record.supports.swap(incidence);
        record.evidence_version = next_evidence_version_++;
        incidence_ = incidence_ - old_bytes + new_bytes;
        // The scratch now owns the old buffer. Transfer the observations
        // together before observing, retaining the real old+new overlap.
        work_.live_incidence_bytes = work_.live_incidence_bytes - scratch.bytes + old_bytes;
        scratch.bytes = old_bytes;
        ++work_.stats.vertex_record_constructions;
        ++work_.stats.vertex_evidence_extensions;
        observe();
    }
    std::size_t live_records() const { return live_; }
    std::size_t record_capacity() const { return pages_.size() * 2; }
    std::size_t slot_capacity() const { return slots_.size() * 4; }

private:
    static constexpr auto absent = std::numeric_limits<std::size_t>::max();
    struct Slot
    {
        Record* record = nullptr;
        std::size_t references = 0, next_free = absent;
    };
    using SlotBlock = std::array<Slot, 4>;
    using Page = std::array<std::optional<Record>, 2>;
    Work& work_;
    std::vector<std::unique_ptr<Page>> pages_;
    std::vector<std::unique_ptr<SlotBlock>> slots_;
    std::size_t live_ = 0, issued_ = 0, free_ = absent, next_construction_ = 0, next_evidence_version_ = 0;
    // A new page admits two first constructions. Only its trailing unused
    // physical position can retain one credit; stable metadata reuse is separate.
    bool virgin_trailing_record_ = false;
    std::size_t observed_arena_ = 0, observed_slots_ = 0, observed_incidence_ = 0, incidence_ = 0;
    Slot& slot(SlotId id) { return (*slots_[id.value / 4])[id.value % 4]; }
    [[noreturn]] static void counter_limit()
    {
        throw ::SimpleFluid::PLIC::GeometryError(::SimpleFluid::PLIC::GeometryErrorCode::ResourceLimit,
            "Ordinary vertex evidence/construction counter exhausted.");
    }
    void admit_version(bool extension) const
    {
        const auto maximum = std::numeric_limits<std::size_t>::max();
        if (next_evidence_version_ == maximum || work_.stats.vertex_record_constructions == maximum ||
            (extension && work_.stats.vertex_evidence_extensions == maximum))
            counter_limit();
    }
    struct IncidenceScratch
    {
        Work& work;
        std::size_t bytes;
        IncidenceScratch(Work& w, std::size_t n) : work(w), bytes(n)
        {
            work.live_incidence_bytes += bytes;
            work.observe_geometry();
        }
        ~IncidenceScratch() { release(); }
        void release() noexcept
        {
            work.live_incidence_bytes -= bytes;
            bytes = 0;
            work.observe_geometry();
        }
    };
    template<class T> void reserve_directory(std::vector<T>& directory, std::size_t required)
    {
        if (required <= directory.capacity())
            return;
        const auto capacity = std::max(required, 2 * directory.capacity());
        work_.charge(capacity * sizeof(T));
        // Include both old and replacement directory allocations at growth.
        work_.observe_geometry(capacity * sizeof(T));
        directory.reserve(capacity);
        observe();
        work_.charge((directory.capacity() - capacity) * sizeof(T));
    }
    Handle make(const Weights& weights, const Weights& errors, std::span<const Support* const> supports,
        const void* source, const void* chart, std::size_t construction, bool extension)
    {
        admit_version(extension);
        const auto new_page = live_ == record_capacity();
        const auto recycled_record = !new_page && !virgin_trailing_record_;
        // Fresh page reservations already admit both physical positions. A
        // released position in a retained page admits its next construction
        // separately, regardless of which stable metadata slot will own it.
        work_.charge((recycled_record ? sizeof(Record) : 0) + supports.size() * sizeof(const Support*));
        std::vector<const Support*> incidence;
        incidence.reserve(supports.size());
        work_.charge((incidence.capacity() - supports.size()) * sizeof(const Support*));
        incidence.insert(incidence.end(), supports.begin(), supports.end());
        IncidenceScratch scratch(work_, incidence.capacity() * sizeof(const Support*));
        if (free_ == absent && issued_ == slot_capacity())
        {
            reserve_directory(slots_, slots_.size() + 1);
            work_.charge(sizeof(SlotBlock));
            slots_.push_back(std::make_unique<SlotBlock>());
            observe();
        }
        if (new_page)
        {
            reserve_directory(pages_, pages_.size() + 1);
            work_.charge(sizeof(Page));
            pages_.push_back(std::make_unique<Page>());
        }
        const SlotId id{free_ == absent ? issued_++ : free_};
        auto& metadata = slot(id);
        if (free_ != absent)
            free_ = metadata.next_free;
        auto& destination = (*pages_[live_ / 2])[live_ % 2];
        destination.emplace(
            weights, errors, std::move(incidence), source, chart, construction, next_evidence_version_, id);
        ++next_evidence_version_;
        metadata = {&*destination, 1, absent};
        scratch.release();
        incidence_ += destination->supports.capacity() * sizeof(const Support*);
        ++live_;
        // No admission or allocation can fail after the record is committed.
        virgin_trailing_record_ = new_page;
        ++work_.stats.vertex_record_constructions;
        if (extension)
            ++work_.stats.vertex_evidence_extensions;
        observe();
        return Handle(*this, id);
    }
    void release(SlotId id) noexcept
    {
        auto& metadata = slot(id);
        if (--metadata.references)
            return;
        auto* hole = metadata.record;
        auto& last = (*pages_[(live_ - 1) / 2])[(live_ - 1) % 2];
        incidence_ -= hole->supports.capacity() * sizeof(const Support*);
        if (hole != &*last)
        {
            *hole = std::move(*last);
            slot(hole->owning_slot).record = hole;
            work_.stats.vertex_relocation_bytes += sizeof(Record);
        }
        last.reset();
        metadata = {nullptr, 0, free_};
        free_ = id.value;
        --live_;
        // The last occupied position was either recycled in the retained page
        // or its page is being freed, discarding any unused credit without refund.
        virgin_trailing_record_ = false;
        if (pages_.size() > (live_ + 1) / 2)
            pages_.pop_back();
        observe();
    }
    void observe() noexcept
    {
        const auto arena = sizeof(*this) + pages_.capacity() * sizeof(typename decltype(pages_)::value_type) +
                           pages_.size() * sizeof(Page);
        const auto slots =
            slots_.capacity() * sizeof(typename decltype(slots_)::value_type) + slots_.size() * sizeof(SlotBlock);
        work_.live_arena_bytes = work_.live_arena_bytes - observed_arena_ + arena;
        work_.live_slot_bytes = work_.live_slot_bytes - observed_slots_ + slots;
        work_.live_incidence_bytes = work_.live_incidence_bytes - observed_incidence_ + incidence_;
        observed_arena_ = arena;
        observed_slots_ = slots;
        observed_incidence_ = incidence_;
        work_.stats.live_vertex_records = live_;
        work_.stats.live_vertex_record_capacity = record_capacity();
        work_.stats.live_vertex_slot_capacity = slot_capacity();
        work_.stats.peak_live_vertex_records = std::max(work_.stats.peak_live_vertex_records, live_);
        work_.stats.peak_vertex_record_capacity = std::max(work_.stats.peak_vertex_record_capacity, record_capacity());
        work_.stats.peak_vertex_slot_capacity = std::max(work_.stats.peak_vertex_slot_capacity, slot_capacity());
        work_.observe_geometry();
    }
};
} // namespace SimpleFluid::geometry::swept_volume::details
