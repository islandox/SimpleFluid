/** @file NativeAllocationEnvelope.hh
 * @brief Exact fixed allocation ownership for weak native observations.
 */
#pragma once
#include "geometry/details/NativeConstructionAdmission.hh"
#include <array>
#include <cstdint>

namespace SimpleFluid::geometry_detail
{
class NativeAllocationEnvelopeAccess;
namespace native_envelope_detail
{
inline bool contains(NativeAllocationExtent extent, const void* object, std::size_t bytes) noexcept
{
    const auto begin = reinterpret_cast<std::uintptr_t>(extent.address);
    const auto address = reinterpret_cast<std::uintptr_t>(object);
    return extent.address && object && address >= begin && address - begin <= extent.bytes &&
           bytes <= extent.bytes - (address - begin);
}
enum class Block : unsigned
{
    RecordObject,
    RecordControl,
    Target,
    Envelope
};
/** No target/controller ownership here: only allocation extents and the
 * detached scalar admission domain shared by the allocation's allocators.
 */
struct SIMPLEFLUID_LOCAL Record
{
    std::array<NativeAllocationExtent, 4> blocks{};
    std::shared_ptr<const NativeAllocationState> state;
};

/** Record-control allocation captures into the heap Record. Its deallocator
 * never reads that address after Record destruction. There is no stack borrow
 * or source/controller capture in retained allocator state.
 */
template<class T> class SIMPLEFLUID_LOCAL RecordAllocator
{
    template<class> friend class RecordAllocator;

public:
    using value_type = T;
    RecordAllocator(NativeConstructionAllocator<T> allocator, Record* record) noexcept
        : allocator_(std::move(allocator)), record_(record)
    {
    }
    template<class U>
    RecordAllocator(const RecordAllocator<U>& other) noexcept : allocator_(other.allocator_), record_(other.record_)
    {
    }
    T* allocate(std::size_t count)
    {
        auto* address = allocator_.allocate(count);
        auto& extent = record_->blocks[static_cast<unsigned>(Block::RecordControl)];
        if (extent.address)
        {
            allocator_.deallocate(address, count);
            throw std::invalid_argument("Native record requires a single qualified shared-control allocation.");
        }
        extent = {address, count * sizeof(T)};
        return address;
    }
    void deallocate(T* address, std::size_t count) noexcept { allocator_.deallocate(address, count); }
    template<class U> bool operator==(const RecordAllocator<U>& other) const noexcept
    {
        return allocator_ == other.allocator_ && record_ == other.record_;
    }

private:
    NativeConstructionAllocator<T> allocator_;
    Record* record_;
};
template<class T> class SIMPLEFLUID_LOCAL BlockAllocator
{
    template<class> friend class BlockAllocator;

public:
    using value_type = T;
    BlockAllocator(NativeConstructionAllocator<T> allocator, std::shared_ptr<Record> record, Block block) noexcept
        : allocator_(std::move(allocator)), record_(std::move(record)), block_(block)
    {
    }
    template<class U>
    BlockAllocator(const BlockAllocator<U>& other) noexcept
        : allocator_(other.allocator_), record_(other.record_), block_(other.block_)
    {
    }
    T* allocate(std::size_t count)
    {
        auto* address = allocator_.allocate(count);
        auto& extent = record_->blocks[static_cast<unsigned>(block_)];
        if (extent.address)
        {
            allocator_.deallocate(address, count);
            throw std::invalid_argument("Native envelope requires a single qualified combined allocation.");
        }
        extent = {address, count * sizeof(T)};
        return address;
    }
    void deallocate(T* address, std::size_t count) noexcept { allocator_.deallocate(address, count); }
    template<class U> bool operator==(const BlockAllocator<U>& other) const noexcept
    {
        return allocator_ == other.allocator_ && record_ == other.record_ && block_ == other.block_;
    }

private:
    NativeConstructionAllocator<T> allocator_;
    std::shared_ptr<Record> record_;
    Block block_;
};
} // namespace native_envelope_detail

/** While this envelope is alive, its four measured fixed allocations are all
 * alive. It retains the actual target resource weakly. Object destruction can
 * release every dynamic child while the combined target/control allocation
 * remains observable. Strong-lifetime child projections are separate roots.
 *
 * Factories retain this same canonical envelope in the target's source owner
 * projection and in weak observers. Do not create an additional independent
 * envelope for an existing shared control or bill dynamic children here.
 */
class SIMPLEFLUID_LOCAL NativeAllocationEnvelope final
{
    class ConstructionKey
    {
        friend class NativeAllocationEnvelopeAccess;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
        ConstructionKey(ConstructionKey&&) = default;
    };

public:
    NativeAllocationEnvelope(ConstructionKey, std::weak_ptr<const void> target, const void* address,
        std::shared_ptr<const native_envelope_detail::Record> record)
        : target_(std::move(target)), address_(address), record_(std::move(record))
    {
    }
    NativeAllocationEnvelope(const NativeAllocationEnvelope&) = delete;
    NativeAllocationEnvelope& operator=(const NativeAllocationEnvelope&) = delete;
    NativeAllocationEnvelope(NativeAllocationEnvelope&&) = delete;
    NativeAllocationEnvelope& operator=(NativeAllocationEnvelope&&) = delete;
    [[nodiscard]] bool authentic_owner(const std::shared_ptr<const NativeAllocationEnvelope>& owner) const noexcept
    {
        return owner.get() == this && !self_.owner_before(owner) && !owner.owner_before(self_);
    }
    [[nodiscard]] const void* observed_target_address() const noexcept { return address_; }
    [[nodiscard]] bool object_alive() const noexcept { return !target_.expired(); }
    template<class T> [[nodiscard]] bool shares_control(const std::shared_ptr<T>& owner) const noexcept
    {
        return owner && !target_.owner_before(owner) && !owner.owner_before(target_);
    }
    template<class T> [[nodiscard]] bool matches(const std::shared_ptr<T>& owner) const noexcept
    {
        return owner.get() == address_ && shares_control(owner);
    }
    [[nodiscard]] std::size_t fixed_allocation_bytes() const
    {
        if (!native_envelope_detail::contains(
                record_->blocks[static_cast<unsigned>(native_envelope_detail::Block::Envelope)], this,
                sizeof(NativeAllocationEnvelope)))
            throw std::logic_error("Native allocation envelope is copied or foreign to its actual recorded block.");
        std::size_t bytes = 0;
        for (const auto& extent : record_->blocks)
        {
            if (!extent.address || extent.bytes > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::logic_error("Native fixed allocation envelope is incomplete or excessive.");
            bytes += extent.bytes;
        }
        return bytes;
    }
    template<class Visit> void visit_allocations(const Visit& visit) const
    {
        (void) fixed_allocation_bytes();
        for (const auto& extent : record_->blocks)
            visit(extent);
    }
    [[nodiscard]] const std::shared_ptr<const NativeAllocationState>& state() const noexcept { return record_->state; }

private:
    friend class NativeAllocationEnvelopeAccess;
    std::weak_ptr<const NativeAllocationEnvelope> self_;
    std::weak_ptr<const void> target_;
    const void* address_;
    std::shared_ptr<const native_envelope_detail::Record> record_;
};
template<class T> struct SIMPLEFLUID_LOCAL NativeTrackedOwner
{
    std::shared_ptr<T> owner;
    std::shared_ptr<const NativeAllocationEnvelope> envelope;
};

/** Allocation provenance only; this does not issue kernel or geometry proof.
 * All three shared controls and the record object use the same live admission.
 */
class SIMPLEFLUID_LOCAL NativeAllocationEnvelopeAccess
{
public:
    template<class T, class... Args>
    [[nodiscard]] static NativeTrackedOwner<T> make(const NativeConstructionAdmission& admission, Args&&... args)
    {
        static_assert(!std::is_array_v<T>);
        namespace detail = native_envelope_detail;
        auto memory = admission.unique<detail::Record>();
        memory->state = admission.state();
        memory->blocks[static_cast<unsigned>(detail::Block::RecordObject)] = {memory.get(), sizeof(detail::Record)};
        auto* raw = memory.release();
        auto deleter = std::move(memory.get_deleter());
        std::shared_ptr<detail::Record> record(raw, std::move(deleter),
            detail::RecordAllocator<detail::Record>(admission.allocator<detail::Record>(), raw));
        auto owner =
            std::allocate_shared<T>(detail::BlockAllocator<T>(admission.allocator<T>(), record, detail::Block::Target),
                std::forward<Args>(args)...);
        if (!detail::contains(record->blocks[static_cast<unsigned>(detail::Block::Target)], owner.get(), sizeof(T)))
            throw std::invalid_argument("Native target does not use its recorded qualified combined allocation.");
        auto envelope = std::allocate_shared<NativeAllocationEnvelope>(
            detail::BlockAllocator<NativeAllocationEnvelope>(
                admission.allocator<NativeAllocationEnvelope>(), record, detail::Block::Envelope),
            NativeAllocationEnvelope::ConstructionKey{}, std::shared_ptr<const void>(owner), owner.get(), record);
        envelope->self_ = envelope;
        if (!detail::contains(record->blocks[static_cast<unsigned>(detail::Block::Envelope)], envelope.get(),
                sizeof(NativeAllocationEnvelope)))
            throw std::invalid_argument("Native envelope does not use its recorded qualified combined allocation.");
        return {std::move(owner), std::move(envelope)};
    }
};
} // namespace SimpleFluid::geometry_detail
