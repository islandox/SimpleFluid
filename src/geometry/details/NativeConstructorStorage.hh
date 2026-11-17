/** @file NativeConstructorStorage.hh
 * @brief Pre-allocation receipts for concrete public vectors and private node containers.
 */
#pragma once
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/details/NativeAllocationEnvelope.hh"
#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace SimpleFluid::geometry_detail
{
/** Restrict concrete STL buffers to the inspected libstdc++ reserve/string
 * profile. Other profiles fail before allocating a destination buffer. */
inline constexpr bool native_constructor_capacity_profile_supported() noexcept
{
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE == 16 && defined(__GLIBCXX__) && __GLIBCXX__ == 20260810 && \
    !defined(_GLIBCXX_DEBUG) && !defined(_GLIBCXX_DEBUG_PEDANTIC) && defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI
    return true;
#else
    return false;
#endif
}
class NativeConstructorStorage;
template<class T> class NativeStorageAllocator;
class NativeConstructorStorage final
{
    struct Node
    {
        Node* previous = nullptr;
        Node* next = nullptr;
        NativeAllocationExtent allocation;
    };
    template<class T> struct alignas(std::max(alignof(T), alignof(Node))) Unit
    { std::byte bytes[std::max(alignof(T), alignof(Node))]; };
    struct Buffer
    {
        Buffer* next = nullptr;
        NativeAllocationExtent allocation;
        bool string = false;
    };
public:
    explicit NativeConstructorStorage(NativeConstructionAdmission admission) : admission_(std::move(admission))
    {
        require_profile();
    }
    NativeConstructorStorage(const NativeConstructorStorage&) = delete;
    NativeConstructorStorage& operator=(const NativeConstructorStorage&) = delete;
    ~NativeConstructorStorage()
    {
        if (nodes_)
            std::terminate(); // Every node container must die before its detached storage domain.
        while (buffers_)
        {
            auto* receipt = buffers_;
            buffers_ = receipt->next;
            admission_.release(receipt->allocation.bytes);
            admission_.allocator<Buffer>().deallocate(receipt, 1);
        }
    }
    static void require_profile()
    {
        if (!native_constructor_capacity_profile_supported())
            throw std::invalid_argument("Unsupported native constructor capacity profile.");
    }
    const NativeConstructionAdmission& admission() const noexcept { return admission_; }
    /** Empty concrete vectors are reserved exactly once. Buffer ownership is
     * moved only into an object whose storage field is destroyed after it. */
    template<class T> void reserve(std::vector<T>& destination, std::size_t count)
    {
        require_profile();
        if (!destination.empty() || destination.capacity())
            throw std::invalid_argument("Native concrete buffer must be empty before admission.");
        if (!count)
            return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        auto receipt = admission_.unique<Buffer>();
        const auto bytes = count * sizeof(T);
        admission_.charge(bytes);
        try { destination.reserve(count); }
        catch (...) { admission_.release(bytes); throw; }
        if (destination.capacity() != count)
        {
            // The inspected profile promised exactly count. Release the actual
            // buffer before its charge, then reject rather than publish drift.
            std::vector<T>().swap(destination);
            admission_.release(bytes);
            throw std::invalid_argument("Native concrete vector reserve profile drift.");
        }
        receipt->allocation = {destination.data(), bytes};
        receipt->next = buffers_;
        buffers_ = receipt.release();
        exclusive_bytes_ += sizeof(Buffer) + bytes;
        changed();
    }
    /** Release a concrete phase buffer at its actual capacity lifetime. */
    template<class T> void reset_buffer(std::vector<T>& buffer)
    {
        if (!buffer.capacity())
            return;
        auto** link = &buffers_;
        while (*link && ((*link)->allocation.address != buffer.data() || (*link)->string))
            link = &(*link)->next;
        if (!*link || (*link)->allocation.bytes != buffer.capacity() * sizeof(T))
            throw std::invalid_argument("Native phase buffer has no exact allocation receipt.");
        auto* receipt = *link;
        std::vector<T>().swap(buffer); // Destroy actual elements/storage before retirement.
        *link = receipt->next;
        exclusive_bytes_ -= sizeof(Buffer) + receipt->allocation.bytes;
        changed();
        admission_.release(receipt->allocation.bytes);
        admission_.allocator<Buffer>().deallocate(receipt, 1);
    }
    void reset_buffer(std::string& buffer)
    {
        const std::string local;
        if (buffer.capacity() <= local.capacity())
        {
            buffer.clear();
            return;
        }
        auto** link = &buffers_;
        while (*link && ((*link)->allocation.address != buffer.data() || !(*link)->string))
            link = &(*link)->next;
        if (!*link || (*link)->allocation.bytes != buffer.capacity() + 1)
            throw std::invalid_argument("Native phase string has no exact allocation receipt.");
        auto* receipt = *link;
        std::string().swap(buffer);
        *link = receipt->next;
        exclusive_bytes_ -= sizeof(Buffer) + receipt->allocation.bytes;
        changed();
        admission_.release(receipt->allocation.bytes);
        admission_.allocator<Buffer>().deallocate(receipt, 1);
    }
    void retire_freed_buffer(const void* address, std::size_t bytes) noexcept
    {
        if (!bytes) { if (address) std::terminate(); return; }
        auto** link = &buffers_;
        while (*link && ((*link)->allocation.address != address || (*link)->string)) link = &(*link)->next;
        if (!*link || (*link)->allocation.bytes != bytes) std::terminate();
        auto* receipt = *link;
        *link = receipt->next;
        exclusive_bytes_ -= sizeof(Buffer) + bytes;
        changed();
        admission_.release(bytes);
        admission_.allocator<Buffer>().deallocate(receipt, 1);
    }
    void retire_all_freed_buffers() noexcept
    {
        if (nodes_)
            std::terminate();
        while (buffers_)
        {
            auto* receipt = buffers_;
            buffers_ = receipt->next;
            exclusive_bytes_ -= sizeof(Buffer) + receipt->allocation.bytes;
            changed();
            admission_.release(receipt->allocation.bytes);
            admission_.allocator<Buffer>().deallocate(receipt, 1);
        }
        if (exclusive_bytes_)
            std::terminate();
    }
    /** Default-deleter native payloads keep their public unique_ptr type.
     * Their scalar allocation is precharged and pinned by this storage receipt;
     * the containing backing destroys the payload before its storage field. */
    template<class T, class... Args> std::unique_ptr<T> object(Args&&... args)
    {
        auto receipt = admission_.unique<Buffer>();
        admission_.charge(sizeof(T));
        std::unique_ptr<T> result;
        try { result = std::make_unique<T>(std::forward<Args>(args)...); }
        catch (...) { admission_.release(sizeof(T)); throw; }
        receipt->allocation = {result.get(), sizeof(T)};
        receipt->next = buffers_;
        buffers_ = receipt.release();
        exclusive_bytes_ += sizeof(Buffer) + sizeof(T);
        changed();
        return result;
    }
    template<class T> void copy(std::vector<T>& destination, std::span<const T> source)
    {
        reserve(destination, source.size());
        destination.insert(destination.end(), source.begin(), source.end());
    }
    /** Basic-string construction requires an empty SSO destination and uses
     * local capacity or one exact length+terminator allocation. Retained heap
     * storage must first be released through reset_buffer(), while its exact
     * receipt is still live. An empty string alone does not imply fresh storage. */
    void string(std::string& destination, std::string_view source)
    {
        require_profile();
        // The default string is the profile's allocation-free SSO object.
        const std::string local;
        if (!destination.empty() || destination.capacity() != local.capacity())
            throw std::invalid_argument("Native string destination must be empty and use local storage.");
        if (source.size() <= local.capacity())
        {
            destination.assign(source);
            return;
        }
        if (source.size() == std::numeric_limits<std::size_t>::max())
            throw std::bad_alloc();
        auto receipt = admission_.unique<Buffer>();
        const auto bytes = source.size() + 1;
        admission_.charge(bytes);
        try
        {
            // The range constructor calls _M_create(length, 0); assign/reserve
            // from SSO can instead grow to twice the local capacity.
            std::string exact(source.data(), source.size());
            destination.swap(exact);
        }
        catch (...) { admission_.release(bytes); throw; }
        if (destination.capacity() != source.size())
        {
            std::string().swap(destination);
            admission_.release(bytes);
            throw std::invalid_argument("Native concrete string capacity profile drift.");
        }
        receipt->allocation = {destination.data(), bytes};
        receipt->string = true;
        receipt->next = buffers_;
        buffers_ = receipt.release();
        exclusive_bytes_ += sizeof(Buffer) + bytes;
        changed();
    }
    template<class Visit> void visit_allocations(const Visit& visit) const
    {
        for (auto* node = nodes_; node; node = node->next)
            visit(node->allocation);
        for (auto* buffer = buffers_; buffer; buffer = buffer->next)
        {
            visit(NativeAllocationExtent{buffer, sizeof(Buffer)});
            visit(buffer->allocation);
        }
    }
    std::size_t exclusive_bytes() const noexcept { return exclusive_bytes_; }
    std::size_t revision() const noexcept { return revision_; }
    template<class T> T* allocate(std::size_t count)
    {
        // Intrusive allocation headers make nodes and hash buckets observable
        // without an unbudgeted side table or quadratic deallocation lookup.
        constexpr auto alignment = std::max(alignof(T), alignof(Node));
        constexpr auto offset = (sizeof(Node) + alignment - 1) / alignment * alignment;
        if (count > (std::numeric_limits<std::size_t>::max() - offset) / sizeof(T))
            throw std::bad_alloc();
        const auto bytes = offset + count * sizeof(T);
        const auto units = bytes / sizeof(Unit<T>) + bool(bytes % sizeof(Unit<T>));
        auto* allocation = admission_.allocator<Unit<T>>().allocate(units);
        auto* node = ::new (static_cast<void*>(allocation)) Node;
        node->allocation = {allocation, units * sizeof(Unit<T>)};
        node->next = nodes_;
        if (nodes_) nodes_->previous = node;
        nodes_ = node;
        exclusive_bytes_ += node->allocation.bytes;
        changed();
        return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(allocation) + offset);
    }
    template<class T> void deallocate(T* address, std::size_t count) noexcept
    {
        constexpr auto alignment = std::max(alignof(T), alignof(Node));
        constexpr auto offset = (sizeof(Node) + alignment - 1) / alignment * alignment;
        auto* node = reinterpret_cast<Node*>(reinterpret_cast<std::byte*>(address) - offset);
        if (node->previous) node->previous->next = node->next;
        else nodes_ = node->next;
        if (node->next) node->next->previous = node->previous;
        const auto units = node->allocation.bytes / sizeof(Unit<T>);
        exclusive_bytes_ -= node->allocation.bytes;
        changed();
        (void) count;
        admission_.allocator<Unit<T>>().deallocate(reinterpret_cast<Unit<T>*>(node), units);
    }
private:
    NativeConstructionAdmission admission_;
    Node* nodes_ = nullptr;
    Buffer* buffers_ = nullptr;
    size_t exclusive_bytes_ = 0, revision_ = 0;
    void changed() noexcept
    {
        // Saturation invalidates every prior receipt and forbids future
        // certification; retirement remains allocation-free and noexcept.
        if (revision_ != std::numeric_limits<std::size_t>::max())
            ++revision_;
    }
};
/** The default path explicitly uses std::allocator. Qualified factories always
 * supply an owned storage domain; rebinding preserves it for nodes/buckets. */
template<class T> class NativeStorageAllocator
{
    template<class> friend class NativeStorageAllocator;
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    NativeStorageAllocator() noexcept = default;
    explicit NativeStorageAllocator(std::shared_ptr<NativeConstructorStorage> storage) : storage_(std::move(storage)) {}
    template<class U> NativeStorageAllocator(const NativeStorageAllocator<U>& other) noexcept : storage_(other.storage_) {}
    T* allocate(std::size_t count)
    { return storage_ ? storage_->template allocate<T>(count) : std::allocator<T>().allocate(count); }
    void deallocate(T* address, std::size_t count) noexcept
    {
        if (storage_) storage_->template deallocate<T>(address, count);
        else std::allocator<T>().deallocate(address, count);
    }
    template<class U> bool operator==(const NativeStorageAllocator<U>& other) const noexcept
    { return storage_ == other.storage_; }
private:
    std::shared_ptr<NativeConstructorStorage> storage_;
};
struct NativeConstructorStorageOwner
{
    std::shared_ptr<NativeConstructorStorage> owner;
    std::shared_ptr<const NativeAllocationEnvelope> envelope;
};
inline NativeConstructorStorageOwner native_constructor_storage(const NativeConstructionAdmission& admission)
{
    NativeConstructorStorage::require_profile();
    auto tracked = NativeAllocationEnvelopeAccess::make<NativeConstructorStorage>(admission, admission);
    return {std::move(tracked.owner), std::move(tracked.envelope)};
}
} // namespace SimpleFluid::geometry_detail
