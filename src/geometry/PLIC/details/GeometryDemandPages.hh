/** @file GeometryDemandPages.hh
 * @brief Private persistent geometry selectors with cumulative allocation admission.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/NativePaidExtent.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace SimpleFluid::PLIC::details::geometry_demand
{
/** Qualified private page capacities and conservative shared-control allowance. */
inline constexpr std::size_t control_bytes = 128, leaf_capacity = 64, branch_capacity = 32;
/** Defaulting bounded scratch must neither allocate nor execute user code.
 * Nontrivial component rows may explicitly attest their empty initialization. */
/** Private opt-in: every exclusive control total for this row must be even.
 * Arbitrary/custom rows keep their unmodified numeric admission field. */
template<class Row> inline constexpr bool native_flag_in_owned_admission = false;
template<class Row> inline constexpr bool stack_safe_default = std::is_trivially_default_constructible_v<Row>;

/** The edit callbacks are borrowed. Native pages may retain detached scalar State; no page retains its caller. */
struct Admission
{
    void* context;
    void (*retain_bytes)(void*, std::size_t);
    void (*owner_created)(void*, const std::shared_ptr<const void>&, std::size_t, std::size_t) = nullptr;
    std::shared_ptr<const geometry_detail::NativeAllocationState> native;
    // Called only after this Admission actually allocated a native metadata
    // object and all its buffers; the callback issues the paid owner receipt.
    void (*native_owner_created)(void*, const std::shared_ptr<const void>&, std::size_t) = nullptr;
    template<class T> [[nodiscard]] geometry_detail::NativeAdministrationAllocator<T> allocator() const noexcept
    {
        return geometry_detail::NativeAdministrationAllocator<T>(native);
    }
    template<class T, class... Args> [[nodiscard]] std::shared_ptr<T> shared(Args&&...) const;
    void created(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted) const
    {
        if (native && native_owner_created)
            native_owner_created(context, owner, payload);
        if (owner_created)
            owner_created(context, owner, payload, admitted);
    }
    void charge(std::size_t count, std::size_t width) const;
};
/** Sort compact edit references. Generated charges live in fixed blocks;
 * existing receipt pages can supply their own charge storage. */
class ReceiptEdits
{
public:
    explicit ReceiptEdits(Admission);
    ReceiptEdits(const ReceiptEdits&) = delete;
    ReceiptEdits& operator=(const ReceiptEdits&) = delete;
    ~ReceiptEdits();

    void append(const NativePaidExtentCharge&, NativeOwnerReceipts::Action);
    // The caller keeps the receipt pages alive until the batch is consumed.
    void append_borrowed(NativePaidExtentRef, NativeOwnerReceipts::Action);
    void append_borrowed(const NativePaidExtentCharge& charge, NativeOwnerReceipts::Action action)
    { append_borrowed(&charge, action); }
    [[nodiscard]] std::span<NativeOwnerReceipts::Edit> rows() noexcept;
    void retain_prefix(std::size_t) noexcept;
    void clear() noexcept;

private:
    template<class Value> struct Charges
    {
        struct Block
        {
            std::array<Value, 8> values;
            Block* previous;
        };
        std::array<Value, 8> local{};
        Block* tail = nullptr;
        std::size_t size = 0;
        const Value& append(const Value&, const Admission&);
        void clear(const Admission&) noexcept;
    };

    Admission admission_;
    std::array<NativeOwnerReceipts::Edit, 8> inline_{};
    std::size_t size_ = 0;
    geometry_detail::NativeAdministrationVector<NativeOwnerReceipts::Edit> overflow_;
    Charges<NativeReceiptFact> facts_;
    Charges<NativeReceiptWithWitness> witnesses_;
    Charges<NativePaidExtentCharge> proofs_;
};
/** Borrowed first/last physical-owner visitor. Returning false proves this
 * exact owner's child edges are already held by the destination allowance.
 * Payload and admission sizes remain distinct for physical census. */
struct MetadataSink
{
    void* context;
    bool (*visit)(void*, const std::shared_ptr<const void>&, std::size_t payload, std::size_t admitted);
    // Detached allocation-State owners are physical body dependencies, not
    // index-role credits. Role-aware visitors supply this separate callback.
    bool (*native_visit)(void*, const std::shared_ptr<const void>&, std::size_t bytes) = nullptr;
    void (*paid_extent)(void*, const NativePaidExtentCharge&) = nullptr;
    void (*financial_pin)(void*, const NativePaidExtentCharge&) = nullptr;
    GeometryOwnerGraphEditor* financial_graph = nullptr;
    bool financial_retain = true;
    [[nodiscard]] MetadataSink with_financial(GeometryOwnerGraphEditor& graph, bool retain) const noexcept
    { auto result = *this; result.financial_graph = &graph; result.financial_retain = retain; return result; }
    void financial(const NativePaidExtentCharge&) const;
    bool project_native_state(const std::shared_ptr<const geometry_detail::NativeAllocationState>&) const;
    void paid(const NativePaidExtentCharge& charge) const
    {
        if (paid_extent && charge.valid())
            paid_extent(context, charge);
    }
    bool native_owner(const std::shared_ptr<const void>& owner, std::size_t bytes) const
    {
        return native_visit ? native_visit(context, owner, bytes) : visit(context, owner, bytes, bytes);
    }
    bool operator()(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted) const
    {
        if (paid_extent)
            paid(NativePaidExtentCharge::metadata(owner, payload));
        return visit(context, owner, payload, admitted);
    }
};
/** Diagnostic borrows preserve the physical metadata parent edge. */
struct BorrowedMetadataSink
{
    void* context;
    bool (*visit)(void*, const void* owner, const void* parent, std::size_t payload);
    bool operator()(const void* owner, const void* parent, std::size_t payload) const
    {
        return visit(context, owner, parent, payload);
    }
};
/** Actual allocator-State edges of changed native index owners only. */
struct NativeAdministrationSink
{
    void* context;
    void (*visit)(void*, const std::shared_ptr<const geometry_detail::NativeAllocationState>&, std::size_t references,
        bool retain);
    // Actual changed metadata owner, once per physical root edge. Its receipt
    // includes its real shared control and concrete backing allocations.
    void (*metadata_owner)(void*, const std::shared_ptr<const void>&, bool retain) = nullptr;
    void (*financial_pin)(void*, NativePaidExtentRef, bool retain) = nullptr;
    void financial(NativePaidExtentRef value, bool retain) const
    { if (value.has_lifetime_children() && financial_pin) financial_pin(context, value, retain); }
    void owner(const std::shared_ptr<const void>& value, bool retain) const
    {
        if (metadata_owner && value)
            metadata_owner(context, value, retain);
    }
    void operator()(const std::shared_ptr<const geometry_detail::NativeAllocationState>& state, std::size_t references,
        bool retain) const
    {
        if (state)
            visit(context, state, references, retain);
    }
};
struct PageWork
{
    std::size_t copied_pages = 0, copied_bytes = 0, copied_payload_bytes = 0;
    std::size_t visited_pages = 0, retired_page_visits = 0;
    std::size_t visited_source_rows = 0, copied_source_rows = 0;
};
struct NativeMetadataDelete
{
    std::shared_ptr<const geometry_detail::NativeAllocationState> state;
    void* object;
    std::size_t control_bytes = 0;
    bool constructed = false;
    std::size_t (*extent_bytes)(const void*, std::size_t) noexcept;
    void (*destroy)(void*, const std::shared_ptr<const geometry_detail::NativeAllocationState>&, bool) noexcept;
    void operator()(void* value) const noexcept { destroy(value, state, constructed); }
};
inline std::shared_ptr<const geometry_detail::NativeAllocationState> metadata_state(
    const std::shared_ptr<const void>& owner)
{
    const auto* proof = std::get_deleter<NativeMetadataDelete>(owner);
    return proof ? proof->state : nullptr;
}
inline std::size_t metadata_control_bytes(const std::shared_ptr<const void>& owner) noexcept
{
    const auto* proof = std::get_deleter<NativeMetadataDelete>(owner);
    return proof ? proof->control_bytes : control_bytes;
}
/** Exact construction scratch for a bounded row count. Bounded demand pages
 * work in groups of at most 64 rows; using that room on the stack removes a
 * transient heap allocation without changing what is retained. Larger inputs
 * fall back to an exactly sized heap buffer. */
template<class T, std::size_t Inline> class ScratchRows
{
public:
    ScratchRows(std::size_t count, std::size_t& used, std::size_t limit,
        std::shared_ptr<const geometry_detail::NativeAllocationState> native = {})
        : count_(count), native_(std::move(native)), heap_(geometry_detail::NativeAdministrationAllocator<T>(native_))
    {
        if (count > Inline)
        {
            if (used > limit || count > (limit - used) / sizeof(T))
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry scratch row budget is exhausted.");
            used += count * sizeof(T);
            heap_.resize(count);
            if (heap_.capacity() != count)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry scratch row capacity is unqualified.");
        }
    }
    T& operator[](std::size_t index) noexcept { return data()[index]; }
    const T& operator[](std::size_t index) const noexcept { return data()[index]; }
    [[nodiscard]] T* data() noexcept { return count_ > Inline ? heap_.data() : inline_.data(); }
    [[nodiscard]] const T* data() const noexcept { return count_ > Inline ? heap_.data() : inline_.data(); }
    [[nodiscard]] std::span<T> span() noexcept { return {data(), count_}; }
    [[nodiscard]] operator std::span<const T>() const noexcept { return {data(), count_}; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    std::size_t count_ = 0;
    std::shared_ptr<const geometry_detail::NativeAllocationState> native_;
    std::array<T, Inline> inline_{};
    geometry_detail::NativeAdministrationVector<T> heap_;
};
/** Row copies must allocate nothing and be noexcept. Variable geometry, key and
 * certificate data belong to immutable source owners referenced by rows.
 * Include GeometryDemandPages.tcc only from a compiled implementation provider
 * or a private component test that owns its template instantiation. */
struct NoLeafWitnesses {};
template<class Row, class Less, bool CompactBranches = false, class Witnesses = NoLeafWitnesses> class OrderedPages;
} // namespace SimpleFluid::PLIC::details::geometry_demand
