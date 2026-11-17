/** @file GeometryDemandPages.hh
 * @brief Private persistent geometry selectors with cumulative allocation admission.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
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
template<class Row> inline constexpr bool stack_safe_default = std::is_trivially_default_constructible_v<Row>;

/** Borrowed only while editing. Immutable pages retain no counter or caller. */
struct Admission
{
    void* context;
    void (*retain_bytes)(void*, std::size_t);
    void (*owner_created)(void*, const std::shared_ptr<const void>&, std::size_t, std::size_t) = nullptr;
    void created(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted) const
    {
        if (owner_created)
            owner_created(context, owner, payload, admitted);
    }
    void charge(std::size_t count, std::size_t width) const;
};
/** Borrowed first/last physical-owner visitor. Returning false proves this
 * exact owner's child edges are already held by the destination allowance.
 * Payload and admission sizes remain distinct for physical census. */
struct MetadataSink
{
    void* context;
    bool (*visit)(void*, const std::shared_ptr<const void>&, std::size_t payload, std::size_t admitted);
    bool operator()(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted) const
    {
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
struct PageWork
{
    std::size_t copied_pages = 0, copied_bytes = 0, copied_payload_bytes = 0;
    std::size_t visited_pages = 0, retired_page_visits = 0;
    std::size_t visited_source_rows = 0, copied_source_rows = 0;
};
/** Exact construction scratch for a bounded row count. Bounded demand pages
 * work in groups of at most 64 rows; using that room on the stack removes a
 * transient heap allocation without changing what is retained. Larger inputs
 * fall back to an exactly sized heap buffer. */
template<class T, std::size_t Inline> class ScratchRows
{
public:
    ScratchRows(std::size_t count, std::size_t& used, std::size_t limit) : count_(count)
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
    std::array<T, Inline> inline_{};
    std::vector<T> heap_;
};
/** Row copies must allocate nothing and be noexcept. Variable geometry, key and
 * certificate data belong to immutable source owners referenced by rows.
 * Include GeometryDemandPages.tcc only from a compiled implementation provider
 * or a private component test that owns its template instantiation. */
template<class Row, class Less> class OrderedPages;
} // namespace SimpleFluid::PLIC::details::geometry_demand
