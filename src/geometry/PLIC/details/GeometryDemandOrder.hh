/** @file GeometryDemandOrder.hh
 * @brief Compressed source-only physical selectors in persistent demand order.
 */
#pragma once
#include "geometry/PLIC/details/GeometryDemandPages.hh"
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace SimpleFluid::PLIC
{
class CartesianCellSource;
}
namespace SimpleFluid::PLIC::details::geometry_demand
{
struct OrderStamp
{
    std::uint64_t generation = 0, position = 0;
    bool operator==(const OrderStamp&) const = default;
};
enum class SourceKind : std::uint8_t
{
    Explicit,
    Cartesian
};
struct SourceSlot
{
    std::shared_ptr<const void> source;
    std::uint64_t row = 0;
    SourceKind kind = SourceKind::Explicit;
};
struct SourceSink
{
    void* context;
    void (*visit)(void*, SourceKind, const std::shared_ptr<const void>&);
    void operator()(SourceKind kind, const std::shared_ptr<const void>& owner) const { visit(context, kind, owner); }
};
struct BorrowedSourceSink
{
    void* context;
    void (*visit)(void*, const void* parent, SourceKind, const void* owner);
    void operator()(const void* parent, SourceKind kind, const void* owner) const
    {
        visit(context, parent, kind, owner);
    }
};
struct OrderedSourceSlot
{
    OrderStamp stamp;
    SourceSlot physical;
};
/** Owns source-only data pages and metadata. Consumers authenticate a source
 * page's exact geometry type, layout, epoch and controls before selecting it.
 * Returned selectors preserve their actual shared owner and original row.
 * Pages own no predecessor order object, catalog or hierarchy.
 */
class GeometryDemandOrder
{
public:
    GeometryDemandOrder() = default;
    /** The supplied prefix replaces exactly the old stamps listed in retired.
     * Remaining slots preserve their order. Admission is cumulative and failed
     * construction leaves this accepted value unchanged. */
    [[nodiscard]] GeometryDemandOrder prepended(
        std::span<const SourceSlot> prefix, std::span<const OrderStamp> retired, Admission, PageWork* = nullptr) const;
    /** Repack source pointer arrays without moving their logical cells. */
    [[nodiscard]] GeometryDemandOrder rebound(std::span<const OrderedSourceSlot>, Admission, PageWork* = nullptr) const;
    /** Same-stamp source replacements preserve tree height and row counts. */
    [[nodiscard]] std::size_t rebound_byte_bound(std::size_t changed_slots) const;
    [[nodiscard]] std::size_t rebound_owner_bound(std::size_t changed_slots) const;
    /** First/last dependency traversal skips complete inherited subtrees. */
    void visit_dependencies(MetadataSink, SourceSink) const;
    void visit_borrowed_dependencies(BorrowedMetadataSink, BorrowedSourceSink, const void* parent = nullptr) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] OrderedSourceSlot at(std::size_t) const;
    [[nodiscard]] const std::shared_ptr<const CartesianCellSource>& native_owner(std::size_t) const;
    [[nodiscard]] std::size_t rank(OrderStamp) const;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] std::size_t metadata_admission_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_payload_bytes() const noexcept;
    /** Explicit borrowed physical-owner census, independent of cached totals. */
    void visit_metadata(const std::function<void(const void*, std::size_t)>&) const;
    void visit_source_pages(const std::function<void(SourceKind, const std::shared_ptr<const void>&)>&) const;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace SimpleFluid::PLIC::details::geometry_demand
