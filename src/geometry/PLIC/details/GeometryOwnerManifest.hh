/** @file GeometryOwnerManifest.hh
 * @brief Persistent controlled geometry owner union with exact cached payload totals.
 */
#pragma once
#include "geometry/PLIC/details/GeometryDemandPages.hh"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace SimpleFluid::PLIC::details::geometry_demand
{
struct OwnerKey
{
    std::uint8_t channel = 0, kind = 0;
    const void* address = nullptr;
    bool operator==(const OwnerKey&) const = default;
};
struct OwnerKeyLess
{
    bool operator()(const OwnerKey& a, const OwnerKey& b) const noexcept
    {
        if (a.channel != b.channel)
            return a.channel < b.channel;
        if (a.kind != b.kind)
            return a.kind < b.kind;
        return std::less<const void*>{}(a.address, b.address);
    }
};
struct OwnerRecord
{
    OwnerKey key;
    std::shared_ptr<const void> owner;
    std::size_t bytes = 0, references = 0;
    [[nodiscard]] std::size_t exclusive_payload_bytes() const noexcept { return bytes; }
    [[nodiscard]] std::size_t logical_size() const noexcept { return references; }
};
/** Incoming typed edges are a sparse subset of the real owner references.
 * The matching physical row owns and authenticates the shared control. */
struct TypedOwnerReference
{
    OwnerKey key;
    std::size_t references = 0;
};
// OwnerRecord defaults only zero scalars and an empty shared_ptr; it performs
// no allocation and cannot throw. Its copies also preserve shared controls.
template<> inline constexpr bool stack_safe_default<OwnerRecord> = true;
template<> inline constexpr bool native_flag_in_owned_admission<OwnerRecord> = true;

/** Contains only immutable physical owner pins and shared index pages.
 * A caller constructing an accepted snapshot controls when the returned root
 * publishes. Failed edits preserve that caller's accepted root and spent prefix.
 * A manifest body is independent of its editor, counter and hierarchy lifetime.
 * Callers authenticate layout, geometry epoch and construction controls before
 * choosing geometry reuse. Owner identity here certifies allocation credit. */
class GeometryOwnerManifest
{
public:
    GeometryOwnerManifest() = default;
    [[nodiscard]] bool same_root(const GeometryOwnerManifest&) const noexcept;
    [[nodiscard]] bool authenticates(const std::weak_ptr<const void>&) const noexcept;
    [[nodiscard]] std::shared_ptr<const void> root_control() const noexcept;
    [[nodiscard]] std::size_t root_metadata_bytes() const noexcept;

    struct RecordSink
    {
        void* context;
        void (*visit)(void*, const OwnerRecord&);
        void operator()(const OwnerRecord& value) const { visit(context, value); }
    };
    void visit_dependencies(MetadataSink, RecordSink) const;
    [[nodiscard]] bool has_native_administration() const noexcept;
    void visit_native_administration_delta(
        const GeometryOwnerManifest&, NativeAdministrationSink, PageWork* = nullptr) const;
    void visit_records(RecordSink) const;
    /** Pruned unique-owner delta; exact shared B+ controls stop traversal. */
    void visit_added(const GeometryOwnerManifest&, RecordSink, PageWork* = nullptr) const;

    struct BorrowedRecordSink
    {
        void* context;
        void (*visit)(void*, const void* parent, const OwnerRecord&);
        void operator()(const OwnerRecord& value, const void* parent) const { visit(context, parent, value); }
    };
    void visit_borrowed_dependencies(BorrowedMetadataSink, BorrowedRecordSink, const void* parent = nullptr) const;
    /** Build one complete immutable owner set in a single persistent edit.
     * Rows must be in (channel, kind, address) order, carry their final
     * reference multiplicity and admit no alias with conflicting controls. */
    [[nodiscard]] static GeometryOwnerManifest built(std::span<const OwnerRecord>, Admission, PageWork* = nullptr);
    /** Apply sorted final records in one batch. Zero references retire a row. */
    [[nodiscard]] GeometryOwnerManifest edited(
        std::span<const OwnerRecord>, Admission, PageWork* = nullptr, PageWork* lookup_work = nullptr) const;
    [[nodiscard]] GeometryOwnerManifest retained(std::uint8_t channel, std::uint8_t kind, std::shared_ptr<const void>,
        std::size_t bytes, Admission, PageWork* = nullptr) const;
    /** Add `references` slot holds in one persistent edit. A repeated exact
     * owner keeps its single exclusive byte count; multiplicity stays separate
     * from unique physical payload. */
    [[nodiscard]] GeometryOwnerManifest retained(std::uint8_t channel, std::uint8_t kind, std::shared_ptr<const void>,
        std::size_t bytes, std::size_t references, Admission, PageWork* = nullptr) const;
    [[nodiscard]] GeometryOwnerManifest retired(std::uint8_t channel, std::uint8_t kind,
        const std::shared_ptr<const void>&, Admission, PageWork* = nullptr) const;
    /** Release `references` slot holds in one persistent edit. */
    [[nodiscard]] GeometryOwnerManifest retired(std::uint8_t channel, std::uint8_t kind,
        const std::shared_ptr<const void>&, std::size_t references, Admission, PageWork* = nullptr) const;
    [[nodiscard]] bool admits(std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>&,
        std::size_t bytes, PageWork* = nullptr) const;
    [[nodiscard]] std::size_t references(
        std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>&, PageWork* = nullptr) const;
    /** Empty sets have no channel. Mixed-channel sets cannot enter one
     * arithmetic ledger and are rejected by this query. */
    [[nodiscard]] std::optional<std::uint8_t> uniform_channel(PageWork* = nullptr) const;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_admission_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_payload_bytes() const noexcept;
    [[nodiscard]] std::size_t owner_count() const noexcept;
    [[nodiscard]] std::size_t typed_owner_count() const noexcept;
    /** Cached physical pointer-slot count, separate from unique owners. */
    [[nodiscard]] std::size_t physical_edge_count() const noexcept;
    /** Retained State/page count for any valid index of at most rows keys.
     * Includes a possible indexed singleton from individual retained/retired edits. */
    [[nodiscard]] static std::size_t metadata_owner_count_bound(std::size_t rows, std::size_t typed_rows = 0);
    [[nodiscard]] std::size_t metadata_owner_count_bound() const;
    /** Ratio bounding all height-linear batch costs after possible growth.
     * An engaged typed growth bound includes the sparse typed path. */
    [[nodiscard]] std::size_t edit_height_scale_bound(std::size_t anticipated_additions,
        std::optional<std::size_t> anticipated_typed_additions = {}) const;
    /** Conservative allocation envelope for one sorted batch of changed keys. */
    [[nodiscard]] std::size_t batch_edit_byte_bound(std::size_t changed_keys, std::size_t typed_keys = 0) const;
    /** New index allocations: changed search/rebalance paths, root State and
     * the possible inline-to-index promotion page. */
    [[nodiscard]] std::size_t batch_edit_metadata_owner_bound(
        std::size_t changed_keys, std::size_t typed_keys = 0) const;

    /** Explicit diagnostic export. Source/cache mutation uses typed deltas. */
    void visit(const std::function<void(const OwnerRecord&)>&) const;
    void visit_metadata(const std::function<void(const void*, std::size_t)>&) const;
    /** Exact separate C++ object and backing-buffer addresses for offline census. */
    void visit_payload(const std::function<void(const void*, std::size_t)>&) const;

private:
    friend class SimpleFluid::PLIC::details::GeometryOwnerGraphEditor;
    [[nodiscard]] std::size_t typed_references(const OwnerKey&, PageWork* = nullptr) const;
    [[nodiscard]] GeometryOwnerManifest edited_typed(std::span<const OwnerRecord>, std::span<const TypedOwnerReference>,
        Admission, PageWork*, PageWork*) const;
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace SimpleFluid::PLIC::details::geometry_demand
