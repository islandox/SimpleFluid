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
};
struct OwnerRecord
{
    OwnerKey key;
    std::shared_ptr<const void> owner;
    std::size_t bytes = 0, references = 0;
    [[nodiscard]] std::size_t exclusive_payload_bytes() const noexcept { return bytes; }
};
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
    struct RecordSink
    {
        void* context;
        void (*visit)(void*, const OwnerRecord&);
        void operator()(const OwnerRecord& value) const { visit(context, value); }
    };
    void visit_dependencies(MetadataSink, RecordSink) const;
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
    [[nodiscard]] bool admits(
        std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes) const;
    [[nodiscard]] std::size_t references(
        std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>&) const;
    /** Empty sets have no channel. Mixed-channel sets cannot enter one
     * arithmetic ledger and are rejected by this query. */
    [[nodiscard]] std::optional<std::uint8_t> uniform_channel() const;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_admission_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_payload_bytes() const noexcept;
    [[nodiscard]] std::size_t owner_count() const noexcept;
    /** Explicit diagnostic export. Source/cache mutation uses typed deltas. */
    void visit(const std::function<void(const OwnerRecord&)>&) const;
    void visit_metadata(const std::function<void(const void*, std::size_t)>&) const;
    /** Exact separate C++ object and backing-buffer addresses for offline census. */
    void visit_payload(const std::function<void(const void*, std::size_t)>&) const;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace SimpleFluid::PLIC::details::geometry_demand
