/** Exact persistent control allowances for physically retained index owners. */
#pragma once
#include "geometry/PLIC/details/GeometryDemandPages.hh"
#include <functional>
#include <memory>
#include <span>
namespace SimpleFluid::PLIC::details
{
struct IndexControlRecord
{
    std::shared_ptr<const void> owner;
    std::size_t payload = 0, control = 0;
    // A native admin owner remains classified when its exact physical control
    // is already included in payload and its legacy allowance is zero.
    bool native = false;
    [[nodiscard]] std::size_t exclusive_admission_bytes() const noexcept { return control; }
};
namespace geometry_demand
{
template<> inline constexpr bool stack_safe_default<IndexControlRecord> = true;
template<> inline constexpr bool native_flag_in_owned_admission<IndexControlRecord> = true;
}
class GeometryIndexControls
{
public:
    [[nodiscard]] std::size_t control_bytes() const noexcept;
    [[nodiscard]] std::size_t index_admission_bytes() const noexcept;
    /** Construction admission and newly allocated metadata owners for one
     * batch, allowing earlier unpublished row additions before this batch.
     * Both queries walk one tree path and never enumerate credits. */
    [[nodiscard]] std::size_t edit_admission_bound(std::size_t changes, std::size_t anticipated_additions = 0) const;
    [[nodiscard]] std::size_t edit_metadata_owner_bound(
        std::size_t changes, std::size_t anticipated_additions = 0) const;
    [[nodiscard]] bool holds(const std::shared_ptr<const void>&, std::size_t payload) const;
    [[nodiscard]] bool contains(const std::shared_ptr<const void>&, std::size_t payload, std::size_t control) const;
    [[nodiscard]] GeometryIndexControls edited(
        std::span<const IndexControlRecord>, geometry_demand::Admission, geometry_demand::PageWork* = nullptr) const;
    struct RecordSink
    {
        void* context;
        void (*visit)(void*, const IndexControlRecord&);
    };
    struct BorrowedRecordSink
    {
        void* context;
        void (*visit)(void*, const void* parent, const IndexControlRecord&);
    };
    void visit_records(RecordSink) const;
    void visit(const std::function<void(const IndexControlRecord&)>&) const;
    void visit_metadata(geometry_demand::MetadataSink) const;
    [[nodiscard]] bool has_native_administration() const noexcept;
    void visit_native_administration_delta(const GeometryIndexControls&, geometry_demand::NativeAdministrationSink,
        geometry_demand::PageWork* = nullptr) const;
    /** Explicit diagnostic physical parent edges and allocation buffers. */
    void visit_borrowed_dependencies(geometry_demand::BorrowedMetadataSink, BorrowedRecordSink) const;
    void visit_payload(const std::function<void(const void*, std::size_t)>&) const;
    [[nodiscard]] std::shared_ptr<const void> root_control() const noexcept;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace SimpleFluid::PLIC::details
