/** @file OrdinaryIntersection.hh
 * @brief Private operand-bound receipt for one original framed bound screen.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <cstddef>
#include <cstdint>
namespace SimpleFluid::PLIC::details
{
class OrdinaryIntersectionAccess;
/** Scope-local immutable operand facts. Native preparation can only use the
 * actual retained view; callers cannot attach an arbitrary materialized body. */
class SIMPLEFLUID_LOCAL OrdinaryIntersectionOperand
{
public:
    static OrdinaryIntersectionOperand existing(ConvexGeometryPtr<ArithmeticMode::Ordinary>);
    static OrdinaryIntersectionOperand native(CellGeometryView<ArithmeticMode::Ordinary>);
    OrdinaryIntersectionOperand() = default;
    [[nodiscard]] const Frame& frame() const noexcept { return frame_; }
    [[nodiscard]] const CoordinateBox& bounds() const noexcept { return bounds_; }
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry() const noexcept { return geometry_; }
    void prepare(CellGeometryOperationWorkspace&);
    [[nodiscard]] bool prepared_frame_matches() const noexcept { return exact_preparation_; }

private:
    friend class OrdinaryIntersectionAccess;
    friend class OrdinaryIntersectionReceipt;
    Frame frame_{};
    CoordinateBox bounds_{};
    CellGeometryView<ArithmeticMode::Ordinary> source_;
    ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry_;
    const void* source_identity_ = nullptr;
    std::size_t source_slot_ = 0;
    bool exact_preparation_ = true;
};
/** Borrowed receipt: both operand objects and the active Work guard must
 * outlive it. No source/geometry owners are pinned by this receipt. Every
 * decision checks the same Work, monotone counters, operand identities and
 * named numeric controls. Invalid access poisons the receipt. A true separated
 * decision or finish() consumes it; a false query permits one later finish.
 * It never authorizes publication or ownership. */
class SIMPLEFLUID_LOCAL OrdinaryIntersectionReceipt
{
public:
    OrdinaryIntersectionReceipt(const OrdinaryIntersectionReceipt&) = delete;
    OrdinaryIntersectionReceipt& operator=(const OrdinaryIntersectionReceipt&) = delete;
    [[nodiscard]] bool separated();
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Ordinary> finish();

private:
    friend class OrdinaryIntersectionAccess;
    OrdinaryIntersectionReceipt(const OrdinaryIntersectionOperand&, const OrdinaryIntersectionOperand&);
    const OrdinaryIntersectionOperand *first_, *second_;
    struct Snapshot
    {
        const void *geometry = nullptr, *source = nullptr;
        std::size_t source_slot = 0;
        Frame frame{};
        CoordinateBox bounds{};
    };
    Snapshot first_snapshot_, second_snapshot_;
    Work* work_;
    std::size_t operations_ = 0, pairs_ = 0, bytes_ = 0;
    int rounding_ = -1;
    std::uint64_t x87_control_ = 0, simd_control_ = 0;
    bool controls_supported_ = false, screened_ = false, separated_ = false, consumed_ = false;
};
class SIMPLEFLUID_LOCAL OrdinaryIntersectionAccess
{
public:
    static OrdinaryIntersectionReceipt screen(const OrdinaryIntersectionOperand&, const OrdinaryIntersectionOperand&);
    static ConvexGeometryPtr<ArithmeticMode::Ordinary> finish(OrdinaryIntersectionReceipt&);
    static void validate(OrdinaryIntersectionReceipt&);
};
} // namespace SimpleFluid::PLIC::details
