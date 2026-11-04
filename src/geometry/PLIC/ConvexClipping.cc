#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/ConvexClipping.tcc"
#include "geometry/PLIC/details/ordinary/ConvexClipping.tcc"

#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/PLIC/details/OrdinaryIntersection.hh"
#include <cfenv>
#include <cmath>
#include <utility>
namespace SimpleFluid::PLIC::details
{
namespace
{
constexpr auto ordinary_mode = ArithmeticMode::Ordinary;
struct IntersectionControls
{
    int rounding = -1;
    std::uint64_t x87 = 0, simd = 0;
    bool supported = false;
};
IntersectionControls intersection_controls()
{
    IntersectionControls result;
    result.rounding = std::fegetround();
#if defined(__GLIBC__) && defined(__x86_64__)
    std::fenv_t environment{};
    if (result.rounding != -1 && std::fegetenv(&environment) == 0)
    {
        result.x87 = environment.__control_word;
        result.simd = environment.__mxcsr & ~std::uint32_t(0x3f);
        result.supported = true;
    }
#endif
    return result;
}
bool same_number(long double a, long double b)
{
    return a == b && (a != 0 || std::signbit(a) == std::signbit(b));
}
bool same_frame_bounds(const Frame& first, const CoordinateBox& a, const Frame& second, const CoordinateBox& b)
{
    if (first.exponent != second.exponent)
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!same_number(first.origin[axis], second.origin[axis]) || !same_number(a[axis].lo, b[axis].lo) ||
            !same_number(a[axis].hi, b[axis].hi))
            return false;
    return true;
}
} // namespace
OrdinaryIntersectionOperand OrdinaryIntersectionOperand::existing(ConvexGeometryPtr<ordinary_mode> geometry)
{
    if (!geometry)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Intersection operand lacks existing geometry.");
    OrdinaryIntersectionOperand result;
    result.geometry_ = std::move(geometry);
    const auto& body = DynamicGeometryAccess::implementation(*result.geometry_->dynamic_geometry());
    result.frame_ = body->frame;
    result.bounds_ = body->bounds;
    return result;
}
OrdinaryIntersectionOperand OrdinaryIntersectionOperand::native(CellGeometryView<ordinary_mode> view)
{
    const auto facts = NativeCellGeometrySourceAccess::framed_bounds(view);
    OrdinaryIntersectionOperand result;
    const auto binding = NativeCellGeometrySourceAccess::source_cell(view);
    result.source_identity_ = binding.source.get();
    result.source_slot_ = binding.cell_slot;
    result.source_ = std::move(view);
    result.frame_ = {facts.frame_origin, facts.frame_exponent};
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.bounds_[axis] = {facts.bounds[axis][0], facts.bounds[axis][1]};
    return result;
}
void OrdinaryIntersectionOperand::prepare(CellGeometryOperationWorkspace& workspace)
{
    if (geometry_)
        return;
    if (!source_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Intersection operand lacks its native source.");
    geometry_ = source_.materialize(workspace);
    const auto& body = DynamicGeometryAccess::implementation(*geometry_->dynamic_geometry());
    exact_preparation_ = same_frame_bounds(frame_, bounds_, body->frame, body->bounds);
}
OrdinaryIntersectionReceipt::OrdinaryIntersectionReceipt(
    const OrdinaryIntersectionOperand& first, const OrdinaryIntersectionOperand& second)
    : first_(&first), second_(&second), work_(active_work)
{
    if (!work_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Intersection screen requires scoped valid operands.");
    // Match the original Ordinary Operation's parent-policy check before any
    // statistics or arithmetic, including for immediately separated operands.
    require_mode(work_->limits.arithmetic_mode, ordinary_mode);
    if ((!first.geometry_ && !first.source_) || (!second.geometry_ && !second.source_))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Intersection screen requires scoped valid operands.");
    const auto snapshot = [](const OrdinaryIntersectionOperand& operand)
    {
        return Snapshot{
            operand.geometry_.get(), operand.source_identity_, operand.source_slot_, operand.frame_, operand.bounds_};
    };
    first_snapshot_ = snapshot(first);
    second_snapshot_ = snapshot(second);
    operations_ = work_->operations;
    pairs_ = work_->pairs;
    bytes_ = work_->bytes;
    const auto controls = intersection_controls();
    rounding_ = controls.rounding;
    x87_control_ = controls.x87;
    simd_control_ = controls.simd;
    controls_supported_ = controls.supported;
    // Unsupported control encodings use the complete existing factory path;
    // do not perform a speculative screen that would need to be repeated.
    if (!controls_supported_)
        return;
    const auto check = [](const OrdinaryIntersectionOperand& operand)
    {
        if (operand.geometry_ &&
            DynamicGeometryAccess::implementation(*operand.geometry_->dynamic_geometry())->partition_proof_only)
            ordinary_kernel::reject("private partition witness cannot supply a measured intersection.");
    };
    check(first);
    check(second);
    ++statistics.intersections;
    separated_ = ordinary_kernel::separated_framed_bounds<ordinary_mode>(
        first.bounds_, first.frame_, second.bounds_, second.frame_);
    screened_ = true;
    operations_ = work_->operations;
    pairs_ = work_->pairs;
    bytes_ = work_->bytes;
}
OrdinaryIntersectionReceipt OrdinaryIntersectionAccess::screen(
    const OrdinaryIntersectionOperand& first, const OrdinaryIntersectionOperand& second)
{
    return OrdinaryIntersectionReceipt(first, second);
}
ConvexGeometryPtr<ordinary_mode> OrdinaryIntersectionReceipt::finish()
{
    return OrdinaryIntersectionAccess::finish(*this);
}
bool OrdinaryIntersectionReceipt::separated()
{
    OrdinaryIntersectionAccess::validate(*this);
    if (separated_)
        consumed_ = true;
    return separated_;
}
void OrdinaryIntersectionAccess::validate(OrdinaryIntersectionReceipt& receipt)
{
    const auto invalid = [&](const char* message)
    {
        receipt.consumed_ = true;
        throw GeometryError(GeometryErrorCode::InvalidInput, message);
    };
    if (receipt.consumed_)
        invalid("Intersection screen receipt was already consumed.");
    if (!active_work || receipt.work_ != active_work || active_work->operations < receipt.operations_ ||
        active_work->pairs < receipt.pairs_ || active_work->bytes < receipt.bytes_)
        invalid("Intersection screen Work scope or counters changed.");
    if (active_work->limits.arithmetic_mode != ordinary_mode)
    {
        receipt.consumed_ = true;
        require_mode(active_work->limits.arithmetic_mode, ordinary_mode);
    }
    const auto controls = intersection_controls();
    const bool same_controls = controls.rounding == receipt.rounding_ &&
                               ((receipt.controls_supported_ && controls.supported &&
                                    controls.x87 == receipt.x87_control_ && controls.simd == receipt.simd_control_) ||
                                   (!receipt.screened_ && !receipt.controls_supported_ && !controls.supported));
    if (!same_controls)
        invalid("Intersection screen numeric controls changed.");
    const auto unchanged =
        [](const OrdinaryIntersectionOperand& actual, const OrdinaryIntersectionReceipt::Snapshot& snapshot)
    {
        return actual.source_identity_ == snapshot.source && actual.source_slot_ == snapshot.source_slot &&
               (snapshot.source || actual.geometry_.get() == snapshot.geometry) &&
               same_frame_bounds(actual.frame_, actual.bounds_, snapshot.frame, snapshot.bounds);
    };
    if (!unchanged(*receipt.first_, receipt.first_snapshot_) || !unchanged(*receipt.second_, receipt.second_snapshot_))
        invalid("Intersection screen operand changed.");
}
ConvexGeometryPtr<ordinary_mode> OrdinaryIntersectionAccess::finish(OrdinaryIntersectionReceipt& receipt)
{
    validate(receipt);
    receipt.consumed_ = true;
    if (receipt.separated_)
        return {};
    const auto& first = *receipt.first_;
    const auto& second = *receipt.second_;
    if (!first.geometry_ || !second.geometry_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Intersection screen lacks prepared exact operands.");
    if (!receipt.screened_ || !first.exact_preparation_ || !second.exact_preparation_)
        return first.geometry_->intersect(*second.geometry_);
    const auto& original = first.geometry_->dynamic_geometry();
    const auto& other = second.geometry_->dynamic_geometry();
    const auto& body = DynamicGeometryAccess::implementation(*original);
    Operation scope(body->limits);
    return ConvexCellGeometry<ordinary_mode>::from_dynamic(
        ordinary_kernel::GeometryKernel<ordinary_mode>(body).intersect_after_bound_screen(*other));
}
} // namespace SimpleFluid::PLIC::details
