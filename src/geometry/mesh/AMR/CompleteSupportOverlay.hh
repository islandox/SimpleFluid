/** @file CompleteSupportOverlay.hh
 * @brief Shared complete-support validation and overlay kernel.
 */
#pragma once
#include "utils/export_helper.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct SupportOccurrence
{
    AdaptiveCellKey cell;
    int rank = -1, orientation = 0;
    AdaptiveBytes domain;
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> image;
    std::uint64_t native_face = 0;
    bool exterior = false, native_whole = false;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> whole_geometry;
    std::string boundary;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> face;
};
using SupportFace = PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>;
template<class... Args> class BorrowedSupportCallback
{
public:
    template<class F> BorrowedSupportCallback(F& callback) noexcept
        : object_(&callback), invoke_([](void* object, Args... args)
              { (*static_cast<F*>(object))(std::forward<Args>(args)...); })
    {}
    void operator()(Args... args) const { invoke_(object_, std::forward<Args>(args)...); }

private:
    void* object_;
    void (*invoke_)(void*, Args...);
};
using SupportEmit = BorrowedSupportCallback<std::size_t, std::optional<std::size_t>, SupportFace>;
using SupportCharge = BorrowedSupportCallback<std::size_t>;
using SupportChargeFace = BorrowedSupportCallback<const SupportFace&>;

/** Optional live construction envelope around the unchanged geometry kernel.
 * begin must admit before construction and leave no lease on failure;
 * end releases the same scoped lease and
 * must not throw. Surviving output faces are admitted through charge_face. */
enum class SupportConstructionKind
{
    Pair,
    Partition
};
using SupportConstructionBegin =
    BorrowedSupportCallback<SupportConstructionKind, const SupportFace&, std::span<const SupportFace>>;
using SupportConstructionEnd = BorrowedSupportCallback<>;
struct SupportConstructionAdmission
{
    SupportConstructionBegin begin;
    SupportConstructionEnd end;
};
/** Conservative construction envelope for the same represented planar kernel. */
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t support_construction_capacity_bound(
    SupportConstructionKind, const SupportFace&, std::span<const SupportFace>);
/** Entries are a complete support in legacy arrival order. Geometry has already
 * been translated to its authority image and native containment validated.
 * The caller owns the scoped admission and duplicate canonical-face audit.
 * Admission callbacks preserve arithmetic controls; emit may change controls,
 * in which case subsequent candidates use exhaustive original ordering. */
SIMPLEFLUID_LOCAL void overlay_complete_support(std::span<const SupportOccurrence>, std::size_t budget,
    const SupportEmit&, const SupportCharge&, const SupportChargeFace&, const SupportConstructionAdmission* = nullptr);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
