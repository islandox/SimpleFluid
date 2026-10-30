#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/mesh/MultiRegionMesh.hh"

#include "geometry/PLIC/details/bounded/ConvexConstruction.tcc"
#include "geometry/PLIC/details/ordinary/ConvexConstruction.tcc"

// Native definitions consume the common typed helpers before explicit instantiation.
#include "geometry/PLIC/details/ConvexGeometry.tcc"
#include "geometry/PLIC/details/ordinary/NativeCellGeometry.tcc"

namespace SimpleFluid::PLIC
{
std::vector<PlaneAuthority> dynamic_common_supports(std::span<const DynamicPoint> points)
{
    for (const auto& p : points)
        require_mode(p.arithmetic_mode, points.front().arithmetic_mode);
    if (points.empty())
        return {};
    std::vector<PlaneAuthority> result;
    for (const auto& key : points.front().supports())
        if (!key.key.empty() &&
            std::all_of(points.begin() + 1, points.end(), [&](const auto& p)
                { return has_support(p, key); }))
            result.push_back(key);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
bool certification_environment()
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__) &&                           \
    !defined(__FAST_MATH__) && (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0)
    if constexpr (!std::numeric_limits<long double>::is_iec559 || std::numeric_limits<long double>::radix != 2 ||
                  std::numeric_limits<long double>::digits < 64 ||
                  std::numeric_limits<long double>::max_exponent < 16384 || FLT_EVAL_METHOD != 0)
        return false;
    const unsigned csr = _mm_getcsr();
    unsigned short control;
    __asm__ __volatile__("fnstcw %0" : "=m"(control));
    return std::fegetround() == FE_TONEAREST && (csr & ((1u << 15) | (1u << 6) | (3u << 13))) == 0 &&
           (csr & (0x3fu << 7)) == (0x3fu << 7) && (control & 0x3fu) == 0x3fu && (control & (3u << 8)) == (3u << 8) &&
           (control & (3u << 10)) == 0;
#else
    return false;
#endif
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
GeometryStatistics geometry_statistics()
{
    return statistics;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
void reset_geometry_statistics()
{
    statistics = {};
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
void record_ledger_operations(std::uint64_t count)
{
    if (count > std::numeric_limits<std::uint64_t>::max() - statistics.ledger_operations)
        reject("ledger instrumentation counter overflow.");
    statistics.ledger_operations += count;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
std::size_t dynamic_point_storage_bytes(const DynamicPoint& p)
{
    return DynamicGeometryAccess::storage_bytes(p);
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
DynamicConvexGeometry::DynamicConvexGeometry(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
ArithmeticMode DynamicConvexGeometry::arithmetic_mode() const noexcept
{
    return impl_->limits.arithmetic_mode;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
GeometryPoint DynamicConvexGeometry::origin() const
{
    return impl_->reference_origin;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
const std::vector<DynamicPoint>& DynamicConvexGeometry::points() const
{
    return impl_->displayed;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
const std::vector<Face>& DynamicConvexGeometry::faces() const
{
    return impl_->faces;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{

template<ArithmeticMode Mode> long double DynamicGeometryValue::lower() const
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    require_mode(arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<Mode>(*this).lower();
    else
        return certification_kernel::ValueKernel<Mode>(*this).lower();
}

template<ArithmeticMode Mode> long double DynamicGeometryValue::upper() const
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    require_mode(arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<Mode>(*this).upper();
    else
        return certification_kernel::ValueKernel<Mode>(*this).upper();
}

template<ArithmeticMode Mode> double DynamicGeometryValue::as_double() const
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    require_mode(arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<Mode>(*this).as_double();
    else
        return certification_kernel::ValueKernel<Mode>(*this).as_double();
}

template long double DynamicGeometryValue::lower<ArithmeticMode::Ordinary>() const;
template long double DynamicGeometryValue::lower<ArithmeticMode::BoundedCertification>() const;
template long double DynamicGeometryValue::upper<ArithmeticMode::Ordinary>() const;
template long double DynamicGeometryValue::upper<ArithmeticMode::BoundedCertification>() const;
template double DynamicGeometryValue::as_double<ArithmeticMode::Ordinary>() const;
template double DynamicGeometryValue::as_double<ArithmeticMode::BoundedCertification>() const;

long double DynamicGeometryValue::lower() const
{
    ++statistics.runtime_dispatches;
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).lower();
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).lower();
}

long double DynamicGeometryValue::upper() const
{
    ++statistics.runtime_dispatches;
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).upper();
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).upper();
}

double DynamicGeometryValue::as_double() const
{
    ++statistics.runtime_dispatches;
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).as_double();
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).as_double();
}

bool DynamicGeometryValue::relative_error_at_most(long double target, double tolerance) const
{
    ++statistics.runtime_dispatches;
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).relative_error_at_most(target, tolerance);
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).relative_error_at_most(
        target, tolerance);
}

} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC::details
{
GeometrySourceSnapshot prepare_source_geometry(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, GeometryLimits limits)
{
    limits.arithmetic_mode = ArithmeticMode::Ordinary;
    Operation preparation(limits);
    if (source.storage_bytes() > limits.byte_budget)
        reject("source authority exceeds bootstrap storage cap.");
    const auto bytes = active_work->bytes, operations = active_work->operations, pairs = active_work->pairs;
    GeometrySourceSnapshot snapshot;
    snapshot.shell = source.numerical_shell();
    active_work->charge(snapshot.shell.relative_vertices.capacity(), sizeof(snapshot.shell.relative_vertices.front()));
    active_work->charge(snapshot.shell.outward_faces.capacity(), sizeof(std::vector<std::size_t>));
    active_work->charge(
        snapshot.shell.face_support_identities.capacity(), sizeof(PlaneSupportIdentity<ArithmeticMode::Ordinary>));
    for (const auto& face : snapshot.shell.outward_faces)
        active_work->charge(face.capacity(), sizeof(std::size_t));
    for (const auto& support : snapshot.shell.face_support_identities)
        active_work->charge(support.payload_bytes(), 1);
    snapshot.preparation_bytes = active_work->bytes - bytes;
    snapshot.preparation_operations = active_work->operations - operations;
    snapshot.preparation_pairs = active_work->pairs - pairs;
    snapshot.charged_work = preparation.previous ? active_work : nullptr;
    return snapshot;
}
void charge_source_preparation(const GeometrySourceSnapshot& source)
{
    if (source.charged_work == active_work)
        return;
    active_work->charge(source.preparation_bytes, 1);
    if (source.preparation_operations >
        active_work->operation_limit - std::min(active_work->operations, active_work->operation_limit))
        reject("source preparation exhausts the geometry operation cap.");
    if (source.preparation_pairs >
        active_work->limits.maximum_pair_tests - std::min(active_work->pairs, active_work->limits.maximum_pair_tests))
        reject("source preparation exhausts the geometry pair-test cap.");
    active_work->operations += source.preparation_operations;
    active_work->pairs += source.preparation_pairs;
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
template class Point<ArithmeticMode::Ordinary>;
template class Point<ArithmeticMode::BoundedCertification>;
template class ConvexCellGeometry<ArithmeticMode::Ordinary>;
template class ConvexCellGeometry<ArithmeticMode::BoundedCertification>;
template class ConvexPartition<ArithmeticMode::Ordinary>;
template class ConvexPartition<ArithmeticMode::BoundedCertification>;
#define SIMPLEFLUID_INSTANTIATE_TYPED_POINTS(mode)                                                                     \
    template std::vector<PlaneAuthority> common_supports(std::span<const Point<mode>>);                                \
    template Point<mode> point_average(std::span<const Point<mode>>);                                                  \
    template Point<mode> point_interpolate(const Point<mode>&, const Point<mode>&, long double);                       \
    template GeometryValue<mode> orientation(const std::array<Point<mode>, 4>&);                                       \
    template bool points_identical(const Point<mode>&, const Point<mode>&);                                            \
    template Point<mode> point_displaced(const Point<mode>&, const Point<mode>&);                                      \
    template Point<mode> source_point(                                                                                 \
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, const PointIdentity<ArithmeticMode::Ordinary>&);          \
    template Point<mode> source_point(                                                                                 \
        std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>>, const PointIdentity<ArithmeticMode::Ordinary>&); \
    template Point<mode> backtrace_constrained(                                                                        \
        const Point<mode>&, GeometryPoint, double, std::span<const FaceConstraint<mode>>, GeometryLimits);
SIMPLEFLUID_INSTANTIATE_TYPED_POINTS(ArithmeticMode::Ordinary)
SIMPLEFLUID_INSTANTIATE_TYPED_POINTS(ArithmeticMode::BoundedCertification)
#undef SIMPLEFLUID_INSTANTIATE_TYPED_POINTS
} // namespace SimpleFluid::PLIC
