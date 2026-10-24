#include "geometry/details/BoundedGeometryKernels.hh"

namespace SimpleFluid::Meshes
{
std::vector<BoundedPlaneAuthority> bounded_common_supports(std::span<const BoundedPoint> points)
{
    for (const auto& p : points)
        require_mode(p.arithmetic_mode, points.front().arithmetic_mode);
    if (points.empty())
        return {};
    std::vector<BoundedPlaneAuthority> result;
    for (const auto& key : points.front().supports())
        if (!key.key.empty() &&
            std::all_of(points.begin() + 1, points.end(), [&](const auto& p)
                { return std::find(p.supports().begin(), p.supports().end(), key) != p.supports().end(); }))
            result.push_back(key);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
bool bounded_geometry_environment()
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
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
BoundedGeometryStatistics bounded_geometry_statistics()
{
    return statistics;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
void reset_bounded_geometry_statistics()
{
    statistics = {};
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
void bounded_record_ledger_operations(std::uint64_t count)
{
    if (count > std::numeric_limits<std::uint64_t>::max() - statistics.ledger_operations)
        reject("ledger instrumentation counter overflow.");
    statistics.ledger_operations += count;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
std::size_t bounded_point_storage_bytes(const BoundedPoint& p)
{
    return BoundedGeometryAccess::storage_bytes(p);
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
BoundedConvexGeometry::BoundedConvexGeometry(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
ArithmeticMode BoundedConvexGeometry::arithmetic_mode() const noexcept
{
    return impl_->limits.arithmetic_mode;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
GeometryPoint BoundedConvexGeometry::origin() const
{
    return impl_->reference_origin;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
const std::vector<BoundedPoint>& BoundedConvexGeometry::points() const
{
    return impl_->displayed;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
const std::vector<Face>& BoundedConvexGeometry::faces() const
{
    return impl_->faces;
}
} // namespace SimpleFluid::Meshes

#define SIMPLEFLUID_GEOMETRY_KERNEL ordinary_kernel
#include "geometry/details/BoundedConvexConstruction.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL
#define SIMPLEFLUID_GEOMETRY_KERNEL certification_kernel
#include "geometry/details/BoundedConvexConstruction.tcc"
#undef SIMPLEFLUID_GEOMETRY_KERNEL

namespace SimpleFluid::Meshes
{
BoundedPoint bounded_point_average(std::span<const BoundedPoint> vertices)
{
    if ((vertices.empty() ? ArithmeticMode::BoundedCertification : vertices.front().arithmetic_mode) ==
        ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_point_average<ArithmeticMode::Ordinary>(vertices);
    return certification_kernel::bounded_point_average<ArithmeticMode::BoundedCertification>(vertices);
}

BoundedPoint bounded_point_interpolate(const BoundedPoint& a, const BoundedPoint& b, long double fraction)
{
    if ((a.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_point_interpolate<ArithmeticMode::Ordinary>(a, b, fraction);
    return certification_kernel::bounded_point_interpolate<ArithmeticMode::BoundedCertification>(a, b, fraction);
}

template<ArithmeticMode Mode> long double BoundedGeometryValue::lower() const
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    require_mode(arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<Mode>(*this).lower();
    else
        return certification_kernel::ValueKernel<Mode>(*this).lower();
}

template<ArithmeticMode Mode> long double BoundedGeometryValue::upper() const
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    require_mode(arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<Mode>(*this).upper();
    else
        return certification_kernel::ValueKernel<Mode>(*this).upper();
}

template<ArithmeticMode Mode> double BoundedGeometryValue::as_double() const
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    require_mode(arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<Mode>(*this).as_double();
    else
        return certification_kernel::ValueKernel<Mode>(*this).as_double();
}

template long double BoundedGeometryValue::lower<ArithmeticMode::Ordinary>() const;
template long double BoundedGeometryValue::lower<ArithmeticMode::BoundedCertification>() const;
template long double BoundedGeometryValue::upper<ArithmeticMode::Ordinary>() const;
template long double BoundedGeometryValue::upper<ArithmeticMode::BoundedCertification>() const;
template double BoundedGeometryValue::as_double<ArithmeticMode::Ordinary>() const;
template double BoundedGeometryValue::as_double<ArithmeticMode::BoundedCertification>() const;

long double BoundedGeometryValue::lower() const
{
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).lower();
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).lower();
}

long double BoundedGeometryValue::upper() const
{
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).upper();
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).upper();
}

double BoundedGeometryValue::as_double() const
{
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).as_double();
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).as_double();
}

bool BoundedGeometryValue::relative_error_at_most(long double target, double tolerance) const
{
    if (arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::ValueKernel<ArithmeticMode::Ordinary>(*this).relative_error_at_most(target, tolerance);
    return certification_kernel::ValueKernel<ArithmeticMode::BoundedCertification>(*this).relative_error_at_most(
        target, tolerance);
}

bool bounded_points_identical(const BoundedPoint& a, const BoundedPoint& b)
{
    if ((a.arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_points_identical<ArithmeticMode::Ordinary>(a, b);
    return certification_kernel::bounded_points_identical<ArithmeticMode::BoundedCertification>(a, b);
}

BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>& vertices)
{
    if ((vertices.front().arithmetic_mode) == ArithmeticMode::Ordinary)
        return ordinary_kernel::bounded_orientation<ArithmeticMode::Ordinary>(vertices);
    return certification_kernel::bounded_orientation<ArithmeticMode::BoundedCertification>(vertices);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::make(vertices, faces, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::make(vertices, faces, limits);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::make_bounded(
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::make_bounded(vertices, faces, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::make_bounded(
        vertices, faces, limits);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::make_bounded_local(GeometryPoint origin,
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::make_bounded_local(
            origin, vertices, faces, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::make_bounded_local(
        origin, vertices, faces, limits);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::from_geometry(
    const ConvexCellGeometry& source, BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::from_geometry(source, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::from_geometry(source, limits);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::box(
    GeometryPoint lower, GeometryPoint upper, BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::box(lower, upper, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::box(lower, upper, limits);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::tetrahedron(
    const std::array<BoundedPoint, 4>& vertices, BoundedGeometryLimits limits)
{
    if (limits.arithmetic_mode == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>::tetrahedron(vertices, limits);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>::tetrahedron(vertices, limits);
}

BoundedGeometryValue BoundedConvexGeometry::volume() const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).volume();
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).volume();
}

BoundedGeometryMoments BoundedConvexGeometry::moments() const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).moments();
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).moments();
}

BoundedGeometryMoments BoundedConvexGeometry::moments(GeometryPoint origin) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).moments(origin);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).moments(origin);
}

GeometryPoint BoundedConvexGeometry::centroid() const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).centroid();
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).centroid();
}

std::vector<BoundedPoint> BoundedConvexGeometry::local_points() const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).local_points();
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).local_points();
}

std::array<BoundedGeometryValue, 3> BoundedConvexGeometry::axis_spans() const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).axis_spans();
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).axis_spans();
}

BoundedSupport BoundedConvexGeometry::support(GeometryPoint normal, GeometryPoint anchor) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return ordinary_kernel::GeometryKernel<ArithmeticMode::Ordinary>(impl_).support(normal, anchor);
    return certification_kernel::GeometryKernel<ArithmeticMode::BoundedCertification>(impl_).support(normal, anchor);
}
} // namespace SimpleFluid::Meshes
