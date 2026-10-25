/** @file FiniteGeometry.tcc
 * @brief Compiled typed finite geometry adapters over policy-specific kernels.
 */
namespace SimpleFluid::PLIC
{
namespace
{
template<ArithmeticMode Mode>
using SelectedGeometryKernel = std::conditional_t<Mode == ArithmeticMode::Ordinary,
    details::ordinary_kernel::GeometryKernel<Mode>, details::certification_kernel::GeometryKernel<Mode>>;
template<ArithmeticMode Mode>
using SelectedPartitionKernel = std::conditional_t<Mode == ArithmeticMode::Ordinary,
    details::ordinary_kernel::PartitionKernel<Mode>, details::certification_kernel::PartitionKernel<Mode>>;
template<ArithmeticMode Mode> GeometryLimits typed_limits(GeometryLimits limits)
{
    limits.arithmetic_mode = Mode;
    return limits;
}
template<ArithmeticMode Mode> GeometryValue<Mode> typed_value(const DynamicGeometryValue& value)
{
    details::require_mode(value.arithmetic_mode, Mode);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return {value.value};
    else
        return {value.value, value.absolute_error};
}
template<ArithmeticMode Mode> std::vector<DynamicPoint> dynamic_points(std::span<const Point<Mode>> points)
{
    details::active_work->charge(points.size(), sizeof(DynamicPoint));
    std::vector<DynamicPoint> result;
    result.reserve(points.size());
    for (const auto& point : points)
        result.push_back(point.dynamic());
    return result;
}
template<ArithmeticMode Mode> std::vector<Point<Mode>> typed_points(std::span<const DynamicPoint> points)
{
    details::active_work->charge(points.size(), sizeof(Point<Mode>));
    std::vector<Point<Mode>> result;
    result.reserve(points.size());
    for (const auto& point : points)
        result.emplace_back(point);
    return result;
}
template<ArithmeticMode Mode>
std::vector<DynamicConvexGeometryPtr> dynamic_geometries(std::span<const ConvexGeometryPtr<Mode>> sources)
{
    details::active_work->charge(sources.size(), sizeof(DynamicConvexGeometryPtr));
    std::vector<DynamicConvexGeometryPtr> result;
    result.reserve(sources.size());
    for (const auto& source : sources)
        result.push_back(source ? source->dynamic_geometry() : nullptr);
    return result;
}
void preflight_shell(std::size_t vertices, std::size_t faces)
{
    const auto& limits = details::active_work->limits;
    if (vertices < 4 || vertices > limits.maximum_vertices || faces < 4 || faces > limits.maximum_faces)
        details::reject("invalid or excessive shell size before typed conversion allocation.");
}
template<ArithmeticMode Mode> ClipResult<Mode> typed_clip(const DynamicClipResult& source)
{
    ClipResult<Mode> result;
    result.inside = ConvexCellGeometry<Mode>::from_dynamic(source.inside);
    result.outside = ConvexCellGeometry<Mode>::from_dynamic(source.outside);
    // Empty optional sections carry no numerical certificate. Their default
    // runtime tag is immaterial; return a zero value in the selected policy.
    if (!source.section.vertices.empty())
    {
        result.section.vertices = typed_points<Mode>(source.section.vertices);
        result.section.area = typed_value<Mode>(source.section.area);
        result.section.centroid = Point<Mode>(source.section.centroid);
    }
    return result;
}
} // namespace

template<ArithmeticMode Mode> Point<Mode>::Point(const DynamicPoint& point)
{
    details::require_mode(point.arithmetic_mode, Mode);
    this->value = point.value;
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        this->absolute_error = point.absolute_error;
    provenance_ = DynamicGeometryAccess::identity(point);
}
template<ArithmeticMode Mode> DynamicPoint Point<Mode>::dynamic() const
{
    std::array<long double, 3> error{};
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        error = this->absolute_error;
    else if (provenance_)
        error = provenance_->absolute_error; // Original source evidence, not an ordinary numerical bound.
    DynamicPoint result(this->value, error, Mode);
    result.provenance_ = provenance_;
    return result;
}

template<ArithmeticMode Mode>
ConvexCellGeometry<Mode>::ConvexCellGeometry(DynamicConvexGeometryPtr geometry) : geometry_(std::move(geometry))
{
    details::require_mode(geometry_->arithmetic_mode(), Mode);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::from_dynamic(DynamicConvexGeometryPtr geometry)
{
    if (!geometry)
        return {};
    details::require_mode(geometry->arithmetic_mode(), Mode);
    details::Operation scope(geometry->impl_->limits);
    details::active_work->charge(1, sizeof(ConvexCellGeometry));
    return ConvexGeometryPtr<Mode>(new ConvexCellGeometry(std::move(geometry)));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::make(
    std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& faces, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    return from_dynamic(SelectedGeometryKernel<Mode>::make(vertices, faces, typed_limits<Mode>(limits)));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::make_points(
    std::span<const Point<Mode>> vertices, const std::vector<std::vector<std::size_t>>& faces, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    preflight_shell(vertices.size(), faces.size());
    return from_dynamic(
        SelectedGeometryKernel<Mode>::make_bounded(dynamic_points<Mode>(vertices), faces, typed_limits<Mode>(limits)));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::make_local(GeometryPoint origin,
    std::span<const Point<Mode>> vertices, const std::vector<std::vector<std::size_t>>& faces, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    preflight_shell(vertices.size(), faces.size());
    return from_dynamic(SelectedGeometryKernel<Mode>::make_bounded_local(
        origin, dynamic_points<Mode>(vertices), faces, typed_limits<Mode>(limits)));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::from_geometry(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, GeometryLimits limits)
{
    limits = typed_limits<Mode>(limits);
    const auto snapshot = details::prepare_source_geometry(source, limits);
    details::Operation scope(limits);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return from_dynamic(details::ordinary_kernel::GeometryKernel<Mode>::from_snapshot(snapshot, limits));
    else
        return from_dynamic(details::certification_kernel::GeometryKernel<Mode>::from_snapshot(snapshot, limits));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::box(GeometryPoint lower, GeometryPoint upper, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    return from_dynamic(SelectedGeometryKernel<Mode>::box(lower, upper, typed_limits<Mode>(limits)));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::tetrahedron(
    const std::array<Point<Mode>, 4>& points, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    std::array<DynamicPoint, 4> vertices;
    for (std::size_t i = 0; i < 4; ++i)
        vertices[i] = points[i].dynamic();
    return from_dynamic(SelectedGeometryKernel<Mode>::tetrahedron(vertices, typed_limits<Mode>(limits)));
}
template<ArithmeticMode Mode> GeometryValue<Mode> ConvexCellGeometry<Mode>::volume() const
{
    return typed_value<Mode>(SelectedGeometryKernel<Mode>(geometry_->impl_).volume());
}
template<ArithmeticMode Mode> GeometryMoments<Mode> ConvexCellGeometry<Mode>::moments(GeometryPoint origin) const
{
    const auto source = SelectedGeometryKernel<Mode>(geometry_->impl_).moments(origin);
    GeometryMoments<Mode> result;
    result.volume = typed_value<Mode>(source.volume);
    result.origin = source.origin;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.first_moment[axis] = typed_value<Mode>(source.first_moment[axis]);
    return result;
}
template<ArithmeticMode Mode> GeometryPoint ConvexCellGeometry<Mode>::centroid() const
{
    return SelectedGeometryKernel<Mode>(geometry_->impl_).centroid();
}
template<ArithmeticMode Mode>
bool ConvexCellGeometry<Mode>::bounds_prove_disjoint_interiors(const ConvexCellGeometry& other) const
{
    return SelectedGeometryKernel<Mode>(geometry_->impl_).bounds_prove_disjoint_interiors(*other.geometry_);
}
template<ArithmeticMode Mode>
std::vector<char> ConvexCellGeometry<Mode>::encode(const GeometryPacketContext& context, std::size_t budget) const
{
    return SelectedGeometryKernel<Mode>(geometry_->impl_).encode(context, budget);
}
template<ArithmeticMode Mode> std::vector<Point<Mode>> ConvexCellGeometry<Mode>::points() const
{
    details::Operation scope(geometry_->impl_->limits);
    return typed_points<Mode>(geometry_->points());
}
template<ArithmeticMode Mode> std::vector<Point<Mode>> ConvexCellGeometry<Mode>::local_points() const
{
    details::Operation scope(geometry_->impl_->limits);
    return typed_points<Mode>(SelectedGeometryKernel<Mode>(geometry_->impl_).local_points());
}
template<ArithmeticMode Mode> std::array<GeometryValue<Mode>, 3> ConvexCellGeometry<Mode>::axis_spans() const
{
    const auto spans = SelectedGeometryKernel<Mode>(geometry_->impl_).axis_spans();
    return {typed_value<Mode>(spans[0]), typed_value<Mode>(spans[1]), typed_value<Mode>(spans[2])};
}
template<ArithmeticMode Mode>
Support<Mode> ConvexCellGeometry<Mode>::support(GeometryPoint normal, GeometryPoint anchor) const
{
    const auto result = SelectedGeometryKernel<Mode>(geometry_->impl_).support(normal, anchor);
    return {typed_value<Mode>(result.lower), typed_value<Mode>(result.upper)};
}
template<ArithmeticMode Mode>
ClipResult<Mode> ConvexCellGeometry<Mode>::clip(const GeometryHalfspace& plane, bool section) const
{
    details::Operation scope(geometry_->impl_->limits);
    return typed_clip<Mode>(SelectedGeometryKernel<Mode>(geometry_->impl_).clip(plane, section));
}
template<ArithmeticMode Mode>
ClipResult<Mode> ConvexCellGeometry<Mode>::clip_by_face(
    const ConvexCellGeometry& source, std::size_t face, bool section) const
{
    details::Operation scope(geometry_->impl_->limits);
    return typed_clip<Mode>(
        SelectedGeometryKernel<Mode>(geometry_->impl_).clip_by_face(*source.geometry_, face, section));
}
template<ArithmeticMode Mode> PhaseMeasures<Mode> ConvexCellGeometry<Mode>::phases(const GeometryHalfspace& plane) const
{
    const auto result = SelectedGeometryKernel<Mode>(geometry_->impl_).phases(plane);
    return {typed_value<Mode>(result.bulk), typed_value<Mode>(result.gas), typed_value<Mode>(result.carrier)};
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::intersect(const ConvexCellGeometry& source) const
{
    details::Operation scope(geometry_->impl_->limits);
    return from_dynamic(SelectedGeometryKernel<Mode>(geometry_->impl_).intersect(*source.geometry_));
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::translated(GeometryPoint displacement) const
{
    details::Operation scope(geometry_->impl_->limits);
    return from_dynamic(SelectedGeometryKernel<Mode>(geometry_->impl_).translated(displacement));
}
template<ArithmeticMode Mode>
void ConvexCellGeometry<Mode>::certify_partition(std::span<const ConvexGeometryPtr<Mode>> pieces, double budget) const
{
    details::Operation scope(geometry_->impl_->limits);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        details::ordinary_native::certify_cell_partition(*this, pieces, budget);
    else
        SelectedGeometryKernel<Mode>(geometry_->impl_).certify_partition(dynamic_geometries<Mode>(pieces), budget);
}
template<ArithmeticMode Mode>
void ConvexCellGeometry<Mode>::certify_fragment_partition(
    std::span<const ConvexGeometryPtr<Mode>> pieces, double budget) const
{
    details::Operation scope(geometry_->impl_->limits);
    SelectedGeometryKernel<Mode>(geometry_->impl_).certify_partition(dynamic_geometries<Mode>(pieces), budget);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::decode(std::span<const char> packet,
    const GeometryPacketContext& context, std::span<const ConvexGeometryPtr<Mode>> sources, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    if (packet.size() > limits.byte_budget || sources.empty())
        details::reject(Mode, "numerical decode requires capped payload and authenticated source registry.");
    return from_dynamic(SelectedGeometryKernel<Mode>::decode(
        packet, context, dynamic_geometries<Mode>(sources), typed_limits<Mode>(limits)));
}

template<ArithmeticMode Mode>
ConvexPartition<Mode>::ConvexPartition(std::span<const ConvexGeometryPtr<Mode>> donors, GeometryLimits limits)
    : limits_(typed_limits<Mode>(limits))
{
    details::Operation scope(limits_);
    donors_ = SelectedGeometryKernel<Mode>::validate_donors(dynamic_geometries<Mode>(donors), limits_);
}
template<ArithmeticMode Mode>
std::vector<PartitionPiece<Mode>> ConvexPartition<Mode>::partition(
    const ConvexCellGeometry<Mode>& parent, std::size_t maximum) const
{
    details::Operation scope(limits_);
    std::vector<PartitionPiece<Mode>> result;
    auto pieces = SelectedPartitionKernel<Mode>(donors_, limits_).partition(*parent.dynamic_geometry(), maximum);
    details::active_work->charge(pieces.size(), sizeof(PartitionPiece<Mode>));
    result.reserve(pieces.size());
    for (auto& piece : pieces)
        result.push_back({piece.donor, ConvexCellGeometry<Mode>::from_dynamic(std::move(piece.geometry))});
    return result;
}

template<ArithmeticMode Mode> std::vector<PlaneAuthority> common_supports(std::span<const Point<Mode>> points)
{
    if (points.empty())
        return {};
    std::vector<PlaneAuthority> result;
    for (const auto& key : points.front().supports())
        if (!key.key.empty() &&
            std::all_of(points.begin() + 1, points.end(), [&](const auto& point)
                { return std::find(point.supports().begin(), point.supports().end(), key) != point.supports().end(); }))
            result.push_back(key);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
template<ArithmeticMode Mode> Point<Mode> point_average(std::span<const Point<Mode>> points)
{
    // Empty input still uses the selected mode for its rejection diagnostic.
    details::Operation scope(typed_limits<Mode>({}));
    if (points.empty())
        details::reject(Mode, "empty affine average.");
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return Point<Mode>(details::ordinary_kernel::dynamic_point_average<Mode>(dynamic_points<Mode>(points)));
    else
        return Point<Mode>(details::certification_kernel::dynamic_point_average<Mode>(dynamic_points<Mode>(points)));
}
template<ArithmeticMode Mode>
Point<Mode> point_interpolate(const Point<Mode>& a, const Point<Mode>& b, long double fraction)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return Point<Mode>(
            details::ordinary_kernel::dynamic_point_interpolate<Mode>(a.dynamic(), b.dynamic(), fraction));
    else
        return Point<Mode>(
            details::certification_kernel::dynamic_point_interpolate<Mode>(a.dynamic(), b.dynamic(), fraction));
}
template<ArithmeticMode Mode> GeometryValue<Mode> orientation(const std::array<Point<Mode>, 4>& points)
{
    std::array<DynamicPoint, 4> vertices;
    for (std::size_t i = 0; i < 4; ++i)
        vertices[i] = points[i].dynamic();
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return typed_value<Mode>(details::ordinary_kernel::dynamic_orientation<Mode>(vertices));
    else
        return typed_value<Mode>(details::certification_kernel::dynamic_orientation<Mode>(vertices));
}
template<ArithmeticMode Mode> bool points_identical(const Point<Mode>& a, const Point<Mode>& b)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return details::ordinary_kernel::dynamic_points_identical<Mode>(a.dynamic(), b.dynamic());
    else
        return details::certification_kernel::dynamic_points_identical<Mode>(a.dynamic(), b.dynamic());
}
template<ArithmeticMode Mode> Point<Mode> point_displaced(const Point<Mode>& a, const Point<Mode>& b)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return Point<Mode>(details::ordinary_kernel::dynamic_point_displaced<Mode>(a.dynamic(), b.dynamic()));
    else
        return Point<Mode>(details::certification_kernel::dynamic_point_displaced<Mode>(a.dynamic(), b.dynamic()));
}
template<ArithmeticMode Mode>
Point<Mode> source_point(const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, std::span<const char> identity)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return Point<Mode>(details::ordinary_kernel::dynamic_source_point<Mode>(source, identity, Mode));
    else
        return Point<Mode>(details::certification_kernel::dynamic_source_point<Mode>(source, identity, Mode));
}
template<ArithmeticMode Mode>
Point<Mode> source_point(
    std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources, std::span<const char> identity)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return Point<Mode>(details::ordinary_kernel::dynamic_source_point<Mode>(sources, identity, Mode));
    else
        return Point<Mode>(details::certification_kernel::dynamic_source_point<Mode>(sources, identity, Mode));
}
template<ArithmeticMode Mode>
Point<Mode> backtrace_constrained(const Point<Mode>& source, GeometryPoint velocity, double timestep,
    std::span<const FaceConstraint<Mode>> walls, GeometryLimits limits)
{
    details::Operation scope(typed_limits<Mode>(limits));
    details::active_work->charge(walls.size(), sizeof(DynamicFaceConstraint));
    std::vector<DynamicFaceConstraint> dynamic_walls;
    dynamic_walls.reserve(walls.size());
    for (const auto& wall : walls)
        dynamic_walls.push_back({wall.geometry ? wall.geometry->dynamic_geometry() : nullptr, wall.face});
    return Point<Mode>(SelectedGeometryKernel<Mode>::backtrace(
        source.dynamic(), velocity, timestep, dynamic_walls, typed_limits<Mode>(limits)));
}

#include "geometry/PLIC/details/ordinary/NativeCellGeometry.tcc"

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
    template Point<mode> source_point(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::span<const char>);     \
    template Point<mode> source_point(                                                                                 \
        std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>>, std::span<const char>);                          \
    template Point<mode> backtrace_constrained(                                                                        \
        const Point<mode>&, GeometryPoint, double, std::span<const FaceConstraint<mode>>, GeometryLimits);
SIMPLEFLUID_INSTANTIATE_TYPED_POINTS(ArithmeticMode::Ordinary)
SIMPLEFLUID_INSTANTIATE_TYPED_POINTS(ArithmeticMode::BoundedCertification)
#undef SIMPLEFLUID_INSTANTIATE_TYPED_POINTS
} // namespace SimpleFluid::PLIC
