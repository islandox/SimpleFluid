/** Compiled finite geometry kernels. Included only by the owning implementation unit.
 * Namespace operators bind shared storage to the fixed arithmetic policy; Mode
 * templates retain the compiled implementation boundary across geometry units.
 */
namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace SIMPLEFLUID_GEOMETRY_KERNEL SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode>
BoundedPoint source_point_fan(
    std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(mode));
    ++statistics.source_imports;
    if (sources.empty() || identity.size() > active_work->limits.byte_budget)
        reject("source point export requires a capped authenticated source fan.");
    std::size_t faces = 0;
    for (const auto* source : sources)
    {
        if (!source || source->storage_bytes() > active_work->limits.byte_budget ||
            source->num_faces() > active_work->limits.maximum_faces - faces)
            reject("source point authority exceeds bootstrap storage/fan cap.");
        faces += source->num_faces();
    }
    const auto coordinates = numerical_point(identity);
    BoundedPoint result;
    result.arithmetic_mode = mode;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.value[axis] = coordinates[axis].value;
        result.absolute_error[axis] = mode == ArithmeticMode::Ordinary ? 0 : coordinates[axis].absolute_error;
    }
    std::vector<BoundedPlaneAuthority> supports;
    for (const auto* source : sources)
        for (std::size_t f = 0; f < source->num_faces(); ++f)
        {
            const auto face = source->face_geometry(f);
            if (!face->contains_exact_point(identity))
                continue;
            const auto bytes = exact_planar_support(*face);
            BoundedPlaneAuthority authority;
            authority.key.push_back(bytes.size());
            authority.key.resize(1 + (bytes.size() + 7) / 8, 0);
            for (std::size_t i = 0; i < bytes.size(); ++i)
                authority.key[1 + i / 8] |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                                            << (8 * (i % 8));
            active_work->charge(authority.key.size(), sizeof(std::uint64_t));
            supports.push_back(std::move(authority));
        }
    BoundedGeometryAccess::seal(result, std::move(supports));
    return result;
}

template<ArithmeticMode Mode>
BoundedPoint bounded_source_point(const ConvexCellGeometry& source, std::span<const char> identity, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    const std::array<const ConvexCellGeometry*, 1> sources{&source};
    return source_point_fan<Mode>(sources, identity, mode);
}

template<ArithmeticMode Mode>
BoundedPoint bounded_source_point(
    std::span<const ConvexGeometryPtr> sources, std::span<const char> identity, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(mode));
    if (sources.size() > active_work->limits.maximum_faces)
        reject("source point fan exceeds bootstrap source-count cap.");
    active_work->charge(sources.size(), sizeof(const ConvexCellGeometry*));
    std::vector<const ConvexCellGeometry*> pointers;
    for (const auto& source : sources)
        pointers.push_back(source.get());
    return source_point_fan<Mode>(pointers, identity, mode);
}

template<ArithmeticMode Mode>
BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement)
{
    static_assert(Mode == kernel_mode);
    require_mode(source.arithmetic_mode, displacement.arithmetic_mode);
    Operation scope(mode_limits(source.arithmetic_mode));
    const auto delta = point<Mode>(displacement);
    if (std::all_of(delta.begin(), delta.end(), [](auto x) { return x.zero(); }))
        return source;
    auto result = bounded<Mode>(point<Mode>(source) + delta);
    BoundedGeometryAccess::seal(result, {});
    return result;
}

template<ArithmeticMode Mode>
BoundedPoint GeometryKernel<Mode>::backtrace(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    require_mode(source.arithmetic_mode, limits.arithmetic_mode);
    Operation scope(limits);
    if (!std::isfinite(timestep) || timestep < 0 || walls.size() > 256)
        reject("invalid or excessive constrained trajectory inputs.");
    const auto input = point<Mode>(source);
    auto displacement = point<Mode>(velocity) * Interval(timestep);
    const bool zero_motion = std::all_of(displacement.begin(), displacement.end(), [](auto x) { return x.zero(); });
    if (walls.empty())
    {
        if (zero_motion)
            return source;
        auto result = bounded<Mode>(input - displacement);
        BoundedGeometryAccess::seal(result, {});
        return result;
    }
    unsigned rank = 0;
    Point first_normal{}, tangent{};
    const auto all_zero = [](const Point& p)
    { return std::all_of(p.begin(), p.end(), [](auto x) { return x.zero(); }); };
    const auto direction = [](const Point& p)
    {
        std::size_t pivot = 3;
        long double strength = 0;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto x = p[axis];
            const auto lower = x.lo > 0 ? x.lo : x.hi < 0 ? -x.hi : 0;
            if (lower > strength)
            {
                pivot = axis;
                strength = lower;
            }
        }
        if (pivot == 3)
            reject("wall constraint rank/conditioning unresolved.");
        Point q;
        for (std::size_t axis = 0; axis < 3; ++axis)
            q[axis] = axis == pivot ? Interval(1) : p[axis] / p[pivot];
        return q; // The pivot ratio is identically one, including its dependency.
    };
    std::vector<BoundedPlaneAuthority> retained;
    std::set<BoundedPlaneAuthority> independent_supports;
    for (const auto& wall : walls)
    {
        active_work->pair();
        if (!wall.geometry || wall.face >= wall.geometry->impl_->surfaces.size())
            reject("invalid wall support constraint.");
        require_mode(source.arithmetic_mode, wall.geometry->arithmetic_mode());
        const auto& plane = wall.geometry->impl_->surfaces[wall.face].plane;
        bool authenticated = false;
        for (const auto& key : plane->authorities)
            if (std::find(source.supports().begin(), source.supports().end(), key) != source.supports().end())
            {
                authenticated = true;
                retained.push_back(key);
            }
        if (!authenticated)
        {
            const auto local = scaled<Mode>(input - point<Mode>(plane->frame.origin), -plane->frame.exponent);
            if (!(dot<Mode>(plane->normal, local - plane->anchor) - plane->offset).zero())
                reject("wall constraint lacks original point/support incidence.");
        }
        // Every catalog entry is authenticated above, including duplicates.
        // A collision-free original support identity proves equal planes even
        // when independently scaled normals have different numerical bounds.
        const bool repeated_support = std::any_of(plane->authorities.begin(), plane->authorities.end(),
            [&](const auto& key) { return independent_supports.contains(key); });
        independent_supports.insert(plane->authorities.begin(), plane->authorities.end());
        if (zero_motion || repeated_support || rank == 3)
            continue; // Every catalog incidence is still authenticated above.
        const auto& normal = plane->normal;
        if (rank == 0)
        {
            if (!nonzero<Mode>(normal))
                reject("wall constraint rank/conditioning unresolved.");
            first_normal = normal;
            rank = 1;
        }
        else if (rank == 1)
        {
            const auto candidate = cross<Mode>(first_normal, normal);
            if (all_zero(candidate))
                continue;
            if (!nonzero<Mode>(candidate))
                reject("wall constraint rank/conditioning unresolved.");
            tangent = candidate;
            rank = 2;
        }
        else
        {
            const auto determinant = dot<Mode>(normal, tangent);
            if (determinant.zero())
                continue;
            if (!(determinant.lo > 0 || determinant.hi < 0))
                reject("wall constraint rank/conditioning unresolved.");
            rank = 3;
        }
    }
    if (zero_motion || rank == 3)
    {
        // A proven rank-three tangent space is exactly {0}. All catalog
        // incidences were authenticated, so retain the original construction.
        return source;
    }
    auto projected = displacement;
    if (rank == 1)
    {
        const auto q = direction(first_normal);
        projected = cross<Mode>(q, cross<Mode>(displacement, q)) * (Interval(1) / dot<Mode>(q, q));
    }
    else if (rank == 2)
    {
        // Construct the common tangent directly. Subtracting nearly parallel
        // normal projections loses their shared dependency and amplifies the
        // enclosure by the inverse squared angle. A nonzero cross-product
        // pivot proves rank two; its normalized direction retains axial zeros.
        const auto q = direction(tangent);
        projected = q * (dot<Mode>(displacement, q) / dot<Mode>(q, q));
    }
    // Verify the enclosure of every original wall residual before attaching
    // its incidence. The downstream geometry/CFL budget consumes the full
    // propagated coordinate error; this does not introduce a wall tolerance.
    if constexpr (!ordinary())
        for (const auto& wall : walls)
        {
            const auto& normal = wall.geometry->impl_->surfaces[wall.face].plane->normal;
            const auto residual = dot<Mode>(normal, projected);
            if (residual.lo > 0 || residual.hi < 0)
                reject("projected trajectory does not enclose original wall tangency.");
        }
    auto result = bounded<Mode>(input - projected);
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
    BoundedGeometryAccess::seal(result, std::move(retained));
    return result;
}

template BoundedPoint source_point_fan<kernel_mode>(
    std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity, ArithmeticMode mode);
template BoundedPoint bounded_source_point<kernel_mode>(
    const ConvexCellGeometry& source, std::span<const char> identity, ArithmeticMode mode);
template BoundedPoint bounded_source_point<kernel_mode>(
    std::span<const ConvexGeometryPtr> sources, std::span<const char> identity, ArithmeticMode mode);
template BoundedPoint bounded_point_displaced<kernel_mode>(
    const BoundedPoint& source, const BoundedPoint& displacement);
template BoundedPoint GeometryKernel<kernel_mode>::backtrace(const BoundedPoint& source, GeometryPoint velocity,
    double timestep, std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
} // namespace SIMPLEFLUID_GEOMETRY_KERNEL SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes::bounded_geometry_detail
