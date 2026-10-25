/** @file ordinary/ConvexTrajectories.tcc
 * @brief Ordinary source-point construction and constrained backtracing.
 */
namespace SimpleFluid::PLIC::details
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode>
DynamicPoint source_point_fan(std::span<const ConvexCellGeometry<ArithmeticMode::Ordinary>* const> sources,
    std::span<const char> identity, ArithmeticMode mode)
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
    const auto coordinates = numerical_point<ArithmeticMode::Ordinary>(identity);
    DynamicPoint result;
    result.arithmetic_mode = mode;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.value[axis] = coordinates[axis].value;
        result.absolute_error[axis] = 0;
    }
    std::vector<PlaneAuthority> supports;
    for (const auto* source : sources)
        for (std::size_t f = 0; f < source->num_faces(); ++f)
        {
            const auto face = source->face_geometry(f);
            if (!face->contains_point(identity))
                continue;
            const auto bytes = planar_support<ArithmeticMode::Ordinary>(*face);
            PlaneAuthority authority;
            authority.key.push_back(bytes.size());
            authority.key.resize(1 + (bytes.size() + 7) / 8, 0);
            for (std::size_t i = 0; i < bytes.size(); ++i)
                authority.key[1 + i / 8] |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                                            << (8 * (i % 8));
            active_work->charge(authority.key.size(), sizeof(std::uint64_t));
            supports.push_back(std::move(authority));
        }
    if (supports.empty())
        reject("source point has no authenticated source-face membership.");
    active_work->charge(1, sizeof(DynamicPointProvenance));
    active_work->charge(supports.capacity(), sizeof(PlaneAuthority));
    DynamicGeometryAccess::seal_represented_source(result, std::move(supports));
    return result;
}

template<ArithmeticMode Mode>
DynamicPoint dynamic_source_point(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, std::span<const char> identity, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    const std::array<const ConvexCellGeometry<ArithmeticMode::Ordinary>*, 1> sources{&source};
    return source_point_fan<Mode>(sources, identity, mode);
}

template<ArithmeticMode Mode>
DynamicPoint dynamic_source_point(std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources,
    std::span<const char> identity, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(mode));
    if (sources.size() > active_work->limits.maximum_faces)
        reject("source point fan exceeds bootstrap source-count cap.");
    active_work->charge(sources.size(), sizeof(const ConvexCellGeometry<ArithmeticMode::Ordinary>*));
    std::vector<const ConvexCellGeometry<ArithmeticMode::Ordinary>*> pointers;
    for (const auto& source : sources)
        pointers.push_back(source.get());
    return source_point_fan<Mode>(pointers, identity, mode);
}

template<ArithmeticMode Mode>
DynamicPoint dynamic_point_displaced(const DynamicPoint& source, const DynamicPoint& displacement)
{
    static_assert(Mode == kernel_mode);
    require_mode(source.arithmetic_mode, displacement.arithmetic_mode);
    Operation scope(mode_limits(source.arithmetic_mode));
    const auto delta = point<Mode>(displacement);
    if (std::all_of(delta.begin(), delta.end(), [](auto x) { return x.zero(); }))
        return source;
    auto result = bounded<Mode>(point<Mode>(source) + delta);
    DynamicGeometryAccess::seal(result, {});
    return result;
}

template<ArithmeticMode Mode>
DynamicPoint GeometryKernel<Mode>::backtrace(const DynamicPoint& source, GeometryPoint velocity, double timestep,
    std::span<const DynamicFaceConstraint> walls, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    require_mode(source.arithmetic_mode, limits.arithmetic_mode);
    Operation scope(limits);
    if (!std::isfinite(timestep) || timestep < 0 || walls.size() > 256)
        reject("invalid or excessive constrained trajectory inputs.");
    const auto input = point<Mode>(source);
    auto displacement = point<Mode>(velocity) * Scalar(timestep);
    const bool zero_motion = std::all_of(displacement.begin(), displacement.end(), [](auto x) { return x.zero(); });
    if (walls.empty())
    {
        if (zero_motion)
            return source;
        auto result = bounded<Mode>(input - displacement);
        DynamicGeometryAccess::seal(result, {});
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
            const auto lower = x.value > 0 ? x.value : x.value < 0 ? -x.value : 0;
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
            q[axis] = axis == pivot ? Scalar(1) : p[axis] / p[pivot];
        return q; // The pivot ratio is identically one, including its dependency.
    };
    std::vector<PlaneAuthority> retained;
    std::set<PlaneAuthority> independent_supports;
    for (const auto& wall : walls)
    {
        active_work->pair();
        if (!wall.geometry || wall.face >= checked_geometry_storage<Mode>(wall.geometry->impl_)->surfaces.size())
            reject("invalid wall support constraint.");
        require_mode(source.arithmetic_mode, wall.geometry->arithmetic_mode());
        const auto& plane = checked_geometry_storage<Mode>(wall.geometry->impl_)->surfaces[wall.face].plane;
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
            if (!(determinant.value > 0 || determinant.value < 0))
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
        projected = cross<Mode>(q, cross<Mode>(displacement, q)) * (Scalar(1) / dot<Mode>(q, q));
    }
    else if (rank == 2)
    {
        // Construct the common tangent directly. Subtracting nearly parallel
        // normal projections loses precision at small angles. A nonzero
        // cross-product pivot identifies rank two; its normalized direction
        // retains axial zeros.
        const auto q = direction(tangent);
        projected = q * (dot<Mode>(displacement, q) / dot<Mode>(q, q));
    }
    auto result = bounded<Mode>(input - projected);
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
    DynamicGeometryAccess::seal(result, std::move(retained));
    return result;
}

template DynamicPoint source_point_fan<kernel_mode>(
    std::span<const ConvexCellGeometry<ArithmeticMode::Ordinary>* const> sources, std::span<const char> identity,
    ArithmeticMode mode);
template DynamicPoint dynamic_source_point<kernel_mode>(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, std::span<const char> identity, ArithmeticMode mode);
template DynamicPoint dynamic_source_point<kernel_mode>(
    std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources, std::span<const char> identity,
    ArithmeticMode mode);
template DynamicPoint dynamic_point_displaced<kernel_mode>(
    const DynamicPoint& source, const DynamicPoint& displacement);
template DynamicPoint GeometryKernel<kernel_mode>::backtrace(const DynamicPoint& source, GeometryPoint velocity,
    double timestep, std::span<const DynamicFaceConstraint> walls, GeometryLimits limits);
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
