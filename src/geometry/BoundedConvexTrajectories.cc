#include "geometry/details/BoundedConvexGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
BoundedPoint bounded_source_point(const ConvexCellGeometry& source, std::span<const char> identity)
{
    const std::array<const ConvexCellGeometry*, 1> sources{&source};
    return source_point_fan(sources, identity);
}

BoundedPoint bounded_source_point(std::span<const ConvexGeometryPtr> sources, std::span<const char> identity)
{
    Operation scope({});
    if (sources.size() > active_work->limits.maximum_faces)
        reject("source point fan exceeds bootstrap source-count cap.");
    active_work->charge(sources.size(), sizeof(const ConvexCellGeometry*));
    std::vector<const ConvexCellGeometry*> pointers;
    for (const auto& source : sources)
        pointers.push_back(source.get());
    return source_point_fan(pointers, identity);
}

BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement)
{
    Operation scope({});
    const auto delta = point(displacement);
    if (std::all_of(delta.begin(), delta.end(), [](auto x) { return x.zero(); }))
        return source;
    return bounded(point(source) + delta);
}

BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    if (!std::isfinite(timestep) || timestep < 0 || walls.size() > 256)
        reject("invalid or excessive constrained trajectory inputs.");
    const auto input = point(source);
    auto displacement = point(velocity) * Interval(timestep);
    const bool zero_motion = std::all_of(displacement.begin(), displacement.end(), [](auto x) { return x.zero(); });
    if (walls.empty())
    {
        if (zero_motion)
            return source;
        return bounded(input - displacement);
    }
    std::vector<Point> basis;
    std::vector<Interval> squared;
    std::vector<BoundedPlaneAuthority> retained;
    std::set<BoundedPlaneAuthority> independent_supports;
    for (const auto& wall : walls)
    {
        active_work->pair();
        if (!wall.geometry || wall.face >= wall.geometry->impl_->surfaces.size())
            reject("invalid wall support constraint.");
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
            const auto local = scaled(input - point(plane->frame.origin), -plane->frame.exponent);
            if (!dot(plane->normal, local - plane->anchor).zero())
                reject("wall constraint lacks original point/support incidence.");
        }
        // Every catalog entry is authenticated above, including duplicates.
        // A collision-free original support identity proves equal planes even
        // when independently scaled normals have different numerical bounds.
        const bool repeated_support = std::any_of(plane->authorities.begin(), plane->authorities.end(),
            [&](const auto& key) { return independent_supports.contains(key); });
        independent_supports.insert(plane->authorities.begin(), plane->authorities.end());
        if (zero_motion || repeated_support || basis.size() == 3)
            continue; // The proven rank-three tangent space already is {0}.
        auto normal = plane->normal;
        for (std::size_t j = 0; j < basis.size(); ++j)
            normal = normal - basis[j] * (dot(normal, basis[j]) / squared[j]);
        const auto norm = dot(normal, normal);
        if (norm.zero())
            continue;
        if (!(norm.lo > 0) || basis.size() == 3)
            reject("wall constraint rank/conditioning unresolved.");
        basis.push_back(normal);
        squared.push_back(norm);
    }
    if (zero_motion)
        return source; // All catalog incidences were validated; no projection is needed.
    if (basis.size() == 3)
    {
        // A proven rank-three tangent space is exactly {0}; retaining the
        // original point is the structural construction, not a rounded snap.
        return source; // Preserve original construction identity as well as its coordinates.
    }
    auto projected = displacement;
    for (std::size_t j = 0; j < basis.size(); ++j)
        projected = projected - basis[j] * (dot(displacement, basis[j]) / squared[j]);
    // Verify the enclosure of every original wall residual before attaching
    // its incidence. The downstream geometry/CFL budget consumes the full
    // propagated coordinate error; this does not introduce a wall tolerance.
    for (const auto& wall : walls)
    {
        const auto& normal = wall.geometry->impl_->surfaces[wall.face].plane->normal;
        const auto residual = dot(normal, projected);
        if (residual.lo > 0 || residual.hi < 0)
            reject("projected trajectory does not enclose original wall tangency.");
    }
    auto result = bounded(input - projected);
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
    BoundedGeometryAccess::seal(result, std::move(retained));
    return result;
}
} // namespace SimpleFluid::Meshes
