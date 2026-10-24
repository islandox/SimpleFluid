struct Plane
{
    Frame frame;
    Point normal, anchor;
    Interval offset;
    std::optional<GeometryHalfspace> input;
    std::vector<BoundedPlaneAuthority> authorities;
};
using PlanePtr = std::shared_ptr<const Plane>;
struct Vertex
{
    Point coordinates;
    std::vector<PlanePtr> incident;
    std::vector<BoundedPlaneAuthority> authorities;
};
using Face = std::vector<std::size_t>;
struct Split;
struct Ancestry
{
    std::shared_ptr<const Ancestry> parent;
    std::shared_ptr<const Split> split;
    bool outside = false;
};
struct Split
{
    std::shared_ptr<const Ancestry> parent;
};
struct Surface
{
    PlanePtr plane;
    int orientation = 1;
};
int matching_input(const GeometryHalfspace&, const GeometryHalfspace&);
Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
{
    ++statistics.support_distances;
    if (std::find(vertex.incident.begin(), vertex.incident.end(), plane) != vertex.incident.end())
        return Interval{};
    for (const auto& key : plane->authorities)
        if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) != vertex.authorities.end())
            return Interval{};
    if (plane->input)
        for (const auto& original : vertex.incident)
            if (original->input && matching_input(*original->input, *plane->input))
                return Interval{}; // The identical original expression, not a rounded plane fit.
    return dot(plane->normal, transform(vertex.coordinates, frame, plane->frame) - plane->anchor) - plane->offset;
}
Interval square_root(Interval x)
{
    if (x.lo < 0)
        reject("area square root is unresolved.");
    const auto lo = std::sqrt(x.lo), hi = std::sqrt(x.hi);
    finite(lo);
    finite(hi);
    if (x.hi == 0)
        return Interval{};
    if (lo == 0)
        reject("positive area below wider-FP range.");
    return {std::nextafter(lo, -infinity), std::nextafter(hi, infinity)};
}
