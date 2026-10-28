/** @file ConstructionIncidence.tcc
 * @brief Durable source-edge incidence with finite rank and identity evidence.
 */
namespace SimpleFluid::PLIC::details
{
namespace ordinary_kernel::source_witness
{
long double add(long double, long double);
long double add_up(long double, long double);
long double sum_gap(long double, long double, long double);
} // namespace ordinary_kernel::source_witness
namespace construction_incidence
{
using Identity = std::shared_ptr<const DynamicPointProvenance>;
using Edge = std::array<Identity, 2>;

template<class T> void retain(std::vector<T>& values, T value)
{
    if (values.size() == values.capacity())
    {
        if (values.capacity() > std::numeric_limits<std::size_t>::max() / 2)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Construction incidence capacity overflows.");
        const auto next = values.capacity() ? 2 * values.capacity() : 1;
        active_work->charge(next, sizeof(T));
        values.reserve(next);
    }
    values.push_back(std::move(value));
}
bool same_edge(const Edge& a, const Edge& b)
{
    active_work->pair();
    return (DynamicGeometryAccess::same_represented_source(a[0], b[0]) &&
               DynamicGeometryAccess::same_represented_source(a[1], b[1])) ||
           (DynamicGeometryAccess::same_represented_source(a[0], b[1]) &&
               DynamicGeometryAccess::same_represented_source(a[1], b[0]));
}
void retain_edge(std::vector<Edge>& result, const Edge& edge)
{
    for (const auto& existing : result)
        if (same_edge(existing, edge))
            return;
    retain(result, edge);
}

template<ArithmeticMode Mode> bool distinct(const Identity& first, const Identity& second)
{
    const auto& a = DynamicGeometryAccess::borrowed_original_source(first);
    const auto& b = DynamicGeometryAccess::borrowed_original_source(second);
    if (!a || !b || a->constructed || b->constructed || a->numerical_only || b->numerical_only ||
        a->arithmetic_mode != Mode || b->arithmetic_mode != Mode ||
        DynamicGeometryAccess::same_represented_source(a, b) || !certification_environment())
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(a->value[axis]) || !std::isfinite(b->value[axis]) ||
            !std::isfinite(a->absolute_error[axis]) || !std::isfinite(b->absolute_error[axis]) ||
            a->absolute_error[axis] < 0 || b->absolute_error[axis] < 0)
            return false;
        if constexpr (Mode == ArithmeticMode::Ordinary)
        {
            using namespace ordinary_kernel::source_witness;
            const auto difference = add(a->value[axis], -b->value[axis]);
            const auto error = add_up(add_up(a->absolute_error[axis], b->absolute_error[axis]),
                sum_gap(a->value[axis], -b->value[axis], difference));
            if (std::isfinite(difference) && std::isfinite(error) && std::abs(difference) > error)
                return true;
        }
        else
        {
            const auto difference =
                certification_kernel::enclosure<Mode>({a->value[axis], a->absolute_error[axis], Mode}) -
                certification_kernel::enclosure<Mode>({b->value[axis], b->absolute_error[axis], Mode});
            if (difference.lo() > 0 || difference.hi() < 0)
                return true;
        }
    }
    return false;
}

template<ArithmeticMode Mode> bool independent(const StoredPlane<Mode>& a, const StoredPlane<Mode>& b)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        // A cross-product component is the determinant against one coordinate
        // basis vector. Reuse the normal-construction-aware rank filter.
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            StoredPlane<Mode> basis;
            basis.normal[axis] = StoredValue<Mode>(1);
            basis.normal_error = 0;
            if (ordinary_kernel::independent_source_normals(a, b, basis))
                return true;
        }
        return false;
    }
    else
        return certification_kernel::nonzero<Mode>(certification_kernel::cross<Mode>(a.normal, b.normal));
}

template<ArithmeticMode Mode> bool directly_on(const Identity& token, const StoredPlane<Mode>& plane)
{
    if (!token || token->numerical_only || token->arithmetic_mode != Mode)
        return false;
    for (const auto& weak : plane.source_vertices)
    {
        active_work->pair();
        const auto owner = weak.lock();
        if (DynamicGeometryAccess::same_represented_source(token, owner))
            return true;
    }
    for (const auto* point = token.get(); point; point = point->source_origin.get())
        for (const auto& key : plane.authorities)
            if (!key.key.empty())
                for (const auto& support : point->supports)
                {
                    active_work->pair();
                    if (key == support)
                        return true;
                }
    return false;
}

template<ArithmeticMode Mode> bool point_on(const Identity& token, const StoredPlane<Mode>& plane)
{
    if (directly_on<Mode>(token, plane))
        return true;
    const auto& point = DynamicGeometryAccess::borrowed_original_source(token);
    if (!point || point->numerical_only || point->arithmetic_mode != Mode)
        return false;
    // Retained edge records already passed distinctness and rank admission;
    // their endpoints are original roots, so this does not recurse on incidence.
    for (const auto& edge : point->source_edges)
    {
        active_work->pair();
        if (directly_on<Mode>(edge[0], plane) && directly_on<Mode>(edge[1], plane))
            return true;
    }
    return false;
}

bool point_on_edge(const Identity& token, const Edge& edge)
{
    if (DynamicGeometryAccess::same_represented_source(token, edge[0]) ||
        DynamicGeometryAccess::same_represented_source(token, edge[1]))
        return true;
    const auto& point = DynamicGeometryAccess::borrowed_original_source(token);
    if (point && !point->numerical_only)
        for (const auto& other : point->source_edges)
            if (same_edge(other, edge))
                return true;
    return false;
}

template<ArithmeticMode Mode>
std::vector<Edge> infer_edges(const StoredVertex<Mode>& vertex, const StoredPlane<Mode>* query)
{
    if (vertex.incident.size() < 2)
        return {};
    if (query && query->source_vertices.empty() && query->authorities.empty())
        return {};
    std::vector<Identity> candidates;
    const auto include = [&](const StoredPlane<Mode>& plane)
    {
        for (const auto& weak : plane.source_vertices)
        {
            active_work->pair();
            const auto owner = weak.lock();
            const auto& point = DynamicGeometryAccess::borrowed_original_source(owner);
            if (!point || point->constructed || point->numerical_only || point->arithmetic_mode != Mode)
                continue;
            bool duplicate = false;
            for (const auto& previous : candidates)
            {
                active_work->pair();
                duplicate |= DynamicGeometryAccess::same_represented_source(previous, point);
            }
            if (!duplicate)
                retain(candidates, point);
        }
    };
    if (query)
        include(*query);
    // Without authority keys, only the query's own original tokens can prove
    // endpoint membership. Native keyed supports also admit endpoints retained
    // by the intersecting supports.
    if (!query || !query->authorities.empty())
        for (const auto& plane : vertex.incident)
            include(*plane);
    std::vector<Edge> result;
    for (std::size_t a = 0; a < candidates.size(); ++a)
        for (std::size_t b = a + 1; b < candidates.size(); ++b)
        {
            active_work->pair();
            const Edge edge{candidates[a], candidates[b]};
            if ((query && (!directly_on<Mode>(edge[0], *query) || !directly_on<Mode>(edge[1], *query))) ||
                !distinct<Mode>(edge[0], edge[1]))
                continue;
            bool proved = false;
            for (std::size_t i = 0; i < vertex.incident.size() && !proved; ++i)
            {
                const auto& first = *vertex.incident[i];
                if (!directly_on<Mode>(edge[0], first) || !directly_on<Mode>(edge[1], first))
                    continue;
                for (std::size_t j = i + 1; j < vertex.incident.size(); ++j)
                {
                    active_work->pair();
                    const auto& second = *vertex.incident[j];
                    if (directly_on<Mode>(edge[0], second) && directly_on<Mode>(edge[1], second) &&
                        independent<Mode>(first, second))
                    {
                        proved = true;
                        break;
                    }
                }
            }
            if (proved)
            {
                retain_edge(result, edge);
                if (query)
                    return result;
            }
        }
    return result;
}
} // namespace construction_incidence

template<ArithmeticMode Mode>
bool construction_edge_incident(const StoredVertex<Mode>& vertex, const StoredPlanePtr<Mode>& plane)
{
    using namespace construction_incidence;
    if (vertex.numerical_only)
        return false;
    if (vertex.source_identity && point_on<Mode>(vertex.source_identity, *plane))
        return true;
    return !infer_edges<Mode>(vertex, plane.get()).empty();
}

template<ArithmeticMode Mode>
void preserve_crossing_incidence(StoredVertex<Mode>& vertex, const StoredVertex<Mode>& a, const StoredVertex<Mode>& b,
    const Frame& frame, bool numerical_only)
{
    using namespace construction_incidence;
    if (vertex.source_identity)
        return;
    auto edges = numerical_only ? std::vector<Edge>{} : infer_edges<Mode>(vertex, nullptr);
    if (!numerical_only)
        if (const auto& first = DynamicGeometryAccess::borrowed_original_source(a.source_identity))
            for (const auto& edge : first->source_edges)
                if (point_on_edge(b.source_identity, edge))
                    retain_edge(edges, edge);
    DynamicPoint point;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        using namespace ordinary_kernel;
        point = bounded<Mode>(
            scaled<Mode>(vertex.coordinates, frame.exponent) + ordinary_kernel::point<Mode>(frame.origin));
    }
    else
    {
        using namespace certification_kernel;
        point = bounded<Mode>(
            scaled<Mode>(vertex.coordinates, frame.exponent) + certification_kernel::point<Mode>(frame.origin));
    }
    DynamicGeometryAccess::seal_generated(point, std::move(edges), vertex.authorities, numerical_only);
    active_work->charge(DynamicGeometryAccess::identity_bytes(DynamicGeometryAccess::identity(point)), 1);
    vertex.source_identity = DynamicGeometryAccess::identity(point);
}

template bool construction_edge_incident<ArithmeticMode::Ordinary>(
    const StoredVertex<ArithmeticMode::Ordinary>&, const StoredPlanePtr<ArithmeticMode::Ordinary>&);
template bool construction_edge_incident<ArithmeticMode::BoundedCertification>(
    const StoredVertex<ArithmeticMode::BoundedCertification>&,
    const StoredPlanePtr<ArithmeticMode::BoundedCertification>&);
template void preserve_crossing_incidence<ArithmeticMode::Ordinary>(StoredVertex<ArithmeticMode::Ordinary>&,
    const StoredVertex<ArithmeticMode::Ordinary>&, const StoredVertex<ArithmeticMode::Ordinary>&, const Frame&, bool);
template void preserve_crossing_incidence<ArithmeticMode::BoundedCertification>(
    StoredVertex<ArithmeticMode::BoundedCertification>&, const StoredVertex<ArithmeticMode::BoundedCertification>&,
    const StoredVertex<ArithmeticMode::BoundedCertification>&, const Frame&, bool);
} // namespace SimpleFluid::PLIC::details
