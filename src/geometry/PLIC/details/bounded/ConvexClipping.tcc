/** Compiled certification geometry kernels. Included only by the owning implementation unit.
 * Directed interval arithmetic and retained certificates remain confined to this
 * provider; Mode templates retain the compiled implementation boundary.
 */
namespace SimpleFluid::PLIC::details
{
namespace certification_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b)
{
    static_assert(Mode == kernel_mode);
    if (a.anchor != b.anchor)
        return 0;
    bool equal = a.offset == b.offset, reversed = a.offset == -b.offset;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        equal = equal && a.normal[axis] == b.normal[axis];
        reversed = reversed && a.normal[axis] == -b.normal[axis];
    }
    return equal ? 1 : reversed ? -1 : 0;
}

template<ArithmeticMode Mode> bool incident(const Vertex& v, const PlanePtr& plane)
{
    static_assert(Mode == kernel_mode);
    if (std::find(v.incident.begin(), v.incident.end(), plane) != v.incident.end())
        return true;
    if (v.source_identity)
    for (const auto& weak : plane->source_vertices)
    {
        active_work->pair();
        const auto identity = weak.lock();
        if (DynamicGeometryAccess::same_represented_source(identity, v.source_identity))
            return true;
    }
    return construction_edge_incident<Mode>(v, plane);
}

template<ArithmeticMode Mode>
Interval refine_triangle_distance(const Point& coordinates, const Frame& frame, const Vertex& a, const Vertex& b,
    const Vertex& c, const Frame& triangle_frame, const PlanePtr& plane, int orientation, Interval result)
{
    static_assert(Mode == kernel_mode);
    active_work->pair();
    if (!incident<Mode>(a, plane) || !incident<Mode>(b, plane) || !incident<Mode>(c, plane))
        return result;
    auto w = transform<Mode>(coordinates, frame, triangle_frame) - a.coordinates;
    for (std::size_t k = 0; k < 3; ++k)
        if (plane->normal[k].zero())
            w[k] = Interval{}; // This coordinate cannot affect the original equation.
    const auto u = b.coordinates - a.coordinates, v = c.coordinates - a.coordinates;
    const auto normal = cross<Mode>(u, v);
    std::size_t axis = 3;
    long double largest = 0;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const auto n = normal[k], original = plane->normal[k];
        const auto strength = n.lo() > 0 ? n.lo() : n.hi() < 0 ? -n.hi() : 0;
        if (strength > largest && (original.lo() > 0 || original.hi() < 0))
        {
            axis = k;
            largest = strength;
        }
    }
    if (axis == 3)
        return result;
    const std::array forms{dot<Mode>(normal, w), dot<Mode>(u, cross<Mode>(v, w)), dot<Mode>(v, cross<Mode>(w, u))};
    auto determinant = forms.front();
    for (const auto& form : forms)
    {
        const auto lo = std::max(determinant.lo(), form.lo()), hi = std::min(determinant.hi(), form.hi());
        if (lo > hi)
            reject("equivalent source-face determinants are disjoint.");
        determinant = Interval(lo, hi);
    }
    // The admitted triangle lies on the original plane. Restore its signed
    // normal scale and frame units before using distances for edge fractions.
    const auto refined = scaled<Mode>(determinant * (plane->normal[axis] / normal[axis]),
                             triangle_frame.exponent - plane->frame.exponent) *
                         Interval(orientation);
    const auto lo = std::max(result.lo(), refined.lo()), hi = std::min(result.hi(), refined.hi());
    if (lo > hi)
        reject("source-face support refinement has inconsistent distance enclosures.");
    return Interval(lo, hi);
}

template<ArithmeticMode Mode>
DynamicGeometrySection section(
    const StoredGeometry& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop)
{
    static_assert(Mode == kernel_mode);
    DynamicGeometrySection result;
    result.area = bounded<Mode>(Interval{});
    result.centroid = bounded<Mode>(Point{});
    for (const auto i : loop)
        result.vertices.push_back(
            bounded<Mode>(scaled<Mode>(vertices[i].coordinates, poly.frame.exponent) + point<Mode>(poly.frame.origin)));
    if (loop.size() < 3)
        return result;
    const auto& a = vertices[loop[0]].coordinates;
    Point area_vector;
    for (std::size_t i = 1; i + 1 < loop.size(); ++i)
        area_vector =
            area_vector + cross<Mode>(vertices[loop[i]].coordinates - a, vertices[loop[i + 1]].coordinates - a);
    area_vector = area_vector * Interval(.5L);
    const auto squared_area = dot<Mode>(area_vector, area_vector);
    if (squared_area.zero())
        return result; // A proven line/point contact retains its vertices.
    result.area = bounded<Mode>(scaled<Mode>(square_root<Mode>(squared_area), 2 * poly.frame.exponent));
    Point weighted;
    Interval total_weight;
    for (std::size_t i = 1; i + 1 < loop.size(); ++i)
    {
        const auto& b = vertices[loop[i]].coordinates;
        const auto& c = vertices[loop[i + 1]].coordinates;
        const auto weight = dot<Mode>(cross<Mode>(b - a, c - a), area_vector);
        total_weight = total_weight + weight;
        weighted = weighted + (a + b + c) * (weight / Interval(3));
    }
    result.centroid = bounded<Mode>(
        scaled<Mode>(weighted * (Interval(1) / total_weight), poly.frame.exponent) + point<Mode>(poly.frame.origin));
    return result;
}

template<ArithmeticMode Mode>
DynamicClipResult GeometryKernel<Mode>::clip(const GeometryHalfspace& input, bool measure_section) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (!nonzero<Mode>(point<Mode>(input.normal)))
        reject("clipping requires a nonzero plane normal.");
    // Reuse the original, oriented source expression, never a comparison of
    // rounded vertex coordinates with a geometrically nearby plane.
    for (const auto& vertex : impl_->vertices)
        for (const auto& existing : vertex.incident)
            if (existing->input)
                if (const int orientation = matching_input<Mode>(input, *existing->input))
                {
                    auto source = std::make_shared<Impl>();
                    source->limits = impl_->limits;
                    source->surfaces.push_back({existing, orientation});
                    return clip_impl(source, 0, measure_section);
                }
    // A numerical rebase is not physical motion. Subtract the original frame
    // anchors directly before adding local coordinates; a world-coordinate
    // round trip would irreversibly lose the minority layer's local scale.
    // Retain the public moment origin, support incidence, split ancestry and
    // valid volume cache. First moments belong to the new numerical frame.
    if (impl_->frame.origin != input.anchor)
    {
        active_work->charge(storage_bytes(), 1);
        auto rebased = std::make_shared<Impl>(*impl_);
        rebased->frame.origin = input.anchor;
        active_work->charge(1, sizeof(GeometryConstruction));
        auto recipe = std::make_shared<GeometryConstruction>();
        recipe->kind = GeometryConstruction::Kind::Rebase;
        recipe->parent = construction_source<Mode>(impl_);
        recipe->rebase_input = input;
        rebased->construction = std::move(recipe);
        rebased->moment_cache = std::make_shared<MomentCache>();
        for (auto& vertex : rebased->vertices)
            vertex.coordinates = transform<Mode>(vertex.coordinates, impl_->frame, rebased->frame);
        rebased->bounds = transform<Mode>(impl_->bounds, impl_->frame, rebased->frame);
        ++statistics.constructions;
        return GeometryKernel<Mode>(std::move(rebased)).clip(input, measure_section);
    }
    auto plane = std::make_shared<Plane>();
    plane->frame = impl_->frame;
    plane->normal = point<Mode>(input.normal);
    plane->anchor = scaled<Mode>(point<Mode>(input.anchor) - point<Mode>(impl_->frame.origin), -impl_->frame.exponent);
    plane->offset = scaled<Mode>(Interval(input.offset), -impl_->frame.exponent);
    plane->input = input;
    auto source = std::make_shared<Impl>();
    source->limits = impl_->limits;
    source->surfaces.push_back({plane, 1});
    return clip_impl(source, 0, measure_section);
}

template<ArithmeticMode Mode>
DynamicClipResult GeometryKernel<Mode>::clip_by_face(
    const DynamicConvexGeometry& source, std::size_t face, bool measure_section) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    require_mode(arithmetic_mode(), source.arithmetic_mode());
    if (face >= source.impl_->faces.size())
        reject("invalid source face index.");
    return clip_impl(source.impl_, face, measure_section);
}

template<ArithmeticMode Mode>
DynamicClipResult GeometryKernel<Mode>::clip_impl(const std::shared_ptr<const DynamicConvexGeometry::Impl>& source_base,
    std::size_t plane_id, bool measure_section, bool construct_outside, bool partition_proof, int orientation_override) const
{
    static_assert(Mode == kernel_mode);
    const auto source = checked_geometry_storage<Mode>(source_base);
    require_mode(impl_->limits.arithmetic_mode, source->limits.arithmetic_mode);
    if ((impl_->partition_proof_only && !partition_proof) ||
        (partition_proof && (measure_section || !construct_outside)))
        reject("invalid private partition witness clip mode.");
    auto support = source->surfaces.at(plane_id);
    if (orientation_override)
    {
        if (orientation_override != 1 && orientation_override != -1)
            reject("invalid construction replay orientation.");
        support.orientation = orientation_override;
    }
    const auto& plane = support.plane;
    const auto refine_face_distance = [&](const Point& coordinates, const Frame& frame, Interval result)
    {
        const auto& face = source->faces[plane_id];
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            result =
                refine_triangle_distance<Mode>(coordinates, frame, source->vertices[face[0]], source->vertices[face[i]],
                    source->vertices[face[i + 1]], source->frame, plane, support.orientation, result);
            if (result.lo() > 0 || result.hi() < 0 || result.zero())
                break;
        }
        return result;
    };
    const auto original_point = [&](const std::shared_ptr<const DynamicPointProvenance>& identity)
    {
        Vertex result;
        result.coordinates = transform<Mode>(
            point<Mode>(DynamicPoint{identity->value, identity->absolute_error}), Frame{}, source->frame);
        result.source_identity = identity;
        active_work->charge(identity->supports.size(), sizeof(PlaneAuthority));
        for (const auto& key : identity->supports)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
        result.authorities = identity->supports;
        return result;
    };
    const auto retained_endpoint_distance = [&](const Vertex& endpoint, const PlanePtr& original)
    {
        auto result = distance<Mode>(endpoint, source->frame, original);
        if (result.lo() > 0 || result.hi() < 0 || result.zero())
            return result;
        active_work->charge(original->source_vertices.size(), sizeof(Vertex));
        std::vector<Vertex> defining;
        defining.reserve(original->source_vertices.size());
        for (const auto& weak : original->source_vertices)
        {
            active_work->pair();
            if (const auto identity = weak.lock())
                defining.push_back(original_point(identity));
        }
        for (std::size_t i = 1; i + 1 < defining.size(); ++i)
        {
            result = refine_triangle_distance<Mode>(endpoint.coordinates, source->frame, defining[0], defining[i],
                defining[i + 1], source->frame, original, 1, result);
            if (result.lo() > 0 || result.hi() < 0 || result.zero())
                break;
        }
        return result; // Deliberately no affine or clipping recursion.
    };
    const auto endpoint_distance = [&](const Vertex& endpoint)
    {
        auto result = distance<Mode>(endpoint, source->frame, plane) * Interval(support.orientation);
        if (result.lo() <= 0 && result.hi() >= 0 && !result.zero())
            result = refine_face_distance(endpoint.coordinates, source->frame, result);
        return result;
    };
    std::vector<Interval> distances;
    std::vector<int> signs;
    bool negative = false, positive = false;
    for (const auto& vertex : impl_->vertices)
    {
        auto d = distance<Mode>(vertex, impl_->frame, plane) * Interval(support.orientation);
        if (d.lo() <= 0 && d.hi() >= 0 && !d.zero() && plane_id < source->faces.size())
        {
            // Three noncollinear shared points determine the same affine
            // plane, even when separate swept pieces use opposite windings.
            // This proves zero distance only, not either signed halfspace.
            const auto& face = source->faces[plane_id];
            const auto& a = source->vertices[face[0]];
            for (const auto& original : vertex.incident)
            {
                active_work->pair();
                if (!incident<Mode>(a, plane) ||
                    !(incident<Mode>(a, original) || raw_distance<Mode>(a, source->frame, original).zero()))
                    continue;
                for (std::size_t i = 1; i + 1 < face.size() && !d.zero(); ++i)
                {
                    active_work->pair();
                    const auto& b = source->vertices[face[i]];
                    const auto& c = source->vertices[face[i + 1]];
                    if (incident<Mode>(b, plane) && incident<Mode>(c, plane) &&
                        (incident<Mode>(b, original) || raw_distance<Mode>(b, source->frame, original).zero()) &&
                        (incident<Mode>(c, original) || raw_distance<Mode>(c, source->frame, original).zero()) &&
                        nonzero<Mode>(cross<Mode>(b.coordinates - a.coordinates, c.coordinates - a.coordinates)))
                        d = Interval{};
                }
                if (d.zero())
                    break;
            }
        }
        if (d.lo() <= 0 && d.hi() >= 0 && !d.zero() && plane_id < source->faces.size())
        {
            // A reconstructed crossing can be an original source vertex even
            // when its independent coordinate enclosure straddles this plane.
            // Three shared independent support equations prove that identity;
            // coordinate proximity or an uncertain residual supplies no proof.
            for (const auto index : source->faces[plane_id])
            {
                active_work->pair();
                const auto& candidate = source->vertices[index];
                if (!incident<Mode>(candidate, plane))
                    continue;
                active_work->charge(vertex.incident.size(), sizeof(PlanePtr));
                std::vector<PlanePtr> common;
                common.reserve(vertex.incident.size());
                for (const auto& original : vertex.incident)
                    if (incident<Mode>(candidate, original) ||
                        raw_distance<Mode>(candidate, source->frame, original).zero())
                        common.push_back(original);
                bool identical = false;
                for (std::size_t a = 0; a < common.size() && !identical; ++a)
                    for (std::size_t b = a + 1; b < common.size() && !identical; ++b)
                        for (std::size_t c = b + 1; c < common.size() && !identical; ++c)
                        {
                            active_work->pair();
                            const auto determinant =
                                dot<Mode>(common[a]->normal, cross<Mode>(common[b]->normal, common[c]->normal));
                            identical = determinant.lo() > 0 || determinant.hi() < 0;
                        }
                if (identical)
                {
                    const auto coordinates = transform<Mode>(candidate.coordinates, source->frame, impl_->frame);
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        if (coordinates[axis].hi() < vertex.coordinates[axis].lo() ||
                            vertex.coordinates[axis].hi() < coordinates[axis].lo())
                            reject("shared-support point identity has inconsistent coordinate enclosures.");
                    d = Interval{};
                    break; // Existing zero-sign propagation retains this incidence.
                }
            }
        }
        if (d.lo() <= 0 && d.hi() >= 0 && !d.zero() && plane_id < source->faces.size())
        {
            // Two independent target supports define a line. Two distinct
            // source-face points on both supports prove that the query plane
            // contains that entire line, including points beyond the edge.
            const auto& face = source->faces[plane_id];
            for (std::size_t i = 0; i < face.size() && !d.zero(); ++i)
            {
                active_work->pair();
                const auto& a = source->vertices[face[i]];
                const auto& b = source->vertices[face[(i + 1) % face.size()]];
                if (!incident<Mode>(a, plane) || !incident<Mode>(b, plane) ||
                    !nonzero<Mode>(a.coordinates - b.coordinates))
                    continue;
                active_work->charge(vertex.incident.size(), sizeof(PlanePtr));
                std::vector<PlanePtr> common;
                common.reserve(vertex.incident.size());
                for (const auto& original : vertex.incident)
                    if ((incident<Mode>(a, original) || raw_distance<Mode>(a, source->frame, original).zero()) &&
                        (incident<Mode>(b, original) || raw_distance<Mode>(b, source->frame, original).zero()))
                        common.push_back(original);
                for (std::size_t first = 0; first < common.size() && !d.zero(); ++first)
                    for (std::size_t second = first + 1; second < common.size() && !d.zero(); ++second)
                    {
                        active_work->pair();
                        if (nonzero<Mode>(cross<Mode>(common[first]->normal, common[second]->normal)))
                            d = Interval{};
                    }
            }
        }
        if (d.lo() <= 0 && d.hi() >= 0 && !d.zero() && plane_id < source->faces.size())
            d = refine_face_distance(vertex.coordinates, impl_->frame, d);
        if (d.lo() <= 0 && d.hi() >= 0 && !d.zero() && plane_id < source->faces.size())
        {
            // P and Q share an edge (a,b). On P, x-a = s*(b-a)+t*(p-a),
            // so Q(x)=t*Q(p). Retain this affine dependency instead of
            // subtracting independently enclosed nearly parallel normals.
            const auto& face = source->faces[plane_id];
            const auto x = transform<Mode>(vertex.coordinates, impl_->frame, source->frame);
            for (const auto& original : vertex.incident)
            {
                for (std::size_t i = 0; i < face.size(); ++i)
                {
                    active_work->pair();
                    const auto& a = source->vertices[face[i]];
                    const auto& b = source->vertices[face[(i + 1) % face.size()]];
                    if (!incident<Mode>(a, plane) || !incident<Mode>(b, plane) ||
                        !(incident<Mode>(a, original) || raw_distance<Mode>(a, source->frame, original).zero()) ||
                        !(incident<Mode>(b, original) || raw_distance<Mode>(b, source->frame, original).zero()))
                        continue;
                    const auto edge = b.coordinates - a.coordinates;
                    for (const auto& weak : original->source_vertices)
                    {
                        active_work->pair();
                        const auto identity = weak.lock();
                        if (!identity)
                            continue;
                        // The immutable global snapshot was validated before
                        // import, and P issued its incidence after planarity.
                        const auto p = original_point(identity);
                        const auto normal = cross<Mode>(edge, p.coordinates - a.coordinates);
                        std::size_t axis = 3;
                        long double largest = 0;
                        for (std::size_t k = 0; k < 3; ++k)
                        {
                            const auto n = normal[k];
                            const auto strength = n.lo() > 0 ? n.lo() : n.hi() < 0 ? -n.hi() : 0;
                            if (strength > largest)
                            {
                                axis = k;
                                largest = strength;
                            }
                        }
                        if (axis == 3)
                            continue;
                        auto t = cross<Mode>(edge, x - a.coordinates)[axis] / normal[axis];
                        const auto residual = endpoint_distance(p);
                        auto refined = t * residual;
                        if (refined.lo() <= 0 && refined.hi() >= 0 && !refined.zero())
                        {
                            // On this retained plane, x=a+s*(b-a)+t*(p-a).
                            // Other authenticated supports give two linear
                            // equations for s,t, without using rounded x.
                            struct Row
                            {
                                Interval at_anchor, along_edge, transverse;
                            };
                            active_work->charge(vertex.incident.size(), sizeof(Row));
                            std::vector<Row> rows;
                            rows.reserve(vertex.incident.size());
                            for (const auto& constraint : vertex.incident)
                            {
                                if (constraint == original)
                                    continue;
                                active_work->pair();
                                const auto at_a = retained_endpoint_distance(a, constraint);
                                const auto at_b = retained_endpoint_distance(b, constraint);
                                const auto at_p = retained_endpoint_distance(p, constraint);
                                rows.push_back({at_a, at_b - at_a, at_p - at_a});
                            }
                            for (std::size_t first = 0; first < rows.size(); ++first)
                            {
                                for (std::size_t second = first + 1; second < rows.size(); ++second)
                                {
                                    active_work->pair();
                                    const auto& u = rows[first];
                                    const auto& v = rows[second];
                                    const auto determinant = u.along_edge * v.transverse - u.transverse * v.along_edge;
                                    if (!(determinant.lo() > 0 || determinant.hi() < 0))
                                        continue;
                                    const auto solved =
                                        (u.at_anchor * v.along_edge - u.along_edge * v.at_anchor) / determinant;
                                    const auto lo = std::max(t.lo(), solved.lo()), hi = std::min(t.hi(), solved.hi());
                                    if (lo > hi)
                                        reject("retained-plane affine coordinates have inconsistent enclosures.");
                                    t = Interval(lo, hi);
                                    refined = t * residual;
                                    if (refined.lo() > 0 || refined.hi() < 0 || refined.zero())
                                        break;
                                }
                                if (refined.lo() > 0 || refined.hi() < 0 || refined.zero())
                                    break;
                            }
                        }
                        const auto lo = std::max(d.lo(), refined.lo()), hi = std::min(d.hi(), refined.hi());
                        if (lo > hi)
                            reject("shared-edge affine support has inconsistent distance enclosures.");
                        d = Interval(lo, hi);
                        if (d.lo() > 0 || d.hi() < 0 || d.zero())
                            break;
                    }
                    if (d.lo() > 0 || d.hi() < 0 || d.zero())
                        break;
                }
                if (d.lo() > 0 || d.hi() < 0 || d.zero())
                    break;
            }
        }
        if (d.lo() <= 0 && d.hi() >= 0 && !d.zero() && plane_id < source->faces.size())
        {
            // Two independent retained supports define a line through their
            // original points a,b. Evaluate Q along that line, including its
            // extensions, instead of losing the x/y coordinate dependency.
            const auto x = transform<Mode>(vertex.coordinates, impl_->frame, source->frame);
            for (const auto& first : vertex.incident)
            {
                if (first->source_vertices.size() < 2)
                    continue;
                for (const auto& second : vertex.incident)
                {
                    if (first == second)
                        continue;
                    active_work->pair();
                    if (!nonzero<Mode>(cross<Mode>(first->normal, second->normal)))
                        continue;
                    for (std::size_t i = 0; i < first->source_vertices.size(); ++i)
                    {
                        active_work->pair();
                        const auto first_identity = first->source_vertices[i].lock();
                        if (!first_identity)
                            continue;
                        const auto a = original_point(first_identity);
                        if (!distance<Mode>(a, source->frame, second).zero())
                            continue;
                        for (std::size_t j = i + 1; j < first->source_vertices.size(); ++j)
                        {
                            active_work->pair();
                            const auto second_identity = first->source_vertices[j].lock();
                            if (!second_identity)
                                continue;
                            const auto b = original_point(second_identity);
                            if (!distance<Mode>(b, source->frame, second).zero())
                                continue;
                            const auto edge = b.coordinates - a.coordinates;
                            std::size_t axis = 3;
                            long double largest = 0;
                            for (std::size_t k = 0; k < 3; ++k)
                            {
                                const auto e = edge[k];
                                const auto strength = e.lo() > 0 ? e.lo() : e.hi() < 0 ? -e.hi() : 0;
                                if (strength > largest)
                                {
                                    axis = k;
                                    largest = strength;
                                }
                            }
                            if (axis == 3)
                                continue;
                            const auto first_weight = (b.coordinates[axis] - x[axis]) / edge[axis];
                            const auto second_weight = (x[axis] - a.coordinates[axis]) / edge[axis];
                            const auto refined =
                                first_weight * endpoint_distance(a) + second_weight * endpoint_distance(b);
                            const auto lo = std::max(d.lo(), refined.lo()), hi = std::min(d.hi(), refined.hi());
                            if (lo > hi)
                                reject("retained-line support has inconsistent distance enclosures.");
                            d = Interval(lo, hi);
                            if (d.lo() > 0 || d.hi() < 0 || d.zero())
                                break;
                        }
                        if (d.lo() > 0 || d.hi() < 0 || d.zero())
                            break;
                    }
                    if (d.lo() > 0 || d.hi() < 0 || d.zero())
                        break;
                }
                if (d.lo() > 0 || d.hi() < 0 || d.zero())
                    break;
            }
        }
        const auto s = sign<Mode>(d);
        distances.push_back(d);
        signs.push_back(s);
        negative = negative || s < 0;
        positive = positive || s > 0;
    }
    DynamicClipResult result;
    result.section.area = bounded<Mode>(Interval{});
    result.section.centroid = bounded<Mode>(Point{});
    if (!negative || !positive)
    {
        // A support face/edge/point is a proven zero-volume contact. Preserve
        // its section dimension rather than manufacturing a positive piece.
        std::vector<std::size_t> contact;
        std::map<std::pair<std::size_t, std::size_t>, int> contact_edges;
        for (const auto& face : impl_->faces)
            if (std::all_of(face.begin(), face.end(), [&](auto i) { return signs[i] == 0; }))
                for (std::size_t i = 0; i < face.size(); ++i)
                {
                    const auto a = face[i], b = face[(i + 1) % face.size()];
                    contact_edges[std::minmax(a, b)] += a < b ? 1 : -1;
                }
        std::map<std::size_t, std::set<std::size_t>> boundary;
        for (const auto& [edge, coefficient] : contact_edges)
            if (coefficient)
            {
                if (std::abs(coefficient) != 1)
                    reject("support section has repeated oriented boundary coverage.");
                boundary[edge.first].insert(edge.second);
                boundary[edge.second].insert(edge.first);
            }
        if (!boundary.empty())
        {
            for (const auto& [vertex, neighbors] : boundary)
                if (neighbors.size() != 2)
                    reject("support section is not a closed boundary.");
            const auto first = boundary.begin()->first;
            auto previous = first, current = *boundary.at(first).begin();
            contact.push_back(first);
            while (current != first)
            {
                if (contact.size() >= boundary.size())
                    reject("support section boundary repeats a vertex.");
                contact.push_back(current);
                const auto& next = boundary.at(current);
                const auto following = *next.begin() == previous ? *next.rbegin() : *next.begin();
                previous = current;
                current = following;
            }
            if (contact.size() != boundary.size())
                reject("support section has disconnected atomic boundaries.");
        }
        if (contact.empty())
            for (std::size_t i = 0; i < signs.size(); ++i)
                if (signs[i] == 0)
                    contact.push_back(i);
        if (measure_section)
            result.section = section<Mode>(*impl_, impl_->vertices, contact);
        active_work->charge(1, sizeof(DynamicConvexGeometry));
        auto self = DynamicConvexGeometryPtr(new DynamicConvexGeometry(impl_));
        if (!positive)
            result.inside = self;
        if (!negative && construct_outside)
            result.outside = self;
        // A valid positive-volume shell cannot lie entirely on this plane.
        if (!negative && !positive)
            reject("positive-volume shell has entirely unresolved support.");
        return result;
    }

    std::vector<Vertex> vertices = impl_->vertices;
    // Authenticate every zero-distance input incidence on the new shared plane.
    for (std::size_t i = 0; i < signs.size(); ++i)
        if (signs[i] == 0)
        {
            // Materialize the proved relation even when a local point token
            // already establishes it; numerical packets retain plane pointers
            // as indexed incidences, but deliberately omit those local tokens.
            if (std::find(vertices[i].incident.begin(), vertices[i].incident.end(), plane) ==
                vertices[i].incident.end())
                vertices[i].incident.push_back(plane);
            for (const auto& key : plane->authorities)
                if (std::find(vertices[i].authorities.begin(), vertices[i].authorities.end(), key) ==
                    vertices[i].authorities.end())
                    vertices[i].authorities.push_back(key);
        }
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> crossings;
    const auto crossing = [&](std::size_t a, std::size_t b)
    {
        const std::pair<std::size_t, std::size_t> key{std::min(a, b), std::max(a, b)};
        if (auto found = crossings.find(key); found != crossings.end())
            return found->second;
        a = key.first;
        b = key.second;
        // Strict opposite endpoint distances prove a unique intersection in
        // the open edge. An independently enclosed da/(da-db) loses the shared
        // denominator dependency near an endpoint and need not exclude 0/1.
        if (!((distances[a].lo() > 0 && distances[b].hi() < 0) || (distances[b].lo() > 0 && distances[a].hi() < 0)))
            reject("edge intersection lacks certified opposite endpoint signs.");
        Vertex vertex;
        vertex.numerical_only = impl_->numerical_only || source->numerical_only;
        const auto denominator = distances[a] - distances[b];
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            // Delay the common division. A nonbinary edge parameter can
            // produce an exactly representable coordinate; rounding the
            // parameter first would destroy that cancellation and its support.
            vertex.coordinates[axis] =
                (vertices[b].coordinates[axis] * distances[a] - vertices[a].coordinates[axis] * distances[b]) /
                denominator;
        }
        for (const auto& p : vertices[a].incident)
            if (incident<Mode>(vertices[b], p))
                vertex.incident.push_back(p);
        if (std::find(vertex.incident.begin(), vertex.incident.end(), plane) == vertex.incident.end())
            vertex.incident.push_back(plane);
        for (const auto& key : vertices[a].authorities)
            if (std::find(vertices[b].authorities.begin(), vertices[b].authorities.end(), key) !=
                vertices[b].authorities.end())
                vertex.authorities.push_back(key);
        for (const auto& key : plane->authorities)
            if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) == vertex.authorities.end())
                vertex.authorities.push_back(key);
        tighten_incident_coordinates<Mode>(vertex, impl_->frame);
        preserve_crossing_incidence<Mode>(vertex, vertices[a], vertices[b], impl_->frame,
            impl_->numerical_only || source->numerical_only);
        if (vertices.size() >= impl_->limits.maximum_vertices)
            reject("intersection vertex cap exhausted.");
        const auto index = vertices.size();
        vertices.push_back(std::move(vertex));
        crossings.emplace(key, index);
        return index;
    };
    std::array<std::vector<Face>, 2> output_faces;
    std::array<std::vector<Surface>, 2> output_surfaces;
    struct CapEdge
    {
        unsigned occurrences = 0;
        int orientation = 0;
    };
    std::map<std::pair<std::size_t, std::size_t>, CapEdge> cap_edges;
    for (std::size_t f = 0; f < impl_->faces.size(); ++f)
    {
        const auto& face = impl_->faces[f];
        for (unsigned outside = 0; outside < (construct_outside ? 2u : 1u); ++outside)
        {
            Face loop;
            for (std::size_t i = 0; i < face.size(); ++i)
            {
                const auto a = face[i], b = face[(i + 1) % face.size()];
                if (outside ? signs[a] >= 0 : signs[a] <= 0)
                    loop.push_back(a);
                if (signs[a] * signs[b] < 0)
                    loop.push_back(crossing(a, b));
            }
            if (loop.size() >= 3 &&
                !std::all_of(loop.begin(), loop.end(), [&](auto i) { return incident<Mode>(vertices[i], plane); }))
            {
                // A source face can meet the cut in an edge containing three
                // or more propagated collinear vertices. It is not a positive
                // area output face; the retained adjacent face contributes the
                // same authenticated edge chain to the cap.
                output_faces[outside].push_back(loop);
                output_surfaces[outside].push_back(impl_->surfaces[f]);
                if (!outside)
                    for (std::size_t i = 0; i < loop.size(); ++i)
                    {
                        const auto a = loop[i], b = loop[(i + 1) % loop.size()];
                        if (a != b && incident<Mode>(vertices[a], plane) && incident<Mode>(vertices[b], plane))
                        {
                            active_work->pair();
                            const std::pair key{std::min(a, b), std::max(a, b)};
                            auto edge = cap_edges.find(key);
                            if (edge == cap_edges.end())
                            {
                                active_work->charge(
                                    1, sizeof(typename decltype(cap_edges)::value_type) + 4 * sizeof(void*));
                                edge = cap_edges.emplace(key, CapEdge{}).first;
                            }
                            if (++edge->second.occurrences > 2)
                                reject("section edge has more than two face incidences.");
                            edge->second.orientation += a < b ? 1 : -1;
                        }
                    }
            }
        }
    }
    std::map<std::size_t, std::set<std::size_t>> adjacency;
    for (const auto& [edge, incidence] : cap_edges)
    {
        active_work->pair();
        if (incidence.occurrences == 2)
        {
            if (incidence.orientation != 0)
                reject("section edge has repeated oriented boundary coverage.");
            continue; // Opposite inherited face edges are internal.
        }
        if (std::abs(incidence.orientation) != 1)
            reject("section edge has invalid oriented boundary coverage.");
        const auto [a, b] = edge;
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }
    if (adjacency.size() < 3)
        reject("positive complementary cut has no closed section.");
    for (const auto& [vertex, neighbors] : adjacency)
        if (neighbors.size() != 2)
            reject("section edge incidence is not one closed cycle.");
    Face cap;
    const auto first = adjacency.begin()->first;
    std::size_t previous = first, current = *adjacency.at(first).begin();
    cap.push_back(first);
    while (current != first)
    {
        if (cap.size() >= adjacency.size())
            reject("section cycle repeats a vertex.");
        cap.push_back(current);
        const auto& neighbors = adjacency.at(current);
        const auto next = *neighbors.begin() == previous ? *neighbors.rbegin() : *neighbors.begin();
        previous = current;
        current = next;
    }
    if (cap.size() != adjacency.size())
        reject("section has disconnected boundaries.");
    // The retained faces inherit outward winding. Their unmatched edges
    // determine the cap's opposite winding without a numerical normal test.
    // Strict signs on both sides already prove a positive-area section.
    bool same_direction = false, opposite_direction = false;
    for (std::size_t i = 0; i < cap.size(); ++i)
    {
        active_work->pair();
        const auto a = cap[i], b = cap[(i + 1) % cap.size()];
        const auto orientation = cap_edges.at(std::minmax(a, b)).orientation;
        const auto direction = a < b ? 1 : -1;
        if (direction != orientation && direction != -orientation)
            reject("section cycle contains an unproved boundary edge.");
        same_direction |= direction == orientation;
        opposite_direction |= direction == -orientation;
    }
    if (same_direction && opposite_direction)
        reject("section boundary has inconsistent directed winding.");
    if (same_direction)
        std::reverse(cap.begin(), cap.end());
    if (measure_section)
        result.section = section<Mode>(*impl_, vertices, cap);
    output_faces[0].push_back(cap);
    output_surfaces[0].push_back(support);
    if (construct_outside)
    {
        std::reverse(cap.begin(), cap.end());
        output_faces[1].push_back(cap);
        output_surfaces[1].push_back({plane, -support.orientation});
    }
    // Charge immutable proof nodes and geometry wrappers as well as the shell
    // payload charged by finish<Mode>(). Private partition clips have no outer FVM
    // wrapper allocator to account for discarded temporary children.
    active_work->charge(1, sizeof(Split));
    auto split = std::make_shared<Split>();
    split->parent = impl_->ancestry;
    if (impl_->ancestry->depth == std::numeric_limits<std::size_t>::max())
        reject("construction ancestry depth overflows.");
    active_work->charge(1, sizeof(GeometryClipConstruction));
    auto construction = std::make_shared<GeometryClipConstruction>();
    construction->parent = construction_source<Mode>(impl_);
    construction->plane = plane;
    construction->mode = Mode;
    construction->orientation = support.orientation;
    construction->measure_section = measure_section;
    construction->construct_outside = construct_outside;
    construction->partition_proof = partition_proof;
    active_work->charge(plane->source_vertices.size(), sizeof(std::shared_ptr<const DynamicPointProvenance>));
    for (const auto& weak : plane->source_vertices)
        if (const auto point = weak.lock())
            construction->retained_points.push_back(point);
    const auto child = [&](unsigned outside)
    {
        active_work->charge(1, sizeof(Impl) + 2 * sizeof(Ancestry) + sizeof(DynamicConvexGeometry));
        auto poly = std::make_shared<Impl>();
        poly->frame = impl_->frame;
        poly->reference_origin = impl_->reference_origin;
        poly->limits = impl_->limits;
        poly->numerical_only = impl_->numerical_only || source->numerical_only;
        active_work->charge(1, sizeof(GeometryConstruction));
        auto recipe = std::make_shared<GeometryConstruction>();
        recipe->kind = GeometryConstruction::Kind::Clip;
        recipe->clip = construction;
        recipe->outside = outside != 0;
        poly->construction = std::move(recipe);
        poly->ancestry =
            std::make_shared<Ancestry>(Ancestry{impl_->ancestry, split, outside != 0, impl_->ancestry->depth + 1});
        std::set<std::size_t> used;
        for (const auto& face : output_faces[outside])
            used.insert(face.begin(), face.end());
        std::map<std::size_t, std::size_t> remap;
        for (const auto i : used)
        {
            remap.emplace(i, poly->vertices.size());
            poly->vertices.push_back(vertices[i]);
        }
        poly->faces = std::move(output_faces[outside]);
        poly->surfaces = std::move(output_surfaces[outside]);
        for (auto& face : poly->faces)
            for (auto& i : face)
                i = remap.at(i);
        // Certified strict signs on both sides of a full-dimensional convex
        // parent prove that both children have positive interiors. Private
        // partition witnesses use only that geometry and never their measures.
        finish<Mode>(*poly, partition_proof, true);
        return DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(poly)));
    };
    result.inside = child(0);
    if (construct_outside)
        result.outside = child(1);
    return result;
}

template<ArithmeticMode Mode> DynamicPhaseMeasures GeometryKernel<Mode>::phases(const GeometryHalfspace& gas) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    const auto split = clip(gas, false);
    return {volume(), split.inside ? GeometryKernel<Mode>(split.inside->impl_).volume() : bounded<Mode>(Interval{}),
        split.outside ? GeometryKernel<Mode>(split.outside->impl_).volume() : bounded<Mode>(Interval{})};
}

template<ArithmeticMode Mode>
bool construction_contains(const std::shared_ptr<const Ancestry>& parent, std::shared_ptr<const Ancestry> child)
{
    static_assert(Mode == kernel_mode);
    while (child && child->depth > parent->depth)
    {
        active_work->pair();
        child = child->parent;
    }
    active_work->pair();
    return child == parent;
}

template<ArithmeticMode Mode>
bool construction_disjoint(std::shared_ptr<const Ancestry> a, std::shared_ptr<const Ancestry> b)
{
    static_assert(Mode == kernel_mode);
    while (a && b && a != b)
    {
        active_work->pair();
        if (a->depth > b->depth)
            a = a->parent;
        else if (b->depth > a->depth)
            b = b->parent;
        else
        {
            if (a->split && a->split == b->split && a->outside != b->outside)
                return true;
            a = a->parent;
            b = b->parent;
        }
    }
    return false;
}

template<ArithmeticMode Mode>
bool separated_bounds(const DynamicConvexGeometry::Impl& a, const DynamicConvexGeometry::Impl& b)
{
    static_assert(Mode == kernel_mode);
    const auto other = transform<Mode>(b.bounds, b.frame, a.frame);
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (a.bounds[axis].hi <= other[axis].lo || other[axis].hi <= a.bounds[axis].lo)
            return true; // At most a zero-volume contact, including equality.
    return false;
}

template<ArithmeticMode Mode>
bool GeometryKernel<Mode>::bounds_prove_disjoint_interiors(const DynamicConvexGeometry& other) const
{
    static_assert(Mode == kernel_mode);
    require_mode(arithmetic_mode(), other.arithmetic_mode());
    Operation scope(impl_->limits);
    return separated_bounds<Mode>(*impl_, *other.impl_);
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::intersect(const DynamicConvexGeometry& other) const
{
    static_assert(Mode == kernel_mode);
    require_mode(arithmetic_mode(), other.arithmetic_mode());
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only || other.impl_->partition_proof_only)
        reject("private partition witness cannot supply a measured intersection.");
    ++statistics.intersections;
    if (separated_bounds<Mode>(*impl_, *other.impl_))
        return {};
    if (construction_disjoint<Mode>(impl_->ancestry, other.impl_->ancestry))
        return {};
    // An authenticated common support with opposing interior halfspaces
    // proves disjoint interiors before any clipping. Otherwise, clipping by
    // unrelated faces first can create poorly conditioned intermediates even
    // though a later shared support already separates the original pieces.
    for (const auto& a : impl_->surfaces)
        for (const auto& b : checked_geometry_storage<Mode>(other.impl_)->surfaces)
        {
            active_work->pair();
            int normal_orientation = 0;
            if (a.plane == b.plane)
                normal_orientation = 1;
            else if (a.plane->input && b.plane->input)
                normal_orientation = matching_input<Mode>(*a.plane->input, *b.plane->input);
            if (!normal_orientation)
            {
                const bool common_authority = std::any_of(a.plane->authorities.begin(), a.plane->authorities.end(),
                    [&](const auto& key)
                    {
                        return std::find(b.plane->authorities.begin(), b.plane->authorities.end(), key) !=
                               b.plane->authorities.end();
                    });
                if (common_authority)
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        const auto na = a.plane->normal[axis], nb = b.plane->normal[axis];
                        const int sa = na.lo() > 0 ? 1 : na.hi() < 0 ? -1 : 0;
                        const int sb = nb.lo() > 0 ? 1 : nb.hi() < 0 ? -1 : 0;
                        if (sa && sb)
                        {
                            // The key proves collinearity of the true normals.
                            // One certified common component establishes their
                            // relative orientation without a norm/dot overflow.
                            normal_orientation = sa * sb;
                            break;
                        }
                    }
            }
            if (normal_orientation * a.orientation * b.orientation < 0)
                return {};
        }
    auto result = DynamicConvexGeometryPtr(new DynamicConvexGeometry(impl_));
    for (std::size_t face = 0; result && face < other.impl_->faces.size(); ++face)
    {
        if (separated_bounds<Mode>(*result->impl_, *other.impl_))
            return {};
        // Intersection consumes only the inside. Certifying an unused thin
        // outside measure can reject an otherwise resolved intersection.
        result = GeometryKernel<Mode>(result->impl_).clip_impl(other.impl_, face, false, false).inside;
    }
    return result;
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::translated(GeometryPoint displacement) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only)
        reject("private partition witness cannot supply translated measured geometry.");
    auto result = std::make_shared<Impl>(*impl_);
    result->moment_cache = std::make_shared<MomentCache>();
    // Translation is a new source expression, with a fresh structural root.
    // Both vertex and support anchors carry the same displacement enclosure.
    result->ancestry = std::make_shared<Ancestry>();
    result->partition_certificate.reset();
    result->construction.reset();
    std::map<PlanePtr, PlanePtr> planes;
    for (auto& vertex : result->vertices)
        for (auto& plane : vertex.incident)
        {
            if (!planes.contains(plane))
            {
                auto shifted = std::make_shared<Plane>(*plane);
                shifted->anchor = shifted->anchor + scaled<Mode>(point<Mode>(displacement), -shifted->frame.exponent);
                shifted->input.reset();
                shifted->authorities.clear();
                shifted->source_vertices.clear();
                planes.emplace(plane, shifted);
            }
            plane = planes.at(plane);
        }
    for (auto& surface : result->surfaces)
        surface.plane = planes.at(surface.plane);
    for (auto& vertex : result->vertices)
    {
        vertex.coordinates = vertex.coordinates + scaled<Mode>(point<Mode>(displacement), -result->frame.exponent);
        vertex.authorities.clear();
        vertex.source_identity.reset();
    }
    finish<Mode>(*result, false, true);
    return DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(result)));
}

template int matching_input<kernel_mode>(const GeometryHalfspace& a, const GeometryHalfspace& b);
template bool incident<kernel_mode>(const Vertex& v, const PlanePtr& plane);
template Interval refine_triangle_distance<kernel_mode>(const Point& coordinates, const Frame& frame, const Vertex& a,
    const Vertex& b, const Vertex& c, const Frame& triangle_frame, const PlanePtr& plane, int orientation,
    Interval result);
template DynamicGeometrySection section<kernel_mode>(
    const StoredGeometry& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop);
template DynamicClipResult GeometryKernel<kernel_mode>::clip(
    const GeometryHalfspace& input, bool measure_section) const;
template DynamicClipResult GeometryKernel<kernel_mode>::clip_by_face(
    const DynamicConvexGeometry& source, std::size_t face, bool measure_section) const;
template DynamicClipResult GeometryKernel<kernel_mode>::clip_impl(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& source, std::size_t plane_id, bool measure_section,
    bool construct_outside, bool partition_proof, int orientation_override) const;
template DynamicPhaseMeasures GeometryKernel<kernel_mode>::phases(const GeometryHalfspace& gas) const;
template bool construction_contains<kernel_mode>(
    const std::shared_ptr<const Ancestry>& parent, std::shared_ptr<const Ancestry> child);
template bool construction_disjoint<kernel_mode>(std::shared_ptr<const Ancestry> a, std::shared_ptr<const Ancestry> b);
template bool separated_bounds<kernel_mode>(const DynamicConvexGeometry::Impl& a, const DynamicConvexGeometry::Impl& b);
template bool GeometryKernel<kernel_mode>::bounds_prove_disjoint_interiors(const DynamicConvexGeometry& other) const;
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::intersect(const DynamicConvexGeometry& other) const;
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::translated(GeometryPoint displacement) const;
} // namespace certification_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
