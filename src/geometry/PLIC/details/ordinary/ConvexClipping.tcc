/** @file ordinary/ConvexClipping.tcc
 * @brief Ordinary convex clipping with retained topology and construction provenance.
 */
namespace SimpleFluid::PLIC::details
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
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

bool same_source_plane(const Plane& a, const Plane& b)
{
    if (a.source_vertices.size() != 3 || b.source_vertices.size() != 3 || !certification_environment())
        return false;
    const std::array<const Plane*, 2> planes{&a, &b};
    std::array<std::array<std::shared_ptr<const DynamicPointProvenance>, 3>, 2> tokens;
    for (std::size_t side = 0; side < planes.size(); ++side)
    {
        const auto& source = *planes[side];
        if (!source.normal_error || !std::isfinite(*source.normal_error) || *source.normal_error < 0)
            return false;
        long double maximum = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            active_work->pair();
            if (!std::isfinite(source.normal[i].value))
                return false;
            maximum = std::max(maximum, std::abs(source.normal[i].value));
            tokens[side][i] = source.source_vertices[i].lock();
            if (!tokens[side][i] || tokens[side][i]->arithmetic_mode != ArithmeticMode::Ordinary)
                return false;
            for (std::size_t j = 0; j < i; ++j)
            {
                active_work->pair();
                if (DynamicGeometryAccess::same_represented_source(tokens[side][j], tokens[side][i]))
                    return false;
            }
        }
        if (!(maximum > *source.normal_error))
            return false;
    }
    for (const auto& original : tokens[0])
    {
        bool matched = false;
        for (const auto& candidate : tokens[1])
        {
            active_work->pair();
            if (DynamicGeometryAccess::same_represented_source(candidate, original))
            {
                matched = true;
                break;
            }
        }
        if (!matched)
            return false;
    }
    // The same three noncollinear original points define one unoriented
    // support. Neither rounded normal equality nor a raw residual is evidence.
    return true;
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
    for (const auto& retained : v.incident)
    {
        active_work->pair();
        if (same_source_plane(*retained, *plane))
            return true;
    }
    return construction_edge_incident<Mode>(v, plane);
}

bool recover_source_identity(Vertex& vertex, const Frame& frame)
{
    if (vertex.numerical_only || vertex.source_identity || vertex.incident.size() < 3 || !certification_environment())
        return false;
    for (const auto& plane : vertex.incident)
    {
        active_work->pair();
        if (!plane->normal_error || !std::isfinite(*plane->normal_error) || *plane->normal_error < 0)
            return false;
    }
    const auto authenticated =
        [&](const std::shared_ptr<const DynamicPointProvenance>& candidate, const PlanePtr& plane)
    {
        for (const auto& weak : plane->source_vertices)
        {
            active_work->pair();
            if (DynamicGeometryAccess::same_represented_source(weak.lock(), candidate))
                return true;
        }
        for (const auto& key : plane->authorities)
            if (!key.key.empty())
                for (const auto& original : candidate->supports)
                {
                    active_work->pair();
                    if (key == original)
                        return true;
                }
        return false;
    };
    std::shared_ptr<const DynamicPointProvenance> unique;
    for (const auto& plane : vertex.incident)
        for (const auto& weak : plane->source_vertices)
        {
            active_work->pair();
            const auto candidate = weak.lock();
            if (!candidate || DynamicGeometryAccess::same_represented_source(candidate, unique) ||
                candidate->arithmetic_mode != ArithmeticMode::Ordinary)
                continue;
            bool covers_every_support = true;
            for (const auto& support : vertex.incident)
                if (!authenticated(candidate, support))
                {
                    covers_every_support = false;
                    break;
                }
            if (covers_every_support)
            {
                if (unique)
                    return false; // Distinct original tokens must not be merged.
                unique = candidate;
            }
        }
    if (!unique)
        return false;
    bool independent = false;
    for (std::size_t a = 0; a + 2 < vertex.incident.size() && !independent; ++a)
        for (std::size_t b = a + 1; b + 1 < vertex.incident.size() && !independent; ++b)
            for (std::size_t c = b + 1; c < vertex.incident.size(); ++c)
            {
                active_work->pair();
                if (independent_source_normals(*vertex.incident[a], *vertex.incident[b], *vertex.incident[c]))
                {
                    independent = true;
                    break;
                }
            }
    if (!independent)
        return false;
    // Three independent original supports identify this source point uniquely.
    // No rounded residual or coordinate proximity participates in the proof.
    const auto original = DynamicPoint{unique->value, unique->absolute_error, ArithmeticMode::Ordinary};
    const auto coordinates = transform<kernel_mode>(point<kernel_mode>(original), Frame{}, frame);
    for (const auto& key : unique->supports)
    {
        bool retained = false;
        for (const auto& current : vertex.authorities)
        {
            active_work->pair();
            if (key == current)
            {
                retained = true;
                break;
            }
        }
        if (!retained)
        {
            active_work->charge(1, sizeof(PlaneAuthority));
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
            vertex.authorities.push_back(key);
        }
    }
    vertex.coordinates = coordinates;
    vertex.source_identity = std::move(unique);
    return true;
}

template<ArithmeticMode Mode>
DynamicGeometrySection section(
    const StoredGeometry& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop)
{
    static_assert(Mode == kernel_mode);
    DynamicGeometrySection result;
    result.area = bounded<Mode>(Scalar{});
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
    area_vector = area_vector * Scalar(.5L);
    const auto squared_area = dot<Mode>(area_vector, area_vector);
    if (squared_area.zero())
        return result; // A proven line/point contact retains its vertices.
    result.area = bounded<Mode>(scaled<Mode>(square_root<Mode>(squared_area), 2 * poly.frame.exponent));
    Point weighted;
    Scalar total_weight;
    for (std::size_t i = 1; i + 1 < loop.size(); ++i)
    {
        const auto& b = vertices[loop[i]].coordinates;
        const auto& c = vertices[loop[i + 1]].coordinates;
        const auto weight = dot<Mode>(cross<Mode>(b - a, c - a), area_vector);
        total_weight = total_weight + weight;
        weighted = weighted + (a + b + c) * (weight / Scalar(3));
    }
    result.centroid = bounded<Mode>(
        scaled<Mode>(weighted * (Scalar(1) / total_weight), poly.frame.exponent) + point<Mode>(poly.frame.origin));
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
    plane->offset = scaled<Mode>(Scalar(input.offset), -impl_->frame.exponent);
    plane->input = input;
    plane->native_support = details::ordinary_native::input_support_packet(input.normal, input.anchor, input.offset);
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
    std::vector<Scalar> distances;
    std::vector<int> signs;
    bool negative = false, positive = false;
    for (const auto& vertex : impl_->vertices)
    {
        auto d = distance<Mode>(vertex, impl_->frame, plane) * Scalar(support.orientation);

        const auto s = sign<Mode>(d);
        distances.push_back(d);
        signs.push_back(s);
        negative = negative || s < 0;
        positive = positive || s > 0;
    }
    DynamicClipResult result;
    result.section.area = bounded<Mode>(Scalar{});
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
        if (!((distances[a].value > 0 && distances[b].value < 0) || (distances[b].value > 0 && distances[a].value < 0)))
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
        recover_source_identity(vertex, impl_->frame);
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
    return {volume(), split.inside ? GeometryKernel<Mode>(split.inside->impl_).volume() : bounded<Mode>(Scalar{}),
        split.outside ? GeometryKernel<Mode>(split.outside->impl_).volume() : bounded<Mode>(Scalar{})};
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
                        const int sa = na.value > 0 ? 1 : na.value < 0 ? -1 : 0;
                        const int sb = nb.value > 0 ? 1 : nb.value < 0 ? -1 : 0;
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
    // Both vertex and support anchors carry the same displacement.
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
                shifted->normal_error.reset();
                if (!shifted->native_support.empty())
                    shifted->native_support = details::ordinary_native::translated_support_packet(
                        shifted->native_support, details::ordinary_native::wide(displacement));
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
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
