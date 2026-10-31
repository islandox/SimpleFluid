/** @file ordinary/ConvexClipping.tcc
 * @brief Ordinary convex clipping with retained topology and construction provenance.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <exception>
#include <sstream>

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

bool recover_source_identity(Vertex& vertex, const Frame& frame, const StoredGeometry* parent, const Vertex* edge_first,
    const Vertex* edge_second)
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
            const auto owner = weak.lock();
            if (DynamicGeometryAccess::same_represented_source(owner, candidate))
                return true;
        }
        for (const auto& key : plane->authorities)
            if (!key.empty())
                for (const auto& original : candidate->supports())
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
    const auto& source_point = DynamicGeometryAccess::borrowed_original_source(unique);
    if (!source_point)
    {
        ++statistics.rejections;
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Source recovery lost its original point authority.");
    }
    for (const auto error : source_point->absolute_error)
        if (error != 0)
        {
            ++statistics.rejections;
            throw GeometryError(std::isfinite(error) && error > 0 ? GeometryErrorCode::UnresolvedArithmetic
                                                                  : GeometryErrorCode::InvalidInput,
                "Original source point does not have a resolved represented placement.");
        }
    if (parent)
    {
        if (!edge_first || !edge_second)
        {
            ++statistics.rejections;
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Source recovery requires the original edge endpoints.");
        }
        // A unique intersection of infinite support planes need not belong to
        // the retained parent or edge. Check the proposed original point before
        // assigning any recovered coordinates, identity or support authority.
        Vertex candidate;
        candidate.source_identity = unique;
        for (std::size_t axis = 0; axis < 3; ++axis)
            candidate.coordinates[axis] = StorageScalar(source_point->value[axis]);
        for (const auto& surface : parent->surfaces)
        {
            active_work->pair();
            if (sign<kernel_mode>(
                    distance<kernel_mode>(candidate, Frame{}, surface.plane) * Scalar(surface.orientation)) > 0)
            {
                ++statistics.rejections;
                throw GeometryError(GeometryErrorCode::InvalidTopology,
                    "Recovered source point lies outside an inherited parent halfspace.");
            }
        }
        const ordinary_predicates::FramedPoint proposed{source_point->value, {}, 0};
        const auto endpoint = [&](const Vertex& v)
        {
            return ordinary_predicates::FramedPoint{
                {v.coordinates[0].value, v.coordinates[1].value, v.coordinates[2].value}, parent->frame.origin,
                parent->frame.exponent};
        };
        const auto first = endpoint(*edge_first), second = endpoint(*edge_second);
        bool coordinate_bracket = true;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            ordinary_predicates::Point unit{};
            unit[axis] = 1;
            const auto a =
                ordinary_predicates::require_sign(ordinary_predicates::plane_distance(proposed, first, unit, 0));
            const auto b =
                ordinary_predicates::require_sign(ordinary_predicates::plane_distance(proposed, second, unit, 0));
            if (a != 0 && a == b)
            {
                coordinate_bracket = false;
                break;
            }
        }
        if (!coordinate_bracket)
        {
            // Rounded generated endpoints can lie just beyond their original
            // endpoint support. A shared root edge alone does not prove current
            // segment membership: require a proved current endpoint or two
            // transverse current-parent faces that bound the common source line.
            const auto exact_anchor = [&](const PlanePtr& support)
            {
                if (!support->normal_error || *support->normal_error != 0 || support->offset.value != 0 ||
                    support->source_vertices.empty())
                    return false;
                const auto owner = support->source_vertices.front().lock();
                const auto& token = DynamicGeometryAccess::borrowed_original_source(owner);
                if (!token || token->constructed || token->arithmetic_mode != ArithmeticMode::Ordinary ||
                    token->absolute_error != std::array<long double, 3>{})
                    return false;
                const ordinary_predicates::FramedPoint anchor{
                    {support->anchor[0].value, support->anchor[1].value, support->anchor[2].value},
                    support->frame.origin, support->frame.exponent};
                const ordinary_predicates::FramedPoint original_anchor{token->value, {}, 0};
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    ordinary_predicates::Point unit{};
                    unit[axis] = 1;
                    const auto residual = ordinary_predicates::plane_distance(anchor, original_anchor, unit, 0);
                    if (residual.value != 0 || residual.absolute_error != 0)
                        return false;
                }
                return true;
            };
            const auto common_support = [&](const PlanePtr& support)
            {
                return incident<kernel_mode>(*edge_first, support) && incident<kernel_mode>(*edge_second, support) &&
                       incident<kernel_mode>(candidate, support);
            };
            struct EndpointBounds
            {
                bool coincident = false;
                unsigned directions = 0;
            };
            const auto endpoint_bounds = [&](const Vertex& endpoint, const Vertex& other, const PlanePtr& first_support,
                                             const PlanePtr& second_support)
            {
                EndpointBounds result;
                for (const auto& bound : endpoint.incident)
                {
                    active_work->pair();
                    if (incident<kernel_mode>(other, bound))
                        continue;
                    const int derivative = source_normal_orientation(*first_support, *second_support, *bound);
                    if (!derivative)
                        continue;
                    if (incident<kernel_mode>(candidate, bound))
                    {
                        result.coincident = true;
                        continue;
                    }
                    // Numerical signs require an exact original plane
                    // expression, including its anchor, not just a rank bound.
                    if (!exact_anchor(bound))
                        continue;
                    const int candidate_side = sign<kernel_mode>(distance<kernel_mode>(candidate, Frame{}, bound));
                    if (!candidate_side)
                    {
                        result.coincident = true;
                        continue;
                    }
                    for (const auto& surface : parent->surfaces)
                    {
                        active_work->pair();
                        if (surface.plane != bound)
                            continue;
                        const int other_side = sign<kernel_mode>(distance<kernel_mode>(other, parent->frame, bound));
                        if (other_side * surface.orientation >= 0)
                            continue;
                        if (candidate_side * surface.orientation > 0)
                        {
                            ++statistics.rejections;
                            throw GeometryError(GeometryErrorCode::InvalidTopology,
                                "Recovered source point violates an original endpoint bound.");
                        }
                        result.directions |= derivative * surface.orientation < 0 ? 1u : 2u;
                    }
                }
                return result;
            };
            const auto source_segment = [&]
            {
                for (std::size_t a = 0; a + 1 < vertex.incident.size(); ++a)
                {
                    const auto& first_support = vertex.incident[a];
                    if (!common_support(first_support))
                        continue;
                    for (std::size_t b = a + 1; b < vertex.incident.size(); ++b)
                    {
                        const auto& second_support = vertex.incident[b];
                        if (!common_support(second_support))
                            continue;
                        const auto first_bound =
                            endpoint_bounds(*edge_first, *edge_second, first_support, second_support);
                        const auto second_bound =
                            endpoint_bounds(*edge_second, *edge_first, first_support, second_support);
                        if (first_bound.coincident || second_bound.coincident)
                            return true;
                        // Interior admission needs two current parent faces
                        // cutting the common line in opposite outward senses.
                        // Arbitrary ancestral planes or same-direction bounds
                        // cannot widen a retained subsegment.
                        if ((first_bound.directions == 1 && second_bound.directions == 2) ||
                            (first_bound.directions == 2 && second_bound.directions == 1))
                            return true;
                    }
                }
                return false;
            };
            bool source_segment_admitted = false;
            try
            {
                source_segment_admitted = source_segment();
            }
            catch (const GeometryError& error)
            {
                if (error.code() != GeometryErrorCode::UnresolvedArithmetic)
                    throw;
            }
            if (!source_segment_admitted)
            {
                ++statistics.rejections;
                throw GeometryError(GeometryErrorCode::InvalidTopology,
                    "Recovered source point lies outside the original edge bracket.");
            }
        }
    }
    // Three independent original supports identify this source point uniquely.
    // No rounded residual or coordinate proximity participates in the proof.
    const ordinary_predicates::FramedPoint original{source_point->value, {}, 0};
    const ordinary_predicates::FramedPoint destination{{}, frame.origin, frame.exponent};
    Point coordinates;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        ordinary_predicates::Point unit{};
        unit[axis] = 1;
        const auto placed = ordinary_predicates::plane_distance(original, destination, unit, 0);
        if (placed.absolute_error != 0)
        {
            ++statistics.rejections;
            throw GeometryError(GeometryErrorCode::UnresolvedArithmetic,
                "Original source point cannot be placed exactly in the construction frame.");
        }
        coordinates[axis] = StorageScalar(placed.value);
    }
    for (const auto& key : unique->supports())
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
            active_work->allocations.retain(key, active_work->bytes, active_work->limits.byte_budget);
            vertex.authorities.append(key);
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
        rebased->export_source = impl_->export_source ? impl_->export_source : impl_;
        rebased->frame.origin = input.anchor;
        rebased->construction = construction_rebase(construction_source<Mode>(impl_), input);
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
    if (!negative || !positive)
    {
        DynamicClipResult result;
        result.section.area = bounded<Mode>(Scalar{});
        result.section.centroid = bounded<Mode>(Point{});
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

    // Match source-frame coordinates only where relocation is exact for every
    // immutable parent vertex and does not increase that axis's maximum local
    // magnitude. This keeps small source-relative displacements before any
    // crossing is rounded, without moving the parent or selecting a tolerance.
    Frame aligned_frame = impl_->frame;
    std::vector<Point> aligned;
    const auto relocated = [&](const Point& point, const Frame& destination, std::size_t axis)
    {
        const ordinary_predicates::FramedPoint original{
            {point[0].value, point[1].value, point[2].value}, impl_->frame.origin, impl_->frame.exponent};
        const ordinary_predicates::FramedPoint anchor{{}, destination.origin, destination.exponent};
        ordinary_predicates::Point unit{};
        unit[axis] = 1;
        return ordinary_predicates::plane_distance(original, anchor, unit, 0);
    };
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (plane->frame.origin[axis] == impl_->frame.origin[axis])
            continue;
        auto candidate = aligned_frame;
        candidate.origin[axis] = plane->frame.origin[axis];
        const auto maximum = std::max(std::abs(impl_->bounds[axis].lo), std::abs(impl_->bounds[axis].hi));
        bool eligible = true;
        try
        {
            for (const auto bound : {impl_->bounds[axis].lo, impl_->bounds[axis].hi})
            {
                Point endpoint{};
                endpoint[axis] = StorageScalar(bound);
                const auto converted = relocated(endpoint, candidate, axis);
                if (converted.absolute_error != 0 || std::abs(converted.value) > maximum)
                {
                    eligible = false;
                    break;
                }
            }
            if (eligible)
            {
                if (aligned.empty())
                {
                    active_work->charge(impl_->vertices.size(), sizeof(Point));
                    aligned.reserve(impl_->vertices.size());
                    for (const auto& vertex : impl_->vertices)
                        aligned.push_back(vertex.coordinates);
                }
                for (std::size_t i = 0; i < impl_->vertices.size(); ++i)
                {
                    const auto converted = relocated(impl_->vertices[i].coordinates, candidate, axis);
                    if (converted.absolute_error != 0)
                    {
                        eligible = false;
                        break;
                    }
                    aligned[i][axis] = StorageScalar(converted.value);
                }
            }
        }
        catch (const GeometryError& error)
        {
            if (error.code() != GeometryErrorCode::UnresolvedArithmetic)
                throw;
            eligible = false;
        }
        if (eligible)
            aligned_frame = candidate;
        else if (!aligned.empty())
            for (std::size_t i = 0; i < impl_->vertices.size(); ++i)
                aligned[i][axis] = impl_->vertices[i].coordinates[axis];
    }
    const auto build_split = [&](auto&& build_split, const Frame& frame, std::span<const Point> reframed,
                                 bool reconstruction_attempt) -> DynamicClipResult
    {
        DynamicClipResult result;
        result.section.area = bounded<Mode>(Scalar{});
        result.section.centroid = bounded<Mode>(Point{});
        std::vector<Vertex> vertices = impl_->vertices;
        for (std::size_t i = 0; i < reframed.size(); ++i)
            vertices[i].coordinates = reframed[i];
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
                        vertices[i].authorities.append(key);
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
            if (!((distances[a].value > 0 && distances[b].value < 0) ||
                    (distances[b].value > 0 && distances[a].value < 0)))
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
                    vertex.authorities.append(key);
            for (const auto& key : plane->authorities)
                if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) == vertex.authorities.end())
                    vertex.authorities.append(key);
            const bool recovered_source =
                recover_source_identity(vertex, frame, impl_.get(), &impl_->vertices[a], &impl_->vertices[b]);
            preserve_crossing_incidence<Mode>(
                vertex, vertices[a], vertices[b], frame, impl_->numerical_only || source->numerical_only);
            // Distinct source edges can recover the same original point. Reuse
            // that authenticated identity in every incident face and both children;
            // equal rounded coordinates alone never authorize this replacement.
            if (recovered_source)
                for (std::size_t i = 0; i < vertices.size(); ++i)
                {
                    active_work->pair();
                    if (!DynamicGeometryAccess::same_represented_source(
                            vertex.source_identity, vertices[i].source_identity) ||
                        !incident<Mode>(vertices[i], plane))
                        continue;
                    for (const auto& support : vertex.incident)
                        if (std::find(vertices[i].incident.begin(), vertices[i].incident.end(), support) ==
                            vertices[i].incident.end())
                            vertices[i].incident.push_back(support);
                    for (const auto& authority : vertex.authorities)
                        if (std::find(vertices[i].authorities.begin(), vertices[i].authorities.end(), authority) ==
                            vertices[i].authorities.end())
                            vertices[i].authorities.append(authority);
                    crossings.emplace(key, i);
                    return i;
                }
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
                // Only repeated references are removed. Keep all established edge
                // subdivisions, including collinear ones, conforming across faces.
                loop.erase(std::unique(loop.begin(), loop.end()), loop.end());
                if (loop.size() > 1 && loop.front() == loop.back())
                    loop.pop_back();
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
        output_faces[0].push_back(cap);
        output_surfaces[0].push_back(support);
        if (construct_outside)
        {
            auto outside_cap = cap;
            std::reverse(outside_cap.begin(), outside_cap.end());
            output_faces[1].push_back(std::move(outside_cap));
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
        const auto resize_scratch = [](auto& buffer, std::size_t count)
        {
            using Value = typename std::remove_reference_t<decltype(buffer)>::value_type;
            if (count > buffer.max_size() || count > std::numeric_limits<std::size_t>::max() / sizeof(Value))
            {
                ++statistics.rejections;
                throw GeometryError(GeometryErrorCode::ResourceLimit,
                    "Ordinary FP geometry: clipping scratch size overflows.");
            }
            if (count > buffer.capacity())
            {
                // Work is cumulative: a replacement allocation is charged in full.
                active_work->charge(count, sizeof(Value));
                buffer.reserve(count);
                if (buffer.capacity() > count)
                    active_work->charge(buffer.capacity() - count, sizeof(Value));
            }
            buffer.resize(count);
        };
        std::array<std::shared_ptr<Impl>, 2> drafts;
        {
            // Crossing construction is complete. Include generated vertex IDs and
            // reuse these arrays only across the requested children of this split.
            std::vector<unsigned char> used;
            std::vector<std::size_t> old_to_new;
            resize_scratch(used, vertices.size());
            resize_scratch(old_to_new, vertices.size());
            const auto draft_child = [&](unsigned outside)
            {
                active_work->charge(1, sizeof(Impl) + 2 * sizeof(Ancestry) + sizeof(DynamicConvexGeometry));
                auto poly = std::make_shared<Impl>();
                poly->frame = frame;
                poly->reference_origin = impl_->reference_origin;
                poly->limits = impl_->limits;
                poly->numerical_only = impl_->numerical_only || source->numerical_only;
                poly->construction = construction_node(GeometryConstruction::Clip{construction, outside != 0});
                poly->ancestry =
                    std::make_shared<Ancestry>(Ancestry{impl_->ancestry, split, outside != 0, impl_->ancestry->depth + 1});
                std::fill(used.begin(), used.end(), 0);
                for (const auto& face : output_faces[outside])
                    for (const auto i : face)
                    {
                        if (i >= vertices.size())
                            reject("complementary child has an invalid face vertex.");
                        used[i] = 1;
                    }
                for (std::size_t i = 0; i < vertices.size(); ++i)
                    if (used[i])
                    {
                        old_to_new[i] = poly->vertices.size();
                        poly->vertices.push_back(vertices[i]);
                    }
                poly->faces = std::move(output_faces[outside]);
                poly->surfaces = std::move(output_surfaces[outside]);
                for (auto& face : poly->faces)
                    for (auto& i : face)
                        i = old_to_new[i];
                return poly;
            };
            drafts[0] = draft_child(0);
            if (construct_outside)
                drafts[1] = draft_child(1);
        } // Release compaction scratch before validation or recursive recovery.

        // Validate the whole requested split before evaluating either child. The
        // cap already has the directed winding inherited from the parent faces;
        // no near-degenerate numerical normal is needed to orient it again.
        const auto validate_draft = [&](const Impl& poly)
        {
            if (poly.vertices.size() < 4 || poly.faces.size() < 4 || poly.faces.size() != poly.surfaces.size())
                reject("positive complementary child has an incomplete boundary.");
            std::map<std::pair<std::size_t, std::size_t>, CapEdge> edges;
            std::vector<unsigned char> distinct;
            resize_scratch(distinct, poly.vertices.size());
            for (const auto& face : poly.faces)
            {
                if (face.size() < 3)
                    reject("complementary child face has fewer than three vertices.");
                for (std::size_t i = 0; i < face.size(); ++i)
                {
                    active_work->pair();
                    const auto a = face[i], b = face[(i + 1) % face.size()];
                    if (a >= poly.vertices.size() || b >= poly.vertices.size() || distinct[a])
                        reject("complementary child has an invalid or repeated face vertex.");
                    distinct[a] = 1;
                    const std::pair edge{std::min(a, b), std::max(a, b)};
                    auto found = edges.find(edge);
                    if (found == edges.end())
                    {
                        active_work->charge(1, sizeof(typename decltype(edges)::value_type) + 4 * sizeof(void*));
                        found = edges.emplace(edge, CapEdge{}).first;
                    }
                    if (++found->second.occurrences > 2)
                        reject("complementary child edge has more than two face incidences.");
                    found->second.orientation += a < b ? 1 : -1;
                }
                for (const auto i : face)
                    distinct[i] = 0;
            }
            for (const auto& [edge, incidence] : edges)
                if (incidence.occurrences != 2 || incidence.orientation != 0)
                    reject("complementary child is not an oriented closed two-manifold.");
            validate_connected_boundary<Mode>(poly.vertices.size(), poly.faces);
        };
        for (const auto& draft : drafts)
            if (draft)
                validate_draft(*draft);
        for (const auto& draft : drafts)
            if (draft)
                prepare_finish<Mode>(*draft, partition_proof);

        // Keep numerical statuses until every requested child has been measured.
        // Parent signs, not a zero/unresolved rounded child, establish contact.
        // A strict two-sided split must therefore still reject a nonpositive shell.
        enum class MeasureStatus
        {
            Positive,
            Zero,
            Negative,
            Unresolved,
            PrivateWitness
        };
        struct DraftMeasure
        {
            MeasureStatus status = MeasureStatus::PrivateWitness;
            std::exception_ptr unresolved;
        };
        std::array<DraftMeasure, 2> measures;
        if (!partition_proof)
            for (std::size_t side = 0; side < drafts.size(); ++side)
                if (const auto& draft = drafts[side])
                {
                    try
                    {
                        ++statistics.volume_evaluations;
                        draft->volume = integrate<Mode>(*draft, false).volume;
                        measures[side].status = draft->volume.value > 0   ? MeasureStatus::Positive
                                                : draft->volume.value < 0 ? MeasureStatus::Negative
                                                                          : MeasureStatus::Zero;
                    }
                    catch (const GeometryError& error)
                    {
                        if (error.code() != GeometryErrorCode::UnresolvedArithmetic)
                            throw;
                        measures[side].status = MeasureStatus::Unresolved;
                        measures[side].unresolved = std::current_exception();
                    }
                }
        // A thin cut can be unrepresentable in the parent's distant frame
        // even though its positive intersection is representable near a retained
        // source vertex. Retry once from the immutable parent and cut, never
        // from the rejected child. The frame change must preserve every parent
        // coordinate exactly; only Ordinary crossing rounding is evaluated anew.
        const auto failed_measure = [](MeasureStatus status)
        {
            return status == MeasureStatus::Zero || status == MeasureStatus::Negative ||
                   status == MeasureStatus::Unresolved;
        };
        if ((failed_measure(measures[0].status) || failed_measure(measures[1].status)) && !reconstruction_attempt &&
            !partition_proof && !impl_->numerical_only && !source->numerical_only && impl_->vertices.size() == 4 &&
            impl_->faces.size() == 4 &&
            std::all_of(impl_->faces.begin(), impl_->faces.end(), [](const auto& face) { return face.size() == 3; }) &&
            std::find(signs.begin(), signs.end(), 0) == signs.end())
        {
            // Choose the side nearest the cut from original distance data,
            // independent of which outputs were requested or failed. Sorted
            // source-coordinate ties preserve reversal and permutation symmetry.
            const auto key = [&](std::size_t i)
            {
                const auto& p = impl_->vertices[i].coordinates;
                return std::array<long double, 3>{p[0].value, p[1].value, p[2].value};
            };
            struct Side
            {
                long double maximum_distance = 0;
                std::array<std::size_t, 4> vertices{};
                std::size_t count = 0;
            };
            std::array<Side, 2> sides;
            for (std::size_t i = 0; i < signs.size(); ++i)
            {
                active_work->pair();
                auto& side = sides[signs[i] > 0 ? 1 : 0];
                side.maximum_distance = std::max(side.maximum_distance, std::abs(distances[i].value));
                side.vertices[side.count++] = i;
            }
            const auto coordinate_less = [&](std::size_t a, std::size_t b)
            {
                active_work->pair();
                return key(a) < key(b);
            };
            for (auto& side : sides)
                std::sort(side.vertices.begin(), side.vertices.begin() + side.count, coordinate_less);
            const bool inside_nearer =
                sides[0].maximum_distance < sides[1].maximum_distance ||
                (sides[0].maximum_distance == sides[1].maximum_distance &&
                    std::lexicographical_compare(sides[0].vertices.begin(), sides[0].vertices.begin() + sides[0].count,
                        sides[1].vertices.begin(), sides[1].vertices.begin() + sides[1].count, coordinate_less));
            const auto corner = sides[inside_nearer ? 0 : 1].vertices[0];
            Frame candidate_frame = impl_->frame;
            const auto global = scaled<Mode>(impl_->vertices[corner].coordinates, impl_->frame.exponent) +
                                point<Mode>(impl_->frame.origin);
            bool eligible = true;
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                candidate_frame.origin[axis] = static_cast<double>(global[axis].value);
                eligible = eligible && std::isfinite(candidate_frame.origin[axis]);
            }
            eligible = eligible && candidate_frame.origin != frame.origin;
            std::array<Point, 4> original_in_frame;
            try
            {
                if (eligible)
                    for (std::size_t i = 0; i < impl_->vertices.size() && eligible; ++i)
                    {
                        const auto& coordinate = impl_->vertices[i].coordinates;
                        const ordinary_predicates::FramedPoint original{
                            {coordinate[0].value, coordinate[1].value, coordinate[2].value}, impl_->frame.origin,
                            impl_->frame.exponent};
                        const ordinary_predicates::FramedPoint anchor{
                            {}, candidate_frame.origin, candidate_frame.exponent};
                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            ordinary_predicates::Point unit{};
                            unit[axis] = 1;
                            const auto transformed = ordinary_predicates::plane_distance(original, anchor, unit, 0);
                            if (transformed.absolute_error != 0)
                            {
                                eligible = false;
                                break;
                            }
                            original_in_frame[i][axis] = Scalar(transformed.value);
                        }
                    }
            }
            catch (const GeometryError& error)
            {
                if (error.code() != GeometryErrorCode::UnresolvedArithmetic)
                    throw;
                eligible = false;
            }
            if (eligible)
                return build_split(build_split, candidate_frame, original_in_frame, true);
        }
        for (std::size_t side = 0; side < drafts.size(); ++side)
            if (drafts[side])
            {
                if (measures[side].status == MeasureStatus::Zero || measures[side].status == MeasureStatus::Negative)
                {
                    std::ostringstream reason;
                    reason.precision(std::numeric_limits<long double>::max_digits10);
                    reason << "computed signed volume is not positive (volume=" << drafts[side]->volume.value << ").";
                    reject(reason.str());
                }
            }
        // A resolved inconsistency on either side takes precedence over a
        // recoverable numerical trial failure on its requested complement.
        for (const auto& measure : measures)
            if (measure.status == MeasureStatus::Unresolved)
                std::rethrow_exception(measure.unresolved);
        if (measure_section)
            result.section = section<Mode>(*drafts[0], vertices, cap);
        for (const auto& draft : drafts)
            if (draft)
                ++statistics.constructions;
        result.inside = DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(drafts[0])));
        if (construct_outside)
            result.outside = DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(drafts[1])));
        return result;
    };
    if (aligned_frame.origin != impl_->frame.origin)
        return build_split(build_split, aligned_frame, aligned, false);
    return build_split(build_split, impl_->frame, {}, false);
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
                if (shifted->native_support.has_value())
                    shifted->native_support = details::ordinary_native::translated_support_packet(
                        *shifted->native_support, details::ordinary_native::wide(displacement));
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
