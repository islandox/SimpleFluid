#include "geometry/swept_volume/details/exact/SweptGeometryInternal.hh"

namespace SimpleFluid::geometry::swept_volume
{
namespace exact_swept_detail
{
void target_sweep_shell(
    const SweepRecipient<ArithmeticMode::Exact>& target, SweepBudget& budget, std::map<P, P>& shared)
{
    if (!target.geometry)
        throw std::invalid_argument("Missing actual recipient geometry.");
    std::vector<PlanarFacePtr<ArithmeticMode::Exact>> faces;
    for (const auto& oriented : target.outward_faces)
    {
        if (!oriented.face || (oriented.outward_sign != 1 && oriented.outward_sign != -1))
            throw std::invalid_argument("Invalid recipient outward sweep incidence.");
        auto face = oriented.face->origin_face();
        if (oriented.outward_sign < 0)
            face = face->reversed();
        faces.push_back(std::move(face));
        for (const auto& trace : oriented.face->trajectories())
        {
            budget.add(trace.point.size() + trace.departure.size() + 2 * sizeof(P));
            const auto p = authority_point(trace.point), back = authority_point(trace.departure);
            auto [it, inserted] = shared.emplace(p, back);
            if (!inserted && it->second != back)
                throw std::invalid_argument("Incident sweeps disagree on a shared exact node trajectory.");
        }
    }
    if (!PlanarFace<ArithmeticMode::Exact>::cell_shell(faces)->equivalent(*target.geometry))
        throw std::invalid_argument("Recipient face sweeps do not close its actual oriented native shell.");
}

void departure_charge_q(const Q& q, SweepBudget& budget)
{
    std::size_t bytes = sizeof(Q);
    exact_scalar_storage(bytes, q);
    budget.add(bytes);
}

void departure_charge_point(const P& p, SweepBudget& budget)
{
    budget.add(sizeof(P));
    for (const auto& q : p)
        departure_charge_q(q, budget);
}

void departure_edge(DepartureEdgeChain& chain, const P& a, const P& b, int sign, SweepBudget& budget)
{
    budget.test();
    if (a == b)
        return;
    // Pending canonicalization and persistent map ownership are distinct copies.
    departure_charge_point(a, budget);
    departure_charge_point(b, budget);
    std::array<P, 2> key{a, b};
    if (key[1] < key[0])
    {
        std::swap(key[0], key[1]);
        sign = -sign;
    }
    departure_coefficient(chain, key, sign, budget);
}

void departure_triangle(
    DepartureTriangleChain& chain, const P& a, const P& b, const P& c, int sign, SweepBudget& budget)
{
    budget.test();
    if (a == b || b == c || a == c)
        return;
    for (const auto* p : {&a, &b, &c})
        departure_charge_point(*p, budget);
    DepartureTriangle key{a, b, c};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = i + 1; j < 3; ++j)
            if (key[j] < key[i])
            {
                std::swap(key[i], key[j]);
                sign = -sign;
            }
    departure_coefficient(chain, key, sign, budget);
}

void departure_triangle(DepartureTriangleChain& chain, const DepartureTriangle& t, int sign, SweepBudget& budget)
{
    departure_triangle(chain, t[0], t[1], t[2], sign, budget);
}

void departure_boundary_edges(DepartureEdgeChain& edges, const DepartureTriangle& t, SweepBudget& budget)
{
    for (unsigned a = 0; a < 3; ++a)
        departure_edge(edges, t[a], t[(a + 1) % 3], 1, budget);
}

// Authenticate C(B) against the original immutable signed-sweep representation.
// For the explicitly CLOSED boundary B, the polyhedral-current homotopy identity
// is d C(B)=B-C(dB)=B. A det==0 cone is a zero THREE-current. Its own formal
// triangle boundary need not be an empty identical-triangle multiset; we do not
// make that false assertion or rely on volume/moments to prove chain equality.
void departure_authenticate_cones(const SweptFace<ArithmeticMode::Exact>& face, const Polygon& current, const P& center,
    const std::vector<DepartureTriangle>& boundary, SweepBudget& budget)
{
    DepartureEdgeChain edges;
    std::map<DepartureTetraKey, std::array<std::size_t, 2>> expected, observed;
    const auto insert = [&](auto& map, const std::array<const P*, 4>& input, int sign)
    {
        for (const auto* p : input)
            departure_charge_point(*p, budget);
        DepartureTetraKey key{*input[0], *input[1], *input[2], *input[3]};
        std::sort(key.begin(), key.end());
        auto found = map.find(key);
        if (found == map.end())
        {
            departure_charge_key(key, budget);
            budget.add(sizeof(std::size_t));
            found = map.emplace(key, std::array<std::size_t, 2>{}).first;
        }
        ++found->second[sign > 0 ? 1 : 0];
    };
    for (const auto& t : boundary)
    {
        departure_boundary_edges(edges, t, budget);
        const Q determinant = dot(sub(t[0], center), cross(sub(t[1], center), sub(t[2], center)));
        departure_charge_q(determinant, budget);
        if (determinant != 0)
            insert(expected, {&center, &t[0], &t[1], &t[2]}, determinant > 0 ? 1 : -1);
    }
    if (!edges.empty())
        throw std::invalid_argument("Original face sweep boundary is not a closed exact edge chain.");
    Q observed_volume = 0;
    for (const auto& piece : face.simplices())
    {
        if (!piece.geometry || (piece.sign != 1 && piece.sign != -1))
            throw std::invalid_argument("Original face sweep has a null/invalid signed simplex.");
        const auto& poly = SweepAccess<ArithmeticMode::Exact>::poly(*piece.geometry);
        if (poly.vertices.size() != 4 || poly.faces.size() != 4)
            throw std::invalid_argument("Original signed simplex is not its authenticated tetrahedron.");
        insert(observed, {&poly.vertices[0], &poly.vertices[1], &poly.vertices[2], &poly.vertices[3]}, piece.sign);
        observed_volume += piece.sign * exact_moments(poly, center).volume;
        departure_charge_q(observed_volume, budget);
    }
    if (observed_volume != SweepAccess<ArithmeticMode::Exact>::value(face.volume()))
        throw std::invalid_argument("Original signed sweep volume differs from its prescribed exact control.");
    if (expected != observed)
        throw std::invalid_argument("Original signed sweep differs from its exact closed-boundary cones.");
    std::vector<std::array<std::vector<char>, 3>> sides;
    budget.array(2 * current.size(), sizeof(decltype(sides)::value_type));
    sides.reserve(2 * current.size());
    for (std::size_t i = 0; i < current.size(); ++i)
        for (unsigned a = 2; a < 4; ++a)
        {
            std::array<std::vector<char>, 3> side;
            for (unsigned v = 0; v < 3; ++v)
            {
                side[v] = swept_point(boundary[4 * i + a][v]);
                budget.add(side[v].capacity());
            }
            sides.push_back(std::move(side));
        }
    if (sides != face.edge_time_triangles())
        throw std::invalid_argument("Original shared edge-time certificate differs from its exact sweep.");
}

void departure_admit_boundary_storage(
    const Polygon& current, const Polygon& previous, const P& center, const P& backcenter, SweepBudget& budget)
{
    const auto bytes = [](const P& p)
    {
        std::size_t n = sizeof(P);
        for (const auto& q : p)
            exact_scalar_storage(n, q);
        return n;
    };
    std::size_t maximum = std::max(bytes(center), bytes(backcenter));
    for (const auto* polygon : {&current, &previous})
        for (const auto& p : *polygon)
            maximum = std::max(maximum, bytes(p));
    if (current.size() > std::numeric_limits<std::size_t>::max() / 18)
        throw std::invalid_argument("Departure boundary capacity overflow.");
    budget.array(4 * current.size(), sizeof(DepartureTriangle));
    // 12n stored point copies plus6n temporary side-triangle point copies.
    budget.array(18 * current.size(), maximum);
}

// Unconditional entry proof, BEFORE any optional count/topology/kernel exit.
void departure_authenticate_recipient_sweeps(const SweepRecipient<ArithmeticMode::Exact>& target, SweepBudget& budget)
{
    for (const auto& incidence : target.outward_faces)
    {
        const auto& face = *incidence.face;
        const auto& current = SweepAccess<ArithmeticMode::Exact>::polygon(*face.origin_face());
        std::map<P, P> mapped;
        for (const auto& trace : face.trajectories())
        {
            const auto p = authority_point(trace.point), back = authority_point(trace.departure);
            for (unsigned copy = 0; copy < 2; ++copy)
            {
                departure_charge_point(p, budget);
                departure_charge_point(back, budget);
            }
            budget.add(4 * sizeof(void*));
            if (!mapped.emplace(p, back).second)
                throw std::invalid_argument("Original sweep repeats a source node.");
        }
        Polygon previous;
        budget.array(current.size(), sizeof(P));
        previous.reserve(current.size());
        P center{};
        for (const auto& p : current)
        {
            const auto found = mapped.find(p);
            if (found == mapped.end())
                throw std::invalid_argument("Original sweep omits a canonical source point.");
            departure_charge_point(found->second, budget);
            previous.push_back(found->second);
            center = add(center, p);
        }
        center = mul(center, Q(1) / current.size());
        departure_charge_point(center, budget);
        const auto encoded = face.corrected_departure_center();
        budget.add(encoded.capacity());
        const auto backcenter = authority_point(encoded);
        departure_charge_point(backcenter, budget);
        departure_admit_boundary_storage(current, previous, center, backcenter, budget);
        const auto boundary = swept_boundary(current, previous, center, backcenter, nullptr);
        departure_authenticate_cones(face, current, center, boundary, budget);
    }
}

// No interpolation of absent coarse-edge nodes. Exact shared-side cancellation
// is a prerequisite; a valid hanging-edge configuration lacking that OPTIONAL
// derived topology simply falls back to the signed overlay.
std::optional<DepartureFanCandidate> departure_fan_candidate(
    const SweepRecipient<ArithmeticMode::Exact>& target, std::size_t recipient, SweepBudget& budget)
{
    const auto& native = SweepAccess<ArithmeticMode::Exact>::poly(*target.geometry);
    const auto native_moments = exact_moments(native, native.vertices[0]);
    if (!(native_moments.volume > 0))
        throw std::invalid_argument("Original recipient native volume is not positive.");
    const P kernel = add(native.vertices[0], mul(native_moments.first, Q(1) / native_moments.volume));
    departure_charge_point(kernel, budget);
    const auto native_about_kernel = exact_moments(native, kernel);
    Moments original = native_about_kernel;
    departure_charge_q(original.volume, budget);
    departure_charge_point(original.first, budget);
    DepartureTriangleChain boundary_identity;
    std::map<P, std::size_t> source_nodes, departure_nodes;
    std::vector<P> points;
    std::vector<std::array<std::size_t, 3>> triangles;
    std::size_t point_bound = target.outward_faces.size(), triangle_bound = 0;
    for (const auto& incidence : target.outward_faces)
    {
        const auto n = SweepAccess<ArithmeticMode::Exact>::polygon(*incidence.face->origin_face()).size();
        if (n > std::numeric_limits<std::size_t>::max() - triangle_bound)
            throw std::invalid_argument("Departure triangle count overflow.");
        triangle_bound += n;
    }
    if (point_bound > std::numeric_limits<std::size_t>::max() - triangle_bound)
        throw std::invalid_argument("Departure topology count overflow.");
    point_bound += triangle_bound;
    budget.array(point_bound, sizeof(P));
    budget.array(triangle_bound, sizeof(decltype(triangles)::value_type));
    points.reserve(point_bound);
    triangles.reserve(triangle_bound);
    bool unique_departure = true;
    const auto append = [&](const P& p) -> std::size_t
    {
        budget.test();
        if (departure_nodes.contains(p))
            unique_departure = false;
        departure_charge_point(p, budget);
        budget.add(sizeof(std::size_t) + 4 * sizeof(void*));
        const auto id = points.size();
        departure_charge_point(p, budget); // Separate map key and points[] owned payloads.
        departure_nodes.emplace(p, id);
        points.push_back(p);
        return id;
    };
    for (const auto& incidence : target.outward_faces)
    {
        const auto& face = *incidence.face;
        const auto& current = SweepAccess<ArithmeticMode::Exact>::polygon(*face.origin_face());
        std::map<P, P> trajectories;
        for (const auto& trace : face.trajectories())
        {
            const auto p = authority_point(trace.point), back = authority_point(trace.departure);
            departure_charge_point(p, budget);
            departure_charge_point(back, budget);
            budget.add(4 * sizeof(void*));
            departure_charge_point(p, budget);
            departure_charge_point(back, budget);
            if (!trajectories.emplace(p, back).second)
                throw std::invalid_argument("Original sweep has duplicate trajectory authority.");
        }
        Polygon previous;
        std::vector<std::size_t> node_ids;
        budget.array(current.size(), sizeof(P) + sizeof(std::size_t));
        previous.reserve(current.size());
        node_ids.reserve(current.size());
        P center{};
        for (const auto& p : current)
        {
            auto trace = trajectories.find(p);
            if (trace == trajectories.end())
                throw std::invalid_argument("Original sweep omits an actual canonical node.");
            const P& back = trace->second;
            departure_charge_point(back, budget);
            previous.push_back(back);
            center = add(center, p);
            auto source = source_nodes.find(p);
            if (source == source_nodes.end())
            {
                const auto id = append(back);
                departure_charge_point(p, budget);
                budget.add(sizeof(std::size_t) + 4 * sizeof(void*));
                source = source_nodes.emplace(p, id).first;
            }
            else if (points[source->second] != back)
                throw std::invalid_argument("Original incident sweeps disagree on a shared trajectory.");
            node_ids.push_back(source->second);
        }
        center = mul(center, Q(1) / current.size());
        const auto encoded = face.corrected_departure_center();
        budget.add(encoded.capacity());
        const P backcenter = authority_point(encoded);
        departure_admit_boundary_storage(current, previous, center, backcenter, budget);
        const auto boundary = swept_boundary(current, previous, center, backcenter, nullptr);
        for (const auto& triangle : boundary)
            for (const auto& point : triangle)
                departure_charge_point(point, budget);
        const auto center_id = append(backcenter);
        for (std::size_t i = 0; i < current.size(); ++i)
        {
            const auto j = (i + 1) % current.size();
            departure_triangle(boundary_identity, center, current[i], current[j], incidence.outward_sign, budget);
            departure_triangle(
                boundary_identity, backcenter, previous[i], previous[j], -incidence.outward_sign, budget);
            budget.array(1, sizeof(decltype(triangles)::value_type));
            auto t = std::array<std::size_t, 3>{center_id, node_ids[i], node_ids[j]};
            if (incidence.outward_sign < 0)
                std::swap(t[1], t[2]);
            triangles.push_back(t);
        }
        for (const auto& t : boundary)
            departure_triangle(boundary_identity, t, -incidence.outward_sign, budget);
        for (const auto& piece : face.simplices())
        {
            const auto m = exact_moments(SweepAccess<ArithmeticMode::Exact>::poly(*piece.geometry), kernel);
            original.volume -= incidence.outward_sign * piece.sign * m.volume;
            original.first = sub(original.first, mul(m.first, Q(incidence.outward_sign * piece.sign)));
            departure_charge_q(original.volume, budget);
            departure_charge_point(original.first, budget);
        }
    }
    // Source shell authority plus exact shared-side cancellation proves
    // d(native - outward sweeps) == this oriented departure boundary CURRENT.
    if (!unique_departure || !boundary_identity.empty())
        return {};
    if (triangles.empty())
        return {};
    using Edge = std::array<std::size_t, 2>;
    struct Use
    {
        std::size_t triangle;
        int sign;
    };
    std::map<Edge, std::vector<Use>> edges;
    budget.array(triangles.size(), sizeof(std::vector<std::size_t>));
    std::vector<std::vector<std::size_t>> neighbors(triangles.size());
    std::set<std::array<std::size_t, 3>> unique_triangles;
    for (std::size_t t = 0; t < triangles.size(); ++t)
    {
        auto key = triangles[t];
        std::sort(key.begin(), key.end());
        budget.add(sizeof(key) + 4 * sizeof(void*));
        if (!unique_triangles.insert(key).second)
            return {};
        for (unsigned a = 0; a < 3; ++a)
        {
            budget.test();
            Edge e{triangles[t][a], triangles[t][(a + 1) % 3]};
            const int sign = e[0] < e[1] ? 1 : -1;
            if (sign < 0)
                std::swap(e[0], e[1]);
            if (!edges.contains(e))
                budget.add(sizeof(Edge) + sizeof(std::vector<Use>) + 4 * sizeof(void*));
            budget.array(1, sizeof(Use));
            departure_push(edges[e], Use{t, sign}, budget);
        }
    }
    for (const auto& [edge, uses] : edges)
    {
        if (uses.size() != 2 || uses[0].sign + uses[1].sign != 0)
            return {};
        budget.array(2, sizeof(std::size_t));
        departure_push(neighbors[uses[0].triangle], uses[1].triangle, budget);
        departure_push(neighbors[uses[1].triangle], uses[0].triangle, budget);
    }
    budget.array(triangles.size(), sizeof(bool) + sizeof(std::size_t));
    std::vector<bool> seen(triangles.size());
    std::vector<std::size_t> queue;
    queue.reserve(triangles.size());
    queue.push_back(0);
    seen[0] = true;
    for (std::size_t i = 0; i < queue.size(); ++i)
        for (auto next : neighbors[queue[i]])
        {
            budget.test();
            if (!seen[next])
            {
                seen[next] = true;
                queue.push_back(next);
            }
        }
    if (queue.size() != triangles.size())
        return {};
    // Edge incidence alone misses a pinched vertex. Every vertex link must be
    // one connected simple cycle, not two disjoint cycles with the same center.
    for (std::size_t v = 0; v < points.size(); ++v)
    {
        std::map<std::size_t, std::vector<std::size_t>> link;
        for (const auto& t : triangles)
            for (unsigned a = 0; a < 3; ++a)
            {
                budget.test();
                if (t[a] == v)
                {
                    const auto x = t[(a + 1) % 3], y = t[(a + 2) % 3];
                    for (auto id : {x, y})
                        if (!link.contains(id))
                            budget.add(sizeof(std::size_t) + sizeof(std::vector<std::size_t>) + 4 * sizeof(void*));
                    budget.array(2, sizeof(std::size_t));
                    departure_push(link[x], y, budget);
                    departure_push(link[y], x, budget);
                }
            }
        if (link.size() < 3)
            return {};
        for (const auto& [node, adjacent] : link)
            if (adjacent.size() != 2 || adjacent[0] == adjacent[1])
                return {};
        budget.array(link.size(), sizeof(std::size_t) + 4 * sizeof(void*));
        std::set<std::size_t> reached;
        auto current = link.begin()->first, previous = std::numeric_limits<std::size_t>::max();
        while (!reached.contains(current))
        {
            budget.test();
            reached.insert(current);
            const auto& n = link.at(current);
            const auto next = n[0] == previous ? n[1] : n[0];
            previous = current;
            current = next;
        }
        if (current != link.begin()->first || reached.size() != link.size())
            return {};
    }
    if (points.size() + triangles.size() != edges.size() + 2)
        return {};
    DepartureFanCandidate candidate;
    candidate.recipient = recipient;
    budget.array(triangles.size(), sizeof(ConvexFragmentPtr<ArithmeticMode::Exact>));
    candidate.tetrahedra.reserve(triangles.size());
    Moments fan{};
    for (const auto& t : triangles)
    {
        budget.test();
        const Q determinant =
            dot(sub(points[t[0]], kernel), cross(sub(points[t[1]], kernel), sub(points[t[2]], kernel)));
        departure_charge_q(determinant, budget);
        if (!(determinant > 0))
            return {}; // Exact native centroid is not a strict kernel.
        // maximum_simplices belongs to each ORIGINAL face sweep. Derived
        // positive tetrahedra obey the existing atom/byte/work construction
        // limits; resource exhaustion is never an absent-geometry fallback.
        budget.atom(candidate.tetrahedra.size());
        // Admit known construction capacities before copying the four exact
        // vertices/face indices. Existing private validation/backend temporaries
        // retain their original accounting scope; this is not an RSS promise.
        budget.add(sizeof(Poly));
        budget.array(4, sizeof(Face));
        budget.array(12, sizeof(std::size_t));
        for (const auto* p : std::array<const P*, 4>{&kernel, &points[t[0]], &points[t[1]], &points[t[2]]})
        {
            departure_charge_point(*p, budget);
            departure_charge_point(*p, budget);
        }
        // Fresh tetrahedra still pass the unchanged strict private convex
        // admission. No native/nonconvex shell is asserted by this fast path.
        auto tetra = swept_tetra(kernel, points[t[0]], points[t[1]], points[t[2]]);
        if (!tetra || tetra->sign != 1)
            throw std::logic_error("Positive departure tetra construction disagrees.");
        budget.poly(SweepAccess<ArithmeticMode::Exact>::poly(*tetra->geometry));
        const auto m = exact_moments(SweepAccess<ArithmeticMode::Exact>::poly(*tetra->geometry), kernel);
        fan.volume += m.volume;
        fan.first = add(fan.first, m.first);
        departure_charge_q(fan.volume, budget);
        departure_charge_point(fan.first, budget);
        candidate.tetrahedra.push_back(std::move(tetra->geometry));
    }
    // Independent signed-pullback moments validate the set certificate; they
    // never replace closed-current, manifold, strict-kernel or disjointness proofs.
    if (fan.volume != original.volume || fan.first != original.first)
        throw std::invalid_argument("Departure fan and authenticated signed pullback moments disagree.");
    if (original.volume != native_about_kernel.volume)
        throw std::invalid_argument("Departure fan occupancy requires qualified canonical flux closure.");
    return candidate;
}

std::optional<std::vector<TransportAtom<ArithmeticMode::Exact>>> optional_departure_fan_partition(
    std::span<const SweepDonor<ArithmeticMode::Exact>> donors,
    std::span<const SweepRecipient<ArithmeticMode::Exact>> recipients, SweepBudget& budget)
{
    // Original integrity is authenticated for EVERY recipient before any
    // optional star/topology certificate can decline the route.
    for (const auto& target : recipients)
        departure_authenticate_recipient_sweeps(target, budget);
    std::vector<DepartureFanCandidate> fans;
    budget.array(recipients.size(), sizeof(DepartureFanCandidate));
    fans.reserve(recipients.size());
    bool available = true;
    for (std::size_t r = 0; r < recipients.size(); ++r)
    {
        auto fan = departure_fan_candidate(recipients[r], r, budget);
        if (!fan)
            available = false;
        else
            fans.push_back(std::move(*fan));
    }
    if (!available)
        return {}; // Attempt/source-proof work stays charged.

    // Across ALL local recipients, not merely adjacent cones of one target.
    // A positive-interior collision means this derived positive decomposition
    // is unavailable; original overlay remains the authority for that case.
    for (std::size_t r = 0; r < fans.size(); ++r)
        for (std::size_t t = 0; t < fans[r].tetrahedra.size(); ++t)
            for (std::size_t s = 0; s <= r; ++s)
                for (std::size_t u = 0; u < fans[s].tetrahedra.size() && (s < r || u < t); ++u)
                {
                    budget.test();
                    if (SweepAccess<ArithmeticMode::Exact>::intersect(
                            SweepAccess<ArithmeticMode::Exact>::certified(*fans[r].tetrahedra[t]),
                            SweepAccess<ArithmeticMode::Exact>::certified(*fans[s].tetrahedra[u]), budget))
                        return {};
                }
    std::vector<TransportAtom<ArithmeticMode::Exact>> atoms;
    budget.array(recipients.size(), sizeof(Q));
    std::vector<Q> received(recipients.size());
    for (const auto& fan : fans)
        for (const auto& tetra : fan.tetrahedra)
        {
            std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> coverage;
            budget.array(donors.size(), sizeof(ConvexFragmentPtr<ArithmeticMode::Exact>));
            coverage.reserve(donors.size());
            for (const auto& donor : donors)
            {
                budget.test();
                auto intersection =
                    SweepAccess<ArithmeticMode::Exact>::intersect(SweepAccess<ArithmeticMode::Exact>::certified(*tetra),
                        SweepAccess<ArithmeticMode::Exact>::certified(*donor.geometry), budget);
                if (!intersection)
                    continue;
                budget.poly(intersection->get());
                auto fragment = SweepAccess<ArithmeticMode::Exact>::fragment(std::move(*intersection));
                coverage.push_back(fragment);
                budget.atom(atoms.size());
                budget.add(sizeof(TransportAtom<ArithmeticMode::Exact>));
                received[fan.recipient] += sweep_volume(*fragment);
                departure_charge_q(received[fan.recipient], budget);
                departure_push(atoms,
                    TransportAtom<ArithmeticMode::Exact>{donor.id, recipients[fan.recipient].id, std::move(fragment)},
                    budget);
            }
            // Complete, contained and pairwise interior-disjoint per tetra.
            // Original donor-cover failures are errors, not optional fallbacks.
            certify_sweep_partition(SweepAccess<ArithmeticMode::Exact>::certified(*tetra), coverage, budget);
        }
    for (std::size_t r = 0; r < recipients.size(); ++r)
        if (received[r] != sweep_volume(*recipients[r].geometry))
            throw std::invalid_argument("Departure fan received occupancy differs from actual recipient volume.");
    return atoms;
}
} // namespace exact_swept_detail
} // namespace SimpleFluid::geometry::swept_volume
namespace SimpleFluid::geometry::swept_volume
{
TransportPartition<ArithmeticMode::Exact> TransportPartition<ArithmeticMode::Exact>::local(
    std::span<const SweepDonor<ArithmeticMode::Exact>> donors,
    std::span<const SweepRecipient<ArithmeticMode::Exact>> recipients, SweepLimits limits)
{
    SweepBudget budget(limits);
    budget.array(recipients.size(), sizeof(Q) + sizeof(int));
    sweep_donors(donors, budget);
    std::map<std::uint64_t, const SweepDonor<ArithmeticMode::Exact>*> originals;
    for (const auto& donor : donors)
        originals.emplace(donor.id, &donor);
    std::set<std::uint64_t> targets;
    std::map<P, P> shared;
    std::set<const SweptFace<ArithmeticMode::Exact>*> proved;
    for (const auto& target : recipients)
    {
        if (!targets.insert(target.id).second)
            throw std::invalid_argument("Duplicate recipient identity.");
        auto old = originals.find(target.id);
        if (old == originals.end() || !target.geometry || !old->second->geometry->equivalent(*target.geometry))
            throw std::invalid_argument("Transport recipient must bind its actual unchanged source cell.");
        target_sweep_shell(target, budget, shared);
        for (const auto& oriented : target.outward_faces)
            if (proved.insert(oriented.face.get()).second)
            {
                if (oriented.face->simplices().size() > limits.maximum_simplices)
                    throw std::invalid_argument("Recipient sweep exceeds simplex budget.");
                for (std::size_t simplex = 0; simplex < oriented.face->simplices().size(); ++simplex)
                {
                    const auto& piece = oriented.face->simplices()[simplex];
                    std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> coverage;
                    for (const auto& donor : donors)
                    {
                        budget.test();
                        if (auto p = SweepAccess<ArithmeticMode::Exact>::intersect(
                                SweepAccess<ArithmeticMode::Exact>::certified(*piece.geometry),
                                SweepAccess<ArithmeticMode::Exact>::certified(*donor.geometry), budget))
                        {
                            budget.poly(p->get());
                            coverage.push_back(SweepAccess<ArithmeticMode::Exact>::fragment(std::move(*p)));
                        }
                    }
                    for (std::size_t i = 0; i < coverage.size(); ++i)
                        for (std::size_t j = 0; j < i; ++j)
                            budget.test();
                    try
                    {
                        certify_sweep_partition(
                            SweepAccess<ArithmeticMode::Exact>::certified(*piece.geometry), coverage, budget);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        throw std::invalid_argument("Recipient local coverage target=" + std::to_string(target.id) +
                                                    ": " + error.what() +
                                                    sweep_cover_context(*oriented.face, piece, donors, simplex));
                    }
                }
            }
    }
    TransportPartition<ArithmeticMode::Exact> result;
    if (auto positive = optional_departure_fan_partition(donors, recipients, budget))
    {
        result.atoms_ = std::move(*positive);
        result.used_departure_fan_ = true;
        return result;
    }
    std::vector<Q> volumes(recipients.size());
    for (const auto& donor : donors)
    {
        budget.array(recipients.size(), sizeof(int));
        budget.add(sizeof(SweepAtom));
        std::vector<int> initial(recipients.size());
        for (std::size_t r = 0; r < recipients.size(); ++r)
            initial[r] = recipients[r].id == donor.id ? 1 : 0;
        budget.poly(SweepAccess<ArithmeticMode::Exact>::poly(*donor.geometry));
        std::vector<SweepAtom> atoms{
            {ConvexFragment<ArithmeticMode::Exact>::from_geometry(*donor.geometry), std::move(initial)}};
        for (std::size_t r = 0; r < recipients.size(); ++r)
            for (const auto& oriented : recipients[r].outward_faces)
                for (const auto& piece : oriented.face->simplices())
                    atoms = overlay_sweep(std::move(atoms), piece, r, -oriented.outward_sign, budget);
        for (auto& atom : atoms)
        {
            std::optional<std::size_t> owner;
            for (std::size_t r = 0; r < recipients.size(); ++r)
            {
                const auto c = atom.coefficient[r];
                if (c != 0 && c != 1)
                    throw std::invalid_argument("Closed-cell pullback has negative or repeated signed multiplicity.");
                if (c)
                {
                    if (owner)
                        throw std::invalid_argument("Closed-cell pullbacks overlap between recipients.");
                    owner = r;
                }
            }
            if (owner)
            {
                budget.atom(result.atoms_.size());
                budget.add(sizeof(TransportAtom<ArithmeticMode::Exact>));
                volumes[*owner] += sweep_volume(*atom.geometry);
                result.atoms_.push_back({donor.id, recipients[*owner].id, std::move(atom.geometry)});
            }
        }
    }
    for (std::size_t r = 0; r < recipients.size(); ++r)
        if (volumes[r] != sweep_volume(*recipients[r].geometry))
            throw std::invalid_argument("Exact recipient occupancy differs from its native volume; projected flux "
                                        "divergence needs a qualified canonical correction.");
    return result;
}

void TransportPartition<ArithmeticMode::Exact>::certify_donor_cover(const SweepDonor<ArithmeticMode::Exact>& donor,
    std::span<const TransportAtom<ArithmeticMode::Exact>> atoms, SweepLimits limits)
{
    SweepBudget budget(limits);
    if (!donor.geometry)
        throw std::invalid_argument("Missing donor coverage geometry.");
    std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> pieces;
    for (const auto& atom : atoms)
    {
        budget.atom(pieces.size());
        if (atom.donor != donor.id || !atom.geometry)
            throw std::invalid_argument("Foreign/null atom in donor coverage proof.");
        budget.add(sizeof(ConvexFragmentPtr<ArithmeticMode::Exact>));
        pieces.push_back(atom.geometry);
    }
    for (std::size_t i = 0; i < pieces.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            budget.test();
    certify_sweep_partition(SweepAccess<ArithmeticMode::Exact>::certified(*donor.geometry), pieces, budget);
}

TransportPartition<ArithmeticMode::Exact> TransportPartition<ArithmeticMode::Exact>::make(
    std::span<const SweepDonor<ArithmeticMode::Exact>> donors,
    std::span<const SweepRecipient<ArithmeticMode::Exact>> recipients, SweepLimits limits)
{
    auto result = local(donors, recipients, limits);
    for (const auto& donor : donors)
    {
        std::vector<TransportAtom<ArithmeticMode::Exact>> atoms;
        for (const auto& atom : result.atoms_)
            if (atom.donor == donor.id)
                atoms.push_back(atom);
        certify_donor_cover(donor, atoms, limits);
    }
    return result;
}

PhaseMeasures<ArithmeticMode::Exact> TransportPartition<ArithmeticMode::Exact>::phases(
    std::span<const TransportAtom<ArithmeticMode::Exact>> atoms, const GeometryHalfspace& plane)
{
    std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> pieces;
    for (const auto& atom : atoms)
    {
        if (!atom.geometry)
            throw std::invalid_argument("Null transport atom.");
        pieces.push_back(atom.geometry);
    }
    return sweep_phases(pieces, plane);
}

} // namespace SimpleFluid::geometry::swept_volume

namespace SimpleFluid::PLIC
{
template<> const void* backend_statistics_domain<ArithmeticMode::Exact>()
{
    return &exact_geometry_counts;
}
} // namespace SimpleFluid::PLIC
