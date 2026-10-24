#include "geometry/details/ConvexCellGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
std::vector<char> swept_point(const P& point)
{
    WireWriter writer;
    writer.word(0x5346504f494e5431ULL);
    for (const auto& q : point)
        writer.rational(q);
    return std::move(writer.data);
}

P sparse_support_normal(const Poly& poly, const Face& face)
{
    const auto& anchor = poly.vertices[face[0]];
    P normal{};
    for (std::size_t i = 1; i + 1 < face.size(); ++i)
    {
        normal = cross(sub(poly.vertices[face[i]], anchor), sub(poly.vertices[face[i + 1]], anchor));
        if (!zero(normal))
            return normal;
    }
    throw std::logic_error("Certified convex face lost its supporting plane.");
}

std::string cover_scalar(const Q& q, std::size_t maximum_bits)
{
    const I numerator = boost::multiprecision::numerator(q), denominator = boost::multiprecision::denominator(q);
    const I magnitude = numerator < 0 ? -numerator : numerator;
    const auto nb = magnitude == 0 ? std::size_t(0) : std::size_t(boost::multiprecision::msb(magnitude)) + 1;
    const auto db = std::size_t(boost::multiprecision::msb(denominator)) + 1;
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<long double>::max_digits10) << q.convert_to<long double>() << "{sign="
        << (q < 0      ? -1
               : q > 0 ? 1
                       : 0)
        << ",bits=" << nb << '/' << db;
    if (nb <= maximum_bits && db <= maximum_bits - nb)
        out << ",exact=" << numerator << '/' << denominator;
    out << '}';
    return out.str();
}

std::string cover_point(const P& p)
{
    return '[' + cover_scalar(p[0], 2048) + ',' + cover_scalar(p[1], 2048) + ',' + cover_scalar(p[2], 2048) + ']';
}

std::string sweep_cover_context(const ExactSweptFace& face, const SignedSweepSimplex& piece,
    std::span<const ExactSweepDonor> donors, std::size_t simplex)
{
    const auto& poly = ExactSweepAccess::poly(*piece.geometry);
    std::uint64_t face_hash = 1469598103934665603ULL;
    for (unsigned char byte : face.origin_face()->encode())
        face_hash = (face_hash ^ byte) * 1099511628211ULL;
    std::ostringstream out;
    out << " simplex=" << simplex << " sign=" << piece.sign
        << " signed_volume=" << cover_scalar(exact_moments(poly, poly.vertices[0]).volume * piece.sign)
        << " corrected_departure_center=" << cover_point(authority_point(face.corrected_departure_center()))
        << " origin_face_hash=" << face_hash << " donors=[";
    for (std::size_t i = 0; i < donors.size() && i < 128; ++i)
        out << (i ? "," : "") << donors[i].id;
    out << "] donor_count=" << donors.size() << " simplex_vertices=[";
    for (std::size_t i = 0; i < poly.vertices.size() && i < 16; ++i)
    {
        out << (i ? "," : "") << cover_point(poly.vertices[i]) << " contained_by=[";
        for (const auto& donor : donors)
        {
            bool inside = true;
            for (const auto& plane : planes(ExactSweepAccess::poly(*donor.geometry)))
                if (distance(plane, poly.vertices[i]) > 0)
                {
                    inside = false;
                    break;
                }
            if (inside)
                out << donor.id << ',';
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

// Axis bounds of admitted convex shells are attained at vertices. Equality
// means only a zero-volume contact; any positive penetration survives exactly.
// Fixed rational locals avoid retaining an additional per-fragment cache.
bool axis_separates(CertifiedPolyRef first, CertifiedPolyRef second)
{
    const auto& a = first.get().vertices;
    const auto& b = second.get().vertices;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        Q lo_a = a[0][axis], hi_a = lo_a;
        Q lo_b = b[0][axis], hi_b = lo_b;
        for (const auto& vertex : a)
        {
            lo_a = std::min(lo_a, vertex[axis]);
            hi_a = std::max(hi_a, vertex[axis]);
        }
        for (const auto& vertex : b)
        {
            lo_b = std::min(lo_b, vertex[axis]);
            hi_b = std::max(hi_b, vertex[axis]);
        }
        if (hi_a <= lo_b || hi_b <= lo_a)
            return true;
    }
    return false;
}

// Both references carry immutable strict convex-shell admission. Every face is
// a nonzero convex planar polygon with outward winding. A nonzero fan triangle
// therefore supplies the same outward supporting halfspace as its full area
// vector, without summing all polygon cross products. No derived/raw Poly can
// enter this certificate without the existing private admission bridge.
bool support_plane_separates(const P& normal, const P& anchor, const Poly& other, ClipScratch* scratch)
{
    // a lies on the nonpositive side. Convexity makes every point of b a
    // convex combination of its vertices. If all are nonnegative, the
    // intersection is confined to this exact plane and has zero volume.
    // Equality retains face/edge/vertex contacts as zero-volume contacts;
    // even an arbitrarily thin positive overlap prevents this shortcut.
    std::array<unsigned, 3> active{};
    unsigned active_count = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (normal[axis] != 0)
            active[active_count++] = axis;
    // This geometry-only summary belongs to this immutable plane query. When
    // caller scratch is present, lease its fixed storage alongside the live
    // support-plane cache. An unavailable summary simply retains the existing
    // integer/Q evaluation; no acceptance condition depends on this cache.
    std::optional<ClipCacheLease> filter_lease;
    std::optional<detail::CertifiedFilterPlane> filter;
    if (active_count > 1 && (!scratch || sizeof(detail::CertifiedFilterPlane) <= scratch->limit - scratch->used))
    {
        if (scratch)
        {
            filter_lease.emplace(*scratch);
            scratch->add(sizeof(detail::CertifiedFilterPlane));
        }
        filter.emplace(detail::certified_filter_plane(normal, anchor, Q(0)));
    }
    bool separated = true;
    for (const auto& vertex : other.vertices)
    {
        bool negative = false;
        if (active_count == 1)
        {
            // Only a sign is needed. The nonzero normal coefficient
            // cannot change equality, so compare exact coordinates without
            // constructing any rational difference, product or sum.
            const auto axis = active[0];
            negative = normal[axis] > 0 ? vertex[axis] < anchor[axis] : vertex[axis] > anchor[axis];
        }
        else
        {
            if (filter)
                if (const auto sign = detail::try_certified_support_sign(*filter, vertex))
                {
                    if (*sign < 0)
                        return false;
                    continue;
                }
            std::optional<int> integer_sign;
            if (scratch)
            {
                // Account optional integer work together with the live plane
                // cache. All integer storage dies before this lease releases,
                // and the original rational fallback runs after that release.
                ClipCacheLease lease(*scratch);
                integer_sign = detail::try_integer_support_sign(normal, anchor, vertex,
                    [&](std::size_t bytes)
                    {
                        if (bytes > scratch->limit - scratch->used)
                            return false;
                        scratch->add(bytes);
                        return true;
                    });
            }
            if (integer_sign)
                negative = *integer_sign < 0;
            else
            {
                // Original exact predicate, including all inconclusive integer
                // attempts. Zero normal components contribute exactly zero.
                const auto first = active[0];
                Q value = normal[first] * (vertex[first] - anchor[first]);
                for (unsigned term = 1; term < active_count; ++term)
                {
                    const auto axis = active[term];
                    value += normal[axis] * (vertex[axis] - anchor[axis]);
                }
                negative = value < 0;
            }
        }
        if (negative)
        {
            separated = false;
            break;
        }
    }
    return separated;
}

bool support_separates(CertifiedPolyRef boundary, CertifiedPolyRef other)
{
    const auto& a = boundary.get();
    const auto& b = other.get();
    for (const auto& face : a.faces)
        if (support_plane_separates(sparse_support_normal(a, face), a.vertices[face[0]], b))
            return true;
    return false;
}

bool support_separates(
    CertifiedPolyRef boundary, CertifiedPolyRef other, std::span<const Plane> support_planes, ClipScratch& scratch)
{
    const auto& a = boundary.get();
    const auto& b = other.get();
    if (support_planes.size() != a.faces.size())
        throw std::logic_error("Certified support-plane cache does not match its immutable shell.");
    for (const auto& plane : support_planes)
    {
        if (zero(plane.normal))
            throw std::logic_error("Certified convex face lost its supporting plane.");
        if (support_plane_separates(plane.normal, plane.anchor, b, &scratch))
            return true;
    }
    return false;
}

bool cached_pair_intersects(CertifiedPolyRef first, std::size_t first_index, CertifiedPolyRef second,
    std::size_t second_index, const SupportPlaneCache& cache, ClipScratch& scratch)
{
    if (axis_separates(first, second) || support_separates(first, second, cache.for_piece(first_index), scratch) ||
        support_separates(second, first, cache.for_piece(second_index), scratch))
        return false;
    return certified_intersection(first, cache.for_piece(second_index), scratch).has_value();
}

// Only immutable admitted convex shells can enter this predicate. For each
// outward convex planar face, its first nonzero fan triangle is a positive
// multiple of the full area normal. It therefore defines exactly the same
// closed halfspace, including equality and arbitrarily thin outside slivers.
// Construct one supporting normal per face, then test every piece vertex;
// avoid rebuilding and summing the donor's polygon normals for each piece.
bool certified_cover_contained(std::span<const ExactConvexFragmentPtr> pieces, CertifiedPolyRef outer)
{
    const auto& b = outer.get();
    for (const auto& piece : pieces)
        if (!piece)
            return false;
    for (const auto& face : b.faces)
    {
        const auto& anchor = b.vertices[face[0]];
        P normal{};
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            normal = cross(sub(b.vertices[face[i]], anchor), sub(b.vertices[face[i + 1]], anchor));
            if (!zero(normal))
                break;
        }
        if (zero(normal))
            throw std::logic_error("Certified convex face lost its containment plane.");
        const Plane plane{std::move(normal), anchor};
        for (const auto& piece : pieces)
            for (const auto& vertex : ExactSweepAccess::certified(*piece).get().vertices)
                if (distance(plane, vertex) > 0)
                    return false;
    }
    return true;
}

void certify_sweep_partition(
    CertifiedPolyRef expected_certificate, std::span<const ExactConvexFragmentPtr> pieces, SweepBudget& budget)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.cover_certifications;
    const auto& expected = expected_certificate.get();
    const auto origin = expected.vertices[0];
    const auto whole = exact_moments(expected, origin);
    ExactMoments sum{};
    if (!certified_cover_contained(pieces, expected_certificate))
        throw std::invalid_argument("Exact signed fragment lies outside its actual donor.");
    ClipScratch support_scratch{budget.limits.byte_budget - budget.bytes};
    std::optional<SupportPlaneCache> support_cache;
    std::string support_cache_failure;
    if (pieces.size() > 1)
    {
        try
        {
            support_cache.emplace(pieces, support_scratch);
        }
        catch (const ClipCacheExhausted& error)
        {
            // The support cache is optional scratch only. Its lease is fully
            // released by the failed constructor before the unchanged
            // predicate/intersection path is selected below.
            if (support_scratch.fallbacks != std::numeric_limits<std::size_t>::max())
                ++support_scratch.fallbacks;
            support_cache_failure = error.what();
            if (support_scratch.used)
                throw std::logic_error("Exact support-plane cache lease leaked during construction failure.");
        }
    }
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
        const auto part = exact_moments(ExactSweepAccess::poly(*pieces[i]), origin);
        sum.volume += part.volume;
        sum.first = add(sum.first, part.first);
        for (std::size_t j = 0; j < i; ++j)
        {
            budget.test();
            const auto a = ExactSweepAccess::certified(*pieces[i]);
            const auto b = ExactSweepAccess::certified(*pieces[j]);
            const bool intersects = support_cache ? cached_pair_intersects(a, i, b, j, *support_cache, support_scratch)
                                                  : ExactSweepAccess::intersect(a, b, budget).has_value();
            if (intersects)
            {
                const bool cache_active = support_cache.has_value();
                const auto cache_peak = support_scratch.peak;
                const auto cache_fallbacks = support_scratch.fallbacks;
                support_cache.reset();
                if (support_scratch.used)
                    throw std::logic_error("Exact support-plane cache lease leaked before overlap rejection.");
                std::ostringstream detail;
                detail << "Exact signed fragment cover overlaps. support_cache="
                       << (cache_active ? "cached" : "fallback") << " support_cache_peak=" << cache_peak
                       << " support_cache_fallbacks=" << cache_fallbacks;
                if (!support_cache_failure.empty())
                    detail << " support_cache_failure=\"" << support_cache_failure << '"';
                throw std::invalid_argument(detail.str());
            }
        }
    }
    if (sum.volume != whole.volume || sum.first != whole.first)
    {
        std::ostringstream detail;
        detail << "Exact signed fragment cover has a real volume or first-moment gap. expected_volume="
               << cover_scalar(whole.volume) << " covered_volume=" << cover_scalar(sum.volume)
               << " missing_volume=" << cover_scalar(whole.volume - sum.volume)
               << " missing_first_local=" << cover_point(sub(whole.first, sum.first))
               << " moment_anchor=" << cover_point(origin) << " pieces=" << pieces.size();
        throw std::invalid_argument(detail.str());
    }
    support_cache.reset();
    if (support_scratch.used)
        throw std::logic_error("Exact support-plane cache lease leaked before certification completed.");
    budget.cache(support_scratch);
}

std::optional<SignedSweepSimplex> swept_tetra(const P& origin, const P& a, const P& b, const P& c)
{
    const Q signed_volume = dot(sub(a, origin), cross(sub(b, origin), sub(c, origin))) / 6;
    if (signed_volume == 0)
        return {};
    Poly p;
    p.vertices = {origin, a, b, c};
    p.faces = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const int sign = signed_volume > 0 ? 1 : -1;
    if (sign < 0)
        for (auto& face : p.faces)
            std::reverse(face.begin(), face.end());
    return SignedSweepSimplex{sign, ExactSweepAccess::fragment(std::move(p))};
}

std::vector<SweptTriangle> swept_boundary(const Polygon& current, const Polygon& previous, const P& center,
    const P& backcenter, std::vector<std::array<std::vector<char>, 3>>* sides)
{
    std::vector<SweptTriangle> result;
    result.reserve(4 * current.size());
    for (std::size_t i = 0; i < current.size(); ++i)
    {
        const auto j = (i + 1) % current.size();
        result.push_back({center, current[i], current[j]});
        result.push_back({backcenter, previous[j], previous[i]});
        std::array<SweptTriangle, 2> edge;
        if (swept_point(current[i]) < swept_point(current[j]))
            edge = {SweptTriangle{current[i], previous[i], previous[j]},
                SweptTriangle{current[i], previous[j], current[j]}};
        else
            edge = {SweptTriangle{current[j], current[i], previous[i]},
                SweptTriangle{current[j], previous[i], previous[j]}};
        for (const auto& triangle : edge)
        {
            result.push_back(triangle);
            if (sides)
                sides->push_back({swept_point(triangle[0]), swept_point(triangle[1]), swept_point(triangle[2])});
        }
    }
    return result;
}

Q signed_boundary_volume(const P& origin, const std::vector<SweptTriangle>& triangles)
{
    Q volume = 0;
    for (const auto& t : triangles)
        volume += dot(sub(t[0], origin), cross(sub(t[1], origin), sub(t[2], origin))) / 6;
    return volume;
}

void sweep_donors(std::span<const ExactSweepDonor> donors, SweepBudget& budget)
{
    std::set<std::uint64_t> ids;
    for (std::size_t i = 0; i < donors.size(); ++i)
    {
        if (!donors[i].geometry || !ids.insert(donors[i].id).second)
            throw std::invalid_argument("Exact sweep donor geometry/identity is missing or duplicated.");
        budget.add(sizeof(ExactSweepDonor));
        for (std::size_t j = 0; j < i; ++j)
        {
            budget.test();
            if (ExactSweepAccess::intersect(ExactSweepAccess::certified(*donors[i].geometry),
                    ExactSweepAccess::certified(*donors[j].geometry), budget))
                throw std::invalid_argument("Exact sweep donor interiors overlap.");
        }
    }
}

std::vector<SweepAtom> overlay_sweep(
    std::vector<SweepAtom> atoms, const SignedSweepSimplex& piece, std::size_t column, int sign, SweepBudget& budget)
{
    std::vector<SweepAtom> next;
    for (auto& atom : atoms)
    {
        budget.test();
        const auto& a = ExactSweepAccess::poly(*atom.geometry);
        const auto& b = ExactSweepAccess::poly(*piece.geometry);
        if (!ExactSweepAccess::intersect(
                ExactSweepAccess::certified(*atom.geometry), ExactSweepAccess::certified(*piece.geometry), budget))
        {
            budget.atom(next.size());
            next.push_back(std::move(atom));
            continue;
        }
        if (contained(a, b))
        {
            atom.coefficient.at(column) += sign * piece.sign;
            budget.atom(next.size());
            next.push_back(std::move(atom));
            continue;
        }
        std::optional<CertifiedPoly> remaining = ExactSweepAccess::copy(ExactSweepAccess::certified(*atom.geometry));
        for (const auto& plane : planes(b))
        {
            Plane other = plane;
            other.normal = mul(other.normal, Q(-1));
            other.offset = -other.offset;
            auto outside = ExactSweepAccess::clip(remaining->ref(), other, budget);
            if (outside)
            {
                budget.atom(next.size());
                budget.poly(outside->get());
                budget.add(atom.coefficient.size() * sizeof(int) + sizeof(SweepAtom));
                next.push_back({ExactSweepAccess::fragment(std::move(*outside)), atom.coefficient});
            }
            remaining = ExactSweepAccess::clip(remaining->ref(), plane, budget);
            if (!remaining)
                break;
        }
        if (remaining)
        {
            budget.atom(next.size());
            budget.poly(remaining->get());
            auto coefficient = atom.coefficient;
            coefficient.at(column) += sign * piece.sign;
            budget.add(coefficient.size() * sizeof(int) + sizeof(SweepAtom));
            next.push_back({ExactSweepAccess::fragment(std::move(*remaining)), std::move(coefficient)});
        }
    }
    return next;
}

ExactPhaseMeasures sweep_phases(std::span<const ExactConvexFragmentPtr> atoms, const GeometryHalfspace& halfspace)
{
    ExactPhaseMeasures result;
    Plane plane{point(halfspace.normal), point(halfspace.anchor), exact(halfspace.offset)};
    if (zero(plane.normal))
        throw std::invalid_argument("Phase plane normal must be nonzero.");
    Plane outside = plane;
    outside.normal = mul(outside.normal, Q(-1));
    outside.offset = -outside.offset;
    for (const auto& atom : atoms)
    {
        result.bulk = result.bulk + atom->volume();
        if (auto cut = certified_clip_uncached(ExactSweepAccess::certified(*atom), plane))
            result.gas = result.gas + ExactSweepAccess::measure(exact_volume(cut->get(), cut->get().vertices[0]));
        if (auto cut = certified_clip_uncached(ExactSweepAccess::certified(*atom), outside))
            result.carrier =
                result.carrier + ExactSweepAccess::measure(exact_volume(cut->get(), cut->get().vertices[0]));
    }
    if ((result.gas + result.carrier).compare(result.bulk) != 0)
        throw std::logic_error("Exact independent phase clips lost complementarity.");
    return result;
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
ExactConvexFragment::ExactConvexFragment(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

ExactConvexFragmentPtr ExactConvexFragment::from_geometry(const ConvexCellGeometry& cell)
{
    return ExactSweepAccess::fragment(ExactSweepAccess::certified(cell));
}

ExactGeometryMeasure ExactConvexFragment::volume() const
{
    return ExactSweepAccess::measure(sweep_volume(*this));
}

ExactPhaseMeasures ExactConvexFragment::phases(const GeometryHalfspace& plane) const
{
    const std::array<ExactConvexFragmentPtr, 1> fragment{ExactConvexFragmentPtr(new ExactConvexFragment(impl_))};
    return sweep_phases(fragment, plane);
}

ExactConvexFragmentPtr ExactConvexFragment::translated_exact(std::span<const char> displacement) const
{
    auto poly = impl_->poly.get();
    const auto shift = displacement_point(displacement);
    for (auto& p : poly.vertices)
        p = add(p, shift);
    return ExactSweepAccess::fragment(std::move(poly));
}

ConvexGeometryPtr ExactConvexFragment::as_geometry() const
{
    return ExactSweepAccess::geometry(impl_->poly.get());
}

std::vector<char> ExactConvexFragment::encode(std::size_t byte_budget) const
{
    SweepBudget budget({4096, 65536, 262144, byte_budget});
    budget.add(56);
    for (const auto& vertex : impl_->poly.get().vertices)
        for (const auto& q : vertex)
            budget.add(40 + (q.backend().num().capacity() + q.backend().denom().capacity()) *
                                sizeof(boost::multiprecision::limb_type));
    for (const auto& face : impl_->poly.get().faces)
        budget.add(8 + face.size() * 8);
    WireWriter w;
    w.word(0x5346455841435431ULL);
    w.word(1);
    for (unsigned a = 0; a < 3; ++a)
        w.real(0);
    w.word(impl_->poly.get().vertices.size());
    for (const auto& p : impl_->poly.get().vertices)
        for (const auto& q : p)
            w.rational(q);
    w.word(impl_->poly.get().faces.size());
    for (const auto& face : impl_->poly.get().faces)
    {
        w.word(face.size());
        for (auto v : face)
            w.word(v);
    }
    if (w.data.size() > byte_budget)
        throw std::invalid_argument("Exact fragment packet exceeds byte budget.");
    return std::move(w.data);
}

ExactConvexFragmentPtr ExactConvexFragment::decode(std::span<const char> data, std::size_t byte_budget)
{
    if (data.size() > byte_budget)
        throw std::invalid_argument("Exact fragment packet exceeds byte budget.");
    SweepBudget budget({4096, 65536, 262144, byte_budget});
    WireReader r{data};
    if (r.word() != 0x5346455841435431ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact fragment tag/version.");
    GeometryPoint anchor{};
    for (auto& q : anchor)
        q = r.real();
    const auto origin = point(anchor);
    Poly poly;
    auto count = r.count();
    budget.array(count, sizeof(P));
    poly.vertices.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        P p;
        for (auto& q : p)
        {
            q = r.rational();
            budget.add((q.backend().num().capacity() + q.backend().denom().capacity()) *
                       sizeof(boost::multiprecision::limb_type));
        }
        poly.vertices.push_back(add(p, origin));
    }
    count = r.count();
    budget.array(count, sizeof(Face));
    poly.faces.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        Face face;
        auto n = r.count();
        budget.array(n, sizeof(std::size_t));
        face.reserve(n);
        for (std::size_t j = 0; j < n; ++j)
            face.push_back(r.word());
        poly.faces.push_back(std::move(face));
    }
    r.finish();
    return ExactSweepAccess::fragment(std::move(poly));
}

ExactGeometryMeasure::ExactGeometryMeasure() : impl_(std::make_shared<const Impl>()) {}

ExactGeometryMeasure::ExactGeometryMeasure(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

ExactGeometryMeasure ExactGeometryMeasure::from_value(double value)
{
    return ExactSweepAccess::measure(exact(value));
}

ExactGeometryMeasure exact_point_projection(std::span<const char> identity, GeometryPoint normal, GeometryPoint anchor)
{
    return ExactSweepAccess::measure(dot(point(normal), sub(authority_point(identity), point(anchor))));
}

ExactGeometryMeasure ExactGeometryMeasure::volume(const ConvexCellGeometry& cell)
{
    return ExactSweepAccess::measure(sweep_volume(cell));
}

CertifiedGeometryValue ExactGeometryMeasure::certified() const
{
    return measured(impl_->value);
}

CertifiedGeometryValue ExactGeometryMeasure::enclosure() const
{
    return measured(impl_->value, true);
}

bool ExactGeometryMeasure::relative_error_at_most(const ExactGeometryMeasure& reference, double tolerance) const
{
    const auto factor = exact(tolerance);
    if (factor < 0)
        throw std::invalid_argument("Relative measure tolerance must be nonnegative.");
    Q difference = impl_->value - reference.impl_->value;
    if (difference < 0)
        difference = -difference;
    Q scale = reference.impl_->value;
    if (scale < 0)
        scale = -scale;
    return difference <= factor * scale;
}

double ExactGeometryMeasure::log_ratio(const ExactGeometryMeasure& denominator) const
{
    if (impl_->value <= 0 || denominator.impl_->value <= 0)
        throw std::invalid_argument("Logarithmic measure ratio requires positive operands.");
    const Q ratio = impl_->value / denominator.impl_->value;
    const auto numerator = boost::multiprecision::numerator(ratio), divisor = boost::multiprecision::denominator(ratio);
    const std::int64_t exponent = static_cast<std::int64_t>(boost::multiprecision::msb(numerator)) -
                                  static_cast<std::int64_t>(boost::multiprecision::msb(divisor));
    const Q power = exponent >= 0 ? Q(I{1} << exponent) : Q(1, I{1} << (-exponent));
    const double mantissa = Q(ratio / power).convert_to<double>();
    const double result = std::log(mantissa) + static_cast<double>(exponent) * std::log(2.0);
    if (!std::isfinite(result))
        throw std::invalid_argument("Logarithmic measure ratio is outside interpolation range.");
    return result;
}

std::vector<char> ExactGeometryMeasure::encode(std::size_t byte_budget) const
{
    const auto& q = impl_->value;
    const auto limbs = q.backend().num().capacity() + q.backend().denom().capacity();
    if (byte_budget < 56 || limbs > (byte_budget - 56) / sizeof(boost::multiprecision::limb_type))
        throw std::invalid_argument("Exact measure certificate exceeds byte budget.");
    WireWriter w;
    w.word(0x53464d4541533731ULL);
    w.word(1);
    w.rational(q);
    if (w.data.size() > byte_budget)
        throw std::invalid_argument("Exact measure certificate exceeds byte budget.");
    return std::move(w.data);
}

ExactGeometryMeasure ExactGeometryMeasure::decode(std::span<const char> data, std::size_t byte_budget)
{
    if (data.size() > byte_budget)
        throw std::invalid_argument("Exact measure certificate exceeds byte budget.");
    WireReader r{data};
    if (r.word() != 0x53464d4541533731ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact measure tag/version.");
    auto value = r.rational();
    r.finish();
    return ExactSweepAccess::measure(std::move(value));
}

int ExactGeometryMeasure::sign() const
{
    return impl_->value > 0 ? 1 : impl_->value < 0 ? -1 : 0;
}

int ExactGeometryMeasure::compare(const ExactGeometryMeasure& other) const
{
    return impl_->value > other.impl_->value ? 1 : impl_->value < other.impl_->value ? -1 : 0;
}

ExactGeometryMeasure ExactGeometryMeasure::operator+(const ExactGeometryMeasure& other) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    return ExactSweepAccess::measure(impl_->value + other.impl_->value);
}

ExactGeometryMeasure ExactGeometryMeasure::operator-(const ExactGeometryMeasure& other) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    return ExactSweepAccess::measure(impl_->value - other.impl_->value);
}

ExactGeometryMeasure ExactGeometryMeasure::scaled(double factor) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    return ExactSweepAccess::measure(impl_->value * exact(factor));
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
std::size_t exact_integer_bits(I value)
{
    if (value < 0)
        value = -value;
    return value == 0 ? 0 : static_cast<std::size_t>(boost::multiprecision::msb(value)) + 1;
}

void exact_measure_arithmetic_preflight(const Q& a, const Q& b, bool division, std::size_t budget)
{
    const auto overhead = sizeof(ExactGeometryMeasure) + sizeof(ExactGeometryMeasure::Impl);
    if (budget < overhead)
        throw std::invalid_argument("Exact measure arithmetic result exceeds byte budget.");
    const auto available = budget - overhead;
    if (available > std::numeric_limits<std::size_t>::max() / 8)
        throw std::invalid_argument("Exact measure arithmetic budget overflows bit count.");
    const auto limit = available * 8;
    const auto an = exact_integer_bits(boost::multiprecision::numerator(a)),
               ad = exact_integer_bits(boost::multiprecision::denominator(a));
    const auto bn = exact_integer_bits(boost::multiprecision::numerator(b)),
               bd = exact_integer_bits(boost::multiprecision::denominator(b));
    const auto first = division ? bd : bn, second = division ? bn : bd;
    if (an > limit || first > limit - an)
        throw std::invalid_argument("Exact measure arithmetic numerator exceeds bit budget.");
    const auto numerator = an + first;
    if (ad > limit - numerator || second > limit - numerator - ad)
        throw std::invalid_argument("Exact measure arithmetic denominator exceeds bit budget.");
}

void exact_storage_add(std::size_t& bytes, std::size_t count, std::size_t size)
{
    if (size && count > (std::numeric_limits<std::size_t>::max() - bytes) / size)
        throw std::overflow_error("Exact storage capacity count overflow.");
    bytes += count * size;
}

void exact_scalar_storage(std::size_t& bytes, const Q& q)
{
    exact_storage_add(bytes, q.backend().num().capacity(), sizeof(boost::multiprecision::limb_type));
    exact_storage_add(bytes, q.backend().denom().capacity(), sizeof(boost::multiprecision::limb_type));
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
ExactGeometryMeasure ExactGeometryMeasure::divided_by(
    const ExactGeometryMeasure& denominator, std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    if (denominator.impl_->value == 0)
        throw std::invalid_argument("Exact measure division by zero.");
    exact_measure_arithmetic_preflight(impl_->value, denominator.impl_->value, true, byte_budget);
    auto result = ExactSweepAccess::measure(impl_->value / denominator.impl_->value);
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact measure quotient capacity exceeds byte budget.");
    return result;
}

ExactGeometryMeasure ExactGeometryMeasure::squared(std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    exact_measure_arithmetic_preflight(impl_->value, impl_->value, false, byte_budget);
    auto result = ExactSweepAccess::measure(impl_->value * impl_->value);
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact measure squared capacity exceeds byte budget.");
    return result;
}

std::size_t ExactGeometryMeasure::storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + sizeof(Impl);
    exact_scalar_storage(bytes, impl_->value);
    return bytes;
}

ExactGeometryMeasure ExactGeometryMeasure::compacted(std::size_t operation_byte_budget) const
{
    // A fresh cpp_int destination starts with inline limbs. Its checked copy
    // resize uses max(4*initial_capacity, used_limbs) when those limbs overflow.
    // Charge both immutable source and complete fresh destination before copying.
    std::size_t copied = sizeof(*this) + sizeof(Impl);
    const auto charge = [&](const auto& source, const auto& destination)
    {
        const auto initial = destination.capacity();
        if (initial > std::numeric_limits<std::size_t>::max() / 4)
            throw std::invalid_argument("Exact measure compact copy capacity overflows.");
        const auto capacity = source.size() <= initial ? initial : std::max(4 * initial, source.size());
        exact_storage_add(copied, capacity, sizeof(boost::multiprecision::limb_type));
    };
    {
        const Q fresh;
        charge(impl_->value.backend().num(), fresh.backend().num());
        charge(impl_->value.backend().denom(), fresh.backend().denom());
    }
    const auto source_bytes = storage_bytes();
    if (source_bytes > operation_byte_budget || copied > operation_byte_budget - source_bytes)
        throw std::invalid_argument("Exact measure compact copy exceeds its simultaneous storage budget.");
    auto compact = std::make_shared<Impl>();
    compact->value = impl_->value;
    ExactGeometryMeasure result(std::move(compact));
    if (result.impl_->value != impl_->value || result.storage_bytes() > copied ||
        result.storage_bytes() > operation_byte_budget - source_bytes)
        throw std::invalid_argument("Exact measure compact copy failed its exact value or capacity certificate.");
    return result;
}

ExactGeometryMeasure ExactGeometryMeasure::rounded_dyadic(
    unsigned significant_bits, std::size_t operation_byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    if (significant_bits < 2)
        throw std::invalid_argument("Exact dyadic representation needs at least two significand bits.");
    const auto source_bytes = storage_bytes();
    const auto precision_bytes = std::size_t(significant_bits / 8) + (significant_bits % 8 != 0);
    // The largest shifted integer has at most max(numerator,denominator) bits
    // plus the requested precision. Reserve all copies, quotient, dyadic power,
    // rational normalization and conservative backend growth before allocating.
    if (source_bytes > operation_byte_budget / 16 || precision_bytes > operation_byte_budget / 16 - source_bytes)
        throw std::length_error("Exact dyadic representation exceeds its operation lease.");
    if (!sign())
        return from_value(0);
    I numerator = boost::multiprecision::numerator(impl_->value);
    const bool negative = numerator < 0;
    if (negative)
        numerator = -numerator;
    I denominator = boost::multiprecision::denominator(impl_->value);
    const auto numerator_bits = exact_integer_bits(numerator), denominator_bits = exact_integer_bits(denominator);
    if (numerator_bits > std::size_t(std::numeric_limits<std::int64_t>::max()) ||
        denominator_bits > std::size_t(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("Exact dyadic exponent exceeds the signed index range.");
    const auto exponent = std::int64_t(numerator_bits) - std::int64_t(denominator_bits);
    if (exponent < std::int64_t(significant_bits) - 1 - std::numeric_limits<std::int64_t>::max())
        throw std::invalid_argument("Exact dyadic shift exceeds the signed index range.");
    const auto shift = std::int64_t(significant_bits) - 1 - exponent;
    if (shift >= 0)
        numerator <<= std::size_t(shift);
    else
        denominator <<= std::size_t(-shift);
    I significand = numerator / denominator;
    if (negative)
        significand = -significand;
    Q rounded;
    if (shift >= 0)
        rounded = Q(significand, I{1} << std::size_t(shift));
    else
        rounded = Q(significand << std::size_t(-shift));
    auto result = ExactSweepAccess::measure(std::move(rounded));
    if (result.sign() != sign())
        throw std::logic_error("Exact dyadic representation failed its nonzero sign certificate.");
    if (result.storage_bytes() > operation_byte_budget / 16)
        throw std::length_error("Exact dyadic representation exceeds its retained storage certificate.");
    return result;
}

std::size_t ExactConvexFragment::storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + sizeof(Impl);
    exact_storage_add(bytes, impl_->poly.get().vertices.capacity(), sizeof(P));
    exact_storage_add(bytes, impl_->poly.get().faces.capacity(), sizeof(Face));
    for (const auto& point : impl_->poly.get().vertices)
        for (const auto& q : point)
            exact_scalar_storage(bytes, q);
    for (const auto& face : impl_->poly.get().faces)
        exact_storage_add(bytes, face.capacity(), sizeof(std::size_t));
    return bytes;
}

ExactNodeTrajectory ExactNodeTrajectory::backtrace(
    std::span<const char> identity, GeometryPoint velocity, double timestep)
{
    if (!(timestep >= 0) || !std::isfinite(timestep))
        throw std::invalid_argument("Backward trajectory needs finite nonnegative timestep.");
    const auto p = authority_point(identity);
    const P exact_velocity{exact(velocity[0]), exact(velocity[1]), exact(velocity[2])};
    const auto back = sub(p, mul(exact_velocity, exact(timestep)));
    return {swept_point(p), swept_point(back)};
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
bool trajectory_zero(const P& p)
{
    return p[0] == 0 && p[1] == 0 && p[2] == 0;
}

P bounded_polygon_vector(const Polygon& polygon, TrajectoryArithmetic& arithmetic)
{
    P result{};
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        const auto area = arithmetic.cross_product(
            arithmetic.subtract(polygon[i], polygon[0]), arithmetic.subtract(polygon[i + 1], polygon[0]));
        for (unsigned axis = 0; axis < 3; ++axis)
            result[axis] = arithmetic.binary(result[axis], area[axis], 0);
    }
    return arithmetic.scale(result, Q(1, 2));
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
ExactGeometryMeasure ExactPlanarFace::coplanar_area_ratio(
    const ExactPlanarFace& denominator, std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    ExactTrajectoryLimits limits;
    limits.byte_budget = byte_budget;
    TrajectoryArithmetic arithmetic(limits);
    const auto support = bounded_polygon_vector(denominator.impl_->polygon, arithmetic);
    unsigned axis = 0;
    while (axis < 3 && support[axis] == 0)
        ++axis;
    if (axis == 3)
        throw std::invalid_argument("Exact face area denominator is degenerate.");
    for (const auto& point : impl_->polygon)
        if (arithmetic.inner(support, arithmetic.subtract(point, denominator.impl_->polygon[0])) != 0)
            throw std::invalid_argument("Exact face area ratio requires identical support planes.");
    const auto area = bounded_polygon_vector(impl_->polygon, arithmetic);
    auto ratio = arithmetic.binary(area[axis], support[axis], 3);
    if (ratio < 0)
        ratio = arithmetic.binary(Q(0), ratio, 1);
    if (ratio == 0)
        throw std::invalid_argument("Exact face area numerator is degenerate.");
    auto result = ExactSweepAccess::measure(std::move(ratio));
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact face area ratio result exceeds byte budget.");
    return result;
}

ExactGeometryMeasure ExactPlanarFace::area_vector_dot(GeometryPoint velocity, std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    ExactTrajectoryLimits limits;
    limits.byte_budget = byte_budget;
    TrajectoryArithmetic arithmetic(limits);
    const P represented{arithmetic.retain(exact(velocity[0])), arithmetic.retain(exact(velocity[1])),
        arithmetic.retain(exact(velocity[2]))};
    const auto area = bounded_polygon_vector(impl_->polygon, arithmetic);
    auto result = ExactSweepAccess::measure(arithmetic.inner(area, represented));
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact face area flux result exceeds byte budget.");
    return result;
}

ExactNodeTrajectory ExactNodeTrajectory::backtrace_constrained(std::span<const char> identity, GeometryPoint velocity,
    double timestep, std::span<const ExactPlanarFacePtr> walls, ExactTrajectoryLimits limits)
{
    TrajectoryArithmetic arithmetic(limits);
    if (!std::isfinite(timestep) || timestep < 0 || walls.size() > limits.maximum_constraints)
        throw std::invalid_argument("Invalid exact wall trajectory timestep or constraint count.");
    arithmetic.charge(identity.size());
    const auto source = authority_point(identity);
    for (const auto& q : source)
        (void) arithmetic.bits(q);
    const P exact_velocity{arithmetic.retain(exact(velocity[0])), arithmetic.retain(exact(velocity[1])),
        arithmetic.retain(exact(velocity[2]))};
    const auto original = arithmetic.scale(exact_velocity, arithmetic.retain(exact(timestep)));
    if (walls.size() > limits.byte_budget / sizeof(P))
        throw std::invalid_argument("Exact trajectory constraint storage exceeds budget.");
    arithmetic.charge(walls.size() * sizeof(P));
    std::vector<P> normals;
    normals.reserve(walls.size());
    for (const auto& wall : walls)
    {
        if (!wall)
            throw std::invalid_argument("Null exact wall constraint.");
        const auto& polygon = ExactSweepAccess::polygon(*wall);
        P normal{};
        for (std::size_t i = 1; i + 1 < polygon.size() && trajectory_zero(normal); ++i)
            normal = arithmetic.cross_product(
                arithmetic.subtract(polygon[i], polygon[0]), arithmetic.subtract(polygon[i + 1], polygon[0]));
        if (trajectory_zero(normal) || arithmetic.inner(normal, arithmetic.subtract(source, polygon[0])) != 0)
            throw std::invalid_argument("Trajectory source is not on the exact wall plane.");
        for (std::size_t i = 0; i < polygon.size(); ++i)
            if (arithmetic.inner(
                    normal, arithmetic.cross_product(arithmetic.subtract(polygon[(i + 1) % polygon.size()], polygon[i]),
                                arithmetic.subtract(source, polygon[i]))) < 0)
                throw std::invalid_argument("Trajectory source is outside the closed wall polygon.");
        unsigned pivot = 0;
        while (normal[pivot] == 0)
            ++pivot;
        const auto divisor = normal[pivot];
        for (auto& q : normal)
            q = arithmetic.binary(q, divisor, 3);
        normals.push_back(std::move(normal));
    }
    std::sort(normals.begin(), normals.end());
    normals.erase(std::unique(normals.begin(), normals.end()), normals.end());
    std::vector<P> basis;
    basis.reserve(3);
    for (const auto& normal : normals)
    {
        auto candidate = normal;
        for (const auto& b : basis)
            candidate = arithmetic.subtract(candidate,
                arithmetic.scale(b, arithmetic.binary(arithmetic.inner(candidate, b), arithmetic.inner(b, b), 3)));
        if (!trajectory_zero(candidate))
            basis.push_back(std::move(candidate));
        if (basis.size() > 3)
            throw std::logic_error("Exact wall normal rank exceeds three.");
    }
    auto displacement = original;
    for (const auto& b : basis)
        displacement = arithmetic.subtract(displacement,
            arithmetic.scale(b, arithmetic.binary(arithmetic.inner(original, b), arithmetic.inner(b, b), 3)));
    for (const auto& normal : normals)
        if (arithmetic.inner(normal, displacement) != 0)
            throw std::logic_error("Exact constrained trajectory failed tangency.");
    if (arithmetic.inner(displacement, displacement) > arithmetic.inner(original, original))
        throw std::logic_error("Exact constrained trajectory increased displacement norm.");
    const auto departure = arithmetic.subtract(source, displacement);
    const auto encoded_bound = [&](const P& value)
    {
        std::size_t count = 8;
        for (const auto& q : value)
        {
            const auto bits = arithmetic.bits(q);
            count += 32 + (bits[0] + 7) / 8 + (bits[1] + 7) / 8 + 2;
        }
        return 2 * count; // vector growth and zero-integer framing allowance
    };
    arithmetic.charge(encoded_bound(source));
    arithmetic.charge(encoded_bound(departure));
    auto start_bytes = swept_point(source), end_bytes = swept_point(departure);
    return {std::move(start_bytes), std::move(end_bytes)};
}

ExactGeometryMeasure ExactNodeTrajectory::displacement_squared(ExactTrajectoryLimits limits) const
{
    TrajectoryArithmetic arithmetic(limits);
    arithmetic.charge(point.size());
    arithmetic.charge(departure.size());
    const auto displacement = arithmetic.subtract(authority_point(point), authority_point(departure));
    return ExactSweepAccess::measure(arithmetic.inner(displacement, displacement));
}

ExactSweptFace::ExactSweptFace(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

std::shared_ptr<const ExactSweptFace> ExactSweptFace::make(
    const ExactPlanarFace& face, std::span<const ExactNodeTrajectory> traces, double volume, ExactSweepLimits limits)
{
    return make(face, traces, ExactGeometryMeasure::from_value(volume), limits);
}

std::shared_ptr<const ExactSweptFace> ExactSweptFace::make(const ExactPlanarFace& face,
    std::span<const ExactNodeTrajectory> traces, const ExactGeometryMeasure& volume, ExactSweepLimits limits)
{
    SweepBudget budget(limits);
    const auto& current = ExactSweepAccess::polygon(face);
    if (current.size() > limits.maximum_simplices / 4)
        throw std::invalid_argument("Exact sweep simplex budget exhausted before construction.");
    if (traces.size() != current.size())
        throw std::invalid_argument("Every canonical face node needs exactly one shared trajectory.");
    std::map<P, P> mapped;
    for (const auto& trace : traces)
    {
        budget.add(trace.point.size() + trace.departure.size() + sizeof(ExactNodeTrajectory));
        if (!mapped.emplace(authority_point(trace.point), authority_point(trace.departure)).second)
            throw std::invalid_argument("Duplicate shared node trajectory.");
    }
    Polygon previous;
    P center{}, backcenter{};
    for (const auto& p : current)
    {
        auto found = mapped.find(p);
        if (found == mapped.end())
            throw std::invalid_argument("Shared trajectory does not bind canonical exact face point.");
        previous.push_back(found->second);
        center = add(center, p);
        backcenter = add(backcenter, found->second);
    }
    center = mul(center, Q(1) / current.size());
    backcenter = mul(backcenter, Q(1) / current.size());
    const auto original_triangles = swept_boundary(current, previous, center, backcenter, nullptr);
    const auto original = signed_boundary_volume(center, original_triangles);
    const auto requested = ExactSweepAccess::value(volume);
    const auto normal = polygon_vector(current);
    const Q coefficient =
        signed_boundary_volume(center, swept_boundary(current, previous, center, add(backcenter, normal), nullptr)) -
        original;
    if (requested != original)
    {
        if (coefficient == 0)
            throw std::invalid_argument("Projected face correction has no finite exact interior degree of freedom.");
        backcenter = add(backcenter, mul(normal, Q((requested - original) / coefficient)));
    }
    auto result = std::make_shared<Impl>();
    result->volume = requested;
    result->original = original;
    result->center = backcenter;
    result->native = ExactSweepAccess::retain(face);
    result->traces.assign(traces.begin(), traces.end());
    auto triangles = swept_boundary(current, previous, center, backcenter, &result->edges);
    Q check = 0;
    for (const auto& triangle : triangles)
        if (auto piece = swept_tetra(center, triangle[0], triangle[1], triangle[2]))
        {
            budget.poly(ExactSweepAccess::poly(*piece->geometry));
            check += piece->sign * sweep_volume(*piece->geometry);
            result->pieces.push_back(std::move(*piece));
        }
    if (check != requested)
        throw std::logic_error("Exact private face correction did not match prescribed integrated volume.");
    return std::shared_ptr<const ExactSweptFace>(new ExactSweptFace(std::move(result)));
}

const std::vector<SignedSweepSimplex>& ExactSweptFace::simplices() const noexcept
{
    return impl_->pieces;
}

ExactPlanarFacePtr ExactSweptFace::origin_face() const noexcept
{
    return impl_->native;
}

const std::vector<ExactNodeTrajectory>& ExactSweptFace::trajectories() const noexcept
{
    return impl_->traces;
}

ExactGeometryMeasure ExactSweptFace::volume() const
{
    return ExactSweepAccess::measure(impl_->volume);
}

ExactGeometryMeasure ExactSweptFace::uncorrected_volume() const
{
    return ExactSweepAccess::measure(impl_->original);
}

std::vector<char> ExactSweptFace::corrected_departure_center() const
{
    return swept_point(impl_->center);
}

const std::vector<std::array<std::vector<char>, 3>>& ExactSweptFace::edge_time_triangles() const noexcept
{
    return impl_->edges;
}

std::vector<ExactSweepIntersection> ExactSweptFace::partition(
    std::span<const ExactSweepDonor> donors, ExactSweepLimits limits) const
{
    SweepBudget budget(limits);
    sweep_donors(donors, budget);
    std::vector<ExactSweepIntersection> result;
    for (std::size_t simplex = 0; simplex < impl_->pieces.size(); ++simplex)
    {
        const auto& piece = impl_->pieces[simplex];
        std::vector<ExactConvexFragmentPtr> coverage;
        for (const auto& donor : donors)
        {
            budget.test();
            if (auto poly = ExactSweepAccess::intersect(
                    ExactSweepAccess::certified(*piece.geometry), ExactSweepAccess::certified(*donor.geometry), budget))
            {
                budget.atom(result.size());
                budget.poly(poly->get());
                auto geometry = ExactSweepAccess::fragment(std::move(*poly));
                coverage.push_back(geometry);
                result.push_back({donor.id, piece.sign, std::move(geometry)});
            }
        }
        try
        {
            certify_sweep_partition(ExactSweepAccess::certified(*piece.geometry), coverage, budget);
        }
        catch (const std::invalid_argument& error)
        {
            throw std::invalid_argument(std::string("Face partition coverage: ") + error.what() +
                                        sweep_cover_context(*this, piece, donors, simplex));
        }
    }
    return result;
}

ExactPhaseMeasures ExactSweptFace::phases(
    std::span<const ExactSweepIntersection> pieces, const GeometryHalfspace& plane)
{
    ExactPhaseMeasures result;
    for (const auto& piece : pieces)
    {
        if (!piece.geometry || (piece.sign != 1 && piece.sign != -1))
            throw std::invalid_argument("Invalid signed actual donor intersection.");
        const std::array<ExactConvexFragmentPtr, 1> atom{piece.geometry};
        const auto p = sweep_phases(atom, plane);
        result.bulk = result.bulk + p.bulk.scaled(piece.sign);
        result.gas = result.gas + p.gas.scaled(piece.sign);
        result.carrier = result.carrier + p.carrier.scaled(piece.sign);
    }
    return result;
}

ExactDonorExport::ExactDonorExport(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

ExactDonorExport ExactDonorExport::make(
    const ConvexCellGeometry& donor, std::span<const OrientedFaceSweep> faces, ExactSweepLimits limits)
{
    SweepBudget budget(limits);
    budget.poly(ExactSweepAccess::poly(donor));
    std::vector<SweepAtom> atoms{{ExactConvexFragment::from_geometry(donor), {0}}};
    for (const auto& face : faces)
    {
        if (!face.face || (face.outward_sign != 1 && face.outward_sign != -1))
            throw std::invalid_argument("Invalid outward sweep incidence.");
        for (const auto& piece : face.face->simplices())
            atoms = overlay_sweep(std::move(atoms), piece, 0, face.outward_sign, budget);
    }
    auto result = std::make_shared<Impl>();
    std::vector<ExactConvexFragmentPtr> coverage;
    for (auto& atom : atoms)
    {
        if (atom.coefficient[0] != 0 && atom.coefficient[0] != 1)
            throw std::invalid_argument("Signed donor export has negative or repeated multiplicity.");
        coverage.push_back(atom.geometry);
        (atom.coefficient[0] ? result->exported : result->retained).push_back(std::move(atom.geometry));
    }
    certify_sweep_partition(ExactSweepAccess::certified(donor), coverage, budget);
    return ExactDonorExport(std::move(result));
}

ExactGeometryMeasure ExactDonorExport::exported_volume() const
{
    ExactGeometryMeasure v;
    for (const auto& p : impl_->exported)
        v = v + p->volume();
    return v;
}

ExactGeometryMeasure ExactDonorExport::retained_volume() const
{
    ExactGeometryMeasure v;
    for (const auto& p : impl_->retained)
        v = v + p->volume();
    return v;
}

ExactPhaseMeasures ExactDonorExport::exported_phases(const GeometryHalfspace& p) const
{
    return sweep_phases(impl_->exported, p);
}

ExactPhaseMeasures ExactDonorExport::retained_phases(const GeometryHalfspace& p) const
{
    return sweep_phases(impl_->retained, p);
}

const std::vector<ExactConvexFragmentPtr>& ExactDonorExport::exported_atoms() const noexcept
{
    return impl_->exported;
}

const std::vector<ExactConvexFragmentPtr>& ExactDonorExport::retained_atoms() const noexcept
{
    return impl_->retained;
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
void target_sweep_shell(const ExactSweepRecipient& target, SweepBudget& budget, std::map<P, P>& shared)
{
    if (!target.geometry)
        throw std::invalid_argument("Missing actual recipient geometry.");
    std::vector<ExactPlanarFacePtr> faces;
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
    if (!ExactPlanarFace::cell_shell(faces)->equivalent(*target.geometry))
        throw std::invalid_argument("Recipient face sweeps do not close its actual oriented native shell.");
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
SweepBudget::SweepBudget(ExactSweepLimits l) : limits(l)
{
    if (!l.maximum_simplices || !l.maximum_atoms || !l.maximum_pair_tests || !l.byte_budget)
        throw std::invalid_argument("Exact sweep requires finite positive resource limits.");
}

void SweepBudget::add(std::size_t n)
{
    if (n > limits.byte_budget - bytes)
        throw std::invalid_argument("Exact sweep byte budget exhausted: admitted=" + std::to_string(bytes) +
                                    ", request=" + std::to_string(n) + ", limit=" + std::to_string(limits.byte_budget) +
                                    ", optional_cache_peak=" + std::to_string(cache_peak) +
                                    ", admitted_plus_live_cache_peak=" + std::to_string(cache_simultaneous_peak) +
                                    ", cache_fallbacks=" + std::to_string(cache_fallbacks));
    bytes += n;
}

void SweepBudget::cache(const ClipScratch& scratch)
{
    if (scratch.used || scratch.peak > limits.byte_budget - bytes)
        throw std::logic_error("Exact clipping cache lease was not released or exceeded remaining capacity.");
    cache_peak = std::max(cache_peak, scratch.peak);
    cache_simultaneous_peak = std::max(cache_simultaneous_peak, bytes + scratch.peak);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    cache_fallbacks = scratch.fallbacks > maximum - cache_fallbacks ? maximum : cache_fallbacks + scratch.fallbacks;
    // The cache vectors are already destroyed. Keep only diagnostics, not
    // a permanent charge for capacity freed between planes/intersections.
    // Exact clipping/backend temporaries retain their existing accounting
    // scope; this is not a process or whole-clipping allocator peak.
}

void SweepBudget::array(std::size_t n, std::size_t size)
{
    if (size && n > (limits.byte_budget - bytes) / size)
        throw std::invalid_argument("Exact sweep allocation count exceeds budget.");
    add(n * size);
}

void SweepBudget::test()
{
    if (tests == limits.maximum_pair_tests)
        throw std::invalid_argument("Exact sweep intersection work budget exhausted.");
    ++tests;
}

void SweepBudget::atom(std::size_t n)
{
    if (n >= limits.maximum_atoms)
        throw std::invalid_argument("Exact sweep arrangement atom budget exhausted.");
}

void SweepBudget::poly(const Poly& p)
{
    add(sizeof(Poly) + p.vertices.capacity() * sizeof(P) + p.faces.capacity() * sizeof(Face));
    for (const auto& v : p.vertices)
        for (const auto& q : v)
            add((q.backend().num().capacity() + q.backend().denom().capacity()) *
                sizeof(boost::multiprecision::limb_type));
    for (const auto& f : p.faces)
        add(f.capacity() * sizeof(std::size_t));
}

SupportPlaneCache::SupportPlaneCache(std::span<const ExactConvexFragmentPtr> pieces, ClipScratch& scratch)
    : lease_(scratch), scratch_(scratch)
{
    scratch_.add(sizeof(SupportPlaneCache));
    const auto plane_capacity = scratch_.limit / sizeof(Plane);
    std::size_t face_count = 0;
    for (const auto& piece : pieces)
    {
        const auto piece_faces = ExactSweepAccess::poly(*piece).faces.size();
        if (piece_faces > plane_capacity - face_count)
            throw ClipCacheExhausted("Optional support-plane face count exceeds its live capacity lease.");
        face_count += piece_faces;
    }
    reserve_support_planes(ranges_, pieces.size(), scratch_, "support-plane ranges");
    reserve_support_planes(planes_, face_count, scratch_, "support-plane array");
    for (const auto& piece : pieces)
    {
        const auto& poly = ExactSweepAccess::poly(*piece);
        const auto begin = planes_.size();
        for (const auto& face : poly.faces)
        {
            Plane plane{sparse_support_normal(poly, face), poly.vertices[face[0]]};
            for (const auto& value : plane.normal)
                scratch_.scalar(value);
            for (const auto& value : plane.anchor)
                scratch_.scalar(value);
            scratch_.scalar(plane.offset);
            planes_.push_back(std::move(plane));
        }
        ranges_.push_back({begin, planes_.size()});
    }
}

std::span<const Plane> SupportPlaneCache::for_piece(std::size_t piece) const
{
    const auto range = ranges_.at(piece);
    return {planes_.data() + range.begin, range.end - range.begin};
}

TrajectoryArithmetic::TrajectoryArithmetic(ExactTrajectoryLimits value) : limits(value)
{
    if (!limits.maximum_operations || !limits.maximum_integer_bits || !limits.byte_budget ||
        limits.maximum_integer_bits > std::numeric_limits<std::size_t>::max() / 16)
        throw std::invalid_argument("Exact trajectory needs positive arithmetic limits.");
    charge(sizeof(*this) + 12 * sizeof(P));
}

void TrajectoryArithmetic::charge(std::size_t count)
{
    if (bytes > limits.byte_budget || count > limits.byte_budget - bytes)
        throw std::invalid_argument("Exact trajectory capacity budget exhausted.");
    bytes += count;
}

std::array<std::size_t, 2> TrajectoryArithmetic::bits(const Q& q) const
{
    const std::array<std::size_t, 2> result{exact_integer_bits(boost::multiprecision::numerator(q)),
        exact_integer_bits(boost::multiprecision::denominator(q))};
    if (result[0] > limits.maximum_integer_bits || result[1] > limits.maximum_integer_bits)
        throw std::invalid_argument("Exact trajectory integer-bit budget exhausted.");
    return result;
}

Q TrajectoryArithmetic::retain(Q q)
{
    (void) bits(q);
    std::size_t storage = sizeof(Q);
    exact_scalar_storage(storage, q);
    charge(storage);
    return q;
}

Q TrajectoryArithmetic::binary(const Q& a, const Q& b, unsigned operation)
{
    if (operations >= limits.maximum_operations)
        throw std::invalid_argument("Exact trajectory arithmetic-work budget exhausted.");
    ++operations;
    auto x = bits(a), y = bits(b);
    const auto bound = limits.maximum_integer_bits;
    const auto sum = [&](std::size_t first, std::size_t second, std::size_t extra = 0)
    {
        if (first > bound || second > bound - first || extra > bound - first - second)
            throw std::invalid_argument("Exact trajectory intermediate-bit budget exhausted.");
        return first + second + extra;
    };
    std::size_t numerator_bits = 0, denominator_bits = 0;
    if (operation < 2)
    {
        numerator_bits = sum(std::max(sum(x[0], y[1]), sum(y[0], x[1])), 0, 1);
        denominator_bits = sum(x[1], y[1]);
    }
    else if (operation == 2)
    {
        numerator_bits = sum(x[0], y[0]);
        denominator_bits = sum(x[1], y[1]);
    }
    else
    {
        if (b == 0)
            throw std::invalid_argument("Exact trajectory zero denominator.");
        numerator_bits = sum(x[0], y[1]);
        denominator_bits = sum(x[1], y[0]);
    }
    // Four limb-sized copies conservatively bound the admitted operation's
    // rational intermediates before multiplication/cross multiplication.
    const auto estimated = sizeof(Q) + 4 * ((numerator_bits + denominator_bits + 7) / 8 + 2 * sizeof(I));
    if (estimated > limits.byte_budget - bytes)
        throw std::invalid_argument("Exact trajectory arithmetic-byte preflight exhausted.");
    if (operation == 0)
        return retain(Q(a + b));
    if (operation == 1)
        return retain(Q(a - b));
    if (operation == 2)
        return retain(Q(a * b));
    return retain(Q(a / b));
}

Q TrajectoryArithmetic::inner(const P& a, const P& b)
{
    auto result = binary(a[0], b[0], 2);
    for (unsigned i = 1; i < 3; ++i)
        result = binary(result, binary(a[i], b[i], 2), 0);
    return result;
}

P TrajectoryArithmetic::cross_product(const P& a, const P& b)
{
    P result;
    for (unsigned i = 0; i < 3; ++i)
        result[i] = binary(binary(a[(i + 1) % 3], b[(i + 2) % 3], 2), binary(a[(i + 2) % 3], b[(i + 1) % 3], 2), 1);
    return result;
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
CertifiedPolyRef ExactSweepAccess::certified(const ConvexCellGeometry& geometry)
{
    return CertifiedPolyRef(geometry.impl_->poly);
}

ExactConvexFragmentPtr ExactSweepAccess::fragment(CertifiedPoly poly)
{
    auto impl = std::make_shared<ExactConvexFragment::Impl>(std::move(poly));
    return ExactConvexFragmentPtr(new ExactConvexFragment(std::move(impl)));
}

ExactGeometryMeasure ExactSweepAccess::measure(Q value)
{
    sample_exact_measure_result(value);
    auto p = std::make_shared<ExactGeometryMeasure::Impl>();
    p->value = std::move(value);
    return ExactGeometryMeasure(std::move(p));
}

ConvexGeometryPtr ExactSweepAccess::geometry(Poly poly)
{
    validate(poly);
    auto impl = std::make_shared<ConvexCellGeometry::Impl>();
    for (std::size_t a = 0; a < 3; ++a)
    {
        impl->origin[a] = poly.vertices[0][a].convert_to<double>();
        if (!std::isfinite(impl->origin[a]))
            throw std::invalid_argument("Sweep local moment anchor is not finite.");
    }
    impl->poly = std::move(poly);
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
}

ExactPlanarFacePtr ExactSweepAccess::retain(const ExactPlanarFace& face)
{
    return ExactPlanarFacePtr(new ExactPlanarFace(face.impl_));
}

ConvexGeometryPtr ExactSweepAccess::retain(const ConvexCellGeometry& cell)
{
    return ConvexGeometryPtr(new ConvexCellGeometry(cell.impl_));
}
} // namespace SimpleFluid::Meshes
