#include "geometry/swept_volume/details/exact/SweptGeometryInternal.hh"

namespace SimpleFluid::geometry::swept_volume::exact_swept_detail
{
PLIC::PointIdentity<ArithmeticMode::Exact> swept_point(const P& point)
{
    WireWriter writer;
    writer.word(0x5346504f494e5431ULL);
    for (const auto& q : point)
        writer.rational(q);
    return PLIC::details::GeometryIdentityAccess::make<PLIC::PointIdentity<ArithmeticMode::Exact>>(
        std::move(writer.data));
}
} // namespace SimpleFluid::geometry::swept_volume::exact_swept_detail

namespace SimpleFluid::PLIC::exact_geometry_detail
{
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
} // namespace SimpleFluid::PLIC::exact_geometry_detail

namespace SimpleFluid::geometry::swept_volume::exact_swept_detail
{
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

std::string sweep_cover_context(const SweptFace<ArithmeticMode::Exact>& face,
    const SignedSweepSimplex<ArithmeticMode::Exact>& piece, std::span<const SweepDonor<ArithmeticMode::Exact>> donors,
    std::size_t simplex)
{
    const auto& poly = SweepAccess<ArithmeticMode::Exact>::poly(*piece.geometry);
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
            for (const auto& plane : planes(SweepAccess<ArithmeticMode::Exact>::poly(*donor.geometry)))
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
} // namespace SimpleFluid::geometry::swept_volume::exact_swept_detail

namespace SimpleFluid::PLIC::exact_geometry_detail
{
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
} // namespace SimpleFluid::PLIC::exact_geometry_detail

namespace SimpleFluid::geometry::swept_volume::exact_swept_detail
{
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
bool certified_cover_contained(std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> pieces, CertifiedPolyRef outer)
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
            for (const auto& vertex : SweepAccess<ArithmeticMode::Exact>::certified(*piece).get().vertices)
                if (distance(plane, vertex) > 0)
                    return false;
    }
    return true;
}

void certify_sweep_partition(CertifiedPolyRef expected_certificate,
    std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> pieces, SweepBudget& budget)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.cover_certifications;
    const auto& expected = expected_certificate.get();
    const auto origin = expected.vertices[0];
    const auto whole = exact_moments(expected, origin);
    Moments sum{};
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
        const auto part = exact_moments(SweepAccess<ArithmeticMode::Exact>::poly(*pieces[i]), origin);
        sum.volume += part.volume;
        sum.first = add(sum.first, part.first);
        for (std::size_t j = 0; j < i; ++j)
        {
            budget.test();
            const auto a = SweepAccess<ArithmeticMode::Exact>::certified(*pieces[i]);
            const auto b = SweepAccess<ArithmeticMode::Exact>::certified(*pieces[j]);
            const bool intersects = support_cache
                                        ? cached_pair_intersects(a, i, b, j, *support_cache, support_scratch)
                                        : SweepAccess<ArithmeticMode::Exact>::intersect(a, b, budget).has_value();
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

std::optional<SignedSweepSimplex<ArithmeticMode::Exact>> swept_tetra(
    const P& origin, const P& a, const P& b, const P& c)
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
    return SignedSweepSimplex<ArithmeticMode::Exact>{sign, SweepAccess<ArithmeticMode::Exact>::fragment(std::move(p))};
}

std::vector<SweptTriangle> swept_boundary(const Polygon& current, const Polygon& previous, const P& center,
    const P& backcenter, std::vector<std::array<PLIC::PointIdentity<ArithmeticMode::Exact>, 3>>* sides)
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

void sweep_donors(std::span<const SweepDonor<ArithmeticMode::Exact>> donors, SweepBudget& budget)
{
    std::set<std::uint64_t> ids;
    for (std::size_t i = 0; i < donors.size(); ++i)
    {
        if (!donors[i].geometry || !ids.insert(donors[i].id).second)
            throw std::invalid_argument("Exact sweep donor geometry/identity is missing or duplicated.");
        budget.add(sizeof(SweepDonor<ArithmeticMode::Exact>));
        for (std::size_t j = 0; j < i; ++j)
        {
            budget.test();
            if (SweepAccess<ArithmeticMode::Exact>::intersect(
                    SweepAccess<ArithmeticMode::Exact>::certified(*donors[i].geometry),
                    SweepAccess<ArithmeticMode::Exact>::certified(*donors[j].geometry), budget))
                throw std::invalid_argument("Exact sweep donor interiors overlap.");
        }
    }
}

std::vector<SweepAtom> overlay_sweep(std::vector<SweepAtom> atoms,
    const SignedSweepSimplex<ArithmeticMode::Exact>& piece, std::size_t column, int sign, SweepBudget& budget)
{
    std::vector<SweepAtom> next;
    for (auto& atom : atoms)
    {
        budget.test();
        const auto& a = SweepAccess<ArithmeticMode::Exact>::poly(*atom.geometry);
        const auto& b = SweepAccess<ArithmeticMode::Exact>::poly(*piece.geometry);
        if (!SweepAccess<ArithmeticMode::Exact>::intersect(
                SweepAccess<ArithmeticMode::Exact>::certified(*atom.geometry),
                SweepAccess<ArithmeticMode::Exact>::certified(*piece.geometry), budget))
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
        std::optional<CertifiedPoly> remaining =
            SweepAccess<ArithmeticMode::Exact>::copy(SweepAccess<ArithmeticMode::Exact>::certified(*atom.geometry));
        for (const auto& plane : planes(b))
        {
            Plane other = plane;
            other.normal = mul(other.normal, Q(-1));
            other.offset = -other.offset;
            auto outside = SweepAccess<ArithmeticMode::Exact>::clip(remaining->ref(), other, budget);
            if (outside)
            {
                budget.atom(next.size());
                budget.poly(outside->get());
                budget.add(atom.coefficient.size() * sizeof(int) + sizeof(SweepAtom));
                next.push_back({SweepAccess<ArithmeticMode::Exact>::fragment(std::move(*outside)), atom.coefficient});
            }
            remaining = SweepAccess<ArithmeticMode::Exact>::clip(remaining->ref(), plane, budget);
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
            next.push_back(
                {SweepAccess<ArithmeticMode::Exact>::fragment(std::move(*remaining)), std::move(coefficient)});
        }
    }
    return next;
}

PhaseMeasures<ArithmeticMode::Exact> sweep_phases(
    std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> atoms, const GeometryHalfspace& halfspace)
{
    PhaseMeasures<ArithmeticMode::Exact> result;
    Plane plane{point(halfspace.normal), point(halfspace.anchor), exact(halfspace.offset)};
    if (zero(plane.normal))
        throw std::invalid_argument("Phase plane normal must be nonzero.");
    Plane outside = plane;
    outside.normal = mul(outside.normal, Q(-1));
    outside.offset = -outside.offset;
    for (const auto& atom : atoms)
    {
        result.bulk = result.bulk + atom->volume();
        if (auto cut = certified_clip_uncached(SweepAccess<ArithmeticMode::Exact>::certified(*atom), plane))
            result.gas = result.gas +
                         SweepAccess<ArithmeticMode::Exact>::measure(exact_volume(cut->get(), cut->get().vertices[0]));
        if (auto cut = certified_clip_uncached(SweepAccess<ArithmeticMode::Exact>::certified(*atom), outside))
            result.carrier = result.carrier + SweepAccess<ArithmeticMode::Exact>::measure(
                                                  exact_volume(cut->get(), cut->get().vertices[0]));
    }
    if ((result.gas + result.carrier).compare(result.bulk) != 0)
        throw std::logic_error("Exact independent phase clips lost complementarity.");
    return result;
}
} // namespace SimpleFluid::geometry::swept_volume::exact_swept_detail

namespace SimpleFluid::geometry::swept_volume
{
ConvexFragment<ArithmeticMode::Exact>::ConvexFragment(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

ConvexFragmentPtr<ArithmeticMode::Exact> ConvexFragment<ArithmeticMode::Exact>::from_geometry(
    const ConvexCellGeometry<ArithmeticMode::Exact>& cell)
{
    return SweepAccess<ArithmeticMode::Exact>::fragment(SweepAccess<ArithmeticMode::Exact>::certified(cell));
}

GeometryMeasure<ArithmeticMode::Exact> ConvexFragment<ArithmeticMode::Exact>::volume() const
{
    return SweepAccess<ArithmeticMode::Exact>::measure(sweep_volume(*this));
}

PhaseMeasures<ArithmeticMode::Exact> ConvexFragment<ArithmeticMode::Exact>::phases(const GeometryHalfspace& plane) const
{
    const std::array<ConvexFragmentPtr<ArithmeticMode::Exact>, 1> fragment{
        ConvexFragmentPtr<ArithmeticMode::Exact>(new ConvexFragment<ArithmeticMode::Exact>(impl_))};
    return sweep_phases(fragment, plane);
}

ConvexFragmentPtr<ArithmeticMode::Exact> ConvexFragment<ArithmeticMode::Exact>::translated(
    const PLIC::DisplacementIdentity<ArithmeticMode::Exact>& displacement) const
{
    auto poly = impl_->poly.get();
    const auto shift = displacement_point(displacement);
    for (auto& p : poly.vertices)
        p = add(p, shift);
    return SweepAccess<ArithmeticMode::Exact>::fragment(std::move(poly));
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexFragment<ArithmeticMode::Exact>::as_geometry() const
{
    return SweepAccess<ArithmeticMode::Exact>::geometry(impl_->poly.get());
}

std::vector<char> ConvexFragment<ArithmeticMode::Exact>::encode(std::size_t byte_budget) const
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

ConvexFragmentPtr<ArithmeticMode::Exact> ConvexFragment<ArithmeticMode::Exact>::decode(
    std::span<const char> data, std::size_t byte_budget)
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
    return SweepAccess<ArithmeticMode::Exact>::fragment(std::move(poly));
}
} // namespace SimpleFluid::geometry::swept_volume

namespace SimpleFluid::PLIC
{
GeometryMeasure<ArithmeticMode::Exact>::GeometryMeasure(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::from_value(double value)
{
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(exact(value));
}

template<>
GeometryMeasure<ArithmeticMode::Exact> point_projection<ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>& identity, GeometryPoint normal, GeometryPoint anchor)
{
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(
        dot(point(normal), sub(authority_point(identity), point(anchor))));
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::volume(
    const ConvexCellGeometry<ArithmeticMode::Exact>& cell)
{
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(::SimpleFluid::geometry::swept_volume::exact_swept_detail::sweep_volume(cell));
}

GeometryValue<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::certified() const
{
    return measured(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this));
}

GeometryValue<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::enclosure() const
{
    return measured(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this), true);
}

bool GeometryMeasure<ArithmeticMode::Exact>::relative_error_at_most(
    const GeometryMeasure<ArithmeticMode::Exact>& reference, double tolerance) const
{
    const auto factor = exact(tolerance);
    if (factor < 0)
        throw std::invalid_argument("Relative measure tolerance must be nonnegative.");
    Q difference =
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) - ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(reference);
    if (difference < 0)
        difference = -difference;
    Q scale = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(reference);
    if (scale < 0)
        scale = -scale;
    return difference <= factor * scale;
}

double GeometryMeasure<ArithmeticMode::Exact>::log_ratio(
    const GeometryMeasure<ArithmeticMode::Exact>& denominator) const
{
    if (::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) <= 0 ||
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(denominator) <= 0)
        throw std::invalid_argument("Logarithmic measure ratio requires positive operands.");
    const Q ratio =
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) / ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(denominator);
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

std::vector<char> GeometryMeasure<ArithmeticMode::Exact>::encode(std::size_t byte_budget) const
{
    const auto& q = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this);
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

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::decode(
    std::span<const char> data, std::size_t byte_budget)
{
    if (data.size() > byte_budget)
        throw std::invalid_argument("Exact measure certificate exceeds byte budget.");
    WireReader r{data};
    if (r.word() != 0x53464d4541533731ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact measure tag/version.");
    auto value = r.rational();
    r.finish();
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(std::move(value));
}

int GeometryMeasure<ArithmeticMode::Exact>::sign() const
{
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) > 0   ? 1
           : ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) < 0 ? -1
                                                                  : 0;
}

int GeometryMeasure<ArithmeticMode::Exact>::compare(const GeometryMeasure<ArithmeticMode::Exact>& other) const
{
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) > ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(other)   ? 1
           : ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) < ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(other) ? -1
                                                                                                                 : 0;
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::operator+(
    const GeometryMeasure<ArithmeticMode::Exact>& other) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) + ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(other));
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::operator-(
    const GeometryMeasure<ArithmeticMode::Exact>& other) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) - ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(other));
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::scaled(double factor) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    return ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) * exact(factor));
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC::exact_geometry_detail
{
std::size_t exact_integer_bits(I value)
{
    if (value < 0)
        value = -value;
    return value == 0 ? 0 : static_cast<std::size_t>(boost::multiprecision::msb(value)) + 1;
}

void exact_measure_arithmetic_preflight(const Q& a, const Q& b, bool division, std::size_t budget)
{
    const auto overhead =
        sizeof(GeometryMeasure<ArithmeticMode::Exact>) + sizeof(GeometryMeasure<ArithmeticMode::Exact>::Impl);
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
} // namespace SimpleFluid::PLIC::exact_geometry_detail

namespace SimpleFluid::PLIC
{
GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::divided_by(
    const GeometryMeasure<ArithmeticMode::Exact>& denominator, std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    if (::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(denominator) == 0)
        throw std::invalid_argument("Exact measure division by zero.");
    exact_measure_arithmetic_preflight(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this),
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(denominator), true, byte_budget);
    auto result = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) / ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(denominator));
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact measure quotient capacity exceeds byte budget.");
    return result;
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::squared(std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.measure_arithmetic;
    exact_measure_arithmetic_preflight(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this),
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this), false, byte_budget);
    auto result = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(
        ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) * ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this));
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact measure squared capacity exceeds byte budget.");
    return result;
}

std::size_t GeometryMeasure<ArithmeticMode::Exact>::storage_bytes() const
{
    if (!impl_)
        return sizeof(*this);
    std::size_t bytes = sizeof(*this) + sizeof(Impl);
    exact_scalar_storage(bytes, ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this));
    return bytes;
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::compacted(
    std::size_t operation_byte_budget) const
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
        charge(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this).backend().num(), fresh.backend().num());
        charge(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this).backend().denom(), fresh.backend().denom());
    }
    const auto source_bytes = storage_bytes();
    if (source_bytes > operation_byte_budget || copied > operation_byte_budget - source_bytes)
        throw std::invalid_argument("Exact measure compact copy exceeds its simultaneous storage budget.");
    auto compact = std::make_shared<Impl>();
    compact->value = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this);
    GeometryMeasure<ArithmeticMode::Exact> result(std::move(compact));
    if (::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(result) != ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this) ||
        result.storage_bytes() > copied || result.storage_bytes() > operation_byte_budget - source_bytes)
        throw std::invalid_argument("Exact measure compact copy failed its exact value or capacity certificate.");
    return result;
}

GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::rounded_dyadic(
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
    I numerator = boost::multiprecision::numerator(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this));
    const bool negative = numerator < 0;
    if (negative)
        numerator = -numerator;
    I denominator = boost::multiprecision::denominator(::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::value(*this));
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
    auto result = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(std::move(rounded));
    if (result.sign() != sign())
        throw std::logic_error("Exact dyadic representation failed its nonzero sign certificate.");
    if (result.storage_bytes() > operation_byte_budget / 16)
        throw std::length_error("Exact dyadic representation exceeds its retained storage certificate.");
    return result;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::geometry::swept_volume
{
std::size_t ConvexFragment<ArithmeticMode::Exact>::storage_bytes() const
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

NodeTrajectory<ArithmeticMode::Exact> NodeTrajectory<ArithmeticMode::Exact>::backtrace(
    const PLIC::PointIdentity<ArithmeticMode::Exact>& identity, GeometryPoint velocity, double timestep)
{
    if (!(timestep >= 0) || !std::isfinite(timestep))
        throw std::invalid_argument("Backward trajectory needs finite nonnegative timestep.");
    const auto p = authority_point(identity);
    const P exact_velocity{exact(velocity[0]), exact(velocity[1]), exact(velocity[2])};
    const auto back = sub(p, mul(exact_velocity, exact(timestep)));
    return {swept_point(p), swept_point(back)};
}
} // namespace SimpleFluid::geometry::swept_volume

namespace SimpleFluid::geometry::swept_volume::exact_swept_detail
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
} // namespace SimpleFluid::geometry::swept_volume::exact_swept_detail

namespace SimpleFluid::PLIC
{
GeometryMeasure<ArithmeticMode::Exact> PlanarFace<ArithmeticMode::Exact>::coplanar_area_ratio(
    const PlanarFace<ArithmeticMode::Exact>& denominator, std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    ::SimpleFluid::geometry::swept_volume::TrajectoryLimits limits;
    limits.byte_budget = byte_budget;
    ::SimpleFluid::geometry::swept_volume::exact_swept_detail::TrajectoryArithmetic arithmetic(limits);
    const auto support = ::SimpleFluid::geometry::swept_volume::exact_swept_detail::bounded_polygon_vector(denominator.impl_->polygon, arithmetic);
    unsigned axis = 0;
    while (axis < 3 && support[axis] == 0)
        ++axis;
    if (axis == 3)
        throw std::invalid_argument("Exact face area denominator is degenerate.");
    for (const auto& point : impl_->polygon)
        if (arithmetic.inner(support, arithmetic.subtract(point, denominator.impl_->polygon[0])) != 0)
            throw std::invalid_argument("Exact face area ratio requires identical support planes.");
    const auto area = ::SimpleFluid::geometry::swept_volume::exact_swept_detail::bounded_polygon_vector(impl_->polygon, arithmetic);
    auto ratio = arithmetic.binary(area[axis], support[axis], 3);
    if (ratio < 0)
        ratio = arithmetic.binary(Q(0), ratio, 1);
    if (ratio == 0)
        throw std::invalid_argument("Exact face area numerator is degenerate.");
    auto result = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(std::move(ratio));
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact face area ratio result exceeds byte budget.");
    return result;
}

GeometryMeasure<ArithmeticMode::Exact> PlanarFace<ArithmeticMode::Exact>::area_vector_dot(
    GeometryPoint velocity, std::size_t byte_budget) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    ::SimpleFluid::geometry::swept_volume::TrajectoryLimits limits;
    limits.byte_budget = byte_budget;
    ::SimpleFluid::geometry::swept_volume::exact_swept_detail::TrajectoryArithmetic arithmetic(limits);
    const P represented{arithmetic.retain(exact(velocity[0])), arithmetic.retain(exact(velocity[1])),
        arithmetic.retain(exact(velocity[2]))};
    const auto area = ::SimpleFluid::geometry::swept_volume::exact_swept_detail::bounded_polygon_vector(impl_->polygon, arithmetic);
    auto result = ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>::measure(arithmetic.inner(area, represented));
    if (result.storage_bytes() > byte_budget)
        throw std::invalid_argument("Exact face area flux result exceeds byte budget.");
    return result;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::geometry::swept_volume
{
NodeTrajectory<ArithmeticMode::Exact> NodeTrajectory<ArithmeticMode::Exact>::backtrace_constrained(
    const PLIC::PointIdentity<ArithmeticMode::Exact>& identity, GeometryPoint velocity, double timestep,
    std::span<const PlanarFacePtr<ArithmeticMode::Exact>> walls, TrajectoryLimits limits)
{
    TrajectoryArithmetic arithmetic(limits);
    if (!std::isfinite(timestep) || timestep < 0 || walls.size() > limits.maximum_constraints)
        throw std::invalid_argument("Invalid exact wall trajectory timestep or constraint count.");
    arithmetic.charge(canonical_key_size(identity));
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
        const auto& polygon = SweepAccess<ArithmeticMode::Exact>::polygon(*wall);
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

GeometryMeasure<ArithmeticMode::Exact> NodeTrajectory<ArithmeticMode::Exact>::displacement_squared(
    TrajectoryLimits limits) const
{
    TrajectoryArithmetic arithmetic(limits);
    arithmetic.charge(canonical_key_size(point));
    arithmetic.charge(canonical_key_size(departure));
    const auto displacement = arithmetic.subtract(authority_point(point), authority_point(departure));
    return SweepAccess<ArithmeticMode::Exact>::measure(arithmetic.inner(displacement, displacement));
}

SweptFace<ArithmeticMode::Exact>::SweptFace(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

std::shared_ptr<const SweptFace<ArithmeticMode::Exact>> SweptFace<ArithmeticMode::Exact>::make(
    const PlanarFace<ArithmeticMode::Exact>& face, std::span<const NodeTrajectory<ArithmeticMode::Exact>> traces,
    double volume, SweepLimits limits)
{
    return make(face, traces, GeometryMeasure<ArithmeticMode::Exact>::from_value(volume), limits);
}

std::shared_ptr<const SweptFace<ArithmeticMode::Exact>> SweptFace<ArithmeticMode::Exact>::make(
    const PlanarFace<ArithmeticMode::Exact>& face, std::span<const NodeTrajectory<ArithmeticMode::Exact>> traces,
    const GeometryMeasure<ArithmeticMode::Exact>& volume, SweepLimits limits)
{
    SweepBudget budget(limits);
    const auto& current = SweepAccess<ArithmeticMode::Exact>::polygon(face);
    if (current.size() > limits.maximum_simplices / 4)
        throw std::invalid_argument("Exact sweep simplex budget exhausted before construction.");
    if (traces.size() != current.size())
        throw std::invalid_argument("Every canonical face node needs exactly one shared trajectory.");
    std::map<P, P> mapped;
    for (const auto& trace : traces)
    {
        budget.add(canonical_key_size(trace.point) + canonical_key_size(trace.departure) +
                   sizeof(NodeTrajectory<ArithmeticMode::Exact>));
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
    const auto requested = SweepAccess<ArithmeticMode::Exact>::value(volume);
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
    result->native = SweepAccess<ArithmeticMode::Exact>::retain(face);
    result->traces.assign(traces.begin(), traces.end());
    auto triangles = swept_boundary(current, previous, center, backcenter, &result->edges);
    Q check = 0;
    for (const auto& triangle : triangles)
        if (auto piece = swept_tetra(center, triangle[0], triangle[1], triangle[2]))
        {
            budget.poly(SweepAccess<ArithmeticMode::Exact>::poly(*piece->geometry));
            check += piece->sign * sweep_volume(*piece->geometry);
            result->pieces.push_back(std::move(*piece));
        }
    if (check != requested)
        throw std::logic_error("Exact private face correction did not match prescribed integrated volume.");
    return std::shared_ptr<const SweptFace<ArithmeticMode::Exact>>(
        new SweptFace<ArithmeticMode::Exact>(std::move(result)));
}

const std::vector<SignedSweepSimplex<ArithmeticMode::Exact>>&
SweptFace<ArithmeticMode::Exact>::simplices() const noexcept
{
    return impl_->pieces;
}

PlanarFacePtr<ArithmeticMode::Exact> SweptFace<ArithmeticMode::Exact>::origin_face() const noexcept
{
    return impl_->native;
}

const std::vector<NodeTrajectory<ArithmeticMode::Exact>>&
SweptFace<ArithmeticMode::Exact>::trajectories() const noexcept
{
    return impl_->traces;
}

GeometryMeasure<ArithmeticMode::Exact> SweptFace<ArithmeticMode::Exact>::volume() const
{
    return SweepAccess<ArithmeticMode::Exact>::measure(impl_->volume);
}

GeometryMeasure<ArithmeticMode::Exact> SweptFace<ArithmeticMode::Exact>::uncorrected_volume() const
{
    return SweepAccess<ArithmeticMode::Exact>::measure(impl_->original);
}

PLIC::PointIdentity<ArithmeticMode::Exact> SweptFace<ArithmeticMode::Exact>::corrected_departure_center() const
{
    return swept_point(impl_->center);
}

const std::vector<std::array<PLIC::PointIdentity<ArithmeticMode::Exact>, 3>>&
SweptFace<ArithmeticMode::Exact>::edge_time_triangles() const noexcept
{
    return impl_->edges;
}

std::vector<SweepIntersection<ArithmeticMode::Exact>> SweptFace<ArithmeticMode::Exact>::partition(
    std::span<const SweepDonor<ArithmeticMode::Exact>> donors, SweepLimits limits) const
{
    SweepBudget budget(limits);
    sweep_donors(donors, budget);
    std::vector<SweepIntersection<ArithmeticMode::Exact>> result;
    for (std::size_t simplex = 0; simplex < impl_->pieces.size(); ++simplex)
    {
        const auto& piece = impl_->pieces[simplex];
        std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> coverage;
        for (const auto& donor : donors)
        {
            budget.test();
            if (auto poly = SweepAccess<ArithmeticMode::Exact>::intersect(
                    SweepAccess<ArithmeticMode::Exact>::certified(*piece.geometry),
                    SweepAccess<ArithmeticMode::Exact>::certified(*donor.geometry), budget))
            {
                budget.atom(result.size());
                budget.poly(poly->get());
                auto geometry = SweepAccess<ArithmeticMode::Exact>::fragment(std::move(*poly));
                coverage.push_back(geometry);
                result.push_back({donor.id, piece.sign, std::move(geometry)});
            }
        }
        try
        {
            certify_sweep_partition(SweepAccess<ArithmeticMode::Exact>::certified(*piece.geometry), coverage, budget);
        }
        catch (const std::invalid_argument& error)
        {
            throw std::invalid_argument(std::string("Face partition coverage: ") + error.what() +
                                        sweep_cover_context(*this, piece, donors, simplex));
        }
    }
    return result;
}

PhaseMeasures<ArithmeticMode::Exact> SweptFace<ArithmeticMode::Exact>::phases(
    std::span<const SweepIntersection<ArithmeticMode::Exact>> pieces, const GeometryHalfspace& plane)
{
    PhaseMeasures<ArithmeticMode::Exact> result;
    for (const auto& piece : pieces)
    {
        if (!piece.geometry || (piece.sign != 1 && piece.sign != -1))
            throw std::invalid_argument("Invalid signed actual donor intersection.");
        const std::array<ConvexFragmentPtr<ArithmeticMode::Exact>, 1> atom{piece.geometry};
        const auto p = sweep_phases(atom, plane);
        result.bulk = result.bulk + p.bulk.scaled(piece.sign);
        result.gas = result.gas + p.gas.scaled(piece.sign);
        result.carrier = result.carrier + p.carrier.scaled(piece.sign);
    }
    return result;
}

DonorExport<ArithmeticMode::Exact>::DonorExport(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}

DonorExport<ArithmeticMode::Exact> DonorExport<ArithmeticMode::Exact>::make(
    const ConvexCellGeometry<ArithmeticMode::Exact>& donor,
    std::span<const OrientedFaceSweep<ArithmeticMode::Exact>> faces, SweepLimits limits)
{
    SweepBudget budget(limits);
    budget.poly(SweepAccess<ArithmeticMode::Exact>::poly(donor));
    std::vector<SweepAtom> atoms{{ConvexFragment<ArithmeticMode::Exact>::from_geometry(donor), {0}}};
    for (const auto& face : faces)
    {
        if (!face.face || (face.outward_sign != 1 && face.outward_sign != -1))
            throw std::invalid_argument("Invalid outward sweep incidence.");
        for (const auto& piece : face.face->simplices())
            atoms = overlay_sweep(std::move(atoms), piece, 0, face.outward_sign, budget);
    }
    auto result = std::make_shared<Impl>();
    std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> coverage;
    for (auto& atom : atoms)
    {
        if (atom.coefficient[0] != 0 && atom.coefficient[0] != 1)
            throw std::invalid_argument("Signed donor export has negative or repeated multiplicity.");
        coverage.push_back(atom.geometry);
        (atom.coefficient[0] ? result->exported : result->retained).push_back(std::move(atom.geometry));
    }
    certify_sweep_partition(SweepAccess<ArithmeticMode::Exact>::certified(donor), coverage, budget);
    return DonorExport<ArithmeticMode::Exact>(std::move(result));
}

GeometryMeasure<ArithmeticMode::Exact> DonorExport<ArithmeticMode::Exact>::exported_volume() const
{
    GeometryMeasure<ArithmeticMode::Exact> v;
    for (const auto& p : impl_->exported)
        v = v + p->volume();
    return v;
}

GeometryMeasure<ArithmeticMode::Exact> DonorExport<ArithmeticMode::Exact>::retained_volume() const
{
    GeometryMeasure<ArithmeticMode::Exact> v;
    for (const auto& p : impl_->retained)
        v = v + p->volume();
    return v;
}

PhaseMeasures<ArithmeticMode::Exact> DonorExport<ArithmeticMode::Exact>::exported_phases(
    const GeometryHalfspace& p) const
{
    return sweep_phases(impl_->exported, p);
}

PhaseMeasures<ArithmeticMode::Exact> DonorExport<ArithmeticMode::Exact>::retained_phases(
    const GeometryHalfspace& p) const
{
    return sweep_phases(impl_->retained, p);
}

const std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>>&
DonorExport<ArithmeticMode::Exact>::exported_atoms() const noexcept
{
    return impl_->exported;
}

const std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>>&
DonorExport<ArithmeticMode::Exact>::retained_atoms() const noexcept
{
    return impl_->retained;
}
} // namespace SimpleFluid::geometry::swept_volume

namespace SimpleFluid::geometry::swept_volume::exact_swept_detail
{
SweepBudget::SweepBudget(SweepLimits l) : limits(l)
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

SupportPlaneCache::SupportPlaneCache(
    std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> pieces, ClipScratch& scratch)
    : lease_(scratch), scratch_(scratch)
{
    scratch_.add(sizeof(SupportPlaneCache));
    const auto plane_capacity = scratch_.limit / sizeof(Plane);
    std::size_t face_count = 0;
    for (const auto& piece : pieces)
    {
        const auto piece_faces = SweepAccess<ArithmeticMode::Exact>::poly(*piece).faces.size();
        if (piece_faces > plane_capacity - face_count)
            throw ClipCacheExhausted("Optional support-plane face count exceeds its live capacity lease.");
        face_count += piece_faces;
    }
    reserve_support_planes(ranges_, pieces.size(), scratch_, "support-plane ranges");
    reserve_support_planes(planes_, face_count, scratch_, "support-plane array");
    for (const auto& piece : pieces)
    {
        const auto& poly = SweepAccess<ArithmeticMode::Exact>::poly(*piece);
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

TrajectoryArithmetic::TrajectoryArithmetic(TrajectoryLimits value) : limits(value)
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
} // namespace SimpleFluid::geometry::swept_volume::exact_swept_detail

namespace SimpleFluid::geometry::swept_volume
{
CertifiedPolyRef SweepAccess<ArithmeticMode::Exact>::certified(
    const ConvexCellGeometry<ArithmeticMode::Exact>& geometry)
{
    return CertifiedPolyRef(geometry.impl_->poly);
}

ConvexFragmentPtr<ArithmeticMode::Exact> SweepAccess<ArithmeticMode::Exact>::fragment(CertifiedPoly poly)
{
    auto impl = std::make_shared<ConvexFragment<ArithmeticMode::Exact>::Impl>(std::move(poly));
    return ConvexFragmentPtr<ArithmeticMode::Exact>(new ConvexFragment<ArithmeticMode::Exact>(std::move(impl)));
}

GeometryMeasure<ArithmeticMode::Exact> SweepAccess<ArithmeticMode::Exact>::measure(Q value)
{
    sample_exact_measure_result(value);
    auto p = std::make_shared<GeometryMeasure<ArithmeticMode::Exact>::Impl>();
    p->value = std::move(value);
    return GeometryMeasure<ArithmeticMode::Exact>(std::move(p));
}

ConvexGeometryPtr<ArithmeticMode::Exact> SweepAccess<ArithmeticMode::Exact>::geometry(Poly poly)
{
    validate(poly);
    auto impl = std::make_shared<ConvexCellGeometry<ArithmeticMode::Exact>::Impl>();
    for (std::size_t a = 0; a < 3; ++a)
    {
        impl->origin[a] = poly.vertices[0][a].convert_to<double>();
        if (!std::isfinite(impl->origin[a]))
            throw std::invalid_argument("Sweep local moment anchor is not finite.");
    }
    impl->poly = std::move(poly);
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
}

PlanarFacePtr<ArithmeticMode::Exact> SweepAccess<ArithmeticMode::Exact>::retain(
    const PlanarFace<ArithmeticMode::Exact>& face)
{
    return PlanarFacePtr<ArithmeticMode::Exact>(new PlanarFace<ArithmeticMode::Exact>(face.impl_));
}

ConvexGeometryPtr<ArithmeticMode::Exact> SweepAccess<ArithmeticMode::Exact>::retain(
    const ConvexCellGeometry<ArithmeticMode::Exact>& cell)
{
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(cell.impl_));
}
} // namespace SimpleFluid::geometry::swept_volume

namespace SimpleFluid::PLIC
{
GeometryMeasure<ArithmeticMode::Exact> GeometryMeasure<ArithmeticMode::Exact>::volume(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source)
{
    return volume(*ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(source));
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::geometry::swept_volume
{
std::shared_ptr<const ConvexFragment<ArithmeticMode::Exact>> ConvexFragment<ArithmeticMode::Exact>::from_geometry(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source)
{
    return from_geometry(*ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(source));
}
} // namespace SimpleFluid::geometry::swept_volume

namespace SimpleFluid::geometry::swept_volume
{
SweepDonor<ArithmeticMode::Exact>::SweepDonor(
    std::uint64_t source, const ConvexGeometryPtr<ArithmeticMode::Ordinary>& cell)
    : id(source), geometry(cell ? ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(*cell) : nullptr)
{
}

SweepRecipient<ArithmeticMode::Exact>::SweepRecipient(std::uint64_t target,
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>& cell,
    std::vector<OrientedFaceSweep<ArithmeticMode::Exact>> faces)
    : id(target), geometry(cell ? ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(*cell) : nullptr),
      outward_faces(std::move(faces))
{
}

DonorExport<ArithmeticMode::Exact> DonorExport<ArithmeticMode::Exact>::make(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source,
    std::span<const OrientedFaceSweep<ArithmeticMode::Exact>> faces, SweepLimits limits)
{
    return make(*ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(source), faces, limits);
}
} // namespace SimpleFluid::geometry::swept_volume
