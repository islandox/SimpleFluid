/** @file BoundedVoFTransport.hh
 * @brief Bounded signed face chains and complementary closed-cell overlays.
 */
#pragma once
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include "geometry/ExactSweptGeometry.hh"
#include <map>
#include <set>
#include <span>
namespace SimpleFluid::VoF::finite_detail
{
using Geometry = Meshes::BoundedConvexGeometry;
using GeometryPtr = Meshes::BoundedConvexGeometryPtr;
struct Work
{
    Meshes::ExactSweepLimits limits;
    std::size_t bytes = 0, pair_tests = 0;
    void charge(std::size_t n)
    {
        if (n > limits.byte_budget - std::min(bytes, limits.byte_budget))
            throw std::invalid_argument("Finite VoF cumulative construction byte budget is exhausted.");
        bytes += n;
    }
    void test()
    {
        if (++pair_tests > limits.maximum_pair_tests)
            throw std::invalid_argument("Finite VoF pair-test budget is exhausted.");
    }
    template<class T> void push(std::vector<T>& values, T value)
    {
        if (values.size() >= limits.maximum_atoms)
            throw std::invalid_argument("Finite VoF atom buffer capacity is exhausted.");
        if (values.size() == values.capacity())
        {
            const auto capacity = values.capacity();
            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                throw std::invalid_argument("Finite VoF atom capacity overflows.");
            const auto next = capacity ? 2 * capacity : 1;
            if (next > std::numeric_limits<std::size_t>::max() / sizeof(T))
                throw std::invalid_argument("Finite VoF atom byte capacity overflows.");
            charge(next * sizeof(T));
            values.reserve(next);
        }
        values.push_back(std::move(value));
    }
    void created(const GeometryPtr& p, const GeometryPtr& source = {})
    {
        charge(sizeof(Geometry)); // Every clip result owns its wrapper, even when its immutable payload aliases.
        if (!source || p->storage_identity() != source->storage_identity())
            charge(p->storage_bytes());
    }
    void atom(const GeometryPtr& p)
    {
        charge(sizeof(GeometryPtr));
        created(p);
    }

};
struct SignedPiece
{
    GeometryPtr geometry;
    int sign = 1;
};
struct Sweep
{
    std::uint64_t face = 0;
    std::vector<SignedPiece> pieces;
    std::vector<Point> current, departure;
    Point center, backcenter;
    Value control;
};
inline bool same_point(const Point& a, const Point& b)
{
    return Meshes::bounded_points_identical(a, b);
}
inline Value cone_determinant(const Point& origin, const std::array<Point, 3>& triangle)
{
    return Meshes::bounded_orientation({origin, triangle[0], triangle[1], triangle[2]});
}
inline std::vector<std::array<Point, 3>> sweep_boundary(std::span<const std::uint64_t> nodes,
    const std::vector<Point>& current, const std::vector<Point>& departure, const Point& center, const Point& backcenter)
{
    std::vector<std::array<Point, 3>> result;
    result.reserve(3 * nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const auto j = (i + 1) % nodes.size();
        // The source cap lies in its authenticated source plane. Its cone
        // about center has identically zero three-dimensional measure.
        result.push_back({backcenter, departure[j], departure[i]});
        if (nodes[i] < nodes[j])
        {
            result.push_back({current[i], departure[i], departure[j]});
            result.push_back({current[i], departure[j], current[j]});
        }
        else
        {
            result.push_back({current[j], current[i], departure[i]});
            result.push_back({current[j], departure[i], departure[j]});
        }
    }
    return result;
}
inline Sweep make_sweep(std::uint64_t face, std::span<const std::uint64_t> nodes,
    const std::map<std::uint64_t, std::pair<Point, Point>>& traces, Value control, Work& work)
{
    if (nodes.size() < 3 || nodes.size() > work.limits.maximum_simplices / 4)
        throw std::invalid_argument("Finite VoF face simplex limit is exhausted.");
    Sweep result;
    result.face = face;
    result.control = control;
    std::set<std::uint64_t> unique;
    for (auto node : nodes)
    {
        if (!unique.insert(node).second)
            throw std::invalid_argument("Finite VoF canonical face repeats a node.");
        const auto& trace = traces.at(node);
        result.current.push_back(trace.first);
        result.departure.push_back(trace.second);
        result.center = add(result.center, trace.first);
        result.backcenter = add(result.backcenter, trace.second);
    }
    result.center = Meshes::bounded_point_average(result.current);
    result.backcenter = Meshes::bounded_point_average(result.departure);
    Point normal{};
    for (std::size_t i = 0; i < nodes.size(); ++i)
        normal = add(normal, cross(subtract(result.current[i], result.center),
            subtract(result.current[(i + 1) % nodes.size()], result.center)));
    const auto triangles = sweep_boundary(nodes, result.current, result.departure, result.center, result.backcenter);
    Sum original;
    for (const auto& triangle : triangles)
        original.add(cone_determinant(result.center, triangle));
    // Only the departure cap depends on the private center. Its linear
    // coefficient follows directly from the determinant's multilinearity.
    Sum coefficient;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const auto j = (i + 1) % nodes.size();
        coefficient.add(dot(normal, cross(subtract(result.departure[j], result.center),
            subtract(result.departure[i], result.center))));
    }
    const auto difference = subtract(multiply(control, {6, 0}), original.value());
    if (difference.value != 0 || difference.absolute_error != 0)
        result.backcenter = Meshes::bounded_point_displaced(result.backcenter,
            scale(normal, divide(difference, coefficient.value())));
    Sum check;
    const auto corrected = sweep_boundary(nodes, result.current, result.departure, result.center, result.backcenter);
    for (const auto& triangle : corrected)
    {
        const auto volume = cone_determinant(result.center, triangle);
        if (volume.value == 0 && volume.absolute_error == 0)
            continue;
        if (!(std::abs(volume.value) > volume.absolute_error))
            throw std::invalid_argument("Finite VoF swept simplex orientation is arithmetically unresolved.");
        auto geometry = Geometry::tetrahedron({result.center, triangle[0], triangle[1], triangle[2]});
        if (!geometry)
            throw std::invalid_argument("Finite VoF positive swept simplex disappeared.");
        work.atom(geometry);
        result.pieces.push_back({std::move(geometry), volume.value > 0 ? 1 : -1});
        check.add(volume);
    }
    const auto residual = subtract(divide(check.value(), {6, 0}), control);
    if (std::abs(residual.value) > residual.absolute_error)
        throw std::invalid_argument("Finite VoF corrected face chain does not match its canonical control.");
    return result;
}
struct OverlayAtom
{
    GeometryPtr geometry;
    int coefficient = 0;
};
/** Each overlay replaces one convex set by complementary descendants. Thus
 * disjointness/containment is inherited; integer multiplicity is never inferred
 * from a net volume sum. Every outside piece retains its original coefficient.
 */
inline std::vector<OverlayAtom> overlay(std::vector<OverlayAtom> atoms, const SignedPiece& piece,
    int incidence, Work& work)
{
    std::vector<OverlayAtom> result;
    for (const auto& atom : atoms)
    {
        work.test();
        auto inside = atom.geometry;
        for (std::size_t face = 0; inside && face < piece.geometry->faces().size(); ++face)
        {
            const auto split = inside->clip_by_face(*piece.geometry, face);
            if (split.outside)
            {
                work.created(split.outside, inside);
                work.push(result, OverlayAtom{split.outside, atom.coefficient});
            }
            if (split.inside) work.created(split.inside, inside);
            inside = split.inside;
        }
        if (inside)
        {
            work.push(result, OverlayAtom{inside, atom.coefficient + incidence * piece.sign});
        }
    }
    return result;
}
struct Atom
{
    std::uint64_t donor = 0, recipient = 0;
    GeometryPtr geometry;
};
inline std::vector<Atom> recipient_partition(std::uint64_t recipient,
    const std::map<std::uint64_t, GeometryPtr>& donors, const std::vector<std::pair<const Sweep*, int>>& faces,
    Work& work, double physical_budget)
{
    std::vector<Atom> result;
    Sum volume;
    for (const auto& [donor, geometry] : donors)
    {
        std::vector<OverlayAtom> atoms{{geometry, donor == recipient ? 1 : 0}};
        for (const auto& [sweep, incidence] : faces)
            for (const auto& piece : sweep->pieces)
                atoms = overlay(std::move(atoms), piece, -incidence, work);
        for (const auto& atom : atoms)
        {
            if (atom.coefficient != 0 && atom.coefficient != 1)
                throw std::invalid_argument("Finite VoF closed-cell pullback has invalid signed multiplicity.");
            if (atom.coefficient)
            {
                volume.add(atom.geometry->volume());
                work.push(result, Atom{donor, recipient, atom.geometry});
            }
        }
    }
    const auto target = donors.at(recipient)->volume();
    const auto difference = subtract(volume.value(), target);
    if (std::abs(difference.value) > difference.absolute_error ||
        difference.absolute_error > std::abs(target.value) * physical_budget)
        throw std::invalid_argument("Finite VoF recipient closure exceeds its original arithmetic/physical budget.");
    return result;
}
} // namespace SimpleFluid::VoF::finite_detail
