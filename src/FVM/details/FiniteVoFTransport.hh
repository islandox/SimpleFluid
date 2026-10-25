/** @file FiniteVoFTransport.hh
 * @brief Finite signed face chains and complementary closed-cell overlays.
 */
#pragma once
#include "geometry/PLIC/SweptGeometry.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <span>
namespace SimpleFluid::VoF::finite_detail
{
using Geometry = PLIC::DynamicConvexGeometry;
using GeometryPtr = PLIC::DynamicConvexGeometryPtr;
struct Work
{
    SimpleFluid::PLIC::SweepLimits limits;
    ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
    double volume_tolerance = 8e-15;
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
    Value constructed_volume, signed_control_difference;
    long double ordinary_volume_budget = 0;
};
inline bool same_point(const Point& a, const Point& b)
{
    return PLIC::dynamic_points_identical(a, b);
}
inline Value cone_determinant(const Point& origin, const std::array<Point, 3>& triangle)
{
    return PLIC::dynamic_orientation({origin, triangle[0], triangle[1], triangle[2]});
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
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_EXPORT Sweep make_sweep(std::uint64_t face, std::span<const std::uint64_t> nodes,
    const std::map<std::uint64_t, std::pair<Point, Point>>& traces, Value control, Work& work,
    long double ordinary_volume_budget = 0);

struct OverlayAtom
{
    GeometryPtr geometry;
    int coefficient = 0;
};
struct OverlayFace
{
    std::size_t index;
    long double priority;
};
inline std::vector<OverlayFace> overlay_face_order(const Geometry& geometry, Work& work)
{
    const auto charge = [&](std::size_t count, std::size_t size)
    {
        if (count > std::numeric_limits<std::size_t>::max() / size)
            throw std::invalid_argument("Finite VoF face-order capacity overflows.");
        work.charge(count * size);
    };
    charge(geometry.faces().size(), sizeof(OverlayFace));
    std::vector<OverlayFace> result;
    result.reserve(geometry.faces().size());
    for (std::size_t f = 0; f < geometry.faces().size(); ++f)
        result.push_back({f, 0});
    charge(geometry.points().size(), sizeof(Point));
    const auto points = geometry.local_points();
    long double magnitude = 0;
    for (const auto& p : points)
        for (const auto x : p.value)
        {
            if (!std::isfinite(x))
                return result;
            magnitude = std::max(magnitude, std::abs(x));
        }
    if (!(magnitude > 0))
        return result;
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    const auto coordinates = [&](std::size_t index)
    {
        std::array<long double, 3> p;
        for (std::size_t axis = 0; axis < 3; ++axis)
            p[axis] = std::scalbn(points[index].value[axis], -exponent);
        return p;
    };
    for (auto& entry : result)
    {
        const auto& face = geometry.faces()[entry.index];
        const auto a = coordinates(face.front());
        std::array<long double, 3> normal{};
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            work.test(); // Preprocessing shares the existing pair-work limit.
            auto b = coordinates(face[i]), c = coordinates(face[i + 1]);
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                b[axis] -= a[axis];
                c[axis] -= a[axis];
            }
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const auto next = (axis + 1) % 3, last = (axis + 2) % 3;
                normal[axis] += b[next] * c[last] - b[last] * c[next];
            }
        }
        for (const auto n : normal)
        {
            if (!std::isfinite(n))
                return result; // Original indices still have their initial order.
            entry.priority = std::max(entry.priority, std::abs(n));
        }
    }
    // Nominal areas schedule work only. Every geometric decision still uses
    // the selected finite kernel. Broad faces first avoid extending thin-piece side
    // planes unnecessarily through the full donor; original indices break ties.
    std::sort(result.begin(), result.end(), [&](const auto& a, const auto& b)
    {
        work.test();
        return a.priority != b.priority ? a.priority > b.priority : a.index < b.index;
    });
    return result;
}
/** Each overlay replaces one convex set by complementary descendants. Thus
 * disjointness/containment is inherited; integer multiplicity is never inferred
 * from a net volume sum. Every outside piece retains its original coefficient.
 */
inline std::vector<OverlayAtom> overlay(std::vector<OverlayAtom> atoms, const SignedPiece& piece,
    int incidence, Work& work)
{
    std::vector<OverlayAtom> result;
    std::vector<OverlayFace> order;
    for (const auto& atom : atoms)
    {
        work.test();
        if (atom.geometry->bounds_prove_disjoint_interiors(*piece.geometry))
        {
            // Disjoint spatial bounds leave this coefficient
            // unchanged. Retain even zero atoms for later signed pieces, and
            // charge the output buffer through the same resource guard.
            work.push(result, atom);
            continue;
        }
        if (order.empty())
            order = overlay_face_order(*piece.geometry, work);
        const auto first_output = result.size();
        auto inside = atom.geometry;
        for (const auto& face : order)
        {
            if (!inside)
                break;
            const auto split = inside->clip_by_face(*piece.geometry, face.index);
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
        else
        {
            // The emitted complements cover the original atom with the same
            // coefficient. Empty intersection permits reconstruction
            // reuse; retain all charges for the work already performed.
            result.resize(first_output);
            work.push(result, atom);
        }
    }
    return result;
}
struct Atom
{
    std::uint64_t donor = 0, recipient = 0;
    GeometryPtr geometry;
};
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_EXPORT std::vector<Atom> recipient_partition(std::uint64_t recipient,
    const std::map<std::uint64_t, GeometryPtr>& donors, const std::vector<std::pair<const Sweep*, int>>& faces,
    Work& work, double physical_budget);

inline Sweep make_sweep(std::uint64_t face, std::span<const std::uint64_t> nodes,
    const std::map<std::uint64_t, std::pair<Point, Point>>& traces, Value control, Work& work,
    long double ordinary_volume_budget = 0)
{
    if (work.arithmetic_mode == ArithmeticMode::Ordinary)
        return make_sweep<ArithmeticMode::Ordinary>(face, nodes, traces, control, work, ordinary_volume_budget);
    return make_sweep<ArithmeticMode::BoundedCertification>(face, nodes, traces, control, work, ordinary_volume_budget);
}
inline std::vector<Atom> recipient_partition(std::uint64_t recipient,
    const std::map<std::uint64_t, GeometryPtr>& donors, const std::vector<std::pair<const Sweep*, int>>& faces,
    Work& work, double physical_budget)
{
    if (work.arithmetic_mode == ArithmeticMode::Ordinary)
        return recipient_partition<ArithmeticMode::Ordinary>(recipient, donors, faces, work, physical_budget);
    return recipient_partition<ArithmeticMode::BoundedCertification>(recipient, donors, faces, work, physical_budget);
}
} // namespace SimpleFluid::VoF::finite_detail
