/** @file FiniteVoFTransport.hh
 * @brief Finite signed face chains and complementary closed-cell overlays.
 */
#pragma once
#include "FVM/details/FiniteGeometryAccess.hh"
#include "FVM/details/VoFFailure.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/SweptGeometry.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <span>
namespace SimpleFluid::VoF::finite_detail
{
template<ArithmeticMode Mode> using Geometry = PLIC::ConvexCellGeometry<Mode>;
template<ArithmeticMode Mode> using GeometryPtr = PLIC::ConvexGeometryPtr<Mode>;
struct Work
{
    SimpleFluid::PLIC::SweepLimits limits;
    ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
    double volume_tolerance = 8e-15;
    std::size_t bytes = 0, pair_tests = 0;
    PLIC::GeometryAllocationLedger allocations;
    void charge(std::size_t n)
    {
        if (n > limits.byte_budget - std::min(bytes, limits.byte_budget))
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                "Finite VoF cumulative construction byte budget is exhausted.");
        bytes += n;
    }
    void test()
    {
        if (++pair_tests > limits.maximum_pair_tests)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF pair-test budget is exhausted.");
    }
    template<class T> void push(std::vector<T>& values, T value)
    {
        if (values.size() >= limits.maximum_atoms)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF atom buffer capacity is exhausted.");
        if (values.size() == values.capacity())
        {
            const auto capacity = values.capacity();
            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF atom capacity overflows.");
            const auto next = capacity ? 2 * capacity : 1;
            if (next > std::numeric_limits<std::size_t>::max() / sizeof(T))
                throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF atom byte capacity overflows.");
            charge(next * sizeof(T));
            values.reserve(next);
        }
        values.push_back(std::move(value));
    }
    template<ArithmeticMode Mode> void created(const GeometryPtr<Mode>& p, const GeometryPtr<Mode>& source = {})
    {
        if (!source || p->storage_identity() != source->storage_identity())
            allocations.retain<Mode>(p, bytes, limits.byte_budget);
        else
            charge(sizeof(Geometry<Mode>) + sizeof(PLIC::DynamicConvexGeometry));
    }
    template<ArithmeticMode Mode> void atom(const GeometryPtr<Mode>& p)
    {
        charge(sizeof(GeometryPtr<Mode>));
        created(p);
    }

};
template<ArithmeticMode Mode> struct SignedPiece
{
    GeometryPtr<Mode> geometry;
    int sign = 1;
};
template<ArithmeticMode Mode> struct Sweep
{
    std::uint64_t face = 0;
    std::vector<SignedPiece<Mode>> pieces;
    std::vector<PLIC::Point<Mode>> current, departure;
    PLIC::Point<Mode> center, backcenter;
    Value control;
    Value constructed_volume, signed_control_difference;
    long double ordinary_volume_budget = 0;
};
template<ArithmeticMode Mode> inline bool same_point(const PLIC::Point<Mode>& a, const PLIC::Point<Mode>& b)
{
    return PLIC::points_identical<Mode>(a, b);
}
template<ArithmeticMode Mode>
inline Value cone_determinant(const PLIC::Point<Mode>& origin, const std::array<PLIC::Point<Mode>, 3>& triangle)
{
    return runtime_value<Mode>(PLIC::orientation<Mode>({origin, triangle[0], triangle[1], triangle[2]}));
}
template<ArithmeticMode Mode>
inline std::vector<std::array<PLIC::Point<Mode>, 3>> sweep_boundary(std::span<const std::uint64_t> nodes,
    const std::vector<PLIC::Point<Mode>>& current, const std::vector<PLIC::Point<Mode>>& departure,
    const PLIC::Point<Mode>& center, const PLIC::Point<Mode>& backcenter)
{
    std::vector<std::array<PLIC::Point<Mode>, 3>> result;
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
SIMPLEFLUID_FVM_LOCAL Sweep<Mode> prepare_sweep(std::uint64_t face, std::span<const std::uint64_t> nodes,
    const std::map<std::uint64_t, std::pair<PLIC::Point<Mode>, PLIC::Point<Mode>>>& traces, Value control, Work& work,
    long double ordinary_volume_budget = 0);

/** Instantiate the signed tetrahedra only after the scalar recipe needs geometry. */
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL Sweep<Mode> materialize_sweep(
    Sweep<Mode> recipe, std::span<const std::uint64_t> nodes, Work& work);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_EXPORT Sweep<Mode> make_sweep(std::uint64_t face, std::span<const std::uint64_t> nodes,
    const std::map<std::uint64_t, std::pair<PLIC::Point<Mode>, PLIC::Point<Mode>>>& traces, Value control, Work& work,
    long double ordinary_volume_budget = 0);

template<ArithmeticMode Mode> struct OverlayAtom
{
    GeometryPtr<Mode> geometry;
    int coefficient = 0;
};
struct OverlayFace
{
    std::size_t index;
    long double priority;
};
template<ArithmeticMode Mode>
inline std::vector<OverlayFace> overlay_face_order(const Geometry<Mode>& geometry, Work& work)
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
    charge(geometry.points().size(), sizeof(PLIC::Point<Mode>));
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
template<ArithmeticMode Mode>
inline std::vector<OverlayAtom<Mode>> overlay(
    std::vector<OverlayAtom<Mode>> atoms, const SignedPiece<Mode>& piece, int incidence, Work& work)
{
    std::vector<OverlayAtom<Mode>> result;
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
                work.push(result, OverlayAtom<Mode>{split.outside, atom.coefficient});
            }
            if (split.inside) work.created(split.inside, inside);
            inside = split.inside;
        }
        if (inside)
        {
            work.push(result, OverlayAtom<Mode>{inside, atom.coefficient + incidence * piece.sign});
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
template<ArithmeticMode Mode> struct Atom
{
    std::uint64_t donor = 0, recipient = 0;
    GeometryPtr<Mode> geometry;
};
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_EXPORT std::vector<Atom<Mode>> recipient_partition(std::uint64_t recipient,
    const std::map<std::uint64_t, GeometryPtr<Mode>>& donors,
    const std::vector<std::pair<const Sweep<Mode>*, int>>& faces, Work& work, double physical_budget);

} // namespace SimpleFluid::VoF::finite_detail
