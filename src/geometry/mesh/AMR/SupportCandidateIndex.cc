#include "geometry/mesh/AMR/SupportCandidateIndex.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
struct Rectangle
{
    std::array<unsigned, 2> axes{};
    std::array<long double, 2> lower{}, upper{};
    unsigned normal = 0;
    long double plane = 0;
};
std::optional<Rectangle> rectangle(const SupportFace& face)
{
    if (!face || face->points().size() != 4)
        return {};
    const auto& points = face->points();
    Rectangle result;
    unsigned fixed = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (std::all_of(points.begin(), points.end(), [&](const auto& point) { return point[axis] == points[0][axis]; }))
        {
            result.normal = axis;
            ++fixed;
        }
    if (fixed != 1)
        return {};
    result.plane = points[0][result.normal];
    unsigned dimension = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (axis != result.normal)
            result.axes[dimension++] = axis;
    for (unsigned d = 0; d < 2; ++d)
    {
        result.lower[d] = result.upper[d] = points[0][result.axes[d]];
        for (const auto& point : points)
        {
            result.lower[d] = std::min(result.lower[d], point[result.axes[d]]);
            result.upper[d] = std::max(result.upper[d], point[result.axes[d]]);
        }
        if (!(result.lower[d] < result.upper[d]))
            return {};
    }
    unsigned corners = 0;
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        unsigned corner = 0, changed = 0;
        for (unsigned d = 0; d < 2; ++d)
        {
            const auto coordinate = points[i][result.axes[d]];
            if (coordinate == result.upper[d])
                corner |= 1u << d;
            else if (coordinate != result.lower[d])
                return {};
            changed += coordinate != points[(i + 1) % points.size()][result.axes[d]];
        }
        if (changed != 1 || (corners & (1u << corner)))
            return {};
        corners |= 1u << corner;
    }
    return corners == 15 ? std::optional<Rectangle>(result) : std::nullopt;
}
bool qualified(const GeometryConstructionControls& controls)
{
    return std::numeric_limits<long double>::radix == 2 && std::numeric_limits<long double>::digits == 64 &&
           controls.supported && controls.rounding == FE_TONEAREST && (controls.x87_control & 0xf3f) == 0x33f &&
           (controls.simd_control & 0xffc0) == 0x1f80;
}
struct RestoreEnvironment
{
    std::fenv_t value{};
    bool saved = std::fegetenv(&value) == 0;
    ~RestoreEnvironment()
    {
        if (saved)
            (void) std::fesetenv(&value);
    }
};
/** Raw coordinates are N*q, |N|<=2^14, q=2^e with -128<=e<=128.
 * Raw differences and rectangle distances are exact within 64-bit precision.
 * Earlier orthogonal cuts retain a separating component exactly; identical
 * endpoint order yields identical parallel-edge cut coordinates. A separating
 * edge therefore empties the loop without interpolation. The bounded exponent
 * window keeps crossing denominators and degree-four intermediates normal and
 * finite. No bare box claim is made for other represented polygons. */
bool certified_group(std::span<const SupportOccurrence> entries, Rectangle& frame)
{
    if (entries.empty())
        return false;
    RestoreEnvironment restore;
    if (!restore.saved)
        return false;
    const auto first = rectangle(entries.front().face);
    if (!first)
        return false;
    frame = *first;
    int exponent = std::numeric_limits<int>::max();
    for (const auto& entry : entries)
    {
        const auto shape = rectangle(entry.face);
        if (!shape || shape->normal != frame.normal || shape->plane != frame.plane)
            return false;
        for (const auto& point : entry.face->points())
            for (const auto coordinate : point)
            {
                if (!std::isfinite(coordinate))
                    return false;
                if (coordinate == 0)
                    continue;
                int power = 0;
                const auto fraction = std::frexp(std::abs(coordinate), &power);
                const auto significand = static_cast<std::uint64_t>(std::scalbn(fraction, 64));
                exponent = std::min(exponent, power - 64 + int(std::countr_zero(significand)));
            }
    }
    if (exponent < -128 || exponent > 128)
        return false;
    for (const auto& entry : entries)
        for (const auto& point : entry.face->points())
            for (const auto coordinate : point)
            {
                const auto integer = std::scalbn(coordinate, -exponent);
                if (!std::isfinite(integer) || std::abs(integer) > 16384 || std::trunc(integer) != integer)
                    return false;
            }
    return true;
}
} // namespace

SupportCandidateIndex::SupportCandidateIndex(
    std::span<const SupportOccurrence> entries, std::size_t budget, const SupportCharge& charge)
    : size_(entries.size())
{
    const auto controls = geometry_construction_controls();
    Rectangle frame;
    if (size_ < 16 || !qualified(controls) || !certified_group(entries, frame))
        return;
    if (size_ > budget / (2 * sizeof(Node) + sizeof(std::size_t)) || size_ > nodes_.max_size() / 2 ||
        size_ > members_.max_size())
        throw std::invalid_argument("Support candidate index budget exhausted.");
    charge(2 * size_ * sizeof(Node) + size_ * sizeof(std::size_t));
    nodes_.reserve(2 * size_);
    members_.reserve(size_);
    if (nodes_.capacity() != 2 * size_ || members_.capacity() != size_)
        throw std::invalid_argument("Support candidate index capacity differs from admission.");
    for (std::size_t i = 0; i < size_; ++i)
    {
        const auto shape = *rectangle(entries[i].face);
        nodes_.push_back({{shape.lower, shape.upper}, i, absent, absent});
        members_.push_back(i);
    }
    root_ = build(0, size_, 0);
    rounding_ = controls.rounding;
    x87_ = controls.x87_control;
    simd_ = controls.simd_control;
    certified_ = true;
}
bool SupportCandidateIndex::stable() const noexcept
{
    const auto controls = geometry_construction_controls();
    return certified_ && controls.supported && controls.rounding == rounding_ && controls.x87_control == x87_ &&
           controls.simd_control == simd_;
}
std::size_t SupportCandidateIndex::build(std::size_t first, std::size_t last, unsigned axis)
{
    if (last - first == 1)
        return members_[first];
    const auto middle = first + (last - first) / 2;
    std::nth_element(members_.begin() + first, members_.begin() + middle, members_.begin() + last,
        [&](std::size_t a, std::size_t b)
        {
            const auto& x = nodes_[a].bounds;
            const auto& y = nodes_[b].bounds;
            if (x.lower[axis] != y.lower[axis])
                return x.lower[axis] < y.lower[axis];
            if (x.upper[axis] != y.upper[axis])
                return x.upper[axis] < y.upper[axis];
            if (x.lower[axis ^ 1] != y.lower[axis ^ 1])
                return x.lower[axis ^ 1] < y.lower[axis ^ 1];
            return a < b;
        });
    const auto left = build(first, middle, axis ^ 1), right = build(middle, last, axis ^ 1);
    Bounds bounds;
    for (unsigned d = 0; d < 2; ++d)
    {
        bounds.lower[d] = std::min(nodes_[left].bounds.lower[d], nodes_[right].bounds.lower[d]);
        bounds.upper[d] = std::max(nodes_[left].bounds.upper[d], nodes_[right].bounds.upper[d]);
    }
    const auto result = nodes_.size();
    nodes_.push_back({bounds, absent, left, right});
    return result;
}
void SupportCandidateIndex::collect(std::size_t node, const Bounds& bounds, std::size_t limit)
{
    const auto& value = nodes_[node];
    for (unsigned d = 0; d < 2; ++d)
        if (value.bounds.upper[d] < bounds.lower[d] || bounds.upper[d] < value.bounds.lower[d])
            return;
    if (value.member != absent)
    {
        if (value.member < limit)
            members_.push_back(value.member);
        return;
    }
    collect(value.left, bounds, limit);
    collect(value.right, bounds, limit);
}
void SupportCandidateIndex::visit(
    std::size_t source, std::size_t limit, const BorrowedSupportCallback<std::size_t>& visitor)
{
    if (source >= size_ || limit > size_)
        throw std::out_of_range("Support candidate query exceeds its immutable input.");
    const auto exhaustive = [&](std::size_t first)
    {
        for (auto i = first; i < limit; ++i)
            visitor(i);
    };
    if (!stable())
    {
        exhaustive(0);
        return;
    }
    members_.clear();
    collect(root_, nodes_[source].bounds, limit);
    std::sort(members_.begin(), members_.end());
    std::size_t next = 0;
    for (const auto member : members_)
    {
        if (!stable())
        {
            exhaustive(next);
            return;
        }
        visitor(member);
        next = member + 1;
    }
    if (!stable())
        exhaustive(next);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
