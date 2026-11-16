/** @file CartesianIncidenceSource.cc
 * @brief Exact constructor replay over compressed local Cartesian ownership.
 */
#include "geometry/mesh/AMR/CartesianIncidenceSource.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/CartesianIncidenceBudget.hh"
#include "geometry/mesh/AMR/CartesianSupportKey.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include "geometry/mesh/OrthogonalIndexer.hh"
#include <algorithm>
#include <bit>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
class FrozenEnvironment
{
public:
    explicit FrozenEnvironment(const std::fenv_t& environment)
    {
        if (std::fegetenv(&previous_) != 0)
            throw std::invalid_argument("Cannot capture Cartesian replay caller environment.");
        if (std::fesetenv(&environment) != 0)
        {
            (void) std::fesetenv(&previous_);
            throw std::invalid_argument("Cannot install certified Cartesian replay environment.");
        }
    }
    ~FrozenEnvironment() { (void) std::fesetenv(&previous_); }

private:
    std::fenv_t previous_{};
};
constexpr std::size_t owner_bytes = 128;
using ID = std::uint64_t;
using cartesian_detail::add;
using cartesian_detail::product;
using Scope = cartesian_detail::Allocation;
ID scale(unsigned level)
{
    if (level >= 63)
        throw std::invalid_argument("Cartesian incidence level exceeds precision.");
    return ID(1) << level;
}
bool equal(std::span<const char> a, std::span<const char> b)
{
    return std::ranges::equal(a, b);
}
/** Invert only the integer fields written by CartesianKernel::surfaces_from_shape.
 * This is a candidate locator, never an incidence predicate: every candidate is
 * evaluated by that same kernel and its entire emitted key is compared below.
 * In particular the axis/sign are the ACTUAL normal's choices in the key; no
 * assertion about a physical normal or reciprocal arithmetic is needed here. */
struct SupportLocator
{
    std::span<const char> domain;
    ID region = 0, axis = 0, level = 0, plane = 0;
    std::array<ID, 3> transverse{};
};
std::optional<SupportLocator> support_locator(std::span<const char> key)
{
    // The typed decoder is the single grammar owner; it rejects truncated,
    // inconsistent, out-of-range and future frames before any field is used.
    const auto typed = decode_cartesian_support_key(key);
    if (!typed)
        return {};
    SupportLocator result;
    result.domain = typed->domain;
    result.region = typed->region;
    result.axis = typed->axis;
    result.level = typed->level;
    result.plane = typed->plane;
    result.transverse = typed->transverse;
    return result;
}
void write_context(Writer& w, const ContactHaloContext& c)
{
    for (auto n : {c.epoch, c.topology, c.partition, c.round, c.transaction})
        w.word(n);
}
bool inside(const CartesianIncidenceRange& range, const CartesianCellAncestry& c)
{
    if (range.level != c.level)
        return false;
    for (std::size_t a = 0; a < 3; ++a)
        if (c.index[a] < range.lower[a] || c.index[a] >= range.upper[a])
            return false;
    return true;
}
bool cell_less(const CartesianCellAncestry& a, const CartesianCellAncestry& b)
{
    return std::tie(a.level, a.index) < std::tie(b.level, b.index);
}
void compact_ranges(std::vector<CartesianIncidenceRange>& ranges, unsigned axis)
{
    const auto others = std::array<unsigned, 2>{unsigned((axis + 1) % 3), unsigned((axis + 2) % 3)};
    auto less = [&](const auto& a, const auto& b)
    {
        return std::tuple(a.level, a.lower[others[0]], a.upper[others[0]], a.lower[others[1]], a.upper[others[1]],
                   a.lower[axis], a.upper[axis]) < std::tuple(b.level, b.lower[others[0]], b.upper[others[0]],
                                                       b.lower[others[1]], b.upper[others[1]], b.lower[axis],
                                                       b.upper[axis]);
    };
    std::sort(ranges.begin(), ranges.end(), less);
    std::size_t size = 0;
    for (const auto next : ranges)
    {
        bool merge = size && ranges[size - 1].level == next.level && ranges[size - 1].upper[axis] == next.lower[axis];
        for (auto other : others)
            merge = merge && ranges[size - 1].lower[other] == next.lower[other] &&
                    ranges[size - 1].upper[other] == next.upper[other];
        if (merge)
            ranges[size - 1].upper[axis] = next.upper[axis];
        else
            ranges[size++] = next;
    }
    ranges.resize(size);
}
/** Compare the exact existing native scalar descriptor without allocating a
 * block axis vector. The trusted grid factory, not parsing, authorizes replay. */
struct NativeCompare
{
    std::span<const char> bytes;
    template<class T> void scalar(const T& expected)
    {
        if (bytes.size() < sizeof(T) || std::memcmp(bytes.data(), &expected, sizeof(T)))
            throw CartesianSourceAuthenticationFailure("Cartesian native descriptor differs from typed ancestry.");
        bytes = bytes.subspan(sizeof(T));
    }
    void string(std::string_view expected)
    {
        scalar(ID(expected.size()));
        if (bytes.size() < expected.size() || !std::equal(expected.begin(), expected.end(), bytes.begin()))
            throw CartesianSourceAuthenticationFailure("Cartesian native descriptor name differs from typed ancestry.");
        bytes = bytes.subspan(expected.size());
    }
    void done() const
    {
        if (!bytes.empty())
            throw CartesianSourceAuthenticationFailure("Trailing Cartesian native descriptor bytes.");
    }
};
double source_coordinate(const AMRGrid& grid, std::size_t axis, ID index, unsigned level)
{
    return native_cartesian_coordinate(grid, axis, index, level);
}
void validate_native_source(const AMRGrid& grid, const AdaptiveLeaf& leaf)
{
    const auto* cell = std::get_if<CartesianCellAncestry>(&leaf.key.ancestry);
    if (!cell || leaf.key.region != grid.region_id || cell->level > grid.max_level)
        throw CartesianSourceAuthenticationFailure("Foreign Cartesian cell ancestry.");
    for (unsigned axis = 0; axis < 3; ++axis)
        if (cell->index[axis] >= product(product(grid.root_blocks[axis], grid.block_cells[axis]), scale(cell->level)))
            throw CartesianSourceAuthenticationFailure("Cartesian cell lies outside root grid.");
    const auto ordinal = cell->index[0] % grid.block_cells[0] +
                         grid.block_cells[0] * (cell->index[1] % grid.block_cells[1] +
                                                   grid.block_cells[1] * (cell->index[2] % grid.block_cells[2]));
    if (leaf.provider != AdaptiveProviderKind::Cartesian || leaf.material != grid.material_id || leaf.domain.empty() ||
        !leaf.native_recipe || leaf.native_cell != ordinal)
        throw CartesianSourceAuthenticationFailure(
            "Cartesian leaf membership or native ordinal differs from typed ancestry.");
    std::array<ID, 3> block{};
    for (unsigned axis = 0; axis < 3; ++axis)
        block[axis] = cell->index[axis] / grid.block_cells[axis];
    // Native topology validates its arithmetic layout before axis evaluation.
    const OrthogonalIndexer topology(unsigned(grid.block_cells[0]), unsigned(grid.block_cells[1]),
        unsigned(grid.block_cells[2]), false, false, false);
    (void) topology;
    // Legacy recipe construction evaluates every axis before geometry validates
    // any axis. Preserve that arithmetic/validation order without retaining it.
    for (unsigned axis = 0; axis < 3; ++axis)
        for (ID i = 0; i <= grid.block_cells[axis]; ++i)
        {
            volatile double evaluated =
                source_coordinate(grid, axis, product(block[axis], grid.block_cells[axis]) + i, cell->level);
            (void) evaluated;
        }
    std::size_t vertices = 1;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const auto count = add(grid.block_cells[axis], 1);
        if (vertices > std::numeric_limits<std::size_t>::max() / count / 3)
            throw std::overflow_error("Rectilinear entity counts overflow.");
        vertices *= count;
        double previous = 0;
        for (ID i = 0; i <= grid.block_cells[axis]; ++i)
        {
            const auto value =
                source_coordinate(grid, axis, product(block[axis], grid.block_cells[axis]) + i, cell->level);
            if (!std::isfinite(value) || (i && value <= previous))
                throw CartesianSourceAuthenticationFailure("Rectilinear edges must be finite and strictly increasing.");
            previous = value;
        }
    }
    std::array<char, 128> name{};
    char* position = name.data();
    auto text = [&](std::string_view value)
    {
        if (std::size_t(name.data() + name.size() - position) < value.size())
            throw CartesianSourceAuthenticationFailure("Cartesian block name exceeds fixed format bound.");
        position = std::copy(value.begin(), value.end(), position);
    };
    auto number = [&](ID value)
    {
        const auto result = std::to_chars(position, name.data() + name.size(), value);
        if (result.ec != std::errc{})
            throw CartesianSourceAuthenticationFailure("Cartesian block name exceeds fixed format bound.");
        position = result.ptr;
    };
    text("block_");
    number(cell->level);
    for (auto value : block)
    {
        text("_");
        number(value);
    }
    const auto& descriptor = *leaf.native_recipe;
    if (descriptor.kind != NativeFragmentDescriptor::Kind::Cartesian || descriptor.parts.size() != 4)
        throw CartesianSourceAuthenticationFailure("Cartesian native block recipe differs from typed ancestry.");
    NativeCompare view{descriptor.view};
    view.string({name.data(), std::size_t(position - name.data())});
    view.done();
    for (std::size_t part = 0; part < 4; ++part)
    {
        if (!descriptor.parts[part])
            throw CartesianSourceAuthenticationFailure("Null Cartesian native descriptor part.");
        NativeCompare compare{*descriptor.parts[part]};
        if (!part)
        {
            for (auto count : grid.block_cells)
                compare.scalar(ID(count));
            for (unsigned axis = 0; axis < 3; ++axis)
                compare.scalar(std::uint8_t(0));
            for (auto boundary : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
                compare.string(boundary);
        }
        else
        {
            const auto axis = part - 1;
            compare.scalar(ID(grid.block_cells[axis] + 1));
            for (ID i = 0; i <= grid.block_cells[axis]; ++i)
                compare.scalar(
                    source_coordinate(grid, axis, product(block[axis], grid.block_cells[axis]) + i, cell->level));
        }
        compare.done();
    }
}
AdaptiveBytes member_key(
    const CartesianIncidenceSource& source, const AdaptiveLeaf& leaf, std::size_t face, Scope& allocation)
{
    allocation.retain(cartesian_detail::growth(72, 1));
    const auto cell = encode_adaptive_key(leaf.key);
    Writer w;
    allocation.reserve(w.data, add(add(source.identity().size(), cell.size()), 40));
    w.word(0x534643494d454d31ULL);
    w.bytes(source.identity());
    w.bytes(cell);
    w.word(leaf.revision);
    w.word(face);
    return std::move(w.data);
}
void emit_edges(std::span<const char> domain, std::span<const char> member, const SupportFace& original,
    const PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>& second_image, Budget& budget,
    const CartesianEdgeEmit& emit, std::span<const char> line)
{
    for (unsigned image = 0; image < (second_image.is_zero() ? 1U : 2U); ++image)
    {
        Scope construction(budget);
        const auto vertices = original->points().size();
        if (image)
            construction.retain(add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(vertices),
                add(product(vertices, 3 * sizeof(long double)),
                    product(PLIC::canonical_key_size(PLIC::planar_support<ArithmeticMode::Ordinary>(*original)), 4))));
        const auto physical =
            image ? original->translated(PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, second_image))
                  : original;
        construction.retain(cartesian_detail::growth(vertices, sizeof(LinePoint)));
        construction.retain(cartesian_detail::growth(vertices, sizeof(PLIC::LineIdentity<ArithmeticMode::Ordinary>)));
        const auto points = physical->vertex_identities();
        const auto lines = physical->edge_support_identities();
        if (points.size() != lines.size())
            throw std::invalid_argument("Invalid Cartesian tile edge incidence.");
        for (std::size_t edge = 0; edge < points.size(); ++edge)
        {
            Scope encoding(budget);
            const auto canonical = cartesian_detail::identity_key(lines[edge], encoding);
            Writer key;
            encoding.reserve(key.data, add(add(domain.size(), canonical.size()), 16));
            key.bytes(domain);
            key.bytes(canonical);
            if (!line.empty() && !equal(key.data, line))
                continue;
            const auto first = cartesian_detail::identity_key(points[edge], encoding);
            const auto last = cartesian_detail::identity_key(points[(edge + 1) % points.size()], encoding);
            Writer full;
            encoding.reserve(full.data, add(add(member.size(), first.size()), add(last.size(), 40)));
            full.bytes(member);
            full.word(image);
            full.word(edge);
            full.bytes(first);
            full.bytes(last);
            emit({std::move(key.data), std::move(full.data), points[edge], points[(edge + 1) % points.size()], image,
                edge});
        }
    }
}

} // namespace

CartesianEvaluationEnvironment cartesian_evaluation_environment()
{
    CartesianEvaluationEnvironment result;
    const auto rounding = std::fegetround();
    if (rounding < 0)
        throw std::invalid_argument("Cannot inspect Cartesian evaluation rounding control.");
    result.rounding = std::uint64_t(rounding);
#if defined(__i386__) || defined(__x86_64__)
    unsigned short control = 0;
    __asm__ volatile("fnstcw %0" : "=m"(control));
    result.platform = 1;
    result.primary = control;
    result.secondary = _mm_getcsr() & 0xffc0U;
#endif
    return result;
}
double cartesian_frozen_coordinate(
    const AMRGrid& grid, std::size_t axis, std::uint64_t index, unsigned level, const std::fenv_t& environment)
{
    // Preserve exception flags as well as controls, including when controls
    // already match: an inexact native division must not dirty its caller.
    FrozenEnvironment frozen(environment);
    return native_cartesian_coordinate(grid, axis, index, level);
}
bool cartesian_contact_arithmetic_environment_supported() noexcept
{
    // Fast/reassociated arithmetic and excess binary64 evaluation are outside
    // this certificate. Coordinate contraction need not be assumed absent:
    // source, kernel and contact use one out-of-line native evaluation boundary.
#if defined(__FAST_MATH__) || FLT_EVAL_METHOD != 0
    return false;
#else
    if (!std::numeric_limits<double>::is_iec559 || std::numeric_limits<double>::radix != 2 ||
        std::numeric_limits<double>::digits != 53 || std::numeric_limits<long double>::radix != 2 ||
        std::numeric_limits<long double>::digits < 64 || std::numeric_limits<long double>::max_exponent < 1024 ||
        std::numeric_limits<long double>::min_exponent > -1021)
        return false;
    try
    {
        const auto environment = cartesian_evaluation_environment();
        // The current proof recognizes x86 extended arithmetic only. Require
        // nearest rounding, masked exceptions, 64-bit x87 precision, and no
        // SSE flush-to-zero/denormals-are-zero controls.
        return environment.platform == 1 && environment.rounding == FE_TONEAREST &&
               (environment.primary & 0x0f3fU) == 0x033fU && environment.secondary == 0x1f80U;
    }
    catch (const std::invalid_argument&)
    {
        return false;
    }
#endif
}
bool cartesian_grid_identity_equal(const AMRGrid& a, const AMRGrid& b) noexcept
{
    if (a.root_blocks != b.root_blocks || a.block_cells != b.block_cells || a.periodic != b.periodic ||
        a.max_level != b.max_level || a.region_id != b.region_id || a.material_id != b.material_id)
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (std::bit_cast<ID>(a.origin[axis]) != std::bit_cast<ID>(b.origin[axis]) ||
            std::bit_cast<ID>(a.extent[axis]) != std::bit_cast<ID>(b.extent[axis]))
            return false;
    return true;
}
bool cartesian_contact_arithmetic_eligible(const AMRGrid& grid, unsigned level)
{
    if (!cartesian_contact_arithmetic_environment_supported() || level > grid.max_level || grid.max_level >= 63)
        return false;
    constexpr ID exact_integer_limit = ID(1) << 52;
    const auto finest_scale = ID(1) << grid.max_level;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (grid.periodic[axis] || !grid.root_blocks[axis] || grid.block_cells[axis] < 2 ||
            grid.block_cells[axis] % 2 || grid.block_cells[axis] >= std::numeric_limits<unsigned>::max() ||
            !std::isfinite(grid.origin[axis]) || !std::isfinite(grid.extent[axis]) || !(grid.extent[axis] > 0) ||
            grid.root_blocks[axis] > exact_integer_limit / grid.block_cells[axis])
            return false;
        const auto base = ID(grid.root_blocks[axis]) * ID(grid.block_cells[axis]);
        if (base > exact_integer_limit / finest_scale)
            return false;
        const auto count = base << level;
        // Certification is an evaluated-axis proof, not an unbounded scan of
        // a compact logical description. Bound its work before entering it;
        // larger grids retain the explicit path without attempting the proof.
        if (count > (ID(1) << 20))
            return false;
        auto previous = native_cartesian_coordinate(grid, axis, 0, level);
        if (!std::isfinite(previous) || std::abs(previous) > 0x1p100)
            return false;
        for (ID index = 1; index <= count; ++index)
        {
            const auto current = native_cartesian_coordinate(grid, axis, index, level);
            if (!std::isfinite(current) || std::abs(current) > 0x1p100 || !(current > previous))
                return false;
            if (previous != 0 && current != 0)
            {
                int before = 0, after = 0;
                (void) std::frexp(previous, &before);
                (void) std::frexp(current, &after);
                // Two 53-bit binary64 operands spanning at most ten binary
                // exponents have a difference representable in 64 bits,
                // including a possible carry when the signs differ.
                if (std::abs(before - after) > 10)
                    return false;
            }
            const auto width = static_cast<long double>(current) - static_cast<long double>(previous);
            // These deliberately conservative bounds keep all origin/frame
            // translations, binary scalings, face and volume products normal.
            if (!std::isfinite(width) || width < 0x1p-100L || width > 0x1p100L)
                return false;
            previous = current;
        }
    }
    return true;
}
bool CartesianIncidenceSource::eligible(const AdaptiveLeaf& leaf) noexcept
{
    return leaf.provider == AdaptiveProviderKind::Cartesian && !leaf.geometry && leaf.surfaces.empty() &&
           leaf.native_recipe && !leaf.factor_recipe && !leaf.join_recipe &&
           std::holds_alternative<CartesianCellAncestry>(leaf.key.ancestry);
}
bool CartesianIncidenceSource::authenticates_native_leaf(const AMRGrid& grid, const AdaptiveLeaf& leaf)
{
    if (!eligible(leaf) || grid.max_level >= 63)
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!grid.root_blocks[axis] || !grid.block_cells[axis])
            return false;
    try
    {
        validate_native_source(grid, leaf);
        return true;
    }
    catch (const std::invalid_argument&)
    {
        return false;
    }
    catch (const std::overflow_error&)
    {
        return false;
    }
}
std::shared_ptr<const CartesianIncidenceSource> CartesianIncidenceSource::make(const AMRGrid& grid,
    std::span<const AdaptiveLeaf> leaves, const AdaptiveBytes& domain, int owner, std::uint64_t version,
    ContactHaloContext context, Budget& budget, std::shared_ptr<Budget> keeper)
{
    if (owner < 0 || domain.empty())
        throw std::invalid_argument("Cartesian incidence source needs owner and connected domain.");
    cartesian_detail::check_keeper(budget, keeper);
    Scope retained(budget), scratch(budget);
    constexpr auto compatibility_kernel = owner_bytes + sizeof(AMRGrid) + sizeof(void*);
    const std::shared_ptr<Budget> empty;
    const bool native_replay = keeper && (keeper.owner_before(empty) || empty.owner_before(keeper)) &&
        geometry_detail::native_constructor_capacity_profile_supported();
    retained.retain(sizeof(CartesianIncidenceSource) + (native_replay ? 0 : compatibility_kernel));
    auto result = contact_funding_detail::make_controlled<source_control_bytes>(
        budget, keeper, [] { return new CartesianIncidenceSource; }).first;
    result->grid_ = grid;
    result->evaluation_ = cartesian_evaluation_environment();
    result->context_ = context;
    result->owner_ = owner;
    result->version_ = version;
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    retained.reserve(result->domain_, domain.size());
    result->domain_.assign(domain.begin(), domain.end());
    if (native_replay)
    {
        const geometry_detail::NativeConstructionAdmission admission(
            geometry_detail::NativeConstructionCounterAccess::create<Budget>(
                budget.limit, 0, NativeSourceOwnerStore::mirror(result->keeper_)));
        auto replay = cartesian_incidence_replay_kernel(grid, admission);
        result->kernel_ = std::move(replay.owner);
        result->kernel_allocation_ = std::move(replay.envelope);
    }
    else
    {
        scratch.retain(compatibility_kernel);
        result->kernel_ = cartesian_incidence_replay_kernel(grid);
    }
    std::vector<CartesianIncidenceRange> temporary;
    std::vector<ID> revisions;
    scratch.reserve(temporary, leaves.size());
    scratch.reserve(revisions, leaves.size());
    for (const auto& leaf : leaves)
    {
        if (!eligible(leaf) || leaf.domain != result->domain_)
            throw std::invalid_argument("Unsupported Cartesian incidence source leaf remains explicit.");
        validate_native_source(grid, leaf);
        const auto& cell = std::get<CartesianCellAncestry>(leaf.key.ancestry);
        CartesianIncidenceRange range{cell.level, cell.index, cell.index};
        for (auto& upper : range.upper)
            ++upper;
        temporary.push_back(range);
        revisions.push_back(leaf.revision);
    }
    std::sort(temporary.begin(), temporary.end(),
        [](const auto& a, const auto& b) { return std::tie(a.level, a.lower) < std::tie(b.level, b.lower); });
    for (std::size_t i = 1; i < temporary.size(); ++i)
        if (temporary[i].level == temporary[i - 1].level && temporary[i].lower == temporary[i - 1].lower)
            throw std::invalid_argument("Duplicate Cartesian source ancestry.");
    // Search only possible typed parents; uniform grids remain O(N log N).
    for (const auto& entry : temporary)
    {
        auto parent = entry.lower;
        for (unsigned level = entry.level; level;)
        {
            --level;
            for (auto& value : parent)
                value /= 2;
            const auto target = std::pair(level, parent);
            const auto at =
                std::lower_bound(temporary.begin(), temporary.end(), target, [](const auto& candidate, const auto& key)
                    { return std::pair(candidate.level, candidate.lower) < key; });
            if (at != temporary.end() && at->level == level && at->lower == parent)
                throw std::invalid_argument("Overlapping active Cartesian source ancestry.");
        }
    }
    std::sort(revisions.begin(), revisions.end());
    std::size_t best_count = 0;
    for (std::size_t i = 0; i < revisions.size();)
    {
        auto last = std::upper_bound(revisions.begin() + i, revisions.end(), revisions[i]);
        const auto count = std::size_t(last - revisions.begin()) - i;
        if (count > best_count)
        {
            best_count = count;
            result->base_revision_ = revisions[i];
        }
        i += count;
    }
    retained.reserve(result->revisions_, leaves.size() - best_count);
    for (const auto& leaf : leaves)
        if (leaf.revision != result->base_revision_)
            result->revisions_.push_back({std::get<CartesianCellAncestry>(leaf.key.ancestry), leaf.revision});
    std::sort(result->revisions_.begin(), result->revisions_.end(),
        [](const auto& a, const auto& b) { return cell_less(a.cell, b.cell); });
    for (unsigned axis = 0; axis < 3; ++axis)
        compact_ranges(temporary, axis);
    std::sort(temporary.begin(), temporary.end(), [](const auto& a, const auto& b)
        { return std::tie(a.level, a.lower, a.upper) < std::tie(b.level, b.lower, b.upper); });
    retained.reserve(result->ranges_, temporary.size());
    result->ranges_.assign(temporary.begin(), temporary.end());
    result->cell_count_ = leaves.size();
    struct Interval
    {
        unsigned level = 0, axis = 0;
        ID first = 0, last = 0;
    };
    std::vector<Interval> intervals;
    scratch.reserve(intervals, product(temporary.size(), 3));
    for (const auto& range : temporary)
        for (unsigned axis = 0; axis < 3; ++axis)
            intervals.push_back({range.level, axis, range.lower[axis], range.upper[axis]});
    std::sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b)
        { return std::tie(a.level, a.axis, a.first, a.last) < std::tie(b.level, b.axis, b.first, b.last); });
    std::size_t occupied = 0;
    for (const auto next : intervals)
        if (occupied && intervals[occupied - 1].level == next.level && intervals[occupied - 1].axis == next.axis &&
            next.first <= intervals[occupied - 1].last)
            intervals[occupied - 1].last = std::max(intervals[occupied - 1].last, next.last);
        else
            intervals[occupied++] = next;
    intervals.resize(occupied);
    retained.reserve(result->axes_, intervals.size());
    for (const auto& interval : intervals)
    {
        result->axes_.push_back({interval.level, interval.axis, interval.first, {}});
        auto& axis = result->axes_.back();
        const auto count = add(std::size_t(interval.last - interval.first), 1);
        retained.reserve(axis.values, count);
        for (ID index = interval.first;; ++index)
        {
            axis.values.push_back(source_coordinate(grid, interval.axis, index, interval.level));
            if (index == interval.last)
                break;
        }
    }
    // Fixed exact serialization size; every dynamic buffer is admitted before allocation.
    result->build_identity(budget, retained.bytes);
    result->charge_ = retained.bytes + source_control_bytes;
    retained.bytes = 0;
    return result;
}
void CartesianIncidenceSource::build_identity(Budget& budget, std::size_t& retained_bytes)
{
    Scope retained(budget);
    Scope descriptor_scratch(budget);
    descriptor_scratch.retain(cartesian_detail::growth(152, 1));
    const auto kernel_descriptor = kernel_->descriptor();
    std::size_t descriptor_size = add(304, domain_.size());
    descriptor_size = add(descriptor_size, product(ranges_.size(), 56));
    descriptor_size = add(descriptor_size, product(axes_.size(), 32));
    for (const auto& axis : axes_)
        descriptor_size = add(descriptor_size, product(axis.values.size(), 8));
    descriptor_size = add(descriptor_size, product(revisions_.size(), 40));
    Writer w;
    retained.reserve(w.data, descriptor_size);
    w.word(0x5346434953524331ULL);
    w.word(evaluation_.platform);
    w.word(evaluation_.rounding);
    w.word(evaluation_.primary);
    w.word(evaluation_.secondary);
    w.bytes(kernel_descriptor);
    w.bytes(domain_);
    w.word(owner_);
    w.word(version_);
    write_context(w, context_);
    w.word(base_revision_);
    w.word(cell_count_);
    w.word(ranges_.size());
    for (const auto& range : ranges_)
    {
        w.word(range.level);
        for (const auto n : range.lower)
            w.word(n);
        for (const auto n : range.upper)
            w.word(n);
    }
    w.word(axes_.size());
    for (const auto& axis : axes_)
    {
        w.word(axis.level);
        w.word(axis.axis);
        w.word(axis.first);
        w.word(axis.values.size());
        for (const auto value : axis.values)
            w.word(std::bit_cast<ID>(value));
    }
    w.word(revisions_.size());
    for (const auto& revision : revisions_)
    {
        w.word(revision.cell.level);
        for (auto index : revision.cell.index)
            w.word(index);
        w.word(revision.revision);
    }
    if (w.data.size() != descriptor_size)
        throw std::logic_error("Cartesian source descriptor count differs.");
    identity_ = std::move(w.data);
    retained_bytes = add(retained_bytes, retained.bytes);
    retained.bytes = 0;
}
std::shared_ptr<const CartesianIncidenceSource> CartesianIncidenceSource::reevaluated(
    Budget& budget, std::shared_ptr<Budget> keeper) const
{
    cartesian_detail::check_keeper(budget, keeper);
    if (version_ == UINT64_MAX)
        throw std::invalid_argument("Cartesian source version exhausted.");
    if (!keeper && &budget == budget_)
        keeper = keeper_;
    Scope retained(budget);
    constexpr auto compatibility_kernel = owner_bytes + sizeof(AMRGrid) + sizeof(void*);
    retained.retain(sizeof(CartesianIncidenceSource) + (kernel_allocation_ ? 0 : compatibility_kernel));
    auto result = contact_funding_detail::make_controlled<source_control_bytes>(
        budget, keeper, [] { return new CartesianIncidenceSource; }).first;
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    result->grid_ = grid_;
    result->kernel_ = kernel_;
    result->kernel_allocation_ = kernel_allocation_;
    result->owner_ = owner_;
    result->context_ = context_;
    result->version_ = version_ + 1;
    result->base_revision_ = base_revision_;
    result->cell_count_ = cell_count_;
    result->evaluation_ = cartesian_evaluation_environment();
    retained.copy(result->domain_, domain_);
    retained.reserve(result->ranges_, ranges_.size());
    result->ranges_.assign(ranges_.begin(), ranges_.end());
    retained.reserve(result->revisions_, revisions_.size());
    result->revisions_.assign(revisions_.begin(), revisions_.end());
    retained.reserve(result->axes_, axes_.size());
    for (const auto& source_axis : axes_)
    {
        result->axes_.push_back({source_axis.level, source_axis.axis, source_axis.first, {}});
        auto& axis = result->axes_.back();
        retained.reserve(axis.values, source_axis.values.size());
        for (std::size_t i = 0; i < source_axis.values.size(); ++i)
            axis.values.push_back(source_coordinate(grid_, axis.axis, add(axis.first, i), axis.level));
    }
    result->build_identity(budget, retained.bytes);
    result->charge_ = retained.bytes + source_control_bytes;
    retained.bytes = 0;
    return result;
}
NativeContactOwnerFunding CartesianIncidenceSource::native_funding(
    const std::shared_ptr<const CartesianIncidenceSource>& owner, std::size_t bytes) const
{
    auto funding = NativeContactFundingAccess::split_named<source_control_bytes>(owner, bytes,
        [](const CartesianIncidenceSource& value) { return value.charge_ - source_control_bytes; },
        [](const CartesianIncidenceSource& value) -> const auto& { return value.keeper_; });
    if (kernel_allocation_ && !kernel_allocation_->shares_control(kernel_))
        throw std::invalid_argument("Contact replay differs from its actual compact kernel allocation.");
    funding.dependent_fixed = kernel_allocation_;
    return funding;
}
CartesianIncidenceSource::~CartesianIncidenceSource()
{
    kernel_.reset();
    AdaptiveBytes().swap(domain_);
    AdaptiveBytes().swap(identity_);
    decltype(ranges_)().swap(ranges_);
    decltype(axes_)().swap(axes_);
    decltype(revisions_)().swap(revisions_);
    if (charge_)
        budget_->used -= charge_ - source_control_bytes;
}
bool CartesianIncidenceSource::contains(const CartesianCellAncestry& cell) const noexcept
{
    return std::ranges::any_of(ranges_, [&](const auto& range) { return inside(range, cell); });
}
std::uint64_t CartesianIncidenceSource::revision(const CartesianCellAncestry& cell) const
{
    if (!contains(cell))
        throw std::invalid_argument("Foreign Cartesian source recipe selector.");
    const auto at = std::lower_bound(revisions_.begin(), revisions_.end(), cell,
        [](const auto& entry, const auto& key) { return cell_less(entry.cell, key); });
    return at != revisions_.end() && !cell_less(cell, at->cell) ? at->revision : base_revision_;
}
void CartesianIncidenceSource::visit_cell(
    const CartesianCellAncestry& cell, Budget& budget, const CartesianSurfaceEmit& emit) const
{
    Scope construction(budget);
    const auto face_capacity = PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(4);
    construction.retain(add(PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box_construction_capacity_bound(),
        product(6, face_capacity)));
    construction.retain(add(sizeof(AdaptiveLeaf), add(domain_.size(), product(6, sizeof(AdaptiveSurface)))));
    construction.retain(product(6, add(add(64, domain_.size()), std::string{}.capacity())));
    AdaptiveLeaf leaf;
    leaf.key = {grid_.region_id, cell};
    leaf.domain = domain_;
    leaf.material = grid_.material_id;
    leaf.revision = revision(cell);
    leaf.provider = AdaptiveProviderKind::Cartesian;
    leaf.surfaces = kernel_->surfaces(leaf);
    for (std::size_t face = 0; face < leaf.surfaces.size(); ++face)
    {
        Scope encoding(budget);
        const auto packet = cartesian_detail::surface_packet(leaf, face, encoding);
        const auto member = member_key(*this, leaf, face, encoding);
        emit({leaf.key, leaf.revision, face, leaf.surfaces[face], packet, member});
    }
}
CartesianNativeIncidenceCell::~CartesianNativeIncidenceCell()
{
    leaf_ = {};
    // IdentityBudget's pins must die before their originating admission does.
    // The ordinary ledger is scoped with the cell; no canonical rows escape it.
    identities_ = IdentityBudget{};
    if (budget_)
        budget_->used -= charge_;
}
std::unique_ptr<const CartesianNativeIncidenceCell> CartesianIncidenceSource::evaluate_cell(
    std::shared_ptr<const CartesianIncidenceSource> source, const CartesianCellAncestry& cell, Budget& budget,
    std::shared_ptr<Budget> keeper)
{
    if (!source || !source->contains(cell))
        throw std::invalid_argument("Foreign native source evaluation selector.");
    cartesian_detail::check_keeper(budget, keeper);
    if (source->allocation_budget() != &budget && (!keeper || !source->has_arena_keeper()))
        throw std::invalid_argument("Cross-arena native evaluation requires both source and output arena keepers.");
    if (!keeper && source->allocation_budget() == &budget)
        keeper = source->keeper_;
    Scope retained(budget), construction(budget);
    retained.retain(add(sizeof(CartesianNativeIncidenceCell) + owner_bytes, source->domain_.size()));
    if (source->allocation_budget() != &budget)
        retained.retain(source->metadata_bytes());
    auto result = std::unique_ptr<CartesianNativeIncidenceCell>(new CartesianNativeIncidenceCell);
    result->source_ = std::move(source);
    result->keeper_ = std::move(keeper);
    result->evaluation_ = cartesian_evaluation_environment();
    if (std::fegetenv(&result->environment_))
        throw std::invalid_argument("Cannot capture actual native cell evaluation environment.");
    auto& leaf = result->leaf_;
    leaf.key = {result->source_->grid_.region_id, cell};
    leaf.domain = result->source_->domain_;
    leaf.material = result->source_->grid_.material_id;
    leaf.revision = result->source_->revision(cell);
    leaf.provider = AdaptiveProviderKind::Cartesian;
    const auto face_capacity = PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(4);
    construction.retain(add(PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box_construction_capacity_bound(),
        product(6, face_capacity)));
    construction.retain(
        product(6, add(sizeof(AdaptiveSurface), add(64 + leaf.domain.size(), std::string{}.capacity()))));
    // This is the original Cartesian surfaces call, once, with the same typed
    // ancestry/domain/material/revision. The source factory authenticated every
    // original native descriptor before this opt-in can be used.
    leaf.surfaces = result->source_->kernel_->surfaces(leaf);
    retained.retain(product(leaf.surfaces.capacity(), sizeof(AdaptiveSurface)));
    const auto before_ledger = budget.used;
    try
    {
        for (const auto& surface : leaf.surfaces)
        {
            result->identities_.face(budget, *surface.geometry);
            if (surface.native_whole_geometry && surface.native_whole_geometry != surface.geometry)
                result->identities_.face(budget, *surface.native_whole_geometry);
        }
    }
    catch (...)
    {
        retained.bytes = add(retained.bytes, budget.used - before_ledger);
        throw;
    }
    retained.bytes = add(retained.bytes, budget.used - before_ledger);
    for (const auto& surface : leaf.surfaces)
        retained.retain(add(surface.key.capacity(), surface.boundary.capacity()));
    result->budget_ = &budget;
    result->charge_ = std::exchange(retained.bytes, 0);
    return result;
}
void CartesianIncidenceSource::visit_surfaces(Budget& budget, const CartesianSurfaceEmit& emit,
    std::span<const char> support, std::optional<std::size_t> only_range,
    CartesianSourceVisitStatistics* statistics) const
{
    if (only_range && *only_range >= ranges_.size())
        throw std::invalid_argument("Foreign Cartesian source range selector.");
    const auto first = only_range.value_or(0), last = only_range ? first + 1 : ranges_.size();
    const auto locator = support.empty() ? std::optional<SupportLocator>{} : support_locator(support);
    if (locator && (locator->region != grid_.region_id || !equal(locator->domain, domain_)))
        return;
    for (auto r = first; r < last; ++r)
    {
        const auto& range = ranges_[r];
        auto lower = range.lower, upper = range.upper;
        std::array<ID, 3> plane_cells{};
        std::size_t planes = 0;
        if (locator)
        {
            if (range.level < locator->level)
                continue;
            const auto factor = scale(range.level - unsigned(locator->level));
            bool empty = false;
            for (unsigned a = 0; a < 3; ++a)
                if (a != locator->axis)
                {
                    const auto transverse = locator->transverse[a];
                    if (transverse >= std::numeric_limits<ID>::max() / factor)
                    {
                        empty = true;
                        break;
                    }
                    lower[a] = std::max(lower[a], transverse * factor);
                    upper[a] = std::min(upper[a], (transverse + 1) * factor);
                    empty = empty || lower[a] >= upper[a];
                }
            if (empty || locator->plane > std::numeric_limits<ID>::max() / factor)
                continue;
            const auto axis = unsigned(locator->axis);
            const auto plane = locator->plane * factor;
            auto add_plane = [&](ID index)
            {
                if (index >= lower[axis] && index < upper[axis])
                    plane_cells[planes++] = index;
            };
            if (plane)
                add_plane(plane - 1);
            add_plane(plane);
            // Only the upper physical seam wraps to plane zero in the key.
            // The physical authority displacement is still produced by the
            // unchanged kernel; it is never inferred by this integer locator.
            if (grid_.periodic[axis] && !plane)
            {
                const auto total = grid_.root_blocks[axis] * grid_.block_cells[axis] * scale(range.level);
                if (total > 1)
                    add_plane(total - 1);
            }
            if (!planes)
                continue;
        }
        std::array<ID, 3> counts{};
        for (unsigned a = 0; a < 3; ++a)
            counts[a] = locator && a == locator->axis ? planes : upper[a] - lower[a];
        auto index = [&](unsigned axis, ID offset)
        { return locator && axis == locator->axis ? plane_cells[offset] : lower[axis] + offset; };
        for (ID z = 0; z < counts[2]; ++z)
            for (ID y = 0; y < counts[1]; ++y)
                for (ID x = 0; x < counts[0]; ++x)
                {
                    if (statistics)
                        statistics->candidate_cells = add(statistics->candidate_cells, 1);
                    visit_cell({range.level, {index(0, x), index(1, y), index(2, z)}}, budget,
                        [&](const auto& surface)
                        {
                            if (support.empty() || equal(surface.surface.key, support))
                            {
                                if (statistics)
                                    statistics->emitted_surfaces = add(statistics->emitted_surfaces, 1);
                                emit(surface);
                            }
                        });
                }
    }
}
bool CartesianIncidenceSource::visit_support_members(std::span<const char> support_key, Budget& budget,
    const CartesianSurfaceEmit& emit, CartesianSourceVisitStatistics* statistics) const
{
    if (support_key.empty())
        throw std::invalid_argument("Empty Cartesian support membership query.");
    const auto locator = support_locator(support_key);
    if (!locator || locator->region != grid_.region_id || !std::ranges::equal(locator->domain, domain_))
        return false;
    std::size_t emitted = 0;
    visit_surfaces(
        budget,
        [&](const CartesianIncidenceSurface& surface)
        {
            ++emitted;
            emit(surface);
        },
        support_key, {}, statistics);
    return emitted != 0;
}
void visit_cartesian_source_edges(
    const CartesianIncidenceSource& source, Budget& budget, const CartesianEdgeEmit& emit, std::span<const char> line)
{
    source.visit_surfaces(budget,
        [&](const auto& surface)
        {
            // Discovery uses the actual native physical face. It is deliberately
            // distinct from complete overlay tile membership and never certifies it.
            emit_edges(source.domain(), surface.member, surface.surface.geometry, {}, budget, emit, line);
        });
}
void visit_cartesian_tile_edges(std::span<const char> domain, std::span<const char> tile_member,
    const PrivateTileRecipe& tile, Budget& budget, const CartesianEdgeEmit& emit, std::span<const char> line)
{
    if (!tile.face || domain.empty() || tile_member.empty())
        throw std::invalid_argument("Incomplete certified Cartesian tile edge input.");
    emit_edges(domain, tile_member, tile.face, tile.second_image, budget, emit, line);
}
void visit_cartesian_support_members(std::span<const std::shared_ptr<const CartesianIncidenceSource>> sources,
    std::span<const CartesianExplicitSupportMember> exceptions, std::span<const char> support, Budget& budget,
    const CartesianSupportMemberEmit& emit)
{
    if (support.empty())
        throw std::invalid_argument("Empty complete Cartesian support query key.");
    for (const auto& source : sources)
    {
        if (!source)
            throw std::invalid_argument("Null complete Cartesian support source.");
        source->visit_surfaces(
            budget, [&](const auto& occurrence) { emit(occurrence.member, occurrence.packet, source->owner()); },
            support);
    }
    for (const auto& exception : exceptions)
    {
        if (!exception.leaf || exception.owner < 0 || exception.identity.empty() ||
            exception.face >= exception.leaf->surfaces.size())
            throw std::invalid_argument("Incomplete explicit Cartesian support exception.");
        if (!equal(exception.leaf->surfaces[exception.face].key, support))
            continue;
        Scope encoding(budget);
        const auto packet = cartesian_detail::surface_packet(*exception.leaf, exception.face, encoding);
        emit(exception.identity, packet, exception.owner);
    }
}
void visit_cartesian_line_members(std::span<const CartesianCertifiedTileInput> tiles,
    std::span<const CartesianExplicitLineMember> exceptions, std::span<const char> line, Budget& budget,
    const CartesianEdgeEmit& emit)
{
    if (line.empty())
        throw std::invalid_argument("Empty complete Cartesian line query key.");
    for (const auto& input : tiles)
    {
        if (!input.tile)
            throw std::invalid_argument("Null complete Cartesian tile source.");
        visit_cartesian_tile_edges(input.domain, input.member, *input.tile, budget, emit, line);
    }
    for (const auto& exception : exceptions)
        if (equal(exception.line, line))
        {
            if (exception.member.empty())
                throw std::invalid_argument("Empty explicit Cartesian line member identity.");
            Scope scratch(budget);
            scratch.retain(sizeof(CartesianIncidenceEdge) + exception.line.size() + exception.member.size() +
                           exception.subscription.first.payload_bytes() + exception.subscription.last.payload_bytes());
            const CartesianIncidenceEdge edge{
                exception.line, exception.member, exception.subscription.first, exception.subscription.last, 0, 0};
            if (edge.key.capacity() != exception.line.size() || edge.member.capacity() != exception.member.size())
                throw std::invalid_argument("Explicit Cartesian line capacity differs from admission.");
            emit(edge);
        }
}
void visit_cartesian_changed_supports(const CartesianIncidenceSource* previous, const CartesianIncidenceSource* next,
    Budget& budget, const CartesianBorrowedCallback<std::span<const char>>& emit)
{
    for (const auto* source : {previous, next})
        if (source)
            source->visit_surfaces(budget, [&](const auto& occurrence) { emit(occurrence.surface.key); });
}
void replay_cartesian_recipe(const CartesianReplayRecipe& recipe, Budget& budget, const CartesianIdentityEmit& emit)
{
    for (const auto& selector : recipe.selectors)
    {
        if (selector.source >= recipe.sources.size() || !recipe.sources[selector.source])
            throw std::invalid_argument("Foreign Cartesian replay source selector.");
        if (selector.cell && selector.range)
            throw std::invalid_argument("Ambiguous Cartesian replay range and cell selector.");
        const auto& source = *recipe.sources[selector.source];
        auto surface_emit = [&](const CartesianIncidenceSurface& occurrence)
        {
            if ((selector.face && *selector.face != occurrence.ordinal) ||
                (!selector.support.empty() && occurrence.surface.key != selector.support))
                return;
            if (selector.output == CartesianRecipeOutput::SurfacePacket)
                return emit(occurrence.packet);
            if (selector.output == CartesianRecipeOutput::SurfaceKey)
                return emit(occurrence.surface.key);
            Scope construction(budget);
            auto face = occurrence.surface.geometry;
            const auto vertices = face->points().size();
            if (selector.authority_image)
            {
                construction.retain(
                    add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(vertices),
                        product(vertices, 3 * sizeof(long double))));
                face = face->translated(occurrence.surface.authority_image);
            }
            if (selector.reverse)
            {
                construction.retain(
                    add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(vertices),
                        product(vertices, 3 * sizeof(long double))));
                face = face->reversed();
            }
            if (selector.output == CartesianRecipeOutput::FaceKey)
            {
                const auto key = cartesian_detail::face_key(face, construction);
                emit(key);
                return;
            }
            const auto count = selector.count ? selector.count : vertices;
            if (selector.first >= vertices || count > vertices)
                throw std::invalid_argument("Foreign Cartesian replay vertex selector.");
            construction.retain(cartesian_detail::growth(vertices, sizeof(LinePoint)));
            construction.retain(
                cartesian_detail::growth(vertices, sizeof(PLIC::LineIdentity<ArithmeticMode::Ordinary>)));
            const auto points = face->vertex_identities();
            const auto lines = face->edge_support_identities();
            for (std::size_t i = 0; i < count; ++i)
            {
                Scope encoding(budget);
                const auto index = (selector.first + i) % vertices;
                const auto key = selector.output == CartesianRecipeOutput::PointKeys
                                     ? cartesian_detail::identity_key(points[index], encoding)
                                     : cartesian_detail::identity_key(lines[index], encoding);
                emit(key);
            }
        };
        if (selector.cell)
            source.visit_cell(*selector.cell, budget, surface_emit);
        else
            source.visit_surfaces(budget, surface_emit, {}, selector.range);
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
