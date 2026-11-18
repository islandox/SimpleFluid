#include "geometry/mesh/NativeFragmentDecodePlan.hh"
#include "geometry/PLIC/GeometryCodecTags.hh"
#include "geometry/PLIC/details/NativeConstructionWireAccess.hh"
#include "geometry/VoFWireProtocol.hh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
namespace
{
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Native constructor count overflows.");
    return a + b;
}
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Native constructor extent overflows.");
    return a * b;
}
struct Cursor
{
    std::span<const char> bytes;
    std::size_t position = 0;
    std::uint8_t byte()
    {
        if (position == bytes.size())
            throw std::invalid_argument("Truncated native constructor preflight.");
        return static_cast<std::uint8_t>(bytes[position++]);
    }
    std::uint64_t word()
    {
        std::uint64_t value = 0;
        for (unsigned index = 0; index != 8; ++index)
            value |= std::uint64_t(byte()) << (8 * index);
        return value;
    }
    template<class T> T native()
    {
        if (sizeof(T) > bytes.size() - position)
            throw std::invalid_argument("Truncated native constructor scalar.");
        T value;
        std::memcpy(&value, bytes.data() + position, sizeof(T));
        position += sizeof(T);
        return value;
    }
    std::size_t count()
    {
        const auto value = word();
        if (value > 65536 || value > bytes.size() - position)
            throw std::invalid_argument("Excessive constructed native count.");
        return static_cast<std::size_t>(value);
    }
    std::span<const char> take(std::size_t count)
    {
        if (count > bytes.size() - position)
            throw std::invalid_argument("Truncated native constructor subpacket.");
        auto result = bytes.subspan(position, count);
        position += count;
        return result;
    }
    std::span<const char> native_bytes()
    {
        const auto count = native<std::uint64_t>();
        if (count > bytes.size() - position)
            throw std::invalid_argument("Native constructor byte extent exceeds its packet.");
        return take(static_cast<std::size_t>(count));
    }
    double ordinary_scalar() { return PLIC::details::NativeConstructionWireAccess::ordinary_scalar(bytes, position); }
    void finish() const
    {
        if (position != bytes.size())
            throw std::invalid_argument("Trailing native constructor preflight bytes.");
    }
};
void tag(Cursor& cursor, PLIC::GeometryTag expected)
{
    if (cursor.byte() != static_cast<std::uint8_t>(expected) || cursor.byte() != 2)
        throw std::invalid_argument("Unsupported constructed native preflight schema.");
}
void scan_topology(std::span<const char> packet, NativeFragmentDecodePlan::Counts& counts)
{
    Cursor cursor{packet};
    packet_codec::read_header(
        cursor, packet_codec::Family::ExtrudedTopologyBase, std::uint8_t{2}, packet_codec::Record::Value);
    const auto nodes = cursor.word();
    counts.cells = cursor.count();
    if (!nodes || nodes > std::numeric_limits<unsigned>::max() || !counts.cells)
        throw std::invalid_argument("Invalid constructed native base extent.");
    counts.nodes = static_cast<std::size_t>(nodes);
    for (std::size_t cell = 0; cell < counts.cells; ++cell)
    {
        const auto count = cursor.count();
        if (count < 3)
            throw std::invalid_argument("Native base polygon has fewer than three vertices.");
        counts.loop_incidence = sum(counts.loop_incidence, count);
        counts.maximum_loop = std::max(counts.maximum_loop, count);
        for (std::size_t index = 0; index < count; ++index)
            if (cursor.word() >= nodes)
                throw std::invalid_argument("Native base node ordinal exceeds its extent.");
    }
    counts.boundary_edges = cursor.count();
    for (std::size_t edge = 0; edge < counts.boundary_edges; ++edge)
    {
        if (cursor.word() >= nodes || cursor.word() >= nodes)
            throw std::invalid_argument("Native boundary edge ordinal exceeds its extent.");
        const auto length = cursor.word();
        if (length > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Native boundary label extent overflows.");
        counts.boundary_bytes = sum(counts.boundary_bytes, static_cast<std::size_t>(length));
        (void) cursor.take(static_cast<std::size_t>(length));
    }
    cursor.finish();
}
void scan_base(std::span<const char> packet, std::span<const char> topology, NativeFragmentDecodePlan::Counts& counts)
{
    Cursor cursor{packet};
    tag(cursor, PLIC::GeometryTag::ExtrusionBase);
    const auto length = cursor.word();
    if (length > std::numeric_limits<std::size_t>::max() ||
        !std::ranges::equal(cursor.take(static_cast<std::size_t>(length)), topology))
        throw std::invalid_argument("Constructed native base belongs to different topology bytes.");
    counts.point_recipes = cursor.count();
    for (std::size_t recipe = 0; recipe < counts.point_recipes; ++recipe)
    {
        for (unsigned axis = 0; axis != 3; ++axis)
            (void) cursor.ordinary_scalar();
        const auto parents = cursor.count();
        counts.point_parents = sum(counts.point_parents, parents);
        for (std::size_t parent = 0; parent < parents; ++parent)
            if (cursor.word() >= recipe)
                throw std::invalid_argument("Native point recipe is not a forward-only ancestry DAG.");
    }
    counts.active_points = cursor.count();
    if (counts.active_points != counts.nodes)
        throw std::invalid_argument("Constructed native active point extent differs from topology.");
    for (std::size_t point = 0; point < counts.active_points; ++point)
        if (cursor.word() >= counts.point_recipes)
            throw std::invalid_argument("Native active point references an unknown recipe.");
    cursor.finish();
}
void scan_axis(std::span<const char> values, std::span<const char> selection, NativeFragmentDecodePlan::Counts& counts)
{
    Cursor cursor{values};
    tag(cursor, PLIC::GeometryTag::AxisValues);
    counts.axial_recipes = cursor.count();
    for (std::size_t recipe = 0; recipe < counts.axial_recipes; ++recipe)
    {
        (void) cursor.ordinary_scalar();
        const auto parents = cursor.count();
        counts.axial_parents = sum(counts.axial_parents, parents);
        for (std::size_t parent = 0; parent < parents; ++parent)
            if (cursor.word() >= recipe)
                throw std::invalid_argument("Native axial recipe is not a forward-only ancestry DAG.");
    }
    cursor.finish();
    Cursor selected{selection};
    counts.selected_axial = selected.count();
    if (counts.selected_axial < 2)
        throw std::invalid_argument("Constructed native axis selection is incomplete.");
    for (std::size_t point = 0; point < counts.selected_axial; ++point)
        if (selected.word() >= counts.axial_recipes)
            throw std::invalid_argument("Native axis selection references an unknown recipe.");
    selected.finish();
}
} // namespace

NativeFragmentDecodePlan NativeFragmentDecodePlan::scan(const std::shared_ptr<const NativeFragmentDescriptor>& recipe)
{
    if (!recipe || recipe->parts.size() != 4 ||
        (recipe->kind != NativeFragmentDescriptor::Kind::Cartesian &&
            recipe->kind != NativeFragmentDescriptor::Kind::ConstructedProduct))
        throw std::invalid_argument("Native descriptor requires a separate admitted legacy storage planner.");
    NativeFragmentDecodePlan plan;
    plan.recipe_ = recipe;
    plan.kind_ = recipe->kind;
    plan.view_ = recipe->view;
    for (std::size_t part = 0; part != 4; ++part)
    {
        if (!recipe->parts[part])
            throw std::invalid_argument("Native constructor descriptor has a null part.");
        plan.parts_[part] = recipe->parts[part];
        plan.spans_[part] = *recipe->parts[part];
    }
    Cursor view{plan.view_};
    plan.counts_.name_bytes = view.native_bytes().size();
    if (recipe->kind == NativeFragmentDescriptor::Kind::Cartesian)
    {
        Cursor topology{plan.spans_[0]};
        std::array<std::size_t, 3> extents{};
        plan.counts_.cells = plan.counts_.nodes = 1;
        for (auto& extent : extents)
        {
            const auto count = topology.native<std::uint64_t>();
            if (!count || count >= std::numeric_limits<unsigned>::max())
                throw std::invalid_argument("Invalid Cartesian native constructor extent.");
            extent = static_cast<std::size_t>(count);
            plan.counts_.cells = product(plan.counts_.cells, extent);
            plan.counts_.nodes = product(plan.counts_.nodes, sum(extent, 1));
        }
        for (unsigned axis = 0; axis != 3; ++axis)
            if (topology.byte())
                throw std::invalid_argument("Cartesian native fragments require nonperiodic local topology.");
        for (unsigned boundary = 0; boundary != 6; ++boundary)
            plan.counts_.boundary_bytes = sum(plan.counts_.boundary_bytes, topology.native_bytes().size());
        topology.finish();
        for (unsigned axis = 0; axis != 3; ++axis)
        {
            Cursor coordinates{plan.spans_[axis + 1]};
            const auto count = coordinates.native<std::uint64_t>();
            if (count != extents[axis] + 1)
                throw std::invalid_argument("Cartesian native axis extent differs from its topology.");
            double previous = 0;
            for (std::size_t point = 0; point < count; ++point)
            {
                const auto value = coordinates.native<double>();
                if (!std::isfinite(value) || (point && !(value > previous)))
                    throw std::invalid_argument("Invalid Cartesian native coordinate sequence.");
                previous = value;
            }
            coordinates.finish();
        }
        view.finish();
        return plan;
    }
    plan.counts_.layers = view.native<std::uint64_t>();
    if (!plan.counts_.layers || plan.counts_.layers > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Invalid constructed native layer extent.");
    const auto aliases = view.native<std::uint64_t>();
    if (aliases > (view.bytes.size() - view.position) / (2 * sizeof(std::uint64_t)))
        throw std::invalid_argument("Native alias count exceeds its packet.");
    plan.counts_.aliases = static_cast<std::size_t>(aliases);
    for (std::size_t alias = 0; alias < plan.counts_.aliases; ++alias)
    {
        plan.counts_.alias_bytes = sum(plan.counts_.alias_bytes, view.native_bytes().size());
        plan.counts_.alias_bytes = sum(plan.counts_.alias_bytes, view.native_bytes().size());
    }
    Cursor run{view.native_bytes()};
    tag(run, PLIC::GeometryTag::ProductView);
    plan.counts_.first_layer = run.word();
    if (run.word() != plan.counts_.layers)
        throw std::invalid_argument("Native product view and layer extent disagree.");
    run.finish();
    view.finish();
    scan_topology(plan.spans_[0], plan.counts_);
    scan_base(plan.spans_[1], plan.spans_[0], plan.counts_);
    scan_axis(plan.spans_[2], plan.spans_[3], plan.counts_);
    if (plan.counts_.first_layer >= plan.counts_.selected_axial ||
        plan.counts_.layers >= plan.counts_.selected_axial - plan.counts_.first_layer)
        throw std::invalid_argument("Native product view exceeds its selected axis.");
    return plan;
}
bool NativeFragmentDecodePlan::matches(const std::shared_ptr<const NativeFragmentDescriptor>& recipe) const noexcept
{
    if (!recipe || recipe->kind != kind_ || recipe.get() != recipe_.get() || recipe.owner_before(recipe_) ||
        recipe_.owner_before(recipe) || recipe->view.data() != view_.data() || recipe->view.size() != view_.size() ||
        recipe->parts.size() != 4)
        return false;
    for (std::size_t part = 0; part != 4; ++part)
        if (!recipe->parts[part] || recipe->parts[part].get() != parts_[part].get() ||
            recipe->parts[part].owner_before(parts_[part]) || parts_[part].owner_before(recipe->parts[part]) ||
            recipe->parts[part]->data() != spans_[part].data() || recipe->parts[part]->size() != spans_[part].size())
            return false;
    return true;
}
} // namespace SimpleFluid::Meshes
