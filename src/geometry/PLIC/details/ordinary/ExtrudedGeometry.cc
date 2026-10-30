#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/ordinary/ExtrudedConstructionInternal.hh"

namespace SimpleFluid::PLIC
{
namespace ordinary_extrusion_detail
{
void write_point_recipes(WireWriter& w, const ExtrudedBaseData& base)
{
    w.word(base.recipes.size());
    for (const auto& p : base.recipes)
    {
        for (auto x : p.root)
            w.real(x);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(base.active.size());
    for (auto id : base.active)
        w.word(id);
}

void read_point_recipes(WireReader& r, std::vector<PointRecipe>& points, std::vector<std::size_t>& active)
{
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        PointRecipe p;
        for (auto& x : p.root)
            x = r.real();
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(r.word());
        points.push_back(std::move(p));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        active.push_back(r.word());
}

void read_axis_parts(std::span<const char> packet, std::vector<ScalarRecipe>& roots, std::vector<std::size_t>& active)
{
    WireReader r{packet};
    if (r.word() != 0x5346415849534331ULL || r.word() != recipe_packet_version)
        throw std::invalid_argument("Unknown shared axis packet.");
    auto values = r.word();
    if (values > r.bytes.size() - r.cursor)
        throw std::invalid_argument("Invalid axis values length.");
    WireReader v{r.bytes.subspan(r.cursor, values)};
    r.cursor += values;
    if (v.word() != 0x5346415856414c31ULL || v.word() != recipe_packet_version)
        throw std::invalid_argument("Unknown axis value DAG packet.");
    auto n = v.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ScalarRecipe p;
        p.root = v.real();
        auto m = v.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(v.word());
        roots.push_back(std::move(p));
    }
    v.finish();
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        active.push_back(r.word());
    r.finish();
}
} // namespace ordinary_extrusion_detail
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{
std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::encode_base() const
{
    WireWriter w;
    w.word(0x534647454f424131ULL);
    w.word(recipe_packet_version);
    w.word(impl_->base->topology.size());
    w.bytes.insert(w.bytes.end(), impl_->base->topology.begin(), impl_->base->topology.end());
    write_point_recipes(w, *impl_->base);
    return std::move(w.bytes);
}

std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::encode_axis_values() const
{
    WireWriter w;
    w.word(0x5346415856414c31ULL);
    w.word(recipe_packet_version);
    w.word(impl_->axis->values->zrecipes.size());
    for (const auto& p : impl_->axis->values->zrecipes)
    {
        w.real(p.root);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    return std::move(w.bytes);
}

std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::encode_axis_selection() const
{
    WireWriter w;
    w.word(impl_->axis->zactive.size());
    for (auto id : impl_->axis->zactive)
        w.word(id);
    return std::move(w.bytes);
}

std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::combine_axis_parts(
    std::span<const char> values, std::span<const char> selection)
{
    WireWriter w;
    w.word(0x5346415849534331ULL);
    w.word(recipe_packet_version);
    w.word(values.size());
    w.bytes.insert(w.bytes.end(), values.begin(), values.end());
    w.bytes.insert(w.bytes.end(), selection.begin(), selection.end());
    return std::move(w.bytes);
}

std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::encode_axis() const
{
    return combine_axis_parts(encode_axis_values(), encode_axis_selection());
}

std::vector<char> ExtrudedConstruction<ArithmeticMode::Ordinary>::encode_view() const
{
    WireWriter w;
    w.word(0x5346505256494531ULL);
    w.word(recipe_packet_version);
    w.word(impl_->begin);
    w.word(impl_->layers);
    return std::move(w.bytes);
}

std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>
ExtrudedConstruction<ArithmeticMode::Ordinary>::shared_view(std::shared_ptr<const ExtrudedTopology> topology,
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> base,
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> axis, std::size_t begin, std::size_t layers)
{
    if (!topology || !base || !axis || !layers || topology->indexer().num_layers != layers ||
        begin >= axis->impl_->axis->zactive.size() || layers >= axis->impl_->axis->zactive.size() - begin ||
        topology->encode_base() != base->impl_->base->topology)
        throw std::invalid_argument("Shared product view has foreign base topology or invalid axial extent.");
    auto impl = std::make_shared<Impl>();
    impl->topology = std::move(topology);
    impl->base = base->impl_->base;
    impl->axis = axis->impl_->axis;
    impl->begin = begin;
    impl->layers = layers;
    auto result = std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>(
        new ExtrudedConstruction<ArithmeticMode::Ordinary>(std::move(impl)));
    // Base visibility and axis representability were certified independently;
    // paired native volume/face outputs still require representable products.
    Scalar minimum = 0, maximum = 0;
    for (std::size_t k = 0; k < layers; ++k)
    {
        const Scalar width = result->impl_->axis->values->z[result->impl_->axis->zactive[begin + k + 1]] -
                             result->impl_->axis->values->z[result->impl_->axis->zactive[begin + k]];
        if (k == 0 || width < minimum)
            minimum = width;
        if (width > maximum)
            maximum = width;
    }
    for (std::size_t c = 0; c < result->impl_->base->areas.size(); ++c)
    {
        (void) positive_measure(Scalar(result->impl_->base->areas[c] * minimum)).as_double();
        (void) positive_measure(Scalar(result->impl_->base->areas[c] * maximum)).as_double();
    }
    for (const auto& edge : result->impl_->base->deltas)
    {
        const Scalar length = dot(edge, edge);
        (void) square_root(Scalar(length * minimum * minimum)).as_double();
        (void) square_root(Scalar(length * maximum * maximum)).as_double();
    }
    return result;
}

std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>
ExtrudedConstruction<ArithmeticMode::Ordinary>::shared_axis_selection(std::shared_ptr<const ExtrudedTopology> topology,
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> base,
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> coordinate_anchor,
    std::vector<std::size_t> active)
{
    if (!topology || !base || !coordinate_anchor || active.size() != std::size_t(topology->indexer().num_layers) + 1 ||
        topology->encode_base() != base->impl_->base->topology)
        throw std::invalid_argument("Shared axis selection belongs to a different base or layer layout.");
    auto impl = std::make_shared<Impl>();
    impl->topology = std::move(topology);
    impl->base = base->impl_->base;
    impl->axis = std::make_shared<ExtrudedAxisData>();
    impl->axis->values = coordinate_anchor->impl_->axis->values;
    impl->axis->zactive = std::move(active);
    impl->layers = impl->axis->zactive.size() - 1;
    for (std::size_t k = 0; k < impl->axis->zactive.size(); ++k)
    {
        const auto id = impl->axis->zactive[k];
        if (id >= impl->axis->values->z.size())
            throw std::invalid_argument("Shared axis selection has an unknown ordinary point.");
        const auto value = measured(impl->axis->values->z[id]).as_double();
        if (k)
        {
            const auto previous = impl->axis->zactive[k - 1];
            const auto lo = impl->axis->display.back();
            const auto center =
                measured(Scalar((impl->axis->values->z[id] + impl->axis->values->z[previous]) / 2)).as_double();
            if (!(value > lo && center > lo && center < value))
                throw std::invalid_argument("Shared axis selection has unrepresentable native widths/centers.");
        }
        impl->axis->display.push_back(value);
    }
    auto selection = std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>(
        new ExtrudedConstruction<ArithmeticMode::Ordinary>(std::move(impl)));
    return shared_view(selection->impl_->topology, selection, selection, 0, selection->impl_->layers);
}

std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>
ExtrudedConstruction<ArithmeticMode::Ordinary>::decode_shared(std::shared_ptr<const ExtrudedTopology> topology,
    std::span<const char> base_packet, std::span<const char> axis_packet, std::span<const char> view_packet,
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> base_anchor,
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> axis_anchor)
{
    if (!topology)
        throw std::invalid_argument("Shared product decoder requires retained topology.");
    WireReader view{view_packet};
    if (view.word() != 0x5346505256494531ULL || view.word() != recipe_packet_version)
        throw std::invalid_argument("Unknown product run view packet.");
    const auto begin = view.word(), layers = view.word();
    view.finish();
    if (base_anchor && base_anchor->encode_base() != std::vector<char>(base_packet.begin(), base_packet.end()))
        throw std::invalid_argument("Shared product base reference does not match ordinary descriptor.");
    std::vector<ScalarRecipe> roots;
    std::vector<std::size_t> active;
    read_axis_parts(axis_packet, roots, active);
    if (active.size() < 2 || active.size() > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Shared axis selection extent is invalid.");
    WireReader base{base_packet};
    if (base.word() != 0x534647454f424131ULL || base.word() != recipe_packet_version)
        throw std::invalid_argument("Unknown ordinary product base packet.");
    auto bytes = base.word();
    if (bytes > base.bytes.size() - base.cursor)
        throw std::invalid_argument("Invalid shared base topology binding.");
    auto expected = topology->encode_base();
    if (bytes != expected.size() || !std::equal(expected.begin(), expected.end(), base.bytes.begin() + base.cursor))
        throw std::invalid_argument("Ordinary shared product base topology mismatch.");
    base.cursor += bytes;
    std::vector<PointRecipe> points;
    std::vector<std::size_t> point_ids;
    read_point_recipes(base, points, point_ids);
    base.finish();
    std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>> source;
    if (!base_anchor || !axis_anchor)
    {
        auto full_topology = topology->with_layers(active.size() - 1);
        source = make(full_topology, std::move(points), std::move(point_ids), roots, active);
    }
    if (!base_anchor)
        base_anchor = source;
    if (axis_anchor)
    {
        const auto expected_values = [&]
        {
            WireWriter w;
            w.word(0x5346415856414c31ULL);
            w.word(recipe_packet_version);
            w.word(roots.size());
            for (const auto& p : roots)
            {
                w.real(p.root);
                w.word(p.parents.size());
                for (auto id : p.parents)
                    w.word(id);
            }
            return w.bytes;
        }();
        if (axis_anchor->encode_axis_values() != expected_values)
            throw std::invalid_argument(
                "Shared product axis reference has different ordinary coordinate constructions.");
        if (axis_anchor->encode_axis_selection() !=
            [&]
            {
                WireWriter w;
                w.word(active.size());
                for (auto id : active)
                    w.word(id);
                return w.bytes;
            }())
        {
            // Form the layout before moving its selection. Function argument
            // evaluation must not observe active after another argument moved it.
            auto selection_topology = topology->with_layers(active.size() - 1);
            axis_anchor =
                shared_axis_selection(std::move(selection_topology), base_anchor, axis_anchor, std::move(active));
        }
    }
    else
        axis_anchor = source;
    return shared_view(std::move(topology), std::move(base_anchor), std::move(axis_anchor), begin, layers);
}

const void* ExtrudedConstruction<ArithmeticMode::Ordinary>::base_storage_identity() const noexcept
{
    return impl_->base.get();
}

const void* ExtrudedConstruction<ArithmeticMode::Ordinary>::axis_storage_identity() const noexcept
{
    return impl_->axis.get();
}

const void* ExtrudedConstruction<ArithmeticMode::Ordinary>::axis_values_storage_identity() const noexcept
{
    return impl_->axis->values.get();
}

std::size_t ExtrudedConstruction<ArithmeticMode::Ordinary>::view_storage_bytes() const noexcept
{
    return sizeof(ExtrudedConstruction<ArithmeticMode::Ordinary>) + sizeof(Impl);
}

std::size_t ExtrudedConstruction<ArithmeticMode::Ordinary>::base_storage_bytes() const
{
    const auto& b = *impl_->base;
    std::size_t bytes = sizeof(b) + b.recipes.capacity() * sizeof(PointRecipe) +
                        b.active.capacity() * sizeof(std::size_t) + b.display.capacity() * sizeof(MeshUtils::Vec3) +
                        b.topology.capacity();
    for (const auto& r : b.recipes)
        bytes += r.parents.capacity() * sizeof(std::size_t);
    for (const auto* values : {&b.points, &b.centers, &b.deltas, &b.edge_centers})
    {
        bytes += values->capacity() * sizeof(P);
    }
    bytes += b.areas.capacity() * sizeof(Scalar);
    bytes += b.edge_supports.capacity() * sizeof(PlaneSupportIdentity<ArithmeticMode::Ordinary>) +
             b.support_normals.capacity() * sizeof(P);
    GeometryAllocationLedger allocations;
    std::size_t admitted = 0;
    for (const auto& support : b.edge_supports)
        allocations.retain(support, admitted, std::numeric_limits<std::size_t>::max());
    return bytes + allocations.payload_bytes();
}

std::size_t ExtrudedConstruction<ArithmeticMode::Ordinary>::axis_values_storage_bytes() const
{
    const auto& a = *impl_->axis->values;
    std::size_t bytes = sizeof(a) + a.zrecipes.capacity() * sizeof(ScalarRecipe) + a.z.capacity() * sizeof(Scalar);
    for (const auto& r : a.zrecipes)
        bytes += r.parents.capacity() * sizeof(std::size_t);
    return bytes;
}

std::size_t ExtrudedConstruction<ArithmeticMode::Ordinary>::axis_storage_bytes() const
{
    return sizeof(ExtrudedAxisData) + impl_->axis->zactive.capacity() * sizeof(std::size_t) +
           impl_->axis->display.capacity() * sizeof(double) + axis_values_storage_bytes();
}
} // namespace SimpleFluid::PLIC
