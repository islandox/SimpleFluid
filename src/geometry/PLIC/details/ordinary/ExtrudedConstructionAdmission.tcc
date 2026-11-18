/** Admitted Ordinary Product construction; included by its existing provider. */
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include "geometry/mesh/details/NativeTopologyAccess.hh"
#include <iterator>
namespace SimpleFluid::PLIC::ordinary_extrusion_detail
{
namespace ga = geometry_detail;
template<class T> using AdmittedVector = ga::NativeConstructorVector<T>;
template<class K, class V> using AdmittedMap = ga::NativeConstructorMap<K, V>;
template<class K> using AdmittedSet = std::set<K, std::less<K>, ga::NativeConstructionAllocator<K>>;
template<class K, class V> auto admitted_map(const ga::NativeConstructionAdmission& admission)
{
    return AdmittedMap<K, V>(std::less<K>{}, admission.allocator<std::pair<const K, V>>());
}
template<class K> auto admitted_set(const ga::NativeConstructionAdmission& admission)
{
    return AdmittedSet<K>(std::less<K>{}, admission.allocator<K>());
}

PlaneSupportIdentity<ArithmeticMode::Ordinary> segment_recipe_support_admitted(const std::vector<PointRecipe>& recipes,
    size_t first, size_t second, const ga::NativeConstructionAdmission& admission)
{
    auto selected = admitted_set<size_t>(admission);
    AdmittedVector<size_t> pending(admission.allocator<size_t>());
    pending.push_back(first);
    pending.push_back(second);
    while (!pending.empty())
    {
        const auto id = pending.back();
        pending.pop_back();
        if (selected.insert(id).second)
            pending.insert(pending.end(), recipes.at(id).parents.begin(), recipes.at(id).parents.end());
    }
    auto depth = admitted_map<size_t, size_t>(admission), canonical = admitted_map<size_t, size_t>(admission);
    auto levels = admitted_map<size_t, AdmittedVector<size_t>>(admission);
    for (const auto id : selected)
    {
        size_t level = 0;
        for (const auto parent : recipes.at(id).parents)
            level = std::max(level, depth.at(parent) + 1);
        depth[id] = level;
        levels.try_emplace(level, admission.allocator<size_t>()).first->second.push_back(id);
    }
    auto storage = ga::native_constructor_storage(admission);
    AdmittedVector<details::OrdinarySupportRecipe> emitted(admission.allocator<details::OrdinarySupportRecipe>());
    for (const auto& [level, ids] : levels)
    {
        struct Entry
        {
            AdmittedVector<size_t> ids;
            details::OrdinarySupportRecipe recipe;
            explicit Entry(ga::NativeConstructionAllocator<size_t> allocator) : ids(std::move(allocator)) {}
        };
        auto unique = admitted_map<AdmittedVector<char>, Entry>(admission);
        for (const auto id : ids)
        {
            const auto& recipe = recipes.at(id);
            details::ordinary_native::BasicWriter<AdmittedVector<char>> signature{
                AdmittedVector<char>(admission.allocator<char>())};
            details::OrdinarySupportRecipe native;
            AdmittedVector<size_t> parents(admission.allocator<size_t>());
            native.original = recipe.parents.empty();
            signature.byte(static_cast<uint8_t>(
                recipe.parents.empty() ? SupportRecipeKind::Original : SupportRecipeKind::Derived));
            if (recipe.parents.empty())
            {
                native.root = details::ordinary_native::wide(recipe.root);
                for (const auto value : recipe.root)
                    signature.real(value);
            }
            else
            {
                for (const auto parent : recipe.parents)
                    parents.push_back(canonical.at(parent));
                std::sort(parents.begin(), parents.end());
                signature.word(parents.size());
                for (const auto parent : parents)
                    signature.word(parent);
            }
            auto [entry, fresh] = unique.try_emplace(std::move(signature.bytes), admission.allocator<size_t>());
            if (fresh)
            {
                if (!native.original)
                {
                    storage.owner->reserve(native.parents, parents.size());
                    native.parents.insert(native.parents.end(), parents.begin(), parents.end());
                }
                entry->second.recipe = std::move(native);
            }
            entry->second.ids.push_back(id);
        }
        for (auto& [signature, entry] : unique)
        {
            for (const auto id : entry.ids)
                canonical[id] = emitted.size();
            emitted.push_back(std::move(entry.recipe));
        }
    }
    std::vector<details::OrdinarySupportRecipe> records;
    storage.owner->reserve(records, emitted.size());
    records.insert(records.end(), std::make_move_iterator(emitted.begin()), std::make_move_iterator(emitted.end()));
    auto a = canonical.at(first), b = canonical.at(second);
    if (a > b)
        std::swap(a, b);
    return details::GeometryIdentityAccess::support_dag_admitted(
        std::move(records), a, b, std::move(storage), admission);
}
std::pair<PlaneSupportIdentity<ArithmeticMode::Ordinary>, P> ancestral_edge_support_admitted(
    const ExtrudedBaseData& base, size_t first, size_t second, const ga::NativeConstructionAdmission& admission)
{
    auto ancestry = admitted_set<size_t>(admission);
    AdmittedVector<size_t> pending(admission.allocator<size_t>());
    pending.push_back(first);
    pending.push_back(second);
    auto candidates = admitted_set<std::pair<size_t, size_t>>(admission);
    candidates.emplace(std::minmax(first, second));
    while (!pending.empty())
    {
        const auto id = pending.back();
        pending.pop_back();
        if (!ancestry.insert(id).second)
            continue;
        const auto& recipe = base.recipes.at(id);
        auto parents = admitted_set<size_t>(admission);
        parents.insert(recipe.parents.begin(), recipe.parents.end());
        if (parents.size() == 2)
            candidates.emplace(*parents.begin(), *parents.rbegin());
        pending.insert(pending.end(), parents.begin(), parents.end());
    }
    const auto membership = [&](size_t a, size_t b)
    {
        AdmittedVector<unsigned char> contained(base.recipes.size(), 0, admission.allocator<unsigned char>());
        for (size_t i = 0; i < base.recipes.size(); ++i)
        {
            const auto& parents = base.recipes[i].parents;
            contained[i] = i == a || i == b ||
                           (!parents.empty() && std::all_of(parents.begin(), parents.end(),
                                                    [&](auto parent) { return contained[parent]; }));
        }
        return contained;
    };
    const auto source = std::minmax(first, second);
    std::pair<size_t, size_t> selected{source.first, source.second};
    for (const auto& [a, b] : candidates)
    {
        const auto contained = membership(a, b);
        if (contained[first] && contained[second] && contained[selected.first] && contained[selected.second])
            selected = {a, b};
    }
    const auto edge = sub(base.points.at(selected.second), base.points.at(selected.first));
    const P normal{edge[1], -edge[0], 0};
    if (zero(normal))
        details::ordinary_native::reject("native ancestral edge is degenerate.");
    if (edge[0] == 0 || edge[1] == 0)
    {
        const auto axis = edge[0] == 0 ? 0u : 1u;
        return {details::GeometryIdentityAccess::support_axis_admitted(
                    axis, base.points.at(selected.first)[axis], admission),
            normal};
    }
    return {segment_recipe_support_admitted(base.recipes, selected.first, selected.second, admission), normal};
}
struct UnitPrismReservation
{
    ga::NativeConstructionAdmission admission;
    size_t bytes;
    UnitPrismReservation(const ga::NativeConstructionAdmission& domain, size_t count)
        : admission(domain),
          bytes(ConvexCellGeometry<ArithmeticMode::Ordinary>::unit_prism_construction_capacity_bound(count))
    {
        admission.charge(bytes);
    }
    ~UnitPrismReservation() { admission.release(bytes); }
};
} // namespace SimpleFluid::PLIC::ordinary_extrusion_detail

namespace SimpleFluid::PLIC::details
{
struct ExtrudedPreparedRecipes
{
    geometry_detail::NativeConstructorStorageOwner base_storage, value_storage, axis_storage;
    std::vector<PointRecipe> recipes;
    std::vector<size_t> active;
    std::vector<ScalarRecipe> zrecipes;
    std::vector<size_t> zactive;
};
std::shared_ptr<const ExtrudedConstructionAccess::Construction> ExtrudedConstructionAccess::make(
    std::shared_ptr<const Meshes::ExtrudedTopology> topology, std::span<const PointRecipe> recipes,
    std::span<const size_t> active, std::span<const ScalarRecipe> zrecipes, std::span<const size_t> zactive,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    using namespace ordinary_extrusion_detail;
    geometry_detail::NativeConstructorStorage::require_profile();
    if (details::active_work)
        throw std::invalid_argument("Admitted Product construction requires a fresh numerical operation.");
    if (!topology || active.size() != topology->indexer().num_nodes_per_layer ||
        zactive.size() != size_t(topology->indexer().num_layers) + 1)
        throw std::invalid_argument("Ordinary extrusion construction and retained topology differ.");
    // Count/index/scalar preflight is allocation-free. Geometric uniqueness,
    // convexity and signed-distance checks remain in the original numeric body.
    for (size_t i = 0; i < recipes.size(); ++i)
    {
        (void) point(recipes[i].root);
        for (const auto parent : recipes[i].parents)
            if (parent >= i)
                throw std::invalid_argument("Ordinary point construction is not an acyclic ancestry DAG.");
    }
    for (const auto id : active)
        if (id >= recipes.size())
            throw std::invalid_argument("Ordinary active point index is invalid.");
    for (size_t i = 0; i < zrecipes.size(); ++i)
    {
        (void) represented(zrecipes[i].root);
        for (const auto parent : zrecipes[i].parents)
            if (parent >= i)
                throw std::invalid_argument("Ordinary axial construction is not acyclic.");
    }
    for (const auto id : zactive)
        if (id >= zrecipes.size())
            throw std::invalid_argument("Ordinary axial point index is invalid.");
    ExtrudedPreparedRecipes input;
    input.base_storage = geometry_detail::native_constructor_storage(admission);
    input.value_storage = geometry_detail::native_constructor_storage(admission);
    input.axis_storage = geometry_detail::native_constructor_storage(admission);
    input.base_storage.owner->reserve(input.recipes, recipes.size());
    for (const auto& source : recipes)
    {
        PointRecipe recipe;
        recipe.root = source.root;
        input.base_storage.owner->copy(recipe.parents, std::span<const size_t>(source.parents));
        input.recipes.push_back(std::move(recipe));
    }
    input.base_storage.owner->copy(input.active, active);
    input.value_storage.owner->reserve(input.zrecipes, zrecipes.size());
    for (const auto& source : zrecipes)
    {
        ScalarRecipe recipe;
        recipe.root = source.root;
        input.value_storage.owner->copy(recipe.parents, std::span<const size_t>(source.parents));
        input.zrecipes.push_back(std::move(recipe));
    }
    input.axis_storage.owner->copy(input.zactive, zactive);
    return make_owned(std::move(topology), std::move(input), admission);
}
std::shared_ptr<const ExtrudedConstructionAccess::Construction> ExtrudedConstructionAccess::make_owned(
    std::shared_ptr<const Meshes::ExtrudedTopology> topology, ExtrudedPreparedRecipes&& input,
    const geometry_detail::NativeConstructionAdmission& admission, std::shared_ptr<const Construction> base_anchor,
    std::shared_ptr<const Construction> values_anchor, std::shared_ptr<const Construction> selection_anchor)
{
    using namespace ordinary_extrusion_detail;
    if (!topology || (!base_anchor && input.active.size() != topology->indexer().num_nodes_per_layer) ||
        (!selection_anchor && input.zactive.size() != size_t(topology->indexer().num_layers) + 1))
        throw std::invalid_argument("Ordinary extrusion construction and retained topology differ.");
    std::shared_ptr<ExtrudedBaseData> base;
    if (base_anchor)
        base = base_anchor->impl_->base;
    else
    {
        auto owner = geometry_detail::NativeAllocationEnvelopeAccess::make<ExtrudedBaseData>(admission);
        owner.owner->storage = std::move(input.base_storage);
        owner.owner->allocation_envelope = std::move(owner.envelope);
        owner.owner->recipes = std::move(input.recipes);
        owner.owner->active = std::move(input.active);
        base = std::move(owner.owner);
    }
    std::shared_ptr<ExtrudedAxisValues> values;
    if (values_anchor)
        values = values_anchor->impl_->axis->values;
    else
    {
        auto owner = geometry_detail::NativeAllocationEnvelopeAccess::make<ExtrudedAxisValues>(admission);
        owner.owner->storage = std::move(input.value_storage);
        owner.owner->allocation_envelope = std::move(owner.envelope);
        owner.owner->zrecipes = std::move(input.zrecipes);
        values = std::move(owner.owner);
    }
    std::shared_ptr<ExtrudedAxisData> axis;
    if (selection_anchor)
    {
        axis = selection_anchor->impl_->axis;
        if (axis->values != values)
            throw std::invalid_argument("Foreign immutable axis selection values.");
    }
    else
    {
        auto owner = geometry_detail::NativeAllocationEnvelopeAccess::make<ExtrudedAxisData>(admission, values);
        owner.owner->storage = std::move(input.axis_storage);
        owner.owner->allocation_envelope = std::move(owner.envelope);
        owner.owner->zactive = std::move(input.zactive);
        axis = std::move(owner.owner);
    }
    auto impl_owner = geometry_detail::NativeAllocationEnvelopeAccess::make<Construction::Impl>(
        admission, std::move(topology), std::move(base), std::move(axis), 0, 0);
    auto impl = std::move(impl_owner.owner);
    impl->allocation_envelope = std::move(impl_owner.envelope);
    const auto cells = impl->topology->indexer().num_cells_per_layer;
    const auto edges = impl->topology->native_topology().side_faces().size();
    if (!base_anchor)
    {
        auto& receipt = *impl->base->storage.owner;
        impl->base->topology =
            Meshes::details::NativeTopologyAccess::encode_base(*impl->topology, impl->base->storage.owner);
        receipt.reserve(impl->base->points, impl->base->recipes.size());
        receipt.reserve(impl->base->centers, cells);
        receipt.reserve(impl->base->areas, cells);
        receipt.reserve(impl->base->deltas, edges);
        receipt.reserve(impl->base->edge_centers, edges);
        receipt.reserve(impl->base->edge_supports, edges);
        receipt.reserve(impl->base->support_normals, edges);
        receipt.reserve(impl->base->display, impl->base->active.size());
    }
    if (!values_anchor)
        impl->axis->values->storage.owner->reserve(impl->axis->values->z, impl->axis->values->zrecipes.size());
    if (!selection_anchor)
        impl->axis->storage.owner->reserve(impl->axis->display, impl->axis->zactive.size());
    if (!base_anchor)
    {
        for (std::size_t i = 0; i < impl->base->recipes.size(); ++i)
        {
            const auto& recipe = impl->base->recipes[i];
            P p{};
            (void) point(recipe.root);
            if (recipe.parents.empty())
                p = point(recipe.root);
            else
            {
                for (auto parent : recipe.parents)
                {
                    if (parent >= i)
                        throw std::invalid_argument("Ordinary point construction is not an acyclic ancestry DAG.");
                    p = add(p, impl->base->points[parent]);
                }
                p = mul(p, (1.L / static_cast<long double>(recipe.parents.size())));
            }
            if (p[2] != 0)
                throw std::invalid_argument("Ordinary extrusion base must lie in XY.");
            impl->base->points.push_back(std::move(p));
        }
        auto unique = admitted_set<size_t>(admission);
        auto unique_points = admitted_set<P>(admission);
        auto display = admitted_set<GeometryPoint>(admission);
        for (auto id : impl->base->active)
        {
            if (id >= impl->base->points.size() || !unique.insert(id).second ||
                !unique_points.insert(impl->base->points[id]).second)
                throw std::invalid_argument("Ordinary extrusion has invalid/duplicate active points.");
            const auto p = rounded_point(impl->base->points[id]);
            if (!display.insert({p.x, p.y, p.z}).second)
                throw std::invalid_argument("Ordinary extrusion display vertices are indistinguishable.");
        }
    }
    if (!values_anchor)
    {
        for (std::size_t i = 0; i < impl->axis->values->zrecipes.size(); ++i)
        {
            const auto& recipe = impl->axis->values->zrecipes[i];
            Scalar z = 0;
            (void) represented(recipe.root);
            if (recipe.parents.empty())
                z = represented(recipe.root);
            else
            {
                for (auto parent : recipe.parents)
                {
                    if (parent >= i)
                        throw std::invalid_argument("Ordinary axial construction is not acyclic.");
                    z += impl->axis->values->z[parent];
                }
                z /= recipe.parents.size();
            }
            impl->axis->values->z.push_back(z);
        }
    }
    if (!selection_anchor)
    {
        Scalar previous = 0;
        double previous_display = 0;
        for (std::size_t k = 0; k < impl->axis->zactive.size(); ++k)
        {
            const auto id = impl->axis->zactive[k];
            if (id >= impl->axis->values->z.size())
                throw std::invalid_argument("Ordinary axial point index is invalid.");
            const auto value = measured(impl->axis->values->z[id]).as_double();
            if (k && (!(impl->axis->values->z[id] > previous) || !(value > previous_display)))
                throw std::invalid_argument("Ordinary axial/display edges must be distinct and increasing.");
            previous = impl->axis->values->z[id];
            previous_display = value;
            if (k)
            {
                const auto lo = measured(impl->axis->values->z[impl->axis->zactive[k - 1]]).as_double();
                const auto center = measured(
                    Scalar((impl->axis->values->z[id] + impl->axis->values->z[impl->axis->zactive[k - 1]]) / 2))
                                        .as_double();
                if (!(center > lo && center < value))
                    throw std::invalid_argument(
                        "Ordinary extrusion has unrepresentable signed axial metric distances.");
            }
        }
    }
    if (!base_anchor)
    {
        for (std::size_t c = 0; c < impl->topology->indexer().num_cells_per_layer; ++c)
        {
            const auto q = impl->topology->base_cell_nodes(c).size();
            UnitPrismReservation prism_admission(admission, q);
            Product product;
            product.base.reserve(q);
            product.lower = 0;
            product.upper = 1;
            for (auto n : impl->topology->base_cell_nodes(c))
                product.base.push_back(impl->base->points[impl->base->active[n]]);
            for (std::size_t i = 0; i < product.base.size(); ++i)
                if (cross(sub(product.base[(i + 1) % product.base.size()], product.base[i]),
                        sub(product.base[(i + 2) % product.base.size()],
                            product.base[(i + 1) % product.base.size()]))[2] <= 0)
                    throw std::invalid_argument(
                        "Ordinary extrusion base must be strictly convex and counterclockwise.");
            (void) product_poly(product);
            AdmittedVector<P> display_polygon(admission.allocator<P>());
            for (const auto& p : product.base)
            {
                auto v = rounded_point(p);
                display_polygon.push_back(point({v.x, v.y, 0}));
            }
            for (std::size_t i = 0; i < display_polygon.size(); ++i)
            {
                const auto a = display_polygon[i], b = display_polygon[(i + 1) % display_polygon.size()],
                           c = display_polygon[(i + 2) % display_polygon.size()];
                if (cross(sub(b, a), sub(c, b))[2] <= 0)
                    throw std::invalid_argument(
                        "Ordinary extrusion display polygon is not a distinct strictly convex native loop.");
                for (const auto& p : display_polygon)
                    if (cross(sub(b, a), sub(p, a))[2] < 0)
                        throw std::invalid_argument("Ordinary extrusion display polygon is nonsimple.");
            }
            const auto m = product_moments(product, P{});
            impl->base->areas.push_back(m.volume);
            impl->base->centers.push_back({m.first[0] / m.volume, m.first[1] / m.volume, 0});
        }
        for (const auto& edge : impl->topology->native_topology().side_faces())
        {
            const auto a = impl->base->points[impl->base->active[edge.nodes[0]]],
                       b = impl->base->points[impl->base->active[edge.nodes[1]]];
            impl->base->deltas.push_back(sub(b, a));
            impl->base->edge_centers.push_back(mul(add(a, b), .5L));
            auto [support, normal] = ancestral_edge_support_admitted(
                *impl->base, impl->base->active[edge.nodes[0]], impl->base->active[edge.nodes[1]], admission);
            impl->base->edge_supports.push_back(std::move(support));
            impl->base->support_normals.push_back(normal);
        }
        for (std::size_t e = 0; e < impl->base->deltas.size(); ++e)
        {
            const auto& edge = impl->topology->native_topology().side_face(e);
            const auto face = rounded_point(impl->base->edge_centers[e]);
            const auto owner = rounded_point(impl->base->centers.at(edge.owner));
            const auto& d = impl->base->deltas[e];
            const auto normal = rounded_point({d[1], -d[0], 0});
            const auto owner_distance = (face - owner).dot(normal);
            if (!std::isfinite(owner_distance) || !(owner_distance > 0))
                throw std::invalid_argument("Ordinary extrusion has unrepresentable signed base metric distances.");
            if (edge.neighbor != SemiStructMeshTopo::invalid_ordinal)
            {
                const auto neighbor = rounded_point(impl->base->centers.at(edge.neighbor));
                const auto neighbor_distance = (face - neighbor).dot(normal);
                if (!std::isfinite(neighbor_distance) || !(neighbor_distance < 0))
                    throw std::invalid_argument("Ordinary extrusion neighbor metric distance is unresolved.");
            }
        }
    }
    impl->layers = impl->axis->zactive.size() - 1;
    if (!base_anchor)
        for (auto id : impl->base->active)
            impl->base->display.push_back(rounded_point(impl->base->points[id]));
    if (!selection_anchor)
        for (auto id : impl->axis->zactive)
            impl->axis->display.push_back(measured(impl->axis->values->z[id]).as_double());
    if (!base_anchor)
        freeze_constructor_buffer_bytes(*impl->base);
    if (!selection_anchor)
        freeze_constructor_buffer_bytes(*impl->axis);
    if (!values_anchor)
        freeze_constructor_buffer_bytes(*impl->axis->values);
    auto result = geometry_detail::NativeAllocationEnvelopeAccess::make<Construction>(
        admission, Construction::AdmittedStorageTag{}, std::move(impl));
    result.owner->allocation_envelope_ = std::move(result.envelope);
    return std::move(result.owner);
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::ordinary_extrusion_detail
{
struct ConstructionCompareBytes
{
    std::span<const char> input;
    size_t cursor = 0;
    bool equal = true;
    void push_back(char value)
    {
        if (cursor >= input.size() || input[cursor] != value)
            equal = false;
        ++cursor;
    }
    const char* end() const noexcept { return nullptr; }
    template<class Iterator> void insert(const char*, Iterator first, Iterator last)
    {
        for (; first != last; ++first)
            push_back(*first);
    }
};
template<class Emit> bool construction_packet_equal(std::span<const char> packet, Emit&& emit)
{
    details::ordinary_native::BasicWriter<ConstructionCompareBytes> writer{ConstructionCompareBytes{packet}};
    emit(writer);
    return writer.bytes.equal && writer.bytes.cursor == packet.size();
}
bool base_packet_equal(const ExtrudedBaseData& base, std::span<const char> packet)
{
    return construction_packet_equal(packet,
        [&](auto& writer)
        {
            writer.tag(GeometryTag::ExtrusionBase);
            writer.byte(recipe_packet_version);
            writer.word(base.topology.size());
            writer.bytes.insert(writer.bytes.end(), base.topology.begin(), base.topology.end());
            writer.word(base.recipes.size());
            for (const auto& recipe : base.recipes)
            {
                for (const auto x : recipe.root)
                    writer.real(x);
                writer.word(recipe.parents.size());
                for (const auto id : recipe.parents)
                    writer.word(id);
            }
            writer.word(base.active.size());
            for (const auto id : base.active)
                writer.word(id);
        });
}
bool values_packet_equal(const ExtrudedAxisValues& values, std::span<const char> packet)
{
    return construction_packet_equal(packet,
        [&](auto& writer)
        {
            writer.tag(GeometryTag::AxisValues);
            writer.byte(recipe_packet_version);
            writer.word(values.zrecipes.size());
            for (const auto& recipe : values.zrecipes)
            {
                writer.real(recipe.root);
                writer.word(recipe.parents.size());
                for (const auto id : recipe.parents)
                    writer.word(id);
            }
        });
}
bool selection_packet_equal(const ExtrudedAxisData& axis, std::span<const char> packet)
{
    return construction_packet_equal(packet,
        [&](auto& writer)
        {
            writer.word(axis.zactive.size());
            for (const auto id : axis.zactive)
                writer.word(id);
        });
}
} // namespace SimpleFluid::PLIC::ordinary_extrusion_detail
namespace SimpleFluid::PLIC::details
{
std::shared_ptr<const ExtrudedConstructionAccess::Construction> ExtrudedConstructionAccess::decode_parts(
    std::shared_ptr<const Meshes::ExtrudedTopology> topology, std::span<const char> base_packet,
    std::span<const char> values_packet, std::span<const char> selection_packet, std::span<const char> view_packet,
    const geometry_detail::NativeConstructionAdmission& admission, std::shared_ptr<const Construction> base_anchor,
    std::shared_ptr<const Construction> values_anchor, std::shared_ptr<const Construction> selection_anchor)
{
    using namespace ordinary_extrusion_detail;
    geometry_detail::NativeConstructorStorage::require_profile();
    if (!topology)
        throw std::invalid_argument("Shared product decoder requires retained topology.");
    WireReader view{view_packet};
    if (view.tag() != GeometryTag::ProductView || view.byte() != recipe_packet_version)
        throw std::invalid_argument("Unknown product run view packet.");
    const auto begin = view.word(), layers = view.word();
    view.finish();
    if (!layers || layers != topology->indexer().num_layers)
        throw std::invalid_argument("Product run view and topology have different layer extents.");
    WireReader base{base_packet};
    if (base.tag() != GeometryTag::ExtrusionBase || base.byte() != recipe_packet_version)
        throw std::invalid_argument("Unknown ordinary product base packet.");
    const auto bytes = base.word();
    if (bytes > base.bytes.size() - base.cursor ||
        !Meshes::details::NativeTopologyAccess::matches_base(*topology, base.bytes.subspan(base.cursor, bytes)))
        throw std::invalid_argument("Ordinary shared product base topology mismatch.");
    base.cursor += bytes;
    if (base_anchor && !base_packet_equal(*base_anchor->impl_->base, base_packet))
        throw std::invalid_argument("Shared product base reference does not match ordinary descriptor.");
    if (values_anchor && !values_packet_equal(*values_anchor->impl_->axis->values, values_packet))
        throw std::invalid_argument("Shared product values reference does not match ordinary descriptor.");
    if (selection_anchor)
    {
        if (!values_packet_equal(*selection_anchor->impl_->axis->values, values_packet) ||
            !selection_packet_equal(*selection_anchor->impl_->axis, selection_packet) ||
            (values_anchor && values_anchor->impl_->axis->values != selection_anchor->impl_->axis->values))
            throw std::invalid_argument(
                "Shared product selection reference does not match its exact values/selection.");
        if (!values_anchor)
            values_anchor = selection_anchor;
    }
    if (!selection_anchor && values_anchor && selection_packet_equal(*values_anchor->impl_->axis, selection_packet))
        selection_anchor = values_anchor;
    // Every matched immutable part retains its actual producer control. No
    // re-encoding vectors, copied descriptors or discarded factor owners.
    if (base_anchor && values_anchor && selection_anchor)
        return shared_view(
            std::move(topology), std::move(base_anchor), std::move(selection_anchor), begin, layers, admission);
    const auto scan_recipes = [](WireReader& reader, bool points)
    {
        const auto count = reader.count();
        for (size_t i = 0; i < count; ++i)
        {
            for (unsigned axis = 0; axis < (points ? 3u : 1u); ++axis)
                (void) reader.real();
            const auto parents = reader.count();
            for (size_t j = 0; j < parents; ++j)
                if (reader.word() >= i)
                    throw std::invalid_argument("Ordinary construction is not an acyclic ancestry DAG.");
        }
        return count;
    };
    size_t scanned_points = base_anchor ? base_anchor->impl_->base->recipes.size() : 0;
    if (!base_anchor)
    {
        auto scan = base;
        scanned_points = scan_recipes(scan, true);
        const auto count = scan.count();
        if (count != topology->indexer().num_nodes_per_layer)
            throw std::invalid_argument("Ordinary active point count differs from topology.");
        for (size_t i = 0; i < count; ++i)
            if (scan.word() >= scanned_points)
                throw std::invalid_argument("Ordinary active point index is invalid.");
        scan.finish();
    }
    size_t scanned_values = values_anchor ? values_anchor->impl_->axis->values->zrecipes.size() : 0;
    if (!values_anchor)
    {
        WireReader scan{values_packet};
        if (scan.tag() != GeometryTag::AxisValues || scan.byte() != recipe_packet_version)
            throw std::invalid_argument("Unknown axis value DAG packet.");
        scanned_values = scan_recipes(scan, false);
        scan.finish();
    }
    if (!selection_anchor)
    {
        WireReader scan{selection_packet};
        const auto count = scan.count();
        if (count < 2 || begin >= count || layers >= count - begin)
            throw std::invalid_argument("Shared product view has invalid axial extent.");
        for (size_t i = 0; i < count; ++i)
            if (scan.word() >= scanned_values)
                throw std::invalid_argument("Ordinary axial point index is invalid.");
        scan.finish();
    }
    if (!base_anchor && details::active_work)
        throw std::invalid_argument("Admitted Product base decode requires a fresh numerical operation.");
    ExtrudedPreparedRecipes input;
    if (!base_anchor)
    {
        input.base_storage = geometry_detail::native_constructor_storage(admission);
        const auto points = base.count();
        input.base_storage.owner->reserve(input.recipes, points);
        for (size_t i = 0; i < points; ++i)
        {
            PointRecipe recipe;
            for (auto& x : recipe.root)
                x = base.real();
            const auto count = base.count();
            input.base_storage.owner->reserve(recipe.parents, count);
            for (size_t j = 0; j < count; ++j)
            {
                const auto id = base.word();
                if (id >= i)
                    throw std::invalid_argument("Ordinary point construction is not an acyclic ancestry DAG.");
                recipe.parents.push_back(id);
            }
            input.recipes.push_back(std::move(recipe));
        }
        const auto active = base.count();
        input.base_storage.owner->reserve(input.active, active);
        for (size_t i = 0; i < active; ++i)
            input.active.push_back(base.word());
        base.finish();
    }
    if (!values_anchor)
    {
        input.value_storage = geometry_detail::native_constructor_storage(admission);
        WireReader values{values_packet};
        if (values.tag() != GeometryTag::AxisValues || values.byte() != recipe_packet_version)
            throw std::invalid_argument("Unknown axis value DAG packet.");
        const auto count = values.count();
        input.value_storage.owner->reserve(input.zrecipes, count);
        for (size_t i = 0; i < count; ++i)
        {
            ScalarRecipe recipe;
            recipe.root = values.real();
            const auto parents = values.count();
            input.value_storage.owner->reserve(recipe.parents, parents);
            for (size_t j = 0; j < parents; ++j)
            {
                const auto id = values.word();
                if (id >= i)
                    throw std::invalid_argument("Ordinary axial construction is not acyclic.");
                recipe.parents.push_back(id);
            }
            input.zrecipes.push_back(std::move(recipe));
        }
        values.finish();
    }
    size_t edges = 0;
    if (selection_anchor)
        edges = selection_anchor->impl_->axis->zactive.size();
    else
    {
        input.axis_storage = geometry_detail::native_constructor_storage(admission);
        WireReader selection{selection_packet};
        edges = selection.count();
        if (edges < 2 || edges > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Shared axis selection extent is invalid.");
        input.axis_storage.owner->reserve(input.zactive, edges);
        for (size_t i = 0; i < edges; ++i)
            input.zactive.push_back(selection.word());
        selection.finish();
    }
    if (begin >= edges || layers >= edges - begin)
        throw std::invalid_argument("Shared product view has invalid axial extent.");
    auto full_topology = topology;
    if (full_topology->indexer().num_layers != edges - 1)
        full_topology = Meshes::details::NativeTopologyAccess::with_layers(
            *topology, static_cast<unsigned>(edges - 1), {}, admission);
    auto source = make_owned(std::move(full_topology), std::move(input), admission, std::move(base_anchor),
        std::move(values_anchor), std::move(selection_anchor));
    return shared_view(std::move(topology), source, source, begin, layers, admission);
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
std::shared_ptr<const ExtrudedConstructionAccess::Construction> ExtrudedConstructionAccess::decode(
    std::shared_ptr<const Meshes::ExtrudedTopology> topology, std::span<const char> packet,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    using namespace ordinary_extrusion_detail;
    geometry_detail::NativeConstructorStorage::require_profile();
    if (details::active_work)
        throw std::invalid_argument("Admitted Product decode requires a fresh numerical operation.");
    if (!topology)
        throw std::invalid_argument("Ordinary extrusion decode needs retained topology.");
    WireReader reader{packet};
    if (reader.tag() != GeometryTag::ExtrudedConstruction || reader.byte() != recipe_packet_version)
        throw std::invalid_argument("Unknown ordinary extrusion packet.");
    const auto bytes = reader.word();
    if (bytes > packet.size() - reader.cursor ||
        !construction_packet_equal(packet.subspan(reader.cursor, bytes), [&](auto& writer)
            { Meshes::details::NativeTopologyAccess::emit_construction_topology(*topology, writer); }))
        throw std::invalid_argument("Ordinary extrusion packet belongs to different topology/order.");
    reader.cursor += bytes;
    const auto scan_section = [](WireReader& scan, unsigned dimensions)
    {
        const auto count = scan.count();
        for (size_t i = 0; i < count; ++i)
        {
            for (unsigned axis = 0; axis < dimensions; ++axis)
                (void) scan.real();
            const auto parents = scan.count();
            for (size_t j = 0; j < parents; ++j)
                if (scan.word() >= i)
                    throw std::invalid_argument("Ordinary construction is not acyclic.");
        }
        const auto active = scan.count();
        for (size_t i = 0; i < active; ++i)
            if (scan.word() >= count)
                throw std::invalid_argument("Ordinary active index is invalid.");
        return active;
    };
    auto scan = reader;
    if (scan_section(scan, 3) != topology->indexer().num_nodes_per_layer ||
        scan_section(scan, 1) != size_t(topology->indexer().num_layers) + 1)
        throw std::invalid_argument("Ordinary extrusion construction and retained topology differ.");
    scan.finish();
    ExtrudedPreparedRecipes input;
    input.base_storage = geometry_detail::native_constructor_storage(admission);
    input.value_storage = geometry_detail::native_constructor_storage(admission);
    input.axis_storage = geometry_detail::native_constructor_storage(admission);
    const auto points = reader.count();
    input.base_storage.owner->reserve(input.recipes, points);
    for (size_t i = 0; i < points; ++i)
    {
        PointRecipe recipe;
        for (auto& x : recipe.root)
            x = reader.real();
        const auto parents = reader.count();
        input.base_storage.owner->reserve(recipe.parents, parents);
        for (size_t j = 0; j < parents; ++j)
            recipe.parents.push_back(reader.word());
        input.recipes.push_back(std::move(recipe));
    }
    const auto active = reader.count();
    input.base_storage.owner->reserve(input.active, active);
    for (size_t i = 0; i < active; ++i)
        input.active.push_back(reader.word());
    const auto values = reader.count();
    input.value_storage.owner->reserve(input.zrecipes, values);
    for (size_t i = 0; i < values; ++i)
    {
        ScalarRecipe recipe;
        recipe.root = reader.real();
        const auto parents = reader.count();
        input.value_storage.owner->reserve(recipe.parents, parents);
        for (size_t j = 0; j < parents; ++j)
            recipe.parents.push_back(reader.word());
        input.zrecipes.push_back(std::move(recipe));
    }
    const auto edges = reader.count();
    input.axis_storage.owner->reserve(input.zactive, edges);
    for (size_t i = 0; i < edges; ++i)
        input.zactive.push_back(reader.word());
    reader.finish();
    return make_owned(std::move(topology), std::move(input), admission);
}
} // namespace SimpleFluid::PLIC::details
