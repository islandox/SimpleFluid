/** @file AdaptiveGeometryBindings.tcc
 * @brief Compiled source-only binding pages and ordered selectors.
 */
namespace geometry_snapshot_detail
{
namespace binding_detail
{
constexpr std::size_t binding_page_size = 64;

bool same_native_descriptor(const std::shared_ptr<const NativeFragmentDescriptor>& first,
    const std::shared_ptr<const NativeFragmentDescriptor>& second)
{
    if (first == second)
        return true;
    if (!first || !second || first->kind != second->kind || first->view != second->view ||
        first->parts.size() != second->parts.size())
        return false;
    return std::equal(first->parts.begin(), first->parts.end(), second->parts.begin(),
        [](const auto& a, const auto& b) { return a && b && *a == *b; });
}
bool fact_equal(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
std::uint8_t policy_bits(GeometryEvaluationPolicy policy)
{
    return std::uint8_t(policy.original_integrals | (policy.ordinary_integrals << 1) | (policy.bounded_integrals << 2));
}
GeometryEvaluationPolicy decode_policy(std::uint8_t bits)
{
    return {bool(bits & 1), bool(bits & 2), bool(bits & 4)};
}
template<class T> struct BindingColumn
{
    struct Override
    {
        std::uint8_t row;
        T value;
    };
    T first{}, step{};
    std::uint8_t repeat = 1, period = binding_page_size;
    std::variant<std::monostate, std::vector<Override>, std::vector<T>> exceptions;

    template<class Value> void build(std::size_t count, Value value, adaptive_detail::Budget& budget)
    {
        first = value(0);
        if constexpr (std::is_integral_v<T>)
        {
            std::size_t equal = 1;
            while (equal < count && value(equal) == first)
                ++equal;
            if (equal < count)
            {
                repeat = std::uint8_t(equal);
                step = T(value(equal) - first);
                for (std::size_t row = 2 * equal; row < count; row += equal)
                    if (value(row) == first)
                    {
                        period = std::uint8_t(row / equal);
                        break;
                    }
            }
        }
        const auto differs = [&](std::size_t row)
        {
            if constexpr (std::is_integral_v<T>)
                return value(row) != pattern(row);
            else
                return !fact_equal(value(row), first);
        };
        std::size_t mismatches = 0;
        for (std::size_t row = 1; row < count; ++row)
            mismatches += differs(row);
        if (!mismatches)
            return;
        if (mismatches * sizeof(Override) < count * sizeof(T))
        {
            auto& overrides = exceptions.template emplace<std::vector<Override>>();
            reserve_checked(overrides, mismatches, budget);
            for (std::size_t row = 1; row < count; ++row)
                if (differs(row))
                    overrides.push_back({std::uint8_t(row), value(row)});
            return;
        }
        auto& dense = exceptions.template emplace<std::vector<T>>();
        reserve_checked(dense, count, budget);
        for (std::size_t row = 0; row < count; ++row)
            dense.push_back(value(row));
    }
    T at(std::size_t row) const noexcept
    {
        if (const auto* dense = std::get_if<std::vector<T>>(&exceptions))
            return (*dense)[row];
        if (const auto* overrides = std::get_if<std::vector<Override>>(&exceptions))
        {
            const auto found = std::lower_bound(overrides->begin(), overrides->end(), row,
                [](const Override& item, std::size_t wanted) { return item.row < wanted; });
            if (found != overrides->end() && found->row == row)
                return found->value;
        }
        return pattern(row);
    }
    std::size_t bytes() const noexcept
    {
        if (const auto* dense = std::get_if<std::vector<T>>(&exceptions))
            return dense->capacity() * sizeof(T);
        if (const auto* overrides = std::get_if<std::vector<Override>>(&exceptions))
            return overrides->capacity() * sizeof(Override);
        return 0;
    }

private:
    T pattern(std::size_t row) const noexcept
    {
        if constexpr (std::is_integral_v<T>)
            return T(first + step * T((row / repeat) % period));
        else
            return first;
    }
};
struct BindingSource
{
    std::uint64_t region, material, revision;
    AdaptiveProviderKind provider;
    bool product_factor;
    AdaptiveBytes domain;
    std::shared_ptr<const NativeFragmentDescriptor> native;
    std::shared_ptr<const AdaptiveBytes> join, product_root;
    std::shared_ptr<const AdaptiveFactorRecipe> factor;

    [[nodiscard]] bool matches(const AdaptiveLeaf& leaf) const
    {
        return region == leaf.key.region && material == leaf.material && revision == leaf.revision &&
               provider == leaf.provider && domain == leaf.domain &&
               same_native_descriptor(native, leaf.native_recipe) && same_capsule(join, leaf.join_recipe);
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return sizeof(*this) + domain.capacity(); }
    template<class Visitor> void visit_children(const Visitor& visit) const
    {
        if (native)
        {
            visit(native,
                sizeof(*native) + native->view.capacity() + native->parts.capacity() * sizeof(native->parts[0]));
            for (const auto& part : native->parts)
                if (part)
                    visit(part, sizeof(*part) + part->capacity());
        }
        const auto bytes = [&](const auto& value)
        {
            if (value)
                visit(value, sizeof(*value) + value->capacity());
        };
        bytes(join);
        bytes(product_root);
        if (factor)
        {
            visit(factor, sizeof(*factor));
            bytes(factor->root);
            bytes(factor->base);
            bytes(factor->axis);
        }
    }
};
bool factorized(const AdaptiveLeaf& leaf)
{
    return std::holds_alternative<ProductCellAncestry>(leaf.key.ancestry) && leaf.factor_recipe;
}
struct BindingSourceLess
{
    bool operator()(const AdaptiveLeaf* a, const AdaptiveLeaf* b) const
    {
        const auto parts = [](const AdaptiveLeaf& leaf)
        {
            const auto address = [](const auto* owner) { return reinterpret_cast<std::uintptr_t>(owner); };
            const bool product = factorized(leaf);
            return std::tuple(leaf.key.region, leaf.material, leaf.revision, leaf.provider, product,
                address(leaf.native_recipe.get()), address(leaf.join_recipe.get()),
                address(product ? leaf.factor_recipe->root.get() : nullptr),
                address(product ? nullptr : leaf.factor_recipe.get()));
        };
        if (const auto comparison = parts(*a) <=> parts(*b); comparison != 0)
            return comparison < 0;
        return a->domain < b->domain;
    }
};
using SourceTable = std::map<const AdaptiveLeaf*, std::shared_ptr<const BindingSource>, BindingSourceLess>;
struct CartesianBindingKeys
{
    BindingColumn<unsigned> level;
    std::array<BindingColumn<std::uint64_t>, 3> indices;
    CartesianCellAncestry at(std::size_t row) const noexcept
    {
        return {level.at(row), {indices[0].at(row), indices[1].at(row), indices[2].at(row)}};
    }
    std::size_t bytes() const noexcept
    {
        return level.bytes() + indices[0].bytes() + indices[1].bytes() + indices[2].bytes();
    }
};
struct StaticBindingKeys
{
    BindingColumn<std::uint64_t> root, native_cell;
    StaticCellAncestry at(std::size_t row) const noexcept { return {root.at(row), native_cell.at(row)}; }
    std::size_t bytes() const noexcept { return root.bytes() + native_cell.bytes(); }
};
struct ProductBindingKeys
{
    template<class Key> struct Factor
    {
        Key key;
        std::shared_ptr<const AdaptiveBytes> capsule;
    };
    std::vector<Factor<BaseCellKey>> bases;
    std::vector<Factor<AxialIntervalKey>> axes;
    BindingColumn<std::uint8_t> base, axis;
    std::size_t bytes() const noexcept
    {
        auto total =
            bases.capacity() * sizeof(bases[0]) + axes.capacity() * sizeof(axes[0]) + base.bytes() + axis.bytes();
        for (const auto& value : bases)
            total += value.key.path.capacity() * sizeof(value.key.path[0]);
        for (const auto& value : axes)
            total += value.key.path.capacity() * sizeof(value.key.path[0]);
        return total;
    }
};
struct ExplicitBindingKeys
{
    std::vector<ConvexCellAncestry> keys;
    std::size_t bytes() const noexcept
    {
        auto total = keys.capacity() * sizeof(ConvexCellAncestry);
        for (const auto& key : keys)
            total += key.path.capacity() * sizeof(key.path[0]);
        return total;
    }
};
struct BindingCertificate
{
    std::uint8_t row;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    std::vector<GeometrySnapshotSurface> surfaces;
};
struct BindingPage
{
    std::size_t row_count = 0;
    std::vector<std::shared_ptr<const BindingSource>> sources;
    BindingColumn<std::uint8_t> source, policy;
    BindingColumn<std::uint64_t> native_cell;
    std::variant<CartesianBindingKeys, ExplicitBindingKeys, ProductBindingKeys, StaticBindingKeys> keys;
    std::optional<std::array<BindingColumn<double>, 4>> metrics;
    std::vector<BindingCertificate> certificates;

    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::strong_ordering compare(std::size_t, const AdaptiveCellKey&) const;
    [[nodiscard]] bool matches(std::size_t, const AdaptiveLeaf&) const;
    [[nodiscard]] AdaptiveCellMetrics metric(std::size_t) const;
    [[nodiscard]] AdaptiveCellKey copy_key(std::size_t, adaptive_detail::Budget&) const;
    [[nodiscard]] AdaptiveLeaf copy_leaf(std::size_t, AdaptiveCellKey, adaptive_detail::Budget&) const;
    template<class Visitor> void visit_children(const Visitor& visit) const
    {
        for (const auto& value : sources)
        {
            visit(value, value->bytes());
            value->visit_children(visit);
        }
        if (const auto* product = std::get_if<ProductBindingKeys>(&keys))
        {
            const auto visit_factors = [&](const auto& factors)
            {
                for (const auto& factor : factors)
                    if (factor.capsule)
                        visit(factor.capsule, sizeof(*factor.capsule) + factor.capsule->capacity());
            };
            visit_factors(product->bases);
            visit_factors(product->axes);
        }
    }

private:
    [[nodiscard]] const BindingCertificate* certificate(std::size_t) const;
};
const BindingCertificate* BindingPage::certificate(std::size_t row) const
{
    const auto found = std::lower_bound(certificates.begin(), certificates.end(), row,
        [](const BindingCertificate& value, std::size_t wanted) { return value.row < wanted; });
    return found != certificates.end() && found->row == row ? &*found : nullptr;
}
std::size_t BindingPage::bytes() const noexcept
{
    auto total = sizeof(*this) + sources.capacity() * sizeof(sources[0]) + source.bytes() + policy.bytes() +
                 native_cell.bytes() + std::visit([](const auto& value) { return value.bytes(); }, keys) +
                 certificates.capacity() * sizeof(BindingCertificate);
    if (metrics)
        for (const auto& column : *metrics)
            total += column.bytes();
    for (const auto& certificate : certificates)
    {
        total += certificate.surfaces.capacity() * sizeof(GeometrySnapshotSurface);
        for (const auto& value : certificate.surfaces)
            total += value.surface.key.capacity() + value.surface.boundary.capacity();
    }
    return total;
}
std::strong_ordering BindingPage::compare(std::size_t row, const AdaptiveCellKey& key) const
{
    if (const auto region = sources[source.at(row)]->region <=> key.region; region != 0)
        return region;
    if (const auto kind = keys.index() <=> key.ancestry.index(); kind != 0)
        return kind;
    if (const auto* cartesian = std::get_if<CartesianBindingKeys>(&keys))
        return cartesian->at(row) <=> std::get<CartesianCellAncestry>(key.ancestry);
    if (const auto* product = std::get_if<ProductBindingKeys>(&keys))
    {
        const auto& ancestry = std::get<ProductCellAncestry>(key.ancestry);
        if (const auto base = product->bases[product->base.at(row)].key <=> ancestry.base; base != 0)
            return base;
        return product->axes[product->axis.at(row)].key <=> ancestry.axial;
    }
    if (const auto* stationary = std::get_if<StaticBindingKeys>(&keys))
        return stationary->at(row) <=> std::get<StaticCellAncestry>(key.ancestry);
    return std::get<ExplicitBindingKeys>(keys).keys[row] <=> std::get<ConvexCellAncestry>(key.ancestry);
}
bool BindingPage::matches(std::size_t row, const AdaptiveLeaf& leaf) const
{
    const auto& descriptor = *sources[source.at(row)];
    if (!descriptor.matches(leaf) || native_cell.at(row) != leaf.native_cell)
        return false;
    if (const auto* product = std::get_if<ProductBindingKeys>(&keys); product && !descriptor.factor)
    {
        const auto& base = product->bases[product->base.at(row)].capsule;
        const auto& axis = product->axes[product->axis.at(row)].capsule;
        if (descriptor.product_factor != bool(leaf.factor_recipe))
            return false;
        if (leaf.factor_recipe &&
            (!same_capsule(descriptor.product_root, leaf.factor_recipe->root) ||
                !same_capsule(base, leaf.factor_recipe->base) || !same_capsule(axis, leaf.factor_recipe->axis)))
            return false;
    }
    else if (!same_factor_recipe(descriptor.factor, leaf.factor_recipe))
        return false;
    const auto* stored = certificate(row);
    if ((stored ? stored->geometry : nullptr) != leaf.geometry)
        return false;
    if (!requires_surface_binding(leaf))
        return true;
    if (!stored || stored->surfaces.size() != leaf.surfaces.size())
        return false;
    return std::equal(stored->surfaces.begin(), stored->surfaces.end(), leaf.surfaces.begin(),
        [](const auto& retained, const auto& incoming) { return same_surface(retained.surface, incoming); });
}
AdaptiveCellMetrics BindingPage::metric(std::size_t row) const
{
    const auto& columns = metrics.value();
    return {columns[0].at(row), {columns[1].at(row), columns[2].at(row), columns[3].at(row)}};
}
AdaptiveCellKey BindingPage::copy_key(std::size_t row, adaptive_detail::Budget& budget) const
{
    const auto region = sources[source.at(row)]->region;
    if (const auto* cartesian = std::get_if<CartesianBindingKeys>(&keys))
        return {region, cartesian->at(row)};
    if (const auto* product = std::get_if<ProductBindingKeys>(&keys))
    {
        const auto& base = product->bases[product->base.at(row)].key;
        const auto& axis = product->axes[product->axis.at(row)].key;
        charge_items(budget, base.path.capacity() + axis.path.capacity(), sizeof(base.path[0]));
        return {region, ProductCellAncestry{base, axis}};
    }
    if (const auto* stationary = std::get_if<StaticBindingKeys>(&keys))
        return {region, stationary->at(row)};
    const auto& key = std::get<ExplicitBindingKeys>(keys).keys[row];
    charge_items(budget, key.path.capacity(), sizeof(key.path[0]));
    return {region, key};
}
AdaptiveLeaf BindingPage::copy_leaf(std::size_t row, AdaptiveCellKey key, adaptive_detail::Budget& budget) const
{
    AdaptiveLeaf leaf;
    const auto& descriptor = *sources[source.at(row)];
    budget.add(descriptor.domain.capacity());
    leaf.key = std::move(key);
    leaf.domain = descriptor.domain;
    leaf.material = descriptor.material;
    leaf.revision = descriptor.revision;
    leaf.provider = descriptor.provider;
    leaf.native_recipe = descriptor.native;
    leaf.native_cell = native_cell.at(row);
    leaf.join_recipe = descriptor.join;
    leaf.factor_recipe = descriptor.factor;
    if (descriptor.product_factor)
    {
        const auto& product = std::get<ProductBindingKeys>(keys);
        const auto& base = product.bases[product.base.at(row)].capsule;
        const auto& axis = product.axes[product.axis.at(row)].capsule;
        budget.add(sizeof(AdaptiveFactorRecipe));
        leaf.factor_recipe =
            std::make_shared<const AdaptiveFactorRecipe>(AdaptiveFactorRecipe{descriptor.product_root, base, axis});
    }
    if (const auto* stored = certificate(row))
    {
        leaf.geometry = stored->geometry;
        reserve_checked(leaf.surfaces, stored->surfaces.size(), budget);
        for (const auto& value : stored->surfaces)
        {
            budget.add(value.surface.key.capacity() + value.surface.boundary.capacity());
            leaf.surfaces.push_back(value.surface);
        }
    }
    return leaf;
}
template<class Factor, class Get>
void product_factors(std::vector<Factor>& dictionary, BindingColumn<std::uint8_t>& column, std::size_t count, Get get,
    adaptive_detail::Budget& budget)
{
    std::array<std::uint8_t, binding_page_size> selectors;
    std::array<std::uint8_t, binding_page_size> distinct;
    std::size_t unique = 0;
    for (std::size_t row = 0; row < count; ++row)
    {
        const auto [key, capsule] = get(row);
        std::size_t index = 0;
        for (; index < unique; ++index)
        {
            const auto [first_key, first_capsule] = get(distinct[index]);
            if (key == first_key && capsule == first_capsule)
                break;
        }
        if (index == unique)
            distinct[unique++] = std::uint8_t(row);
        selectors[row] = std::uint8_t(index);
    }
    reserve_checked(dictionary, unique, budget);
    for (std::size_t i = 0; i < unique; ++i)
    {
        const auto [key, capsule] = get(distinct[i]);
        charge_items(budget, key.path.capacity(), sizeof(key.path[0]));
        dictionary.push_back({key, capsule});
    }
    column.build(count, [&](auto i) { return selectors[i]; }, budget);
}
std::shared_ptr<const BindingPage> make_binding_page(std::span<const AdaptiveLeaf* const> leaves,
    std::span<const std::size_t> rows, const BuildFacts& facts, SourceTable& known, adaptive_detail::Budget& budget)
{
    budget.add(sizeof(BindingPage));
    auto page = std::make_shared<BindingPage>();
    const auto get = [&](auto row) -> const AdaptiveLeaf& { return *leaves[rows[row]]; };
    const auto count = rows.size();
    page->row_count = count;
    std::array<std::shared_ptr<const BindingSource>, binding_page_size> sources;
    std::array<std::uint8_t, binding_page_size> selectors;
    std::size_t source_count = 0, cert_count = 0;
    for (std::size_t row = 0; row < count; ++row)
    {
        const auto& leaf = get(row);
        auto found = known.find(&leaf);
        if (found == known.end())
        {
            budget.add(
                sizeof(SourceTable::value_type) + 4 * sizeof(void*) + sizeof(BindingSource) + leaf.domain.capacity());
            auto source = std::make_shared<const BindingSource>(BindingSource{leaf.key.region, leaf.material,
                leaf.revision, leaf.provider, factorized(leaf), leaf.domain, leaf.native_recipe, leaf.join_recipe,
                factorized(leaf) ? leaf.factor_recipe->root : nullptr,
                factorized(leaf) ? nullptr : leaf.factor_recipe});
            found = known.emplace(&leaf, std::move(source)).first;
        }
        const auto owner = std::find(sources.begin(), sources.begin() + source_count, found->second);
        if (owner == sources.begin() + source_count)
            sources[source_count++] = found->second;
        selectors[row] = std::uint8_t(owner - sources.begin());
        cert_count += bool(leaf.geometry);
    }
    reserve_checked(page->sources, source_count, budget);
    page->sources.assign(sources.begin(), sources.begin() + source_count);
    page->source.build(count, [&](auto row) { return selectors[row]; }, budget);
    page->policy.build(count, [&](auto row) { return policy_bits(facts.policy[rows[row]]); }, budget);
    page->native_cell.build(count, [&](auto row) { return get(row).native_cell; }, budget);
    if (std::holds_alternative<CartesianCellAncestry>(get(0).key.ancestry))
    {
        auto& keys = page->keys.emplace<CartesianBindingKeys>();
        const auto key = [&](auto row) -> const auto&
        { return std::get<CartesianCellAncestry>(get(row).key.ancestry); };
        keys.level.build(count, [&](auto row) { return key(row).level; }, budget);
        for (unsigned axis = 0; axis < 3; ++axis)
            keys.indices[axis].build(count, [&](auto row) { return key(row).index[axis]; }, budget);
    }
    else if (std::holds_alternative<ProductCellAncestry>(get(0).key.ancestry))
    {
        auto& keys = page->keys.emplace<ProductBindingKeys>();
        const auto key = [&](auto row) -> const auto& { return std::get<ProductCellAncestry>(get(row).key.ancestry); };
        product_factors(
            keys.bases, keys.base, count,
            [&](auto row)
            {
                const auto& factor = get(row).factor_recipe;
                return std::pair<const BaseCellKey&, std::shared_ptr<const AdaptiveBytes>>(
                    key(row).base, factor ? factor->base : nullptr);
            },
            budget);
        product_factors(
            keys.axes, keys.axis, count,
            [&](auto row)
            {
                const auto& factor = get(row).factor_recipe;
                return std::pair<const AxialIntervalKey&, std::shared_ptr<const AdaptiveBytes>>(
                    key(row).axial, factor ? factor->axis : nullptr);
            },
            budget);
    }
    else if (std::holds_alternative<StaticCellAncestry>(get(0).key.ancestry))
    {
        auto& keys = page->keys.emplace<StaticBindingKeys>();
        const auto key = [&](auto row) -> const auto& { return std::get<StaticCellAncestry>(get(row).key.ancestry); };
        keys.root.build(count, [&](auto row) { return key(row).provider_root; }, budget);
        keys.native_cell.build(count, [&](auto row) { return key(row).native_cell; }, budget);
    }
    else
    {
        auto& keys = page->keys.emplace<ExplicitBindingKeys>().keys;
        reserve_checked(keys, count, budget);
        for (std::size_t row = 0; row < count; ++row)
        {
            const auto& key = std::get<ConvexCellAncestry>(get(row).key.ancestry);
            charge_items(budget, key.path.capacity(), sizeof(key.path[0]));
            keys.push_back(key);
        }
    }
    if (!facts.metrics.empty())
    {
        auto& metrics = page->metrics.emplace();
        metrics[0].build(count, [&](auto row) { return facts.metrics[rows[row]].volume; }, budget);
        for (unsigned axis = 0; axis < 3; ++axis)
            metrics[axis + 1].build(count, [&](auto row) { return facts.metrics[rows[row]].centroid[axis]; }, budget);
    }
    reserve_checked(page->certificates, cert_count, budget);
    for (std::size_t row = 0; row < count; ++row)
    {
        const auto& leaf = get(row);
        if (!leaf.geometry)
            continue;
        auto& certificate = page->certificates.emplace_back(BindingCertificate{std::uint8_t(row), leaf.geometry, {}});
        if (!requires_surface_binding(leaf))
            continue;
        reserve_checked(certificate.surfaces, leaf.surfaces.size(), budget);
        for (const auto& surface : leaf.surfaces)
        {
            budget.add(surface.key.capacity() + surface.boundary.capacity());
            certificate.surfaces.push_back({surface});
        }
    }
    return page;
}
} // namespace binding_detail
using namespace binding_detail;

/** Compiled compressed complete-key groups and stable binding order. */
namespace binding_detail
{
namespace G = PLIC::details::geometry_demand;
struct SharedLocator
{
    std::shared_ptr<const BindingPage> page;
    std::uint8_t row = 0;
};
std::strong_ordering compare(SharedLocator a, SharedLocator b)
{
    const auto& first = *a.page;
    const auto& second = *b.page;
    if (const auto region =
            first.sources[first.source.at(a.row)]->region <=> second.sources[second.source.at(b.row)]->region;
        region != 0)
        return region;
    if (const auto kind = first.keys.index() <=> second.keys.index(); kind != 0)
        return kind;
    if (const auto* keys = std::get_if<CartesianBindingKeys>(&first.keys))
        return keys->at(a.row) <=> std::get<CartesianBindingKeys>(second.keys).at(b.row);
    if (const auto* keys = std::get_if<ProductBindingKeys>(&first.keys))
    {
        const auto& other = std::get<ProductBindingKeys>(second.keys);
        if (const auto base = keys->bases[keys->base.at(a.row)].key <=> other.bases[other.base.at(b.row)].key;
            base != 0)
            return base;
        return keys->axes[keys->axis.at(a.row)].key <=> other.axes[other.axis.at(b.row)].key;
    }
    if (const auto* keys = std::get_if<StaticBindingKeys>(&first.keys))
        return keys->at(a.row) <=> std::get<StaticBindingKeys>(second.keys).at(b.row);
    return std::get<ExplicitBindingKeys>(first.keys).keys[a.row] <=>
           std::get<ExplicitBindingKeys>(second.keys).keys[b.row];
}
struct SelectedBinding
{
    SharedLocator location;
    G::OrderStamp stamp;
};
G::Admission binding_admission(adaptive_detail::Budget& budget)
{
    return {&budget, [](void* raw, std::size_t bytes) { static_cast<adaptive_detail::Budget*>(raw)->add(bytes); }};
}
struct BindingKeyPage
{
    std::vector<std::shared_ptr<const BindingPage>> sources;
    BindingColumn<std::uint8_t> source, row;
    BindingColumn<std::uint64_t> generation, position;
    std::size_t count = 0;

    SelectedBinding at(std::size_t slot) const
    {
        return {{sources[source.at(slot)], row.at(slot)}, {generation.at(slot), position.at(slot)}};
    }
    std::size_t bytes() const noexcept
    {
        return sizeof(*this) + sources.capacity() * sizeof(sources[0]) + source.bytes() + row.bytes() +
               generation.bytes() + position.bytes();
    }
    static std::shared_ptr<const BindingKeyPage> make(
        std::span<const SelectedBinding> rows, adaptive_detail::Budget& budget)
    {
        if (rows.empty() || rows.size() > binding_page_size)
            throw std::invalid_argument("Invalid binding key page size.");
        std::array<std::shared_ptr<const BindingPage>, binding_page_size> owners{};
        std::array<std::uint8_t, binding_page_size> selectors{};
        std::size_t unique = 0;
        for (std::size_t slot = 0; slot < rows.size(); ++slot)
        {
            if (!rows[slot].location.page || (slot && compare(rows[slot - 1].location, rows[slot].location) >= 0))
                throw std::invalid_argument("Invalid or duplicate binding key page row.");
            const auto& page = rows[slot].location.page;
            std::size_t index = 0;
            while (index < unique && owners[index].get() != page.get())
                ++index;
            if (index == unique)
                owners[unique++] = page;
            else if (owners[index].owner_before(page) || page.owner_before(owners[index]))
                throw std::invalid_argument("Binding data address has conflicting owners.");
            selectors[slot] = std::uint8_t(index);
        }
        budget.add(sizeof(BindingKeyPage));
        auto result = std::make_shared<BindingKeyPage>();
        reserve_checked(result->sources, unique, budget);
        result->sources.assign(owners.begin(), owners.begin() + unique);
        result->count = rows.size();
        result->source.build(rows.size(), [&](auto slot) { return selectors[slot]; }, budget);
        result->row.build(rows.size(), [&](auto slot) { return rows[slot].location.row; }, budget);
        result->generation.build(rows.size(), [&](auto slot) { return rows[slot].stamp.generation; }, budget);
        result->position.build(rows.size(), [&](auto slot) { return rows[slot].stamp.position; }, budget);
        return result;
    }
};
struct BindingKeyGroup
{
    std::shared_ptr<const BindingKeyPage> page;
    std::size_t logical_size() const noexcept { return page->count; }
    std::size_t exclusive_payload_bytes() const noexcept { return page->bytes(); }
};
struct BindingKeyLess
{
    bool operator()(const BindingKeyGroup& a, const BindingKeyGroup& b) const
    {
        return compare(a.page->at(a.page->count - 1).location, b.page->at(b.page->count - 1).location) < 0;
    }
};
class BindingSelection
{
public:
    using Pages = G::OrderedPages<BindingKeyGroup, BindingKeyLess>;
    Pages::Ptr keys;
    /** A one-group key index is held inline: a singleton key page needs no B+
     * tree node, so it owns and charges no index-page layer. */
    std::optional<BindingKeyGroup> single_group;
    G::GeometryDemandOrder order;
    G::GeometryOwnerManifest data_pages;
    /** The two highest-key pages of an ancestry kind. They stay fluid so partial
     * tails are coalesced in bounded ancestry groups; every earlier page is a
     * frozen complete key range. The window is recorded, never rediscovered by
     * scanning the key index, so an expansion costs O(1) index work per kind. */
    struct FluidTrail
    {
        std::uint8_t kind = 0;
        std::shared_ptr<const BindingPage> below, tail;
    };
    std::vector<FluidTrail> trails;

    [[nodiscard]] const FluidTrail* trail(std::size_t kind) const
    {
        const auto found = std::find_if(trails.begin(), trails.end(),
            [&](const auto& value) { return value.kind == kind; });
        return found == trails.end() ? nullptr : &*found;
    }
    FluidTrail& trail_for(std::size_t kind, adaptive_detail::Budget& budget)
    {
        if (const auto found = std::find_if(trails.begin(), trails.end(),
                [&](const auto& value) { return value.kind == kind; });
            found != trails.end())
            return *found;
        if (trails.size() == trails.capacity())
        {
            const auto next = trails.capacity() ? 2 * trails.capacity() : 2;
            budget.add(next * sizeof(FluidTrail));
            trails.reserve(next);
        }
        trails.push_back(FluidTrail{std::uint8_t(kind), {}, {}});
        return trails.back();
    }

    std::optional<SelectedBinding> key(const AdaptiveCellKey& wanted) const
    {
        if (single_group)
            return search(*single_group, wanted);
        const auto* group = Pages::lower_bound(keys,
            [&](const auto& value)
            {
                const auto last = value.page->at(value.page->count - 1).location;
                return last.page->compare(last.row, wanted) < 0;
            });
        if (!group)
            return std::nullopt;
        return search(*group, wanted);
    }
    static std::optional<SelectedBinding> search(const BindingKeyGroup& group, const AdaptiveCellKey& wanted)
    {
        std::size_t low = 0, high = group.page->count;
        while (low < high)
        {
            const auto mid = low + (high - low) / 2;
            const auto value = group.page->at(mid).location;
            if (value.page->compare(value.row, wanted) < 0)
                low = mid + 1;
            else
                high = mid;
        }
        if (low == group.page->count)
            return std::nullopt;
        const auto found = group.page->at(low);
        return found.location.page->compare(found.location.row, wanted) == 0 ? std::optional(found) : std::nullopt;
    }
    SharedLocator at(std::size_t slot) const
    {
        const auto value = order.at(slot).physical;
        return {std::static_pointer_cast<const BindingPage>(value.source), std::uint8_t(value.row)};
    }
    std::size_t metadata_bytes() const noexcept
    {
        const auto groups = keys ? keys->subtree_payload_bytes + keys->subtree_owner_bytes
                                 : (single_group ? single_group->page->bytes() : 0);
        return sizeof(*this) + order.metadata_payload_bytes() + data_pages.metadata_payload_bytes() +
               data_pages.payload_bytes() + groups;
    }
    void patch_keys(std::span<SelectedBinding> changed, adaptive_detail::Budget& budget)
    {
        std::sort(changed.begin(), changed.end(),
            [](const auto& a, const auto& b) { return compare(a.location, b.location) < 0; });
        for (std::size_t slot = 1; slot < changed.size(); ++slot)
            if (compare(changed[slot - 1].location, changed[slot].location) == 0)
                throw std::invalid_argument("Duplicate binding update key.");
        const Pages editor(binding_admission(budget));
        if (single_group)
        {
            const auto value = *single_group;
            single_group.reset();
            keys = editor.edit(keys, value, &value, false);
        }
        std::size_t cursor = 0;
        while (cursor < changed.size())
        {
            const auto* found = Pages::lower_bound(keys, [&](const auto& value)
                { return compare(value.page->at(value.page->count - 1).location, changed[cursor].location) < 0; });
            bool last = false;
            if (!found && keys)
            {
                found = &Pages::at(keys, keys->size - 1);
                last = true;
            }
            std::optional<BindingKeyGroup> old;
            if (found)
                old = *found;
            std::size_t end = cursor;
            while (end < changed.size() &&
                   (!old || last || compare(changed[end].location, old->page->at(old->page->count - 1).location) <= 0))
                ++end;
            if (end == cursor)
                throw std::logic_error("Binding key update has no target page.");
            G::ScratchRows<SelectedBinding, 2 * binding_page_size> merged(
                (old ? old->page->count : 0) + end - cursor, budget.used, budget.limit);
            std::size_t merged_size = 0;
            std::size_t row = 0, update = cursor;
            while ((old && row < old->page->count) || update < end)
            {
                if (!old || row == old->page->count)
                    merged[merged_size++] = changed[update++];
                else if (update == end)
                    merged[merged_size++] = old->page->at(row++);
                else
                {
                    const auto previous = old->page->at(row);
                    const auto comparison = compare(previous.location, changed[update].location);
                    if (comparison < 0)
                    {
                        merged[merged_size++] = previous;
                        ++row;
                    }
                    else
                    {
                        merged[merged_size++] = changed[update++];
                        row += comparison == 0;
                    }
                }
            }
            if (old)
                keys = editor.edit(keys, *old, nullptr, true);
            const auto groups = merged_size / binding_page_size + (merged_size % binding_page_size != 0);
            std::size_t offset = 0;
            for (std::size_t group = 0; group < groups; ++group)
            {
                const auto count = merged_size / groups + (group < merged_size % groups);
                BindingKeyGroup value{BindingKeyPage::make(merged.span().subspan(offset, count), budget)};
                keys = editor.edit(keys, value, &value, false);
                offset += count;
            }
            cursor = end;
        }
        if (keys && keys->size == 1)
        {
            single_group = Pages::at(keys, 0);
            keys.reset();
        }
    }
    static std::shared_ptr<BindingSelection> make(
        std::span<const SharedLocator> selected, adaptive_detail::Budget& budget)
    {
        budget.add(sizeof(BindingSelection));
        auto result = std::make_shared<BindingSelection>();
        constexpr auto kinds = std::variant_size_v<decltype(AdaptiveCellKey::ancestry)>;
        // Bounded construction scratch stays on the stack; larger selections
        // fall back to an exactly reserved buffer.
        G::ScratchRows<G::SourceSlot, binding_page_size> physical(selected.size(), budget.used, budget.limit);
        for (std::size_t slot = 0; slot < selected.size(); ++slot)
            physical[slot] = {selected[slot].page, selected[slot].row};
        // One manifest row per physical page: slot multiplicity is carried by
        // the reference count, never by duplicate owner rows. The page table is
        // built in one pass so a cold build stays O(N log pages).
        {
            std::map<const BindingPage*, std::pair<SharedLocator, std::size_t>> pages;
            for (const auto& value : selected)
            {
                const auto found = pages.find(value.page.get());
                if (found == pages.end())
                {
                    budget.add(sizeof(typename decltype(pages)::value_type) + 4 * sizeof(void*));
                    pages.emplace(value.page.get(), std::pair{value, 1});
                }
                else
                    ++found->second.second;
            }
            for (const auto& [address, record] : pages)
                result->data_pages = result->data_pages.retained(
                    0, 0, record.first.page, record.first.page->bytes(), record.second, binding_admission(budget));
        }
        result->order = G::GeometryDemandOrder{}.prepended(physical.span(), {}, binding_admission(budget));
        G::ScratchRows<SelectedBinding, binding_page_size> indexed(selected.size(), budget.used, budget.limit);
        for (std::size_t slot = 0; slot < selected.size(); ++slot)
            indexed[slot] = {selected[slot], result->order.at(slot).stamp};
        result->patch_keys(indexed.span(), budget);
        // The two highest-key *selected* pages of each kind stay fluid. A reused
        // page can hold unselected physical rows, so the anchors must be rows
        // this selection actually references; the expansion path only repacks
        // rows whose key-index location still names their physical page.
        std::array<SharedLocator, kinds> first_anchor{};
        std::array<SharedLocator, kinds> second_anchor{};
        std::array<bool, kinds> has_first{}, has_second{};
        for (const auto& value : selected)
        {
            const auto kind = std::size_t(value.page->keys.index());
            if (!has_first[kind] || compare(value, first_anchor[kind]) > 0)
            {
                const auto previous_first = first_anchor[kind];
                const bool had_first = has_first[kind];
                first_anchor[kind] = value;
                has_first[kind] = true;
                // The displaced anchor is the second fluid page only when it
                // lives on a different physical page.
                if (had_first && previous_first.page.get() != value.page.get() &&
                    (!has_second[kind] || compare(previous_first, second_anchor[kind]) > 0))
                {
                    second_anchor[kind] = previous_first;
                    has_second[kind] = true;
                }
            }
            else if (value.page.get() != first_anchor[kind].page.get() &&
                     (!has_second[kind] || compare(value, second_anchor[kind]) > 0))
            {
                second_anchor[kind] = value;
                has_second[kind] = true;
            }
        }
        for (std::size_t kind = 0; kind < kinds; ++kind)
        {
            if (!has_first[kind])
                continue;
            auto& entry = result->trail_for(kind, budget);
            entry.tail = first_anchor[kind].page;
            entry.below = has_second[kind] ? second_anchor[kind].page : nullptr;
        }
        return result;
    }
    template<class Visitor> void visit_metadata(const Visitor& visitor) const
    {
        // Explicit diagnostic/census traversal. Mutation uses bounded key groups.
        if (single_group)
        {
            visitor(single_group->page, single_group->page->bytes());
            return;
        }
        Pages::visit_pages(keys,
            [&](const auto& node)
            {
                visitor(node, node->payload_bytes);
                for (const auto& group : node->rows)
                    visitor(group.page, group.page->bytes());
            });
    }
};
} // namespace binding_detail

class Bindings
{
public:
    static std::shared_ptr<const Bindings> build(
        std::span<const AdaptiveLeaf* const>, const BuildFacts&, const Bindings*, bool, adaptive_detail::Budget&);
    static std::shared_ptr<const Bindings> extended(const Bindings&, std::span<const AdaptiveLeaf* const>,
        const BuildFacts&, const AdaptiveLeafLayout&, adaptive_detail::Budget&);
    [[nodiscard]] std::size_t size() const noexcept { return selection_->order.size(); }
    [[nodiscard]] std::optional<std::size_t> find(const AdaptiveLeaf&) const;
    [[nodiscard]] std::optional<std::size_t> find_key(const AdaptiveCellKey&) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return sizeof(*this) + selection_->metadata_bytes(); }
    [[nodiscard]] std::size_t exclusive_bytes() const noexcept { return sizeof(*this); }
    [[nodiscard]] SharedLocator shared_location(std::size_t slot) const { return selection_->at(slot); }
    [[nodiscard]] AdaptiveCellMetrics metric(std::size_t slot) const
    {
        const auto row = shared_location(slot);
        return row.page->metric(row.row);
    }
    [[nodiscard]] GeometryEvaluationPolicy policy(std::size_t slot) const
    {
        const auto row = shared_location(slot);
        return decode_policy(row.page->policy.at(row.row));
    }
    [[nodiscard]] std::size_t unselected(std::span<const AdaptiveLeaf* const>) const;
    void append_unselected(std::span<const AdaptiveLeaf* const>, const AdaptiveLeafLayout&,
        std::vector<std::unique_ptr<AdaptiveLeaf>>&, std::vector<const AdaptiveLeaf*>&, adaptive_detail::Budget&) const;

    template<class Visitor> void visit_children(const Visitor& visitor, bool recipes = true) const
    {
        visitor(selection_, sizeof(BindingSelection));
        selection_->visit_metadata(visitor);
        struct Context
        {
            const Visitor& visitor;
            bool recipes;
        } context{visitor, recipes};
        const G::MetadataSink metadata{&context,
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t)
            {
                static_cast<Context*>(raw)->visitor(owner, payload);
                return true;
            }};
        selection_->data_pages.visit_dependencies(metadata,
            {&context,
                [](void* raw, const G::OwnerRecord& row)
                {
                    const auto& visit = static_cast<Context*>(raw)->visitor;
                    const auto page = std::static_pointer_cast<const BindingPage>(row.owner);
                    visit(page, page->bytes());
                    // A binding source object is construction-charged with its
                    // page; only its immutable recipe children were not.
                    for (const auto& source : page->sources)
                    {
                        visit(source, source->bytes());
                        if (static_cast<Context*>(raw)->recipes)
                            source->visit_children(visit);
                    }
                }});
        selection_->order.visit_dependencies(metadata, {nullptr, [](void*, G::SourceKind,
            const std::shared_ptr<const void>&) {}});
    }
    template<class Visitor> void visit_certificates(const Visitor& visitor) const
    {
        selection_->data_pages.visit(
            [&](const G::OwnerRecord& row)
            {
                const auto* page = static_cast<const BindingPage*>(row.owner.get());
                for (const auto& certificate : page->certificates)
                    visitor(certificate);
            });
    }
    void visit_source_descriptors(const std::function<void(const void*)>& visitor) const
    {
        selection_->data_pages.visit(
            [&](const G::OwnerRecord& row)
            {
                const auto* page = static_cast<const BindingPage*>(row.owner.get());
                for (const auto& source : page->sources)
                    visitor(source.get());
            });
    }

private:
    std::shared_ptr<const BindingSelection> selection_;
};
std::optional<std::size_t> Bindings::find_key(const AdaptiveCellKey& key) const
{
    const auto found = selection_->key(key);
    return found ? std::optional(selection_->order.rank(found->stamp)) : std::nullopt;
}
std::optional<std::size_t> Bindings::find(const AdaptiveLeaf& leaf) const
{
    const auto found = selection_->key(leaf.key);
    return found && found->location.page->matches(found->location.row, leaf)
               ? std::optional(selection_->order.rank(found->stamp))
               : std::nullopt;
}
std::size_t Bindings::unselected(std::span<const AdaptiveLeaf* const> wanted) const
{
    std::size_t selected = 0;
    for (const auto* leaf : wanted)
        selected += selection_->key(leaf->key).has_value();
    return size() - selected;
}
void Bindings::append_unselected(std::span<const AdaptiveLeaf* const> wanted, const AdaptiveLeafLayout& layout,
    std::vector<std::unique_ptr<AdaptiveLeaf>>& detached, std::vector<const AdaptiveLeaf*>& output,
    adaptive_detail::Budget& budget) const
{
    // Cross-purpose cold initialization retains the established full-selection
    // contract. Same-purpose demand expansion never calls this materializer.
    for (std::size_t slot = 0; slot < size(); ++slot)
    {
        const auto row = shared_location(slot);
        const auto found = std::lower_bound(wanted.begin(), wanted.end(), row.row,
            [&](const auto* leaf, auto local) { return row.page->compare(local, leaf->key) > 0; });
        if (found != wanted.end() && row.page->compare(row.row, (*found)->key) == 0)
            continue;
        auto key = row.page->copy_key(row.row, budget);
        if (const auto* owned = layout.find(key); owned && row.page->matches(row.row, *owned))
        {
            output.push_back(owned);
            continue;
        }
        if (detached.size() == detached.capacity())
            reserve_checked(detached, std::min(size(), std::max(std::size_t{1}, 2 * detached.capacity())), budget);
        budget.add(sizeof(AdaptiveLeaf));
        detached.push_back(std::make_unique<AdaptiveLeaf>(row.page->copy_leaf(row.row, std::move(key), budget)));
        output.push_back(detached.back().get());
    }
}
std::shared_ptr<const Bindings> Bindings::build(std::span<const AdaptiveLeaf* const> leaves, const BuildFacts& facts,
    const Bindings* previous, bool, adaptive_detail::Budget& budget)
{
    budget.add(sizeof(Bindings));
    auto result = std::make_shared<Bindings>();
    G::ScratchRows<std::size_t, binding_page_size> order(leaves.size(), budget.used, budget.limit);
    std::iota(order.data(), order.data() + order.size(), std::size_t{0});
    std::sort(order.data(), order.data() + order.size(), [&](auto a, auto b) { return leaves[a]->key < leaves[b]->key; });
    for (std::size_t slot = 1; slot < order.size(); ++slot)
        if (leaves[order[slot - 1]]->key == leaves[order[slot]]->key)
            throw std::invalid_argument("Geometry snapshot contains a duplicate persistent cell key.");
    G::ScratchRows<SharedLocator, binding_page_size> locations(leaves.size(), budget.used, budget.limit);
    constexpr auto kinds = std::variant_size_v<decltype(AdaptiveCellKey::ancestry)>;
    std::array<std::array<std::size_t, binding_page_size>, kinds> pending{};
    std::array<std::size_t, kinds> counts{};
    SourceTable sources;
    const auto publish = [&](std::size_t kind)
    {
        const auto rows = std::span(pending[kind]).first(counts[kind]);
        auto page = make_binding_page(leaves, rows, facts, sources, budget);
        for (std::size_t row = 0; row < rows.size(); ++row)
            locations[rows[row]] = {page, std::uint8_t(row)};
        counts[kind] = 0;
    };
    for (std::size_t position = 0; position < order.size(); ++position)
    {
        const auto slot = order[position];
        if (previous)
            if (const auto old = previous->find(*leaves[slot]))
            {
                bool same = true;
                if (!facts.metrics.empty())
                {
                    const auto metric = previous->metric(*old);
                    same = fact_equal(metric.volume, facts.metrics[slot].volume) &&
                           std::equal(metric.centroid.begin(), metric.centroid.end(),
                               facts.metrics[slot].centroid.begin(), fact_equal);
                }
                if (same && policy_bits(previous->policy(*old)) == policy_bits(facts.policy[slot]))
                {
                    locations[slot] = previous->shared_location(*old);
                    continue;
                }
            }
        const auto kind = leaves[slot]->key.ancestry.index();
        pending[kind][counts[kind]++] = slot;
        if (counts[kind] == binding_page_size)
            publish(kind);
    }
    for (std::size_t kind = 0; kind < kinds; ++kind)
        if (counts[kind])
            publish(kind);
    result->selection_ = BindingSelection::make(locations.span(), budget);
    return result;
}
namespace
{
struct RetainedPage
{
    std::shared_ptr<const BindingPage> page;
    std::size_t references = 0;
};
/** Replace the physical pages of a selection in two bounded manifest edits per
 * page instead of two persistent edits per slot. */
void apply_page_holds(PLIC::details::geometry_demand::GeometryOwnerManifest& pages,
    std::span<const SharedLocator> replaced, std::span<const RetainedPage> inserted, adaptive_detail::Budget& budget)
{
    using namespace PLIC::details::geometry_demand;
    const auto admission = binding_admission(budget);
    std::vector<std::pair<std::shared_ptr<const BindingPage>, std::size_t>> released;
    for (const auto& value : replaced)
    {
        const auto found = std::find_if(
            released.begin(), released.end(), [&](const auto& row) { return row.first.get() == value.page.get(); });
        if (found == released.end())
            released.push_back({value.page, 1});
        else
            ++found->second;
    }
    for (const auto& [page, references] : released)
        pages = pages.retired(0, 0, page, references, admission);
    for (const auto& value : inserted)
        pages = pages.retained(0, 0, value.page, value.page->bytes(), value.references, admission);
}
} // namespace
std::shared_ptr<const Bindings> Bindings::extended(const Bindings& previous,
    std::span<const AdaptiveLeaf* const> leaves, const BuildFacts& facts, const AdaptiveLeafLayout& layout,
    adaptive_detail::Budget& budget)
{
    if (previous.selection_->order.generation() == std::numeric_limits<std::uint64_t>::max())
        throw std::invalid_argument("Binding demand generation overflows.");
    std::vector<std::size_t> sorted;
    reserve_checked(sorted, leaves.size(), budget);
    sorted.resize(leaves.size());
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](auto a, auto b) { return leaves[a]->key < leaves[b]->key; });
    for (std::size_t row = 1; row < sorted.size(); ++row)
        if (leaves[sorted[row - 1]]->key == leaves[sorted[row]]->key)
            throw std::invalid_argument("Duplicate geometry demand key.");
    budget.add(sizeof(Bindings) + sizeof(BindingSelection));
    auto result = std::make_shared<Bindings>();
    auto next = std::make_shared<BindingSelection>(*previous.selection_);
    std::vector<SharedLocator> prefix;
    std::vector<std::optional<SelectedBinding>> original;
    std::vector<unsigned char> needs_data;
    reserve_checked(prefix, leaves.size(), budget);
    reserve_checked(original, leaves.size(), budget);
    reserve_checked(needs_data, leaves.size(), budget);
    prefix.resize(leaves.size());
    original.resize(leaves.size());
    needs_data.resize(leaves.size(), 1);
    constexpr auto kinds = std::variant_size_v<decltype(AdaptiveCellKey::ancestry)>;
    std::array<std::size_t, kinds> added{};
    std::vector<G::OrderStamp> retired;
    reserve_checked(retired, leaves.size(), budget);
    for (std::size_t slot = 0; slot < leaves.size(); ++slot)
    {
        original[slot] = previous.selection_->key(leaves[slot]->key);
        if (original[slot])
        {
            const auto& old = *original[slot];
            retired.push_back(old.stamp);
            bool same = old.location.page->matches(old.location.row, *leaves[slot]) &&
                        old.location.page->policy.at(old.location.row) == policy_bits(facts.policy[slot]);
            if (same && !facts.metrics.empty())
            {
                const auto metric = old.location.page->metric(old.location.row);
                same = fact_equal(metric.volume, facts.metrics[slot].volume) &&
                       std::equal(metric.centroid.begin(), metric.centroid.end(), facts.metrics[slot].centroid.begin(),
                           fact_equal);
            }
            if (same)
            {
                needs_data[slot] = 0;
                prefix[slot] = old.location;
            }
        }
        if (needs_data[slot])
            ++added[leaves[slot]->key.ancestry.index()];
    }
    std::vector<SelectedBinding> key_changes;
    std::vector<G::OrderedSourceSlot> rebound;
    std::array<std::array<std::shared_ptr<const BindingPage>, 2>, kinds> fluid{};
    std::size_t repacked_rows = 0;
    for (std::size_t kind = 0; kind < kinds; ++kind)
    {
        if (!added[kind])
            continue;
        if (const auto* trail = next->trail(kind))
        {
            fluid[kind][1] = trail->tail;
            fluid[kind][0] = trail->below && trail->below.get() != trail->tail.get() ? trail->below : nullptr;
        }
        for (const auto& page : fluid[kind])
            if (page)
                repacked_rows += page->row_count;
    }
    reserve_checked(key_changes, repacked_rows + leaves.size(), budget);
    reserve_checked(rebound, repacked_rows, budget);
    const auto requested_slot = [&](const BindingPage& page, std::size_t row) -> std::optional<std::size_t>
    {
        const auto found = std::lower_bound(sorted.begin(), sorted.end(), row,
            [&](auto slot, auto local) { return page.compare(local, leaves[slot]->key) > 0; });
        return found != sorted.end() && page.compare(row, leaves[*found]->key) == 0 ? std::optional(*found)
                                                                                    : std::nullopt;
    };
    for (std::size_t kind = 0; kind < kinds; ++kind)
    {
        if (!added[kind])
            continue;
        const auto& old_pages = fluid[kind];
        std::size_t old_rows = 0;
        for (const auto& page : old_pages)
            if (page)
                old_rows += page->row_count;
        struct RowOrigin
        {
            std::optional<std::size_t> requested;
            std::optional<SelectedBinding> old;
        };
        const auto capacity = old_rows + added[kind];
        std::vector<const AdaptiveLeaf*> assembled;
        std::vector<std::unique_ptr<AdaptiveLeaf>> detached;
        std::vector<RowOrigin> origins;
        BuildFacts packed;
        reserve_checked(assembled, capacity, budget);
        reserve_checked(detached, old_rows, budget);
        reserve_checked(origins, capacity, budget);
        reserve_checked(packed.policy, capacity, budget);
        if (!facts.metrics.empty())
            reserve_checked(packed.metrics, capacity, budget);
        SourceTable known_sources;
        for (const auto& old_tail : old_pages)
        {
            if (!old_tail)
                continue;
            for (std::size_t row = 0; row < old_tail->row_count; ++row)
            {
                auto key = old_tail->copy_key(row, budget);
                const auto old = previous.selection_->key(key);
                // A reused page can hold physical rows outside this selection.
                // Only rows this selection still binds to this very page carry
                // a manifest hold and may be repacked.
                if (!old || old->location.page.get() != old_tail.get() || old->location.row != row)
                    continue;
                const auto wanted = requested_slot(*old_tail, row);
                if (wanted && needs_data[*wanted])
                    continue;
                const AdaptiveLeaf* leaf = layout.find(key);
                if (!leaf || !old_tail->matches(row, *leaf))
                {
                    budget.add(sizeof(AdaptiveLeaf));
                    detached.push_back(
                        std::make_unique<AdaptiveLeaf>(old_tail->copy_leaf(row, std::move(key), budget)));
                    leaf = detached.back().get();
                }
                assembled.push_back(leaf);
                origins.push_back({wanted, old});
                packed.policy.push_back(decode_policy(old_tail->policy.at(row)));
                if (!facts.metrics.empty())
                    packed.metrics.push_back(old_tail->metric(row));
                if (!known_sources.contains(leaf))
                {
                    budget.add(sizeof(SourceTable::value_type) + 4 * sizeof(void*));
                    known_sources.emplace(leaf, old_tail->sources[old_tail->source.at(row)]);
                }
            }
        }
        for (const auto slot : sorted)
            if (needs_data[slot] && leaves[slot]->key.ancestry.index() == kind)
            {
                assembled.push_back(leaves[slot]);
                origins.push_back({slot, original[slot]});
                packed.policy.push_back(facts.policy[slot]);
                if (!facts.metrics.empty())
                    packed.metrics.push_back(facts.metrics[slot]);
            }
        // A repacked tail is a page of the same kind as the full-build pages:
        // its rows must stay in complete persistent-key order. Building the new
        // page from "old tail order then requested order" would break that
        // invariant (BindingPage::compare and requested_slot binary-search the
        // rows) and force every compressed key/order column into its dense
        // fallback. Rank the merged rows once; pages then index through it
        // without materializing permuted copies of the row metadata.
        std::vector<std::size_t> rank;
        reserve_checked(rank, assembled.size(), budget);
        rank.resize(assembled.size());
        std::iota(rank.begin(), rank.end(), 0);
        std::sort(rank.begin(), rank.end(), [&](auto a, auto b) { return assembled[a]->key < assembled[b]->key; });
        if (const auto* trail = next->trail(kind))
            if (trail->tail || trail->below)
            {
                auto& entry = next->trail_for(kind, budget);
                entry.tail.reset();
                entry.below.reset();
            }
        std::vector<SharedLocator> replaced;
        std::vector<RetainedPage> inserted;
        reserve_checked(replaced, assembled.size(), budget);
        reserve_checked(inserted, assembled.size(), budget);
        for (std::size_t begin = 0; begin < assembled.size(); begin += binding_page_size)
        {
            const auto count = std::min(binding_page_size, assembled.size() - begin);
            std::array<std::size_t, binding_page_size> rows{};
            std::copy(rank.begin() + begin, rank.begin() + begin + count, rows.begin());
            auto page = make_binding_page(assembled, std::span(rows).first(count), packed, known_sources, budget);
            inserted.push_back({page, count});
            for (std::size_t row = 0; row < count; ++row)
            {
                const auto& origin = origins[rank[begin + row]];
                const SharedLocator replacement{page, std::uint8_t(row)};
                if (origin.old)
                    replaced.push_back(origin.old->location);
                if (origin.requested)
                    prefix[*origin.requested] = replacement;
                else
                {
                    key_changes.push_back({replacement, origin.old->stamp});
                    rebound.push_back({origin.old->stamp, {page, row}});
                }
            }
        }
        // Every repacked tail row moves out of its prior page, so each replaced
        // page drops exactly the references this repack held for it. Slot
        // multiplicity is released once per page, not once per row.
        apply_page_holds(next->data_pages, replaced, inserted, budget);
        // Only the two highest-key pages stay fluid; every earlier page is frozen.
        if (!inserted.empty())
        {
            auto& trail = next->trail_for(kind, budget);
            trail.tail = inserted.back().page;
            trail.below = inserted.size() > 1 ? inserted[inserted.size() - 2].page : nullptr;
        }
    }
    std::vector<G::SourceSlot> physical;
    reserve_checked(physical, prefix.size(), budget);
    for (const auto& row : prefix)
    {
        if (!row.page)
            throw std::logic_error("Geometry demand lost its binding data.");
        physical.push_back({row.page, row.row});
    }
    next->order = next->order.prepended(physical, retired, binding_admission(budget));
    if (!rebound.empty())
        next->order = next->order.rebound(rebound, binding_admission(budget));
    for (std::size_t slot = 0; slot < prefix.size(); ++slot)
        key_changes.push_back({prefix[slot], next->order.at(slot).stamp});
    next->patch_keys(key_changes, budget);
    result->selection_ = std::move(next);
    return result;
}
} // namespace geometry_snapshot_detail
