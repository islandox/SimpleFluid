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

class Bindings
{
public:
    static std::shared_ptr<const Bindings> build(std::span<const AdaptiveLeaf* const>, const BuildFacts&,
        const Bindings*, bool compact_expansion, adaptive_detail::Budget&);
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::optional<std::size_t> find(const AdaptiveLeaf&) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t exclusive_bytes() const noexcept;
    [[nodiscard]] AdaptiveCellMetrics metric(std::size_t) const;
    [[nodiscard]] std::size_t unselected(std::span<const AdaptiveLeaf* const>) const;
    void append_unselected(std::span<const AdaptiveLeaf* const>, const AdaptiveLeafLayout&,
        std::vector<std::unique_ptr<AdaptiveLeaf>>&, std::vector<const AdaptiveLeaf*>&, adaptive_detail::Budget&) const;
    [[nodiscard]] GeometryEvaluationPolicy policy(std::size_t) const;
    template<class Visitor> void visit_children(const Visitor& visit) const
    {
        for (const auto& page : pages_)
        {
            visit(page, page->bytes());
            page->visit_children(visit);
        }
    }
    template<class Visitor> void visit_certificates(const Visitor& visit) const
    {
        for (const auto& page : pages_)
            for (const auto& certificate : page->certificates)
                visit(certificate);
    }

private:
    struct Locator
    {
        std::size_t page;
        std::uint8_t row;
    };
    struct Selection
    {
        Locator first;
        std::variant<std::monostate, std::vector<std::uint32_t>, std::vector<Locator>> irregular;
    };
    struct ReverseOrder
    {
    };
    [[nodiscard]] Locator locate(std::size_t) const;
    [[nodiscard]] bool has_key(Locator, std::span<const AdaptiveLeaf* const>) const;
    std::vector<std::shared_ptr<const BindingPage>> pages_;
    std::vector<Selection> selection_;
    std::variant<std::monostate, ReverseOrder, std::vector<std::uint16_t>, std::vector<std::size_t>> ordered_;
    std::size_t size_ = 0;
};
Bindings::Locator Bindings::locate(std::size_t slot) const
{
    if (slot >= size_)
        throw std::out_of_range("Invalid geometry snapshot binding slot.");
    const auto& block = selection_[slot / binding_page_size];
    const auto local = slot % binding_page_size;
    if (const auto* compact = std::get_if<std::vector<std::uint32_t>>(&block.irregular))
    {
        const auto value = (*compact)[local];
        return {value / binding_page_size, std::uint8_t(value % binding_page_size)};
    }
    if (const auto* wide = std::get_if<std::vector<Locator>>(&block.irregular))
        return (*wide)[local];
    return {block.first.page, std::uint8_t(block.first.row + local)};
}
bool Bindings::has_key(Locator row, std::span<const AdaptiveLeaf* const> ordered) const
{
    const auto& page = *pages_[row.page];
    const auto found = std::lower_bound(ordered.begin(), ordered.end(), row.row,
        [&](const auto* leaf, auto slot) { return page.compare(slot, leaf->key) > 0; });
    return found != ordered.end() && page.compare(row.row, (*found)->key) == 0;
}
std::size_t Bindings::unselected(std::span<const AdaptiveLeaf* const> ordered) const
{
    std::size_t count = 0;
    for (std::size_t slot = 0; slot < size_; ++slot)
        count += !has_key(locate(slot), ordered);
    return count;
}
void Bindings::append_unselected(std::span<const AdaptiveLeaf* const> ordered, const AdaptiveLeafLayout& layout,
    std::vector<std::unique_ptr<AdaptiveLeaf>>& detached, std::vector<const AdaptiveLeaf*>& selection,
    adaptive_detail::Budget& budget) const
{
    for (std::size_t slot = 0; slot < size_; ++slot)
    {
        const auto row = locate(slot);
        if (has_key(row, ordered))
            continue;
        const auto& page = *pages_[row.page];
        auto key = page.copy_key(row.row, budget);
        if (const auto* owned = layout.find(key); owned && page.matches(row.row, *owned))
        {
            selection.push_back(owned);
            continue;
        }
        if (detached.size() == detached.capacity())
        {
            const auto old = detached.capacity();
            const auto next = old + std::min(std::max(old, std::size_t(1)), size_ - old);
            reserve_checked(detached, next, budget);
        }
        budget.add(sizeof(AdaptiveLeaf));
        detached.push_back(std::make_unique<AdaptiveLeaf>(page.copy_leaf(row.row, std::move(key), budget)));
        selection.push_back(detached.back().get());
    }
}
AdaptiveCellMetrics Bindings::metric(std::size_t slot) const
{
    const auto locator = locate(slot);
    return pages_[locator.page]->metric(locator.row);
}
GeometryEvaluationPolicy Bindings::policy(std::size_t slot) const
{
    const auto locator = locate(slot);
    return decode_policy(pages_[locator.page]->policy.at(locator.row));
}
std::optional<std::size_t> Bindings::find(const AdaptiveLeaf& leaf) const
{
    std::size_t lower = 0, upper = size_;
    while (lower < upper)
    {
        const auto middle = std::midpoint(lower, upper);
        const auto slot = std::visit(
            [&](const auto& order) -> std::size_t
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(order)>, ReverseOrder>)
                    return size_ - 1 - middle;
                else if constexpr (requires { order[middle]; })
                    return order[middle];
                else
                    return middle;
            },
            ordered_);
        const auto locator = locate(slot);
        const auto& page = *pages_[locator.page];
        const auto compare = page.compare(locator.row, leaf.key);
        if (compare == 0)
            return page.matches(locator.row, leaf) ? std::optional(slot) : std::nullopt;
        if (compare < 0)
            lower = middle + 1;
        else
            upper = middle;
    }
    return std::nullopt;
}
std::size_t Bindings::exclusive_bytes() const noexcept
{
    auto bytes = sizeof(*this) + pages_.capacity() * sizeof(pages_[0]) + selection_.capacity() * sizeof(Selection);
    const auto count = [&](const auto& value)
    {
        if constexpr (requires { value.capacity(); })
            bytes += value.capacity() * sizeof(typename std::decay_t<decltype(value)>::value_type);
    };
    std::visit(count, ordered_);
    for (const auto& block : selection_)
        std::visit(count, block.irregular);
    return bytes;
}
std::size_t Bindings::metadata_bytes() const noexcept
{
    auto bytes = exclusive_bytes();
    for (const auto& page : pages_)
        bytes += page->bytes();
    return bytes;
}
std::shared_ptr<const Bindings> Bindings::build(std::span<const AdaptiveLeaf* const> leaves, const BuildFacts& facts,
    const Bindings* previous, bool compact_expansion, adaptive_detail::Budget& budget)
{
    budget.add(sizeof(Bindings));
    auto result = std::make_shared<Bindings>();
    result->size_ = leaves.size();
    std::vector<std::size_t> order;
    reserve_checked(order, leaves.size(), budget);
    order.resize(leaves.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return leaves[a]->key < leaves[b]->key; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (leaves[order[i - 1]]->key == leaves[order[i]]->key)
            throw std::invalid_argument("Geometry snapshot contains a duplicate persistent cell key.");

    std::vector<Locator> locations;
    reserve_checked(locations, leaves.size(), budget);
    locations.resize(leaves.size());
    SourceTable known_sources;
    std::map<const BindingPage*, std::size_t> reused_pages;
    const auto append_page = [&](auto page)
    {
        if (result->pages_.size() == result->pages_.capacity())
        {
            const auto old = result->pages_.capacity();
            const auto next = old + std::min(std::max(old, std::size_t(1)), leaves.size() - old);
            charge_items(budget, next, sizeof(result->pages_[0]));
            result->pages_.reserve(next);
            if (result->pages_.capacity() > next)
                charge_items(budget, result->pages_.capacity() - next, sizeof(result->pages_[0]));
        }
        result->pages_.push_back(std::move(page));
        return result->pages_.size() - 1;
    };
    const auto reused = [&](std::size_t slot) -> std::optional<Locator>
    {
        if (!previous)
            return std::nullopt;
        const auto old = previous->find(*leaves[slot]);
        if (!old || policy_bits(previous->policy(*old)) != policy_bits(facts.policy[slot]))
            return std::nullopt;
        if (!facts.metrics.empty())
        {
            const auto old_metric = previous->metric(*old);
            const auto& metric = facts.metrics[slot];
            if (!fact_equal(old_metric.volume, metric.volume) ||
                !std::equal(
                    old_metric.centroid.begin(), old_metric.centroid.end(), metric.centroid.begin(), fact_equal))
                return std::nullopt;
        }
        return previous->locate(*old);
    };
    constexpr auto no_page = std::numeric_limits<std::size_t>::max();
    constexpr auto ancestry_kinds = std::variant_size_v<decltype(AdaptiveCellKey::ancestry)>;
    std::array<bool, ancestry_kinds> changed_kinds{};
    for (const auto slot : order)
    {
        if (const auto old = reused(slot))
            locations[slot] = *old;
        else
        {
            locations[slot] = {no_page, 0};
            changed_kinds[leaves[slot]->key.ancestry.index()] = true;
        }
    }
    std::array<std::array<std::size_t, binding_page_size>, ancestry_kinds> pending;
    std::array<std::size_t, ancestry_kinds> pending_counts{};
    const auto publish = [&](std::size_t kind)
    {
        const auto rows = std::span(pending[kind]).first(pending_counts[kind]);
        for (const auto slot : rows)
        {
            const auto old = locations[slot];
            if (old.page == no_page || known_sources.contains(leaves[slot]))
                continue;
            const auto& page = *previous->pages_[old.page];
            budget.add(sizeof(SourceTable::value_type) + 4 * sizeof(void*));
            known_sources.emplace(leaves[slot], page.sources[page.source.at(old.row)]);
        }
        const auto page = append_page(make_binding_page(leaves, rows, facts, known_sources, budget));
        for (std::size_t row = 0; row < rows.size(); ++row)
            locations[rows[row]] = {page, std::uint8_t(row)};
        pending_counts[kind] = 0;
    };
    for (const auto slot : order)
    {
        const auto old = locations[slot];
        const auto kind = leaves[slot]->key.ancestry.index();
        if (old.page != no_page)
        {
            const auto& page = previous->pages_[old.page];
            if (!compact_expansion || !changed_kinds[kind] || page->row_count == binding_page_size)
            {
                auto found = reused_pages.find(page.get());
                if (found == reused_pages.end())
                {
                    budget.add(sizeof(decltype(reused_pages)::value_type) + 4 * sizeof(void*));
                    found = reused_pages.emplace(page.get(), append_page(page)).first;
                }
                locations[slot] = {found->second, old.row};
                continue;
            }
        }
        pending[kind][pending_counts[kind]++] = slot;
        if (pending_counts[kind] == binding_page_size)
            publish(kind);
    }
    for (std::size_t kind = 0; kind < ancestry_kinds; ++kind)
        if (pending_counts[kind])
            publish(kind);
    const auto blocks = leaves.size() / binding_page_size + (leaves.size() % binding_page_size != 0);
    reserve_checked(result->selection_, blocks, budget);
    for (std::size_t begin = 0; begin < locations.size();)
    {
        const auto count = std::min(binding_page_size, locations.size() - begin);
        const auto first = locations[begin];
        auto& block = result->selection_.emplace_back(Selection{first, {}});
        for (std::size_t row = 1; row < count; ++row)
            if (locations[begin + row].page != first.page || locations[begin + row].row != first.row + row)
            {
                const auto view = std::span(locations).subspan(begin, count);
                const auto compact_page_limit = std::numeric_limits<std::uint32_t>::max() / binding_page_size;
                if (std::all_of(
                        view.begin(), view.end(), [&](const auto& item) { return item.page <= compact_page_limit; }))
                {
                    auto& compact = block.irregular.emplace<std::vector<std::uint32_t>>();
                    reserve_checked(compact, count, budget);
                    for (const auto& item : view)
                        compact.push_back(std::uint32_t(item.page * binding_page_size + item.row));
                }
                else
                {
                    auto& wide = block.irregular.emplace<std::vector<Locator>>();
                    reserve_checked(wide, count, budget);
                    wide.assign(view.begin(), view.end());
                }
                break;
            }
        begin += count;
    }
    if (std::is_sorted(order.begin(), order.end()))
        return result;
    if (std::is_sorted(order.rbegin(), order.rend()))
        result->ordered_ = ReverseOrder{};
    else
    {
        if (order.size() - 1 <= std::numeric_limits<std::uint16_t>::max())
        {
            auto& compact = result->ordered_.emplace<std::vector<std::uint16_t>>();
            reserve_checked(compact, order.size(), budget);
            for (const auto slot : order)
                compact.push_back(std::uint16_t(slot));
        }
        else
            result->ordered_ = std::move(order);
    }
    return result;
}
} // namespace geometry_snapshot_detail
