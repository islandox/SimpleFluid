/** Compiled persistent exact recipe/page and source-context prepared lookup. */
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotInternal.hh"
#include <bit>
namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
{
namespace prepared_detail
{
namespace G = PLIC::details::geometry_demand;
struct Row
{
    std::uint8_t tag = 0;
    PreparedCartesianPage page;
};
struct Less
{
    bool operator()(const Row& a, const Row& b) const noexcept
    {
        if (a.tag != b.tag)
            return a.tag < b.tag;
        const void* first = a.tag ? static_cast<const void*>(a.page.source.get()) : a.page.recipe.get();
        const void* second = b.tag ? static_cast<const void*>(b.page.source.get()) : b.page.recipe.get();
        if (first != second)
            return std::less<const void*>{}(first, second);
        return !a.tag && a.page.first < b.page.first;
    }
};
using Pages = G::OrderedPages<Row, Less>;
using Root = Pages::Ptr;
G::Admission admission(adaptive_detail::Budget& budget)
{
    return {&budget, [](void* raw, std::size_t bytes) { static_cast<adaptive_detail::Budget*>(raw)->add(bytes); }};
}
Root root(const std::shared_ptr<const void>& pointer)
{
    return std::static_pointer_cast<const Pages::Node>(pointer);
}
const Row* find(const Root& root, const Row& key)
{
    const auto* row = Pages::lower_bound(root, [&](const auto& value) { return Less{}(value, key); });
    return row && !Less{}(key, *row) ? row : nullptr;
}
} // namespace prepared_detail
const PreparedCartesianPage* PreparedCartesianPages::find(
    const std::shared_ptr<const NativeFragmentDescriptor>& recipe, std::uint64_t first) const
{
    if (!root_ || !recipe)
        return nullptr;
    const prepared_detail::Row key{0, {recipe, {}, first, 0}};
    const auto* row = prepared_detail::find(prepared_detail::root(root_), key);
    if (!row || row->page.recipe.owner_before(recipe) || recipe.owner_before(row->page.recipe))
        return nullptr;
    return &row->page;
}
bool PreparedCartesianPages::contains_source(const std::shared_ptr<const PLIC::CartesianCellSource>& source) const
{
    if (!root_ || !source)
        return false;
    const prepared_detail::Row key{1, {{}, source, 0, 0}};
    const auto* row = prepared_detail::find(prepared_detail::root(root_), key);
    return row && !row->page.source.owner_before(source) && !source.owner_before(row->page.source);
}
std::size_t PreparedCartesianPages::size() const noexcept
{
    const auto value = prepared_detail::root(root_);
    return value ? value->size / 2 : 0;
}
std::size_t PreparedCartesianPages::insertion_byte_bound() const
{
    using namespace prepared_detail;
    const auto current = root(root_);
    std::size_t height = 1;
    auto node = current.get();
    while (node && !node->leaf())
    {
        ++height;
        node = node->children.front().get();
    }
    const auto nodes = 2 * (2 * height + 1);
    const auto maximum =
        sizeof(Pages::Node) + G::control_bytes + std::max(G::leaf_capacity * sizeof(Row), G::branch_capacity * sizeof(Root));
    const auto construction = checked_product(nodes, maximum);
    // Admit construction and retained node payload/bookkeeping before optional
    // source work. Two tagged inserts preserve exact source membership lookup.
    return 2 * (G::control_bytes + construction) + PLIC::details::identity_ledger_bookkeeping_bound(nodes + 2);
}
PreparedCartesianPages PreparedCartesianPages::appended(
    const PreparedCartesianPages* previous, PreparedCartesianPage page, adaptive_detail::Budget& budget)
{
    using namespace prepared_detail;
    if (!page.recipe || !page.source || page.first % prepared_cartesian_page_size || !page.rows ||
        std::popcount(page.rows) != page.source->size() ||
        (previous && (previous->find(page.recipe, page.first) || previous->contains_source(page.source))))
        throw std::invalid_argument("Invalid or duplicate prepared Cartesian source page.");
    auto state = previous ? root(previous->root_) : Root{};
    const Pages editor(admission(budget));
    for (std::uint8_t tag = 0; tag < 2; ++tag)
    {
        const Row row{tag, page};
        state = editor.edit(state, row, &row, false);
    }
    PreparedCartesianPages result;
    result.root_ = std::move(state);
    return result;
}
void PreparedCartesianPages::visit_pages(const std::function<void(const PreparedCartesianPage&)>& visitor) const
{
    if (!root_)
        return;
    prepared_detail::Pages::visit_from(
        prepared_detail::root(root_), [](const auto&) { return false; },
        [&](const auto& row)
        {
            if (!row.tag)
                visitor(row.page);
            return true;
        });
}
void PreparedCartesianPages::visit_metadata(
    const std::function<void(const std::shared_ptr<const void>&, std::size_t)>& visitor) const
{
    if (!root_)
        return;
    prepared_detail::Pages::visit_pages(prepared_detail::root(root_),
        [&](const auto& node) { visitor(node, node->payload_bytes); });
}
std::size_t PreparedCartesianPages::metadata_bytes() const noexcept
{
    const auto value = prepared_detail::root(root_);
    return value ? value->subtree_payload_bytes : 0;
}
} // namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
