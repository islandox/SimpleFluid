/** @file LocalProductAMR.hh
 * @brief Immutable connected base-patch times axial-run provider state.
 */
#pragma once
#include "geometry/ExtrudedAMR.hh"
#include <stdexcept>
namespace SimpleFluid::Meshes
{
template<class Key> class ProductKeyView
{
public:
    ProductKeyView() = default;
    ProductKeyView(std::shared_ptr<const std::vector<Key>> keys, std::size_t first, std::size_t count)
        : keys_(std::move(keys)), first_(first), count_(count)
    {
        if (!keys_ || first_ > keys_->size() || count_ > keys_->size() - first_)
            throw std::out_of_range("Invalid immutable product-key view.");
    }
    std::span<const Key> span() const noexcept
    {
        return keys_ ? std::span<const Key>(*keys_).subspan(first_, count_) : std::span<const Key>{};
    }
    auto begin() const noexcept { return span().begin(); }
    auto end() const noexcept { return span().end(); }
    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    const Key& operator[](std::size_t i) const { return span()[i]; }
    const Key& front() const { return span().front(); }
    const Key& back() const { return span().back(); }
    const void* storage_identity() const noexcept { return keys_.get(); }
    std::size_t storage_bytes() const noexcept
    {
        if (!keys_)
            return 0;
        std::size_t bytes = sizeof(std::vector<Key>) + keys_->capacity() * sizeof(Key);
        for (const auto& k : *keys_)
            bytes += k.path.capacity() * sizeof(std::uint64_t);
        return bytes;
    }

private:
    std::shared_ptr<const std::vector<Key>> keys_;
    std::size_t first_ = 0, count_ = 0;
};
struct LocalProductPlaneKey
{
    enum class Kind
    {
        RootPoint,
        IntervalMidpoint
    };
    Kind kind = Kind::RootPoint;
    std::uint64_t root_point = 0;
    AxialIntervalKey creator;
    auto operator<=>(const LocalProductPlaneKey&) const = default;
};
struct LocalProductBoundary
{
    enum class Kind
    {
        Side,
        LowerCap,
        UpperCap
    };
    Kind kind = Kind::Side;
    int boundary = -1;
    bool physical = false;
    BaseEdgeKey edge;
    LocalProductPlaneKey plane;
};
struct LocalProductRun
{
    ExtrudedRegion region;
    ProductKeyView<BaseCellKey> base_keys;
    ProductKeyView<AxialIntervalKey> intervals;
    std::vector<LocalProductBoundary> boundaries;
};
/** Provider state only: a mixed/native assembler must join the exported seam
 * patches before publishing fields. No per-3D-cell provider/certificate cache.
 * Marks use persistent product keys; coarsening names the corresponding parent
 * factor and requires complete immediate siblings on the unchanged other factor.
 */
class LocalProductAMR
{
public:
    struct Data;
    explicit LocalProductAMR(ExtrudedAMRGrid grid);
    [[nodiscard]] const std::vector<LocalProductRun>& runs() const noexcept;
    [[nodiscard]] AdaptiveCellKey cell_key(std::size_t run, std::uint64_t local_cell) const;
    [[nodiscard]] ConvexGeometryPtr cell_geometry(std::size_t run, std::uint64_t local_cell) const;
    [[nodiscard]] std::uint64_t material_id() const noexcept;
    [[nodiscard]] const std::shared_ptr<const void>& identity() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] std::shared_ptr<const LocalProductAMR> adapt(const std::vector<ProductCellAncestry>& refine_xy = {},
        const std::vector<ProductCellAncestry>& refine_z = {}, const std::vector<ProductCellAncestry>& coarsen_xy = {},
        const std::vector<ProductCellAncestry>& coarsen_z = {}) const;
    using ContactViolation = ProductContactViolation;
    /** Local admission query including closed face/edge/vertex contacts. Uses exact
     * factor provenance; retains no expanded cell or certificate cache. */
    [[nodiscard]] std::vector<ContactViolation> contact_violations(
        double maximum_span_ratio, unsigned maximum_trace_jump = 1, std::size_t contact_budget = 65536) const;
    struct StorageCounts
    {
        std::size_t cells, runs, base_templates, axis_templates, axial_values, base_vertices, point_recipes, z_recipes,
            seam_descriptors, selection_entries;
    };
    [[nodiscard]] StorageCounts storage_counts() const;
    [[nodiscard]] MeshStorageReport storage_report() const;

private:
    friend class ExtrudedAMR;
    explicit LocalProductAMR(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::Meshes
