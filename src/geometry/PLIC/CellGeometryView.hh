/** @file CellGeometryView.hh
 * @brief Owning immutable indexed geometry views and explicit shell materialization.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid::PLIC
{
namespace details
{
class NativeCellGeometrySourceAccess;
class CellGeometryWorkspaceAccess;
struct NativeOriginalFace;
struct NativeFramedBounds;
enum class NativeCellGeometryMoments;
} // namespace details
/** Compact original Cartesian source recipe. Support slots reference the
 * context's original-support table; absent slots reconstruct canonical supports. */
struct CartesianSourceCellSlot
{
    static constexpr std::uint32_t canonical_support = std::numeric_limits<std::uint32_t>::max();
    std::array<std::uint32_t, 3> lower{}, upper{};
    std::array<std::uint32_t, 6> support_slots{canonical_support, canonical_support, canonical_support,
        canonical_support, canonical_support, canonical_support};
};
/** Construction input transferred into one immutable source context. Distinct
 * contexts retain independent ownership even for equal coordinate tables. */
struct CartesianSourceDescription
{
    std::array<std::vector<double>, 3> axes;
    std::vector<CartesianSourceCellSlot> cells;
    std::vector<PlaneSupportIdentity<ArithmeticMode::Ordinary>> original_supports;
};
template<ArithmeticMode Mode> class CellGeometryRange;
class CellGeometryOperationWorkspace;
/** Native face numerical outputs, preserving the existing planar query algorithms. */
struct CellGeometryFaceMetrics
{
    GeometryValue<ArithmeticMode::Ordinary> area;
    GeometryPoint centroid{}, area_vector{};
    int orientation = 0;
};
/** Immutable Cartesian source and necessary primed represented metric facts.
 * It owns no mesh, hierarchy, general planes, topology or point-export cache.
 * Volumes/moments are represented PLIC measures, independent of native binary64
 * mesh metrics. The higher-level adapter authenticates eligibility and ancestry. */
class CartesianCellSource : public std::enable_shared_from_this<CartesianCellSource>
{
public:
    struct Impl;
    CartesianCellSource(const CartesianCellSource&) = delete;
    CartesianCellSource& operator=(const CartesianCellSource&) = delete;
    [[nodiscard]] static std::shared_ptr<const CartesianCellSource> make(
        CartesianSourceDescription, GeometryLimits = {});
    /** Fresh raw query state over the same immutable coordinates/support facts. */
    [[nodiscard]] static std::shared_ptr<const CartesianCellSource> clone_deferred(
        const std::shared_ptr<const CartesianCellSource>&, GeometryLimits = {});
    [[nodiscard]] const void* base_identity() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    /** Exclusive buffers; shared original-support ancestry is admitted by a ledger. */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] GeometryValue<ArithmeticMode::Ordinary> volume(std::size_t) const;
    [[nodiscard]] GeometryMoments<ArithmeticMode::Ordinary> moments(std::size_t, GeometryPoint) const;
    [[nodiscard]] GeometryPoint origin(std::size_t) const;
    [[nodiscard]] GeometryPoint centroid(std::size_t) const;
    [[nodiscard]] std::array<long double, 3> corner(std::size_t, std::size_t) const;
    [[nodiscard]] std::array<GeometryValue<ArithmeticMode::Ordinary>, 3> axis_spans(std::size_t) const;
    [[nodiscard]] std::array<std::array<long double, 2>, 3> closed_contact_bounds(std::size_t) const;
    [[nodiscard]] Support<ArithmeticMode::Ordinary> support(std::size_t, GeometryPoint, GeometryPoint) const;
    [[nodiscard]] PlaneSupportIdentity<ArithmeticMode::Ordinary> source_support(std::size_t, std::size_t) const;
    [[nodiscard]] GeometryValue<ArithmeticMode::Ordinary> face_area(std::size_t, std::size_t) const;
    [[nodiscard]] GeometryPoint face_centroid(std::size_t, std::size_t) const;
    [[nodiscard]] GeometryPoint face_area_vector(std::size_t, std::size_t) const;
    [[nodiscard]] int face_orientation(std::size_t, std::size_t) const;
    [[nodiscard]] bool face_contains(std::size_t, std::size_t, const PlanarFace<ArithmeticMode::Ordinary>&) const;
    [[nodiscard]] bool contains_source_point(
        std::size_t, std::size_t, const PointIdentity<ArithmeticMode::Ordinary>&) const;
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Ordinary> materialize(
        std::size_t, CellGeometryOperationWorkspace&) const;

private:
    friend class GeometryAllocationLedger;
    friend class CellGeometryOperationWorkspace;
    friend class details::RetentionAccess;
    friend class details::NativeCellGeometrySourceAccess;
    [[nodiscard]] static std::shared_ptr<const CartesianCellSource> make_authenticated(
        CartesianSourceDescription, GeometryLimits, details::NativeCellGeometryMoments);
    [[nodiscard]] std::array<std::size_t, 3> retention_bytes() const;
    [[nodiscard]] std::array<std::size_t, 3> base_retention_bytes() const;
    [[nodiscard]] std::shared_ptr<const void> base_owner() const;
    [[nodiscard]] std::span<const PlaneSupportIdentity<ArithmeticMode::Ordinary>> retained_supports() const;
    [[nodiscard]] GeometryLimits query_limits() const;
    [[nodiscard]] std::pair<ConvexGeometryPtr<ArithmeticMode::Ordinary>, ConvexGeometryPtr<ArithmeticMode::Ordinary>>
    materialize_inputs(std::size_t, GeometryLimits) const;
    [[nodiscard]] details::NativeOriginalFace original_face(std::size_t, std::size_t) const;
    [[nodiscard]] details::NativeFramedBounds framed_bounds(std::size_t) const;
    void visit_local_coordinates(
        std::size_t, void*, void (*)(void*, std::size_t, const PointCoordinates<ArithmeticMode::Ordinary>&)) const;
    explicit CartesianCellSource(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
/** A caller operation's cumulative remaining work/storage admission. Temporary
 * owners are scoped here; a surviving result needs its own output ledger. */
class CellGeometryOperationWorkspace
{
public:
    struct Impl;
    explicit CellGeometryOperationWorkspace(GeometryLimits = {});
    ~CellGeometryOperationWorkspace();
    CellGeometryOperationWorkspace(const CellGeometryOperationWorkspace&) = delete;
    CellGeometryOperationWorkspace& operator=(const CellGeometryOperationWorkspace&) = delete;
    [[nodiscard]] GeometryLimits remaining_limits(ArithmeticMode) const;
    [[nodiscard]] std::size_t bytes_used() const noexcept;
    [[nodiscard]] std::size_t operations_used() const noexcept;
    [[nodiscard]] std::size_t pair_tests_used() const noexcept;
    /** Monotonically restrict remaining quotas. Zero remaining work/pairs means
     * exhaustion here, rather than selecting a fresh derived default guard. */
    void restrict_remaining(GeometryLimits);
    /** Admit caller-owned numeric/descriptor storage before allocating it. */
    void charge_temporary_storage(std::size_t count, std::size_t width);
    template<ArithmeticMode Mode>
        requires(Mode != ArithmeticMode::Exact)
    void retain_temporary(const ConvexGeometryPtr<Mode>&);

private:
    friend class CartesianCellSource;
    friend class details::CellGeometryWorkspaceAccess;
    template<ArithmeticMode M>
    friend std::vector<Point<M>> source_points(const CellGeometryRange<ArithmeticMode::Ordinary>&,
        std::span<const std::size_t>, std::span<const NativeSourcePointRequest>, CellGeometryOperationWorkspace&);
    std::unique_ptr<Impl> impl_;
};
template<ArithmeticMode Mode> class CellGeometryCatalog;
template<ArithmeticMode Mode> class CellGeometryView;
template<ArithmeticMode Mode> class CellGeometryRange;
template<ArithmeticMode Mode> class CellGeometryFace;
template<ArithmeticMode Mode> class CellGeometryFaces;
/** Native entry references an actual immutable source context and its cell slot. */
struct NativeCellGeometrySlot
{
    std::size_t source_slot = 0, cell_slot = 0;
};
namespace details
{
/** Stack admission state shared by catalog factories before metadata allocation. */
class SIMPLEFLUID_LOCAL CellCatalogAdmission
{
public:
    explicit CellCatalogAdmission(GeometryLimits limits) : limits_(limits) {}
    void charge(std::size_t count, std::size_t width);

private:
    GeometryLimits limits_;
    std::size_t used_ = 0;
};
} // namespace details

/** Catalog roots pin either a shared Cartesian context or explicit fallback
 * shells. Cartesian slots are available only for Ordinary arithmetic. */
template<ArithmeticMode Mode> class CellGeometryCatalog
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification ||
                  Mode == ArithmeticMode::Exact);

public:
    using SourcePtr = std::shared_ptr<const CartesianCellSource>;
    using SourceTable = std::vector<SourcePtr>;
    using NativeEntry = NativeCellGeometrySlot;
    using Entry = std::variant<NativeEntry, ConvexGeometryPtr<Mode>>;
    CellGeometryCatalog(const CellGeometryCatalog&) = delete;
    CellGeometryCatalog& operator=(const CellGeometryCatalog&) = delete;
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> make(
        SourceTable, std::vector<Entry>, GeometryLimits = {});
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> make(
        SourcePtr, std::vector<Entry>, GeometryLimits = {});
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> cartesian(
        std::shared_ptr<const CartesianCellSource>, GeometryLimits = {})
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> materialized(
        std::vector<ConvexGeometryPtr<Mode>>, GeometryLimits = {});
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> single_existing(
        ConvexGeometryPtr<Mode>, GeometryLimits = {});
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> reordered(
        std::shared_ptr<const CellGeometryCatalog>, std::span<const std::size_t>, GeometryLimits = {});
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] std::size_t source_count() const noexcept { return sources_.size(); }
    [[nodiscard]] std::size_t storage_bytes() const noexcept
    {
        return sizeof(*this) + entries_.capacity() * sizeof(Entry) + sources_.capacity() * sizeof(SourcePtr);
    }

private:
    friend class GeometryAllocationLedger;
    friend class details::RetentionAccess;
    friend class CellGeometryView<Mode>;
    friend class CellGeometryFace<Mode>;
    CellGeometryCatalog() = default;
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog> build(
        SourceTable, std::vector<Entry>, details::CellCatalogAdmission&, bool distinct_sources = false);
    SourceTable sources_;
    std::vector<Entry> entries_;
};
/** One lifetime-safe geometry value, without implicit shell conversion. */
template<ArithmeticMode Mode> class CellGeometryView
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification ||
                  Mode == ArithmeticMode::Exact);

public:
    CellGeometryView() noexcept = default;
    CellGeometryView(std::shared_ptr<const CellGeometryCatalog<Mode>>, std::size_t);
    /** Explicit existing-owner adaptation. A null owner yields an empty view. */
    [[nodiscard]] static CellGeometryView from_existing(ConvexGeometryPtr<Mode>, GeometryLimits = {});
    explicit operator bool() const noexcept { return bool(catalog_); }
    [[nodiscard]] static constexpr ArithmeticMode arithmetic_mode() noexcept { return Mode; }
    [[nodiscard]] bool is_procedural() const;
    bool operator==(const CellGeometryView&) const = default;
    [[nodiscard]] GeometryValue<Mode> volume() const;
    [[nodiscard]] GeometryMoments<Mode> moments() const;
    [[nodiscard]] GeometryMoments<Mode> moments(GeometryPoint) const
        requires(Mode != ArithmeticMode::Exact);
    [[nodiscard]] GeometryPoint centroid() const;
    [[nodiscard]] GeometryPoint origin() const
        requires(Mode != ArithmeticMode::Exact);
    [[nodiscard]] std::array<GeometryValue<Mode>, 3> axis_spans() const;
    [[nodiscard]] std::size_t num_faces() const;
    [[nodiscard]] std::size_t vertex_count() const
        requires(Mode != ArithmeticMode::Exact);
    [[nodiscard]] std::array<long double, 3> corner(std::size_t) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] CellGeometryFace<Mode> ordered_face(std::size_t) const
        requires(Mode != ArithmeticMode::Exact);
    [[nodiscard]] CellGeometryFaces<Mode> faces() const
        requires(Mode != ArithmeticMode::Exact);
    [[nodiscard]] std::array<std::array<long double, 2>, 3> closed_contact_bounds() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] Support<Mode> support(GeometryPoint, GeometryPoint) const
        requires(Mode != ArithmeticMode::Exact);
    [[nodiscard]] PlaneSupportIdentity<ArithmeticMode::Ordinary> source_support(std::size_t) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] bool face_contains(std::size_t, const PlanarFace<ArithmeticMode::Ordinary>&) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] bool contains_source_point(std::size_t, const PointIdentity<ArithmeticMode::Ordinary>&) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] ConvexGeometryPtr<Mode> materialize(CellGeometryOperationWorkspace&) const
        requires(Mode != ArithmeticMode::Exact);
    /** Existing Exact owner, with no new construction or finite-workspace admission. */
    [[nodiscard]] ConvexGeometryPtr<Mode> existing_geometry() const
        requires(Mode == ArithmeticMode::Exact)
    {
        return fallback();
    }
    /** Check native-source authority and source-size metadata before membership queries. */
    void validate_source_authority(GeometryLimits) const
        requires(Mode == ArithmeticMode::Ordinary);

private:
    friend class CellGeometryFace<Mode>;
    friend class details::NativeCellGeometrySourceAccess;
    void require_valid() const;
    const ConvexGeometryPtr<Mode>& fallback() const;
    const typename CellGeometryCatalog<Mode>::SourcePtr& native_source() const;
    std::size_t source_slot() const;
    std::shared_ptr<const CellGeometryCatalog<Mode>> catalog_;
    std::size_t slot_ = 0;
};
/** An ordered face whose enclosing geometry catalog remains pinned. */
template<ArithmeticMode Mode> class CellGeometryFace
{
public:
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] std::size_t at(std::size_t) const;
    [[nodiscard]] std::size_t operator[](std::size_t vertex) const { return at(vertex); }
    struct Iterator
    {
        using value_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        CellGeometryView<Mode> view;
        std::size_t face = 0, vertex = 0;
        value_type operator*() const { return view.ordered_face(face).at(vertex); }
        Iterator& operator++()
        {
            ++vertex;
            return *this;
        }
        void operator++(int) { ++vertex; }
        bool operator==(const Iterator&) const = default;
    };
    [[nodiscard]] Iterator begin() const { return {view_, face_, 0}; }
    [[nodiscard]] Iterator end() const { return {view_, face_, size()}; }
    [[nodiscard]] std::array<long double, 3> corner(std::size_t i) const
        requires(Mode == ArithmeticMode::Ordinary)
    {
        return view_.corner((*this)[i]);
    }
    [[nodiscard]] PlaneSupportIdentity<ArithmeticMode::Ordinary> support() const
        requires(Mode == ArithmeticMode::Ordinary)
    {
        return view_.source_support(face_);
    }
    [[nodiscard]] bool contains(const PlanarFace<ArithmeticMode::Ordinary>& face) const
        requires(Mode == ArithmeticMode::Ordinary)
    {
        return view_.face_contains(face_, face);
    }
    [[nodiscard]] CellGeometryFaceMetrics metrics() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] GeometryValue<Mode> area() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] GeometryPoint centroid() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] GeometryPoint area_vector() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] int orientation() const
        requires(Mode == ArithmeticMode::Ordinary);

private:
    friend class CellGeometryView<Mode>;
    CellGeometryFace(CellGeometryView<Mode> view, std::size_t face) : view_(std::move(view)), face_(face) {}
    CellGeometryView<Mode> view_;
    std::size_t face_;
};
/** An owning range of ordered cell faces. */
template<ArithmeticMode Mode> class CellGeometryFaces
{
public:
    CellGeometryFaces() noexcept = default;
    explicit CellGeometryFaces(CellGeometryView<Mode> view) : view_(std::move(view)) {}
    [[nodiscard]] std::size_t size() const { return view_ ? view_.num_faces() : 0; }
    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] CellGeometryFace<Mode> at(std::size_t face) const
    {
        if (face >= size())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry face range index.");
        return view_.ordered_face(face);
    }
    [[nodiscard]] CellGeometryFace<Mode> operator[](std::size_t face) const { return at(face); }
    struct Iterator
    {
        using value_type = CellGeometryFace<Mode>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        CellGeometryView<Mode> view;
        std::size_t face = 0;
        value_type operator*() const { return view.ordered_face(face); }
        Iterator& operator++()
        {
            ++face;
            return *this;
        }
        void operator++(int) { ++face; }
        bool operator==(const Iterator& other) const { return view == other.view && face == other.face; }
    };
    [[nodiscard]] Iterator begin() const { return {view_, 0}; }
    [[nodiscard]] Iterator end() const { return {view_, size()}; }

private:
    CellGeometryView<Mode> view_;
};
/** Sized, indexed owning range; every returned value independently pins it. */
template<ArithmeticMode Mode> class CellGeometryRange
{
public:
    CellGeometryRange() noexcept = default;
    explicit CellGeometryRange(std::shared_ptr<const CellGeometryCatalog<Mode>>);
    /** Own a contiguous catalog slice. Shape caps remain per cell/fan; indexing
     * preserves the slice's existing order without rebuilding its catalog. */
    CellGeometryRange(std::shared_ptr<const CellGeometryCatalog<Mode>>, std::size_t begin, std::size_t count);
    [[nodiscard]] std::size_t size() const noexcept { return catalog_ ? count_ : 0; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] CellGeometryView<Mode> at(std::size_t index) const;
    [[nodiscard]] CellGeometryView<Mode> operator[](std::size_t index) const { return at(index); }
    struct Iterator
    {
        using value_type = CellGeometryView<Mode>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        std::shared_ptr<const CellGeometryCatalog<Mode>> catalog;
        std::size_t index = 0;
        value_type operator*() const { return {catalog, index}; }
        Iterator& operator++()
        {
            ++index;
            return *this;
        }
        void operator++(int) { ++index; }
        bool operator==(const Iterator&) const = default;
    };
    [[nodiscard]] Iterator begin() const { return {catalog_, catalog_ ? first_ : 0}; }
    [[nodiscard]] Iterator end() const { return {catalog_, catalog_ ? first_ + count_ : 0}; }

private:
    std::shared_ptr<const CellGeometryCatalog<Mode>> catalog_;
    std::size_t first_ = 0, count_ = 0;
};
/** Explicit runtime policy wrapper; typed queries retain their mode contracts. */
class DynamicCellGeometryView
{
public:
    DynamicCellGeometryView() noexcept = default;
    explicit operator bool() const noexcept
    {
        return std::visit([](const auto& view) { return bool(view); }, view_);
    }
    template<ArithmeticMode Mode> explicit DynamicCellGeometryView(CellGeometryView<Mode> view) : view_(std::move(view))
    {
    }
    [[nodiscard]] ArithmeticMode arithmetic_mode() const
    {
        if (!*this)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Empty runtime cell geometry view.");
        return std::visit([](const auto& view) { return view.arithmetic_mode(); }, view_);
    }
    template<ArithmeticMode Mode> [[nodiscard]] CellGeometryView<Mode> get() const
    {
        if (const auto* value = std::get_if<CellGeometryView<Mode>>(&view_))
            return *value;
        throw GeometryError(GeometryErrorCode::InvalidInput, "Cell geometry view arithmetic mode mismatch.");
    }
    template<class Visitor> decltype(auto) visit(Visitor&& visitor) const
    {
        return std::visit(std::forward<Visitor>(visitor), view_);
    }

private:
    std::variant<CellGeometryView<ArithmeticMode::Ordinary>, CellGeometryView<ArithmeticMode::BoundedCertification>,
        CellGeometryView<ArithmeticMode::Exact>>
        view_;
};
template<ArithmeticMode Mode>
    requires(Mode != ArithmeticMode::Exact)
[[nodiscard]] ConvexGeometryPtr<Mode> materialize_cell(
    const CellGeometryView<Mode>& view, CellGeometryOperationWorkspace& workspace)
{
    return view.materialize(workspace);
}
/** Explicit ordered finite batch. Descriptor capacity and every cell's work
 * share the workspace; failed prefixes publish no vector and remain spent. */
template<ArithmeticMode Mode>
    requires(Mode != ArithmeticMode::Exact)
[[nodiscard]] std::vector<ConvexGeometryPtr<Mode>> materialize_cells(
    const CellGeometryRange<Mode>&, CellGeometryOperationWorkspace&);
template<ArithmeticMode Mode>
[[nodiscard]] std::vector<Point<Mode>> source_points(const CellGeometryRange<ArithmeticMode::Ordinary>&,
    std::span<const std::size_t>, std::span<const NativeSourcePointRequest>, CellGeometryOperationWorkspace&);
namespace details
{
GeometryValue<ArithmeticMode::Ordinary> cell_view_face_area(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t);
GeometryPoint cell_view_face_centroid(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t);
GeometryPoint cell_view_face_area_vector(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t);
int cell_view_face_orientation(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t);
bool cell_view_face_contains(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t, const PlanarFace<ArithmeticMode::Ordinary>&);
void cell_view_validate_source_authority(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, GeometryLimits);
PlaneSupportIdentity<ArithmeticMode::Ordinary> cell_view_source_support(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t);
bool cell_view_contains_source_point(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t, const PointIdentity<ArithmeticMode::Ordinary>&);
std::size_t cell_view_box_face_vertex(std::size_t, std::size_t);
std::array<long double, 3> cell_view_corner(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::size_t);
} // namespace details
// The three supported arithmetic catalogs are provided by CellGeometryView.cc.
extern template class CellGeometryCatalog<ArithmeticMode::Ordinary>;
extern template class CellGeometryCatalog<ArithmeticMode::BoundedCertification>;
extern template class CellGeometryCatalog<ArithmeticMode::Exact>;
} // namespace SimpleFluid::PLIC
#include "geometry/PLIC/CellGeometryView.ipp"
