/** @file GeometryOwnerGraph.hh
 * @brief Private first/last physical dependency editing for immutable owner graphs.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include <array>
#include <exception>
#include <map>
#include <span>
#include <vector>
namespace SimpleFluid::PLIC::details
{
struct GeometryOwnerGraphWork
{
    std::size_t dependency_entries = 0, edge_visits = 0;
    std::size_t first_transitions = 0, last_transitions = 0;
    geometry_demand::PageWork index, lookup;
};
/** Borrowed mutation context. Only the returned manifest retains physical owners.
 * Add replacement roots before retiring accepted roots to prune shared frontiers.
 * The caller publishes the completed manifest after every throwing edit succeeds.
 * Accepted values must be complete physical dependency graphs emitted by this
 * closed typed protocol, with exact incoming multiplicities. A flat allocation
 * census/export or arbitrary external-owner admission does not certify closure.
 */
class GeometryOwnerGraphEditor
{
public:
    GeometryOwnerGraphEditor(geometry_demand::GeometryOwnerManifest, std::uint8_t channel, std::size_t& used,
        std::size_t limit, GeometryAllocationLedger* = nullptr, GeometryOwnerGraphWork* = nullptr);
    ~GeometryOwnerGraphEditor();
    GeometryOwnerGraphEditor(const GeometryOwnerGraphEditor&) = delete;
    GeometryOwnerGraphEditor& operator=(const GeometryOwnerGraphEditor&) = delete;
    GeometryOwnerGraphEditor(GeometryOwnerGraphEditor&&) = delete;
    GeometryOwnerGraphEditor& operator=(GeometryOwnerGraphEditor&&) = delete;

    [[nodiscard]] static std::uint8_t external_kind() noexcept;
    [[nodiscard]] static std::size_t edit_byte_bound(
        const geometry_demand::GeometryOwnerManifest&, std::size_t incoming_edges, std::size_t changed_keys);
    /** One actual shared pointer edge. The result gates child traversal. */
    bool external(const std::shared_ptr<const void>&, std::size_t bytes, bool retain, bool prepaid = false);
    /** Admit the one physical shared pointer held by a manifest record.
     * Stored payload reference multiplicity describes its graph, not this edge. */
    bool owner(const geometry_demand::OwnerRecord&, bool retain, bool prepaid = false);
    template<ArithmeticMode M> void geometry(const ConvexGeometryPtr<M>&, bool retain);
    template<ArithmeticMode M>
    void catalog(const std::shared_ptr<const CellGeometryCatalog<M>>&, bool retain, bool metadata_prepaid = false);
    /** Construction receipt only, without an incoming graph edge or child
     * closure assertion. The later catalog walk still descends on graph first. */
    template<ArithmeticMode M>
    void prepay_catalog_page(const std::shared_ptr<const typename CellGeometryCatalog<M>::ExplicitPage>&);
    void cartesian(const std::shared_ptr<const CartesianCellSource>&, bool retain);
    void planar(const PlanarFacePtr<ArithmeticMode::Ordinary>&, bool retain);
    [[nodiscard]] geometry_demand::GeometryOwnerManifest finish();

private:
    // Bounded mutation scratch belongs to the caller's stack, never heap Impl.
    // Larger changes use charged maps/vectors without changing the protocol.
    struct PendingOwner
    {
        std::uint8_t kind = 0;
        std::shared_ptr<const void> owner;
    };
    std::array<geometry_demand::OwnerRecord, geometry_demand::leaf_capacity> rows_{};
    std::array<PendingOwner, geometry_demand::leaf_capacity> pending_{};
    struct Impl
    {
        geometry_demand::GeometryOwnerManifest result;
        std::uint8_t channel;
        std::size_t& used;
        std::size_t limit;
        GeometryAllocationLedger* ledger;
        GeometryOwnerGraphWork* work;
        std::span<geometry_demand::OwnerRecord> rows;
        std::span<PendingOwner> inline_pending;
        std::size_t row_count = 0, pending_count = 0;
        std::vector<PendingOwner> pending;
        struct KeyLess
        {
            bool operator()(const geometry_demand::OwnerKey&, const geometry_demand::OwnerKey&) const noexcept;
        };
        std::map<geometry_demand::OwnerKey, geometry_demand::OwnerRecord, KeyLess> overlay;
        bool adding = true, failed = false, finished = false;
        std::size_t last_counter = 0;
        int construction_exceptions = std::uncaught_exceptions();
        struct Call
        {
            Impl& editor;
            int exceptions = std::uncaught_exceptions();
            explicit Call(Impl&);
            ~Call();
        };
        geometry_demand::Admission admission();
        template<class Kind>
        bool edge(Kind, const std::shared_ptr<const void>&, std::size_t bytes, bool retain, bool prepaid = false);
        template<class Kind, class T> void push(Kind, const std::shared_ptr<const T>&, std::size_t&, std::size_t);
        [[nodiscard]] static bool source_base_children(bool transition) noexcept { return transition; }
        template<GeometryIdentityKind Kind, ArithmeticMode Mode>
        void retain_identity(const GeometryIdentity<Kind, Mode>&, std::size_t&, std::size_t);
        void retain_authorities(const std::vector<PlaneAuthority>&, std::size_t&, std::size_t);
        template<ArithmeticMode Mode> void drain(bool retain);
    } state_;
    Impl* impl_ = &state_;
};
} // namespace SimpleFluid::PLIC::details
