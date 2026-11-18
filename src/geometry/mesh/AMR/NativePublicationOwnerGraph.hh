/** @file NativePublicationOwnerGraph.hh
 * @brief First/last native fixed, scalar-state and strong schema projections.
 */
#pragma once
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/details/NativeSchemaOwner.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
namespace SimpleFluid::Meshes
{
struct NativeFragmentDescriptor;
class NativeFragmentRecord;
}
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativeSourceSubsetCoverage;
class SIMPLEFLUID_LOCAL NativePublicationOwnerGraph
{
public:
    /** Borrowed callback context. Closed receipt hooks are installed only when
     * this graph belongs to the operation's live admission domain. */
    struct Context
    {
        PLIC::details::GeometryOwnerGraphEditor& graph;
        bool retain, prepaid;
        PLIC::details::NativeGeometryAccounting* accounting;
        geometry_detail::NativeConstructorOwnerSink sink;
        Context(PLIC::details::GeometryOwnerGraphEditor&, bool, bool,
            PLIC::details::NativeGeometryAccounting* = nullptr);
        static void identity(void*, const PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>&);
        static void geometry(void*, const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
        static void coverage(void*, const std::shared_ptr<const NativeSourceSubsetCoverage>&);
    };
    static void descriptor(const std::shared_ptr<const NativeFragmentDescriptor>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, bool prepaid = false,
        PLIC::details::NativeGeometryAccounting* = nullptr);
    static void fragment(const std::shared_ptr<const NativeFragmentRecord>&, PLIC::details::GeometryOwnerGraphEditor&,
        bool retain, bool prepaid, PLIC::details::NativeGeometryAccounting* = nullptr);
    static void validate_state(const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    static void state(const std::shared_ptr<const geometry_detail::NativeAllocationState>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, bool prepaid);
    /** Measured fixed-allocation vertex. This never locks/retains the target or
     * traverses the target's strong-lifetime dynamic children.
     */
    static void fixed(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, bool prepaid);
    /** A zero-payload semantic anchor gates strong-lifetime children separately
     * from the fixed envelope. A weak/fixed root cannot suppress this traversal.
     */
    static void schema(const std::shared_ptr<const geometry_detail::NativeSchemaOwner>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, bool prepaid);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
