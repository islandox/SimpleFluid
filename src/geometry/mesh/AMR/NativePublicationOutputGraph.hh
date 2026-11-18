/** @file NativePublicationOutputGraph.hh
 * @brief Closed native result subset projection and maintained output summary.
 */
#pragma once
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/mesh/AMR/NativePublicationNumbering.hh"
#include "geometry/mesh/AMR/NativePublicationViewData.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativePublicationOutputGraph final
{
public:
    static void catalog(const NativeRemoteKeyCatalog&, PLIC::details::GeometryOwnerGraphEditor&,
        bool retain, PLIC::details::NativeGeometryAccounting*);
    static void numbering(const std::shared_ptr<const NativePublicationNumbering>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, PLIC::details::NativeGeometryAccounting*);
    static void payload(const std::shared_ptr<const NativePublicationPayload>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, PLIC::details::NativeGeometryAccounting*);
    [[nodiscard]] static std::shared_ptr<const NativePublicationRoot> make(
        std::shared_ptr<NativePublicationPayload>, std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> view,
        const std::shared_ptr<const NativePublicationRoot>& previous, PLIC::details::NativeGeometryAccounting&);
public:
    // Closed typed subset visitors used by the native geometry/partition factories.
    static void fragments(const std::shared_ptr<const NativeSparseFragments>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool, PLIC::details::NativeGeometryAccounting*);
    static void numeric(const std::shared_ptr<const NativeSparseNumericIndex>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool, PLIC::details::NativeGeometryAccounting*);
private:
    template<class Owner, class Vector> static void vector(const std::shared_ptr<const Owner>& owner,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& allocation, const Vector& values,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, PLIC::details::NativeGeometryAccounting* accounting)
    {
        if (!values.capacity()) return;
        if (!allocation || !allocation->matches(owner) || values.get_allocator().state() != allocation->state() ||
            values.capacity() > SIZE_MAX / sizeof(typename Vector::value_type))
            throw std::invalid_argument("Native output vector differs from its immutable target/allocator receipt.");
        const auto bytes = values.capacity() * sizeof(typename Vector::value_type);
        const auto alias = std::shared_ptr<const void>(owner, values.data());
        if (accounting)
            accounting->authorize_buffer(PLIC::details::NativeActualBufferProof(owner, allocation,
                values.get_allocator().state(), {values.data(), bytes}));
        graph.external(alias, bytes, retain, bool(accounting));
        graph.native_state(values.get_allocator().state(), retain, bool(accounting));
    }

};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
