/** @file NativePublicationProof.hh
 * @brief Detached exact source selectors and privately issued publication roots.
 */
#pragma once
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include <array>
#include <memory>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativePublicationSession;
class NativeSparseAccess;
struct NativePublicationIndexes;
struct NativePublicationPayload;
class NativePublicationOutputGraph;

/** An exact detached selector, not a validation capability. It owns only this
 * source row's immutable dependencies, never an enclosing source layout/page.
 */
class SIMPLEFLUID_LOCAL NativeSourceBinding final
{
    class ConstructionKey
    {
        friend class NativeSourceBinding;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
        ConstructionKey(ConstructionKey&&) = default;
    };

public:
    // Public only to permit std::allocate_shared to call it; the argument type
    // and its construction remain private; callers use the admitted capture.
    NativeSourceBinding(
        ConstructionKey, geometry_detail::NativeConstructionAdmission);
    [[nodiscard]] static std::shared_ptr<const NativeSourceBinding> capture(
        const AdaptiveLeaf&, const geometry_detail::NativeConstructionAdmission&);
    /** Convenience for standalone selector tests. Sessions share one existing
     * admission domain rather than creating one per selector.
     */
    [[nodiscard]] static std::shared_ptr<const NativeSourceBinding> capture(
        const AdaptiveLeaf&, std::shared_ptr<Budget>);
    [[nodiscard]] bool matches(const AdaptiveLeaf&) const noexcept;
    [[nodiscard]] const AdaptiveLeaf& source() const noexcept { return source_; }
    [[nodiscard]] std::size_t exclusive_bytes() const noexcept { return payload_; }
    [[nodiscard]] std::size_t admission_bytes() const noexcept { return control_allocation_ + charge_; }
    static void graph(const std::shared_ptr<const NativeSourceBinding>&, PLIC::details::GeometryOwnerGraphEditor&,
        bool retain, bool prepaid = false, PLIC::details::NativeGeometryAccounting* = nullptr);
    ~NativeSourceBinding();
    NativeSourceBinding(const NativeSourceBinding&) = delete;
    NativeSourceBinding& operator=(const NativeSourceBinding&) = delete;

private:
    // Destroy the copied variable buffers and owner pins before releasing their
    // operation charge. No layout or cache controller is held here.
    geometry_detail::NativeConstructionAdmission admission_;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
    std::size_t charge_ = 0, payload_ = 0, control_allocation_ = 0;
    void graph_children(PLIC::details::GeometryOwnerGraphEditor&, bool retain,
        PLIC::details::NativeGeometryAccounting*) const;
    AdaptiveLeaf source_;
    std::array<std::shared_ptr<const AdaptiveBytes>, 3> factor_parts_;
    NativeFragmentDescriptor::Kind recipe_kind_ = NativeFragmentDescriptor::Kind::Cartesian;
    NativeFragmentBytes recipe_view_;
    std::vector<std::shared_ptr<const NativeFragmentBytes>> recipe_parts_;
};

/** Owner-issued immutable source instance; fields on the wire are not proof.
 * Only the private exchange can accept a ticket after a current owner confirms
 * the exact outstanding key/revision/transaction and retained proof instance.
 */
struct NativeSourceTicket
{
    int issuer = -1;
    std::uint64_t transaction = 0, ordinal = 0;
    auto operator<=>(const NativeSourceTicket&) const = default;
};

/** Native result subset only; source/cache stores are separate roots. */
class SIMPLEFLUID_LOCAL NativePublicationRoot final
{
public:
    [[nodiscard]] const PLIC::details::OwnerManifestPair& owners() const noexcept { return owners_; }
    [[nodiscard]] const std::shared_ptr<const NativePublicationPayload>& payload() const noexcept { return payload_; }
    [[nodiscard]] const auto& allocation() const noexcept { return allocation_; }
    [[nodiscard]] const auto& paid_receipts() const noexcept { return paid_; }
private:
    friend class NativePublicationSession;
    friend class NativePublicationOperation;
    friend class NativePublicationOutputGraph;
    std::shared_ptr<const NativePublicationPayload> payload_;
    PLIC::details::OwnerManifestPair owners_;
    PLIC::details::NativeOwnerReceipts paid_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> view_allocation_;
};

/** The sole native root issuer recognized by the Stage 3 physical union. The
 * global union remains in the hierarchy cache; this store never retains an old
 * union containing unrelated retired snapshot roots.
 */
class SIMPLEFLUID_LOCAL NativeGeometryRootStore final
{
public:
    NativeGeometryRootStore() = default;
    [[nodiscard]] const std::shared_ptr<const NativePublicationRoot>& root() const noexcept { return root_; }
    [[nodiscard]] const PLIC::details::GeometryOwnerRootBinding& binding() const noexcept { return binding_; }
    void certify(const PLIC::details::GeometryOwnerUnion&, PLIC::GeometryAllocationLedger&,
        PLIC::GeometryAllocationLedger&, std::size_t& used, std::size_t limit) const;

private:
    friend class NativePublicationSession;
    friend class NativePublicationOperation;
    /** Pin the completed output and its exact binding for the union edit. */
    [[nodiscard]] static NativeGeometryRootStore staged(std::shared_ptr<const NativePublicationRoot>);
    void swap(NativeGeometryRootStore&) noexcept;
    std::shared_ptr<const NativePublicationRoot> root_;
    PLIC::details::GeometryOwnerRootBinding binding_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
