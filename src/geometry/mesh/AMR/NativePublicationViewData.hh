/** @file NativePublicationViewData.hh
 * @brief Private persistent remote membership and scalar publication output.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveLeafLayout.hh"
#include "geometry/mesh/AMR/NativePublicationProof.hh"
#include "geometry/mesh/AMR/NativePublicationView.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"
#include <optional>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeRemoteKey final
{
public:
    [[nodiscard]] const AdaptiveCellKey& key() const noexcept { return key_; }

private:
    friend class NativePublicationSession;
    friend class NativeRemoteKeyCatalog;
    friend class NativePublicationOutputGraph;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    AdaptiveCellKey key_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
class SIMPLEFLUID_LOCAL NativeRemoteKeyCatalog
{
    struct Row
    {
        std::shared_ptr<const NativeRemoteKey> binding;
        [[nodiscard]] const AdaptiveCellKey& key() const noexcept { return binding->key(); }
    };
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const { return a.key() < b.key(); }
    };
    using Pages = persistent_ordered_detail::OrderedPages<Row, Less>;

public:
    NativeRemoteKeyCatalog() = default;
    [[nodiscard]] static std::shared_ptr<const NativeRemoteKey> capture(
        const AdaptiveCellKey&, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] NativeRemoteKeyCatalog with_key(
        const AdaptiveCellKey&, const geometry_detail::NativeConstructionAdmission&) const;
    [[nodiscard]] NativeRemoteKeyCatalog without_key(
        const AdaptiveCellKey&, const geometry_detail::NativeConstructionAdmission&) const;
    [[nodiscard]] std::size_t size() const noexcept { return root_ ? root_->size : 0; }
    [[nodiscard]] const AdaptiveCellKey& at_ordinal(std::size_t) const;
    [[nodiscard]] std::optional<std::size_t> ordinal_of(const AdaptiveCellKey&) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return root_ ? root_->subtree_bytes : 0; }

private:
    friend class NativePublicationSession;
    friend class NativePublicationOutputGraph;
    [[nodiscard]] NativeRemoteKeyCatalog edited(const AdaptiveCellKey&, std::shared_ptr<const NativeRemoteKey>,
        std::shared_ptr<Budget>, persistent_ordered_detail::OrderedPageWork* = nullptr,
        std::shared_ptr<const geometry_detail::NativeAllocationState> = {}) const;
    Pages::Ptr root_;
};

using NativePublicationKeyCatalog = NativeRemoteKeyCatalog;
class NativePublicationNumbering;
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
namespace SimpleFluid::Meshes { class NativeSparseFragments; struct NativeSparseNumericIndex; }
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct NativePublicationStorageSummary
{
    // The current maintained rich owner total includes geometry, keys,
    // topology and administration. It is reported together as indexing;
    // physical_geometry/fixed are not separate measured-zero claims.
    // library_upper is an explicit library profile, outside actual C++ output.
    std::size_t metadata = 0, physical_geometry = 0, scalar_numeric = 0;
    std::size_t library_upper = 0, fixed = 0;
    [[nodiscard]] std::size_t retained_bytes() const;
    [[nodiscard]] MeshStorageReport report() const noexcept;
};
/** Current scalar routes. Rich key owners are retained by the catalogs. */
struct SIMPLEFLUID_LOCAL NativePublicationKeyRoute
{
    std::uint64_t gid = 0;
    std::size_t ordinal = 0;
    bool remote = false;
};
/** Native-only payload. Never owns layouts, source indexes or controllers. */
struct SIMPLEFLUID_LOCAL NativePublicationPayload
{
    NativePublicationKeyCatalog owned, remote;
    std::uint64_t cell_begin = 0;
    geometry_detail::NativeConstructorStorageOwner route_storage;
    std::vector<NativePublicationKeyRoute> keys;
    std::shared_ptr<const CompositePartition<>> partition;
    std::shared_ptr<const MeshHandle<>> mesh;
    std::shared_ptr<const NativePublicationNumbering> numbering;
    std::shared_ptr<const NativeSparseFragments> fragments;
    std::shared_ptr<const NativeSparseNumericIndex> numeric;
    NativePublicationStorageSummary summary;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    [[nodiscard]] const AdaptiveCellKey& key(const NativePublicationKeyRoute&) const;
};
struct SIMPLEFLUID_LOCAL NativePublicationViewData
{
    std::shared_ptr<NativePublicationPayload> payload;
    std::shared_ptr<const AdaptiveNativePublication> compatibility;
    std::shared_ptr<const NativePublicationRoot> proof_root;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    std::size_t compatibility_resident_bytes = 0;
    [[nodiscard]] std::size_t resident_bytes() const
    { return compatibility ? compatibility_resident_bytes : (payload ? payload->summary.retained_bytes() : 0); }
    [[nodiscard]] const AdaptiveCellKey& key(const NativePublicationKeyRoute& route) const;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
