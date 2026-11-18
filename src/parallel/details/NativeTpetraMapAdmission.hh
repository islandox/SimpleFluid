/** @file NativeTpetraMapAdmission.hh
 * @brief Explicit finite map profiles and collectively staged actual map owners.
 */
#pragma once
#include "dataclass/TpetraTypes.hh"
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/details/NativeAllocationEnvelope.hh"
#include <array>
#include <span>
namespace SimpleFluid::geometry_detail
{
class NativeMapDelete;
class SIMPLEFLUID_LOCAL NativeTpetraMapAdmission final
{
public:
    using Pack = DefaultTpetraTypes;
    using Map = Pack::map_type;
    using GO = Pack::global_ordinal_type;
    enum class Kind : unsigned char
    {
        ContiguousOwned,
        Noncontiguous
    };
    struct Plan
    {
        Kind kind = Kind::Noncontiguous;
        std::size_t n = 0, m = 0, k = 0, b = 0;
        std::size_t map_object = 0, directory_object = 0, rcp_node = 0;
        std::size_t gid_data = 0, hash_offsets = 0, hash_pairs = 0;
        std::size_t retained_kokkos_headers = 0, retained_kokkos_records = 0, retained_labels = 0;
        std::size_t retained_teuchos_bookkeeping = 0;
        std::size_t temporary_gid_copy = 0, temporary_noncontiguous_gids = 0, temporary_hash_counts = 0;
        std::size_t temporary_minmax = 0, temporary_kokkos_headers_records_labels = 0;
        std::size_t temporary_comm_helpers = 0, temporary_profiling_helpers = 0;
        std::size_t native_receipts = 0;
        std::uint64_t fingerprint = 0;
        bool host_mirror_alias = true;
        [[nodiscard]] std::size_t retained_upper() const;
        [[nodiscard]] std::size_t scratch_upper() const;
    };
    struct Owner
    {
        Teuchos::RCP<const Map> map;
        Plan plan;
    };
    [[nodiscard]] static Plan plan(Kind, std::span<const GO>);
    /** All native allocation/admission happens in prepare(), under the caller's
     * local collective failure gate. construct() enters the library only after
     * every rank has completed prepare(). */
    class Prepared;
    [[nodiscard]] static std::shared_ptr<Prepared> prepare(const Plan&, const NativeConstructionAdmission&,
        const std::shared_ptr<Prepared>& same_batch = {});
    [[nodiscard]] static Owner construct(
        Prepared&, std::span<const GO>, Tpetra::global_size_t, const Teuchos::RCP<const Teuchos::Comm<int>>&);
    static void visit(const Teuchos::RCP<const Map>&, PLIC::details::GeometryOwnerGraphEditor&, bool,
        PLIC::details::NativeGeometryAccounting*);
    [[nodiscard]] static bool admits(const Teuchos::RCP<const Map>&);
    /** Weak result receipts retain the actual RCP deallocator/control tail.
     * They do not promote the map or retain its conditional live body profile. */
    static void visit_weak(const Teuchos::RCP<const Map>&, PLIC::details::GeometryOwnerGraphEditor&, bool,
        PLIC::details::NativeGeometryAccounting*);
    [[nodiscard]] static std::size_t retained_upper(const Teuchos::RCP<const Map>&);
    [[nodiscard]] static std::size_t library_context_bytes() noexcept;
    /** Import the exact active process-runtime contexts when they are already
     * live, including those started by earlier, unrelated source hierarchies. */
    static void capture_context_prefix(PLIC::details::SealedSourcePrefix&, const std::shared_ptr<const void>& arena);
    /** Allocation-free qualification query. Unsupported profiles use the public
     * complete verifier rather than changing factory semantics. */
    [[nodiscard]] static bool profile_supported() noexcept;
    /** Preallocation profile validation; rejects unsupported runtime knobs. */
    static void require_profile();

private:
    static void visit_control(const NativeMapDelete&, PLIC::details::GeometryOwnerGraphEditor&, bool,
        PLIC::details::NativeGeometryAccounting*);
};
} // namespace SimpleFluid::geometry_detail
