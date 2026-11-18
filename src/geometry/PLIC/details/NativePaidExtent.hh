/** Closed per-physical-owner mirror credits; never derived from Counter.used. */
#pragma once
#include "geometry/PLIC/details/NativeWeakOwnerObservation.hh"
#include "geometry/details/NativeConstructorStorage.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
#include <cstdint>
#include <span>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativeSourceCoverage;
class NativeSourceOwnerStore;
class NativeContactFundingAccess;
class NativePublicationOperation;
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
namespace SimpleFluid::PLIC::details
{
class NativeActualBufferProof;
class NativeGeometryAccounting;
class GeometryAllocationOwnerAccess;
class GeometryOwnerGraphEditor;
namespace geometry_demand
{
struct Admission;
struct MetadataSink;
struct NativeAdministrationSink;
class OwnerEdits;
struct OwnerRecord;
struct PageWork;
} // namespace geometry_demand
/** Separate diagnostic seam: concrete issuers expose actual typed fields;
 * Census never calls the production owner projector. */
namespace retention { struct BorrowedRoot; }
struct NativeFinancialInspectionSink
{
    void* context;
    void (*owner)(void*, std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes);
    void (*fixed)(void*, const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    void (*storage)(void*, const geometry_detail::NativeConstructorStorageOwner&);
    void (*state)(void*, const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    void (*root)(void*, const retention::BorrowedRoot&) = nullptr;
    void (*weak)(void*, const NativeWeakOwnerObservation&) = nullptr;
    void weak_observation(const NativeWeakOwnerObservation& observation) const
    {
        if (!observation) return;
        if (!weak) throw std::invalid_argument("Native financial inspection lacks fixed observation projection.");
        weak(context, observation);
    }
};
class NativePaidExtentCharge
{
public:
    NativePaidExtentCharge() = default;
    [[nodiscard]] bool valid() const;
    [[nodiscard]] bool matches(std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes) const;
    [[nodiscard]] std::shared_ptr<const void> owner() const noexcept { return owner_.lock(); }
    [[nodiscard]] std::shared_ptr<const void> origin() const noexcept { return origin_.lock(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::uint8_t kind() const noexcept { return kind_; }
    [[nodiscard]] const void* owner_address() const noexcept { return address_; }
    [[nodiscard]] const void* origin_address() const noexcept { return origin_address_; }
    [[nodiscard]] const void* allocation_state_address() const noexcept { return state_address_; }
    [[nodiscard]] bool source_only() const noexcept { return proof_ == 4; }
    [[nodiscard]] bool direct_h() const noexcept { return direct_h_; }
    [[nodiscard]] bool publication_h() const noexcept { return publication_h_; }
    [[nodiscard]] bool borrowed_financial() const noexcept { return borrowed_financial_; }
    [[nodiscard]] const void* financial_address() const noexcept { return financial_address_; }
    [[nodiscard]] bool same_financial_control(const NativePaidExtentCharge&) const noexcept;
    [[nodiscard]] bool same_physical_origin(const NativePaidExtentCharge& other) const noexcept
    { return same_owner_fact(other) && same_origin_fact(other); }
    [[nodiscard]] const NativeWeakOwnerObservation& physical_observation() const noexcept
    {
        return physical_fixed_;
    }
    /** Build a fixed/scalar witness only for a selected, still-live detached
     * source lease. Other receipt rows remain nonowning observations. */
    [[nodiscard]] NativeWeakOwnerObservation detached_source_observation() const;
    [[nodiscard]] const NativeWeakOwnerObservation& financial_observation() const noexcept
    {
        return financial_fixed_;
    }
    using FinancialProjector = void (*)(const std::shared_ptr<const void>&,
        const geometry_detail::NativeConstructorOwnerSink&);
    using FinancialInspector = void (*)(const std::shared_ptr<const void>&, const NativeFinancialInspectionSink&);
    using FinancialValidator = bool (*)(const std::shared_ptr<const void>& financial,
        const std::shared_ptr<const void>& physical, const std::shared_ptr<const void>& arena,
        std::uint8_t kind, std::size_t bytes, std::size_t snapshot, std::uint64_t revision);
    [[nodiscard]] bool has_financial_pin() const noexcept { return bool(lease_pin_); }
    [[nodiscard]] bool has_lifetime_children() const noexcept
    {
        return lease_pin_ || physical_fixed_ || financial_fixed_;
    }
    void project_financial(const geometry_detail::NativeConstructorOwnerSink&) const;
    void inspect_financial(const NativeFinancialInspectionSink&) const;
    [[nodiscard]] static NativePaidExtentCharge fixed(
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    [[nodiscard]] static NativePaidExtentCharge state(
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    [[nodiscard]] static NativePaidExtentCharge counter(
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    [[nodiscard]] static NativePaidExtentCharge mirror(
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    [[nodiscard]] static NativePaidExtentCharge storage(const geometry_detail::NativeConstructorStorageOwner&);
    [[nodiscard]] static NativePaidExtentCharge metadata(const std::shared_ptr<const void>&, std::size_t bytes);
    [[nodiscard]] static NativePaidExtentCharge buffer(const NativeActualBufferProof&);

private:
    friend class NativeGeometryAccounting;
    friend class GeometryAllocationOwnerAccess;
    friend class GeometryOwnerGraphEditor;
    friend class NativeOwnerReceipts;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeContactFundingAccess;
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativePublicationOperation;
    [[nodiscard]] static NativePaidExtentCharge retained_profile(
        const std::shared_ptr<const void>&, const std::shared_ptr<const geometry_detail::NativeRetainedExtent>&,
        const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&);
    // SourceStore verifies this exact immutable qualified lease token against
    // its canonical body/arena ledger before issuing the receipt.
    [[nodiscard]] static NativePaidExtentCharge source_profile(std::uint8_t kind,
        const std::shared_ptr<const void>& owner, std::size_t bytes, const std::shared_ptr<const void>& actual_arena,
        const std::shared_ptr<const void>& immutable_lease, std::size_t charge_snapshot, std::uint64_t revision, FinancialProjector, FinancialInspector, FinancialValidator);
    [[nodiscard]] static NativePaidExtentCharge detached_subset(std::uint8_t kind,
        const std::shared_ptr<const void>& owner, std::size_t bytes, const std::shared_ptr<const void>& actual_arena,
        const std::shared_ptr<const void>& detached_subset_lease, std::size_t charge_snapshot, std::uint64_t revision, FinancialProjector, FinancialInspector, FinancialValidator);
    [[nodiscard]] static NativePaidExtentCharge intrinsic(std::uint8_t kind,
        const std::shared_ptr<const void>& owner, std::size_t bytes, const std::shared_ptr<const void>& actual_arena,
        const std::shared_ptr<const void>& financial_owner, std::size_t snapshot, std::uint64_t revision,
        FinancialProjector, FinancialInspector, FinancialValidator,
        const NativeWeakOwnerObservation& physical = {}, const NativeWeakOwnerObservation& financial = {});
    [[nodiscard]] static NativePaidExtentCharge contact_fixed(const NativeWeakOwnerObservation&);
    [[nodiscard]] NativePaidExtentCharge with_weak_observations(
        const NativeWeakOwnerObservation& physical, const NativeWeakOwnerObservation& financial) const;
    [[nodiscard]] NativePaidExtentCharge as_borrowed_financial() const;
    [[nodiscard]] bool same_owner_fact(const NativePaidExtentCharge&) const noexcept;
    [[nodiscard]] bool same_origin_fact(const NativePaidExtentCharge&) const noexcept;
    [[nodiscard]] bool same_receipt_fact(const NativePaidExtentCharge&) const noexcept;
    static NativePaidExtentCharge made(std::uint8_t, const std::shared_ptr<const void>&, std::size_t,
        const std::shared_ptr<const geometry_detail::NativeAllocationState>&);
    const void* address_ = nullptr;
    const void* origin_address_ = nullptr;
    std::weak_ptr<const void> owner_, origin_;
    std::shared_ptr<const void> lease_pin_;
    std::weak_ptr<const void> financial_observation_;
    NativeWeakOwnerObservation physical_fixed_, financial_fixed_;
    const void* financial_address_ = nullptr;
    bool direct_h_ = true, borrowed_financial_ = false, publication_h_ = false;
    FinancialProjector financial_projector_ = nullptr;
    FinancialInspector financial_inspector_ = nullptr;
    FinancialValidator financial_validator_ = nullptr;
    // Identity fact only, never dereferenced or retained. The actual allocator
    // and owning object keep their domain alive; a proof creates no extra pin.
    const void* state_address_ = nullptr;
    std::weak_ptr<const geometry_detail::NativeRetainedExtent> profile_;
    std::weak_ptr<const geometry_detail::NativeConstructorStorage> storage_;
    std::size_t bytes_ = 0, storage_bytes_ = 0, storage_revision_ = 0, charge_snapshot_ = 0;
    std::uint64_t revision_ = 0;
    std::uint8_t kind_ = 0, proof_ = 0;
    enum class FixedOwner : std::uint8_t { None, Envelope, State, Counter, Mirror };
    FixedOwner fixed_owner_ = FixedOwner::None;
};
/** Closed lazy source-coverage certificate. Source roots may own it;
 * native outputs retain detached subset receipts, never this membership root. */
class NativePaidExtentGroup
{
public:
    NativePaidExtentGroup() noexcept = default;
    struct MemberSink
    {
        void* context;
        void (*visit)(
            void*, std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes, bool retain);
    };
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool contains(std::uint8_t, const std::shared_ptr<const void>&, std::size_t) const;
    [[nodiscard]] bool same_membership(const NativePaidExtentGroup&) const noexcept;
    void visit_delta(const NativePaidExtentGroup&, MemberSink) const;
    [[nodiscard]] const std::shared_ptr<const void>& origin() const noexcept { return origin_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    friend class SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceOwnerStore;
    using Membership = SimpleFluid::Meshes::AMR::adaptive_detail::NativeSourceCoverage;
    NativePaidExtentGroup(std::shared_ptr<const Membership> membership_root,
        std::shared_ptr<const void> detached_coverage,
        std::shared_ptr<const void> actual_arena, std::size_t covered_bytes, std::uint64_t revision,
        bool (*contains)(const Membership*, std::uint8_t, const std::shared_ptr<const void>&, std::size_t),
        bool (*same_membership)(const Membership*, const Membership*) noexcept,
        void (*visit_delta)(const Membership* previous, const Membership* next, MemberSink))
        : membership_(std::move(membership_root)), coverage_(std::move(detached_coverage)),
          origin_(std::move(actual_arena)), bytes_(covered_bytes), revision_(revision), contains_(contains),
          same_(same_membership), delta_(visit_delta)
    {
    }
    std::shared_ptr<const Membership> membership_;
    std::shared_ptr<const void> coverage_, origin_;
    std::size_t bytes_ = 0;
    std::uint64_t revision_ = 0;
    bool (*contains_)(const Membership*, std::uint8_t, const std::shared_ptr<const void>&, std::size_t) = nullptr;
    bool (*same_)(const Membership*, const Membership*) noexcept = nullptr;
    void (*delta_)(const Membership*, const Membership*, MemberSink) = nullptr;
};
/** Persistent native-only receipt rows. Default value allocates nothing.
 * Physical-owner and arena controls are weak. Financial pins retain only
 * closed, selected physical funding holders, and are real leaf children.
 * They never retain a source membership/controller to fund a subset. */
class NativeOwnerReceipts
{
public:
    enum class Action : std::uint8_t { Record, Retain, Retire, RetireIfRecorded };
    struct Edit
    {
        const NativePaidExtentCharge* charge = nullptr;
        Action action = Action::Record;
        std::size_t sequence = 0;
    };
    NativeOwnerReceipts() = default;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool same_root(const NativeOwnerReceipts&) const noexcept;
    [[nodiscard]] bool contains(const NativePaidExtentCharge&) const;
    [[nodiscard]] bool contains_any(const NativePaidExtentCharge&) const;
    [[nodiscard]] bool contains_direct_h(const NativePaidExtentCharge&) const;
    [[nodiscard]] bool contains_publication_h(const NativePaidExtentCharge&) const;
    [[nodiscard]] bool source_only() const noexcept;
    [[nodiscard]] NativeOwnerReceipts recorded(const NativePaidExtentCharge&, geometry_demand::Admission) const;
    [[nodiscard]] NativeOwnerReceipts retained(const NativePaidExtentCharge&, geometry_demand::Admission) const;
    [[nodiscard]] NativeOwnerReceipts retired(const NativePaidExtentCharge&, geometry_demand::Admission) const;
    /** Stage checked references in input order for each exact receipt, then
     * copy the affected ordered pages once. Scratch is sorted in place.
     * Charge references must remain valid through the edit and its caller's
     * subsequent consumption of the sorted batch. */
    [[nodiscard]] NativeOwnerReceipts edited(std::span<Edit>, geometry_demand::Admission,
        std::span<const geometry_demand::OwnerRecord> owner_changes = {}) const;
    struct Sink
    {
        void* context;
        void (*visit)(void*, const NativePaidExtentCharge&, std::size_t references);
    };
    void visit_owner(std::uint8_t kind, const std::shared_ptr<const void>&, Sink) const;
    void visit(Sink) const;
    void visit_added(const NativeOwnerReceipts&, Sink, geometry_demand::PageWork* = nullptr) const;
    void visit_administration(geometry_demand::MetadataSink) const;
    void visit_native_administration_delta(const NativeOwnerReceipts&, geometry_demand::NativeAdministrationSink,
        geometry_demand::PageWork* = nullptr) const;
    [[nodiscard]] std::size_t metadata_admission_bytes() const noexcept;

private:
    friend class GeometryOwnerGraphEditor;
    [[nodiscard]] NativeOwnerReceipts edited_graph(std::span<Edit>, geometry_demand::Admission,
        const geometry_demand::OwnerEdits&) const;
    template<class Owners, class Find>
    [[nodiscard]] NativeOwnerReceipts edited_with(std::span<Edit>, geometry_demand::Admission,
        const Owners& owner_changes, const Find&) const;
    enum class Category { Any, DirectH, PublicationH };
    [[nodiscard]] bool contains_category(const NativePaidExtentCharge&, Category) const;
    struct State;
    std::shared_ptr<const State> state_;
};
class NativePaidOriginTotals
{
public:
    struct Assignment
    {
        std::shared_ptr<const void> arena;
        std::size_t bytes = 0;
    };
    [[nodiscard]] std::size_t bytes(const std::shared_ptr<const void>& actual_arena) const;
    [[nodiscard]] NativePaidOriginTotals edited(std::span<const Assignment>, geometry_demand::Admission) const;
    void visit_administration(geometry_demand::MetadataSink) const;
    void visit_native_administration_delta(const NativePaidOriginTotals&, geometry_demand::NativeAdministrationSink,
        geometry_demand::PageWork* = nullptr) const;
    [[nodiscard]] std::size_t metadata_admission_bytes() const noexcept;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace SimpleFluid::PLIC::details
