/** @file PointSnapshot.hh
 * @brief Frozen mode-typed provenance tables; geometry and construction owners
 * are deliberately absent from this ownership graph.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/OriginalSupportAuthority.hh"
#include <algorithm>
#include <exception>
#include <map>
#include <stdexcept>

namespace SimpleFluid::PLIC
{
namespace details
{
void charge_point_snapshot(std::size_t count, std::size_t size);
}
struct PointSnapshotAllocation
{
    void* state = nullptr;
    void (*charge)(void*, std::size_t, std::size_t) = nullptr;
    void add(std::size_t count, std::size_t size) const
    {
        if (charge)
            charge(state, count, size);
        else
            details::charge_point_snapshot(count, size);
    }
};
struct PointSnapshotRange
{
    std::uint32_t first = 0, count = 0;
};
template<ArithmeticMode Mode> class PrivateAuthorityRef
{
public:
    PrivateAuthorityRef() = default;
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    bool operator==(const PrivateAuthorityRef&) const = default;

private:
    friend class PointSnapshotBuilder<Mode>;
    PrivateAuthorityRef(std::uint32_t index) : index_(index) {}
    std::uint32_t index_ = std::numeric_limits<std::uint32_t>::max();
};
template<ArithmeticMode Mode> class AdmittedPointIncidence
{
public:
    AdmittedPointIncidence() = default;

private:
    friend class DynamicGeometryAccess;
    friend class PointSnapshotBuilder<Mode>;
    explicit AdmittedPointIncidence(PointAuthorityRange keys) : keys_(keys) {}
    PointAuthorityRange keys_;
};
template<ArithmeticMode Mode> class LocalAuthorityRange
{
public:
    LocalAuthorityRange() = default;
    [[nodiscard]] std::uint32_t first() const noexcept { return first_; }
    [[nodiscard]] std::uint32_t size() const noexcept { return count_; }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
    bool operator==(const LocalAuthorityRange&) const = default;

private:
    friend class PointSnapshotBuilder<Mode>;
    LocalAuthorityRange(std::uint32_t first, std::uint32_t count) : first_(first), count_(count) {}
    std::uint32_t first_ = 0, count_ = 0;
};
struct PointErrorWitness
{
    PointProvenanceRole role = PointProvenanceRole::Native;
    std::uint32_t index = 0;
    std::array<long double, 3> error{};
};
template<ArithmeticMode Mode> struct NativePointFactBlock
{
    std::vector<PointCoordinates<Mode>> points;
    // Ordinary retains rare source-error evidence out of line. Bounded stores
    // its enclosures directly in PointCoordinates<Mode>.
    std::vector<PointErrorWitness> witnesses;
};
template<ArithmeticMode Mode> struct NativePointRefRow
{
    std::uint32_t fact_block = 0, fact_index = 0;
    LocalAuthorityRange<Mode> incidence;
    bool represented = false;
};
template<ArithmeticMode Mode> struct GeneratedPointRecipe
{
    LocalAuthorityRange<Mode> incidence;
    PointSnapshotRange edges;
    bool constructed = true;
};
template<ArithmeticMode Mode> struct ExportPointAlias
{
    StoredPointRef<Mode> source;
    LocalAuthorityRange<Mode> incidence;
    bool constructed = false, numerical_only = false;
};
template<ArithmeticMode Mode> using StoredPointEdge = std::array<StoredPointRef<Mode>, 2>;
struct PointSnapshotContextBase : std::enable_shared_from_this<PointSnapshotContextBase>
{
    explicit PointSnapshotContextBase(ArithmeticMode requested) : mode(requested) {}
    virtual ~PointSnapshotContextBase() = default;
    const ArithmeticMode mode;
    bool frozen = false;
    std::vector<std::shared_ptr<const PointSnapshotContextBase>> dependencies;
};
template<ArithmeticMode Mode> struct PointSnapshotContext final : PointSnapshotContextBase
{
    PointSnapshotContext() : PointSnapshotContextBase(Mode) {}
    [[nodiscard]] std::size_t exclusive_storage_bytes() const;
    std::vector<std::shared_ptr<const NativePointFactBlock<Mode>>> native_fact_blocks;
    std::vector<NativePointRefRow<Mode>> native_refs;
    std::vector<GeneratedPointRecipe<Mode>> generated_recipes;
    std::vector<ExportPointAlias<Mode>> export_aliases;
    std::vector<PointCoordinates<Mode>> generated_points, export_points, numerical_points;
    std::vector<PointErrorWitness> witnesses;
    std::vector<PlaneAuthority> authority_table;
    std::vector<PrivateAuthorityRef<Mode>> incidence;
    std::vector<StoredPointEdge<Mode>> edges;
};
class PointEdgeRange;
/** A borrowed, context-bound token. No owner increments occur while resolving
 * ancestors, source-edge endpoints, or native incidence witnesses. */
class PointProvenanceView
{
public:
    PointProvenanceView() = default;
    [[nodiscard]] explicit operator bool() const noexcept { return context_ && bool(ref_); }
    [[nodiscard]] PointProvenanceView view() const { return *this; }
    [[nodiscard]] const void* context_identity() const noexcept { return context_; }
    [[nodiscard]] const PointSnapshotContextBase* context() const noexcept { return context_; }
    [[nodiscard]] DynamicStoredPointRef local_ref() const noexcept { return ref_; }
    [[nodiscard]] PointProvenanceRef owning() const;
    [[nodiscard]] ArithmeticMode arithmetic_mode() const;
    [[nodiscard]] const std::array<long double, 3>& value() const;
    [[nodiscard]] const std::array<long double, 3>& absolute_error() const;
    [[nodiscard]] bool represented_native_source() const;
    [[nodiscard]] bool constructed() const;
    [[nodiscard]] bool numerical_only() const;
    [[nodiscard]] PointAuthorityRange supports() const;
    [[nodiscard]] PointProvenanceView source_origin() const;
    [[nodiscard]] PointEdgeRange source_edges() const;
    [[nodiscard]] bool proves_incidence(const PlaneAuthority&) const;
    bool operator==(const PointProvenanceView&) const = default;
    bool operator==(const PointProvenanceRef& other) const { return *this == other.view(); }

private:
    friend class PointProvenanceRef;
    friend PointProvenanceView resolve_point(const PointSnapshotContextBase*, DynamicStoredPointRef);
    template<ArithmeticMode Mode>
    friend PointProvenanceView resolve_point(const PointSnapshotContext<Mode>*, StoredPointRef<Mode>);
    template<ArithmeticMode Mode>
    friend PointProvenanceView resolve_point(
        const std::shared_ptr<const PointSnapshotContext<Mode>>&, StoredPointRef<Mode>);
    PointProvenanceView(const PointSnapshotContextBase*, DynamicStoredPointRef);
    const PointSnapshotContextBase* context_ = nullptr;
    DynamicStoredPointRef ref_;
};
class PointEdgeRange
{
public:
    struct Iterator
    {
        PointProvenanceView owner;
        std::size_t index = 0;
        using value_type = std::array<PointProvenanceView, 2>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        using reference = value_type;
        using pointer = void;
        std::array<PointProvenanceView, 2> operator*() const;
        Iterator& operator++()
        {
            ++index;
            return *this;
        }
        Iterator operator++(int)
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const Iterator&) const = default;
    };
    PointEdgeRange() = default;
    PointEdgeRange(PointProvenanceView owner, std::size_t count) : owner_(owner), count_(count) {}
    Iterator begin() const { return {owner_, 0}; }
    Iterator end() const { return {owner_, count_}; }
    bool empty() const { return count_ == 0; }
    std::size_t size() const { return count_; }
    auto operator[](std::size_t index) const { return *Iterator{owner_, index}; }

private:
    PointProvenanceView owner_;
    std::size_t count_ = 0;
};
inline PointProvenanceView resolve_point(const PointSnapshotContextBase* context, DynamicStoredPointRef ref)
{
    return {context, ref};
}
inline PointProvenanceRef bind_point(std::shared_ptr<const PointSnapshotContextBase> context, DynamicStoredPointRef ref)
{
    return {std::move(context), ref};
}
template<ArithmeticMode Mode> DynamicStoredPointRef erase_point_ref(StoredPointRef<Mode> ref)
{
    return {ref.dependency_slot(), ref.index(), ref.role()};
}
template<ArithmeticMode Mode>
PointProvenanceView resolve_point(const PointSnapshotContext<Mode>* context, StoredPointRef<Mode> ref)
{
    return {context, erase_point_ref(ref)};
}
template<ArithmeticMode Mode>
PointProvenanceView resolve_point(
    const std::shared_ptr<const PointSnapshotContext<Mode>>& context, StoredPointRef<Mode> ref)
{
    return {context.get(), erase_point_ref(ref)};
}
template<ArithmeticMode Mode>
PointProvenanceRef bind_point(std::shared_ptr<const PointSnapshotContext<Mode>> context, StoredPointRef<Mode> ref)
{
    return bind_point(std::move(context), erase_point_ref(ref));
}

class PointAuthorityRangeAccess
{
public:
    template<ArithmeticMode Mode>
    static PointAuthorityRange range(const PointSnapshotContext<Mode>* context, LocalAuthorityRange<Mode> range)
    {
        return {context, range.first(), range.size(), [](const void* owner, std::size_t index) -> const PlaneAuthority&
            {
                const auto& typed = *static_cast<const PointSnapshotContext<Mode>*>(owner);
                return typed.authority_table.at(typed.incidence.at(index).index());
            }};
    }
};
template<ArithmeticMode Mode>
PointAuthorityRange resolve_authorities(const PointSnapshotContext<Mode>* context, LocalAuthorityRange<Mode> range)
{
    return context ? PointAuthorityRangeAccess::range(context, range) : PointAuthorityRange{};
}
template<ArithmeticMode Mode>
PointAuthorityRange resolve_authorities(
    const std::shared_ptr<const PointSnapshotContext<Mode>>& context, LocalAuthorityRange<Mode> range)
{
    return resolve_authorities(context.get(), range);
}

/** A builder accepts only already-frozen external dependencies. Local draft
 * views are usable for validation but cannot escape as owning public tokens. */
template<ArithmeticMode Mode> class PointSnapshotBuilder
{
public:
    using Context = PointSnapshotContext<Mode>;
    explicit PointSnapshotBuilder(PointSnapshotAllocation = {});
    PointSnapshotBuilder(const PointSnapshotBuilder&) = delete;
    PointSnapshotBuilder& operator=(const PointSnapshotBuilder&) = delete;
    PointSnapshotBuilder(PointSnapshotBuilder&&) = delete;
    PointSnapshotBuilder& operator=(PointSnapshotBuilder&&) = delete;
    [[nodiscard]] const Context* context() const noexcept { return context_.get(); }
    [[nodiscard]] StoredPointRef<Mode> import(PointProvenanceView);
    [[nodiscard]] StoredPointRef<Mode> import(const PointProvenanceRef& ref) { return import(ref.view()); }
    /** Explicitly pin a plane/source context even if none of its original
     * vertices survives in the newly constructed geometry. */
    void retain_context(const PointSnapshotContextBase*);
    [[nodiscard]] LocalAuthorityRange<Mode> authority_range(AdmittedPointIncidence<Mode> proof)
    {
        require_draft();
        Mutation mutation{*this};
        return append_incidence(proof.keys_);
    }
    [[nodiscard]] LocalAuthorityRange<Mode> import_authority_range(PointAuthorityRange range)
    {
        require_draft();
        Mutation mutation{*this};
        return append_incidence(range);
    }
    [[nodiscard]] LocalAuthorityRange<Mode> append_authority(LocalAuthorityRange<Mode>, AdmittedPointIncidence<Mode>);
    [[nodiscard]] PrivateAuthorityRef<Mode> register_authority(const PlaneAuthority&);
    [[nodiscard]] PointProvenanceView native_source(std::array<long double, 3>, std::array<long double, 3>,
        AdmittedPointIncidence<Mode>, PointProvenanceView facts = {});
    [[nodiscard]] PointProvenanceView sealed(
        std::array<long double, 3>, std::array<long double, 3>, AdmittedPointIncidence<Mode>);
    [[nodiscard]] PointProvenanceView generated(std::array<long double, 3>, std::array<long double, 3>,
        std::span<const std::array<PointProvenanceView, 2>>, AdmittedPointIncidence<Mode>);
    [[nodiscard]] PointProvenanceView export_alias(std::array<long double, 3>, std::array<long double, 3>,
        PointProvenanceView, AdmittedPointIncidence<Mode>, bool numerical = false);
    [[nodiscard]] PointProvenanceView numerical(std::array<long double, 3>, std::array<long double, 3>);
    [[nodiscard]] std::shared_ptr<const Context> freeze();

private:
    void require_draft() const;
    LocalAuthorityRange<Mode> append_incidence(PointAuthorityRange);
    void witness(PointProvenanceRole, std::uint32_t, const std::array<long double, 3>&);
    PointSnapshotAllocation allocation_;
    template<class T> void append(std::vector<T>& values, T value);
    std::shared_ptr<Context> context_;
    std::shared_ptr<NativePointFactBlock<Mode>> own_facts_;
    std::map<PlaneAuthority, std::uint32_t> authority_index_;
    bool aborted_ = false;
    struct Mutation
    {
        PointSnapshotBuilder& builder;
        int exceptions = std::uncaught_exceptions();
        ~Mutation()
        {
            if (std::uncaught_exceptions() > exceptions)
                builder.aborted_ = true;
        }
    };
};
} // namespace SimpleFluid::PLIC

#include "geometry/PLIC/details/PointSnapshot.ipp"
