#pragma once
#include "AdaptiveContactQuery.hh"
#include "AdaptiveLeafLayout.hh"
#include "ContactBackend.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/AdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/CartesianIncidenceBudget.hh"
#include "geometry/mesh/AMR/CartesianIncidenceSource.hh"
#include "geometry/mesh/AMR/ContactIncidenceRowView.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Operation-local selection. Pointer selections borrow the existing hierarchy
 * overlay and cannot escape into a published snapshot. */
class CartesianLeafSelection
{
public:
    CartesianLeafSelection(std::span<const AdaptiveLeaf> rows) : rows_(rows) {}
    CartesianLeafSelection(const std::vector<AdaptiveLeaf>& rows) : rows_(rows) {}
    CartesianLeafSelection(const std::vector<const AdaptiveLeaf*>& rows) : pointers_(rows) {}
    [[nodiscard]] std::size_t size() const noexcept { return rows_.size() + pointers_.size(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] const AdaptiveLeaf& operator[](std::size_t i) const
    {
        return pointers_.empty() ? rows_[i] : *pointers_[i];
    }
    [[nodiscard]] const AdaptiveLeaf& front() const { return (*this)[0]; }
    struct Iterator
    {
        const CartesianLeafSelection* source;
        std::size_t index;
        const AdaptiveLeaf& operator*() const { return (*source)[index]; }
        Iterator& operator++()
        {
            ++index;
            return *this;
        }
        bool operator==(const Iterator&) const = default;
    };
    Iterator begin() const { return {this, 0}; }
    Iterator end() const { return {this, size()}; }

private:
    std::span<const AdaptiveLeaf> rows_;
    std::span<const AdaptiveLeaf* const> pointers_;
};

/** Factory-issued promise that, for one authenticated Cartesian source, complete
 * normalized contact equals closed logical-cell contact. An enum tag or a
 * successfully decoded support key cannot carry this. */
struct CartesianContactCapability
{
    AdaptiveBytes frame; ///< exact kernel descriptor, reused as the trace stem
    AdaptiveBytes domain;
    AMRGrid grid; ///< authenticated source recipe; no per-cell coordinate storage
    CartesianEvaluationEnvironment evaluation;
    std::fenv_t environment{};                     ///< replay controls; never canonical identity bytes
    std::array<long double, 3> origin{};           ///< source origin (diagnostic compatibility)
    std::array<long double, 3> cell_spans{};       ///< represented widths of cell index zero only
    std::array<std::uint64_t, 3> factors{0, 0, 0}; ///< Cartesian closure split per axis (Isotropic)
    std::array<std::uint64_t, 3> counts{};         ///< total cells per axis at `level`
    unsigned level = 0;
    std::uint64_t region = 0;
    /** Actual native evaluation, never origin + index * an analytic width. */
    [[nodiscard]] double coordinate(std::size_t axis, std::uint64_t index) const
    {
        return cartesian_frozen_coordinate(grid, axis, index, level, environment);
    }
    /** Exact represented width under the strict arithmetic certificate below.
     * The constructor's local translation and power-of-two frame scaling, and
     * the decoded geometry certificate's rescaling, preserve this value. */
    [[nodiscard]] long double axis_span(std::size_t axis, std::uint64_t index) const
    {
        return static_cast<long double>(coordinate(axis, index + 1)) -
               static_cast<long double>(coordinate(axis, index));
    }
};
/** Machine-checked sufficient arithmetic certificate, deliberately narrower
 * than finite positive boxes: IEEE binary64 native coordinates from one shared
 * evaluation function; nearest, gradual x86 arithmetic with >=64-bit binary
 * long double; exact adjacent translation; bounded normal frame/face/volume
 * operations; strictly increasing source axes; uniform nonperiodic ancestry.
 * Native box construction translates by its first corner and rescales by a
 * binary exponent. Exact subtraction and scaling preserve both source corners
 * and decoded axis_spans. Consequently distinct lattice planes stay distinct,
 * full uniform native rectangles need no interior vertices, and their
 * axis-aligned line cross-products have no subtractive cancellation. Thus
 * normalized closed contact equals logical closed contact. The evaluated-axis
 * proof is bounded to at most 2^20 cells per axis before scanning. Recipe
 * authentication and collective complete
 * membership are additional requirements, never inferred from this predicate.
 * Coordinate contraction is permitted because every path uses the same native
 * evaluation boundary and the predicate checks its actual evaluated values. */
[[nodiscard]] inline std::optional<CartesianContactCapability> capability_from_grid(
    const AMRGrid& grid, const AdaptiveBytes& domain, unsigned level)
{
    if (domain.empty() || !cartesian_contact_arithmetic_eligible(grid, level))
        return std::nullopt;
    CartesianContactCapability capability;
    capability.grid = grid;
    capability.evaluation = cartesian_evaluation_environment();
    if (std::fegetenv(&capability.environment) != 0)
        return std::nullopt;
    capability.domain = domain;
    capability.region = grid.region_id;
    capability.level = level;
    for (std::size_t axis = 0; axis != 3; ++axis)
    {
        // The arithmetic predicate checks max_level and every product before
        // these shifts, including the native finest-level integer period.
        const auto base = std::uint64_t(grid.root_blocks[axis]) * grid.block_cells[axis];
        capability.counts[axis] = std::uint64_t(base) << level;
        capability.cell_spans[axis] = capability.axis_span(axis, 0);
        capability.origin[axis] = static_cast<long double>(grid.origin[axis]);
    }
    return capability;
}
/** Total logical cells of one certified frame; saturated on overflow. */
[[nodiscard]] inline std::optional<std::uint64_t> capability_cells(const CartesianContactCapability& capability)
{
    std::uint64_t product = 1;
    for (const auto count : capability.counts)
    {
        if (!count || product > std::numeric_limits<std::uint64_t>::max() / count)
            return std::nullopt;
        product *= count;
    }
    return product;
}
/** Narrow first support class: complete source membership and native recipes
 * passing the strict arithmetic predicate above. Merely positive finite spans
 * and uniform nonperiodic ancestry do not establish this theorem. */
[[nodiscard]] inline std::optional<CartesianContactCapability> issue_cartesian_contact_capability(
    const AdaptiveProviderKernel& kernel, const CartesianIncidenceSource& source)
{
    const auto grid = SimpleFluid::Meshes::AMR::trusted_cartesian_incidence_grid(kernel);
    if (!grid)
        return std::nullopt;
    if (grid->periodic[0] || grid->periodic[1] || grid->periodic[2])
        return std::nullopt;
    if (!cartesian_grid_identity_equal(*grid, source.grid()) ||
        source.evaluation_environment() != cartesian_evaluation_environment())
        return std::nullopt;
    if (source.ranges().empty())
        return std::nullopt;
    const auto level = source.ranges().front().level;
    for (const auto& range : source.ranges())
        if (range.level != level)
            return std::nullopt;
    auto capability = capability_from_grid(*grid, source.domain(), level);
    if (!capability)
        return std::nullopt;
    const auto cells = capability_cells(*capability);
    if (!cells || *cells != source.cell_count())
        return std::nullopt;
    // The immutable source retained the actual native evaluation, rather than
    // an analytic reconstruction. Bind every occupied axis value as well as
    // the complete grid/ancestry/refinement recipe and evaluation environment.
    for (const auto& axis : source.axes())
    {
        if (axis.axis >= 3 || axis.level != level || axis.first > capability->counts[axis.axis] ||
            axis.values.size() > capability->counts[axis.axis] - axis.first + 1)
            return std::nullopt;
        for (std::size_t i = 0; i < axis.values.size(); ++i)
            if (std::bit_cast<std::uint64_t>(axis.values[i]) !=
                std::bit_cast<std::uint64_t>(capability->coordinate(axis.axis, axis.first + i)))
                return std::nullopt;
    }
    capability->frame = kernel.descriptor();
    return capability;
}
/** Topology-only issuance: the caller must separately authenticate actual native
 * leaves before constructing a state. Keys never authenticate native recipes.
 * Local stores may cover a partition subset; the caller proves complete global
 * coverage collectively before any neighbor is treated as absent. */
[[nodiscard]] inline std::optional<CartesianContactCapability> issue_cartesian_contact_capability(
    const AdaptiveProviderKernel& kernel, std::span<const AdaptiveCellKey> keys, const AdaptiveBytes& domain)
{
    const auto grid = SimpleFluid::Meshes::AMR::trusted_cartesian_incidence_grid(kernel);
    if (!grid || grid->periodic[0] || grid->periodic[1] || grid->periodic[2])
        return std::nullopt;
    if (keys.empty() || domain.empty())
        return std::nullopt;
    const auto* first = std::get_if<CartesianCellAncestry>(&keys.front().ancestry);
    if (!first)
        return std::nullopt;
    const auto level = first->level;
    auto capability = capability_from_grid(*grid, domain, level);
    if (!capability)
        return std::nullopt;
    const auto cells = capability_cells(*capability);
    if (!cells || *cells < keys.size())
        return std::nullopt;
    for (const auto& key : keys)
    {
        const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
        if (!ancestry || ancestry->level != level || key.region != grid->region_id)
            return std::nullopt;
        for (std::size_t axis = 0; axis != 3; ++axis)
            if (ancestry->index[axis] >= capability->counts[axis])
                return std::nullopt;
    }
    capability->frame = kernel.descriptor();
    return capability;
}
/** Cartesian contact backend: grading views derived from the captured
 * capability. It retains one shared source/topology snapshot and no per-cell
 * contact objects. Views borrow backend members and are valid only for the
 * synchronous pair evaluation that requested them. */
class SIMPLEFLUID_LOCAL CartesianContactBackend final : public ContactBackend
{
public:
    CartesianContactBackend() = default;
    CartesianContactBackend(CartesianContactCapability capability, int owner)
        : capability_(std::move(capability)), owner_(owner)
    {
        trace_face_.stem = capability_.frame;
        trace_face_.depth = capability_.level;
        trace_face_.split = AdaptiveSplit::Isotropic;
        trace_face_.terminal = true;
    }
    [[nodiscard]] ContactBackendKind kind() const noexcept override { return ContactBackendKind::Cartesian; }
    [[nodiscard]] const CartesianContactCapability& capability() const noexcept { return capability_; }
    [[nodiscard]] int owner() const noexcept { return owner_; }
    [[nodiscard]] const AdaptiveTraceFace& trace_face() const noexcept { return trace_face_; }
    bool closure_views(const AdaptiveCellKey& first, const AdaptiveCellKey& second, ContactClosureView& view_a,
        ContactClosureView& view_b) const noexcept override
    {
        if (!cartesian_contact_arithmetic_environment_supported())
            return false;
        try
        {
            if (capability_.evaluation != cartesian_evaluation_environment())
                return false;
        }
        catch (const std::invalid_argument&)
        {
            return false;
        }
        const auto a = view_for(first, first_spans_);
        const auto b = view_for(second, second_spans_);
        if (!a || !b)
            return false;
        view_a = *a;
        view_b = *b;
        return true;
    }

private:
    [[nodiscard]] std::optional<ContactClosureView> view_for(
        const AdaptiveCellKey& key, std::array<long double, 3>& spans) const noexcept
    {
        const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
        if (!ancestry || key.region != capability_.region || ancestry->level != capability_.level)
            return std::nullopt;
        for (std::size_t axis = 0; axis != 3; ++axis)
        {
            if (ancestry->index[axis] >= capability_.counts[axis])
                return std::nullopt;
            spans[axis] = capability_.axis_span(axis, ancestry->index[axis]);
        }
        return ContactClosureView{owner_, std::span<const long double>(spans),
            std::span<const std::uint64_t>(capability_.factors), std::span<const AdaptiveTraceFace>(&trace_face_, 1),
            true};
    }
    CartesianContactCapability capability_;
    AdaptiveTraceFace trace_face_{};
    // Both views must survive the synchronous pair evaluation independently.
    // A second view must never overwrite the first cell's represented widths.
    mutable std::array<long double, 3> first_spans_{}, second_spans_{};
    int owner_ = -1;
};
/** The 26 closed-contact stencil offsets of a regular Cartesian cell. */
[[nodiscard]] constexpr std::array<std::array<int, 3>, 26> cartesian_contact_offsets()
{
    std::array<std::array<int, 3>, 26> result{};
    std::size_t count = 0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                if (x || y || z)
                    result[count++] = {x, y, z};
    return result;
}
/** Allocator used only by allocate_shared's initial control/object allocation.
 * The measurement pointer is construction-only; deallocation uses its own exact
 * rebound count, and the copied keeper pins the arena through weak-pin release. */
template<class T> struct CartesianContactAllocator
{
    using value_type = T;
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t* construction_bytes = nullptr;
    CartesianContactAllocator(Budget& b, std::shared_ptr<Budget> pin, std::size_t& bytes)
        : budget(&b), keeper(std::move(pin)), construction_bytes(&bytes)
    {
    }
    template<class U>
    CartesianContactAllocator(const CartesianContactAllocator<U>& other) noexcept
        : budget(other.budget), keeper(other.keeper), construction_bytes(other.construction_bytes)
    {
    }
    [[nodiscard]] T* allocate(std::size_t count)
    {
        const auto bytes = cartesian_detail::product(count, sizeof(T));
        budget->add(bytes);
        try
        {
            auto* allocation = std::allocator<T>{}.allocate(count);
            *construction_bytes = bytes;
            return allocation;
        }
        catch (...)
        {
            budget->used -= bytes;
            throw;
        }
    }
    void deallocate(T* allocation, std::size_t count) noexcept
    {
        std::allocator<T>{}.deallocate(allocation, count);
        budget->used -= count * sizeof(T);
    }
    template<class U> bool operator==(const CartesianContactAllocator<U>& other) const noexcept
    {
        return budget == other.budget;
    }
};
/** Immutable after preparation: one admitted snapshot is physically shared by
 * accepted/candidate/sibling sessions. Local construction, owner installation
 * and grading finish before publication through shared_ptr<const State>.
 * The retained arena lease dies after all state containers; the allocator keeps
 * the same arena alive through destruction of the shared control allocation. */
class SIMPLEFLUID_LOCAL CartesianContactState
{
    struct ConstructionTag
    {
    };

public:
    // Public for allocate_shared; only this factory can create ConstructionTag.
    CartesianContactState(ConstructionTag, Budget& budget, std::shared_ptr<Budget> keeper, std::uint64_t generation)
        : keeper_(std::move(keeper)), allocation_(budget), generation_(generation)
    {
    }
    CartesianContactState(const CartesianContactState&) = delete;
    CartesianContactState& operator=(const CartesianContactState&) = delete;
    CartesianContactState(CartesianContactState&&) = delete;
    CartesianContactState& operator=(CartesianContactState&&) = delete;
    [[nodiscard]] static std::shared_ptr<CartesianContactState> build(const AdaptiveProviderKernel& kernel,
        CartesianLeafSelection owned, int owner, Budget& budget, std::shared_ptr<Budget> keeper,
        std::uint64_t generation, const CartesianContactCapability* elected = nullptr,
        std::shared_ptr<const AdaptiveLeafLayout> layout = {})
    {
        cartesian_detail::check_keeper(budget, keeper);
        const auto grid = SimpleFluid::Meshes::AMR::trusted_cartesian_incidence_grid(kernel);
        if (owner < 0 || !grid || (owned.empty() && !elected))
            return {};
        const auto& domain = owned.empty() ? elected->domain : owned.front().domain;
        const auto* first = owned.empty() ? nullptr : std::get_if<CartesianCellAncestry>(&owned.front().key.ancestry);
        if ((!owned.empty() && !first) || domain.empty())
            return {};
        const unsigned level = first ? first->level : elected->level;
        if (elected && (!cartesian_grid_identity_equal(*grid, elected->grid) || elected->domain != domain ||
                           elected->level != level || elected->evaluation != cartesian_evaluation_environment()))
            return {};
        if (!cartesian_contact_arithmetic_eligible(*grid, level))
            return {};
        // Authentication itself retains no geometry, keys or axis buffers.
        for (const auto& leaf : owned)
        {
            const auto* ancestry = std::get_if<CartesianCellAncestry>(&leaf.key.ancestry);
            if (leaf.domain != domain || !ancestry || ancestry->level != level ||
                !CartesianIncidenceSource::authenticates_native_leaf(*grid, leaf))
                return {};
        }
        // Admission precedes the exact rebound shared object/control allocation.
        std::size_t shared_bytes = 0;
        auto state = std::allocate_shared<CartesianContactState>(
            CartesianContactAllocator<CartesianContactState>(budget, keeper, shared_bytes), ConstructionTag{}, budget,
            keeper, generation);
        state->shared_allocation_bytes_ = shared_bytes;
        state->layout_ = std::move(layout);
        if (state->layout_)
        {
            bool exact = state->layout_->size() == owned.size();
            for (const auto& leaf : owned)
            {
                const auto* accepted = state->layout_->find(leaf.key);
                exact = exact && accepted == &leaf;
            }
            if (!exact)
            {
                state->pending_selection_ = true;
                state->allocation_.reserve(state->pending_leaves_, owned.size());
                for (const auto& leaf : owned)
                    state->pending_leaves_.push_back(&leaf);
                std::ranges::sort(
                    state->pending_leaves_, {}, [](const auto* leaf) -> const AdaptiveCellKey& { return leaf->key; });
                for (std::size_t i = 1; i < state->pending_leaves_.size(); ++i)
                    if (state->pending_leaves_[i - 1]->key == state->pending_leaves_[i]->key)
                        return {};
            }
        }
        else
        {
            // Only the standalone halo compatibility route owns these catalogs.
            state->allocation_.reserve(state->keys_, owned.size());
            state->allocation_.reserve(state->revisions_, owned.size());
            for (const auto& leaf : owned)
            {
                if (state->index_.contains(leaf.key))
                    return {};
                state->allocation_.retain(node_bytes<typename decltype(index_)::value_type>());
                state->index_.emplace(leaf.key, state->keys_.size());
                state->keys_.push_back(leaf.key);
                state->revisions_.push_back(leaf.revision);
            }
        }
        // The native descriptor has 152 fixed bytes. Prepay its complete writer
        // growth history while the old/new capacities can overlap. Transfer only
        // the actual retained frame/domain buffers into the snapshot's lease.
        cartesian_detail::Allocation scratch(budget);
        scratch.retain(domain.size());
        scratch.retain(cartesian_detail::growth(152, 1));
        auto capability = capability_from_grid(*grid, domain, level);
        if (!capability)
            return {};
        capability->frame = kernel.descriptor();
        if (capability->domain.capacity() != domain.size() ||
            capability->frame.capacity() > cartesian_detail::growth(152, 1))
            throw std::invalid_argument("Cartesian capability exceeded its admitted native capacity.");
        if (elected && (elected->frame != capability->frame || elected->counts != capability->counts ||
                           elected->region != capability->region || elected->factors != capability->factors))
            return {};
        const auto transferred = cartesian_detail::add(capability->domain.capacity(), capability->frame.capacity());
        state->allocation_.bytes = cartesian_detail::add(state->allocation_.bytes, transferred);
        scratch.bytes -= transferred;
        state->allocation_.retain(capability->frame.size()); // backend trace-stem copy
        state->backend_ = CartesianContactBackend(std::move(*capability), owner);
        if (state->backend_.trace_face().stem.capacity() != state->capability().frame.size())
            throw std::invalid_argument("Cartesian trace-stem capacity differs from admission.");
        return state;
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return shared_allocation_bytes_ + allocation_.bytes; }
    [[nodiscard]] std::size_t admitted_bytes() const noexcept { return metadata_bytes(); }
    [[nodiscard]] const Budget* allocation_budget() const noexcept { return &allocation_.budget; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] const AdaptiveBytes& domain() const noexcept { return capability().domain; }
    [[nodiscard]] const CartesianContactBackend& backend() const noexcept { return backend_; }
    /** Known admission beyond visitor payload: exact measured shared control
     * allocation remainder and the qualified tree-node header allowance. */
    [[nodiscard]] std::size_t admission_allowance_bytes() const noexcept
    {
        return shared_allocation_bytes_ - sizeof(*this) +
               (index_.size() + remote_.size() + failing_.size()) * node_header_bytes;
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(
            visitor, this, GeometryContactPayloadRole::CartesianContactStateObject, sizeof(*this), true);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianContactKeys, keys_);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianContactRevisions, revisions_);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianContactKeys, pending_leaves_);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianContactFrame, capability().frame);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianContactDomain, capability().domain);
        contact_payload_buffer(
            visitor, this, GeometryContactPayloadRole::CartesianContactTraceStem, backend_.trace_face().stem);
        contact_payload_buffer(visitor, this, GeometryContactPayloadRole::CartesianContactMarks, marks_);
        for (const auto& value : index_)
            contact_payload_node_value(visitor, this, GeometryContactPayloadRole::CartesianContactIndex, value);
        for (const auto& value : remote_)
            contact_payload_node_value(visitor, this, GeometryContactPayloadRole::CartesianContactRemoteOwners, value);
        for (const auto& value : failing_)
            contact_payload_node_value(visitor, this, GeometryContactPayloadRole::CartesianContactFailures, value);
    }
    class KeyRange
    {
    public:
        const CartesianContactState* state;
        std::size_t size() const noexcept { return state->key_count(); }
        bool empty() const noexcept { return size() == 0; }
        const AdaptiveCellKey& operator[](std::size_t i) const { return state->key_at(i); }
        const AdaptiveCellKey& at(std::size_t i) const { return state->key_at(i); }
        struct Iterator
        {
            const CartesianContactState* state;
            std::size_t index;
            const AdaptiveCellKey& operator*() const { return state->key_at(index); }
            Iterator& operator++()
            {
                ++index;
                return *this;
            }
            bool operator==(const Iterator&) const = default;
        };
        Iterator begin() const { return {state, 0}; }
        Iterator end() const { return {state, size()}; }
    };
    [[nodiscard]] KeyRange keys() const noexcept { return {this}; }
    [[nodiscard]] bool requirements_empty() const noexcept { return marks_.empty(); }
    [[nodiscard]] bool pending_selection() const noexcept { return pending_selection_; }
    [[nodiscard]] const std::shared_ptr<const AdaptiveLeafLayout>& layout() const noexcept { return layout_; }
    /** Validate before any publication, then release operation-local overlay
     * borrows. This private draft has never escaped through a reader. */
    void validate_layout(const std::shared_ptr<const AdaptiveLeafLayout>& next) const
    {
        if (!layout_ || !next || next->size() != key_count())
            throw std::invalid_argument("Cartesian seal requires the exact complete target layout.");
        for (std::size_t i = 0; i < key_count(); ++i)
        {
            const auto& key = key_at(i);
            const auto* leaf = next->find(key);
            if (!leaf || leaf->revision != revision(key) || leaf->domain != domain())
                throw std::invalid_argument("Cartesian seal disagrees with the authenticated leaf selection.");
        }
    }
    void bind_validated_layout(std::shared_ptr<const AdaptiveLeafLayout> next) noexcept
    {
        const auto bytes = pending_leaves_.capacity() * sizeof(const AdaptiveLeaf*);
        std::vector<const AdaptiveLeaf*>().swap(pending_leaves_);
        allocation_.bytes -= bytes;
        allocation_.budget.used -= bytes;
        layout_ = std::move(next);
        pending_selection_ = false;
    }
    [[nodiscard]] const CartesianContactCapability& capability() const noexcept { return backend_.capability(); }
    [[nodiscard]] int owner() const noexcept { return backend_.owner(); }
    [[nodiscard]] std::size_t total_cells() const noexcept
    {
        const auto cells = capability_cells(capability());
        return cells ? std::size_t(*cells) : 0;
    }
    [[nodiscard]] std::size_t record_id(const AdaptiveCellKey& key) const
    {
        if (pending_selection_)
        {
            const auto found = std::lower_bound(pending_leaves_.begin(), pending_leaves_.end(), key,
                [](const AdaptiveLeaf* leaf, const AdaptiveCellKey& value) { return leaf->key < value; });
            if (found != pending_leaves_.end() && (*found)->key == key)
                return std::size_t(found - pending_leaves_.begin());
        }
        else if (layout_)
        {
            if (const auto ordinal = layout_->ordinal_of(key))
                return *ordinal;
        }
        else if (const auto found = index_.find(key); found != index_.end())
            return found->second;
        throw std::invalid_argument("Foreign Cartesian contact cell.");
    }
    [[nodiscard]] bool owns(const AdaptiveCellKey& key) const
    {
        if (pending_selection_)
        {
            const auto found = std::lower_bound(pending_leaves_.begin(), pending_leaves_.end(), key,
                [](const AdaptiveLeaf* leaf, const AdaptiveCellKey& value) { return leaf->key < value; });
            return found != pending_leaves_.end() && (*found)->key == key;
        }
        return layout_ ? layout_->contains(key) : index_.contains(key);
    }
    [[nodiscard]] std::uint64_t revision(const AdaptiveCellKey& key) const
    {
        const auto ordinal = record_id(key);
        if (pending_selection_)
            return pending_leaves_[ordinal]->revision;
        return layout_ ? layout_->at_ordinal(ordinal).revision : revisions_.at(ordinal);
    }
    [[nodiscard]] std::pair<int, std::uint64_t> owner_revision(const AdaptiveCellKey& key) const
    {
        if (owns(key))
            return {owner(), revision(key)};
        const auto& ancestry = std::get<CartesianCellAncestry>(key.ancestry);
        const auto& receipt = remote_.at(ancestry.index);
        if (!receipt.authenticated)
            throw std::invalid_argument("Contact snapshot needs an authenticated remote revision.");
        return {receipt.owner, receipt.revision};
    }
    [[nodiscard]] std::optional<AdaptiveCellKey> neighbor_key(
        const AdaptiveCellKey& key, const std::array<int, 3>& offset) const
    {
        return offset_key(key, offset);
    }
    [[nodiscard]] double coordinate(std::size_t axis, std::uint64_t index) const
    {
        return capability().coordinate(axis, index);
    }
    void visit_closed_contacts(const AdaptiveCellKey& key, const ContactWitnessVisitor& visit) const
    {
        if (pending_selection_ || !owns(key))
            throw std::invalid_argument("Cartesian contacts require a sealed owned selection.");
        const auto& index = std::get<CartesianCellAncestry>(key.ancestry).index;
        const auto points = ordered_points(record_id(key));
        for (const auto& point : points)
            for (const auto& offset : cartesian_contact_offsets())
            {
                const auto other = offset_key(key, offset);
                if (!other)
                    continue;
                const auto& other_index = std::get<CartesianCellAncestry>(other->ancestry).index;
                bool shared = true;
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    const auto x = PLIC::details::GeometryIdentityAccess::point(point)[axis];
                    const auto low = std::max(index[axis], other_index[axis]);
                    const auto high = std::min(index[axis] + 1, other_index[axis] + 1);
                    shared = shared && (x == coordinate(axis, low) || (low != high && x == coordinate(axis, high)));
                }
                if (shared)
                {
                    const auto [peer, revision] = owner_revision(*other);
                    visit({*other, peer, revision, domain(), point, point, point});
                }
            }
    }
    /** In-frame stencil neighbours that are not locally owned, sorted and unique. */
    void boundary_keys(std::vector<AdaptiveCellKey>& out, cartesian_detail::Allocation& lease) const
    {
        out.clear();
        for (const auto& key : keys())
            for (const auto& offset : cartesian_contact_offsets())
            {
                const auto neighbour = offset_key(key, offset);
                if (neighbour && !owns(*neighbour))
                {
                    lease.append(out);
                    out.push_back(*neighbour);
                }
            }
        std::ranges::sort(out);
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }
    /** One authenticated ownership claim; a second, conflicting claim or a
     * self-claim is a partition fault, never a silent override. */
    void install_owner(const AdaptiveCellKey& key, int owner) { install_owner_impl(key, owner, 0, false); }
    void install_owner(const AdaptiveCellKey& key, int owner, std::uint64_t revision)
    {
        install_owner_impl(key, owner, revision, true);
    }
    void install_owner_impl(const AdaptiveCellKey& key, int owner, std::uint64_t revision, bool authenticated)
    {
        const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
        if (!ancestry || ancestry->level != capability().level || key.region != capability().region)
            throw std::invalid_argument("Owner claim lies outside the certified Cartesian frame.");
        for (std::size_t axis = 0; axis != 3; ++axis)
            if (ancestry->index[axis] >= capability().counts[axis])
                throw std::invalid_argument("Owner claim lies outside the certified Cartesian frame.");
        if (owner == backend_.owner() || owner < 0)
            throw std::invalid_argument("Owner claim names the local rank or an invalid rank.");
        if (owns(key))
            throw std::invalid_argument("Owner claim names a locally owned Cartesian cell.");
        if (evaluated_)
            throw std::invalid_argument("Cannot install owners into an evaluated Cartesian snapshot.");
        if (const auto found = remote_.find(ancestry->index); found != remote_.end())
            throw std::invalid_argument(found->second.owner == owner ? "Duplicate Cartesian cell ownership claim."
                                                                     : "Conflicting Cartesian cell ownership claims.");
        allocation_.retain(node_bytes<typename decltype(remote_)::value_type>());
        remote_.emplace(ancestry->index, RemoteReceipt{owner, revision, authenticated});
    }
    [[nodiscard]] bool has_owner(const AdaptiveCellKey& key) const
    {
        const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
        return ancestry && remote_.contains(ancestry->index);
    }
    [[nodiscard]] std::size_t remote_cells() const noexcept { return remote_.size(); }
    /** Stencil pair enumeration evaluated on the owner side: every pair with at
     * least one locally owned cell is graded from capability facts; a local-local
     * pair is visited once from its lower record id, and a local-remote pair is
     * visited by each participating rank. Uniform nonperiodic grids make
     * adjacency exactly index differences in {-1,0,1} on all axes, so an
     * in-frame neighbour must be locally owned or have an authenticated owner. */
    void evaluate(const AdaptiveClosureOptions& options, std::size_t cumulative_work_baseline = 0)
    {
        if (evaluated_)
        {
            if (options.maximum_span_ratio != evaluated_ratio_ || options.maximum_trace_jump != evaluated_trace_jump_ ||
                work_ > options.contact_budget || cumulative_work_baseline > work_)
                throw std::invalid_argument("Cartesian snapshot grading options changed after evaluation.");
            return;
        }
        if (cumulative_work_baseline > options.contact_budget)
            throw std::invalid_argument("Cartesian contact work budget exhausted.");
        work_ = cumulative_work_baseline;
        auto charge_work = [&]
        {
            if (work_ == options.contact_budget)
                throw std::invalid_argument("Cartesian contact work budget exhausted.");
            ++work_;
        };
        const auto identity = PLIC::details::GeometryIdentityAccess::point({0, 0, 0});
        for (std::size_t i = 0; i != keys().size(); ++i)
        {
            for (const auto& offset : cartesian_contact_offsets())
            {
                const auto neighbour = offset_key(keys()[i], offset);
                if (!neighbour)
                    continue;
                int other_owner = -1;
                if (owns(*neighbour))
                {
                    if (record_id(*neighbour) < i)
                        continue;
                    other_owner = backend_.owner();
                }
                else
                {
                    const auto remote = remote_.find(std::get<CartesianCellAncestry>(neighbour->ancestry).index);
                    if (remote == remote_.end())
                        throw std::invalid_argument("Unresolved Cartesian contact neighbour.");
                    other_owner = remote->second.owner;
                }
                charge_work();
                ++unique_;
                std::size_t shared = 1;
                for (const auto delta : offset)
                    shared *= 2 - std::size_t(delta < 0 ? -delta : delta);
                logical_ += shared;
                ContactClosureView view_a, view_b;
                if (!backend_.closure_views(keys()[i], *neighbour, view_a, view_b))
                    throw std::invalid_argument("Cartesian contact neighbour lies outside the certified frame.");
                view_b.owner = other_owner;
                ClosureRequirements required;
                ++span_evaluations_;
                grade_contact_spans(view_a, view_b, options.maximum_span_ratio, backend_.owner(), required);
                if (view_a.traces_point_independent && view_b.traces_point_independent)
                {
                    charge_work();
                    ++trace_evaluations_;
                    grade_contact_trace_point(
                        view_a, view_b, options.maximum_trace_jump, backend_.owner(), identity, identity, required);
                }
                if (required.first || required.second)
                {
                    allocation_.retain(node_bytes<typename decltype(failing_)::value_type>());
                    failing_.emplace(std::make_pair(keys()[i], *neighbour), required);
                }
            }
        }
        cache_marks();
        evaluated_ratio_ = options.maximum_span_ratio;
        evaluated_trace_jump_ = options.maximum_trace_jump;
        evaluated_ = true;
    }
    [[nodiscard]] std::size_t unique_pairs() const noexcept { return unique_; }
    [[nodiscard]] std::size_t logical_pairs() const noexcept { return logical_; }
    [[nodiscard]] std::size_t span_evaluations() const noexcept { return span_evaluations_; }
    [[nodiscard]] std::size_t trace_evaluations() const noexcept { return trace_evaluations_; }
    [[nodiscard]] std::size_t work() const noexcept { return work_; }
    [[nodiscard]] std::size_t coordinate_points() const noexcept
    {
        std::size_t result = 1;
        for (const auto count : backend_.capability().counts)
            result *= std::size_t(count) + 1;
        return result;
    }
    void collect_marks(std::vector<AdaptiveMark>& out, cartesian_detail::Allocation& lease) const
    {
        if (!evaluated_)
            throw std::invalid_argument("Cartesian marks require a completed snapshot evaluation.");
        lease.reserve(out, cartesian_detail::add(out.size(), marks_.size()));
        out.insert(out.end(), marks_.begin(), marks_.end());
        std::ranges::sort(out, {}, &AdaptiveMark::key);
    }
    /** On-demand row export: eight physical corners derived from the capability;
     * normalized identities equal physical identities on the supported
     * nonperiodic class. The row is returned by value and never enrolled in
     * groups or fans. */
    [[nodiscard]] ContactIncidenceRow make_row(std::size_t id, cartesian_detail::Allocation& lease) const
    {
        const auto& key = keys().at(id);
        return make_row_from(key, ordered_points(id), revision(keys().at(id)), lease);
    }
    /** No new ownership/control allocation: both the row and generated points
     * copy the exact accepted snapshot pin supplied by the halo. */
    [[nodiscard]] static ContactIncidenceRowView row_view(const std::shared_ptr<const void>& snapshot, std::size_t id)
    {
        if (!snapshot)
            throw std::invalid_argument("Cartesian row view needs an accepted snapshot pin.");
        const auto& state = *static_cast<const CartesianContactState*>(snapshot.get());
        if (state.pending_selection_)
            throw std::invalid_argument("Unsealed Cartesian overlay cannot escape through a reader.");
        ContactIncidenceRowView row;
        row.snapshot = snapshot;
        row.key = &state.keys().at(id);
        row.revision = state.revision(state.keys().at(id));
        row.domain = state.domain();
        row.incidence_generation = state.generation_;
        row.points = ContactPointArray::generated(snapshot, id, 8, &read_point);
        return row;
    }
    [[nodiscard]] static ContactPointArray::Reference read_point(
        const void* snapshot, std::size_t id, std::size_t corner)
    {
        const auto& state = *static_cast<const CartesianContactState*>(snapshot);
        const auto points = state.ordered_points(id);
        const auto& point = points.at(corner);
        return {point, point};
    }

private:
    [[nodiscard]] std::array<ContactPointIdentity, 8> ordered_points(std::size_t id) const
    {
        const auto* ancestry = std::get_if<CartesianCellAncestry>(&keys().at(id).ancestry);
        if (!ancestry)
            throw std::invalid_argument("Cartesian row export needs Cartesian ancestry.");
        auto physical = zero_points(std::make_index_sequence<8>{});
        for (std::size_t corner = 0; corner != 8; ++corner)
        {
            std::array<long double, 3> coordinate{};
            for (std::size_t axis = 0; axis != 3; ++axis)
                coordinate[axis] = static_cast<long double>(
                    backend_.capability().coordinate(axis, ancestry->index[axis] + ((corner >> axis) & 1U)));
            physical[corner] = PLIC::details::GeometryIdentityAccess::point(coordinate);
        }
        // The explicit row's point map is ordered by full canonical physical
        // identity. Binary corner numbering is a different order.
        std::ranges::sort(physical);
        return physical;
    }
    /** Neighbour key for one stencil offset, or nullopt outside the closed frame. */
    [[nodiscard]] std::optional<AdaptiveCellKey> offset_key(
        const AdaptiveCellKey& key, const std::array<int, 3>& offset) const
    {
        const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
        if (!ancestry)
            return std::nullopt;
        CartesianCellAncestry neighbour{ancestry->level, ancestry->index};
        for (std::size_t axis = 0; axis != 3; ++axis)
        {
            const auto next = std::int64_t(ancestry->index[axis]) + offset[axis];
            if (next < 0 || std::uint64_t(next) >= capability().counts[axis])
                return std::nullopt;
            neighbour.index[axis] = std::uint64_t(next);
        }
        return AdaptiveCellKey{key.region, neighbour};
    }
    template<std::size_t... I> static std::array<ContactPointIdentity, 8> zero_points(std::index_sequence<I...>)
    {
        return {((void) I, PLIC::details::GeometryIdentityAccess::point({0, 0, 0}))...};
    }
    template<std::size_t... I>
    static ContactPointArray make_points(const std::array<ContactPointIdentity, 8>& physical, std::index_sequence<I...>)
    {
        return ContactPointArray{ContactPointIncidence{physical[I], physical[I]}...};
    }
    ContactIncidenceRow make_row_from(const AdaptiveCellKey& key, const std::array<ContactPointIdentity, 8>& physical,
        std::uint64_t revision, cartesian_detail::Allocation& lease) const
    {
        ContactIncidenceRow row;
        row.key = key;
        row.revision = revision;
        lease.copy(row.domain, domain());
        lease.retain(8 * sizeof(ContactPointIdentity));
        row.incidence_generation = generation_;
        row.points = make_points(physical, std::make_index_sequence<8>{});
        return row;
    }
    static constexpr std::size_t node_header_bytes = 4 * sizeof(void*);
    template<class Pair> static std::size_t node_bytes()
    {
        if (!PLIC::geometry_encoding_capacity_profile_supported())
            throw std::invalid_argument("Unqualified Cartesian tree allocation profile.");
        return cartesian_detail::add(sizeof(Pair), node_header_bytes);
    }
    void cache_marks()
    {
        cartesian_detail::Allocation scratch(allocation_.budget);
        std::vector<std::optional<AdaptiveSplit>> required;
        scratch.reserve(required, keys().size());
        required.resize(keys().size());
        constexpr std::array<std::uint64_t, 3> factors{0, 0, 0};
        auto join = [&](const AdaptiveCellKey& key, const std::optional<AdaptiveSplit>& split)
        {
            if (!split)
                return;
            const auto ordinal = record_id(key);
            join_closure_requirement(std::span<const std::uint64_t>(factors), required[ordinal], std::uint64_t(*split));
        };
        for (const auto& [keys, request] : failing_)
        {
            join(keys.first, request.first);
            join(keys.second, request.second);
        }
        std::size_t count = 0;
        for (const auto& request : required)
            count += bool(request);
        allocation_.reserve(marks_, count);
        for (std::size_t i = 0; i < required.size(); ++i)
            if (required[i])
                marks_.push_back({keys()[i], *required[i]});
        std::ranges::sort(marks_, {}, &AdaptiveMark::key);
    }
    // Keeper and credit precede every allocation they protect in destruction
    // order. The object/control block itself is credited by its exact allocator.
    std::shared_ptr<Budget> keeper_;
    cartesian_detail::Allocation allocation_;
    CartesianContactBackend backend_;
    [[nodiscard]] std::size_t key_count() const noexcept
    {
        return pending_selection_ ? pending_leaves_.size() : layout_ ? layout_->size() : keys_.size();
    }
    [[nodiscard]] const AdaptiveCellKey& key_at(std::size_t i) const
    {
        if (pending_selection_)
            return pending_leaves_.at(i)->key;
        return layout_ ? layout_->at_ordinal(i).key : keys_.at(i);
    }
    std::shared_ptr<const AdaptiveLeafLayout> layout_;
    std::vector<const AdaptiveLeaf*> pending_leaves_;
    bool pending_selection_ = false;
    std::vector<AdaptiveCellKey> keys_;
    std::map<AdaptiveCellKey, std::size_t> index_;
    std::vector<std::uint64_t> revisions_;
    struct RemoteReceipt
    {
        int owner;
        std::uint64_t revision;
        bool authenticated;
    };
    std::map<std::array<std::uint64_t, 3>, RemoteReceipt> remote_;
    std::map<std::pair<AdaptiveCellKey, AdaptiveCellKey>, ClosureRequirements> failing_;
    std::vector<AdaptiveMark> marks_;
    std::size_t shared_allocation_bytes_ = 0;
    std::uint64_t generation_ = 0;
    bool evaluated_ = false;
    double evaluated_ratio_ = 0;
    unsigned evaluated_trace_jump_ = 0;
    std::size_t unique_ = 0, logical_ = 0, span_evaluations_ = 0, trace_evaluations_ = 0, work_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
