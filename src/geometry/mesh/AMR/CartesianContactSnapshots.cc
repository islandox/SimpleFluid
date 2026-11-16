#include "AdaptiveSubfaceSnapshot.hh"
#include "CartesianContactBackend.hh"
#include "CartesianSupportKey.hh"
#include "DistributedAdaptiveIdentityBudget.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/OrthogonalIndexer.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::array<std::array<unsigned, 4>, 6> corners{
    {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}}};
constexpr std::array<std::string_view, 6> boundary_names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};

struct FrozenEnvironment
{
    std::fenv_t previous{};
    explicit FrozenEnvironment(const std::fenv_t& environment)
    {
        if (std::fegetenv(&previous) != 0)
            throw std::invalid_argument("Cannot capture Cartesian subface caller controls.");
        if (std::fesetenv(&environment) != 0)
        {
            (void) std::fesetenv(&previous);
            throw std::invalid_argument("Cannot install Cartesian subface construction controls.");
        }
    }
    ~FrozenEnvironment() { (void) std::fesetenv(&previous); }
};
struct FaceDescriptor
{
    AdaptiveCellKey first;
    std::optional<AdaptiveCellKey> second;
    unsigned face = 0;
    CartesianSupportKey support;
};
FaceDescriptor describe(const CartesianContactState& state, const AdaptiveCellKey& owned, unsigned face)
{
    if (face >= 6)
        throw std::out_of_range("Cartesian source surface is out of range.");
    FaceDescriptor result{owned, {}, face, {state.domain()}};
    const auto axis = face / 2;
    std::array<int, 3> offset{};
    offset[axis] = face % 2 ? 1 : -1;
    if (const auto other = state.neighbor_key(owned, offset))
    {
        if (*other < owned)
        {
            result.first = *other;
            result.second = owned;
            result.face ^= 1;
        }
        else
            result.second = *other;
    }
    const auto& cell = std::get<CartesianCellAncestry>(result.first.ancestry);
    result.support.region = result.first.region;
    result.support.axis = axis;
    result.support.level = cell.level;
    result.support.plane = cell.index[axis] + result.face % 2;
    while (result.support.level && result.support.plane % 2 == 0)
    {
        result.support.plane /= 2;
        --result.support.level;
    }
    for (unsigned a = 0; a < 3; ++a)
        if (a != axis)
            result.support.transverse[a] = cell.index[a] >> (cell.level - result.support.level);
    return result;
}
GeneratedSubfaceIncident incident(
    const CartesianContactState& state, const AdaptiveCellKey& key, unsigned surface, int resident)
{
    const auto [owner, revision] = state.owner_revision(key);
    GeneratedSubfaceIncident result{key, owner, revision, surface, {}};
    if (owner == resident)
    {
        const auto& grid = state.capability().grid;
        const auto& cell = std::get<CartesianCellAncestry>(key.ancestry);
        std::array<unsigned, 3> location{};
        for (unsigned a = 0; a < 3; ++a)
            location[a] = unsigned(cell.index[a] % grid.block_cells[a]);
        location[surface / 2] += surface % 2;
        const OrthogonalIndexer indexer(unsigned(grid.block_cells[0]), unsigned(grid.block_cells[1]),
            unsigned(grid.block_cells[2]), false, false, false);
        result.native = indexer.face_ordinal({location[0], location[1], location[2], std::uint8_t(surface / 2)});
    }
    return result;
}
// Compare the actual little-endian support frame without allocating a key for
// every comparison. Common domain/region bytes are identical within this source.
std::array<char, 40> order_key(const FaceDescriptor& face)
{
    std::array<std::uint64_t, 5> words{face.support.axis, face.support.level, face.support.plane, 0, 0};
    std::size_t next = 3;
    for (unsigned a = 0; a < 3; ++a)
        if (a != face.support.axis)
            words[next++] = face.support.transverse[a];
    std::array<char, 40> result{};
    for (std::size_t i = 0; i < words.size(); ++i)
        for (unsigned byte = 0; byte < 8; ++byte)
            result[i * 8 + byte] = char(words[i] >> (8 * byte));
    return result;
}
std::uint64_t global_emission(const FaceDescriptor& face)
{
    const auto& cell = std::get<CartesianCellAncestry>(face.first.ancestry);
    const auto shift = cell.level - unsigned(face.support.level);
    const auto width = std::uint64_t(1) << shift;
    std::uint64_t result = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (axis != face.support.axis)
        {
            if (result > (std::numeric_limits<std::uint64_t>::max() - (cell.index[axis] % width)) / width)
                throw std::invalid_argument("Cartesian subface emission overflows.");
            result = result * width + cell.index[axis] % width;
        }
    return result;
}
class CartesianSubfacePiece final : public GeneratedSubfacePiece
{
public:
    CartesianSubfacePiece(std::shared_ptr<const CartesianContactState> state, FaceDescriptor face, int resident,
        std::shared_ptr<Budget> arena, std::uint64_t emission)
        : state_(std::move(state)), keeper_(std::move(arena)), memory_(*keeper_), face_(std::move(face)),
          first_(adaptive_detail::incident(*state_, face_.first, face_.face, resident)), emission_(emission)
    {
        if (face_.second)
            second_ = adaptive_detail::incident(*state_, *face_.second, face_.face ^ 1, resident);
        memory_.retain(cartesian_support_key_size(state_->domain()));
        support_ = encode_cartesian_support_key(face_.support);
        if (support_.capacity() != cartesian_support_key_size(state_->domain()))
            throw std::invalid_argument("Cartesian subface support exceeded its admitted extent.");
    }
    const GeneratedSubfaceIncident& incident(bool second) const override
    {
        if (second && !face_.second)
            throw std::invalid_argument("Exterior Cartesian subface has no second incident.");
        return second ? second_ : first_;
    }
    bool has_second() const noexcept override { return bool(face_.second); }
    std::span<const char> support() const noexcept override { return support_; }
    std::span<const char> domain() const noexcept override { return state_->domain(); }
    std::string_view boundary() const noexcept override
    {
        return face_.second ? std::string_view{} : boundary_names[face_.face];
    }
    std::uint64_t generation() const noexcept override { return state_->generation(); }
    std::uint64_t emission() const noexcept override { return emission_; }
    PLIC::PointIdentity<ordinary> point(std::size_t index) const override
    {
        if (index >= 4)
            throw std::out_of_range("Cartesian subface point is out of range.");
        const auto& cell = std::get<CartesianCellAncestry>(face_.first.ancestry);
        const auto bits = corners[face_.face][index];
        PLIC::details::OrdinaryIdentityPoint value{};
        for (unsigned a = 0; a < 3; ++a)
            value[a] = state_->coordinate(a, cell.index[a] + ((bits >> a) & 1));
        return PLIC::details::GeometryIdentityAccess::point(value);
    }
    MaterializedCanonicalSubface materialize(Budget& budget, IdentityBudget& identities) const override
    {
        const FrozenEnvironment environment(state_->capability().environment);
        cartesian_detail::Allocation scratch(budget);
        using Cell = PLIC::ConvexCellGeometry<ordinary>;
        using Face = PLIC::PlanarFace<ordinary>;
        scratch.retain(Cell::box_construction_capacity_bound());
        scratch.retain(cartesian_detail::product(2, Face::construction_capacity_bound(4)));
        const auto& cell = std::get<CartesianCellAncestry>(face_.first.ancestry);
        PLIC::GeometryPoint lower{}, upper{};
        for (unsigned a = 0; a < 3; ++a)
        {
            lower[a] = state_->coordinate(a, cell.index[a]);
            upper[a] = state_->coordinate(a, cell.index[a] + 1);
        }
        const auto shape = Cell::box(lower, upper);
        auto geometry = shape->face_geometry(face_.face);
        const auto reversed = geometry->reversed();
        scratch.retain(cartesian_detail::product(geometry->encoded_incidence_size(), 6));
        scratch.retain(geometry->encoding_scratch_bytes());
        const auto forward = PLIC::canonical_face_key(*geometry), backward = PLIC::canonical_face_key(*reversed);
        const auto& canonical = std::min(forward, backward);
        const auto bytes = cartesian_detail::add(16, cartesian_detail::add(support_.size(), canonical.size()));
        const auto retained = cartesian_detail::add(bytes, sizeof(Face));
        scratch.retain(retained);
        Writer identity;
        identity.data.reserve(bytes);
        if (identity.data.capacity() != bytes)
            throw std::invalid_argument("Cartesian canonical identity exceeded admission.");
        identity.bytes(support_);
        identity.bytes(canonical);
        identities.face(budget, *geometry);
        scratch.bytes -= retained;
        return {std::move(geometry), std::move(identity.data)};
    }

private:
    std::shared_ptr<const CartesianContactState> state_;
    std::shared_ptr<Budget> keeper_;
    cartesian_detail::Allocation memory_;
    FaceDescriptor face_;
    GeneratedSubfaceIncident first_, second_;
    std::uint64_t emission_;
    AdaptiveBytes support_;
};
} // namespace

class CartesianSubfaceSource final : public GeneratedSubfaceSnapshot,
                                     public std::enable_shared_from_this<CartesianSubfaceSource>
{
public:
    CartesianSubfaceSource(std::shared_ptr<const CartesianContactState> state,
        std::shared_ptr<const AdaptiveLeafLayout> layout, int resident, std::shared_ptr<Budget> arena)
        : state_(std::move(state)), layout_(std::move(layout)), arena_(std::move(arena)), resident_(resident)
    {
    }
    void visit_pieces(const CanonicalSubfaceEmit& emit) const override
    {
        cartesian_detail::Allocation scratch(*arena_);
        std::vector<std::size_t> selected;
        scratch.reserve(selected, cartesian_detail::product(6, layout_->size()));
        for (std::size_t i = 0; i < layout_->size(); ++i)
            for (unsigned face = 0; face < 6; ++face)
            {
                const auto& key = layout_->at_ordinal(i).key;
                const auto description = describe(*state_, key, face);
                if (description.first != key && state_->owns(description.first))
                    continue;
                selected.push_back(6 * i + face);
            }
        const auto description = [&](std::size_t code)
        { return describe(*state_, layout_->at_ordinal(code / 6).key, unsigned(code % 6)); };
        std::ranges::sort(selected,
            [&](std::size_t a, std::size_t b)
            {
                const auto first = description(a), second = description(b);
                const auto x = order_key(first), y = order_key(second);
                if (x != y)
                    return x < y;
                return first.first < second.first;
            });
        for (const auto code : selected)
        {
            const auto current = description(code);
            emit_piece(current, global_emission(current), false, emit);
        }
    }
    void visit_face_pieces(
        const AdaptiveCellKey& key, std::uint64_t surface, const CanonicalSubfaceEmit& emit) const override
    {
        if (surface >= 6 || !layout_->contains(key))
            return;
        const auto current = describe(*state_, key, unsigned(surface));
        emit_piece(current, global_emission(current), current.first != key, emit);
    }
    void set_allocation_bytes(std::size_t bytes) noexcept { allocation_bytes_ = bytes; }
    void visit_owners(const SubfaceOwnerVisit& visit) const override
    {
        visit(std::static_pointer_cast<const void>(shared_from_this()), arena_.get(), allocation_bytes_);
        visit(std::static_pointer_cast<const void>(state_), state_->allocation_budget(), state_->admitted_bytes());
        visit(std::static_pointer_cast<const void>(layout_), nullptr, 0);
    }
    bool imports_source_owner(const void* value) const noexcept override
    {
        return value == state_.get() || value == layout_.get();
    }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visit) const override
    {
        contact_payload_object(
            visit, this, GeometryContactPayloadRole::CartesianSubfaceSourceObject, sizeof(*this), true);
        state_->visit_borrowed_payload(visit);
    }

private:
    void emit_piece(FaceDescriptor face, std::uint64_t emission, bool side, const CanonicalSubfaceEmit& emit) const
    {
        std::size_t bytes = 0;
        const auto piece = std::allocate_shared<CartesianSubfacePiece>(
            CartesianContactAllocator<CartesianSubfacePiece>(*arena_, arena_, bytes), state_, std::move(face),
            resident_, arena_, emission);
        emit(CanonicalSubfaceView(piece, side));
    }
    std::shared_ptr<const CartesianContactState> state_;
    std::shared_ptr<const AdaptiveLeafLayout> layout_;
    std::shared_ptr<Budget> arena_;
    int resident_;
    std::size_t allocation_bytes_ = 0;
};

std::shared_ptr<const AdaptiveSubfaceSnapshot> AdaptiveSubfaceSnapshot::cartesian(
    std::shared_ptr<const CartesianContactState> state, std::shared_ptr<const AdaptiveLeafLayout> layout,
    ContactHaloContext context, GeometryConstructionControls controls, int resident, Budget& budget,
    std::shared_ptr<Budget> arena)
{
    if (!state || !layout || !arena || arena.get() != &budget || resident < 0)
        throw std::invalid_argument("Cartesian subface publication needs authenticated immutable owners.");
    cartesian_detail::Allocation memory(budget);
    memory.retain(sizeof(AdaptiveSubfaceSnapshot) + 128);
    auto result =
        std::shared_ptr<AdaptiveSubfaceSnapshot>(new AdaptiveSubfaceSnapshot(context, controls, resident, arena));
    result->charge_ = std::exchange(memory.bytes, 0);
    std::size_t bytes = 0;
    auto source = std::allocate_shared<CartesianSubfaceSource>(
        CartesianContactAllocator<CartesianSubfaceSource>(budget, arena, bytes), state, layout, resident, arena);
    source->set_allocation_bytes(bytes);
    for (const auto& leaf : *layout)
        for (unsigned face = 0; face < 6; ++face)
        {
            const auto description = describe(*state, leaf.key, face);
            if (description.first == leaf.key || !state->owns(description.first))
                ++result->size_;
        }
    result->index_bytes_ = result->charge_ + bytes;
    result->exclusive_payload_bytes_ = result->index_bytes_;
    result->generated_ = std::move(source);
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
