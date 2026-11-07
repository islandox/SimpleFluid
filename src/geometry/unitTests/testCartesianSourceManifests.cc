#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/CartesianSourceInventory.hh"
#include "geometry/mesh/AMR/CartesianSourceManifests.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
static_assert(!std::is_default_constructible_v<CartesianSourceManifestAdmission>);
static_assert(
    std::is_same_v<decltype(CartesianSourceManifestSet::validate_untrusted_wire(std::declval<const AMRGrid&>(),
                       std::declval<ContactHaloContext>(), std::declval<std::uint64_t>(),
                       std::declval<std::span<const OwnerPacket>>(), 2, std::declval<Budget&>())),
        void>);
// Independent literal protocol writer. The decoder-only seam cannot turn these
// caller bytes into an authorized manifest set or a procedural geometry answer.
AdaptiveBytes descriptor(const AMRGrid& value)
{
    Writer writer;
    writer.word(0x5346434152544b31ULL);
    for (unsigned a = 0; a != 3; ++a)
    {
        writer.word(value.root_blocks[a]);
        writer.word(value.block_cells[a]);
        writer.word(std::bit_cast<std::uint64_t>(value.origin[a]));
        writer.word(std::bit_cast<std::uint64_t>(value.extent[a]));
        writer.word(value.periodic[a]);
    }
    writer.word(value.max_level);
    writer.word(value.region_id);
    writer.word(value.material_id);
    return std::move(writer.data);
}
OwnerPacket receipt(const AMRGrid& value, ContactHaloContext context, int owner, std::uint64_t generation,
    const std::shared_ptr<const CartesianIncidenceSource>& source = {})
{
    Writer writer;
    writer.word(0x534643534d414e31ULL);
    writer.word(1);
    for (auto word : {context.epoch, context.topology, context.partition, context.round, context.transaction})
        writer.word(word);
    writer.word(owner);
    writer.word(source ? source->cell_count() : 0);
    writer.word(source ? source->version() : 0);
    writer.word(!source);
    writer.word(generation);
    writer.bytes(descriptor(value));
    writer.bytes(source ? std::span<const char>(source->identity()) : std::span<const char>{});
    return {owner, std::move(writer.data)};
}
TEST(CartesianSourceManifests, CompleteInventoryRequiresEveryEmptyOwnerAndExactContextGridVersionAndBytes)
{
    const auto value = grid();
    const ContactHaloContext context{1, 2, 3, 4, 5};
    const auto native = leaves(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 7, context, *budget, budget);
    const std::vector<OwnerPacket> valid{receipt(value, context, 0, 11, source), receipt(value, context, 1, 11)};
    auto validate = [&](std::span<const OwnerPacket> packets)
    { CartesianSourceManifestSet::validate_untrusted_wire(value, context, 11, packets, 2, *budget); };
    const auto before = budget->used;
    EXPECT_NO_THROW(validate(valid));
    EXPECT_EQ(budget->used, before);
    EXPECT_THROW(validate(std::span(valid).first(1)), std::exception);
    auto duplicate = valid;
    duplicate[1] = duplicate[0];
    EXPECT_THROW(validate(duplicate), std::exception);
    for (const auto byte : {std::size_t(0), std::size_t(8), std::size_t(16), std::size_t(56), std::size_t(64),
             std::size_t(72), std::size_t(80), std::size_t(88), std::size_t(104)})
    {
        auto corrupted = valid;
        corrupted.front().bytes[byte] ^= 1;
        EXPECT_THROW(validate(corrupted), std::exception) << byte;
        EXPECT_EQ(budget->used, before);
    }
    auto malformed = valid;
    malformed.front().bytes.back() ^= 1;
    EXPECT_THROW(validate(malformed), std::exception);
    auto trailing = valid;
    trailing[1].bytes.push_back('x');
    EXPECT_THROW(validate(trailing), std::exception);
    for (std::size_t cut : {std::size_t(0), std::size_t(7), std::size_t(95), std::size_t(263)})
    {
        auto truncated = valid;
        truncated.front().bytes.resize(cut);
        EXPECT_THROW(validate(truncated), std::exception) << cut;
        EXPECT_EQ(budget->used, before);
    }
    auto reversed = valid;
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_NO_THROW(validate(reversed)); // Per-peer completion order is unrelated to identity authority.
    EXPECT_EQ(budget->used, before);
    Budget rejected{0};
    EXPECT_THROW(
        CartesianSourceManifestSet::validate_untrusted_wire(value, context, 11, valid, 2, rejected), std::exception);
    EXPECT_EQ(rejected.used, 0);
}
// Schema 1 above remains a decoder compatibility case. Schema 2 below carries
// the full current inventory, including unchanged sources from older contexts.
// These independent wire builders never issue the session-only Admission.
struct LiteralBatch
{
    AdaptiveBytes original;
    std::uint64_t cells;
    std::vector<CartesianIncidenceRange> ranges;
};
struct LiteralInventory
{
    ContactHaloContext context;
    int owner;
    std::uint64_t version, cells;
    std::vector<LiteralBatch> batches;
    std::uint64_t reserved = 0;
};
AdaptiveBytes inventory_bytes(const LiteralInventory& value)
{
    Writer writer;
    writer.word(0x534643494e563031ULL);
    writer.word(1);
    for (auto word : {value.context.epoch, value.context.topology, value.context.partition, value.context.round,
             value.context.transaction})
        writer.word(word);
    writer.word(value.owner);
    writer.word(value.version);
    writer.word(value.cells);
    writer.word(value.batches.size());
    writer.word(value.reserved);
    // The 96-byte header is followed by length-framed full native identities,
    // active cell/range counts, and exactly seven uint64 words per active box.
    for (const auto& batch : value.batches)
    {
        writer.bytes(batch.original);
        writer.word(batch.cells);
        writer.word(batch.ranges.size());
        for (const auto& range : batch.ranges)
        {
            writer.word(range.level);
            for (auto word : range.lower)
                writer.word(word);
            for (auto word : range.upper)
                writer.word(word);
        }
    }
    return std::move(writer.data);
}
OwnerPacket inventory_receipt(const AMRGrid& value, ContactHaloContext context, int owner, std::uint64_t generation,
    std::uint64_t version, std::uint64_t cells, std::span<const char> identity, std::uint64_t empty)
{
    Writer writer;
    writer.word(0x534643534d414e31ULL);
    writer.word(2);
    for (auto word : {context.epoch, context.topology, context.partition, context.round, context.transaction})
        writer.word(word);
    writer.word(owner);
    writer.word(cells);
    writer.word(version);
    writer.word(empty);
    writer.word(generation);
    writer.bytes(descriptor(value));
    writer.bytes(identity);
    return {owner, std::move(writer.data)};
}
// Schema 3 adds only sparse explicit contributors after the unchanged schema-2
// native inventory. Literal framing is independent of production serialization;
// support keys and original packet sizes below come from the actual kernel.
struct LiteralExplicitMember
{
    AdaptiveBytes support;
    std::uint64_t encoded_size;
};
struct LiteralExplicitCell
{
    AdaptiveBytes key;
    std::uint64_t revision;
    std::vector<LiteralExplicitMember> members;
};
OwnerPacket explicit_receipt(
    const AMRGrid& value, const LiteralInventory& native, std::span<const LiteralExplicitCell> explicit_cells)
{
    Writer writer;
    writer.word(0x534643534d414e31ULL);
    writer.word(3);
    for (auto word : {native.context.epoch, native.context.topology, native.context.partition, native.context.round,
             native.context.transaction})
        writer.word(word);
    writer.word(native.owner);
    writer.word(native.cells);
    writer.word(native.version);
    writer.word(native.cells == 0);
    writer.word(native.version);
    writer.bytes(descriptor(value));
    writer.bytes(inventory_bytes(native));
    writer.word(explicit_cells.size());
    for (const auto& cell : explicit_cells)
    {
        writer.bytes(cell.key);
        writer.word(cell.revision);
        writer.word(cell.members.size());
        for (const auto& member : cell.members)
        {
            writer.bytes(member.support);
            writer.word(member.encoded_size);
        }
    }
    return {native.owner, std::move(writer.data)};
}
CartesianSourceInventory::CapturedSource captured(std::shared_ptr<const CartesianIncidenceSource> source)
{
    CartesianSourceInventory::CapturedSource result;
    result.source = std::move(source);
    if (std::fegetenv(&result.environment))
        throw std::runtime_error("Cannot capture manifest test source environment.");
    return result;
}
void replace_word(AdaptiveBytes& bytes, std::size_t offset, std::uint64_t value)
{
    // Bounds checking keeps a malformed test offset from becoming undefined behavior.
    for (unsigned byte = 0; byte != 8; ++byte)
        bytes.at(offset + byte) = char(value >> (byte * 8));
}
class CartesianSourceInventoryManifests : public testing::Test
{
protected:
    RestoreEnvironment restore;
    AMRGrid value = grid();
    ContactHaloContext context{1, 2, 3, 4, 5};
    static constexpr std::uint64_t generation = 11;
    std::shared_ptr<Budget> budget = std::make_shared<Budget>(Budget{ample});
    std::shared_ptr<CartesianAuthorityArena> arena;
    std::array<AdaptiveLeaf, 2> coarse;
    std::shared_ptr<const CartesianIncidenceSource> original;
    std::shared_ptr<const CartesianSourceInventory> inventory, empty_inventory;
    AdaptiveBytes original_identity, immutable_inventory;
    LiteralInventory literal{context, 0, generation, 2, {}};
    std::vector<OwnerPacket> valid;

    void SetUp() override
    {
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
        arena = CartesianGroupAuthority::make_arena(*budget, budget);
        coarse = {cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {0, 0, 0}}}, {'d'}),
            cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {1, 0, 0}}}, {'d'})};
        original = CartesianIncidenceSource::make(value, coarse, {'d'}, 0, 7, context, *budget, budget);
        original_identity = original->identity();
        const std::array sources{captured(original)};
        inventory = CartesianSourceInventory::make(context, 0, generation, sources, *budget, budget, arena);
        empty_inventory = CartesianSourceInventory::make(context, 1, generation, {}, *budget, budget, arena);
        immutable_inventory = inventory->encode_identity(*budget).bytes();
        // Independent expected membership: two touching level-0 cells form this box.
        literal.batches = {{original_identity, 2, {{0, {0, 0, 0}, {2, 1, 1}}}}};
        valid = {packet(literal), packet({context, 1, generation, 0, {}})};
    }
    OwnerPacket packet(const LiteralInventory& expected) const
    {
        return inventory_receipt(value, expected.context, expected.owner, generation, expected.version, expected.cells,
            inventory_bytes(expected), expected.cells == 0);
    }
    void validate(std::span<const OwnerPacket> packets, int ranks = 2)
    {
        CartesianSourceManifestSet::validate_untrusted_wire(value, context, generation, packets, ranks, *budget);
    }
    void expect_rejected(const std::vector<OwnerPacket>& packets)
    {
        const auto retained = budget->used;
        const auto immutable_packets = packets;
        EXPECT_THROW(validate(packets), std::exception);
        EXPECT_EQ(budget->used, retained);
        ASSERT_EQ(packets.size(), immutable_packets.size());
        for (std::size_t i = 0; i != packets.size(); ++i)
        {
            EXPECT_EQ(packets[i].peer, immutable_packets[i].peer);
            EXPECT_EQ(packets[i].bytes, immutable_packets[i].bytes);
        }
        EXPECT_EQ(original->identity(), original_identity);
        EXPECT_EQ(inventory->encode_identity(*budget).bytes(), immutable_inventory);
        EXPECT_EQ(std::fegetround(), FE_TONEAREST);
    }
    void expect_bad_inventory(const LiteralInventory& bad)
    {
        auto packets = valid;
        packets.front() = packet(bad);
        expect_rejected(packets);
    }
};
TEST_F(CartesianSourceInventoryManifests, Schema2CarriesExactCurrentInventoryAndEveryEmptyOwner)
{
    EXPECT_EQ(inventory->encode_identity(*budget).bytes(), inventory_bytes(literal));
    EXPECT_EQ(empty_inventory->encode_identity(*budget).bytes(), inventory_bytes({context, 1, generation, 0, {}}));
    EXPECT_EQ(valid.front().bytes.size(), 264 + inventory->identity_size());
    EXPECT_EQ(valid.back().bytes.size(), 264 + 96);
    EXPECT_GT(empty_inventory->version(), 0);
    EXPECT_EQ(empty_inventory->version(), generation);
    const auto retained = budget->used;
    EXPECT_NO_THROW(validate(valid));
    EXPECT_EQ(budget->used, retained);
    auto reversed = valid;
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_NO_THROW(validate(reversed));
    EXPECT_EQ(budget->used, retained);
    // Even an owner with no native source must provide its bound current inventory.
    const std::vector<OwnerPacket> all_empty{
        packet({context, 0, generation, 0, {}}), packet({context, 1, generation, 0, {}})};
    EXPECT_NO_THROW(validate(all_empty));
    EXPECT_EQ(budget->used, retained);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
}
TEST_F(CartesianSourceInventoryManifests, DerivedInventoryKeepsOriginalContextAndNearestSourceBesideUpwardChildren)
{
    auto next = context;
    ++next.round;
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    std::vector<AdaptiveLeaf> fine;
    for (std::uint64_t z = 0; z != 2; ++z)
        for (std::uint64_t y = 0; y != 2; ++y)
            for (std::uint64_t x = 0; x != 2; ++x)
                fine.push_back(
                    cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{1, {x, y, z}}}, {'d'}));
    auto children = CartesianIncidenceSource::make(value, fine, {'d'}, 0, 8, next, *budget, budget);
    const auto child_identity = children->identity();
    const std::array added{captured(children)};
    const std::array retired{CartesianSourceInventory::Retirement{coarse.front().key, coarse.front().revision}};
    auto derived = inventory->derive(next, 12, retired, added, *budget, budget, arena);
    auto empty_next = CartesianSourceInventory::make(next, 1, 12, {}, *budget, budget, arena);
    const LiteralInventory expected{next, 0, 12, 9,
        {{original_identity, 1, {{0, {1, 0, 0}, {2, 1, 1}}}}, {child_identity, 8, {{1, {0, 0, 0}, {2, 2, 2}}}}}};
    EXPECT_EQ(derived->encode_identity(*budget).bytes(), inventory_bytes(expected));
    ASSERT_EQ(derived->batches().size(), 2);
    EXPECT_EQ(derived->batches().front().source(), original);
    EXPECT_EQ(derived->batches().front().source()->context(), context);
    EXPECT_NE(original->evaluation_environment(), children->evaluation_environment());
    EXPECT_EQ(original->identity(), original_identity);
    EXPECT_EQ(inventory->encode_identity(*budget).bytes(), immutable_inventory);
    EXPECT_EQ(derived->batches().front().source()->identity(), original_identity);
    EXPECT_EQ(derived->batches().back().source()->identity(), child_identity);
    EXPECT_EQ(derived->batches().front().source()->version(), 7);
    EXPECT_EQ(derived->batches().back().source()->version(), 8);
    EXPECT_TRUE(derived->contains(coarse.back().key));
    EXPECT_FALSE(derived->contains(coarse.front().key));
    const std::vector<OwnerPacket> packets{
        inventory_receipt(value, next, 0, 12, 12, 9, derived->encode_identity(*budget).bytes(), 0),
        inventory_receipt(value, next, 1, 12, 12, 0, empty_next->encode_identity(*budget).bytes(), 1)};
    const auto retained = budget->used;
    EXPECT_NO_THROW(CartesianSourceManifestSet::validate_untrusted_wire(value, next, 12, packets, 2, *budget));
    EXPECT_EQ(budget->used, retained);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    // Parsed signatures are metadata, never permission to activate a source's
    // local fenv_t; validation also works in another caller rounding environment.
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    EXPECT_NO_THROW(CartesianSourceManifestSet::validate_untrusted_wire(value, next, 12, packets, 2, *budget));
    EXPECT_EQ(budget->used, retained);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
    EXPECT_EQ(original->identity(), original_identity);
    EXPECT_EQ(children->identity(), child_identity);
}
TEST_F(CartesianSourceInventoryManifests, RejectsIncompleteOwnerReceiptsAndStaleOuterBindings)
{
    expect_rejected({valid.front()});
    expect_rejected({valid.back()});
    expect_rejected({});
    auto packets = valid;
    packets.back() = packets.front();
    expect_rejected(packets);
    for (int peer : {-1, 2})
    {
        SCOPED_TRACE(peer);
        packets = valid;
        packets.back().peer = peer;
        expect_rejected(packets);
    }
    // Outer schema: magic0, schema8, context16..48, owner56, cells64,
    // inventory-version72, empty80, generation88, grid-length96, grid104.
    for (auto offset : {std::size_t(0), std::size_t(8), std::size_t(16), std::size_t(24), std::size_t(32),
             std::size_t(40), std::size_t(48), std::size_t(56), std::size_t(64), std::size_t(72), std::size_t(80),
             std::size_t(88), std::size_t(104)})
    {
        SCOPED_TRACE(offset);
        packets = valid;
        packets.front().bytes[offset] ^= 1;
        expect_rejected(packets);
    }
    packets = valid;
    replace_word(packets.back().bytes, 80, 0); // Empty owner presented as nonempty.
    expect_rejected(packets);
    packets = valid;
    replace_word(packets.front().bytes, 80, 2); // Empty flag is a canonical bit.
    expect_rejected(packets);
    auto stale = literal;
    stale.version = generation - 1; // Matching inner/outer versions still fail this generation.
    expect_bad_inventory(stale);
    stale.version = 0;
    expect_bad_inventory(stale);
    packets = valid;
    packets.back() = packet({context, 1, 0, 0, {}});
    expect_rejected(packets);
    auto foreign_grid = value;
    foreign_grid.origin[0] = 0.125;
    packets = valid;
    packets.front() = inventory_receipt(
        foreign_grid, context, 0, generation, generation, 2, inventory->encode_identity(*budget).bytes(), 0);
    expect_rejected(packets); // Compare with the actual supplied trusted grid.
}
TEST_F(CartesianSourceInventoryManifests, RejectsStaleMalformedAndTruncatedNestedInventory)
{
    // Inventory header: magic0, schema8, context16..48, owner56, version64,
    // active-count72, batch-count80, reserved88; full identity begins at104.
    for (auto offset :
        {std::size_t(0), std::size_t(8), std::size_t(16), std::size_t(24), std::size_t(32), std::size_t(40),
            std::size_t(48), std::size_t(56), std::size_t(64), std::size_t(72), std::size_t(80), std::size_t(88)})
    {
        SCOPED_TRACE(offset);
        auto bad = inventory->encode_identity(*budget).bytes();
        bad[offset] ^= 1;
        auto packets = valid;
        packets.front() = inventory_receipt(value, context, 0, generation, generation, 2, bad, 0);
        expect_rejected(packets);
    }
    const auto source_end = 104 + original_identity.size();
    const auto range_start = source_end + 16;
    for (auto cut : {std::size_t(0), std::size_t(7), std::size_t(15), std::size_t(55), std::size_t(95), std::size_t(96),
             std::size_t(103), source_end - 1, source_end, source_end + 7, source_end + 15, range_start,
             range_start + 7, range_start + 31, inventory->identity_size() - 1})
    {
        SCOPED_TRACE(cut);
        auto truncated = inventory->encode_identity(*budget).bytes();
        truncated.resize(cut);
        auto packets = valid;
        packets.front() = inventory_receipt(value, context, 0, generation, generation, 2, truncated, 0);
        expect_rejected(packets);
    }
    auto trailing = inventory->encode_identity(*budget).bytes();
    trailing.push_back('x');
    auto packets = valid;
    packets.front() = inventory_receipt(value, context, 0, generation, generation, 2, trailing, 0);
    expect_rejected(packets);
    for (auto offset : {std::size_t(96), source_end, source_end + 8})
    {
        SCOPED_TRACE(offset);
        auto bad = inventory->encode_identity(*budget).bytes();
        replace_word(bad, offset, 0); // No native identity, no active cells, or no active ranges.
        packets = valid;
        packets.front() = inventory_receipt(value, context, 0, generation, generation, 2, bad, 0);
        expect_rejected(packets);
    }
    for (auto cut : {std::size_t(0), std::size_t(7), std::size_t(95), std::size_t(103), std::size_t(255),
             std::size_t(263), valid.front().bytes.size() - 1})
    {
        SCOPED_TRACE(cut);
        packets = valid;
        packets.front().bytes.resize(cut);
        expect_rejected(packets);
    }
    packets = valid;
    packets.back().bytes.push_back('x');
    expect_rejected(packets);
}
TEST_F(CartesianSourceInventoryManifests, RejectsInvalidFullNativeSourceAndUncoveredOrRepeatedSelectors)
{
    // Native identity: family0, four environment words8..32, grid-length40,
    // full 152-byte grid48, domain-length200, domain208, then owner/version.
    const auto native_owner = std::size_t(208) + original->domain().size();
    for (auto offset : {std::size_t(0), std::size_t(48), native_owner})
    {
        SCOPED_TRACE(offset);
        auto bad = literal;
        bad.batches.front().original[offset] ^= 1;
        expect_bad_inventory(bad); // Invalid family, trusted grid, or original owner.
    }
    auto foreign_grid = value;
    foreign_grid.origin[0] = 0.125;
    auto foreign_descriptor = descriptor(foreign_grid);
    auto foreign_source = literal;
    std::copy(
        foreign_descriptor.begin(), foreign_descriptor.end(), foreign_source.batches.front().original.begin() + 48);
    expect_bad_inventory(foreign_source); // Valid native family, different complete trusted grid descriptor.
    for (auto cut : {std::size_t(0), std::size_t(7), std::size_t(39), std::size_t(47), std::size_t(199),
             std::size_t(207), native_owner, original_identity.size() - 1})
    {
        SCOPED_TRACE(cut);
        auto bad = literal;
        bad.batches.front().original.resize(cut);
        expect_bad_inventory(bad);
    }
    auto bad = literal;
    bad.batches.front().ranges.front() = {0, {0, 1, 0}, {2, 2, 1}};
    expect_bad_inventory(bad); // In-grid ancestry, outside the original source's occupied boxes.
    bad = literal;
    bad.batches.front().ranges.front().level = value.max_level + 1;
    expect_bad_inventory(bad);
    bad = literal;
    bad.batches.front().ranges.front().upper[0] = 3;
    expect_bad_inventory(bad); // Outside the grid itself.
    bad = literal;
    bad.batches.front().ranges.front().upper[0] = 0;
    expect_bad_inventory(bad); // Empty range.
    bad = literal;
    bad.batches.front().cells = 1;
    expect_bad_inventory(bad); // Active batch count differs from its complete selectors.
    bad = literal;
    bad.cells = 3;
    expect_bad_inventory(bad); // Owner total differs from active batch sum.
    bad = literal;
    bad.cells = bad.batches.front().cells = 4;
    bad.batches.front().ranges.push_back(bad.batches.front().ranges.front());
    expect_bad_inventory(bad); // Duplicate membership within one original source.
    bad = literal;
    bad.cells = 4;
    bad.batches.push_back(bad.batches.front());
    expect_bad_inventory(bad); // Duplicate membership across separate source batches.
    const auto child = cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{1, {0, 0, 0}}}, {'d'});
    auto child_source =
        CartesianIncidenceSource::make(value, std::span(&child, 1), {'d'}, 0, 8, context, *budget, budget);
    bad = literal;
    bad.cells = 3;
    bad.batches.push_back({child_source->identity(), 1, {{1, {0, 0, 0}, {1, 1, 1}}}});
    expect_bad_inventory(bad); // Different levels still overlap as ancestor and descendant.
}
TEST_F(CartesianSourceInventoryManifests, RejectedTinyBudgetsRestoreAdmissionAndImmutableInputs)
{
    for (auto limit : {std::size_t(0), std::size_t(1), std::size_t(64)})
    {
        SCOPED_TRACE(limit);
        Budget rejected{limit};
        EXPECT_THROW(
            CartesianSourceManifestSet::validate_untrusted_wire(value, context, generation, valid, 2, rejected),
            std::exception);
        EXPECT_EQ(rejected.used, 0);
        EXPECT_EQ(original->identity(), original_identity);
        EXPECT_EQ(inventory->encode_identity(*budget).bytes(), immutable_inventory);
        EXPECT_EQ(empty_inventory->encode_identity(*budget).bytes(), inventory_bytes({context, 1, generation, 0, {}}));
        EXPECT_EQ(std::fegetround(), FE_TONEAREST);
    }
}
TEST_F(CartesianSourceInventoryManifests, DuplicateOriginalBoxesCannotProveCoverageOfAnAdjacentMissingCell)
{
    // Keep original_cells=2 and all exact native header/axis/revision bytes.
    // Replace its [0,2)x[0,1)x[0,1) source box with two copies of the first
    // unit cell. A naive intersection sum equals active volume2 but covers
    // only x=0, so x=1 must never acquire an authenticated native recipe.
    Reader reader{original_identity};
    for (unsigned i = 0; i != 5; ++i)
        (void) reader.word();
    (void) reader.bytes_view();
    (void) reader.bytes_view();
    for (unsigned i = 0; i != 9; ++i)
        (void) reader.word(); // owner/version/context/base revision/cells.
    const auto range_count_offset = reader.offset;
    const auto old_ranges = reader.word();
    ASSERT_EQ(old_ranges, 1);
    const auto old_ranges_end = reader.offset + 56 * old_ranges;
    const auto forge = [&](bool ancestor)
    {
        Writer writer;
        writer.data.assign(original_identity.begin(), original_identity.begin() + range_count_offset);
        writer.word(2);
        for (unsigned copy = 0; copy != 2; ++copy)
        {
            writer.word(ancestor && copy ? 1 : 0);
            for (unsigned axis = 0; axis != 3; ++axis)
                writer.word(0);
            for (unsigned axis = 0; axis != 3; ++axis)
                writer.word(1);
        }
        writer.data.insert(writer.data.end(), original_identity.begin() + old_ranges_end, original_identity.end());
        auto bad = literal;
        bad.batches.front().original = std::move(writer.data);
        expect_bad_inventory(bad);
    };
    forge(false);
    forge(true); // The same proof must also reject cross-level ancestor overlap.
}
TEST_F(CartesianSourceInventoryManifests, RejectsUnorderedDuplicateAndForeignOriginalRevisionOverrides)
{
    Reader last{std::span<const char>(original_identity).last(8)};
    ASSERT_EQ(last.word(), 0); // This literal fixture has no original overrides.
    const auto check = [&](std::span<const CartesianCellAncestry> overrides)
    {
        Writer writer;
        writer.data.assign(original_identity.begin(), original_identity.end() - 8);
        writer.word(overrides.size());
        for (const auto& cell : overrides)
        {
            writer.word(cell.level);
            for (const auto index : cell.index)
                writer.word(index);
            writer.word(7);
        }
        auto bad = literal;
        bad.batches.front().original = std::move(writer.data);
        expect_bad_inventory(bad);
    };
    const CartesianCellAncestry first{0, {0, 0, 0}}, second{0, {1, 0, 0}}, foreign{0, {0, 1, 0}};
    check(std::array{second, first});
    check(std::array{first, first});
    check(std::array{foreign}); // In-grid but absent from this source's occupied boxes.
}
class CartesianSourceExplicitManifests : public CartesianSourceInventoryManifests
{
protected:
    LiteralInventory partial;
    LiteralExplicitCell explicit_a;
    std::vector<OwnerPacket> complete;

    void SetUp() override
    {
        CartesianSourceInventoryManifests::SetUp();
        partial = literal;
        partial.cells = partial.batches.front().cells = 1;
        partial.batches.front().ranges = {{0, {1, 0, 0}, {2, 1, 1}}};
        const auto kernel = cartesian_adaptive_kernel(value);
        auto materialized = coarse.front();
        materialized.geometry = kernel->geometry(materialized);
        materialized.surfaces = kernel->surfaces(materialized);
        ASSERT_EQ(materialized.surfaces.size(), 6);
        explicit_a = {encode_adaptive_key(materialized.key), materialized.revision, {}};
        for (std::size_t ordinal = 0; ordinal != materialized.surfaces.size(); ++ordinal)
        {
            const auto original_packet = encode_support_surface(materialized, ordinal);
            explicit_a.members.push_back({materialized.surfaces[ordinal].key, original_packet.size()});
        }
        complete = {explicit_receipt(value, partial, std::span(&explicit_a, 1)),
            explicit_receipt(value, {context, 1, generation, 0, {}}, {})};
    }
    void expect_bad_cell(const LiteralExplicitCell& bad)
    {
        auto packets = complete;
        packets.front() = explicit_receipt(value, partial, std::span(&bad, 1));
        expect_rejected(packets);
    }
    void expect_bad_member(std::size_t offset, std::uint64_t word)
    {
        auto bad = explicit_a;
        replace_word(bad.members.front().support, offset, word);
        expect_bad_cell(bad);
    }
};
TEST_F(CartesianSourceExplicitManifests, Schema3CarriesSparseExplicitHeadersBesidePartialAndEmptyNativeInventories)
{
    const auto retained = budget->used;
    EXPECT_EQ(complete.front().bytes.size(), 264 + inventory_bytes(partial).size() + 8 + 8 + explicit_a.key.size() +
                                                 16 + 6 * (16 + explicit_a.members.front().support.size()));
    EXPECT_NO_THROW(validate(complete));
    EXPECT_EQ(budget->used, retained);
    auto reversed = complete;
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_NO_THROW(validate(reversed));
    EXPECT_EQ(budget->used, retained);
    // Native-empty is independent of sparse explicit contribution count.
    auto explicit_only = complete;
    explicit_only.front() = explicit_receipt(value, {context, 0, generation, 0, {}}, std::span(&explicit_a, 1));
    EXPECT_NO_THROW(validate(explicit_only));
    EXPECT_EQ(budget->used, retained);
    EXPECT_EQ(original->identity(), original_identity);
    EXPECT_EQ(inventory->encode_identity(*budget).bytes(), immutable_inventory);
}
TEST_F(CartesianSourceExplicitManifests, RejectsIncompleteRepeatedAndNativeOverlappingExplicitContributors)
{
    const auto explicit_count = 264 + inventory_bytes(partial).size();
    const auto cell_key = explicit_count + 8;
    const auto revision = cell_key + 8 + explicit_a.key.size();
    const auto member_count = revision + 8;
    const auto first_member = member_count + 8;
    const auto first_size = first_member + 8 + explicit_a.members.front().support.size();
    for (const auto cut : {explicit_count, explicit_count + 7, cell_key, cell_key + 7, revision - 1, revision,
             revision + 7, member_count, member_count + 7, first_member, first_member + 7, first_size - 1, first_size,
             first_size + 7, complete.front().bytes.size() - 1})
    {
        SCOPED_TRACE(cut);
        auto packets = complete;
        packets.front().bytes.resize(cut);
        expect_rejected(packets);
    }
    for (const auto count : {5u, 7u})
    {
        auto bad = explicit_a;
        if (count == 5)
            bad.members.pop_back();
        else
            bad.members.push_back(bad.members.front());
        expect_bad_cell(bad); // Complete framing still cannot change the trusted six-face count.
    }
    auto packets = complete;
    packets.front().bytes.push_back('x');
    expect_rejected(packets);
    packets = complete;
    const std::array duplicate{explicit_a, explicit_a};
    packets.front() = explicit_receipt(value, partial, duplicate);
    expect_rejected(packets);
    packets = complete;
    packets.front() = explicit_receipt(value, literal, std::span(&explicit_a, 1));
    expect_rejected(packets); // A cannot be both a native recipe and an explicit contributor.
    auto bad = explicit_a;
    bad.members[1] = bad.members[0];
    expect_bad_cell(bad);
    bad = explicit_a;
    bad.members.front().encoded_size = 0;
    expect_bad_cell(bad);
    bad = explicit_a;
    bad.key = encode_adaptive_key({value.region_id, CartesianCellAncestry{0, {2, 0, 0}}});
    for (auto& member : bad.members)
    {
        Reader support{member.support};
        (void) support.word();
        (void) support.bytes_view();
        (void) support.word();
        const auto axis = support.word();
        (void) support.word();
        const auto plane_offset = support.offset;
        const auto plane = support.word();
        if (axis == 0)
            replace_word(member.support, plane_offset, plane + 2);
        else
            replace_word(member.support, support.offset, 2); // x is the first transverse coordinate.
    }
    expect_bad_cell(bad); // Self-consistent primitive headers still cannot place a cell outside the grid.
    bad = explicit_a;
    bad.key = encode_adaptive_key({value.region_id + 1, CartesianCellAncestry{0, {0, 0, 0}}});
    expect_bad_cell(bad);
    bad = explicit_a;
    bad.key = encode_adaptive_key({value.region_id, CartesianCellAncestry{value.max_level + 1, {0, 0, 0}}});
    expect_bad_cell(bad);
}
TEST_F(CartesianSourceExplicitManifests, RejectsMalformedPrimitiveSupportIdentityAndStaleOuterBindings)
{
    // Primitive support layout: family, domain length/data, region, axis,
    // reduced level, plane, then two transverse cell coordinates.
    Reader first{explicit_a.members.front().support};
    ASSERT_EQ(first.word(), 0x5346434152543631ULL);
    const auto domain = first.bytes_view();
    ASSERT_FALSE(domain.empty());
    const auto region = first.offset;
    EXPECT_EQ(first.word(), value.region_id);
    const auto axis = first.offset;
    EXPECT_LT(first.word(), 3);
    const auto level = first.offset;
    (void) first.word();
    const auto plane = first.offset;
    (void) first.word();
    const auto transverse = first.offset;
    expect_bad_member(0, 0);
    expect_bad_member(region, value.region_id + 1);
    expect_bad_member(axis, 3);
    expect_bad_member(level, value.max_level + 1);
    expect_bad_member(plane, 17);
    expect_bad_member(transverse, 17);
    expect_bad_member(transverse + 8, 17);
    auto bad = explicit_a;
    // Drop the framed domain contents and retain every other literal word.
    Writer empty_domain;
    empty_domain.word(0x5346434152543631ULL);
    empty_domain.bytes({});
    empty_domain.data.insert(
        empty_domain.data.end(), bad.members.front().support.begin() + region, bad.members.front().support.end());
    bad.members.front().support = std::move(empty_domain.data);
    expect_bad_cell(bad);
    bad = explicit_a;
    bad.members[1].support[16] ^= 1; // One face claims another connected domain.
    expect_bad_cell(bad);
    for (const auto offset : {std::size_t(16), std::size_t(24), std::size_t(32), std::size_t(40), std::size_t(48),
             std::size_t(56), std::size_t(72), std::size_t(88)})
    {
        SCOPED_TRACE(offset);
        auto packets = complete;
        packets.front().bytes[offset] ^= 1;
        expect_rejected(packets);
    }
    for (const auto limit : {std::size_t(0), std::size_t(1), std::size_t(64)})
    {
        Budget rejected{limit};
        EXPECT_THROW(
            CartesianSourceManifestSet::validate_untrusted_wire(value, context, generation, complete, 2, rejected),
            std::exception);
        EXPECT_EQ(rejected.used, 0);
        EXPECT_EQ(original->identity(), original_identity);
        EXPECT_EQ(inventory->encode_identity(*budget).bytes(), immutable_inventory);
    }
}
} // namespace

TEST_F(CartesianSourceInventoryManifests, ReceiptMaterializationExactCreditAndOneByteShortRestoreArena)
{
    Budget measured{budget->limit};
    CartesianSourceManifestSet::validate_receipt_storage(value, context, generation, valid, 2, measured);
    EXPECT_EQ(measured.used, 0u);
    ASSERT_GT(measured.peak, 1u);
    Budget exact{measured.peak};
    EXPECT_NO_THROW(CartesianSourceManifestSet::validate_receipt_storage(value, context, generation, valid, 2, exact));
    EXPECT_EQ(exact.used, 0u);
    Budget short_budget{measured.peak - 1};
    EXPECT_THROW(
        CartesianSourceManifestSet::validate_receipt_storage(value, context, generation, valid, 2, short_budget),
        std::invalid_argument);
    EXPECT_EQ(short_budget.used, 0u);

    auto foreign = valid;
    foreign.front().bytes[16] ^= 1;
    EXPECT_THROW(CartesianSourceManifestSet::validate_receipt_storage(value, context, generation, foreign, 2, measured),
        std::invalid_argument);
    EXPECT_EQ(measured.used, 0u);
}

TEST_F(CartesianSourceInventoryManifests, ReorderedRepeatedSourcePreservesOrdinalsAndIrregularProofBytes)
{
    using Index = std::array<std::uint64_t, 3>;
    const auto leaf_at = [&](Index index)
    { return cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, index}}, {'d'}); };
    const auto point_range = [](Index index)
    { return CartesianIncidenceRange{0, index, {index[0] + 1, index[1] + 1, index[2] + 1}}; };

    const std::array a{Index{1, 0, 0}, Index{0, 1, 0}, Index{0, 0, 0}};
    const std::array b{Index{1, 0, 1}, Index{1, 1, 0}};
    const std::array retained{leaf_at(a[0]), leaf_at(a[1]), leaf_at(a[2]), leaf_at(b[0]), leaf_at(b[1])};
    const std::array removed{leaf_at({0, 0, 1})};
    const auto repeated = CartesianIncidenceSource::make(value, retained, {'d'}, 0, 7, context, *budget, budget);
    const auto prefix = CartesianIncidenceSource::make(value, removed, {'d'}, 0, 8, context, *budget, budget);
    const LiteralInventory initial{context, 0, generation, 6,
        {{prefix->identity(), 1, {point_range({0, 0, 1})}},
            {repeated->identity(), 3, {point_range(a[0]), point_range(a[1]), point_range(a[2])}},
            {repeated->identity(), 2, {point_range(b[0]), point_range(b[1])}}}};
    auto next = context;
    ++next.round;
    const auto version = generation + 1;
    const LiteralInventory edited{next, 0, version, 5,
        {{repeated->identity(), 2, {point_range(b[0]), point_range(b[1])}},
            {repeated->identity(), 3, {point_range(a[1]), point_range(a[0]), point_range(a[2])}}}};
    const auto packets = [&](const LiteralInventory& inventory)
    {
        return std::array{explicit_receipt(value, inventory, {}),
            explicit_receipt(value, {inventory.context, 1, inventory.version, 0, {}}, {})};
    };
    const auto before = packets(initial);
    auto after = packets(edited);
    std::swap(after[0], after[1]); // Receive ordinal is independent of owner identity.
    const auto retained_bytes = budget->used;
    CartesianSourceManifestSet::inspect_receipt_transition(value, {context, generation, before}, {next, version, after},
        2, *budget,
        [&](const auto& previous, const auto& current)
        {
            ASSERT_EQ(previous.source_count(), 4u);
            ASSERT_EQ(current.source_count(), 3u);
            EXPECT_EQ(current.owner(0), 1);
            EXPECT_TRUE(current.identity(0).empty());
            const auto expect = [](const auto& receipt, std::size_t ordinal, const auto& expected)
            {
                ASSERT_EQ(receipt.range_count(ordinal), expected.size());
                for (std::size_t i = 0; i < expected.size(); ++i)
                {
                    const auto actual = receipt.range(ordinal, i);
                    EXPECT_EQ(actual.level, expected[i].level);
                    EXPECT_EQ(actual.lower, expected[i].lower);
                    EXPECT_EQ(actual.upper, expected[i].upper);
                }
            };
            expect(previous, 1, initial.batches[1].ranges);
            expect(previous, 2, initial.batches[2].ranges);
            expect(current, 1, edited.batches[0].ranges);
            expect(current, 2, edited.batches[1].ranges);
            EXPECT_EQ(current.owner(1), 0);
            EXPECT_EQ(current.owner(2), 0);
            EXPECT_EQ(current.version(1), 7u);
            EXPECT_EQ(current.version(2), 7u);
            EXPECT_EQ(current.identity(1).data(), previous.identity(2).data());
            EXPECT_TRUE(current.identity(2).data() == previous.identity(1).data() ||
                        current.identity(2).data() == previous.identity(2).data());
            ASSERT_GT(current.irregular_range_admission(2), 0u);
            // Only the changed noncanonical range order needs a new byte body.
            EXPECT_EQ(current.work().copied_body_bytes, current.irregular_range_admission(2));
        });
    EXPECT_EQ(budget->used, retained_bytes);
}
