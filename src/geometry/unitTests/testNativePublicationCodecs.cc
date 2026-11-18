#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/NativeAdaptiveLeafCodec.hh"
#include "geometry/mesh/AMR/NativeAdaptiveTileCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/PLIC/details/NativePlanarCodecAccess.hh"
#include "native_constructor_allocation_operators.inc"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <optional>
namespace
{
namespace M = SimpleFluid::Meshes::AMR;
namespace A = M::adaptive_detail;
namespace G = SimpleFluid::geometry_detail;
namespace P = SimpleFluid::PLIC;
using Mode = SimpleFluid::ArithmeticMode;
struct Counter
{
    std::size_t limit = 0, used = 0, peak = 0, cumulative = 0;
    G::NativeCounterAllocationEnvelope native_allocation;
    void add(std::size_t bytes)
    {
        if (used > limit || bytes > limit - used)
            throw std::invalid_argument("test codec cap");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
};
struct Domain
{
    G::NativeOwnedCounter<Counter> counter;
    G::NativeConstructionAdmission admission;
    explicit Domain(std::size_t limit = 4 * 1024 * 1024)
        : counter(G::NativeConstructionCounterAccess::create<Counter>(limit)), admission(counter)
    {
    }
};
struct Key
{
    G::NativeConstructorStorageOwner storage;
    M::AdaptiveCellKey key;
};
struct Bytes
{
    G::NativeConstructorStorageOwner storage;
    std::vector<char> bytes;
};
void word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.push_back(static_cast<char>(value >> (8 * i)));
}
TEST(NativePublicationCodecs, CanonicalProductBytesUseIndependentLiteralFieldOrder)
{
    Domain domain;
    M::AdaptiveCellKey key{19, M::ProductCellAncestry{{23, 29, {31, 37}}, {41, {43, 47, 53}}}};
    std::vector<char> expected;
    for (auto value :
        std::array<std::uint64_t, 15>{0x5346414450543631ULL, 1, 0, 19, 2, 23, 29, 2, 31, 37, 41, 3, 43, 47, 53})
        word(expected, value);
    auto storage = G::native_constructor_storage(domain.admission);
    A::Writer writer;
    storage.owner->reserve(writer.data, expected.size() + 8);
    A::NativeAdaptiveKeyCodec::write(writer, key);
    std::vector<char> framed;
    word(framed, expected.size());
    framed.insert(framed.end(), expected.begin(), expected.end());
    EXPECT_EQ(writer.data, framed);
    EXPECT_EQ(A::NativeAdaptiveKeyCodec::decode(expected, storage), key);
    EXPECT_EQ(A::NativeAdaptiveKeyCodec::copy(key, storage), key);
    EXPECT_EQ(M::decode_adaptive_key(expected), key);
    A::Writer unadmitted;
    EXPECT_THROW(A::NativeAdaptiveKeyCodec::write(unadmitted, key), std::invalid_argument);
}
TEST(NativePublicationCodecs, DeepKeyCppAllocationsArePrechargedAndRetireWithActualStorage)
{
    M::ConvexCellAncestry ancestry{71, std::vector<std::uint64_t>(513, 3)};
    const M::AdaptiveCellKey key{61, std::move(ancestry)};
    const auto packet = M::encode_adaptive_key(key);
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    std::optional<Key> decoded;
    decoded.emplace();
    decoded->storage = G::native_constructor_storage(domain.admission);
    decoded->key = A::NativeAdaptiveKeyCodec::decode(packet, decoded->storage);
    const auto live = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(live, trace.all_live);
    EXPECT_EQ(decoded->key, key);
    decoded.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    for (std::size_t ordinal = 0; ordinal < trace.phase[0].allocations; ++ordinal)
    {
        bool failed = false;
        {
            native_constructor_test::Failure failure(ordinal);
            try
            {
                Key value{G::native_constructor_storage(domain.admission), {}};
                value.key = A::NativeAdaptiveKeyCodec::decode(packet, value.storage);
            }
            catch (const std::bad_alloc&)
            {
                failed = true;
            }
        }
        EXPECT_TRUE(failed) << ordinal;
        EXPECT_EQ(domain.counter.owner()->used, prefix);
        Key retry{G::native_constructor_storage(domain.admission), {}};
        retry.key = A::NativeAdaptiveKeyCodec::decode(packet, retry.storage);
        EXPECT_EQ(retry.key, key);
    }
}
TEST(NativePublicationCodecs, FacePacketBytesAndRawCppAllocationsMatchPublicCodec)
{
    const std::array<P::GeometryPoint, 5> points{{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 3, 0}, {0, 3, 0}}};
    const auto face = P::PlanarFace<Mode::Ordinary>::make(points);
    for (const bool incidence : {false, true})
    {
        const auto expected = incidence ? face->encode_incidence() : face->encode();
        Domain domain;
        const auto prefix = domain.counter.owner()->used;
        native_constructor_test::Capture capture(domain.counter.owner()->used);
        std::optional<Bytes> encoded;
        encoded.emplace();
        encoded->storage = G::native_constructor_storage(domain.admission);
        encoded->bytes = P::details::NativePlanarCodecAccess::encode(*face, incidence, encoded->storage);
        const auto live = domain.counter.owner()->used - prefix;
        const auto trace = capture.finish();
        EXPECT_EQ(native_constructor_test::uncovered, 0u);
        EXPECT_FALSE(pipeline_trace::overflow.load());
        EXPECT_EQ(live, trace.all_live);
        EXPECT_EQ(encoded->bytes, expected);
        encoded.reset();
        EXPECT_EQ(domain.counter.owner()->used, prefix);
    }
}
TEST(NativePublicationCodecs, MalformedAncestryPreservesAcceptedAllocationAndRetry)
{
    Domain domain;
    const auto accepted = domain.admission.shared<unsigned>(89);
    const auto prefix = domain.counter.owner()->used;
    const M::AdaptiveCellKey key{7, M::ProductCellAncestry{{1, 2, {3}}, {4, {5}}}};
    auto packet = M::encode_adaptive_key(key);
    packet.pop_back();
    {
        Key result{G::native_constructor_storage(domain.admission), {}};
        EXPECT_THROW((void) A::NativeAdaptiveKeyCodec::decode(packet, result.storage), std::invalid_argument);
    }
    EXPECT_EQ(*accepted, 89u);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    packet = M::encode_adaptive_key(key);
    {
        Key result{G::native_constructor_storage(domain.admission), {}};
        result.key = A::NativeAdaptiveKeyCodec::decode(packet, result.storage);
        EXPECT_EQ(result.key, key);
    }
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, FullLeafDirectEmissionPreservesPublicBytesAndOwnsEveryCppBuffer)
{
    M::AdaptiveLeaf leaf;
    leaf.key = {13, M::ConvexCellAncestry{17, {1, 3, 5}}};
    leaf.domain = {'d', 'o', 'm', 'a', 'i', 'n'};
    leaf.material = 19;
    leaf.revision = 23;
    leaf.geometry = P::ConvexCellGeometry<Mode::Ordinary>::box({-2, 3, 7}, {5, 11, 13});
    for (std::size_t face = 0; face < leaf.geometry->num_faces(); ++face)
    {
        M::AdaptiveSurface surface;
        surface.key = {'s', char(face)};
        surface.geometry = leaf.geometry->face_geometry(face);
        surface.native_whole_geometry = surface.geometry;
        surface.require_native_whole_tile = true;
        surface.exterior = true;
        surface.boundary = std::string(89 + face, char('a' + face));
        surface.native_face = face;
        leaf.surfaces.push_back(std::move(surface));
    }
    const auto expected = M::encode_adaptive_leaf(leaf);
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    std::optional<Bytes> encoded;
    encoded.emplace();
    encoded->storage = G::native_constructor_storage(domain.admission);
    A::Writer output;
    encoded->storage.owner->reserve(output.data, 8 + A::NativeAdaptiveLeafCodec::size(leaf, domain.admission));
    A::NativeAdaptiveLeafCodec::write(output, leaf, domain.admission);
    encoded->bytes = std::move(output.data);
    const auto live = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(live, trace.all_live);
    A::Reader input{encoded->bytes};
    const auto body = input.bytes_view();
    input.done();
    EXPECT_EQ(std::vector<char>(body.begin(), body.end()), expected);
    encoded.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, RetainedPlanarDecodeTranslationAndWeakControlMatchRawCppLifetimes)
{
    const std::array<P::GeometryPoint, 4> points{{{0, 0, 0}, {2, 0, 0}, {2, 3, 0}, {0, 3, 0}}};
    auto original = P::PlanarFace<Mode::Ordinary>::make(points);
    for (unsigned i = 0; i < 73; ++i)
        original = original->translated(P::displacement<Mode::Ordinary>({0, 0, 0}, {0, 0, 1}));
    const auto packet = original->encode_incidence();
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    auto decoded = P::details::NativePlanarOwnerAccess::decode(packet, 64 * 1024 * 1024, domain.admission);
    auto reversed = P::details::NativePlanarOwnerAccess::reversed(*decoded, domain.admission);
    auto translated = P::details::NativePlanarOwnerAccess::translated(
        *reversed, P::displacement<Mode::Ordinary>({0, 0, 0}, {1, 2, 3}), domain.admission);
    const auto live = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(live, trace.all_live);
    EXPECT_TRUE(P::details::NativePlanarOwnerAccess::admitted(decoded));
    EXPECT_EQ(decoded->encode_incidence(), packet);
    EXPECT_TRUE(decoded->equivalent(*original));
    std::weak_ptr<const P::PlanarFace<Mode::Ordinary>> weak = decoded;
    translated.reset();
    reversed.reset();
    decoded.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_GT(domain.counter.owner()->used, prefix);
    weak.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, RetainedCellDecodeCopiesActualOwnersAndPreservesCertificateAndMetrics)
{
    auto original = P::ConvexCellGeometry<Mode::Ordinary>::box({-3, 5, 7}, {11, 13, 17});
    const auto packet = original->encode_certificate();
    const auto volume = original->volume().value;
    const auto center = original->centroid();
    std::weak_ptr<const P::ConvexCellGeometry<Mode::Ordinary>> old = original;
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    auto decoded = P::details::NativeCellCodecAccess::decode(packet, 64 * 1024 * 1024, domain.admission);
    const auto live = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(live, trace.all_live);
    EXPECT_EQ(domain.admission.statistics().reserved_live, 0u);
    EXPECT_EQ(domain.admission.statistics().profiled_live, 0u);
    EXPECT_GT(domain.admission.statistics().profiled_cumulative_upper, 0u);
    EXPECT_EQ(decoded->encode_certificate(), packet);
    EXPECT_EQ(decoded->volume().value, volume);
    EXPECT_EQ(decoded->centroid(), center);
    original.reset();
    EXPECT_TRUE(old.expired());
    EXPECT_EQ(decoded->encode_certificate(), packet);
    std::weak_ptr<const P::ConvexCellGeometry<Mode::Ordinary>> weak = decoded;
    decoded.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_GT(domain.counter.owner()->used, prefix);
    weak.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, SeparateVectorControlKeepsOnlyRawObjectMemoryThroughWeakLifetime)
{
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    auto tracked = G::NativeObjectOwnerAccess<std::vector<char>>::make(domain.admission);
    const auto storage = G::NativeObjectOwnerAccess<std::vector<char>>::storage(tracked.owner);
    storage.owner->reserve(*tracked.owner, 137);
    tracked.owner->resize(137, 's');
    std::weak_ptr<const std::vector<char>> weak = tracked.owner;
    tracked.owner.reset();
    const auto fixed = tracked.envelope->fixed_allocation_bytes();
    const auto remaining = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(storage.owner->exclusive_bytes(), 0u);
    EXPECT_EQ(remaining, trace.all_live);
    EXPECT_GE(remaining, fixed);
    tracked.envelope.reset();
    weak.reset();
    // An independent storage observer remains, but no freed payload is charged.
    EXPECT_EQ(storage.owner->exclusive_bytes(), 0u);
}
TEST(NativePublicationCodecs, NativeTileFullCodecPreservesAllWireFieldsAndCppOwnership)
{
    M::AdaptiveTile tile;
    tile.face_gid = 103;
    tile.domain = std::vector<char>{'d'};
    tile.surface = std::vector<char>{'s'};
    tile.identity = {'i', 'd'};
    const std::array<P::GeometryPoint, 4> points{{{0, 0, 0}, {2, 0, 0}, {2, 3, 0}, {0, 3, 0}}};
    tile.geometry = P::PlanarFace<Mode::Ordinary>::make(points);
    tile.first = {107, M::ProductCellAncestry{{109, 113, {1, 2}}, {127, {0, 1}}}};
    tile.second = M::AdaptiveCellKey{131, M::ConvexCellAncestry{137, {3, 4, 5}}};
    tile.first_owner = 0;
    tile.second_owner = 1;
    tile.first_native_face = 139;
    tile.second_native_face = 149;
    tile.boundary = std::string(73, 'b');
    tile.second_image_identity = P::displacement<Mode::Ordinary>({0, 0, 0}, {1, 2, 3});
    tile.second_image = P::displacement_display<Mode::Ordinary>(tile.second_image_identity);
    tile.node_gids = std::vector<std::uint64_t>{151, 157, 163, 167};
    tile.second_node_gids = std::vector<std::uint64_t>{173, 179, 181, 191};
    const auto packet = M::encode_adaptive_tile(tile);
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    struct OwnedTile
    {
        G::NativeConstructorStorageOwner storage;
        M::AdaptiveTile tile;
    };
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    std::optional<OwnedTile> decoded;
    decoded.emplace();
    decoded->storage = G::native_constructor_storage(domain.admission);
    decoded->tile = A::NativeAdaptiveTileCodec::decode(packet, 64 * 1024 * 1024, decoded->storage);
    const auto live = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(live, trace.all_live);
    EXPECT_EQ(M::encode_adaptive_tile(decoded->tile), packet);
    EXPECT_EQ(decoded->tile.first, tile.first);
    EXPECT_EQ(decoded->tile.second, tile.second);
    EXPECT_EQ(decoded->tile.node_gids, tile.node_gids);
    EXPECT_EQ(decoded->tile.second_node_gids, tile.second_node_gids);
    EXPECT_EQ(decoded->tile.boundary, tile.boundary);
    decoded.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, SeparateControlEveryHeapPrefixAndConstructorFailureRetireExactly)
{
    const auto operation = [](const G::NativeConstructionAdmission& admission)
    {
        auto owner = G::NativeObjectOwnerAccess<std::vector<char>>::make(admission);
        const auto& storage = G::NativeObjectOwnerAccess<std::vector<char>>::storage(owner.owner);
        storage.owner->reserve(*owner.owner, 97);
        owner.owner->resize(97, 'v');
        return owner;
    };
    Domain measured;
    native_constructor_test::Capture capture(measured.counter.owner()->used);
    {
        const auto value = operation(measured.admission);
    }
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    for (std::size_t ordinal = 0; ordinal < trace.phase[0].allocations; ++ordinal)
    {
        Domain domain;
        const auto accepted = domain.admission.shared<unsigned>(211);
        const auto prefix = domain.counter.owner()->used;
        bool failed = false;
        {
            native_constructor_test::Failure failure(ordinal);
            try
            {
                const auto value = operation(domain.admission);
            }
            catch (const std::bad_alloc&)
            {
                failed = true;
            }
        }
        EXPECT_TRUE(failed) << ordinal;
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
        EXPECT_EQ(*accepted, 211u);
        {
            const auto retry = operation(domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix);
    }
    struct Throwing
    {
        Throwing() { throw std::bad_alloc(); }
    };
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    EXPECT_THROW((void) G::NativeObjectOwnerAccess<Throwing>::make_without_buffers(domain.admission), std::bad_alloc);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, SeparateOveralignedObjectBlockSurvivesUntilLastWeakControl)
{
    struct alignas(256) Aligned
    {
        std::array<std::byte, 256> values{};
    };
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    auto tracked = G::NativeObjectOwnerAccess<Aligned>::make_without_buffers(domain.admission);
    const auto address = reinterpret_cast<std::uintptr_t>(tracked.owner.get());
    std::weak_ptr<const Aligned> weak = tracked.owner;
    tracked.owner.reset();
    const auto live = domain.counter.owner()->used - prefix;
    const auto fixed = tracked.envelope->fixed_allocation_bytes();
    const auto trace = capture.finish();
    EXPECT_EQ(address % alignof(Aligned), 0u);
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_EQ(live, trace.all_live);
    EXPECT_EQ(live, fixed);
    EXPECT_TRUE(weak.expired());
    tracked.envelope.reset();
    weak.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
} // namespace
