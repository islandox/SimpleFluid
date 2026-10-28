/** @file testGeneralVoFOwnerPlaneRequests.cc
 * @brief Actual Ordinary fits, authenticated cache reuse and bounded owner replies.
 */
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <tuple>
#include <utility>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
namespace local = SimpleFluid::VoF::general_vof_detail;
namespace AMR = SimpleFluid::Meshes::AMR;
namespace wire = local::wire;
constexpr auto mode = ArithmeticMode::Ordinary;
constexpr std::size_t allowance = 1024 * 1024;
constexpr std::uint64_t revision = 17;
using Geometry = PLIC::ConvexCellGeometry<mode>;
using Sample = PLIC::PolyhedralPhaseSample<mode>;
using Diagnostics = GeneralVoFOwnerPlaneRequestDiagnostics;
using Work = SimpleFluid::geometry::swept_volume::Work;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

AMR::AdaptiveCellKey key(unsigned region = 3)
{
    AMR::AdaptiveCellKey result;
    result.region = region;
    return result;
}
AMR::AdaptiveCellKey deep_key(bool product)
{
    auto result = key(product ? 9 : 8);
    std::vector<std::uint64_t> path(129);
    for (std::size_t i = 0; i < path.size(); ++i)
        path[i] = 3 * i + 1;
    if (product)
        result.ancestry = AMR::ProductCellAncestry{{13, 17, path}, {19, path}};
    else
        result.ancestry = AMR::ConvexCellAncestry{11, path};
    return result;
}
PLIC::GeometryPacketContext context()
{
    return {{11, 23, 37, 53}};
}
OwnerPacket request(int peer, AMR::AdaptiveCellKey requested = key(), std::uint64_t requested_revision = revision,
    PLIC::GeometryPacketContext requested_context = context())
{
    wire::Writer writer;
    writer.bytes(AMR::encode_adaptive_key(requested));
    writer.word(requested_revision);
    for (const auto word : requested_context.publication)
        writer.word(word);
    return {peer, std::move(writer.data)};
}
Work work(std::size_t limit = allowance)
{
    Work result;
    result.arithmetic_mode = mode;
    result.limits.byte_budget = limit;
    result.charge(29); // Unrelated caller storage must survive every service scope.
    return result;
}

/** The analytic gas region is x <= fraction. Its target and ten neighbors
 * provide independent expected volumes and three-dimensional gradient support.
 */
struct PlanarFixture
{
    double fraction = .25;
    Sample target;
    std::vector<Sample> stencil;
    explicit PlanarFixture(double gas = .25) : fraction(gas), target{Geometry::box({0, 0, 0}, {1, 1, 1}), gas, 1 - gas}
    {
        for (const auto index : std::vector<std::array<int, 3>>{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0},
                 {0, 0, -1}, {0, 0, 1}, {-1, -1, -1}, {1, 1, 1}, {-1, 1, 1}, {1, -1, -1}})
        {
            const PLIC::GeometryPoint lower{double(index[0]), double(index[1]), double(index[2])};
            const PLIC::GeometryPoint upper{double(index[0] + 1), double(index[1] + 1), double(index[2] + 1)};
            const double amount = index[0] < 0 ? 1 : index[0] > 0 ? 0 : gas;
            stencil.push_back({Geometry::box(lower, upper), amount, 1 - amount});
        }
    }
    void change_phase(double gas)
    {
        fraction = gas;
        target.gas_volume = gas;
        target.carrier_volume = 1 - gas;
        for (auto& sample : stencil)
            if (sample.gas_volume > 0 && sample.gas_volume < 1)
            {
                sample.gas_volume = gas;
                sample.carrier_volume = 1 - gas;
            }
    }
    void check(const PLIC::PolyhedralInterface& fitted) const
    {
        EXPECT_FALSE(fitted.pure_gas);
        EXPECT_FALSE(fitted.pure_carrier);
        EXPECT_TRUE(fitted.finite_geometry);
        EXPECT_NEAR(fitted.gas.normal[0], 1, 1e-8);
        EXPECT_NEAR(fitted.gas.normal[1], 0, 1e-8);
        EXPECT_NEAR(fitted.gas.normal[2], 0, 1e-8);
        // The scalar offset is relative to the represented anchor.
        EXPECT_NEAR(fitted.gas.offset + fitted.gas.anchor[0], fraction, 1e-8);
        const auto measured = target.geometry->phases(fitted.gas);
        EXPECT_LE(std::abs(measured.gas.value - fraction),
            fraction * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
        EXPECT_LE(std::abs(measured.carrier.value - (1 - fraction)),
            (1 - fraction) * PLIC::PolyhedralReconstructionOptions{}.volume_tolerance);
        EXPECT_EQ(target.gas_volume, fraction);
        EXPECT_EQ(target.carrier_volume, 1 - fraction);
    }
    PLIC::PolyhedralInterface fit(const AMR::AdaptiveCellKey&, Diagnostics& diagnostics) const
    {
        ++diagnostics.owner_reconstruction_calls;
        ++diagnostics.mixed_cell_fits;
        auto fitted = PLIC::PolyhedralReconstruction<mode>::reconstruct(target, stencil);
        EXPECT_TRUE(fitted.resolution.admitted());
        EXPECT_TRUE(fitted.fit_converged);
        EXPECT_EQ(fitted.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
        check(fitted);
        return fitted;
    }
};

/** An independent test payload, padded to the production interface envelope.
 * This qualifies helper reuse and storage, not the unchanged production codec.
 * All fit fields inspected below come from an actual Ordinary reconstruction.
 */
void serialize(wire::Writer& writer, const PLIC::PolyhedralInterface& fitted)
{
    const auto begin = writer.data.size();
    const auto real = [&](double value) { writer.word(std::bit_cast<std::uint64_t>(value)); };
    for (const auto value : fitted.gas.normal)
        real(value);
    for (const auto value : fitted.gas.anchor)
        real(value);
    real(fitted.gas.offset);
    writer.word(fitted.pure_gas);
    writer.word(fitted.pure_carrier);
    real(fitted.stencil_volume_residual);
    writer.word(fitted.stencil_volume_residual_measured);
    real(fitted.stencil_condition);
    real(fitted.orientation_step);
    writer.word(fitted.fit_iterations);
    writer.word(std::uint64_t(fitted.orientation_method));
    writer.word(fitted.fit_converged);
    real(double(fitted.target_volume_numerical_error_estimate));
    writer.word(fitted.finite_geometry);
    writer.word(std::uint64_t(fitted.finite_target_volume_difference.arithmetic_mode));
    real(double(fitted.finite_target_volume_difference.value));
    real(double(fitted.finite_target_volume_difference.absolute_error));
    writer.data.resize(begin + 216, char(0x5a));
}
PLIC::PolyhedralInterface decode(const OwnerPacket& packet, const AMR::AdaptiveCellKey& expected_key = key())
{
    wire::Reader reader{packet.bytes};
    EXPECT_EQ(AMR::decode_adaptive_key(reader.bytes()), expected_key);
    EXPECT_EQ(reader.word(), revision);
    for (const auto expected : context().publication)
        EXPECT_EQ(reader.word(), expected);
    const auto begin = reader.offset;
    const auto real = [&] { return std::bit_cast<double>(reader.word()); };
    PLIC::PolyhedralInterface fitted;
    for (auto& value : fitted.gas.normal)
        value = real();
    for (auto& value : fitted.gas.anchor)
        value = real();
    fitted.gas.offset = real();
    fitted.pure_gas = reader.word();
    fitted.pure_carrier = reader.word();
    fitted.stencil_volume_residual = real();
    fitted.stencil_volume_residual_measured = reader.word();
    fitted.stencil_condition = real();
    fitted.orientation_step = real();
    fitted.fit_iterations = reader.word();
    fitted.orientation_method = PLIC::PolyhedralOrientationMethod(reader.word());
    fitted.fit_converged = reader.word();
    fitted.target_volume_numerical_error_estimate = real();
    fitted.finite_geometry = reader.word();
    fitted.finite_target_volume_difference.arithmetic_mode = ArithmeticMode(reader.word());
    fitted.finite_target_volume_difference.value = real();
    fitted.finite_target_volume_difference.absolute_error = real();
    EXPECT_EQ(fitted.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
    EXPECT_TRUE(fitted.fit_converged);
    EXPECT_EQ(fitted.finite_target_volume_difference.arithmetic_mode, mode);
    while (reader.offset < begin + 216)
        EXPECT_EQ(reader.data[reader.offset++], char(0x5a));
    reader.done();
    return fitted;
}
void validate(const AMR::AdaptiveCellKey& requested, std::uint64_t requested_revision, int)
{
    if ((requested != key() && requested != key(4)) || requested_revision != revision)
        throw std::invalid_argument("Test owner has no current requested cell.");
}
template<class Serialize = decltype(&serialize)>
std::vector<OwnerPacket> replies(const PlanarFixture& fixture, const local::RequestedPlaneMap& result,
    std::span<const OwnerPacket> incoming, local::PlaneImportStorage& storage, Diagnostics& diagnostics,
    Serialize&& encode = &serialize, int ranks = 3)
{
    return local::owner_plane_replies(
        result, incoming, context(), 0, ranks, storage, diagnostics, validate,
        [&](const auto& requested) { return fixture.fit(requested, diagnostics); }, std::forward<Serialize>(encode));
}
void check_reply_storage(const Work& retained, const std::vector<OwnerPacket>& packets)
{
    std::size_t actual = packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
        actual += packet.bytes.capacity();
    EXPECT_GE(retained.bytes, 29u + actual);
    EXPECT_LE(retained.bytes, retained.limits.byte_budget);
}

TEST(GeneralVoFOwnerPlaneRequests, LocalOwnedResultServesBothPeersWithoutAnotherFitOrReturnedKey)
{
    const PlanarFixture fixture;
    auto retained = work();
    Diagnostics diagnostics;
    local::RequestedPlaneMap result;
    diagnostics.local_owned_requests = 1;
    result.emplace(key(), fixture.fit(key(), diagnostics));
    const std::vector incoming{request(1), request(2)};
    {
        local::PlaneImportStorage storage(retained, diagnostics);
        const auto outgoing = replies(fixture, result, incoming, storage, diagnostics);
        ASSERT_EQ(outgoing.size(), 2u);
        EXPECT_EQ(outgoing[0].peer, 1);
        EXPECT_EQ(outgoing[1].peer, 2);
        EXPECT_EQ(outgoing[0].bytes, outgoing[1].bytes);
        fixture.check(decode(outgoing[0]));
        fixture.check(decode(outgoing[1]));
        check_reply_storage(retained, outgoing);
        EXPECT_EQ(diagnostics.local_result_hits, 2u);
        EXPECT_EQ(diagnostics.peer_cache_admissions, 0u);
        EXPECT_EQ(diagnostics.peer_cache_hits, 0u);
        EXPECT_EQ(diagnostics.peer_cache_peak_bytes, 0u);
    }
    EXPECT_EQ(retained.bytes, 29u);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result.contains(key()));
    fixture.check(result.at(key()));
    EXPECT_EQ(diagnostics.incoming_peer_requests, 2u);
    EXPECT_EQ(diagnostics.owner_reconstruction_calls, 1u);
    EXPECT_EQ(diagnostics.mixed_cell_fits, 1u);
    EXPECT_EQ(diagnostics.local_owned_requests + diagnostics.peer_cache_admissions, 1u); // Known union: {key()}.
}

TEST(GeneralVoFOwnerPlaneRequests, PeerOnlyDuplicateFitsOnceAndLeavesRequestedResultEmpty)
{
    const PlanarFixture fixture;
    auto retained = work();
    Diagnostics diagnostics;
    const local::RequestedPlaneMap result;
    const std::vector incoming{request(1), request(2)};
    {
        local::PlaneImportStorage storage(retained, diagnostics);
        const auto outgoing = replies(fixture, result, incoming, storage, diagnostics);
        ASSERT_EQ(outgoing.size(), 2u);
        EXPECT_EQ(outgoing[0].bytes, outgoing[1].bytes);
        fixture.check(decode(outgoing[0]));
        fixture.check(decode(outgoing[1]));
        check_reply_storage(retained, outgoing);
        EXPECT_EQ(diagnostics.peer_cache_peak_bytes, local::owner_plane_cache_entry_bytes(key()));
        EXPECT_EQ(diagnostics.peer_cache_admissions, 1u);
        EXPECT_EQ(diagnostics.peer_cache_hits, 1u);
        EXPECT_EQ(diagnostics.local_result_hits, 0u);
    }
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(retained.bytes, 29u);
    EXPECT_EQ(diagnostics.owner_reconstruction_calls, 1u);
    EXPECT_EQ(diagnostics.mixed_cell_fits, 1u);
    EXPECT_EQ(diagnostics.local_owned_requests + diagnostics.peer_cache_admissions, 1u); // Known union: {key()}.
}

TEST(GeneralVoFOwnerPlaneRequests, ExistingCacheNeverBypassesRevisionContextOrPacketCompleteness)
{
    const PlanarFixture fixture;
    for (unsigned defect = 0; defect < 3; ++defect)
    {
        SCOPED_TRACE(defect);
        auto retained = work();
        Diagnostics diagnostics;
        auto invalid = request(2, key(), defect == 0 ? revision + 1 : revision);
        if (defect == 1)
        {
            auto stale = context();
            ++stale.publication[3];
            invalid = request(2, key(), revision, stale);
        }
        if (defect == 2)
            invalid.bytes.push_back('x');
        const std::vector incoming{request(1), std::move(invalid)};
        {
            local::PlaneImportStorage storage(retained, diagnostics);
            EXPECT_THROW((void) replies(fixture, {}, incoming, storage, diagnostics), std::invalid_argument);
            // The ordered first request populated the cache before the rejection.
            EXPECT_EQ(diagnostics.owner_reconstruction_calls, 1u);
            EXPECT_EQ(diagnostics.peer_cache_admissions, 1u);
            EXPECT_EQ(diagnostics.peer_cache_hits, 0u);
            EXPECT_EQ(diagnostics.local_result_hits, 0u);
            EXPECT_LE(retained.bytes, retained.limits.byte_budget);
        }
        EXPECT_EQ(retained.bytes, 29u);
    }
}

TEST(GeneralVoFOwnerPlaneRequests, NextInvocationRefitsChangedPhaseOnIdenticalGeometryKeyAndRevision)
{
    PlanarFixture fixture;
    const auto geometry_identity = fixture.target.geometry.get();
    auto retained = work();
    std::vector<char> previous;
    Diagnostics diagnostics;
    const std::vector incoming{request(1), request(2)};
    for (const double fraction : {.25, .75})
    {
        fixture.change_phase(fraction);
        EXPECT_EQ(fixture.target.geometry.get(), geometry_identity);
        {
            local::PlaneImportStorage storage(retained, diagnostics);
            const auto outgoing = replies(fixture, {}, incoming, storage, diagnostics);
            ASSERT_EQ(outgoing.size(), 2u);
            EXPECT_EQ(outgoing[0].bytes, outgoing[1].bytes);
            fixture.check(decode(outgoing[0]));
            if (!previous.empty())
                EXPECT_NE(previous, outgoing[0].bytes);
            previous = outgoing[0].bytes;
        }
        EXPECT_EQ(retained.bytes, 29u);
    }
    EXPECT_EQ(diagnostics.owner_reconstruction_calls, 2u);
    EXPECT_EQ(diagnostics.mixed_cell_fits, 2u);
    EXPECT_EQ(diagnostics.peer_cache_admissions, 2u);
    EXPECT_EQ(diagnostics.peer_cache_hits, 2u);
}

TEST(GeneralVoFOwnerPlaneRequests, CacheAdmissionBoundaryFallsBackWithoutChangingRepliesOrBudget)
{
    const PlanarFixture fixture;
    const std::vector incoming{request(1), request(2)};
    const auto mandatory = local::owner_plane_request_workspace_bytes(incoming);
    const auto entry = local::owner_plane_cache_entry_bytes(key());
    std::vector<char> previous;
    for (const bool admit : {false, true})
    {
        SCOPED_TRACE(admit);
        auto retained = work(29 + mandatory + entry - (admit ? 0 : 1));
        Diagnostics diagnostics;
        {
            local::PlaneImportStorage storage(retained, diagnostics);
            const auto outgoing = replies(fixture, {}, incoming, storage, diagnostics,
                [&](wire::Writer& writer, const PLIC::PolyhedralInterface& fitted)
                {
                    // A successfully admitted value is still charged when serialized.
                    EXPECT_EQ(retained.bytes, 29 + mandatory + (admit ? entry : 0));
                    serialize(writer, fitted);
                });
            ASSERT_EQ(outgoing.size(), 2u);
            EXPECT_EQ(outgoing[0].bytes, outgoing[1].bytes);
            fixture.check(decode(outgoing[0]));
            if (!previous.empty())
                EXPECT_EQ(previous, outgoing[0].bytes);
            previous = outgoing[0].bytes;
            check_reply_storage(retained, outgoing);
            EXPECT_EQ(diagnostics.owner_reconstruction_calls, admit ? 1u : 2u);
            EXPECT_EQ(diagnostics.mixed_cell_fits, admit ? 1u : 2u);
            EXPECT_EQ(diagnostics.peer_cache_admissions, admit ? 1u : 0u);
            EXPECT_EQ(diagnostics.peer_cache_hits, admit ? 1u : 0u);
            EXPECT_EQ(diagnostics.cache_admission_denials, admit ? 0u : 2u);
            EXPECT_EQ(diagnostics.workspace_peak_bytes, 29 + mandatory + (admit ? entry : 0));
            EXPECT_LE(diagnostics.workspace_peak_bytes, retained.limits.byte_budget);
        }
        EXPECT_EQ(retained.bytes, 29u);
    }
}

TEST(GeneralVoFOwnerPlaneRequests, FirstCacheEntryCannotConsumeFinalDistinctReplyReservation)
{
    const PlanarFixture fixture;
    const std::vector incoming{request(1), request(2, key(4))};
    const auto mandatory = local::owner_plane_request_workspace_bytes(incoming);
    const auto entry = local::owner_plane_cache_entry_bytes(key());
    auto retained = work(29 + mandatory + entry);
    Diagnostics diagnostics;
    {
        local::PlaneImportStorage storage(retained, diagnostics);
        const auto outgoing = replies(fixture, {}, incoming, storage, diagnostics);
        ASSERT_EQ(outgoing.size(), 2u);
        fixture.check(decode(outgoing[0]));
        fixture.check(decode(outgoing[1], key(4)));
        check_reply_storage(retained, outgoing);
        EXPECT_EQ(diagnostics.owner_reconstruction_calls, 2u);
        EXPECT_EQ(diagnostics.peer_cache_admissions, 1u);
        EXPECT_EQ(diagnostics.cache_admission_denials, 1u);
        EXPECT_EQ(diagnostics.workspace_peak_bytes, retained.limits.byte_budget);
    }
    EXPECT_EQ(retained.bytes, 29u);
}

TEST(GeneralVoFOwnerPlaneRequests, SerializationExceptionRetainsCacheChargeUntilScopeUnwinds)
{
    const PlanarFixture fixture;
    const std::vector incoming{request(1), request(2)};
    const auto mandatory = local::owner_plane_request_workspace_bytes(incoming);
    const auto entry = local::owner_plane_cache_entry_bytes(key());
    auto retained = work(29 + mandatory + entry);
    Diagnostics diagnostics;
    {
        local::PlaneImportStorage storage(retained, diagnostics);
        EXPECT_THROW((void) replies(fixture, {}, incoming, storage, diagnostics,
                         [&](wire::Writer&, const PLIC::PolyhedralInterface& fitted)
                         {
                             fixture.check(fitted);
                             EXPECT_EQ(retained.bytes, 29 + mandatory + entry);
                             throw std::runtime_error("Test reply serialization rejected.");
                         }),
            std::runtime_error);
        EXPECT_EQ(diagnostics.peer_cache_admissions, 1u);
        EXPECT_EQ(diagnostics.owner_reconstruction_calls, 1u);
        EXPECT_LE(retained.bytes, retained.limits.byte_budget);
    }
    EXPECT_EQ(retained.bytes, 29u);
    // A failed service leaves the same caller usable for another invocation.
    {
        local::PlaneImportStorage storage(retained, diagnostics);
        const auto outgoing = replies(fixture, {}, incoming, storage, diagnostics);
        EXPECT_EQ(outgoing.size(), 2u);
    }
    EXPECT_EQ(retained.bytes, 29u);
    EXPECT_EQ(diagnostics.owner_reconstruction_calls, 2u);
}

TEST(GeneralVoFOwnerPlaneRequests, KeyWireStorageMatchesActualEncodingForEveryAncestryVariant)
{
    auto cartesian = key(5), fixed = key(6);
    cartesian.ancestry = AMR::CartesianCellAncestry{4, {7, 11, 13}};
    fixed.ancestry = AMR::StaticCellAncestry{23, 29};
    for (const auto& requested : std::array{cartesian, deep_key(false), deep_key(true), fixed})
    {
        SCOPED_TRACE(requested.ancestry.index());
        const auto encoded = AMR::encode_adaptive_key(requested);
        EXPECT_EQ(local::owner_plane_key_wire_bytes(requested), encoded.size());
        EXPECT_EQ(AMR::decode_adaptive_key(encoded), requested);
    }
}

TEST(GeneralVoFOwnerPlaneRequests, DeepKeyGrowthAndOuterWorkspaceRemainReservedAtCacheBoundary)
{
    const PlanarFixture fixture;
    for (const bool product : {false, true})
    {
        SCOPED_TRACE(product);
        const auto requested = deep_key(product);
        const auto decoded = AMR::decode_adaptive_key(AMR::encode_adaptive_key(requested));
        std::size_t decoded_slots = 0, transient_slots = 0;
        if (product)
        {
            const auto& ancestry = std::get<AMR::ProductCellAncestry>(decoded.ancestry);
            ASSERT_EQ(ancestry.base.path.size(), 129u);
            ASSERT_EQ(ancestry.axial.path.size(), 129u);
            ASSERT_EQ(ancestry.base.path.capacity(), 256u);
            ASSERT_EQ(ancestry.axial.path.capacity(), 256u);
            decoded_slots = 256 + 256;
            // The first decoded path survives while the second grows 128 -> 256.
            transient_slots = 256 + 128 + 256;
        }
        else
        {
            const auto& ancestry = std::get<AMR::ConvexCellAncestry>(decoded.ancestry);
            ASSERT_EQ(ancestry.path.size(), 129u);
            ASSERT_EQ(ancestry.path.capacity(), 256u);
            decoded_slots = 256;
            transient_slots = 128 + 256; // Old and new allocation coexist at push 129.
        }
        const auto dynamic_bytes = decoded_slots * sizeof(std::uint64_t);
        const auto transient_bytes = transient_slots * sizeof(std::uint64_t);
        const auto entry = sizeof(local::RequestedPlaneMap::value_type) + 4 * sizeof(void*) + dynamic_bytes;
        EXPECT_EQ(local::owner_plane_key_storage_bytes(decoded), dynamic_bytes);
        EXPECT_EQ(local::owner_plane_cache_entry_bytes(decoded), entry);

        const std::vector incoming{request(1, requested), request(2, requested)};
        std::size_t reply_envelope = incoming.size() * sizeof(OwnerPacket);
        std::size_t outer_bytes = incoming.capacity() * sizeof(OwnerPacket) + 89;
        std::size_t largest_packet = 0;
        for (const auto& packet : incoming)
        {
            reply_envelope += packet.bytes.size() + 216;
            outer_bytes += packet.bytes.capacity();
            largest_packet = std::max(largest_packet, packet.bytes.size());
        }
        const auto key_envelope = sizeof(AMR::AdaptiveCellKey) + 3 * largest_packet;
        // Independent allocation sizes expose the old 2*wire underestimate;
        // neither this transient lower bound nor the caller bytes use a helper.
        EXPECT_GT(transient_bytes, 2 * largest_packet);
        EXPECT_GE(key_envelope, sizeof(AMR::AdaptiveCellKey) + transient_bytes);
        const auto mandatory = reply_envelope + key_envelope;
        EXPECT_EQ(local::owner_plane_request_workspace_bytes(incoming), mandatory);

        std::vector<char> previous;
        for (const bool admit : {false, true})
        {
            SCOPED_TRACE(admit);
            // The caller's retained packets and other workspace already share
            // this ledger when the owner helper considers optional admission.
            const auto caller_bytes = 29 + outer_bytes;
            auto retained = work(caller_bytes + mandatory + entry - (admit ? 0 : 1));
            retained.charge(outer_bytes);
            Diagnostics diagnostics;
            const auto validate_deep = [&](const auto& candidate, std::uint64_t candidate_revision, int)
            {
                if (candidate != requested || candidate_revision != revision)
                    throw std::invalid_argument("Test deep ancestry is not owner authenticated.");
                EXPECT_EQ(local::owner_plane_key_storage_bytes(candidate), dynamic_bytes);
            };
            {
                local::PlaneImportStorage storage(retained, diagnostics);
                const auto outgoing = local::owner_plane_replies(
                    {}, incoming, context(), 0, 3, storage, diagnostics, validate_deep,
                    [&](const auto& candidate) { return fixture.fit(candidate, diagnostics); },
                    [&](wire::Writer& writer, const PLIC::PolyhedralInterface& fitted)
                    {
                        EXPECT_EQ(retained.bytes, caller_bytes + mandatory + (admit ? entry : 0));
                        serialize(writer, fitted);
                    });
                ASSERT_EQ(outgoing.size(), 2u);
                EXPECT_EQ(outgoing[0].bytes, outgoing[1].bytes);
                fixture.check(decode(outgoing[0], requested));
                fixture.check(decode(outgoing[1], requested));
                if (!previous.empty())
                    EXPECT_EQ(previous, outgoing[0].bytes);
                previous = outgoing[0].bytes;
                // Both decoded scratch and the dynamic cache key died before
                // their charges were released; all output/caller bytes remain.
                EXPECT_EQ(retained.bytes, caller_bytes + reply_envelope);
                EXPECT_EQ(diagnostics.owner_reconstruction_calls, admit ? 1u : 2u);
                EXPECT_EQ(diagnostics.peer_cache_admissions, admit ? 1u : 0u);
                EXPECT_EQ(diagnostics.peer_cache_hits, admit ? 1u : 0u);
                EXPECT_EQ(diagnostics.cache_admission_denials, admit ? 0u : 2u);
                EXPECT_EQ(diagnostics.peer_cache_peak_bytes, admit ? entry : 0u);
                EXPECT_EQ(diagnostics.workspace_peak_bytes, caller_bytes + mandatory + (admit ? entry : 0));
                EXPECT_LE(diagnostics.workspace_peak_bytes, retained.limits.byte_budget);
            }
            EXPECT_EQ(retained.bytes, caller_bytes);
            if (admit)
            {
                local::PlaneImportStorage storage(retained, diagnostics);
                EXPECT_THROW((void) local::owner_plane_replies(
                                 {}, incoming, context(), 0, 3, storage, diagnostics, validate_deep,
                                 [&](const auto& candidate) { return fixture.fit(candidate, diagnostics); },
                                 [&](wire::Writer&, const PLIC::PolyhedralInterface&)
                                 {
                                     EXPECT_EQ(retained.bytes, caller_bytes + mandatory + entry);
                                     throw std::runtime_error("Test deep-key serialization rejected.");
                                 }),
                    std::runtime_error);
            }
            EXPECT_EQ(retained.bytes, caller_bytes);
        }
    }
}

TEST(GeneralVoFOwnerPlaneRequests, ExchangedOwnerCacheAndCollectiveRejectIncludeIdleRank)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), ranks = comm->getSize();
    if (ranks != 2 && ranks != 4)
        GTEST_SKIP() << "This registered MPI case requires exactly two or four ranks.";
    const PlanarFixture fixture;
    for (const bool stale : {false, true})
    {
        SCOPED_TRACE(stale);
        std::vector<OwnerPacket> sent;
        if (rank == 1)
        {
            sent.push_back(request(0));
            if (ranks == 2)
                sent.push_back(request(0, key(), stale ? revision + 1 : revision));
        }
        if (ranks == 4 && rank == 2)
            sent.push_back(request(0, key(), stale ? revision + 1 : revision));
        auto incoming = exchange_owner_packets(sent, comm, allowance);
        // This driver explicitly orders authenticated senders, then packet bytes.
        // It does not assume any cross-peer arrival order from the exchange.
        std::sort(incoming.packets.begin(), incoming.packets.end(),
            [](const auto& a, const auto& b) { return std::tie(a.peer, a.bytes) < std::tie(b.peer, b.bytes); });
        auto retained = work();
        Diagnostics diagnostics;
        int threw = 0;
        {
            local::PlaneImportStorage storage(retained, diagnostics);
            std::vector<OwnerPacket> outgoing;
            try
            {
                wire::collective(*comm,
                    [&]
                    {
                        outgoing = local::owner_plane_replies(
                            {}, incoming.packets, context(), rank, ranks, storage, diagnostics,
                            [&](const auto& requested, std::uint64_t requested_revision, int peer)
                            {
                                if (rank != 0)
                                    throw std::invalid_argument("Test request reached a foreign owner.");
                                validate(requested, requested_revision, peer);
                            },
                            [&](const auto& requested) { return fixture.fit(requested, diagnostics); }, serialize);
                    });
            }
            catch (const std::invalid_argument&)
            {
                threw = 1;
            }
            int total_threw = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &threw, &total_threw);
            EXPECT_EQ(total_threw, stale ? ranks : 0);
            if (!stale && total_threw == 0)
            {
                if (rank == 0)
                {
                    EXPECT_EQ(outgoing.size(), 2u);
                    if (outgoing.size() == 2)
                        EXPECT_EQ(outgoing[0].bytes, outgoing[1].bytes);
                }
                const auto returned = exchange_owner_packets(outgoing, comm, allowance);
                const auto expected = rank == 1 ? (ranks == 2 ? 2u : 1u) : rank == 2 ? 1u : 0u;
                EXPECT_EQ(returned.packets.size(), expected);
                for (const auto& packet : returned.packets)
                {
                    EXPECT_EQ(packet.peer, 0);
                    fixture.check(decode(packet));
                }
            }
            EXPECT_EQ(diagnostics.owner_reconstruction_calls, rank == 0 ? 1u : 0u);
            EXPECT_EQ(diagnostics.mixed_cell_fits, rank == 0 ? 1u : 0u);
            EXPECT_EQ(diagnostics.peer_cache_admissions, rank == 0 ? 1u : 0u);
            EXPECT_EQ(diagnostics.peer_cache_hits, rank == 0 && !stale ? 1u : 0u);
            EXPECT_EQ(diagnostics.incoming_peer_requests, rank == 0 ? 2u : 0u);
            EXPECT_EQ(diagnostics.local_result_hits, 0u);
            EXPECT_LE(diagnostics.workspace_peak_bytes, retained.limits.byte_budget);
            if (rank == 3)
            {
                EXPECT_TRUE(sent.empty());
                EXPECT_TRUE(incoming.packets.empty());
                EXPECT_TRUE(outgoing.empty());
            }
        }
        EXPECT_EQ(retained.bytes, 29u);
    }
}
} // namespace
