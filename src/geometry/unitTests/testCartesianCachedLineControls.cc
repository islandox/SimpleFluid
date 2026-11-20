/** Literal cached native-line reads on real serial/MPI communicators. */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include "geometry/mesh/AMR/ContactHaloContext.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/IncidencePools.hh"
#include "geometry/mesh/AMR/PooledLineExchange.hh"
#include "geometry/mesh/AMR/PooledLineRegistry.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
using Access = P::details::GeometryIdentityAccess;
using Wide = std::array<long double, 3>;
using Results = std::map<std::uint64_t, std::vector<AMR::AdaptiveBytes>>;
constexpr std::size_t budget_limit = 8 * 1024 * 1024, original_page = 64 * 1024;
constexpr D::ContactHaloContext kernel_context{7, 11, 13, 17, 19};
constexpr D::ContactHaloContext read_context{23, 29, 31, 37, 41};
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

AMR::AdaptiveBytes line_key()
{
    D::Writer writer;
    writer.bytes(std::array<char, 1>{'d'});
    writer.bytes(P::canonical_key(Access::line({0, {1, 0, 0}, {0, 0, 0}})));
    return std::move(writer.data);
}
AMR::AdaptiveBytes support_key(unsigned query)
{
    return {'s', char(query)};
}
auto point(unsigned x)
{
    return Access::point(Wide{static_cast<long double>(x), 0, 0});
}
unsigned first_x(unsigned query)
{
    return query ? query - 1 : 0;
}
unsigned last_x(unsigned query)
{
    return query ? query : 16;
}
unsigned result_count(unsigned query)
{
    return query ? 2 : 17;
}
unsigned result_x(unsigned query, unsigned position)
{
    return query ? first_x(query) + position : position;
}

// Values are checked against the independently specified integer geometry,
// never against the cached result being tested or a rank-local pool numbering.
void check_point(const D::IncidencePools::Point& actual, unsigned x)
{
    EXPECT_EQ(Access::point(actual), (Wide{static_cast<long double>(x), 0, 0}));
    EXPECT_EQ(P::canonical_key(actual), P::canonical_key(point(x)));
}
void check_original(const D::PooledLineSubscriber& actual, const D::PooledLineSubscriber& expected)
{
    EXPECT_EQ(actual.support, expected.support);
    EXPECT_EQ(actual.first, expected.first);
    EXPECT_EQ(actual.last, expected.last);
    EXPECT_EQ(actual.resident, expected.resident);
    EXPECT_EQ(actual.support_authority, expected.support_authority);
    EXPECT_EQ(actual.emission, expected.emission);
    EXPECT_EQ(actual.image, expected.image);
    EXPECT_EQ(actual.edge, expected.edge);
    EXPECT_EQ(actual.source_generation, expected.source_generation);
    EXPECT_EQ(actual.native_source, expected.native_source);
    EXPECT_EQ(actual.result_state, D::PooledLineSubscriber::ResultState::Unset);
    EXPECT_EQ(actual.result.capacity(), 0u);
}

struct RegistrySnapshot
{
    std::size_t processed, retained, metadata;
    std::optional<std::size_t> dirty;
    std::array<std::size_t, 21> storage;
    AMR::AdaptiveBytes values;
    bool operator==(const RegistrySnapshot&) const = default;
};

struct Fixture
{
    const Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    std::shared_ptr<D::Budget> budget;
    std::shared_ptr<D::CartesianAuthorityArena> arena;
    std::unique_ptr<D::IdentityBudget> identities;
    std::unique_ptr<D::IncidencePools> pools;
    std::unique_ptr<D::PooledLineRegistry> registry;
    std::vector<D::PooledLineCompletionRequest> requests;
    Results original_results;
    std::size_t original_callbacks = 0;
    unsigned witness_layer;

    explicit Fixture(bool empty = false, unsigned initial_layer = 0,
        std::shared_ptr<D::Budget> caller_budget = std::make_shared<D::Budget>(D::Budget{budget_limit}),
        D::IdentityPoolStorage storage = D::IdentityPoolStorage::Flat)
        : budget(std::move(caller_budget)), arena(std::make_shared<D::CartesianAuthorityArena>(*budget, budget)),
          identities(std::make_unique<D::IdentityBudget>()),
          pools(std::make_unique<D::IncidencePools>(*budget, *identities, budget, storage)),
          registry(std::make_unique<D::PooledLineRegistry>(*budget, *pools)), witness_layer(initial_layer)
    {
        std::vector<D::PooledLineEdit> edits;
        D::collective(*comm,
            [&]
            {
                registry->begin_replay(true);
                // Deliberately distinct local slot namespaces on empty owners.
                // These are ordinary literal points, with no source admission.
                for (int i = 0; i < comm->getRank(); ++i)
                    (void) pools->points.intern(point(100 + unsigned(i)));
                if (!empty && comm->getRank() == 0)
                    for (unsigned query = 0; query <= 16; ++query)
                    {
                        auto member = literal_member(query);
                        const auto line = pools->lines.intern(line_key());
                        requests.push_back({line, member});
                        edits.push_back({line, std::move(member), false});
                    }
            });
        std::size_t cursor = 0;
        D::exchange_pooled_line_edits(
            *registry, *pools,
            [&](D::PooledLineEdit& edit)
            {
                if (cursor == edits.size())
                    return false;
                edit = edits[cursor++];
                return true;
            },
            witness_layer, kernel_context, comm, *budget, original_page,
            [&](std::size_t line, const D::PooledLineSubscriber& member)
            {
                EXPECT_EQ(comm->getRank(), 0);
                EXPECT_EQ(pools->lines.get(line), line_key());
                EXPECT_TRUE(member.native_source);
                EXPECT_EQ(member.source_generation, 100u + member.emission);
                EXPECT_LE(member.emission, 16u);
                const auto query = unsigned(member.emission);
                EXPECT_EQ(member.result_size(), result_count(query));
                auto& result = original_results[member.emission];
                for (std::size_t i = 0; i < member.result_size(); ++i)
                {
                    const auto value = pools->points.get(member.result_at(i));
                    check_point(value, result_x(query, unsigned(i)));
                    result.push_back(P::canonical_key(value));
                }
                ++original_callbacks;
            },
            {}, true);
        D::collective(*comm, [&] { registry->finish_replay(witness_layer); });
        EXPECT_EQ(original_callbacks, !empty && comm->getRank() == 0 ? 17u : 0u);
    }
    ~Fixture()
    {
        registry.reset();
        pools.reset();
        // The literal identity ledger owns external pins until its natural
        // destructor. Only then retire its caller admission scope, never its
        // cumulative counter or any still-live receipt allocation.
        const auto external_identity_charge = identities->admitted_bytes();
        identities.reset();
        budget->used -= external_identity_charge;
        arena->reap();
    }
    D::PooledLineSubscriber literal_member(unsigned query)
    {
        return literal_member(query, support_key(query));
    }
    D::PooledLineSubscriber literal_member(unsigned query, const AMR::AdaptiveBytes& support)
    {
        D::PooledLineSubscriber member;
        member.support = pools->supports.intern(support);
        member.first = pools->points.intern(point(first_x(query)));
        member.last = pools->points.intern(point(last_x(query)));
        member.resident = 0;
        member.support_authority = D::authority(support, comm->getSize());
        member.emission = query;
        member.edge = 1000 + query;
        member.source_generation = 100 + query;
        member.native_source = true;
        return member;
    }
    int authority_rank() const { return D::authority(line_key(), comm->getSize()); }
    std::size_t fragmented_page() const
    {
        // Account for every peer's envelope and the complete member reference.
        // At most three literal points fit in a reply page: long output needs
        // six data pages, followed by an authenticated empty terminal page.
        const auto request_bytes = 88 + line_key().size() + support_key(0).size() + P::encoded_identity_size(point(0)) +
                                   P::encoded_identity_size(point(16));
        auto minimum_point_bytes = 8 + P::encoded_identity_size(point(0));
        auto maximum_point_bytes = minimum_point_bytes;
        for (unsigned x = 1; x <= 16; ++x)
        {
            const auto point_bytes = 8 + P::encoded_identity_size(point(x));
            minimum_point_bytes = std::min(minimum_point_bytes, point_bytes);
            maximum_point_bytes = std::max(maximum_point_bytes, point_bytes);
        }
        EXPECT_GT(4 * minimum_point_bytes, 3 * maximum_point_bytes);
        const auto bytes = std::size_t(comm->getSize()) * (14 * 8 + sizeof(SimpleFluid::OwnerPacket)) + request_bytes +
                           32 + 3 * maximum_point_bytes;
        EXPECT_LT(bytes, original_page);
        return bytes;
    }
    RegistrySnapshot snapshot() const
    {
        const auto s = registry->active_storage();
        RegistrySnapshot saved{registry->processed_lines(), registry->retained_bytes(), registry->metadata_bytes(),
            registry->next_dirty(),
            {s.lines, s.explicit_records, s.run_records, s.runs, s.explicit_bytes, s.run_bytes, s.source_records,
                s.source_families, s.source_bytes, s.source_catalog_bytes, s.source_maps, s.source_mapped_components,
                s.source_nonforward_records, s.source_result_selectors, s.retained_explicit_records,
                s.retained_run_records, s.retained_source_records, pools->points.size(), pools->lines.size(),
                pools->supports.size(), original_callbacks},
            {}};
        D::Writer writer;
        if (comm->getRank() == authority_rank() && s.lines && !saved.dirty)
            for (std::size_t ordinal = 0; ordinal < 17; ++ordinal)
            {
                const auto view = registry->query_result(*pools->lines.find(line_key()), ordinal);
                const auto& member = view.subscriber;
                const auto support = pools->supports.get(member.support);
                writer.bytes(support.bytes());
                for (const auto word : {std::uint64_t(member.resident), std::uint64_t(member.support_authority),
                         member.emission, std::uint64_t(member.image), member.edge, member.source_generation,
                         std::uint64_t(member.native_source), std::uint64_t(member.result_state),
                         std::uint64_t(view.result_size())})
                    writer.word(word);
                writer.bytes(P::canonical_key(pools->points.get(member.first)));
                writer.bytes(P::canonical_key(pools->points.get(member.last)));
                for (std::size_t i = 0; i < view.result_size(); ++i)
                    writer.bytes(P::canonical_key(pools->points.get(view.result_at(i))));
            }
        saved.values = std::move(writer.data);
        return saved;
    }
    void check_completion(std::size_t line, const D::PooledLineCompletion& completion, unsigned layer) const
    {
        EXPECT_EQ(comm->getRank(), 0);
        const auto& proof = *completion.proof();
        const auto query = unsigned(proof.original_request().emission);
        ASSERT_LE(query, 16u);
        const auto& original = requests.at(query);
        EXPECT_EQ(line, original.line);
        check_original(proof.original_request(), original.original);
        EXPECT_EQ(proof.context(), read_context);
        EXPECT_EQ(proof.layer(), layer);
        EXPECT_EQ(proof.kernel_witness_layer(), witness_layer);
        EXPECT_TRUE(proof.is_cached_read());
        EXPECT_EQ((AMR::AdaptiveBytes{proof.line_key().begin(), proof.line_key().end()}), line_key());
        EXPECT_EQ((AMR::AdaptiveBytes{proof.support_key().begin(), proof.support_key().end()}), support_key(query));
        check_point(proof.endpoint(0), first_x(query));
        check_point(proof.endpoint(1), last_x(query));
        EXPECT_EQ(proof.point_pool_identity(), &pools->points);
        EXPECT_EQ(proof.allocation_budget(), budget.get());
        EXPECT_TRUE(proof.has_arena_keeper());
        EXPECT_EQ(completion.allocation_budget(), budget.get());
        EXPECT_TRUE(completion.has_arena_keeper());
        ASSERT_EQ(completion.result().size(), result_count(query));
        EXPECT_EQ(completion.result_capacity(), completion.result().size());
        std::vector<AMR::AdaptiveBytes> observed;
        for (std::size_t i = 0; i < completion.result().size(); ++i)
        {
            const auto slot = completion.result()[i];
            const auto expected_x = result_x(query, unsigned(i));
            const auto expected_slot = pools->points.find(point(expected_x));
            ASSERT_TRUE(expected_slot);
            EXPECT_EQ(slot, *expected_slot);
            check_point(pools->points.get(slot), expected_x);
            observed.push_back(P::canonical_key(pools->points.get(slot)));
        }
        EXPECT_EQ(observed, original_results.at(query));
    }
    std::size_t fetch(
        D::PooledCachedLineFault fault = {}, unsigned layer = 1, const D::PooledLineCompletionReceive& receive = {})
    {
        std::size_t cursor = 0, callbacks = 0;
        D::fetch_pooled_line_completions(
            *registry, *pools,
            [&](D::PooledLineCompletionRequest& request)
            {
                if (cursor == requests.size())
                    return false;
                const auto& saved = requests[cursor++];
                request = saved;
                // Copying an empty vector does not preserve its reserved
                // capacity. Deliver that intentionally invalid fixture input
                // at the same callback boundary the production code validates.
                if (saved.original.result.capacity())
                    request.original.result.reserve(saved.original.result.capacity());
                return true;
            },
            layer, read_context, comm, *budget, fragmented_page(),
            [&](std::size_t line, D::PooledLineCompletion&& completion)
            {
                check_completion(line, completion, layer);
                ++callbacks;
                if (receive)
                    receive(line, std::move(completion));
            },
            budget, arena, fault);
        return callbacks;
    }
};

void expect_all_ranks_reject(const std::function<void()>& operation)
{
    int local = 0;
    try
    {
        operation();
    }
    catch (const std::exception&)
    {
        local = 1;
    }
    int all = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &local, &all);
    EXPECT_EQ(all, 1);
}
void clean_retry(bool empty = false)
{
    Fixture clean(empty);
    const auto before = clean.snapshot();
    const auto used = clean.budget->used;
    EXPECT_EQ(clean.fetch(), !empty && clean.comm->getRank() == 0 ? 17u : 0u);
    clean.arena->reap();
    EXPECT_EQ(clean.snapshot(), before);
    EXPECT_EQ(clean.budget->used, used);
}
void reject_fault(D::PooledCachedLineFault fault, bool empty)
{
    Fixture fixture(empty);
    const auto before = fixture.snapshot();
    const auto used = fixture.budget->used, spent = fixture.budget->cumulative;
    expect_all_ranks_reject([&] { fixture.fetch(fault); });
    fixture.arena->reap();
    EXPECT_EQ(fixture.snapshot(), before);
    EXPECT_EQ(fixture.budget->used, used);
    EXPECT_GE(fixture.budget->cumulative, spent);
    clean_retry(empty);
}

TEST(CartesianCachedLineControls, LiteralFragmentedResultsPreserveCompleteOriginalAndKernelWitness)
{
    Fixture fixture;
    const auto before = fixture.snapshot();
    const auto used = fixture.budget->used, spent = fixture.budget->cumulative;
    EXPECT_EQ(fixture.fetch(), fixture.comm->getRank() == 0 ? 17u : 0u);
    fixture.arena->reap();
    EXPECT_EQ(fixture.snapshot(), before);
    EXPECT_EQ(fixture.budget->used, used);
    EXPECT_GT(fixture.budget->cumulative, spent);
    RecordProperty("cached_control_cumulative_admission_bytes", std::to_string(fixture.budget->cumulative - spent));
    RecordProperty("cached_control_page_bytes", std::to_string(fixture.fragmented_page()));
    RecordProperty("long_result_literal_points", 17);
    RecordProperty("long_result_data_pages_lower_bound", 6);
    RecordProperty("kernel_witness_layer", 0);
    RecordProperty("current_attestation_layer", 1);
    EXPECT_EQ(fixture.registry->processed_lines(), fixture.comm->getRank() == fixture.authority_rank() ? 1u : 0u);
    // A second read attests another effective layer without running the kernel.
    EXPECT_EQ(fixture.fetch({}, 2), fixture.comm->getRank() == 0 ? 17u : 0u);
    EXPECT_EQ(fixture.snapshot(), before);
    EXPECT_EQ(fixture.budget->used, used);
}

TEST(CartesianCachedLineControls, EmptyInvocationAuthenticatesBothEnvelopeFamiliesAndHasMeasuredControlCost)
{
    Fixture fixture(true);
    const auto before = fixture.snapshot();
    const auto used = fixture.budget->used, spent = fixture.budget->cumulative;
    EXPECT_EQ(fixture.fetch(), 0u);
    fixture.arena->reap();
    EXPECT_EQ(fixture.snapshot(), before);
    EXPECT_EQ(fixture.budget->used, used);
    EXPECT_GT(fixture.budget->cumulative, spent);
    RecordProperty("empty_control_cumulative_admission_bytes", std::to_string(fixture.budget->cumulative - spent));
    RecordProperty("empty_logical_request_count", 0);
    RecordProperty("empty_envelope_families", 2);
}

TEST(CartesianCachedLineControls, RankSelectiveMalformedRequestAndReplyEnvelopesRejectCollectivelyAndRecover)
{
    const auto comm = Tpetra::getDefaultComm();
    for (bool reply : {false, true})
        for (auto kind :
            {D::PooledCachedLineFault::Kind::MissingEnvelope, D::PooledCachedLineFault::Kind::DuplicateEnvelope,
                D::PooledCachedLineFault::Kind::StaleContext, D::PooledCachedLineFault::Kind::TruncatedEnvelope})
        {
            SCOPED_TRACE(reply);
            SCOPED_TRACE(static_cast<int>(kind));
            D::PooledCachedLineFault fault;
            fault.kind = kind;
            fault.reply = reply;
            fault.rank = reply ? D::authority(line_key(), comm->getSize()) : 0;
            // Explicit peers ensure the nonempty first-wave record is targeted.
            fault.peer = reply ? 0 : D::authority(line_key(), comm->getSize());
            reject_fault(fault, false);
        }
    for (auto kind : {D::PooledCachedLineFault::Kind::MissingCompletion, D::PooledCachedLineFault::Kind::DuplicateChunk,
             D::PooledCachedLineFault::Kind::StaleGeneration})
    {
        SCOPED_TRACE(static_cast<int>(kind));
        D::PooledCachedLineFault fault;
        fault.kind = kind;
        fault.reply = true;
        fault.rank = D::authority(line_key(), comm->getSize());
        fault.peer = 0;
        reject_fault(fault, false);
    }
}

TEST(CartesianCachedLineControls, TrulyEmptyRequestAndTerminalReplyControlsRejectMissingAndMalformedEnvelopes)
{
    for (bool reply : {false, true})
        for (auto kind : {D::PooledCachedLineFault::Kind::MissingEnvelope,
                 D::PooledCachedLineFault::Kind::DuplicateEnvelope, D::PooledCachedLineFault::Kind::StaleContext,
                 D::PooledCachedLineFault::Kind::TruncatedEnvelope, D::PooledCachedLineFault::Kind::InvalidEmptyCount})
        {
            SCOPED_TRACE(reply);
            SCOPED_TRACE(static_cast<int>(kind));
            D::PooledCachedLineFault fault;
            fault.kind = kind;
            fault.reply = reply;
            fault.rank = Tpetra::getDefaultComm()->getSize() - 1;
            fault.peer = 0;
            reject_fault(fault, true);
        }
}

TEST(CartesianCachedLineControls, StaleOriginalMetadataAndPhysicalEndpointsRejectExactLookup)
{
    // Each mutation starts with an independent completed actual kernel pass.
    for (unsigned mutation = 0; mutation < 13; ++mutation)
    {
        SCOPED_TRACE(mutation);
        Fixture fixture;
        D::collective(*fixture.comm,
            [&]
            {
                if (fixture.comm->getRank() != 0)
                    return;
                auto& request = fixture.requests.front();
                auto& member = request.original;
                switch (mutation)
                {
                    case 0:
                        ++member.emission;
                        break;
                    case 1:
                        ++member.edge;
                        break;
                    case 2:
                        member.image = 1;
                        break;
                    case 3:
                        ++member.source_generation;
                        break;
                    case 4:
                        member.native_source = false;
                        break;
                    case 5:
                        member.resident = fixture.comm->getSize();
                        break;
                    case 6:
                        member.support_authority = fixture.comm->getSize();
                        break;
                    case 7:
                        member.first = *fixture.pools->points.find(point(1));
                        break;
                    case 8:
                        member.last = *fixture.pools->points.find(point(15));
                        break;
                    case 9:
                        std::swap(member.first, member.last);
                        break;
                    case 10:
                        member.support = fixture.requests.at(1).original.support;
                        break;
                    case 11:
                        member.result_state = D::PooledLineSubscriber::ResultState::ForwardEndpoints;
                        break;
                    case 12:
                        member.result.reserve(1);
                        break;
                }
            });
        const auto before = fixture.snapshot();
        const auto used = fixture.budget->used;
        expect_all_ranks_reject([&] { fixture.fetch(); });
        fixture.arena->reap();
        EXPECT_EQ(fixture.snapshot(), before);
        EXPECT_EQ(fixture.budget->used, used);
        clean_retry();
    }
}

TEST(CartesianCachedLineControls, MissingForeignWrongLayerAndUncomputedGroupsRejectWithoutNumericalWork)
{
    for (unsigned mutation = 0; mutation < 4; ++mutation)
    {
        SCOPED_TRACE(mutation);
        Fixture fixture(false, mutation == 2 ? 2 : 0);
        D::collective(*fixture.comm,
            [&]
            {
                if (mutation == 0 && fixture.comm->getRank() == 0)
                {
                    auto absent = line_key();
                    absent.push_back('x');
                    fixture.requests.front().line = fixture.pools->lines.intern(std::move(absent));
                }
                if (mutation == 1 && fixture.comm->getRank() == 0)
                {
                    auto foreign = support_key(17);
                    auto& member = fixture.requests.front().original;
                    member.support = fixture.pools->supports.intern(foreign);
                    member.support_authority = D::authority(foreign, fixture.comm->getSize());
                }
                if (mutation == 3 && fixture.comm->getRank() == fixture.authority_rank())
                {
                    // An actual new subscription is admitted but intentionally
                    // not recomputed. It makes this complete line dirty; no
                    // previous completed member can attest that changed group.
                    fixture.registry->begin_replay(false);
                    fixture.registry->set_layer(1);
                    auto member = fixture.literal_member(16);
                    member.emission = 99;
                    const D::PooledLineEdit edit{*fixture.pools->lines.find(line_key()), std::move(member), false};
                    fixture.registry->apply(std::span{&edit, 1});
                    EXPECT_TRUE(fixture.registry->next_dirty());
                }
            });
        const auto before = fixture.snapshot();
        const auto used = fixture.budget->used;
        expect_all_ranks_reject([&] { fixture.fetch(); });
        fixture.arena->reap();
        EXPECT_EQ(fixture.snapshot(), before);
        EXPECT_EQ(fixture.budget->used, used);
        clean_retry();
    }
}

TEST(CartesianCachedLineControls, RankLocalTinyAdmissionRejectsBeforeTransportAndRetiresTemporaryStorage)
{
    Fixture fixture;
    const auto before = fixture.snapshot();
    const auto used = fixture.budget->used, spent = fixture.budget->cumulative;
    const auto limit = fixture.budget->limit;
    if (fixture.comm->getRank() == 0)
        fixture.budget->limit = used + 1; // Cannot admit even the first complete request-set node.
    expect_all_ranks_reject([&] { fixture.fetch(); });
    fixture.budget->limit = limit;
    fixture.arena->reap();
    EXPECT_EQ(fixture.budget->used, used);
    EXPECT_GE(fixture.budget->cumulative, spent);
    EXPECT_EQ(fixture.snapshot(), before);
    RecordProperty("rank_zero_remaining_admission_bytes", 1);
    EXPECT_EQ(fixture.fetch(), fixture.comm->getRank() == 0 ? 17u : 0u);
    EXPECT_EQ(fixture.snapshot(), before);
    EXPECT_EQ(fixture.budget->used, used);
    clean_retry();
}

TEST(CartesianCachedLineControls, OneImmutableResultOwnerPreservesAllocationAndSeparatelyReleasesProof)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{budget_limit});
    std::weak_ptr<D::Budget> budget_observer = budget;
    D::PooledLineCompletionView retained;
    std::weak_ptr<const D::PooledLineCompletion> completion_observer;
    std::shared_ptr<const D::PooledLineCompletion::Proof> proof;
    std::weak_ptr<const D::PooledLineCompletion::Proof> proof_observer;
    std::size_t result_bytes = 0;
    {
        Fixture fixture(false, 0, budget);
        const auto before = fixture.snapshot();
        const auto used = budget->used;
        fixture.fetch({}, 1,
            [&](std::size_t, D::PooledLineCompletion&& completion)
            {
                if (completion.proof()->original_request().emission)
                    return;
                const auto* data = completion.result().data();
                const auto capacity = completion.result_capacity();
                retained = std::move(completion).share_immutable();
                EXPECT_EQ(retained->result().data(), data);
                EXPECT_EQ(retained->result_capacity(), capacity);
                result_bytes = capacity * sizeof(std::size_t);
                completion_observer = retained;
                proof = retained->proof();
                proof_observer = proof;
            });
        EXPECT_EQ(fixture.snapshot(), before);
        if (fixture.comm->getRank() == 0)
        {
            ASSERT_TRUE(retained);
            EXPECT_EQ(result_bytes, 17 * sizeof(std::size_t));
            EXPECT_GE(budget->used, used + result_bytes);
            auto second_handle = retained;
            EXPECT_EQ(second_handle.get(), retained.get());
            EXPECT_EQ(second_handle->result().data(), retained->result().data());
            const auto retained_used = budget->used;
            second_handle.reset();
            EXPECT_EQ(budget->used, retained_used);
            EXPECT_FALSE(completion_observer.expired());
        }
        fixture.arena->reap();
    } // Registry, pools, literal ledger pins and arena are naturally destroyed.
    const auto comm = Tpetra::getDefaultComm();
    const auto spent = budget->cumulative;
    if (comm->getRank() == 0)
    {
        ASSERT_TRUE(retained);
        EXPECT_GT(budget->used, result_bytes);
        const auto before_drop = budget->used;
        budget.reset(); // Receipt/proof keepers now own the source budget lifetime.
        EXPECT_FALSE(budget_observer.expired());
        retained.reset();
        EXPECT_TRUE(completion_observer.expired());
        EXPECT_FALSE(proof_observer.expired());
        EXPECT_EQ(proof->context(), read_context);
        check_point(proof->endpoint(0), 0);
        check_point(proof->endpoint(1), 16);
        EXPECT_LE(budget_observer.lock()->used, before_drop - result_bytes);
        completion_observer.reset(); // Retire the actual shared deleter/control.
        EXPECT_GT(budget_observer.lock()->used, 0u);
        proof.reset();
        EXPECT_TRUE(proof_observer.expired());
        // Observe final accounting across the last weak-control destruction.
        auto final_accounting = budget_observer.lock();
        ASSERT_TRUE(final_accounting);
        proof_observer.reset();
        EXPECT_EQ(final_accounting->used, 0u);
        EXPECT_EQ(final_accounting->cumulative, spent);
        final_accounting.reset();
    }
    else
    {
        EXPECT_EQ(budget->used, 0u);
        EXPECT_EQ(budget->cumulative, spent);
        budget.reset();
    }
    EXPECT_TRUE(budget_observer.expired());
}

// A borrow cannot be forged from arbitrary descriptors or copied into an owner.
static_assert(!std::is_default_constructible_v<D::PooledLineBorrowedCompletion>);
static_assert(!std::is_copy_constructible_v<D::PooledLineBorrowedCompletion>);
static_assert(!std::is_move_constructible_v<D::PooledLineBorrowedCompletion>);
static_assert(!std::is_constructible_v<D::PooledLineBorrowedCompletion, std::size_t,
    const D::PooledLineSubscriber&, std::span<const std::size_t>, D::ContactHaloContext, unsigned,
    const D::IncidencePools&>);

struct BorrowedRun
{
    std::vector<std::pair<bool, AMR::AdaptiveBytes>> data;
    std::vector<unsigned> attempted, sealed;
    std::size_t consumed = 0;
    bool rejected = false, retained_at_failure = false;
};
AMR::AdaptiveBytes borrowed_support_key(unsigned query)
{
    D::Writer key;
    key.word(0x5346434152543631ULL);
    key.bytes(std::array<char, 1>{'d'});
    // Region, axis, level, plane and the two transverse coordinates.
    for (const auto value : std::array<std::uint64_t, 6>{0, 0, 0, query, 0, query == 16 ? 256u : 0u})
        key.word(value);
    return std::move(key.data);
}
std::vector<unsigned> expected_borrowed_order()
{
    std::vector<unsigned> all;
    for (unsigned query = 0; query <= 16; ++query)
        all.push_back(query);
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    // Member order is resident, support authority, full support bytes, then
    // emission/image/edge. Do not assume one rank's local insertion order.
    std::sort(all.begin(), all.end(),
        [&](unsigned a, unsigned b)
        {
            const auto aa = D::authority(support_key(a), ranks), ba = D::authority(support_key(b), ranks);
            return aa == ba ? support_key(a) < support_key(b) : aa < ba;
        });
    return all;
}
BorrowedRun fresh_borrowed_exchange(bool use_consumer, D::PooledNativeLineFault fault = {}, bool throw_consumer = false,
    bool late_throw = false, D::IdentityPoolStorage storage = D::IdentityPoolStorage::Flat, bool framed = false)
{
    const auto key_for = framed ? borrowed_support_key : support_key;
    BorrowedRun run;
    auto budget = std::make_shared<D::Budget>(D::Budget{budget_limit});
    D::PooledLineCompletionView retained;
    std::size_t retained_result_bytes = 0;
    {
        Fixture fixture(true, 0, budget, storage);
        auto& pools = *fixture.pools;
        std::vector<D::PooledLineEdit> edits;
        D::collective(*fixture.comm,
            [&]
            {
                fixture.registry->begin_replay(true);
                if (fixture.comm->getRank() != 0)
                    return;
                for (unsigned query = 0; query <= 16; ++query)
                {
                    auto original = fixture.literal_member(query, key_for(query));
                    EXPECT_EQ(pools.supports.is_compact(original.support),
                        framed && storage == D::IdentityPoolStorage::Persistent &&
                            std::numeric_limits<std::size_t>::digits > 32 && query != 16);
                    // One long member appears between endpoint-only members in
                    // full support order. Integer points independently determine
                    // its 17 results and the two results of every other member.
                    original.first = pools.points.intern(point(query == 8 ? 0 : query < 8 ? query : query - 1));
                    original.last = pools.points.intern(point(query == 8 ? 16 : query < 8 ? query + 1 : query));
                    const auto line = pools.lines.intern(line_key());
                    fixture.requests.push_back({line, original});
                    edits.push_back({line, std::move(original), false});
                }
            });
        std::size_t cursor = 0;
        bool materialized = false;
        const auto check = [&](std::size_t line, const auto& proof, std::span<const std::size_t> result)
        {
            const auto& original = proof.original_request();
            const auto query = unsigned(original.emission);
            EXPECT_LE(query, 16u);
            if (query > 16)
                throw std::invalid_argument("Foreign test member.");
            check_original(original, fixture.requests.at(query).original);
            EXPECT_EQ(line, fixture.requests.at(query).line);
            EXPECT_EQ(proof.context(), kernel_context);
            EXPECT_EQ(proof.layer(), 0u);
            EXPECT_EQ(proof.kernel_witness_layer(), 0u);
            EXPECT_EQ(proof.point_pool_identity(), &pools.points);
            EXPECT_TRUE(std::ranges::equal(proof.line_key(), line_key()));
            EXPECT_TRUE(std::ranges::equal(proof.support_key(), key_for(query)));
            const auto first = query == 8 ? 0 : query < 8 ? query : query - 1;
            const auto last = query == 8 ? 16 : query < 8 ? query + 1 : query;
            check_point(proof.endpoint(0), first);
            check_point(proof.endpoint(1), last);
            EXPECT_EQ(result.size(), query == 8 ? 17u : 2u);
            for (std::size_t i = 0; i < result.size(); ++i)
                check_point(pools.points.get(result[i]), query == 8 ? unsigned(i) : i ? last : first);
            return query;
        };
        D::PooledLineCompletionConsume consume;
        if (use_consumer)
            consume = [&](std::size_t line, const D::PooledLineBorrowedCompletion& completion)
            {
                const auto query = check(line, completion, completion.result());
                run.attempted.push_back(query);
                if (throw_consumer || (late_throw && run.consumed && !run.sealed.empty()))
                    throw std::invalid_argument("Intentional authenticated-consumer failure.");
                if (materialized || query == 8)
                    return false;
                ++run.consumed;
                return true;
            };
        const auto spent = budget->cumulative;
        try
        {
            D::exchange_pooled_line_edits(
                *fixture.registry, pools,
                [&](D::PooledLineEdit& edit)
                {
                    if (cursor == edits.size())
                        return false;
                    edit = edits[cursor++];
                    return true;
                },
                0, kernel_context, fixture.comm, *budget, original_page,
                [&](std::size_t, const D::PooledLineSubscriber&) { ADD_FAILURE() << "Native fallback lost receipt."; },
                {}, true, fault,
                [&](bool reply, unsigned, std::uint64_t, std::uint64_t, std::span<const SimpleFluid::OwnerPacket> packets)
                {
                    for (const auto& packet : packets)
                        run.data.emplace_back(reply, packet.bytes);
                },
                [&](std::size_t line, const D::PooledLineSubscriber& decoded) -> std::optional<D::PooledLineSubscriber>
                {
                    const auto query = std::size_t(decoded.emission);
                    if (query >= fixture.requests.size() || line != fixture.requests[query].line)
                        return std::nullopt;
                    return fixture.requests[query].original;
                },
                [&](std::size_t line, D::PooledLineCompletion&& completion)
                {
                    run.sealed.push_back(check(line, *completion.proof(), completion.result()));
                    materialized = true;
                    if (late_throw)
                    {
                        // Keep the genuine owning fallback alive while a later
                        // borrow throws and all other pending outputs unwind.
                        retained_result_bytes = completion.result_capacity() * sizeof(std::size_t);
                        retained = std::move(completion).share_immutable();
                    }
                },
                budget, fixture.arena, consume);
        }
        catch (const std::exception&)
        {
            run.rejected = true;
            run.retained_at_failure = bool(retained);
        }
        int rejected = run.rejected, all_rejected = 0;
        Teuchos::reduceAll(*fixture.comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
        EXPECT_EQ(all_rejected, run.rejected);
        EXPECT_GE(budget->cumulative, spent);
    }
    // Registry, pools, external identity ledger and temporary controls have
    // died. An explicitly retained owning fallback must still keep its exact
    // result/proof admission alive; releasing it must not refund cumulative.
    if (retained)
    {
        EXPECT_TRUE(run.retained_at_failure);
        EXPECT_EQ(retained->proof()->original_request().emission, 8u);
        EXPECT_EQ(retained->result().size(), 17u);
        EXPECT_EQ(retained_result_bytes, 17 * sizeof(std::size_t));
        EXPECT_GT(budget->used, retained_result_bytes);
        check_point(retained->proof()->endpoint(0), 0);
        check_point(retained->proof()->endpoint(1), 16);
        const auto spent = budget->cumulative;
        retained.reset();
        EXPECT_EQ(budget->cumulative, spent);
    }
    // This checks exact admission retirement, not allocator-requested-live bytes.
    EXPECT_EQ(budget->used, 0u);
    EXPECT_GT(budget->cumulative, 0u);
    return run;
}

TEST(CartesianCachedLineControls, AuthenticatedBorrowAvoidsReceiptsAndPreservesMixedCallbackOrderAndV2Bytes)
{
    const auto normal = fresh_borrowed_exchange(false);
    const auto borrowed = fresh_borrowed_exchange(true);
    const auto literal_frames = fresh_borrowed_exchange(true, {}, false, false, D::IdentityPoolStorage::Flat, true);
    const auto compact = fresh_borrowed_exchange(true, {}, false, false, D::IdentityPoolStorage::Persistent, true);
    EXPECT_FALSE(normal.rejected);
    EXPECT_FALSE(borrowed.rejected);
    EXPECT_FALSE(literal_frames.rejected);
    EXPECT_FALSE(compact.rejected);
    EXPECT_EQ(normal.data, borrowed.data);
    EXPECT_EQ(compact.data, literal_frames.data);
    EXPECT_EQ(compact.attempted, literal_frames.attempted);
    EXPECT_EQ(compact.sealed, literal_frames.sealed);
    EXPECT_EQ(compact.consumed, literal_frames.consumed);
    EXPECT_EQ(borrowed.attempted, normal.sealed);
    if (Tpetra::getDefaultComm()->getRank() == 0)
    {
        const auto all = expected_borrowed_order();
        const auto changed = std::find(all.begin(), all.end(), 8u);
        const std::vector<unsigned> fallback(changed, all.end());
        EXPECT_EQ(normal.sealed, all);
        EXPECT_EQ(borrowed.sealed, fallback);
        EXPECT_EQ(borrowed.consumed, std::size_t(changed - all.begin()));
        EXPECT_GT(borrowed.consumed, 0u);
        EXPECT_GT(fallback.size(), 1u);
    }
    else
    {
        EXPECT_TRUE(borrowed.attempted.empty());
        EXPECT_TRUE(borrowed.sealed.empty());
        EXPECT_EQ(borrowed.consumed, 0u);
    }
}

TEST(CartesianCachedLineControls, LateBorrowFailureRetiresMixedConsumedPendingAndRetainedOwningOutputs)
{
    Fixture source;
    const auto accepted = source.snapshot();
    const auto source_used = source.budget->used, source_spent = source.budget->cumulative;
    const auto failed = fresh_borrowed_exchange(true, {}, false, true);
    EXPECT_TRUE(failed.rejected);
    if (source.comm->getRank() == 0)
    {
        const auto all = expected_borrowed_order();
        const auto changed = std::find(all.begin(), all.end(), 8u);
        const auto consumed = std::size_t(changed - all.begin());
        EXPECT_GT(consumed, 0u);
        EXPECT_GT(all.end() - changed, 1);
        // The long result is sealed after a positive endpoint-consumed prefix.
        // The next authenticated borrow throws with that owner still live and
        // the failing reply buffer and remaining pending outputs admitted.
        if (all.end() - changed > 1)
        {
            const std::vector<unsigned> attempted(all.begin(), changed + 2);
            EXPECT_EQ(failed.attempted, attempted);
        }
        EXPECT_EQ(failed.consumed, consumed);
        EXPECT_EQ(failed.sealed, (std::vector<unsigned>{8u}));
        EXPECT_TRUE(failed.retained_at_failure);
    }
    else
    {
        EXPECT_TRUE(failed.attempted.empty());
        EXPECT_TRUE(failed.sealed.empty());
        EXPECT_EQ(failed.consumed, 0u);
        EXPECT_FALSE(failed.retained_at_failure);
    }
    EXPECT_EQ(source.snapshot(), accepted);
    EXPECT_EQ(source.budget->used, source_used);
    EXPECT_EQ(source.budget->cumulative, source_spent);
    EXPECT_EQ(source.fetch(), source.comm->getRank() == 0 ? 17u : 0u);
    EXPECT_EQ(source.snapshot(), accepted);
    EXPECT_EQ(source.budget->used, source_used);
    const auto retry = fresh_borrowed_exchange(true);
    EXPECT_FALSE(retry.rejected);
}

TEST(CartesianCachedLineControls, BorrowCannotEscapeMalformedDataOrTerminalControlsAndConsumerFailureUnwinds)
{
    for (bool reply : {false, true})
        for (auto kind : {D::PooledNativeLineFault::Kind::MissingCompletion,
                 D::PooledNativeLineFault::Kind::DuplicateCompletion, D::PooledNativeLineFault::Kind::StaleContext,
                 D::PooledNativeLineFault::Kind::TruncatedCompletion})
            for (std::uint64_t page : {0u, 1u})
            {
                // Edit data has no second page in this fixture. Reply page one
                // is the authenticated empty terminal page, after all results.
                if (!reply && page)
                    continue;
                D::PooledNativeLineFault fault;
                fault.kind = kind;
                fault.reply = reply;
                fault.page = page;
                fault.rank = D::authority(line_key(), Tpetra::getDefaultComm()->getSize());
                if (!reply)
                    fault.rank = 0;
                const auto failed = fresh_borrowed_exchange(true, fault);
                EXPECT_TRUE(failed.rejected);
                EXPECT_TRUE(failed.attempted.empty());
                EXPECT_TRUE(failed.sealed.empty());
                EXPECT_EQ(failed.consumed, 0u);
            }
    const auto failed = fresh_borrowed_exchange(true, {}, true);
    EXPECT_TRUE(failed.rejected);
    EXPECT_TRUE(failed.sealed.empty());
    EXPECT_EQ(failed.consumed, 0u);
    const auto retry = fresh_borrowed_exchange(true);
    EXPECT_FALSE(retry.rejected);
}
} // namespace
