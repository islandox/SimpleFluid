/** @file DistributedAdaptiveBalance.hh
 * @brief Measured owner-local work recording and explicitly evaluated migration proposals.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <map>
namespace SimpleFluid::Meshes::AMR
{
enum class AdaptiveWorkUnit : std::uint64_t
{
    Seconds,
    Operations
};
/** All three recurring observations use the same declared unit. Migration bytes
 * are measured standalone record sizes. Their sum estimates batched transfer
 * cost; shared dictionaries and framing require a packed-stream recheck. */
struct AdaptiveMeasuredWork
{
    double reconstruction = 0, interface = 0, population = 0;
    std::uint64_t migration_bytes = 0;
    AdaptiveWorkUnit unit = AdaptiveWorkUnit::Operations;
    [[nodiscard]] double total() const;
};
/** Local recorder; callers synchronize concurrent observations. A topology or
 * state revision requires a new recorder or an explicit clear(). */
class AdaptiveWorkRecorder
{
public:
    explicit AdaptiveWorkRecorder(AdaptiveWorkUnit unit) : unit_(unit) {}
    void observe(
        const AdaptiveCellKey&, std::uint64_t revision, double reconstruction, double interface, double population);
    void record_payload(const AdaptiveCellKey&, std::uint64_t revision, std::uint64_t standalone_encoded_bytes);
    [[nodiscard]] AdaptiveMeasuredWork snapshot(const AdaptiveCellKey&, std::uint64_t revision) const;
    void clear() noexcept { records_.clear(); }

private:
    struct Record
    {
        std::uint64_t revision = 0, samples = 0;
        AdaptiveMeasuredWork work;
    };
    AdaptiveWorkUnit unit_;
    std::map<AdaptiveCellKey, Record> records_;
};
struct AdaptiveBalanceCell
{
    AdaptiveCellKey key;
    std::uint64_t revision = 0;
    AdaptiveMeasuredWork work;
};
/** Each canonical interior edge is supplied exactly once, at its first owner.
 * communication_bytes is measured transfer work per recurring evaluation. */
struct AdaptiveBalanceEdge
{
    AdaptiveBytes identity;
    AdaptiveCellKey first, second;
    int first_owner = -1, second_owner = -1;
    std::uint64_t communication_bytes = 0;
};
struct AdaptiveBalanceSource
{
    std::shared_ptr<const void> identity;
    std::uint64_t topology = 0, partition = 0;
};
struct AdaptiveBalanceOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024;
    AdaptiveWorkUnit work_unit = AdaptiveWorkUnit::Operations;
    double compute_weight = 1, communication_weight = 0, migration_weight = 0, minimum_relative_improvement = 1e-6;
    bool operator==(const AdaptiveBalanceOptions&) const = default;
};
struct AdaptiveBalanceMetrics
{
    double total_compute = 0, maximum_compute = 0, mean_compute = 0, imbalance = 0, objective = 0;
    std::uint64_t communication_cut_bytes = 0, moved_bytes = 0, moved_cells = 0;
    bool wire_bytes_exact = false;
};
struct AdaptiveBalancePlan
{
    AdaptiveBalanceSource source;
    AdaptiveBalanceOptions objective_options;
    std::vector<AdaptiveCellKey> keys;
    std::vector<std::uint64_t> revisions;
    std::vector<int> localdestinations;
    bool accepted = false;
    AdaptiveBalanceMetrics before, proposed, final;
};
/** Evaluate a measured-weight heuristic or an explicit owner-local proposal.
 * Only compact rank loads are replicated; cell/edge assignments use sharded
 * exact-key directories. The objective initially uses measured standalone payload-size estimates.
 * Rejected proposals retain the existing destinations. This is a heuristic,
 * with no optimality claim. Cells must be in strictly increasing key order. */
[[nodiscard]] AdaptiveBalancePlan balance_owned_cells(std::span<const AdaptiveBalanceCell>,
    std::span<const AdaptiveBalanceEdge>, AdaptiveBalanceSource, Teuchos::RCP<const Teuchos::Comm<int>>,
    AdaptiveBalanceOptions = {}, std::optional<std::span<const int>> external_proposal = std::nullopt);
/** Recheck an accepted candidate using actual packed streams including unique
 * dictionaries, state/history and framing, excluding self-target packets.
 * This can reject a proposal but cannot revive an already rejected proposal.
 * Call before publication; source identity remains pinned in the returned plan. */
[[nodiscard]] AdaptiveBalancePlan qualify_balance_wire_bytes(const AdaptiveBalancePlan&,
    std::uint64_t local_packed_migration_bytes, Teuchos::RCP<const Teuchos::Comm<int>>, AdaptiveBalanceOptions = {});
} // namespace SimpleFluid::Meshes::AMR
