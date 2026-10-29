/** @file GeneralVoFBalance.hh
 * @brief Measured partition proposals and detached initial phase experiments.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
namespace SimpleFluid::VoF
{
enum class GeneralVoFReceiverEstimator : std::uint64_t
{
    Mean,
    Maximum
};
struct GeneralVoFBalanceOptions
{
    Meshes::AMR::AdaptiveBalanceOptions objective = []
    {
        Meshes::AMR::AdaptiveBalanceOptions result;
        result.work_unit = Meshes::AMR::AdaptiveWorkUnit::Seconds;
        return result;
    }();
    /** Existing requester ranks retain their actual validation time. New remote
     * requester ranks use this per-donor estimator; no byte-to-time calibration.
     * Maximum is conservative among observed consumers, not a bound on future execution. */
    GeneralVoFReceiverEstimator receiver_estimator = GeneralVoFReceiverEstimator::Maximum;
};
enum class GeneralVoFBalanceRejection
{
    None,
    NoCoreImprovement,
    IncompleteReceiverModel,
    ObservationClassificationChanged,
    NoFullComputeImprovement,
    NativeWireQualification
};
enum class GeneralVoFBalanceBinding : std::uint64_t
{
    /** Measurements describe this exact phase publication. */
    ObservedSource,
    /** Measurements describe the authenticated predecessor of this accepted
     * publication; their costs and dependencies are a one-step-lag forecast. */
    AcceptedSuccessor
};
struct GeneralVoFBalanceLoad
{
    /** False when the receiver model is incomplete; compute fields are unavailable. */
    bool compute_available = false;
    double total_core_seconds = 0, total_receiver_seconds = 0;
    double maximum_compute_seconds = 0, mean_compute_seconds = 0;
    /** Complete request/reply payload envelopes, excluding exchange peer framing and MPI. */
    std::uint64_t request_payload_bytes = 0, reply_payload_bytes = 0, remote_donor_consumers = 0;
    std::uint64_t canonical_cut_faces = 0;
};
struct GeneralVoFBalanceProposal
{
    GeneralVoFBalanceBinding binding = GeneralVoFBalanceBinding::ObservedSource;
    /** Forecast target and historical measured source remain distinct. */
    std::shared_ptr<const void> current_phase_identity, observed_source_identity;
    std::uint64_t current_phase_control = 0, observed_source_control = 0;
    std::shared_ptr<const GeneralVoFWorkSample> observed_work;
    /** Qualified only for the native hierarchy domain. wire_bytes_exact does
     * not include phase/state migration or establish a complete VoF wire cost. */
    Meshes::AMR::AdaptiveBalancePlan native_plan;
    GeneralVoFBalanceLoad measured_before, forecast_proposed;
    GeneralVoFBalanceRejection rejection = GeneralVoFBalanceRejection::None;
    bool accepted = false;
    /** Exact current native hierarchy domain only: payload + 8/record + 16/peer. */
    std::uint64_t local_native_application_bytes = 0, global_native_application_bytes = 0;
};
struct GeneralVoFBalanceResult : GeneralVoFBalanceProposal
{
    std::shared_ptr<const GeneralVoF> candidate;
    /** Global native application bytes: payload + 8/record + 16/peer. */
    std::uint64_t native_application_bytes = 0;
    /** Maximum rank elapsed times, measured independently of the work forecast. */
    double native_migration_seconds = 0, phase_migration_seconds = 0, total_migration_seconds = 0;
};
/** Collective proposal only; performs no hierarchy or phase migration.
 * AcceptedSuccessor authenticates the retained predecessor observation against
 * the exact current accepted phase. It does not rewrite the historical sample.
 * Native bytes are exact for the current hierarchy; recurring work/dependency
 * costs remain historical measurements and an explicitly bound forecast.
 * AcceptedSuccessor rejects when a currently inferred two-phase cell lacks an
 * observed mixed model. This public declaration/inventory check is conservative
 * for inferred cells with retained pure geometry; no new costs are synthesized.
 * Only compute seconds are scored; communication/migration coefficients must be zero.
 * A solver must revalidate its accepted State and migrate all fields atomically. */
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT GeneralVoFBalanceProposal plan_partition_proposal(
    std::shared_ptr<const GeneralVoF> current, const GeneralVoFPreparedTransport& observed, GeneralVoFBalanceBinding,
    GeneralVoFBalanceOptions = {});
/** Collective initial-only partition planner. Requires an authenticated complete
 * Ordinary/SweptPartition/InterfaceLocalSwept observation from this exact source.
 * Every rejection retains the exact source pointer. Accepted candidates use the
 * existing phase-only ownership remap and retain source options.
 * The source is immutable; this does not repartition an attached solver.
 * Only compute seconds are scored: communication/migration coefficients must be zero. */
[[nodiscard]] SIMPLEFLUID_FVM_EXPORT GeneralVoFBalanceResult plan_initial_partition(
    std::shared_ptr<const GeneralVoF> source, const GeneralVoFPreparedTransport& observed,
    GeneralVoFBalanceOptions = {});
} // namespace SimpleFluid::VoF
