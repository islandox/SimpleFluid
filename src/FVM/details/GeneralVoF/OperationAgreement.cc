/** @file OperationAgreement.cc
 * @brief Collective operation identity allocation and algorithm agreement.
 */
#include "FVM/details/GeneralVoF/PhaseCleanup.hh"
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
std::uint64_t transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::uint64_t id = 0;
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                static std::uint64_t next = 0;
                if (next == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("General VoF transaction identity exhausted.");
                id = ++next;
            }
        });
    Teuchos::broadcast(*comm, 0, 1, &id);
    return id;
}
void agree_transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm, std::uint64_t id)
{
    std::uint64_t low = 0, high = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &id, &low);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &id, &high);
    wire::collective(*comm,
        [&]
        {
            if (!id || low != high)
                throw std::invalid_argument("General VoF mixed collective trial controls.");
        });
}
void agree_options(const GeneralVoFOptions& options, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{options.byte_budget};
            if (std::uint64_t(options.arithmetic_mode) > std::uint64_t(ArithmeticMode::Exact))
                throw std::invalid_argument("Unknown General VoF arithmetic mode.");
            if (std::uint64_t(options.transport_backend) > std::uint64_t(GeneralVoFTransportBackend::GeometricFace) ||
                (options.transport_backend != GeneralVoFTransportBackend::SweptPartition &&
                    options.arithmetic_mode != ArithmeticMode::Ordinary))
                throw std::invalid_argument("Face transport currently supports Ordinary arithmetic only.");
            if (std::uint64_t(options.reconstruction.model_policy) >
                    std::uint64_t(PLIC::PlicModelPolicy::ConservativeApproximation) ||
                (options.reconstruction.model_policy == PLIC::PlicModelPolicy::ConservativeApproximation &&
                    options.arithmetic_mode != ArithmeticMode::Ordinary))
                throw std::invalid_argument("General VoF model policy requires Ordinary arithmetic.");
            const auto ratio = options.reconstruction.minimum_optimized_phase_length_ratio;
            if (!std::isfinite(ratio) || ratio < 0 || ratio > 1)
                throw std::invalid_argument("General VoF optimized phase length ratio must be finite in [0,1].");
            if (!options.maximum_face_geometry_operations)
                throw std::invalid_argument("General VoF face geometry operation limit must be positive.");
            validate_phase_cleanup_options(options);
            budget.packet(27 * 8);
            wire::Writer w;
            w.word(8); // Includes the Ordinary plane-model admission policy.
            for (auto value : {std::uint64_t(options.maximum_stencil_cells), std::uint64_t(options.maximum_halo_cells),
                     std::uint64_t(options.support_rings), std::uint64_t(options.arithmetic_mode),
                     std::uint64_t(options.transport_backend), std::uint64_t(options.reconstruction.measure_only),
                     std::uint64_t(options.reconstruction.model_policy),
                     std::uint64_t(options.reconstruction.maximum_offset_iterations),
                     std::uint64_t(options.reconstruction.maximum_fit_iterations),
                     std::uint64_t(options.reconstruction.maximum_stencil_cells),
                     std::uint64_t(options.reconstruction.maximum_clip_evaluations),
                     std::uint64_t(options.sweep.maximum_simplices), std::uint64_t(options.sweep.maximum_atoms),
                     std::uint64_t(options.sweep.maximum_pair_tests),
                     std::uint64_t(options.maximum_face_geometry_operations),
                     std::uint64_t(options.record_balance_work)})
                w.word(value);
            real(w, options.phase_cleanup.relative_cutoff);
            for (double value : options.phase_cleanup.maximum_absolute_volume)
                real(w, value);
            real(w, options.reconstruction.volume_tolerance);
            real(w, options.reconstruction.fit_tolerance);
            real(w, options.reconstruction.maximum_condition);
            real(w, options.reconstruction.orientation_tolerance);
            real(w, options.reconstruction.maximum_model_residual);
            real(w, options.reconstruction.minimum_phase_support_fraction);
            real(w, options.reconstruction.minimum_optimized_phase_length_ratio);
            packets.push_back({0, std::move(w.data)});
        });
    const auto received = exchange_owner_packets(packets, comm, options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                if (received.packets.size() != std::size_t(comm->getSize()))
                    throw std::invalid_argument("General VoF missing algorithm controls.");
                for (const auto& packet : received.packets)
                    if (packet.bytes != received.packets.front().bytes)
                        throw std::invalid_argument("General VoF algorithm controls disagree across ranks.");
            }
        });
}
} // namespace SimpleFluid::VoF::general_vof_detail
