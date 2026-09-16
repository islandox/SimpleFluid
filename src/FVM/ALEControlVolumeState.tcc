/** @file ALEControlVolumeState.tcc @brief Compiled ALE validation. */
#pragma once

#include "FVM/ALEControlVolumeState.hh"

namespace SimpleFluid::FVM
{
template<class MeshType>
void ALEControlVolumeState::validate(const MeshType& mesh) const
{
    using local_ordinal_type = typename MeshType::local_ordinal_type;

    const auto communicator = mesh.owned_cell_map()->getComm();
    const auto& diagnostics = d_motion->diagnostics();
    const int local_identity_error =
        std::addressof(mesh) != d_mesh_view_identity || detail::ale_geometry_identity(mesh) != d_geometry_identity
            ? 1
            : 0;
    const int local_trial_error = !d_motion->has_active_trial() || !diagnostics.trial_active ? 1 : 0;
    const int local_size_error = d_old_cell_volumes.size() != mesh.num_local_cells() ||
                                         d_new_cell_volumes.size() != mesh.num_local_cells() ||
                                         d_face_mesh_fluxes.size() != mesh.num_faces()
                                     ? 1
                                     : 0;
    const int local_epoch_error = mesh_geometry_epoch(mesh) != d_new_geometry_epoch ||
                                          diagnostics.old_geometry_epoch != d_old_geometry_epoch ||
                                          diagnostics.new_geometry_epoch != d_new_geometry_epoch
                                      ? 1
                                      : 0;
    const int local_time_error =
        !std::isfinite(d_time_step) || d_time_step <= real_t{} || diagnostics.time_step != d_time_step ? 1 : 0;
    const int local_tolerance_error =
        !std::isfinite(d_gcl_absolute_tolerance) || d_gcl_absolute_tolerance < real_t{} ||
                !std::isfinite(d_gcl_relative_tolerance) || d_gcl_relative_tolerance < real_t{}
            ? 1
            : 0;

    std::array<int, 6> local_state{local_identity_error, local_trial_error, local_size_error, local_epoch_error,
        local_time_error, local_tolerance_error};
    std::array<int, 6> global_state{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_state.size()), local_state.data(),
        global_state.data());
    if (global_state[0] != 0)
    {
        throw std::invalid_argument("ALE control-volume state belongs to another concrete mesh geometry.");
    }
    if (global_state[1] != 0)
    {
        throw std::logic_error("ALE control-volume state requires its originating motion trial to remain active.");
    }
    if (global_state[2] != 0)
    {
        throw std::invalid_argument(
            "ALE control-volume state dimensions do not match mesh-local cell and face order.");
    }
    if (global_state[3] != 0)
    {
        throw std::invalid_argument(
            "ALE control-volume state does not represent the mesh's current trial geometry epoch.");
    }
    if (global_state[4] != 0)
    {
        throw std::invalid_argument("ALE control-volume state requires one finite positive trial time step.");
    }
    if (global_state[5] != 0)
    {
        throw std::invalid_argument("ALE control-volume state requires finite non-negative GCL tolerances.");
    }

    int local_non_finite = 0;
    for (size_t local = 0; local < d_old_cell_volumes.size(); ++local)
    {
        local_non_finite = local_non_finite || !std::isfinite(d_old_cell_volumes[local]) ||
                           d_old_cell_volumes[local] <= real_t{} || !std::isfinite(d_new_cell_volumes[local]) ||
                           d_new_cell_volumes[local] <= real_t{};
    }
    for (const auto flux : d_face_mesh_fluxes)
    {
        local_non_finite = local_non_finite || !std::isfinite(flux);
    }

    int local_gcl_failure = 0;
    real_t local_maximum_gcl_residual{};
    if (local_non_finite == 0)
    {
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            real_t mesh_flux_balance{};
            for (const auto face_lid : mesh.faces(cell_lid))
            {
                const auto mesh_flux = d_face_mesh_fluxes[static_cast<size_t>(face_lid)];
                mesh_flux_balance += mesh.owner_cell(face_lid) == cell_lid ? mesh_flux : -mesh_flux;
            }
            const auto volume_rate = (d_new_cell_volumes[owned] - d_old_cell_volumes[owned]) / d_time_step;
            const auto residual = volume_rate - mesh_flux_balance;
            const auto scale = std::max(std::abs(volume_rate), std::abs(mesh_flux_balance));
            const auto tolerance = d_gcl_absolute_tolerance + d_gcl_relative_tolerance * scale;
            local_non_finite = local_non_finite || !std::isfinite(volume_rate) ||
                               !std::isfinite(mesh_flux_balance) || !std::isfinite(residual);
            local_gcl_failure = local_gcl_failure || std::abs(residual) > tolerance;
            local_maximum_gcl_residual = std::max(local_maximum_gcl_residual, std::abs(residual));
        }
    }

    std::array<int, 2> local_validation{local_non_finite, local_gcl_failure};
    std::array<int, 2> global_validation{};
    real_t global_maximum_gcl_residual{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_validation.size()),
        local_validation.data(), global_validation.data());
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1, &local_maximum_gcl_residual, &global_maximum_gcl_residual);
    if (global_validation[0] != 0)
    {
        throw std::invalid_argument("ALE control-volume state contains invalid volume, mesh-flux, or GCL data.");
    }
    if (global_validation[1] != 0)
    {
        throw std::invalid_argument(
            "ALE control-volume state violates the cellwise geometric conservation law; maximum residual is " +
            std::to_string(global_maximum_gcl_residual) + " m^3/s.");
    }
}
} // namespace SimpleFluid::FVM
