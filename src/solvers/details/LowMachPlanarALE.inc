/** Compiled implementation fragment; include inside SimpleFluid::Reacting after flow Implementation. */
std::shared_ptr<LowMachMixtureSolver::ALEGeometryCertificate> LowMachMixtureSolver::capture_accepted_geometry(
    const PlanarMotion& motion) const
{
    const auto& mesh = *d->mesh;
    collective_detail::collective_local_validation(mesh, "Accepted planar geometry authority",
        [&]
        {
            check(d->initialized && d->epoch == mesh_geometry_epoch(mesh) && !motion.has_active_trial() &&
                      motion.mesh_ptr().get() == d->mesh.get(),
                "Planar authority must be live, idle and bound to this exact current accepted mesh.");
        });
    const auto snapshot = motion.snapshot(); // Collective exact native-axis/motion-lease proof, no EOS.
    std::shared_ptr<ALEGeometryCertificate> result;
    collective_detail::collective_local_validation(mesh, "Owned planar geometry certificate",
        [&]
        {
            result = std::make_shared<ALEGeometryCertificate>();
            result->owner = this;
            result->motion_identity = std::addressof(motion);
            result->geometry_identity = mesh.geometry_identity();
            result->source = snapshot;
            result->logical_geometry = d->logical_geometry;
            result->old_volumes.resize(mesh.num_local_cells());
            result->new_volumes.resize(mesh.num_local_cells());
            result->swept_flux.assign(mesh.num_faces(), 0.);
            for (size_t c = 0; c < mesh.num_local_cells(); ++c)
            {
                const LO cell = static_cast<LO>(c);
                result->old_volumes[c] = result->new_volumes[c] = mesh.cell_volume(cell);
                result->cell_ids.push_back(mesh.cell_global_id(cell));
                result->cell_geometry_ids.push_back(mesh.cell_geometry_global_id(cell));
            }
            for (size_t f = 0; f < mesh.num_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                result->face_ids.push_back(mesh.face_global_id(face));
                result->face_geometry_ids.push_back(mesh.face_geometry_global_id(face));
            }
        });
    return result;
}

void LowMachMixtureSolver::restore_accepted_after_motion(
    PlanarMotion& motion, const PlanarMotion::StateSnapshot& snapshot, std::span<const double> absolute_flux)
{
    if (motion.has_active_trial())
        motion.rollback_trial();
    else
        motion.restore(snapshot);
    d->refresh_numeric_geometry();
    d->publish_fields(d->accepted, true);
    for (size_t f = 0; f < absolute_flux.size(); ++f)
        d->flux.set_owned_value(static_cast<LO>(f), absolute_flux[f]);
    d->flux.sync_ghosts();
    // No logical accepted geometry, accepted state or trial_nonce change.
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_ale(double dt, PlanarMotion& motion, double target_surface,
    const std::vector<CellSources>& sources, const ControllerLease* controller)
{
    return preview_planar_impl(dt, motion, target_surface, sources, nullptr, controller);
}
LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_ale_with_sources(double dt, PlanarMotion& motion,
    double target_surface, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    collective_detail::collective_local_validation(*d->mesh, "Planar source provider ownership",
        [&] { check(bool(provider), "Planar source preview requires an immutable provider."); });
    return preview_planar_impl(dt, motion, target_surface, {}, std::move(provider), controller);
}
LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_planar_impl(double dt, PlanarMotion& motion,
    double target_surface, const std::vector<CellSources>& sources, std::shared_ptr<const SourceStageProvider> provider,
    const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    collective_detail::collective_local_validation(mesh, "Prescribed planar preview input",
        [&]
        {
            check(d->initialized && d->epoch == mesh_geometry_epoch(mesh) && motion.mesh_ptr().get() == d->mesh.get() &&
                      !motion.has_active_trial(),
                "Planar preview requires initialized accepted geometry and its explicit idle motion authority.");
            check(std::isfinite(dt) && dt > 0 && std::isfinite(target_surface),
                "Planar preview needs a finite target and positive finite timestep.");
            mesh.visit(
                [&](const auto& native)
                {
                    using Native = std::remove_cvref_t<decltype(native)>;
                    check(std::is_same_v<Native, Mesh::Cartesian> || std::is_same_v<Native, Mesh::Cylindrical> ||
                              std::is_same_v<Native, Mesh::SemiStructured>,
                        "Low-Mach planar ALE requires native orthogonal mapped geometry.");
                    if constexpr (requires { native.indexer().periodic_dimensions[0]; })
                        for (size_t axis = 0; axis < 3; ++axis)
                            check(!native.indexer().periodic_dimensions[axis],
                                "Initial planar ALE rejects periodic topology.");
                    if constexpr (requires { native.indexer().axial_periodic; })
                        check(!native.indexer().axial_periodic, "Initial planar ALE rejects axial periodic topology.");
                });
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                if (!mesh.is_geometry_exterior_face(face))
                    continue;
                for (const auto& slip : d->accepted.cells[mesh.owner_cell(face)].drift_velocity)
                    check(slip.dot(mesh.face_area_vector(face)) == 0.,
                        "Impermeable wall requires zero normal phase slip.");
            }
        });
    collective_detail::require_uniform_value(mesh, dt, "Prescribed planar timestep");
    collective_detail::require_uniform_value(mesh, target_surface, "Prescribed planar target");
    auto certificate = capture_accepted_geometry(motion);
    std::vector<double> accepted_flux;
    collective_detail::collective_local_validation(mesh, "Planar accepted absolute flux receipt",
        [&]
        {
            accepted_flux.resize(mesh.num_owned_faces());
            for (size_t f = 0; f < accepted_flux.size(); ++f)
                accepted_flux[f] = d->flux.local_value(static_cast<LO>(f));
            certificate->target_surface = target_surface;
            certificate->time_step = dt;
        });
    try
    {
        motion.begin_trial(target_surface, dt);
        const auto ale = FVM::make_validated_planar_ale_control_volume_state(mesh, motion);
        collective_detail::collective_local_validation(mesh, "Planar accepted-old storage match",
            [&]
            {
                check(certificate->old_volumes.size() == ale.old_cell_volumes().size() &&
                          std::equal(certificate->old_volumes.begin(), certificate->old_volumes.end(),
                              ale.old_cell_volumes().begin()),
                    "Planar trial did not start from actual accepted-old CV storage.");
                certificate->new_volumes.assign(ale.new_cell_volumes().begin(), ale.new_cell_volumes().end());
                certificate->swept_flux.assign(ale.face_mesh_fluxes().begin(), ale.face_mesh_fluxes().end());
            });
        d->refresh_numeric_geometry();
        auto result = preview_impl(dt, sources, std::move(provider), controller, &ale);
        result.ale_geometry = certificate;
        restore_accepted_after_motion(motion, certificate->source, accepted_flux);
        return result;
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        restore_accepted_after_motion(motion, certificate->source, accepted_flux);
        std::rethrow_exception(failure);
    }
}

void LowMachMixtureSolver::commit_ale(const Trial& trial, PlanarMotion& motion, const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    State candidate;
    Diagnostics statistics;
    std::vector<double> candidate_flux, accepted_flux;
    std::shared_ptr<const ALEGeometryCertificate> certificate;
    collective_detail::collective_local_validation(mesh, "Explicit planar trial commit preflight",
        [&]
        {
            certificate = trial.ale_geometry;
            check(certificate && certificate->owner == this && certificate->motion_identity == std::addressof(motion) &&
                      certificate->geometry_identity == mesh.geometry_identity() && trial.owner == this &&
                      trial.nonce == d->trial_nonce && trial.base_revision == d->accepted.revision &&
                      trial.logical_geometry == d->logical_geometry &&
                      certificate->logical_geometry == d->logical_geometry &&
                      d->logical_geometry != std::numeric_limits<std::uint64_t>::max() &&
                      d->epoch == mesh_geometry_epoch(mesh) && motion.mesh_ptr().get() == d->mesh.get() &&
                      !motion.has_active_trial() && trial.face_flux.size() == mesh.num_owned_faces(),
                "Moving trial requires its exact explicit live motion and unconsumed logical accepted authority.");
            candidate = trial.candidate;
            statistics = trial.statistics;
            candidate_flux = trial.face_flux;
            accepted_flux.resize(mesh.num_owned_faces());
            for (size_t f = 0; f < accepted_flux.size(); ++f)
                accepted_flux[f] = d->flux.local_value(static_cast<LO>(f));
        });
    const auto rollback_geometry = motion.snapshot();
    try
    {
        motion.begin_trial(certificate->target_surface, certificate->time_step);
        const auto ale = FVM::make_validated_planar_ale_control_volume_state(mesh, motion);
        collective_detail::collective_local_validation(mesh, "Exact retained planar geometry restaging",
            [&]
            {
                const auto same = [](const auto& saved, auto actual)
                { return saved.size() == actual.size() && std::equal(saved.begin(), saved.end(), actual.begin()); };
                check(same(certificate->old_volumes, ale.old_cell_volumes()) &&
                          same(certificate->new_volumes, ale.new_cell_volumes()) &&
                          same(certificate->swept_flux, ale.face_mesh_fluxes()) &&
                          certificate->cell_ids.size() == mesh.num_local_cells() &&
                          certificate->face_ids.size() == mesh.num_faces(),
                    "Re-staged CV/swept receipt differs from the retained immutable geometry certificate.");
                for (size_t c = 0; c < mesh.num_local_cells(); ++c)
                {
                    const LO cell = static_cast<LO>(c);
                    check(certificate->cell_ids[c] == mesh.cell_global_id(cell) &&
                              certificate->cell_geometry_ids[c] == mesh.cell_geometry_global_id(cell),
                        "Planar cell maps changed.");
                }
                for (size_t f = 0; f < mesh.num_faces(); ++f)
                {
                    const LO face = static_cast<LO>(f);
                    check(certificate->face_ids[f] == mesh.face_global_id(face) &&
                              certificate->face_geometry_ids[f] == mesh.face_geometry_global_id(face),
                        "Planar face maps changed.");
                }
            });
        d->refresh_numeric_geometry();
        d->publish_fields(candidate, true);
        for (size_t f = 0; f < candidate_flux.size(); ++f)
            d->flux.set_owned_value(static_cast<LO>(f), candidate_flux[f]);
        d->flux.sync_ghosts();
        motion.accept_trial();
        d->accepted = std::move(candidate);
        d->diagnostics = std::move(statistics);
        ++d->logical_geometry;
        ++d->trial_nonce;
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        restore_accepted_after_motion(motion, rollback_geometry, accepted_flux);
        std::rethrow_exception(failure);
    }
}

LowMachMixtureSolver::Checkpoint LowMachMixtureSolver::checkpoint(const PlanarMotion& motion) const
{
    auto geometry = capture_accepted_geometry(motion);
    auto result = checkpoint();
    result.ale_geometry = std::move(geometry);
    return result;
}
void LowMachMixtureSolver::restore(const Checkpoint& saved, PlanarMotion& motion, const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    State candidate;
    Diagnostics statistics;
    std::vector<double> candidate_flux, accepted_flux;
    collective_detail::collective_local_validation(mesh, "Planar checkpoint authority",
        [&]
        {
            check(saved.owner == this && saved.ale_geometry && saved.ale_geometry->owner == this &&
                      saved.ale_geometry->motion_identity == std::addressof(motion) &&
                      saved.ale_geometry->geometry_identity == mesh.geometry_identity() &&
                      motion.mesh_ptr().get() == d->mesh.get() && !motion.has_active_trial() &&
                      d->epoch == mesh_geometry_epoch(mesh) &&
                      d->logical_geometry != std::numeric_limits<std::uint64_t>::max() &&
                      saved.state.cells.size() == mesh.num_owned_cells() &&
                      saved.face_flux.size() == mesh.num_owned_faces(),
                "Planar checkpoint requires its exact explicit live motion and unchanged native topology.");
            candidate = saved.state;
            statistics = saved.diagnostics;
            candidate_flux = saved.face_flux;
            accepted_flux.resize(mesh.num_owned_faces());
            for (size_t f = 0; f < accepted_flux.size(); ++f)
                accepted_flux[f] = d->flux.local_value(static_cast<LO>(f));
        });
    const auto rollback_geometry = motion.snapshot();
    try
    {
        motion.restore(saved.ale_geometry->source);
        d->refresh_numeric_geometry();
        d->publish_fields(candidate, true);
        for (size_t f = 0; f < candidate_flux.size(); ++f)
            d->flux.set_owned_value(static_cast<LO>(f), candidate_flux[f]);
        d->flux.sync_ghosts();
        d->accepted = std::move(candidate);
        d->diagnostics = std::move(statistics);
        ++d->logical_geometry;
        ++d->trial_nonce;
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        restore_accepted_after_motion(motion, rollback_geometry, accepted_flux);
        std::rethrow_exception(failure);
    }
}
