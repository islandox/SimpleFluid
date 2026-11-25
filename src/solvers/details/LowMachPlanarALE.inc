/** Compiled implementation fragment; include inside SimpleFluid::Reacting after flow Implementation. */
struct LowMachMixtureSolver::PlanarMotionFrame
{
    std::uint64_t epoch;
    std::vector<double> old_volumes, new_volumes, swept_rates;
    MeshMotionDiagnostics diagnostics;
};

LowMachMixtureSolver::PlanarMotionFrame LowMachMixtureSolver::capture_planar_motion_frame(const PlanarMotion& motion) const
{
    PlanarMotionFrame frame{};
    frame.epoch = mesh_geometry_epoch(*d->mesh);
    if (uses_qbmm_linear_transport(d->options.qbmm_model))
        collective_detail::collective_local_validation(*d->mesh, "Accepted planar motion history",
            [&]
            {
                frame.old_volumes = motion.d_old_cell_volumes;
                frame.new_volumes = motion.d_new_cell_volumes;
                frame.swept_rates = motion.d_face_mesh_fluxes;
                frame.diagnostics = motion.d_diagnostics;
            });
    return frame;
}

std::shared_ptr<LowMachMixtureSolver::ALEGeometryCertificate> LowMachMixtureSolver::capture_accepted_geometry(
    const PlanarMotion& motion) const
{
    const auto& mesh = *d->mesh;
    collective_detail::collective_local_validation(mesh, "Accepted planar geometry authority",
        [&]
        {
            qbmm_detail::validate_capability(d->options, mesh, true, acceleration_vector(d->active_body_acceleration));
            d->validate_planar_controller_authority(motion);
            check(d->initialized && d->epoch == mesh_geometry_epoch(mesh) && !motion.has_active_trial() &&
                      motion.mesh_ptr().get() == d->mesh.get() &&
                      (!d->accepted_planar_motion || d->accepted_planar_motion == motion.d_authority),
                "Planar authority must be live, idle and bound to this exact current accepted mesh.");
        });
    const auto snapshot = motion.snapshot(); // Collective exact native-axis/motion-lease proof, no EOS.
    std::shared_ptr<ALEGeometryCertificate> result;
    collective_detail::collective_local_validation(mesh, "Owned planar geometry certificate",
        [&]
        {
            result = std::make_shared<ALEGeometryCertificate>();
            result->owner = d->planar_owner;
            result->motion_identity = motion.d_authority;
            result->geometry_identity = mesh.geometry_identity();
            result->source = snapshot;
            result->logical_geometry = d->logical_geometry;
            result->old_geometry_epoch = result->new_geometry_epoch = d->epoch;
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
    PlanarMotion& motion, std::span<const double> absolute_flux, PlanarMotionFrame& accepted_frame)
{
    rollback_planar_geometry(motion, accepted_frame);
    if (d->initialized)
        d->publish_fields(d->accepted);
    for (size_t f = 0; f < absolute_flux.size(); ++f)
        d->flux.set_owned_value(static_cast<LO>(f), absolute_flux[f]);
    d->flux.sync_ghosts();
    // No logical accepted geometry, accepted state or trial_nonce change.
}

LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_ale(double dt, PlanarMotion& motion, double target_surface,
    const std::vector<CellSources>& sources, const ControllerLease* controller)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::PreviewPlanar);
    return preview_planar_impl(dt, motion, target_surface, sources, nullptr, controller);
}
LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_ale_with_sources(double dt, PlanarMotion& motion,
    double target_surface, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::PreviewPlanarWithSources);
    collective_detail::collective_local_validation(*d->mesh, "Planar source provider ownership",
        [&] { check(bool(provider), "Planar source preview requires an immutable provider."); });
    return preview_planar_impl(dt, motion, target_surface, {}, std::move(provider), controller);
}
LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_ale_with_qbmm_sources(double dt, PlanarMotion& motion,
    double target_surface, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::PreviewPlanarWithQbmmSources);
    collective_detail::collective_local_validation(*d->mesh, "Planar source provider ownership",
        [&] { check(bool(provider), "Planar source preview requires an immutable provider."); });
    return preview_planar_impl(dt, motion, target_surface, {}, std::move(provider), controller, true);
}
LowMachMixtureSolver::Trial LowMachMixtureSolver::preview_planar_impl(double dt, PlanarMotion& motion,
    double target_surface, const std::vector<CellSources>& sources, std::shared_ptr<const SourceStageProvider> provider,
    const ControllerLease* controller, bool coherent_qbmm)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    collective_detail::collective_local_validation(mesh, "Prescribed planar preview input",
        [&]
        {
            check(d->initialized && d->epoch == mesh_geometry_epoch(mesh) && motion.mesh_ptr().get() == d->mesh.get() &&
                      !motion.has_active_trial(),
                "Planar preview requires initialized accepted geometry and its explicit idle motion authority.");
            const bool has_qbmm = !d->options.qbmm_families.empty();
            qbmm_detail::validate_capability(d->options, mesh, true,
                d->active_body_acceleration ? acceleration_vector(d->active_body_acceleration)
                                            : (provider ? provider->body_acceleration : std::nullopt));
            check(!has_qbmm || !provider || coherent_qbmm,
                "Planar QBMM source providers require the coherent reacting controller.");
            check(!coherent_qbmm || (has_qbmm && controller && controller->reacting_source_authority && provider),
                "Planar QBMM source preview requires the reacting controller and an immutable source provider.");
            check(std::isfinite(dt) && dt > 0 && std::isfinite(target_surface),
                "Planar preview needs a finite target and positive finite timestep.");
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
    const bool linear = uses_qbmm_linear_transport(d->options.qbmm_model);
    std::vector<qbmm_linear_detail::PlanarFaceGeometry> old_faces;
    collective_detail::collective_local_validation(mesh, "Planar accepted absolute flux receipt",
        [&]
        {
            accepted_flux.resize(mesh.num_owned_faces());
            for (size_t f = 0; f < accepted_flux.size(); ++f)
                accepted_flux[f] = d->flux.local_value(static_cast<LO>(f));
            if (linear)
                old_faces = qbmm_linear_detail::capture_planar_face_geometry(mesh);
            certificate->target_surface = target_surface;
            certificate->time_step = dt;
        });
    auto accepted_frame = capture_planar_motion_frame(motion);
    Trial result;
    try
    {
        motion.begin_trial_impl(target_surface, dt, linear);
        const auto ale = FVM::make_validated_planar_ale_control_volume_state(mesh, motion);
        collective_detail::collective_local_validation(mesh, "Planar accepted-old storage match",
            [&]
            {
                check(certificate->old_volumes.size() == ale.old_cell_volumes().size() &&
                          certificate->old_geometry_epoch == ale.old_geometry_epoch() &&
                          std::equal(certificate->old_volumes.begin(), certificate->old_volumes.end(),
                              ale.old_cell_volumes().begin()),
                    "Planar trial did not start from actual accepted-old CV storage.");
                certificate->new_volumes.assign(ale.new_cell_volumes().begin(), ale.new_cell_volumes().end());
                certificate->swept_flux.assign(ale.face_mesh_fluxes().begin(), ale.face_mesh_fluxes().end());
                certificate->new_geometry_epoch = ale.new_geometry_epoch();
            });
        d->refresh_numeric_geometry();
        const qbmm_linear_detail::PlanarTransportGeometry geometry{ale, old_faces};
        result = preview_impl(
            dt, sources, std::move(provider), controller, &ale, coherent_qbmm, linear ? &geometry : nullptr);
        result.ale_geometry = certificate;
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        restore_accepted_after_motion(motion, accepted_flux, accepted_frame);
        std::rethrow_exception(failure);
    }
    restore_accepted_after_motion(motion, accepted_flux, accepted_frame);
    return result;
}

void LowMachMixtureSolver::commit_ale(const Trial& trial, PlanarMotion& motion, const ControllerLease* controller)
{
    commit_planar_impl(trial, motion, controller, nullptr);
}

void LowMachMixtureSolver::commit_planar_impl(const Trial& trial, PlanarMotion& motion,
    const ControllerLease* controller, const ControllerRestartDiagnostics* controller_diagnostics)
{
    require_uniform_advancement(*d->mesh, AdvancementOperation::CommitPlanar);
    require_controller(controller);
    const auto& mesh = *d->mesh;
    const bool linear = uses_qbmm_linear_transport(d->options.qbmm_model);
    State candidate;
    Diagnostics statistics;
    std::vector<double> candidate_flux, accepted_flux;
    std::shared_ptr<const ALEGeometryCertificate> certificate;
    std::shared_ptr<const QbmmPublication> publication;
    collective_detail::collective_local_validation(mesh, "Explicit planar trial commit preflight",
        [&]
        {
            d->validate_planar_controller_authority(motion);
            certificate = trial.ale_geometry;
            check(certificate && certificate->owner == d->planar_owner &&
                      certificate->motion_identity == motion.d_authority &&
                      certificate->geometry_identity == mesh.geometry_identity() && trial.owner == this &&
                      trial.nonce == d->trial_nonce && trial.base_revision == d->accepted.revision &&
                      trial.logical_geometry == d->logical_geometry &&
                      certificate->logical_geometry == d->logical_geometry &&
                      d->logical_geometry != std::numeric_limits<std::uint64_t>::max() &&
                      d->epoch == mesh_geometry_epoch(mesh) && motion.mesh_ptr().get() == d->mesh.get() &&
                      (!d->accepted_planar_motion || d->accepted_planar_motion == motion.d_authority) &&
                      !motion.has_active_trial() && trial.preview_id != 0 &&
                      trial.face_flux.size() == mesh.num_owned_faces(),
                "Moving trial requires its exact explicit live motion and unconsumed logical accepted authority.");
            qbmm_detail::validate_state(d->options, trial.candidate);
            candidate = trial.candidate;
            statistics = trial.statistics;
            candidate_flux = trial.face_flux;
            publication = prepare_qbmm_publication(trial, controller_diagnostics);
            accepted_flux.resize(mesh.num_owned_faces());
            for (size_t f = 0; f < accepted_flux.size(); ++f)
                accepted_flux[f] = d->flux.local_value(static_cast<LO>(f));
        });
    collective_detail::require_uniform_value(mesh, trial.preview_id, "Planar collective trial selection");
    auto accepted_frame = capture_planar_motion_frame(motion);
    try
    {
        motion.begin_trial_impl(certificate->target_surface, certificate->time_step, linear);
        const auto ale = FVM::make_validated_planar_ale_control_volume_state(mesh, motion);
        collective_detail::collective_local_validation(mesh, "Exact retained planar geometry restaging",
            [&]
            {
                const auto same = [](const auto& saved, auto actual)
                { return saved.size() == actual.size() && std::equal(saved.begin(), saved.end(), actual.begin()); };
                check(same(certificate->old_volumes, ale.old_cell_volumes()) &&
                          same(certificate->new_volumes, ale.new_cell_volumes()) &&
                          same(certificate->swept_flux, ale.face_mesh_fluxes()),
                    "Re-staged CV/swept receipt differs from the retained immutable geometry certificate.");
                certificate->validate_maps(mesh);
            });
        d->refresh_numeric_geometry();
        d->stage_publication_fields(candidate, candidate_flux);
        motion.accept_trial();
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        restore_accepted_after_motion(motion, accepted_flux, accepted_frame);
        std::rethrow_exception(failure);
    }
    d->publish_staged_fields();
    d->accepted = std::move(candidate);
    d->diagnostics = std::move(statistics);
    if (publication)
    {
        d->qbmm_publication = std::move(publication);
        d->accepted_qbmm_controller = d->active_qbmm_controller;
        d->accepted_planar_motion = motion.d_authority;
        d->explicit_family_force_recorded = linear;
        d->needs_restart_controller = false;
    }
    ++d->logical_geometry;
    ++d->trial_nonce;
}

LowMachMixtureSolver::Checkpoint LowMachMixtureSolver::checkpoint(const PlanarMotion& motion) const
{
    auto geometry = capture_accepted_geometry(motion);
    Checkpoint result;
    collective_detail::collective_local_validation(*d->mesh, "Planar checkpoint state ownership",
        [&]
        {
            d->validate_planar_checkpoint_provenance(motion);
            result = checkpoint_state();
        });
    result.ale_geometry = std::move(geometry);
    return result;
}

void LowMachMixtureSolver::rollback_planar_geometry(PlanarMotion& motion, PlanarMotionFrame& accepted_frame)
{
    const bool active = motion.has_active_trial();
    const bool linear = uses_qbmm_linear_transport(d->options.qbmm_model);
    if (active)
    {
        if (linear)
            motion.rollback_trial_preserving_epoch(accepted_frame.epoch);
        else
            motion.rollback_trial();
    }
    if (linear)
    {
        motion.d_old_cell_volumes = std::move(accepted_frame.old_volumes);
        motion.d_new_cell_volumes = std::move(accepted_frame.new_volumes);
        motion.d_face_mesh_fluxes = std::move(accepted_frame.swept_rates);
        motion.d_diagnostics = accepted_frame.diagnostics;
    }
    // A failed refresh can leave some trial caches rebuilt before it updates d->epoch.
    if (active || d->epoch != mesh_geometry_epoch(*d->mesh))
        d->refresh_numeric_geometry();
}

void LowMachMixtureSolver::restore(const Checkpoint& saved, PlanarMotion& motion, const ControllerLease* controller)
{
    require_controller(controller);
    const auto& mesh = *d->mesh;
    const bool linear = uses_qbmm_linear_transport(d->options.qbmm_model);
    State candidate;
    Diagnostics statistics;
    std::shared_ptr<const QbmmPublication> publication;
    collective_detail::collective_local_validation(mesh, "Planar checkpoint authority",
        [&]
        {
            d->validate_planar_controller_authority(motion);
            qbmm_detail::validate_capability(d->options, mesh, true, acceleration_vector(d->active_body_acceleration));
            check(saved.owner == this && saved.ale_geometry && saved.ale_geometry->owner == d->planar_owner &&
                      saved.ale_geometry->motion_identity == motion.d_authority &&
                      saved.ale_geometry->geometry_identity == mesh.geometry_identity() &&
                      saved.ale_geometry->logical_geometry == saved.logical_geometry &&
                      (!d->accepted_planar_motion || d->accepted_planar_motion == motion.d_authority) &&
                      motion.mesh_ptr().get() == d->mesh.get() && !motion.has_active_trial() && d->initialized &&
                      d->epoch == mesh_geometry_epoch(mesh) &&
                      d->logical_geometry != std::numeric_limits<std::uint64_t>::max() &&
                      saved.state.cells.size() == mesh.num_owned_cells() &&
                      saved.face_flux.size() == mesh.num_owned_faces(),
                "Planar checkpoint requires its exact explicit live motion and unchanged native topology.");
            check(!d->controller_active || d->active_qbmm_controller == saved.qbmm_controller_signature,
                "Planar QBMM checkpoint belongs to a different coupled source configuration.");
            check(!saved.qbmm_legacy_default_only || !d->controller_active || d->active_legacy_compatible,
                "Legacy QBMM checkpoint requires default force and constitutive-law configuration.");
            qbmm_detail::validate_state(d->options, saved.state);
            candidate = saved.state;
            statistics = saved.diagnostics;
            if (!d->options.qbmm_families.empty())
            {
                auto metadata = saved.qbmm_publication ? std::make_shared<QbmmPublication>(*saved.qbmm_publication)
                                                       : std::make_shared<QbmmPublication>();
                metadata->origin = QbmmStateOrigin::Checkpoint;
                publication = std::move(metadata);
            }
        });
    collective_detail::require_uniform_value(mesh, saved.generation, "Planar checkpoint generation");
    collective_detail::require_uniform_value(mesh, saved.logical_geometry, "Planar checkpoint logical geometry");
    collective_detail::require_uniform_value(mesh, candidate.thermodynamic_pressure, "Planar checkpoint pressure");
    collective_detail::require_uniform_value(mesh, candidate.time, "Planar checkpoint time");
    collective_detail::require_uniform_value(mesh, candidate.revision, "Planar checkpoint revision");
    collective_detail::require_uniform_value(
        mesh, int(saved.qbmm_controller_signature.has_value()), "Planar checkpoint controller presence");
    collective_detail::require_uniform_value(
        mesh, saved.qbmm_controller_signature.value_or(0), "Planar checkpoint controller signature");
    collective_detail::require_uniform_value(
        mesh, int(saved.qbmm_legacy_default_only), "Planar checkpoint legacy configuration restriction");
    auto accepted_frame = capture_planar_motion_frame(motion);
    try
    {
        motion.stage_restart_geometry(saved.ale_geometry->source.surface_elevation(), linear);
        collective_detail::collective_local_validation(mesh, "Planar checkpoint storage geometry",
            [&]
            {
                const auto& volumes = saved.ale_geometry->new_volumes;
                const auto live = motion.new_cell_volumes();
                check(volumes.size() == live.size() && std::equal(volumes.begin(), volumes.end(), live.begin()),
                    "Restored planar checkpoint volumes differ from the saved geometry.");
            });
        d->refresh_numeric_geometry();
        d->stage_publication_fields(candidate, saved.face_flux);
        motion.accept_trial();
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        rollback_planar_geometry(motion, accepted_frame);
        std::rethrow_exception(failure);
    }
    d->publish_staged_fields();
    d->accepted = std::move(candidate);
    d->diagnostics = std::move(statistics);
    d->qbmm_publication = std::move(publication);
    d->accepted_qbmm_controller = saved.qbmm_controller_signature;
    d->accepted_count_only_body_acceleration = saved.count_only_body_acceleration;
    d->count_only_force_provenance_recorded = d->options.qbmm_families.empty();
    d->needs_restart_controller = saved.qbmm_controller_signature.has_value();
    d->qbmm_legacy_default_only = saved.qbmm_legacy_default_only;
    d->explicit_family_force_recorded = saved.explicit_family_force_recorded;
    if (!d->options.qbmm_families.empty())
        d->accepted_planar_motion = motion.d_authority;
    ++d->logical_geometry;
    ++d->trial_nonce;
}
