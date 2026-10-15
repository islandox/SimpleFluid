/**
 * @file MultiphysicsSolverCouplerPlanarALE.tcc
 * @brief Private planar-ALE transaction state, physical trial phases, and replay.
 *
 * Included only by the compiled coupler implementation. This step-local state
 * retains preview lifetimes through each corrector and performs no destructor
 * rollback or MPI work.
 */

namespace SimpleFluid
{

/** @brief Own one accepted snapshot and the trial schedule for a planar-ALE step. */
template<TpetraTypePack Pack> struct MultiphysicsSolverCoupler<Pack>::PlanarALEStep
{
    /** Preview ownership lasts through commit or internal replay of one corrector. */
    struct TrialState
    {
        std::optional<typename liquid_mass_inventory_type::PhaseChangePreview> temperature_mass_preview;
        std::vector<scalar_type> next_density_guess, next_heat_capacity_guess, current_gas_state;
        std::optional<typename liquid_mass_inventory_type::PhaseChangePreview> liquid_preview;
        std::optional<typename free_surface_model_type::UpdatePreview> surface_preview;
        const field_type* trial_mass = nullptr;
        std::vector<scalar_type> new_material_volume;
        std::optional<face_flux_field_type> carrier_material_volume_flux;
        std::optional<typename volume_continuity_model_type::Trial> volume_trial;
        scalar_type material_state_residual{}, gas_state_residual{}, gas_inventory_residual{};
        std::array<scalar_type, ALEConvergenceObservation::gas_field_count> gas_field_residual{};
        scalar_type boundary_energy{}, energy_residual{}, energy_scale{};
        scalar_type level_residual{}, level_tolerance{}, target_tolerance{};
        continuity_residual_type final_continuity;
        scalar_type pool_volume_mismatch{}, pool_volume_tolerance{}, thermal_residual{};
        bool converged = false;
    };

    MultiphysicsSolverCoupler& coupler;
    solver_type& d_solver;
    FieldStateSnapshot<field_type> pressure_snapshot;
    FieldStateSnapshot<field_type> pressure_correction_snapshot;
    FieldStateSnapshot<velocity_field_type> velocity_snapshot;
    FieldStateSnapshot<velocity_field_type> predictor_gradient_snapshot;
    FieldStateSnapshot<velocity_field_type> predictor_velocity_snapshot;
    FieldStateSnapshot<face_flux_field_type> old_flux_snapshot;
    FieldStateSnapshot<face_flux_field_type> projected_flux_snapshot;
    FieldStateSnapshot<face_flux_field_type> relative_flux_snapshot;
    FieldStateSnapshot<face_flux_field_type> bubble_slip_flux_snapshot;
    FieldStateSnapshot<field_type> mesh_volume_rate_snapshot;
    FieldStateSnapshot<field_type> temperature_snapshot;
    FieldStateSnapshot<field_type> clear_level_snapshot;
    FieldStateSnapshot<field_type> pool_level_snapshot;
    FieldStateSnapshot<field_type> headspace_pressure_snapshot;
    FieldStateSnapshot<field_type> occupancy_snapshot;
    FieldStateSnapshot<field_type> continuity_residual_snapshot;
    std::optional<typename prescribed_heat_source_type::StateSnapshot> prescribed_heat_snapshot;
    const typename material_type::StateSnapshot material_snapshot;
    std::optional<typename material_feedback_model_type::StateSnapshot> material_feedback_snapshot;
    const typename liquid_mass_inventory_type::StateSnapshot liquid_snapshot;
    const typename free_surface_model_type::StateSnapshot free_surface_snapshot;
    const typename volume_continuity_model_type::StateSnapshot volume_model_snapshot;
    std::optional<typename reacting_gas_model_type::StateSnapshot> reacting_snapshot;
    std::optional<typename scalar_void_fraction_model_type::StateSnapshot> void_snapshot;
    const step_statistics_type accepted_statistics;
    const residual_type accepted_residuals;
    const continuity_residual_type accepted_volume_residuals;
    const scalar_type accepted_time;
    const int accepted_step;
    const std::size_t accepted_history_size;
    const PlanarALEStepDiagnostics accepted_ale_diagnostics;
    const scalar_type accepted_occupancy_volume_error;

    std::vector<real_t> accepted_cell_volumes;
    scalar_type accepted_mesh_volume{}, old_energy{}, time_step{};
    std::vector<scalar_type> old_material_volume;
    free_surface_diagnostics_type accepted_surface;
    SP<const typename thermal_participant_type::State> accepted_thermal, substep_thermal;
    SP<const typename temperature_equation_type::boundary_cache_type> accepted_thermal_flux;
    std::unique_ptr<field_type> thermal_guess;
    std::vector<scalar_type> target_guess, density_guess, heat_capacity_guess, previous_gas_state;
    scalar_type accepted_mesh_level{}, top_area{}, candidate_level{}, mesh_volume_correction_rate{};
    scalar_type last_level_residual{}, last_target_change{}, last_continuity_maximum{};
    scalar_type last_level_tolerance{}, last_target_tolerance{}, last_pool_volume_mismatch{},
        last_pool_volume_tolerance{};
    scalar_type last_material_state_residual{}, last_gas_state_residual{}, last_thermal_state_residual{};
    static constexpr scalar_type state_relative_tolerance{1.0e-10};
    std::vector<scalar_type> level_residual_history, target_change_history, continuity_maximum_history;
    std::vector<scalar_type> material_state_residual_history, gas_state_residual_history;

    template<class Model>
    static std::optional<typename Model::StateSnapshot> capture_optional(const std::unique_ptr<Model>& model)
    {
        if (model)
            return model->snapshot();
        return std::nullopt;
    }

    explicit PlanarALEStep(MultiphysicsSolverCoupler& owner)
        : coupler(owner), d_solver(*owner.d_solver), pressure_snapshot(d_solver.pressure()),
          pressure_correction_snapshot(d_solver.pressure_correction()), velocity_snapshot(d_solver.velocity()),
          predictor_gradient_snapshot(d_solver.predictor_pressure_gradient()),
          predictor_velocity_snapshot(d_solver.predictor_velocity()), old_flux_snapshot(d_solver.old_face_fluxes()),
          projected_flux_snapshot(d_solver.projected_face_fluxes()),
          relative_flux_snapshot(*coupler.d_mesh_relative_face_flux),
          bubble_slip_flux_snapshot(*coupler.d_bubble_slip_volume_flux),
          mesh_volume_rate_snapshot(*coupler.d_mesh_volume_rate), temperature_snapshot(d_solver.temperature()),
          clear_level_snapshot(*coupler.d_clear_level), pool_level_snapshot(*coupler.d_pool_level),
          headspace_pressure_snapshot(*coupler.d_headspace_pressure), occupancy_snapshot(*coupler.d_pool_occupancy),
          continuity_residual_snapshot(*coupler.d_continuity_residual),
          prescribed_heat_snapshot(capture_optional(coupler.d_prescribed_heat_source)),
          material_snapshot(d_solver.stored_material_properties().snapshot()),
          material_feedback_snapshot(capture_optional(coupler.d_material_feedback_model)),
          liquid_snapshot(coupler.d_liquid_mass_inventory->snapshot()),
          free_surface_snapshot(coupler.d_free_surface_model->snapshot()),
          volume_model_snapshot(coupler.d_volume_continuity_model->snapshot()),
          reacting_snapshot(capture_optional(coupler.d_reacting_gas_model)),
          void_snapshot(capture_optional(coupler.d_scalar_void_fraction_model)),
          accepted_statistics(d_solver.d_last_step_statistics),
          accepted_residuals(d_solver.pressure_velocity_residuals()),
          accepted_volume_residuals(d_solver.last_volume_continuity_residuals()), accepted_time(d_solver.d_time),
          accepted_step(d_solver.d_step_index), accepted_history_size(coupler.d_free_surface_history.size()),
          accepted_ale_diagnostics(coupler.d_planar_ale_diagnostics),
          accepted_occupancy_volume_error(coupler.d_pool_occupancy_volume_error)
    {
        capture_geometry_and_energy();
        capture_thermal_state();
        initialize_iteration();
    }

    /** @brief Run the coupled corrector schedule with one outward rollback boundary. */
    void run()
    {
        try
        {
            begin_thermal_substep();
            for (int corrector = 1; corrector <= coupler.d_free_surface_options.ale.maximum_correctors; ++corrector)
            {
                const auto timing = d_solver.d_solver_timings.scope(SolverPhase::ALETrial);
                TrialState trial;
                prepare_trial();
                advance_flow();
                advance_temperature(trial);
                advance_gas();
                evaluate_material_state(trial);
                evaluate_gas_state(trial);
                preview_surface(trial);
                preview_volume_balance(trial);
                validate_energy_balance(trial);
                evaluate_convergence(trial, corrector);
                if (trial.converged)
                {
                    commit_trial(trial, corrector);
                    return;
                }
                replay_trial(trial);
            }
            throw_nonconvergence();
        }
        catch (...)
        {
            reject(std::current_exception());
        }
    }

    /** @brief Freeze old geometry, storage coefficients, and extensive energy. */
    void capture_geometry_and_energy()
    {
        accepted_cell_volumes.resize(d_solver.d_mesh->num_local_cells());
        for (size_t local = 0; local < accepted_cell_volumes.size(); ++local)
        {
            accepted_cell_volumes[local] = d_solver.d_mesh->cell_volume(static_cast<local_ordinal_type>(local));
        }
        scalar_type local_accepted_mesh_volume{};
        for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
        {
            local_accepted_mesh_volume += static_cast<scalar_type>(accepted_cell_volumes[owned]);
        }
        accepted_mesh_volume = {};
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1,
            &local_accepted_mesh_volume, &accepted_mesh_volume);
        {
            const auto density = d_solver.stored_material_properties().density.owned_read_view();
            const auto heat_capacity = d_solver.stored_material_properties().specific_heat_capacity.owned_read_view();
            auto old_density = coupler.d_ale_old_density->owned_write_view();
            auto old_heat_capacity = coupler.d_ale_old_heat_capacity->owned_write_view();
            for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
            {
                old_density(owned, 0) = density(owned, 0);
                old_heat_capacity(owned, 0) = heat_capacity(owned, 0);
            }
        }
        coupler.d_ale_old_density->sync_ghosts();
        coupler.d_ale_old_heat_capacity->sync_ghosts();
        old_material_volume = coupler.material_volumes(coupler.d_liquid_mass_inventory->cellMassInventory(),
            coupler.d_liquid_mass_inventory->pureLiquidDensity(), accepted_cell_volumes);
        old_energy = coupler.total_sensible_energy(coupler.d_liquid_mass_inventory->cellMassInventory(),
            *coupler.d_ale_old_heat_capacity, accepted_cell_volumes);
        accepted_surface = coupler.d_free_surface_model->diagnostics();
        time_step = static_cast<scalar_type>(d_solver.d_problem.time_options().time_step);
    }

    /** @brief Capture external thermal state and initialize its Picard guess. */
    void capture_thermal_state()
    {
        accepted_thermal = coupler.d_thermal_participant ? coupler.d_thermal_participant->capture_state() : nullptr;
        if (coupler.d_thermal_participant)
            collective_detail::collective_local_validation(*d_solver.d_mesh, "ALE thermal accepted state",
                [&]
                {
                    if (coupler.d_thermal_participant && !accepted_thermal)
                        throw std::invalid_argument("Thermal participant returned a null accepted state.");
                });
        accepted_thermal_flux = d_solver.temperature_equation().boundary_heat_flux();
        if (coupler.d_thermal_participant)
        {
            thermal_guess = std::make_unique<field_type>(d_solver.d_mesh, "ale_thermal_guess");
            {
                const auto source = d_solver.temperature().owned_read_view();
                auto destination = thermal_guess->owned_write_view();
                for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
                    destination(owned, 0) = source(owned, 0);
            }
            thermal_guess->sync_ghosts();
        }
    }

    /** @brief Seed target and material guesses from the accepted state. */
    void initialize_iteration()
    {
        target_guess.assign(d_solver.d_mesh->num_owned_cells(), scalar_type{});
        density_guess.resize(d_solver.d_mesh->num_owned_cells());
        heat_capacity_guess.resize(d_solver.d_mesh->num_owned_cells());
        {
            const auto density = coupler.d_ale_old_density->owned_read_view();
            const auto heat_capacity = coupler.d_ale_old_heat_capacity->owned_read_view();
            for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
            {
                density_guess[owned] = density(owned, 0);
                heat_capacity_guess[owned] = heat_capacity(owned, 0);
            }
        }
        // Pressure prescribes the exact swept flux on the moving top. Its global
        // source and trial geometry must therefore describe the same volume change.
        // The thermodynamic pool level can differ from accepted mesh coordinates
        // within the outer-corrector tolerance; it is not an exact geometric origin.
        accepted_mesh_level = coupler.d_ale_boundary->diagnostics().surface_elevation;
        top_area = coupler.d_ale_boundary->diagnostics().global_area;
        candidate_level = accepted_mesh_level;
        // A rate-only physical ledger preserves any accepted pool/mesh offset.
        // Remove that offset over this step in both pressure and swept geometry.
        // Recompute from restored accepted state and the current dt; a previous
        // step's numerical correction must not become the new physical predictor.
        mesh_volume_correction_rate =
            (static_cast<scalar_type>(accepted_surface.pool_volume) - accepted_mesh_volume) / time_step;
        for (size_t owned = 0; owned < target_guess.size(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            if (coupler.d_volume_continuity_model->generation() != 0)
            {
                target_guess[owned] = coupler.d_volume_continuity_model->material_source_field().value(cell) +
                                      coupler.d_volume_continuity_model->slip_contribution_field().value(cell);
            }
            target_guess[owned] += mesh_volume_correction_rate *
                                   (static_cast<scalar_type>(accepted_cell_volumes[owned]) / accepted_mesh_volume);
        }

        // Compare accepted-to-trial gas state first, then successive new-time trials.
        previous_gas_state = capture_gas_state();
    }

    /** @brief Read authoritative hydrogen/population fields in their existing order. */
    std::vector<scalar_type> capture_gas_state() const
    {
        std::vector<scalar_type> state;
        if (coupler.d_reacting_gas_model && coupler.d_reacting_gas_model->enabled())
        {
            const std::array<const field_type*, 5> fields{
                &coupler.d_reacting_gas_model->dissolved_hydrogen_inventory(),
                &coupler.d_reacting_gas_model->micro_number_density(), &coupler.d_reacting_gas_model->micro_moles(),
                &coupler.d_reacting_gas_model->large_number_density(),
                &coupler.d_reacting_gas_model->large_moles()};
            state.reserve(fields.size() * d_solver.d_mesh->num_owned_cells());
            for (const auto* field : fields)
            {
                const auto values = field->owned_read_view();
                for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
                    state.push_back(values(owned, 0));
            }
        }
        return state;
    }

    /** @brief Restore accepted physics; internal replay may defer metric reconstruction. */
    void restore_accepted(bool refresh_metrics = true)
    {
        const auto restore_timing = d_solver.d_solver_timings.scope(SolverPhase::ALERestore);
        d_solver.clear_volume_continuity_target();
        coupler.clear_ale_pressure_boundary();
        coupler.d_ale_temperature_density = nullptr;
        coupler.d_active_ale.reset();
        if (coupler.d_ale_motion->has_active_trial())
        {
            coupler.d_ale_motion->rollback_trial();
        }
        if (refresh_metrics)
        {
            coupler.refresh_geometry_dependent_state();
            coupler.d_ale_boundary->refresh(coupler.d_free_surface_model->volumeMap());
        }
        pressure_snapshot.restore(d_solver.pressure());
        pressure_correction_snapshot.restore(d_solver.pressure_correction());
        velocity_snapshot.restore(d_solver.velocity());
        predictor_gradient_snapshot.restore(d_solver.predictor_pressure_gradient());
        predictor_velocity_snapshot.restore(d_solver.predictor_velocity());
        old_flux_snapshot.restore(d_solver.old_face_fluxes());
        projected_flux_snapshot.restore(d_solver.projected_face_fluxes());
        relative_flux_snapshot.restore(*coupler.d_mesh_relative_face_flux);
        bubble_slip_flux_snapshot.restore(*coupler.d_bubble_slip_volume_flux);
        mesh_volume_rate_snapshot.restore(*coupler.d_mesh_volume_rate);
        temperature_snapshot.restore(d_solver.temperature());
        if (coupler.d_thermal_participant && accepted_thermal)
            coupler.d_thermal_participant->restore_state(
                refresh_metrics || !substep_thermal ? *accepted_thermal : *substep_thermal);
        if (coupler.d_thermal_participant)
            d_solver.temperature_equation().set_boundary_heat_flux_snapshot(accepted_thermal_flux);
        clear_level_snapshot.restore(*coupler.d_clear_level);
        pool_level_snapshot.restore(*coupler.d_pool_level);
        headspace_pressure_snapshot.restore(*coupler.d_headspace_pressure);
        occupancy_snapshot.restore(*coupler.d_pool_occupancy);
        continuity_residual_snapshot.restore(*coupler.d_continuity_residual);
        if (prescribed_heat_snapshot)
            coupler.d_prescribed_heat_source->restore(*prescribed_heat_snapshot);
        d_solver.stored_material_properties().restore(material_snapshot);
        if (material_feedback_snapshot)
        {
            coupler.d_material_feedback_model->restore(*material_feedback_snapshot);
        }
        coupler.d_liquid_mass_inventory->restore(liquid_snapshot);
        coupler.d_free_surface_model->restore(free_surface_snapshot);
        coupler.d_volume_continuity_model->restore(volume_model_snapshot);
        if (reacting_snapshot)
        {
            const auto gas_restore_timing = d_solver.d_solver_timings.scope(SolverPhase::GasRestore);
            if (refresh_metrics)
                coupler.d_reacting_gas_model->restore(*reacting_snapshot);
            else
                coupler.d_reacting_gas_model->restore_for_ale_replay(*reacting_snapshot);
        }
        if (void_snapshot)
        {
            if (refresh_metrics)
                coupler.d_scalar_void_fraction_model->restore(*void_snapshot);
            else
                coupler.d_scalar_void_fraction_model->restore_for_ale_replay(*void_snapshot);
        }
        d_solver.d_last_step_statistics = accepted_statistics;
        d_solver.pressure_velocity_residuals() = accepted_residuals;
        d_solver.d_last_volume_continuity_residuals = accepted_volume_residuals;
        d_solver.d_time = accepted_time;
        d_solver.d_step_index = accepted_step;
        coupler.d_free_surface_history.resize(accepted_history_size);
        coupler.d_planar_ale_diagnostics = accepted_ale_diagnostics;
        coupler.d_pool_occupancy_volume_error = accepted_occupancy_volume_error;
    }

    /** @brief Begin the external thermal transaction inside the rollback boundary. */
    void begin_thermal_substep()
    {
        if (coupler.d_thermal_participant)
        {
            coupler.d_thermal_participant->begin_substep(d_solver.d_time, time_step);
            substep_thermal = coupler.d_thermal_participant->capture_state();
            collective_detail::collective_local_validation(*d_solver.d_mesh, "ALE thermal substep state",
                [&]
                {
                    if (!substep_thermal)
                        throw std::invalid_argument("Thermal participant returned a null substep state.");
                });
        }
    }

    /** @brief Move to a candidate geometry and publish its Picard coefficients. */
    void prepare_trial()
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEPrepare);
        scalar_type local_target_integral{};
        for (const auto value : target_guess)
            local_target_integral += value;
        scalar_type target_integral{};
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1,
            &local_target_integral, &target_integral);
        candidate_level = accepted_mesh_level + time_step * target_integral / top_area;
        {
            const auto geometry_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEGeometry);
            coupler.d_ale_motion->begin_trial(candidate_level, time_step);
            coupler.d_active_ale.emplace(
                FVM::make_validated_planar_ale_control_volume_state(*d_solver.d_mesh, *coupler.d_ale_motion));
            coupler.refresh_geometry_dependent_state();
        }
        if (coupler.d_prescribed_heat_source && coupler.d_prescribed_heat_source->has_interval_energy())
            coupler.d_prescribed_heat_source->refresh_interval_energy(d_solver.d_time, time_step);
        // Picard data from the preceding outer trial supplies the
        // new-time rho/cp coefficients. Accepted-old coefficients remain
        // in d_ale_old_* for the conservative transient RHS.
        {
            auto density = d_solver.stored_material_properties().density.owned_write_view();
            auto heat_capacity = d_solver.stored_material_properties().specific_heat_capacity.owned_write_view();
            for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
            {
                density(owned, 0) = density_guess[owned];
                heat_capacity(owned, 0) = heat_capacity_guess[owned];
            }
        }
        d_solver.stored_material_properties().density.sync_ghosts();
        d_solver.stored_material_properties().specific_heat_capacity.sync_ghosts();
        coupler.d_ale_boundary->refresh(coupler.d_free_surface_model->volumeMap());
        coupler.d_ale_boundary->apply_kinematic_velocity(
            *coupler.d_active_ale, d_solver.boussinesq_velocity_boundary_cache());
    }

    /** @brief Solve and refine pressure/velocity against the same swept-volume target. */
    void advance_flow()
    {
        if (coupler.d_ale_target_generation == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("planarALE continuity-target generation exhausted.");
        }
        const auto solve_generation = ++coupler.d_ale_target_generation;
        const continuity_target_type solve_target(d_solver.d_mesh, target_guess, solve_generation);
        d_solver.set_volume_continuity_target(solve_target);
        coupler.configure_ale_pressure_boundary(solve_generation);
        d_solver.solve_flow(
            {coupler.d_active_ale ? &*coupler.d_active_ale : nullptr, coupler.d_mesh_relative_face_flux.get()});
        scalar_type local_solve_target_scale{};
        for (const auto value : target_guess)
        {
            local_solve_target_scale = std::max(local_solve_target_scale, std::abs(value));
        }
        scalar_type solve_target_scale{};
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
            &local_solve_target_scale, &solve_target_scale);
        const auto strict_pressure_tolerance =
            static_cast<scalar_type>(coupler.d_free_surface_options.coupling.volume_absolute_tolerance) / time_step +
            static_cast<scalar_type>(coupler.d_free_surface_options.coupling.volume_relative_tolerance) *
                std::max(scalar_type{1}, solve_target_scale);
        d_solver.refine_volume_continuity(
            coupler.d_free_surface_options.ale.maximum_correctors, strict_pressure_tolerance);
        coupler.form_mesh_relative_flux(d_solver.projected_face_fluxes());
    }

    /** @brief Advance conservative liquid energy while retaining its mass preview. */
    void advance_temperature(TrialState& trial)
    {
        // Temperature stores sensible energy in the transported liquid,
        // not in the bubble-displaced mixture volume.  Build the
        // conservative trial liquid-mass density on the candidate mesh
        // before assembling rho*cp*T at the new time level.
        {
            const auto mass_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEMassPreview);
            trial.temperature_mass_preview.emplace(
                coupler.d_liquid_mass_inventory->previewCellwiseAdvance(time_step, *coupler.d_mesh_relative_face_flux,
                    nullptr, nullptr, d_solver.d_problem.linear_options(), &*coupler.d_active_ale));
        }
        coupler.d_ale_temperature_density =
            &coupler.d_liquid_mass_inventory->trialCellMassInventory(*trial.temperature_mass_preview);
        if (trial.temperature_mass_preview->transportStatistics())
        {
            d_solver.d_last_step_statistics.add(*trial.temperature_mass_preview->transportStatistics());
        }
        try
        {
            if (coupler.d_thermal_participant)
                d_solver.temperature_equation().set_boundary_heat_flux_snapshot(
                    coupler.d_thermal_participant->preview_trial(d_solver.d_mesh, *thermal_guess));
            coupler.advance_temperature_transport(time_step);
        }
        catch (...)
        {
            coupler.d_ale_temperature_density = nullptr;
            throw;
        }
        coupler.d_ale_temperature_density = nullptr;
    }

    /** @brief Advance reaction_source with trial temperature, pressure, and relative transport. */
    void advance_gas()
    {
        if (coupler.d_reacting_gas_model && coupler.d_reacting_gas_model->enabled())
        {
            const auto pressure_mode = coupler.d_reacting_gas_model->options().pressure_mode;
            auto offset = static_cast<scalar_type>(coupler.d_free_surface_model->headspacePressure());
            if (pressure_mode == ReactingPressureMode::Reconstructed)
            {
                // The reacting reconstruction subtracts the gauge mean;
                // this offset therefore makes p_abs=p_h on the p_gauge=0 top.
                offset += coupler.volume_weighted_mean_pressure();
            }
            coupler.d_reacting_gas_model->set_absolute_pressure_offset(offset);
            {
                const auto gas_timing = d_solver.d_solver_timings.scope(SolverPhase::Gas);
                coupler.d_reacting_gas_model->advance(d_solver.d_time + time_step, time_step, d_solver.temperature(),
                    d_solver.pressure(), d_solver.velocity(), *coupler.d_mesh_relative_face_flux,
                    d_solver.stored_material_properties(),
                    coupler.d_prescribed_heat_source ? &coupler.d_prescribed_heat_source->field() : nullptr,
                    &*coupler.d_active_ale, coupler.d_free_surface_options.gravity_axis);
            }
            {
                const auto material_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEMaterial);
                coupler.update_void_fraction_models(time_step);
            }
        }
    }

    /** @brief Capture next material coefficients and their normalized Picard residual. */
    void evaluate_material_state(TrialState& trial)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEMaterial);
        coupler.refresh_material_feedback(d_solver.d_time + time_step);
        trial.next_density_guess.resize(d_solver.d_mesh->num_owned_cells());
        trial.next_heat_capacity_guess.resize(d_solver.d_mesh->num_owned_cells());
        {
            const auto density = d_solver.stored_material_properties().density.owned_read_view();
            const auto heat_capacity = d_solver.stored_material_properties().specific_heat_capacity.owned_read_view();
            for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
            {
                trial.next_density_guess[owned] = density(owned, 0);
                trial.next_heat_capacity_guess[owned] = heat_capacity(owned, 0);
            }
        }
        scalar_type local_material_state_residual{};
        for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
        {
            const auto density_scale =
                std::max({scalar_type{1}, std::abs(density_guess[owned]), std::abs(trial.next_density_guess[owned])});
            const auto heat_capacity_scale = std::max({scalar_type{1}, std::abs(heat_capacity_guess[owned]),
                std::abs(trial.next_heat_capacity_guess[owned])});
            local_material_state_residual = std::max(local_material_state_residual,
                std::max(std::abs(trial.next_density_guess[owned] - density_guess[owned]) / density_scale,
                    std::abs(trial.next_heat_capacity_guess[owned] - heat_capacity_guess[owned]) /
                        heat_capacity_scale));
        }
        trial.material_state_residual = {};
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
            &local_material_state_residual, &trial.material_state_residual);
    }

    /** @brief Capture next gas state and its normalized Picard residual. */
    void evaluate_gas_state(TrialState& trial)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEResidual);
        trial.current_gas_state = capture_gas_state();
        // Entry zero retains the original aggregate comparison. The remaining
        // entries attribute that same arithmetic to each authoritative field.
        std::array<scalar_type, 1 + ALEConvergenceObservation::gas_field_count> local{}, global{};
        if (coupler.d_reacting_gas_model && coupler.d_reacting_gas_model->enabled())
        {
            if (previous_gas_state.size() != trial.current_gas_state.size())
            {
                local.fill(scalar_type{1});
            }
            else
            {
                for (size_t entry = 0; entry < trial.current_gas_state.size(); ++entry)
                {
                    const auto scale = std::max({scalar_type{1}, std::abs(previous_gas_state[entry]),
                        std::abs(trial.current_gas_state[entry])});
                    const auto residual = std::abs(trial.current_gas_state[entry] - previous_gas_state[entry]) / scale;
                    local[0] = std::max(local[0], residual);
                    const auto field = entry / d_solver.d_mesh->num_owned_cells();
                    local[1 + field] = std::max(local[1 + field], residual);
                }
            }
        }
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
            static_cast<int>(local.size()), local.data(), global.data());
        trial.gas_state_residual = global[0];
        std::copy(global.begin() + 1, global.end(), trial.gas_field_residual.begin());
    }

    /** @brief Preview liquid and surface ledgers after validating hydrogen closure. */
    void preview_surface(TrialState& trial)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALESurfacePreview);
        {
            const auto mass_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEMassVolumeRefresh);
            trial.liquid_preview.emplace(
                coupler.d_liquid_mass_inventory->refreshCellwiseDensity(*trial.temperature_mass_preview,
                    [this](local_ordinal_type cell) { return coupler.pure_liquid_density(cell); }));
        }
        FreeSurfaceAccountingPreview accounting;
        accounting.liquid = trial.liquid_preview->diagnostics();

        trial.gas_inventory_residual = {};
        if (coupler.d_reacting_gas_model && coupler.d_reacting_gas_model->enabled())
        {
            const auto& gas_statistics = coupler.d_reacting_gas_model->last_statistics();
            trial.gas_inventory_residual = gas_statistics.inventory_error;
            const auto gas_scale = std::abs(gas_statistics.hydrogen_before) +
                                   std::abs(gas_statistics.hydrogen_produced) +
                                   std::abs(gas_statistics.hydrogen_escaped) + std::abs(gas_statistics.hydrogen_after);
            const auto gas_tolerance =
                static_cast<scalar_type>(coupler.d_free_surface_options.coupling.gas_absolute_tolerance) +
                static_cast<scalar_type>(coupler.d_free_surface_options.coupling.gas_relative_tolerance) * gas_scale;
            if (std::abs(trial.gas_inventory_residual) > gas_tolerance)
            {
                std::ostringstream message;
                message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                        << "planarALE reacting H2 step-inventory closure exceeded its physical tolerance: residual="
                        << trial.gas_inventory_residual << " mol, tolerance=" << gas_tolerance << " mol.";
                throw std::runtime_error(message.str());
            }
        }
        trial.surface_preview.emplace(coupler.d_free_surface_model->previewUpdate(
            coupler.make_free_surface_update(d_solver.d_time + time_step, false, &accounting)));
    }

    /** @brief Publish transported bubble slip and reduce its outward escape rate. */
    scalar_type update_bubble_slip_flux()
    {
        if (coupler.d_reacting_gas_model && coupler.d_reacting_gas_model->enabled())
        {
            const auto& transported_slip = coupler.d_reacting_gas_model->transported_bubble_slip_volume_flux();
            for (const auto face_lid : coupler.d_bubble_slip_volume_flux->owned_face_ids())
            {
                coupler.d_bubble_slip_volume_flux->set_owned_value(face_lid, transported_slip.value(face_lid));
            }
            coupler.d_bubble_slip_volume_flux->sync_ghosts();
        }
        else
        {
            coupler.d_bubble_slip_volume_flux->put_scalar(scalar_type{});
            coupler.d_bubble_slip_volume_flux->sync_ghosts();
        }

        scalar_type local_escape_volume_rate{};
        for (const auto face_lid : coupler.d_bubble_slip_volume_flux->owned_face_ids())
        {
            if (coupler.d_ale_boundary->contains(face_lid))
            {
                local_escape_volume_rate += std::max(coupler.d_bubble_slip_volume_flux->value(face_lid), scalar_type{});
            }
        }
        scalar_type escape_volume_rate{};
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1,
            &local_escape_volume_rate, &escape_volume_rate);

        return escape_volume_rate;
    }

    /** @brief Build the material carrier flux and preview volume continuity. */
    void preview_volume_balance(TrialState& trial)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEVolumeBalance);
        trial.trial_mass = &coupler.d_liquid_mass_inventory->trialCellMassInventory(*trial.liquid_preview);
        const auto escape_volume_rate = update_bubble_slip_flux();

        trial.new_material_volume = coupler.material_volumes(*trial.trial_mass,
            coupler.d_liquid_mass_inventory->pureLiquidDensity(), coupler.d_active_ale->new_cell_volumes());
        trial.carrier_material_volume_flux.emplace(d_solver.d_mesh, 0.0, "carrierMaterialVolumeFlux");
        const auto* transported_bubble_carrier =
            coupler.d_reacting_gas_model && coupler.d_reacting_gas_model->enabled()
                ? &coupler.d_reacting_gas_model->transported_bubble_carrier_volume_flux()
                : nullptr;
        for (const auto face_lid : trial.carrier_material_volume_flux->owned_face_ids())
        {
            const auto carrier_flux = coupler.d_mesh_relative_face_flux->value(face_lid);
            const auto owner = d_solver.d_mesh->owner_cell(face_lid);
            auto upwind = owner;
            if (carrier_flux < scalar_type{} && d_solver.d_mesh->is_interior_face(face_lid))
            {
                upwind = d_solver.d_mesh->opposite_or_periodic_neighbor_cell(face_lid, owner);
            }
            const auto liquid_fraction = trial.trial_mass->local_value(upwind) /
                                         coupler.d_liquid_mass_inventory->pureLiquidDensity().local_value(upwind);
            const auto bubble_carrier_flux =
                transported_bubble_carrier == nullptr ? scalar_type{} : transported_bubble_carrier->value(face_lid);
            trial.carrier_material_volume_flux->set_owned_value(
                face_lid, carrier_flux * liquid_fraction + bubble_carrier_flux);
        }
        trial.carrier_material_volume_flux->sync_ghosts();
        if (coupler.d_ale_target_generation == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("planarALE continuity-target generation exhausted.");
        }
        const auto ledger_generation = ++coupler.d_ale_target_generation;
        const typename volume_continuity_model_type::Inputs volume_inputs{.ale = *coupler.d_active_ale,
            .old_material_volume = old_material_volume,
            .new_material_volume = trial.new_material_volume,
            .carrier_relative_flux = *coupler.d_mesh_relative_face_flux,
            .carrier_material_volume_flux = &*trial.carrier_material_volume_flux,
            .bubble_slip_volume_flux = coupler.d_bubble_slip_volume_flux.get(),
            .old_pool_volume = static_cast<scalar_type>(accepted_surface.pool_volume),
            .new_pool_volume = static_cast<scalar_type>(trial.surface_preview->diagnostics().pool_volume),
            .bubble_escape_volume_rate = escape_volume_rate,
            .other_outflow_volume_rate = scalar_type{},
            .previous_target = target_guess,
            .mesh_volume_correction_rate = mesh_volume_correction_rate};
        {
            const auto volume_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEVolumePreview);
            trial.volume_trial.emplace(coupler.d_volume_continuity_model->preview(volume_inputs, ledger_generation));
        }
    }

    /** @brief Check extensive sensible energy against prescribed source and boundary work. */
    void validate_energy_balance(TrialState& trial)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEEnergyBalance);
        const auto new_energy = coupler.total_sensible_energy(*trial.trial_mass,
            d_solver.stored_material_properties().specific_heat_capacity, coupler.d_active_ale->new_cell_volumes());
        scalar_type local_added_energy{};
        for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            local_added_energy += static_cast<scalar_type>(coupler.d_active_ale->new_cell_volumes()[owned]) *
                                  time_step * d_solver.stored_temperature_sources().total_power_density(cell);
        }
        scalar_type added_energy{};
        Teuchos::reduceAll(
            *d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_added_energy, &added_energy);
        trial.boundary_energy = time_step * d_solver.temperature_equation().last_prescribed_boundary_power();
        trial.energy_residual = new_energy - old_energy - added_energy + trial.boundary_energy;
        trial.energy_scale = std::max({scalar_type{1}, std::abs(new_energy), std::abs(old_energy),
            std::abs(added_energy), std::abs(trial.boundary_energy)});
        const auto energy_tolerance =
            scalar_type{4096} * std::numeric_limits<scalar_type>::epsilon() * trial.energy_scale +
            scalar_type{1.0e-9} * trial.energy_scale;
        if (std::abs(trial.energy_residual) > energy_tolerance)
        {
            throw std::runtime_error("planarALE sensible-energy closure exceeded its physical tolerance: residual=" +
                                     std::to_string(trial.energy_residual) +
                                     " J, tolerance=" + std::to_string(energy_tolerance) + " J.");
        }
    }

    /** @brief Publish cellwise continuity and reduce its physical residual. */
    continuity_residual_type evaluate_continuity(const TrialState& trial)
    {
        scalar_type local_final_l2_squared{};
        scalar_type local_final_normalization_squared{};
        scalar_type local_final_maximum{};
        for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto balance = FVM::cell_flux_balance<Pack>(*d_solver.d_mesh, d_solver.projected_face_fluxes(), cell);
            const auto target = trial.volume_trial->target().integrated_rate(cell);
            const auto residual = balance - target;
            coupler.d_continuity_residual->set_owned_value(cell, residual);
            local_final_l2_squared += residual * residual;
            local_final_normalization_squared += std::max(balance * balance, target * target);
            local_final_maximum = std::max(local_final_maximum, std::abs(residual));
        }
        coupler.d_continuity_residual->sync_ghosts();
        const std::array<scalar_type, 2> local_continuity_squared{
            local_final_l2_squared, local_final_normalization_squared};
        std::array<scalar_type, 2> global_continuity_squared{};
        scalar_type global_final_maximum{};
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
            static_cast<int>(local_continuity_squared.size()), local_continuity_squared.data(),
            global_continuity_squared.data());
        Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_final_maximum,
            &global_final_maximum);
        const auto final_l2 = std::sqrt(global_continuity_squared[0]);
        const auto final_normalization = std::sqrt(global_continuity_squared[1]);
        return continuity_residual_type{final_l2, global_final_maximum,
            final_normalization > scalar_type{} ? final_l2 / final_normalization : final_l2, final_normalization};
    }

    /** @brief Validate and reduce the external thermal convergence residual. */
    scalar_type evaluate_thermal_residual()
    {
        scalar_type thermal_residual{};
        if (coupler.d_thermal_participant)
        {
            thermal_residual = coupler.d_thermal_participant->residual(d_solver.temperature());
            collective_detail::collective_local_validation(*d_solver.d_mesh, "ALE thermal residual",
                [&]
                {
                    if (!std::isfinite(thermal_residual) || thermal_residual < scalar_type{})
                        throw std::runtime_error("Thermal participant returned invalid residual.");
                });
            scalar_type global_thermal_residual{};
            Teuchos::reduceAll(*d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &thermal_residual,
                &global_thermal_residual);
            thermal_residual = global_thermal_residual;
        }
        return thermal_residual;
    }

    /** @brief Record residual histories and apply all physical acceptance gates. */
    void evaluate_convergence(TrialState& trial, int corrector)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEResidual);
        trial.level_residual =
            static_cast<scalar_type>(trial.surface_preview->diagnostics().pool_level) - candidate_level;
        last_level_residual = trial.level_residual;
        const auto level_scale = std::max({scalar_type{1}, std::abs(candidate_level),
            std::abs(static_cast<scalar_type>(trial.surface_preview->diagnostics().pool_level))});
        trial.level_tolerance =
            static_cast<scalar_type>(coupler.d_free_surface_options.ale.level_absolute_tolerance) +
            static_cast<scalar_type>(coupler.d_free_surface_options.ale.level_relative_tolerance) * level_scale;
        scalar_type local_target_scale{};
        for (size_t owned = 0; owned < target_guess.size(); ++owned)
        {
            local_target_scale = std::max(local_target_scale,
                std::max(std::abs(target_guess[owned]),
                    std::abs(trial.volume_trial->target().integrated_rate(static_cast<local_ordinal_type>(owned)))));
        }
        scalar_type target_scale{};
        Teuchos::reduceAll(
            *d_solver.d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_target_scale, &target_scale);
        trial.target_tolerance =
            static_cast<scalar_type>(coupler.d_free_surface_options.coupling.volume_absolute_tolerance) / time_step +
            static_cast<scalar_type>(coupler.d_free_surface_options.coupling.volume_relative_tolerance) *
                std::max(scalar_type{1}, target_scale);
        last_target_tolerance = trial.target_tolerance;
        trial.final_continuity = evaluate_continuity(trial);
        const auto continuity_tolerance = trial.target_tolerance;
        trial.pool_volume_mismatch = coupler.d_ale_boundary->diagnostics().global_mesh_volume -
                                     static_cast<scalar_type>(trial.surface_preview->diagnostics().pool_volume);
        const auto pool_volume_scale =
            std::max({scalar_type{1}, std::abs(coupler.d_ale_boundary->diagnostics().global_mesh_volume),
                std::abs(static_cast<scalar_type>(trial.surface_preview->diagnostics().pool_volume))});
        trial.pool_volume_tolerance =
            static_cast<scalar_type>(coupler.d_free_surface_options.coupling.volume_absolute_tolerance) +
            static_cast<scalar_type>(coupler.d_free_surface_options.coupling.volume_relative_tolerance) *
                pool_volume_scale;
        if (coupler.d_free_surface_options.ale.level_tolerance_from_volume)
            trial.level_tolerance = trial.pool_volume_tolerance / top_area;
        last_level_tolerance = trial.level_tolerance;
        last_target_change = trial.volume_trial->diagnostics().maximum_target_change;
        last_continuity_maximum = trial.final_continuity.maximum;
        last_pool_volume_mismatch = trial.pool_volume_mismatch;
        last_pool_volume_tolerance = trial.pool_volume_tolerance;
        last_material_state_residual = trial.material_state_residual;
        last_gas_state_residual = trial.gas_state_residual;
        level_residual_history.push_back(trial.level_residual);
        target_change_history.push_back(last_target_change);
        continuity_maximum_history.push_back(last_continuity_maximum);
        material_state_residual_history.push_back(trial.material_state_residual);
        gas_state_residual_history.push_back(trial.gas_state_residual);
        trial.thermal_residual = evaluate_thermal_residual();
        last_thermal_state_residual = trial.thermal_residual;
        trial.converged = trial.thermal_residual <= scalar_type{1} &&
                          std::abs(trial.level_residual) <= trial.level_tolerance &&
                          trial.volume_trial->diagnostics().maximum_target_change <= trial.target_tolerance &&
                          trial.final_continuity.maximum <= continuity_tolerance &&
                          std::abs(trial.pool_volume_mismatch) <= trial.pool_volume_tolerance &&
                          trial.material_state_residual <= state_relative_tolerance &&
                          trial.gas_state_residual <= state_relative_tolerance;
        const auto normalized = [](scalar_type residual, scalar_type tolerance) {
            return tolerance > scalar_type{} ? static_cast<double>(std::abs(residual) / tolerance)
                : residual == scalar_type{} ? 0. : std::numeric_limits<double>::infinity();
        };
        // Diagnostic division never replaces the original comparisons at a tolerance boundary.
        coupler.d_ale_convergence_statistics.record(
            {static_cast<double>(trial.thermal_residual), normalized(trial.level_residual,trial.level_tolerance),
             normalized(last_target_change,trial.target_tolerance), normalized(trial.final_continuity.maximum,continuity_tolerance),
             normalized(trial.pool_volume_mismatch,trial.pool_volume_tolerance),
             normalized(trial.material_state_residual,state_relative_tolerance), normalized(trial.gas_state_residual,state_relative_tolerance)},
            {!(trial.thermal_residual <= scalar_type{1}), !(std::abs(trial.level_residual) <= trial.level_tolerance),
             !(last_target_change <= trial.target_tolerance), !(trial.final_continuity.maximum <= continuity_tolerance),
             !(std::abs(trial.pool_volume_mismatch) <= trial.pool_volume_tolerance),
             !(trial.material_state_residual <= state_relative_tolerance), !(trial.gas_state_residual <= state_relative_tolerance)});
        ALEConvergenceObservation observation;
        observation.check = coupler.d_ale_convergence_statistics.checks;
        observation.step = d_solver.d_step_index;
        observation.corrector = corrector;
        observation.time = static_cast<double>(d_solver.d_time);
        observation.time_step = static_cast<double>(time_step);
        observation.gas_tolerance = static_cast<double>(state_relative_tolerance);
        const std::array<bool, ALEConvergenceStatistics::criterion_count> failed{
            !(trial.thermal_residual <= scalar_type{1}), !(std::abs(trial.level_residual) <= trial.level_tolerance),
            !(last_target_change <= trial.target_tolerance), !(trial.final_continuity.maximum <= continuity_tolerance),
            !(std::abs(trial.pool_volume_mismatch) <= trial.pool_volume_tolerance),
            !(trial.material_state_residual <= state_relative_tolerance), !(trial.gas_state_residual <= state_relative_tolerance)};
        for (size_t i = 0; i < failed.size(); ++i)
            if (failed[i]) observation.blocking_mask |= size_t{1} << i;
        for (size_t i = 0; i < trial.gas_field_residual.size(); ++i)
            observation.gas_relative_change[i] = static_cast<double>(trial.gas_field_residual[i]);
        collective_detail::collective_local_validation(*d_solver.d_mesh, "ALE convergence observer", [&] {
            if (coupler.d_ale_convergence_observer) coupler.d_ale_convergence_observer(observation);
        });
    }

    /** @brief Publish one accepted diagnostic record with its corrector histories. */
    void publish_diagnostics(const TrialState& trial, const MeshMotionDiagnostics& motion_diagnostics, int corrector)
    {
        PlanarALEStepDiagnostics accepted_diagnostics;
        accepted_diagnostics.initialized = true;
        accepted_diagnostics.old_geometry_epoch = motion_diagnostics.old_geometry_epoch;
        accepted_diagnostics.new_geometry_epoch = motion_diagnostics.new_geometry_epoch;
        accepted_diagnostics.old_mesh_volume = accepted_mesh_volume;
        accepted_diagnostics.new_mesh_volume = coupler.d_ale_boundary->diagnostics().global_mesh_volume;
        accepted_diagnostics.mesh_vessel_mismatch = trial.pool_volume_mismatch;
        accepted_diagnostics.maximum_gcl_residual = motion_diagnostics.maximum_absolute_gcl_residual;
        accepted_diagnostics.maximum_normalized_gcl_residual = motion_diagnostics.maximum_normalized_gcl_residual;
        accepted_diagnostics.mesh_quality = motion_diagnostics.mesh_quality;
        accepted_diagnostics.outer_correctors = corrector;
        accepted_diagnostics.level_residual = trial.level_residual;
        accepted_diagnostics.pressure_residual = static_cast<scalar_type>(
            trial.surface_preview->headspacePressure() - coupler.d_free_surface_model->headspacePressure());
        accepted_diagnostics.material_state_residual = trial.material_state_residual;
        accepted_diagnostics.gas_state_residual = trial.gas_state_residual;
        accepted_diagnostics.thermal_state_residual = trial.thermal_residual;
        accepted_diagnostics.outward_boundary_energy = trial.boundary_energy;
        accepted_diagnostics.energy_residual = trial.energy_residual;
        accepted_diagnostics.normalized_energy_residual = trial.energy_residual / trial.energy_scale;
        accepted_diagnostics.liquid_mass_residual = trial.liquid_preview->diagnostics().step_mass_balance_residual;
        accepted_diagnostics.gas_inventory_residual = trial.gas_inventory_residual;
        accepted_diagnostics.volume_source = trial.volume_trial->diagnostics();
        accepted_diagnostics.continuity = trial.final_continuity;
        accepted_diagnostics.level_residual_history = std::move(level_residual_history);
        accepted_diagnostics.target_change_history = std::move(target_change_history);
        accepted_diagnostics.continuity_maximum_history = std::move(continuity_maximum_history);
        accepted_diagnostics.material_state_residual_history = std::move(material_state_residual_history);
        accepted_diagnostics.gas_state_residual_history = std::move(gas_state_residual_history);
        accepted_diagnostics.rejected_transactions = accepted_ale_diagnostics.rejected_transactions;
        coupler.d_planar_ale_diagnostics = std::move(accepted_diagnostics);
    }

    /** @brief Commit ledgers and publication before releasing the active geometry trial. */
    void commit_trial(const TrialState& trial, int corrector)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALECommit);
        if (coupler.d_thermal_participant)
            coupler.d_thermal_participant->validate_commit();
        const auto motion_diagnostics = coupler.d_ale_motion->diagnostics();
        for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            coupler.d_mesh_volume_rate->set_owned_value(
                cell, static_cast<scalar_type>((coupler.d_active_ale->new_cell_volumes()[owned] -
                                                   coupler.d_active_ale->old_cell_volumes()[owned]) /
                                               time_step));
        }
        coupler.d_mesh_volume_rate->sync_ghosts();
        coupler.d_free_surface_model->commitUpdate(*trial.surface_preview);
        coupler.d_liquid_mass_inventory->commitPhaseChange(*trial.liquid_preview);
        if (trial.liquid_preview->transportStatistics())
        {
            d_solver.d_last_step_statistics.add(*trial.liquid_preview->transportStatistics());
        }
        coupler.d_volume_continuity_model->commit(*trial.volume_trial);
        d_solver.d_last_volume_continuity_residuals = trial.final_continuity;
        d_solver.pressure_velocity_residuals().continuity = trial.final_continuity.l2;

        publish_diagnostics(trial, motion_diagnostics, corrector);

        coupler.publish_free_surface_fields();
        coupler.record_free_surface_history();
        // Leave the geometry trial active until every other
        // potentially throwing accepted-state publication has
        // completed.  A failure above or in finish_step() can then
        // still roll geometry, fields, ledgers, time, and history
        // back together.
        d_solver.finish_step();
        if (coupler.d_thermal_participant)
            coupler.d_thermal_participant->commit();
        coupler.d_ale_motion->accept_trial();
        coupler.d_ale_temperature_density = nullptr;
        coupler.d_active_ale.reset();
        d_solver.clear_volume_continuity_target();
        coupler.clear_ale_pressure_boundary();
    }

    /** @brief Preserve next guesses across accepted-state restoration. */
    void replay_trial(TrialState& trial)
    {
        const auto phase_timing = d_solver.d_solver_timings.scope(SolverPhase::ALEReplay);
        std::vector<scalar_type> next_target(target_guess.size());
        const auto relaxation = static_cast<scalar_type>(coupler.d_free_surface_options.ale.relaxation);
        for (size_t owned = 0; owned < next_target.size(); ++owned)
        {
            next_target[owned] = target_guess[owned] + relaxation * (trial.volume_trial->target().integrated_rate(
                                                                         static_cast<local_ordinal_type>(owned)) -
                                                                        target_guess[owned]);
        }
        // No geometry-dependent consumer runs between this internal
        // rollback and the next candidate's begin_trial/refresh. Preserve
        // all accepted physical state now; reconstruct metrics only for
        // the candidate. Any exception (including begin/refresh failure)
        // reaches the eager restore below before leaving this function.
        if (thermal_guess)
        {
            {
                const auto source = d_solver.temperature().owned_read_view();
                auto destination = thermal_guess->owned_write_view();
                for (size_t owned = 0; owned < d_solver.d_mesh->num_owned_cells(); ++owned)
                    destination(owned, 0) = source(owned, 0);
            }
            thermal_guess->sync_ghosts();
        }
        restore_accepted(false);
        target_guess = std::move(next_target);
        density_guess = std::move(trial.next_density_guess);
        heat_capacity_guess = std::move(trial.next_heat_capacity_guess);
        previous_gas_state = std::move(trial.current_gas_state);
    }

    /** @brief Report the last physical residuals when correctors are exhausted. */
    [[noreturn]] void throw_nonconvergence() const
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                << "planarALE outer level/continuity corrector did not converge in "
                << coupler.d_free_surface_options.ale.maximum_correctors
                << " iterations: level residual=" << last_level_residual << " m (tolerance=" << last_level_tolerance
                << "), target change=" << last_target_change << " m^3/s (tolerance=" << last_target_tolerance
                << "), continuity maximum=" << last_continuity_maximum
                << " m^3/s, mesh/pool-volume mismatch=" << last_pool_volume_mismatch
                << " m^3 (tolerance=" << last_pool_volume_tolerance
                << " m^3), material-state residual=" << last_material_state_residual
                << " (tolerance=" << state_relative_tolerance << "), gas-state residual=" << last_gas_state_residual
                << " (tolerance=" << state_relative_tolerance
                << "), thermal-state residual=" << last_thermal_state_residual << " (tolerance=1).";
        throw std::runtime_error(message.str());
    }

    /** @brief Eagerly roll back every accepted ledger, then rethrow the trial failure. */
    [[noreturn]] void reject(std::exception_ptr failure)
    {
        std::string reason = "unknown planarALE transaction failure";
        try
        {
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
        catch (const std::exception& error)
        {
            reason = error.what();
        }
        catch (...)
        {
        }
        restore_accepted();
        coupler.d_planar_ale_diagnostics.rejected_transactions = accepted_ale_diagnostics.rejected_transactions + 1;
        coupler.d_planar_ale_diagnostics.last_rejection_reason = std::move(reason);
        std::rethrow_exception(failure);
    }
};

/** @brief Initialize a planar-ALE step, then delegate its rollback-safe trial schedule. */
template<TpetraTypePack Pack> void MultiphysicsSolverCoupler<Pack>::step_planar_ale()
{
    const auto timing = d_solver->d_solver_timings.scope(SolverPhase::ALE);
    d_ale_temperature_density = nullptr;
    if (d_solver->d_step_index == 0)
    {
        d_solver->temperature().sync_ghosts();
    }
    if (d_solver->d_physical_model_enabled)
    {
        refresh_physical_models();
    }
    initialize_free_surface_if_needed(true, d_solver->d_physical_model_enabled);
    initialize_planar_ale_if_needed();
    if (!d_ale_motion || !d_ale_boundary || !d_volume_continuity_model || !d_mesh_relative_face_flux ||
        !d_bubble_slip_volume_flux || !d_mesh_volume_rate || !d_continuity_residual || !d_ale_old_density ||
        !d_ale_old_heat_capacity)
    {
        throw std::logic_error("planarALE did not initialize its geometry, flux, and volume-source owners.");
    }

    PlanarALEStep transaction(*this);
    transaction.run();
}

} // namespace SimpleFluid
