#pragma once
// This implementation owns direct diagnostic/custom-cache pressure-kernel instantiations.
#include "FVM/details/FieldStoredPressureWeightedFaceFlux.tcc"
#include "FVM/details/DiagnosticCapture.hh"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
namespace SimpleFluid {
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::evaluate_diagnostics(const SolverDiagnosticsOptions& options) const -> SolverDiagnosticsResult<scalar_type>
{
    validate_diagnostic_options(options);
    field_type viscosity(d_mesh, "diagnostic_laminar_viscosity");
    viscosity.owned_data().putScalar(static_cast<scalar_type>(d_problem.time_options().kinematic_viscosity));
    viscosity.sync_ghosts();
    return evaluate_flow_diagnostics(options, viscosity, scalar_type{1});
}
template<TpetraTypePack Pack>
auto FluidSolver<Pack>::evaluate_flow_diagnostics(const SolverDiagnosticsOptions& options,
    const field_type& viscosity, scalar_type rho, const velocity_field_type* k_gradient,
    const FVM::MeshBoundaryCache<Pack,mesh_type>* boundary_viscosity,
    const typename momentum_equation_type::source_type& extra,
    const std::vector<scalar_type>* mesh_flux, const std::vector<scalar_type>* retained_target,
    const velocity_boundary_cache_type* retained_velocity_boundary, const BoundaryConditionSet* retained_boundaries) const -> SolverDiagnosticsResult<scalar_type>
{
    SolverDiagnosticsResult<scalar_type> result;
    result.time = time(); result.step_index = step_index();
    result.stored_flux_available = step_index() > 0;
    if(!d_diagnostic_history_enabled) {
        result.geometry_kind="current_geometry_without_history";
        result.unavailable.push_back("accepted temporal storage and GCL history capture is disabled");
    }
    result.pressure_coefficient_time_step = options.pressure_coefficient_time_step.value_or(d_problem.time_options().time_step);
    collective_detail::collective_local_validation(*d_mesh, "Diagnostic pressure coefficient", [&] {
        if (!std::isfinite(result.pressure_coefficient_time_step) || result.pressure_coefficient_time_step < scalar_type{})
            throw std::invalid_argument("Diagnostic pressure coefficient time step must be finite and nonnegative.");
    });
    collective_detail::require_uniform_value(*d_mesh, result.pressure_coefficient_time_step, "Diagnostic pressure coefficient agreement");
    collective_detail::require_uniform_value(*d_mesh, options.include_cell_data ? 1 : 0, "Diagnostic cells agreement");
    collective_detail::require_uniform_value(*d_mesh, options.include_face_data ? 1 : 0, "Diagnostic faces agreement");
    result.pressure_coefficient = result.pressure_coefficient_time_step / rho;
    velocity_field_type u(d_mesh, "diagnostic_velocity");
    field_type p(d_mesh, "diagnostic_pressure");
    u.owned_data().update(scalar_type{1}, velocity().owned_data(), scalar_type{});
    p.owned_data().update(scalar_type{1}, pressure().owned_data(), scalar_type{});
    u.sync_ghosts(); p.sync_ghosts();
    face_flux_field_type stored(d_mesh, "diagnostic_stored_flux"), reconstructed(d_mesh, "diagnostic_reconstructed_flux"),
        velocity_flux(d_mesh, "diagnostic_velocity_flux"), pressure_flux(d_mesh, "diagnostic_pressure_flux");
    stored.data().update(scalar_type{1}, pressure_corrected_face_fluxes().data(), scalar_type{}); stored.sync_ghosts();
    const auto& bc=retained_boundaries ? *retained_boundaries : d_problem.boundary_conditions();
    const auto boundaries = retained_velocity_boundary ? *retained_velocity_boundary : FVM::cache_velocity_boundary_conditions<Pack>(d_mesh,bc);
    SimpleFluid::diagnostics::arm_pressure_gradient_build_capture();
    native_face_flux_workspace_type workspace(d_mesh);
    velocity_field_type gradient(d_mesh, "diagnostic_pressure_gradient");
    FVM::cell_gradient(p, bc.pressure, gradient, workspace.gradient_cache(),
        d_problem.time_options().pressure_gradient_scheme);
    SimpleFluid::diagnostics::disarm_pressure_gradient_build_capture();
    // Opt-in frozen-operand capture of the pressure-gradient reconstruction.
    // Read-only second pass over the same cached rows; never affects the gate.
    if (SimpleFluid::diagnostics::pressure_gradient_eval_capture_enabled()
            && d_problem.time_options().pressure_gradient_scheme == FVM::CellGradientScheme::LeastSquares)
    {
        const auto& capture_configuration = SimpleFluid::diagnostics::capture_configuration();
        const auto& capture_prefix = capture_configuration.pressure_gradient_prefix;
        const auto& cache = workspace.gradient_cache();
        const auto p_owned = p.owned_read_view();
        const auto p_local = p.local_read_view();
        const auto g_values = gradient.owned_read_view();
        const int rank = d_mesh->owned_cell_map()->getComm()->getRank();
        const int size = d_mesh->owned_cell_map()->getComm()->getSize();
        std::ofstream out(capture_prefix + "_gradient_eval_np" + std::to_string(size)
                + "_rank" + std::to_string(rank) + ".csv",
            std::ios::trunc);
        out << std::hexfloat
            << "cell_gid,kind,p_i,sample_index,sample_kind,other_lid,other_gid,"
               "batch,in_batch,cond_type,cond_value,normal_distance,"
               "w_x,w_y,w_z,p_other,delta,grad_x,grad_y,grad_z\n";
        const auto boundary_condition = [&](int batch_id, size_t)
        {
            const auto it = bc.pressure.find(d_mesh->boundary_batch_name(batch_id));
            return it == bc.pressure.end() ? BoundaryCondition{} : it->second;
        };
        FVM::detail::visit_cached_gradient_rows<true>(cache,
            [&](local_ordinal_type cell_lid, const auto& geometry)
        {
            const auto gid = [&]
            {
                if constexpr (requires { d_mesh->cell_geometry_global_id(cell_lid); })
                    return static_cast<std::uint64_t>(d_mesh->cell_geometry_global_id(cell_lid));
                else
                    return static_cast<std::uint64_t>(
                        d_mesh->owned_cell_map()->getGlobalElement(cell_lid));
            }();
            if (!capture_configuration.targets_pressure_gradient_cell(gid)) return;
            std::string_view kind = "generic";
            if constexpr (requires { cache.structured_geometry(); })
                if (cache.structured_geometry()) kind = "structured";
            if constexpr (requires { cache.layered_geometry(); })
                if (cache.layered_geometry()) kind = "layered";
            const auto p_i = p_owned(cell_lid, 0);
            const auto gx = g_values(cell_lid, 0);
            const auto gy = g_values(cell_lid, 1);
            const auto gz = g_values(cell_lid, 2);
            size_t index = 0;
            for (const auto& sample : geometry.interior_samples)
            {
                const auto p_other = p_local(sample.other_lid, 0);
                const auto other_gid = [&]
                {
                    if constexpr (requires { d_mesh->cell_geometry_global_id(sample.other_lid); })
                        return static_cast<std::uint64_t>(
                            d_mesh->cell_geometry_global_id(sample.other_lid));
                    else
                        return static_cast<std::uint64_t>(
                            d_mesh->overlap_cell_map()->getGlobalElement(sample.other_lid));
                }();
                out << gid << ',' << kind << ',' << p_i << ',' << index++
                    << ",interior," << sample.other_lid << ',' << other_gid
                    << ",,,,,," << sample.weight.x << ',' << sample.weight.y << ','
                    << sample.weight.z << ',' << p_other << ','
                    << (p_other - p_i) << ',' << gx << ',' << gy << ',' << gz << '\n';
            }
            for (const auto& sample : geometry.boundary_samples)
            {
                const auto condition = boundary_condition(
                    sample.location.batch_id, sample.location.in_batch_id);
                const auto increment = condition.type == BoundaryConditionType::Neumann
                    ? static_cast<scalar_type>(condition.value)
                        * static_cast<scalar_type>(sample.normal_distance)
                    : static_cast<scalar_type>(condition.value) - p_i;
                out << gid << ',' << kind << ',' << p_i << ',' << index++
                    << ",boundary,,," << sample.location.batch_id << ','
                    << sample.location.in_batch_id << ','
                    << static_cast<int>(condition.type) << ',' << condition.value << ','
                    << sample.normal_distance << ',' << sample.weight.x << ','
                    << sample.weight.y << ',' << sample.weight.z << ',' << ','
                    << increment << ',' << gx << ',' << gy << ',' << gz << '\n';
            }
        });
    }
    auto flux_pressure_bc=bc.pressure;
    std::set<std::string> kinematic_patches;
    if(mesh_flux) for(auto& [name,condition]:flux_pressure_bc) {
        const auto velocity_type=boundaries.type_by_name.find(name);
        if(condition.type==BoundaryConditionType::Dirichlet && velocity_type!=boundaries.type_by_name.end()
            && velocity_type->second!=BoundaryConditionType::Neumann) {
            // The ALE free surface prescribes both pressure and swept normal
            // volume. Its flux is imposed by the same kinematic contract as
            // production, while the physical pressure gradient remains Dirichlet.
            condition={BoundaryConditionType::Neumann,0.0};kinematic_patches.insert(name);
        }
    }
    FVM::detail::pressure_weighted_stored_face_fluxes_impl(u,p,result.pressure_coefficient,boundaries,
        &flux_pressure_bc,workspace,&gradient,reconstructed,d_problem.time_options().pressure_gradient_scheme,
        &velocity_flux,&pressure_flux);
    if(mesh_flux) {
        for(const auto& [id,batch]:d_mesh->boundary_batches()) if(kinematic_patches.contains(d_mesh->boundary_batch_name(id)))
            for(auto f:batch.face_lids) if(d_mesh->is_owned_face(f)) {
                reconstructed.set_owned_value(f,mesh_flux->at(f));velocity_flux.set_owned_value(f,mesh_flux->at(f));pressure_flux.set_owned_value(f,scalar_type{});
            }
        reconstructed.sync_ghosts();velocity_flux.sync_ghosts();pressure_flux.sync_ghosts();
    }
    if (!result.stored_flux_available) {
        stored.data().update(scalar_type{1}, reconstructed.data(), scalar_type{}); stored.sync_ghosts();
        result.unavailable.push_back("stored accepted face flux unavailable before accepted steps; transport uses reconstructed flux");
    }
    const auto grad = gradient.owned_read_view();
    // Opt-in isolation capture of the reconstructed pressure gradient and its input.
    // Off unless diagnostics::transpose_stress_capture_enabled(). Never affects the gates.
    std::unique_ptr<std::ofstream> pressure_gradient_dump;
    if (SimpleFluid::diagnostics::transpose_stress_capture_enabled())
    {
        const auto& capture_prefix = SimpleFluid::diagnostics::capture_configuration().transpose_stress_prefix;
        const int dump_rank = d_mesh->owned_cell_map()->getComm()->getRank();
        const int dump_size = d_mesh->owned_cell_map()->getComm()->getSize();
        pressure_gradient_dump = std::make_unique<std::ofstream>(
            capture_prefix + "_pressure_gradient_np" + std::to_string(dump_size)
                + "_rank" + std::to_string(dump_rank) + ".csv", std::ios::trunc);
        if (pressure_gradient_dump->good())
            *pressure_gradient_dump << std::setprecision(17)
                                    << "cell_geometry_gid,pressure,grad_x,grad_y,grad_z\n";
    }
    if (pressure_gradient_dump)
    {
        for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
        {
            const auto lid = static_cast<local_ordinal_type>(cell);
            if constexpr (requires { d_mesh->cell_geometry_global_id(lid); })
                *pressure_gradient_dump << d_mesh->cell_geometry_global_id(lid);
            else
                *pressure_gradient_dump << d_mesh->owned_cell_map()->getGlobalElement(lid);
            *pressure_gradient_dump << ',' << p.value(lid) << ',' << grad(lid, 0) << ',' << grad(lid, 1)
                                    << ',' << grad(lid, 2) << '\n';
        }
    }
    auto source = [&](local_ordinal_type c) -> vec_type {
        vec_type a{-grad(c,0)/rho, -grad(c,1)/rho, -grad(c,2)/rho};
        if (k_gradient) {
            const auto g = k_gradient->owned_read_view();
            a.x -= scalar_type{2.0/3.0}*g(c,0); a.y -= scalar_type{2.0/3.0}*g(c,1); a.z -= scalar_type{2.0/3.0}*g(c,2);
        }
        if (extra) a = a + extra(c);
        return a;
    };
    face_flux_field_type relative(d_mesh,"diagnostic_relative_flux");
    const face_flux_field_type* transport=options.pressure_coefficient_time_step ? &reconstructed : &stored;
    result.transport_flux_kind=options.pressure_coefficient_time_step || !result.stored_flux_available ? "reconstructed" : "stored_accepted";
    if(mesh_flux) {
        for(size_t i=0;i<d_mesh->num_owned_faces();++i) relative.set_owned_value(static_cast<local_ordinal_type>(i),transport->value(static_cast<local_ordinal_type>(i))-mesh_flux->at(i));
        relative.sync_ghosts();transport=&relative;result.geometry_kind="frozen_ALE_geometry";
        if(options.include_face_data) for(size_t i=0;i<d_mesh->num_owned_faces();++i) result.relative_face_flux.push_back(relative.value(static_cast<local_ordinal_type>(i)));
    }
    momentum_equation_type equation(d_mesh);
    const auto system = equation.assemble_physical_system(u, *transport, boundaries, d_problem.time_options(),
        viscosity, rho, source, &u, boundary_viscosity, nullptr, FVM::AssemblyMode::Steady);
    std::vector<scalar_type> volumes(d_mesh->num_owned_cells());
    for (size_t i=0;i<volumes.size();++i) volumes[i] = d_mesh->cell_volume(static_cast<local_ordinal_type>(i));
    for (size_t c=0;c<3;++c) result.momentum[c] = FVM::steady_residual_report<Pack>(*system.matrix,
        *system.rhs->getVector(c), *u.owned_data().getVector(c), volumes, {}, true);
    scalar_type local_max{}, local_square{};
    for (size_t i=0;i<volumes.size();++i) {
        scalar_type square{};
        for (const auto& r:result.momentum) square += r.density[i]*r.density[i];
        local_square += square*volumes[i]; local_max = std::max(local_max, std::sqrt(square));
    }
    scalar_type global_square{};
    const auto comm = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_square, &global_square);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max, &result.momentum_vector_max);
    if (result.momentum[0].volume > scalar_type{}) result.momentum_vector_rms = std::sqrt(global_square/result.momentum[0].volume);
    auto continuity = [&](const face_flux_field_type& flux) {
        std::vector<scalar_type> residual(volumes.size()), scale(volumes.size());
        for (size_t i=0;i<volumes.size();++i) {
            const auto lid=static_cast<local_ordinal_type>(i);
            const auto target=retained_target ? retained_target->at(i) : volume_continuity_target() ? volume_continuity_target()->integrated_rate(lid) : scalar_type{};
            residual[i]=FVM::detail::stored_cell_flux_balance(*d_mesh,flux,lid)-target;
            scale[i]=std::abs(target);
            // An owned cell can include faces owned by another rank.
            for (const auto face : d_mesh->faces(lid))
                scale[i] += std::abs(flux.local_value(face));
        }
        return FVM::integrated_residual_report<scalar_type>(residual,scale,volumes,*comm,options.include_cell_data);
    };
    result.stored_continuity=continuity(stored); result.reconstructed_continuity=continuity(reconstructed);
    scalar_type local_target{};
    if(retained_target) for(auto q:*retained_target) local_target+=q;
    else if (volume_continuity_target()) for (const auto q:volume_continuity_target()->owned_integrated_rates()) local_target+=q;
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&local_target,&result.continuity_target_sum);
    // Physical boundary batches can be rank-local (including empty-row ranks).
    // Build one sorted union before entering one collective per global patch.
    std::string packed_names;
    for(const auto& [id,batch]:d_mesh->boundary_batches()) {
        packed_names+=d_mesh->boundary_batch_name(id);packed_names.push_back('\0');
    }
    const int local_name_size=static_cast<int>(packed_names.size());int maximum_name_size{};
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&local_name_size,&maximum_name_size);
    maximum_name_size=std::max(maximum_name_size,1);
    std::vector<char> local_names(static_cast<size_t>(maximum_name_size),0);
    std::copy(packed_names.begin(),packed_names.end(),local_names.begin());
    std::vector<char> all_names(static_cast<size_t>(maximum_name_size)*comm->getSize());
    Teuchos::gatherAll(*comm,maximum_name_size,local_names.data(),static_cast<int>(all_names.size()),all_names.data());
    std::set<std::string> global_names;
    for(int rank=0;rank<comm->getSize();++rank) {
        std::string name;
        for(int i=0;i<maximum_name_size;++i) {
            const char character=all_names[static_cast<size_t>(rank)*maximum_name_size+i];
            if(character) name.push_back(character);
            else if(!name.empty()) {global_names.insert(name);name.clear();}
        }
    }
    for (const auto& name:global_names) {
        DiagnosticBoundaryBalance<scalar_type> b; b.name=name;
        scalar_type local[2]{},global[2]{};
        for(const auto& [id,batch]:d_mesh->boundary_batches()) if(d_mesh->boundary_batch_name(id)==name)
            for (const auto f:batch.face_lids) if (d_mesh->is_owned_face(f)) {local[0]+=stored.value(f);local[1]+=reconstructed.value(f);}
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,2,local,global);
        b.stored_volume=global[0];b.reconstructed_volume=global[1];b.stored_mass=rho*global[0];b.reconstructed_mass=rho*global[1];
        result.stored_boundary_volume+=b.stored_volume;result.reconstructed_boundary_volume+=b.reconstructed_volume;
        result.boundaries.push_back(b);
    }
    result.pressure_update_l2_history.assign(d_pressure_update_history.begin(),
        d_pressure_update_history.begin()+std::min(d_pressure_update_history.size(),static_cast<size_t>(step_index())));
    if (!result.pressure_update_l2_history.empty()) {
        result.last_pressure_update_l2=result.pressure_update_l2_history.back();
        scalar_type local{},global{};
        for (size_t i=0;i<d_accumulated_pressure_update.size();++i) {
            const auto delta=pressure().value(static_cast<local_ordinal_type>(i))-d_accumulated_pressure_update[i];local+=delta*delta;
        }
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&local,&global);result.accumulated_pressure_update_l2=std::sqrt(global);
    } else result.unavailable.push_back("pressure update history unavailable before an accepted observed step (including replay)");
    if (options.include_face_data) for(size_t i=0;i<d_mesh->num_owned_faces();++i) {
        const auto f=static_cast<local_ordinal_type>(i);
        result.stored_face_flux.push_back(stored.value(f));result.reconstructed_face_flux.push_back(reconstructed.value(f));
        result.velocity_face_flux.push_back(velocity_flux.value(f));result.pressure_face_flux.push_back(pressure_flux.value(f));
    }
    if (options.include_cell_data) result.cell_volume=volumes;
    else for(auto& r:result.momentum) {r.integrated.clear();r.density.clear();r.algebraic_scale.clear();}
    if(step_index()>0 && d_correction_norms_by_step.size()>=static_cast<size_t>(step_index())) {
        result.pressure_correction_l2_history=d_correction_norms_by_step[static_cast<size_t>(step_index()-1)];
        if(!result.pressure_correction_l2_history.empty()) result.correction_last_l2=result.pressure_correction_l2_history.back();
    }
    if(!result.correction_last_l2) result.unavailable.push_back("pressure correction history unavailable for replay, pre-step, or coupled direct solves");
    return result;
}
}

namespace SimpleFluid {
template<TpetraTypePack Pack> void FluidSolver<Pack>::record_diagnostic_pressure_correction()
{
    if(!d_diagnostic_history_enabled) return;
    const auto values=pressure_correction().owned_read_view();
    scalar_type local{};
    for(size_t i=0;i<d_mesh->num_owned_cells();++i) local+=values(i,0)*values(i,0);
    d_pending_correction_norms.push_back(std::sqrt(global_sum(local)));
}
}

namespace SimpleFluid {
template<TpetraTypePack Pack> void FluidSolver<Pack>::validate_diagnostic_options(const SolverDiagnosticsOptions& options) const
{
    collective_detail::require_uniform_value(*d_mesh,options.include_cell_data ? 1 : 0,"Diagnostic cell selection");
    collective_detail::require_uniform_value(*d_mesh,options.include_face_data ? 1 : 0,"Diagnostic face selection");
    collective_detail::require_uniform_value(*d_mesh,options.pressure_coefficient_time_step ? 1 : 0,"Diagnostic coefficient override selection");
    const auto coefficient=options.pressure_coefficient_time_step.value_or(d_problem.time_options().time_step);
    collective_detail::collective_local_validation(*d_mesh,"Diagnostic coefficient validation",[&] {
        if(!std::isfinite(coefficient) || coefficient<0) throw std::invalid_argument("Diagnostic coefficient time step must be finite and nonnegative.");
    });
    collective_detail::require_uniform_value(*d_mesh,coefficient,"Diagnostic coefficient time step agreement");
}
}
