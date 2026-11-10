/** @file SolverDiagnosticsOutput.hh Collective, opt-in JSON/CSV diagnostic output. */
#pragma once
#include "solvers/SolverDiagnostics.hh"
#include "equations/CollectiveValidation.hh"
#include <algorithm>
#include <fstream>
#include <limits>
#include <iomanip>
#include <sstream>

namespace SimpleFluid::diagnostic_output
{
inline std::string quote(const std::string& text)
{
    std::ostringstream out; out << '"';
    for (const unsigned char c : text)
    {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setfill('0') << std::setw(4) << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
template<class T> void optional_number(std::ostream& out, const std::optional<T>& value)
{ if (value) out << *value; else out << "null"; }
template<class T> void report(std::ostream& out, const FVM::SteadyResidualReport<T>& r)
{
    out << "{\"signed_sum\":" << r.signed_sum << ",\"volume_rms\":" << r.volume_rms
        << ",\"max_density\":" << r.max_density << ",\"scale\":" << r.scale
        << ",\"scale_kind\":\"algebraic_absolute_row_terms\",\"dimensionless\":";
    if (r.dimensionless_available) out << r.dimensionless; else out << "null";
    out << ",\"included_count\":" << r.included_count << ",\"excluded_count\":" << r.excluded_count
        << ",\"volume\":" << r.volume
        << ",\"residual_evaluation\":{\"kind\":"
        << quote(r.row_accumulation_available ? "crs_row_dot_minus_rhs" : "caller_supplied_integrated")
        << ",\"scalar_radix\":" << std::numeric_limits<T>::radix
        << ",\"scalar_digits\":" << std::numeric_limits<T>::digits
        << ",\"unit_roundoff\":";
    constexpr bool binary_ieee = std::numeric_limits<T>::is_iec559 && std::numeric_limits<T>::radix == 2;
    if constexpr (binary_ieee) out << std::numeric_limits<T>::epsilon() / T{2};
    else out << "null";
    out << ",\"round_to_nearest\":" << (r.evaluation_round_to_nearest ? "true" : "false")
        << ",\"arithmetic_assumptions\":{\"iec559_binary\":" << (binary_ieee ? "true" : "false")
        << ",\"fast_math_disabled\":"
        << (FVM::steady_diagnostic_detail::standard_arithmetic_compilation ? "true" : "false")
        << ",\"underflow_checks\":\"normal_or_zero_intermediates\"}}}";
}
template<class T> void optional_report(std::ostream& out, const std::optional<FVM::SteadyResidualReport<T>>& r)
{ if (r) report(out,*r); else out << "null"; }

// Cell scales use residual-density units. Empty arrays leave the CSV field blank.
template<class T> void algebraic_scale_density(std::ostream& out, const FVM::SteadyResidualReport<T>& r,
    size_t cell, T volume)
{
    if (!r.algebraic_scale.empty())
        out << r.algebraic_scale.at(cell) / volume;
}

// These raw arrays have integrated equation units, independent of CSV cell volume.
// Unavailable row metadata stays blank. Eligibility is serialized as the Boolean 0 or 1.
template<class T> void integrated_cell_values(std::ostream& out, const FVM::SteadyResidualReport<T>& r,
    std::size_t cell)
{
    out << ',';
    if (!r.integrated.empty()) out << r.integrated.at(cell);
    out << ',';
    if (!r.algebraic_scale.empty()) out << r.algebraic_scale.at(cell);
    out << ',';
    if (r.row_accumulation_available && !r.accumulation_terms.empty())
        out << r.accumulation_terms.at(cell);
    out << ',';
    if (r.row_accumulation_available && !r.accumulation_eligible.empty())
        out << static_cast<unsigned int>(r.accumulation_eligible.at(cell));
}

/** All ranks call with matching prefix. Each file contains owned entities only. */
template<class MeshType, class Scalar>
void write(const MeshType& mesh, const SolverDiagnosticsResult<Scalar>& result, const std::string& prefix,
    const std::string& secondary_units = "m^2/s^4")
{
    const int output_precision = std::max(17, std::numeric_limits<Scalar>::max_digits10);
    const int rank=mesh.owned_cell_map()->getComm()->getRank();
    collective_detail::collective_local_validation(mesh,"Diagnostic output",[&]
    {
        if (rank==0)
        {
            std::ofstream out(prefix+".json"); if(!out) throw std::runtime_error("Cannot write diagnostic JSON");
            out << std::setprecision(output_precision) << "{\"schema_version\":1,\"qualification\":\"not_assessed\","
                << "\"time_since_replay_s\":" << result.time << ",\"step_index\":" << result.step_index
                << ",\"pressure_coefficient_dt_s\":" << result.pressure_coefficient_time_step
                << ",\"pressure_coefficient\":" << result.pressure_coefficient
                << ",\"geometry_kind\":" << quote(result.geometry_kind)
                << ",\"transport_flux_kind\":" << quote(result.transport_flux_kind)
                << ",\"residual_density_units\":{\"momentum\":\"m/s^2\",\"k\":\"m^2/s^3\",\"secondary\":"
                << quote(result.turbulence_secondary_residual_units.empty()?secondary_units:result.turbulence_secondary_residual_units) << ",\"continuity\":\"1/s\"},\"momentum\":[";
            for(size_t i=0;i<3;++i) {if(i) out<<',';report(out,result.momentum[i]);}
            out << "],\"momentum_vector_rms\":" << result.momentum_vector_rms
                << ",\"momentum_vector_max\":" << result.momentum_vector_max << ",\"temperature\":";
            optional_report(out,result.temperature);
            out<<",\"momentum_storage\":[";
            for(size_t i=0;i<3;++i){if(i)out<<',';optional_report(out,result.momentum_storage[i]);}
            out<<"],\"momentum_total\":[";
            for(size_t i=0;i<3;++i){if(i)out<<',';optional_report(out,result.momentum_total[i]);}
            out<<"],\"temperature_storage\":";optional_report(out,result.temperature_storage);
            out<<",\"temperature_total\":";optional_report(out,result.temperature_total);
            out<<",\"k\":";optional_report(out,result.turbulent_k);
            out<<",\"secondary\":";optional_report(out,result.turbulent_secondary);
            out<<",\"k_constraints\":";optional_report(out,result.turbulent_k_constraints);
            out<<",\"secondary_constraints\":";optional_report(out,result.turbulent_secondary_constraints);
            out<<",\"stored_continuity\":";
            if(result.stored_flux_available) report(out,result.stored_continuity);else out<<"null";
            out<<",\"reconstructed_continuity\":";report(out,result.reconstructed_continuity);
            out<<",\"stored_flux_available\":"<<(result.stored_flux_available?"true":"false")
               <<",\"temperature_residual_units\":"<<quote(result.temperature_residual_units)
               <<",\"turbulence_model\":"<<quote(result.turbulence_model_name)
               <<",\"pressure_correction_last_l2\":";optional_number(out,result.correction_last_l2);
            out<<",\"pressure_correction_l2_history\":[";
            for(size_t i=0;i<result.pressure_correction_l2_history.size();++i){if(i)out<<',';out<<result.pressure_correction_l2_history[i];}
            out<<"],\"retained_nut_closure_l2\":";optional_number(out,result.retained_nut_closure_l2);
            out<<",\"retained_nut_closure_max\":";optional_number(out,result.retained_nut_closure_max);
            out<<",\"gcl_max\":";optional_number(out,result.gcl_max);
            out<<",\"mesh_volume_rate\":";optional_number(out,result.mesh_volume_rate);
            out<<",\"continuity_target_sum\":"<<result.continuity_target_sum
               <<",\"stored_boundary_volume\":";
            if(result.stored_flux_available)out<<result.stored_boundary_volume;else out<<"null";
            out<<",\"reconstructed_boundary_volume\":"<<result.reconstructed_boundary_volume
               <<",\"last_pressure_update_l2\":";optional_number(out,result.last_pressure_update_l2);
            out<<",\"net_pressure_change_from_history_origin_l2\":";optional_number(out,result.accumulated_pressure_update_l2);
            out<<",\"pressure_update_l2_history\":[";
            for(size_t i=0;i<result.pressure_update_l2_history.size();++i)
            {if(i)out<<',';out<<result.pressure_update_l2_history[i];}
            out<<"],\"boundaries\":[";
            for(size_t i=0;i<result.boundaries.size();++i)
            {
                const auto& b=result.boundaries[i];if(i)out<<',';
                out<<"{\"name\":"<<quote(b.name)<<",\"reconstructed_volume\":"<<b.reconstructed_volume
                   <<",\"reconstructed_mass\":"<<b.reconstructed_mass<<",\"stored_volume\":";
                if(result.stored_flux_available)out<<b.stored_volume;else out<<"null";
                out<<",\"stored_mass\":";if(result.stored_flux_available)out<<b.stored_mass;else out<<"null";out<<'}';
            }
            out<<"],\"unavailable\":[";
            for(size_t i=0;i<result.unavailable.size();++i){if(i)out<<',';out<<quote(result.unavailable[i]);}
            out<<"]}\n";out.flush();if(!out)throw std::runtime_error("Diagnostic JSON write failed");
        }
        if (!result.cell_volume.empty())
        {
            std::ofstream out(prefix+"_cells_rank"+std::to_string(rank)+".csv");
            if(!out)throw std::runtime_error("Cannot write diagnostic cell CSV");
            out<<std::setprecision(output_precision)<<"cell_id,x,y,z,volume,momentum_x,momentum_y,momentum_z,continuity,k,secondary"
                <<",cell_geometry_gid,momentum_x_algebraic_scale_density,momentum_y_algebraic_scale_density"
                <<",momentum_z_algebraic_scale_density,continuity_algebraic_scale_density"
                <<",k_algebraic_scale_density,secondary_algebraic_scale_density";
            for (const auto* component : {"momentum_x", "momentum_y", "momentum_z", "continuity", "k", "secondary"})
                out << ',' << component << "_integrated_residual"
                    << ',' << component << "_integrated_algebraic_scale"
                    << ',' << component << "_accumulation_terms"
                    << ',' << component << "_accumulation_eligible";
            out << '\n';
            for(size_t i=0;i<result.cell_volume.size();++i)
            {
                const auto lid=static_cast<typename MeshType::local_ordinal_type>(i);
                const auto c=mesh.cell_centroid(lid);
                out<<mesh.owned_cell_map()->getGlobalElement(lid)<<','<<c.x<<','<<c.y<<','<<c.z<<','<<result.cell_volume[i];
                for(const auto& m:result.momentum)out<<','<<m.density.at(i);
                out<<','<<result.reconstructed_continuity.density.at(i)<<',';
                if(result.turbulent_k)out<<result.turbulent_k->density.at(i);
                out<<',';if(result.turbulent_secondary)out<<result.turbulent_secondary->density.at(i);
                out<<',';
                // Geometry identity survives repartitioning. cell_id stores the algebraic map ID.
                if constexpr (requires { mesh.cell_geometry_global_id(lid); })
                    out<<mesh.cell_geometry_global_id(lid);
                else if constexpr (requires { mesh.mesh_gid_to_tpetra_gid(mesh.cell_global_id(lid)); })
                    out<<mesh.cell_global_id(lid); // Legacy Mesh stores geometry IDs separately from Tpetra IDs.
                for(const auto& m:result.momentum)
                {out<<',';algebraic_scale_density(out,m,i,result.cell_volume[i]);}
                out<<',';algebraic_scale_density(out,result.reconstructed_continuity,i,result.cell_volume[i]);
                out<<',';if(result.turbulent_k)algebraic_scale_density(out,*result.turbulent_k,i,result.cell_volume[i]);
                out<<',';if(result.turbulent_secondary)algebraic_scale_density(out,*result.turbulent_secondary,i,result.cell_volume[i]);
                for (const auto& m : result.momentum) integrated_cell_values(out,m,i);
                integrated_cell_values(out,result.reconstructed_continuity,i);
                if (result.turbulent_k) integrated_cell_values(out,*result.turbulent_k,i);
                else out << ",,,,";
                if (result.turbulent_secondary) integrated_cell_values(out,*result.turbulent_secondary,i);
                else out << ",,,,";
                out<<'\n';
            }
            out.flush();if(!out)throw std::runtime_error("Diagnostic cell CSV write failed");
        }
        if (!result.reconstructed_face_flux.empty())
        {
            std::ofstream out(prefix+"_faces_rank"+std::to_string(rank)+".csv");
            if(!out)throw std::runtime_error("Cannot write diagnostic face CSV");
            out<<std::setprecision(output_precision)<<"face_id,x,y,z,area_m2,area_x_m2,area_y_m2,area_z_m2,velocity_flux_m3_s,pressure_flux_m3_s,reconstructed_flux_m3_s,stored_flux_m3_s,relative_flux_m3_s\n";
            for(size_t i=0;i<result.reconstructed_face_flux.size();++i)
            {
                const auto lid=static_cast<typename MeshType::local_ordinal_type>(i);
                const auto center=mesh.face_centroid(lid), area=mesh.face_area_vector(lid);
                out<<mesh.owned_face_map()->getGlobalElement(lid)
                   <<','<<center.x<<','<<center.y<<','<<center.z<<','<<mesh.face_area(lid)
                   <<','<<area.x<<','<<area.y<<','<<area.z<<','<<result.velocity_face_flux.at(i)<<','<<result.pressure_face_flux.at(i)
                   <<','<<result.reconstructed_face_flux.at(i)<<',';
                if(result.stored_flux_available)out<<result.stored_face_flux.at(i);out<<',';
                if(!result.relative_face_flux.empty())out<<result.relative_face_flux.at(i);out<<'\n';
            }
            out.flush();if(!out)throw std::runtime_error("Diagnostic face CSV write failed");
        }
    });
}
} // namespace SimpleFluid::diagnostic_output
