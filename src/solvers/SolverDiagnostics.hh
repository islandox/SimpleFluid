/** @file SolverDiagnostics.hh Independent final-state equation evidence. */
#pragma once
#include "FVM/SteadyDiagnostics.hh"
#include <array>
#include <optional>
#include <string>
#include <vector>
namespace SimpleFluid {
struct SolverDiagnosticsOptions {
    bool include_cell_data = false;
    bool include_face_data = false;
    std::optional<double> pressure_coefficient_time_step;
};
template<class Scalar> struct DiagnosticBoundaryBalance {
    std::string name;
    Scalar stored_volume{}, reconstructed_volume{}, stored_mass{}, reconstructed_mass{};
};
template<class Scalar> struct SolverDiagnosticsResult {
    using report_type = FVM::SteadyResidualReport<Scalar>;
    Scalar time{}, pressure_coefficient_time_step{}, pressure_coefficient{};
    int step_index{};
    bool stored_flux_available = false;
    std::string temperature_residual_units, turbulence_secondary_residual_units, turbulence_model_name;
    std::optional<Scalar> correction_last_l2, retained_nut_closure_l2, retained_nut_closure_max;
    std::optional<Scalar> gcl_max, mesh_volume_rate;
    std::vector<Scalar> relative_face_flux;

    std::string transport_flux_kind;
    std::string geometry_kind = "fixed_geometry";
    std::array<report_type,3> momentum;
    Scalar momentum_vector_rms{}, momentum_vector_max{};
    std::optional<report_type> temperature, turbulent_k, turbulent_secondary;
    std::array<std::optional<report_type>,3> momentum_storage, momentum_total;
    std::optional<report_type> temperature_storage, temperature_total;
    std::optional<report_type> turbulent_k_constraints, turbulent_secondary_constraints;
    report_type stored_continuity, reconstructed_continuity;
    std::vector<DiagnosticBoundaryBalance<Scalar>> boundaries;
    Scalar continuity_target_sum{}, stored_boundary_volume{}, reconstructed_boundary_volume{};
    std::optional<Scalar> last_pressure_update_l2, accumulated_pressure_update_l2;
    std::vector<Scalar> pressure_update_l2_history;
    std::vector<Scalar> pressure_correction_l2_history;
    std::vector<Scalar> stored_face_flux, reconstructed_face_flux, velocity_face_flux, pressure_face_flux;
    std::vector<Scalar> cell_volume;
    std::vector<std::string> unavailable;
};
}
