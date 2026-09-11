/** Per-step nonlinear work and physical acceptance diagnostics. */
#pragma once

#include "solvers/NoxNonlinearSolver.hh"

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace SimpleFluid::Verification
{
class NonlinearSolverHistory
{
public:
    explicit NonlinearSolverHistory(const std::filesystem::path& output)
    {
        std::filesystem::create_directories(output);
        d_stream.exceptions(std::ios::badbit | std::ios::failbit);
        d_stream.open(output / "nonlinear_solver_statistics.csv");
        d_stream << std::setprecision(17)
                 << "step,time_s,converged,nonlinear_iterations,linear_solves,krylov_iterations,"
                    "residual_evaluations,line_search_rejections,scaled_residual,achieved_linear_tolerance,"
                    "continuity_max_m3_s,total_seconds,residual_seconds,linearization_seconds,linear_solve_seconds,"
                    "native_workspace_builds,native_workspace_reuses,native_geometry_builds,native_operator_builds,"
                    "native_graph_reuses,native_schur_builds,native_preconditioner_builds,"
                    "native_preconditioner_refreshes,native_setup_seconds\n";
    }

    void write(int step, double time, const NonlinearSolveResult& result, double continuity_max)
    {
        d_stream << step << ',' << time << ',' << result.converged << ',' << result.nonlinear_iterations << ','
                 << result.linear_solves << ',' << result.krylov_iterations << ',' << result.residual_evaluations << ','
                 << result.line_search_rejections << ',' << result.scaled_residual << ','
                 << result.achieved_linear_tolerance << ',' << continuity_max << ',' << result.total_seconds << ','
                 << result.residual_seconds << ',' << result.linearization_seconds << ',' << result.linear_solve_seconds
                 << ',' << result.native_workspace_builds << ',' << result.native_workspace_reuses << ','
                 << result.native_geometry_builds << ',' << result.native_operator_builds << ','
                 << result.native_graph_reuses << ',' << result.native_schur_builds << ','
                 << result.native_preconditioner_builds << ',' << result.native_preconditioner_refreshes << ','
                 << result.native_setup_seconds
                 << '\n';
        d_stream.flush();
    }

private:
    std::ofstream d_stream;
};
} // namespace SimpleFluid::Verification
