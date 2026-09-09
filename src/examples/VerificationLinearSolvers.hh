/** @file VerificationLinearSolvers.hh
 * @brief Optional linear-algebra controls for the matched verification cases.
 */
#pragma once

#include "solvers/BelosLinearSolver.hh"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string_view>

namespace SimpleFluid::Verification
{

/** @brief Override algorithms without changing the case's residual tolerances. */
struct LinearSolverControls
{
    std::optional<LinearSolverBackend> transport_backend;
    std::optional<LinearPreconditioner> transport_preconditioner;
    std::optional<LinearSolverBackend> pressure_backend;
    std::optional<LinearPreconditioner> pressure_preconditioner;

    bool parse(std::string_view argument, std::string_view value, bool has_pressure = true)
    {
        if (argument == "--transport-solver")
            transport_backend = parse_linear_solver_backend(value);
        else if (argument == "--transport-preconditioner")
            transport_preconditioner = parse_linear_preconditioner(value);
        else if (has_pressure && argument == "--pressure-solver")
            pressure_backend = parse_linear_solver_backend(value);
        else if (has_pressure && argument == "--pressure-preconditioner")
            pressure_preconditioner = parse_linear_preconditioner(value);
        else
            return false;
        return true;
    }

    void apply_transport(LinearSolverOptions& options) const
    {
        if (transport_backend)
            options.backend = *transport_backend;
        if (transport_preconditioner)
            options.preconditioner = *transport_preconditioner;
    }

    template<class Solver> void apply_flow(Solver& solver) const
    {
        auto transport = solver.linear_solver_options();
        apply_transport(transport);
        solver.set_linear_solver_options(transport);
        auto pressure = solver.pressure_linear_solver_options();
        if (pressure_backend)
            pressure.backend = *pressure_backend;
        if (pressure_preconditioner)
            pressure.preconditioner = *pressure_preconditioner;
        solver.set_pressure_linear_solver_options(pressure);
        print("momentum/temperature", transport);
        print("pressure", pressure);
    }

    template<class GasModel> void apply_gas(GasModel& model) const
    {
        auto options = model.transport_linear_solver_options();
        apply_transport(options);
        model.set_transport_linear_solver_options(options);
        print("gas", options);
    }

    static void print(std::string_view group, const LinearSolverOptions& options)
    {
        std::cout << group << " linear solver: " << to_string(options.backend) << '/'
                  << to_string(options.preconditioner) << " tolerance=" << options.tolerance
                  << " max_iterations=" << options.max_iterations << '\n';
    }
};

/** @brief Retain existing flow aggregate and gas iteration/residual diagnostics. */
class LinearSolverHistory
{
public:
    void flush() { d_stream.flush(); }
    explicit LinearSolverHistory(const std::filesystem::path& output)
    {
        std::filesystem::create_directories(output);
        d_stream.exceptions(std::ios::badbit | std::ios::failbit);
        d_stream.open(output / "linear_solver_statistics.csv");
        d_stream << std::setprecision(17)
                 << "step,time_s,flow_solves,flow_iterations,flow_max_relative_residual,"
                    "gas_solves,gas_iterations,gas_max_relative_residual\n";
    }

    template<class FlowStatistics>
    void write(int step, double time, const FlowStatistics& flow, const LinearSolveSummary& gas = {})
    {
        d_stream << step << ',' << time << ',' << flow.linear_solves << ',' << flow.krylov_iterations << ','
                 << flow.achieved_tolerance << ',' << gas.solves << ',' << gas.iterations << ','
                 << gas.achieved_tolerance << '\n';
    }

    void write_gas(int step, double time, const LinearSolveSummary& gas)
    {
        d_stream << step << ',' << time << ",0,0,0," << gas.solves << ',' << gas.iterations << ','
                 << gas.achieved_tolerance << '\n';
    }

private:
    std::ofstream d_stream;
};

} // namespace SimpleFluid::Verification
