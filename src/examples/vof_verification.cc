/**
 * @file vof_verification.cc
 * @brief Reproducible sphere and capillary-wave measurements for the VoF core.
 *
 * These scaled, source-free cases verify numerical behavior. They do not
 * qualify centimetric bubbly-flow, reacting_gas, AMR, or phase-change models.
 */

#include "solvers/VoFSolver.hh"

#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using SimpleFluid::VoFMaterial;
using SimpleFluid::VoFSolver;
using SimpleFluid::VoFSolverOptions;
using SimpleFluid::VoF::UniformGrid;
using SimpleFluid::VoF::Vector;

constexpr double pi = 3.141592653589793238462643383279502884;

struct Options
{
    std::string mode = "sphere";
    std::vector<std::size_t> resolutions{12, 24};
    double density_ratio = 10000.0;
    double end_time = 0.025;
    double wave_periods = 1.0;
    bool trace = false;
    bool help = false;
};

double positive_number(const std::string& value)
{
    std::size_t end = 0;
    const double parsed = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(parsed) || parsed <= 0.0)
        throw std::invalid_argument("Expected a positive finite number: " + value);
    return parsed;
}

Options parse(int argc, char** argv)
{
    Options options;
    for (int arg = 1; arg < argc; ++arg)
    {
        const std::string name = argv[arg];
        if (name == "--help")
        {
            std::cout << "Uniform-grid VoF numerical verification (serial).\n"
                         "  --case sphere|wave      default sphere\n"
                         "  --resolutions N,N,...   cells per unit length, default 12,24\n"
                         "  --density-ratio R       rho_l/rho_g, default 10000\n"
                         "  --end-time T            common sphere horizon, default 0.025\n"
                         "  --periods P             capillary wave periods, default 1\n"
                         "  --trace                 write wave history to stderr as CSV\n"
                         "Sphere: R=.25, rho_l=1, sigma=1, inviscid, unit slip box.\n"
                         "Wave: amplitude=.01, wavelength=1, depths=.5, sigma=1, inviscid;\n"
                         "      x/z periodic, y slip walls, two cells in z.\n"
                         "Summary CSV reports measured errors; solver failure returns nonzero.\n";
            options.help = true;
            return options;
        }
        if (name == "--trace")
        {
            options.trace = true;
            continue;
        }
        if (++arg == argc)
            throw std::invalid_argument("Missing value for " + name);
        const std::string value = argv[arg];
        if (name == "--case")
            options.mode = value;
        else if (name == "--density-ratio")
            options.density_ratio = positive_number(value);
        else if (name == "--end-time")
            options.end_time = positive_number(value);
        else if (name == "--periods")
            options.wave_periods = positive_number(value);
        else if (name == "--resolutions")
        {
            options.resolutions.clear();
            std::size_t begin = 0;
            while (begin < value.size())
            {
                const auto comma = value.find(',', begin);
                const auto token = value.substr(begin, comma == std::string::npos ? comma : comma - begin);
                const double n = positive_number(token);
                if (std::floor(n) != n || n < 8.0 || n > 256.0)
                    throw std::invalid_argument("Resolution must be an integer in [8,256].");
                options.resolutions.push_back(static_cast<std::size_t>(n));
                if (comma == std::string::npos)
                    break;
                begin = comma + 1;
                if (begin == value.size())
                    throw std::invalid_argument("Missing resolution after comma.");
            }
            if (options.resolutions.empty())
                throw std::invalid_argument("Empty resolution list.");
        }
        else
            throw std::invalid_argument("Unknown argument: " + name);
    }
    if (options.mode != "sphere" && options.mode != "wave")
        throw std::invalid_argument("--case must be sphere or wave.");
    return options;
}

UniformGrid grid_for(std::size_t n)
{
    UniformGrid grid;
    grid.cells = {n, n, n};
    grid.spacing = {1.0 / n, 1.0 / n, 1.0 / n};
    return grid;
}

VoFMaterial material_for(double density_ratio)
{
    VoFMaterial material;
    material.liquid_density = 1.0;
    material.gas_density = 1.0 / density_ratio;
    material.liquid_viscosity = 0.0;
    material.gas_viscosity = 0.0;
    material.surface_tension = 1.0;
    return material;
}

double volume(const UniformGrid& grid, const std::vector<double>& fraction)
{
    return grid.cell_volume() * std::accumulate(fraction.begin(), fraction.end(), 0.0);
}

struct QuadraturePoint
{
    double node;
    double weight;
};

const std::array<QuadraturePoint, 32>& sphere_quadrature()
{
    static const auto rule = []
    {
        constexpr std::size_t order = 64;
        std::array<QuadraturePoint, order / 2> result{};
        const auto polynomial = [](double x)
        {
            double previous = 1.0, current = x;
            for (std::size_t degree = 2; degree <= order; ++degree)
            {
                const double next = ((2.0 * degree - 1.0) * x * current - (degree - 1.0) * previous) / degree;
                previous = current;
                current = next;
            }
            return std::array<double, 2>{current, order * (x * current - previous) / (x * x - 1.0)};
        };
        for (std::size_t q = 0; q < result.size(); ++q)
        {
            double x = std::cos(pi * (q + 0.75) / (order + 0.5));
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                const auto [value, derivative] = polynomial(x);
                const double update = value / derivative;
                x -= update;
                if (std::abs(update) < 4.0 * std::numeric_limits<double>::epsilon())
                    break;
            }
            const double derivative = polynomial(x)[1];
            result[q] = {x, 2.0 / ((1.0 - x * x) * derivative * derivative)};
        }
        return result;
    }();
    return rule;
}

std::vector<double> sphere(const UniformGrid& grid)
{
    constexpr double radius = 0.25;
    // Exact x-chords and high-order y-z quadrature avoid a noisy sphere input
    // contaminating the second derivatives used by curvature reconstruction.
    const auto& quadrature = sphere_quadrature();
    std::vector<double> f(grid.cell_count());
    for (std::size_t c = 0; c < f.size(); ++c)
    {
        const auto x = grid.cell_center(c);
        double nearest2 = 0.0, farthest2 = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
        {
            const double d = std::abs(x[a] - 0.5);
            nearest2 += std::pow(std::max(0.0, d - 0.5 * grid.spacing[a]), 2);
            farthest2 += std::pow(d + 0.5 * grid.spacing[a], 2);
        }
        if (nearest2 >= radius * radius)
            continue;
        if (farthest2 <= radius * radius)
        {
            f[c] = 1.0;
            continue;
        }
        for (std::size_t j = 0; j < quadrature.size(); ++j)
            for (int sj : {-1, 1})
                for (std::size_t k = 0; k < quadrature.size(); ++k)
                    for (int sk : {-1, 1})
                    {
                        const double dy = x[1] - 0.5 + sj * 0.5 * grid.spacing[1] * quadrature[j].node;
                        const double dz = x[2] - 0.5 + sk * 0.5 * grid.spacing[2] * quadrature[k].node;
                        const double chord2 = radius * radius - dy * dy - dz * dz;
                        if (chord2 <= 0.0)
                            continue;
                        const double chord = std::sqrt(chord2);
                        const double length = std::max(0.0, std::min(x[0] + 0.5 * grid.spacing[0], 0.5 + chord) -
                                                                std::max(x[0] - 0.5 * grid.spacing[0], 0.5 - chord));
                        f[c] += 0.25 * quadrature[j].weight * quadrature[k].weight * length / grid.spacing[0];
                    }
    }
    return f;
}

void sphere_case(const Options& options, std::size_t n)
{
    const auto grid = grid_for(n);
    const auto material = material_for(options.density_ratio);
    VoFSolverOptions controls;
    controls.pressure_relative_tolerance = 1e-12;
    controls.divergence_tolerance = 1e-7;
    VoFSolver solver(grid, material, controls);
    const auto initial = sphere(grid);
    const double initial_volume = volume(grid, initial);
    solver.initialize(initial);
    double maximum_speed = 0.0, maximum_divergence = 0.0;
    std::size_t steps = 0;
    while (solver.time() < options.end_time - 1e-14)
    {
        const auto report = solver.step(std::min(0.5 * solver.stable_timestep(), options.end_time - solver.time()));
        maximum_divergence = std::max(maximum_divergence, report.maximum_divergence);
        maximum_speed = std::max(maximum_speed, report.maximum_velocity);
        if (++steps > 100000)
            throw std::runtime_error("Sphere exceeded 100000 steps.");
    }
    double pgas = 0.0, pliquid = 0.0;
    std::size_t ngas = 0, nliquid = 0;
    double curvature_sum = 0.0, curvature_error = 0.0, interface_weight = 0.0;
    for (std::size_t c = 0; c < grid.cell_count(); ++c)
    {
        const auto x = grid.cell_center(c);
        const double r2 = std::pow(x[0] - 0.5, 2) + std::pow(x[1] - 0.5, 2) + std::pow(x[2] - 0.5, 2);
        if (r2 < 0.15 * 0.15)
        {
            pgas += solver.pressure()[c];
            ++ngas;
        }
        if (r2 > 0.375 * 0.375)
        {
            pliquid += solver.pressure()[c];
            ++nliquid;
        }
        const double f = solver.gas_fraction()[c];
        const double weight = f * (1.0 - f);
        curvature_sum += weight * solver.curvature()[c];
        curvature_error += weight * std::pow(solver.curvature()[c] - 8.0, 2);
        interface_weight += weight;
    }
    const double jump = pgas / ngas - pliquid / nliquid;
    const double capillary_speed = std::sqrt(material.surface_tension / (material.liquid_density * 0.25));
    std::cout << "sphere," << n << ',' << options.density_ratio << ',' << solver.time() << ',' << steps << ',' << jump
              << ',' << std::abs(jump / 8.0 - 1.0) << ',' << maximum_speed << ',' << maximum_speed / capillary_speed
              << ',' << maximum_divergence << ',' << volume(grid, solver.gas_fraction()) / initial_volume - 1.0 << ','
              << initial_volume / (4.0 * pi * std::pow(0.25, 3) / 3.0) - 1.0 << ',' << curvature_sum / interface_weight
              << ',' << std::sqrt(curvature_error / interface_weight) / 8.0 << '\n'
              << std::flush;
}

std::vector<double> wave(const UniformGrid& grid)
{
    constexpr std::size_t samples = 256;
    std::vector<double> f(grid.cell_count());
    for (std::size_t c = 0; c < f.size(); ++c)
    {
        const auto x = grid.cell_center(c);
        const double bottom = x[1] - 0.5 * grid.spacing[1];
        for (std::size_t q = 0; q < samples; ++q)
        {
            const double xq = x[0] + grid.spacing[0] * ((q + 0.5) / samples - 0.5);
            f[c] += std::clamp((0.5 + 0.01 * std::cos(2.0 * pi * xq) - bottom) / grid.spacing[1], 0.0, 1.0) / samples;
        }
    }
    return f;
}

double amplitude(const UniformGrid& grid, const std::vector<double>& f)
{
    double sum = 0.0;
    for (std::size_t c = 0; c < f.size(); ++c)
        sum += f[c] * grid.spacing[1] * std::cos(2.0 * pi * grid.cell_center(c)[0]);
    return 2.0 * sum / (grid.cells[0] * grid.cells[2]);
}

void wave_case(const Options& options, std::size_t n)
{
    auto grid = grid_for(n);
    grid.cells[2] = 2;
    grid.periodic = {true, false, true};
    const auto material = material_for(options.density_ratio);
    VoFSolver solver(grid, material);
    solver.initialize(wave(grid));
    const double initial_volume = volume(grid, solver.gas_fraction());
    const double initial_amplitude = amplitude(grid, solver.gas_fraction());
    const double k = 2.0 * pi;
    const double omega = std::sqrt(material.surface_tension * std::pow(k, 3) /
                                   ((material.liquid_density + material.gas_density) / std::tanh(0.5 * k)));
    const double end_time = options.wave_periods * 2.0 * pi / omega;
    double old_time = 0.0, old_amplitude = initial_amplitude;
    double first_down = -1.0, first_up = -1.0;
    double squared_error_integral = 0.0, maximum_divergence = 0.0;
    std::size_t steps = 0;
    if (options.trace)
        std::cerr << "wave," << n << ",0," << initial_amplitude << ',' << initial_amplitude << '\n';
    while (solver.time() < end_time - 1e-14)
    {
        const double dt = std::min(0.65 * solver.stable_timestep(), end_time - solver.time());
        const auto report = solver.step(dt);
        const double current = amplitude(grid, solver.gas_fraction());
        if (first_down < 0.0 && old_amplitude > 0.0 && current <= 0.0)
            first_down = old_time + dt * old_amplitude / (old_amplitude - current);
        if (first_down >= 0.0 && first_up < 0.0 && old_amplitude < 0.0 && current >= 0.0)
            first_up = old_time + dt * old_amplitude / (old_amplitude - current);
        const double exact = initial_amplitude * std::cos(omega * solver.time());
        squared_error_integral += dt * std::pow((current - exact) / initial_amplitude, 2);
        maximum_divergence = std::max(maximum_divergence, report.maximum_divergence);
        if (options.trace)
            std::cerr << "wave," << n << ',' << solver.time() << ',' << current << ',' << exact << '\n';
        old_amplitude = current;
        old_time = solver.time();
        if (++steps > 100000)
            throw std::runtime_error("Wave exceeded 100000 steps.");
    }
    const double missing = std::numeric_limits<double>::quiet_NaN();
    const double measured = first_up > first_down && first_down > 0.0 ? pi / (first_up - first_down) : missing;
    const double first_zero_frequency = first_down > 0.0 ? 0.5 * pi / first_down : missing;
    std::cout << "wave," << n << ',' << options.density_ratio << ',' << solver.time() << ',' << steps << ',' << omega
              << ',' << measured << ',' << measured / omega - 1.0 << ',' << first_zero_frequency / omega - 1.0 << ','
              << std::sqrt(squared_error_integral / solver.time()) << ',' << old_amplitude / initial_amplitude << ','
              << maximum_divergence << ',' << volume(grid, solver.gas_fraction()) / initial_volume - 1.0 << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    try
    {
        const auto options = parse(argc, argv);
        if (options.help)
            return EXIT_SUCCESS;
        if (Tpetra::getDefaultComm()->getSize() != 1)
            throw std::invalid_argument("vof_verification requires one MPI rank.");
        std::cout << std::setprecision(17);
        std::cerr << std::setprecision(17);
        if (options.mode == "sphere")
            std::cout << "case,n,density_ratio,time,steps,pressure_jump,relative_pressure_error,"
                         "maximum_speed,relative_spurious_speed,maximum_divergence,relative_volume_drift,"
                         "initial_volume_quadrature_error,mean_curvature,relative_curvature_rms_error\n";
        else
        {
            std::cout << "case,n,density_ratio,time,steps,analytical_omega,measured_omega,relative_frequency_error,"
                         "first_zero_frequency_error,relative_amplitude_rms_error,final_normalized_amplitude,"
                         "maximum_divergence,relative_volume_drift\n";
            if (options.trace)
                std::cerr << "case,n,time,measured_amplitude,analytical_amplitude\n";
        }
        std::cout << std::flush;
        for (const auto n : options.resolutions)
            if (options.mode == "sphere")
                sphere_case(options, n);
            else
                wave_case(options, n);
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vof_verification: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
