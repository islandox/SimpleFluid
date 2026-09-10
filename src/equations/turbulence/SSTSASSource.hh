/** @file SSTSASSource.hh @brief Local SAS omega source for the existing SST-1994 parent. */
#pragma once

#include "equations/turbulence/SSTKOmegaEquation.hh"

#include <array>
#include <limits>

namespace SimpleFluid
{
/** Dimensionless controls; the timestep passed to evaluate must be physical. */
struct SSTSASOptions
{
    bool enabled = true;
    bool diagnostics = false;
    real_t zeta2 = 3.51;
    real_t sigma_phi = 2.0 / 3.0;
    real_t c = 2.0;
    real_t cs = 0.11;
    real_t delta_multiplier = 1.0;
    bool time_limiter = true;
    real_t cap_time_fraction = 0.1;
};

/** Supplied invariants, independent of mesh operations and FV integration. */
struct SSTSASInputs
{
    KOmegaState state;
    real_t strain_squared = 0; ///< 2 S_ij S_ij, in 1/s2.
    real_t velocity_laplacian_magnitude = 0; ///< |laplacian(U)|, in 1/(m s).
    real_t grad_k_magnitude = 0; ///< |grad(k)|, in m/s2.
    real_t grad_omega_magnitude = 0; ///< |grad(omega)|, in 1/(m s).
    real_t delta = 1; ///< Filter width in m.
    real_t time_step = 1; ///< Physical timestep in s.
    real_t beta_star = 0.09;
    real_t beta = 0.075;
    real_t gamma = 0.5531666666666667;
    real_t kappa = 0.41; ///< The active parent coefficient, also coordinated with walls.
};

struct SSTSASResult
{
    real_t L{}, Lvk_flow{}, Lvk_grid{}, Lvk{}, delta{};
    real_t production{}, damping{}, Q_raw{}, Q_applied{};
    bool grid_limiter_active{}, time_limiter_active{};
    std::array<real_t, 11> values() const
    {
        return {L, Lvk_flow, Lvk_grid, Lvk, delta, production, damping,
                Q_raw, Q_applied, real_t(grid_limiter_active), real_t(time_limiter_active)};
    }
};

/** Global owned-cell statistics for the source assembled in the accepted step. */
struct SSTSASStatistics
{
    bool valid = false; ///< False before an accepted step or after a field-only restart.
    real_t active_cell_fraction{}, cap_active_cell_fraction{}; ///< Strict Q>0 and Qraw>cap.
    real_t min_source{}, max_source{}; ///< Applied source, in 1/s2.
    real_t volume_mean_source{}, volume_integrated_source{}; ///< 1/s2 and m3/s2.
    real_t active_volume_fraction{}, cap_active_volume_fraction{};
};

/**
 * Algebra reference: OpenFOAM Foundation v13 kOmegaSSTSAS::Qsas.
 * No OpenFOAM code or parent closure is imported. See docs/modeling/sst_sas.md.
 */
class SSTSASSource
{
public:
    explicit SSTSASSource(SSTSASOptions options = {}) : d_options(options)
    {
        for (const auto value : {options.zeta2, options.sigma_phi, options.cs,
                                options.delta_multiplier, options.cap_time_fraction})
            turbulence_detail::require_positive(value, "SAS coefficient");
        turbulence_detail::require_non_negative(options.c, "SAS damping coefficient");
    }

    real_t filter_width(real_t volume) const
    {
        turbulence_detail::require_positive(volume, "SAS cell volume");
        return checked(static_cast<long double>(d_options.delta_multiplier) * std::cbrt(volume));
    }

    SSTSASResult evaluate(const SSTSASInputs& in) const
    {
        turbulence_detail::validate(in.state);
        for (const auto value : {in.delta, in.time_step, in.beta_star, in.beta, in.gamma, in.kappa})
            turbulence_detail::require_positive(value, "SAS positive input");
        for (const auto value : {in.strain_squared, in.velocity_laplacian_magnitude,
                                in.grad_k_magnitude, in.grad_omega_magnitude})
            turbulence_detail::require_non_negative(value, "SAS invariant");
        const long double b = static_cast<long double>(in.beta) / in.beta_star - in.gamma;
        if (!std::isfinite(b) || b <= 0)
            throw std::invalid_argument("SAS requires beta/beta_star - gamma > 0.");

        SSTSASResult out;
        if (!d_options.enabled)
            return out;
        const long double k = in.state.k, omega = in.state.omega;
        const long double s = std::sqrt(static_cast<long double>(in.strain_squared));
        const long double h = in.velocity_laplacian_magnitude;
        const long double length = std::sqrt(k) / std::sqrt(std::sqrt(in.beta_star)) / omega;
        const long double grid = d_options.cs * static_cast<long double>(in.delta)
            * std::sqrt(static_cast<long double>(in.kappa) * d_options.zeta2 / b);
        // Analytic H=0 limit: production is exactly zero. No dimensional epsilon.
        const long double flow = h == 0 ? std::numeric_limits<long double>::infinity()
                                        : in.kappa * s / h;
        const long double lvk = std::max(flow, grid);
        // Compute S/Lvk directly. This avoids both 0/0 and an overflowing L/Lvk.
        const long double s_over_lvk = std::min(h / in.kappa, s / grid);
        const long double amplitude = length * s_over_lvk;
        const long double production = d_options.zeta2 * static_cast<long double>(in.kappa)
                                       * amplitude * amplitude;
        const long double normalized_gradient = std::max(
            static_cast<long double>(in.grad_k_magnitude) / k,
            static_cast<long double>(in.grad_omega_magnitude) / omega);
        const long double damping = (2 * static_cast<long double>(d_options.c) / d_options.sigma_phi)
                                    * k * normalized_gradient * normalized_gradient;
        const long double raw = std::max(production - damping, 0.0L);
        const long double cap = omega / d_options.cap_time_fraction / in.time_step;
        out.L = checked(length);
        // Largest finite real is the documented output encoding for an unbounded length.
        out.Lvk_flow = display_length(flow);
        out.Lvk_grid = checked(grid);
        out.Lvk = display_length(lvk);
        out.delta = in.delta;
        out.production = checked(production);
        out.damping = checked(damping);
        out.Q_raw = checked(raw);
        out.Q_applied = checked(d_options.time_limiter ? std::min(raw, cap) : raw);
        out.grid_limiter_active = grid > flow;
        out.time_limiter_active = d_options.time_limiter && raw > cap;
        return out;
    }

private:
    static real_t checked(long double value)
    {
        if (!std::isfinite(value) || value > std::numeric_limits<real_t>::max())
            throw std::overflow_error("SAS result is not representable.");
        return static_cast<real_t>(value);
    }
    static real_t display_length(long double value)
    {
        return static_cast<real_t>(std::min(value, static_cast<long double>(std::numeric_limits<real_t>::max())));
    }
    SSTSASOptions d_options;
};
} // namespace SimpleFluid
