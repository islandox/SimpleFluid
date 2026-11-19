#include "solvers/details/LowMachQbmmExplicitTransport.hh"

#include "equations/CollectiveValidation.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace SimpleFluid::Reacting::qbmm_explicit_detail
{
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

void Product::multiply(double value)
{
    require(std::isfinite(value), "Nonfinite QBMM kinetic packet operand.");
    int shift;
    fraction *= std::frexp(value, &shift);
    exponent += shift;
}

void Product::divide(double positive)
{
    int shift;
    require(std::isfinite(positive) && positive > 0, "Invalid QBMM kinetic packet divisor.");
    fraction /= std::frexp(positive, &shift);
    exponent -= shift;
}

FaceSum Product::sum() const
{
    FaceSum result;
    result.add_scaled_value(fraction, exponent);
    return result;
}

void Product::add_weighted(FaceSum& result, const FaceSum& value) const
{
    result.add_scaled_weighted(value, fraction, exponent);
}

Product product(std::initializer_list<double> numerator, std::initializer_list<double> denominator)
{
    Product result;
    for (const auto value : numerator)
        result.multiply(value);
    for (const auto value : denominator)
        result.divide(value);
    return result;
}

double finite_rate(double value, const char* message)
{
    require(std::isfinite(value), message);
    return value;
}

double relative_difference(const FaceSum& difference, const FaceSum& magnitude)
{
    if (!difference.sign())
        return 0;
    if (magnitude.sign() <= 0)
        return std::numeric_limits<double>::infinity();
    return std::abs(difference.rounded(magnitude));
}

void add_magnitude(FaceSum& sum, const FaceSum& operand)
{
    sum.add_weighted(operand, operand.sign() < 0 ? -1. : 1.);
}

double represented(const FaceSum& amount, double divisor)
{
    const auto value = finite_rate(amount.rounded(divisor), "QBMM kinetic transfer overflows binary64.");
    require(value != 0 || !amount.sign(), "QBMM kinetic transfer underflows binary64.");
    if (std::fpclassify(value) == FP_SUBNORMAL)
    {
        FaceSum error = amount;
        error.add_product(-value, divisor);
        FaceSum magnitude = amount;
        if (amount.sign() < 0)
        {
            magnitude = {};
            magnitude.add_weighted(amount, -1.);
        }
        require(relative_difference(error, magnitude) <= eps, "QBMM kinetic transfer loses relative precision.");
    }
    return value;
}

double represented(const Product& amount)
{
    return represented(amount.sum());
}

double rounded_rate(const FaceSum& amount, double duration)
{
    return finite_rate(amount.rounded(duration), "QBMM kinetic face receipt overflows binary64.");
}

static double mass(const PhaseThermalState& phase)
{
    FaceSum sum;
    const auto& species = phase.thermodynamics->catalog().species();
    for (size_t s = 0; s < phase.moles.size(); ++s)
        sum.add_product(phase.moles[s], species[s].molarMass);
    return represented(sum);
}

StageThermal thermal_trace(PhaseThermalState& phase, bool recover_arrival)
{
    StageThermal result;
    result.material_mass = mass(phase);
    if (result.material_mass == 0)
        return result;
    if (recover_arrival)
        phase.temperature = recover_phase_temperatures(
            std::span(&phase, 1), phase.internal_energy, PhaseThermalMode::SeparatePhaseTemperatures)
                                .phases[0]
                                .temperature;
    auto composition = phase.moles;
    const double scale = *std::ranges::max_element(composition);
    for (auto& value : composition)
        value /= scale;
    const double sum = std::accumulate(composition.begin(), composition.end(), 0.);
    for (auto& value : composition)
        value /= sum;
    const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, composition);
    result.density = point.density;
    result.volume = represented(product({result.material_mass}, {point.density}));
    result.caloric_specific = represented(product({phase.temperature, point.internal_energy_temperature}));
    require(result.caloric_specific >= 0, "QBMM kinetic caloric transport characteristic is negative.");
    return result;
}

double exact_interval_substep(double remaining, double limit)
{
    if (limit >= remaining * .5)
        return limit;
    // Both remaining endpoints are within a factor of two, so their
    // difference is an exact binary64 duration. Round that duration down
    // to honor the rate cap and make the full interval telescope exactly.
    double next = remaining - limit;
    double h = remaining - next;
    if (h > limit)
    {
        next = std::nextafter(next, remaining);
        h = remaining - next;
    }
    return h;
}

struct SIMPLEFLUID_SOLVERS_LOCAL ClosedBalance
{
    FaceSum change, magnitude;
    double scale = 0;

    void add(double before, double after)
    {
        change.add_product(after, 1.);
        change.add_product(-before, 1.);
        magnitude.add_product(std::abs(before), 1.);
        magnitude.add_product(std::abs(after), 1.);
        scale = std::max({scale, std::abs(before), std::abs(after)});
    }
};

static Flow::QbmmDirectionalFlux direction_receipt(const FaceInterval& interval, size_t side, double dt)
{
    Flow::QbmmDirectionalFlux result;
    const auto& channels = interval.direction[side];
    const auto& momentum = interval.momentum[side];
    result.species_molar_flux.resize(channels.size() - FirstSpecies);
    for (size_t s = 0; s < result.species_molar_flux.size(); ++s)
        result.species_molar_flux[s] = represented(channels[FirstSpecies + s], dt);
    for (size_t k = 0; k < result.moment_flux.size(); ++k)
        result.moment_flux[k] = represented(channels[k], dt);
    result.integrated_count = represented(channels[H0]);
    result.integrated_mass = rounded_rate(momentum[Mass]);
    result.mass_flux = rounded_rate(momentum[Mass], dt);
    result.internal_energy_flux = represented(channels[U], dt);
    result.material_volume_flux = represented(channels[W], dt);
    for (size_t axis = 0; axis < dimensions; ++axis)
    {
        result.carrier_momentum_flux.component(axis) = rounded_rate(momentum[CarrierMomentum + axis], dt);
        result.drift_momentum_flux.component(axis) = rounded_rate(momentum[DriftMomentum + axis], dt);
    }
    return result;
}

Flow::QbmmFaceFlux face_receipt(const FaceInterval& interval, double dt, double carrier_flux)
{
    Flow::QbmmFaceFlux result;
    result.kinetic.emplace();
    auto& kinetic = *result.kinetic;
    kinetic.forward = direction_receipt(interval, 0, dt);
    kinetic.backward = direction_receipt(interval, 1, dt);
    const auto net = [&](const auto& streams, size_t channel)
    {
        FaceSum total = streams[0][channel];
        total.add_weighted(streams[1][channel], 1.);
        return rounded_rate(total, dt);
    };
    result.species_molar_flux.resize(kinetic.forward.species_molar_flux.size());
    for (size_t s = 0; s < result.species_molar_flux.size(); ++s)
        result.species_molar_flux[s] = net(interval.direction, FirstSpecies + s);
    result.population_number_flux = net(interval.direction, H0);
    result.shape_flux = {net(interval.direction, H2), net(interval.direction, H3)};
    result.internal_energy_flux = net(interval.direction, U);
    result.material_volume_flux = net(interval.direction, W);
    result.derived_mass_flux = net(interval.momentum, Mass);
    for (size_t axis = 0; axis < dimensions; ++axis)
        result.derived_drift_momentum_flux.component(axis) = net(interval.momentum, DriftMomentum + axis);

    FaceSum physical_defect, physical_scale;
    physical_defect.add_weighted(interval.theta, carrier_flux);
    physical_defect.add_weighted(interval.slip, 1.);
    physical_defect.add_weighted(interval.direction[0][W], -1.);
    physical_defect.add_weighted(interval.direction[1][W], -1.);
    physical_scale.add_weighted(interval.theta, std::abs(carrier_flux));
    physical_scale.add_weighted(interval.slip, interval.slip.sign() < 0 ? -1. : 1.);
    physical_scale.add_weighted(interval.direction[0][W], 1.);
    physical_scale.add_weighted(interval.direction[1][W], -1.);
    require(relative_difference(physical_defect, physical_scale) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM kinetic pressure coefficients disagree with the directly transported volume packet.");

    kinetic.theta = rounded_rate(interval.theta, dt);
    require(kinetic.theta >= 0, "QBMM kinetic pressure coefficient is negative.");
    kinetic.slip_volume_flux = rounded_rate(interval.slip, dt);
    // Coefficient rounding can lose material flux near the subnormal floor or
    // when two normal operands almost cancel. Use a separate representation
    // term at the frozen carrier flux, leaving physical slip independently recorded.
    const auto imprecisely_rounded = [](double rounded, const FaceSum& exact)
    { return std::fpclassify(rounded) == FP_SUBNORMAL || (rounded == 0 && exact.sign()); };
    FaceSum residual, direct_magnitude;
    residual.add_product(result.material_volume_flux, 1.);
    residual.add_product(-kinetic.theta, carrier_flux);
    residual.add_product(-kinetic.slip_volume_flux, 1.);
    direct_magnitude.add_product(kinetic.forward.material_volume_flux, 1.);
    direct_magnitude.add_product(-kinetic.backward.material_volume_flux, 1.);
    if (imprecisely_rounded(kinetic.theta, interval.theta) ||
        imprecisely_rounded(kinetic.slip_volume_flux, interval.slip) ||
        relative_difference(residual, direct_magnitude) > 2 * qbmm_reconstruction_tolerance)
    {
        kinetic.pressure_rounding_flux = rounded_rate(residual);

        FaceSum theta_error = interval.theta, slip_error = interval.slip, packet_error = interval.direction[0][W];
        theta_error.add_product(-kinetic.theta, dt);
        slip_error.add_product(-kinetic.slip_volume_flux, dt);
        packet_error.add_weighted(interval.direction[1][W], 1.);
        packet_error.add_product(-result.material_volume_flux, dt);
        FaceSum rounding_bound;
        rounding_bound.add_weighted(
            theta_error, theta_error.sign() < 0 ? -std::abs(carrier_flux) : std::abs(carrier_flux));
        add_magnitude(rounding_bound, slip_error);
        add_magnitude(rounding_bound, packet_error);
        add_magnitude(rounding_bound, physical_defect);
        rounding_bound.add_product(std::numeric_limits<double>::denorm_min(), dt, .5);
        rounding_bound.add_product(-std::abs(kinetic.pressure_rounding_flux), dt, 1. - 2. * eps);
        require(rounding_bound.sign() >= 0, "QBMM kinetic pressure correction exceeds its coefficient rounding bound.");
    }

    FaceSum defect, magnitude;
    defect.add_product(kinetic.theta, carrier_flux);
    defect.add_product(kinetic.slip_volume_flux, 1.);
    defect.add_product(kinetic.pressure_rounding_flux, 1.);
    defect.add_product(-result.material_volume_flux, 1.);
    magnitude.add_product(kinetic.theta, std::abs(carrier_flux));
    magnitude.add_product(std::abs(kinetic.slip_volume_flux), 1.);
    magnitude.add_product(std::abs(kinetic.pressure_rounding_flux), 1.);
    magnitude.add_product(kinetic.forward.material_volume_flux, 1.);
    magnitude.add_product(-kinetic.backward.material_volume_flux, 1.);
    require(relative_difference(defect, magnitude) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM kinetic pressure split disagrees with the directly transported volume packet.");
    return result;
}

void validate_closed_balances(const Mesh& mesh, const Flow::Options& options, const Flow::State& accepted,
    const Flow::State& candidate, const std::vector<size_t>& phases, size_t packet_columns,
    const std::vector<std::vector<double>>& accepted_volumes, const TransportResult& result)
{
    const auto& comm = *mesh.owned_cell_map()->getComm();
    const size_t channels = phases.size() * packet_columns;
    const size_t operands = 2 * channels;
    std::vector<ClosedBalance> budget;
    std::vector<double> local_scale, scale, local, totals;
    collective_detail::collective_local_validation(mesh, "QBMM kinetic closed balance operands",
        [&]
        {
            require(operands <= std::numeric_limits<int>::max(), "QBMM kinetic balance layout overflows.");
            budget.resize(channels);
            local_scale.resize(channels);
            scale.resize(channels);
            local.resize(operands);
            totals.resize(operands);
            for (size_t a = 0; a < phases.size(); ++a)
            {
                const auto p = phases[a];
                const auto& family = *options.qbmm_families[p];
                auto* packet = budget.data() + a * packet_columns;
                for (size_t c = 0; c < candidate.cells.size(); ++c)
                {
                    const auto& old_cell = accepted.cells[c];
                    const auto& new_cell = candidate.cells[c];
                    const auto before =
                        qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, old_cell, p));
                    const auto after =
                        qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, new_cell, p));
                    for (size_t k = 0; k < before.size(); ++k)
                        packet[k].add(before[k], after[k]);
                    packet[U].add(old_cell.phases[p].internal_energy, new_cell.phases[p].internal_energy);
                    packet[W].add(accepted_volumes[c][p], result.advected_material_volumes[c][p]);
                    for (size_t s = 0; s < old_cell.phases[p].moles.size(); ++s)
                        packet[FirstSpecies + s].add(old_cell.phases[p].moles[s], new_cell.phases[p].moles[s]);
                }
            }
            std::ranges::transform(budget, local_scale.begin(), &ClosedBalance::scale);
        });
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, static_cast<int>(channels), local_scale.data(), scale.data());
    collective_detail::collective_local_validation(mesh, "QBMM kinetic scaled closed balance",
        [&]
        {
            for (size_t q = 0; q < channels; ++q)
                if (scale[q] > 0)
                {
                    local[2 * q] = budget[q].change.rounded(scale[q]);
                    local[2 * q + 1] = budget[q].magnitude.rounded(scale[q]);
                }
        });
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, static_cast<int>(operands), local.data(), totals.data());
    const double tolerance = (64. + 2. * comm.getSize()) * eps;
    for (size_t q = 0; q < channels; ++q)
        require(std::abs(totals[2 * q]) <= tolerance * totals[2 * q + 1],
            "QBMM kinetic closed-family transport budget exceeds binary64 roundoff.");
}
} // namespace SimpleFluid::Reacting::qbmm_explicit_detail
