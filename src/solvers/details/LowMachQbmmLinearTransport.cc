#include "solvers/details/LowMachQbmmLinearTransport.hh"

#include "equations/CollectiveValidation.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Reacting::qbmm_linear_detail
{
namespace
{
using namespace qbmm_explicit_detail;
using Block = std::vector<double>;
using Vec3 = Mesh::Vec3;

struct FaceStencil
{
    LO face;
    size_t side;
    double outward;
    std::vector<double> weights;
    size_t neighbor;
    std::vector<double> center_weights;
    Vec3 offset;
};

struct CellStencil
{
    double volume = 0.;
    FaceSum outgoing_rate;
    struct Neighbor
    {
        LO cell;
        double volume;
        Vec3 displacement;
    };
    std::vector<Neighbor> neighbors;
    std::vector<FaceStencil> traces;
};

std::vector<CellStencil> cell_stencils(
    const Mesh& mesh, const Flow::Traits::scalar_face_type& flux, TraceLimiter limiter, bool tvd_mc)
{
    std::vector<CellStencil> stencils(mesh.num_owned_cells());
    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const double phi = flux.local_value(face);
        require(std::isfinite(phi) && (!mesh.is_geometry_exterior_face(face) || phi == 0),
            "QBMM linear carrier flux differs from a closed mesh.");
    }
    for (size_t c = 0; c < stencils.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        auto& stencil = stencils[c];
        const auto center = mesh.cell_centroid(cell);
        stencil.volume = mesh.cell_volume(cell);
        require(std::isfinite(stencil.volume) && stencil.volume > 0, "Invalid QBMM linear bulk volume.");

        std::vector<Vec3> displacements;
        for (const auto face : mesh.faces(cell))
        {
            if (mesh.is_geometry_exterior_face(face))
                continue;
            const auto neighbor = mesh.opposite_or_periodic_neighbor_cell(face, cell);
            require(neighbor >= 0 && static_cast<size_t>(neighbor) < mesh.num_local_cells() && neighbor != cell,
                "QBMM linear reconstruction requires the adjacent cell in its one-ring halo.");
            const double neighbor_volume = mesh.cell_volume(neighbor);
            require(std::isfinite(neighbor_volume) && neighbor_volume > 0, "Invalid QBMM linear neighbor volume.");
            const auto displacement =
                tvd_mc ? mesh.cell_center_vector(face, cell) : mesh.cell_centroid(neighbor) - center;
            displacements.push_back(displacement);
            stencil.neighbors.push_back({neighbor, neighbor_volume, displacement});
        }
        const FaceOffsetFit fit(displacements);
        size_t next_neighbor = 0;
        for (const auto face : mesh.faces(cell))
        {
            if (mesh.is_geometry_exterior_face(face))
                continue;
            const size_t neighbor = next_neighbor++;
            const double phi = flux.local_value(face);
            const size_t side = mesh.owner_cell(face) == cell ? 0 : 1;
            const double outward = side == 0 ? phi : -phi;
            require(std::isfinite(outward), "Nonfinite QBMM linear overlap carrier flux.");
            if (outward < 0 && limiter == TraceLimiter::Outgoing)
                continue;
            auto& wanted = stencil.traces.emplace_back(FaceStencil{face, side, outward, {}, neighbor, {}, {}});
            if (outward > 0.)
                stencil.outgoing_rate.add_product(outward, 1.);
            wanted.offset = tvd_mc ? mesh.face_center_vector(face, cell) : mesh.face_centroid(face) - center;
            const bool resolved = fit.weights(wanted.offset, wanted.weights);
            require(
                resolved || outward <= 0, "QBMM linear one-ring geometry cannot reproduce this outgoing face offset.");
            if (!resolved)
                wanted.weights.clear(); // A non-transporting side may use its cell-center trace.
            else if (tvd_mc && !fit.weights(displacements[neighbor], wanted.center_weights))
            {
                // The center difference itself is a one-sided projection
                // when the fit loses this direction to numerical rank.
                wanted.center_weights.assign(displacements.size(), 0.);
                wanted.center_weights[neighbor] = 1.;
            }
        }
    }
    return stencils;
}

double local_substep_limit(const std::vector<CellStencil>& stencils, double cap)
{
    double limit = std::numeric_limits<double>::infinity();
    for (const auto& stencil : stencils)
    {
        if (stencil.outgoing_rate.sign() == 0)
            continue;
        FaceSum cap_volume;
        cap_volume.add_product(cap, stencil.volume);
        double bound = cap_volume.rounded(stencil.outgoing_rate);
        if (std::isfinite(bound) && bound > 0)
        {
            cap_volume.add_weighted(stencil.outgoing_rate, -bound);
            if (cap_volume.sign() < 0)
                bound = std::nextafter(bound, 0.);
        }
        limit = std::min(limit, bound);
    }
    return limit;
}

double local_cfl(const std::vector<CellStencil>& stencils, double h, double cap)
{
    double maximum = 0;
    for (const auto& stencil : stencils)
    {
        FaceSum swept, excess;
        swept.add_weighted(stencil.outgoing_rate, h);
        excess = swept;
        excess.add_product(-cap, stencil.volume);
        double fraction = swept.rounded(stencil.volume);
        if (excess.sign() > 0)
            fraction = std::max(fraction, std::nextafter(cap, std::numeric_limits<double>::infinity()));
        maximum = std::max(maximum, fraction);
    }
    return maximum;
}

Block cell_block(const QbmmPopulationSchema& family, const Flow::CellState& stock, size_t phase)
{
    const auto& thermal = stock.phases[phase];
    const auto moments = qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, stock, phase));
    Block block(FirstSpecies + thermal.moles.size());
    std::copy(moments.begin(), moments.end(), block.begin());
    block[U] = thermal.internal_energy;
    std::ranges::copy(thermal.moles, block.begin() + FirstSpecies);
    return block;
}

FaceSum material_mass(const QbmmPopulationSchema& family, std::span<const double> moles)
{
    FaceSum mass;
    for (size_t s = 0; s < moles.size(); ++s)
        mass.add_product(moles[s], family.catalog().species()[s].molarMass);
    return mass;
}

std::span<const double> moles(const Block& block)
{
    return std::span(block).subspan(FirstSpecies);
}

void check_material_block(const QbmmPopulationSchema& family, Block& block)
{
    block[H1] = family.scaled_mass(moles(block));
    const auto moments = std::span(block).first(H3 + 1);
    require(std::ranges::all_of(moments, [](double value) { return std::isfinite(value) && value > 0; }) ||
                std::ranges::all_of(moments, [](double value) { return value == 0; }),
        "A represented QBMM linear material block must have coincident positive moments or exact absence.");
    require(std::isfinite(block[U]) && (block[H0] != 0 || block[U] == 0),
        "A represented QBMM linear vacuum must have zero energy.");
}

void check_realizable(const QbmmPopulationSchema& family, Block& block, const char* context)
{
    check_material_block(family, block);
    const auto quadrature = invert_qbmm_moments({block[H0], block[H1], block[H2], block[H3]});
    if (!quadrature.valid())
    {
        std::ostringstream message;
        message << "QBMM linear " << context << " is not realizable (" << qbmm_inversion_status_name(quadrature.status)
                << "). H0 through H3 = " << std::setprecision(std::numeric_limits<double>::max_digits10) << block[H0]
                << ", " << block[H1] << ", " << block[H2] << ", " << block[H3] << '.';
        throw std::invalid_argument(message.str());
    }
}

FaceSum density_jump(const CellStencil& stencil, size_t neighbor, double center, double other)
{
    FaceSum difference, magnitude, result;
    const double other_volume = stencil.neighbors[neighbor].volume;
    difference.add_product(stencil.volume, other);
    difference.add_product(-other_volume, center);
    magnitude.add_product(stencil.volume, other);
    magnitude.add_product(other_volume, center);
    // A constant density can differ in its last bits when each cell stores
    // its own rounded extensive amount. A zero-to-positive jump stays active.
    if (relative_difference(difference, magnitude) > 2 * eps)
        product({}, {other_volume}).add_weighted(result, difference);
    return result;
}

double unit_ratio(const FaceSum& numerator, const FaceSum& denominator)
{
    if (numerator.sign() != denominator.sign())
        return 0.;
    FaceSum left, right, excess;
    add_magnitude(left, numerator);
    add_magnitude(right, denominator);
    excess = left;
    excess.add_weighted(right, -1.);
    return excess.sign() >= 0 ? 1. : left.rounded(right);
}

double mc_factor(const FaceSum& jump, const FaceSum& projected)
{
    FaceSum upstream;
    upstream.add_weighted(projected, 2.);
    upstream.add_weighted(jump, -1.);
    if (upstream.sign() != jump.sign())
        return 0.;

    // r = upstream/jump, phi_MC = min(2r, (1+r)/2, 2).
    // Divide by the centered branch so the result scales the existing
    // physical face increment. This also avoids overflowing a large r.
    FaceSum twice_upstream, twice_jump;
    twice_upstream.add_weighted(upstream, 2.);
    twice_jump.add_weighted(jump, 2.);
    return std::min(unit_ratio(twice_upstream, projected), unit_ratio(twice_jump, projected));
}

bool limits_density(size_t channel)
{
    return channel == H0 || channel == H2 || channel == H3 || channel >= FirstSpecies;
}

struct DensityBounds
{
    struct Stock
    {
        double amount, volume;

        bool operator<(const Stock& other) const
        {
            FaceSum difference;
            difference.add_product(amount, other.volume);
            difference.add_product(-other.amount, volume);
            return difference.sign() < 0;
        }
    };
    std::vector<size_t> channels;
    std::vector<Stock> minimum, maximum;

    DensityBounds(const CellStencil& stencil, const Block& center, const std::vector<Block>& neighbors)
    {
        for (size_t q = 0; q < center.size(); ++q)
        {
            if (!limits_density(q))
                continue;
            Stock low{center[q], stencil.volume}, high = low;
            for (size_t k = 0; k < neighbors.size(); ++k)
            {
                const Stock sample{neighbors[k][q], stencil.neighbors[k].volume};
                low = std::min(low, sample);
                high = std::max(high, sample);
            }
            if (high.amount != 0.)
            {
                channels.push_back(q);
                minimum.push_back(low);
                maximum.push_back(high);
            }
        }
    }

    void check(const Block& stock, const FaceSum& volume) const
    {
        for (size_t i = 0; i < channels.size(); ++i)
        {
            const auto bounded = [&](const Stock& limit, double sign)
            {
                FaceSum bound, value;
                bound.add_weighted(volume, limit.amount);
                value.add_product(stock[channels[i]], limit.volume);
                FaceSum excess, scale;
                excess.add_weighted(value, sign);
                excess.add_weighted(bound, -sign);
                scale.add_weighted(value, 1.);
                scale.add_weighted(bound, 1.);
                excess.add_weighted(scale, -2 * eps);
                return excess.sign() <= 0;
            };
            require(bounded(minimum[i], -1.) && bounded(maximum[i], 1.),
                "QBMM TVD-MC density exceeds the donor stencil bounds.");
        }
    }
};

struct EnergyBounds
{
    struct Stock
    {
        double energy;
        FaceSum mass;
    };
    std::optional<Stock> minimum, maximum;

    static FaceSum difference(const Stock& lhs, const Stock& rhs)
    {
        FaceSum cross;
        cross.add_weighted(rhs.mass, lhs.energy);
        cross.add_weighted(lhs.mass, -rhs.energy);
        return cross;
    }

    void include(const QbmmPopulationSchema& family, const Block& block)
    {
        if (block[H0] == 0)
            return;
        const Stock stock{block[U], material_mass(family, moles(block))};
        require(stock.mass.sign() > 0, "An occupied QBMM linear stencil stock needs positive mass.");
        if (!minimum || difference(stock, *minimum).sign() < 0)
            minimum = stock;
        if (!maximum || difference(stock, *maximum).sign() > 0)
            maximum = stock;
    }

    static bool at_most(const Stock& lhs, const Stock& rhs, double relative_slack = 2 * eps)
    {
        auto excess = difference(lhs, rhs);
        if (excess.sign() <= 0)
            return true;
        // The stock representation bounds each rounding by one epsilon.
        // At C <= 1/2 the outgoing rounding contributes at most one epsilon
        // relative to a first-order retained channel, and rounding that
        // channel contributes at most one more. Measure the resulting
        // envelope against the cross operands, with no absolute floor.
        FaceSum scale;
        scale.add_weighted(rhs.mass, std::abs(lhs.energy));
        scale.add_weighted(lhs.mass, std::abs(rhs.energy));
        excess.add_weighted(scale, -relative_slack);
        return excess.sign() <= 0;
    }

    static bool indistinguishable(const Stock& lhs, const Stock& rhs)
    {
        // Independent species and energy sums can round a constant ratio
        // differently. Ignore this uncertainty when selecting the MC ratio.
        // The face and remainder energy certificates keep their own bounds.
        constexpr double slope_slack = 8 * eps;
        return at_most(lhs, rhs, slope_slack) && at_most(rhs, lhs, slope_slack);
    }

    void check(const QbmmPopulationSchema& family, const Block& block) const
    {
        if (block[H0] == 0)
            return;
        const Stock stock{block[U], material_mass(family, moles(block))};
        require(stock.mass.sign() > 0 && minimum && maximum && at_most(*minimum, stock) && at_most(stock, *maximum),
            "QBMM linear specific energy exceeds the occupied stencil bounds.");
    }
};

std::vector<double> tvd_factors(const QbmmPopulationSchema& family, const CellStencil& stencil, const Block& center,
    const std::vector<Block>& neighbors, const std::vector<std::vector<FaceSum>>& slopes)
{
    std::vector<double> factors(stencil.traces.size(), 1.);
    if (center[H0] == 0)
        return factors;

    std::vector<std::vector<FaceSum>> jumps(neighbors.size(), std::vector<FaceSum>(center.size()));
    std::vector<EnergyBounds::Stock> energy;
    std::vector<Vec3> occupied_displacements;
    std::vector<FaceSum> energy_jumps;
    energy.reserve(neighbors.size());
    const EnergyBounds::Stock center_energy{center[U], material_mass(family, moles(center))};
    for (size_t k = 0; k < neighbors.size(); ++k)
    {
        const auto& neighbor = neighbors[k];
        for (size_t q = 0; q < center.size(); ++q)
            if (limits_density(q))
                jumps[k][q] = density_jump(stencil, k, center[q], neighbor[q]);

        energy.push_back({neighbor[U], material_mass(family, moles(neighbor))});
        if (neighbor[H0] == 0)
            continue;
        // Store the specific-energy difference times a common mass scale.
        // The common scale cancels in the MC ratio. A vacuum has no sample.
        auto& jump = energy_jumps.emplace_back();
        const auto& sample = energy.back();
        if (!EnergyBounds::indistinguishable(center_energy, sample))
            product({}, {neighbor[H1]}).add_weighted(jump, EnergyBounds::difference(sample, center_energy));
        occupied_displacements.push_back(stencil.neighbors[k].displacement);
    }
    const FaceOffsetFit energy_fit(occupied_displacements);
    std::vector<double> energy_weights;
    for (size_t f = 0; f < stencil.traces.size(); ++f)
    {
        const auto& face = stencil.traces[f];
        const auto adjacent = face.neighbor;
        auto& factor = factors[f];
        if (face.center_weights.empty())
            continue;
        for (size_t q = 0; q < center.size() && factor > 0; ++q)
        {
            const auto& jump = jumps[adjacent][q];
            if (jump.sign() == 0)
                continue; // H1, U, W, and flat or excluded species cannot constrain the component pass.
            if (center[q] == 0)
            {
                factor = 0.;
                break;
            }
            FaceSum projected;
            for (size_t k = 0; k < face.center_weights.size(); ++k)
                projected.add_weighted(jumps[k][q], face.center_weights[k]);
            factor = std::min(factor, mc_factor(jump, projected));
            if (slopes[f][q].sign() != 0)
                factor = std::min(factor, unit_ratio(jump, slopes[f][q]));
            if (factor > 0 && slopes[f][q].sign() > 0)
            {
                FaceSum donor;
                donor.add_product(center[q], 1.);
                factor = std::min(factor, unit_ratio(donor, slopes[f][q]));
            }
        }
        const auto& downstream = energy[adjacent];
        if (factor == 0 || downstream.mass.sign() == 0 || EnergyBounds::indistinguishable(center_energy, downstream))
            continue;

        const auto jump = EnergyBounds::difference(downstream, center_energy);
        FaceSum local_jump, projected;
        product({}, {neighbors[adjacent][H1]}).add_weighted(local_jump, jump);
        if (energy_fit.weights(stencil.neighbors[adjacent].displacement, energy_weights))
            for (size_t k = 0; k < energy_weights.size(); ++k)
                projected.add_weighted(energy_jumps[k], energy_weights[k]);
        else
            projected = local_jump;
        const double energy_mc = mc_factor(local_jump, projected);
        factor = std::min(factor, energy_mc);
        if (factor == 0.)
            continue;

        FaceSum face_projection;
        if (energy_fit.weights(face.offset, energy_weights))
            for (size_t k = 0; k < energy_weights.size(); ++k)
                face_projection.add_weighted(energy_jumps[k], energy_weights[k]);
        else
        {
            // Only the donor-to-neighbor direction is determined by the
            // occupied stocks. Project the face offset on that direction.
            const auto& displacement = stencil.neighbors[adjacent].displacement;
            const double length = std::hypot(displacement.x, displacement.y, displacement.z);
            face_projection.add_weighted(projected, (face.offset / length).dot(displacement / length));
        }
        FaceSum limited_projection;
        limited_projection.add_weighted(face_projection, energy_mc);
        const double interpolation = unit_ratio(limited_projection, local_jump);

        // The trace transports energy and species mass, so its specific
        // energy is a quotient. Bound the quotient at the MC face value
        // (1-t)*e_center + t*e_downstream using the cross-products with both
        // stocks. Their positive masses supply the common denominator.
        const auto cross_slope = [&](const EnergyBounds::Stock& reference)
        {
            const auto base = EnergyBounds::difference(center_energy, reference);
            FaceSum slope;
            for (size_t k = 0; k < face.weights.size(); ++k)
            {
                FaceSum diff;
                diff.add_weighted(base, -1.);
                product({stencil.volume}, {stencil.neighbors[k].volume})
                    .add_weighted(diff, EnergyBounds::difference(energy[k], reference));
                slope.add_weighted(diff, face.weights[k]);
            }
            return slope;
        };
        const auto center_slope = cross_slope(center_energy);
        if (center_slope.sign() != 0 && center_slope.sign() != jump.sign())
        {
            factor = 0.;
            continue;
        }
        FaceSum against_center, against_downstream, bound, slope;
        against_center.add_weighted(center_slope, neighbors[adjacent][H1]);
        against_downstream.add_weighted(cross_slope(downstream), center[H1]);
        bound.add_weighted(jump, center[H1]);
        slope.add_weighted(against_center, 1. - interpolation);
        slope.add_weighted(against_downstream, interpolation);
        FaceSum limit;
        limit.add_weighted(bound, interpolation);
        if (slope.sign() == jump.sign())
            factor = std::min(factor, unit_ratio(limit, slope));
    }
    return factors;
}

template<class View> Block read_block(const View& view, LO row, size_t first, size_t width)
{
    Block block(width);
    for (size_t q = 0; q < width; ++q)
        block[q] = view(row, first + q);
    return block;
}

bool loses_packet_precision(const FaceSum& amount, double divisor, double rounded)
{
    if (amount.sign() == 0)
        return false;
    if (rounded == 0.)
        return true;
    if (std::fpclassify(rounded) != FP_SUBNORMAL)
        return false;
    FaceSum error = amount, magnitude;
    error.add_product(-rounded, divisor);
    add_magnitude(magnitude, amount);
    return relative_difference(error, magnitude) > eps;
}

bool loses_receipt_precision(const FaceSum& contribution, double dt)
{
    return loses_packet_precision(contribution, 1., contribution.rounded()) ||
           loses_packet_precision(contribution, dt, contribution.rounded(dt));
}

bool loses_interval_precision(const FaceSum& amount, double dt)
{
    FaceSum half;
    half.add_weighted(amount, .5);
    return loses_receipt_precision(half, dt);
}

bool loses_packet_channel(const FaceSum& amount, double divisor, double reference_volume, double dt)
{
    const double rounded = amount.rounded(divisor);
    if (loses_packet_precision(amount, divisor, rounded))
        return true;
    if (!std::isfinite(rounded))
        return false; // The regular packet certificate still rejects overflow.
    FaceSum packet;
    packet.add_product(rounded, 1.);
    const double concentration = packet.rounded(reference_volume);
    return !std::isfinite(concentration) || loses_packet_precision(packet, reference_volume, concentration) ||
           loses_interval_precision(packet, dt);
}

bool return_material_packet(const Block& trace, double h, double outward, double volume, double reference_volume,
    double dt)
{
    constexpr double count_floor = 1e-200, total_moles_floor = 1e-200, species_floor = 1e-220;
    const auto below_floor = [volume](const FaceSum& numerator, double floor)
    {
        if (numerator.sign() <= 0)
            return false;
        auto difference = numerator;
        difference.add_product(-floor, volume);
        return difference.sign() < 0;
    };
    FaceSum total_moles;
    for (size_t q = 0; q < trace.size(); ++q)
    {
        if (q == W || trace[q] == 0.)
            continue;
        FaceSum amount;
        amount.add_product(h, outward, trace[q]);
        if (loses_packet_channel(amount, volume, reference_volume, dt) ||
            (q == H0 && below_floor(amount, count_floor)) ||
            (q >= FirstSpecies && below_floor(amount, species_floor)))
            return true;
        if (q >= FirstSpecies)
            total_moles.add_weighted(amount, 1.);
    }
    return below_floor(total_moles, total_moles_floor);
}

FaceSum material_volume(const FaceSum& mass, double density)
{
    FaceSum volume;
    product({}, {density}).add_weighted(volume, mass);
    return volume;
}

Product pressure_theta_weight(double h, double stage_weight, double donor_volume, double density)
{
    return product({h, stage_weight}, {donor_volume, density});
}

bool return_physical_packet(const FaceSum& mass, const FaceSum& half_stage_theta, const StageThermal& point,
    const Vector& velocity, double reference_volume, double dt)
{
    if (loses_packet_channel(material_volume(mass, point.density), 1., reference_volume, dt) ||
        loses_interval_precision(mass, dt) || loses_receipt_precision(half_stage_theta, dt))
        return true;
    FaceSum caloric;
    caloric.add_weighted(mass, point.caloric_specific);
    if (loses_interval_precision(caloric, dt))
        return true;
    for (size_t axis = 0; axis < dimensions; ++axis)
    {
        FaceSum momentum;
        momentum.add_weighted(mass, velocity.component(axis));
        if (loses_interval_precision(momentum, dt))
            return true;
    }
    return false;
}

StageThermal trace_thermal(PhaseThermalState thermal, const Block& trace, double pressure)
{
    thermal.moles.assign(moles(trace).begin(), moles(trace).end());
    thermal.internal_energy = trace[U];
    thermal.absolute_pressure = pressure;
    const auto point = thermal_trace(thermal, true);
    require((point.material_mass != 0) == (trace[H0] != 0),
        "QBMM linear trace count and EOS material presence differ.");
    return point;
}

Block withdrawal(const QbmmPopulationSchema& family, const Block& trace, double h, double outward, double volume)
{
    Block transfer(trace.size());
    for (size_t q = 0; q < transfer.size(); ++q)
    {
        if (q == H1 || q == W)
            continue; // H1 is derived below. W is the physical EOS volume packet.
        FaceSum amount;
        amount.add_product(h, outward, trace[q]);
        transfer[q] = represented(amount, volume);
    }
    // The transfer is the certified trace times the positive factor
    // h*phi/V. Its rounded channels are balance receipts. Re-inverting
    // those independently rounded channels can move an exact singleton
    // across the shifted determinant boundary solely by rescaling it.
    check_material_block(family, transfer);
    FaceSum expected_mass, represented_mass, scale;
    expected_mass.add_product(h, outward, trace[H1]);
    represented_mass.add_product(volume, transfer[H1]);
    scale = expected_mass;
    scale.add_weighted(represented_mass, 1.);
    expected_mass.add_weighted(represented_mass, -1.);
    require(relative_difference(expected_mass, scale) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM linear species-derived H1 transfer differs from the certified face trace.");
    return transfer;
}

bool preserves_singleton(const Block& center, const std::vector<FaceSum>& slope)
{
    // A positive measure left in a singleton donor and every positive
    // withdrawal must have its original support. The only slopes tangent
    // to its moment ray are changes in the common weight. Checking this
    // before limiting avoids passing a small unsupported support change
    // through the numerical rank-one tolerance.
    for (size_t q = H2; q < center.size(); ++q)
    {
        if (q == U || q == W)
            continue;
        FaceSum channel, count, difference, magnitude;
        channel.add_weighted(slope[q], center[H0]);
        count.add_weighted(slope[H0], center[q]);
        difference = channel;
        difference.add_weighted(count, -1.);
        add_magnitude(magnitude, channel);
        add_magnitude(magnitude, count);
        if (relative_difference(difference, magnitude) > 16 * eps)
            return false;
    }
    return true;
}

struct LimitedTrace
{
    Block block;
    std::optional<StageThermal> thermal;
    bool returned = false;
};

std::vector<LimitedTrace> limit_traces(const QbmmPopulationSchema& family, const CellStencil& stencil,
    const Block& center, const std::vector<Block>& neighbors, const PhaseThermalState& thermal_guess,
    const Vector& velocity, double pressure, double h, double dt, TraceLimiter limiter, bool tvd_mc)
{
    EnergyBounds energy;
    energy.include(family, center);
    for (const auto& neighbor : neighbors)
        energy.include(family, neighbor);
    std::optional<DensityBounds> density_bounds;
    if (tvd_mc)
        density_bounds.emplace(stencil, center, neighbors);
    std::vector<LimitedTrace> traces(stencil.traces.size());
    for (auto& trace : traces)
        trace.block.resize(center.size());
    std::vector<std::vector<FaceSum>> slopes(stencil.traces.size(), std::vector<FaceSum>(center.size()));
    if (center[H0] != 0)
        for (size_t f = 0; f < stencil.traces.size(); ++f)
            for (size_t k = 0; k < stencil.traces[f].weights.size(); ++k)
                for (size_t q = 0; q < center.size(); ++q)
                {
                    if (q == H1 || q == W)
                        continue;
                    const double weight = stencil.traces[f].weights[k];
                    const double other_volume = stencil.neighbors[k].volume;
                    // All columns use the same density stencil in donor-volume
                    // units. The cross-products cancel a constant exactly.
                    FaceSum difference;
                    difference.add_product(stencil.volume, neighbors[k][q]);
                    difference.add_product(-other_volume, center[q]);
                    product({weight}, {other_volume}).add_weighted(slopes[f][q], difference);
                }

    const auto mc = tvd_mc ? tvd_factors(family, stencil, center, neighbors, slopes) : std::vector<double>{};
    FaceSum bulk_volume, retained_volume;
    bulk_volume.add_product(stencil.volume, 1.);
    retained_volume = bulk_volume;
    retained_volume.add_weighted(stencil.outgoing_rate, -h);
    require(retained_volume.sign() > 0, "QBMM linear all-face donor retention must be positive.");

    const auto reconstruct = [&](size_t face, double factor)
    {
        auto& trace = traces[face].block;
        if (tvd_mc)
            factor *= mc[face];
        for (size_t q = 0; q < trace.size(); ++q)
        {
            if (q == H1 || q == W)
                continue;
            FaceSum value;
            value.add_product(center[q], 1.);
            value.add_weighted(slopes[face][q], factor);
            trace[q] = represented(value);
        }
        check_realizable(family, trace, "face trace");
        energy.check(family, trace);
        if (density_bounds)
            density_bounds->check(trace, bulk_volume);
    };
    const auto certify_donor = [&](double factor)
    {
        auto applied_retained_volume = retained_volume;
        std::vector<FaceSum> retained(center.size());
        for (size_t q = 0; q < center.size(); ++q)
            retained[q].add_product(center[q], 1.);
        for (size_t f = 0; f < traces.size(); ++f)
        {
            const auto& face = stencil.traces[f];
            const auto outward = face.outward;
            // The forced corrector certifies every potential donor trace with the same factor.
            // A vanishing rate can change direction without changing the set of trace constraints.
            if (outward > 0. || limiter == TraceLimiter::AllIncident)
                reconstruct(f, factor);
            if (outward <= 0.)
                continue;
            auto& trace = traces[f];
            trace.thermal.reset();
            const double reference_volume = std::min(stencil.volume, stencil.neighbors[face.neighbor].volume);
            trace.returned = tvd_mc && trace.block[H0] != 0 &&
                             return_material_packet(trace.block, h, outward, stencil.volume, reference_volume, dt);
            Block transfer;
            if (!trace.returned)
            {
                transfer = withdrawal(family, trace.block, h, outward, stencil.volume);
                if (tvd_mc && trace.block[H0] != 0)
                {
                    trace.thermal = trace_thermal(thermal_guess, trace.block, pressure);
                    FaceSum half_stage_theta;
                    pressure_theta_weight(h, .5, stencil.volume, trace.thermal->density)
                        .add_weighted(half_stage_theta, material_mass(family, moles(trace.block)));
                    trace.returned = return_physical_packet(material_mass(family, moles(transfer)), half_stage_theta,
                        *trace.thermal, velocity, reference_volume, dt);
                }
            }
            if (trace.returned)
            {
                applied_retained_volume.add_product(h, outward);
                continue;
            }
            for (size_t q = 0; q < center.size(); ++q)
                retained[q].add_product(-transfer[q], 1.);
        }

        // Certify the retained stock after the packets that actually leave
        // the donor. Its extensive units avoid independently rounding
        // density channels at rank changes.
        Block remainder(center.size());
        for (size_t q = 0; q < center.size(); ++q)
            if (q != H1 && q != W)
                remainder[q] = represented(retained[q]);
        check_realizable(family, remainder, "retained donor stock");
        energy.check(family, remainder);
        if (density_bounds)
            density_bounds->check(remainder, applied_retained_volume);
    };

    const auto limit = [&](auto&& certify, double initial)
    {
        constexpr int halvings = std::numeric_limits<double>::digits;
        for (int attempt = 0; initial > 0 && attempt <= halvings; ++attempt)
        {
            const double factor = std::scalbn(initial, -attempt);
            try
            {
                certify(factor);
                return factor;
            }
            catch (const std::invalid_argument&)
            {
                // A common factor must pass every trace and the combined
                // retained block. No component or individual face is repaired.
            }
        }
        certify(0.);
        return 0.;
    };
    double initial = center[H0] != 0 ? 1. : 0.;
    if (invert_qbmm_moments({center[H0], center[H1], center[H2], center[H3]}).active_atoms == 1)
        for (size_t f = 0; f < traces.size(); ++f)
            if ((stencil.traces[f].outward > 0 || limiter == TraceLimiter::AllIncident) &&
                (!tvd_mc || mc[f] > 0) &&
                !preserves_singleton(center, slopes[f]))
                initial = 0.;
    const double factor = limit(certify_donor, initial);
    if (limiter == TraceLimiter::Outgoing)
        for (size_t f = 0; f < traces.size(); ++f)
            if (stencil.traces[f].outward == 0)
            {
                // The unforced policy permits its two tied pressure traces to
                // reduce their own slopes independently of the donor's outgoing traces.
                limit([&](double tie_factor) { reconstruct(f, tie_factor); }, factor);
            }
    return traces;
}

template<class View>
void stage_stocks(const Flow::Options& options, const Flow::State& stage, const std::vector<size_t>& phases,
    size_t width, const View& values)
{
    for (size_t c = 0; c < stage.cells.size(); ++c)
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto cell = static_cast<LO>(c);
            const auto& family = *options.qbmm_families[phases[a]];
            auto block = cell_block(family, stage.cells[c], phases[a]);
            check_realizable(family, block, "stage cell stock");
            for (size_t q = 0; q < width; ++q)
                values(cell, a * width + q) = block[q];
        }
}

struct TraceColumns
{
    static constexpr double returned_marker = -1.;
    size_t block;
    size_t density() const { return block; }
    size_t caloric() const { return block + 1; }
    size_t marker() const { return block + 2; }
    size_t volume() const { return block + 3; }
    size_t carrier() const { return block + 4; }
    size_t width() const { return carrier() + dimensions; }
};

template<class Stocks, class Traces>
void stage_traces(const Flow::Options& options, const Flow::State& guess, const std::vector<size_t>& phases,
    const std::vector<CellStencil>& stencils, const TraceColumns& columns, const Stocks& stocks, const Traces& traces,
    double h, double dt, TraceLimiter limiter, bool tvd_mc)
{
    for (size_t c = 0; c < stencils.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto& stencil = stencils[c];
        const auto& donor = guess.cells[c];
        const double pressure = guess.thermodynamic_pressure + donor.mechanical_pressure;
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            const auto center = read_block(stocks, cell, a * columns.block, columns.block);
            std::vector<Block> neighbors;
            neighbors.reserve(stencil.neighbors.size());
            for (const auto& neighbor : stencil.neighbors)
                neighbors.push_back(read_block(stocks, neighbor.cell, a * columns.block, columns.block));
            const auto limited = limit_traces(family, stencil, center, neighbors, donor.phases[phase], donor.velocity,
                pressure, h, dt, limiter, tvd_mc);
            for (size_t f = 0; f < stencil.traces.size(); ++f)
            {
                const auto& face = stencil.traces[f];
                if (face.outward < 0.)
                    continue;
                const auto first = (a * sides + face.side) * columns.width();
                const auto& trace = limited[f].block;
                if (limited[f].returned)
                {
                    traces(face.face, first + columns.marker()) = TraceColumns::returned_marker;
                    continue;
                }
                const auto point =
                    limited[f].thermal ? *limited[f].thermal : trace_thermal(donor.phases[phase], trace, pressure);
                for (size_t q = 0; q < columns.block; ++q)
                    traces(face.face, first + q) = trace[q];
                traces(face.face, first + columns.density()) = point.density;
                traces(face.face, first + columns.caloric()) = point.caloric_specific;
                traces(face.face, first + columns.marker()) = 1.;
                traces(face.face, first + columns.volume()) = stencil.volume;
                for (size_t axis = 0; axis < dimensions; ++axis)
                    traces(face.face, first + columns.carrier() + axis) = donor.velocity.component(axis);
            }
        }
    }
}

template<class Traces, class Packets>
void stage_packets(const Mesh& mesh, const Flow::Options& options, const Flow::Traits::scalar_face_type& flux,
    const std::vector<size_t>& phases, const TraceColumns& columns, const Traces& traces, const Packets& packets,
    std::vector<FaceInterval>& intervals, FaceSum& caloric, FaceSum& energy, double h, bool tvd_mc,
    unsigned long long& returned_packets)
{
    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const bool wall = mesh.is_geometry_exterior_face(face);
        const double phi = flux.local_value(face);
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto& family = *options.qbmm_families[phases[a]];
            auto& interval = intervals[f * phases.size() + a];
            for (size_t side = 0; side < sides; ++side)
            {
                const double sign = side == 0 ? 1. : -1.;
                const double outward = sign * phi;
                const bool selected = !wall && outward >= 0;
                const size_t first = (a * sides + side) * columns.width();
                const double marker = traces(face, first + columns.marker());
                const bool returned = tvd_mc && selected && outward > 0 && marker == TraceColumns::returned_marker;
                require(returned || marker == (selected ? 1. : 0.),
                    "QBMM linear face has no unique certified donor trace.");
                // The canonical owner publishes zeros for both cells and counts
                // one return, independent of which rank owns the donor.
                returned_packets += returned;
                Block transfer(columns.block);
                if (selected && !returned)
                {
                    auto trace = read_block(traces, face, first, columns.block);
                    check_realizable(family, trace, "face trace");
                    if (trace[H0] != 0)
                    {
                        const auto density = traces(face, first + columns.density());
                        require(std::isfinite(density) && density > 0, "Invalid QBMM linear face EOS density.");
                        const auto donor_volume = traces(face, first + columns.volume());
                        const auto trace_mass = material_mass(family, moles(trace));
                        pressure_theta_weight(h, outward == 0 ? .25 : .5, donor_volume, density)
                            .add_weighted(interval.theta, trace_mass);
                        if (outward > 0)
                        {
                            transfer = withdrawal(family, trace, h, outward, donor_volume);
                            const auto mass = material_mass(family, moles(transfer));
                            transfer[W] = represented(material_volume(mass, density));
                            interval.momentum[side][Mass].add_weighted(mass, sign * .5);
                            for (size_t axis = 0; axis < dimensions; ++axis)
                            {
                                FaceSum momentum;
                                momentum.add_weighted(mass, traces(face, first + columns.carrier() + axis));
                                interval.momentum[side][CarrierMomentum + axis].add_weighted(momentum, sign * .5);
                            }
                            FaceSum caloric_transfer;
                            caloric_transfer.add_weighted(mass, traces(face, first + columns.caloric()));
                            caloric.add_weighted(caloric_transfer, .5);
                            energy.add_product(.5, std::abs(transfer[U]));
                        }
                    }
                }
                for (size_t q = 0; q < columns.block; ++q)
                {
                    packets(face, (a * sides + side) * columns.block + q) = transfer[q];
                    interval.direction[side][q].add_product(sign * .5, transfer[q]);
                }
            }
        }
    }
}

void store_block(Flow::CellState& stock, size_t phase, const Block& block)
{
    *stock.phase_numbers[phase] = block[H0];
    *stock.qbmm_shape[phase] = {block[H2], block[H3]};
    stock.phases[phase].internal_energy = block[U];
    std::ranges::copy(moles(block), stock.phases[phase].moles.begin());
}

double balance(double before, double after, const FaceSum& transfer, const FaceSum& gross, double tolerance)
{
    FaceSum defect = transfer, scale = gross;
    defect.add_product(before, 1.);
    defect.add_product(-after, 1.);
    scale.add_product(std::abs(before), 1.);
    scale.add_product(std::abs(after), 1.);
    const double normalized = relative_difference(defect, scale) / tolerance;
    require(normalized <= 1., "QBMM linear transport fails its represented cell/face balance.");
    return normalized;
}

void record_residual(TransportResult& result, size_t channel, double residual)
{
    if (channel == H0)
        result.maximum_population_residual = std::max(result.maximum_population_residual, residual);
    else if (channel == H2 || channel == H3)
        result.maximum_shape_residual = std::max(result.maximum_shape_residual, residual);
    else if (channel == U)
        result.maximum_energy_residual = std::max(result.maximum_energy_residual, residual);
    else if (channel == H1 || channel >= FirstSpecies)
        result.maximum_species_residual = std::max(result.maximum_species_residual, residual);
}

struct StepTransfer
{
    FaceSum net, gross;

    void add(double amount, double sign)
    {
        net.add_product(amount, sign);
        gross.add_product(std::abs(amount), std::abs(sign));
    }
};

template<class Packets>
void apply_stage(const Mesh& mesh, const Flow::Options& options, Flow::State& stage, const std::vector<size_t>& phases,
    size_t width, const Packets& packets, std::vector<StepTransfer>& step, TransportResult& result)
{
    const double tolerance = options.conservation_relative_tolerance;
    for (size_t c = 0; c < stage.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            const auto before = cell_block(family, stage.cells[c], phase);
            auto after = before;
            std::vector<StepTransfer> stage_transfer(width);
            for (const auto face : mesh.faces(cell))
            {
                const size_t side = mesh.owner_cell(face) == cell ? 0 : 1;
                const size_t out = (a * sides + side) * width, in = (a * sides + 1 - side) * width;
                for (size_t q = 0; q < width; ++q)
                {
                    const double incoming = packets(face, in + q), outgoing = packets(face, out + q);
                    stage_transfer[q].add(incoming, 1.);
                    stage_transfer[q].add(outgoing, -1.);
                    auto& average = step[(c * phases.size() + a) * width + q];
                    average.add(incoming, .5);
                    average.add(outgoing, -.5);
                }
            }
            if (stage_transfer[H0].gross.sign() == 0)
                continue;
            // The donor owner certified the part retained after these same
            // represented outgoing packets. Incoming packets are nonnegative
            // measures. Apply their sum in canonical extensive stock units.
            for (size_t q = 0; q < width; ++q)
            {
                if (q == H1 || q == W)
                    continue;
                auto endpoint = stage_transfer[q].net;
                endpoint.add_product(before[q], 1.);
                after[q] = represented(endpoint);
            }
            check_realizable(family, after, "forward-Euler endpoint");
            for (size_t q = 0; q < width; ++q)
                if (q != W)
                {
                    const double residual =
                        balance(before[q], after[q], stage_transfer[q].net, stage_transfer[q].gross, tolerance);
                    result.maximum_stage_residual = std::max(result.maximum_stage_residual, residual);
                    record_residual(result, q, residual);
                }
            store_block(stage.cells[c], phase, after);
        }
    }
}

void complete_substep(const Flow::Options& options, const Flow::State& start, Flow::State& stage,
    const std::vector<size_t>& phases, size_t width, const std::vector<StepTransfer>& step,
    std::vector<CellInterval>& volumes, TransportResult& result)
{
    const double tolerance = options.conservation_relative_tolerance;
    for (size_t c = 0; c < stage.cells.size(); ++c)
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            const auto before = cell_block(family, start.cells[c], phase);
            const auto second_euler = cell_block(family, stage.cells[c], phase);
            auto after = second_euler;
            const auto base = (c * phases.size() + a) * width;
            // Evaluate the conservative form of SSPRK2 directly, rounding
            // the accepted stock once after both actual face increments.
            // The convex stage average is checked independently below.
            for (size_t q = 0; q < width; ++q)
                if (q != H1 && q != W)
                {
                    auto update = step[base + q].net;
                    update.add_product(before[q], 1.);
                    after[q] = represented(update);
                }
            check_realizable(family, after, "SSPRK2 endpoint");
            for (size_t q = 0; q < width; ++q)
                if (q != W)
                {
                    const auto& transfer = step[base + q];
                    FaceSum half_change;
                    half_change.add_product(-.5, before[q]);
                    half_change.add_product(.5, second_euler[q]);
                    const double residual =
                        std::max(balance(before[q], after[q], transfer.net, transfer.gross, tolerance),
                            balance(before[q], after[q], half_change, transfer.gross, tolerance));
                    result.maximum_step_residual = std::max(result.maximum_step_residual, residual);
                    record_residual(result, q, residual);
                }
            auto& volume = volumes[c * phases.size() + a];
            volume.work_volume.add_weighted(step[base + W].net, 1.);
            volume.gross_volume.add_weighted(step[base + W].gross, 1.);
            store_block(stage.cells[c], phase, after);
        }
}
} // namespace

TransportResult transport(const SP<const Flow::Mesh>& grid, const Flow::Options& options, const Flow::State& accepted,
    const Flow::State& thermodynamic_guess, Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector,
    TraceLimiter limiter)
{
    const auto& mesh = *grid;
    const auto owned_cells = mesh.owned_cell_map(), overlap_cells = mesh.overlap_cell_map();
    const auto owned_faces = mesh.owned_face_map(), overlap_faces = mesh.overlap_face_map();
    const auto& comm = *owned_cells->getComm();
    const size_t nc = mesh.num_owned_cells(), nf = mesh.num_owned_faces(), np = options.qbmm_families.size();
    const double cap = options.qbmm_model.linear_cfl;
    const int max_substeps = options.qbmm_model.maximum_linear_substeps;
    const auto slope_limiter = options.qbmm_model.linear_slope_limiter;
    const bool tvd_mc = slope_limiter == QbmmModelOptions::LinearSlopeLimiter::TvdMc;
    const auto validate = [&](const char* context, auto&& operation)
    { collective_detail::collective_local_validation(mesh, context, operation); };

    TransportResult result;
    std::vector<size_t> phases;
    std::vector<CellStencil> stencils;
    TraceColumns columns{};
    std::vector<FaceInterval> faces;
    std::vector<CellInterval> volumes;
    FaceSum caloric, energy;
    unsigned long long returned_packets = 0;
    validate("QBMM linear transport setup",
        [&]
        {
            require(accepted.cells.size() == nc && candidate.cells.size() == nc &&
                        thermodynamic_guess.cells.size() == nc && accepted_material_volumes.size() == nc,
                "QBMM linear owned-cell layout differs.");
            require(std::isfinite(dt) && dt > 0 && cap > 0 && cap <= .5 && max_substeps > 0,
                "Invalid QBMM linear interval or CFL policy.");
            require(tvd_mc || slope_limiter == QbmmModelOptions::LinearSlopeLimiter::RealizabilityOnly,
                "Invalid QBMM linear slope limiter.");
            for (size_t p = 0; p < np; ++p)
                if (options.qbmm_families[p])
                {
                    phases.push_back(p);
                    columns.block = FirstSpecies + options.qbmm_families[p]->catalog().species().size();
                }
            require(!phases.empty(), "QBMM linear mode has no bound phase.");
            stencils = cell_stencils(mesh, carrier_flux, limiter, tvd_mc);
            result.owned_flux.resize(np);
            result.integrated_faces.resize(nf);
            result.advected_material_volumes = accepted_material_volumes;
            faces.resize(nf * phases.size());
            volumes.resize(nc * phases.size());
            for (const auto p : phases)
                result.owned_flux[p].resize(nf);
            for (auto& face : faces)
                for (auto& direction : face.direction)
                    direction.resize(columns.block);
            for (size_t c = 0; c < nc; ++c)
            {
                require(accepted_material_volumes[c].size() == np, "QBMM linear material-volume layout differs.");
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const double volume = accepted_material_volumes[c][phases[a]];
                    require(std::isfinite(volume) && volume >= 0, "Invalid QBMM linear accepted material volume.");
                    auto& cell = volumes[c * phases.size() + a];
                    cell.work_volume.add_product(volume, 1.);
                    cell.gross_volume.add_product(volume, 1.);
                }
            }
        });

    // These maps and exchange buffers are private to one corrector and its
    // fixed family/catalog order. Each stage overwrites them before use.
    collective_detail::require_uniform_value(mesh, corrector, "QBMM linear corrector index");
    collective_detail::require_uniform_value(mesh, static_cast<int>(limiter), "QBMM linear trace limiter");
    collective_detail::require_uniform_value(mesh, static_cast<int>(slope_limiter), "QBMM linear slope limiter");
    collective_detail::require_uniform_value(mesh, dt, "QBMM linear requested interval");
    collective_detail::require_uniform_value(mesh, cap, "QBMM linear CFL cap");
    collective_detail::require_uniform_value(mesh, max_substeps, "QBMM linear substep cap");
    collective_detail::require_uniform_value(mesh, phases.size(), "QBMM linear family count");
    for (const auto phase : phases)
    {
        const auto& family = *options.qbmm_families[phase];
        collective_detail::require_uniform_value(mesh, phase, "QBMM linear phase index");
        collective_detail::require_uniform_value(mesh, family.fingerprint(), "QBMM linear family fingerprint");
        collective_detail::require_uniform_value(
            mesh, family.catalog().fingerprint(), "QBMM linear catalog fingerprint");
    }
    const Import cell_import(owned_cells, overlap_cells), face_import(owned_faces, overlap_faces);
    std::optional<Overlap> stocks, packets;
    std::optional<MultiVector> donor_traces, canonical_traces;
    validate("QBMM linear owner/ghost buffers",
        [&]
        {
            stocks.emplace(cell_import, owned_cells, overlap_cells, phases.size() * columns.block);
            packets.emplace(face_import, owned_faces, overlap_faces, phases.size() * sides * columns.block);
            donor_traces.emplace(overlap_faces, phases.size() * sides * columns.width());
            canonical_traces.emplace(owned_faces, phases.size() * sides * columns.width());
        });

    const auto maximum = [&](double value)
    {
        double global = 0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &value, &global);
        return global;
    };
    double local_limit = 0, limit = 0;
    validate("QBMM linear all-face CFL limit", [&] { local_limit = local_substep_limit(stencils, cap); });
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local_limit, &limit);

    double remaining = dt;
    while (remaining > 0)
    {
        if (result.substeps >= max_substeps)
            throw std::runtime_error("QBMM linear transport exhausted its subcycle cap.");
        double h = std::min(remaining, limit);
        for (;;)
        {
            h = exact_interval_substep(remaining, h);
            require(h > 0 && (h == remaining || remaining - h < remaining),
                "QBMM linear subcycle has no representable time progress.");
            double local_used = 0;
            validate("QBMM linear fractions actually used", [&] { local_used = local_cfl(stencils, h, cap); });
            const double used = maximum(local_used);
            if (used <= cap)
            {
                result.maximum_cfl = std::max(result.maximum_cfl, used);
                break;
            }
            h = std::nextafter(h, 0.);
        }

        Flow::State start;
        std::vector<StepTransfer> step;
        validate("QBMM linear SSPRK2 substep setup",
            [&]
            {
                start = candidate;
                step.resize(nc * phases.size() * columns.block);
                result.substep_durations.push_back(h);
            });
        for (int rk_stage = 0; rk_stage < 2; ++rk_stage)
        {
            validate("QBMM linear owned stage stocks",
                [&] { stage_stocks(options, candidate, phases, columns.block, stocks->write()); });
            stocks->synchronize();
            const char* trace_context =
                rk_stage == 0 ? "QBMM SSPRK2 stage 1 donor traces" : "QBMM SSPRK2 stage 2 donor traces";
            validate(trace_context,
                [&]
                {
                    donor_traces->putScalar(0.);
                    canonical_traces->putScalar(0.);
                    try
                    {
                        stage_traces(options, thermodynamic_guess, phases, stencils, columns, stocks->read(),
                            donor_traces->getLocalViewHost(Tpetra::Access::ReadWrite), h, dt, limiter, tvd_mc);
                    }
                    catch (const std::exception& error)
                    {
                        throw std::invalid_argument(std::string(trace_context) + ": " + error.what());
                    }
                });
            // ADD combines the two disjoint direction slots. Each nonzero
            // slot has exactly one cell-owner writer, verified by its marker.
            canonical_traces->doExport(*donor_traces, face_import, Tpetra::ADD);
            validate("QBMM linear canonical face packets",
                [&]
                {
                    stage_packets(mesh, options, carrier_flux, phases, columns,
                        canonical_traces->getLocalViewHost(Tpetra::Access::ReadOnly), packets->write(), faces, caloric,
                        energy, h, tvd_mc, returned_packets);
                });
            packets->synchronize();
            validate("QBMM linear represented forward-Euler endpoint",
                [&] { apply_stage(mesh, options, candidate, phases, columns.block, packets->read(), step, result); });
        }
        validate("QBMM linear represented SSPRK2 endpoint",
            [&] { complete_substep(options, start, candidate, phases, columns.block, step, volumes, result); });
        ++result.substeps;
        remaining -= h;
    }

    validate("QBMM linear interval receipts and work volumes",
        [&]
        {
            FaceSum interval;
            interval.add_product(dt, 1.);
            for (const auto h : result.substep_durations)
                interval.add_product(-h, 1.);
            require(!interval.sign(), "QBMM linear substeps do not cover the requested interval exactly.");
            for (size_t c = 0; c < nc; ++c)
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto phase = phases[a];
                    const auto& cell = volumes[c * phases.size() + a];
                    const auto& state = candidate.cells[c];
                    if (*state.phase_numbers[phase] == 0)
                    {
                        require(state.phases[phase].internal_energy == 0 &&
                                    relative_difference(cell.work_volume, cell.gross_volume) <=
                                        2 * qbmm_reconstruction_tolerance,
                            "An empty QBMM linear endpoint has inconsistent energy or work volume.");
                        result.advected_material_volumes[c][phase] = 0.;
                    }
                    else
                    {
                        require(cell.work_volume.sign() > 0, "QBMM linear material work volume is nonpositive.");
                        result.advected_material_volumes[c][phase] = represented(cell.work_volume);
                    }
                }
            for (size_t f = 0; f < nf; ++f)
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto& interval = faces[f * phases.size() + a];
                    auto& physical = result.integrated_faces[f];
                    physical.forward_mass.add_weighted(interval.momentum[0][Mass], 1.);
                    physical.backward_mass.add_weighted(interval.momentum[1][Mass], -1.);
                    physical.theta.add_weighted(interval.theta, 1.);
                    physical.material_volume.add_weighted(interval.direction[0][W], 1.);
                    physical.material_volume.add_weighted(interval.direction[1][W], 1.);
                    result.owned_flux[phases[a]][f] =
                        face_receipt(interval, dt, carrier_flux.local_value(static_cast<LO>(f)));
                }
            result.local_caloric_transport_magnitude = represented(caloric);
            result.local_energy_transport_magnitude = represented(energy);
        });
    validate_closed_balances(
        mesh, options, accepted, candidate, phases, columns.block, accepted_material_volumes, result);
    result.maximum_stage_residual = maximum(result.maximum_stage_residual);
    result.maximum_step_residual = maximum(result.maximum_step_residual);
    result.maximum_species_residual = maximum(result.maximum_species_residual);
    result.maximum_population_residual = maximum(result.maximum_population_residual);
    result.maximum_shape_residual = maximum(result.maximum_shape_residual);
    result.maximum_energy_residual = maximum(result.maximum_energy_residual);
    if (tvd_mc)
    {
        unsigned long long global_returned = 0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &returned_packets, &global_returned);
        result.upstream_returned_packets = global_returned;
    }
    return result;
}
} // namespace SimpleFluid::Reacting::qbmm_linear_detail
