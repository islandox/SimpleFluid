#include "solvers/details/LowMachQbmmLinearTransport.hh"

#include "FVM/ALEControlVolumeState.hh"
#include "equations/CollectiveValidation.hh"
#include "geometry/GeometryExecutionGuard.hh"
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
using WideBlock = std::vector<long double>;
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
    Accumulator outgoing_rate;
    struct Neighbor
    {
        LO cell;
        double volume;
        Vec3 displacement;
    };
    std::vector<Neighbor> neighbors;
    std::vector<FaceStencil> traces;
};

struct CarrierRates
{
    const Flow::Traits::scalar_face_type& absolute;
    std::optional<decltype(std::declval<const Overlap&>().read())> relative;

    double local_value(LO face) const { return relative ? (*relative)(face, 0) : absolute.local_value(face); }
};

Vec3 planar_face_offset(const Mesh& mesh, LO face, LO cell)
{
    if (std::holds_alternative<Mesh::CartesianPtr>(mesh.variant()))
    {
        // Cartesian faces share the cell's transverse center. A half width
        // preserves the offset even at a large coordinate origin.
        const double sign = mesh.owner_cell(face) == cell ? 1. : -1.;
        return mesh.face_normal(face) * (sign * mesh.cell_to_face_distance(face, cell));
    }
    return mesh.face_center_vector(face, cell);
}

class StageGeometry
{
    // Planar motion changes volumes and offsets affinely. Use the same
    // stage geometry for both trace policies, including non-donor traces.
    const Mesh& mesh;
    const PlanarTransportGeometry* planar;
    bool from_old;
    double fraction;

    double interpolate(double old_value, double new_value) const
    {
        return from_old ? std::lerp(old_value, new_value, fraction) : std::lerp(new_value, old_value, fraction);
    }

    Vec3 interpolate(const Vec3& old_offset, const Vec3& new_offset) const
    {
        return {interpolate(old_offset.x, new_offset.x), interpolate(old_offset.y, new_offset.y),
            interpolate(old_offset.z, new_offset.z)};
    }

public:
    StageGeometry(const Mesh& mesh_, const PlanarTransportGeometry* planar_, double dt, double remaining)
        : mesh(mesh_), planar(planar_), from_old(remaining >= dt * .5),
          fraction((from_old ? dt - remaining : remaining) / dt)
    {
    }

    double volume(LO cell) const
    {
        return planar ? interpolate(planar->ale.old_cell_volumes()[cell], planar->ale.new_cell_volumes()[cell])
                      : mesh.cell_volume(cell);
    }

    Vec3 cell_displacement(LO face, LO cell, LO neighbor, const Vec3& center, bool tvd_mc) const
    {
        if (planar)
        {
            const auto old = planar->old_faces[face].owner_to_neighbor;
            return interpolate(mesh.owner_cell(face) == cell ? old : old * -1., mesh.cell_center_vector(face, cell));
        }
        return tvd_mc ? mesh.cell_center_vector(face, cell) : mesh.cell_centroid(neighbor) - center;
    }

    Vec3 face_offset(LO face, LO cell, const Vec3& center, bool tvd_mc) const
    {
        if (planar)
        {
            const auto& old = planar->old_faces[face];
            return interpolate(mesh.owner_cell(face) == cell ? old.owner_to_face : old.neighbor_to_face,
                planar_face_offset(mesh, face, cell));
        }
        return tvd_mc ? mesh.face_center_vector(face, cell) : mesh.face_centroid(face) - center;
    }
};

void cell_stencils(const Mesh& mesh, const CarrierRates& flux, const StageGeometry& geometry, TraceLimiter limiter,
    bool tvd_mc, std::vector<CellStencil>& stencils, std::vector<Vec3>& displacements, FaceOffsetFit& fit)
{
    stencils.resize(mesh.num_owned_cells());
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
        stencil.volume = geometry.volume(cell);
        require(std::isfinite(stencil.volume) && stencil.volume > 0, "Invalid QBMM linear bulk volume.");

        stencil.outgoing_rate = {};
        stencil.neighbors.clear();
        displacements.clear();
        for (const auto face : mesh.faces(cell))
        {
            if (mesh.is_geometry_exterior_face(face))
                continue;
            const auto neighbor = mesh.opposite_or_periodic_neighbor_cell(face, cell);
            require(neighbor >= 0 && static_cast<size_t>(neighbor) < mesh.num_local_cells() && neighbor != cell,
                "QBMM linear reconstruction requires the adjacent cell in its one-ring halo.");
            const double neighbor_volume = geometry.volume(neighbor);
            require(std::isfinite(neighbor_volume) && neighbor_volume > 0, "Invalid QBMM linear neighbor volume.");
            const auto displacement = geometry.cell_displacement(face, cell, neighbor, center, tvd_mc);
            displacements.push_back(displacement);
            stencil.neighbors.push_back({neighbor, neighbor_volume, displacement});
        }
        fit.assign(displacements);
        size_t next_neighbor = 0, next_trace = 0;
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
            if (next_trace == stencil.traces.size())
                stencil.traces.emplace_back();
            auto& wanted = stencil.traces[next_trace++];
            wanted.face = face;
            wanted.side = side;
            wanted.outward = outward;
            wanted.neighbor = neighbor;
            if (outward > 0.)
                stencil.outgoing_rate += outward;
            wanted.offset = geometry.face_offset(face, cell, center, tvd_mc);
            const bool resolved = fit.weights(wanted.offset, wanted.weights);
            require(
                resolved || outward <= 0, "QBMM linear one-ring geometry cannot reproduce this outgoing face offset.");
            if (!resolved)
            {
                // A non-transporting side may use its cell-center trace.
                wanted.weights.clear();
                wanted.center_weights.clear();
            }
            else if (tvd_mc && !fit.weights(displacements[neighbor], wanted.center_weights))
            {
                // The center difference itself is a one-sided projection
                // when the fit loses this direction to numerical rank.
                wanted.center_weights.assign(displacements.size(), 0.);
                wanted.center_weights[neighbor] = 1.;
            }
        }
        stencil.traces.resize(next_trace);
    }
}

double fused(double a, double b, double c)
{
#if defined(__GNUC__) && defined(__FMA__)
    return __builtin_fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

bool product_has_range(double product, double a, double b)
{
    // The residual of two 53-bit significands can be much smaller than the
    // rounded product. Leave enough binary64 range for that residual too.
    constexpr double minimum = 0x1p-968;
    return (std::abs(product) >= minimum && std::isfinite(product)) || a == 0. || b == 0.;
}

// The fused residual distinguishes products that round to the same value.
double product_difference(double a, double b, double c, double d, double left, double right)
{
    const double difference = left - right;
    if (a == c && b == d)
        return 0.;
    if (std::abs(difference) > std::max(std::abs(left), std::abs(right)) * (8 * eps))
        return difference;
    return fused(a, b, -right) + fused(-c, d, right);
}

double local_substep_limit(const std::vector<CellStencil>& stencils, double cap, const PlanarTransportGeometry* planar)
{
    double limit = std::numeric_limits<double>::infinity();
    for (size_t c = 0; c < stencils.size(); ++c)
    {
        const auto& stencil = stencils[c];
        const auto rate = stencil.outgoing_rate.value();
        if (rate == 0.)
            continue;
        long double numerator = static_cast<long double>(cap) * stencil.volume;
        long double denominator = rate;
        if (planar && planar->ale.old_cell_volumes()[c] > planar->ale.new_cell_volumes()[c])
        {
            // On contraction the minimum volume is at the end of this substep.
            // Rearrange h*rate <= cap*(V_start - h*(V_old - V_new)/dt).
            const auto dt = planar->ale.time_step();
            const auto contraction =
                static_cast<long double>(planar->ale.old_cell_volumes()[c]) - planar->ale.new_cell_volumes()[c];
            numerator *= dt;
            denominator = rate * dt + cap * contraction;
        }
        const auto exact_bound = numerator / denominator;
        double bound = static_cast<double>(exact_bound);
        if (std::isfinite(bound) && bound > 0. && bound > exact_bound)
            bound = std::nextafter(bound, 0.);
        limit = std::min(limit, bound);
    }
    return limit;
}

double local_cfl(const std::vector<CellStencil>& stencils, double h, double cap, const StageGeometry* end_geometry)
{
    double maximum = 0;
    for (size_t c = 0; c < stencils.size(); ++c)
    {
        const auto& stencil = stencils[c];
        const double volume =
            end_geometry ? std::min(stencil.volume, end_geometry->volume(static_cast<LO>(c))) : stencil.volume;
        require(std::isfinite(volume) && volume > 0, "Invalid QBMM linear interval volume.");
        const auto rate = stencil.outgoing_rate.value();
        const auto swept = rate * h;
        const auto allowed = static_cast<long double>(cap) * volume;
        double fraction = static_cast<double>(swept / volume);
        bool exceeds = swept > allowed;
        if (fraction == cap && rate > 0.)
        {
            const double narrow_rate = static_cast<double>(rate);
            const double left = narrow_rate * h, right = cap * volume;
            const auto difference =
                narrow_rate == rate && product_has_range(left, narrow_rate, h) && product_has_range(right, cap, volume)
                    ? product_difference(narrow_rate, h, cap, volume, left, right)
                    : swept - allowed;
            // The rounded aggregate can hide positive outflow on another face.
            auto remainder = stencil.outgoing_rate;
            remainder += -rate;
            Accumulator excess;
            excess += difference;
            excess += h * remainder.value();
            exceeds = excess.value() > 0.;
        }
        if (exceeds)
            fraction = std::max(fraction, std::nextafter(cap, std::numeric_limits<double>::infinity()));
        maximum = std::max(maximum, fraction);
    }
    return maximum;
}

void cell_block(const QbmmPopulationSchema& family, const Flow::CellState& stock, size_t phase, Block& block)
{
    const auto& thermal = stock.phases[phase];
    const auto moments = qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, stock, phase));
    std::copy(moments.begin(), moments.end(), block.begin());
    block[U] = thermal.internal_energy;
    std::ranges::copy(thermal.moles, block.begin() + FirstSpecies);
    block[W] = 0.;
}

long double material_mass(const QbmmPopulationSchema& family, std::span<const double> moles)
{
    Accumulator mass;
    for (size_t s = 0; s < moles.size(); ++s)
        mass += static_cast<long double>(moles[s]) * family.catalog().species()[s].molarMass;
    return mass.value();
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

long double stock_difference(const CellStencil& stencil, size_t neighbor, double center, double other, bool singleton)
{
    const double other_volume = stencil.neighbors[neighbor].volume;
    if (stencil.volume == other_volume)
        return static_cast<long double>(other) - center;

    const double left = stencil.volume * other, right = other_volume * center;
    if (product_has_range(left, stencil.volume, other) && product_has_range(right, other_volume, center))
    {
        const double difference = product_difference(stencil.volume, other, other_volume, center, left, right);
        const double rounded = difference / other_volume;
        if (!singleton && std::isnormal(rounded))
            return rounded;
        return static_cast<long double>(difference) / other_volume;
    }
    return (static_cast<long double>(stencil.volume) * other - static_cast<long double>(other_volume) * center) /
           other_volume;
}

long double density_jump(
    const CellStencil& stencil, size_t neighbor, double center, double other, long double difference)
{
    const double other_volume = stencil.neighbors[neighbor].volume;
    // A constant density can differ in its last bits when each cell stores
    // its own rounded extensive amount. A zero-to-positive jump stays active.
    if (difference == 0.)
        return 0.;
    const auto magnitude = static_cast<long double>(stencil.volume) * other / other_volume + center;
    return relative_difference(difference, magnitude) > 2 * eps ? difference : 0.;
}

double unit_ratio(long double numerator, long double denominator)
{
    if (numerator == 0. || denominator == 0.)
        return numerator == denominator ? 1. : 0.;
    if (std::signbit(numerator) != std::signbit(denominator))
        return 0.;
    const auto left = std::abs(numerator), right = std::abs(denominator);
    return left >= right ? 1. : static_cast<double>(left / right);
}

double mc_factor(long double jump, long double projected)
{
    // r = upstream/jump, phi_MC = min(2r, (1+r)/2, 2).
    // Divide by the centered branch so the result scales the existing
    // physical face increment. This also avoids overflowing a large r.
    const auto magnitude = std::max(std::abs(jump), std::abs(projected));
    if (magnitude == 0.)
        return 0.;
    const auto minimum = std::numeric_limits<double>::min();
    const auto small = [minimum](long double value) { return value != 0. && std::abs(value) < minimum; };
    // Scale both operands when either lacks range or doubling can overflow.
    const int shift = small(jump) || small(projected) || magnitude > std::numeric_limits<double>::max() / 8.
                          ? -std::ilogb(magnitude)
                          : 0;
    const double j = shift == 0 ? static_cast<double>(jump) : static_cast<double>(std::scalbn(jump, shift));
    const double p = shift == 0 ? static_cast<double>(projected) : static_cast<double>(std::scalbn(projected, shift));
    // After rescaling, a still-tiny jump limits the MC factor by its ratio.
    // Evaluate that ratio before rounding the jump to subnormal precision.
    if (jump != 0. && !std::isnormal(j))
        return unit_ratio(2 * jump, projected);
    const double upstream = 2 * p - j;
    if (upstream == 0. || std::signbit(upstream) != std::signbit(j) || p == 0.)
        return 0.;
    return std::min(unit_ratio(2 * upstream, p), unit_ratio(2 * j, p));
}

long double project_column(
    std::span<const WideBlock> increments, std::span<const double> weights, size_t column, bool singleton = false)
{
    if (!singleton)
    {
        detail::CompensatedSum<double> projection;
        bool native = true;
        for (size_t k = 0; k < weights.size(); ++k)
        {
            if (weights[k] == 0. || increments[k][column] == 0.)
                continue;
            const double term = weights[k] * static_cast<double>(increments[k][column]);
            if (!std::isnormal(term))
            {
                native = false;
                break;
            }
            projection += term;
        }
        if (native && std::isfinite(projection.value()))
            return projection.value();
    }

    // Keep the moment-ray tangent or a packet that is outside binary64
    // intermediate range until reconstruction reaches the packet boundary.
    long double wide = 0.;
    for (size_t k = 0; k < weights.size(); ++k)
        wide += weights[k] * increments[k][column];
    return wide;
}

double reconstruct_channel(double center, long double slope, double factor)
{
    if (factor == 0. || slope == 0.)
        return center;
    const double narrow_slope = static_cast<double>(slope);
    if (narrow_slope == slope)
    {
        const double value = fused(factor, narrow_slope, center);
        if (std::isnormal(value))
            return value;
    }
    return represented(center + factor * slope);
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
            const double left = amount * other.volume, right = other.amount * volume;
            if (product_has_range(left, amount, other.volume) && product_has_range(right, other.amount, volume))
            {
                if (left != right)
                    return left < right;
                return product_difference(amount, other.volume, other.amount, volume, left, right) < 0.;
            }
            return static_cast<long double>(amount) * other.volume < static_cast<long double>(other.amount) * volume;
        }
    };
    struct Channel
    {
        size_t column;
        Stock minimum, maximum;
    };
    std::vector<Channel> channels;

    void assign(const CellStencil& stencil, const Block& center, std::span<const Block> neighbors)
    {
        channels.clear();
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
                channels.push_back({q, low, high});
        }
    }

    void check(const Block& stock, long double volume) const
    {
        for (const auto& channel : channels)
        {
            const auto bounded = [&](const Stock& limit, double sign)
            {
                const auto bound = volume * limit.amount;
                const auto value = static_cast<long double>(stock[channel.column]) * limit.volume;
                return sign * (value - bound) <= (2 * eps) * (value + bound);
            };
            require(bounded(channel.minimum, -1.) && bounded(channel.maximum, 1.),
                "QBMM TVD-MC density exceeds the donor stencil bounds.");
        }
    }
};

struct EnergyBounds
{
    struct Stock
    {
        double energy;
        long double mass;
    };
    std::optional<Stock> minimum, maximum;

    static long double difference(const Stock& lhs, const Stock& rhs)
    {
        const double left_mass = static_cast<double>(lhs.mass), right_mass = static_cast<double>(rhs.mass);
        if (left_mass == lhs.mass && right_mass == rhs.mass)
        {
            const double left = right_mass * lhs.energy, right = left_mass * rhs.energy;
            if (product_has_range(left, right_mass, lhs.energy) && product_has_range(right, left_mass, rhs.energy))
                return product_difference(right_mass, lhs.energy, left_mass, rhs.energy, left, right);
        }
        return rhs.mass * lhs.energy - lhs.mass * rhs.energy;
    }

    void include(const QbmmPopulationSchema& family, const Block& block)
    {
        if (block[H0] == 0)
            return;
        const Stock stock{block[U], material_mass(family, moles(block))};
        require(stock.mass > 0., "An occupied QBMM linear stencil stock needs positive mass.");
        if (!minimum || difference(stock, *minimum) < 0.)
            minimum = stock;
        if (!maximum || difference(stock, *maximum) > 0.)
            maximum = stock;
    }

    static bool at_most(const Stock& lhs, const Stock& rhs, double relative_slack = 2 * eps)
    {
        const auto excess = difference(lhs, rhs);
        if (excess <= 0.)
            return true;
        // The stock representation bounds each rounding by one epsilon.
        // At C <= 1/2 the outgoing rounding contributes at most one epsilon
        // relative to a first-order retained channel, and rounding that
        // channel contributes at most one more. Measure the resulting
        // envelope against the cross operands, with no absolute floor.
        const auto scale = rhs.mass * std::abs(lhs.energy) + lhs.mass * std::abs(rhs.energy);
        return excess <= relative_slack * scale;
    }

    static bool indistinguishable(const Stock& lhs, const Stock& rhs)
    {
        // Independent species and energy sums can round a constant ratio
        // differently. Ignore this uncertainty when selecting the MC ratio.
        // The face and remainder energy certificates keep their own bounds.
        constexpr double slope_slack = 8 * eps;
        return at_most(lhs, rhs, slope_slack) && at_most(rhs, lhs, slope_slack);
    }

    double reflected_slope_factor(const Stock& center, double energy_slope, double mass_slope, double factor) const
    {
        const double mass = static_cast<double>(center.mass);
        if (!std::isfinite(mass) || !std::isfinite(mass_slope) || !std::isfinite(energy_slope))
            return factor;
        // Keep both the trace and its reflection nonnegative in mass.
        if (std::abs(mass_slope) > mass)
            factor = std::min(factor, mass / std::abs(mass_slope));
        const auto clip = [&](const Stock& bound, double sign)
        {
            const double bound_mass = static_cast<double>(bound.mass);
            const double left = energy_slope * bound_mass, right = mass_slope * bound.energy;
            const double room = static_cast<double>(sign * difference(center, bound));
            if (!std::isfinite(left) || !std::isfinite(right) || !std::isfinite(room))
                return;
            const double debit =
                std::abs(product_difference(energy_slope, bound_mass, mass_slope, bound.energy, left, right));
            if (debit > 0.)
                factor = std::min(factor, std::max(0., room) / debit);
        };
        clip(*minimum, 1.);
        clip(*maximum, -1.);
        return factor;
    }

    void check(const QbmmPopulationSchema& family, const Block& block) const
    {
        if (block[H0] == 0)
            return;
        const Stock stock{block[U], material_mass(family, moles(block))};
        require(stock.mass > 0. && minimum && maximum && at_most(*minimum, stock) && at_most(stock, *maximum),
            "QBMM linear specific energy exceeds the occupied stencil bounds.");
    }
};

void tvd_factors(const QbmmPopulationSchema& family, const CellStencil& stencil, const Block& center,
    std::span<const Block> neighbors, std::span<const WideBlock> increments, std::span<const WideBlock> slopes,
    std::vector<WideBlock>& jumps, std::vector<EnergyBounds::Stock>& energy, std::vector<Vec3>& occupied_displacements,
    std::vector<long double>& energy_jumps, std::vector<double>& energy_weights, std::vector<double>& factors,
    FaceOffsetFit& energy_fit)
{
    std::ranges::fill(factors, 1.);
    if (center[H0] == 0)
        return;

    energy.clear();
    occupied_displacements.clear();
    energy_jumps.clear();
    const EnergyBounds::Stock center_energy{center[U], material_mass(family, moles(center))};
    for (size_t k = 0; k < neighbors.size(); ++k)
    {
        const auto& neighbor = neighbors[k];
        for (size_t q = 0; q < center.size(); ++q)
            if (limits_density(q))
                jumps[k][q] = density_jump(stencil, k, center[q], neighbor[q], increments[k][q]);

        energy.push_back({neighbor[U], material_mass(family, moles(neighbor))});
        if (neighbor[H0] == 0)
            continue;
        // Store the specific-energy difference times a common mass scale.
        // The common scale cancels in the MC ratio. A vacuum has no sample.
        const auto& sample = energy.back();
        energy_jumps.push_back(EnergyBounds::indistinguishable(center_energy, sample)
                                   ? 0.
                                   : EnergyBounds::difference(sample, center_energy) / neighbor[H1]);
        occupied_displacements.push_back(stencil.neighbors[k].displacement);
    }
    energy_fit.assign(occupied_displacements);
    const auto project_energy = [&]
    {
        long double projection = 0.;
        for (size_t k = 0; k < energy_weights.size(); ++k)
            projection += energy_jumps[k] * energy_weights[k];
        return projection;
    };
    for (size_t f = 0; f < stencil.traces.size(); ++f)
    {
        const auto& face = stencil.traces[f];
        const auto adjacent = face.neighbor;
        auto& factor = factors[f];
        if (face.center_weights.empty())
            continue;
        for (size_t q = 0; q < center.size() && factor > 0; ++q)
        {
            const auto jump = jumps[adjacent][q];
            if (jump == 0.)
                continue; // H1, U, W, and flat or excluded species cannot constrain the component pass.
            if (center[q] == 0)
            {
                factor = 0.;
                break;
            }
            const auto projected = project_column(jumps, face.center_weights, q);
            factor = std::min(factor, mc_factor(jump, projected));
            if (slopes[f][q] != 0.)
                factor = std::min(factor, unit_ratio(jump, slopes[f][q]));
            if (factor > 0 && slopes[f][q] > 0.)
                factor = std::min(factor, unit_ratio(center[q], slopes[f][q]));
        }
        const auto& downstream = energy[adjacent];
        if (factor == 0 || downstream.mass == 0. || EnergyBounds::indistinguishable(center_energy, downstream))
            continue;

        const auto jump = EnergyBounds::difference(downstream, center_energy);
        const auto local_jump = jump / neighbors[adjacent][H1];
        const auto projected = energy_fit.weights(stencil.neighbors[adjacent].displacement, energy_weights)
                                   ? project_energy()
                                   : local_jump;
        const double energy_mc = mc_factor(local_jump, projected);
        factor = std::min(factor, energy_mc);
        if (factor == 0.)
            continue;

        long double face_projection;
        if (energy_fit.weights(face.offset, energy_weights))
            face_projection = project_energy();
        else
        {
            // Only the donor-to-neighbor direction is determined by the
            // occupied stocks. Project the face offset on that direction.
            const auto& displacement = stencil.neighbors[adjacent].displacement;
            const double length = std::hypot(displacement.x, displacement.y, displacement.z);
            face_projection = projected * (face.offset / length).dot(displacement / length);
        }
        const double interpolation = unit_ratio(face_projection * energy_mc, local_jump);

        // The trace transports energy and species mass, so its specific
        // energy is a quotient. Bound the quotient at the MC face value
        // (1-t)*e_center + t*e_downstream using the cross-products with both
        // stocks. Their positive masses supply the common denominator.
        const auto cross_slope = [&](const EnergyBounds::Stock& reference)
        {
            const auto base = EnergyBounds::difference(center_energy, reference);
            long double slope = 0.;
            for (size_t k = 0; k < face.weights.size(); ++k)
            {
                const auto diff = EnergyBounds::difference(energy[k], reference);
                const auto other_volume = stencil.neighbors[k].volume;
                const auto relative_volume = static_cast<long double>(stencil.volume) / other_volume;
                slope += face.weights[k] * (diff * relative_volume - base);
            }
            return slope;
        };
        const auto center_slope = cross_slope(center_energy);
        if (center_slope != 0. && std::signbit(center_slope) != std::signbit(jump))
        {
            factor = 0.;
            continue;
        }
        const auto against_center = center_slope * neighbors[adjacent][H1];
        const auto against_downstream = cross_slope(downstream) * center[H1];
        const auto bound = jump * center[H1];
        const auto slope = (1. - interpolation) * against_center + interpolation * against_downstream;
        if (slope != 0. && std::signbit(slope) == std::signbit(jump))
            factor = std::min(factor, unit_ratio(bound * interpolation, slope));
    }
}

template<class View> void read_block(const View& view, LO row, size_t first, Block& block)
{
    for (size_t q = 0; q < block.size(); ++q)
        block[q] = view(row, first + q);
}

bool loses_packet_precision(long double amount, double divisor, double rounded)
{
    if (amount == 0.)
        return false;
    if (rounded == 0.)
        return true;
    if (std::fpclassify(rounded) != FP_SUBNORMAL)
        return false;
    const auto error = amount - static_cast<long double>(rounded) * divisor;
    return relative_difference(error, std::abs(amount)) > eps;
}

bool loses_receipt_precision(long double contribution, double dt)
{
    return loses_packet_precision(contribution, 1., static_cast<double>(contribution)) ||
           loses_packet_precision(contribution, dt, static_cast<double>(contribution / dt));
}

bool loses_interval_precision(long double amount, double dt)
{
    return loses_receipt_precision(amount * .5L, dt);
}

bool loses_packet_channel(long double amount, double divisor, double reference_volume, double dt)
{
    const double rounded = static_cast<double>(amount / divisor);
    if (loses_packet_precision(amount, divisor, rounded))
        return true;
    if (!std::isfinite(rounded))
        return false; // The regular packet certificate still rejects overflow.
    const double concentration = static_cast<double>(static_cast<long double>(rounded) / reference_volume);
    return !std::isfinite(concentration) || loses_packet_precision(rounded, reference_volume, concentration) ||
           loses_interval_precision(rounded, dt);
}

bool return_material_packet(const Block& trace, long double swept, double volume, double reference_volume, double dt)
{
    constexpr double count_floor = 1e-200, total_moles_floor = 1e-200, species_floor = 1e-220;
    const auto below_floor = [volume](long double amount, double floor)
    { return amount > 0. && amount < static_cast<long double>(floor) * volume; };
    Accumulator total_moles;
    for (size_t q = 0; q < trace.size(); ++q)
    {
        if (q == W || trace[q] == 0.)
            continue;
        const auto amount = swept * trace[q];
        if (loses_packet_channel(amount, volume, reference_volume, dt) ||
            (q == H0 && below_floor(amount, count_floor)) || (q >= FirstSpecies && below_floor(amount, species_floor)))
            return true;
        if (q >= FirstSpecies)
            total_moles += amount;
    }
    return below_floor(total_moles.value(), total_moles_floor);
}

long double material_volume(long double mass, double density)
{
    return mass / density;
}

long double pressure_theta(long double mass, double h, double stage_weight, double donor_volume, double density)
{
    return material_volume(mass, density) * (static_cast<long double>(h) * stage_weight / donor_volume);
}

bool return_physical_packet(long double mass, long double half_stage_theta, const StageThermal& point,
    const Vector& velocity, double reference_volume, double dt)
{
    if (loses_packet_channel(material_volume(mass, point.density), 1., reference_volume, dt) ||
        loses_interval_precision(mass, dt) || loses_receipt_precision(half_stage_theta, dt))
        return true;
    if (loses_interval_precision(mass * point.caloric_specific, dt))
        return true;
    for (size_t axis = 0; axis < dimensions; ++axis)
        if (loses_interval_precision(mass * velocity.component(axis), dt))
            return true;
    return false;
}

StageThermal trace_thermal(PhaseThermalState& thermal, std::vector<double>& composition, const PhaseThermalState& guess,
    const Block& trace, double pressure)
{
    thermal.thermodynamics = guess.thermodynamics;
    thermal.moles.assign(moles(trace).begin(), moles(trace).end());
    thermal.temperature = guess.temperature;
    thermal.internal_energy = trace[U];
    thermal.absolute_pressure = pressure;
    const auto point = thermal_trace(thermal, true, composition);
    require(
        (point.material_mass != 0) == (trace[H0] != 0), "QBMM linear trace count and EOS material presence differ.");
    return point;
}

void withdrawal(
    const QbmmPopulationSchema& family, const Block& trace, long double swept, double volume, Block& transfer)
{
    for (size_t q = 0; q < transfer.size(); ++q)
    {
        if (q == H1 || q == W)
            continue; // H1 is derived below. W is the physical EOS volume packet.
        transfer[q] = represented(swept * trace[q], volume);
    }
    // The transfer is the certified trace times the positive factor
    // h*phi/V. Its rounded channels are balance receipts. Re-inverting
    // those independently rounded channels can move an exact singleton
    // across the shifted determinant boundary solely by rescaling it.
    transfer[W] = 0.;
    check_material_block(family, transfer);
    const auto expected = swept * trace[H1];
    const auto actual = static_cast<long double>(volume) * transfer[H1];
    require(relative_difference(expected - actual, expected + actual) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM linear species-derived H1 transfer differs from the certified face trace.");
}

bool preserves_singleton(const Block& center, const WideBlock& slope)
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
        const auto channel = static_cast<long double>(slope[q]) * center[H0];
        const auto count = static_cast<long double>(slope[H0]) * center[q];
        if (relative_difference(channel - count, std::abs(channel) + std::abs(count)) > 16 * eps)
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

struct TraceScratch
{
    Block center, transfer, remainder;
    std::vector<Block> neighbors;
    std::vector<LimitedTrace> traces;
    std::vector<WideBlock> increments, slopes, jumps;
    std::vector<Accumulator> retained;
    DensityBounds bounds;
    std::vector<EnergyBounds::Stock> energy;
    std::vector<Vec3> occupied_displacements;
    std::vector<long double> energy_jumps;
    std::vector<double> energy_weights, mc;
    PhaseThermalState thermal;
    std::vector<double> composition;
    FaceOffsetFit energy_fit;

    void resize(size_t width, const std::vector<CellStencil>& stencils, bool tvd_mc)
    {
        size_t n_neighbors = 0, n_traces = 0;
        for (const auto& stencil : stencils)
        {
            n_neighbors = std::max(n_neighbors, stencil.neighbors.size());
            n_traces = std::max(n_traces, stencil.traces.size());
        }
        center.resize(width);
        transfer.resize(width);
        remainder.resize(width);
        retained.resize(width);
        neighbors.resize(n_neighbors);
        increments.resize(n_neighbors);
        slopes.resize(n_traces);
        traces.resize(n_traces);
        for (size_t k = 0; k < n_neighbors; ++k)
        {
            neighbors[k].resize(width);
            increments[k].resize(width);
        }
        for (size_t f = 0; f < n_traces; ++f)
        {
            slopes[f].resize(width);
            traces[f].block.resize(width);
        }
        thermal.moles.resize(width - FirstSpecies);
        composition.resize(width - FirstSpecies);
        if (tvd_mc)
        {
            jumps.resize(n_neighbors);
            for (auto& jump : jumps)
                jump.resize(width);
            bounds.channels.reserve(width);
            energy.reserve(n_neighbors);
            occupied_displacements.reserve(n_neighbors);
            energy_jumps.reserve(n_neighbors);
            energy_weights.reserve(n_neighbors);
            mc.resize(n_traces);
            energy_fit.reserve(n_neighbors);
        }
    }
};

void limit_traces(const QbmmPopulationSchema& family, const CellStencil& stencil,
    const PhaseThermalState& thermal_guess, const Vector& velocity, double pressure, double h, double dt,
    TraceLimiter limiter, bool tvd_mc, TraceScratch& scratch)
{
    const auto& center = scratch.center;
    const auto neighbors = std::span(scratch.neighbors).first(stencil.neighbors.size());
    auto traces = std::span(scratch.traces).first(stencil.traces.size());
    auto slopes = std::span(scratch.slopes).first(stencil.traces.size());
    const bool singleton =
        center[H0] != 0. && invert_qbmm_moments({center[H0], center[H1], center[H2], center[H3]}).active_atoms == 1;
    EnergyBounds energy;
    energy.include(family, center);
    for (const auto& neighbor : neighbors)
        energy.include(family, neighbor);
    if (tvd_mc)
        scratch.bounds.assign(stencil, center, neighbors);

    for (auto& slope : slopes)
        std::ranges::fill(slope, 0.);
    if (center[H0] != 0.)
    {
        // The singleton tangent test compares small differences against one
        // another. Keep extra mantissa bits through both subtractions so
        // cancellation of large stocks cannot erase a valid weight slope.
        for (size_t k = 0; k < neighbors.size(); ++k)
            for (size_t q = 0; q < center.size(); ++q)
                if (q != H1 && q != W)
                    scratch.increments[k][q] = stock_difference(stencil, k, center[q], neighbors[k][q], singleton);

        // Accumulate in donor-volume units without narrowing an intermediate
        // density jump, which can exceed binary64 even with a finite trace.
        for (size_t f = 0; f < traces.size(); ++f)
            for (size_t q = 0; q < center.size(); ++q)
                if (q != H1 && q != W)
                    slopes[f][q] = project_column(scratch.increments, stencil.traces[f].weights, q, singleton);
    }
    if (tvd_mc)
        tvd_factors(family, stencil, center, neighbors, scratch.increments, slopes, scratch.jumps, scratch.energy,
            scratch.occupied_displacements, scratch.energy_jumps, scratch.energy_weights, scratch.mc,
            scratch.energy_fit);
    const auto rate = stencil.outgoing_rate.value();
    const auto retained_volume = static_cast<long double>(stencil.volume) - h * rate;
    require(retained_volume > 0., "QBMM linear all-face donor retention must be positive.");

    const auto reconstruct = [&](size_t face, double factor)
    {
        auto& trace = traces[face].block;
        if (tvd_mc)
            factor *= scratch.mc[face];
        for (size_t q = 0; q < trace.size(); ++q)
        {
            if (q == H1 || q == W)
                continue;
            trace[q] = reconstruct_channel(center[q], slopes[face][q], factor);
        }
        check_realizable(family, trace, "face trace");
        energy.check(family, trace);
        if (tvd_mc)
            scratch.bounds.check(trace, stencil.volume);
    };
    const auto certify_donor = [&](double factor)
    {
        Accumulator applied_retained_volume;
        applied_retained_volume += retained_volume;
        auto& retained = scratch.retained;
        auto& transfer = scratch.transfer;
        std::ranges::fill(retained, Accumulator{});
        for (size_t q = 0; q < center.size(); ++q)
            retained[q] += center[q];
        for (size_t f = 0; f < traces.size(); ++f)
        {
            const auto& face = stencil.traces[f];
            const auto outward = face.outward;
            // The forced corrector certifies every potential donor trace with the same factor.
            // A vanishing rate can change direction without changing the set of trace constraints.
            if (outward > 0. || limiter == TraceLimiter::AllIncident)
                reconstruct(f, factor);
            auto& trace = traces[f];
            trace.thermal.reset();
            trace.returned = false;
            if (outward <= 0.)
                continue;
            const auto swept = static_cast<long double>(h) * outward;
            // The precision test uses the same reference on either side of a face,
            // including when the two stage volumes come from different MPI owners.
            const double reference_volume = std::min(stencil.volume, stencil.neighbors[face.neighbor].volume);
            trace.returned = tvd_mc && trace.block[H0] != 0 &&
                             return_material_packet(trace.block, swept, stencil.volume, reference_volume, dt);
            if (!trace.returned)
            {
                withdrawal(family, trace.block, swept, stencil.volume, transfer);
                if (tvd_mc && trace.block[H0] != 0)
                {
                    trace.thermal =
                        trace_thermal(scratch.thermal, scratch.composition, thermal_guess, trace.block, pressure);
                    const auto half_stage_theta = pressure_theta(
                        material_mass(family, moles(trace.block)), h, .5, stencil.volume, trace.thermal->density);
                    trace.returned = return_physical_packet(material_mass(family, moles(transfer)), half_stage_theta,
                        *trace.thermal, velocity, reference_volume, dt);
                }
            }
            if (trace.returned)
            {
                applied_retained_volume += swept;
                continue;
            }
            for (size_t q = 0; q < center.size(); ++q)
                retained[q] += -transfer[q];
        }

        // Certify the retained stock after the packets that actually leave
        // the donor. Its extensive units avoid independently rounding
        // density channels at rank changes.
        auto& remainder = scratch.remainder;
        for (size_t q = 0; q < center.size(); ++q)
            if (q != H1 && q != W)
                remainder[q] = represented(retained[q].value());
        check_realizable(family, remainder, "retained donor stock");
        energy.check(family, remainder);
        if (tvd_mc)
            scratch.bounds.check(remainder, applied_retained_volume.value());
    };

    const auto limit = [&](auto&& certify, double initial)
    {
        constexpr int halvings = std::numeric_limits<double>::digits;
        double factor = initial;
        for (int attempt = 0; factor > 0 && attempt <= halvings; ++attempt, factor *= .5)
        {
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
    if (singleton)
        for (size_t f = 0; f < traces.size(); ++f)
            if ((stencil.traces[f].outward > 0 || limiter == TraceLimiter::AllIncident) &&
                (!tvd_mc || scratch.mc[f] > 0) && !preserves_singleton(center, slopes[f]))
                initial = 0.;
    if (initial > 0. && tvd_mc && limiter == TraceLimiter::AllIncident &&
        !EnergyBounds::indistinguishable(*energy.minimum, *energy.maximum))
    {
        const EnergyBounds::Stock center_energy{center[U], material_mass(family, moles(center))};
        const auto& species = family.catalog().species();
        for (size_t f = 0; f < traces.size(); ++f)
        {
            auto& factor = scratch.mc[f];
            if (factor == 0.)
                continue;
            detail::CompensatedSum<double> mass_slope;
            for (size_t s = 0; s < species.size(); ++s)
                mass_slope += static_cast<double>(slopes[f][FirstSpecies + s]) * species[s].molarMass;
            // At most half the donor volume can leave. The retained stock is then
            // a nonnegative combination of the center and the reflected face stocks
            // for any subset of exporting faces. Certify the actual rounded packets below.
            factor = energy.reflected_slope_factor(
                center_energy, static_cast<double>(slopes[f][U]), mass_slope.value(), factor);
        }
    }
    const double factor = limit(certify_donor, initial);
    if (limiter == TraceLimiter::Outgoing)
        for (size_t f = 0; f < traces.size(); ++f)
            if (stencil.traces[f].outward == 0)
            {
                // The unforced policy permits its two tied pressure traces to
                // reduce their own slopes independently of the donor's outgoing traces.
                limit([&](double tie_factor) { reconstruct(f, tie_factor); }, factor);
            }
}

template<class View>
void stage_stocks(const Flow::Options& options, const Flow::State& stage, const std::vector<size_t>& phases,
    size_t width, const View& values, Block& block)
{
    for (size_t c = 0; c < stage.cells.size(); ++c)
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto cell = static_cast<LO>(c);
            const auto& family = *options.qbmm_families[phases[a]];
            cell_block(family, stage.cells[c], phases[a], block);
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
    double h, double dt, TraceLimiter limiter, bool tvd_mc, TraceScratch& scratch)
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
            read_block(stocks, cell, a * columns.block, scratch.center);
            for (size_t k = 0; k < stencil.neighbors.size(); ++k)
                read_block(stocks, stencil.neighbors[k].cell, a * columns.block, scratch.neighbors[k]);
            limit_traces(
                family, stencil, donor.phases[phase], donor.velocity, pressure, h, dt, limiter, tvd_mc, scratch);
            for (size_t f = 0; f < stencil.traces.size(); ++f)
            {
                const auto& face = stencil.traces[f];
                if (face.outward < 0.)
                    continue;
                const auto first = (a * sides + face.side) * columns.width();
                const auto& limited = scratch.traces[f];
                const auto& trace = limited.block;
                if (limited.returned)
                {
                    traces(face.face, first + columns.marker()) = TraceColumns::returned_marker;
                    continue;
                }
                const auto point = limited.thermal ? *limited.thermal
                                                   : trace_thermal(scratch.thermal, scratch.composition,
                                                         donor.phases[phase], trace, pressure);
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
void stage_packets(const Mesh& mesh, const Flow::Options& options, const CarrierRates& flux,
    const std::vector<size_t>& phases, const TraceColumns& columns, const Traces& traces, const Packets& packets,
    std::vector<FaceInterval>& intervals, Accumulator& caloric, Accumulator& energy, double h, bool tvd_mc,
    unsigned long long& returned_packets, Block& trace, Block& transfer)
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
                std::ranges::fill(transfer, 0.);
                if (selected && !returned)
                {
                    read_block(traces, face, first, trace);
                    check_realizable(family, trace, "face trace");
                    if (trace[H0] != 0)
                    {
                        const auto density = traces(face, first + columns.density());
                        require(std::isfinite(density) && density > 0, "Invalid QBMM linear face EOS density.");
                        const auto donor_volume = traces(face, first + columns.volume());
                        const auto trace_mass = material_mass(family, moles(trace));
                        // Tied relative rates supply two half-branch derivatives
                        // to the anchor without withdrawing any stock.
                        interval.theta +=
                            pressure_theta(trace_mass, h, outward == 0. ? .25 : .5, donor_volume, density);
                        if (outward > 0.)
                        {
                            withdrawal(family, trace, static_cast<long double>(h) * outward, donor_volume, transfer);
                            const auto mass = material_mass(family, moles(transfer));
                            transfer[W] = represented(material_volume(mass, density));
                            interval.momentum[side][Mass] += mass * (sign * .5);
                            for (size_t axis = 0; axis < dimensions; ++axis)
                                interval.momentum[side][CarrierMomentum + axis] +=
                                    mass * traces(face, first + columns.carrier() + axis) * (sign * .5);
                            caloric += mass * traces(face, first + columns.caloric()) * .5L;
                            energy += .5L * std::abs(transfer[U]);
                        }
                    }
                }
                for (size_t q = 0; q < columns.block; ++q)
                {
                    packets(face, (a * sides + side) * columns.block + q) = transfer[q];
                    interval.direction[side][q] += (sign * .5L) * transfer[q];
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

double balance(double before, double after, Accumulator defect, const Accumulator& gross, double tolerance)
{
    defect += before;
    defect += -after;
    const auto scale = gross.value() + std::abs(static_cast<long double>(before)) + std::abs(after);
    const double normalized = relative_difference(defect.value(), scale) / tolerance;
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
    Accumulator net, gross;

    void add(double amount, double sign)
    {
        net += static_cast<long double>(amount) * sign;
        gross += static_cast<long double>(std::abs(amount)) * std::abs(sign);
    }
};

template<class Packets>
void apply_stage(const Mesh& mesh, const Flow::Options& options, Flow::State& stage, const std::vector<size_t>& phases,
    size_t width, const Packets& packets, std::vector<StepTransfer>& step, TransportResult& result, Block& before,
    Block& after, std::vector<StepTransfer>& stage_transfer)
{
    const double tolerance = options.conservation_relative_tolerance;
    for (size_t c = 0; c < stage.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            cell_block(family, stage.cells[c], phase, before);
            std::ranges::fill(stage_transfer, StepTransfer{});
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
            if (stage_transfer[H0].gross.value() == 0.)
                continue;
            // The donor owner certified the part retained after these same
            // represented outgoing packets. Incoming packets are nonnegative
            // measures. Apply their sum in canonical extensive stock units.
            for (size_t q = 0; q < width; ++q)
            {
                if (q == H1 || q == W)
                    continue;
                auto endpoint = stage_transfer[q].net;
                endpoint += before[q];
                after[q] = represented(endpoint.value());
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
    std::vector<CellInterval>& volumes, TransportResult& result, Block& before, Block& second_euler, Block& after)
{
    const double tolerance = options.conservation_relative_tolerance;
    for (size_t c = 0; c < stage.cells.size(); ++c)
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            cell_block(family, start.cells[c], phase, before);
            cell_block(family, stage.cells[c], phase, second_euler);
            const auto base = (c * phases.size() + a) * width;
            // Evaluate the conservative form of SSPRK2 directly, rounding
            // the accepted stock once after both actual face increments.
            // The convex stage average is checked independently below.
            for (size_t q = 0; q < width; ++q)
                if (q != H1 && q != W)
                {
                    auto update = step[base + q].net;
                    update += before[q];
                    after[q] = represented(update.value());
                }
            check_realizable(family, after, "SSPRK2 endpoint");
            for (size_t q = 0; q < width; ++q)
                if (q != W)
                {
                    const auto& transfer = step[base + q];
                    Accumulator half_change;
                    half_change += -.5L * before[q];
                    half_change += .5L * second_euler[q];
                    const double residual =
                        std::max(balance(before[q], after[q], transfer.net, transfer.gross, tolerance),
                            balance(before[q], after[q], half_change, transfer.gross, tolerance));
                    result.maximum_step_residual = std::max(result.maximum_step_residual, residual);
                    record_residual(result, q, residual);
                }
            auto& volume = volumes[c * phases.size() + a];
            volume.work_volume.append(step[base + W].net);
            volume.gross_volume.append(step[base + W].gross);
            store_block(stage.cells[c], phase, after);
        }
}
} // namespace

std::vector<PlanarFaceGeometry> capture_planar_face_geometry(const Mesh& mesh)
{
    const auto execution = acquire_mesh_execution(mesh);
    std::vector<PlanarFaceGeometry> faces(mesh.num_faces());
    for (size_t f = 0; f < faces.size(); ++f)
    {
        const auto face = static_cast<LO>(f);
        if (mesh.is_geometry_exterior_face(face))
            continue;
        const auto owner = mesh.owner_cell(face), neighbor = mesh.neighbor_cell(face);
        if (owner < 0 || neighbor < 0)
            continue; // Fringe faces have no owned row in the reconstruction.
        faces[f] = {mesh.cell_center_vector(face, owner), planar_face_offset(mesh, face, owner),
            planar_face_offset(mesh, face, neighbor)};
    }
    return faces;
}

TransportResult transport(const SP<const Flow::Mesh>& grid, const Flow::Options& options, const Flow::State& accepted,
    const Flow::State& thermodynamic_guess, Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector,
    TraceLimiter limiter, const PlanarTransportGeometry* planar)
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

    collective_detail::require_uniform_value(mesh, int(planar != nullptr), "QBMM linear planar transport selection");
    if (planar)
        planar->ale.validate(mesh, dt);
    CarrierRates transport_flux{carrier_flux, std::nullopt};

    TransportResult result;
    std::vector<size_t> phases;
    std::vector<CellStencil> stencils;
    std::vector<Vec3> displacements;
    FaceOffsetFit stencil_fit;
    TraceColumns columns{};
    std::vector<FaceInterval> faces;
    std::vector<CellInterval> volumes;
    Accumulator caloric, energy;
    TraceScratch trace_scratch;
    Block before, after, second_euler;
    std::vector<StepTransfer> stage_transfer, step;
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
                    const size_t width = FirstSpecies + options.qbmm_families[p]->catalog().species().size();
                    require(phases.empty() || columns.block == width, "QBMM linear family catalog sizes differ.");
                    columns.block = width;
                    phases.push_back(p);
                }
            require(!phases.empty(), "QBMM linear mode has no bound phase.");
            if (!planar)
                cell_stencils(mesh, transport_flux, StageGeometry(mesh, nullptr, dt, dt), limiter, tvd_mc, stencils,
                    displacements, stencil_fit);
            result.owned_flux.resize(np);
            result.integrated_faces.resize(nf);
            result.advected_material_volumes = accepted_material_volumes;
            faces.resize(nf * phases.size());
            volumes.resize(nc * phases.size());
            before.resize(columns.block);
            after.resize(columns.block);
            second_euler.resize(columns.block);
            stage_transfer.resize(columns.block);
            step.resize(nc * phases.size() * columns.block);
            if (!planar)
                trace_scratch.resize(columns.block, stencils, tvd_mc);
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
                    cell.work_volume += volume;
                    cell.gross_volume += volume;
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
    std::optional<Overlap> stocks, packets, relative_carrier;
    std::optional<MultiVector> donor_traces, canonical_traces;
    validate("QBMM linear owner/ghost buffers",
        [&]
        {
            stocks.emplace(cell_import, owned_cells, overlap_cells, phases.size() * columns.block);
            packets.emplace(face_import, owned_faces, overlap_faces, phases.size() * sides * columns.block);
            donor_traces.emplace(overlap_faces, phases.size() * sides * columns.width());
            canonical_traces.emplace(owned_faces, phases.size() * sides * columns.width());
        });
    if (planar)
    {
        validate("QBMM linear planar carrier and geometry",
            [&]
            {
                require(options.qbmm_model.velocity == QbmmModelOptions::Velocity::SharedCarrier,
                    "QBMM linear planar transport requires a shared carrier.");
                require(
                    planar->old_faces.size() == mesh.num_faces(), "QBMM linear planar face geometry layout differs.");
                const auto finite = [](const Vec3& value)
                { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); };
                require(std::ranges::all_of(planar->old_faces,
                            [&](const PlanarFaceGeometry& face)
                            {
                                return finite(face.owner_to_neighbor) && finite(face.owner_to_face) &&
                                       finite(face.neighbor_to_face);
                            }),
                    "QBMM linear planar face geometry is nonfinite.");
                relative_carrier.emplace(face_import, owned_faces, overlap_faces, 1);
                const auto values = relative_carrier->write();
                for (size_t f = 0; f < nf; ++f)
                {
                    const auto face = static_cast<LO>(f);
                    const double relative = carrier_flux.local_value(face) - planar->ale.face_mesh_fluxes()[f];
                    require(std::isfinite(relative), "QBMM linear relative carrier flux is nonfinite.");
                    require(!mesh.is_geometry_exterior_face(face) || (mesh.is_boundary_face(face) && relative == 0.),
                        "QBMM linear planar transport requires impermeable physical walls.");
                    values(face, 0) = relative == 0. ? 0. : relative;
                }
            });
        relative_carrier->synchronize();
        transport_flux.relative = relative_carrier->read();
    }

    const auto maximum = [&](double value)
    {
        double global = 0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &value, &global);
        return global;
    };
    double local_limit = 0, limit = 0;
    const auto update_substep_limit = [&]
    {
        validate("QBMM linear all-face CFL limit", [&] { local_limit = local_substep_limit(stencils, cap, planar); });
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local_limit, &limit);
    };
    const auto rebuild_stencils = [&](const StageGeometry& geometry)
    {
        validate("QBMM linear planar stage geometry",
            [&]
            {
                const auto execution = acquire_mesh_execution(mesh);
                cell_stencils(mesh, transport_flux, geometry, limiter, tvd_mc, stencils, displacements, stencil_fit);
                trace_scratch.resize(columns.block, stencils, tvd_mc);
            });
    };
    if (planar)
        rebuild_stencils(StageGeometry(mesh, planar, dt, dt));
    else
        update_substep_limit();

    Flow::State start;
    double remaining = dt;
    while (remaining > 0)
    {
        if (result.substeps >= max_substeps)
            throw std::runtime_error("QBMM linear transport exhausted its subcycle cap.");
        if (planar)
            update_substep_limit();
        double h = std::min(remaining, limit);
        for (;;)
        {
            h = exact_interval_substep(remaining, h);
            require(h > 0 && (h == remaining || remaining - h < remaining),
                "QBMM linear subcycle has no representable time progress.");
            double local_used = 0;
            validate("QBMM linear fractions actually used",
                [&]
                {
                    const StageGeometry end_geometry(mesh, planar, dt, remaining - h);
                    local_used = local_cfl(stencils, h, cap, planar ? &end_geometry : nullptr);
                });
            const double used = maximum(local_used);
            if (used <= cap)
            {
                result.maximum_cfl = std::max(result.maximum_cfl, used);
                break;
            }
            h = std::nextafter(h, 0.);
        }

        validate("QBMM linear SSPRK2 substep setup",
            [&]
            {
                start = candidate;
                std::ranges::fill(step, StepTransfer{});
                result.substep_durations.push_back(h);
            });
        for (int rk_stage = 0; rk_stage < 2; ++rk_stage)
        {
            // The first Euler stock belongs to the end geometry. That stencil is also
            // the start geometry for the next substep after the extensive RK average.
            if (planar && rk_stage == 1)
                rebuild_stencils(StageGeometry(mesh, planar, dt, remaining - h));
            validate("QBMM linear owned stage stocks",
                [&] { stage_stocks(options, candidate, phases, columns.block, stocks->write(), before); });
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
                            donor_traces->getLocalViewHost(Tpetra::Access::ReadWrite), h, dt, limiter, tvd_mc,
                            trace_scratch);
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
                    stage_packets(mesh, options, transport_flux, phases, columns,
                        canonical_traces->getLocalViewHost(Tpetra::Access::ReadOnly), packets->write(), faces, caloric,
                        energy, h, tvd_mc, returned_packets, trace_scratch.center, trace_scratch.transfer);
                });
            packets->synchronize();
            validate("QBMM linear represented forward-Euler endpoint",
                [&]
                {
                    apply_stage(mesh, options, candidate, phases, columns.block, packets->read(), step, result, before,
                        after, stage_transfer);
                });
        }
        validate("QBMM linear represented SSPRK2 endpoint",
            [&]
            {
                complete_substep(options, start, candidate, phases, columns.block, step, volumes, result, before,
                    second_euler, after);
            });
        ++result.substeps;
        remaining -= h;
    }

    validate("QBMM linear interval receipts and work volumes",
        [&]
        {
            Accumulator interval;
            interval += dt;
            for (const auto h : result.substep_durations)
                interval += -h;
            require(interval.value() == 0., "QBMM linear substeps do not cover the requested interval exactly.");
            for (size_t c = 0; c < nc; ++c)
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto phase = phases[a];
                    const auto& cell = volumes[c * phases.size() + a];
                    const auto& state = candidate.cells[c];
                    if (*state.phase_numbers[phase] == 0)
                    {
                        require(state.phases[phase].internal_energy == 0 &&
                                    relative_difference(cell.work_volume.value(), cell.gross_volume.value()) <=
                                        2 * qbmm_reconstruction_tolerance,
                            "An empty QBMM linear endpoint has inconsistent energy or work volume.");
                        result.advected_material_volumes[c][phase] = 0.;
                    }
                    else
                    {
                        require(cell.work_volume.value() > 0., "QBMM linear material work volume is nonpositive.");
                        result.advected_material_volumes[c][phase] = represented(cell.work_volume.value());
                    }
                }
            for (size_t f = 0; f < nf; ++f)
            {
                Accumulator forward, backward, net_mass, theta, material_volume;
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto& interval = faces[f * phases.size() + a];
                    forward.append(interval.momentum[0][Mass]);
                    backward.append(interval.momentum[1][Mass]);
                    net_mass.append(interval.momentum[0][Mass]);
                    net_mass.append(interval.momentum[1][Mass]);
                    theta.append(interval.theta);
                    material_volume.append(interval.direction[0][W]);
                    material_volume.append(interval.direction[1][W]);
                    result.owned_flux[phases[a]][f] =
                        face_receipt(interval, dt, transport_flux.local_value(static_cast<LO>(f)));
                }
                auto& physical = result.integrated_faces[f];
                physical.forward_mass = rounded_rate(forward.value());
                physical.backward_mass = rounded_rate(-backward.value());
                physical.net_mass = rounded_rate(net_mass.value());
                physical.theta = rounded_rate(theta.value());
                physical.material_volume = rounded_rate(material_volume.value());
            }
            result.local_caloric_transport_magnitude = represented(caloric.value());
            result.local_energy_transport_magnitude = represented(energy.value());
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
