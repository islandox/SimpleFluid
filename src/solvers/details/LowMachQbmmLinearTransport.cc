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
};

struct CellStencil
{
    double volume = 0.;
    FaceSum outgoing_rate;
    struct Neighbor
    {
        LO cell;
        double volume;
    };
    std::vector<Neighbor> neighbors;
    std::vector<FaceStencil> traces;
};

std::vector<CellStencil> cell_stencils(
    const Mesh& mesh, const Flow::Traits::scalar_face_type& flux, TraceLimiter limiter)
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
            displacements.push_back(mesh.cell_centroid(neighbor) - center);
            stencil.neighbors.push_back({neighbor, neighbor_volume});
        }
        const FaceOffsetFit fit(displacements);
        for (const auto face : mesh.faces(cell))
        {
            if (mesh.is_geometry_exterior_face(face))
                continue;
            const double phi = flux.local_value(face);
            const size_t side = mesh.owner_cell(face) == cell ? 0 : 1;
            const double outward = side == 0 ? phi : -phi;
            require(std::isfinite(outward), "Nonfinite QBMM linear overlap carrier flux.");
            if (outward < 0 && limiter == TraceLimiter::Outgoing)
                continue;
            auto& wanted = stencil.traces.emplace_back(FaceStencil{face, side, outward, {}});
            if (outward > 0.)
                stencil.outgoing_rate.add_product(outward, 1.);
            const bool resolved = fit.weights(mesh.face_centroid(face) - center, wanted.weights);
            require(
                resolved || outward <= 0, "QBMM linear one-ring geometry cannot reproduce this outgoing face offset.");
            if (!resolved)
                wanted.weights.clear(); // A non-transporting side may use its cell-center trace.
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

    static bool at_most(const Stock& lhs, const Stock& rhs)
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
        excess.add_weighted(scale, -2 * eps);
        return excess.sign() <= 0;
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

template<class View> Block read_block(const View& view, LO row, size_t first, size_t width)
{
    Block block(width);
    for (size_t q = 0; q < width; ++q)
        block[q] = view(row, first + q);
    return block;
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

std::vector<Block> limit_traces(const QbmmPopulationSchema& family, const CellStencil& stencil, const Block& center,
    const std::vector<Block>& neighbors, double h, TraceLimiter limiter)
{
    EnergyBounds energy;
    energy.include(family, center);
    for (const auto& neighbor : neighbors)
        energy.include(family, neighbor);
    std::vector<Block> traces(stencil.traces.size(), Block(center.size()));
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

    FaceSum retained_volume;
    retained_volume.add_product(stencil.volume, 1.);
    retained_volume.add_weighted(stencil.outgoing_rate, -h);
    require(retained_volume.sign() > 0, "QBMM linear all-face donor retention must be positive.");

    const auto reconstruct = [&](size_t face, double factor)
    {
        auto& trace = traces[face];
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
    };
    const auto certify_donor = [&](double factor)
    {
        std::vector<FaceSum> retained(center.size());
        for (size_t q = 0; q < center.size(); ++q)
            retained[q].add_product(center[q], 1.);
        for (size_t f = 0; f < traces.size(); ++f)
        {
            const auto outward = stencil.traces[f].outward;
            // The forced corrector certifies every potential donor trace with the same factor.
            // A vanishing rate can change direction without changing the set of trace constraints.
            if (outward > 0. || limiter == TraceLimiter::AllIncident)
                reconstruct(f, factor);
            if (outward <= 0.)
                continue;
            const auto transfer = withdrawal(family, traces[f], h, outward, stencil.volume);
            for (size_t q = 0; q < center.size(); ++q)
                retained[q].add_product(-transfer[q], 1.);
        }

        // This is (V - sum(h*phi))*R, the retained part expressed in canonical
        // extensive units. Positive scaling does not alter either certificate.
        // It avoids independently rounding density channels at rank changes.
        Block remainder(center.size());
        for (size_t q = 0; q < center.size(); ++q)
            if (q != H1 && q != W)
                remainder[q] = represented(retained[q]);
        check_realizable(family, remainder, "retained donor stock");
        energy.check(family, remainder);
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
    double h, TraceLimiter limiter)
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
            const auto limited = limit_traces(family, stencil, center, neighbors, h, limiter);
            for (size_t f = 0; f < stencil.traces.size(); ++f)
            {
                const auto& face = stencil.traces[f];
                if (face.outward < 0.)
                    continue;
                const auto first = (a * sides + face.side) * columns.width();
                const auto& trace = limited[f];
                auto thermal = donor.phases[phase];
                thermal.moles.assign(moles(trace).begin(), moles(trace).end());
                thermal.internal_energy = trace[U];
                thermal.absolute_pressure = pressure;
                const auto point = thermal_trace(thermal, true);
                require((point.material_mass != 0) == (trace[H0] != 0),
                    "QBMM linear trace count and EOS material presence differ.");
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
    std::vector<FaceInterval>& intervals, FaceSum& caloric, FaceSum& energy, double h)
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
                require(traces(face, first + columns.marker()) == (selected ? 1. : 0.),
                    "QBMM linear face has no unique certified donor trace.");
                Block transfer(columns.block);
                if (selected)
                {
                    auto trace = read_block(traces, face, first, columns.block);
                    check_realizable(family, trace, "face trace");
                    if (trace[H0] != 0)
                    {
                        const auto density = traces(face, first + columns.density());
                        require(std::isfinite(density) && density > 0, "Invalid QBMM linear face EOS density.");
                        const auto donor_volume = traces(face, first + columns.volume());
                        const auto trace_mass = material_mass(family, moles(trace));
                        product({h, outward == 0 ? .25 : .5}, {donor_volume, density})
                            .add_weighted(interval.theta, trace_mass);
                        if (outward > 0)
                        {
                            transfer = withdrawal(family, trace, h, outward, donor_volume);
                            const auto mass = material_mass(family, moles(transfer));
                            FaceSum volume;
                            product({}, {density}).add_weighted(volume, mass);
                            transfer[W] = represented(volume);
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
    const auto validate = [&](const char* context, auto&& operation)
    { collective_detail::collective_local_validation(mesh, context, operation); };

    TransportResult result;
    std::vector<size_t> phases;
    std::vector<CellStencil> stencils;
    TraceColumns columns{};
    std::vector<FaceInterval> faces;
    std::vector<CellInterval> volumes;
    FaceSum caloric, energy;
    validate("QBMM linear transport setup",
        [&]
        {
            require(accepted.cells.size() == nc && candidate.cells.size() == nc &&
                        thermodynamic_guess.cells.size() == nc && accepted_material_volumes.size() == nc,
                "QBMM linear owned-cell layout differs.");
            require(std::isfinite(dt) && dt > 0 && cap > 0 && cap <= .5 && max_substeps > 0,
                "Invalid QBMM linear interval or CFL policy.");
            for (size_t p = 0; p < np; ++p)
                if (options.qbmm_families[p])
                {
                    phases.push_back(p);
                    columns.block = FirstSpecies + options.qbmm_families[p]->catalog().species().size();
                }
            require(!phases.empty(), "QBMM linear mode has no bound phase.");
            stencils = cell_stencils(mesh, carrier_flux, limiter);
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
                            donor_traces->getLocalViewHost(Tpetra::Access::ReadWrite), h, limiter);
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
                        energy, h);
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
    return result;
}
} // namespace SimpleFluid::Reacting::qbmm_linear_detail
