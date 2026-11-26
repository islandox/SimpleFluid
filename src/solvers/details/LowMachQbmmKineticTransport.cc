#include "solvers/details/LowMachQbmmKineticTransport.hh"
#include "equations/CollectiveValidation.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"
#include "solvers/details/LowMachQbmmState.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Reacting::qbmm_kinetic_detail
{
namespace
{
using namespace qbmm_explicit_detail;
using Evaluation = QbmmApplicabilityEvaluation;

constexpr size_t nodes = 2;

// Index the columns shared by all kinetic families in each synchronized exchange.
enum TraceColumn : size_t
{
    BulkVolume,
    Rank,
    Density,
    MaterialMass,
    Energy,
    CaloricSpecific,
    Carrier,
    FirstAtom = Carrier + dimensions
};
enum AtomColumn : size_t
{
    Weight,
    Support,
    Slip,
    AtomColumns = Slip + dimensions
};
constexpr size_t trace_columns = FirstAtom + nodes * AtomColumns;

Vector make_vector(double x, double y, double z)
{
    Vector result;
    result.component(0) = x;
    result.component(1) = y;
    result.component(2) = z;
    return result;
}

template<class View> Vector trace_vector(const View& trace, LO cell, size_t first)
{
    return make_vector(trace(cell, first), trace(cell, first + 1), trace(cell, first + 2));
}

template<class View> void set_trace_vector(const View& trace, LO cell, size_t first, const Vector& vector)
{
    for (size_t axis = 0; axis < dimensions; ++axis)
        trace(cell, first + axis) = vector.component(axis);
}

struct LawTrace
{
    Flow::QbmmKineticClosure closure;
    std::array<Evaluation, nodes> observations;
};

struct TraceScratch
{
    Flow::CellState donor;
    std::vector<StageThermal> thermal;
    std::vector<double> volumes, composition;
};

template<class View>
void stage_traces(const Mesh& mesh, const Flow::Options& options, const Flow::State& accepted, const Flow::State& guess,
    const Flow::State& candidate, const std::vector<size_t>& phases, const View& trace,
    std::vector<LawTrace>& law_traces, TraceScratch& scratch)
{
    auto& [donor, thermal, volumes, composition] = scratch;
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const double bulk_volume = mesh.cell_volume(cell);
        const auto location = qbmm_native_law_detail::kinetic_location(mesh, cell);
        donor = guess.cells[c];
        const auto& stock = candidate.cells[c];
        const double pressure = guess.thermodynamic_pressure + donor.mechanical_pressure;
        for (size_t p = 0; p < donor.phases.size(); ++p)
        {
            auto& phase = donor.phases[p];
            bool occupied = std::ranges::any_of(phase.moles, [](double n) { return n != 0; });
            phase.absolute_pressure = pressure;
            if (options.qbmm_families[p])
            {
                occupied = *accepted.cells[c].phase_numbers[p] > 0;
                phase.moles = stock.phases[p].moles;
                phase.internal_energy = stock.phases[p].internal_energy;
                donor.phase_numbers[p] = stock.phase_numbers[p];
                donor.qbmm_shape[p] = stock.qbmm_shape[p];
            }
            thermal[p] = thermal_trace(phase, !occupied || !(phase.temperature > 0), composition);
            volumes[p] = thermal[p].volume;
        }
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const size_t phase = phases[a], column = a * trace_columns;
            const auto& family = *options.qbmm_families[phase];
            const auto quadrature = invert_qbmm_moments(
                qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, donor, phase)));
            require(quadrature.valid(), "QBMM kinetic donor reconstruction failed.");
            auto& law = law_traces[c * phases.size() + a];
            law.closure = qbmm_native_law_detail::evaluate_node_slip(
                options, mesh, donor, phase, bulk_volume, volumes, location, &law.observations);
            require(law.closure.active_atoms == quadrature.active_atoms, "QBMM kinetic donor atom layout differs.");
            trace(cell, column + BulkVolume) = bulk_volume;
            trace(cell, column + Rank) = quadrature.active_atoms;
            trace(cell, column + Density) = thermal[phase].density;
            trace(cell, column + MaterialMass) = thermal[phase].material_mass;
            trace(cell, column + Energy) = donor.phases[phase].internal_energy;
            trace(cell, column + CaloricSpecific) = thermal[phase].caloric_specific;
            set_trace_vector(trace, cell, column + Carrier, donor.velocity);
            for (size_t i = 0; i < nodes; ++i)
            {
                const size_t atom = column + FirstAtom + i * AtomColumns;
                trace(cell, atom + Weight) = quadrature.atoms[i].count;
                trace(cell, atom + Support) = quadrature.atoms[i].mass_over_scale;
                set_trace_vector(trace, cell, atom + Slip, law.closure.node_velocities[i]);
            }
        }
    }
}

size_t outgoing_column(size_t family, size_t direction, size_t atom)
{
    return (family * sides + direction) * nodes + atom;
}

double active_speed(double speed, size_t direction)
{
    return std::max(0., direction == 0 ? speed : -speed);
}

struct UnresolvedOffset
{
    LO face;
    size_t side;
};

std::vector<UnresolvedOffset> unresolved_offsets(const Mesh& mesh)
{
    std::vector<UnresolvedOffset> unresolved;
    std::vector<Mesh::Vec3> displacements;
    std::vector<double> weights;
    FaceOffsetFit fit;
    size_t maximum_faces = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
        maximum_faces = std::max(maximum_faces, static_cast<size_t>(mesh.faces(static_cast<LO>(c)).size()));
    displacements.reserve(maximum_faces);
    weights.reserve(maximum_faces);
    fit.reserve(maximum_faces);
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto center = mesh.cell_centroid(cell);
        displacements.clear();
        for (const auto face : mesh.faces(cell))
            if (!mesh.is_geometry_exterior_face(face))
                displacements.push_back(mesh.cell_center_vector(face, cell));
        fit.assign(displacements);
        for (const auto face : mesh.faces(cell))
            if (!mesh.is_geometry_exterior_face(face) && !fit.weights(mesh.face_centroid(face) - center, weights))
                unresolved.push_back({face, mesh.owner_cell(face) == cell ? 0u : 1u});
    }
    return unresolved;
}

template<class Trace, class Speeds>
void stage_speeds(const Mesh& mesh, const Flow::Traits::scalar_face_type& carrier, const Trace& trace,
    const Speeds& speeds, size_t families)
{
    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const bool wall = mesh.is_geometry_exterior_face(face);
        const double phi = carrier.local_value(face);
        require(std::isfinite(phi) && (!wall || phi == 0), "QBMM kinetic carrier flux differs from a closed mesh.");
        const auto left = mesh.owner_cell(face);
        const auto right = wall ? left : mesh.opposite_or_periodic_neighbor_cell(face, left);
        const auto area = mesh.face_area_vector(face);
        for (size_t a = 0; a < families; ++a)
            for (size_t side = 0; side < sides; ++side)
            {
                const LO donor = side == 0 ? left : right;
                const size_t base = a * trace_columns;
                const auto rank = trace(donor, base + Rank);
                require(rank == 0 || rank == 1 || rank == 2, "Invalid exchanged QBMM kinetic atom mask.");
                for (size_t i = 0; i < nodes; ++i)
                {
                    double speed = 0;
                    if (!wall && i < rank)
                    {
                        const auto atom = base + FirstAtom + i * AtomColumns;
                        const auto slip = trace_vector(trace, donor, atom + Slip);
                        long double projected = phi;
                        for (size_t axis = 0; axis < dimensions; ++axis)
                        {
                            const long double velocity = slip.component(axis);
                            const long double measure = area.component(axis);
                            const long double contribution = velocity * measure;
                            const long double sum = projected + contribution;
                            // Preserve the sign of a nearly cancelling carrier and slip contribution.
                            projected = contribution != 0 && std::abs(sum) <= eps * std::abs(projected)
                                            ? std::fma(velocity, measure, projected)
                                            : sum;
                        }
                        speed = represented(projected);
                    }
                    speeds(face, outgoing_column(a, side, i)) = speed;
                }
            }
    }
}

template<class Speeds> double reserve(const Mesh& mesh, const Speeds& speeds, size_t families, double cfl)
{
    double local_limit = std::numeric_limits<double>::infinity();
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto volume = mesh.cell_volume(cell);
        require(std::isfinite(volume) && volume > 0, "Invalid QBMM kinetic bulk volume.");
        const long double cap_volume = static_cast<long double>(cfl) * volume;
        for (size_t a = 0; a < families; ++a)
            for (size_t i = 0; i < nodes; ++i)
            {
                Accumulator rate;
                for (const auto face : mesh.faces(cell))
                {
                    const size_t direction = mesh.owner_cell(face) == cell ? 0 : 1;
                    rate += active_speed(speeds(face, outgoing_column(a, direction, i)), direction);
                }
                const long double outgoing_rate = rate.value();
                require(std::isfinite(outgoing_rate), "Nonfinite QBMM kinetic outgoing rate.");
                if (outgoing_rate > 0)
                {
                    const long double duration = cap_volume / outgoing_rate;
                    if (duration <= std::numeric_limits<double>::max())
                    {
                        double bound = static_cast<double>(duration);
                        if (bound > duration)
                            bound = std::nextafter(bound, 0.);
                        local_limit = std::min(local_limit, bound);
                    }
                }
            }
    }
    return local_limit;
}

template<class Speeds>
double verify_reservations(
    const Mesh& mesh, const Speeds& speeds, size_t families, double h, double cap, std::vector<long double>& remainders)
{
    double local_maximum = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto volume = mesh.cell_volume(cell);
        for (size_t a = 0; a < families; ++a)
            for (size_t i = 0; i < nodes; ++i)
            {
                Accumulator used, remainder;
                remainder += 1.L;
                for (const auto face : mesh.faces(cell))
                {
                    const size_t direction = mesh.owner_cell(face) == cell ? 0 : 1;
                    const double speed = active_speed(speeds(face, outgoing_column(a, direction, i)), direction);
                    const long double fraction = static_cast<long double>(h) * speed / volume;
                    used += fraction;
                    remainder += -fraction;
                }
                auto excess = used;
                excess += -cap;
                const long double used_fraction = used.value();
                const double actual = rounded_rate(used_fraction);
                bool exceeds_cap = excess.value() > 0;
                long double retained = remainder.value();
                if (actual == cap && used_fraction > 0)
                {
                    // On the cap, retain product roundoff to distinguish a small
                    // withdrawal excess from complete or almost complete evacuation.
                    const long double cap_volume = static_cast<long double>(cap) * volume;
                    Accumulator margin;
                    margin += cap_volume;
                    margin += std::fma(static_cast<long double>(cap), volume, -cap_volume);
                    for (const auto face : mesh.faces(cell))
                    {
                        const size_t direction = mesh.owner_cell(face) == cell ? 0 : 1;
                        const double speed = active_speed(speeds(face, outgoing_column(a, direction, i)), direction);
                        if (speed == 0)
                            continue;
                        const long double swept_volume = static_cast<long double>(h) * speed;
                        margin += -swept_volume;
                        margin += -std::fma(static_cast<long double>(h), speed, -swept_volume);
                    }
                    const long double remaining_volume = margin.value();
                    exceeds_cap = remaining_volume < 0;
                    retained = (1.L - cap) + remaining_volume / volume;
                }
                remainders[(c * families + a) * nodes + i] = retained;
                local_maximum = std::max(local_maximum,
                    exceeds_cap ? std::nextafter(actual, std::numeric_limits<double>::infinity()) : actual);
            }
    }
    return local_maximum;
}

void fill_species_packet(const QbmmPopulationSchema& family, double scaled_mass, std::span<double> moles)
{
    // Use the family composition and the same roundoff budget as its owning packet API.
    constexpr double composition_tolerance = 64 * eps;
    const auto fractions = family.mass_fractions();
    const auto& species = family.catalog().species();
    const long double mass = static_cast<long double>(scaled_mass) * family.mass_scale_kg();
    for (size_t s = 0; s < moles.size(); ++s)
    {
        if (fractions[s] == 0)
        {
            moles[s] = 0;
            continue;
        }
        const long double amount = mass * fractions[s] / species[s].molarMass;
        require(std::isfinite(amount) && amount <= std::numeric_limits<double>::max(),
            "QBMM value exceeds the representable range.");
        moles[s] = rounded_rate(amount);
        require(moles[s] != 0, "A nonzero QBMM value underflows to zero.");
        const long double represented_moles = moles[s];
        require(std::abs(represented_moles - amount) <= composition_tolerance * std::max(represented_moles, amount),
            "The QBMM value cannot be represented within the roundoff budget.");
    }
    const double recovered = family.scaled_mass_change(moles);
    require(std::abs(static_cast<long double>(recovered) - scaled_mass) <=
                composition_tolerance * static_cast<long double>(std::max(recovered, scaled_mass)),
        "The QBMM packet cannot represent its requested mass.");
}

template<class Trace, class Speeds, class Packet>
void stage_packets(const Mesh& mesh, const Flow::Options& options, const std::vector<size_t>& phases,
    const Trace& trace, const Speeds& speeds, const Packet& packet, size_t packet_columns,
    std::vector<FaceInterval>& intervals, Accumulator& caloric, Accumulator& energy, double h,
    std::span<Accumulator> channels, std::span<double> moles)
{
    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const bool wall = mesh.is_geometry_exterior_face(face);
        const auto left = mesh.owner_cell(face);
        const auto right = wall ? left : mesh.opposite_or_periodic_neighbor_cell(face, left);
        const auto area = mesh.face_area_vector(face);
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto& family = *options.qbmm_families[phases[a]];
            const size_t base = a * trace_columns;
            auto& interval = intervals[f * phases.size() + a];
            for (size_t side = 0; side < sides; ++side)
            {
                const double sign = side == 0 ? 1. : -1.;
                const auto donor = side == 0 ? left : right;
                const auto rank = static_cast<size_t>(trace(donor, base + Rank));
                const size_t first = (a * sides + side) * packet_columns;
                std::ranges::fill(channels, Accumulator{});
                for (size_t i = 0; !wall && i < rank; ++i)
                {
                    const auto atom = base + FirstAtom + i * AtomColumns;
                    const double speed = speeds(face, outgoing_column(a, side, i));
                    const double outward = active_speed(speed, side);
                    if (outward == 0 && speed != 0)
                        continue;
                    const double weight = trace(donor, atom + Weight), support = trace(donor, atom + Support);
                    const auto slip = trace_vector(trace, donor, atom + Slip);
                    const double volume = trace(donor, base + BulkVolume);
                    const double density = trace(donor, base + Density);
                    const double scale = family.mass_scale_kg();
                    require(std::isfinite(volume) && volume > 0 && std::isfinite(density) && density > 0,
                        "Invalid QBMM kinetic packet divisor.");
                    const long double swept = static_cast<long double>(h) / volume;
                    const long double alpha = swept * (speed == 0 ? .5L : 1.L) * weight * support * scale / density;
                    interval.theta += alpha;
                    for (size_t axis = 0; axis < dimensions; ++axis)
                        interval.slip += alpha * slip.component(axis) * area.component(axis);
                    if (outward == 0)
                        continue;
                    const auto carrier = trace_vector(trace, donor, base + Carrier);
                    const double count = represented(static_cast<long double>(h) * outward / volume * weight);
                    const long double first_moment = static_cast<long double>(count) * support;
                    const long double second_moment = first_moment * support;
                    const double h1 = represented(first_moment);
                    channels[H0] += count;
                    channels[H1] += h1;
                    channels[H2] += represented(second_moment);
                    channels[H3] += represented(second_moment * support);
                    fill_species_packet(family, h1, moles);
                    auto& momentum = interval.momentum[side];
                    Accumulator material;
                    for (size_t s = 0; s < moles.size(); ++s)
                    {
                        channels[FirstSpecies + s] += moles[s];
                        const double molar_mass = family.catalog().species()[s].molarMass;
                        const long double species_mass = static_cast<long double>(moles[s]) * molar_mass;
                        material += species_mass;
                        if constexpr (std::numeric_limits<long double>::digits == std::numeric_limits<double>::digits)
                        {
                            // Preserve the product residual in the directional budget on binary64-only targets.
                            momentum[Mass] += sign * species_mass;
                            momentum[Mass] += sign * std::fma(moles[s], molar_mass, -static_cast<double>(species_mass));
                        }
                    }
                    const long double mass = material.value();
                    require(std::isfinite(mass) && mass > 0, "QBMM kinetic atom count and material transfer disagree.");
                    if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
                        momentum[Mass] += sign * mass;
                    const double donor_mass = trace(donor, base + MaterialMass);
                    require(std::isfinite(donor_mass) && donor_mass > 0, "Invalid QBMM kinetic packet divisor.");
                    const long double energy_packet =
                        static_cast<long double>(trace(donor, base + Energy)) / donor_mass * mass;
                    channels[U] += energy_packet;
                    energy += std::abs(energy_packet);
                    channels[W] += mass / density;
                    caloric += mass * trace(donor, base + CaloricSpecific);
                    for (size_t axis = 0; axis < dimensions; ++axis)
                    {
                        momentum[CarrierMomentum + axis] += sign * mass * carrier.component(axis);
                        momentum[DriftMomentum + axis] += sign * mass * slip.component(axis);
                    }
                }
                for (size_t q = 0; q < packet_columns; ++q)
                {
                    const double amount = represented(channels[q].value());
                    packet(face, first + q) = amount;
                    interval.direction[side][q] += sign * amount;
                }
            }
        }
    }
}

void compare(long double actual, long double constructive)
{
    require(relative_difference(actual - constructive, actual + constructive) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM kinetic canonical endpoint disagrees with the constructive measure.");
}

double updated_amount(double before, long double incoming, long double outgoing, const Accumulator& net, bool depleted,
    bool nonnegative, double tolerance, double& maximum_residual)
{
    auto result = net;
    result += before;
    // A completely exhausted reconstructed measure can differ from the stored
    // donor by its inversion/product roundoff. Check that discrepancy against
    // the donor before adopting the incoming measure, including exact vacuum.
    if (depleted)
    {
        const long double remainder = before - outgoing;
        const long double donor = std::abs(before) + std::abs(outgoing);
        require(relative_difference(remainder, donor) <= 2 * qbmm_reconstruction_tolerance,
            "QBMM kinetic complete evacuation disagrees with the authoritative stock.");
    }
    else
        require(!nonnegative || result.value() >= 0, "QBMM kinetic withdrawal exceeds its canonical stock.");
    const double actual = represented(depleted ? incoming : result.value());
    if (actual == before && before != 0)
    {
        const bool withdrawal = before > 0 ? net.value() < 0 : net.value() > 0;
        require(!withdrawal, "QBMM kinetic net transfer is too small to update its authoritative stock.");
    }
    auto defect = result;
    defect += -actual;
    const long double magnitude = static_cast<long double>(std::abs(before)) + std::abs(actual) +
                                  std::abs(incoming) + std::abs(outgoing);
    const double residual = relative_difference(defect.value(), magnitude) / tolerance;
    require(residual <= 1, "QBMM kinetic component fails its canonical cell/face balance.");
    maximum_residual = std::max(maximum_residual, residual);
    return actual;
}

template<class Trace, class Packet>
void apply_packets(const Mesh& mesh, const Flow::Options& options, Flow::State& candidate,
    const std::vector<size_t>& phases, const Trace& trace, const Packet& packet, size_t packet_columns,
    const std::vector<long double>& remainders, std::vector<CellInterval>& intervals, TransportResult& result,
    std::array<std::vector<Accumulator>, 3>& channels)
{
    auto& [incoming, outgoing, net] = channels;
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        auto& stock = candidate.cells[c];
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            const auto rank = static_cast<size_t>(trace(cell, a * trace_columns + Rank));
            for (auto& channel : channels)
                std::ranges::fill(channel, Accumulator{});
            for (const auto face : mesh.faces(cell))
            {
                const size_t side = mesh.owner_cell(face) == cell ? 0 : 1;
                const size_t out = (a * sides + side) * packet_columns;
                const size_t in = (a * sides + 1 - side) * packet_columns;
                for (size_t q = 0; q < packet_columns; ++q)
                {
                    const double credit = packet(face, in + q);
                    const double debit = packet(face, out + q);
                    incoming[q] += credit;
                    outgoing[q] += debit;
                    net[q] += credit;
                    net[q] += -debit;
                }
            }
            if (incoming[H0].value() == 0 && outgoing[H0].value() == 0)
                continue;
            std::array<Accumulator, 4> constructive;
            bool exhausted = rank > 0;
            for (size_t i = 0; i < rank; ++i)
            {
                const auto remainder = remainders[(c * phases.size() + a) * nodes + i];
                require(remainder >= 0, "QBMM kinetic constructive donor remainder is negative.");
                exhausted &= remainder == 0;
                const size_t atom = a * trace_columns + FirstAtom + i * AtomColumns;
                const double support = trace(cell, atom + Support);
                long double moment = remainder * trace(cell, atom + Weight);
                for (auto& channel : constructive)
                {
                    channel += moment;
                    moment *= support;
                }
            }
            for (size_t k = 0; k < constructive.size(); ++k)
                constructive[k].append(incoming[k]);
            const auto before =
                qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, stock, phase));
            auto& shape = *stock.qbmm_shape[phase];
            auto& thermal = stock.phases[phase];
            const double tolerance = options.conservation_relative_tolerance;
            const auto update = [&](size_t channel, double before_amount, bool nonnegative, double& residual)
            {
                return updated_amount(before_amount, incoming[channel].value(), outgoing[channel].value(), net[channel],
                    exhausted, nonnegative, tolerance, residual);
            };
            *stock.phase_numbers[phase] = update(H0, before[0], true, result.maximum_population_residual);
            shape.H2 = update(H2, before[2], true, result.maximum_shape_residual);
            shape.H3 = update(H3, before[3], true, result.maximum_shape_residual);
            thermal.internal_energy = update(U, thermal.internal_energy, false, result.maximum_energy_residual);
            for (size_t s = 0; s < thermal.moles.size(); ++s)
                thermal.moles[s] = update(FirstSpecies + s, thermal.moles[s], true, result.maximum_species_residual);
            const auto after =
                qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, stock, phase));
            for (size_t k = 0; k < constructive.size(); ++k)
                compare(after[k], constructive[k].value());
            const auto quadrature = invert_qbmm_moments(after);
            if (!quadrature.valid())
                throw std::invalid_argument(
                    std::string("QBMM kinetic compression failed at the represented endpoint (") +
                    qbmm_inversion_status_name(quadrature.status) + ").");
            auto& interval = intervals[c * phases.size() + a];
            interval.work_volume.append(net[W]);
            interval.gross_volume.append(incoming[W]);
            interval.gross_volume.append(outgoing[W]);
        }
    }
}

} // namespace

TransportResult transport(const SP<const Flow::Mesh>& grid, const Flow::Options& options, const Flow::State& accepted,
    const Flow::State& thermodynamic_guess, Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector)
{
    const auto& mesh = *grid;
    const auto owned_cells = mesh.owned_cell_map(), overlap_cells = mesh.overlap_cell_map();
    const auto owned_faces = mesh.owned_face_map(), overlap_faces = mesh.overlap_face_map();
    const auto& comm = *owned_cells->getComm();
    const size_t nphase = options.qbmm_families.size(), nc = mesh.num_owned_cells(), nf = mesh.num_owned_faces();
    std::vector<size_t> phases;
    size_t packet_columns = 0;
    TransportResult result;
    TraceScratch trace_scratch;
    std::vector<LawTrace> laws;
    std::vector<long double> remainders;
    std::vector<FaceInterval> faces;
    std::vector<CellInterval> cells;
    std::vector<Accumulator> face_channels;
    std::vector<double> species_packet;
    std::array<std::vector<Accumulator>, 3> cell_channels;
    Accumulator caloric, energy;
    const double cap = options.qbmm_model.kinetic_cfl;
    const int max_substeps = options.qbmm_model.maximum_kinetic_substeps;
    std::vector<UnresolvedOffset> unresolved;
    collective_detail::collective_local_validation(mesh, "QBMM kinetic transport setup",
        [&]
        {
            require(accepted.cells.size() == nc && candidate.cells.size() == nc &&
                        thermodynamic_guess.cells.size() == nc && accepted_material_volumes.size() == nc,
                "QBMM kinetic owned-cell layout differs.");
            require(std::isfinite(dt) && dt > 0 && std::isfinite(cap) && cap > 0 && cap <= 1 && max_substeps > 0,
                "Invalid QBMM kinetic interval or CFL policy.");
            if (qbmm_detail::uses_axial_kinetic_transport(options, mesh))
            {
                qbmm_detail::validate_capability(options, mesh, false, {}, false);
                unresolved = unresolved_offsets(mesh);
            }
            for (size_t p = 0; p < nphase; ++p)
                if (options.qbmm_families[p])
                {
                    phases.push_back(p);
                    packet_columns = FirstSpecies + options.qbmm_families[p]->catalog().species().size();
                }
            require(!phases.empty(), "QBMM kinetic mode has no bound phase.");
            result.owned_flux.resize(nphase);
            result.advected_material_volumes = accepted_material_volumes;
            result.integrated_faces.resize(nf);
            faces.resize(nf * phases.size());
            cells.resize(nc * phases.size());
            laws.resize(nc * phases.size());
            remainders.resize(nc * phases.size() * nodes);
            trace_scratch.thermal.resize(nphase);
            trace_scratch.volumes.resize(nphase);
            if (nc != 0)
            {
                trace_scratch.donor = thermodynamic_guess.cells.front();
                size_t maximum_species = 0;
                for (const auto& phase : trace_scratch.donor.phases)
                    maximum_species = std::max(maximum_species, phase.moles.size());
                trace_scratch.composition.reserve(maximum_species);
            }
            face_channels.resize(packet_columns);
            species_packet.resize(packet_columns - FirstSpecies);
            for (auto& channel : cell_channels)
                channel.resize(packet_columns);
            for (const auto p : phases)
            {
                require(options.qbmm_families[p]->catalog().species().size() == species_packet.size(),
                    "QBMM kinetic family catalog sizes differ.");
                result.owned_flux[p].resize(nf);
            }
            for (auto& face : faces)
                for (auto& direction : face.direction)
                    direction.resize(packet_columns);
            for (size_t c = 0; c < nc; ++c)
            {
                require(thermodynamic_guess.cells[c].phases.size() == nphase &&
                            accepted_material_volumes[c].size() == nphase,
                    "QBMM kinetic phase or material-volume layout differs.");
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const double volume = accepted_material_volumes[c][phases[a]];
                    require(std::isfinite(volume) && volume >= 0, "Invalid QBMM kinetic accepted material volume.");
                    auto& cell = cells[c * phases.size() + a];
                    cell.work_volume += volume;
                    cell.gross_volume += volume;
                }
            }
        });
    // The imports below are private to this invocation, tied to these maps,
    // this immutable family/catalog order, and the synchronized substep loop.
    collective_detail::require_uniform_value(mesh, corrector, "QBMM kinetic corrector index");
    collective_detail::require_uniform_value(mesh, dt, "QBMM kinetic requested interval");
    collective_detail::require_uniform_value(mesh, cap, "QBMM kinetic CFL cap");
    collective_detail::require_uniform_value(mesh, max_substeps, "QBMM kinetic substep cap");
    collective_detail::require_uniform_value(mesh, phases.size(), "QBMM kinetic family count");
    for (const auto phase : phases)
    {
        const auto& family = *options.qbmm_families[phase];
        collective_detail::require_uniform_value(mesh, phase, "QBMM kinetic phase index");
        collective_detail::require_uniform_value(mesh, family.fingerprint(), "QBMM kinetic family fingerprint");
        collective_detail::require_uniform_value(
            mesh, family.catalog().fingerprint(), "QBMM kinetic catalog fingerprint");
    }
    const Import cell_import(owned_cells, overlap_cells), face_import(owned_faces, overlap_faces);
    std::optional<Overlap> traces, speeds, packets;
    collective_detail::collective_local_validation(mesh, "QBMM kinetic owner/ghost buffers",
        [&]
        {
            traces.emplace(cell_import, owned_cells, overlap_cells, phases.size() * trace_columns);
            speeds.emplace(face_import, owned_faces, overlap_faces, phases.size() * sides * nodes);
            packets.emplace(face_import, owned_faces, overlap_faces, phases.size() * sides * packet_columns);
        });

    const auto maximum = [&](double value)
    {
        double global = 0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &value, &global);
        return global;
    };
    double remaining = dt;
    while (remaining > 0)
    {
        if (result.substeps >= max_substeps)
            throw std::runtime_error("QBMM kinetic transport exhausted its subcycle cap.");
        collective_detail::collective_local_validation(mesh, "QBMM kinetic owned donor laws and caloric traces",
            [&]
            {
                stage_traces(mesh, options, accepted, thermodynamic_guess, candidate, phases, traces->write(), laws,
                    trace_scratch);
            });
        traces->synchronize();
        const auto trace = traces->read();
        collective_detail::collective_local_validation(mesh, "QBMM kinetic canonical face speeds",
            [&] { stage_speeds(mesh, carrier_flux, trace, speeds->write(), phases.size()); });
        speeds->synchronize();
        const auto speed = speeds->read();
        double local_limit = std::numeric_limits<double>::infinity();
        collective_detail::collective_local_validation(mesh, "QBMM kinetic all-face owner reserved rates",
            [&]
            {
                for (const auto& offset : unresolved)
                    for (size_t a = 0; a < phases.size(); ++a)
                        for (size_t i = 0; i < nodes; ++i)
                            require(
                                active_speed(speed(offset.face, outgoing_column(a, offset.side, i)), offset.side) == 0.,
                                "QBMM kinetic one-ring geometry cannot reproduce this outgoing face offset.");
                local_limit = reserve(mesh, speed, phases.size(), cap);
            });
        // The minimum of cap / max_local(sum |q| / V) is the same limit as
        // the global maximum outgoing rate. Reducing the reciprocal bound
        // keeps an otherwise representable duration when q/V overflows.
        double limit = 0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local_limit, &limit);
        double h = std::min(remaining, limit);
        for (;;)
        {
            h = exact_interval_substep(remaining, h);
            require(h > 0 && (h == remaining || remaining - h < remaining),
                "QBMM kinetic subcycle has no representable time progress.");
            double local_cfl = 0;
            collective_detail::collective_local_validation(mesh, "QBMM kinetic fractions actually used",
                [&] { local_cfl = verify_reservations(mesh, speed, phases.size(), h, cap, remainders); });
            const double used = maximum(local_cfl);
            if (used <= cap)
            {
                result.maximum_cfl = std::max(result.maximum_cfl, used);
                break;
            }
            h = std::nextafter(h, 0.);
        }
        collective_detail::collective_local_validation(mesh, "QBMM kinetic per-atom interval applicability",
            [&]
            {
                for (size_t c = 0; c < nc; ++c)
                    for (size_t a = 0; a < phases.size(); ++a)
                    {
                        const auto phase = phases[a];
                        auto& law = laws[c * phases.size() + a];
                        qbmm_native_law_detail::validate_node_slip_interval(
                            options, phase, law.closure, h, &law.observations);
                        const auto cell = static_cast<LO>(c);
                        for (size_t i = 0; i < law.closure.active_atoms; ++i)
                        {
                            const auto atom = a * trace_columns + FirstAtom + i * AtomColumns;
                            const double mass_kg = represented(static_cast<long double>(trace(cell, atom + Support)) *
                                                               options.qbmm_families[phase]->mass_scale_kg());
                            result.law_receipts.push_back(
                                {mesh.cell_geometry_global_id(cell), phase, static_cast<size_t>(result.substeps), i, h,
                                    mass_kg, law.closure.node_velocities[i], law.observations[i]});
                        }
                    }
                result.substep_durations.push_back(h);
            });
        collective_detail::collective_local_validation(mesh, "QBMM kinetic canonical directional packets",
            [&]
            {
                stage_packets(mesh, options, phases, trace, speed, packets->write(), packet_columns, faces, caloric,
                    energy, h, face_channels, species_packet);
            });
        packets->synchronize();
        collective_detail::collective_local_validation(mesh, "QBMM kinetic constructive compression and balances",
            [&]
            {
                apply_packets(mesh, options, candidate, phases, trace, packets->read(), packet_columns, remainders,
                    cells, result, cell_channels);
            });
        ++result.substeps;
        remaining -= h;
    }
    collective_detail::collective_local_validation(mesh, "QBMM kinetic interval receipts and work volumes",
        [&]
        {
            Accumulator covered;
            covered += dt;
            for (const auto h : result.substep_durations)
                covered += -h;
            require(covered.value() == 0, "QBMM kinetic substeps do not cover the requested interval exactly.");
            for (size_t c = 0; c < nc; ++c)
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto phase = phases[a];
                    const auto& cell = cells[c * phases.size() + a];
                    const auto& phase_state = candidate.cells[c].phases[phase];
                    if (*candidate.cells[c].phase_numbers[phase] == 0)
                    {
                        require(phase_state.internal_energy == 0 &&
                                    relative_difference(cell.work_volume.value(), cell.gross_volume.value()) <=
                                        2 * qbmm_reconstruction_tolerance,
                            "Locally empty kinetic phase has inconsistent canonical energy/work-volume packets.");
                        result.advected_material_volumes[c][phase] = 0;
                    }
                    else
                        result.advected_material_volumes[c][phase] = represented(cell.work_volume.value());
                }
            for (size_t f = 0; f < nf; ++f)
            {
                std::array<Accumulator, sides> mass;
                std::array<Accumulator, dimensions> drift;
                Accumulator theta, slip, material_volume;
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto& interval = faces[f * phases.size() + a];
                    theta.append(interval.theta);
                    slip.append(interval.slip);
                    for (size_t side = 0; side < sides; ++side)
                    {
                        material_volume.append(interval.direction[side][W]);
                        mass[side].append(interval.momentum[side][Mass]);
                        for (size_t axis = 0; axis < dimensions; ++axis)
                            drift[axis].append(interval.momentum[side][DriftMomentum + axis]);
                    }
                    result.owned_flux[phases[a]][f] =
                        face_receipt(interval, dt, carrier_flux.local_value(static_cast<LO>(f)));
                }
                auto net_mass = mass[0];
                net_mass.append(mass[1]);
                auto& momentum = result.integrated_faces[f];
                momentum.forward_mass = rounded_rate(mass[0].value());
                momentum.backward_mass = rounded_rate(-mass[1].value());
                momentum.net_mass = rounded_rate(net_mass.value());
                momentum.theta = rounded_rate(theta.value());
                momentum.slip = rounded_rate(slip.value());
                momentum.material_volume = rounded_rate(material_volume.value());
                for (size_t axis = 0; axis < dimensions; ++axis)
                    momentum.drift[axis] = rounded_rate(drift[axis].value());
            }
            result.local_caloric_transport_magnitude = represented(caloric.value());
            result.local_energy_transport_magnitude = represented(energy.value());
        });
    validate_closed_balances(
        mesh, options, accepted, candidate, phases, packet_columns, accepted_material_volumes, result);
    result.maximum_species_residual = maximum(result.maximum_species_residual);
    result.maximum_population_residual = maximum(result.maximum_population_residual);
    result.maximum_shape_residual = maximum(result.maximum_shape_residual);
    result.maximum_energy_residual = maximum(result.maximum_energy_residual);
    return result;
}
} // namespace SimpleFluid::Reacting::qbmm_kinetic_detail
