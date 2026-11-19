#include "solvers/details/LowMachQbmmKineticTransport.hh"
#include "equations/CollectiveValidation.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
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

template<class View>
void stage_traces(const Mesh& mesh, const Flow::Options& options, const Flow::State& accepted, const Flow::State& guess,
    const Flow::State& candidate, const std::vector<size_t>& phases, const View& trace,
    std::vector<LawTrace>& law_traces)
{
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const double bulk_volume = mesh.cell_volume(cell);
        const auto location = qbmm_native_law_detail::kinetic_location(mesh, cell);
        auto donor = guess.cells[c];
        const auto& stock = candidate.cells[c];
        std::vector<StageThermal> thermal(donor.phases.size());
        std::vector<double> volumes(donor.phases.size());
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
            thermal[p] = thermal_trace(phase, !occupied || !(phase.temperature > 0));
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
                options, donor, phase, bulk_volume, volumes, location, &law.observations);
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
                        FaceSum total;
                        total.add_product(phi, 1.);
                        for (size_t axis = 0; axis < dimensions; ++axis)
                            total.add_product(slip.component(axis), area.component(axis));
                        speed = represented(total);
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
        FaceSum cap_volume;
        cap_volume.add_product(cfl, volume);
        for (size_t a = 0; a < families; ++a)
            for (size_t i = 0; i < nodes; ++i)
            {
                FaceSum rate;
                for (const auto face : mesh.faces(cell))
                {
                    const size_t direction = mesh.owner_cell(face) == cell ? 0 : 1;
                    rate.add_product(active_speed(speeds(face, outgoing_column(a, direction, i)), direction), 1.);
                }
                if (rate.sign() > 0)
                {
                    double bound = cap_volume.rounded(rate);
                    if (std::isfinite(bound) && bound > 0)
                    {
                        FaceSum slack = cap_volume;
                        slack.add_weighted(rate, -bound);
                        if (slack.sign() < 0)
                            bound = std::nextafter(bound, 0.);
                    }
                    local_limit = std::min(local_limit, bound);
                }
            }
    }
    return local_limit;
}

template<class Speeds>
double verify_reservations(
    const Mesh& mesh, const Speeds& speeds, size_t families, double h, double cap, std::vector<FaceSum>& fractions)
{
    double local_maximum = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto volume = mesh.cell_volume(cell);
        for (size_t a = 0; a < families; ++a)
            for (size_t i = 0; i < nodes; ++i)
            {
                auto& used = fractions[(c * families + a) * nodes + i];
                used = {};
                for (const auto face : mesh.faces(cell))
                {
                    const size_t direction = mesh.owner_cell(face) == cell ? 0 : 1;
                    const double speed = active_speed(speeds(face, outgoing_column(a, direction, i)), direction);
                    used.add_weighted(product({h, speed}, {volume}).sum(), 1.);
                }
                FaceSum excess = used;
                excess.add_product(-cap, 1.);
                const auto actual = used.rounded();
                local_maximum = std::max(local_maximum,
                    excess.sign() > 0 ? std::nextafter(actual, std::numeric_limits<double>::infinity()) : actual);
            }
    }
    return local_maximum;
}

template<class Trace, class Speeds, class Packet>
void stage_packets(const Mesh& mesh, const Flow::Options& options, const std::vector<size_t>& phases,
    const Trace& trace, const Speeds& speeds, const Packet& packet, size_t packet_columns,
    std::vector<FaceInterval>& intervals, FaceSum& caloric, FaceSum& energy, double h)
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
                std::vector<FaceSum> channels(packet_columns);
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
                    const auto alpha =
                        product({h, speed == 0 ? .5 : 1., weight, support, scale}, {volume, density}).sum();
                    interval.theta.add_weighted(alpha, 1.);
                    for (size_t axis = 0; axis < dimensions; ++axis)
                    {
                        FaceSum aligned;
                        aligned.add_weighted(alpha, slip.component(axis));
                        interval.slip.add_weighted(aligned, area.component(axis));
                    }
                    if (outward == 0)
                        continue;
                    const auto carrier = trace_vector(trace, donor, base + Carrier);
                    FaceSum withdrawal;
                    withdrawal.add_weighted(product({h, outward}, {volume}).sum(), weight);
                    const double count = represented(withdrawal);
                    channels[H0].add_product(count, 1.);
                    const double h1 = represented(product({count, support}));
                    channels[H1].add_product(h1, 1.);
                    channels[H2].add_product(represented(product({count, support, support})), 1.);
                    channels[H3].add_product(represented(product({count, support, support, support})), 1.);
                    const auto moles = family.species_moles(h1);
                    FaceSum material;
                    for (size_t s = 0; s < moles.size(); ++s)
                    {
                        channels[FirstSpecies + s].add_product(moles[s], 1.);
                        material.add_product(moles[s], family.catalog().species()[s].molarMass);
                    }
                    require(material.sign() > 0, "QBMM kinetic atom count and material transfer disagree.");
                    auto& momentum = interval.momentum[side];
                    momentum[Mass].add_weighted(material, sign);
                    const double donor_mass = trace(donor, base + MaterialMass);
                    FaceSum energy_packet;
                    product({trace(donor, base + Energy)}, {donor_mass}).add_weighted(energy_packet, material);
                    channels[U].add_weighted(energy_packet, 1.);
                    energy.add_weighted(energy_packet, energy_packet.sign() < 0 ? -1. : 1.);
                    product({}, {density}).add_weighted(channels[W], material);
                    caloric.add_weighted(material, trace(donor, base + CaloricSpecific));
                    for (size_t axis = 0; axis < dimensions; ++axis)
                    {
                        momentum[CarrierMomentum + axis].add_weighted(material, sign * carrier.component(axis));
                        momentum[DriftMomentum + axis].add_weighted(material, sign * slip.component(axis));
                    }
                }
                for (size_t q = 0; q < packet_columns; ++q)
                {
                    const double amount = represented(channels[q]);
                    packet(face, first + q) = amount;
                    interval.direction[side][q].add_product(amount, sign);
                }
            }
        }
    }
}

void compare(const FaceSum& actual, const FaceSum& constructive)
{
    FaceSum difference = actual;
    difference.add_weighted(constructive, -1.);
    FaceSum magnitude = constructive;
    magnitude.add_weighted(actual, 1.);
    require(relative_difference(difference, magnitude) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM kinetic canonical endpoint disagrees with the constructive measure.");
}

double updated_amount(double before, const FaceSum& incoming, const FaceSum& outgoing, bool depleted, bool nonnegative,
    double tolerance, double& maximum_residual)
{
    FaceSum result;
    result.add_product(before, 1.);
    result.add_weighted(incoming, 1.);
    result.add_weighted(outgoing, -1.);
    // A completely exhausted reconstructed measure can differ from the stored
    // donor by its inversion/product roundoff. Check that discrepancy against
    // the donor before adopting the incoming measure, including exact vacuum.
    if (depleted)
    {
        FaceSum remainder = result;
        remainder.add_weighted(incoming, -1.);
        FaceSum donor;
        donor.add_product(std::abs(before), 1.);
        donor.add_weighted(outgoing, outgoing.sign() < 0 ? -1. : 1.);
        require(relative_difference(remainder, donor) <= 2 * qbmm_reconstruction_tolerance,
            "QBMM kinetic complete evacuation disagrees with the authoritative stock.");
    }
    else
        require(!nonnegative || result.sign() >= 0, "QBMM kinetic withdrawal exceeds its canonical stock.");
    const double actual = represented(depleted ? incoming : result);
    if (actual == before && before != 0)
    {
        FaceSum net = incoming;
        net.add_weighted(outgoing, -1.);
        const bool withdrawal = before > 0 ? net.sign() < 0 : net.sign() > 0;
        require(!withdrawal, "QBMM kinetic net transfer is too small to update its authoritative stock.");
    }
    FaceSum defect = result;
    defect.add_product(-actual, 1.);
    FaceSum magnitude;
    magnitude.add_product(std::abs(before), 1.);
    magnitude.add_product(std::abs(actual), 1.);
    magnitude.add_weighted(incoming, incoming.sign() < 0 ? -1. : 1.);
    magnitude.add_weighted(outgoing, outgoing.sign() < 0 ? -1. : 1.);
    const double residual = relative_difference(defect, magnitude) / tolerance;
    require(residual <= 1, "QBMM kinetic component fails its canonical cell/face balance.");
    maximum_residual = std::max(maximum_residual, residual);
    return actual;
}

template<class Trace, class Packet>
void apply_packets(const Mesh& mesh, const Flow::Options& options, Flow::State& candidate,
    const std::vector<size_t>& phases, const Trace& trace, const Packet& packet, size_t packet_columns,
    const std::vector<FaceSum>& fractions, std::vector<CellInterval>& intervals, TransportResult& result)
{
    for (size_t c = 0; c < candidate.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        auto& stock = candidate.cells[c];
        for (size_t a = 0; a < phases.size(); ++a)
        {
            const auto phase = phases[a];
            const auto& family = *options.qbmm_families[phase];
            const auto rank = static_cast<size_t>(trace(cell, a * trace_columns + Rank));
            std::vector<FaceSum> incoming(packet_columns), outgoing(packet_columns);
            for (const auto face : mesh.faces(cell))
            {
                const size_t side = mesh.owner_cell(face) == cell ? 0 : 1;
                const size_t out = (a * sides + side) * packet_columns;
                const size_t in = (a * sides + 1 - side) * packet_columns;
                for (size_t q = 0; q < packet_columns; ++q)
                {
                    incoming[q].add_product(packet(face, in + q), 1.);
                    outgoing[q].add_product(packet(face, out + q), 1.);
                }
            }
            if (!incoming[H0].sign() && !outgoing[H0].sign())
                continue;
            std::array<FaceSum, 4> constructive;
            bool exhausted = rank > 0;
            for (size_t i = 0; i < rank; ++i)
            {
                const auto& fraction = fractions[(c * phases.size() + a) * nodes + i];
                FaceSum remainder;
                remainder.add_product(1., 1.);
                remainder.add_weighted(fraction, -1.);
                require(remainder.sign() >= 0, "QBMM kinetic constructive donor remainder is negative.");
                exhausted &= remainder.sign() == 0;
                const size_t atom = a * trace_columns + FirstAtom + i * AtomColumns;
                const double support = trace(cell, atom + Support);
                auto weight = remainder;
                for (size_t k = 0; k < constructive.size(); ++k)
                {
                    auto moment = weight;
                    weight = {};
                    weight.add_weighted(moment, support);
                    constructive[k].add_weighted(moment, trace(cell, atom + Weight));
                }
            }
            for (size_t k = 0; k < constructive.size(); ++k)
                constructive[k].add_weighted(incoming[k], 1.);
            const auto before =
                qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, stock, phase));
            auto& shape = *stock.qbmm_shape[phase];
            auto& thermal = stock.phases[phase];
            const double tolerance = options.conservation_relative_tolerance;
            *stock.phase_numbers[phase] = updated_amount(
                before[0], incoming[H0], outgoing[H0], exhausted, true, tolerance, result.maximum_population_residual);
            shape.H2 = updated_amount(
                before[2], incoming[H2], outgoing[H2], exhausted, true, tolerance, result.maximum_shape_residual);
            shape.H3 = updated_amount(
                before[3], incoming[H3], outgoing[H3], exhausted, true, tolerance, result.maximum_shape_residual);
            thermal.internal_energy = updated_amount(thermal.internal_energy, incoming[U], outgoing[U], exhausted,
                false, tolerance, result.maximum_energy_residual);
            for (size_t s = 0; s < thermal.moles.size(); ++s)
                thermal.moles[s] = updated_amount(thermal.moles[s], incoming[FirstSpecies + s],
                    outgoing[FirstSpecies + s], exhausted, true, tolerance, result.maximum_species_residual);
            const auto after =
                qbmm_cell_moments(family, qbmm_native_law_detail::kinetic_cell_view(family, stock, phase));
            for (size_t k = 0; k < constructive.size(); ++k)
            {
                FaceSum canonical;
                canonical.add_product(after[k], 1.);
                compare(canonical, constructive[k]);
            }
            const auto quadrature = invert_qbmm_moments(after);
            if (!quadrature.valid())
                throw std::invalid_argument(
                    std::string("QBMM kinetic compression failed at the represented endpoint (") +
                    qbmm_inversion_status_name(quadrature.status) + ").");
            auto& interval = intervals[c * phases.size() + a];
            interval.work_volume.add_weighted(incoming[W], 1.);
            interval.work_volume.add_weighted(outgoing[W], -1.);
            interval.gross_volume.add_weighted(incoming[W], 1.);
            interval.gross_volume.add_weighted(outgoing[W], 1.);
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
    std::vector<LawTrace> laws;
    std::vector<FaceSum> fractions;
    std::vector<FaceInterval> faces;
    std::vector<CellInterval> cells;
    FaceSum caloric, energy;
    const double cap = options.qbmm_model.kinetic_cfl;
    const int max_substeps = options.qbmm_model.maximum_kinetic_substeps;
    collective_detail::collective_local_validation(mesh, "QBMM kinetic transport setup",
        [&]
        {
            require(accepted.cells.size() == nc && candidate.cells.size() == nc &&
                        thermodynamic_guess.cells.size() == nc && accepted_material_volumes.size() == nc,
                "QBMM kinetic owned-cell layout differs.");
            require(std::isfinite(dt) && dt > 0 && std::isfinite(cap) && cap > 0 && cap <= 1 && max_substeps > 0,
                "Invalid QBMM kinetic interval or CFL policy.");
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
            fractions.resize(nc * phases.size() * nodes);
            for (const auto p : phases)
                result.owned_flux[p].resize(nf);
            for (auto& face : faces)
                for (auto& direction : face.direction)
                    direction.resize(packet_columns);
            for (size_t c = 0; c < nc; ++c)
            {
                require(accepted_material_volumes[c].size() == nphase, "QBMM kinetic material-volume layout differs.");
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const double volume = accepted_material_volumes[c][phases[a]];
                    require(std::isfinite(volume) && volume >= 0, "Invalid QBMM kinetic accepted material volume.");
                    auto& cell = cells[c * phases.size() + a];
                    cell.work_volume.add_product(volume, 1.);
                    cell.gross_volume.add_product(volume, 1.);
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
        collective_detail::collective_local_validation(mesh, "QBMM kinetic owned donor laws and caloric traces", [&]
            { stage_traces(mesh, options, accepted, thermodynamic_guess, candidate, phases, traces->write(), laws); });
        traces->synchronize();
        const auto trace = traces->read();
        collective_detail::collective_local_validation(mesh, "QBMM kinetic canonical face speeds",
            [&] { stage_speeds(mesh, carrier_flux, trace, speeds->write(), phases.size()); });
        speeds->synchronize();
        const auto speed = speeds->read();
        double local_limit = std::numeric_limits<double>::infinity();
        collective_detail::collective_local_validation(mesh, "QBMM kinetic all-face owner reserved rates",
            [&] { local_limit = reserve(mesh, speed, phases.size(), cap); });
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
                [&] { local_cfl = verify_reservations(mesh, speed, phases.size(), h, cap, fractions); });
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
                            const double mass_kg = represented(
                                product({trace(cell, atom + Support), options.qbmm_families[phase]->mass_scale_kg()}));
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
                stage_packets(
                    mesh, options, phases, trace, speed, packets->write(), packet_columns, faces, caloric, energy, h);
            });
        packets->synchronize();
        collective_detail::collective_local_validation(mesh, "QBMM kinetic constructive compression and balances",
            [&]
            {
                apply_packets(
                    mesh, options, candidate, phases, trace, packets->read(), packet_columns, fractions, cells, result);
            });
        ++result.substeps;
        remaining -= h;
    }
    collective_detail::collective_local_validation(mesh, "QBMM kinetic interval receipts and work volumes",
        [&]
        {
            FaceSum interval;
            interval.add_product(dt, 1.);
            for (const auto h : result.substep_durations)
                interval.add_product(-h, 1.);
            require(!interval.sign(), "QBMM kinetic substeps do not cover the requested interval exactly.");
            for (size_t c = 0; c < nc; ++c)
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto phase = phases[a];
                    const auto& cell = cells[c * phases.size() + a];
                    const auto& phase_state = candidate.cells[c].phases[phase];
                    if (*candidate.cells[c].phase_numbers[phase] == 0)
                    {
                        require(phase_state.internal_energy == 0 &&
                                    relative_difference(cell.work_volume, cell.gross_volume) <=
                                        2 * qbmm_reconstruction_tolerance,
                            "Locally empty kinetic phase has inconsistent canonical energy/work-volume packets.");
                        result.advected_material_volumes[c][phase] = 0;
                    }
                    else
                        result.advected_material_volumes[c][phase] = represented(cell.work_volume);
                }
            for (size_t f = 0; f < nf; ++f)
                for (size_t a = 0; a < phases.size(); ++a)
                {
                    const auto& interval = faces[f * phases.size() + a];
                    auto& momentum = result.integrated_faces[f];
                    momentum.forward_mass.add_weighted(interval.momentum[0][Mass], 1.);
                    momentum.backward_mass.add_weighted(interval.momentum[1][Mass], -1.);
                    momentum.theta.add_weighted(interval.theta, 1.);
                    momentum.slip.add_weighted(interval.slip, 1.);
                    momentum.material_volume.add_weighted(interval.direction[0][W], 1.);
                    momentum.material_volume.add_weighted(interval.direction[1][W], 1.);
                    for (size_t axis = 0; axis < dimensions; ++axis)
                    {
                        momentum.drift[axis].add_weighted(interval.momentum[0][DriftMomentum + axis], 1.);
                        momentum.drift[axis].add_weighted(interval.momentum[1][DriftMomentum + axis], 1.);
                    }
                    result.owned_flux[phases[a]][f] =
                        face_receipt(interval, dt, carrier_flux.local_value(static_cast<LO>(f)));
                }
            result.local_caloric_transport_magnitude = represented(caloric);
            result.local_energy_transport_magnitude = represented(energy);
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
