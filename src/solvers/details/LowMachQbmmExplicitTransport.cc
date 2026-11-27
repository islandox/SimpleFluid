#include "solvers/details/LowMachQbmmExplicitTransport.hh"

#include "FVM/details/OperatorDetails.hh"
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/PredicateSigns2D.hh"
#include "solvers/details/LowMachQbmmLawInputs.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Reacting::qbmm_explicit_detail
{
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}

namespace
{
using Vec3 = Mesh::Vec3;

double length(const Vec3& vector)
{
    return std::hypot(vector.x, vector.y, vector.z);
}

bool finite(const Vec3& vector)
{
    return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

constexpr auto invalid_base_index = Mesh::SemiStructured::Topology::invalid_ordinal;

double xy_cross(const Vec3& a, const Vec3& b)
{
    return std::fma(a.x, b.y, -a.y * b.x);
}

double xy_length(const Vec3& a)
{
    return std::hypot(a.x, a.y);
}

struct CircularBoundary
{
    std::vector<Vec3> corners;
    Vec3 center;
    double coordinate_roundoff;
    int orientation;
};

CircularBoundary circular_boundary(const std::vector<Vec3>& points)
{
    const auto& origin = points.front();
    double scale = 0., magnitude = 0.;
    for (const auto& point : points)
    {
        scale = std::max(scale, xy_length(point - origin));
        magnitude = std::max({magnitude, std::abs(point.x), std::abs(point.y)});
    }
    require(scale > 0. && std::isfinite(scale), "Cylindrical QBMM requires a finite polygonal wall.");
    // Compare local shapes in units of their diameter. Input rounding also depends on the absolute
    // coordinate magnitude when a small vessel is translated far from the origin.
    const double coordinate_roundoff = 16 * std::numeric_limits<double>::epsilon() * std::max(magnitude, scale);
    const double roundoff = coordinate_roundoff / scale;
    std::vector<Vec3> local;
    local.reserve(points.size());
    for (const auto& point : points)
        local.push_back((point - origin) / scale);
    std::vector<size_t> corners;
    for (size_t i = 0; i < local.size(); ++i)
    {
        const auto before = local[i] - local[(i + local.size() - 1) % local.size()];
        const auto after = local[(i + 1) % local.size()] - local[i];
        if (before.dot(after) <= 0. ||
            std::abs(xy_cross(before, after)) > roundoff * (xy_length(before) + xy_length(after)))
            corners.push_back(i);
    }
    require(corners.size() >= 3, "Cylindrical QBMM requires at least three distinct circular wall corners.");

    // Use long chords for the circumcenter so clustered angular samples do not determine the fit.
    const auto& first = local[corners.front()];
    const auto longest = *std::ranges::max_element(corners, {},
        [&](size_t i)
        {
            const auto offset = local[i] - first;
            return offset.dot(offset);
        });
    const auto u = local[longest] - first;
    const auto widest =
        *std::ranges::max_element(corners, {}, [&](size_t i) { return std::abs(xy_cross(u, local[i] - first)); });
    const auto v = local[widest] - first;
    const double determinant = xy_cross(u, v);
    require(std::abs(determinant) > 4 * roundoff, "Cylindrical QBMM wall curvature is unresolved.");
    const double u2 = u.dot(u), v2 = v.dot(v);
    const auto center = first + Vec3{u2 * v.y - v2 * u.y, v2 * u.x - u2 * v.x, 0.} / (2 * determinant);
    const double radius = xy_length(first - center);
    const double radial_roundoff = 8 * roundoff;
    const int orientation = xy_cross(first - center, local[corners[1]] - center) > 0. ? 1 : -1;
    CircularBoundary ring{{}, origin + center * scale, radial_roundoff * scale, orientation};
    ring.corners.reserve(corners.size());
    size_t positive_axis_crossings = 0;
    for (size_t k = 0; k < corners.size(); ++k)
    {
        const size_t i = corners[k], j = corners[(k + 1) % corners.size()];
        const auto a = local[i] - center, b = local[j] - center;
        require(std::abs(xy_length(a) - radius) <= radial_roundoff && orientation * xy_cross(a, b) > 0.,
            "Cylindrical QBMM walls must follow ordered circular facets about their center.");
        positive_axis_crossings += orientation > 0 ? a.y <= 0. && b.y > 0. : a.y > 0. && b.y <= 0.;
        ring.corners.push_back(points[i]);
        // Check all omitted subdivisions against the complete chord, including their order.
        const auto chord = local[j] - local[i];
        const double length = xy_length(chord);
        const auto tangent = chord / length;
        double preceding = 0.;
        for (size_t s = (i + 1) % points.size(); s != j; s = (s + 1) % points.size())
        {
            const auto offset = local[s] - local[i];
            const double along = tangent.dot(offset);
            require(along > preceding && along < length && std::abs(xy_cross(tangent, offset)) <= roundoff,
                "Cylindrical QBMM wall subdivisions must follow their circular facet in order.");
            preceding = along;
        }
    }
    require(positive_axis_crossings == 1, "Cylindrical QBMM requires simple circular wall cycles.");
    return ring;
}

void validate_annular_boundaries(const CircularBoundary& outer, const CircularBoundary& inner)
{
    require(xy_length(outer.center - inner.center) <= outer.coordinate_roundoff + inner.coordinate_roundoff,
        "Cylindrical QBMM annular walls must be concentric.");
    using Meshes::detail::orient2d_sign;
    using Meshes::detail::PredicateSign;
    const auto upper_half = [&](const Vec3& point)
    {
        return point.y > outer.center.y || (point.y == outer.center.y && point.x >= outer.center.x);
    };
    const auto angle_less = [&](const Vec3& a, const Vec3& b)
    {
        const bool upper_a = upper_half(a), upper_b = upper_half(b);
        return upper_a != upper_b ? upper_a : orient2d_sign(outer.center, a, b) == PredicateSign::Positive;
    };
    // Locate the crossed edge by exact ray order so a rounded angle cannot select a neighboring wall.
    auto corners = outer.corners;
    std::ranges::rotate(corners, std::ranges::min_element(corners, angle_less));
    require(std::ranges::adjacent_find(corners, [&](const Vec3& a, const Vec3& b) { return !angle_less(a, b); }) ==
                corners.end(),
        "Cylindrical QBMM walls must follow ordered circular facets about their center.");
    for (size_t k = 0; k < corners.size(); ++k)
        require(orient2d_sign(outer.center, corners[k], corners[(k + 1) % corners.size()]) == PredicateSign::Positive,
            "Cylindrical QBMM outer facets must enclose their common center.");
    for (const auto& point : inner.corners)
    {
        if (point.x == outer.center.x && point.y == outer.center.y)
            continue;
        const auto next = std::ranges::upper_bound(corners, point, angle_less);
        const auto& b = next == corners.end() ? corners.front() : *next;
        const auto& a = next == corners.begin() ? corners.back() : *(next - 1);
        require(orient2d_sign(a, b, point) == PredicateSign::Positive,
            "Cylindrical QBMM annular inner walls must lie strictly inside the outer wall.");
    }
}

} // namespace

void validate_cylindrical_base(const Flow::Mesh::SemiStructured& mesh)
{
    const auto& topology = mesh.topology();
    std::vector<bool> visited(mesh.xy_cell_nodes().size());
    require(!visited.empty(), "Cylindrical QBMM requires a nonempty polygonal base.");
    std::vector<unsigned> connected{0};
    visited[0] = true;
    for (size_t row = 0; row < connected.size(); ++row)
        for (const auto face : topology.cell_side_faces(connected[row]))
        {
            const auto& side = topology.side_face(face);
            const auto other = side.owner == connected[row] ? side.neighbor : side.owner;
            if (other != invalid_base_index && !visited[other])
            {
                visited[other] = true;
                connected.push_back(other);
            }
        }
    require(connected.size() == visited.size(), "Cylindrical QBMM requires an edge-connected polygonal base.");

    std::vector<unsigned> next(mesh.xy_nodes().size(), invalid_base_index), previous = next;
    for (const auto& side : topology.side_faces())
        if (side.neighbor == invalid_base_index)
        {
            const auto [a, b] = side.nodes;
            require(next[a] == invalid_base_index && previous[b] == invalid_base_index,
                "Cylindrical QBMM requires disjoint polygonal wall cycles.");
            next[a] = b;
            previous[b] = a;
        }
    std::optional<CircularBoundary> outer, inner;
    for (size_t i = 0; i < next.size(); ++i)
    {
        if (next[i] == invalid_base_index)
            continue;
        std::vector<Vec3> points;
        auto node = static_cast<unsigned>(i);
        do
        {
            require(next[node] != invalid_base_index, "Cylindrical QBMM requires closed polygonal wall cycles.");
            const auto& point = mesh.xy_nodes()[node];
            points.push_back({point.x, point.y, 0.});
            node = std::exchange(next[node], invalid_base_index);
        } while (node != i);
        auto ring = circular_boundary(points);
        auto& slot = ring.orientation > 0 ? outer : inner;
        require(!slot, "Cylindrical QBMM accepts one polygonal outer wall and at most one inner wall.");
        slot = std::move(ring);
    }
    require(outer.has_value(), "Cylindrical QBMM requires a polygonal outer wall.");
    if (inner)
        validate_annular_boundaries(*outer, *inner);
}

void validate_axial_kinetic_geometry(const Mesh& mesh)
{
    constexpr double roundoff = 64 * eps;
    const auto local = [&](LO cell) { return cell >= 0 && static_cast<size_t>(cell) < mesh.num_local_cells(); };
    const auto owned = [&](LO cell) { return cell >= 0 && static_cast<size_t>(cell) < mesh.num_owned_cells(); };
    const auto valid_offset = [&](LO face, LO cell, const Vec3& face_center)
    {
        const double distance = mesh.cell_to_face_distance(face, cell);
        const double length_from_centers = length(face_center - mesh.cell_centroid(cell));
        return std::isfinite(distance) && std::isfinite(length_from_centers) &&
               (length_from_centers == 0. ? distance == 0.
                                          : distance > 0. && std::abs(distance / length_from_centers - 1.) <= roundoff);
    };
    for (size_t c = 0; c < mesh.num_local_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const double volume = mesh.cell_volume(cell);
        require(std::isfinite(volume) && volume > 0 && finite(mesh.cell_centroid(cell)),
            "Cylindrical kinetic QBMM requires finite cell centers and positive volumes.");
    }
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const auto owner = mesh.owner_cell(face), neighbor = mesh.neighbor_cell(face);
        if (f >= mesh.num_owned_faces() && !owned(owner) && !owned(neighbor))
            continue;
        const bool wall = mesh.is_geometry_exterior_face(face);
        require(local(owner) && (wall ? mesh.is_boundary_face(face)
                                      : !mesh.is_boundary_face(face) && local(neighbor) && neighbor != owner &&
                                            mesh.opposite_or_periodic_neighbor_cell(face, owner) == neighbor),
            "Cylindrical kinetic QBMM requires physical walls and a complete adjacent cell halo.");

        const double area = mesh.face_area(face);
        const auto normal = mesh.face_normal(face);
        const auto area_vector = mesh.face_area_vector(face);
        const auto center = mesh.face_centroid(face);
        require(std::isfinite(area) && area > 0 && finite(area_vector) && finite(center) &&
                    std::abs(length(normal) - 1.) <= roundoff,
            "Cylindrical kinetic QBMM requires finite face centroids and positive scalar and vector areas.");
        for (size_t axis = 0; axis < dimensions; ++axis)
            require(std::abs(area_vector.component(axis) / area - normal.component(axis)) <= roundoff,
                "Cylindrical kinetic QBMM face area vector disagrees with its scalar area and normal.");
        // The side measure can be curved. Its Cartesian axial component is exactly zero.
        require(normal.z == 0. ? area_vector.z == 0.
                               : normal.x == 0. && normal.y == 0. && area_vector.x == 0. && area_vector.y == 0. &&
                                     std::abs(area_vector.z) == area,
            "Cylindrical kinetic QBMM requires axial caps and side faces with zero axial projection.");
        require(valid_offset(face, owner, center) && (wall || valid_offset(face, neighbor, center)),
            "Cylindrical kinetic QBMM requires consistent finite cell-to-face distances.");
        if (wall)
            continue;

        const auto displacement = mesh.cell_center_vector(face, owner);
        const double center_distance = length(displacement);
        const double pressure_distance = mesh.face_cell_center_distance(face);
        require(std::isfinite(center_distance) && center_distance > 0 && std::isfinite(pressure_distance) &&
                    pressure_distance > 0 && std::abs(pressure_distance / center_distance - 1.) <= roundoff &&
                    (displacement / center_distance).dot(normal) >= 1. - 1e-10,
            "Cylindrical kinetic QBMM requires positive pressure distances and orthogonal owner-oriented faces.");
    }
}

void FaceOffsetFit::assign(std::span<const Vec3> displacements)
{
    std::array<std::array<real_t, dimensions>, dimensions> matrix{};
    samples.clear();
    samples.reserve(displacements.size());
    for (const auto& displacement : displacements)
    {
        const double distance = length(displacement);
        require(std::isfinite(distance) && distance > 0, "Unresolved QBMM cell-center displacement.");
        const auto direction = displacement / distance;
        for (size_t i = 0; i < dimensions; ++i)
            for (size_t j = 0; j < dimensions; ++j)
                matrix[i][j] += direction.component(i) * direction.component(j);
        samples.push_back({displacement, direction, distance});
    }
    for (auto& sample : samples)
    {
        auto local_matrix = matrix;
        sample.basis = FVM::detail::solve_3x3(local_matrix, sample.basis);
    }
}

bool FaceOffsetFit::weights(const Vec3& offset, std::vector<double>& result) const
{
    const double distance = length(offset);
    require(std::isfinite(distance), "Invalid QBMM cell-to-face displacement.");
    result.clear();
    if (distance == 0)
        return true;
    result.reserve(samples.size());
    for (const auto& sample : samples)
        result.push_back(
            finite_rate(sample.basis.dot(offset / sample.distance), "Unrepresentable QBMM linear stencil weight."));
    // The pseudoinverse can fit a rank-deficient one-ring only in its resolved directions.
    bool resolved = true;
    for (size_t i = 0; i < dimensions; ++i)
    {
        Accumulator defect;
        defect += -offset.component(i);
        for (size_t k = 0; k < samples.size(); ++k)
            defect += static_cast<long double>(result[k]) * samples[k].displacement.component(i);
        resolved &= relative_difference(defect.value(), distance) <= 4096 * eps;
    }
    return resolved;
}

double finite_rate(double value, const char* message)
{
    require(std::isfinite(value), message);
    return value;
}

double relative_difference(long double difference, long double magnitude)
{
    constexpr double infinity = std::numeric_limits<double>::infinity();
    if (!std::isfinite(difference) || !std::isfinite(magnitude))
        return infinity;
    if (difference == 0)
        return 0;
    if (magnitude <= 0)
        return infinity;
    const long double relative = std::abs(difference) / magnitude;
    return relative <= std::numeric_limits<double>::max() ? static_cast<double>(relative) : infinity;
}

static long double checked_rate(long double amount, long double duration, const char* message)
{
    require(std::isfinite(duration) && duration > 0, "Invalid QBMM kinetic packet divisor.");
    const long double rate = amount / duration;
    require(std::isfinite(rate) && std::abs(rate) <= std::numeric_limits<double>::max(), message);
    return rate;
}

double represented(long double amount, long double divisor)
{
    const long double rate = checked_rate(amount, divisor, "QBMM kinetic transfer overflows binary64.");
    const double value = static_cast<double>(rate);
    require(value != 0 || amount == 0, "QBMM kinetic transfer underflows binary64.");
    if (std::fpclassify(value) == FP_SUBNORMAL)
        require(relative_difference(value - rate, std::abs(rate)) <= eps, "QBMM kinetic transfer loses relative precision.");
    return value;
}

double rounded_rate(long double amount, long double duration)
{
    return static_cast<double>(checked_rate(amount, duration, "QBMM kinetic face receipt overflows binary64."));
}

static double mass(const PhaseThermalState& phase)
{
    Accumulator sum;
    const auto& species = phase.thermodynamics->catalog().species();
    for (size_t s = 0; s < phase.moles.size(); ++s)
        sum += static_cast<long double>(phase.moles[s]) * species[s].molarMass;
    return represented(sum.value());
}

StageThermal thermal_trace(PhaseThermalState& phase, bool recover_arrival, std::vector<double>& composition)
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
    composition = phase.moles;
    const double scale = *std::ranges::max_element(composition);
    for (auto& value : composition)
        value /= scale;
    const double sum = std::accumulate(composition.begin(), composition.end(), 0.);
    for (auto& value : composition)
        value /= sum;
    const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, composition);
    result.density = point.density;
    require(std::isfinite(point.density) && point.density > 0, "Invalid QBMM kinetic packet divisor.");
    const double volume = result.material_mass / point.density;
    result.volume = std::isnormal(volume) ? volume
                                         : represented(static_cast<long double>(result.material_mass) / point.density);
    const double caloric = phase.temperature * point.internal_energy_temperature;
    result.caloric_specific = std::isnormal(caloric)
        ? caloric
        : represented(static_cast<long double>(phase.temperature) * point.internal_energy_temperature);
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
    Accumulator change, magnitude;
    double scale = 0;

    void add(double before, double after)
    {
        change += after;
        change += -before;
        magnitude += std::abs(before);
        magnitude += std::abs(after);
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
        result.species_molar_flux[s] = represented(channels[FirstSpecies + s].value(), dt);
    for (size_t k = 0; k < result.moment_flux.size(); ++k)
        result.moment_flux[k] = represented(channels[k].value(), dt);
    result.integrated_count = represented(channels[H0].value());
    result.integrated_mass = rounded_rate(momentum[Mass].value());
    result.mass_flux = rounded_rate(momentum[Mass].value(), dt);
    result.internal_energy_flux = represented(channels[U].value(), dt);
    result.material_volume_flux = represented(channels[W].value(), dt);
    for (size_t axis = 0; axis < dimensions; ++axis)
    {
        result.carrier_momentum_flux.component(axis) = rounded_rate(momentum[CarrierMomentum + axis].value(), dt);
        result.drift_momentum_flux.component(axis) = rounded_rate(momentum[DriftMomentum + axis].value(), dt);
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
        auto total = streams[0][channel];
        total.append(streams[1][channel]);
        return rounded_rate(total.value(), dt);
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

    const long double theta = interval.theta.value(), slip = interval.slip.value();
    const long double forward_volume = interval.direction[0][W].value();
    const long double backward_volume = interval.direction[1][W].value();
    auto packet = interval.direction[0][W];
    packet.append(interval.direction[1][W]);
    const long double physical_carrier = theta * carrier_flux;
    const long double physical_defect = physical_carrier + slip - packet.value();
    const long double physical_scale = std::abs(physical_carrier) + std::abs(slip) + forward_volume - backward_volume;
    require(relative_difference(physical_defect, physical_scale) <= 2 * qbmm_reconstruction_tolerance,
        "QBMM kinetic pressure coefficients disagree with the directly transported volume packet.");

    kinetic.theta = rounded_rate(theta, dt);
    require(kinetic.theta >= 0, "QBMM kinetic pressure coefficient is negative.");
    kinetic.slip_volume_flux = rounded_rate(slip, dt);
    // Coefficient rounding can lose material flux near the subnormal floor or
    // when two normal operands almost cancel. Use a separate representation
    // term at the frozen carrier flux, leaving physical slip independently recorded.
    const auto imprecisely_rounded = [](double rounded, long double exact)
    { return std::fpclassify(rounded) == FP_SUBNORMAL || (rounded == 0 && exact != 0); };
    const long double pressure_carrier = static_cast<long double>(kinetic.theta) * carrier_flux;
    const double pressure_volume_flux = std::fma(kinetic.theta, carrier_flux, kinetic.slip_volume_flux);
    const long double residual = static_cast<long double>(result.material_volume_flux) - pressure_volume_flux;
    const long double direct_magnitude =
        static_cast<long double>(kinetic.forward.material_volume_flux) - kinetic.backward.material_volume_flux;
    if (imprecisely_rounded(kinetic.theta, theta) || imprecisely_rounded(kinetic.slip_volume_flux, slip) ||
        relative_difference(residual, direct_magnitude) > 2 * qbmm_reconstruction_tolerance)
    {
        kinetic.pressure_rounding_flux = rounded_rate(residual);

        const long double theta_error = theta - static_cast<long double>(kinetic.theta) * dt;
        const long double slip_error = slip - static_cast<long double>(kinetic.slip_volume_flux) * dt;
        const long double packet_error =
            packet.value() - static_cast<long double>(result.material_volume_flux) * dt;
        // The published pressure expression also rounds its fused binary64 evaluation.
        const long double evaluation_error = pressure_carrier + kinetic.slip_volume_flux - pressure_volume_flux;
        const long double half_minimum_packet =
            static_cast<long double>(std::numeric_limits<double>::denorm_min()) * dt * .5L;
        const long double rounding_bound = std::abs(theta_error * carrier_flux) + std::abs(slip_error) +
                                           std::abs(packet_error) + std::abs(evaluation_error) * dt +
                                           std::abs(physical_defect) + half_minimum_packet;
        const long double correction =
            static_cast<long double>(std::abs(kinetic.pressure_rounding_flux)) * dt * (1.L - 2.L * eps);
        require(correction <= rounding_bound,
            "QBMM kinetic pressure correction exceeds its coefficient rounding bound.");
    }

    const long double defect = kinetic.pressure_rounding_flux - residual;
    const long double magnitude = std::abs(pressure_carrier) + std::abs(kinetic.slip_volume_flux) +
                                  std::abs(kinetic.pressure_rounding_flux) + direct_magnitude;
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
                    local[2 * q] = rounded_rate(budget[q].change.value(), scale[q]);
                    local[2 * q + 1] = rounded_rate(budget[q].magnitude.value(), scale[q]);
                }
        });
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, static_cast<int>(operands), local.data(), totals.data());
    const double tolerance = (64. + 2. * comm.getSize()) * eps;
    for (size_t q = 0; q < channels; ++q)
        require(std::abs(totals[2 * q]) <= tolerance * totals[2 * q + 1],
            "QBMM kinetic closed-family transport budget exceeds binary64 roundoff.");
}
} // namespace SimpleFluid::Reacting::qbmm_explicit_detail
