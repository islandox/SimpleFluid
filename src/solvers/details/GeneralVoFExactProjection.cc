/** @file GeneralVoFFlowWorkspace.cc
 * @brief Conditioned native gradients, stress transport and nonorthogonal projection.
 */
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/BoundedConvexGeometry.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactPressure.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cfenv>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "solvers/details/GeneralVoFFlowSupport.hh"
namespace SimpleFluid::detail
{
using namespace flow_support;
namespace Exact
{
void GeneralVoFFlowWorkspace::project(detail::GeneralVoFFlowWorkspace& self, std::vector<double>& face_velocity,
    std::vector<Vector>& velocity, std::vector<double>& pressure, const std::vector<double>& density,
    const std::vector<double>& fraction, const std::vector<double>& kappa, double dt, double sigma,
    const VoFSolverOptions& options, VoFStepDiagnostics& diagnostics, GeneralVoFProjectionProvenance* provenance,
    std::optional<std::span<const Meshes::ExactGeometryMeasure>> physical_mass,
    const GeneralVoFProjectionClosureTopology* closure_topology)
{
    self.require(!self.finite_arithmetic_,
        "Exact projection requires an exact workspace; finite policy cannot switch arithmetic.");
    int closure_requested = bool(closure_topology), closure_minimum = 0, closure_maximum = 0;
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &closure_requested, &closure_minimum);
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &closure_requested, &closure_maximum);
    self.require(self.current_geometry() && closure_minimum == closure_maximum && (!closure_topology || provenance),
        "General VoF rate-closure request differs across ranks, lacks provenance or has stale workspace geometry.");
    int mass_requested = bool(physical_mass), mass_minimum = 0, mass_maximum = 0;
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &mass_requested, &mass_minimum);
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &mass_requested, &mass_maximum);
    self.require(mass_minimum == mass_maximum && (!physical_mass || physical_mass->size() == self.volumes_.size()),
        "General VoF physical mixture-mass request differs across ranks or has wrong shape.");
    int arithmetic_requested = bool(provenance), arithmetic_minimum = 0, arithmetic_maximum = 0;
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &arithmetic_requested, &arithmetic_minimum);
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &arithmetic_requested, &arithmetic_maximum);
    self.require(arithmetic_minimum == arithmetic_maximum && (!provenance || provenance->byte_budget > 0),
        "General VoF projection arithmetic request differs across ranks or has no budget.");
    bool valid = velocity.size() == self.volumes_.size() && fraction.size() == self.volumes_.size() &&
                 kappa.size() == self.volumes_.size() && density.size() == self.volumes_.size() &&
                 face_velocity.size() == self.geometry_.size() && std::isfinite(dt) && dt > 0 && std::isfinite(sigma) &&
                 sigma >= 0;
    for (double rho : density)
        valid = valid && std::isfinite(rho) && rho > 0;
    for (double value : fraction)
        valid = valid && std::isfinite(value) && value >= 0 && value <= 1;
    for (double value : kappa)
        valid = valid && std::isfinite(value);
    for (const auto& u : velocity)
        for (double value : u)
            valid = valid && std::isfinite(value);
    self.require(valid, "General VoF projection has invalid physical dimensions or density.");
    for (std::size_t id = 0; id < self.mesh_->num_owned_faces(); ++id)
        valid = valid && std::isfinite(face_velocity[id]) && (!self.geometry_[id].boundary || face_velocity[id] == 0);
    self.require(valid, "General VoF projection requires finite face velocities and impermeable walls.");
    const auto arithmetic_guard = [&](auto work)
    {
        bool accepted = true;
        try
        {
            work();
        }
        catch (...)
        {
            accepted = false;
        }
        self.require(accepted, "General VoF projection arithmetic exceeded its range or budget.");
    };
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(provenance ? provenance->byte_budget : 64 * 1024 * 1024);
    std::size_t scalar_budget = 0, admission_budget = 0;
    arithmetic_guard(
        [&]
        {
            const auto n = self.volumes_.size(), nf = self.geometry_.size(),
                       owned_faces = provenance ? self.mesh_->num_owned_faces() : 0, nc = self.mesh_->num_cells();
            storage.capacity(Domain::Input, self.geometry_.capacity(), sizeof(Face));
            storage.capacity(Domain::Input, self.volumes_.capacity(), sizeof(double));
            storage.capacity(Domain::Input, self.spans_.capacity(), sizeof(Vector));
            storage.capacity(Domain::Input, self.exact_volumes_.capacity(), sizeof(Meshes::ExactGeometryMeasure));
            for (const auto& value : self.exact_volumes_)
                storage.measure(Domain::Input, value);
            if (physical_mass)
            {
                storage.capacity(Domain::Input, physical_mass->size(), sizeof(Meshes::ExactGeometryMeasure));
                for (const auto& value : *physical_mass)
                    storage.measure(Domain::Input, value);
            }
            storage.capacity(Domain::Input, n, 4 * sizeof(double) + sizeof(Vector));
            storage.capacity(Domain::Input, face_velocity.capacity(), sizeof(double));
            storage.capacity(Domain::Input, pressure.capacity(), sizeof(double));
            storage.capacity(Domain::Scratch, nc, 8 * sizeof(Vector) + 8 * sizeof(double));
            storage.capacity(Domain::Scratch, nf, 8 * sizeof(double));
            storage.capacity(Domain::Scratch, n, 12 * sizeof(double) + 2 * sizeof(Vector));
            storage.capacity(Domain::Result, owned_faces, sizeof(GeneralVoFRoundedValue));
            if (closure_topology)
            {
                storage.capacity(Domain::Input, closure_topology->cells.capacity(), sizeof(GeneralVoFPhiCell));
                storage.capacity(Domain::Input, closure_topology->faces.capacity(), sizeof(GeneralVoFPhiFace));
                storage.capacity(Domain::Scratch, closure_topology->cells.size(), sizeof(GeneralVoFPhiCell));
                storage.capacity(Domain::Scratch, closure_topology->faces.size(), sizeof(GeneralVoFPhiFace));
                for (const auto& cell : closure_topology->cells)
                    storage.capacity(Domain::Scratch, cell.identity.capacity(), 2);
                for (const auto& face : closure_topology->faces)
                    storage.capacity(Domain::Scratch, face.identity.capacity(), 2);
                storage.capacity(Domain::Result, owned_faces, sizeof(Meshes::ExactGeometryMeasure));
            }
            const auto helper_pool = provenance ? storage.remaining() / 2 : 0;
            storage.capacity(Domain::Scratch, helper_pool, 1);
            // One half retains the current certificate; the other is reused
            // serially by forest, exact pressure or scalar reduction buffers.
            admission_budget = helper_pool / 2;
            if (owned_faces > (std::numeric_limits<std::size_t>::max() - 32) / 2)
                throw std::runtime_error("Projection exact scalar capacity overflow.");
            // At most two retained interval measures per owned face. Thirty-two
            // extra scalar slots cover interval corners, nested operation inputs,
            // physical-impulse normalization and scalar publication temporaries.
            // Sixteen additional slots form a separate synchronous operation
            // workspace, never borrowed from pending corner/intermediate slots.
            storage.capacity(Domain::Scratch, 2 * owned_faces + 6 * n, sizeof(Meshes::ExactGeometryMeasure));
            storage.capacity(Domain::Scratch, owned_faces, sizeof(GeneralVoFExactPressureFace));
            const auto result_slots = (closure_topology ? 3 : 2) * owned_faces;
            const auto temporary_slots = std::size_t(48) + 3 * owned_faces + 6 * n;
            scalar_budget = storage.remaining() / (result_slots + temporary_slots);
            if (scalar_budget < 1024 || (provenance && admission_budget < 1024))
                throw std::runtime_error("Projection cumulative exact resource reserve exhausted.");
            storage.capacity(Domain::Result, result_slots, scalar_budget);
            storage.capacity(Domain::Scratch, temporary_slots, scalar_budget);
        });
    arithmetic_guard(
        [&]
        {
            if (physical_mass)
                for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                {
                    const auto& mass = (*physical_mass)[c];
                    if (mass.sign() <= 0 ||
                        vof_exact::leased_divide(mass, self.exact_volumes_[c], scalar_budget).certified().as_double() !=
                            density[c])
                        throw std::runtime_error(
                            "Exact physical mixture mass disagrees with its represented density provenance.");
                }
        });
    GeneralVoFProjectionArithmetic arithmetic(scalar_budget);
    std::vector<GeneralVoFRoundedValue> rounded(provenance ? self.mesh_->num_owned_faces() : 0);
    arithmetic_guard(
        [&]
        {
            if (provenance)
                for (std::size_t face = 0; face < rounded.size(); ++face)
                    rounded[face] = arithmetic.leaf(face_velocity[face]);
        });
    using M = Meshes::ExactGeometryMeasure;
    const auto exact_product = [&](const M& a, const M& b) -> M
    {
        if (!a.sign() || !b.sign())
            return {};
        return vof_exact::leased_divide(a, vof_exact::leased_divide(M::from_value(1), b, scalar_budget), scalar_budget);
    };
    std::vector<M> accepted_rate, primary_rate, primary_radius;
    std::vector<M> exact_pressure, exact_divergence, exact_rhs;
    std::vector<std::array<M, 3>> exact_force;
    GeneralVoFProjectionRateCertificate current_certificate;
    if (closure_topology)
    {
        arithmetic_guard(
            [&]
            {
                for (const auto& cell : closure_topology->cells)
                    if (cell.arithmetic_radius.sign())
                        throw std::runtime_error("Rate topology contains a nonzero cell placeholder.");
                for (const auto& face : closure_topology->faces)
                    if (face.volume.sign() || face.arithmetic_radius.sign())
                        throw std::runtime_error("Rate topology contains nonzero face placeholders.");
            });
        // Authenticate metadata independently of current pressure feasibility.
        (void) close_general_vof_phi(self.mesh_, closure_topology->cells, closure_topology->faces, admission_budget);
    }
    const auto capacity_context = [&]
    {
        GeneralVoFExactStorage live(provenance ? provenance->byte_budget : 64 * 1024 * 1024);
        const auto retain = [&](const auto& values)
        {
            live.capacity(Domain::Scratch, values.capacity(), sizeof(M));
            for (const auto& value : values)
                live.measure(Domain::Scratch, value);
        };
        retain(self.exact_volumes_);
        if (physical_mass)
            for (const auto& value : *physical_mass)
                live.measure(Domain::Input, value);
        for (const auto* values : {&accepted_rate, &primary_rate, &primary_radius, &exact_pressure, &exact_divergence,
                 &exact_rhs, &current_certificate.owned_rate, &current_certificate.owned_face_radius,
                 &current_certificate.owned_cell_radius, &current_certificate.owned_net_rate})
            retain(*values);
        live.capacity(Domain::Scratch, rounded.capacity(), sizeof(GeneralVoFRoundedValue));
        for (const auto& value : rounded)
        {
            live.measure(Domain::Scratch, value.lower);
            live.measure(Domain::Scratch, value.upper);
        }
        live.capacity(Domain::Scratch, exact_force.capacity(), sizeof(std::array<M, 3>));
        for (const auto& force : exact_force)
            for (const auto& value : force)
                live.measure(Domain::Scratch, value);
        const auto report = storage.finish();
        return " workspace_scalar_budget=" + std::to_string(scalar_budget) +
               ", helper_budget=" + std::to_string(admission_budget) +
               ", outer_reserved=" + std::to_string(report.total()) +
               ", outer_remaining=" + std::to_string(storage.remaining()) +
               ", retained_exact_and_container_bytes=" + std::to_string(live.finish().total());
    };
    std::string closure_reason = "not attempted";
    unsigned correction_count = 0;
    const auto rate_admitted = [&]
    {
        if (!provenance)
            return true;
        current_certificate = {};
        try
        {
            current_certificate = certify_general_vof_projection_rate(self.mesh_, rounded, admission_budget,
                primary_rate.empty() && self.mesh_->num_owned_faces() ? nullptr : &primary_rate,
                primary_radius.empty() && self.mesh_->num_owned_faces() ? nullptr : &primary_radius);
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error(std::string("Primary rate certificate: ") + error.what() + capacity_context());
        }
        const auto& certificate = current_certificate;
        if (!closure_topology)
            return certificate.arithmetic_admitted;
        if (!certificate.arithmetic_admitted)
        {
            const auto comm = self.mesh_->owned_cell_map()->getComm();
            int local_violations = 0, violations = 0, local_zero_faces = 0, zero_faces = 0;
            int witness = comm->getSize(), owner = 0;
            std::string detail;
            arithmetic_guard(
                [&]
                {
                    for (std::size_t f = 0; f < certificate.owned_face_radius.size(); ++f)
                        local_zero_faces += certificate.owned_face_radius[f].sign() == 0;
                    for (std::size_t c = 0; c < certificate.owned_net_rate.size(); ++c)
                    {
                        const auto& net = certificate.owned_net_rate[c];
                        const auto magnitude = net.sign() < 0 ? vof_exact::leased_scale(net, -1, scalar_budget) : net;
                        if (magnitude.compare(certificate.owned_cell_radius[c]) <= 0)
                            continue;
                        ++local_violations;
                        if (detail.empty())
                        {
                            std::ostringstream out;
                            out << std::setprecision(18) << "cell=" << self.mesh_->cell_global_id(LO(c))
                                << ", net_rate=" << net.enclosure().value
                                << ", radius=" << certificate.owned_cell_radius[c].enclosure().value
                                << ", net_bytes=" << net.storage_bytes()
                                << ", radius_bytes=" << certificate.owned_cell_radius[c].storage_bytes();
                            detail = out.str();
                            witness = comm->getRank();
                        }
                    }
                });
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_violations, &violations);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_zero_faces, &zero_faces);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &witness, &owner);
            int count = comm->getRank() == owner ? int(detail.size()) : 0;
            Teuchos::broadcast(*comm, owner, 1, &count);
            arithmetic_guard([&] { detail.resize(std::size_t(count)); });
            Teuchos::broadcast(*comm, owner, count, detail.data());
            closure_reason = "cell operation envelope exceeded: violations=" + std::to_string(violations) +
                             ", zero_radius_owned_faces=" + std::to_string(zero_faces) + ", " + detail;
            return false;
        }
        std::vector<GeneralVoFPhiCell> cells;
        std::vector<GeneralVoFPhiFace> faces;
        arithmetic_guard(
            [&]
            {
                cells = closure_topology->cells;
                faces = closure_topology->faces;
                for (auto& cell : cells)
                {
                    const auto row = self.mesh_->owned_cell_map()->getLocalElement(cell.cell);
                    if (row < 0)
                        throw std::runtime_error("Foreign rate-closure cell.");
                    cell.arithmetic_radius = certificate.owned_cell_radius[row];
                }
                for (auto& face : faces)
                {
                    const auto local = self.mesh_->owned_face_map()->getLocalElement(face.face);
                    if (local < 0)
                        throw std::runtime_error("Foreign rate-closure face.");
                    face.volume = certificate.owned_rate[local];
                    face.arithmetic_radius = certificate.owned_face_radius[local];
                }
            });
        std::optional<GeneralVoFPhiClosure> closed;
        try
        {
            closed = close_general_vof_bounded_rate(self.mesh_, cells, faces, admission_budget);
        }
        catch (const GeneralVoFPhiInfeasible& error)
        {
            closure_reason = error.what();
            return false;
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error(std::string("Primary rate forest: ") + error.what() + capacity_context());
        }
        closure_reason = "feasible";
        arithmetic_guard(
            [&]
            {
                accepted_rate = certificate.owned_rate;
                for (std::size_t f = 0; f < faces.size(); ++f)
                    accepted_rate[self.mesh_->owned_face_map()->getLocalElement(faces[f].face)] = closed->volume[f];
            });
        return true;
    };
    const auto dot_arithmetic = [&](const Vector& left, const Vector& right)
    {
        auto value = arithmetic.multiply(arithmetic.leaf(left[0]), arithmetic.leaf(right[0]));
        for (unsigned axis = 1; axis < 3; ++axis)
            value =
                arithmetic.add(value, arithmetic.multiply(arithmetic.leaf(left[axis]), arithmetic.leaf(right[axis])));
        return value;
    };
    const auto rho = self.cells(density), f = self.cells(fraction), k = self.cells(kappa);
    const auto grad_f = self.vectors(self.gradient(fraction));
    std::vector<double> beta(self.geometry_.size()), capillary(self.geometry_.size()),
        fraction_face(self.geometry_.size()), curvature_face(self.geometry_.size());
    arithmetic_guard(
        [&]
        {
            for (std::size_t id = 0; id < self.mesh_->num_owned_faces(); ++id)
            {
                const auto& g = self.geometry_[id];
                if (g.boundary)
                    continue;
                const double mass_length = rho[g.owner] * g.left + rho[g.neighbor] * g.right;
                beta[id] = g.distance / mass_length;
                const double left = f[g.owner] + dot(grad_f[g.owner], g.tangent_left);
                const double right = f[g.neighbor] + dot(grad_f[g.neighbor], g.tangent_right);
                fraction_face[id] = (rho[g.neighbor] * g.right * left + rho[g.owner] * g.left * right) / mass_length;
                curvature_face[id] = (g.right * k[g.owner] + g.left * k[g.neighbor]) / g.distance;
                capillary[id] = sigma * curvature_face[id] * (right - left) / g.distance;
                if (provenance)
                {
                    const auto left_value =
                        arithmetic.add(arithmetic.leaf(f[g.owner]), dot_arithmetic(grad_f[g.owner], g.tangent_left));
                    const auto right_value = arithmetic.add(
                        arithmetic.leaf(f[g.neighbor]), dot_arithmetic(grad_f[g.neighbor], g.tangent_right));
                    const auto curvature_value = arithmetic.divide(
                        arithmetic.add(arithmetic.multiply(arithmetic.leaf(g.right), arithmetic.leaf(k[g.owner])),
                            arithmetic.multiply(arithmetic.leaf(g.left), arithmetic.leaf(k[g.neighbor]))),
                        arithmetic.leaf(g.distance));
                    const auto capillary_value = arithmetic.divide(
                        arithmetic.multiply(arithmetic.multiply(arithmetic.leaf(sigma), curvature_value),
                            arithmetic.subtract(right_value, left_value)),
                        arithmetic.leaf(g.distance));
                    rounded[id] = arithmetic.add(rounded[id],
                        arithmetic.multiply(
                            arithmetic.multiply(arithmetic.leaf(dt), arithmetic.leaf(beta[id])), capillary_value));
                    face_velocity[id] = rounded[id].value;
                }
                else
                    face_velocity[id] += dt * beta[id] * capillary[id];
            }
        });
    beta = self.faces(beta);
    fraction_face = self.faces(fraction_face);
    curvature_face = self.faces(curvature_face);
    face_velocity = self.faces(face_velocity);
    std::vector<Vector> force(self.volumes_.size());
    {
        const auto lease = self.mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < self.volumes_.size(); ++c)
            for (auto fid : self.mesh_->faces(LO(c)))
            {
                const auto& g = self.geometry_[fid];
                if (g.boundary)
                {
                    // Use exactly the pressure boundary extrapolation for the
                    // capillary normal reaction. For constant kappa, p=sigma*
                    // kappa*f then cancels on the complete control volume.
                    const double value =
                        sigma * kappa[c] * dot(grad_f[c], convert(self.mesh_->face_center_vector(fid, LO(c)))) * g.area;
                    for (unsigned a = 0; a < 3; ++a)
                        force[c][a] += value * g.normal[a];
                    continue;
                }
                const double value = (g.owner == LO(c) ? 1 : -1) * g.area * sigma * curvature_face[fid] *
                                     (fraction_face[fid] - fraction[c]);
                for (unsigned a = 0; a < 3; ++a)
                    force[c][a] += value * g.normal[a];
            }
    }
    pressure.assign(self.volumes_.size(), 0);
    arithmetic_guard(
        [&]
        {
            exact_pressure.resize(self.volumes_.size());
            exact_divergence.resize(self.volumes_.size());
            exact_rhs.resize(self.volumes_.size());
            exact_force.resize(self.volumes_.size());
            if (provenance)
            {
                primary_rate.resize(rounded.size());
                primary_radius.resize(rounded.size());
                for (std::size_t f = 0; f < rounded.size(); ++f)
                {
                    primary_rate[f] =
                        vof_exact::leased_scale(M::from_value(rounded[f].value), self.geometry_[f].area, scalar_budget);
                    primary_radius[f] =
                        vof_exact::leased_scale(arithmetic.radius(rounded[f]), self.geometry_[f].area, scalar_budget);
                }
                for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                    for (unsigned a = 0; a < 3; ++a)
                        exact_force[c][a] = M::from_value(force[c][a]);
            }
        });
    auto div = self.divergence(face_velocity);
    const auto norm = [&](const std::vector<double>& values)
    {
        double local = 0;
        for (double v : values)
            local = std::isfinite(v) ? std::max(local, std::abs(v)) : std::numeric_limits<double>::infinity();
        return self.maximum(local);
    };
    diagnostics.pressure_iterations = 0;
    bool arithmetic_admitted = rate_admitted();
    const auto primary_divergence_maximum = [&]
    {
        arithmetic_guard(
            [&]
            {
                for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                {
                    exact_divergence[c] = vof_exact::leased_divide(
                        current_certificate.owned_net_rate[c], M::from_value(self.volumes_[c]), scalar_budget);
                    div[c] = static_cast<double>(exact_divergence[c].enclosure().value);
                }
            });
        return vof_exact::leased_maximum_absolute(
            exact_divergence, *self.mesh_->owned_cell_map(), scalar_budget, admission_budget);
    };
    // Fix the relative reference once from the actual initial primary rate.
    // Algebraically dt*(abs+rel*(initial/dt))=dt*abs+rel*initial;
    // no rounded zero or division underflow can erase that reference.
    const auto initial_maximum = provenance ? primary_divergence_maximum() : M::from_value(norm(div));
    M exact_limit;
    arithmetic_guard(
        [&]
        {
            exact_limit =
                general_vof_projection_divergence_limit(initial_maximum, dt, options.pressure_absolute_tolerance,
                    options.pressure_relative_tolerance, options.divergence_tolerance, scalar_budget);
        });
    const double limit = static_cast<double>(exact_limit.enclosure().value);
    self.require(exact_limit.sign() > 0, "General VoF projection has invalid physical tolerance.");
    const auto physical_admission = [&]
    {
        if (!provenance)
            return norm(div) <= limit;
        return primary_divergence_maximum().compare(exact_limit) <= 0;
    };
    bool physical_admitted = physical_admission();
    const auto accumulate_exact_pressure = [&](const std::vector<M>& p)
    {
        arithmetic_guard(
            [&]
            {
                if (p.size() != self.mesh_->num_cells())
                    throw std::runtime_error("Exact pressure halo shape mismatch.");
                const auto lease = self.mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                {
                    // Freeze the same represented geometry-only WLS matrix and
                    // solve its pressure response exactly; no tiny p is narrowed.
                    std::array<std::array<double, 3>, 3> metric{};
                    std::array<std::array<M, 4>, 3> matrix{};
                    for (auto fid : self.mesh_->faces(LO(c)))
                    {
                        const auto& g = self.geometry_[fid];
                        if (g.boundary)
                            continue;
                        auto r = convert(self.mesh_->cell_center_vector(fid, LO(c)));
                        for (unsigned a = 0; a < 3; ++a)
                            r[a] /= self.spans_[c][a];
                        const double w = 1 / dot(r, r);
                        const auto delta = vof_exact::leased_add(p[self.mesh_->opposite_cell(fid, LO(c))],
                            vof_exact::leased_scale(p[c], -1, scalar_budget), scalar_budget);
                        for (unsigned a = 0; a < 3; ++a)
                        {
                            matrix[a][3] = vof_exact::leased_add(
                                matrix[a][3], vof_exact::leased_scale(delta, w * r[a], scalar_budget), scalar_budget);
                            for (unsigned b = 0; b < 3; ++b)
                                metric[a][b] += w * r[a] * r[b];
                        }
                    }
                    for (unsigned a = 0; a < 3; ++a)
                        for (unsigned b = 0; b < 3; ++b)
                            matrix[a][b] = M::from_value(metric[a][b]);
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        unsigned pivot = a;
                        for (unsigned row = a + 1; row < 3; ++row)
                        {
                            const auto magnitude = matrix[row][a].sign() < 0
                                                       ? vof_exact::leased_scale(matrix[row][a], -1, scalar_budget)
                                                       : matrix[row][a];
                            const auto previous = matrix[pivot][a].sign() < 0
                                                      ? vof_exact::leased_scale(matrix[pivot][a], -1, scalar_budget)
                                                      : matrix[pivot][a];
                            if (magnitude.compare(previous) > 0)
                                pivot = row;
                        }
                        if (!matrix[pivot][a].sign())
                            throw std::runtime_error("Exact pressure WLS matrix is singular.");
                        std::swap(matrix[a], matrix[pivot]);
                        const auto diagonal = matrix[a][a];
                        for (unsigned b = a; b < 4; ++b)
                            matrix[a][b] = vof_exact::leased_divide(matrix[a][b], diagonal, scalar_budget);
                        for (unsigned row = 0; row < 3; ++row)
                            if (row != a)
                            {
                                const auto factor = matrix[row][a];
                                for (unsigned b = a; b < 4; ++b)
                                    matrix[row][b] = vof_exact::leased_add(matrix[row][b],
                                        vof_exact::leased_scale(exact_product(factor, matrix[a][b]), -1, scalar_budget),
                                        scalar_budget);
                            }
                    }
                    std::array<M, 3> grad;
                    for (unsigned a = 0; a < 3; ++a)
                        grad[a] =
                            vof_exact::leased_divide(matrix[a][3], M::from_value(self.spans_[c][a]), scalar_budget);
                    exact_pressure[c] = vof_exact::leased_add(exact_pressure[c], p[c], scalar_budget);
                    for (auto fid : self.mesh_->faces(LO(c)))
                    {
                        const auto& g = self.geometry_[fid];
                        M value;
                        if (g.boundary)
                        {
                            const auto offset = convert(self.mesh_->face_center_vector(fid, LO(c)));
                            for (unsigned a = 0; a < 3; ++a)
                                value = vof_exact::leased_add(
                                    value, vof_exact::leased_scale(grad[a], -offset[a], scalar_budget), scalar_budget);
                        }
                        else
                        {
                            const auto left_weight =
                                vof_exact::leased_scale(M::from_value(rho[g.neighbor]), g.right, scalar_budget);
                            const auto right_weight =
                                vof_exact::leased_scale(M::from_value(rho[g.owner]), g.left, scalar_budget);
                            const auto face_p =
                                vof_exact::leased_divide(vof_exact::leased_add(exact_product(left_weight, p[g.owner]),
                                                             exact_product(right_weight, p[g.neighbor]), scalar_budget),
                                    vof_exact::leased_add(left_weight, right_weight, scalar_budget), scalar_budget);
                            value = vof_exact::leased_scale(
                                vof_exact::leased_add(
                                    face_p, vof_exact::leased_scale(p[c], -1, scalar_budget), scalar_budget),
                                g.owner == LO(c) ? -1 : 1, scalar_budget);
                        }
                        value = vof_exact::leased_scale(value, g.area, scalar_budget);
                        for (unsigned a = 0; a < 3; ++a)
                            exact_force[c][a] = vof_exact::leased_add(exact_force[c][a],
                                vof_exact::leased_scale(value, g.normal[a], scalar_budget), scalar_budget);
                    }
                }
            });
    };
    int local_orthogonal = 1, orthogonal = 0;
    for (std::size_t f = 0; f < self.mesh_->num_owned_faces(); ++f)
        if (!self.geometry_[f].boundary)
            for (unsigned a = 0; a < 3; ++a)
                local_orthogonal = local_orthogonal && self.geometry_[f].tangent_left[a] == 0 &&
                                   self.geometry_[f].tangent_right[a] == 0;
    Teuchos::reduceAll(
        *self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_orthogonal, &orthogonal);
    bool exact_fallback_used = false;
    if (!physical_admitted || !arithmetic_admitted)
    {
        self.projection_.refresh(beta, nullptr, dt, self.gauge_);
        LinearSolverOptions linear;
        linear.backend = LinearSolverBackend::Cg;
        linear.preconditioner = LinearPreconditioner::Jacobi;
        linear.reuse_preconditioner = true;
        linear.tolerance = 1e-8;
        linear.verbosity = 0;
        for (unsigned correction = 0; correction < 64 && (!physical_admitted || !arithmetic_admitted); ++correction)
        {
            ++correction_count;
            linear.max_iterations = options.max_pressure_iterations - diagnostics.pressure_iterations;
            if (linear.max_iterations <= 0)
                break;
            int local_zero_radius_defect = 0, zero_radius_defect = 0;
            if (provenance)
                for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                    local_zero_radius_defect =
                        local_zero_radius_defect || (current_certificate.owned_net_rate[c].sign() != 0 &&
                                                        current_certificate.owned_cell_radius[c].sign() == 0);
            Teuchos::reduceAll(*self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
                &local_zero_radius_defect, &zero_radius_defect);
            if (provenance && closure_topology && orthogonal && zero_radius_defect && !exact_fallback_used)
            {
                exact_fallback_used = true;
                {
                    // The old certificate occupies one helper half; the exact
                    // solve/result occupies the other. Destroy its result before
                    // another certificate/forest allocation uses that half.
                    std::vector<GeneralVoFExactPressureFace> exact_faces;
                    arithmetic_guard(
                        [&]
                        {
                            exact_faces.reserve(self.mesh_->num_owned_faces());
                            for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                                exact_rhs[c] =
                                    vof_exact::leased_scale(current_certificate.owned_net_rate[c], -1, scalar_budget);
                            for (std::size_t f = 0; f < self.mesh_->num_owned_faces(); ++f)
                            {
                                const auto& g = self.geometry_[f];
                                if (g.boundary)
                                    continue;
                                const auto mobility = vof_exact::leased_divide(
                                    vof_exact::leased_scale(
                                        vof_exact::leased_scale(M::from_value(dt), beta[f], scalar_budget), g.area,
                                        scalar_budget),
                                    M::from_value(g.distance), scalar_budget);
                                if (mobility.sign() <= 0)
                                    throw std::runtime_error("Nonpositive exact orthogonal pressure mobility.");
                                exact_faces.push_back(
                                    {self.mesh_->face_global_id(LO(f)), self.mesh_->cell_global_id(g.owner),
                                        self.mesh_->cell_global_id(g.neighbor), mobility});
                            }
                        });
                    const auto result = solve_general_vof_exact_pressure(
                        self.mesh_, exact_faces, exact_rhs, linear.max_iterations, admission_budget);
                    self.require(result.iterations >= 0 && result.iterations <= linear.max_iterations,
                        "Exact pressure helper exceeded remaining shared iterations.");
                    self.require(
                        result.admitted_storage_bytes <= admission_budget && result.storage_bytes <= admission_budget,
                        "Exact pressure helper result exceeded its simultaneous workspace lease.");
                    diagnostics.pressure_iterations += result.iterations;
                    accumulate_exact_pressure(result.overlap_pressure);
                    arithmetic_guard(
                        [&]
                        {
                            if (result.face_rate_increment.size() != exact_faces.size())
                                throw std::runtime_error("Incomplete exact pressure rate response.");
                            for (std::size_t i = 0; i < exact_faces.size(); ++i)
                            {
                                const auto f = self.mesh_->owned_face_map()->getLocalElement(exact_faces[i].face);
                                primary_rate[f] = vof_exact::leased_add(
                                    primary_rate[f], result.face_rate_increment[i], scalar_budget);
                                const auto view = vof_exact::leased_divide(
                                    primary_rate[f], M::from_value(self.geometry_[f].area), scalar_budget);
                                face_velocity[f] = static_cast<double>(view.enclosure().value);
                                if (!std::isfinite(face_velocity[f]))
                                    throw std::runtime_error("Exact pressure rate view overflows.");
                                rounded[f] = arithmetic.leaf(face_velocity[f]);
                            }
                        });
                }
                face_velocity = self.faces(face_velocity);
                div = self.divergence(face_velocity);
                arithmetic_admitted = rate_admitted();
                physical_admitted = physical_admission();
                continue;
            }
            M pressure_scale = M::from_value(1);
            if (provenance)
            {
                arithmetic_guard(
                    [&]
                    {
                        for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                            exact_rhs[c] = vof_exact::leased_divide(
                                vof_exact::leased_scale(current_certificate.owned_net_rate[c], -1, scalar_budget),
                                M::from_value(dt), scalar_budget);
                    });
                const auto maximum = vof_exact::leased_maximum_absolute(
                    exact_rhs, *self.mesh_->owned_cell_map(), scalar_budget, admission_budget);
                std::optional<M> proposed_scale;
                int local_scale_admitted = 1, scale_admitted = 0;
                arithmetic_guard(
                    [&]
                    {
                        try
                        {
                            proposed_scale.emplace(general_vof_pressure_normalization_scale(maximum, scalar_budget));
                        }
                        catch (const std::length_error&)
                        {
                            local_scale_admitted = 0;
                        }
                    });
                Teuchos::reduceAll(*self.mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1,
                    &local_scale_admitted, &scale_admitted);
                arithmetic_guard(
                    [&]
                    {
                        // Rank-local backend capacities must not choose
                        // different normalizers for the same pressure solve.
                        pressure_scale =
                            scale_admitted ? *proposed_scale : (maximum.sign() ? maximum : M::from_value(1));
                    });
            }
            arithmetic_guard(
                [&]
                {
                    for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                    {
                        double rhs = -div[c] * self.volumes_[c] / dt;
                        if (provenance)
                            rhs = static_cast<double>(
                                vof_exact::leased_divide(exact_rhs[c], pressure_scale, scalar_budget)
                                    .enclosure()
                                    .value);
                        self.projection_.rhs.replaceLocalValue(
                            LO(c), self.mesh_->cell_global_id(LO(c)) == self.gauge_ ? 0 : rhs);
                    }
                });
            const auto stats = self.projection_.solver.solve_from_zero_with_statistics(
                Teuchos::rcp_implicit_cast<const Pack::operator_type>(self.projection_.matrix), self.projection_.rhs,
                self.projection_.solution, linear);
            diagnostics.pressure_iterations += stats.iterations;
            self.require(std::isfinite(stats.achieved_tolerance) && std::isfinite(stats.rhs_norm),
                "General VoF pressure solve is nonfinite.");
            M rate_pressure_scale;
            arithmetic_guard([&] { rate_pressure_scale = vof_exact::leased_scale(pressure_scale, dt, scalar_budget); });
            const auto data = self.projection_.solution.getData();
            std::vector<double> p_owned(data.begin(), data.end());
            const auto p = self.cells(p_owned);
            const auto grad_p = self.vectors(self.gradient(p_owned));
            std::vector<double> p_face(self.geometry_.size());
            arithmetic_guard(
                [&]
                {
                    for (std::size_t id = 0; id < self.mesh_->num_owned_faces(); ++id)
                    {
                        const auto& g = self.geometry_[id];
                        if (g.boundary)
                            continue;
                        const double left = p[g.owner] + dot(grad_p[g.owner], g.tangent_left);
                        const double right = p[g.neighbor] + dot(grad_p[g.neighbor], g.tangent_right);
                        if (provenance)
                        {
                            const auto left_value = arithmetic.add(
                                arithmetic.leaf(p[g.owner]), dot_arithmetic(grad_p[g.owner], g.tangent_left));
                            const auto right_value = arithmetic.add(
                                arithmetic.leaf(p[g.neighbor]), dot_arithmetic(grad_p[g.neighbor], g.tangent_right));
                            const auto correction_value =
                                arithmetic.divide(arithmetic.multiply(arithmetic.leaf(beta[id]),
                                                      arithmetic.subtract(right_value, left_value)),
                                    arithmetic.leaf(g.distance));
                            // Evaluate the normalized correction once. Exact physical
                            // scaling and accumulation preserve corrections below a
                            // stored face velocity's ulp; no u-dc rounding is invented.
                            const auto decrement = vof_exact::leased_scale(
                                exact_product(rate_pressure_scale, M::from_value(correction_value.value)), g.area,
                                scalar_budget);
                            primary_rate[id] = vof_exact::leased_add(
                                primary_rate[id], vof_exact::leased_scale(decrement, -1, scalar_budget), scalar_budget);
                            primary_radius[id] = vof_exact::leased_add(primary_radius[id],
                                vof_exact::leased_scale(
                                    exact_product(rate_pressure_scale, arithmetic.radius(correction_value)), g.area,
                                    scalar_budget),
                                scalar_budget);
                            const auto view =
                                vof_exact::leased_divide(primary_rate[id], M::from_value(g.area), scalar_budget);
                            face_velocity[id] = static_cast<double>(view.enclosure().value);
                            if (!std::isfinite(face_velocity[id]))
                                throw std::runtime_error("Primary pressure rate diagnostic overflow.");
                            rounded[id] = arithmetic.leaf(face_velocity[id]);
                        }
                        else
                            face_velocity[id] -= dt * beta[id] * (right - left) / g.distance;
                        p_face[id] = (rho[g.neighbor] * g.right * left + rho[g.owner] * g.left * right) /
                                     (rho[g.owner] * g.left + rho[g.neighbor] * g.right);
                    }
                });
            face_velocity = self.faces(face_velocity);
            p_face = self.faces(p_face);
            arithmetic_guard(
                [&]
                {
                    const auto lease = self.mesh_->acquire_execution_view();
                    for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                    {
                        if (provenance)
                            exact_pressure[c] = vof_exact::leased_add(exact_pressure[c],
                                vof_exact::leased_scale(pressure_scale, p_owned[c], scalar_budget), scalar_budget);
                        else
                            pressure[c] += p_owned[c];
                        for (auto fid : self.mesh_->faces(LO(c)))
                        {
                            const auto& g = self.geometry_[fid];
                            // Extrapolate the same physical pressure gradient to
                            // an impermeable face: its reaction completes the
                            // pressure force on a boundary control volume.
                            const double value =
                                g.boundary
                                    ? -dot(grad_p[c], convert(self.mesh_->face_center_vector(fid, LO(c)))) * g.area
                                    : -(g.owner == LO(c) ? 1 : -1) * (p_face[fid] - p_owned[c]) * g.area;
                            for (unsigned a = 0; a < 3; ++a)
                                if (provenance)
                                    exact_force[c][a] = vof_exact::leased_add(exact_force[c][a],
                                        vof_exact::leased_scale(
                                            vof_exact::leased_scale(pressure_scale, value, scalar_budget), g.normal[a],
                                            scalar_budget),
                                        scalar_budget);
                                else
                                    force[c][a] += value * g.normal[a];
                        }
                    }
                });
            div = self.divergence(face_velocity);
            arithmetic_admitted = rate_admitted();
            physical_admitted = physical_admission();
        }
    }
    const double residual = norm(div);
    if (!(std::isfinite(residual) && physical_admitted && arithmetic_admitted))
    {
        long double local_exact_divergence = 0, diagnostic_exact_divergence = 0;
        int local_erased = 0, erased = 0;
        arithmetic_guard(
            [&]
            {
                const auto lease = self.mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                {
                    Meshes::ExactGeometryMeasure rate;
                    for (auto fid : self.mesh_->faces(LO(c)))
                    {
                        const auto& g = self.geometry_[fid];
                        const auto contribution = vof_exact::leased_scale(
                            vof_exact::leased_scale(
                                Meshes::ExactGeometryMeasure::from_value(face_velocity[fid]), g.area, scalar_budget),
                            g.owner == LO(c) ? 1 : -1, scalar_budget);
                        rate = vof_exact::leased_add(rate, contribution, scalar_budget);
                    }
                    const auto exact =
                        provenance ? exact_divergence[c]
                                   : vof_exact::leased_divide(rate,
                                         Meshes::ExactGeometryMeasure::from_value(self.volumes_[c]), scalar_budget);
                    local_exact_divergence = std::max(local_exact_divergence, std::abs(exact.enclosure().value));
                    local_erased += exact.sign() != 0 && div[c] == 0;
                }
            });
        const auto comm = self.mesh_->owned_cell_map()->getComm();
        // Teuchos has no portable long-double packet serialization. Reduce a
        // diagnostic binary exponent followed by its mantissa, retaining tiny
        // values without making this diagnostic part of numerical acceptance.
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        double local_mantissa = 0, mantissa = 0;
        if (local_exact_divergence > 0)
            local_mantissa = static_cast<double>(std::frexp(local_exact_divergence, &local_exponent));
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        if (local_exponent != exponent)
            local_mantissa = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_mantissa, &mantissa);
        if (mantissa > 0)
            diagnostic_exact_divergence = std::ldexp(static_cast<long double>(mantissa), exponent);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_erased, &erased);
        std::ostringstream message;
        message << std::setprecision(18)
                << "General VoF nonorthogonal projection failed physical or arithmetic divergence acceptance: residual="
                << residual << ", limit=" << limit << ", arithmetic=" << arithmetic_admitted
                << ", closure=" << closure_reason << ", corrections=" << correction_count
                << ", iterations=" << diagnostics.pressure_iterations
                << ", exact_rate_divergence=" << diagnostic_exact_divergence
                << ", nonzero_exact_rows_with_zero_float_rhs=" << erased;
        throw std::runtime_error(message.str());
    }
    arithmetic_guard(
        [&]
        {
            for (std::size_t c = 0; c < self.volumes_.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto physical_force = provenance ? exact_force[c][a] : M::from_value(force[c][a]);
                    if (!physical_force.sign())
                        continue;
                    // Normalize before narrowing: a represented density and volume
                    // may have a nonzero product below binary64 carrier range.
                    const auto mass = physical_mass ? (*physical_mass)[c]
                                                    : vof_exact::leased_scale(
                                                          Meshes::ExactGeometryMeasure::from_value(self.volumes_[c]),
                                                          density[c], scalar_budget);
                    const auto impulse = vof_exact::leased_divide(
                        vof_exact::leased_scale(physical_force, dt, scalar_budget), mass, scalar_budget);
                    const auto updated = vof_exact::leased_add(
                        Meshes::ExactGeometryMeasure::from_value(velocity[c][a]), impulse, scalar_budget);
                    velocity[c][a] = updated.sign() ? updated.certified().as_double() : 0;
                }
        });
    if (provenance)
        arithmetic_guard(
            [&]
            {
                for (std::size_t c = 0; c < pressure.size(); ++c)
                    pressure[c] = exact_pressure[c].sign() ? exact_pressure[c].certified().as_double() : 0;
            });
    for (const auto& u : velocity)
        for (double value : u)
            valid = valid && std::isfinite(value);
    for (double value : pressure)
        valid = valid && std::isfinite(value);
    for (double value : face_velocity)
        valid = valid && std::isfinite(value);
    self.require(valid, "General VoF projection produced a nonfinite candidate.");
    diagnostics.maximum_divergence = residual;
    diagnostics.pressure_residual = residual / dt;
    if (provenance)
    {
        provenance->face_velocity = std::move(rounded);
        provenance->face_rate = std::move(accepted_rate);
    }
}
} // namespace Exact
} // namespace SimpleFluid::detail
