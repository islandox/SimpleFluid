/** @file GeneralVoFFlowWorkspace.cc
 * @brief Conditioned native gradients, stress transport and nonorthogonal projection.
 */
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "geometry/ConvexCellGeometry.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactPressure.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using Vector = VoF::Vector;
double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector convert(const MeshUtils::Vec3& a)
{
    return {a.x, a.y, a.z};
}
SP<const MeshHandle<>> validated_native_flow_mesh(SP<const MeshHandle<>> mesh)
{
    bool accepted = true;
    try
    {
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            (void) mesh->faces(static_cast<DefaultTpetraTypes::local_ordinal_type>(c)).size();
    }
    catch (...)
    {
        accepted = false;
    }
    int local = accepted, global = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("General VoF native source failed collective workspace preflight.");
    return mesh;
}
} // namespace
void GeneralVoFFlowWorkspace::require(bool valid, const char* message) const
{
    int local = valid, global = 0;
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error(message);
}
double GeneralVoFFlowWorkspace::maximum(double value) const
{
    double global = 0;
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &value, &global);
    return global;
}
GeneralVoFFlowWorkspace::GeneralVoFFlowWorkspace(SP<const MeshHandle<>> mesh)
    : mesh_(validated_native_flow_mesh(std::move(mesh))),
      cell_import_(mesh_->owned_cell_map(), mesh_->overlap_cell_map()),
      face_import_(mesh_->owned_face_map(), mesh_->overlap_face_map()), cell_owned_(mesh_->owned_cell_map()),
      cell_overlap_(mesh_->overlap_cell_map()), face_owned_(mesh_->owned_face_map()),
      face_overlap_(mesh_->overlap_face_map()), projection_(mesh_)
{
    bool valid = true;
    auto local_gauge = std::numeric_limits<Pack::global_ordinal_type>::max();
    try
    {
        const auto lease = mesh_->acquire_execution_view();
        volumes_.resize(mesh_->num_owned_cells());
        exact_volumes_.resize(mesh_->num_owned_cells());
        spans_.resize(volumes_.size());
        for (std::size_t c = 0; c < volumes_.size(); ++c)
        {
            const LO cell = LO(c);
            volumes_[c] = mesh_->cell_volume(cell);
            local_gauge = std::min(local_gauge, mesh_->cell_global_id(cell));
            const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&mesh_->variant());
            if (!native || !*native)
            {
                valid = false;
                continue;
            }
            const auto exact_cell =
                Meshes::ConvexCellGeometry::from_native_cell(**native, mesh_->cell_geometry_global_id(cell));
            exact_volumes_[c] = Meshes::ExactGeometryMeasure::volume(*exact_cell);
            const auto exact_spans = exact_cell->axis_spans();
            for (unsigned a = 0; a < 3; ++a)
            {
                spans_[c][a] = exact_spans[a].as_double();
                valid = valid && std::isfinite(spans_[c][a]) && spans_[c][a] > 0;
            }
        }
        geometry_.resize(mesh_->num_faces());
        for (std::size_t id = 0; id < geometry_.size(); ++id)
        {
            const LO fid = LO(id);
            auto& f = geometry_[id];
            f.owner = mesh_->owner_cell(fid);
            f.neighbor = mesh_->neighbor_cell(fid);
            f.boundary = mesh_->is_geometry_exterior_face(fid);
            if (f.owner < 0 || (!f.boundary && f.neighbor < 0))
            {
                valid = valid && !mesh_->is_owned_face(fid);
                continue;
            }
            f.area = mesh_->face_area(fid);
            f.normal = convert(mesh_->face_normal(fid));
            auto r = convert(mesh_->face_center_vector(fid, f.owner));
            f.left = dot(r, f.normal);
            f.tangent_left = r;
            if (!f.boundary)
            {
                r = convert(mesh_->face_center_vector(fid, f.neighbor));
                f.right = -dot(r, f.normal);
                f.tangent_right = r;
            }
            for (unsigned a = 0; a < 3; ++a)
            {
                f.tangent_left[a] -= f.left * f.normal[a];
                f.tangent_right[a] += f.right * f.normal[a];
            }
            f.distance = f.left + f.right;
            valid = valid && f.left > 0 && (f.boundary || f.right > 0) && f.area > 0;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(valid, "General VoF native physical spans or face distances are invalid.");
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_gauge, &gauge_);
}
std::vector<double> GeneralVoFFlowWorkspace::cells(const std::vector<double>& owned) const
{
    require(owned.size() == mesh_->num_owned_cells(), "General VoF owned scalar shape mismatch.");
    {
        auto d = cell_owned_.getDataNonConst();
        std::copy(owned.begin(), owned.end(), d.begin());
    }
    cell_overlap_.doImport(cell_owned_, cell_import_, Tpetra::INSERT);
    auto d = cell_overlap_.getData();
    return {d.begin(), d.end()};
}
std::vector<double> GeneralVoFFlowWorkspace::faces(const std::vector<double>& overlap) const
{
    require(overlap.size() == mesh_->num_faces(), "General VoF overlap face shape mismatch.");
    {
        auto d = face_owned_.getDataNonConst();
        std::copy_n(overlap.begin(), d.size(), d.begin());
    }
    face_overlap_.doImport(face_owned_, face_import_, Tpetra::INSERT);
    auto d = face_overlap_.getData();
    return {d.begin(), d.end()};
}
std::vector<Vector> GeneralVoFFlowWorkspace::vectors(const std::vector<Vector>& owned) const
{
    require(owned.size() == mesh_->num_owned_cells(), "General VoF owned vector shape mismatch.");
    std::vector<Vector> result(mesh_->num_cells());
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<double> scalar(owned.size());
        for (std::size_t c = 0; c < owned.size(); ++c)
            scalar[c] = owned[c][a];
        auto halo = cells(scalar);
        for (std::size_t c = 0; c < halo.size(); ++c)
            result[c][a] = halo[c];
    }
    return result;
}
std::vector<Vector> GeneralVoFFlowWorkspace::gradient(const std::vector<double>& owned) const
{
    const auto halo = cells(owned);
    std::vector<Vector> result(owned.size());
    bool valid = true;
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < owned.size(); ++c)
        {
            std::array<std::array<double, 7>, 3> m{};
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto& f = geometry_[fid];
                if (f.boundary)
                    continue;
                auto r = convert(mesh_->cell_center_vector(fid, LO(c)));
                for (unsigned a = 0; a < 3; ++a)
                    r[a] /= spans_[c][a];
                const double w = 1 / dot(r, r), delta = halo[mesh_->opposite_cell(fid, LO(c))] - owned[c];
                for (unsigned a = 0; a < 3; ++a)
                {
                    m[a][3] += w * r[a] * delta;
                    for (unsigned b = 0; b < 3; ++b)
                        m[a][b] += w * r[a] * r[b];
                }
            }
            double norm = 0;
            for (unsigned row = 0; row < 3; ++row)
            {
                double sum = 0;
                for (unsigned column = 0; column < 3; ++column)
                    sum += std::abs(m[row][column]);
                norm = std::max(norm, sum);
                m[row][4 + row] = 1;
            }
            for (unsigned a = 0; a < 3; ++a)
            {
                unsigned pivot = a;
                for (unsigned b = a + 1; b < 3; ++b)
                    if (std::abs(m[b][a]) > std::abs(m[pivot][a]))
                        pivot = b;
                if (!(std::abs(m[pivot][a]) > 1e-12))
                {
                    valid = false;
                    break;
                }
                std::swap(m[a], m[pivot]);
                const double d = m[a][a];
                for (unsigned b = a; b < 7; ++b)
                    m[a][b] /= d;
                for (unsigned row = 0; row < 3; ++row)
                    if (row != a)
                    {
                        const double factor = m[row][a];
                        for (unsigned b = a; b < 7; ++b)
                            m[row][b] -= factor * m[a][b];
                    }
            }
            double inverse_norm = 0;
            for (unsigned row = 0; row < 3; ++row)
            {
                double sum = 0;
                for (unsigned column = 4; column < 7; ++column)
                    sum += std::abs(m[row][column]);
                inverse_norm = std::max(inverse_norm, sum);
            }
            valid = valid && std::isfinite(norm * inverse_norm) && norm * inverse_norm <= 1e8;
            for (unsigned a = 0; a < 3; ++a)
            {
                result[c][a] = m[a][3] / spans_[c][a];
                valid = valid && std::isfinite(result[c][a]);
            }
        }
    }
    require(valid, "General VoF physical least-squares stencil is deficient or ill-conditioned.");
    return result;
}
std::vector<double> GeneralVoFFlowWorkspace::curvature(const std::vector<Vector>& normals) const
{
    require(normals.size() == volumes_.size(), "General VoF curvature normal shape mismatch.");
    std::vector<double> result(volumes_.size());
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<double> component(normals.size());
        for (std::size_t c = 0; c < normals.size(); ++c)
            component[c] = normals[c][a];
        auto derivative = gradient(component);
        for (std::size_t c = 0; c < normals.size(); ++c)
            result[c] += derivative[c][a];
    }
    return result;
}
std::vector<double> GeneralVoFFlowWorkspace::interpolate(const std::vector<Vector>& velocity) const
{
    const auto halo = vectors(velocity);
    std::array<std::vector<Vector>, 3> gradients;
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<double> component(velocity.size());
        for (std::size_t c = 0; c < velocity.size(); ++c)
            component[c] = velocity[c][a];
        gradients[a] = vectors(gradient(component));
    }
    std::vector<double> result(geometry_.size());
    for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
    {
        const auto& f = geometry_[id];
        if (f.boundary)
            continue;
        for (unsigned a = 0; a < 3; ++a)
            result[id] += f.normal[a] *
                          (f.right * (halo[f.owner][a] + dot(gradients[a][f.owner], f.tangent_left)) +
                              f.left * (halo[f.neighbor][a] + dot(gradients[a][f.neighbor], f.tangent_right))) /
                          f.distance;
    }
    return faces(result);
}
std::vector<double> GeneralVoFFlowWorkspace::divergence(const std::vector<double>& velocity) const
{
    std::vector<double> result(volumes_.size());
    const auto lease = mesh_->acquire_execution_view();
    for (std::size_t c = 0; c < volumes_.size(); ++c)
    {
        CompensatedSum<> value;
        for (auto fid : mesh_->faces(LO(c)))
            value += (geometry_[fid].owner == LO(c) ? 1 : -1) * geometry_[fid].area * velocity[fid];
        result[c] = static_cast<double>(value.value() / volumes_[c]);
    }
    return result;
}
std::vector<Vector> GeneralVoFFlowWorkspace::viscous_force(
    const std::vector<Vector>& velocity, const std::vector<double>& viscosity) const
{
    const auto u = vectors(velocity);
    const auto mu = cells(viscosity);
    std::array<std::vector<Vector>, 3> derivatives;
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<double> scalar(velocity.size());
        for (std::size_t c = 0; c < velocity.size(); ++c)
            scalar[c] = velocity[c][a];
        derivatives[a] = vectors(gradient(scalar));
    }
    std::array<std::vector<double>, 3> traction;
    for (auto& t : traction)
        t.resize(geometry_.size());
    bool valid = true;
    for (double value : viscosity)
        valid = valid && std::isfinite(value) && value >= 0;
    require(valid, "General VoF viscosity must be finite and nonnegative.");
    for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
    {
        const auto& f = geometry_[id];
        if (f.boundary)
        {
            // Free slip removes tangential traction. The normal Newtonian
            // traction remains and is completed by the impermeability reaction.
            double strain = 0;
            for (unsigned a = 0; a < 3; ++a)
                for (unsigned b = 0; b < 3; ++b)
                    strain += f.normal[a] * derivatives[a][f.owner][b] * f.normal[b];
            for (unsigned a = 0; a < 3; ++a)
                traction[a][id] = 2 * mu[f.owner] * f.area * strain * f.normal[a];
            continue;
        }
        const double face_mu = (f.right * mu[f.owner] + f.left * mu[f.neighbor]) / f.distance;
        for (unsigned a = 0; a < 3; ++a)
        {
            const double left = u[f.owner][a] + dot(derivatives[a][f.owner], f.tangent_left);
            const double right = u[f.neighbor][a] + dot(derivatives[a][f.neighbor], f.tangent_right);
            double stress = (right - left) / f.distance;
            // Symmetric Newtonian stress: mu*(grad(u)+grad(u)^T)*n.
            for (unsigned b = 0; b < 3; ++b)
                stress += f.normal[b] *
                          (f.right * derivatives[b][f.owner][a] + f.left * derivatives[b][f.neighbor][a]) / f.distance;
            traction[a][id] = face_mu * f.area * stress;
        }
    }
    for (auto& t : traction)
        t = faces(t);
    std::vector<Vector> result(volumes_.size());
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < volumes_.size(); ++c)
            for (auto fid : mesh_->faces(LO(c)))
                for (unsigned a = 0; a < 3; ++a)
                    result[c][a] += (geometry_[fid].owner == LO(c) ? 1 : -1) * traction[a][fid];
    }
    return result;
}
void GeneralVoFFlowWorkspace::project(std::vector<double>& face_velocity, std::vector<Vector>& velocity,
    std::vector<double>& pressure, const std::vector<double>& density, const std::vector<double>& fraction,
    const std::vector<double>& kappa, double dt, double sigma, const VoFSolverOptions& options,
    VoFStepDiagnostics& diagnostics, GeneralVoFProjectionProvenance* provenance,
    std::optional<std::span<const Meshes::ExactGeometryMeasure>> physical_mass,
    const GeneralVoFProjectionClosureTopology* closure_topology)
{
    int closure_requested = bool(closure_topology), closure_minimum = 0, closure_maximum = 0;
    Teuchos::reduceAll(
        *mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &closure_requested, &closure_minimum);
    Teuchos::reduceAll(
        *mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &closure_requested, &closure_maximum);
    require(closure_minimum == closure_maximum && (!closure_topology || provenance),
        "General VoF rate-closure request differs across ranks or lacks provenance output.");
    int mass_requested = bool(physical_mass), mass_minimum = 0, mass_maximum = 0;
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &mass_requested, &mass_minimum);
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &mass_requested, &mass_maximum);
    require(mass_minimum == mass_maximum && (!physical_mass || physical_mass->size() == volumes_.size()),
        "General VoF physical mixture-mass request differs across ranks or has wrong shape.");
    int arithmetic_requested = bool(provenance), arithmetic_minimum = 0, arithmetic_maximum = 0;
    Teuchos::reduceAll(
        *mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &arithmetic_requested, &arithmetic_minimum);
    Teuchos::reduceAll(
        *mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &arithmetic_requested, &arithmetic_maximum);
    require(arithmetic_minimum == arithmetic_maximum && (!provenance || provenance->byte_budget > 0),
        "General VoF projection arithmetic request differs across ranks or has no budget.");
    bool valid = velocity.size() == volumes_.size() && fraction.size() == volumes_.size() &&
                 kappa.size() == volumes_.size() && density.size() == volumes_.size() &&
                 face_velocity.size() == geometry_.size() && std::isfinite(dt) && dt > 0 && std::isfinite(sigma) &&
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
    require(valid, "General VoF projection has invalid physical dimensions or density.");
    for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
        valid = valid && std::isfinite(face_velocity[id]) && (!geometry_[id].boundary || face_velocity[id] == 0);
    require(valid, "General VoF projection requires finite face velocities and impermeable walls.");
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
        require(accepted, "General VoF projection arithmetic exceeded its range or budget.");
    };
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(provenance ? provenance->byte_budget : 64 * 1024 * 1024);
    std::size_t scalar_budget = 0, admission_budget = 0;
    arithmetic_guard(
        [&]
        {
            const auto n = volumes_.size(), nf = geometry_.size(),
                       owned_faces = provenance ? mesh_->num_owned_faces() : 0, nc = mesh_->num_cells();
            storage.capacity(Domain::Input, geometry_.capacity(), sizeof(Face));
            storage.capacity(Domain::Input, volumes_.capacity(), sizeof(double));
            storage.capacity(Domain::Input, spans_.capacity(), sizeof(Vector));
            storage.capacity(Domain::Input, exact_volumes_.capacity(), sizeof(Meshes::ExactGeometryMeasure));
            for (const auto& value : exact_volumes_)
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
                for (std::size_t c = 0; c < volumes_.size(); ++c)
                {
                    const auto& mass = (*physical_mass)[c];
                    if (mass.sign() <= 0 ||
                        vof_exact::leased_divide(mass, exact_volumes_[c], scalar_budget).certified().as_double() !=
                            density[c])
                        throw std::runtime_error(
                            "Exact physical mixture mass disagrees with its represented density provenance.");
                }
        });
    GeneralVoFProjectionArithmetic arithmetic(scalar_budget);
    std::vector<GeneralVoFRoundedValue> rounded(provenance ? mesh_->num_owned_faces() : 0);
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
        (void) close_general_vof_phi(mesh_, closure_topology->cells, closure_topology->faces, admission_budget);
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
        retain(exact_volumes_);
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
            current_certificate = certify_general_vof_projection_rate(mesh_, rounded, admission_budget,
                primary_rate.empty() && mesh_->num_owned_faces() ? nullptr : &primary_rate,
                primary_radius.empty() && mesh_->num_owned_faces() ? nullptr : &primary_radius);
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
            const auto comm = mesh_->owned_cell_map()->getComm();
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
                            out << std::setprecision(18) << "cell=" << mesh_->cell_global_id(LO(c))
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
                    const auto row = mesh_->owned_cell_map()->getLocalElement(cell.cell);
                    if (row < 0)
                        throw std::runtime_error("Foreign rate-closure cell.");
                    cell.arithmetic_radius = certificate.owned_cell_radius[row];
                }
                for (auto& face : faces)
                {
                    const auto local = mesh_->owned_face_map()->getLocalElement(face.face);
                    if (local < 0)
                        throw std::runtime_error("Foreign rate-closure face.");
                    face.volume = certificate.owned_rate[local];
                    face.arithmetic_radius = certificate.owned_face_radius[local];
                }
            });
        std::optional<GeneralVoFPhiClosure> closed;
        try
        {
            closed = close_general_vof_phi(mesh_, cells, faces, admission_budget);
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
                    accepted_rate[mesh_->owned_face_map()->getLocalElement(faces[f].face)] = closed->volume[f];
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
    const auto rho = cells(density), f = cells(fraction), k = cells(kappa);
    const auto grad_f = vectors(gradient(fraction));
    std::vector<double> beta(geometry_.size()), capillary(geometry_.size()), fraction_face(geometry_.size()),
        curvature_face(geometry_.size());
    arithmetic_guard(
        [&]
        {
            for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
            {
                const auto& g = geometry_[id];
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
    beta = faces(beta);
    fraction_face = faces(fraction_face);
    curvature_face = faces(curvature_face);
    face_velocity = faces(face_velocity);
    std::vector<Vector> force(volumes_.size());
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < volumes_.size(); ++c)
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto& g = geometry_[fid];
                if (g.boundary)
                {
                    // Use exactly the pressure boundary extrapolation for the
                    // capillary normal reaction. For constant kappa, p=sigma*
                    // kappa*f then cancels on the complete control volume.
                    const double value =
                        sigma * kappa[c] * dot(grad_f[c], convert(mesh_->face_center_vector(fid, LO(c)))) * g.area;
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
    pressure.assign(volumes_.size(), 0);
    arithmetic_guard(
        [&]
        {
            exact_pressure.resize(volumes_.size());
            exact_divergence.resize(volumes_.size());
            exact_rhs.resize(volumes_.size());
            exact_force.resize(volumes_.size());
            if (provenance)
            {
                primary_rate.resize(rounded.size());
                primary_radius.resize(rounded.size());
                for (std::size_t f = 0; f < rounded.size(); ++f)
                {
                    primary_rate[f] =
                        vof_exact::leased_scale(M::from_value(rounded[f].value), geometry_[f].area, scalar_budget);
                    primary_radius[f] =
                        vof_exact::leased_scale(arithmetic.radius(rounded[f]), geometry_[f].area, scalar_budget);
                }
                for (std::size_t c = 0; c < volumes_.size(); ++c)
                    for (unsigned a = 0; a < 3; ++a)
                        exact_force[c][a] = M::from_value(force[c][a]);
            }
        });
    auto div = divergence(face_velocity);
    const auto norm = [&](const std::vector<double>& values)
    {
        double local = 0;
        for (double v : values)
            local = std::isfinite(v) ? std::max(local, std::abs(v)) : std::numeric_limits<double>::infinity();
        return maximum(local);
    };
    diagnostics.pressure_iterations = 0;
    bool arithmetic_admitted = rate_admitted();
    const auto primary_divergence_maximum = [&]
    {
        arithmetic_guard(
            [&]
            {
                for (std::size_t c = 0; c < volumes_.size(); ++c)
                {
                    exact_divergence[c] = vof_exact::leased_divide(
                        current_certificate.owned_net_rate[c], M::from_value(volumes_[c]), scalar_budget);
                    div[c] = static_cast<double>(exact_divergence[c].enclosure().value);
                }
            });
        return vof_exact::leased_maximum_absolute(
            exact_divergence, *mesh_->owned_cell_map(), scalar_budget, admission_budget);
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
    require(exact_limit.sign() > 0, "General VoF projection has invalid physical tolerance.");
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
                if (p.size() != mesh_->num_cells())
                    throw std::runtime_error("Exact pressure halo shape mismatch.");
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < volumes_.size(); ++c)
                {
                    // Freeze the same represented geometry-only WLS matrix and
                    // solve its pressure response exactly; no tiny p is narrowed.
                    std::array<std::array<double, 3>, 3> metric{};
                    std::array<std::array<M, 4>, 3> matrix{};
                    for (auto fid : mesh_->faces(LO(c)))
                    {
                        const auto& g = geometry_[fid];
                        if (g.boundary)
                            continue;
                        auto r = convert(mesh_->cell_center_vector(fid, LO(c)));
                        for (unsigned a = 0; a < 3; ++a)
                            r[a] /= spans_[c][a];
                        const double w = 1 / dot(r, r);
                        const auto delta = vof_exact::leased_add(p[mesh_->opposite_cell(fid, LO(c))],
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
                        grad[a] = vof_exact::leased_divide(matrix[a][3], M::from_value(spans_[c][a]), scalar_budget);
                    exact_pressure[c] = vof_exact::leased_add(exact_pressure[c], p[c], scalar_budget);
                    for (auto fid : mesh_->faces(LO(c)))
                    {
                        const auto& g = geometry_[fid];
                        M value;
                        if (g.boundary)
                        {
                            const auto offset = convert(mesh_->face_center_vector(fid, LO(c)));
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
    for (std::size_t f = 0; f < mesh_->num_owned_faces(); ++f)
        if (!geometry_[f].boundary)
            for (unsigned a = 0; a < 3; ++a)
                local_orthogonal =
                    local_orthogonal && geometry_[f].tangent_left[a] == 0 && geometry_[f].tangent_right[a] == 0;
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_orthogonal, &orthogonal);
    bool exact_fallback_used = false;
    if (!physical_admitted || !arithmetic_admitted)
    {
        projection_.refresh(beta, nullptr, dt, gauge_);
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
                for (std::size_t c = 0; c < volumes_.size(); ++c)
                    local_zero_radius_defect =
                        local_zero_radius_defect || (current_certificate.owned_net_rate[c].sign() != 0 &&
                                                        current_certificate.owned_cell_radius[c].sign() == 0);
            Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_zero_radius_defect,
                &zero_radius_defect);
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
                            exact_faces.reserve(mesh_->num_owned_faces());
                            for (std::size_t c = 0; c < volumes_.size(); ++c)
                                exact_rhs[c] =
                                    vof_exact::leased_scale(current_certificate.owned_net_rate[c], -1, scalar_budget);
                            for (std::size_t f = 0; f < mesh_->num_owned_faces(); ++f)
                            {
                                const auto& g = geometry_[f];
                                if (g.boundary)
                                    continue;
                                const auto mobility = vof_exact::leased_divide(
                                    vof_exact::leased_scale(
                                        vof_exact::leased_scale(M::from_value(dt), beta[f], scalar_budget), g.area,
                                        scalar_budget),
                                    M::from_value(g.distance), scalar_budget);
                                if (mobility.sign() <= 0)
                                    throw std::runtime_error("Nonpositive exact orthogonal pressure mobility.");
                                exact_faces.push_back({mesh_->face_global_id(LO(f)), mesh_->cell_global_id(g.owner),
                                    mesh_->cell_global_id(g.neighbor), mobility});
                            }
                        });
                    const auto result = solve_general_vof_exact_pressure(
                        mesh_, exact_faces, exact_rhs, linear.max_iterations, admission_budget);
                    require(result.iterations >= 0 && result.iterations <= linear.max_iterations,
                        "Exact pressure helper exceeded remaining shared iterations.");
                    require(
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
                                const auto f = mesh_->owned_face_map()->getLocalElement(exact_faces[i].face);
                                primary_rate[f] = vof_exact::leased_add(
                                    primary_rate[f], result.face_rate_increment[i], scalar_budget);
                                const auto view = vof_exact::leased_divide(
                                    primary_rate[f], M::from_value(geometry_[f].area), scalar_budget);
                                face_velocity[f] = static_cast<double>(view.enclosure().value);
                                if (!std::isfinite(face_velocity[f]))
                                    throw std::runtime_error("Exact pressure rate view overflows.");
                                rounded[f] = arithmetic.leaf(face_velocity[f]);
                            }
                        });
                }
                face_velocity = faces(face_velocity);
                div = divergence(face_velocity);
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
                        for (std::size_t c = 0; c < volumes_.size(); ++c)
                            exact_rhs[c] = vof_exact::leased_divide(
                                vof_exact::leased_scale(current_certificate.owned_net_rate[c], -1, scalar_budget),
                                M::from_value(dt), scalar_budget);
                    });
                pressure_scale = vof_exact::leased_maximum_absolute(
                    exact_rhs, *mesh_->owned_cell_map(), scalar_budget, admission_budget);
                if (!pressure_scale.sign())
                    pressure_scale = M::from_value(1);
            }
            arithmetic_guard(
                [&]
                {
                    for (std::size_t c = 0; c < volumes_.size(); ++c)
                    {
                        double rhs = -div[c] * volumes_[c] / dt;
                        if (provenance)
                            rhs = static_cast<double>(
                                vof_exact::leased_divide(exact_rhs[c], pressure_scale, scalar_budget)
                                    .enclosure()
                                    .value);
                        projection_.rhs.replaceLocalValue(LO(c), mesh_->cell_global_id(LO(c)) == gauge_ ? 0 : rhs);
                    }
                });
            const auto stats = projection_.solver.solve_from_zero_with_statistics(
                Teuchos::rcp_implicit_cast<const Pack::operator_type>(projection_.matrix), projection_.rhs,
                projection_.solution, linear);
            diagnostics.pressure_iterations += stats.iterations;
            require(std::isfinite(stats.achieved_tolerance) && std::isfinite(stats.rhs_norm),
                "General VoF pressure solve is nonfinite.");
            M rate_pressure_scale;
            arithmetic_guard([&] { rate_pressure_scale = vof_exact::leased_scale(pressure_scale, dt, scalar_budget); });
            const auto data = projection_.solution.getData();
            std::vector<double> p_owned(data.begin(), data.end());
            const auto p = cells(p_owned);
            const auto grad_p = vectors(gradient(p_owned));
            std::vector<double> p_face(geometry_.size());
            arithmetic_guard(
                [&]
                {
                    for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
                    {
                        const auto& g = geometry_[id];
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
            face_velocity = faces(face_velocity);
            p_face = faces(p_face);
            arithmetic_guard(
                [&]
                {
                    const auto lease = mesh_->acquire_execution_view();
                    for (std::size_t c = 0; c < volumes_.size(); ++c)
                    {
                        if (provenance)
                            exact_pressure[c] = vof_exact::leased_add(exact_pressure[c],
                                vof_exact::leased_scale(pressure_scale, p_owned[c], scalar_budget), scalar_budget);
                        else
                            pressure[c] += p_owned[c];
                        for (auto fid : mesh_->faces(LO(c)))
                        {
                            const auto& g = geometry_[fid];
                            // Extrapolate the same physical pressure gradient to
                            // an impermeable face: its reaction completes the
                            // pressure force on a boundary control volume.
                            const double value =
                                g.boundary ? -dot(grad_p[c], convert(mesh_->face_center_vector(fid, LO(c)))) * g.area
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
            div = divergence(face_velocity);
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
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < volumes_.size(); ++c)
                {
                    Meshes::ExactGeometryMeasure rate;
                    for (auto fid : mesh_->faces(LO(c)))
                    {
                        const auto& g = geometry_[fid];
                        const auto contribution = vof_exact::leased_scale(
                            vof_exact::leased_scale(
                                Meshes::ExactGeometryMeasure::from_value(face_velocity[fid]), g.area, scalar_budget),
                            g.owner == LO(c) ? 1 : -1, scalar_budget);
                        rate = vof_exact::leased_add(rate, contribution, scalar_budget);
                    }
                    const auto exact = provenance
                                           ? exact_divergence[c]
                                           : vof_exact::leased_divide(rate,
                                                 Meshes::ExactGeometryMeasure::from_value(volumes_[c]), scalar_budget);
                    local_exact_divergence = std::max(local_exact_divergence, std::abs(exact.enclosure().value));
                    local_erased += exact.sign() != 0 && div[c] == 0;
                }
            });
        const auto comm = mesh_->owned_cell_map()->getComm();
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
            for (std::size_t c = 0; c < volumes_.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto physical_force = provenance ? exact_force[c][a] : M::from_value(force[c][a]);
                    if (!physical_force.sign())
                        continue;
                    // Normalize before narrowing: a represented density and volume
                    // may have a nonzero product below binary64 carrier range.
                    const auto mass =
                        physical_mass ? (*physical_mass)[c]
                                      : vof_exact::leased_scale(Meshes::ExactGeometryMeasure::from_value(volumes_[c]),
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
    require(valid, "General VoF projection produced a nonfinite candidate.");
    diagnostics.maximum_divergence = residual;
    diagnostics.pressure_residual = residual / dt;
    if (provenance)
    {
        provenance->face_velocity = std::move(rounded);
        provenance->face_rate = std::move(accepted_rate);
    }
}
double GeneralVoFFlowWorkspace::stable_timestep(
    const std::vector<double>& velocity, double rho, double mu, double sigma, const VoFSolverOptions& options) const
{
    require(velocity.size() == geometry_.size() && std::isfinite(rho) && rho > 0 && std::isfinite(mu) && mu >= 0 &&
                std::isfinite(sigma) && sigma >= 0 && std::isfinite(options.max_courant) && options.max_courant > 0 &&
                std::isfinite(options.viscous_safety) && options.viscous_safety > 0 &&
                std::isfinite(options.capillary_safety) && options.capillary_safety > 0,
        "General VoF timestep has invalid material/face data.");
    require(std::all_of(velocity.begin(), velocity.end(), [](double value) { return std::isfinite(value); }),
        "General VoF timestep requires finite face velocities.");
    double local_inverse = 0;
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < volumes_.size(); ++c)
        {
            double flux = 0, viscous = 0, area = 0;
            double span = std::min({spans_[c][0], spans_[c][1], spans_[c][2]});
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto& f = geometry_[fid];
                flux += f.area * std::abs(velocity[fid]);
                area += f.area;
                if (!f.boundary)
                    viscous += 2 * mu * f.area / (rho * f.distance * volumes_[c]);
            }
            // Twice the volume/surface ratio never exceeds the smallest
            // thickness of a convex cell, including rotated slender cells.
            span = std::min(span, 2 * volumes_[c] / area);
            local_inverse = std::max(local_inverse, flux / (2 * volumes_[c] * options.max_courant));
            local_inverse = std::max(local_inverse, viscous / options.viscous_safety);
            if (sigma > 0)
                local_inverse =
                    std::max(local_inverse, std::sqrt(sigma / (rho * span * span * span)) / options.capillary_safety);
        }
    }
    const double inverse = maximum(local_inverse);
    return inverse > 0 ? 1 / inverse : std::numeric_limits<double>::infinity();
}
} // namespace SimpleFluid::detail
