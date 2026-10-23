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
GeneralVoFFlowWorkspace::GeneralVoFFlowWorkspace(SP<const MeshHandle<>> mesh, bool finite_arithmetic,
    std::vector<Vector> finite_spans)
    : mesh_(validated_native_flow_mesh(std::move(mesh))),
      cell_import_(mesh_->owned_cell_map(), mesh_->overlap_cell_map()),
      face_import_(mesh_->owned_face_map(), mesh_->overlap_face_map()), cell_owned_(mesh_->owned_cell_map()),
      cell_overlap_(mesh_->overlap_cell_map()), face_owned_(mesh_->owned_face_map()),
      face_overlap_(mesh_->overlap_face_map()), projection_(mesh_)
{
    finite_arithmetic_ = finite_arithmetic;
    bool valid = !finite_arithmetic || finite_spans.size() == mesh_->num_owned_cells();
    auto local_gauge = std::numeric_limits<Pack::global_ordinal_type>::max();
    try
    {
        const auto lease = mesh_->acquire_execution_view();
        volumes_.resize(mesh_->num_owned_cells());
        if (!finite_arithmetic_)
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
            if (finite_arithmetic_)
            {
                if (finite_spans.size() != volumes_.size())
                {
                    valid = false;
                    continue;
                }
                spans_[c] = finite_spans[c];
                for (double span : spans_[c])
                    valid = valid && std::isfinite(span) && span > 0;
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
    geometry_epoch_ = mesh_->geometry_epoch();
    native_owner_ = std::get<MeshHandle<>::MultiRegionPtr>(mesh_->variant());
    prepare_gradient_geometry();
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_gauge, &gauge_);
}
std::vector<double> GeneralVoFFlowWorkspace::cells(const std::vector<double>& owned) const
{
    require(current_geometry() && owned.size() == mesh_->num_owned_cells(),
        "General VoF owned scalar shape mismatch or stale workspace geometry/partition.");
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
    require(current_geometry() && overlap.size() == mesh_->num_faces(),
        "General VoF overlap face shape mismatch or stale workspace geometry/partition.");
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
    require(current_geometry() && owned.size() == mesh_->num_owned_cells(),
        "General VoF owned vector shape mismatch or stale workspace geometry/partition.");
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
bool GeneralVoFFlowWorkspace::current_geometry() const noexcept
{
    bool current = false;
    try
    {
        const auto lease = mesh_->acquire_execution_view();
        const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&mesh_->variant());
        current = native && native->get() == native_owner_.get() && mesh_->geometry_epoch() == geometry_epoch_ &&
                  mesh_->owned_cell_map().getRawPtr() == cell_owned_.getMap().getRawPtr() &&
                  mesh_->overlap_cell_map().getRawPtr() == cell_overlap_.getMap().getRawPtr() &&
                  mesh_->owned_face_map().getRawPtr() == face_owned_.getMap().getRawPtr() &&
                  mesh_->overlap_face_map().getRawPtr() == face_overlap_.getMap().getRawPtr();
    }
    catch (...)
    {
        current = false;
    }
    return current;
}
void GeneralVoFFlowWorkspace::prepare_gradient_geometry()
{
    // Cache only geometry and the elimination schedule. Applying that schedule
    // preserves the original RHS arithmetic order for every scalar field.
    std::array<double, 3> low, high, global_low, global_high;
    std::array<int, 3> sampled{}, global_sampled{};
    low.fill(std::numeric_limits<double>::infinity());
    high.fill(-std::numeric_limits<double>::infinity());
    bool finite = true;
    try
    {
        const auto lease = mesh_->acquire_execution_view();
        gradient_stencils_.resize(volumes_.size());
        for (std::size_t c = 0; c < volumes_.size(); ++c)
        {
            const auto x = convert(mesh_->cell_centroid(LO(c)));
            for (unsigned a = 0; a < 3; ++a)
            {
                finite = finite && std::isfinite(x[a]);
                low[a] = std::min(low[a], x[a]);
                high[a] = std::max(high[a], x[a]);
            }
            auto& stencil = gradient_stencils_[c];
            for (auto fid : mesh_->faces(LO(c)))
                if (!geometry_[fid].boundary)
                {
                    auto r = convert(mesh_->cell_center_vector(fid, LO(c)));
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        sampled[a] = sampled[a] || r[a] != 0;
                        r[a] /= spans_[c][a];
                    }
                    stencil.samples.push_back({mesh_->opposite_cell(fid, LO(c)), r, 1 / dot(r, r)});
                }
        }
    }
    catch (...)
    {
        finite = false;
    }
    require(finite, "General VoF gradient samples must be finite.");
    const auto comm = mesh_->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, low.data(), global_low.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, high.data(), global_high.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, sampled.data(), global_sampled.data());
    for (unsigned a = 0; a < 3; ++a)
        absent_gradient_axis_[a] = global_low[a] == global_high[a] && !global_sampled[a];
    for (auto& stencil : gradient_stencils_)
    {
        std::array<std::array<double, 7>, 3> m{};
        for (const auto& sample : stencil.samples)
            for (unsigned a = 0; a < 3; ++a)
                for (unsigned b = 0; b < 3; ++b)
                    m[a][b] += sample.weight * sample.displacement[a] * sample.displacement[b];
        double norm = 0;
        for (unsigned a = 0; a < 3; ++a)
            if (absent_gradient_axis_[a])
            {
                for (unsigned b = 0; b < 4; ++b)
                    m[a][b] = 0;
                for (unsigned b = 0; b < 3; ++b)
                    m[b][a] = 0;
                m[a][a] = 1;
            }
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
            stencil.pivot[a] = pivot;
            if (!(std::abs(m[pivot][a]) > 1e-12))
            {
                stencil.valid = false;
                break;
            }
            std::swap(m[a], m[pivot]);
            const double d = m[a][a];
            stencil.divisor[a] = d;
            for (unsigned b = a; b < 7; ++b)
                m[a][b] /= d;
            for (unsigned row = 0; row < 3; ++row)
                if (row != a)
                {
                    const double factor = m[row][a];
                    stencil.factor[a][row] = factor;
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
        stencil.valid = stencil.valid && std::isfinite(norm * inverse_norm) && norm * inverse_norm <= 1e8;
    }
}
std::vector<Vector> GeneralVoFFlowWorkspace::gradient(const std::vector<double>& owned) const
{
    const auto halo = cells(owned);
    std::vector<Vector> result(owned.size());
    double low = std::numeric_limits<double>::infinity(), high = -std::numeric_limits<double>::infinity();
    bool valid = true;
    for (double value : owned)
    {
        valid = valid && std::isfinite(value);
        low = std::min(low, value);
        high = std::max(high, value);
    }
    require(valid, "General VoF gradient samples must be finite.");
    double global_low = 0, global_high = 0;
    const auto comm = mesh_->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &low, &global_low);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &high, &global_high);
    if (global_low == global_high && std::isfinite(global_low))
        return result;
    for (std::size_t c = 0; c < owned.size(); ++c)
    {
        const auto& stencil = gradient_stencils_[c];
        valid = valid && stencil.valid;
        Vector rhs{};
        for (const auto& sample : stencil.samples)
        {
            const double delta = halo[sample.neighbor] - owned[c];
            for (unsigned a = 0; a < 3; ++a)
                rhs[a] += sample.weight * sample.displacement[a] * delta;
        }
        for (unsigned a = 0; a < 3; ++a)
            if (absent_gradient_axis_[a])
                rhs[a] = 0;
        if (stencil.valid)
            for (unsigned a = 0; a < 3; ++a)
            {
                std::swap(rhs[a], rhs[stencil.pivot[a]]);
                rhs[a] /= stencil.divisor[a];
                for (unsigned row = 0; row < 3; ++row)
                    if (row != a)
                        rhs[row] -= stencil.factor[a][row] * rhs[a];
            }
        for (unsigned a = 0; a < 3; ++a)
        {
            result[c][a] = rhs[a] / spans_[c][a];
            valid = valid && std::isfinite(result[c][a]);
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
    VoFStepDiagnostics& diagnostics, GeneralVoFFiniteProjectionProvenance* provenance,
    std::optional<std::span<const long double>> physical_mass,
    const GeneralVoFFiniteProjectionClosureTopology* closure_topology,
    std::optional<std::span<const long double>> physical_mass_error)
{
    const auto comm = mesh_->owned_cell_map()->getComm();
    const auto n = volumes_.size(), nf = mesh_->num_owned_faces();
    const auto budget = provenance ? provenance->byte_budget : 64 * 1024 * 1024;
    bool valid = Meshes::bounded_geometry_environment() && finite_arithmetic_ && current_geometry() &&
                 velocity.size() == n && density.size() == n && fraction.size() == n && kappa.size() == n &&
                 face_velocity.size() == geometry_.size() && (!physical_mass || physical_mass->size() == n) &&
                 (!physical_mass_error || (physical_mass && physical_mass_error->size() == n)) && std::isfinite(dt) &&
                 dt > 0 && std::isfinite(sigma) && sigma >= 0 && std::numeric_limits<long double>::is_iec559 &&
                 std::numeric_limits<long double>::digits >= std::numeric_limits<double>::digits &&
                 std::numeric_limits<long double>::digits <= 106 && std::fegetround() == FE_TONEAREST;
    valid = valid && std::isfinite(options.pressure_absolute_tolerance) && options.pressure_absolute_tolerance >= 0 &&
            std::isfinite(options.pressure_relative_tolerance) && options.pressure_relative_tolerance >= 0 &&
            std::isfinite(options.divergence_tolerance) && options.divergence_tolerance > 0 &&
            options.max_pressure_iterations > 0;
    for (std::size_t c = 0; c < n; ++c)
        valid = valid && std::isfinite(density[c]) && density[c] > 0 && std::isfinite(fraction[c]) &&
                fraction[c] >= 0 && fraction[c] <= 1 && std::isfinite(kappa[c]) &&
                (!physical_mass || (std::isfinite((*physical_mass)[c]) && (*physical_mass)[c] > 0)) &&
                (!physical_mass_error || (std::isfinite((*physical_mass_error)[c]) && (*physical_mass_error)[c] >= 0 &&
                                             (*physical_mass)[c] > (*physical_mass_error)[c]));
    for (const auto& u : velocity)
        for (double value : u)
            valid = valid && std::isfinite(value);
    for (std::size_t f = 0; f < nf; ++f)
        valid = valid && std::isfinite(face_velocity[f]) && (!geometry_[f].boundary || face_velocity[f] == 0);
    // Fixed-width vector/halo/gradient/force buffers occupy one half; the
    // authenticated sparse closure uses the other half. Backend allocations
    // remain outside this application-payload accounting, as in the exact API.
    const auto cells_count = mesh_->num_cells(), faces_count = mesh_->num_faces();
    const auto payload_units = 256 * (static_cast<long double>(cells_count) + faces_count + n + nf + 1);
    valid = valid && budget > 0 && payload_units <= budget / 2;
    require(valid,
        "Finite General VoF projection has invalid inputs, unsupported FP environment or exhausted payload cap.");
    int local_requests[3] = {bool(physical_mass), bool(closure_topology), bool(physical_mass_error)},
        minimum_requests[3]{}, maximum_requests[3]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, local_requests, minimum_requests);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, local_requests, maximum_requests);
    require(minimum_requests[0] == maximum_requests[0] && minimum_requests[1] == maximum_requests[1] &&
                minimum_requests[2] == maximum_requests[2],
        "Finite projection requests differ across ranks.");
    const auto wide_maximum = [&](long double local)
    {
        int exponent = std::numeric_limits<int>::min(), global_exponent = 0;
        double mantissa = 0, global_mantissa = 0;
        if (local > 0)
        {
            const auto wide = std::frexp(local, &exponent);
            mantissa = static_cast<double>(wide);
            if (static_cast<long double>(mantissa) < wide)
                mantissa = std::nextafter(mantissa, std::numeric_limits<double>::infinity());
        }
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &exponent, &global_exponent);
        if (exponent != global_exponent)
            mantissa = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &mantissa, &global_mantissa);
        return global_mantissa > 0 ? std::ldexp(static_cast<long double>(global_mantissa), global_exponent) : 0.L;
    };
    const auto halo_rates = [&](const std::vector<long double>& owned)
    {
        std::vector<double> high(geometry_.size()), low(geometry_.size()), exponent(geometry_.size());
        for (std::size_t f = 0; f < nf; ++f)
        {
            int e = 0;
            const auto m = std::frexp(owned[f], &e);
            high[f] = static_cast<double>(m);
            low[f] = static_cast<double>(m - high[f]);
            exponent[f] = e;
        }
        high = faces(high);
        low = faces(low);
        exponent = faces(exponent);
        std::vector<long double> result(geometry_.size());
        for (std::size_t f = 0; f < result.size(); ++f)
            result[f] = std::ldexp(static_cast<long double>(high[f]) + low[f], int(exponent[f]));
        return result;
    };
    const auto rho = cells(density), fraction_halo = cells(fraction), curvature_halo = cells(kappa);
    const auto grad_f = vectors(gradient(fraction));
    std::vector<double> beta(geometry_.size()), fraction_face(geometry_.size()), curvature_face(geometry_.size());
    std::vector<long double> rate(nf), radius(nf);
    constexpr auto eps = std::numeric_limits<long double>::epsilon();
    constexpr auto double_eps = std::numeric_limits<double>::epsilon();
    const auto outward = [](long double bound)
    { return bound == 0 ? 0.L : std::nextafter(bound, std::numeric_limits<long double>::infinity()); };
    const auto record_update = [&](std::size_t f, long double increment, long double construction_magnitude)
    {
        const auto old = rate[f];
        rate[f] += increment;
        // gamma_64 bounds the explicitly evaluated binary64 face expression;
        // the wide accumulation has a separate outward operation allowance.
        const long double gamma = 64.L * double_eps / (1 - 64.L * double_eps);
        radius[f] =
            outward(radius[f] + gamma * construction_magnitude + 8 * eps * (std::abs(old) + std::abs(increment)));
        valid = valid && std::isfinite(rate[f]) && std::isfinite(radius[f]) &&
                (increment == 0 || construction_magnitude > 0);
    };
    for (std::size_t f = 0; f < nf; ++f)
    {
        const auto& g = geometry_[f];
        if (g.boundary)
            continue;
        rate[f] = static_cast<long double>(face_velocity[f]) * g.area;
        radius[f] = 4 * eps * std::abs(rate[f]);
        const double mass_length = rho[g.owner] * g.left + rho[g.neighbor] * g.right;
        beta[f] = g.distance / mass_length;
        const double left = fraction_halo[g.owner] + dot(grad_f[g.owner], g.tangent_left);
        const double right = fraction_halo[g.neighbor] + dot(grad_f[g.neighbor], g.tangent_right);
        fraction_face[f] = (rho[g.neighbor] * g.right * left + rho[g.owner] * g.left * right) / mass_length;
        curvature_face[f] = (g.right * curvature_halo[g.owner] + g.left * curvature_halo[g.neighbor]) / g.distance;
        const long double multiplier =
            static_cast<long double>(dt) * beta[f] * sigma * curvature_face[f] * g.area / g.distance;
        const long double capillary = multiplier * (static_cast<long double>(right) - left);
        record_update(f, capillary, std::abs(multiplier) * (std::abs(left) + std::abs(right)));
        valid = valid && std::isfinite(beta[f]) && beta[f] > 0;
    }
    require(valid, "Finite projection face construction exceeds its supported range.");
    beta = faces(beta);
    fraction_face = faces(fraction_face);
    curvature_face = faces(curvature_face);
    std::vector<std::array<long double, 3>> force(n), force_error(n), velocity_error(n);
    const long double force_gamma = 64.L * double_eps / (1 - 64.L * double_eps);
    const auto add_force = [&](std::size_t c, unsigned a, long double contribution, long double expression_magnitude)
    {
        const auto old = force[c][a];
        force[c][a] += contribution;
        force_error[c][a] = outward(force_error[c][a] + force_gamma * expression_magnitude +
                                    8 * eps * (std::abs(old) + std::abs(contribution)));
    };
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto& g = geometry_[fid];
                const long double value =
                    g.boundary ? static_cast<long double>(sigma) * kappa[c] *
                                     dot(grad_f[c], convert(mesh_->face_center_vector(fid, LO(c)))) * g.area
                               : static_cast<long double>(g.owner == LO(c) ? 1 : -1) * g.area * sigma *
                                     curvature_face[fid] * (fraction_face[fid] - fraction[c]);
                long double magnitude = 0;
                if (g.boundary)
                {
                    const auto r = convert(mesh_->face_center_vector(fid, LO(c)));
                    for (unsigned a = 0; a < 3; ++a)
                        magnitude += std::abs(static_cast<long double>(grad_f[c][a]) * r[a]);
                    magnitude *= std::abs(static_cast<long double>(sigma) * kappa[c] * g.area);
                }
                else
                    magnitude = std::abs(static_cast<long double>(g.area) * sigma * curvature_face[fid]) *
                                (std::abs(fraction_face[fid]) + std::abs(fraction[c]));
                for (unsigned a = 0; a < 3; ++a)
                    add_force(c, a, value * g.normal[a], magnitude * std::abs(g.normal[a]));
            }
    }
    std::vector<long double> net(n), cell_radius(n), overlap_rate, overlap_radius;
    const auto refresh_rates = [&]
    {
        overlap_rate = halo_rates(rate);
        overlap_radius = halo_rates(radius);
        std::fill(net.begin(), net.end(), 0);
        std::fill(cell_radius.begin(), cell_radius.end(), 0);
        long double local_divergence = 0;
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
        {
            long double sum = 0, compensation = 0, magnitude = 0, error = 0;
            std::size_t degree = 0;
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto value = (geometry_[fid].owner == LO(c) ? 1 : -1) * overlap_rate[fid];
                const auto next = sum + value;
                compensation += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
                sum = next;
                magnitude += std::abs(value);
                error += overlap_radius[fid];
                ++degree;
            }
            net[c] = sum + compensation;
            cell_radius[c] = outward(error * (1 + 8 * eps * degree) + 16 * eps * magnitude);
            local_divergence = std::max(local_divergence, std::abs(net[c]) / volumes_[c]);
        }
        return wide_maximum(local_divergence);
    };
    auto residual = refresh_rates();
    const long double limit = std::min(static_cast<long double>(options.divergence_tolerance),
        static_cast<long double>(dt) * options.pressure_absolute_tolerance +
            static_cast<long double>(options.pressure_relative_tolerance) * residual);
    require(std::isfinite(limit) && limit > 0, "Finite projection has invalid original physical threshold.");
    pressure.assign(n, 0);
    std::vector<long double> wide_pressure(n);
    diagnostics.pressure_iterations = 0;
    bool closed = false;
    std::size_t closure_storage = 0;
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Cg;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    linear.tolerance = 1e-8;
    linear.verbosity = 0;
    for (unsigned correction = 0; correction <= 64; ++correction)
    {
        if (residual <= limit)
        {
            try
            {
                auto closure =
                    close_general_vof_finite_rate(mesh_, rate, radius, cell_radius, budget / 2, closure_topology);
                for (std::size_t f = 0; f < nf; ++f)
                    radius[f] = outward(radius[f] + std::abs(closure.correction[f]) +
                                        8 * eps * (std::abs(rate[f]) + std::abs(closure.rate[f])));
                rate = std::move(closure.rate);
                closure_storage = closure.admitted_storage_bytes;
                residual = refresh_rates();
                closed = residual <= limit;
            }
            catch (const GeneralVoFPhiInfeasible&)
            {
                closed = false;
            }
            if (closed)
                break;
        }
        linear.max_iterations = options.max_pressure_iterations - diagnostics.pressure_iterations;
        if (correction == 64 || linear.max_iterations <= 0)
            break;
        if (correction == 0)
            projection_.refresh(beta, nullptr, dt, gauge_);
        long double local_scale = 0;
        for (auto value : net)
            local_scale = std::max(local_scale, std::abs(value / dt));
        const auto scale = wide_maximum(local_scale);
        require(
            std::isfinite(scale) && scale > 0, "Finite projection cannot resolve a nonzero original rate residual.");
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto rhs = -net[c] / dt / scale;
            projection_.rhs.replaceLocalValue(LO(c), mesh_->cell_global_id(LO(c)) == gauge_ ? 0 : double(rhs));
        }
        const auto stats = projection_.solver.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(projection_.matrix), projection_.rhs,
            projection_.solution, linear);
        diagnostics.pressure_iterations += stats.iterations;
        require(std::isfinite(stats.achieved_tolerance) && std::isfinite(stats.rhs_norm),
            "Finite General VoF pressure correction is nonfinite.");
        const auto data = projection_.solution.getData();
        std::vector<double> p_owned(data.begin(), data.end());
        const auto p = cells(p_owned);
        const auto grad_p = vectors(gradient(p_owned));
        std::vector<double> p_face(geometry_.size());
        for (std::size_t f = 0; f < nf; ++f)
        {
            const auto& g = geometry_[f];
            if (g.boundary)
                continue;
            const double left = p[g.owner] + dot(grad_p[g.owner], g.tangent_left);
            const double right = p[g.neighbor] + dot(grad_p[g.neighbor], g.tangent_right);
            const long double multiplier = static_cast<long double>(dt) * scale * beta[f] * g.area / g.distance;
            record_update(f, -multiplier * (static_cast<long double>(right) - left),
                std::abs(multiplier) * (std::abs(right) + std::abs(left)));
            p_face[f] = (rho[g.neighbor] * g.right * left + rho[g.owner] * g.left * right) /
                        (rho[g.owner] * g.left + rho[g.neighbor] * g.right);
        }
        require(valid, "Finite pressure correction exceeds its arithmetic range.");
        p_face = faces(p_face);
        {
            const auto lease = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
            {
                wide_pressure[c] += scale * p_owned[c];
                for (auto fid : mesh_->faces(LO(c)))
                {
                    const auto& g = geometry_[fid];
                    const double value = g.boundary
                                             ? -dot(grad_p[c], convert(mesh_->face_center_vector(fid, LO(c)))) * g.area
                                             : -(g.owner == LO(c) ? 1 : -1) * (p_face[fid] - p_owned[c]) * g.area;
                    long double magnitude = 0;
                    if (g.boundary)
                    {
                        const auto r = convert(mesh_->face_center_vector(fid, LO(c)));
                        for (unsigned a = 0; a < 3; ++a)
                            magnitude += std::abs(static_cast<long double>(grad_p[c][a]) * r[a]);
                    }
                    else
                        magnitude = std::abs(p_face[fid]) + std::abs(p_owned[c]);
                    magnitude *= std::abs(scale) * g.area;
                    for (unsigned a = 0; a < 3; ++a)
                        add_force(c, a, scale * value * g.normal[a], magnitude * std::abs(g.normal[a]));
                }
            }
        }
        residual = refresh_rates();
    }
    if (!closed)
    {
        std::ostringstream message;
        message << std::setprecision(18)
                << "Finite General VoF projection failed original physical/arithmetic acceptance: residual=" << residual
                << ", limit=" << limit << ", iterations=" << diagnostics.pressure_iterations;
        throw std::runtime_error(message.str());
    }
    for (std::size_t f = 0; f < nf; ++f)
        face_velocity[f] = geometry_[f].boundary ? 0 : double(rate[f] / geometry_[f].area);
    face_velocity = faces(face_velocity);
    try
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            pressure[c] = double(wide_pressure[c]);
            const long double mass =
                physical_mass ? (*physical_mass)[c] : static_cast<long double>(density[c]) * volumes_[c];
            const long double mass_error =
                physical_mass_error ? (*physical_mass_error)[c] : (physical_mass ? 0 : 4 * eps * std::abs(mass));
            const auto mass_lower = mass - mass_error;
            valid = valid && std::isfinite(mass) && mass_lower > 0 && std::isfinite(pressure[c]);
            for (unsigned a = 0; a < 3; ++a)
            {
                namespace fp = VoF::finite_detail;
                const auto impulse =
                    fp::divide(fp::multiply({force[c][a], force_error[c][a]}, {dt, 0}), {mass, mass_error});
                const auto update = fp::add({velocity[c][a], 0}, impulse);
                velocity[c][a] = double(update.value);
                velocity_error[c][a] = fp::bound_add(update.absolute_error, std::abs(update.value - velocity[c][a]));
                valid = valid && std::isfinite(velocity[c][a]) && std::isfinite(velocity_error[c][a]) &&
                        (update.value == 0 || velocity[c][a] != 0);
            }
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(valid, "Finite projection candidate cannot be represented without range loss.");
    diagnostics.maximum_divergence = double(residual);
    diagnostics.pressure_residual = double(residual / dt);
    if (provenance)
    {
        provenance->face_rate = std::move(rate);
        provenance->face_radius = std::move(radius);
        provenance->cell_radius = std::move(cell_radius);
        provenance->velocity_error = std::move(velocity_error);
        provenance->closure_storage_bytes = closure_storage;
    }
}
double GeneralVoFFlowWorkspace::stable_timestep(
    const std::vector<double>& velocity, double rho, double mu, double sigma, const VoFSolverOptions& options) const
{
    require(current_geometry() && velocity.size() == geometry_.size() && std::isfinite(rho) && rho > 0 && std::isfinite(mu) && mu >= 0 &&
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
#include "solvers/details/GeneralVoFExactProjection.inc"
} // namespace SimpleFluid::detail
