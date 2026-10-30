#include "solvers/details/GeneralVoFFlowSupport.hh"
/** @file GeneralVoFFlowWorkspace.cc
 * @brief Conditioned native gradients, stress transport and nonorthogonal projection.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "FVM/details/GeneralVoF/OccupancyAdmission.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FlowTrace.hh"
#include "solvers/details/GeneralVoFExactPressure.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFFiniteProjectionArithmetic.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "solvers/details/GeneralVoFCompatibleProjection.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using namespace flow_support;
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
GeneralVoFFlowWorkspace::GeneralVoFFlowWorkspace(
    SP<const MeshHandle<>> mesh, ArithmeticMode arithmetic_mode, std::vector<Vector> finite_spans)
    : mesh_(validated_native_flow_mesh(std::move(mesh))),
      cell_import_(mesh_->owned_cell_map(), mesh_->overlap_cell_map()),
      face_import_(mesh_->owned_face_map(), mesh_->overlap_face_map()), cell_owned_(mesh_->owned_cell_map()),
      cell_overlap_(mesh_->overlap_cell_map()), face_owned_(mesh_->owned_face_map()),
      face_overlap_(mesh_->overlap_face_map()), projection_(mesh_)
{
    arithmetic_mode_ = arithmetic_mode;
    const auto comm = mesh_->owned_cell_map()->getComm();
    int local_mode = int(arithmetic_mode), minimum_mode = 0, maximum_mode = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_mode, &minimum_mode);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_mode, &maximum_mode);
    require(minimum_mode == maximum_mode && minimum_mode >= int(ArithmeticMode::Ordinary) &&
                maximum_mode <= int(ArithmeticMode::Exact),
        "General VoF workspace arithmetic modes are invalid or differ across ranks.");
    bool valid = finite_spans.empty() || finite_spans.size() == mesh_->num_owned_cells();
    auto local_gauge = std::numeric_limits<Pack::global_ordinal_type>::max();
    try
    {
        const auto lease = mesh_->acquire_execution_view();
        volumes_.resize(mesh_->num_owned_cells());
        if (arithmetic_mode_ == ArithmeticMode::Exact)
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
            if (arithmetic_mode_ != ArithmeticMode::Exact)
            {
                if (finite_spans.empty())
                {
                    Vector lower{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()},
                        upper{-lower[0], -lower[1], -lower[2]};
                    for (auto node : (*native)->cell_nodes(mesh_->cell_geometry_global_id(cell)))
                    {
                        const auto point = convert((*native)->node_coordinates(node));
                        for (unsigned axis = 0; axis < 3; ++axis)
                        {
                            lower[axis] = std::min(lower[axis], double(point[axis]));
                            upper[axis] = std::max(upper[axis], double(point[axis]));
                        }
                    }
                    for (unsigned axis = 0; axis < 3; ++axis)
                        spans_[c][axis] = upper[axis] - lower[axis];
                }
                else
                    spans_[c] = finite_spans[c];
                for (double span : spans_[c])
                    valid = valid && std::isfinite(span) && span > 0;
                continue;
            }
            const auto exact_cell =
                SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::from_native_cell(
                    **native, mesh_->cell_geometry_global_id(cell));
            exact_volumes_[c] =
                SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(*exact_cell);
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
std::vector<unsigned char> GeneralVoFFlowWorkspace::capillary_extension_required(
    const std::vector<double>& fraction, std::size_t byte_budget) const
{
    const general_vof::Execution execution(*mesh_);
    const auto owned_count = mesh_->num_owned_cells(), overlap_count = mesh_->num_cells();
    std::vector<double> fraction_halo, flat_owned, flat_halo;
    std::vector<unsigned char> required;
    execution.collective(
        [&]
        {
            if (arithmetic_mode_ != ArithmeticMode::Ordinary || !current_geometry() || fraction.size() != owned_count ||
                geometry_.size() != mesh_->num_faces() || !byte_budget)
                throw std::runtime_error(
                    "Capillary dependency query requires current Ordinary geometry and owned fractions.");
            for (double value : fraction)
                if (!std::isfinite(value) || value < 0 || value > 1)
                    throw std::runtime_error("Capillary dependency query fractions must be finite and in [0,1].");
            auto remaining = byte_budget;
            const auto charge = [&](std::size_t count, std::size_t bytes)
            {
                if (count > remaining / bytes)
                    throw std::length_error("Capillary dependency query exceeds its byte budget.");
                remaining -= count * bytes;
            };
            // Two overlap imports, one owned binary marker and the returned mask.
            // Preflight all counts before allocating or starting either import.
            charge(overlap_count, 2 * sizeof(double));
            charge(owned_count, sizeof(double) + sizeof(unsigned char));
            fraction_halo.assign(overlap_count, 0);
            flat_halo.assign(overlap_count, 0);
            flat_owned.assign(owned_count, 0);
            required.assign(owned_count, 1);
        },
        "capillary_dependency_preflight");
    // Use the same cached scalar import as cells(), with explicit local
    // preparation/publication boundaries on either side of the MPI operation.
    // A host-view allocation failure must be reduced before any rank imports.
    const auto import_scalar = [&](const std::vector<double>& owned, std::vector<double>& overlap)
    {
        execution.collective(
            [&]
            {
                if (!current_geometry())
                    throw std::runtime_error("Capillary dependency geometry changed before scalar import.");
                auto data = cell_owned_.getDataNonConst();
                if (static_cast<std::size_t>(data.size()) != owned.size())
                    throw std::runtime_error("Capillary dependency source differs from its cached scalar map.");
                std::copy(owned.begin(), owned.end(), data.begin());
            },
            "capillary_dependency_import_prepare");
        cell_overlap_.doImport(cell_owned_, cell_import_, Tpetra::INSERT);
        execution.collective(
            [&]
            {
                const auto data = cell_overlap_.getData();
                if (static_cast<std::size_t>(data.size()) != overlap.size())
                    throw std::runtime_error("Capillary dependency target differs from its cached scalar map.");
                std::copy(data.begin(), data.end(), overlap.begin());
            },
            "capillary_dependency_import_publish");
    };
    import_scalar(fraction, fraction_halo);
    execution.collective(
        [&]
        {
            if (fraction_halo.size() != overlap_count)
                throw std::runtime_error("Capillary dependency query imported an invalid scalar shape.");
            const auto lease = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                // Equality is structural. In particular, no phase-fraction cutoff
                // can turn a positive physical trace into an inactive pure cell.
                bool flat = fraction[c] == 0 || fraction[c] == 1;
                for (const auto face : mesh_->faces(LO(c)))
                {
                    const auto& geometry = geometry_.at(face);
                    if (geometry.boundary)
                        continue;
                    if (geometry.owner != LO(c) && geometry.neighbor != LO(c))
                        throw std::runtime_error("Capillary dependency face has inconsistent cached cell incidence.");
                    const auto neighbor = geometry.owner == LO(c) ? geometry.neighbor : geometry.owner;
                    if (neighbor < 0 || std::size_t(neighbor) >= fraction_halo.size())
                        throw std::runtime_error("Capillary dependency neighbor is outside the imported scalar map.");
                    flat = flat && fraction_halo[neighbor] == fraction[c];
                }
                flat_owned[c] = flat ? 1 : 0;
            }
        },
        "capillary_dependency_flat_stencil");
    import_scalar(flat_owned, flat_halo);
    execution.collective(
        [&]
        {
            if (flat_halo.size() != overlap_count)
                throw std::runtime_error("Capillary dependency query imported an invalid marker shape.");
            const auto lease = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                if (flat_owned[c] == 0)
                    continue;
                required[c] = 0;
                for (const auto face : mesh_->faces(LO(c)))
                {
                    const auto& geometry = geometry_.at(face);
                    if (geometry.boundary)
                        continue;
                    const auto neighbor = geometry.owner == LO(c) ? geometry.neighbor : geometry.owner;
                    if (flat_halo.at(neighbor) == 1)
                        continue;
                    // This is the NEIGHBOR'S reconstruction displacement used
                    // by project_finite, not the current cell's displacement.
                    const auto& tangent = geometry.owner == LO(c) ? geometry.tangent_right : geometry.tangent_left;
                    if (std::any_of(tangent.begin(), tangent.end(), [](double value) { return value != 0; }))
                    {
                        required[c] = 1;
                        break;
                    }
                }
            }
            if (!current_geometry())
                throw std::runtime_error("Capillary dependency query geometry changed during the operation.");
        },
        "capillary_dependency_operator_support");
    return required;
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
        face_gradient_stencils_.resize(volumes_.size());
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
            {
                face_gradient_stencils_[c].has_exterior = face_gradient_stencils_[c].has_exterior ||
                    geometry_[fid].boundary;
                if (!geometry_[fid].boundary)
                {
                    auto r = convert(mesh_->cell_center_vector(fid, LO(c)));
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        sampled[a] = sampled[a] || r[a] != 0;
                        r[a] /= spans_[c][a];
                    }
                    stencil.samples.push_back({mesh_->opposite_cell(fid, LO(c)), r, 1 / dot(r, r)});
                    auto face_r = convert(mesh_->face_center_vector(fid, LO(c)));
                    for (unsigned a = 0; a < 3; ++a)
                        face_r[a] /= spans_[c][a];
                    finite = finite && std::isfinite(dot(face_r, face_r)) && dot(face_r, face_r) > 0;
                    face_gradient_stencils_[c].samples.push_back({fid, face_r, 1 / dot(face_r, face_r)});
                }
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
    // An unsampled coordinate direction can still carry a pressure gradient
    // between its two impermeable caps. Qualify a straight closed prism: cap
    // normals are axial, while all other face normals have zero axial component.
    // This is a geometry support contract, not a tolerance-based normal snap.
    std::array<int, 3> local_wall_axis{1, 1, 1}, global_wall_axis{};
    bool wall_geometry_valid = true;
    try
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < volumes_.size(); ++c)
            for (unsigned a = 0; a < 3; ++a)
            {
                if (!absent_gradient_axis_[a])
                    continue;
                bool lower = false, upper = false;
                for (const auto fid : mesh_->faces(LO(c)))
                {
                    const auto& face = geometry_[fid];
                    if (face.normal[a] == 0)
                        continue;
                    bool cap = face.boundary && face.owner == LO(c);
                    for (unsigned b = 0; b < 3; ++b)
                        if (b != a)
                            cap = cap && face.normal[b] == 0;
                    local_wall_axis[a] = local_wall_axis[a] && cap;
                    lower = lower || (cap && face.normal[a] < 0);
                    upper = upper || (cap && face.normal[a] > 0);
                }
                local_wall_axis[a] = local_wall_axis[a] && lower && upper;
            }
    }
    catch (...)
    {
        wall_geometry_valid = false;
    }
    require(wall_geometry_valid, "General VoF wall-pressure geometry preparation failed.");
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, local_wall_axis.data(), global_wall_axis.data());
    for (unsigned a = 0; a < 3; ++a)
        wall_reaction_axis_[a] = absent_gradient_axis_[a] && global_wall_axis[a];
    for (auto& stencil : gradient_stencils_)
        factor_gradient_stencil(stencil);
    for (auto& stencil : face_gradient_stencils_)
        factor_gradient_stencil(stencil);
}
std::size_t GeneralVoFFlowWorkspace::boundary_gradient_storage_bytes() const
{
    std::size_t result = face_gradient_stencils_.capacity() * sizeof(GradientStencil);
    for (const auto& stencil : face_gradient_stencils_)
        result += stencil.samples.capacity() * sizeof(GradientSample);
    return result;
}
void GeneralVoFFlowWorkspace::factor_gradient_stencil(GradientStencil& stencil) const
{
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
#include "solvers/details/GeneralVoFFiniteProjection.tcc"
#include "solvers/details/GeneralVoFCompatibleFlow.tcc"
void GeneralVoFFlowWorkspace::project(std::vector<double>& face_velocity, std::vector<Vector>& velocity,
    std::vector<double>& pressure, const std::vector<double>& density, const std::vector<double>& fraction,
    const std::vector<double>& kappa, double dt, double sigma, const VoFSolverOptions& options,
    VoFStepDiagnostics& diagnostics, GeneralVoFFiniteProjectionProvenance* provenance,
    std::optional<std::span<const long double>> physical_mass,
    const GeneralVoFFiniteProjectionClosureTopology* closure_topology,
    std::optional<std::span<const long double>> physical_mass_error)
{
    if (arithmetic_mode_ == ArithmeticMode::Ordinary)
        return project_compatible(face_velocity, velocity, pressure, density, fraction, kappa, dt,
            sigma, options, diagnostics, provenance, physical_mass, closure_topology, physical_mass_error);
    return project_finite<ArithmeticMode::BoundedCertification>(face_velocity, velocity, pressure, density, fraction,
        kappa, dt, sigma, options, diagnostics, provenance, physical_mass, closure_topology, physical_mass_error);
}
double GeneralVoFFlowWorkspace::stable_timestep(
    const std::vector<double>& velocity, double rho, double mu, double sigma, const VoFSolverOptions& options) const
{
    require(current_geometry() && velocity.size() == geometry_.size() && std::isfinite(rho) && rho > 0 &&
                std::isfinite(mu) && mu >= 0 && std::isfinite(sigma) && sigma >= 0 &&
                std::isfinite(options.max_courant) && options.max_courant > 0 &&
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
            general_vof::flow_trace(mesh_->owned_cell_map()->getComm()->getRank(), "physical_timestep", [&](auto& out)
            {
                out << ",\"cell\":" << mesh_->cell_global_id(LO(c))
                    << ",\"face_cfl_inverse\":" << flux / (2 * volumes_[c] * options.max_courant)
                    << ",\"viscous_inverse\":" << viscous / options.viscous_safety
                    << ",\"capillary_inverse\":"
                    << (sigma > 0 ? std::sqrt(sigma / (rho * span * span * span)) / options.capillary_safety : 0);
            });
        }
    }
    const double inverse = maximum(local_inverse);
    return inverse > 0 ? 1 / inverse : std::numeric_limits<double>::infinity();
}
} // namespace SimpleFluid::detail
