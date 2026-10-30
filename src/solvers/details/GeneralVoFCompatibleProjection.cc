/** @file GeneralVoFCompatibleProjection.cc
 * @brief Unregularized compatible physical-mass KKT assembly and wide defects.
 */
#include "solvers/details/GeneralVoFCompatibleProjection.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Map = Pack::map_type;
using WideVector = GeneralVoFCompatibleProjection::WideVector;
long double dot(const WideVector& a, const WideVector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
std::array<double, 3> components(const MeshUtils::Vec3& value)
{
    return {value.x, value.y, value.z};
}
} // namespace

struct GeneralVoFCompatibleProjection::Impl
{
    struct Entry
    {
        LO face;
        int sign;
        WideVector coefficient;
    };
    struct Cell
    {
        long double mass = 0;
        std::vector<Entry> entries;
    };
    struct Face
    {
        LO owner = -1, neighbor = -1;
        WideVector left{}, right{};
    };
    SP<const MeshHandle<>> mesh;
    general_vof::Execution execution;
    std::vector<Cell> cells;
    std::vector<Face> faces;
    std::vector<long double> halo_mass;
    std::vector<unsigned char> boundary;
    Teuchos::RCP<const Map> joint_map;
    Teuchos::RCP<Pack::matrix_type> matrix;
    std::vector<long double> scale;
    Teuchos::RCP<Pack::vector_type> rhs, solution;
    BelosLinearSolver<> solver;
    GO face_offset = 0, gauge = 0;
    std::size_t budget = 0, bytes = 0;
    std::uint64_t epoch = 0;

    explicit Impl(SP<const MeshHandle<>> source) : mesh(std::move(source)), execution(*mesh) {}

    template<class T> void agree(T value, const char* message) const
    {
        T minimum{}, maximum{};
        const auto comm = mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &value, &minimum);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &value, &maximum);
        execution.require(minimum == maximum, message);
    }
    void current() const
    {
        execution.require(epoch == mesh->geometry_epoch(), "Compatible projection geometry is stale.");
    }
    void build(std::span<const long double> mass, std::size_t requested_budget)
    {
        budget = requested_budget;
        execution.require(budget > 0 && mass.size() == mesh->num_owned_cells(),
            "Compatible projection requires owned physical masses and a positive payload budget.");
        // Reject impossible geometry payloads before materializing any cache.
        // Use full incident counts, including fixed boundary faces, as a
        // conservative bound for the subsequently retained interior stencil.
        execution.collective(
            [&]
            {
                const auto lease = mesh->acquire_execution_view();
                long double estimate = static_cast<long double>(mesh->num_owned_cells()) * sizeof(Cell) +
                                       static_cast<long double>(mesh->num_owned_faces()) * sizeof(Face) +
                                       static_cast<long double>(mesh->num_faces()) +
                                       static_cast<long double>(mesh->num_cells()) * sizeof(long double);
                for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                    estimate += static_cast<long double>(mesh->faces(LO(c)).size()) * sizeof(Entry);
                if (estimate > budget)
                    throw std::runtime_error("Compatible projection geometry payload exceeds its byte budget.");
            },
            "compatible_projection_geometry_budget");
        GO local_min = std::numeric_limits<GO>::max(), local_max = -1, local_face_max = -1;
        std::size_t local_max_faces = 0, global_max_faces = 0;
        execution.collective(
            [&]
            {
                const auto lease = mesh->acquire_execution_view();
                cells.resize(mesh->num_owned_cells());
                faces.resize(mesh->num_owned_faces());
                boundary.resize(mesh->num_faces());
                for (std::size_t f = 0; f < boundary.size(); ++f)
                {
                    boundary[f] = mesh->is_geometry_exterior_face(LO(f));
                    const auto gid = mesh->face_global_id(LO(f));
                    if (gid < 0)
                        throw std::runtime_error("Compatible projection requires nonnegative face IDs.");
                    local_face_max = std::max(local_face_max, gid);
                    if (f < faces.size() && !boundary[f])
                    {
                        auto& face = faces[f];
                        face.owner = mesh->owner_cell(LO(f));
                        face.neighbor = mesh->neighbor_cell(LO(f));
                        if (face.owner < 0 || face.neighbor < 0)
                            throw std::runtime_error("Compatible projection lacks an owned-face adjacent cell.");
                        const long double vl = mesh->cell_volume(face.owner), vr = mesh->cell_volume(face.neighbor);
                        const auto left = components(mesh->face_center_vector(LO(f), face.owner));
                        const auto right = components(mesh->face_center_vector(LO(f), face.neighbor));
                        if (!std::isfinite(vl) || vl <= 0 || !std::isfinite(vr) || vr <= 0)
                            throw std::runtime_error("Compatible projection has invalid adjacent native volumes.");
                        for (unsigned a = 0; a < 3; ++a)
                        {
                            face.left[a] = static_cast<long double>(left[a]) / vl;
                            face.right[a] = -static_cast<long double>(right[a]) / vr;
                            if (!std::isfinite(face.left[a]) || !std::isfinite(face.right[a]))
                                throw std::runtime_error("Compatible projection found invalid native face centroids.");
                        }
                    }
                }
                for (std::size_t c = 0; c < cells.size(); ++c)
                {
                    const LO lid = LO(c);
                    const GO gid = mesh->cell_global_id(lid);
                    const long double volume = mesh->cell_volume(lid);
                    if (gid < 0 || !std::isfinite(mass[c]) || mass[c] <= 0 || !std::isfinite(volume) || volume <= 0)
                        throw std::runtime_error("Compatible projection has invalid physical mass or native volume.");
                    local_min = std::min(local_min, gid);
                    local_max = std::max(local_max, gid);
                    auto& cell = cells[c];
                    cell.mass = mass[c];
                    cell.entries.reserve(mesh->faces(lid).size());
                    for (const auto face : mesh->faces(lid))
                    {
                        if (boundary[face])
                            continue;
                        const int sign = mesh->owner_cell(face) == lid ? 1 : -1;
                        if (sign < 0 && mesh->neighbor_cell(face) != lid)
                            throw std::runtime_error("Compatible projection found inconsistent face incidence.");
                        const auto displacement = components(mesh->face_center_vector(face, lid));
                        Entry entry{face, sign, {}};
                        for (unsigned a = 0; a < 3; ++a)
                        {
                            entry.coefficient[a] = sign * static_cast<long double>(displacement[a]) / volume;
                            if (!std::isfinite(entry.coefficient[a]))
                                throw std::runtime_error("Compatible projection found invalid native face centroid.");
                        }
                        cell.entries.push_back(entry);
                    }
                    local_max_faces = std::max(local_max_faces, cell.entries.size());
                }
            },
            "compatible_projection_geometry");
        const auto comm = mesh->owned_cell_map()->getComm();
        GO maximum_cell = -1, maximum_face = -1;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_min, &gauge);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max, &maximum_cell);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_face_max, &maximum_face);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max_faces, &global_max_faces);
        execution.require(maximum_cell >= 0 && maximum_face >= 0 && maximum_cell < std::numeric_limits<GO>::max() &&
                              maximum_face <= std::numeric_limits<GO>::max() - maximum_cell - 1,
            "Compatible projection joint cell/face global-ID range overflows.");
        face_offset = maximum_cell + 1;
        const auto nc = mesh->num_owned_cells(), nf = mesh->num_owned_faces();
        const auto nj = nc + nf;
        // Two incident cell cliques and the two pressure columns bound every
        // interior face row, including duplicates before fillComplete merges.
        const std::size_t row_capacity = 2 * global_max_faces + 2;
        execution.collective(
            [&]
            {
                long double estimate = cells.capacity() * sizeof(Cell) + boundary.capacity() +
                                       faces.capacity() * sizeof(Face) + mesh->num_cells() * sizeof(long double);
                for (const auto& cell : cells)
                    estimate += cell.entries.capacity() * sizeof(Entry);
                estimate +=
                    static_cast<long double>(nj) * row_capacity * (2 * sizeof(double) + 2 * sizeof(GO) + sizeof(LO));
                // Joint maps, row pointers, scales, import/export high+low
                // buffers, wide results/defects, and row assembly scratch.
                estimate += static_cast<long double>(nj + mesh->num_faces() + mesh->num_cells()) * 256;
                estimate += static_cast<long double>(row_capacity) * 64;
                if (estimate > budget || nj > std::size_t(std::numeric_limits<LO>::max()))
                    throw std::runtime_error("Compatible projection explicit payload exceeds its byte budget.");
                bytes = static_cast<std::size_t>(estimate);
            },
            "compatible_projection_budget");
        Teuchos::Array<GO> gids;
        execution.collective(
            [&]
            {
                gids.reserve(nj);
                for (std::size_t c = 0; c < nc; ++c)
                    gids.push_back(mesh->cell_global_id(LO(c)));
                for (std::size_t f = 0; f < nf; ++f)
                    gids.push_back(face_offset + mesh->face_global_id(LO(f)));
            },
            "compatible_projection_joint_ids");
        joint_map = Teuchos::rcp(new Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), gids(), 0, comm));
        halo_mass = import_wide(mass, mesh->owned_cell_map(), mesh->overlap_cell_map());
        std::vector<long double> owned_face_scale, owned_cell_scale;
        execution.collective(
            [&]
            {
                owned_face_scale.resize(nf);
                owned_cell_scale.resize(nc);
                scale.resize(nj);
                for (std::size_t f = 0; f < nf; ++f)
                {
                    long double diagonal = 1;
                    if (!boundary[f])
                    {
                        const auto& face = faces[f];
                        diagonal = halo_mass[face.owner] * dot(face.left, face.left) +
                                   halo_mass[face.neighbor] * dot(face.right, face.right);
                    }
                    if (!std::isfinite(diagonal) || diagonal <= 0)
                        throw std::runtime_error("Compatible projection has an invalid physical face diagonal.");
                    owned_face_scale[f] = 1 / std::sqrt(diagonal);
                    if (!std::isfinite(owned_face_scale[f]) || owned_face_scale[f] <= 0)
                        throw std::runtime_error("Compatible projection has an invalid wide face scale.");
                    scale[nc + f] = owned_face_scale[f];
                }
            },
            "compatible_projection_wide_diagonal");
        const auto face_scale = import_wide(owned_face_scale, mesh->owned_face_map(), mesh->overlap_face_map());
        execution.collective(
            [&]
            {
                for (std::size_t c = 0; c < nc; ++c)
                {
                    long double inverse_diagonal = 0;
                    for (const auto& entry : cells[c].entries)
                        inverse_diagonal += face_scale[entry.face] * face_scale[entry.face];
                    const long double value =
                        mesh->cell_global_id(LO(c)) == gauge ? 1 : 1 / std::sqrt(inverse_diagonal);
                    if (!std::isfinite(value) || value <= 0)
                        throw std::runtime_error("Compatible projection has an invalid wide pressure scale.");
                    owned_cell_scale[c] = value;
                    scale[c] = value;
                }
            },
            "compatible_projection_wide_pressure_scale");
        const auto cell_scale = import_wide(owned_cell_scale, mesh->owned_cell_map(), mesh->overlap_cell_map());
        execution.collective(
            [&]
            {
                matrix = Teuchos::rcp(new Pack::matrix_type(joint_map, row_capacity));
                rhs = Teuchos::rcp(new Pack::vector_type(joint_map));
                solution = Teuchos::rcp(new Pack::vector_type(joint_map));
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t c = 0; c < nc; ++c)
                {
                    const auto cg = mesh->cell_global_id(LO(c));
                    const auto& cell = cells[c];
                    for (const auto& left : cell.entries)
                    {
                        Teuchos::Array<GO> columns;
                        Teuchos::Array<double> values;
                        columns.reserve(cell.entries.size());
                        values.reserve(cell.entries.size());
                        for (const auto& right : cell.entries)
                        {
                            columns.push_back(face_offset + mesh->face_global_id(right.face));
                            const auto value = cell.mass * dot(left.coefficient, right.coefficient) *
                                               (face_scale[left.face] * face_scale[right.face]);
                            values.push_back(static_cast<double>(value));
                            if (!std::isfinite(values.back()))
                                throw std::runtime_error("Compatible projection scaled mass coefficient is nonfinite.");
                        }
                        matrix->insertGlobalValues(face_offset + mesh->face_global_id(left.face), columns(), values());
                    }
                    Teuchos::Array<GO> columns;
                    Teuchos::Array<double> values;
                    if (cg == gauge)
                    {
                        columns.push_back(cg);
                        values.push_back(1);
                    }
                    else
                        for (const auto& entry : cell.entries)
                        {
                            columns.push_back(face_offset + mesh->face_global_id(entry.face));
                            values.push_back(static_cast<double>(-entry.sign * scale[c] * face_scale[entry.face]));
                        }
                    matrix->insertGlobalValues(cg, columns(), values());
                }
                for (std::size_t f = 0; f < nf; ++f)
                {
                    const LO lid = LO(f);
                    const GO fg = face_offset + mesh->face_global_id(lid);
                    Teuchos::Array<GO> columns;
                    Teuchos::Array<double> values;
                    if (boundary[f])
                    {
                        columns.push_back(fg);
                        values.push_back(1);
                    }
                    else
                    {
                        const auto owner = mesh->owner_cell(lid), neighbor = mesh->neighbor_cell(lid);
                        if (owner < 0 || neighbor < 0)
                            throw std::runtime_error("Compatible projection lacks an owned-face adjacent cell.");
                        for (const auto [cell, coefficient] : {std::pair{owner, -1.0}, std::pair{neighbor, 1.0}})
                        {
                            const auto cg = mesh->cell_global_id(cell);
                            if (cg != gauge)
                            {
                                columns.push_back(cg);
                                values.push_back(
                                    static_cast<double>(coefficient * owned_face_scale[f] * cell_scale[cell]));
                            }
                        }
                    }
                    matrix->insertGlobalValues(fg, columns(), values());
                }
            },
            "compatible_projection_assembly");
        matrix->fillComplete();
        epoch = mesh->geometry_epoch();
    }

    std::vector<long double> import_wide(std::span<const long double> owned, const Teuchos::RCP<const Map>& source,
        const Teuchos::RCP<const Map>& target) const
    {
        execution.require(owned.size() == source->getLocalNumElements(), "Compatible projection halo shape mismatch.");
        Teuchos::RCP<Pack::multi_vector_type> input, output;
        execution.collective(
            [&]
            {
                input = Teuchos::rcp(new Pack::multi_vector_type(source, 3));
                output = Teuchos::rcp(new Pack::multi_vector_type(target, 3));
                auto high = input->getDataNonConst(0), low = input->getDataNonConst(1),
                     exponent = input->getDataNonConst(2);
                for (std::size_t i = 0; i < owned.size(); ++i)
                {
                    if (!std::isfinite(owned[i]))
                        throw std::runtime_error("Compatible projection halo value is nonfinite.");
                    int power = 0;
                    const long double mantissa = std::frexp(owned[i], &power);
                    high[i] = static_cast<double>(mantissa);
                    low[i] = static_cast<double>(mantissa - high[i]);
                    exponent[i] = power;
                }
            },
            "compatible_projection_halo_pack");
        Pack::import_type importer(source, target);
        output->doImport(*input, importer, Tpetra::INSERT);
        std::vector<long double> result;
        execution.collective(
            [&]
            {
                result.resize(target->getLocalNumElements());
                auto high = output->getData(0), low = output->getData(1), exponent = output->getData(2);
                for (std::size_t i = 0; i < result.size(); ++i)
                    result[i] = std::ldexp(static_cast<long double>(high[i]) + low[i], static_cast<int>(exponent[i]));
            },
            "compatible_projection_halo_unpack");
        return result;
    }

    std::vector<long double> transpose(std::span<const WideVector> velocity) const
    {
        std::vector<long double> result, component;
        std::vector<WideVector> halo_velocity;
        execution.collective(
            [&]
            {
                result.resize(mesh->num_owned_faces());
                component.resize(cells.size());
                halo_velocity.resize(mesh->num_cells());
            },
            "compatible_projection_transpose_pack");
        for (unsigned a = 0; a < 3; ++a)
        {
            for (std::size_t c = 0; c < cells.size(); ++c)
                component[c] = velocity[c][a];
            const auto imported = import_wide(component, mesh->owned_cell_map(), mesh->overlap_cell_map());
            for (std::size_t c = 0; c < imported.size(); ++c)
                halo_velocity[c][a] = imported[c];
        }
        execution.collective(
            [&]
            {
                for (std::size_t f = 0; f < result.size(); ++f)
                    if (!boundary[f])
                    {
                        const auto& face = faces[f];
                        result[f] = halo_mass[face.owner] * dot(face.left, halo_velocity[face.owner]) +
                                    halo_mass[face.neighbor] * dot(face.right, halo_velocity[face.neighbor]);
                    }
            },
            "compatible_projection_transpose_unpack");
        return result;
    }

    std::vector<WideVector> reconstruct(std::span<const long double> face_rate) const
    {
        current();
        const auto q = import_wide(face_rate, mesh->owned_face_map(), mesh->overlap_face_map());
        std::vector<WideVector> result;
        execution.collective(
            [&]
            {
                result.resize(cells.size());
                for (std::size_t c = 0; c < cells.size(); ++c)
                    for (const auto& entry : cells[c].entries)
                        for (unsigned a = 0; a < 3; ++a)
                            result[c][a] += entry.coefficient[a] * q[entry.face];
                for (std::size_t f = 0; f < face_rate.size(); ++f)
                    if (boundary[f] && face_rate[f] != 0)
                        throw std::runtime_error("Compatible projection requires zero exterior face rates.");
            },
            "compatible_projection_reconstruction");
        return result;
    }

    void physical_inputs(std::span<const Vector> predictor, std::span<const long double> target,
        std::span<const long double> force, double dt) const
    {
        current();
        execution.require(predictor.size() == cells.size() && target.size() == cells.size() &&
                              force.size() == mesh->num_owned_faces() && std::isfinite(dt) && dt > 0,
            "Compatible projection physical input shape or timestep is invalid.");
        agree(dt, "Compatible projection timesteps differ across ranks.");
        execution.collective(
            [&]
            {
                for (const auto& value : predictor)
                    for (const auto component : value)
                        if (!std::isfinite(component))
                            throw std::runtime_error("Compatible projection predictor is nonfinite.");
                for (auto value : target)
                    if (!std::isfinite(value))
                        throw std::runtime_error("Compatible projection cell target is nonfinite.");
                for (std::size_t f = 0; f < force.size(); ++f)
                    if (!std::isfinite(force[f]) || (boundary[f] && force[f] != 0))
                        throw std::runtime_error("Compatible projection exterior force must be zero and finite.");
            },
            "compatible_projection_physical_inputs");
    }
};

GeneralVoFCompatibleProjection::GeneralVoFCompatibleProjection(
    SP<const MeshHandle<>> mesh, std::span<const long double> mass, std::size_t budget)
    : impl_(std::make_unique<Impl>(std::move(mesh)))
{
    impl_->build(mass, budget);
}
GeneralVoFCompatibleProjection::~GeneralVoFCompatibleProjection() = default;
std::size_t GeneralVoFCompatibleProjection::storage_bytes() const noexcept
{
    return impl_->bytes;
}
std::vector<GeneralVoFCompatibleProjection::WideVector> GeneralVoFCompatibleProjection::reconstruct(
    std::span<const long double> q) const
{
    return impl_->reconstruct(q);
}

GeneralVoFCompatibleProjection::Result GeneralVoFCompatibleProjection::solve_algebraic(
    std::span<const long double> face_rhs, std::span<const long double> cell_rhs, const LinearSolverOptions& requested)
{
    auto& p = *impl_;
    p.current();
    p.execution.require(face_rhs.size() == p.mesh->num_owned_faces() && cell_rhs.size() == p.cells.size(),
        "Compatible projection algebraic RHS shape mismatch.");
    p.execution.collective([&] { BelosLinearSolver<>::validate_options(requested); }, "compatible_projection_options");
    p.agree(requested.max_iterations, "Compatible projection iteration limits differ across ranks.");
    p.agree(requested.tolerance, "Compatible projection linear tolerances differ across ranks.");
    p.agree(requested.verbosity, "Compatible projection linear verbosity differs across ranks.");
    const auto nc = p.cells.size();
    const auto nj = nc + face_rhs.size();
    // Belos PseudoBlockGMRES's configured default restart space is 300.
    // Include its basis, Hessenberg/QR storage and wrapper defect vectors.
    constexpr std::size_t krylov_blocks = 300;
    const long double solve_bytes = p.bytes + static_cast<long double>(nj) * (krylov_blocks + 16) * sizeof(double) +
                                    (krylov_blocks + 1) * (krylov_blocks + 8) * sizeof(double);
    p.execution.require(solve_bytes <= p.budget, "Compatible projection Krylov work exceeds its byte budget.");
    long double local_norm = 0;
    p.execution.collective(
        [&]
        {
            const auto& d = p.scale;
            for (std::size_t i = 0; i < nj; ++i)
            {
                const auto value = i < nc ? -cell_rhs[i] : face_rhs[i - nc];
                if (!std::isfinite(value) || (i >= nc && p.boundary[i - nc] && value != 0))
                    throw std::runtime_error("Compatible projection algebraic RHS is invalid.");
                const long double scaled = value * d[i];
                if (!std::isfinite(scaled))
                    throw std::runtime_error("Compatible projection scaled RHS is nonfinite.");
                if (i < nc && p.mesh->cell_global_id(LO(i)) == p.gauge)
                    continue;
                local_norm = std::max(local_norm, std::abs(scaled));
            }
        },
        "compatible_projection_rhs_norm");
    // Teuchos's configured communicator serialization does not provide a
    // long-double reduction. Reduce the exponent and normalized mantissa,
    // retaining the wide dynamic range before narrowing the normalized RHS.
    int local_exponent = std::numeric_limits<int>::min(), global_exponent = 0;
    double local_mantissa = 0, global_mantissa = 0;
    if (local_norm > 0)
        local_mantissa = static_cast<double>(std::frexp(local_norm, &local_exponent));
    const auto comm = p.mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_exponent, &global_exponent);
    if (local_exponent != global_exponent)
        local_mantissa = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_mantissa, &global_mantissa);
    const long double norm =
        global_mantissa > 0 ? std::ldexp(static_cast<long double>(global_mantissa), global_exponent) : 0;
    p.execution.collective(
        [&]
        {
            auto b = p.rhs->getDataNonConst();
            const auto& d = p.scale;
            for (std::size_t i = 0; i < nj; ++i)
            {
                const auto value = i < nc ? -cell_rhs[i] : face_rhs[i - nc];
                b[i] = norm == 0 ? 0 : static_cast<double>(value * d[i] / norm);
                if (i < nc && p.mesh->cell_global_id(LO(i)) == p.gauge)
                    b[i] = 0;
            }
        },
        "compatible_projection_rhs_normalization");
    auto options = requested;
    options.backend = LinearSolverBackend::Gmres;
    options.preconditioner = LinearPreconditioner::None;
    auto op = Teuchos::rcp_implicit_cast<const Pack::operator_type>(p.matrix);
    const auto statistics = p.solver.solve_from_zero_with_statistics(op, *p.rhs, *p.solution, options);
    Result result;
    result.linear.add(statistics);
    result.storage_bytes = static_cast<std::size_t>(solve_bytes);
    p.execution.collective(
        [&]
        {
            result.face_rate.resize(face_rhs.size());
            result.pressure_impulse.resize(nc);
            const auto x = p.solution->getData();
            const auto& d = p.scale;
            for (std::size_t c = 0; c < nc; ++c)
                result.pressure_impulse[c] = static_cast<long double>(x[c]) * d[c] * norm;
            for (std::size_t f = 0; f < face_rhs.size(); ++f)
                result.face_rate[f] = p.boundary[f] ? 0 : static_cast<long double>(x[nc + f]) * d[nc + f] * norm;
        },
        "compatible_projection_solution");
    result.velocity = p.reconstruct(result.face_rate);
    return result;
}

GeneralVoFCompatibleProjection::Result GeneralVoFCompatibleProjection::solve(std::span<const Vector> predictor,
    std::span<const long double> target, std::span<const long double> force, double dt,
    const LinearSolverOptions& options)
{
    auto& p = *impl_;
    p.physical_inputs(predictor, target, force, dt);
    std::vector<WideVector> wide;
    p.execution.collective(
        [&]
        {
            wide.resize(predictor.size());
            for (std::size_t c = 0; c < wide.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                    wide[c][a] = predictor[c][a];
        },
        "compatible_projection_predictor");
    auto face_rhs = p.transpose(wide);
    p.execution.collective(
        [&]
        {
            for (std::size_t f = 0; f < face_rhs.size(); ++f)
                face_rhs[f] += dt * force[f];
        },
        "compatible_projection_face_force");
    return solve_algebraic(face_rhs, target, options);
}

GeneralVoFCompatibleProjection::Residual GeneralVoFCompatibleProjection::residual(std::span<const long double> rate,
    std::span<const long double> impulse, std::span<const Vector> predictor, std::span<const long double> target,
    std::span<const long double> force, double dt) const
{
    auto& p = *impl_;
    p.physical_inputs(predictor, target, force, dt);
    const auto q = p.import_wide(rate, p.mesh->owned_face_map(), p.mesh->overlap_face_map());
    const auto lambda = p.import_wide(impulse, p.mesh->owned_cell_map(), p.mesh->overlap_cell_map());
    std::vector<WideVector> difference;
    Residual result;
    p.execution.collective(
        [&]
        {
            difference.resize(p.cells.size());
            result.cell_rhs.assign(target.begin(), target.end());
            for (std::size_t c = 0; c < p.cells.size(); ++c)
            {
                for (unsigned a = 0; a < 3; ++a)
                    difference[c][a] = predictor[c][a];
                for (const auto& entry : p.cells[c].entries)
                {
                    result.cell_rhs[c] -= entry.sign * q[entry.face];
                    for (unsigned a = 0; a < 3; ++a)
                        difference[c][a] -= entry.coefficient[a] * q[entry.face];
                }
                result.max_constraint =
                    std::max(result.max_constraint, static_cast<double>(std::abs(result.cell_rhs[c])));
                if (p.mesh->cell_global_id(LO(c)) == p.gauge)
                    result.cell_rhs[c] = 0;
            }
        },
        "compatible_projection_physical_defect");
    result.face_rhs = p.transpose(difference);
    p.execution.collective(
        [&]
        {
            const auto lease = p.mesh->acquire_execution_view();
            for (std::size_t f = 0; f < result.face_rhs.size(); ++f)
            {
                if (p.boundary[f])
                {
                    if (rate[f] != 0)
                        throw std::runtime_error("Compatible projection residual has nonzero exterior rate.");
                    result.face_rhs[f] = 0;
                    continue;
                }
                const auto owner = p.mesh->owner_cell(LO(f)), neighbor = p.mesh->neighbor_cell(LO(f));
                const auto left = p.mesh->cell_global_id(owner) == p.gauge ? 0 : lambda[owner];
                const auto right = p.mesh->cell_global_id(neighbor) == p.gauge ? 0 : lambda[neighbor];
                result.face_rhs[f] += dt * force[f] + left - right;
                result.max_momentum = std::max(result.max_momentum, static_cast<double>(std::abs(result.face_rhs[f])));
            }
        },
        "compatible_projection_momentum_defect");
    result.max_constraint = p.execution.maximum(result.max_constraint);
    result.max_momentum = p.execution.maximum(result.max_momentum);
    return result;
}
} // namespace SimpleFluid::detail
