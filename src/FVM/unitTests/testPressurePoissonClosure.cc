/**
 * @file testPressurePoissonClosure.cc
 * @brief Verification-contract gates for geometric roundoff isolation of the
 *        multi-region pressure Poisson operator.
 *
 * Covers, on multi-region meshes with skewed extruded regions and a
 * nonconforming (coarse/fine) Cartesian interface under pure-Neumann
 * pressure boundaries:
 *   1. Discrete GCL closure of the stored face area vectors.
 *   2. Algebraic conservation: exact row-sum and column-sum locks.
 *   3. Zeroth-order constant-pressure reproduction of the assembled matrix.
 *   4. First-order linear-field reproduction without coordinate-origin
 *      leakage (domain shifted by +1e5 in x/y/z).
 *   5. Cold-start near-zero-RHS behavior of the pressure projection.
 *   6. Full corrected operator (A - C) conservation and constant
 *      preservation through the production gradient machinery.
 *   7. Small-time-step scaling safety of the absolute short-circuit guards.
 *   8. High-aspect-ratio LSQ gradient conditioning (normal equations).
 *   9. Multi-rank closure gates across partition boundaries.
 */
#include <gtest/gtest.h>

#include "FVM/MatrixOperators.hh"
#include "FVM/details/OperatorDetails.hh"
#include "FVM/CellOperators.hh"
#include "FVM/details/FieldStoredFaceFlux.hh"
#include "equations/PressureProjectionEquation.hh"
#include "fields/FieldStored.hh"
#include "fields/VectorCellField.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "parallel/CompositePartition.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/CompensatedSum.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace SimpleFluid::detail
{

/** @brief Test-only access to the pressure projection's post-guard RHS. */
template<TpetraTypePack Pack>
struct PressureProjectionEquationTestAccess
{
    static Teuchos::RCP<const typename Pack::vector_type> projection_base_rhs(
        const PressureProjectionEquation<Pack>& equation)
    {
        return equation.d_non_orthogonal_base_rhs;
    }
};

} // namespace SimpleFluid::detail

namespace
{

using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using MultiMesh = Meshes::MultiRegionMesh;
using Region = Meshes::MultiRegionMesh::Region;
using Interface = Meshes::MultiRegionMesh::Interface;
using real = real_t;
using Vec3 = MeshUtils::Vec3;

testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr real coordinate_shift = 100000.0;

ArrReal shift_edges(ArrReal edges, real shift)
{
    for (auto& edge : edges) edge += shift;
    return edges;
}

/** @brief Local axis edges of one Cartesian region of the closure mesh. */
struct CartesianSpec
{
    std::array<ArrReal, 3> edges;
    std::string name;
};

Meshes::CartesianRegion cartesian_from_spec(const CartesianSpec& spec, const std::array<real, 3>& shift)
{
    return Meshes::cartesian_region(spec.name,
        {{shift_edges(spec.edges[0], shift[0]), shift_edges(spec.edges[1], shift[1]),
            shift_edges(spec.edges[2], shift[2])}});
}

/**
 * @brief Two Cartesian regions joined by one 2:1-refined nonconforming plane.
 *
 * Non-dyadic 0.3 spacing keeps the fine faces exact power-of-two halves of
 * the coarse cells while making raw global-coordinate subtractions lossy at
 * the shifted origin.
 */
struct CoarseFineMesh
{
    SP<const MultiMesh> mesh;
    CartesianSpec coarse;
    CartesianSpec fine;
    std::array<real, 3> shift{};
};

CoarseFineMesh build_coarse_fine_mesh(const std::array<real, 3>& shift)
{
    CoarseFineMesh result;
    result.coarse = {std::array<ArrReal, 3>{{{0.0, 0.3}, {0.0, 0.3}, {0.0, 0.3}}}, "coarse"};
    result.fine = {std::array<ArrReal, 3>{{{0.3, 0.45, 0.6}, {0.0, 0.15, 0.3}, {0.0, 0.15, 0.3}}}, "fine"};
    auto coarse = cartesian_from_spec(result.coarse, shift);
    auto fine = cartesian_from_spec(result.fine, shift);

    Meshes::NonconformingInterface interface{0, 1, 1, 0, {}};
    Meshes::CoarseFineFaceMapping mapping{0, {}};
    for (size_t face = 0; face < coarse.layout().faces; ++face)
        if (coarse.topology().boundary_id(face) == 1) mapping.coarse_face = face;
    for (size_t face = 0; face < fine.layout().faces; ++face)
        if (fine.topology().boundary_id(face) == 0) mapping.fine_faces.push_back(face);
    if (mapping.fine_faces.size() != 4)
        throw std::logic_error("The closure mesh expects four fine subfaces per coarse face.");
    interface.faces.push_back(std::move(mapping));

    // The 1e5 shift costs low bits of the stored global edges themselves, so
    // construction-time tile matching uses a shift-aware input tolerance
    // (far above the ~1e-11 coordinate fuzz at 1e5, far below the 0.15
    // geometric scale). All numerical gates below stay at machine precision.
    const Meshes::InterfaceTolerance tolerance{1e-9, 1e-9};
    result.mesh = std::make_shared<const MultiMesh>(
        std::vector<Region>{std::move(coarse), std::move(fine)}, std::vector<Interface>{std::move(interface)},
        tolerance);
    result.shift = shift;
    return result;
}

/** @brief Two skewed extruded prism regions with a conforming shared plane. */
struct ExtrudedMesh
{
    SP<const MultiMesh> mesh;
    Arr<Vec3> nodes;
    Arr<Vec3> partner_nodes;
    ArrReal z_edges;
};

ExtrudedMesh build_extruded_mesh(const std::array<real, 3>& shift, ArrReal z_edges = ArrReal{0.0, 0.3, 0.6})
{
    ExtrudedMesh result;
    // Strictly convex, counter-clockwise triangle pair with non-dyadic
    // coordinates in the x/y plane.
    result.nodes = Arr<Vec3>{Vec3{shift[0] + 0.0, shift[1] + 0.0, 0.0},
        Vec3{shift[0] + 0.3, shift[1] + 0.0, 0.0}, Vec3{shift[0] + 0.3, shift[1] + 0.3, 0.0},
        Vec3{shift[0] + 0.0, shift[1] + 0.3, 0.0}};
    result.z_edges = shift_edges(std::move(z_edges), shift[2]);
    auto topology = std::make_shared<const Meshes::ExtrudedTopology>(4,
        Arr<Arr<unsigned>>{{0, 1, 2}, {0, 2, 3}}, 2,
        Arr<Meshes::SemiStructMeshTopo::BoundaryEdge>{{0, 1, "b0"}, {1, 2, "b1"}, {2, 3, "b2"}, {3, 0, "b3"}});
    // The partner mirrors across the x=0.3 plane so both join edges carry
    // the same physical segment with opposite outward normals.
    const Arr<Vec3> partner{Vec3{shift[0] + 0.6, shift[1] + 0.3, 0.0}, result.nodes[2], result.nodes[1],
        Vec3{shift[0] + 0.6, shift[1] + 0.0, 0.0}};
    result.partner_nodes = partner;
    auto first = Meshes::extruded_region("first", topology, result.nodes, result.z_edges);
    auto second = Meshes::extruded_region("second", topology, partner, result.z_edges);
    int join = -1;
    for (auto id : topology->boundary_batch_ids())
        if (topology->boundary_batch_name(id) == "b1") join = id;
    if (join < 0) throw std::logic_error("The extruded closure mesh requires its join boundary.");
    Meshes::ExplicitConformingInterface interface{0, 1, join, join, {}};
    for (size_t face = 0; face < first.layout().faces; ++face)
        if (first.topology().boundary_id(face) == static_cast<size_t>(join))
            interface.faces.emplace_back(face, face);
    result.mesh = std::make_shared<const MultiMesh>(std::vector<Region>{std::move(first), std::move(second)},
        std::vector<Interface>{std::move(interface)});
    return result;
}

/** @brief Exact per-cell displacement residual of the discrete GCL. */
real gcl_closure_residual(const MultiMesh& mesh)
{
    real worst = 0.0;
    for (size_t cell = 0; cell < mesh.num_cells(); ++cell)
    {
        Vec3 sum{};
        real magnitude = 0.0;
        for (const auto face : mesh.cell_faces(cell))
        {
            const auto area_vector = mesh.face_area_vector_outward(face, cell);
            sum = sum + area_vector;
            magnitude += area_vector.norm();
        }
        if (magnitude > 0.0) worst = std::max(worst, sum.norm() / magnitude);
    }
    return worst;
}

/** @brief Largest magnitude diagonal entry of an owned pressure matrix row. */
real maximum_diagonal(const Pack::matrix_type& matrix)
{
    real largest = 0.0;
    for (size_t row = 0; row < matrix.getLocalNumRows(); ++row)
    {
        typename Pack::matrix_type::local_inds_host_view_type columns;
        typename Pack::matrix_type::values_host_view_type values;
        matrix.getLocalRowView(static_cast<typename Pack::local_ordinal_type>(row), columns, values);
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            if (static_cast<size_t>(columns(entry)) == row)
                largest = std::max(largest, std::abs(values(entry)));
        }
    }
    return largest;
}

struct MatrixClosure
{
    real row_sum = 0.0;
    real column_sum = 0.0;
};

/** @brief Compensated row-sum and column-sum conservation residuals. */
MatrixClosure matrix_conservation_residuals(const Pack::matrix_type& matrix)
{
    MatrixClosure result;
    std::vector<SimpleFluid::detail::CompensatedSum<long double>> column_sums(
        matrix.getColMap()->getLocalNumElements());
    for (size_t row = 0; row < matrix.getLocalNumRows(); ++row)
    {
        typename Pack::matrix_type::local_inds_host_view_type columns;
        typename Pack::matrix_type::values_host_view_type values;
        matrix.getLocalRowView(static_cast<typename Pack::local_ordinal_type>(row), columns, values);
        SimpleFluid::detail::CompensatedSum<long double> row_sum;
        real diagonal = 0.0;
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            row_sum += values(entry);
            column_sums[static_cast<size_t>(columns(entry))] += values(entry);
            if (static_cast<size_t>(columns(entry)) == row) diagonal = std::abs(values(entry));
        }
        if (diagonal > 0.0)
        {
            const auto residual = std::abs(static_cast<real_t>(row_sum.value()));
            result.row_sum = std::max(result.row_sum, residual / diagonal);
        }
    }
    for (size_t column = 0; column < column_sums.size(); ++column)
    {
        // Compare against the owning row's diagonal magnitude.
        typename Pack::matrix_type::local_inds_host_view_type columns;
        typename Pack::matrix_type::values_host_view_type values;
        if (!matrix.getRowMap()->isNodeLocalElement(static_cast<typename Pack::local_ordinal_type>(column)))
            continue;
        matrix.getLocalRowView(static_cast<typename Pack::local_ordinal_type>(column), columns, values);
        real diagonal = 0.0;
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
            if (static_cast<size_t>(columns(entry)) == column) diagonal = std::abs(values(entry));
        if (diagonal > 0.0)
        {
            const auto residual = std::abs(static_cast<real_t>(column_sums[column].value()));
            result.column_sum = std::max(result.column_sum, residual / diagonal);
        }
    }
    return result;
}

/**
 * @brief Cell centers of the coarse/fine closure mesh in stored-local
 *        frames, recovering the mesh's stored edges exactly.
 *
 * The mesh stores `fl(spec + shift)` edge coordinates; the stored value
 * minus the shift is recovered bit-exactly (both sit within a factor of
 * two). Injecting the linear field from the stored geometry isolates what
 * the gate must catch: any operator arithmetic that re-amplifies
 * origin-scale rounding beyond the input representation itself.
 */
std::vector<real> analytic_linear_field(const CoarseFineMesh& mesh_data, const Vec3& gradient)
{
    std::vector<real> values;
    const auto& mesh = *mesh_data.mesh;
    const std::array<const CartesianSpec*, 2> specs{{&mesh_data.coarse, &mesh_data.fine}};
    std::array<std::array<ArrReal, 2>, 3> stored_local{};
    for (size_t axis = 0; axis < 3; ++axis)
        for (size_t region = 0; region < 2; ++region)
        {
            const auto stored = shift_edges(specs[region]->edges[axis], mesh_data.shift[axis]);
            for (size_t i = 0; i < stored.size(); ++i)
                stored_local[axis][region].push_back(stored[i] - mesh_data.shift[axis]);
        }
    for (size_t cell = 0; cell < mesh.num_cells(); ++cell)
    {
        const auto [region, native] = mesh.native_cell(cell);
        const auto& spec = *specs.at(region);
        const auto ni = spec.edges[0].size() - 1;
        const auto nj = spec.edges[1].size() - 1;
        const std::array<size_t, 3> indices{native % ni, (native / ni) % nj, native / (ni * nj)};
        std::array<real, 3> center{};
        for (size_t axis = 0; axis < 3; ++axis)
            center[axis] = std::midpoint(stored_local[axis][region][indices[axis]],
                stored_local[axis][region][indices[axis] + 1]);
        values.push_back(gradient.x * center[0] + gradient.y * center[1] + gradient.z * center[2]);
    }
    return values;
}

/** @brief Analytic prism-cell centers of the extruded closure mesh.
 *
 * A triangular prism's centroid is the mean of its six vertices; base-cell
 * and layer indices follow the extruded indexer (ij fastest). Positions are
 * measured in the mesh's own first-node local frame. */
std::vector<real> analytic_linear_field(const ExtrudedMesh& mesh_data, const Vec3& gradient)
{
    const Arr<Arr<unsigned>> loops{{0, 1, 2}, {0, 2, 3}};
    std::vector<real> values;
    const auto& mesh = *mesh_data.mesh;
    for (size_t cell = 0; cell < mesh.num_cells(); ++cell)
    {
        const auto [region, native] = mesh.native_cell(cell);
        const auto& nodes = region == 0 ? mesh_data.nodes : mesh_data.partner_nodes;
        const auto& frame = mesh_data.nodes[0];
        const auto ij = native % 2;
        const auto k = native / 2;
        // Stored-local node offsets are recovered exactly relative to the
        // region's first node; the prism centroid is the vertex mean.
        Vec3 center{};
        for (const auto node : loops[ij])
        {
            center.x += (nodes[node].x - frame.x) / 3.0;
            center.y += (nodes[node].y - frame.y) / 3.0;
        }
        center.z = std::midpoint(mesh_data.z_edges[k] - mesh_data.z_edges[0],
            mesh_data.z_edges[k + 1] - mesh_data.z_edges[0]);
        values.push_back(gradient.x * center.x + gradient.y * center.y + gradient.z * center.z);
    }
    return values;
}

/**
 * @brief Worst per-face reproduction error of the full corrected operator on
 *        an exact linear field, decomposed exactly as the assembly computes
 *        it.
 *
 * For each interior face the over-relaxed coefficient forms the implicit
 * flux and the tangential correction is closed by subtraction from the
 * stored face area vector. With T = S - alpha*d the per-face pair must
 * reproduce g . S_f, regardless of the domain's distance from the origin:
 * the error isolates injection consistency and closure-by-subtraction
 * roundoff, nothing else.
 */
real linear_face_reproduction_error(const Handle& mesh, const std::vector<real>& values, const Vec3& gradient)
{
    using local_ordinal_type = Pack::local_ordinal_type;
    real worst = 0.0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        for (const auto face : mesh.faces(cell))
        {
            const auto face_lid = static_cast<local_ordinal_type>(face);
            if (!mesh.is_interior_face(face_lid)) continue;
            const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell);
            const auto coefficient =
                FVM::detail::interior_pressure_coefficient(mesh, face_lid, cell, other);
            const auto center_delta = mesh.cell_center_vector(face_lid, cell);
            const auto area_vector = mesh.face_area_vector_outward(face_lid, cell);
            const auto distance_squared = center_delta.dot(center_delta);
            const auto alpha = coefficient * distance_squared / distance_squared;
            const Vec3 tangential = area_vector - center_delta * alpha;
            const auto implicit_flux =
                coefficient * (values[static_cast<size_t>(mesh.cell_geometry_global_id(other))]
                    - values[static_cast<size_t>(mesh.cell_geometry_global_id(cell))]);
            const auto correction = gradient.x * tangential.x + gradient.y * tangential.y +
                                    gradient.z * tangential.z;
            const auto reconstructed = gradient.x * area_vector.x + gradient.y * area_vector.y +
                                       gradient.z * area_vector.z;
            const auto error = implicit_flux + correction - reconstructed;
            const auto scale = std::abs(implicit_flux) + std::abs(correction) + std::abs(reconstructed);
            if (scale > 0.0) worst = std::max(worst, std::abs(error) / scale);
        }
    }
    return worst;
}

Teuchos::RCP<const Pack::matrix_type> pure_neumann_matrix(const Handle& mesh)
{
    auto homogeneous_neumann = [](int, size_t) { return BoundaryCondition{}; };
    return FVM::pressure_poisson_matrix<Pack>(mesh, std::optional<Pack::global_ordinal_type>{},
        homogeneous_neumann);
}

// ---------------------------------------------------------------------------
// Gate 1: discrete GCL closure, including coarse/fine interface cells.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, MultiRegionGclClosureIncludingCoarseFineCells)
{
    const std::pair<const char*, SP<const MultiMesh>> meshes[]{
        {"mixed", test::mixed_regions()}, {"coarse_fine", test::coarse_fine_regions()},
        {"extruded_pair", test::independent_extruded_regions()}};
    for (const auto& [label, mesh] : meshes)
    {
        const auto residual = gcl_closure_residual(*mesh);
        EXPECT_LE(residual, 1e-15);
    }
}

TEST(PressurePoissonClosureTest, ShiftedOriginGclClosureHoldsForCoarseFineMesh)
{
    const auto shifted = build_coarse_fine_mesh({coordinate_shift, coordinate_shift, coordinate_shift});
    // The coarse cell visits four retained fine subfaces; its closure is the
    // bit-level sum of the fine region's stored subface vectors.
    const auto shifted_residual = gcl_closure_residual(*shifted.mesh);
    EXPECT_LE(shifted_residual, 1e-15);
}

// ---------------------------------------------------------------------------
// Gate 2: algebraic conservation on pure-Neumann multi-region meshes.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, PureNeumannRowAndColumnSumLocks)
{
    for (const auto& composite : {test::mixed_regions(), test::coarse_fine_regions(),
             test::independent_extruded_regions()})
    {
        const Handle mesh(std::static_pointer_cast<const MultiMesh>(composite));
        const auto matrix = pure_neumann_matrix(mesh);
        const auto closure = matrix_conservation_residuals(*matrix);
        EXPECT_LE(closure.row_sum, 1e-15);
        EXPECT_LE(closure.column_sum, 1e-15);
    }
}

TEST(PressurePoissonClosureTest, ShiftedOriginRowAndColumnSumLocks)
{
    const auto shifted = build_coarse_fine_mesh({coordinate_shift, coordinate_shift, coordinate_shift});
    const Handle mesh(shifted.mesh);
    const auto matrix = pure_neumann_matrix(mesh);
    const auto closure = matrix_conservation_residuals(*matrix);
    EXPECT_LE(closure.row_sum, 1e-15);
    EXPECT_LE(closure.column_sum, 1e-15);
}

// ---------------------------------------------------------------------------
// Gate 3: zeroth-order constant-pressure reproduction.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, ConstantBackgroundPressureResidual)
{
    for (const auto& composite : {test::mixed_regions(), test::coarse_fine_regions(),
             test::independent_extruded_regions()})
    {
        const Handle mesh(std::static_pointer_cast<const MultiMesh>(composite));
        const auto matrix = pure_neumann_matrix(mesh);
        const auto diagonal = maximum_diagonal(*matrix);
        ASSERT_GT(diagonal, 0.0);

        Pack::vector_type ones(mesh.owned_cell_map(), true);
        ones.putScalar(1.0);
        Pack::vector_type residual(mesh.owned_cell_map(), true);
        const real background = 1.0e5;
        ones.scale(background);
        residual.putScalar(0.0);
        matrix->apply(ones, residual);
        const auto values = residual.getData();
        real worst = 0.0;
        for (const auto value : values) worst = std::max(worst, std::abs(value));
        const auto metric = worst / (std::abs(background) * diagonal);
        EXPECT_LE(metric, 1e-15);
    }
}

TEST(PressurePoissonClosureTest, ShiftedOriginConstantBackgroundPressureResidual)
{
    const auto shifted = build_coarse_fine_mesh({coordinate_shift, coordinate_shift, coordinate_shift});
    const Handle mesh(shifted.mesh);
    const auto matrix = pure_neumann_matrix(mesh);
    const auto diagonal = maximum_diagonal(*matrix);
    ASSERT_GT(diagonal, 0.0);

    Pack::vector_type ones(mesh.owned_cell_map(), true);
    ones.putScalar(1.0);
    Pack::vector_type residual(mesh.owned_cell_map(), true);
    const real background = 1.0e5;
    ones.scale(background);
    residual.putScalar(0.0);
    matrix->apply(ones, residual);
    const auto values = residual.getData();
    real worst = 0.0;
    for (const auto value : values) worst = std::max(worst, std::abs(value));
    const auto metric = worst / (std::abs(background) * diagonal);
    EXPECT_LE(metric, 1e-15);
}

// ---------------------------------------------------------------------------
// Gate 4: first-order linear hydrostatic reproduction across shifted origins.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, LinearHydrostaticFieldReproductionAcrossShiftedOrigins)
{
    const Vec3 gradient{0.7, -1.3, 0.9};
    const auto unshifted = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const auto shifted = build_coarse_fine_mesh(
        {coordinate_shift, coordinate_shift, coordinate_shift});

    {
        const Handle mesh(unshifted.mesh);
        const auto values = analytic_linear_field(unshifted, gradient);
        const auto error = linear_face_reproduction_error(mesh, values, gradient);
        EXPECT_LE(error, 1e-13);
    }
    {
        const Handle mesh(shifted.mesh);
        const auto values = analytic_linear_field(shifted, gradient);
        // No geometric roundoff leakage: the shifted domain reproduces the
        // linear field at the same machine-precision level as the local one.
        const auto error = linear_face_reproduction_error(mesh, values, gradient);
        EXPECT_LE(error, 1e-13);
    }
}

TEST(PressurePoissonClosureTest, LinearFieldOnShiftedSkewedExtrudedMesh)
{
    const Vec3 gradient{0.7, -1.3, 0.9};
    const auto unshifted = build_extruded_mesh({0.0, 0.0, 0.0});
    const auto shifted = build_extruded_mesh(
        {coordinate_shift, coordinate_shift, coordinate_shift});

    const auto evaluate = [&gradient](const ExtrudedMesh& mesh_data, const char* label)
    {
        const Handle mesh(mesh_data.mesh);
        const auto values = analytic_linear_field(mesh_data, gradient);
        const auto error = linear_face_reproduction_error(mesh, values, gradient);
        return error;
    };
    EXPECT_LE(evaluate(unshifted, "unshifted"), 1e-13);
    EXPECT_LE(evaluate(shifted, "shifted"), 1e-13);
}

// ---------------------------------------------------------------------------
// Gate 5: cold-start near-zero RHS handling.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, ColdStartZeroRhsPreservesAccumulatedPressure)
{
    const auto mesh_data = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const SP<const Handle> mesh = std::make_shared<const Handle>(mesh_data.mesh);

    CellField<Pack> pressure(mesh, "pressure");
    CellField<Pack> correction(mesh, "correction");
    VectorCellField<Pack> velocity(mesh, "velocity");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        pressure.set_owned_value(cell, 1.0e5);
        correction.set_owned_value(cell, 0.0);
        velocity.set_owned_value(cell, {0.0, 0.0, 0.0});
    }
    pressure.sync_ghosts();

    const BoundaryConditionSet boundary_conditions;
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    LinearSolverOptions options;
    options.tolerance = 1.0e-10;
    PressureProjectionEquation<Pack> equation(mesh, options);

    const auto small_time_step = 1.0e-6;
    const auto result = equation.project(pressure, correction, small_time_step, 1000.0, cache, velocity);

    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_EQ(result.linear_solve.iterations, 0);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        EXPECT_EQ(correction.local_value(cell), 0.0);
        // The accumulated physical pressure is never overwritten with zero.
        EXPECT_EQ(pressure.local_value(cell), 1.0e5);
        EXPECT_TRUE(std::isfinite(pressure.local_value(cell)));
    }
}

TEST(PressurePoissonClosureTest, ColdStartTinyRhsExitsWithinTolerance)
{
    const auto mesh_data = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const SP<const Handle> mesh = std::make_shared<const Handle>(mesh_data.mesh);

    CellField<Pack> pressure(mesh, "pressure");
    CellField<Pack> correction(mesh, "correction");
    VectorCellField<Pack> velocity(mesh, "velocity");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        pressure.set_owned_value(cell, 0.0);
        correction.set_owned_value(cell, 0.0);
        velocity.set_owned_value(cell, {0.0, 0.0, 0.0});
    }
    pressure.sync_ghosts();

    const BoundaryConditionSet boundary_conditions;
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    LinearSolverOptions options;
    options.tolerance = 1.0e-10;
    PressureProjectionEquation<Pack> equation(mesh, options);

    // A tiny globally compatible integrated target: ||b|| ~ 1e-14.
    std::vector<real> rates(mesh->num_owned_cells());
    for (size_t cell = 0; cell < rates.size(); ++cell)
        rates[cell] = (cell % 2 == 0 ? 1.0 : -1.0) * 1.0e-14;
    const VolumeContinuityTarget<Pack> target(mesh, rates, 1);

    const auto small_time_step = 1.0e-6;
    const auto result = equation.project(pressure, correction, small_time_step, 1000.0, cache, velocity, target);

    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_LT(result.linear_solve.iterations, 200);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        EXPECT_TRUE(std::isfinite(correction.local_value(cell)));
        EXPECT_TRUE(std::isfinite(pressure.local_value(cell)));
        EXPECT_TRUE(std::isfinite(velocity.local_value(cell).x));
    }
    // The tiny target stays tiny: no amplified geometric closure noise.
    EXPECT_LT(std::abs(result.pressure_correction), 1.0e-4);
}


// ---------------------------------------------------------------------------
// High-aspect-ratio LSQ conditioning mesh: a triangulated strip extruded
// into two thin axial layers, mirrored across its single join edge.
// ---------------------------------------------------------------------------

struct HighRatioMesh
{
    SP<const MultiMesh> mesh;
    Arr<Vec3> nodes;
    Arr<Vec3> partner_nodes;
    ArrReal z_edges;
    Arr<Arr<unsigned>> loops;
};

HighRatioMesh build_high_ratio_mesh(const std::array<real, 3>& shift)
{
    // Two quad rows of four columns (width/height 0.075), each quad split
    // into two counter-clockwise triangles; axial slabs of 0.0003 give
    // aspect ratio 250. Every base cell keeps at least two independent
    // in-plane sampling directions.
    constexpr real width = 0.075;
    constexpr real slab = 0.0003;
    constexpr size_t columns = 4;
    constexpr size_t rows = 2;
    HighRatioMesh result;
    const auto node_index = [](size_t i, size_t j) { return i + (columns + 1) * j; };
    result.loops = Arr<Arr<unsigned>>{};
    for (size_t j = 0; j < rows; ++j)
        for (size_t i = 0; i < columns; ++i)
        {
            // Quad base cells keep at least two independent in-plane sampling
            // directions everywhere, including at the strip corners.
            const auto a = static_cast<unsigned>(node_index(i, j));
            const auto b = static_cast<unsigned>(node_index(i + 1, j));
            const auto c = static_cast<unsigned>(node_index(i + 1, j + 1));
            const auto d = static_cast<unsigned>(node_index(i, j + 1));
            result.loops.push_back(Arr<unsigned>{a, b, c, d});
        }
    result.nodes = Arr<Vec3>{};
    for (size_t j = 0; j <= rows; ++j)
        for (size_t i = 0; i <= columns; ++i)
            result.nodes.push_back(Vec3{shift[0] + static_cast<real>(i) * width,
                shift[1] + static_cast<real>(j) * width, 0.0});
    // The partner is the same strip rotated onto [0.3, 0.6]: same node index
    // mapping, x mirrored and y reversed so both orientations stay CCW and
    // the join edges carry opposite outward normals.
    result.partner_nodes = Arr<Vec3>{};
    for (size_t j = 0; j <= rows; ++j)
        for (size_t i = 0; i <= columns; ++i)
            result.partner_nodes.push_back(
                Vec3{shift[0] + 2 * static_cast<real>(columns) * width - static_cast<real>(i) * width,
                    shift[1] + (static_cast<real>(rows - j)) * width, 0.0});
    result.z_edges = shift_edges(ArrReal{0.0, slab, 2 * slab}, shift[2]);

    Arr<Meshes::SemiStructMeshTopo::BoundaryEdge> boundary_edges;
    for (size_t i = 0; i < columns; ++i)
    {
        boundary_edges.push_back({static_cast<unsigned>(node_index(i, 0)),
            static_cast<unsigned>(node_index(i + 1, 0)), "bottom"});
        boundary_edges.push_back({static_cast<unsigned>(node_index(i, rows)),
            static_cast<unsigned>(node_index(i + 1, rows)), "top"});
    }
    for (size_t j = 0; j < rows; ++j)
    {
        boundary_edges.push_back({static_cast<unsigned>(node_index(0, j)),
            static_cast<unsigned>(node_index(0, j + 1)), "left"});
        boundary_edges.push_back({static_cast<unsigned>(node_index(columns, j)),
            static_cast<unsigned>(node_index(columns, j + 1)), "join"});
    }
    auto topology = std::make_shared<const Meshes::ExtrudedTopology>(
        result.nodes.size(), result.loops, 2, boundary_edges);
    auto first = Meshes::extruded_region("thin_first", topology, result.nodes, result.z_edges);
    auto second = Meshes::extruded_region("thin_second", topology, result.partner_nodes, result.z_edges);
    int join = -1;
    for (auto id : topology->boundary_batch_ids())
        if (topology->boundary_batch_name(id) == "join") join = id;
    if (join < 0) throw std::logic_error("The high-ratio mesh requires its join boundary.");
    Meshes::ExplicitConformingInterface interface{0, 1, join, join, {}};
    // Pair the join faces by physical centroid: the partner is y-reversed,
    // so its face ordinals do not line up column-for-column.
    Arr<size_t> unmatched;
    for (size_t face = 0; face < first.layout().faces; ++face)
    {
        if (first.topology().boundary_id(face) != static_cast<size_t>(join)) continue;
        const auto centroid = first.geometry().face_centroid(face);
        bool found = false;
        for (size_t other = 0; other < second.layout().faces; ++other)
        {
            if (second.topology().boundary_id(other) != static_cast<size_t>(join)) continue;
            if ((centroid - second.geometry().face_centroid(other)).norm() < 1e-6)
            {
                interface.faces.emplace_back(face, other);
                found = true;
                break;
            }
        }
        if (!found) unmatched.push_back(face);
    }
    if (!unmatched.empty()) throw std::logic_error("The high-ratio mesh has unpaired join faces.");
    result.mesh = std::make_shared<const MultiMesh>(
        std::vector<Region>{std::move(first), std::move(second)},
        std::vector<Interface>{std::move(interface)});
    return result;
}

/** @brief Cell values of g . (x - x_ref) in the strip mesh's local frames,
 *         computed from the construction arrays the mesh itself stores. */
std::vector<real> local_linear_field(const HighRatioMesh& mesh_data, const Vec3& gradient)
{
    std::vector<real> values;
    const auto& mesh = *mesh_data.mesh;
    const auto base = mesh_data.nodes[0];
    const auto z_reference = mesh_data.z_edges[0];
    for (size_t cell = 0; cell < mesh.num_cells(); ++cell)
    {
        const auto [region, native] = mesh.native_cell(cell);
        const auto& nodes = region == 0 ? mesh_data.nodes : mesh_data.partner_nodes;
        const auto loop = mesh_data.loops[native % mesh_data.loops.size()];
        const auto layer = native / mesh_data.loops.size();
        Vec3 center{};
        for (const auto node : loop)
        {
            center.x += (nodes[node].x - base.x) / static_cast<real>(loop.size());
            center.y += (nodes[node].y - base.y) / static_cast<real>(loop.size());
        }
        // Recover the stored-local z values first; midpoints of the shifted
        // values themselves would round at the origin scale.
        const auto lower = mesh_data.z_edges[layer] - z_reference;
        const auto upper = mesh_data.z_edges[layer + 1] - z_reference;
        center.z = std::midpoint(lower, upper);
        values.push_back(gradient.x * center.x + gradient.y * center.y + gradient.z * center.z);
    }
    return values;
}

// ---------------------------------------------------------------------------
// Gate 6: full corrected operator (A - C) through the production machinery.
// ---------------------------------------------------------------------------

/**
 * @brief Row and conservation residuals of the full corrected operator.
 *
 * The correction term C(x) is evaluated exactly as the production
 * non-orthogonal correction computes it: cached boundary-aware LSQ
 * gradients of the injected field, owner/neighbor weights that sum to one,
 * and the tangential area vector closed by subtraction from the stored
 * face area vector. The implicit part comes from the assembled pure-
 * Neumann matrix. Returns the worst owned-row residual normalized by the
 * value scale times the diagonal magnitude, and the absolute global sum of
 * all row residuals (pairwise flux cancellation across owner/neighbor rows,
 * including across rank boundaries).
 */
struct FullOperatorMetrics
{
    real row_residual = 0.0;
    real conservation = 0.0;
    real correction_zero = 0.0;
};

FullOperatorMetrics full_operator_metrics(const SP<const Handle>& mesh, const std::vector<real>& values,
    real value_scale)
{
    using local_ordinal_type = Pack::local_ordinal_type;
    const auto& owned_map = *mesh->owned_cell_map();
    const auto comm = owned_map.getComm();

    ScalarCellFieldStored<Pack> field(mesh, "closure_full_operator_field");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<local_ordinal_type>(owned);
        field.set_owned_value(owned,
            values[static_cast<size_t>(mesh->cell_geometry_global_id(lid))]);
    }
    field.sync_ghosts();

    BoundaryConditionMap gradient_conditions;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        static_cast<void>(batch);
        gradient_conditions[mesh->boundary_batch_name(batch_id)] = {BoundaryConditionType::Neumann, 0.0};
    }
    VectorCellFieldStored<Pack> gradients(mesh, "closure_full_operator_gradient");
    FVM::CellGradientCache<Pack, Handle> cache(mesh);
    FVM::cell_gradient(field, gradient_conditions, gradients, cache);
    // Production consumers synchronize ghosts before reading local views.
    gradients.sync_ghosts();

    const auto matrix = pure_neumann_matrix(*mesh);
    const auto diagonal = maximum_diagonal(*matrix);

    // Implicit (matrix) part on owned rows.
    Pack::vector_type field_owned(mesh->owned_cell_map(), true);
    {
        const auto view = field_owned.getDataNonConst();
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            view[owned] = field.local_value(static_cast<local_ordinal_type>(owned));
    }
    Pack::vector_type implicit_rows(mesh->owned_cell_map(), true);
    implicit_rows.putScalar(0.0);
    matrix->apply(field_owned, implicit_rows);

    // Correction part: one owner-oriented flux per interior face, accumulated
    // with the balance-sign convention (+ for the owner row, - for the
    // neighbor row), exactly as the stored flux-balance cache does.
    std::vector<real> correction_rows(mesh->num_owned_cells(), 0.0);
    std::vector<real> row_magnitudes(mesh->num_owned_cells(), 0.0);
    real correction_magnitude = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        for (const auto face : mesh->faces(cell))
        {
            const auto face_lid = static_cast<local_ordinal_type>(face);
            if (!mesh->is_interior_face(face_lid)) continue;
            const auto owner = mesh->owner_cell(face_lid);
            const auto neighbor = mesh->opposite_or_periodic_neighbor_cell(face_lid, owner);
            const auto weights =
                FVM::detail::stored_interior_face_linear_weights(*mesh, face_lid, owner, neighbor);
            const auto interpolated = gradients.local_value(owner) * weights.first
                                     + gradients.local_value(neighbor) * weights.second;
            const auto coefficient =
                FVM::detail::interior_pressure_coefficient(*mesh, face_lid, owner, neighbor);
            const auto center_delta = mesh->cell_center_vector(face_lid, owner);
            const auto area_vector = mesh->face_area_vector_outward(face_lid, owner);
            const auto distance_squared = center_delta.dot(center_delta);
            const auto alpha = coefficient * distance_squared / distance_squared;
            const Vec3 tangential = area_vector - center_delta * alpha;
            const auto flux = interpolated.dot(tangential);
            const auto sign = owner == cell ? 1.0 : -1.0;
            correction_rows[owned] += sign * flux;
            row_magnitudes[owned] += std::abs(flux);
            correction_magnitude += std::abs(flux);
        }
    }

    FullOperatorMetrics metrics;
    const auto row_view = implicit_rows.getData();
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto residual = row_view[owned] - correction_rows[owned];
        const auto normalized = std::abs(residual) / (std::abs(value_scale) * diagonal);
        metrics.row_residual = std::max(metrics.row_residual, normalized);
        metrics.correction_zero = std::max(metrics.correction_zero,
            std::abs(correction_rows[owned]) / (std::abs(value_scale) * diagonal));
    }
    // Pairwise flux cancellation (the matrix-free column-sum lock): the
    // global sum over all rows vanishes bit-exactly when every face flux
    // enters its two rows with identical bits, including across ranks.
    SimpleFluid::detail::CompensatedSum<long double> local_sum;
    real local_magnitude = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_sum += static_cast<long double>(row_view[owned]) - static_cast<long double>(correction_rows[owned]);
        local_magnitude += std::abs(row_view[owned]) + row_magnitudes[owned];
    }
    const std::array<real, 2> local_values{
        static_cast<real>(local_sum.value()), local_magnitude};
    std::array<real, 2> global_values{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(local_values.size()), local_values.data(),
        global_values.data());
    // Denominator keeps the metric meaningful for a constant field, whose
    // row magnitudes are themselves at the roundoff floor: the value scale
    // times the diagonal bounds every row's representable contribution.
    const auto global_rows = static_cast<real>(owned_map.getGlobalNumElements());
    const auto denominator =
        global_values[1] + std::abs(value_scale) * diagonal * global_rows;
    metrics.conservation = denominator > 0.0 ? std::abs(global_values[0]) / denominator : 0.0;
    return metrics;
}

TEST(PressurePoissonClosureTest, FullOperatorConstantPreservation)
{
    const real background = 1.0e5;
    const auto unshifted = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const auto shifted = build_coarse_fine_mesh(
        {coordinate_shift, coordinate_shift, coordinate_shift});
    const auto extruded = build_extruded_mesh({0.0, 0.0, 0.0});
    const auto extruded_shifted = build_extruded_mesh(
        {coordinate_shift, coordinate_shift, coordinate_shift});

    const auto check = [&](const SP<const Handle>& mesh, const char* label)
    {
        const std::vector<real> values(mesh->num_cells(), background);
        const auto metrics = full_operator_metrics(mesh, values, background);
        std::cout << "METRIC gate6_full_constant_" << label << "_row " << std::setprecision(17)
                  << metrics.row_residual << "\n";
        std::cout << "METRIC gate6_full_constant_" << label << "_conservation " << std::setprecision(17)
                  << metrics.conservation << "\n";
        // A constant field carries no gradient: the correction term C must be
        // bit-exactly zero through the production LSQ machinery.
        EXPECT_EQ(metrics.correction_zero, 0.0);
        EXPECT_LE(metrics.row_residual, 1e-15);
        EXPECT_LE(metrics.conservation, 1e-15);
    };
    check(std::make_shared<const Handle>(unshifted.mesh), "coarse_fine");
    check(std::make_shared<const Handle>(shifted.mesh), "coarse_fine_shifted");
    check(std::make_shared<const Handle>(extruded.mesh), "extruded");
    check(std::make_shared<const Handle>(extruded_shifted.mesh), "extruded_shifted");
}

TEST(PressurePoissonClosureTest, FullOperatorLinearFluxConservation)
{
    const Vec3 gradient{0.7, -1.3, 0.9};
    const auto unshifted = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const auto shifted = build_coarse_fine_mesh(
        {coordinate_shift, coordinate_shift, coordinate_shift});

    const auto check = [&](const SP<const Handle>& mesh, const std::vector<real>& values,
        const char* label)
    {
        const auto metrics = full_operator_metrics(mesh, values, 1.0);
        std::cout << "METRIC gate6_full_linear_" << label << "_conservation " << std::setprecision(17)
                  << metrics.conservation << "\n";
        // Every interior face flux enters its two rows with identical bits
        // and opposite balance signs, so the assembled operator conserves the
        // linear field pairwise, exactly as it conserves any field.
        EXPECT_LE(metrics.conservation, 1e-15);
    };
    {
        const SP<const Handle> mesh = std::make_shared<const Handle>(unshifted.mesh);
        check(mesh, analytic_linear_field(unshifted, gradient), "coarse_fine");
    }
    {
        const SP<const Handle> mesh = std::make_shared<const Handle>(shifted.mesh);
        check(mesh, analytic_linear_field(shifted, gradient), "coarse_fine_shifted");
    }
}

// ---------------------------------------------------------------------------
// Gate 7: small-time-step scaling safety of the absolute guards.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, SmallTimeStepTargetIsSolvedNotShortCircuited)
{
    const auto mesh_data = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const SP<const Handle> mesh = std::make_shared<const Handle>(mesh_data.mesh);

    // A genuine physical integrated target, orders above any noise floor.
    // Pairs cancel exactly so the globally compatible check passes.
    std::vector<real> rates(mesh->num_owned_cells());
    for (size_t cell = 0; cell < rates.size(); ++cell)
        rates[cell] = cell + 1 < rates.size() ? (cell % 2 == 0 ? 1.0 : -1.0) * 1.0e-7 : 0.0;

    const auto project_with = [&](real time_step)
    {
        CellField<Pack> pressure(mesh, "pressure");
        CellField<Pack> correction(mesh, "correction");
        VectorCellField<Pack> velocity(mesh, "velocity");
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            pressure.set_owned_value(cell, 0.0);
            correction.set_owned_value(cell, 0.0);
            velocity.set_owned_value(cell, {0.0, 0.0, 0.0});
        }
        pressure.sync_ghosts();
        const BoundaryConditionSet boundary_conditions;
        const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
        LinearSolverOptions options;
        options.tolerance = 1.0e-10;
        PressureProjectionEquation<Pack> equation(mesh, options);
        const VolumeContinuityTarget<Pack> target(mesh, rates, 1);
        return equation.project(pressure, correction, time_step, 1000.0, cache, velocity, target);
    };

    const auto coarse_step = project_with(1.0e-3);
    const auto fine_step = project_with(1.0e-5);
    EXPECT_TRUE(coarse_step.linear_solve.converged);
    EXPECT_TRUE(fine_step.linear_solve.converged);
    // Both solves are real Krylov solves: the physical target is far above
    // the flux-scale noise floor, so the absolute short-circuit cannot fire.
    EXPECT_GT(coarse_step.linear_solve.iterations, 0);
    EXPECT_GT(fine_step.linear_solve.iterations, 0);
    // With dt on the left side the normalized correction scales as 1/dt;
    // a short-circuit or scaling mismatch would destroy this ratio.
    ASSERT_GT(coarse_step.pressure_correction, 0.0);
    ASSERT_GT(fine_step.pressure_correction, 0.0);
    const auto ratio = fine_step.pressure_correction / coarse_step.pressure_correction;
    std::cout << "METRIC gate7_small_dt_correction_ratio " << std::setprecision(17) << ratio << "\n";
    EXPECT_NEAR(ratio, 100.0, 25.0);
}

// ---------------------------------------------------------------------------
// Gate 8: high-aspect-ratio LSQ conditioning of the normal-equation solve.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, HighAspectRatioLsqGradientReproduction)
{
    const Vec3 gradient{0.7, -1.3, 0.9};
    const auto unshifted = build_high_ratio_mesh({0.0, 0.0, 0.0});
    const auto shifted = build_high_ratio_mesh({coordinate_shift, coordinate_shift, coordinate_shift});
    ASSERT_EQ(unshifted.loops.size(), 8u);
    ASSERT_EQ(unshifted.mesh->num_cells(), 32u);

    const auto evaluate = [&](const HighRatioMesh& mesh_data, const char* label)
    {
        const SP<const Handle> mesh = std::make_shared<const Handle>(mesh_data.mesh);
        const auto values = local_linear_field(mesh_data, gradient);
        ScalarCellFieldStored<Pack> field(mesh, "high_ar_linear");
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            field.set_owned_value(owned, values[static_cast<size_t>(mesh->cell_geometry_global_id(
                static_cast<Pack::local_ordinal_type>(owned)))]);
        field.sync_ghosts();
        BoundaryConditionMap gradient_conditions;
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
        {
            static_cast<void>(batch);
            gradient_conditions[mesh->boundary_batch_name(batch_id)] = {BoundaryConditionType::Neumann, 0.0};
        }
        VectorCellFieldStored<Pack> gradients(mesh, "high_ar_gradient");
        FVM::CellGradientCache<Pack, Handle> cache(mesh);
        FVM::cell_gradient(field, gradient_conditions, gradients, cache);
        gradients.sync_ghosts();

        const auto gradient_norm = std::sqrt(gradient.x * gradient.x + gradient.y * gradient.y
            + gradient.z * gradient.z);
        // Cached boundary-aware scheme error (includes the standard LSQ
        // Neumann-sample directions in the normal matrix): recorded as
        // scheme-level evidence; it is a first-order discretization
        // property of the all-Neumann fit, not roundoff or conditioning.
        {
            real scheme = 0.0;
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto value = gradients.local_value(static_cast<Pack::local_ordinal_type>(owned));
                const Vec3 error{value.x - gradient.x, value.y - gradient.y, value.z - gradient.z};
                scheme = std::max(scheme, error.norm() / gradient_norm);
            }
            std::cout << "METRIC gate8_high_ar_cached_scheme_" << label << " " << std::setprecision(17)
                      << scheme << "\n";
        }
        // Conditioning metric: the interior-only LSQ path isolates the
        // normal-equation linear algebra (solve_3x3) from the Neumann
        // sample treatment. An exact linear field must be reproduced to
        // roundoff even at aspect ratio 250.
        real worst = 0.0;
        {
            VectorCellFieldStored<Pack> fresh(mesh, "high_ar_fresh");
            FVM::cell_gradient(field, fresh);
            fresh.sync_ghosts();
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto value = fresh.local_value(static_cast<Pack::local_ordinal_type>(owned));
                const Vec3 error{value.x - gradient.x, value.y - gradient.y, value.z - gradient.z};
                const auto magnitude = error.norm() / gradient_norm;
                if (magnitude > worst)
                {
                    worst = magnitude;
                    std::cout << "ZDEBUG8 " << label << " worst cell=" << owned << " grad=(" << value.x
                              << "," << value.y << "," << value.z << ") err=" << magnitude << "\n";
                    const auto stencils = FVM::detail::least_squares_gradient_stencils(*mesh);
                    for (const auto& entry : stencils[owned])
                        std::cout << "ZDEBUG8 entry lid=" << entry.cell_lid << " coeff=("
                                  << entry.coefficient.x << "," << entry.coefficient.y << ","
                                  << entry.coefficient.z << ") value=" << field.local_value(entry.cell_lid)
                                  << " cellvalue=" << field.local_value(static_cast<Pack::local_ordinal_type>(owned))
                                  << "\n";
                }
            }
        }
        std::cout << "METRIC gate8_high_ar_lsq_" << label << " " << std::setprecision(17) << worst << "\n";
        // Per-face linear reproduction through the assembled operator stays
        // at machine precision independent of the stencil solver.
        const auto reproduction = linear_face_reproduction_error(*mesh, values, gradient);
        std::cout << "METRIC gate8_high_ar_face_" << label << " " << std::setprecision(17)
                  << reproduction << "\n";
        EXPECT_LE(reproduction, 1e-12);
        return worst;
    };
    const auto unshifted_error = evaluate(unshifted, "unshifted");
    const auto shifted_error = evaluate(shifted, "shifted");
    EXPECT_LE(unshifted_error, 1e-12);
    EXPECT_LE(shifted_error, 1e-12);
}

// ---------------------------------------------------------------------------
// Gate 9: multi-rank closure across partition boundaries.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, MultiRankShiftedClosureAndColdStart)
{
    const auto shifted = build_coarse_fine_mesh(
        {coordinate_shift, coordinate_shift, coordinate_shift});
    const auto comm = Teuchos::DefaultComm<int>::getComm();
    const auto ranks = static_cast<size_t>(comm->getSize());
    if (ranks < 2)
    {
        GTEST_SKIP() << "This gate requires at least two MPI ranks.";
    }
    // Topology-preserving composite partition: the shifted multi-region
    // source is supplied by one rank and split across the communicator, so
    // the coarse/fine interface crosses rank boundaries.
    const auto source = comm->getRank() == 0 ? shifted.mesh : nullptr;
    const auto partition = MeshPartitioner<Pack>::partition(CompositeMeshSource(source), {}, comm);
    const SP<const Handle> mesh = std::make_shared<Handle>(partition);

    // Cross-rank local-frame displacements and per-face linear reproduction:
    // every owned row exercises faces whose neighbor cell is a ghost
    // maintained by another rank.
    const Vec3 gradient{0.7, -1.3, 0.9};
    const auto values = analytic_linear_field(shifted, gradient);
    const auto reproduction = linear_face_reproduction_error(*mesh, values, gradient);
    std::cout << "METRIC gate9_multi_rank_face_reproduction " << std::setprecision(17) << reproduction
              << "\n";
    EXPECT_LE(reproduction, 1e-13);

    // Full corrected operator (A - C) constant preservation across ranks,
    // including the collective pairwise-flux conservation sum. Values are
    // indexed by the partition-independent geometry id.
    std::vector<real> constant_values(shifted.mesh->num_cells(), 1.0e5);
    const auto metrics = full_operator_metrics(mesh, constant_values, 1.0e5);
    std::cout << "METRIC gate9_multi_rank_full_row " << std::setprecision(17) << metrics.row_residual
              << "\n";
    std::cout << "METRIC gate9_multi_rank_full_conservation " << std::setprecision(17)
              << metrics.conservation << "\n";
    EXPECT_EQ(metrics.correction_zero, 0.0);
    EXPECT_LE(metrics.row_residual, 1e-15);
    EXPECT_LE(metrics.conservation, 1e-15);

    // Collective cold start on the distributed mesh: the compensated
    // compatibility reduction and the pure-Neumann branch must agree on
    // every rank, and the accumulated pressure must stay bit-preserved.
    CellField<Pack> pressure(mesh, "pressure");
    CellField<Pack> correction(mesh, "correction");
    VectorCellField<Pack> velocity(mesh, "velocity");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        pressure.set_owned_value(cell, 1.0e5);
        correction.set_owned_value(cell, 0.0);
        velocity.set_owned_value(cell, {0.0, 0.0, 0.0});
    }
    pressure.sync_ghosts();
    const BoundaryConditionSet boundary_conditions;
    const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    LinearSolverOptions options;
    options.tolerance = 1.0e-10;
    PressureProjectionEquation<Pack> equation(mesh, options);
    const auto result = equation.project(pressure, correction, 1.0e-6, 1000.0, cache, velocity);
    EXPECT_TRUE(result.linear_solve.converged);
    EXPECT_EQ(result.linear_solve.iterations, 0);
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        EXPECT_EQ(correction.local_value(cell), 0.0);
        EXPECT_EQ(pressure.local_value(cell), 1.0e5);
    }
}



// ---------------------------------------------------------------------------
// Pure-Neumann mean-projection guard: cell-count-invariant boundary.
// ---------------------------------------------------------------------------

TEST(PressurePoissonClosureTest, MeanProjectionGuardIsInvariantToCellCount)
{
    const auto mesh_data = build_coarse_fine_mesh({0.0, 0.0, 0.0});
    const SP<const Handle> mesh = std::make_shared<const Handle>(mesh_data.mesh);
    ASSERT_GT(mesh->num_owned_cells(), 1u);
    const auto& owned_map = *mesh->owned_cell_map();
    // Pure-Neumann gauge: the globally smallest owned row.
    const auto gauge = owned_map.getMinAllGlobalIndex();
    const auto poisson_rows = owned_map.getGlobalNumElements() - 1;
    ASSERT_GT(poisson_rows, 0u);

    // Zero velocity and zero pressure give bit-exact zero predictor fluxes,
    // so the integral RHS equals the continuity target row by row.
    const real pair_rate = 1.0e-9;
    const auto project_with_net = [&](real net_source)
    {
        std::vector<real> rates(mesh->num_owned_cells(), 0.0);
        // Canceling pairs across the Poisson rows; the gauge row stays at
        // zero and the net rides the last pair.
        for (size_t cell = 1; cell + 1 < rates.size(); cell += 2)
        {
            rates[cell] = pair_rate;
            rates[cell + 1] = -pair_rate;
        }
        rates.back() += net_source;
        real magnitude = 0.0;
        for (const auto rate : rates) magnitude += std::abs(rate);

        CellField<Pack> pressure(mesh, "pressure");
        CellField<Pack> correction(mesh, "correction");
        VectorCellField<Pack> velocity(mesh, "velocity");
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            pressure.set_owned_value(cell, 0.0);
            correction.set_owned_value(cell, 0.0);
            velocity.set_owned_value(cell, {0.0, 0.0, 0.0});
        }
        pressure.sync_ghosts();
        const BoundaryConditionSet boundary_conditions;
        const auto cache = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
        LinearSolverOptions options;
        options.tolerance = 1.0e-10;
        PressureProjectionEquation<Pack> equation(mesh, options);
        const VolumeContinuityTarget<Pack> target(mesh, rates, 1);
        const auto result = equation.project(pressure, correction, 1.0e-3, 1000.0, cache, velocity, target);
        EXPECT_TRUE(result.linear_solve.converged);

        // The base RHS is exactly what the Krylov solve received after the
        // mean-projection guard decision.
        const auto base_rhs =
            SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::projection_base_rhs(equation);
        SimpleFluid::detail::CompensatedSum<long double> sum;
        const auto values = base_rhs->getData();
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto row_gid = owned_map.getGlobalElement(static_cast<Pack::local_ordinal_type>(owned));
            if (row_gid == gauge)
            {
                EXPECT_EQ(values[owned], 0.0);
                continue;
            }
            sum += static_cast<long double>(values[owned]);
        }
        const auto floor =
            static_cast<real>(64) * std::numeric_limits<real>::epsilon() * magnitude;
        return std::pair<real, real>{static_cast<real>(sum.value()), floor};
    };

    // Below the boundary: a net source at half the noise floor of the global
    // sum is pure drift. The guard projects it to zero mean, so the solved
    // RHS sums only to the subtraction roundoff (far below the source).
    {
        const auto drift = project_with_net(0.0);
        const auto floor = drift.second;
        const auto stripped = project_with_net(0.5 * floor);
        std::cout << "METRIC guard_drift_net " << std::setprecision(17) << 0.5 * floor
                  << " post_sum " << std::abs(stripped.first) << "\n";
        EXPECT_LE(std::abs(stripped.first), 1e-23);
    }

    // In the cell-count gap: a subtle net source with
    // floor < |sum| < N_poisson * floor is genuine signal. The guard must
    // NOT strip it: the solved RHS keeps the net source exactly. (The
    // previous mean-vs-floor comparison would have stripped it, since its
    // mean stays below the floor on any mesh with N_poisson > 1.)
    {
        const auto reference = project_with_net(0.0);
        const auto floor = reference.second;
        const auto subtle = project_with_net(4.0 * floor);
        std::cout << "METRIC guard_gap_net " << std::setprecision(17) << 4.0 * floor
                  << " post_sum " << subtle.first << "\n";
        EXPECT_NEAR(std::abs(subtle.first), 4.0 * floor, 1e-24);
    }
}

} // namespace
