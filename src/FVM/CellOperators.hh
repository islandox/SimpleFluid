/**
 * @file CellOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cell-centered finite-volume gradient and divergence operators.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include "utils/export_helper.hh"

#include "geometry/MeshConcepts.hh"
#include "utils/export_helper.hh"

#include "FVM/CellGradientScheme.hh"
#include "FVM/GaussLinearGradientCache.hh"
#include "FVM/details/FieldStoredCellOperators.hh"
#include "FVM/details/OperatorDetails.hh"
#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/GeometryEpoch.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Mesh-bound least-squares weights for repeated cell gradients.
 *
 * The cache owns a shared reference to the mesh and precomputes two
 * reconstruction variants: an interior-neighbor-only stencil and a
 * boundary-aware stencil. Boundary condition types and values remain dynamic;
 * only topology, directions, normal distances, and least-squares weights are
 * cached. Fixed-topology geometry motion invalidates the cache until refresh()
 * rebuilds its numeric data; topology changes require a new cache.
 *
 * Cached data are read-only between refreshes, so one cache may be read by
 * concurrent evaluations provided refresh() is not running and their fields
 * and callbacks are independent.
 *
 * @tparam Pack Tpetra type pack used by the mesh and fields.
 * @tparam MeshType Runtime or statically dispatched mesh interface.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType = MeshHandle<Pack>>
class SIMPLEFLUID_FVM_EXPORT CellGradientCache
{
public:
    using mesh_type = MeshType;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;
    using boundary_location_type =
        detail::BoundaryFaceLocation<mesh_type>;
    using traverse_plan_type = MeshTraversePlan<mesh_type>;

    /** @brief One cached interior-neighbor contribution. */
    struct InteriorSample
    {
        local_ordinal_type other_lid{};
        vec_type weight{};
    };

    /** @brief One cached boundary-face contribution. */
    struct BoundarySample
    {
        boundary_location_type location{};
        vec_type weight{};
        real_t normal_distance{};
    };

    /** @brief Face-ordered weights for one owned cell. */
    struct CellGeometry
    {
        std::vector<InteriorSample> interior_samples;
        std::vector<BoundarySample> boundary_samples;
    };

    /** Six regular contributions; boundary metadata lives in separate arrays. */
    struct StructuredInteriorRow
    {
        std::array<vec_type, 6> weights;
        std::array<local_ordinal_type, 6> neighbors;
    };

    struct StructuredBoundaryRow
    {
        size_t interior_begin{}, boundary_interior_begin{}, boundary_begin{};
        unsigned char interior_count{}, boundary_count{};
    };

    struct StructuredGeometry
    {
        explicit StructuredGeometry(::SimpleFluid::detail::StructuredTraverseLayout source) : layout(std::move(source)) {}
        ::SimpleFluid::detail::StructuredTraverseLayout layout;
        // Low bit marks a boundary row; remaining bits index the matching array.
        std::vector<size_t> rows;
        std::vector<StructuredInteriorRow> interior_rows;
        std::vector<StructuredBoundaryRow> boundary_rows;
        std::vector<InteriorSample> boundary_interior_only, boundary_interior;
        std::vector<BoundarySample> boundary_samples;

        struct InteriorRange
        {
            const StructuredInteriorRow* row;
            struct Iterator
            {
                const StructuredInteriorRow* row;
                size_t slot;
                InteriorSample operator*() const { return {row->neighbors[slot], row->weights[slot]}; }
                Iterator& operator++() { ++slot; return *this; }
                bool operator==(const Iterator&) const = default;
            };
            Iterator begin() const { return {row, 0}; }
            Iterator end() const { return {row, 6}; }
            size_t size() const noexcept { return 6; }
        };
        struct InteriorView
        {
            InteriorRange interior_samples;
            std::span<const BoundarySample> boundary_samples;
        };
        struct BoundaryView
        {
            std::span<const InteriorSample> interior_samples;
            std::span<const BoundarySample> boundary_samples;
        };

        template<bool IncludeBoundary, class Visitor>
        void visit_rows(Visitor&& visitor) const
        {
            for (size_t cell = 0; cell < rows.size(); ++cell)
            {
                const auto encoded = rows[cell];
                if (!(encoded & size_t{1}))
                    visitor(cell, InteriorView{{&interior_rows[encoded >> 1]}, {}});
                else
                {
                    const auto& row = boundary_rows[encoded >> 1];
                    if constexpr (IncludeBoundary)
                    {
                        if (row.boundary_count)
                            visitor(cell, BoundaryView{
                                std::span<const InteriorSample>(boundary_interior).subspan(
                                    row.boundary_interior_begin, row.interior_count),
                                std::span<const BoundarySample>(boundary_samples).subspan(
                                    row.boundary_begin, row.boundary_count)});
                        else
                            visitor(cell, BoundaryView{
                                std::span<const InteriorSample>(boundary_interior_only).subspan(
                                    row.interior_begin, row.interior_count), {}});
                    }
                    else
                        visitor(cell, BoundaryView{
                            std::span<const InteriorSample>(boundary_interior_only).subspan(
                                row.interior_begin, row.interior_count), {}});
                }
            }
        }

        size_t storage_bytes() const noexcept
        {
            return sizeof(*this) + rows.capacity() * sizeof(size_t)
                + interior_rows.capacity() * sizeof(StructuredInteriorRow)
                + boundary_rows.capacity() * sizeof(StructuredBoundaryRow)
                + (boundary_interior_only.capacity() + boundary_interior.capacity()) * sizeof(InteriorSample)
                + boundary_samples.capacity() * sizeof(BoundarySample);
        }
    };

    /** Variable-degree flat numerical rows for polygonal XY/Z prisms. */
    struct LayeredGeometry
    {
        std::shared_ptr<const ::SimpleFluid::detail::LayeredTraverseLayout> layout;
        struct Row
        {
            size_t interior_begin{}, interior_count{};
            size_t boundary_interior_begin{}, boundary_begin{}, boundary_count{};
        };
        std::vector<Row> rows;
        std::vector<InteriorSample> interior, boundary_interior;
        std::vector<BoundarySample> boundary;
        struct View
        {
            std::span<const InteriorSample> interior_samples;
            std::span<const BoundarySample> boundary_samples;
        };
        template<bool IncludeBoundary, class Visitor>
        void visit_rows(Visitor&& visitor) const
        {
            for (size_t cell = 0; cell < rows.size(); ++cell)
            {
                const auto& row = rows[cell];
                if constexpr (IncludeBoundary)
                    if (row.boundary_count)
                    {
                        visitor(cell, View{
                            std::span<const InteriorSample>(boundary_interior).subspan(row.boundary_interior_begin, row.interior_count),
                            std::span<const BoundarySample>(boundary).subspan(row.boundary_begin, row.boundary_count)});
                        continue;
                    }
                visitor(cell, View{std::span<const InteriorSample>(interior).subspan(row.interior_begin, row.interior_count), {}});
            }
        }
        size_t storage_bytes() const noexcept
        {
            return sizeof(*this) + rows.capacity() * sizeof(Row)
                + (interior.capacity() + boundary_interior.capacity()) * sizeof(InteriorSample)
                + boundary.capacity() * sizeof(BoundarySample);
        }
    };

    /**
     * @brief Precompute least-squares weights for an assembled mesh.
     * @param mesh Shared mesh whose lifetime and geometry the cache retains.
     * @throws std::invalid_argument if @p mesh is null.
     */
    using boundary_direction_provider_type = std::function<vec_type(local_ordinal_type, local_ordinal_type)>;

    explicit CellGradientCache(SP<const mesh_type> mesh);

    /** Optional displacement override for mixed vector boundary constraints. */
    CellGradientCache(SP<const mesh_type> mesh, boundary_direction_provider_type boundary_direction);

    /** Rebuild numeric weights while sharing the same fixed-topology traversal. */
    CellGradientCache(SP<const mesh_type> mesh, boundary_direction_provider_type boundary_direction,
        std::shared_ptr<const traverse_plan_type> traverse_plan);

    CellGradientCache(const CellGradientCache&) = delete;
    CellGradientCache& operator=(const CellGradientCache&) = delete;
    CellGradientCache(CellGradientCache&&) = default;
    CellGradientCache& operator=(CellGradientCache&&) = default;

    /** @brief Return the mesh retained by this cache. */
    const SP<const mesh_type>& mesh_ptr() const noexcept
    {
        return d_mesh;
    }

    /**
     * @brief Throw unless @p mesh is the exact cached mesh instance.
     */
    void require_mesh(const mesh_type& mesh) const
    {
        if (&mesh != d_mesh.get())
        {
            throw std::invalid_argument(
                "Cell-gradient cache belongs to another mesh.");
        }
        d_traverse_plan->require_valid();
        if (mesh_geometry_epoch(mesh) != d_geometry_epoch)
        {
            throw std::invalid_argument("Cell-gradient cache is stale for the mesh geometry epoch.");
        }
    }

    /** @brief Rebuild geometry-dependent weights at the current mesh epoch. */
    void refresh();

    /** @brief Geometry epoch represented by this cache. */
    std::uint64_t geometry_epoch() const noexcept { return d_geometry_epoch; }

    const traverse_plan_type& traverse_plan() const noexcept { return *d_traverse_plan; }
    const std::shared_ptr<const traverse_plan_type>& shared_traverse_plan() const noexcept
    { return d_traverse_plan; }

    /** @brief Return interior-neighbor-only reconstruction weights. */
    const std::vector<CellGeometry>& interior_geometry() const
    {
        require_mesh(*d_mesh);
        materialize_geometry();
        return d_geometry->interior;
    }

    /** @brief Return boundary-aware reconstruction weights. */
    const std::vector<CellGeometry>& boundary_geometry() const
    {
        require_mesh(*d_mesh);
        materialize_geometry();
        return d_geometry->boundary;
    }

    /** @brief Return the cached per-face boundary lookup. */
    const std::vector<boundary_location_type>& boundary_locations() const
    {
        require_mesh(*d_mesh);
        materialize_locations();
        return *d_geometry->locations;
    }

    bool uses_structured_storage() const noexcept { return d_geometry && bool(d_geometry->structured); }
    bool uses_layered_storage() const noexcept { return d_geometry && bool(d_geometry->layered); }
    const LayeredGeometry* layered_geometry() const
    { require_mesh(*d_mesh); return d_geometry->layered.get(); }
    bool legacy_geometry_materialized() const noexcept
    { return !(uses_structured_storage() || uses_layered_storage()) || d_geometry->legacy_ready.load(std::memory_order_acquire); }
    const StructuredGeometry* structured_geometry() const
    { require_mesh(*d_mesh); return d_geometry->structured.get(); }
    /** Numerical capacities; lazily requested compatibility arrays are excluded. */
    size_t numerical_storage_bytes() const noexcept;

private:
    struct GeometrySnapshot
    {
        std::unique_ptr<StructuredGeometry> structured;
        std::unique_ptr<LayeredGeometry> layered;
        mutable std::vector<CellGeometry> interior, boundary;
        mutable std::shared_ptr<const std::vector<boundary_location_type>> locations;
        mutable std::once_flag legacy_once, locations_once;
        mutable std::atomic<bool> legacy_ready{false};
    };
    void materialize_geometry() const;
    void materialize_locations() const;
    std::shared_ptr<GeometrySnapshot> build_snapshot() const;
    void build_region_geometry(GeometrySnapshot& next) const;
    void build_layered_geometry(GeometrySnapshot& next) const;
    void build_structured_geometry(GeometrySnapshot& next,
        const std::vector<boundary_location_type>* locations) const;
    static std::shared_ptr<const traverse_plan_type> require_traverse_plan(const mesh_type& mesh,
        std::shared_ptr<const traverse_plan_type> plan)
    {
        if (!plan) return std::make_shared<const traverse_plan_type>(mesh, MeshTraverseEntities::AllWithCellFaces);
        if (&plan->mesh() != &mesh || plan->entities() != MeshTraverseEntities::AllWithCellFaces)
            throw std::invalid_argument("Cell-gradient cache requires cell-face traversal for the same mesh.");
        plan->require_valid();
        return plan;
    }

    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "CellGradientCache requires a non-null mesh.");
        }
        return mesh;
    }

    SIMPLEFLUID_FVM_LOCAL
    static std::array<vec_type, 3> inverse_columns(
        const std::array<std::array<real_t, 3>, 3>& normal);

    static vec_type apply_inverse(
        const std::array<vec_type, 3>& inverse,
        const vec_type& direction)
    {
        return inverse[0] * direction.x
             + inverse[1] * direction.y
             + inverse[2] * direction.z;
    }

    SIMPLEFLUID_FVM_LOCAL
    static std::vector<CellGeometry> build_geometry(
        const mesh_type& mesh,
        const std::vector<boundary_location_type>& boundary_locations,
        const traverse_plan_type& plan, bool include_boundary_samples,
        const boundary_direction_provider_type& boundary_direction);

    SP<const mesh_type> d_mesh;
    std::shared_ptr<const traverse_plan_type> d_traverse_plan;
    boundary_direction_provider_type d_boundary_direction;
    std::shared_ptr<GeometrySnapshot> d_geometry;
    std::uint64_t d_geometry_epoch = 0;
};
extern template class CellGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

/** @brief Reconstruct a scalar Gauss gradient with retained geometry and live boundaries. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType, class BoundaryConditionProvider,
    class BoundaryValueProvider>
void gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients, const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    detail::cached_gauss_linear_scalar_gradient<Pack>(
        field, boundary_condition, boundary_value, gradients, cache);
}

/** @brief Reconstruct a cached Gauss gradient using a boundary map. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
void gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions, VectorCellFieldStored<Pack, MeshType>& gradients,
    const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto& name = field.mesh().boundary_batch_name(batch_id);
        const auto iter = boundary_conditions.find(name);
        return iter == boundary_conditions.end() ? BoundaryCondition{} : iter->second;
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        return static_cast<typename Pack::scalar_type>(boundary_condition(batch_id, in_batch_id).value);
    };
    gauss_linear_cell_gradient(field, boundary_condition, boundary_value, gradients, cache);
}

/** @brief Reconstruct a cached Gauss gradient with zero-normal walls. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
void gauss_linear_cell_gradient(const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients, const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    gauss_linear_cell_gradient(field, [](int, size_t) { return BoundaryCondition{}; },
        [](int, size_t) { return typename Pack::scalar_type{}; }, gradients, cache);
}

/** @brief Reconstruct a vector Gauss gradient with retained geometry and live boundaries. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType, class BoundaryValueProvider>
void gauss_linear_cell_gradient(const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value, TensorCellFieldStored<Pack, MeshType>& gradients,
    const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    detail::cached_gauss_linear_vector_gradient<Pack>(field, boundary_value, gradients, cache);
}

/**
 * @brief Reconstruct an interior least-squares gradient in mesh-aware stored
 *        fields.
 *
 * This overload covers `MeshHandle` fields backed by orthogonal or
 * semi-structured meshes as well as statically typed partitioned meshes.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Reconstruct a stored scalar gradient from cached mesh geometry. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache);

/** @brief Reconstruct a boundary-aware stored scalar gradient. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Reconstruct a cached boundary-aware stored scalar gradient. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache);

/** @brief Reconstruct a stored scalar gradient from dynamic boundary data. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
    requires requires(BoundaryConditionProvider condition,
                      BoundaryValueProvider value,
                      int batch_id,
                      size_t in_batch_id)
    {
        { condition(batch_id, in_batch_id) }
            -> std::convertible_to<BoundaryCondition>;
        { value(batch_id, in_batch_id) }
            -> std::convertible_to<typename Pack::scalar_type>;
    }
void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Reconstruct a cached stored scalar gradient from boundary data. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
    requires requires(BoundaryConditionProvider condition,
                      BoundaryValueProvider value,
                      int batch_id,
                      size_t in_batch_id)
    {
        { condition(batch_id, in_batch_id) }
            -> std::convertible_to<BoundaryCondition>;
        { value(batch_id, in_batch_id) }
            -> std::convertible_to<typename Pack::scalar_type>;
    }
void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache);

/** @brief Reconstruct component gradients into a stored tensor field. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Reconstruct cached component gradients into tensor storage. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field,
    TensorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache);

/** @brief Reconstruct boundary-aware vector gradients into tensor storage. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryValueProvider>
    requires requires(BoundaryValueProvider provider,
                      int batch_id,
                      size_t in_batch_id)
    {
        { provider(batch_id, in_batch_id) }
            -> std::convertible_to<
                typename VectorCellFieldStored<Pack, MeshType>::value_type>;
    }
void cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value,
    TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Reconstruct cached boundary-aware vector gradients. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryValueProvider>
    requires requires(BoundaryValueProvider provider,
                      int batch_id,
                      size_t in_batch_id)
    {
        { provider(batch_id, in_batch_id) }
            -> std::convertible_to<
                typename VectorCellFieldStored<Pack, MeshType>::value_type>;
    }
void cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value,
    TensorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache);

/** @brief Compute a Gauss-linear stored scalar gradient. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
void gauss_linear_cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Compute a Gauss-linear stored scalar gradient from a boundary map. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void gauss_linear_cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Compute a zero-normal-boundary Gauss-linear scalar gradient. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void gauss_linear_cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    VectorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Compute a Gauss-linear stored vector gradient. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryValueProvider>
void gauss_linear_cell_gradient(
    const VectorCellFieldStored<Pack, MeshType>& field,
    BoundaryValueProvider boundary_value,
    TensorCellFieldStored<Pack, MeshType>& gradients);

/** @brief Select the stored scalar gradient reconstruction scheme. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT void cell_gradient(
    const ScalarCellFieldStored<Pack, MeshType>& field,
    const BoundaryConditionMap& boundary_conditions,
    VectorCellFieldStored<Pack, MeshType>& gradients,
    const CellGradientCache<Pack, MeshType>& cache,
    CellGradientScheme scheme);

/** @brief Compute one stored face field's net flux around a cell. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
auto cell_flux_balance(
    const MeshType& mesh,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid)
    -> typename Pack::scalar_type
{
    if (&mesh != &face_fluxes.mesh())
    {
        throw std::invalid_argument(
            "cell_flux_balance requires the face field mesh.");
    }
    return detail::stored_cell_flux_balance(
        mesh, face_fluxes, cell_lid);
}

/**
 * @brief Compute a stored balance from a cached overlap or owned face view.
 * An owned-only view uses the field's published overlap values for remote faces.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType, class View>
auto cell_flux_balance(
    const MeshType& mesh,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    const View& face_values,
    typename Pack::local_ordinal_type cell_lid)
    -> typename Pack::scalar_type
{
    if (&mesh != &face_fluxes.mesh())
        throw std::invalid_argument("cell_flux_balance requires the face field mesh.");
    return detail::stored_cell_flux_balance(mesh, face_fluxes, face_values, cell_lid);
}

/** @brief Select mesh-local flux storage for the backend's balance convention. */
template<class FaceField>
auto face_flux_balance_read_view(const FaceField& field)
{
    if constexpr (requires { field.local_read_view(); })
        return field.local_read_view();
    else
        return field.owned_read_view();
}

/**
 * @brief Immutable ordered face/sign slots for repeated owned-cell balances.
 * Geometry motion leaves these topology-only slots valid; topology replacement
 * requires a new cache. Views are scoped by callers and never retained here.
 */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
class CellFluxBalanceCache
{
public:
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    template<class FaceField>
    explicit CellFluxBalanceCache(const FaceField& field)
    {
        const auto& mesh = field.mesh();
        d_offsets.reserve(mesh.num_owned_cells() + 1);
        d_offsets.push_back(0);
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto cell_id = detail::query_cell_id(mesh, cell_lid);
            for (const auto face_id : mesh.faces(cell_id))
            {
                const auto face_lid = static_cast<local_ordinal_type>(detail::packed_face_local_id(mesh, face_id));
                const auto owner = detail::packed_cell_local_id(mesh, mesh.owner_cell(face_id));
                d_faces.push_back({face_lid, owner == cell_lid ? scalar_type{1} : scalar_type{-1}});
            }
            d_offsets.push_back(d_faces.size());
        }
    }

    template<class View>
    scalar_type balance(const View& mesh_local_values, local_ordinal_type cell_lid) const
    {
        const auto row = static_cast<size_t>(cell_lid);
        if (row >= d_offsets.size() - 1)
            throw std::out_of_range("Cell flux-balance cache requires an owned cell.");
        scalar_type result{};
        for (auto slot = d_offsets[row]; slot < d_offsets[row + 1]; ++slot)
        {
            const auto& face = d_faces[slot];
            result += face.sign * mesh_local_values(face.row, 0);
        }
        return result;
    }

private:
    struct FaceSlot { local_ordinal_type row; scalar_type sign; };
    std::vector<size_t> d_offsets;
    std::vector<FaceSlot> d_faces;
};

/** @brief Compute volume-normalized divergence from stored face fluxes. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_FVM_EXPORT std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes(
    const MeshType& mesh,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes);

} // namespace SimpleFluid::FVM

#include "FVM/FieldStoredCellOperators.ipp"
