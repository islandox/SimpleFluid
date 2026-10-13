/**
 * @file GaussLinearGradientCache.hh
 * @brief  Reusable face-ordered Gauss geometry.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/details/OperatorDetails.hh"
#include "geometry/GeometryEpoch.hh"
#include "geometry/Mesh.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshTraverse.hh"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Immutable-between-refreshes geometry for Gauss-linear reconstruction.
 *
 * Only mesh data are retained: boundary types and values are evaluated on each
 * application. Face order and interpolation arithmetic match the uncached
 * operator. Motion invalidates the cache until refresh(); topology changes
 * require a new cache. The ordered traversal plan is retained across geometry
 * refreshes and cache copies. Independent field evaluations may share a const
 * cache.
 */
template<TpetraTypePack Pack, class MeshType = Mesh<Pack>>
class SIMPLEFLUID_FVM_EXPORT GaussLinearGradientCache
{
public:
    using mesh_type = MeshType;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using boundary_location_type = detail::BoundaryFaceLocation<mesh_type>;
    using traverse_plan_type = MeshTraversePlan<mesh_type>;

    struct FaceGeometry
    {
        local_ordinal_type face_lid{};
        local_ordinal_type other_lid{};
        typename mesh_type::Vec3 area{};
        real_t cell_distance{};
        real_t other_distance{};
        real_t normal_distance{};
        boundary_location_type boundary{};
        bool interior = false;
    };

    struct Geometry
    {
        std::vector<size_t> offsets;
        std::vector<real_t> volumes;
        std::vector<FaceGeometry> faces;
    };

    /** Six native-ordered interior faces, without per-face boundary metadata. */
    struct StructuredInteriorRow
    {
        std::array<typename mesh_type::Vec3, 6> areas;
        std::array<real_t, 6> cell_distances;
        std::array<real_t, 6> other_distances;
        std::array<local_ordinal_type, 6> other_lids;
    };

    struct StructuredGeometry
    {
        std::optional<SimpleFluid::detail::StructuredTraverseLayout> layout;
        std::vector<real_t> volumes;
        // Low bit selects boundary rows; the remaining bits are the row index.
        std::vector<size_t> rows;
        std::vector<StructuredInteriorRow> interior;
        std::vector<std::array<FaceGeometry, 6>> boundary;
    };

    /** Cache-owned immutable topology shared by geometry snapshots and copies. */
    struct LayeredTopology
    {
        struct BoundaryFace
        {
            local_ordinal_type face_lid{}, other_lid{};
            boundary_location_type boundary{};
            bool interior = false;
        };
        std::shared_ptr<const traverse_plan_type> plan;
        std::vector<size_t> rows, interior_offsets, boundary_offsets;
        std::vector<local_ordinal_type> other_lids;
        std::vector<BoundaryFace> boundary;
        size_t storage_bytes() const noexcept
        {
            return sizeof(*this) + (rows.capacity() + interior_offsets.capacity()
                + boundary_offsets.capacity()) * sizeof(size_t)
                + other_lids.capacity() * sizeof(local_ordinal_type)
                + boundary.capacity() * sizeof(BoundaryFace);
        }
    };

    /** Variable-degree metric payload; topology is retained across refresh. */
    struct LayeredGeometry
    {
        struct BoundaryMetric
        {
            typename mesh_type::Vec3 area{};
            real_t cell_distance{}, other_distance{}, normal_distance{};
        };
        std::shared_ptr<const LayeredTopology> topology;
        std::vector<real_t> volumes;
        std::vector<typename mesh_type::Vec3> areas;
        std::vector<real_t> cell_distances, other_distances;
        std::vector<BoundaryMetric> boundary;
    };

    explicit GaussLinearGradientCache(SP<const mesh_type> mesh);

    const SP<const mesh_type>& mesh_ptr() const noexcept { return d_mesh; }
    std::uint64_t geometry_epoch() const noexcept { return d_geometry_epoch; }
    /** @brief Immutable topology schedule shared by refreshes and cache copies. */
    const traverse_plan_type& traverse_plan() const noexcept { return *d_traverse_plan; }

    void require_mesh(const mesh_type& mesh) const;

    /** Ordered compatibility view, materialized only when explicitly requested. */
    const Geometry& geometry() const;

    bool uses_layered_storage() const noexcept { return bool(d_snapshot->layered); }
    const LayeredGeometry* layered_geometry() const noexcept { return d_snapshot->layered.get(); }

    bool uses_structured_storage() const noexcept { return bool(d_snapshot->structured); }
    bool legacy_geometry_materialized() const noexcept
    { return d_snapshot->legacy_materialized.load(std::memory_order_acquire); }
    /** Native numerical allocation estimate, excluding a requested compatibility view. */
    size_t numerical_storage_bytes() const noexcept;
    /** Caller must validate the mesh before evaluating this immutable snapshot. */
    const StructuredGeometry* structured_geometry() const noexcept
    { return d_snapshot->structured.get(); }

    /** @brief Replace all cached geometry after a successful rebuild. */
    void refresh();

private:
    std::shared_ptr<const LayeredTopology> build_layered_topology(
        const MeshTraverseExecution<mesh_type>& traversal) const;
    void fill_layered_geometry(LayeredGeometry& geometry) const;

    struct Snapshot
    {
        std::unique_ptr<StructuredGeometry> structured;
        std::unique_ptr<LayeredGeometry> layered;
        mutable Geometry geometry;
        mutable std::once_flag legacy_once;
        mutable std::atomic<bool> legacy_materialized{false};
    };

    SP<const mesh_type> d_mesh;
    std::shared_ptr<const traverse_plan_type> d_traverse_plan;
    std::vector<boundary_location_type> d_boundary_locations;
    std::shared_ptr<Snapshot> d_snapshot;
    std::uint64_t d_geometry_epoch = 0;
};

extern template class GaussLinearGradientCache<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class GaussLinearGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

namespace detail
{

/** @brief Shared scalar kernel for stored and legacy fields. */
template<TpetraTypePack Pack, class ScalarField, class BoundaryConditionProvider,
    class BoundaryValueProvider, class VectorField, class MeshType>
void cached_gauss_linear_scalar_gradient(const ScalarField& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    VectorField& gradients, const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (&field.mesh() != &gradients.mesh())
        throw std::invalid_argument("gauss_linear_cell_gradient requires input and output fields on one mesh.");
    const auto execution = acquire_mesh_execution(field.mesh());
    cache.require_mesh(field.mesh());
    if (const auto* structured = cache.structured_geometry())
    {
        const auto local_values = field.local_read_view();
        auto gradient_values = gradients.owned_write_view();
        for (size_t owned = 0; owned < structured->volumes.size(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto cell_value = local_values(cell_lid, 0);
            vec3<scalar_type> gradient{};
            const auto row = structured->rows[owned];
            if ((row & 1) == 0)
            {
                const auto& faces = structured->interior[row >> 1];
                for (size_t face = 0; face < 6; ++face)
                {
                    const auto total_distance = faces.cell_distances[face] + faces.other_distances[face];
                    const auto face_value = total_distance > scalar_type{}
                        ? (faces.other_distances[face] * cell_value
                            + faces.cell_distances[face] * local_values(faces.other_lids[face], 0)) / total_distance
                        : scalar_type{0.5} * (cell_value + local_values(faces.other_lids[face], 0));
                    gradient = gradient + faces.areas[face] * face_value;
                }
            }
            else
                for (const auto& face : structured->boundary[row >> 1])
                {
                    scalar_type face_value = cell_value;
                    if (face.interior)
                    {
                        const auto total_distance = face.cell_distance + face.other_distance;
                        face_value = total_distance > scalar_type{}
                            ? (face.other_distance * cell_value
                                + face.cell_distance * local_values(face.other_lid, 0)) / total_distance
                            : scalar_type{0.5} * (cell_value + local_values(face.other_lid, 0));
                    }
                    else if (face.boundary.active)
                    {
                        const auto& location = face.boundary;
                        const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
                        if (condition.type == BoundaryConditionType::Dirichlet)
                            face_value = boundary_value(location.batch_id, location.in_batch_id);
                        else if (condition.type == BoundaryConditionType::Neumann)
                            face_value = cell_value + static_cast<scalar_type>(condition.value)
                                * static_cast<scalar_type>(face.normal_distance);
                        else
                            throw std::invalid_argument("Gauss-linear scalar gradients support only "
                                                        "Dirichlet and Neumann boundaries.");
                    }
                    gradient = gradient + face.area * face_value;
                }
            gradient = gradient / static_cast<scalar_type>(structured->volumes[owned]);
            gradient_values(cell_lid, 0) = gradient.x;
            gradient_values(cell_lid, 1) = gradient.y;
            gradient_values(cell_lid, 2) = gradient.z;
        }
        return;
    }
    if (const auto* layered = cache.layered_geometry())
    {
        const auto& topology = *layered->topology;
        const auto local_values = field.local_read_view();
        auto gradient_values = gradients.owned_write_view();
        for (size_t owned = 0; owned < layered->volumes.size(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto cell_value = local_values(cell_lid, 0);
            vec3<scalar_type> gradient{};
            const auto row = topology.rows[owned];
            if ((row & 1) == 0)
            {
                const auto& faces = *layered;
                for (size_t face = topology.interior_offsets[row >> 1];
                     face < topology.interior_offsets[(row >> 1) + 1]; ++face)
                {
                    const auto total_distance = faces.cell_distances[face] + faces.other_distances[face];
                    const auto face_value = total_distance > scalar_type{}
                        ? (faces.other_distances[face] * cell_value
                            + faces.cell_distances[face] * local_values(topology.other_lids[face], 0)) / total_distance
                        : scalar_type{0.5} * (cell_value + local_values(topology.other_lids[face], 0));
                    gradient = gradient + faces.areas[face] * face_value;
                }
            }
            else
                for (size_t entry = topology.boundary_offsets[row >> 1];
                     entry < topology.boundary_offsets[(row >> 1) + 1]; ++entry)
                {
                    const auto& face = layered->boundary[entry];
                    const auto& metadata = topology.boundary[entry];
                    scalar_type face_value = cell_value;
                    if (metadata.interior)
                    {
                        const auto total_distance = face.cell_distance + face.other_distance;
                        face_value = total_distance > scalar_type{}
                            ? (face.other_distance * cell_value
                                + face.cell_distance * local_values(metadata.other_lid, 0)) / total_distance
                            : scalar_type{0.5} * (cell_value + local_values(metadata.other_lid, 0));
                    }
                    else if (metadata.boundary.active)
                    {
                        const auto& location = metadata.boundary;
                        const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
                        if (condition.type == BoundaryConditionType::Dirichlet)
                            face_value = boundary_value(location.batch_id, location.in_batch_id);
                        else if (condition.type == BoundaryConditionType::Neumann)
                            face_value = cell_value + static_cast<scalar_type>(condition.value)
                                * static_cast<scalar_type>(face.normal_distance);
                        else
                            throw std::invalid_argument("Gauss-linear scalar gradients support only "
                                                        "Dirichlet and Neumann boundaries.");
                    }
                    gradient = gradient + face.area * face_value;
                }
            gradient = gradient / static_cast<scalar_type>(layered->volumes[owned]);
            gradient_values(cell_lid, 0) = gradient.x;
            gradient_values(cell_lid, 1) = gradient.y;
            gradient_values(cell_lid, 2) = gradient.z;
        }
        return;
    }
    const auto& geometry = cache.geometry();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < geometry.volumes.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto cell_value = local_values(cell_lid, 0);
        vec3<scalar_type> gradient{};
        for (size_t entry = geometry.offsets[owned]; entry < geometry.offsets[owned + 1]; ++entry)
        {
            const auto& face = geometry.faces[entry];
            scalar_type face_value = cell_value;
            if (face.interior)
            {
                const auto total_distance = face.cell_distance + face.other_distance;
                face_value = total_distance > scalar_type{}
                    ? (face.other_distance * cell_value + face.cell_distance * local_values(face.other_lid, 0)) /
                          total_distance
                    : scalar_type{0.5} * (cell_value + local_values(face.other_lid, 0));
            }
            else if (face.boundary.active)
            {
                const auto& location = face.boundary;
                const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
                if (condition.type == BoundaryConditionType::Dirichlet)
                    face_value = boundary_value(location.batch_id, location.in_batch_id);
                else if (condition.type == BoundaryConditionType::Neumann)
                    face_value = cell_value + static_cast<scalar_type>(condition.value) *
                                                 static_cast<scalar_type>(face.normal_distance);
                else
                    throw std::invalid_argument("Gauss-linear scalar gradients support only "
                                                "Dirichlet and Neumann boundaries.");
            }
            gradient = gradient + face.area * face_value;
        }
        gradient = gradient / static_cast<scalar_type>(geometry.volumes[owned]);
        gradient_values(cell_lid, 0) = gradient.x;
        gradient_values(cell_lid, 1) = gradient.y;
        gradient_values(cell_lid, 2) = gradient.z;
    }
}

/** @brief Shared vector kernel with the original component and face order. */
template<TpetraTypePack Pack, class VectorField, class BoundaryValueProvider, class TensorField, class MeshType>
void cached_gauss_linear_vector_gradient(const VectorField& field,
    BoundaryValueProvider boundary_value, TensorField& gradients,
    const GaussLinearGradientCache<Pack, MeshType>& cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<scalar_type>;
    if (&field.mesh() != &gradients.mesh())
        throw std::invalid_argument("gauss_linear_cell_gradient requires input and output fields on one mesh.");
    const auto execution = acquire_mesh_execution(field.mesh());
    cache.require_mesh(field.mesh());
    if (const auto* structured = cache.structured_geometry())
    {
        const auto local_values = field.local_read_view();
        auto gradient_values = gradients.owned_write_view();
        for (size_t owned = 0; owned < structured->volumes.size(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const vec_type cell_value{local_values(cell_lid, 0), local_values(cell_lid, 1), local_values(cell_lid, 2)};
            std::array<vec_type, 3> gradient{};
            const auto row = structured->rows[owned];
            if ((row & 1) == 0)
            {
                const auto& faces = structured->interior[row >> 1];
                for (size_t face = 0; face < 6; ++face)
                {
                    const auto other = faces.other_lids[face];
                    const vec_type other_value{local_values(other, 0), local_values(other, 1), local_values(other, 2)};
                    const auto total_distance = faces.cell_distances[face] + faces.other_distances[face];
                    const auto face_value = total_distance > scalar_type{}
                        ? (cell_value * faces.other_distances[face] + other_value * faces.cell_distances[face]) / total_distance
                        : (cell_value + other_value) / scalar_type{2};
                    for (size_t component = 0; component < 3; ++component)
                        gradient[component] = gradient[component] + faces.areas[face] * face_value.component(component);
                }
            }
            else
                for (const auto& face : structured->boundary[row >> 1])
                {
                    auto face_value = cell_value;
                    if (face.interior)
                    {
                        const vec_type other_value{local_values(face.other_lid, 0),
                            local_values(face.other_lid, 1), local_values(face.other_lid, 2)};
                        const auto total_distance = face.cell_distance + face.other_distance;
                        face_value = total_distance > scalar_type{}
                            ? (cell_value * face.other_distance + other_value * face.cell_distance) / total_distance
                            : (cell_value + other_value) / scalar_type{2};
                    }
                    else if (face.boundary.active)
                        face_value = boundary_value(face.boundary.batch_id, face.boundary.in_batch_id);
                    for (size_t component = 0; component < 3; ++component)
                        gradient[component] = gradient[component] + face.area * face_value.component(component);
                }
            const auto inverse_volume = scalar_type{1} / static_cast<scalar_type>(structured->volumes[owned]);
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] * inverse_volume;
                gradient_values(cell_lid, component * 3) = gradient[component].x;
                gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
                gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
            }
        }
        return;
    }
    if (const auto* layered = cache.layered_geometry())
    {
        const auto& topology = *layered->topology;
        const auto local_values = field.local_read_view();
        auto gradient_values = gradients.owned_write_view();
        for (size_t owned = 0; owned < layered->volumes.size(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const vec_type cell_value{local_values(cell_lid, 0), local_values(cell_lid, 1), local_values(cell_lid, 2)};
            std::array<vec_type, 3> gradient{};
            const auto row = topology.rows[owned];
            if ((row & 1) == 0)
            {
                const auto& faces = *layered;
                for (size_t face = topology.interior_offsets[row >> 1];
                     face < topology.interior_offsets[(row >> 1) + 1]; ++face)
                {
                    const auto other = topology.other_lids[face];
                    const vec_type other_value{local_values(other, 0), local_values(other, 1), local_values(other, 2)};
                    const auto total_distance = faces.cell_distances[face] + faces.other_distances[face];
                    const auto face_value = total_distance > scalar_type{}
                        ? (cell_value * faces.other_distances[face] + other_value * faces.cell_distances[face]) / total_distance
                        : (cell_value + other_value) / scalar_type{2};
                    for (size_t component = 0; component < 3; ++component)
                        gradient[component] = gradient[component] + faces.areas[face] * face_value.component(component);
                }
            }
            else
                for (size_t entry = topology.boundary_offsets[row >> 1];
                     entry < topology.boundary_offsets[(row >> 1) + 1]; ++entry)
                {
                    const auto& face = layered->boundary[entry];
                    const auto& metadata = topology.boundary[entry];
                    auto face_value = cell_value;
                    if (metadata.interior)
                    {
                        const vec_type other_value{local_values(metadata.other_lid, 0),
                            local_values(metadata.other_lid, 1), local_values(metadata.other_lid, 2)};
                        const auto total_distance = face.cell_distance + face.other_distance;
                        face_value = total_distance > scalar_type{}
                            ? (cell_value * face.other_distance + other_value * face.cell_distance) / total_distance
                            : (cell_value + other_value) / scalar_type{2};
                    }
                    else if (metadata.boundary.active)
                        face_value = boundary_value(metadata.boundary.batch_id, metadata.boundary.in_batch_id);
                    for (size_t component = 0; component < 3; ++component)
                        gradient[component] = gradient[component] + face.area * face_value.component(component);
                }
            const auto inverse_volume = scalar_type{1} / static_cast<scalar_type>(layered->volumes[owned]);
            for (size_t component = 0; component < 3; ++component)
            {
                gradient[component] = gradient[component] * inverse_volume;
                gradient_values(cell_lid, component * 3) = gradient[component].x;
                gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
                gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
            }
        }
        return;
    }
    const auto& geometry = cache.geometry();
    const auto local_values = field.local_read_view();
    auto gradient_values = gradients.owned_write_view();
    for (size_t owned = 0; owned < geometry.volumes.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const vec_type cell_value{local_values(cell_lid, 0), local_values(cell_lid, 1), local_values(cell_lid, 2)};
        std::array<vec_type, 3> gradient{};
        for (size_t entry = geometry.offsets[owned]; entry < geometry.offsets[owned + 1]; ++entry)
        {
            const auto& face = geometry.faces[entry];
            auto face_value = cell_value;
            if (face.interior)
            {
                const vec_type other_value{local_values(face.other_lid, 0),
                    local_values(face.other_lid, 1), local_values(face.other_lid, 2)};
                const auto total_distance = face.cell_distance + face.other_distance;
                face_value = total_distance > scalar_type{}
                    ? (cell_value * face.other_distance + other_value * face.cell_distance) / total_distance
                    : (cell_value + other_value) / scalar_type{2};
            }
            else if (face.boundary.active)
                face_value = boundary_value(face.boundary.batch_id, face.boundary.in_batch_id);
            for (size_t component = 0; component < 3; ++component)
                gradient[component] = gradient[component] + face.area * face_value.component(component);
        }
        const auto inverse_volume = scalar_type{1} / static_cast<scalar_type>(geometry.volumes[owned]);
        for (size_t component = 0; component < 3; ++component)
        {
            gradient[component] = gradient[component] * inverse_volume;
            gradient_values(cell_lid, component * 3) = gradient[component].x;
            gradient_values(cell_lid, component * 3 + 1) = gradient[component].y;
            gradient_values(cell_lid, component * 3 + 2) = gradient[component].z;
        }
    }
}

} // namespace detail
} // namespace SimpleFluid::FVM
