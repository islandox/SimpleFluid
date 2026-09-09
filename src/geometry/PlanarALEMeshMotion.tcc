/**
 * @file PlanarALEMeshMotion.tcc
 * @brief Template implementation of planar fixed-topology mesh motion.
 */

#pragma once

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid
{
namespace planar_ale_detail
{

template<class Comm, class Value> bool collectively_equal(const Comm& communicator, const Value& value)
{
    Value minimum{};
    Value maximum{};
    Teuchos::reduceAll(communicator, Teuchos::REDUCE_MIN, 1, &value, &minimum);
    Teuchos::reduceAll(communicator, Teuchos::REDUCE_MAX, 1, &value, &maximum);
    return minimum == maximum;
}

template<class Comm> bool collectively_equal_values(const Comm& communicator, const std::vector<real_t>& values)
{
    if (values.empty())
    {
        return true;
    }
    std::vector<real_t> minimum(values.size());
    std::vector<real_t> maximum(values.size());
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MIN, static_cast<int>(values.size()), values.data(), minimum.data());
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MAX, static_cast<int>(values.size()), values.data(), maximum.data());
    return minimum == maximum;
}

inline bool finite_non_negative(real_t value) noexcept
{
    return std::isfinite(value) && value >= 0.0;
}

} // namespace planar_ale_detail

template<TpetraTypePack Pack>
PlanarALEMeshMotion<Pack>::PlanarALEMeshMotion(SP<mesh_type> mesh, PlanarALEMeshMotionOptions options)
    : d_mesh(std::move(mesh)), d_options(std::move(options))
{
    if (!d_mesh)
    {
        throw std::invalid_argument("PlanarALEMeshMotion requires a non-null mutable MeshHandle.");
    }

    d_family = detect_family();
    d_reference_geometry_edges = geometry_edge_coordinates();
    const auto axis = static_cast<int>(d_options.axis);
    const auto axis_is_valid = axis >= static_cast<int>(Dimension::X) && axis <= static_cast<int>(Dimension::Z);
    const auto family_axis_is_valid =
        d_family == Family::Cartesian ||
        ((d_family == Family::Cylindrical || d_family == Family::SemiStructured) && d_options.axis == Dimension::Z);
    if (d_family != Family::Unsupported && axis_is_valid && family_axis_is_valid)
    {
        d_reference_axis_edges = current_axis_edges();
    }
    validate_collective_construction();

    switch (d_family)
    {
        case Family::Cartesian:
            d_family_name = "OrthogonalCartesian3D";
            break;
        case Family::Cylindrical:
            d_family_name = "OrthogonalCylindrial3D";
            break;
        case Family::SemiStructured:
            d_family_name = "SemiStructuredXY_Z";
            break;
        case Family::Unsupported:
            throw std::logic_error("Planar ALE construction reached an unsupported mesh family.");
    }

    d_reference_surface_elevation = d_reference_axis_edges.back();
    d_accepted_surface_elevation = d_reference_surface_elevation;
    const auto deformation_start = d_options.deformation_start_elevation.value_or(d_reference_axis_edges.front());
    d_deformation_weights.reserve(d_reference_axis_edges.size());
    for (const auto coordinate : d_reference_axis_edges)
    {
        const auto weight = coordinate <= deformation_start ? real_t{}
                                                            : (coordinate - deformation_start) /
                                                                  (d_reference_surface_elevation - deformation_start);
        d_deformation_weights.push_back(std::clamp(weight, 0.0, 1.0));
    }

    d_expected_geometry_epoch = d_mesh->geometry_epoch();
    d_accepted_quality = evaluate_mesh_quality(*d_mesh);
    const MeshQualityGate gate(d_options.quality_limits);
    const auto assessment = gate.assess(d_accepted_quality);
    if (!assessment.accepted())
    {
        throw std::invalid_argument("PlanarALEMeshMotion initial " + assessment.report(d_accepted_quality));
    }
    reset_stationary_state();
    claim_geometry_motion();
    d_geometry_motion_claimed = true;
}

template<TpetraTypePack Pack> PlanarALEMeshMotion<Pack>::~PlanarALEMeshMotion()
{
    try
    {
        if (d_trial_active && geometry_motion_owned() && d_mesh->geometry_epoch() == d_expected_geometry_epoch &&
            geometry_edge_coordinates() == d_trial_geometry_edges)
        {
            rollback_impl();
        }
    }
    catch (...)
    {
    }
    release_geometry_motion();
}

template<TpetraTypePack Pack> auto PlanarALEMeshMotion<Pack>::detect_family() const noexcept -> Family
{
    return d_mesh->visit(
        []<class Mesh>(const Mesh&) noexcept
        {
            using mesh_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<mesh_type, typename PlanarALEMeshMotion::mesh_type::Cartesian>)
            {
                return Family::Cartesian;
            }
            else if constexpr (std::same_as<mesh_type, typename PlanarALEMeshMotion::mesh_type::Cylindrical>)
            {
                return Family::Cylindrical;
            }
            else if constexpr (std::same_as<mesh_type, typename PlanarALEMeshMotion::mesh_type::SemiStructured>)
            {
                return Family::SemiStructured;
            }
            else
            {
                return Family::Unsupported;
            }
        });
}

template<TpetraTypePack Pack> ArrReal PlanarALEMeshMotion<Pack>::current_axis_edges() const
{
    return d_mesh->visit(
        [this]<class Mesh>(const Mesh& mesh) -> ArrReal
        {
            using concrete_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian>)
            {
                const auto axis = static_cast<int>(d_options.axis);
                if (axis < static_cast<int>(Dimension::X) || axis > static_cast<int>(Dimension::Z))
                {
                    throw std::invalid_argument("PlanarALEMeshMotion Cartesian axis is invalid.");
                }
                return mesh.cell_edges()[static_cast<size_t>(axis)];
            }
            else if constexpr (std::same_as<concrete_type, typename mesh_type::Cylindrical>)
            {
                return mesh.cell_edges()[concrete_type::AXIAL];
            }
            else if constexpr (std::same_as<concrete_type, typename mesh_type::SemiStructured>)
            {
                return mesh.z_edges();
            }
            else
            {
                throw std::invalid_argument("PlanarALEMeshMotion does not support this mesh family.");
            }
        });
}

template<TpetraTypePack Pack> std::array<ArrReal, 3> PlanarALEMeshMotion<Pack>::geometry_edge_coordinates() const
{
    return d_mesh->visit(
        []<class Mesh>(const Mesh& mesh) -> std::array<ArrReal, 3>
        {
            using concrete_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian> ||
                          std::same_as<concrete_type, typename mesh_type::Cylindrical>)
            {
                return mesh.cell_edges();
            }
            else if constexpr (std::same_as<concrete_type, typename mesh_type::SemiStructured>)
            {
                std::array<ArrReal, 3> result;
                result[static_cast<size_t>(Dimension::Z)] = mesh.z_edges();
                return result;
            }
            else
            {
                return {};
            }
        });
}

template<TpetraTypePack Pack> bool PlanarALEMeshMotion<Pack>::geometry_motion_available() const noexcept
{
    return d_mesh->visit(
        []<class Mesh>(const Mesh& mesh) noexcept
        {
            using concrete_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian> ||
                          std::same_as<concrete_type, typename mesh_type::Cylindrical> ||
                          std::same_as<concrete_type, typename mesh_type::SemiStructured>)
            {
                return Meshes::PlanarALEGeometryAccess::motion_available(mesh);
            }
            else
            {
                return false;
            }
        });
}

template<TpetraTypePack Pack> bool PlanarALEMeshMotion<Pack>::geometry_motion_owned() const noexcept
{
    return d_mesh->visit(
        [this]<class Mesh>(const Mesh& mesh) noexcept
        {
            using concrete_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian> ||
                          std::same_as<concrete_type, typename mesh_type::Cylindrical> ||
                          std::same_as<concrete_type, typename mesh_type::SemiStructured>)
            {
                return Meshes::PlanarALEGeometryAccess::motion_owned_by(mesh, this);
            }
            else
            {
                return false;
            }
        });
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::claim_geometry_motion()
{
    d_mesh->visit_mutable(
        [this]<class Mesh>(Mesh& mesh)
        {
            using concrete_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian> ||
                          std::same_as<concrete_type, typename mesh_type::Cylindrical> ||
                          std::same_as<concrete_type, typename mesh_type::SemiStructured>)
            {
                Meshes::PlanarALEGeometryAccess::claim_motion(mesh, this);
            }
            else
            {
                throw std::logic_error("Planar ALE cannot claim unsupported geometry.");
            }
        });
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::release_geometry_motion() noexcept
{
    if (!d_geometry_motion_claimed)
    {
        return;
    }
    try
    {
        d_mesh->visit_mutable(
            [this]<class Mesh>(Mesh& mesh)
            {
                using concrete_type = std::remove_cvref_t<Mesh>;
                if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian> ||
                              std::same_as<concrete_type, typename mesh_type::Cylindrical> ||
                              std::same_as<concrete_type, typename mesh_type::SemiStructured>)
                {
                    Meshes::PlanarALEGeometryAccess::release_motion(mesh, this);
                }
            });
    }
    catch (...)
    {
    }
    d_geometry_motion_claimed = false;
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::replace_axis_edges(ArrReal edges)
{
    d_mesh->visit_mutable(
        [this, &edges]<class Mesh>(Mesh& mesh)
        {
            using concrete_type = std::remove_cvref_t<Mesh>;
            if constexpr (std::same_as<concrete_type, typename mesh_type::Cartesian>)
            {
                Meshes::PlanarALEGeometryAccess::require_motion_owner(mesh, this);
                Meshes::PlanarALEGeometryAccess::replace_axis_edges(
                    mesh, static_cast<size_t>(d_options.axis), std::move(edges));
            }
            else if constexpr (std::same_as<concrete_type, typename mesh_type::Cylindrical> ||
                               std::same_as<concrete_type, typename mesh_type::SemiStructured>)
            {
                Meshes::PlanarALEGeometryAccess::require_motion_owner(mesh, this);
                Meshes::PlanarALEGeometryAccess::replace_axial_edges(mesh, std::move(edges));
            }
            else
            {
                throw std::invalid_argument("PlanarALEMeshMotion supports only Cartesian, cylindrical, "
                                            "and semi-structured extruded geometry.");
            }
        });
}

template<TpetraTypePack Pack> ArrReal PlanarALEMeshMotion<Pack>::candidate_axis_edges(real_t surface_elevation) const
{
    ArrReal result(d_reference_axis_edges.size());
    const auto displacement = surface_elevation - d_reference_surface_elevation;
    for (size_t edge = 0; edge < result.size(); ++edge)
    {
        result[edge] = d_reference_axis_edges[edge] + d_deformation_weights[edge] * displacement;
    }
    return result;
}

template<TpetraTypePack Pack>
std::array<ArrReal, 3> PlanarALEMeshMotion<Pack>::candidate_geometry_edges(real_t surface_elevation) const
{
    auto result = d_reference_geometry_edges;
    const auto moved_axis =
        d_family == Family::Cartesian ? static_cast<size_t>(d_options.axis) : static_cast<size_t>(Dimension::Z);
    result[moved_axis] = candidate_axis_edges(surface_elevation);
    return result;
}

template<TpetraTypePack Pack> std::vector<real_t> PlanarALEMeshMotion<Pack>::capture_cell_volumes() const
{
    std::vector<real_t> result(d_mesh->num_local_cells());
    for (size_t local = 0; local < result.size(); ++local)
    {
        result[local] = d_mesh->cell_volume(static_cast<local_ordinal_type>(local));
    }
    return result;
}

template<TpetraTypePack Pack>
std::vector<typename PlanarALEMeshMotion<Pack>::mesh_type::Vec3>
PlanarALEMeshMotion<Pack>::capture_face_centroids() const
{
    std::vector<typename mesh_type::Vec3> result(d_mesh->num_faces());
    for (size_t local = 0; local < result.size(); ++local)
    {
        result[local] = d_mesh->face_centroid(static_cast<local_ordinal_type>(local));
    }
    return result;
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::validate_collective_construction()
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    int local_valid = 1;
    const auto axis = static_cast<int>(d_options.axis);
    local_valid = local_valid && d_mesh->has_mutable_geometry() && geometry_motion_available() &&
                  d_family != Family::Unsupported && axis >= static_cast<int>(Dimension::X) &&
                  axis <= static_cast<int>(Dimension::Z) &&
                  (d_family == Family::Cartesian || d_options.axis == Dimension::Z) &&
                  d_reference_axis_edges.size() >= 2 &&
                  d_reference_axis_edges.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                  planar_ale_detail::finite_non_negative(d_options.gcl_absolute_tolerance) &&
                  planar_ale_detail::finite_non_negative(d_options.gcl_relative_tolerance);

    std::array<int, 3> geometry_edge_counts{};
    for (size_t axis_index = 0; axis_index < d_reference_geometry_edges.size(); ++axis_index)
    {
        const auto count = d_reference_geometry_edges[axis_index].size();
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            local_valid = 0;
            geometry_edge_counts[axis_index] = -1;
        }
        else
        {
            geometry_edge_counts[axis_index] = static_cast<int>(count);
        }
    }

    for (size_t edge = 0; edge < d_reference_axis_edges.size(); ++edge)
    {
        local_valid = local_valid && std::isfinite(d_reference_axis_edges[edge]) &&
                      (edge == 0 || d_reference_axis_edges[edge] > d_reference_axis_edges[edge - 1]);
    }

    const auto reference_bottom = d_reference_axis_edges.empty() ? real_t{} : d_reference_axis_edges.front();
    const auto reference_top = d_reference_axis_edges.empty() ? real_t{} : d_reference_axis_edges.back();
    const auto deformation_start = d_options.deformation_start_elevation.value_or(reference_bottom);
    local_valid = local_valid && std::isfinite(deformation_start) && deformation_start >= reference_bottom &&
                  deformation_start < reference_top;
    if (d_options.maximum_level_change)
    {
        local_valid =
            local_valid && std::isfinite(*d_options.maximum_level_change) && *d_options.maximum_level_change > 0.0;
    }

    try
    {
        static_cast<void>(MeshQualityGate(d_options.quality_limits));
    }
    catch (const std::invalid_argument&)
    {
        local_valid = 0;
    }

    int any_invalid = 0;
    const int local_invalid = local_valid == 0 ? 1 : 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);

    const std::array<int, 8> integer_state{{static_cast<int>(d_family), axis, d_mesh->has_mutable_geometry() ? 1 : 0,
        d_options.deformation_start_elevation ? 1 : 0, d_options.maximum_level_change ? 1 : 0,
        d_options.quality_limits.maximum_growth_ratio ? 1 : 0,
        d_options.quality_limits.maximum_non_orthogonality_degrees ? 1 : 0,
        d_options.quality_limits.maximum_skewness ? 1 : 0}};
    std::array<int, integer_state.size()> minimum_integer_state{};
    std::array<int, integer_state.size()> maximum_integer_state{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(integer_state.size()), integer_state.data(),
        minimum_integer_state.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(integer_state.size()), integer_state.data(),
        maximum_integer_state.data());

    const std::array<int, 1> extra_presence{{d_options.quality_limits.maximum_aspect_ratio ? 1 : 0}};
    std::array<int, 1> minimum_extra_presence{};
    std::array<int, 1> maximum_extra_presence{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, extra_presence.data(), minimum_extra_presence.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, extra_presence.data(), maximum_extra_presence.data());

    const std::vector<real_t> option_values{deformation_start, d_options.maximum_level_change.value_or(0.0),
        d_options.gcl_absolute_tolerance, d_options.gcl_relative_tolerance,
        d_options.quality_limits.maximum_growth_ratio.value_or(0.0),
        d_options.quality_limits.maximum_non_orthogonality_degrees.value_or(0.0),
        d_options.quality_limits.maximum_skewness.value_or(0.0),
        d_options.quality_limits.maximum_aspect_ratio.value_or(0.0)};
    const auto option_values_match = planar_ale_detail::collectively_equal_values(*communicator, option_values);

    std::array<int, 3> minimum_geometry_edge_counts{};
    std::array<int, 3> maximum_geometry_edge_counts{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(geometry_edge_counts.size()),
        geometry_edge_counts.data(), minimum_geometry_edge_counts.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(geometry_edge_counts.size()),
        geometry_edge_counts.data(), maximum_geometry_edge_counts.data());
    const auto geometry_edge_counts_match = minimum_geometry_edge_counts == maximum_geometry_edge_counts;
    bool geometry_edge_values_match = true;
    if (geometry_edge_counts_match)
    {
        for (size_t axis_index = 0; axis_index < d_reference_geometry_edges.size(); ++axis_index)
        {
            if (geometry_edge_counts[axis_index] > 0)
            {
                geometry_edge_values_match =
                    geometry_edge_values_match &&
                    planar_ale_detail::collectively_equal_values(*communicator, d_reference_geometry_edges[axis_index]);
            }
        }
    }

    const auto epoch = static_cast<unsigned long long>(d_mesh->geometry_epoch());
    const auto epoch_matches = planar_ale_detail::collectively_equal(*communicator, epoch);

    if (any_invalid != 0)
    {
        throw std::invalid_argument("PlanarALEMeshMotion requires valid options and a mutable "
                                    "Cartesian mesh, axial cylindrical mesh, or serial axial "
                                    "SemiStructuredXY_Z mesh on every rank.");
    }
    if (minimum_integer_state != maximum_integer_state || minimum_extra_presence != maximum_extra_presence ||
        !option_values_match || !geometry_edge_counts_match || !geometry_edge_values_match || !epoch_matches)
    {
        throw std::invalid_argument("PlanarALEMeshMotion mesh family, reference geometry, options, "
                                    "and geometry epoch must match on every rank.");
    }
}

template<TpetraTypePack Pack>
void PlanarALEMeshMotion<Pack>::validate_collective_trial(real_t surface_elevation, real_t time_step) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto current_epoch = d_mesh->geometry_epoch();
    const auto current_geometry = geometry_edge_coordinates();
    const auto expected_geometry = candidate_geometry_edges(d_accepted_surface_elevation);
    const auto proposed_edges = candidate_axis_edges(surface_elevation);

    int local_invalid = d_trial_active || !geometry_motion_owned() || current_epoch != d_expected_geometry_epoch ||
                                current_geometry != expected_geometry || !std::isfinite(surface_elevation) ||
                                !std::isfinite(time_step) || time_step <= 0.0 ||
                                proposed_edges.size() != d_reference_axis_edges.size() ||
                                current_epoch > std::numeric_limits<std::uint64_t>::max() - 2
                            ? 1
                            : 0;
    if (d_options.maximum_level_change &&
        std::abs(surface_elevation - d_accepted_surface_elevation) > *d_options.maximum_level_change)
    {
        local_invalid = 1;
    }
    for (size_t edge = 0; edge < proposed_edges.size(); ++edge)
    {
        if (!std::isfinite(proposed_edges[edge]) || (edge > 0 && proposed_edges[edge] <= proposed_edges[edge - 1]))
        {
            local_invalid = 1;
        }
    }

    int any_invalid = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    const auto surface_matches = planar_ale_detail::collectively_equal(*communicator, surface_elevation);
    const auto time_step_matches = planar_ale_detail::collectively_equal(*communicator, time_step);
    const auto epoch_value = static_cast<unsigned long long>(current_epoch);
    const auto epoch_matches = planar_ale_detail::collectively_equal(*communicator, epoch_value);
    const int active_value = d_trial_active ? 1 : 0;
    const auto active_matches = planar_ale_detail::collectively_equal(*communicator, active_value);

    if (any_invalid != 0)
    {
        throw std::invalid_argument("PlanarALEMeshMotion trial requires an idle, unchanged geometry, "
                                    "a finite positive time step, a finite non-inverting target, and "
                                    "a displacement within the configured limit on every rank.");
    }
    if (!surface_matches || !time_step_matches || !epoch_matches || !active_matches)
    {
        throw std::invalid_argument("PlanarALEMeshMotion target, time step, geometry epoch, and "
                                    "transaction state must match on every rank.");
    }
}

template<TpetraTypePack Pack>
void PlanarALEMeshMotion<Pack>::validate_collective_transaction(TransactionAction action) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const int active = d_trial_active ? 1 : 0;
    const auto active_matches = planar_ale_detail::collectively_equal(*communicator, active);
    const auto epoch = static_cast<unsigned long long>(d_mesh->geometry_epoch());
    const auto epoch_matches = planar_ale_detail::collectively_equal(*communicator, epoch);
    const auto action_value = static_cast<int>(action);
    const auto action_matches = planar_ale_detail::collectively_equal(*communicator, action_value);
    const auto current_geometry = geometry_edge_coordinates();
    const int local_invalid = !d_trial_active || !geometry_motion_owned() ||
                                      d_mesh->geometry_epoch() != d_expected_geometry_epoch ||
                                      current_geometry != d_trial_geometry_edges
                                  ? 1
                                  : 0;
    int any_invalid = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (!active_matches || !epoch_matches || !action_matches || any_invalid != 0)
    {
        throw std::logic_error("PlanarALEMeshMotion action, transaction state, geometry, and "
                               "geometry epoch must match an active trial on every rank.");
    }
}

template<TpetraTypePack Pack>
void PlanarALEMeshMotion<Pack>::compute_trial_state(
    const std::vector<typename mesh_type::Vec3>& old_face_centroids, real_t time_step)
{
    d_new_cell_volumes = capture_cell_volumes();
    d_face_mesh_fluxes.assign(d_mesh->num_faces(), 0.0);
    int local_invalid =
        old_face_centroids.size() != d_face_mesh_fluxes.size() || d_old_cell_volumes.size() != d_new_cell_volumes.size()
            ? 1
            : 0;
    for (size_t face = 0; face < d_face_mesh_fluxes.size(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        const auto displacement = d_mesh->face_centroid(face_lid) - old_face_centroids[face];
        const auto swept_volume = displacement.dot(d_mesh->face_area_vector(face_lid));
        const auto flux = swept_volume / time_step;
        d_face_mesh_fluxes[face] = flux;
        local_invalid = local_invalid || !std::isfinite(flux);
    }

    real_t local_maximum_absolute_residual = {};
    real_t local_maximum_normalized_residual = {};
    int local_tolerance_failure = 0;
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        real_t mesh_flux_balance = {};
        for (const auto face_lid : d_mesh->faces(cell_lid))
        {
            const auto flux = d_face_mesh_fluxes.at(static_cast<size_t>(face_lid));
            mesh_flux_balance += d_mesh->owner_cell(face_lid) == cell_lid ? flux : -flux;
        }
        const auto volume_rate = (d_new_cell_volumes[owned] - d_old_cell_volumes[owned]) / time_step;
        const auto residual = volume_rate - mesh_flux_balance;
        const auto scale = std::max(std::abs(volume_rate), std::abs(mesh_flux_balance));
        const auto tolerance = d_options.gcl_absolute_tolerance + d_options.gcl_relative_tolerance * scale;
        const auto normalized = scale > 0.0 ? std::abs(residual) / scale : std::abs(residual);
        local_invalid = local_invalid || !std::isfinite(volume_rate) || !std::isfinite(mesh_flux_balance) ||
                        !std::isfinite(residual);
        local_tolerance_failure = local_tolerance_failure || std::abs(residual) > tolerance;
        local_maximum_absolute_residual = std::max(local_maximum_absolute_residual, std::abs(residual));
        local_maximum_normalized_residual = std::max(local_maximum_normalized_residual, normalized);
    }

    const auto communicator = d_mesh->owned_cell_map()->getComm();
    int any_invalid = 0;
    int any_tolerance_failure = 0;
    real_t global_maximum_absolute_residual = {};
    real_t global_maximum_normalized_residual = {};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_tolerance_failure, &any_tolerance_failure);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1, &local_maximum_absolute_residual, &global_maximum_absolute_residual);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1, &local_maximum_normalized_residual, &global_maximum_normalized_residual);
    if (any_invalid != 0)
    {
        throw std::runtime_error("PlanarALEMeshMotion produced non-finite swept-volume or GCL data.");
    }
    if (any_tolerance_failure != 0)
    {
        std::ostringstream message;
        message << "PlanarALEMeshMotion GCL residual " << global_maximum_absolute_residual
                << " m^3/s exceeds its configured tolerance.";
        throw std::runtime_error(message.str());
    }

    const auto quality = evaluate_mesh_quality(*d_mesh);
    const MeshQualityGate gate(d_options.quality_limits);
    const auto assessment = gate.assess(quality);
    if (!assessment.accepted())
    {
        throw std::runtime_error("PlanarALEMeshMotion trial " + assessment.report(quality));
    }

    d_diagnostics.maximum_absolute_gcl_residual = global_maximum_absolute_residual;
    d_diagnostics.maximum_normalized_gcl_residual = global_maximum_normalized_residual;
    d_diagnostics.mesh_quality = quality;
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::begin_trial(real_t surface_elevation, real_t time_step)
{
    validate_collective_trial(surface_elevation, time_step);
    auto candidate_edges = candidate_axis_edges(surface_elevation);
    d_trial_geometry_edges = candidate_geometry_edges(surface_elevation);
    d_pre_trial_axis_edges = current_axis_edges();
    d_old_cell_volumes = capture_cell_volumes();
    const auto old_face_centroids = capture_face_centroids();
    const auto old_epoch = d_mesh->geometry_epoch();

    bool geometry_replaced = false;
    try
    {
        replace_axis_edges(std::move(candidate_edges));
        geometry_replaced = true;
        d_expected_geometry_epoch = d_mesh->geometry_epoch();
        d_trial_active = true;

        d_diagnostics = {};
        d_diagnostics.old_surface_elevation = d_accepted_surface_elevation;
        d_diagnostics.new_surface_elevation = surface_elevation;
        d_diagnostics.time_step = time_step;
        d_diagnostics.old_geometry_epoch = old_epoch;
        d_diagnostics.new_geometry_epoch = d_expected_geometry_epoch;
        d_diagnostics.trial_active = true;
        compute_trial_state(old_face_centroids, time_step);
    }
    catch (...)
    {
        const auto failure = std::current_exception();
        if (geometry_replaced)
        {
            rollback_impl();
        }
        std::rethrow_exception(failure);
    }
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::accept_trial()
{
    validate_collective_transaction(TransactionAction::Accept);
    d_accepted_surface_elevation = d_diagnostics.new_surface_elevation;
    d_accepted_quality = d_diagnostics.mesh_quality;
    d_pre_trial_axis_edges.clear();
    d_trial_geometry_edges = {};
    d_trial_active = false;
    d_diagnostics.trial_active = false;
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::rollback_trial()
{
    validate_collective_transaction(TransactionAction::Rollback);
    rollback_impl();
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::rollback_impl()
{
    if (!d_trial_active && d_pre_trial_axis_edges.empty())
    {
        return;
    }
    replace_axis_edges(std::move(d_pre_trial_axis_edges));
    d_expected_geometry_epoch = d_mesh->geometry_epoch();
    d_trial_active = false;
    d_trial_geometry_edges = {};
    reset_stationary_state();
}

template<TpetraTypePack Pack> void PlanarALEMeshMotion<Pack>::reset_stationary_state()
{
    d_old_cell_volumes = capture_cell_volumes();
    d_new_cell_volumes = d_old_cell_volumes;
    d_face_mesh_fluxes.assign(d_mesh->num_faces(), 0.0);
    d_diagnostics = {};
    d_diagnostics.old_surface_elevation = d_accepted_surface_elevation;
    d_diagnostics.new_surface_elevation = d_accepted_surface_elevation;
    d_diagnostics.old_geometry_epoch = d_mesh->geometry_epoch();
    d_diagnostics.new_geometry_epoch = d_mesh->geometry_epoch();
    d_diagnostics.mesh_quality = d_accepted_quality;
}

} // namespace SimpleFluid
