/**
 * @file PrescribedFluxResponse.hh
 * @brief Frozen prescribed-flux RHS response, including affine non-orthogonal terms.
 */
#pragma once

#include "FVM/details/FieldStoredTransportSystem.hh"
#include <exception>
#include <map>

namespace SimpleFluid::FVM::detail
{
/** Boundary-sized application plan for a fixed physical transport session.
 * Geometry and coefficients are captured after the native operator is assembled.
 * Mutable gradient scratch belongs to one equation owner; returned vectors are independent.
 */
template<TpetraTypePack Pack, class MeshType> class PrescribedFluxResponsePlan
{
    using Scalar = typename Pack::scalar_type;
    using Local = typename Pack::local_ordinal_type;
    using Vec = typename MeshType::Vec3;
    using Cache = MeshBoundaryCache<Pack, MeshType>;
    using Gradient = VectorCellFieldStored<Pack, MeshType>;
    struct Input
    {
        int batch;
        size_t slot;
        Local owner;
        Scalar conductivity, area;
    };
    struct Sample
    {
        Local cell;
        size_t input;
        Vec basis;
        real_t distance;
    };
    enum class Kind
    {
        Flux,
        OwnedGradient,
        ImportedGradient,
        ExplicitGradient
    };
    struct Operation
    {
        Kind kind;
        Local row, gradient, other;
        size_t input;
        Scalar scale;
        Vec area;
    };
    template<class Work> static void validate_local(const MeshType& mesh, const char* context, Work&& work)
    {
        std::exception_ptr error;
        try
        {
            work();
        }
        catch (...)
        {
            error = std::current_exception();
        }
        const auto failed = reduce_transport_validation_state(mesh, std::array<int, 1>{error ? 1 : 0});
        if (error)
            std::rethrow_exception(error);
        if (failed[0])
            throw std::runtime_error(std::string(context) + " failed on another rank.");
    }
    SP<const MeshType> d_mesh;
    std::vector<Input> d_inputs;
    std::vector<Sample> d_samples;
    std::vector<Operation> d_implicit, d_explicit;
    mutable std::vector<Scalar> d_normalized_flux;
    mutable std::vector<Vec> d_owned_gradient;
    std::unique_ptr<Gradient> d_gradient;

public:
    template<class Conductivity, class BoundaryCoefficient, class BoundaryCondition>
    PrescribedFluxResponsePlan(SP<const MeshType> mesh, const Cache& support,
        const TransportGeometryCache<MeshType>& geometry, Conductivity conductivity, BoundaryCoefficient boundary_k,
        BoundaryCondition condition, NonOrthogonalTreatment treatment, FaceCoefficientInterpolation interpolation)
        : d_mesh(std::move(mesh))
    {
        const auto& m = *d_mesh;
        std::map<std::pair<int, size_t>, size_t> indices;
        validate_local(m, "Prepared flux response geometry",
            [&]
            {
                geometry.require_mesh(m);
                for (const auto& [batch, values] : support.value)
                {
                    const auto& faces = m.boundary_face_batch(batch).face_lids;
                    for (size_t slot = 0; slot < values.size(); ++slot)
                    {
                        const auto owner = m.owner_cell(faces[slot]);
                        if (!m.is_owned_cell(owner))
                            continue;
                        indices[{batch, slot}] = d_inputs.size();
                        d_inputs.push_back({batch, slot, owner, boundary_k(batch, slot, conductivity(owner)),
                            static_cast<Scalar>(m.face_area(faces[slot]))});
                    }
                }
                auto homogeneous = [&](int batch, size_t slot)
                {
                    auto value = condition(batch, slot);
                    value.value = 0;
                    return value;
                };
                auto observe = [&](Local cell, const auto& location, const Vec& basis, real_t distance)
                {
                    const auto found = indices.find({location.batch_id, location.in_batch_id});
                    if (found != indices.end())
                        d_samples.push_back({cell, found->second, basis, distance});
                };
                bool robin = false;
                for (const auto& [batch, faces] : m.boundary_batches())
                    for (size_t slot = 0; slot < faces.face_lids.size(); ++slot)
                        robin |= m.is_boundary_face(faces.face_lids[slot]) &&
                                 homogeneous(batch, slot).type == BoundaryConditionType::Robin;
                const auto zero = [](int, size_t) { return Scalar{}; };
                if (robin)
                    scalar_affine_gradient_stencils(m, homogeneous, zero, geometry.traverse_plan(), observe);
                else
                    materialize_scalar_affine_gradient_stencils<MeshType>(
                        geometry.boundary_geometry(), homogeneous, zero, observe);
            });
        d_normalized_flux.resize(d_inputs.size());
        const int local[2] = {!d_inputs.empty(), geometry.has_non_orthogonal_faces()};
        int global[2]{};
        Teuchos::reduceAll(*m.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 2, local, global);
        const bool gradients = global[0] != 0;
        if (gradients)
        {
            d_owned_gradient.resize(m.num_owned_cells());
            d_gradient = std::make_unique<Gradient>(d_mesh, "prepared_flux_response_gradient");
            d_gradient->owned_data().putScalar(0.);
            {
                auto data = d_gradient->owned_data().getLocalViewHost(Tpetra::Access::ReadWrite);
                for (const auto& sample : d_samples)
                    data(sample.cell, 0) = 1.;
            }
            d_gradient->sync_ghosts();
        }
        const auto weights = non_orthogonal_weights<Scalar>(treatment);
        validate_local(m, "Prepared flux response operations",
            [&]
            {
                const auto active =
                    gradients ? d_gradient->local_read_view() : decltype(d_gradient->local_read_view()){};
                auto has_gradient = [&](Local cell) { return gradients && active(cell, 0) != 0.; };
                auto append = [&](Local row, Local cell, Scalar scale, const Vec& area)
                {
                    if (!has_gradient(cell) || scale == Scalar{} || area.dot(area) == 0.)
                        return;
                    d_implicit.push_back({m.is_owned_cell(cell) ? Kind::OwnedGradient : Kind::ImportedGradient, row,
                        cell, Local{}, 0, scale, area});
                };
                const auto execution = acquire_mesh_execution(m);
                visit_transport_assembly_rows(m, geometry.assembly_geometry(),
                    [&](Local cell, auto, const auto& visit_faces)
                    {
                        visit_faces(
                            [&](const auto& face, const auto& metrics)
                            {
                                using Face = std::remove_cvref_t<decltype(face)>;
                                const auto id = face.face_lid;
                                if constexpr (Face::interior)
                                {
                                    const auto other = face.other;
                                    const Scalar k = face_coefficient_value(metrics, id, cell, other,
                                        conductivity(cell), conductivity(other), interpolation);
                                    if (k <= Scalar{})
                                        return;
                                    const Vec area =
                                        non_orthogonal_area_vector(metrics.face_area_vector_outward(id, cell),
                                            metrics.cell_center_vector(id, cell));
                                    append(cell, cell, -weights.implicit * k * Scalar{0.5}, area);
                                    // The assembler adds imported constants, and subtracts owned affine constants.
                                    const Scalar sign = m.is_owned_cell(other) ? Scalar{-1} : Scalar{1};
                                    if (m.is_owned_cell(other) || global[1])
                                        append(cell, other, sign * weights.implicit * k * Scalar{0.5}, area);
                                    if (global[1] && weights.explicit_ > Scalar{} &&
                                        (has_gradient(cell) || has_gradient(other)) && area.dot(area) != 0.)
                                        d_explicit.push_back({Kind::ExplicitGradient, cell, cell, other, 0,
                                            weights.explicit_ * k, area});
                                }
                                else if constexpr (Face::boundary)
                                {
                                    const auto& location = geometry.boundary_locations().at(static_cast<size_t>(id));
                                    if (!location.active)
                                        return;
                                    const auto bc = condition(location.batch_id, location.in_batch_id);
                                    const Scalar k =
                                        boundary_k(location.batch_id, location.in_batch_id, conductivity(cell));
                                    if (bc.type == BoundaryConditionType::Neumann)
                                    {
                                        const auto found = indices.find({location.batch_id, location.in_batch_id});
                                        if (found != indices.end())
                                            d_implicit.push_back({Kind::Flux, cell, Local{}, Local{}, found->second, k,
                                                Vec{static_cast<Scalar>(metrics.face_area(id)), 0, 0}});
                                    }
                                    else if (bc.type == BoundaryConditionType::Dirichlet ||
                                             bc.type == BoundaryConditionType::Robin)
                                    {
                                        const Scalar effective =
                                            k * (bc.type == BoundaryConditionType::Robin
                                                        ? robin_boundary_weight(metrics, id, cell, bc.robin_coefficient)
                                                        : Scalar{1});
                                        const Vec area =
                                            non_orthogonal_area_vector(metrics.face_area_vector_outward(id, cell),
                                                metrics.face_centroid(id) - metrics.cell_centroid(cell));
                                        append(cell, cell, -weights.implicit * effective, area);
                                        if (global[1] && weights.explicit_ > Scalar{} && has_gradient(cell) &&
                                            area.dot(area) != 0.)
                                            d_explicit.push_back({Kind::ExplicitGradient, cell, cell,
                                                MeshType::invalid_local_id(), 0, weights.explicit_ * effective, area});
                                    }
                                }
                            });
                    });
            });
        const int local_gradient_ops = !d_explicit.empty() || std::any_of(d_implicit.begin(), d_implicit.end(),
                                                                  [](const auto& op) { return op.kind != Kind::Flux; });
        int global_gradient_ops = 0;
        Teuchos::reduceAll(
            *m.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_gradient_ops, &global_gradient_ops);
        if (!global_gradient_ops)
        {
            d_gradient.reset();
            d_owned_gradient.clear();
            d_samples.clear();
        }
    }

    /** Normalize increments during the caller's collective local validation. */
    void validate_increment(const Cache& increment) const
    {
        for (size_t i = 0; i < d_inputs.size(); ++i)
        {
            const auto& input = d_inputs[i];
            const auto found = increment.value.find(input.batch);
            const Scalar q = found == increment.value.end() ? Scalar{} : found->second[input.slot];
            if (input.conductivity <= Scalar{} && q != Scalar{})
                throw std::invalid_argument("Nonzero boundary response requires positive conductivity.");
            d_normalized_flux[i] = input.conductivity > Scalar{} ? -q / input.conductivity : Scalar{};
            if (!std::isfinite(d_normalized_flux[i]))
                throw std::overflow_error("Boundary response normalization overflow.");
        }
    }

    /** Apply validated increments without cell-face or matrix assembly. */
    Teuchos::RCP<const typename Pack::vector_type> response() const
    {
        auto result = Teuchos::rcp(new typename Pack::vector_type(d_mesh->owned_cell_map(), true));
        if (d_gradient)
        {
            std::fill(d_owned_gradient.begin(), d_owned_gradient.end(), Vec{});
            for (const auto& s : d_samples)
            {
                const auto increment = d_normalized_flux[s.input] * s.distance;
                d_owned_gradient[s.cell].x += s.basis.x * increment;
                d_owned_gradient[s.cell].y += s.basis.y * increment;
                d_owned_gradient[s.cell].z += s.basis.z * increment;
            }
            {
                auto values = d_gradient->owned_data().getLocalViewHost(Tpetra::Access::ReadWrite);
                for (size_t i = 0; i < d_owned_gradient.size(); ++i)
                {
                    values(i, 0) = d_owned_gradient[i].x;
                    values(i, 1) = d_owned_gradient[i].y;
                    values(i, 2) = d_owned_gradient[i].z;
                }
            }
            d_gradient->sync_ghosts();
        }
        const auto gradients = d_gradient ? d_gradient->local_read_view() : decltype(d_gradient->local_read_view()){};
        auto gradient = [&](Local cell) { return Vec{gradients(cell, 0), gradients(cell, 1), gradients(cell, 2)}; };
        auto values = result->getLocalViewHost(Tpetra::Access::ReadWrite);
        for (const auto& op : d_implicit)
        {
            if (op.kind == Kind::Flux)
                values(op.row, 0) += op.scale * d_normalized_flux[op.input] * op.area.x;
            else if (op.kind == Kind::OwnedGradient)
                values(op.row, 0) -= op.scale * d_owned_gradient[op.gradient].dot(op.area);
            else
                values(op.row, 0) += op.scale * gradient(op.gradient).dot(op.area);
        }
        for (const auto& op : d_explicit)
        {
            const auto g = op.other == MeshType::invalid_local_id()
                               ? gradient(op.gradient)
                               : (gradient(op.gradient) + gradient(op.other)) / Scalar{2};
            values(op.row, 0) += op.scale * g.dot(op.area);
        }
        return result;
    }
};
} // namespace SimpleFluid::FVM::detail
