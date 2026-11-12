/**
 * @file PrescribedFluxResponse.hh
 * @brief Frozen prescribed-flux RHS response, including affine non-orthogonal terms.
 */
#pragma once

#include "FVM/BoundaryCache.hh"
#include "FVM/FaceCoefficientInterpolation.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/TransportCoefficientAccess.hh"
#include "FVM/details/TransportValidation.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_RCP.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{
template<class MeshType> class TransportGeometryCache;
}

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
    PrescribedFluxResponsePlan(SP<const MeshType> mesh, const Cache& support,
        const TransportGeometryCache<MeshType>& geometry, TransportCoefficientView<Pack> conductivity,
        const MeshBoundaryCache<Pack, MeshType>* boundary_conductivity,
        const std::map<std::pair<int, size_t>, SimpleFluid::BoundaryCondition>& frozen_conditions,
        NonOrthogonalTreatment treatment, FaceCoefficientInterpolation interpolation);

    /** Normalize increments during the caller's collective local validation. */
    void validate_increment(const Cache& increment) const;

    /** Apply validated increments without cell-face or matrix assembly. */
    Teuchos::RCP<const typename Pack::vector_type> response() const;
};

extern template class PrescribedFluxResponsePlan<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM::detail
