/**
 * @file VolumeContinuityTarget.hh
 * @brief Named integrated per-cell target for low-Mach volume continuity.
 */

#pragma once

#include "dataclass/TpetraTypes.hh"
#include "dataclass/typedefs.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/GeometryEpoch.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/** Detailed norms of the integrated continuity residual, in m^3/s. */
template<class Scalar> struct VolumeContinuityResiduals
{
    Scalar l2 = {};            ///< Global L2 norm [m^3/s].
    Scalar maximum = {};       ///< Global maximum absolute value [m^3/s].
    Scalar normalized_l2 = {}; ///< L2 divided by @ref normalization.
    Scalar normalization = {}; ///< Global flux/target reference norm [m^3/s].
};

/**
 * @brief Immutable snapshot of one integrated cellwise volume target.
 *
 * The target represents @f$Q_{V,c}=\int_{V_c}S_V\,dV@f$ in m^3/s and the
 * pressure-velocity constraint is @f$\sum_f\phi_{abs,f}-Q_{V,c}=0@f$.
 * Values use owned-cell order. A generation identifies the physical target
 * state for predictor-cache validation; callers must change it whenever any
 * target value changes.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> class VolumeContinuityTarget
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_cell_type;

    /** Construct a zero target at the mesh's current geometry epoch. */
    explicit VolumeContinuityTarget(SP<const mesh_type> mesh, std::uint64_t generation = 0)
        : VolumeContinuityTarget(std::move(mesh), {}, generation, ZeroTag{})
    {
    }

    /** Construct from owned-cell integrated rates in mesh-owned order. */
    VolumeContinuityTarget(
        SP<const mesh_type> mesh, std::vector<scalar_type> integrated_rates, std::uint64_t generation)
        : d_mesh(require_mesh(std::move(mesh))), d_integrated_rates(std::move(integrated_rates)),
          d_generation(generation), d_geometry_epoch(mesh_geometry_epoch(*d_mesh))
    {
    }

    /** Copy an integrated-rate field into an immutable target snapshot. */
    static VolumeContinuityTarget from_field(const field_type& integrated_rates, std::uint64_t generation)
    {
        auto mesh = integrated_rates.mesh_ptr();
        std::vector<scalar_type> values(mesh->num_owned_cells());
        for (size_t owned = 0; owned < values.size(); ++owned)
        {
            values[owned] = integrated_rates.value(static_cast<local_ordinal_type>(owned));
        }
        return VolumeContinuityTarget(std::move(mesh), std::move(values), generation);
    }

    /**
     * Build a target from a rank-local integrated-rate provider, converting
     * provider failures into one collective failure before later collectives.
     */
    template<class Provider>
    static VolumeContinuityTarget from_integrated_rate_provider(SP<const mesh_type> mesh, Provider&& provider,
        std::uint64_t generation, std::string_view context = "Volume continuity target")
    {
        mesh = require_mesh(std::move(mesh));
        std::vector<scalar_type> values(mesh->num_owned_cells());
        std::exception_ptr local_error;
        try
        {
            for (size_t owned = 0; owned < values.size(); ++owned)
            {
                values[owned] = provider(static_cast<local_ordinal_type>(owned));
            }
        }
        catch (...)
        {
            local_error = std::current_exception();
        }

        const int local_failed = local_error ? 1 : 0;
        int any_failed = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_failed, &any_failed);
        if (any_failed != 0)
        {
            if (local_error)
            {
                std::rethrow_exception(local_error);
            }
            throw std::runtime_error(std::string(context) + " provider failed on another mesh rank.");
        }
        return VolumeContinuityTarget(std::move(mesh), std::move(values), generation);
    }

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    const SP<const mesh_type>& mesh_ptr() const noexcept { return d_mesh; }

    scalar_type integrated_rate(local_ordinal_type cell_lid) const
    {
        return d_integrated_rates.at(static_cast<size_t>(cell_lid));
    }

    std::span<const scalar_type> owned_integrated_rates() const noexcept { return d_integrated_rates; }

    std::uint64_t generation() const noexcept { return d_generation; }
    std::uint64_t geometry_epoch() const noexcept { return d_geometry_epoch; }

    /** Validate mesh identity, epoch, generation parity, size, and finiteness. */
    void validate(const mesh_type& expected_mesh, std::string_view context = "Volume continuity target") const
    {
        const auto communicator = expected_mesh.owned_cell_map()->getComm();
        int local_invalid = d_mesh.get() != &expected_mesh ||
                                    d_integrated_rates.size() != expected_mesh.num_owned_cells() ||
                                    d_geometry_epoch != mesh_geometry_epoch(expected_mesh)
                                ? 1
                                : 0;
        for (const auto value : d_integrated_rates)
        {
            local_invalid = local_invalid || !std::isfinite(value);
        }
        int any_invalid = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);

        const auto generation = static_cast<unsigned long long>(d_generation);
        const auto epoch = static_cast<unsigned long long>(d_geometry_epoch);
        unsigned long long minimum_generation = 0;
        unsigned long long maximum_generation = 0;
        unsigned long long minimum_epoch = 0;
        unsigned long long maximum_epoch = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &generation, &minimum_generation);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &generation, &maximum_generation);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &epoch, &minimum_epoch);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &epoch, &maximum_epoch);

        if (any_invalid != 0)
        {
            throw std::invalid_argument(std::string(context) + " must contain finite integrated rates in owned-cell "
                                                               "order on the current target mesh epoch.");
        }
        if (minimum_generation != maximum_generation || minimum_epoch != maximum_epoch)
        {
            throw std::invalid_argument(std::string(context) + " generation and geometry epoch must match on every "
                                                               "mesh rank.");
        }
    }

private:
    struct ZeroTag
    {
    };

    VolumeContinuityTarget(SP<const mesh_type> mesh, std::vector<scalar_type>, std::uint64_t generation, ZeroTag)
        : d_mesh(require_mesh(std::move(mesh))), d_integrated_rates(d_mesh->num_owned_cells(), scalar_type{}),
          d_generation(generation), d_geometry_epoch(mesh_geometry_epoch(*d_mesh))
    {
    }

    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh)
    {
        if (!mesh || mesh->owned_cell_map().is_null())
        {
            throw std::invalid_argument("VolumeContinuityTarget requires a mesh with an owned-cell map.");
        }
        return mesh;
    }

    SP<const mesh_type> d_mesh;
    std::vector<scalar_type> d_integrated_rates;
    std::uint64_t d_generation = 0;
    std::uint64_t d_geometry_epoch = 0;
};

} // namespace SimpleFluid
