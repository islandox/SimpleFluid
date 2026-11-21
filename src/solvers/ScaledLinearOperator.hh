/**
 * @file ScaledLinearOperator.hh
 * @brief Internal constant-scaled CRS operator wrapper used by Krylov solves.
 */
#pragma once

#include "dataclass/TpetraTypes.hh"
#include "utils/export_helper.hh"

#include <Teuchos_RCP.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{

/**
 * @brief Internal constant-scaled CRS operator wrapper.
 *
 * Multiplies the wrapped matrix into the Krylov operator so the time step
 * stays on the linear system's left side without rebuilding the cached
 * geometric matrix or invalidating its preconditioner. Preconditioning can
 * keep using the unscaled matrix object.
 */
template<TpetraTypePack Pack>
class SIMPLEFLUID_SOLVERS_EXPORT ScaledLinearOperator final : public Pack::operator_type
{
public:
    using scalar_type = typename Pack::scalar_type;
    using map_type = typename Pack::map_type;
    using matrix_type = typename Pack::matrix_type;
    using multi_vector_type = typename Pack::multi_vector_type;

    ScaledLinearOperator(Teuchos::RCP<const matrix_type> matrix, scalar_type scale)
        : d_matrix(std::move(matrix)), d_scale(scale)
    {
        if (d_matrix.is_null() || !std::isfinite(d_scale) || d_scale <= scalar_type{})
            throw std::invalid_argument("ScaledLinearOperator requires a matrix and a positive finite scale.");
    }

    // Out-of-line key function: SimpleFluidSolvers owns the vtable and type
    // information so cross-DSO dynamic casts cannot compare two private copies.
    ~ScaledLinearOperator() override;

    Teuchos::RCP<const map_type> getDomainMap() const override { return d_matrix->getDomainMap(); }
    Teuchos::RCP<const map_type> getRangeMap() const override { return d_matrix->getRangeMap(); }
    bool hasTransposeApply() const override { return false; }

    /** @brief The wrapped unscaled CRS matrix, for compensated residual evaluation. */
    const Teuchos::RCP<const matrix_type>& scaled_crs_matrix() const noexcept { return d_matrix; }

    /** @brief The positive constant multiplied into every application. */
    scalar_type crs_scale() const noexcept { return d_scale; }

    void apply(const multi_vector_type& input, multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS, scalar_type alpha = scalar_type{1},
        scalar_type beta = scalar_type{}) const override
    {
        if (mode != Teuchos::NO_TRANS)
            throw std::invalid_argument("ScaledLinearOperator supports only non-transpose application.");
        d_matrix->apply(input, output, mode, alpha * d_scale, beta);
    }

private:
    Teuchos::RCP<const matrix_type> d_matrix;
    scalar_type d_scale;
};

// Compiled into SimpleFluidSolvers for the default pack so every consumer
// references that DSO's vtable and type information instead of emitting a
// private copy. A non-default pack must include "solvers/ScaledLinearOperator.tcc"
// in the translation unit that instantiates it.
extern template class ScaledLinearOperator<DefaultTpetraTypes>;

} // namespace SimpleFluid::detail
