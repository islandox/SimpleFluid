/** @file CoupledBlockOperator.hh @brief Coupled action without a monolithic CRS allocation. */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/NumericAssemblyLease.hh"
#include "dataclass/TpetraTypes.hh"

#include <array>
#include <memory>
#include <optional>

namespace SimpleFluid
{
/**
 * True velocity-pressure action. Blocks are frozen for the lifetime of a
 * retained generation. All halo exchange is delegated to block apply().
 * Mutable per-instance workspace makes apply sequential and non-reentrant.
 * Scratch is replaced only when the RHS column count changes. X/Y may alias,
 * including overlapping column views: all inputs are packed before writing Y.
 */
template<TpetraTypePack Pack> class SIMPLEFLUID_SOLVERS_EXPORT CoupledBlockOperator final : public Pack::operator_type
{
public:
    using scalar_type = typename Pack::scalar_type;
    using map_type = typename Pack::map_type;
    using matrix_type = typename Pack::matrix_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    CoupledBlockOperator(Teuchos::RCP<const map_type> map, Teuchos::RCP<const matrix_type> momentum,
        std::array<Teuchos::RCP<matrix_type>, 3> gradient, std::array<Teuchos::RCP<matrix_type>, 3> divergence,
        Teuchos::RCP<const matrix_type> stabilization, std::optional<global_ordinal_type> gauge,
        std::shared_ptr<FVM::NumericAssemblyLease> lease = {});

    Teuchos::RCP<const map_type> getDomainMap() const override { return d_map; }
    Teuchos::RCP<const map_type> getRangeMap() const override { return d_map; }
    bool hasTransposeApply() const override { return false; }
    size_t scratch_allocations() const noexcept { return d_allocations; }
    size_t scratch_payload_bytes() const noexcept
    {
        return 10 * d_cells->getLocalNumElements() * d_columns * sizeof(scalar_type);
    }

    void apply(const multi_vector_type& input, multi_vector_type& output, Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha = scalar_type{1}, scalar_type beta = scalar_type{}) const override;

private:
    SIMPLEFLUID_SOLVERS_LOCAL
    void ensure_scratch(size_t columns) const;
    Teuchos::RCP<const map_type> d_map, d_cells;
    Teuchos::RCP<const matrix_type> d_momentum;
    std::array<Teuchos::RCP<matrix_type>, 3> d_gradient, d_divergence;
    Teuchos::RCP<const matrix_type> d_stabilization;
    std::shared_ptr<FVM::NumericAssemblyLease> d_lease;
    local_ordinal_type d_gauge;
    mutable Teuchos::RCP<multi_vector_type> d_u, d_yu, d_q, d_yq, d_temp, d_component;
    mutable size_t d_columns = 0, d_allocations = 0;
};
extern template class CoupledBlockOperator<DefaultTpetraTypes>;

} // namespace SimpleFluid
