/** @file CoupledBlockOperator.hh @brief Coupled action without a monolithic CRS allocation. */
#pragma once

#include "FVM/NumericAssemblyLease.hh"
#include "dataclass/TpetraTypes.hh"

#include <Tpetra_Access.hpp>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>

namespace SimpleFluid
{
/**
 * True velocity-pressure action. Blocks are frozen for the lifetime of a
 * retained generation. All halo exchange is delegated to block apply().
 * Mutable per-instance workspace makes apply sequential and non-reentrant.
 * Scratch is replaced only when the RHS column count changes. X/Y may alias,
 * including overlapping column views: all inputs are packed before writing Y.
 */
template<TpetraTypePack Pack> class CoupledBlockOperator final : public Pack::operator_type
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
        std::shared_ptr<FVM::NumericAssemblyLease> lease = {})
        : d_map(std::move(map)), d_momentum(std::move(momentum)), d_gradient(std::move(gradient)),
          d_divergence(std::move(divergence)), d_stabilization(std::move(stabilization)), d_lease(std::move(lease))
    {
        if (d_map.is_null() || d_momentum.is_null() || d_stabilization.is_null())
            throw std::invalid_argument("CoupledBlockOperator requires maps and complete blocks.");
        d_cells = d_momentum->getDomainMap();
        if (d_map->getLocalNumElements() != 4 * d_cells->getLocalNumElements() ||
            d_map->getGlobalNumElements() != 4 * d_cells->getGlobalNumElements())
            throw std::invalid_argument("CoupledBlockOperator requires four entries per owned cell.");
        for (size_t cell = 0; cell < d_cells->getLocalNumElements(); ++cell)
        {
            const auto gid = d_cells->getGlobalElement(static_cast<local_ordinal_type>(cell));
            for (int c = 0; c < 4; ++c)
                if (d_map->getGlobalElement(static_cast<local_ordinal_type>(4 * cell + c)) != 4 * gid + c)
                    throw std::invalid_argument("CoupledBlockOperator map is not cell-interleaved.");
        }
        auto validate = [&](const auto& block)
        {
            if (block.is_null() || !block->isFillComplete() || !block->getDomainMap()->isSameAs(*d_cells) ||
                !block->getRangeMap()->isSameAs(*d_cells))
                throw std::invalid_argument("CoupledBlockOperator block maps do not match.");
        };
        validate(d_momentum);
        validate(d_stabilization);
        for (size_t c = 0; c < 3; ++c)
        {
            validate(d_gradient[c]);
            validate(d_divergence[c]);
        }
        d_gauge = gauge ? d_cells->getLocalElement(*gauge) : Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    Teuchos::RCP<const map_type> getDomainMap() const override { return d_map; }
    Teuchos::RCP<const map_type> getRangeMap() const override { return d_map; }
    bool hasTransposeApply() const override { return false; }
    size_t scratch_allocations() const noexcept { return d_allocations; }
    size_t scratch_payload_bytes() const noexcept
    {
        return 10 * d_cells->getLocalNumElements() * d_columns * sizeof(scalar_type);
    }

    void apply(const multi_vector_type& input, multi_vector_type& output, Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha = scalar_type{1}, scalar_type beta = scalar_type{}) const override
    {
        if (mode != Teuchos::NO_TRANS)
            throw std::invalid_argument("CoupledBlockOperator does not support transpose apply.");
        if (!input.getMap()->isSameAs(*d_map) || !output.getMap()->isSameAs(*d_map) ||
            input.getNumVectors() != output.getNumVectors())
            throw std::invalid_argument("CoupledBlockOperator input/output maps or column counts do not match.");
        if (alpha == scalar_type{})
        {
            if (beta == scalar_type{})
                output.putScalar(scalar_type{});
            else if (beta != scalar_type{1})
                output.scale(beta);
            return;
        }
        const size_t columns = input.getNumVectors();
        ensure_scratch(columns);
        const size_t cells = d_cells->getLocalNumElements();
        using execution_space = typename Pack::execution_space;
        using policy = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_t>>;
        {
            const auto u = d_u->getLocalViewDevice(Tpetra::Access::OverwriteAll);
            const auto q = d_q->getLocalViewDevice(Tpetra::Access::OverwriteAll);
            // getVector resolves nonconstant-stride column selections in the
            // linked Trilinos; raw MultiVector device views do not resolve them.
            for (size_t j = 0; j < columns; ++j)
            {
                const auto column = input.getVector(j);
                const auto x = column->getLocalViewDevice(Tpetra::Access::ReadOnly);
                Kokkos::parallel_for(
                    "CoupledBlockOperator::pack", policy(0, cells), KOKKOS_LAMBDA(size_t cell) {
                        for (size_t c = 0; c < 3; ++c)
                            u(cell, 3 * j + c) = x(4 * cell + c, 0);
                        q(cell, j) = x(4 * cell + 3, 0);
                    });
            }
            execution_space().fence();
        }
        d_momentum->apply(*d_u, *d_yu);
        d_stabilization->apply(*d_q, *d_yq);
        for (size_t c = 0; c < 3; ++c)
        {
            d_gradient[c]->apply(*d_q, *d_temp);
            {
                const auto g = d_temp->getLocalViewDevice(Tpetra::Access::ReadOnly);
                const auto u = d_u->getLocalViewDevice(Tpetra::Access::ReadOnly);
                const auto yu = d_yu->getLocalViewDevice(Tpetra::Access::ReadWrite);
                const auto component = d_component->getLocalViewDevice(Tpetra::Access::OverwriteAll);
                Kokkos::parallel_for(
                    "CoupledBlockOperator::component", policy(0, cells * columns), KOKKOS_LAMBDA(size_t k) {
                        const size_t cell = k / columns, j = k % columns;
                        yu(cell, 3 * j + c) += g(cell, j);
                        component(cell, j) = u(cell, 3 * j + c);
                    });
                execution_space().fence();
            }
            d_divergence[c]->apply(*d_component, *d_yq, Teuchos::NO_TRANS, scalar_type{1}, scalar_type{1});
        }
        const auto yu = d_yu->getLocalViewDevice(Tpetra::Access::ReadOnly);
        const auto yq = d_yq->getLocalViewDevice(Tpetra::Access::ReadOnly);
        const auto q = d_q->getLocalViewDevice(Tpetra::Access::ReadOnly);
        const auto gauge = d_gauge;
        for (size_t j = 0; j < columns; ++j)
        {
            auto column = output.getVectorNonConst(j);
            const auto y = beta == scalar_type{} ? column->getLocalViewDevice(Tpetra::Access::OverwriteAll)
                                                 : column->getLocalViewDevice(Tpetra::Access::ReadWrite);
            Kokkos::parallel_for(
                "CoupledBlockOperator::unpack", policy(0, cells), KOKKOS_LAMBDA(size_t cell) {
                    for (size_t c = 0; c < 4; ++c)
                    {
                        const auto value =
                            c < 3 ? yu(cell, 3 * j + c)
                                  : (static_cast<local_ordinal_type>(cell) == gauge ? q(cell, j) : yq(cell, j));
                        if (beta == scalar_type{})
                            y(4 * cell + c, 0) = alpha * value;
                        else
                            y(4 * cell + c, 0) = alpha * value + beta * y(4 * cell + c, 0);
                    }
                });
        }
        execution_space().fence();
    }

private:
    void ensure_scratch(size_t columns) const
    {
        if (columns == d_columns && !d_u.is_null())
            return;
        d_u = Teuchos::rcp(new multi_vector_type(d_cells, 3 * columns, false));
        d_yu = Teuchos::rcp(new multi_vector_type(d_cells, 3 * columns, false));
        d_q = Teuchos::rcp(new multi_vector_type(d_cells, columns, false));
        d_yq = Teuchos::rcp(new multi_vector_type(d_cells, columns, false));
        d_temp = Teuchos::rcp(new multi_vector_type(d_cells, columns, false));
        d_component = Teuchos::rcp(new multi_vector_type(d_cells, columns, false));
        d_columns = columns;
        ++d_allocations;
    }
    Teuchos::RCP<const map_type> d_map, d_cells;
    Teuchos::RCP<const matrix_type> d_momentum;
    std::array<Teuchos::RCP<matrix_type>, 3> d_gradient, d_divergence;
    Teuchos::RCP<const matrix_type> d_stabilization;
    std::shared_ptr<FVM::NumericAssemblyLease> d_lease;
    local_ordinal_type d_gauge;
    mutable Teuchos::RCP<multi_vector_type> d_u, d_yu, d_q, d_yq, d_temp, d_component;
    mutable size_t d_columns = 0, d_allocations = 0;
};
} // namespace SimpleFluid
