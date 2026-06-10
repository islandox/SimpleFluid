/**
 * @file CoupledEquation.hh
 * @brief Structural coupled-equation containers and solver backends.
 */

#pragma once

#include "dataclass/TpetraTypes.hh"

#include <Teuchos_RCP.hpp>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/** @brief Type-erased interface for an assembled diagonal equation block. */
template<TpetraTypePack Pack>
class AssembledEquationConcept
{
public:
    virtual ~AssembledEquationConcept() = default;
    virtual const std::string& name() const noexcept = 0;
    virtual Teuchos::RCP<const typename Pack::map_type>
    solution_map() const = 0;
    virtual bool solve() = 0;
};

/** @brief Type-erased owner for a concrete assembled equation. */
template<class EquationType, TpetraTypePack Pack>
class AssembledEquationModel final
    : public AssembledEquationConcept<Pack>
{
public:
    explicit AssembledEquationModel(EquationType equation)
        : d_equation(std::move(equation))
    {
    }

    const std::string& name() const noexcept override
    {
        return d_equation.name();
    }

    Teuchos::RCP<const typename Pack::map_type>
    solution_map() const override
    {
        return d_equation.solution().map();
    }

    bool solve() override { return d_equation.solve(); }

    EquationType& equation() noexcept { return d_equation; }

private:
    EquationType d_equation;
};

template<TpetraTypePack Pack = DefaultTpetraTypes>
class AssembledCoupledEquation;

/** @brief Strategy interface for solving an assembled block system. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class CoupledSolverBackend
{
public:
    virtual ~CoupledSolverBackend() = default;
    virtual bool solve(AssembledCoupledEquation<Pack>& equation) = 0;
};

/**
 * @brief Solve uncoupled diagonal blocks independently.
 *
 * This backend rejects systems containing off-diagonal matrices.
 */
template<TpetraTypePack Pack>
class IndependentBlockSolver final : public CoupledSolverBackend<Pack>
{
public:
    bool solve(AssembledCoupledEquation<Pack>& equation) override
    {
        if (!equation.off_diagonal_blocks().empty())
        {
            throw std::logic_error(
                "IndependentBlockSolver cannot solve off-diagonal "
                "couplings.");
        }
        bool converged = true;
        for (auto& block : equation.diagonal_blocks())
        {
            converged = block->solve() && converged;
        }
        return converged;
    }
};

/**
 * @brief Validated block representation of a coupled equation system.
 *
 * Diagonal blocks are type-erased assembled equations. Off-diagonal blocks
 * describe row-to-column coupling matrices interpreted by the selected
 * CoupledSolverBackend.
 */
template<TpetraTypePack Pack>
class AssembledCoupledEquation
{
public:
    using block_type = AssembledEquationConcept<Pack>;
    using matrix_type = typename Pack::matrix_type;

    /** @brief Matrix coupling one unknown block into another equation row. */
    struct OffDiagonalBlock
    {
        size_t row = 0;
        size_t column = 0;
        Teuchos::RCP<matrix_type> matrix;
    };

    /**
     * @brief Construct and validate an assembled block system.
     * @throws std::invalid_argument For empty or malformed block systems.
     * @throws std::out_of_range If a coupling block index is invalid.
     */
    AssembledCoupledEquation(
        std::vector<std::unique_ptr<block_type>> diagonal,
        std::vector<OffDiagonalBlock> off_diagonal,
        std::shared_ptr<CoupledSolverBackend<Pack>> backend =
            std::make_shared<IndependentBlockSolver<Pack>>())
        : d_diagonal(std::move(diagonal)),
          d_off_diagonal(std::move(off_diagonal)),
          d_backend(std::move(backend))
    {
        validate();
    }

    std::vector<std::unique_ptr<block_type>>&
    diagonal_blocks() noexcept
    {
        return d_diagonal;
    }

    const std::vector<std::unique_ptr<block_type>>&
    diagonal_blocks() const noexcept
    {
        return d_diagonal;
    }

    const std::vector<OffDiagonalBlock>&
    off_diagonal_blocks() const noexcept
    {
        return d_off_diagonal;
    }

    /**
     * @brief Replace the strategy used by solve().
     * @throws std::invalid_argument If @p backend is null.
     */
    void set_backend(std::shared_ptr<CoupledSolverBackend<Pack>> backend)
    {
        if (!backend)
        {
            throw std::invalid_argument(
                "Coupled solver backend cannot be null.");
        }
        d_backend = std::move(backend);
    }

    /**
     * @brief Delegate the block system to the configured backend.
     * @return Whether the backend reports convergence.
     */
    bool solve() { return d_backend->solve(*this); }

private:
    void validate() const
    {
        if (d_diagonal.empty())
        {
            throw std::invalid_argument(
                "Coupled equation requires at least one diagonal block.");
        }
        for (const auto& block : d_off_diagonal)
        {
            if (block.row >= d_diagonal.size()
                || block.column >= d_diagonal.size())
            {
                throw std::out_of_range(
                    "Coupled equation block index is out of range.");
            }
            if (block.row == block.column)
            {
                throw std::invalid_argument(
                    "Off-diagonal block cannot target the diagonal.");
            }
            if (block.matrix.is_null())
            {
                throw std::invalid_argument(
                    "Off-diagonal block matrix cannot be null.");
            }
            if (!block.matrix->getRowMap()->isSameAs(
                    *d_diagonal[block.row]->solution_map()))
            {
                throw std::invalid_argument(
                    "Off-diagonal block row map mismatch.");
            }
        }
    }

    std::vector<std::unique_ptr<block_type>> d_diagonal;
    std::vector<OffDiagonalBlock> d_off_diagonal;
    std::shared_ptr<CoupledSolverBackend<Pack>> d_backend;
};

/**
 * @brief Deferred builder for a coupled equation system.
 *
 * Diagonal builders are invoked by assemble(), allowing each equation to
 * produce its matrix and right-hand side at assembly time.
 */
template<TpetraTypePack Pack>
class CoupledEquation
{
public:
    using assembled_type = AssembledCoupledEquation<Pack>;
    using block_type = AssembledEquationConcept<Pack>;
    using builder_type = std::function<std::unique_ptr<block_type>()>;

    /**
     * @brief Append a diagonal block builder.
     * @throws std::invalid_argument If @p builder is empty.
     */
    CoupledEquation& add_diagonal(builder_type builder)
    {
        if (!builder)
        {
            throw std::invalid_argument(
                "Coupled equation block builder cannot be empty.");
        }
        d_builders.push_back(std::move(builder));
        return *this;
    }

    CoupledEquation& add_off_diagonal(
        typename assembled_type::OffDiagonalBlock block)
    {
        d_off_diagonal.push_back(std::move(block));
        return *this;
    }

    /** @brief Materialize all blocks and validate the assembled system. */
    assembled_type assemble(
        std::shared_ptr<CoupledSolverBackend<Pack>> backend =
            std::make_shared<IndependentBlockSolver<Pack>>()) const
    {
        std::vector<std::unique_ptr<block_type>> diagonal;
        diagonal.reserve(d_builders.size());
        for (const auto& builder : d_builders)
        {
            diagonal.push_back(builder());
        }
        return assembled_type(
            std::move(diagonal), d_off_diagonal, std::move(backend));
    }

private:
    std::vector<builder_type> d_builders;
    std::vector<typename assembled_type::OffDiagonalBlock> d_off_diagonal;
};

/** @brief Wrap a concrete equation in the coupled type-erasure API. */
template<class EquationType, TpetraTypePack Pack>
std::unique_ptr<AssembledEquationConcept<Pack>>
make_assembled_equation_model(EquationType equation)
{
    return std::make_unique<
        AssembledEquationModel<EquationType, Pack>>(
            std::move(equation));
}

} // namespace SimpleFluid
