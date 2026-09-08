/**
 * @file DICPreconditioner.hh
 * @brief Serial diagonal incomplete-Cholesky inverse for CRS matrices.
 */
#pragma once

#include "dataclass/TpetraTypes.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::detail
{

/**
 * @brief Apply the diagonal incomplete-Cholesky inverse in local row order.
 *
 * For A = L + diag(a) + L^H, retain L unchanged and compute
 * d_i = a_i - sum_{j<i} |L_ij|^2 / d_j. The approximate factorization is
 * (diag(d) + L) diag(d)^-1 (diag(d) + L)^H. This is the diagonal-only
 * recurrence used by OpenFOAM DIC, rather than a general ILU(0) factorization.
 *
 * The complete row ordering must reside on one rank. Distributed matrices
 * are rejected on every rank before inspecting their local entries. An
 * incomplete factor can break down even for an SPD input; nonpositive or
 * nonfinite pivots fail explicitly rather than silently changing the factor.
 * Like the retained Belos solver, this object is not thread-safe: apply()
 * reuses mutable workspace and must not execute concurrently on one instance.
 * One instance retains a work vector and must not be applied concurrently.
 */
template<TpetraTypePack Pack>
class DICPreconditioner final : public Pack::operator_type
{
public:
    using scalar_type = typename Pack::scalar_type;
    using traits = Teuchos::ScalarTraits<scalar_type>;
    using magnitude_type = typename traits::magnitudeType;
    using map_type = typename Pack::map_type;
    using matrix_type = typename Pack::matrix_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    explicit DICPreconditioner(const matrix_type& matrix)
        : d_map(matrix.getRowMap())
    {
        if (d_map->getComm()->getSize() != 1)
        {
            throw std::invalid_argument(
                "DIC preconditioning currently requires a single MPI rank.");
        }
        if (!matrix.isFillComplete()
            || !d_map->isSameAs(*matrix.getDomainMap())
            || !d_map->isSameAs(*matrix.getRangeMap()))
        {
            throw std::invalid_argument(
                "DIC requires a fill-complete square matrix with matching "
                "row, domain, and range maps.");
        }

        struct Entry
        {
            local_ordinal_type column;
            scalar_type value;
        };
        const auto count = matrix.getLocalNumRows();
        std::vector<std::vector<Entry>> rows(count);
        d_inverse_diagonal.resize(count);
        d_workspace.resize(count);
        d_offsets.reserve(count + 1);
        d_offsets.push_back(0);

        for (std::size_t row = 0; row < count; ++row)
        {
            typename matrix_type::local_inds_host_view_type columns;
            typename matrix_type::values_host_view_type values;
            matrix.getLocalRowView(
                static_cast<local_ordinal_type>(row), columns, values);
            auto& entries = rows[row];
            entries.reserve(columns.size());
            for (std::size_t entry = 0; entry < columns.size(); ++entry)
            {
                const auto column = d_map->getLocalElement(
                    matrix.getColMap()->getGlobalElement(columns[entry]));
                if (column
                        == Teuchos::OrdinalTraits<local_ordinal_type>::invalid()
                    || !std::isfinite(traits::magnitude(values[entry])))
                {
                    throw std::invalid_argument(
                        "DIC requires finite entries within its row map.");
                }
                entries.push_back({column, values[entry]});
            }
            std::sort(entries.begin(), entries.end(),
                [](const Entry& first, const Entry& second)
                { return first.column < second.column; });
        }

        const auto symmetry_tolerance =
            magnitude_type{64} * std::numeric_limits<magnitude_type>::epsilon();
        for (std::size_t row = 0; row < count; ++row)
        {
            scalar_type diagonal{};
            bool found_diagonal = false;
            for (const auto& entry : rows[row])
            {
                if (entry.column == static_cast<local_ordinal_type>(row))
                {
                    diagonal = entry.value;
                    found_diagonal = true;
                    continue;
                }
                const auto& transpose_row = rows[entry.column];
                const auto transpose = std::lower_bound(
                    transpose_row.begin(), transpose_row.end(),
                    static_cast<local_ordinal_type>(row),
                    [](const Entry& value, local_ordinal_type column)
                    { return value.column < column; });
                const scalar_type transpose_value =
                    transpose != transpose_row.end()
                            && transpose->column
                                   == static_cast<local_ordinal_type>(row)
                        ? transpose->value : scalar_type{};
                if (traits::magnitude(entry.value
                        - traits::conjugate(transpose_value))
                    > symmetry_tolerance
                          * std::max(traits::magnitude(entry.value),
                                     traits::magnitude(transpose_value)))
                {
                    throw std::invalid_argument(
                        "DIC requires a symmetric (Hermitian) matrix.");
                }
                if (entry.column < static_cast<local_ordinal_type>(row)
                    && entry.value != scalar_type{})
                {
                    d_columns.push_back(entry.column);
                    d_values.push_back(entry.value);
                }
            }
            d_offsets.push_back(d_columns.size());
            for (auto entry = d_offsets[row];
                 entry < d_offsets[row + 1]; ++entry)
            {
                diagonal -= d_values[entry]
                    * traits::conjugate(d_values[entry])
                    * d_inverse_diagonal[d_columns[entry]];
            }
            if (!found_diagonal
                || !std::isfinite(traits::magnitude(diagonal))
                || traits::real(diagonal) <= magnitude_type{}
                || std::abs(traits::imag(diagonal))
                       > symmetry_tolerance * traits::magnitude(diagonal))
            {
                throw std::invalid_argument(
                    "DIC requires positive finite incomplete-Cholesky pivots.");
            }
            d_inverse_diagonal[row] = scalar_type{1} / diagonal;
            if (!std::isfinite(traits::magnitude(d_inverse_diagonal[row])))
            {
                throw std::invalid_argument(
                    "DIC incomplete-Cholesky pivot reciprocal overflowed.");
            }
        }
    }

    Teuchos::RCP<const map_type> getDomainMap() const override { return d_map; }
    Teuchos::RCP<const map_type> getRangeMap() const override { return d_map; }

    void apply(
        const multi_vector_type& input, multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha = scalar_type{1},
        scalar_type beta = scalar_type{}) const override
    {
        if (mode != Teuchos::NO_TRANS)
        {
            throw std::invalid_argument(
                "DIC preconditioner supports only NO_TRANS application.");
        }
        if (!d_map->isSameAs(*input.getMap())
            || !d_map->isSameAs(*output.getMap())
            || input.getNumVectors() != output.getNumVectors())
        {
            throw std::invalid_argument(
                "DIC input and output maps and vector counts must match.");
        }
        if (alpha == scalar_type{})
        {
            if (beta == scalar_type{})
                output.putScalar(scalar_type{});
            else
                output.scale(beta);
            return;
        }

        for (std::size_t column = 0; column < input.getNumVectors(); ++column)
        {
            // Stage each RHS before writing output, including aliased input.
            {
                const auto source = input.getData(column);
                for (std::size_t row = 0; row < d_workspace.size(); ++row)
                {
                    auto value = source[row];
                    for (auto entry = d_offsets[row];
                         entry < d_offsets[row + 1]; ++entry)
                        value -= d_values[entry] * d_workspace[d_columns[entry]];
                    d_workspace[row] = value * d_inverse_diagonal[row];
                }
            }
            for (auto row = d_workspace.size(); row-- > 0;)
            {
                for (auto entry = d_offsets[row];
                     entry < d_offsets[row + 1]; ++entry)
                {
                    const auto lower = d_columns[entry];
                    d_workspace[lower] -= traits::conjugate(d_values[entry])
                        * d_inverse_diagonal[lower] * d_workspace[row];
                }
            }
            auto destination = output.getDataNonConst(column);
            for (std::size_t row = 0; row < d_workspace.size(); ++row)
            {
                destination[row] = beta == scalar_type{}
                    ? alpha * d_workspace[row]
                    : alpha * d_workspace[row] + beta * destination[row];
            }
        }
    }

private:
    Teuchos::RCP<const map_type> d_map;
    std::vector<std::size_t> d_offsets;
    std::vector<local_ordinal_type> d_columns;
    std::vector<scalar_type> d_values;
    std::vector<scalar_type> d_inverse_diagonal;
    mutable std::vector<scalar_type> d_workspace;
};

} // namespace SimpleFluid::detail
