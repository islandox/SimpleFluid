#pragma once

#include "dataclass/TpetraTypes.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{

/** Internal corrected pressure operator. Its correction callback is scoped to one solve. */
template<TpetraTypePack Pack>
class PressureProjectionOperator final : public Pack::operator_type
{
public:
    using scalar_type = typename Pack::scalar_type;
    using map_type = typename Pack::map_type;
    using matrix_type = typename Pack::matrix_type;
    using vector_type = typename Pack::vector_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using correction_type = std::function<void(const vector_type&, vector_type&)>;

    PressureProjectionOperator(Teuchos::RCP<const matrix_type> compact_matrix, correction_type correction)
        : d_compact_matrix(std::move(compact_matrix)), d_correction(std::move(correction)),
          d_correction_result(d_compact_matrix->getRangeMap(), false)
    {
    }

    Teuchos::RCP<const map_type> getDomainMap() const override { return d_compact_matrix->getDomainMap(); }
    Teuchos::RCP<const map_type> getRangeMap() const override { return d_compact_matrix->getRangeMap(); }
    bool hasTransposeApply() const override { return false; }

    /** Release solve-local captures while keeping retained operator maps valid. */
    void deactivate() noexcept { d_correction = {}; }

    std::size_t applications() const noexcept { return d_applications; }
    std::size_t correction_applications() const noexcept { return d_correction_applications; }

    void apply(const multi_vector_type& input, multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS, scalar_type alpha = scalar_type{1},
        scalar_type beta = scalar_type{}) const override
    {
        const bool input_map_matches = input.getMap()->isSameAs(*getDomainMap());
        const bool output_map_matches = output.getMap()->isSameAs(*getRangeMap());
        const int local_invalid = mode != Teuchos::NO_TRANS || !input_map_matches || !output_map_matches ||
            input.getNumVectors() != output.getNumVectors() || !d_correction;
        int invalid = 0;
        const auto comm = getDomainMap()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_invalid, &invalid);
        const auto columns = input.getNumVectors();
        decltype(input.getNumVectors()) minimum_columns{}, maximum_columns{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &columns, &minimum_columns);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &columns, &maximum_columns);
        if (invalid || minimum_columns != maximum_columns)
            throw std::invalid_argument("PressureProjectionOperator requires an active non-transpose "
                                        "application with matching maps and multivector columns on every rank.");
        if (alpha == scalar_type{})
        {
            if (beta == scalar_type{}) output.putScalar(scalar_type{});
            else output.scale(beta);
            return;
        }
        if (d_result.is_null() || d_result->getNumVectors() != columns)
            d_result = Teuchos::rcp(new multi_vector_type(getRangeMap(), columns, false));
        // Complete all reads before touching output, including overlapping subviews.
        d_compact_matrix->apply(input, *d_result);
        ++d_applications;
        for (std::size_t column = 0; column < columns; ++column)
        {
            const auto input_column = input.getVector(column);
            d_correction(*input_column, d_correction_result);
            ++d_correction_applications;
            d_result->getVectorNonConst(column)->update(-scalar_type{1}, d_correction_result, scalar_type{1});
        }
        output.update(alpha, *d_result, beta);
    }

private:
    Teuchos::RCP<const matrix_type> d_compact_matrix;
    correction_type d_correction;
    mutable vector_type d_correction_result;
    mutable Teuchos::RCP<multi_vector_type> d_result;
    mutable std::size_t d_applications = 0;
    mutable std::size_t d_correction_applications = 0;
};

} // namespace SimpleFluid::detail
