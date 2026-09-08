/**
 * @file DICPreconditioner.hh
 * @brief Diagonal incomplete-Cholesky inverse for serial and distributed CRS matrices.
 */
#pragma once

#include "dataclass/TpetraTypes.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Tpetra_RowMatrixTransposer.hpp>
#include <Tpetra_Util.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid::detail
{

template<TpetraTypePack Pack>
struct DICPreconditionerTestAccess;

/**
 * @brief Apply a diagonal incomplete-Cholesky inverse with all matrix couplings.
 *
 * For A = L + diag(a) + L^H, retain L unchanged and compute
 * d_i = a_i - sum_{j<i} |L_ij|^2 / d_j. The approximate factorization is
 * (diag(d) + L) diag(d)^-1 (diag(d) + L)^H. This is the diagonal-only
 * recurrence used by OpenFOAM DIC, rather than a general ILU(0) factorization.
 *
 * Serial matrices retain local row order. Distributed matrices use ascending
 * global row IDs. Tpetra imports communicate inverse pivots during setup and
 * triangular-solve values during apply; the matrix is never gathered. Each
 * communication stage completes every locally ready row before exchanging
 * remote dependencies. Application retains directional, remote-only imports
 * for the dependencies consumed in each stage. Stage count depends on the
 * partition and ordering.
 * An incomplete factor can break down even for an SPD input; nonpositive or
 * nonfinite pivots fail collectively rather than silently changing the factor.
 * Like the retained Belos solver, this object is not thread-safe: apply()
 * reuses mutable workspace and must not execute concurrently on one instance.
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
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using import_type = typename Pack::import_type;

    explicit DICPreconditioner(const matrix_type& matrix)
        : d_map(matrix.getRowMap())
    {
        if (d_map.is_null())
        {
            throw std::invalid_argument("DIC requires a non-null row map.");
        }
        if (d_map->getComm()->getSize() > 1)
        {
            initialize_distributed(matrix);
            d_distributed = true;
            return;
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
        if (d_distributed)
        {
            apply_distributed(input, output, mode, alpha, beta);
            return;
        }
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
    friend struct DICPreconditionerTestAccess<Pack>;

    struct GlobalEntry
    {
        global_ordinal_type column;
        scalar_type value;
    };

    struct DistributedEntry
    {
        local_ordinal_type column;
        local_ordinal_type local_row;
        scalar_type value;
    };

    struct StageTransfer
    {
        Teuchos::RCP<const import_type> importer;
        mutable Teuchos::RCP<multi_vector_type> halo;
    };

    static constexpr local_ordinal_type invalid_row()
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    /** Local map comparison: Map::isSameAs can take rank-dependent fast exits. */
    bool local_map_matches(const Teuchos::RCP<const map_type>& other) const
    {
        if (other.is_null() || other->getComm().is_null()
            || other->getComm()->getSize() != d_map->getComm()->getSize()
            || other->getComm()->getRank() != d_map->getComm()->getRank()
            || other->getGlobalNumElements() != d_map->getGlobalNumElements()
            || other->getIndexBase() != d_map->getIndexBase()
            || other->getLocalNumElements() != d_map->getLocalNumElements())
            return false;
        if (other->getComm().getRawPtr() != d_map->getComm().getRawPtr()
            && !Tpetra::Details::congruent(*other->getComm(), *d_map->getComm()))
            return false;
        const auto expected = d_map->getLocalElementList();
        const auto actual = other->getLocalElementList();
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    void require_collectively(bool valid, const char* message) const
    {
        const int local_invalid = valid ? 0 : 1;
        int invalid = 0;
        Teuchos::reduceAll(*d_map->getComm(), Teuchos::REDUCE_MAX,
            1, &local_invalid, &invalid);
        if (invalid != 0)
            throw std::invalid_argument(message);
    }

    /** Extract owned rows in global-column order without communication. */
    std::vector<std::vector<GlobalEntry>> global_rows(const matrix_type& matrix) const
    {
        std::vector<std::vector<GlobalEntry>> rows(d_map->getLocalNumElements());
        for (std::size_t row = 0; row < rows.size(); ++row)
        {
            const auto gid = d_map->getGlobalElement(static_cast<local_ordinal_type>(row));
            const auto matrix_row = matrix.getRowMap()->getLocalElement(gid);
            typename matrix_type::local_inds_host_view_type columns;
            typename matrix_type::values_host_view_type values;
            matrix.getLocalRowView(matrix_row, columns, values);
            rows[row].reserve(columns.size());
            for (std::size_t entry = 0; entry < columns.size(); ++entry)
                rows[row].push_back({matrix.getColMap()->getGlobalElement(columns[entry]), values[entry]});
            std::sort(rows[row].begin(), rows[row].end(),
                [](const auto& first, const auto& second) { return first.column < second.column; });
        }
        return rows;
    }

    /** Build a global DIC factor with only owned rows and column-map halos. */
    void initialize_distributed(const matrix_type& matrix)
    {
        require_collectively(matrix.isFillComplete() && !matrix.getColMap().is_null()
                && local_map_matches(matrix.getDomainMap())
                && local_map_matches(matrix.getRangeMap()),
            "Distributed DIC requires a fill-complete square matrix with matching row, domain and range maps.");
        const bool unique_ownership = d_map->isOneToOne();
        require_collectively(unique_ownership, "Distributed DIC requires one-to-one row ownership.");
        const auto column_map = matrix.getColMap();
        Teuchos::Array<int> owners(column_map->getLocalNumElements());
        const auto lookup = d_map->getRemoteIndexList(column_map->getLocalElementList(), owners());
        require_collectively(lookup != Tpetra::IDNotPresent,
            "Distributed DIC matrix columns must belong to its global row map.");

        const auto rows = global_rows(matrix);
        bool valid_entries = true;
        for (const auto& row : rows)
            for (const auto& entry : row)
                valid_entries = valid_entries && std::isfinite(traits::magnitude(entry.value));
        require_collectively(valid_entries, "Distributed DIC requires finite matrix entries.");

        // Transpose only the distributed sparse matrix; no rank owns a global copy.
        Tpetra::RowMatrixTransposer<scalar_type, local_ordinal_type,
            global_ordinal_type, typename Pack::node_type> transposer(Teuchos::rcpFromRef(matrix));
        auto parameters = Teuchos::rcp(new Teuchos::ParameterList());
        parameters->set("conjugate values", false);
        const auto transpose = transposer.createTranspose(parameters);
        require_collectively(local_map_matches(transpose->getRowMap()),
            "Distributed DIC transpose must preserve row ownership.");
        const auto transposed_rows = global_rows(*transpose);

        const auto count = rows.size();
        const auto tolerance = magnitude_type{64} * std::numeric_limits<magnitude_type>::epsilon();
        std::vector<scalar_type> diagonal(count, scalar_type{});
        d_lower_offsets.push_back(0);
        d_upper_offsets.push_back(0);
        bool symmetric = true;
        bool valid_diagonal = true;
        bool valid_columns = true;
        for (std::size_t row = 0; row < count; ++row)
        {
            const auto gid = d_map->getGlobalElement(static_cast<local_ordinal_type>(row));
            const auto& original = rows[row];
            const auto& transposed = transposed_rows[row];
            std::size_t a = 0, t = 0;
            while (a < original.size() || t < transposed.size())
            {
                scalar_type value{}, reverse{};
                if (t == transposed.size() || (a < original.size() && original[a].column < transposed[t].column))
                    value = original[a++].value;
                else if (a == original.size() || transposed[t].column < original[a].column)
                    reverse = transposed[t++].value;
                else
                {
                    value = original[a++].value;
                    reverse = transposed[t++].value;
                }
                const auto difference = traits::magnitude(value - traits::conjugate(reverse));
                symmetric = symmetric && std::isfinite(difference)
                    && difference <= tolerance * std::max(traits::magnitude(value), traits::magnitude(reverse));
            }
            bool found_diagonal = false;
            for (const auto& entry : original)
            {
                if (entry.column == gid)
                {
                    diagonal[row] = entry.value;
                    found_diagonal = true;
                }
                else if (entry.column < gid && entry.value != scalar_type{})
                {
                    const auto column = column_map->getLocalElement(entry.column);
                    valid_columns = valid_columns && column != invalid_row();
                    d_lower.push_back({column, d_map->getLocalElement(entry.column), entry.value});
                }
            }
            // Reverse global order matches serial backward subtraction order.
            for (auto entry = transposed.rbegin(); entry != transposed.rend(); ++entry)
            {
                if (entry->column > gid && entry->value != scalar_type{})
                {
                    const auto column = column_map->getLocalElement(entry->column);
                    valid_columns = valid_columns && column != invalid_row();
                    d_upper.push_back({column, d_map->getLocalElement(entry->column), traits::conjugate(entry->value)});
                }
            }
            valid_diagonal = valid_diagonal && found_diagonal;
            d_lower_offsets.push_back(d_lower.size());
            d_upper_offsets.push_back(d_upper.size());
        }
        require_collectively(symmetric, "Distributed DIC requires a symmetric (Hermitian) matrix across all ranks.");
        require_collectively(valid_diagonal && valid_columns,
            "Distributed DIC requires diagonal entries and complete factor column maps.");

        const import_type pivot_import(d_map, column_map);
        multi_vector_type pivots(d_map, 1, true);
        multi_vector_type halo_pivots(column_map, 1, true);
        d_inverse_diagonal.assign(count, scalar_type{});
        std::vector<local_ordinal_type> ordered_rows(count);
        std::iota(ordered_rows.begin(), ordered_rows.end(), local_ordinal_type{});
        std::sort(ordered_rows.begin(), ordered_rows.end(), [&](auto first, auto second)
            { return d_map->getGlobalElement(first) < d_map->getGlobalElement(second); });
        std::size_t remaining = count;
        d_stage_offsets.push_back(0);
        while (true)
        {
            int invalid_pivot = 0;
            int progressed = 0;
            {
                const auto remote = halo_pivots.getLocalViewHost(Tpetra::Access::ReadOnly);
                const auto owned = pivots.getLocalViewHost(Tpetra::Access::ReadWrite);
                for (const auto row : ordered_rows)
                {
                    if (d_inverse_diagonal[row] != scalar_type{})
                        continue;
                    bool ready = true;
                    for (auto index = d_lower_offsets[row]; index < d_lower_offsets[row + 1]; ++index)
                    {
                        const auto& entry = d_lower[index];
                        const auto inverse = entry.local_row == invalid_row()
                            ? remote(entry.column, 0) : d_inverse_diagonal[entry.local_row];
                        ready = ready && inverse != scalar_type{};
                    }
                    if (!ready)
                        continue;
                    auto pivot = diagonal[row];
                    for (auto index = d_lower_offsets[row]; index < d_lower_offsets[row + 1]; ++index)
                    {
                        const auto& entry = d_lower[index];
                        const auto inverse = entry.local_row == invalid_row()
                            ? remote(entry.column, 0) : d_inverse_diagonal[entry.local_row];
                        pivot -= entry.value * traits::conjugate(entry.value) * inverse;
                    }
                    if (!std::isfinite(traits::magnitude(pivot)) || traits::real(pivot) <= magnitude_type{}
                        || std::abs(traits::imag(pivot)) > tolerance * traits::magnitude(pivot))
                    {
                        invalid_pivot = 1;
                        break;
                    }
                    const auto inverse = scalar_type{1} / pivot;
                    if (!std::isfinite(traits::magnitude(inverse)) || inverse == scalar_type{})
                    {
                        invalid_pivot = 1;
                        break;
                    }
                    d_inverse_diagonal[row] = inverse;
                    owned(row, 0) = inverse;
                    d_stage_rows.push_back(row);
                    --remaining;
                    progressed = 1;
                }
            }
            const std::array<int, 3> local{invalid_pivot, remaining != 0 ? 1 : 0, progressed};
            std::array<int, 3> global{};
            Teuchos::reduceAll(*d_map->getComm(), Teuchos::REDUCE_MAX,
                static_cast<int>(local.size()), local.data(), global.data());
            if (global[0] != 0)
                throw std::invalid_argument("Distributed DIC requires positive finite incomplete-Cholesky pivots.");
            d_stage_offsets.push_back(d_stage_rows.size());
            if (global[1] == 0)
                break;
            if (global[2] == 0)
                throw std::invalid_argument("Distributed DIC factor dependencies made no progress.");
            halo_pivots.doImport(pivots, pivot_import, Tpetra::INSERT);
        }
        d_forward_transfers = make_stage_transfers(column_map, d_lower_offsets, d_lower);
        d_backward_transfers = make_stage_transfers(column_map, d_upper_offsets, d_upper);
    }

    /**
     * Retain only remote values consumed by each triangular-solve stage.
     *
     * The established global factor schedule guarantees that every requested
     * value has already been computed. Owned dependencies stay in the work
     * vector; opposite-direction and other-stage halo values are not copied or
     * sent. Remapping column slots leaves row and subtraction order unchanged.
     * All ranks construct every plan, including ranks with no receiving rows:
     * those ranks may still export values needed by another rank.
     */
    std::vector<StageTransfer> make_stage_transfers(
        const Teuchos::RCP<const map_type>& column_map,
        const std::vector<std::size_t>& offsets,
        std::vector<DistributedEntry>& entries) const
    {
        const auto stages = d_stage_offsets.size() - 1;
        std::vector<StageTransfer> transfers;
        transfers.reserve(stages);
        for (std::size_t stage = 0; stage < stages; ++stage)
        {
            Teuchos::Array<global_ordinal_type> remote_ids;
            for (auto position = d_stage_offsets[stage]; position < d_stage_offsets[stage + 1]; ++position)
            {
                const auto row = d_stage_rows[position];
                for (auto index = offsets[row]; index < offsets[row + 1]; ++index)
                    if (entries[index].local_row == invalid_row())
                        remote_ids.push_back(column_map->getGlobalElement(entries[index].column));
            }
            std::sort(remote_ids.begin(), remote_ids.end());
            remote_ids.erase(std::unique(remote_ids.begin(), remote_ids.end()), remote_ids.end());
            const auto target = Teuchos::rcp(new map_type(
                Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                remote_ids(), d_map->getIndexBase(), d_map->getComm()));
            transfers.push_back({Teuchos::rcp(new import_type(d_map, target)), Teuchos::null});
            for (auto position = d_stage_offsets[stage]; position < d_stage_offsets[stage + 1]; ++position)
            {
                const auto row = d_stage_rows[position];
                for (auto index = offsets[row]; index < offsets[row + 1]; ++index)
                    if (entries[index].local_row == invalid_row())
                        entries[index].column = target->getLocalElement(
                            column_map->getGlobalElement(entries[index].column));
            }
        }
        return transfers;
    }

    /** Replay the factor schedule using its directional remote-only plans. */
    void apply_distributed(const multi_vector_type& input, multi_vector_type& output,
        Teuchos::ETransp mode, scalar_type alpha, scalar_type beta) const
    {
        const auto vectors = input.getNumVectors();
        const bool count_fits = vectors <= static_cast<std::size_t>(std::numeric_limits<int>::max());
        const int count = count_fits ? static_cast<int>(vectors) : 0;
        const int zero_alpha = alpha == scalar_type{} ? 1 : 0;
        const std::array<int, 5> local{
            mode != Teuchos::NO_TRANS || !local_map_matches(input.getMap())
                    || !local_map_matches(output.getMap()) || vectors != output.getNumVectors() || !count_fits ? 1 : 0,
            count, -count, zero_alpha, -zero_alpha};
        std::array<int, 5> global{};
        Teuchos::reduceAll(*d_map->getComm(), Teuchos::REDUCE_MAX,
            static_cast<int>(local.size()), local.data(), global.data());
        if (global[0] != 0 || global[1] != -global[2] || global[3] != -global[4])
            throw std::invalid_argument("Distributed DIC requires matching maps, NO_TRANS, and rank-consistent "
                                        "vector counts and zero-alpha selection.");
        if (zero_alpha != 0)
        {
            if (beta == scalar_type{})
                output.putScalar(scalar_type{});
            else
                output.scale(beta);
            return;
        }
        if (vectors == 0)
            return;
        if (d_distributed_work.is_null() || d_distributed_work->getNumVectors() != vectors)
        {
            d_distributed_work = Teuchos::rcp(new multi_vector_type(d_map, vectors, false));
            for (const auto* transfers : {&d_forward_transfers, &d_backward_transfers})
                for (const auto& transfer : *transfers)
                    transfer.halo = Teuchos::rcp(new multi_vector_type(
                        transfer.importer->getTargetMap(), vectors, false));
        }
        // Copy every input before publishing any output, including overlapping
        // differently permuted column views. getData honors nonconstant stride.
        for (std::size_t column = 0; column < vectors; ++column)
        {
            const auto source = input.getData(column);
            auto destination = d_distributed_work->getDataNonConst(column);
            std::copy(source.begin(), source.end(), destination.begin());
        }
        const auto stages = d_stage_offsets.size() - 1;
        for (std::size_t stage = 0; stage < stages; ++stage)
        {
            const auto& transfer = d_forward_transfers[stage];
            if (stage != 0)
                transfer.halo->doImport(*d_distributed_work, *transfer.importer, Tpetra::INSERT);
            const auto owned = d_distributed_work->getLocalViewHost(Tpetra::Access::ReadWrite);
            const auto remote = transfer.halo->getLocalViewHost(Tpetra::Access::ReadOnly);
            for (auto position = d_stage_offsets[stage]; position < d_stage_offsets[stage + 1]; ++position)
            {
                const auto row = d_stage_rows[position];
                for (std::size_t column = 0; column < vectors; ++column)
                {
                    auto value = owned(row, column);
                    for (auto index = d_lower_offsets[row]; index < d_lower_offsets[row + 1]; ++index)
                    {
                        const auto& entry = d_lower[index];
                        value -= entry.value * (entry.local_row == invalid_row()
                            ? remote(entry.column, column) : owned(entry.local_row, column));
                    }
                    owned(row, column) = value * d_inverse_diagonal[row];
                }
            }
        }
        for (auto stage = stages; stage-- > 0;)
        {
            const auto& transfer = d_backward_transfers[stage];
            if (stage + 1 != stages)
                transfer.halo->doImport(*d_distributed_work, *transfer.importer, Tpetra::INSERT);
            const auto owned = d_distributed_work->getLocalViewHost(Tpetra::Access::ReadWrite);
            const auto remote = transfer.halo->getLocalViewHost(Tpetra::Access::ReadOnly);
            for (auto position = d_stage_offsets[stage + 1]; position-- > d_stage_offsets[stage];)
            {
                const auto row = d_stage_rows[position];
                for (std::size_t column = 0; column < vectors; ++column)
                {
                    auto value = owned(row, column);
                    for (auto index = d_upper_offsets[row]; index < d_upper_offsets[row + 1]; ++index)
                    {
                        const auto& entry = d_upper[index];
                        value -= entry.value * d_inverse_diagonal[row] * (entry.local_row == invalid_row()
                            ? remote(entry.column, column) : owned(entry.local_row, column));
                    }
                    owned(row, column) = value;
                }
            }
        }
        for (std::size_t column = 0; column < vectors; ++column)
        {
            const auto source = d_distributed_work->getData(column);
            auto destination = output.getDataNonConst(column);
            for (std::size_t row = 0; row < d_map->getLocalNumElements(); ++row)
                destination[row] = beta == scalar_type{} ? alpha * source[row]
                    : alpha * source[row] + beta * destination[row];
        }
    }

    Teuchos::RCP<const map_type> d_map;
    std::vector<std::size_t> d_offsets;
    std::vector<local_ordinal_type> d_columns;
    std::vector<scalar_type> d_values;
    std::vector<scalar_type> d_inverse_diagonal;
    mutable std::vector<scalar_type> d_workspace;
    bool d_distributed = false;
    std::vector<std::size_t> d_lower_offsets, d_upper_offsets;
    std::vector<DistributedEntry> d_lower, d_upper;
    std::vector<std::size_t> d_stage_offsets;
    std::vector<local_ordinal_type> d_stage_rows;
    std::vector<StageTransfer> d_forward_transfers, d_backward_transfers;
    mutable Teuchos::RCP<multi_vector_type> d_distributed_work;
};

} // namespace SimpleFluid::detail
