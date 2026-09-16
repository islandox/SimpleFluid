/**
 * @file DICPreconditioner.hh
 * @brief Diagonal incomplete-Cholesky inverse for serial and distributed CRS matrices.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include "dataclass/TpetraTypes.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <cstddef>
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
class SIMPLEFLUID_FVM_EXPORT DICPreconditioner final : public Pack::operator_type
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

    explicit DICPreconditioner(const matrix_type& matrix);

    Teuchos::RCP<const map_type> getDomainMap() const override { return d_map; }
    Teuchos::RCP<const map_type> getRangeMap() const override { return d_map; }

    void apply(
        const multi_vector_type& input, multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha = scalar_type{1},
        scalar_type beta = scalar_type{}) const override;

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

    SIMPLEFLUID_FVM_LOCAL
    static constexpr local_ordinal_type invalid_row()
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    /** Local map comparison: Map::isSameAs can take rank-dependent fast exits. */
    SIMPLEFLUID_FVM_LOCAL
    bool local_map_matches(const Teuchos::RCP<const map_type>& other) const;

    SIMPLEFLUID_FVM_LOCAL
    void require_collectively(bool valid, const char* message) const;

    /** Extract owned rows in global-column order without communication. */
    SIMPLEFLUID_FVM_LOCAL
    std::vector<std::vector<GlobalEntry>> global_rows(const matrix_type& matrix) const;

    /** Build a global DIC factor with only owned rows and column-map halos. */
    SIMPLEFLUID_FVM_LOCAL
    void initialize_distributed(const matrix_type& matrix);

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
    SIMPLEFLUID_FVM_LOCAL
    std::vector<StageTransfer> make_stage_transfers(
        const Teuchos::RCP<const map_type>& column_map,
        const std::vector<std::size_t>& offsets,
        std::vector<DistributedEntry>& entries) const;

    /** Replay the factor schedule using its directional remote-only plans. */
    SIMPLEFLUID_FVM_LOCAL
    void apply_distributed(const multi_vector_type& input, multi_vector_type& output,
        Teuchos::ETransp mode, scalar_type alpha, scalar_type beta) const;

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

extern template class DICPreconditioner<DefaultTpetraTypes>;

} // namespace SimpleFluid::detail
