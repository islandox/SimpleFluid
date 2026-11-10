/** @file SteadyDiagnostics.hh Read-only, owned-row steady residual reductions. */
#pragma once
#include "FVM/TransportSystem.hh"
#include <Tpetra_Import.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace SimpleFluid::FVM
{
namespace steady_diagnostic_detail
{
#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__) || defined(_M_FP_FAST) \
    || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
inline constexpr bool standard_arithmetic_compilation = false;
#else
inline constexpr bool standard_arithmetic_compilation = true;
#endif

template<class Scalar> bool normal_or_zero(Scalar value)
{
    // Inspect supported IEEE encodings directly. DAZ can affect floating comparisons.
    if constexpr (std::is_floating_point_v<Scalar> && sizeof(Scalar) == sizeof(std::uint64_t)
        && std::numeric_limits<Scalar>::is_iec559 && std::numeric_limits<Scalar>::digits == 53)
    {
        const auto magnitude = std::bit_cast<std::uint64_t>(value) & UINT64_C(0x7fffffffffffffff);
        return magnitude == 0
            || (magnitude > UINT64_C(0x0010000000000000) && magnitude < UINT64_C(0x7ff0000000000000));
    }
    else if constexpr (std::is_floating_point_v<Scalar> && sizeof(Scalar) == sizeof(std::uint32_t)
        && std::numeric_limits<Scalar>::is_iec559 && std::numeric_limits<Scalar>::digits == 24)
    {
        const auto magnitude = std::bit_cast<std::uint32_t>(value) & UINT32_C(0x7fffffff);
        return magnitude == 0
            || (magnitude > UINT32_C(0x00800000) && magnitude < UINT32_C(0x7f800000));
    }
    else
    {
        const auto category = std::fpclassify(value);
        return category == FP_ZERO
            || (category == FP_NORMAL && std::abs(value) > std::numeric_limits<Scalar>::min());
    }
}
}

/** Residuals have integrated equation units; densities divide by cell volume.
 * scale is the volume-weighted RMS of (sum_j |Aij qj| + |bi|)/Vi.
 * The scale is algebraic: it does not claim a physical-term decomposition.
 * Excluded rows (for example wall constraints) contribute to none of the norms.
 * Optional cell arrays retain owned-map indexing, with zero for excluded rows.
 * integrated and algebraic_scale have integrated equation units. density divides by volume.
 * accumulation_terms counts stored products. Twice this count bounds product/add operations.
 * accumulation_eligible marks rows satisfying the relative-error arithmetic assumptions.
 * Caller-provided integrated reports leave row accumulation metadata unavailable.
 */
template<class Scalar> struct SteadyResidualReport
{
    Scalar signed_sum{}, volume_rms{}, max_density{}, scale{}, dimensionless{}, volume{};
    bool dimensionless_available = true;
    std::size_t included_count{}, excluded_count{};
    static constexpr std::string_view scale_kind = "algebraic_absolute_row_terms";
    std::vector<Scalar> integrated, density, algebraic_scale;
    std::vector<std::size_t> accumulation_terms;
    std::vector<unsigned char> accumulation_eligible;
    bool row_accumulation_available = false;
    bool evaluation_round_to_nearest = std::fegetround() == FE_TONEAREST;
};

/** Reduce caller-provided integrated conservation defects and absolute algebraic scales.
 * All arrays use owned-cell ordering. Scales are integrated, nonnegative quantities.
 */
template<class Scalar, class Comm>
SteadyResidualReport<Scalar> integrated_residual_report(
    const std::vector<Scalar>& residual, const std::vector<Scalar>& algebraic_scale,
    const std::vector<Scalar>& volumes, const Comm& comm, bool include_cells = false,
    const std::vector<unsigned char>& excluded = {})
{
    int invalid = residual.size() != volumes.size() || algebraic_scale.size() != residual.size()
        || (!excluded.empty() && excluded.size() != residual.size());
    int any_invalid{};
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &invalid, &any_invalid);
    if (any_invalid) throw std::invalid_argument("Integrated residual arrays require equal owned-cell extents.");
    SteadyResidualReport<Scalar> result;
    Scalar sums[6]{}, totals[6]{}, maximum{};
    if (include_cells)
    { result.integrated.resize(residual.size()); result.density.resize(residual.size()); result.algebraic_scale.resize(residual.size()); }
    for (std::size_t i = 0; i < residual.size(); ++i)
    {
        if (!std::isfinite(residual[i]) || !std::isfinite(algebraic_scale[i]) || algebraic_scale[i] < Scalar{}
            || !std::isfinite(volumes[i]) || !(volumes[i] > Scalar{})) { invalid = 1; continue; }
        if (!excluded.empty() && excluded[i]) { sums[5] += Scalar{1}; continue; }
        const auto density = residual[i] / volumes[i], scale_density = algebraic_scale[i] / volumes[i];
        sums[0] += residual[i]; sums[1] += density * density * volumes[i];
        sums[2] += scale_density * scale_density * volumes[i]; sums[3] += volumes[i]; sums[4] += Scalar{1};
        maximum = std::max(maximum, std::abs(density));
        if (include_cells)
        { result.integrated[i] = residual[i]; result.density[i] = density; result.algebraic_scale[i] = algebraic_scale[i]; }
    }
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 6, sums, totals);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &maximum, &result.max_density);
    for (const auto value : totals) if (!std::isfinite(value)) invalid = 1;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &invalid, &any_invalid);
    if (any_invalid) throw std::invalid_argument("Integrated residual encountered invalid or nonfinite input.");
    result.signed_sum = totals[0]; result.volume = totals[3];
    result.included_count = static_cast<std::size_t>(totals[4]); result.excluded_count = static_cast<std::size_t>(totals[5]);
    if (result.volume > Scalar{})
    { result.volume_rms = std::sqrt(totals[1] / result.volume); result.scale = std::sqrt(totals[2] / result.volume); }
    result.dimensionless_available = result.scale > Scalar{} || result.volume_rms == Scalar{};
    result.dimensionless = result.scale > Scalar{} ? result.volume_rms / result.scale : Scalar{};
    return result;
}

/** Evaluate A q - b without modifying the operator, fields, or any solver cache.
 * Collective over the matrix communicator, including ranks with no owned rows.
 * Input volumes and optional exclusion mask use the matrix owned-row ordering.
 * Zero residual and zero scale give ratio zero; a nonzero residual with zero
 * scale has dimensionless_available=false and ratio zero. Nonfinite inputs/results reject on all ranks.
 */
template<TpetraTypePack Pack>
SteadyResidualReport<typename Pack::scalar_type> steady_residual_report(
    const typename Pack::matrix_type& matrix, const typename Pack::vector_type& rhs,
    const typename Pack::vector_type& values, const std::vector<typename Pack::scalar_type>& owned_volumes,
    const std::vector<unsigned char>& excluded = {}, bool include_cells = false)
{
    using Scalar = typename Pack::scalar_type;
    using LO = typename Pack::local_ordinal_type;
    using GO = typename Pack::global_ordinal_type;
    using Node = typename Pack::node_type;
    const auto comm = matrix.getRowMap()->getComm();
    const auto n = matrix.getLocalNumRows();
    int invalid = !matrix.isFillComplete() || !matrix.getRowMap()->isSameAs(*matrix.getRangeMap()) || owned_volumes.size() != n
        || (!excluded.empty() && excluded.size() != n)
        || !rhs.getMap()->isSameAs(*matrix.getRangeMap())
        || !values.getMap()->isSameAs(*matrix.getDomainMap());
    int any_invalid = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &invalid, &any_invalid);
    if (any_invalid) throw std::invalid_argument("Steady residual requires fill-complete compatible maps, volumes and mask.");
    typename Pack::vector_type columns(matrix.getColMap());
    Tpetra::Import<LO, GO, Node> importer(matrix.getDomainMap(), matrix.getColMap());
    columns.doImport(values, importer, Tpetra::INSERT);
    const auto q = columns.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto b = rhs.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (std::size_t i = 0; i < q.extent(0); ++i)
        if (!std::isfinite(q(i, 0))) invalid = 1;
    SteadyResidualReport<Scalar> report;
    report.row_accumulation_available = true;
    if (include_cells)
    {
        report.integrated.resize(n); report.density.resize(n); report.algebraic_scale.resize(n);
        report.accumulation_terms.resize(n);
        report.accumulation_eligible.resize(n);
    }
    Scalar sums[6]{}; // signed, density squared * V, scale squared * V, V, included, excluded
    Scalar maximum{};
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto volume = owned_volumes[i];
        Scalar residual = -b(i, 0), scale = std::abs(b(i, 0));
        bool accumulation_eligible = report.evaluation_round_to_nearest
            && std::numeric_limits<Scalar>::is_iec559 && std::numeric_limits<Scalar>::radix == 2
            && steady_diagnostic_detail::standard_arithmetic_compilation
            && steady_diagnostic_detail::normal_or_zero(b(i, 0));
        typename Pack::matrix_type::local_inds_host_view_type indices;
        typename Pack::matrix_type::values_host_view_type coefficients;
        matrix.getLocalRowView(static_cast<LO>(i), indices, coefficients);
        for (std::size_t j = 0; j < indices.extent(0); ++j)
        {
            const auto term = coefficients(j) * q(indices(j), 0);
            const auto previous_residual = residual;
            residual += term; scale += std::abs(term);
            accumulation_eligible = accumulation_eligible
                && steady_diagnostic_detail::normal_or_zero(coefficients(j))
                && steady_diagnostic_detail::normal_or_zero(q(indices(j), 0))
                && steady_diagnostic_detail::normal_or_zero(term)
                && steady_diagnostic_detail::normal_or_zero(residual)
                && steady_diagnostic_detail::normal_or_zero(scale);
            if (std::fpclassify(term) == FP_ZERO
                && std::fpclassify(coefficients(j)) != FP_ZERO
                && std::fpclassify(q(indices(j), 0)) != FP_ZERO)
                accumulation_eligible = false;
            if (std::fpclassify(residual) == FP_ZERO && previous_residual != -term)
                accumulation_eligible = false;
        }
        if (!(volume > Scalar{}) || !std::isfinite(volume) || !std::isfinite(residual) || !std::isfinite(scale))
        { invalid = 1; continue; }
        if (!excluded.empty() && excluded[i]) { sums[5] += Scalar{1}; continue; }
        const auto density = residual / volume;
        const auto scale_density = scale / volume;
        sums[0] += residual; sums[1] += density * density * volume;
        sums[2] += scale_density * scale_density * volume; sums[3] += volume; sums[4] += Scalar{1};
        maximum = std::max(maximum, std::abs(density));
        if (include_cells)
        {
            report.integrated[i] = residual; report.density[i] = density; report.algebraic_scale[i] = scale;
            report.accumulation_terms[i] = indices.extent(0);
            report.accumulation_eligible[i] = static_cast<unsigned char>(accumulation_eligible);
        }
    }
    Scalar totals[6]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 6, sums, totals);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &maximum, &report.max_density);
    for (const auto value : totals) if (!std::isfinite(value)) invalid = 1;
    if (!std::isfinite(report.max_density)) invalid = 1;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &invalid, &any_invalid);
    if (any_invalid) throw std::invalid_argument("Steady residual encountered invalid volume or nonfinite equation terms.");
    report.signed_sum = totals[0]; report.volume = totals[3];
    report.included_count = static_cast<std::size_t>(totals[4]);
    report.excluded_count = static_cast<std::size_t>(totals[5]);
    if (report.volume > Scalar{})
    { report.volume_rms = std::sqrt(totals[1] / report.volume); report.scale = std::sqrt(totals[2] / report.volume); }
    report.dimensionless_available = report.scale > Scalar{} || report.volume_rms == Scalar{};
    report.dimensionless = report.scale > Scalar{} ? report.volume_rms / report.scale : Scalar{};
    return report;
}
}
