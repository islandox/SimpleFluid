#include "equations/GrayDiffuseEnclosure.hh"

#include <Teuchos_LAPACK.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
namespace
{
constexpr real_t sigma = 5.670374419e-8;
constexpr real_t tolerance = 1.0e-10;
real_t checked(real_t value)
{
    if (!std::isfinite(value))
        throw std::overflow_error("Gray diffuse enclosure produced a non-finite value.");
    return value;
}
void positive(real_t value)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("Gray diffuse enclosure requires positive finite areas and emissivities.");
}
} // namespace

GrayDiffuseEnclosure::GrayDiffuseEnclosure(std::vector<real_t> areas,
    std::vector<real_t> exchange_areas, std::vector<real_t> emissivities)
    : areas_(std::move(areas)), exchange_(std::move(exchange_areas)), emissivities_(std::move(emissivities))
{
    const auto count = size();
    if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        count > std::numeric_limits<std::size_t>::max() / count ||
        exchange_.size() != count * count || emissivities_.size() != count)
        throw std::invalid_argument("Gray diffuse enclosure requires nonempty, matching patch and matrix sizes.");
    for (std::size_t i = 0; i < count; ++i)
    {
        positive(areas_[i]);
        positive(emissivities_[i]);
        if (emissivities_[i] > 1.0)
            throw std::invalid_argument("Gray diffuse enclosure emissivity exceeds one.");
    }
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = i; j < count; ++j)
        {
            const auto a = exchange_[i * count + j], b = exchange_[j * count + i];
            if (!std::isfinite(a) || !std::isfinite(b) || a < 0.0 || b < 0.0)
                throw std::invalid_argument("Gray diffuse enclosure exchange areas must be finite and nonnegative.");
            // Relative to the exchange itself: zero entries cannot hide a finite asymmetric entry.
            if (std::abs(a - b) > tolerance * std::max(a, b))
                throw std::invalid_argument("Gray diffuse enclosure exchange areas violate reciprocity.");
            exchange_[i * count + j] = exchange_[j * count + i] = 0.5 * a + 0.5 * b;
        }
    matrix_.resize(count * count);
    real_t norm = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        long double row = 0.0;
        for (std::size_t j = 0; j < count; ++j)
        {
            row += exchange_[i * count + j];
            matrix_[j * count + i] = (i == j ? 1.0 : 0.0) -
                (1.0 - emissivities_[i]) * (exchange_[i * count + j] / areas_[i]);
        }
        if (std::abs(row - areas_[i]) > tolerance * areas_[i])
            throw std::invalid_argument("Gray diffuse enclosure exchange areas do not close each row.");
    }
    for (std::size_t j = 0; j < count; ++j)
    {
        real_t column = 0.0;
        for (std::size_t i = 0; i < count; ++i)
            column += std::abs(matrix_[j * count + i]);
        norm = std::max(norm, column);
    }
    lu_ = matrix_;
    pivots_.resize(count);
    const int n = static_cast<int>(count);
    const Teuchos::LAPACK<int, real_t> lapack;
    int info = 0;
    lapack.GETRF(n, n, lu_.data(), n, pivots_.data(), &info);
    if (info != 0)
        throw std::domain_error("Gray diffuse enclosure LAPACK factorization failed or is singular.");
    // Strict row diagonal dominance bounds ||A^-1||inf by 1/min(delta).
    // Therefore ||A^-1||1 <= n/min(delta). Use a conservative roundoff
    // allowance before screening; uncertain/reflective cases retain the
    // previous inverse-column metric and its unchanged rejection threshold.
    const auto epsilon = std::numeric_limits<real_t>::epsilon();
    const long double roundoff = 64.0L * epsilon * count * count;
    long double minimum_delta = std::numeric_limits<long double>::infinity();
    long double norm_upper = 0.0L;
    for (std::size_t i = 0; i < count; ++i)
    {
        long double off_diagonal = 0.0L;
        for (std::size_t j = 0; j < count; ++j)
            if (i != j) off_diagonal += std::abs(matrix_[j * count + i]);
        const auto diagonal = static_cast<long double>(std::abs(matrix_[i * count + i]));
        minimum_delta = std::min(minimum_delta,
            diagonal - off_diagonal - roundoff * (diagonal + off_diagonal));
    }
    for (std::size_t j = 0; j < count; ++j)
    {
        long double column_sum = 0.0L;
        for (std::size_t i = 0; i < count; ++i)
            column_sum += std::abs(matrix_[j * count + i]);
        norm_upper = std::max(norm_upper, column_sum * (1.0L + roundoff));
    }
    const auto threshold = 64.0 * epsilon;
    bool condition_is_estimate = false;
    const auto reciprocal_lower_bound = minimum_delta / (count * norm_upper);
    if (minimum_delta > 0.0L && std::isfinite(reciprocal_lower_bound) && reciprocal_lower_bound > threshold)
    {
        std::vector<real_t> work(4 * count);
        std::vector<int> integer_work(count);
        lapack.GECON('1', n, lu_.data(), n, norm, &reciprocal_condition_, work.data(), integer_work.data(), &info);
        condition_is_estimate = info == 0 && std::isfinite(reciprocal_condition_) && reciprocal_condition_ > threshold;
    }
    if (!condition_is_estimate)
    {
        std::vector<real_t> column(count);
        real_t inverse_norm = 0.0;
        for (std::size_t j = 0; j < count; ++j)
        {
            std::fill(column.begin(), column.end(), 0.0);
            column[j] = 1.0;
            lapack.GETRS('N', n, 1, lu_.data(), n, pivots_.data(), column.data(), n, &info);
            if (info != 0)
                throw std::runtime_error("Gray diffuse enclosure LAPACK condition solve failed.");
            real_t sum = 0.0;
            for (const auto value : column)
                sum = checked(sum + std::abs(checked(value)));
            inverse_norm = std::max(inverse_norm, sum);
        }
        reciprocal_condition_ = 1.0 / (norm * inverse_norm);
    }
    if (!std::isfinite(reciprocal_condition_) ||
        reciprocal_condition_ <= threshold)
        throw std::domain_error("Gray diffuse enclosure is numerically singular.");
}

real_t GrayDiffuseEnclosure::blackbody_delta(real_t temperature, real_t reference_temperature)
{
    positive(temperature);
    positive(reference_temperature);
    return checked(sigma * (temperature - reference_temperature) * (temperature + reference_temperature) *
        (temperature * temperature + reference_temperature * reference_temperature));
}

GrayDiffuseEnclosureResult GrayDiffuseEnclosure::evaluate(std::span<const real_t> emission,
    real_t reference_blackbody) const
{
    if (emission.size() != size() || !std::isfinite(reference_blackbody) || reference_blackbody < 0.0)
        throw std::invalid_argument("Gray diffuse enclosure requires matching emission size and finite nonnegative reference.");
    const auto count = size();
    GrayDiffuseEnclosureResult result;
    result.radiosity_delta.resize(count);
    std::vector<real_t> rhs(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!std::isfinite(emission[i]) || checked(reference_blackbody + emission[i]) < 0.0)
            throw std::invalid_argument("Gray diffuse enclosure requires finite nonnegative absolute emission.");
        rhs[i] = checked(emissivities_[i] * emission[i]);
    }
    result.radiosity_delta = rhs;
    const int n = static_cast<int>(count);
    const Teuchos::LAPACK<int, real_t> lapack;
    int info = 0;
    lapack.GETRS('N', n, 1, lu_.data(), n, pivots_.data(), result.radiosity_delta.data(), n, &info);
    if (info != 0)
        throw std::runtime_error("Gray diffuse enclosure LAPACK radiosity solve failed.");
    result.irradiation_delta.assign(count, 0.0);
    result.outward_heat_rate.assign(count, 0.0);
    result.radiosity.resize(count);
    result.irradiation.resize(count);
    real_t scale = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        long double residual = -rhs[i], irradiation = 0.0;
        real_t row_scale = std::abs(rhs[i]);
        for (std::size_t j = 0; j < count; ++j)
        {
            const auto radiosity = checked(result.radiosity_delta[j]);
            irradiation += static_cast<long double>(exchange_[i * count + j] / areas_[i]) * radiosity;
            residual += static_cast<long double>(matrix_[j * count + i]) * radiosity;
            row_scale += std::abs(matrix_[j * count + i] * radiosity);
        }
        result.irradiation_delta[i] = checked(static_cast<real_t>(irradiation));
        result.radiosity[i] = checked(reference_blackbody + result.radiosity_delta[i]);
        result.irradiation[i] = checked(reference_blackbody + result.irradiation_delta[i]);
        result.equation_residual = std::max(result.equation_residual, checked(std::abs(static_cast<real_t>(residual))));
        scale = std::max(scale, checked(row_scale));
    }
    if (result.equation_residual > 256.0 * std::numeric_limits<real_t>::epsilon() * scale)
        throw std::runtime_error("Gray diffuse enclosure radiosity equation residual is too large.");
    for (std::size_t i = 0; i < count; ++i)
        for (auto j = i + 1; j < count; ++j)
        {
            const auto heat = checked(exchange_[i * count + j] *
                (result.radiosity_delta[i] - result.radiosity_delta[j]));
            result.outward_heat_rate[i] = checked(result.outward_heat_rate[i] + heat);
            result.outward_heat_rate[j] = checked(result.outward_heat_rate[j] - heat);
        }
    return result;
}
} // namespace SimpleFluid
