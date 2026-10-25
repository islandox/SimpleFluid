/** @file GeometryMeasure.hh
 * @brief Policy-selected retained geometry measures behind the public PLIC interface.
 */
#pragma once
#include "geometry/PLIC/CertifiedGeometryProperties.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include <span>
#include <vector>

namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> class ConvexCellGeometry;

template<> class GeometryMeasure<ArithmeticMode::Ordinary>
{
public:
    static constexpr ArithmeticMode arithmetic_mode = ArithmeticMode::Ordinary;
    GeometryMeasure() = default;
    [[nodiscard]] static GeometryMeasure from_value(double value) { return from_nominal(value); }
    [[nodiscard]] static GeometryMeasure from_nominal(long double);
    [[nodiscard]] static GeometryMeasure volume(const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
    [[nodiscard]] GeometryValue<ArithmeticMode::Ordinary> certified() const { return {value_}; }
    [[nodiscard]] GeometryValue<ArithmeticMode::Ordinary> enclosure() const { return {value_}; }
    [[nodiscard]] double as_double() const { return certified().as_double(); }
    [[nodiscard]] long double nominal() const noexcept { return value_; }
    [[nodiscard]] int sign() const noexcept { return (value_ > 0) - (value_ < 0); }
    [[nodiscard]] int compare(const GeometryMeasure& other) const noexcept
    {
        return (value_ > other.value_) - (value_ < other.value_);
    }
    [[nodiscard]] GeometryMeasure operator+(const GeometryMeasure&) const;
    [[nodiscard]] GeometryMeasure operator-(const GeometryMeasure&) const;
    [[nodiscard]] GeometryMeasure scaled(double) const;
    [[nodiscard]] GeometryMeasure divided_by(const GeometryMeasure&, std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] GeometryMeasure squared(std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] bool relative_error_at_most(const GeometryMeasure& reference, double tolerance) const;
    [[nodiscard]] double log_ratio(const GeometryMeasure& denominator) const;
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] static GeometryMeasure decode(std::span<const char>, std::size_t byte_budget = 1024 * 1024);
    [[nodiscard]] GeometryMeasure compacted(std::size_t operation_byte_budget) const;
    [[nodiscard]] GeometryMeasure rounded_dyadic(unsigned significant_bits, std::size_t operation_byte_budget) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return sizeof(*this); }
    [[nodiscard]] const void* storage_identity() const noexcept { return this; }

private:
    explicit GeometryMeasure(long double value) : value_(value) {}
    long double value_ = 0;
};
} // namespace SimpleFluid::PLIC

#include "geometry/PLIC/details/exact/GeometryMeasure.hh"
