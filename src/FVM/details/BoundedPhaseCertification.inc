/** @file BoundedPhaseCertification.inc
 * @brief Bounded receipt and phase-publication certification.
 * Included only by BoundedPhaseArithmetic.hh inside SimpleFluid::VoF::finite_detail.
 */
inline std::pair<double, double> publish(Value value)
{
    if (!std::isfinite(value.value) || value.value < 0 || value.absolute_error < 0 ||
        (value.value == 0 && value.absolute_error != 0) ||
        (value.value != 0 && !(value.value - value.absolute_error > 0)))
        throw std::invalid_argument("Finite VoF phase positivity is unresolved.");
    const auto result = static_cast<double>(value.value);
    if (!std::isfinite(result) || (value.value != 0 && result == 0))
        throw std::invalid_argument("Finite VoF phase publication is outside binary64 range.");
    return {result, upper_double(bound_add(value.absolute_error, std::abs(value.value - result)))};
}
