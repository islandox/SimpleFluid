/** @file GeneralVoFBoundaryGradient.ipp
 * @brief Caller-arithmetic response of the geometry-conditioned face stencil.
 */
#pragma once
namespace SimpleFluid::detail
{
template<class Scalar, class Difference, class Add, class Scale, class Divide, class IsZero>
std::vector<std::array<Scalar, 3>> GeneralVoFFlowWorkspace::boundary_gradient(
    Scalar zero, Difference difference, Add add, Scale scale, Divide divide, IsZero is_zero) const
{
    std::vector<std::array<Scalar, 3>> result(volumes_.size(), {zero, zero, zero});
    for (std::size_t c = 0; c < volumes_.size(); ++c)
    {
        const auto& stencil = face_gradient_stencils_[c];
        if (!stencil.has_exterior)
            continue;
        auto& rhs = result[c];
        bool zero_response = true;
        for (const auto& sample : stencil.samples)
        {
            const auto delta = difference(c, sample.neighbor);
            zero_response = zero_response && is_zero(delta);
            for (unsigned a = 0; a < 3; ++a)
                rhs[a] = add(rhs[a], scale(delta, sample.weight * sample.displacement[a]));
        }
        // Only exact zero input differences prove the response independently of
        // rank. Cancellation of a nonconstant RHS never bypasses conditioning.
        if (zero_response)
            continue;
        if (!stencil.valid)
            throw std::runtime_error("General VoF one-sided internal-face gradient is deficient or ill-conditioned.");
        for (unsigned a = 0; a < 3; ++a)
            if (absent_gradient_axis_[a])
                rhs[a] = zero;
        for (unsigned a = 0; a < 3; ++a)
        {
            std::swap(rhs[a], rhs[stencil.pivot[a]]);
            rhs[a] = divide(rhs[a], stencil.divisor[a]);
            for (unsigned row = 0; row < 3; ++row)
                if (row != a)
                    rhs[row] = add(rhs[row], scale(rhs[a], -stencil.factor[a][row]));
        }
        for (unsigned a = 0; a < 3; ++a)
            rhs[a] = divide(rhs[a], spans_[c][a]);
    }
    return result;
}
} // namespace SimpleFluid::detail
