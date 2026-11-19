#include "equations/QbmmMoments.hh"
#include "equations/QbmmPopulationSchema.hh"
#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace SimpleFluid::Reacting
{
namespace
{
using Sum = detail::CompensatedSum<long double>;

bool positive(long double value)
{
    return std::isfinite(value) && value > 0;
}

double positive_double(long double value)
{
    if (!positive(value) || value > std::numeric_limits<double>::max())
        throw std::invalid_argument("QBMM positive quantity is outside the representable range.");
    const auto result = static_cast<double>(value);
    if (std::abs(value - result) / value > qbmm_reconstruction_tolerance)
        throw std::invalid_argument("QBMM positive quantity is outside the representable range.");
    return result;
}

std::array<Sum, 4> sum_moments(std::span<const QbmmAtom> atoms)
{
    std::array<Sum, 4> sums{};
    for (const auto& atom : atoms)
    {
        long double term = atom.count;
        sums[0] += term;
        for (std::size_t k = 1; k < sums.size(); ++k)
        {
            term *= atom.mass_over_scale;
            sums[k] += term;
        }
    }
    return sums;
}

// Kahan's difference of products retains the sign at a cancellation boundary.
long double determinant(long double a, long double b, long double d)
{
    const long double bb = b * b;
    return std::fma(a, d, -bb) + std::fma(-b, b, bb);
}

void verify(QbmmQuadrature& result, const QbmmMomentBlock& h, std::span<const QbmmAtom> atoms)
{
    for (const auto& atom : atoms)
        if (!positive(atom.count) || !positive(atom.mass_over_scale))
        {
            result.status = InversionStatus::IllConditioned;
            return;
        }
    const auto sums = sum_moments(atoms);
    bool consistent = true;
    for (std::size_t k = 0; k < sums.size(); ++k)
    {
        const long double error = std::abs(sums[k].value() - h[k]);
        result.reconstruction.absolute[k] = error;
        result.reconstruction.relative[k] = error / h[k];
        consistent &= result.reconstruction.relative[k] <= qbmm_reconstruction_tolerance;
    }
    if (!consistent)
    {
        result.status = InversionStatus::ReconstructionFailure;
        return;
    }
    result.active_atoms = static_cast<unsigned>(atoms.size());
    std::ranges::copy(atoms, result.atoms.begin());
    result.status = atoms.size() == 1 ? InversionStatus::RankOne : InversionStatus::TwoNodes;
}
} // namespace

QbmmMomentBlock qbmm_atom_moments(std::span<const QbmmAtom> atoms)
{
    if (atoms.empty())
        return {};
    for (const auto& atom : atoms)
        if (!positive(atom.count) || !positive(atom.mass_over_scale))
            throw std::invalid_argument("QBMM active atom needs positive finite count and mass.");
    const auto sums = sum_moments(atoms);
    QbmmMomentBlock result;
    for (std::size_t k = 0; k < result.size(); ++k)
        result[k] = positive_double(sums[k].value());
    return result;
}

QbmmMomentBlock qbmm_cell_moments(const QbmmPopulationSchema& schema, QbmmCellView cell)
{
    if (cell.population != schema.population() || cell.catalog_fingerprint != schema.catalog().fingerprint() ||
        cell.schema_fingerprint != schema.fingerprint())
        throw std::invalid_argument("QBMM cell family or catalog differs from the schema.");
    const QbmmMomentBlock h{cell.number, schema.scaled_mass(cell.species_moles), cell.shape.H2, cell.shape.H3};
    if (!std::ranges::all_of(h, positive) && !std::ranges::all_of(h, [](double x) { return x == 0; }))
        throw std::invalid_argument("QBMM occupied family needs finite positive material, count and shape.");
    return h;
}

QbmmQuadrature invert_qbmm_moments(QbmmMomentBlock h) noexcept
{
    QbmmQuadrature result;
    if (std::ranges::all_of(h, [](double x) { return x == 0; }))
    {
        result.status = InversionStatus::Absent;
        return result;
    }
    if (!std::ranges::all_of(h, positive))
        return result;

    // Scale count and coordinate by exact powers of two, centering the mean at O(1).
    // H0 and H1 individually can differ by hundreds of orders of magnitude.
    const int count_exponent = std::ilogb(h[0]);
    const int coord = std::ilogb(h[1]) - count_exponent;
    auto& report = result.conditioning;
    report.coordinate_exponent = coord;
    std::array<long double, 4> m;
    for (int k = 0; k < 4; ++k)
        m[k] = std::scalbn(static_cast<long double>(h[k]), -count_exponent - k * coord);
    const long double d0 = determinant(m[0], m[1], m[2]), d1 = determinant(m[1], m[2], m[3]);
    report.hankel_margin = d0 / (m[0] * m[2] + m[1] * m[1]);
    report.shifted_hankel_margin = d1 / (m[1] * m[3] + m[2] * m[2]);
    result.status = InversionStatus::IllConditioned;
    if (!std::ranges::all_of(m, positive) || !std::isfinite(report.hankel_margin) ||
        !std::isfinite(report.shifted_hankel_margin))
        return result;

    const long double a0 = m[1] / m[0];
    report.a0 = std::scalbn(a0, coord);
    if (std::abs(report.hankel_margin) <= qbmm_rank_tolerance)
    {
        const long double mono2 = m[1] * a0, mono3 = mono2 * a0;
        if (std::abs(mono2 - m[2]) > qbmm_rank_tolerance * m[2] || std::abs(mono3 - m[3]) > qbmm_rank_tolerance * m[3])
        {
            result.status = d0 > 0 && d1 > 0 ? InversionStatus::IllConditioned : InversionStatus::InconsistentRank;
            return result;
        }
        const QbmmAtom atom{h[0], static_cast<double>(report.a0)};
        const std::span singleton(&atom, 1);
        if (d0 > 0 && d1 == 0)
        {
            // Binary64 rounding can produce an exactly singular shifted matrix
            // even from a single positive atom (e.g. count 0.4 at size 5).
            // At this ambiguous boundary require that one positive atom, with
            // the entire original count, rounds back to every input exactly.
            const auto sums = sum_moments(singleton);
            for (std::size_t k = 0; k < h.size(); ++k)
                if (static_cast<double>(sums[k].value()) != h[k])
                {
                    result.status = InversionStatus::NonpositiveSupport;
                    return result;
                }
        }
        verify(result, h, singleton);
        return result;
    }
    if (d0 <= 0 || d1 <= 0)
    {
        result.status = d0 <= 0 ? InversionStatus::InconsistentRank : InversionStatus::NonpositiveSupport;
        return result;
    }
    if (report.shifted_hankel_margin <= qbmm_rank_tolerance)
        return result; // A positive lower endpoint cannot be resolved at the input precision.

    const long double b1 = d0 / (m[0] * m[0]), det_j = d1 / d0;
    // det(J) = a0*a1 - b1. Recover a1 using positive terms only.
    const long double a1 = (det_j + b1) / a0;
    report.a1 = std::scalbn(a1, coord);
    report.b1 = std::scalbn(b1, 2 * coord);
    const long double offdiag = std::sqrt(b1), half_gap = (a1 - a0) / 2;
    const long double radius = std::hypot(half_gap, offdiag);
    const long double upper = (a0 + a1) / 2 + radius, lower = det_j / upper;
    // Squared first components of the Jacobi rotation. Neither weight is
    // obtained by subtracting a nearly total weight from the original count.
    const long double tangent = offdiag / (std::abs(half_gap) + radius);
    const long double major = static_cast<long double>(h[0]) / (1 + tangent * tangent);
    const long double minor = (major * tangent) * tangent;
    const std::array<QbmmAtom, 2> atoms{
        {{static_cast<double>(half_gap >= 0 ? major : minor), static_cast<double>(std::scalbn(lower, coord))},
            {static_cast<double>(half_gap >= 0 ? minor : major), static_cast<double>(std::scalbn(upper, coord))}}};
    if (!(atoms[0].mass_over_scale < atoms[1].mass_over_scale))
        return result;
    verify(result, h, atoms);
    return result;
}

QbmmGeometryView qbmm_geometry(const QbmmPopulationSchema& schema, QbmmCellView cell, double cell_volume_m3,
    std::optional<double> phase_specific_volume_m3_per_kg)
{
    if (!positive(cell_volume_m3))
        throw std::invalid_argument("QBMM geometry needs finite positive bulk volume.");
    const auto h = qbmm_cell_moments(schema, cell);
    const auto quadrature = invert_qbmm_moments(h);
    if (!quadrature.valid())
        throw std::invalid_argument("QBMM geometry needs a realizable population.");
    QbmmGeometryView view;
    if (quadrature.status == InversionStatus::Absent)
        return view;
    if (!phase_specific_volume_m3_per_kg || !positive(*phase_specific_volume_m3_per_kg))
        throw std::invalid_argument("Occupied QBMM geometry needs a positive common-pressure EOS specific volume.");

    view.active_atoms = quadrature.active_atoms;
    Sum material_mass;
    for (std::size_t s = 0; s < cell.species_moles.size(); ++s)
        material_mass += static_cast<long double>(cell.species_moles[s]) * schema.catalog().species()[s].molarMass;
    view.mass_kg = positive_double(material_mass.value());
    view.number_density = positive_double(static_cast<long double>(cell.number) / cell_volume_m3);
    std::array<Sum, 4> diameter_moments{};
    for (unsigned i = 0; i < view.active_atoms; ++i)
    {
        const auto atom = quadrature.atoms[i];
        const long double mass = static_cast<long double>(schema.mass_scale_kg()) * atom.mass_over_scale;
        const long double diameter =
            std::cbrt(6 / std::numbers::pi_v<long double> * mass * *phase_specific_volume_m3_per_kg);
        view.mass_kg_per_bubble[i] = positive_double(mass);
        view.diameter_m[i] = positive_double(diameter);
        view.radius_m[i] = positive_double(diameter / 2);
        long double term = atom.count;
        for (std::size_t k = 1; k < diameter_moments.size(); ++k)
        {
            term *= diameter;
            diameter_moments[k] += term;
        }
    }
    view.diameter_moment_density[0] = view.number_density;
    for (std::size_t k = 1; k < diameter_moments.size(); ++k)
        view.diameter_moment_density[k] = positive_double(diameter_moments[k].value() / cell_volume_m3);
    const long double area = std::numbers::pi_v<long double> * diameter_moments[2].value();
    view.area_density = positive_double(area / cell_volume_m3);
    view.d32 = positive_double(6 * material_mass.value() * *phase_specific_volume_m3_per_kg / area);
    view.min_diameter = view.diameter_m[0];
    view.max_diameter = view.diameter_m[view.active_atoms - 1];
    return view;
}
} // namespace SimpleFluid::Reacting
