/** @file GeneralVoFTransferBinding.hh
 * @brief Bind positive donor routes to the original canonical face mass equation.
 */
#pragma once
#include "solvers/details/ConservativeVoFTransport.hh"
#include <map>
namespace SimpleFluid::detail
{
/** Original incident face stream retained on its endpoint owner. */
struct GeneralVoFOriginalFaceTerm
{
    DefaultTpetraTypes::global_ordinal_type face = 0, donor = 0;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> amount; ///< Includes outward cell incidence.
};
struct GeneralVoFOriginalRow
{
    std::map<DefaultTpetraTypes::global_ordinal_type,
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>
        coefficients;
    std::vector<GeneralVoFOriginalFaceTerm> faces;
};
/** Collective coefficient authentication with a bounded sparse owner exchange.
 * Only a counted floating-point arithmetic bound is admitted. Physical residual
 * tolerances never authorize a mismatch between the face and transfer equations.
 * When unrounded face measures are supplied, their exact signed row aggregate
 * must narrow to the unique routed coefficient; the display-roundoff envelope
 * cannot hide a different microscopic physical transfer.
 */
SIMPLEFLUID_SOLVERS_LOCAL std::vector<GeneralVoFOriginalRow> verify_general_vof_transfer_binding(
    const SP<const MeshHandle<>>&, const std::vector<double>& outgoing,
    const std::vector<std::map<DefaultTpetraTypes::global_ordinal_type, double>>& incoming,
    const std::vector<ConservativeVoFTransport::FaceContribution>& contributions, std::size_t byte_budget,
    const std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>* exact_outgoing = nullptr,
    const std::vector<std::map<DefaultTpetraTypes::global_ordinal_type,
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>>* exact_incoming = nullptr,
    bool exact_only = false);
} // namespace SimpleFluid::detail

namespace SimpleFluid::detail
{
SIMPLEFLUID_SOLVERS_LOCAL std::vector<GeneralVoFOriginalRow> verify_general_vof_exact_transfer_binding(
    const SP<const MeshHandle<>>&,
    const std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>&,
    const std::vector<std::map<DefaultTpetraTypes::global_ordinal_type,
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>>&,
    const std::vector<ConservativeVoFTransport::ExactFaceContribution>&, std::size_t);
}
