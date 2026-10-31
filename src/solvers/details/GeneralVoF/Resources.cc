/** @file Resources.cc
 * @brief Compiled general VoF Resources operation.
 */
#include "solvers/details/GeneralVoF/Resources.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
namespace
{
std::shared_ptr<const MeshHandle<>> validated_phase_mesh(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    const auto mesh = phases->mesh();
    int local = phases->hierarchy()->geometry_unchanged(), global = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("General VoF source geometry is stale before workspace construction.");
    return mesh;
}
template<ArithmeticMode Mode>
std::vector<Vector> finite_spans_mode(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    std::vector<Vector> result;
    for (const auto& geometry : phases->owned_finite_geometry<Mode>())
    {
        const auto spans = geometry->axis_spans();
        result.push_back({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
    }
    return result;
}
std::vector<Vector> finite_spans(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    if (phases->arithmetic_mode() == ArithmeticMode::Exact)
        return {};
    if (phases->arithmetic_mode() == ArithmeticMode::Ordinary)
        return finite_spans_mode<ArithmeticMode::Ordinary>(phases);
    return finite_spans_mode<ArithmeticMode::BoundedCertification>(phases);
}

template<ArithmeticMode Mode>
void initialize_geometry(
    Resources& resources, std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o)
{
    auto& mesh = resources.mesh;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& volume = resources.volume;
    auto& thickness = resources.thickness;
    const auto arithmetic_mode_ = resources.arithmetic_mode;
    const Execution execution(*resources.mesh);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    bool valid = options.byte_budget > 0 && options.maximum_momentum_iterations > 0 &&
                 std::isfinite(options.momentum_relative_tolerance) && options.momentum_relative_tolerance > 0;
    for (double value : {m.gas_density, m.liquid_density})
        valid = valid && std::isfinite(value) && value > 0;
    for (double value : {m.gas_viscosity, m.liquid_viscosity, m.surface_tension})
        valid = valid && std::isfinite(value) && value >= 0;
    for (double value : o.flow.gravity)
        valid = valid && std::isfinite(value);
    for (double value : {o.flow.max_courant, o.flow.capillary_safety, o.flow.viscous_safety,
             o.flow.pressure_absolute_tolerance, o.flow.divergence_tolerance})
        valid = valid && std::isfinite(value) && value > 0;
    valid = valid && std::isfinite(o.flow.pressure_relative_tolerance) && o.flow.pressure_relative_tolerance >= 0 &&
            o.flow.max_pressure_iterations > 0;
    execution.require(valid, "General VoF flow has invalid material or numerical controls.");
    std::array<std::uint64_t, 18> controls{};
    const std::array<double, 15> real_controls{m.gas_density, m.liquid_density, m.gas_viscosity, m.liquid_viscosity,
        m.surface_tension, o.flow.gravity[0], o.flow.gravity[1], o.flow.gravity[2], o.flow.max_courant,
        o.flow.capillary_safety, o.flow.viscous_safety, o.flow.pressure_relative_tolerance,
        o.flow.pressure_absolute_tolerance, o.flow.divergence_tolerance, o.momentum_relative_tolerance};
    for (std::size_t i = 0; i < real_controls.size(); ++i)
        controls[i] = std::bit_cast<std::uint64_t>(real_controls[i]);
    controls[15] = std::uint64_t(o.maximum_momentum_iterations);
    controls[16] = std::uint64_t(o.flow.max_pressure_iterations);
    controls[17] = std::uint64_t(arithmetic_mode_);
    auto agreed = controls;
    Teuchos::broadcast(
        *mesh->owned_cell_map()->getComm(), 0, int(sizeof(agreed)), reinterpret_cast<char*>(agreed.data()));
    execution.require(controls == agreed, "General VoF material/numerical controls differ across ranks.");
    const auto capillary_model = o.capillary.model;
    execution.require((capillary_model == VoF::GeneralVoFCapillaryModel::Reconstructed ||
                          capillary_model == VoF::GeneralVoFCapillaryModel::FilteredBinary) &&
                          std::isfinite(o.capillary.filter_width) && o.capillary.filter_width > 0 &&
                          (arithmetic_mode_ == ArithmeticMode::Ordinary ||
                              capillary_model == VoF::GeneralVoFCapillaryModel::Reconstructed),
        "General VoF capillary model or filter width is invalid for its arithmetic mode.");
    std::array<std::uint64_t, 2> capillary_controls{
        std::uint64_t(capillary_model), std::bit_cast<std::uint64_t>(o.capillary.filter_width)};
    auto agreed_capillary = capillary_controls;
    Teuchos::broadcast(*mesh->owned_cell_map()->getComm(), 0, int(sizeof(agreed_capillary)),
        reinterpret_cast<char*>(agreed_capillary.data()));
    execution.require(capillary_controls == agreed_capillary, "General VoF capillary controls differ across ranks.");
    execution.require(phases->hierarchy()->geometry_unchanged(), "General VoF source geometry is stale.");
    const auto geometry = phases->owned_geometry();
    const auto bounded_geometry = arithmetic_mode_ != ArithmeticMode::Exact
                                      ? phases->owned_finite_geometry<Mode>()
                                      : std::vector<PLIC::ConvexGeometryPtr<Mode>>{};
    volume.resize(mesh->num_owned_cells());
    thickness.resize(volume.size());
    execution.collective(
        [&]
        {
            if (geometry.size() != volume.size())
                throw std::runtime_error("General VoF phase/native geometry order differs.");
            PLIC::GeometryAllocationLedger point_exports;
            std::size_t point_export_bytes = 0;
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                if (!bounded_geometry.empty())
                {
                    const auto exported =
                        bounded_geometry[c]->export_points(point_exports, point_export_bytes, o.byte_budget);
                    const auto p = exported.points();
                    long double area = 0;
                    for (const auto& face : bounded_geometry[c]->faces())
                    {
                        auto normal = A::point({});
                        for (std::size_t i = 1; i + 1 < face.size(); ++i)
                            normal = A::add(normal, A::cross(A::subtract(p[face[i]].dynamic(), p[face[0]].dynamic()),
                                                        A::subtract(p[face[i + 1]].dynamic(), p[face[0]].dynamic())));
                        const auto square = A::dot(normal, normal);
                        area += std::sqrt(square.value) / 2;
                    }
                    const auto v = SimpleFluid::geometry::swept_volume::runtime_value<Mode>(bounded_geometry[c]->volume());
                    volume[c] = A::as_double(v);
                    thickness[c] = static_cast<double>(2 * v.value / area);
                    if (!std::isfinite(thickness[c]) || !(thickness[c] > 0))
                        throw std::runtime_error("Finite VoF cell has no physical thickness.");
                    continue;
                }
                volume[c] = geometry[c]->moments().volume.as_double();
                long double area = 0;
                for (std::size_t face = 0; face < geometry[c]->num_faces(); ++face)
                    area += geometry[c]->face_geometry(face)->area().value;
                thickness[c] = static_cast<double>(2 * geometry[c]->moments().volume.value / area);
                if (!std::isfinite(thickness[c]) || thickness[c] <= 0)
                    throw std::runtime_error("General VoF cell has no positive physical thickness.");
            }
        });
}
} // namespace

Resources::Resources(std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o)
    : mesh(validated_phase_mesh(phases)), arithmetic_mode(phases->arithmetic_mode()), material(m), options(o),
      flow(mesh, arithmetic_mode, finite_spans(phases)), transport(mesh)
{
    if (arithmetic_mode == ArithmeticMode::BoundedCertification)
        initialize_geometry<ArithmeticMode::BoundedCertification>(*this, std::move(phases), m, o);
    else
        initialize_geometry<ArithmeticMode::Ordinary>(*this, std::move(phases), m, o);
}

} // namespace SimpleFluid::detail::general_vof
