/** @file Initialization.tcc
 * @brief Compiled general VoF Initialization operation.
 */
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include "solvers/details/GeneralVoF/Initialization.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include <cmath>
#include <utility>

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
template<ArithmeticMode Mode>
std::shared_ptr<State> Initialization<Mode>::prepare(Resources& resources,
    std::shared_ptr<const VoF::GeneralVoF> phases, const std::vector<VoF::Vector>& owned_velocity,
    const std::vector<double>& face_velocity)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& volume = resources.volume;
    const auto arithmetic_mode_ = resources.arithmetic_mode;
    const Execution execution(*resources.mesh);
    execution.require(
        phases->arithmetic_mode() == arithmetic_mode_, "General VoF initialization changed arithmetic mode.");
    // Empty owners still participate in the same scalar/face imports. Local
    // vector emptiness cannot select whether a field was supplied globally.
    const int local_fields[2] = {!owned_velocity.empty(), !face_velocity.empty()};
    int supplied_fields[2] = {};
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 2, local_fields, supplied_fields);
    const bool supplied_velocity = supplied_fields[0], supplied_faces = supplied_fields[1];
    execution.require((!supplied_velocity || owned_velocity.size() == volume.size()) &&
                          (!supplied_faces || face_velocity.size() == mesh->num_faces()),
        "General VoF initialization supplied field shape mismatch.");
    auto next = std::make_shared<State>();
    next->phases = std::move(phases);
    next->velocity = supplied_velocity ? owned_velocity : std::vector<Vector>(volume.size());
    if (!supplied_velocity && supplied_faces)
    {
        const auto canonical = flow.faces(face_velocity);
        execution.collective(
            [&]
            {
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t c = 0; c < volume.size(); ++c)
                    for (auto face : mesh->faces(LO(c)))
                    {
                        const auto offset = mesh->face_center_vector(face, LO(c));
                        const double amount = (mesh->owner_cell(face) == LO(c) ? 1 : -1) * mesh->face_area(face) *
                                              canonical[face] / volume[c];
                        next->velocity[c][0] += amount * offset.x;
                        next->velocity[c][1] += amount * offset.y;
                        next->velocity[c][2] += amount * offset.z;
                    }
            });
    }
    bool valid = next->velocity.size() == volume.size();
    for (const auto& velocity : next->velocity)
        for (double value : velocity)
            valid = valid && std::isfinite(value);
    execution.require(valid, "General VoF initialization requires finite owned cell velocities.");
    if constexpr (Mode == ArithmeticMode::Exact)
    {
        auto storage = PhaseStorage<ArithmeticMode::Exact>::evaluate(
            *resources.mesh, *next->phases, resources.material, resources.options.byte_budget);
        next->fraction = std::move(storage.fraction);
        next->density = std::move(storage.density);
        next->physical_mass = std::move(storage.mass);
    }
    else
    {
        auto storage = PhaseStorage<Mode>::evaluate(*resources.mesh, *next->phases, resources.material);
        next->fraction = std::move(storage.fraction);
        next->density = std::move(storage.density);
        next->finite_mass = std::move(storage.mass);
        next->finite_mass_error = std::move(storage.mass_error);
        next->physical_mass.clear();
    }
    next->curvature = Curvature<Mode>::assemble(
        *resources.mesh, *next->phases, resources.material.surface_tension, resources.options.byte_budget);
    next->face_velocity = supplied_faces ? flow.faces(face_velocity) : flow.interpolate(next->velocity);
    next->projection_arithmetic.byte_budget = options.byte_budget;
    if constexpr (Mode != ArithmeticMode::Exact)
    {
        const auto closure_topology =
            projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*mesh, *next->phases);
        next->finite_projection.byte_budget = options.byte_budget;
        flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
            next->curvature, 1, 0, options.flow, next->diagnostics, &next->finite_projection,
            std::span<const long double>(next->finite_mass), &closure_topology,
            std::span<const long double>(next->finite_mass_error));
    }
    else
    {
        const auto closure_topology =
            projection_topology<detail::Exact::GeneralVoFProjectionClosureTopology>(*mesh, *next->phases);
        detail::Exact::GeneralVoFFlowWorkspace::project(flow, next->face_velocity, next->velocity, next->pressure,
            next->density, next->fraction, next->curvature, 1, 0, options.flow, next->diagnostics,
            &next->projection_arithmetic,
            std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(
                next->physical_mass),
            &closure_topology);
    }
    if constexpr (Mode == ArithmeticMode::Exact)
        next->diagnostics =
            Diagnostics<ArithmeticMode::Exact>::evaluate(*resources.mesh, *next->phases, resources.material,
                next->physical_mass, next->velocity, next->density, resources.options.byte_budget, next->diagnostics);
    else
        next->diagnostics = Diagnostics<Mode>::evaluate(
            *resources.mesh, *next->phases, resources.material, next->finite_mass, next->velocity, next->diagnostics);
    return next;
}

} // namespace SimpleFluid::detail::general_vof
