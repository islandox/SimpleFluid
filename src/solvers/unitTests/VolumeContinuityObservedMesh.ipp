/** @file VolumeContinuityObservedMesh.ipp
 * @brief Caller-local observation and native execution lease definitions.
 */
#pragma once

#include "VolumeContinuityObservedMesh.hh"

#include <stdexcept>

namespace SimpleFluid::test
{

inline VolumeContinuityObservedMesh::Execution::Execution(const VolumeContinuityObservedMesh& source) : mesh(source)
{
    if (mesh.use_lease)
        native.reset(new NativeExecution(mesh.Handle::acquire_execution_view()));
    if (mesh.observe)
    {
        ++mesh.acquisitions;
        if (mesh.fail_acquisition) throw std::logic_error("Injected continuity acquisition failure.");
        ++mesh.active;
    }
}

inline VolumeContinuityObservedMesh::Execution::~Execution() { if (mesh.observe) --mesh.active; }

inline auto VolumeContinuityObservedMesh::acquire_execution_view() const -> Execution { return Execution(*this); }

inline auto VolumeContinuityObservedMesh::faces(local_ordinal_type cell) const
{
    if (observe)
    {
        ++visits;
        if (active != 1) throw std::logic_error("Continuity traversal escaped its execution scope.");
        if (fail_traversal) throw std::out_of_range("Injected continuity traversal failure.");
    }
    return Handle::faces(cell);
}

} // namespace SimpleFluid::test
