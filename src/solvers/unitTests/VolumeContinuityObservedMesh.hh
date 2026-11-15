/** @file VolumeContinuityObservedMesh.hh
 * @brief Named internal mesh view shared by continuity tests and their providers.
 */
#pragma once

#include "utils/export_helper.hh"
#include "geometry/MeshHandle.hh"

#include <memory>
#include <utility>

namespace SimpleFluid::test
{

/** @brief Observe execution scope without changing the native mesh identity.
 *
 * Keep execution construction, destruction and acquisition inline in every
 * caller: the inherited MeshHandle lease uses image-local bookkeeping.
 */
class SIMPLEFLUID_LOCAL VolumeContinuityObservedMesh : public MeshHandle<DefaultTpetraTypes>
{
public:
    using Handle = MeshHandle<DefaultTpetraTypes>;
    using Handle::Handle;
    using NativeExecution = decltype(std::declval<const Handle&>().acquire_execution_view());
    mutable int acquisitions = 0, active = 0, visits = 0;
    bool observe = false, use_lease = true, fail_acquisition = false, fail_traversal = false;
    struct Execution
    {
        const VolumeContinuityObservedMesh& mesh;
        std::unique_ptr<NativeExecution> native;
        explicit inline Execution(const VolumeContinuityObservedMesh& source);
        inline ~Execution();
        Execution(const Execution&) = delete;
        Execution& operator=(const Execution&) = delete;
    };
    inline Execution acquire_execution_view() const;
    inline auto faces(local_ordinal_type cell) const;
};

} // namespace SimpleFluid::test

#include "VolumeContinuityObservedMesh.ipp"
