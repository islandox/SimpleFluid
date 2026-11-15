/** @file STKBackendBridge.cc
 * @brief Construct Ioss properties inside the static archive owner.
 */
#include "STKBackendBridge.hh"
#include <Ioss_Property.h>
#include <stk_io/StkMeshIoBroker.hpp>

extern "C" SIMPLEFLUID_STK_IO_BRIDGE_EXPORT void simplefluid_trilinos_stk_io_set_string_property(
    void* broker, const char* name, const char* value)
{
    if (!broker || !name || !value)
        throw std::invalid_argument("Invalid STK I/O property request.");
    static_cast<stk::io::StkMeshIoBroker*>(broker)->property_add(Ioss::Property(name, value));
}
