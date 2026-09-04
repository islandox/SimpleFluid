/** Regenerate the common material snapshot using the actual optional library.
 * From the repository root, after building SimpleFluidIF97:
 * c++ -std=c++20 -Isrc verification/openfoam/export_reference_water.cc \
 *     build/gcc/lib/Debug/libSimpleFluidIF97.a -o /tmp/export_reference_water
 * /tmp/export_reference_water > verification/openfoam/reference_water.properties
 * No Trilinos/MPI initialization, upstream headers, or flow-solver run is needed.
 */
#include "examples/IF97ReferenceWater.hh"

#include <exception>
#include <iostream>

int main()
{
    try
    {
        SimpleFluid::Verification::write_if97_reference_water(
            std::cout, SimpleFluid::Verification::query_if97_reference_water(300.0, 101325.0));
        return std::cout.good() ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Cannot export IF97 reference water: " << error.what() << '\n';
        return 1;
    }
}
