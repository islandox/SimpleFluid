/** @file if97_verification_unavailable.cc
 * @brief Reports that water verification requires the optional IF97 dependency.
 */
#include <iostream>

int main()
{
    std::cerr << "This water verification case requires SIMPLEFLUID_ENABLE_IF97=ON.\n"
                 "Configure with cmake --preset GCC-ninja-multi -DSIMPLEFLUID_ENABLE_IF97=ON,\n"
                 "then rebuild the verification target.\n";
    return 2;
}
