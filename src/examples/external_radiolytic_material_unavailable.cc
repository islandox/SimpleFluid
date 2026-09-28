#include <iostream>
int main()
{
    std::cerr << "This verification requires SIMPLEFLUID_RADIOLYTIC_MATERIAL_LIBRARY pointing to "
                 "libthermal_radiolytic_properties.so from given submodule.\n";
    return 1;
}
