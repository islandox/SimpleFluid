#include <iostream>
int main()
{
    std::cerr << "This verification requires SIMPLEFLUID_REACTING_MATERIAL_LIBRARY pointing to "
                 "libthermal_reacting_properties.so from given submodule.\n";
    return 1;
}
