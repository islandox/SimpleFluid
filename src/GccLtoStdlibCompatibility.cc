/**
 * @file GccLtoStdlibCompatibility.cc
 * @brief Preserve libstdc++ COMDATs used by non-LTO STK/Ioss archives.
 *
 * GCC's LTO plugin can choose an LTO copy of these weak definitions as the
 * prevailing copy and then remove it, although regular STK/Ioss objects still
 * reference it. This translation unit is compiled without LTO and linked
 * directly into the optimized shared libraries.
 */

#include <vector>

template class std::vector<bool>;
template class std::vector<unsigned long>;
