/**
 * @file ClangLtoTrilinosCompatibility.cc
 * @brief Preserve complete MueLu COMDAT definitions across Clang ThinLTO.
 *
 * The GCC-built MueLu ETI archive groups its primary vtable, VTT, and
 * construction vtables into one COMDAT.  Clang ThinLTO can emit an incomplete
 * competing group from a consumer translation unit, causing the linker to
 * discard the archive's complete definition.  This deliberately non-LTO
 * instantiation emits the required tables as independently selectable COMDATs.
 */

#include "dataclass/TpetraTypes.hh"

#include <MueLu_TpetraOperator.hpp>

template class MueLu::TpetraOperator<
    SimpleFluid::real_t,
    SimpleFluid::local_index_t,
    SimpleFluid::global_index_t,
    SimpleFluid::DefaultTpetraNode>;
