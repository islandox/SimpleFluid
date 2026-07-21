/**
 * @file Tpetra.hh
 * @brief Header and module boundary for the Tpetra API used by SimpleFluid.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "modules/Trilinos.hh";
#else
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#endif
