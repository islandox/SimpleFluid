/**
 * @file Teuchos.hh
 * @brief Header and module boundary for the Teuchos API used by SimpleFluid.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "modules/Trilinos.hh";
#else
#include <Teuchos_Array.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#endif
