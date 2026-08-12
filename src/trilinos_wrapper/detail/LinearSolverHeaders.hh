/**
 * @file LinearSolverHeaders.hh
 * @brief Unconditional textual solver declarations for the Trilinos boundary.
 */
#pragma once

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosSolverManager.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
