#include "solvers/NoxNonlinearSolver.hh"
#include "solvers/BelosLinearSolver.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#ifdef SIMPLEFLUID_ENABLE_NOX
#include <NOX_Config.h>
#ifndef HAVE_NOX_THYRA
#error "SIMPLEFLUID_ENABLE_NOX requires Trilinos NOX with Thyra support"
#endif
#include <NOX.H>
#include <NOX_LineSearch_Generic.H>
#include <NOX_LineSearch_UserDefinedFactory.H>
#include <NOX_Thyra.H>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_ParameterListAcceptorDefaultBase.hpp>
#include <Thyra_LinearOpSourceBase.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryBase.hpp>
#include <Thyra_StateFuncModelEvaluatorBase.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#endif

namespace SimpleFluid
{
#ifdef SIMPLEFLUID_ENABLE_NOX
namespace
{
using Pack = DefaultTpetraTypes;
using Vector = Pack::vector_type;
using MultiVector = Pack::multi_vector_type;
using Operator = Pack::operator_type;
using Map = Pack::map_type;
using Extract =
    Thyra::TpetraOperatorVectorExtraction<double, Pack::local_ordinal_type, Pack::global_ordinal_type, Pack::node_type>;
using Space = Thyra::VectorSpaceBase<double>;
using ThyraOp = Thyra::LinearOpBase<double>;
using Teuchos::RCP;
using Teuchos::rcp;
using WallClock = std::chrono::steady_clock;

/** Account native work even if an evaluation or solve throws. */
class AccumulateTime
{
public:
    explicit AccumulateTime(double& total) : total_(total), start_(WallClock::now()) {}
    ~AccumulateTime() { total_ += std::chrono::duration<double>(WallClock::now() - start_).count(); }

private:
    double& total_;
    WallClock::time_point start_;
};

bool collectively(const Map& map, bool local)
{
    int input = local ? 1 : 0, output = 0;
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MIN, 1, &input, &output);
    return output != 0;
}

/** Communicator comparison is local; validate it on the control communicator
 * before entering any collective through callback-owned maps. */
bool comm_congruent(const Teuchos::Comm<int>& first, const Teuchos::Comm<int>& second)
{
#ifdef HAVE_TEUCHOS_MPI
    const auto* first_mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&first);
    const auto* second_mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&second);
    if (first_mpi || second_mpi)
    {
        if (!first_mpi || !second_mpi)
            return false;
        const auto first_raw = first_mpi->getRawMpiComm();
        const auto second_raw = second_mpi->getRawMpiComm();
        if (first_raw.is_null() || second_raw.is_null())
            return false;
        int comparison = MPI_UNEQUAL;
        return MPI_Comm_compare(*first_raw, *second_raw, &comparison) == MPI_SUCCESS &&
               (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
    }
#endif
    return dynamic_cast<const Teuchos::SerialComm<int>*>(&first) &&
           dynamic_cast<const Teuchos::SerialComm<int>*>(&second);
}

/** Map::isSameAs has fast exits that assume pointer and metadata choices are
 * rank-consistent. Validate caller metadata locally, then reduce explicitly. */
bool locally_same_map(const Map& first, const Map& second)
{
    if (!comm_congruent(*first.getComm(), *second.getComm()) ||
        first.getGlobalNumElements() != second.getGlobalNumElements() ||
        first.getLocalNumElements() != second.getLocalNumElements() || first.getIndexBase() != second.getIndexBase() ||
        first.isDistributed() != second.isDistributed())
        return false;
    const auto first_gids = first.getLocalElementList();
    const auto second_gids = second.getLocalElementList();
    return std::equal(first_gids.begin(), first_gids.end(), second_gids.begin());
}

bool finite(const Vector& v, bool positive = false)
{
    const auto values = v.getData(0);
    bool valid = true;
    for (const double value : values)
        valid = valid && std::isfinite(value) && (!positive || value > 0.0);
    return collectively(*v.getMap(), valid);
}

void multiply(MultiVector& value, const RCP<const Vector>& scale, bool inverse)
{
    if (scale.is_null())
        return;
    const auto s = scale->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto v = value.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (std::size_t j = 0; j < value.getNumVectors(); ++j)
        for (std::size_t i = 0; i < value.getLocalLength(); ++i)
            v(i, j) = inverse ? v(i, j) / s(i, 0) : v(i, j) * s(i, 0);
}

/** Dr J Dx, or Dx^-1 P^-1 Dr^-1; no native object is scaled in place. */
class ScaledOperator final : public Operator
{
public:
    ScaledOperator(RCP<const Operator> op, RCP<const Vector> left, RCP<const Vector> right, bool inverse)
        : op_(std::move(op)), left_(std::move(left)), right_(std::move(right)), inverse_(inverse)
    {
    }
    RCP<const Map> getDomainMap() const override { return op_->getDomainMap(); }
    RCP<const Map> getRangeMap() const override { return op_->getRangeMap(); }
    void apply(const MultiVector& x, MultiVector& y, Teuchos::ETransp mode = Teuchos::NO_TRANS, double alpha = 1.0,
        double beta = 0.0) const override
    {
        if (mode != Teuchos::NO_TRANS)
            throw std::invalid_argument("NOX scaled operator has no transpose action");
        if (alpha == 0.0)
        {
            if (beta == 0.0)
                y.putScalar(0.0);
            else
                y.scale(beta);
            return;
        }
        // NOX applies a retained generation sequentially. Scratch is private to
        // this scaled wrapper and never aliases a native or accepted vector.
        if (input_.is_null() || input_->getNumVectors() != x.getNumVectors())
        {
            input_ = rcp(new MultiVector(getDomainMap(), x.getNumVectors()));
            output_ = rcp(new MultiVector(getRangeMap(), x.getNumVectors()));
        }
        input_->assign(x);
        multiply(*input_, right_, inverse_);
        op_->apply(*input_, *output_);
        multiply(*output_, left_, inverse_);
        if (beta == 0.0)
        {
            y.assign(*output_);
            if (alpha != 1.0)
                y.scale(alpha);
        }
        else
            y.update(alpha, *output_, beta);
    }

private:
    RCP<const Operator> op_;
    RCP<const Vector> left_, right_;
    bool inverse_;
    mutable RCP<MultiVector> input_, output_;
};

struct Generation
{
    RCP<const Operator> jacobian, preconditioner;
};

class LinearizationOperator final : public ThyraOp
{
public:
    explicit LinearizationOperator(RCP<const Space> space) : space_(std::move(space)) {}
    RCP<const Space> range() const override { return space_; }
    RCP<const Space> domain() const override { return space_; }
    RCP<const Generation> generation;

protected:
    bool opSupportedImpl(Thyra::EOpTransp mode) const override { return mode == Thyra::NOTRANS; }
    void applyImpl(Thyra::EOpTransp mode, const Thyra::MultiVectorBase<double>& x,
        const Teuchos::Ptr<Thyra::MultiVectorBase<double>>& y, double alpha, double beta) const override
    {
        if (mode != Thyra::NOTRANS || generation.is_null())
            throw std::logic_error("NOX Jacobian is unavailable");
        const auto tx = Extract::getConstTpetraMultiVector(Teuchos::rcpFromRef(x));
        const auto ty = Extract::getTpetraMultiVector(Teuchos::rcpFromRef(*y));
        generation->jacobian->apply(*tx, *ty, Teuchos::NO_TRANS, alpha, beta);
    }

private:
    RCP<const Space> space_;
};

struct SolveContext
{
    NonlinearSolveResult result;
    NonlinearSolverOptions nonlinear;
    LinearSolverOptions linear;
    NonlinearSolverCacheStatistics cache_statistics;
    RCP<const Generation> active_generation;
    RCP<Vector> scaled_x, accepted_x, gate_x, jacobian_direction;
};

class NativeLinearSolve final : public Thyra::LinearOpWithSolveBase<double>
{
public:
    explicit NativeLinearSolve(RCP<SolveContext> context) : context_(std::move(context)) {}
    RCP<const Thyra::LinearOpSourceBase<double>> source;
    RCP<const Generation> generation;
    RCP<const Space> space;
    RCP<const Space> range() const override { return space; }
    RCP<const Space> domain() const override { return space; }
    void invalidate()
    {
        source = Teuchos::null;
        generation = Teuchos::null;
        if (!problem_.is_null())
        {
            problem_->setOperator(Teuchos::null);
            problem_->setRightPrec(Teuchos::null);
            problem_->setLHS(Teuchos::null);
            problem_->setRHS(Teuchos::null);
            problem_->setInitResVec(Teuchos::null);
            problem_->setInitPrecResVec(Teuchos::null);
        }
    }

protected:
    bool opSupportedImpl(Thyra::EOpTransp t) const override { return t == Thyra::NOTRANS; }
    bool solveSupportsImpl(Thyra::EOpTransp t) const override { return t == Thyra::NOTRANS; }
    bool solveSupportsNewImpl(Thyra::EOpTransp t, const Teuchos::Ptr<const Thyra::SolveCriteria<double>>) const override
    {
        return t == Thyra::NOTRANS;
    }
    bool solveSupportsSolveMeasureTypeImpl(Thyra::EOpTransp t, const Thyra::SolveMeasureType&) const override
    {
        return t == Thyra::NOTRANS;
    }
    void applyImpl(Thyra::EOpTransp t, const Thyra::MultiVectorBase<double>& x,
        const Teuchos::Ptr<Thyra::MultiVectorBase<double>>& y, double alpha, double beta) const override
    {
        if (t != Thyra::NOTRANS || generation.is_null())
            throw std::logic_error("NOX linear solve operator unavailable");
        generation->jacobian->apply(*Extract::getConstTpetraMultiVector(Teuchos::rcpFromRef(x)),
            *Extract::getTpetraMultiVector(Teuchos::rcpFromRef(*y)), Teuchos::NO_TRANS, alpha, beta);
    }
    Thyra::SolveStatus<double> solveImpl(Thyra::EOpTransp t, const Thyra::MultiVectorBase<double>& rhs,
        const Teuchos::Ptr<Thyra::MultiVectorBase<double>>& solution,
        const Teuchos::Ptr<const Thyra::SolveCriteria<double>> criteria) const override
    {
        AccumulateTime elapsed(context_->result.linear_solve_seconds);
        if (t != Thyra::NOTRANS || generation.is_null())
            throw std::logic_error("NOX correction solve unavailable");
        auto b = Extract::getConstTpetraMultiVector(Teuchos::rcpFromRef(rhs));
        auto x = Extract::getTpetraMultiVector(Teuchos::rcpFromRef(*solution));
        const double tolerance = criteria.is_null() ? context_->linear.tolerance : criteria->requestedTol;
        if (!(tolerance > 0.0 && tolerance < 1.0))
            throw std::invalid_argument("NOX requested an invalid linear tolerance");
        x->putScalar(0.0);
        if (problem_.is_null())
            problem_ = rcp(new Problem);
        problem_->setOperator(generation->jacobian);
        problem_->setLHS(x);
        problem_->setRHS(b);
        problem_->setRightPrec(generation->preconditioner);
        // The correction always starts at zero, hence its initial residual is
        // exactly b. Refresh this pointer on every Newton solve.
        problem_->setInitResVec(b);
        if (!problem_->setProblem())
            throw std::runtime_error("Invalid NOX Belos problem");
        auto parameters = rcp(new Teuchos::ParameterList);
        parameters->set("Maximum Iterations", context_->linear.max_iterations);
        parameters->set("Convergence Tolerance", tolerance);
        parameters->set("Verbosity", context_->linear.verbosity);
        parameters->set("Implicit Residual Scaling", "Norm of RHS");
        parameters->set("Explicit Residual Scaling", "Norm of RHS");
        if (context_->linear.backend == LinearSolverBackend::Gmres)
        {
            parameters->set("Num Blocks", context_->nonlinear.krylov_restart);
            parameters->set("Maximum Restarts", context_->linear.max_iterations);
        }
        if (solver_.is_null() || backend_ != context_->linear.backend)
        {
            ++context_->cache_statistics.linear_solver_builds;
            backend_ = context_->linear.backend;
            if (context_->linear.backend == LinearSolverBackend::BiCGStab)
                solver_ = rcp(new Belos::BiCGStabSolMgr<double, MultiVector, Operator>(problem_, parameters));
            else
                solver_ = rcp(new Belos::PseudoBlockGmresSolMgr<double, MultiVector, Operator>(problem_, parameters));
        }
        else
            solver_->setParameters(parameters);
        ++context_->result.linear_solves;
        const auto belos_status = solver_->solve();
        const int iterations = solver_->getNumIters();
        context_->result.krylov_iterations += iterations;
        if (residual_.is_null() || residual_->getNumVectors() != b->getNumVectors())
            residual_ = rcp(new MultiVector(b->getMap(), b->getNumVectors()));
        generation->jacobian->apply(*x, *residual_);
        residual_->update(1.0, *b, -1.0);
        Teuchos::Array<double> norms(b->getNumVectors()), rhs_norms(b->getNumVectors());
        residual_->norm2(norms());
        b->norm2(rhs_norms());
        double achieved = 0.0;
        for (int j = 0; j < norms.size(); ++j)
        {
            const double relative = rhs_norms[j] > 0.0 ? norms[j] / rhs_norms[j] : norms[j];
            achieved = std::isfinite(relative) ? std::max(achieved, relative) : std::numeric_limits<double>::infinity();
        }
        context_->result.achieved_linear_tolerance = achieved;
        Thyra::SolveStatus<double> status;
        status.achievedTol = achieved;
        status.solveStatus = belos_status == Belos::Converged && achieved <= tolerance
                                 ? Thyra::SOLVE_STATUS_CONVERGED
                                 : Thyra::SOLVE_STATUS_UNCONVERGED;
        status.message = "Explicit scaled Newton-system residual / RHS norm";
        status.extraParameters = rcp(new Teuchos::ParameterList);
        status.extraParameters->set("Iteration Count", iterations);
        if (status.solveStatus != Thyra::SOLVE_STATUS_CONVERGED)
            context_->result.reason = "Correction solve failed the explicit residual forcing tolerance";
        return status;
    }

private:
    using Problem = Belos::LinearProblem<double, MultiVector, Operator>;
    RCP<SolveContext> context_;
    mutable RCP<Problem> problem_;
    mutable RCP<Belos::SolverManager<double, MultiVector, Operator>> solver_;
    mutable RCP<MultiVector> residual_;
    mutable std::optional<LinearSolverBackend> backend_;
};

class NativeLinearSolveFactory final : public Thyra::LinearOpWithSolveFactoryBase<double>,
                                       public Teuchos::ParameterListAcceptorDefaultBase
{
public:
    explicit NativeLinearSolveFactory(RCP<NativeLinearSolve> solve) : solve_(std::move(solve)) {}
    void setParameterList(const RCP<Teuchos::ParameterList>& list) override { setMyParamList(list); }
    RCP<const Teuchos::ParameterList> getValidParameters() const override { return rcp(new Teuchos::ParameterList); }
    bool isCompatible(const Thyra::LinearOpSourceBase<double>& source) const override
    {
        return dynamic_cast<const LinearizationOperator*>(source.getOp().get()) != nullptr;
    }
    RCP<Thyra::LinearOpWithSolveBase<double>> createOp() const override { return solve_; }
    void initializeOp(const RCP<const Thyra::LinearOpSourceBase<double>>& source,
        Thyra::LinearOpWithSolveBase<double>* op,
        Thyra::ESupportSolveUse = Thyra::SUPPORT_SOLVE_UNSPECIFIED) const override
    {
        auto& solve = dynamic_cast<NativeLinearSolve&>(*op);
        const auto& linearization = dynamic_cast<const LinearizationOperator&>(*source->getOp());
        solve.source = source;
        solve.generation = linearization.generation;
        solve.space = linearization.domain();
    }
    void uninitializeOp(Thyra::LinearOpWithSolveBase<double>* op,
        RCP<const Thyra::LinearOpSourceBase<double>>* source = nullptr,
        RCP<const Thyra::PreconditionerBase<double>>* prec = nullptr,
        RCP<const Thyra::LinearOpSourceBase<double>>* approximate = nullptr,
        Thyra::ESupportSolveUse* use = nullptr) const override
    {
        auto& solve = dynamic_cast<NativeLinearSolve&>(*op);
        if (source)
            *source = solve.source;
        if (prec)
            *prec = Teuchos::null;
        if (approximate)
            *approximate = Teuchos::null;
        if (use)
            *use = Thyra::SUPPORT_SOLVE_UNSPECIFIED;
        solve.source = Teuchos::null;
        solve.generation = Teuchos::null;
    }

private:
    RCP<NativeLinearSolve> solve_;
};

class NativeModel final : public Thyra::StateFuncModelEvaluatorBase<double>
{
public:
    NativeModel(NonlinearCallbacks callbacks, RCP<SolveContext> context)
        : callbacks_(std::move(callbacks)), context_(std::move(context)),
          space_(Thyra::createVectorSpace<double, Pack::local_ordinal_type, Pack::global_ordinal_type, Pack::node_type>(
              callbacks_.map)),
          physical_x_(callbacks_.map), linear_solve_(rcp(new NativeLinearSolve(context_)))
    {
    }
    void set_callbacks(NonlinearCallbacks callbacks) { callbacks_ = std::move(callbacks); }
    void invalidate_linearization()
    {
        context_->active_generation = Teuchos::null;
        linear_solve_->invalidate();
    }
    RCP<const Space> get_x_space() const override { return space_; }
    RCP<const Space> get_f_space() const override { return space_; }
    InArgs<double> createInArgs() const override
    {
        InArgsSetup<double> in;
        in.setModelEvalDescription("SimpleFluid native coupled residual");
        in.setSupports(IN_ARG_x, true);
        return in;
    }
    RCP<ThyraOp> create_W_op() const override { return rcp(new LinearizationOperator(space_)); }
    RCP<const Thyra::LinearOpWithSolveFactoryBase<double>> get_W_factory() const override
    {
        return rcp(new NativeLinearSolveFactory(linear_solve_));
    }

protected:
    OutArgs<double> createOutArgsImpl() const override
    {
        OutArgsSetup<double> out;
        out.setModelEvalDescription("SimpleFluid native coupled residual");
        out.setSupports(OUT_ARG_f, true);
        out.setSupports(OUT_ARG_W_op, true);
        return out;
    }
    void evalModelImpl(const InArgs<double>& in, const OutArgs<double>& out) const override
    {
        physical_x_.assign(*Extract::getConstTpetraVector(in.get_x()));
        multiply(physical_x_, callbacks_.state_scale, false);
        if (!finite(physical_x_))
        {
            out.setFailed();
            return;
        }
        if (!out.get_f().is_null())
        {
            auto f = Extract::getTpetraVector(out.get_f());
            ++context_->result.residual_evaluations;
            bool valid = false;
            {
                AccumulateTime elapsed(context_->result.residual_seconds);
                valid = callbacks_.residual(physical_x_, *f);
            }
            if (!collectively(*callbacks_.map, valid))
            {
                out.setFailed();
                return;
            }
            multiply(*f, callbacks_.residual_scale, false);
            if (!finite(*f))
            {
                out.setFailed();
                return;
            }
        }
        if (!out.get_W_op().is_null())
        {
            NonlinearLinearization native;
            {
                AccumulateTime elapsed(context_->result.linearization_seconds);
                native = callbacks_.linearize(physical_x_);
            }
            if (!collectively(*callbacks_.map, !native.jacobian.is_null()))
                throw std::runtime_error("Native Jacobian is null on a rank");
            const auto check_maps = [this](const Operator& op)
            {
                const auto domain = op.getDomainMap(), range = op.getRangeMap();
                if (!collectively(*callbacks_.map, !domain.is_null() && !range.is_null()))
                    throw std::runtime_error("Native nonlinear operator has a null map");
                if (!collectively(*callbacks_.map, comm_congruent(*domain->getComm(), *callbacks_.map->getComm()) &&
                                                       comm_congruent(*range->getComm(), *callbacks_.map->getComm())))
                    throw std::runtime_error("Native nonlinear operator communicator mismatch");
                // Compare local ownership, then agree on the callback communicator.
                const bool domain_matches = locally_same_map(*domain, *callbacks_.map);
                const bool range_matches = locally_same_map(*range, *callbacks_.map);
                if (!collectively(*callbacks_.map, domain_matches && range_matches))
                    throw std::runtime_error("Native nonlinear operator map mismatch");
            };
            check_maps(*native.jacobian);
            int has_prec = native.right_preconditioner.is_null() ? 0 : 1;
            int prec_min = 0, prec_max = 0;
            Teuchos::reduceAll(*callbacks_.map->getComm(), Teuchos::REDUCE_MIN, 1, &has_prec, &prec_min);
            Teuchos::reduceAll(*callbacks_.map->getComm(), Teuchos::REDUCE_MAX, 1, &has_prec, &prec_max);
            if (prec_min != prec_max)
                throw std::runtime_error("Native nonlinear preconditioner presence differs between ranks");
            if (has_prec)
                check_maps(*native.right_preconditioner);
            auto generation = rcp(new Generation);
            generation->jacobian =
                rcp(new ScaledOperator(native.jacobian, callbacks_.residual_scale, callbacks_.state_scale, false));
            if (!native.right_preconditioner.is_null())
                generation->preconditioner = rcp(new ScaledOperator(
                    native.right_preconditioner, callbacks_.state_scale, callbacks_.residual_scale, true));
            context_->active_generation = generation;
            ++context_->cache_statistics.linearization_generations;
            dynamic_cast<LinearizationOperator&>(*out.get_W_op()).generation = generation;
        }
    }

private:
    NonlinearCallbacks callbacks_;
    RCP<SolveContext> context_;
    RCP<const Space> space_;
    mutable Vector physical_x_;
    RCP<NativeLinearSolve> linear_solve_;
};

class Armijo final : public NOX::LineSearch::Generic
{
public:
    explicit Armijo(RCP<SolveContext> context) : context_(std::move(context)) {}
    bool compute(NOX::Abstract::Group& trial, double& step, const NOX::Abstract::Vector& direction,
        const NOX::Solver::Generic& solver) override
    {
        const auto& base = solver.getPreviousSolutionGroup();
        // NOX transfers shared-Jacobian ownership when it copies the previous
        // group. Its isJacobian() flag is therefore unsuitable here. Retain and
        // apply the exact generation that produced this correction directly.
        if (context_->active_generation.is_null())
            throw std::logic_error("Armijo has no retained nonlinear linearization");
        const auto& td = dynamic_cast<const NOX::Thyra::Vector&>(direction);
        const auto& tf = dynamic_cast<const NOX::Thyra::Vector&>(base.getF());
        context_->active_generation->jacobian->apply(
            *Extract::getConstTpetraVector(td.getThyraRCPVector()), *context_->jacobian_direction);
        const double slope = Extract::getConstTpetraVector(tf.getThyraRCPVector())->dot(*context_->jacobian_direction);
        const double norm = base.getNormF();
        const double merit = 0.5 * norm * norm;
        if (!std::isfinite(slope) || !std::isfinite(merit) || slope >= 0.0)
        {
            context_->result.reason = "Nonlinear correction is not a descent direction";
            trial = base;
            step = 0.0;
            return false;
        }
        step = 1.0;
        for (int attempt = 0; attempt <= context_->nonlinear.maximum_backtracks; ++attempt)
        {
            trial.computeX(base, direction, step);
            const bool valid = trial.computeF() == NOX::Abstract::Group::Ok;
            const double trial_norm = valid ? trial.getNormF() : std::numeric_limits<double>::infinity();
            if (std::isfinite(trial_norm) &&
                0.5 * trial_norm * trial_norm <= merit + context_->nonlinear.armijo * step * slope)
                return true;
            ++context_->result.line_search_rejections;
            step *= context_->nonlinear.backtrack_factor;
            if (step < context_->nonlinear.minimum_step)
                break;
        }
        trial = base;
        step = 0.0;
        context_->result.reason = "Armijo line search exhausted admissible decreasing trials";
        return false;
    }

private:
    RCP<SolveContext> context_;
};

class ArmijoFactory final : public NOX::LineSearch::UserDefinedFactory
{
public:
    explicit ArmijoFactory(RCP<SolveContext> context) : context_(std::move(context)) {}
    RCP<NOX::LineSearch::Generic> buildLineSearch(const RCP<NOX::GlobalData>&, Teuchos::ParameterList&) const override
    {
        return rcp(new Armijo(context_));
    }

private:
    RCP<SolveContext> context_;
};

/** Separate component RMS checks prevent a large momentum scale hiding mass. */
class ComponentStatus final : public NOX::StatusTest::Generic
{
public:
    ComponentStatus(RCP<SolveContext> context, const NonlinearCallbacks& callbacks)
        : context_(std::move(context)), gate_(callbacks.convergence_gate), state_scale_(callbacks.state_scale),
          physical_x_(context_->gate_x)
    {
    }
    NOX::StatusTest::StatusType checkStatus(
        const NOX::Solver::Generic& solver, NOX::StatusTest::CheckType check) override
    {
        using namespace NOX::StatusTest;
        if (check == None)
            return status_ = Unevaluated;
        context_->result.nonlinear_iterations = solver.getNumIterations();
        const auto& group = solver.getSolutionGroup();
        if (!group.isF())
            return status_ = Failed;
        const auto& tf = dynamic_cast<const NOX::Thyra::Vector&>(group.getF());
        const auto f = Extract::getConstTpetraVector(tf.getThyraRCPVector());
        if (!finite(*f))
            return status_ = Failed;
        std::array<double, 4> local_max{}, global_max{};
        std::array<double, 8> local{}, global{};
        const auto values = f->getData(0);
        const auto component_of = [&f](std::size_t i)
        {
            const auto gid = f->getMap()->getGlobalElement(static_cast<Pack::local_ordinal_type>(i));
            return static_cast<std::size_t>((gid % 4 + 4) % 4);
        };
        for (std::size_t i = 0; i < f->getLocalLength(); ++i)
        {
            const auto c = component_of(i);
            local_max[c] = std::max(local_max[c], std::abs(values[i]));
        }
        Teuchos::reduceAll(*f->getMap()->getComm(), Teuchos::REDUCE_MAX, 4, local_max.data(), global_max.data());
        // Normalize before squaring so a finite large residual cannot overflow
        // its reference norm and falsely satisfy an infinite relative target.
        for (std::size_t i = 0; i < f->getLocalLength(); ++i)
        {
            const auto c = component_of(i);
            const double scaled = global_max[c] > 0.0 ? values[i] / global_max[c] : 0.0;
            local[c] += scaled * scaled;
            local[4 + c] += 1.0;
        }
        Teuchos::reduceAll(*f->getMap()->getComm(), Teuchos::REDUCE_SUM, 8, local.data(), global.data());
        bool converged = true;
        context_->result.scaled_residual = 0.0;
        for (std::size_t c = 0; c < 4; ++c)
        {
            const double norm = global[4 + c] > 0 ? global_max[c] * std::sqrt(global[c] / global[4 + c]) : 0.0;
            if (!initialized_)
                initial_[c] = norm;
            converged = converged && norm <= context_->nonlinear.absolute_tolerance +
                                                 context_->nonlinear.relative_tolerance * initial_[c];
            context_->result.scaled_residual = std::max(context_->result.scaled_residual, norm);
        }
        initialized_ = true;
        bool physical_gate_failed = false;
        if (converged && gate_)
        {
            const auto& tx = dynamic_cast<const NOX::Thyra::Vector&>(group.getX());
            physical_x_->assign(*Extract::getConstTpetraVector(tx.getThyraRCPVector()));
            multiply(*physical_x_, state_scale_, false);
            if (!finite(*physical_x_))
            {
                context_->result.reason = "Nonfinite physical state at convergence gate";
                return status_ = Failed;
            }
            const bool local_acceptance = gate_(*physical_x_);
            physical_gate_failed = !collectively(*physical_x_->getMap(), local_acceptance);
            converged = !physical_gate_failed;
        }
        if (converged)
            return status_ = Converged;
        if (solver.getNumIterations() >= context_->nonlinear.maximum_iterations)
        {
            context_->result.reason = physical_gate_failed
                                          ? "Maximum nonlinear iterations reached before physical acceptance"
                                          : "Maximum nonlinear iterations reached";
            return status_ = Failed;
        }
        return status_ = Unconverged;
    }
    NOX::StatusTest::StatusType getStatus() const override { return status_; }
    std::ostream& print(std::ostream& stream, int indent = 0) const override
    {
        return stream << std::string(indent, ' ') << "Coupled component RMS: " << status_ << '\n';
    }

private:
    RCP<SolveContext> context_;
    std::function<bool(const Vector&)> gate_;
    RCP<const Vector> state_scale_;
    RCP<Vector> physical_x_;
    std::array<double, 4> initial_{};
    bool initialized_ = false;
    NOX::StatusTest::StatusType status_ = NOX::StatusTest::Unevaluated;
};

void validate_callbacks(const NonlinearCallbacks& c, const Map& map, bool require_same_map)
{
    // Use the caller state communicator for every validation collective: an
    // invalid or missing callback map must not strand another rank in a solve.
    const auto require = [&map](bool condition, const char* reason)
    {
        if (!collectively(map, condition))
            throw std::invalid_argument(reason);
    };
    require(!c.map.is_null() && bool(c.residual) && bool(c.linearize),
        "NOX requires map, residual and linearization callbacks on every rank");
    require(comm_congruent(*c.map->getComm(), *map.getComm()), "NOX callback and state communicators differ");
    if (require_same_map)
        require(locally_same_map(*c.map, map), "NOX state map mismatch");
    int has_gate = c.convergence_gate ? 1 : 0, gate_min = 0, gate_max = 0;
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MIN, 1, &has_gate, &gate_min);
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MAX, 1, &has_gate, &gate_max);
    require(gate_min == gate_max, "NOX convergence gate presence differs between ranks");
    for (const auto& scale : {c.state_scale, c.residual_scale})
    {
        int present = scale.is_null() ? 0 : 1, minimum = 0, maximum = 0;
        Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MIN, 1, &present, &minimum);
        Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MAX, 1, &present, &maximum);
        require(minimum == maximum, "NOX scale presence differs between ranks");
        if (present)
        {
            require(comm_congruent(*scale->getMap()->getComm(), *map.getComm()),
                "NOX scale and state communicators differ");
            require(locally_same_map(*c.map, *scale->getMap()), "NOX scale map mismatch");
            require(finite(*scale, true), "NOX scales require finite positive values");
        }
    }
}

void validate(
    const NonlinearCallbacks& c, const Vector& x, const NonlinearSolverOptions& n, const LinearSolverOptions& l)
{
    const auto& map = *x.getMap();
    validate_callbacks(c, map, true);
    const auto require = [&map](bool condition, const char* reason)
    {
        if (!collectively(map, condition))
            throw std::invalid_argument(reason);
    };
    const auto positive = [](double v) { return std::isfinite(v) && v > 0.0; };
    require(n.backend == NonlinearBackend::NOX &&
                (n.linearization == CoupledLinearization::Picard ||
                    n.linearization == CoupledLinearization::AnalyticNewton) &&
                (l.backend == LinearSolverBackend::Gmres || l.backend == LinearSolverBackend::BiCGStab),
        "NOX requires a valid linearization and GMRES or BiCGStab corrections");
    require(n.maximum_iterations > 0 && n.maximum_backtracks >= 0 && n.krylov_restart > 0 &&
                positive(n.absolute_tolerance) && positive(n.relative_tolerance) && positive(n.forcing_minimum) &&
                std::isfinite(n.forcing_initial) && std::isfinite(n.forcing_maximum) &&
                n.forcing_minimum <= n.forcing_initial && n.forcing_initial <= n.forcing_maximum &&
                n.forcing_maximum < 1.0 && positive(n.armijo) && n.armijo < 1.0 && positive(n.backtrack_factor) &&
                n.backtrack_factor < 1.0 && positive(n.minimum_step) && n.minimum_step <= 1.0 &&
                positive(n.velocity_scale) && positive(n.pressure_scale) && positive(n.momentum_residual_scale) &&
                positive(n.continuity_residual_scale) && positive(n.continuity_tolerance) && l.max_iterations > 0 &&
                positive(l.tolerance) && l.tolerance < 1.0,
        "Invalid NOX nonlinear or linear solve controls");
    const std::array<double, 24> controls = {double(n.backend), double(n.linearization), double(n.maximum_iterations),
        double(n.maximum_backtracks), double(n.krylov_restart), n.relative_tolerance, n.absolute_tolerance,
        n.continuity_tolerance, n.velocity_scale, n.pressure_scale, n.momentum_residual_scale,
        n.continuity_residual_scale, n.forcing_initial, n.forcing_minimum, n.forcing_maximum, n.armijo,
        n.backtrack_factor, n.minimum_step, double(l.max_iterations), l.tolerance, double(l.verbosity),
        double(l.backend), double(l.preconditioner), double(l.reuse_preconditioner)};
    std::array<double, controls.size()> minima{}, maxima{};
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MIN, int(controls.size()), controls.data(), minima.data());
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MAX, int(controls.size()), controls.data(), maxima.data());
    require(minima == maxima, "NOX solve controls differ between ranks");
}
} // namespace
#endif

struct NOXNonlinearSolver::Impl
{
    explicit Impl(NonlinearCallbacks supplied) : callbacks(std::move(supplied)) {}
    NonlinearCallbacks callbacks;
#ifdef SIMPLEFLUID_ENABLE_NOX
    RCP<SolveContext> context = rcp(new SolveContext);
    RCP<NativeModel> model;
    RCP<NOX::Thyra::Group> group;
    RCP<NOX::Solver::Generic> solver;
    RCP<Thyra::VectorBase<double>> initial;
    RCP<Teuchos::ParameterList> parameters;
    std::array<double, 4> forcing_controls{};

    void clear_numerical_state()
    {
        context->active_generation = Teuchos::null;
        if (!model.is_null())
            model->invalidate_linearization();
        if (!group.is_null())
        {
            auto op = group->getNonconstJacobianOperator();
            dynamic_cast<LinearizationOperator&>(*op).generation = Teuchos::null;
            group->setX(NOX::Thyra::Vector(initial));
        }
    }

    void prepare_vectors()
    {
        if (!model.is_null() && !context->scaled_x.is_null() &&
            collectively(*callbacks.map, locally_same_map(*context->scaled_x->getMap(), *callbacks.map)))
            return;
        // Drop objects that retain the previous vector space before rebuilding.
        solver = Teuchos::null;
        group = Teuchos::null;
        model = Teuchos::null;
        initial = Teuchos::null;
        context->active_generation = Teuchos::null;
        context->scaled_x = rcp(new Vector(callbacks.map));
        context->accepted_x = rcp(new Vector(callbacks.map));
        context->gate_x = rcp(new Vector(callbacks.map));
        context->jacobian_direction = rcp(new Vector(callbacks.map));
        ++context->cache_statistics.vector_cache_builds;
        model = rcp(new NativeModel(callbacks, context));
        initial = Thyra::createVector<double, Pack::local_ordinal_type, Pack::global_ordinal_type, Pack::node_type>(
            context->scaled_x, model->get_x_space());
    }
#endif
};

NOXNonlinearSolver::NOXNonlinearSolver(NonlinearCallbacks callbacks)
    : d_impl(std::make_unique<Impl>(std::move(callbacks)))
{
}
NOXNonlinearSolver::~NOXNonlinearSolver() = default;
NOXNonlinearSolver::NOXNonlinearSolver(NOXNonlinearSolver&&) noexcept = default;
NOXNonlinearSolver& NOXNonlinearSolver::operator=(NOXNonlinearSolver&&) noexcept = default;

void NOXNonlinearSolver::set_callbacks(NonlinearCallbacks callbacks)
{
    if (!d_impl)
        throw std::logic_error("Cannot reset a moved-from NOXNonlinearSolver");
#ifdef SIMPLEFLUID_ENABLE_NOX
    const auto control_map = d_impl->callbacks.map.is_null() ? callbacks.map : d_impl->callbacks.map;
    if (control_map.is_null())
        throw std::invalid_argument("NOX callback replacement requires a map");
    validate_callbacks(callbacks, *control_map, false);
    const bool compatible =
        !d_impl->model.is_null() && !d_impl->context->scaled_x.is_null() &&
        collectively(*control_map, locally_same_map(*d_impl->context->scaled_x->getMap(), *callbacks.map));
    d_impl->clear_numerical_state();
    d_impl->callbacks = std::move(callbacks);
    if (compatible)
    {
        d_impl->model->set_callbacks(d_impl->callbacks);
        if (!d_impl->solver.is_null())
            d_impl->solver->reset(
                NOX::Thyra::Vector(d_impl->initial), rcp(new ComponentStatus(d_impl->context, d_impl->callbacks)));
    }
    else
    {
        d_impl->solver = Teuchos::null;
        d_impl->group = Teuchos::null;
        d_impl->model = Teuchos::null;
        d_impl->initial = Teuchos::null;
        // prepare_vectors detects the new map and rebuilds lazily on solve.
    }
#else
    d_impl->callbacks = std::move(callbacks);
#endif
}

NonlinearSolverCacheStatistics NOXNonlinearSolver::cache_statistics() const noexcept
{
#ifdef SIMPLEFLUID_ENABLE_NOX
    return d_impl ? d_impl->context->cache_statistics : NonlinearSolverCacheStatistics{};
#else
    return {};
#endif
}

NonlinearSolveResult NOXNonlinearSolver::solve(DefaultTpetraTypes::vector_type& physical_x,
    const NonlinearSolverOptions& nonlinear, const LinearSolverOptions& linear)
{
    if (!d_impl)
        throw std::logic_error("Cannot solve with a moved-from NOXNonlinearSolver");
#ifndef SIMPLEFLUID_ENABLE_NOX
    (void) physical_x;
    (void) nonlinear;
    (void) linear;
    throw std::invalid_argument("NOX backend requires SIMPLEFLUID_ENABLE_NOX=ON");
#else
    const auto started = WallClock::now();
    auto& impl = *d_impl;
    const auto& callbacks = impl.callbacks;
    validate(callbacks, physical_x, nonlinear, linear);
    auto context = impl.context;
    context->result = {};
    context->nonlinear = nonlinear;
    context->linear = linear;
    try
    {
        impl.prepare_vectors();
        impl.clear_numerical_state();
        context->scaled_x->assign(physical_x);
        multiply(*context->scaled_x, callbacks.state_scale, true);
        if (impl.group.is_null())
            impl.group = rcp(new NOX::Thyra::Group(NOX::Thyra::Vector(impl.initial), impl.model));
        auto status = rcp(new ComponentStatus(context, callbacks));
        const std::array<double, 4> forcing_controls = {
            nonlinear.forcing_initial, nonlinear.forcing_minimum, nonlinear.forcing_maximum, linear.tolerance};
        if (impl.solver.is_null() || forcing_controls != impl.forcing_controls)
        {
            // NOX captures forcing controls in its Newton direction object;
            // rebuild that control layer if they change, retaining the group
            // vectors and the native LOWS/Belos allocation underneath it.
            impl.group->setX(NOX::Thyra::Vector(impl.initial));
            impl.parameters = rcp(new Teuchos::ParameterList);
            auto parameters = impl.parameters;
            parameters->set("Nonlinear Solver", "Line Search Based");
            parameters->sublist("Printing").set("Output Information", NOX::Utils::Error);
            auto& direction = parameters->sublist("Direction");
            direction.set("Method", "Newton");
            auto& newton = direction.sublist("Newton");
            newton.set("Forcing Term Method", "Type 2");
            newton.set("Forcing Term Initial Tolerance", nonlinear.forcing_initial);
            newton.set("Forcing Term Minimum Tolerance", nonlinear.forcing_minimum);
            newton.set("Forcing Term Maximum Tolerance", nonlinear.forcing_maximum);
            newton.set("Rescue Bad Newton Solve", false);
            newton.sublist("Linear Solver").set("Tolerance", linear.tolerance);
            newton.sublist("Linear Solver").set("Solve Measure Denominator", "Norm RHS");
            auto& search = parameters->sublist("Line Search");
            search.set("Method", "User Defined");
            RCP<NOX::LineSearch::UserDefinedFactory> factory = rcp(new ArmijoFactory(context));
            search.set("User Defined Line Search Factory", factory);
            impl.solver = NOX::Solver::buildSolver(impl.group, status, parameters);
            impl.forcing_controls = forcing_controls;
        }
        else
            impl.solver->reset(NOX::Thyra::Vector(impl.initial), status);
        const auto outcome = impl.solver->solve();
        context->result.nonlinear_iterations = impl.solver->getNumIterations();
        context->result.converged = outcome == NOX::StatusTest::Converged;
        if (context->result.converged)
        {
            const auto& final_x = dynamic_cast<const NOX::Thyra::Vector&>(impl.solver->getSolutionGroup().getX());
            context->accepted_x->assign(*Extract::getConstTpetraVector(final_x.getThyraRCPVector()));
            multiply(*context->accepted_x, callbacks.state_scale, false);
            if (!finite(*context->accepted_x))
                throw std::runtime_error("Converged NOX state is nonfinite");
            physical_x.assign(*context->accepted_x);
            context->result.reason = callbacks.convergence_gate
                                         ? "Scaled residual components and physical convergence gate passed"
                                         : "All four scaled residual components converged";
        }
        else if (context->result.reason.empty())
            context->result.reason = "NOX nonlinear solve failed";
    }
    catch (const std::exception& error)
    {
        context->result.converged = false;
        if (context->result.reason.empty())
            context->result.reason = error.what();
    }
    context->result.total_seconds = std::chrono::duration<double>(WallClock::now() - started).count();
    const std::array<double, 4> local_times = {context->result.total_seconds, context->result.residual_seconds,
        context->result.linearization_seconds, context->result.linear_solve_seconds};
    std::array<double, 4> maximum_times{};
    Teuchos::reduceAll(*physical_x.getMap()->getComm(), Teuchos::REDUCE_MAX, int(local_times.size()),
        local_times.data(), maximum_times.data());
    context->result.total_seconds = maximum_times[0];
    context->result.residual_seconds = maximum_times[1];
    context->result.linearization_seconds = maximum_times[2];
    context->result.linear_solve_seconds = maximum_times[3];
    return context->result;
#endif
}

NonlinearSolveResult solve_nox(const NonlinearCallbacks& callbacks, DefaultTpetraTypes::vector_type& physical_x,
    const NonlinearSolverOptions& nonlinear, const LinearSolverOptions& linear)
{
    NOXNonlinearSolver solver(callbacks);
    return solver.solve(physical_x, nonlinear, linear);
}
} // namespace SimpleFluid
