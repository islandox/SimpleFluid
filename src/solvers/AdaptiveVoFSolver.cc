#include "solvers/details/AdaptiveVoFContext.hh"
/** @file AdaptiveVoFSolver.cc
 * @brief Distributed synchronous-level geometric two-phase flow.
 */
#include "solvers/AdaptiveVoFSolver.hh"

#include "equations/ReactingGasProperties.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/VoFProjectionWorkspace.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
using namespace detail::adaptive_vof;

struct AdaptiveVoFSolver::Impl
{
    using State = detail::adaptive_vof::AdaptiveVoFState;
    using Context = detail::adaptive_vof::AdaptiveVoFContext;


    struct Published
    {
        State state;
        ScalarField f, p;
        VectorField u;
        Published(State candidate, const SP<const Handle>& mesh)
            : state(std::move(candidate)), f(mesh, "resolvedGasFraction", false), p(mesh, "pressure", false), u(mesh, "velocity", false)
        {
            for (std::size_t c = 0; c < state.f.size(); ++c)
            {
                const auto id = static_cast<LO>(c);
                f.set_owned_value(id, state.f[c]);
                p.set_owned_value(id, state.p[c]);
                u.set_owned_value(id, vec3<double>{state.u[c][0], state.u[c][1], state.u[c][2]});
            }
            f.sync_ghosts();
            p.sync_ghosts();
            u.sync_ghosts();
        }
    };

    std::shared_ptr<Context> context;
    std::shared_ptr<Published> published, saved;
    Checkpoint active_checkpoint = 0, next_checkpoint = 1;
    std::uint64_t low_trial_nonce = 0;

    struct LowTrialData
    {
        const Impl* owner = nullptr;
        std::shared_ptr<Context> context;
        std::shared_ptr<const Published> accepted;
        std::uint64_t nonce = 0;
        SimpleFluid::Meshes::AMR::HierarchyObservation observation;
        bool representation = false;
        VoF::RepresentationDiagnostics representation_diagnostics;
        State candidate;
    };

    void invalidate_low_trials()
    {
        require_all(
            low_trial_nonce < std::numeric_limits<std::uint64_t>::max(), "Low-Mach trial generation exhausted.");
        ++low_trial_nonce;
    }

    Impl(std::shared_ptr<const Hierarchy> h, VoFMaterial material, VoFSolverOptions options)
    {
        validate_options(material, options);
        context = std::make_shared<Context>(std::move(h), material, options);
        State initial;
        initial.f.assign(context->owned(), 0);
        initial.p = initial.f;
        initial.k = initial.f;
        initial.u.resize(context->owned());
        initial.face_u.assign(context->faces.size(), 0);
        context->budget(initial);
        published = std::make_shared<Published>(std::move(initial), context->mesh);
    }

    void initialize(const std::vector<double>& f, std::vector<Vector> velocity, std::vector<double> faces,
        Context::BoundaryProjection boundary)
    {
        require_all(!saved, "AdaptiveVoFSolver cannot reinitialize while a checkpoint is active.");
        require_all(!published->state.low || !published->state.low->pending_representation_relaxation,
            "AdaptiveVoFSolver cannot erase pending representation pressure histories by reinitializing.");
        context->validate_fraction(f);
        State next;
        next.f = f;
        next.u = std::move(velocity);
        next.face_u = context->checked_faces(faces);
        next.k = context->geometry->curvature(f);
        next.p.assign(f.size(), 0);
        context->project(next, context->density(next), 1, std::vector<Vector>(f.size()), false, boundary);
        context->budget(next);
        auto candidate = std::make_shared<Published>(std::move(next), context->mesh);
        invalidate_low_trials();
        published.swap(candidate);
        context->low_options.reset();
    }

    void regrid(const std::vector<std::size_t>& refine, const std::vector<SimpleFluid::Meshes::AMR::AMRBlock>& coarsen,
        VoF::RefinementOptions protection, VoF::RefinementHints hints)
    {
        require_all(!saved, "AdaptiveVoFSolver topology is frozen while a checkpoint is active.");
        require_all(!published->state.low || !published->state.low->pending_representation_relaxation,
            "AdaptiveVoFSolver requires physical representation relaxation before regridding.");
        identical(
            static_cast<double>(coarsen.size()), "AdaptiveVoFSolver coarsening request counts disagree across ranks.");
        if (!coarsen.empty())
        {
            const auto safe = context->geometry->refinement_plan(
                published->state.f, published->state.face_u, protection, std::move(hints));
            const bool valid = std::all_of(coarsen.begin(), coarsen.end(), [&](const auto& key)
                { return std::find(safe.coarsen.begin(), safe.coarsen.end(), key) != safe.coarsen.end(); });
            require_all(valid, "AdaptiveVoFSolver refuses to coarsen an interface, its stencil/motion guard, or a "
                               "nonuniform sibling group.");
        }
        const auto& source = context->geometry->hierarchy();
        const auto adaptation = source->plan_adaptation(refine, coarsen);
        const auto hierarchy = source->apply_adaptation(adaptation);
        if (hierarchy->topology_generation() == context->geometry->hierarchy()->topology_generation())
            return;
        auto candidate_context = std::make_shared<Context>(hierarchy, context->material, context->options);
        if (published->state.low)
        {
            auto next = context->remap_low(published->state, *candidate_context);
            auto candidate = std::make_shared<Published>(std::move(next), candidate_context->mesh);
            invalidate_low_trials();
            context.swap(candidate_context);
            published.swap(candidate);
            return;
        }
        auto quantities = published->state.inventories;
        const auto rho = context->density(published->state);
        for (std::size_t a = 0; a < 3; ++a)
        {
            Inventory momentum;
            momentum.name = "__vof_momentum_" + std::to_string(a);
            momentum.carrier = Carrier::MixtureMass;
            momentum.nonnegative = false;
            momentum.amount.resize(context->owned());
            for (std::size_t c = 0; c < context->owned(); ++c)
                momentum.amount[c] = rho[c] * context->volume[c] * published->state.u[c][a];
            quantities.push_back(std::move(momentum));
        }
        Inventory pressure_history;
        pressure_history.name = "__vof_pressure_history";
        pressure_history.carrier = Carrier::BulkVolume;
        pressure_history.nonnegative = false;
        pressure_history.amount.resize(context->owned());
        for (std::size_t c = 0; c < context->owned(); ++c)
            pressure_history.amount[c] = published->state.p[c] * context->volume[c];
        quantities.push_back(std::move(pressure_history));
        auto remapped = context->geometry->remap(*candidate_context->geometry, published->state.f, quantities,
            context->material.liquid_density, context->material.gas_density);
        State next;
        next.f = std::move(remapped.fraction);
        next.p.assign(next.f.size(), 0);
        next.k = candidate_context->geometry->curvature(next.f);
        next.u.resize(next.f.size());
        next.steps = published->state.steps;
        next.diagnostics.time = published->state.diagnostics.time;
        const auto new_rho = candidate_context->density(next);
        const auto passive_count = published->state.inventories.size();
        for (std::size_t c = 0; c < next.f.size(); ++c)
            for (std::size_t a = 0; a < 3; ++a)
                next.u[c][a] =
                    remapped.inventories[passive_count + a].amount[c] / (new_rho[c] * candidate_context->volume[c]);
        next.inventories.assign(std::make_move_iterator(remapped.inventories.begin()),
            std::make_move_iterator(remapped.inventories.begin() + static_cast<std::ptrdiff_t>(passive_count)));
        next.face_u = candidate_context->interpolate(next.u);
        candidate_context->project(
            next, new_rho, 1, std::vector<Vector>(next.f.size()), false, Context::BoundaryProjection::InitialCell);
        // Carry the previous multiplier as diagnostic history. Only the new
        // flux projection correction changes momentum; the next physical step
        // recomputes capillary equilibrium on the new hierarchy.
        for (std::size_t c = 0; c < next.f.size(); ++c)
            next.p[c] += remapped.inventories[passive_count + 3].amount[c] / candidate_context->volume[c];
        candidate_context->budget(next);
        const auto& config = hierarchy->config();
        const double domain_volume = config.extent[0] * config.extent[1] * config.extent[2];
        require_all(
            std::abs(next.diagnostics.gas_volume - published->state.diagnostics.gas_volume) <= 5e-12 * domain_volume,
            "AdaptiveVoFSolver regrid failed its phase-volume conservation gate.");
        const auto old_momentum_l1 = context->momentum_l1(published->state);
        const auto new_momentum_l1 = candidate_context->momentum_l1(next);
        for (std::size_t a = 0; a < 3; ++a)
            if (config.periodic[a])
                require_all(std::abs(next.diagnostics.momentum[a] - published->state.diagnostics.momentum[a]) <=
                                5e-11 * std::max(old_momentum_l1[a] + new_momentum_l1[a], 1e-30),
                    "AdaptiveVoFSolver regrid failed periodic momentum conservation.");
        for (std::size_t q = 0; q < passive_count; ++q)
        {
            const double before = context->inventory_total(published->state.inventories[q]);
            const double after = candidate_context->inventory_total(next.inventories[q]);
            const double scale = context->inventory_l1(published->state.inventories[q]) +
                                 candidate_context->inventory_l1(next.inventories[q]);
            require_all(std::abs(after - before) <= 5e-12 * std::max(scale, 1e-30),
                "AdaptiveVoFSolver regrid failed a passive-inventory conservation gate.");
        }
        auto candidate = std::make_shared<Published>(std::move(next), candidate_context->mesh);
        invalidate_low_trials();
        context.swap(candidate_context);
        published.swap(candidate);
    }
};

AdaptiveVoFSolver::AdaptiveVoFSolver(
    std::shared_ptr<const Hierarchy> hierarchy, VoFMaterial material, VoFSolverOptions options)
    : impl_(std::make_unique<Impl>(std::move(hierarchy), material, options))
{
}
AdaptiveVoFSolver::~AdaptiveVoFSolver() = default;
AdaptiveVoFSolver::AdaptiveVoFSolver(AdaptiveVoFSolver&&) noexcept = default;
AdaptiveVoFSolver& AdaptiveVoFSolver::operator=(AdaptiveVoFSolver&&) noexcept = default;

void AdaptiveVoFSolver::initialize(const std::vector<double>& f, const std::vector<Vector>& input)
{
    auto velocity = input;
    if (velocity.empty())
        velocity.resize(impl_->context->owned());
    bool valid = velocity.size() == impl_->context->owned();
    for (const auto& v : velocity)
        for (double value : v)
            valid = valid && std::isfinite(value);
    require_all(valid, "AdaptiveVoFSolver initial cell velocities must be finite and cover owned cells.");
    auto faces = impl_->context->interpolate(velocity);
    impl_->initialize(f, std::move(velocity), std::move(faces), Impl::Context::BoundaryProjection::InitialCell);
}

void AdaptiveVoFSolver::initialize_faces(const std::vector<double>& f, const std::vector<double>& velocity)
{
    auto faces = impl_->context->checked_faces(velocity);
    auto cell = impl_->context->average_faces(faces);
    impl_->initialize(f, std::move(cell), std::move(faces), Impl::Context::BoundaryProjection::InitialFace);
}

VoFStepDiagnostics AdaptiveVoFSolver::step(double dt)
{
    auto next = impl_->context->advance(impl_->published->state, dt);
    const auto diagnostics = next.diagnostics;
    auto candidate = std::make_shared<Impl::Published>(std::move(next), impl_->context->mesh);
    impl_->invalidate_low_trials();
    impl_->published.swap(candidate);
    return diagnostics;
}

double AdaptiveVoFSolver::stable_timestep() const
{
    return impl_->context->stable_timestep(impl_->published->state);
}

void AdaptiveVoFSolver::initialize_low_mach(VoF::LowMachOptions options, const VoF::LowMachInitialState& initial)
{
    require_all(!impl_->saved, "AdaptiveVoFSolver cannot reinitialize while a checkpoint is active.");
    require_all(!impl_->published->state.low || !impl_->published->state.low->pending_representation_relaxation,
        "AdaptiveVoFSolver cannot erase pending representation pressure histories by reinitializing.");
    impl_->context->validate_low_options(options);
    const auto previous = impl_->context->low_options;
    impl_->context->low_options = std::make_shared<const VoF::LowMachOptions>(std::move(options));
    try
    {
        auto next = impl_->context->initialize_low(initial);
        auto candidate = std::make_shared<Impl::Published>(std::move(next), impl_->context->mesh);
        impl_->invalidate_low_trials();
        impl_->published.swap(candidate);
    }
    catch (...)
    {
        impl_->context->low_options = previous;
        throw;
    }
}

bool AdaptiveVoFSolver::low_mach_enabled() const noexcept
{
    return impl_->published->state.low.has_value();
}

const VoF::LowMachState& AdaptiveVoFSolver::low_mach_state() const
{
    if (!low_mach_enabled())
        throw std::logic_error("AdaptiveVoFSolver low-Mach mode is not initialized.");
    return *impl_->published->state.low;
}

const VoF::LowMachDiagnostics& AdaptiveVoFSolver::low_mach_diagnostics() const
{
    (void) low_mach_state();
    return impl_->published->state.low_diagnostics;
}

const VoF::LowMachOptions& AdaptiveVoFSolver::low_mach_options() const
{
    (void) low_mach_state();
    return *impl_->context->low_options;
}

AdaptiveVoFSolver::LowMachTrial::LowMachTrial(std::shared_ptr<const void> candidate) : candidate_(std::move(candidate))
{
}

const VoF::LowMachState& AdaptiveVoFSolver::LowMachTrial::state() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return *std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.low;
}
const VoF::LowMachDiagnostics& AdaptiveVoFSolver::LowMachTrial::diagnostics() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.low_diagnostics;
}
const VoF::LowMachTransport& AdaptiveVoFSolver::LowMachTrial::transport() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.low_flux;
}
const std::vector<double>& AdaptiveVoFSolver::LowMachTrial::face_velocity() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.face_u;
}
const std::vector<Vector>& AdaptiveVoFSolver::LowMachTrial::velocity() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.u;
}
const std::vector<double>& AdaptiveVoFSolver::LowMachTrial::mechanical_pressure() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.p;
}

const std::vector<double>& AdaptiveVoFSolver::LowMachTrial::gas_fraction() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->candidate.f;
}

bool AdaptiveVoFSolver::LowMachTrial::is_representation_change() const
{
    if (!candidate_)
        throw std::logic_error("AdaptiveVoFSolver low-Mach trial is empty.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->representation;
}

const VoF::RepresentationDiagnostics& AdaptiveVoFSolver::LowMachTrial::representation_diagnostics() const
{
    if (!candidate_ || !is_representation_change())
        throw std::logic_error("AdaptiveVoFSolver trial is not a representation conversion.");
    return std::static_pointer_cast<const Impl::LowTrialData>(candidate_)->representation_diagnostics;
}

std::uint64_t AdaptiveVoFSolver::state_revision() const noexcept
{
    return impl_->low_trial_nonce;
}

bool AdaptiveVoFSolver::pending_representation_relaxation() const noexcept
{
    return impl_->published->state.low && impl_->published->state.low->pending_representation_relaxation;
}

AdaptiveVoFSolver::LowMachTrial AdaptiveVoFSolver::preview_representation_change(
    const VoF::RepresentationChangeRequest& request)
{
    require_all(low_mach_enabled(), "AdaptiveVoFSolver representation changes require initialized low-Mach state.");
    require_all(request.source_hierarchy && request.source_hierarchy.get() == hierarchy().get() &&
                    request.accepted_state_revision == impl_->low_trial_nonce,
        "AdaptiveVoFSolver representation request has stale state revision or hierarchy.");
    auto candidate = std::make_shared<Impl::LowTrialData>();
    candidate->owner = impl_.get();
    candidate->context = impl_->context;
    candidate->accepted = impl_->published;
    candidate->nonce = impl_->low_trial_nonce;
    candidate->observation = hierarchy()->observation();
    candidate->representation = true;
    candidate->candidate =
        impl_->context->change_representation(impl_->published->state, request, candidate->representation_diagnostics);
    return LowMachTrial(std::move(candidate));
}

AdaptiveVoFSolver::LowMachTrial AdaptiveVoFSolver::preview_low_mach_step(
    double dt, const VoF::LowMachSources& sources, const VoF::LowMachState* guess)
{
    require_all(low_mach_enabled(), "AdaptiveVoFSolver low-Mach mode is not initialized.");
    auto candidate = std::make_shared<Impl::LowTrialData>();
    candidate->owner = impl_.get();
    candidate->context = impl_->context;
    candidate->accepted = impl_->published;
    candidate->nonce = impl_->low_trial_nonce;
    candidate->observation = hierarchy()->observation();
    candidate->candidate = impl_->context->advance_low(impl_->published->state, dt, sources, guess);
    return LowMachTrial(std::move(candidate));
}

void AdaptiveVoFSolver::commit_low_mach_trial(const LowMachTrial& trial)
{
    const auto data = std::static_pointer_cast<const Impl::LowTrialData>(trial.candidate_);
    require_all(data && data->owner == impl_.get() && data->context == impl_->context &&
                    data->accepted == impl_->published && data->nonce == impl_->low_trial_nonce &&
                    data->observation == hierarchy()->observation(),
        "AdaptiveVoFSolver low-Mach trial is stale or belongs to another solver.");
    auto candidate = std::make_shared<Impl::Published>(data->candidate, impl_->context->mesh);
    impl_->invalidate_low_trials();
    impl_->published.swap(candidate);
}

void AdaptiveVoFSolver::regrid(const std::vector<std::size_t>& refine,
    const std::vector<SimpleFluid::Meshes::AMR::AMRBlock>& coarsen, VoF::RefinementOptions protection,
    VoF::RefinementHints hints)
{
    impl_->regrid(refine, coarsen, protection, std::move(hints));
}
void AdaptiveVoFSolver::regrid(const VoF::RefinementPlan& plan)
{
    require_all(plan.source_hierarchy && plan.source_hierarchy.get() == hierarchy().get() &&
                    plan.source_observation == hierarchy()->observation() &&
                    (!plan.source_state_revision ||
                        (*plan.source_state_revision == impl_->low_trial_nonce && plan.source_state == impl_->published)),
        "AdaptiveVoFSolver refinement plan belongs to a different or replaced hierarchy or accepted state.");
    impl_->regrid(plan.refine, plan.coarsen, {}, {});
}
VoF::RefinementPlan AdaptiveVoFSolver::refinement_plan(VoF::RefinementOptions options, VoF::RefinementHints hints) const
{
    auto plan = impl_->context->geometry->refinement_plan(gas_fraction(), face_velocity(), options, std::move(hints));
    plan.source_state = impl_->published;
    plan.source_state_revision = impl_->low_trial_nonce;
    return plan;
}

void AdaptiveVoFSolver::register_inventory(Inventory inventory)
{
    require_all(!low_mach_enabled() || inventory.carrier != Carrier::MixtureMass,
        "Low-Mach mode uses its typed EOS mass inventories; the legacy constant-density passive MixtureMass carrier "
        "is unavailable.");
    identical(static_cast<double>(impl_->published->state.inventories.size()),
        "AdaptiveVoFSolver inventory registries disagree.");
    impl_->context->validate_inventory(inventory, impl_->published->state);
    require_all(!inventory.name.starts_with("__vof_") &&
                    std::none_of(impl_->published->state.inventories.begin(), impl_->published->state.inventories.end(),
                        [&](const auto& q) { return q.name == inventory.name; }),
        "AdaptiveVoFSolver inventory name is reserved or already registered.");
    auto next = impl_->published->state;
    next.inventories.push_back(std::move(inventory));
    auto candidate = std::make_shared<Impl::Published>(std::move(next), impl_->context->mesh);
    impl_->invalidate_low_trials();
    impl_->published.swap(candidate);
}

const Inventory& AdaptiveVoFSolver::inventory(const std::string& name) const
{
    const auto& quantities = impl_->published->state.inventories;
    const auto found =
        std::find_if(quantities.begin(), quantities.end(), [&](const auto& q) { return q.name == name; });
    if (found == quantities.end())
        throw std::out_of_range("Unknown AdaptiveVoFSolver inventory: " + name);
    return *found;
}
const std::vector<Inventory>& AdaptiveVoFSolver::inventories() const noexcept
{
    return impl_->published->state.inventories;
}

AdaptiveVoFSolver::Checkpoint AdaptiveVoFSolver::checkpoint()
{
    require_all(!impl_->saved, "AdaptiveVoFSolver already has an active checkpoint.");
    Checkpoint lo = 0, hi = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &impl_->next_checkpoint, &lo);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &impl_->next_checkpoint, &hi);
    require_all(lo == hi && lo > 0 && lo < std::numeric_limits<Checkpoint>::max(),
        "AdaptiveVoFSolver checkpoint counter disagrees across ranks or is exhausted.");
    impl_->active_checkpoint = impl_->next_checkpoint++;
    impl_->saved = impl_->published;
    return impl_->active_checkpoint;
}
void AdaptiveVoFSolver::restore(Checkpoint checkpoint)
{
    require_all(
        impl_->saved && checkpoint == impl_->active_checkpoint, "AdaptiveVoFSolver checkpoint token is not active.");
    impl_->invalidate_low_trials();
    impl_->published = impl_->saved;
}
void AdaptiveVoFSolver::release_checkpoint(Checkpoint checkpoint)
{
    require_all(
        impl_->saved && checkpoint == impl_->active_checkpoint, "AdaptiveVoFSolver checkpoint token is not active.");
    impl_->saved.reset();
    impl_->active_checkpoint = 0;
}
bool AdaptiveVoFSolver::has_checkpoint() const noexcept
{
    return bool(impl_->saved);
}
const std::shared_ptr<const AdaptiveVoFSolver::Hierarchy>& AdaptiveVoFSolver::hierarchy() const noexcept
{
    return impl_->context->geometry->hierarchy();
}
const SP<const AdaptiveVoFSolver::Handle>& AdaptiveVoFSolver::mesh() const noexcept
{
    return impl_->context->mesh;
}
std::uint64_t AdaptiveVoFSolver::topology_generation() const noexcept
{
    return hierarchy()->topology_generation();
}
std::uint64_t AdaptiveVoFSolver::partition_generation() const noexcept
{
    return hierarchy()->partition_generation();
}
const VoFMaterial& AdaptiveVoFSolver::material() const noexcept
{
    return impl_->context->material;
}
const VoFSolverOptions& AdaptiveVoFSolver::options() const noexcept
{
    return impl_->context->options;
}
double AdaptiveVoFSolver::time() const noexcept
{
    return impl_->published->state.diagnostics.time;
}
const VoFStepDiagnostics& AdaptiveVoFSolver::diagnostics() const noexcept
{
    return impl_->published->state.diagnostics;
}
const std::vector<double>& AdaptiveVoFSolver::gas_fraction() const noexcept
{
    return impl_->published->state.f;
}
const std::vector<Vector>& AdaptiveVoFSolver::velocity() const noexcept
{
    return impl_->published->state.u;
}
const std::vector<double>& AdaptiveVoFSolver::pressure() const noexcept
{
    return impl_->published->state.p;
}
const std::vector<double>& AdaptiveVoFSolver::curvature() const noexcept
{
    return impl_->published->state.k;
}
const std::vector<double>& AdaptiveVoFSolver::face_velocity() const noexcept
{
    return impl_->published->state.face_u;
}
const AdaptiveVoFSolver::ScalarField& AdaptiveVoFSolver::fraction_field() const noexcept
{
    return impl_->published->f;
}
const AdaptiveVoFSolver::ScalarField& AdaptiveVoFSolver::pressure_field() const noexcept
{
    return impl_->published->p;
}
const AdaptiveVoFSolver::VectorField& AdaptiveVoFSolver::velocity_field() const noexcept
{
    return impl_->published->u;
}

} // namespace SimpleFluid
