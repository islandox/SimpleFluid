// Included in the controller's anonymous namespace.
std::shared_ptr<const InterphaseConfiguration> configure_qbmm_sources(
    const Solver& flow, ReactingMixtureOptions& options)
{
    auto thermal = options.interphase ? options.interphase->configuration : nullptr;
    const auto& families = flow.options().qbmm_families;
    const double tolerance = options.qbmm_source_substep_relative_tolerance;
    require(std::isfinite(tolerance) && tolerance >= 0 && tolerance < 1 &&
                options.qbmm_source_maximum_substeps >= 1,
        "QBMM source substep tolerance must be in [0, 1) and its stage budget positive.");
    require(tolerance == 0 || (!families.empty() && options.qbmm_population),
        "QBMM source substeps require bound families and a QBMM source provider.");
    if (families.empty())
    {
        require(!options.qbmm_population, "A QBMM law requires native QBMM family bindings.");
        return thermal;
    }
    const auto& catalog = options.mechanism->catalog();
    require(options.phases.size() == families.size(), "QBMM physical and chemical phase layouts differ.");
    if (options.population_events)
    {
        std::set<std::string> selected;
        const auto& names = options.population_events->populations;
        require(!names.empty(), "A coexisting legacy count law must declare its unbound population names.");
        for (const auto& name : names)
        {
            const auto& roles = flow.options().phase_roles;
            const auto it = std::ranges::find(roles, name, &Solver::PhaseRoleDescriptor::identity);
            require(it != roles.end() && it->role == Solver::PhaseRole::DispersedGas && !families[it - roles.begin()] &&
                        selected.insert(name).second,
                "Legacy count callback claims a QBMM, duplicate, or unknown population.");
        }
    }
    for (const auto& reaction : options.mechanism->reactions())
        for (const auto& term : reaction.changes)
            for (const auto& family : families)
                require(!family || term.phase != SpeciesPhase::Dispersed || term.population != family->population(),
                    "Reaction stoichiometry on a QBMM family requires a paired allocation.");
    for (const auto& face : options.face_transport)
        require(face.physical_phase < families.size() && (!face.diffusion || !families[face.physical_phase]),
            "Unpaired species diffusion on a QBMM phase is unsupported.");
    if (thermal)
        for (const auto& stage : thermal->stages)
            for (const auto& slot : stage.material)
                require(slot.donor < families.size() && slot.recipient < families.size() && !families[slot.donor] &&
                            !families[slot.recipient],
                    "Generic interphase material slots cannot change a QBMM family.");
    if (!options.qbmm_population)
        return thermal;
    const auto& provider = *options.qbmm_population;
    require((provider.evaluate || !provider.laws.empty() || provider.evaluate_joint) &&
                !provider.metadata.identity.empty() && provider.metadata.revision > 0 &&
                provider.metadata.schema_fingerprint == catalog.fingerprint(),
        "QBMM provider needs a valid identity, revision, matching species catalog and local law.");
    require(!provider.evaluate_joint || (!provider.evaluate && provider.laws.empty()),
        "A joint QBMM provider owns all bound families and requires exclusive evaluation.");
    for (const auto& family : families)
        if (family)
        {
            const bool selected = provider.evaluate || provider.evaluate_joint ||
                                  std::ranges::any_of(provider.laws,
                                      [&](const auto& law) { return law && family->declares_law(law->metadata); });
            require(family->catalog().fingerprint() == catalog.fingerprint() &&
                        (!selected || (family->closure_versions().source_identity == provider.metadata.identity &&
                                          family->closure_versions().source == provider.metadata.revision)),
                "QBMM source policy differs from the selected family.");
        }
    for (std::size_t i = 0; i < provider.laws.size(); ++i)
    {
        const auto& law = provider.laws[i];
        require(law && law->evaluate, "A QBMM constitutive law requires an immutable local evaluator.");
        validate_qbmm_law_metadata(law->metadata, catalog.species().size());
        require(
            !qbmm_is_transport_law(law->metadata), "Slip and property laws cannot own integrated population events.");
        for (std::size_t j = 0; j < i; ++j)
            require(law->metadata != provider.laws[j]->metadata,
                "A QBMM law contract may have only one evaluator; use distinct family contracts for distinct laws.");
        bool bound = false;
        for (const auto& family : families)
            if (family)
                bound = bound || family->declares_law(law->metadata);
        require(bound, "Each supplied QBMM constitutive law must be declared by at least one bound family.");
    }
    for (const auto& family : families)
        if (family)
            for (const auto& declared : family->closure_versions().laws)
                if (!qbmm_is_transport_law(declared))
                    require(provider.evaluate || provider.evaluate_joint ||
                                std::ranges::any_of(
                                    provider.laws, [&](const auto& law) { return law->metadata == declared; }),
                        "A declared population law needs a matching evaluator or the integrated provider adapter.");
    if (!provider.transfers)
        return thermal;
    const auto transfers = std::make_shared<InterphaseConfiguration>(*provider.transfers);
    require(transfers->stages.size() == 1 && transfers->stages[0].kind == InterphaseStageKind::Material &&
                transfers->metadata.schema_fingerprint == catalog.fingerprint() &&
                transfers->phase_enthalpy.size() == families.size(),
        "QBMM phase transfer requires one simultaneous material stage in native phase order.");
    std::set<std::tuple<SpeciesId, size_t, size_t>> material_slots;
    for (const auto& slot : transfers->stages[0].material)
    {
        require(material_slots.emplace(slot.species, slot.donor, slot.recipient).second,
            "Each directed QBMM material slot must be unique.");
        const auto s = catalog.index(slot.species);
        require(slot.donor < families.size() && slot.recipient < families.size() &&
                    bool(families[slot.donor]) != bool(families[slot.recipient]) &&
                    catalog.allows(slot.species, compartment(options.phases[slot.donor], s)) &&
                    catalog.allows(slot.species, compartment(options.phases[slot.recipient], s)),
            "QBMM material slots must link a bound family and a compatible counterpart.");
        const auto other = families[slot.donor] ? slot.recipient : slot.donor;
        require(flow.options().phase_roles[other].role == Solver::PhaseRole::BulkLiquid,
            "Native QBMM internal material exchange requires the liquid carrier; bulk-gas conversion is unsupported.");
        require(options.phases[other].population.empty(),
            "A QBMM material counterpart must match the unnamed chemical compartment in the Q1 transfer.");
    }
    if (provider.evaluate_joint)
    {
        // A simultaneous joint stage uses the same thermal visitation order
        // when the physical phase slots or supplied transfer slots are permuted.
        const auto& roles = flow.options().phase_roles;
        std::ranges::sort(transfers->stages.front().material, {}, [&](const MaterialTransferSlot& slot)
            {
                const auto& donor = roles[slot.donor];
                const auto& recipient = roles[slot.recipient];
                return std::tuple{slot.species, donor.role, std::string_view(donor.identity), recipient.role,
                    std::string_view(recipient.identity)};
            });
    }
    if (!flow.state().cells.empty())
        validate_interphase_configuration(*transfers, flow.state().cells.front().phases);
    options.qbmm_population = std::make_shared<const MixtureQbmmProvider>(
        MixtureQbmmProvider{provider.metadata, transfers, provider.evaluate, provider.laws, provider.evaluate_joint});
    if (!thermal)
        return transfers;
    auto combined = std::make_shared<InterphaseConfiguration>(*thermal);
    require(combined->phase_enthalpy.size() == families.size(), "Generic interphase enthalpy layout differs.");
    combined->stages.push_back(transfers->stages.front());
    for (const auto& slot : transfers->stages.front().material)
    {
        auto& law = combined->phase_enthalpy.at(slot.donor);
        const auto& supplied = transfers->phase_enthalpy.at(slot.donor);
        require(bool(supplied), "A QBMM material donor needs a partial enthalpy law.");
        const bool shared_donor = std::ranges::any_of(thermal->stages,
            [&](const auto& stage)
            {
                return std::ranges::any_of(
                    stage.material, [&](const auto& generic) { return generic.donor == slot.donor; });
            });
        if (shared_donor)
            require(law && law->metadata == supplied->metadata &&
                        law->bounds.minimum_temperature == supplied->bounds.minimum_temperature &&
                        law->bounds.maximum_temperature == supplied->bounds.maximum_temperature &&
                        law->bounds.minimum_pressure == supplied->bounds.minimum_pressure &&
                        law->bounds.maximum_pressure == supplied->bounds.maximum_pressure,
                "A shared material donor requires the same partial enthalpy law and validity domain across stages.");
        else
            law = supplied;
    }
    if (transfers->caloric_domain)
    {
        require(!combined->caloric_domain || combined->caloric_domain->metadata == transfers->caloric_domain->metadata,
            "Combined thermal configuration requires a common whole-plan caloric domain provider.");
        combined->caloric_domain = transfers->caloric_domain;
    }
    if (!flow.state().cells.empty())
        validate_interphase_configuration(*combined, flow.state().cells.front().phases);
    return combined;
}
