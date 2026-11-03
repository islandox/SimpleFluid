/** Test/probe support only. Census scratch and reporting never enter production budgets. */
#pragma once
#include "FVM/details/GeneralVoF/Retention.hh"
#include <cstdlib>
#include <iostream>
namespace SimpleFluid::VoF::test::retention_probe
{
namespace detail = general_vof_detail;
namespace census = PLIC::details::retention;
inline void write_report(const char* workload, const char* checkpoint, const census::Report& r)
{
    const auto flag = [](bool value) { return value ? "true" : "false"; };
    const auto counts = [](const char* name, const census::ByteCount& value)
    { std::cout << ",\"" << name << "\":{\"owners\":" << value.owners << ",\"bytes\":" << value.bytes << '}'; };
    std::cout << "RETENTION {\"workload\":\"" << workload << "\",\"checkpoint\":\"" << checkpoint
              << "\",\"status\":" << unsigned(r.status) << ",\"detail\":\"" << r.detail
              << "\",\"complete_root_manifest\":" << flag(r.complete_root_manifest) << ",\"roots\":" << r.root_count
              << ",\"ledgers\":" << r.ledger_count << ",\"failed_ledgers\":" << r.failed_ledgers
              << ",\"pending_pins\":" << r.pending_pins;
    counts("application_only", r.application_only);
    counts("intersection", r.intersection);
    counts("ledger_only", r.ledger_only);
    counts("owner_union", r.owner_union);
    counts("required_ledger_only", r.required_ledger_only);
    counts("unneeded_ledger_only", r.unneeded_ledger_only);
    std::cout
        << ",\"application_bytes\":" << r.application_bytes << ",\"ledger_bytes\":" << r.ledger_bytes
        << ",\"union_bytes\":" << r.owner_union.bytes << ",\"ledger_only_bytes\":" << r.ledger_only.bytes
        << ",\"required_ledger_only_bytes\":" << r.required_ledger_only.bytes
        << ",\"unneeded_ledger_only_bytes\":" << r.unneeded_ledger_only.bytes
        << ",\"group_overlap_bytes\":" << r.group_overlap_bytes << ",\"retained_records\":" << r.retained_records
        << ",\"required_records\":" << r.required_records
        << ",\"required_record_bytes_lower_bound\":" << r.required_record_bytes_lower_bound
        << ",\"record_byte_lower_bound\":" << r.required_record_bytes_lower_bound
        << ",\"uncovered_required_owners\":" << r.uncovered_required_owners
        << ",\"maximum_recipe_depth\":" << r.maximum_recipe_depth
        << ",\"maximum_identity_depth\":" << r.maximum_identity_depth
        << ",\"maximum_owner_dependency_depth\":" << r.maximum_owner_dependency_depth
        << ",\"maximum_context_dependency_depth\":" << r.maximum_context_dependency_depth
        << ",\"maximum_required_point_dependency_depth\":" << r.maximum_required_point_dependency_depth
        << ",\"maximum_identity_dependency_depth\":" << r.maximum_identity_dependency_depth
        << ",\"maximum_ancestry_dependency_depth\":" << r.maximum_ancestry_dependency_depth
        << ",\"ledger_tracking_bytes\":" << r.ledger_tracking_bytes
        << ",\"ledger_payload_admission\":" << r.ledger_payload_admission
        << ",\"ledger_bookkeeping_admission\":" << r.ledger_bookkeeping_admission
        << ",\"current_ledger_tracking_bytes\":" << r.ledger_tracking_bytes
        << ",\"cumulative_ledger_payload_admission\":" << r.ledger_payload_admission
        << ",\"cumulative_ledger_bookkeeping_admission\":" << r.ledger_bookkeeping_admission
        << ",\"probe_used_bytes\":" << r.probe_used_bytes << ",\"probe_capacity_bytes\":" << r.probe_capacity_bytes
        << ",\"scratch_bytes\":" << r.probe_used_bytes << ",\"scratch_capacity_bytes\":" << r.probe_capacity_bytes
        << ",\"manifest\":{\"scope\":\"PLIC retained owners reachable from supplied quiescent application roots\","
           "\"observation\":\"application boundary; transient FVM planning peak not observed\","
           "\"covered\":[\"cell_shells\",\"hierarchy_explicit_shells\",\"topology_tiles_walls_identities\","
           "\"canonical_point_snapshot\",\"spatial_cache_shells\",\"prepared_source_candidate\","
           "\"copied_point_trajectory_PLIC_provenance\",\"supplied_work_ledgers\"],"
           "\"excluded\":[\"native_recipes_mesh_partition_fields\",\"non_PLIC_application_buffers\","
           "\"trajectory_owner_entry_buffers\",\"allocator_control_blocks\",\"MPI_allocations\","
           "\"unprovided_external_roots\"]},\"groups\":[";
    static constexpr std::array labels{"accepted", "candidate", "prepared_source", "prepared_candidate",
        "copied_points", "copied_trajectories", "active_work", "external_fixture"};
    bool first = true;
    for (const auto& g : r.groups)
    {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << "{\"group\":" << g.group << ",\"label\":\""
                  << (g.group < labels.size() ? labels[g.group] : "unknown") << "\",\"roots\":" << g.roots
                  << ",\"inclusive_bytes\":" << g.inclusive_bytes << ",\"exclusive_bytes\":" << g.exclusive_bytes
                  << '}';
    }
    std::cout << "],\"owners\":[";
    first = true;
    for (const auto& o : r.owners)
    {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << "{\"id\":" << o.id << ",\"kind\":\"" << o.kind << "\",\"recipe_kind\":\"" << o.recipe_kind
                  << "\",\"depth\":" << o.depth << ",\"dependency_depth\":" << o.dependency_depth
                  << ",\"mode\":" << unsigned(o.mode) << ",\"bytes\":" << o.bytes
                  << ",\"metadata_bytes\":" << o.metadata_bytes << ",\"populated_bytes\":" << o.populated_bytes
                  << ",\"unused_capacity_bytes\":" << o.unused_capacity_bytes << ",\"root_groups\":" << o.root_groups
                  << ",\"application\":" << flag(o.application) << ",\"ledger\":" << flag(o.ledger)
                  << ",\"required\":" << flag(o.required) << '}';
    }
    std::cout << "],\"contexts\":[";
    first = true;
    for (const auto& c : r.contexts)
    {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << "{\"owner_id\":" << c.owner_id << ",\"mode\":" << unsigned(c.mode) << ",\"retained\":[";
        for (unsigned i = 0; i < 4; ++i)
            std::cout << (i ? "," : "") << c.retained_records[i];
        std::cout << "],\"required\":[";
        for (unsigned i = 0; i < 4; ++i)
            std::cout << (i ? "," : "") << c.required_records[i];
        std::cout << "],\"authorities\":" << c.authorities << ",\"required_authorities\":" << c.required_authorities
                  << ",\"incidence\":" << c.incidence << ",\"required_incidence\":" << c.required_incidence
                  << ",\"edges\":" << c.edges << ",\"required_edges\":" << c.required_edges
                  << ",\"fact_rows\":" << c.fact_rows << ",\"required_fact_rows\":" << c.required_fact_rows << '}';
    }
    std::cout << "]}\n";
}
template<class Collect> void checkpoint(const char* workload, const char* name, Collect&& collect)
{
    // Explicit test-only opt-in; baseline test invocations do not allocate census scratch.
    if (!std::getenv("SIMPLEFLUID_RETENTION_CENSUS"))
        return;
    try
    {
        std::vector<std::byte> storage(64 * 1024 * 1024);
        census::ProbeScratch scratch(storage);
        detail::RetentionRoots roots(scratch);
        collect(roots);
        auto r = census::inspect_retention(
            roots.roots, roots.ledgers, scratch, {.complete_root_manifest = roots.complete_manifest});
        write_report(workload, name, r);
    }
    catch (...)
    {
        // A diagnostic failure must not turn a production acceptance into rejection.
        std::cout << "RETENTION {\"workload\":\"" << workload << "\",\"checkpoint\":\"" << name
                  << "\",\"status\":\"adapter_incomplete\",\"complete_root_manifest\":false}\n";
    }
}
} // namespace SimpleFluid::VoF::test::retention_probe
