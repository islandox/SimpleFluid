/** @file ExactGeometryInstrumentation.hh
 * @brief Opt-in exact-kernel operation counts for numerical-path qualification.
 */
#pragma once
#include <cstdint>

namespace SimpleFluid::Meshes
{
/** Thread-local API/kernel event counts, not counts of primitive bigint
 * operations. Operand maxima sample retained measure results; transient
 * expression temporaries and unrelated exact libraries are not covered.
 * Source coordinate/support export belongs to the mesh authority bridge. */
struct ExactGeometryStatistics
{
    std::uint64_t source_coordinate_exports = 0, source_support_exports = 0;
    std::uint64_t volume_evaluations = 0, moment_evaluations = 0;
    std::uint64_t clip_constructions = 0, intersection_constructions = 0;
    std::uint64_t face_measure_evaluations = 0, source_membership_predicates = 0;
    std::uint64_t measure_values = 0, measure_arithmetic = 0, cover_certifications = 0;
    std::uint64_t maximum_result_numerator_bits = 0, maximum_result_denominator_bits = 0;
};
/** Disabled by default so ordinary benchmarks do not include operand sampling.
 * Reset does not alter enablement. Each MPI rank controls its own calling thread. */
void enable_exact_geometry_statistics(bool enabled);
void reset_exact_geometry_statistics();
[[nodiscard]] ExactGeometryStatistics exact_geometry_statistics();
/** Calling image/thread identity; use only for duplicate-domain suppression. */
[[nodiscard]] const void* exact_geometry_statistics_domain();
} // namespace SimpleFluid::Meshes
