/** @file TrilinosBackend.cc
 * @brief Exposes the ABI marker for the selected Trilinos backend archives.
 */
// The dependency implementations are supplied by the selected PIC archives.
// This stable C entry point also lets the backend audit identify the owner.
extern "C" unsigned simplefluid_trilinos_backend_abi_version()
{
    return 1;
}
