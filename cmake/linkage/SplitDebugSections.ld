/* Split DWARF keeps the full names/types in reusable .dwo files. GCC also
 * emits optional name indexes (especially large with PCH) into every object.
 * Leave those indexes out of native binaries; retain the DWARF skeletons,
 * line tables and addresses needed to locate and read the .dwo files. */
SECTIONS
{
    /DISCARD/ : { *(.debug_gnu_pubnames) *(.debug_gnu_pubtypes) }
}
INSERT AFTER .text;
