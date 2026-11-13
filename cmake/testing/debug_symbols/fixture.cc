#include <cstdio>

struct DebugSymbolsFixtureDeliberatelyDescriptiveState
{
    volatile int value_that_must_remain_visible_at_the_source_breakpoint;
    int independent_expected_increment;
};

extern "C" __attribute__((noinline)) int debug_symbols_fixture_compute(int input)
{
    DebugSymbolsFixtureDeliberatelyDescriptiveState state{input, 5};
    state.value_that_must_remain_visible_at_the_source_breakpoint =
        state.value_that_must_remain_visible_at_the_source_breakpoint + state.independent_expected_increment; // SOURCE_BREAKPOINT
    return state.value_that_must_remain_visible_at_the_source_breakpoint;
}

#ifdef FIXTURE_EXECUTABLE
int main()
{
    const int result = debug_symbols_fixture_compute(37);
    std::printf("debug-symbol-fixture=%d\n", result);
    return result == 42 ? 0 : 1;
}
#endif
