#include <cstddef>
extern "C" {
#include "fastest/custom_tests.h"
#include "fastest/tests.h"
}

FASTEST_CUSTOMTEST_INLINE("standalone/1", FASTEST_TIME_NS, NULL, {
    out->exit_status |= FASTEST_ERROR_ASSERT;
    out->test_flags |= FASTEST_ASSERT_EQ;
});

FASTEST_CUSTOMTEST_INLINE("standalone/2", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
});

FASTEST_CUSTOMTEST_INLINE("standalone/3", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
});

FASTEST_CUSTOMTEST_INLINE("standalone/4", FASTEST_TIME_NS, NULL, {
    out->exit_status |= FASTEST_ERROR_MEMORY;
});

FASTEST_CUSTOMTEST_INLINE("standalone/5", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
});
