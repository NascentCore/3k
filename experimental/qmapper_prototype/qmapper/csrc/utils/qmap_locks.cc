#include "qmap_locks.h"
#include <atomic>

void qmap::utils::SpinLock::lock() {
    while (flag.test_and_set(std::memory_order_acquire)) {}
}

void qmap::utils::SpinLock::unlock() {
    flag.clear(std::memory_order_release);
}

