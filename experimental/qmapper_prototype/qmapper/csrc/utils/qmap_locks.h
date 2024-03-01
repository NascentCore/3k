#pragma once

#include <atomic>

#ifndef QMAP_LOCKS
#define QMAP_LOCKS

namespace qmap {
namespace utils {

class SpinLock {
public:
    void lock();
    void unlock();

private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

}; // namespace utils
}; // namespace qmap

#endif