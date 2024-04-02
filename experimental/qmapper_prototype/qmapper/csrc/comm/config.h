#pragma once

#include <cstddef>
#include <vector>

#ifndef QMAP_COMM_CONFIG
#define QMAP_COMM_CONFIG

namespace qmap {
namespace comm {

typedef struct register_info {
    void *elem;
    size_t elem_size;
} register_info_t;

class CommConfig {
public:
    static std::vector<register_info_t> mc_libs;
};

}; // namespace comm
}; // namespace qmap

#endif