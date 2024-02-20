#ifndef QMAP_COMPILER_DEFS
#define QMAP_COMPILER_DEFS

namespace qmap {
namespace utils {

#define qmap_compiler_fence()  asm volatile("":: "memory")
#define class_offset_of(class, member) (reinterpret_cast<char *>(&static_cast<class *>(0)->member))
#define class_container_of(class, member, member_p) (reinterpret_cast<class *>(reinterpret_cast<char *>(member_p) - class_offset_of(class, member)))

}; // namespace utils
}; // namespace qmap

#endif