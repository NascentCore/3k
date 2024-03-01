#pragma once

#ifndef QMAP_CPU_IFACE_H
#define QMAP_CPU_IFACE_H

#define CACHE_LINE_SIZE 128
#define memory_bus_store_fence() asm volatile ("sfence" ::: "memory")
#define memory_bus_load_fence() asm volatile ("lfence" ::: "memory")
#define cpu_fence() asm volatile("" ::: "memory")
#define cpu_load_fence() asm volatile("" ::: "memory")
#define cpu_mfence() asm volatile("" ::: "memory")

#endif