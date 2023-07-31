/*
 * SKI - Systematic Kernel Interleaving explorer (http://ski.mpi-sws.org)
 *
 * Copyright (c) 2013-2015 Pedro Fonseca
 *
 *
 * This work is licensed under the terms of the GNU GPL, version 3.  See
 * the GPL3 file in SKI's top-level directory.
 *
 */


#ifndef IPFILTER_H
#define IPFILTER_H



#include "ski-config.h"
#include "ski-by-eip.h"

#ifdef SKI_IPFILTER_HASH

//#define DHASH_DEBUG 1

#include "uthash.h"

typedef struct struct_ski_ipfilter_hash_entry{
    int eip;
    UT_hash_handle hh; /* makes this structure hashable */
} ski_ipfilter_hash_entry;

extern ski_ipfilter_hash_entry * ski_ipfilter_hash; 

typedef struct struct_pmc_hash_entry_by_mem_addr {
    unsigned int mem_addr;

    UT_hash_handle hh; /* makes this structure hashable */
} pmc_hash_entry_by_mem_addr;

typedef struct struct_pmc_hash_entry_by_cpu1_eip {
    unsigned int cpu1_ins_addr;
    pmc_hash_entry_by_mem_addr *by_this_mem_addr;
    int num_by_this_mem_addr;

    UT_hash_handle hh; /* makes this structure hashable */
} pmc_hash_entry_by_cpu1_eip;

typedef struct struct_pmc_hash_entry_by_cpu0_eip {
    unsigned int cpu0_ins_addr;
    pmc_hash_entry_by_cpu1_eip *by_this_cpu1_ins;
    int num_by_this_cpu1_ins;

    UT_hash_handle hh; /* makes this structure hashable */
} pmc_hash_entry_by_cpu0_eip;

#else

#define MAX_SKI_IPFILTER_RANGES 100

typedef struct struct_ski_range{
    int start;
    int end;
} ski_ipfilter_range;

extern ski_ipfilter_range ski_init_ipfilter_ranges[];
extern int ski_init_ipfilter_ranges_count;
#endif






#ifdef SKI_IPFILTER_HASH
extern ski_ipfilter_hash_entry * ski_ipfilter_hash;
extern ski_preeemption_by_eip_entry * ski_preeemption_by_eip;

#define MAX_IPFILTER_HASH (128*1024)
extern ski_ipfilter_hash_entry ski_ipfilter_hash_values[MAX_IPFILTER_HASH];
extern int ski_ipfilter_hash_values_n;

// Backup up variables 
extern ski_ipfilter_hash_entry *ski_ipfilter_hash_backup;
extern ski_ipfilter_hash_entry ski_ipfilter_hash_values_backup[MAX_IPFILTER_HASH];
extern int ski_ipfilter_hash_values_backup_n;

#else // SKI_IPFILTER_HASH

extern ski_ipfilter_range ski_init_ipfilter_ranges[MAX_SKI_IPFILTER_RANGES];
extern int ski_init_ipfilter_ranges_count;

#endif // SKI_IPFILTER_HASH










#endif
