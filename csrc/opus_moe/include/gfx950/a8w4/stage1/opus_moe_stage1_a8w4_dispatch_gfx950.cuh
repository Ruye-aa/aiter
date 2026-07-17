// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_pipeline_group_split_gfx950.cuh"
#include "opus_moe_stage1_a8w4_pipeline_pair_kwave_gfx950.cuh"
#include "opus_moe_stage1_a8w4_manifest.h"

#include "opus/hip_minimal.hpp"

namespace opus_moe
{
namespace stage1_a8w4
{

constexpr bool kid_is_valid(int kid)
{ return opus_moe::stage1_a8w4_kid_is_valid(kid); }
constexpr const char* kid_name(int kid)
{ return opus_moe::stage1_a8w4_kid_name(kid); }
constexpr int kid_sort_block_m(int kid)
{ return opus_moe::stage1_a8w4_kid_sort_block_m(kid); }

template<typename Traits>
inline void launch(const OpusMoeStage1A8W4Kargs& kargs, hipStream_t stream)
{
    dim3 grid(Traits::STAGE1_COL_TILES, kargs.sorted_blocks, 1);
    dim3 block(Traits::BLOCK_SIZE);
    if constexpr(Traits::GATE_UP_GROUP_SPLIT)
    {
        pipeline_group_split::opus_moe_stage1_a8w4_kernel_group_split_gfx950<Traits>
            <<<grid, block, 0, stream>>>(kargs);
    }
    else
    {
        static_assert(Traits::PAIR_GATE_UP_SINGLE_GROUP,
                      "stage1 dispatch requires group-split or pair-kwave policy");
        pipeline_pair_kwave::opus_moe_stage1_a8w4_kernel_pair_kwave_gfx950<Traits>
            <<<grid, block, 0, stream>>>(kargs);
    }
}

inline void dispatch(const OpusMoeStage1A8W4Kargs& kargs, hipStream_t stream)
{
    switch(kargs.kernel_id)
    {
    GENERATE_OPUS_MOE_STAGE1_A8W4_DISPATCH_CASES
    default: AITER_CHECK(false, "unreachable A8W4 stage1 kernel dispatch");
    }
}

} // namespace stage1_a8w4
} // namespace opus_moe
