RemovalRecordsIntegrity-2in-2out:
| Subroutine                                                                           |            Processor |             Op Stack |                  RAM |                 Hash |                  U32 |
|:-------------------------------------------------------------------------------------|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| tasmlib_mmr_bag_peaks                                                                |         384 (  1.2%) |         664 (  3.0%) |         262 (  0.9%) |         300 (  1.5%) |          16 (  0.1%) |
| ··tasmlib_mmr_bag_peaks_length_is_not_zero                                           |         354 (  1.1%) |         638 (  2.9%) |         260 (  0.9%) |         300 (  1.5%) |           0 (  0.0%) |
| ····tasmlib_mmr_bag_peaks_length_is_not_zero_or_one                                  |         338 (  1.1%) |         626 (  2.8%) |         260 (  0.9%) |         300 (  1.5%) |           0 (  0.0%) |
| ······tasmlib_mmr_bag_peaks_loop                                                     |         302 (  1.0%) |         600 (  2.7%) |         250 (  0.8%) |         300 (  1.5%) |           0 (  0.0%) |
| tasmlib_hashing_merkle_verify                                                        |          72 (  0.2%) |          52 (  0.2%) |           0 (  0.0%) |          36 (  0.2%) |          37 (  0.3%) |
| ··tasmlib_hashing_merkle_verify_tree_height_is_not_zero                              |          24 (  0.1%) |           4 (  0.0%) |           0 (  0.0%) |          36 (  0.2%) |          24 (  0.2%) |
| ····tasmlib_hashing_merkle_verify_traverse_tree                                      |          14 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          36 (  0.2%) |          24 (  0.2%) |
| tasmlib_hashing_algebraic_hasher_hash_varlen                                         |       16676 ( 52.9%) |       11146 ( 50.5%) |       27294 ( 92.7%) |       16394 ( 83.4%) |          33 (  0.3%) |
| ··tasmlib_hashing_absorb_multiple                                                    |       16648 ( 52.8%) |       11116 ( 50.4%) |       27294 ( 92.7%) |       16380 ( 83.3%) |          33 (  0.3%) |
| ····tasmlib_hashing_absorb_multiple_hash_all_full_chunks                             |       16380 ( 51.9%) |       10920 ( 49.5%) |       27280 ( 92.6%) |       16368 ( 83.2%) |           0 (  0.0%) |
| ····tasmlib_hashing_absorb_multiple_pad_varnum_zeros                                 |          56 (  0.2%) |          36 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ····tasmlib_hashing_absorb_multiple_read_remainder                                   |         138 (  0.4%) |          78 (  0.4%) |          14 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| for_all_utxos                                                                        |       13936 ( 44.2%) |        9742 ( 44.1%) |        1848 (  6.3%) |        1400 (  7.1%) |       13037 ( 99.3%) |
| ··tasmlib_hashing_algebraic_hasher_hash_varlen                                       |         312 (  1.0%) |         234 (  1.1%) |          38 (  0.1%) |          38 (  0.2%) |           6 (  0.0%) |
| ····tasmlib_hashing_absorb_multiple                                                  |         284 (  0.9%) |         204 (  0.9%) |          38 (  0.1%) |          24 (  0.1%) |           6 (  0.0%) |
| ······tasmlib_hashing_absorb_multiple_hash_all_full_chunks                           |          24 (  0.1%) |          16 (  0.1%) |          20 (  0.1%) |          12 (  0.1%) |           0 (  0.0%) |
| ······tasmlib_hashing_absorb_multiple_pad_varnum_zeros                               |          12 (  0.0%) |           8 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_hashing_absorb_multiple_read_remainder                                 |         174 (  0.6%) |          98 (  0.4%) |          18 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_neptune_mutator_set_commit                                                 |           8 (  0.0%) |          20 (  0.1%) |           0 (  0.0%) |          24 (  0.1%) |           0 (  0.0%) |
| ··tasmlib_mmr_verify_from_secret_in_leaf_index_on_stack                              |        3860 ( 12.2%) |        2008 (  9.1%) |          10 (  0.0%) |         756 (  3.8%) |        3954 ( 30.1%) |
| ····tasmlib_mmr_leaf_index_to_mt_index_and_peak_index                                |         248 (  0.8%) |         166 (  0.8%) |           0 (  0.0%) |           0 (  0.0%) |         783 (  6.0%) |
| ······tasmlib_arithmetic_u64_lt_preserve_args                                        |          26 (  0.1%) |          22 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         129 (  1.0%) |
| ······tasmlib_arithmetic_u64_log_2_floor                                             |          26 (  0.1%) |          18 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |          66 (  0.5%) |
| ········tasmlib_arithmetic_u64_log_2_floor_hi_not_zero                               |          14 (  0.0%) |          10 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          66 (  0.5%) |
| ······tasmlib_arithmetic_u64_pow2                                                    |          10 (  0.0%) |           6 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          40 (  0.3%) |
| ······tasmlib_arithmetic_u64_decr                                                    |          40 (  0.1%) |          32 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_arithmetic_u64_decr_carry                                            |          24 (  0.1%) |          20 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_arithmetic_u64_and                                                     |          24 (  0.1%) |           8 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |         196 (  1.5%) |
| ······tasmlib_arithmetic_u64_add                                                     |          30 (  0.1%) |          16 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         128 (  1.0%) |
| ······tasmlib_arithmetic_u64_popcount                                                |          24 (  0.1%) |           4 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          95 (  0.7%) |
| ····tasmlib_mmr_verify_from_secret_in_leaf_index_on_stack_auth_path_loop             |        3556 ( 11.3%) |        1780 (  8.1%) |           0 (  0.0%) |         756 (  3.8%) |        3167 ( 24.1%) |
| ······tasmlib_arithmetic_u64_eq                                                      |         896 (  2.8%) |         384 (  1.7%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_hashing_merkle_step_u64_index                                          |        1890 (  6.0%) |         756 (  3.4%) |           0 (  0.0%) |         756 (  3.8%) |        3167 ( 24.1%) |
| ····tasmlib_list_get_element___digest                                                |          28 (  0.1%) |          30 (  0.1%) |          10 (  0.0%) |           0 (  0.0%) |           4 (  0.0%) |
| ··tasmlib_neptune_mutator_get_swbf_indices_1048576_45                                |        9214 ( 29.2%) |        6910 ( 31.3%) |         938 (  3.2%) |          86 (  0.4%) |        9077 ( 69.2%) |
| ····tasmlib_arithmetic_u128_shift_right_static_3                                     |          50 (  0.2%) |          24 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         128 (  1.0%) |
| ····tasmlib_arithmetic_u128_shift_left_static_12                                     |          46 (  0.1%) |          24 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         128 (  1.0%) |
| ····tasmlib_hashing_algebraic_hasher_sample_indices                                  |        5538 ( 17.6%) |        3980 ( 18.0%) |         478 (  1.6%) |          60 (  0.3%) |        5818 ( 44.3%) |
| ······tasmlib_list_new___u32                                                         |          52 (  0.2%) |          40 (  0.2%) |           6 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_memory_dyn_malloc                                                    |          72 (  0.2%) |          62 (  0.3%) |           8 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··········tasmlib_memory_dyn_malloc_initialize                                       |           4 (  0.0%) |           2 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_hashing_algebraic_hasher_sample_indices_main_loop                      |        5470 ( 17.3%) |        3932 ( 17.8%) |         472 (  1.6%) |          60 (  0.3%) |        5818 ( 44.3%) |
| ········tasmlib_list_length___u32                                                    |         448 (  1.4%) |         224 (  1.0%) |         112 (  0.4%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_hashing_algebraic_hasher_sample_indices_then_reduce_and_save         |        3150 ( 10.0%) |        1890 (  8.6%) |         360 (  1.2%) |           0 (  0.0%) |        5818 ( 44.3%) |
| ··········tasmlib_list_push___u32                                                    |        1710 (  5.4%) |        1260 (  5.7%) |         360 (  1.2%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_hashing_algebraic_hasher_sample_indices_else_drop_tip                |          60 (  0.2%) |          10 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ····tasmlib_list_higher_order_u32_map_u32_to_u128_add_another_u128                   |        3530 ( 11.2%) |        2788 ( 12.6%) |         460 (  1.6%) |           0 (  0.0%) |        3003 ( 22.9%) |
| ······tasmlib_list_new___u128                                                        |          48 (  0.2%) |          38 (  0.2%) |           6 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_list_higher_order_u32_map_u32_to_u128_add_another_u128_loop            |        3432 ( 10.9%) |        2708 ( 12.3%) |         450 (  1.5%) |           0 (  0.0%) |        3003 ( 22.9%) |
| ··tasmlib_hashing_algebraic_hasher_hash_static_size_180                              |         228 (  0.7%) |         188 (  0.9%) |         720 (  2.4%) |         484 (  2.5%) |           0 (  0.0%) |
| ····tasmlib_hashing_absorb_multiple_static_size_180                                  |         168 (  0.5%) |         128 (  0.6%) |         720 (  2.4%) |         456 (  2.3%) |           0 (  0.0%) |
| Total                                                                                |       31548 (100.0%) |       22067 (100.0%) |       29459 (100.0%) |       19667 (100.0%) |       13123 (100.0%) |
