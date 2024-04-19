tasm_neptune_transaction_removal_records_integrity:
| Subroutine                                                                                      |            Processor |             Op Stack |                  RAM |                 Hash |                  U32 |
|:------------------------------------------------------------------------------------------------|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| tasm_neptune_transaction_transaction_kernel_mast_hash                                           |        4723 ( 16.4%) |        8500 ( 31.2%) |        2692 ( 48.3%) |        1639 ( 29.4%) |         157 (  1.2%) |
| ··tasmlib_list_new___digest                                                                     |          96 (  0.3%) |          78 (  0.3%) |          12 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |
| ····tasmlib_memory_dyn_malloc                                                                   |         164 (  0.6%) |         152 (  0.6%) |          20 (  0.4%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_memory_dyn_malloc_initialize                                                      |           3 (  0.0%) |           2 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_set_length___digest                                                              |           5 (  0.0%) |           3 (  0.0%) |           1 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_hashing_algebraic_hasher_hash_varlen                                                  |        4126 ( 14.3%) |        7869 ( 28.9%) |        2527 ( 45.4%) |        1597 ( 28.6%) |          55 (  0.4%) |
| ····tasmlib_hashing_absorb_multiple                                                             |        4028 ( 14.0%) |        7764 ( 28.5%) |        2527 ( 45.4%) |        1548 ( 27.7%) |          55 (  0.4%) |
| ······tasmlib_hashing_absorb_multiple_hash_all_full_chunks                                      |        3047 ( 10.6%) |        7056 ( 25.9%) |        2510 ( 45.1%) |        1506 ( 27.0%) |           0 (  0.0%) |
| ······tasmlib_hashing_absorb_multiple_pad_varnum_zeros                                          |         541 (  1.9%) |         350 (  1.3%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_hashing_absorb_multiple_read_remainder                                            |         188 (  0.7%) |         113 (  0.4%) |          17 (  0.3%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_set_element___digest                                                             |         120 (  0.4%) |         165 (  0.6%) |          75 (  1.3%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_get_element___digest                                                             |         195 (  0.7%) |         225 (  0.8%) |          75 (  1.3%) |           0 (  0.0%) |         102 (  0.8%) |
| tasmlib_memory_push_ram_to_stack___digest                                                       |          10 (  0.0%) |          16 (  0.1%) |          10 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |
| tasmlib_mmr_bag_peaks                                                                           |        1140 (  4.0%) |        1196 (  4.4%) |         212 (  3.8%) |         240 (  4.3%) |         198 (  1.5%) |
| ··tasmlib_list_length___digest                                                                  |          12 (  0.0%) |           8 (  0.0%) |           4 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_mmr_bag_peaks_length_is_not_zero                                                      |        1114 (  3.9%) |        1178 (  4.3%) |         210 (  3.8%) |         240 (  4.3%) |         198 (  1.5%) |
| ····tasmlib_mmr_bag_peaks_length_is_not_zero_or_one                                             |        1098 (  3.8%) |        1166 (  4.3%) |         210 (  3.8%) |         240 (  4.3%) |         198 (  1.5%) |
| ······tasmlib_list_get_element___digest                                                         |          52 (  0.2%) |          60 (  0.2%) |          20 (  0.4%) |           0 (  0.0%) |          18 (  0.1%) |
| ······tasmlib_mmr_bag_peaks_loop                                                                |         998 (  3.5%) |        1072 (  3.9%) |         190 (  3.4%) |         228 (  4.1%) |         180 (  1.3%) |
| ········tasmlib_list_get_element___digest                                                       |         520 (  1.8%) |         600 (  2.2%) |         200 (  3.6%) |           0 (  0.0%) |         180 (  1.3%) |
| tasmlib_list_contiguous_list_get_pointer_list                                                   |         303 (  1.1%) |         228 (  0.8%) |          27 (  0.5%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_contiguous_list_get_length                                                       |           9 (  0.0%) |           6 (  0.0%) |           3 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_new___void_pointer                                                               |          92 (  0.3%) |          76 (  0.3%) |          12 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_set_length___void_pointer                                                        |          15 (  0.1%) |           9 (  0.0%) |           3 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_contiguous_list_get_pointer_list_loop                                            |         165 (  0.6%) |         132 (  0.5%) |          12 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |
| ····tasmlib_list_set_element___void_pointer                                                     |          36 (  0.1%) |          30 (  0.1%) |           6 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_hash_utxo                            |         418 (  1.5%) |         356 (  1.3%) |          57 (  1.0%) |          38 (  0.7%) |          11 (  0.1%) |
| ··tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_hash_utxo_loop                     |         369 (  1.3%) |         316 (  1.2%) |          52 (  0.9%) |          38 (  0.7%) |          11 (  0.1%) |
| ····tasm_neptune_transaction_hash_utxo                                                          |         332 (  1.2%) |         280 (  1.0%) |          40 (  0.7%) |          38 (  0.7%) |          11 (  0.1%) |
| ······tasmlib_hashing_algebraic_hasher_hash_varlen                                              |        1836 (  6.4%) |        2786 ( 10.2%) |         758 ( 13.6%) |         522 (  9.4%) |          20 (  0.1%) |
| ········tasmlib_hashing_absorb_multiple                                                         |        1752 (  6.1%) |        2696 (  9.9%) |         758 ( 13.6%) |         480 (  8.6%) |          20 (  0.1%) |
| ··········tasmlib_hashing_absorb_multiple_hash_all_full_chunks                                  |         918 (  3.2%) |        2096 (  7.7%) |         740 ( 13.3%) |         444 (  8.0%) |           0 (  0.0%) |
| ··········tasmlib_hashing_absorb_multiple_pad_varnum_zeros                                      |         426 (  1.5%) |         276 (  1.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··········tasmlib_hashing_absorb_multiple_read_remainder                                        |         192 (  0.7%) |         114 (  0.4%) |          18 (  0.3%) |           0 (  0.0%) |           0 (  0.0%) |
| tasmlib_list_higher_order_u32_zip_digest_with_void_pointer                                      |         102 (  0.4%) |          99 (  0.4%) |          30 (  0.5%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_new___tuple_Ldigest___void_pointerR                                              |          23 (  0.1%) |          19 (  0.1%) |           3 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_higher_order_u32_zip_digest_with_void_pointer_loop                               |          45 (  0.2%) |          52 (  0.2%) |          24 (  0.4%) |           0 (  0.0%) |           0 (  0.0%) |
| tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_compute_indices                      |        9516 ( 33.0%) |        7136 ( 26.2%) |         995 ( 17.9%) |          86 (  1.5%) |        9079 ( 66.8%) |
| ··tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_compute_indices_loop               |        9471 ( 32.9%) |        7100 ( 26.1%) |         990 ( 17.8%) |          86 (  1.5%) |        9079 ( 66.8%) |
| ····tasm_neptune_transaction_compute_indices                                                    |        9432 ( 32.7%) |        7062 ( 25.9%) |         976 ( 17.5%) |          86 (  1.5%) |        9079 ( 66.8%) |
| ······tasmlib_memory_push_ram_to_stack___u64                                                    |          10 (  0.0%) |          10 (  0.0%) |           4 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_memory_push_ram_to_stack___digest                                                 |          40 (  0.1%) |          64 (  0.2%) |          40 (  0.7%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_neptune_mutator_get_swbf_indices_1048576_45                                       |        9208 ( 31.9%) |        6908 ( 25.4%) |         938 ( 16.8%) |          86 (  1.5%) |        9079 ( 66.8%) |
| ········tasmlib_arithmetic_u128_shift_right_static_3                                            |          48 (  0.2%) |          24 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         128 (  0.9%) |
| ········tasmlib_arithmetic_u128_shift_left_static_12                                            |          44 (  0.2%) |          24 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         130 (  1.0%) |
| ········tasmlib_hashing_algebraic_hasher_sample_indices                                         |        5532 ( 19.2%) |        3978 ( 14.6%) |         478 (  8.6%) |          60 (  1.1%) |        5815 ( 42.8%) |
| ··········tasmlib_list_new___u32                                                                |          46 (  0.2%) |          38 (  0.1%) |           6 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ············tasmlib_memory_dyn_malloc                                                           |          64 (  0.2%) |          60 (  0.2%) |           8 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··········tasmlib_hashing_algebraic_hasher_sample_indices_main_loop                             |        5468 ( 19.0%) |        3932 ( 14.4%) |         472 (  8.5%) |          60 (  1.1%) |        5815 ( 42.8%) |
| ············tasmlib_list_length___u32                                                           |         336 (  1.2%) |         224 (  0.8%) |         112 (  2.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ············tasmlib_hashing_algebraic_hasher_sample_indices_then_reduce_and_save                |        3060 ( 10.6%) |        1890 (  6.9%) |         360 (  6.5%) |           0 (  0.0%) |        5815 ( 42.8%) |
| ··············tasmlib_list_push___u32                                                           |        1620 (  5.6%) |        1260 (  4.6%) |         360 (  6.5%) |           0 (  0.0%) |           0 (  0.0%) |
| ············tasmlib_hashing_algebraic_hasher_sample_indices_else_drop_tip                       |          50 (  0.2%) |          10 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_list_higher_order_u32_map_u32_to_u128_add_another_u128                          |        3528 ( 12.2%) |        2788 ( 10.2%) |         460 (  8.3%) |           0 (  0.0%) |        3006 ( 22.1%) |
| ··········tasmlib_list_new___u128                                                               |          46 (  0.2%) |          38 (  0.1%) |           6 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··········tasmlib_list_higher_order_u32_map_u32_to_u128_add_another_u128_loop                   |        3430 ( 11.9%) |        2708 (  9.9%) |         450 (  8.1%) |           0 (  0.0%) |        3006 ( 22.1%) |
| tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_hash_index_list                      |         862 (  3.0%) |        1344 (  4.9%) |         379 (  6.8%) |         242 (  4.3%) |           9 (  0.1%) |
| ··tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_hash_index_list_loop               |         813 (  2.8%) |        1304 (  4.8%) |         374 (  6.7%) |         242 (  4.3%) |           9 (  0.1%) |
| ····tasm_neptune_transaction_hash_index_list                                                    |         776 (  2.7%) |        1268 (  4.7%) |         362 (  6.5%) |         242 (  4.3%) |           9 (  0.1%) |
| tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_hash_removal_record_indices          |         866 (  3.0%) |        1348 (  5.0%) |         379 (  6.8%) |         242 (  4.3%) |           0 (  0.0%) |
| ··tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_hash_removal_record_indices_loop   |         817 (  2.8%) |        1308 (  4.8%) |         374 (  6.7%) |         242 (  4.3%) |           0 (  0.0%) |
| ····tasm_neptune_transaction_hash_removal_record_indices                                        |         780 (  2.7%) |        1272 (  4.7%) |         362 (  6.5%) |         242 (  4.3%) |           0 (  0.0%) |
| tasmlib_list_multiset_equality                                                                  |         553 (  1.9%) |         503 (  1.8%) |          34 (  0.6%) |          44 (  0.8%) |           5 (  0.0%) |
| ··tasmlib_list_multiset_equality_continue                                                       |         532 (  1.8%) |         490 (  1.8%) |          32 (  0.6%) |          44 (  0.8%) |           5 (  0.0%) |
| ····tasmlib_hashing_algebraic_hasher_hash_varlen                                                |         352 (  1.2%) |         306 (  1.1%) |          20 (  0.4%) |          38 (  0.7%) |           5 (  0.0%) |
| ······tasmlib_hashing_absorb_multiple                                                           |         324 (  1.1%) |         276 (  1.0%) |          20 (  0.4%) |          24 (  0.4%) |           5 (  0.0%) |
| ········tasmlib_hashing_absorb_multiple_hash_all_full_chunks                                    |          34 (  0.1%) |          64 (  0.2%) |          20 (  0.4%) |          12 (  0.2%) |           0 (  0.0%) |
| ········tasmlib_hashing_absorb_multiple_pad_varnum_zeros                                        |         208 (  0.7%) |         134 (  0.5%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_hashing_absorb_multiple_read_remainder                                          |          10 (  0.0%) |           8 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ····tasmlib_list_multiset_equality_running_product                                              |         142 (  0.5%) |         138 (  0.5%) |          12 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |
| ······tasmlib_list_multiset_equality_running_product_loop                                       |         114 (  0.4%) |         120 (  0.4%) |          12 (  0.2%) |           0 (  0.0%) |           0 (  0.0%) |
| tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_compute_commitment                   |         278 (  1.0%) |         230 (  0.8%) |          57 (  1.0%) |          36 (  0.6%) |           0 (  0.0%) |
| ··tasmlib_list_new___tuple_Lvoid_pointer___digestR                                              |          23 (  0.1%) |          19 (  0.1%) |           3 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_higher_order_u32_map_tasm_neptune_transaction_compute_commitment_loop            |         227 (  0.8%) |         188 (  0.7%) |          52 (  0.9%) |          36 (  0.6%) |           0 (  0.0%) |
| ····tasm_neptune_transaction_compute_commitment                                                 |         186 (  0.6%) |         140 (  0.5%) |          28 (  0.5%) |          36 (  0.6%) |           0 (  0.0%) |
| ······tasmlib_neptune_mutator_set_commit                                                        |           6 (  0.0%) |          20 (  0.1%) |           0 (  0.0%) |          24 (  0.4%) |           0 (  0.0%) |
| tasmlib_list_higher_order_u32_all_tasm_neptune_transaction_verify_aocl_membership               |        9761 ( 33.9%) |        6002 ( 22.0%) |         669 ( 12.0%) |         756 ( 13.6%) |        4131 ( 30.4%) |
| ··tasmlib_list_length___tuple_Lvoid_pointer___digestR                                           |           3 (  0.0%) |           2 (  0.0%) |           1 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··tasmlib_list_higher_order_u32_all_tasm_neptune_transaction_verify_aocl_membership_loop        |        9751 ( 33.8%) |        5996 ( 22.0%) |         668 ( 12.0%) |         756 ( 13.6%) |        4131 ( 30.4%) |
| ····tasmlib_list_get_element___tuple_Lvoid_pointer___digestR                                    |          28 (  0.1%) |          32 (  0.1%) |          12 (  0.2%) |           0 (  0.0%) |           9 (  0.1%) |
| ····tasm_neptune_transaction_verify_aocl_membership                                             |        9688 ( 33.6%) |        5938 ( 21.8%) |         656 ( 11.8%) |         756 ( 13.6%) |        4122 ( 30.3%) |
| ······tasmlib_mmr_verify_from_memory                                                            |        9542 ( 33.1%) |        5790 ( 21.3%) |         640 ( 11.5%) |         756 ( 13.6%) |        4122 ( 30.3%) |
| ········tasmlib_mmr_leaf_index_to_mt_index_and_peak_index                                       |         248 (  0.9%) |         154 (  0.6%) |           0 (  0.0%) |           0 (  0.0%) |         727 (  5.3%) |
| ··········tasmlib_arithmetic_u64_lt                                                             |          12 (  0.0%) |          10 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          66 (  0.5%) |
| ··········tasmlib_arithmetic_u64_xor                                                            |          10 (  0.0%) |           4 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |         131 (  1.0%) |
| ··········tasmlib_arithmetic_u64_log_2_floor                                                    |          30 (  0.1%) |          18 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |          66 (  0.5%) |
| ············tasmlib_arithmetic_u64_log_2_floor_then                                             |          16 (  0.1%) |          10 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          66 (  0.5%) |
| ··········tasmlib_arithmetic_u64_pow2                                                           |           8 (  0.0%) |           6 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          40 (  0.3%) |
| ··········tasmlib_arithmetic_u64_decr                                                           |          38 (  0.1%) |          32 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ············tasmlib_arithmetic_u64_decr_carry                                                   |          22 (  0.1%) |          20 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··········tasmlib_arithmetic_u64_and                                                            |          20 (  0.1%) |           8 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |         196 (  1.4%) |
| ··········tasmlib_arithmetic_u64_add                                                            |          28 (  0.1%) |          16 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |         131 (  1.0%) |
| ··········tasmlib_arithmetic_u64_popcount                                                       |          20 (  0.1%) |           4 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |          97 (  0.7%) |
| ········tasmlib_mmr_verify_from_memory_while                                                    |        9182 ( 31.9%) |        5560 ( 20.4%) |         630 ( 11.3%) |         756 ( 13.6%) |        3395 ( 25.0%) |
| ··········tasmlib_arithmetic_u64_eq                                                             |         768 (  2.7%) |         384 (  1.4%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ··········tasmlib_list_get_element___digest                                                     |        1638 (  5.7%) |        1890 (  6.9%) |         630 ( 11.3%) |           0 (  0.0%) |         228 (  1.7%) |
| ··········tasmlib_arithmetic_u32_isodd                                                          |         756 (  2.6%) |         252 (  0.9%) |           0 (  0.0%) |           0 (  0.0%) |        2048 ( 15.1%) |
| ··········tasmlib_arithmetic_u64_div2                                                           |        1764 (  6.1%) |         756 (  2.8%) |           0 (  0.0%) |           0 (  0.0%) |        1119 (  8.2%) |
| ··········tasmlib_hashing_swap_digest                                                           |         784 (  2.7%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| ········tasmlib_hashing_eq_digest                                                               |          30 (  0.1%) |          18 (  0.1%) |           0 (  0.0%) |           0 (  0.0%) |           0 (  0.0%) |
| Total                                                                                           |       28824 (100.0%) |       27231 (100.0%) |        5570 (100.0%) |        5579 (100.0%) |       13590 (100.0%) |
