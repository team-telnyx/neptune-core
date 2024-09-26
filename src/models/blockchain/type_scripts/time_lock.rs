use std::collections::HashMap;

use get_size::GetSize;
use itertools::Itertools;
use num_traits::CheckedSub;
use num_traits::Zero;
use proptest::arbitrary::Arbitrary;
use proptest::collection::vec;
use proptest::strategy::BoxedStrategy;
use proptest::strategy::Strategy;
use proptest_arbitrary_interop::arb;
use serde::Deserialize;
use serde::Serialize;
use tasm_lib::memory::encode_to_memory;
use tasm_lib::memory::FIRST_NON_DETERMINISTICALLY_INITIALIZED_MEMORY_ADDRESS;
use tasm_lib::triton_vm::prelude::*;
use tasm_lib::twenty_first::math::b_field_element::BFieldElement;
use tasm_lib::twenty_first::math::bfield_codec::BFieldCodec;
use tasm_lib::twenty_first::math::tip5::Tip5;
use tasm_lib::twenty_first::prelude::AlgebraicHasher;
use tasm_lib::Digest;

use super::neptune_coins::NeptuneCoins;
use super::TypeScriptWitness;
use crate::models::blockchain::transaction::primitive_witness::PrimitiveWitness;
use crate::models::blockchain::transaction::primitive_witness::SaltedUtxos;
use crate::models::blockchain::transaction::transaction_kernel::TransactionKernel;
use crate::models::blockchain::transaction::transaction_kernel::TransactionKernelField;
use crate::models::blockchain::transaction::utxo::Coin;
use crate::models::blockchain::transaction::PublicAnnouncement;
use crate::models::blockchain::type_scripts::TypeScriptAndWitness;
use crate::models::proof_abstractions::mast_hash::MastHash;
use crate::models::proof_abstractions::tasm::builtins as tasm;
use crate::models::proof_abstractions::tasm::program::ConsensusProgram;
use crate::models::proof_abstractions::timestamp::Timestamp;
use crate::models::proof_abstractions::SecretWitness;
use crate::Hash;

#[derive(Debug, Clone, Deserialize, Serialize, BFieldCodec, GetSize, PartialEq, Eq)]
pub struct TimeLock;

impl TimeLock {
    /// Create a `TimeLock` type-script-and-state-pair that releases the coins at the
    /// given release date, which corresponds to the number of milliseconds that passed
    /// since the unix epoch started (00:00 am UTC on Jan 1 1970).
    pub fn until(date: Timestamp) -> Coin {
        Coin {
            type_script_hash: TimeLock.hash(),
            state: vec![date.0],
        }
    }
}

impl ConsensusProgram for TimeLock {
    #[allow(clippy::needless_return)]
    fn source(&self) {
        // get in the current program's hash digest
        let self_digest: Digest = tasm::own_program_digest();

        // read standard input:
        //  - transaction kernel mast hash
        //  - input salted utxos digest
        //  - output salted utxos digest
        // (All type scripts take this triple as input.)
        let tx_kernel_digest: Digest = tasm::tasmlib_io_read_stdin___digest();
        let input_utxos_digest: Digest = tasm::tasmlib_io_read_stdin___digest();
        let _output_utxos_digest: Digest = tasm::tasmlib_io_read_stdin___digest();

        // divine the timestamp and authenticate it against the kernel mast hash
        let leaf_index: u32 = 5;
        let timestamp: BFieldElement = tasm::tasmlib_io_read_secin___bfe();
        let leaf: Digest = Hash::hash_varlen(&timestamp.encode());
        let tree_height: u32 = 3;
        tasm::tasmlib_hashing_merkle_verify(tx_kernel_digest, leaf_index, leaf, tree_height);

        // get pointers to objects living in nondeterministic memory:
        //  - input Salted UTXOs
        let input_utxos_pointer: u64 = tasm::tasmlib_io_read_secin___bfe().value();

        // it's important to read the outputs digest too, but we actually don't care about
        // the output UTXOs (in this type script)
        let _output_utxos_pointer: u64 = tasm::tasmlib_io_read_secin___bfe().value();

        // authenticate salted input UTXOs against the digest that was read from stdin
        let input_salted_utxos: SaltedUtxos =
            tasm::decode_from_memory(BFieldElement::new(input_utxos_pointer));
        let input_salted_utxos_digest: Digest = Tip5::hash(&input_salted_utxos);
        assert_eq!(input_salted_utxos_digest, input_utxos_digest);

        // iterate over inputs
        let input_utxos = input_salted_utxos.utxos;
        let mut i = 0;
        while i < input_utxos.len() {
            // get coins
            let coins: &Vec<Coin> = &input_utxos[i].coins;

            // if this typescript is present
            let mut j: usize = 0;
            while j < coins.len() {
                let coin: &Coin = &coins[j];
                if coin.type_script_hash == self_digest {
                    // extract state
                    let state: &Vec<BFieldElement> = &coin.state;

                    // assert format
                    assert!(state.len() == 1);

                    // extract timestamp
                    let release_date: BFieldElement = state[0];

                    // test time lock
                    assert!(release_date.value() < timestamp.value());
                }
                j += 1;
            }
            i += 1;
        }

        return;
    }

    fn code(&self) -> Vec<LabelledInstruction> {
        // Generated by tasm-lang compiler
        // `tasm-lang typescript-timelock.rs type-script-time-lock.tasm`
        // 2024-09-09
        triton_asm! {
        call main
        halt
        main:
        call tasmlib_verifier_own_program_digest
        dup 4
        dup 4
        dup 4
        dup 4
        dup 4
        push -6
        write_mem 5
        pop 1
        hint self_digest = stack[0..5]
        call tasmlib_io_read_stdin___digest
        dup 4
        dup 4
        dup 4
        dup 4
        dup 4
        push -11
        write_mem 5
        pop 1
        hint tx_kernel_digest = stack[0..5]
        call tasmlib_io_read_stdin___digest
        dup 4
        dup 4
        dup 4
        dup 4
        dup 4
        push -16
        write_mem 5
        pop 1
        hint input_utxos_digest = stack[0..5]
        call tasmlib_io_read_stdin___digest
        hint _output_utxos_digest = stack[0..5]
        push 5
        hint leaf_index = stack[0]
        call tasmlib_io_read_secin___bfe
        dup 0
        push -17
        write_mem 1
        pop 1
        hint timestamp = stack[0]
        push -17
        read_mem 1
        pop 1
        call encode_BField
        call tasm_langs_hash_varlen
        hint leaf = stack[0..5]
        push 3
        hint tree_height = stack[0]
        push -7
        read_mem 5
        pop 1
        dup 5
        dup 13
        dup 12
        dup 12
        dup 12
        dup 12
        dup 12
        call tasmlib_hashing_merkle_verify
        call tasmlib_io_read_secin___bfe
        hint input_utxos_pointer = stack[0]
        call tasmlib_io_read_secin___bfe
        split
        hint _output_utxos_pointer = stack[0..2]
        dup 2
        hint input_salted_utxos = stack[0]
        dup 0
        call tasm_langs_hash_varlen_boxed_value___SaltedUtxos
        hint input_salted_utxos_digest = stack[0..5]
        dup 4
        dup 4
        dup 4
        dup 4
        dup 4
        push -12
        read_mem 5
        pop 1
        call tasmlib_hashing_eq_digest
        assert
        dup 5
        addi 4
        hint input_utxos = stack[0]
        push 0
        hint i = stack[0]
        call _binop_Lt__LboolR_bool_32_while_loop
        pop 5
        pop 5
        pop 5
        pop 5
        pop 5
        pop 5
        pop 5
        pop 4
        return
        _binop_Eq__LboolR_bool_49_then:
        pop 1
        dup 0
        addi 1
        hint state = stack[0]
        dup 0
        read_mem 1
        pop 1
        push 1
        eq
        assert
        dup 0
        push 0
        push 1
        mul
        push 1
        add
        push 00000000001073741824
        dup 1
        lt
        assert
        add
        read_mem 1
        pop 1
        hint release_date = stack[0]
        dup 0
        split
        push -17
        read_mem 1
        pop 1
        split
        swap 3
        swap 1
        swap 3
        swap 2
        call tasmlib_arithmetic_u64_lt
        assert
        pop 1
        pop 1
        push 0
        return
        _binop_Eq__LboolR_bool_49_else:
        return
        _binop_Lt__LboolR_bool_42_while_loop:
        dup 0
        dup 2
        read_mem 1
        pop 1
        swap 1
        lt
        push 0
        eq
        skiz
        return
        dup 1
        push 1
        add
        dup 1
        call tasm_langs_dynamic_list_element_finder
        pop 1
        addi 1
        hint coin = stack[0]
        dup 0
        read_mem 1
        push 00000000001073741824
        dup 2
        lt
        assert
        addi 2
        add
        push 4
        add
        read_mem 5
        pop 1
        push -2
        read_mem 5
        pop 1
        call tasmlib_hashing_eq_digest
        push 1
        swap 1
        skiz
        call _binop_Eq__LboolR_bool_49_then
        skiz
        call _binop_Eq__LboolR_bool_49_else
        dup 1
        push 1
        call tasmlib_arithmetic_u32_safeadd
        swap 2
        pop 1
        pop 1
        recurse
        _binop_Lt__LboolR_bool_32_while_loop:
        dup 0
        dup 2
        read_mem 1
        pop 1
        swap 1
        lt
        push 0
        eq
        skiz
        return
        dup 1
        push 1
        add
        dup 1
        call tasm_langs_dynamic_list_element_finder
        pop 1
        addi 1
        addi 1
        hint coins = stack[0]
        push 0
        hint j = stack[0]
        call _binop_Lt__LboolR_bool_42_while_loop
        dup 2
        push 1
        call tasmlib_arithmetic_u32_safeadd
        swap 3
        pop 1
        pop 1
        pop 1
        recurse
        encode_BField:
        call tasmlib_memory_dyn_malloc
        push 1
        swap 1
        write_mem 1
        write_mem 1
        push -2
        add
        return
        tasm_langs_dynamic_list_element_finder:
        dup 0
        push 0
        eq
        skiz
        return
        swap 1
        read_mem 1
        push 00000000001073741824
        dup 2
        lt
        assert
        addi 2
        add
        swap 1
        addi -1
        recurse
        tasm_langs_hash_varlen:
        read_mem 1
        push 2
        add
        swap 1
        call tasmlib_hashing_algebraic_hasher_hash_varlen
        return
        tasm_langs_hash_varlen_boxed_value___SaltedUtxos:
        dup 0
        push 0
        addi 3
        swap 1
        addi 3
        swap 1
        dup 1
        read_mem 1
        pop 1
        push 00000000001073741824
        dup 1
        lt
        assert
        addi 1
        dup 2
        dup 1
        add
        swap 3
        pop 1
        add
        swap 1
        pop 1
        call tasmlib_hashing_algebraic_hasher_hash_varlen
        return
        tasmlib_arithmetic_u32_safeadd:
        hint input_lhs: u32 = stack[0]
        hint input_rhs: u32 = stack[1]
        add
        dup 0
        split
        pop 1
        push 0
        eq
        assert
        return
        tasmlib_arithmetic_u64_lt:
        hint lhs: u64 = stack[0..2]
        hint rhs: u64 = stack[2..4]
        swap 3
        swap 2
        dup 2
        dup 2
        lt
        swap 4
        lt
        swap 2
        eq
        mul
        add
        return
        tasmlib_hashing_absorb_multiple:
        hint len: u32 = stack[0]
        hint _sequence: void_pointer = stack[1]
        dup 0
        push 10
        swap 1
        div_mod
        swap 1
        pop 1
        swap 1
        dup 1
        push -1
        mul
        dup 3
        add
        add
        swap 1
        swap 2
        push 0
        push 0
        push 0
        push 0
        swap 4
        call tasmlib_hashing_absorb_multiple_hash_all_full_chunks
        pop 5
        push -1
        add
        push 9
        dup 2
        push -1
        mul
        add
        call tasmlib_hashing_absorb_multiple_pad_varnum_zeros
        pop 1
        push 1
        swap 2
        dup 1
        add
        call tasmlib_hashing_absorb_multiple_read_remainder
        pop 2
        sponge_absorb
        return
        tasmlib_hashing_absorb_multiple_hash_all_full_chunks:
        dup 5
        dup 1
        eq
        skiz
        return
        sponge_absorb_mem
        recurse
        tasmlib_hashing_absorb_multiple_pad_varnum_zeros:
        dup 0
        push 0
        eq
        skiz
        return
        push 0
        swap 3
        swap 2
        swap 1
        push -1
        add
        recurse
        tasmlib_hashing_absorb_multiple_read_remainder:
        dup 1
        dup 1
        eq
        skiz
        return
        read_mem 1
        swap 1
        swap 2
        swap 1
        recurse
        tasmlib_hashing_algebraic_hasher_hash_varlen:
        hint length: u32 = stack[0]
        hint _addr: void_pointer = stack[1]
        sponge_init
        call tasmlib_hashing_absorb_multiple
        sponge_squeeze
        swap 5
        pop 1
        swap 5
        pop 1
        swap 5
        pop 1
        swap 5
        pop 1
        swap 5
        pop 1
        return
        tasmlib_hashing_eq_digest:
        hint input_a4: digest = stack[0..5]
        hint input_b4: digest = stack[5..10]
        swap 6
        eq
        swap 6
        eq
        swap 6
        eq
        swap 6
        eq
        swap 2
        eq
        mul
        mul
        mul
        mul
        return
        tasmlib_hashing_merkle_verify:
        hint leaf: digest = stack[0..5]
        hint leaf_index: u32 = stack[5]
        hint tree_height: u32 = stack[6]
        hint root: digest = stack[7..12]
        dup 6
        push 2
        pow
        dup 0
        dup 7
        lt
        assert
        dup 6
        add
        swap 6
        pop 1
        dup 6
        skiz
        call tasmlib_hashing_merkle_verify_tree_height_is_not_zero
        swap 2
        swap 4
        swap 6
        pop 1
        swap 2
        swap 4
        pop 1
        assert_vector
        pop 5
        return
        tasmlib_hashing_merkle_verify_tree_height_is_not_zero:
        push 1
        swap 7
        pop 1
        call tasmlib_hashing_merkle_verify_traverse_tree
        return
        tasmlib_hashing_merkle_verify_traverse_tree:
        merkle_step
        recurse_or_return
        tasmlib_io_read_secin___bfe:
        divine 1
        return
        tasmlib_io_read_stdin___digest:
        read_io 5
        return
        tasmlib_memory_dyn_malloc:
        push -1
        read_mem 1
        pop 1
        dup 0
        push 0
        eq
        skiz
        call tasmlib_memory_dyn_malloc_initialize
        push 00000000002147483647
        dup 1
        lt
        assert
        dup 0
        push 1
        add
        push -1
        write_mem 1
        pop 1
        push 00000000004294967296
        mul
        return
        tasmlib_memory_dyn_malloc_initialize:
        pop 1
        push 1
        return
        tasmlib_verifier_own_program_digest:
        dup 15
        dup 15
        dup 15
        dup 15
        dup 15
        return
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, BFieldCodec, GetSize, PartialEq, Eq)]
pub struct TimeLockWitness {
    /// One timestamp for every input UTXO. Inputs that do not have a time lock are
    /// assigned timestamp 0, which is automatically satisfied.
    release_dates: Vec<Timestamp>,
    input_utxos: SaltedUtxos,
    transaction_kernel: TransactionKernel,
}

impl SecretWitness for TimeLockWitness {
    fn nondeterminism(&self) -> NonDeterminism {
        let mut memory: HashMap<BFieldElement, BFieldElement> = HashMap::new();
        let input_salted_utxos_address = FIRST_NON_DETERMINISTICALLY_INITIALIZED_MEMORY_ADDRESS;
        let output_salted_utxos_address =
            encode_to_memory(&mut memory, input_salted_utxos_address, &self.input_utxos);
        encode_to_memory(
            &mut memory,
            output_salted_utxos_address,
            &SaltedUtxos::empty(),
        );
        let individual_tokens = vec![
            self.transaction_kernel.timestamp.0,
            input_salted_utxos_address,
            output_salted_utxos_address,
        ];
        let mast_path = self
            .transaction_kernel
            .mast_path(TransactionKernelField::Timestamp)
            .clone();
        NonDeterminism::new(individual_tokens)
            .with_digests(mast_path)
            .with_ram(memory)
    }

    fn standard_input(&self) -> PublicInput {
        self.type_script_standard_input()
    }

    fn program(&self) -> Program {
        TimeLock.program()
    }
}

impl TypeScriptWitness for TimeLockWitness {
    fn transaction_kernel(&self) -> TransactionKernel {
        self.transaction_kernel.clone()
    }

    fn salted_input_utxos(&self) -> SaltedUtxos {
        self.input_utxos.clone()
    }

    fn salted_output_utxos(&self) -> SaltedUtxos {
        SaltedUtxos::empty()
    }
    fn type_script_and_witness(&self) -> TypeScriptAndWitness {
        TypeScriptAndWitness::new_with_nondeterminism(TimeLock.program(), self.nondeterminism())
    }
}

impl From<PrimitiveWitness> for TimeLockWitness {
    fn from(primitive_witness: PrimitiveWitness) -> Self {
        let release_dates = primitive_witness
            .input_utxos
            .utxos
            .iter()
            .map(|utxo| {
                utxo.coins
                    .iter()
                    .find(|coin| coin.type_script_hash == TimeLock.hash())
                    .cloned()
                    .map(|coin| {
                        coin.state
                            .first()
                            .copied()
                            .unwrap_or_else(|| BFieldElement::new(0))
                    })
                    .unwrap_or_else(|| BFieldElement::new(0))
            })
            .map(Timestamp)
            .collect_vec();
        let transaction_kernel = TransactionKernel::from(primitive_witness.clone());
        let input_utxos = primitive_witness.input_utxos.clone();
        Self {
            release_dates,
            input_utxos,
            transaction_kernel,
        }
    }
}

impl Arbitrary for TimeLockWitness {
    /// Parameters are:
    ///  - release_dates : Vec<u64> One release date per input UTXO. 0 if the time lock
    ///    coin is absent.
    ///  - num_outputs : usize Number of outputs.
    ///  - num_public_announcements : usize Number of public announcements.
    ///  - transaction_timestamp: Timestamp determining when the transaction takes place.
    type Parameters = (Vec<Timestamp>, usize, usize, Timestamp);

    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(parameters: Self::Parameters) -> Self::Strategy {
        let (release_dates, num_outputs, num_public_announcements, transaction_timestamp) =
            parameters;
        let num_inputs = release_dates.len();
        (
            vec(arb::<Digest>(), num_inputs),
            vec(arb::<NeptuneCoins>(), num_inputs),
            vec(arb::<Digest>(), num_outputs),
            vec(arb::<NeptuneCoins>(), num_outputs),
            vec(arb::<PublicAnnouncement>(), num_public_announcements),
            arb::<Option<NeptuneCoins>>(),
            arb::<NeptuneCoins>(),
        )
            .prop_flat_map(
                move |(
                    input_address_seeds,
                    input_amounts,
                    output_address_seeds,
                    mut output_amounts,
                    public_announcements,
                    maybe_coinbase,
                    mut fee,
                )| {
                    // generate inputs
                    let (mut input_utxos, input_lock_scripts_and_witnesses) =
                        PrimitiveWitness::transaction_inputs_from_address_seeds_and_amounts(
                            &input_address_seeds,
                            &input_amounts,
                        );
                    let total_inputs = input_amounts.into_iter().sum::<NeptuneCoins>();

                    // add time locks to input UTXOs
                    for (utxo, release_date) in input_utxos.iter_mut().zip(release_dates.iter()) {
                        if !release_date.is_zero() {
                            let time_lock_coin = TimeLock::until(*release_date);
                            utxo.coins.push(time_lock_coin);
                        }
                    }

                    // generate valid output amounts
                    PrimitiveWitness::find_balanced_output_amounts_and_fee(
                        total_inputs,
                        maybe_coinbase,
                        &mut output_amounts,
                        &mut fee,
                    );

                    // generate output UTXOs
                    let output_utxos =
                        PrimitiveWitness::valid_transaction_outputs_from_amounts_and_address_seeds(
                            &output_amounts,
                            &output_address_seeds,
                        );

                    // generate primitive transaction witness and time lock witness from there
                    PrimitiveWitness::arbitrary_primitive_witness_with(
                        &input_utxos,
                        &input_lock_scripts_and_witnesses,
                        &output_utxos,
                        &public_announcements,
                        NeptuneCoins::zero(),
                        maybe_coinbase,
                    )
                    .prop_map(move |mut transaction_primitive_witness| {
                        transaction_primitive_witness.kernel.timestamp = transaction_timestamp;
                        TimeLockWitness::from(transaction_primitive_witness)
                    })
                    .boxed()
                },
            )
            .boxed()
    }
}

/// Generate a `Strategy` for a [`PrimitiveWitness`] with the given numbers of
/// inputs, outputs, and public announcements. The UTXOs are timelocked with a
/// release date set between `now` and six months from `now`.
pub fn arbitrary_primitive_witness_with_active_timelocks(
    num_inputs: usize,
    num_outputs: usize,
    num_announcements: usize,
    now: Timestamp,
) -> BoxedStrategy<PrimitiveWitness> {
    vec(
        Timestamp::arbitrary_between(now, now + Timestamp::months(6)),
        num_inputs + num_outputs,
    )
    .prop_flat_map(move |release_dates| {
        arbitrary_primitive_witness_with_timelocks(
            num_inputs,
            num_outputs,
            num_announcements,
            now,
            release_dates,
        )
    })
    .boxed()
}

/// Generate a `Strategy` for a [`PrimitiveWitness`] with the given numbers of
/// inputs, outputs, and public announcements. The UTXOs are timelocked with a
/// release date set between six months in the past relative to `now` and `now`.
pub fn arbitrary_primitive_witness_with_expired_timelocks(
    num_inputs: usize,
    num_outputs: usize,
    num_announcements: usize,
    now: Timestamp,
) -> BoxedStrategy<PrimitiveWitness> {
    vec(
        Timestamp::arbitrary_between(now - Timestamp::months(6), now - Timestamp::millis(1)),
        num_inputs + num_outputs,
    )
    .prop_flat_map(move |release_dates| {
        arbitrary_primitive_witness_with_timelocks(
            num_inputs,
            num_outputs,
            num_announcements,
            now,
            release_dates,
        )
    })
    .boxed()
}

#[expect(unused_variables, reason = "under development")]
fn arbitrary_primitive_witness_with_timelocks(
    num_inputs: usize,
    num_outputs: usize,
    num_announcements: usize,
    now: Timestamp,
    release_dates: Vec<Timestamp>,
) -> BoxedStrategy<PrimitiveWitness> {
    (
        arb::<NeptuneCoins>(),
        vec(arb::<Digest>(), num_inputs),
        vec(arb::<u64>(), num_inputs),
        vec(arb::<Digest>(), num_outputs),
        vec(arb::<u64>(), num_outputs),
        vec(arb::<PublicAnnouncement>(), num_announcements),
        arb::<u64>(),
        arb::<Option<u64>>(),
    )
        .prop_flat_map(
            move |(
                total_amount,
                input_address_seeds,
                input_dist,
                output_address_seeds,
                output_dist,
                public_announcements,
                fee_dist,
                maybe_coinbase_dist,
            )| {
                // distribute total amount across inputs (+ coinbase)
                let mut input_denominator = input_dist.iter().map(|u| *u as f64).sum::<f64>();
                if let Some(d) = maybe_coinbase_dist {
                    input_denominator += d as f64;
                }
                let input_weights = input_dist
                    .into_iter()
                    .map(|u| (u as f64) / input_denominator)
                    .collect_vec();
                let mut input_amounts = input_weights
                    .into_iter()
                    .map(|w| total_amount.to_nau_f64() * w)
                    .map(|f| NeptuneCoins::try_from(f).unwrap())
                    .collect_vec();
                let maybe_coinbase = if maybe_coinbase_dist.is_some() || input_amounts.is_empty() {
                    Some(
                        total_amount
                            .checked_sub(&input_amounts.iter().cloned().sum::<NeptuneCoins>())
                            .unwrap(),
                    )
                } else {
                    let sum_of_all_but_last = input_amounts
                        .iter()
                        .rev()
                        .skip(1)
                        .cloned()
                        .sum::<NeptuneCoins>();
                    *input_amounts.last_mut().unwrap() =
                        total_amount.checked_sub(&sum_of_all_but_last).unwrap();
                    None
                };

                // distribute total amount across outputs
                let output_denominator =
                    output_dist.iter().map(|u| *u as f64).sum::<f64>() + (fee_dist as f64);
                let output_weights = output_dist
                    .into_iter()
                    .map(|u| (u as f64) / output_denominator)
                    .collect_vec();
                let output_amounts = output_weights
                    .into_iter()
                    .map(|w| total_amount.to_nau_f64() * w)
                    .map(|f| NeptuneCoins::try_from(f).unwrap())
                    .collect_vec();
                let total_outputs = output_amounts.iter().cloned().sum::<NeptuneCoins>();
                let fee = total_amount.checked_sub(&total_outputs).unwrap();

                let (mut input_utxos, input_lock_scripts_and_witnesses) =
                    PrimitiveWitness::transaction_inputs_from_address_seeds_and_amounts(
                        &input_address_seeds,
                        &input_amounts,
                    );
                let total_inputs = input_amounts.iter().copied().sum::<NeptuneCoins>();

                assert_eq!(
                    total_inputs + maybe_coinbase.unwrap_or(NeptuneCoins::new(0)),
                    total_outputs + fee
                );
                let mut output_utxos =
                    PrimitiveWitness::valid_transaction_outputs_from_amounts_and_address_seeds(
                        &output_amounts,
                        &output_address_seeds,
                    );
                let mut counter = 0usize;
                for utxo in input_utxos.iter_mut() {
                    let release_date = release_dates[counter];
                    let time_lock = TimeLock::until(release_date);
                    utxo.coins.push(time_lock);
                    counter += 1;
                }
                for utxo in output_utxos.iter_mut() {
                    utxo.coins.push(TimeLock::until(release_dates[counter]));
                    counter += 1;
                }
                let release_dates = release_dates.clone();
                PrimitiveWitness::arbitrary_primitive_witness_with_timestamp_and(
                    &input_utxos,
                    &input_lock_scripts_and_witnesses,
                    &output_utxos,
                    &public_announcements,
                    fee,
                    maybe_coinbase,
                    now,
                )
                .prop_map(move |primitive_witness_template| {
                    let mut primitive_witness = primitive_witness_template.clone();
                    let time_lock_witness = TimeLockWitness::from(primitive_witness.clone());
                    primitive_witness.type_scripts_and_witnesses.push(
                        TypeScriptAndWitness::new_with_nondeterminism(
                            TimeLock.program(),
                            time_lock_witness.nondeterminism(),
                        ),
                    );
                    primitive_witness.kernel.timestamp = now;
                    primitive_witness
                })
            },
        )
        .boxed()
}

#[cfg(test)]
mod test {
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::strategy::Just;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;
    use tokio::runtime::Runtime;

    use super::TimeLockWitness;
    use crate::models::blockchain::transaction::primitive_witness::PrimitiveWitness;
    use crate::models::blockchain::type_scripts::time_lock::arbitrary_primitive_witness_with_active_timelocks;
    use crate::models::blockchain::type_scripts::time_lock::arbitrary_primitive_witness_with_expired_timelocks;
    use crate::models::blockchain::type_scripts::time_lock::TimeLock;
    use crate::models::proof_abstractions::tasm::program::ConsensusProgram;
    use crate::models::proof_abstractions::timestamp::Timestamp;
    use crate::models::proof_abstractions::SecretWitness;

    #[proptest(cases = 20)]
    fn test_unlocked(
        #[strategy(1usize..=3)] _num_inputs: usize,
        #[strategy(1usize..=3)] _num_outputs: usize,
        #[strategy(1usize..=3)] _num_public_announcements: usize,
        #[strategy(vec(Just(Timestamp::zero()), #_num_inputs))] _release_dates: Vec<Timestamp>,
        #[strategy(Just::<Timestamp>(#_release_dates.iter().copied().min().unwrap()))]
        _transaction_timestamp: Timestamp,
        #[strategy(TimeLockWitness::arbitrary_with((#_release_dates, #_num_outputs, #_num_public_announcements, #_transaction_timestamp)))]
        time_lock_witness: TimeLockWitness,
    ) {
        let rust_result = TimeLock.run_rust(
            &time_lock_witness.standard_input(),
            time_lock_witness.nondeterminism(),
        );
        prop_assert!(
            rust_result.is_ok(),
            "time lock program did not halt gracefully"
        );
        let tasm_result = TimeLock.run_tasm(
            &time_lock_witness.standard_input(),
            time_lock_witness.nondeterminism(),
        );
        prop_assert!(
            tasm_result.is_ok(),
            "time lock program did not halt gracefully"
        );
        prop_assert_eq!(rust_result.unwrap(), tasm_result.unwrap());
    }

    #[proptest(cases = 20)]
    fn test_locked(
        #[strategy(1usize..=3)] _num_inputs: usize,
        #[strategy(1usize..=3)] _num_outputs: usize,
        #[strategy(1usize..=3)] _num_public_announcements: usize,
        #[strategy(vec(Timestamp::arbitrary_between(Timestamp::now()-Timestamp::days(7),Timestamp::now()-Timestamp::days(1)), #_num_inputs))]
        _release_dates: Vec<Timestamp>,
        #[strategy(Just::<Timestamp>(#_release_dates.iter().copied().max().unwrap()))]
        _tx_timestamp: Timestamp,
        #[strategy(TimeLockWitness::arbitrary_with((#_release_dates, #_num_outputs, #_num_public_announcements, #_tx_timestamp)))]
        time_lock_witness: TimeLockWitness,
    ) {
        println!("now: {}", Timestamp::now());
        prop_assert!(
            TimeLock {}
                .run_rust(
                    &time_lock_witness.standard_input(),
                    time_lock_witness.nondeterminism(),
                )
                .is_err(),
            "time lock program failed to panic"
        );
        prop_assert!(
            TimeLock {}
                .run_tasm(
                    &time_lock_witness.standard_input(),
                    time_lock_witness.nondeterminism(),
                )
                .is_err(),
            "time lock program failed to panic"
        );
    }

    #[proptest(cases = 20)]
    fn test_released(
        #[strategy(1usize..=3)] _num_inputs: usize,
        #[strategy(1usize..=3)] _num_outputs: usize,
        #[strategy(1usize..=3)] _num_public_announcements: usize,
        #[strategy(vec(Timestamp::arbitrary_between(Timestamp::now()-Timestamp::days(7),Timestamp::now()-Timestamp::days(1)), #_num_inputs))]
        _release_dates: Vec<Timestamp>,
        #[strategy(Just::<Timestamp>(#_release_dates.iter().copied().max().unwrap()))]
        _tx_timestamp: Timestamp,
        #[strategy(TimeLockWitness::arbitrary_with((#_release_dates, #_num_outputs, #_num_public_announcements, #_tx_timestamp + Timestamp::days(1))))]
        time_lock_witness: TimeLockWitness,
    ) {
        println!("now: {}", Timestamp::now());
        let rust_result = TimeLock.run_rust(
            &time_lock_witness.standard_input(),
            time_lock_witness.nondeterminism(),
        );
        prop_assert!(
            rust_result.is_ok(),
            "time lock program did not halt gracefully"
        );
        let tasm_result = TimeLock.run_tasm(
            &time_lock_witness.standard_input(),
            time_lock_witness.nondeterminism(),
        );
        prop_assert!(
            tasm_result.is_ok(),
            "time lock program did not halt gracefully"
        );
        prop_assert_eq!(rust_result.unwrap(), tasm_result.unwrap());
    }

    #[proptest(cases = 5)]
    fn primitive_witness_with_timelocks_is_valid(
        #[strategy(arb::<Timestamp>())] _now: Timestamp,
        #[strategy(arbitrary_primitive_witness_with_active_timelocks(2, 2, 2, #_now))]
        primitive_witness: PrimitiveWitness,
    ) {
        prop_assert!(Runtime::new()
            .unwrap()
            .block_on(primitive_witness.validate()));
    }

    #[proptest(cases = 10)]
    fn arbitrary_primitive_witness_with_active_timelocks_fails(
        #[strategy(arb::<Timestamp>())] _now: Timestamp,
        #[strategy(arbitrary_primitive_witness_with_active_timelocks(2, 2, 2, #_now))]
        primitive_witness: PrimitiveWitness,
    ) {
        let time_lock_witness = TimeLockWitness::from(primitive_witness);

        prop_assert!(
            TimeLock {}
                .run_rust(
                    &time_lock_witness.standard_input(),
                    time_lock_witness.nondeterminism(),
                )
                .is_err(),
            "time lock program failed to panic"
        );
        prop_assert!(
            TimeLock {}
                .run_tasm(
                    &time_lock_witness.standard_input(),
                    time_lock_witness.nondeterminism(),
                )
                .is_err(),
            "time lock program failed to panic"
        );
    }

    #[proptest(cases = 10)]
    fn arbitrary_primitive_witness_with_expired_timelocks_passes(
        #[strategy(arb::<Timestamp>())] _now: Timestamp,
        #[strategy(arbitrary_primitive_witness_with_expired_timelocks(2, 2, 2, #_now))]
        primitive_witness: PrimitiveWitness,
    ) {
        let time_lock_witness = TimeLockWitness::from(primitive_witness);

        let rust_result = TimeLock.run_rust(
            &time_lock_witness.standard_input(),
            time_lock_witness.nondeterminism(),
        );
        prop_assert!(
            rust_result.is_ok(),
            "time lock program did not halt gracefully"
        );
        let tasm_result = TimeLock.run_tasm(
            &time_lock_witness.standard_input(),
            time_lock_witness.nondeterminism(),
        );
        prop_assert!(
            tasm_result.is_ok(),
            "time lock program did not halt gracefully"
        );
        prop_assert_eq!(tasm_result.unwrap(), rust_result.unwrap());
    }
}
