use num_bigint::BigUint;
use tasm_lib::{
    triton_vm::prelude::{BFieldElement, Digest},
    twenty_first::prelude::U32s,
};

use crate::models::{
    blockchain::block::block_header::{
        DIFFICULTY_NUM_LIMBS, MINIMUM_DIFFICULTY, TARGET_BLOCK_INTERVAL,
    },
    proof_abstractions::timestamp::Timestamp,
};

use super::block_height::BlockHeight;

/// Convert a difficulty to a target threshold so as to test a block's
/// proof-of-work.
pub(crate) fn target(difficulty: U32s<DIFFICULTY_NUM_LIMBS>) -> Digest {
    let difficulty_as_bui: BigUint = difficulty.into();
    let max_threshold_as_bui: BigUint =
        Digest([BFieldElement::new(BFieldElement::MAX); Digest::LEN]).into();
    let threshold_as_bui: BigUint = max_threshold_as_bui / difficulty_as_bui;

    threshold_as_bui.try_into().unwrap()
}

/// Control system for block difficulty.
///
/// This function computes the new block's difficulty from the block's
/// timestamp, the previous block's difficulty, and the previous block's
/// timestamp. It regulates the block interval by tuning the difficulty.
/// It assumes that the block timestamp is valid.
///
/// This mechanism is a PID controller with P = -2^-4 (and I = D = 0) and with
/// the relative error being clamped within [-1;4]. The following diagram
/// describes the mechanism.
///
/// ```notest
///                          --------------
///                         |              |--- new timestamp ------
///  --- new difficulty --->|  blockchain  |--- old timestamp ----  |
/// |   (control signal)    |              |--- old difficulty -  | |
/// |                        --------------                     | | |
/// |   ---                                                     | | |
///  --| * |<---------------------------------------------------  | |
///     ---                                                     - v v
///      ^                                                        ---
///      |                                                       | + |
///     ---                                                       ---
///    | + |<--- 1.0                                    (process   |
///     ---                              (setpoint:)    variable:) |
///      ^                                 target         observed |
///      |                                  block       block time |
///      |                                interval                 v
///      |                                   |                 -  ---
///      |                                   |------------------>| + |
///      |                                   |                    ---
///      |                                   |                     |
///      |                                   v                     |
///      |                                 -----                   |
///      |                                | 1/x |                  |
///      |      _                          -----                   |
///      |     / |                           v                     |
///      |    /  |    ---------------       ---     absolute error |
///       ---(P* |<--| clamp [-1; 4] |<----| * |<------------------
///           \  |    ---------------  rel. ---
///            \_|                    error
///``
/// The P-controller (without clamping) does have a systematic error up to -5%
/// of the setpoint, whose exact magnitude depends on the relation between
/// proving and guessing time. This bias could be eliminated in principle by
/// setting I and D; but the resulting controller is more complex (=> difficult
/// to implement), generates overshoot (=> bad for liveness), and periodicity
/// (=> attack vector). Most importantly, the bias is counteracted to some
/// degree by the clamping.
/// ```
pub(crate) fn difficulty_control(
    new_timestamp: Timestamp,
    old_timestamp: Timestamp,
    old_difficulty: U32s<DIFFICULTY_NUM_LIMBS>,
    target_block_interval: Option<Timestamp>,
    previous_block_height: BlockHeight,
) -> U32s<DIFFICULTY_NUM_LIMBS> {
    // no adjustment if the previous block is the genesis block
    if previous_block_height.is_genesis() {
        return old_difficulty;
    }

    // otherwise, compute PID control signal

    // target; signal to follow
    let target_block_interval = target_block_interval.unwrap_or(TARGET_BLOCK_INTERVAL);

    // most recent observed block time
    let delta_t = new_timestamp - old_timestamp;

    // distance to target
    let absolute_error = (delta_t.0.value() as i64) - (target_block_interval.0.value() as i64);
    let relative_error = absolute_error * ((1i64 << 32) / (target_block_interval.0.value() as i64));
    let clamped_error = relative_error.clamp(-1 << 32, 4 << 32);

    // change to control signal
    // adjustment_factor = (1 + P * error)
    // const P: f64 = -1.0 / 16.0;
    let one_plus_p_times_error = (1i64 << 32) + ((-clamped_error) >> 4);
    let lo = one_plus_p_times_error as u32;
    let hi = (one_plus_p_times_error >> 32) as u32;

    let mut new_difficulty = [0u32; DIFFICULTY_NUM_LIMBS + 1];
    let mut carry = 0u32;
    for (old_difficulty_i, new_difficulty_i) in old_difficulty
        .as_ref()
        .iter()
        .zip(new_difficulty.iter_mut().take(DIFFICULTY_NUM_LIMBS))
    {
        let sum = (carry as u64) + (*old_difficulty_i as u64) * (lo as u64);
        *new_difficulty_i = sum as u32;
        carry = (sum >> 32) as u32;
    }
    new_difficulty[DIFFICULTY_NUM_LIMBS] = carry;
    carry = 0u32;
    for (old_difficulty_i, new_difficulty_i_plus_one) in old_difficulty
        .as_ref()
        .iter()
        .zip(new_difficulty.iter_mut().skip(1))
    {
        let sum = (carry as u64) + (*old_difficulty_i as u64) * (hi as u64);
        let (digit, carry_bit) = new_difficulty_i_plus_one.overflowing_add(sum as u32);
        *new_difficulty_i_plus_one = digit;
        carry = ((sum >> 32) as u32) + (carry_bit as u32);
    }
    let new_difficulty =
        U32s::<DIFFICULTY_NUM_LIMBS>::new(new_difficulty[1..].to_owned().try_into().unwrap());

    if new_difficulty < MINIMUM_DIFFICULTY.into() {
        MINIMUM_DIFFICULTY.into()
    } else {
        new_difficulty
    }
}

#[cfg(test)]
mod test {
    use num_bigint::{BigInt, BigUint};
    use num_rational::BigRational;
    use num_traits::ToPrimitive;
    use rand::{rngs::StdRng, thread_rng, SeedableRng};
    use rand_distr::{Distribution, Geometric};
    use tasm_lib::twenty_first::prelude::U32s;

    use crate::models::{
        blockchain::block::{
            block_header::{DIFFICULTY_NUM_LIMBS, MINIMUM_DIFFICULTY},
            block_height::BlockHeight,
        },
        proof_abstractions::timestamp::Timestamp,
    };

    use super::difficulty_control;

    fn sample_block_time(
        hash_rate: f64,
        difficulty: U32s<DIFFICULTY_NUM_LIMBS>,
        proving_time: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let p_rational = BigRational::from_integer(1.into())
            / BigRational::from_integer(BigInt::from(BigUint::from(difficulty)));
        let p = p_rational
            .to_f64()
            .expect("difficulty-to-target conversion from `BigRational` to `f64` should succeed");
        let geo = Geometric::new(p).unwrap();
        let num_hashes = 1u64 + geo.sample(rng);
        let guessing_time = (num_hashes as f64) / hash_rate;
        proving_time + guessing_time
    }

    #[derive(Debug, Clone, Copy)]
    struct SimulationEpoch {
        log_hash_rate: f64,
        proving_time: f64,
        num_iterations: usize,
    }

    #[test]
    fn block_time_tracks_target() {
        // declare epochs
        let epochs = [
            SimulationEpoch {
                log_hash_rate: 2.0,
                proving_time: 300.0,
                num_iterations: 2000,
            },
            SimulationEpoch {
                log_hash_rate: 3.0,
                proving_time: 300.0,
                num_iterations: 2000,
            },
            SimulationEpoch {
                log_hash_rate: 3.0,
                proving_time: 60.0,
                num_iterations: 2000,
            },
            SimulationEpoch {
                log_hash_rate: 5.0,
                proving_time: 60.0,
                num_iterations: 2000,
            },
            SimulationEpoch {
                log_hash_rate: 2.0,
                proving_time: 0.0,
                num_iterations: 2000,
            },
        ];

        // run simulation
        let mut rng: StdRng = SeedableRng::from_rng(thread_rng()).unwrap();
        let mut block_times = vec![];
        let mut difficulty = U32s::<DIFFICULTY_NUM_LIMBS>::from(MINIMUM_DIFFICULTY);
        let target_block_time = 600f64;
        let target_block_interval = Timestamp::seconds(target_block_time.round() as u64);
        let mut new_timestamp = Timestamp::now();
        let mut block_height = BlockHeight::genesis();
        for SimulationEpoch {
            log_hash_rate,
            proving_time,
            num_iterations,
        } in epochs
        {
            let hash_rate = 10f64.powf(log_hash_rate);
            for _ in 0..num_iterations {
                let block_time = sample_block_time(hash_rate, difficulty, proving_time, &mut rng);
                block_times.push(block_time);
                let old_timestamp = new_timestamp;
                new_timestamp = new_timestamp + Timestamp::seconds(block_time.round() as u64);

                difficulty = difficulty_control(
                    new_timestamp,
                    old_timestamp,
                    difficulty,
                    Some(target_block_interval),
                    block_height,
                );
                block_height = block_height.next();
            }
        }

        // filter out monitored block times
        let allowed_adjustment_period = 1000usize;
        let mut monitored_block_times = vec![];
        let mut counter = 0;
        for epoch in epochs {
            monitored_block_times.append(
                &mut block_times
                    [counter + allowed_adjustment_period..counter + epoch.num_iterations]
                    .to_owned(),
            );
            counter += epoch.num_iterations;
        }

        // perform statistical test on block times
        let n = monitored_block_times.len();
        let mean = monitored_block_times.into_iter().sum::<f64>() / (n as f64);
        println!("mean block time: {mean}\ntarget is: {target_block_time}");

        let margin = 0.05;
        assert!(target_block_time * (1.0 - margin) < mean);
        assert!(mean < target_block_time * (1.0 + margin));
    }
}
