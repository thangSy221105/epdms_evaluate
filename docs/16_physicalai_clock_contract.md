# PhysicalAI clock contract

The official Alpamayo loader sets `t0_us=5_100_000` and queries the PhysicalAI egomotion interpolator at `t0 + 100000*k` for `k=1..64`. `ego_future_xyz` is produced from those egomotion poses and transformed into the ego-local frame at the t0 pose.

The official PhysicalAI loader uses timestamp columns in microseconds, but the public code does not declare that zero is the clip origin. The pilot egomotion table contains negative timestamps and starts at `-190795`, so the origin cannot be asserted from field names or magnitude.

The public NCore PAI converter uses the egomotion timestamps to define the sequence interval and filters obstacle `timestamp_us` against that interval. Its implementation preserves the non-negative PAI timestamp values; it does not apply a scale or offset. This establishes same-domain handling inside the PAI→NCore path.

No public NCore→NuRec/NRE export mapping to `clipgt/*.parquet:key.timestamp_micros` was found in the inspected sources. Therefore the PhysicalAI clock is documented, but the cross-domain bridge remains unresolved.
