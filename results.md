## Ryzen 7 9700x

OPENBLAS_NUM_THREADS=8 (same as p-cores)

Command being timed: "./target/release/train"
User time (seconds): 52.20
System time (seconds): 0.02
Percent of CPU this job got: 799%
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:06.53
Average shared text size (kbytes): 0
Average unshared data size (kbytes): 0
Average stack size (kbytes): 0
Average total size (kbytes): 0
Maximum resident set size (kbytes): 78144
Average resident set size (kbytes): 0
Major (requiring I/O) page faults: 0
Minor (reclaiming a frame) page faults: 4479
Voluntary context switches: 7
Involuntary context switches: 189
Swaps: 0
File system inputs: 0
File system outputs: 2728
Socket messages sent: 0
Socket messages received: 0
Signals delivered: 0
Page size (bytes): 4096
Exit status: 0

Command being timed: "./target/release/train_large"
User time (seconds): 166.87
System time (seconds): 0.02
Percent of CPU this job got: 798%
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.90
Average shared text size (kbytes): 0
Average unshared data size (kbytes): 0
Average stack size (kbytes): 0
Average total size (kbytes): 0
Maximum resident set size (kbytes): 117740
Average resident set size (kbytes): 0
Major (requiring I/O) page faults: 0
Minor (reclaiming a frame) page faults: 9208
Voluntary context switches: 16
Involuntary context switches: 516
Swaps: 0
File system inputs: 0
File system outputs: 0
Socket messages sent: 0
Socket messages received: 0
Signals delivered: 0
Page size (bytes): 4096
Exit status: 0

Performance counter stats for './target/release/train_large':

                0      context-switches:u               #      0,0 cs/sec  cs_per_second
                0      cpu-migrations:u                 #      0,0 migrations/sec  migrations_per_second
            8.368      page-faults:u                    #     49,9 faults/sec  page_faults_per_second
      167.627,97 msec task-clock:u                     #      8,0 CPUs  CPUs_utilized
  116.819.810.219      L1-dcache-load-misses:u          #     16,7 %  l1d_miss_rate            (42,86%)
      49.707.849      branch-misses:u                  #      0,1 %  branch_miss_rate         (42,85%)
  41.027.900.437      branches:u                       #    244,8 M/sec  branch_frequency     (42,86%)
  876.596.766.356      cpu-cycles:u                     #      5,2 GHz  cycles_frequency       (42,86%)
1.998.160.756.363      instructions:u                   #      2,3 instructions  insn_per_cycle  (42,86%)
    5.854.495.391      stalled-cycles-frontend:u        #     0,01 frontend_cycles_idle        (42,86%)

    20,957570558 seconds time elapsed

    166,588455000 seconds user
      0,048340000 seconds sys

Performance counter stats for './target/release/train':

                0      context-switches:u               #      0,0 cs/sec  cs_per_second
                0      cpu-migrations:u                 #      0,0 migrations/sec  migrations_per_second
            3.327      page-faults:u                    #     63,3 faults/sec  page_faults_per_second
        52.533,48 msec task-clock:u                     #      8,0 CPUs  CPUs_utilized
  28.802.835.956      L1-dcache-load-misses:u          #     15,0 %  l1d_miss_rate            (42,86%)
      63.151.151      branch-misses:u                  #      0,3 %  branch_miss_rate         (42,86%)
  18.926.899.056      branches:u                       #    360,3 M/sec  branch_frequency     (42,86%)
  281.204.792.657      cpu-cycles:u                     #      5,4 GHz  cycles_frequency       (42,86%)
  467.294.683.664      instructions:u                   #      1,7 instructions  insn_per_cycle  (42,85%)
    2.254.545.520      stalled-cycles-frontend:u        #     0,01 frontend_cycles_idle        (42,86%)

      6,568182168 seconds time elapsed

    52,330256000 seconds user
      0,022754000 seconds sys

## Mac M3 (same as listed on README, time changes on hot run)

train

   8.82 real         8.79 user         0.03 sys
            69058560  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                4398  page reclaims
                   0  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   7  voluntary context switches
                  90  involuntary context switches
        110235167995  instructions retired
         28978515842  cycles elapsed
            64520600  peak memory footprint

train_large

       29.90 real        29.83 user         0.05 sys
           109150208  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                6848  page reclaims
                   0  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                  23  involuntary context switches
        357870075378  instructions retired
        110223328911  cycles elapsed
           104677832  peak memory footprint

## Prediction

OPENBLAS_NUM_THREADS=1 /usr/bin/time --verbose ./target/release/predict
............................
............................
............................
............................
...........................+
......++++++++++++++++++++++
+++++++++++++++++++++.......
............................
..........++++..............
...........+++#+............
..............+#+...........
...............#+...........
...........+++#+............
...........+##+#+...........
...............+#+..........
................+#+.........
.................#+.+.......
.................#+.........
............+..++#+.........
...........+++#+++..........
............................
........++.+++++++++++++++++
+++++++++++++++++++++++++...
............................
............................
............................
............................
............................

=== ReLU Model ===
Predicted: 8 | Actual: 3
Class Probabilities:
  0: 0.0317
  1: 0.0089
  2: 0.0450
  3: 0.2453
  4: 0.0051
  5: 0.1054
  6: 0.0069
  7: 0.0224
  8: 0.4662  <-- predicted
  9: 0.0631

=== Sigmoid Model ===
Predicted: 3 | Actual: 3
Class Probabilities:
  0: 0.0019
  1: 0.0011
  2: 0.0062
  3: 0.4820  <-- predicted
  4: 0.0001
  5: 0.0253
  6: 0.0001
  7: 0.0001
  8: 0.4811
  9: 0.0022
        Command being timed: "./target/release/predict"
        User time (seconds): 0.00
        System time (seconds): 0.00
        Percent of CPU this job got: 100%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 5720
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 936
        Voluntary context switches: 1
        Involuntary context switches: 1
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

OPENBLAS_NUM_THREADS=1 hyperfine -N --warmup 50 -m 5000 './target/release/predict'
Benchmark 1: ./target/release/predict
  Time (mean ± σ):     741.8 µs ±  78.3 µs    [User: 351.5 µs, System: 368.3 µs]
  Range (min … max):   654.3 µs … 1402.1 µs    5000 runs

  Warning: Statistical outliers were detected. Consider re-running this benchmark on a quiet system without any interferences from other programs. It might help to use the '--warmup' or '--prepare' options.
