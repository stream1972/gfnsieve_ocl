     1. Introduction
    -----------------

This is OpenCL port of Generalized Fermat Number (GFN) sieving program.
Original program was written by Anand Nair. OpenCL port by Roman Trunov.

Original program was written for NVIDIA CUDA only. This port could be run
on any OpenCL-enabled GPU.

Additional improvements over original version are:

- Better performance (less CPU usage) for low-N sieves (GFN-15,-16,-17);
- A 64-bit version of program which introduces even less CPU usage;
- An additional benchmark/simulation utility can be used to find maximum
possible performance of your CPU for low-N sieves (GFN-15,-16,-17), which
are usually limited by CPU and must use few CPU cores to feed GPU).

Although in can be run on any OpenCL device, NVIDIA still may have better
throughput because it's using optimized GPU assembly code, while other
cards have to rely on optimization of plain C code done by their OpenCL
compiler (driver).

For NVIDIA, compute capability 2.0 or better card is mandatory.

The high CPU usage problem on NVIDIA has been worked around (note: you
must specify a command-line option to enable this workaround). This problem
should not exist for other vendors.


     2. Command-line options
    -------------------------

All command-line options are compatible with original program.

A new command option "W<n>" has been added to work around 100% CPU usage
problem. By default the workaround is disabled, you must enable it manually.
To find do you need this workaround or not, run a test:

  gfnsvocl_xxx.exe 21 2000 2001

If your driver is affected, sieve program (actually, NVIDIA driver)
will consume full CPU core. Then enable workaround by running:

  gfnsvocl_xxx.exe 21 2000 2001 W1

A CPU usage now should be close to zero.

Note that this test must be run for GFN-21 or GFN-22 because low-N sieves
like GFN-15,-16,-17 will consume up to full CPU core even when workaround is
enabled. This is normal because lot of CPU work is required on these sieves.

Note that usage of the workaround will reduce your GPU load (and P/day) a bit.
To avoid this, use, if possible, high 'B' values (this reduces relative amount
of GPU time lost in the workaround) or run few instances of the program.


     3. CPU Benchmarks
    -------------------

Following is important for low-N sieves like GFN-15,-16,-17.

Small GFN-"N" requires lot of CPU power. If you're planning to run these
sieves, be aware that even a fastest CPU cannot feed even a average GPU.
You must run at least two processes on different ranges, until you reach
100% GPU usage.

For example, a GTX750ti GPU can do 55 P/day at fastest core (see below for
cores). Let's compare this value with speed of CPU (i7-4770K at 4000 MHz):

-------------+--------------------------------------------------------------
   Program   | GFN-15   16     17     18     19      20       21     22
-------------+--------------------------------------------------------------
gfnsvocl_w32 |   7.6   15.3   28.6   56.5   111.7   219.0   422.0   782.0
gfnsvocl_w64 |  12.1   24.5   44.6   88.0   174.7   342.0   660.0   1220.0
-------------+--------------------------------------------------------------

As can be seen from the table, to fully utilize this quite average GPU at
GFN-16, four instances of 32-bit or three instances of 64-bit program must
be run. A GFN-18 and above can be served by a single instance. It's also
clear that 64-bit version of the program must be used whenever possible.

To find maximum CPU throughput of your computer, use benchmark/simulation
program "gfnsvsim_xxx.exe". It runs only CPU part of sieve so numbers it
reports are maximum throughput which could be obtained by single CPU core
for given sieve range. Note: it can not find any factors (so it'll not
write checkpoint file to avoid occasional skip of a subrange).


     4. GPU Benchmarks
    -------------------

For NVIDIA cards, speed should be same or little better then original
program, probably due to better GPU utilization or better optimization
of recent compiler/driver.

Technically, program consist of 3 separate cores, applicable for different
sieve ranges:

    "63-bit" core - 0P to 9223P
    "64-bit" core - 9223P to about 18446P
    "79-bit" core - 18446P and above

where 63-bit core is fastest and 79-bit is slowest. Program automatically
chooses (or changes) a core for a range being sieved.

Average benchmarks for NVIDIA GTX750ti, GFN-21, B10, single instance (P/day):

      Core      |  GFNSvCUDA  |   gfnsvocl
----------------+-------------+--------------
 63-bit NVIDIA  |     47.0    |     55.1
 64-bit NVIDIA  |     40.0    |     48.9
 64-bit C [1]   |     n/a     |     42.3
 79-bit NVIDIA  |     27.3    |     30.8
 79-bit C [1]   |     n/a     |     23.5
----------------+-------------+---------------
[1] A "Plain C" core will be used on all non-NVIDIA GPUs. It also has
been manually benchmarked on NVIDIA to estimate slowdown, comparing to
hand-written NVIDIA assembly.

GPU throughput does not changes for 32-bit and 64-bit applications.

