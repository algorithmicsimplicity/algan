# Metal bloom: measured speedup and arena-border correction

The remaining Mac performance gap includes an application dispatch error.
`can_use_bloom_taichi` rejected every MPS tensor, although the patched compiler
can import its Metal storage. CPU and CUDA used the fused resize kernels; MPS
ran Python loops over individual image rows and columns.

Two UHD post-processing intervals in [run 73](https://github.com/algorithmicsimplicity/algan/actions/runs/34181920758)
executed 23,974 additions and 12,000 scalar interpolations. The native trace in
[run 75](https://github.com/algorithmicsimplicity/algan/actions/runs/34183492487)
recorded 37,504 MPSGraph encodes in those two intervals. These are graph
**executions**, not fresh compilations: the public compile API also services
cached executable lookups.

## Measured performance

Run 75 used one MPS process, identical seeded scenes, exact 1720 MiB arenas,
18 UHD frames in 18 chunks, CPU animation and software encoding. Its final
three renders alternated warm kernel/fallback/kernel without the detailed
profiler:

| Bloom resize path | Resize host wall | Full render wall |
| --- | ---: | ---: |
| Metal kernels | 1.021 s | 105.328 s |
| Original MPS fallback | 34.137 s | 151.203 s |
| Metal kernels | 1.054 s | 115.690 s |

Resizing saves approximately 33.1 seconds (32.4–33.4 times faster); total
render time falls 23.5–30.3%. These are measurements on a shared virtual Mac,
not a universal GPU/CPU speed ratio. Queue waits include useful device work,
and framework encoding times must not be labeled pure virtualization cost.
The runner exposes hardware acceleration through Apple's paravirtualized
Metal interface; a matched physical-Mac control is still needed to measure
virtualization's penalty.

## Why comparison with the original fallback initially failed

The small fixture in run 75 accidentally selected an identity resize and
provided no kernel-parity evidence. [Run 76](https://github.com/algorithmicsimplicity/algan/actions/runs/34184823324)
corrected that fixture and found large discrepancies. Independent CPU
references in [run 77](https://github.com/algorithmicsimplicity/algan/actions/runs/34185474243)
then localized the error to the **old MPS upsampling fallback**, not the
newly selected Metal kernels:

- Three nontrivial resize shapes, including non-divisible dimensions, match
  CPU interpolation within 4.77e-7 for Metal downsampling and 1.20e-7 for
  Metal upsampling.
- Initializing the upsample destination to -91 leaves that sentinel in the
  old MPS fallback's border rows. Metal writes all pixels correctly.
- All full-bloom intermediates agree with CPU; the final Metal image differs
  by at most 9.54e-7. The old fallback's final difference reaches 0.871.

PyTorch 2.7.1's
[`add_sub_lerp_template`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/mps/operations/BinaryOps.mm)
returns early when the interpolation weight is zero and `self.is_alias_of(output)`.
Distinct tensors in Algan's arena share storage and satisfy that alias test,
even when their ranges do not overlap. The optimization incorrectly skips
writing the destination. Clamped border coordinates produce exactly zero
weights, leaving unrelated scratch values in the bloom image.

## Production change

MPS float32 bloom now uses the existing kernels when the active backend is
Metal and both the patched import capability and conversion hook are present.
The fallback explicitly copies zero-weight MPS samples. This also keeps
fallback rendering correct when the fused path is unavailable. CPU/CUDA
selection and arithmetic, kernel bodies, FFT convolution, arena sizing and
synchronization boundaries are unchanged.

Regression tests compare both resize paths against independent CPU
interpolation with sentinel outputs, shared storage and nonzero offsets.
They cover three shapes, one/two-frame batches, three/four resized channels,
and complete four/five-channel bloom. Complete bloom allows only one output
byte of quantization difference from its CPU reference; the numerical
rounding differences are visually imperceptible. Larger differences from the
old fallback are repaired border pixels, not a precision tradeoff.

[Run 78](https://github.com/algorithmicsimplicity/algan/actions/runs/34185825573)
passes all 12 MPS regression tests on the exact production code. Its corrected
fallback and Metal fixtures have identical quantized output for four and five
channels. The final warm UHD comparison is 193.219 s fallback versus 127.246 s
Metal (34.1% less total time); resize time falls from 48.249 s to 1.030 s.

Across all 18 decoded frames, 1,562 of 447,897,600 channel values differ
(0.000349%); mean absolute difference is 0.00000656/255 and maximum is 6/255.
Only 367 values differ by more than 2. Inspection of full frames and the
maximum-difference crop finds no perceptible visual change. Exact video byte
identity is not required for acceptance. The full local suite passes 3,409 tests with 150 skips (including unavailable
heavy-render baselines). Its 14 compiler-setting tests pass separately without
the forced-eager environment override; the full run includes the fast CPU
image comparison. CPU/CUDA behavior is unchanged.
