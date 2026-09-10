# Coupled rough-glass energy compensation

Implemented 2026-09-08. This is an approximate reciprocal compensation lobe,
not an exact simulation of microscopic multiple scattering. It preserves the
existing GGX single-scatter reflection/refraction response and adds missing
neutral power without splitting paths or changing the material API.

## The quantity being recovered

For incidence in medium i, integrate the **power** of the existing neutral
single-scatter dielectric over both outgoing hemispheres:

    E_i(v) = integral_reflection f_ss |cos(theta)| dw
           + integral_transmission f_ss |cos(theta)| dw / eta^2
    L_i(v) = 1 - E_i(v),       eta = n_i / n_t.

The second integral removes the radiance index-squared factor; otherwise an
entering interface would incorrectly appear to absorb most of the light.
The LUT generator integrates both Fresnel outcomes of every visible GGX facet,
including invalid outcomes as zero contribution, with the same correlated
Smith masking used by the renderer. It is not a reflection-only metal lookup.

Let barL_i = 2 integral_0^1 mu L_i(mu) dmu and
D = n_i^2 barL_i + n_t^2 barL_t. For either destination hemisphere j, add the
following **radiance** BSDF:

    f_ms(i,v -> j,w) = weight_rgb * n_i^2 L_i(v) L_j(w) / (pi D).

The squared indices can be normalised by their maximum before forming D.
The implementation does this to avoid overflow at extreme relative indices.
The coefficient is zero for a delta interface, matched indices, no compensable
neutral fraction or a vanishing D. Neither a runtime random walk nor an
unbounded iterative solve is needed.

### Shared power budget

The reflected integral is weight_rgb L_i(v) n_i^2 barL_i / D.
Dividing the transmitted radiance integral by eta^2 gives
weight_rgb L_i(v) n_t^2 barL_t / D. Their sum is exactly
weight_rgb L_i(v), to the accuracy of the tabulated single-scatter loss.
There is **one** recovered budget, not one for each lobe. Adding the term to
neutral white glass restores its integrated unit power.

The product L_i(v)L_j(w) is symmetric. Reflection is reciprocal. Swapping the
two media and directions for transmission gives f_forward = eta^2 f_reverse,
the same radiance reciprocity as the existing refractive single-scatter term.
The model's broad angular shape is a mean-field approximation; that identity
and the shared energy integral are algebraic, not assumptions about a fit.

## Preserve intended tint and absorption

Compensating every coloured response to white would remove authored absorption.
Instead, extract a neutral lossless component with per-channel weight w from
the existing facet response. Set m=clamp(metalness), p=((eta-1)/(eta+1))^2,
d=f0-m*albedo and t=clamp(T)*clamp(albedo). Use

    w = clamp(min(1-m, (1-m)*t, d/p, ((1-m)-d)/(1-p)), 0, 1).

The implementation guards the denominators; matched-index interfaces keep their
existing delta handling. Subtracting w*F from reflection and w*(1-F) from
transmission leaves nonnegative residual responses with total at most 1-w.
The final facet-transmission clamp is retained for boosted authored specular.
Only the neutral fraction gets the missing-energy correction; the residual
continues to use single-scatter transport. For default white full glass w=1.
For opaque glass, conductors and channels without a neutral fraction it is zero.
The extraction is invariant under eta -> 1/eta, preserving reciprocity.

This is deliberately conservative for strongly coloured or partly transmitting
materials. It does not claim the exact colour evolution of many microscopic
reflections/refractions. Beer-Lambert attenuation along scene segments remains
unchanged, as do coverage opacity, nested-medium transitions and roulette's
index-squared cancellation.

## Sampling and MIS

Choose the new lobe with p_ms = clamp(max(w)*L_i(v), 0, 0.95). Inside it choose
the reflected side with n_i^2 barL_i / D, otherwise the opposite side, then
sample that hemisphere with a cosine distribution. The existing independent
branch scalar selects and remaps these choices. The existing direction pair
samples either this cosine direction or the original visible-normal facet.
There is no added random dimension, shared mutable RNG, retry loop or path split.

For any candidate direction the conditional interface PDF is

    (1-p_ms) * pdf_single_scatter
      + p_ms * p_destination_side * abs(cos(theta)) / pi.

The full BSDF is always evaluated, not just the selected lobe. Its compensated
PDF participates in the outer diffuse/interface mixture and both NEE and
continuation MIS. The new term is evaluated even where the single-scatter
half-vector is invalid: microscopic multiple scattering has broader support
than a single facet, particularly above the macro critical angle. Invalid
single-scatter samples remain null samples, not a hidden resampling strategy.
Smooth delta reflection/TIR and matched-index straight transmission retain their
existing behavior and medium-stack rules. Geometric-normal support checks still
apply to perturbed shading normals.

## Lookup, generation and renderer memory

The bundled `_glass_energy_lut.npz` stores bounded losses quantised to uint8
(14,393 compressed bytes; at most 0.5/255 absolute quantisation error). The
loader decodes to float32 for all device arithmetic. Axes are
17 roughness values, 33 values of sqrt(abs((eta-1)/(eta+1))), and 65 values of
sqrt(abs(cosine)). Columns distinguish incidence from the low- and high-index
medium; each block has one additional cosine-average row. Shape is [37026, 2],
296,208 uncompressed bytes. The final ratio node uses a finite approximation
to the infinite-index limit. No data is downloaded or generated while rendering.

Trilinear interpolation gives the directional loss; the two means are bilinear.
The loader calculates the means of the decoded piecewise-linear sqrt(mu)
interpolant exactly with a small fixed weighted sum, not a different quadrature
grid or separately quantised mean. The CPU loader checks shape and uint8 dtype,
caches one immutable-by-convention NumPy array, and the host makes one scoped
arena copy per render. Arena slot 37, formerly the removed nonphysical area-light
falloff table, now carries this lookup; the binding indices, dtype and rank are
unchanged. No extra kernel argument, path-state byte or per-scene specialization
is introduced. The scoped allocation participates in preflight and OOM retry.

Regenerate from the repository root:

    .venv/bin/python scripts/generate_glass_energy.py

The generator requires NumPy only, does not import/initialise Algan, and uses
8,192 deterministic Hammersley visible-normal samples per directional row,
then rounds the bounded directional losses to uint8 for storage.
`--samples` accepts 1,024 through 65,536 for bounded-memory convergence checks.
A missing or corrupted shipped asset is an installation error, not a silent
fallback to dark single-scatter glass.

## Validation and limits

Local algebra validation used the generated table and a separate 65,536-sample
quadrature at 150 off-grid roughness/IOR/incidence combinations. Compensated
neutral power ranged from 0.9976869 to 1.0020714 (worst absolute error 0.0023131).
Single-scatter power reached 0.3090 in the same sample set. All stored means
matched the interpolant's exact cosine integral within 4.80e-8.

The actual helper and sampler source was also executed as scalar NumPy algebra,
without its compiler decorators, to independently integrate each hemisphere's
PDF, compare valid sample frequencies, check refractive reciprocity and verify
neutral-component facet budgets. This checks the formulas, **not Taichi codegen,
GPU behavior or rendered images**. Reciprocity's maximum relative discrepancy
in that double-precision probe was 4.15e-14.

`test_rough_dielectric.py` supplies compiled furnace, PDF-mass, reciprocity,
Snell/TIR and facet-budget regressions. API/geometry tests and existing panel
render regressions cover the accompanying physical-emission change. Compiled
Algan tests and image comparisons were not run in the implementation container:
the requested Library install skill/assets were not accessible and no Quadrants
or Taichi compiler was installed. No full suite, remote CI monitoring or
rebaseline was performed.

Finite table interpolation has nonzero error; the numerical sample above is
not a global bound over all roughness, grazing angles and indices. Very close
to index matching or critical angles, detailed angular multiple scattering is
still approximate. A microscopic random-walk reference and a GPU timing sweep
would be useful future validation; neither is claimed here. The change does not
add a caustic estimator or bend straight transparent-shadow connections.

## Measured on hardware, and the alternative that was not taken

The validation note above was written without a compiler available. It has
since been run. `benchmarks/rough_glass_furnace.py` sweeps 112 configurations
(4 roughnesses x 7 relative indices, both sides x 4 incidences) through the
shipped sampler and evaluator, and again through a zeroed table, which
disables the compensation lobe and leaves plain single scatter. Results are
recorded under `benchmarks/results/`. Uncompensated glass loses up to **69%**
of its power (mean 14%); compensated, the furnace closes to **0.73%** worst
case and 0.09% mean. `test_rough_glass_furnace_render.py` is the render-level
form of the same claim: a closed rough slab in a uniform environment must come
back within 1/255 of the environment radiance, unsaturated, at roughness 0.35,
0.65 and 1.0.

A second, independent implementation of this feature exists on
`codex/rough-glass-energy-compensation`. It reaches the same physical model by
the same route, differing in how the table is parameterised and stored: a
65 x 33 x 65 f32 grid, linear in `|(eta-1)/(eta+1)|` and in cosine, held in a
1.1 MB `.npy` appended to the `nee_meta` arena vector. Measured against a
65536-sample quadrature reference at 400 random points, that table has a lower
RMS error (0.00041 against this one's 0.00095) but a slightly *worse* worst
case (0.00504 against 0.00464), and it closes the furnace sweep above to 0.25%.
It was not adopted: the accuracy difference is far below the visible
threshold, this implementation passes the other's own strictest render-level
test unchanged, and the table it ships is 79 times larger in the wheel. Its
furnace benchmark and render test were taken; its table and kernels were not.

That comparison did locate a real improvement, which is **not** applied here
because it would move path-traced output and those baselines are
release-hosted. The error above is quantisation-limited, not grid-limited --
uint8 storage has a +-0.00196 floor, and the measured RMS sits right on it --
while the sqrt-warped eta axis already resolves index better than the
alternative's 65 linear samples. Raising `GLASS_ROUGHNESS_SIZE` from 17 to 33
and storing 16-bit measures at 0.00202 max / 0.00027 RMS, beating **both**
shipped tables on every metric at 283 KB raw, a quarter of the alternative's.
Roughness density is what buys this; widening the index axis to 65 as well
changes RMS only from 0.00027 to 0.00023 and is not worth the bytes.
