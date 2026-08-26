# Read-only audit: texture and material-parameter transport in the renderer

Read /home/user/algan/CLAUDE.md first. READ-ONLY: modify no file except writing
your report to /home/user/algan/scratch_perf/ox/REPORT_struct_textures.md. No
renders, no pytest. Cite file:line for every claim. CPU-only container; no
wall-clock claims. Other read-only audit sessions share this tree — ignore them.

Context: ranking STRUCTURAL speedup candidates (memory traffic, deduplication,
upload cost) in texture/material transport on the default deterministic path.
Sources: algan/rendering/raytracing/{scene_builder.py, primitives.py,
tracer.py, wavefront_kernels_taichi.py, sheet_resolve_taichi.py},
algan/rendering/primitives/triangle_primitive.py, algan/rendering/shaders/.

Claims (CONFIRMED/REFUTED + line numbers):
1. A texture image used by N primitives across T frames is stored once per
   batch in the merged upload, not N or T times. If refuted, say exactly what
   is duplicated and by what factor.
2. Two Mobs built from the same image file/tensor share one texture in the
   merge — state whether dedup exists, and its key (content hash? tensor
   identity? none).
3. Texture texels are uploaded and sampled as f32 (4 bytes/channel); no u8/f16
   texture storage exists on the default path.
4. An ANIMATED texture (ImageMob whose pixels animate) is stored one image per
   frame of the batch, with no dedup of identical consecutive frames.
5. Per-corner attributes (colors, uvs, material parameter columns) are carried
   per (frame, corner) in the merged arrays even when constant over the batch's
   frames — the merged time axis is dense for them. Name the arrays, dtypes and
   widths.
6. `_sample_tex_vec5` bilinear-samples with no LOD, and the sheet resolve
   shades one dominant fragment per sheet, so a minified texture is effectively
   point-sampled per sheet (RENDERER_WORK_QUEUE.md item 4). Confirm the call
   path, and confirm the sheet record already carries the exact covered area
   from which a mip level could be derived without derivatives.

Questions (answer from source only):
A. Walk the life of one texture from Mob attribute to kernel sample: every
   copy, device move, dtype conversion, and re-upload on the way — labelling
   each as per-batch, per-chunk, or per-frame.
B. From the code (not measured): what is the largest byte multiplier in this
   transport for (i) a static textured quad and (ii) an animated ~1774x887
   ImageMob over a 30-frame batch?

End the report with a "What I did not verify" section.
