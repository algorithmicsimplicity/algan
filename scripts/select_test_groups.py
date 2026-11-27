r"""Regenerate ``tests/groups/ci.txt`` and ``tests/groups/pre-pr.txt``.

The two groups are chosen to cover as much of ``algan/`` as their time
budgets allow, so a change that breaks something far from where it was made
still has a good chance of tripping a test. That is a measurement, not a
judgement, and this script is the measurement: give it one run of the suite
with per-test coverage contexts and per-test durations, and it picks greedily
by lines newly covered per second until each budget is spent.

    .venv/bin/python -m pytest -q -p no:cacheprovider tests/unit_tests tests/fast \
        -n 1 --durations=0 --cov=algan --cov-context=test --cov-report= \
        > durations.txt
    .venv/bin/python scripts/select_test_groups.py --durations durations.txt \
        --coverage .coverage

Run it against a WARM kernel cache (run the suite once before measuring):
both groups are run warm in practice -- CI restores the compiled kernels, and
a development machine has them from its last run -- and a cold run charges
each kernel's compile to whichever test happens to reach it first, which says
nothing about that test.

The CI group starts from the hand-curated fast suite (the tests marked
``fast``; see tests/README.md), chosen for exactly this purpose by reading the
code rather than by measuring it, plus the tests pinned in ``ALWAYS_CI``; the
pre-PR group starts from the whole CI group plus the tests pinned in
``ALWAYS_PRE_PR``. The greedy pass then adds whatever buys the most new
coverage per second.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT / "tests" / "groups"

# Seconds each group may spend, counted as a test's measured setup + call +
# teardown (a GitHub ubuntu-latest runner, -n 1, under coverage) plus
# PER_TEST_OVERHEAD. The overhead is what a runner actually charged per test
# beyond its reported duration when the first lists, which ignored it, ran:
# 5700 tests reporting 80 s took 4 minutes. With it, the CI budget plus
# collection comes to about two minutes on a runner, and the pre-PR budget to
# about nine, under the ten it promises.
PER_TEST_OVERHEAD = 0.02
DEFAULT_BUDGETS = {"ci": 90.0, "pre-pr": 480.0}

# Tests that add next to no lines of their own, so the greedy pass never picks
# them, yet are exactly what trips on a change made somewhere else. Two kinds:
# guards that read the source rather than run it -- kernel specs, launch
# arities, compiled-in settings, curation tables -- and the regression tests of
# a fix, which re-run lines other tests already cover (a regeneration dropped
# the sheet resolve's that way). They join the CI group by name, a prefix of a
# node id: this script seeds them, and ``--group`` runs every test they name
# whether a list carries it yet or not (tests/conftest.py).
ALWAYS_CI = (
    "tests/unit_tests/test_kernel_precompile.py::test_every_builtin_spec_still_matches_a_live_kernel",
    "tests/unit_tests/test_arena_args.py::test_prologue_matches_the_spec_the_host_packs_from",
    "tests/unit_tests/test_arena_args.py::test_no_kernel_asks_for_more_ndarrays_than_metal_can_bind",
    "tests/unit_tests/test_compiled_in_settings.py::test_every_static_gate_over_a_setting_is_declared",
    "tests/unit_tests/test_cuda_codegen_switches.py::test_each_megakernel_gates_its_main_loop",
    "tests/unit_tests/test_taichi_early_return.py::test_no_function_in_algans_own_kernel_modules_is_rewritten",
    "tests/unit_tests/test_taichi_source_key.py::test_every_algan_kernel_body_walks_without_poison",
    "tests/unit_tests/test_fast_suite_curation.py::",
    "tests/unit_tests/test_test_groups.py::",
    "tests/unit_tests/test_render_coverage_audit.py::test_every_native_renderable_class_appears_in_a_scene",
    # A script cannot change BLUE or ORIGIN for the code after it, on the warm
    # daemon included; Manim idioms on a compatibility Mob (animate, get_color,
    # a scalar argument); integer colour tuples; and the root adapters' degree
    # and stroke-width conventions in methods as well as constructors.
    "tests/unit_tests/test_shared_constants.py::",
    "tests/unit_tests/test_manim_compat_idioms.py::",
    "tests/unit_tests/test_color_arguments.py::test_an_integer_tuple_is_manims_0_255_under_algan_manim",
    "tests/unit_tests/test_color_arguments.py::test_an_integer_tuple_above_one_is_refused_at_the_root",
    "tests/unit_tests/test_color_arguments.py::test_whole_number_channels_in_range_are_still_a_color",
    "tests/unit_tests/test_color_arguments.py::test_a_color_compares_with_anything_and_hashes_by_identity",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_rotate_and_orbit_warn_like_the_constructors",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_rotate_takes_ordinary_degrees_in_silence",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_rotate_in_radians_on_purpose_does_not_warn",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_the_rotate_warning_points_at_the_script",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_get_angle_and_set_angle_are_in_degrees_at_the_root",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_rotate_about_origin_is_in_degrees_at_the_root",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_set_stroke_width_is_algan_units_at_the_root",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_a_nested_style_dict_stroke_width_is_algan_units_at_the_root",
    "tests/unit_tests/test_manim_opengl_surface.py::test_a_mesh_in_the_scene_registers_its_lines_as_actors",
    "tests/unit_tests/test_manim_adapter_conventions.py::test_every_adapted_class_has_a_derived_angle_table",
    "tests/unit_tests/test_manim_animation_audit.py::test_high_value_animation_api_is_exported",
    "tests/unit_tests/test_scene_pixel_lengths.py::test_every_public_scene_method_answers_on_the_class",
    # Timeline regressions, recorded and materialized without a render:
    # removing an updater inside a block that rescales on exit, a block that
    # raises, the context tree and the segment caches after materializing,
    # the parameter-name memo's hold on a script, a context's repr, and
    # Wiggle beside a move.
    "tests/unit_tests/test_updater_removal.py::",
    "tests/unit_tests/test_failed_block_rollback.py::",
    "tests/unit_tests/test_updater_mob_creation.py::test_materializing_frames_leaves_the_context_tree_as_it_was",
    "tests/unit_tests/test_texture_time_lerp.py::test_a_render_drops_the_segment_window_caches_when_it_ends",
    "tests/unit_tests/test_animated_function_arguments.py::test_a_recorded_function_does_not_keep_its_script_alive",
    "tests/unit_tests/test_animated_function_arguments.py::test_functions_sharing_a_code_object_keep_their_own_parameter_names",
    "tests/unit_tests/test_edge_case_timeline.py::test_a_context_prints_without_the_tree_it_belongs_to",
    "tests/unit_tests/test_indication_animations.py::test_wiggle_composes_with_a_move_running_alongside_it",
    "tests/unit_tests/test_shape_style_profiles.py::test_snapshot_matches_the_installed_manim",
    "tests/unit_tests/test_renderer_audit_contract.py::test_every_audit_scene_normalizes_identically_in_both_bridges",
    # FXAA returns a frame with no edge in it unchanged: tensors only.
    "tests/unit_tests/test_fxaa.py::",
    # The sheet resolve's event pass compiles none of its continuation fans.
    "tests/unit_tests/test_sheet_resolve_variants.py::test_mode_1_unrolls_no_continuation_fan",
    # Nor, under the glossy prefilter, spreads one over the GGX lobe.
    "tests/unit_tests/test_sheet_resolve_variants.py::test_only_the_tap_fan_modes_spread_a_continuation_over_the_lobe",
    # The glossy prefilter's split-sum weight is the direct lobe's albedo
    # table: a rough white metal keeps nearly all its reflection, the
    # roughness-0 row is Schlick, and no lookup reflects more than arrives.
    # One small probe kernel, a fraction of a second.
    "tests/unit_tests/test_glossy_prefilter.py::test_env_brdf_beats_the_mirror_share_throttle_on_a_rough_metal",
    "tests/unit_tests/test_glossy_prefilter.py::test_env_brdf_reduces_to_fresnel_at_zero_roughness",
    "tests/unit_tests/test_glossy_prefilter.py::test_env_brdf_never_reflects_more_than_arrives",
    # Its launches name every argument, so none can land in another's place.
    "tests/unit_tests/test_sheet_resolve_variants.py::test_a_resolve_launch_names_every_argument",
    # The sequential oracle's band, claim, residue and NaN rules, which the
    # kernel follows (test_sheet_resolve_parity.py): tensors only.
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_band_occludes_by_each_siblings_own_alpha",
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_band_writes_when_its_last_sibling_claims_nothing",
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_band_the_list_leaves_open_still_occludes",
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_zero_area_sibling_keeps_its_band_open",
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_claims_no_more_than_the_pixel_still_shows",
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_redistributes_only_outside_the_sheets_mask",
    "tests/unit_tests/test_sheet_compaction.py::test_oracle_drops_a_nan_area",
    # What the docs and the agent skill say about the repository: every public
    # class has a reference page, and the skill names one commit and the
    # package's version. Reads files; no Scene.
    "tests/unit_tests/test_docs_consistency.py::",
    # MPS-friendly mode spends the closed-shell ceiling per segment.
    "tests/unit_tests/test_sheet_chain_kernels.py::test_float32_shell_ceiling_spend_does_not_degrade_with_stream_length",
    "tests/unit_tests/test_sheet_chain_kernels.py::test_float32_segment_prefix_matches_the_float64_global_difference",
    # The closed-shell ceiling clamps each segment on that segment alone, which
    # is what lets a stream with a closed solid take the stroke-pixel split; the
    # split's kernels match its torch arm bit for bit.
    "tests/unit_tests/test_sheet_chain_kernels.py::test_shell_ceiling_clamp_reads_only_its_own_segment",
    "tests/unit_tests/test_stroke_sheets.py::test_stroke_pixel_split_holds_with_a_closed_solid",
    "tests/unit_tests/test_stroke_sheets.py::test_stroke_pixel_split_kernels_match_the_torch_arm",
    # A material, a light and a fragment stage are checked when they are made:
    # constructors and one set_material call, no render.
    "tests/unit_tests/test_material_validation.py::",
    "tests/unit_tests/test_lights.py::test_hemisphere_ground_color_accepts_hex_like_every_material_color",
    "tests/unit_tests/test_lights.py::test_hemisphere_ground_color_is_validated_at_construction",
    "tests/unit_tests/test_lights.py::test_a_light_on_its_own_target_says_it_emits_nothing",
    "tests/unit_tests/test_lights.py::test_a_light_aimed_at_a_mob_where_it_stands_says_so",
    "tests/unit_tests/test_lights.py::test_a_light_off_its_target_does_not_warn",
    "tests/unit_tests/test_fragment_shaders.py::test_a_stage_parameter_named_like_a_mob_attribute_is_refused",
    "tests/unit_tests/test_fragment_shaders.py::test_the_reserved_state_names_are_the_ones_a_mob_is_built_with",
    "tests/unit_tests/test_fragment_shaders.py::test_material_slot_names_and_earlier_stage_parameters_stay_legal",
    "tests/unit_tests/test_fragment_shaders.py::test_a_stage_the_kernel_could_not_compile_fails_when_made",
    # A vertex shader parameter named like a Mob attribute warns: one
    # set_shader call on an unspawned Sphere each, no render.
    "tests/unit_tests/test_custom_shader_signature.py::test_a_material_parameter_named_like_a_mob_attribute_warns",
    "tests/unit_tests/test_custom_shader_signature.py::test_a_material_parameter_named_like_a_subclass_attribute_warns",
    "tests/unit_tests/test_custom_shader_signature.py::test_setting_the_same_custom_shader_twice_does_not_warn",
    # A fragment the ceiling does not clip comes back bit for bit, which on an
    # Apple GPU it did not while the kernel divided it by its own area.
    "tests/unit_tests/test_sheet_chain_kernels.py::test_solid_shell_ceiling_returns_what_it_does_not_clip_bit_for_bit",
    # Integer gathers that MPS rounds past 2**24, with the rounding emulated on
    # the CPU: a circuit's or triangle's ref through the compaction, its sample
    # depth rule and the fragment gather; the Bezier header's bounds, bin sizes
    # and intervals; and the advanced indexing ``gather_exact`` stands on.
    "tests/unit_tests/test_mps_fragment_refs.py::test_circuit_refs_past_the_ceiling_leave_the_compaction_exact",
    "tests/unit_tests/test_mps_fragment_refs.py::test_a_triangle_ref_past_the_ceiling_keeps_its_surface_in_the_sample_depth_rule",
    "tests/unit_tests/test_mps_fragment_refs.py::test_the_fragment_gather_carries_refs_past_the_ceiling_exactly",
    "tests/unit_tests/test_bezier_header_mps_gather.py::test_the_bounds_and_bin_sizes_survive_the_mps_integer_gather",
    "tests/unit_tests/test_bezier_header_mps_gather.py::test_a_point_inside_a_circuit_reads_inside_through_the_mps_integer_gather",
    "tests/unit_tests/test_bezier_header_mps_gather.py::test_the_bin_intervals_survive_the_mps_repeat_interleave",
    "tests/unit_tests/test_mps_friendly.py::test_advanced_indexing_is_exact_above_the_mps_ceiling",
    # Mob and Scene authoring guards that build nothing heavier than a Square:
    # the grid's rows, placement arguments, a failed set, the Scene stack and
    # show_frame's settings.
    "tests/unit_tests/test_group_layout.py::",
    "tests/unit_tests/test_mob_layout_validation.py::",
    "tests/unit_tests/test_scene_stack.py::",
    "tests/unit_tests/test_mob_set_guards.py::test_a_failed_set_non_recursive_still_lets_transforms_carry_children",
    "tests/unit_tests/test_mob_set_guards.py::test_detaching_or_morphing_an_unspawned_mob_leaves_it_unspawned",
    "tests/unit_tests/test_scene_side_outputs.py::test_show_frame_renders_at_the_settings_save_frame_would_use",
    "tests/unit_tests/test_scene_side_outputs.py::test_show_frame_without_matplotlib_says_how_to_get_it",
    # A Group grown by adds is the Group built at once, and an add costs the
    # same however large the Group is (counted, not timed).
    "tests/unit_tests/test_group_add.py::",
    # A Mob scaled to nothing grows back to its size, shape and orientation,
    # and a shrink to nothing still ends a video.
    "tests/unit_tests/test_scale_to_zero.py::",
    # More of the same, and the ops MPS gets wrong around it, emulated on the
    # CPU and, on an Apple machine, run through the real op: a BVH level's
    # ranges and the sliver strips, the depth-order scan's owners, the depth
    # gate's surfaces and the edge tables' circuits, the exact unique and the
    # two tables built on it, and the int64 extrema MPS cannot reduce.
    "tests/unit_tests/test_bvh_mps_gathers.py::",
    "tests/unit_tests/test_depth_order_warning.py::test_owners_past_2_24_stay_apart_through_the_mps_gather",
    "tests/unit_tests/test_ids_mps_gathers.py::test_the_depth_gate_tells_surfaces_apart_past_the_ceiling",
    "tests/unit_tests/test_ids_mps_gathers.py::test_the_edge_tables_file_each_edge_under_its_circuit_past_the_ceiling",
    "tests/unit_tests/test_mps_friendly.py::test_the_exact_unique_matches_unique",
    "tests/unit_tests/test_mps_friendly.py::test_the_exact_unique_handles_empty_and_single_tables",
    "tests/unit_tests/test_mps_friendly.py::test_the_exact_unique_does_not_go_through_unique",
    "tests/unit_tests/test_mps_friendly.py::test_the_exact_unique_is_exact_above_the_mps_ceiling",
    "tests/unit_tests/test_hybrid_transport.py::test_shell_table_keeps_surface_ids_past_2_24_apart_through_mps_unique",
    "tests/unit_tests/test_hybrid_transport.py::test_shell_table_keeps_surface_ids_past_2_24_apart_on_the_device",
    "tests/unit_tests/test_sheet_rank_groups.py::test_the_torch_arm_keeps_keys_past_2_24_apart_through_mps_unique",
    "tests/unit_tests/test_sheet_rank_groups.py::test_the_torch_arm_keeps_keys_past_2_24_apart_on_the_device",
    "tests/unit_tests/test_mps_int64_reductions.py::",
    # Model import and surface morph rules, on files written to tmp_path and
    # small grids: a shared material decoded once, the factor's precision,
    # channels bound by index, a mirrored node's winding, the skin's top four,
    # and set_shape_to taking the target's shape as it is.
    "tests/unit_tests/test_gltf_import_semantics.py::test_a_material_shared_by_several_primitives_is_decoded_once",
    "tests/unit_tests/test_model_import.py::test_a_mirrored_node_is_rewound_so_derived_normals_face_out",
    "tests/unit_tests/test_gltf_import_semantics.py::test_dark_base_color_factors_keep_their_float_precision",
    "tests/unit_tests/test_gltf_import_semantics.py::test_float_vertex_colors_beside_a_material_keep_their_precision",
    "tests/unit_tests/test_model_import.py::test_gltf_channels_bind_to_their_node_by_index_not_by_name",
    "tests/unit_tests/test_fbx_import.py::test_skin_weights_keep_the_four_strongest_influences_per_vertex",
    "tests/unit_tests/test_surface_shape_and_image_mob.py::test_set_shape_to_takes_a_user_surfaces_own_function",
    "tests/unit_tests/test_surface_shape_and_image_mob.py::test_set_shape_to_takes_a_turned_targets_shape_as_it_is",
    # A packed Cone keeps its direction, a packed capped solid caps every
    # member, and a Surface child from_batches cannot pack is refused.
    "tests/unit_tests/test_batched_surface_mobs.py::test_a_packed_cone_keeps_its_direction",
    "tests/unit_tests/test_batched_surface_mobs.py::test_a_packed_capped_solid_caps_every_member",
    "tests/unit_tests/test_batched_surface_mobs.py::test_from_batches_refuses_a_surface_child_it_cannot_pack",
    # A bad coordinate function (NaN, wrong shape) is named up front rather
    # than read as a grid too coarse.
    "tests/unit_tests/test_surface_autotune.py::test_a_bad_coordinate_function_is_named_before_the_sizing_search",
    "tests/unit_tests/test_surface_autotune.py::test_a_bad_coordinate_function_is_named_at_a_fixed_resolution_too",
    # Node animation turns a mesh's authored normals rather than recomputing
    # them, so a UV seam stays smooth while a clip plays.
    "tests/unit_tests/test_model_import.py::test_playback_keeps_the_authored_normals_across_a_uv_seam",
    "tests/unit_tests/test_fbx_import.py::test_play_animation_turns_the_authored_normals_rather_than_recomputing_them",
    # A sheet's float32 area, and the weights pooled from it, come back bit
    # for bit across the stroke split's arms and across replays of a frame.
    "tests/unit_tests/test_sheet_area_order.py::",
    # Which batches the sheet route takes, and which fall back.
    "tests/unit_tests/test_sheet_route_vetoes.py::",
    # The sheet route's tile out-of-memory retry: a stand-in arena, no kernel.
    "tests/unit_tests/test_sparse_tile_oom_retry.py::",
    # A setting that no longer changes anything refuses a write.
    "tests/unit_tests/test_settings_api.py::test_an_inert_field_is_readable_but_refuses_a_write",
    # The daemon, CLI, viewer and settings regressions, on stubs, loopback
    # sockets and two-frame clips: joining clips that differ or have no sound;
    # the handshake that keeps the environment from an impostor, the request
    # cap and the registration a second daemon may not take; cheap commands
    # that do not import torch; a restore that leaves untouched fields alone;
    # and a viewer that recovers from one failed render.
    "tests/unit_tests/test_project.py::test_project_concat_keeps_narration_after_a_silent_first_scene",
    "tests/unit_tests/test_project.py::test_project_concat_refuses_to_stream_copy_different_resolutions",
    "tests/unit_tests/test_project.py::test_project_concat_reencodes_different_resolutions_with_a_warning",
    "tests/unit_tests/test_daemon_client.py::test_nothing_is_sent_to_a_listener_that_cannot_prove_the_token",
    "tests/unit_tests/test_daemon_client.py::test_a_daemon_of_another_protocol_is_not_contacted",
    "tests/unit_tests/test_daemon_client.py::test_a_cuda_visible_devices_mismatch_is_refused",
    "tests/unit_tests/test_daemon_client.py::test_the_daemon_home_and_log_are_owner_only",
    "tests/unit_tests/test_daemon_socket.py::test_a_proved_verb_is_queued",
    "tests/unit_tests/test_daemon_socket.py::test_a_verb_that_carries_the_token_itself_is_refused",
    "tests/unit_tests/test_daemon_socket.py::test_a_caller_with_the_wrong_token_is_refused",
    "tests/unit_tests/test_daemon_socket.py::test_a_proof_for_one_verb_does_not_authorise_another",
    "tests/unit_tests/test_daemon_client.py::test_a_listener_that_cannot_prove_the_token_is_not_reachable",
    "tests/unit_tests/test_cli.py::test_a_daemon_from_before_the_upgrade_can_still_be_stopped",
    "tests/unit_tests/test_daemon_socket.py::test_a_run_is_refused_without_the_clients_proof",
    "tests/unit_tests/test_daemon_socket.py::test_a_run_request_too_large_is_refused_before_it_is_read",
    "tests/unit_tests/test_daemon_socket.py::test_a_protocol_4_client_is_told_to_restart_the_daemon",
    "tests/unit_tests/test_daemon_socket.py::test_a_second_daemon_does_not_take_a_live_daemons_registration",
    "tests/unit_tests/test_daemon_socket.py::test_a_registration_nobody_answers_is_claimed",
    "tests/unit_tests/test_daemon_socket.py::test_an_unreadable_state_file_is_claimed",
    "tests/unit_tests/test_daemon_cancellation.py::test_a_cancel_interrupts_a_blocking_wait_at_once",
    "tests/unit_tests/test_cli_startup.py::test_a_daemon_trigger_does_not_import_torch",
    "tests/unit_tests/test_cli_startup.py::test_launching_a_script_does_not_import_torch",
    "tests/unit_tests/test_cli.py::test_new_appends_py_to_a_name_without_it",
    "tests/unit_tests/test_project_selection_and_cli.py::test_run_cli_names_an_action_it_did_not_recognise",
    "tests/unit_tests/test_project_selection_and_cli.py::test_run_cli_stays_quiet_about_a_launchers_own_arguments",
    "tests/unit_tests/test_settings_sections_and_audio_effect.py::test_global_override_keeps_the_identity_of_fields_it_did_not_change",
    "tests/unit_tests/test_settings_sections_and_audio_effect.py::test_root_override_undoes_a_change_made_inside_it_to_another_section",
    "tests/unit_tests/test_settings_sections_and_audio_effect.py::test_section_override_keeps_a_change_made_inside_it_to_another_field",
    "tests/unit_tests/test_settings_sections_and_audio_effect.py::test_restore_rewrites_only_the_fields_that_differ",
    "tests/unit_tests/test_settings_sections_and_audio_effect.py::test_restore_still_puts_back_a_field_changed_in_place",
    "tests/unit_tests/test_viewer_frame_cache.py::test_one_failed_render_does_not_fail_every_later_frame",
    "tests/unit_tests/test_viewer_frame_cache.py::test_a_new_resolution_clears_the_error_of_the_old_one",
    "tests/unit_tests/test_viewer_frame_cache.py::test_inspections_nobody_waits_for_any_more_are_dropped",
    "tests/unit_tests/test_viewer_server.py::test_the_daemon_stops_a_viewer_its_run_left_serving",
    # The closed-shell coverage ceiling, on synthetic streams: tensors only.
    "tests/unit_tests/test_closed_shell_ceiling.py::",
    # The one-mesh ceiling behind a refracted sheet (the glass rim), at the
    # oracle and the emission: tensors and two small kernels.
    "tests/unit_tests/test_one_solid_ceiling.py::",
    # A triangle too big for the coverage lattice is sent to the ray cast.
    "tests/unit_tests/test_raster_lattice_bound.py::test_lattice_demotion_keeps_safe_triangles_and_sends_the_rest_to_half",
    # The circuit wedge against exact areas: a corner's two walls through any
    # screen map, and by identity rather than the nearest's repeat, an edge
    # through pixel centres, a corner on one, and stems and slits narrower
    # than a pixel. Small raster renders, a fraction of a second each.
    "tests/unit_tests/test_tilted_circuit_coverage.py::test_a_corner_keeps_both_walls_through_any_screen_map",
    "tests/unit_tests/test_tilted_circuit_coverage.py::test_an_edge_through_pixel_centres_half_covers_its_pixels_up_to_the_corners",
    "tests/unit_tests/test_tilted_circuit_coverage.py::test_a_small_square_keeps_both_walls_at_its_corners",
    "tests/unit_tests/test_tilted_circuit_coverage.py::test_a_corner_on_a_pixel_centre_covers_a_quarter_of_its_pixel",
    "tests/unit_tests/test_tilted_circuit_coverage.py::test_a_stem_and_a_slit_narrower_than_a_pixel_cover_their_areas",
    # Text and 2D shapes, without a render: a right-to-left line Pango drops,
    # the SVG recipe cache as plain data that is never unpickled, a private
    # tessellation RNG and whole cache entries, typesetting that is bounded
    # and keyed on its compiler, a minus sign only on digits that are not
    # zero, and degenerate radii, arcs and control points reported by name.
    "tests/unit_tests/test_text_right_to_left.py::",
    "tests/unit_tests/test_manim_svg_cache.py::test_the_in_process_recipe_memo_is_bounded",
    "tests/unit_tests/test_manim_svg_cache.py::test_a_cache_file_is_never_unpickled",
    "tests/unit_tests/test_manim_svg_cache.py::test_a_recipe_naming_a_class_outside_vendored_manim_is_refused",
    "tests/unit_tests/test_manim_svg_cache.py::test_recipes_are_saved_as_plain_data_and_rebuild_the_same_glyphs",
    "tests/unit_tests/test_manim_svg_cache.py::test_two_saves_of_one_key_write_through_different_temporary_files",
    "tests/unit_tests/test_tessellation_cache.py::test_tessellating_leaves_the_global_rng_alone",
    "tests/unit_tests/test_tessellation_cache.py::test_an_entry_is_published_whole",
    "tests/unit_tests/test_typeset_cache.py::test_every_typesetting_run_is_bounded_and_reads_no_stdin",
    "tests/unit_tests/test_typeset_cache.py::test_a_non_positive_timeout_means_no_limit",
    "tests/unit_tests/test_typeset_cache.py::test_one_document_under_two_compilers_is_cached_twice",
    "tests/unit_tests/test_numeric_display.py::test_a_value_that_rounds_to_zero_has_no_minus_sign",
    "tests/unit_tests/test_numeric_display.py::test_a_count_through_zero_never_shows_minus_zero",
    "tests/unit_tests/test_numeric_display.py::test_a_value_beyond_float32_range_says_so_rather_than_infinite",
    "tests/unit_tests/test_shape_accessors.py::test_a_zero_radius_circle_warns_that_it_cannot_grow",
    "tests/unit_tests/test_shape_accessors.py::test_a_negative_radius_circle_warns_that_it_faces_inward",
    "tests/unit_tests/test_shape_accessors.py::test_a_positive_radius_circle_does_not_warn",
    "tests/unit_tests/test_shape_accessors.py::test_a_full_turn_line_arc_is_refused",
    "tests/unit_tests/test_shape_accessors.py::test_other_line_arcs_still_build_finite_points",
    "tests/unit_tests/test_shape_accessors.py::test_non_finite_control_points_are_refused_by_name",
    # An extent past int32 takes the torch arm, or is refused, instead of wrapping.
    "tests/unit_tests/test_sheet_stream.py::test_kernel_extents_stop_below_2_31_on_each_extent",
    "tests/unit_tests/test_sheet_stream.py::test_a_chunk_past_the_emissions_int32_extents_is_refused",
    "tests/unit_tests/test_sheet_compaction.py::test_the_lane_owner_table_bounds_its_entries_not_its_rows",
    "tests/unit_tests/test_sheet_compaction.py::test_a_lane_table_at_the_extent_limit_takes_the_torch_arm",
    "tests/unit_tests/test_sheet_compaction.py::test_a_stream_past_every_kernel_extent_compacts_in_torch",
    "tests/unit_tests/test_sheet_compaction.py::test_the_raster_fragment_passes_past_the_extent_limit_take_torch",
    "tests/unit_tests/test_sheet_chain_kernels.py::test_class_pairs_flat_past_the_extent_limit_expands_in_torch",
    # The opaque-prefix truncation's torch arm keeps each pixel's own prefix
    # past 2**24 covered pixels, where MPS rounds a pixel ordinal it gathers
    # or repeats: tensors only, 17 million rows, under a second.
    "tests/unit_tests/test_sheet_compaction.py::test_the_opaque_prefix_keeps_each_pixels_own_past_2_24_covered_pixels",
    # A cgroup's page cache is no memory pressure, and a trace-counter reading
    # survives the reset real pressure brings: files and stubs, no kernel.
    "tests/unit_tests/test_manual_memory.py::test_a_cgroup_counts_its_working_set_not_its_page_cache",
    "tests/unit_tests/test_trace_counters.py::test_a_reading_spans_a_program_teardown",
    # The renderer's infrastructure, on stubs and temporary directories: a
    # kernel cache that cannot be written or lists what it cannot load, the
    # early-return rewrite in the source key, what the precompile context and
    # stamp cover, Inductor's cache per CPU, the cgroup-sized CPU budget, the
    # arena-prefix bound, a volume chunk's transforms, OptiX said once, and a
    # video published by rename with temporaries of its own.
    "tests/unit_tests/test_taichi_source_key.py::test_a_miss_tells_the_progress_account",
    "tests/unit_tests/test_taichi_source_key.py::test_an_indexed_artifact_that_will_not_load_is_reported_once",
    "tests/unit_tests/test_taichi_source_key.py::test_an_index_that_cannot_be_written_is_a_miss_not_a_crash",
    "tests/unit_tests/test_taichi_source_key.py::test_the_early_return_rewrite_revision_is_keyed",
    "tests/unit_tests/test_kernel_precompile.py::test_the_context_moves_only_with_what_the_source_key_folds_in",
    "tests/unit_tests/test_kernel_precompile.py::test_the_stamp_covers_what_reaches_a_kernel_and_not_the_authoring_layer",
    "tests/unit_tests/test_clear_cache.py::test_the_inductor_cache_is_split_by_cpu_model_and_flags",
    "tests/unit_tests/test_manual_memory.py::test_the_cpu_budget_is_sized_from_the_cgroup_limit_inside_a_container",
    "tests/unit_tests/test_preflight_overlap.py::test_prefix_search_bounds_projection_by_the_measured_model",
    "tests/unit_tests/test_pt_volumes.py::test_a_chunk_of_the_window_reads_the_same_transforms_as_the_whole",
    "tests/unit_tests/test_optix.py::test_a_runtime_optix_cannot_have_is_said_once",
    "tests/unit_tests/test_video_writer_failures.py::test_a_destination_that_cannot_be_replaced_keeps_the_finished_video",
    "tests/unit_tests/test_video_writer_failures.py::test_save_video_leaves_a_video_it_could_not_publish_in_place",
    "tests/unit_tests/test_video_writer_failures.py::test_an_odd_sized_h264_video_says_it_will_not_play_in_a_browser",
    "tests/unit_tests/test_video_writer_failures.py::test_save_video_temporaries_are_its_own_and_hidden",
    "tests/unit_tests/test_video_writer_failures.py::test_a_dead_renders_temporaries_are_removed_and_a_live_ones_kept",
    # A render on MPS holds torch's allocator to Metal's working set and gives
    # torch its own limit back after, a failed render too; and import algan
    # starts that allocator with the low watermark the hold relies on, unless
    # the user chose one or the allocator already started: stubs, no GPU.
    "tests/unit_tests/test_mps_working_set_hold.py::test_a_render_on_mps_holds_torch_to_the_working_set_until_it_ends",
    "tests/unit_tests/test_mps_working_set_hold.py::test_a_render_that_fails_still_gives_the_limit_back",
    "tests/unit_tests/test_mps_working_set_hold.py::test_algan_gives_torchs_mps_allocator_a_low_watermark_before_it_starts",
    "tests/unit_tests/test_mps_working_set_hold.py::test_a_watermark_already_chosen_stands",
    "tests/unit_tests/test_mps_working_set_hold.py::test_an_allocator_that_already_started_is_left_alone",
    # While a render job runs, only its thread drains torch's MPS caches, and a
    # drain the batch-prep worker asks for waits for that thread: stubs, no GPU.
    "tests/unit_tests/test_mps_cache_drains.py::test_a_drain_asked_for_off_the_render_thread_waits_for_the_render_thread",
    # The repository audits tests/README.md says gate every pull request. They
    # read source, docs or committed files rather than run Algan, so they cover
    # no lines and the greedy pass never chose one: the render-coverage and
    # scene-convention audits, the committed baseline pointer, the static tier
    # of the doc examples, the two source walks of the environment registry,
    # and the viewer's JavaScript, which Python coverage cannot see at all.
    # Each file or test here runs in under three seconds.
    "tests/unit_tests/test_render_coverage_audit.py::",
    "tests/unit_tests/test_baseline_store.py::",
    "tests/unit_tests/test_doc_examples.py::test_doc_example_uses_public_api",
    "tests/unit_tests/test_environment.py::test_package_reaches_algan_variables_only_through_the_accessors",
    "tests/unit_tests/test_environment.py::test_import_time_variables_match_where_the_code_reads_them",
    "tests/unit_tests/test_viewer_client.py::",
    # Files added after the groups were last measured (2026-10-23), so the
    # greedy pass has never seen them; pinned until the next regeneration
    # measures them. MNEE caustics' chain solver and its derivatives, and the
    # Animatable accessors: about two seconds together on a warm cache.
    "tests/unit_tests/test_pt_caustics.py::",
    "tests/unit_tests/test_animatable_accessors.py::",
    # The torch floor in pyproject.toml admits every torch function algan/
    # calls: reads the two, well under a second.
    "tests/unit_tests/test_dependency_floors.py::",
    # Every algan name a benchmark script imports still exists: an AST walk of
    # benchmarks/ against the live package, about two seconds.
    "tests/unit_tests/test_benchmark_imports.py::",
    # Synthesised narration reaches its cache whole or not at all: a
    # stand-in engine, no audio.
    "tests/unit_tests/test_speech_cache.py::",
    # Where the conftest pipes a run's log, and that a collection writes none.
    "tests/unit_tests/test_conftest_logs.py::",
    # Two files the groups had no test of, which other changes now add pinned
    # tests to: the file leaves NOT_IN_GROUPS (tests/unit_tests/
    # test_test_groups.py) the moment it has one, so each has one here already.
    # The CLI's version and entry point, and the direct specular lobe on a
    # material without one and behind the surface: under a second together.
    "tests/unit_tests/test_cli_startup.py::test_version_is_read_from_metadata_not_from_the_package",
    "tests/unit_tests/test_cli_startup.py::test_the_entry_point_points_outside_the_package",
    "tests/unit_tests/test_direct_specular_lobe.py::test_the_lobe_is_off_for_a_non_pbr_material",
    "tests/unit_tests/test_direct_specular_lobe.py::test_a_back_facing_hit_does_not_explode",
    # The angle between two vectors is finite at 0 and 180 degrees, so look()
    # at a target the Mob already faces does not raise: tensors and Squares.
    "tests/unit_tests/test_rotation_between_vectors.py::",
    # Timeline regressions recorded and materialized without a render: a
    # failed replay leaves authoring state and names its updater, a spawn
    # before time zero raises, and an opacity updater ends with its Mob.
    "tests/unit_tests/test_timeline_error_paths.py::",
    "tests/unit_tests/test_updater_lifespan.py::",
    # Authoring guards with no render: assigning a name that is not a Mob
    # attribute, the wording of a missing attribute, color inputs that would
    # draw the wrong thing, and timing keywords a context cannot take.
    "tests/unit_tests/test_mob_attribute_guards.py::",
    "tests/unit_tests/test_color_input_validation.py::",
    "tests/unit_tests/test_context_keyword_errors.py::",
    # A viewer frame clears a stale error, the speech cache keys on voice and
    # rate, an unknown ray-tracing setting gets a suggestion, and `algan
    # check` follows the encoder's ffmpeg lookup: stand-ins, no render.
    "tests/unit_tests/test_viewer_and_speech_fixes.py::",
    # A render warns about a name set on a Mob that looks like a misspelt
    # Algan attribute, and about nothing the library itself set: record-only.
    "tests/unit_tests/test_unknown_mob_attribute_lint.py::",
)

# The same for regression tests too slow for the CI budget: they render, or
# compile a kernel variant nothing in the CI group does. The pre-PR group always
# carries them, as well as the whole CI group, and the greedy pass spends the
# rest of its budget around them.
ALWAYS_PRE_PR = (
    # Which resolve variant each launch specialises on, and which scenes reach it.
    "tests/unit_tests/test_sheet_resolve_variants.py::",
    # A live CameraView's windows share one arena, one prep worker and each
    # pass's batch-sizing models, and render the frames fresh passes do.
    "tests/unit_tests/test_camera_views.py::test_camera_view_windows_share_one_arena_models_and_prep_worker",
    "tests/unit_tests/test_camera_views.py::test_shared_window_resources_render_the_same_frames",
    # A tiled denoise is the whole-frame one at its seams: random weights of
    # the real shapes, an aligned and an unaligned tile size.
    "tests/unit_tests/test_denoise.py::test_a_tiled_denoise_is_the_whole_frame_denoise_at_its_seams",
    # Partial coverage resolved the way a supersampled render is.
    "tests/unit_tests/test_display_referred_coverage.py::",
    # The resolve kernel against its sequential oracle.
    "tests/unit_tests/test_sheet_resolve_parity.py::",
    # The agent skill's example scripts build their scenes: two of them
    # construct 3D Mobs or compile a stage, so they are not CI-sized.
    "tests/unit_tests/test_doc_examples.py::test_skill_example_authors_without_error",
    # Big triangles just in front of the camera plane, and the emission's
    # write-pass miss and chunk cap.
    "tests/unit_tests/test_raster_lattice_bound.py::",
    "tests/unit_tests/test_near_camera_plane_render.py::",
    "tests/unit_tests/test_emission_write_miss.py::",
    # The path tracer's clearcoat attenuates the sheen under it.
    "tests/unit_tests/test_pt_clearcoat_sheen.py::test_the_clearcoat_attenuates_the_sheen_under_it",
    # The resolve's shading against renders: the glass rim, the refracting
    # variant against the oracle and the solids its ceiling must leave alone,
    # a NaN area, the byte buffer under a bright sky, partial-coverage tint,
    # and rough glass with and without the prefilter.
    "tests/unit_tests/test_sheet_resolve_shading.py::",
    "tests/unit_tests/test_glossy_prefilter.py::test_rough_glass_reflects_the_same_with_and_without_the_prefilter",
    # An opaque metal keeps its direct-light highlight when the render bounces.
    "tests/unit_tests/test_direct_specular_lobe.py::test_an_opaque_metal_keeps_its_highlight_when_the_render_has_bounces",
    # The split-sum weight against a brute-force integral of the direct lobe
    # over a grid (three seconds of quadrature).
    "tests/unit_tests/test_glossy_prefilter.py::test_env_brdf_is_the_albedo_of_the_direct_lobe",
    # A part metal is the mix of its dielectric and its metal: the diffuse
    # layer loses the dielectric Fresnel share only. Pure torch, milliseconds.
    "tests/unit_tests/test_materials.py::test_a_part_metal_is_the_mix_of_the_dielectric_and_the_metal",
    # The closed-shell ceiling's torch arm and its segment order keep their
    # integer gathers exact past MPS's 2**24: a real frame's fragments in a 4K
    # window, and a float32 spend over 2**24 rows.
    "tests/unit_tests/test_sheet_chain_kernels.py::test_the_shell_ceiling_keeps_its_segments_past_the_mps_gather_ceiling",
    "tests/unit_tests/test_sheet_chain_kernels.py::test_the_float32_segment_prefix_finds_its_rows_past_the_mps_gather_ceiling",
    # The compaction keeps pixel ids exact past MPS's 2**24: a real frame's
    # fragments in a chunk's third 4K frame, and the reference depth gate's
    # sample keys inside one 4K frame.
    "tests/unit_tests/test_sheet_chain_kernels.py::test_the_compaction_keeps_neighbouring_pixels_apart_past_the_mps_gather_ceiling",
    "tests/unit_tests/test_sheet_chain_kernels.py::test_the_reference_depth_gate_keeps_its_samples_apart_in_a_4k_frame",
    # Mob guards that build a Sphere or a Cube, whose first construction
    # compiles: a set() only another Mob could take, shader setup under a
    # spawned member, and a geometry flag changed after spawn.
    "tests/unit_tests/test_mob_set_guards.py::test_set_refuses_an_attribute_only_another_mob_has",
    "tests/unit_tests/test_materials.py::test_shader_setup_on_unspawned_group_with_spawned_members_raises",
    "tests/unit_tests/test_geometry_declarations.py::",
    # More integer gathers MPS rounds past 2**24, where a test takes a render
    # or a header past 2**24 words: four squares drawn as circuits 65,536 on,
    # a near cut's mask word, the Bezier header's bin offsets, and the
    # polylines' repeated run starts.
    "tests/unit_tests/test_mps_fragment_refs.py::test_circuits_past_the_65536th_render_as_they_do_with_small_ids",
    "tests/unit_tests/test_sheet_route_former_vetoes.py::test_a_near_cut_mask_word_survives_the_mps_integer_gather",
    "tests/unit_tests/test_bezier_header_mps_gather.py::test_bin_offsets_past_the_ceiling_survive_the_mps_integer_gather",
    "tests/unit_tests/test_bezier_polyline_mps_repeat.py::test_frame_local_edges_keep_their_slots_past_the_ceiling",
    "tests/unit_tests/test_bezier_polyline_mps_repeat.py::test_the_vertex_parameters_survive_the_mps_repeat_interleave",
    "tests/unit_tests/test_bezier_polyline_mps_repeat.py::test_the_cache_places_every_edge_past_the_ceiling",
    # A few seconds each: a DecimalNumber says when its 32-bit value shows
    # other digits than it was given, a truncated tessellation entry is
    # rebuilt, and a formula that never finishes is stopped (a real LaTeX run).
    "tests/unit_tests/test_numeric_display.py::test_a_value_float32_cannot_show_warns_with_the_digits_drawn",
    "tests/unit_tests/test_numeric_display.py::test_a_value_float32_shows_exactly_does_not_warn",
    "tests/unit_tests/test_tessellation_cache.py::test_a_truncated_entry_is_re_tessellated",
    "tests/unit_tests/test_typeset_cache.py::test_a_formula_that_never_finishes_is_stopped",
    # And the ids the scene build gathers or repeats past 2**24, each test in
    # a batch past the ceiling: a member's surface, a circuit's edges, gaps,
    # bounds and z bias, a promoted group's triangle and a sampled vertex.
    "tests/unit_tests/test_ids_mps_gathers.py::",
    # A tube stretched through zero length keeps its round cross-section: it
    # builds a Line3D, whose geometry search is what takes the time.
    "tests/unit_tests/test_3d_shape_ends.py::test_a_line_through_zero_length_keeps_its_round_cross_section",
    # A point cloud built point by point packs its spheres once, and a
    # spawned one still rebuilds at the edit: Dot3D's geometry search is the
    # cost.
    "tests/unit_tests/test_point_cloud_api.py::test_building_a_cloud_point_by_point_packs_the_spheres_once",
    "tests/unit_tests/test_point_cloud_api.py::test_an_edit_to_a_spawned_cloud_still_rebuilds_at_once",
    # The zero-copy cache evicts an owner the garbage collector frees while its
    # lock is held, instead of deadlocking: a subprocess, with a timeout.
    "tests/unit_tests/test_mps_zero_copy.py::test_an_owner_freed_while_the_cache_lock_is_held_is_evicted_without_deadlock",
    # The glossy pyramid's fused finish sums each texel in its launches' order.
    "tests/unit_tests/test_glossy_fused_finish.py::test_the_fused_finish_matches_the_launches_it_replaces",
    # The rest of the environment audit: how each variable parses, and the
    # subprocess that checks an unknown ALGAN_ name warns at import. About
    # eight seconds for the file.
    "tests/unit_tests/test_environment.py::",
    # Three more files added after the last measurement (see ALWAYS_CI), too
    # slow to pin whole: a representative subset of each, about 34 seconds
    # together on a warm cache against about 370 for the three files whole.
    # Caustics rendered through a glass plate against geometric optics, and
    # straight subsurface shadows under delta lights (about 14 s of 151 s).
    "tests/unit_tests/test_pt_caustics_render.py::test_a_point_light_through_a_glass_plate_lights_the_floor_as_optics_says",
    "tests/unit_tests/test_pt_caustics_render.py::test_subsurface_delta_lights_leave_glass_shadows_straight",
    # Volumes: the density field, the majorant and the delta-tracking
    # estimators against closed forms, and one render through an absorbing
    # medium; not the scene renders (about 11 s of 124 s).
    "tests/unit_tests/test_pt_volumes.py::test_majorant_bounds_the_interpolated_density_everywhere",
    "tests/unit_tests/test_pt_volumes.py::test_absorbing_density_dims_the_environment_by_its_optical_depth",
    "tests/unit_tests/test_pt_volumes.py::test_spectral_tracking_",
    "tests/unit_tests/test_pt_volumes.py::test_ratio_tracking_matches_the_optical_depth_per_channel",
    "tests/unit_tests/test_pt_volumes.py::test_the_box_clips_the_segment_and_the_outside_is_vacuum",
    "tests/unit_tests/test_pt_volumes.py::test_each_frame_reads_its_own_transform",
    "tests/unit_tests/test_pt_volumes.py::test_kernel_density_is_cell_centred_trilinear_clamped_to_the_edges",
    "tests/unit_tests/test_pt_volumes.py::test_a_segment_that_exhausts_the_step_budget_is_absorbed_and_flagged",
    # Shared Bezier acceleration bins: the cheap checks whole, and one case of
    # each moving fraction of the 72-case sweep (about 9 s of 99 s).
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_unchanged_means_the_same_bits",
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_one_frame_and_no_edges_match_the_previous_buffer",
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_a_fixed_circuit_reads_its_frame_zero_bins_in_every_frame",
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_the_disabled_arm_is_the_previous_buffer[0]",
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_shared_bins_answer_every_query_as_before[0-None-40]",
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_shared_bins_answer_every_query_as_before[0-0.3-1]",
    "tests/unit_tests/test_bezier_acceleration_dedup.py::test_shared_bins_answer_every_query_as_before[0-1.0-4194304]",
)

_DURATION_LINE = re.compile(r"^\s*([\d.]+)s (setup|call|teardown)\s+(\S.*?)\s*$")

HEADER = """\
# The {name} test group: `pytest --group {name}`. Generated by
# scripts/select_test_groups.py -- regenerate rather than hand-edit, except to
# drop a test that no longer exists. `--group` also runs every test the script
# pins, listed here or not. See tests/README.md, "Test groups".
#
# {count} tests, {seconds:.0f} s of estimated runner time (measured duration
# plus per-test overhead, -n 1 on ubuntu-latest), covering {lines} of the
# {total} lines of algan/ the whole measured run covers ({share:.0%}).
"""


def read_durations(path: Path) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _DURATION_LINE.match(line)
        if m:
            total[m.group(3)] += float(m.group(1))
    return dict(total)


def read_coverage(path: Path) -> dict[str, set[tuple[str, int]]]:
    """Lines of algan/ each test executed, keyed by node id."""
    from coverage import CoverageData

    data = CoverageData(basename=str(path))
    data.read()
    by_test: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for filename in data.measured_files():
        path = Path(filename)
        key = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        for lineno, contexts in data.contexts_by_lineno(filename).items():
            for context in contexts:
                # pytest-cov names contexts "<nodeid>|setup", "|run", "|teardown".
                nodeid = context.rsplit("|", 1)[0]
                if nodeid:
                    by_test[nodeid].add((key, lineno))
    return dict(by_test)


def collect(*extra: str) -> set[str]:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            *extra,
            "-p",
            "no:cacheprovider",
            "tests/unit_tests",
            "tests/fast",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if "::" in line}


def greedy(seed, candidates, durations, coverage, budget):
    chosen = list(dict.fromkeys(t for t in seed if t in durations))
    covered = (
        set().union(*(coverage.get(t, set()) for t in chosen)) if chosen else set()
    )
    spent = sum(durations[t] for t in chosen)
    pool = {t for t in candidates if t not in set(chosen)}
    # Lazy greedy: a test's gain only shrinks as coverage grows, so a stale
    # score is an upper bound and only the leader needs recomputing.
    import heapq

    heap = []
    for t in pool:
        gain = len(coverage.get(t, set()) - covered)
        if gain:
            heap.append((-gain / max(durations[t], 0.05), t))
    heapq.heapify(heap)
    while heap:
        _, t = heapq.heappop(heap)
        cost = durations[t]
        if spent + cost > budget:
            continue
        gain = len(coverage.get(t, set()) - covered)
        if not gain:
            continue
        score = -gain / max(cost, 0.05)
        if heap and score > heap[0][0]:
            heapq.heappush(heap, (score, t))
            continue
        chosen.append(t)
        covered |= coverage[t]
        spent += cost
    return chosen, covered, spent


def write_group(name, tests, covered, spent, total_lines):
    GROUPS.mkdir(exist_ok=True)
    header = HEADER.format(
        name=name,
        count=len(tests),
        seconds=spent,
        lines=len(covered),
        total=total_lines,
        share=len(covered) / max(total_lines, 1),
    )
    (GROUPS / f"{name}.txt").write_text(header + "\n" + "\n".join(sorted(tests)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--durations", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--ci-budget", type=float, default=DEFAULT_BUDGETS["ci"])
    parser.add_argument(
        "--pre-pr-budget", type=float, default=DEFAULT_BUDGETS["pre-pr"]
    )
    parser.add_argument(
        "--per-test-overhead",
        type=float,
        default=PER_TEST_OVERHEAD,
        help="Seconds each test costs on top of its reported duration",
    )
    args = parser.parse_args()

    # pytest's --durations=0 hides every phase under 5 ms, so a test that is
    # missing from the report is one of the cheapest there are.
    collected = collect()
    measured = read_durations(args.durations)
    durations = {t: measured.get(t, 0.005) + args.per_test_overhead for t in collected}
    coverage = read_coverage(args.coverage)
    candidates = set(durations)
    total = set().union(*coverage.values())
    # A pin that matches nothing names a test that was renamed or deleted.
    stale = [
        p
        for p in ALWAYS_CI + ALWAYS_PRE_PR
        if not any(t.startswith(p) for t in durations)
    ]
    if stale:
        parser.error(f"ALWAYS_CI / ALWAYS_PRE_PR name no collected test: {stale}")
    guards = sorted(t for t in durations if t.startswith(ALWAYS_CI))
    seed = sorted(collect("--fast")) + guards
    pre_guards = sorted(t for t in durations if t.startswith(ALWAYS_PRE_PR))

    ci, ci_cov, ci_spent = greedy(seed, candidates, durations, coverage, args.ci_budget)
    pre, pre_cov, pre_spent = greedy(
        ci + pre_guards, candidates, durations, coverage, args.pre_pr_budget
    )
    write_group("ci", ci, ci_cov, ci_spent, len(total))
    write_group("pre-pr", pre, pre_cov, pre_spent, len(total))
    full = sum(durations.values())
    print(f"whole run: {len(durations)} tests, {full:.0f} s, {len(total)} lines")
    for name, tests, cov, spent in (
        ("ci", ci, ci_cov, ci_spent),
        ("pre-pr", pre, pre_cov, pre_spent),
    ):
        print(
            f"{name}: {len(tests)} tests, {spent:.0f} s, {len(cov)} lines ({len(cov) / len(total):.0%})"
        )


if __name__ == "__main__":
    main()
