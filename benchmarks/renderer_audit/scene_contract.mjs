import fs from 'node:fs';

export const SCENE_DEFAULTS = JSON.parse(fs.readFileSync(new URL('./scene_defaults.json', import.meta.url), 'utf8'));

export function normalizeSpec(spec) {
  const unknown = Object.keys(spec.camera || {}).filter((key) => !(key in SCENE_DEFAULTS.camera));
  if (unknown.length) throw new Error(`Unsupported camera fields: ${unknown.join(', ')}`);
  return {
    ...spec,
    render: {...SCENE_DEFAULTS.render, ...(spec.render || {})},
    camera: {...SCENE_DEFAULTS.camera, ...(spec.camera || {})},
    objects: (spec.objects || []).map((object) => ({
      ...object, material: {...SCENE_DEFAULTS.material, ...(object.material || {})},
    })),
  };
}
