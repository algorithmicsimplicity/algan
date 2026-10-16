import fs from 'node:fs';

export const SCENE_DEFAULTS = JSON.parse(fs.readFileSync(new URL('./scene_defaults.json', import.meta.url), 'utf8'));

// The JavaScript half of the scene contract; `_normalize_spec` in
// algan_render.py is the Python half, and the two must agree field for field,
// error message included (tests/unit_tests/test_renderer_audit_contract.py).
// A key starting with '_' is an annotation (JSON has no comments) and passes
// through anywhere; any other key a section does not define is rejected, so a
// typo cannot silently fall back to a default in one engine.

function checkFields(where, given, allowed, required = []) {
  const unknown = Object.keys(given).filter((key) => !allowed.includes(key) && !key.startsWith('_')).sort();
  if (unknown.length) throw new Error(`Unsupported ${where} fields: ${unknown.join(', ')}`);
  const missing = required.filter((key) => !(key in given));
  if (missing.length) throw new Error(`Missing ${where} fields: ${missing.join(', ')}`);
}

function normalizeLight(light, index) {
  const where = `lights[${index}]`;
  const kind = light.type;
  if (typeof kind !== 'string' || !Object.hasOwn(SCENE_DEFAULTS.light, kind)) {
    throw new Error(`Unsupported ${where} type: ${JSON.stringify(kind ?? null)}`);
  }
  const defaults = SCENE_DEFAULTS.light[kind];
  const required = SCENE_DEFAULTS.light_required[kind] || [];
  const optional = SCENE_DEFAULTS.light_optional[kind] || [];
  checkFields(where, light, ['type', ...Object.keys(defaults), ...required, ...optional], required);
  if (kind === 'directional' && ('direction' in light) === ('position' in light)) {
    throw new Error(`${where}: a directional light takes exactly one form: 'direction', or 'position' + 'target'`);
  }
  return { ...defaults, ...light };
}

function normalizeObject(object, index) {
  const where = `objects[${index}]`;
  checkFields(where, object, ['geometry', 'material', ...Object.keys(SCENE_DEFAULTS.object)], ['geometry']);
  const geometry = object.geometry;
  const kind = geometry.type;
  if (typeof kind !== 'string' || !Object.hasOwn(SCENE_DEFAULTS.geometry, kind)) {
    throw new Error(`Unsupported ${where}.geometry type: ${JSON.stringify(kind ?? null)}`);
  }
  const geometryDefaults = SCENE_DEFAULTS.geometry[kind];
  const geometryRequired = SCENE_DEFAULTS.geometry_required[kind] || [];
  checkFields(`${where}.geometry`, geometry, ['type', ...Object.keys(geometryDefaults), ...geometryRequired], geometryRequired);
  checkFields(`${where}.material`, object.material || {}, Object.keys(SCENE_DEFAULTS.material));
  const material = { ...SCENE_DEFAULTS.material, ...(object.material || {}) };
  if (!SCENE_DEFAULTS.material_types.includes(material.type)) {
    throw new Error(`Unsupported ${where}.material type: ${JSON.stringify(material.type)}`);
  }
  return { ...SCENE_DEFAULTS.object, ...object, geometry: { ...geometryDefaults, ...geometry }, material };
}

export function normalizeSpec(spec) {
  checkFields('scene', spec, ['name', 'render', 'camera', 'lights', 'objects']);
  checkFields('render', spec.render || {}, Object.keys(SCENE_DEFAULTS.render));
  checkFields('camera', spec.camera || {}, Object.keys(SCENE_DEFAULTS.camera));
  return {
    ...spec,
    render: { ...SCENE_DEFAULTS.render, ...(spec.render || {}) },
    camera: { ...SCENE_DEFAULTS.camera, ...(spec.camera || {}) },
    lights: (spec.lights || []).map(normalizeLight),
    objects: (spec.objects || []).map(normalizeObject),
  };
}
