// three_render.mjs — Three.js reference renderer for renderer-audit scene specs.
//
// Renders one scene-spec JSON (see SPEC.md) to PNG with two independent back ends:
//   * "raster"     — THREE.WebGLRenderer, ordinary out-of-the-box Three.js
//   * "pathtrace"  — three-gpu-pathtracer (WebGLPathTracer), physically based ground truth
//
// Usage:
//   node three_render.mjs <scene.json> --out <dir> [--mode raster|pathtrace|both] [--samples N]
//                          [--gl swiftshader|hardware] [--tiles N]
//
// Prints a one-line JSON summary to stdout: {"scene":...,"outputs":[...],"samples":...,"seconds":...}
// All diagnostics go to stderr.
//
// DEPENDENCIES: this script needs an npm project containing node_modules with
//   three, three-gpu-pathtracer (+ its peer deps three-mesh-bvh, xatlas-web) and playwright.
//   Recreate it anywhere with:  npm install three three-gpu-pathtracer playwright
//   The directory is resolved from (first hit wins):
//     1. --node-modules <dir> CLI flag
//     2. $AUDIT_THREE_NODE_MODULES  (deliberately NOT ALGAN_-prefixed: Algan warns about
//        any ALGAN_ variable it does not itself honour, and this one is the audit's)
//     3. ./node_modules next to this script
//     4. the scratch project this tool was developed against (path below)
//   Chromium is found via PLAYWRIGHT_BROWSERS_PATH; if Playwright cannot resolve a
//   browser binary (revision mismatch) we fall back to globbing /opt/pw-browsers/chromium-*/.
//   WebGL2 backend: --gl swiftshader (default) launches Chromium's software rasterizer via
//     --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader
//     --no-sandbox --disable-dev-shm-usage --headless=new
//   which is the only option on a machine without a GPU (the session this tool was written
//   in). --gl hardware asks ANGLE for the real device instead (d3d11 on Windows, gl
//   elsewhere); it is tens of times faster for the path tracer and is what makes a
//   64-sample pass of every scene practical. The images agree -- see REPORT.md section 1.
//
// CONVENTIONS PINNED FOR THE AUDIT (do not change these to make images match):
//   * Colour management: Three.js defaults — THREE.ColorManagement.enabled = true,
//     renderer.outputColorSpace = THREE.SRGBColorSpace. Spec colours are authored sRGB
//     values and are fed to THREE.Color via setRGB(r,g,b,THREE.SRGBColorSpace), so spec
//     [0.8,0.8,0.8] means "the sRGB value 0.8". The only transfer function in play is
//     the sRGB OETF at output.
//   * renderer.toneMapping = THREE.NoToneMapping in both modes.
//   * Lights: directional -> DirectionalLight(position = -direction*50, target = origin);
//     point -> PointLight(color, intensity, distance, decay); ambient -> AmbientLight.
//     Installed three is r185 (0.185.1): since r155 punctual-light intensities are
//     physical (no legacy PI scaling; useLegacyLights was removed in r165).
//     PointLight intensity is in candela and the shader attenuation is
//     1/max(pow(d,decay),0.01), windowed by pow2(saturate(1-pow4(d/distance))) when
//     distance>0 — so decay=0 & distance=0 is literally attenuation 1 everywhere.
//     DirectionalLight irradiance = color*intensity*dotNL (no PI factor either).
//   * SpotLight: spec `angle` is a half-angle in DEGREES, three wants radians;
//     spec `decay`/`distance` default to 0 (no falloff), overriding three's own
//     constructor defaults (angle pi/3, decay 2). castShadow = true.
//   * RectAreaLight: positioned and lookAt(target); three renders it BLACK
//     until RectAreaLightUniformsLib.init() runs once before the first render
//     (imported via the page's import map below). It cannot cast shadows here.
//   * HemisphereLight: HemisphereLight(skyColor, groundColor, intensity); no
//     shadows either side.

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FALLBACK_NODE_MODULES =
  '/tmp/claude-0/-home-user-algan/51960e50-b094-5117-954b-e7b85c715502/scratchpad/three/node_modules';

function parseArgs(argv) {
  const args = { mode: 'both', samples: 64, gl: 'swiftshader', tiles: null };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') args.out = argv[++i];
    else if (a === '--mode') args.mode = argv[++i];
    else if (a === '--samples') args.samples = Number(argv[++i]);
    else if (a === '--node-modules') args.nodeModules = argv[++i];
    else if (a === '--gl') args.gl = argv[++i];
    else if (a === '--tiles') args.tiles = Number(argv[++i]);
    else rest.push(a);
  }
  if (!rest[0]) die('usage: node three_render.mjs <scene.json> --out <dir> [--mode raster|pathtrace|both] [--samples N] [--gl swiftshader|hardware]');
  args.scenePath = path.resolve(rest[0]);
  if (!['raster', 'pathtrace', 'both'].includes(args.mode)) die(`bad --mode ${args.mode}`);
  if (!['swiftshader', 'hardware'].includes(args.gl)) die(`bad --gl ${args.gl}`);
  // Default per backend: a hardware device needs the frame split or a single
  // over-long draw trips the display driver's watchdog (see renderPathTrace);
  // SwiftShader has no watchdog and every extra tile is pure per-draw overhead.
  if (args.tiles === null) args.tiles = args.gl === 'hardware' ? 4 : 1;
  if (!Number.isInteger(args.tiles) || args.tiles < 1) die('bad --tiles');
  if (!Number.isFinite(args.samples) || args.samples < 1) die('bad --samples');
  return args;
}

// Chromium switches selecting the WebGL2 implementation. 'swiftshader' is the software
// path and works anywhere; 'hardware' asks ANGLE for the real device -- d3d11 on Windows,
// the platform GL driver elsewhere -- which is what makes a 64-sample path-traced pass of
// every scene finish in minutes rather than days.
function glArgs(kind) {
  const common = ['--no-sandbox', '--disable-dev-shm-usage'];
  if (kind === 'hardware') {
    const backend = process.platform === 'win32' ? 'd3d11' : 'gl';
    return [
      '--use-gl=angle', `--use-angle=${backend}`,
      '--enable-gpu', '--ignore-gpu-blocklist',
      // Prefer the discrete GPU where the machine has both; ignored otherwise.
      '--force_high_performance_gpu',
      ...common,
    ];
  }
  return [
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
    ...common, '--headless=new',
  ];
}

function die(msg) {
  console.error(msg);
  process.exit(1);
}

function resolveNodeModules(flagValue) {
  const candidates = [
    flagValue,
    process.env.AUDIT_THREE_NODE_MODULES,
    path.join(__dirname, 'node_modules'),
    FALLBACK_NODE_MODULES,
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'three', 'build', 'three.module.js'))) return c;
  }
  die(`no usable node_modules found (tried:\n  ${candidates.join('\n  ')}\n)` +
      '\nrecreate with: npm install three three-gpu-pathtracer playwright');
}

function findChromiumExecutable() {
  // Playwright's pinned revision may not match what is installed; fall back to any
  // full chromium binary under /opt/pw-browsers (highest revision wins).
  const roots = ['/opt/pw-browsers'];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    const dirs = fs.readdirSync(root)
      .filter(d => d.startsWith('chromium-'))
      .sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]));
    for (const d of dirs) {
      const exe = path.join(root, d, 'chrome-linux', 'chrome');
      if (fs.existsSync(exe)) return exe;
    }
  }
  return undefined; // let playwright resolve normally
}

function startServer(nodeModulesDir) {
  const MIME = {
    '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
    '.wasm': 'application/wasm', '.map': 'application/json',
  };
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://127.0.0.1');
    if (process.env.ALGAN_THREE_DEBUG) console.error('[srv]', req.url);
    if (url.pathname === '/' || url.pathname === '/index.html') {
      const html = pageHtml(server.address().port);
      if (process.env.ALGAN_THREE_DEBUG) fs.writeFileSync('/tmp/opencode/three_page.html', html);
      res.writeHead(200, { 'content-type': 'text/html' });
      res.end(html);
      return;
    }
    if (url.pathname.startsWith('/node_modules/')) {
      const rel = decodeURIComponent(url.pathname.slice('/node_modules/'.length));
      const file = path.resolve(nodeModulesDir, rel);
      if (!file.startsWith(path.resolve(nodeModulesDir)) || !fs.existsSync(file)) {
        res.writeHead(404); res.end('not found'); return;
      }
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
      fs.createReadStream(file).pipe(res);
      return;
    }
    res.writeHead(404); res.end('not found');
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

function pageHtml(port) {
  const base = `http://127.0.0.1:${port}/node_modules`;
  const imports = {
    'three': `${base}/three/build/three.module.js`,
    'three/addons/': `${base}/three/examples/jsm/`,
    'three/examples/jsm/': `${base}/three/examples/jsm/`,
    'three-mesh-bvh': `${base}/three-mesh-bvh/build/index.module.js`,
    'xatlas-web': `${base}/xatlas-web/dist/xatlas-web.js`,
    // three-gpu-pathtracer's own entry points. Its package `module` field is
    // `src/index.js`, and its internals import each other by relative path, so
    // the bare specifier and the `src/` prefix both have to resolve.
    'three-gpu-pathtracer': `${base}/three-gpu-pathtracer/src/index.js`,
    'three-gpu-pathtracer/': `${base}/three-gpu-pathtracer/`,
  };
  return `<!doctype html><html><head><meta charset="utf-8">
<script type="importmap">${JSON.stringify({ imports })}</script>
</head><body>
<script type="module">
import * as THREE from 'three';
import { WebGLPathTracer } from 'three-gpu-pathtracer';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';

// A RectAreaLight renders BLACK until the LTC lookup tables are built; this
// must run once before the first render (see SPEC.md, "rect_area").
RectAreaLightUniformsLib.init();

const col = (c) => new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);

function buildMaterial(m) {
  // Spec defaults (SPEC.md): every field optional, these are the defaults both back ends apply.
  const d = {
    color: [0.5, 0.5, 0.54], roughness: 0.85, metalness: 0.0, ior: 1.5, transmission: 0.0,
    clearcoat: 0.0, clearcoat_roughness: 0.0, sheen: 0.0, sheen_roughness: 1.0,
    sheen_color: [0, 0, 0], emissive: [0, 0, 0], emissive_intensity: 1.0,
    specular_intensity: 1.0, specular_color: [1, 1, 1], opacity: 1.0,
    attenuation_color: [1, 1, 1], attenuation_distance: 0,
    // phong / toon / depth fields (SPEC.md). phong's specular default is
    // three's own 0x111111; depth's near/far exist only on the Algan side --
    // three derives depth from the camera and takes no such fields.
    specular: [0.067, 0.067, 0.067], shininess: 30, bands: 3, near: 0.1, far: 100,
  };
  const p = { ...d, ...(m || {}) };
  let mat;
  if (p.type === 'physical') mat = new THREE.MeshPhysicalMaterial();
  else if (p.type === 'basic') mat = new THREE.MeshBasicMaterial();
  else if (p.type === 'lambert') mat = new THREE.MeshLambertMaterial();
  else if (p.type === 'phong') mat = new THREE.MeshPhongMaterial();
  else if (p.type === 'toon') mat = new THREE.MeshToonMaterial();
  else if (p.type === 'normal') mat = new THREE.MeshNormalMaterial();
  else if (p.type === 'matcap') mat = new THREE.MeshMatcapMaterial();
  else if (p.type === 'depth') mat = new THREE.MeshDepthMaterial();
  else mat = new THREE.MeshStandardMaterial();
  // normal/depth discard the base colour in-shader on BOTH engines, so it is
  // not set here either.
  if (p.type !== 'normal' && p.type !== 'depth') {
    mat.color.copy(col(p.color));
  }
  const hasEmissive =
    p.type === 'lambert' || p.type === 'phong' || p.type === 'toon' ||
    p.type === 'standard' || p.type === 'physical';
  if (hasEmissive) {
    mat.emissive.copy(col(p.emissive));
    mat.emissiveIntensity = p.emissive_intensity;
  }
  if (p.type === 'standard' || p.type === 'physical') {
    mat.roughness = p.roughness;
    mat.metalness = p.metalness;
  }
  if (p.type === 'phong') {
    mat.specular.copy(col(p.specular));
    mat.shininess = p.shininess;
  }
  if (p.type === 'toon' && m && m.bands !== undefined) {
    // SPEC.md: the documented three.js way to get N toon bands is a
    // gradientMap -- a DataTexture of N texels ramping 0..1 with NearestFilter
    // on both min and mag, so the ramp quantises into N steps. Algan instead
    // quantises dotNL directly (its 'bands' argument); without 'bands' three
    // keeps its own default toon (a 2-step smoothstep at 0.7).
    const n = Math.max(1, Math.round(p.bands));
    const data = new Uint8Array(n);
    for (let i = 0; i < n; i++) data[i] = Math.round((i / Math.max(n - 1, 1)) * 255);
    const gradientMap = new THREE.DataTexture(data, n, 1, THREE.RedFormat);
    gradientMap.minFilter = THREE.NearestFilter;
    gradientMap.magFilter = THREE.NearestFilter;
    gradientMap.needsUpdate = true;
    mat.gradientMap = gradientMap;
  }
  // depth: deliberately nothing beyond construction -- three.js's
  // MeshDepthMaterial takes no near/far (it uses the camera's near/far and
  // writes non-linear gl_FragCoord.z). SPEC.md records this panel as
  // informational rather than a parity test; we do not hack a custom shader
  // to force agreement.
  if (p.type === 'matcap') {
    // No matcap texture is sampled on either engine (SPEC.md): with none
    // assigned, three's shader substitutes its built-in default
    // vec4(vec3(mix(0.2, 0.8, uv.y)), 1.0) and multiplies the base colour in.
  }
  if (p.type === 'physical') {
    mat.ior = p.ior;
    mat.transmission = p.transmission;
    mat.clearcoat = p.clearcoat;
    mat.clearcoatRoughness = p.clearcoat_roughness;
    mat.sheen = p.sheen;
    mat.sheenRoughness = p.sheen_roughness;
    mat.sheenColor.copy(col(p.sheen_color));
    mat.specularIntensity = p.specular_intensity;
    mat.specularColor.copy(col(p.specular_color));
    mat.attenuationColor.copy(col(p.attenuation_color));
    // Spec default attenuation_distance is 0; three.js means "no attenuation" by Infinity
    // and would divide by zero at 0, so <= 0 maps to Infinity (decision recorded in notes).
    mat.attenuationDistance = p.attenuation_distance > 0 ? p.attenuation_distance : Infinity;
    // thickness is NOT in the spec but transmission without it samples the backdrop at the
    // un-refracted position (thin-film look, no inversion). A competent user modelling a
    // solid glass object sets it to the volume depth; we use twice the geometry bounding-
    // sphere radius (= full diameter for spheres). Decision recorded in OX_THREE_NOTES.md.
    mat.thickness = p.thickness !== undefined ? p.thickness : undefined;
  }
  if (p.opacity < 1.0) { mat.opacity = p.opacity; mat.transparent = true; }
  return mat;
}

function buildScene(spec) {
  const scene = new THREE.Scene();
  scene.background = col(spec.render.background);
  for (const l of spec.lights || []) {
    if (l.type === 'directional') {
      // Exactly one form required (SPEC.md): 'direction' (pointing from the
      // light toward the scene; position becomes -direction*50) or
      // 'position' + 'target'.
      if ((l.direction !== undefined) === (l.position !== undefined)) {
        throw new Error("directional light takes exactly one form: 'direction', or 'position' + 'target'");
      }
      const dl = new THREE.DirectionalLight(col(l.color), l.intensity);
      if (l.direction !== undefined) {
        dl.position.set(-l.direction[0], -l.direction[1], -l.direction[2]).multiplyScalar(50);
        dl.target.position.set(0, 0, 0);
      } else {
        dl.position.set(l.position[0], l.position[1], l.position[2]);
        dl.target.position.set(l.target[0], l.target[1], l.target[2]);
      }
      dl.castShadow = true;
      dl.shadow.mapSize.set(2048, 2048);
      dl.shadow.camera.left = -30; dl.shadow.camera.right = 30;
      dl.shadow.camera.top = 30; dl.shadow.camera.bottom = -30;
      dl.shadow.camera.near = 0.5; dl.shadow.camera.far = 200;
      dl.shadow.camera.updateProjectionMatrix();
      scene.add(dl); scene.add(dl.target);
    } else if (l.type === 'point') {
      const pl = new THREE.PointLight(col(l.color), l.intensity, l.distance ?? 0, l.decay ?? 2);
      pl.position.set(l.position[0], l.position[1], l.position[2]);
      pl.castShadow = true;
      pl.shadow.mapSize.set(1024, 1024);
      pl.shadow.camera.near = 0.5; pl.shadow.camera.far = 200;
      scene.add(pl);
    } else if (l.type === 'spot') {
      // Spec 'angle' is a half-angle in DEGREES (three wants radians); spec
      // decay/distance default to 0 -- passed explicitly because three's own
      // constructor defaults are angle pi/3 and decay 2.
      const sl = new THREE.SpotLight(
        col(l.color), l.intensity,
        l.distance ?? 0,
        THREE.MathUtils.degToRad(l.angle ?? 30),
        l.penumbra ?? 0,
        l.decay ?? 0,
      );
      sl.position.set(l.position[0], l.position[1], l.position[2]);
      sl.target.position.set(l.target[0], l.target[1], l.target[2]);
      sl.castShadow = true;
      sl.shadow.mapSize.set(1024, 1024);
      sl.shadow.camera.near = 0.5; sl.shadow.camera.far = 200;
      scene.add(sl); scene.add(sl.target);
    } else if (l.type === 'rect_area') {
      // RectAreaLightUniformsLib.init() ran once at page load; without it this
      // light renders black. It cannot cast shadows in three (SPEC.md).
      const rl = new THREE.RectAreaLight(col(l.color), l.intensity, l.width, l.height);
      rl.position.set(l.position[0], l.position[1], l.position[2]);
      scene.add(rl);
      rl.lookAt(new THREE.Vector3(l.target[0], l.target[1], l.target[2]));
    } else if (l.type === 'hemisphere') {
      // No shadow support on either engine (SPEC.md).
      const hl = new THREE.HemisphereLight(col(l.color), col(l.ground_color), l.intensity);
      scene.add(hl);
    } else if (l.type === 'ambient') {
      scene.add(new THREE.AmbientLight(col(l.color), l.intensity));
    } else {
      throw new Error('unknown light type ' + l.type);
    }
  }
  for (const o of spec.objects || []) {
    let geo;
    if (o.geometry.type === 'sphere') {
      const seg = o.geometry.segments ?? 64;
      geo = new THREE.SphereGeometry(o.geometry.radius, seg, Math.floor(seg / 2));
    } else if (o.geometry.type === 'box') {
      geo = new THREE.BoxGeometry(o.geometry.size[0], o.geometry.size[1], o.geometry.size[2]);
    } else {
      throw new Error('unknown geometry type ' + o.geometry.type);
    }
    const mesh = new THREE.Mesh(geo, buildMaterial(o.material));
    mesh.name = o.name || '';
    mesh.position.set(o.position[0], o.position[1], o.position[2]);
    mesh.rotation.y = THREE.MathUtils.degToRad(o.rotation_y || 0);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (mesh.material.isMeshPhysicalMaterial && mesh.material.thickness === undefined) {
      geo.computeBoundingSphere();
      mesh.material.thickness = geo.boundingSphere.radius * 2;
    }
    scene.add(mesh);
  }
  const cam = new THREE.PerspectiveCamera(
    spec.camera.fov, spec.render.width / spec.render.height, spec.camera.near, spec.camera.far);
  cam.up.set(spec.camera.up[0], spec.camera.up[1], spec.camera.up[2]);
  cam.position.set(spec.camera.position[0], spec.camera.position[1], spec.camera.position[2]);
  cam.lookAt(new THREE.Vector3(...spec.camera.target));
  return { scene, camera: cam };
}

function makeRenderer(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  document.body.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(1);
  // Defaults, stated explicitly for the audit (see header comment).
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  return renderer;
}

async function renderRaster(spec) {
  const t0 = performance.now();
  const renderer = makeRenderer(spec.render.width, spec.render.height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const { scene, camera } = buildScene(spec);
  renderer.render(scene, camera);
  const dataUrl = renderer.domElement.toDataURL('image/png');
  renderer.dispose();
  return { dataUrl, ms: performance.now() - t0 };
}

// three-gpu-pathtracer's material table reads material.color.r unguarded
// (MaterialsTexture.js:193 -- every other field goes through getField with a
// default), and MeshNormalMaterial and MeshDepthMaterial have no color: they
// are not surface descriptions. Without this the whole pass throws before the
// first sample, taking the scene's other ten objects with it.
//
// So the tracer is given a colour for those two, and ONLY those two, so the
// rest of the frame can be path-traced. White, because that is what the Algan
// back end gives the same two mobs (algan_render.py: a bare Surface would
// otherwise bake its own default green into the vertices). Everything else
// falls to the tracer's own defaults.
//
// What this does NOT do is produce a reference for those materials. The path
// tracer has no normal-packing, no depth ramp, no toon banding and no matcap;
// it converts every material to its own PBR model. Those panels are the
// tracer's PBR stand-in, not three.js's material, and the returned
// substitutedMaterials list is what says so beside the image.
function substituteUnsupportedMaterials(scene) {
  const substituted = [];
  scene.traverse(o => {
    const m = o.material;
    if (!m || m.color) return;
    m.color = new THREE.Color(1, 1, 1);
    substituted.push((o.name || '<unnamed>') + ' (' + m.type + ')');
  });
  return substituted;
}

async function renderPathTrace(spec, samples, opts) {
  opts = opts || {};
  const t0 = performance.now();
  const renderer = makeRenderer(spec.render.width, spec.render.height);
  // A lost context does not raise: renderSample() returns, pt.samples counts up, and
  // toDataURL hands back a fully transparent frame. Left undetected that is written to
  // disk and compared as though it were a render. Fail instead.
  let contextLost = false;
  renderer.domElement.addEventListener('webglcontextlost', () => { contextLost = true; });
  const pt = new WebGLPathTracer(renderer);
  pt.renderToCanvas = true;
  pt.renderDelay = 0;
  pt.minSamples = 1;
  pt.fadeDuration = 0;
  pt.dynamicLowRes = false;
  // How many draws one sample is split into. The path tracer's own default is 3x3;
  // 1x1 draws the whole frame in a single call, which is what this used until a
  // hardware GL device made it a problem: on Windows a draw call that runs longer
  // than the display driver's watchdog (TDR, two seconds by default) resets the
  // device, and a WebGL context killed that way keeps accepting renderSample()
  // calls -- pt.samples counts up to 64 and the canvas is never presented. Splitting
  // one sample across N*N scissored draws keeps each one short. It does not change
  // what is accumulated: the seed and the stratified sequence advance once per
  // sample, not per tile (PathTracingRenderer._renderSample yields per tile).
  const tiles = Math.max(1, Math.round(opts.tiles || 1));
  pt.tiles.set(tiles, tiles);
  // Flat spec-colour background, zero IBL: scene.background as a Color makes the path
  // tracer use a constant background map; scene.environment stays null so the tracer's
  // environmentIntensity resolves to 0 (WebGLPathTracer.updateEnvironment). No HDRI, no
  // DataTexture fallback needed. backgroundBlur comes from scene.backgroundBlurriness = 0.
  const { scene, camera } = buildScene(spec);
  // three-gpu-pathtracer's getLights() only picks up rect-area/spot/point/directional
  // lights — AmbientLight is silently ignored. Flag it so the comparison knows.
  // three-gpu-pathtracer's getLights() (core/utils/sceneUpdateUtils.js) collects
  // ONLY rectArea / spot / point / directional. An AmbientLight or a
  // HemisphereLight is dropped without a word, so a scene carrying one is being
  // path-traced with fewer lights than it declares -- which silently changes
  // every shadow in the frame, not just the overall level.
  const droppedTypes = ['ambient', 'hemisphere']
    .filter(t => (spec.lights || []).some(l => l.type === t));
  const ambientIgnored = droppedTypes.length > 0;
  if (ambientIgnored) {
    console.error('WARNING: path tracer does not support ' + droppedTypes.join(' or ')
      + ' lights; that contribution is missing from this pass, and any comparison '
      + 'against it is against a scene with ' + droppedTypes.length + ' fewer light(s)');
  }
  // MeshBasicMaterial is unlit on both engines, and the path tracer has no unlit
  // model: it traces one as a smooth PBR dielectric (MaterialsTexture defaults
  // metalness and roughness to 0), so in a scene with no lights it comes out black.
  // That is the engine's answer, not a broken render -- say which objects it applies
  // to so a black panel is read for what it is.
  const unlit = [];
  scene.traverse(o => { if (o.material && o.material.isMeshBasicMaterial) unlit.push(o.name || '<unnamed>'); });
  if (unlit.length) {
    console.error('WARNING: path tracer has no unlit material model; ' + unlit.join(', ')
      + ' traced as PBR dielectrics (black without lights) -- those objects are NOT a reference');
  }
  const substitutedMaterials = substituteUnsupportedMaterials(scene);
  if (substitutedMaterials.length) {
    console.error('WARNING: path tracer has no shading model for ' + substitutedMaterials.join(', ')
      + '; traced as its own PBR default with a white base colour -- those objects are NOT a reference');
  }
  await pt.setScene(scene, camera);
  // renderSample() no-ops while the tracer's shaders are still compiling, and
  // compilation only progresses when the event loop turns -- so a tight
  // synchronous loop spins forever at 0 samples. Yield between samples and
  // bound the pass by wall clock rather than by iteration count (SwiftShader
  // does tens of seconds per sample on a scene of any size).
  const budgetMs = Number(opts.budgetMs || 25 * 60 * 1000);
  const start = performance.now();
  let lastLog = 0;
  while (pt.samples < samples) {
    pt.renderSample();
    await new Promise(r => setTimeout(r, 0));
    const elapsed = performance.now() - start;
    if (elapsed - lastLog > 30000) {
      lastLog = elapsed;
      console.log('path tracer: ' + pt.samples + '/' + samples + ' samples, '
        + (elapsed / 1000).toFixed(0) + 's');
    }
    if (elapsed > budgetMs) {
      console.log('path tracer: budget reached at ' + pt.samples + '/' + samples + ' samples');
      break;
    }
  }
  if (pt.samples < 1) throw new Error('path tracer produced no samples');
  if (contextLost || renderer.getContext().isContextLost()) {
    throw new Error(
      'WebGL context was lost during the path-traced pass (' + pt.samples + ' samples '
      + 'counted, frame is empty). On a hardware device this is usually the display '
      + 'driver watchdog killing an over-long draw call: raise --tiles, or fall back '
      + 'to --gl swiftshader.');
  }
  const dataUrl = renderer.domElement.toDataURL('image/png');
  renderer.dispose();
  return {
    dataUrl, ms: performance.now() - t0, samples: pt.samples,
    ambientIgnored, droppedTypes, substitutedMaterials,
  };
}

window.renderThreeScene = async function (spec, opts) {
  const out = {};
  if (opts.mode === 'raster' || opts.mode === 'both') out.raster = await renderRaster(spec);
  if (opts.mode === 'pathtrace' || opts.mode === 'both') out.pathtrace = await renderPathTrace(spec, opts.samples, opts);
  return out;
};
window.THREE_VERSION = THREE.REVISION;
window.dispatchEvent(new Event('three-ready'));
</script></body></html>`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const spec = JSON.parse(fs.readFileSync(args.scenePath, 'utf8'));
  const name = spec.name || path.basename(args.scenePath, '.json');
  const outDir = path.resolve(args.out || '.');
  fs.mkdirSync(outDir, { recursive: true });

  const nmDir = resolveNodeModules(args.nodeModules);
  const { chromium } = await import(pathToFileURL(path.join(nmDir, 'playwright', 'index.mjs')));

  const server = await startServer(nmDir);
  const port = server.address().port;

  const launchOpts = { headless: true, args: glArgs(args.gl) };
  const exe = findChromiumExecutable();
  if (exe) launchOpts.executablePath = exe;
  const browser = await chromium.launch(launchOpts);
  try {
    const page = await browser.newPage({
      viewport: { width: Math.max(spec.render.width, 640) + 80, height: Math.max(spec.render.height, 480) + 80 },
    });
    page.on('console', m => console.error('[page]', m.text()));
    page.on('pageerror', e => console.error('[pageerror]', e.message));
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.waitForFunction('typeof window.renderThreeScene === "function"', { timeout: 60000 });
    const revision = await page.evaluate('window.THREE_VERSION');
    // Which WebGL2 implementation actually answered, recorded beside the image:
    // --gl asks, the browser decides, and a silent fallback to SwiftShader is the
    // difference between a four-minute pass and a four-hour one.
    const glRenderer = await page.evaluate(() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (!gl) return 'none';
      const d = gl.getExtension('WEBGL_debug_renderer_info');
      return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    });
    console.error(`webgl2: ${glRenderer}`);

    console.error(`rendering ${name} (${spec.render.width}x${spec.render.height}, mode=${args.mode}, samples=${args.samples}) ...`);
    const t0 = Date.now();
    const result = await page.evaluate(
      ({ spec, opts }) => window.renderThreeScene(spec, opts),
      { spec, opts: { mode: args.mode, samples: args.samples, tiles: args.tiles } },
    );
    const seconds = (Date.now() - t0) / 1000;

    const outputs = [];
    if (result.raster) {
      const p = path.join(outDir, `${name}.three_raster.png`);
      fs.writeFileSync(p, Buffer.from(result.raster.dataUrl.split(',')[1], 'base64'));
      outputs.push(p);
      console.error(`raster: ${result.raster.ms.toFixed(0)} ms -> ${p}`);
    }
    if (result.pathtrace) {
      const p = path.join(outDir, `${name}.three_pathtrace.png`);
      fs.writeFileSync(p, Buffer.from(result.pathtrace.dataUrl.split(',')[1], 'base64'));
      outputs.push(p);
      console.error(`pathtrace: ${result.pathtrace.samples} samples in ${result.pathtrace.ms.toFixed(0)} ms -> ${p}`);
    }

    const summary = {
      scene: name,
      outputs,
      samples: result.pathtrace ? result.pathtrace.samples : null,
      seconds,
      width: spec.render.width,
      height: spec.render.height,
      mode: args.mode,
      three_revision: revision,
      gl: args.gl,
      gl_renderer: glRenderer,
      tiles: args.tiles,
      ambient_ignored_in_pathtrace: result.pathtrace ? result.pathtrace.ambientIgnored : false,
      // Which light types the tracer dropped, by spec type name.
      pathtrace_dropped_light_types: result.pathtrace ? result.pathtrace.droppedTypes : [],
      // Objects the path tracer cannot express, rendered as its PBR default so the
      // rest of the frame can be traced. Not a reference -- see the note above
      // substituteUnsupportedMaterials.
      pathtrace_substituted_materials: result.pathtrace ? result.pathtrace.substitutedMaterials : [],
    };
    process.stdout.write(JSON.stringify(summary) + '\n');
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch(e => { console.error(e); process.exit(1); });