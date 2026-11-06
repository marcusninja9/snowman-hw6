const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aNormal;
in vec3 aTangent;
in vec2 aTexCoord;

uniform mat4 uModelView;
uniform mat4 uProjection;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vTexCoord;

void main() {
  vec4 p = uModelView * vec4(aPosition, 1.0);
  vPosition = p.xyz;

  // Inverse transpose for potential non-uniform scalings
  mat3 mv = mat3(uModelView);
  vNormal = transpose(inverse(mv)) * aNormal;


  vTangent = mv * aTangent;
  vTexCoord = aTexCoord;
  gl_Position = uProjection * p;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vTexCoord;

uniform vec3 uLightPos;
uniform sampler2D uTexture;    // color (albedo) map, texture unit 0
uniform sampler2D uDetailMap;  // grayscale height or tangent-space normal, unit 1
uniform int uMapping;          // 0=bump, 1=normal, 2=ordinary texture
uniform vec2 uUVOffset;        // animated UV scrolling for the background / ground
uniform bool uUnlit;           // keep forest painting its original colors
uniform float uSpecStrength;   // snow is matte; coal and carrot are shinier

out vec4 fragColor;

const float bumpStrength = 12.0;

float heightAt(vec2 uv) {
  // Height texture is grayscale. It is NOT the same as the white color map.
  return texture(uDetailMap, uv).r;
}

void main() {
  vec2 uv = vTexCoord + uUVOffset;
  vec3 baseColor = texture(uTexture, uv).rgb;

  if (uUnlit) {
    fragColor = vec4(baseColor, 1.0);
    return;
  }

  // TBN transforms tangent-space perturbations into camera/view-space normals.
  vec3 N = normalize(vNormal);
  vec3 T = normalize(vTangent - N * dot(N, vTangent));
  vec3 B = normalize(cross(N, T));
  mat3 TBN = mat3(T, B, N);

  if (uMapping == 0) {
    // BUMP MAPPING: derive the normal by sampling neighboring height texels.
    // The sphere's snow texture has genuine variation in surface height.
    vec2 pixel = 1.0 / vec2(textureSize(uDetailMap, 0));
    float dx = (heightAt(uv + vec2(pixel.x, 0.0))
              - heightAt(uv - vec2(pixel.x, 0.0))) * 0.5;
    float dy = (heightAt(uv + vec2(0.0, pixel.y))
              - heightAt(uv - vec2(0.0, pixel.y))) * 0.5;
    vec3 bumpNormal = normalize(vec3(-bumpStrength * dx,
                                     -bumpStrength * dy, 1.0));
    N = normalize(TBN * bumpNormal);
  } else if (uMapping == 1) {
    // NORMAL MAPPING: read precomputed tangent-space XYZ from RGB channels.
    // (0.5,0.5,1.0) represents a flat surface; other values add carrot ridges.
    vec3 tangentNormal = normalize(texture(uDetailMap, uv).rgb * 2.0 - 1.0);
    N = normalize(TBN * tangentNormal);
  }
  // uMapping == 2: use the geometric normal without perturbation.

  vec3 L = normalize(uLightPos - vPosition);
  vec3 V = normalize(-vPosition);
  vec3 R = reflect(-L, N);
  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(V, R), 0.0), 24.0);

  vec3 ambient = 0.28 * baseColor;
  vec3 diffuse = 0.72 * baseColor * diff;
  vec3 specular = vec3(uSpecStrength * spec);
  fragColor = vec4(ambient + diffuse + specular, 1.0);
}
`;

// WebGL setup
const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl2");

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const ATTRIBUTES = { aPosition: 0, aNormal: 1, aTexCoord: 2, aTangent: 3 };

function createProgram(vsSource, fsSource) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
  for (const name in ATTRIBUTES) gl.bindAttribLocation(prog, ATTRIBUTES[name], name);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// Shader editors
const vertEditor = document.getElementById("vertEditor");
const fragEditor = document.getElementById("fragEditor");
const shaderError = document.getElementById("shaderError");
vertEditor.value = VERTEX_SHADER;
fragEditor.value = FRAGMENT_SHADER;

let program, uModelView, uProjection, uLightPos, uMapping, uUVOffset, uUnlit, uSpecStrength;

function updateProgram() {
  try {
    const prog = createProgram(vertEditor.value, fragEditor.value);
    gl.deleteProgram(program);
    program = prog;
    gl.useProgram(program);
    uModelView = gl.getUniformLocation(program, "uModelView");
    uProjection = gl.getUniformLocation(program, "uProjection");
    uLightPos = gl.getUniformLocation(program, "uLightPos");
    uMapping = gl.getUniformLocation(program, "uMapping");
    uUVOffset = gl.getUniformLocation(program, "uUVOffset");
    uUnlit = gl.getUniformLocation(program, "uUnlit");
    uSpecStrength = gl.getUniformLocation(program, "uSpecStrength");
    gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(program, "uDetailMap"), 1);
    shaderError.textContent = "";
  } catch (e) {
    shaderError.textContent = e.message.replace(/\0/g, "");
  }
}
updateProgram();
vertEditor.addEventListener("input", updateProgram);
fragEditor.addEventListener("input", updateProgram);

function bindAttribute(name, data, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.vertexAttribPointer(ATTRIBUTES[name], size, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(ATTRIBUTES[name]);
}

function createMesh({ positions, normals, tangents, texCoords, indices }) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  bindAttribute("aPosition", positions, 3);
  bindAttribute("aNormal", normals, 3);
  bindAttribute("aTangent", tangents, 3);
  bindAttribute("aTexCoord", texCoords, 2);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  return { vao, count: indices.length };
}

// One sphere mesh is reused for the two snowballs and both eyes.
const sphereMesh = createMesh(createSphere());
const coneMesh = createMesh(createCone());
// Four repeats preserve the forest photo's proportions on the long wall.
const backgroundMesh = createMesh(createQuad(4, 1));
const groundMesh = createMesh(createQuad(8, 6));      // repeat over the snow floor

// 0: bump mapping via height-gradient samples; 1: precomputed RGB normal map;
// 2: regular texture mapping (used for scenery and the eyes).
const BUMP_MAPPING = 0, NORMAL_MAPPING = 1, COLOR_MAPPING = 2;

// A 1-pixel color texture for the smooth black eyes.
function createColorTexture(r, g, b) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([r, g, b, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return texture;
}

const coalTexture = createColorTexture(18, 20, 25);

// Load the two JPEGs from the SAME folder as hw6.html / main.js.
// MIRRORED_REPEAT prevents a hard cut between the left and right edges of
// the forest photo (which was not originally designed as a seamless tile).
// This still scrolls the image via uUVOffset in the fragment shader.
function loadImageTexture(filename, mirrored = false, placeholder = [231, 241, 248, 255]) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  // A pale 1x1 pixel shows until the JPEG finishes loading.
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE,
                new Uint8Array(placeholder));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,
                   mirrored ? gl.MIRRORED_REPEAT : gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,
                   mirrored ? gl.MIRRORED_REPEAT : gl.REPEAT);

  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // JPEG origin is at top left; our UV origin is at bottom left.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB,
                  gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                     gl.LINEAR_MIPMAP_LINEAR);
  };
  image.onerror = () => {
    console.error(`Could not load ${filename}. Check its name and location.`);
    shaderError.textContent += `\nCould not load ${filename}. Make sure it is in the same folder as hw6.html.`;
  };
  image.src = filename;
  return texture;
}

// ANIMATED TEXTURE: redraw the user's forest photo + falling snow into an
// offscreen 2D canvas, then upload its changing pixels to WebGL. Unlike just
// scrolling UVs, this is also genuine frame-by-frame TEXTURE CONTENT animation.
// The snow is light so the original uploaded image remains the main artwork.
function createAnimatedForestTexture(filename) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([231, 241, 248, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const offscreen = document.createElement("canvas");
  const ctx = offscreen.getContext("2d");
  const image = new Image();
  let ready = false;
  let textureUploaded = false;
  let lastFrame = -1;
  image.onload = () => {
    offscreen.width = image.width;
    offscreen.height = image.height;
    ready = true;
    update(0);
  };
  image.onerror = () => {
    shaderError.textContent += `\nCould not load ${filename}. Use localhost and check the file name.`;
  };
  image.src = filename;

  // Deterministic particles: no sprite files or external dependencies.
  const flakes = Array.from({ length: 70 }, (_, i) => ({
    x: ((i * 0.618033989 + 0.11) % 1),
    y: ((i * 0.414213562 + 0.17) % 1),
    speed: 0.025 + (i % 7) * 0.006,
    size: 1 + (i % 3) * 0.45,
    alpha: 0.30 + (i % 5) * 0.10
  }));

  function update(time) {
    if (!ready || (lastFrame >= 0 && time - lastFrame < 1 / 15)) return;
    lastFrame = time;
    const w = offscreen.width, h = offscreen.height;
    ctx.drawImage(image, 0, 0, w, h);
    for (const f of flakes) {
      const y = ((f.y + time * f.speed) % 1) * h;
      const x = ((f.x + time * 0.008) % 1) * w;
      ctx.fillStyle = `rgba(255,255,255,${f.alpha})`;
      ctx.beginPath();
      ctx.arc(x, y, f.size * 1.3, 0, Math.PI * 2);
      ctx.fill();
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    if (!textureUploaded) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                    gl.UNSIGNED_BYTE, offscreen);
      textureUploaded = true;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA,
                       gl.UNSIGNED_BYTE, offscreen);
    }
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                     gl.LINEAR_MIPMAP_LINEAR);
  }
  return { texture, update };
}

// Your exact two uploaded photos, not computer-generated substitutes.
const animatedForest = createAnimatedForestTexture("SnowmanBackground.jpg");
const forestTexture = animatedForest.texture;
const groundTexture = loadImageTexture("SnowmanGround.jpg", true);

// These maps were generated from the uploaded snow photo / a textured carrot.
// SnowHeight.png stores HEIGHT in its red channel for runtime bump mapping.
// CarrotNormal.png stores XYZ normals encoded in RGB for normal mapping.
const snowColorTexture = loadImageTexture("SnowColor.png");
const snowHeightTexture = loadImageTexture("SnowHeight.png", false, [128, 128, 128, 255]);
const carrotColorTexture = loadImageTexture("CarrotColor.png");
const carrotNormalTexture = loadImageTexture("CarrotNormal.png", false, [128, 128, 255, 255]);

const snowMaterial = { color: snowColorTexture, detail: snowHeightTexture,
                       mapping: BUMP_MAPPING, spec: 0.13 };
const carrotMaterial = { color: carrotColorTexture, detail: carrotNormalTexture,
                         mapping: NORMAL_MAPPING, spec: 0.30 };
const coalMaterial = { color: coalTexture, detail: null,
                       mapping: COLOR_MAPPING, spec: 0.38 };
let useSurfaceDetail = true; // toggle to see what the maps change


// Environment transforms deliberately do NOT belong to the snowman hierarchy.
// The background is a vertical XY rectangle at z=-5.5.
// The ground is the same rectangle rotated onto the XZ plane, at snowball level.
const forestWorld = multiply(translate(0, 2.6, -5.5), scale(24, 4.4, 1));
const groundWorld = multiply(
  multiply(translate(0, -1.77, -0.8), rotateX(-Math.PI / 2)),
  scale(24, 7, 1)
);

function drawScenery(mesh, texture, world, view, uScroll, unlit) {
  gl.uniformMatrix4fv(uModelView, false, multiply(view, world));
  gl.uniform1i(uMapping, COLOR_MAPPING);
  gl.uniform2f(uUVOffset, uScroll, 0);
  gl.uniform1i(uUnlit, unlit ? 1 : 0);
  gl.uniform1f(uSpecStrength, 0.12);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.bindVertexArray(mesh.vao);
  gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
}


// Scene graph: a child inherits all of its ancestors' transformations.
// Pivot nodes have no mesh; they position other nodes without scaling them.
function node(transform, mesh = null, material = null, children = []) {
  return { transform, mesh, material, children };
}

const snowman = node(translate(0, 0, 0), null, null, [
  // The body pivot is the parent of the head pivot.
  node(translate(0, -0.7, 0), null, null, [
    node(scale(1.05, 1.05, 1.05), sphereMesh, snowMaterial),

    // Moving or rotating this head pivot also moves the eyes and nose.
    node(translate(0, 1.65, 0), null, null, [
      node(scale(0.66, 0.66, 0.66), sphereMesh, snowMaterial),
      node(multiply(translate(-0.23, 0.12, 0.63), scale(0.09, 0.09, 0.09)),
           sphereMesh, coalMaterial),
      node(multiply(translate(0.23, 0.12, 0.63), scale(0.09, 0.09, 0.09)),
           sphereMesh, coalMaterial),
      node(multiply(multiply(translate(0, -0.08, 0.79), rotateX(Math.PI / 2)),
                    scale(0.17, 0.42, 0.17)), coneMesh, carrotMaterial)
    ])
  ])
]);

// Traverse the hierarchy. Each material may use either a height map (bump),
// a tangent-space normal map, or just its ordinary color texture.
function drawNode(current, parentWorld, view) {
  const world = multiply(parentWorld, current.transform);
  if (current.mesh) {
    const material = current.material;
    gl.uniformMatrix4fv(uModelView, false, multiply(view, world));
    gl.uniform1i(uMapping, useSurfaceDetail ? material.mapping : COLOR_MAPPING);
    gl.uniform2f(uUVOffset, 0, 0);
    gl.uniform1i(uUnlit, 0);
    gl.uniform1f(uSpecStrength, material.spec);
    gl.activeTexture(gl.TEXTURE0);                  // base color
    gl.bindTexture(gl.TEXTURE_2D, material.color);
    gl.activeTexture(gl.TEXTURE1);                  // height / normal data
    gl.bindTexture(gl.TEXTURE_2D, material.detail);
    gl.bindVertexArray(current.mesh.vao);
    gl.drawElements(gl.TRIANGLES, current.mesh.count, gl.UNSIGNED_SHORT, 0);
  }
  for (const child of current.children) drawNode(child, world, view);
}

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0.055, 0.085, 0.15, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld = [3, 4, 5];

// Interactive controls
let dragRotX = 0, dragRotY = 0;      // mouse drag
let camX = 0, camY = 0, camZ = -7;      // arrow keys, w, s
let sliderAngle = 0;                    // degrees
let spinning = false, spinAngle = 0;    // Start/Stop button
// One animation clock drives BOTH the hops and scrolling textures.
// Keeping them synchronized creates the illusion of running to the right.
let moving = true;
let movementTime = 0;

// The body's center is at local y=-0.7 and its radius is 1.05,
// so its lowest point is y=-1.75. The ground sits at y=-1.77.
// Lowering the whole hierarchy by 0.12 puts it just into the snow
// (by about 0.10 units), avoiding a visible gap at takeoff/landing.
const SNOWMAN_REST_Y = -0.12;
const HOP_HEIGHT = 0.53;       // how high the snowman rises, in world units
const HOPS_PER_SECOND = 1.2;  // number of hops each second

let mouseDown = false, lastX, lastY;
canvas.addEventListener("mousedown", e => { mouseDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", e => {
  if (!mouseDown) return;
  dragRotY += (e.clientX - lastX) * 0.01;
  dragRotX += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});

document.addEventListener("keydown", e => {
  if (e.target === vertEditor || e.target === fragEditor) return;
  const step = 0.2;
  switch (e.key) {
    case "ArrowUp": camY -= step; break;
    case "ArrowDown": camY += step; break;
    case "ArrowLeft": camX += step; break;
    case "ArrowRight": camX -= step; break;
    case "w": camZ += step; break;
    case "s": camZ -= step; break;
    default: return;
  }
  e.preventDefault();
});

const axisSelect = document.getElementById("axisSelect");
document.getElementById("slider").addEventListener("input", e => {
  sliderAngle = Number(e.target.value);
  document.getElementById("sliderValue").textContent = sliderAngle;
});
document.getElementById("spinButton").addEventListener("click", () => spinning = !spinning);
const surfaceDetailCheckbox = document.getElementById("surfaceDetail");
surfaceDetailCheckbox.addEventListener("change", () => {
  useSurfaceDetail = surfaceDetailCheckbox.checked;
});
const moveButton = document.getElementById("moveButton");
moveButton.addEventListener("click", () => {
  moving = !moving;
  moveButton.textContent = moving ? "Pause Movement" : "Resume Movement";
});

// Render loop
let lastTime = 0;
function render(time) {
  const t = time / 1000;
  const dt = Math.min(Math.max(t - lastTime, 0), 0.05);
  if (spinning) spinAngle += dt;
  if (moving) movementTime += dt;
  lastTime = t;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const view = translate(camX, camY, camZ);
  gl.uniformMatrix4fv(uProjection, false, projection);
  gl.uniform3fv(uLightPos, transformPoint(view, lightWorld));

  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);
  const rotation = multiply(multiply(rotateY(dragRotY), rotateX(dragRotX)), axisRotation);

  gl.activeTexture(gl.TEXTURE0);

  // The forest image itself gains new snowflake pixels at 15 fps.
  animatedForest.update(movementTime);

  // Increasing the UV offset makes the pattern travel toward screen-left.
  // The distant forest scrolls more slowly than the nearby snow (parallax).
  drawScenery(backgroundMesh, forestTexture, forestWorld, view,
              movementTime * 0.035, true);
  drawScenery(groundMesh, groundTexture, groundWorld, view,
              movementTime * 0.18, false);

  // The face originally points along +Z. Turning +90 degrees around Y
  // makes the carrot point toward +X (screen-right).
  // This rotates the whole hierarchy, including the attached eyes and nose.
  // A full hop uses sin^2: y=0 when landing and y=HOP_HEIGHT at the apex.
  // Only the root moves, so the body, head, eyes and nose hop together.
  // The snowman stays at a fixed screen X while the background slides left.
  const hopPhase = Math.PI * HOPS_PER_SECOND * movementTime;
  const hopY = HOP_HEIGHT * Math.pow(Math.sin(hopPhase), 2);
  const facingRight = multiply(rotation, rotateY(Math.PI / 2));
  snowman.transform = multiply(translate(0, SNOWMAN_REST_Y + hopY, 0), facingRight);
  drawNode(snowman, translate(0, 0, 0), view);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
