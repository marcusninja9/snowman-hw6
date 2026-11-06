function createCube() {
  const positions = new Float32Array([
    -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,   // front
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,   1,  1, -1,   // back
    -1,  1,  1,   1,  1,  1,   1,  1, -1,  -1,  1, -1,   // top
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1,  1,   // bottom
     1, -1,  1,   1, -1, -1,   1,  1, -1,   1,  1,  1,   // right
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1    // left
  ]);
  const faceNormals = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]];
  const normals = new Float32Array(faceNormals.flatMap(n => [...n, ...n, ...n, ...n]));

  // Tangent; bitangent = cross(N, T)
  const faceTangents = [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]];
  const tangents = new Float32Array(faceTangents.flatMap(t => [...t, ...t, ...t, ...t]));

  const texCoords = new Float32Array(faceNormals.flatMap(() => [0, 0, 1, 0, 1, 1, 0, 1]));

  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,   // front
     4,  5,  6,   4,  6,  7,   // back
     8,  9, 10,   8, 10, 11,   // top
    12, 13, 14,  12, 14, 15,   // bottom
    16, 17, 18,  16, 18, 19,   // right
    20, 21, 22,  20, 22, 23    // left
  ]);
  return { positions, normals, tangents, texCoords, indices };
}


function createCylinder(segments = 64, uRepeat = 3) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  // Side: theta grows with u, so T points along the circle and B = cross(N, T) points up
  for (let i = 0; i <= segments; i++) {
    const u = i / segments, theta = 2 * Math.PI * u;
    const s = Math.sin(theta), c = Math.cos(theta);
    for (const y of [-1, 1]) {
      positions.push(s, y, c);
      normals.push(s, 0, c);
      tangents.push(c, 0, -s);
      texCoords.push(u * uRepeat, (y + 1) / 2);
    }
  }
  for (let i = 0; i < segments; i++) {
    const k = 2 * i;
    indices.push(k, k + 2, k + 3, k, k + 3, k + 1);
  }

  // Caps: planar mapping of x, z into the texture, T = +x on both
  for (const y of [1, -1]) {
    const center = positions.length / 3;
    positions.push(0, y, 0);
    normals.push(0, y, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5, 0.5);
    for (let i = 0; i <= segments; i++) {
      const theta = 2 * Math.PI * i / segments;
      const s = Math.sin(theta), c = Math.cos(theta);
      positions.push(s, y, c);
      normals.push(0, y, 0);
      tangents.push(1, 0, 0);
      texCoords.push(0.5 + 0.5 * s, 0.5 - 0.5 * y * c);  // keeps cross(N, T) along +v
    }
    for (let i = 0; i < segments; i++) {
      const a = center + 1 + i, b = a + 1;
      if (y > 0) indices.push(center, a, b);
      else indices.push(center, b, a);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// A unit sphere centered at the origin. UVs and tangents allow texture mapping
// (and later bump/normal mapping) without changing the geometry.
function createSphere(stacks = 32, slices = 48) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  // v goes from the south pole to the north pole. The tangent points in +u.
  for (let i = 0; i <= stacks; i++) {
    const v = i / stacks;
    const phi = Math.PI * (1 - v);
    const y = Math.cos(phi);
    const radius = Math.sin(phi);

    for (let j = 0; j <= slices; j++) {
      const u = j / slices;
      const theta = 2 * Math.PI * u;
      const s = Math.sin(theta), c = Math.cos(theta);
      const x = radius * s, z = radius * c;

      positions.push(x, y, z);
      normals.push(x, y, z);
      tangents.push(c, 0, -s);
      texCoords.push(u, v);
    }
  }

  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i * (slices + 1) + j;
      const b = a + slices + 1;
      indices.push(a, a + 1, b + 1, a, b + 1, b);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// A unit cone along +Y: the base is at y = -1, and its tip is at y = +1.
// Rotating it 90 degrees around X makes it point toward the camera (+Z).
function createCone(segments = 48) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];
  const slope = 0.5;  // radius falls from 1 to 0 over a height of 2
  const normalLength = Math.hypot(1, slope);

  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const theta = 2 * Math.PI * u;
    const s = Math.sin(theta), c = Math.cos(theta);

    // Separate bottom/top vertices give the cone a clean UV seam.
    for (const y of [-1, 1]) {
      const radius = (1 - y) / 2;
      positions.push(radius * s, y, radius * c);
      normals.push(s / normalLength, slope / normalLength, c / normalLength);
      tangents.push(c, 0, -s);
      texCoords.push(u, (y + 1) / 2);
    }
  }

  for (let i = 0; i < segments; i++) {
    const k = 2 * i;
    indices.push(k, k + 2, k + 3); // one triangle per side wedge
  }

  // Flat bottom cap, with its own normal and planar UV coordinates.
  const center = positions.length / 3;
  positions.push(0, -1, 0);
  normals.push(0, -1, 0);
  tangents.push(1, 0, 0);
  texCoords.push(0.5, 0.5);
  for (let i = 0; i <= segments; i++) {
    const theta = 2 * Math.PI * i / segments;
    const s = Math.sin(theta), c = Math.cos(theta);
    positions.push(s, -1, c);
    normals.push(0, -1, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5 + 0.5 * s, 0.5 + 0.5 * c);
  }
  for (let i = 0; i < segments; i++) {
    // Clockwise when viewed from above => outward/downward facing cap.
    indices.push(center, center + i + 2, center + i + 1);
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// A textured rectangle centered at the origin in the XY plane.
// Its front side points toward +Z. To make horizontal ground, rotate it
// -90 degrees around X so the front side points upward (+Y).
function createQuad(uRepeat = 1, vRepeat = 1) {
  return {
    positions: new Float32Array([
      -1, -1, 0,   1, -1, 0,   1, 1, 0,   -1, 1, 0
    ]),
    normals: new Float32Array([
      0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1
    ]),
    tangents: new Float32Array([
      1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0
    ]),
    texCoords: new Float32Array([
      0, 0,   uRepeat, 0,   uRepeat, vRepeat,   0, vRepeat
    ]),
    indices: new Uint16Array([0, 1, 2, 0, 2, 3])
  };
}
