////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
struct CameraUniforms
{
  mvp: mat4x4<f32>, // view * projection matrix
  right : vec3<f32>, // camera.right and up are used to determine the direction the particle "sprites" face
  _pad0 : f32,
  up : vec3<f32>,
  _pad1 : f32,
}; 

struct VertexInput {
  @location(0) instPos : vec3<f32>,
  @location(1) instColor : vec4<f32>,
  @location(2) corner : vec2<f32>,
};

struct VertexOutput
{
    @builtin(position) pos : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) corner : vec2<f32>,
};

@group(0) @binding(0) var<uniform> cam : CameraUniforms;
@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  let size = 0.02;
  let worldPos = in.instPos + cam.right * (in.corner.x * size) + cam.up * (in.corner.y * size);

  var out: VertexOutput;
  out.pos = cam.mvp * vec4<f32>(worldPos, 1.0);
  out.color = in.instColor;
  out.corner = in.corner; 
  return out;

}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
  var color = in.color;
  
  //var color = vec4f(1,1,1,1);
  color.a = color.a * max(1.0 - length(in.corner), 0.0);
  return color;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////

struct SimulationParams {
  dt : f32,
  bounce : f32,
  count: u32,
  gravity: f32,

  //sph related
  smoothing_radius: f32,
  smoothing_radius2: f32,
  mass: f32,
  rest_density: f32,
  pressure_constant: f32,
  viscosity_constant: f32,
  
  boxMin : vec3<f32>,
  boxMax : vec3<f32>,

  // Kernel precomputations
  poly6_constant: f32,
  spiky_constant: f32,

};

struct Particle {
  //pos life color vel and pad 1
  pos : vec3<f32>,
  life: f32,
  color: vec4<f32>,
  vel: vec3<f32>,
  _pad2: f32,
};

struct SPHData {
  density: f32,
  pressure: f32,
  _pad0: vec2<f32>,
  force: vec3<f32>,
  _pad1: f32,
};

const count = 5000;
@group(0) @binding(0) var<uniform> sim : SimulationParams;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> sph_data : array<SPHData>;


// helper functions

fn poly6_kernel(r2: f32) -> f32
{
  let h2 = sim.smoothing_radius2;
  if (r2 >= h2) {
    return 0.0;
  }
  return sim.poly6_constant * pow(h2 - r2, 3.0);
}

fn spikey_kernel_gradient(r: f32, r_norm: vec3<f32>) -> vec3<f32>
{
  let h = sim.smoothing_radius;
  if (r >= h || r <= 0.0) {
    return vec3<f32>(0.0);
  }
  return sim.spiky_constant * pow(h - r, 2.0) * r_norm;
  //return  pow(h - r, 2.0) * r_norm;
}

fn viscosity_kernel(r: f32) -> f32 {
  let h = sim.smoothing_radius;
  let h2 = sim.smoothing_radius2;
  let h3 = h2 * h;
  
  if (r >= h || r <= 0.0) {
    return 0.0;
  }
  
  return -(r * r * r) / (2.0 * h3) + (r * r) / h2 + h / (2.0 * r) - 1.0;
} 

// first phase: computing densities and pressure
@compute @workgroup_size(64)
fn compute_density_pressure(@builtin(global_invocation_id) gid: vec3u) 
{
  let i = gid.x;
  if(i >= count)
  {  return;       }

  var particleA = particles[i];
  var data = sph_data[i];

  data.density = 0.0;

  for (var j = 0u; j < count; j+= 1u)
  {
    let particleB = particles[j];
    let dis = particleA.pos - particleB.pos;

    let r2 = dot(dis, dis);

    if(r2 < sim.smoothing_radius2)
    {
      data.density += sim.mass * poly6_kernel(r2);
    }
  }

  data.density = max(1e-6, data.density);

  data.pressure = sim.pressure_constant * (data.density - sim.rest_density);

  sph_data[i] = data;
  
}


//phase 2: compute forces: 
@compute @workgroup_size(64)
fn compute_forces(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if(i >= count)
  {  return;       }

  var particleA = particles[i];
  var data = sph_data[i];

  data.force = vec3<f32>(0.0);

  for(var j = 0u; j < count; j+= 1u)
  {
    if (i == j) 
    {continue;}

    let particleB = particles[j];
    let dataB = sph_data[j];

    let dis = particleA.pos - particleB.pos;
    let r2 = dot(dis, dis);
    let r = sqrt(r2);

    if (r > 0.0 && r < sim.smoothing_radius)
    {
      let r_norm = dis / r;

      //pressure force
      let pressure_force = -sim.mass * (data.pressure / (data.density * data.density) +
               dataB.pressure / (dataB.density * dataB.density))  * spikey_kernel_gradient(r, r_norm);

      //viscosity
      let viscosity_force = (1.0 / dataB.density) * (particleB.vel - particleA.vel) * viscosity_kernel(r);

      data.force += pressure_force + (sim.viscosity_constant *  viscosity_force);
    }

  }

  sph_data[i] = data;

}


//phase 3: integrate
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if(i >= count)
  {  return;       }

  var p = particles[i];
  let data = sph_data[i];

  let gravity = vec3<f32>(0.0, sim.gravity * sim.mass, 0.0);
  p.vel += sim.dt * (data.force / data.density + gravity);
  p.pos += sim.dt * p.vel;
  

  // boundary box collision handling
  if (p.pos.x < sim.boxMin.x) {
    p.pos.x = sim.boxMin.x;
    p.vel.x = abs(p.vel.x) * sim.bounce;
  } else if (p.pos.x > sim.boxMax.x) {
    p.pos.x = sim.boxMax.x;
    p.vel.x = -abs(p.vel.x) * sim.bounce;
  }

  if (p.pos.y < sim.boxMin.y) {
    p.pos.y = sim.boxMin.y;
    p.vel.y = abs(p.vel.y) * sim.bounce;
  } else if (p.pos.y > sim.boxMax.y) {
    p.pos.y = sim.boxMax.y;
    p.vel.y = -abs(p.vel.y) * sim.bounce;
  }

  if (p.pos.z < sim.boxMin.z) {
    p.pos.z = sim.boxMin.z;
    p.vel.z = abs(p.vel.z) * sim.bounce;
  } else if (p.pos.z > sim.boxMax.z) {
    p.pos.z = sim.boxMax.z;
    p.vel.z = -abs(p.vel.z) * sim.bounce;
  }

  particles[i] = p;

}



// // old
// @compute @workgroup_size(64) // test other workgroup sizes
// fn simulate(@builtin(global_invocation_id) gid : vec3u) {
//    let i = gid.x;
//    if (i >= count) {return;}

//    var p = particles[i];
//    p.vel = p.vel + vec3(0,0, sim.gravity * sim.dt); 
//    p.pos = p.pos + p.vel * sim.dt;
//    // collision handling
//   if (p.pos.x < sim.boxMin.x) {
//     p.pos.x = sim.boxMin.x;
//     p.vel.x = abs(p.vel.x) * sim.bounce;
//   } else if (p.pos.x > sim.boxMax.x) {
//     p.pos.x = sim.boxMax.x;
//     p.vel.x = -abs(p.vel.x) * sim.bounce;
//   }

//   if (p.pos.y < sim.boxMin.y) {
//     p.pos.y = sim.boxMin.y;
//     p.vel.y = abs(p.vel.y) * sim.bounce;
//   } else if (p.pos.y > sim.boxMax.y) {
//     p.pos.y = sim.boxMax.y;
//     p.vel.y = -abs(p.vel.y) * sim.bounce;
//   }

//   if (p.pos.z < sim.boxMin.z) {
//     p.pos.z = sim.boxMin.z;
//     p.vel.z = abs(p.vel.z) * sim.bounce;
//   } else if (p.pos.z > sim.boxMax.z) {
//     p.pos.z = sim.boxMax.z;
//     p.vel.z = -abs(p.vel.z) * sim.bounce;
//   }

//   particles[i] = p;
//   }





