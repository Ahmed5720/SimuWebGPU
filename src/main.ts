import { mat4, vec3 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import './style.css'
import particleWGSL from './particle.wgsl?raw';

const numParticles = 5000;
const particlePositionOffset = 0;
const particleColorOffset = 4 * 4;
const particleInstanceByteSize = 48;
const sphDataSize = 32;


const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const adapter = await navigator.gpu?.requestAdapter({
  featureLevel: 'compatibility',
});
const device = await adapter?.requestDevice();
if (!navigator.gpu) throw new Error('WebGPU not available: navigator.gpu is missing.');
if (!adapter) throw new Error('WebGPU not available: requestAdapter() returned null.');
if (!device) throw new Error('WebGPU not available: requestDevice() returned null.');


const context = canvas.getContext('webgpu');

const devicePixelRatio = window.devicePixelRatio;
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const presentationFormat = 'rgba16float';

function configureContext() {
  context.configure({
    device,
    format: presentationFormat,
  });
}


// buffer used in vertex and compute shader we also need to be able to copy into it with a staging buffer to initialize it.
const particlesBuffer = device.createBuffer({
  size: numParticles * particleInstanceByteSize,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const sphDataBuffer = device.createBuffer({
  size: numParticles * sphDataSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
})
const boxMin = {x: -1.0, y: -1.0, z: -1.0};
const boxMax = {x: 1.0, y: 1.0, z: 1.0}

const floatPerParticle = particleInstanceByteSize / 4;
const initData = new Float32Array(numParticles * floatPerParticle);

function rand(a: number, b: number) // random in range a , b 
{
  return a + Math.random() * (b-a);
}

function initParticles()
{
  for (let i = 0; i < numParticles; i++)
  {
    const base = i * floatPerParticle;
    const px = rand(boxMin.x, boxMax.x);
    const py = rand(boxMin.y, boxMax.y);
    const pz = rand(boxMin.z, boxMax.z);

    const vx = rand(-0.2, 0.2);
    const vy = rand(-0.2, 0.2);
    const vz = rand(-0.2, 0.2);

    const r = 0.2;
    const g = 0.1;
    const b = 0.9;
    const a = 1.0;

    initData[base + 0] = px;
    initData[base + 1] = py;
    initData[base + 2] = pz;
    initData[base + 3] = 1.0; // lifetime 
    initData[base + 4] = r;
    initData[base + 5] = g;
    initData[base + 6] = b;
    initData[base + 7] = a;
    initData[base + 8] = vx;
    initData[base + 9] = vy;
    initData[base + 10] = vz;
    initData[base + 11] = 0.0; // padding


    
  }

  //staging buffer needed to transfer initData to the particlesBuffer
  const staging = device.createBuffer(
    {
      size: initData.byteLength,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    }
  );
  new Float32Array(staging.getMappedRange()).set(initData);
  staging.unmap();

  const Encoder = device.createCommandEncoder();
  Encoder.copyBufferToBuffer(staging, 0, particlesBuffer, 0, initData.byteLength);
  device.queue.submit([Encoder.finish()]);


  // init sph data

  const sphData = new Float32Array(8 * numParticles);
  for (let i = 0; i < numParticles; i++)
  {
    const base = i * 8;
    sphData[base] = 1.0 // density
    sphData[base + 1] = 0.0 // pressure
    sphData[base + 2] = 0.0 // pad
    sphData[base + 3] = 0.0 // padd
    sphData[base + 4] = 0.0 // force x
    sphData[base + 5] = 0.0 // force y
    sphData[base + 6] = 0.0 // force z
    sphData[base + 7] = 0.0 // padd
  }

  const staging2 = device.createBuffer(
    {
      size: sphData.byteLength,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    }
  );
  new Float32Array(staging2.getMappedRange()).set(sphData);
  staging2.unmap();

  const Encoder2 = device.createCommandEncoder();
  Encoder2.copyBufferToBuffer(staging2, 0, sphDataBuffer, 0, sphData.byteLength);
  device.queue.submit([Encoder2.finish()]);


}


initParticles();

const renderPipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module: device.createShaderModule({
      code: particleWGSL,
    }),
    buffers: [
      {
        // instanced particles buffer
        arrayStride: particleInstanceByteSize,
        stepMode: 'instance',
        attributes: [
          {
            // position
            shaderLocation: 0,
            offset: particlePositionOffset,
            format: 'float32x3',
          },
          {
            // color
            shaderLocation: 1,
            offset: particleColorOffset,
            format: 'float32x4',
          },
        ],
      },
      {
        // quad vertex buffer
        arrayStride: 2 * 4, // vec2f
        stepMode: 'vertex',
        attributes: [
          {
            // vertex positions
            shaderLocation: 2,
            offset: 0,
            format: 'float32x2',
          },
        ],
      },
    ],
  },
  fragment: {
    module: device.createShaderModule({
      code: particleWGSL,
    }),
    targets: [
      {
        format: presentationFormat,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'zero',
            dstFactor: 'one',
            operation: 'add',
          },
        },
      },
    ],
  },
  primitive: {
    topology: 'triangle-list',
  },

  depthStencil: {
    depthWriteEnabled: false,
    depthCompare: 'less',
    format: 'depth24plus',
  },
});

let depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: 'depth24plus',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const uniformBufferSize =
  4 * 4 * 4 + // modelViewProjectionMatrix : mat4x4f
  3 * 4 + // right : vec3f
  4 + // padding
  3 * 4 + // up : vec3f
  4 + // padding
  0;
const uniformBuffer = device.createBuffer({
  size: uniformBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const uniformBindGroup = device.createBindGroup({
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: uniformBuffer,
      },
    },
  ],
});

const renderPassDescriptor: GPURenderPassDescriptor = {
  colorAttachments: [
    {
      view: undefined, // Assigned later
      clearValue: [0, 0, 0, 1],
      loadOp: 'clear',
      storeOp: 'store',
    },
  ],
  depthStencilAttachment: {
    view: depthTexture.createView(),

    depthClearValue: 1.0,
    depthLoadOp: 'clear',
    depthStoreOp: 'store',
  },
};




//////////////////////////////////////////////////////////////////////////////
// Quad vertex buffer
//////////////////////////////////////////////////////////////////////////////
const quadVertexBuffer = device.createBuffer({
  size: 6 * 2 * 4, // 6x vec2f
  usage: GPUBufferUsage.VERTEX,
  mappedAtCreation: true,
});
// prettier-ignore
const vertexData = [
  -1.0, -1.0, +1.0, -1.0, -1.0, +1.0, -1.0, +1.0, +1.0, -1.0, +1.0, +1.0,
];
new Float32Array(quadVertexBuffer.getMappedRange()).set(vertexData);
quadVertexBuffer.unmap();


//////////////////////////////////////////////////////////////////////////////
// Simulation compute pipeline
//////////////////////////////////////////////////////////////////////////////



//constant calculations
var h = 0.01; // smoothing radius
var h2 = h * h;
var h6 = h2 * h2 * h2;
var h9 = h6 * h2 * h;
var PI = Math.PI;

var poly6_constant = 315.0 / (64.0 * PI * h9);
var spiky_constant = -45.0 / (PI * h6);


const simulationParams = {
  simulate: true,
  deltaTime: 0.01,
  bounce: 0.5,
  count: numParticles,
  gravity: -9.8,

  smoothing_radius: h,
  smoothing_radius2: h2,
  mass: 1.0,
  rest_density: 625,
  pressure_constant: 1.0,
  viscosity_constant: 0.0,
  
  pad0: 0.0,
  pad1: 0.0,

  boxMin : [-1,-1,-1],
  pad2: 0.0,
  boxMax : [1,1,1],
  pad3: 0.0,

  // Kernel precomputations
  poly6_constant: poly6_constant,
  spiky_constant: spiky_constant,
  pad4: 0.0,
  pad5: 0.0,
};

const simulationUBOBufferSize = 96; 
const simulationUBOBuffer = device.createBuffer({
  size: simulationUBOBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const gui = new GUI();
gui.width = 325;
gui.add(simulationParams, 'simulate');
gui.add(simulationParams, 'deltaTime', 0.001, 0.05, 0.001);
gui.add(simulationParams, 'gravity', -30, 0, 30);
gui.add(simulationParams, 'bounce', 0.0, 1.0, 0.01);

const fSPH = gui.addFolder('SPH');
fSPH.add(simulationParams, 'mass', 0.001, 10.0, 0.001);
fSPH.add(simulationParams, 'rest_density', 0.001, 50.0, 0.001);
fSPH.add(simulationParams, 'pressure_constant', 0.0, 5000.0, 1.0);
fSPH.add(simulationParams, 'viscosity_constant', 0.0, 1.0, 0.0005);

// smoothing radius affects both smoothing_radius and smoothing_radius2
fSPH.add(simulationParams, 'smoothing_radius', 0.01, 5.0, 0.01).onChange((h: number) => {
  simulationParams.smoothing_radius = h;
  simulationParams.smoothing_radius2 = h * h;
});

fSPH.add(simulationParams, 'smoothing_radius2').listen(); // read-only display basically
fSPH.open();

// const fBox = gui.addFolder('Bounds (read-only unless you wire updates)');
// fBox.add(simulationParams, 'boxMin').listen();
// fBox.add(simulationParams, 'boxMax').listen();
// fBox.open();

const fKernel = gui.addFolder('Kernel constants');
fKernel.add(simulationParams, 'poly6_constant').listen();
fKernel.add(simulationParams, 'spiky_constant').listen();
fKernel.open();
const computeBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'uniform',
      },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'storage',
      },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'storage',
      },
    },
  ],
});

const computePipelineLayout = device.createPipelineLayout({
  bindGroupLayouts: [computeBindGroupLayout],
});


const densityPipeline = device.createComputePipeline({
  layout: computePipelineLayout,
  compute: {
    module: device.createShaderModule({
      code: particleWGSL,
    }),
    entryPoint: 'compute_density_pressure',
  },
});

const forcesPipeline = device.createComputePipeline({
  layout: computePipelineLayout,
  compute: {
    module: device.createShaderModule({
      code: particleWGSL,
    }),
    entryPoint: 'compute_forces',
  },
});

const integratePipeline = device.createComputePipeline({
  layout: computePipelineLayout,
  compute: {
    module: device.createShaderModule({
      code: particleWGSL,
    }),
    entryPoint: 'integrate',
  },
});




const computeBindGroup = device.createBindGroup({
  layout: computeBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: {
        buffer: simulationUBOBuffer,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: particlesBuffer,
        offset: 0,
        size: numParticles * particleInstanceByteSize,
      },
    },
    {
      binding: 2,
      resource: {
        buffer: sphDataBuffer,
        offset: 0,
        size: numParticles * sphDataSize, //size?
      },
    }
  ],
});

const aspect = canvas.width / canvas.height;
const projection = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);
const view = mat4.create();
const mvp = mat4.create();

function updateSimParameters()
{
 device.queue.writeBuffer(
    simulationUBOBuffer,
    0,
    new Float32Array([
      simulationParams.simulate ? simulationParams.deltaTime : 0.0,
      simulationParams.bounce,
      simulationParams.count >>> 0,
      simulationParams.gravity, 
      simulationParams.smoothing_radius,
      simulationParams.smoothing_radius2,
      simulationParams.mass,
      simulationParams.rest_density,
      simulationParams.pressure_constant,
      simulationParams.viscosity_constant,

      0.0,
      0.0,

      boxMin.x,
      boxMin.y,
      boxMin.z,
      0.0, // pad

      boxMax.x,
      boxMax.y,
      boxMax.z,
      0.0, // pad
      
      simulationParams.poly6_constant,
      simulationParams.spiky_constant,

      0.0,
      0.0,
    ])
  );
}
function frame() {

  updateSimParameters();

 

  mat4.identity(view);
  mat4.translate(view, vec3.fromValues(0, 0, -3), view);
  mat4.rotateX(view, Math.PI * -0.2, view);
  mat4.multiply(projection, view, mvp);

  // prettier-ignore
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([
      // modelViewProjectionMatrix
      mvp[0], mvp[1], mvp[2], mvp[3],
      mvp[4], mvp[5], mvp[6], mvp[7],
      mvp[8], mvp[9], mvp[10], mvp[11],
      mvp[12], mvp[13], mvp[14], mvp[15],

      view[0], view[4], view[8], // right

      0, // padding

      view[1], view[5], view[9], // up

      0, // padding
    ])
  );
  const swapChainTexture = context.getCurrentTexture();
  // prettier-ignore
  renderPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();

  const commandEncoder = device.createCommandEncoder();
  { 

    // computes densities + pressures
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(densityPipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
    
    // computes forces
    computePass.setPipeline(forcesPipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
    
    // integration pass
    computePass.setPipeline(integratePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));

    //end
    computePass.end();
  }
  {
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.setVertexBuffer(0, particlesBuffer);
    passEncoder.setVertexBuffer(1, quadVertexBuffer);
    passEncoder.draw(6, numParticles, 0, 0);
    passEncoder.end(); 
  }

  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(frame);
}
configureContext();
requestAnimationFrame(frame);

function assert(cond: boolean, msg = '') {
  if (!cond) {
    throw new Error(msg);
  }
}
