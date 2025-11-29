import { mat4, vec3 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import './style.css'
import particleWGSL from './particle.wgsl?raw';

const numParticles = 10000;
const particlePositionOffset = 0;
const particleColorOffset = 4 * 4;
const particleInstanceByteSize =
  3 * 4 + // position
  1 * 4 + // lifetime
  4 * 4 + // color
  3 * 4 + // velocity
  1 * 4 + // padding
  0;


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
    //toneMapping: { mode: simulationParams.toneMappingMode },
  });
  //hdrFolder.name = getHdrFolderName();
}


// buffer used in vertex and compute shader we also need to be able to copy into it with a staging buffer to initialize it.
const particlesBuffer = device.createBuffer({
  size: numParticles * particleInstanceByteSize,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

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
    initData[base + 11] = 0.0; //pad
    
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
const simulationParams = {
  simulate: true,
  deltaTime: 0.01,
  bounce: 0.7,
  count: numParticles,
  gravity: -9.8,
  toneMappingMode: 'standard' as GPUCanvasToneMappingMode,
  brightnessFactor: 1,
};

const simulationUBOBufferSize = 16 * 4; // why do we need 16 floats? whats the size of simulate & tonemappingmode? 1 byte for each?
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


const computePipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: device.createShaderModule({
      code: particleWGSL,
    }),
    entryPoint: 'simulate',
  },
});
const computeBindGroup = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(0),
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
  ],
});

const aspect = canvas.width / canvas.height;
const projection = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);
const view = mat4.create();
const mvp = mat4.create();

function frame() {

  

  device.queue.writeBuffer(
    simulationUBOBuffer,
    0,
    new Float32Array([
      simulationParams.simulate ? simulationParams.deltaTime : 0.0,
      simulationParams.bounce,
      0.0,
      0.0, // pad
      
      simulationParams.count,
      simulationParams.gravity, 
      0.0, // pad
      0.0,

      boxMin.x,
      boxMin.y,
      boxMin.z,
      0.0, // pad

      boxMax.x,
      boxMax.y,
      boxMax.z,
      0.0, // pad
    ])
  );

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
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, computeBindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numParticles / 64));
    passEncoder.end();
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
