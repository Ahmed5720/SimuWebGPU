# SimuWebGPU

This project is a port of my OpenGL particle simulator to WebGPU. OpenGL is pretty archaic technology by today's standards and I am beggining to be convinced of the wonders of cross platform gpu development on the web. Because of this, I have been spending a good amount of time learning webgpu and I plan to continiously update this project as I learn new things. 

### features (so far):
1. An implementation of smoothed particle hydrodynamics in a compute shader (n^2) complexity.
1. GUI to tune parameters
2. particles are instance renderered
3. particles rendered as billboards (quads that always face the camera to give the appearance of a 3D sphere)
4. fragment shader turns quad into a particle by alpha clipping.

### future work (TBD):
1. An optimized SPH implementation using a spatial hashgrid for efficient niehborhood search.
2. interativity with mouse cursor as a force field
3. A GPU implementation of Barnes-hut for N-body simulations
> [!Note]
> how to run:

```bash
npm install
npm i wgpu-matrix dat.gui
npm i -D @types/dat.gui @webgpu/types
npm run dev
```

![](public/assets/img/thumb.png)
