const PARTICLES_TOTAL = 64;

async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('No appropriate GPUAdapter found');
    }

    const device = await adapter.requestDevice();
    const canvas = document.getElementById('webgpu-canvas');
    const context = canvas.getContext('webgpu');

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    return { device, context, canvas, canvasFormat };
}

function createParticleBuffer(device) {
    // Создаем буфер для частиц
    const particleData = new Float32Array(PARTICLES_TOTAL * 4); // xyz + padding
    for (let i = 0; i < PARTICLES_TOTAL; i++) {
        // Генерируем случайные позиции для частиц
        particleData[i * 4 + 0] = (Math.random() - 0.5) * 8; // x
        particleData[i * 4 + 1] = (Math.random() - 0.5) * 8; // y
        particleData[i * 4 + 2] = (Math.random() - 0.5) * 8; // z
        particleData[i * 4 + 3] = 0; // padding
    }

    const particleBuffer = device.createBuffer({
        size: particleData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    new Float32Array(particleBuffer.getMappedRange()).set(particleData);
    particleBuffer.unmap();

    return particleBuffer;
}

function createBindGroupLayout(device) {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'read-only-storage' }
            }
        ]
    });
}

function createPipelineLayout(device, bindGroupLayout) {
    return device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });
}

function createShaderModule(device) {
    const shaderCode = `
        struct Uniforms {
            resolution: vec2<f32>,
            time: f32,
        };

        struct Particle {
            position: vec3<f32>,
            padding: f32,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read> particles: array<Particle>;

        const particlesTotal: u32 = ${PARTICLES_TOTAL}u;

        fn rot(a: f32) -> mat3x3<f32> {
            return mat3x3<f32>(
                vec3<f32>(cos(a), sin(a/2.0)*sin(a), sin(a)*cos(a/2.0)),
                vec3<f32>(0.0, cos(a/2.0), -sin(a/2.0)),
                vec3<f32>(-sin(a), sin(a/2.0)*cos(a), cos(a/2.0)*cos(a))
            );
        }

        fn sdDroplet(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
            let pa = p - a;
            let ba = b - a;
            let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            return length(pa - ba * h) - r * h;
        }

        fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
            let k_scaled = k * 4.0;
            let h = max(k_scaled - abs(d1 - d2), 0.0);
            return min(d1, d2) - h * h * 0.25 / k_scaled;
        }

        fn SDF(p: vec3<f32>) -> f32 {
            var sphere = length(p) - 2.0;

            for (var i: u32 = 0u; i < particlesTotal; i = i + 1u) {
                let pos = particles[i].position;
                sphere = opSmoothUnion(sphere, sdDroplet(p, vec3<f32>(0.0, 0.0, 0.0), pos, 0.1), 0.175);
            }

            return sphere;
        }

        fn getColor(p: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
            let eps = 0.001;
            let normal = normalize(vec3<f32>(
                SDF(p + vec3<f32>(eps, 0.0, 0.0)) - SDF(p - vec3<f32>(eps, 0.0, 0.0)),
                SDF(p + vec3<f32>(0.0, eps, 0.0)) - SDF(p - vec3<f32>(0.0, eps, 0.0)),
                SDF(p + vec3<f32>(0.0, 0.0, eps)) - SDF(p - vec3<f32>(0.0, 0.0, eps))
            ));

            let color_mix = clamp(length(p) / 1.0 - 2.0, 0.0, 1.0);
            let color = mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), color_mix);
            let specular = pow(max(0.0, dot(normal, normalize(-rd))), 4.0);
            return 1.0 - color * specular;
        }

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
            var pos = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>( 1.0,  1.0),
                vec2<f32>(-1.0,  1.0)
            );
            return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
        }

        @fragment
        fn fragmentMain(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
            let time = uniforms.time / 2.0 + 1.0;
            let uv = (fragCoord.xy * 2.0 - uniforms.resolution) / min(uniforms.resolution.x,
uniforms.resolution.y);

            var ro = vec3<f32>(0.0, 0.0, -12.0);
            var rd = normalize(vec3<f32>(uv, 3.0));

            let rotation = rot(time);
            ro = rotation * ro;
            rd = rotation * rd;

            var t = 0.0;
            var p = vec3<f32>(0.0, 0.0, 0.0);

            for (var i: u32 = 0u; i < 64u; i = i + 1u) {
                p = ro + rd * t;
                let d = SDF(p);
                if (d < 0.001) {
                    break;
                }
                if (t > 20.0) {
                    break;
                }
                t += d;
            }

            if (t < 20.0) {
                return vec4<f32>(getColor(p, rd), 1.0);
            } else {
                let bg_color = mix(
                    vec3<f32>(1.0, 1.0, 1.0),
                    vec3<f32>(0.1, 0.5, 0.7),
                    length(uv) / 2.0
                );
                return vec4<f32>(bg_color, 1.0);
            }
        }
    `;

    return device.createShaderModule({
        code: shaderCode
    });
}

function createRenderPipeline(device, pipelineLayout, shaderModule, canvasFormat) {
    return device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain'
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: canvasFormat
            }]
        },
        primitive: {
            topology: 'triangle-list'
        }
    });
}

function createUniformBuffer(device) {
    const uniformBufferSize = 4 * 4; // vec2 + f32 + padding
    return device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

function createBindGroup(device, bindGroupLayout, uniformBuffer, particleBuffer) {
    return device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: particleBuffer }
            }
        ]
    });
}

async function main() {
    try {
        const { device, context, canvas } = await initWebGPU();

        const particleBuffer = createParticleBuffer(device);
        const bindGroupLayout = createBindGroupLayout(device);
        const pipelineLayout = createPipelineLayout(device, bindGroupLayout);
        const shaderModule = createShaderModule(device);
        const renderPipeline = createRenderPipeline(device, pipelineLayout, shaderModule,
navigator.gpu.getPreferredCanvasFormat());
        const uniformBuffer = createUniformBuffer(device);
        const bindGroup = createBindGroup(device, bindGroupLayout, uniformBuffer, particleBuffer);

        let startTime = Date.now();

        function updateUniforms() {
            const time = (Date.now() - startTime) / 1000;
            const uniformData = new Float32Array([
                canvas.width, canvas.height,  // resolution
                time,                          // time
                0                              // padding
            ]);

            device.queue.writeBuffer(uniformBuffer, 0, uniformData);
        }

        function render() {
            updateUniforms();

            const commandEncoder = device.createCommandEncoder();
            const textureView = context.getCurrentTexture().createView();

            const renderPassDescriptor = {
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }]
            };

            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(renderPipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.draw(6);
            passEncoder.end();

            device.queue.submit([commandEncoder.finish()]);

            requestAnimationFrame(render);
        }

        render();
    } catch (error) {
        console.error('Error initializing WebGPU:', error);
    }
}

main();