const PARTICLES_TOTAL = 64;
const ROTATION_SPEED = 0.3;
const EXTENSION_SPEED = 2.0;
const EXTENSION_FREQUENCY = 0.5;

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
    // Создаем буфер для частиц с начальными позициями
    const particleData = new Float32Array(PARTICLES_TOTAL * 8); // 8 float значения на частицу

    for (let i = 0; i < PARTICLES_TOTAL; i++) {
        // Начальные позиции ближе к центру
        const radius = Math.random() * 2.0;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        particleData[i * 8 + 0] = radius * Math.sin(phi) * Math.cos(theta); // x
        particleData[i * 8 + 1] = radius * Math.sin(phi) * Math.sin(theta); // y
        particleData[i * 8 + 2] = radius * Math.cos(phi); // z

        // Оси вращения для рандомного движения
        particleData[i * 8 + 3] = (Math.random() - 0.5) * 2; // rotation x
        particleData[i * 8 + 4] = (Math.random() - 0.5) * 2; // rotation y
        particleData[i * 8 + 5] = (Math.random() - 0.5) * 2; // rotation z

        // Скорости расширения
        particleData[i * 8 + 6] = 0.5 + Math.random() * 2.0; // extension speed

        particleData[i * 8 + 7] = 0; // заполнение нулями
    }

    const particleBuffer = device.createBuffer({
        size: particleData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(particleBuffer, 0, particleData);

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
            deltaTime: f32,
        };

        struct Particle {
            position: vec3<f32>,
            rotation_axis: vec3<f32>,
            extension_speed: f32,
            padding: f32,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read> particles: array<Particle>;

        const particlesTotal: u32 = ${PARTICLES_TOTAL}u;
        const ROTATION_SPEED: f32 = ${ROTATION_SPEED}f;
        const EXTENSION_SPEED: f32 = ${EXTENSION_SPEED}f;
        const EXTENSION_FREQUENCY: f32 = ${EXTENSION_FREQUENCY}f;

        fn rotAxis(axis: vec3<f32>, angle: f32) -> mat3x3<f32> {
            let normalized_axis = normalize(axis);
            let s = sin(angle);
            let c = cos(angle);
            let oc = 1.0 - c;

            return mat3x3<f32>(
                oc * normalized_axis.x * normalized_axis.x + c,
                oc * normalized_axis.x * normalized_axis.y - normalized_axis.z * s,
                oc * normalized_axis.z * normalized_axis.x + normalized_axis.y * s,

                oc * normalized_axis.x * normalized_axis.y + normalized_axis.z * s,
                oc * normalized_axis.y * normalized_axis.y + c,
                oc * normalized_axis.y * normalized_axis.z - normalized_axis.x * s,

                oc * normalized_axis.z * normalized_axis.x - normalized_axis.y * s,
                oc * normalized_axis.y * normalized_axis.z + normalized_axis.x * s,
                oc * normalized_axis.z * normalized_axis.z + c
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

        fn getCurrentParticlePosition(i: u32) -> vec3<f32> {
            let particle = particles[i];
            let base_pos = particle.position;

            // Рандомное вращение
            let rotation_angle = uniforms.time * ROTATION_SPEED;
            let rotation_matrix = rotAxis(particle.rotation_axis, rotation_angle);
            let rotated_pos = rotation_matrix * base_pos;

            // Движение от центра (щупальца)
            let extension_factor = (sin(uniforms.time * EXTENSION_FREQUENCY) + 1.0) * 0.5;
            let extension_distance = extension_factor * particle.extension_speed;
            let direction_from_center = normalize(rotated_pos);
            let extended_pos = rotated_pos + direction_from_center * extension_distance;

            return extended_pos;
        }

        fn SDF(p: vec3<f32>) -> f32 {
            var sphere = length(p) - 2.0;

            for (var i: u32 = 0u; i < particlesTotal; i = i + 1u) {
                let pos = getCurrentParticlePosition(i);
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

            // Общее вращение сцены
            let rotation_y = rotAxis(vec3<f32>(0.0, 1.0, 0.0), time * 0.5);
            let rotation_x = rotAxis(vec3<f32>(1.0, 0.0, 0.0), time * 0.3);
            let total_rotation = rotation_y * rotation_x;

            ro = total_rotation * ro;
            rd = total_rotation * rd;

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
    const uniformBufferSize = 4 * 5; // vec2 + f32 + f32 + padding
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

class FPSCounter {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
    }

    update() {
        this.frameCount++;
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastTime;

        if (deltaTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / deltaTime);
            this.frameCount = 0;
            this.lastTime = currentTime;
        }

        return this.fps;
    }
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

        const fpsCounter = new FPSCounter();
        const infoElement = document.getElementById('info');

        let startTime = Date.now();
        let lastFrameTime = performance.now();

        function updateUniforms() {
            const currentTime = Date.now();
            const time = (currentTime - startTime) / 1000;
            const currentFrameTime = performance.now();
            const deltaTime = (currentFrameTime - lastFrameTime) / 1000;
            lastFrameTime = currentFrameTime;

            const uniformData = new Float32Array([
                canvas.width, canvas.height,  // resolution (vec2)
                time,                          // time (f32)
                deltaTime,                     // deltaTime (f32)
                0                              // заполнение (f32)
            ]);

            device.queue.writeBuffer(uniformBuffer, 0, uniformData);
        }

        function render() {
            updateUniforms();

            const fps = fpsCounter.update();
            infoElement.textContent = `FPS: ${fps}`;

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
        document.getElementById('info').textContent = 'Error: ' + error.message;
    }
}

main();