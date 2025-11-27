import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import {
    Fn, float, vec2, vec3, vec4,
    uniform, storage,
    length, normalize, dot, cross,
    sin, cos, clamp, min, max, abs, pow,
    mix,
    Loop, If, Return,
    texture,
    time
} from 'three/tsl';

// Параметры
const PARTICLES_TOTAL = 32;
const ROTATION_SPEED = 0.3;
const EXTENSION_SPEED = 2.0;
const EXTENSION_FREQUENCY = 0.5;
const MAX_STEPS = 32;

// Создание частиц
function createParticles() {
    const particles = new Float32Array(PARTICLES_TOTAL * 8);

    for (let i = 0; i < PARTICLES_TOTAL; i++) {
        const radius = Math.random() * 2.0;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        particles[i * 8 + 0] = radius * Math.sin(phi) * Math.cos(theta);
        particles[i * 8 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        particles[i * 8 + 2] = radius * Math.cos(phi);

        particles[i * 8 + 3] = (Math.random() - 0.5) * 2;
        particles[i * 8 + 4] = (Math.random() - 0.5) * 2;
        particles[i * 8 + 5] = (Math.random() - 0.5) * 2;

        particles[i * 8 + 6] = 0.5 + Math.random() * 2.0;
        particles[i * 8 + 7] = 0;
    }

    return particles;
}

// Функция вращения вокруг оси
const rotAxis = Fn(([axis, angle]) => {
    const normalized_axis = normalize(axis);
    const s = sin(angle);
    const c = cos(angle);
    const oc = float(1.0).sub(c);

    return vec3(
        vec3(
            oc.mul(normalized_axis.x).mul(normalized_axis.x).add(c),
            oc.mul(normalized_axis.x).mul(normalized_axis.y).sub(normalized_axis.z.mul(s)),
            oc.mul(normalized_axis.z).mul(normalized_axis.x).add(normalized_axis.y.mul(s))
        ),
        vec3(
            oc.mul(normalized_axis.x).mul(normalized_axis.y).add(normalized_axis.z.mul(s)),
            oc.mul(normalized_axis.y).mul(normalized_axis.y).add(c),
            oc.mul(normalized_axis.y).mul(normalized_axis.z).sub(normalized_axis.x.mul(s))
        ),
        vec3(
            oc.mul(normalized_axis.z).mul(normalized_axis.x).sub(normalized_axis.y.mul(s)),
            oc.mul(normalized_axis.y).mul(normalized_axis.z).add(normalized_axis.x.mul(s)),
            oc.mul(normalized_axis.z).mul(normalized_axis.z).add(c)
        )
    );
});

// SDF для капли
const sdDroplet = Fn(([p, a, b, r]) => {
    const pa = p.sub(a);
    const ba = b.sub(a);
    const h = clamp(dot(pa, ba).div(dot(ba, ba)), 0.0, 1.0);
    return length(pa.sub(ba.mul(h))).sub(r.mul(h));
});

// Гладкое объединение
const opSmoothUnion = Fn(([d1, d2, k]) => {
    const k_scaled = k.mul(4.0);
    const h = max(k_scaled.sub(abs(d1.sub(d2))), 0.0);
    return min(d1, d2).sub(h.mul(h).mul(0.25).div(k_scaled));
});

// Получение текущей позиции частицы
const getCurrentParticlePosition = Fn(([particles, i, timeValue]) => {
    const particle = particles.element(i);
    const base_pos = particle.position;

    const rotation_angle = timeValue.mul(ROTATION_SPEED);
    const rotation_matrix = rotAxis(particle.rotation_axis, rotation_angle);
    const rotated_pos = rotation_matrix.mul(base_pos);

    const extension_factor = sin(timeValue.mul(EXTENSION_FREQUENCY)).add(1.0).mul(0.5);
    const extension_distance = extension_factor.mul(particle.extension_speed);
    const direction_from_center = normalize(rotated_pos);
    const extended_pos = rotated_pos.add(direction_from_center.mul(extension_distance));

    return extended_pos;
});

// SDF функция
const SDF = Fn(([p, particles, particlesTotal]) => {
    let sphere = length(p).sub(2.0);

    Loop({ start: 0, end: particlesTotal, type: 'uint' }, ({ i }) => {
        const pos = getCurrentParticlePosition(particles, i, time);
        sphere = opSmoothUnion(sphere, sdDroplet(p, vec3(0.0), pos, 0.1), 0.175);
    });

    Return(sphere);
});

// Функция цвета
const getColor = Fn(([p, rd]) => {
    const eps = 0.001;
    const normal = normalize(vec3(
        SDF(p.add(vec3(eps, 0.0, 0.0))).sub(SDF(p.sub(vec3(eps, 0.0, 0.0)))),
        SDF(p.add(vec3(0.0, eps, 0.0))).sub(SDF(p.sub(vec3(0.0, eps, 0.0)))),
        SDF(p.add(vec3(0.0, 0.0, eps))).sub(SDF(p.sub(vec3(0.0, 0.0, eps))))
    ));

    const color_mix = clamp(length(p).div(1.0).sub(2.0), 0.0, 1.0);
    const colorValue = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0), color_mix);
    const specular = pow(max(0.0, dot(normal, normalize(rd.negate()))), 4.0);
    Return(vec3(1.0).sub(colorValue.mul(specular)));
});

// Основной шейдер
const fragmentShader = Fn(() => {
    const resolution = uniform(vec2());
    const timeUniform = uniform(float());

    const fragCoord = THREE.fragmentPosition;
    const timeValue = timeUniform.div(2.0).add(1.0);
    const uv = fragCoord.xy.mul(2.0).sub(resolution).div(resolution.x.min(resolution.y));

    let ro = vec3(0.0, 0.0, -12.0);
    let rd = normalize(vec3(uv, 3.0));

    const rotation_y = rotAxis(vec3(0.0, 1.0, 0.0), timeValue.mul(0.5));
    const rotation_x = rotAxis(vec3(1.0, 0.0, 0.0), timeValue.mul(0.3));
    const total_rotation = rotation_y.mul(rotation_x);

    ro = total_rotation.mul(ro);
    rd = total_rotation.mul(rd);

    let t = float(0.0);
    let p = vec3(0.0);

    Loop({ start: 0, end: MAX_STEPS, type: 'uint' }, ({ i }) => {
        p = ro.add(rd.mul(t));
        const d = SDF(p);
        If(d.lessThan(0.001), () => {
            Break;
        });
        If(t.greaterThan(20.0), () => {
            Break;
        });
        t = t.add(d.mul(0.8));
    });

    If(t.lessThan(20.0), () => {
        Return(vec4(getColor(p, rd), 1.0));
    }, () => {
        const bg_color = mix(
            vec3(1.0),
            vec3(0.1, 0.5, 0.7),
            length(uv).div(2.0)
        );
        Return(vec4(bg_color, 1.0));
    });
});

class FPSCounter {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
        this.avgFps = 0;
        this.frameTimes = [];
    }

    update() {
        this.frameCount++;
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastTime;
        this.frameTimes.push(deltaTime);

        if (this.frameTimes.length > 60) {
            this.frameTimes.shift();
        }

        if (deltaTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / deltaTime);
            this.frameCount = 0;
            this.lastTime = currentTime;
        }

        const avgDeltaTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        this.avgFps = Math.round(1000 / avgDeltaTime);

        return this.avgFps;
    }
}

async function init() {
    // Создание сцены
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    // Создание рендерера
    const renderer = new WebGPURenderer({ antialias: true });
    await renderer.init(); // Важно: дождаться инициализации
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setAnimationLoop(animate);
    document.body.appendChild(renderer.domElement);

    // Создание частиц
    const particleData = createParticles();
    const particlesStorage = storage(particleData, 'vec4', PARTICLES_TOTAL * 2); // 2 vec4 на частицу

    // Создание uniform для разрешения
    const resolutionUniform = uniform(vec2(window.innerWidth, window.innerHeight));
    const timeUniform = uniform(float(0));

    // Создание материала с TSL шейдером
    const material = new THREE.MeshBasicMaterial();
    material.fragmentNode = fragmentShader();

    // Создание полноэкранного квадрата
    const geometry = new THREE.PlaneGeometry(2, 2);
    const quad = new THREE.Mesh(geometry, material);
    scene.add(quad);

    // FPS счетчик
    const fpsCounter = new FPSCounter();
    const infoElement = document.getElementById('info');

    let startTime = Date.now();

    function animate() {
        const currentTime = (Date.now() - startTime) / 1000;
        timeUniform.value = currentTime;

        renderer.render(scene, camera);

        const fps = fpsCounter.update();
        if (infoElement) {
            infoElement.textContent = `FPS: ${fps} | Particles: ${PARTICLES_TOTAL} | Steps: ${MAX_STEPS}`;
        }
    }

    // Обработчик изменения размера
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        resolutionUniform.value.set(window.innerWidth, window.innerHeight);
    }

    window.addEventListener('resize', onWindowResize);
}

// Проверка поддержки WebGPU
async function checkWebGPUSupport() {
    try {
        if (navigator.gpu) {
            await init();
        } else {
            const infoElement = document.getElementById('info');
            if (infoElement) {
                infoElement.textContent = 'WebGPU not supported in this browser';
            }
            console.error('WebGPU not supported');
        }
    } catch (error) {
        console.error('Error initializing WebGPU:', error);
        const infoElement = document.getElementById('info');
        if (infoElement) {
            infoElement.textContent = 'Error: ' + error.message;
        }
    }
}

// Запуск приложения
checkWebGPUSupport();