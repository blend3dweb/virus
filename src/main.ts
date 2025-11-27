import * as THREE from 'three/webgpu'
import { color, Fn, If, Return, instancedArray, instanceIndex, uniform, select, attribute, uint, Loop, float,
transformNormalToView, cross, triNoise3D, time } from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import WebGPU from 'three/addons/capabilities/WebGPU.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

// Параметры
const PARTICLES_TOTAL = 16; // Уменьшено для производительности
const ROTATION_SPEED = 0.3;
const EXTENSION_SPEED = 2.0;
const EXTENSION_FREQUENCY = 0.5;
const MAX_STEPS = 32;

// Структура для частицы
class ParticleData {
    constructor(
        public position: THREE.Vector3,
        public rotationAxis: THREE.Vector3,
        public extensionSpeed: number,
        public phase: number
    ) {}
}

// Создание частиц
function createParticles(): ParticleData[] {
    const particles: ParticleData[] = [];

    for (let i = 0; i < PARTICLES_TOTAL; i++) {
        const radius = 1.5 + Math.random() * 0.5;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        const position = new THREE.Vector3(
            radius * Math.sin(phi) * Math.cos(theta),
            radius * Math.sin(phi) * Math.sin(theta),
            radius * Math.cos(phi)
        );

        const rotationAxis = new THREE.Vector3(
            (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 2
        ).normalize();

        const extensionSpeed = 0.5 + Math.random() * 2.0;
        const phase = Math.random() * Math.PI * 2;

        particles.push(new ParticleData(position, rotationAxis, extensionSpeed, phase));
    }

    return particles;
}

// Функция вращения вокруг оси
const rotAxis = Fn(([axis, angle]) => {
    const normalized_axis = normalize(axis);
    const s = sin(angle);
    const c = cos(angle);
    const oc = float(1.0).sub(c);

    return mat3(
        oc.mul(normalized_axis.x).mul(normalized_axis.x).add(c),
        oc.mul(normalized_axis.x).mul(normalized_axis.y).sub(normalized_axis.z.mul(s)),
        oc.mul(normalized_axis.z).mul(normalized_axis.x).add(normalized_axis.y.mul(s)),
        
        oc.mul(normalized_axis.x).mul(normalized_axis.y).add(normalized_axis.z.mul(s)),
        oc.mul(normalized_axis.y).mul(normalized_axis.y).add(c),
        oc.mul(normalized_axis.y).mul(normalized_axis.z).sub(normalized_axis.x.mul(s)),
        
        oc.mul(normalized_axis.z).mul(normalized_axis.x).sub(normalized_axis.y.mul(s)),
        oc.mul(normalized_axis.y).mul(normalized_axis.z).add(normalized_axis.x.mul(s)),
        oc.mul(normalized_axis.z).mul(normalized_axis.z).add(c)
    );
});

// SDF для капли
const sdDroplet = Fn(([p, a, b, r]) => {
    const pa = p.sub(a);
    const ba = b.sub(a);
    const h = clamp(dot(pa, ba).div(dot(ba, ba)), 0.0, 1.0);
    return length(pa.sub(ba.mul(h))).sub(r);
});

// Гладкое объединение
const opSmoothUnion = Fn(([d1, d2, k]) => {
    const h = clamp(float(0.5).add(float(0.5).mul(d2.sub(d1)).div(k)), 0.0, 1.0);
    return mix(d2, d1, h).sub(k.mul(h).mul(float(1.0).sub(h)));
});

// Получение текущей позиции частицы
const getCurrentParticlePosition = Fn(([basePos, rotationAxis, extensionSpeed, phaseOffset, timeValue]) => {
    const rotationAngle = timeValue.mul(ROTATION_SPEED).add(phaseOffset);
    const rotationMatrix = rotAxis(rotationAxis, rotationAngle);
    const rotatedPos = rotationMatrix.mul(basePos);

    const extensionFactor = sin(timeValue.mul(EXTENSION_FREQUENCY).add(phaseOffset)).add(1.0).mul(0.5);
    const extensionDistance = extensionFactor.mul(extensionSpeed);
    const directionFromCenter = normalize(rotatedPos);
    const extendedPos = rotatedPos.add(directionFromCenter.mul(extensionDistance));

    return extendedPos;
});

// Основной шейдер
const fragmentShader = Fn(() => {
    const resolution = uniform(vec2(window.innerWidth, window.innerHeight));
    const timeUniform = uniform(float(0));

    const fragCoord = vec2(THREE.fragmentCoordinate.xy);
    const timeValue = timeUniform;
    const uv = fragCoord.mul(2.0).sub(resolution).div(min(resolution.x, resolution.y));

    let ro = vec3(0.0, 0.0, -8.0);
    let rd = normalize(vec3(uv, 2.0));

    const rotationY = rotAxis(vec3(0.0, 1.0, 0.0), timeValue.mul(0.2));
    const rotationX = rotAxis(vec3(1.0, 0.0, 0.0), timeValue.mul(0.1));
    const totalRotation = rotationY.mul(rotationX);

    ro = totalRotation.mul(ro);
    rd = totalRotation.mul(rd);

    let t = float(0.0);
    let p = vec3(0.0);
    let hit = float(0.0);

    Loop({ start: 0, end: MAX_STEPS, type: 'uint' }, ({ i }) => {
        p = ro.add(rd.mul(t));
        
        // Базовая сфера
        let dist = length(p).sub(1.5);
        
        // Добавляем капли
        Loop({ start: 0, end: PARTICLES_TOTAL, type: 'uint' }, ({ i }) => {
            const particleIndex = float(i);
            const basePos = vec3(
                sin(particleIndex.mul(2.39)).mul(1.5),
                cos(particleIndex.mul(1.97)).mul(1.5),
                sin(particleIndex.mul(1.71)).mul(1.5)
            );
            const rotationAxis = normalize(basePos);
            const extensionSpeed = float(0.3).add(sin(particleIndex.mul(3.14)).mul(0.2).add(1.0).mul(0.5));
            const phaseOffset = particleIndex.mul(2.0);
            
            const particlePos = getCurrentParticlePosition(basePos, rotationAxis, extensionSpeed, phaseOffset, timeValue);
            const dropletDist = sdDroplet(p, vec3(0.0), particlePos, float(0.3));
            dist = opSmoothUnion(dist, dropletDist, float(0.2));
        });
        
        If(dist.lessThan(0.001), () => {
            hit = float(1.0);
            Break();
        });
        If(t.greaterThan(20.0), () => {
            Break();
        });
        t = t.add(dist.mul(0.8));
    });

    If(hit.equal(1.0), () => {
        const eps = float(0.001);
        const normal = normalize(vec3(
            length(p.add(vec3(eps, 0.0, 0.0))).sub(length(p.sub(vec3(eps, 0.0, 0.0)))),
            length(p.add(vec3(0.0, eps, 0.0))).sub(length(p.sub(vec3(0.0, eps, 0.0)))),
            length(p.add(vec3(0.0, 0.0, eps))).sub(length(p.sub(vec3(0.0, 0.0, eps))))
        ));

        const colorMix = clamp(length(p).div(2.0).sub(0.5), 0.0, 1.0);
        const colorValue = mix(vec3(0.0, 0.8, 1.0), vec3(0.2, 0.2, 0.8), colorMix);
        const specular = pow(max(float(0.0), dot(normal, normalize(rd.negate()))), 8.0);
        
        Return(vec4(colorValue.add(vec3(1.0).mul(specular)), 1.0));
    }, () => {
        const bgColor = mix(
            vec3(0.05, 0.1, 0.2),
            vec3(0.0, 0.3, 0.6),
            length(uv).div(3.0)
        );
        Return(vec4(bgColor, 1.0));
    });
});

class FPSCounter {
    private frameCount = 0;
    private lastTime = performance.now();
    private fps = 0;
    private frameTimes: number[] = [];

    update(): number {
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
        return Math.round(1000 / avgDeltaTime);
    }
}

async function init() {
    try {
        // Создание сцены
        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        // Создание рендерера
        const renderer = new WebGPURenderer({ 
            antialias: true,
            powerPreference: 'high-performance'
        });
        
        await renderer.init();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

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
            
            // Обновляем uniform времени
            const timeUniform = fragmentShader.getUniform('timeUniform');
            if (timeUniform) {
                timeUniform.value = currentTime;
            }

            renderer.render(scene, camera);

            const fps = fpsCounter.update();
            if (infoElement) {
                infoElement.textContent = `FPS: ${fps} | Particles: ${PARTICLES_TOTAL} | Steps: ${MAX_STEPS}`;
            }
        }

        renderer.setAnimationLoop(animate);

        // Обработчик изменения размера
        function onWindowResize() {
            renderer.setSize(window.innerWidth, window.innerHeight);
            
            // Обновляем uniform разрешения
            const resolutionUniform = fragmentShader.getUniform('resolution');
            if (resolutionUniform) {
                resolutionUniform.value.set(window.innerWidth, window.innerHeight);
            }
        }

        window.addEventListener('resize', onWindowResize);

    } catch (error) {
        console.error('Error initializing WebGPU:', error);
        const infoElement = document.getElementById('info');
        if (infoElement) {
            infoElement.textContent = 'Error: ' + (error as Error).message;
        }
    }
}

// Проверка поддержки WebGPU
async function checkWebGPUSupport() {
    try {
        if (navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                await init();
            } else {
                throw new Error('No suitable GPU adapter found');
            }
        } else {
            throw new Error('WebGPU not supported in this browser');
        }
    } catch (error) {
        console.error('WebGPU support check failed:', error);
        const infoElement = document.getElementById('info');
        if (infoElement) {
            infoElement.textContent = 'WebGPU not supported: ' + (error as Error).message;
        }
    }
}

// Запуск приложения
checkWebGPUSupport();