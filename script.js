'use strict'


const canvas = document.querySelector('#canvas')

const gpu = navigator.gpu
if (!gpu) throw new Error('Please view with WebGPU-enabled browsers and devices.')

const adapterResult = async (device) => {
    const context = canvas.getContext('webgpu')
    const presentationFormat = gpu.getPreferredCanvasFormat()
    context.configure({
        device: device,
        format: presentationFormat,
    })

    // Position, Normal, Binormal, Coordinate
    const vertex = new Float32Array([
        -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,   0.0,  1.0,  0.0,   0.0, 1.0, // Front
         0.5, -0.5, -0.5,   0.0,  0.0, -1.0,   0.0,  1.0,  0.0,   1.0, 1.0,
        -0.5,  0.5, -0.5,   0.0,  0.0, -1.0,   0.0,  1.0,  0.0,   0.0, 0.0,
         0.5,  0.5, -0.5,   0.0,  0.0, -1.0,   0.0,  1.0,  0.0,   1.0, 0.0,
        
         0.5, -0.5,  0.5,   0.0,  0.0,  1.0,   0.0,  1.0,  0.0,   0.0, 1.0, // Back
        -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,   0.0,  1.0,  0.0,   1.0, 1.0,
         0.5,  0.5,  0.5,   0.0,  0.0,  1.0,   0.0,  1.0,  0.0,   0.0, 0.0,
        -0.5,  0.5,  0.5,   0.0,  0.0,  1.0,   0.0,  1.0,  0.0,   1.0, 0.0,
         
        -0.5, -0.5,  0.5,  -1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   0.0, 1.0, // Left
        -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   1.0, 1.0,
        -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   0.0, 0.0,
        -0.5,  0.5, -0.5,  -1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   1.0, 0.0,
         
         0.5, -0.5, -0.5,   1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   0.0, 1.0, // Right
         0.5, -0.5,  0.5,   1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   1.0, 1.0,
         0.5,  0.5, -0.5,   1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   0.0, 0.0,
         0.5,  0.5,  0.5,   1.0,  0.0,  0.0,   0.0,  1.0,  0.0,   1.0, 0.0,
         
        -0.5, -0.5,  0.5,   0.0, -1.0,  0.0,   0.0,  0.0, -1.0,   0.0, 1.0, // Bottom
         0.5, -0.5,  0.5,   0.0, -1.0,  0.0,   0.0,  0.0, -1.0,   1.0, 1.0,
        -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,   0.0,  0.0, -1.0,   0.0, 0.0,
         0.5, -0.5, -0.5,   0.0, -1.0,  0.0,   0.0,  0.0, -1.0,   1.0, 0.0,
         
        -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,   0.0,  0.0,  1.0,   0.0, 1.0, // Top
         0.5,  0.5, -0.5,   0.0,  1.0,  0.0,   0.0,  0.0,  1.0,   1.0, 1.0,
        -0.5,  0.5,  0.5,   0.0,  1.0,  0.0,   0.0,  0.0,  1.0,   0.0, 0.0,
         0.5,  0.5,  0.5,   0.0,  1.0,  0.0,   0.0,  0.0,  1.0,   1.0, 0.0,
         
    ])
    
    const index = new Uint16Array([
        0, 1, 2,
        3, 2, 1,
        
        4, 5, 6,
        7, 6, 5,
        
        8, 9, 10,
        11, 10, 9,
        
        12, 13, 14,
        15, 14, 13,
        
        16, 17, 18,
        19, 18, 17,
        
        20, 21, 22,
        23, 22, 21,
    ])

    const vertexBuffer = device.createBuffer({
        label: "Cube",
        size: vertex.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(vertexBuffer, 0, vertex)

    const vertexBufferLayout = {
        arrayStride: 11 * 4,
        attributes: [
            {
                format: "float32x3",
                offset: 0 * 4,
                shaderLocation: 0,
            },
            {
                format: "float32x3",
                offset: 3 * 4,
                shaderLocation: 1,
            },
            {
                format: "float32x3",
                offset: 6 * 4,
                shaderLocation: 2,
            },
            {
                format: "float32x2",
                offset: 9 * 4,
                shaderLocation: 3,
            },
        ],
    }
    
    const indexBuffer = device.createBuffer({
        size: index.byteLength,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
    })
    new Uint16Array(indexBuffer.getMappedRange()).set(index)
    indexBuffer.unmap()

    // Image
    const normalImage = document.createElement('img');
    normalImage.crossOrigin = 'Anonymous';
    normalImage.src = './normal.png';
    await normalImage.decode();
    const normalBitmap = await createImageBitmap(normalImage);

    const texture = device.createTexture({
        size: [normalBitmap.width, normalBitmap.height, 1],
        format: 'rgba8unorm',
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
        { source: normalBitmap },
        { texture: texture },
        [normalBitmap.width, normalBitmap.height]
    );

    // Sampler
    const sampler = device.createSampler({
        magFilter: 'nearest',
        minFilter: 'nearest',
    });

    // ShaderModule
    const shaderModule = device.createShaderModule({
        label: "Normal Pixelart Cube shader",
        code: `
            struct Uniforms {
                light : vec3<f32>,
                matrix : mat4x4<f32>,
                normalMatrix : mat4x4<f32>
            }
            @binding(0) @group(0) var<uniform> uniforms : Uniforms;
            @binding(1) @group(0) var normalTexture: texture_2d<f32>;
            @binding(2) @group(0) var normalSampler: sampler;
            
            struct VertexOutput {
                @builtin(position) Position : vec4<f32>,
                @location(0) vertexNormal: vec3f,
                @location(1) vertexBinormal: vec3f,
                @location(2) vertexCoordinate: vec2f
            }
            
            @vertex
            fn vertexMain(
                @location(0) vertexPosision: vec3f,
                @location(1) vertexNormal: vec3f,
                @location(2) vertexBinormal: vec3f,
                @location(3) vertexCoordinate: vec2f
            ) -> VertexOutput {
                var output : VertexOutput;
                output.Position = uniforms.matrix * vec4f(vertexPosision, 1.0);
                output.vertexNormal = vertexNormal;
                output.vertexBinormal = vertexBinormal;
                output.vertexCoordinate = vertexCoordinate;
                return output;
            }
            
            @fragment
            fn fragmentMain(
                @location(0) vertexNormal: vec3f,
                @location(1) vertexBinormal: vec3f,
                @location(2) vertexCoordinate: vec2f
            ) -> @location(0) vec4f {
                var output : VertexOutput;

                var vertexTangent : vec3f = cross(vertexNormal, vertexBinormal);

                var normal4 : vec4f = uniforms.normalMatrix * vec4f(vertexNormal, 0.0);
                var binormal4 : vec4f = uniforms.normalMatrix * vec4f(vertexBinormal, 0.0);
                var tangent4 : vec4f = uniforms.normalMatrix * vec4f(vertexTangent, 0.0);

                var normal : vec3f = vec3f(normal4.x, normal4.y, normal4.z);
                var binormal : vec3f = vec3f(binormal4.x, binormal4.y, binormal4.z);
                var tangent : vec3f = vec3f(tangent4.x, tangent4.y, tangent4.z);

                var normalColor : vec4f = textureSample(normalTexture, normalSampler, vertexCoordinate);
                normalColor.r = (normalColor.r - 0.5) * 2.0;
                normalColor.g = (normalColor.g - 0.5) * 2.0;
                normalColor.b = (normalColor.b - 0.5) * 2.0;

                var n : vec3f = vec3f(0.0, 0.0, 0.0);
                n += normalColor.r * tangent;
                n += normalColor.g * binormal;
                n += normalColor.b * normal;
                n = normalize(n);

                var l : f32 = max(dot(n, normalize(uniforms.light)), 0.0);
                
                return vec4f(l, l, l, 1.0);
            }
        `
    })
    
    // Pipeline
    const renderPipeline = device.createRenderPipeline({
        label: "Normal Pixelart Cube pipeline",
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vertexMain",
            buffers: [
                vertexBufferLayout,
            ]
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fragmentMain",
            targets: [{
                format: presentationFormat
            }]
        },
        primitive: {
            topology: 'triangle-list',
            frontFace: 'ccw',
            cullMode: 'back',
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
    })
    
    /*
    // Resize Window
    let depthTexture
    const resize = () => {
        const r = window.devicePixelRatio
        const w = window.innerWidth * r
        const h = window.innerHeight * r
        if (w / h <= 1 / 1) {
            canvas.width = w
            canvas.height = w
        }
        if (w / h >  1 / 1) {
            canvas.width = h
            canvas.height = h
        }
        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })
    }
    resize()
    window.addEventListener('resize', resize)
    */

    // DepthTexture
    const depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    
    const setScale = (x, y, z) => {
        const m = new Float32Array([
            x,
            0,
            0,
            0,
            
            0,
            y,
            0,
            0,
            
            0,
            0,
            z,
            0,
            
            0,
            0,
            0,
            1,
        ])
        return m
    }
    
    const setTranslateZ = (near) => {
        const m = new Float32Array([
            1,
            0,
            0,
            0,
            
            0,
            1,
            0,
            0,
            
            0,
            0,
            1,
            0,
            
            0,
            0,
            near,
            1,
        ])
        return m
    }
    
    const setRotateX = (radian) => {
        const m = new Float32Array([
            1,
            0,
            0,
            0,

            0,
            Math.cos(radian),
            Math.sin(radian),
            0,

            0,
            -Math.sin(radian),
            Math.cos(radian),
            0,

            0,
            0,
            0,
            1,
        ])
        return m
    }
    
    const setRotateY = (radian) => {
        const m = new Float32Array([
            Math.cos(radian),
            0,
            -Math.sin(radian),
            0,

            0,
            1,
            0,
            0,

            Math.sin(radian),
            0,
            Math.cos(radian),
            0,

            0,
            0,
            0,
            1,
        ])
        return m
    }
    
    const multiply = (l, r) => {
        const m = new Float32Array([
            l[ 0] * r[ 0] + l[ 4] * r[ 1] + l[ 8] * r[ 2] + l[12] * r[ 3],
            l[ 1] * r[ 0] + l[ 5] * r[ 1] + l[ 9] * r[ 2] + l[13] * r[ 3],
            l[ 2] * r[ 0] + l[ 6] * r[ 1] + l[10] * r[ 2] + l[14] * r[ 3],
            l[ 3] * r[ 0] + l[ 7] * r[ 1] + l[11] * r[ 2] + l[15] * r[ 3],
            
            l[ 0] * r[ 4] + l[ 4] * r[ 5] + l[ 8] * r[ 6] + l[12] * r[ 7],
            l[ 1] * r[ 4] + l[ 5] * r[ 5] + l[ 9] * r[ 6] + l[13] * r[ 7],
            l[ 2] * r[ 4] + l[ 6] * r[ 5] + l[10] * r[ 6] + l[14] * r[ 7],
            l[ 3] * r[ 4] + l[ 7] * r[ 5] + l[11] * r[ 6] + l[15] * r[ 7],
            
            l[ 0] * r[ 8] + l[ 4] * r[ 9] + l[ 8] * r[10] + l[12] * r[11],
            l[ 1] * r[ 8] + l[ 5] * r[ 9] + l[ 9] * r[10] + l[13] * r[11],
            l[ 2] * r[ 8] + l[ 6] * r[ 9] + l[10] * r[10] + l[14] * r[11],
            l[ 3] * r[ 8] + l[ 7] * r[ 9] + l[11] * r[10] + l[15] * r[11],
            
            l[ 0] * r[12] + l[ 4] * r[13] + l[ 8] * r[14] + l[12] * r[15],
            l[ 1] * r[12] + l[ 5] * r[13] + l[ 9] * r[14] + l[13] * r[15],
            l[ 2] * r[12] + l[ 6] * r[13] + l[10] * r[14] + l[14] * r[15],
            l[ 3] * r[12] + l[ 7] * r[13] + l[11] * r[14] + l[15] * r[15],
        ])
        return m
    }
    
    let light = new Float32Array([0.0, 0.5, -1.0])
    
    let matrix = new Float32Array([
        1,
        0,
        0,
        0,
        
        0,
        1,
        0,
        0,
        
        0,
        0,
        1,
        0,
        
        0,
        0,
        0,
        1,
    ])
    
    let normalMatrix = new Float32Array([
        1,
        0,
        0,
        0,
        
        0,
        1,
        0,
        0,
        
        0,
        0,
        1,
        0,
        
        0,
        0,
        0,
        1,
    ])
    
    const translateZMatrix = setTranslateZ(0.5)
    const scaleMatrix = setScale(0.5, 0.5, 0.5)
    matrix = multiply(matrix, translateZMatrix)
    matrix = multiply(matrix, scaleMatrix)

    // UniformBuffer
    const uniformBuffer = device.createBuffer({
        label: "Uniform",
        size: (4 + 16 + 16) * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(uniformBuffer, 0 * 4, light)
    
    // BindBroup
    const bindGroup = device.createBindGroup({
        label: "BindGroup",
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: texture.createView(),
            },
            {
                binding: 2,
                resource: sampler,
            },
        ],
    })
    
    const renderPassDescriptor = {
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        },
    }
    
    let prevTimestamp = 0
    const frame = (timestamp) => {
        if (!prevTimestamp) prevTimestamp = timestamp
        const deltaTime = (timestamp - prevTimestamp) / 1000
        
        requestAnimationFrame(frame)

        const rotateX = setRotateX(deltaTime / 12 * Math.PI * 2)
        const rotateY = setRotateY(deltaTime / 8 * Math.PI * 2)
        
        matrix = multiply(matrix, rotateX)
        matrix = multiply(matrix, rotateY)
        normalMatrix = multiply(normalMatrix, rotateX)
        normalMatrix = multiply(normalMatrix, rotateY)
        
        device.queue.writeBuffer(uniformBuffer, 4 * 4, matrix)
        device.queue.writeBuffer(uniformBuffer, (4 + 16) * 4, normalMatrix)
        
        const commandEncoder = device.createCommandEncoder()

        renderPassDescriptor.colorAttachments[0].view =
            context
            .getCurrentTexture()
            .createView()
        
        const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor)

        renderPass.setPipeline(renderPipeline)
        renderPass.setVertexBuffer(0, vertexBuffer)
        renderPass.setBindGroup(0, bindGroup)
        renderPass.setIndexBuffer(indexBuffer, 'uint16')
        renderPass.drawIndexed(index.length)
        renderPass.end()
        
        device.queue.submit([commandEncoder.finish()])
        
        prevTimestamp = timestamp
    }
    requestAnimationFrame(frame)
};

const gpuResult = async (adapter) => {
    if (!adapter) throw new Error("No appropriate GPUAdapter found.")
    adapter.requestDevice().then(adapterResult)
};

gpu.requestAdapter().then(gpuResult)