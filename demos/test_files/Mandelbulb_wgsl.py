import os
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
import wgpu


def render_rotation(
    config,
    frame_count=120,
    radius=4.5,
    height=1.5,
    output_directory="mandelbulb_frames_vert_300",
):
    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    target = config.camera_target

    for frame in range(frame_count):

        # 0.0 -> almost 360.0
        # We don't render 360 because it is
        # identical to frame 0.
        angle = (
            frame / frame_count
        ) * 360.0

        config.camera_position = orbit_camera(
            angle_degrees=angle,
            radius=radius,
            height=height,
            vertical_motion=0.35,
            target=target,
        )

        config.output_file = os.path.join(
            output_directory,
            f"frame_{frame:04d}.png",
        )

        print(
            f"\nFrame {frame + 1}/{frame_count}"
            f" | angle={angle:.1f}°"
            f" | camera={config.camera_position}"
        )

        render(config)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RenderConfig:
    # Output
    width: int = 1280
    height: int = 1024
    output_file: str = "mandelbulb.png"

    # Camera
    camera_position = (3.2, 2.2, 3.2)
    camera_target = (0.0, 0.0, 0.0)
    fov_degrees: float = 45.0

    # Mandelbulb
    power: float = 8.0
    fractal_iterations: int = 12
    bailout: float = 4.0

    # Ray marching
    max_steps: int = 400
    max_distance: float = 20.0
    hit_epsilon: float = 0.0005

    # Surface normal
    normal_epsilon: float = 0.001

    # Lighting
    light_direction = (-0.5, 0.8, -0.6)
    ambient_strength: float = 0.15
    diffuse_strength: float = 1.0
    specular_strength: float = 0.35
    shininess: float = 32.0

    # Shadows
    enable_shadows: bool = True
    shadow_softness: float = 12.0

    # Ambient occlusion
    enable_ao: bool = True
    ao_strength: float = 1.0

    # Coloring
    object_color = (0.95, 0.55, 0.12)
    background_top = (0.04, 0.06, 0.10)
    background_bottom = (0.005, 0.008, 0.015)

    # Gamma
    gamma: float = 2.2

def orbit_camera(
    angle_degrees: float,
    radius: float = 4.5,
    height: float = 1.5,
    vertical_motion: float = 0.35,
    target=(0.0, 0.0, 0.0),
):
    """
    Calculate a camera position that orbits around the target.

    angle_degrees:
        Rotation around the Mandelbulb.
        0 -> 360 gives one complete rotation.

    radius:
        Distance from the Mandelbulb.

    height:
        Camera height.

    target:
        Point that the camera orbits around.
    """
    angle = math.radians(angle_degrees)

    x = target[0] + radius * math.cos(angle)

    y = (
        target[1]
        + height
        + vertical_motion * math.sin(angle * 2.0)
    )

    z = target[2] + radius * math.sin(angle)

    return (x, y, z)

def create_gpu_device():
    """
    Create a WebGPU device in a platform-independent way.

    Expected backends:
        macOS   -> Metal
        Windows -> Vulkan / DirectX 12
        Linux   -> Vulkan

    The actual backend is selected by wgpu-native.
    """

    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to request a WebGPU adapter.\n"
            "Make sure your GPU drivers and wgpu installation "
            "are working correctly."
        ) from exc

    if adapter is None:
        raise RuntimeError(
            "No compatible WebGPU GPU adapter was found."
        )

    try:
        device = adapter.request_device_sync(
            required_features=[],
            required_limits={},
        )
    except Exception as exc:
        raise RuntimeError(
            "A GPU was found, but wgpu could not create "
            "a WebGPU device."
        ) from exc

    info = adapter.info

    print()
    print("=" * 60)
    print("WebGPU Renderer")
    print("=" * 60)

    for key in (
        "vendor",
        "architecture",
        "device",
        "description",
        "adapter_type",
        "backend_type",
    ):
        if key in info:
            print(f"{key:14}: {info[key]}")

    print("=" * 60)
    print()

    return adapter, device

# ============================================================
# WGSL COMPUTE SHADER
# ============================================================

WGSL_SHADER = r"""
struct Params {
    width: u32,
    height: u32,
    max_steps: u32,
    fractal_iterations: u32,

    power: f32,
    bailout: f32,
    max_distance: f32,
    hit_epsilon: f32,

    normal_epsilon: f32,
    fov: f32,
    ambient_strength: f32,
    diffuse_strength: f32,

    specular_strength: f32,
    shininess: f32,
    shadow_softness: f32,
    ao_strength: f32,

    camera_position: vec4<f32>,
    camera_target: vec4<f32>,
    light_direction: vec4<f32>,
    object_color: vec4<f32>,

    background_top: vec4<f32>,
    background_bottom: vec4<f32>,

    enable_shadows: u32,
    enable_ao: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> params: Params;


// ============================================================
// MANDELBULB DISTANCE ESTIMATOR
// ============================================================

fn mandelbulb_de(pos: vec3<f32>) -> f32 {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;

    for (
        var i: u32 = 0u;
        i < params.fractal_iterations;
        i = i + 1u
    ) {
        r = length(z);

        if (r > params.bailout) {
            break;
        }

        // Avoid division by zero near origin.
        let safe_r = max(r, 0.000001);

        var theta = acos(clamp(z.z / safe_r, -1.0, 1.0));
        var phi = atan2(z.y, z.x);

        let zr = pow(safe_r, params.power);

        dr =
            pow(safe_r, params.power - 1.0)
            * params.power
            * dr
            + 1.0;

        theta = theta * params.power;
        phi = phi * params.power;

        z =
            zr *
            vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            )
            + pos;
    }

    let safe_r = max(r, 0.000001);

    return 0.5 * log(safe_r) * safe_r / dr;
}


// ============================================================
// NORMAL CALCULATION
// ============================================================

fn get_normal(p: vec3<f32>) -> vec3<f32> {
    let e = params.normal_epsilon;

    let ex = vec3<f32>(e, 0.0, 0.0);
    let ey = vec3<f32>(0.0, e, 0.0);
    let ez = vec3<f32>(0.0, 0.0, e);

    let nx =
        mandelbulb_de(p + ex) -
        mandelbulb_de(p - ex);

    let ny =
        mandelbulb_de(p + ey) -
        mandelbulb_de(p - ey);

    let nz =
        mandelbulb_de(p + ez) -
        mandelbulb_de(p - ez);

    return normalize(vec3<f32>(nx, ny, nz));
}


// ============================================================
// SOFT SHADOW
// ============================================================

fn soft_shadow(
    origin: vec3<f32>,
    direction: vec3<f32>
) -> f32 {

    if (params.enable_shadows == 0u) {
        return 1.0;
    }

    var result = 1.0;
    var t = 0.02;

    for (var i = 0; i < 64; i = i + 1) {

        let p = origin + direction * t;
        let h = mandelbulb_de(p);

        if (h < 0.0005) {
            return 0.0;
        }

        result = min(
            result,
            params.shadow_softness * h / t
        );

        t = t + clamp(h, 0.01, 0.2);

        if (t > 10.0) {
            break;
        }
    }

    return clamp(result, 0.0, 1.0);
}


// ============================================================
// AMBIENT OCCLUSION
// ============================================================

fn ambient_occlusion(
    p: vec3<f32>,
    normal: vec3<f32>
) -> f32 {

    if (params.enable_ao == 0u) {
        return 1.0;
    }

    var occlusion = 0.0;
    var weight = 1.0;

    for (var i = 1; i <= 5; i = i + 1) {

        let fi = f32(i);

        let distance_sample = 0.02 + 0.06 * fi;

        let sample_point =
            p + normal * distance_sample;

        let distance_field =
            mandelbulb_de(sample_point);

        occlusion =
            occlusion +
            (distance_sample - distance_field)
            * weight;

        weight = weight * 0.6;
    }

    return clamp(
        1.0 - params.ao_strength * occlusion,
        0.0,
        1.0
    );
}


// ============================================================
// BACKGROUND
// ============================================================

fn background_color(ray_direction: vec3<f32>) -> vec3<f32> {

    let t =
        clamp(
            ray_direction.y * 0.5 + 0.5,
            0.0,
            1.0
        );

    return mix(
        params.background_bottom.xyz,
        params.background_top.xyz,
        t
    );
}


// ============================================================
// RAY MARCH
// ============================================================

fn ray_march(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>
) -> vec4<f32> {

    var total_distance = 0.0;

    for (
        var step: u32 = 0u;
        step < params.max_steps;
        step = step + 1u
    ) {

        let p =
            ray_origin +
            ray_direction * total_distance;

        let distance = mandelbulb_de(p);

        if (distance < params.hit_epsilon) {

            let normal = get_normal(p);

            let light_direction =
                normalize(-params.light_direction.xyz);

            let view_direction =
                normalize(ray_origin - p);

            // ----------------------------
            // Diffuse
            // ----------------------------

            let diffuse =
                max(
                    dot(normal, light_direction),
                    0.0
                );

            // ----------------------------
            // Shadow
            // ----------------------------

            let shadow =
                soft_shadow(
                    p + normal * 0.003,
                    light_direction
                );

            // ----------------------------
            // Ambient occlusion
            // ----------------------------

            let ao =
                ambient_occlusion(
                    p,
                    normal
                );

            // ----------------------------
            // Specular
            // ----------------------------

            let half_direction =
                normalize(
                    light_direction +
                    view_direction
                );

            let specular =
                pow(
                    max(
                        dot(normal, half_direction),
                        0.0
                    ),
                    params.shininess
                );

            // ----------------------------
            // Rim lighting
            // ----------------------------

            let rim =
                pow(
                    1.0 -
                    max(
                        dot(normal, view_direction),
                        0.0
                    ),
                    3.0
                );

            // ----------------------------
            // Color
            // ----------------------------

            var color =
                params.object_color.xyz
                * (
                    params.ambient_strength
                    +
                    diffuse
                    * params.diffuse_strength
                    * shadow
                );

            color = color * ao;

            color =
                color +
                vec3<f32>(1.0)
                * specular
                * params.specular_strength
                * shadow;

            color =
                color +
                params.object_color.xyz
                * rim
                * 0.15;

            return vec4<f32>(
                color,
                1.0
            );
        }

        total_distance =
            total_distance + distance;

        if (total_distance > params.max_distance) {
            break;
        }
    }

    return vec4<f32>(
        background_color(ray_direction),
        1.0
    );
}


// ============================================================
// CAMERA
// ============================================================

fn get_camera_ray(
    pixel: vec2<f32>
) -> vec3<f32> {

    let resolution =
        vec2<f32>(
            f32(params.width),
            f32(params.height)
        );

    var uv =
        (
            pixel + vec2<f32>(0.5)
        ) / resolution;

    uv = uv * 2.0 - 1.0;

    // Correct aspect ratio
    uv.x =
        uv.x *
        resolution.x /
        resolution.y;

    // Flip Y so image isn't upside-down.
    uv.y = -uv.y;

    let camera_position =
        params.camera_position.xyz;

    let target_to = params.camera_target.xyz;

    let forward =
        normalize(
            target_to -
            camera_position
        );

    let world_up =
        vec3<f32>(
            0.0,
            1.0,
            0.0
        );

    let right =
        normalize(
            cross(
                forward,
                world_up
            )
        );

    let up =
        normalize(
            cross(
                right,
                forward
            )
        );

    let scale =
        tan(
            params.fov * 0.5
        );

    return normalize(
        forward
        +
        right * uv.x * scale
        +
        up * uv.y * scale
    );
}


// ============================================================
// COMPUTE ENTRY POINT
// ============================================================

@compute
@workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id)
    id: vec3<u32>
) {

    if (
        id.x >= params.width ||
        id.y >= params.height
    ) {
        return;
    }

    let ray_origin =
        params.camera_position.xyz;

    let ray_direction =
        get_camera_ray(
            vec2<f32>(
                f32(id.x),
                f32(id.y)
            )
        );

    var color =
        ray_march(
            ray_origin,
            ray_direction
        );

    // Gamma correction
    color = vec4<f32>(
        pow(
            max(color.rgb, vec3<f32>(0.0)),
            vec3<f32>(1.0 / 2.2)
        ),
        1.0
    );

    textureStore(
        output_texture,
        vec2<i32>(
            i32(id.x),
            i32(id.y)
        ),
        color
    );
}
"""


# ============================================================
# UNIFORM BUFFER
# ============================================================

def create_uniform_data(config: RenderConfig) -> bytes:
    """
    WGSL uniform structs have alignment requirements.
    We build the buffer manually using uint32/float32 values.
    """

    values = bytearray()

    def u32(x):
        values.extend(
            np.array([x], dtype=np.uint32).tobytes()
        )

    def f32(x):
        values.extend(
            np.array([x], dtype=np.float32).tobytes()
        )

    def vec4(v):
        values.extend(
            np.array(v, dtype=np.float32).tobytes()
        )

    # First vec4-equivalent block
    u32(config.width)
    u32(config.height)
    u32(config.max_steps)
    u32(config.fractal_iterations)

    # floats
    f32(config.power)
    f32(config.bailout)
    f32(config.max_distance)
    f32(config.hit_epsilon)

    f32(config.normal_epsilon)
    f32(math.radians(config.fov_degrees))
    f32(config.ambient_strength)
    f32(config.diffuse_strength)

    f32(config.specular_strength)
    f32(config.shininess)
    f32(config.shadow_softness)
    f32(config.ao_strength)

    # vec4s
    vec4((*config.camera_position, 0.0))
    vec4((*config.camera_target, 0.0))
    vec4((*config.light_direction, 0.0))
    vec4((*config.object_color, 1.0))

    vec4((*config.background_top, 1.0))
    vec4((*config.background_bottom, 1.0))

    # u32 flags
    u32(1 if config.enable_shadows else 0)
    u32(1 if config.enable_ao else 0)
    u32(0)
    u32(0)

    return bytes(values)


# ============================================================
# GPU RENDERER
# ============================================================

def render(config: RenderConfig):
    print(
        f"Rendering Mandelbulb "
        f"{config.width}x{config.height}"
    )

    # --------------------------------------------------------
    # GPU adapter / device
    # --------------------------------------------------------

    adapter, device = create_gpu_device()

    print("GPU:", adapter.info)

    # --------------------------------------------------------
    # Shader
    # --------------------------------------------------------

    shader = device.create_shader_module(
        code=WGSL_SHADER
    )

    # --------------------------------------------------------
    # Output texture
    # --------------------------------------------------------

    texture = device.create_texture(
        size=(
            config.width,
            config.height,
            1
        ),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.COPY_SRC
        ),
    )

    texture_view = texture.create_view()

    # --------------------------------------------------------
    # Uniform buffer
    # --------------------------------------------------------

    uniform_data = create_uniform_data(config)

    uniform_buffer = device.create_buffer_with_data(
        data=uniform_data,
        usage=wgpu.BufferUsage.UNIFORM,
    )

    # --------------------------------------------------------
    # Bind group layout
    # --------------------------------------------------------

    bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility":
                        wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access":
                            wgpu.StorageTextureAccess.write_only,
                        "format":
                            wgpu.TextureFormat.rgba8unorm,
                        "view_dimension":
                            wgpu.TextureViewDimension.d2,
                    },
                },
                {
                    "binding": 1,
                    "visibility":
                        wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type":
                            wgpu.BufferBindingType.uniform
                    },
                },
            ]
        )

    pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[
                bind_group_layout
            ]
        )

    # --------------------------------------------------------
    # Compute pipeline
    # --------------------------------------------------------

    pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader,
                "entry_point": "main",
            },
        )

    # --------------------------------------------------------
    # Bind group
    # --------------------------------------------------------

    bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource":
                        texture_view,
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer":
                            uniform_buffer,
                        "offset": 0,
                        "size":
                            len(uniform_data),
                    },
                },
            ],
        )

    # --------------------------------------------------------
    # Encode compute work
    # --------------------------------------------------------

    encoder = device.create_command_encoder()

    compute_pass = encoder.begin_compute_pass()

    compute_pass.set_pipeline(
        pipeline
    )

    compute_pass.set_bind_group(
        0,
        bind_group
    )

    workgroups_x = math.ceil(
            config.width / 8
        )

    workgroups_y = math.ceil(
            config.height / 8
        )

    compute_pass.dispatch_workgroups(
        workgroups_x,
        workgroups_y,
        1
    )

    compute_pass.end()

    device.queue.submit(
        [encoder.finish()]
    )

    # --------------------------------------------------------
    # Read texture back
    # --------------------------------------------------------

    print("Reading result from GPU...")

    raw_data = device.queue.read_texture(
        {
            "texture": texture,
            "origin": (0, 0, 0),
            "mip_level": 0,
        },
        {
            "offset": 0,
            "bytes_per_row": config.width * 4,
            "rows_per_image": config.height,
        },
        (
            config.width,
            config.height,
            1,
        ),
    )

    # --------------------------------------------------------
    # Convert to image
    # --------------------------------------------------------

    image_array = np.frombuffer(
        raw_data,
        dtype=np.uint8
    )

    image_array = image_array.reshape(
        config.height,
        config.width,
        4
    )

    image = Image.fromarray(
        image_array,
        mode="RGBA"
    )

    image.save(
        config.output_file
    )

    print(
        f"Saved: {config.output_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    config = RenderConfig()
    render_rotation(
        config,
        frame_count=300,
        radius=4.5,
        height=1.5,
    )
    # config = RenderConfig()
    # config.width = 1920
    # config.height = 1536

    # config.fractal_iterations = 16
    # config.max_steps = 400
    # config.hit_epsilon = 0.00025
    # config.normal_epsilon = 0.0005
    # render(config)
    # outputfile = config.output_file.split(".")[0]

    # x, y, z = config.camera_position
    # for num in range(5):
    #     config.output_file = outputfile + f"_pos_{num}.png"
    #     render(config)

    #     config.camera_position = (x + 1, y + 1, z + 1)

