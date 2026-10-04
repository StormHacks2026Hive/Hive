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