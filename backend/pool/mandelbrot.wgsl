// Fixed M1 ABI: 48-byte uniform at binding 0; packed RGBA pixels at binding 1.
struct Params {
    tile_x: u32, tile_y: u32, tile_width: u32, tile_height: u32,
    image_width: u32, image_height: u32, max_iterations: u32, padding: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.tile_width || id.y >= params.tile_height) { return; }
    let x = params.tile_x + id.x;
    let y = params.tile_y + id.y;
    let cr = params.xmin + (f32(x) + 0.5) / f32(params.image_width) * (params.xmax - params.xmin);
    let ci = params.ymin + (f32(y) + 0.5) / f32(params.image_height) * (params.ymax - params.ymin);
    var zr = 0.0;
    var zi = 0.0;
    var iterations = 0u;
    // The service caps max_iterations at 1024. No uploaded code runs on the host.
    for (var step = 0u; step < 1024u; step++) {
        if (step >= params.max_iterations || zr * zr + zi * zi > 4.0) { break; }
        let next_real = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = next_real;
        iterations++;
    }
    var red = 12u;
    var green = 20u;
    var blue = 23u;
    if (iterations < params.max_iterations) {
        let t = sqrt(f32(iterations) / f32(params.max_iterations));
        red = u32(255.0 * t);
        green = u32(210.0 * t * t);
        blue = u32(100.0 + 155.0 * (1.0 - t));
    }
    pixels[id.y * params.tile_width + id.x] = red | (green << 8u) | (blue << 16u) | (255u << 24u);
}
