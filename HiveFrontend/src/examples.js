export const pythonExample = `def transform(values):
    results = []
    for x in values:
        results.append(x * 2 + 1)
    return results
`;
export const mandelbrotKernel = `def mandelbrot(offset: u32, count: u32, seed: u32, width: u32, height: u32, xmin: f32, xmax: f32, ymin: f32, ymax: f32, result: Array[u32]):
    i = global_id()
    if i < count:
        pixel = offset + i
        cx = xmin + f32(pixel % width) / f32(width) * (xmax - xmin)
        cy = ymin + f32(pixel // width) / f32(height) * (ymax - ymin)
        zr = 0.0
        zi = 0.0
        iterations = u32(0)
        for step in range(256):
            if zr * zr + zi * zi > 4.0:
                break
            next_real = zr * zr - zi * zi + cx
            zi = 2.0 * zr * zi + cy
            zr = next_real
            iterations += 1
        result[i] = iterations
`;
