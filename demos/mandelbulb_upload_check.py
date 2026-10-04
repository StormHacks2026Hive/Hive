"""Check real upload UI + two browser contributors against a full WebGPU render.

Start uvicorn first. Run python -m demos.mandelbulb_upload_check.
This checks correctness, not speedup: both contributors share this machine's GPU.
"""

import base64
import io
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright

from backend.pool.image_workloads import ImageAnalysisRequest, extract, uniform_data

LOG = logging.getLogger(__name__)
RENDERER = Path(__file__).resolve().parent / "test_files/Mandelbulb_wgsl.py"

# Reference uses the unmodified shader and original bindings, in one full dispatch.
REFERENCE = """async ({source, uniform, width, height}) => {
  const adapter = await navigator.gpu.requestAdapter();
  const d = await adapter.requestDevice();
  d.pushErrorScope('validation');
  const module = d.createShaderModule({code: source});
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === 'error');
  if (errors.length) throw new Error(errors.map(m => m.message).join('\\n'));
  const pipeline = await d.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
  const bytes = Uint8Array.from(atob(uniform), c => c.charCodeAt(0));
  const params = d.createBuffer({size: bytes.length, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  d.queue.writeBuffer(params, 0, bytes);
  const texture = d.createTexture({size: [width,height], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC});
  const stride = Math.ceil(width * 4 / 256) * 256;
  const readback = d.createBuffer({size: stride * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const bind = d.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
    {binding:0, resource:texture.createView()}, {binding:1, resource:{buffer:params}}]});
  const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0,bind);
  pass.dispatchWorkgroups(Math.ceil(width/8), Math.ceil(height/8)); pass.end();
  encoder.copyTextureToBuffer({texture}, {buffer:readback,bytesPerRow:stride}, [width,height]);
  d.queue.submit([encoder.finish()]);
  const error = await d.popErrorScope(); if (error) throw new Error(error.message);
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readback.getMappedRange()), pixels = new Uint8Array(width * height * 4);
  for (let y=0; y<height; y++) pixels.set(mapped.subarray(y*stride,y*stride+width*4), y*width*4);
  let binary = ''; for(let i=0;i<pixels.length;i+=8192) binary += String.fromCharCode(...pixels.subarray(i,i+8192));
  readback.unmap(); params.destroy(); readback.destroy(); texture.destroy(); d.destroy();
  return btoa(binary);
}"""


def main():
    base = os.getenv("HIVE_UI_URL", "http://localhost:8019")
    width = int(os.getenv("HIVE_RENDER_WIDTH", "259"))
    height = int(os.getenv("HIVE_RENDER_HEIGHT", "131"))
    expected_tiles = ((width + 127) // 128) * ((height + 127) // 128)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=True, args=["--enable-unsafe-webgpu"]
        )
        contexts = [browser.new_context() for _ in range(2)]

        def delay_tile(route):
            response = route.fetch()
            source = response.text().replace(
                "  async textureRender(chunk) {",
                "  async textureRender(chunk) {\n    await new Promise(resolve => setTimeout(resolve, 500));",
            )
            route.fulfill(response=response, body=source)

        errors = []
        nodes = []
        for index, context in enumerate(contexts):
            # Delay only to ensure both contributors participate in this tiny job.
            context.route("**/node/gpu.js", delay_tile)
            node = context.new_page()
            node.on("pageerror", lambda error: errors.append(str(error)))
            node.goto(base + "/node/")
            node.locator("#label").fill(f"Mandelbulb check {index + 1}")
            node.get_by_role("button", name="Start contributing").click()
            node.wait_for_function(
                "document.querySelector('#status').textContent === 'Connected · idle'",
                timeout=30000,
            )
            nodes.append(node)
        ui = contexts[0].new_page()
        ui.on("pageerror", lambda error: errors.append(str(error)))
        ui.goto(base)
        ui.locator("input[type=file]").set_input_files(RENDERER)
        expect(ui.locator("#workload-input")).to_have_count(0)
        ui.get_by_label("Image width", exact=True).fill(str(width))
        ui.get_by_label("Image height", exact=True).fill(str(height))
        ui.get_by_role("button", name="Analyze source", exact=True).click()
        expect(ui.get_by_role("button", name="Run on team GPUs")).to_be_enabled(
            timeout=30000
        )
        with ui.expect_response(
            lambda r: r.url == base + "/pool/jobs" and r.request.method == "POST"
        ) as created:
            ui.get_by_role("button", name="Run on team GPUs").click()
        assert created.value.status == 202, created.value.text()
        job_id = created.value.json()["job_id"]
        download = ui.get_by_role("button", name="Download image PNG", exact=True)
        expect(download).to_be_enabled(timeout=90000)
        status = contexts[0].request.get(base + "/pool/jobs/" + job_id).json()
        assert len(status["contributions"]) == 2, status
        assert status["completed_chunks"] == expected_tiles, status
        pixels = contexts[0].request.get(base + status["result_url"]).body()
        shader, config = extract(
            ImageAnalysisRequest(
                source=RENDERER.read_text(), width=width, height=height
            )
        )
        reference = base64.b64decode(
            ui.evaluate(
                REFERENCE,
                {
                    "source": shader,
                    "uniform": base64.b64encode(uniform_data(config)).decode(),
                    "width": width,
                    "height": height,
                },
            )
        )
        split = np.frombuffer(pixels, np.uint8).reshape(height, width, 4)
        single = np.frombuffer(reference, np.uint8).reshape(height, width, 4)
        difference = np.abs(split.astype(np.int16) - single.astype(np.int16))
        assert difference.max() <= 1, f"Max channel difference {difference.max()}"
        with ui.expect_download() as saved:
            download.click()
        png = Image.open(io.BytesIO(Path(saved.value.path()).read_bytes())).convert(
            "RGBA"
        )
        assert png.size == (width, height)
        assert np.array_equal(np.asarray(png), split)
        assert not errors, errors
        LOG.info(
            "Original .py upload, two contributors, stitched tiles, PNG download: passed; max channel diff=%d",
            difference.max(),
        )
        browser.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
