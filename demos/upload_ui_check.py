"""Optional Playwright check; run backend + Vite first, set HIVE_UI_URL if needed."""
import base64
import os
import struct
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True, args=['--enable-unsafe-webgpu'])
    base = os.getenv('HIVE_UI_URL', 'http://localhost:5173')
    page = browser.new_page(viewport={'width':1280,'height':1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    gpu_nodes = [browser.new_page() for _ in range(2)]
    for node in gpu_nodes:
        node.goto(base + '/legacy-node/')
        node.wait_for_function("document.querySelector('#status').textContent === 'Connected · idle'", timeout=30000)
    page.goto(base)
    page.get_by_role('button', name='Python kernel prototype').click()
    upload = page.locator('input[type=file]')
    upload.set_input_files({'name':'unsupported.py','mimeType':'text/x-python','buffer':b'import numpy as np\nz = 0j\nwhile True:\n    z += 1\n'})
    page.get_by_role('button', name='Analyze compatibility').click()
    expect(page.locator('.badge')).to_have_text('manual conversion required')
    expect(page.locator('#kernel')).to_have_value('')
    source = b'def transform(values):\n    results = []\n    for x in values:\n        results.append(x * 2 + 1)\n    return results\n'
    upload.set_input_files({'name':'transform.py','mimeType':'text/x-python','buffer':source})
    page.get_by_role('button', name='Analyze compatibility').click()
    expect(page.locator('.badge')).to_have_text('conversion available')
    page.get_by_role('button', name='Validate kernel').click()
    expect(page.get_by_role('button', name='Submit GPU job')).to_be_visible()
    page.locator('#kernel').fill(page.locator('#kernel').input_value() + '\n')
    expect(page.get_by_role('button', name='Submit GPU job')).not_to_be_visible()
    page.get_by_role('button', name='Validate kernel').click()
    page.get_by_role('button', name='Submit GPU job').click()
    expect(page.get_by_role('button', name='Download result')).to_be_visible(timeout=30000)
    assert '[\n  3,\n  5,\n  7,\n  9,\n  11\n]' in page.locator('pre').inner_text()
    print('Uploaded Python → report → editable preview → GPU output [3, 5, 7, 9, 11] passed')
    page.get_by_role('button', name='Load Mandelbrot kernel').click()
    page.get_by_role('button', name='Validate kernel').click()
    page.get_by_role('button', name='Submit GPU job').click()
    expect(page.get_by_role('button', name='Download result')).to_be_visible(timeout=30000)
    expect(page.locator('canvas')).to_be_visible()
    job_id = page.locator('.mono').inner_text()
    result = page.request.get(base + '/jobs/' + job_id).json()['result']
    values = [v[0] for v in struct.iter_unpack('<I', base64.b64decode(result['data']))]
    assert len(values) == 65536
    # Stable sample coordinates, away from the fractal boundary, checked against scalar Python.
    for x, y in [(0,0),(128,128),(255,255),(32,128),(200,10)]:
        cr, ci = -2+x/256*3, -1.5+y/256*3
        zr=zi=0.; iterations=0
        for _ in range(256):
            if zr*zr+zi*zi>4: break
            zr,zi=zr*zr-zi*zi+cr,2*zr*zi+ci
            iterations+=1
        assert values[y*256+x] == iterations, (x,y,values[y*256+x],iterations)
    assert page.locator('canvas').evaluate('(canvas) => { const a = canvas.getContext("2d").getImageData(0,0,256,256).data; return a.some(v => v !== 0); }')
    assert not errors, errors
    page.screenshot(path='/private/tmp/hive-upload-workflow.png', full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    print('Mandelbrot: 65,536 GPU pixels, sampled Python reference checks, canvas, mobile layout passed')
    browser.close()
