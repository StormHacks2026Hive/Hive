"""Optional integration check: pip install playwright; uses installed Chrome."""
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True, args=['--enable-unsafe-webgpu'])
    pages = [browser.new_page() for _ in range(2)]
    for page in pages:
        page.goto('http://localhost:8000/node/')
        page.wait_for_function("document.querySelector('#status').textContent === 'Connected · idle'", timeout=30000)
    for demo in ['demos.monte_carlo', 'demos.elementwise']:
        process = subprocess.Popen([sys.executable, '-m', demo], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 180
        while process.poll() is None and time.monotonic() < deadline:
            pages[0].wait_for_timeout(100)
        if process.poll() is None:
            process.kill()
            raise TimeoutError(demo)
        print(process.stdout.read())
        if process.returncode:
            print([page.locator('#error').inner_text() for page in pages])
            raise RuntimeError(demo + ' failed')
    completed = [int(page.locator('#completed').inner_text()) for page in pages]
    assert all(n >= 2 for n in completed), completed
    print('Both browser tabs contributed:', completed)
    browser.close()
