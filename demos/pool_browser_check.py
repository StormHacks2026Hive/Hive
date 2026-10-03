"""Optional M1 Chrome integration check; run backend with built frontend first.

pip install playwright; HIVE_UI_URL=http://localhost:8000 python -m demos.pool_browser_check
A test-only delay before dispatch keeps leases observable during disconnect tests.
"""
import math
import os
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True,args=['--enable-unsafe-webgpu'])
    base=os.getenv('HIVE_UI_URL','http://localhost:8000')
    contexts=[browser.new_context() for _ in range(3)]
    def slow_gpu(route):
        response=route.fetch()
        source=response.text().replace('const start = performance.now();','const start = performance.now(); await new Promise(resolve => setTimeout(resolve, 100));')
        route.fulfill(response=response,body=source)
    for context in contexts[1:]: context.route('**/node/gpu.js',slow_gpu)
    submitter=contexts[0].new_page()
    errors=[]
    submitter.on('pageerror',lambda e:errors.append(str(e)))
    nodes=[c.new_page() for c in contexts[1:]]
    for index,node in enumerate(nodes):
        node.goto(base+'/node/')
        node.locator('#label').fill(f'Chrome GPU {index+1}')
        node.get_by_role('button',name='Start contributing').click()
        node.wait_for_function("document.querySelector('#status').textContent === 'Connected · idle'",timeout=30000)
        assert 'maxBufferSize' in node.locator('#capabilities').inner_text()
    submitter.goto(base)
    expect(submitter.get_by_role('heading',name='Connected contributors · 2')).to_be_visible(timeout=10000)
    submitter.get_by_role('button',name='Render on team GPUs').click()
    # Live tiles should appear before completion, not only as a final image.
    submitter.wait_for_function("document.querySelector('[role=status]')?.textContent.includes('running') && !document.querySelector('[role=status]').textContent.includes('0/64 displayed')",timeout=30000)
    expect(submitter.get_by_role('button',name='Download PNG')).to_be_enabled(timeout=60000)
    assert all(int(n.locator('#completed').inner_text())>0 for n in nodes)
    job_id=submitter.locator('.mono').inner_text().removeprefix('Job ')
    job=submitter.request.get(base+'/pool/jobs/'+job_id).json()
    assert job['completed_chunks']==64 and len(job['contributions'])==2
    rgba=submitter.request.get(base+job['result_url']).body()
    assert len(rgba)==512*512*4
    for x,y in [(0,0),(256,256),(511,511),(128,256),(400,20)]:
        cr=-2+(x+.5)/512*3; ci=-1.5+(y+.5)/512*3
        zr=zi=0.; count=0
        for _ in range(256):
            if zr*zr+zi*zi>4: break
            zr,zi=zr*zr-zi*zi+cr,2*zr*zi+ci; count+=1
        if count==256: expected=(12,20,23,255)
        else:
            t=math.sqrt(count/256); expected=(int(255*t),int(210*t*t),int(100+155*(1-t)),255)
        actual=tuple(rgba[(y*512+x)*4:(y*512+x)*4+4])
        assert all(abs(a-b)<=1 for a,b in zip(actual,expected)), (x,y,actual,expected)
    with submitter.expect_download() as download:
        submitter.get_by_role('button',name='Download PNG').click()
    assert download.value.suggested_filename.endswith('.png')
    print('64 tiles rendered live by two GPU workers; assembled image samples match Python; PNG download passed')
    submitter.reload()
    submitter.locator('summary').click()
    submitter.get_by_label('Job ID').fill(job_id)
    submitter.get_by_role('button',name='Load job',exact=True).click()
    expect(submitter.get_by_role('button',name='Download PNG')).to_be_enabled(timeout=30000)
    print('Reload and job snapshot restored all accepted tiles')
    submitter.get_by_role('button',name='Render on team GPUs').click()
    nodes[0].wait_for_function("document.querySelector('#status').textContent === 'Working on GPU'",timeout=20000)
    nodes[0].close()
    expect(submitter.get_by_role('button',name='Download PNG')).to_be_enabled(timeout=60000)
    second_id=submitter.locator('.mono').inner_text().removeprefix('Job ')
    second=submitter.request.get(base+'/pool/jobs/'+second_id).json()
    assert second['status']=='done' and second['retries']>=1
    print('Worker closed during GPU assignment; tile reassigned; image completed; retries:',second['retries'])
    submitter.screenshot(path='/private/tmp/hive-m1-pool.png',full_page=True)
    nodes[1].get_by_role('button',name='Stop',exact=True).click()
    expect(nodes[1].locator('#status')).to_have_text('Stopped')
    expect(submitter.get_by_role('heading',name='Connected contributors · 0')).to_be_visible(timeout=10000)
    submitter.get_by_role('button',name='Render on team GPUs').click()
    expect(submitter.get_by_role('button',name='Cancel job')).to_be_visible(timeout=10000)
    submitter.get_by_role('button',name='Cancel job').click()
    submitter.wait_for_function("document.querySelector('[role=status]').textContent.startsWith('cancelled')")
    print('Stop removed worker; queued job cancellation passed')
    submitter.set_viewport_size({'width':390,'height':844})
    assert submitter.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    browser.close()
