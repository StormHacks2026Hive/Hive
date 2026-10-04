"""Run with Hive server running: HIVE_UI_URL=http://localhost:8017 python -m demos.workloads_browser_check."""
import base64
import os
import re
import struct
import time
from onnx import helper, TensorProto
from playwright.sync_api import sync_playwright, expect
from backend.models import TypedArray

WGSL = '''@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    result[i] = values[i] * 2.0 + f32(i);
}'''
PYTHON = '''def transform(values):
    result = []
    # hive:parallel begin
    for value in values:
        scaled = value * 2
        result.append(scaled + 1)
    # hive:parallel end
    return result
'''


def main():
    base=os.getenv('HIVE_UI_URL','http://localhost:8017')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True,args=['--enable-unsafe-webgpu'])
        contexts=[browser.new_context() for _ in range(2)]
        def slow_gpu(route):
            response = route.fetch()
            source = response.text().replace('  async render(chunk) {', '  async render(chunk) {\n    await new Promise(resolve => setTimeout(resolve, 100));')
            route.fulfill(response=response, body=source)
        for context in contexts:
            context.route('**/node/gpu.js', slow_gpu)
        nodes=[c.new_page() for c in contexts]
        errors=[]
        for index, node in enumerate(nodes):
            node.on('pageerror',lambda e:errors.append(str(e)))
            node.goto(base+'/node/')
            node.locator('#label').fill(f'Workload GPU {index+1}')
            node.get_by_role('button',name='Start contributing').click()
            node.wait_for_function("document.querySelector('#status').textContent === 'Connected · idle'",timeout=30000)
        client=contexts[0].request
        def run(request):
            created=client.post(base+'/pool/jobs',data=request)
            assert created.status==202,created.text()
            job_id=created.json()['job_id']
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                status=client.get(base+'/pool/jobs/'+job_id).json()
                if status['status'] in ('done','failed','cancelled'):break
                time.sleep(.1)
            assert status['status']=='done',(status, [n.locator('#error').inner_text() for n in nodes])
            return status,client.get(base+status['result_url']).body()
        graph=helper.make_graph([helper.make_node('Relu',['X'],['Y'])], 'relu',
            [helper.make_tensor_value_info('X',TensorProto.FLOAT,['batch',2])],
            [helper.make_tensor_value_info('Y',TensorProto.FLOAT,['batch',2])])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
        encoded=base64.b64encode(model.SerializeToString()).decode()
        if os.getenv('HIVE_CHECK_UI_ONLY') != '1':
            values=list(range(10240))
            typed=TypedArray.encode(values).model_dump()
            status,data=run({'kind':'wgsl','wgsl':WGSL,'input':typed,'chunk_size':1024})
            assert list(struct.unpack('<10240f',data))==[v*3 for v in values]
            print('Custom WGSL auto analysis, global offsets, ordered output: passed; contributors:',len(status['contributions']))
            status,data=run({'kind':'python','source':PYTHON,'input':typed,'chunk_size':1024})
            assert list(struct.unpack('<10240f',data))==[v*2+1 for v in values]
            print('Marked Python on pool WebGPU workers: passed')
            for distribution in ['tiles','frames']:
                frames=[{'max_iterations':32},{'max_iterations':64,'xmin':-1.5,'xmax':.5}]
                status,data=run({'kind':'animation','frames':frames,'distribution':distribution})
                assert len(data)==512*512*4*2 and data[:512*512*4]!=data[512*512*4:]
                assert client.get(base+'/pool/jobs/'+status['job_id']+'/frames/1').body()==data[512*512*4:]
                print(f'Animation {distribution} distribution, ordered frames: passed')
            values=[float(i%17-8) for i in range(1026)]
            request={'kind':'onnx','model':encoded,'input':TypedArray.encode(values).model_dump(),
                'input_shape':[513,2],'batch_size':32,'independent':True}
            status,data=run(request)
            assert list(struct.unpack('<1026f',data))==[max(0,v) for v in values]
            assert len(status['contributions'])==2, status
            print('ONNX WebGPU, two workers, dynamic final batch, ordered tensor assembly: passed')
            # A repeated job exercises per-worker model/session caching.
            _, repeated=run(request)
            assert data==repeated
            print('ONNX cached repeat inference: passed')
        ui_context = browser.new_context()
        page = ui_context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(base)
        page.get_by_role('button',name='Analyze source',exact=True).click()
        expect(page.get_by_role('button',name='Run on team GPUs')).to_be_enabled()
        def ui_run():
            with page.expect_response(lambda r:r.url == base+'/pool/jobs' and r.request.method=='POST') as response:
                page.get_by_role('button',name='Run on team GPUs').click()
            created=response.value.json()
            assert response.value.status==202,created
            expect(page.get_by_text(created['job_id'],exact=True)).to_be_visible()
            expect(page.get_by_role('status')).to_contain_text('done',timeout=60000)
        ui_run()
        expect(page.get_by_label('Output preview')).to_have_text('[3,5,7,9]')
        page.get_by_label('Source',exact=True).fill(WGSL.replace('values[i]', 'values[i-1]'))
        expect(page.get_by_role('button',name='Run on team GPUs')).to_be_disabled()
        page.get_by_label('Workload',exact=True).select_option('python')
        page.get_by_role('button',name='Find and mark parallel loop').click()
        expect(page.get_by_label('Source',exact=True)).to_have_value(re.compile('.*# hive:parallel begin.*', re.S))
        ui_run()
        expect(page.get_by_label('Output preview')).to_have_text('[3,5,7,9]')
        page.get_by_label('Workload',exact=True).select_option('onnx')
        page.get_by_label('Upload model (.onnx, up to 4 MiB)').set_input_files({
            'name':'relu.onnx','mimeType':'application/octet-stream','buffer':model.SerializeToString()})
        expect(page.get_by_text('relu.onnx',exact=False)).to_be_visible()
        page.get_by_label('Input array (flat JSON numbers)').fill('[-1, 2, -3, 4, -5, 6, -7, 8]')
        page.get_by_role('button',name='Analyze model').click()
        expect(page.get_by_label('Input shape (comma separated)')).to_have_value('4, 2')
        ui_run()
        expect(page.get_by_label('Output preview')).to_have_text('[0,2,0,4,0,6,0,8]')
        with page.expect_download() as download:
            page.get_by_role('button',name='Download output',exact=True).click()
        assert download.value.suggested_filename.endswith('.f32.bin')
        page.get_by_label('Workload',exact=True).select_option('animation')
        page.get_by_label('Frames',exact=True).fill('2')
        ui_run()
        expect(page.get_by_role('button',name='Download frame PNG')).to_be_enabled()
        page.get_by_role('button',name='Play animation').click()
        expect(page.get_by_role('button',name='Pause animation')).to_be_visible()
        page.get_by_role('button',name='Pause animation').click()
        with page.expect_download() as download:
            page.get_by_role('button',name='Download frame PNG').click()
        assert download.value.suggested_filename.endswith('.png')
        page.screenshot(path='/private/tmp/hive-workloads-desktop.png',full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/private/tmp/hive-workloads-mobile.png',full_page=True)
        print('React workflows: analyze WGSL, invalidate edits, mark Python, upload ONNX, submit, download, animate, mobile layout: passed')
        assert not errors,errors
        browser.close()


if __name__=='__main__':
    main()
