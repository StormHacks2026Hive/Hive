import pytest
from fastapi.testclient import TestClient
from backend.converter import analyze, validate_preview

SOURCE = '''def transform(values):
    results = []
    for x in values:
        results.append(x * 2 + 1)
    return results
'''


def test_convert_complete_function():
    report = analyze(SOURCE)
    assert report.status == 'conversion_available'
    assert 'values[i]' in report.kernel and '2.0' in report.kernel
    assert report.mode == 'data-slice'
    assert validate_preview(report.kernel).valid
    assert any('float32' in f.message for f in report.findings)


@pytest.mark.parametrize('expression', ['x / 2', 'x ** 2', 'other + x', 'abs(x)', 'values[0]', 'x % 2'])
def test_unsupported_expression_never_silently_converted(expression):
    report = analyze(SOURCE.replace('x * 2 + 1', expression))
    assert report.status == 'manual_conversion_required'
    assert report.kernel is None


@pytest.mark.parametrize('source', [
    SOURCE + '\nprint(transform([1]))',
    SOURCE.replace('    return results', '    print(results)\n    return results'),
    SOURCE.replace('def transform(values):', '@decorator\ndef transform(values):'),
    SOURCE.replace('for x in values:', 'for values in values:').replace('x * 2 + 1', 'values * 2'),
    SOURCE.replace('def transform(values):', 'def transform(values, scale=2):'),
    SOURCE.replace('results.append(x * 2 + 1)', 'results.append(x * 2 + 1)\n        x += 1'),
])
def test_requires_exact_pattern(source):
    assert analyze(source).kernel is None


def test_manual_report_explains_imports_complex_and_while():
    report = analyze('import numpy as np\nz = 0j\nwhile True:\n    z += 1\n')
    assert report.status == 'manual_conversion_required'
    messages = ' '.join(f.message for f in report.findings)
    assert 'Imports' in messages and 'Complex' in messages and 'While' in messages
    assert report.kernel is None


def test_never_execute_upload(tmp_path):
    marker = tmp_path / 'executed'
    report = analyze(f"open({str(marker)!r}, 'w').write('bad')")
    assert report.kernel is None and not marker.exists()


def test_syntax_error_and_missing_contract():
    assert analyze('def broken(').findings[0].severity == 'error'
    result = validate_preview('def k(result: Array[u32]):\n    result[global_id()] = u32(1)')
    assert not result.valid and 'offset' in result.error


def test_existing_kernel_recognized():
    converted = analyze(SOURCE).kernel
    report = analyze(converted)
    assert report.status == 'compatible' and report.kernel == converted


def test_analysis_http_contract():
    from backend.main import app
    with TestClient(app) as client:
        response = client.post('/kernels/analyze', json={'source': SOURCE})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'conversion_available'
        validated = client.post('/kernels/validate', json={'source': data['kernel']}).json()
        assert validated['valid'] and validated['bindings'][0]['access'] == 'read'
        assert client.post('/kernels/analyze', json={'source': 'x' * 32001}).status_code == 422
        assert client.post('/kernels/validate', json={'source': 'import os'}).json()['valid'] is False
