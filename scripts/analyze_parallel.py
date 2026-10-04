"""Usage: python -m scripts.analyze_parallel program.py [--mark] [--line N]."""
import argparse
from pathlib import Path
from backend.marked_python import AnalysisRequest, analyze_marked


def main():
    parser = argparse.ArgumentParser(description='Find independent Python array loops without executing the file')
    parser.add_argument('file', type=Path)
    parser.add_argument('--mark', action='store_true', help='Include suggested comment markers in the JSON report')
    parser.add_argument('--line', type=int, help='Select a loop by its starting line when multiple candidates exist')
    args = parser.parse_args()
    try:
        source = args.file.read_text()
        report = analyze_marked(AnalysisRequest(source=source, auto_mark=args.mark, candidate_line=args.line))
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(report.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
