"""Check frozen evidence, chart numbers, report layout, PDF and Git portability.

Uses report dependencies only; never loads models or raw photos.
"""
import argparse
import base64
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import statistics
import subprocess
from urllib.parse import unquote, urlsplit

import numpy as np
import pymupdf as fitz
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
checks = {}


def check(name, value):
    checks[name] = bool(value)
    if not value:
        raise AssertionError(name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


class ReportHTML(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.images, self.ids, self.links, self.figures = [], [], [], 0
        self.embedded_only = True
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'img' and 'src' in attrs:
            self.images.append(attrs['src'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
        if tag == 'figure':
            self.figures += 1
        if (tag == 'script' and attrs.get('src')) or (tag == 'link' and attrs.get('href')):
            self.embedded_only = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--git-ref', help='staged, HEAD, or another commit/tree')
    parser.add_argument('--check-only', action='store_true', help='Do not rewrite summaries or manifest')
    args = parser.parse_args()
    diagnostics = args.diagnostics
    if diagnostics is None:
        fresh = OUT / '.report_check'
        diagnostics = fresh if (fresh / 'render_summary.json').exists() else OUT / 'data'

    index = read(OUT / 'data/evidence_index.json')
    check('31_evidence_sources', len(index) == 31 and len({r['id'] for r in index}) == 31)
    evidence, original_matches = {}, 0
    for row in index:
        snapshot = OUT / row['snapshot']
        check('snapshot_sha256:' + row['id'], sha(snapshot) == row['sha256'])
        original = ROOT / row['path']
        if original.exists():
            check('original_bytes:' + row['id'], original.read_bytes() == snapshot.read_bytes())
            original_matches += 1
        evidence[row['id']] = read(snapshot)

    data = read(OUT / 'data/report_data.json')
    check('measurement_source_binding', data['source_sha256'] == evidence['S25']['source_sha256'])
    check('measurement_commit_binding', data['commit'] == evidence['S25']['code_commit'])
    check('no_new_inference', data['inference_runs_for_report'] == 0)
    with (OUT / 'data/retrieval_metrics.csv').open() as stream:
        retrieval_rows = list(csv.DictReader(stream))
    check('14_retrieval_rows', len(retrieval_rows) == len(data['data']['retrieval']) == 14)
    for row, bound in zip(retrieval_rows, data['data']['retrieval']):
        prefix = row['backend'] + ':' + row['variant']
        expected = evidence['S02']['results'][row['backend']][row['variant']]['all']['metrics']
        check('retrieval_row_binding:' + prefix, all(row[key] == str(bound[key]) for key in row))
        for key in ['count', 'correct_at_1', 'correct_at_3', 'recall_at_1', 'recall_at_3', 'macro_f1']:
            check('retrieval_metric:' + prefix + ':' + key, float(row[key]) == expected[key])
        check('retrieval_denominator:' + prefix, int(row['count']) == 262 and
              int(row['correct_at_1']) / 262 == float(row['recall_at_1']) and
              int(row['correct_at_3']) / 262 == float(row['recall_at_3']))

    latest = evidence['S25']
    runs = {'baseline': evidence['S26'], 'candidate': evidence['S27']}
    with (OUT / 'data/u2_repeat_metrics.csv').open() as stream:
        repeats = list(csv.DictReader(stream))
    check('all_10_pipeline_repeats_in_csv', len(repeats) == 10 and
          {(r['mode'], int(r['repeat'])) for r in repeats} ==
          {(mode, number) for mode in runs for number in range(1, 6)})
    for row in repeats:
        expected = runs[row['mode']]['pipeline_runs'][int(row['repeat']) - 1]
        for key in ['seconds', 'rss_peak_bytes', *expected['stage_timings']]:
            value = expected.get(key, expected['stage_timings'].get(key))
            check(f"repeat:{row['mode']}:{row['repeat']}:{key}", float(row[key]) == value)
    for kind in ['feature_runs', 'pipeline_runs']:
        timing = latest['timing'][kind]
        for mode, key in [('baseline', 'baseline'), ('candidate', 'batch')]:
            values = [r['seconds'] for r in runs[mode][kind]]
            check(f'five_repeats:{kind}:{mode}', len(values) == 5)
            check(f'median:{kind}:{mode}', statistics.median(values) == timing[key + '_median_seconds'])
            check(f'p95:{kind}:{mode}', abs(float(np.percentile(values, 95)) -
                  timing['distribution'][key]['p95_seconds']) < 1e-8)
        check('speedup:' + kind, abs(timing['speedup'] - timing['baseline_median_seconds'] /
              timing['batch_median_seconds']) < 1e-10)
    check('latest_performance_gate_preserved', latest['performance_status'] == 'not_passed' and
          [key for key, value in latest['checks'].items() if not value] == ['feature_speedup'])
    all_runs = runs['baseline']['pipeline_runs'] + runs['candidate']['pipeline_runs']
    check('decisions_and_scores_equal_in_all_runs', len({r['decisions_sha256'] for r in all_runs}) == 1 and
          len({r['quality_sha256'] for r in all_runs}) == 1)
    check('154_kept_and_no_errors_in_all_runs', all(r['kept'] == 154 and r['errors'] == 0 for r in all_runs))
    check('cache_zero_forwards', latest['cache_hit_forward_calls'] == {'image': 0, 'text': 0} and
          latest['pipeline_cache_hit_forward_calls'] == {'image': 0, 'text': 0})
    check('independent_quality_not_claimed', latest['quality'] == 'not_evaluated_requires_verified_labels')
    check('81_regression_results_preserved', evidence['S30']['total_passed'] == 81)
    check('dali_not_adopted', evidence['S24']['status'] == 'not_adopted')
    check('dali_changes_preserved', sum(row[0].endswith('/organized_name') for row in evidence['S31']) == 9)

    document = (OUT / 'technical_report.html').read_text()
    parsed = ReportHTML(document)
    check('29_html_pages', len(re.findall(r'data-page="\d+"', document)) == 29)
    check('15_numbered_body_figures', parsed.figures == 15)
    check('no_duplicate_html_ids', len(parsed.ids) == len(set(parsed.ids)))
    check('all_html_citation_targets_exist', all(target[1:] in parsed.ids for target in parsed.links if target.startswith('#')))
    check('all_html_images_embedded', len(parsed.images) >= 16 and all(src.startswith('data:image/') for src in parsed.images))
    check('no_external_script_or_stylesheet', parsed.embedded_only)
    for number, src in enumerate(parsed.images, 1):
        encoded = base64.b64decode(src.split(',', 1)[1], validate=True)
        check(f'embedded_image:{number}', len(encoded) > 100)
    for path in (OUT / 'figures').iterdir():
        if path.suffix in ['.png', '.jpg']:
            with Image.open(path) as picture:
                picture.verify()
            check('image_valid:' + path.name, True)
        elif path.suffix == '.svg':
            check('svg_has_display_png:' + path.stem, path.with_suffix('.png').exists())
    markdown_files = [ROOT / 'docs/technical_report_20261026.md', ROOT / 'docs/technical_report_20261024.md',
                      *sorted((ROOT / 'docs/reports').rglob('*.md'))]
    targets = set()
    image_count = 0
    for file in markdown_files:
        source = file.read_text()
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', source):
            url = urlsplit(target.strip('<>'))
            if url.scheme or not url.path:
                continue
            path = (file.parent / unquote(url.path)).resolve()
            if not args.check_only and not path.exists() and path in [
                OUT / 'data/verification_summary.json', OUT / 'artifact_manifest.json'
            ]:
                targets.add(path)
                continue  # These two files are produced after all input checks pass.
            check('local_link:' + str(file.relative_to(ROOT)) + ':' + target, path.exists())
            check('link_within_repo:' + str(path), path.is_relative_to(ROOT))
            check('report_asset_not_in_ignored_outputs:' + target, not path.is_relative_to(ROOT / 'outputs'))
            targets.add(path)
        if file.name == 'technical_report_20261026.md':
            images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', source)
            image_count = len(images)
            check('github_has_15_raster_figures', len(images) == 15 and all(Path(p).suffix in ['.png', '.jpg'] for p in images))
            check('github_cites_all_31_snapshots', all(f'data/evidence/{row["id"]}.json' in source for row in index))
            table = [line.split('|')[1:-1] for line in source.splitlines() if re.match(r'\| (SigLIP2|Gemma 2) \|', line)]
            check('github_retrieval_table_has_14_rows', len(table) == 14)
            for actual, expected in zip(table, retrieval_rows):
                values = [str(expected['correct_at_1']), f"{100 * float(expected['recall_at_1']):.2f}%",
                          str(expected['correct_at_3']), f"{100 * float(expected['recall_at_3']):.2f}%",
                          f"{100 * float(expected['macro_f1']):.2f}%"]
                check('github_table_numbers:' + expected['backend'] + ':' + expected['variant'],
                      [item.strip() for item in actual[2:]] == values)
    for file in [ROOT / 'README.md', ROOT / 'docs/README.md', ROOT / 'WORK_STATE.md']:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', file.read_text()):
            if 'technical_report_' not in target:
                continue
            path = (file.parent / target.split('#')[0]).resolve()
            check('navigation:' + file.name + ':' + target, path.exists())
            targets.add(path)

    render = read(diagnostics / 'render_summary.json')
    layout = read(diagnostics / 'layout_check.json')
    check('browser_passed_offline_and_zoom', render['status'] == 'passed' and render['zoom'] and
          render['imagesLoaded'] and not render['errors'] and not render['externalRequests'])
    check('390px_mobile_no_overflow', render['mobile']['width'] == render['mobile']['scrollWidth'] == 390)
    check('29_desktop_pages_fit', len(layout['layout']) == 29 and not layout['errors'] and
          not any(row['overflow'] or row['horizontal'] for row in layout['layout']))
    pdf = fitz.open(OUT / 'technical_report.pdf')
    check('29_pdf_pages', len(pdf) == 29)
    check('pdf_bookmarks_for_each_page', set(range(1, 30)).issubset({entry[2] for entry in pdf.get_toc()}))
    text = '\n'.join(page.get_text() for page in pdf)
    check('pdf_no_replacement_characters', '\ufffd' not in text)
    check('pdf_contains_latest_and_historical_results', all(value in text for value in
          ['76.34%', '69.85%', '71.76%', '53.44%', '59.088', '42.655', '167.209', '122.834',
           '26.54%', '1.385', 'not_passed', 'DALI不采用', '172.316', '181.181', '0.99999893']))
    for i, page in enumerate(pdf, 1):
        check(f'A4:{i}', abs(page.rect.width - 595.28) < 1 and abs(page.rect.height - 841.89) < 1)
        check(f'pdf_text:{i}', len(page.get_text()) > 100)
        if i > 1:
            check(f'footer:{i}', f'{i:02d} / 29' in page.get_text())
        spans = [span for block in page.get_text('dict')['blocks'] if block['type'] == 0
                 for line in block['lines'] for span in line['spans']]
        check(f'text_within_page:{i}', all(s['bbox'][0] >= -1 and s['bbox'][1] >= -1 and
              s['bbox'][2] <= page.rect.width + 1 and s['bbox'][3] <= page.rect.height + 1 for s in spans))
        if i > 1:
            check(f'body_above_footer:{i}', all(s['bbox'][3] <= page.rect.height - 38 for s in spans
                  if s['bbox'][1] < page.rect.height - 40))

    artifacts = sorted(p for bundle in [OUT, OUT.parent / 'technical_report_20261024']
                       for p in bundle.rglob('*') if p.is_file() and
                       not any(part in ['.report_check', '__pycache__'] for part in p.parts))
    if args.git_ref:
        names = git('ls-files', '-z') if args.git_ref == 'staged' else git('ls-tree', '-r', '-z', '--name-only', args.git_ref)
        tracked = {name.decode() for name in names.split(b'\0') if name}
        for path in targets | set(artifacts) | set(markdown_files):
            rel = path.relative_to(ROOT).as_posix()
            check('git_contains:' + rel, rel in tracked if path.is_file() else any(n.startswith(rel + '/') for n in tracked))
            if path.is_file() and (args.check_only or path.name not in ['verification_summary.json', 'artifact_manifest.json']):
                selector = ':' + rel if args.git_ref == 'staged' else args.git_ref + ':' + rel
                check('git_bytes:' + rel, git('show', selector) == path.read_bytes())
    check('report_diff_has_no_whitespace_errors', subprocess.run(
        ['git', 'diff', '--check', '--', 'README.md', 'docs/README.md', 'docs/technical_report_20261024.md',
         'docs/technical_report_20261026.md', 'docs/reports'], cwd=ROOT, capture_output=True).returncode == 0)
    manifest_path = OUT / 'artifact_manifest.json'
    if args.check_only:
        check('manifest_covers_all_bundle_files', set(read(manifest_path)) ==
              {p.relative_to(OUT).as_posix() for p in artifacts if p.is_relative_to(OUT) and p != manifest_path})
        for name, digest in read(manifest_path).items():
            check('manifest_sha256:' + name, sha(OUT / name) == digest)
    else:
        for name in ['render_summary.json', 'layout_check.json']:
            target = OUT / 'data' / name
            source = diagnostics / name
            if source.resolve() != target.resolve():
                target.write_bytes(source.read_bytes())
        result = {'status': 'passed', 'check_count': len(checks), 'checks': checks,
                  'pdf_pages': len(pdf), 'pdf_bookmarks': len(pdf.get_toc()), 'body_figures': image_count,
                  'evidence_sources': len(index), 'retrieval_rows': len(retrieval_rows), 'pipeline_repeat_rows': len(repeats),
                  'original_source_byte_matches': original_matches, 'git_ref_checked': args.git_ref,
                  'pdf_bytes': (OUT / 'technical_report.pdf').stat().st_size,
                  'html_bytes': (OUT / 'technical_report.html').stat().st_size, 'inference_runs_for_report': 0}
        (OUT / 'data/verification_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        included = sorted(p for p in OUT.rglob('*') if p.is_file() and p != manifest_path and
                          not any(part in ['.report_check', '__pycache__'] for part in p.parts))
        manifest_path.write_text(json.dumps({p.relative_to(OUT).as_posix(): sha(p) for p in included},
                                           ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'pages': len(pdf), 'body_figures': image_count,
                      'evidence_sources': len(index), 'git_ref': args.git_ref, 'check_only': args.check_only}))


if __name__ == '__main__':
    main()
