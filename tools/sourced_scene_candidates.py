"""Build retrieval documents from hash-bound HTML quotations, without query text."""
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse

PARSER_VERSION = 'sourced_scene_candidates/v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _TextParser(HTMLParser):
    ignored = {'script', 'style', 'noscript', 'svg', 'template'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.fragments = []

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in self.ignored and self.hidden:
            self.hidden -= 1

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_data(self, data):
        text = re.sub(r'\s+', ' ', data).strip()
        if text and not self.hidden:
            self.fragments.append(text)


def extract_html_text(body, encoding='utf-8'):
    """Deterministic visible text; scripts, comments and image attributes excluded."""
    parser = _TextParser()
    parser.feed(body.decode(encoding))
    parser.close()
    return '\n'.join(parser.fragments) + '\n'


def _scene_key(value):
    from core_modules.organize.play_knowledge import chinese_number
    value = re.sub(r'[\s：:·•《》“”‘’]', '', value)
    number = r'([一二三四五六七八九十百\d]+)'
    match = re.match(r'^第' + number + r'幕(?:第)?' + number + r'[场景](.*)$', value)
    if match:
        return (chinese_number(match[1]), chinese_number(match[2]), match[3])
    match = re.match(r'^第' + number + r'([幕场景])(.*)$', value)
    if match:
        index = chinese_number(match[1])
        return (index, None, match[3]) if match[2] == '幕' else (None, index, match[3])
    return (None, None, value)


def _inside_bundle(folder, value):
    relative = Path(value)
    path = (folder / relative).resolve()
    if relative.is_absolute() or folder not in path.parents:
        raise ValueError('source_snapshot_must_be_inside_bundle')
    return path


def apply_candidate_bundle(manifest, base_path, bundle_path, registry):
    """Replace only explicit source matches; expanded pools retain every old scene."""
    from core_modules.organize.identity_registry import stable_id
    base_path, bundle_path = Path(base_path).resolve(), Path(bundle_path).resolve()
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    allowed = {'schema_version', 'version', 'base_manifest_sha256', 'pool_mode',
               'sources', 'replacements', 'additions', 'notes'}
    if set(bundle) - allowed:
        raise ValueError('unexpected_candidate_bundle_fields')
    if bundle.get('schema_version') != 1 or not bundle.get('version'):
        raise ValueError('invalid_candidate_bundle_version')
    if bundle.get('base_manifest_sha256') != digest(base_path):
        raise ValueError('candidate_base_manifest_changed')
    if manifest != json.loads(base_path.read_text(encoding='utf-8')) or 'candidate_texts' in manifest:
        raise ValueError('candidate_bundle_requires_original_base_manifest')
    mode = bundle.get('pool_mode')
    if mode not in {'same_pool', 'expanded_pool'}:
        raise ValueError('invalid_candidate_pool_mode')
    if mode == 'same_pool' and bundle.get('additions'):
        raise ValueError('same_pool_cannot_add_scenes')

    texts, sources, files = {}, {}, {}
    for source in bundle['sources']:
        source_id = source['source_id']
        if source_id in sources:
            raise ValueError('duplicate_candidate_source')
        parsed = urlparse(source['url'])
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('invalid_candidate_source_url')
        if source['kind'] not in {'official_plot', 'official_performance_report',
                                  'media_performance_report', 'numbering_reference'}:
            raise ValueError('invalid_candidate_source_kind')
        path = _inside_bundle(bundle_path.parent, source['snapshot'])
        if digest(path) != source['snapshot_sha256']:
            raise ValueError('candidate_source_snapshot_changed')
        texts[source_id] = extract_html_text(path.read_bytes(), source.get('encoding', 'utf-8'))
        sources[source_id] = source
        files[str(path)] = source['snapshot_sha256']

    def quotation(evidence, document=False):
        source_id = evidence['source_id']
        if source_id not in texts:
            raise ValueError('unknown_candidate_source')
        if document and sources[source_id]['kind'] == 'numbering_reference':
            raise ValueError('numbering_reference_is_not_plot_text')
        start, end = evidence['start'], evidence['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(texts[source_id]):
            raise ValueError('invalid_candidate_quote_span')
        if texts[source_id][start:end] != evidence['quote']:
            raise ValueError('candidate_quote_does_not_match_snapshot')
        return evidence['quote']

    original = {(s['play_id'], s['scene_id']): s for s in manifest['scenes']}
    if len(original) != len(manifest['scenes']):
        raise ValueError('duplicate_base_candidate')
    queries = {row['play_id'] for row in manifest['rows']}
    documents, changes, additions, seen = deepcopy(manifest['scenes']), [], [], set()
    positions = {(s['play_id'], s['scene_id']): i for i, s in enumerate(documents)}
    ids = {s['scene_id'] for s in documents}
    vocabulary = {(s['play_id'], _scene_key(s['label'])) for s in documents}
    for is_new, rows in ((False, bundle['replacements']), (True, bundle.get('additions', []))):
        for row in rows:
            if set(row) - {'play_id', 'scene_id', 'label', 'evidence', 'mapping', 'notes'}:
                raise ValueError('unexpected_candidate_document_fields')
            key = (row['play_id'], row['scene_id'])
            if key in seen:
                raise ValueError('duplicate_candidate_replacement')
            seen.add(key)
            if row['play_id'] not in queries or row['play_id'] not in registry.plays:
                raise ValueError('candidate_play_has_no_queries')
            if not is_new and (key not in original or row['label'] != original[key]['label']):
                raise ValueError('candidate_vocabulary_changed')
            if is_new:
                if row['scene_id'] in ids or (row['play_id'], _scene_key(row['label'])) in vocabulary:
                    raise ValueError('duplicate_added_candidate')
                if row['scene_id'] != stable_id('scene', row['play_id'] + ':' + row['label']):
                    raise ValueError('added_candidate_id_does_not_match_label')
            mapping = row['mapping']
            if mapping.get('method') != 'explicit_source_numbering':
                raise ValueError('candidate_mapping_is_not_explicit')
            context = quotation(mapping)
            heading = mapping['source_heading']
            if heading not in context:
                raise ValueError('candidate_heading_not_in_source')
            label_key, heading_key = _scene_key(row['label']), _scene_key(heading)
            if label_key[:2] != heading_key[:2] or (label_key[2] and label_key[2] not in heading_key[2]):
                raise ValueError('candidate_scene_number_or_title_mismatch')
            if not row['evidence']:
                raise ValueError('candidate_has_no_plot_evidence')
            if any(e['source_id'] != mapping['source_id'] or
                   not mapping['start'] <= e['start'] < e['end'] <= mapping['end']
                   for e in row['evidence']):
                raise ValueError('candidate_quote_outside_mapped_section')
            quotes = [quotation(e, document=True) for e in row['evidence']]
            text = registry.plays[row['play_id']] + ' ' + row['label'] + ' ' + ' '.join(
                re.sub(r'\s+', ' ', quote).strip() for quote in quotes)
            refs = [sources[e['source_id']]['url'] + f"#text-{e['start']}-{e['end']}" for e in row['evidence']]
            document = {'play_id': row['play_id'], 'scene_id': row['scene_id'], 'label': row['label'],
                        'text': text, 'text_source': 'sourced_plot_excerpt_and_scene_label',
                        'source_ref': refs[0], 'source_refs': refs, 'evidence': row['evidence'],
                        'mapping': mapping}
            if is_new:
                documents.append(document)
                additions.append(row['scene_id'])
                ids.add(row['scene_id'])
                vocabulary.add((row['play_id'], label_key))
            else:
                documents[positions[key]] = document
                changes.append(row['scene_id'])
    result = deepcopy(manifest)
    result['scenes'] = documents
    result['candidate_texts'] = {
        'version': bundle['version'], 'parser': PARSER_VERSION, 'pool_mode': mode,
        'bundle_path': str(bundle_path), 'bundle_sha256': digest(bundle_path),
        'base_manifest_path': str(base_path), 'base_manifest_sha256': digest(base_path),
        'source_files_sha256': files, 'replaced_scene_ids': changes, 'added_scene_ids': additions,
    }
    return result


def candidate_files_unchanged(manifest):
    meta = manifest.get('candidate_texts')
    if not meta:
        return True
    files = {**meta['source_files_sha256'], meta['bundle_path']: meta['bundle_sha256'],
             meta['base_manifest_path']: meta['base_manifest_sha256']}
    try:
        return all(digest(path) == value for path, value in files.items())
    except OSError:
        return False


def validate_sourced_manifest(manifest, registry):
    """Rebuild saved sourced inputs, detecting changes before a paired model run."""
    meta = manifest.get('candidate_texts')
    if not meta:
        return
    if not candidate_files_unchanged(manifest):
        raise ValueError('prepared_candidate_sources_changed')
    base = json.loads(Path(meta['base_manifest_path']).read_text(encoding='utf-8'))
    rebuilt = apply_candidate_bundle(base, meta['base_manifest_path'], meta['bundle_path'], registry)
    if manifest != rebuilt:
        raise ValueError('prepared_candidate_manifest_changed')
