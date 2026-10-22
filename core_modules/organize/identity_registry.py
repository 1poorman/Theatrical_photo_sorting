"""Explicit actor identities and play-specific roles; legacy names remain hints.

The optional identity_registry.json in a gallery is read-only here. Only
human-verified assignments are used. A face score confirms an identity, never
the character played in a different anchor photograph.
"""
import hashlib
import json
import math
from pathlib import Path
import re


REGISTRY_NAME = 'identity_registry.json'


def stable_id(kind, value):
    return kind + '-' + hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]


def legacy_actor(person):
    return re.split(r'[（(]|饰', Path(person).name, maxsplit=1)[0].strip()


def registry_path(db_root, explicit=None):
    if explicit:
        path = Path(explicit).absolute()
        if not path.is_file():
            raise ValueError('identity_registry_missing')
        return path
    return Path(db_root).resolve() / REGISTRY_NAME


def registry_digest(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('identity_registry_must_be_regular_file')
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError('identity_registry_must_be_regular_file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('identity_registry_invalid_' + field)
    return value.strip()


def _id(value, field):
    value = _text(value, field)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value):
        raise ValueError('identity_registry_invalid_' + field)
    return value


class IdentityRegistry:
    def __init__(self, data=None):
        self.by_dir, self.identities, self.plays, self.play_aliases = {}, {}, {}, {}
        self._display_ids = {}
        if data is None:
            return
        if not isinstance(data, dict) or data.get('schema_version') != 1:
            raise ValueError('identity_registry_schema_version')
        if not isinstance(data.get('identities'), list) or not isinstance(data.get('plays'), list):
            raise ValueError('identity_registry_requires_identities_and_plays')
        for play in data['plays']:
            play_id = _id(play['play_id'], 'play_id')
            title = _text(play['title'], 'play_title')
            if play_id in self.plays or not isinstance(play.get('aliases', []), list):
                raise ValueError('identity_registry_duplicate_or_invalid_play')
            self.plays[play_id] = title
            for alias in [play_id, title] + play.get('aliases', []):
                alias = _text(alias, 'play_alias')
                if alias in self.play_aliases and self.play_aliases[alias] != play_id:
                    raise ValueError('identity_registry_ambiguous_play_alias')
                self.play_aliases[alias] = play_id
        assigned_dirs, role_names = set(), {}
        for row in data['identities']:
            identity_id = _id(row['identity_id'], 'identity_id')
            actor_name = _text(row['actor_name'], 'actor_name')
            if identity_id in self.identities or type(row.get('verified')) is not bool:
                raise ValueError('identity_registry_duplicate_or_unverified_identity_field')
            if not isinstance(row.get('gallery_dirs'), list) or not isinstance(row.get('roles', []), list):
                raise ValueError('identity_registry_invalid_lists')
            roles = []
            for role in row.get('roles', []):
                play_id, role_id = _id(role['play_id'], 'play_id'), _id(role['role_id'], 'role_id')
                role_name = _text(role['role_name'], 'role_name')
                if play_id not in self.plays or type(role.get('verified')) is not bool:
                    raise ValueError('identity_registry_invalid_role')
                source = _text(role.get('source_ref'), 'source_ref')
                if (play_id, role_id) in role_names and role_names[play_id, role_id] != role_name:
                    raise ValueError('identity_registry_conflicting_role_name')
                role_names[play_id, role_id] = role_name
                roles.append({'play_id': play_id, 'role_id': role_id, 'role_name': role_name,
                              'source_ref': source, 'verified': role['verified']})
            identity = {'identity_id': identity_id, 'actor_name': actor_name,
                        'identity_source': 'verified_registry' if row['verified'] else 'unverified_registry',
                        'verified': row['verified'], 'roles': roles}
            self.identities[identity_id] = identity
            if row['verified']:
                labels = {actor_name}
                labels.update(f"{actor_name}（饰{role['role_name']}）" for role in roles if role['verified'])
                for label in labels:
                    self._display_ids.setdefault(label, set()).add(identity_id)
            for directory in row['gallery_dirs']:
                directory = _text(directory, 'gallery_dir')
                if directory in ('.', '..') or '/' in directory or '\\' in directory or directory in assigned_dirs:
                    raise ValueError('identity_registry_duplicate_or_invalid_gallery_dir')
                assigned_dirs.add(directory)
                if row['verified']:
                    self.by_dir[directory] = identity

    @classmethod
    def load(cls, path):
        path = Path(path)
        if registry_digest(path) is None:
            return cls()
        try:
            return cls(json.loads(path.read_text(encoding='utf-8')))
        except (KeyError, TypeError) as exc:
            raise ValueError('identity_registry_invalid_structure') from exc

    def identity(self, directory):
        directory = Path(directory).name
        if directory in self.by_dir:
            return self.by_dir[directory]
        actor = legacy_actor(directory)
        # Explicitly separated names must not be silently joined by a fallback.
        collision = any(row['actor_name'] == actor and row['verified'] for row in self.identities.values())
        return {'identity_id': stable_id('legacy', directory if collision else actor),
                'actor_name': actor, 'identity_source': 'gallery_name_unverified',
                'verified': False, 'roles': []}

    def resolve(self, directory, play=None):
        identity = self.identity(directory)
        play_id = self.play_aliases.get(play)
        roles = {(r['role_id'], r['role_name']): r for r in identity['roles']
                 if r['verified'] and r['play_id'] == play_id}
        role = next(iter(roles.values())) if len(roles) == 1 else None
        reason = 'matched' if role else ('unknown_play' if play_id is None else
                                       'ambiguous_role' if len(roles) > 1 else 'no_verified_role')
        return {'identity_id': identity['identity_id'], 'actor_name': identity['actor_name'],
                'identity_source': identity['identity_source'], 'play_id': play_id,
                'role_id': role['role_id'] if role else None, 'role_name': role['role_name'] if role else None,
                'role_source': role['source_ref'] if role else None, 'role_status': reason}

    def display_name(self, identity):
        """Disambiguate known labels across photos and plays, including solo faces."""
        label = identity['actor_name']
        if identity['role_name']:
            label += f"（饰{identity['role_name']}）"
        if any(other != identity['identity_id'] for other in self._display_ids.get(label, ())):
            return f"{label} [{identity['identity_id']}]"
        return label


def decide_identity(scores, persons, registry, threshold, margin):
    """Rank identities over the full gallery before album filtering or naming."""
    if len(scores) != len(persons):
        raise ValueError('face_anchor_count_mismatch')
    best = {}
    for score, person in zip(scores, persons):
        score = float(score)
        if not math.isfinite(score):
            continue
        identity_id = registry.identity(person)['identity_id']
        current = best.get(identity_id)
        if current is None or score > current[0] or (score == current[0] and person < current[1]):
            best[identity_id] = (score, person)
    ranked = sorted(best.items(), key=lambda pair: (-pair[1][0], pair[0]))
    result = {'status': 'unknown', 'reason': 'no_valid_anchor', 'identity_id': None,
              'anchor_dir': None, 'score': None, 'second_score': None, 'margin': None,
              'threshold': float(threshold), 'required_margin': float(margin)}
    if not ranked:
        return result
    identity_id, (score, person) = ranked[0]
    second = ranked[1][1][0] if len(ranked) > 1 else None
    result.update(score=score, second_score=second, margin=score - second if second is not None else None)
    if score < threshold:
        result['reason'] = 'below_threshold'
    elif second is not None and score - second < margin:
        result['reason'] = 'ambiguous_identity'
    else:
        result.update(status='recognized', reason=None, identity_id=identity_id, anchor_dir=person)
    return result


def resolve_hits(hits, registry, play=None):
    """Return display hits plus stable identities without inferring anchor roles."""
    identities = {}
    for directory, score in hits.items():
        row = registry.resolve(directory, play)
        row.update(score=float(score), anchor_dir=directory)
        if row['score'] > identities.get(row['identity_id'], {'score': -float('inf')})['score']:
            identities[row['identity_id']] = row
    rows = sorted(identities.values(), key=lambda row: (-row['score'], row['identity_id']))
    labels = {}
    for row in rows:
        label = registry.display_name(row)
        labels.setdefault(label, []).append(row)
    display = {}
    for label, members in labels.items():
        for row in members:
            row['display_name'] = label if len(members) == 1 else f"{label} [{row['identity_id']}]"
            row['score'] = round(row['score'], 4)
            display[row['display_name']] = row['score']
    return display, rows
