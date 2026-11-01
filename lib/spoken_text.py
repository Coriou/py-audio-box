"""Pinned, model-free spoken text and frozen-plan corroboration."""
import hashlib
import json
import re
from decimal import Decimal

PROFILE = 'numeric-v1'
VERSION = 1
MAX_CHARS = 4096
_FR = 'zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize'.split()
_EN = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()


def _fr(n, before_mille=False):
    if n < 17:
        return _FR[n]
    if n < 20:
        return 'dix-' + _FR[n - 10]
    if n < 70:
        tens = {2: 'vingt', 3: 'trente', 4: 'quarante', 5: 'cinquante', 6: 'soixante'}[n // 10]
        r = n % 10
        return tens + (' et un' if r == 1 else '-' + _fr(r) if r else '')
    if n < 80:
        return 'soixante' + (' et onze' if n == 71 else '-' + _fr(n - 60))
    if n < 100:
        return 'quatre-vingt' + ('s' if n == 80 and not before_mille else '' if n == 80 else '-' + _fr(n - 80))
    if n < 1000:
        q, r = divmod(n, 100)
        return ('' if q == 1 else _fr(q) + ' ') + 'cent' + ('s' if q > 1 and not r and not before_mille else '') + (' ' + _fr(r, before_mille) if r else '')
    q, r = divmod(n, 1000)
    return ('' if q == 1 else _fr(q, True) + ' ') + 'mille' + (' ' + _fr(r) if r else '')


def _en(n):
    if n < 20:
        return _EN[n]
    if n < 100:
        q, r = divmod(n, 10)
        return ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'][q] + ('-' + _en(r) if r else '')
    if n < 1000:
        q, r = divmod(n, 100)
        return _en(q) + ' hundred' + (' and ' + _en(r) if r else '')
    q, r = divmod(n, 1000)
    return _en(q) + ' thousand' + ((' and ' if r < 100 else ' ') + _en(r) if r else '')


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _keys(obj, required, optional=()):
    if not isinstance(obj, dict) or not set(required) <= obj.keys() or obj.keys() - set(required) - set(optional):
        raise ValueError('missing or unknown fields')


def _amount(a, locale):
    _keys(a, ('type', 'start', 'end', 'literal', 'currency'), ('amount', 'minor_units'))
    if a['currency'] != 'EUR' or ('amount' in a) == ('minor_units' in a):
        raise ValueError('EUR and exactly one amount representation required')
    if 'amount' in a:
        value = a['amount']
        if not isinstance(value, str) or not re.fullmatch(r'(0|[1-9][0-9]{0,5})(\.[0-9]{1,2})?', value):
            raise ValueError('amount must be unsigned exact decimal string')
        minor = int(Decimal(value) * 100)
    else:
        minor = a['minor_units']
        if type(minor) is not int:
            raise ValueError('minor_units must be integer, not bool')
    if not 0 <= minor <= 99999999:
        raise ValueError('amount out of range')
    whole, cents = divmod(minor, 100)
    # Literal is locale-specific, includes the currency, and must corroborate value.
    sep, decimal = (' ', ',') if locale == 'fr-FR' else (',', '.')
    plain = str(whole)
    grouped = format(whole, ',').replace(',', sep)
    fractions = [''] if not cents else []
    fractions += [decimal + f'{cents:02d}']
    if cents % 10 == 0:
        fractions += [decimal + str(cents // 10)]
    if a['literal'] not in {w + f + ' euros' for w in (plain, grouped) for f in fractions}:
        raise ValueError('literal does not match locale/value (include euros)')
    number = _fr if locale == 'fr-FR' else _en
    result = number(whole) + (' euro' if whole == 1 or (locale == 'fr-FR' and whole == 0) else ' euros')
    if cents:
        result += (' et ' if locale == 'fr-FR' else ' and ') + number(cents)
        result += (' centime' if cents == 1 else ' centimes') if locale == 'fr-FR' else (' cent' if cents == 1 else ' cents')
    return result


# Explicit names, including multiword letters.
_FR_LETTERS = ['a','bé','cé','dé','e','effe','gé','ache','i','ji','ka','elle','emme','enne','o','pé','ku','erre','esse','té','u','vé','double vé','ixe','i grec','zède']
_EN_LETTERS = ['A','bee','cee','dee','E','ef','gee','aitch','I','jay','kay','el','em','en','O','pee','cue','ar','ess','tee','U','vee','double U','ex','why','zed']


def _identifier(a, locale):
    _keys(a, ('type', 'start', 'end', 'literal', 'policy'))
    if a['policy'] != 'spell-letters-digits' or not re.fullmatch(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', a['literal']):
        raise ValueError('unsupported identifier/policy')
    letters = _FR_LETTERS if locale == 'fr-FR' else _EN_LETTERS
    digits = _FR if locale == 'fr-FR' else _EN
    groups = []
    for part in a['literal'].split('-'):
        groups.append(' '.join(digits[int(c)] if c.isdigit() else letters[ord(c.upper()) - 65] for c in part))
    return ', '.join(groups)


def render(request):
    """Render exact code-point spans; reject unknown fields and all implicit policies."""
    _keys(request, ('original_text', 'locale', 'profile', 'annotations'))
    text = request['original_text']
    locale = request['locale']
    if request['profile'] != PROFILE or locale not in ('fr-FR', 'en-GB'):
        raise ValueError('unsupported profile/locale')
    if not isinstance(text, str) or not 0 < len(text) <= MAX_CHARS or not text.isprintable():
        raise ValueError('original_text must be printable, 1..4096 characters')
    annotations = request['annotations']
    if not isinstance(annotations, list) or len(annotations) > 64:
        raise ValueError('annotations must be a bounded list')
    pieces, previous = [], 0
    for a in annotations:
        if not isinstance(a, dict):
            raise ValueError('annotation must be object')
        start, end = a.get('start'), a.get('end')
        if type(start) is not int or type(end) is not int or not previous <= start < end <= len(text):
            raise ValueError('spans must be ordered, disjoint and in bounds')
        if a.get('literal') != text[start:end]:
            raise ValueError('literal drift')
        if a.get('type') == 'amount':
            replacement = _amount(a, locale)
        elif a.get('type') == 'identifier':
            replacement = _identifier(a, locale)
        else:
            raise ValueError('unsupported annotation type')
        pieces.extend((text[previous:start], replacement))
        previous = end
    pieces.append(text[previous:])
    spoken = ''.join(pieces)
    if len(spoken) > MAX_CHARS:
        raise ValueError('spoken_text exceeds 4096 characters')
    result = dict(original_text=text, spoken_text=spoken, locale=locale,
                  profile=PROFILE, version=VERSION,
                  annotations=json.loads(canonical_json(annotations)))
    result['render_sha256'] = hashlib.sha256(canonical_json(result).encode('utf-8')).hexdigest()
    return result


def rendering_request(request):
    """Validate the closed public option and render without changing request.text."""
    option = request['text_rendering']
    _keys(option, ('profile', 'version', 'locale', 'annotations'))
    if type(option['version']) is not int or option['version'] != VERSION:
        raise ValueError('unsupported rendering version')
    return render(dict(original_text=request['text'], **{k: option[k] for k in ('profile', 'locale', 'annotations')}))


def freeze(plan):
    """Trusted resolution only, after request replacement, before validation."""
    plan.pop('speech_text', None)
    if 'text_rendering' in plan['request']:
        if plan['operation'] != 'speech_generate':
            raise ValueError('speech_text_mismatch')
        plan['speech_text'] = rendering_request(plan['request'])
    speech_text(plan)


def speech_text(plan):
    """Pure fail-closed accessor. Recompute to corroborate; return retained bytes."""
    request = plan['request']
    opted = 'text_rendering' in request
    if opted != ('speech_text' in plan):
        raise ValueError('speech_text_mismatch')
    if not opted:
        return request.get('text', request.get('reference_text', ''))
    if plan['operation'] != 'speech_generate':
        raise ValueError('speech_text_mismatch')
    expected = rendering_request(request)
    record = plan['speech_text']
    if canonical_json(record) != canonical_json(expected):
        raise ValueError('speech_text_mismatch')
    resolved = plan['resolved']
    language = {'fr-FR': 'French', 'en-GB': 'English'}[record['locale']] if resolved['identity']['effective']['engine'] == 'qwen' else {'fr-FR': 'fr', 'en-GB': 'en'}[record['locale']]
    if resolved['settings'].get('language') != language:
        raise ValueError('speech_text_language_mismatch')
    limit = resolved['profile']['limits']['text_chars']
    if max(len(request['text']), len(record['spoken_text'])) > limit:
        raise ValueError('profile_limit')
    return record['spoken_text']
