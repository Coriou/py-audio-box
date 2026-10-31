import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from lib.spoken_text import canonical_json, render


def amount(value, locale='fr-FR', literal=None):
    whole, _, fraction = value.partition('.')
    literal = literal or whole + ((',' if locale == 'fr-FR' else '.') + fraction if fraction else '') + ' euros'
    return dict(original_text=literal, locale=locale, profile='numeric-v1', annotations=[dict(type='amount', start=0, end=len(literal), literal=literal, currency='EUR', amount=value)])


class SpokenTextTests(unittest.TestCase):
    def test_french_grammar(self):
        for value, expected in [('0','zéro euro'), ('1','un euro'), ('71','soixante et onze euros'), ('80','quatre-vingts euros'), ('81','quatre-vingt-un euros'), ('91','quatre-vingt-onze euros'), ('200','deux cents euros'), ('1000','mille euros'), ('1250.50','mille deux cent cinquante euros et cinquante centimes'), ('200000','deux cent mille euros'), ('80000','quatre-vingt mille euros'), ('999999.99','neuf cent quatre-vingt-dix-neuf mille neuf cent quatre-vingt-dix-neuf euros et quatre-vingt-dix-neuf centimes'), ('0.01','zéro euro et un centime')]:
            with self.subTest(value=value):
                self.assertEqual(render(amount(value))['spoken_text'], expected)

    def test_english(self):
        for value, expected in [('1250.50','one thousand two hundred and fifty euros and fifty cents'), ('1001.01','one thousand and one euros and one cent'), ('80','eighty euros'), ('1','one euro'), ('0','zero euros')]:
            self.assertEqual(render(amount(value, 'en-GB'))['spoken_text'], expected)

    def test_identifiers_and_untouched_text(self):
        for locale, expected in [('fr-FR','a, zéro zéro deux zéro quatre sept, version bé; 12:30 le 01/02.'), ('en-GB','A, zero zero two zero four seven, version bee; 12:30 le 01/02.')]:
            text = 'A-002047, version B; 12:30 le 01/02.'
            annotations = [dict(type='identifier', start=s, end=e, literal=text[s:e], policy='spell-letters-digits') for s,e in [(0,8),(18,19)]]
            self.assertEqual(render(dict(original_text=text, locale=locale, profile='numeric-v1', annotations=annotations))['spoken_text'], expected)

    def test_provenance_determinism_and_no_mutation(self):
        request = amount('1250.50', literal='1 250,50 euros')
        before = copy.deepcopy(request)
        result = render(request)
        self.assertEqual(request, before)
        self.assertEqual(canonical_json(result), canonical_json(render(request)))
        fixture = amount('1250.50', literal='1 250,50 euros')
        fixture['original_text'] = 'Le budget est de 1 250,50 euros, sans dépasser ce montant.'
        fixture['annotations'][0].update(start=17, end=31)
        self.assertEqual(render(fixture)['render_sha256'], '98f4fd6218346d02ffdf39c0e093992e1a3762b9803ff849ffbddf504e79cd9e')
        digest = result.pop('render_sha256')
        self.assertEqual(digest, hashlib.sha256(canonical_json(result).encode()).hexdigest())
        for field, value in [('version',2), ('original_text','changed'), ('spoken_text','changed'), ('locale','en-GB'), ('profile','future'), ('annotations',[])]:
            changed = dict(result, **{field:value})
            self.assertNotEqual(digest, hashlib.sha256(canonical_json(changed).encode()).hexdigest())
        result['annotations'][0]['literal'] = 'changed'
        self.assertEqual(request, before)

    def test_minor_units_and_zero_cents(self):
        request = amount('1.00')
        a = request['annotations'][0]
        del a['amount']
        a['minor_units'] = 100
        self.assertEqual(render(request)['spoken_text'], 'un euro')
        for bad in [True, -1, 100000000, 1.0]:
            a['minor_units'] = bad
            with self.assertRaises(ValueError): render(request)

    def test_rejections(self):
        base = amount('1250.50')
        changes = [dict(amount=x) for x in ['-1','+1','1e3','1,250.50','01','NaN','1000000','0.001',1.2,True]]
        changes += [dict(currency='USD'), dict(extra=1), dict(start=True), dict(end=999), dict(literal='drift'), dict(type='date'), dict(minor_units=125050)]
        for change in changes:
            request = copy.deepcopy(base)
            request['annotations'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): render(request)
        for literal in ['1,250.50 euros','1.250,50 euros','1250,50 EUR','1 250 euros']:
            with self.assertRaises(ValueError): render(amount('1250.50', literal=literal))
        for change in [dict(profile='signed-v1'), dict(locale='fr'), dict(extra=True), dict(original_text='x\n'), dict(original_text='x'*4097)]:
            with self.assertRaises(ValueError): render(dict(base, **change))
        for spans in [[(0,1),(0,1)], [(-1,1)], [(0,0)]]:
            request = dict(original_text='AA', locale='en-GB', profile='numeric-v1', annotations=[dict(type='identifier', start=s,end=e,literal='A',policy='spell-letters-digits') for s,e in spans])
            with self.assertRaises(ValueError): render(request)

    def test_repeated_spans_and_expansion_limit(self):
        request = dict(original_text='B B', locale='fr-FR', profile='numeric-v1', annotations=[dict(type='identifier', start=s,end=s+1,literal='B',policy='spell-letters-digits') for s in (0,2)])
        self.assertEqual(render(request)['spoken_text'], 'bé bé')
        request['original_text'] = 'W'*4096
        request['annotations'] = [dict(type='identifier',start=0,end=4096,literal=request['original_text'],policy='spell-letters-digits')]
        with self.assertRaises(ValueError): render(request)
        request = dict(original_text='12:30 01/02 -5',locale='fr-FR',profile='numeric-v1',annotations=[])
        self.assertEqual(render(request)['spoken_text'], request['original_text'])

    def test_preview(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/preview-spoken-text.py'
        payload = canonical_json(amount('1250.50'))
        first = subprocess.run([sys.executable,str(script)],input=payload,text=True,capture_output=True,check=True).stdout
        second = subprocess.run([sys.executable,str(script)],input=payload,text=True,capture_output=True,check=True).stdout
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), render(amount('1250.50')))
        bad = subprocess.run([sys.executable,str(script)],input='{"profile":1,"profile":2}',text=True,capture_output=True)
        self.assertEqual(bad.returncode,2)
        self.assertEqual(bad.stdout,'')


if __name__ == '__main__':
    unittest.main()
