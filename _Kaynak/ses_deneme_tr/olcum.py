"""Türkçe ses denemesi: bir dersin seslendirmesini konuşma tanımayla ölçer (anlaşılırlık).

Kullanım: python -X utf8 olcum.py <video_klasoru> <cikis.json>
Ses ayarı ses.py ile aynı yoldan gelir (AKADEMI_SES_AYAR_TR); cümle sesleri _tts önbelleğinden okunur.
Her cümle faster-whisper "medium" ile Türkçe yazıya dökülür; okunması gereken metinle karşılaştırılır:
  KHO (kelime hata oranı) ve HHO (harf hata oranı), noktalama ve büyük/küçük harf yok sayılarak.
Rakamlar iki tarafta da yazıya çevrilir (whisper "4000" ya da "dört bin" yazabilir).
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

MOTOR = pathlib.Path(__file__).resolve().parents[1] / 'motor'
sys.path.insert(0, str(MOTOR))
from ses import MARKA, FF, cumle_al, okunus, onbellek_yolu  # noqa: E402


def yaziya(metin):
    from num2words import num2words
    metin = re.sub(r'(\d+)[,.](\d+)', lambda m: f"{m.group(1)} virgül {m.group(2)}", metin)
    return re.sub(r'\d+', lambda m: ' ' + num2words(int(m.group()), lang='tr') + ' ', metin)


def sade(metin):
    metin = yaziya(metin).replace('İ', 'i').replace('I', 'ı').lower()
    return ''.join(ch if ch.isalnum() else ' ' for ch in metin).split()


def mesafe(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        o, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            o, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, o + (x != y))
    return d[-1]


def main(klasor, cikis):
    klasor = pathlib.Path(klasor).resolve()
    v = json.loads((klasor / 'metin.json').read_text(encoding='utf-8'))
    sa = json.loads(os.environ['AKADEMI_SES_AYAR_TR']) if os.environ.get('AKADEMI_SES_AYAR_TR') else dict(MARKA['ses']['tr'])
    sa.setdefault('hiz', '')
    model = WhisperModel('medium', device='cpu', compute_type='int8')
    # önbellek dosyası yoksa (ör. ses modeli betikleri sonradan değişti) cümle, bitmiş dersin _ses.wav'ından
    # zamanlama.json'daki yerine göre kesilir
    zaman = json.loads((klasor / 'zamanlama.json').read_text(encoding='utf-8')) if (klasor / 'zamanlama.json').exists() else None
    tum = None
    yerler = [(sz['bas'] + c['b'], c['sure']) for sz in zaman['sahneler'] for c in sz['cumleler']] if zaman else []
    sira = 0
    kelime_hata = kelime = harf_hata = harf = 0
    sure = 0.0
    cumleler = []
    for s in v['sahneler']:
        for c in s.get('cumleler', []):
            yazi, sm = cumle_al(c)
            sm = okunus(sm or yazi, 'tr')
            dosya, _ = onbellek_yolu(sa, sm, klasor / '_tts')
            if dosya.exists():
                ham = subprocess.run([FF, '-v', 'error', '-i', str(dosya), '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'],
                                     capture_output=True, check=True).stdout
                x = np.frombuffer(ham, dtype=np.float32)
            else:
                if tum is None:
                    ham = subprocess.run([FF, '-v', 'error', '-i', str(klasor / '_ses.wav'), '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'],
                                         capture_output=True, check=True).stdout
                    tum = np.frombuffer(ham, dtype=np.float32)
                b, d = yerler[sira]
                x = tum[int(b * 16000):int((b + d) * 16000)]
            sira += 1
            sure += len(x) / 16000
            parcalar, bilgi = model.transcribe(x, language='tr', beam_size=5)
            duyulan = ' '.join(p.text.strip() for p in parcalar)
            a, b = sade(sm), sade(duyulan)
            kh = mesafe(a, b); hh = mesafe(' '.join(a), ' '.join(b))
            kelime_hata += kh; kelime += len(a); harf_hata += hh; harf += len(' '.join(a))
            cumleler.append({'yazi': yazi, 'duyulan': duyulan, 'kho': round(kh / max(1, len(a)), 3)})
    sonuc = {'ses': sa.get('ses'), 'motor': sa.get('motor'), 'cumle': len(cumleler),
             'kho': round(kelime_hata / max(1, kelime), 4), 'hho': round(harf_hata / max(1, harf), 4),
             'konusma_sn': round(sure, 1), 'harf_sn': round(harf / max(1, sure), 2),
             'en_kotu': sorted(cumleler, key=lambda c: -c['kho'])[:12]}
    t = klasor / '_tts' / '_tts_sure.json'
    if t.exists():
        sonuc['seslendirme_sn'] = json.loads(t.read_text(encoding='utf-8'))['saniye']
    pathlib.Path(cikis).write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"KHO {sonuc['kho']:.1%}  HHO {sonuc['hho']:.1%}  konuşma {sure / 60:.1f} dk")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
