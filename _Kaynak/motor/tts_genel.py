"""Genel seslendirme işçisi: Türkçe ses denemesi için açık modeller (Pocket TTS Turkish, EMA Lightning, Chatterbox).
ses.py "genel" motorunda çağırır; motor kendi sanal ortamında çalışır.

Kullanım: <ortam python> -X utf8 tts_genel.py is.json
is.json: [{"metin": "...", "wav": "çıkış yolu"}, ...]
Ses ortam değişkeniyle gelir: AKADEMI_TTS_SES = "<motor>:<ses>"
  pocket:male_1 | pocket:female_2      wite-tech/pocket-tts-turkish (CC-BY-4.0), hazır sentetik sesler
  ema:varsayilan                       canberkkkkkk/ema-lightning (Apache-2.0), tek ses
  chatterbox:male_1 | chatterbox:female_2   ResembleAI/chatterbox çok dilli (MIT); referans ses Pocket TTS'in aynı adlı
                                       sentetik sesi (gerçek bir kişinin sesi kopyalanmaz)
Kelime zamanı vermez; ses.py zamanları harf sayısına göre orantılı dağıtır (Piper ve Kokoro gibi).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time


def sayilar_yaziyla(metin):
    """Chatterbox rakamları okuyamıyor: sayılar Türkçe yazıya çevrilir (ondalık virgül ve nokta ile)."""
    from num2words import num2words

    def ondalik(m):
        a, b = m.group(1), m.group(2)
        return f"{num2words(int(a), lang='tr')} virgül {' '.join(num2words(int(c), lang='tr') for c in b) if b.startswith('0') else num2words(int(b), lang='tr')}"
    metin = re.sub(r'(\d+)[,.](\d+)', ondalik, metin)
    return re.sub(r'\d+', lambda m: num2words(int(m.group()), lang='tr'), metin)


def pocket(isler, ses):
    """Önce Python arayüzü denenir; olmazsa komut satırı (her cümle için ayrı çağrı)."""
    try:
        from pocket_tts_turkish import TurkishTTS
        import soundfile as sf
        m = TurkishTTS.from_pretrained()
        for j in isler:
            x = m.generate(j['metin'], voice=ses)
            sr = getattr(m, 'sample_rate', None) or getattr(m, 'sr', 24000)
            if isinstance(x, tuple):
                x, sr = x
            if hasattr(x, 'cpu'):
                x = x.squeeze().cpu().numpy()
            sf.write(j['wav'], x, sr, subtype='PCM_16')
        return
    except ImportError:
        pass
    for j in isler:
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(j['metin'])
        subprocess.run(['pocket-tts-turkish', 'generate', '--text-file', f.name, '--voice', ses, '-o', j['wav']], check=True)


def ema(isler, ses):
    from ema_lightning import EMA
    m = EMA()
    for j in isler:
        m.say(j['metin'], path=j['wav'])


def chatterbox(isler, ses):
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    ref = hf_hub_download('wite-tech/pocket-tts-turkish', f'voices/{ses}.wav')
    try:
        m = ChatterboxMultilingualTTS.from_pretrained(device='cpu', t3_model='v3')
    except TypeError:
        m = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
    for j in isler:
        wav = m.generate(sayilar_yaziyla(j['metin']), language_id='tr', audio_prompt_path=ref, exaggeration=0.5, cfg_weight=0.4)
        # torchaudio.save yeni sürümde torchcodec istiyor; soundfile ile yazılır
        sf.write(j['wav'], wav.squeeze().cpu().numpy(), m.sr, subtype='PCM_16')


def main(is_dosyasi):
    isler = json.load(open(is_dosyasi, encoding='utf-8'))
    motor, _, ses = os.environ['AKADEMI_TTS_SES'].partition(':')
    bas = time.time()
    {'pocket': pocket, 'ema': ema, 'chatterbox': chatterbox}[motor](isler, ses)
    sure = time.time() - bas
    # ses denemesi sayfası için: seslendirme süresi (hız ölçütü)
    with open(os.path.join(os.path.dirname(os.path.abspath(is_dosyasi)), '_tts_sure.json'), 'w', encoding='utf-8') as f:
        json.dump({'ses': os.environ['AKADEMI_TTS_SES'], 'cumle': len(isler), 'saniye': round(sure, 1)}, f)
    print(f'tamam: {len(isler)} cümle, {sure:.0f} sn')


if __name__ == '__main__':
    main(sys.argv[1])
