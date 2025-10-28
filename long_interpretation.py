#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uzun rüya tabiri üretici (veri tabanına dayalı)
- Veri: _interpretations_contains_all.json (word, meaning)
- Girdi: Serbest metin rüya anlatımı veya anahtar kelimeler
- Çıktı: Uzatılmış, bölümlere ayrılmış yorum (özet + detay)

Kullanım:
  python long_interpretation.py --text "Rüyamda yılan ve deniz gördüm" --length long

Notlar:
- Bu script ML kullanmaz; veri tabanı + şablon ile uzunlaştırır.
- Daha da doğal metin için bir LLM ile birleştirilebilir (RAG tarzı).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

DATA_FILE = "_interpretations_contains_all.json"

# Basit olumlu/olumsuz kelime listeleri (heuristic)
POS_WORDS = {
    "hayır", "hayra", "müjde", "sevinç", "ferah", "bereket", "rızık", "nimet",
    "afiyet", "şifa", "başarı", "muvaffakiyet", "izzet", "şeref", "refah",
}
NEG_WORDS = {
    "keder", "üzüntü", "musibet", "hastalık", "zarar", "kötü", "belâ", "tehlike",
    "kaygı", "sıkıntı", "düşman", "fitne", "yoksulluk", "hüzün", "felâket",
}

# Bazı yardımcı fonksiyonlar

def normalize_text(s: str) -> str:
    """Küçük harfe çevir, unicode normalize et, fazla boşlukları sadeleştir."""
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    # Yeni satırları boşluğa çevir
    s = re.sub(r"\s+", " ", s)
    return s

_parenthet_re = re.compile(r"\s*\([^\)]*\)")

def strip_parentheticals(s: str) -> str:
    """Parantez içlerini kaldır (ACVE (..)-> ACVE)."""
    return _parenthet_re.sub("", s).strip()

_word_re = re.compile(r"[\wçğıöşüâîûÇĞİÖŞÜ]+", re.IGNORECASE)

def tokenize(s: str) -> List[str]:
    return _word_re.findall(s)

def fold_tr(s: str) -> str:
    """Türkçe için kaba bir eşleme: diakritikleri kaldır, özel harfleri yakın ASCII'ye çevir."""
    # Önce casefold ve NFD
    s = unicodedata.normalize("NFKC", s).casefold()
    # Özel Türkçe harfleri eşle
    trans = str.maketrans({
        "ı": "i", "İ": "i", "I": "i",
        "ş": "s", "Ş": "s",
        "ğ": "g", "Ğ": "g",
        "ç": "c", "Ç": "c",
        "ö": "o", "Ö": "o",
        "ü": "u", "Ü": "u",
        "â": "a", "î": "i", "û": "u",
    })
    s = s.translate(trans)
    # Bir de kombinasyon işaretlerini kaldır (genel amaçlı)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)
    # Boşluk sadeleştirme
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Türkçe kök çıkarımı için basit kurallar (heuristic) ---
VOWELS_A = set("aıou")
VOWELS_E = set("eiöü")
VOWELS = VOWELS_A | VOWELS_E

def last_vowel(stem: str) -> str:
    for ch in reversed(stem):
        if ch in VOWELS:
            return ch
    return "a"  # varsayılan

def guess_inf_suffix(stem: str) -> str:
    v = last_vowel(stem)
    return "mak" if v in VOWELS_A else "mek"

SUFFIXES = [
    "yormuş", "yordu", "yorsun", "yorum", "yorsam", "yorsak",
    "yorm",  # nadir
    "miş", "mış", "muş", "müş",
    "eceğim", "acağım", "ecek", "acak",
    "dik", "dık", "duk", "dük", "tik", "tık", "tuk", "tük",
    "dim", "dım", "dum", "düm", "tim", "tım", "tum", "tüm",
    "sin", "sın", "sun", "sün",
    "siniz", "sınız", "sunuz", "sünüz",
    "ler", "lar",
    "yor",  # en yaygın
]
SUFFIXES_F = sorted({fold_tr(s) for s in SUFFIXES}, key=len, reverse=True)

def tr_stem_candidates(token: str) -> List[str]:
    """Basit sapma: çekim eklerini budayıp kök/lemma adayları üretir.
    Dönüş örn: savaşıyormuş -> ["savaşı", "savaş", "savaşmak"]
    """
    # token zaten folded gelmeli
    t = token
    out = {t}
    # genel budama denemeleri
    for suf in SUFFIXES_F:
        if t.endswith(suf):
            base = t[: -len(suf)]
            out.add(base)
            # -yor* eklerinden sonra kalan sondaki ünlüyü de at (savaSi -> savas)
            if suf.startswith("yor") and base:
                if base[-1] in VOWELS:
                    out.add(base[:-1])
    # -mak/-mek varsa kaldırıp kök ekle
    for base in list(out):
        if base.endswith("mak") or base.endswith("mek"):
            out.add(base[:-3])
    # çok kısa olanları ele
    out = {s for s in out if len(s) >= 2}
    # kökten mastar oluştur (mak/mek)
    more = set()
    for s in out:
        if not (s.endswith("mak") or s.endswith("mek")):
            more.add(s + guess_inf_suffix(s))
    out |= more
    # ayrıca son harf uyumu: ı->i düzeltmesi gibi küçük katlamayı tekrar uygula
    out_folded = {fold_tr(s) for s in out}
    return sorted(out_folded)

# Bazı eş anlam/kök yakınlık haritası (folded formlar üzerinden)
SYN_EQUIV: Dict[str, List[str]] = {
    # öpüşmek -> öpmek
    "opusmek": ["opmek"],
    # savaşmak -> savaş
    "savasmak": ["savas"],
    # kavga etmek -> kavga
    "kavga etmek": ["kavga"],
}


def load_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Beklenen alanlar: word, meaning
    cleaned = []
    for row in data:
        w = row.get("word")
        m = row.get("meaning")
        if isinstance(w, str) and isinstance(m, str):
            cleaned.append({"word": w.strip(), "meaning": m.strip()})
    return cleaned


def build_index(entries: List[Dict[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Key: normalize edilmiş anahtar; Value: (orijinal_word, meaning) listesi.
    Aynı anahtar birden fazla maddeyi temsil edebilir.
    """
    idx: Dict[str, List[Tuple[str, str]]] = {}
    for e in entries:
        w = e["word"]
        m = e["meaning"]
        keys = set()
        # Orijinal ve parantezsiz halleri ekle
        keys.add(normalize_text(w))
        wp = strip_parentheticals(w)
        if wp != w:
            keys.add(normalize_text(wp))
        # Çok sözcüklü ifadeleri de anahtar olarak ekle (ör. "ADA TAVŞANI")
        # Ayrıca tekil kelimeleri de (ör. ADA, TAVŞANI) aşırı gürültü yaratır, o yüzden eklemiyoruz.
        for k in keys:
            idx.setdefault(k, []).append((w, m))
    return idx


def find_matches(text: str, index: Dict[str, List[Tuple[str, str]]], max_phr_len: int = 5) -> List[Tuple[str, str]]:
    """Metin içinde geçebilecek söz öbeklerini (dataset anahtarlarına göre) bul.
    - Token tabanlı n-gram eşleşmesi kullanır; bu sayede 'nar' ~ 'kenarında' gibi yanlış eşleşmelerden kaçınılır.
    - Önce uzun ifadeler (çok sözcüklü) sonra kısa olanlar aranır.
    """
    ntext = normalize_text(text)
    # "X gibi hisset..." kalıbını yakala ve X'i adaylara ekle
    m = re.search(r"([\wçğıöşüâîûÇĞİÖŞÜ]+)\s+gibi\s+hisset", ntext, re.IGNORECASE)
    extracted = []
    if m:
        extracted.append(m.group(1))

    ntext_f = fold_tr(ntext)
    toks = tokenize(ntext_f)
    for ex in extracted:
        toks.append(fold_tr(ex))

    # Tüm olası n-gramları çıkar (1..max_phr_len) ve bir küme halinde tut
    ngrams: List[str] = []
    L = len(toks)
    for n in range(min(max_phr_len, L), 0, -1):  # önce uzun n-gramlar
        for i in range(0, L - n + 1):
            ngrams.append(" ".join(toks[i : i + n]))
    ngram_set = set(ngrams)
    # Tekil tokenlardan kök/mastar adaylarını da ekle (örn. savaşıyordum -> savaş, savaşmak)
    stem_cands: List[str] = []
    for t in toks:
        stem_cands.extend(tr_stem_candidates(t))
    ngram_set |= set(stem_cands)
    # Eş anlam/denk kök genişletmesi
    syn_add = set()
    for c in list(ngram_set):
        if c in SYN_EQUIV:
            syn_add.update(SYN_EQUIV[c])
    ngram_set |= syn_add

    # İfadeleri uzunluğa göre sırala (daha uzun ifadeler önce)
    # Folded key -> (orijinal_word, meaning) listesi haritası oluştur
    folded_index: Dict[str, List[Tuple[str, str]]] = {}
    for k, v in index.items():
        fk = fold_tr(k)
        folded_index.setdefault(fk, []).extend(v)

    keys = sorted(folded_index.keys(), key=lambda k: len(k.split()), reverse=True)
    seen: set[str] = set()
    matches: List[Tuple[str, str]] = []
    for k in keys:
        if len(k.split()) > max_phr_len:
            continue
        if len(k) < 2:
            continue
        if k in ngram_set:
            for ow, m in folded_index[k]:
                uid = f"{ow}::{hash(m)}"
                if uid not in seen:
                    matches.append((ow, m))
                    seen.add(uid)
    # Son çare: uzun tek-kelime anahtarları metin içinde alt dize olarak ara (yanlış pozitifleri azaltmak için >=4)
    if not matches:
        for k in keys:
            if len(k.split()) == 1 and len(k) >= 4:
                if k in ntext_f:
                    for ow, m in folded_index[k]:
                        uid = f"{ow}::{hash(m)}"
                        if uid not in seen:
                            matches.append((ow, m))
                            seen.add(uid)

    return matches


def sentiment_hint(meaning: str) -> str:
    n = normalize_text(meaning)
    pos = sum(1 for w in POS_WORDS if w in n)
    neg = sum(1 for w in NEG_WORDS if w in n)
    if pos > neg and pos > 0:
        return "olumlu"
    if neg > pos and neg > 0:
        return "olumsuz"
    return "nötr/karışık"


def build_long_interpretation(
    text: str,
    matches: List[Tuple[str, str]],
    length: str = "long",
    top_k: int = 8,
) -> str:
    """Şablonlu uzun yorum metni üret."""
    if not matches:
        return (
            "Rüya anlatımında veri tabanımızdaki belirgin sembollere doğrudan bir eşleşme bulamadım. "
            "Daha net anahtar kelimeler (örn. yılan, deniz, altın, koç vb.) içeren bir anlatım paylaşırsan, "
            "daha isabetli ve detaylı bir yorum üretebilirim."
        )

    # En alakalı ilk N madde (şimdilik basitçe ilk gelenler)
    matches = matches[: max(1, top_k)]

    # Özet bölümü
    symbols = ", ".join(sorted({w for w, _ in matches}, key=len)[:6])
    summary = (
        f"Rüyanın genel teması şu semboller etrafında şekilleniyor: {symbols}. "
        "Bu semboller çoğunlukla niyetin, içsel hâlin ve yaklaşan gelişmelerin işaretleri olarak ele alınır."
    )

    # Semboller ve anlamlar (madde madde)
    lines = []
    for w, m in matches:
        hint = sentiment_hint(m)
        lines.append(f"- {w}: ({hint}) {m}")
    sym_block = "\n".join(lines)

    # Muhtemel yorum (uzatma)
    if length == "short":
        body = (
            "Sembollerin ortak anlamları bir araya getirildiğinde; mevcut dönemde niyetleri berraklaştırma, "
            "yakın çevreyle iletişimi güçlendirme ve somut adımlar atma vurgusu öne çıkıyor."
        )
    elif length == "medium":
        body = (
            "Bu rüya, bilinçaltının güncel kaygı ve beklentileri bir arada işlemesi gibi görünüyor. "
            "Özellikle öne çıkan semboller; ilişkilerde denge kurma, fırsatları kaçırmama ve riskleri tedricen yönetme çağrısı yapıyor. "
            "Bu süreçte sezgileri küçümsememek ve küçük sinyalleri ciddiye almak faydalı olacaktır."
        )
    else:  # long
        body = (
            "Sembollerin kesişim kümesi; niyet temizliği, kaynağa dönme ve adım adım ilerleme fikrini işaret ediyor. "
            "Rüya; maddi-manevi dengede kalmayı, aşırı uçlardan kaçınmayı ve ilişkilerde netlik aramayı öğütlüyor. "
            "Gündelik düzlemde bu, planların yazılı hâle getirilmesi, küçük ama sürdürülebilir alışkanlıklar kurulması ve "
            "duygusal refleksler tetiklendiğinde kısa bir durup nefesi düzenleme pratikleriyle desteklenebilir."
        )

    # Olumlu/olumsuz işaretler
    pos_count = sum(1 for _, m in matches if sentiment_hint(m) == "olumlu")
    neg_count = sum(1 for _, m in matches if sentiment_hint(m) == "olumsuz")
    if pos_count > neg_count:
        tone = "Genel ton olumlu; fırsat ve açılımlar biraz daha baskın görünüyor."
    elif neg_count > pos_count:
        tone = "Genel ton temkinli/olumsuz; riskleri fark edip koruyucu önlemler almak öne çıkıyor."
    else:
        tone = "Genel ton dengeli; hem fırsat hem de dikkat gerektiren işaretler birlikte görünüyor."

    # Öneriler
    recs = (
        "- Kısa vadeli (1-2 hafta) yapılabilir iki adım belirle.\n"
        "- Duygusal yoğunlukta 3 derin nefes + 10 dakikalık yürüyüş gibi bir ara ritüel uygula.\n"
        "- Bir cümlelik niyet yaz: ‘Şu konuda şu tarihe dek şu adımı atıyorum.’\n"
    )

    out = (
        "RÜYA TABİRİ (Uzun)\n"
        "====================\n\n"
        f"Özet\n----\n{summary}\n\n"
        f"İlgili Semboller ve Anlamları\n-------------------------------\n{sym_block}\n\n"
        f"Muhtemel Yorum\n---------------\n{body}\n\n"
        f"Genel Ton\n---------\n{tone}\n\n"
        f"Öneriler\n--------\n{recs}"
    )
    return out


def build_long_interpretation_structured(
    text: str,
    matches: List[Tuple[str, str]],
    length: str = "long",
    top_k: int = 8,
) -> Dict[str, object]:
    """Yapılandırılmış (structured) yorum döndür.
    Dönüş:
      {
        summary: str,
        items: [{word, hint, meaning}],
        body: str,
        tone: str,
        recommendations: [str],
      }
    Eşleşme yoksa items boş olur ve summary/body bilgilendirici metin içerir.
    """
    if not matches:
        msg = (
            "Rüya anlatımında veri tabanımızdaki belirgin sembollere doğrudan bir eşleşme bulamadım. "
            "Daha net anahtar kelimeler (örn. yılan, deniz, altın, koç vb.) içeren bir anlatım paylaşırsan, "
            "daha isabetli ve detaylı bir yorum üretebilirim."
        )
        return {
            "summary": msg,
            "items": [],
            "body": "",
            "tone": "nötr/karışık",
            "recommendations": [],
        }

    matches = matches[: max(1, top_k)]
    symbols = ", ".join(sorted({w for w, _ in matches}, key=len)[:6])
    summary = (
        f"Rüyanın genel teması şu semboller etrafında şekilleniyor: {symbols}. "
        "Bu semboller çoğunlukla niyetin, içsel hâlin ve yaklaşan gelişmelerin işaretleri olarak ele alınır."
    )

    items = []
    for w, m in matches:
        items.append({
            "word": w,
            "hint": sentiment_hint(m),
            "meaning": m,
        })

    if length == "short":
        body = (
            "Sembollerin ortak anlamları bir araya getirildiğinde; mevcut dönemde niyetleri berraklaştırma, "
            "yakın çevreyle iletişimi güçlendirme ve somut adımlar atma vurgusu öne çıkıyor."
        )
    elif length == "medium":
        body = (
            "Bu rüya, bilinçaltının güncel kaygı ve beklentileri bir arada işlemesi gibi görünüyor. "
            "Özellikle öne çıkan semboller; ilişkilerde denge kurma, fırsatları kaçırmama ve riskleri tedricen yönetme çağrısı yapıyor. "
            "Bu süreçte sezgileri küçümsememek ve küçük sinyalleri ciddiye almak faydalı olacaktır."
        )
    else:
        body = (
            "Sembollerin kesişim kümesi; niyet temizliği, kaynağa dönme ve adım adım ilerleme fikrini işaret ediyor. "
            "Rüya; maddi-manevi dengede kalmayı, aşırı uçlardan kaçınmayı ve ilişkilerde netlik aramayı öğütlüyor. "
            "Gündelik düzlemde bu, planların yazılı hâle getirilmesi, küçük ama sürdürülebilir alışkanlıklar kurulması ve "
            "duygusal refleksler tetiklendiğinde kısa bir durup nefesi düzenleme pratikleriyle desteklenebilir."
        )

    pos_count = sum(1 for it in items if it["hint"] == "olumlu")
    neg_count = sum(1 for it in items if it["hint"] == "olumsuz")
    if pos_count > neg_count:
        tone = "Genel ton olumlu; fırsat ve açılımlar biraz daha baskın görünüyor."
    elif neg_count > pos_count:
        tone = "Genel ton temkinli/olumsuz; riskleri fark edip koruyucu önlemler almak öne çıkıyor."
    else:
        tone = "Genel ton dengeli; hem fırsat hem de dikkat gerektiren işaretler birlikte görünüyor."

    recommendations = [
        "Kısa vadeli (1-2 hafta) yapılabilir iki adım belirle.",
        "Duygusal yoğunlukta 3 derin nefes + 10 dakikalık yürüyüş gibi bir ara ritüel uygula.",
        "Bir cümlelik niyet yaz: ‘Şu konuda şu tarihe dek şu adımı atıyorum.’",
    ]

    return {
        "summary": summary,
        "items": items,
        "body": body,
        "tone": tone,
        "recommendations": recommendations,
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Rüya tabirini veri tabanına dayanarak uzunlaştır.")
    parser.add_argument("--text", type=str, required=True, help="Rüya anlatımı veya anahtar kelimeler")
    parser.add_argument("--length", type=str, default="long", choices=["short", "medium", "long"], help="Detay seviyesi")
    parser.add_argument("--top-k", type=int, default=8, help="Alınacak en fazla sembol sayısı")
    parser.add_argument("--data", type=str, default=DATA_FILE, help="JSON veri yolu (varsayılan: _interpretations_contains_all.json)")
    args = parser.parse_args(argv)

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"Veri dosyası bulunamadı: {data_path}", file=sys.stderr)
        return 2

    try:
        entries = load_dataset(data_path)
    except Exception as e:
        print(f"Veri dosyası okunamadı: {e}", file=sys.stderr)
        return 3

    index = build_index(entries)
    matches = find_matches(args.text, index)
    output = build_long_interpretation(args.text, matches, length=args.length, top_k=args.top_k)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
