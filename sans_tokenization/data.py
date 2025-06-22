# Sanskrit Token Collector Using Vidyut
# Extracts dhatus, tiṅ-pratyayas, kṛt‑pratyayas, taddhita‑pratyayas, vibhaktis, upasargas, avyayas

import json
from vidyut import kosha
from vidyut.prakriya import (
    Vyakarana, Dhatu, Gana, Prayoga, Lakara, Purusha, Vacana, Pada, Krt, Taddhita
)

# --- Load Database ---
print("🔄 Loading Vidyut Kosha...")
db = kosha.Kosha("/home/kush/cooking/ml/sans_tokenization/data/kosha")
tokens = []

# --- Dhātus ---
print("🔍 Collecting unique dhatus...")
seen_dhatus = set()
for entry in db.dhatus():
    root = entry.dhatu.aupadeshika
    if root not in seen_dhatus:
        seen_dhatus.add(root)
        tokens.append({"token": root, "category": "dhatu", "meaning": entry.artha_sa or ""})

# --- Tiṅ-Pratyayas (sample extraction) ---
print("🔍 Collecting tiṅ-pratyayas...")
seen_ting = set()
vy = Vyakarana()
for root in list(seen_dhatus)[:50]:
    dh_obj = Dhatu.mula(root, Gana.Bhvadi)
    forms = vy.derive(Pada.Tinanta(
        dhatu=dh_obj, prayoga=Prayoga.Kartari,
        lakara=Lakara.Lat, purusha=Purusha.Prathama, vacana=Vacana.Eka
    ))
    for f in forms:
        last = f.history[-1].result[-1]
        if last not in seen_ting:
            seen_ting.add(last)
            tokens.append({"token": last, "category": "ting_pratyaya", "meaning": "verb suffix"})

# --- Kṛt-Pratyayas ---
print("🔍 Collecting kṛt-pratyayas...")
seen_krt = set()
for k in Krt.choices():
    suffix = str(k)
    if suffix not in seen_krt:
        seen_krt.add(suffix)
        tokens.append({"token": suffix, "category": "krt_pratyaya", "meaning": "kṛt suffix"})

# --- Taddhita-Pratyayas ---
print("🔍 Collecting taddhita-pratyayas...")
seen_tad = set()
for t in Taddhita.choices():
    suffix = str(t)
    if suffix not in seen_tad:
        seen_tad.add(suffix)
        tokens.append({"token": suffix, "category": "taddhita_pratyaya", "meaning": "taddhita suffix"})

# --- Upasargas ---
print("🔍 Adding upasargas...")
upasargas = ["pra", "pari", "apa", "sam", "abhi", "ni", "vi", "ud", "anu", "ava", "su", "durg", "prati", "ati", "adhi"]
for u in upasargas:
    tokens.append({"token": u, "category": "upasarga", "meaning": "verbal prefix"})

# --- Vibhaktis ---
print("🔍 Adding vibhaktis...")
vibhakti = ["h", "m", "ah", "au", "ni", "bhyam", "bhih", "sy", "yo", "nam"]
for v in vibhakti:
    tokens.append({"token": v, "category": "vibhakti", "meaning": "case/number marker"})

# --- Avyayas ---
print("🔍 Adding avyayas...")
avyayas = ["ca", "hi", "va", "na", "nahi", "tu", "api", "kila", "eva", "sma"]
for a in avyayas:
    tokens.append({"token": a, "category": "avyaya", "meaning": "indeclinable"})

# --- Sandhi Marker ---
tokens.append({"token": "<+>", "category": "sandhi_marker", "meaning": "sandhi join"})

# --- Save JSON ---
out = "/home/kush/cooking/ml/sans_tokenization/token/sanskrit_tokens.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(tokens, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(tokens)} tokens to {out}")
