"""
FAQ BEST - Recherche sémantique dans les vidéos de formation
"""

import streamlit as st
import json
import pickle
import numpy as np
import subprocess
import threading
import re
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import os

FAQ_DIR = Path(__file__).parent

# URL de l'API Ollama (configurable via variable d'environnement)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

st.set_page_config(page_title="FAQ BESTT", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .timestamp-badge { background:#1976D2; color:white; padding:3px 8px; border-radius:4px; }
    .result-card { border:1px solid #ddd; border-radius:10px; padding:15px; margin:10px 0; }
    .highlight-bar { height:8px; background:#2196F3; position:absolute; opacity:0.7; }
    .processing { padding:20px; background:#fff3cd; border-radius:10px; margin:20px 0; }
</style>
""", unsafe_allow_html=True)


def extract_video_id(url: str) -> str:
    patterns = [r'(?:v=|/)([0-9A-Za-z_-]{11}).*', r'(?:youtu\.be/)([0-9A-Za-z_-]{11})']
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return url


def get_status_file():
    return FAQ_DIR / ".processing_status.json"


def get_log_file():
    return FAQ_DIR / ".processing_log.txt"


def is_processing():
    status_file = get_status_file()
    if status_file.exists():
        try:
            with open(status_file) as f:
                content = f.read().strip()
                if not content:
                    return False
                status = json.loads(content)
            return status.get("processing", False)
        except (json.JSONDecodeError, Exception):
            return False
    return False


def get_processing_status():
    status_file = get_status_file()
    if status_file.exists():
        try:
            with open(status_file) as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, Exception):
            return {}
    return {}


def run_pipeline(url: str, name: str):
    """Lance le pipeline en arrière-plan"""
    status_file = get_status_file()
    log_file = get_log_file()

    def log(msg: str):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    def update_status(step: str, progress: int):
        with open(status_file, 'w') as f:
            json.dump({"processing": True, "step": step, "progress": progress, "name": name}, f)
        log(f"[{progress}%] {step}")

    def finish(success: bool, error: str = None):
        with open(status_file, 'w') as f:
            json.dump({"processing": False, "success": success, "error": error}, f)

    # Clear log
    with open(log_file, 'w') as f:
        f.write("")

    try:
        video_id = extract_video_id(url)
        output_name = name or video_id
        log(f"Video ID: {video_id}")
        log(f"Output: {output_name}")

        # 1. Download
        update_status("Téléchargement audio...", 10)
        audio_path = FAQ_DIR / f"{output_name}.mp3"
        if not audio_path.exists():
            log("yt-dlp en cours...")
            proc = subprocess.Popen([
                "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0",
                "-o", str(audio_path), url
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log(line.strip())
            proc.wait()
            if proc.returncode != 0:
                raise Exception("yt-dlp failed")
        else:
            log(f"Audio existant: {audio_path}")

        # 2. Whisper
        update_status("Transcription Whisper...", 30)
        json_path = FAQ_DIR / f"{output_name}.json"
        if not json_path.exists():
            log("Whisper large-v3 en cours...")
            proc = subprocess.Popen([
                "whisper", str(audio_path), "--model", "large-v3", "--language", "fr",
                "--output_dir", str(FAQ_DIR), "--output_format", "json"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log(line.strip())
            proc.wait()
            if proc.returncode != 0:
                raise Exception("Whisper failed")
        else:
            log(f"Transcription existante: {json_path}")

        # 3. Embeddings
        update_status("Création embeddings...", 70)
        emb_path = FAQ_DIR / f"{output_name}_embeddings.pkl"
        if not emb_path.exists():
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get('segments', [])
            embeddings = model.encode([s['text'] for s in segments], show_progress_bar=False)
            with open(emb_path, 'wb') as f:
                pickle.dump((embeddings, segments), f)

        # 4. Chapters - détection automatique via Llama
        update_status("Détection des chapitres (Llama)...", 85)
        chap_path = FAQ_DIR / f"{output_name}_chapters.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = data.get('segments', [])

        log("Analyse du contenu pour détecter les sections...")
        chapters = detect_chapters_llama(segments, log_func=log)

        if not chapters:
            log("Fallback: chapitres basés sur la durée")
            # Fallback si Llama échoue
            total_duration = segments[-1]['end'] if segments else 0
            num_chapters = min(6, max(2, int(total_duration / 300)))
            chapter_duration = total_duration / num_chapters if num_chapters > 0 else total_duration
            chapters = []
            for i in range(num_chapters):
                chapters.append({
                    "title": f"Partie {i+1}",
                    "start": i * chapter_duration,
                    "end": (i + 1) * chapter_duration
                })

        with open(chap_path, 'w', encoding='utf-8') as f:
            json.dump(chapters, f, ensure_ascii=False, indent=2)
        log(f"Chapitres détectés: {len(chapters)}")

        # 5. Pré-génération des résumés via Llama
        update_status("Génération des résumés (Llama 3.2)...", 92)
        summaries = {}
        for i, ch in enumerate(chapters):
            log(f"Résumé {i+1}/{len(chapters)}: {ch['title']}")
            chapter_segments = [s for s in segments if ch['start'] <= s['start'] < ch['end']]
            if chapter_segments:
                full_text = ' '.join(seg['text'] for seg in chapter_segments)
                summary = generate_summary_llama(full_text, ch['title'])
                if summary:
                    ch['summary'] = summary
                else:
                    ch['summary'] = full_text[:300] + '...'

        # Sauvegarder chapitres avec résumés
        with open(chap_path, 'w', encoding='utf-8') as f:
            json.dump(chapters, f, ensure_ascii=False, indent=2)

        # 5b. Extraction mots-clés métier via Llama
        update_status("Extraction mots-clés (Llama)...", 95)
        keywords_path = FAQ_DIR / f"{output_name}_keywords.json"
        keywords_list = []
        for i, seg in enumerate(segments):
            if i % 50 == 0:
                log(f"Keywords {i}/{len(segments)}")
            text = seg['text'].strip()
            if len(text) < 10:
                keywords = []
            else:
                keywords = extract_keywords_for_segment(text)
            keywords_list.append({
                "idx": i,
                "start": seg['start'],
                "keywords": keywords
            })
        with open(keywords_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_list, f, ensure_ascii=False, indent=2)
        log(f"Keywords extraits: {sum(len(k['keywords']) for k in keywords_list)} mots-clés")

        # 6. Config
        update_status("Finalisation...", 98)
        config_path = FAQ_DIR / "config.json"
        config = {"videos": []} if not config_path.exists() else json.load(open(config_path))

        config["videos"] = [v for v in config.get("videos", []) if v["video_id"] != video_id]
        config["videos"].append({
            "video_id": video_id, "name": output_name,
            "json": f"{output_name}.json",
            "embeddings": f"{output_name}_embeddings.pkl",
            "chapters": f"{output_name}_chapters.json"
        })
        config["current"] = video_id

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        finish(True)

    except Exception as e:
        finish(False, str(e))


@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def load_config():
    config_path = FAQ_DIR / "config.json"
    return json.load(open(config_path)) if config_path.exists() else None


def save_config(config):
    config_path = FAQ_DIR / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_video_data(vc):
    with open(FAQ_DIR / vc["json"], 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    with open(FAQ_DIR / vc["embeddings"], 'rb') as f:
        embeddings, segments = pickle.load(f)
    with open(FAQ_DIR / vc["chapters"], 'r', encoding='utf-8') as f:
        chapters = json.load(f)

    # Charger les keywords si disponibles
    keywords_path = FAQ_DIR / f"{vc['name']}_keywords.json"
    keywords = None
    if keywords_path.exists():
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords = json.load(f)

    return transcript, embeddings, segments, chapters, keywords


def format_time(s):
    return f"{int(s//60):02d}:{int(s%60):02d}"


def parse_time(time_str: str) -> int:
    """Parse mm:ss format to seconds, returns 0 if invalid"""
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        return int(time_str)
    except (ValueError, IndexError):
        return 0


def get_best_llama_model() -> str:
    """Détecte la VRAM GPU et retourne le meilleur modèle Llama disponible"""
    # Modèles par ordre de préférence (du meilleur au plus léger)
    # Format: (nom, VRAM requise en MB, quantification)
    MODEL_TIERS = [
        ("llama3.1:70b-instruct-q8_0", 75000, "Q8"),       # 70B Q8 - Excellente qualité
        ("llama3.1:70b-instruct-q4_K_M", 42000, "Q4"),     # 70B Q4 - Très bonne qualité
        ("llama3.1:8b-instruct-fp16", 16000, "FP16"),      # 8B FP16 - Bonne qualité
        ("llama3.1:8b-instruct-q8_0", 9000, "Q8"),         # 8B Q8
        ("llama3.2:latest", 2500, "Q4_K_M"),               # 3B - Bon rapport qualité/taille
        ("llama3.2:1b", 1500, "Q4_K_M"),                   # 1B - Minimum
    ]

    # Détecter VRAM disponible
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        vram_free = int(result.stdout.strip().split('\n')[0])
    except:
        vram_free = 4000  # Fallback si pas de GPU

    # Vérifier modèles installés
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        installed = result.stdout.lower()
    except:
        installed = ""

    # Sélectionner le meilleur modèle disponible
    for model, vram_needed, quant in MODEL_TIERS:
        if vram_free >= vram_needed:
            model_short = model.split(":")[0]
            if model_short in installed or model in installed:
                return model
            # Modèle non installé mais assez de VRAM - on peut l'installer
            if vram_free >= vram_needed + 1000:  # Marge de sécurité
                return model

    return "llama3.2:latest"  # Fallback


def ensure_model_installed(model: str) -> bool:
    """Vérifie et installe le modèle si nécessaire"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model.lower() in result.stdout.lower():
            return True
        # Installer le modèle
        subprocess.run(["ollama", "pull", model], timeout=600)
        return True
    except:
        return False


# Cache du modèle sélectionné
_SELECTED_MODEL = None

def get_llama_model():
    global _SELECTED_MODEL
    if _SELECTED_MODEL is None:
        _SELECTED_MODEL = get_best_llama_model()
        ensure_model_installed(_SELECTED_MODEL)
    return _SELECTED_MODEL


def extract_keywords_for_segment(text: str) -> list:
    """Extrait les mots-clés métier d'un segment via Llama"""
    model = get_llama_model()
    prompt = f"""Extrais les mots-clés métier de ce texte sur le logiciel BESTT (gestion intérim).

EXEMPLES: contrat, facture, client, intérimaire, mission, heure, horaire, paie, salaire, relevé, DPAE, agence, entreprise, salarié, candidat, document, signature, mail, planning, mutuelle

IGNORE: "tout à l'heure", "en temps et en heure", bonjour, voilà, donc, pareil

Texte: "{text}"

Mots-clés (max 5, séparés par virgules, ou AUCUN):"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 60}
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            if "aucun" in result.lower() or not result:
                return []
            keywords = [kw.strip().lower() for kw in result.split(",")
                       if kw.strip() and 2 < len(kw.strip()) < 30]
            return keywords[:5]
    except:
        pass
    return []


def detect_chapters_llama(segments: list, log_func=None) -> list:
    """
    Détecte automatiquement les chapitres/sections via approche HYBRIDE:
    1. Détection lexicale des termes métier spécifiques (relevés d'heures, DPAE, etc.)
    2. Llama pour le contexte global des grandes sections
    """
    if not segments:
        return []

    def log(msg):
        if log_func:
            log_func(msg)

    # === ÉTAPE 1: Détection lexicale des fonctionnalités clés ===
    # Termes métier BESTT à détecter (regex insensible à la casse)
    feature_patterns = {
        'CONTRATS > Relevés d\'heures': r'relev[ée]s?\s+d\'?\s*heures?',
        'CONTRATS > DPAE': r'\bDPAE\b',
        'CONTRATS > Création contrat': r'(cr[ée]er?|nouveau|faire)\s+(un\s+)?contrat',
        'ADMINISTRATION > DSN': r'\bDSN\b',
        'SALARIES > Fiche salarié': r'fiche\s+(de\s+l\'?\s*)?int[ée]rimaire|fiche\s+salari[ée]',
        'ENTREPRISES > Fiche entreprise': r'fiche\s+(de\s+l\'?\s*)?(entreprise|client)',
        'FACTURATION > Factures': r'(faire|[ée]diter|envoyer)\s+(les?\s+)?factures?',
        'PAIE > Bulletins': r'(faire|[ée]diter)\s+(les?\s+)?payes?|bulletins?\s+de\s+paie',
        'MUTUELLE > Prévoyance': r'mutuelle|pr[ée]voyance|int[ée]rimaire\s+sant[ée]',
    }

    # Créer des fenêtres de 30 secondes pour une granularité fine
    window_size = 30
    total_duration = segments[-1]['end']

    detected_features = []  # Liste de {start, end, feature}

    for start in range(0, int(total_duration), window_size):
        end = min(start + window_size, total_duration)
        window_segments = [s for s in segments if s['start'] >= start and s['start'] < end]
        if not window_segments:
            continue

        text = ' '.join(s['text'] for s in window_segments).lower()

        # Chercher les patterns
        for feature_name, pattern in feature_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_features.append({
                    'start': start,
                    'end': end,
                    'feature': feature_name,
                    'confidence': 'lexical'
                })
                log(f"[Lexical] {start/60:.1f}min: {feature_name}")
                break  # Une seule feature par fenêtre

    # === ÉTAPE 2: Llama pour les grandes sections (fenêtres de 2min) ===
    model = get_llama_model()
    window_size_llama = 120

    llama_sections = []
    for start in range(0, int(total_duration), window_size_llama):
        end = min(start + window_size_llama, total_duration)
        window_segments = [s for s in segments if s['start'] >= start and s['start'] < end]
        if not window_segments:
            continue

        text = ' '.join(s['text'] for s in window_segments)[:2000]

        log(f"[Llama] Analyse {start/60:.0f}min-{end/60:.0f}min")

        prompt = f"""Analyse ce passage de formation sur le logiciel BESTT (gestion intérim).

Texte: "{text}"

Identifie le MENU PRINCIPAL du logiciel concerné parmi:
- ACCUEIL (présentation générale, navigation)
- ENTREPRISES (fiches clients, contacts, organigramme)
- SALARIES (intérimaires, candidats, CV, documents)
- CONTRATS (création, DPAE, relevés d'heures)
- FACTURATION (édition, envoi, suivi factures)
- PAIE (bulletins, calcul, prélèvement source)
- ADMINISTRATION (DSN, exports, statistiques)

Réponds avec UN SEUL nom de menu (ex: "CONTRATS" ou "FACTURATION"). Pas d'explication."""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 30}
                },
                timeout=30
            )
            if response.status_code == 200:
                raw_response = response.json().get("response", "").strip()
                # Nettoyer la réponse (enlever guillemets, ponctuation, etc.)
                topic = raw_response.upper()
                topic = re.sub(r'["\'\-\.\,\:\;]', '', topic)  # Enlever ponctuation
                topic = topic.split('\n')[0].strip()  # Première ligne
                # Chercher un menu valide dans la réponse
                valid_menus = ['ACCUEIL', 'ENTREPRISES', 'SALARIES', 'CONTRATS', 'FACTURATION', 'PAIE', 'ADMINISTRATION']
                found_menu = None
                for menu in valid_menus:
                    if menu in topic:
                        found_menu = menu
                        break
                if found_menu:
                    llama_sections.append({
                        'start': start,
                        'end': end,
                        'menu': found_menu
                    })
                    log(f"  → {found_menu}")
        except Exception as e:
            log(f"  → Erreur: {e}")

    # === ÉTAPE 3: Fusion intelligente ===
    # Stratégie: Combiner Llama (couverture) + Lexical (précision) sans perdre les trous

    # Fusionner les sections Llama consécutives avec le même menu
    llama_merged = []
    if llama_sections:
        current = llama_sections[0].copy()
        for sec in llama_sections[1:]:
            if sec['menu'] == current['menu'] and sec['start'] <= current['end'] + 120:
                current['end'] = sec['end']
            else:
                llama_merged.append(current)
                current = sec.copy()
        llama_merged.append(current)

    # Fusionner les lexicaux consécutifs avec le même titre
    merged_lexical = []
    if detected_features:
        sorted_lex = sorted(detected_features, key=lambda x: x['start'])
        current = sorted_lex[0].copy()
        for feat in sorted_lex[1:]:
            if feat['feature'] == current['feature'] and feat['start'] <= current['end'] + 60:
                current['end'] = feat['end']
            else:
                merged_lexical.append(current)
                current = feat.copy()
        merged_lexical.append(current)

    # Créer la timeline complète en découpant les chapitres Llama là où il y a des lexicaux
    final_chapters = []

    for llama_ch in llama_merged:
        llama_start = llama_ch['start']
        llama_end = llama_ch['end']
        llama_title = llama_ch['menu']

        # Trouver les lexicaux qui chevauchent ce chapitre Llama
        overlapping_lex = []
        for lex in merged_lexical:
            lex_start = lex['start']
            lex_end = lex['end']
            if not (lex_end <= llama_start or lex_start >= llama_end):
                overlapping_lex.append(lex)

        if not overlapping_lex:
            # Pas de chevauchement, garder le chapitre Llama tel quel
            final_chapters.append({
                'title': llama_title,
                'start': llama_start,
                'end': llama_end,
                'source': 'llama'
            })
        else:
            # Découper le chapitre Llama autour des lexicaux
            overlapping_lex.sort(key=lambda x: x['start'])
            cursor = llama_start

            for lex in overlapping_lex:
                lex_start = max(lex['start'], llama_start)
                lex_end = min(lex['end'], llama_end)

                # Partie Llama avant le lexical
                if cursor < lex_start:
                    final_chapters.append({
                        'title': llama_title,
                        'start': cursor,
                        'end': lex_start,
                        'source': 'llama'
                    })

                # Le chapitre lexical
                final_chapters.append({
                    'title': lex['feature'],
                    'start': lex_start,
                    'end': lex_end,
                    'source': 'lexical'
                })
                cursor = lex_end

            # Partie Llama après le dernier lexical
            if cursor < llama_end:
                final_chapters.append({
                    'title': llama_title,
                    'start': cursor,
                    'end': llama_end,
                    'source': 'llama'
                })

    # Ajouter les lexicaux qui ne sont dans aucun chapitre Llama
    for lex in merged_lexical:
        in_llama = any(
            llama_ch['start'] <= lex['start'] < llama_ch['end']
            for llama_ch in llama_merged
        )
        if not in_llama:
            final_chapters.append({
                'title': lex['feature'],
                'start': lex['start'],
                'end': lex['end'],
                'source': 'lexical'
            })

    # Trier par timestamp
    final_chapters.sort(key=lambda x: x['start'])

    # Fusionner les chapitres consécutifs avec le même titre
    if not final_chapters:
        return []

    merged = [final_chapters[0].copy()]
    for ch in final_chapters[1:]:
        if ch['title'] == merged[-1]['title'] and ch['start'] <= merged[-1]['end'] + 30:
            merged[-1]['end'] = ch['end']
        else:
            merged.append(ch.copy())

    # Formater le résultat final
    result = []
    for ch in merged:
        result.append({
            'title': ch['title'],
            'start': ch['start'],
            'end': ch['end']
        })

    lexical_count = sum(1 for c in merged if c.get('source') == 'lexical')
    llama_count = sum(1 for c in merged if c.get('source') == 'llama')
    log(f"Détecté {len(result)} chapitres ({lexical_count} lexical, {llama_count} Llama)")
    return result


def generate_summary_llama(text: str, title: str) -> str:
    """Génère un résumé via le meilleur modèle Llama disponible"""
    model = get_llama_model()

    prompt = f"""Tu es un assistant qui résume des transcriptions de vidéos de formation sur le logiciel BEST (gestion d'intérim).

Résume en 2-3 phrases ce passage de la section "{title}". Sois concis et informatif.

Transcription:
{text[:3000]}

Résumé:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        pass
    return None


def find_chapter_for_time(chapters, time):
    """Trouve le chapitre correspondant à un timestamp"""
    for ch in chapters:
        if ch['start'] <= time < ch['end']:
            return ch
    return None


def stem_french(word):
    """Stemming français simplifié - normalise singulier/pluriel"""
    w = word.lower()
    # Pluriels en -aux -> -al (travaux -> travail)
    if w.endswith('aux') and len(w) > 4:
        return w[:-3] + 'al'
    # Pluriels en -eaux -> -eau
    if w.endswith('eaux') and len(w) > 5:
        return w[:-1]
    # Pluriels simples -s, -x
    if w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
        return w[:-1]
    if w.endswith('x') and len(w) > 3:
        return w[:-1]
    return w


def tokenize(text):
    """Tokenize text for BM25 avec stemming français"""
    words = re.findall(r'\w+', text.lower())
    return [stem_french(w) for w in words]


@st.cache_resource
def build_bm25_index(segment_texts, keywords_data=None):
    """Build BM25 index from segment texts or keywords (cached)

    Si keywords_data est fourni, utilise les mots-clés métier extraits par Llama.
    Sinon, utilise le texte brut.
    """
    if keywords_data:
        # Utiliser keywords + texte pour meilleure couverture
        corpus = []
        for i, text in enumerate(segment_texts):
            tokens = tokenize(text)
            if i < len(keywords_data) and keywords_data[i].get('keywords'):
                # Ajouter les keywords (avec boost en les répétant)
                kw_tokens = keywords_data[i]['keywords']
                tokens = kw_tokens * 3 + tokens  # Keywords 3x plus importants
            corpus.append(tokens)
        return BM25Okapi(corpus)
    else:
        corpus = [tokenize(text) for text in segment_texts]
        return BM25Okapi(corpus)


# Expressions idiomatiques à ignorer (faux positifs fréquents)
IDIOM_PATTERNS = [
    r"tout à l'heure",
    r"à l'heure actuelle",
    r"en temps et en heure",
    r"de bonne heure",
    r"à la bonne heure",
    r"sur l'heure",
    r"d'heure en heure",
]


def contains_idiom(text, query_words):
    """Vérifie si le texte contient une expression idiomatique pour les mots recherchés"""
    text_lower = text.lower()
    # Ne filtrer que si la query contient des mots sensibles aux idiomes
    idiom_sensitive = ['heure', 'heures', 'temps']
    if not any(w in query_words for w in idiom_sensitive):
        return False

    for pattern in IDIOM_PATTERNS:
        if pattern in text_lower:
            # Vérifier si le mot apparaît AUSSI en dehors de l'idiome
            # Retirer l'idiome et vérifier si "heure" reste
            cleaned = re.sub(pattern, '', text_lower)
            if not any(w in cleaned for w in idiom_sensitive if w in query_words):
                return True  # Seulement l'idiome, pas de vrai match
    return False


def search(query, embeddings, segments, model, chapters, bm25=None, top_k=30, alpha=0.5):
    """Recherche hybride (sémantique + BM25) avec fusion par chapitre

    Args:
        alpha: poids de la recherche sémantique (1-alpha = poids BM25)
        top_k: nombre de segments candidats à récupérer
    """
    # Déterminer la plage valide des chapitres (pas avant le premier, pas après le dernier)
    if chapters:
        chapters_start = min(ch['start'] for ch in chapters)
        chapters_end = max(ch['end'] for ch in chapters)
    else:
        chapters_start = 0
        chapters_end = float('inf')

    # Déterminer si requête courte (1-2 mots) = exiger match lexical
    query_tokens = tokenize(query)
    is_keyword_query = len(query_tokens) <= 2

    # 1. Recherche sémantique
    q = model.encode([query])
    q_norm = q / np.linalg.norm(q)
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sem_scores = np.dot(emb_norm, q_norm.T).flatten()

    # 2. Recherche BM25 (lexicale)
    bm25_scores_raw = None
    if bm25 is not None:
        bm25_scores_raw = bm25.get_scores(query_tokens)
        bm25_scores = bm25_scores_raw.copy()
        # Normaliser BM25 scores [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        # Combiner les scores
        scores = alpha * sem_scores + (1 - alpha) * bm25_scores
    else:
        scores = sem_scores

    idx = np.argsort(scores)[::-1][:top_k]

    # Mots de la query pour filtrage idiomes
    query_words = set(query_tokens)

    # Collecter les résultats bruts avec leur chapitre
    raw_results = []
    for i in idx:
        if scores[i] > 0.3:
            seg_start = segments[i]["start"]
            seg_end = segments[i]["end"]

            # Filtrer les résultats hors de la plage des chapitres
            if seg_start < chapters_start or seg_end > chapters_end:
                continue

            # Pour requêtes courtes: exiger un match BM25 (mot présent dans texte/keywords)
            if is_keyword_query and bm25_scores_raw is not None and bm25_scores_raw[i] == 0:
                continue
            text = segments[i]["text"]
            # Filtrer les expressions idiomatiques
            if contains_idiom(text, query_words):
                continue
            chapter = find_chapter_for_time(chapters, seg_start)
            raw_results.append({
                "text": text,
                "start": seg_start,
                "end": seg_end,
                "score": float(scores[i]),
                "chapter": chapter,
            })

    if not raw_results:
        return []

    # Fusionner par chapitre
    chapter_results = {}
    for r in raw_results:
        ch_key = r["chapter"]["title"] if r["chapter"] else "Inconnu"
        if ch_key not in chapter_results:
            chapter_results[ch_key] = {
                "chapter": r["chapter"],
                "segments": [],
                "best_score": 0,
            }
        chapter_results[ch_key]["segments"].append(r)
        chapter_results[ch_key]["best_score"] = max(
            chapter_results[ch_key]["best_score"], r["score"]
        )

    # Construire les résultats fusionnés par chapitre
    merged = []
    for ch_key, data in chapter_results.items():
        # Trouver le segment avec le meilleur score (le plus pertinent)
        best_seg = max(data["segments"], key=lambda x: x["score"])

        # Trier les segments par temps pour l'affichage du texte
        data["segments"].sort(key=lambda x: x["start"])

        # Construire le texte avec les extraits pertinents (triés par score)
        top_segments = sorted(data["segments"], key=lambda x: -x["score"])[:3]
        texts = [s["text"] for s in top_segments]
        text = " [...] ".join(texts)
        if len(data["segments"]) > 3:
            text += f" [...] (+{len(data['segments'])-3} autres)"

        merged.append({
            "chapter": data["chapter"],
            "start": best_seg["start"],  # Position au segment le plus pertinent
            "end": data["segments"][-1]["end"],
            "score": data["best_score"],
            "text": text,
            "segment_count": len(data["segments"]),
        })

    # Trier par score (meilleur score en premier)
    merged.sort(key=lambda x: -x["score"])

    return merged[:10]  # Top 10 résultats par score


def main():
    st.title("FAQ BESTT")

    # Sidebar: Ajouter vidéo
    with st.sidebar:
        st.header("Ajouter une vidéo")
        url = st.text_input("URL YouTube", placeholder="https://youtube.com/watch?v=...")
        name = st.text_input("Nom (optionnel)", placeholder="BEST_demo")

        if st.button("Traiter", type="primary", disabled=is_processing()):
            if url:
                thread = threading.Thread(target=run_pipeline, args=(url, name or None))
                thread.start()
                st.rerun()
            else:
                st.error("URL requise")

        # Status
        if is_processing():
            status = get_processing_status()
            progress = status.get('progress', 0)
            step = status.get('step', 'Initialisation...')

            st.progress(progress / 100)
            st.info(f"**{step}**")

            # Dernière ligne de log
            log_file = get_log_file()
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    last = lines[-1]
                    # Nettoyer timestamp whisper
                    if '-->' in last:
                        last = last.split(']')[-1].strip() if ']' in last else last
                    st.caption(last[:100])

            if st.button("🔄 Actualiser", use_container_width=True):
                st.rerun()

        st.divider()
        st.header("Vidéos disponibles")

    config = load_config()

    if not config or not config.get("videos"):
        st.info("Ajoutez une vidéo YouTube via la sidebar pour commencer.")
        return

    # Sélection vidéo
    videos = config["videos"]

    with st.sidebar:
        st.subheader(f"{len(videos)} vidéo(s)")

    if len(videos) > 1:
        vc = st.selectbox("Sélectionner une vidéo", videos, format_func=lambda v: v["name"])
    else:
        vc = videos[0]
        st.info(f"Vidéo : **{vc['name']}**")

    model = load_model()

    try:
        _, embeddings_all, segments_all, chapters_all, keywords_all = load_video_data(vc)
    except FileNotFoundError as e:
        st.error(f"Fichier manquant: {e}")
        return

    total = segments_all[-1]["end"] if segments_all else 3600

    # Calculer les temps de coupe basés sur les chapitres détectés
    if chapters_all:
        chapters_start = min(ch['start'] for ch in chapters_all)
        chapters_end = max(ch['end'] for ch in chapters_all)
    else:
        chapters_start = 0
        chapters_end = total

    # Temps de coupe (intro/conclusion)
    with st.sidebar:
        st.divider()
        st.subheader("Temps de coupe")
        st.caption("Basé sur les chapitres détectés")

        # Valeurs par défaut = plage des chapitres
        default_start = format_time(chapters_start)
        default_end = format_time(chapters_end)

        col_s, col_e = st.columns(2)
        with col_s:
            cut_start_str = st.text_input("Début", value=default_start, key="cut_start_input", help="mm:ss")
        with col_e:
            cut_end_str = st.text_input("Fin", value=default_end, key="cut_end_input", help="mm:ss")

        cut_start = parse_time(cut_start_str)
        cut_end = parse_time(cut_end_str)
        if cut_end == 0:
            cut_end = chapters_end

        # Sauvegarder si changement
        if cut_start != vc.get("cut_start", 0) or cut_end != vc.get("cut_end", total):
            vc["cut_start"] = cut_start
            vc["cut_end"] = cut_end
            save_config(config)
            st.success(f"Coupe: {format_time(cut_start)} - {format_time(cut_end)}")

    # Filtrer segments, embeddings et chapitres par temps de coupe
    indices = [i for i, s in enumerate(segments_all) if s["start"] >= cut_start and s["end"] <= cut_end]
    segments = [segments_all[i] for i in indices]
    embeddings = embeddings_all[indices] if len(indices) > 0 else embeddings_all
    # Chapitres: inclure ceux qui commencent dans la plage
    chapters = [c for c in chapters_all if c["start"] >= cut_start and c["start"] < cut_end]
    keywords = [keywords_all[i] for i in indices] if keywords_all and len(indices) > 0 else keywords_all

    # Build BM25 index après filtrage
    bm25 = build_bm25_index(tuple(s['text'] for s in segments), keywords)
    if keywords_all:
        st.sidebar.success(f"Keywords Llama chargés ({len(segments)} segments)")

    # Recalculer total après filtrage
    total_filtered = segments[-1]["end"] - cut_start if segments else total

    if "time" not in st.session_state:
        st.session_state.time = 0
    if "hl" not in st.session_state:
        st.session_state.hl = []
    if "selected_chapter" not in st.session_state:
        st.session_state.selected_chapter = None
    if "query" not in st.session_state:
        st.session_state.query = ""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.components.v1.iframe(
            f"https://www.youtube.com/embed/{vc['video_id']}?start={st.session_state.time}&autoplay=1",
            height=400
        )

        # Timeline
        html = '<div style="position:relative;height:12px;background:#e0e0e0;border-radius:6px;margin:10px 0;">'
        for s, e in st.session_state.hl:
            html += f'<div class="highlight-bar" style="left:{s/total*100}%;width:{(e-s)/total*100}%"></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        # Barre de recherche sous la vidéo
        query = st.text_input("Rechercher", placeholder="Comment rentrer les heures ?", key="query")

        if query:
            results = search(query, embeddings, segments, model, chapters, bm25=bm25)
            st.session_state.hl = [(r["start"], r["end"]) for r in results]

            if results:
                total_segments = sum(r.get("segment_count", 1) for r in results)
                st.markdown(f"**{total_segments} passage(s) dans {len(results)} chapitre(s)**")
                for i, r in enumerate(results):
                    chapter_name = r["chapter"]["title"] if r["chapter"] else "Inconnu"
                    score_pct = min(99, int(r["score"] * 100))  # Cap à 99%
                    seg_count = r.get("segment_count", 1)
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f'''<div class="result-card">
                            <div style="margin-bottom:5px;">
                                <span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">📂 {chapter_name}</span>
                                <span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-left:5px;">{score_pct}%</span>
                                <span style="background:#9E9E9E;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-left:5px;">{seg_count} passage(s)</span>
                            </div>
                            <span class="timestamp-badge">{format_time(r["start"])}</span> {r["text"]}
                        </div>''', unsafe_allow_html=True)
                    with c2:
                        if st.button("▶", key=f"p{i}"):
                            # Démarrer au début du chapitre, pas au segment trouvé
                            if r["chapter"]:
                                st.session_state.time = int(r["chapter"]["start"])
                            else:
                                st.session_state.time = int(r["start"])
                            st.session_state.selected_chapter = r["chapter"]
                            st.rerun()

    with col2:
        st.subheader("Chapitres")
        with st.container(height=400):
            for ch in chapters:
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f'''<div style="padding:3px 0;">
                        <span class="timestamp-badge">{format_time(ch['start'])}</span>
                        <span style="margin-left:5px;font-size:0.9em;">{ch['title']}</span>
                    </div>''', unsafe_allow_html=True)
                with c2:
                    if st.button("▶", key=f"c{ch['start']}"):
                        st.session_state.time = int(ch['start'])
                        st.session_state.selected_chapter = ch
                        st.rerun()

    # Afficher résumé et transcription en pleine largeur
    if st.session_state.selected_chapter:
        ch = st.session_state.selected_chapter
        st.divider()
        st.subheader(f"📖 {ch['title']}")

        # Récupérer les segments de ce chapitre
        chapter_segments = [s for s in segments if ch['start'] <= s['start'] < ch['end']]

        if chapter_segments:
            full_text = ' '.join(s['text'] for s in chapter_segments)

            # Résumé pré-calculé ou fallback
            summary = ch.get('summary')
            if summary:
                st.success(f"**Résumé:** {summary}")
            else:
                st.info(f"**Extrait:** {full_text[:300]}...")

            # Transcription complète
            st.markdown("**Transcription:**")
            with st.container(height=400):
                for seg in chapter_segments:
                    st.markdown(f"**{format_time(seg['start'])}** {seg['text']}")


if __name__ == "__main__":
    main()
