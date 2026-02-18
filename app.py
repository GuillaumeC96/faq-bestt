"""
FAQ BESTT - Version Cloud (lecture seule)
Recherche sémantique dans les vidéos de formation BESTT
"""

import streamlit as st
import json
import pickle
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

FAQ_DIR = Path(__file__).parent

st.set_page_config(page_title="FAQ BESTT", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .timestamp-badge { background:#1976D2; color:white; padding:3px 8px; border-radius:4px; }
    .result-card { border:1px solid #ddd; border-radius:10px; padding:15px; margin:10px 0; }
    .highlight-bar { height:8px; background:#2196F3; position:absolute; opacity:0.7; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def load_config():
    config_path = FAQ_DIR / "config.json"
    return json.load(open(config_path)) if config_path.exists() else None


def load_video_data(vc):
    with open(FAQ_DIR / vc["json"], 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    with open(FAQ_DIR / vc["embeddings"], 'rb') as f:
        embeddings, segments = pickle.load(f)
    with open(FAQ_DIR / vc["chapters"], 'r', encoding='utf-8') as f:
        chapters = json.load(f)

    keywords_path = FAQ_DIR / f"{vc['name']}_keywords.json"
    keywords = None
    if keywords_path.exists():
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords = json.load(f)

    return transcript, embeddings, segments, chapters, keywords


def format_time(s):
    return f"{int(s//60):02d}:{int(s%60):02d}"


def find_chapter_for_time(chapters, time):
    for ch in chapters:
        if ch['start'] <= time < ch['end']:
            return ch
    return None


def stem_french(word):
    w = word.lower()
    if w.endswith('aux') and len(w) > 4:
        return w[:-3] + 'al'
    if w.endswith('eaux') and len(w) > 5:
        return w[:-1]
    if w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
        return w[:-1]
    if w.endswith('x') and len(w) > 3:
        return w[:-1]
    return w


def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return [stem_french(w) for w in words]


@st.cache_resource
def build_bm25_index(segment_texts, keywords_data=None):
    if keywords_data:
        corpus = []
        for i, text in enumerate(segment_texts):
            tokens = tokenize(text)
            if i < len(keywords_data) and keywords_data[i].get('keywords'):
                kw_tokens = keywords_data[i]['keywords']
                tokens = kw_tokens * 3 + tokens
            corpus.append(tokens)
        return BM25Okapi(corpus)
    else:
        corpus = [tokenize(text) for text in segment_texts]
        return BM25Okapi(corpus)


IDIOM_PATTERNS = [
    r"tout à l'heure", r"à l'heure actuelle", r"en temps et en heure",
    r"de bonne heure", r"à la bonne heure", r"sur l'heure", r"d'heure en heure",
]


def contains_idiom(text, query_words):
    text_lower = text.lower()
    idiom_sensitive = ['heure', 'heures', 'temps']
    if not any(w in query_words for w in idiom_sensitive):
        return False
    for pattern in IDIOM_PATTERNS:
        if pattern in text_lower:
            cleaned = re.sub(pattern, '', text_lower)
            if not any(w in cleaned for w in idiom_sensitive if w in query_words):
                return True
    return False


def search(query, embeddings, segments, model, chapters, bm25=None, top_k=30, alpha=0.5):
    if chapters:
        chapters_start = min(ch['start'] for ch in chapters)
        chapters_end = max(ch['end'] for ch in chapters)
    else:
        chapters_start = 0
        chapters_end = float('inf')

    query_tokens = tokenize(query)
    is_keyword_query = len(query_tokens) <= 2

    q = model.encode([query])
    q_norm = q / np.linalg.norm(q)
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sem_scores = np.dot(emb_norm, q_norm.T).flatten()

    bm25_scores_raw = None
    if bm25 is not None:
        bm25_scores_raw = bm25.get_scores(query_tokens)
        bm25_scores = bm25_scores_raw.copy()
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        scores = alpha * sem_scores + (1 - alpha) * bm25_scores
    else:
        scores = sem_scores

    idx = np.argsort(scores)[::-1][:top_k]
    query_words = set(query_tokens)

    raw_results = []
    for i in idx:
        if scores[i] > 0.3:
            seg_start = segments[i]["start"]
            seg_end = segments[i]["end"]

            if seg_start < chapters_start or seg_end > chapters_end:
                continue

            if is_keyword_query and bm25_scores_raw is not None and bm25_scores_raw[i] == 0:
                continue
            text = segments[i]["text"]
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

    merged = []
    for ch_key, data in chapter_results.items():
        best_seg = max(data["segments"], key=lambda x: x["score"])
        data["segments"].sort(key=lambda x: x["start"])
        top_segments = sorted(data["segments"], key=lambda x: -x["score"])[:3]
        texts = [s["text"] for s in top_segments]
        text = " [...] ".join(texts)
        if len(data["segments"]) > 3:
            text += f" [...] (+{len(data['segments'])-3} autres)"

        merged.append({
            "chapter": data["chapter"],
            "start": best_seg["start"],
            "end": data["segments"][-1]["end"],
            "score": data["best_score"],
            "text": text,
            "segment_count": len(data["segments"]),
        })

    merged.sort(key=lambda x: -x["score"])
    return merged[:10]


def main():
    st.title("FAQ BESTT")

    config = load_config()

    if not config or not config.get("videos"):
        st.error("Aucune vidéo configurée.")
        return

    videos = config["videos"]

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

    # Plage des chapitres
    if chapters_all:
        chapters_start = min(ch['start'] for ch in chapters_all)
        chapters_end = max(ch['end'] for ch in chapters_all)
    else:
        chapters_start = 0
        chapters_end = total

    # Filtrer segments et chapitres
    indices = [i for i, s in enumerate(segments_all) if s["start"] >= chapters_start and s["end"] <= chapters_end]
    segments = [segments_all[i] for i in indices]
    embeddings = embeddings_all[indices] if len(indices) > 0 else embeddings_all
    chapters = [c for c in chapters_all if c["start"] >= chapters_start and c["start"] < chapters_end]
    keywords = [keywords_all[i] for i in indices] if keywords_all and len(indices) > 0 else keywords_all

    bm25 = build_bm25_index(tuple(s['text'] for s in segments), keywords)

    if "time" not in st.session_state:
        st.session_state.time = 0
    if "hl" not in st.session_state:
        st.session_state.hl = []
    if "selected_chapter" not in st.session_state:
        st.session_state.selected_chapter = None

    col1, col2 = st.columns([2, 1])

    with col1:
        st.components.v1.iframe(
            f"https://www.youtube.com/embed/{vc['video_id']}?start={st.session_state.time}&autoplay=1",
            height=400
        )

        html = '<div style="position:relative;height:12px;background:#e0e0e0;border-radius:6px;margin:10px 0;">'
        for s, e in st.session_state.hl:
            html += f'<div class="highlight-bar" style="left:{s/total*100}%;width:{(e-s)/total*100}%"></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        query = st.text_input("Rechercher", placeholder="Comment rentrer les heures ?")

        if query:
            results = search(query, embeddings, segments, model, chapters, bm25=bm25)
            st.session_state.hl = [(r["start"], r["end"]) for r in results]

            if results:
                total_segments = sum(r.get("segment_count", 1) for r in results)
                st.markdown(f"**{total_segments} passage(s) dans {len(results)} chapitre(s)**")
                for i, r in enumerate(results):
                    chapter_name = r["chapter"]["title"] if r["chapter"] else "Inconnu"
                    score_pct = min(99, int(r["score"] * 100))
                    seg_count = r.get("segment_count", 1)
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f'''<div class="result-card">
                            <div style="margin-bottom:5px;">
                                <span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">{chapter_name}</span>
                                <span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-left:5px;">{score_pct}%</span>
                                <span style="background:#9E9E9E;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-left:5px;">{seg_count} passage(s)</span>
                            </div>
                            <span class="timestamp-badge">{format_time(r["start"])}</span> {r["text"]}
                        </div>''', unsafe_allow_html=True)
                    with c2:
                        if st.button("▶", key=f"p{i}"):
                            if r["chapter"]:
                                st.session_state.time = int(r["chapter"]["start"])
                            else:
                                st.session_state.time = int(r["start"])
                            st.session_state.selected_chapter = r["chapter"]
                            st.rerun()

    with col2:
        st.subheader("Chapitres")
        for ch in chapters:
            if st.button(f"{format_time(ch['start'])} {ch['title']}", key=f"c{ch['start']}", use_container_width=True):
                st.session_state.time = int(ch['start'])
                st.session_state.selected_chapter = ch
                st.rerun()

    if st.session_state.selected_chapter:
        ch = st.session_state.selected_chapter
        st.divider()
        st.subheader(f"{ch['title']}")

        chapter_segments = [s for s in segments if ch['start'] <= s['start'] < ch['end']]

        if chapter_segments:
            summary = ch.get('summary')
            if summary:
                st.success(f"**Résumé:** {summary}")

            st.markdown("**Transcription:**")
            with st.container(height=400):
                for seg in chapter_segments:
                    st.markdown(f"**{format_time(seg['start'])}** {seg['text']}")


if __name__ == "__main__":
    main()
