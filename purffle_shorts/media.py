"""Visual sourcing: per-scene stock footage and photos (Pexels, Pixabay), your own media folder,
and AI-generated images (OpenAI gpt-image-1 / DALL·E, Pollinations). Clips are never reused across
videos, portrait footage is preferred, and a scene always gets *something* (animated gradient)."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from .asset_library import get_or_create_asset
from .config import Settings
from .utils import PermanentError, download, http, raise_for_status, redact, with_retries
from .visual_library import visual_type

log = logging.getLogger("purffle")

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STOP_WORDS = {"the", "a", "an", "of", "and", "in", "on", "with", "for", "to", "at", "by", "from", "closeup", "how", "what", "why", "when", "where", "who", "show", "explain", "describe", "tell", "make", "create", "communicate", "communicates", "communicating", "communicated"}


@dataclass
class MediaItem:
    source: str
    id: str
    kind: str                     # video | image | generated
    url: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    credit: str = ""
    path: Path | None = None
    query: str = ""
    extra: dict = field(default_factory=dict)
    rank: int = 0                 # position in the search results (the API's own relevance order)
    text: str = ""                # what the provider says is in it: page slug, alt text, tags

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def portrait(self) -> bool:
        return self.height > self.width


# ------------------------------------------------------------------------------------------ background video
def background_item(settings: Settings, seconds: float) -> MediaItem | None:
    """One continuous stretch of BACKGROUND_VIDEO (a file, or a random video from a folder) long enough for
    the whole Short, starting at a random point so consecutive videos don't all open on the same frames."""
    from . import ffmpeg
    if not settings.background_video:
        return None
    root = Path(settings.background_video).expanduser()
    if root.is_dir():
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in VIDEO_EXT and not f.name.startswith("."))
    else:
        files = [root] if root.is_file() else []
    if not files:
        log.warning("BACKGROUND_VIDEO=%s has no video files; using footage instead", settings.background_video)
        return None
    f = random.choice(files)
    info = ffmpeg.probe(f)
    if not info["has_video"]:
        log.warning("%s has no picture; using footage instead", f)
        return None
    dur = info["duration"]
    start = random.uniform(0.0, dur - seconds - 0.5) if dur > seconds + 1.0 else 0.0
    return MediaItem("background", f.name, "video", width=info["width"], height=info["height"], duration=dur,
                     path=f, query="background", extra={"start": round(start, 2)})


# ------------------------------------------------------------------------------------------ Pexels
def _pexels_pick_file(files: list[dict], prefer_4k: bool) -> dict | None:
    files = [f for f in files if f.get("link") and f.get("width") and f.get("height")
             and f.get("file_type", "video/mp4") == "video/mp4"]
    if not files:
        return None
    cap = 4096 if prefer_4k else 2160

    def short(f):
        return min(f["width"], f["height"])
    good = [f for f in files if 1080 <= short(f) <= cap]
    if good:
        return min(good, key=short) if not prefer_4k else max(good, key=short)
    return max(files, key=short)


def pexels_videos(query: str, key: str, prefer_4k: bool = False, orientation: str = "portrait") -> list[MediaItem]:
    params = {"query": query, "per_page": 15, "size": "medium"}
    if orientation:
        params["orientation"] = orientation
    r = http().get("https://api.pexels.com/videos/search", params=params, headers={"Authorization": key}, timeout=30)
    raise_for_status(r, "Pexels video search")
    items = []
    for rank, v in enumerate(r.json().get("videos", [])):
        f = _pexels_pick_file(v.get("video_files", []), prefer_4k)
        if not f:
            continue
        tags = " ".join(str(t) for t in v.get("tags") or [])
        items.append(MediaItem("pexels", str(v["id"]), "video", f["link"], f["width"], f["height"],
                               float(v.get("duration") or 0), (v.get("user") or {}).get("name", ""), query=query,
                               rank=rank, text=f"{_slug_words(v.get('url', ''))} {tags}".strip()))
    return items


def pexels_photos(query: str, key: str, orientation: str = "portrait") -> list[MediaItem]:
    params = {"query": query, "per_page": 15}
    if orientation:
        params["orientation"] = orientation
    r = http().get("https://api.pexels.com/v1/search", params=params, headers={"Authorization": key}, timeout=30)
    raise_for_status(r, "Pexels photo search")
    items = []
    for rank, p in enumerate(r.json().get("photos", [])):
        src = p.get("src") or {}
        url = src.get("original")
        if not url:
            continue
        url += ("&" if "?" in url else "?") + "auto=compress&cs=tinysrgb&h=2400"
        items.append(MediaItem("pexels-photo", str(p["id"]), "image", url, p.get("width", 0), p.get("height", 0),
                               credit=p.get("photographer", ""), query=query, rank=rank,
                               text=f"{p.get('alt') or ''} {_slug_words(p.get('url', ''))}".strip()))
    return items




# ------------------------------------------------------------------------------------------ Wikimedia Commons
def commons_search(query: str, orientation: str = "portrait") -> list[MediaItem]:
    """Search Commons for media with suitable reuse licences."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 25,
        "prop": "imageinfo|categories",
        "iiprop": "url|size|mime|extmetadata",
        "cllimit": "max",
    }
    r = http().get(
        "https://commons.wikimedia.org/w/api.php",
        params=params,
        timeout=30,
    )
    raise_for_status(r, "Wikimedia Commons search")
    pages = (r.json().get("query") or {}).get("pages") or {}
    items = []

    def metadata_value(metadata, name):
        value = (metadata.get(name) or {}).get("value", "")
        return re.sub(r"<[^>]*>", "", str(value)).strip()

    for rank, page in enumerate(pages.values()):
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = str(info.get("mime") or "").lower()
        is_video = mime in {"video/webm", "video/ogg", "video/mp4"}
        is_image = mime in {"image/jpeg", "image/png"}
        if not (is_video or is_image) or not info.get("url"):
            continue

        metadata = info.get("extmetadata") or {}
        license_name = metadata_value(metadata, "LicenseShortName")
        terms = metadata_value(metadata, "UsageTerms")
        license_text = (license_name + " " + terms).lower()

        # Conservative first pass: exclude non-commercial and no-derivatives files.
        if not license_name or "non-commercial" in license_text:
            continue
        if "no-derivatives" in license_text or "cc by-nd" in license_text or "cc-by-nd" in license_text:
            continue
        if not any(x in license_text for x in (
            "cc0", "public domain", "cc by", "cc-by", "cc by-sa", "cc-by-sa"
        )):
            continue

        title = str(page.get("title") or "")
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if orientation == "portrait" and width and height and width > height * 2:
            continue

        artist = metadata_value(metadata, "Artist")
        description = metadata_value(metadata, "ImageDescription")
        categories = " ".join(
            str(category.get("title", "")).replace("Category:", "")
            for category in page.get("categories", [])
        )
        page_url = "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"))
        filename = title.removeprefix("File:").replace("_", " ")
        searchable_text = " ".join(
            part for part in (filename, description, categories) if part
        )
        items.append(MediaItem(
            "commons", title.removeprefix("File:"),
            "video" if is_video else "image",
            info["url"], width, height,
            float(info.get("duration") or 0),
            credit=artist, query=query, rank=rank,
            text=searchable_text,
            extra={
                "license": license_name,
                "license_url": metadata_value(metadata, "LicenseUrl"),
                "artist": artist,
                "source_page": page_url,
                "mime": mime,
                "description": description,
                "categories": categories,
            },
        ))
    return items

# ------------------------------------------------------------------------------------------ Pixabay
def pixabay_videos(query: str, key: str, prefer_4k: bool = False) -> list[MediaItem]:
    r = http().get("https://pixabay.com/api/videos/", timeout=30, params={
        "key": key, "q": query[:100], "per_page": 20, "safesearch": "true"})
    raise_for_status(r, "Pixabay video search")
    order = ["large", "medium", "small"] if prefer_4k else ["medium", "large", "small"]
    items = []
    for rank, h in enumerate(r.json().get("hits", [])):
        vids = h.get("videos") or {}
        f = next((vids[q] for q in order if (vids.get(q) or {}).get("url")), None)
        if not f:
            continue
        items.append(MediaItem("pixabay", str(h["id"]), "video", f["url"], f.get("width", 0), f.get("height", 0),
                               float(h.get("duration") or 0), h.get("user", ""), query=query, rank=rank,
                               text=str(h.get("tags") or "")))
    return items


def pixabay_photos(query: str, key: str, orientation: str = "portrait") -> list[MediaItem]:
    pix = {"portrait": "vertical", "landscape": "horizontal"}.get(orientation, "all")
    r = http().get("https://pixabay.com/api/", timeout=30, params={
        "key": key, "q": query[:100], "image_type": "photo", "orientation": pix,
        "per_page": 20, "safesearch": "true"})
    raise_for_status(r, "Pixabay photo search")
    return [MediaItem("pixabay-photo", str(h["id"]), "image", h["largeImageURL"], h.get("imageWidth", 0),
                      h.get("imageHeight", 0), credit=h.get("user", ""), query=query, rank=rank,
                      text=str(h.get("tags") or ""))
            for rank, h in enumerate(r.json().get("hits", [])) if h.get("largeImageURL")]


# ------------------------------------------------------------------------------------------ AI images
def openai_image(prompt: str, dest: Path, key: str, model: str, orientation: str = "portrait",
                 reference_path: Path | None = None) -> Path:
    body = {"model": model, "prompt": prompt, "n": 1}
    size = {"portrait": "1024x1536", "landscape": "1536x1024"}.get(orientation, "1024x1024")
    use_reference = bool(reference_path and reference_path.is_file() and model.startswith("gpt-image"))
    if model.startswith("dall-e"):
        size = {"portrait": "1024x1792", "landscape": "1792x1024"}.get(orientation, "1024x1024")
        body.update(size=size, response_format="b64_json", quality="hd" if model == "dall-e-3" else "standard")
    else:
        body.update(size=size, quality="medium")

    def _do():
        headers = {"Authorization": f"Bearer {key}"}
        if use_reference:
            # OpenAI image edits accepts a source image and preserves its subject identity
            # while generating a new scene. Use it only for models supporting this endpoint.
            with reference_path.open("rb") as reference_file:
                response = http().post(
                    "https://api.openai.com/v1/images/edits",
                    data={"model": model, "prompt": prompt, "n": 1, "size": size, "quality": "medium"},
                    files={"image": (reference_path.name, reference_file, "image/png")},
                    timeout=240, headers=headers,
                )
        else:
            response = http().post("https://api.openai.com/v1/images/generations", json=body,
                                   timeout=240, headers=headers)
        raise_for_status(response, "OpenAI image")
        d = response.json()["data"][0]
        if d.get("b64_json"):
            dest.write_bytes(base64.b64decode(d["b64_json"]))
        else:
            download(d["url"], dest)
        return dest

    return with_retries(_do, attempts=2, label="OpenAI image")


def cloudflare_image(prompt: str, dest: Path, account_id: str, token: str) -> Path:
    """Generate one image with Cloudflare Workers AI FLUX.1 schnell."""
    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + quote(account_id, safe="")
        + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    response = http().post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"prompt": prompt[:5000], "steps": 4},
        timeout=240,
    )
    raise_for_status(response, "Cloudflare Workers AI image")
    result = response.json()
    if not result.get("success"):
        errors = result.get("errors") or []
        raise RuntimeError("Cloudflare image generation failed: " + json.dumps(errors)[:500])
    image_data = (result.get("result") or {}).get("image")
    if not image_data:
        raise RuntimeError("Cloudflare image response did not contain image data")
    dest.write_bytes(base64.b64decode(image_data))
    return dest


_POLLINATIONS_LOCK = threading.Lock()


def pollinations_image(prompt: str, dest: Path, width: int, height: int, key: str = "") -> Path:
    url = (f"https://image.pollinations.ai/prompt/{quote(prompt[:900])}"
           f"?width={width}&height={height}&nologo=true&model=flux&seed={random.randint(1, 10**9)}")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    # One request at a time: parallel scenes would all be rate-limited. Since September 2026 anonymous
    # use is limited to roughly one image every few minutes (HTTP 402 "payment required" otherwise).
    with _POLLINATIONS_LOCK:
        return download(url, dest, headers=headers, timeout=180, max_bytes=30 * 1024 * 1024)


# ------------------------------------------------------------------------------------------ selection
def simplify_query(query: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", query.lower()) if w not in STOP_WORDS]
    return " ".join(words[:2])



def expand_commons_queries(query: str) -> list[str]:
    """Generate focused alternative searches for scientific visuals."""
    q = query.lower()
    alternatives = []

    if any(w in q for w in ("astrocyte", "neuron", "microscopy")):
        alternatives.extend([
            "astrocytes neurons microscopy",
            "astrocyte immunofluorescence",
            "neurons glial cells microscopy",
            "brain cells fluorescence microscopy",
        ])

    if any(w in q for w in ("myelin", "myelinated", "nerve fiber", "nerve fibers")):
        alternatives.extend([
            "myelinated nerve fibers microscopy",
            "myelin immunofluorescence",
            "nerve fiber microscopy",
        ])

    if "brain" in q and any(w in q for w in ("glowing", "neon", "light", "illustration", "visual")):
        alternatives.extend([
            "brain neural network illustration",
            "human brain MRI scan",
            "brain scientific illustration",
        ])

    if "brain" in q and any(w in q for w in ("sagittal", "anatomy", "section", "mri")):
        alternatives.extend([
            "human brain sagittal section",
            "brain sagittal MRI",
            "brain anatomy medical illustration",
        ])

    if "black hole" in q or "black holes" in q:
        alternatives.extend([
            "black hole accretion disk",
            "black hole space astronomy",
            "black hole illustration",
        ])

    result = []
    seen = set()
    for candidate in [query, *alternatives]:
        candidate = " ".join(candidate.split())
        key = candidate.casefold()
        if candidate and key not in seen:
            seen.add(key)
            result.append(candidate)
    return result




def _director_search_query(image_prompt: str) -> str:
    """Extract a concise, scene-specific search phrase from the director's visual brief."""
    prompt = (image_prompt or "").strip()
    if not prompt:
        return ""
    # Remove the internal mode labels before sending a human-readable search to stock providers.
    prompt = re.sub(
        r"^(?:DIAGRAM / EXPLANATORY GRAPHIC|ORIGINAL DRAWN ILLUSTRATION|"
        r"SYMBOLIC VISUAL METAPHOR|REAL-WORLD DOCUMENTARY FOOTAGE|"
        r"REALISTIC EDITORIAL PHOTOGRAPH):\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\bREUSABLE CHARACTER:.*$", "", prompt, flags=re.IGNORECASE | re.DOTALL)
    # The director is instructed to put the actual subject/action first. Ignore style boilerplate
    # after the first sentence and trim very long descriptions to keep search queries useful.
    first_sentence = re.split(r"(?<=[.!?])\s+", prompt, maxsplit=1)[0]
    first_sentence = re.sub(
        r"\b(cinematic|documentary-style|real-world|realistic|editorial|photograph|"
        r"portrait|vertical|framing|no text|no letters|no watermark|clear|specific|"
        r"visible|action|lighting|composition|illustration|footage|photography)\b",
        " ",
        first_sentence,
        flags=re.IGNORECASE,
    )
    words = re.findall(r"[A-Za-z0-9'-]+", first_sentence)
    # Keep the director's concrete noun/action phrase, not a whole prompt full of style terms.
    candidate = " ".join(words[:9]).strip(" -'")
    return candidate if len(candidate.split()) >= 2 else ""


def _slug_words(url: str) -> str:
    """'https://www.pexels.com/video/octopus-swimming-in-the-sea-2169880/' -> 'octopus swimming in the sea'."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return " ".join(w for w in re.split(r"[-_]+", slug) if w and not w.isdigit())


def _stem(word: str) -> str:
    """A tiny English stemmer, enough to match 'octopuses'/'octopus', 'swimming'/'swims', 'cities'/'city',
    'diving'/'dives' between a scene's query and a clip's description."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "shes", "ches", "xes", "zes", "uses")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "ls":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z]+", text.lower()) if w not in STOP_WORDS and len(w) > 1}


def relevance(item: MediaItem, query: str) -> float:
    """Conservative subject match; reject incidental metadata matches."""
    want = _terms(query)
    if not want:
        return 0.0

    presentation = _terms(
        "beautiful amazing surprising photo picture photograph image images "
        "illustration video clip"
    )
    subject = want - presentation
    if not subject:
        return 0.0

    # Preserve neutral scoring for media without searchable descriptive text.
    # Commons still requires strict subject evidence from its filename/categories.
    if item.source != "commons" and not (
        item.text or item.extra.get("description") or item.extra.get("categories")
    ):
        return 0.5

    if item.source == "commons":
        filename = _terms(item.id)
        description = _terms(str(item.extra.get("description", "") or ""))
        categories = _terms(str(item.extra.get("categories", "") or ""))

        filename_hits = subject & filename
        category_hits = subject & categories
        description_hits = subject & description
        strong_hits = filename_hits | category_hits

        # Require a filename match and enough independent subject evidence.
        # Place names such as Antarctica must match the filename/categories;
        # description-only keyword coincidences cannot rescue a bad result.
        if not filename_hits:
            return 0.0

        missing = subject - strong_hits
        place_terms = {"antarctica", "arctic", "moon", "mars", "earth"}
        if missing & place_terms:
            return 0.0

        if len(subject) >= 3 and len(strong_hits) < 2:
            return 0.0

        filename_score = len(filename_hits) / len(subject)
        category_score = len(category_hits) / len(subject)
        description_score = len(description_hits) / len(subject)

        score = (
            0.70 * filename_score
            + 0.20 * category_score
            + 0.10 * description_score
        )
        return min(1.0, score)

    evidence = _terms(item.text)
    hits = subject & evidence
    if not hits:
        return 0.0
    return len(hits) / len(subject)


def _strong_stock_match(item: MediaItem, query: str, directed_query: str = "",
                        image_prompt: str = "") -> bool:
    """Require scene-specific metadata and reject misleading styles for documentary requests."""
    text = " ".join((item.text or "", str(item.extra.get("description", "")),
                     str(item.extra.get("categories", "")))).strip()
    if not text:
        return False
    tags = _terms(text)
    mode = (image_prompt or "").upper()
    if "REAL-WORLD DOCUMENTARY FOOTAGE:" in mode or "REALISTIC EDITORIAL PHOTOGRAPH:" in mode:
        unwanted = _terms("cartoon animation animated anime illustration 3d render cgi christmas santa toy mascot logo ai generated abstract")
        if tags & unwanted:
            return False
    # The search phrase and the director's shot brief serve different purposes: the former
    # anchors the story subject, while the latter describes what must actually be visible.
    # Score against the shot brief first; never let a broad topic ("island mystery", "space")
    # make unrelated high-ranked stock footage look relevant.
    generic = _terms(
        "beautiful amazing surprising photo picture photograph image images illustration "
        "video clip footage cinematic documentary realistic real world portrait vertical "
        "framing visible scene shot showing show closeup close up"
    )
    query_terms = _terms(query) - generic
    target_terms = _terms(directed_query or query) - generic
    if not target_terms:
        target_terms = query_terms
    target_hits = target_terms & tags
    query_hits = query_terms & tags

    # Require multiple independent clues for longer shot briefs, and meaningful evidence
    # for the story's subject too. A lone incidental tag must not pass the gate.
    target_required = 1 if len(target_terms) <= 2 else 2
    if len(target_hits) < target_required:
        return False
    if target_hits and len(target_hits) / max(1, len(target_terms)) < 0.30:
        return False
    if query_terms and not query_hits:
        # Some director prompts use a clear synonym or a specific action rather than the
        # script's broader query. Permit this only when the shot brief itself is well matched.
        if len(target_hits) < max(2, target_required):
            return False
    if relevance(item, query) < 0.34:
        return False
    return True


def _shape(item: MediaItem) -> str:
    if item.height > item.width:
        return "portrait"
    return "landscape" if item.width > item.height else "square"


def _score(item: MediaItem, need: float, orientation: str = "portrait", query: str = "") -> float:
    """Higher is better. Relevance to the scene matters most, then the search engine's own ranking, then
    whether the clip fits the frame and lasts long enough."""
    s = 4.0 * relevance(item, query or item.query)
    s += 2.0 / (1.0 + 0.25 * item.rank)
    if _shape(item) == orientation or (orientation == "square" and item.width and item.height):
        s += 2.5
    if item.kind == "video":
        s += 2 if item.duration >= need else (1 if item.duration >= need * 0.6 else 0)
    if min(item.width, item.height) >= 1080:
        s += 1
    return s + random.uniform(0, 0.6)  # a little variety between near-identical candidates


class Visuals:
    def __init__(self, settings: Settings, used_keys: set[str] | None = None):
        self.s = settings
        self.used = set(used_keys or ())
        self.lock = threading.Lock()
        self._search_cache: dict[tuple[str, str], list[MediaItem]] = {}
        self.sources = [x for x in settings.visual_sources if x]
        # Commons first for normal stock scenes; Pixabay is the first fallback.
        # Keep only explicitly configured providers, but normalize their priority.
        priority = {"commons": 0, "pixabay": 1, "pixabay-photos": 2,
                    "pexels": 3, "pexels-photos": 4}
        self.sources = sorted(enumerate(self.sources), key=lambda pair: (priority.get(pair[1], 5), pair[0]))
        self.sources = [source for _, source in self.sources]
        if settings.offline:
            self.sources = ["local"]
        self.credits: set[str] = set()

    # -- search per source (cached per query) --
    def _search(self, source: str, query: str) -> list[MediaItem]:
        ck = (source, query.lower())
        if ck in self._search_cache:
            return self._search_cache[ck]
        # Persistent 24-hour cache: empty results are cached too, preventing repeat
        # API calls for scenes that have already failed to find suitable media.
        disk_cache = Path(self.s.media_dir) / ".purffle_search_cache.json"
        cache = {}
        cache_key = f"{source}|{query.strip().casefold()}"
        try:
            cache = json.loads(disk_cache.read_text(encoding="utf-8")) if disk_cache.exists() else {}
            record = cache.get(cache_key)
            if record and time.time() - float(record.get("saved_at", 0)) < 86400:
                restored = [MediaItem(**entry) for entry in record.get("items", [])]
                self._search_cache[ck] = restored
                return restored
        except (OSError, ValueError, TypeError, KeyError) as e:
            log.debug("Visual search cache read skipped: %s", redact(e))
        s = self.s
        items: list[MediaItem] = []
        try:
            o = s.orientation
            if source == "pexels" and s.pexels_api_key:
                items = pexels_videos(query, s.pexels_api_key, s.prefer_4k, orientation=o)
                if len(items) < 3:
                    items += pexels_videos(query, s.pexels_api_key, s.prefer_4k, orientation="")
                if not items:
                    items = pexels_photos(query, s.pexels_api_key, orientation=o)
            elif source == "pexels-photos" and s.pexels_api_key:
                items = pexels_photos(query, s.pexels_api_key, orientation=o)
            elif source == "pixabay" and s.pixabay_api_key:
                items = (pixabay_videos(query, s.pixabay_api_key, s.prefer_4k)
                         or pixabay_photos(query, s.pixabay_api_key, orientation=o))
            elif source == "pixabay-photos" and s.pixabay_api_key:
                items = pixabay_photos(query, s.pixabay_api_key, orientation=o)
            elif source == "commons":
                items = commons_search(query, orientation=o)
            elif source == "local":
                items = self._local(query)
            seen: set[str] = set()
            items = [i for i in items if not (i.key in seen or seen.add(i.key))]
        except PermanentError as e:
            log.warning("%s search failed permanently (%s); disabling it for this video", source, redact(e))
            self.sources = [x for x in self.sources if x != source]
        except Exception as e:
            log.warning("%s search for %r failed: %s", source, query, redact(e))
        self._search_cache[ck] = items
        try:
            disk_cache.parent.mkdir(parents=True, exist_ok=True)
            cache[cache_key] = {
                "saved_at": time.time(),
                "items": [{
                    "source": item.source, "id": item.id, "kind": item.kind,
                    "url": item.url, "text": item.text, "credit": item.credit,
                    "width": item.width, "height": item.height, "duration": item.duration,
                    "rank": item.rank, "query": item.query, "extra": item.extra,
                } for item in items],
            }
            cutoff = time.time() - 86400
            cache = {k: v for k, v in cache.items() if float(v.get("saved_at", 0)) >= cutoff}
            if len(cache) > 500:
                cache = dict(sorted(cache.items(), key=lambda pair: pair[1].get("saved_at", 0), reverse=True)[:500])
            temp_cache = disk_cache.with_suffix(".tmp")
            temp_cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            temp_cache.replace(disk_cache)
        except (OSError, ValueError, TypeError) as e:
            log.debug("Visual search cache write skipped: %s", redact(e))
        return items

    def _local(self, query: str) -> list[MediaItem]:
        root = Path(self.s.media_dir)
        if not root.is_dir():
            return []
        words = set(simplify_query(query).split()) | set(query.lower().split())
        # background videos (media/backgrounds, BACKGROUND_VIDEO) play under stories, never as scene footage
        skip = [(root / "backgrounds").resolve(), (root / ".purffle_assets").resolve()]
        if self.s.background_video:
            skip.append(Path(self.s.background_video).expanduser().resolve())
        items = []
        for p in root.rglob("*"):
            ext = p.suffix.lower()
            if ext not in VIDEO_EXT | IMAGE_EXT:
                continue
            rp = p.resolve()
            if any(rp == d or d in rp.parents for d in skip):
                continue
            stem = set(re.findall(r"[a-z]+", p.stem.lower()))
            item = MediaItem("local", str(p), "video" if ext in VIDEO_EXT else "image", path=p, query=query)
            item.extra["overlap"] = len(words & stem)
            items.append(item)
        items.sort(key=lambda i: i.extra["overlap"], reverse=True)
        return items

    def _claim(self, candidates: list[MediaItem], need: float, query: str = "") -> MediaItem | None:
        with self.lock:
            fresh = [c for c in candidates if c.key not in self.used]
            if not fresh:
                return None
            if fresh[0].source == "local":
                best = fresh[0] if fresh[0].extra.get("overlap") else random.choice(fresh)
            else:
                best = max(fresh, key=lambda c: _score(c, need, self.s.orientation, query))
            self.used.add(best.key)
            return best

    def _fetch(self, item: MediaItem, dest_dir: Path, idx: int) -> MediaItem:
        if item.source == "local":
            return item
        if item.source == "commons":
            ext = {
                "video/webm": ".webm",
                "video/ogg": ".ogv",
                "video/mp4": ".mp4",
                "image/png": ".png",
                "image/jpeg": ".jpg",
            }.get(item.extra.get("mime", ""), ".jpg")
        else:
            ext = ".mp4" if item.kind == "video" else ".jpg"
        item.path = download(item.url, dest_dir / f"scene{idx:02d}_{item.source}_{item.id}{ext}")
        return item

    def _ai(self, source: str, prompt: str, dest_dir: Path, idx: int, reference_path: Path | None = None) -> MediaItem | None:
        s = self.s
        full = f"{prompt}. {s.visual_style}. No text, no letters, no watermark."
        dest = dest_dir / f"scene{idx:02d}_ai.png"
        try:
            if source in ("openai-images", "ai", "dalle", "gpt-image") and s.openai_api_key:
                openai_image(full, dest, s.openai_api_key, s.image_model, s.orientation, reference_path=reference_path)
            elif source == "pollinations":
                # Anonymous Pollinations requests can sit for minutes and are now rate-limited.
                # Do not let a free, unkeyed endpoint stall every fantasy scene in the queue.
                if not s.pollinations_api_key:
                    log.warning(
                        "Skipping anonymous Pollinations for scene %d: configure POLLINATIONS_API_KEY "
                        "or an OpenAI image key. Refusing a long wait on the anonymous endpoint.",
                        idx + 1,
                    )
                    with self.lock:
                        self.sources = [x for x in self.sources if x != source]
                    return None
                pollinations_image(full, dest.with_suffix(".jpg"), s.width, s.height, s.pollinations_api_key)
                dest = dest.with_suffix(".jpg")
            elif source == "cloudflare" and s.cloudflare_account_id and s.cloudflare_api_token:
                cloudflare_image(full, dest, s.cloudflare_account_id, s.cloudflare_api_token)
            else:
                return None
        except PermanentError as e:
            # A refused key or an exhausted free tier won't recover within this video: stop asking.
            with self.lock:
                if source in self.sources:
                    self.sources = [x for x in self.sources if x != source]
                    hint = (" Anonymous Pollinations use is now heavily rate-limited; set POLLINATIONS_API_KEY "
                            "or use Pexels/Pixabay (free keys)." if source == "pollinations" and " 402" in str(e)
                            else "")
                    log.warning("AI images from %s refused (%s); skipping it for this video.%s",
                                source, redact(e)[:160], hint)
            return None
        except Exception as e:
            log.warning("AI image (%s) for scene %d failed: %s", source, idx + 1, redact(e))
            return None
        log.info("Scene %d: %s image", idx + 1, source)
        return MediaItem(source, dest.name, "image", path=dest, width=s.width, height=s.height, query=prompt)

    def for_scene(self, idx: int, query: str, image_prompt: str, topic: str, need: float, dest_dir: Path) -> MediaItem:
        # Use the built-in illustration/diagram library for a meaningful share of scenes.
        # Explanatory scenes prefer diagrams; atmospheric/story scenes use drawings often,
        # while a smaller share can still use real footage/photos for realism and variety.
        # An explicit gradient is a designed visual choice, not a failed search result.
        fantasy_mode = "FANTASY CINEMATIC FOOTAGE:" in (image_prompt or "").upper()
        character_marker = "REUSABLE CHARACTER:"
        character_mode = character_marker in (image_prompt or "").upper()
        character_reference_path: Path | None = None
        style_key = ""
        if character_mode:
            parts = re.split(re.escape(character_marker), image_prompt or "", maxsplit=1, flags=re.IGNORECASE)
            style_key = (parts[-1].splitlines()[0].strip() if len(parts) > 1 else "")[:240]
        if fantasy_mode and not style_key:
            style_key = f"{topic} | {query} | original fantasy character"
        if fantasy_mode:
            # The character is the subject, not a background search term. Generate an
            # original reference sheet, then prefer AI generation for every scene.
            ref_root = Path(self.s.media_dir) / ".purffle_assets" / "character_references"
            ref_root.mkdir(parents=True, exist_ok=True)
            ref_key = __import__("hashlib").sha256(style_key.casefold().encode("utf-8")).hexdigest()[:20]
            ref_path = ref_root / f"{ref_key}.png"
            character_reference_path = ref_path
            # Only create a reference sheet when the configured image endpoint can actually
            # feed that image back into later generations. Pollinations/stock prompts do not
            # accept this reference in our integration, so spending an extra image request there
            # would create an unused portrait and falsely imply stronger identity consistency.
            # Anonymous Pollinations is intentionally excluded: it is heavily rate-limited and
            # has caused multi-minute stalls before the first scene can render.
            can_reuse_reference = bool(
                self.s.openai_api_key
                and self.s.image_model.startswith("gpt-image")
                and any(x in self.sources for x in ("openai-images", "ai", "dalle", "gpt-image"))
            )
            if can_reuse_reference and not ref_path.is_file():
                ref_prompt = (
                    "ORIGINAL FICTIONAL FANTASY CHARACTER MODEL SHEET, one character only, clean neutral "
                    "background, one clear face portrait and one full-body view of the SAME design, "
                    "distinctive silhouette, readable eyes and expression, consistent costume and colors. "
                    "No text, logos, watermarks or other characters. Character identity: " + style_key
                )
                made = self._ai("openai-images", ref_prompt, ref_root, int(ref_key[:6], 16) % 100000)
                if made and made.path and made.path.is_file():
                    try:
                        shutil.copy2(made.path, ref_path)
                    except OSError:
                        pass
            if can_reuse_reference and not ref_path.is_file():
                log.warning("Reusable fantasy character reference unavailable; using the repeated textual character bible.")
            visual_kind = "character"
            directed_mode = "character"
            stable = int(__import__("hashlib").sha1(f"fantasy|{topic}|{idx}|{query}".encode()).hexdigest()[:8], 16)
        else:
            if "ABSTRACT GRADIENT BACKGROUND:" in (image_prompt or "").upper():
                return MediaItem("generated", f"intentional-gradient-{idx}", "generated",
                                 query=query or topic, extra={"intentional_gradient": True})
            visual_kind = visual_type(query or topic, image_prompt)
            directed_mode = ""
            for marker, mode in (
                ("DIAGRAM / EXPLANATORY GRAPHIC:", "diagram"),
                ("ORIGINAL DRAWN ILLUSTRATION:", "illustration"),
                ("SYMBOLIC VISUAL METAPHOR:", "symbolic"),
                ("REAL-WORLD DOCUMENTARY FOOTAGE:", "footage"),
                ("FANTASY CINEMATIC FOOTAGE:", "footage"),
                ("REALISTIC EDITORIAL PHOTOGRAPH:", "photo"),
            ):
                if marker in (image_prompt or "").upper():
                    directed_mode = mode
                    break
            if directed_mode in ("diagram", "illustration", "symbolic"):
                visual_kind = directed_mode
            stable = int(__import__("hashlib").sha1(f"{topic}|{idx}|{query}".encode()).hexdigest()[:8], 16)
        # Explicit director choices override the usual library/stock balance; otherwise keep variety.
        # Respect the configured source order: a Pexels-only or Pollinations-only setup
        # must actually try that source for ordinary scenes. The built-in library is the
        # default when local artwork is requested, for explicit director illustration/
        # diagram choices, and as the final fallback when no source yields a usable match.
        # Respect explicit provider choices: Pexels/Pixabay/AI sources should be tried
        # instead of being silently bypassed by local artwork. The library remains the
        # preferred source for director-led illustrations/diagrams and when no provider exists.
        # Factual videos should not silently substitute generic cartoon assets for real subject
        # footage. Use the built-in library for explicit mechanism diagrams; only use illustration
        # or symbolic artwork when no external visual provider is configured. If a provider is
        # configured, attempt a metadata-matched asset first and keep the offline library as a
        # last resort rather than letting a director style tag override factual relevance.
        use_library = (
            (directed_mode == "diagram" and not self.sources)
            or (directed_mode in ("illustration", "symbolic") and not self.sources)
            or (not directed_mode and visual_kind == "diagram" and not self.sources)
            or (not directed_mode and not self.sources and stable % 100 < 78)
        )
        if character_mode and not fantasy_mode:
            ref_root = Path(self.s.media_dir) / ".purffle_assets" / "character_references"
            ref_root.mkdir(parents=True, exist_ok=True)
            ref_key = __import__("hashlib").sha256(style_key.casefold().encode("utf-8")).hexdigest()[:20]
            ref_path = ref_root / f"{ref_key}.png"
            character_reference_path = ref_path
            if not ref_path.is_file():
                ref_prompt = "ORIGINAL FICTIONAL CHARACTER REFERENCE SHEET. Character design: " + style_key
                for provider in self.sources:
                    if provider in ("openai-images", "ai", "dalle", "gpt-image", "pollinations", "cloudflare"):
                        made = self._ai(provider, ref_prompt, ref_root, int(ref_key[:6], 16) % 100000)
                        if made and made.path and made.path.is_file():
                            try:
                                shutil.copy2(made.path, ref_path)
                                break
                            except OSError:
                                pass
            directed_mode = "character"
            use_library = False
        if fantasy_mode:
            use_library = False
        if use_library:
            try:
                library_root = Path(self.s.media_dir) / ".purffle_assets"
                asset = get_or_create_asset(
                    library_root, query or topic, image_prompt, self.s.width, self.s.height,
                    variation=idx, mode=directed_mode or visual_kind,
                    style_key=style_key or self.s.visual_style,
                )
                library_path = dest_dir / f"scene{idx:02d}_library.png"
                shutil.copy2(asset["path"], library_path)
                log.info("Scene %d: persistent %s asset (%s)", idx + 1, asset["visual_type"], asset["id"])
                return MediaItem(
                    "library", library_path.name, "image", path=library_path,
                    width=self.s.width, height=self.s.height, query=query or topic,
                    text=f"reusable {asset['visual_type']} artwork",
                    extra={"visual_type": asset["visual_type"], "asset_id": asset["id"],
                           "character": asset.get("character")},
                )
            except Exception as e:
                log.warning("Persistent visual library failed for scene %d: %s", idx + 1, redact(e))
        queries = []
        # Search first for the script's chosen subject, then for the director's concrete visible
        # action. Broad topic-only searches come last so they cannot dominate scene-specific results.
        directed_query = _director_search_query(image_prompt)
        for q in (directed_query, query, simplify_query(query), simplify_query(topic), topic):
            q = (q or "").strip()
            if q and q.lower() not in [x.lower() for x in queries]:
                queries.append(q)
        source_order = list(self.sources)
        fantasy_prompt = (
            "MAIN SUBJECT MUST BE A CLEARLY VISIBLE ORIGINAL FANTASY CHARACTER, filling at least 45% "
            "of the vertical frame. NOT a landscape-only shot. Show the creature/person's face, eyes, "
            "mouth and body in the same recognizable design; cinematic realistic CGI/live-action fantasy "
            "film still, expressive face, readable silhouette, dramatic but clear lighting. "
            "If dialogue is being spoken, frame the character's face/muzzle prominently. "
            "Preserve this exact character identity: " + (style_key or query or topic) +
            ". Scene action: " + (image_prompt or query or topic)
        ) if fantasy_mode else (image_prompt or query)
        if fantasy_mode:
            # Fantasy requires original character art. Prefer explicitly configured AI providers,
            # including Cloudflare Workers AI (the intended low-cost image provider). The UI hides
            # Cloudflare from the selectable list because its credentials are server-side secrets,
            # so include it here whenever both credentials are configured.
            ai_sources = [
                x for x in source_order
                if x in ("openai-images", "ai", "dalle", "gpt-image", "pollinations", "cloudflare")
            ]
            if self.s.cloudflare_account_id and self.s.cloudflare_api_token and "cloudflare" not in ai_sources:
                ai_sources.insert(0, "cloudflare")
            if self.s.openai_api_key and not any(x in ai_sources for x in ("openai-images", "ai", "dalle", "gpt-image")):
                ai_sources.insert(0, "openai-images")
            # Do not silently fall back to anonymous Pollinations: it is rate-limited and can
            # stall the job. Only use it when explicitly selected and a key is configured.
            if not self.s.pollinations_api_key:
                ai_sources = [x for x in ai_sources if x != "pollinations"]
            # Never fall through to unrelated stock people just because AI generation is unavailable.
            source_order = ai_sources
        for source in source_order:
            if source in ("openai-images", "ai", "dalle", "gpt-image", "pollinations", "cloudflare"):
                item = self._ai(source, fantasy_prompt, dest_dir, idx, reference_path=character_reference_path if fantasy_mode else None)
                if item:
                    return item
                continue
            if source == "commons":
                # Search the most scene-specific phrases first; stop on a credible match.
                search_queries = []
                for candidate_query in (directed_query, simplify_query(query), simplify_query(topic)):
                    candidate_query = (candidate_query or "").strip()
                    if candidate_query and candidate_query.casefold() not in {x.casefold() for x in search_queries}:
                        search_queries.append(candidate_query)
                combined = {}
                item = None
                for search_query in search_queries[:2]:
                    try:
                        for candidate in self._search(source, search_query):
                            combined.setdefault(candidate.key, candidate)
                    except Exception as e:
                        log.warning("Commons search failed for %r: %s", search_query, redact(e))
                    cand = [candidate for candidate in combined.values() if _strong_stock_match(candidate, query, directed_query, image_prompt)]
                    if cand:
                        item = self._claim(cand, need, directed_query or query)
                        if item:
                            q = search_query
                            break
                if not item:
                    continue
            else:
                # At most two targeted searches per provider; later providers run only if needed.
                combined = {}
                item = None
                for search_query in queries[:2]:
                    try:
                        for candidate in self._search(source, search_query):
                            combined.setdefault(candidate.key, candidate)
                    except Exception as e:
                        log.warning("%s search failed for %r: %s", source, search_query, redact(e))
                    cand = [candidate for candidate in combined.values() if ((candidate.source == "local" and candidate.extra.get("overlap", 0) > 0) or (candidate.source != "local" and _strong_stock_match(candidate, query, directed_query, image_prompt)))]
                    if cand:
                        item = self._claim(cand, need, directed_query or query)
                        if item:
                            q = search_query
                            break
                if not item:
                    continue
            try:
                self._fetch(item, dest_dir, idx)
            except Exception as e:
                log.warning("Download failed for %s: %s", item.key, redact(e))
                continue

            if item.source.startswith("pexels"):
                self.credits.add("Pexels")
            elif item.source.startswith("pixabay"):
                self.credits.add("Pixabay")
            elif item.source == "commons":
                license_name = item.extra.get("license", "Licence not listed")
                artist = item.extra.get("artist") or item.credit or "Creator not listed"
                source_page = item.extra.get("source_page", "")
                license_url = item.extra.get("license_url", "")
                self.credits.add(
                    f"{item.text} — {artist} — {license_name}"
                    f" — {source_page} — {license_url}"
                )

            log.info("Scene %d: %s %s (%r)", idx + 1, item.source, item.kind, q)
            return item
        # Fantasy role-play explicitly prefers real cinematic footage. If no credible footage or
        # configured AI visual source exists, do not silently replace it with random vector drawings.
        fantasy_footage = fantasy_mode
        if fantasy_footage:
            log.error("Scene %d: character image generation failed. Refusing to render a blank/gradient scene as if it contained a character.", idx + 1)
            raise RuntimeError(
                f"Fantasy scene {idx + 1} could not generate a character image. "
                "Check the configured image provider. Fantasy needs Cloudflare Workers AI credentials or an OpenAI image key; anonymous Pollinations is not a reliable fallback."
            )
        # Last resort: meaningful offline artwork beats a generic gradient, especially on
        # low-cost deployments with no usable external footage or image API.
        try:
            library_root = Path(self.s.media_dir) / ".purffle_assets"
            asset = get_or_create_asset(
                library_root, query or topic, image_prompt, self.s.width, self.s.height,
                variation=idx, mode=directed_mode or visual_kind, style_key=self.s.visual_style,
            )
            library_path = dest_dir / f"scene{idx:02d}_library_fallback.png"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset["path"], library_path)
            log.info("Scene %d: offline visual-library fallback (%s)", idx + 1, asset["id"])
            return MediaItem(
                "library", library_path.name, "image", path=library_path,
                width=self.s.width, height=self.s.height, query=query or topic,
                text=f"offline {asset['visual_type']} artwork",
                extra={"visual_type": asset["visual_type"], "asset_id": asset["id"],
                       "character": asset.get("character")},
            )
        except Exception as e:
            log.warning("Offline visual fallback failed for scene %d: %s", idx + 1, redact(e))
        log.info("Scene %d: no usable visual found for %r — using an animated background", idx + 1, query)
        return MediaItem("generated", f"gradient-{idx}", "generated", query=query)

    def for_scenes(self, jobs: list[tuple[str, str, float]], topic: str, dest_dir: Path,
                   progress=None) -> list[MediaItem]:
        """jobs: [(search_query, image_prompt, seconds_needed)] -> one MediaItem per scene. ``progress`` gets
        the fraction of scenes that have their footage."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        workers = 2 if any(s in ("openai-images", "ai", "pollinations", "cloudflare") for s in self.sources) else 6
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self.for_scene, i, q, p, topic, need, dest_dir) for i, (q, p, need) in enumerate(jobs)]
            for done, _ in enumerate(concurrent.futures.as_completed(futs), 1):
                if progress:
                    progress(done / len(futs))
            return [f.result() for f in futs]
