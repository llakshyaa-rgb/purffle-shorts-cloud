"""Visual sourcing: per-scene stock footage and photos (Pexels, Pixabay), your own media folder,
and AI-generated images (OpenAI gpt-image-1 / DALL·E, Pollinations). Clips are never reused across
videos, portrait footage is preferred, and a scene always gets *something* (animated gradient)."""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from .config import Settings
from .utils import PermanentError, download, http, raise_for_status, redact, with_retries

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
def openai_image(prompt: str, dest: Path, key: str, model: str, orientation: str = "portrait") -> Path:
    body = {"model": model, "prompt": prompt, "n": 1}
    if model.startswith("dall-e"):
        size = {"portrait": "1024x1792", "landscape": "1792x1024"}.get(orientation, "1024x1024")
        body.update(size=size, response_format="b64_json", quality="hd" if model == "dall-e-3" else "standard")
    else:
        size = {"portrait": "1024x1536", "landscape": "1536x1024"}.get(orientation, "1024x1024")
        body.update(size=size, quality="medium")

    def _do():
        r = http().post("https://api.openai.com/v1/images/generations", json=body, timeout=240,
                        headers={"Authorization": f"Bearer {key}"})
        raise_for_status(r, "OpenAI image")
        d = r.json()["data"][0]
        if d.get("b64_json"):
            dest.write_bytes(base64.b64decode(d["b64_json"]))
        else:
            download(d["url"], dest)
        return dest

    return with_retries(_do, attempts=2, label="OpenAI image")


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
        if settings.offline:
            self.sources = ["local"]
        self.credits: set[str] = set()

    # -- search per source (cached per query) --
    def _search(self, source: str, query: str) -> list[MediaItem]:
        ck = (source, query.lower())
        if ck in self._search_cache:
            return self._search_cache[ck]
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
        return items

    def _local(self, query: str) -> list[MediaItem]:
        root = Path(self.s.media_dir)
        if not root.is_dir():
            return []
        words = set(simplify_query(query).split()) | set(query.lower().split())
        # background videos (media/backgrounds, BACKGROUND_VIDEO) play under stories, never as scene footage
        skip = [(root / "backgrounds").resolve()]
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

    def _ai(self, source: str, prompt: str, dest_dir: Path, idx: int) -> MediaItem | None:
        s = self.s
        full = f"{prompt}. {s.visual_style}. No text, no letters, no watermark."
        dest = dest_dir / f"scene{idx:02d}_ai.png"
        try:
            if source in ("openai-images", "ai", "dalle", "gpt-image") and s.openai_api_key:
                openai_image(full, dest, s.openai_api_key, s.image_model, s.orientation)
            elif source == "pollinations":
                pollinations_image(full, dest.with_suffix(".jpg"), s.width, s.height, s.pollinations_api_key)
                dest = dest.with_suffix(".jpg")
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
        queries = []
        for q in (query, simplify_query(query), simplify_query(topic), topic):
            q = (q or "").strip()
            if q and q.lower() not in [x.lower() for x in queries]:
                queries.append(q)
        for source in list(self.sources):
            if source in ("openai-images", "ai", "dalle", "gpt-image", "pollinations"):
                item = self._ai(source, image_prompt or query, dest_dir, idx)
                if item:
                    return item
                continue
            if source == "commons":
                search_queries = expand_commons_queries(query or topic)
                seen_queries = {q.casefold() for q in search_queries}
                for extra_query in queries:
                    if extra_query and extra_query.casefold() not in seen_queries:
                        search_queries.append(extra_query)
                        seen_queries.add(extra_query.casefold())

                combined = {}
                for search_query in search_queries:
                    try:
                        for candidate in self._search(source, search_query):
                            combined.setdefault(candidate.key, candidate)
                    except Exception as e:
                        log.warning(
                            "Commons search failed for %r: %s",
                            search_query, redact(e)
                        )

                candidates = list(combined.values())
                ranked = sorted(
                    candidates,
                    key=lambda candidate: relevance(candidate, query),
                    reverse=True,
                )
                cand = [
                    candidate for candidate in ranked
                    if relevance(candidate, query) >= 0.45
                ]
                q = query
                item = self._claim(cand, need, query)
                if not item:
                    continue
            else:
                combined = {}
                for q in queries:
                    try:
                        for candidate in self._search(source, q):
                            combined.setdefault(candidate.key, candidate)
                    except Exception as e:
                        log.warning(
                            "%s search failed for %r: %s",
                            source, q, redact(e)
                        )

                cand = [
                    candidate for candidate in combined.values()
                    if candidate.source == "local"
                    or relevance(candidate, query) > 0
                ]
                item = self._claim(cand, need, query)
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
        log.info("Scene %d: no footage found for %r — using an animated background", idx + 1, query)
        return MediaItem("generated", f"gradient-{idx}", "generated", query=query)

    def for_scenes(self, jobs: list[tuple[str, str, float]], topic: str, dest_dir: Path,
                   progress=None) -> list[MediaItem]:
        """jobs: [(search_query, image_prompt, seconds_needed)] -> one MediaItem per scene. ``progress`` gets
        the fraction of scenes that have their footage."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        workers = 2 if any(s in ("openai-images", "ai", "pollinations") for s in self.sources) else 6
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self.for_scene, i, q, p, topic, need, dest_dir) for i, (q, p, need) in enumerate(jobs)]
            for done, _ in enumerate(concurrent.futures.as_completed(futs), 1):
                if progress:
                    progress(done / len(futs))
            return [f.result() for f in futs]
