"""Transparent, lightweight recurring narrator mascot for Purffle Shorts.

The mascot is drawn locally with Pillow (no image API, model download, or paid service).
FFmpeg composites these RGBA sprites over the finished video. Its hooded silhouette,
subtle cyan eyes, gestures, brief appearances, and tiny topic callouts are intentionally
simple so the story visuals remain the focus.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/system/fonts/Roboto-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _mascot(path: Path, width: int, height: int, pose: str) -> Path:
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # A dark, mysterious hooded presenter. All shapes stay on a transparent canvas.
    cx = width // 2
    ink = (12, 15, 25, 245)
    edge = (85, 104, 132, 235)
    glow = (95, 232, 255, 255)
    # Hood and head
    d.ellipse((cx-width*.27, height*.03, cx+width*.27, height*.38), fill=ink, outline=edge, width=max(2,width//65))
    d.ellipse((cx-width*.19, height*.105, cx+width*.19, height*.335), fill=(3, 7, 15, 255))
    # Eyes glow without requiring external assets
    eye_y = int(height*.225)
    eye_dx = int(width*.075)
    d.ellipse((cx-eye_dx-width*.035, eye_y-width*.025, cx-eye_dx+width*.035, eye_y+width*.025), fill=glow)
    d.ellipse((cx+eye_dx-width*.035, eye_y-width*.025, cx+eye_dx+width*.035, eye_y+width*.025), fill=glow)
    # Cloak
    d.polygon([(cx-width*.22,height*.30),(cx+width*.22,height*.30),
               (cx+width*.43,height*.96),(cx-width*.43,height*.96)], fill=ink, outline=edge)
    d.line((cx,height*.34,cx,height*.90), fill=(52,67,91,220), width=max(2,width//90))
    # Arms differ by pose; joints are rounded to keep the silhouette clean.
    stroke=max(8,width//14)
    if pose == "point":
        d.line((cx-width*.12,height*.49,cx+width*.10,height*.56,cx+width*.44,height*.40),
               fill=ink,width=stroke,joint="curve")
        d.ellipse((cx+width*.39,height*.35,cx+width*.49,height*.45),fill=ink)
        d.line((cx-width*.15,height*.49,cx-width*.28,height*.68),fill=ink,width=stroke)
    elif pose == "startled":
        d.line((cx-width*.16,height*.48,cx-width*.35,height*.36),fill=ink,width=stroke)
        d.line((cx+width*.16,height*.48,cx+width*.35,height*.36),fill=ink,width=stroke)
    elif pose == "explain":
        d.line((cx-width*.15,height*.48,cx-width*.40,height*.57),fill=ink,width=stroke)
        d.line((cx+width*.15,height*.48,cx+width*.36,height*.56),fill=ink,width=stroke)
        d.ellipse((cx+width*.30,height*.49,cx+width*.42,height*.61),fill=ink)
    else:  # observing / thinking
        d.line((cx-width*.15,height*.48,cx-width*.32,height*.62),fill=ink,width=stroke)
        d.line((cx+width*.15,height*.48,cx+width*.28,height*.59),fill=ink,width=stroke)
    # Feet and a restrained outline highlight
    d.line((cx-width*.18,height*.92,cx-width*.25,height*.99),fill=ink,width=stroke)
    d.line((cx+width*.18,height*.92,cx+width*.25,height*.99),fill=ink,width=stroke)
    im.save(path)
    return path


def _callout(path: Path, width: int, height: int, text: str) -> Path:
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = max(8, width//24)
    radius = max(10, width//18)
    d.rounded_rectangle((2,2,width-3,height-8), radius=radius,
                        fill=(8,13,24,225), outline=(89,220,244,235), width=max(2,width//100))
    d.polygon([(width*.20,height-10),(width*.12,height-2),(width*.31,height-10)], fill=(8,13,24,225))
    font = _font(max(12, width//18))
    words = re.sub(r"[^\w\s'-]", "", text).split()
    lines, line = [], ""
    max_width = width - 2*pad
    for word in words[:8]:
        trial = (line+" "+word).strip()
        if d.textbbox((0,0),trial,font=font)[2] > max_width and line:
            lines.append(line); line=word
        else:
            line=trial
    if line:
        lines.append(line)
    line_h = max(15, int(height*.25))
    y = max(5,(height-len(lines)*line_h)//2-2)
    for line in lines[:3]:
        d.text((pad,y),line,font=font,fill=(245,250,255,255),stroke_width=0)
        y += line_h
    im.save(path)
    return path


def build_assets(work: Path, width: int, height: int, total: float, topic: str = "") -> list[dict]:
    """Create a handful of timed, transparent character/callout overlays.

    The mascot appears in short beats rather than occupying the screen continuously.
    Callouts are optional and use the topic phrase only, never invented narration.
    """
    root = work / "narrator"
    root.mkdir(parents=True, exist_ok=True)
    sw, sh = max(120, int(width*.205)), max(220, int(height*.235))
    poses = {}
    for name in ("observe", "point", "startled", "explain"):
        poses[name] = _mascot(root / f"{name}.png", sw, sh, name)
    # The character gets quiet gaps and shifts sides/poses so it doesn't feel like a watermark.
    windows = [
        (0.8, min(3.4,total), "observe", "right"),
        (max(0.0,total*.19), min(total*.19+2.2,total), "point", "left"),
        (max(0.0,total*.39), min(total*.39+1.8,total), "startled", "right"),
        (max(0.0,total*.61), min(total*.61+2.6,total), "explain", "left"),
        (max(0.0,total*.83), min(total*.83+1.8,total), "observe", "right"),
    ]
    assets = []
    for start,end,pose,side in windows:
        if end-start >= .45:
            assets.append({"path":poses[pose],"start":round(start,3),"end":round(end,3),
                           "kind":"mascot","side":side})
    clean = re.sub(r"\s+"," ",topic).strip(" -:|")
    if clean and total > 8:
        # A short topic label, not a fabricated explanatory sentence.
        label = " ".join(clean.split()[:5])[:48]
        bw, bh = max(240,int(width*.43)), max(78,int(height*.075))
        bubble = _callout(root / "topic.png",bw,bh,label)
        for start in (max(2.8,total*.32), max(4.5,total*.70)):
            end = min(total,start+1.65)
            if end-start >= .45:
                assets.append({"path":bubble,"start":round(start,3),"end":round(end,3),
                               "kind":"callout","side":"left" if start < total*.5 else "right"})
    return assets
