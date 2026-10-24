"""Built-in non-photographic visuals for Purffle Shorts.

Creates lightweight, reusable-style illustrated frames and explanatory diagrams with Pillow.
No image API, network access, or extra dependency is required beyond Purffle's existing Pillow.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DIAGRAM_WORDS = {
    "diagram", "how", "mechanism", "inside", "inside", "cross-section", "cross section",
    "layout", "map", "route", "path", "process", "system", "steps", "flow", "timeline",
    "orbit", "network", "structure", "blueprint", "explained", "works", "cause", "causes",
    "connected", "hidden room", "floor plan", "anatomy", "cycle", "before", "after",
}
HORROR_WORDS = {
    "horror", "haunted", "ghost", "shadow", "dark", "night", "creepy", "scary", "door",
    "wardrobe", "basement", "monster", "strange", "mystery", "secret", "whisper", "knock",
    "boy", "girl", "room", "discovered", "behind", "missing", "vanished", "disappear",
}
OBJECTS = {
    "door": "door", "wardrobe": "door", "room": "room", "house": "house", "home": "house",
    "boy": "person", "girl": "person", "person": "person", "man": "person", "woman": "person",
    "phone": "phone", "message": "phone", "text": "phone", "clock": "clock", "time": "clock",
    "eye": "eye", "watch": "eye", "book": "book", "letter": "book", "list": "book",
    "planet": "planet", "space": "planet", "moon": "planet", "earth": "planet", "mars": "planet",
    "brain": "brain", "mind": "brain", "heart": "heart", "fire": "fire", "flame": "fire",
    "water": "water", "ocean": "water", "sea": "water", "mountain": "mountain",
    "tree": "tree", "forest": "tree", "car": "car", "machine": "gear", "engine": "gear",
    "gear": "gear", "key": "key", "lock": "key", "skull": "skull", "death": "skull",
}
PALETTE = [
    ((12, 18, 34), (102, 214, 208), (244, 174, 102)),
    ((24, 20, 39), (180, 148, 255), (255, 112, 130)),
    ((16, 28, 36), (133, 220, 165), (246, 205, 108)),
    ((29, 22, 28), (255, 157, 119), (231, 221, 194)),
]


def _font(size: int):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def visual_type(query: str, image_prompt: str = "") -> str:
    """Choose diagram for explanatory/structural scenes; otherwise use illustration."""
    text = f"{query} {image_prompt}".lower()
    words = _words(text)
    diagram_hits = sum(1 for term in DIAGRAM_WORDS if term in text or term in words)
    # Stories with uncanny imagery should remain atmospheric drawings, not be forced into diagrams.
    story_hits = sum(1 for term in HORROR_WORDS if term in text or term in words)
    return "diagram" if diagram_hits >= 2 and diagram_hits >= story_hits else "illustration"


def _centered(draw: ImageDraw.ImageDraw, xy, text: str, font, fill):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def _draw_object(draw, kind: str, cx: int, cy: int, scale: float, ink, accent):
    """Simple bold line-art symbols that remain legible on a phone screen."""
    w = max(3, int(5 * scale))
    r = int(70 * scale)
    if kind == "door":
        box = (cx-r, cy-r*1.45, cx+r, cy+r*1.45)
        draw.rounded_rectangle(box, radius=int(8*scale), outline=accent, width=w)
        draw.line((cx-r+int(12*scale), cy-r*1.25, cx-r+int(12*scale), cy+r*1.25), fill=ink, width=max(2,w-1))
        draw.ellipse((cx+r-int(22*scale), cy, cx+r-int(10*scale), cy+int(12*scale)), fill=accent)
        draw.line((cx-r*1.3, cy+r*1.48, cx+r*1.3, cy+r*1.48), fill=ink, width=w)
    elif kind == "person":
        draw.ellipse((cx-int(28*scale), cy-r*1.45, cx+int(28*scale), cy-r*.9), outline=accent, width=w)
        draw.line((cx, cy-r*.85, cx, cy+r*.35), fill=accent, width=w)
        draw.line((cx, cy-r*.35, cx-r*.65, cy+r*.1), fill=ink, width=w)
        draw.line((cx, cy-r*.35, cx+r*.65, cy+r*.1), fill=ink, width=w)
        draw.line((cx, cy+r*.3, cx-r*.55, cy+r*1.1), fill=accent, width=w)
        draw.line((cx, cy+r*.3, cx+r*.55, cy+r*1.1), fill=accent, width=w)
    elif kind == "phone":
        draw.rounded_rectangle((cx-r*.62,cy-r*1.35,cx+r*.62,cy+r*1.35),radius=int(14*scale),outline=accent,width=w)
        draw.line((cx-r*.35,cy-r*.7,cx+r*.35,cy-r*.7),fill=ink,width=w)
        draw.rounded_rectangle((cx-r*.38,cy-r*.35,cx+r*.35,cy+r*.35),radius=int(8*scale),outline=ink,width=max(2,w-1))
        draw.ellipse((cx-4*scale,cy+r*.9,cx+4*scale,cy+r*.9+8*scale),fill=accent)
    elif kind == "clock":
        draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline=accent,width=w)
        draw.line((cx,cy,cx,cy-r*.62),fill=ink,width=w)
        draw.line((cx,cy,cx+r*.52,cy+r*.25),fill=accent,width=w)
        for a in range(12):
            ang=math.pi*a/6
            x1,y1=cx+math.sin(ang)*r*.82,cy-math.cos(ang)*r*.82
            x2,y2=cx+math.sin(ang)*r*.95,cy-math.cos(ang)*r*.95
            draw.line((x1,y1,x2,y2),fill=ink,width=max(2,w-1))
    elif kind in ("planet","eye","brain","heart","fire","water","mountain","tree","house","book","gear","key","skull"):
        if kind == "planet":
            draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline=accent,width=w)
            draw.arc((cx-r*1.5,cy-r*.45,cx+r*1.5,cy+r*.45),0,360,fill=ink,width=w)
            draw.ellipse((cx+r*.35,cy-r*.5,cx+r*.55,cy-r*.3),fill=accent)
        elif kind == "eye":
            draw.arc((cx-r,cy-r*.55,cx+r,cy+r*.55),180,360,fill=accent,width=w)
            draw.arc((cx-r,cy-r*.55,cx+r,cy+r*.55),0,180,fill=accent,width=w)
            draw.ellipse((cx-r*.28,cy-r*.28,cx+r*.28,cy+r*.28),outline=ink,width=w)
            draw.ellipse((cx-r*.1,cy-r*.1,cx+r*.1,cy+r*.1),fill=accent)
        elif kind == "brain":
            for ox,oy,rr in [(-.45,-.2,.45),(-.15,-.48,.43),(.25,-.4,.42),(.48,-.05,.4),(.2,.28,.43),(-.25,.3,.42)]:
                draw.ellipse((cx+(ox-rr)*r,cy+(oy-rr)*r,cx+(ox+rr)*r,cy+(oy+rr)*r),outline=accent,width=w)
            draw.line((cx,cy-r*.7,cx,cy+r*.65),fill=ink,width=max(2,w-1))
        elif kind == "heart":
            pts=[(cx,cy+r*.8),(cx-r*1.05,cy-r*.1),(cx-r*.9,cy-r*.65),(cx-r*.45,cy-r*.8),(cx,cy-r*.35),(cx+r*.45,cy-r*.8),(cx+r*.9,cy-r*.65),(cx+r*1.05,cy-r*.1),(cx,cy+r*.8)]
            draw.line(pts,fill=accent,width=w,joint="curve")
        elif kind == "fire":
            pts=[(cx,cy+r),(cx-r*.75,cy+r*.25),(cx-r*.45,cy-r*.25),(cx-r*.2,cy-r*.05),(cx-r*.05,cy-r),(cx+r*.55,cy-r*.2),(cx+r*.75,cy+r*.25),(cx,cy+r)]
            draw.line(pts,fill=accent,width=w,joint="curve")
            draw.line([(cx,cy+r*.55),(cx-r*.2,cy+r*.05),(cx+r*.15,cy-r*.25),(cx+r*.3,cy+r*.3),(cx,cy+r*.55)],fill=ink,width=max(2,w-1),joint="curve")
        elif kind == "water":
            for j in range(3):
                yy=cy+(j-1)*r*.42
                draw.arc((cx-r,yy-r*.3,cx+r,yy+r*.3),0,180,fill=accent if j%2==0 else ink,width=w)
        elif kind == "mountain":
            draw.line([(cx-r*1.2,cy+r*.75),(cx-r*.25,cy-r),(cx+r*.15,cy-r*.2),(cx+r*.55,cy-r*.7),(cx+r*1.2,cy+r*.75),(cx-r*1.2,cy+r*.75)],fill=accent,width=w,joint="curve")
        elif kind == "tree":
            draw.line((cx,cy-r*.3,cx,cy+r),fill=ink,width=w)
            for yy,ww in [(-.65,.55),(-.2,.8),(.2,.65)]:
                draw.polygon([(cx,cy+(yy-.55)*r),(cx-ww*r,cy+(yy+.25)*r),(cx+ww*r,cy+(yy+.25)*r)],outline=accent)
        elif kind == "house":
            draw.line([(cx-r,cy),(cx,cy-r),(cx+r,cy),(cx+r*.78,cy+r),(cx-r*.78,cy+r),(cx-r,cy)],fill=accent,width=w,joint="curve")
            draw.rectangle((cx-r*.2,cy+r*.25,cx+r*.2,cy+r),outline=ink,width=max(2,w-1))
        elif kind == "book":
            draw.line([(cx,cy-r),(cx-r,cy-r*.75),(cx-r,cy+r),(cx,cy+r*.7),(cx+r,cy+r),(cx+r,cy-r*.75),(cx,cy-r)],fill=accent,width=w,joint="curve")
            draw.line((cx,cy-r,cx,cy+r*.7),fill=ink,width=max(2,w-1))
        elif kind == "gear":
            draw.ellipse((cx-r*.7,cy-r*.7,cx+r*.7,cy+r*.7),outline=accent,width=w)
            draw.ellipse((cx-r*.25,cy-r*.25,cx+r*.25,cy+r*.25),outline=ink,width=w)
            for a in range(8):
                ang=math.pi*a/4
                draw.line((cx+math.cos(ang)*r*.72,cy+math.sin(ang)*r*.72,cx+math.cos(ang)*r,cy+math.sin(ang)*r),fill=accent,width=w)
        elif kind == "key":
            draw.ellipse((cx-r,cy-r*.55,cx,cy+r*.45),outline=accent,width=w)
            draw.line((cx-r*.05,cy,cx+r*1.15,cy),fill=accent,width=w)
            draw.line((cx+r*.65,cy,cx+r*.65,cy+r*.35),fill=ink,width=w)
            draw.line((cx+r*.95,cy,cx+r*.95,cy+r*.25),fill=ink,width=w)
        else:
            draw.ellipse((cx-r*.75,cy-r*.9,cx+r*.75,cy+r*.9),outline=accent,width=w)
            draw.ellipse((cx-r*.4,cy-r*.25,cx-r*.12,cy+r*.03),outline=ink,width=w)
            draw.ellipse((cx+r*.12,cy-r*.25,cx+r*.4,cy+r*.03),outline=ink,width=w)
            draw.arc((cx-r*.35,cy+r*.1,cx+r*.35,cy+r*.55),0,180,fill=ink,width=w)


def _diagram(im: Image.Image, draw, query: str, ink, accent, second):
    w,h=im.size
    margin=int(w*.12)
    title = "HOW IT CONNECTS" if any(x in query.lower() for x in ("connect","network","system")) else "INSIDE THE MYSTERY"
    draw.rounded_rectangle((margin,int(h*.14),w-margin,int(h*.86)),radius=28,outline=accent,width=max(3,w//180))
    # Three clean nodes and directional links, legible as a vertical explainer.
    node_w=int(w*.36); node_h=int(h*.105)
    centers=[int(h*.29),int(h*.50),int(h*.71)]
    labels=["TRIGGER","HIDDEN LINK","RESULT"]
    for i,cy in enumerate(centers):
        x0=(w-node_w)//2; y0=cy-node_h//2
        draw.rounded_rectangle((x0,y0,x0+node_w,y0+node_h),radius=18,
                               fill=(24,31,49),outline=accent if i!=1 else second,width=max(3,w//240))
        _centered(draw,(w//2,y0+node_h*.28),labels[i],_font(max(16,w//34)),ink)
        if i<2:
            x=w//2; y1=y0+node_h+8; y2=centers[i+1]-node_h//2-8
            draw.line((x,y1,x,y2),fill=second,width=max(3,w//220))
            draw.polygon([(x,y2),(x-12,y2-20),(x+12,y2-20)],fill=second)
    # Tiny side marks make it feel like a designed technical illustration.
    for side in (margin+28,w-margin-28):
        draw.line((side,int(h*.24),side,int(h*.76)),fill=(60,76,100),width=max(2,w//360))


def make_visual(query: str, image_prompt: str, destination: Path, width: int, height: int) -> str:
    """Write a scene-specific non-photographic PNG and return 'diagram' or 'illustration'."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.sha1((query + "|" + image_prompt).encode("utf-8")).hexdigest()[:8], 16)
    bg, accent, second = PALETTE[seed % len(PALETTE)]
    im = Image.new("RGB", (max(360, width), max(640, height)), bg)
    draw = ImageDraw.Draw(im)
    w,h=im.size
    # Subtle illustrated atmosphere: glows, specks, and a framed composition.
    for i in range(34):
        x=(seed*(i+7)*31) % w
        y=(seed*(i+13)*17) % h
        rr=2+(i%4)
        draw.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(34,43,62))
    draw.rounded_rectangle((int(w*.055),int(h*.045),int(w*.945),int(h*.955)),
                           radius=28,outline=(61,72,94),width=max(2,w//400))
    kind=visual_type(query,image_prompt)
    if kind == "diagram":
        _diagram(im,draw,query,(235,239,248),accent,second)
    else:
        text=f"{query} {image_prompt}".lower()
        words=_words(text)
        obj=next((v for k,v in OBJECTS.items() if k in words), "person" if any(x in words for x in ("someone","stranger","figure","creature")) else "eye")
        cx,cy=w//2,int(h*.46)
        # Floor/ground plane and a cinematic pool of light.
        draw.ellipse((int(w*.16),int(h*.68),int(w*.84),int(h*.83)),outline=(49,60,82),width=max(2,w//260))
        _draw_object(draw,obj,cx,cy,min(w/720,h/1280)*1.65,(235,239,248),accent)
        # Small framing lines and a second, smaller symbol make the image read as authored art.
        draw.line((int(w*.16),int(h*.2),int(w*.29),int(h*.2)),fill=second,width=max(3,w//220))
        draw.line((int(w*.71),int(h*.76),int(w*.84),int(h*.76)),fill=second,width=max(3,w//220))
        draw.ellipse((int(w*.76),int(h*.2),int(w*.79),int(h*.2)+int(w*.03)),outline=accent,width=max(2,w//300))
        if any(x in text for x in HORROR_WORDS):
            # An ominous shadow figure behind the main symbol, intentionally symbolic.
            sx,sy=int(w*.24),int(h*.62)
            draw.ellipse((sx-18,sy-35,sx+18,sy+1),fill=(8,12,24))
            draw.polygon([(sx-28,sy+65),(sx-16,sy-2),(sx+16,sy-2),(sx+28,sy+65)],fill=(8,12,24))
    im.save(destination, format="PNG", optimize=True)
    return kind
