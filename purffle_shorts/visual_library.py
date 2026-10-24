"""Offline vector-style illustration and diagram library for Purffle Shorts.

All artwork is drawn with Pillow: no stock-photo dependency, network calls, or extra packages.
Scene-specific compositions are selected from narration/search prompts and rendered as portrait PNGs.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DIAGRAM_PHRASES = (
    "diagram", "flowchart", "how it works", "how does", "how the", "mechanism",
    "process", "step by step", "steps of", "cause and effect", "timeline",
    "cross section", "cross-section", "floor plan", "map of", "route of",
    "orbit", "food chain", "life cycle", "network", "system architecture",
    "anatomy", "structure of", "inside the engine", "explained",
)
PALETTES = [
    ((11, 17, 32), (91, 220, 211), (255, 174, 105), (235, 239, 248)),
    ((19, 16, 35), (179, 143, 255), (255, 104, 133), (244, 240, 255)),
    ((13, 27, 34), (124, 222, 169), (255, 207, 105), (235, 246, 242)),
    ((30, 19, 25), (255, 147, 113), (160, 190, 255), (255, 238, 222)),
]
STOP = {"the","and","for","with","from","that","this","into","your","you","was","were","are","his","her","they","their","then","when","where","what","why","how","someone","something","very","just","real","photo","image","illustration","cinematic","drawing","style","vertical","portrait"}
HORROR = {"horror","haunted","ghost","shadow","dark","night","creepy","scary","door","wardrobe","basement","monster","mystery","secret","whisper","knock","boy","girl","room","discovered","behind","missing","vanished","disappear","stranger","haunting","eerie","attic","corridor"}
OBJECTS = {
    "wardrobe":"wardrobe", "closet":"wardrobe", "cupboard":"wardrobe",
    "door":"door", "room":"room", "bedroom":"bedroom", "bed":"bedroom",
    "boy":"person", "girl":"person", "person":"person", "man":"person", "woman":"person",
    "ghost":"ghost", "shadow":"ghost", "monster":"ghost", "creature":"ghost",
    "house":"house", "phone":"phone", "message":"phone", "clock":"clock", "time":"clock",
    "eye":"eye", "watching":"eye", "letter":"letter", "list":"letter", "book":"letter",
    "key":"key", "lock":"key", "window":"window", "mirror":"mirror", "stairs":"stairs",
    "forest":"forest", "tree":"forest", "mountain":"mountain", "ocean":"ocean", "sea":"ocean",
    "planet":"planet", "moon":"planet", "mars":"planet", "earth":"planet", "space":"planet",
    "brain":"brain", "heart":"heart", "fire":"fire", "flame":"fire", "water":"ocean",
    "car":"car", "engine":"gear", "machine":"gear", "gear":"gear", "skull":"skull",
    "stairs":"stairs", "staircase":"stairs", "city":"city", "building":"city", "street":"city",
    "laboratory":"lab", "lab":"lab", "science":"lab", "microscope":"microscope",
    "atom":"atom", "electricity":"circuit", "circuit":"circuit", "computer":"computer",
    "robot":"robot", "airplane":"airplane", "plane":"airplane", "train":"train", "railway":"train",
    "bridge":"bridge", "cave":"cave", "tunnel":"cave", "storm":"storm", "lightning":"storm",
    "rain":"storm", "cloud":"storm", "dog":"animal", "cat":"animal", "bird":"animal", "wolf":"animal",
    "fish":"animal", "snake":"animal", "dinosaur":"animal", "insect":"animal",
    "dna":"dna", "cell":"cell", "bacteria":"cell", "virus":"cell", "plant":"plant", "flower":"plant",
    "book":"book", "document":"letter", "map":"map", "compass":"compass", "rocket":"rocket",
    "satellite":"satellite", "volcano":"volcano", "desert":"desert", "snow":"snow",
}

def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))

def visual_type(query: str, image_prompt: str = "") -> str:
    """Prefer diagrams only when the wording explicitly asks to explain structure/process."""
    text = f"{query} {image_prompt}".lower().replace("_", " ")
    if any(p in text for p in DIAGRAM_PHRASES):
        # A story prompt that happens to say 'inside' should still be illustrated.
        if not (_words(text) & HORROR and not any(p in text for p in ("diagram", "flowchart", "how it works", "step by step", "cause and effect", "anatomy", "timeline"))):
            return "diagram"
    return "illustration"


def character_kind(text: str) -> str:
    """Stable, reusable character archetype inferred from the explicit character brief."""
    words = _words(text)
    if words & {"narrator", "hooded", "mascot", "host", "purffle"}:
        return "hooded_narrator"
    if words & {"scientist", "researcher", "laboratory"}:
        return "scientist"
    if words & {"detective", "investigator", "sleuth"}:
        return "detective"
    if words & {"explorer", "astronaut", "adventurer"}:
        return "explorer"
    if words & {"ghost", "creature", "monster", "shadow"}:
        return "mysterious_figure"
    if words & {"boy", "girl", "person", "character", "witness", "stranger"}:
        return "human_silhouette"
    return "symbolic_mascot"


def _character(draw, w: int, h: int, ink, accent, second, seed: int, archetype: str) -> None:
    """Draw a consistent, simple recurring character silhouette with pose variation."""
    cx, cy = w // 2, int(h * .48)
    scale = min(w * .56, h * .42)
    line_w = max(5, w // 150)
    # Identity stays stable across poses: hood shape, face glow, eye spacing and accent trim.
    if archetype in ("hooded_narrator", "mysterious_figure", "detective"):
        head_r = scale * .23
        hood = [(cx, cy-scale*.64), (cx-scale*.28, cy-scale*.35), (cx-scale*.24, cy+scale*.06),
                (cx-scale*.15, cy+scale*.18), (cx+scale*.15, cy+scale*.18),
                (cx+scale*.24, cy+scale*.06), (cx+scale*.28, cy-scale*.35)]
        draw.polygon(hood, fill=(7, 10, 20), outline=accent)
        draw.ellipse((cx-head_r, cy-scale*.43, cx+head_r, cy-scale*.03),
                     fill=(12, 17, 30), outline=second, width=line_w)
        eye_y = cy-scale*.24
        eye_dx = scale*.09
        glow = accent if archetype != "detective" else second
        draw.ellipse((cx-eye_dx-scale*.035, eye_y-scale*.02, cx-eye_dx+scale*.035, eye_y+scale*.035), fill=glow)
        draw.ellipse((cx+eye_dx-scale*.035, eye_y-scale*.02, cx+eye_dx+scale*.035, eye_y+scale*.035), fill=glow)
        torso = [(cx-scale*.16, cy+scale*.1), (cx-scale*.36, cy+scale*.55),
                 (cx-scale*.29, cy+scale*.72), (cx+scale*.29, cy+scale*.72),
                 (cx+scale*.36, cy+scale*.55), (cx+scale*.16, cy+scale*.1)]
        draw.polygon(torso, fill=(10, 15, 27), outline=accent)
        # Repeatable left/right gesture poses.
        pose = seed % 3
        arm_y = cy + scale*(.25 if pose == 1 else .42)
        if pose == 0:
            _line(draw, [(cx-scale*.25, arm_y), (cx-scale*.43, cy+scale*.14),
                         (cx-scale*.48, cy-scale*.02)], second, line_w)
            _line(draw, [(cx+scale*.25, arm_y), (cx+scale*.43, cy+scale*.14),
                         (cx+scale*.48, cy-scale*.02)], accent, line_w)
        elif pose == 1:
            _line(draw, [(cx-scale*.24, arm_y), (cx-scale*.42, cy+scale*.5)], second, line_w)
            _line(draw, [(cx+scale*.24, arm_y), (cx+scale*.47, cy+scale*.03),
                         (cx+scale*.61, cy-scale*.04)], accent, line_w)
        else:
            _line(draw, [(cx-scale*.24, arm_y), (cx-scale*.48, cy+scale*.02),
                         (cx-scale*.59, cy-scale*.02)], accent, line_w)
            _line(draw, [(cx+scale*.24, arm_y), (cx+scale*.42, cy+scale*.5)], second, line_w)
    else:
        # General character silhouette; same body language and visual grammar across topics.
        _person(draw, cx, int(cy+scale*.12), scale*.85, ink, accent, frightened=False)
        draw.ellipse((cx-scale*.22,cy-scale*.62,cx+scale*.22,cy-scale*.18),outline=second,width=line_w)
    _glow(draw, cx, int(cy-scale*.18), int(scale*.38), second)


def _font(size: int):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, max(10, size))
        except OSError:
            pass
    return ImageFont.load_default()

def _center(draw, xy, text, font, fill):
    x, y = xy
    b = draw.textbbox((0, 0), text, font=font)
    draw.text((x-(b[2]-b[0])/2, y), text, font=font, fill=fill)

def _line(draw, points, color, width):
    draw.line(points, fill=color, width=max(2, width), joint="curve")

def _glow(draw, cx, cy, radius, color):
    # Layered rings create a soft graphic halo without expensive blur filters.
    for n in range(5, 0, -1):
        rr = int(radius*n/5)
        draw.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=color, width=max(1, radius//18))

def _person(draw, cx, cy, s, ink, accent, frightened=False):
    # Readable character silhouette with head, shoulders, torso, arms and legs.
    r=max(12,int(s*.12)); w=max(4,int(s*.045))
    draw.ellipse((cx-r,cy-s*.48-r,cx+r,cy-s*.48+r), fill=(8,12,24), outline=accent, width=w)
    draw.rounded_rectangle((cx-s*.17,cy-s*.31,cx+s*.17,cy+s*.15), radius=int(s*.08), fill=(8,12,24), outline=accent, width=w)
    _line(draw,[(cx-s*.12,cy-s*.23),(cx-s*.32,cy-s*.02),(cx-s*.24,cy+s*.08)],ink,w)
    _line(draw,[(cx+s*.12,cy-s*.23),(cx+s*.32,cy-s*.02),(cx+s*.24,cy+s*.08)],accent,w)
    _line(draw,[(cx-s*.08,cy+s*.14),(cx-s*.22,cy+s*.43)],accent,w)
    _line(draw,[(cx+s*.08,cy+s*.14),(cx+s*.22,cy+s*.43)],accent,w)
    if frightened:
        draw.ellipse((cx-r*.45,cy-s*.48-r*.1,cx-r*.12,cy-s*.48+r*.18),fill=ink)
        draw.ellipse((cx+r*.12,cy-s*.48-r*.1,cx+r*.45,cy-s*.48+r*.18),fill=ink)

def _wardrobe(draw, cx, cy, s, ink, accent, second, open_door=False):
    w=max(4,int(s*.035)); left=cx-s*.34; right=cx+s*.34; top=cy-s*.48; bottom=cy+s*.43
    draw.rounded_rectangle((left,top,right,bottom),radius=int(s*.025),fill=(24,27,43),outline=accent,width=w)
    mid=cx
    draw.line((mid,top+8,mid,bottom-8),fill=ink,width=max(3,w-1))
    draw.rectangle((left+s*.07,top+s*.1,left+s*.12,bottom-s*.1),outline=second,width=max(2,w//2))
    draw.rectangle((right-s*.12,top+s*.1,right-s*.07,bottom-s*.1),outline=second,width=max(2,w//2))
    draw.ellipse((mid-s*.045,cy,mid-s*.015,cy+s*.03),fill=accent)
    if open_door:
        # Perspective slab opens toward viewer; the dark impossible space is visible behind it.
        draw.polygon([(mid,top),(mid+s*.48,top+s*.13),(mid+s*.48,bottom-s*.06),(mid,bottom)],fill=(8,10,22),outline=second)
        draw.ellipse((mid+s*.12,cy-s*.2,mid+s*.35,cy+s*.15),outline=accent,width=w)

def _bedroom(draw, w, h, ink, accent, second):
    # Perspective room, floorboards, window and bed: a scene, not an isolated pictogram.
    top=int(h*.2); horizon=int(h*.64)
    _line(draw,[(w*.1,top),(w*.9,top),(w*.9,horizon),(w*.1,horizon),(w*.1,top)],(68,76,101),max(3,w//240))
    _line(draw,[(w*.1,top),(w*.5,h*.34),(w*.9,top)],(68,76,101),max(3,w//240))
    draw.rectangle((w*.68,h*.28,w*.82,h*.48),fill=(16,25,47),outline=second,width=max(3,w//240))
    _line(draw,[(w*.75,h*.28),(w*.75,h*.48)],second,max(2,w//300))
    _line(draw,[(w*.68,h*.38),(w*.82,h*.38)],second,max(2,w//300))
    # bed in the foreground
    draw.polygon([(w*.16,h*.61),(w*.68,h*.61),(w*.82,h*.75),(w*.23,h*.75)],fill=(31,37,58),outline=accent)
    draw.polygon([(w*.23,h*.57),(w*.65,h*.57),(w*.68,h*.61),(w*.16,h*.61)],fill=(68,76,105),outline=accent)
    _line(draw,[(w*.23,h*.75),(w*.21,h*.82)],ink,max(4,w//180))
    _line(draw,[(w*.77,h*.75),(w*.79,h*.82)],ink,max(4,w//180))

def _illustration(im, draw, text, ink, accent, second, seed):
    w,h=im.size; words=_words(text)
    ordered_words=re.findall(r"[a-z]+", text.lower())
    obj=next((OBJECTS[token] for token in ordered_words if token in OBJECTS), "person" if words & {"someone","stranger","figure","waiting","knocked"} else "eye")
    horror=bool(words & HORROR)
    # Narrative-aware compositions.
    if obj in ("wardrobe","door","bedroom","room") or (horror and words & {"behind","knock","sealed","opened"}):
        _bedroom(draw,w,h,ink,accent,second)
        if obj == "wardrobe" or words & {"wardrobe","closet","cupboard"}:
            _wardrobe(draw,int(w*.49),int(h*.51),min(w*.57,h*.54),ink,accent,second,
                      bool(words & {"open","opened","inside","crack","secret","revealed"}))
        elif obj == "door" or words & {"door","sealed","knock"}:
            # Large door framed in a shadowy hallway.
            draw.polygon([(w*.25,h*.22),(w*.62,h*.18),(w*.62,h*.65),(w*.25,h*.7)],fill=(25,28,45),outline=accent)
            _line(draw,[(w*.25,h*.22),(w*.62,h*.18),(w*.62,h*.65),(w*.25,h*.7),(w*.25,h*.22)],accent,max(4,w//180))
            draw.ellipse((w*.54,h*.43,w*.565,h*.46),fill=second)
            if words & {"knock","knocks","tapping","tapped"}:
                for rr in (w*.05,w*.09,w*.13):
                    draw.arc((w*.43-rr,h*.36-rr,w*.43+rr,h*.36+rr),300,60,fill=second,width=max(3,w//260))
        if words & {"boy","girl","person","someone","figure","standing"}:
            _person(draw,int(w*.72),int(h*.62),min(w*.22,h*.26),ink,second,True)
    elif obj in ("person","ghost"):
        # Figure silhouetted against a doorway/light source, with atmospheric depth.
        _glow(draw,int(w*.52),int(h*.39),int(w*.22),second)
        draw.polygon([(w*.36,h*.73),(w*.36,h*.28),(w*.64,h*.28),(w*.64,h*.73)],fill=(7,10,21),outline=(61,72,94))
        if obj=="ghost":
            draw.ellipse((w*.40,h*.26,w*.60,h*.47),fill=(8,12,24),outline=accent,width=max(4,w//180))
            draw.polygon([(w*.40,h*.39),(w*.60,h*.39),(w*.57,h*.62),(w*.51,h*.56),(w*.46,h*.64)],fill=(8,12,24),outline=accent)
            draw.ellipse((w*.445,h*.36,w*.47,h*.39),fill=ink); draw.ellipse((w*.53,h*.36,w*.555,h*.39),fill=ink)
        else:
            _person(draw,w//2,int(h*.53),min(w*.48,h*.48),ink,accent,True)
    elif obj in ("window","mirror"):
        draw.rounded_rectangle((w*.22,h*.22,w*.78,h*.67),radius=10,fill=(15,25,48),outline=accent,width=max(5,w//150))
        _line(draw,[(w*.5,h*.23),(w*.5,h*.66)],second,max(3,w//220))
        _line(draw,[(w*.23,h*.45),(w*.77,h*.45)],second,max(3,w//220))
        if obj=="mirror":
            _person(draw,w//2,int(h*.48),min(w*.30,h*.30),ink,second,True)
    elif obj=="stairs":
        for i in range(7):
            x0=w*(.2+i*.075); y0=h*(.66-i*.055)
            _line(draw,[(x0,y0),(w*.76,y0)],accent,max(3,w//200))
            _line(draw,[(x0,y0),(x0,y0+h*.055)],second,max(3,w//200))
        _glow(draw,int(w*.69),int(h*.27),int(w*.11),second)
        draw.polygon([(w*.59,h*.45),(w*.59,h*.23),(w*.79,h*.23),(w*.79,h*.45)],fill=(8,12,24),outline=accent)
    elif obj in ("city","lab","microscope","atom","circuit","computer","robot","airplane","train","bridge","cave","storm","animal","dna","cell","plant","map","compass","rocket","satellite","volcano","desert","snow"):
        _symbol(draw,obj,w//2,int(h*.46),min(w*.46,h*.40),ink,accent,second)
        # Add a small secondary element to make the scene feel composed, not like a lone icon.
        _line(draw,[(w*.16,h*.72),(w*.84,h*.72)],(63,74,96),max(3,w//240))
        if obj in ("city","train","airplane","rocket","satellite"):
            for i in range(5):
                x=w*(.16+i*.16); y=h*(.76+(i%2)*.035)
                draw.ellipse((x,y,x+w*.018,y+w*.018),fill=second)
    else:
        cx,cy=w//2,int(h*.46); s=min(w*.43,h*.42); sw=max(4,w//180)
        if obj=="phone":
            draw.rounded_rectangle((cx-s*.42,cy-s*.75,cx+s*.42,cy+s*.75),radius=int(s*.12),fill=(22,29,47),outline=accent,width=sw)
            draw.rounded_rectangle((cx-s*.29,cy-s*.49,cx+s*.29,cy+s*.39),radius=8,fill=(11,18,34),outline=second,width=max(2,sw//2))
            draw.line((cx-s*.17,cy-s*.28,cx+s*.17,cy-s*.28),fill=ink,width=sw)
            draw.line((cx-s*.17,cy-s*.1,cx+s*.12,cy-s*.1),fill=accent,width=sw)
        elif obj=="letter":
            draw.polygon([(cx-s*.62,cy-s*.42),(cx+s*.62,cy-s*.42),(cx+s*.62,cy+s*.42),(cx-s*.62,cy+s*.42)],fill=(30,33,49),outline=accent)
            _line(draw,[(cx-s*.62,cy-s*.42),(cx,cy+s*.06),(cx+s*.62,cy-s*.42)],second,sw)
            for yy in (-.18,.02,.22): _line(draw,[(cx-s*.35,cy+yy*s),(cx+s*.32,cy+yy*s)],ink,max(2,sw//2))
        elif obj in ("key","gear","clock","eye","planet","brain","heart","fire","ocean","mountain","forest","house","car","skull"):
            _symbol(draw,obj,cx,cy,s,ink,accent,second)
        else:
            _symbol(draw,"eye",cx,cy,s,ink,accent,second)
    # Film-frame labels are deliberately avoided: generated story art remains text-free.
    # Foreground vignette and hand-drawn marks tie assets together visually.
    draw.ellipse((w*.17,h*.78,w*.83,h*.84),outline=(48,59,81),width=max(2,w//280))
    for i in range(12):
        x=(seed*(i+5)*31)%w; y=(seed*(i+9)*17)%int(h*.78)
        r=2+i%3; draw.ellipse((x-r,y-r,x+r,y+r),fill=(37,46,66))

def _symbol(draw, kind, cx, cy, s, ink, accent, second):
    r=s*.42; lw=max(4,int(s*.035))
    if kind=="planet":
        draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline=accent,width=lw)
        draw.arc((cx-r*1.45,cy-r*.42,cx+r*1.45,cy+r*.42),0,360,fill=second,width=lw)
        draw.ellipse((cx+r*.25,cy-r*.35,cx+r*.42,cy-r*.18),fill=ink)
    elif kind=="clock":
        draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(23,30,48),outline=accent,width=lw)
        _line(draw,[(cx,cy),(cx,cy-r*.55)],ink,lw); _line(draw,[(cx,cy),(cx+r*.45,cy+r*.22)],second,lw)
    elif kind=="eye":
        draw.arc((cx-r,cy-r*.48,cx+r,cy+r*.48),180,360,fill=accent,width=lw)
        draw.arc((cx-r,cy-r*.48,cx+r,cy+r*.48),0,180,fill=accent,width=lw)
        draw.ellipse((cx-r*.23,cy-r*.23,cx+r*.23,cy+r*.23),fill=second)
        draw.ellipse((cx-r*.08,cy-r*.08,cx+r*.08,cy+r*.08),fill=ink)
    elif kind=="gear":
        draw.ellipse((cx-r*.72,cy-r*.72,cx+r*.72,cy+r*.72),outline=accent,width=lw)
        draw.ellipse((cx-r*.24,cy-r*.24,cx+r*.24,cy+r*.24),outline=ink,width=lw)
        for a in range(8):
            ang=a*math.pi/4; _line(draw,[(cx+math.cos(ang)*r*.74,cy+math.sin(ang)*r*.74),(cx+math.cos(ang)*r,cy+math.sin(ang)*r)],second,lw)
    elif kind=="heart":
        pts=[(cx,cy+r),(cx-r,cy),(cx-r*.9,cy-r*.55),(cx-r*.4,cy-r*.75),(cx,cy-r*.35),(cx+r*.4,cy-r*.75),(cx+r*.9,cy-r*.55),(cx+r,cy),(cx,cy+r)]
        _line(draw,pts,accent,lw)
    elif kind=="fire":
        _line(draw,[(cx,cy+r),(cx-r*.7,cy+r*.2),(cx-r*.45,cy-r*.35),(cx-r*.1,cy-r*.05),(cx,cy-r),(cx+r*.55,cy-r*.15),(cx+r*.7,cy+r*.25),(cx,cy+r)],accent,lw)
        _line(draw,[(cx,cy+r*.6),(cx-r*.2,cy+r*.12),(cx+r*.2,cy-r*.28),(cx+r*.3,cy+r*.3),(cx,cy+r*.6)],second,lw)
    elif kind in ("ocean","mountain","forest","house","car","skull","brain","key"):
        if kind=="ocean":
            for j in range(4):
                yy=cy+(j-1.5)*r*.4; draw.arc((cx-r,yy-r*.25,cx+r,yy+r*.25),0,180,fill=accent if j%2==0 else second,width=lw)
        elif kind=="mountain":
            _line(draw,[(cx-r,cy+r*.65),(cx-r*.25,cy-r),(cx+r*.12,cy-r*.15),(cx+r*.48,cy-r*.62),(cx+r,cy+r*.65),(cx-r,cy+r*.65)],accent,lw)
            _line(draw,[(cx-r*.35,cy-r*.72),(cx-r*.17,cy-r*.42),(cx-r*.02,cy-r*.54)],ink,lw)
        elif kind=="forest":
            for dx,sc in [(-.5,.65),(0,1),(.48,.7)]:
                draw.polygon([(cx+dx*r,cy-r*sc),(cx+(dx-.43*sc)*r,cy+r*.15),(cx+(dx+.43*sc)*r,cy+r*.15)],fill=(20,43,48),outline=accent)
                _line(draw,[(cx+dx*r,cy+r*.1),(cx+dx*r,cy+r*.7)],second,lw)
        elif kind=="house":
            _line(draw,[(cx-r,cy),(cx,cy-r*.8),(cx+r,cy),(cx+r*.78,cy+r*.8),(cx-r*.78,cy+r*.8),(cx-r,cy)],accent,lw)
            draw.rectangle((cx-r*.18,cy+r*.28,cx+r*.18,cy+r*.8),outline=second,width=lw)
        elif kind=="car":
            draw.rounded_rectangle((cx-r,cy-r*.25,cx+r,cy+r*.35),radius=int(r*.18),outline=accent,width=lw)
            draw.polygon([(cx-r*.55,cy-r*.25),(cx-r*.25,cy-r*.65),(cx+r*.35,cy-r*.65),(cx+r*.65,cy-r*.25)],outline=second)
            for dx in (-.55,.55): draw.ellipse((cx+dx*r-r*.18,cy+r*.18,cx+dx*r+r*.18,cy+r*.55),fill=ink,outline=accent,width=lw)
        elif kind=="key":
            draw.ellipse((cx-r,cy-r*.5,cx,cy+r*.5),outline=accent,width=lw); _line(draw,[(cx,cy),(cx+r,cy)],accent,lw)
            _line(draw,[(cx+r*.55,cy),(cx+r*.55,cy+r*.3)],second,lw); _line(draw,[(cx+r*.82,cy),(cx+r*.82,cy+r*.23)],second,lw)
        elif kind=="stairs":
            for i in range(6):
                x=cx-r+i*r*.28; y=cy+r-i*r*.28
                _line(draw,[(x,y),(cx+r,y)],accent,lw)
                _line(draw,[(x,y),(x,y-r*.28)],second,lw)
        elif kind=="city":
            for i,(dx,bh) in enumerate([(-.85,.55),(-.48,.9),(-.08,.68),(.32,1),(.68,.62)]):
                x=cx+dx*r; bw=r*.29; top=cy+r*.55-bh*r
                draw.rectangle((x,top,x+bw,cy+r*.55),fill=(20,28,46),outline=accent,width=max(2,lw//2))
                for wy in range(3):
                    for wx in range(2):
                        xx=x+bw*(.22+wx*.42); yy=top+bh*r*(.18+wy*.22)
                        draw.rectangle((xx,yy,xx+bw*.1,yy+bw*.12),fill=second)
        elif kind=="lab":
            _line(draw,[(cx-r*.35,cy-r),(cx+r*.35,cy-r)],accent,lw)
            _line(draw,[(cx-r*.2,cy-r),(cx-r*.2,cy-r*.25),(cx-r*.65,cy+r*.55),(cx-r*.5,cy+r*.8),(cx+r*.5,cy+r*.8),(cx+r*.65,cy+r*.55),(cx+r*.2,cy-r*.25),(cx+r*.2,cy-r)],accent,lw)
            draw.polygon([(cx-r*.52,cy+r*.35),(cx+r*.52,cy+r*.35),(cx+r*.42,cy+r*.68),(cx-r*.42,cy+r*.68)],fill=(20,91,94))
            for dx,dy in [(-.2,.5),(.1,.43),(.25,.58)]: draw.ellipse((cx+dx*r-r*.06,cy+dy*r-r*.06,cx+dx*r+r*.06,cy+dy*r+r*.06),fill=second)
        elif kind=="microscope":
            _line(draw,[(cx-r*.4,cy-r*.8),(cx+r*.05,cy-r*.8),(cx+r*.25,cy-r*.35),(cx+r*.05,cy-r*.2)],accent,lw)
            _line(draw,[(cx+r*.05,cy-r*.2),(cx-r*.25,cy+r*.05),(cx-r*.18,cy+r*.45),(cx+r*.25,cy+r*.45)],second,lw)
            _line(draw,[(cx-r*.5,cy+r*.65),(cx+r*.55,cy+r*.65)],accent,lw)
        elif kind=="atom":
            draw.ellipse((cx-r,cy-r*.38,cx+r,cy+r*.38),outline=accent,width=lw)
            draw.ellipse((cx-r*.38,cy-r,cx+r*.38,cy+r),outline=second,width=lw)
            draw.ellipse((cx-r*.76,cy-r*.76,cx+r*.76,cy+r*.76),outline=accent,width=max(2,lw//2))
            draw.ellipse((cx-r*.12,cy-r*.12,cx+r*.12,cy+r*.12),fill=ink)
        elif kind=="circuit":
            for yy in (-.55,0,.55):
                _line(draw,[(cx-r,cy+yy*r),(cx-r*.35,cy+yy*r),(cx-r*.1,cy+yy*r*.4),(cx+r*.2,cy+yy*r*.4),(cx+r*.45,cy+yy*r),(cx+r,cy+yy*r)],accent,lw)
                for xx in (-.72,.7): draw.ellipse((cx+xx*r-r*.08,cy+yy*r-r*.08,cx+xx*r+r*.08,cy+yy*r+r*.08),fill=second)
        elif kind=="computer":
            draw.rounded_rectangle((cx-r,cy-r*.75,cx+r,cy+r*.45),radius=int(r*.08),outline=accent,width=lw)
            _line(draw,[(cx,cy+r*.45),(cx,cy+r*.75),(cx-r*.4,cy+r*.75),(cx+r*.4,cy+r*.75)],second,lw)
            draw.rectangle((cx-r*.75,cy-r*.55,cx+r*.75,cy+r*.25),fill=(17,34,49))
        elif kind=="robot":
            draw.rounded_rectangle((cx-r*.62,cy-r*.75,cx+r*.62,cy+r*.05),radius=int(r*.12),outline=accent,width=lw)
            draw.ellipse((cx-r*.35,cy-r*.45,cx-r*.12,cy-r*.22),fill=second); draw.ellipse((cx+r*.12,cy-r*.45,cx+r*.35,cy-r*.22),fill=second)
            _line(draw,[(cx,cy-r*.75),(cx,cy-r),(cx+r*.22,cy-r)],accent,lw)
            draw.rounded_rectangle((cx-r*.45,cy+r*.12,cx+r*.45,cy+r*.68),radius=int(r*.1),outline=second,width=lw)
            _line(draw,[(cx-r*.45,cy+r*.22),(cx-r*.8,cy+r*.55)],accent,lw); _line(draw,[(cx+r*.45,cy+r*.22),(cx+r*.8,cy+r*.55)],accent,lw)
        elif kind=="airplane":
            _line(draw,[(cx-r,cy),(cx+r,cy)],accent,lw)
            _line(draw,[(cx-r*.1,cy),(cx-r*.35,cy-r*.65),(cx+r*.05,cy-r*.25),(cx+r*.45,cy-r*.2),(cx+r*.8,cy-r*.4),(cx+r*.7,cy),(cx+r*.8,cy+r*.4),(cx+r*.45,cy+r*.2),(cx+r*.05,cy+r*.25),(cx-r*.35,cy+r*.65),(cx-r*.1,cy)],second,lw)
        elif kind=="train":
            draw.rounded_rectangle((cx-r*.75,cy-r*.65,cx+r*.75,cy+r*.48),radius=int(r*.12),outline=accent,width=lw)
            for dx in (-.45,0,.45): draw.rectangle((cx+dx*r-r*.13,cy-r*.4,cx+dx*r+r*.13,cy-r*.08),outline=second,width=max(2,lw//2))
            for dx in (-.45,.45): draw.ellipse((cx+dx*r-r*.15,cy+r*.35,cx+dx*r+r*.15,cy+r*.65),fill=ink,outline=accent,width=max(2,lw//2))
        elif kind=="bridge":
            _line(draw,[(cx-r,cy+r*.5),(cx+r,cy+r*.5)],accent,lw)
            _line(draw,[(cx-r*.75,cy-r*.4),(cx-r*.75,cy+r*.5)],second,lw); _line(draw,[(cx+r*.75,cy-r*.4),(cx+r*.75,cy+r*.5)],second,lw)
            draw.arc((cx-r*.75,cy-r*.65,cx+r*.75,cy+r*.75),180,360,fill=accent,width=lw)
            for dx in (-.5,-.25,0,.25,.5): _line(draw,[(cx+dx*r,cy-r*.1),(cx+dx*r,cy+r*.5)],second,max(2,lw//2))
        elif kind=="cave":
            draw.arc((cx-r,cy-r,cx+r,cy+r),180,360,fill=accent,width=lw)
            draw.polygon([(cx-r,cy+r*.65),(cx-r*.72,cy-r*.25),(cx-r*.25,cy-r*.75),(cx+r*.2,cy-r*.4),(cx+r*.55,cy-r*.7),(cx+r,cy+r*.65)],fill=(10,14,25),outline=accent)
            _glow(draw,cx+r*.15,cy+r*.25,int(r*.17),second)
        elif kind=="storm":
            draw.ellipse((cx-r*.75,cy-r*.4,cx-r*.05,cy+r*.1),outline=accent,width=lw)
            draw.ellipse((cx-r*.3,cy-r*.65,cx+r*.35,cy+r*.05),outline=accent,width=lw)
            draw.ellipse((cx+r*.05,cy-r*.35,cx+r*.75,cy+r*.1),outline=accent,width=lw)
            _line(draw,[(cx+r*.1,cy),(cx-r*.2,cy+r*.45),(cx+r*.08,cy+r*.35),(cx-r*.1,cy+r*.8)],second,lw)
        elif kind=="animal":
            draw.ellipse((cx-r*.5,cy-r*.45,cx+r*.5,cy+r*.4),outline=accent,width=lw)
            draw.ellipse((cx-r*.7,cy-r*.65,cx-r*.25,cy-r*.25),outline=second,width=lw); draw.ellipse((cx+r*.25,cy-r*.65,cx+r*.7,cy-r*.25),outline=second,width=lw)
            draw.ellipse((cx-r*.2,cy-r*.1,cx-r*.08,cy+.02*r),fill=ink); draw.ellipse((cx+r*.08,cy-r*.1,cx+r*.2,cy+.02*r),fill=ink)
            draw.arc((cx-r*.2,cy+ r*.05,cx+r*.2,cy+r*.3),0,180,fill=second,width=max(2,lw//2))
        elif kind=="dna":
            for i in range(9):
                yy=cy-r+i*r*.25; x1=cx+math.sin(i*.8)*r*.55; x2=cx-math.sin(i*.8)*r*.55
                _line(draw,[(x1,yy),(x2,yy)],second,max(2,lw//2))
            _line(draw,[(cx+math.sin(i*.8)*r*.55,cy-r+i*r*.25) for i in range(9)],accent,lw)
            _line(draw,[(cx-math.sin(i*.8)*r*.55,cy-r+i*r*.25) for i in range(9)],accent,lw)
        elif kind=="cell":
            draw.ellipse((cx-r,cy-r*.75,cx+r,cy+r*.75),outline=accent,width=lw)
            draw.ellipse((cx-r*.25,cy-r*.2,cx+r*.15,cy+r*.2),outline=second,width=lw)
            for dx,dy in [(-.5,-.2),(.5,-.3),(.45,.35),(-.35,.4)]: draw.ellipse((cx+dx*r-r*.07,cy+dy*r-r*.07,cx+dx*r+r*.07,cy+dy*r+r*.07),fill=ink)
        elif kind=="plant":
            _line(draw,[(cx,cy+r*.75),(cx,cy-r*.45)],accent,lw)
            draw.ellipse((cx-r*.65,cy-r*.5,cx,cy+r*.02),outline=second,width=lw); draw.ellipse((cx,cy-r*.7,cx+r*.65,cy-r*.15),outline=accent,width=lw)
            draw.ellipse((cx-r*.3,cy-r*.9,cx+r*.3,cy-r*.35),outline=second,width=lw)
        elif kind=="map":
            draw.rounded_rectangle((cx-r,cy-r*.7,cx+r,cy+r*.7),radius=8,outline=accent,width=lw)
            pts=[(cx-r*.65,cy-r*.35),(cx-r*.15,cy-r*.05),(cx+r*.15,cy-r*.4),(cx+r*.62,cy-r*.15),(cx+r*.4,cy+r*.45)]
            _line(draw,pts,second,lw)
            draw.ellipse((pts[-1][0]-r*.1,pts[-1][1]-r*.1,pts[-1][0]+r*.1,pts[-1][1]+r*.1),fill=ink)
        elif kind=="compass":
            draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline=accent,width=lw)
            _line(draw,[(cx,cy-r*.72),(cx-r*.2,cy),(cx,cy+r*.72),(cx+r*.2,cy),(cx,cy-r*.72)],second,lw)
        elif kind=="rocket":
            draw.polygon([(cx,cy-r),(cx+r*.42,cy-r*.2),(cx+r*.35,cy+r*.4),(cx,cy+r*.7),(cx-r*.35,cy+r*.4),(cx-r*.42,cy-r*.2)],outline=accent)
            draw.ellipse((cx-r*.13,cy-r*.2,cx+r*.13,cy+r*.05),outline=second,width=lw)
            _line(draw,[(cx-r*.2,cy+r*.45),(cx-r*.45,cy+r*.8),(cx,cy+r*.62),(cx+r*.45,cy+r*.8),(cx+r*.2,cy+r*.45)],second,lw)
        elif kind=="satellite":
            draw.rectangle((cx-r*.2,cy-r*.2,cx+r*.2,cy+r*.2),outline=accent,width=lw)
            draw.rectangle((cx-r,cy-r*.45,cx-r*.25,cy+r*.45),outline=second,width=lw)
            draw.rectangle((cx+r*.25,cy-r*.45,cx+r,cy+r*.45),outline=second,width=lw)
            _line(draw,[(cx,cy-r*.2),(cx+r*.35,cy-r*.7)],accent,lw)
        elif kind=="volcano":
            _line(draw,[(cx-r,cy+r*.65),(cx-r*.4,cy-r*.45),(cx-r*.12,cy-r*.2),(cx+r*.12,cy-r*.2),(cx+r*.4,cy-r*.45),(cx+r,cy+r*.65)],accent,lw)
            for dx in (-.2,0,.2): draw.ellipse((cx+dx*r-r*.05,cy-r*.55,cx+dx*r+r*.05,cy-r*.4),fill=second)
        elif kind=="desert":
            _line(draw,[(cx-r,cy+r*.35),(cx-r*.45,cy-r*.2),(cx+r*.15,cy+r*.25),(cx+r*.7,cy-r*.35),(cx+r,cy+r*.35)],accent,lw)
            draw.ellipse((cx+r*.2,cy-r*.85,cx+r*.55,cy-r*.5),outline=second,width=lw)
        elif kind=="snow":
            for dx,dy in [(-.45,-.35),(.2,-.55),(.55,.05)]:
                x=cx+dx*r; y=cy+dy*r
                _line(draw,[(x-r*.16,y),(x+r*.16,y)],second,lw); _line(draw,[(x,y-r*.16),(x,y+r*.16)],second,lw)
                _line(draw,[(x-r*.11,y-r*.11),(x+r*.11,y+r*.11)],accent,max(2,lw//2))
        elif kind=="skull":
            draw.ellipse((cx-r*.65,cy-r*.8,cx+r*.65,cy+r*.35),outline=accent,width=lw)
            for dx in (-.27,.27): draw.ellipse((cx+(dx-.12)*r,cy-r*.18,cx+(dx+.12)*r,cy+r*.08),fill=second)
            draw.rectangle((cx-r*.32,cy+r*.25,cx+r*.32,cy+r*.58),outline=ink,width=lw)
        else:
            for dx,dy,rr in [(-.4,-.1,.38),(-.12,-.4,.38),(.23,-.32,.4),(.43,.02,.35),(.15,.28,.38),(-.22,.25,.38)]:
                draw.ellipse((cx+(dx-rr)*r,cy+(dy-rr)*r,cx+(dx+rr)*r,cy+(dy+rr)*r),outline=accent,width=lw)
            _line(draw,[(cx,cy-r*.65),(cx,cy+r*.65)],second,lw)
    else:
        draw.rounded_rectangle((cx-r*.65,cy-r,cx+r*.65,cy+r),radius=10,outline=accent,width=lw)

def _diagram(im, draw, query, ink, accent, second):
    """Render a subject-specific infographic layout rather than one universal flowchart."""
    w,h=im.size; text=query.lower()
    if any(k in text for k in ("orbit","planet","solar system","moon")):
        title="ORBITAL SYSTEM"; mode="orbit"; labels=["STAR","ORBIT","PLANET"]
    elif any(k in text for k in ("brain","neuron","memory","nervous system")):
        title="SIGNAL PATHWAY"; mode="flow"; labels=["SIGNAL IN","PROCESSING","RESPONSE"]
    elif any(k in text for k in ("engine","machine","mechanism","gear","how it works")):
        title="HOW IT WORKS"; mode="flow"; labels=["INPUT","MECHANISM","OUTPUT"]
    elif any(k in text for k in ("water cycle","life cycle","cycle of","cycle")):
        title="REPEATING CYCLE"; mode="cycle"; labels=["START","TRANSFORM","RETURN"]
    elif any(k in text for k in ("food chain","ecosystem")):
        title="FOOD CHAIN"; mode="flow"; labels=["ENERGY SOURCE","CONSUMER","PREDATOR"]
    elif any(k in text for k in ("timeline","history","evolution")):
        title="TIMELINE"; mode="timeline"; labels=["ORIGIN","CHANGE","OUTCOME"]
    elif any(k in text for k in ("floor plan","room layout","house layout","map","route")):
        title="SPACE & ROUTE"; mode="map"; labels=["START","LANDMARK","DESTINATION"]
    elif any(k in text for k in ("network","system architecture","internet")):
        title="CONNECTED NETWORK"; mode="network"; labels=["NODE A","NODE B","NODE C"]
    elif any(k in text for k in ("anatomy","cross section","structure of","cell")):
        title="STRUCTURE"; mode="structure"; labels=["OUTER LAYER","INNER PART","CORE"]
    else:
        title="CAUSE & EFFECT"; mode="flow"; labels=["TRIGGER","PROCESS","RESULT"]
    _center(draw,(w/2,h*.105),title,_font(max(18,w//22)),ink)
    draw.rounded_rectangle((w*.07,h*.17,w*.93,h*.88),radius=24,outline=(61,72,94),width=max(3,w//200))
    lw=max(3,w//220)
    if mode=="orbit":
        cx,cy=w*.5,h*.49; r=min(w*.29,h*.19)
        draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline=accent,width=lw)
        draw.ellipse((cx-r*.67,cy-r*.67,cx+r*.67,cy+r*.67),outline=second,width=max(2,lw-1))
        draw.ellipse((cx-r*.16,cy-r*.16,cx+r*.16,cy+r*.16),fill=(255,187,93),outline=ink,width=lw)
        draw.ellipse((cx+r*.88,cy-r*.12,cx+r*1.12,cy+r*.12),fill=second,outline=ink,width=lw)
        _center(draw,(w*.5,h*.74),labels[0]+"  •  "+labels[1]+"  •  "+labels[2],_font(max(13,w//38)),ink)
    elif mode=="timeline":
        yy=h*.49; _line(draw,[(w*.18,yy),(w*.82,yy)],accent,lw)
        for i,label in enumerate(labels):
            xx=w*(.2+i*.3); draw.ellipse((xx-w*.025,yy-w*.025,xx+w*.025,yy+w*.025),fill=second if i==1 else accent)
            _line(draw,[(xx,yy-w*.025),(xx,yy-h*.1)],second,lw)
            _center(draw,(xx,yy-h*.17),label,_font(max(12,w//38)),ink)
        _center(draw,(w*.5,h*.68),"THEN  →  NOW  →  NEXT",_font(max(12,w//40)),second)
    elif mode=="cycle":
        cx,cy=w*.5,h*.49; r=min(w*.27,h*.18)
        pts=[]
        for a in (-90,30,150):
            rad=math.radians(a); pts.append((cx+math.cos(rad)*r,cy+math.sin(rad)*r))
        for i in range(3):
            p1=pts[i]; p2=pts[(i+1)%3]
            _line(draw,[p1,p2],accent,lw)
            mx=(p1[0]+p2[0])*.5; my=(p1[1]+p2[1])*.5
            _glow(draw,int(mx),int(my),int(w*.035),second)
        for (xx,yy),label in zip(pts,labels):
            draw.ellipse((xx-w*.075,yy-w*.055,xx+w*.075,yy+w*.055),fill=(24,31,49),outline=second,width=lw)
            _center(draw,(xx,yy-h*.012),label,_font(max(10,w//46)),ink)
    elif mode=="map":
        # Schematic map with route, blocks and a clear destination pin.
        for row in range(4):
            for col in range(3):
                x0=w*(.15+col*.24); y0=h*(.27+row*.12)
                draw.rounded_rectangle((x0,y0,x0+w*.16,y0+h*.075),radius=7,fill=(21,30,46),outline=(55,70,91),width=2)
        pts=[(w*.2,h*.72),(w*.35,h*.6),(w*.48,h*.65),(w*.62,h*.43),(w*.8,h*.3)]
        _line(draw,pts,accent,max(4,lw+1))
        for i,(xx,yy) in enumerate((pts[0],pts[2],pts[-1])):
            draw.ellipse((xx-w*.025,yy-w*.025,xx+w*.025,yy+w*.025),fill=second if i==2 else accent)
        _center(draw,(w*.5,h*.79),"START  →  LANDMARK  →  END",_font(max(11,w//42)),ink)
    elif mode=="network":
        pts=[(w*.5,h*.34),(w*.27,h*.51),(w*.73,h*.51),(w*.35,h*.72),(w*.65,h*.72)]
        for i in range(1,len(pts)):
            _line(draw,[pts[0],pts[i]],(78,103,132),lw)
        for i,(xx,yy) in enumerate(pts):
            rr=w*(.035 if i==0 else .027)
            draw.ellipse((xx-rr,yy-rr,xx+rr,yy+rr),fill=accent if i==0 else (24,31,49),outline=second,width=lw)
    elif mode=="structure":
        cx,cy=w*.5,h*.49
        for scale,col in ((.28,accent),(.19,second),(.09,ink)):
            rr=min(w,h)*scale
            draw.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=col,width=lw)
        _line(draw,[(cx+min(w,h)*.28,cy),(w*.8,cy-h*.12)],second,lw)
        _line(draw,[(cx+min(w,h)*.19,cy),(w*.8,cy)],accent,lw)
        _line(draw,[(cx+min(w,h)*.09,cy),(w*.8,cy+h*.12)],ink,lw)
    else:
        node_w=w*.62; node_h=h*.105; ys=[h*.31,h*.51,h*.71]
        for i,(label,yy) in enumerate(zip(labels,ys)):
            x0=(w-node_w)/2; y0=yy-node_h/2
            draw.rounded_rectangle((x0,y0,x0+node_w,y0+node_h),radius=18,fill=(24,31,49),outline=accent if i!=1 else second,width=lw)
            _center(draw,(w/2,y0+node_h*.25),label,_font(max(13,w//34)),ink)
            xx=x0+w*.08; rr=w*.025
            draw.ellipse((xx-rr,yy-rr,xx+rr,yy+rr),outline=second if i==1 else accent,width=max(2,w//280))
            if i<2:
                _line(draw,[(w/2,y0+node_h+6),(w/2,ys[i+1]-node_h*.65)],second,lw)
                draw.polygon([(w/2,ys[i+1]-node_h*.42),(w/2-w*.025,ys[i+1]-node_h*.62),(w/2+w*.025,ys[i+1]-node_h*.62)],fill=second)
    for x in (w*.14,w*.84):
        _line(draw,[(x,h*.22),(x+w*.045,h*.22)],accent,max(2,w//280))
        _line(draw,[(x,h*.82),(x+w*.045,h*.82)],second,max(2,w//280))

def make_visual(
    query: str, image_prompt: str, destination: Path, width: int, height: int,
    variation: int = 0, mode: str | None = None, style_key: str = "",
) -> str:
    """Create reusable story artwork, a diagram, or a consistent character portrait."""
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    character_mode = (mode or "").lower() == "character"
    seed_text = (f"character:{style_key.casefold()}" if character_mode and style_key
                 else query + "|" + image_prompt + "|style:" + style_key + "|variation:" + str(variation))
    seed=int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8],16)
    bg,accent,second,ink=PALETTES[seed%len(PALETTES)]
    w=max(360,int(width)); h=max(640,int(height))
    im=Image.new("RGB",(w,h),bg); draw=ImageDraw.Draw(im)
    for i in range(44):
        x=(seed*(i+7)*31)%w; y=(seed*(i+13)*17)%h; rr=1+i%3
        draw.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(31,40,60))
    draw.rounded_rectangle((w*.045,h*.035,w*.955,h*.965),radius=26,outline=(55,67,90),width=max(2,w//400))
    kind=(mode or visual_type(query,image_prompt)).lower()
    if kind == "character":
        _character(draw,w,h,ink,accent,second,seed,character_kind(style_key or query+" "+image_prompt))
    elif kind == "diagram":
        _diagram(im,draw,f"{query} {image_prompt}",ink,accent,second)
    else:
        _illustration(im,draw,f"{query} {image_prompt}",ink,accent,second,seed)
    im.save(destination,format="PNG",optimize=True)
    return kind
