"""Large reusable recipe library for cinematic fantasy role-play Shorts.

These are decision menus and production rules for the scene director, not a fixed template.
The director should select a coherent subset per episode and vary choices between episodes.
"""
FANTASY_LOOK_OPTIONS = (
    "grounded live-action medieval fantasy",
    "dark gothic fantasy film",
    "high fantasy epic with monumental scale",
    "photoreal creature-feature fantasy",
    "mysterious enchanted-forest fantasy",
    "mythic ancient-world fantasy",
    "snowbound northern fantasy",
    "desert kingdom fantasy",
    "underwater kingdom fantasy",
    "sky-island / floating-citadel fantasy",
    "ruined empire / post-war fantasy",
    "court intrigue with restrained magic",
    "dark fairytale with practical creature effects",
    "realistic premium fantasy-game cinematic",
    "swashbuckling adventure fantasy",
    "cosmic / celestial fantasy",
)

WORLD_OPTIONS = (
    "rain-soaked medieval castle and narrow torchlit corridors",
    "busy medieval market at dawn",
    "throne room with tall windows and banners",
    "wind-battered battlements above a stormy sea",
    "dragon nesting grounds in a volcanic mountain",
    "ancient forest with enormous roots and ground fog",
    "snow-covered fortress during a blizzard",
    "desert citadel with sandstone arches and blowing dust",
    "underground dwarven hall with colossal machinery",
    "ruined temple covered in vines",
    "moonlit lake and a half-submerged shrine",
    "floating islands connected by rope bridges",
    "royal library with ladders, maps and candlelight",
    "medieval village during an evacuation",
    "battlefield after the fighting has stopped",
    "hidden stable / creature sanctuary",
    "crystal cavern lit by reflected natural light",
    "ancient bridge above a bottomless ravine",
    "healer's chamber with practical herbs and tools",
    "secret passage behind a tapestry",
    "crowded royal banquet interrupted by a creature",
    "ship deck in rough seas",
    "abandoned watchtower at sunrise",
    "underground prison with wet stone and iron doors",
)

CAMERA_SHOT_OPTIONS = (
    "extreme wide establishing shot",
    "wide tracking shot following a character",
    "medium two-shot showing a relationship",
    "over-the-shoulder reveal",
    "low-angle creature reveal emphasizing scale",
    "eye-level dialogue coverage",
    "tight facial reaction close-up",
    "close-up of hands performing a specific action",
    "macro insert of a named clue or prop",
    "side-profile tracking shot",
    "foreground-obstructed spying shot",
    "slow push-in as the truth lands",
    "pull-back revealing a hidden threat",
    "controlled handheld movement during danger",
    "silhouette framed by a doorway or lightning",
    "rack-focus from clue to character reaction",
    "top-down geography shot only when spatial clarity matters",
    "whip-pan only for a motivated discovery or sudden attack",
)

LIGHTING_OPTIONS = (
    "warm torchlight against cool moonlight",
    "overcast daylight with realistic soft shadows",
    "storm flashes briefly revealing the creature",
    "dawn backlight through mist",
    "candlelight with deep but readable shadows",
    "firelight reflecting on wet armor",
    "cold blue snow light with warm windows in the distance",
    "dusty late-afternoon sunbeams",
    "subtle bioluminescence limited to the magical subject",
    "eclipse light with a clear readable silhouette",
    "natural window light in a quiet emotional scene",
    "smoke and ember light during a battle aftermath",
)

WEATHER_AND_ATMOSPHERE_OPTIONS = (
    "fine rain and wet stone",
    "heavy rain driven sideways by wind",
    "low ground fog that does not hide the action",
    "snow and wind around armor and banners",
    "ash drifting from a distant volcano",
    "dust clouds from approaching riders",
    "sea spray on the battlements",
    "wind moving cloaks, banners and loose hair",
    "quiet clear night with visible stars",
    "morning mist slowly lifting",
    "smoke from a recently extinguished fire",
    "dry heat shimmer above desert stone",
    "falling leaves in a forest passage",
    "subtle floating magical particles only when story-relevant",
)

CREATURE_OPTIONS = (
    "ancient dragon with believable weight, scales and wing mechanics",
    "small injured dragon with expressive eyes and physical vulnerability",
    "massive guardian golem with stone joints and heavy movement",
    "wolf-like shadow creature with readable anatomy",
    "griffin with feathered wings and believable takeoff",
    "phoenix with physical feathers and controlled flame",
    "giant serpent moving through real environmental obstacles",
    "armored troll with practical-looking skin and heavy footsteps",
    "unicorn behaving like a real animal rather than a toy",
    "spectral knight with worn armor and restrained supernatural effects",
    "sea creature emerging from believable water and spray",
    "enchanted forest creature partially concealed by foliage",
    "wyvern landing with weight and dust displacement",
    "ancient giant whose scale is shown against architecture",
    "tiny magical creature interacting physically with a human prop",
)

MAGIC_EFFECT_OPTIONS = (
    "a protective barrier visibly deflects one incoming strike",
    "runes ignite one by one along an actual stone surface",
    "a sword freezes from hilt to tip as a character grips it",
    "a locked door opens only after a physical magical cause",
    "a dragon's breath lights nearby surfaces before the flame arrives",
    "a healing glow fades as the wound visibly closes",
    "a shadow moves against the direction of the light",
    "water rises and forms a brief physical shape",
    "a portal distorts the background and affects nearby cloth and dust",
    "a spell knocks loose objects across the room",
    "a magical disguise breaks in stages, revealing the real face",
    "an enchanted object reacts to a character's touch",
    "a frost line spreads across stone from a clear point of contact",
    "a spell fails, with a visible consequence for its caster",
    "magic remains subtle: light, wind, heat, or object movement rather than neon particles",
)

STORY_ENGINE_OPTIONS = (
    "secret identity exposed at the worst possible moment",
    "reluctant protector hiding a dangerous secret",
    "enemy forced to cooperate to survive",
    "a tiny clue changes the meaning of an earlier scene",
    "a promise tested by an impossible choice",
    "a creature feared by everyone protects one child",
    "a royal order contradicts what the hero witnessed",
    "a rescue attempt reveals the person rescued was hiding something",
    "a harmless object turns out to be the key to the danger",
    "a powerful character loses control of their own magic",
    "a betrayal is implied but not confirmed until the final beat",
    "a character chooses mercy and pays an immediate cost",
    "a quiet conversation is interrupted by a physical threat",
    "a victory creates a larger problem",
    "a familiar location is revealed to contain a hidden second purpose",
    "a mystery solved too early creates a more unsettling question",
    "a role reversal: the guarded person saves the guard",
    "a creature recognizes a symbol no human should know",
)

TRANSITION_OPTIONS = (
    "hard cut on movement",
    "match cut between similar shapes or gestures",
    "foreground wipe caused by a passing character or object",
    "sound-led cut just before the visual reveal",
    "brief dissolve only for a clear time or memory change",
    "cut from wide geography to a meaningful close-up",
    "hold a reaction for a beat before cutting",
    "whip-pan only when the action motivates it",
)

FANTASY_SHOT_SEQUENCE = (
    "establishing wide",
    "specific character action",
    "reaction close-up or prop insert",
    "opposing response",
    "consequence or reveal",
    "payoff or cliffhanger",
)

FANTASY_VISUAL_RULES = """
CINEMATIC FANTASY VIDEO RECIPE LIBRARY — CHOOSE, DON'T RANDOMLY MIX
Goal: make the viewer feel they are watching a coherent scene from a live-action fantasy film
or premium photoreal CGI cinematic, not a slideshow of drawings. For each episode, select ONE
dominant look, ONE coherent world, a restrained lighting/weather palette, and a story engine.
Choose different options across episodes so the library creates variety without losing continuity.
Use the option menus in this module as creative choices; do not dump all choices into one prompt.

SHOT GRAMMAR — select the shot that matches the exact narrated beat:
1. ESTABLISHING WIDE: show the real geography of the castle, forest, room or battlefield.
2. CHARACTER ACTION: show a readable physical verb and its result.
3. REACTION CLOSE-UP: show the same character registering what just happened.
4. PROP INSERT: show the exact named clue, key, egg, seal, wound or weapon.
5. SCALE REVEAL: show the actual creature or threat against a familiar size reference.
6. TWO-SHOT / OVER-SHOULDER: make relationships, conflict and power dynamics legible.
7. MAGIC CAUSE-AND-EFFECT: show what triggers magic and what it physically changes.
8. PAYOFF: fulfil the visual promise made by the hook.
9. CLIFFHANGER: introduce one new visual question or threat, then cut.

EPISODE LOOK AND VARIETY:
- Choose a style appropriate to the topic: grounded medieval, gothic, epic-scale, creature-feature,
  enchanted forest, ancient myth, snowbound, desert, underwater, sky-island, ruined empire,
  court intrigue, dark fairytale, game cinematic, swashbuckling or celestial.
- Rotate environments, camera shot sizes, lighting and weather across episodes. Within an episode,
  changes must be motivated by the story, not random aesthetic changes.
- Build a shot list with a mix of wides, mediums, close-ups, inserts and over-the-shoulder views.
  Avoid repeating centered portraits or the same angle in consecutive shots.
- Prefer 2–4 seconds per shot for fast Shorts pacing, but let emotional reveals breathe. Cut on
  movement or a meaningful reaction; don't add transitions merely for decoration.
- Each shot must show the exact narrated event. A query should include subject + physical action +
  location/context, not vague terms like "fantasy", "magic", "epic" or "dragon aesthetic".
- For every magical event, establish trigger, visible effect and consequence. No meaningless
  neon particles covering the screen.
- Creatures must have believable anatomy, weight, contact with the environment, and consistent size.
  Show wing beats, footfalls, displaced dust/water, broken objects or shadows when appropriate.
- Keep recurring characters consistent in face, age, hair, clothing/armor, palette, scars, species,
  height and signature props. Keep prop damage and costume state continuous from shot to shot.
- Keep geography, time, weather, light direction and screen direction coherent across consecutive
  shots. Do not teleport characters or change armor colors between cuts.
- Prefer genuine moving footage when available. For stills, use photoreal movie-frame imagery and
  deliberate camera motion. Never pretend a still is video footage.
- No flat vector art, cartoon icons, diagrams, emoji, generic gradients, unrelated stock footage,
  collage, baked-in words, fake subtitles, logos, watermarks or game HUDs.
- Captions are added separately. Compose for vertical 9:16; keep faces, hands, creatures and key
  props away from caption-heavy lower areas when possible.
- Every shot should be useful. If no source matches the action, search with a more specific query
  or use a clearly identified photoreal still fallback; do not claim the mismatch is a match.
- The library guides planning, not source availability. Actual moving footage depends on the
  configured provider and whether suitable assets exist.
""".strip()
