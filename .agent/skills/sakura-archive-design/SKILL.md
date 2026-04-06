---
name: sakura-archive-design
description: Use when building, styling, or modifying UI components for the VNCCS / Sakura Archive project. Covers the complete design system including dark-theme colors, glassmorphism, layout columns, pose cards, animations, and strict layout rules that prevent common breakage.
---

# Sakura Archive Design System

Premium dark-anime aesthetic for the VNCCS Character Creator. Every UI element MUST follow these rules.

## Core Philosophy

- **Premium Dark Theme**: Deep backgrounds with subtle gradients. No flat blacks.
- **Glassmorphism**: Translucent surfaces with `backdrop-filter: blur`, luminous borders.
- **Selective Accents**: Sakura Pink `#ff8fa3` for interactions, glows, active states only.
- **Micro-animations**: Hover states lift, glow, and shift. No abrupt state changes.
- **Two-Layer Rule**: Layout sizing and visual styling are ALWAYS on separate elements.

---

## 1. CSS Variables (The Only Source of Truth)

NEVER hardcode colors. Always use variables.

```css
:root {
  /* Backgrounds */
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-elevated: #1a1a26;
  --bg-surface: #22222e;
  --bg-hover: #2a2a38;

  /* Text */
  --text-primary: #e8e8f0;
  --text-secondary: #9898a8;
  --text-muted: #5e5e70;

  /* Accent (Sakura Signature) */
  --accent: #ff8fa3;
  --accent-hover: #ffb6c8;
  --accent-glow: rgba(255, 143, 163, 0.3);
  --accent-subtle: rgba(255, 143, 163, 0.1);
  --accent-border: rgba(255, 143, 163, 0.22);
  --accent-lavender: #b8a9e8;

  /* Semantic */
  --success: #00d68f;
  --warning: #ffaa00;
  --error: #ff4757;

  /* Borders */
  --border: rgba(255, 255, 255, 0.06);
  --border-hover: rgba(255, 255, 255, 0.12);

  /* Typography */
  --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Spacing */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
  --radius-xl: 24px;

  /* Shadows */
  --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.3);
  --shadow-elevated: 0 8px 32px rgba(0, 0, 0, 0.5);

  /* Transitions */
  --transition: 0.2s ease;
}
```

---

## 2. Typography

- **UI text**: `var(--font)` (Sora)
- **Technical data** (seeds, phases, tags, numbers): `var(--font-mono)` (JetBrains Mono)
- Uppercase labels: `letter-spacing: 0.05em` or wider
- Inputs/buttons: `font-weight: 500-600`
- Hierarchy: Headings (`--text-primary`, 600-700), Body (400-500), Meta (`--text-secondary`, 400)

---

## 3. Layout System (CRITICAL)

### Container

All screens share ONE container: `char-creator__body` (flex row, `gap: 16px`, `align-items: stretch`).

```css
.char-creator__body {
  display: flex;
  gap: 16px;
  align-items: stretch;
  justify-content: flex-start;
  height: 100%;
  min-height: 0;
  flex: 1;
}
```

### Column Types

Every screen is a horizontal row of columns. Pick from these types:

| Class | Behavior | When to Use |
|---|---|---|
| `cc-col--sidebar` | `flex: 1 1 280px` (grows) | Form with tabs. ONLY Base Character screen. |
| `cc-col--canvas` | `flex: 1` (fills remaining space) | Main preview. ONLY Base Character screen. |
| `cc-col--stage` | `flex: 0 0 auto` (content-sized) | Generated image preview. ALL other screens. |
| `cc-col--source` | `flex: 0 0 calc(...)` (aspect-based) | Reference image display. |
| `cc-col--aside` | `flex: 0 0 240px` (fixed) | Settings panel. |
| `cc-col--nav` | `flex: 0 0 84px` (fixed) | Navigation button (back/forward). |

### The canvas vs stage Trap

```
cc-col--canvas = STRETCHES to fill all available space (flex: 1)
cc-col--stage  = SIZES TO CONTENT (flex: 0 0 auto)
```

**WRONG**: Using `cc-col--canvas` on any screen except Base Character.
Result: preview stretches, nav button flies to far edge, massive gaps appear.

**RIGHT**: Using `cc-col--stage` on Poses, Emotions, Outfits, Sprites.
Result: blocks hug each other, no gaps.

```
CORRECT (stage):   [ back ][ reference ][ preview ][ next ]
                    ← blocks touching each other

BROKEN (canvas):   [ back ][ reference ][    preview STRETCHED    ][ next ]
                    ← huge empty gap, button at screen edge
```

### Two-Layer Rule

Every column = wrapper (sizing) + content (visual). NEVER mix them.

```jsx
{/* CORRECT */}
<div className="cc-col cc-col--aside">
    <div className="cc-aside-panel">...settings...</div>
</div>

{/* BROKEN — mixing layout and visual on same element */}
<div className="cc-col cc-col--aside cc-aside-panel">
    ...settings...
</div>
```

### Column CSS

```css
.cc-col {
  position: relative;
  height: 100%;
  min-height: 0;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.cc-col--sidebar { flex: 1 1 280px; }
.cc-col--canvas { flex: 1; transition: flex-basis 0.42s cubic-bezier(0.22, 1, 0.36, 1); }
.cc-col--canvas-locked { flex: 0 0 calc((100vh - 150px) * 640 / 1536); }
.cc-col--aside { flex: 0 0 240px; }
.cc-col--nav { flex: 0 0 84px; }
.cc-col--source { flex: 0 0 calc((100vh - 150px) * 640 / 1536); width: calc((100vh - 150px) * 640 / 1536); }
.cc-col--stage { flex: 0 0 auto; width: auto; gap: 16px; align-items: flex-start; justify-content: flex-start; }
```

### Stage Width Measurement (MANDATORY)

CSS cannot reliably calculate stage width. You MUST use ResizeObserver:

```jsx
const stageRef = useRef(null);
const [stageWidth, setStageWidth] = useState(0);

useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            setStageWidth(entry.contentRect.width);
        }
    });
    observer.observe(el);
    return () => observer.disconnect();
}, []);
```

Apply measured width to the COLUMN and FOOTER, NOT to the preview container:

```jsx
<div className="cc-col cc-col--stage"
     style={stageWidth ? { width: `${stageWidth}px` } : undefined}>
    <div className="cc-stage-preview">
        <div ref={stageRef} className="cc-canvas cc-stage-canvas">...</div>
    </div>
    <div className="cc-stage-footer"
         style={stageWidth ? { width: `${stageWidth}px` } : undefined}>
        <button className="cc-btn cc-btn--primary cc-btn--large">Generate</button>
    </div>
</div>
```

**WARNING**: Setting measured width on `cc-stage-preview` causes recursive ResizeObserver loop.

---

## 4. Aspect-Ratio Rule for Cards with Header/Footer

> **NEVER put `aspect-ratio` on a card that contains a header (variant bar), image area (surface), and footer.**
> `aspect-ratio` goes ONLY on the surface.

### Why

```
┌─────────────────────┐
│  variant bar (30px)  │  ← header
├─────────────────────┤
│                     │
│     surface         │  ← image area (flex: 1)
│                     │
├─────────────────────┤
│  footer (40px)      │  ← buttons
└─────────────────────┘
```

If `aspect-ratio: 2/3` is on the **card**: width is calculated from TOTAL card height (including bar + footer). But the image lives only in surface, which is ~70px shorter. Result:
- `object-fit: contain` → empty strips on sides
- `object-fit: cover` → head/feet cropped under header/footer

If `aspect-ratio: 2/3` is on the **surface**: width is calculated from surface height alone (already excluding bar + footer). Image fits exactly.

### Correct CSS

```css
.cc-pose-card {
  display: flex;
  flex-direction: column;
  height: 100%;
  /* NO aspect-ratio here! */
}

.cc-pose-card__surface {
  flex: 1;
  min-height: 0;
  aspect-ratio: var(--pose-aspect-ratio, 2 / 3);
  position: relative;
  overflow: hidden;
}

.cc-pose-card__image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
}
```

### What NOT to do

| Action | Result |
|---|---|
| `aspect-ratio` on `.cc-pose-card` | Empty strips or image cropping |
| `object-fit: cover` on image | Head/feet hidden under header/footer |
| `position: absolute` on image | Image escapes layout |

---

## 5. Component Reference

### Section Titles

```css
.cc-section-title {
  font-size: 10px;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.cc-section-title::before {
  content: '';
  width: 3px;
  height: 12px;
  background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
  border-radius: 2px;
  box-shadow: 0 0 8px var(--accent-glow);
}
```

### Glassmorphic Panels

```css
.cc-aside-panel {
  flex: 1;
  background: rgba(20, 16, 30, 0.88);
  border-radius: var(--radius-lg);
  border: 1px solid var(--accent-border);
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: relative;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.cc-aside-panel::before {
  content: '';
  position: absolute;
  top: 0;
  left: 18%;
  right: 18%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
  border-radius: 1px;
}
```

### Primary Button (with shimmer)

```css
.cc-btn--primary {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
  color: #1a1525;
  box-shadow: 0 4px 20px rgba(255, 143, 163, 0.25);
  position: relative;
  overflow: hidden;
}
.cc-btn--primary::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.22) 45%, rgba(255,255,255,0.32) 50%, rgba(255,255,255,0.22) 55%, transparent 100%);
  transform: translateX(-120%) skewX(-15deg);
  animation: ccBtnShimmer 3.5s ease-in-out infinite;
  pointer-events: none;
}
.cc-btn--primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(255, 143, 163, 0.45), 0 0 0 1px var(--accent-glow);
  background: linear-gradient(135deg, #ff9fb3 0%, #ffc6d8 100%);
}
```

### Form Inputs

```css
.cc-input, .cc-select {
  width: 100%;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  color: #fff;
  font-family: var(--font);
  font-size: 13px;
  transition: all var(--transition);
}
.cc-input:focus, .cc-select:focus {
  outline: none;
  border-color: var(--accent-border);
  background: rgba(255, 143, 163, 0.03);
  box-shadow: 0 0 0 3px rgba(255, 143, 163, 0.05);
}
```

### Pose Card (Full)

```css
.cc-pose-card {
  display: flex;
  flex-direction: column;
  min-width: 0;
  height: 100%;
  border: 1px solid var(--accent-border);
  border-radius: var(--radius-lg);
  background: rgba(20, 16, 30, 0.7);
  overflow: hidden;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
  transition: all var(--transition);
}
.cc-pose-card:hover {
  border-color: var(--accent);
  transform: translateY(-4px);
  box-shadow: 0 16px 42px rgba(0, 0, 0, 0.35), 0 0 15px var(--accent-subtle);
}

.cc-pose-card__variant-bar {
  display: flex;
  flex-shrink: 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.cc-pose-card__variant-btn {
  flex: 1;
  padding: 5px 0;
  border: none;
  background: transparent;
  color: rgba(240, 232, 245, 0.35);
  font-weight: 600;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  cursor: pointer;
  transition: all 0.2s;
}
.cc-pose-card__variant-btn.active {
  background: rgba(255, 143, 163, 0.1);
  color: #ffb6c8;
}

.cc-pose-card__surface {
  flex: 1;
  min-height: 0;
  aspect-ratio: var(--pose-aspect-ratio, 2 / 3);
  border: 0;
  padding: 0;
  margin: 0;
  background: #000;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.cc-pose-card__image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
  display: block;
}

.cc-pose-card__footer {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.cc-pose-card__generate-btn {
  flex: 1;
  min-height: 26px;
  padding: 0 10px;
  border-radius: 6px;
  background: var(--accent);
  color: #fff;
  border: none;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  cursor: pointer;
  box-shadow: 0 4px 12px var(--accent-glow);
  transition: all var(--transition);
}

.cc-pose-card__delete-btn {
  width: 26px;
  height: 26px;
  flex-shrink: 0;
  border-radius: 6px;
  background: rgba(255, 71, 87, 0.15);
  color: #ff4757;
  border: 1px solid rgba(255, 71, 87, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 14px;
}
.cc-pose-card__delete-btn:hover {
  background: #ff4757;
  color: #fff;
}

.cc-pose-card__add {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  border: 1px dashed rgba(255, 143, 163, 0.3);
  background: rgba(255, 143, 163, 0.04);
  color: var(--accent);
  cursor: pointer;
}
.cc-pose-card__add:hover {
  background: rgba(255, 143, 163, 0.08);
  border-color: var(--accent);
}
```

### Pose Grid

```css
.cc-pose-grid {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-rows: repeat(var(--cc-grid-rows, 2), 1fr);
  grid-auto-flow: column;
  grid-auto-columns: min-content;
  gap: 16px;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 4px 4px 12px;
  scrollbar-width: thin;
  scrollbar-color: var(--accent-border) transparent;
}
```

### Navigation CTA

```css
.cc-pose-cta {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px dashed rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-lg);
  color: var(--text-muted);
  cursor: pointer;
  transition: all var(--transition);
  padding: 24px 12px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.cc-pose-cta:hover:not(.cc-pose-cta--saved) {
  background: rgba(255, 143, 163, 0.08);
  border-color: rgba(255, 143, 163, 0.22);
  color: #ffb6c8;
}
.cc-pose-cta__label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  writing-mode: vertical-rl;
  text-orientation: mixed;
}
.cc-pose-cta--saved {
  border-color: rgba(74, 222, 128, 0.4);
  background: linear-gradient(165deg, rgba(74, 222, 128, 0.12) 0%, rgba(45, 212, 191, 0.08) 100%);
}
```

### Canvas (empty, with result, loading)

```css
.cc-canvas {
  flex: 1;
  min-height: 0;
  background: radial-gradient(circle, rgba(255, 143, 163, 0.055) 1px, transparent 1px), rgba(15, 12, 22, 0.6);
  background-size: 22px 22px, 100% 100%;
  border-radius: var(--radius-lg);
  border: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.3);
}
.cc-canvas--with-result {
  flex: 1 1 auto;
  min-height: 0;
  padding: 0;
  box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.24), 0 18px 48px rgba(0, 0, 0, 0.18);
}
.cc-canvas__img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center top;
  border-radius: var(--radius-lg);
  animation: fadeIn 0.5s ease-out;
}
.cc-canvas__loading {
  position: absolute;
  inset: 0;
  background: rgba(14, 11, 20, 0.88);
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1.2rem;
  z-index: 10;
}
```

### Spinner

```css
.cc-spinner {
  width: 44px;
  height: 44px;
  position: relative;
}
.cc-spinner::before, .cc-spinner::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 3px solid transparent;
}
.cc-spinner::before {
  border-top-color: #ff8fa3;
  border-right-color: rgba(255, 143, 163, 0.3);
  animation: spin 1s linear infinite;
  box-shadow: 0 0 18px rgba(255, 143, 163, 0.2);
}
.cc-spinner::after {
  inset: 7px;
  border-bottom-color: rgba(184, 169, 232, 0.6);
  border-left-color: rgba(184, 169, 232, 0.2);
  animation: spin 1.4s linear infinite reverse;
}
```

---

## 6. Animations

### Utility Classes

```css
.animate-slide-up   { animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.animate-slide-down { animation: slideDownFade 0.35s cubic-bezier(0.22, 1, 0.36, 1) forwards; }
.animate-fade-in    { animation: fadeIn 0.3s ease; }
.animate-reveal     { animation: sakuraReveal 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.animate-pulse      { animation: pulse 1.5s ease-in-out infinite; }
```

### Key Keyframes

```css
@keyframes ccBtnShimmer {
  0%  { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
  35% { transform: translateX(120%) skewX(-15deg); opacity: 1; }
  100%{ transform: translateX(120%) skewX(-15deg); opacity: 0; }
}
@keyframes ccRevealImage {
  0%   { opacity: 0; transform: scale(1.035); filter: saturate(0.8) blur(8px); }
  65%  { opacity: 1; }
  100% { opacity: 1; transform: scale(1); filter: saturate(1) blur(0); }
}
@keyframes ccIconFloat {
  0%,100% { transform: translateY(0); opacity: 0.45; }
  50%     { transform: translateY(-7px); opacity: 0.65; filter: drop-shadow(0 0 28px rgba(255,143,163,0.58)); }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUpFade { from { opacity:0; transform:translateY(15px); } to { opacity:1; transform:translateY(0); } }
```

---

## 7. Screen Recipes

### Base Character (ONLY screen with canvas)

```
char-creator__body
  ├── cc-col--sidebar  → form with tabs (GROWS)
  ├── cc-col--canvas   → preview (GROWS)
  ├── cc-col--aside    → upscale settings (conditional)
  └── cc-col--nav      → "Create Poses" button (conditional)
```

### Pose Studio / Emotions / Outfits / Sprites

```
char-creator__body
  ├── cc-col--nav      → back button
  ├── cc-col--source   → reference character
  └── cc-col--stage    → generated result + button  ← STAGE, NOT CANVAS!
```

### Adding a New Screen

1. Add value to `creatorView` state
2. Use `char-creator__body` (same container, no custom wrappers)
3. Pick column types from table above
4. For preview: use `cc-col--stage` (NOT `cc-col--canvas`)
5. Wire up ResizeObserver for stage width
6. NO new CSS needed — all styles already exist

---

## 8. Responsive Breakpoints

| Width | Behavior |
|---|---|
| `> 2200px` | Sidebar tabs become side-by-side columns |
| `1200-2200px` | Standard horizontal layout |
| `< 1200px` | Stack vertically, stage gets `width: 100%` |
| `< 1100px` | Nav CTA becomes horizontal pill |
| `< 900px` | Compact workspace cards |

---

## 9. Forbidden Actions

| Action | Result |
|---|---|
| `cc-col--canvas` on any screen except Base Character | Gaps, buttons fly to edges |
| `width: 100%` on canvas inside `cc-col--stage` | Stage stretches to full width |
| `justify-content: center` on `char-creator__body` | Blocks float with gaps |
| Mix layout and visual on same element | Unpredictable sizing |
| `flex: 1` on content inside `cc-col--stage` | Same as canvas — stretching |
| New wrapper instead of `char-creator__body` | Breaks gap, stretch, etc. |
| Modifiers on `char-creator__body` | May break other screens |
| `aspect-ratio` on `.cc-pose-card` (the card itself) | Empty strips or image cropping |
| `object-fit: cover` on `.cc-pose-card__image` | Head/feet under header/footer |
| `position: absolute` on `.cc-pose-card__image` | Image escapes layout |
| Setting measured width on `cc-stage-preview` | Recursive ResizeObserver loop |

---

## 10. Implementation Workflow

When building new UI for this project:

1. **Structure**: Semantic JSX. Group controls in `.cc-section`.
2. **Primitives first**: Use existing classes (`.cc-input`, `.cc-btn`, `.cc-section-title`) before writing new CSS.
3. **Layer aesthetics**: Glassmorphic base → top highlight border → hover states.
4. **Add motion**: `.animate-slide-up` on major regions with staggered `animation-delay`.
5. **Icons**: 1.5-2px stroke SVG. Hover changes stroke to `--accent`.
