---
name: Speech editor fixture
description: Operate workbench for synthetic production editing
colors:
  primary: "#145c57"
  background: "#f4f6f6"
  surface: "#ffffff"
  ink: "#202e32"
  muted: "#506268"
  border: "#c9d3d4"
  selected: "#b9dcd5"
typography:
  title:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "20px"
    fontWeight: 650
    letterSpacing: "-0.025em"
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "14px"
  label:
    fontSize: "12px"
rounded:
  control: "5px"
  clip: "4px"
spacing:
  compact: "8px"
  panel: "16px"
  outer: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    height: "34px"
    padding: "6px 10px"
---

# Design System: Speech editor fixture

## Overview

The editorial workbench keeps source text, fixed placement and precise properties visible together. This is the provisional system evidenced by `studio/src/style.css`, not a final product brand. Mode and route strategy live in the studio surface brief.

## Colors

Cool neutral surfaces carry sustained editing. Deep teal marks selection and primary actions. Muted states also use words, dashed borders or pressed controls; color is never the only signal.

## Typography

One system sans stack; no external font requests. Titles are 20px, section headings 15px, body 14px and labels 12px. Exact source uses preserved whitespace. Timing fields and rulers use tabular numbers.

## Layout

Desktop: 240px script rail, flexible timeline, 288px inspector. At 1100px the rails reduce to 190px and 260px. Below 760px the workspace stacks for legible inspection. Timeline scroll stays inside its own container; script rows and clips render a bounded window. Desktop remains the precision editing target.

## Elevation & Depth

Flat tonal regions and hairline separators. Selection uses an inset outline. No decorative elevation or motion.

## Shapes

Controls have 5px corners, clips 4px. Script rows are rectangular. Native selects and numeric fields retain familiar behavior.

## Components

Primary buttons use teal with white text; ordinary buttons use a white surface and neutral border. Focus is a 3px teal outline with 3px offset. Disabled actions reduce opacity. Each clip is a keyboard-focusable button; arrow movement has textual instructions. Errors name recovery in the persistent status region.

## Do's and Don'ts

Preserve exact script text and identity. Keep time and revisions visible. Use explicit actions for replacement and undo. Do not add decorative dashboards, musical grids, hidden ripple, or generation controls to this evaluation surface.
