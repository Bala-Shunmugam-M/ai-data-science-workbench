import { clsx, type ClassValue } from "clsx"
import { extendTailwindMerge } from "tailwind-merge"

/**
 * The nine type-scale roles in globals.css are custom `@utility` classes named
 * `text-metric` … `text-label`. Stock tailwind-merge does not know them, so it
 * files them under `text-color` alongside `text-success` / `text-text` and keeps
 * only the last one — `cn('text-metric', 'text-success')` silently returned just
 * `text-success`, dropping the 48px/700 type entirely.
 *
 * Declaring the nine as font-size utilities puts them in their own group, so a
 * size and a colour coexist while size-vs-size and colour-vs-colour still
 * override correctly. This list must stay in sync with the `@utility` blocks in
 * `app/globals.css`; `lib/utils.test.ts` fails if it drifts.
 */
const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      "font-size": [
        "text-metric",
        "text-display",
        "text-h1",
        "text-h2",
        "text-h3",
        "text-body-lg",
        "text-body",
        "text-caption",
        "text-label",
      ],
    },
  },
})

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
