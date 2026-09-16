/**
 * Guard for a failure that produces no error, just wrong-looking text.
 *
 * If a type-scale utility is added to globals.css but not to the font-size
 * class group in utils.ts, tailwind-merge silently drops it whenever it meets a
 * text colour — the element renders at the default 16px/400 and nothing warns.
 *
 * Run: node lib/utils.test.ts     (Node >=23.6 strips the types natively)
 */
import assert from 'node:assert/strict'

import { cn } from './utils.ts'

const SIZES = [
  'text-metric',
  'text-display',
  'text-h1',
  'text-h2',
  'text-h3',
  'text-body-lg',
  'text-body',
  'text-caption',
  'text-label',
]

// A size and a colour must coexist.
for (const size of SIZES) {
  const result = cn(size, 'text-success')
  assert.ok(result.includes(size), `${size} was dropped when combined with a colour: "${result}"`)
  assert.ok(result.includes('text-success'), `text-success was dropped by ${size}: "${result}"`)
}

// Size still beats size, colour still beats colour — the override behaviour
// tailwind-merge exists for must survive the extension.
assert.equal(cn('text-h1', 'text-h2'), 'text-h2')
assert.equal(cn('text-success', 'text-danger'), 'text-danger')

console.log(`ok — ${SIZES.length} type-scale utilities survive colour merging`)
