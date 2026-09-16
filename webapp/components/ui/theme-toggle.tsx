'use client'

import { Monitor, Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTheme, type Theme } from '@/lib/theme'
import { cn } from './cn'

const OPTIONS: { value: Theme; label: string; Icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'System', Icon: Monitor },
]

/** Three-way segmented control: light / dark / system. */
export function ThemeToggle({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme()

  // ThemeProvider seeds `theme` from localStorage in a useState initializer,
  // so the server always says 'system' while the client's first render may
  // say 'light'/'dark'. React 19 refuses to patch mismatched attributes, and
  // the selected segment would then be wrong until two theme changes later.
  // Gating on `mounted` (false on the server *and* on the first client
  // render) keeps the two renders identical; the highlight lands one frame in.
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  const current = mounted ? theme : null

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className={cn(
        'inline-flex items-center gap-0.5 rounded-md border border-border bg-bg-subtle p-0.5',
        className,
      )}
    >
      {OPTIONS.map(({ value, label, Icon }) => {
        const selected = current === value
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={label}
            title={label}
            onClick={() => setTheme(value)}
            className={cn(
              'flex size-8 items-center justify-center rounded-sm',
              'transition-colors duration-fast ease-standard',
              'outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
              selected
                ? 'bg-surface text-brand shadow-[var(--shadow-xs)]'
                : 'text-text-tertiary hover:bg-surface-hover hover:text-text active:bg-bg-subtle',
            )}
          >
            <Icon aria-hidden className="size-4" />
          </button>
        )
      })}
    </div>
  )
}
