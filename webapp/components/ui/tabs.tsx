'use client'

import { useId, useState } from 'react'

import { cn } from './cn'

export interface TabItem {
  id: string
  label: string
  /** Small count hint shown after the label. */
  hint?: string
  content: React.ReactNode
}

/**
 * Minimal tabs, keyboard-operable.
 *
 * Exists because a stage that shows raw, processed, train, validation, test
 * and engineered previews as six stacked tables is a wall, not a page — the
 * Streamlit original tabs them for the same reason. All panels are rendered
 * and the inactive ones are hidden, so this costs nothing on the server and
 * switching is instant.
 *
 * Arrow keys move between tabs, matching the WAI-ARIA tabs pattern; Home/End
 * jump to the ends.
 */
export function Tabs({ items, className }: { items: TabItem[]; className?: string }) {
  const [active, setActive] = useState(items[0]?.id)
  const base = useId()

  if (!items.length) return null

  const onKeyDown = (event: React.KeyboardEvent, index: number) => {
    const keys: Record<string, number> = {
      ArrowRight: index + 1,
      ArrowLeft: index - 1,
      Home: 0,
      End: items.length - 1,
    }
    const next = keys[event.key]
    if (next === undefined) return
    event.preventDefault()
    const wrapped = (next + items.length) % items.length
    setActive(items[wrapped].id)
    document.getElementById(`${base}-tab-${items[wrapped].id}`)?.focus()
  }

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      <div
        role="tablist"
        aria-label="Dataset previews"
        className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1"
      >
        {items.map((item, index) => {
          const selected = item.id === active
          return (
            <button
              key={item.id}
              id={`${base}-tab-${item.id}`}
              role="tab"
              type="button"
              aria-selected={selected}
              aria-controls={`${base}-panel-${item.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => setActive(item.id)}
              onKeyDown={(event) => onKeyDown(event, index)}
              className={cn(
                'shrink-0 rounded-md px-3 py-2 text-caption whitespace-nowrap transition-colors duration-fast',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring',
                selected
                  ? 'bg-brand-subtle font-medium text-brand'
                  : 'text-text-secondary hover:bg-surface-hover hover:text-text',
              )}
            >
              {item.label}
              {item.hint && (
                <span className="ml-2 text-label text-text-tertiary">{item.hint}</span>
              )}
            </button>
          )
        })}
      </div>

      {items.map((item) => (
        <div
          key={item.id}
          id={`${base}-panel-${item.id}`}
          role="tabpanel"
          aria-labelledby={`${base}-tab-${item.id}`}
          hidden={item.id !== active}
        >
          {item.id === active && item.content}
        </div>
      ))}
    </div>
  )
}
