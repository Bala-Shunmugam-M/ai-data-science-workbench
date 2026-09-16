import type { ReactNode } from 'react'

/**
 * One title per screen, styled in one place.
 *
 * `actions` sits top-right on desktop and wraps under the title on narrow
 * screens rather than squeezing the heading. The title is what orients you.
 */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-h1 text-text">{title}</h1>
        {description && <p className="mt-2 max-w-2xl text-body text-text-secondary">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}
