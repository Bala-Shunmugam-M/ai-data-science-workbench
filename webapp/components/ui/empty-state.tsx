import type { ReactNode } from 'react'
import { cn } from './cn'

export interface EmptyStateProps {
  /** A lucide icon element, e.g. <FileSearch />. Sized by this component. */
  icon?: ReactNode
  title: string
  /** One or two sentences saying what to do next — never a bare "no data". */
  body: string
  /** Usually a <Button>. */
  action?: ReactNode
  className?: string
}

export function EmptyState({
  icon,
  title,
  body,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border-strong bg-bg-subtle px-6 py-12 text-center',
        className,
      )}
    >
      {icon && (
        <span
          aria-hidden
          className="text-text-tertiary [&_svg]:size-8 [&_svg]:stroke-[1.5]"
        >
          {icon}
        </span>
      )}
      <h3 className="text-h3 text-text">{title}</h3>
      <p className="max-w-prose text-body text-text-secondary">{body}</p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}
