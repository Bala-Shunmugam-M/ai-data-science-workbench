import { forwardRef, type HTMLAttributes } from 'react'
import { cn } from './cn'

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** 24px padding (§2.2 card padding). Default true. */
  padded?: boolean
  /** Hover/focus affordance for cards that behave like links or buttons. */
  interactive?: boolean
}

/**
 * --surface, `lg` radius, `sm` shadow. In dark mode --shadow-sm is
 * redefined by globals.css as a 1px --border ring, so the same class
 * gives the plan's "border instead of shadow" behaviour for free.
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(function Card(
  { className, padded = true, interactive, ...props },
  ref,
) {
  return (
    <div
      ref={ref}
      className={cn(
        'rounded-lg bg-surface text-text shadow-[var(--shadow-sm)]',
        padded && 'p-6',
        interactive && [
          'cursor-pointer transition-colors duration-fast ease-standard',
          'hover:bg-surface-hover active:bg-bg-subtle',
          'outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
        ],
        className,
      )}
      {...props}
    />
  )
})

export function CardHeader({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('mb-4 flex flex-col gap-1', className)} {...props} />
}

export function CardTitle({
  className,
  ...props
}: HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn('text-h3 text-text', className)} {...props} />
}

export function CardDescription({
  className,
  ...props
}: HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn('text-caption text-text-secondary', className)} {...props} />
  )
}
