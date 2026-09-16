import { cva, type VariantProps } from 'class-variance-authority'
import type { HTMLAttributes } from 'react'
import { cn } from './cn'

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-full font-medium whitespace-nowrap [&_svg]:size-3 [&_svg]:shrink-0',
  {
    variants: {
      tone: {
        neutral: 'bg-bg-subtle text-text-secondary',
        good: 'bg-success-subtle text-success',
        warn: 'bg-warning-subtle text-warning',
        bad: 'bg-danger-subtle text-danger',
      },
      size: {
        sm: 'text-label px-2 py-0.5',
        md: 'text-caption px-2.5 py-1',
      },
    },
    defaultVariants: { tone: 'neutral', size: 'sm' },
  },
)

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

/** Subtle background + solid semantic foreground. Status only, never chrome. */
export function Badge({ className, tone, size, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ tone, size }), className)} {...props} />
  )
}

export { badgeVariants }
