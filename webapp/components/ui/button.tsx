'use client'

import { Loader2 } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'

import { buttonVariants, type ButtonVariantProps } from './button-variants'
import { cn } from './cn'

// ponytail: hand-rolled rather than `npx shadcn add button`. The shadcn
// CLI rewrites app/globals.css and package.json, both owned by another
// package, and its variant/size names (default|outline|link, default|icon)
// don't match the ones WEBAPP_PLAN.md §4 specifies anyway.

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    ButtonVariantProps {
  /** Shows a spinner and disables the button. Width does not change. */
  loading?: boolean
  /** Leading icon. Hidden while `loading`. */
  icon?: ReactNode
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant, size, loading, icon, children, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {/* Content stays in flow (invisible) so the button keeps its width. */}
      <span className={cn('inline-flex items-center gap-2', loading && 'invisible')}>
        {icon}
        {children}
      </span>
      {loading && <Loader2 aria-hidden className="absolute animate-spin" />}
    </button>
  )
})

export { buttonVariants }
