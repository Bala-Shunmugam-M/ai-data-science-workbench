import type { HTMLAttributes } from 'react'
import { cn } from './cn'

const RADII = {
  sm: 'rounded-sm',
  md: 'rounded-md',
  lg: 'rounded-lg',
  xl: 'rounded-xl',
  full: 'rounded-full',
} as const

export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  /** Any CSS width, e.g. '100%', '12rem', 240. */
  w?: string | number
  /** Any CSS height. Defaults to 1rem. */
  h?: string | number
  radius?: keyof typeof RADII
}

/**
 * The default loading affordance. Uses Tailwind's `animate-pulse`; the
 * global prefers-reduced-motion rule in globals.css neutralises it.
 */
export function Skeleton({
  className,
  w,
  h = '1rem',
  radius = 'md',
  style,
  ...props
}: SkeletonProps) {
  return (
    <div
      aria-hidden
      className={cn('animate-pulse bg-border', RADII[radius], className)}
      style={{ width: w, height: h, ...style }}
      {...props}
    />
  )
}
