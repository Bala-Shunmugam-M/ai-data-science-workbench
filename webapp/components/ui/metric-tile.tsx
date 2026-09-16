import { ArrowDownRight, ArrowUpRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from './cn'
import { Card } from './card'

export type MetricTone = 'neutral' | 'good' | 'warn' | 'bad'

const VALUE_TONE: Record<MetricTone, string> = {
  neutral: 'text-text',
  good: 'text-success',
  warn: 'text-warning',
  bad: 'text-danger',
}

export interface MetricTileProps {
  label: string
  /** Pre-formatted. Use lib/format.ts (formatMetric / formatNumber / formatPercent). */
  value: ReactNode
  tone?: MetricTone
  /** One short line under the value explaining what it means. */
  hint?: string
  trend?: { direction: 'up' | 'down'; label: string }
  className?: string
}

/** Headline number in the `metric` type scale (48/56, 700, tabular-nums). */
export function MetricTile({
  label,
  value,
  tone = 'neutral',
  hint,
  trend,
  className,
}: MetricTileProps) {
  const TrendIcon = trend?.direction === 'down' ? ArrowDownRight : ArrowUpRight
  return (
    <Card className={cn('flex flex-col gap-1', className)}>
      <span className="text-label text-text-tertiary uppercase tracking-wide">
        {label}
      </span>
      <span
        className={cn(
          'text-metric break-words',
          VALUE_TONE[tone],
        )}
      >
        {value}
      </span>
      {trend && (
        <span
          className={cn(
            'inline-flex items-center gap-1 text-caption',
            trend.direction === 'up' ? 'text-success' : 'text-danger',
          )}
        >
          <TrendIcon aria-hidden className="size-4" />
          {trend.label}
        </span>
      )}
      {hint && <span className="text-caption text-text-secondary">{hint}</span>}
    </Card>
  )
}
