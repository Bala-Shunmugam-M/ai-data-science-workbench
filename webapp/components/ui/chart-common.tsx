'use client'

import type { ReactElement, ReactNode } from 'react'
import { ResponsiveContainer } from 'recharts'
import { cn } from './cn'

export interface ChartSeries {
  /** Key into each datum. */
  key: string
  label: string
}

export type ChartDatum = Record<string, string | number>

/**
 * The §2.1 categorical ramp, max 6, referenced as CSS variables rather
 * than resolved values. SVG presentation attributes accept `var(...)`,
 * so a theme switch recolours every mark with no JS and no re-render.
 */
export const CHART_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
  'var(--chart-6)',
] as const

export const MAX_SERIES = CHART_COLORS.length

export function seriesColor(index: number): string {
  return CHART_COLORS[index % CHART_COLORS.length]
}

export const AXIS_STYLE = {
  stroke: 'var(--border-strong)',
  fontSize: 12,
  tick: { fill: 'var(--text-secondary)', fontSize: 12 },
} as const

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--text)',
    fontSize: 14,
    boxShadow: 'var(--shadow-md)',
  },
  labelStyle: { color: 'var(--text-secondary)', fontSize: 12 },
  itemStyle: { color: 'var(--text)' },
  cursor: { fill: 'var(--surface-hover)', stroke: 'var(--border-strong)' },
} as const

export const LEGEND_STYLE = {
  fontSize: 12,
  color: 'var(--text-secondary)',
} as const

/**
 * Shared chrome for every chart: a responsive plot area, the mandatory
 * one-line caption, and a visually-hidden data table so the numbers are
 * available to screen readers rather than locked inside the SVG.
 */
export function ChartFrame({
  caption,
  height = 260,
  labelKey,
  labelHeader,
  series,
  data,
  className,
  children,
}: {
  caption: string
  height?: number
  labelKey: string
  labelHeader: string
  series: ChartSeries[]
  data: ChartDatum[]
  className?: string
  children: ReactElement
}) {
  return (
    <figure className={cn('flex flex-col gap-2', className)}>
      <div style={{ height }} className="w-full">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
      <figcaption className="text-caption text-text-secondary">
        {caption}
      </figcaption>
      {/* A <table> ignores sr-only's width:1px (table layout sizes to
          content) and would widen the page, so clip it with a div. */}
      <div className="sr-only">
        <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">{labelHeader}</th>
            {series.map((s) => (
              <th key={s.key} scope="col">
                {s.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <th scope="row">{String(row[labelKey] ?? '')}</th>
              {series.map((s) => (
                <td key={s.key}>{String(row[s.key] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
        </table>
      </div>
    </figure>
  )
}

export function ChartLegendText({ children }: { children: ReactNode }) {
  return <span className="text-caption text-text-secondary">{children}</span>
}
