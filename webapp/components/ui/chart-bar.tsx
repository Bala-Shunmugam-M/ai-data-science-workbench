'use client'

import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis } from 'recharts'
import {
  AXIS_STYLE,
  ChartFrame,
  LEGEND_STYLE,
  MAX_SERIES,
  TOOLTIP_STYLE,
  seriesColor,
  type ChartDatum,
  type ChartSeries,
} from './chart-common'

export interface ChartBarProps {
  data: ChartDatum[]
  /** Key holding the category name (rendered down the left edge). */
  xKey: string
  /** Max 6; extras are dropped. */
  series: ChartSeries[]
  /** Required plain-language line rendered under the chart. */
  caption: string
  height?: number
  /** Width reserved for category labels. Default 110. */
  categoryWidth?: number
  className?: string
}

/** Horizontal bars only (§1.6). Flat fills — no gradients, no shadows. */
export function ChartBar({
  data,
  xKey,
  series,
  caption,
  height = 260,
  categoryWidth = 110,
  className,
}: ChartBarProps) {
  const shown = series.slice(0, MAX_SERIES)
  return (
    <ChartFrame
      caption={caption}
      height={height}
      labelKey={xKey}
      labelHeader={xKey}
      series={shown}
      data={data}
      className={className}
    >
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 12, bottom: 4, left: 0 }}
        barCategoryGap="20%"
      >
        <CartesianGrid horizontal={false} stroke="var(--border)" />
        <XAxis
          type="number"
          stroke={AXIS_STYLE.stroke}
          tick={AXIS_STYLE.tick}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey={xKey}
          width={categoryWidth}
          stroke={AXIS_STYLE.stroke}
          tick={AXIS_STYLE.tick}
          tickLine={false}
        />
        <Tooltip {...TOOLTIP_STYLE} />
        {shown.length > 1 && (
          <Legend wrapperStyle={LEGEND_STYLE} verticalAlign="bottom" />
        )}
        {shown.map((s, i) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.label}
            fill={seriesColor(i)}
            radius={[0, 4, 4, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ChartFrame>
  )
}
