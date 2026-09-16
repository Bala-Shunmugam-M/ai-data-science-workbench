'use client'

import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from 'recharts'
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

export interface ChartLineProps {
  data: ChartDatum[]
  /** Key holding the x-axis value. */
  xKey: string
  /** Max 6; extras are dropped. */
  series: ChartSeries[]
  /** Required plain-language line rendered under the chart. */
  caption: string
  height?: number
  className?: string
}

export function ChartLine({
  data,
  xKey,
  series,
  caption,
  height = 260,
  className,
}: ChartLineProps) {
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
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={AXIS_STYLE.stroke}
          tick={AXIS_STYLE.tick}
          tickLine={false}
          minTickGap={24}
          interval="preserveStartEnd"
        />
        <YAxis
          stroke={AXIS_STYLE.stroke}
          tick={AXIS_STYLE.tick}
          tickLine={false}
          width={44}
        />
        <Tooltip {...TOOLTIP_STYLE} cursor={{ stroke: 'var(--border-strong)' }} />
        {shown.length > 1 && (
          <Legend wrapperStyle={LEGEND_STYLE} verticalAlign="bottom" />
        )}
        {shown.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={seriesColor(i)}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartFrame>
  )
}
