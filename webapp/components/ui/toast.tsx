'use client'

import {
  AlertTriangle,
  CheckCircle2,
  Info,
  X,
  XCircle,
} from 'lucide-react'
import { useCallback, useEffect, useRef, type ReactNode } from 'react'
import { cn } from './cn'

export type ToastTone = 'neutral' | 'good' | 'warn' | 'bad'

const TONE: Record<ToastTone, { border: string; icon: string; Icon: typeof Info }> =
  {
    neutral: { border: 'border-l-border-strong', icon: 'text-text-tertiary', Icon: Info },
    good: { border: 'border-l-success', icon: 'text-success', Icon: CheckCircle2 },
    warn: { border: 'border-l-warning', icon: 'text-warning', Icon: AlertTriangle },
    bad: { border: 'border-l-danger', icon: 'text-danger', Icon: XCircle },
  }

export interface ToastProps {
  tone?: ToastTone
  title: string
  body?: ReactNode
  /** ms until auto-dismiss. 0 disables it. Default 5000. */
  duration?: number
  onDismiss?: () => void
  className?: string
}

/**
 * Auto-dismisses after `duration`, pausing while hovered or focused.
 * No portal: render inside <ToastRegion>, which owns the live region.
 */
export function Toast({
  tone = 'neutral',
  title,
  body,
  duration = 5000,
  onDismiss,
  className,
}: ToastProps) {
  const { border, icon, Icon } = TONE[tone]
  const remaining = useRef(duration)
  const startedAt = useRef(0)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const pause = useCallback(() => {
    if (timer.current === null) return
    clearTimeout(timer.current)
    timer.current = null
    remaining.current -= Date.now() - startedAt.current
  }, [])

  const resume = useCallback(() => {
    if (!duration || timer.current !== null || !onDismiss) return
    startedAt.current = Date.now()
    timer.current = setTimeout(onDismiss, Math.max(0, remaining.current))
  }, [duration, onDismiss])

  useEffect(() => {
    resume()
    return pause
  }, [resume, pause])

  return (
    <div
      onMouseEnter={pause}
      onMouseLeave={resume}
      onFocus={pause}
      onBlur={resume}
      className={cn(
        'pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-lg border-l-4 bg-surface p-4 shadow-[var(--shadow-lg)]',
        'animate-fade-in',
        border,
        className,
      )}
    >
      <Icon aria-hidden className={cn('mt-0.5 size-5 shrink-0', icon)} />
      <div className="min-w-0 flex-1">
        <p className="text-body font-medium text-text">{title}</p>
        {body && <div className="text-caption text-text-secondary">{body}</div>}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss notification"
          className={cn(
            'rounded-sm p-0.5 text-text-tertiary',
            'transition-colors duration-fast ease-standard',
            'hover:bg-surface-hover hover:text-text active:bg-bg-subtle',
            'outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
          )}
        >
          <X aria-hidden className="size-4" />
        </button>
      )}
    </div>
  )
}

/**
 * Fixed, bottom-right stack. Announces its children politely — mount it
 * once per screen and render <Toast> elements inside it.
 */
export function ToastRegion({
  children,
  className,
}: {
  children?: ReactNode
  className?: string
}) {
  return (
    <div
      aria-live="polite"
      aria-relevant="additions text"
      className={cn(
        'pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col items-end gap-2',
        className,
      )}
    >
      {children}
    </div>
  )
}
