import { Check, Loader2, X } from 'lucide-react'
import { cn } from './cn'

export type StepState = 'pending' | 'active' | 'done' | 'failed'

export interface ProgressStep {
  id: string
  label: string
  state: StepState
  /** Optional one-liner under the label — e.g. the failure reason. */
  note?: string
}

export interface ProgressStepsProps {
  steps: ProgressStep[]
  /** Marks the current step for assistive tech (`aria-current="step"`). */
  activeId?: string
  /**
   * `responsive` (default) stacks on mobile and goes horizontal from `md`.
   *
   * Pass `vertical` when the component sits in a narrow column: five steps
   * sharing 420px leaves ~84px each, which is not enough for a label like
   * "Scanning document" and makes adjacent labels visually run together.
   * Width here is a property of the *container*, which a media query cannot
   * see, so it has to be the caller's decision.
   */
  orientation?: 'responsive' | 'vertical'
  className?: string
}

const MARKER: Record<StepState, string> = {
  pending: 'border-border-strong bg-surface text-text-tertiary',
  active: 'border-brand bg-brand-subtle text-brand',
  done: 'border-success bg-success-subtle text-success',
  // Solid fill: the failed step must not be mistakable for any other state.
  failed: 'border-danger bg-danger text-bg',
}

const LABEL: Record<StepState, string> = {
  pending: 'text-text-tertiary',
  active: 'text-text font-medium',
  done: 'text-text-secondary',
  failed: 'text-danger font-medium',
}

const CONNECTOR: Record<StepState, string> = {
  pending: 'bg-border',
  active: 'bg-border',
  done: 'bg-success',
  failed: 'bg-danger',
}

const STATE_WORD: Record<StepState, string> = {
  pending: 'Pending',
  active: 'In progress',
  done: 'Done',
  failed: 'Failed',
}

export function ProgressSteps({
  steps,
  activeId,
  orientation = 'responsive',
  className,
}: ProgressStepsProps) {
  const horizontal = orientation === 'responsive'
  return (
    <ol className={cn('flex flex-col', horizontal && 'md:flex-row', className)}>
      {steps.map((step, i) => {
        const isLast = i === steps.length - 1
        return (
          <li
            key={step.id}
            aria-current={step.id === activeId ? 'step' : undefined}
            className={cn(
              'flex gap-3',
              horizontal && 'md:min-w-0 md:flex-1 md:flex-col md:gap-2',
              !isLast && (horizontal ? 'pb-5 md:pb-0' : 'pb-5'),
            )}
          >
            {/* Marker + connector track. Column when stacked, row when not. */}
            <div
              className={cn(
                'flex flex-col items-center',
                horizontal && 'md:w-full md:flex-row',
              )}
            >
              <span
                className={cn(
                  'flex size-8 shrink-0 items-center justify-center rounded-full border-2',
                  MARKER[step.state],
                )}
              >
                {step.state === 'done' && (
                  <Check aria-hidden className="size-4" />
                )}
                {step.state === 'failed' && <X aria-hidden className="size-4" />}
                {step.state === 'active' && (
                  <Loader2 aria-hidden className="size-4 animate-spin" />
                )}
                {step.state === 'pending' && (
                  <span className="text-label">{i + 1}</span>
                )}
              </span>
              {!isLast && (
                <span
                  aria-hidden
                  className={cn(
                    'mt-1 w-0.5 grow rounded-full',
                    horizontal && 'md:mt-0 md:ml-2 md:h-0.5 md:w-auto',
                    CONNECTOR[step.state],
                  )}
                />
              )}
            </div>

            <div className={cn('min-w-0 flex-1', horizontal && 'md:pr-6')}>
              <p className={cn('text-body break-words', LABEL[step.state])}>{step.label}</p>
              <p
                className={cn(
                  // A failure note carries a Python exception message. Clamped
                  // so a long traceback cannot stretch the step out of shape —
                  // the full text is shown in the failure card below the list.
                  'text-caption break-words',
                  step.note && step.state === 'failed' && 'line-clamp-2',
                  step.state === 'failed' ? 'text-danger' : 'text-text-tertiary',
                )}
                title={step.note}
              >
                {step.note ?? STATE_WORD[step.state]}
              </p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
