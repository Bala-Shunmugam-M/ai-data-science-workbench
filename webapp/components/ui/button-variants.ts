import { cva, type VariantProps } from 'class-variance-authority'

/**
 * Split out of `button.tsx` so Server Components can style a `<Link>` with
 * `buttonVariants({ variant: 'secondary' })`.
 *
 * `button.tsx` carries `'use client'`, and a client module's exports cannot be
 * *called* from the server, only rendered. Since this is a pure string builder
 * with no React in it, it belongs on the boundary-free side.
 */
export const buttonVariants = cva(
  [
    'relative inline-flex items-center justify-center gap-2 whitespace-nowrap',
    'rounded-md font-medium select-none',
    'transition-colors duration-fast ease-standard',
    'outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
    'disabled:pointer-events-none disabled:opacity-50',
    '[&_svg]:shrink-0',
  ].join(' '),
  {
    variants: {
      variant: {
        primary: 'bg-brand text-brand-fg hover:bg-brand-hover active:bg-brand-hover',
        secondary:
          'bg-surface text-text border border-border-strong hover:bg-surface-hover active:bg-bg-subtle',
        ghost:
          'bg-transparent text-text-secondary hover:bg-surface-hover hover:text-text active:bg-bg-subtle',
        destructive:
          'bg-danger-subtle text-danger border border-danger hover:bg-danger hover:text-bg active:bg-danger active:text-bg',
      },
      size: {
        sm: 'h-8 px-3 text-caption [&_svg]:size-4',
        md: 'h-10 px-4 text-body [&_svg]:size-4',
        lg: 'h-12 px-6 text-body-lg [&_svg]:size-5',
      },
    },
    defaultVariants: { variant: 'primary', size: 'md' },
  },
)

export type ButtonVariantProps = VariantProps<typeof buttonVariants>
