'use client'

/**
 * Sidebar + topbar, and the only place page chrome is decided.
 *
 * Collapse state persists in localStorage but is read in an effect, not in the
 * initial state: reading it during render would make the server and the first
 * client render disagree, which is exactly the hydration mismatch the theme
 * toggle had to be fixed for. The sidebar therefore always renders expanded on
 * the server and settles a frame later.
 *
 * Below 1024px the sidebar becomes a slide-over rather than a rail. A permanent
 * sidebar on a phone would eat the width the results table needs.
 */
import { ChevronLeft, ChevronRight, Menu, Search, Settings, Upload, X } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useCallback, useEffect, useState } from 'react'

import { ThemeToggle, cn } from '@/components/ui'

import { NAV_ITEMS, isActive } from './nav-items'

const STORAGE_KEY = 'workbench-sidebar-collapsed'

const ICONS = {
  '/': Upload,
  '/settings': Settings,
} as const

function NavLinks({
  pathname,
  collapsed,
  onNavigate,
}: {
  pathname: string
  collapsed: boolean
  onNavigate?: () => void
}) {
  return (
    <nav aria-label="Main" className="flex flex-col gap-1 p-3">
      {NAV_ITEMS.map((item) => {
        const active = isActive(item, pathname)
        const Icon = ICONS[item.href as keyof typeof ICONS]
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            aria-current={active ? 'page' : undefined}
            title={collapsed ? item.label : undefined}
            className={cn(
              'flex items-center gap-3 rounded-md px-3 py-2 text-body transition-colors duration-fast ease-standard',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
              active
                ? 'bg-brand-subtle font-medium text-brand'
                : 'text-text-secondary hover:bg-surface-hover hover:text-text',
              collapsed && 'justify-center px-0',
            )}
          >
            <Icon className="size-5 shrink-0" />
            <span className={cn(collapsed && 'sr-only')}>{item.label}</span>
          </Link>
        )
      })}
    </nav>
  )
}

function Wordmark({ collapsed }: { collapsed: boolean }) {
  return (
    <Link
      href="/"
      className={cn(
        'flex items-center gap-2.5 rounded-md px-3 py-2',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring',
        collapsed && 'justify-center px-0',
      )}
    >
      <span
        aria-hidden
        className="grid size-8 shrink-0 place-items-center rounded-md bg-brand text-label text-brand-fg"
      >
        W
      </span>
      <span className={cn('text-body font-semibold text-text', collapsed && 'sr-only')}>
        Workbench
      </span>
    </Link>
  )
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)

  useEffect(() => {
    setCollapsed(window.localStorage.getItem(STORAGE_KEY) === '1')
  }, [])

  const toggleCollapsed = useCallback(() => {
    setCollapsed((previous) => {
      const next = !previous
      window.localStorage.setItem(STORAGE_KEY, next ? '1' : '0')
      return next
    })
  }, [])

  // Route change closes the drawer. Leaving it open over the new page is the
  // classic mobile-nav bug.
  useEffect(() => setDrawerOpen(false), [pathname])

  useEffect(() => {
    if (!drawerOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDrawerOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [drawerOpen])

  return (
    <div className="flex min-h-dvh bg-bg">
      <aside
        className={cn(
          'hidden shrink-0 flex-col border-r border-border bg-surface lg:flex',
          'transition-[width] duration-base ease-standard',
          collapsed ? 'w-[72px]' : 'w-60',
        )}
      >
        <div className="flex h-16 items-center border-b border-border px-3">
          <Wordmark collapsed={collapsed} />
        </div>
        <NavLinks pathname={pathname} collapsed={collapsed} />
        <button
          type="button"
          onClick={toggleCollapsed}
          aria-expanded={!collapsed}
          className={cn(
            'mt-auto flex items-center gap-3 border-t border-border px-4 py-3 text-caption',
            'text-text-tertiary transition-colors duration-fast hover:text-text',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus-ring',
            collapsed && 'justify-center px-0',
          )}
        >
          {collapsed ? (
            <ChevronRight className="size-5" />
          ) : (
            <>
              <ChevronLeft className="size-5" />
              <span>Collapse</span>
            </>
          )}
          <span className="sr-only">{collapsed ? 'Expand sidebar' : 'Collapse sidebar'}</span>
        </button>
      </aside>

      {drawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setDrawerOpen(false)}
            className="absolute inset-0 bg-black/40"
          />
          <div className="relative flex h-full w-64 flex-col border-r border-border bg-surface">
            <div className="flex h-16 items-center justify-between border-b border-border px-3">
              <Wordmark collapsed={false} />
              <button
                type="button"
                onClick={() => setDrawerOpen(false)}
                aria-label="Close navigation"
                className="rounded-md p-2 text-text-secondary hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
              >
                <X className="size-5" />
              </button>
            </div>
            <NavLinks pathname={pathname} collapsed={false} onNavigate={() => setDrawerOpen(false)} />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-3 border-b border-border bg-bg px-4 lg:px-8">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open navigation"
            className="rounded-md p-2 text-text-secondary hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring lg:hidden"
          >
            <Menu className="size-5" />
          </button>

          <div className="relative hidden max-w-sm flex-1 md:block">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-text-tertiary" />
            <input
              type="search"
              placeholder="Search is not wired up yet"
              aria-label="Search projects"
              // Disabled on purpose. A search box that silently does nothing is
              // worse than one that admits it, and wiring it is a later call.
              disabled
              className="w-full rounded-md border border-border bg-bg-subtle py-2 pl-9 pr-3 text-caption text-text placeholder:text-text-tertiary disabled:cursor-not-allowed"
            />
          </div>

          <div className="ml-auto flex items-center gap-2">
            <ThemeToggle />
          </div>
        </header>

        <main className="mx-auto w-full max-w-content flex-1 px-4 py-8 lg:px-8">{children}</main>
      </div>
    </div>
  )
}
