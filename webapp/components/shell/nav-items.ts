/**
 * Two destinations, deliberately.
 *
 * The nine pipeline stages are *not* here: they belong to a project and are
 * navigated by the stage rail inside it. Putting them in the global sidebar
 * would offer nine dead links whenever no project is open, which is the
 * structure this redesign replaced.
 */
export interface NavItem {
  href: string
  label: string
  matches?: (pathname: string) => boolean
}

export const NAV_ITEMS: NavItem[] = [
  {
    href: '/',
    label: 'New analysis',
    // A project's journey still belongs to the analysis flow.
    matches: (pathname) =>
      pathname === '/' || pathname.startsWith('/project') || pathname.startsWith('/results'),
  },
  { href: '/settings', label: 'Settings', matches: (pathname) => pathname.startsWith('/settings') },
]

export function isActive(item: NavItem, pathname: string): boolean {
  return item.matches ? item.matches(pathname) : pathname === item.href
}
