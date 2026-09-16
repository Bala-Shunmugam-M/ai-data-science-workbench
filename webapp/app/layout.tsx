import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ThemeProvider } from '@/lib/theme'
import './globals.css'

const inter = Inter({
  variable: '--font-inter',
  subsets: ['latin'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Workbench',
  description:
    'Upload, scan, extract, and analyze a dataset end to end, backed by the real Python ML pipeline running on this machine.',
}

// Must match THEME_STORAGE_KEY in lib/theme.tsx. Duplicated here (rather
// than imported) because this file is a Server Component and lib/theme.tsx
// is 'use client' — a server module cannot call a function exported from
// a client module. Runs before hydration so a persisted dark theme (or
// the OS preference, on first visit) is applied before first paint —
// no light-then-dark flash on reload.
const ANTI_FLASH_SCRIPT = `(function(){try{
  var key='workbench-theme';
  var stored=localStorage.getItem(key);
  var theme=stored==='light'||stored==='dark'||stored==='system'?stored:'system';
  var resolved=theme==='system'
    ?(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')
    :theme;
  var root=document.documentElement;
  if(resolved==='dark')root.classList.add('dark');
  root.style.colorScheme=resolved;
}catch(e){}})();`

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: ANTI_FLASH_SCRIPT }} />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  )
}
