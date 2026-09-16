import { redirect } from 'next/navigation'

/**
 * Upload is the front door now, not a separate destination.
 *
 * Kept as a redirect rather than deleted so older links and bookmarks still
 * land somewhere sensible.
 */
export default function UploadRedirect() {
  redirect('/')
}
