/**
 * Single place the frontend talks to the backend.
 *
 * Previously `http://localhost:8000/api` was hardcoded in nine files across 38 fetch
 * sites, with no timeout and no shared error handling, so the app could only ever
 * point at a dev backend and a hung request hung forever.
 */

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000/api';

/** Requests that outlive this are aborted rather than hanging the UI. */
const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
    readonly path: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

/**
 * Fetch JSON from the API. Throws ApiError on timeout, network failure, or non-2xx.
 * `path` is relative to API_BASE and should start with "/".
 */
export async function fetchJSON<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...init } = options;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // Respect a caller-supplied signal alongside the timeout
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', () => controller.abort(), { once: true });
  }

  try {
    const response = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });

    if (!response.ok) {
      throw new ApiError(`Request failed (${response.status})`, response.status, path);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(`Request timed out after ${timeoutMs}ms`, null, path);
    }
    throw new ApiError(error instanceof Error ? error.message : 'Network error', null, path);
  } finally {
    clearTimeout(timer);
  }
}

/** POST with no body, for the trigger-style endpoints. */
export function postJSON<T>(path: string, options: RequestOptions = {}): Promise<T> {
  return fetchJSON<T>(path, { ...options, method: 'POST' });
}
