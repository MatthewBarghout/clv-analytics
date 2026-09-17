import { useEffect, useRef } from 'react';

/**
 * Run `callback` immediately, then on an interval — but only while the tab is visible.
 *
 * The Markets and Pred Markets tabs previously polled every 60s regardless of whether
 * anyone was looking, so a backgrounded dashboard kept hitting the API all day. When
 * the tab is hidden the timer is torn down; on return it fires once straight away so
 * the view is never stale.
 */
export function usePolling(callback: () => void, intervalMs: number): void {
  const savedCallback = useRef(callback);
  savedCallback.current = callback;

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;

    const stop = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    };

    const start = () => {
      stop();
      savedCallback.current();
      timer = setInterval(() => savedCallback.current(), intervalMs);
    };

    const onVisibilityChange = () => {
      if (document.hidden) stop();
      else start();
    };

    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [intervalMs]);
}
