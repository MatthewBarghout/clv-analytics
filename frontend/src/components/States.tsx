/**
 * The three states every data panel needs, defined once.
 *
 * Each panel used to hand-roll its own spinner and message, which is why the
 * accent colour and sizing drifted between tabs. Styling these in one place also
 * means the visual pass only has to touch one file.
 */

interface LoadingStateProps {
  message?: string;
  /** Fill the viewport instead of sitting inline in a panel. */
  fullScreen?: boolean;
}

export function LoadingState({ message = 'Loading...', fullScreen = false }: LoadingStateProps) {
  const spinner = (
    <>
      <div
        className="animate-spin rounded-full h-8 w-8 border-2 border-line-strong border-t-white/70"
        role="status"
        aria-label="Loading"
      />
      <span className="text-sm text-gray-400">{message}</span>
    </>
  );

  if (fullScreen) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-4">
        {spinner}
      </div>
    );
  }

  return <div className="flex items-center justify-center gap-3 py-16">{spinner}</div>;
}

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
  fullScreen?: boolean;
}

export function ErrorState({ message, onRetry, fullScreen = false }: ErrorStateProps) {
  const body = (
    <div className="text-center max-w-md">
      <div className="text-sm font-medium text-red-400 mb-1">Something went wrong</div>
      <div className="text-sm text-gray-400 mb-4 break-words">{message}</div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-3 py-1.5 text-sm rounded border border-line-strong text-gray-300 hover:bg-panel-raised transition-colors"
        >
          Try again
        </button>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        {body}
      </div>
    );
  }

  return <div className="flex items-center justify-center py-16">{body}</div>;
}

interface EmptyStateProps {
  title: string;
  hint?: string;
}

export function EmptyState({ title, hint }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="text-sm text-gray-400">{title}</div>
      {hint && <div className="text-xs text-gray-600 mt-1 max-w-sm">{hint}</div>}
    </div>
  );
}
