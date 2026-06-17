import type { ReasoningSummary } from '../api';

type Props = {
  reasoning: ReasoningSummary | null;
  isLoading: boolean;
  mode: string;
  streamingAnswer?: string | null;
  reasoningTokens?: string | null;
};

export default function ProcessTrace({ reasoning, isLoading, mode, streamingAnswer, reasoningTokens }: Props) {
  const summaryItems = reasoning?.reasoning_summary ?? [];
  const sourceNotes = reasoning?.source_notes ?? [];
  const limitations = reasoning?.limitations ?? [];

  const isStreamingActive = isLoading && mode === 'ai_answer';
  const hasReasoning = summaryItems.length > 0 || sourceNotes.length > 0 || limitations.length > 0;

  return (
    <section className="trace-panel">
      {reasoningTokens && (
        <details className="reasoning reasoning-raw" open={true}>
          <summary>
            <h4>Model reasoning (raw)</h4>
          </summary>
          <pre className="reasoning-raw-content">{reasoningTokens}</pre>
        </details>
      )}

      {mode === 'ai_answer' && hasReasoning && (
        <details className="reasoning" open={false}>
          <summary>
            <h4>Model reasoning</h4>
            {isLoading && <span className="badge">running</span>}
          </summary>

          {summaryItems.length > 0 && (
            <>
              <h5>Reasoning summary</h5>
              <ul>
                {summaryItems.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
          {sourceNotes.length > 0 && (
            <>
              <h5>Source notes</h5>
              <ul>
                {sourceNotes.map((note, idx) => {
                  const source = String(note.source ?? 'unknown');
                  const usedFor = String(note.used_for ?? '');
                  return (
                    <li key={`${source}-${idx}`}>
                      <code>{source}</code>
                      {usedFor && <span> — {usedFor}</span>}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
          {limitations.length > 0 && (
            <>
              <h5>Limitations</h5>
              <ul>
                {limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
        </details>
      )}

      {streamingAnswer && (
        <article className="reasoning streaming-answer">
          <h5>{isStreamingActive ? 'Generating answer...' : 'Answer'}</h5>
          <p className={`streaming-content ${isStreamingActive ? 'streaming-active' : ''}`}>
            {streamingAnswer}
          </p>
        </article>
      )}

      {mode === 'quick_search' && (
        <p className="empty">Reasoning is shown only for AI answer mode.</p>
      )}
    </section>
  );
}
