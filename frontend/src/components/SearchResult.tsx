type SearchData = {
  answer: string | null;
  reasoning_content?: string | null;
  chunks: Array<{
    text?: string;
    metadata?: Record<string, unknown>;
    score?: number;
  }>;
};

type Props = { data: SearchData };

export default function SearchResult({ data }: Props) {
  return (
    <div>
      {data.reasoning_content && (
        <details className="reasoning reasoning-raw" open>
          <summary>
            <h4>Reasoning (chain-of-thought)</h4>
          </summary>
          <pre className="reasoning-raw-content">{data.reasoning_content}</pre>
        </details>
      )}

      {data.answer && (
        <article className="answer">
          <h3>Answer</h3>
          <p>{data.answer}</p>
        </article>
      )}

      {data.chunks.length ? (
        <>
          <h3>Sources</h3>
          <div className="chunks">
            {data.chunks.map((chunk, idx) => {
              const meta = chunk.metadata ?? {};
              const relPath = (meta.rel_path as string) ?? 'unknown';
              const chunkIndex = (meta.chunk_index as number) ?? 0;
              return (
                <details className="chunk" key={idx}>
                  <summary>
                    <code>{relPath}:{chunkIndex}</code>
                    {typeof chunk.score === 'number' && (
                      <span className="score">score: {chunk.score.toFixed(4)}</span>
                    )}
                  </summary>
                  <pre>{chunk.text}</pre>
                </details>
              );
            })}
          </div>
        </>
      ) : (
        <p className="empty">No relevant chunks found.</p>
      )}
    </div>
  );
}
