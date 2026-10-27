import type { SearchData } from '../api';

type Props = { data: SearchData };

export default function ProcessTrace({ data }: Props) {
  if (!data.chunks.length) {
    return <p className="empty">No relevant chunks found.</p>;
  }

  return (
    <div>
      {data.answer && (
        <article className="answer">
          <h3>Answer</h3>
          <p>{data.answer}</p>
        </article>
      )}

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
    </div>
  );
}
