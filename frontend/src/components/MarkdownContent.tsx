import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type Props = {
  content: string;
  className?: string;
};

export default function MarkdownContent({ content, className }: Props) {
  return (
    <div className={className ? `markdown-content ${className}` : 'markdown-content'}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
