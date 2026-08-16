import fs from "node:fs";
import path from "node:path";
import { marked } from "marked";

/** Lee docs/metodologia.md (raiz del proyecto, un nivel arriba de panel/) y lo
 * convierte a HTML en build. Contenido propio y confiable (no input de
 * usuario): dangerouslySetInnerHTML es seguro aqui. */
function renderMetodologia(): string {
  const mdPath = path.join(process.cwd(), "..", "docs", "metodologia.md");
  const md = fs.readFileSync(mdPath, "utf-8");
  return marked.parse(md, { async: false }) as string;
}

export default function MetodologiaPage() {
  const html = renderMetodologia();
  return (
    <div className="mx-auto max-w-2xl">
      <div className="prose-metodologia" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
