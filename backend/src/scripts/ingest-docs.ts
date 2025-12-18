import * as fs from "fs/promises";
import * as path from "path";
import { v4 as uuidv4 } from "uuid";
import { initializeVectorDB, indexDocuments } from "../services/vectordb.js";
import { generateEmbedding } from "../services/llm.js";

interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    source: string;
    section: string;
    chapter: number;
    difficulty: string;
  };
  embedding: number[];
}

const DOCS_DIR = path.join(
  "/Users/younghwa.jin/Documents/GitHub/MathForLLM/LLM_math"
);

async function main() {
  try {
    console.log("📚 시작: 문서 인제스션 파이프라인");
    console.log(`📁 문서 디렉토리: ${DOCS_DIR}`);

    // Initialize vector DB
    await initializeVectorDB();

    // Read all markdown files
    const files = await fs.readdir(DOCS_DIR);
    const mdFiles = files.filter((f) => f.endsWith(".md"));

    console.log(`✓ ${mdFiles.length}개의 마크다운 파일 발견`);

    const chunks: DocumentChunk[] = [];

    // Process each file
    for (const file of mdFiles) {
      const filePath = path.join(DOCS_DIR, file);
      const content = await fs.readFile(filePath, "utf-8");

      // Extract chapter number from filename (e.g., "Day_01_수의_체계.md" -> 1)
      const match = file.match(/Day_(\d+)/);
      const chapter = match ? parseInt(match[1]) : 0;

      // Split content by headings
      const sections = content.split(/^#+\s+/m);

      for (let i = 1; i < sections.length; i++) {
        const [title, ...contentLines] = sections[i].split("\n");
        const sectionContent = contentLines.join("\n").trim();

        if (sectionContent.length > 50) {
          // Only process substantial sections
          // Generate embedding for this section
          const embedding = await generateEmbedding(sectionContent);

          const chunk: DocumentChunk = {
            id: uuidv4(),
            content: sectionContent.slice(0, 2000), // Limit chunk size
            metadata: {
              source: file,
              section: title.trim(),
              chapter: chapter,
              difficulty: determineDifficulty(chapter),
            },
            embedding,
          };

          chunks.push(chunk);
        }
      }

      console.log(`✓ 처리됨: ${file} (${chunks.length}개 청크)`);
    }

    // Index all chunks
    if (chunks.length > 0) {
      await indexDocuments(chunks);
      console.log(`\n✅ 완료: ${chunks.length}개 청크 인덱싱됨`);
    } else {
      console.warn("⚠️ 인덱싱할 청크가 없습니다");
    }
  } catch (error) {
    console.error("❌ 오류 발생:", error);
    process.exit(1);
  }
}

function determineDifficulty(chapter: number): string {
  if (chapter <= 10) return "basic";
  if (chapter <= 25) return "intermediate";
  return "advanced";
}

main();
