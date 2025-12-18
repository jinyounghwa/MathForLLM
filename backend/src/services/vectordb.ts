import Vectra from "vectra";
import * as fs from "fs/promises";
import * as path from "path";
import { v4 as uuidv4 } from "uuid";

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

interface StoredDocument {
  id: string;
  vector: number[];
  metadata: DocumentChunk["metadata"];
  data: string;
}

const VECTRA_DB_PATH = "/tmp/vectra_db";
const INDEX_FILE = path.join(VECTRA_DB_PATH, "documents.json");

let documents: StoredDocument[] = [];

export async function initializeVectorDB() {
  try {
    // Ensure directory exists
    await fs.mkdir(VECTRA_DB_PATH, { recursive: true });

    // Load existing index if it exists
    try {
      const data = await fs.readFile(INDEX_FILE, "utf-8");
      documents = JSON.parse(data);
      console.log(`✓ Vector DB initialized with ${documents.length} existing documents`);
    } catch {
      console.log("✓ Vector DB initialized (empty)");
    }
  } catch (error) {
    console.error("Failed to initialize vector DB:", error);
    throw error;
  }
}

export async function indexDocuments(chunks: DocumentChunk[]) {
  try {
    // Add new documents
    for (const chunk of chunks) {
      documents.push({
        id: chunk.id,
        vector: chunk.embedding,
        metadata: chunk.metadata,
        data: chunk.content,
      });
    }

    // Save to file
    await fs.writeFile(INDEX_FILE, JSON.stringify(documents, null, 2));
    console.log(`✓ Indexed ${chunks.length} document chunks (total: ${documents.length})`);
  } catch (error) {
    console.error("Failed to index documents:", error);
    throw error;
  }
}

export async function searchDocuments(query: string, embedding: number[], topK: number = 3) {
  try {
    if (documents.length === 0) {
      return [];
    }

    // Compute similarity scores using cosine similarity
    const results = documents
      .map((doc) => ({
        id: doc.id,
        content: doc.data,
        metadata: doc.metadata,
        similarity: cosineSimilarity(embedding, doc.vector),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);

    return results.map((r) => ({
      id: r.id,
      content: r.content,
      relevance: Math.max(0, r.similarity),
      metadata: r.metadata,
    }));
  } catch (error) {
    console.error("Failed to search documents:", error);
    return [];
  }
}

export async function saveVectorIndex() {
  try {
    // Already saved in indexDocuments
    console.log("✓ Vector index saved");
  } catch (error) {
    console.error("Failed to save vector index:", error);
    throw error;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) return 0;

  const minLength = Math.min(a.length, b.length);
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < minLength; i++) {
    dotProduct += a[i] * b[i];
    magnitudeA += a[i] * a[i];
    magnitudeB += b[i] * b[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) return 0;

  return dotProduct / (magnitudeA * magnitudeB);
}
