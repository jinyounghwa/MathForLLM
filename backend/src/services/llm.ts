import axios from "axios";
import { enhanceEducationalResponse, validateResponse } from "./response-cleaner.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const MODEL = "qwen2.5:7b";

interface OllamaRequest {
  model: string;
  prompt: string;
  stream: boolean;
}

interface OllamaResponse {
  response: string;
  done: boolean;
}

export async function generateResponse(
  systemPrompt: string,
  userMessage: string,
  context: string,
  history: Array<{ role: string; content: string }> = []
): Promise<string> {
  try {
    // Format conversation history
    let conversationPrompt = systemPrompt + "\n\n";

    if (context) {
      conversationPrompt += `### 참고 자료\n${context}\n\n`;
    }

    // Add previous messages
    for (const msg of history) {
      if (msg.role === "user") {
        conversationPrompt += `사용자: ${msg.content}\n`;
      } else {
        conversationPrompt += `어시스턴트: ${msg.content}\n`;
      }
    }

    // Add current message
    conversationPrompt += `사용자: ${userMessage}\n어시스턴트:`;

    const response = await axios.post(
      `${OLLAMA_URL}/api/generate`,
      {
        model: MODEL,
        prompt: conversationPrompt,
        stream: false,
      } as OllamaRequest,
      { timeout: 60000 }
    );

    // Validate and clean response
    const rawResponse = response.data.response.trim();
    const validationResult = validateResponse(rawResponse);

    if (!validationResult.isValid) {
      console.warn(`Response validation failed: ${validationResult.message}`);
      throw new Error(`LLM response validation failed: ${validationResult.message}`);
    }

    // Enhance and clean the response
    const cleanedResponse = enhanceEducationalResponse(validationResult.cleaned || rawResponse);

    return cleanedResponse;
  } catch (error) {
    console.error("LLM 생성 오류:", error);
    throw new Error("LLM 응답 생성 실패");
  }
}

export async function generateEmbedding(text: string): Promise<number[]> {
  try {
    // For now, use a simple TF-IDF-like embedding
    // In production, you'd use Ollama's embedding model or sentence-transformers
    return generateSimpleEmbedding(text);
  } catch (error) {
    console.error("임베딩 생성 오류:", error);
    return [];
  }
}

// Simple embedding function using TF-IDF-like approach
function generateSimpleEmbedding(text: string): number[] {
  const words = text.toLowerCase().split(/\s+/);
  const embedding = new Array(384).fill(0); // 384-dim embedding

  for (const word of words) {
    const hash = hashWord(word);
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] += Math.sin(hash + i) * 0.1;
    }
  }

  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
  return embedding.map((v) => (magnitude > 0 ? v / magnitude : 0));
}

function hashWord(word: string): number {
  let hash = 0;
  for (let i = 0; i < word.length; i++) {
    const char = word.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash;
}
