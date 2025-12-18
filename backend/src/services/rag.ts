import { generateResponse, generateEmbedding } from "./llm.js";
import { searchDocuments } from "./vectordb.js";

const NORMAL_SYSTEM_PROMPT = `당신은 LLM/AI 수학 강사입니다. 간결하고 정확하게 답변하세요.

형식:
1) 정의 (1문장)
2) 수식 (LaTeX)
3) 예제

제약: 300자 이내, 직결한 설명만 제공`;

const ROLEPLAY_SYSTEM_PROMPT = `당신은 AI 개발 시뮬레이션 멘토입니다. 실무 상황에서 필요한 개념을 설명하세요.

형식:
【역할극】
상황: (1줄 상황)
역할: (1줄)
미션: (1줄)
---
개념 설명: (간결하게)
적용: (예시)
【종료】

제약: 500자 이내, 직결하고 실용적으로`;

interface RAGResponse {
  answer: string;
  sources: Array<{
    file: string;
    section: string;
    relevance: number;
  }>;
  suggestedQuestions: string[];
}

export async function queryRAG(
  userQuery: string,
  mode: "normal" | "roleplay" = "normal",
  history: Array<{ role: string; content: string }> = []
): Promise<RAGResponse> {
  try {
    // Generate embedding for the query
    const queryEmbedding = await generateEmbedding(userQuery);

    // Search for relevant documents
    const searchResults = await searchDocuments(userQuery, queryEmbedding, 3);

    // Format context from search results
    const context = searchResults
      .map(
        (result: any) =>
          `[${result.metadata.source} - ${result.metadata.section}]\n${result.content}`
      )
      .join("\n\n");

    // Select system prompt based on mode
    const systemPrompt =
      mode === "roleplay" ? ROLEPLAY_SYSTEM_PROMPT : NORMAL_SYSTEM_PROMPT;

    // Generate response using LLM
    const answer = await generateResponse(systemPrompt, userQuery, context, history);

    // Format sources
    const sources = searchResults.map((result: any) => ({
      file: result.metadata.source,
      section: result.metadata.section,
      relevance: result.relevance,
    }));

    // Generate suggested follow-up questions
    const suggestedQuestions = generateSuggestedQuestions(userQuery, answer);

    return {
      answer,
      sources,
      suggestedQuestions,
    };
  } catch (error) {
    console.error("RAG 쿼리 오류:", error);
    throw error;
  }
}

function generateSuggestedQuestions(query: string, answer: string): string[] {
  // Simple heuristic-based suggestion
  const suggestions = [
    `이것과 관련된 다른 개념이 있나요?`,
    `실제 구현에서는 어떻게 사용되나요?`,
    `더 자세한 예제가 있을까요?`,
  ];

  return suggestions.slice(0, 2);
}
