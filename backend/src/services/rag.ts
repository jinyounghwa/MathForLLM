import { generateResponse, generateEmbedding } from "./llm.js";
import { searchDocuments } from "./vectordb.js";

const NORMAL_SYSTEM_PROMPT = `당신은 LLM/AI 개발을 위한 수학 강사입니다. 명확하고 정확한 설명을 제공하세요.

답변 구조 (간결하게):
1) 핵심 개념 - 정의와 직관적 이해
2) 수학적 표현 - 주요 공식 (LaTeX)
3) 구체적 예제 - 명확한 예시와 계산 과정
4) 실무 연관성 - AI/LLM 분야 응용

원칙: 정확성, 명확한 수식, 구체적 예제. 적절한 길이로 설명 (600-800자)`;

const ROLEPLAY_SYSTEM_PROMPT = `당신은 AI 개발팀의 경험 많은 멘토입니다. 실무 관점에서 수학 개념을 설명하세요.

답변 구조:
【실무 상황】 간단한 상황 설명 및 핵심 개념
【수학 원리】 필요한 공식 및 수학적 기초
【실제 적용】 코드 관점의 접근과 예제
【주의사항】 흔한 실수와 최적화 팁

원칙: 실용적 설명, 이론과 실전 균형, 명확한 수식. 적절한 길이 (800-1000자)`;

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

    // Filter results with minimum relevance threshold (0.7)
    const relevantResults = searchResults.filter((result: any) => result.relevance >= 0.7);

    // Format context from search results (or use empty context if no relevant results)
    let context = "";
    let sources: Array<{ file: string; section: string; relevance: number }> = [];

    if (relevantResults.length > 0) {
      context = relevantResults
        .map(
          (result: any) =>
            `[${result.metadata.source} - ${result.metadata.section}]\n${result.content}`
        )
        .join("\n\n");

      sources = relevantResults.map((result: any) => ({
        file: result.metadata.source,
        section: result.metadata.section,
        relevance: result.relevance,
      }));
    } else {
      // RAG miss - model will use its own knowledge
      console.log(`RAG miss for query: "${userQuery}" - using model's own knowledge`);
    }

    // Select system prompt based on mode
    const systemPrompt =
      mode === "roleplay" ? ROLEPLAY_SYSTEM_PROMPT : NORMAL_SYSTEM_PROMPT;

    // Generate response using LLM (with or without context)
    const answer = await generateResponse(systemPrompt, userQuery, context, history);

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
