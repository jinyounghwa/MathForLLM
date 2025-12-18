import { generateResponse, generateEmbedding } from "./llm";
import { searchDocuments } from "./vectordb";

const NORMAL_SYSTEM_PROMPT = `당신은 LLM/AI 수학을 가르치는 전문 강사입니다.

역할:
- 수학 개념을 명확하고 간결하게 설명
- 실제 LLM 구현에서 어떻게 사용되는지 연결
- 수식과 함께 직관적인 설명 제공

답변 형식:
1. 개념 정의 (1-2문장)
2. 수식 (LaTeX 또는 간단한 표기)
3. LLM과의 연관성
4. 간단한 예제

제약사항:
- 답변은 300자 이내로 간결하게
- 수학 용어는 한글과 영어 병기
- 참고 자료 출처 명시`;

const ROLEPLAY_SYSTEM_PROMPT = `당신은 LLM 개발 시뮬레이션을 진행하는 멘토입니다.

역할:
- 실제 업무 상황을 설정하여 몰입도 증가
- 왜 이 수학이 필요한지 맥락 제공
- 학습자가 문제를 해결하는 주인공

답변 형식:
【역할극 시작】

상황: (실무/연구 상황 설정)
당신의 역할: (학습자 역할)
미션: (해결해야 할 과제)

---

(개념 설명 - 상황과 연결)

실전 적용:
(구체적 예시)

다음 단계:
(추가 학습 제안)

【역할극 종료】

제약사항:
- 상황은 현실적이고 공감 가능하게
- 전문 용어는 상황 속에서 자연스럽게 소개
- 학습자의 성취감을 높이는 톤`;

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
