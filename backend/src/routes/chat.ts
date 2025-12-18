import { Hono } from "hono";
import { queryRAG } from "../services/rag.js";

const router = new Hono();

interface ChatRequest {
  message: string;
  mode: "normal" | "roleplay";
  learningMode: "free" | "curriculum";
  curriculumDay?: number;
  history?: Array<{ role: string; content: string }>;
}

interface CachedResponse {
  answer: string;
  sources: any[];
  suggestedQuestions: string[];
  timestamp: number;
}

// 간단한 메모리 캐시 (최대 100개 항목)
const responseCache = new Map<string, CachedResponse>();
const MAX_CACHE_SIZE = 100;
const CACHE_DURATION = 1000 * 60 * 60; // 1시간

// 캐시 키 생성
function generateCacheKey(message: string, mode: string): string {
  return `${mode}:${message.toLowerCase()}`;
}

// 캐시 정리 (오래된 항목 제거)
function cleanCache() {
  const now = Date.now();
  const entriesToDelete: string[] = [];

  responseCache.forEach((value, key) => {
    if (now - value.timestamp > CACHE_DURATION) {
      entriesToDelete.push(key);
    }
  });

  entriesToDelete.forEach(key => responseCache.delete(key));

  // 캐시 크기 제한 (FIFO)
  if (responseCache.size > MAX_CACHE_SIZE) {
    const entriesToRemove = responseCache.size - MAX_CACHE_SIZE;
    let removed = 0;
    for (const key of responseCache.keys()) {
      if (removed >= entriesToRemove) break;
      responseCache.delete(key);
      removed++;
    }
  }
}

router.post("/", async (c) => {
  try {
    const body = (await c.req.json()) as ChatRequest;
    const { message, mode = "normal", history = [] } = body;

    // 히스토리가 없는 경우만 캐시 사용 (초기 질문)
    let result;
    const cacheKey = generateCacheKey(message, mode);

    if (history.length === 0) {
      // 캐시에서 조회
      const cachedResult = responseCache.get(cacheKey);
      if (cachedResult) {
        console.log(`✓ Cache hit for: "${message.slice(0, 30)}..."`);
        return c.json({
          answer: cachedResult.answer,
          sources: cachedResult.sources,
          suggestedQuestions: cachedResult.suggestedQuestions,
          cached: true,
        });
      }
    }

    // 캐시 미스 - RAG 쿼리 실행
    const startTime = Date.now();
    result = await queryRAG(message, mode, history);
    const duration = Date.now() - startTime;

    console.log(`⏱️ Query time: ${duration}ms for "${message.slice(0, 30)}..."`);

    // 캐시에 저장 (히스토리가 없는 경우만)
    if (history.length === 0) {
      responseCache.set(cacheKey, {
        ...result,
        timestamp: Date.now(),
      });

      // 캐시 크기 관리
      cleanCache();
      console.log(`💾 Cached response (cache size: ${responseCache.size}/${MAX_CACHE_SIZE})`);
    }

    return c.json({
      answer: result.answer,
      sources: result.sources,
      suggestedQuestions: result.suggestedQuestions,
      cached: false,
      duration,
    });
  } catch (error) {
    console.error("Chat endpoint error:", error);
    return c.json(
      {
        error: "An error occurred while processing your request",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      500
    );
  }
});

export default router;
