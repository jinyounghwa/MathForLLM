import { Hono } from "hono";
import { queryRAG } from "../services/rag";

const router = new Hono();

interface ChatRequest {
  message: string;
  mode: "normal" | "roleplay";
  learningMode: "free" | "curriculum";
  curriculumDay?: number;
  history?: Array<{ role: string; content: string }>;
}

router.post("/", async (c) => {
  try {
    const body = (await c.req.json()) as ChatRequest;
    const { message, mode = "normal", history = [] } = body;

    // Query RAG system
    const result = await queryRAG(message, mode, history);

    return c.json({
      answer: result.answer,
      sources: result.sources,
      suggestedQuestions: result.suggestedQuestions,
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
