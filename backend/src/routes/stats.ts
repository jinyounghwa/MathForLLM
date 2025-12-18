import { Hono } from "hono";
import {
  calculateSessionStats,
  calculateCurriculumStats,
  getTopTopics,
  getOverallStats,
  calculateStreak,
  type OverallStats,
} from "../services/stats.js";

const router = new Hono();

interface StatsRequest {
  sessions: any[];
  curriculum: any;
  dayMessages: any;
}

// Get overall statistics
router.post("/overall", async (c) => {
  try {
    const body = (await c.req.json()) as StatsRequest;
    const { sessions, curriculum, dayMessages } = body;

    const stats = await getOverallStats(sessions, curriculum, dayMessages);

    return c.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    console.error("Stats API error:", error);
    return c.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

// Get session statistics only
router.post("/session", async (c) => {
  try {
    const body = (await c.req.json()) as { sessions: any[] };
    const { sessions } = body;

    const stats = await calculateSessionStats(sessions);

    return c.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    console.error("Session stats error:", error);
    return c.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

// Get curriculum statistics only
router.post("/curriculum", async (c) => {
  try {
    const body = (await c.req.json()) as {
      curriculum: any;
      dayMessages: any;
    };
    const { curriculum, dayMessages } = body;

    const stats = await calculateCurriculumStats(curriculum, dayMessages);

    return c.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    console.error("Curriculum stats error:", error);
    return c.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

// Get top topics
router.post("/topics", async (c) => {
  try {
    const body = (await c.req.json()) as { sessions: any[] };
    const { sessions } = body;

    const topics = await getTopTopics(sessions);

    return c.json({
      success: true,
      data: topics,
    });
  } catch (error) {
    console.error("Topics stats error:", error);
    return c.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

// Get learning streak
router.post("/streak", async (c) => {
  try {
    const body = (await c.req.json()) as { sessions: any[] };
    const { sessions } = body;

    const streak = await calculateStreak(sessions);

    return c.json({
      success: true,
      data: { streak },
    });
  } catch (error) {
    console.error("Streak calculation error:", error);
    return c.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

export default router;
