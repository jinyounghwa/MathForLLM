// Statistics service for learning analytics

export interface SessionStats {
  totalSessions: number;
  totalMessages: number;
  totalLearningTime: number; // in minutes
  averageSessionDuration: number; // in minutes
  modes: {
    normal: number;
    roleplay: number;
  };
  lastSessionDate?: string;
}

export interface CurriculumStats {
  totalCurriculum: number;
  activeCurriculum: number;
  completedDays: number;
  totalDays: number;
  completionPercentage: number;
  averageCompletionTime: number; // in minutes per day
  lastLearningDate?: string;
}

export interface LearningTopic {
  topic: string;
  frequency: number;
  lastAsked: string;
}

export interface OverallStats {
  session: SessionStats;
  curriculum: CurriculumStats;
  topTopics: LearningTopic[];
  totalLearningTime: number; // in minutes
  joinDate: string;
  streak: number; // consecutive days of learning
}

// Calculate session statistics from localStorage data
export async function calculateSessionStats(sessions: any[]): Promise<SessionStats> {
  if (!sessions || sessions.length === 0) {
    return {
      totalSessions: 0,
      totalMessages: 0,
      totalLearningTime: 0,
      averageSessionDuration: 0,
      modes: {
        normal: 0,
        roleplay: 0,
      },
    };
  }

  let totalMessages = 0;
  let normalCount = 0;
  let roleplayCount = 0;
  const sessionDurations: number[] = [];

  sessions.forEach((session: any) => {
    totalMessages += session.messages?.length || 0;
    const mode = session.mode || "normal";

    if (mode === "normal") {
      normalCount++;
    } else {
      roleplayCount++;
    }

    // Calculate session duration
    if (session.messages && session.messages.length > 0) {
      const firstMsg = new Date(session.messages[0].timestamp).getTime();
      const lastMsg = new Date(
        session.messages[session.messages.length - 1].timestamp
      ).getTime();
      const durationMs = lastMsg - firstMsg;
      const durationMins = Math.max(1, Math.round(durationMs / 60000));
      sessionDurations.push(durationMins);
    }
  });

  const totalLearningTime =
    sessionDurations.reduce((a, b) => a + b, 0);
  const averageDuration =
    sessionDurations.length > 0
      ? Math.round(totalLearningTime / sessionDurations.length)
      : 0;

  const lastSession = sessions[sessions.length - 1];
  const lastSessionDate =
    lastSession.messages && lastSession.messages.length > 0
      ? lastSession.messages[lastSession.messages.length - 1].timestamp
      : undefined;

  return {
    totalSessions: sessions.length,
    totalMessages,
    totalLearningTime,
    averageSessionDuration: averageDuration,
    modes: {
      normal: normalCount,
      roleplay: roleplayCount,
    },
    lastSessionDate,
  };
}

// Calculate curriculum statistics
export async function calculateCurriculumStats(
  curriculum: any,
  dayMessages: any
): Promise<CurriculumStats> {
  if (!curriculum) {
    return {
      totalCurriculum: 0,
      activeCurriculum: 0,
      completedDays: 0,
      totalDays: 0,
      completionPercentage: 0,
      averageCompletionTime: 0,
    };
  }

  const schedule = curriculum.schedule || [];
  const completedDays = schedule.filter((day: any) => day.completed).length;
  const totalDays = schedule.length;
  const completionPercentage =
    totalDays > 0 ? Math.round((completedDays / totalDays) * 100) : 0;

  // Calculate average completion time
  let totalTime = 0;
  let completedCount = 0;

  schedule.forEach((day: any) => {
    if (day.completed && dayMessages && dayMessages[day.day]) {
      const messages = dayMessages[day.day];
      if (messages.length > 0) {
        const firstMsg = new Date(messages[0].timestamp).getTime();
        const lastMsg = new Date(messages[messages.length - 1].timestamp).getTime();
        const durationMs = lastMsg - firstMsg;
        const durationMins = Math.round(durationMs / 60000);
        totalTime += durationMins;
        completedCount++;
      }
    }
  });

  const averageCompletionTime =
    completedCount > 0 ? Math.round(totalTime / completedCount) : 0;

  const lastScheduledDay = schedule
    .filter((d: any) => d.completed)
    .sort((a: any, b: any) => b.day - a.day)[0];

  return {
    totalCurriculum: 1,
    activeCurriculum: completionPercentage < 100 ? 1 : 0,
    completedDays,
    totalDays,
    completionPercentage,
    averageCompletionTime,
    lastLearningDate: lastScheduledDay?.date,
  };
}

// Get top topics from session messages
export async function getTopTopics(sessions: any[]): Promise<LearningTopic[]> {
  const topicMap = new Map<string, { count: number; lastDate: string }>();

  sessions.forEach((session: any) => {
    session.messages?.forEach((msg: any) => {
      if (msg.role === "user") {
        // Extract keywords from message (simple heuristic)
        const keywords = extractKeywords(msg.content);
        keywords.forEach((keyword) => {
          const existing = topicMap.get(keyword) || {
            count: 0,
            lastDate: msg.timestamp,
          };
          existing.count++;
          existing.lastDate = msg.timestamp; // Update to latest
          topicMap.set(keyword, existing);
        });
      }
    });
  });

  return Array.from(topicMap.entries())
    .map(([topic, data]) => ({
      topic,
      frequency: data.count,
      lastAsked: data.lastDate,
    }))
    .sort((a, b) => b.frequency - a.frequency)
    .slice(0, 10); // Top 10 topics
}

// Extract keywords from text (simple implementation)
function extractKeywords(text: string): string[] {
  const keywords = [
    "벡터",
    "행렬",
    "미분",
    "적분",
    "확률",
    "통계",
    "선형대수",
    "미적분",
    "신경망",
    "엔트로피",
    "경사하강",
    "소프트맥스",
    "고유값",
    "함수",
    "수열",
    "극한",
    "급수",
    "이산수학",
    "정보이론",
    "머신러닝",
    "딥러닝",
    "트랜스포머",
    "임베딩",
    "코사인",
    "거리",
  ];

  return keywords.filter((keyword) => text.includes(keyword));
}

// Calculate learning streak
export async function calculateStreak(sessions: any[]): Promise<number> {
  if (!sessions || sessions.length === 0) return 0;

  const now = new Date();
  now.setHours(0, 0, 0, 0);

  const sortedSessions = sessions
    .map((s: any) => ({
      date: s.updatedAt
        ? new Date(s.updatedAt)
        : new Date(s.messages?.[0]?.timestamp || 0),
    }))
    .sort((a, b) => b.date.getTime() - a.date.getTime());

  let streak = 0;
  let currentDate = new Date(now);

  for (const session of sortedSessions) {
    const sessionDate = new Date(session.date);
    sessionDate.setHours(0, 0, 0, 0);

    if (sessionDate.getTime() === currentDate.getTime()) {
      // Same day - continue streak
      continue;
    } else if (
      sessionDate.getTime() ===
      currentDate.getTime() - 24 * 60 * 60 * 1000
    ) {
      // Previous day - continue streak
      streak++;
      currentDate = sessionDate;
    } else if (sessionDate.getTime() < currentDate.getTime()) {
      // Gap in streak
      break;
    }
  }

  return Math.max(0, streak);
}

// Get overall learning statistics
export async function getOverallStats(
  sessions: any[],
  curriculum: any,
  dayMessages: any
): Promise<OverallStats> {
  const sessionStats = await calculateSessionStats(sessions);
  const curriculumStats = await calculateCurriculumStats(curriculum, dayMessages);
  const topTopics = await getTopTopics(sessions);
  const streak = await calculateStreak(sessions);

  const firstSession = sessions?.[0];
  const joinDate = firstSession
    ? new Date(firstSession.createdAt).toISOString().split("T")[0]
    : new Date().toISOString().split("T")[0];

  const totalLearningTime =
    sessionStats.totalLearningTime + curriculumStats.averageCompletionTime;

  return {
    session: sessionStats,
    curriculum: curriculumStats,
    topTopics,
    totalLearningTime,
    joinDate,
    streak,
  };
}
