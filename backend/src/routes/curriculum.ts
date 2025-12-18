import { Hono } from "hono";
import { v4 as uuidv4 } from "uuid";

const router = new Hono();

interface LearningDay {
  day: number;
  date: string;
  topic: string;
  sections: string[];
  estimatedTime: number;
  completed: boolean;
}

interface Curriculum {
  curriculumId: string;
  totalDays: number;
  schedule: LearningDay[];
}

interface CurriculumRequest {
  frequency: 1 | 2 | 3 | 7;
  duration: 30 | 60 | 120;
  startDate: string;
}

const sampleTopics = [
  "수의 체계와 기초 연산",
  "지수와 로그",
  "함수와 그래프",
  "집합과 논리",
  "변수, 방정식, 연립방정식",
  "좌표평면과 벡터 기초",
  "벡터의 길이와 거리",
  "내적",
  "정규화",
  "행렬과 행렬곱셈",
  "중간 복습 - 선형대수 기초",
  "전치행렬",
  "역행렬과 단위행렬",
  "고유값과 고유벡터",
  "행렬식과 노름",
  "선형대수 최종 프로젝트",
  "극한과 연속성",
  "미분 - 도함수",
  "연쇄법칙",
  "편미분과 그래디언트",
  "미분을 이용한 최적화",
  "중간 복습 - 미적분",
  "확률론 기초",
  "확률분포",
  "중간 복습 - 확률",
  "엔트로피와 정보이득",
  "베이즈 정리",
  "최대가능도 추정",
  "신경망 기초",
  "활성화 함수",
  "손실함수와 백프로퍼게이션",
  "경사하강법",
  "중간 복습 - 신경망",
  "경사하강법의 변형",
  "배치 정규화",
  "드롭아웃",
  "정규화 기법",
];

router.post("/", async (c) => {
  try {
    const body = (await c.req.json()) as CurriculumRequest;
    const { frequency, duration, startDate } = body;

    // Calculate schedule
    const start = new Date(startDate);
    const schedule: LearningDay[] = [];

    const learningDayCount = sampleTopics.length;
    let currentDate = new Date(start);
    let dayCounter = 1;

    for (let i = 0; i < learningDayCount; i++) {
      const topic = sampleTopics[i];
      const learningDay: LearningDay = {
        day: dayCounter,
        date: currentDate.toISOString().split("T")[0],
        topic: topic,
        sections: [topic],
        estimatedTime: duration,
        completed: false,
      };

      schedule.push(learningDay);

      // Move to next learning day
      currentDate = new Date(
        currentDate.getTime() + frequency * 24 * 60 * 60 * 1000
      );
      dayCounter++;
    }

    const totalDays = Math.ceil(
      (schedule[schedule.length - 1].day * frequency * 24 * 60 * 60 * 1000) /
        (24 * 60 * 60 * 1000)
    );

    const curriculum: Curriculum = {
      curriculumId: uuidv4(),
      totalDays,
      schedule,
    };

    return c.json(curriculum);
  } catch (error) {
    console.error("Curriculum endpoint error:", error);
    return c.json(
      {
        error: "An error occurred while generating curriculum",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      500
    );
  }
});

export default router;
