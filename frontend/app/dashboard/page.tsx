"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, TrendingUp, BookOpen, Flame, Clock, Target } from "lucide-react";
import axios from "axios";

interface SessionStats {
  totalSessions: number;
  totalMessages: number;
  totalLearningTime: number;
  averageSessionDuration: number;
  modes: { normal: number; roleplay: number };
  lastSessionDate?: string;
}

interface CurriculumStats {
  totalCurriculum: number;
  activeCurriculum: number;
  completedDays: number;
  totalDays: number;
  completionPercentage: number;
  averageCompletionTime: number;
  lastLearningDate?: string;
}

interface LearningTopic {
  topic: string;
  frequency: number;
  lastAsked: string;
}

interface OverallStats {
  session: SessionStats;
  curriculum: CurriculumStats;
  topTopics: LearningTopic[];
  totalLearningTime: number;
  joinDate: string;
  streak: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";
const STORAGE_KEY = "mathForLLM_chat_sessions";
const CURRICULUM_KEY = "mathForLLM_curriculum";
const DAY_MESSAGES_KEY = "mathForLLM_day_messages";

export default function DashboardPage() {
  const [stats, setStats] = useState<OverallStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadStats() {
      try {
        // Load data from localStorage
        const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
        const curriculum = JSON.parse(localStorage.getItem(CURRICULUM_KEY) || "null");
        const dayMessages = JSON.parse(localStorage.getItem(DAY_MESSAGES_KEY) || "{}");

        // Call stats API
        const response = await axios.post(`${API_URL}/api/stats/overall`, {
          sessions,
          curriculum,
          dayMessages,
        });

        if (response.data.success) {
          setStats(response.data.data);
        } else {
          setError("통계를 불러올 수 없습니다");
        }
      } catch (err) {
        console.error("Stats loading error:", err);
        setError("데이터 로드 중 오류가 발생했습니다");
      } finally {
        setLoading(false);
      }
    }

    loadStats();
  }, []);

  const formatTime = (minutes: number) => {
    if (minutes < 60) return `${minutes}분`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}시간 ${mins}분`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "-";
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString("ko-KR", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "-";
    }
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">통계를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="text-center">
          <p className="text-red-600 mb-4">{error || "통계를 불러올 수 없습니다"}</p>
          <Link
            href="/"
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            홈으로 돌아가기
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center gap-4">
          <Link
            href="/"
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">학습 대시보드</h1>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {/* Total Learning Time */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">총 학습 시간</h3>
              <Clock className="w-5 h-5 text-blue-600" />
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {formatTime(stats.totalLearningTime)}
            </p>
            <p className="text-xs text-gray-500 mt-2">
              {stats.session.totalSessions}개 세션
            </p>
          </div>

          {/* Learning Streak */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">연속 학습 일수</h3>
              <Flame className="w-5 h-5 text-orange-600" />
            </div>
            <p className="text-2xl font-bold text-gray-900">{stats.streak}</p>
            <p className="text-xs text-gray-500 mt-2">일</p>
          </div>

          {/* Total Messages */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">총 메시지</h3>
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {stats.session.totalMessages}
            </p>
            <p className="text-xs text-gray-500 mt-2">평균 {Math.round(stats.session.totalMessages / stats.session.totalSessions)}개/세션</p>
          </div>

          {/* Curriculum Progress */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">커리큘럼 진도</h3>
              <Target className="w-5 h-5 text-indigo-600" />
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {stats.curriculum.completionPercentage}%
            </p>
            <p className="text-xs text-gray-500 mt-2">
              {stats.curriculum.completedDays}/{stats.curriculum.totalDays}일
            </p>
          </div>
        </div>

        {/* Session Analytics & Curriculum Progress */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Session Analytics */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-6 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-blue-600" />
              질문 모드 분석
            </h2>

            <div className="space-y-4">
              {/* Normal Mode */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    일반 모드
                  </span>
                  <span className="text-sm font-bold text-gray-900">
                    {stats.session.modes.normal}회
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{
                      width: `${
                        stats.session.totalSessions > 0
                          ? (stats.session.modes.normal / stats.session.totalSessions) * 100
                          : 0
                      }%`,
                    }}
                  ></div>
                </div>
              </div>

              {/* Roleplay Mode */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    역할극 모드
                  </span>
                  <span className="text-sm font-bold text-gray-900">
                    {stats.session.modes.roleplay}회
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-indigo-600 h-2 rounded-full transition-all"
                    style={{
                      width: `${
                        stats.session.totalSessions > 0
                          ? (stats.session.modes.roleplay / stats.session.totalSessions) * 100
                          : 0
                      }%`,
                    }}
                  ></div>
                </div>
              </div>

              {/* Session Duration */}
              <div className="pt-4 border-t border-gray-200">
                <p className="text-sm text-gray-600">
                  평균 세션 시간: <span className="font-bold text-gray-900">{stats.session.averageSessionDuration}분</span>
                </p>
                <p className="text-sm text-gray-600 mt-2">
                  마지막 학습: <span className="font-bold text-gray-900">{formatDate(stats.session.lastSessionDate)}</span>
                </p>
              </div>
            </div>
          </div>

          {/* Curriculum Progress */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-6 flex items-center gap-2">
              <Target className="w-5 h-5 text-indigo-600" />
              학습 계획 진행률
            </h2>

            <div className="space-y-4">
              {/* Main Progress Bar */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    전체 진도
                  </span>
                  <span className="text-sm font-bold text-gray-900">
                    {stats.curriculum.completionPercentage}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-indigo-600 h-4 rounded-full transition-all"
                    style={{
                      width: `${stats.curriculum.completionPercentage}%`,
                    }}
                  ></div>
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
                <div>
                  <p className="text-xs text-gray-600">완료한 일차</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats.curriculum.completedDays}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">총 일차</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats.curriculum.totalDays}
                  </p>
                </div>
                <div className="col-span-2">
                  <p className="text-xs text-gray-600">평균 완료 시간</p>
                  <p className="text-sm font-bold text-gray-900 mt-1">
                    {formatTime(stats.curriculum.averageCompletionTime)}
                  </p>
                </div>
                <div className="col-span-2">
                  <p className="text-xs text-gray-600">마지막 학습</p>
                  <p className="text-sm font-bold text-gray-900 mt-1">
                    {formatDate(stats.curriculum.lastLearningDate)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Top Topics */}
        {stats.topTopics.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6 mb-8">
            <h2 className="text-lg font-bold text-gray-900 mb-6">
              자주 학습한 주제
            </h2>

            <div className="space-y-3">
              {stats.topTopics.slice(0, 8).map((topic, idx) => (
                <div key={idx} className="flex items-center justify-between pb-3 border-b border-gray-100 last:border-b-0">
                  <div className="flex items-center gap-3 flex-1">
                    <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 text-sm font-bold">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{topic.topic}</p>
                      <p className="text-xs text-gray-500">
                        마지막: {formatDate(topic.lastAsked)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-gray-900">{topic.frequency}</p>
                    <p className="text-xs text-gray-500">회</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Summary Card */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg shadow p-6 text-white">
          <h2 className="text-lg font-bold mb-4">학습 요약</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-blue-100 text-sm">가입일</p>
              <p className="font-bold text-lg mt-1">
                {new Date(stats.joinDate).toLocaleDateString("ko-KR")}
              </p>
            </div>
            <div>
              <p className="text-blue-100 text-sm">총 학습 세션</p>
              <p className="font-bold text-lg mt-1">
                {stats.session.totalSessions}회
              </p>
            </div>
            <div>
              <p className="text-blue-100 text-sm">평가</p>
              <p className="font-bold text-lg mt-1">
                {stats.totalLearningTime > 300
                  ? "⭐⭐⭐⭐⭐ 뛰어남"
                  : stats.totalLearningTime > 120
                  ? "⭐⭐⭐⭐ 우수"
                  : stats.totalLearningTime > 60
                  ? "⭐⭐⭐ 좋음"
                  : "⭐⭐ 시작"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
