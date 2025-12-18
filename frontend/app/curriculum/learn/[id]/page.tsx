"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { Send, ArrowLeft, Loader, MessageCircle, CheckCircle2 } from "lucide-react";
import axios from "axios";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ file: string; section: string; relevance: number }>;
  timestamp: string;
}

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

interface DayMessages {
  [dayNumber: number]: Message[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";
const CURRICULUM_KEY = "mathForLLM_curriculum";
const DAY_MESSAGES_KEY = "mathForLLM_day_messages";

export default function CurriculumLearningPage({
  params,
}: {
  params: { id: string };
}) {
  const [curriculum, setCurriculum] = useState<Curriculum | null>(null);
  const [currentDayIndex, setCurrentDayIndex] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [answerMode, setAnswerMode] = useState<"normal" | "roleplay">("roleplay");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load curriculum from localStorage
    const stored = localStorage.getItem(CURRICULUM_KEY);
    if (stored) {
      const data = JSON.parse(stored) as Curriculum;
      setCurriculum(data);

      // Load saved messages for this day
      const dayMessagesStore = JSON.parse(localStorage.getItem(DAY_MESSAGES_KEY) || "{}") as DayMessages;
      const currentDay = data.schedule[currentDayIndex];

      if (currentDay) {
        if (dayMessagesStore[currentDay.day] && dayMessagesStore[currentDay.day].length > 0) {
          // Load saved messages
          setMessages(dayMessagesStore[currentDay.day]);
        } else {
          // Initialize with greeting message
          const initialMessage: Message = {
            id: "1",
            role: "assistant",
            content: `📚 Day ${currentDay.day}: ${currentDay.topic}\n\n당신의 오늘의 학습 목표는 "${currentDay.topic}"입니다. 자유롭게 질문하며 학습해보세요!`,
            timestamp: new Date().toISOString(),
          };
          setMessages([initialMessage]);
        }
      }
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Save messages to localStorage when they change
  useEffect(() => {
    if (!curriculum) return;

    const currentDay = curriculum.schedule[currentDayIndex];
    if (!currentDay) return;

    const dayMessagesStore = JSON.parse(localStorage.getItem(DAY_MESSAGES_KEY) || "{}") as DayMessages;
    dayMessagesStore[currentDay.day] = messages;
    localStorage.setItem(DAY_MESSAGES_KEY, JSON.stringify(dayMessagesStore));
  }, [messages, curriculum, currentDayIndex]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading || !curriculum) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: input,
        mode: answerMode,
        learningMode: "curriculum",
        curriculumDay: curriculum.schedule[currentDayIndex].day,
        history: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      });

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: response.data.answer,
        sources: response.data.sources,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("API 오류:", error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: "죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleCompleteDay = () => {
    if (!curriculum) return;

    const updated = { ...curriculum };
    updated.schedule[currentDayIndex].completed = true;
    localStorage.setItem(CURRICULUM_KEY, JSON.stringify(updated));
    setCurriculum(updated);

    // Move to next day
    if (currentDayIndex < curriculum.schedule.length - 1) {
      setCurrentDayIndex(currentDayIndex + 1);
      const nextDay = updated.schedule[currentDayIndex + 1];
      const initialMessage: Message = {
        id: "1",
        role: "assistant",
        content: `📚 Day ${nextDay.day}: ${nextDay.topic}\n\n다음 학습 주제입니다. 자유롭게 질문하며 학습해보세요!`,
        timestamp: new Date().toISOString(),
      };
      setMessages([initialMessage]);
    }
  };

  if (!curriculum) {
    return (
      <div className="h-screen flex items-center justify-center">
        <p className="text-gray-600">학습 계획을 로드하는 중...</p>
      </div>
    );
  }

  const currentDay = curriculum.schedule[currentDayIndex];
  const progress = Math.round(
    ((currentDayIndex + 1) / curriculum.schedule.length) * 100
  );

  return (
    <div className="h-screen flex flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </Link>
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                Day {currentDay.day}: {currentDay.topic}
              </h1>
              <p className="text-sm text-gray-500">
                예상 학습 시간: {currentDay.estimatedTime}분
              </p>
            </div>
          </div>

          {/* Answer Mode Toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => setAnswerMode("normal")}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                answerMode === "normal"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              일반 모드
            </button>
            <button
              onClick={() => setAnswerMode("roleplay")}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                answerMode === "roleplay"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              역할극 모드
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="max-w-4xl mx-auto px-4 pb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">진행률</span>
            <span className="text-sm text-gray-600">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                  <MessageCircle className="w-4 h-4 text-indigo-600" />
                </div>
              )}
              <div
                className={`max-w-2xl rounded-lg p-4 ${
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-900"
                }`}
              >
                <p className="text-sm sm:text-base whitespace-pre-wrap">
                  {message.content}
                </p>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-300 text-xs text-gray-600">
                    <p className="font-semibold mb-1">📚 참고 자료:</p>
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="text-gray-600">
                        {source.file} - {source.section}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-4 justify-start">
              <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                <Loader className="w-4 h-4 text-indigo-600 animate-spin" />
              </div>
              <div className="bg-gray-100 rounded-lg p-4">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white sticky bottom-0">
        <div className="max-w-4xl mx-auto px-4 py-4 space-y-3">
          {!currentDay.completed && (
            <button
              onClick={handleCompleteDay}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium flex items-center justify-center gap-2"
            >
              <CheckCircle2 className="w-5 h-5" />
              학습 완료하기
            </button>
          )}
          {currentDay.completed && currentDayIndex < curriculum.schedule.length - 1 && (
            <button
              onClick={() => {
                setCurrentDayIndex(currentDayIndex + 1);
                const nextDay = curriculum.schedule[currentDayIndex + 1];
                const dayMessagesStore = JSON.parse(localStorage.getItem(DAY_MESSAGES_KEY) || "{}") as DayMessages;
                if (dayMessagesStore[nextDay.day] && dayMessagesStore[nextDay.day].length > 0) {
                  setMessages(dayMessagesStore[nextDay.day]);
                } else {
                  const initialMessage: Message = {
                    id: "1",
                    role: "assistant",
                    content: `📚 Day ${nextDay.day}: ${nextDay.topic}\n\n다음 학습 주제입니다. 자유롭게 질문하며 학습해보세요!`,
                    timestamp: new Date().toISOString(),
                  };
                  setMessages([initialMessage]);
                }
              }}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              다음 학습하기
            </button>
          )}
          <form onSubmit={handleSendMessage} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문을 입력하세요..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading || currentDay.completed}
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || currentDay.completed}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
