"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { Send, ArrowLeft, Loader, MessageCircle, CheckCircle2 } from "lucide-react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import "katex/dist/katex.min.css";

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
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-primary-50">
        <p className="text-slate-600">학습 계획을 로드하는 중...</p>
      </div>
    );
  }

  const currentDay = curriculum.schedule[currentDayIndex];
  const progress = Math.round(
    ((currentDayIndex + 1) / curriculum.schedule.length) * 100
  );

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-50 to-primary-50">
      {/* Header */}
      <div className="border-b border-slate-200 bg-white/70 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="p-2 text-slate-600 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-all duration-300"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-display font-bold text-slate-900">
                Day {currentDay.day}: {currentDay.topic}
              </h1>
              <p className="text-sm text-slate-600">
                예상 학습 시간: {currentDay.estimatedTime}분
              </p>
            </div>
          </div>

          {/* Answer Mode Toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => setAnswerMode("normal")}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                answerMode === "normal"
                  ? "bg-primary-600 text-white shadow-md"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              일반 모드
            </button>
            <button
              onClick={() => setAnswerMode("roleplay")}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                answerMode === "roleplay"
                  ? "bg-accent-600 text-white shadow-md"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              역할극 모드
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="max-w-5xl mx-auto px-4 pb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700">진행률</span>
            <span className="text-sm text-slate-600">{progress}%</span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
                  <MessageCircle className="w-4 h-4 text-primary-600" />
                </div>
              )}
              <div
                className={`max-w-2xl rounded-lg p-4 ${
                  message.role === "user"
                    ? "bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-md"
                    : "bg-white text-slate-900 border border-slate-200 shadow-base prose prose-sm max-w-none"
                }`}
              >
                {message.role === "user" ? (
                  <p className="text-sm sm:text-base whitespace-pre-wrap">
                    {message.content}
                  </p>
                ) : (
                  <div className="text-slate-900 leading-relaxed">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        code: ({ node, inline, className, children, ...props }: any) => {
                          const match = /language-(\w+)/.exec(className || "");
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={oneDark}
                              language={match[1]}
                              PreTag="div"
                              className="rounded-lg"
                              customStyle={{
                                margin: "1.25rem 0",
                                padding: "1rem",
                                fontSize: "0.875rem",
                                lineHeight: "1.5",
                              }}
                              {...props}
                            >
                              {String(children).replace(/\n$/, "")}
                            </SyntaxHighlighter>
                          ) : (
                            <code
                              className="bg-slate-100 text-primary-700 px-2 py-1 rounded text-xs font-mono"
                              {...props}
                            >
                              {children}
                            </code>
                          );
                        },
                        table: ({ node, ...props }: any) => (
                          <table
                            className="border-collapse w-full text-sm"
                            style={{
                              marginTop: "1.25rem",
                              marginBottom: "1.25rem",
                            }}
                            {...props}
                          />
                        ),
                        thead: ({ node, ...props }: any) => (
                          <thead
                            style={{
                              backgroundColor: "#f3f4f6",
                            }}
                            {...props}
                          />
                        ),
                        th: ({ node, ...props }: any) => (
                          <th
                            className="border border-slate-300 px-3 py-2 text-left font-semibold text-slate-900"
                            {...props}
                          />
                        ),
                        td: ({ node, ...props }: any) => (
                          <td className="border border-slate-300 px-3 py-2" {...props} />
                        ),
                        h1: ({ node, ...props }: any) => (
                          <h1 className="text-2xl font-display font-bold text-slate-900" {...props} />
                        ),
                        h2: ({ node, ...props }: any) => (
                          <h2 className="text-xl font-display font-bold text-slate-900" {...props} />
                        ),
                        h3: ({ node, ...props }: any) => (
                          <h3 className="text-lg font-display font-bold text-slate-900" {...props} />
                        ),
                        h4: ({ node, ...props }: any) => (
                          <h4 className="text-base font-display font-bold text-slate-900" {...props} />
                        ),
                        ul: ({ node, ...props }: any) => (
                          <ul className="list-disc list-outside pl-6 text-slate-800" {...props} />
                        ),
                        ol: ({ node, ...props }: any) => (
                          <ol className="list-decimal list-outside pl-6 text-slate-800" {...props} />
                        ),
                        li: ({ node, ...props }: any) => (
                          <li className="mb-2 leading-relaxed" {...props} />
                        ),
                        blockquote: ({ node, ...props }: any) => (
                          <blockquote
                            className="border-l-4 border-primary-500 pl-4 italic text-slate-700"
                            {...props}
                          />
                        ),
                        a: ({ node, ...props }: any) => (
                          <a className="text-primary-600 underline hover:text-primary-700 transition-colors" {...props} />
                        ),
                        p: ({ node, ...props }: any) => (
                          <p className="text-slate-900 leading-relaxed" {...props} />
                        ),
                        hr: ({ node, ...props }: any) => (
                          <hr className="border-0 border-t border-slate-300 my-6" {...props} />
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                )}
                {message.sources && message.sources.length > 0 && (
                  <div className={`mt-3 pt-3 border-t text-xs ${message.role === "user" ? "border-emerald-400 text-emerald-100" : "border-slate-200 text-slate-600"}`}>
                    <p className="font-semibold mb-1">📚 참고 자료:</p>
                    {message.sources.map((source, idx) => (
                      <div key={idx} className={message.role === "user" ? "text-emerald-100" : "text-slate-600"}>
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
              <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
                <Loader className="w-4 h-4 text-primary-600 animate-spin" />
              </div>
              <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-base">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                  <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-200 bg-white/70 backdrop-blur-md sticky bottom-0">
        <div className="max-w-5xl mx-auto px-4 py-4 space-y-3">
          {!currentDay.completed && (
            <button
              onClick={handleCompleteDay}
              className="w-full px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-lg hover:from-emerald-600 hover:to-emerald-700 transition-all duration-300 font-medium flex items-center justify-center gap-2 shadow-md hover:shadow-lg"
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
              className="w-full px-4 py-2 bg-gradient-to-r from-primary-600 to-primary-700 text-white rounded-lg hover:from-primary-700 hover:to-primary-800 transition-all duration-300 font-medium shadow-md hover:shadow-lg"
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
              className="flex-1 px-4 py-2 border-2 border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-slate-900 bg-white transition-all duration-300"
              disabled={loading || currentDay.completed}
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || currentDay.completed}
              className="px-4 py-2 bg-gradient-to-br from-primary-600 to-primary-700 text-white rounded-lg hover:from-primary-700 hover:to-primary-800 disabled:from-slate-400 disabled:to-slate-500 disabled:cursor-not-allowed transition-all duration-300 shadow-md hover:shadow-lg"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
