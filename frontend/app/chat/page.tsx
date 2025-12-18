"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Send, ArrowLeft, Loader, MessageCircle } from "lucide-react";
import axios from "axios";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ file: string; section: string; relevance: number }>;
  timestamp: Date;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "안녕하세요! LLM 수학에 대해 무엇이든 물어보세요. 📚 참고: 저장된 교재 44개",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [answerMode, setAnswerMode] = useState<"normal" | "roleplay">("normal");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: input,
        mode: answerMode,
        learningMode: "free",
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
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("API 오류:", error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: "죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

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
            <h1 className="text-xl font-bold text-gray-900">일반 질문 모드</h1>
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
                        {source.file} - {source.section} ({Math.round(source.relevance * 100)}%)
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
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={handleSendMessage} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문을 입력하세요..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
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
