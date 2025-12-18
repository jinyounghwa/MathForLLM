"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Send, ArrowLeft, Loader, MessageCircle, Trash2 } from "lucide-react";
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

interface ChatSession {
  sessionId: string;
  mode: "normal" | "roleplay";
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";
const STORAGE_KEY = "mathForLLM_chat_sessions";
const CURRENT_SESSION_KEY = "mathForLLM_current_session";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [answerMode, setAnswerMode] = useState<"normal" | "roleplay">("normal");
  const [sessionId, setSessionId] = useState<string>("");
  const [isLoaded, setIsLoaded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // LocalStorage에서 세션 로드
  useEffect(() => {
    const savedSessionId = localStorage.getItem(CURRENT_SESSION_KEY);
    const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as ChatSession[];

    let session: ChatSession | undefined;

    if (savedSessionId) {
      session = sessions.find((s) => s.sessionId === savedSessionId);
    }

    if (session) {
      setMessages(session.messages);
      setAnswerMode(session.mode);
      setSessionId(session.sessionId);
    } else {
      // 새로운 세션 생성
      const newSessionId = `session_${Date.now()}`;
      const initialMessage: Message = {
        id: "1",
        role: "assistant",
        content: "안녕하세요! LLM 수학에 대해 무엇이든 물어보세요. 📚 참고: 저장된 교재 44개",
        timestamp: new Date().toISOString(),
      };
      setMessages([initialMessage]);
      setSessionId(newSessionId);
      localStorage.setItem(CURRENT_SESSION_KEY, newSessionId);
    }

    setIsLoaded(true);
  }, []);

  // 메시지가 변경될 때마다 저장
  useEffect(() => {
    if (!isLoaded || !sessionId) return;

    const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as ChatSession[];
    const existingIndex = sessions.findIndex((s) => s.sessionId === sessionId);

    const session: ChatSession = {
      sessionId,
      mode: answerMode,
      messages,
      createdAt: existingIndex >= 0 ? sessions[existingIndex].createdAt : new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    if (existingIndex >= 0) {
      sessions[existingIndex] = session;
    } else {
      sessions.push(session);
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  }, [messages, answerMode, sessionId, isLoaded]);

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
      timestamp: new Date().toISOString(),
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

  const handleNewSession = () => {
    if (window.confirm("새로운 세션을 시작하시겠습니까?")) {
      const newSessionId = `session_${Date.now()}`;
      const initialMessage: Message = {
        id: "1",
        role: "assistant",
        content: "안녕하세요! LLM 수학에 대해 무엇이든 물어보세요. 📚 참고: 저장된 교재 44개",
        timestamp: new Date().toISOString(),
      };
      setMessages([initialMessage]);
      setSessionId(newSessionId);
      setAnswerMode("normal");
      localStorage.setItem(CURRENT_SESSION_KEY, newSessionId);
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
            <div>
              <h1 className="text-xl font-bold text-gray-900">일반 질문 모드</h1>
              <p className="text-xs text-gray-500">세션 ID: {sessionId.slice(-8)}</p>
            </div>
          </div>

          {/* Answer Mode Toggle & Session Controls */}
          <div className="flex gap-2 items-center">
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
            <button
              onClick={handleNewSession}
              className="px-3 py-2 bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
              title="새로운 세션 시작"
            >
              <Trash2 className="w-5 h-5" />
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
                {message.role === "user" ? (
                  <p className="text-sm sm:text-base whitespace-pre-wrap">
                    {message.content}
                  </p>
                ) : (
                  <div className="prose prose-sm max-w-none dark:prose-invert">
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
                              {...props}
                            >
                              {String(children).replace(/\n$/, "")}
                            </SyntaxHighlighter>
                          ) : (
                            <code className="bg-gray-800 text-orange-300 px-2 py-1 rounded text-xs" {...props}>
                              {children}
                            </code>
                          );
                        },
                        table: ({ node, ...props }: any) => (
                          <table className="border-collapse border border-gray-400 w-full my-2" {...props} />
                        ),
                        th: ({ node, ...props }: any) => (
                          <th className="border border-gray-400 bg-gray-200 px-3 py-2 text-left" {...props} />
                        ),
                        td: ({ node, ...props }: any) => (
                          <td className="border border-gray-400 px-3 py-2" {...props} />
                        ),
                        h1: ({ node, ...props }: any) => (
                          <h1 className="text-2xl font-bold my-3" {...props} />
                        ),
                        h2: ({ node, ...props }: any) => (
                          <h2 className="text-xl font-bold my-2" {...props} />
                        ),
                        h3: ({ node, ...props }: any) => (
                          <h3 className="text-lg font-bold my-2" {...props} />
                        ),
                        ul: ({ node, ...props }: any) => (
                          <ul className="list-disc list-inside my-2 space-y-1" {...props} />
                        ),
                        ol: ({ node, ...props }: any) => (
                          <ol className="list-decimal list-inside my-2 space-y-1" {...props} />
                        ),
                        blockquote: ({ node, ...props }: any) => (
                          <blockquote className="border-l-4 border-gray-400 pl-4 italic my-2" {...props} />
                        ),
                        a: ({ node, ...props }: any) => (
                          <a className="text-blue-600 underline hover:text-blue-800" {...props} />
                        ),
                        p: ({ node, ...props }: any) => (
                          <p className="my-2" {...props} />
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                )}
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
