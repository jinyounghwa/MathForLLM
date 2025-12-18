"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Send, ArrowLeft, Loader, Trash2, Zap, BookOpen } from "lucide-react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { motion, AnimatePresence } from "framer-motion";
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

// AI 아바타 컴포넌트
const AIAvatar = () => (
  <motion.div
    initial={{ scale: 0 }}
    animate={{ scale: 1 }}
    className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0 shadow-lg"
  >
    <span className="text-lg">🤖</span>
  </motion.div>
);

// 사용자 아바타 컴포넌트
const UserAvatar = () => (
  <motion.div
    initial={{ scale: 0 }}
    animate={{ scale: 1 }}
    className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0 shadow-lg"
  >
    <span className="text-lg">👤</span>
  </motion.div>
);

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
        content: "안녕하세요! 👋 LLM 수학에 대해 무엇이든 물어보세요. 저는 44개의 전문 교재로부터 답변을 제공합니다. 📚",
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
        content: "안녕하세요! 👋 LLM 수학에 대해 무엇이든 물어보세요. 저는 44개의 전문 교재로부터 답변을 제공합니다. 📚",
        timestamp: new Date().toISOString(),
      };
      setMessages([initialMessage]);
      setSessionId(newSessionId);
      setAnswerMode("normal");
      localStorage.setItem(CURRENT_SESSION_KEY, newSessionId);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="border-b border-slate-700 bg-gradient-to-r from-slate-800 via-slate-800 to-slate-800 sticky top-0 z-10 backdrop-blur-sm bg-opacity-80"
      >
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="p-2 hover:bg-slate-700 rounded-lg transition-all hover:scale-110"
            >
              <ArrowLeft className="w-5 h-5 text-slate-300" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  LLM 수학 튜터
                </h1>
              </div>
              <p className="text-xs text-slate-400">세션: {sessionId.slice(-6)}</p>
            </div>
          </div>

          {/* Mode Controls */}
          <div className="flex gap-2 items-center flex-wrap">
            <div className="hidden sm:flex gap-1 bg-slate-700 rounded-lg p-1">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setAnswerMode("normal")}
                className={`px-3 py-1.5 rounded transition-all text-sm font-medium ${
                  answerMode === "normal"
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-500/50"
                    : "text-slate-300 hover:text-white"
                }`}
              >
                일반
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setAnswerMode("roleplay")}
                className={`px-3 py-1.5 rounded transition-all text-sm font-medium ${
                  answerMode === "roleplay"
                    ? "bg-purple-600 text-white shadow-lg shadow-purple-500/50"
                    : "text-slate-300 hover:text-white"
                }`}
              >
                역할극
              </motion.button>
            </div>
            <motion.button
              whileHover={{ scale: 1.1, rotate: 180 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleNewSession}
              className="p-2 hover:bg-slate-700 rounded-lg transition-all text-slate-300 hover:text-red-400"
              title="새 세션"
            >
              <Trash2 className="w-5 h-5" />
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.role === "assistant" && <AIAvatar />}

                <motion.div
                  className={`max-w-2xl lg:max-w-3xl ${
                    message.role === "user" ? "order-2" : "order-1"
                  }`}
                >
                  <div
                    className={`rounded-2xl px-5 py-4 shadow-xl backdrop-blur-sm ${
                      message.role === "user"
                        ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-br-none"
                        : "bg-slate-700 text-slate-50 rounded-bl-none border border-slate-600"
                    }`}
                  >
                    {message.role === "user" ? (
                      <p className="text-sm sm:text-base leading-relaxed">
                        {message.content}
                      </p>
                    ) : (
                      <div className="prose prose-sm prose-invert max-w-none text-slate-50 leading-relaxed">
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
                                  className="rounded-lg my-2"
                                  {...props}
                                >
                                  {String(children).replace(/\n$/, "")}
                                </SyntaxHighlighter>
                              ) : (
                                <code
                                  className="bg-slate-900 text-orange-300 px-2 py-1 rounded text-xs font-mono"
                                  {...props}
                                >
                                  {children}
                                </code>
                              );
                            },
                            table: ({ node, ...props }: any) => (
                              <table
                                className="border-collapse border border-slate-500 w-full my-3 text-sm"
                                {...props}
                              />
                            ),
                            th: ({ node, ...props }: any) => (
                              <th
                                className="border border-slate-500 bg-slate-800 px-3 py-2 text-left font-semibold"
                                {...props}
                              />
                            ),
                            td: ({ node, ...props }: any) => (
                              <td className="border border-slate-500 px-3 py-2" {...props} />
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
                              <ul className="list-disc list-inside my-2 space-y-1 ml-2" {...props} />
                            ),
                            ol: ({ node, ...props }: any) => (
                              <ol className="list-decimal list-inside my-2 space-y-1 ml-2" {...props} />
                            ),
                            blockquote: ({ node, ...props }: any) => (
                              <blockquote
                                className="border-l-4 border-blue-400 pl-4 italic my-2 text-slate-300"
                                {...props}
                              />
                            ),
                            a: ({ node, ...props }: any) => (
                              <a className="text-blue-300 underline hover:text-blue-200" {...props} />
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
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="mt-4 pt-4 border-t border-slate-600 text-xs text-slate-300"
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <BookOpen className="w-4 h-4 text-yellow-400" />
                          <span className="font-semibold">참고 자료</span>
                        </div>
                        <div className="space-y-1 ml-6">
                          {message.sources.map((source, idx) => (
                            <div key={idx} className="text-slate-300 hover:text-blue-300 transition-colors">
                              <span className="font-medium">{source.file}</span> →{" "}
                              <span className="text-slate-400">{source.section}</span>
                              <span className="text-slate-500">
                                {" "}
                                ({Math.round(source.relevance * 100)}%)
                              </span>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </div>
                </motion.div>

                {message.role === "user" && <UserAvatar />}
              </motion.div>
            ))}
          </AnimatePresence>

          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3 justify-start"
            >
              <AIAvatar />
              <div className="bg-slate-700 rounded-2xl rounded-bl-none px-5 py-4 shadow-xl border border-slate-600">
                <div className="flex gap-2">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity }}
                    className="w-2.5 h-2.5 bg-blue-400 rounded-full"
                  />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                    className="w-2.5 h-2.5 bg-purple-400 rounded-full"
                  />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                    className="w-2.5 h-2.5 bg-pink-400 rounded-full"
                  />
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="border-t border-slate-700 bg-gradient-to-t from-slate-900 via-slate-800 to-slate-800 sticky bottom-0 backdrop-blur-sm bg-opacity-80"
      >
        <div className="max-w-5xl mx-auto px-4 py-6">
          <form onSubmit={handleSendMessage} className="flex gap-3 items-end">
            <motion.input
              whileFocus={{ scale: 1.02 }}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문을 입력하세요... (예: 벡터란 무엇인가요?)"
              className="flex-1 px-6 py-4 text-base bg-slate-700 border border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-50 placeholder-slate-400 transition-all shadow-lg hover:border-slate-500"
              disabled={loading}
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              disabled={loading || !input.trim()}
              className="p-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-700 hover:to-blue-800 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-blue-500/50 disabled:shadow-none flex-shrink-0"
            >
              <Send className="w-6 h-6" />
            </motion.button>
          </form>

          {/* Mode Toggle for Mobile */}
          <div className="sm:hidden mt-3 flex gap-2 bg-slate-700 rounded-lg p-1">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setAnswerMode("normal")}
              className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-all ${
                answerMode === "normal"
                  ? "bg-blue-600 text-white"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              일반
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setAnswerMode("roleplay")}
              className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-all ${
                answerMode === "roleplay"
                  ? "bg-purple-600 text-white"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              역할극
            </motion.button>
          </div>
        </div>
      </motion.div>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(100, 116, 139, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(100, 116, 139, 0.8);
        }
      `}</style>
    </div>
  );
}
