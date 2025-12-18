"use client";

import { useState } from "react";
import Link from "next/link";
import { BookOpen, Settings } from "lucide-react";

export default function Home() {
  const [selectedMode, setSelectedMode] = useState<"free" | "curriculum" | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <BookOpen className="w-10 h-10 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-900">MathForLLM</h1>
          </div>
          <p className="text-xl text-gray-600">LLM 개발을 위한 수학 기초 학습</p>
          <p className="text-sm text-gray-500 mt-2">Qwen 2.5 7B 기반 AI 멘토와 함께하는 대화형 학습</p>
        </div>

        {/* Settings Button */}
        <div className="absolute top-6 right-6">
          <Link
            href="/settings"
            className="p-2 hover:bg-white rounded-lg transition-colors"
            title="설정"
          >
            <Settings className="w-6 h-6 text-gray-600" />
          </Link>
        </div>

        {/* Mode Selection Cards */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Free Question Mode */}
          <Link
            href="/chat"
            onClick={(e) => {
              setSelectedMode("free");
            }}
            className="group bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden cursor-pointer hover:scale-105"
          >
            <div className="p-8 h-full flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                  <span className="text-2xl">💬</span>
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">일반 질문 모드</h2>
                <p className="text-gray-600">
                  자유롭게 질문하고 즉시 답변받기
                </p>
              </div>
              <div className="mt-6 pt-6 border-t border-gray-100">
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold">•</span>
                    <span>RAG 기반 정확한 답변</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold">•</span>
                    <span>질문 히스토리 자동 저장</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold">•</span>
                    <span>참고 자료 출처 표시</span>
                  </li>
                </ul>
              </div>
            </div>
          </Link>

          {/* Curriculum Mode */}
          <Link
            href="/curriculum/setup"
            onClick={(e) => {
              setSelectedMode("curriculum");
            }}
            className="group bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden cursor-pointer hover:scale-105"
          >
            <div className="p-8 h-full flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center mb-4">
                  <span className="text-2xl">📚</span>
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">계획된 학습 모드</h2>
                <p className="text-gray-600">
                  체계적인 커리큘럼 기반 학습 경로
                </p>
              </div>
              <div className="mt-6 pt-6 border-t border-gray-100">
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-start gap-2">
                    <span className="text-amber-600 font-bold">•</span>
                    <span>일차별 학습 내용 자동 할당</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-amber-600 font-bold">•</span>
                    <span>학습 진도 추적 및 시각화</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-amber-600 font-bold">•</span>
                    <span>복습 알림 기능</span>
                  </li>
                </ul>
              </div>
            </div>
          </Link>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>💡 팁: 일반 질문 모드에서는 자유롭게, 계획된 학습 모드에서는 체계적으로 학습하세요.</p>
        </div>
      </div>
    </div>
  );
}
