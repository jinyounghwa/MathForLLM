"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { BookOpen, Settings, BarChart3, Sparkles, ArrowRight } from "lucide-react";

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <div className="min-h-screen overflow-hidden">
      {/* Animated background grid */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-primary-50 to-slate-100" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary-200 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-accent-200 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float animation-delay-2000" style={{ animationDelay: "2s" }} />
      </div>

      {/* Top Navigation */}
      <nav className="fixed top-0 right-0 z-50 flex items-center gap-2 p-6">
        <Link
          href="/dashboard"
          className="p-2 rounded-lg bg-white/80 backdrop-blur-sm border border-slate-200 hover:border-primary-400 text-slate-600 hover:text-primary-600 transition-all duration-300 hover:shadow-md"
          title="학습 대시보드"
        >
          <BarChart3 className="w-6 h-6" />
        </Link>
        <Link
          href="/settings"
          className="p-2 rounded-lg bg-white/80 backdrop-blur-sm border border-slate-200 hover:border-primary-400 text-slate-600 hover:text-primary-600 transition-all duration-300 hover:shadow-md"
          title="설정"
        >
          <Settings className="w-6 h-6" />
        </Link>
      </nav>

      {/* Main Content */}
      <div className="min-h-screen flex items-center justify-center px-4 py-20">
        <div className="max-w-5xl w-full">
          {/* Hero Section */}
          <div
            className={`text-center mb-16 transition-all duration-1000 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
          >
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100/50 border border-primary-200 mb-6 hover:bg-primary-100 transition-colors">
              <Sparkles className="w-4 h-4 text-primary-600" />
              <span className="text-sm font-medium text-primary-700">AI 기반 지능형 학습 플랫폼</span>
            </div>

            {/* Title */}
            <div className="flex items-center justify-center gap-4 mb-6">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg">
                <BookOpen className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-5xl md:text-6xl font-display font-bold bg-gradient-to-r from-primary-900 via-primary-700 to-accent-600 bg-clip-text text-transparent">
                MathForLLM
              </h1>
            </div>

            {/* Tagline */}
            <p className="text-xl md:text-2xl text-slate-700 mb-3 font-medium">
              LLM 개발을 위한 수학 기초 마스터하기
            </p>
            <p className="text-base text-slate-600 max-w-2xl mx-auto">
              Qwen 2.5 7B 기반 AI 멘토와 함께 대화형으로 배우는 똑똑한 수학 학습 경험
            </p>
          </div>

          {/* Mode Selection Cards */}
          <div className={`grid gap-8 lg:grid-cols-2 transition-all duration-1000 delay-200 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-12"}`}>
            {/* Free Question Mode Card */}
            <Link
              href="/chat"
              className="group relative overflow-hidden rounded-2xl bg-white/70 backdrop-blur-sm border border-slate-200 hover:border-primary-400 transition-all duration-500 hover:shadow-hover"
            >
              {/* Gradient overlay on hover */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

              <div className="relative p-8 lg:p-10">
                {/* Icon */}
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br from-primary-100 to-primary-50 group-hover:from-primary-200 group-hover:to-primary-100 transition-colors duration-300 mb-6">
                  <span className="text-3xl">💬</span>
                </div>

                {/* Content */}
                <div>
                  <h2 className="text-2xl font-display font-bold text-slate-900 mb-2 group-hover:text-primary-700 transition-colors">
                    자유로운 질문 모드
                  </h2>
                  <p className="text-slate-600 mb-6 leading-relaxed">
                    궁금한 내용을 언제든 자유롭게 물어보고 즉시 상세한 답변받기
                  </p>

                  {/* Features */}
                  <div className="space-y-3 mb-8">
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-primary-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-white text-xs font-bold">✓</span>
                      </div>
                      <span className="text-sm text-slate-700">RAG 시스템 기반 정확하고 신뢰할 수 있는 답변</span>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-primary-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-white text-xs font-bold">✓</span>
                      </div>
                      <span className="text-sm text-slate-700">질문 기록이 자동으로 저장되는 깔끔한 히스토리</span>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-primary-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-white text-xs font-bold">✓</span>
                      </div>
                      <span className="text-sm text-slate-700">수식 렌더링과 참고 자료 출처 표시</span>
                    </div>
                  </div>

                  {/* CTA */}
                  <div className="flex items-center gap-2 text-primary-600 font-medium group-hover:gap-3 transition-all">
                    시작하기
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </div>
            </Link>

            {/* Curriculum Mode Card */}
            <Link
              href="/curriculum/setup"
              className="group relative overflow-hidden rounded-2xl bg-white/70 backdrop-blur-sm border border-slate-200 hover:border-accent-400 transition-all duration-500 hover:shadow-hover"
            >
              {/* Gradient overlay on hover */}
              <div className="absolute inset-0 bg-gradient-to-br from-accent-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

              <div className="relative p-8 lg:p-10">
                {/* Icon */}
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br from-accent-100 to-accent-50 group-hover:from-accent-200 group-hover:to-accent-100 transition-colors duration-300 mb-6">
                  <span className="text-3xl">📚</span>
                </div>

                {/* Content */}
                <div>
                  <h2 className="text-2xl font-display font-bold text-slate-900 mb-2 group-hover:text-accent-700 transition-colors">
                    체계적 커리큘럼 모드
                  </h2>
                  <p className="text-slate-600 mb-6 leading-relaxed">
                    전문가가 설계한 체계적인 학습 경로를 따라 단계적으로 성장하기
                  </p>

                  {/* Features */}
                  <div className="space-y-3 mb-8">
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-accent-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-white text-xs font-bold">✓</span>
                      </div>
                      <span className="text-sm text-slate-700">일차별로 할당된 맞춤형 학습 내용</span>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-accent-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-white text-xs font-bold">✓</span>
                      </div>
                      <span className="text-sm text-slate-700">학습 진도를 한눈에 보는 시각화 대시보드</span>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-accent-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-white text-xs font-bold">✓</span>
                      </div>
                      <span className="text-sm text-slate-700">학습 일정 설정과 스마트 복습 관리</span>
                    </div>
                  </div>

                  {/* CTA */}
                  <div className="flex items-center gap-2 text-accent-600 font-medium group-hover:gap-3 transition-all">
                    시작하기
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </div>
            </Link>
          </div>

          {/* Footer CTA */}
          <div
            className={`mt-16 text-center transition-all duration-1000 delay-300 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
          >
            <div className="inline-block rounded-xl bg-white/50 backdrop-blur-sm border border-slate-200 px-6 py-4">
              <p className="text-sm text-slate-600 flex items-center gap-2">
                <span>✨</span>
                <span>
                  자유로운 질문 모드로 시작하거나 체계적인 커리큘럼을 통해 깊이 있게 배워보세요
                </span>
              </p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes animationDelay {
          animation-delay: 2s;
        }
      `}</style>
    </div>
  );
}
