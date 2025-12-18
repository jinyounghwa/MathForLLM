"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Moon, Sun } from "lucide-react";

export default function SettingsPage() {
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-primary-50 to-slate-100">
      {/* Header */}
      <div className="border-b border-slate-200 bg-white/70 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-4 py-4 flex items-center gap-3">
          <Link
            href="/"
            className="p-2 text-slate-600 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-all duration-300"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-display font-bold text-slate-900">설정</h1>
            <p className="text-xs text-slate-600 mt-0.5">계정과 앱 설정을 관리하세요</p>
          </div>
        </div>
      </div>

      {/* Settings Content */}
      <div className="max-w-2xl mx-auto px-4 py-8">
        {/* Theme Settings */}
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-base border border-slate-200 p-6 mb-6">
          <h2 className="text-lg font-display font-semibold text-slate-900 mb-4">테마</h2>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {darkMode ? (
                <Moon className="w-5 h-5 text-primary-600" />
              ) : (
                <Sun className="w-5 h-5 text-primary-600" />
              )}
              <span className="text-slate-700 font-medium">
                {darkMode ? "다크 모드" : "라이트 모드"}
              </span>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="relative inline-flex items-center h-8 w-14 rounded-full transition-colors duration-300"
              style={{
                backgroundColor: darkMode ? "#6b7eea" : "#e2e8f0",
              }}
            >
              <span
                className="inline-block h-6 w-6 bg-white rounded-full transition-transform duration-300 shadow-md"
                style={{
                  transform: darkMode ? "translateX(28px)" : "translateX(2px)",
                }}
              />
            </button>
          </div>
          <p className="text-sm text-slate-600 mt-2">
            💡 현재 버전에서는 라이트 모드를 권장합니다
          </p>
        </div>

        {/* About Section */}
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-base border border-slate-200 p-6 mb-6">
          <h2 className="text-lg font-display font-semibold text-slate-900 mb-4">정보</h2>
          <div className="space-y-4 text-sm">
            <div className="flex justify-between items-start pb-4 border-b border-slate-200 last:border-b-0 last:pb-0">
              <p className="font-medium text-slate-700">애플리케이션</p>
              <p className="text-slate-600">MathForLLM v1.0.0</p>
            </div>
            <div className="flex justify-between items-start pb-4 border-b border-slate-200 last:border-b-0 last:pb-0">
              <p className="font-medium text-slate-700">설명</p>
              <p className="text-slate-600 text-right">LLM을 위한 수학 기초 학습 웹서비스</p>
            </div>
            <div className="flex justify-between items-start pb-4 border-b border-slate-200 last:border-b-0 last:pb-0">
              <p className="font-medium text-slate-700">기술 스택</p>
              <p className="text-slate-600 text-right">Next.js 16, Hono, Qwen 2.5 7B</p>
            </div>
            <div className="flex justify-between items-start">
              <p className="font-medium text-slate-700">벡터 DB</p>
              <p className="text-slate-600 text-right">Vectra (로컬 JSON 기반)</p>
            </div>
          </div>
        </div>

        {/* Help Section */}
        <div className="bg-gradient-to-br from-primary-50 to-primary-100 rounded-2xl p-6 border border-primary-200">
          <h3 className="font-display font-semibold text-primary-900 mb-3 flex items-center gap-2">
            <span>❓</span>
            도움말
          </h3>
          <ul className="text-sm text-primary-800 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-primary-600 font-bold">•</span>
              <span>README.md 파일을 참고하세요</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-600 font-bold">•</span>
              <span>QUICKSTART.md에서 빠른 시작 가이드를 확인하세요</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-600 font-bold">•</span>
              <span>문제 발생 시 GitHub Issues를 이용하세요</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
