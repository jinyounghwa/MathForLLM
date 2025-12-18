"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Moon, Sun } from "lucide-react";

export default function SettingsPage() {
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-4 py-4 flex items-center gap-3">
          <Link
            href="/"
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">설정</h1>
        </div>
      </div>

      {/* Settings Content */}
      <div className="max-w-2xl mx-auto px-4 py-8">
        {/* Theme Settings */}
        <div className="bg-gray-50 rounded-lg p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">테마</h2>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {darkMode ? (
                <Moon className="w-5 h-5 text-gray-600" />
              ) : (
                <Sun className="w-5 h-5 text-gray-600" />
              )}
              <span className="text-gray-700">
                {darkMode ? "다크 모드" : "라이트 모드"}
              </span>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="relative inline-flex items-center h-8 w-14 bg-gray-300 rounded-full transition-colors"
              style={{
                backgroundColor: darkMode ? "#4f46e5" : "#d1d5db",
              }}
            >
              <span
                className="inline-block h-6 w-6 bg-white rounded-full transition-transform"
                style={{
                  transform: darkMode ? "translateX(28px)" : "translateX(2px)",
                }}
              />
            </button>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            (현재 버전에서는 라이트 모드만 지원)
          </p>
        </div>

        {/* About Section */}
        <div className="bg-gray-50 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">정보</h2>
          <div className="space-y-3 text-sm text-gray-700">
            <div>
              <p className="font-medium text-gray-900">애플리케이션</p>
              <p>MathForLLM v1.0.0</p>
            </div>
            <div>
              <p className="font-medium text-gray-900">설명</p>
              <p>LLM을 위한 수학 기초 학습 웹서비스</p>
            </div>
            <div>
              <p className="font-medium text-gray-900">기술 스택</p>
              <p>Next.js 16, Hono, Qwen 2.5 7B</p>
            </div>
            <div>
              <p className="font-medium text-gray-900">벡터 DB</p>
              <p>Vectra (로컬 JSON 기반)</p>
            </div>
          </div>
        </div>

        {/* Help Section */}
        <div className="bg-blue-50 rounded-lg p-6 mt-6 border border-blue-200">
          <h3 className="font-semibold text-blue-900 mb-2">도움말</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• README.md 파일을 참고하세요</li>
            <li>• QUICKSTART.md에서 빠른 시작 가이드를 확인하세요</li>
            <li>• 문제 발생 시 GitHub Issues를 이용하세요</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
