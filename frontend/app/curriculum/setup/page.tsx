"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

export default function CurriculumSetupPage() {
  const [frequency, setFrequency] = useState<1 | 2 | 3 | 7>(2);
  const [duration, setDuration] = useState<30 | 60 | 120>(60);
  const [startDate, setStartDate] = useState(
    new Date().toISOString().split("T")[0]
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState({
    totalDays: 0,
    completionDate: "",
  });

  const calculatePreview = () => {
    const start = new Date(startDate);
    let days = 0;
    const learningDayCount = Math.ceil(45 / (duration / 60)); // Estimate based on 45 lecture days
    const totalDays =
      learningDayCount * frequency + (learningDayCount - 1) * (frequency - 1);
    const end = new Date(start.getTime() + totalDays * 24 * 60 * 60 * 1000);

    setPreview({
      totalDays,
      completionDate: end.toLocaleDateString("ko-KR"),
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await axios.post(`${API_URL}/api/curriculum`, {
        frequency,
        duration,
        startDate,
      });

      // Save curriculum to localStorage
      localStorage.setItem("curriculum", JSON.stringify(response.data));

      // Redirect to curriculum page
      window.location.href = `/curriculum/learn/${response.data.curriculumId}`;
    } catch (err) {
      setError("학습 계획 생성 중 오류가 발생했습니다. 다시 시도해주세요.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 to-orange-100">
      <div className="max-w-2xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <Link
            href="/"
            className="p-2 hover:bg-white rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <h1 className="text-3xl font-bold text-gray-900">학습 계획 설정</h1>
        </div>

        {/* Form Card */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Learning Frequency */}
            <div>
              <label className="block text-lg font-semibold text-gray-900 mb-4">
                📅 학습 주기
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {([1, 2, 3, 7] as const).map((freq) => (
                  <label
                    key={freq}
                    className="cursor-pointer"
                  >
                    <input
                      type="radio"
                      name="frequency"
                      value={freq}
                      checked={frequency === freq}
                      onChange={() => {
                        setFrequency(freq);
                        calculatePreview();
                      }}
                      className="sr-only"
                    />
                    <div
                      className={`p-3 rounded-lg border-2 transition-colors text-center font-medium ${
                        frequency === freq
                          ? "border-amber-600 bg-amber-50 text-amber-900"
                          : "border-gray-200 bg-white text-gray-700 hover:border-gray-300"
                      }`}
                    >
                      {freq === 1
                        ? "매일"
                        : freq === 2
                        ? "2일마다"
                        : freq === 3
                        ? "3일마다"
                        : "주 1회"}
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Learning Duration */}
            <div>
              <label className="block text-lg font-semibold text-gray-900 mb-4">
                ⏱️ 학습 시간
              </label>
              <div className="grid grid-cols-3 gap-3">
                {([30, 60, 120] as const).map((dur) => (
                  <label
                    key={dur}
                    className="cursor-pointer"
                  >
                    <input
                      type="radio"
                      name="duration"
                      value={dur}
                      checked={duration === dur}
                      onChange={() => {
                        setDuration(dur);
                        calculatePreview();
                      }}
                      className="sr-only"
                    />
                    <div
                      className={`p-3 rounded-lg border-2 transition-colors text-center font-medium ${
                        duration === dur
                          ? "border-amber-600 bg-amber-50 text-amber-900"
                          : "border-gray-200 bg-white text-gray-700 hover:border-gray-300"
                      }`}
                    >
                      {dur === 30 ? "30분" : dur === 60 ? "1시간" : "2시간"}
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-lg font-semibold text-gray-900 mb-4">
                📆 시작일
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => {
                  setStartDate(e.target.value);
                  calculatePreview();
                }}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>

            {/* Preview */}
            <div className="bg-amber-50 rounded-lg p-6 border border-amber-200">
              <h3 className="font-semibold text-gray-900 mb-3">📊 예상 일정</h3>
              <div className="space-y-2 text-sm text-gray-700">
                <p>
                  • <strong>총 예상 기간:</strong> {preview.totalDays}일
                </p>
                <p>
                  • <strong>학습일 수:</strong> 약{" "}
                  {Math.ceil(preview.totalDays / frequency)}일
                </p>
                <p>
                  • <strong>완료 예정:</strong> {preview.completionDate}
                </p>
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full px-6 py-3 bg-amber-600 text-white rounded-lg font-semibold hover:bg-amber-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "계획 생성 중..." : "학습 계획 생성하기"}
            </button>
          </form>
        </div>

        {/* Info */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6 border border-blue-200">
          <h3 className="font-semibold text-blue-900 mb-2">💡 팁</h3>
          <p className="text-sm text-blue-800">
            • 충분한 학습 시간을 확보할 수 있는 주기를 선택하세요.
            <br />• 하루 1시간 학습을 기준으로 설정되었습니다.
            <br />• 중도에 변경 가능합니다.
          </p>
        </div>
      </div>
    </div>
  );
}
