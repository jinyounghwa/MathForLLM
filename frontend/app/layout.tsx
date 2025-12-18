import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MathForLLM - LLM을 위한 수학 학습",
  description: "LLM/AI 개발을 위한 수학 기초를 효과적으로 학습하는 대화형 웹 서비스",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className="bg-gray-50">{children}</body>
    </html>
  );
}
