// Response cleaner service to remove Chinese characters, weird text, and normalize output

interface CleaningOptions {
  removeChinese: boolean;
  removeJapanese: boolean;
  normalizeWhitespace: boolean;
  fixCommonErrors: boolean;
  removeWeirdCharacters: boolean;
}

// Default cleaning options
const DEFAULT_OPTIONS: CleaningOptions = {
  removeChinese: true,
  removeJapanese: false,
  normalizeWhitespace: true,
  fixCommonErrors: true,
  removeWeirdCharacters: true,
};

// Chinese character ranges
const CHINESE_RANGES = [
  /[\u4E00-\u9FFF]/g, // CJK Unified Ideographs
  /[\u3400-\u4DBF]/g, // CJK Extension A
  /[\u3040-\u309F]/g, // Hiragana (often mixed with Chinese in Qwen output)
  /[\u30A0-\u30FF]/g, // Katakana (often mixed with Chinese in Qwen output)
];

// Weird character patterns that shouldn't appear in mathematical/educational content
const WEIRD_PATTERNS = [
  /[\u200B-\u200D\uFEFF]/g, // Zero-width characters
  /[\u0000-\u0008\u000B-\u000C\u000E-\u001F]/g, // Control characters
  /&#x[0-9a-fA-F]+;/g, // HTML entities
  /\\u[0-9a-fA-F]{4}/g, // Unicode escapes
];

// Common OCR/Encoding errors
const COMMON_ERRORS: { [key: string]: string } = {
  "Ⅴ": "V",
  "Ⅳ": "IV",
  "Ⅲ": "III",
  "Ⅱ": "II",
  "Ⅰ": "I",
  "①": "1",
  "②": "2",
  "③": "3",
  "④": "4",
  "⑤": "5",
  "⑥": "6",
  "⑦": "7",
  "⑧": "8",
  "⑨": "9",
  "⑩": "10",
  "α": "alpha",
  "β": "beta",
  "γ": "gamma",
  "δ": "delta",
  "ε": "epsilon",
  "ζ": "zeta",
  "η": "eta",
  "θ": "theta",
  "ι": "iota",
  "κ": "kappa",
  "λ": "lambda",
  "μ": "mu",
  "ν": "nu",
  "ξ": "xi",
  "ο": "omicron",
  "π": "pi",
  "ρ": "rho",
  "σ": "sigma",
  "τ": "tau",
  "υ": "upsilon",
  "φ": "phi",
  "χ": "chi",
  "ψ": "psi",
  "ω": "omega",
};

// Korean particles and connecting words that might need removal in certain contexts
const FILLER_PATTERNS = [
  /\n\s*\n\s*\n+/g, // Multiple blank lines -> two newlines
  /[ \t]+\n/g, // Trailing whitespace before newline
  /\n[ \t]+/g, // Leading whitespace after newline
];

/**
 * Remove Chinese characters from text
 */
function removeChinese(text: string): string {
  let cleaned = text;
  CHINESE_RANGES.forEach((range) => {
    cleaned = cleaned.replace(range, "");
  });
  return cleaned;
}

/**
 * Remove weird/control characters
 */
function removeWeirdCharacters(text: string): string {
  let cleaned = text;
  WEIRD_PATTERNS.forEach((pattern) => {
    cleaned = cleaned.replace(pattern, "");
  });
  return cleaned;
}

/**
 * Fix common encoding errors and character replacements
 */
function fixCommonErrors(text: string): string {
  let cleaned = text;
  Object.entries(COMMON_ERRORS).forEach(([from, to]) => {
    cleaned = cleaned.split(from).join(to);
  });
  return cleaned;
}

/**
 * Normalize whitespace and formatting
 */
function normalizeWhitespace(text: string): string {
  let cleaned = text;

  // Fix filler patterns
  FILLER_PATTERNS.forEach((pattern) => {
    cleaned = cleaned.replace(pattern, (match) => {
      if (pattern.source.includes("\\n\\s*\\n")) {
        return "\n\n"; // Multiple newlines -> two
      } else if (pattern.source.includes("\\n")) {
        return "\n"; // Remove trailing/leading whitespace around newlines
      }
      return match;
    });
  });

  // Normalize spaces around punctuation
  cleaned = cleaned
    .replace(/\s+([.,!?;:])/g, "$1") // No space before punctuation
    .replace(/([.,!?;:])\s+/g, "$1 "); // One space after punctuation

  // Fix spacing around brackets
  cleaned = cleaned
    .replace(/\[\s+/g, "[")
    .replace(/\s+\]/g, "]")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .replace(/\{\s+/g, "{")
    .replace(/\s+\}/g, "}");

  // Remove excessive spaces
  cleaned = cleaned.replace(/  +/g, " ");

  return cleaned.trim();
}

/**
 * Detect if text contains mostly weird/corrupt content
 */
function isCorruptContent(text: string): boolean {
  if (!text || text.length < 10) return false;

  // Count Chinese characters
  const chineseCount = (text.match(/[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]/g) || []).length;
  const chineseRatio = chineseCount / text.length;

  // Count weird characters
  const weirdCount = (text.match(/[\u200B-\u200D\uFEFF]/g) || []).length;

  // If more than 30% Chinese or too many weird characters, likely corrupt
  return chineseRatio > 0.3 || weirdCount > 5;
}

/**
 * Clean entire response with fallback handling
 */
function cleanResponse(text: string, options: Partial<CleaningOptions> = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Check if content is severely corrupt
  if (isCorruptContent(text)) {
    console.warn("Warning: Response appears to be corrupt or contains excessive Chinese characters");
  }

  let cleaned = text;

  // Apply cleaners in order
  if (opts.removeChinese) {
    cleaned = removeChinese(cleaned);
  }

  if (opts.removeWeirdCharacters) {
    cleaned = removeWeirdCharacters(cleaned);
  }

  if (opts.fixCommonErrors) {
    cleaned = fixCommonErrors(cleaned);
  }

  if (opts.normalizeWhitespace) {
    cleaned = normalizeWhitespace(cleaned);
  }

  return cleaned;
}

/**
 * Validate and clean response with quality checks
 */
function validateResponse(text: string): { isValid: boolean; message?: string; cleaned?: string } {
  if (!text || text.trim().length === 0) {
    return {
      isValid: false,
      message: "Empty response",
    };
  }

  const cleaned = cleanResponse(text);

  if (cleaned.trim().length === 0) {
    return {
      isValid: false,
      message: "Response became empty after cleaning (likely corrupt Chinese text)",
    };
  }

  // Check if too much was removed
  const removalRatio = 1 - cleaned.length / text.length;
  if (removalRatio > 0.5) {
    console.warn(
      `Warning: Removed ${(removalRatio * 100).toFixed(1)}% of response text - may indicate corrupt input`
    );
  }

  return {
    isValid: true,
    cleaned,
  };
}

/**
 * Extract and clean specific sections from response
 */
function cleanMarkdownResponse(text: string): string {
  let cleaned = cleanResponse(text);

  // Fix markdown formatting
  cleaned = cleaned
    .replace(/#+\s+/g, (match) => match) // Keep markdown headers
    .replace(/\*\*(.+?)\*\*/g, "**$1**") // Keep bold
    .replace(/\*(.+?)\*/g, "*$1*") // Keep italics
    .replace(/`(.+?)`/g, "`$1`"); // Keep code

  // Fix list formatting
  cleaned = cleaned.replace(/^[-*+]\s+/gm, "- "); // Normalize list items

  // Fix code blocks
  cleaned = cleaned.replace(/```\n/g, "```\n").replace(/\n```/g, "\n```");

  return cleaned;
}

/**
 * Clean and enhance response for educational content
 */
export function enhanceEducationalResponse(text: string): string {
  let enhanced = cleanResponse(text);

  // Remove any remaining line noise
  enhanced = enhanced
    .split("\n")
    .filter((line) => {
      // Remove lines that are only special characters or noise
      const cleaned = line.replace(/[^\w가-힣\s.,!?;:()[\]{}<>_*-]/g, "");
      return cleaned.trim().length > 0;
    })
    .join("\n");

  // Clean markdown if present
  if (enhanced.includes("**") || enhanced.includes("#") || enhanced.includes("```")) {
    enhanced = cleanMarkdownResponse(enhanced);
  }

  return enhanced.trim();
}

/**
 * Batch clean multiple responses (for testing)
 */
export function batchCleanResponses(texts: string[]): string[] {
  return texts.map((text) => enhanceEducationalResponse(text));
}

// Export main functions
export { cleanResponse, validateResponse, isCorruptContent, cleanMarkdownResponse };

// Export type
export type { CleaningOptions };
