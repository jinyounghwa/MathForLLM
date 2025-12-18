import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { initializeVectorDB } from "./services/vectordb";
import chatRoute from "./routes/chat";
import curriculumRoute from "./routes/curriculum";

const app = new Hono();

// CORS middleware
app.use("*", cors());

// Routes
app.route("/api/chat", chatRoute);
app.route("/api/curriculum", curriculumRoute);

// Health check
app.get("/health", (c) => c.json({ status: "ok" }));

const port = 3001;

async function startServer() {
  try {
    // Initialize vector database
    await initializeVectorDB();

    console.log(`🚀 Server is running on http://localhost:${port}`);

    serve({
      fetch: app.fetch,
      port,
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}

startServer();
