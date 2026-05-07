import express from "express";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "node:fs";

import pdfParse from "pdf-parse/lib/pdf-parse.js";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import * as z from "zod";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { DynamicStructuredTool } from "langchain/tools";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let geminiDown = false;

dotenv.config();

const resumePdfPath = path.join(__dirname, "roshan_2026.pdf");
const resumeText = (await pdfParse(fs.readFileSync(resumePdfPath))).text;

const model = new ChatGoogleGenerativeAI({
  model: "models/gemini-2.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.7,
  apiKey: process.env.GEMENI_API_KEY,
});

export const getResumeInfoTool = new DynamicStructuredTool({
  name: "getResumeInfo",
  description:
    "Returns the full text of Roshan Poudel's resume parsed from his PDF. Call this tool whenever the user asks about Roshan — including contact info, skills, experience, education, certifications, or projects — and answer using only the returned content.",
  schema: z.object({
    question: z.string().describe("User's question about Roshan's resume"),
  }),
  func: async () => resumeText,
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a helpful assistant that answers questions about Roshan Poudel using his resume. Call the getResumeInfo tool to retrieve the resume content, then answer the user's question concisely based only on that content. If the answer is not in the resume, say you don't know.",
  ],
  ["human", "{input}"],
  ["ai", "{agent_scratchpad}"],
]);

const agent = await createToolCallingAgent({
  llm: model,
  tools: [getResumeInfoTool],
  prompt: prompt,
});

const executor = await AgentExecutor.fromAgentAndTools({
  agent,
  tools: [getResumeInfoTool],
  maxIterations: 3,
  verbose: true,
  returnIntermediateSteps: true,
});

const app = express();
const port = 9000;

app.use(express.json());

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "chat.html"));
});

app.post("/chat", async (req, res) => {
  const input = req.body?.message;
  if (!input) {
    return res.status(400).json({ error: "Missing input message" });
  }

  try {
    if (geminiDown) {
      return res.json({
        response: `[Gemini quota exhausted, returning raw resume content]\n\n${resumeText}`,
      });
    }

    const result = await executor.invoke({ input });
    return res.json({ response: result.output });
  } catch (error) {
    if (error.status === 429) {
      console.warn("Gemini quota exhausted, switching to fallback mode");
      geminiDown = true;
      return res.json({
        response: `[Gemini quota exceeded, returning raw resume content]\n\n${resumeText}`,
      });
    }

    console.error("Agent Error:", error);
    res.status(500).json({ error: "Something went wrong." });
  }
});

app.listen(port, () => {
  console.log(`✅ Server started at: http://localhost:${port}`);
});
