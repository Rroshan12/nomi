import express from "express";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "node:fs"


import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import * as z from "zod";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { DynamicStructuredTool } from "langchain/tools";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const resumePath = path.join(__dirname, "resume.json");

const resumeRaw = fs.readFileSync(resumePath, "utf-8");
const resumeData = JSON.parse(resumeRaw);


dotenv.config();


const model = new ChatGoogleGenerativeAI({
  model: "models/gemini-2.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.7,
  apiKey: process.env.GEMENI_API_KEY, // Make sure this key is set
});


const getResumeInfoTool = new DynamicStructuredTool({
  name: "getResumeInfo",
  description: "Answers user questions about Roshan Poudel's resume, like work experience, contact info, email, skills, etc.",
  schema: z.object({
    question: z.string().describe("User's question about the resume")
  }),
  func: async ({ question }) => {
    const lower = question.toLowerCase();

    if (lower.includes("email")) {
      return `Roshan's email is ${resumeData.contact.email}`;
    }

    if (lower.includes("experience") || lower.includes("worked")) {
      return resumeData.workExperience.map((job) => {
        return `${job.role} at ${job.company} (${job.duration})`;
      }).join("\n");
    }

    if (lower.includes("phone")) {
      return `Roshan's phone number is ${resumeData.contact.phone}`;
    }

    if (lower.includes("projects")) {
      return resumeData.projects.map((p) => `- ${p.name}: ${p.description}`).join("\n");
    }

    return "Sorry, I couldn't find an exact match for your question. Try asking about email, experience, phone, or projects.";
  }
});
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that uses tools when needed."],
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
    const result = await executor.invoke({ input });
    res.json({ response: result.output });
  } catch (error) {
    console.error("Agent Error:", error);
    res.status(500).json({ error: "Something went wrong." });
  }
});

app.listen(port, () => {
  console.log(`✅ Server started at: http://localhost:${port}`);
});
