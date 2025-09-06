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


// Export the tool
export const getResumeInfoTool = new DynamicStructuredTool({
  name: "getResumeInfo",
  description:
    "Answers user questions about Roshan Poudel's resume, including contact info, skills, experience, education, certifications, and projects.",
  schema: z.object({
    question: z.string().describe("User's question about Roshan's resume")
  }),

  func: async ({ question }) => {
    const lower = question.toLowerCase();

    // 🔹 Roshan keyword summary
    if (
      lower.includes("roshan") &&
      (lower.includes("who") ||
        lower.includes("about") ||
        lower.includes("profile") ||
        lower.includes("what") ||
        lower.includes("is"))
    ) {
      return `Roshan Poudel is a Senior Full Stack Software Engineer with over 5 years of experience specializing in Node.js, .NET, JavaScript, and React.\n\n${resumeData.objective}`;
    }

    // 🔹 Contact Info
    if (lower.includes("email")) return `Roshan's email is ${resumeData.contact.email}`;
    if (lower.includes("phone")) return `Roshan's phone number is ${resumeData.contact.phone}`;
    if (lower.includes("linkedin")) return `Roshan's LinkedIn: ${resumeData.contact.linkedin}`;
    if (lower.includes("github")) return `Roshan's GitHub: ${resumeData.contact.github}`;
    if (lower.includes("portfolio")) return `Roshan's portfolio: ${resumeData.contact.portfolio}`;
    if (lower.includes("address")) return `Roshan lives in ${resumeData.contact.address}`;
    if (lower.includes("birth") || lower.includes("dob")) return `Roshan was born on ${resumeData.contact.dateOfBirth}`;

    // 🔹 Education
    if (lower.includes("education") || lower.includes("study")) {
      return `Roshan completed his ${resumeData.education.degree} from ${resumeData.education.institution}`;
    }

    // 🔹 Certifications
    if (lower.includes("certification") || lower.includes("certified")) {
      return `Roshan has the following certifications:\n- ${resumeData.certifications.join("\n- ")}`;
    }

    // 🔹 Objective
    if (lower.includes("objective") || lower.includes("goal")) {
      return resumeData.objective;
    }

    // 🔹 Skills
    if (
      lower.includes("skills") ||
      lower.includes("technologies") ||
      lower.includes("tech") ||
      lower.includes("stack")
    ) {
      const skills = resumeData.technicalSkills;
      return `
Backend: ${skills.backend.join(", ")}
Frontend: ${skills.frontend.join(", ")}
Databases: ${skills.databases.join(", ")}
Cloud & DevOps: ${skills.cloudAndDevOps.join(", ")}
Real-Time & Microservices: ${skills.realTime.join(", ")}
Security: ${skills.security.join(", ")}
CI/CD & Monitoring: ${skills.ciCdMonitoring.join(", ")}
      `.trim();
    }

    // 🔹 Experience
    if (
      lower.includes("experience") ||
      lower.includes("worked") ||
      lower.includes("job") ||
      lower.includes("career") ||
      lower.includes("history")
    ) {
      return resumeData.workExperience
        .map((job) => `${job.role} at ${job.company} (${job.duration})`)
        .join("\n");
    }

    // 🔹 Projects
    if (lower.includes("project")) {
      return resumeData.projects
        .map((p) => `- ${p.name}: ${p.description}`)
        .join("\n");
    }

    // 🔹 Fallback
    return "Sorry, I couldn't find an exact match for your question. Try asking about email, experience, phone, skills, projects, or certifications.";
  }
})
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
  maxIterations: 5, 
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
