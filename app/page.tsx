"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Send, Download, User, Bot, Sparkles } from "lucide-react"

interface Message {
  id: string
  type: "user" | "bot"
  content: string
  timestamp: Date
}

const RESUME_DATA = {
  name: "Ameesha Priya",
  title: "Software Engineer – Backend, Distributed & Full-Stack Systems",
  experience: "4+ years",
  education: "MS Software Engineering, CMU",
  specialization: "Distributed Systems",
  recentRole: "Software Development Engineer (Recent)",
  skills: ["Java", "Spring Boot", "Kafka", "AWS", "Kubernetes", "Python"],
  links: {
    linkedin: "LinkedIn",
    github: "GitHub",
  },
}

const QUICK_QUESTIONS = [
  "Tell me about your experience",
  "What projects have you worked on?",
  "What are your technical skills?",
  "Tell me about your education",
]

export default function ResumeChatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "bot",
      content:
        "Hi! I'm an interactive resume assistant. Ask about experience, skills, projects, or education. I will not share phone numbers or private emails; please use the Contact form for outreach.",
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [contactForm, setContactForm] = useState({ name: "", email: "", company: "", message: "" })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (message: string = inputValue) => {
    if (!message.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: message,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    // Simulate API call - replace with actual backend call
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: generateResponse(message),
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, botResponse])
      setIsLoading(false)
    }, 1000)
  }

  const generateResponse = (question: string): string => {
    const q = question.toLowerCase()

    if (q.includes("experience") || q.includes("background")) {
      return "I have 4+ years of experience building distributed, real-time, and big data systems. I've worked across finance, healthcare, and e-commerce domains, focusing on backend development and system architecture."
    }

    if (q.includes("project")) {
      return "I've worked on various distributed systems projects including real-time data processing pipelines, microservices architectures, and scalable APIs. My recent work involves building high-throughput systems using Java, Spring Boot, and Kafka."
    }

    if (q.includes("skill") || q.includes("technical")) {
      return "My core technical skills include Java, Spring Boot, Kafka, AWS, Kubernetes, Python, distributed systems design, microservices architecture, and API development. I'm experienced with both backend and full-stack development."
    }

    if (q.includes("education")) {
      return "I have an MS in Software Engineering from Carnegie Mellon University (CMU), with a specialization in Distributed Systems. This gave me a strong foundation in system design and scalable architecture patterns."
    }

    return "I'd be happy to discuss my background! Feel free to ask about my experience, technical skills, projects, or education. For specific inquiries, please use the contact form."
  }

  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log("Contact form submitted:", contactForm)
    alert("Thank you! Your message has been sent.")
    setContactForm({ name: "", email: "", company: "", message: "" })
  }

  const handleDownload = () => {
    // In a real app, this would download the actual PDF
    alert("Resume download would start here. Please use the contact form to request the resume.")
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">{RESUME_DATA.name}</h1>
              <p className="text-muted-foreground text-sm">{RESUME_DATA.title}</p>
            </div>
          </div>
          <Button onClick={handleDownload} className="gap-2">
            <Download className="w-4 h-4" />
            Download Resume
          </Button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {/* Quick Facts */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Quick Facts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-sm space-y-2">
                <p>
                  <span className="font-medium">Experience:</span> {RESUME_DATA.experience}
                </p>
                <p>
                  <span className="font-medium">Education:</span> {RESUME_DATA.education}
                </p>
                <p>
                  <span className="font-medium">Specialization:</span> {RESUME_DATA.specialization}
                </p>
                <p>
                  <span className="font-medium">Recent Role:</span> {RESUME_DATA.recentRole}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Core Skills */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Core Skills</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {RESUME_DATA.skills.map((skill) => (
                  <Badge key={skill} variant="secondary" className="text-xs">
                    {skill}
                  </Badge>
                ))}
                <Badge variant="outline" className="text-xs">
                  Python
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Links */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Links</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button variant="ghost" className="w-full justify-start text-primary hover:text-primary/80">
                {RESUME_DATA.links.linkedin}
              </Button>
              <Button variant="ghost" className="w-full justify-start text-primary hover:text-primary/80">
                {RESUME_DATA.links.github}
              </Button>
            </CardContent>
          </Card>

          {/* Contact Form */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Contact</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleContactSubmit} className="space-y-4">
                <Input
                  placeholder="Name"
                  value={contactForm.name}
                  onChange={(e) => setContactForm((prev) => ({ ...prev, name: e.target.value }))}
                  required
                />
                <Input
                  type="email"
                  placeholder="Email"
                  value={contactForm.email}
                  onChange={(e) => setContactForm((prev) => ({ ...prev, email: e.target.value }))}
                  required
                />
                <Input
                  placeholder="Company (optional)"
                  value={contactForm.company}
                  onChange={(e) => setContactForm((prev) => ({ ...prev, company: e.target.value }))}
                />
                <Textarea
                  placeholder="Message"
                  rows={4}
                  value={contactForm.message}
                  onChange={(e) => setContactForm((prev) => ({ ...prev, message: e.target.value }))}
                  required
                />
                <Button type="submit" className="w-full">
                  Send Message
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        {/* Chat Interface */}
        <div className="lg:col-span-2">
          <Card className="shadow-sm h-[700px] flex flex-col">
            <CardHeader>
              <CardTitle className="text-lg">Chat with Ameesha</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.type === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.type === "bot" && (
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <Bot className="w-4 h-4 text-primary" />
                      </div>
                    )}
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-3 text-sm ${
                        message.type === "user" ? "bg-primary text-primary-foreground" : "bg-card border border-border"
                      }`}
                    >
                      {message.content}
                    </div>
                    {message.type === "user" && (
                      <div className="w-8 h-8 rounded-full bg-secondary/10 flex items-center justify-center flex-shrink-0">
                        <User className="w-4 h-4 text-secondary" />
                      </div>
                    )}
                  </div>
                ))}
                {isLoading && (
                  <div className="flex gap-3 justify-start">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-primary" />
                    </div>
                    <div className="bg-card border border-border rounded-lg px-4 py-3 text-sm">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="space-y-3">
                <div className="flex gap-2">
                  <Input
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask about my background..."
                    onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                    disabled={isLoading}
                  />
                  <Button onClick={() => handleSendMessage()} disabled={isLoading || !inputValue.trim()} size="icon">
                    <Send className="w-4 h-4" />
                  </Button>
                </div>

                {/* Quick Questions */}
                <div className="flex flex-wrap gap-2">
                  {QUICK_QUESTIONS.map((question) => (
                    <Button
                      key={question}
                      variant="outline"
                      size="sm"
                      onClick={() => handleSendMessage(question)}
                      disabled={isLoading}
                      className="text-xs"
                    >
                      {question}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
