'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { GitCommit, GitFork, Clock, MessageSquare, Layers } from "lucide-react"

export function Homepage() {
  const components = [
    {
      title: "Memento Fractals Time Travel",
      description: "Explore conversation histories in an immersive 3D environment",
      icon: <Layers className="w-6 h-6" />,
      href: "/memento-fractals"
    },
    {
      title: "Side-by-Side Chat Viewer",
      description: "Compare and interact with multiple conversation branches simultaneously",
      icon: <GitFork className="w-6 h-6" />,
      href: "/side-by-side-chat"
    },
    {
      title: "Time Portal Conversations",
      description: "Navigate through conversation timelines with an intuitive portal interface",
      icon: <Clock className="w-6 h-6" />,
      href: "/time-portal-conversations"
    },
    {
      title: "Git-Versioned Forum",
      description: "Participate in discussions with advanced version control features",
      icon: <GitCommit className="w-6 h-6" />,
      href: "/git-versioned-forum"
    },
    {
      title: "Enhanced Chat Interface",
      description: "Experience an advanced chat UI with AI-powered features",
      icon: <MessageSquare className="w-6 h-6" />,
      href: "/enhanced-chat"
    }
  ]

  return (
    <div className="container mx-auto px-4 py-8">
      <header className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Conversation Time Travel Hub</h1>
        <p className="text-xl text-muted-foreground">
          Explore innovative ways to interact with and navigate complex conversations
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {components.map((component, index) => (
          <Card key={index} className="flex flex-col h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {component.icon}
                <span>{component.title}</span>
              </CardTitle>
              <CardDescription>{component.description}</CardDescription>
            </CardHeader>
            <CardContent className="flex-grow flex items-end">
              <Link href={component.href} passHref>
                <Button className="w-full">Explore</Button>
              </Link>
            </CardContent>
          </Card>
        ))}
      </div>

      <footer className="mt-16 text-center text-muted-foreground">
        <p>&copy; 2024 Dratos.com. All rights reserved.</p>
      </footer>
    </div>
  )
}