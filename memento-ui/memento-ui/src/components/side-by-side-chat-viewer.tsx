'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { GitBranch, Send, Edit2, Check, X } from "lucide-react"

type Message = {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: string
}

type Conversation = {
  id: string
  messages: Message[]
}

type SideBySideChatViewerProps = {
  conversations: Conversation[]
  activeConversations: [string, string]
  onCreateBranch: (conversationId: string) => void
  onMergeBranches: (sourceBranch: string, targetBranch: string) => void
  onSendMessage: (conversationId: string, content: string, sender: 'user' | 'ai') => void
  onEditMessage: (conversationId: string, messageId: string, newContent: string) => void
}

export function SideBySideChatViewer({
  conversations,
  activeConversations,
  onCreateBranch,
  onMergeBranches,
  onSendMessage,
  onEditMessage
}: SideBySideChatViewerProps) {
  const [newMessage, setNewMessage] = useState('')
  const [editingMessage, setEditingMessage] = useState<string | null>(null)

  const handleSendMessage = (conversationId: string) => {
    if (newMessage.trim() === '') return
    onSendMessage(conversationId, newMessage, 'user')
    setNewMessage('')
  }

  const handleEditMessage = (conversationId: string, messageId: string, newContent: string) => {
    onEditMessage(conversationId, messageId, newContent)
    setEditingMessage(null)
  }

  const renderConversation = (conversationId: string) => {
    const conversation = conversations.find(conv => conv.id === conversationId)
    if (!conversation) return null

    return (
      <Card className="flex flex-col h-full">
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            Conversation {conversation.id}
            <Button variant="outline" size="sm" onClick={() => onCreateBranch(conversation.id)}>
              <GitBranch className="w-4 h-4 mr-2" />
              Branch
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-grow overflow-hidden">
          <ScrollArea className="h-[calc(100vh-300px)]">
            {conversation.messages.map(message => (
              <div key={message.id} className={`mb-4 ${message.sender === 'user' ? 'text-right' : 'text-left'}`}>
                <div className={`inline-block p-2 rounded-lg ${message.sender === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
                  {editingMessage === message.id ? (
                    <div className="flex items-center">
                      <Textarea
                        value={message.content}
                        onChange={(e) => handleEditMessage(conversation.id, message.id, e.target.value)}
                        className="min-w-[200px]"
                      />
                      <Button variant="ghost" size="sm" onClick={() => setEditingMessage(null)}>
                        <Check className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => setEditingMessage(null)}>
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  ) : (
                    <>
                      <p>{message.content}</p>
                      <Button variant="ghost" size="sm" onClick={() => setEditingMessage(message.id)}>
                        <Edit2 className="w-4 h-4" />
                      </Button>
                    </>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">{new Date(message.timestamp).toLocaleString()}</p>
              </div>
            ))}
          </ScrollArea>
        </CardContent>
        <CardFooter>
          <div className="flex w-full items-center space-x-2">
            <Input
              type="text"
              placeholder="Type your message..."
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(conversation.id)}
            />
            <Button type="submit" size="icon" onClick={() => handleSendMessage(conversation.id)}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardFooter>
      </Card>
    )
  }

  return (
    <div className="container mx-auto p-4">
      <Tabs defaultValue="side-by-side" className="w-full">
        <TabsList>
          <TabsTrigger value="side-by-side">Side by Side</TabsTrigger>
          <TabsTrigger value="single">Single View</TabsTrigger>
        </TabsList>
        <TabsContent value="side-by-side" className="w-full">
          <div className="grid grid-cols-2 gap-4">
            {renderConversation(activeConversations[0])}
            {renderConversation(activeConversations[1])}
          </div>
        </TabsContent>
        <TabsContent value="single" className="w-full">
          <div className="grid grid-cols-1 gap-4">
            {renderConversation(activeConversations[0])}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}